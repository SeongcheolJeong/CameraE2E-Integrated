from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import pyisetcam.camera as camera_module
import pyisetcam.illuminant as illuminant_module
import pyisetcam.metrics as metrics_module
import pyisetcam.optics as optics_module
import pyisetcam.sensor as sensor_module
import imageio.v3 as iio
from scipy.io import loadmat, savemat
from scipy import ndimage
from scipy.signal import convolve2d

from pyisetcam.exceptions import UnsupportedOptionError
from pyisetcam.parity import run_python_case_with_context
from pyisetcam.utils import energy_to_quanta, tile_pattern
from pyisetcam.utils import quanta_to_energy
from pyisetcam import (
    ApplyFilters,
    autoExposure,
    analog2digital,
    blackbody,
    binNoiseColumnFPN,
    binNoiseFPN,
    binNoiseRead,
    binSensorCompute,
    binSensorComputeImage,
    camera_acutance,
    camera_color_accuracy,
    cameraClearData,
    camera_compute,
    cameraComputeSequence,
    cameraComputesrgb,
    camera_create,
    cameraFullReference,
    camera_get,
    cameraMoire,
    camera_mtf,
    camera_set,
    camera_vsnr,
    cameraVSNR_SL,
    chartPatchData,
    chartPatchCompare,
    changeColorSpace,
    cct2sun,
    cmatrix,
    dac_to_rgb,
    macbeth_color_error,
    macbethChartCreate,
    macbethDrawRects,
    macbethEvaluationGraphs,
    macbethGretagSGCreate,
    macbethIdealColor,
    macbethLuminanceNoise,
    macbethPatchData,
    macbethROIs,
    macbethRectangles,
    macbethSelect,
    macbethSensorValues,
    chromaticity_xy,
    cpiq_csf,
    adobergb_parameters,
    daylight,
    delta_e_ab,
    deltaEuv,
    displayDescription,
    display_create,
    display_get,
    displayList,
    displayMaxContrast,
    displaySetMaxLuminance,
    displaySetWhitePoint,
    displayShowImage,
    edge_to_mtf,
    exposureValue,
    FOTParams,
    FloydSteinberg,
    gaborP,
    hc_basis,
    HalfToneImage,
    dlCore,
    dlMTF,
    imgDeadleaves,
    imgDiskArray,
    imgMackay,
    imgRadialRamp,
    imgRamp,
    imgSquareArray,
    imgSweep,
    imgZonePlate,
    imageDeadLeaves,
    imageVernier,
    image_flip,
    image_increase_image_rgb_size,
    image_linear_transform,
    initDefaultSpectrum,
    illuminant_create,
    illuminant_get,
    illuminantModernize,
    illuminantRead,
    illuminant_set,
    ie_reflectance_samples,
    ie_read_color_filter,
    ie_read_spectra,
    ie_save_color_filter,
    ie_save_multispectral_image,
    ie_save_si_data_file,
    ieCalculateMonitorDPI,
    ieCheckerboard,
    ieCirclePoints,
    ieCovEllipsoid,
    ieConv2FFT,
    ieCTemp2SRGB,
    ie_cxcorr,
    ie_luminance_to_radiance,
    ie_n_to_megapixel,
    ieSQRI,
    ie_responsivity_convert,
    ieSpectraSphere,
    ie_scotopic_luminance_from_energy,
    ie_iso12233,
    ieISO12233v1,
    ie_xyz_from_photons,
    ieFieldHeight2Index,
    ie_field_height_to_index,
    ie_rect2_locs,
    ieCookTorrance,
    ip_get,
    ip_plot,
    ip_set,
    ip_compute,
    ip_create,
    ie_mvnrnd,
    iso_find_slanted_bar,
    iso_acutance,
    iso12233,
    ISO12233v1,
    ieLAB2XYZ,
    lrgb_to_srgb,
    LoadRawSensorData,
    lms2srgb,
    lms2xyz,
    luminance_from_energy,
    luminance_from_photons,
    macbeth_read_reflectance,
    macbeth_compare_ideal,
    mperdot2dpi,
    mkInvGammaTable,
    MOTarget,
    lensList,
    makeCmatrix,
    makeCombinedOtf,
    optics_build_2d_otf,
    opticsClearData,
    optics_coc,
    opticsCreate,
    optics_defocus_core,
    opticsDefocusDepth,
    optics_depth_defocus,
    optics_defocus_displacement,
    opticsDLCompute,
    opticsDescription,
    optics_dof,
    opticsGet,
    opticsPlotDefocus,
    opticsPlotOffAxis,
    opticsPlotTransmittance,
    optics_psf_to_otf,
    optics_ray_trace,
    opticsSet,
    opticsSICompute,
    optics2wvf,
    airy_disk,
    oiAdd,
    oiAdjustIlluminance,
    oiBirefringentDiffuser,
    oiCalculateOTF,
    oiCalculateIrradiance,
    oiCameraMotion,
    oi_calculate_illuminance,
    oiClearData,
    oiCombineDepths,
    oiCustomCompute,
    oiDepthCombine,
    oiDepthCompute,
    oiDepthEdges,
    oiDepthSegmentMap,
    oi_diffuser,
    oi_compute,
    oi_crop,
    oi_create,
    oiExtractBright,
    oiExtractWaveband,
    oi_frequency_resolution,
    oiFromFile,
    oi_get,
    oiIlluminantPattern,
    oiIlluminantSS,
    oiInterpolateW,
    oiMakeEvenRowCol,
    oiPad,
    oiPadDepthMap,
    oiPadValue,
    oiPhotonNoise,
    oiPSF,
    oiPreviewVideo,
    oiSaveImage,
    oiShowImage,
    oi_plot,
    oi_space,
    oi_spatial_support,
    oi_spatial_resample,
    oi_set,
    oiWBCompute,
    s3dRenderDepthDefocus,
    lsf2circularpsf,
    psf2lsf,
    psfAverageMultiple,
    psfCenter,
    psfCircularlyAverage,
    psfFindCriterionRadius,
    psfFindPeak,
    psfVolume,
    psf_to_zcoeff_error,
    rt_angle_lut,
    rt_block_center,
    rt_choose_block_size,
    rt_di_interp,
    rt_extract_block,
    rt_filtered_block_support,
    rt_file_names,
    rt_geometry,
    rtImageRotate,
    rt_import_data,
    rt_insert_block,
    rt_otf,
    rt_psf_apply,
    rt_psf_edit,
    rt_psf_grid,
    rt_psf_interp,
    rt_precompute_psf,
    rt_precompute_psf_apply,
    rt_ri_interp,
    rtRootPath,
    rt_sample_heights,
    rt_synthetic,
    retinalImage,
    rgb_to_xw_format,
    colorTransformMatrix,
    gauss,
    getPlanes,
    scComputeSCIELAB,
    scOpponentFilter,
    scResize,
    sc_params,
    sc_prepare_filters,
    scielab,
    scielab_rgb,
    separableConv,
    separableFilters,
    sceCreate,
    sceGet,
    siConvertRTdata,
    si_synthetic,
    run_python_case,
    hdr_render,
    sceneAddGrid,
    sceneAdjustPixelSize,
    sceneAdjustReflectance,
    sceneIlluminantScale,
    sceneRamp,
    sceneSPDScale,
    scene_adjust_illuminant,
    scene_combine,
    scene_create,
    sceneDescription,
    sceneEnergyFromVector,
    sceneFrequencySupport,
    sceneHDRChart,
    sceneHDRImage,
    scene_from_file,
    scene_adjust_luminance,
    scene_add,
    scene_add_grid,
    sceneCrop,
    scene_get,
    sceneFromBasis,
    scene_illuminant_ss,
    sceneInitGeometry,
    sceneInitSpatial,
    sceneInsert,
    scene_interpolate_w,
    sceneList,
    sceneMakeVideo,
    scene_plot,
    sceneExtractWaveband,
    scenePhotonNoise,
    scenePhotonsFromVector,
    sceneRadianceFromVector,
    sceneRadianceChart,
    scene_reflectance_chart,
    scene_ramp,
    scene_rotate,
    sceneSaveImage,
    scene_show_image,
    sceneSpatialResample,
    sceneSpatialSupport,
    scene_set,
    sceneThumbnail,
    sceneToFile,
    sceneTranslate,
    sceneVernier,
    sceneWBCreate,
    signal_current,
    SignalCurrentDensity,
    srgb2xyz,
    srgb_to_lrgb,
    srgb_parameters,
    spatialIntegration,
    vc_get_roi_data,
    xyy2xyz,
    xyz2lms,
    xyz2vSNR,
    xyz2srgb,
    imx490_compute,
    mlDescription,
    mlGetCurrent,
    mlImportParams,
    mlAnalyzeArrayEtendue,
    mlPrint,
    ml_radiance,
    mlSetCurrent,
    mlens_create,
    mlens_get,
    mlens_set,
    metricsCamera,
    metricsClose,
    metricsCompareROI,
    metricsCompute,
    metricsDescription,
    metricsGet,
    metricsGetVciPair,
    metricsKeyPress,
    metricsMaskedError,
    metricsROI,
    metricsRefresh,
    metricsSaveData,
    metricsSaveImage,
    metricsSet,
    metricsShowImage,
    metricsShowMetric,
    metrics_spd,
    noiseColumnFPN,
    noiseFPN,
    pad4conv,
    photometricExposure,
    preSCIELAB,
    scApplyFilters,
    iePixelWellCapacity,
    sc_apply_filters,
    binPixel,
    binPixelPost,
    pixelCenterFillPD,
    pixelCreate,
    pixelDescription,
    pixelDR,
    pixelGet,
    pixelIdeal,
    pixelPositionPD,
    pixelSet,
    pixelSR,
    pixelTransmittance,
    pixel_snr_luxsec,
    pixel_v_per_lux_sec,
    ptInterfaceMatrix,
    ptPoyntingFactor,
    ptPropagationMatrix,
    pvFullOverlap,
    pvReduction,
    ptReflectionAndTransmission,
    ptScatteringMatrix,
    ptSnellsLaw,
    ptTransmittance,
    sensor_ccm,
    sensorAddFilter,
    sensorCFANameList,
    sensorCfaSave,
    sensorAddNoise,
    sensorMacbethDaylightEstimate,
    sensorClearData,
    sensorCheckArray,
    sensorCheckHuman,
    sensor_compute,
    sensor_compute_array,
    sensorComputeFullArray,
    sensorComputeImage,
    sensorComputeMEV,
    sensorComputeNoise,
    sensorComputeNoiseFree,
    sensorComputeSVFilters,
    sensor_add_noise,
    sensor_compute_full_array,
    sensor_compute_image,
    sensor_compute_mev,
    sensor_compute_noise_free,
    plane2mosaic,
    regridOI2ISA,
    sensor_compute_samples,
    sensor_color_filter,
    sensorColorOrder,
    sensor_crop,
    sensor_create,
    sensor_create_ideal,
    sensor_create_array,
    sensorCreateConeMosaic,
    sensorCreateIMX490,
    sensorCreateIMECSSM4x4vis,
    sensor_dr,
    sensor_dng_read,
    sensorDetermineCFA,
    sensorDeleteFilter,
    sensorDisplayTransform,
    sensorEquateTransmittances,
    sensorFilterRGB,
    sensorFromFile,
    sensorGainOffset,
    sensor_formats,
    sensor_get,
    sensorHumanResize,
    sensorIMX363V2,
    sensorImageColorArray,
    sensorInterleaved,
    sensorJiggle,
    sensorLightField,
    sensorMPE30,
    sensorMT9V024,
    sensorNoNoise,
    sensorPDArray,
    sensorPixelCoord,
    sensor_plot,
    sensorRGB2Plane,
    sensorReadColorFilters,
    sensorReadFilter,
    sensorRescale,
    sensorResampleWave,
    sensorReplaceFilter,
    sensorSaveImage,
    sensorSNR,
    sensorSNRluxsec,
    sensorStats,
    sensorVignetting,
    sensorShowCFA,
    sensorShowCFAWeights,
    sensorShowImage,
    sensorWBCompute,
    sensor_set_size_to_fov,
    sensor_set,
    session_add_and_select_object,
    session_create,
    spd_to_cct,
    srgb_to_color_temp,
    wvf_aperture,
    wvfApply,
    wvf2SiPsf,
    wvf_compute,
    wvf_compute_psf,
    wvf_clear_data,
    wvf_create,
    wvfComputeConeAverageCriterionRadius,
    wvfComputeConePSF,
    wvfComputePupilFunctionCustomLCA,
    wvfComputePupilFunctionCustomLCAFromMaster,
    wvfComputePupilFunctionFromMaster,
    wvf_defocus_diopters_to_microns,
    wvf_defocus_microns_to_diopters,
    wvf_get,
    wvfKeySynonyms,
    wvf_load_thibos_virtual_eyes,
    wvfOSAIndexToName,
    wvf_osa_index_to_vector_index,
    wvf_osa_index_to_zernike_nm,
    wvf_pupil_function,
    wvfPrint,
    wvfRootPath,
    wvf_plot,
    wvf_set,
    wvfSummarize,
    wvf2optics,
    wvf2PSF,
    wvfPupilAmplitude,
    wvf_to_oi,
    wvf_wave_to_idx,
    wvf_zernike_nm_to_osa_index,
    xw_to_rgb_format,
    xyz_from_energy,
    xyz_to_srgb,
    y_to_lstar,
    visualAngle,
    vcLoadObject,
    vcSaveObject,
    zemax_load,
    zemax_read_header,
)


def _write_mock_zemax_bundle(
    tmp_path,
    *,
    lens_file: str = "CookeLens.ZMX",
    base_lens_file_name: str = "CookeLens",
    wave_assignment: str = "500:100:600",
    base_lens_has_semicolon: bool = True,
    psf_size_assignment: int = 2,
    params_file_name: str = "ISETPARAMS.txt",
    psf_spacing_assignment_mm: float | None = 0.00025,
):
    params_file = tmp_path / params_file_name
    base_lens_line = f"baseLensFileName='{base_lens_file_name}'"
    if base_lens_has_semicolon:
        base_lens_line += ";"
    psf_spacing_line = "" if psf_spacing_assignment_mm is None else f"psfSpacing={psf_spacing_assignment_mm:.7f};\n"
    params_file.write_text(
        "".join(
            [
                f"lensFile='{lens_file}';\n",
                f"psfSize={psf_size_assignment};\n",
                psf_spacing_line,
                f"wave={wave_assignment};\n",
                "imgHeightNum=2;\n",
                "imgHeightMax=1.0;\n",
                "objDist=250.0;\n",
                "mag=-0.1;\n",
                f"{base_lens_line}\n",
                "refWave=550.0;\n",
                "fov=15.0;\n",
                "efl=6.0;\n",
                "fnumber_eff=1.8;\n",
                "fnumber=2.0;\n",
            ]
        ),
        encoding="latin1",
    )
    (tmp_path / "CookeLens_DI_.dat").write_text("0.0 0.05 0.2 0.3\n", encoding="latin1")
    (tmp_path / "CookeLens_RI_.dat").write_text("1.0 0.9 0.8 0.7\n", encoding="latin1")
    kernels = {
        "CookeLens_2D_PSF_Fld1_Wave1.dat": "1 0 0 0\n",
        "CookeLens_2D_PSF_Fld1_Wave2.dat": "0 1 0 0\n",
        "CookeLens_2D_PSF_Fld2_Wave1.dat": "0 0 1 0\n",
        "CookeLens_2D_PSF_Fld2_Wave2.dat": "0 0 0 1\n",
    }
    for name, data in kernels.items():
        (tmp_path / name).write_text(
            "spacing is 0.5000 microns\n"
            "area is 1.0000 microns\n"
            "normalized.\n"
            f"{data}",
            encoding="latin1",
        )
    return params_file


def test_oi_compute_matches_scene_wave(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    assert oi.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.array_equal(oi.fields["wave"], scene.fields["wave"])


def test_wvf_load_thibos_virtual_eyes_drives_pupil_size_smoke(asset_store) -> None:
    default_mean = wvf_load_thibos_virtual_eyes(asset_store=asset_store)
    assert default_mean.shape == (36,)

    sample_mean, sample_cov, subject_coeffs = wvf_load_thibos_virtual_eyes(7.5, asset_store=asset_store, full=True)
    assert sample_mean.shape == (66,)
    assert sample_cov.shape == (66, 66)
    assert subject_coeffs["left_eye"].shape == (66, 70)
    assert subject_coeffs["right_eye"].shape == (66, 70)
    assert subject_coeffs["both_eyes"].shape == (66, 140)

    wvf = wvf_create(
        wave=np.array([550.0], dtype=float),
        zcoeffs=sample_mean,
        measured_pupil_diameter_mm=7.5,
        calc_pupil_diameter_mm=3.0,
    )
    wvf = wvf_compute(wvf)
    psf = np.asarray(wvf_get(wvf, "psf", 550.0), dtype=float)

    assert psf.shape == (int(wvf_get(wvf, "spatial samples")), int(wvf_get(wvf, "spatial samples")))
    assert np.isclose(np.sum(psf), 1.0)
    assert np.isclose(wvf_get(wvf, "measured pupil size", "mm"), 7.5)
    assert np.isclose(wvf_get(wvf, "calc pupil size", "mm"), 3.0)


def test_ie_mvnrnd_supports_explicit_standard_normal_samples() -> None:
    mu = np.array([0.5, -1.0], dtype=float)
    sigma = np.array([[4.0, 1.0], [1.0, 9.0]], dtype=float)
    standard_normal = np.array([[0.25, -0.5], [1.0, 0.75], [-0.25, 0.5]], dtype=float)

    samples = ie_mvnrnd(mu, sigma, standard_normal_samples=standard_normal)
    expected = standard_normal @ np.linalg.cholesky(sigma).T + np.repeat(mu.reshape(1, -1), 3, axis=0)

    assert samples.shape == (3, 2)
    assert np.allclose(samples, expected)


def test_wvf_thibos_model_workflow_supports_mean_and_example_subject_psfs(asset_store) -> None:
    measured_pupil_mm = 4.5
    calc_pupil_mm = 3.0
    measured_wavelength_nm = 550.0
    calc_waves = np.arange(450.0, 651.0, 100.0, dtype=float)
    sample_mean, sample_cov, _ = wvf_load_thibos_virtual_eyes(measured_pupil_mm, asset_store=asset_store, full=True)
    standard_normal = np.arange(1, 10 * sample_mean.size + 1, dtype=float).reshape(10, sample_mean.size, order="F")
    standard_normal = np.sqrt(-2.0 * np.log(np.clip(np.mod(standard_normal * 0.7548776662466927, 1.0), 1e-6, 1.0 - 1e-6))) * np.cos(
        2.0 * np.pi * np.mod(standard_normal * 0.5698402909980532, 1.0)
    )
    example_coeffs = ie_mvnrnd(sample_mean, sample_cov, standard_normal_samples=standard_normal)

    assert example_coeffs.shape == (10, sample_mean.size)

    zcoeffs = np.zeros(65, dtype=float)
    zcoeffs[:13] = np.asarray(sample_mean[:13], dtype=float)
    mean_subject = wvf_create()
    mean_subject = wvf_set(mean_subject, "zcoeffs", zcoeffs)
    mean_subject = wvf_set(mean_subject, "measured pupil", measured_pupil_mm)
    mean_subject = wvf_set(mean_subject, "calculated pupil", calc_pupil_mm)
    mean_subject = wvf_set(mean_subject, "measured wavelength", measured_wavelength_nm)
    mean_subject = wvf_set(mean_subject, "calc wave", calc_waves)
    mean_subject = wvf_compute(mean_subject)

    psf_rows = []
    for wavelength_nm in calc_waves:
        psf = np.asarray(wvf_get(mean_subject, "psf", float(wavelength_nm)), dtype=float)
        psf_rows.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))

    assert len(psf_rows) == 3
    assert all(row.shape == psf_rows[0].shape for row in psf_rows)

    subject = wvf_create()
    subject = wvf_set(subject, "measured pupil", measured_pupil_mm)
    subject = wvf_set(subject, "calculated pupil", calc_pupil_mm)
    subject = wvf_set(subject, "measured wavelength", measured_wavelength_nm)

    which_subjects = np.arange(0, 10, 3, dtype=int)
    for subject_index in which_subjects:
        subject_zcoeffs = np.zeros(65, dtype=float)
        subject_zcoeffs[:13] = example_coeffs[subject_index, :13]
        subject = wvf_set(subject, "zcoeffs", subject_zcoeffs)
        subject = wvf_set(subject, "calc wave", 450.0)
        subject = wvf_compute(subject)
        psf = np.asarray(wvf_get(subject, "psf", 450.0), dtype=float)
        assert psf.shape[0] == psf.shape[1]
        assert np.isclose(float(np.sum(psf)), 1.0)


def test_wvf_create_key_value_and_human_lca_smoke(asset_store) -> None:
    zcoeffs = wvf_load_thibos_virtual_eyes(7.5, asset_store=asset_store)

    wvf = wvf_create(
        "calc wavelengths",
        np.array([520.0], dtype=float),
        "zcoeffs",
        zcoeffs,
        "measured pupil size",
        7.5,
        "calc pupil size",
        3.0,
        "name",
        "7-pupil",
    )
    wvf = wvf_set(wvf, "lcaMethod", "human")
    wvf = wvf_compute(wvf)

    wvf_no_lca = wvf_create(
        "calc wavelengths",
        np.array([520.0], dtype=float),
        "zcoeffs",
        zcoeffs,
        "measured pupil size",
        7.5,
        "calc pupil size",
        3.0,
    )
    wvf_no_lca = wvf_compute(wvf_no_lca)

    psf_human = np.asarray(wvf_get(wvf, "psf", 520.0), dtype=float)
    psf_none = np.asarray(wvf_get(wvf_no_lca, "psf", 520.0), dtype=float)

    assert wvf_get(wvf, "name") == "7-pupil"
    assert np.isclose(wvf_get(wvf, "measured pupil size", "mm"), 7.5)
    assert np.isclose(wvf_get(wvf, "calc pupil size", "mm"), 3.0)
    assert wvf_get(wvf, "lcaMethod") == "human"
    assert np.isclose(np.sum(psf_human), 1.0)
    assert float(np.max(np.abs(psf_human - psf_none))) > 1e-8


def test_wvf_measured_pupil_variation_smoke(asset_store) -> None:
    measured_pupils = [7.5, 6.0, 4.5, 3.0]
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    psf_mid_rows = []

    for measured_pupil in measured_pupils:
        zcoeffs = wvf_load_thibos_virtual_eyes(measured_pupil, asset_store=asset_store)
        wvf = wvf_create(
            "calc wavelengths",
            wave,
            "zcoeffs",
            zcoeffs,
            "measured pupil size",
            measured_pupil,
            "calc pupil size",
            3.0,
            "name",
            f"{measured_pupil:g}-pupil",
        )
        wvf = wvf_set(wvf, "lcaMethod", "human")
        wvf = wvf_compute(wvf)
        psf = np.asarray(wvf_get(wvf, "psf", 550.0), dtype=float)
        psf_mid_rows.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))
        assert np.isclose(np.sum(psf), 1.0)

    max_diffs = [
        float(np.max(np.abs(psf_mid_rows[0] - row)))
        for row in psf_mid_rows[1:]
    ]
    assert any(diff > 1e-9 for diff in max_diffs)


def test_wvf_psf_sample_spacing_setter_matches_formula() -> None:
    wavelength_nm = 550.0
    focal_length_mm = 4.0
    f_number = 4.0
    n_pixels = 1024
    psf_spacing_mm = 1e-3

    wvf = wvf_create()
    wvf = wvf_set(wvf, "wave", wavelength_nm)
    wvf = wvf_set(wvf, "focal length", focal_length_mm, "mm")
    wvf = wvf_set(wvf, "calc pupil diameter", focal_length_mm / f_number, "mm")
    wvf = wvf_set(wvf, "spatial samples", n_pixels)
    wvf = wvf_set(wvf, "psf sample spacing", psf_spacing_mm)

    lambda_mm = float(np.asarray(wvf_get(wvf, "wave", "mm"), dtype=float).reshape(-1)[0])
    pupil_spacing_mm = lambda_mm * focal_length_mm / (psf_spacing_mm * n_pixels)
    expected_field_size_mm = pupil_spacing_mm * n_pixels

    assert np.isclose(float(wvf_get(wvf, "field size mm", "mm")), expected_field_size_mm)
    assert np.isclose(float(wvf_get(wvf, "pupil sample spacing", "mm", wavelength_nm)), pupil_spacing_mm)
    assert np.isclose(
        float(wvf_get(wvf, "psf sample spacing")),
        float(wvf_get(wvf, "ref psf sample interval")),
    )


def test_scene_get_depth_map_defaults_to_scene_distance(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    depth_map = scene_get(scene, "depth map")

    assert depth_map.shape == scene_get(scene, "size")
    assert np.allclose(depth_map, scene_get(scene, "distance"))
    assert np.allclose(scene_get(scene, "depth range"), np.array([scene_get(scene, "distance"), scene_get(scene, "distance")]))


def test_ie_field_height_to_index_matches_matlab_rules() -> None:
    heights = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)

    assert ie_field_height_to_index(heights, 0.6) == 2
    assert ie_field_height_to_index(heights, 0.9) == 3
    assert ie_field_height_to_index(heights, 0.1, bounding=True) == (1, 2)
    assert ie_field_height_to_index(heights, 0.9, bounding=True) == (2, 3)


def test_raytrace_struct_uses_normalized_keys_recognizes_blocks_per_field_height() -> None:
    assert optics_module._raytrace_struct_uses_normalized_keys({"blocks_per_field_height": 7}) is True
    assert optics_module._raytrace_struct_uses_normalized_keys({"blocksPerFieldHeight": 7}) is False


def test_raytrace_struct_uses_normalized_keys_recognizes_scalar_normalized_fields() -> None:
    assert optics_module._raytrace_struct_uses_normalized_keys({"f_number": 3.1}) is True
    assert optics_module._raytrace_struct_uses_normalized_keys({"magnification": -0.2}) is True
    assert optics_module._raytrace_struct_uses_normalized_keys({"fNumber": 3.1}) is False
    assert optics_module._raytrace_struct_uses_normalized_keys({"mag": -0.2}) is False


def test_oi_compute_tracks_output_geometry(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene)
    rows, cols = oi.data["photons"].shape[:2]
    assert np.isclose(oi.fields["width_m"], cols * oi.fields["sample_spacing_m"])
    assert np.isclose(oi.fields["height_m"], rows * oi.fields["sample_spacing_m"])


def test_oi_compute_crop_matches_matlab_crop_geometry(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi_uncropped = oi_compute(oi_create(), scene, crop=False)
    oi_cropped = oi_compute(oi_create(), scene, crop=True)

    image_distance = float(oi_uncropped.fields["image_distance_m"])
    focal_length = float(oi_uncropped.fields["optics"]["focal_length_m"])
    expected_scale = image_distance / focal_length

    assert np.isclose(
        oi_cropped.fields["sample_spacing_m"],
        oi_uncropped.fields["sample_spacing_m"] * expected_scale,
    )
    assert oi_cropped.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_compute_tracks_padded_and_cropped_depth_maps(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    rows, cols = scene_get(scene, "size")
    depth_map = np.linspace(1.0, 1.4, rows * cols, dtype=float).reshape(rows, cols)
    scene = scene_set(scene, "depth map", depth_map)

    oi_uncropped = oi_compute(oi_create(), scene, crop=False)
    uncropped_depth = oi_get(oi_uncropped, "depth map")
    pad_rows, pad_cols = oi_uncropped.fields["padding_pixels"]

    assert uncropped_depth.shape == oi_uncropped.data["photons"].shape[:2]
    assert np.allclose(uncropped_depth[:pad_rows, :], 0.0)
    assert np.allclose(uncropped_depth[:, :pad_cols], 0.0)
    assert np.allclose(uncropped_depth[pad_rows:-pad_rows, pad_cols:-pad_cols], depth_map)

    oi_cropped = oi_compute(oi_create(), scene, crop=True)
    assert np.allclose(oi_get(oi_cropped, "depth map"), depth_map)


def test_oi_depth_helpers_match_manual_depth_map_selection(asset_store) -> None:
    wave = np.array([500.0, 600.0], dtype=float)
    scene = scene_create("uniform ee", 8, wave, asset_store=asset_store)
    scene_depth = np.vstack(
        [
            np.full((4, 8), 1.0, dtype=float),
            np.full((4, 8), 3.0, dtype=float),
        ]
    )
    scene = scene_set(scene, "depth map", scene_depth)

    padded = oiPadDepthMap(scene)
    expected_padded = np.pad(scene_depth, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)

    assert np.array_equal(padded, expected_padded)

    depth_edges = np.array([1.0, 3.0], dtype=float)
    segmented = oiDepthSegmentMap(padded, depth_edges)
    expected_segmented = np.where(
        np.abs(padded - depth_edges[0]) <= np.abs(padded - depth_edges[1]),
        1,
        2,
    )

    assert np.array_equal(segmented, expected_segmented)

    def build_depth_oi(photons: np.ndarray, depth_map: np.ndarray) -> object:
        oi = oi_create()
        oi = oi_set(oi, "wave", wave)
        oi = oi_set(oi, "photons", photons)
        return oi_set(oi, "depth map", depth_map)

    combined = oiDepthCombine(
        [
            build_depth_oi(np.full((*padded.shape, wave.size), 10.0, dtype=float), np.zeros_like(padded)),
            build_depth_oi(np.full((*padded.shape, wave.size), 20.0, dtype=float), np.zeros_like(padded)),
        ],
        scene,
        depth_edges,
    )
    expected_combined = np.where(segmented[:, :, np.newaxis] == 1, 10.0, 20.0)

    assert np.allclose(np.asarray(oi_get(combined, "photons"), dtype=float), expected_combined)
    assert np.array_equal(np.asarray(oi_get(combined, "depth map"), dtype=float), expected_padded)
    assert oi_get(combined, "name") == "Combined"

    near_mask = np.pad(
        np.vstack(
            [
                np.ones((4, 8), dtype=bool),
                np.zeros((4, 8), dtype=bool),
            ]
        ),
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=False,
    )
    far_mask = np.pad(
        np.vstack(
            [
                np.zeros((4, 8), dtype=bool),
                np.ones((4, 8), dtype=bool),
            ]
        ),
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=False,
    )
    near_photons = np.where(
        near_mask[:, :, np.newaxis],
        np.array([3.0, 5.0], dtype=float).reshape(1, 1, -1),
        0.0,
    )
    far_photons = np.where(
        far_mask[:, :, np.newaxis],
        np.array([7.0, 11.0], dtype=float).reshape(1, 1, -1),
        0.0,
    )
    summed = oiCombineDepths(
        [
            build_depth_oi(near_photons, np.where(near_mask, 1.0, 0.0)),
            build_depth_oi(far_photons, np.where(far_mask, 3.0, 0.0)),
        ]
    )
    expected_depth = np.where(near_mask, 1.0, np.where(far_mask, 3.0, 0.0))

    assert np.allclose(np.asarray(oi_get(summed, "photons"), dtype=float), near_photons + far_photons)
    assert np.allclose(np.asarray(oi_get(summed, "depth map"), dtype=float), expected_depth)


def test_oi_depth_defocus_wrappers_match_manual_replay(asset_store) -> None:
    wave = np.array([500.0, 600.0], dtype=float)
    scene = scene_create("uniform ee", 8, wave, asset_store=asset_store)

    scene_photons = np.zeros((8, 8, wave.size), dtype=float)
    base_plane = np.outer(np.linspace(1.0, 2.0, 8, dtype=float), np.linspace(0.5, 1.5, 8, dtype=float))
    scene_photons[:, :, 0] = base_plane
    scene_photons[:, :, 1] = np.flipud(base_plane) * 1.3
    depth_map = np.vstack(
        [
            np.full((4, 8), 1.2, dtype=float),
            np.full((4, 8), 2.4, dtype=float),
        ]
    )
    scene = scene_set(scene, "photons", scene_photons)
    scene = scene_set(scene, "depth map", depth_map)

    oi = oi_create("diffraction limited", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    oi_wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    focal_length = float(opticsGet(optics, "focal length", "m"))

    requested_defocus = np.array([-0.25, -0.75], dtype=float)
    depth_edges, image_distance, object_defocus = oiDepthEdges(oi, requested_defocus, 1.6)

    expected_edges = np.asarray(opticsDefocusDepth(requested_defocus, optics, focal_length), dtype=float).reshape(-1)
    expected_index = int(np.argmin(np.abs(expected_edges - 1.6)))
    _, expected_image_distance = optics_depth_defocus(expected_edges[expected_index], optics, focal_length)
    expected_object_defocus, _ = optics_depth_defocus(expected_edges, optics, expected_image_distance)

    assert np.allclose(depth_edges, expected_edges)
    assert np.isclose(image_distance, expected_image_distance)
    assert np.allclose(object_defocus, np.asarray(expected_object_defocus, dtype=float).reshape(-1))

    stack_edges = np.array([1.0, 1.8, 2.8], dtype=float)
    combined, depth_ois, stack_defocus = s3dRenderDepthDefocus(scene, oi, image_distance, stack_edges)

    depth_centers = stack_edges[:-1] + (np.diff(stack_edges) / 2.0)
    expected_stack_defocus, _ = optics_depth_defocus(depth_centers, optics, image_distance)
    expected_stack_defocus = np.asarray(expected_stack_defocus, dtype=float).reshape(-1)
    max_sf = float(scene_get(scene, "maxfreqres", "cpd"))
    n_steps = int(min(max(np.ceil(max_sf), 1.0), 70.0))
    sample_sf = np.linspace(0.0, max_sf, n_steps, dtype=float)

    manual_depth_ois = [oi.clone() for _ in range(depth_centers.size)]
    for depth_index in range(depth_centers.size - 1, -1, -1):
        defocus = np.full(oi_wave.size, expected_stack_defocus[depth_index], dtype=float)
        otf_rows, sample_sf_mm = optics_defocus_core(optics, sample_sf, defocus)
        current_optics = optics_build_2d_otf(optics, otf_rows, sample_sf_mm)
        current_oi = oi_set(oi.clone(), "optics", current_optics)

        slab_mask = (stack_edges[depth_index] <= depth_map) & (depth_map < stack_edges[depth_index + 1])
        slab_scene = scene.clone()
        slab_photons = scene_photons.copy()
        slab_photons[~slab_mask, :] = 0.0
        slab_scene = scene_set(slab_scene, "photons", slab_photons)
        slab_scene = scene_set(slab_scene, "depth map", depth_map * slab_mask.astype(float))
        manual_depth_ois[depth_index] = oi_compute(current_oi, slab_scene)

    manual_combined = oiCombineDepths(manual_depth_ois)

    assert np.allclose(stack_defocus, expected_stack_defocus)
    for computed, expected in zip(depth_ois, manual_depth_ois, strict=True):
        assert np.allclose(np.asarray(oi_get(computed, "photons"), dtype=float), np.asarray(oi_get(expected, "photons"), dtype=float))
        assert np.allclose(np.asarray(oi_get(computed, "depth map"), dtype=float), np.asarray(oi_get(expected, "depth map"), dtype=float))
    assert np.allclose(np.asarray(oi_get(combined, "photons"), dtype=float), np.asarray(oi_get(manual_combined, "photons"), dtype=float))
    assert np.allclose(np.asarray(oi_get(combined, "depth map"), dtype=float), np.asarray(oi_get(manual_combined, "depth map"), dtype=float))

    requested_planes = np.array([1.2, 2.6], dtype=float)
    oi_stack, last_defocus = oiDepthCompute(oi, scene, image_distance, requested_planes)

    expected_last_defocus = np.empty(0, dtype=float)
    for current_oi, plane_depth in zip(oi_stack, requested_planes, strict=True):
        plane_scene = scene.clone()
        plane_scene = scene_set(plane_scene, "depth map", np.full_like(depth_map, plane_depth, dtype=float))
        expected_oi, _, current_defocus = s3dRenderDepthDefocus(plane_scene, oi, image_distance)
        current_defocus = np.asarray(current_defocus, dtype=float).reshape(-1)

        assert oi_get(current_oi, "name") == f"Defocus {float(current_defocus[0]):.2f}"
        assert np.allclose(np.asarray(oi_get(current_oi, "photons"), dtype=float), np.asarray(oi_get(expected_oi, "photons"), dtype=float))
        assert np.allclose(np.asarray(oi_get(current_oi, "depth map"), dtype=float), np.asarray(oi_get(expected_oi, "depth map"), dtype=float))
        expected_last_defocus = current_defocus

    assert np.allclose(last_defocus, expected_last_defocus)


def test_oi_get_reports_matlab_style_geometry_vectors(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    rows, cols = oi.data["photons"].shape[:2]
    sample_spacing = oi_get(oi, "sample spacing")
    spatial_resolution = oi_get(oi, "distance per sample")
    angular_resolution = oi_get(oi, "angular resolution")

    assert oi_get(oi, "rows") == rows
    assert oi_get(oi, "cols") == cols
    assert oi_get(oi, "size") == (rows, cols)
    assert np.isclose(sample_spacing[0], oi_get(oi, "width") / cols)
    assert np.isclose(sample_spacing[1], oi_get(oi, "height") / rows)
    assert np.isclose(spatial_resolution[0], oi_get(oi, "height") / rows)
    assert np.isclose(spatial_resolution[1], oi_get(oi, "width") / cols)
    assert angular_resolution.shape == (2,)
    assert np.all(angular_resolution > 0.0)


def test_oi_set_updates_geometry_and_optics_accessors() -> None:
    oi = oi_create()
    oi = oi_set(oi, "photons", np.ones((4, 6, 3), dtype=float))
    oi = oi_set(oi, "focal length", 0.02)
    oi = oi_set(oi, "fov", 5.0)
    oi = oi_set(oi, "compute method", "opticspsf")
    oi = oi_set(oi, "diffuser method", "skip")
    oi = oi_set(oi, "off axis method", "cos4th")

    expected_width = 2.0 * oi_get(oi, "image distance") * np.tan(np.deg2rad(2.5))

    assert np.isclose(oi_get(oi, "width"), expected_width)
    assert np.isclose(oi_get(oi, "sample size"), expected_width / 6.0)
    assert oi_get(oi, "compute method") == "opticspsf"
    assert oi_get(oi, "diffuser method") == "skip"
    assert oi_get(oi, "off axis method") == "cos4th"


def test_oi_get_reports_spatial_and_frequency_support(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    spatial = oi_get(oi, "spatial support linear", "mm")
    mesh = oi_get(oi, "spatial support", "mm")
    angular = oi_get(oi, "angular support", "radians")
    freq = oi_get(oi, "frequency resolution", "mm")
    fsupport = oi_get(oi, "frequency support", "mm")

    rows, cols = oi.data["photons"].shape[:2]
    assert spatial["x"].shape == (cols,)
    assert spatial["y"].shape == (rows,)
    assert mesh.shape == (rows, cols, 2)
    assert angular.shape == (rows, cols, 2)
    assert freq["fx"].shape == (cols,)
    assert freq["fy"].shape == (rows,)
    assert fsupport.shape == (rows, cols, 2)
    assert np.isclose(oi_get(oi, "max frequency resolution", "mm"), max(freq["fx"].max(), freq["fy"].max()))


def test_oi_geometry_helpers_match_existing_getters(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    spatial = oi_spatial_support(oi, "mm")
    freq = oi_frequency_resolution(oi, "mm")
    sample_position = np.array([1.0, 1.0], dtype=float)
    dpos = oi_space(oi, sample_position, "mm")

    assert np.allclose(spatial["x"], np.asarray(oi_get(oi, "spatial support linear", "mm")["x"], dtype=float))
    assert np.allclose(spatial["y"], np.asarray(oi_get(oi, "spatial support linear", "mm")["y"], dtype=float))
    assert np.allclose(freq["fx"], np.asarray(oi_get(oi, "frequency resolution", "mm")["fx"], dtype=float))
    assert np.allclose(freq["fy"], np.asarray(oi_get(oi, "frequency resolution", "mm")["fy"], dtype=float))

    size = np.asarray(oi_get(oi, "size"), dtype=float)
    spacing = np.asarray(oi_get(oi, "distance per sample", "mm"), dtype=float)
    expected = np.array(
        [
            (size[0] / 2.0 - sample_position[0]) * spacing[0],
            (sample_position[1] - size[1] / 2.0) * spacing[1],
        ],
        dtype=float,
    )
    assert np.allclose(dpos, expected)


def test_oi_wrapper_compatibility_surface(asset_store, tmp_path: Path) -> None:
    scene = scene_create("uniform ee", np.array([8, 8], dtype=int), asset_store=asset_store)
    scene = scene_set(scene, "fov", 4.0)
    oi = oi_compute(oi_create("diffraction limited", asset_store=asset_store), scene)

    wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    support_mm = np.asarray(oi_get(oi, "frequency support", "mm"), dtype=float)
    target_wave = float(wave[np.argmin(np.abs(wave - 550.0))])
    target_index = int(np.argmin(np.abs(wave - target_wave)))

    otf_plane, support = oiCalculateOTF(oi, target_wave, "mm")
    otf_mm_all, support_mm_all = oiCalculateOTF(oi, wave, "mm")
    otf_all, support_all = oiCalculateOTF(oi)

    assert otf_plane.shape == otf_mm_all[:, :, target_index].shape
    assert np.allclose(otf_plane, otf_mm_all[:, :, target_index], atol=1e-12, rtol=1e-12)
    assert np.allclose(support, support_mm, atol=1e-12, rtol=1e-12)
    assert np.allclose(support_mm_all, support_mm, atol=1e-12, rtol=1e-12)
    assert otf_all.shape[2] == wave.size
    assert np.allclose(support_all, np.asarray(oi_get(oi, "frequency support", "cyclesPerDegree"), dtype=float), atol=1e-12, rtol=1e-12)

    custom_enabled, custom_method = oiCustomCompute(oi)
    assert custom_enabled is False
    assert custom_method is None
    custom_oi = oi_set(oi.clone(), "custom compute method", "myCustomOI")
    custom_enabled, custom_method = oiCustomCompute(custom_oi)
    assert custom_enabled is True
    assert custom_method == "myCustomOI"
    assert bool(oi_get(custom_oi, "custom compute")) is True
    assert oi_get(custom_oi, "custom compute method") == "myCustomOI"

    wb_scene = scene_create("uniform ee", 4, np.array([500.0, 600.0], dtype=float), asset_store=asset_store)
    wb_scene = scene_set(
        wb_scene,
        "photons",
        np.arange(1, 4 * 4 * 2 + 1, dtype=float).reshape(4, 4, 2),
    )
    work_dir = Path(sceneWBCreate(wb_scene, tmp_path / "scene-wb"))
    oi_dir = Path(oiWBCompute(work_dir, oi_create("diffraction limited", asset_store=asset_store), asset_store=asset_store))
    oi_files = sorted(oi_dir.glob("oi*.mat"))

    assert oi_dir == work_dir
    assert len(oi_files) == 2
    loaded_oi, loaded_path = vcLoadObject("oi", oi_files[0])
    assert Path(loaded_path) == oi_files[0]
    assert np.array_equal(np.asarray(oi_get(loaded_oi, "wave"), dtype=float).reshape(-1), np.array([500.0], dtype=float))

    preview_path = Path(oiPreviewVideo(oi_dir, tmp_path / "oi-preview", asset_store=asset_store))
    frames = list(iio.imiter(preview_path))

    assert preview_path.suffix == ".gif"
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0
    assert len(frames) == len(oi_files)


def test_oi_pad_value_matches_legacy_mean_zero_and_border_padding() -> None:
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=float,
    )
    oi = oi_create()
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    oi = oi_set(oi, "photons", photons)
    oi = oi_set(oi, "image distance", 0.02)
    oi = oi_set(oi, "fov", 5.0)

    mean_padded = oiPadValue(oi, [1, 2, 0], "mean photons")
    zero_padded = oiPadValue(oi, [1, 2, 0], "zero photons")
    border_padded = oiPadValue(oi, [1, 2, 0], "border photons", direction="post")

    mean_band = np.mean(photons, axis=(0, 1), dtype=float)
    assert mean_padded.data["photons"].shape == (4, 6, 2)
    assert np.allclose(mean_padded.data["photons"][1:3, 2:4, :], photons)
    assert np.allclose(mean_padded.data["photons"][0, 0, :], mean_band)
    assert mean_padded.fields["pad_value"] == "mean photons"
    assert mean_padded.fields["padding_pixels"] == (1, 2)

    width = float(oi_get(oi, "width"))
    expected_width = width * (1.0 + (4.0 / float(oi_get(oi, "cols"))))
    expected_hfov = np.rad2deg(2.0 * np.arctan2(expected_width / 2.0, float(oi_get(oi, "image distance"))))
    assert np.isclose(float(oi_get(mean_padded, "hfov")), expected_hfov)

    assert np.allclose(zero_padded.data["photons"][0, 0, :], 0.0)
    assert np.allclose(zero_padded.data["photons"][1:3, 2:4, :], photons)

    assert border_padded.data["photons"].shape == (3, 4, 2)
    assert np.allclose(border_padded.data["photons"][:2, :2, :], photons)
    assert np.allclose(border_padded.data["photons"][-1, -1, :], photons[0, 0, :])


def test_oi_pad_deprecated_wrapper_uses_near_zero_padding() -> None:
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=float,
    )
    oi = oi_create()
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    oi = oi_set(oi, "photons", photons)

    padded = oiPad(oi, [1, 1, 0], direction="post")
    near_zero = float(np.max(photons)) * 1.0e-9

    assert padded.data["photons"].shape == (3, 3, 2)
    assert np.allclose(padded.data["photons"][:2, :2, :], photons)
    assert np.allclose(padded.data["photons"][-1, -1, :], near_zero)
    assert np.isclose(float(padded.fields["pad_value"]), near_zero)


def test_oi_make_even_row_col_only_pads_odd_dimensions() -> None:
    odd_photons = np.arange(3 * 5 * 2, dtype=float).reshape(3, 5, 2)
    odd = oi_create()
    odd = oi_set(odd, "wave", np.array([500.0, 600.0], dtype=float))
    odd = oi_set(odd, "photons", odd_photons)

    evened = oiMakeEvenRowCol(odd)

    assert evened.data["photons"].shape == (4, 6, 2)
    assert np.allclose(evened.data["photons"][:3, :5, :], odd_photons)
    assert evened.fields["padding_pixels"] == (1, 1)

    even_photons = np.arange(4 * 6 * 2, dtype=float).reshape(4, 6, 2)
    even = oi_create()
    even = oi_set(even, "wave", np.array([500.0, 600.0], dtype=float))
    even = oi_set(even, "photons", even_photons)

    unchanged = oiMakeEvenRowCol(even)

    assert unchanged is not even
    assert unchanged.data["photons"].shape == (4, 6, 2)
    assert np.allclose(unchanged.data["photons"], even_photons)


def test_diffraction_otf_matches_oi_frequency_support(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=False)
    optics = oi.fields["optics"]
    wave = oi_get(oi, "wave")
    sample_spacing = float(oi_get(oi, "sample spacing")[0])

    otf = optics_module._diffraction_otf(oi.data["photons"].shape[:2], sample_spacing, wave, optics, scene)
    frequency_support = oi_get(oi, "frequency support", "m")
    rho = np.sqrt(frequency_support[:, :, 0] ** 2 + frequency_support[:, :, 1] ** 2)
    cutoff = (
        (float(optics["focal_length_m"]) / float(optics["f_number"]) / float(optics["focal_length_m"]))
        / np.maximum(np.asarray(wave, dtype=float) * 1e-9, 1e-12)
    )
    expected = np.zeros_like(otf)
    for index, cutoff_frequency in enumerate(cutoff):
        normalized = rho / max(float(cutoff_frequency), 1e-12)
        clipped = np.clip(normalized, 0.0, 1.0)
        current = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
        current[normalized >= 1.0] = 0.0
        expected[:, :, index] = np.fft.ifftshift(current)

    assert np.allclose(otf, expected)


def test_dl_core_and_dl_mtf_compatibility_surface(asset_store) -> None:
    scene = scene_create("uniform ee", np.array([8, 8], dtype=int), asset_store=asset_store)
    scene = scene_set(scene, "fov", 4.0)
    oi = oi_compute(oi_create("diffraction limited", asset_store=asset_store), scene)

    wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    support_mm = np.asarray(oi_get(oi, "frequency support", "mm"), dtype=float)
    target_wave = float(wave[np.argmin(np.abs(wave - 550.0))])

    otf_plane, returned_support = oiCalculateOTF(oi, target_wave, "mm")
    otf_helper, helper_support, cutoff = dlMTF(oi, support_mm, target_wave, "mm")
    otf_optics, optics_support, optics_cutoff = dlMTF(dict(oi.fields["optics"]), support_mm, target_wave, "mm")
    rho = np.sqrt(np.asarray(support_mm[:, :, 0], dtype=float) ** 2 + np.asarray(support_mm[:, :, 1], dtype=float) ** 2)
    otf_core = dlCore(rho, cutoff)

    assert np.allclose(returned_support, support_mm, atol=1e-12, rtol=1e-12)
    assert np.allclose(helper_support, support_mm, atol=1e-12, rtol=1e-12)
    assert np.allclose(optics_support, support_mm, atol=1e-12, rtol=1e-12)
    assert np.allclose(otf_helper, otf_plane, atol=1e-12, rtol=1e-12)
    assert np.allclose(otf_optics, otf_plane, atol=1e-12, rtol=1e-12)
    assert np.allclose(otf_core, otf_plane, atol=1e-12, rtol=1e-12)
    assert np.allclose(optics_cutoff, cutoff, atol=1e-12, rtol=1e-12)


def test_optics_plot_wrappers_and_si_convert_rt_data(asset_store, tmp_path: Path) -> None:
    optics = opticsCreate()
    defocus = np.array([-0.5, 0.0, 0.5], dtype=float)
    sample_sf = np.array([0.0, 10.0, 20.0], dtype=float)
    defocus_surface = opticsPlotDefocus(defocus, sample_sf, 550.0, optics)

    focal_length_m = float(optics["focal_length_m"])
    f_number = float(optics["f_number"])
    diopters = 1.0 / focal_length_m
    pupil_radius_m = focal_length_m / (2.0 * f_number)
    deg_per_meter = diopters / np.tan(np.deg2rad(1.0))
    reduced_sf = (deg_per_meter * 550.0e-9 / max(diopters * pupil_radius_m, 1.0e-12)) * sample_sf
    expected_surface = np.zeros_like(defocus_surface)
    for index, defocus_diopters in enumerate(defocus):
        w20 = ((pupil_radius_m**2) / 2.0) * (diopters * defocus_diopters) / max(diopters + defocus_diopters, 1.0e-12)
        alpha = (4.0 * np.pi / 550.0e-9) * w20 * np.abs(reduced_sf)
        expected_surface[index, :] = optics_module._optics_defocused_mtf(reduced_sf, np.abs(alpha))

    assert defocus_surface.shape == (defocus.size, sample_sf.size)
    assert np.all(np.isfinite(defocus_surface))
    assert np.allclose(defocus_surface, expected_surface, atol=1e-12, rtol=1e-12)

    scene = scene_set(scene_create("uniform ee", 16, asset_store=asset_store), "fov", 5.0)
    oi = oi_compute(oi_create("diffraction limited", asset_store=asset_store), scene, crop=True)
    off_axis = opticsPlotOffAxis(oi)
    expected_cos4th = optics_module._cos4th_factor(*np.asarray(oi_get(oi, "size"), dtype=int), off_axis.fields["optics"], scene)

    assert off_axis is not oi
    assert np.allclose(np.asarray(off_axis.fields["optics"]["cos4th_data"], dtype=float), expected_cos4th, atol=5e-5, rtol=5e-5)
    assert np.allclose(np.asarray(off_axis.fields["optics"]["cos4th_value"], dtype=float), expected_cos4th, atol=5e-5, rtol=5e-5)

    rt_oi = oi_create("ray trace", asset_store=asset_store)
    field_height = float(np.asarray(oi_get(rt_oi, "rtpsffieldheight"), dtype=float).reshape(-1)[0])
    first_wave = float(np.asarray(oi_get(rt_oi, "rtpsfwavelength"), dtype=float).reshape(-1)[0])
    converted, input_path, output_path = siConvertRTdata(dict(rt_oi.fields["optics"]), field_height, tmp_path / "si-rt")
    expected_psf = np.asarray(oi_get(rt_oi, "rtpsfdata", field_height, first_wave), dtype=float)
    expected_otf = np.fft.fftshift(np.fft.fft2(expected_psf / np.sum(expected_psf, dtype=float)))
    saved_path = Path(output_path) if output_path is not None else None

    assert input_path is None
    assert saved_path is not None
    assert saved_path.suffix == ".mat"
    assert saved_path.exists()
    assert converted["model"] == "shiftinvariant"
    assert opticsGet(converted, "compute method") == "opticsotf"
    assert np.array_equal(
        np.asarray(opticsGet(converted, "otfwave"), dtype=float),
        np.asarray(oi_get(rt_oi, "rtpsfwavelength"), dtype=float),
    )
    assert np.allclose(np.asarray(opticsGet(converted, "otfdata"))[:, :, 0], expected_otf, atol=1e-12, rtol=1e-12)
    assert "optics" in loadmat(saved_path, squeeze_me=True, struct_as_record=False)


def test_oi_get_image_distance_uses_depth_map_when_geometry_is_not_precomputed() -> None:
    oi = oi_create()
    oi = oi_set(oi, "photons", np.ones((4, 6, 3), dtype=float))
    oi = oi_set(oi, "depth map", np.full((4, 6), 1.2, dtype=float))
    oi.fields.pop("image_distance_m", None)
    oi.fields.pop("width_m", None)
    oi.fields.pop("height_m", None)
    oi.fields.pop("sample_spacing_m", None)

    focal_length = oi_get(oi, "focal length")
    expected = 1.0 / ((1.0 / focal_length) - (1.0 / 1.2))

    assert np.isclose(oi_get(oi, "image distance"), expected)


def test_oi_compute_skip_model_avoids_blur(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_create()
    oi.fields["optics"]["model"] = "skip"
    oi.fields["optics"]["offaxis_method"] = "skip"
    oi = oi_compute(oi, scene, crop=True)

    scene_photons = np.asarray(scene.data["photons"], dtype=float)
    oi_photons = np.asarray(oi.data["photons"], dtype=float)
    scale = oi_photons[0, 0, 0] / scene_photons[0, 0, 0]
    assert np.allclose(oi_photons, scene_photons * scale)


def test_oi_compute_border_padding_matches_corner_photons(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_create()
    oi.fields["optics"]["model"] = "skip"
    oi.fields["optics"]["offaxis_method"] = "skip"
    oi = oi_compute(oi, scene, pad_value="border", crop=False)

    pad_rows, pad_cols = oi.fields["padding_pixels"]
    corner = oi.data["photons"][pad_rows, pad_cols, :]

    assert np.allclose(oi.data["photons"][0, 0, :], corner)
    assert np.allclose(oi.data["photons"][0, -1, :], corner)
    assert np.allclose(oi.data["photons"][-1, 0, :], corner)
    assert np.allclose(oi.data["photons"][-1, -1, :], corner)


def test_oi_compute_pixel_size_matches_requested_spacing(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True, pixel_size=2e-6)

    assert np.isclose(oi_get(oi, "sample size"), 2e-6)
    assert np.isclose(oi_get(oi, "distance per sample")[1], 2e-6)
    assert np.isclose(oi.fields["requested_pixel_size_m"], 2e-6)


def test_oi_transmittance_scales_and_interpolates(asset_store) -> None:
    wave = np.array([400.0, 550.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 8, wave, asset_store=asset_store)

    baseline_oi = oi_create()
    baseline_oi.fields["optics"]["model"] = "skip"
    baseline_oi.fields["optics"]["offaxis_method"] = "skip"
    baseline = oi_compute(baseline_oi, scene, crop=True)

    scaled_oi = oi_create()
    scaled_oi.fields["optics"]["model"] = "skip"
    scaled_oi.fields["optics"]["offaxis_method"] = "skip"
    scaled_oi = oi_set(scaled_oi, "transmittance wave", wave)
    scaled_oi = oi_set(scaled_oi, "transmittance scale", np.array([0.5, 1.0, 0.25], dtype=float))
    scaled = oi_compute(scaled_oi, scene, crop=True)

    center = (scaled.data["photons"].shape[0] // 2, scaled.data["photons"].shape[1] // 2)
    ratio = scaled.data["photons"][center[0], center[1], :] / baseline.data["photons"][center[0], center[1], :]

    assert np.allclose(ratio, np.array([0.5, 1.0, 0.25], dtype=float))
    assert np.allclose(oi_get(scaled_oi, "transmittance", np.array([475.0, 625.0], dtype=float)), np.array([0.75, 0.625]))
    assert np.array_equal(oi_get(scaled_oi, "transmittance wave"), wave)
    assert oi_get(scaled_oi, "transmittance nwave") == 3


def test_oi_create_raytrace_loads_upstream_optics(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    raytrace = oi_get(oi, "raytrace")

    assert oi_get(oi, "model") == "raytrace"
    assert oi_get(oi, "compute method") == ""
    assert np.isclose(oi_get(oi, "focal length"), 0.001999989, rtol=1e-6, atol=1e-9)
    assert np.isclose(oi_get(oi, "fnumber"), 4.999973)
    assert np.isclose(oi_get(oi, "rt object distance"), 2.0)
    assert np.isclose(oi_get(oi, "rtfov"), 38.72116733777534)
    assert oi_get(oi, "raytrace optics name") == "Asphere 2mm"
    assert optics["model"] == "raytrace"
    assert np.isclose(optics["fNumber"], oi_get(oi, "fnumber"))
    assert np.isclose(optics["focalLength"], oi_get(oi, "focal length"))
    assert optics["rayTrace"]["lensFile"].endswith(".ZMX")
    assert raytrace["lensFile"].endswith(".ZMX")
    assert np.isclose(raytrace["objectDistance"], 2000.0)
    assert oi_get(oi, "rtpsffieldheight").shape == (21,)
    assert np.allclose(oi_get(oi, "rtpsffieldheight", "mm"), raytrace["psf"]["fieldHeight"])
    assert np.allclose(oi_get(oi, "rtpsfsamplespacing"), np.array([2.5e-7, 2.5e-7]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.25, 0.25]))
    assert np.array_equal(oi_get(oi, "rtpsfwavelength"), np.array([400.0, 475.0, 550.0, 625.0, 700.0]))
    assert oi_get(oi, "optics rtpsfsize") == oi_get(oi, "rtpsf")["function"].shape
    assert oi_get(oi, "rtpsfsize") == (0, 0)
    assert "fieldHeight" in oi_get(oi, "rtpsf")
    assert "sampleSpacing" in oi_get(oi, "rtpsf")


def test_oi_create_raytrace_exposes_raw_psf_support_axes(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)

    support_x = np.asarray(oi_get(oi, "rtpsfsupportx", "um"), dtype=float)
    support_y = np.asarray(oi_get(oi, "rtpsfsupporty", "um"), dtype=float).reshape(-1)
    freq_x = np.asarray(oi_get(oi, "rtfreqsupportx", "mm"), dtype=float)
    freq_y = np.asarray(oi_get(oi, "rtfreqsupporty", "mm"), dtype=float).reshape(-1)
    spacing_mm = np.asarray(oi_get(oi, "rtpsfspacing", "mm"), dtype=float)

    assert support_x.shape == (128,)
    assert support_y.shape == (128,)
    assert np.isclose(support_x[63], 0.0)
    assert np.isclose(support_y[63], 0.0)
    assert np.isclose(support_x[0], -63.0 * 0.25)
    assert np.isclose(support_x[-1], 64.0 * 0.25)
    assert np.isclose(freq_x[1] - freq_x[0], 1.0 / (128.0 * spacing_mm[1]))
    assert np.isclose(freq_y[1] - freq_y[0], 1.0 / (128.0 * spacing_mm[0]))


def test_oi_create_raytrace_exposes_raw_geometry_and_relillum_tables(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    geometry = oi_get(oi, "rtgeometry")
    rel_illum = oi_get(oi, "rtrelillum")

    assert oi_get(oi, "rtname") == "Asphere 2mm"
    assert oi_get(oi, "rtopticsprogram") == "Zemax"
    assert oi_get(oi, "rtlensfile").endswith(".ZMX")
    assert np.isclose(oi_get(oi, "rtefl", "mm"), 1.999989, atol=1e-6)
    assert np.isclose(oi_get(oi, "rteffectivefnumber"), 4.895375)
    assert np.isclose(oi_get(oi, "rtfnumber"), 4.999973)
    assert np.isclose(oi_get(oi, "rtmagnification"), -0.001)
    assert np.isclose(oi_get(oi, "rtrefwave"), 450.0)
    assert np.isclose(oi_get(oi, "rtobjdist", "mm"), 2000.0)
    assert np.isclose(oi_get(oi, "rtmaxfov"), oi_get(oi, "rtfov"))
    assert geometry["function"].shape == (21, 5)
    assert rel_illum["function"].shape == (21, 5)
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), rel_illum["fieldHeight"])
    assert np.allclose(oi_get(oi, "rtgeomfieldheight", "mm"), geometry["fieldHeight"])
    assert np.isclose(oi_get(oi, "rtgeommaxfieldheight", "mm"), np.max(geometry["fieldHeight"]))
    assert np.array_equal(oi_get(oi, "rtriwavelength"), rel_illum["wavelength"])
    assert np.array_equal(oi_get(oi, "rtgeomwavelength"), geometry["wavelength"])
    assert np.allclose(oi_get(oi, "rtgeomfunction"), geometry["function"])
    assert np.allclose(oi_get(oi, "rtgeomfunction", 550.0, "mm"), geometry["function"][:, 2])
    assert np.allclose(oi_get(oi, "rtgeomfunction", 550.0), geometry["function"][:, 2] / 1e3)
    assert np.allclose(oi_get(oi, "rtrifunction"), rel_illum["function"])


def test_oi_compute_raytrace_applies_lens_shading_and_blur(asset_store) -> None:
    scene = scene_create("uniform ee", 32, asset_store=asset_store)
    raytrace = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    skip = oi_create()
    skip.fields["optics"]["model"] = "skip"
    skip.fields["optics"]["offaxis_method"] = "skip"
    baseline = oi_compute(skip, scene, crop=True)

    band = raytrace.data["photons"].shape[2] // 2
    center = float(raytrace.data["photons"][raytrace.data["photons"].shape[0] // 2, raytrace.data["photons"].shape[1] // 2, band])
    corner = float(raytrace.data["photons"][0, 0, band])

    assert raytrace.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.allclose(oi_get(raytrace, "depth map"), 2.0)
    assert center > corner
    assert not np.allclose(raytrace.data["photons"], baseline.data["photons"])


def test_oi_compute_raytrace_builds_precomputed_psf_structure(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 30)
    result = oi_compute(oi, scene, crop=True)

    psf_struct = oi_get(result, "psf struct")
    sampled = oi_get(result, "sampledRTpsf")

    assert np.isclose(oi_get(result, "psf angle step"), 30.0)
    assert isinstance(psf_struct, dict)
    assert "sampAngles" in psf_struct
    assert "imgHeight" in psf_struct
    assert "wavelength" in psf_struct
    assert sampled is not None
    assert sampled.dtype == object
    assert sampled.ndim == 3
    assert np.array_equal(oi_get(result, "psfwavelength"), np.array([550.0]))
    assert oi_get(result, "rtpsfsize") == sampled[0, 0, 0].shape
    assert oi_get(result, "optics rtpsfsize") == oi_get(result, "rtpsf")["function"].shape
    assert oi_get(result, "raytrace optics name") == "Asphere 2mm"


def test_oi_compute_raytrace_crop_false_tracks_padding_and_depth_map(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    result = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=False)

    pad_rows, pad_cols = oi_get(result, "padding pixels")
    photons = np.asarray(result.data["photons"], dtype=float)
    depth_map = np.asarray(oi_get(result, "depth map"), dtype=float)
    base_rows, base_cols = scene.data["photons"].shape[:2]

    assert pad_rows > 0
    assert pad_cols > 0
    assert photons.shape[:2] == (base_rows + 2 * pad_rows, base_cols + 2 * pad_cols)
    assert depth_map.shape == photons.shape[:2]
    assert np.allclose(depth_map[pad_rows:-pad_rows, pad_cols:-pad_cols], 2.0)
    assert np.allclose(depth_map[:pad_rows, :], 0.0)
    assert np.allclose(depth_map[-pad_rows:, :], 0.0)
    assert np.allclose(depth_map[:, :pad_cols], 0.0)
    assert np.allclose(depth_map[:, -pad_cols:], 0.0)


def test_oi_get_set_raytrace_sample_angles_matches_matlab_surface(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    sample_angles = np.arange(0.0, 361.0, 45.0, dtype=float)
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf sample angles", sample_angles)

    assert np.array_equal(oi_get(oi, "psf sample angles"), sample_angles)
    assert np.isclose(oi_get(oi, "psf angle step"), 45.0)

    result = oi_compute(oi, scene, crop=True)

    assert np.array_equal(oi_get(result, "psf sample angles"), sample_angles)
    assert np.isclose(oi_get(result, "psf angle step"), 45.0)
    assert np.allclose(oi_get(result, "psf image heights", "mm"), oi_get(result, "psf image heights") * 1e3)


def test_oi_set_psfstruct_normalizes_matlab_style_metadata(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    psf_cells = np.empty((2, 2, 1), dtype=object)
    psf_cells[0, 0, 0] = np.ones((3, 3), dtype=float)
    psf_cells[0, 1, 0] = np.full((3, 3), 2.0, dtype=float)
    psf_cells[1, 0, 0] = np.full((3, 3), 3.0, dtype=float)
    psf_cells[1, 1, 0] = np.full((3, 3), 4.0, dtype=float)
    psf_struct = {
        "psf": psf_cells,
        "sampAngles": np.array([0.0, 180.0], dtype=float),
        "imgHeight": np.array([0.0, 1.5e-3], dtype=float),
        "wavelength": np.array([550.0], dtype=float),
        "opticsName": "Synthetic RT",
    }

    oi = oi_set(oi, "shift variant structure", psf_struct)

    exported = oi_get(oi, "psf struct")
    sampled = oi_get(oi, "sampledRTpsf")
    assert exported["psf"].shape == (2, 2, 1)
    assert sampled.shape == (2, 2, 1)
    assert sampled.dtype == object
    assert sampled[1, 1, 0].shape == (3, 3)
    assert np.array_equal(oi_get(oi, "psf sample angles"), np.array([0.0, 180.0]))
    assert np.isclose(oi_get(oi, "psf angle step"), 180.0)
    assert np.array_equal(oi_get(oi, "psf image heights"), np.array([0.0, 1.5e-3]))
    assert np.allclose(oi_get(oi, "psf image heights", "mm"), np.array([0.0, 1.5]))
    assert np.array_equal(oi_get(oi, "psf wavelength"), np.array([550.0]))
    assert oi_get(oi, "raytrace optics name") == "Synthetic RT"


def test_oi_get_set_raytrace_psf_metadata_before_compute(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "psf image heights", np.array([0.0, 1.0e-3, 2.0e-3], dtype=float))
    oi = oi_set(oi, "psf wavelength", np.array([450.0, 550.0], dtype=float))
    oi = oi_set(oi, "raytrace optics name", "Manual RT")

    assert np.array_equal(oi_get(oi, "psf image heights"), np.array([0.0, 1.0e-3, 2.0e-3]))
    assert np.allclose(oi_get(oi, "psf image heights", "mm"), np.array([0.0, 1.0, 2.0]))
    assert oi_get(oi, "psf image heights n") == 3
    assert np.array_equal(oi_get(oi, "psf wavelength"), np.array([450.0, 550.0]))
    assert oi_get(oi, "psf wavelength n") == 2
    assert oi_get(oi, "raytrace optics name") == "Manual RT"


def test_oi_compute_reuses_matlab_style_psf_struct(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    baseline = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)
    exported = dict(oi_get(baseline, "psf struct"))
    original_cells = np.asarray(exported["psf"], dtype=object)

    delta_cells = np.empty(original_cells.shape, dtype=object)
    for index in np.ndindex(delta_cells.shape):
        kernel = np.zeros_like(np.asarray(original_cells[index], dtype=float))
        kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
        delta_cells[index] = kernel

    exported["psf"] = delta_cells
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf struct", exported)

    assert np.allclose(oi_get(oi, "psf image heights"), np.asarray(exported["imgHeight"], dtype=float))
    assert np.allclose(oi_get(oi, "psf image heights", "mm"), np.asarray(exported["imgHeight"], dtype=float) * 1e3)

    result = oi_compute(oi, scene, crop=True)
    sampled = oi_get(result, "sampledRTpsf")

    assert np.allclose(np.asarray(sampled[0, 0, 0], dtype=float), np.asarray(delta_cells[0, 0, 0], dtype=float))
    assert np.allclose(oi_get(result, "psf image heights"), np.asarray(exported["imgHeight"], dtype=float))
    assert not np.allclose(result.data["photons"], baseline.data["photons"])


def test_oi_set_raw_raytrace_psf_metadata_updates_optics_and_invalidates_cache(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    assert oi_get(oi, "psf struct") is not None

    oi = oi_set(oi, "rtpsfspacing", np.array([0.0005, 0.00075], dtype=float))
    oi = oi_set(oi, "rtpsffieldheight", np.array([0.0, 0.5, 1.0], dtype=float))
    oi = oi_set(oi, "rtpsfwavelength", np.array([500.0, 600.0], dtype=float))

    assert oi_get(oi, "psf struct") is None
    assert oi_get(oi, "psf image heights").size == 0
    assert oi_get(oi, "psf wavelength").size == 0
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.5, 0.75]))
    assert np.allclose(oi_get(oi, "rtpsffieldheight", "mm"), np.array([0.0, 0.5, 1.0]))
    assert np.array_equal(oi_get(oi, "rtpsfwavelength"), np.array([500.0, 600.0]))


def test_oi_set_raw_raytrace_psf_data_supports_indexed_updates(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    other = np.asarray(oi_get(oi, "rtpsfdata", 0.0, 625.0), dtype=float)
    replacement = np.full_like(np.asarray(oi_get(oi, "rtpsfdata", 0.0, 550.0), dtype=float), 7.0)

    oi = oi_set(oi, "rtpsfdata", replacement, 0.0, 550.0)

    assert np.allclose(oi_get(oi, "rtpsfdata", 0.0, 550.0), replacement)
    assert np.allclose(oi_get(oi, "rtpsfdata", 0.0, 625.0), other)
    assert oi_get(oi, "optics rtpsfsize") == oi_get(oi, "rtpsf")["function"].shape


def test_oi_set_raw_raytrace_geometry_and_relillum_updates_tables(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)

    geometry_function = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    rel_illum_function = np.array([[1.0, 0.9], [0.8, 0.7]], dtype=float)

    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 0.5], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": geometry_function,
        },
    )
    oi = oi_set(
        oi,
        "rtrelillum",
        {
            "fieldHeight": np.array([0.0, 0.25], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": rel_illum_function,
        },
    )

    assert np.allclose(oi_get(oi, "rtgeomfieldheight", "mm"), np.array([0.0, 0.5]))
    assert np.array_equal(oi_get(oi, "rtgeomwavelength"), np.array([500.0, 600.0]))
    assert np.allclose(oi_get(oi, "rtgeomfunction", 600.0, "mm"), geometry_function[:, 1])
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), np.array([0.0, 0.25]))
    assert np.array_equal(oi_get(oi, "rtriwavelength"), np.array([500.0, 600.0]))
    assert np.allclose(oi_get(oi, "rtrifunction"), rel_illum_function)


def test_oi_set_raw_raytrace_geometry_supports_wavelength_index_updates(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    baseline_475 = np.asarray(oi_get(oi, "rtgeomfunction", 475.0, "mm"), dtype=float)
    replacement_550 = np.linspace(0.0, 2.0, baseline_475.size, dtype=float)

    oi = oi_set(oi, "rtgeomfunction", replacement_550, 550.0)

    assert np.allclose(oi_get(oi, "rtgeomfunction", 550.0, "mm"), replacement_550)
    assert np.allclose(oi_get(oi, "rtgeomfunction", 475.0, "mm"), baseline_475)


def test_oi_set_raw_raytrace_scalar_metadata_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "rtname", "Custom RT")
    oi = oi_set(oi, "rtopticsprogram", "Code V")
    oi = oi_set(oi, "rtlensfile", "custom.seq")
    oi = oi_set(oi, "rtefff#", 3.2)
    oi = oi_set(oi, "rtfnumber", 3.5)
    oi = oi_set(oi, "rtmag", -0.25)
    oi = oi_set(oi, "rtrefwave", 520.0)
    oi = oi_set(oi, "rtrefobjdist", 1.5)
    oi = oi_set(oi, "rtmaxfov", 25.0)
    oi = oi_set(oi, "rtefl", 0.004)
    oi = oi_set(oi, "rtcomputespacing", 2e-6)

    assert oi_get(oi, "rtname") == "Custom RT"
    assert oi_get(oi, "raytrace optics name") == "Custom RT"
    assert oi_get(oi, "rtopticsprogram") == "Code V"
    assert oi_get(oi, "rtlensfile") == "custom.seq"
    assert np.isclose(oi_get(oi, "rteffectivefnumber"), 3.2)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.5)
    assert np.isclose(oi_get(oi, "fnumber"), 3.5)
    assert np.isclose(oi_get(oi, "rtmagnification"), -0.25)
    assert np.isclose(oi_get(oi, "rtrefwave"), 520.0)
    assert np.isclose(oi_get(oi, "rtobjdist"), 1.5)
    assert np.isclose(oi_get(oi, "rtobjdist", "mm"), 1500.0)
    assert np.isclose(oi_get(oi, "rtfov"), 25.0)
    assert np.isclose(oi_get(oi, "rtefl"), 0.004)
    assert np.isclose(oi_get(oi, "focal length"), 0.004)
    assert np.isclose(oi_get(oi, "rtcomputespacing"), 2e-6)
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 2.0)


def test_oi_get_set_optics_prefixed_raytrace_parameters(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    geometry_update = np.linspace(0.0, 1.0, oi_get(oi, "rtgeomfieldheight").size, dtype=float)

    assert np.isclose(oi_get(oi, "optics rtfnumber"), oi_get(oi, "rtfnumber"))
    assert np.allclose(oi_get(oi, "optics rtpsfspacing", "um"), oi_get(oi, "rtpsfspacing", "um"))
    assert np.allclose(oi_get(oi, "optics rtgeomfieldheight", "mm"), oi_get(oi, "rtgeomfieldheight", "mm"))

    oi = oi_set(oi, "optics rtrefwave", 530.0)
    oi = oi_set(oi, "optics rtpsfspacing", np.array([0.0004, 0.0006], dtype=float))
    oi = oi_set(oi, "optics rtcomputespacing", 3e-6)
    oi = oi_set(oi, "optics rtgeomfunction", geometry_update, 400.0)

    assert np.isclose(oi_get(oi, "rtrefwave"), 530.0)
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.4, 0.6]))
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 3.0)
    assert np.allclose(oi_get(oi, "rtgeomfunction", 400.0, "mm"), geometry_update)


def test_oi_set_whole_raytrace_struct_normalizes_and_invalidates_cache(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    raytrace = {
        "name": "Whole RT",
        "program": "Code V",
        "lensFile": "whole_rt.seq",
        "effectiveFNumber": 2.8,
        "fNumber": 3.1,
        "referenceWavelength": 600.0,
        "objectDistance": 1500.0,
        "mag": -0.2,
        "effectiveFocalLength": 5.0,
        "maxfov": 12.0,
        "geometry": {
            "fieldHeight": np.array([0.0, 0.5], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        },
        "relIllum": {
            "fieldHeight": np.array([0.0, 0.25], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[1.0, 0.9], [0.8, 0.7]], dtype=float),
        },
        "psf": {
            "fieldHeight": np.array([0.0, 0.5], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "sampleSpacing": np.array([0.0004, 0.0006], dtype=float),
            "function": np.ones((3, 3, 2, 2), dtype=float),
        },
        "computation": {"psfSpacing": 4e-6},
    }

    oi = oi_set(oi, "raytrace", raytrace)

    assert oi_get(oi, "psf struct") is None
    assert oi_get(oi, "psf image heights").size == 0
    assert oi_get(oi, "rtname") == "Whole RT"
    assert oi_get(oi, "rtopticsprogram") == "Code V"
    assert oi_get(oi, "rtlensfile") == "whole_rt.seq"
    assert np.isclose(oi_get(oi, "rteffectivefnumber"), 2.8)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.1)
    assert np.isclose(oi_get(oi, "rtrefwave"), 600.0)
    assert np.isclose(oi_get(oi, "rtobjdist"), 1.5)
    assert np.isclose(oi_get(oi, "rtefl"), 0.005)
    assert np.allclose(oi_get(oi, "rtrifieldheight", "mm"), np.array([0.0, 0.25]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.4, 0.6]))
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 4.0)


def test_oi_set_whole_optics_struct_normalizes_raytrace_payload(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    optics = {
        "name": "Whole Optics",
        "fNumber": 2.9,
        "focalLength": 0.006,
        "transmittance": {
            "wave": np.array([500.0, 600.0], dtype=float),
            "scale": np.array([0.8, 0.9], dtype=float),
        },
        "rayTrace": {
            "name": "Whole Optics RT",
            "fNumber": 2.9,
            "effectiveFNumber": 2.6,
            "effectiveFocalLength": 6.0,
            "referenceWavelength": 550.0,
            "psf": {
                "fieldHeight": np.array([0.0], dtype=float),
                "wavelength": np.array([550.0], dtype=float),
                "sampleSpacing": np.array([0.0007, 0.0007], dtype=float),
                "function": np.ones((3, 3, 1, 1), dtype=float),
            },
        },
    }

    oi = oi_set(oi, "optics", optics)

    assert oi_get(oi, "model") == "raytrace"
    assert oi_get(oi, "psf struct") is None
    assert oi_get(oi, "raytrace optics name") == "Whole Optics RT"
    assert np.isclose(oi_get(oi, "fnumber"), 2.9)
    assert np.isclose(oi_get(oi, "focal length"), 0.006)
    assert np.array_equal(oi_get(oi, "transmittance wave"), np.array([500.0, 600.0]))
    assert np.allclose(oi_get(oi, "transmittance"), np.array([0.85]))
    assert np.allclose(oi_get(oi, "rtpsfspacing", "um"), np.array([0.7, 0.7]))


def test_oi_get_optics_roundtrips_matlab_style_raytrace_struct(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["fNumber"] = 3.4
    optics["rayTrace"]["referenceWavelength"] = 610.0
    optics["rayTrace"]["blocksPerFieldHeight"] = 7

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.4)
    assert np.isclose(oi_get(oi, "rtrefwave"), 610.0)
    assert oi_get(oi, "rt blocks per field height") == 7
    assert roundtrip["rayTrace"]["referenceWavelength"] == 610.0
    assert roundtrip["rayTrace"]["blocksPerFieldHeight"] == 7
    assert "raytrace" not in roundtrip


def test_oi_set_optics_accepts_normalized_nested_raytrace_scalar_fields(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["raytrace"] = {
        "f_number": 3.7,
        "magnification": -0.42,
    }
    optics.pop("rayTrace", None)

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.7)
    assert np.isclose(oi_get(oi, "rtmagnification"), -0.42)
    assert np.isclose(roundtrip["rayTrace"]["fNumber"], 3.7)
    assert np.isclose(roundtrip["rayTrace"]["mag"], -0.42)


def test_oi_set_optics_preserves_existing_raytrace_data_for_normalized_partial_update(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    original = oi_get(oi, "optics")
    original_geometry = np.asarray(original["rayTrace"]["geometry"]["function"], dtype=float).copy()
    original_psf_spacing = np.asarray(original["rayTrace"]["psf"]["sampleSpacing"], dtype=float).copy()

    optics = oi_get(oi, "optics")
    optics["raytrace"] = {
        "lens_file": "CustomLens.zmx",
        "computation": {"psf_spacing_m": 8e-6},
    }
    optics.pop("rayTrace", None)

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert oi_get(oi, "rtlensfile") == "CustomLens.zmx"
    assert np.isclose(oi_get(oi, "rtcomputespacing"), 8e-6)
    assert np.array_equal(roundtrip["rayTrace"]["geometry"]["function"], original_geometry)
    assert np.allclose(roundtrip["rayTrace"]["psf"]["sampleSpacing"], original_psf_spacing)
    assert np.isclose(roundtrip["rayTrace"]["referenceWavelength"], original["rayTrace"]["referenceWavelength"])


def test_oi_set_optics_preserves_existing_raytrace_data_for_raw_partial_update(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    original = oi_get(oi, "optics")
    original_geometry = np.asarray(original["rayTrace"]["geometry"]["function"], dtype=float).copy()
    original_psf_spacing = np.asarray(original["rayTrace"]["psf"]["sampleSpacing"], dtype=float).copy()

    optics = oi_get(oi, "optics")
    optics["rayTrace"] = {
        "lensFile": "CustomLens.zmx",
        "computation": {"psfSpacing": 8e-6},
    }

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert oi_get(oi, "rtlensfile") == "CustomLens.zmx"
    assert np.isclose(oi_get(oi, "rtcomputespacing"), 8e-6)
    assert np.array_equal(roundtrip["rayTrace"]["geometry"]["function"], original_geometry)
    assert np.allclose(roundtrip["rayTrace"]["psf"]["sampleSpacing"], original_psf_spacing)
    assert np.isclose(roundtrip["rayTrace"]["referenceWavelength"], original["rayTrace"]["referenceWavelength"])


def test_oi_set_optics_raw_nested_fnumber_overrides_exported_top_level_fnumber(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["rayTrace"] = {
        "fNumber": 3.3,
    }

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.3)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.3)
    assert np.isclose(roundtrip["rayTrace"]["fNumber"], 3.3)


def test_oi_set_optics_top_level_focal_length_overrides_exported_nested_effective_focal_length(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["focalLength"] = 0.009

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "focal length"), 0.009)
    assert np.isclose(oi_get(oi, "rteffectivefocallength"), 0.009)
    assert np.isclose(roundtrip["rayTrace"]["effectiveFocalLength"], 9.0)


def test_oi_set_optics_normalized_top_level_f_number_updates_raytrace(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["f_number"] = 3.2

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "fnumber"), 3.2)
    assert np.isclose(oi_get(oi, "rtfnumber"), 3.2)
    assert np.isclose(roundtrip["rayTrace"]["fNumber"], 3.2)


def test_oi_set_optics_normalized_top_level_focal_length_updates_raytrace(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["focal_length_m"] = 0.008

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(oi_get(oi, "focal length"), 0.008)
    assert np.isclose(oi_get(oi, "rteffectivefocallength"), 0.008)
    assert np.isclose(roundtrip["rayTrace"]["effectiveFocalLength"], 8.0)


def test_oi_set_optics_normalized_top_level_nominal_focal_length_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["nominal_focal_length_m"] = 0.012

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(roundtrip["nominalFocalLength"], 0.012)


def test_oi_set_optics_raw_top_level_nominal_focal_length_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["nominalFocalLength"] = 0.012

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.isclose(roundtrip["nominalFocalLength"], 0.012)


def test_oi_set_optics_raw_top_level_offaxis_roundtrips(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["offaxis"] = "skip"

    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert oi_get(oi, "offaxis method") == "skip"
    assert roundtrip["offaxis"] == "skip"


def test_oi_set_optics_transmittance_wave_preserves_scale_via_interpolation(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["transmittance"]["wave"] = np.array([450.0, 550.0, 650.0], dtype=float)
    optics["transmittance"]["scale"] = np.array([0.2, 0.5, 0.8], dtype=float)
    oi = oi_set(oi, "optics", optics)

    optics = oi_get(oi, "optics")
    optics["transmittance"]["wave"] = np.array([475.0, 625.0], dtype=float)
    oi = oi_set(oi, "optics", optics)

    roundtrip = oi_get(oi, "optics")
    assert np.array_equal(roundtrip["transmittance"]["wave"], np.array([475.0, 625.0]))
    assert np.allclose(roundtrip["transmittance"]["scale"], np.array([0.275, 0.725]))


def test_oi_set_optics_transmittance_scale_length_mismatch_raises(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)
    optics = oi_get(oi, "optics")
    optics["transmittance"]["scale"] = np.array([0.2, 0.5, 0.8], dtype=float)

    with pytest.raises(ValueError, match="Transmittance must match wave dimension."):
        oi_set(oi, "optics", optics)


def test_oi_compute_raytrace_rotates_psf_with_field_angle(asset_store) -> None:
    wave = np.array([550.0], dtype=float)
    scene = scene_create("uniform ee", 96, wave, asset_store=asset_store)
    scene = scene_set(scene, "fov", 10.0)
    photons = np.zeros_like(scene.data["photons"], dtype=float)
    center = photons.shape[0] // 2
    offset = 18
    photons[center, center + offset, 0] = 1.0
    photons[center - offset, center, 0] = 1.0
    scene = scene_set(scene, "photons", photons)

    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 30)
    raytrace = oi_compute(oi, scene, crop=True)
    plane = np.asarray(raytrace.data["photons"][:, :, 0], dtype=float)

    radius = 4
    right_patch = plane[center - radius : center + radius + 1, center + offset - radius : center + offset + radius + 1]
    top_patch = plane[center - offset - radius : center - offset + radius + 1, center - radius : center + radius + 1]
    right_patch = right_patch / max(float(np.sum(right_patch)), 1e-12)
    top_patch = top_patch / max(float(np.sum(top_patch)), 1e-12)

    plain_error = float(np.mean((top_patch - right_patch) ** 2))
    rotated_error = min(float(np.mean((top_patch - np.rot90(right_patch, k)) ** 2)) for k in (1, 3))

    assert rotated_error < plain_error


def test_rt_psf_interp_matches_raw_psf_without_resampling() -> None:
    oi = oi_create("ray trace")
    kernel = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 4.0],
        ],
        dtype=float,
    )
    oi = oi_set(
        oi,
        "rtpsf",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "sampleSpacing": np.array([0.1, 0.1], dtype=float),
            "function": np.stack((kernel, np.zeros_like(kernel)), axis=2)[:, :, :, None],
        },
    )
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "function": np.array([[0.0], [1.0]], dtype=float),
        },
    )

    interp = rt_psf_interp(oi, field_height_m=0.0, wavelength_nm=550.0)

    assert np.allclose(interp, kernel)


def test_rt_psf_interp_interpolates_field_height_and_rotates() -> None:
    oi = oi_create("ray trace")
    psf_stack = np.zeros((5, 5, 2, 1), dtype=float)
    psf_stack[1, 2, 0, 0] = 1.0
    psf_stack[2, 3, 1, 0] = 1.0
    oi = oi_set(
        oi,
        "rtpsf",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "sampleSpacing": np.array([0.1, 0.1], dtype=float),
            "function": psf_stack,
        },
    )
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "function": np.array([[0.0], [1.0]], dtype=float),
        },
    )

    interp = rt_psf_interp(oi, field_height_m=0.5e-3, wavelength_nm=550.0)
    rotated = rt_psf_interp(oi, field_height_m=0.5e-3, field_angle_deg=90.0, wavelength_nm=550.0)

    expected = 0.5 * psf_stack[:, :, 0, 0] + 0.5 * psf_stack[:, :, 1, 0]
    assert np.allclose(interp, expected)
    assert np.allclose(rotated, np.rot90(expected))


def test_rt_psf_interp_resamples_to_requested_grid() -> None:
    oi = oi_create("ray trace")
    kernel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    oi = oi_set(
        oi,
        "rtpsf",
        {
            "fieldHeight": np.array([0.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "sampleSpacing": np.array([0.1, 0.1], dtype=float),
            "function": kernel[:, :, None, None],
        },
    )
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([550.0], dtype=float),
            "function": np.array([[0.0], [1.0]], dtype=float),
        },
    )

    grid = np.array([0.0, 1.0e-4], dtype=float)
    x_grid, y_grid = np.meshgrid(grid, grid)
    resampled = rt_psf_interp(oi.fields["optics"], field_height_m=0.0, wavelength_nm=550.0, x_grid_m=x_grid, y_grid_m=y_grid)

    assert resampled.shape == x_grid.shape
    assert np.isclose(resampled[0, 0], kernel[1, 1])
    assert np.isclose(resampled[1, 1], kernel[2, 2])


def test_rt_di_interp_and_rt_ri_interp_use_nearest_wavelength() -> None:
    oi = oi_create("ray trace")
    oi = oi_set(
        oi,
        "rtgeometry",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[1.0, 10.0], [2.0, 20.0]], dtype=float),
        },
    )
    oi = oi_set(
        oi,
        "rtrelillum",
        {
            "fieldHeight": np.array([0.0, 1.0], dtype=float),
            "wavelength": np.array([500.0, 600.0], dtype=float),
            "function": np.array([[0.5, 0.9], [0.25, 0.8]], dtype=float),
        },
    )

    assert np.array_equal(rt_di_interp(oi, 540.0), np.array([1.0, 2.0]))
    assert np.array_equal(rt_di_interp(oi.fields["optics"], 580.0), np.array([10.0, 20.0]))
    assert np.array_equal(rt_ri_interp(oi, 520.0), np.array([0.5, 0.25]))
    assert np.array_equal(rt_ri_interp(oi.fields["optics"], 590.0), np.array([0.9, 0.8]))


def test_oi_compute_raytrace_uses_raw_curve_helpers(asset_store, monkeypatch) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_create("ray trace", asset_store=asset_store)

    di_calls: list[float] = []
    ri_calls: list[float] = []
    original_di = optics_module.rt_di_interp
    original_ri = optics_module.rt_ri_interp

    def record_di(optics: object, wavelength_nm: float) -> np.ndarray:
        di_calls.append(float(wavelength_nm))
        return original_di(optics, wavelength_nm)

    def record_ri(optics: object, wavelength_nm: float) -> np.ndarray:
        ri_calls.append(float(wavelength_nm))
        return original_ri(optics, wavelength_nm)

    monkeypatch.setattr(optics_module, "rt_di_interp", record_di)
    monkeypatch.setattr(optics_module, "rt_ri_interp", record_ri)

    result = oi_compute(oi, scene, crop=True)

    assert result.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert di_calls == [550.0]
    assert ri_calls == [550.0]


def test_rt_sample_heights_matches_matlab_truncation_rule() -> None:
    all_heights = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    data_height = np.array([[0.1, 0.9], [1.4, 1.6]], dtype=float)

    img_height, max_data_height = rt_sample_heights(all_heights, data_height)

    assert np.array_equal(img_height, np.array([0.0, 0.5, 1.0, 2.0]))
    assert np.isclose(max_data_height, 1.6)


def test_rt_block_center_matches_matlab_formula() -> None:
    center = rt_block_center(2, 3, np.array([8, 16], dtype=int))

    assert np.allclose(center, np.array([12.0, 40.0]))


def test_rt_extract_block_returns_matlab_style_indices() -> None:
    plane = np.arange(1, 1 + 6 * 8, dtype=float).reshape(6, 8)

    block, r_list, c_list = rt_extract_block(plane, np.array([2, 3], dtype=int), 2, 2)

    assert np.array_equal(r_list, np.array([3, 4]))
    assert np.array_equal(c_list, np.array([4, 5, 6]))
    assert np.array_equal(block, plane[2:4, 3:6])


def test_rt_insert_block_adds_filtered_data_at_matlab_block_origin() -> None:
    img = np.zeros((8, 10), dtype=float)
    filtered = np.ones((4, 5), dtype=float)

    inserted = rt_insert_block(img, filtered, np.array([2, 3], dtype=int), np.array([1, 1], dtype=int), 2, 2)

    expected = np.zeros_like(img)
    expected[2:6, 3:8] = 1.0
    assert np.array_equal(inserted, expected)


def test_rt_choose_block_size_matches_upstream_formula(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    n_blocks, block_samples, irrad_padding = rt_choose_block_size(scene, oi)

    field_heights = np.asarray(oi_get(oi, "rtgeometryfieldheight", "mm"), dtype=float)
    diagonal_mm = float(oi_get(oi, "diagonal")) * 1e3 / 2.0
    n_heights = int(np.argmin(np.abs(field_heights - diagonal_mm))) + 1
    expected_n_blocks = 4 * n_heights + 1
    expected_block_samples = np.array(
        [
            max(1, int(2 ** np.ceil(np.log2(max(scene_get(scene, "rows") / expected_n_blocks, 1.0))))),
            max(1, int(2 ** np.ceil(np.log2(max(scene_get(scene, "cols") / expected_n_blocks, 1.0))))),
        ],
        dtype=int,
    )
    expected_padding = np.ceil(
        (np.array(
            [
                expected_n_blocks * expected_block_samples[0] - scene_get(scene, "rows"),
                expected_n_blocks * expected_block_samples[1] - scene_get(scene, "cols"),
            ],
            dtype=float,
        ))
        / 2.0
    ).astype(int)

    assert n_blocks == expected_n_blocks
    assert np.array_equal(block_samples, expected_block_samples)
    assert np.array_equal(irrad_padding, expected_padding)


def test_rt_choose_block_size_uses_public_field_height_indexing(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    field_heights = np.asarray(oi_get(oi, "rtgeometryfieldheight", "mm"), dtype=float)
    diagonal_mm = float(oi_get(oi, "diagonal")) * 1e3 / 2.0

    n_blocks, _, _ = rt_choose_block_size(scene, oi)

    assert n_blocks == 4 * int(ie_field_height_to_index(field_heights, diagonal_mm)) + 1


def test_rt_otf_returns_padded_filtered_cube(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    result = rt_otf(scene, stage)
    n_blocks, block_samples, _ = rt_choose_block_size(scene, stage)
    block_padding = block_samples // 2
    expected_shape = (
        int(n_blocks * block_samples[0] + 2 * block_padding[0]),
        int(n_blocks * block_samples[1] + 2 * block_padding[1]),
        1,
    )

    assert result.shape == expected_shape
    assert np.all(result >= 0.0)
    assert np.sum(result) > 0.0


def test_rt_otf_uses_rt_blocks_per_field_height_setting(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    stage = oi_set(stage, "rt blocks per field height", 6)

    result = rt_otf(scene, stage)
    n_blocks, block_samples, _ = rt_choose_block_size(scene, stage, steps_fh=6)
    block_padding = block_samples // 2
    expected_shape = (
        int(n_blocks * block_samples[0] + 2 * block_padding[0]),
        int(n_blocks * block_samples[1] + 2 * block_padding[1]),
        1,
    )

    assert oi_get(stage, "rt blocks per field height") == 6
    assert result.shape == expected_shape


def test_rt_filtered_block_support_matches_block_layout(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    _, block_samples, _ = rt_choose_block_size(scene, stage)
    block_padding = block_samples // 2

    block_x, block_y, mm_row, mm_col = rt_filtered_block_support(stage, block_samples, block_padding)

    assert block_x.ndim == 1
    assert block_y.ndim == 1
    assert block_x.size == int(block_samples[1] + 2 * block_padding[1])
    assert block_y.size == int(block_samples[0] + 2 * block_padding[0])
    assert np.isclose(np.diff(block_x)[0], mm_col)
    assert np.isclose(np.diff(block_y)[0], mm_row)
    assert np.any(np.isclose(block_x, 0.0))
    assert np.any(np.isclose(block_y, 0.0))


def test_rt_otf_filtered_block_support_matches_public_helper(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    _, block_samples, _ = rt_choose_block_size(scene, stage)
    block_padding = block_samples // 2

    block_x, block_y, mm_row, mm_col = rt_filtered_block_support(stage, block_samples, block_padding)
    result = rt_otf(scene, stage)

    expected_rows = int(block_samples[0] + 2 * block_padding[0])
    expected_cols = int(block_samples[1] + 2 * block_padding[1])
    assert block_y.size == expected_rows
    assert block_x.size == expected_cols
    assert mm_row > 0.0
    assert mm_col > 0.0
    assert result.shape[0] >= expected_rows
    assert result.shape[1] >= expected_cols


def test_rt_synthetic_builds_normalized_raytrace_optics(asset_store) -> None:
    oi = oi_create("ray trace", asset_store=asset_store)

    optics = rt_synthetic(oi, spread_limits=(3.0, 5.0), xy_ratio=0.3)

    assert optics["model"] == "raytrace"
    assert optics["name"] == "Synthetic Gaussian"
    assert np.array_equal(optics["transmittance"]["wave"], np.array([450.0, 550.0, 650.0]))
    assert optics["raytrace"]["program"] == "Zemax"
    assert optics["raytrace"]["lens_file"] == "Synthetic Gaussian"
    assert optics["raytrace"]["psf"]["function"].shape[0:2] == (128, 128)
    assert optics["raytrace"]["psf"]["function"].shape[2] == 21
    assert optics["raytrace"]["psf"]["function"].shape[3] == 3
    assert np.allclose(np.sum(optics["raytrace"]["psf"]["function"][:, :, 0, 0]), 1.0)


def test_oi_compute_accepts_rt_synthetic_optics(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "optics", rt_synthetic(oi, spread_limits=(2.0, 3.0), xy_ratio=0.5))

    result = oi_compute(oi, scene, crop=True)

    assert result.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert oi_get(result, "raytrace optics name") == "Synthetic Gaussian"
    assert oi_get(result, "sampledRTpsf") is not None


def test_rt_otf_runs_with_rt_synthetic_optics(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_create("ray trace", asset_store=asset_store)
    oi = oi_set(oi, "optics", rt_synthetic(oi, spread_limits=(2.0, 4.0), xy_ratio=0.25))
    stage = rt_geometry(oi, scene)

    result = rt_otf(scene, stage)

    assert result.ndim == 3
    assert result.shape[2] == 1
    assert np.sum(result) > 0.0


def test_optics_rt_synthetic_script_workflow(asset_store) -> None:
    scene = scene_create("point array", 256, asset_store=asset_store)
    scene = scene_set(scene, "h fov", 3.0)
    scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))

    oi = oi_create("ray trace", asset_store=asset_store)
    optics = rt_synthetic(oi, spread_limits=(1.0, 3.0), xy_ratio=1.6)
    oi = oi_set(oi, "optics", optics)
    scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
    oi = oi_compute(oi, scene)

    raytrace = oi.fields["optics"]["raytrace"]
    psf = np.asarray(raytrace["psf"]["function"], dtype=float)
    center_psf = psf[:, :, 0, 1]
    edge_psf = psf[:, :, -1, 1]
    coords = np.arange(psf.shape[1], dtype=float) - (psf.shape[1] // 2)
    center_row = center_psf[center_psf.shape[0] // 2, :]
    edge_row = edge_psf[edge_psf.shape[0] // 2, :]
    center_variance = float(np.sum((coords**2) * center_row) / np.sum(center_row))
    edge_variance = float(np.sum((coords**2) * edge_row) / np.sum(edge_row))

    photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.array([550.0, 650.0], dtype=float))
    assert oi_get(oi, "raytrace optics name") == "Synthetic Gaussian"
    assert photons.shape == (312, 312, 2)
    assert center_variance < edge_variance
    assert np.allclose(np.sum(psf[:, :, 0, 1]), 1.0)
    assert np.allclose(np.sum(psf[:, :, -1, 1]), 1.0)
    assert float(np.max(photons[:, :, 0])) > 0.0


def test_optics_rt_gridlines_script_workflow(asset_store) -> None:
    scene = scene_create("grid lines", [384, 384], 48, asset_store=asset_store)
    scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))
    scene = scene_set(scene, "hfov", 45.0)
    scene = scene_set(scene, "name", "rtDemo-Large-grid")

    oi = oi_create("ray trace", asset_store.resolve("data/optics/zmWideAngle.mat"), asset_store=asset_store)
    oi = oi_set(oi, "wangular", scene_get(scene, "wangular"))
    oi = oi_set(oi, "wave", scene_get(scene, "wave"))
    scene = scene_set(scene, "distance", 2.0)
    oi = oi_set(oi, "optics rtObjectDistance", scene_get(scene, "distance", "mm"))

    raytrace_fov = float(oi_get(oi, "optics rt fov"))
    target_diagonal_fov = max(raytrace_fov - 1.0, 0.1)
    adjusted_hfov = float(
        np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(target_diagonal_fov) / 2.0) / np.sqrt(2.0)))
    )
    scene = scene_set(scene, "hfov", adjusted_hfov)

    geometry_oi = rt_geometry(oi, scene)
    psf_struct = rt_precompute_psf(geometry_oi, angle_step_deg=20.0)
    stepwise_oi = oi_set(geometry_oi, "psf struct", psf_struct)
    stepwise_oi = rt_precompute_psf_apply(stepwise_oi, angle_step_deg=20.0)

    automated_rt = oi_compute(oi_set(oi.clone(), "optics model", "ray trace"), scene)
    diffraction_oi = oi_set(automated_rt.clone(), "optics model", "diffraction limited")
    diffraction_oi = oi_set(diffraction_oi, "optics fnumber", oi_get(automated_rt, "rtfnumber"))
    diffraction_oi = oi_compute(diffraction_oi, scene)

    scene_small = scene_set(scene.clone(), "name", "rt-Small-Grid")
    scene_small = scene_set(scene_small, "fov", 20.0)
    rt_small = oi_compute(automated_rt.clone(), scene_small)
    dl_small = oi_compute(diffraction_oi.clone(), scene_small)

    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.array([550.0, 650.0], dtype=float))
    assert np.isclose(raytrace_fov, float(oi_get(oi, "optics rt fov")))
    assert adjusted_hfov < 45.0
    assert np.array_equal(np.asarray(oi_get(geometry_oi, "size"), dtype=int), np.array([384, 384], dtype=int))
    assert np.asarray(psf_struct["imgHeight"], dtype=float).shape == (6,)
    assert np.asarray(psf_struct["sampAngles"], dtype=float).shape == (19,)
    assert np.array_equal(np.asarray(oi_get(stepwise_oi, "size"), dtype=int), np.array([436, 436], dtype=int))
    assert np.array_equal(np.asarray(oi_get(automated_rt, "size"), dtype=int), np.array([436, 436], dtype=int))
    assert np.array_equal(np.asarray(oi_get(diffraction_oi, "size"), dtype=int), np.array([480, 480], dtype=int))
    assert np.isclose(float(scene_get(scene_small, "fov")), 20.0)
    assert np.array_equal(np.asarray(oi_get(rt_small, "size"), dtype=int), np.array([484, 484], dtype=int))
    assert np.array_equal(np.asarray(oi_get(dl_small, "size"), dtype=int), np.array([480, 480], dtype=int))


def test_optics_rt_psf_script_workflow(asset_store) -> None:
    scene = scene_create("point array", 512, 32, asset_store=asset_store)
    scene = scene_interpolate_w(scene, np.arange(450.0, 651.0, 100.0, dtype=float))
    scene = scene_set(scene, "hfov", 10.0)
    scene = scene_set(scene, "name", "psf Point Array")

    oi = oi_create("ray trace", asset_store.resolve("data/optics/rtZemaxExample.mat"), asset_store=asset_store)
    scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
    oi = oi_set(oi, "name", "ray trace case")
    oi = oi_set(oi, "optics model", "ray trace")
    oi = oi_compute(oi, scene)

    sampled_rt_psf = np.asarray(oi_get(oi, "sampledRTpsf"), dtype=object)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.array([450.0, 550.0, 650.0], dtype=float))
    assert np.isclose(float(scene_get(scene, "fov")), 10.0)
    assert np.array_equal(np.asarray(oi_get(oi, "size"), dtype=int), np.array([564, 564], dtype=int))
    assert np.isclose(float(oi_get(oi, "rtfnumber")), 4.999973)
    assert oi_get(oi, "rtname") == "Asphere 2mm"
    assert np.array_equal(np.asarray(oi_get(oi, "psf sample angles"), dtype=float), np.arange(0.0, 361.0, 10.0))
    assert np.asarray(oi_get(oi, "psf image heights", "mm"), dtype=float).shape == (6,)
    assert np.array_equal(np.asarray(oi_get(oi, "psf wavelength"), dtype=float), np.array([450.0, 550.0, 650.0], dtype=float))
    assert sampled_rt_psf.shape == (37, 6, 3)

    oi_dl = oi_set(oi.clone(), "name", "diffraction case")
    optics = oi_get(oi_dl, "optics")
    oi_dl = oi_set(oi_dl, "optics fnumber", float(optics["rayTrace"]["fNumber"]) * 0.8)
    oi_dl = oi_set(oi_dl, "optics model", "diffraction limited")
    oi_dl = oi_compute(oi_dl, scene)

    assert np.array_equal(np.asarray(oi_get(oi_dl, "size"), dtype=int), np.array([640, 640], dtype=int))
    assert np.isclose(float(oi_get(oi_dl, "fnumber")), 3.9999784)


def test_optics_rt_psf_view_script_workflow(asset_store) -> None:
    scene = scene_create("point array", 384, asset_store=asset_store)
    scene = scene_set(scene, "h fov", 4.0)
    scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics", rt_synthetic(oi, spread_limits=(1.0, 5.0), xy_ratio=1.6))
    oi = oi_compute(oi, scene)

    sampled_rt_psf = np.asarray(oi_get(oi, "sampledRTpsf"), dtype=object)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.array([550.0, 650.0], dtype=float))
    assert np.isclose(float(scene_get(scene, "fov")), 4.0)
    assert np.array_equal(np.asarray(oi_get(oi, "size"), dtype=int), np.array([448, 448], dtype=int))
    assert np.array_equal(np.asarray(oi_get(oi, "psf sample angles"), dtype=float), np.arange(0.0, 361.0, 10.0))
    assert np.allclose(np.asarray(oi_get(oi, "psf image heights", "mm"), dtype=float), np.array([0.0, 0.05, 0.1, 0.15], dtype=float))
    assert np.array_equal(np.asarray(oi_get(oi, "psf wavelength"), dtype=float), np.array([550.0, 650.0], dtype=float))
    assert sampled_rt_psf.shape == (37, 4, 2)


def test_rt_file_names_matches_zemax_naming(tmp_path) -> None:
    di_name, ri_name, psf_name_list, cra_name = rt_file_names(
        "CookeLens.ZMX",
        np.array([500.0, 600.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        directory=tmp_path,
    )

    assert di_name.endswith("CookeLens_DI_.dat")
    assert ri_name.endswith("CookeLens_RI_.dat")
    assert cra_name.endswith("CookeLens_CRA_.dat")
    assert psf_name_list.shape == (2, 2)
    assert str(psf_name_list[1, 1]).endswith("CookeLens_2D_PSF_Fld2_Wave2.dat")


def test_rt_root_path_points_at_upstream_raytrace_snapshot() -> None:
    root = Path(rtRootPath())

    assert root.name == "raytrace"
    assert (root / "rtImageRotate.m").is_file()


def test_rt_image_rotate_matches_legacy_zero_border_behavior() -> None:
    source = np.arange(1.0, 26.0, dtype=float).reshape(5, 5)
    rotated = rtImageRotate(source, 0.0)
    expected = np.zeros_like(source)
    expected[1:4, 1:4] = source[1:4, 1:4]

    center = np.zeros((5, 5), dtype=float)
    center[2, 2] = 1.0
    rotated_center = rtImageRotate(center, 37.0)

    assert np.allclose(rotated, expected)
    assert np.isclose(float(rotated_center[2, 2]), 1.0)
    assert np.isclose(float(rotated_center[1, 2]), float(rotated_center[2, 1]))
    assert np.isclose(float(rotated_center[1, 2]), float(rotated_center[2, 3]))
    assert np.isclose(float(rotated_center[1, 2]), float(rotated_center[3, 2]))


def test_rt_file_names_normalizes_windows_style_lens_paths(tmp_path) -> None:
    di_name, ri_name, psf_name_list, cra_name = rt_file_names(
        r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens.ZMX",
        np.array([500.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        directory=tmp_path,
    )

    assert di_name.endswith("CookeLens_DI_.dat")
    assert ri_name.endswith("CookeLens_RI_.dat")
    assert cra_name.endswith("CookeLens_CRA_.dat")
    assert str(psf_name_list[1, 0]).endswith("CookeLens_2D_PSF_Fld2_Wave1.dat")


def test_zemax_read_header_and_load_parse_text_output(tmp_path) -> None:
    psf_file = tmp_path / "Test_2D_PSF_Fld1_Wave1.dat"
    psf_file.write_text(
        "spacing is 0.5000 microns\n"
        "area is 1.0000 microns\n"
        "normalized.\n"
        "1 2 3 4\n",
        encoding="latin1",
    )

    spacing_um, area_um = zemax_read_header(psf_file)
    kernel = zemax_load(psf_file, 2)

    assert np.isclose(spacing_um, 0.5)
    assert np.isclose(area_um, 1.0)
    assert np.array_equal(kernel, np.array([[2.0, 4.0], [1.0, 3.0]], dtype=float))


def test_rt_import_data_builds_usable_raytrace_optics(tmp_path, asset_store) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)
    oi = oi_create("ray trace", imported_optics, asset_store=asset_store)
    scene = scene_create("uniform ee", 16, np.array([500.0, 600.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi, scene)

    assert optics_file is None
    assert imported_optics["model"] == "raytrace"
    assert imported_optics["raytrace"]["program"] == "zemax"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)
    assert imported_optics["raytrace"]["relative_illumination"]["function"].shape == (2, 2)
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)
    assert np.allclose(imported_optics["raytrace"]["psf"]["sample_spacing_mm"], np.array([0.0005, 0.0005]))
    assert np.isclose(imported_optics["raytrace"]["computation"]["psf_spacing_m"], 2.5e-7)
    assert stage.data["photons"].shape == scene.data["photons"].shape


def test_rt_import_data_preserves_requested_program_label(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)

    imported_optics, optics_file = rt_import_data(p_file_full=params_file, rt_program="ZeMaX")

    assert optics_file is None
    assert imported_optics["raytrace"]["program"] == "ZeMaX"


def test_rt_import_data_normalizes_windows_style_base_lens_paths(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        lens_file=r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens.ZMX",
        base_lens_file_name=r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["raytrace"]["lens_file"] == r"C:\PROGRAM FILES\ZEMAX\LENSES\CookeLens.ZMX"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_parses_multiline_wave_vector_with_continuation(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        wave_assignment="[500 ...\n 600]",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert np.array_equal(imported_optics["raytrace"]["geometry"]["wavelength_nm"], np.array([500.0, 600.0]))
    assert np.array_equal(imported_optics["raytrace"]["psf"]["wavelength_nm"], np.array([500.0, 600.0]))


def test_rt_import_data_parses_column_vector_wave_syntax(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        wave_assignment="[500; 600]",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert np.array_equal(imported_optics["raytrace"]["relative_illumination"]["wavelength_nm"], np.array([500.0, 600.0]))
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_parses_transposed_row_vector_wave_syntax(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        wave_assignment="[500 600]'",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert np.array_equal(imported_optics["raytrace"]["geometry"]["wavelength_nm"], np.array([500.0, 600.0]))
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_parses_legacy_base_lens_line_without_semicolon(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        base_lens_has_semicolon=False,
    )

    imported_optics, optics_file = rt_import_data(p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["raytrace"]["lens_file"] == "CookeLens.ZMX"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)


def test_rt_import_data_rejects_odd_psf_size_from_isetparams(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        psf_size_assignment=3,
    )

    with pytest.raises(ValueError, match="PSF size must be even"):
        rt_import_data(p_file_full=params_file)


def test_rt_import_data_accepts_bundle_directory(tmp_path) -> None:
    _write_mock_zemax_bundle(tmp_path)

    imported_optics, optics_file = rt_import_data(p_file_full=tmp_path)

    assert optics_file is None
    assert imported_optics["raytrace"]["lens_file"] == "CookeLens.ZMX"
    assert imported_optics["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_rt_import_data_accepts_legacy_isetparms_filename(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        params_file_name="ISETPARMS.TXT",
    )

    imported_optics, optics_file = rt_import_data(p_file_full=tmp_path)

    assert optics_file is None
    assert params_file.name == "ISETPARMS.TXT"
    assert imported_optics["raytrace"]["geometry"]["function"].shape == (2, 2)


def test_rt_import_data_preserves_existing_optics_fields_and_effective_top_level_state(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "name": "Existing Optics",
        "compute_method": "customrt",
        "aberration_scale": 0.25,
        "offaxis_method": "cos4th",
        "transmittance": {
            "wave": np.array([500.0, 600.0], dtype=float),
            "scale": np.array([0.7, 0.8], dtype=float),
        },
        "focal_length_m": 0.123,
        "f_number": 9.9,
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing Optics"
    assert imported_optics["compute_method"] == "customrt"
    assert np.isclose(imported_optics["aberration_scale"], 0.25)
    assert imported_optics["offaxis_method"] == "cos4th"
    assert np.array_equal(imported_optics["transmittance"]["wave"], np.array([500.0, 600.0]))
    assert np.array_equal(imported_optics["transmittance"]["scale"], np.array([0.7, 0.8]))
    assert np.isclose(imported_optics["focal_length_m"], 0.006)
    assert np.isclose(imported_optics["f_number"], 1.8)
    assert np.isclose(imported_optics["raytrace"]["f_number"], 2.0)
    assert np.isclose(imported_optics["raytrace"]["effective_f_number"], 1.8)


def test_rt_import_data_preserves_existing_compute_spacing_when_bundle_omits_it(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        psf_spacing_assignment_mm=None,
    )
    existing = {
        "name": "Existing Optics",
        "raytrace": {
            "computation": {
                "psf_spacing_m": 7.5e-6,
            },
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert np.isclose(imported_optics["raytrace"]["computation"]["psf_spacing_m"], 7.5e-6)


def test_rt_import_data_preserves_existing_raytrace_name_independently(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "name": "Existing Optics",
        "raytrace": {
            "name": "Existing RT Name",
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing Optics"
    assert imported_optics["raytrace"]["name"] == "Existing RT Name"


def test_rt_import_data_preserves_raw_matlab_style_optics_fields(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(
        tmp_path,
        psf_spacing_assignment_mm=None,
    )
    existing = {
        "name": "Existing Optics",
        "computeMethod": "customrt",
        "aberrationScale": 0.5,
        "offaxis": "cos4th",
        "transmittance": {
            "wave": np.array([500.0, 600.0], dtype=float),
            "scale": np.array([0.6, 0.7], dtype=float),
        },
        "rayTrace": {
            "name": "Existing RT Name",
            "computation": {
                "psfSpacing": 9e-6,
            },
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing Optics"
    assert imported_optics["compute_method"] == "customrt"
    assert np.isclose(imported_optics["aberration_scale"], 0.5)
    assert imported_optics["offaxis_method"] == "cos4th"
    assert np.array_equal(imported_optics["transmittance"]["wave"], np.array([500.0, 600.0]))
    assert np.array_equal(imported_optics["transmittance"]["scale"], np.array([0.6, 0.7]))
    assert imported_optics["raytrace"]["name"] == "Existing RT Name"
    assert np.isclose(imported_optics["raytrace"]["computation"]["psf_spacing_m"], 9e-6)


def test_rt_import_data_uses_existing_raytrace_name_as_top_level_fallback(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "rayTrace": {
            "name": "Existing RT Name",
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["name"] == "Existing RT Name"
    assert imported_optics["raytrace"]["name"] == "Existing RT Name"


def test_rt_import_data_preserves_existing_blocks_per_field_height(tmp_path) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)
    existing = {
        "rayTrace": {
            "blocksPerFieldHeight": 7,
        },
    }

    imported_optics, optics_file = rt_import_data(existing, p_file_full=params_file)

    assert optics_file is None
    assert imported_optics["raytrace"]["blocks_per_field_height"] == 7


def test_oi_create_raytrace_accepts_isetparams_file(tmp_path, asset_store) -> None:
    params_file = _write_mock_zemax_bundle(tmp_path)

    oi = oi_create("ray trace", params_file, asset_store=asset_store)
    exported = oi_get(oi, "optics")

    assert oi.fields["optics"]["model"] == "raytrace"
    assert np.isclose(oi.fields["optics"]["f_number"], 1.8)
    assert oi_get(oi, "rtlensfile") == "CookeLens.ZMX"
    assert np.isclose(oi_get(oi, "fnumber"), 2.0)
    assert np.isclose(exported["fNumber"], 2.0)
    assert np.isclose(oi_get(oi, "rtcomputespacing", "um"), 0.25)
    assert oi.fields["optics"]["raytrace"]["psf"]["function"].shape == (2, 2, 2, 2)


def test_oi_create_raytrace_accepts_isetparams_directory(tmp_path, asset_store) -> None:
    _write_mock_zemax_bundle(tmp_path)

    oi = oi_create("ray trace", tmp_path, asset_store=asset_store)

    assert oi.fields["optics"]["model"] == "raytrace"
    assert oi_get(oi, "raytraceopticsname") == "CookeLens"


def test_oi_create_raytrace_accepts_legacy_isetparms_directory(tmp_path, asset_store) -> None:
    _write_mock_zemax_bundle(
        tmp_path,
        params_file_name="ISETPARMS.TXT",
    )

    oi = oi_create("ray trace", tmp_path, asset_store=asset_store)

    assert oi.fields["optics"]["model"] == "raytrace"
    assert oi_get(oi, "raytraceopticsname") == "CookeLens"


def test_oi_create_raytrace_directory_without_bundle_stays_unsupported(tmp_path, asset_store) -> None:
    with pytest.raises(UnsupportedOptionError, match="ray trace optics"):
        oi_create("ray trace", tmp_path, asset_store=asset_store)


def test_oi_create_raytrace_directory_surfaces_malformed_bundle_errors(tmp_path, asset_store) -> None:
    (tmp_path / "ISETPARAMS.txt").write_text("lensFile='CookeLens.ZMX';\n", encoding="latin1")

    with pytest.raises(ValueError, match="Missing Zemax parameter"):
        oi_create("ray trace", tmp_path, asset_store=asset_store)


def test_rt_psf_grid_matches_oi_sample_spacing(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=True)

    x_grid, y_grid, sample_spacing = rt_psf_grid(oi, "m")

    assert x_grid.shape == y_grid.shape
    assert x_grid.ndim == 2
    assert sample_spacing.shape == (2,)
    assert np.isclose(np.diff(x_grid[0])[0], sample_spacing[0])
    assert np.isclose(np.diff(y_grid[:, 0])[0], sample_spacing[1])
    assert np.any(np.isclose(x_grid, 0.0))
    assert np.any(np.isclose(y_grid, 0.0))


def test_rt_angle_lut_returns_matlab_style_indices(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 45.0)
    oi = oi_compute(oi, scene, crop=True)

    angle_lut = rt_angle_lut(oi)

    assert angle_lut.shape == (360, 2)
    assert np.all(angle_lut[:, 0] >= 1)
    assert np.all(angle_lut[:, 0] <= len(oi_get(oi, "psf sample angles")) - 1)
    assert np.all(angle_lut[:, 1] >= 0.0)
    assert np.all(angle_lut[:, 1] <= 1.0)


def test_rt_geometry_returns_uncropped_raytrace_stage(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    assert stage.data["photons"].shape == scene.data["photons"].shape
    assert np.allclose(oi_get(stage, "depth map"), 2.0)
    assert oi_get(stage, "padding pixels") == (0, 0)
    assert np.allclose(
        oi_get(stage, "samplespacing"),
        np.array([oi_get(stage, "hspatialresolution"), oi_get(stage, "wspatialresolution")], dtype=float),
    )


def test_rt_precompute_psf_returns_matlab_style_struct(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_set(oi_create("ray trace", asset_store=asset_store), "psf angle step", 30.0), scene)
    psf_struct = rt_precompute_psf(stage)

    assert isinstance(psf_struct, dict)
    assert "psf" in psf_struct
    assert "sampAngles" in psf_struct
    assert "imgHeight" in psf_struct
    assert "wavelength" in psf_struct
    assert np.asarray(psf_struct["psf"], dtype=object).dtype == object
    assert np.array_equal(psf_struct["wavelength"], np.array([550.0]))


def test_rt_precompute_psf_apply_matches_oi_compute_uncropped(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    baseline = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=False)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)
    stage = oi_set(stage, "psf struct", rt_precompute_psf(stage))
    applied = rt_precompute_psf_apply(stage)

    assert applied.data["photons"].shape == baseline.data["photons"].shape
    assert applied.fields["padding_pixels"] == baseline.fields["padding_pixels"]
    assert np.allclose(applied.data["photons"], baseline.data["photons"])
    assert np.allclose(oi_get(applied, "depth map"), oi_get(baseline, "depth map"))


def test_rt_psf_apply_matches_rt_precompute_psf_apply(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    legacy = rt_psf_apply(stage)
    cached = rt_precompute_psf_apply(stage)

    assert legacy.data["photons"].shape == cached.data["photons"].shape
    assert legacy.fields["padding_pixels"] == cached.fields["padding_pixels"]
    assert np.allclose(legacy.data["photons"], cached.data["photons"])
    assert np.allclose(oi_get(legacy, "depth map"), oi_get(cached, "depth map"))


def test_rt_psf_apply_uses_explicit_angle_step(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    stage = rt_geometry(oi_create("ray trace", asset_store=asset_store), scene)

    result = rt_psf_apply(stage, angle_step_deg=30.0)

    assert np.isclose(oi_get(result, "psf angle step"), 30.0)
    assert np.array_equal(oi_get(result, "psf sample angles"), np.arange(0.0, 361.0, 30.0))


def test_rt_psf_edit_centers_and_rotates_raytrace_planes() -> None:
    centered_input = np.zeros((3, 3, 1, 1), dtype=float)
    centered_input[0, 0, 0, 0] = 1.0
    centered_optics = {
        "model": "raytrace",
        "raytrace": {
            "geometry": {},
            "psf": {"function": centered_input},
        },
    }
    centered = rt_psf_edit(centered_optics, cntr=True, visualize_flag=True)

    assert np.allclose(
        centered["raytrace"]["psf"]["function"][:, :, 0, 0],
        np.array(
            [
                [0.25, 0.0, 0.25],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.25],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(centered_optics["raytrace"]["psf"]["function"], centered_input)

    rotate_input = np.arange(1.0, 10.0, dtype=float).reshape(3, 3)
    rotate_optics = {
        "model": "raytrace",
        "raytrace": {
            "geometry": {},
            "psf": {"function": rotate_input[:, :, np.newaxis, np.newaxis]},
        },
    }
    rotated = rt_psf_edit(rotate_optics, rot=1)

    assert np.array_equal(
        rotated["raytrace"]["psf"]["function"][:, :, 0, 0],
        np.rot90(rotate_input, 1),
    )


def test_optics_object_wrappers_round_trip_supported_fields(asset_store) -> None:
    optics = opticsCreate(asset_store=asset_store)
    diffraction_alias = opticsCreate("diffraction", asset_store=asset_store)
    assert optics["type"] == "optics"
    assert optics["name"] == "standard (1/4-inch)"
    assert optics["vignetting"] == 0
    assert diffraction_alias["name"] == optics["name"]
    assert diffraction_alias["type"] == optics["type"]
    assert diffraction_alias["vignetting"] == optics["vignetting"]
    optics = opticsSet(optics, "fnumber", 2.8)
    optics = opticsSet(optics, "focal length", 0.01)
    optics = opticsSet(optics, "transmittance wave", np.array([500.0, 600.0], dtype=float))
    optics = opticsSet(optics, "transmittance scale", np.array([0.25, 0.75], dtype=float))

    assert np.isclose(float(opticsGet(optics, "fnumber")), 2.8)
    assert np.isclose(float(opticsGet(optics, "focal length")), 0.01)
    assert np.array_equal(np.asarray(opticsGet(optics, "transmittance wave"), dtype=float), np.array([500.0, 600.0], dtype=float))
    assert np.allclose(np.asarray(opticsGet(optics, "transmittance scale"), dtype=float), np.array([0.25, 0.75], dtype=float))
    assert np.isclose(float(opticsGet(optics, "aperture diameter")), 0.01 / 2.8)


def test_optics_create_blank_type_placeholder_uses_default_constructor(asset_store) -> None:
    default_optics = opticsCreate(asset_store=asset_store)
    blank_optics = opticsCreate("", asset_store=asset_store)

    assert blank_optics["name"] == default_optics["name"]
    assert blank_optics["type"] == default_optics["type"]
    assert np.isclose(float(blank_optics["f_number"]), float(default_optics["f_number"]))
    assert np.isclose(float(blank_optics["focal_length_m"]), float(default_optics["focal_length_m"]))


def test_optics_create_valid_returns_supported_wrapper_list() -> None:
    assert opticsCreate("valid") == [
        "default",
        "pinhole",
        "diffractionlimited",
        "diffraction",
        "shiftinvariant",
        "raytrace",
        "wvf",
        "psf",
        "human",
        "humanmw",
        "wvfhuman",
        "humanwvf",
        "uniformd65",
        "uniformee",
        "black",
        "empty",
    ]


def test_optics_create_empty_returns_minimal_shell() -> None:
    optics = opticsCreate("empty")

    assert optics["type"] == "optics"
    assert optics["name"] == "empty"
    assert "f_number" not in optics
    assert "focal_length_m" not in optics
    assert "transmittance" not in optics
    assert "offaxis_method" not in optics
    assert "vignetting" not in optics


def test_optics_clear_data_description_and_wvf_bridge(asset_store) -> None:
    optics = opticsCreate("psf", asset_store=asset_store)
    assert "otf_data" in optics

    cleared = opticsClearData(optics)
    description = opticsDescription(cleared)
    bridged_wvf = optics2wvf(cleared)

    assert "otf_data" not in cleared
    assert np.isclose(float(cleared["f_number"]), float(optics["f_number"]))
    assert np.isclose(float(cleared["focal_length_m"]), float(optics["focal_length_m"]))
    assert "Optics:" in description
    assert "Aper Diam" in description
    assert np.isclose(float(wvf_get(bridged_wvf, "fnumber")), float(cleared["f_number"]))
    assert np.isclose(float(wvf_get(bridged_wvf, "focal length")), float(cleared["focal_length_m"]))


def test_optics_create_human_alias_matches_marimont_wandell(asset_store) -> None:
    human_alias = opticsCreate("human", asset_store=asset_store)
    human_mw = opticsCreate("human mw", asset_store=asset_store)
    human_wvf = opticsCreate("wvf human", asset_store=asset_store)

    assert human_alias["type"] == "optics"
    assert human_mw["type"] == "optics"
    assert human_wvf["type"] == "optics"
    assert human_alias["name"] == "human-MW"
    assert human_alias["name"] == human_mw["name"]
    assert human_wvf["name"] == "humanwvf"
    assert human_wvf["name"] != human_alias["name"]
    assert np.isclose(float(human_alias["f_number"]), float(human_mw["f_number"]))
    assert np.isclose(float(human_alias["focal_length_m"]), float(human_mw["focal_length_m"]))
    assert "transmittance" not in human_alias
    assert "transmittance" not in human_mw
    assert "transmittance" not in human_wvf
    assert human_alias["lens"]["type"] == "lens"
    assert human_mw["lens"]["type"] == "lens"
    assert human_wvf["lens"]["type"] == "lens"
    assert np.allclose(
        np.asarray(opticsGet(human_mw, "transmittance"), dtype=float),
        np.asarray(human_mw["lens"]["transmittance"], dtype=float),
    )
    assert np.array_equal(
        np.asarray(opticsGet(human_wvf, "transmittance wave"), dtype=float),
        np.asarray(human_wvf["lens"]["wave"], dtype=float),
    )
    assert "wavefront" not in human_alias
    assert "wavefront" not in human_mw
    assert "wvf" not in human_alias
    assert "wvf" not in human_mw
    assert human_alias["otf_method"] == "human"
    assert human_mw["otf_method"] == "human"
    assert np.array_equal(
        np.asarray(human_wvf["wvf"]["zcoeffs"], dtype=float),
        np.asarray(human_wvf["wavefront"]["zcoeffs"], dtype=float),
    )
    assert human_wvf["otf_method"] == "human"


def test_oi_and_optics_set_lens_replace_camera_transmittance() -> None:
    wave = np.array([500.0, 600.0], dtype=float)
    transmittance = np.array([0.8, 0.6], dtype=float)
    lens = {"wave": wave, "transmittance": transmittance}
    updated_transmittance = np.array([0.7, 0.5], dtype=float)
    rewaved = np.array([450.0, 550.0, 650.0], dtype=float)

    oi = oi_set(oi_create("default"), "lens", lens)
    optics = opticsSet(opticsCreate("default"), "lens", lens)

    assert "transmittance" not in oi.fields["optics"]
    assert "transmittance" not in optics
    assert oi_get(oi, "lens")["type"] == "lens"
    assert opticsGet(optics, "lens")["type"] == "lens"
    assert np.array_equal(np.asarray(oi_get(oi, "transmittance wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(opticsGet(optics, "transmittance wave"), dtype=float), wave)
    assert np.allclose(np.asarray(oi_get(oi, "transmittance", wave), dtype=float), transmittance)
    assert np.allclose(np.asarray(opticsGet(optics, "transmittance", wave), dtype=float), transmittance)

    oi = oi_set(oi, "transmittance", updated_transmittance)
    optics = opticsSet(optics, "transmittance", updated_transmittance)
    assert "transmittance" not in oi.fields["optics"]
    assert "transmittance" not in optics
    assert np.allclose(np.asarray(oi_get(oi, "transmittance", wave), dtype=float), updated_transmittance)
    assert np.allclose(np.asarray(opticsGet(optics, "transmittance", wave), dtype=float), updated_transmittance)

    oi = oi_set(oi, "transmittance wave", rewaved)
    optics = opticsSet(optics, "transmittance wave", rewaved)
    assert "transmittance" not in oi.fields["optics"]
    assert "transmittance" not in optics
    assert np.array_equal(np.asarray(oi_get(oi, "transmittance wave"), dtype=float), rewaved)
    assert np.array_equal(np.asarray(opticsGet(optics, "transmittance wave"), dtype=float), rewaved)
    expected_rewaved = np.interp(rewaved, wave, updated_transmittance)
    assert np.allclose(np.asarray(oi_get(oi, "transmittance", rewaved), dtype=float), expected_rewaved)
    assert np.allclose(np.asarray(opticsGet(optics, "transmittance", rewaved), dtype=float), expected_rewaved)


def test_optics_create_shift_invariant_wrappers_restore_wvf_name(asset_store) -> None:
    shift_invariant = opticsCreate("shift invariant", asset_store=asset_store)
    wavefront = opticsCreate("wvf", asset_store=asset_store)

    assert shift_invariant["type"] == "optics"
    assert wavefront["type"] == "optics"
    assert shift_invariant["name"] == "wvf"
    assert wavefront["name"] == "wvf"
    assert shift_invariant["compute_method"] == "opticspsf"
    assert wavefront["compute_method"] == "opticspsf"
    assert np.array_equal(
        np.asarray(shift_invariant["wvf"]["wave"], dtype=float),
        np.asarray(shift_invariant["wavefront"]["wave"], dtype=float),
    )
    assert np.array_equal(
        np.asarray(wavefront["wvf"]["wave"], dtype=float),
        np.asarray(wavefront["wavefront"]["wave"], dtype=float),
    )


def test_lens_list_reads_pinned_upstream_lens_jsons(asset_store) -> None:
    files = lensList("dgauss*.json", quiet=True, asset_store=asset_store)

    assert files
    assert all(item["name"].endswith(".json") for item in files)
    assert any(item["name"] == "dgauss.22deg.6.0mm.json" for item in files)


def test_optics_ray_trace_matches_oi_compute_uncropped(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    baseline = oi_compute(oi_create("ray trace", asset_store=asset_store), scene, crop=False)
    result = optics_ray_trace(scene, oi_create("ray trace", asset_store=asset_store))

    assert result.data["photons"].shape == baseline.data["photons"].shape
    assert result.fields["padding_pixels"] == baseline.fields["padding_pixels"]
    assert np.allclose(result.data["photons"], baseline.data["photons"])
    assert np.allclose(oi_get(result, "depth map"), oi_get(baseline, "depth map"))
    assert result.fields["illuminance"].shape == result.data["photons"].shape[:2]
    assert np.allclose(result.fields["illuminance"], oi_get(baseline, "illuminance"))
    assert np.isclose(result.fields["mean_illuminance"], oi_get(result, "mean illuminance"))


def test_optics_ray_trace_uses_explicit_angle_step(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    result = optics_ray_trace(scene, oi_create("ray trace", asset_store=asset_store), angle_step_deg=30.0)

    assert np.isclose(oi_get(result, "psf angle step"), 30.0)
    assert np.array_equal(oi_get(result, "psf sample angles"), np.arange(0.0, 361.0, 30.0))


def test_oi_calculate_illuminance_updates_cached_fields(asset_store) -> None:
    scene = scene_create("uniform ee", 32, np.array([550.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    illuminance, mean_illuminance, mean_comp_illuminance = oi_calculate_illuminance(oi)

    assert np.allclose(illuminance, oi_get(oi, "illuminance"))
    assert np.isclose(mean_illuminance, oi_get(oi, "mean illuminance"))
    assert np.isclose(mean_comp_illuminance, oi_get(oi, "mean comp illuminance"))
    assert np.isclose(mean_comp_illuminance, 0.0)


def test_oi_diffuser_blurs_photons_and_returns_kernel(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    baseline = np.asarray(oi.data["photons"], dtype=float).copy()

    oi, sd, blur_filter = oi_diffuser(oi, 2.0)

    assert np.isclose(float(sd), 2.0)
    assert blur_filter.ndim == 2
    assert np.isclose(np.sum(blur_filter), 1.0)
    assert oi.data["photons"].shape == baseline.shape
    assert not np.allclose(oi.data["photons"], baseline)
    assert np.allclose(oi.fields["illuminance"], oi_get(oi, "illuminance"))


def test_oi_birefringent_diffuser_matches_four_corner_average() -> None:
    photons = np.arange(1.0, 17.0, dtype=float).reshape(4, 4, 1)
    oi = oi_create("empty", np.array([550.0], dtype=float))
    oi = oi_set(oi, "photons", photons)
    displacement_um = float(oi_get(oi, "wspatialresolution")) * 1e6

    filtered, returned_disp_um = oiBirefringentDiffuser(oi, displacement_um)

    padded = np.pad(photons[:, :, 0], 1, mode="constant", constant_values=0.0)
    expected = (
        padded[:-2, :-2]
        + padded[:-2, 2:]
        + padded[2:, :-2]
        + padded[2:, 2:]
    ) / 4.0

    assert np.isclose(returned_disp_um, displacement_um)
    assert np.allclose(np.asarray(oi_get(filtered, "photons"), dtype=float)[:, :, 0], expected)
    assert np.allclose(np.asarray(oi_get(filtered, "illuminance"), dtype=float), np.asarray(filtered.fields["illuminance"], dtype=float))


def test_oi_camera_motion_builds_depth_shifted_frame_stack() -> None:
    photons = np.zeros((3, 3, 1), dtype=float)
    photons[1, 1, 0] = 10.0
    focal_length_m = 0.004
    sample_spacing_m = 1e-4
    width_m = photons.shape[1] * sample_spacing_m
    fov_deg = float(np.rad2deg(2.0 * np.arctan2(width_m / 2.0, focal_length_m)))

    oi = oi_create("empty", np.array([550.0], dtype=float))
    oi = oi_set(oi, "optics focal length", focal_length_m)
    oi = oi_set(oi, "photons", photons)
    oi = oi_set(oi, "wangular", fov_deg)
    oi = oi_set(oi, "depth map", np.full((3, 3), focal_length_m, dtype=float))
    oi_calculate_illuminance(oi)

    shifted = oiCameraMotion(
        oi,
        {
            "amount": ((0.0, sample_spacing_m), (sample_spacing_m, 0.0)),
            "focalLength": focal_length_m,
        },
    )

    shifted_photons = np.asarray(oi_get(shifted, "photons"), dtype=float)
    shifted_illuminance = np.asarray(shifted.fields["illuminance"], dtype=float)

    expected_right = np.zeros((3, 3, 1), dtype=float)
    expected_right[1, 2, 0] = 10.0
    expected_down = np.zeros((3, 3, 1), dtype=float)
    expected_down[2, 1, 0] = 10.0

    assert shifted_photons.shape == (3, 3, 1, 3)
    assert np.allclose(shifted_photons[:, :, :, 0], photons)
    assert np.allclose(shifted_photons[:, :, :, 1], expected_right)
    assert np.allclose(shifted_photons[:, :, :, 2], expected_down)
    assert shifted_illuminance.shape == (3, 3, 3)
    assert np.count_nonzero(shifted_illuminance[:, :, 1]) == 1
    assert np.count_nonzero(shifted_illuminance[:, :, 2]) == 1


def test_oi_clear_data_clears_computed_payloads_and_wvf_cache(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create("wvf", wvf_create(wave=scene_get(scene, "wave")), asset_store=asset_store), scene, crop=True)
    oi = oi_set(oi, "mean illuminance", 12.0)

    cleared = oiClearData(oi)

    assert cleared is not oi
    assert cleared.data == {}
    assert cleared.fields["depth_map_m"] is None
    assert cleared.fields.get("mean_illuminance") is None
    assert np.array_equal(np.asarray(oi_get(cleared, "wave"), dtype=float), np.asarray(oi_get(oi, "wave"), dtype=float))
    assert isinstance(cleared.fields["optics"].get("wavefront"), dict)
    assert cleared.fields["optics"]["wavefront"]["psf"] is None
    assert cleared.fields["optics"]["wavefront"]["pupil_function"] is None


def test_oi_show_image_matches_manual_render_and_gray_modes(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene, crop=True)

    photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    wave = np.asarray(oi_get(oi, "wave"), dtype=float)
    expected = xyz_to_srgb(xyz_from_energy(quanta_to_energy(photons, wave), wave, asset_store=asset_store))

    rendered = oiShowImage(oi, 0, 1.0, asset_store=asset_store)
    gamma_rendered = oiShowImage(oi, 1, 0.8, asset_store=asset_store)
    gray = oiShowImage(oi, 2, 1.0, asset_store=asset_store)

    assert rendered is not None
    assert np.allclose(np.asarray(rendered, dtype=float), np.asarray(expected, dtype=float))
    assert np.allclose(np.asarray(gamma_rendered, dtype=float), np.power(np.clip(np.asarray(expected, dtype=float), 0.0, None), 0.8))
    assert gray is not None
    assert np.asarray(gray, dtype=float).shape == np.asarray(rendered, dtype=float).shape
    assert np.allclose(np.asarray(gray, dtype=float)[:, :, 0], np.asarray(gray, dtype=float)[:, :, 1])
    assert np.allclose(np.asarray(gray, dtype=float)[:, :, 1], np.asarray(gray, dtype=float)[:, :, 2])


def test_oi_save_image_appends_png_and_matches_headless_render(asset_store, tmp_path) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene, crop=True)

    saved_path = oiSaveImage(oi, tmp_path / "oi_capture", asset_store=asset_store)
    written = np.asarray(iio.imread(saved_path), dtype=float) / 255.0
    expected = np.asarray(oiShowImage(oi, -1, 1.0, asset_store=asset_store), dtype=float)

    assert saved_path.endswith(".png")
    assert np.allclose(written, np.clip(np.round(np.clip(expected, 0.0, 1.0) * 255.0), 0.0, 255.0) / 255.0, atol=1.0 / 255.0)


def test_oi_from_file_and_extract_bright_match_legacy_contract(asset_store) -> None:
    rgb = np.linspace(0.0, 1.0, 4 * 6 * 3, dtype=float).reshape(4, 6, 3)
    display = display_create("LCD-Apple.mat", asset_store=asset_store)
    wave = np.arange(450.0, 651.0, 50.0, dtype=float)

    oi = oiFromFile(rgb, "rgb", 50.0, display, wave, asset_store=asset_store)

    scene = scene_from_file(rgb, "rgb", 50.0, display, wave, asset_store=asset_store)
    scene = scene_adjust_illuminant(scene, "D65", asset_store=asset_store)

    assert oi_get(oi, "optics model") == "raytrace"
    assert np.array_equal(np.asarray(oi_get(oi, "wave"), dtype=float), np.asarray(scene_get(scene, "wave"), dtype=float))
    assert np.isclose(float(oi_get(oi, "fov")), float(scene_get(scene, "fov")))
    assert np.allclose(
        np.asarray(oi_get(oi, "photons"), dtype=float),
        np.asarray(scene_get(scene, "photons"), dtype=float) / np.pi,
    )

    oi_calculate_illuminance(oi)
    illuminance = np.asarray(oi_get(oi, "illuminance"), dtype=float)
    row_index, col_index = np.unravel_index(int(np.argmax(illuminance)), illuminance.shape)

    bright = oiExtractBright(oi)
    expected = oi_crop(oi, np.array([col_index + 1, row_index + 1, 1, 1], dtype=int))
    expected = oi_set(expected, "compute method", "skip")
    expected = oi_set(expected, "off axis method", "skip")

    assert np.allclose(np.asarray(oi_get(bright, "photons"), dtype=float), np.asarray(oi_get(expected, "photons"), dtype=float))
    assert oi_get(bright, "compute method") == "skip"
    assert oi_get(bright, "off axis method") == "skip"


def test_oi_spectral_helper_wrappers_match_legacy_contract(asset_store) -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 16, wave, asset_store=asset_store)
    oi_base = oi_create("pinhole", asset_store=asset_store)
    oi = oi_compute(oi_base, scene, crop=True)

    expected_irradiance = np.asarray(oi_get(oi, "photons"), dtype=float)
    irradiance = np.asarray(oiCalculateIrradiance(scene, oi_base), dtype=float)
    assert np.allclose(irradiance, expected_irradiance)

    photons = expected_irradiance.copy()
    photons[:, :, 1] *= 2.0
    photons[:, :, 2] *= 4.0
    spectral_oi = oi_set(oi.clone(), "photons", photons)
    oi_calculate_illuminance(spectral_oi)

    adjusted_mean = oiAdjustIlluminance(spectral_oi, 25.0)
    adjusted_peak = oiAdjustIlluminance(spectral_oi, 40.0, "peak")
    assert np.isclose(float(oi_get(adjusted_mean, "mean illuminance")), 25.0, atol=1e-8, rtol=1e-8)
    assert np.isclose(float(np.max(np.asarray(oi_get(adjusted_peak, "illuminance"), dtype=float))), 40.0, atol=1e-8, rtol=1e-8)

    new_wave = np.array([500.0, 550.0, 600.0, 650.0, 700.0], dtype=float)
    interpolated = oiInterpolateW(spectral_oi, new_wave)
    expected_profile = np.interp(new_wave, wave, photons[0, 0, :])
    manual_interpolated = oi_set(spectral_oi.clone(), "wave", new_wave)
    manual_interpolated = oi_set(
        manual_interpolated,
        "photons",
        np.broadcast_to(expected_profile.reshape(1, 1, -1), (photons.shape[0], photons.shape[1], new_wave.size)).copy(),
    )
    oi_calculate_illuminance(manual_interpolated)
    expected_profile = expected_profile * (
        float(oi_get(spectral_oi, "mean illuminance")) / float(oi_get(manual_interpolated, "mean illuminance"))
    )
    assert np.array_equal(np.asarray(oi_get(interpolated, "wave"), dtype=float), new_wave)
    assert np.allclose(np.asarray(oi_get(interpolated, "photons"), dtype=float)[0, 0, :], expected_profile)
    assert np.isclose(
        float(oi_get(interpolated, "mean illuminance")),
        float(oi_get(spectral_oi, "mean illuminance")),
        atol=1e-8,
        rtol=1e-8,
    )

    extracted = oiExtractWaveband(spectral_oi, np.array([550.0, 650.0], dtype=float))
    extracted_with_illum = oiExtractWaveband(spectral_oi, np.array([550.0, 650.0], dtype=float), 1)
    assert np.array_equal(np.asarray(oi_get(extracted, "wave"), dtype=float), np.array([550.0, 650.0], dtype=float))
    assert np.allclose(
        np.asarray(oi_get(extracted, "photons"), dtype=float)[0, 0, :],
        np.interp(np.array([550.0, 650.0], dtype=float), wave, photons[0, 0, :]),
    )
    assert "illuminance" not in extracted.fields
    assert np.asarray(oi_get(extracted_with_illum, "illuminance"), dtype=float).shape == photons.shape[:2]

    added = oiAdd(spectral_oi, spectral_oi)
    assert np.allclose(np.asarray(oi_get(added, "photons"), dtype=float), 2.0 * photons)

    rows = photons.shape[0]
    gradient = np.linspace(-1.0, 1.0, rows, dtype=float).reshape(rows, 1, 1)
    contrast_oi = oi_set(spectral_oi.clone(), "photons", photons * (1.0 + 0.1 * gradient))
    removed = oiAdd(spectral_oi, contrast_oi, "remove spatial mean")
    contrast_photons = np.asarray(oi_get(contrast_oi, "photons"), dtype=float)
    expected_removed = photons + (contrast_photons - np.mean(contrast_photons, axis=(0, 1), keepdims=True))
    assert np.allclose(np.asarray(oi_get(removed, "photons"), dtype=float), expected_removed)


def test_oi_illuminant_helper_wrappers_match_legacy_contract(asset_store) -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 16, wave, asset_store=asset_store)
    oi = oi_compute(oi_create("pinhole", asset_store=asset_store), scene, crop=True)
    illuminant = illuminant_create("equal photons", wave, 100.0, asset_store=asset_store)
    oi = oi_set(oi, "illuminant", illuminant)

    assert oi_get(oi, "illuminant format") == "spectral"
    assert np.array_equal(np.asarray(oi_get(oi, "illuminant photons"), dtype=float), np.asarray(illuminant.data["photons"], dtype=float))

    spatial = oiIlluminantSS(oi)
    rows, cols = oi_get(spatial, "size")
    expected_spectral = np.asarray(illuminant.data["photons"], dtype=float).reshape(1, 1, -1)
    assert oi_get(spatial, "illuminant format") == "spatial spectral"
    assert np.allclose(
        np.asarray(oi_get(spatial, "illuminant photons"), dtype=float),
        np.broadcast_to(expected_spectral, (rows, cols, expected_spectral.shape[2])),
    )

    pattern = np.array([[1.0, 0.5], [0.25, 2.0]], dtype=float)
    patterned = oiIlluminantPattern(spatial, pattern)
    row_positions = np.linspace(0.0, pattern.shape[0] - 1, rows, dtype=float)
    col_positions = np.linspace(0.0, pattern.shape[1] - 1, cols, dtype=float)
    row_grid, col_grid = np.meshgrid(row_positions, col_positions, indexing="ij")
    resized_pattern = ndimage.map_coordinates(pattern, [row_grid, col_grid], order=1, mode="nearest", prefilter=False)
    assert np.allclose(
        np.asarray(oi_get(patterned, "illuminant photons"), dtype=float),
        np.asarray(oi_get(spatial, "illuminant photons"), dtype=float) * resized_pattern[:, :, None],
    )
    assert np.allclose(
        np.asarray(oi_get(patterned, "photons"), dtype=float),
        np.asarray(oi_get(spatial, "photons"), dtype=float) * resized_pattern[:, :, None],
    )


def test_oi_create_valid_returns_upstream_constructor_list() -> None:
    assert oi_create("valid") == [
        "default",
        "pinhole",
        "diffractionlimited",
        "diffraction",
        "shiftinvariant",
        "raytrace",
        "wvf",
        "human",
        "humanmw",
        "wvfhuman",
        "humanwvf",
        "uniformd65",
        "uniformee",
        "black",
    ]


def test_oi_create_supports_uniform_and_black_constructor_wrappers(asset_store) -> None:
    wave = np.arange(500.0, 601.0, 50.0, dtype=float)

    default_oi = oi_create("default", asset_store=asset_store)
    diffraction_oi = oi_create("diffraction", asset_store=asset_store)
    wvf_oi = oi_create("wvf", asset_store=asset_store)
    shift_oi = oi_create("shift invariant", asset_store=asset_store)
    uniform_d65 = oi_create("uniform d65", 16, wave, asset_store=asset_store)
    expected_uniform_d65_scene = scene_set(scene_create("uniform d65", 16, wave, asset_store=asset_store), "hfov", 120.0)
    expected_uniform_d65 = oi_create("default", asset_store=asset_store)
    expected_uniform_d65 = oi_set(expected_uniform_d65, "optics fnumber", 1e-3)
    expected_uniform_d65 = oi_set(expected_uniform_d65, "optics offaxis method", "skip")
    expected_uniform_d65 = oi_set(oi_compute(expected_uniform_d65, expected_uniform_d65_scene), "compute method", "")

    assert default_oi.fields["optics"]["name"] == "standard (1/4-inch)"
    assert diffraction_oi.fields["optics"]["name"] == "standard (1/4-inch)"
    assert wvf_oi.fields["optics"]["name"] == "wvf"
    assert shift_oi.fields["optics"]["name"] == "wvf"
    assert uniform_d65.fields["optics"]["name"] == "standard (1/4-inch)"

    np.testing.assert_allclose(
        np.asarray(oi_get(uniform_d65, "photons"), dtype=float),
        np.asarray(oi_get(expected_uniform_d65, "photons"), dtype=float),
    )
    assert oi_get(uniform_d65, "compute method") == ""
    assert np.isclose(float(oi_get(uniform_d65, "fnumber")), 1e-3)
    assert oi_get(uniform_d65, "optics off axis method") == "skip"
    assert np.isclose(float(oi_get(uniform_d65, "fov")), float(oi_get(expected_uniform_d65, "fov")))

    uniform_ee = oi_create("uniform ee", [], wave, asset_store=asset_store)
    expected_uniform_ee_scene = scene_set(scene_create("uniform ee", 32, wave, asset_store=asset_store), "hfov", 120.0)
    expected_uniform_ee = oi_create("default", asset_store=asset_store)
    expected_uniform_ee = oi_set(expected_uniform_ee, "optics fnumber", 1e-3)
    expected_uniform_ee = oi_set(expected_uniform_ee, "optics offaxis method", "skip")
    expected_uniform_ee = oi_set(oi_compute(expected_uniform_ee, expected_uniform_ee_scene), "compute method", "")

    np.testing.assert_allclose(
        np.asarray(oi_get(uniform_ee, "photons"), dtype=float),
        np.asarray(oi_get(expected_uniform_ee, "photons"), dtype=float),
    )
    assert tuple(np.asarray(oi_get(uniform_ee, "size"), dtype=int)) == tuple(np.asarray(oi_get(expected_uniform_ee, "size"), dtype=int))
    assert oi_get(uniform_ee, "compute method") == ""

    black = oi_create("black", [], wave, asset_store=asset_store)
    expected_black = oi_create("shift invariant", asset_store=asset_store)
    expected_black = oi_set(expected_black, "wave", wave)
    expected_black = oi_set(expected_black, "photons", np.zeros((32, 32, wave.size), dtype=float))
    expected_black = oi_set(expected_black, "fov", 100.0)

    np.testing.assert_allclose(
        np.asarray(oi_get(black, "photons"), dtype=float),
        np.asarray(oi_get(expected_black, "photons"), dtype=float),
    )
    assert np.array_equal(np.asarray(oi_get(black, "wave"), dtype=float).reshape(-1), wave)
    assert np.isclose(float(oi_get(black, "fov")), 100.0)
    assert black.fields["optics"]["name"] == "wvf"


def test_oi_create_supports_human_constructor_wrappers(asset_store) -> None:
    wave = np.arange(500.0, 601.0, 50.0, dtype=float)
    scene = scene_create("uniform ee", 8, wave, asset_store=asset_store)

    human_mw = oi_create("human mw", asset_store=asset_store)
    assert oi_get(human_mw, "name") == "human-MW"
    assert human_mw.fields["name"] == "human-MW"
    assert human_mw.fields["type"] == "opticalimage"
    assert oi_get(human_mw, "compute method") == "humanmw"
    assert oi_get(human_mw, "optics model") == "shiftinvariant"
    assert human_mw.fields["optics"]["name"] == "human-MW"
    assert human_mw.fields["optics"]["lens"]["type"] == "lens"
    assert np.array_equal(
        np.asarray(human_mw.fields["optics"]["lens"]["wave"], dtype=float),
        np.asarray(oi_get(human_mw, "wave"), dtype=float),
    )
    assert np.allclose(
        np.asarray(oi_get(human_mw, "transmittance"), dtype=float),
        np.asarray(human_mw.fields["optics"]["lens"]["transmittance"], dtype=float),
    )
    assert oi_get(human_mw, "lens")["type"] == "lens"
    assert "transmittance" not in human_mw.fields["optics"]

    computed_human_mw = oi_compute(human_mw, scene, crop=True)
    assert np.asarray(oi_get(computed_human_mw, "photons"), dtype=float).shape == (8, 8, wave.size)

    human_wvf = oi_create("wvf human", 4.0, [], wave, asset_store=asset_store)
    human_alias = oi_create("human", 4.0, [], wave, asset_store=asset_store)
    assert human_wvf.name == "human-WVF"
    assert human_wvf.fields["name"] == "human-WVF"
    assert human_wvf.fields["type"] == "opticalimage"
    assert oi_get(human_wvf, "compute method") == "opticspsf"
    assert oi_get(human_wvf, "optics model") == "shiftinvariant"
    assert np.array_equal(np.asarray(oi_get(human_wvf, "wave"), dtype=float).reshape(-1), wave)
    assert human_wvf.fields["optics"]["name"] == "humanwvf"
    assert human_alias.fields["optics"]["name"] == "humanwvf"
    assert human_wvf.fields["optics"]["lens"]["type"] == "lens"
    assert human_alias.fields["optics"]["lens"]["type"] == "lens"
    assert np.array_equal(
        np.asarray(human_wvf.fields["optics"]["lens"]["wave"], dtype=float),
        np.asarray(oi_get(human_wvf, "wave"), dtype=float),
    )
    assert np.allclose(
        np.asarray(oi_get(human_wvf, "transmittance"), dtype=float),
        np.asarray(human_wvf.fields["optics"]["lens"]["transmittance"], dtype=float),
    )
    assert np.array_equal(
        np.asarray(oi_get(human_wvf, "transmittance wave"), dtype=float),
        np.asarray(oi_get(human_wvf, "wave"), dtype=float),
    )
    assert oi_get(human_wvf, "lens")["type"] == "lens"
    assert "transmittance" not in human_wvf.fields["optics"]
    assert "transmittance" not in human_alias.fields["optics"]
    assert np.array_equal(
        np.asarray(oi_get(human_alias, "wave"), dtype=float).reshape(-1),
        np.asarray(oi_get(human_wvf, "wave"), dtype=float).reshape(-1),
    )
    assert oi_get(human_alias, "compute method") == oi_get(human_wvf, "compute method")

    computed_human_wvf = oi_compute(human_wvf, scene, crop=True)
    assert np.asarray(oi_get(computed_human_wvf, "photons"), dtype=float).shape == (8, 8, wave.size)


def test_oi_create_empty_returns_minimal_shell_and_keeps_wave_override() -> None:
    explicit_wave = np.array([450.0, 550.0], dtype=float)

    empty = oi_create("empty")
    explicit = oi_create("empty", explicit_wave)
    renamed = oi_set(empty.clone(), "name", "empty shell")

    assert empty.fields["name"] == "opticalimage"
    assert empty.fields["type"] == "opticalimage"
    assert empty.fields["optics"]["name"] == "empty"
    assert empty.fields["optics"]["type"] == "optics"
    assert "f_number" not in empty.fields["optics"]
    assert "focal_length_m" not in empty.fields["optics"]
    assert "transmittance" not in empty.fields["optics"]
    assert oi_get(empty, "diffuser method") == "skip"
    assert oi_get(empty, "compute method") == ""
    assert np.asarray(oi_get(empty, "wave"), dtype=float).size == 0

    assert explicit.fields["name"] == "opticalimage"
    assert explicit.fields["type"] == "opticalimage"
    assert "transmittance" not in explicit.fields["optics"]
    assert np.array_equal(np.asarray(oi_get(explicit, "wave"), dtype=float), explicit_wave)
    assert renamed.fields["name"] == "empty shell"
    assert renamed.fields["type"] == "opticalimage"


def test_oi_create_treats_empty_type_as_default_and_supports_uniform_ee_specify(asset_store) -> None:
    wave = np.arange(500.0, 601.0, 50.0, dtype=float)

    default_oi = oi_create(asset_store=asset_store)
    placeholder_oi = oi_create([], asset_store=asset_store)
    assert oi_get(placeholder_oi, "compute method") == oi_get(default_oi, "compute method")
    assert oi_get(placeholder_oi, "optics model") == oi_get(default_oi, "optics model")
    assert np.isclose(float(oi_get(placeholder_oi, "fnumber")), float(oi_get(default_oi, "fnumber")))

    alias_oi = oi_create("uniform ee specify", [], wave, asset_store=asset_store)
    reference_oi = oi_create("uniform ee", 32, wave, asset_store=asset_store)
    np.testing.assert_allclose(
        np.asarray(oi_get(alias_oi, "photons"), dtype=float),
        np.asarray(oi_get(reference_oi, "photons"), dtype=float),
    )
    assert oi_get(alias_oi, "compute method") == oi_get(reference_oi, "compute method")


def test_oi_create_treats_blank_type_as_default(asset_store) -> None:
    default_oi = oi_create(asset_store=asset_store)
    blank_oi = oi_create("", asset_store=asset_store)

    assert oi_get(blank_oi, "compute method") == oi_get(default_oi, "compute method")
    assert oi_get(blank_oi, "optics model") == oi_get(default_oi, "optics model")
    assert np.isclose(float(oi_get(blank_oi, "fnumber")), float(oi_get(default_oi, "fnumber")))


def test_oi_create_treats_empty_optional_wave_as_omitted_for_default_and_pinhole(asset_store) -> None:
    default_oi = oi_create("default", [], asset_store=asset_store)
    default_reference = oi_create("default", asset_store=asset_store)
    pinhole_oi = oi_create("pinhole", [], asset_store=asset_store)
    pinhole_reference = oi_create("pinhole", asset_store=asset_store)

    assert np.array_equal(np.asarray(oi_get(default_oi, "wave"), dtype=float), np.asarray(oi_get(default_reference, "wave"), dtype=float))
    assert np.array_equal(np.asarray(oi_get(pinhole_oi, "wave"), dtype=float), np.asarray(oi_get(pinhole_reference, "wave"), dtype=float))
    assert np.isclose(float(oi_get(pinhole_oi, "fnumber")), float(oi_get(pinhole_reference, "fnumber")))
    assert oi_get(pinhole_oi, "name") == "pinhole"
    assert pinhole_oi.fields["optics"]["name"] == "pinhole"
    assert oi_get(pinhole_reference, "name") == "pinhole"
    assert pinhole_reference.fields["optics"]["name"] == "pinhole"


def test_oi_photon_noise_matches_seeded_legacy_contract() -> None:
    photons = np.array(
        [
            [[4.0, 20.0], [9.0, 25.0]],
            [[16.0, 36.0], [1.0, 49.0]],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(7)
    expected_noise = np.sqrt(photons) * rng.standard_normal(photons.shape)
    expected_noisy = np.rint(photons + expected_noise)
    poisson_mask = photons < 15.0
    poisson_samples = rng.poisson(photons[poisson_mask])
    expected_noise[poisson_mask] = poisson_samples
    expected_noisy[poisson_mask] = poisson_samples

    noisy, noise = oiPhotonNoise(photons, seed=7)

    assert np.array_equal(noisy, expected_noisy)
    assert np.array_equal(noise, expected_noise)


def test_legacy_optics_compute_wrappers_match_direct_oi_compute(asset_store) -> None:
    scene = scene_create("uniform ee", 16, 550.0, asset_store=asset_store)

    dl_base = oi_create("diffraction limited", asset_store=asset_store)
    dl_direct = oi_compute(dl_base, scene, crop=True)
    dl_wrapped = opticsDLCompute(scene, dl_base, crop=True)
    assert np.allclose(np.asarray(oi_get(dl_wrapped, "photons"), dtype=float), np.asarray(oi_get(dl_direct, "photons"), dtype=float))
    assert np.allclose(np.asarray(oi_get(dl_wrapped, "illuminance"), dtype=float), np.asarray(oi_get(dl_direct, "illuminance"), dtype=float))

    si_base = oi_create("shift invariant", asset_store=asset_store)
    si_direct = oi_compute(si_base, scene, crop=True)
    si_wrapped = opticsSICompute(scene, si_base, crop=True)
    assert np.allclose(np.asarray(oi_get(si_wrapped, "photons"), dtype=float), np.asarray(oi_get(si_direct, "photons"), dtype=float))
    assert np.allclose(np.asarray(oi_get(si_wrapped, "illuminance"), dtype=float), np.asarray(oi_get(si_direct, "illuminance"), dtype=float))


def test_legacy_optics_defocus_and_transmittance_wrappers_match_contract(asset_store) -> None:
    scene = scene_create("uniform ee", 16, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    oi = oi_create("diffraction limited", asset_store=asset_store)
    oi = oi_set(oi, "transmittance wave", np.array([500.0, 600.0, 700.0], dtype=float))
    oi = oi_set(oi, "transmittance scale", np.array([0.25, 0.5, 0.75], dtype=float))

    udata = opticsPlotTransmittance(oi)
    assert np.array_equal(np.asarray(udata["wave"], dtype=float), np.asarray(oi_get(oi, "wave"), dtype=float))
    assert np.allclose(
        np.asarray(udata["transmittance"], dtype=float),
        np.asarray(oi_get(oi, "transmittance", oi_get(oi, "wave")), dtype=float),
    )

    optics = oi_get(oi, "optics")
    image_plane_distance = float(opticsGet(optics, "focal length", "m")) * 1.1
    object_distance = 1.75
    defocus_diopters, _ = optics_depth_defocus(object_distance, optics, image_plane_distance)
    recovered = opticsDefocusDepth(defocus_diopters, optics, image_plane_distance)
    assert np.isclose(float(recovered), object_distance, atol=1e-8, rtol=1e-8)

    wrapped = opticsDLCompute(scene, oi, crop=True)
    direct = oi_compute(oi, scene, crop=True)
    assert np.allclose(np.asarray(oi_get(wrapped, "photons"), dtype=float), np.asarray(oi_get(direct, "photons"), dtype=float))


def test_optics_ray_trace_blur_matches_public_oi_diffuser(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    blur_m = 2e-6

    manual = rt_precompute_psf_apply(rt_geometry(oi_create("ray trace", asset_store=asset_store), scene))
    manual, _, _ = oi_diffuser(manual, blur_m * 1e6)

    raytrace_oi = oi_set(oi_create("ray trace", asset_store=asset_store), "diffuser method", "blur")
    raytrace_oi = oi_set(raytrace_oi, "diffuser blur", blur_m)
    wrapped = optics_ray_trace(scene, raytrace_oi)

    assert wrapped.data["photons"].shape == manual.data["photons"].shape
    assert np.allclose(wrapped.data["photons"], manual.data["photons"])
    assert np.allclose(oi_get(wrapped, "illuminance"), oi_get(manual, "illuminance"))


def test_wvf_path_preserves_more_checkerboard_contrast_than_diffraction(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi_wvf = oi_compute(oi_create("wvf"), scene, crop=True)
    oi_diffraction = oi_compute(oi_create(), scene, crop=True)

    row = oi_wvf.data["photons"].shape[0] // 2
    dark_col = 8
    band = 0
    dark_wvf = float(oi_wvf.data["photons"][row, dark_col, band])
    dark_diffraction = float(oi_diffraction.data["photons"][row, dark_col, band])

    assert dark_wvf < dark_diffraction


def test_oi_create_wvf_matches_upstream_default_wavefront_metadata() -> None:
    oi = oi_create("wvf")
    wavefront = oi.fields["optics"]["wavefront"]
    assert oi.fields["optics"]["compute_method"] == "opticspsf"
    assert oi.fields["optics"]["model"] == "shiftinvariant"
    assert np.isclose(wavefront["measured_pupil_diameter_mm"], 8.0)
    assert np.isclose(wavefront["measured_wavelength_nm"], 550.0)
    assert wavefront["sample_interval_domain"] == "psf"
    assert wavefront["spatial_samples"] == 201
    assert np.isclose(wavefront["ref_pupil_plane_size_mm"], 16.212)
    assert np.isclose(wavefront["calc_pupil_diameter_mm"], 9.6569e-01)
    assert np.isclose(wavefront["focal_length_m"], 0.003862755099228)
    assert np.isclose(wavefront["f_number"], 4.0)
    assert wavefront["lca_method"] == "none"
    assert wavefront["compute_sce"] is False
    assert np.array_equal(wavefront["zcoeffs"], np.array([0.0]))
    assert np.array_equal(wavefront["sce_params"]["wave"], wavefront["wave"])
    assert np.allclose(wavefront["sce_params"]["rho"], 0.0)
    assert np.isclose(wavefront["sce_params"]["xo_mm"], 0.0)
    assert np.isclose(wavefront["sce_params"]["yo_mm"], 0.0)


def test_wvf_set_and_get_named_zcoeffs_round_trip() -> None:
    wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
    wvf = wvf_set(wvf, "zcoeffs", np.array([2.0, 0.5], dtype=float), ["defocus", "vertical_astigmatism"])

    assert np.isclose(wvf_get(wvf, "zcoeffs", "defocus"), 2.0)
    assert np.isclose(wvf_get(wvf, "zcoeffs", "vertical_astigmatism"), 0.5)
    assert np.array_equal(wvf_get(wvf, "wave"), np.array([450.0, 550.0, 650.0], dtype=float))


def test_wvf_defocus_diopter_micron_round_trip() -> None:
    microns = wvf_defocus_diopters_to_microns(1.5, 4.0)
    diopters = wvf_defocus_microns_to_diopters(microns, 4.0)

    assert np.isclose(float(np.asarray(microns).reshape(-1)[0]), 1.5 * (4.0**2) / (16.0 * np.sqrt(3.0)))
    assert np.isclose(float(np.asarray(diopters).reshape(-1)[0]), 1.5)


def test_wvf_compute_returns_psf_and_pupil_function() -> None:
    wvf = wvf_create(wave=np.array([500.0, 600.0], dtype=float))
    wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
    computed = wvf_compute(wvf)

    assert computed["computed"] is True
    assert computed["psf"].shape == (201, 201, 2)
    assert computed["pupil_function"].shape == (201, 201, 2)
    assert computed["pupil_amplitude"].shape == (201, 201, 2)
    assert computed["pupil_phase"].shape == (201, 201, 2)
    assert np.isclose(float(np.sum(computed["psf"][:, :, 0])), 1.0)
    assert np.isclose(float(wvf_get(computed, "pupil diameter", "mm")), 3.0)
    assert np.asarray(wvf_get(computed, "psf")).shape == (201, 201, 2)
    assert np.asarray(wvf_get(computed, "pupil function")).shape == (201, 201, 2)


def test_wvf_clear_data_preserves_metadata_and_recomputes_on_demand() -> None:
    wvf = wvf_compute(wvf_set(wvf_create(wave=np.array([500.0, 600.0], dtype=float)), "pupil diameter", 3.0, "mm"))

    cleared = wvf_clear_data(wvf)

    assert cleared is not wvf
    assert cleared["computed"] is False
    assert cleared["psf"] is None
    assert cleared["wavefront_aberrations_um"] is None
    assert cleared["pupil_function"] is None
    assert cleared["pupil_amplitude"] is None
    assert cleared["pupil_phase"] is None
    assert cleared["areapix"] is None
    assert cleared["areapixapod"] is None
    assert np.array_equal(np.asarray(wvf_get(cleared, "wave"), dtype=float), np.asarray(wvf_get(wvf, "wave"), dtype=float))
    assert np.asarray(wvf_get(cleared, "psf", 500.0), dtype=float).shape == (201, 201)
    assert np.asarray(wvf_get(wvf, "psf"), dtype=float).shape == (201, 201, 2)


def test_wvf_pupil_function_and_compute_psf_support_explicit_aperture() -> None:
    aperture = np.ones((41, 41), dtype=float)
    aperture[:, 20:] *= 0.5
    aperture[18:23, 10:31] = 0.0

    wvf = wvf_create(wave=np.array([550.0], dtype=float))
    wvf = wvf_set(wvf, "spatial samples", 101)
    wvf = wvf_pupil_function(wvf, "aperture function", aperture)

    stored_aperture = np.asarray(wvf_get(wvf, "aperture function"), dtype=float)
    stored_pupil = np.asarray(wvf_get(wvf, "pupil function", 550.0), dtype=np.complex128)

    assert stored_aperture.shape == aperture.shape
    assert stored_pupil.shape == (101, 101)

    computed = wvf_compute_psf(wvf, "compute pupil func", False)
    psf = np.asarray(wvf_get(computed, "psf", 550.0), dtype=float)

    assert psf.shape == (101, 101)
    assert np.isclose(float(np.sum(psf)), 1.0)
    assert np.asarray(wvf_get(computed, "pupil function", 550.0), dtype=np.complex128).shape == (101, 101)


def test_wvf_aperture_clean_polygon_supports_wvf_pupil_function() -> None:
    wvf = wvf_create(wave=np.array([550.0], dtype=float))
    wvf = wvf_set(wvf, "spatial samples", 101)
    aperture, params = wvf_aperture(
        wvf,
        "n sides",
        8,
        "dot mean",
        0,
        "dot sd",
        0,
        "line mean",
        0,
        "line sd",
        0,
        "image rotate",
        0,
    )

    assert aperture.shape == (101, 101)
    assert np.isclose(float(np.max(aperture)), 1.0)
    assert np.isclose(float(np.min(aperture)), 0.0)
    assert int(params["nsides"]) == 8

    wvf = wvf_pupil_function(wvf, "aperture function", aperture)
    wvf = wvf_compute_psf(wvf, "compute pupil func", False)
    psf = np.asarray(wvf_get(wvf, "psf", 550.0), dtype=float)

    assert psf.shape == (101, 101)
    assert np.isclose(float(np.sum(psf)), 1.0)
    assert np.asarray(wvf_get(wvf, "aperture function"), dtype=float).ndim == 2


def test_wvf_compute_accepts_matlab_style_aperture_option() -> None:
    wvf = wvf_create(wave=np.array([550.0], dtype=float))
    wvf = wvf_set(wvf, "spatial samples", 101)
    aperture, _ = wvf_aperture(
        wvf,
        "n sides",
        8,
        "dot mean",
        0,
        "dot sd",
        0,
        "line mean",
        0,
        "line sd",
        0,
        "image rotate",
        0,
    )

    computed = wvf_compute(wvf, "aperture", aperture, "compute sce", False)
    psf = np.asarray(wvf_get(computed, "psf", 550.0), dtype=float)
    pupil = np.asarray(wvf_get(computed, "pupil function", 550.0), dtype=np.complex128)

    assert computed["computed"] is True
    assert psf.shape == (101, 101)
    assert pupil.shape == (101, 101)
    assert np.isclose(float(np.sum(psf)), 1.0)


def test_wvf_spatial_sampling_getters_match_spatial_model() -> None:
    wvf = wvf_create()
    wvf = wvf_set(wvf, "calc pupil diameter", 7.0 / 4.0, "mm")
    wvf = wvf_set(wvf, "focal length", 7e-3, "m")
    wvf = wvf_compute(wvf)

    wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
    n_pixels = int(wvf_get(wvf, "npixels"))
    pupil_plane_size_mm = float(wvf_get(wvf, "pupil plane size", "mm", wave))
    pupil_sample_spacing_mm = float(wvf_get(wvf, "pupil sample spacing", "mm", wave))
    pupil_positions_mm = np.asarray(wvf_get(wvf, "pupil positions", wave, "mm"), dtype=float)
    pupil_amplitude = np.asarray(wvf_get(wvf, "pupil function amplitude", wave), dtype=float)
    pupil_phase = np.asarray(wvf_get(wvf, "pupil function phase", wave), dtype=float)

    assert int(wvf_get(wvf, "calc nwave")) == int(np.asarray(wvf_get(wvf, "wave"), dtype=float).size)
    assert np.isclose(float(wvf_get(wvf, "psf sample spacing")), float(wvf_get(wvf, "ref psf sample interval")))
    assert np.isclose(pupil_sample_spacing_mm, pupil_plane_size_mm / n_pixels)
    assert pupil_positions_mm.shape == (n_pixels,)
    assert np.isclose(pupil_positions_mm[1] - pupil_positions_mm[0], pupil_sample_spacing_mm)
    assert pupil_amplitude.shape == (n_pixels, n_pixels)
    assert pupil_phase.shape == (n_pixels, n_pixels)


def test_oi_compute_accepts_wvf_input(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    wvf = wvf_create(wave=scene_get(scene, "wave"))
    wvf = wvf_set(wvf, "zcoeffs", np.array([2.0, 0.5], dtype=float), ["defocus", "vertical_astigmatism"])

    oi = oi_compute(wvf, scene, crop=True)

    assert oi.fields["optics"]["model"] == "shiftinvariant"
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "defocus")), 2.0)
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism")), 0.5)
    assert oi_get(oi, "photons").shape[:2] == scene.data["photons"].shape[:2]
    assert wvf_to_oi(wvf).fields["optics"]["model"] == "shiftinvariant"


def test_wvf2psf_returns_shift_invariant_psf_data() -> None:
    wvf = wvf_create(wave=np.array([500.0, 600.0], dtype=float))
    wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")

    psf_data, computed = wvf2PSF(wvf, False)
    psf = np.asarray(psf_data["psf"], dtype=float)

    assert computed["computed"] is True
    assert psf.shape == (128, 128, 2)
    assert np.allclose(np.asarray(psf_data["wave"], dtype=float), np.array([500.0, 600.0], dtype=float))
    assert np.allclose(np.asarray(psf_data["umPerSamp"], dtype=float), np.array([0.25, 0.25], dtype=float))
    assert np.all(psf >= 0.0)
    assert np.all(np.sum(psf, axis=(0, 1)) > 0.0)


def test_wvf2optics_builds_shift_invariant_optics_bundle() -> None:
    wvf = wvf_create(wave=np.array([500.0, 600.0], dtype=float))
    wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
    psf_data, _ = wvf2PSF(wvf, False)

    optics = wvf2optics(wvf)

    assert optics["model"] == "shiftinvariant"
    assert optics["name"] == "wvf"
    assert optics["compute_method"] == "opticsotf"
    assert optics["wavefront"]["computed"] is True
    assert np.asarray(optics["psf_data"]["psf"], dtype=float).shape == np.asarray(psf_data["psf"], dtype=float).shape
    assert np.asarray(optics["otf_data"], dtype=complex).shape == (128, 128, 2)
    assert np.allclose(np.asarray(optics["otf_wave"], dtype=float), np.array([500.0, 600.0], dtype=float))


def test_wvf_pupil_amplitude_alias_matches_wvf_aperture() -> None:
    wvf = wvf_create(wave=np.array([550.0], dtype=float))
    wvf = wvf_set(wvf, "spatial samples", 101)
    args = (
        "n sides",
        8,
        "dot mean",
        0,
        "dot sd",
        0,
        "line mean",
        0,
        "line sd",
        0,
        "image rotate",
        0,
        "seed",
        1,
    )

    aperture, params = wvf_aperture(wvf, *args)
    alias_aperture, alias_params = wvfPupilAmplitude(wvf, *args)

    assert np.allclose(alias_aperture, aperture)
    assert alias_params == params


def test_psf_helper_wrappers_match_legacy_contract() -> None:
    psf = np.array(
        [
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 8.0, 2.0, 0.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    peak_row, peak_col = psfFindPeak(psf)
    assert (peak_row, peak_col) == (2, 3)

    centered, source_row, source_col = psfCenter(psf)
    centered_peak = psfFindPeak(centered)
    assert (source_row, source_col) == (2, 3)
    assert centered.shape == psf.shape
    assert centered_peak == (3, 3)
    assert np.isclose(np.sum(centered), np.sum(psf), atol=1e-10, rtol=1e-10)

    support = np.arange(-2.0, 3.0, 1.0, dtype=float)
    volume, normalized = psfVolume(centered, support, support)
    assert np.isclose(volume, np.sum(centered), atol=1e-10, rtol=1e-10)
    assert np.isclose(np.sum(normalized), 1.0, atol=1e-10, rtol=1e-10)

    radius = float(psfFindCriterionRadius(psf, 0.9))
    assert np.isclose(radius, 1.5, atol=1e-10, rtol=1e-10)


def test_psf_lsf_and_circular_average_wrappers_match_legacy_contract() -> None:
    psf = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 4.0, 2.0, 0.0],
            [1.0, 4.0, 8.0, 4.0, 1.0],
            [0.0, 2.0, 4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    psf_stack = np.stack((psf, 2.0 * psf), axis=2)

    horizontal_lsf = np.asarray(psf2lsf(psf), dtype=float)
    vertical_lsf = np.asarray(psf2lsf(psf, "direction", "vertical"), dtype=float)
    stacked_lsf = np.asarray(psf2lsf(psf_stack), dtype=float)

    assert np.allclose(horizontal_lsf, np.sum(psf, axis=1))
    assert np.allclose(vertical_lsf, np.sum(psf, axis=0))
    assert stacked_lsf.shape == (psf.shape[0], 2)
    assert np.allclose(stacked_lsf[:, 0], horizontal_lsf)
    assert np.allclose(stacked_lsf[:, 1], 2.0 * horizontal_lsf)

    circular_psf = np.asarray(lsf2circularpsf(horizontal_lsf), dtype=float)
    assert circular_psf.shape == psf.shape
    assert np.isclose(np.sum(circular_psf), 1.0, atol=1e-10, rtol=1e-10)
    assert psfFindPeak(circular_psf) == (3, 3)

    yy, xx = np.indices((7, 7), dtype=float)
    astigmatic_psf = np.exp(-(((yy - 3.0) ** 2) / 4.0 + ((xx - 3.0) ** 2) / 1.0))
    circular_average = np.asarray(psfCircularlyAverage(astigmatic_psf), dtype=float)

    assert circular_average.shape == astigmatic_psf.shape
    assert np.isclose(np.sum(circular_average), np.sum(astigmatic_psf), atol=1e-10, rtol=1e-10)
    assert psfFindPeak(circular_average) == (4, 4)
    assert np.isclose(circular_average[3, 2], circular_average[3, 4], atol=1e-10, rtol=1e-10)
    assert np.isclose(circular_average[2, 3], circular_average[4, 3], atol=1e-10, rtol=1e-10)

    averaged = np.asarray(psfAverageMultiple(psf_stack), dtype=float)
    assert np.allclose(averaged, 1.5 * psf)


def test_oi_set_wvf_prefixed_parameter_rebuilds_oi(asset_store) -> None:
    scene = scene_create("point array", 64, 16, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 1.5)
    oi = oi_compute(oi_create("wvf"), scene, crop=True)

    original_wangular = float(oi_get(oi, "wangular"))
    updated = oi_set(oi, "wvf zcoeffs", 1.5, "defocus")

    assert np.isclose(float(oi_get(updated, "wvf zcoeffs", "defocus")), 1.5)
    assert np.isclose(float(oi_get(updated, "wvf pupil diameter", "mm")), float(oi_get(oi, "wvf pupil diameter", "mm")))
    assert np.isclose(float(oi_get(updated, "wangular")), original_wangular)
    assert updated.data == {}

    recomputed = oi_compute(updated, scene, crop=True)
    assert recomputed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_set_optics_wvf_rebuilds_oi_from_wavefront(asset_store) -> None:
    scene = scene_create("point array", 64, 16, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 1.5)
    oi = oi_compute(oi_create("wvf"), scene, crop=True)

    wavefront = wvf_set(oi_get(oi, "optics wvf"), "zcoeffs", 0.75, "defocus")
    updated = oi_set(oi, "optics wvf", wavefront)

    assert np.isclose(float(oi_get(updated, "wvf zcoeffs", "defocus")), 0.75)
    assert updated.data == {}

    recomputed = oi_compute(updated, scene, crop=True)
    assert recomputed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_create_psf_builds_default_shift_invariant_psf_optics() -> None:
    oi = oi_create("psf")

    assert oi.fields["optics"]["compute_method"] == "opticsotf"
    assert oi.fields["optics"]["model"] == "shiftinvariant"
    psf_data = oi_get(oi, "psfdata")
    assert psf_data is not None
    assert np.asarray(psf_data["psf"]).ndim == 3
    assert np.asarray(psf_data["wave"]).shape == (31,)
    assert np.allclose(np.asarray(psf_data["umPerSamp"], dtype=float), np.array([0.25, 0.25], dtype=float))
    assert np.asarray(oi.fields["optics"]["otf_data"]).shape == (129, 129, 31)


def test_oi_create_psf_accepts_custom_shift_invariant_psf_data(asset_store) -> None:
    psf = np.zeros((33, 33, 1), dtype=float)
    psf[16, 16, 0] = 1.0
    oi = oi_create(
        "psf",
        {
            "psf": psf,
            "wave": np.array([550.0], dtype=float),
            "umPerSamp": np.array([0.25, 0.25], dtype=float),
        },
    )
    stored = oi_get(oi, "psfdata")
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    computed = oi_compute(oi, scene, crop=True)

    assert np.asarray(stored["psf"]).shape == (33, 33, 1)
    assert np.array_equal(np.asarray(stored["wave"]), np.array([550.0], dtype=float))
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert computed.data["photons"].shape[2] == scene.data["photons"].shape[2]


def test_oi_psf_area_and_diameter_match_manual_threshold_support() -> None:
    oi = oi_create("psf")
    this_wave = 550.0
    threshold = 0.1

    area = float(oiPSF(oi, "area", "units", "um", "threshold", threshold, "wave", this_wave))
    diameter = float(oiPSF(oi, "diameter", "units", "um", "threshold", threshold, "wave", this_wave))

    psf_data = oi_get(oi, "psfdata")
    psf_stack = np.asarray(psf_data["psf"], dtype=float)
    wave = np.asarray(psf_data["wave"], dtype=float).reshape(-1)
    wave_index = int(np.argmin(np.abs(wave - this_wave)))
    psf = psf_stack[:, :, wave_index]
    um_per_samp = np.asarray(psf_data["umPerSamp"], dtype=float).reshape(-1)
    mask = np.asarray(psf >= (threshold * float(np.max(psf))), dtype=float)
    expected_area = float(np.sum(mask) * np.prod(um_per_samp))
    expected_diameter = float(2.0 * np.sqrt(expected_area / np.pi))

    assert np.isclose(area, expected_area, atol=1e-12, rtol=1e-12)
    assert np.isclose(diameter, expected_diameter, atol=1e-12, rtol=1e-12)


def test_make_combined_otf_and_make_cmatrix_match_closed_form() -> None:
    otf = np.array(
        [
            [1.0, 0.8, 0.5],
            [0.7, 0.4, 0.2],
            [0.6, 0.3, 0.1],
        ],
        dtype=float,
    )
    sample_sf = np.array([0.0, 8.0, 16.0], dtype=float)

    expected_factor = 0.3481 + 0.6519 * np.exp(-0.1212 * sample_sf)
    combined_otf = makeCombinedOtf(otf, sample_sf)
    assert np.allclose(combined_otf, otf * expected_factor.reshape(1, -1), atol=1e-12, rtol=1e-12)

    receptor = np.array(
        [
            [0.8, 0.2, 0.1],
            [0.3, 0.9, 0.2],
            [0.1, 0.4, 0.7],
        ],
        dtype=float,
    )
    monitor_spd = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.2, 0.8, 0.3],
            [0.1, 0.3, 0.9],
        ],
        dtype=float,
    )
    expected_cmatrix = np.column_stack(
        [
            (receptor.T @ (combined_otf[:, index][:, None] * monitor_spd)).reshape(-1, order="F")
            for index in range(combined_otf.shape[1])
        ]
    )

    assert np.allclose(makeCmatrix(combined_otf, receptor, monitor_spd), expected_cmatrix, atol=1e-12, rtol=1e-12)


def test_retinal_image_matches_frequency_domain_contract() -> None:
    image = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.25, 0.15, 0.05],
            [0.40, 0.10, 0.20],
            [0.15, 0.35, 0.25],
            [0.30, 0.45, 0.10],
            [0.05, 0.25, 0.40],
        ],
        dtype=float,
    )
    identity_cmatrices = np.repeat(np.eye(3, dtype=float).reshape(9, 1, order="F"), 4, axis=1)
    assert np.allclose(retinalImage(image, identity_cmatrices), image, atol=1e-12, rtol=1e-12)

    cmatrices = identity_cmatrices.copy()
    cmatrices[:, 1] = np.diag([1.20, 0.85, 0.60]).reshape(-1, order="F")
    cmatrices[:, 2] = np.diag([0.95, 1.10, 0.75]).reshape(-1, order="F")
    cmatrices[:, 3] = np.diag([0.70, 0.90, 1.15]).reshape(-1, order="F")

    image_fft = np.fft.fft(image.T, axis=1)
    expected_fft = np.zeros_like(image_fft, dtype=np.complex128)
    max_sf = cmatrices.shape[1] - 1
    for frequency_index in range(max_sf + 1):
        cmatrix = cmatrices[:, frequency_index].reshape(3, 3, order="F")
        expected_fft[:, frequency_index] = cmatrix @ image_fft[:, frequency_index]
    for frequency_index in range(max_sf + 1, min(2 * max_sf, image_fft.shape[1] - 1) + 1):
        mirrored_index = 2 * max_sf + 1 - frequency_index
        cmatrix = cmatrices[:, mirrored_index - 1].reshape(3, 3, order="F")
        expected_fft[:, frequency_index] = cmatrix @ image_fft[:, frequency_index]
    expected = np.real(np.fft.ifft(expected_fft, axis=1)).T

    assert np.allclose(retinalImage(image, cmatrices), expected, atol=1e-12, rtol=1e-12)


def test_optics_psf_to_otf_builds_custom_otf_struct(asset_store) -> None:
    otf = optics_psf_to_otf(
        asset_store.resolve("data/optics/flare/flare1.png"),
        1.2e-6,
        np.arange(400.0, 701.0, 10.0, dtype=float),
    )

    assert otf["function"] == "custom"
    assert np.asarray(otf["OTF"]).shape[2] == 31
    assert np.asarray(otf["fx"]).ndim == 1
    assert np.asarray(otf["fy"]).ndim == 1
    assert np.isclose(float(np.abs(np.asarray(otf["OTF"])[0, 0, 0])), 1.0)


def test_oi_set_optics_otfstruct_supports_custom_shift_invariant_otf(asset_store) -> None:
    scene = scene_create("point array", 64, 16, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 40.0)
    otf = optics_psf_to_otf(
        asset_store.resolve("data/optics/flare/flare1.png"),
        1.2e-6,
        np.arange(400.0, 701.0, 10.0, dtype=float),
    )
    oi = oi_set(oi_create("shift invariant"), "optics otfstruct", otf)
    computed = oi_compute(oi, scene, crop=True)

    stored = oi_get(oi, "optics otfstruct")
    assert stored is not None
    assert oi.fields["optics"]["compute_method"] == "opticsotf"
    assert np.asarray(stored["OTF"]).shape == np.asarray(otf["OTF"]).shape
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_oi_get_optics_otf_synthesizes_shift_invariant_otf_after_compute(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create("shift invariant"), scene, crop=True)

    otf = oi_get(oi, "optics OTF")
    fx = oi_get(oi, "optics OTF fx")
    fy = oi_get(oi, "optics OTF fy")

    assert otf is not None
    assert fx is not None
    assert fy is not None
    assert np.asarray(otf).ndim == 3
    assert np.asarray(otf).shape[2] == oi.data["photons"].shape[2]
    assert np.asarray(otf).shape[0] == np.asarray(fy).size
    assert np.asarray(otf).shape[1] == np.asarray(fx).size
    otf_plane = np.abs(np.asarray(otf)[:, :, 0])
    dc_value = float(otf_plane[0, 0])
    assert dc_value > 0.5
    assert np.isfinite(np.asarray(otf)).all()
    assert np.isclose(float(np.max(otf_plane[:2, :2])), float(np.max(otf_plane)), atol=1e-6)


def test_oi_set_optics_otf_supports_direct_raw_shift_invariant_otf(asset_store) -> None:
    params = {
        "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
        "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
        "blockSize": 16,
        "contrast": 1.0,
    }
    scene = scene_create("frequency orientation", params, asset_store=asset_store)
    scene = scene_set(scene, "fov", 3.0)
    base = oi_compute(oi_create("shift invariant"), scene, crop=True)
    raw_otf = oi_get(base, "optics OTF")
    assert raw_otf is not None

    custom_otf = np.zeros_like(np.asarray(raw_otf), dtype=complex)
    center_row = custom_otf.shape[0] // 2
    center_col = custom_otf.shape[1] // 2
    custom_otf[center_row, center_col, :] = 1.0
    ideal = oi_set(base, "optics OTF", custom_otf)
    stored = oi_get(ideal, "optics OTF")
    computed = oi_compute(ideal, scene, crop=True)

    assert stored is not None
    assert np.asarray(stored).shape == np.asarray(raw_otf).shape
    assert np.allclose(np.asarray(stored), custom_otf)
    assert ideal.fields["optics"]["compute_method"] == "opticsotf"
    assert not np.allclose(computed.data["photons"], base.data["photons"])


def test_oi_set_optics_otf_repeats_2d_otf_across_wave(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    base = oi_compute(oi_create("shift invariant"), scene, crop=True)
    raw_otf = oi_get(base, "optics OTF")
    assert raw_otf is not None
    samples = np.asarray(raw_otf).shape[0]
    support = np.linspace(-1.0, 1.0, samples, dtype=float)
    xx, yy = np.meshgrid(support, support)
    gaussian = np.exp(-((xx**2) + (yy**2)) / 0.1)

    oi = oi_set(base, "optics OTF", gaussian)
    stored = np.asarray(oi_get(oi, "optics OTF"))

    assert stored.shape == np.asarray(raw_otf).shape
    assert np.allclose(stored[:, :, 0], gaussian)
    assert np.allclose(stored[:, :, -1], gaussian)


def test_si_synthetic_gaussian_builds_anisotropic_shift_invariant_optics() -> None:
    oi = oi_create("shift invariant")
    wave = np.asarray(oi_get(oi, "wave"), dtype=float)

    optics = si_synthetic("gaussian", oi, wave / wave[0], 2.0)
    updated = oi_set(oi, "optics", optics)
    psf_data = oi_get(updated, "psfdata")
    psf = np.asarray(psf_data["psf"], dtype=float)

    assert updated.fields["optics"]["model"] == "shiftinvariant"
    assert updated.fields["optics"]["compute_method"] == "opticsotf"
    assert psf.shape == (129, 129, wave.size)
    center = psf.shape[0] // 2
    horizontal = np.count_nonzero(psf[center, :, 0] > 1e-8)
    vertical = np.count_nonzero(psf[:, center, 0] > 1e-8)
    assert horizontal > vertical


def test_optics_gaussian_psf_point_array_script_flow(asset_store) -> None:
    wave = np.arange(450.0, 651.0, 100.0, dtype=float)
    scene = scene_create("point array", 128, 32, asset_store=asset_store)
    original_mean = float(scene_get(scene, "mean luminance", asset_store=asset_store))

    scene = scene_interpolate_w(scene, wave, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 1.0)
    scene = scene_set(scene, "name", "psfPointArray")

    oi = oi_create()
    oi = oi_set(oi, "wave", scene_get(scene, "wave"))
    optics = si_synthetic("gaussian", oi, wave / wave[0], np.full(wave.size, 3.0, dtype=float))
    oi = oi_set(oi, "optics", optics)
    oi = oi_compute(oi, scene)

    assert scene.name == "psfPointArray"
    assert np.asarray(scene.fields["wave"]).shape == (3,)
    assert scene.data["photons"].shape == (128, 128, 3)
    assert np.asarray(scene.fields["illuminant_photons"]).shape == (3,)
    assert float(scene_get(scene, "mean luminance", asset_store=asset_store)) == pytest.approx(original_mean, rel=1e-6)
    assert np.asarray(oi_get(oi, "wave")).shape == (3,)
    assert oi.data["photons"].shape[2] == 3
    assert oi.data["photons"].shape[:2] == tuple(np.asarray(oi_get(oi, "size"), dtype=int))
    assert oi.data["photons"].shape[0] > scene.data["photons"].shape[0]
    assert np.max(oi.data["photons"]) > 0.0


def test_optics_psf_plot_script_flow() -> None:
    oi = oi_create("diffraction limited")
    optics = dict(oi.fields["optics"])
    optics["f_number"] = 12.0
    oi.fields["optics"] = optics

    psf_data = oi_get(oi, "psf data", 600.0, "um", 100)
    xy = np.asarray(psf_data["xy"], dtype=float)
    psf = np.asarray(psf_data["psf"], dtype=float)

    assert psf.shape == (200, 200)
    assert xy.shape == (200, 200, 2)
    assert np.all(psf >= 0.0)
    assert float(psf[psf.shape[0] // 2, psf.shape[1] // 2]) == pytest.approx(float(np.max(psf)))
    assert float(airy_disk(600.0, float(oi_get(oi, "fnumber")), "units", "um")) == pytest.approx(8.784)


def test_si_synthetic_lorentzian_applies_to_grid_lines_scene(asset_store) -> None:
    scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.0)
    oi = oi_create("psf")
    gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)

    optics = si_synthetic("lorentzian", oi, gamma)
    updated = oi_set(oi, "optics", optics)
    computed = oi_compute(updated, scene, crop=True)

    assert updated.fields["optics"]["compute_method"] == "opticsotf"
    assert updated.fields["optics"]["model"] == "shiftinvariant"
    assert np.asarray(oi_get(updated, "psfdata")["psf"]).shape[2] == np.asarray(oi_get(oi, "wave")).size
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)


def test_oi_si_lorentzian_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_si_lorentzian_small", asset_store=asset_store)

    assert payload["photons"].ndim == 3
    assert payload["photons"].shape[2] == np.asarray(payload["wave"]).size


def test_oi_psf550_si_lorentzian_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_psf550_si_lorentzian_small", asset_store=asset_store)

    assert payload["x"].ndim == 2
    assert payload["y"].ndim == 2
    assert payload["psf"].shape == payload["x"].shape == payload["y"].shape
    assert np.all(payload["psf"] >= 0.0)


def test_oi_si_pillbox_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_si_pillbox_small", asset_store=asset_store)

    assert np.asarray(payload["wave"]).ndim == 1
    assert np.asarray(payload["input_psf_mid_row_550"]).ndim == 1


def test_oi_si_gaussian_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_si_gaussian_small", asset_store=asset_store)

    assert np.asarray(payload["wave"]).ndim == 1
    assert np.asarray(payload["input_psf_mid_row_550"]).ndim == 1


def test_oi_si_gaussian_ratio_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_si_gaussian_ratio_small", asset_store=asset_store)

    assert np.asarray(payload["wave"]).ndim == 1
    assert np.asarray(payload["input_psf_mid_row_550"]).ndim == 1


def test_oi_gaussian_psf_point_array_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_gaussian_psf_point_array_small", asset_store=asset_store)

    assert payload["scene_name"] == "psfPointArray"
    assert np.asarray(payload["wave"]).shape == (3,)
    assert np.asarray(payload["scene_wave"]).shape == (3,)
    assert np.asarray(payload["scene_size"]).shape == (2,)
    assert np.asarray(payload["oi_size"]).shape == (2,)
    assert payload["center_row_normalized"].shape == (int(np.asarray(payload["oi_size"])[1]), 3)
    assert payload["center_col_normalized"].shape == (int(np.asarray(payload["oi_size"])[0]), 3)


def test_oi_psf550_si_gaussian_ratio_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_psf550_si_gaussian_ratio_small", asset_store=asset_store)

    assert payload["x"].ndim == 2
    assert payload["y"].ndim == 2
    assert payload["psf"].shape == payload["x"].shape == payload["y"].shape
    assert np.all(payload["psf"] >= 0.0)


def test_oi_illuminance_lines_si_gaussian_ratio_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_illuminance_lines_si_gaussian_ratio_small", asset_store=asset_store)

    assert payload["xy_middle"].shape == (2,)
    assert payload["v_pos"].ndim == 1
    assert payload["h_pos"].ndim == 1
    assert payload["v_data"].shape == payload["v_pos"].shape
    assert payload["h_data"].shape == payload["h_pos"].shape


def test_oi_si_custom_file_small_parity_case(asset_store) -> None:
    payload = run_python_case("oi_si_custom_file_small", asset_store=asset_store)

    assert payload["photons"].ndim == 3
    assert payload["photons"].shape[2] == np.asarray(payload["wave"]).size
    assert np.asarray(payload["input_psf_mid_row_550"]).ndim == 1


def test_optics_airy_disk_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_airy_disk_small", asset_store=asset_store)

    assert float(payload["radius_um"]) > 0.0
    assert float(payload["diameter_um"]) == pytest.approx(float(payload["radius_um"]) * 2.0)
    assert int(payload["image_rows"]) > 0
    assert int(payload["image_cols"]) > 0


def test_optics_coc_matches_thin_lens_formula() -> None:
    oi = oi_create()
    optics = dict(oi.fields["optics"])
    optics["focal_length_m"] = 0.050
    optics["f_number"] = 2.0

    circ_mm, x_dist_m = optics_coc(optics, 0.5, "unit", "mm", "n samples", 50)
    expected_dof = 2.0 * 2.0 * 20e-6 * (0.5**2) / (0.050**2)

    assert x_dist_m.shape == (50,)
    assert circ_mm.shape == (50,)
    assert np.all(circ_mm >= 0.0)
    assert np.any(x_dist_m < 0.5)
    assert np.any(x_dist_m > 0.5)
    assert np.isclose(float(optics_dof(optics, 0.5, 20e-6)), expected_dof)


def test_optics_defocus_displacement_matches_lens_power_formula() -> None:
    base_diopters = np.array([50.0, 150.0, 250.0, 350.0], dtype=float)
    delta_diopters = np.arange(1.0, 16.0, dtype=float)

    displacement = np.asarray(
        optics_defocus_displacement(base_diopters[:, None], delta_diopters[None, :]),
        dtype=float,
    )
    expected = (1.0 / base_diopters[:, None]) - (1.0 / (base_diopters[:, None] + delta_diopters[None, :]))

    ratio_base_diopters = np.arange(50.0, 301.0, 50.0, dtype=float)
    ratio_displacement = np.asarray(
        optics_defocus_displacement(ratio_base_diopters, ratio_base_diopters / 10.0),
        dtype=float,
    )

    assert displacement.shape == (4, 15)
    assert np.allclose(displacement, expected)
    assert np.allclose(ratio_displacement * ratio_base_diopters, np.full(ratio_base_diopters.shape, 1.0 / 11.0))


def test_optics_dof_script_workflow_matches_coc_crossings() -> None:
    oi = oi_create()
    optics = dict(oi.fields["optics"])
    optics["f_number"] = 2.0
    optics["focal_length_m"] = 0.100

    object_distance_m = 2.0
    coc_diameter_m = 50e-6

    dof_formula_m = float(optics_dof(optics, object_distance_m, coc_diameter_m))
    coc_curve_m, x_dist_m = optics_coc(optics, object_distance_m, "nsamples", 200)
    idx1 = int(np.argmin(np.abs(coc_curve_m[:100] - coc_diameter_m)))
    idx2 = int(np.argmin(np.abs(coc_curve_m[100:] - coc_diameter_m))) + 100
    coc_dof_m = float(x_dist_m[idx2] - x_dist_m[idx1])

    object_distances_m = np.arange(0.5, 20.0 + 1e-12, 0.25, dtype=float)
    f_numbers = np.arange(2.0, 12.0 + 1e-12, 0.25, dtype=float)
    dof_surface_m = np.zeros((object_distances_m.size, f_numbers.size), dtype=float)
    optics_sweep = dict(optics)
    for column_index, f_number in enumerate(f_numbers):
        optics_sweep["f_number"] = float(f_number)
        dof_surface_m[:, column_index] = np.asarray(optics_dof(optics_sweep, object_distances_m, 20e-6), dtype=float)

    assert np.isclose(dof_formula_m, 0.08)
    assert x_dist_m.shape == (200,)
    assert coc_curve_m.shape == (200,)
    assert 0 <= idx1 < 100
    assert 100 <= idx2 < 200
    assert coc_dof_m > 0.0
    assert np.isclose(coc_dof_m / dof_formula_m, 1.0, rtol=0.2)
    assert dof_surface_m.shape == (79, 41)
    assert dof_surface_m[-1, 0] > dof_surface_m[0, 0]
    assert dof_surface_m[0, -1] > dof_surface_m[0, 0]


def test_optics_depth_defocus_matches_thin_lens_script_workflow() -> None:
    optics = dict(oi_create().fields["optics"])
    focal_length_m = float(optics["focal_length_m"])
    lens_power_diopters = 1.0 / focal_length_m
    object_distance_m = np.linspace(focal_length_m * 1.5, 100.0 * focal_length_m, 500, dtype=float)

    defocus_diopters, image_distance_m = optics_depth_defocus(object_distance_m, optics)
    defocus_diopters = np.asarray(defocus_diopters, dtype=float)
    image_distance_m = np.asarray(image_distance_m, dtype=float)

    expected_image_distance_m = (focal_length_m * object_distance_m) / (object_distance_m - focal_length_m)
    expected_defocus_diopters = (1.0 / expected_image_distance_m) - (1.0 / focal_length_m)

    shifted_scale = 1.1
    shifted_defocus_diopters, shifted_image_distance_m = optics_depth_defocus(
        object_distance_m,
        optics,
        shifted_scale * focal_length_m,
    )
    shifted_defocus_diopters = np.asarray(shifted_defocus_diopters, dtype=float)
    shifted_image_distance_m = np.asarray(shifted_image_distance_m, dtype=float)
    focus_index = int(np.argmin(np.abs(shifted_defocus_diopters)))

    pupil_radius_m = focal_length_m / (2.0 * float(optics["f_number"]))
    pupil_radius_scales = np.array([0.5, 1.5, 3.0], dtype=float)
    w20 = ((pupil_radius_scales[None, :] * pupil_radius_m) ** 2 / 2.0) * (
        lens_power_diopters * shifted_defocus_diopters[:, None]
    ) / (lens_power_diopters + shifted_defocus_diopters[:, None])

    assert np.allclose(image_distance_m, expected_image_distance_m)
    assert np.allclose(defocus_diopters, expected_defocus_diopters)
    assert np.allclose(shifted_image_distance_m, expected_image_distance_m)
    assert np.all(image_distance_m > focal_length_m)
    assert np.all(defocus_diopters < 0.0)
    assert np.all(np.diff(defocus_diopters) >= 0.0)
    assert np.any(shifted_defocus_diopters < 0.0)
    assert np.any(shifted_defocus_diopters > 0.0)
    assert np.isclose(object_distance_m[focus_index] / focal_length_m, 11.0, atol=0.05)
    assert w20.shape == (500, 3)
    assert np.isclose(w20[focus_index, 0], 0.0, atol=1e-8)


def test_optics_defocus_core_and_build_2d_otf_support_shift_invariant_bundle() -> None:
    oi = oi_create()
    optics = dict(oi.fields["optics"])
    optics["model"] = "shiftinvariant"
    wave = np.asarray(optics["transmittance"]["wave"], dtype=float).reshape(-1)
    sample_sf = np.linspace(0.0, 40.0, 25, dtype=float)
    defocus = np.full(wave.shape, 5.0, dtype=float)

    otf_rows, sample_sf_mm = optics_defocus_core(optics, sample_sf, defocus)
    updated = optics_build_2d_otf(optics, otf_rows, sample_sf_mm)

    assert otf_rows.shape == (wave.size, sample_sf.size)
    assert sample_sf_mm.shape == sample_sf.shape
    assert np.all(np.isfinite(otf_rows))
    assert np.all(np.diff(sample_sf_mm) >= 0.0)
    assert updated["model"] == "shiftinvariant"
    assert updated["compute_method"] == "opticsotf"
    assert np.asarray(updated["otf_data"]).shape[2] == wave.size
    assert np.isclose(float(np.real(updated["otf_data"][0, 0, 0])), 1.0)


def test_optics_defocus_scene_workflow_supports_defocus_otf_bundle(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    scene_path = asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
    scene = scene_from_file(scene_path, "multispectral", None, None, wave, asset_store=asset_store)
    scene = scene_set(scene, "fov", 5.0)
    max_sf = float(scene_get(scene, "max freq res", "cpd"))
    sample_sf = np.linspace(0.0, max_sf, min(int(np.ceil(max_sf)), 70), dtype=float)
    scene = scene_adjust_illuminant(scene, "D65.mat", asset_store=asset_store)

    base_optics = dict(oi_get(oi_create(), "optics"))
    base_optics["model"] = "shiftinvariant"
    optics_wave = np.asarray(base_optics["transmittance"]["wave"], dtype=float).reshape(-1)

    def _build_oi(defocus_diopters: float) -> tuple[object, float]:
        otf_rows, sample_sf_mm = optics_defocus_core(base_optics, sample_sf, np.full(optics_wave.shape, defocus_diopters, dtype=float))
        current_optics = optics_build_2d_otf(base_optics, otf_rows, sample_sf_mm)
        oi = oi_set(oi_create(), "optics", current_optics)
        oi = oi_compute(oi, scene)
        photons = np.asarray(oi_get(oi, "photons"), dtype=float)
        wave_index = int(np.argmin(np.abs(np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1) - 550.0)))
        peak = float(np.max(photons[:, :, wave_index]))
        return oi, peak

    oi_focus, peak_focus = _build_oi(0.0)
    oi_defocus5, peak_defocus5 = _build_oi(5.0)

    focal_length_m = float(base_optics.get("focal_length_m", base_optics.get("focalLength", 0.0)))
    lens_power = 1.0 / focal_length_m
    d10 = (1.0 / (focal_length_m - 10e-6)) - lens_power
    d40 = (1.0 / (focal_length_m - 40e-6)) - lens_power
    _, peak_10 = _build_oi(float(d10))
    _, peak_40 = _build_oi(float(d40))

    assert np.asarray(oi_get(oi_focus, "wave")).shape == wave.shape
    assert np.asarray(oi_get(oi_defocus5, "photons")).shape == np.asarray(oi_get(oi_focus, "photons")).shape
    assert d40 > d10 > 0.0
    assert peak_focus > peak_10 > peak_40
    assert peak_focus > peak_defocus5


def test_optics_coc_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_coc_small", asset_store=asset_store)

    assert np.array_equal(payload["object_distances_m"], np.array([0.5, 3.0], dtype=float))
    assert np.array_equal(payload["f_numbers"], np.array([2.0, 8.0], dtype=float))
    assert np.isclose(float(payload["focal_length_m"]), 0.050)
    assert payload["x_dist_focus_0_5_m"].shape == (50,)
    assert payload["circ_f2_focus_0_5_mm"].shape == (50,)
    assert payload["circ_f8_focus_0_5_mm"].shape == (50,)
    assert payload["x_dist_focus_3_m"].shape == (50,)
    assert payload["circ_f2_focus_3_mm"].shape == (50,)
    assert payload["circ_f8_focus_3_mm"].shape == (50,)
    assert np.all(payload["circ_f2_focus_0_5_mm"] >= 0.0)
    assert np.all(payload["circ_f8_focus_3_mm"] >= 0.0)


def test_optics_diffraction_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_diffraction_small", asset_store=asset_store)

    assert np.array_equal(payload["scene_wave"], np.arange(400.0, 701.0, 10.0, dtype=float))
    assert np.isclose(float(payload["scene_fov_deg"]), 1.0)
    assert np.isclose(float(payload["default_f_number"]), 4.0)
    assert np.array_equal(payload["default_oi_size"], np.array([160, 160], dtype=int))
    assert np.isclose(float(payload["large_f_number"]), 12.0)
    assert np.array_equal(payload["large_oi_size"], np.array([160, 160], dtype=int))
    assert float(payload["focal_length_mm"]) > 0.0
    assert float(payload["pupil_diameter_mm"]) > 0.0
    assert np.isclose(float(payload["focal_to_pupil_ratio"]), 12.0)
    assert payload["psf_550"].shape == payload["psf_x"].shape == payload["psf_y"].shape
    assert payload["ls_x_um"].shape == (41,)
    assert np.array_equal(payload["ls_wavelength"], np.arange(400.0, 701.0, 10.0, dtype=float))
    assert payload["ls_wave"].shape == (31, 41)
    assert payload["oi_center_row_550_widths"].shape == (3,)


def test_optics_flare_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_flare_small", asset_store=asset_store)

    assert np.isclose(float(payload["pupil_diameter_mm"]), 3.0)
    assert np.isclose(float(payload["focal_length_mm"]), 7.0)
    assert np.isclose(float(payload["f_number"]), 7.0 / 3.0)
    assert np.isclose(float(payload["point_scene_fov_deg"]), 1.0)
    assert np.isclose(float(payload["hdr_scene_fov_deg"]), 1.0)
    assert int(payload["seed_initial"]) == 1
    assert int(payload["seed_five"]) == 2
    assert int(payload["seed_defocus"]) == 3
    assert int(payload["initial_nsides"]) == 3
    assert int(payload["five_nsides"]) == 5
    assert np.isclose(float(payload["defocus_zcoeff"]), 1.0)
    assert int(payload["defocus_nsides"]) == 3
    assert float(payload["initial_aperture_sum"]) > 0.0
    assert float(payload["five_aperture_sum"]) > 0.0
    assert float(payload["defocus_aperture_sum"]) > 0.0
    assert payload["initial_psf_center_row_550_norm"].shape == (129,)
    assert payload["five_psf_center_row_550_norm"].shape == (129,)
    assert payload["defocus_psf_center_row_550_norm"].shape == (129,)
    assert payload["initial_psf_widths"].shape == (3,)
    assert payload["five_psf_widths"].shape == (3,)
    assert payload["defocus_psf_widths"].shape == (3,)
    assert payload["initial_point_oi_size"].shape == (2,)
    assert payload["five_point_oi_size"].shape == (2,)
    assert payload["initial_hdr_oi_size"].shape == (2,)
    assert payload["five_hdr_oi_size"].shape == (2,)
    assert payload["defocus_hdr_oi_size"].shape == (2,)
    assert np.isclose(float(payload["initial_hdr_mean_photons_550_ratio"]), 1.0)
    assert float(payload["five_hdr_mean_photons_550_ratio"]) > 1.0
    assert np.isclose(float(payload["defocus_hdr_mean_photons_550_ratio"]), 1.0, rtol=1e-3)


def test_optics_flare2_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_flare2_small", asset_store=asset_store)

    assert np.isclose(float(payload["pupil_diameter_mm"]), 3.0)
    assert np.isclose(float(payload["focal_length_mm"]), 7.0)
    assert np.isclose(float(payload["f_number"]), 7.0 / 3.0)
    assert np.isclose(float(payload["point_scene_fov_deg"]), 1.0)
    assert np.isclose(float(payload["hdr_scene_fov_deg"]), 3.0)
    assert int(payload["seed_initial"]) == 4
    assert int(payload["seed_five"]) == 5
    assert int(payload["seed_defocus"]) == 6
    assert int(payload["initial_nsides"]) == 6
    assert int(payload["five_nsides"]) == 5
    assert np.isclose(float(payload["defocus_zcoeff"]), 1.5)
    assert int(payload["defocus_nsides"]) == 3
    assert float(payload["initial_aperture_sum"]) > 0.0
    assert float(payload["five_aperture_sum"]) > 0.0
    assert float(payload["defocus_aperture_sum"]) > 0.0
    assert payload["initial_point_oi_size"].shape == (2,)
    assert payload["initial_point_oi_center_row_550_widths"].shape == (3,)
    assert payload["five_point_oi_size"].shape == (2,)
    assert payload["five_point_oi_center_row_550_widths"].shape == (3,)
    assert payload["initial_hdr_oi_size"].shape == (2,)
    assert payload["five_hdr_oi_size"].shape == (2,)
    assert payload["defocus_hdr_oi_size"].shape == (2,)
    assert np.isclose(float(payload["initial_hdr_mean_photons_550_ratio"]), 1.0)
    assert float(payload["five_hdr_mean_photons_550_ratio"]) > 1.0
    assert np.isclose(float(payload["defocus_hdr_mean_photons_550_ratio"]), 1.0, rtol=1e-3)


def test_optics_defocus_displacement_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_defocus_displacement_small", asset_store=asset_store)

    assert np.array_equal(payload["base_diopters"], np.array([50.0, 150.0, 250.0, 350.0], dtype=float))
    assert np.array_equal(payload["delta_diopters"], np.arange(1.0, 16.0, dtype=float))
    assert payload["displacement_curves_m"].shape == (4, 15)
    assert np.array_equal(payload["ratio_base_diopters"], np.arange(50.0, 301.0, 50.0, dtype=float))
    assert np.array_equal(payload["ratio_delta_diopters"], payload["ratio_base_diopters"] / 10.0)
    assert payload["ratio_displacement_m"].shape == (6,)
    assert np.allclose(payload["displacement_to_focal_length_ratio"], np.full(6, 1.0 / 11.0))


def test_optics_dof_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_dof_small", asset_store=asset_store)

    assert np.isclose(float(payload["f_number"]), 2.0)
    assert np.isclose(float(payload["focal_length_m"]), 0.100)
    assert np.isclose(float(payload["object_distance_m"]), 2.0)
    assert np.isclose(float(payload["coc_diameter_m"]), 50e-6)
    assert np.isclose(float(payload["dof_formula_m"]), 0.08)
    assert payload["coc_xdist_m"].shape == (200,)
    assert payload["coc_curve_m"].shape == (200,)
    assert int(payload["coc_idx1"]) < 100
    assert int(payload["coc_idx2"]) >= 100
    assert float(payload["coc_dof_m"]) > 0.0
    assert np.array_equal(payload["object_distances_m"], np.arange(0.5, 20.0 + 1e-12, 0.25, dtype=float))
    assert np.array_equal(payload["f_numbers"], np.arange(2.0, 12.0 + 1e-12, 0.25, dtype=float))
    assert np.isclose(float(payload["sweep_coc_diameter_m"]), 20e-6)
    assert payload["dof_surface_m"].shape == (79, 41)


def test_optics_depth_defocus_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_depth_defocus_small", asset_store=asset_store)

    assert float(payload["focal_length_m"]) > 0.0
    assert np.isclose(float(payload["lens_power_diopters"]), 1.0 / float(payload["focal_length_m"]))
    assert payload["object_distance_m"].shape == (500,)
    assert payload["focal_plane_relative_defocus"].shape == (500,)
    assert payload["image_distance_m"].shape == (500,)
    assert np.isclose(float(payload["shifted_image_plane_scale"]), 1.1)
    assert payload["shifted_defocus_diopters"].shape == (500,)
    assert float(payload["shifted_focus_object_distance_m"]) > float(payload["focal_length_m"])
    assert np.isclose(float(payload["shifted_focus_object_distance_focal_lengths"]), 11.0, atol=0.05)
    assert np.isclose(float(payload["pupil_radius_m"]), float(payload["focal_length_m"]) / 8.0)
    assert np.array_equal(payload["pupil_radius_scales"], np.array([0.5, 1.5, 3.0], dtype=float))
    assert payload["w20"].shape == (500, 3)


def test_optics_defocus_wvf_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_defocus_wvf_small", asset_store=asset_store)

    assert payload["wave"].shape == (31,)
    assert np.isclose(float(payload["scene_fov_deg"]), 1.5)
    assert np.isclose(float(payload["diffraction_limited_focal_length_mm"]), 8.0)
    assert np.isclose(float(payload["diffraction_limited_pupil_diameter_mm"]), 3.0)
    assert np.isclose(float(payload["diffraction_limited_f_number"]), 8.0 / 3.0)
    assert np.isclose(float(payload["defocus_diopters"]), 1.5)
    assert np.isclose(float(payload["explicit_defocus_zcoeff"]), 1.5)
    assert np.isclose(float(payload["oi_method_defocus_zcoeff"]), 1.5)
    assert payload["diffraction_limited_psf_x_um"].shape == payload["diffraction_limited_psf_center_row_550_norm"].shape
    assert payload["explicit_defocus_psf_x_um"].shape == payload["explicit_defocus_psf_center_row_550_norm"].shape
    assert payload["oi_method_defocus_psf_x_um"].shape == payload["oi_method_defocus_psf_center_row_550_norm"].shape
    assert payload["explicit_defocus_oi_center_row_550_norm"].shape == payload["oi_method_defocus_oi_center_row_550_norm"].shape
    assert float(payload["explicit_vs_oi_method_psf_center_row_550_normalized_mae"]) < 1e-8
    assert float(payload["explicit_vs_oi_method_oi_center_row_550_normalized_mae"]) < 1e-8


def test_optics_rt_synthetic_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_rt_synthetic_small", asset_store=asset_store)

    assert np.array_equal(payload["scene_wave"], np.array([550.0, 650.0], dtype=float))
    assert np.isclose(float(payload["scene_fov_deg"]), 3.0)
    assert np.array_equal(payload["spread_limits"], np.array([1.0, 3.0], dtype=float))
    assert np.isclose(float(payload["xy_ratio"]), 1.6)
    assert payload["raytrace_field_height_mm"].shape == (21,)
    assert np.array_equal(payload["raytrace_wave"], np.array([450.0, 550.0, 650.0], dtype=float))
    assert payload["geometry_550"].shape == (21,)
    assert payload["relative_illumination_550"].shape == (21,)
    assert np.isclose(float(payload["center_psf_sum_550"]), 1.0)
    assert np.isclose(float(payload["edge_psf_sum_550"]), 1.0)
    assert payload["center_psf_mid_row_550_norm"].shape == (128,)
    assert payload["edge_psf_mid_row_550_norm"].shape == (128,)
    assert np.array_equal(payload["oi_wave"], np.array([550.0, 650.0], dtype=float))
    assert np.array_equal(payload["oi_photons_shape"], np.array([312.0, 312.0, 2.0], dtype=float))
    assert payload["oi_mean_photons_by_wave"].shape == (2,)
    assert payload["oi_p95_photons_by_wave"].shape == (2,)
    assert payload["oi_max_photons_by_wave"].shape == (2,)
    assert payload["oi_center_row_550_norm"].shape == (129,)


def test_optics_rt_gridlines_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_rt_gridlines_small", asset_store=asset_store)

    assert np.array_equal(payload["scene_wave"], np.array([550.0, 650.0], dtype=float))
    assert np.isclose(float(payload["requested_scene_hfov_deg"]), 45.0)
    assert float(payload["adjusted_scene_hfov_deg"]) < 45.0
    assert float(payload["raytrace_fov_deg"]) > 0.0
    assert float(payload["raytrace_f_number"]) > 0.0
    assert float(payload["raytrace_effective_focal_length_mm"]) > 0.0
    assert np.array_equal(payload["geometry_only_size"], np.array([384, 384], dtype=int))
    assert payload["geometry_center_row_550_norm"].shape == (129,)
    assert payload["psf_struct_sample_angles"].shape == (19,)
    assert payload["psf_struct_img_height_mm"].shape == (6,)
    assert np.array_equal(payload["psf_struct_wavelength"], np.array([550.0, 650.0], dtype=float))
    assert np.array_equal(payload["stepwise_rt_size"], np.array([436, 436], dtype=int))
    assert np.array_equal(payload["stepwise_rt_center_row_550_widths"], np.array([82, 82, 95], dtype=int))
    assert np.array_equal(payload["automated_rt_size"], np.array([436, 436], dtype=int))
    assert np.array_equal(payload["automated_rt_center_row_550_widths"], np.array([95, 95], dtype=int))
    assert np.array_equal(payload["diffraction_large_size"], np.array([480, 480], dtype=int))
    assert np.array_equal(payload["diffraction_large_center_row_550_widths"], np.array([78, 91, 91], dtype=int))
    assert np.isclose(float(payload["small_scene_fov_deg"]), 20.0)
    assert np.array_equal(payload["rt_small_size"], np.array([484, 484], dtype=int))
    assert payload["rt_small_center_row_550_norm"].shape == (129,)
    assert np.array_equal(payload["rt_small_center_row_550_widths"], np.array([89, 89, 91], dtype=int))
    assert np.array_equal(payload["dl_small_size"], np.array([480, 480], dtype=int))
    assert np.array_equal(payload["dl_small_center_row_550_widths"], np.array([78, 91, 93], dtype=int))


def test_optics_rt_psf_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_rt_psf_small", asset_store=asset_store)

    assert np.array_equal(payload["scene_wave"], np.array([450.0, 550.0, 650.0], dtype=float))
    assert np.isclose(float(payload["scene_fov_deg"]), 10.0)
    assert np.array_equal(payload["rt_size"], np.array([564, 564], dtype=int))
    assert np.isclose(float(payload["rt_f_number"]), 4.999973)
    assert payload["rt_optics_name"] == "Asphere 2mm"
    assert np.array_equal(payload["rt_psf_sample_angles_deg"], np.arange(0.0, 361.0, 10.0))
    assert payload["rt_psf_image_heights_mm"].shape == (6,)
    assert np.array_equal(payload["rt_psf_wavelength"], np.array([450.0, 550.0, 650.0], dtype=float))
    assert np.array_equal(payload["rt_sampled_psf_shape"], np.array([37, 6, 3], dtype=int))
    assert payload["rt_center_psf_mid_row_550_norm"].shape == (129,)
    assert payload["rt_edge_psf_mid_row_550_norm"].shape == (129,)
    assert payload["rt_mean_photons_by_wave"].shape == (3,)
    assert payload["rt_max_photons_by_wave"].shape == (3,)
    assert np.array_equal(payload["rt_center_row_550_widths"], np.array([111, 117, 119], dtype=int))
    assert np.array_equal(payload["dl_size"], np.array([640, 640], dtype=int))
    assert np.isclose(float(payload["dl_f_number"]), 3.9999784)
    assert payload["dl_mean_photons_by_wave"].shape == (3,)
    assert payload["dl_max_photons_by_wave"].shape == (3,)
    assert np.array_equal(payload["dl_center_row_550_widths"], np.array([100, 111, 129], dtype=int))


def test_optics_rt_psf_view_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_rt_psf_view_small", asset_store=asset_store)

    assert np.array_equal(payload["scene_wave"], np.array([550.0, 650.0], dtype=float))
    assert np.isclose(float(payload["scene_fov_deg"]), 4.0)
    assert np.array_equal(payload["oi_size"], np.array([448, 448], dtype=int))
    assert np.array_equal(payload["psf_sample_angles_deg"], np.arange(0.0, 361.0, 10.0))
    assert np.allclose(payload["psf_image_heights_mm"], np.array([0.0, 0.05, 0.1, 0.15], dtype=float))
    assert np.array_equal(payload["psf_wavelength"], np.array([550.0, 650.0], dtype=float))
    assert np.array_equal(payload["sampled_rt_psf_shape"], np.array([37, 4, 2], dtype=int))
    assert payload["field_height_psf_mid_rows_550_norm"].shape == (4, 129)
    assert np.array_equal(payload["field_height_psf_widths_10pct"], np.array([18, 22, 24, 28], dtype=int))
    assert payload["angle_sweep_edge_psf_mid_rows_550_norm"].shape == (37, 129)
    assert np.array_equal(
        np.unique(payload["angle_sweep_edge_psf_widths_10pct"]),
        np.array([28, 30, 31, 34, 37, 41, 45, 46, 50, 52], dtype=int),
    )
    assert payload["center_rtplot_psf_mid_row_550_norm"].shape == (129,)
    assert payload["edge_rtplot_psf_mid_row_550_norm"].shape == (129,)


def test_optics_defocus_scene_small_parity_case(asset_store) -> None:
    payload = run_python_case("optics_defocus_scene_small", asset_store=asset_store)

    assert payload["wave"].shape == (31,)
    assert float(payload["max_sf_cpd"]) > 0.0
    assert payload["sample_sf_cpd"].ndim == 1
    assert payload["sample_sf_mm"].shape == payload["sample_sf_cpd"].shape
    assert np.isclose(float(payload["defocus_5_diopters"]), 5.0)
    assert float(payload["defocus_40um_diopters"]) > float(payload["defocus_10um_diopters"]) > 0.0
    assert payload["focus_center_row_550_norm"].shape == payload["defocus5_center_row_550_norm"].shape
    assert payload["focus_center_row_550_norm"].shape == payload["miss10_center_row_550_norm"].shape
    assert float(payload["focus_peak_550"]) > float(payload["miss10_peak_550"]) > float(payload["miss40_peak_550"])


def test_si_synthetic_custom_loads_psf_mat_file(tmp_path, asset_store) -> None:
    psf = np.zeros((129, 129, 1), dtype=float)
    psf[64, 64, 0] = 1.0
    path = tmp_path / "custom_si_psf.mat"
    savemat(
        path,
        {
            "psf": psf,
            "wave": np.array([550.0], dtype=float),
            "umPerSamp": np.array([0.25, 0.25], dtype=float),
        },
    )

    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    optics = si_synthetic("custom", oi_create("shift invariant"), path)
    oi = oi_set(oi_create("shift invariant"), "optics", optics)
    computed = oi_compute(oi, scene, crop=True)

    stored = oi_get(oi, "psfdata")
    assert np.asarray(stored["psf"]).shape == (129, 129, 1)
    assert np.array_equal(np.asarray(stored["wave"]), np.array([550.0], dtype=float))
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]


def test_ie_save_si_data_file_roundtrips_into_si_synthetic_custom(tmp_path, asset_store) -> None:
    samples = np.arange(129, dtype=float) - 64.0
    xx, yy = np.meshgrid(samples, samples, indexing="xy")
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    psf = np.zeros((129, 129, wave.size), dtype=float)
    for idx, wavelength in enumerate(wave):
        sigma = 1.2 + 0.01 * ((wavelength - wave[0]) / 10.0)
        plane = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        psf[:, :, idx] = plane / np.sum(plane)

    path = ie_save_si_data_file(psf, wave, np.array([0.25, 0.25], dtype=float), tmp_path / "custom_si_psf")
    scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.0)
    oi = oi_create("shift invariant")
    optics = si_synthetic("custom", oi, path)
    oi = oi_set(oi, "optics", optics)
    computed = oi_compute(oi, scene, crop=True)

    stored = oi_get(oi, "psfdata")
    assert np.asarray(stored["psf"]).shape == psf.shape
    assert np.array_equal(np.asarray(stored["wave"]), wave)
    assert computed.data["photons"].shape[:2] == scene.data["photons"].shape[:2]

def test_oi_compute_wvf_uses_custom_aperture(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)
    aperture = np.ones((9, 9), dtype=float)
    aperture[:, :4] = 0.0
    custom_oi = oi_compute(oi_create("wvf"), scene, crop=True, aperture=aperture)

    assert custom_oi.data["photons"].shape == default_oi.data["photons"].shape
    assert not np.allclose(custom_oi.data["photons"], default_oi.data["photons"])


def test_oi_compute_wvf_default_aperture_matches_full_open_aperture(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)
    full_aperture = np.ones((9, 9), dtype=float)
    explicit_oi = oi_compute(oi_create("wvf"), scene, crop=True, aperture=full_aperture)

    assert np.allclose(explicit_oi.data["photons"], default_oi.data["photons"])


def test_oi_compute_wvf_zcoeffs_change_wavefront_response(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)

    custom_wvf = wvf_create(
        wave=scene.fields["wave"],
        focal_length_m=0.003862755099228,
        f_number=4.0,
        calc_pupil_diameter_mm=9.6569e-01,
        zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.25], dtype=float),
    )
    custom_oi = oi_compute(oi_create("wvf", custom_wvf), scene, crop=True)

    row = custom_oi.data["photons"].shape[0] // 2
    dark_col = 8
    band = 0
    dark_default = float(default_oi.data["photons"][row, dark_col, band])
    dark_custom = float(custom_oi.data["photons"][row, dark_col, band])

    assert dark_custom > dark_default


def test_oi_compute_wvf_sce_changes_wavefront_response(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    default_oi = oi_compute(oi_create("wvf"), scene, crop=True)

    custom_wvf = wvf_create(
        wave=scene.fields["wave"],
        focal_length_m=0.003862755099228,
        f_number=4.0,
        calc_pupil_diameter_mm=9.6569e-01,
        compute_sce=True,
        sce_params={
            "wave": scene.fields["wave"],
            "rho": np.full(scene.fields["wave"].shape, 200.0, dtype=float),
            "xo_mm": 0.0,
            "yo_mm": 0.0,
        },
    )
    custom_oi = oi_compute(oi_create("wvf", custom_wvf), scene, crop=True)

    assert custom_oi.data["photons"].shape == default_oi.data["photons"].shape
    assert not np.allclose(custom_oi.data["photons"], default_oi.data["photons"])


def test_sensor_compute_noiseless(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0)
    sensor = sensor_compute(sensor, oi, seed=0)
    assert sensor.data["volts"].shape == sensor.fields["size"]
    assert np.all(sensor.data["volts"] >= 0.0)


def test_sensor_compute_noiseless_auto_exposure_matches_regression(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0)
    sensor = sensor_compute(sensor, oi, seed=0)
    assert np.isclose(sensor.fields["integration_time"], 0.05778050422668457, rtol=1e-5, atol=1e-8)


def test_sensor_set_integration_time_disables_auto_exposure(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "autoexposure", True)
    sensor = sensor_set(sensor, "integrationtime", 0.125)
    assert sensor.fields["auto_exposure"] is False
    assert np.isclose(sensor.fields["integration_time"], 0.125)


def test_sensor_get_set_supports_n_samples_per_pixel(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "nsamplesperpixel", 3)

    assert sensor_get(sensor, "nsamplesperpixel") == 3


def test_sensor_get_reports_matlab_style_geometry_and_cfa_metadata(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    rows, cols = sensor.fields["size"]

    support = sensor_get(sensor, "spatialsupport", "um")
    cfa = sensor_get(sensor, "cfa")
    pattern = sensor_get(sensor, "pattern")
    cfa_config = sensor_get(sensor, "unitblockconfig")
    pattern_colors = sensor_get(sensor, "patterncolors")

    assert sensor_get(sensor, "rows") == rows
    assert sensor_get(sensor, "cols") == cols
    assert sensor_get(sensor, "size") == (rows, cols)
    assert np.isclose(sensor_get(sensor, "arrayheight"), rows * pixel_size[0])
    assert np.isclose(sensor_get(sensor, "arraywidth", "mm"), cols * pixel_size[1] * 1e3)
    assert np.allclose(sensor_get(sensor, "dimension", "um"), np.array([rows * pixel_size[0], cols * pixel_size[1]]) * 1e6)
    assert np.isclose(sensor_get(sensor, "wspatialresolution", "um"), pixel_size[1] * 1e6)
    assert np.isclose(sensor_get(sensor, "hspatialresolution"), pixel_size[0])
    assert np.isclose(sensor_get(sensor, "deltax", "um"), pixel_size[1] * 1e6)
    assert np.isclose(sensor_get(sensor, "deltay"), pixel_size[0])
    assert support["x"].shape == (cols,)
    assert support["y"].shape == (rows,)
    assert np.isclose(support["x"][0], -support["x"][-1])
    assert np.isclose(support["y"][0], -support["y"][-1])
    assert sensor_get(sensor, "unitblockrows") == 2
    assert sensor_get(sensor, "unitblockcols") == 2
    assert sensor_get(sensor, "cfasize") == (2, 2)
    assert sensor_get(sensor, "cfaname") == "Bayer RGB"
    assert sensor_get(sensor, "filtercolorletters") == "rgb"
    assert np.array_equal(pattern, np.array([[2, 1], [3, 2]], dtype=int))
    assert np.array_equal(cfa["pattern"], pattern)
    assert cfa["unitBlock"]["rows"] == 2
    assert cfa["unitBlock"]["cols"] == 2
    assert np.allclose(cfa["unitBlock"]["config"], cfa_config)
    assert np.allclose(cfa_config, np.array([[0.0, 0.0], [pixel_size[1], 0.0], [0.0, pixel_size[0]], [pixel_size[1], pixel_size[0]]], dtype=float))
    assert pattern_colors.shape == (2, 2)
    assert np.array_equal(pattern_colors, np.array([["g", "r"], ["b", "g"]], dtype="<U1"))


def test_sensor_cfa_name_list_matches_legacy_popup_options() -> None:
    assert sensorCFANameList() == ["Bayer RGB", "Bayer CMY", "RGBW", "Monochrome", "Other"]


def test_sensor_pixel_coord_matches_even_and_odd_geometry(asset_store) -> None:
    sensor_even = sensor_set(sensor_create(asset_store=asset_store), "rows", 4)
    sensor_even = sensor_set(sensor_even, "cols", 6)
    pitch_x_even = float(sensor_get(sensor_even, "deltax"))
    pitch_y_even = float(sensor_get(sensor_even, "deltay"))
    full_x_even, full_y_even = sensorPixelCoord(sensor_even, "full")
    upper_x_even, upper_y_even = sensorPixelCoord(sensor_even, "upper-right")

    assert np.allclose(full_x_even, pitch_x_even * np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=float))
    assert np.allclose(full_y_even, pitch_y_even * np.array([-1.5, -0.5, 0.5, 1.5], dtype=float))
    assert np.allclose(upper_x_even, pitch_x_even * np.array([0.5, 1.5, 2.5], dtype=float))
    assert np.allclose(upper_y_even, pitch_y_even * np.array([0.5, 1.5], dtype=float))

    sensor_odd = sensor_set(sensor_create("monochrome", asset_store=asset_store), "rows", 5)
    sensor_odd = sensor_set(sensor_odd, "cols", 7)
    pitch_x_odd = float(sensor_get(sensor_odd, "deltax"))
    pitch_y_odd = float(sensor_get(sensor_odd, "deltay"))
    full_x_odd, full_y_odd = sensorPixelCoord(sensor_odd, "full")
    upper_x_odd, upper_y_odd = sensorPixelCoord(sensor_odd, "upper-right")

    assert np.allclose(full_x_odd, pitch_x_odd * np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(full_y_odd, pitch_y_odd * np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float))
    assert np.allclose(upper_x_odd, pitch_x_odd * np.array([0.0, 1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(upper_y_odd, pitch_y_odd * np.array([0.0, 1.0, 2.0], dtype=float))


def test_sensor_get_set_supports_matlab_style_spectrum_metadata(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", np.array([400.0, 500.0, 600.0], dtype=float))
    sensor = sensor_set(sensor, "filterspectra", np.array([[0.0], [1.0], [0.0]], dtype=float))
    sensor = sensor_set(sensor, "pixelspectralqe", np.array([0.2, 0.6, 1.0], dtype=float))
    sensor = sensor_set(sensor, "irfilter", np.array([1.0, 0.5, 0.0], dtype=float))

    spectrum = sensor_get(sensor, "sensorspectrum")
    assert np.array_equal(spectrum["wave"], np.array([400.0, 500.0, 600.0], dtype=float))
    assert sensor_get(sensor, "wavelengthresolution") == 100.0
    assert sensor_get(sensor, "nwaves") == 3

    sensor = sensor_set(sensor, "sensorspectrum", {"wave": np.array([450.0, 550.0], dtype=float), "comment": "test spectrum"})

    assert np.array_equal(sensor_get(sensor, "wavelength"), np.array([450.0, 550.0], dtype=float))
    assert sensor_get(sensor, "binwidth") == 100.0
    assert sensor_get(sensor, "numberofwavelengthsamples") == 2
    assert np.allclose(sensor_get(sensor, "filterspectra"), np.array([[0.5], [0.5]], dtype=float))
    assert np.allclose(sensor_get(sensor, "colorfilters"), np.array([[0.5], [0.5]], dtype=float))
    assert np.allclose(sensor_get(sensor, "pixelspectralqe"), np.array([0.4, 0.8], dtype=float))
    assert np.allclose(sensor_get(sensor, "pixelqe"), np.array([0.4, 0.8], dtype=float))
    assert np.allclose(sensor_get(sensor, "infraredfilter"), np.array([0.75, 0.25], dtype=float))
    assert np.allclose(sensor_get(sensor, "irfilter"), np.array([0.75, 0.25], dtype=float))
    assert np.allclose(sensor_get(sensor, "spectralqe"), np.array([[0.15], [0.1]], dtype=float))
    assert np.allclose(sensor_get(sensor, "sensorspectralsr"), sensor_get(sensor, "sensor spectral sr"))
    assert sensor_get(sensor, "sensorspectrum")["comment"] == "test spectrum"


def test_sensor_get_set_supports_raw_color_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", np.array([500.0, 600.0], dtype=float))
    color = {
        "filterSpectra": np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.8]], dtype=float),
        "filterNames": ["red", "green", "blue"],
        "irFilter": np.array([0.75, 0.25], dtype=float),
    }

    sensor = sensor_set(sensor, "color", color)

    exported = sensor_get(sensor, "color")
    assert np.allclose(exported["filterSpectra"], color["filterSpectra"])
    assert exported["filterNames"] == color["filterNames"]
    assert np.allclose(exported["irFilter"], color["irFilter"])
    assert sensor_get(sensor, "filternames") == ["red", "green", "blue"]
    assert sensor_get(sensor, "filternamescellarray") == ["r", "g", "b"]
    assert sensor_get(sensor, "filtercolornamescellarray") == ["r", "g", "b"]
    assert sensor_get(sensor, "filternamescell") == ["r", "g", "b"]


def test_sensor_get_set_supports_pixel_passthrough_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    assert sensor_get(sensor, "pixel name") == "aps"
    assert sensor_get(sensor, "pixel type") == "pixel"

    sensor = sensor_set(sensor, "pixel name", "custom-pixel")
    sensor = sensor_set(sensor, "pixel type", "custom-type")
    sensor = sensor_set(sensor, "fillfactor", 0.5)
    sensor = sensor_set(sensor, "voltsperelectron", 2.0e-4)
    sensor = sensor_set(sensor, "maxvoltage", 1.5)
    sensor = sensor_set(sensor, "darkvoltageperpixel", 2.0e-3)
    sensor = sensor_set(sensor, "readstandarddeviationvolts", 3.0e-3)
    sensor = sensor_set(sensor, "pixel width", 4.0e-6)
    sensor = sensor_set(sensor, "pixel height", 3.0e-6)
    sensor = sensor_set(sensor, "width between pixels", 0.5e-6)
    sensor = sensor_set(sensor, "height between pixels", 0.25e-6)
    sensor = sensor_set(sensor, "pixelspectralqe", np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))

    pixel_size = np.asarray(sensor_get(sensor, "pixelsize"), dtype=float)
    pd_area = float(sensor_get(sensor, "pdarea"))

    assert np.isclose(sensor_get(sensor, "fillfactor"), 0.5)
    assert sensor_get(sensor, "pixel name") == "custom-pixel"
    assert sensor_get(sensor, "pixel type") == "custom-type"
    assert np.allclose(pixel_size, np.array([3.25e-6, 4.5e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelwidth", "um"), 4.0)
    assert np.isclose(sensor_get(sensor, "pixelheight", "um"), 3.0)
    assert np.isclose(sensor_get(sensor, "pixelwidthmeters"), 4.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheightmeters"), 3.0e-6)
    assert np.isclose(sensor_get(sensor, "widthgap", "um"), 0.5)
    assert np.isclose(sensor_get(sensor, "heightgap", "um"), 0.25)
    assert np.isclose(sensor_get(sensor, "widthbetweenpixels", "um"), 0.5)
    assert np.isclose(sensor_get(sensor, "heightbetweenpixels", "um"), 0.25)
    assert np.allclose(sensor_get(sensor, "xyspacing", "um"), np.array([4.5, 3.25], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelarea"), np.prod(pixel_size))
    assert np.isclose(pd_area, (3.0e-6 * 4.0e-6) * 0.5)
    assert np.allclose(sensor_get(sensor, "pdsize", "um"), np.sqrt(0.5) * np.array([3.0, 4.0], dtype=float))
    assert np.allclose(sensor_get(sensor, "pddimension", "um"), np.sqrt(0.5) * np.array([4.0, 3.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "conversiongain"), 2.0e-4)
    assert np.isclose(sensor_get(sensor, "conversiongainvperelectron"), 2.0e-4)
    assert np.isclose(sensor_get(sensor, "voltsperelectron"), 2.0e-4)
    assert np.isclose(sensor_get(sensor, "voltageswing"), 1.5)
    assert np.isclose(sensor_get(sensor, "maxvoltage"), 1.5)
    assert np.isclose(sensor_get(sensor, "wellcapacity"), 1.5 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "darkvoltage"), 2.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvoltageperpixel"), 2.0e-3)
    assert np.isclose(sensor_get(sensor, "darkelectrons"), 2.0e-3 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "darkcurrent"), sensor_get(sensor, "darkelectrons") * 1.602177e-19)
    assert np.isclose(sensor_get(sensor, "darkcurrentperpixel"), sensor_get(sensor, "darkcurrent"))
    assert np.isclose(sensor_get(sensor, "darkcurrentdensity"), sensor_get(sensor, "darkcurrent") / pd_area)
    assert np.isclose(sensor_get(sensor, "readnoisevolts"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "readnoiseelectrons"), 3.0e-3 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationvolts"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationelectrons"), 3.0e-3 / 2.0e-4)
    assert np.isclose(sensor_get(sensor, "readnoisemillivolts"), 3.0)
    assert np.allclose(sensor_get(sensor, "pdspectralqe"), sensor_get(sensor, "pixelspectralqe"))
    assert np.allclose(sensor_get(sensor, "pixelqe"), sensor_get(sensor, "pixelspectralqe"))
    assert np.allclose(sensor_get(sensor, "pdspectralsr"), sensor_get(sensor, "pixelspectralsr"))
    assert np.allclose(sensor_get(sensor, "spectralsr"), sensor_get(sensor, "pixelspectralsr"))
    assert np.allclose(sensor_get(sensor, "sr"), sensor_get(sensor, "pixelspectralsr"))

    sensor = sensor_set(sensor, "read noise", 10.0)
    assert np.isclose(sensor_get(sensor, "readnoisevolts"), 10.0 * 2.0e-4)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationelectrons"), 10.0)

    sensor = sensor_set(sensor, "readnoisemillivolts", 7.0)
    assert np.isclose(sensor_get(sensor, "readnoisevolts"), 7.0e-3)

    sensor = sensor_set(sensor, "conversiongainvperelectron", 3.0e-4)
    assert np.isclose(sensor_get(sensor, "conversiongain"), 3.0e-4)

    sensor = sensor_set(sensor, "pdwidth", 2.0e-6)
    sensor = sensor_set(sensor, "pdheight", 1.5e-6)

    assert np.isclose(sensor_get(sensor, "pdwidth", "um"), 2.0)
    assert np.isclose(sensor_get(sensor, "pdheight", "um"), 1.5)
    assert np.isclose(sensor_get(sensor, "fillfactor"), (2.0e-6 * 1.5e-6) / (4.0e-6 * 3.0e-6))

    replacement_pixel = dict(sensor_get(sensor, "pixel"))
    replacement_pixel["fill_factor"] = 0.25
    replacement_pixel["conversion_gain_v_per_electron"] = 1.0e-4
    replacement_pixel["name"] = "replacement-pixel"
    replacement_pixel["type"] = "replacement-type"
    sensor = sensor_set(sensor, "pixel", replacement_pixel)

    assert np.isclose(sensor_get(sensor, "fillfactor"), 0.25)
    assert np.isclose(sensor_get(sensor, "conversiongain"), 1.0e-4)
    assert sensor_get(sensor, "pixel name") == "replacement-pixel"
    assert sensor_get(sensor, "pixel type") == "replacement-type"


def test_sensor_get_set_supports_pixel_optical_and_spectral_metadata(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 0.01)
    sensor = sensor_set(sensor, "layerthicknesses", np.array([1.0e-6, 2.0e-6, 0.5e-6], dtype=float))
    sensor = sensor_set(sensor, "refractiveindices", np.array([1.0, 1.5, 3.4], dtype=float))
    sensor = sensor_set(sensor, "pixelspectrum", {"wave": np.array([450.0, 550.0, 650.0], dtype=float), "comment": "pixel spectrum"})
    sensor = sensor_set(sensor, "quantum efficiency", np.array([0.1, 0.2, 0.3], dtype=float))
    sensor = sensor_set(sensor, "darkvoltageperpixelpersec", 2.0e-3)
    sensor = sensor_set(sensor, "readnoisestdvolts", 1.0e-3)
    sensor = sensor_set(sensor, "voltage swing", 1.2)

    assert np.allclose(sensor_get(sensor, "layerthicknesses", "um"), np.array([1.0, 2.0, 0.5], dtype=float))
    assert np.isclose(sensor_get(sensor, "stackheight", "um"), 3.5)
    assert np.isclose(sensor_get(sensor, "pixeldepth", "um"), 3.5)
    assert np.isclose(sensor_get(sensor, "pixeldepthmeters"), 3.5e-6)
    assert np.allclose(sensor_get(sensor, "refractiveindices"), np.array([1.0, 1.5, 3.4], dtype=float))
    assert np.allclose(sensor_get(sensor, "refractiveindex"), np.array([1.0, 1.5, 3.4], dtype=float))
    assert np.allclose(sensor_get(sensor, "n"), np.array([1.0, 1.5, 3.4], dtype=float))
    assert np.array_equal(sensor_get(sensor, "pixelwavelength"), np.array([450.0, 550.0, 650.0], dtype=float))
    assert sensor_get(sensor, "pixelbinwidth") == 100.0
    assert sensor_get(sensor, "pixelnwave") == 3
    assert sensor_get(sensor, "pixelspectrum")["comment"] == "pixel spectrum"
    assert np.allclose(sensor_get(sensor, "quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(sensor_get(sensor, "pixel quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(sensor_get(sensor, "photodetector quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))
    assert np.allclose(sensor_get(sensor, "photodetector spectral quantum efficiency"), np.array([0.1, 0.2, 0.3], dtype=float))

    expected_pixel_dr = 20.0 * np.log10((1.2 - 2.0e-3 * 0.01) / np.sqrt((2.0e-3 * 0.01) + (1.0e-3**2)))
    assert np.isclose(sensor_get(sensor, "pixeldynamicrange"), expected_pixel_dr)
    assert np.isclose(pixelDR(sensor), expected_pixel_dr)
    assert np.isclose(pixelDR(sensor, []), expected_pixel_dr)

    sensor = sensor_set(sensor, "n", np.array([1.0, 2.0, 3.5], dtype=float))
    assert np.allclose(sensor_get(sensor, "refractive indices"), np.array([1.0, 2.0, 3.5], dtype=float))

    sensor = sensor_set(sensor, "conversiongainvpelectron", 2.5e-6)
    assert np.isclose(sensor_get(sensor, "conversiongainvpelectron"), 2.5e-6)
    assert np.isclose(sensor_get(sensor, "conversion gain"), 2.5e-6)

    sensor = sensor_set(sensor, "vswing", 1.3)
    assert np.isclose(sensor_get(sensor, "vswing"), 1.3)
    assert np.isclose(sensor_get(sensor, "max voltage"), 1.3)

    sensor = sensor_set(sensor, "darkvolt", 3.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvolt"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvolts"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "darkvoltageperpixelpersec"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "voltspersecond"), 3.0e-3)
    assert np.isclose(sensor_get(sensor, "readnoisestdvolts"), 1.0e-3)
    assert np.isclose(sensor_get(sensor, "readstandarddeviationelectrons"), 1.0e-3 / 2.5e-6)


def test_sensor_get_set_supports_photodetector_position_passthrough(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel width", 4.0e-6)
    sensor = sensor_set(sensor, "pixel height", 3.0e-6)
    sensor = sensor_set(sensor, "photodetectorsize", np.array([1.0e-6, 2.0e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "pdposition", "um"), np.array([1.0, 1.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "pdxpos", "um"), 1.0)
    assert np.isclose(sensor_get(sensor, "pdypos", "um"), 1.0)
    assert np.allclose(sensor_get(sensor, "photodetectorsize", "um"), np.array([1.0, 2.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "photodetectorwidth", "um"), 2.0)
    assert np.isclose(sensor_get(sensor, "photodetectorheight", "um"), 1.0)

    sensor = sensor_set(sensor, "pdxpos", 0.5e-6)
    sensor = sensor_set(sensor, "pdypos", 0.75e-6)

    assert np.allclose(sensor_get(sensor, "pdposition", "um"), np.array([0.5, 0.75], dtype=float))
    assert np.isclose(sensor_get(sensor, "pdxpos", "um"), 0.5)
    assert np.isclose(sensor_get(sensor, "pdypos", "um"), 0.75)

    sensor = sensor_set(sensor, "photodetectorxposition", 0.25e-6)
    sensor = sensor_set(sensor, "photodetectoryposition", 0.5e-6)

    assert np.isclose(sensor_get(sensor, "photodetectorxposition", "um"), 0.25)
    assert np.isclose(sensor_get(sensor, "photodetectoryposition", "um"), 0.5)

    sensor = sensor_set(sensor, "pdposition", np.array([0.25e-6, 0.5e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "pdposition", "um"), np.array([0.25, 0.5], dtype=float))

    with pytest.raises(ValueError, match="photodetector position must keep the photodetector inside the pixel."):
        sensor_set(sensor, "pdposition", np.array([3.0e-6, 2.5e-6], dtype=float))


def test_sensor_set_supports_matlab_width_height_pair_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "widthandheight", np.array([4.0e-6, 3.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelwidth"), 4.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheight"), 3.0e-6)
    assert np.allclose(sensor_get(sensor, "pixelsize"), np.array([3.0e-6, 4.0e-6], dtype=float))

    sensor = sensor_set(sensor, "widthheight", 5.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelwidth"), 5.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheight"), 5.0e-6)

    sensor = sensor_set(sensor, "pdwidthandheight", np.array([2.0e-6, 1.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "photodetectorwidth"), 2.0e-6)
    assert np.isclose(sensor_get(sensor, "photodetectorheight"), 1.0e-6)
    assert np.allclose(sensor_get(sensor, "photodetectorsize"), np.array([1.0e-6, 2.0e-6], dtype=float))
    assert np.allclose(sensor_get(sensor, "pddimension"), np.array([2.0e-6, 1.0e-6], dtype=float))

    sensor = sensor_set(sensor, "pdwidthandheight", 1.5e-6)
    assert np.isclose(sensor_get(sensor, "photodetectorwidth"), 1.5e-6)
    assert np.isclose(sensor_get(sensor, "photodetectorheight"), 1.5e-6)
    assert np.allclose(sensor_get(sensor, "pddimension"), np.array([1.5e-6, 1.5e-6], dtype=float))


def test_sensor_set_routes_direct_unique_pixel_aliases_without_prefix(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "widthandheight", np.array([4.0e-6, 3.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pixelwidth"), 4.0e-6)
    assert np.isclose(sensor_get(sensor, "pixelheight"), 3.0e-6)

    sensor = sensor_set(sensor, "pdwidthandheight", np.array([2.0e-6, 1.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "pdwidth"), 2.0e-6)
    assert np.isclose(sensor_get(sensor, "pdheight"), 1.0e-6)
    assert np.allclose(sensor_get(sensor, "pddimension"), np.array([2.0e-6, 1.0e-6], dtype=float))

    initial_fill_factor = float(sensor_get(sensor, "fillfactor"))
    sensor = sensor_set(sensor, "sizeconstantfillfactor", np.array([8.0e-6, 6.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "fillfactor"), initial_fill_factor)

    sensor = sensor_set(sensor, "sizekeepfillfactor", np.array([10.0e-6, 8.0e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "fillfactor"), initial_fill_factor)

    sensor = sensor_set(sensor, "dark voltage per pixel per sec", 1.5e-3)
    assert np.isclose(sensor_get(sensor, "dark voltage"), 1.5e-3)


def test_sensor_set_pixel_size_same_fill_factor_scales_photodetector_geometry(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel width", 4.0e-6)
    sensor = sensor_set(sensor, "pixel height", 2.0e-6)
    sensor = sensor_set(sensor, "pd size", np.array([1.0e-6, 2.0e-6], dtype=float))
    sensor = sensor_set(sensor, "pd position", np.array([0.5e-6, 0.25e-6], dtype=float))

    sensor = sensor_set(sensor, "sizesamefillfactor", np.array([4.0e-6, 8.0e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "pixelsize"), np.array([4.0e-6, 8.0e-6], dtype=float))
    assert np.allclose(sensor_get(sensor, "photodetectorsize"), np.array([2.0e-6, 4.0e-6], dtype=float))
    assert np.allclose(sensor_get(sensor, "pdposition"), np.array([1.0e-6, 0.5e-6], dtype=float))
    assert np.isclose(sensor_get(sensor, "fillfactor"), 0.25)

    sensor = sensor_set(sensor, "pixelsize", np.array([8.0e-6, 16.0e-6], dtype=float))

    assert np.allclose(sensor_get(sensor, "photodetectorsize"), np.array([2.0e-6, 4.0e-6], dtype=float))
    assert not np.isclose(sensor_get(sensor, "fillfactor"), 0.25)


def test_sensor_get_set_supports_chart_and_metadata_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    corner_points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    rects = np.array([[10.0, 20.0, 5.0, 6.0]], dtype=float)
    current_rect = np.array([7.0, 8.0, 9.0, 10.0], dtype=float)
    rect_handles = ["r1", "r2"]

    sensor = sensor_set(sensor, "chartparameters", {"name": "Macbeth", "nSquares": 24})
    sensor = sensor_set(sensor, "cornerpoints", corner_points)
    sensor = sensor_set(sensor, "chartrects", rects)
    sensor = sensor_set(sensor, "currentrect", current_rect)
    sensor = sensor_set(sensor, "mccrecthandles", rect_handles)
    sensor = sensor_set(sensor, "metadatasensorname", "sensor-a")
    sensor = sensor_set(sensor, "metadatascenename", "scene-a")
    sensor = sensor_set(sensor, "metadataopticsname", "optics-a")
    sensor = sensor_set(sensor, "metadatacrop", np.array([1, 2, 3, 4], dtype=int))

    chart = sensor_get(sensor, "chartparameters")

    assert chart["name"] == "Macbeth"
    assert chart["nSquares"] == 24
    assert np.array_equal(chart["cornerPoints"], corner_points)
    assert np.array_equal(chart["rects"], rects)
    assert np.array_equal(chart["currentRect"], current_rect)
    assert np.array_equal(sensor_get(sensor, "cornerpoints"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chartcornerpoints"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chartcorners"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chart corners"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chart corner points"), corner_points)
    assert np.array_equal(sensor_get(sensor, "mcccornerpoints"), corner_points)
    assert np.array_equal(sensor_get(sensor, "chartrects"), rects)
    assert np.array_equal(sensor_get(sensor, "chartrectangles"), rects)
    assert np.array_equal(sensor_get(sensor, "chart rectangles"), rects)
    assert np.array_equal(sensor_get(sensor, "currentrect"), current_rect)
    assert np.array_equal(sensor_get(sensor, "chartcurrentrect"), current_rect)
    assert np.array_equal(sensor_get(sensor, "current rect"), current_rect)
    assert sensor_get(sensor, "mccrecthandles") == rect_handles
    assert sensor_get(sensor, "metadatasensorname") == "sensor-a"
    assert sensor_get(sensor, "metadatascenename") == "scene-a"
    assert sensor_get(sensor, "metadataopticsname") == "optics-a"
    assert np.array_equal(sensor_get(sensor, "metadatacrop"), np.array([1, 2, 3, 4], dtype=int))

    sensor = sensor_set(sensor, "mcccornerpoints", corner_points + 1.0)
    sensor = sensor_set(sensor, "chartrectangles", rects + 1.0)
    sensor = sensor_set(sensor, "chartcurrentrect", current_rect + 1.0)

    assert np.array_equal(sensor_get(sensor, "cornerpoints"), corner_points + 1.0)
    assert np.array_equal(sensor_get(sensor, "chart corner points"), corner_points + 1.0)
    assert np.array_equal(sensor_get(sensor, "mcccornerpoints"), corner_points + 1.0)
    assert np.array_equal(sensor_get(sensor, "chartrectangles"), rects + 1.0)
    assert np.array_equal(sensor_get(sensor, "chartcurrentrect"), current_rect + 1.0)


def test_sensor_get_set_supports_diffusion_mtf_storage(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    diffusion = {
        "name": "Gaussian",
        "otf": np.array([1.0, 0.8, 0.6], dtype=float),
        "support": np.array([0.0, 0.5, 1.0], dtype=float),
    }

    sensor = sensor_set(sensor, "diffusionmtf", diffusion)

    stored = sensor_get(sensor, "diffusionmtf")
    assert stored is not None
    assert stored["name"] == "Gaussian"
    assert np.array_equal(stored["otf"], diffusion["otf"])
    assert np.array_equal(stored["support"], diffusion["support"])

    stored["otf"][0] = 9.0
    assert np.array_equal(sensor_get(sensor, "diffusionmtf")["otf"], diffusion["otf"])

    sensor = sensor_set(sensor, "diffusionmtf", None)
    assert sensor_get(sensor, "diffusionmtf") is None


def test_sensor_get_set_supports_movement_metadata_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    positions = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    frames = np.array([2, 3], dtype=int)

    sensor = sensor_set(sensor, "sensormovement", {"name": "shake", "enabled": True})
    sensor = sensor_set(sensor, "movementpositions", positions)
    sensor = sensor_set(sensor, "framesperpositions", frames)

    movement = sensor_get(sensor, "sensormovement")

    assert movement["name"] == "shake"
    assert movement["enabled"] is True
    assert np.array_equal(movement["pos"], positions)
    assert np.array_equal(sensor_get(sensor, "movement positions"), positions)
    assert np.array_equal(sensor_get(sensor, "sensorpositions"), positions)
    assert np.array_equal(sensor_get(sensor, "sensorpositionsx"), positions[:, 0])
    assert np.array_equal(sensor_get(sensor, "sensorpositionsy"), positions[:, 1])
    assert np.array_equal(sensor_get(sensor, "framesperpositions"), frames)
    assert np.array_equal(sensor_get(sensor, "framesperposition"), frames)
    assert np.array_equal(sensor_get(sensor, "exposuretimesperposition"), frames)
    assert np.array_equal(sensor_get(sensor, "etimeperpos"), frames)


def test_sensor_get_set_supports_legacy_human_storage_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    human = {
        "name": "human",
        "coneType": np.array([[1, 2], [3, 4]], dtype=int),
        "densities": np.array([0.0, 0.6, 0.3, 0.1], dtype=float),
        "xy": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "rSeed": 17,
    }

    sensor = sensor_set(sensor, "human", human)

    stored = sensor_get(sensor, "human")
    assert stored is not None
    assert stored["name"] == "human"
    assert np.array_equal(sensor_get(sensor, "conetype"), human["coneType"])
    assert np.array_equal(sensor_get(sensor, "human cone densities"), human["densities"])
    assert np.array_equal(sensor_get(sensor, "conexy"), human["xy"])
    assert np.array_equal(sensor_get(sensor, "conelocs"), human["xy"])
    assert sensor_get(sensor, "humanrseed") == 17

    stored["coneType"][0, 0] = 9
    assert np.array_equal(sensor_get(sensor, "human")["coneType"], human["coneType"])

    cone_type = np.array([[4, 3], [2, 1]], dtype=int)
    densities = np.array([0.1, 0.5, 0.3, 0.1], dtype=float)
    xy = np.array([[0.5, 0.6]], dtype=float)
    sensor = sensor_set(sensor, "conetype", cone_type)
    sensor = sensor_set(sensor, "humanconedensities", densities)
    sensor = sensor_set(sensor, "conexy", xy)
    sensor = sensor_set(sensor, "humanrseed", 23)

    assert np.array_equal(sensor_get(sensor, "conetype"), cone_type)
    assert np.array_equal(sensor_get(sensor, "humanconetype"), cone_type)
    assert np.array_equal(sensor_get(sensor, "humanconedensities"), densities)
    assert np.array_equal(sensor_get(sensor, "humanconelocs"), xy)
    assert np.array_equal(sensor_get(sensor, "conexy"), xy)
    assert np.array_equal(sensor_get(sensor, "conelocs"), xy)
    assert sensor_get(sensor, "humanrseed") == 23


def test_sensor_human_legacy_wrappers_and_light_field(asset_store) -> None:
    base = sensor_create("default", asset_store=asset_store)
    params = {
        "sz": [6, 8],
        "rgbDensities": [0.1, 0.5, 0.3, 0.1],
        "coneAperture": [1.5e-6, 1.5e-6],
        "rSeed": 7,
        "wave": np.array([450.0, 550.0, 650.0], dtype=float),
    }
    human_sensor, xy, cone_type, r_seed, densities = sensorCreateConeMosaic(
        base,
        params["sz"],
        params["rgbDensities"],
        params["coneAperture"],
        params["rSeed"],
        asset_store=asset_store,
    )

    assert sensorCheckHuman(base) is False
    assert sensorCheckHuman(human_sensor) is True
    assert np.array_equal(np.asarray(sensor_get(human_sensor, "pattern"), dtype=int), cone_type)
    assert np.array_equal(np.asarray(sensor_get(human_sensor, "conetype"), dtype=int), cone_type)
    assert xy.shape == (48, 2)
    assert r_seed == 7
    assert np.isclose(float(np.sum(densities, dtype=float)), 1.0)
    assert sensor_get(human_sensor, "filter names")[0] == "kBlack"

    resized = sensorHumanResize(human_sensor, [1, 0], [0, 2])
    resized_pattern = np.asarray(sensor_get(resized, "pattern"), dtype=int)
    assert resized_pattern.shape == (7, 10)
    assert np.all(resized_pattern[0, :] == 1)
    assert np.all(resized_pattern[:, -2:] == 1)
    assert sensor_get(resized, "human")["xy"] is None

    dispatched = sensor_create("human", None, params, asset_store=asset_store)
    overloaded = sensor_create("human", params, asset_store=asset_store)
    explicit_pixel = pixelCreate("human", params["wave"])
    explicit = sensor_create("human", explicit_pixel, params, asset_store=asset_store)
    assert sensorCheckHuman(dispatched) is True
    assert sensorCheckHuman(overloaded) is True
    assert sensorCheckHuman(explicit) is True
    assert np.array_equal(np.asarray(sensor_get(dispatched, "wave"), dtype=float), params["wave"])
    assert np.array_equal(np.asarray(sensor_get(overloaded, "wave"), dtype=float), params["wave"])
    assert np.array_equal(np.asarray(sensor_get(explicit, "wave"), dtype=float), params["wave"])
    assert np.array_equal(np.asarray(sensor_get(dispatched, "pattern"), dtype=int), np.asarray(sensor_get(overloaded, "pattern"), dtype=int))
    assert np.array_equal(np.asarray(sensor_get(dispatched, "humanconetype"), dtype=int), np.asarray(sensor_get(overloaded, "humanconetype"), dtype=int))
    assert np.array_equal(np.asarray(sensor_get(dispatched, "humanconetype"), dtype=int), np.asarray(sensor_get(explicit, "humanconetype"), dtype=int))
    assert np.array_equal(np.asarray(sensor_get(dispatched, "humanconelocs"), dtype=float), np.asarray(sensor_get(overloaded, "humanconelocs"), dtype=float))
    assert np.array_equal(np.asarray(sensor_get(dispatched, "humanconelocs"), dtype=float), np.asarray(sensor_get(explicit, "humanconelocs"), dtype=float))
    assert np.allclose(np.asarray(sensor_get(dispatched, "humanconedensities"), dtype=float), np.asarray(sensor_get(overloaded, "humanconedensities"), dtype=float))
    assert np.allclose(np.asarray(sensor_get(dispatched, "humanconedensities"), dtype=float), np.asarray(sensor_get(explicit, "humanconedensities"), dtype=float))
    assert sensor_get(dispatched, "humanrseed") == params["rSeed"]
    assert sensor_get(overloaded, "humanrseed") == params["rSeed"]
    assert sensor_get(explicit, "humanrseed") == params["rSeed"]
    assert np.isclose(float(pixelGet(sensor_get(dispatched, "pixel"), "voltage swing")), 1.0)
    assert np.isclose(float(pixelGet(sensor_get(overloaded, "pixel"), "voltage swing")), 1.0)
    assert np.isclose(float(pixelGet(sensor_get(explicit, "pixel"), "voltage swing")), 1.0)
    assert np.isclose(float(sensor_get(dispatched, "integration time")), 1.0)
    assert np.isclose(float(sensor_get(overloaded, "integration time")), 1.0)
    assert np.isclose(float(sensor_get(explicit, "integration time")), 1.0)
    assert sensor_get(dispatched, "filter names")[0] == "kBlack"
    assert sensor_get(overloaded, "filter names")[0] == "kBlack"
    assert sensor_get(explicit, "filter names")[0] == "kBlack"


def test_sensor_human_empty_param_fields_replay_default_dispatch(asset_store) -> None:
    base = sensor_create("default", asset_store=asset_store)
    default_mosaic, default_xy, default_cone_type, default_seed, default_densities = sensorCreateConeMosaic(
        base,
        asset_store=asset_store,
    )
    placeholder_mosaic, placeholder_xy, placeholder_cone_type, placeholder_seed, placeholder_densities = sensorCreateConeMosaic(
        base,
        [],
        [],
        [],
        [],
        asset_store=asset_store,
    )

    np.testing.assert_array_equal(np.asarray(sensor_get(placeholder_mosaic, "wave"), dtype=float), np.asarray(sensor_get(default_mosaic, "wave"), dtype=float))
    np.testing.assert_array_equal(np.asarray(sensor_get(placeholder_mosaic, "pattern"), dtype=int), np.asarray(sensor_get(default_mosaic, "pattern"), dtype=int))
    np.testing.assert_array_equal(placeholder_xy, default_xy)
    np.testing.assert_array_equal(placeholder_cone_type, default_cone_type)
    np.testing.assert_allclose(placeholder_densities, default_densities, rtol=0.0, atol=0.0)
    assert placeholder_seed == default_seed

    default_sensor = sensor_create("human", asset_store=asset_store)
    placeholder_sensor = sensor_create(
        "human",
        {
            "sz": [],
            "rgbDensities": [],
            "coneAperture": [],
            "rSeed": [],
            "wave": [],
        },
        asset_store=asset_store,
    )

    np.testing.assert_array_equal(np.asarray(sensor_get(placeholder_sensor, "wave"), dtype=float), np.asarray(sensor_get(default_sensor, "wave"), dtype=float))
    np.testing.assert_array_equal(np.asarray(sensor_get(placeholder_sensor, "pattern"), dtype=int), np.asarray(sensor_get(default_sensor, "pattern"), dtype=int))
    np.testing.assert_array_equal(
        np.asarray(sensor_get(placeholder_sensor, "humanconelocs"), dtype=float),
        np.asarray(sensor_get(default_sensor, "humanconelocs"), dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(sensor_get(placeholder_sensor, "humanconedensities"), dtype=float),
        np.asarray(sensor_get(default_sensor, "humanconedensities"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert sensor_get(placeholder_sensor, "humanrseed") == sensor_get(default_sensor, "humanrseed")

    scene = scene_create("uniform d65", asset_store=asset_store)
    oi = oi_compute(oi_create(), scene)
    oi = oi_set(oi, "name", "lightfield-dispatch")
    lf_sensor = sensorLightField(oi, asset_store=asset_store)
    dispatched = sensor_create("light field", oi, asset_store=asset_store)
    overloaded = sensor_create("light field", {"size": (5, 7)}, oi, asset_store=asset_store)
    oi_size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
    sample_spacing = np.asarray(oi_get(oi, "sample spacing", "m"), dtype=float).reshape(-1)

    assert np.array_equal(np.asarray(sensor_get(lf_sensor, "size"), dtype=int), oi_size[:2])
    assert np.allclose(np.asarray(sensor_get(lf_sensor, "pixel size"), dtype=float), sample_spacing[0])
    assert sensor_get(lf_sensor, "name") == "lightfield"
    assert np.array_equal(np.asarray(sensor_get(dispatched, "size"), dtype=int), oi_size[:2])
    assert np.array_equal(np.asarray(sensor_get(overloaded, "size"), dtype=int), oi_size[:2])
    assert np.allclose(np.asarray(sensor_get(dispatched, "pixel size"), dtype=float), sample_spacing[0])
    assert np.allclose(np.asarray(sensor_get(overloaded, "pixel size"), dtype=float), sample_spacing[0])
    assert sensor_get(dispatched, "name") == "lightfield-dispatch"
    assert sensor_get(overloaded, "name") == "lightfield-dispatch"


def test_sensor_get_set_supports_legacy_scene_and_lens_metadata_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "scenename", "scene-b")
    sensor = sensor_set(sensor, "metadatalensname", "lens-b")
    sensor = sensor_set(sensor, "metadatasensorname", "sensor-b")
    sensor = sensor_set(sensor, "metadatacrop", {"rect": [1, 2, 3, 4]})

    assert sensor_get(sensor, "scene_name") == "scene-b"
    assert sensor_get(sensor, "scenename") == "scene-b"
    assert sensor_get(sensor, "metadatascenename") == "scene-b"
    assert sensor_get(sensor, "lens") == "lens-b"
    assert sensor_get(sensor, "metadatalensname") == "lens-b"
    assert sensor_get(sensor, "metadatalens") == "lens-b"
    assert sensor_get(sensor, "metadata optics name") == "lens-b"
    assert sensor_get(sensor, "metadatasensorname") == "sensor-b"
    assert sensor_get(sensor, "metadatacrop") == {"rect": [1, 2, 3, 4]}

    sensor = sensor_set(sensor, "metadatalens", "lens-c")

    assert sensor_get(sensor, "lens") == "lens-c"
    assert sensor_get(sensor, "metadatalensname") == "lens-c"
    assert sensor_get(sensor, "metadatalens") == "lens-c"


def test_sensor_get_set_supports_microlens_storage_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    microlens = {
        "name": "default",
        "type": "microlens",
        "offset": np.array([0.0, 1.0], dtype=float),
        "wavelength": 500.0,
    }
    ml_offset = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float)

    sensor = sensor_set(sensor, "microlens", microlens)
    sensor = sensor_set(sensor, "microlensoffset", ml_offset)

    stored = sensor_get(sensor, "microlens")

    assert stored["name"] == "default"
    assert stored["type"] == "microlens"
    assert np.array_equal(stored["offset"], np.array([0.0, 1.0], dtype=float))
    assert stored["wavelength"] == 500.0
    assert sensor_get(sensor, "ulens")["name"] == "default"
    assert np.array_equal(sensor_get(sensor, "microlensoffset"), ml_offset)
    assert np.array_equal(sensor_get(sensor, "microlensoffsetmicrons"), ml_offset)

    stored["offset"][0] = 9.0
    assert np.array_equal(sensor_get(sensor, "mlens")["offset"], np.array([0.0, 1.0], dtype=float))


def test_sensor_get_set_supports_column_fpn_storage_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    cols = int(sensor_get(sensor, "cols"))
    column_offset = np.linspace(-0.01, 0.01, cols, dtype=float)
    column_gain = np.linspace(0.9, 1.1, cols, dtype=float)

    assert np.array_equal(sensor_get(sensor, "columnfixedpatternnoise"), np.array([0.0, 0.0], dtype=float))
    assert np.array_equal(sensor_get(sensor, "colfpn"), np.array([0.0, 0.0], dtype=float))
    assert sensor_get(sensor, "columnfpnoffset") == 0.0
    assert sensor_get(sensor, "columnfpngain") == 0.0
    assert sensor_get(sensor, "coloffsetfpn") is None
    assert sensor_get(sensor, "colgainfpn") is None

    sensor = sensor_set(sensor, "columnfpnparameters", np.array([0.05, 0.1], dtype=float))
    assert np.array_equal(sensor_get(sensor, "columnfpn"), np.array([0.05, 0.1], dtype=float))

    sensor = sensor_set(sensor, "offsetnoisevalue", 0.015)
    sensor = sensor_set(sensor, "gainnoisevalue", 3.5)
    sensor = sensor_set(sensor, "columnfixedpatternnoise", np.array([0.125, 0.25], dtype=float))
    sensor = sensor_set(sensor, "coloffsetfpnvector", column_offset)
    sensor = sensor_set(sensor, "colgainfpnvector", column_gain)

    assert np.isclose(sensor_get(sensor, "dsnusigma"), 0.015)
    assert np.isclose(sensor_get(sensor, "dsnulevel"), 0.015)
    assert np.isclose(sensor_get(sensor, "sigmaoffsetfpn"), 0.015)
    assert np.isclose(sensor_get(sensor, "offsetfpn"), 0.015)
    assert np.isclose(sensor_get(sensor, "offsetsd"), 0.015)
    assert np.isclose(sensor_get(sensor, "offsetnoisevalue"), 0.015)
    assert np.isclose(sensor_get(sensor, "prnusigma"), 3.5)
    assert np.isclose(sensor_get(sensor, "prnulevel"), 3.5)
    assert np.isclose(sensor_get(sensor, "sigmagainfpn"), 3.5)
    assert np.isclose(sensor_get(sensor, "gainfpn"), 3.5)
    assert np.isclose(sensor_get(sensor, "gainsd"), 3.5)
    assert np.isclose(sensor_get(sensor, "gainnoisevalue"), 3.5)
    assert np.isclose(sensor_get(sensor, "sigmaprnu"), 3.5)
    assert np.allclose(sensor_get(sensor, "fpnparameters"), np.array([0.015, 3.5], dtype=float))
    assert np.allclose(sensor_get(sensor, "fpnoffsetgain"), np.array([0.015, 3.5], dtype=float))
    assert np.allclose(sensor_get(sensor, "fpnoffsetandgain"), np.array([0.015, 3.5], dtype=float))
    assert np.array_equal(sensor_get(sensor, "columnfixedpatternnoise"), np.array([0.125, 0.25], dtype=float))
    assert np.array_equal(sensor_get(sensor, "colfpn"), np.array([0.125, 0.25], dtype=float))
    assert sensor_get(sensor, "columnfpnoffset") == 0.125
    assert sensor_get(sensor, "columnfpngain") == 0.25
    assert sensor_get(sensor, "columndsnu") == 0.125
    assert sensor_get(sensor, "columnprnu") == 0.25
    assert np.array_equal(sensor_get(sensor, "coloffsetfpn"), column_offset)
    assert np.array_equal(sensor_get(sensor, "coloffsetfpnvector"), column_offset)
    assert np.array_equal(sensor_get(sensor, "coloffset"), column_offset)
    assert np.array_equal(sensor_get(sensor, "colgainfpn"), column_gain)
    assert np.array_equal(sensor_get(sensor, "colgainfpnvector"), column_gain)
    assert np.array_equal(sensor_get(sensor, "colgain"), column_gain)

    stored = sensor_get(sensor, "coloffsetfpn")
    assert stored is not None
    stored[0] = 9.0
    assert np.array_equal(sensor_get(sensor, "coloffsetfpn"), column_offset)

    with pytest.raises(ValueError, match="Column FPN"):
        sensor_set(sensor, "columnfixedpatternnoise", np.array([1.0, 2.0, 3.0], dtype=float))
    with pytest.raises(ValueError, match="Bad column offset data"):
        sensor_set(sensor, "coloffsetfpnvector", np.ones(cols - 1, dtype=float))
    with pytest.raises(ValueError, match="Bad column gain data"):
        sensor_set(sensor, "colgainfpnvector", np.ones(cols - 1, dtype=float))


def test_sensor_get_set_supports_fpn_image_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    rows, cols = sensor_get(sensor, "size")
    dsnu_image = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    prnu_image = np.linspace(0.9, 1.1, rows * cols, dtype=float).reshape(rows, cols)

    sensor = sensor_set(sensor, "dsnuimage", dsnu_image)
    sensor = sensor_set(sensor, "prnuimage", prnu_image)

    stored_dsnu = sensor_get(sensor, "dsnuimage")
    stored_prnu = sensor_get(sensor, "prnuimage")

    assert np.array_equal(stored_dsnu, dsnu_image)
    assert np.array_equal(sensor_get(sensor, "offsetfpnimage"), dsnu_image)
    assert np.array_equal(stored_prnu, prnu_image)
    assert np.array_equal(sensor_get(sensor, "gainfpnimage"), prnu_image)

    stored_dsnu[0, 0] = -1.0
    stored_prnu[0, 0] = -1.0
    assert np.array_equal(sensor_get(sensor, "dsnuimage"), dsnu_image)
    assert np.array_equal(sensor_get(sensor, "prnuimage"), prnu_image)

    sensor = sensor_set(sensor, "offsetfpnimage", None)
    sensor = sensor_set(sensor, "gainfpnimage", None)

    assert sensor_get(sensor, "dsnuimage") is None
    assert sensor_get(sensor, "prnuimage") is None


def test_sensor_get_set_supports_consistency_and_compute_method_storage(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "sensorconsistency") is False
    assert sensor_get(sensor, "sensorcomputemethod") is None

    sensor = sensor_set(sensor, "sensorconsistency", True)
    sensor = sensor_set(sensor, "sensorcomputemethod", {"name": "binning", "factor": 2})

    assert sensor_get(sensor, "sensorconsistency") is True
    assert sensor_get(sensor, "sensorcompute") == {"name": "binning", "factor": 2}
    assert sensor_get(sensor, "sensorcomputemethod") == {"name": "binning", "factor": 2}


def test_sensor_get_set_supports_exposure_plane_and_cds_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "nexposures") == 1
    assert sensor_get(sensor, "exposureplane") == 1
    assert sensor_get(sensor, "cds") is False

    sensor.fields["integration_time"] = np.array([0.01, 0.02, 0.03], dtype=float)
    sensor = sensor_set(sensor, "exposureplane", 3.2)
    sensor = sensor_set(sensor, "cds", True)
    sensor = sensor_set(sensor, "autoexp", "off")

    assert sensor_get(sensor, "nexposures") == 3
    assert sensor_get(sensor, "exposureplane") == 3
    assert sensor_get(sensor, "cds") is True
    assert sensor_get(sensor, "correlateddoublesampling") is True
    assert sensor_get(sensor, "autoexposure") is False

    sensor = sensor_set(sensor, "autoexp", "on")

    assert sensor_get(sensor, "automaticexposure") is True
    assert sensor_get(sensor, "integrationtime") == 0.0


def test_sensor_get_set_supports_exposure_method_and_time_summaries(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    sensor = sensor_set(sensor, "exposuretimes", np.array([0.01, 0.02, 0.04], dtype=float))

    assert np.array_equal(sensor_get(sensor, "exptime"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposuretimes"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposuretime", "ms"), np.array([10.0, 20.0, 40.0], dtype=float))
    assert np.array_equal(sensor_get(sensor, "expduration"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposureduration"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "exposuredurations"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "uniqueintegrationtimes"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "uniqueexptime"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.array_equal(sensor_get(sensor, "uniqueexptimes"), np.array([0.01, 0.02, 0.04], dtype=float))
    assert np.isclose(sensor_get(sensor, "centralexposure"), 0.02)
    assert np.isclose(sensor_get(sensor, "geometricmeanexposuretime"), 0.02)
    assert sensor_get(sensor, "expmethod") == "bracketedExposure"
    assert sensor_get(sensor, "nexposures") == 3

    sensor = sensor_set(sensor, "expmethod", "videoExposure")

    assert sensor_get(sensor, "expmethod") == "videoExposure"

    sensor = sensor_set(sensor, "integrationtime", np.array([[0.01, 0.02], [0.03, 0.04]], dtype=float))
    sensor = sensor_set(sensor, "automaticexposure", "on")

    assert sensor_get(sensor, "automaticexposure") is True
    assert np.array_equal(sensor_get(sensor, "integrationtime"), np.zeros((2, 2), dtype=float))
    assert sensor_get(sensor, "exposuremethod") == "videoExposure"


def test_sensor_compute_supports_multiple_integration_times(asset_store) -> None:
    scene = scene_create("uniform d65")
    oi = oi_compute(oi_create(), scene)
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "integrationtimes", np.array([0.01, 0.02, 0.04], dtype=float))
    sensor = sensor_set(sensor, "exposureplane", 2)

    computed = sensor_compute(sensor, oi)
    volts = np.asarray(sensor_get(computed, "volts"), dtype=float)

    assert volts.ndim == 3
    assert volts.shape[2] == 3
    assert sensor_get(computed, "ncaptures") == 3
    assert sensor_get(computed, "exposureplane") == 2
    assert np.array_equal(sensor_get(computed, "integrationtime"), np.array([0.01, 0.02, 0.04], dtype=float))
    mean_volts = np.mean(volts, axis=(0, 1))
    assert np.all(np.diff(mean_volts) > 0.0)


def test_sensor_compute_supports_cfa_exposure_duration_matrix(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_set(scene, "fov", 4.0)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)

    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "noise flag", 0)

    bluish = np.array([[0.04, 0.03], [0.30, 0.02]], dtype=float)
    sensor_b = sensor_compute(sensor_set(sensor.clone(), "exposure duration", bluish), oi)
    bluish_volt_images = np.asarray(sensor_get(sensor_b, "volt images"), dtype=float)
    bluish_means = np.nanmean(bluish_volt_images, axis=(0, 1))

    reddish = np.array([[0.04, 0.70], [0.03, 0.02]], dtype=float)
    sensor_r = sensor_compute(sensor_set(sensor.clone(), "exposure duration", reddish), oi)
    reddish_volt_images = np.asarray(sensor_get(sensor_r, "volt images"), dtype=float)
    reddish_means = np.nanmean(reddish_volt_images, axis=(0, 1))

    assert np.asarray(sensor_get(sensor_b, "integration time")).shape == (2, 2)
    assert sensor_get(sensor_b, "n captures") == 1
    assert not np.allclose(bluish_means, reddish_means)

    camera = camera_create(asset_store=asset_store)
    camera = camera_set(camera, "sensor noise flag", 0)
    camera = camera_set(camera, "sensor exposure duration", reddish)
    camera = camera_compute(camera, scene)
    camera_sensor = camera_get(camera, "sensor")

    assert np.asarray(sensor_get(camera_sensor, "integration time")).shape == (2, 2)
    camera_volt_images = np.asarray(sensor_get(camera_sensor, "volt images"), dtype=float)
    camera_means = np.nanmean(camera_volt_images, axis=(0, 1))
    assert np.all(camera_means > 0.0)
    assert np.allclose(camera_means, reddish_means, rtol=1.5e-1, atol=1e-8)


def test_sensor_get_set_supports_sampling_and_vignetting_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "pixelsamples") == 1
    assert sensor_get(sensor, "sensorbareetendue") == 0

    sensor = sensor_set(sensor, "spatialsamplesperpixel", 3)
    sensor = sensor_set(sensor, "vignetting", "bare")

    assert sensor_get(sensor, "ngridsamples") == 3
    assert sensor_get(sensor, "nsamplesperpixel") == 3
    assert sensor_get(sensor, "npixelsamplesforcomputing") == 3
    assert sensor_get(sensor, "pixelsamples") == 3
    assert sensor_get(sensor, "vignetting") == "bare"
    assert sensor_get(sensor, "sensorvignetting") == "bare"
    assert sensor_get(sensor, "vignettingflag") == "bare"
    assert sensor_get(sensor, "vignettingname") == "bare"
    assert sensor_get(sensor, "sensorbareetendue") == "bare"
    assert sensor_get(sensor, "nomicrolensetendue") == "bare"


def test_sensor_get_set_supports_noise_seed_reuse_and_response_type(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    assert sensor_get(sensor, "reusenoise") is False
    assert sensor_get(sensor, "noiseseed") == 0
    assert sensor_get(sensor, "responsetype") == "linear"

    sensor = sensor_set(sensor, "reusenoise", True)
    sensor = sensor_set(sensor, "noiseseed", 7)
    sensor = sensor_set(sensor, "responsetype", "LOG")

    assert sensor_get(sensor, "reusenoise") is True
    assert sensor_get(sensor, "noiseseed") == 7
    assert sensor_get(sensor, "responsetype") == "log"


def test_sensor_get_supports_response_and_dynamic_range_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "integrationtime", 0.05)
    sensor = sensor_set(sensor, "dsnusigma", 0.002)

    pixel = sensor_get(sensor, "pixel")
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])
    dsnu_sigma = float(pixel["dsnu_sigma_v"])
    voltage_swing = float(pixel["voltage_swing"])
    expected_noise_sd = np.sqrt(
        ((dark_voltage * 0.05) / conversion_gain) * (conversion_gain**2)
        + (read_noise**2)
        + (dsnu_sigma**2)
    )
    expected_dr = 10.0 * np.log10((voltage_swing - (dark_voltage * 0.05)) / expected_noise_sd)

    assert np.isclose(sensor_get(sensor, "dr"), expected_dr)
    assert np.isclose(sensor_get(sensor, "drdb20"), expected_dr)
    assert np.isclose(sensor_get(sensor, "dynamicrange"), expected_dr)
    assert np.isclose(sensor_get(sensor, "sensordynamicrange"), expected_dr)

    sensor = sensor_set(sensor, "integrationtime", 0.0)
    assert sensor_get(sensor, "sensordynamicrange") is None

    sensor = sensor_set(sensor, "volts", np.array([[0.0, 0.5], [0.25, 0.75]], dtype=float))
    assert np.isclose(sensor_get(sensor, "responsedr"), 0.75 / (1.0 / 4096.0))

    sensor = sensor_set(sensor, "noiseflag", 1)
    assert sensor_get(sensor, "noiseflag") == 1
    assert sensor_get(sensor, "shotnoiseflag") == 1


def test_sensor_get_set_supports_black_level_alias(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    nbits = int(sensor_get(sensor, "nbits"))

    sensor = sensor_set(sensor, "blacklevel", 64)

    assert sensor_get(sensor, "blacklevel") == 64.0
    assert sensor_get(sensor, "zerolevel") == 64.0
    assert sensor_get(sensor, "maxdigitalvalue") == float((2**nbits) - 64)

    sensor = sensor_set(sensor, "zerolevel", 32)

    assert sensor_get(sensor, "blacklevel") == 32.0
    assert sensor_get(sensor, "zerolevel") == 32.0


def test_sensor_get_set_supports_digital_value_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    dv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    sensor = sensor_set(sensor, "digitalvalue", dv)

    assert np.array_equal(sensor_get(sensor, "dv"), dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalue"), dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalues"), dv)

    sensor = sensor_set(sensor, "digitalvalues", dv + 1.0)

    assert np.array_equal(sensor_get(sensor, "digitalvalue"), dv + 1.0)
    assert np.array_equal(sensor_get(sensor, "digitalvalues"), dv + 1.0)


def test_sensor_get_supports_dv_or_volts_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    dv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    sensor = sensor_set(sensor, "volts", volts)
    assert np.array_equal(sensor_get(sensor, "dvorvolts"), volts)
    assert np.array_equal(sensor_get(sensor, "digitalorvolts"), volts)

    sensor = sensor_set(sensor, "dv", dv)
    assert np.array_equal(sensor_get(sensor, "dvorvolts"), dv)
    assert np.array_equal(sensor_get(sensor, "digitalorvolts"), dv)


def test_sensor_get_supports_response_ratio_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.array([[0.25, 0.5], [0.75, 1.0]], dtype=float))

    expected_volts_ratio = 1.0 / float(sensor.fields["pixel"]["voltage_swing"])
    assert np.isclose(sensor_get(sensor, "responseratio"), expected_volts_ratio)
    assert np.isclose(sensor_get(sensor, "volts2maxratio"), expected_volts_ratio)

    sensor = sensor_set(sensor, "digitalvalue", np.array([[64.0, 128.0], [256.0, 512.0]], dtype=float))
    sensor.data.pop("volts", None)
    expected_dv_ratio = 512.0 / float(2 ** int(sensor.fields["nbits"]))

    assert np.isclose(sensor_get(sensor, "responseratio"), expected_dv_ratio)
    assert np.isclose(sensor_get(sensor, "volts2maxratio"), expected_dv_ratio)


def test_sensor_get_supports_volt_images(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 16.0
    sensor = sensor_set(sensor, "volts", volts)

    plane_images = sensor_get(sensor, "voltimages")

    assert plane_images is not None
    assert plane_images.shape == (4, 4, 3)
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    tiled_pattern = tile_pattern(pattern, 4, 4)
    assert np.isnan(plane_images[0, 0, 0])
    assert np.isclose(plane_images[0, 1, 0], volts[0, 1])
    assert np.isclose(plane_images[0, 0, 1], volts[0, 0])
    assert np.isclose(plane_images[1, 0, 2], volts[1, 0])
    assert np.array_equal(~np.isnan(plane_images[:, :, 0]), tiled_pattern == 1)
    assert np.array_equal(~np.isnan(plane_images[:, :, 1]), tiled_pattern == 2)
    assert np.array_equal(~np.isnan(plane_images[:, :, 2]), tiled_pattern == 3)


def test_sensor_get_set_supports_voltage_electron_and_analog_aliases(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.full((2, 2), 0.25, dtype=float)

    sensor = sensor_set(sensor, "ag", 2.5)
    sensor = sensor_set(sensor, "ao", 0.05)
    sensor = sensor_set(sensor, "voltage", volts)

    assert np.isclose(sensor_get(sensor, "analoggain"), 2.5)
    assert np.isclose(sensor_get(sensor, "ag"), 2.5)
    assert np.isclose(sensor_get(sensor, "analogoffset"), 0.05)
    assert np.isclose(sensor_get(sensor, "ao"), 0.05)
    assert np.array_equal(sensor_get(sensor, "volts"), volts)
    assert np.array_equal(sensor_get(sensor, "voltage"), volts)
    assert np.array_equal(sensor_get(sensor, "electron"), sensor_get(sensor, "electrons"))


def test_sensor_get_supports_channel_select_for_sensor_data(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    sensor = sensor_set(sensor, "ag", 2.0)
    sensor = sensor_set(sensor, "ao", 0.1)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 10.0
    dv = np.arange(1, 17, dtype=float).reshape(4, 4)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dv", dv)

    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    tiled_pattern = tile_pattern(pattern, 4, 4)
    expected_volts = volts[tiled_pattern == 2]
    expected_dv = dv[tiled_pattern == 2]
    expected_electrons = np.asarray(sensor_get(sensor, "electrons"))[tiled_pattern == 2]

    assert np.array_equal(sensor_get(sensor, "volts", 2), expected_volts)
    assert np.array_equal(sensor_get(sensor, "voltage", 2), expected_volts)
    assert np.array_equal(sensor_get(sensor, "dv", 2), expected_dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalue", 2), expected_dv)
    assert np.array_equal(sensor_get(sensor, "digitalvalues", 2), expected_dv)
    assert np.array_equal(sensor_get(sensor, "electrons", 2), expected_electrons)
    assert np.array_equal(sensor_get(sensor, "electron", 2), expected_electrons)


def test_sensor_get_supports_electrons_per_area(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    sensor = sensor_set(sensor, "ag", 2.0)
    sensor = sensor_set(sensor, "ao", 0.1)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 10.0
    sensor = sensor_set(sensor, "volts", volts)

    pd_area_m2 = float(sensor_get(sensor, "pixel pd area"))
    electrons = np.asarray(sensor_get(sensor, "electrons"), dtype=float)
    expected_m2 = electrons / pd_area_m2
    expected_um2 = expected_m2 / 1e12
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    tiled_pattern = tile_pattern(pattern, 4, 4)

    assert np.isclose(sensor_get(sensor, "pixel pd area", "um"), pd_area_m2 * 1e12)
    assert np.array_equal(sensor_get(sensor, "electrons per area"), expected_m2)
    assert np.array_equal(sensor_get(sensor, "electrons per area", "um"), expected_um2)
    assert np.array_equal(sensor_get(sensor, "electrons per area", "um", 2), expected_um2[tiled_pattern == 2])


def test_sensor_get_set_supports_quantization_alias_surface(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    lut = np.array([0.0, 1.0, 2.0], dtype=float)

    sensor = sensor_set(sensor, "nbits", 12)
    sensor = sensor_set(sensor, "quantization", "12 bit")
    sensor = sensor_set(sensor, "quantizatonlut", lut)

    quantization = sensor_get(sensor, "quantization")
    quantization_method = sensor_get(sensor, "quantizationmethod")
    quantization_struct = sensor_get(sensor, "quantizationstructure")

    assert quantization == "12 bit"
    assert quantization_method == "12 bit"
    assert sensor_get(sensor, "nbits") == 12
    assert sensor_get(sensor, "bits") == 12
    assert np.array_equal(sensor_get(sensor, "quantizatonlut"), lut)
    assert np.array_equal(sensor_get(sensor, "quantizationlut"), lut)
    assert quantization_struct["bits"] == 12
    assert quantization_struct["method"] == "12 bit"
    assert np.array_equal(quantization_struct["lut"], lut)
    assert sensor_get(sensor, "maxdigital") == float((2**12) - sensor_get(sensor, "zero level"))
    assert sensor_get(sensor, "maxoutput") == sensor.fields["pixel"]["voltage_swing"]

    sensor = sensor_set(sensor, "quantizationstructure", {"bits": 8, "method": "8 bit", "lut": np.array([0.0, 0.5], dtype=float)})

    assert sensor_get(sensor, "quantizationmethod") == "8 bit"
    assert sensor_get(sensor, "nbits") == 8
    assert sensor_get(sensor, "bits") == 8
    assert np.array_equal(sensor_get(sensor, "lut"), np.array([0.0, 0.5], dtype=float))


def test_sensor_compute_uses_stored_noise_seed_when_seed_omitted(asset_store) -> None:
    scene = scene_create("uniform d65")
    oi = oi_compute(oi_create(), scene)

    sensor_a = sensor_set(sensor_create("monochrome", asset_store=asset_store), "noise flag", 2)
    sensor_a = sensor_set(sensor_a, "noise seed", 11)
    sensor_b = sensor_set(sensor_create("monochrome", asset_store=asset_store), "noise flag", 2)
    sensor_b = sensor_set(sensor_b, "noise seed", 11)
    sensor_c = sensor_set(sensor_create("monochrome", asset_store=asset_store), "noise flag", 2)
    sensor_c = sensor_set(sensor_c, "noise seed", 13)

    result_a = sensor_compute(sensor_a, oi)
    result_b = sensor_compute(sensor_b, oi)
    result_c = sensor_compute(sensor_c, oi)

    assert np.allclose(np.asarray(result_a.data["volts"], dtype=float), np.asarray(result_b.data["volts"], dtype=float))
    assert not np.allclose(np.asarray(result_a.data["volts"], dtype=float), np.asarray(result_c.data["volts"], dtype=float))


def test_sensor_set_cfa_round_trips_matlab_style_struct(asset_store) -> None:
    sensor = sensor_create("rgbw", asset_store=asset_store)
    cfa = sensor_get(sensor, "cfa")
    replacement = {
        "pattern": np.array([[4, 3], [2, 1]], dtype=int),
        "unitBlock": cfa["unitBlock"],
    }

    sensor = sensor_set(sensor, "cfa", replacement)

    assert np.array_equal(sensor_get(sensor, "pattern"), replacement["pattern"])
    assert np.array_equal(sensor_get(sensor, "cfa")["pattern"], replacement["pattern"])
    assert sensor_get(sensor, "cfaname") == "RGBW"


def test_sensor_create_rgbw_and_rccc_presets_expose_multichannel_cfas(asset_store) -> None:
    rgbw = sensor_create("rgbw", asset_store=asset_store)
    grbc = sensor_create("grbc", asset_store=asset_store)
    rccc = sensor_create("rccc", asset_store=asset_store)

    assert sensor_get(rgbw, "nfilters") == 4
    assert sensor_get(rgbw, "filtercolorletters") == "rgbw"
    assert sensor_get(rgbw, "filtercolorletterscell") == ["r", "g", "b", "w"]
    assert sensor_get(rgbw, "filterplotcolors") == "rgbk"
    assert np.array_equal(sensor_get(rgbw, "patterncolors"), np.array([["r", "g"], ["b", "w"]], dtype="<U1"))

    assert sensor_get(grbc, "nfilters") == 4
    assert sensor_get(grbc, "filtercolorletters") == "grbc"
    assert sensor_get(grbc, "filtercolorletterscell") == ["g", "r", "b", "c"]
    assert sensor_get(grbc, "filterplotcolors") == "grbc"
    assert np.array_equal(sensor_get(grbc, "patterncolors"), np.array([["g", "r"], ["b", "c"]], dtype="<U1"))
    assert sensor_get(grbc, "cfaname") == "Other"

    assert sensor_get(rccc, "nfilters") == 2
    assert sensor_get(rccc, "filtercolorletters") == "rw"
    assert sensor_get(rccc, "filtercolorletterscell") == ["r", "w"]
    assert sensor_get(rccc, "filterplotcolors") == "rk"
    assert np.array_equal(sensor_get(rccc, "patterncolors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))


def test_sensor_create_supports_parenthesized_bayer_aliases(asset_store) -> None:
    grbg = sensor_create("bayer(grbg)", asset_store=asset_store)
    rggb = sensor_create("bayer(rggb)", asset_store=asset_store)
    bggr = sensor_create("bayer(bggr)", asset_store=asset_store)

    assert np.array_equal(sensor_get(grbg, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert np.array_equal(sensor_get(rggb, "pattern"), np.array([[1, 2], [2, 3]], dtype=int))
    assert np.array_equal(sensor_get(bggr, "pattern"), np.array([[3, 2], [2, 1]], dtype=int))
    assert sensor_get(grbg, "filtercolorletters") == "rgb"
    assert sensor_get(rggb, "filtercolorletters") == "rgb"
    assert sensor_get(bggr, "filtercolorletters") == "rgb"


def test_sensor_create_vendor_models_load_upstream_rgbw_and_rccc_metadata(asset_store) -> None:
    mt9v024_rgbw = sensor_create("mt9v024", "rgbw", asset_store=asset_store)
    mt9v024_rccc = sensor_create("mt9v024", None, "rccc", asset_store=asset_store)
    ar0132at_rgbw = sensor_create("ar0132at", "rgbw", asset_store=asset_store)
    ar0132at_rccc = sensor_create("ar0132at", None, "rccc", asset_store=asset_store)

    assert mt9v024_rgbw.name == "MTV9V024-RGBW"
    assert mt9v024_rgbw.fields["size"] == (480, 752)
    assert np.allclose(mt9v024_rgbw.fields["pixel"]["size_m"], np.array([6e-6, 6e-6]))
    assert sensor_get(mt9v024_rgbw, "filtercolorletters") == "rgbw"
    assert np.array_equal(sensor_get(mt9v024_rgbw, "patterncolors"), np.array([["r", "g"], ["b", "w"]], dtype="<U1"))

    assert mt9v024_rccc.name == "MTV9V024-RCCC"
    assert sensor_get(mt9v024_rccc, "filtercolorletters") == "rw"
    assert np.array_equal(sensor_get(mt9v024_rccc, "patterncolors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))

    assert ar0132at_rgbw.name == "AR0132AT-RGBW"
    assert ar0132at_rgbw.fields["size"] == (960, 1280)
    assert np.allclose(ar0132at_rgbw.fields["pixel"]["size_m"], np.array([3.751e-6, 3.751e-6]))
    assert sensor_get(ar0132at_rgbw, "filtercolorletters") == "rgbw"
    assert np.array_equal(sensor_get(ar0132at_rgbw, "patterncolors"), np.array([["r", "g"], ["w", "b"]], dtype="<U1"))

    assert ar0132at_rccc.name == "AR0132AT-RCCC"
    assert sensor_get(ar0132at_rccc, "filtercolorletters") == "rw"
    assert np.array_equal(sensor_get(ar0132at_rccc, "patterncolors"), np.array([["w", "w"], ["w", "r"]], dtype="<U1"))


def test_sensor_model_wrappers_match_legacy_vendor_contracts(asset_store) -> None:
    mt9v024 = sensorMT9V024(None, "rgbw", asset_store=asset_store)
    direct_mt9v024 = sensor_create("mt9v024", "rgbw", asset_store=asset_store)
    list_mt9v024 = sensor_create("mt9v024", [], ["rgbw"], asset_store=asset_store)
    assert mt9v024.name == direct_mt9v024.name
    assert mt9v024.name == list_mt9v024.name
    assert mt9v024.fields["size"] == direct_mt9v024.fields["size"]
    assert mt9v024.fields["size"] == list_mt9v024.fields["size"]
    assert np.array_equal(sensor_get(mt9v024, "pattern"), sensor_get(direct_mt9v024, "pattern"))
    assert np.array_equal(sensor_get(mt9v024, "pattern"), sensor_get(list_mt9v024, "pattern"))

    direct_ar0132at = sensor_create("ar0132at", "rccc", asset_store=asset_store)
    tuple_ar0132at = sensor_create("ar0132at", [], ("rccc",), asset_store=asset_store)
    assert direct_ar0132at.name == tuple_ar0132at.name
    assert direct_ar0132at.fields["size"] == tuple_ar0132at.fields["size"]
    assert np.array_equal(sensor_get(direct_ar0132at, "pattern"), sensor_get(tuple_ar0132at, "pattern"))

    multi_mt9v024 = sensor_create("mt9v024", [], ["rgb", "mono", "rccc"], asset_store=asset_store)
    multi_ar0132at = sensor_create("ar0132at", [], ("rgb", "rgbw", "rccc"), asset_store=asset_store)
    assert [sensor.name for sensor in multi_mt9v024] == ["MTV9V024-RGB", "MTV9V024-Mono", "MTV9V024-RCCC"]
    assert [sensor.name for sensor in multi_ar0132at] == ["AR0132AT-RGB", "AR0132AT-RGBW", "AR0132AT-RCCC"]
    assert [tuple(sensor_get(sensor, "size")) for sensor in multi_mt9v024] == [(480, 752), (480, 752), (480, 752)]
    assert [tuple(sensor_get(sensor, "size")) for sensor in multi_ar0132at] == [(960, 1280), (960, 1280), (960, 1280)]
    assert [sensor_get(sensor, "filtercolorletters") for sensor in multi_mt9v024] == ["rgb", "w", "rw"]
    assert [sensor_get(sensor, "filtercolorletters") for sensor in multi_ar0132at] == ["rgb", "rgbw", "rw"]

    imx363 = sensorIMX363V2(None, "row col", [12, 16], asset_store=asset_store)
    direct_imx363 = sensor_create("imx363", None, "row col", [12, 16], asset_store=asset_store)
    overload_imx363 = sensor_create("imx363", "row col", [12, 16], asset_store=asset_store)
    assert imx363.name == direct_imx363.name
    assert imx363.name == overload_imx363.name
    assert imx363.fields["size"] == direct_imx363.fields["size"]
    assert imx363.fields["size"] == overload_imx363.fields["size"]
    assert sensor_get(imx363, "quantizationmethod") == sensor_get(direct_imx363, "quantizationmethod")
    assert sensor_get(imx363, "quantizationmethod") == sensor_get(overload_imx363, "quantizationmethod")

    filter_struct = {
        "data": np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ],
            dtype=float,
        ),
        "filterNames": ["c1", "c2", "c3"],
        "wavelength": np.array([500.0, 600.0, 700.0], dtype=float),
    }
    pattern = np.array([[1, 2], [3, 1]], dtype=int)
    interleaved = sensorInterleaved(None, pattern, filter_struct, asset_store=asset_store)
    interleaved_wave = np.asarray(sensor_get(interleaved, "wave"), dtype=float)
    interleaved_filters = np.asarray(sensor_get(interleaved, "color filters"), dtype=float)
    assert interleaved.name == "interleaved"
    assert np.array_equal(sensor_get(interleaved, "pattern"), pattern)
    assert sensor_get(interleaved, "filter names") == ["c1", "c2", "c3"]
    assert sensor_get(interleaved, "nfilters") == 3
    for wavelength, expected in {
        500.0: np.array([1.0, 0.0, 0.5], dtype=float),
        600.0: np.array([0.0, 1.0, 0.5], dtype=float),
        700.0: np.array([0.5, 0.5, 1.0], dtype=float),
    }.items():
        idx = int(np.where(np.isclose(interleaved_wave, wavelength))[0][0])
        np.testing.assert_allclose(interleaved_filters[idx], expected, rtol=1e-10, atol=1e-12)

    wave = np.arange(460.0, 481.0, dtype=float)
    imec = sensorCreateIMECSSM4x4vis("row col", [8, 12], "wave", wave, asset_store=asset_store)
    imec_default = sensorCreateIMECSSM4x4vis(asset_store=asset_store)
    imec_default_placeholder = sensorCreateIMECSSM4x4vis("row col", [], asset_store=asset_store)
    assert imec.name == "IMEC SSM"
    assert imec.fields["size"] == (8, 12)
    assert imec_default.fields["size"] == (1016, 2040)
    assert imec_default_placeholder.fields["size"] == (1016, 2040)
    assert sensor_get(imec, "quantizationmethod") == "10 bit"
    assert sensor_get(imec_default_placeholder, "quantizationmethod") == "10 bit"
    assert np.array_equal(sensor_get(imec, "pattern"), np.arange(1, 17, dtype=int).reshape(4, 4).T)
    assert np.array_equal(sensor_get(imec_default_placeholder, "pattern"), np.arange(1, 17, dtype=int).reshape(4, 4).T)
    assert np.array_equal(np.asarray(sensor_get(imec, "wave"), dtype=float), wave)
    assert sensor_get(imec, "nfilters") == 16
    assert len(sensor_get(imec, "filter names")) == 16

    imec_dispatch = sensor_create("imec44", None, [8, 12], asset_store=asset_store)
    imec_dispatch_placeholder = sensor_create("imec44", [], [8, 12], asset_store=asset_store)
    imec_dispatch_overload = sensor_create("imec44", [8, 12], asset_store=asset_store)
    imec_dispatch_empty_rowcol = sensor_create("imec44", [], [], asset_store=asset_store)
    assert imec_dispatch.name == "IMEC SSM"
    assert imec_dispatch_placeholder.name == "IMEC SSM"
    assert imec_dispatch_overload.name == "IMEC SSM"
    assert imec_dispatch_empty_rowcol.name == "IMEC SSM"
    assert imec_dispatch.fields["size"] == (8, 12)
    assert imec_dispatch_placeholder.fields["size"] == (8, 12)
    assert imec_dispatch_overload.fields["size"] == (8, 12)
    assert imec_dispatch_empty_rowcol.fields["size"] == (1016, 2040)
    assert sensor_get(imec_dispatch, "quantizationmethod") == "10 bit"
    assert sensor_get(imec_dispatch_placeholder, "quantizationmethod") == "10 bit"
    assert sensor_get(imec_dispatch_overload, "quantizationmethod") == "10 bit"
    assert sensor_get(imec_dispatch_empty_rowcol, "quantizationmethod") == "10 bit"
    assert np.array_equal(sensor_get(imec_dispatch, "pattern"), sensor_get(imec, "pattern"))
    assert np.array_equal(sensor_get(imec_dispatch_placeholder, "pattern"), sensor_get(imec, "pattern"))
    assert np.array_equal(sensor_get(imec_dispatch_overload, "pattern"), sensor_get(imec, "pattern"))
    assert np.array_equal(sensor_get(imec_dispatch_empty_rowcol, "pattern"), sensor_get(imec_default, "pattern"))

    custom_wave = np.array([500.0, 600.0, 700.0], dtype=float)
    custom_dispatch = sensor_create("custom", None, pattern, filter_struct, [6, 8], custom_wave, asset_store=asset_store)
    custom_placeholder = sensor_create("custom", [], pattern, filter_struct, [6, 8], custom_wave, asset_store=asset_store)
    custom_overload = sensor_create("custom", pattern, filter_struct, [6, 8], custom_wave, asset_store=asset_store)
    custom_pixel = pixelCreate("default", custom_wave)
    custom_pixel_placeholder = sensor_create("custom", custom_pixel, pattern, filter_struct, [], [], asset_store=asset_store)
    fourcolor = sensor_create("fourcolor", None, pattern, filter_struct, asset_store=asset_store)
    fourcolor_placeholder = sensor_create("fourcolor", [], pattern, filter_struct, asset_store=asset_store)

    assert custom_dispatch.name == "custom"
    assert custom_placeholder.name == "custom"
    assert custom_overload.name == "custom"
    assert custom_pixel_placeholder.name == "custom"
    assert custom_dispatch.fields["size"] == (6, 8)
    assert custom_placeholder.fields["size"] == (6, 8)
    assert custom_overload.fields["size"] == (6, 8)
    assert custom_pixel_placeholder.fields["size"] == pattern.shape
    assert np.array_equal(np.asarray(sensor_get(custom_dispatch, "wave"), dtype=float), custom_wave)
    assert np.array_equal(np.asarray(sensor_get(custom_placeholder, "wave"), dtype=float), custom_wave)
    assert np.array_equal(np.asarray(sensor_get(custom_overload, "wave"), dtype=float), custom_wave)
    assert np.array_equal(np.asarray(sensor_get(custom_pixel_placeholder, "wave"), dtype=float), custom_wave)
    assert np.array_equal(sensor_get(custom_dispatch, "pattern"), pattern)
    assert np.array_equal(sensor_get(custom_placeholder, "pattern"), pattern)
    assert np.array_equal(sensor_get(custom_overload, "pattern"), pattern)
    assert np.array_equal(sensor_get(custom_pixel_placeholder, "pattern"), pattern)
    assert sensor_get(custom_dispatch, "filter names") == ["c1", "c2", "c3"]
    assert sensor_get(custom_placeholder, "filter names") == ["c1", "c2", "c3"]
    assert sensor_get(custom_overload, "filter names") == ["c1", "c2", "c3"]
    assert sensor_get(custom_pixel_placeholder, "filter names") == ["c1", "c2", "c3"]

    assert fourcolor.name == "fourcolor"
    assert fourcolor_placeholder.name == "fourcolor"
    assert fourcolor.fields["size"] == sensor_create(asset_store=asset_store).fields["size"]
    assert fourcolor_placeholder.fields["size"] == sensor_create(asset_store=asset_store).fields["size"]
    assert np.array_equal(sensor_get(fourcolor, "pattern"), pattern)
    assert np.array_equal(sensor_get(fourcolor_placeholder, "pattern"), pattern)
    assert sensor_get(fourcolor, "filter names") == ["c1", "c2", "c3"]
    assert sensor_get(fourcolor_placeholder, "filter names") == ["c1", "c2", "c3"]

    nikon = sensor_create("Nikon D100", asset_store=asset_store)
    assert nikon.name == "Nikon-D100"
    assert tuple(np.asarray(sensor_get(nikon, "size"), dtype=int)) == (288, 352)
    assert float(sensor_get(nikon, "pixel width")) == pytest.approx(2.8e-6)
    assert sensor_get(nikon, "quantizationmethod") == "analog"
    assert np.array_equal(sensor_get(nikon, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert sensor_get(nikon, "filter names") == ["y_custom_", "g_custom_", "b_custom_"]
    np.testing.assert_array_equal(
        np.asarray(sensor_get(nikon, "wave"), dtype=float),
        np.arange(400.0, 701.0, 10.0, dtype=float),
    )


def test_sensor_create_ideal_match_and_match_xyz_replay_legacy_contract(asset_store) -> None:
    example = sensor_set(sensor_create("default", asset_store=asset_store), "exp time", 0.123)
    source_filters = np.asarray(sensor_get(example, "filter spectra"), dtype=float)
    source_names = list(sensor_get(example, "filter names"))
    source_pixel = sensor_get(example, "pixel")

    matched = sensor_create_ideal("match", example, asset_store=asset_store)

    assert isinstance(matched, list)
    assert len(matched) == int(sensor_get(example, "nfilters"))

    for index, current in enumerate(matched):
        assert sensor_get(current, "name") == f"mono-{source_names[index]}"
        assert np.array_equal(np.asarray(sensor_get(current, "pattern"), dtype=int), np.array([[1]], dtype=int))
        assert sensor_get(current, "filter names") == [source_names[index]]
        np.testing.assert_allclose(
            np.asarray(sensor_get(current, "filter spectra"), dtype=float),
            source_filters[:, index : index + 1],
            rtol=0.0,
            atol=0.0,
        )
        assert np.isclose(float(sensor_get(current, "exp time")), 0.123)
        assert sensor_get(current, "auto exposure") is False
        assert tuple(np.asarray(sensor_get(current, "size"), dtype=int)) == tuple(np.asarray(sensor_get(example, "size"), dtype=int))
        current_pixel = sensor_get(current, "pixel")
        assert np.allclose(np.asarray(current_pixel["size_m"], dtype=float), np.asarray(source_pixel["size_m"], dtype=float))
        assert np.isclose(float(current_pixel["fill_factor"]), float(source_pixel["fill_factor"]))

    auto_example = sensor_set(sensor_create("default", asset_store=asset_store), "auto exposure", True)
    matched_auto = sensor_create_ideal("match", auto_example, asset_store=asset_store)
    assert all(np.isclose(float(sensor_get(current, "exp time")), 0.05) for current in matched_auto)

    xyz_reference = sensor_create_ideal("xyz", asset_store=asset_store)
    assert isinstance(xyz_reference, list)
    xyz_filters = np.column_stack([np.asarray(sensor_get(current, "filter spectra"), dtype=float).reshape(-1) for current in xyz_reference])
    xyz_names = [sensor_get(current, "filter names")[0] for current in xyz_reference]
    matched_xyz = sensor_create_ideal("match xyz", example, asset_store=asset_store)

    assert isinstance(matched_xyz, list)
    assert len(matched_xyz) == 3

    for index, current in enumerate(matched_xyz):
        assert sensor_get(current, "name") == f"mono-{source_names[index]}"
        assert sensor_get(current, "filter names") == [xyz_names[index]]
        np.testing.assert_allclose(
            np.asarray(sensor_get(current, "filter spectra"), dtype=float),
            xyz_filters[:, index : index + 1],
            rtol=0.0,
            atol=0.0,
        )
        assert np.isclose(float(sensor_get(current, "exp time")), 0.123)
        assert sensor_get(current, "auto exposure") is False
        assert tuple(np.asarray(sensor_get(current, "size"), dtype=int)) == tuple(np.asarray(sensor_get(example, "size"), dtype=int))
        current_pixel = sensor_get(current, "pixel")
        assert np.allclose(np.asarray(current_pixel["size_m"], dtype=float), np.asarray(source_pixel["size_m"], dtype=float))
        assert np.isclose(float(current_pixel["fill_factor"]), float(source_pixel["fill_factor"]))


def test_sensor_create_ideal_defaults_to_monochrome(asset_store) -> None:
    implicit = sensor_create_ideal(asset_store=asset_store)
    explicit = sensor_create_ideal("monochrome", asset_store=asset_store)

    assert isinstance(implicit, sensor_module.Sensor)
    assert implicit.name == "Monochrome"
    assert explicit.name == "Monochrome"
    assert np.array_equal(np.asarray(sensor_get(implicit, "pattern"), dtype=int), np.array([[1]], dtype=int))
    assert np.array_equal(np.asarray(sensor_get(implicit, "pattern"), dtype=int), np.asarray(sensor_get(explicit, "pattern"), dtype=int))
    assert sensor_get(implicit, "filter names") == ["w"]
    assert sensor_get(implicit, "filter names") == sensor_get(explicit, "filter names")
    assert tuple(np.asarray(sensor_get(implicit, "size"), dtype=int)) == tuple(np.asarray(sensor_get(explicit, "size"), dtype=int))
    assert np.array_equal(np.asarray(sensor_get(implicit, "wave"), dtype=float), np.asarray(sensor_get(explicit, "wave"), dtype=float))


def test_sensor_create_ideal_accepts_empty_placeholder_for_pixel_size(asset_store) -> None:
    monochrome_placeholder = sensor_create_ideal("monochrome", [], 3e-6, asset_store=asset_store)
    monochrome_explicit = sensor_create_ideal("monochrome", None, 3e-6, asset_store=asset_store)
    monochrome_vector = sensor_create_ideal("monochrome", [], [2e-6, 3e-6], asset_store=asset_store)

    assert isinstance(monochrome_placeholder, sensor_module.Sensor)
    assert monochrome_placeholder.name == "Monochrome"
    assert monochrome_explicit.name == "Monochrome"
    assert monochrome_vector.name == "Monochrome"
    assert sensor_get(monochrome_placeholder, "filter names") == ["w"]
    assert sensor_get(monochrome_placeholder, "filter names") == sensor_get(monochrome_explicit, "filter names")
    np.testing.assert_allclose(
        np.asarray(sensor_get(monochrome_placeholder, "pixel")["size_m"], dtype=float),
        np.array([3e-6, 3e-6], dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(sensor_get(monochrome_placeholder, "pixel")["size_m"], dtype=float),
        np.asarray(sensor_get(monochrome_explicit, "pixel")["size_m"], dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(sensor_get(monochrome_vector, "pixel")["size_m"], dtype=float),
        np.array([2e-6, 3e-6], dtype=float),
        rtol=0.0,
        atol=0.0,
    )

    xyz_placeholder = sensor_create_ideal("xyz", [], 3e-6, asset_store=asset_store)
    xyz_explicit = sensor_create_ideal("xyz", None, 3e-6, asset_store=asset_store)
    xyz_vector = sensor_create_ideal("xyz", [], [2e-6, 3e-6], asset_store=asset_store)

    assert isinstance(xyz_placeholder, list)
    assert isinstance(xyz_explicit, list)
    assert isinstance(xyz_vector, list)
    assert [sensor.name for sensor in xyz_placeholder] == ["CIE-X-ideal", "CIE-Y-ideal", "CIE-Z-ideal"]
    assert [sensor.name for sensor in xyz_placeholder] == [sensor.name for sensor in xyz_explicit]
    assert [sensor.name for sensor in xyz_placeholder] == [sensor.name for sensor in xyz_vector]
    assert [sensor_get(sensor, "filter names") for sensor in xyz_placeholder] == [["rX"], ["gY"], ["bZ"]]
    assert [sensor_get(sensor, "filter names") for sensor in xyz_placeholder] == [
        sensor_get(sensor, "filter names") for sensor in xyz_explicit
    ]
    assert [sensor_get(sensor, "filter names") for sensor in xyz_placeholder] == [
        sensor_get(sensor, "filter names") for sensor in xyz_vector
    ]

    for placeholder_sensor, explicit_sensor in zip(xyz_placeholder, xyz_explicit, strict=False):
        np.testing.assert_allclose(
            np.asarray(sensor_get(placeholder_sensor, "pixel")["size_m"], dtype=float),
            np.array([3e-6, 3e-6], dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(sensor_get(placeholder_sensor, "pixel")["size_m"], dtype=float),
            np.asarray(sensor_get(explicit_sensor, "pixel")["size_m"], dtype=float),
            rtol=0.0,
            atol=0.0,
        )
    for vector_sensor in xyz_vector:
        np.testing.assert_allclose(
            np.asarray(sensor_get(vector_sensor, "pixel")["size_m"], dtype=float),
            np.array([2e-6, 3e-6], dtype=float),
            rtol=0.0,
            atol=0.0,
        )


def test_sensor_create_ideal_monochrome_rejects_nonempty_sensor_example(asset_store) -> None:
    example = sensor_create("default", asset_store=asset_store)

    with pytest.raises(ValueError, match=r"Sensor example not used for monochrome case\."):
        sensor_create_ideal("monochrome", example, asset_store=asset_store)


def test_sensor_create_ideal_dispatcher_is_deprecated(asset_store) -> None:
    with pytest.raises(ValueError, match=r"sensorCreate\('ideal'\) is deprecated\. Use sensorCreateIdeal"):
        sensor_create("ideal", asset_store=asset_store)


def test_camera_create_ideal_returns_monochrome_camera(asset_store) -> None:
    ideal = camera_create("ideal", asset_store=asset_store)

    assert not isinstance(ideal, list)
    assert ideal.name == "ideal"
    assert ideal.fields["sensor"].name == "Monochrome"
    assert camera_get(ideal, "sensor filter names") == ["w"]
    assert np.array_equal(np.asarray(camera_get(ideal, "sensor pattern"), dtype=int), np.array([[1]], dtype=int))


def test_camera_create_ideal_monochrome_replays_legacy_camera_name(asset_store) -> None:
    ideal = camera_create("ideal", asset_store=asset_store)
    compact = camera_create("idealmonochrome", asset_store=asset_store)
    spaced = camera_create("ideal monochrome", asset_store=asset_store)

    assert compact.name == "ideal monochrome"
    assert spaced.name == "ideal monochrome"
    assert compact.fields["sensor"].name == "Monochrome"
    assert spaced.fields["sensor"].name == "Monochrome"
    assert compact.fields["sensor"].name == ideal.fields["sensor"].name
    assert spaced.fields["sensor"].name == ideal.fields["sensor"].name
    assert camera_get(compact, "sensor filter names") == ["w"]
    assert camera_get(spaced, "sensor filter names") == ["w"]


def test_sensor_create_imx363_and_crop_support_raw_tutorial_flow(asset_store) -> None:
    sensor = sensor_create("IMX363", None, "row col", [12, 16], asset_store=asset_store)

    assert sensor.name == "IMX363"
    assert sensor.fields["size"] == (12, 16)
    assert sensor_get(sensor, "quantizationmethod") == "10 bit"

    sensor = sensor_set(sensor, "pattern", np.array([[2, 1], [3, 2]], dtype=int))
    sensor = sensor_set(sensor, "wave", np.arange(400.0, 701.0, 10.0, dtype=float))
    dv = np.arange(12 * 16, dtype=float).reshape((12, 16), order="F")
    sensor = sensor_set(sensor, "digital values", dv)

    cropped = sensor_crop(sensor, [2, 3, 7, 5])

    assert sensor_get(cropped, "size") == (6, 8)
    assert np.array_equal(sensor_get(cropped, "metadata crop"), np.array([3, 3, 7, 5], dtype=int))
    assert np.array_equal(sensor_get(cropped, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert np.array_equal(sensor_get(cropped, "digital values"), dv[2:8, 2:10])


def test_sensor_compute_supports_rgbw_and_rccc_presets(asset_store) -> None:
    scene = scene_create("uniform ee", 16, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    rgbw = sensor_set(sensor_create("rgbw", asset_store=asset_store), "noise flag", 0)
    rgbw = sensor_set(rgbw, "integration time", 0.01)
    rccc = sensor_set(sensor_create("rccc", asset_store=asset_store), "noise flag", 0)
    rccc = sensor_set(rccc, "integration time", 0.01)

    rgbw_result = sensor_compute(rgbw, oi, seed=0)
    rccc_result = sensor_compute(rccc, oi, seed=0)

    assert rgbw_result.data["volts"].shape == rgbw.fields["size"]
    assert rccc_result.data["volts"].shape == rccc.fields["size"]
    assert np.all(rgbw_result.data["volts"] >= 0.0)
    assert np.all(rccc_result.data["volts"] >= 0.0)


def test_sensor_compute_supports_vendor_rgbw_qe_sampling(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_set(sensor_create("mt9v024", "rgbw", asset_store=asset_store), "noise flag", 0)
    sensor = sensor_set(sensor, "integration time", 0.01)

    result = sensor_compute(sensor, oi, seed=0)

    assert result.data["volts"].shape == sensor.fields["size"]
    assert np.all(result.data["volts"] >= 0.0)


def test_sensor_set_size_respects_cfa_block_and_clears_cached_data(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.ones((7, 9), dtype=float))

    assert sensor_get(sensor, "size") == (7, 9)

    sensor = sensor_set(sensor, "size", (73, 89))
    assert sensor_get(sensor, "size") == (72, 88)
    assert sensor.data == {}

    sensor = sensor_set(sensor, "rows", 75)
    sensor = sensor_set(sensor, "cols", 91)
    assert sensor_get(sensor, "rows") == 74
    assert sensor_get(sensor, "cols") == 90


def test_sensor_clear_data_clears_computed_payloads(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.arange(1, 37, dtype=float).reshape(6, 6) / 36.0
    dv = np.arange(1, 37, dtype=float).reshape(6, 6)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dv", dv)

    cleared = sensorClearData(sensor)

    assert cleared is not sensor
    assert cleared.data == {}
    assert tuple(sensor_get(cleared, "size")) == (6, 6)
    assert np.array_equal(np.asarray(sensor_get(sensor, "volts"), dtype=float), volts)
    assert np.array_equal(np.asarray(sensor_get(sensor, "dv"), dtype=float), dv)
    assert sensor_get(cleared, "volts") is None
    assert sensor_get(cleared, "dv") is None
    assert np.array_equal(np.asarray(sensor_get(cleared, "wave"), dtype=float), np.asarray(sensor_get(sensor, "wave"), dtype=float))


def test_sensor_no_noise_matches_legacy_parameter_reset(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "prnulevel", 12.5)
    sensor = sensor_set(sensor, "dsnulevel", 0.015)
    sensor = sensor_set(sensor, "quantizationmethod", "coded")
    sensor = sensor_set(sensor, "pixel read noise volts", 1.5e-3)
    sensor = sensor_set(sensor, "pixel dark voltage", 2.5e-3)

    noiseless = sensorNoNoise(sensor)

    assert noiseless is not sensor
    assert np.isclose(float(sensor_get(noiseless, "prnulevel")), 0.0)
    assert np.isclose(float(sensor_get(noiseless, "dsnulevel")), 0.0)
    assert sensor_get(noiseless, "quantizationmethod") == "analog"
    assert np.isclose(float(sensor_get(noiseless, "pixel read noise volts")), 0.0)
    assert np.isclose(float(sensor_get(noiseless, "pixel dark voltage")), 0.0)
    assert np.isclose(float(sensor_get(sensor, "prnulevel")), 12.5)
    assert np.isclose(float(sensor_get(sensor, "pixel read noise volts")), 1.5e-3)


def test_sensor_utility_wrappers_match_legacy_contracts(asset_store, tmp_path) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "quantization method", "analog")

    pd_array = sensorPDArray(sensor, 1.0)
    expected_fill_factor = float(pixelGet(sensor_get(sensor, "pixel"), "fill factor"))
    assert pd_array.shape == (1, 1)
    assert np.isclose(float(pd_array[0, 0]), expected_fill_factor)

    sensor = sensor_set(sensor, "volts", np.arange(1.0, 17.0, dtype=float).reshape(4, 4) / 16.0)
    jiggled = sensorJiggle(sensor, 5)
    assert jiggled is not sensor
    assert np.allclose(np.asarray(sensor_get(jiggled, "volts"), dtype=float), np.asarray(sensor_get(sensor, "volts"), dtype=float))

    monochrome = sensor_create("monochrome", asset_store=asset_store)
    mpe30 = sensorMPE30(monochrome)
    snr, luxsec = sensorSNRluxsec(monochrome, asset_store=asset_store)
    expected_mpe30 = np.interp(30.0, np.asarray(snr, dtype=float), np.asarray(luxsec, dtype=float).reshape(-1))
    assert np.isclose(float(mpe30), float(expected_mpe30))

    vignetting_sensor = sensor_set(sensor_create(asset_store=asset_store), "size", np.array([8, 8], dtype=int))
    skipped = sensorVignetting(vignetting_sensor)
    assert np.allclose(np.asarray(sensor_get(skipped, "etendue"), dtype=float), np.ones((8, 8), dtype=float))

    centered = sensorVignetting(vignetting_sensor, 2, 1, asset_store=asset_store)
    expected_centered = mlAnalyzeArrayEtendue(vignetting_sensor, "centered", 1, asset_store=asset_store)
    assert np.allclose(
        np.asarray(sensor_get(centered, "etendue"), dtype=float),
        np.asarray(sensor_get(expected_centered, "etendue"), dtype=float),
    )

    wave = np.array([500.0, 501.0, 502.0], dtype=float)
    scene = scene_create("uniform ee", 12, wave, asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene, crop=True)
    photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    direct_sensor = sensor_create(asset_store=asset_store)
    direct_sensor = sensor_set(direct_sensor, "noise flag", 0)
    direct_sensor = sensor_set(direct_sensor, "quantization method", "analog")
    direct_sensor = sensor_set(direct_sensor, "integration time", 0.01)
    direct = sensor_compute(direct_sensor, oi)

    for index, wavelength in enumerate(wave, start=1):
        band = oi.clone()
        band.fields["wave"] = np.array([wavelength], dtype=float)
        band.data["photons"] = photons[:, :, index - 1 : index]
        band.name = f"band-{int(wavelength)}"
        vcSaveObject(band, tmp_path / f"oi{index:02d}.mat")

    replay_sensor = sensor_create(asset_store=asset_store)
    replay_sensor = sensor_set(replay_sensor, "noise flag", 0)
    replay_sensor = sensor_set(replay_sensor, "quantization method", "analog")
    replay_sensor = sensor_set(replay_sensor, "integration time", 0.01)
    wb = sensorWBCompute(replay_sensor, tmp_path)

    assert np.allclose(np.asarray(sensor_get(wb, "volts"), dtype=float), np.asarray(sensor_get(direct, "volts"), dtype=float))
    assert sensor_get(wb, "name") == "wb-oi03"


def test_sensor_simulation_legacy_wrappers(asset_store) -> None:
    scene = scene_create("uniform ee", 8, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create(), scene)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 8)
    sensor = sensor_set(sensor, "cols", 8)
    sensor = sensor_set(sensor, "integration time", 0.01)
    sensor = sensor_set(sensor, "auto exposure", False)
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 0.05)
    sensor = sensor_set(sensor, "quantization method", "linear")
    sensor = sensor_set(sensor, "noise flag", 2)
    sensor = sensor_set(sensor, "prnulevel", 5.0)
    sensor = sensor_set(sensor, "dsnulevel", 0.01)
    sensor = sensor_set(sensor, "column fpn", np.array([0.002, 0.01], dtype=float))
    sensor = sensor_set(sensor, "noise seed", 11)
    sensor = sensor_set(sensor, "reuse noise", True)

    noise_free = sensorComputeNoiseFree(sensor, oi)
    noise_free_snake = sensor_compute_noise_free(sensor, oi)
    assert np.allclose(np.asarray(sensor_get(noise_free, "volts"), dtype=float), np.asarray(sensor_get(noise_free_snake, "volts"), dtype=float))
    assert sensor_get(noise_free, "quantization method") == "linear"
    assert np.isclose(float(sensor_get(noise_free, "analog gain")), 2.0)
    assert np.isclose(float(sensor_get(noise_free, "analog offset")), 0.05)
    assert np.isclose(float(sensor_get(noise_free, "pixel voltage swing")), float(sensor_get(sensor, "pixel voltage swing")))

    noisy_a = sensorAddNoise(noise_free)
    noisy_b = sensor_add_noise(noise_free)
    assert np.allclose(np.asarray(sensor_get(noisy_a, "volts"), dtype=float), np.asarray(sensor_get(noisy_b, "volts"), dtype=float))
    assert np.asarray(noisy_a.fields["offset_fpn_image"], dtype=float).shape == tuple(sensor_get(sensor, "size"))
    assert np.asarray(noisy_a.fields["gain_fpn_image"], dtype=float).shape == tuple(sensor_get(sensor, "size"))
    assert np.asarray(noisy_a.fields["column_offset_fpn"], dtype=float).reshape(-1).size == int(sensor_get(sensor, "cols"))
    assert np.asarray(noisy_a.fields["column_gain_fpn"], dtype=float).reshape(-1).size == int(sensor_get(sensor, "cols"))

    noise_stage = sensorComputeNoise(noise_free, [], seed=11)
    expected_volts = np.clip(
        (np.asarray(sensor_get(noisy_a, "volts"), dtype=float) + 0.05) / 2.0,
        0.0,
        float(sensor_get(sensor, "pixel voltage swing")),
    )
    expected_dv, _ = analog2digital(sensor_set(noisy_a.clone(), "volts", expected_volts), "linear")
    assert np.allclose(np.asarray(sensor_get(noise_stage, "volts"), dtype=float), expected_volts)
    assert np.array_equal(np.asarray(sensor_get(noise_stage, "digital values"), dtype=float), np.asarray(expected_dv, dtype=float))

    volt_image, dsnu, prnu = sensorComputeImage(oi, sensor)
    expected_noisy = sensorAddNoise(sensorComputeNoiseFree(sensor, oi))
    expected_volt_image = (np.asarray(sensor_get(expected_noisy, "volts"), dtype=float) + 0.05) / 2.0
    assert np.allclose(np.asarray(volt_image, dtype=float), expected_volt_image)
    assert np.allclose(np.asarray(dsnu, dtype=float), np.asarray(expected_noisy.fields["offset_fpn_image"], dtype=float))
    assert np.allclose(np.asarray(prnu, dtype=float), np.asarray(expected_noisy.fields["gain_fpn_image"], dtype=float))

    filters = np.asarray(sensor_get(sensor, "color filters"), dtype=float)[:, :2]
    full_volts, full_dvs = sensorComputeFullArray(sensor_set(sensor.clone(), "noise flag", 0), oi, filters)
    full_volts_snake, full_dvs_snake = sensor_compute_full_array(sensor_set(sensor.clone(), "noise flag", 0), oi, filters)
    assert np.allclose(full_volts, full_volts_snake)
    assert np.allclose(full_dvs, full_dvs_snake)
    assert full_volts.shape == (8, 8, 2)
    assert full_dvs.shape == (8, 8, 2)


def test_auto_exposure_legacy_wrapper_supported_methods(asset_store) -> None:
    scene = scene_create("uniform ee", 8, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create(), scene)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 8)
    sensor = sensor_set(sensor, "cols", 8)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "auto exposure", False)

    default_time, default_max, small_oi = autoExposure(oi, sensor, [], [])
    luminance_time, luminance_max, luminance_oi = autoExposure(oi, sensor, 0.95, "luminance")
    assert np.isclose(default_time, luminance_time)
    assert np.isclose(default_max, luminance_max)
    assert small_oi is not None
    assert luminance_oi is not None
    assert np.asarray(small_oi.data["photons"], dtype=float).shape[:2] == (2, 2)
    assert np.allclose(np.asarray(small_oi.data["photons"], dtype=float), np.asarray(luminance_oi.data["photons"], dtype=float))

    working = sensor_set(sensor.clone(), "integration time", 1)
    voltage_swing = float(sensor_get(working, "pixel voltage swing"))
    signal_voltage = np.asarray(sensor_module.sensor_compute_image(oi, working, seed=7)[0], dtype=float)

    full_time, full_max, _ = autoExposure(oi, sensor, 0.9, "full", seed=7)
    assert np.isclose(full_max, np.max(signal_voltage))
    assert np.isclose(full_time, (0.9 * voltage_swing) / np.max(signal_voltage))

    mean_time, mean_max, _ = autoExposure(oi, sensor, 0.9, "mean", seed=7)
    expected_mean_time = (0.9 * voltage_swing) / np.mean(signal_voltage)
    assert np.isclose(mean_time, expected_mean_time)
    assert np.isclose(mean_max, expected_mean_time * np.max(signal_voltage))

    cfa_time, cfa_max, _ = autoExposure(oi, sensor, 0.9, "cfa", seed=7)
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    assert np.asarray(cfa_time, dtype=float).shape == pattern.shape
    assert np.asarray(cfa_max, dtype=float).shape == pattern.shape
    for row in range(pattern.shape[0]):
        for col in range(pattern.shape[1]):
            current = signal_voltage[row::pattern.shape[0], col::pattern.shape[1]]
            expected_max = float(np.max(current))
            assert np.isclose(cfa_max[row, col], expected_max)
            assert np.isclose(cfa_time[row, col], (0.9 * voltage_swing) / expected_max)

    specular_time, specular_max, _ = autoExposure(oi, sensor, 0.9, "specular", seed=7)
    assert np.isclose(specular_max, np.max(signal_voltage))
    assert np.isclose(specular_time, voltage_swing / np.percentile(signal_voltage, 90.0))


def test_auto_exposure_legacy_wrapper_weighted_and_video(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 16)
    sensor = sensor_set(sensor, "cols", 16)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "auto exposure", False)
    rect = np.array([4, 4, 5, 5], dtype=int)

    weighted_time, weighted_max, weighted_oi = autoExposure(
        oi,
        sensor,
        0.9,
        "weighted",
        "center rect",
        rect,
        seed=3,
    )
    assert weighted_max is None
    assert weighted_oi is None

    center_oi = optics_module.oi_crop(oi, rect)
    small_sensor = sensor_module.sensor_set_size_to_fov(sensor.clone(), float(optics_module.oi_get(center_oi, "fov")), oi)
    small_sensor = sensor_set(small_sensor, "noise flag", 0)
    small_sensor = sensor_set(small_sensor, "integration time", 1)
    small_sensor = sensor_module.sensor_clear_data(small_sensor)
    voltage_swing = float(sensor_get(small_sensor, "pixel voltage swing"))
    signal_voltage = np.asarray(sensor_module.sensor_compute_image(center_oi, small_sensor, seed=3)[0], dtype=float)
    expected_weighted = (0.9 * voltage_swing) / np.max(signal_voltage)
    assert np.isclose(weighted_time, expected_weighted)

    video_time, video_max, video_oi = autoExposure(
        oi,
        sensor,
        0.9,
        "video",
        "center rect",
        rect,
        "video max",
        weighted_time / 2.0,
        seed=3,
    )
    assert video_max is None
    assert video_oi is None
    assert np.isclose(video_time, weighted_time / 2.0)


def test_sensor_simulation_helper_wrappers_match_legacy_contracts(asset_store, tmp_path) -> None:
    scene = scene_create("uniform ee", 8, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "quantization method", "analog")
    sensor = sensor_set(sensor, "integration time", 0.01)

    oi_rows, oi_cols = tuple(int(v) for v in np.asarray(oi_get(oi, "size"), dtype=int))
    scdi = np.stack(
        [
            np.ones((oi_rows, oi_cols), dtype=float),
            np.full((oi_rows, oi_cols), 2.0, dtype=float),
            np.full((oi_rows, oi_cols), 3.0, dtype=float),
        ],
        axis=2,
    )
    flat_scdi, new_rows, new_cols = regridOI2ISA(scdi, oi, sensor, 1.0)
    cfa_pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), 4, 4)
    expected_flat = np.choose(cfa_pattern - 1, [np.ones((4, 4), dtype=float), np.full((4, 4), 2.0), np.full((4, 4), 3.0)])
    np.testing.assert_allclose(flat_scdi, expected_flat, rtol=1e-10, atol=1e-12)
    assert new_rows.shape == (4,)
    assert new_cols.shape == (4,)

    img = np.arange(1.0, 17.0, dtype=float).reshape(4, 4)
    rgb_format, cfa_vals = plane2mosaic(img, sensor, -1.0)
    for index in range(3):
        expected_plane = np.full((4, 4), -1.0, dtype=float)
        expected_plane[cfa_pattern == (index + 1)] = img[cfa_pattern == (index + 1)]
        np.testing.assert_allclose(rgb_format[:, :, index], expected_plane, rtol=1e-10, atol=1e-12)
    assert np.array_equal(cfa_vals, np.array([1, 2, 3], dtype=int))

    mev_sensor = sensor_set(sensor.clone(), "integration time", np.array([0.01, 0.1], dtype=float))
    mev_sensor = sensor_set(mev_sensor, "pixel voltage swing", 1e-4)
    stack = sensor_compute(mev_sensor, oi)
    mev = sensorComputeMEV(mev_sensor, oi)
    volts = np.asarray(sensor_get(stack, "volts"), dtype=float)
    exp_times = np.asarray(sensor_get(stack, "exp time"), dtype=float).reshape(-1)
    combined_expected = volts[:, :, -1].copy()
    max_time = float(np.max(exp_times))
    max_v = float(sensor_get(stack, "pixel voltage swing")) * 0.95
    for exposure_index in range(exp_times.size - 2, -1, -1):
        saturated = combined_expected > max_v
        if not np.any(saturated):
            break
        scale_factor = max_time / float(exp_times[exposure_index])
        scaled = volts[:, :, exposure_index] * scale_factor
        combined_expected[saturated] = scaled[saturated]
        max_v *= scale_factor
    np.testing.assert_allclose(np.asarray(sensor_get(mev, "volts"), dtype=float), combined_expected, rtol=1e-10, atol=1e-12)
    assert sensor_get(mev, "name") == "combined"
    assert np.isclose(float(sensor_get(mev, "exp time")), max_time)

    optics = oi_get(oi, "optics")
    focal_length = float(opticsGet(optics, "focal length"))
    sensor_deg_map = np.asarray(sensor_get(sensor, "chiefRayAngleDegrees", focal_length), dtype=float)
    max_deg = float(np.max(sensor_deg_map))
    filter_wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    filter_data = np.column_stack(
        [
            np.ones(filter_wave.size, dtype=float),
            np.full(filter_wave.size, 0.7, dtype=float),
            np.full(filter_wave.size, 0.3, dtype=float),
        ]
    )
    filter_file = tmp_path / "sv_filters.mat"
    savemat(
        filter_file,
        {
            "wavelength": filter_wave,
            "data": filter_data,
            "filterNames": np.asarray(["f1", "f2", "f3"], dtype=object),
            "fHeight": np.array([0.0, max_deg * 0.5, max_deg], dtype=float),
        },
    )

    combined_volts, v_images, ir_filters = sensorComputeSVFilters(sensor, oi, filter_file, asset_store=asset_store)
    assert ir_filters.shape[1] == 3

    expected_v_images = np.zeros_like(v_images)
    for index in range(ir_filters.shape[1]):
        current_sensor = sensor_set(sensor.clone(), "irFilter", ir_filters[:, index])
        current_sensor = sensor_compute(current_sensor, oi)
        expected_v_images[:, :, index] = np.asarray(sensor_get(current_sensor, "volts"), dtype=float)
    np.testing.assert_allclose(v_images, expected_v_images, rtol=1e-10, atol=1e-12)

    expected_combined = np.zeros_like(combined_volts)
    field_heights = np.array([0.0, max_deg * 0.5, max_deg], dtype=float)
    for index in range(1, field_heights.size):
        this_band = (field_heights[index - 1] < sensor_deg_map) & (sensor_deg_map < field_heights[index])
        inner_distance = np.abs(sensor_deg_map - field_heights[index - 1])
        inner_weight = 1.0 - (inner_distance / max(field_heights[index] - field_heights[index - 1], 1e-30))
        inner_weight[~this_band] = 0.0
        weighted = (inner_weight * expected_v_images[:, :, index - 1]) + ((1.0 - inner_weight) * expected_v_images[:, :, index])
        expected_combined[this_band] = weighted[this_band]
    np.testing.assert_allclose(combined_volts, expected_combined, rtol=1e-10, atol=1e-12)


def test_sensor_binning_noise_wrappers_match_legacy_contracts(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.arange(1.0, 17.0, dtype=float).reshape(4, 4) / 20.0
    dv = volts * 100.0
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "digital values", dv)
    sensor = sensor_set(sensor, "integration time", 0.02)
    sensor = sensor_set(sensor, "prnulevel", 5.0)
    sensor = sensor_set(sensor, "dsnulevel", 0.01)
    sensor = sensor_set(sensor, "column fpn", np.array([0.002, 0.01], dtype=float))

    noisy_fpn, dsnu, prnu = binNoiseFPN(sensor, seed=7)
    expected_noisy_fpn, expected_dsnu, expected_prnu = noiseFPN(sensor, seed=7)
    np.testing.assert_allclose(noisy_fpn, expected_noisy_fpn, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dsnu, expected_dsnu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prnu, expected_prnu, rtol=1e-10, atol=1e-12)

    noisy_col, col_dsnu, col_prnu = binNoiseColumnFPN(sensor, seed=9)
    expected_noisy_col, expected_col_dsnu, expected_col_prnu = noiseColumnFPN(sensor, seed=9)
    np.testing.assert_allclose(noisy_col, expected_noisy_col, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(col_dsnu, expected_col_dsnu, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(col_prnu, expected_col_prnu, rtol=1e-10, atol=1e-12)

    sigma_read = float(pixelGet(sensor_get(sensor, "pixel"), "readNoiseVolts"))
    rng = np.random.default_rng(13)
    expected_read_noise = rng.normal(0.0, sigma_read, size=dv.shape)
    noisy_read, read_noise = binNoiseRead(sensor, seed=13)
    np.testing.assert_allclose(read_noise, expected_read_noise, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(noisy_read, dv + expected_read_noise, rtol=1e-10, atol=1e-12)


def test_bin_sensor_compute_wrappers_match_legacy_contracts(asset_store) -> None:
    scene = scene_create("uniform ee", 8, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 8)
    sensor = sensor_set(sensor, "cols", 8)
    sensor = sensor_set(sensor, "integration time", 0.01)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 0.05)
    sensor = sensor_set(sensor, "quantization method", "linear")

    dv_stage, volt_image, dsnu, prnu = binSensorComputeImage(oi, sensor, "kodak2008")
    noise_free = sensorComputeNoiseFree(sensor, oi)
    expected_volt_image = np.asarray(sensor_get(noise_free, "volts"), dtype=float)
    expected_stage_sensor = binPixel(sensor_set(sensor.clone(), "volts", expected_volt_image), "kodak2008")
    expected_dv_stage = np.asarray(sensor_get(expected_stage_sensor, "dv"), dtype=float)

    np.testing.assert_allclose(volt_image, expected_volt_image, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dv_stage, expected_dv_stage, rtol=1e-10, atol=1e-12)
    assert dsnu is None
    assert prnu is None

    binned = binSensorCompute(sensor, oi, "kodak2008", 0)
    voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
    expected_linear = np.clip((expected_dv_stage + 0.05) / 2.0, 0.0, voltage_swing)
    quant_sensor = sensor_set(sensor.clone(), "volts", expected_linear)
    expected_quantized, _ = analog2digital(quant_sensor, "linear")
    expected_post = np.asarray(
        sensor_get(
            binPixelPost(sensor_set(quant_sensor, "digital values", expected_quantized), "kodak2008"),
            "dv",
        ),
        dtype=float,
    )

    np.testing.assert_allclose(np.asarray(sensor_get(binned, "volts"), dtype=float), expected_linear, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sensor_get(binned, "dv"), dtype=float), expected_post, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(binned.fields["pre_binning_volts"], dtype=float), expected_volt_image, rtol=1e-10, atol=1e-12)
    assert sensor_get(binned, "sensorcompute") == {"name": "binning", "method": "kodak2008"}


def test_sensor_gain_offset_matches_legacy_voltage_formula(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float)
    sensor = sensor_set(sensor, "volts", volts)

    adjusted = sensorGainOffset(sensor, 2.0, 0.1)

    assert adjusted is not sensor
    assert np.allclose(np.asarray(sensor_get(adjusted, "volts"), dtype=float), (volts + 0.1) / 2.0)
    assert np.isclose(float(sensor_get(adjusted, "analog gain")), 2.0)
    assert np.isclose(float(sensor_get(adjusted, "analog offset")), 0.1)
    assert np.allclose(np.asarray(sensor_get(sensor, "volts"), dtype=float), volts)


def test_sensor_resample_wave_resamples_legacy_spectral_payloads(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "wavelengthsamples", np.array([500.0, 600.0, 700.0], dtype=float))
    sensor = sensor_set(
        sensor,
        "filterSpectra",
        np.array(
            [
                [0.1, 0.8],
                [0.5, 0.4],
                [0.9, 0.0],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(sensor, "irFilter", np.array([1.0, 0.5, 0.0], dtype=float))
    sensor = sensor_set(sensor, "pixelSpectralQE", np.array([0.2, 0.6, 1.0], dtype=float))

    new_wave = np.array([450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0], dtype=float)
    resampled = sensorResampleWave(sensor, new_wave)

    expected_filters = np.column_stack(
        [
            np.interp(new_wave, np.array([500.0, 600.0, 700.0], dtype=float), np.array([0.1, 0.5, 0.9], dtype=float), left=0.0, right=0.0),
            np.interp(new_wave, np.array([500.0, 600.0, 700.0], dtype=float), np.array([0.8, 0.4, 0.0], dtype=float), left=0.0, right=0.0),
        ]
    )
    expected_ir = np.interp(
        new_wave,
        np.array([500.0, 600.0, 700.0], dtype=float),
        np.array([1.0, 0.5, 0.0], dtype=float),
        left=0.0,
        right=0.0,
    )
    expected_qe = np.interp(
        new_wave,
        np.array([500.0, 600.0, 700.0], dtype=float),
        np.array([0.2, 0.6, 1.0], dtype=float),
        left=0.0,
        right=0.0,
    )

    assert resampled is not sensor
    assert np.array_equal(np.asarray(sensor_get(resampled, "wave"), dtype=float), new_wave)
    assert np.array_equal(np.asarray(sensor_get(resampled, "pixel wavelength"), dtype=float), new_wave)
    assert np.allclose(np.asarray(sensor_get(resampled, "filterSpectra"), dtype=float), expected_filters)
    assert np.allclose(np.asarray(sensor_get(resampled, "irFilter"), dtype=float), expected_ir)
    assert np.allclose(np.asarray(sensor_get(resampled, "pixelSpectralQE"), dtype=float), expected_qe)
    assert tuple(sensor_get(resampled, "size")) == tuple(sensor_get(sensor, "size"))
    assert np.array_equal(np.asarray(sensor_get(sensor, "wave"), dtype=float), np.array([500.0, 600.0, 700.0], dtype=float))


def test_analog2digital_matches_legacy_quantization_formula(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.array([[0.0, 0.12], [0.49, 0.98]], dtype=float)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "nbits", 4)
    sensor = sensor_set(sensor, "quantization method", "linear")

    quantized, quantization_error = analog2digital(sensor)

    step = float(sensor_get(sensor, "pixel voltage swing")) / 16.0
    expected_quantized = np.rint(volts / step)
    assert np.allclose(quantized, expected_quantized)
    assert np.allclose(quantization_error, volts - (expected_quantized * step))

    analog_quantized, analog_error = analog2digital(sensor, "analog")
    assert np.allclose(analog_quantized, volts)
    assert np.allclose(analog_error, np.zeros_like(volts))


def test_noise_fpn_matches_legacy_formula_and_zero_exposure_branch(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.array([[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], dtype=float)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dsnulevel", 0.01)
    sensor = sensor_set(sensor, "prnulevel", 5.0)
    sensor = sensor_set(sensor, "integration time", 0.05)
    sensor = sensor_set(sensor, "auto exposure", False)

    noisy_a, dsnu_a, prnu_a = noiseFPN(sensor, seed=9)
    noisy_b, dsnu_b, prnu_b = noiseFPN(sensor, seed=9)

    assert np.array_equal(dsnu_a, dsnu_b)
    assert np.array_equal(prnu_a, prnu_b)
    assert np.allclose(noisy_a, (volts * prnu_a) + dsnu_a)
    assert np.allclose(noisy_a, noisy_b)

    sensor = sensor_set(sensor, "offset fpn image", np.full(volts.shape, 0.03, dtype=float))
    sensor = sensor_set(sensor, "gain fpn image", np.full(volts.shape, 1.1, dtype=float))
    noisy_stored, dsnu_stored, prnu_stored = noiseFPN(sensor, seed=1)
    assert np.allclose(dsnu_stored, 0.03)
    assert np.allclose(prnu_stored, 1.1)
    assert np.allclose(noisy_stored, (volts * 1.1) + 0.03)

    zero_exposure = sensor_set(sensor.clone(), "integration time", 0.0)
    zero_exposure = sensor_set(zero_exposure, "offset fpn image", np.full(volts.shape, 0.02, dtype=float))
    zero_exposure = sensor_set(zero_exposure, "gain fpn image", np.full(volts.shape, 1.4, dtype=float))
    noisy_zero, dsnu_zero, prnu_zero = noiseFPN(zero_exposure, seed=3)
    assert np.allclose(noisy_zero, dsnu_zero)
    assert np.allclose(dsnu_zero, 0.02)
    assert np.allclose(prnu_zero, 1.4)


def test_noise_column_fpn_matches_legacy_formula_and_vector_override(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=float)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "column fpn", np.array([0.02, 0.05], dtype=float))

    noisy_a, dsnu_a, prnu_a = noiseColumnFPN(sensor, seed=11)
    noisy_b, dsnu_b, prnu_b = noiseColumnFPN(sensor, seed=11)

    assert np.array_equal(dsnu_a, dsnu_b)
    assert np.array_equal(prnu_a, prnu_b)
    assert np.allclose(noisy_a, (volts * prnu_a.reshape(1, -1)) + dsnu_a.reshape(1, -1))
    assert np.allclose(noisy_a, noisy_b)

    sensor = sensor_set(sensor, "col offset fpn vector", np.array([0.01, 0.02, 0.03, 0.04], dtype=float))
    sensor = sensor_set(sensor, "col gain fpn vector", np.array([1.0, 0.9, 1.1, 1.2], dtype=float))
    noisy_stored, dsnu_stored, prnu_stored = noiseColumnFPN(sensor, seed=5)
    assert np.allclose(dsnu_stored, np.array([0.01, 0.02, 0.03, 0.04], dtype=float))
    assert np.allclose(prnu_stored, np.array([1.0, 0.9, 1.1, 1.2], dtype=float))
    assert np.allclose(noisy_stored, (volts * prnu_stored.reshape(1, -1)) + dsnu_stored.reshape(1, -1))


def test_sensor_read_color_filters_matches_legacy_special_cases(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)

    rgb_spectra, rgb_names = sensorReadColorFilters(sensor, "rgb", asset_store=asset_store)
    expected_rgb, expected_rgb_names, _ = ie_read_color_filter(wave, "data/sensor/colorfilters/RGB.mat", asset_store=asset_store)
    monochrome_spectra, monochrome_names = sensorReadColorFilters(sensor, "monochrome", asset_store=asset_store)
    xyz_spectra, xyz_names = sensorReadColorFilters(sensor, "xyz", asset_store=asset_store)

    assert np.allclose(rgb_spectra, expected_rgb)
    assert rgb_names == expected_rgb_names
    assert np.allclose(monochrome_spectra, np.ones((wave.size, 1), dtype=float))
    assert monochrome_names == ["w"]
    assert xyz_spectra.shape == (wave.size, 3)
    assert xyz_names == ["rX", "gY", "bZ"]


def test_sensor_read_filter_updates_legacy_sensor_filter_slots(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)

    cfa_updated = sensorReadFilter("colorfilters", sensor, "grbc", asset_store=asset_store)
    expected_grbc, expected_grbc_names = sensorReadColorFilters(sensor, "grbc", asset_store=asset_store)
    qe_updated = sensorReadFilter("pdspectralqe", sensor, "data/sensor/colorfilters/W.mat", asset_store=asset_store)
    expected_qe, _, _ = ie_read_color_filter(wave, "data/sensor/colorfilters/W.mat", asset_store=asset_store)
    ir_updated = sensorReadFilter("irfilter", sensor, "data/sensor/irfilters/ircf_public.mat", asset_store=asset_store)
    expected_ir, _, _ = ie_read_color_filter(wave, "data/sensor/irfilters/ircf_public.mat", asset_store=asset_store)

    assert np.allclose(np.asarray(sensor_get(cfa_updated, "filter spectra"), dtype=float), expected_grbc)
    assert sensor_get(cfa_updated, "filter names") == expected_grbc_names
    assert np.all(np.asarray(sensor_get(cfa_updated, "pattern"), dtype=int) <= expected_grbc.shape[1])
    assert np.allclose(np.asarray(sensor_get(qe_updated, "pixel spectral qe"), dtype=float), expected_qe.reshape(-1))
    assert np.allclose(np.asarray(sensor_get(ir_updated, "ir filter"), dtype=float), expected_ir.reshape(-1))


def test_sensor_filter_edit_helpers_match_legacy_array_updates(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    original_pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    w_filter, w_names, _ = ie_read_color_filter(
        np.asarray(sensor_get(sensor, "wave"), dtype=float),
        "data/sensor/colorfilters/W.mat",
        asset_store=asset_store,
    )

    added = sensorAddFilter(sensor, "data/sensor/colorfilters/W.mat", asset_store=asset_store)
    replaced = sensorReplaceFilter(added, 2, "data/sensor/colorfilters/W.mat", new_filter_name="white", asset_store=asset_store)
    deleted = sensorDeleteFilter(replaced, 2)

    added_spectra = np.asarray(sensor_get(added, "filter spectra"), dtype=float)
    assert added_spectra.shape[1] == int(sensor_get(sensor, "nfilters")) + 1
    assert np.allclose(added_spectra[:, -1], w_filter.reshape(-1))
    assert sensor_get(added, "filter names")[-1] == w_names[0]

    replaced_spectra = np.asarray(sensor_get(replaced, "filter spectra"), dtype=float)
    assert np.allclose(replaced_spectra[:, 1], w_filter.reshape(-1))
    assert sensor_get(replaced, "filter names")[1] == "white"
    assert np.array_equal(np.asarray(sensor_get(replaced, "pattern"), dtype=int), np.asarray(sensor_get(added, "pattern"), dtype=int))

    expected_pattern = original_pattern.copy()
    mask = expected_pattern >= 2
    expected_pattern[mask] = np.maximum(1, expected_pattern[mask] - 1)
    assert int(sensor_get(deleted, "nfilters")) == int(sensor_get(sensor, "nfilters"))
    assert np.array_equal(np.asarray(sensor_get(deleted, "pattern"), dtype=int), expected_pattern)


def test_pixel_create_get_set_and_ideal_match_legacy_contract() -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)
    default_wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    pixel = pixelCreate("default", wave, 1.1e-6)
    default_wave_pixel = pixelCreate("human", [])
    default_pixel = pixelCreate()
    empty_type_pixel = pixelCreate(np.array([], dtype=float), wave)
    blank_type_pixel = pixelCreate("", wave)

    assert pixelGet(pixel, "name") == "aps"
    assert pixelGet(pixel, "type") == "pixel"
    assert np.isclose(float(pixelGet(pixel, "width")), 2.8e-6)
    assert np.isclose(float(pixelGet(pixel, "fill factor")), 0.75)
    assert np.array_equal(np.asarray(pixelGet(pixel, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(pixelGet(pixel, "spectralQE"), dtype=float), np.ones(wave.size, dtype=float))
    assert np.array_equal(np.asarray(pixelGet(default_wave_pixel, "wave"), dtype=float), default_wave)
    assert pixelGet(empty_type_pixel, "name") == pixelGet(default_pixel, "name")
    assert pixelGet(blank_type_pixel, "name") == pixelGet(default_pixel, "name")
    assert np.array_equal(np.asarray(pixelGet(empty_type_pixel, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(pixelGet(blank_type_pixel, "wave"), dtype=float), wave)

    resized = pixelSet(pixel, "size same fill factor", np.array([1.5e-6, 1.5e-6], dtype=float))
    noisy = pixelSet(resized, "readNoiseVolts", 2.0e-3)
    noisy = pixelSet(noisy, "darkVoltage", 3.0e-3)
    idealized = pixelIdeal(noisy)
    created_ideal = pixelCreate("ideal", wave, 1.5e-6)
    created_ideal_rect = pixelCreate("ideal", wave, np.array([1.5e-6, 1.8e-6], dtype=float))
    created_ideal_default_wave = pixelCreate("ideal", [], 1.5e-6)

    assert np.isclose(float(pixelGet(resized, "width")), 1.5e-6)
    assert np.isclose(float(pixelGet(resized, "height")), 1.5e-6)
    assert np.isclose(float(pixelGet(resized, "fill factor")), float(pixelGet(pixel, "fill factor")))
    assert np.isclose(float(pixelGet(noisy, "readNoiseVolts")), 2.0e-3)
    assert np.isclose(float(pixelGet(idealized, "readNoiseVolts")), 0.0)
    assert np.isclose(float(pixelGet(idealized, "darkVoltage")), 0.0)
    assert np.isclose(float(pixelGet(idealized, "voltage swing")), 1.0e6)
    assert np.isclose(float(pixelGet(created_ideal, "width")), 1.5e-6)
    assert np.isclose(float(pixelGet(created_ideal, "pdwidth")), 1.5e-6)
    assert np.isclose(float(pixelGet(created_ideal, "pdheight")), 1.5e-6)
    assert np.isclose(float(pixelGet(created_ideal, "fill factor")), 1.0)
    assert np.isclose(float(pixelGet(created_ideal_rect, "height")), 1.5e-6)
    assert np.isclose(float(pixelGet(created_ideal_rect, "width")), 1.8e-6)
    assert np.isclose(float(pixelGet(created_ideal_rect, "pdheight")), 1.5e-6)
    assert np.isclose(float(pixelGet(created_ideal_rect, "pdwidth")), 1.8e-6)
    assert np.array_equal(np.asarray(pixelGet(created_ideal_default_wave, "wave"), dtype=float), default_wave)


def test_pixel_sr_matches_closed_form_responsivity() -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)
    qe = np.array([0.2, 0.6, 1.0], dtype=float)
    pixel = pixelCreate("human", wave)
    pixel = pixelSet(pixel, "spectralQE", qe)

    sr = pixelSR(pixel)
    q = 1.602177e-19
    h = 6.62607015e-34
    c = 2.99792458e8
    expected = ((wave * 1e-9 * q) / (h * c)) * qe

    assert np.allclose(sr, expected)


def test_ie_pixel_well_capacity_matches_asset_interpolation(asset_store) -> None:
    electrons, table = iePixelWellCapacity(2.35, asset_store=asset_store)
    empty_electrons, empty_table = iePixelWellCapacity([], asset_store=asset_store)

    reference_table = np.asarray(asset_store.load_mat("data/sensor/wellCapacity.mat")["wellCapacity"], dtype=float)
    expected = np.interp(2.35, reference_table[:, 0], reference_table[:, 1])
    left_slope = (reference_table[1, 1] - reference_table[0, 1]) / (reference_table[1, 0] - reference_table[0, 0])
    extrapolated, _ = iePixelWellCapacity(0.05, asset_store=asset_store)
    expected_extrapolated = reference_table[0, 1] + (0.05 - reference_table[0, 0]) * left_slope

    assert np.isclose(float(electrons), expected)
    assert empty_electrons is None
    assert np.allclose(table, reference_table)
    assert np.allclose(empty_table, reference_table)
    assert np.isclose(float(extrapolated), expected_extrapolated)


def test_pixel_pd_helper_wrappers_match_legacy_geometry_contract(asset_store) -> None:
    pixel = pixelCreate("default", np.array([450.0, 550.0, 650.0], dtype=float))
    centered = pixelPositionPD(pixel, "center")
    corner = pixelPositionPD(pixel, "corner")

    expected_center_x = (float(pixelGet(pixel, "width")) - float(pixelGet(pixel, "pdwidth"))) / 2.0
    expected_center_y = (float(pixelGet(pixel, "height")) - float(pixelGet(pixel, "pdheight"))) / 2.0

    assert np.isclose(float(pixelGet(centered, "pdxpos")), expected_center_x)
    assert np.isclose(float(pixelGet(centered, "pdypos")), expected_center_y)
    assert np.isclose(float(pixelGet(corner, "pdxpos")), 0.0)
    assert np.isclose(float(pixelGet(corner, "pdypos")), 0.0)

    centered_fill = pixelCenterFillPD(pixel, 0.25)
    assert np.isclose(float(pixelGet(centered_fill, "fillfactor")), 0.25)
    assert np.isclose(float(pixelGet(centered_fill, "pdwidth")), np.sqrt(0.25) * float(pixelGet(pixel, "deltax")))
    assert np.isclose(float(pixelGet(centered_fill, "pdheight")), np.sqrt(0.25) * float(pixelGet(pixel, "deltay")))
    assert np.isclose(float(pixelGet(centered_fill, "pdxpos")), (float(pixelGet(pixel, "width")) - float(pixelGet(centered_fill, "pdwidth"))) / 2.0)
    assert np.isclose(float(pixelGet(centered_fill, "pdypos")), (float(pixelGet(pixel, "height")) - float(pixelGet(centered_fill, "pdheight"))) / 2.0)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel", pixel)
    sensor = pixelCenterFillPD(sensor, 0.36)
    assert np.isclose(float(pixelGet(sensor_get(sensor, "pixel"), "fillfactor")), 0.36)

    summary = pixelDescription(sensor_get(sensor, "pixel"), sensor)
    assert "Pixel (H,W):" in summary
    assert "PD (H,W):" in summary
    assert "Fill percentage:" in summary
    assert "Well capacity" in summary
    assert "DR (1 ms):" in summary
    assert "Peak SNR:" in summary


def test_pixel_transmittance_helpers_match_legacy_fresnel_contract() -> None:
    theta_in = np.array([0.0, 0.2], dtype=float)
    n_stack = np.array([1.0, 1.5], dtype=float)

    theta_out = ptSnellsLaw(n_stack, theta_in)
    expected_theta_out = np.arcsin(n_stack[0] * np.sin(theta_in) / n_stack[1])
    assert theta_out.shape == (1, 2, 2)
    assert np.allclose(np.real(theta_out[0, 1, :]), expected_theta_out)

    rho_s, tau_s = ptReflectionAndTransmission(1.0, 1.5, theta_in, "s")
    rho_p, tau_p = ptReflectionAndTransmission(1.0, 1.5, theta_in, "p")
    assert np.isclose(float(np.real(rho_s[0])), (1.0 - 1.5) / (1.0 + 1.5))
    assert np.isclose(float(np.real(tau_s[0])), 2.0 / (1.0 + 1.5))
    assert np.isclose(float(np.real(rho_p[0])), (1.5 - 1.0) / (1.5 + 1.0))
    assert np.isclose(float(np.real(tau_p[0])), 2.0 / (1.5 + 1.0))

    interface = ptInterfaceMatrix(rho_s, tau_s)
    propagation = ptPropagationMatrix(1.5, 2.0e-6, theta_out[0, 1, :], 550.0e-9)
    assert interface.shape == (2, 2, theta_in.size)
    assert propagation.shape == (2, 2, theta_in.size)
    assert np.allclose(propagation[0, 0, :] * propagation[1, 1, :], np.ones(theta_in.size, dtype=complex))


def test_pt_transmittance_stack_returns_unity_for_index_matched_layers() -> None:
    n_stack = np.array([1.0, 1.0, 1.0], dtype=float)
    thickness = np.array([1.0e-6, 2.0e-6], dtype=float)
    theta = np.linspace(-0.2, 0.2, 5, dtype=float)
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    scattering = ptScatteringMatrix(n_stack, thickness, theta, 550.0e-9, "s")
    poynting = ptPoyntingFactor(n_stack, theta)
    tunnel = ptTransmittance(n_stack, thickness, wave, theta)

    assert scattering.shape == (2, 2, theta.size)
    assert np.allclose(np.abs(1.0 / scattering[0, 0, :]) ** 2, np.ones(theta.size, dtype=float))
    assert np.allclose(np.real(poynting), np.ones(theta.size, dtype=float))
    assert np.allclose(tunnel["transmission"]["spectral"], np.ones(wave.size, dtype=float))
    assert np.allclose(tunnel["transmission"]["average"], np.ones(theta.size, dtype=float))


def test_pixel_transmittance_scales_optical_image_by_spectral_tunnel_average() -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    photons = np.stack(
        [
            np.full((2, 3), 1.0, dtype=float),
            np.full((2, 3), 2.0, dtype=float),
            np.full((2, 3), 3.0, dtype=float),
        ],
        axis=2,
    )
    oi = oi_set(oi_set(oi_create(), "wave", wave), "photons", photons)
    pixel = pixelCreate("human", wave)
    optics = dict(oi.fields["optics"])
    incidence_angles = np.linspace(-np.arctan(1.0 / (2.0 * float(optics["f_number"]))), np.arctan(1.0 / (2.0 * float(optics["f_number"]))), 25)
    tunnel = ptTransmittance(pixelGet(pixel, "refractiveindex"), pixelGet(pixel, "layerthickness"), wave, incidence_angles)

    filtered_oi, returned_pixel, returned_optics = pixelTransmittance(oi, pixel, optics)
    expected = photons * np.asarray(tunnel["transmission"]["spectral"], dtype=float).reshape(1, 1, -1)

    assert np.allclose(oi_get(filtered_oi, "photons"), expected)
    assert np.allclose(filtered_oi.fields["pixel_transmittance"]["transmission"]["spectral"], tunnel["transmission"]["spectral"])
    assert np.array_equal(np.asarray(pixelGet(returned_pixel, "wave"), dtype=float), wave)
    assert np.isclose(float(returned_optics["f_number"]), float(optics["f_number"]))


def test_pixel_vignetting_and_binning_wrappers_match_legacy_contracts(asset_store) -> None:
    optics = opticsCreate(asset_store=asset_store)
    pixel = pixelCreate("default", np.array([500.0, 600.0], dtype=float))

    overlap, num_grid, num_grid_spot = pvFullOverlap(optics, pixel)
    focal_length = float(opticsGet(optics, "focal length"))
    diameter = float(opticsGet(optics, "aperture diameter"))
    width = float(pixelGet(pixel, "width"))
    depth = float(pixelGet(pixel, "depth"))
    spot_diameter = diameter * (depth / focal_length)

    expected_num_grid = int(np.floor(51.0 * width / spot_diameter))
    if expected_num_grid % 2 == 0:
        expected_num_grid += 1
    diode_x, diode_y = np.meshgrid(
        np.linspace(-width / 2.0, width / 2.0, expected_num_grid, dtype=float),
        np.linspace(-width / 2.0, width / 2.0, expected_num_grid, dtype=float),
    )
    diode_mask = (np.abs(diode_x) < (width / 2.0)) & (np.abs(diode_y) < (width / 2.0))
    diode_grid = diode_mask.astype(float)
    diode_grid /= float(np.count_nonzero(diode_mask))

    expected_num_grid_spot = int(np.floor(spot_diameter / (width / expected_num_grid)))
    if expected_num_grid_spot % 2 == 0:
        expected_num_grid_spot += 1
    spot_x, spot_y = np.meshgrid(
        np.linspace(-spot_diameter / 2.0, spot_diameter / 2.0, expected_num_grid_spot, dtype=float),
        np.linspace(-spot_diameter / 2.0, spot_diameter / 2.0, expected_num_grid_spot, dtype=float),
    )
    spot_mask = np.sqrt(spot_x**2 + spot_y**2) < (spot_diameter / 2.0)
    spot = spot_mask.astype(float)
    spot /= float(np.count_nonzero(spot_mask))
    expected_overlap = convolve2d(diode_grid, spot, mode="full")

    assert num_grid == expected_num_grid
    assert num_grid_spot == expected_num_grid_spot
    np.testing.assert_allclose(overlap, expected_overlap, rtol=1e-10, atol=1e-12)

    diode_location = np.array([[0.25e-6, -0.15e-6]], dtype=float)
    outside_angle = np.array([[0.05, -0.03]], dtype=float)
    ratio = float(pvReduction(overlap, num_grid, num_grid_spot, diode_location, outside_angle, optics, pixel))
    offset_x = diode_location[0, 0] - (focal_length - depth) * np.tan(outside_angle[0, 0])
    offset_y = diode_location[0, 1] - (focal_length - depth) * np.tan(outside_angle[0, 1])
    index_offset_x = int(np.rint((offset_x / width) * num_grid))
    index_offset_y = int(np.rint((offset_y / width) * num_grid))
    index_center_x = ((expected_overlap.shape[1] - 1) / 2.0) + 1.0
    index_center_y = ((expected_overlap.shape[0] - 1) / 2.0) + 1.0
    support = np.arange(-(num_grid - 1) / 2.0, ((num_grid - 1) / 2.0) + 1.0, dtype=float)
    index_range_x = np.rint(index_center_x - index_offset_x + support).astype(int)
    index_range_y = np.rint(index_center_y - index_offset_y + support).astype(int)
    index_range_x[index_range_x <= 0] = 1
    index_range_y[index_range_y <= 0] = 1
    index_range_x[index_range_x > expected_overlap.shape[1]] = 1
    index_range_y[index_range_y > expected_overlap.shape[0]] = 1
    expected_ratio = float(np.sum(expected_overlap[np.ix_(index_range_x - 1, index_range_y - 1)]))
    assert np.isclose(ratio, expected_ratio)

    sensor = sensor_create(asset_store=asset_store)
    volts = np.arange(1.0, 65.0, dtype=float).reshape(8, 8) / 64.0
    dv = np.arange(1.0, 65.0, dtype=float).reshape(8, 8)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "digital values", dv)

    kodak = binPixel(sensor, "kodak2008")
    expected_kodak = np.zeros((8, 4), dtype=float)
    transform = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    for row in range(0, 8, 4):
        for col in range(0, 8, 4):
            expected_kodak[row : row + 4, col // 2 : col // 2 + 2] = volts[row : row + 4, col : col + 4] @ transform
    np.testing.assert_allclose(np.asarray(sensor_get(kodak, "dv"), dtype=float), expected_kodak, rtol=1e-10, atol=1e-12)

    add_adjacent = binPixel(sensor, "addAdjacentBlocks")
    expected_add = np.zeros((4, 4), dtype=float)
    left = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=float)
    for row in range(0, 8, 4):
        for col in range(0, 8, 4):
            expected_add[row // 2 : row // 2 + 2, col // 2 : col // 2 + 2] = left @ volts[row : row + 4, col : col + 4] @ left.T
    np.testing.assert_allclose(np.asarray(sensor_get(add_adjacent, "dv"), dtype=float), expected_add, rtol=1e-10, atol=1e-12)

    kodak_post = binPixelPost(kodak, "kodak2008")
    expected_kodak_post = np.zeros((4, 4), dtype=float)
    for row in range(0, 8, 4):
        for col in range(0, 4, 2):
            expected_kodak_post[row // 2 : row // 2 + 2, col : col + 2] = np.rint(0.5 * (left @ expected_kodak[row : row + 4, col : col + 2]))
    np.testing.assert_allclose(np.asarray(sensor_get(kodak_post, "dv"), dtype=float), expected_kodak_post, rtol=1e-10, atol=1e-12)

    averaged = binPixelPost(sensor, "averageAdjacentDigitalblocks")
    expected_averaged = np.zeros((4, 4), dtype=float)
    for row in range(0, 8, 4):
        for col in range(0, 8, 4):
            expected_averaged[row // 2 : row // 2 + 2, col // 2 : col // 2 + 2] = 0.25 * (left @ dv[row : row + 4, col : col + 4] @ left.T)
    np.testing.assert_allclose(np.asarray(sensor_get(averaged, "dv"), dtype=float), expected_averaged, rtol=1e-10, atol=1e-12)


def test_sensor_show_image_matches_sensor_rgb_rendering(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.full((4, 4), 0.25, dtype=float)
    dv = np.arange(1, 17, dtype=float).reshape(4, 4)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dv", dv)

    rendered = sensorShowImage(sensor, 0.8, True, 0)
    expected = sensor_get(sensor, "rgb", "dv or volts", 0.8, True)

    assert rendered is not None
    assert np.allclose(np.asarray(rendered, dtype=float), np.asarray(expected, dtype=float))
    assert np.asarray(rendered, dtype=float).shape == (4, 4, 3)


def test_sensor_save_image_appends_png_and_normalizes_rendering(asset_store, tmp_path) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 16.0
    sensor = sensor_set(sensor, "volts", volts)

    saved_path = sensorSaveImage(sensor, tmp_path / "sensor_capture", "volts", 1.0, False)
    rendered = np.asarray(sensor_get(sensor, "rgb", "volts", 1.0, False), dtype=float)
    expected = np.clip(rendered / np.max(rendered), 0.0, 1.0)
    written = np.asarray(iio.imread(saved_path), dtype=float) / 255.0

    assert saved_path.endswith(".png")
    assert np.allclose(written, expected, atol=1.0 / 255.0)


def test_sensor_utility_wrappers_match_legacy_geometry_and_file_contract(asset_store, tmp_path) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.full((4, 4), 0.25, dtype=float))

    rescaled = sensorRescale(sensor, sensor_formats("qqcif"), sensor_formats("quarterinch"))
    pixel = sensor_get(rescaled, "pixel")

    assert tuple(np.asarray(sensor_get(rescaled, "size"), dtype=int)) == (72, 88)
    assert np.isclose(float(sensor_get(rescaled, "height")), 0.0024)
    assert np.isclose(float(sensor_get(rescaled, "width")), 0.0032)
    assert np.isclose(float(pixelGet(pixel, "widthGap")), 0.0)
    assert np.isclose(float(pixelGet(pixel, "heightGap")), 0.0)
    assert np.isclose(float(pixelGet(pixel, "width")), 0.0032 / 88.0)
    assert np.isclose(float(pixelGet(pixel, "height")), 0.0024 / 72.0)
    assert np.isclose(float(pixelGet(pixel, "fill factor")), 0.5)
    assert sensor_get(rescaled, "volts") is None
    assert sensor_get(rescaled, "dv") is None

    cfa_path = sensorCfaSave(rescaled, tmp_path / "saved_cfa")
    cfa_payload = loadmat(cfa_path, squeeze_me=True, struct_as_record=False)
    filter_names = [str(name) for name in np.asarray(cfa_payload["color"].filterNames, dtype=object).reshape(-1).tolist()]

    assert cfa_path.endswith(".mat")
    assert {"cfa", "color", "spectrum"} <= set(cfa_payload)
    assert all((not name) or name[0].islower() for name in filter_names)
    assert np.array_equal(np.asarray(cfa_payload["cfa"].pattern, dtype=int), np.asarray(sensor_get(rescaled, "pattern"), dtype=int))
    assert np.array_equal(
        np.asarray(cfa_payload["spectrum"].wave, dtype=float).reshape(-1),
        np.asarray(sensor_get(rescaled, "wave"), dtype=float).reshape(-1),
    )

    saved_sensor_path = Path(vcSaveObject(rescaled, tmp_path / "saved_sensor"))
    loaded = sensorFromFile(saved_sensor_path)
    isa_payload = loadmat(saved_sensor_path, squeeze_me=True, struct_as_record=False)["isa"]
    isa_alias_path = tmp_path / "saved_sensor_isa_alias.mat"
    savemat(isa_alias_path, {"isa_": isa_payload})
    loaded_isa_alias = sensorFromFile(isa_alias_path)

    for restored in (loaded, loaded_isa_alias):
        assert tuple(np.asarray(sensor_get(restored, "size"), dtype=int)) == (72, 88)
        assert np.array_equal(np.asarray(sensor_get(restored, "pattern"), dtype=int), np.asarray(sensor_get(rescaled, "pattern"), dtype=int))
        assert sensor_get(restored, "filter names") == sensor_get(rescaled, "filter names")


def test_load_raw_sensor_data_matches_legacy_8bit_and_10bit_file_contract(tmp_path) -> None:
    raw8 = np.array([0, 5, 127, 255], dtype=np.uint8)
    raw8_path = tmp_path / "sensor8.raw"
    raw8.tofile(raw8_path)
    np.testing.assert_array_equal(LoadRawSensorData(raw8_path), raw8)

    raw10_values = np.array([0x0001, 0x0123, 0x03FF], dtype=np.uint16)
    raw10_le_path = tmp_path / "sensor10_le.raw"
    raw10_be_path = tmp_path / "sensor10_be.raw"
    raw10_values.astype("<u2").tofile(raw10_le_path)
    raw10_values.astype(">u2").tofile(raw10_be_path)

    np.testing.assert_array_equal(LoadRawSensorData(raw10_le_path, 10, "little", 3, 1), raw10_values)
    np.testing.assert_array_equal(LoadRawSensorData(raw10_be_path, 10, "big"), raw10_values)

    with pytest.raises(ValueError, match="Bad bpp"):
        LoadRawSensorData(raw10_le_path, 12)


def test_sensor_show_cfa_matches_plot_sensor_wrappers(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)

    default_payload, _ = sensor_plot(sensor, "cfa")
    full_payload, _ = sensor_plot(sensor, "cfafull")
    default_fig, default_img = sensorShowCFA(sensor)
    full_fig, full_img = sensorShowCFA(sensor, None, [2, 2], 8)

    assert default_fig is None
    assert full_fig is None
    assert np.allclose(np.asarray(default_img, dtype=float), np.asarray(default_payload["img"], dtype=float))
    assert np.allclose(np.asarray(full_img, dtype=float), np.asarray(full_payload["img"], dtype=float))


def test_sensor_check_array_matches_sensor_show_cfa_render(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)

    check_img = sensorCheckArray(sensor, 4)
    _, expected_img = sensorShowCFA(sensor, None, None, 4)

    assert np.allclose(np.asarray(check_img, dtype=float), np.asarray(expected_img, dtype=float))


def test_sensor_rgb_to_plane_matches_legacy_cfa_assignment() -> None:
    rgb_data = np.arange(1, 1 + (5 * 4 * 3), dtype=float).reshape(5, 4, 3)
    cfa_pattern = np.array([[1, 2], [2, 3]], dtype=int)

    sensor_plane, dummy_sensor = sensorRGB2Plane(rgb_data, cfa_pattern)

    expected_rgb = rgb_data[:4, :4, :]
    tiled_pattern = tile_pattern(cfa_pattern, 4, 4)
    expected_plane = np.zeros((4, 4), dtype=float)
    for band_index in range(expected_rgb.shape[2]):
        selector = tiled_pattern == (band_index + 1)
        expected_plane[selector] = expected_rgb[:, :, band_index][selector]

    assert np.allclose(sensor_plane, expected_plane)
    assert np.array_equal(np.asarray(sensor_get(dummy_sensor, "pattern"), dtype=int), cfa_pattern)
    assert tuple(np.asarray(sensor_get(dummy_sensor, "size"), dtype=int)) == (4, 4)
    assert sensor_get(dummy_sensor, "filter names") == ["r", "g", "b"]


def test_sensor_stats_matches_legacy_roi_summary_paths(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 10.0
    dv = np.arange(10, 26, dtype=float).reshape(4, 4)
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dv", dv)
    roi = np.array([2, 2, 1, 1], dtype=int)

    basic_stats, updated = sensorStats(sensor, "basic", "volts", roi, True)
    mean_stats, updated_mean = sensorStats(sensor, "mean", "dv", roi, True)
    roi_volts = np.asarray(sensor_get(updated, "roi volts"), dtype=float)
    roi_dv = np.asarray(sensor_get(updated_mean, "roi dv"), dtype=float)

    expected_mean = []
    expected_std = []
    expected_sem = []
    for index in range(roi_volts.shape[1]):
        column = roi_volts[:, index]
        column = column[np.isfinite(column)]
        expected_mean.append(np.mean(column))
        if column.size > 1:
            expected_std.append(np.std(column, ddof=1))
            expected_sem.append(np.std(column, ddof=1) / np.sqrt(column.size - 1))
        else:
            expected_std.append(0.0)
            expected_sem.append(0.0)

    expected_dv_mean = []
    for index in range(roi_dv.shape[1]):
        column = roi_dv[:, index]
        column = column[np.isfinite(column)]
        expected_dv_mean.append(np.mean(column))

    assert np.allclose(np.asarray(basic_stats["mean"], dtype=float), np.asarray(expected_mean, dtype=float))
    assert np.allclose(np.asarray(basic_stats["std"], dtype=float), np.asarray(expected_std, dtype=float))
    assert np.allclose(np.asarray(basic_stats["sem"], dtype=float), np.asarray(expected_sem, dtype=float))
    assert int(basic_stats["N"]) == int(np.sum(np.isfinite(roi_volts[:, 0])))
    assert np.allclose(np.asarray(mean_stats, dtype=float), np.asarray(expected_dv_mean, dtype=float))


def test_sensor_image_color_array_matches_legacy_color_order() -> None:
    cfa = np.array([["r", "g", "b"], ["c", "y", "o"]], dtype="<U1")

    cfa_numbers, cfa_map = sensorImageColorArray(cfa)

    assert np.array_equal(cfa_numbers, np.array([[1, 2, 3], [4, 5, 12]], dtype=int))
    assert cfa_map.shape == (13, 3)
    assert np.allclose(cfa_map[0], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(cfa_map[7], np.array([0.3, 0.3, 0.3], dtype=float))
    assert np.allclose(cfa_map[11], np.array([1.0, 0.6, 0.0], dtype=float))


def test_sensor_color_order_matches_legacy_hint_map() -> None:
    ordering_cell, cfa_map = sensorColorOrder()
    ordering_string, string_map = sensorColorOrder("string")

    assert ordering_cell == ["r", "g", "b", "c", "y", "m", "w", "i", "u", "x", "z", "o", "k"]
    assert ordering_string == "rgbcymwiuxzok"
    assert np.allclose(string_map, cfa_map)

    numbers, remapped = sensorImageColorArray(np.array([list(ordering_string)], dtype="<U1"))
    assert np.array_equal(numbers, np.arange(1, 14, dtype=int).reshape(1, -1))
    assert np.allclose(remapped, cfa_map)


def test_sensor_determine_cfa_tiles_legacy_pattern_and_colors(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 3)
    sensor = sensor_set(sensor, "cols", 5)
    rows = int(sensor_get(sensor, "rows"))
    cols = int(sensor_get(sensor, "cols"))

    cfa_letters, cfa_numbers, cfa_map = sensorDetermineCFA(sensor)

    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    expected_numbers = tile_pattern(pattern, rows, cols)
    pattern_colors = np.asarray(sensor_get(sensor, "pattern colors"), dtype="<U1")
    expected_letters = np.tile(
        pattern_colors,
        (
            int(np.ceil(rows / pattern_colors.shape[0])),
            int(np.ceil(cols / pattern_colors.shape[1])),
        ),
    )[:rows, :cols]
    ordering_string, ordering_map = sensorColorOrder("string")
    filter_letters = str(sensor_get(sensor, "filter color letters"))
    expected_map = np.vstack([ordering_map[ordering_string.index(letter.lower())] for letter in filter_letters])

    assert np.array_equal(cfa_numbers, expected_numbers)
    assert np.array_equal(cfa_letters, expected_letters)
    assert np.allclose(cfa_map, expected_map)


def test_sensor_show_cfa_weights_matches_legacy_weighted_render(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    weights = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.0],
            [1.0, 0.5, 0.25],
        ],
        dtype=float,
    )
    c_pos = [1, 2]

    weighted_img = sensorShowCFAWeights(weights, sensor, c_pos, "imgScale", 1)

    unit_letters = np.asarray(sensor_get(sensor, "pattern colors"), dtype="<U1")
    row_offsets = np.arange(weights.shape[0], dtype=int) - (weights.shape[0] // 2)
    col_offsets = np.arange(weights.shape[1], dtype=int) - (weights.shape[1] // 2)
    row_indices = np.mod(int(c_pos[0]) - 1 + row_offsets, unit_letters.shape[0])
    col_indices = np.mod(int(c_pos[1]) - 1 + col_offsets, unit_letters.shape[1])
    cfa_letters = unit_letters[np.ix_(row_indices, col_indices)]
    cfa_numbers, cfa_map = sensorImageColorArray(cfa_letters)

    expected = np.zeros(weights.shape + (3,), dtype=float)
    valid = cfa_numbers > 0
    normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    expected[valid] = cfa_map[cfa_numbers[valid] - 1] * normalized[valid, np.newaxis]

    assert np.allclose(np.asarray(weighted_img, dtype=float), expected)
    assert np.allclose(weighted_img[1, 1], expected[1, 1])


def test_sensor_snr_luxsec_matches_manual_reparameterization(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)

    snr, luxsec = sensorSNRluxsec(sensor, asset_store=asset_store)
    expected_snr, volts, *_ = sensorSNR(sensor)
    volts_per_lux_sec, *_ = pixel_v_per_lux_sec(sensor, asset_store=asset_store)
    expected_luxsec = volts.reshape(-1, 1) / np.asarray(volts_per_lux_sec, dtype=float).reshape(1, -1)

    assert snr.shape == expected_snr.shape == (50,)
    assert luxsec.shape == expected_luxsec.shape
    assert luxsec.shape[1] == int(sensor_get(sensor, "ncolors"))
    assert np.allclose(snr, expected_snr)
    assert np.allclose(luxsec, expected_luxsec)
    assert np.all(np.diff(luxsec, axis=0) > 0.0)


def _legacy_sensor_color_block_matrix(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float).reshape(-1)
    default_wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    blue_count = 10
    green_count = 8
    red_count = default_wave.size - blue_count - green_count
    default_matrix = np.column_stack(
        (
            np.concatenate((np.zeros(blue_count + green_count), np.ones(red_count))),
            np.concatenate((np.zeros(blue_count), np.ones(green_count), np.zeros(red_count))),
            np.concatenate((np.ones(blue_count), np.zeros(green_count + red_count))),
        )
    )
    if np.array_equal(wave, default_wave):
        matrix = default_matrix.copy()
    else:
        matrix = np.empty((wave.size, 3), dtype=float)
        for index in range(3):
            matrix[:, index] = np.interp(wave, default_wave, default_matrix[:, index], left=0.2, right=0.2)
    matrix = matrix / np.maximum(np.sum(matrix, axis=0, keepdims=True), 1e-12)
    white_spd = np.asarray(blackbody(wave, 6500.0, kind="quanta"), dtype=float).reshape(-1)
    white_spd = white_spd / max(float(np.max(white_spd)), 1e-12)
    matrix = matrix @ np.diag(1.0 / np.maximum(white_spd @ matrix, 1e-12))
    return matrix


def test_sensor_color_block_matrix_matches_legacy_closed_form() -> None:
    wave = np.arange(410.0, 691.0, 20.0, dtype=float)
    expected = _legacy_sensor_color_block_matrix(wave)

    matrix = sensor_module.colorBlockMatrix(wave, 0.2)

    assert matrix.shape == expected.shape
    assert np.allclose(matrix, expected)


def test_sensor_display_transform_matches_legacy_closed_form(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)

    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    filter_spectra = np.asarray(sensor_get(sensor, "filterSpectra"), dtype=float)
    block_matrix = _legacy_sensor_color_block_matrix(wave)
    expected = (block_matrix.T @ filter_spectra).T
    expected /= max(float(np.max(expected)), 1e-12)

    transform = sensorDisplayTransform(sensor)

    assert transform.shape == expected.shape
    assert np.allclose(transform, expected)
    assert np.isclose(float(np.max(transform)), 1.0)


def test_sensor_equate_transmittances_matches_legacy_scaling() -> None:
    filters = np.array(
        [
            [0.1, 0.4, 0.2],
            [0.3, 0.2, 0.5],
            [0.6, 0.4, 0.3],
        ],
        dtype=float,
    )

    balanced = sensorEquateTransmittances(filters)
    expected = filters / np.sum(filters, axis=0, keepdims=True)
    expected /= max(float(np.max(expected)), 1e-12)

    assert np.allclose(balanced, expected)
    assert np.allclose(np.sum(balanced, axis=0), np.sum(balanced, axis=0)[0])
    assert np.isclose(float(np.max(balanced)), 1.0)


def test_sensor_filter_rgb_matches_legacy_filter_colors(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)

    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    filter_spectra = np.asarray(sensor_get(sensor, "filterSpectra"), dtype=float)
    block_matrix = _legacy_sensor_color_block_matrix(wave)
    background = np.full(3, 0.94, dtype=float)
    expected_full = np.zeros(pattern.shape + (3,), dtype=float)

    for row in range(pattern.shape[0]):
        for col in range(pattern.shape[1]):
            color_filter = filter_spectra[:, int(pattern[row, col]) - 1]
            rgb = np.asarray(block_matrix.T @ color_filter, dtype=float).reshape(-1)
            expected_full[row, col] = rgb / max(float(np.max(rgb)), 1e-12)

    rendered_full = sensorFilterRGB(sensor, 1.0)
    rendered_soft = sensorFilterRGB(sensor, 0.25)

    assert rendered_full.shape == expected_full.shape
    assert np.allclose(rendered_full, expected_full)
    assert np.allclose(rendered_soft, expected_full * 0.25 + background * 0.75)


def test_sensor_set_etendue_scales_noiseless_response(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    baseline_sensor = sensor_create(asset_store=asset_store)
    baseline_sensor = sensor_set(baseline_sensor, "integration time", 0.01)
    baseline_sensor = sensor_set(baseline_sensor, "noise flag", 0)

    attenuated_sensor = sensor_set(
        baseline_sensor.clone(),
        "sensoretendue",
        np.full(baseline_sensor.fields["size"], 0.5, dtype=float),
    )

    baseline = sensor_compute(baseline_sensor, oi, seed=0)
    attenuated = sensor_compute(attenuated_sensor, oi, seed=0)

    assert np.allclose(attenuated.data["volts"], baseline.data["volts"] * 0.5)
    assert np.allclose(sensor_get(attenuated, "sensoretendue"), 0.5)


def test_sensor_compute_applies_legacy_vignetting_modes(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    baseline = sensor_create(asset_store=asset_store)
    baseline = sensor_set(baseline, "integration time", 1.0)
    baseline = sensor_set(baseline, "noise flag", 0)
    vignetted = sensor_set(baseline.clone(), "vignetting", 1)

    baseline = sensor_compute(baseline, oi, seed=0)
    vignetted = sensor_compute(vignetted, oi, seed=0)
    etendue = sensor_get(vignetted, "sensoretendue")

    assert sensor_get(vignetted, "vignettingname") == "bare"
    assert np.asarray(etendue).shape == tuple(sensor_get(vignetted, "size"))
    assert np.all(np.asarray(etendue) <= 1.0)
    assert np.sum(vignetted.data["volts"]) < np.sum(baseline.data["volts"])


def test_sensor_get_fov_uses_scene_distance_when_provided(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    oi = oi_create()
    scene = scene_create(asset_store=asset_store)

    default_fov = float(sensor_get(sensor, "fov", None, oi))
    scene_fov = float(sensor_get(sensor, "fov", scene, oi))

    assert scene_fov < default_fov


def test_sensor_noise_flag_one_includes_fpn(asset_store) -> None:
    scene = scene_create("uniform ee", 16, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    oi.data["photons"][:] = 0.0

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 1.0)
    sensor = sensor_set(sensor, "noise flag", 1)
    sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
    sensor.fields["pixel"]["read_noise_v"] = 0.0
    sensor.fields["pixel"]["dsnu_sigma_v"] = 0.01
    sensor.fields["pixel"]["prnu_sigma"] = 0.0

    noisy = sensor_compute(sensor, oi, seed=0)
    assert np.any(noisy.data["volts"] > 0.0)


def test_sensor_noise_flag_minus_two_keeps_zero_signal_zero(asset_store) -> None:
    scene = scene_create("uniform ee", 16, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    oi.data["photons"][:] = 0.0

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", 1.0)
    sensor = sensor_set(sensor, "noise flag", -2)
    sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.1
    sensor.fields["pixel"]["read_noise_v"] = 0.0
    sensor.fields["pixel"]["dsnu_sigma_v"] = 0.01
    sensor.fields["pixel"]["prnu_sigma"] = 0.25

    noisy = sensor_compute(sensor, oi, seed=0)
    assert np.allclose(noisy.data["volts"], 0.0)


def test_sensor_compute_supersampling_changes_bayer_response(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)

    default_sensor = sensor_create(asset_store=asset_store)
    default_sensor = sensor_set(default_sensor, "integration time", 1.0)
    default_sensor = sensor_set(default_sensor, "noise flag", 0)

    supersampled_sensor = sensor_set(default_sensor.clone(), "n samples per pixel", 3)

    default_result = sensor_compute(default_sensor, oi, seed=0)
    supersampled_result = sensor_compute(supersampled_sensor, oi, seed=0)

    assert default_result.data["volts"].shape == supersampled_result.data["volts"].shape
    assert not np.allclose(default_result.data["volts"], supersampled_result.data["volts"])


def test_ip_compute_default_pipeline(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi_create(), scene, crop=True)
    sensor = sensor_compute(sensor_set(sensor_create(asset_store=asset_store), "noise flag", 0), oi, seed=0)
    ip = ip_compute(ip_create(sensor=sensor, asset_store=asset_store), sensor, asset_store=asset_store)
    assert ip.data["result"].shape[:2] == sensor.fields["size"]
    assert ip.data["result"].shape[2] == 3
    assert np.all((ip.data["result"] >= 0.0) & (ip.data["result"] <= 1.0))


def test_ip_get_set_support_matlab_style_transforms(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    ip = ip_create(sensor=sensor, asset_store=asset_store)

    ip = ip_set(ip, "display dpi", 110)
    ip = ip_set(ip, "render flag", "gray")
    ip = ip_set(ip, "scale display", False)
    ip = ip_set(ip, "sensor conversion matrix", np.eye(3))
    ip = ip_set(ip, "illuminant correction matrix", 2.0 * np.eye(3))
    ip = ip_set(ip, "ics2display transform", 3.0 * np.eye(3))

    assert ip_get(ip, "display dpi") == 110
    assert ip_get(ip, "display spd").shape[1] == 3
    assert ip_get(ip, "render flag") == 3
    assert ip_get(ip, "scale display") is False
    assert np.allclose(ip_get(ip, "combined transform"), 6.0 * np.eye(3))


def test_camera_get_set_routes_matlab_style_subobjects(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)

    camera = camera_set(camera, "sensor integration time", 0.125)
    camera = camera_set(camera, "pixel voltage swing", 1.5)
    camera = camera_set(camera, "pixel size constant fill factor", np.array([1.4e-6, 1.6e-6], dtype=float))
    camera = camera_set(camera, "ip display dpi", 110)
    camera = camera_set(camera, "optics f number", 5.6)

    assert np.isclose(camera_get(camera, "sensor integration time"), 0.125)
    assert np.isclose(camera_get(camera, "pixel voltage swing"), 1.5)
    assert np.allclose(np.asarray(camera_get(camera, "pixel size"), dtype=float), np.array([1.4e-6, 1.6e-6], dtype=float))
    assert camera_get(camera, "ip display dpi") == 110
    assert np.isclose(camera_get(camera, "optics f number"), 5.6)
    assert camera_get(camera, "vci type") == "default"

    camera = camera_set(camera, "auto exposure", True)
    camera = camera_set(camera, "fill factor", 0.5)
    camera = camera_set(camera, "f number", 4.0)
    camera = camera_set(camera, "display dpi", 96)

    assert camera_get(camera, "auto exposure") is True
    assert np.isclose(camera_get(camera, "fill factor"), 0.5)
    assert np.isclose(camera_get(camera, "f number"), 4.0)
    assert camera_get(camera, "display dpi") == 96


def test_camera_compute_end_to_end(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)
    result = camera.fields["ip"].data["result"]
    assert result.shape[:2] == camera.fields["sensor"].fields["size"]
    assert result.shape[2] == 3


def test_camera_get_routes_ip_result_max_and_max_sensor_aliases(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)

    expected_result_max = float(np.max(np.asarray(camera_get(camera, "ip result"), dtype=float)))
    expected_sensor_max = float(np.max(np.asarray(camera_get(camera, "ip input"), dtype=float)))

    assert np.isclose(float(camera_get(camera, "ip result max")), expected_result_max)
    assert np.isclose(float(camera_get(camera, "ip max sensor")), expected_sensor_max)


def test_camera_compute_skips_resize_when_sensor_fov_is_already_close(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_create(asset_store=asset_store)
    sensor = camera.fields["sensor"].clone()
    oi = camera.fields["oi"]

    target_hfov = float(sensor_get(sensor, "fov horizontal", scene, oi))
    target_vfov = float(sensor_get(sensor, "fov vertical", scene, oi))
    scene.fields["fov_deg"] = target_hfov * 1.005
    scene.fields["vfov_deg"] = target_vfov * 0.995
    original_size = sensor.fields["size"]

    camera.fields["sensor"] = sensor
    camera = camera_compute(camera, scene, asset_store=asset_store)

    assert camera.fields["sensor"].fields["size"] == original_size


def test_camera_compute_accepts_empty_mode_placeholder_when_disabling_resize(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.0)
    base_camera = camera_create(asset_store=asset_store)
    original_size = tuple(np.asarray(camera_get(base_camera, "sensor size"), dtype=int))

    explicit_normal = camera_compute(base_camera.clone(), scene, None, False, asset_store=asset_store)
    placeholder_mode = camera_compute(base_camera.clone(), scene, [], False, asset_store=asset_store)
    blank_mode = camera_compute(base_camera.clone(), scene, "", False, asset_store=asset_store)
    whitespace_mode = camera_compute(base_camera.clone(), scene, "   ", False, asset_store=asset_store)

    assert tuple(np.asarray(camera_get(explicit_normal, "sensor size"), dtype=int)) == original_size
    assert tuple(np.asarray(camera_get(placeholder_mode, "sensor size"), dtype=int)) == original_size
    assert tuple(np.asarray(camera_get(blank_mode, "sensor size"), dtype=int)) == original_size
    assert tuple(np.asarray(camera_get(whitespace_mode, "sensor size"), dtype=int)) == original_size
    np.testing.assert_allclose(
        np.asarray(camera_get(placeholder_mode, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_normal, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(blank_mode, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_normal, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(whitespace_mode, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_normal, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


def test_camera_compute_accepts_empty_ptype_placeholder_as_sensor(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    base_camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)

    explicit_sensor = camera_compute(base_camera.clone(), "sensor", asset_store=asset_store)
    placeholder_sensor = camera_compute(base_camera.clone(), [], asset_store=asset_store)
    none_sensor = camera_compute(base_camera.clone(), None, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(camera_get(placeholder_sensor, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_sensor, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(none_sensor, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_sensor, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


def test_camera_compute_accepts_empty_sensor_resize_placeholder_as_default_true(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.0)
    base_camera = camera_create(asset_store=asset_store)

    explicit_default = camera_compute(base_camera.clone(), scene, [], True, asset_store=asset_store)
    placeholder_resize = camera_compute(base_camera.clone(), scene, [], [], asset_store=asset_store)
    none_resize = camera_compute(base_camera.clone(), scene, [], None, asset_store=asset_store)
    blank_resize = camera_compute(base_camera.clone(), scene, [], "", asset_store=asset_store)
    whitespace_resize = camera_compute(base_camera.clone(), scene, [], "   ", asset_store=asset_store)

    assert tuple(np.asarray(camera_get(placeholder_resize, "sensor size"), dtype=int)) == tuple(
        np.asarray(camera_get(explicit_default, "sensor size"), dtype=int)
    )
    assert tuple(np.asarray(camera_get(none_resize, "sensor size"), dtype=int)) == tuple(
        np.asarray(camera_get(explicit_default, "sensor size"), dtype=int)
    )
    assert tuple(np.asarray(camera_get(blank_resize, "sensor size"), dtype=int)) == tuple(
        np.asarray(camera_get(explicit_default, "sensor size"), dtype=int)
    )
    assert tuple(np.asarray(camera_get(whitespace_resize, "sensor size"), dtype=int)) == tuple(
        np.asarray(camera_get(explicit_default, "sensor size"), dtype=int)
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(placeholder_resize, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_default, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(none_resize, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_default, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(blank_resize, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_default, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(whitespace_resize, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_default, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


def test_camera_compute_accepts_empty_string_ptype_placeholder_as_sensor(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    base_camera = camera_compute(camera_create(asset_store=asset_store), scene, asset_store=asset_store)

    explicit_sensor = camera_compute(base_camera.clone(), "sensor", asset_store=asset_store)
    empty_string_sensor = camera_compute(base_camera.clone(), "", asset_store=asset_store)
    whitespace_sensor = camera_compute(base_camera.clone(), "   ", asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(camera_get(empty_string_sensor, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_sensor, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(whitespace_sensor, "ip result"), dtype=float),
        np.asarray(camera_get(explicit_sensor, "ip result"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


def test_camera_parity_case_disables_sensor_noise(asset_store) -> None:
    payload = run_python_case("camera_default_pipeline", asset_store=asset_store)
    assert payload["sensor_volts"].ndim == 2

    scene = scene_create(asset_store=asset_store)
    noiseless_camera = camera_create(asset_store=asset_store)
    noiseless_camera.fields["sensor"] = sensor_set(noiseless_camera.fields["sensor"], "noise flag", 0)
    noiseless_camera = camera_compute(noiseless_camera, scene, asset_store=asset_store)

    assert np.allclose(payload["sensor_volts"], noiseless_camera.fields["sensor"].data["volts"])


def test_camera_create_supports_rgbw_and_rccc_sensor_variants(asset_store) -> None:
    rgbw_camera = camera_create("rgbw", asset_store=asset_store)
    rccc_camera = camera_create("rccc", asset_store=asset_store)

    assert camera_get(rgbw_camera, "sensor filter color letters") == "rgbw"
    assert np.array_equal(
        camera_get(rgbw_camera, "sensor pattern colors"),
        np.array([["r", "g"], ["b", "w"]], dtype="<U1"),
    )
    assert camera_get(rccc_camera, "sensor filter color letters") == "rw"
    assert np.array_equal(
        camera_get(rccc_camera, "sensor pattern colors"),
        np.array([["w", "w"], ["w", "r"]], dtype="<U1"),
    )


def test_camera_create_supports_vendor_sensor_variants(asset_store) -> None:
    mt9v024_rgbw = camera_create("mt9v024", "rgbw", asset_store=asset_store)
    ar0132at_rccc = camera_create("ar0132at", "rccc", asset_store=asset_store)
    mt9v024_multi = camera_create("mt9v024", ["rgb", "mono"], asset_store=asset_store)
    ar0132at_multi = camera_create("ar0132at", ("rgb", "rgbw"), asset_store=asset_store)
    ovt_large = camera_create("ovt-large", asset_store=asset_store)

    assert mt9v024_rgbw.fields["sensor"].name == "MTV9V024-RGBW"
    assert camera_get(mt9v024_rgbw, "sensor filter color letters") == "rgbw"
    assert ar0132at_rccc.fields["sensor"].name == "AR0132AT-RCCC"
    assert camera_get(ar0132at_rccc, "sensor filter color letters") == "rw"
    assert [camera.fields["sensor"].name for camera in mt9v024_multi] == ["MTV9V024-RGB", "MTV9V024-Mono"]
    assert [camera_get(camera, "sensor filter color letters") for camera in mt9v024_multi] == ["rgb", "w"]
    assert [camera.fields["sensor"].name for camera in ar0132at_multi] == ["AR0132AT-RGB", "AR0132AT-RGBW"]
    assert [camera_get(camera, "sensor filter color letters") for camera in ar0132at_multi] == ["rgb", "rgbw"]
    assert [camera.fields["sensor"].name for camera in ovt_large] == ["ovt-LPDLCG", "ovt-LPDHCG"]
    assert mt9v024_multi[0].fields["oi"] is not mt9v024_multi[1].fields["oi"]
    assert mt9v024_multi[0].fields["ip"] is not mt9v024_multi[1].fields["ip"]
    assert ar0132at_multi[0].fields["oi"] is not ar0132at_multi[1].fields["oi"]
    assert ar0132at_multi[0].fields["ip"] is not ar0132at_multi[1].fields["ip"]
    assert ovt_large[0].fields["oi"] is not ovt_large[1].fields["oi"]
    assert ovt_large[0].fields["ip"] is not ovt_large[1].fields["ip"]


def test_camera_create_supports_explicit_l3_payload(asset_store) -> None:
    design_oi = oi_create("pinhole", asset_store=asset_store)
    design_oi.data["photons"] = np.ones((4, 5, 3), dtype=float)
    design_oi.fields["illuminance"] = np.ones((4, 5), dtype=float)
    design_sensor = sensor_create("monochrome", asset_store=asset_store)
    l3_payload = {
        "oi": design_oi,
        "design sensor": design_sensor,
        "kernels": np.arange(4, dtype=float).reshape(2, 2),
        "training illuminant": "D65",
    }

    camera = camera_create("l3", l3_payload, asset_store=asset_store)

    assert camera.name == "L3"
    assert camera_get(camera, "vci type") == "l3"
    assert camera.fields["sensor"] is design_sensor
    assert camera.fields["oi"] is not design_oi
    assert camera.fields["oi"].data == {}
    assert camera.fields["oi"].fields.get("illuminance") is None
    assert camera_get(camera, "l3") is not l3_payload
    assert np.array_equal(np.asarray(camera_get(camera, "l3")["kernels"], dtype=float), l3_payload["kernels"])
    assert camera_get(camera, "l3")["oi"].data == {}
    l3_design_sensor = camera_get(camera, "l3 design sensor")
    assert l3_design_sensor is not design_sensor
    assert l3_design_sensor.name == design_sensor.name
    assert tuple(np.asarray(l3_design_sensor.fields["size"], dtype=int)) == tuple(
        np.asarray(design_sensor.fields["size"], dtype=int)
    )
    assert np.array_equal(np.asarray(camera_get(camera, "l3 kernels"), dtype=float), l3_payload["kernels"])
    assert camera_get(camera, "training illuminant") == "D65"
    ip_l3_design_sensor = camera_get(camera, "ip l3 design sensor")
    assert ip_l3_design_sensor is not design_sensor
    assert ip_l3_design_sensor.name == design_sensor.name
    assert tuple(np.asarray(ip_l3_design_sensor.fields["size"], dtype=int)) == tuple(
        np.asarray(design_sensor.fields["size"], dtype=int)
    )
    assert np.array_equal(np.asarray(camera_get(camera, "ip l3 kernels"), dtype=float), l3_payload["kernels"])

    camera = camera_set(camera, "L3 sensor size", (11, 13))
    assert tuple(np.asarray(camera_get(camera, "sensor size"), dtype=int)) == (11, 13)
    assert tuple(np.asarray(camera_get(camera, "l3 design sensor").fields["size"], dtype=int)) == (11, 13)
    assert tuple(np.asarray(camera_get(camera, "ip l3 design sensor").fields["size"], dtype=int)) == (11, 13)

    camera = camera_set(camera, "L3 sensor fov", 4.0)
    l3_sensor_size = tuple(np.asarray(camera_get(camera, "l3 design sensor").fields["size"], dtype=int))
    assert tuple(np.asarray(camera_get(camera, "sensor size"), dtype=int)) == l3_sensor_size
    assert tuple(np.asarray(camera_get(camera, "ip l3 design sensor").fields["size"], dtype=int)) == l3_sensor_size


def test_camera_compute_supports_vendor_sensor_variants(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    mt9v024_rgbw = camera_compute(camera_create("mt9v024", "rgbw", asset_store=asset_store), scene, asset_store=asset_store)
    ar0132at_rccc = camera_compute(camera_create("ar0132at", "rccc", asset_store=asset_store), scene, asset_store=asset_store)

    assert mt9v024_rgbw.fields["ip"].data["result"].shape[:2] == mt9v024_rgbw.fields["sensor"].fields["size"]
    assert ar0132at_rccc.fields["ip"].data["result"].shape[:2] == ar0132at_rccc.fields["sensor"].fields["size"]


def test_camera_compatibility_wrappers_cover_srgb_sequence_and_clear(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    camera = camera_create(asset_store=asset_store)

    srgb_result, srgb_ideal, raw, updated = cameraComputesrgb(
        camera,
        scene,
        mean_luminance=50.0,
        sz=np.array([16, 16], dtype=int),
        scenefov=5.0,
        scaleoutput=0.8,
        plot_flag=0,
        asset_store=asset_store,
    )

    assert srgb_ideal.shape == (16, 16, 3)
    assert srgb_result.shape[:2] == raw.shape
    assert srgb_result.shape[2] == 3
    assert np.all(np.isfinite(srgb_result))
    assert np.all(np.isfinite(srgb_ideal))
    assert np.all((srgb_result >= 0.0) & (srgb_result <= 1.0))
    assert np.all((srgb_ideal >= 0.0) & (srgb_ideal <= 1.0))
    assert updated.fields["oi"].data["photons"].ndim == 3
    assert updated.fields["sensor"].data["volts"].ndim == 2
    assert updated.fields["ip"].data["result"].shape[:2] == updated.fields["sensor"].data["volts"].shape
    assert updated.fields["ip"].data["result"].shape[2] == 3
    assert updated.fields["ip"].data["result"].shape[0] == srgb_result.shape[0] + 20
    assert updated.fields["ip"].data["result"].shape[1] == srgb_result.shape[1] + 20

    placeholder_result, placeholder_ideal, placeholder_raw, placeholder_updated = cameraComputesrgb(
        camera_create(asset_store=asset_store),
        scene,
        mean_luminance=50.0,
        sz=[],
        scenefov=[],
        scaleoutput=[],
        plot_flag=[],
        asset_store=asset_store,
    )

    explicit_default_result, explicit_default_ideal, explicit_default_raw, explicit_default_updated = cameraComputesrgb(
        camera_create(asset_store=asset_store),
        scene,
        mean_luminance=50.0,
        asset_store=asset_store,
    )

    mean_placeholder_result, mean_placeholder_ideal, mean_placeholder_raw, mean_placeholder_updated = cameraComputesrgb(
        camera_create(asset_store=asset_store),
        scene,
        mean_luminance=[],
        asset_store=asset_store,
    )

    mean_default_result, mean_default_ideal, mean_default_raw, mean_default_updated = cameraComputesrgb(
        camera_create(asset_store=asset_store),
        scene,
        asset_store=asset_store,
    )

    np.testing.assert_allclose(placeholder_result, explicit_default_result, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_ideal, explicit_default_ideal, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_raw, explicit_default_raw, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(camera_get(placeholder_updated, "image"), dtype=float),
        np.asarray(camera_get(explicit_default_updated, "image"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(mean_placeholder_result, mean_default_result, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean_placeholder_ideal, mean_default_ideal, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean_placeholder_raw, mean_default_raw, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(camera_get(mean_placeholder_updated, "image"), dtype=float),
        np.asarray(camera_get(mean_default_updated, "image"), dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )

    sequence_camera, images = cameraComputeSequence(
        camera_create(asset_store=asset_store),
        scenes=[scene, scene],
        exposuretimes=[0.01, 0.02],
        nframes=2,
        asset_store=asset_store,
    )

    kv_sequence_camera, kv_images = cameraComputeSequence(
        camera_create(asset_store=asset_store),
        "scenes",
        [scene, scene],
        "exposuretimes",
        [0.01, 0.02],
        "nframes",
        2,
        asset_store=asset_store,
    )

    placeholder_sequence_camera, placeholder_images = cameraComputeSequence(
        camera_create(asset_store=asset_store),
        "scenes",
        [scene, scene],
        "exposuretimes",
        [],
        "nframes",
        [],
        asset_store=asset_store,
    )

    explicit_default_sequence_camera, explicit_default_images = cameraComputeSequence(
        camera_create(asset_store=asset_store),
        scenes=[scene, scene],
        exposuretimes=1.0,
        asset_store=asset_store,
    )

    session = session_create()
    session_add_and_select_object(session, "scene", scene.clone())
    session_sequence_camera, session_images = cameraComputeSequence(
        camera_create(asset_store=asset_store),
        exposuretimes=[0.01, 0.02],
        nframes=2,
        asset_store=asset_store,
        session=session,
    )

    assert len(images) == 2
    assert images[0].shape == images[1].shape
    assert images[0].ndim == 3 and images[0].shape[2] == 3
    assert np.isclose(camera_get(sequence_camera, "sensor exposure time"), 0.02)
    assert len(kv_images) == 2
    np.testing.assert_allclose(kv_images[0], images[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(kv_images[1], images[1], rtol=1e-10, atol=1e-12)
    assert np.isclose(camera_get(kv_sequence_camera, "sensor exposure time"), 0.02)
    assert len(placeholder_images) == 2
    np.testing.assert_allclose(placeholder_images[0], explicit_default_images[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_images[1], explicit_default_images[1], rtol=1e-10, atol=1e-12)
    assert np.isclose(camera_get(placeholder_sequence_camera, "sensor exposure time"), 1.0)
    assert np.isclose(camera_get(explicit_default_sequence_camera, "sensor exposure time"), 1.0)
    assert len(session_images) == 2
    np.testing.assert_allclose(session_images[0], images[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(session_images[1], images[1], rtol=1e-10, atol=1e-12)
    assert np.isclose(camera_get(session_sequence_camera, "sensor exposure time"), 0.02)

    cleared = cameraClearData(updated)

    assert cleared.data == {}
    assert cleared.fields["oi"].data == {}
    assert cleared.fields["sensor"].data == {}
    assert cleared.fields["ip"].data.get("result") is None
    assert cleared.fields["ip"].fields.get("l3") is None


def test_camera_module_core_matlab_aliases() -> None:
    assert camera_module.cameraCreate is camera_module.camera_create
    assert camera_module.cameraCompute is camera_module.camera_compute
    assert camera_module.cameraComputeSequence is camera_module.camera_compute_sequence
    assert camera_module.cameraComputesrgb is camera_module.camera_compute_srgb
    assert camera_module.cameraClearData is camera_module.camera_clear_data
    assert camera_module.cameraMTF is camera_module.camera_mtf
    assert camera_module.cameraAcutance is camera_module.camera_acutance
    assert camera_module.cameraGet is camera_module.camera_get
    assert camera_module.cameraSet is camera_module.camera_set


def test_camera_create_empty_type_placeholder_uses_default_constructor(asset_store) -> None:
    default_camera = camera_create(asset_store=asset_store)
    placeholder_camera = camera_create([], asset_store=asset_store)

    assert placeholder_camera.name == default_camera.name
    assert camera_get(placeholder_camera, "sensor name") == camera_get(default_camera, "sensor name")
    assert camera_get(placeholder_camera, "oi name") == camera_get(default_camera, "oi name")
    assert camera_get(placeholder_camera, "ip name") == camera_get(default_camera, "ip name")


def test_sensor_create_empty_type_placeholder_uses_default_constructor(asset_store) -> None:
    default_sensor = sensor_create(asset_store=asset_store)
    placeholder_sensor = sensor_create([], asset_store=asset_store)

    assert sensor_get(placeholder_sensor, "name") == sensor_get(default_sensor, "name")
    np.testing.assert_allclose(sensor_get(placeholder_sensor, "wave"), sensor_get(default_sensor, "wave"))
    np.testing.assert_array_equal(sensor_get(placeholder_sensor, "pattern"), sensor_get(default_sensor, "pattern"))
    assert sensor_get(placeholder_sensor, "filter names") == sensor_get(default_sensor, "filter names")


def test_sensor_create_blank_type_placeholder_uses_default_constructor(asset_store) -> None:
    default_sensor = sensor_create(asset_store=asset_store)
    blank_sensor = sensor_create("", asset_store=asset_store)

    assert sensor_get(blank_sensor, "name") == sensor_get(default_sensor, "name")
    np.testing.assert_allclose(sensor_get(blank_sensor, "wave"), sensor_get(default_sensor, "wave"))
    np.testing.assert_array_equal(sensor_get(blank_sensor, "pattern"), sensor_get(default_sensor, "pattern"))
    assert sensor_get(blank_sensor, "filter names") == sensor_get(default_sensor, "filter names")


def test_sensor_module_core_matlab_aliases() -> None:
    assert sensor_module.colorBlockMatrix is sensor_module.color_block_matrix
    assert sensor_module.sensorAddFilter is sensor_module.sensor_add_filter
    assert sensor_module.sensorCfaSave is sensor_module.sensor_cfa_save
    assert sensor_module.sensorCFANameList is sensor_module.sensor_cfa_name_list
    assert sensor_module.sensorCreate is sensor_module.sensor_create
    assert sensor_module.sensorCreateIdeal is sensor_module.sensor_create_ideal
    assert sensor_module.sensorCrop is sensor_module.sensor_crop
    assert sensor_module.sensorDeleteFilter is sensor_module.sensor_delete_filter
    assert sensor_module.sensorDescription is sensor_module.sensor_description
    assert sensor_module.sensorDNGRead is sensor_module.sensor_dng_read
    assert sensor_module.sensorFromFile is sensor_module.sensor_from_file
    assert sensor_module.sensorPixelCoord is sensor_module.sensor_pixel_coord
    assert sensor_module.sensorReadColorFilters is sensor_module.sensor_read_color_filters
    assert sensor_module.sensorReadFilter is sensor_module.sensor_read_filter
    assert sensor_module.sensorReplaceFilter is sensor_module.sensor_replace_filter
    assert sensor_module.sensorRescale is sensor_module.sensor_rescale
    assert sensor_module.pixelCenterFillPD is sensor_module.pixel_center_fill_pd
    assert sensor_module.pixelCreate is sensor_module.pixel_create
    assert sensor_module.pixelDescription is sensor_module.pixel_description
    assert sensor_module.pixelGet is sensor_module.pixel_get
    assert sensor_module.pixelIdeal is sensor_module.pixel_ideal
    assert sensor_module.pixelPositionPD is sensor_module.pixel_position_pd
    assert sensor_module.pixelSet is sensor_module.pixel_set
    assert sensor_module.pixelSR is sensor_module.pixel_sr
    assert sensor_module.pixelSNR is sensor_module.pixel_snr
    assert sensor_module.pixelSNRluxsec is sensor_module.pixel_snr_luxsec
    assert sensor_module.pixelVperLuxSec is sensor_module.pixel_v_per_lux_sec
    assert sensor_module.iePixelWellCapacity is sensor_module.ie_pixel_well_capacity
    assert sensor_module.SignalCurrentDensity is sensor_module.signal_current_density
    assert sensor_module.signalCurrent is sensor_module.signal_current
    assert sensor_module.sensorCompute is sensor_module.sensor_compute
    assert sensor_module.sensorSNR is sensor_module.sensor_snr
    assert sensor_module.sensorGet is sensor_module.sensor_get
    assert sensor_module.sensorSet is sensor_module.sensor_set
    assert sensor_module.sensorSetSizeToFOV is sensor_module.sensor_set_size_to_fov
    assert sensor_module.spatialIntegration is sensor_module.spatial_integration


def test_illuminant_module_core_matlab_aliases() -> None:
    assert illuminant_module.illuminantCreate is illuminant_module.illuminant_create
    assert illuminant_module.illuminantGet is illuminant_module.illuminant_get
    assert illuminant_module.illuminantSet is illuminant_module.illuminant_set


def test_run_python_case_with_context_returns_pipeline_objects(asset_store) -> None:
    case = run_python_case_with_context("camera_default_pipeline", asset_store=asset_store)

    assert case.payload["result"].shape[:2] == tuple(case.context["sensor"].fields["size"])
    assert np.array_equal(case.payload["oi_photons"], case.context["oi"].data["photons"])
    assert np.array_equal(case.payload["sensor_volts"], case.context["sensor"].data["volts"])
    assert case.context["camera"].fields["ip"] is case.context["ip"]


def test_run_python_case_supports_checkerboard_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_checkerboard_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_uniform_bb_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_bb_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_scene_cct_blackbody_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_cct_blackbody_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (65,)
    assert case.payload["single_temperatures_k"].shape == (3,)
    assert case.payload["spd_3500"].shape == (65,)
    assert case.payload["estimated_single_k"].shape == (3,)
    assert case.payload["multi_temperatures_k"].shape == (5,)
    assert case.payload["spd_multi"].shape == (65, 5)
    assert case.payload["estimated_multi_k"].shape == (5,)


def test_run_python_case_supports_scene_daylight_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_daylight_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (371,)
    assert case.payload["cct_k"].shape == (7,)
    assert case.payload["photons"].shape == (371, 7)
    assert case.payload["lum_photons"].shape == (7,)
    assert case.payload["photons_scaled"].shape == (371, 7)
    assert case.payload["energy"].shape == (371, 7)
    assert case.payload["lum_energy"].shape == (7,)
    assert case.payload["energy_scaled"].shape == (371, 7)
    assert case.payload["day_basis"].shape == (371, 3)
    assert case.payload["basis_weights"].shape == (3, 3)
    assert case.payload["basis_examples"].shape == (371, 3)


def test_run_python_case_supports_scene_illuminant_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_illuminant_small", asset_store=asset_store)

    assert case.payload["default_blackbody_wave"].shape == (31,)
    assert case.payload["default_blackbody_photons"].shape == (31,)
    assert case.payload["blackbody_3000_wave"].shape == (301,)
    assert case.payload["blackbody_3000_photons"].shape == (301,)
    assert case.payload["d65_200_wave"].shape == (31,)
    assert case.payload["d65_200_photons"].shape == (31,)
    assert case.payload["equal_energy_wave"].shape == (31,)
    assert case.payload["equal_energy_energy"].shape == (31,)
    assert case.payload["equal_photons_wave"].shape == (31,)
    assert case.payload["equal_photons_photons"].shape == (31,)
    assert case.payload["illuminant_c_photons"].shape == (31,)
    assert case.payload["mono_555_wave"].shape == (31,)
    assert case.payload["mono_555_photons"].shape == (31,)
    assert case.payload["d65_sparse_wave"].shape == (101,)
    assert case.payload["d65_resampled_wave"].shape == (61,)
    assert case.payload["fluorescent_wave"].shape == (61,)
    assert case.payload["tungsten_wave"].shape == (31,)


def test_run_python_case_supports_scene_illuminant_mixtures_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_illuminant_mixtures_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["scene_size"].shape == (2,)
    assert int(case.payload["split_row"]) > 0
    assert case.payload["mixed_illuminant_format"] == "spatial spectral"
    assert case.payload["top_mixed_illuminant_energy"].shape == (31,)
    assert case.payload["bottom_mixed_illuminant_energy"].shape == (31,)
    assert case.payload["top_mixed_reflectance"].shape == (31,)
    assert case.payload["bottom_mixed_reflectance"].shape == (31,)


def test_run_python_case_supports_scene_illuminant_space_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_illuminant_space_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["scene_size"].shape == (2,)
    assert case.payload["initial_illuminant_photons"].shape == (31,)
    assert case.payload["spatial_spectral_shape"].shape == (3,)
    assert case.payload["row_cct_k"].shape[0] == int(case.payload["scene_size"][0])
    assert case.payload["row_top_illuminant_energy"].shape == (31,)
    assert case.payload["row_mid_illuminant_energy"].shape == (31,)
    assert case.payload["row_bottom_illuminant_energy"].shape == (31,)
    assert case.payload["source_mean_reflectance"].shape == (31,)
    assert case.payload["row_mean_reflectance"].shape == (31,)
    assert case.payload["col_scale"].shape[0] == int(case.payload["scene_size"][1])
    assert case.payload["col_scale_norm"].shape == case.payload["col_center_wave_profile_norm"].shape
    assert case.payload["col_mean_reflectance"].shape == (31,)
    assert case.payload["final_center_wave_profile_norm"].shape == case.payload["col_center_wave_profile_norm"].shape


def test_run_python_case_supports_scene_xyz_illuminant_transforms_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_xyz_illuminant_transforms_small", asset_store=asset_store)

    assert case.payload["scene_size"].shape == (2,)
    assert case.payload["xyz_d65_mean_norm"].shape == (3,)
    assert case.payload["xyz_tungsten_mean_norm"].shape == (3,)
    assert case.payload["full_transform"].shape == (3, 3)
    assert case.payload["diagonal_transform"].shape == (3, 3)
    assert case.payload["predicted_full_rmse_ratio"].shape == (3,)
    assert case.payload["predicted_diagonal_rmse_ratio"].shape == (3,)


def test_run_python_case_supports_scene_from_rgb_lcd_apple_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_from_rgb_lcd_apple_small", asset_store=asset_store)

    assert case.payload["display_wave"].shape == (101,)
    assert case.payload["display_spd"].shape == (101, 3)
    assert case.payload["white_spd"].shape == (101,)
    assert case.payload["white_xy"].shape == (2,)
    assert case.payload["scene_size"].shape == (2,)
    assert case.payload["scene_wave"].shape == (101,)
    assert case.payload["scene_mean_photons_norm"].shape == (101,)
    assert case.payload["adjusted_illuminant_energy_norm"].shape == (101,)
    assert case.payload["roi_mean_reflectance"].shape == (101,)


def test_run_python_case_supports_scene_from_multispectral_stuffed_animals_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_from_multispectral_stuffed_animals_small", asset_store=asset_store)

    assert case.payload["scene_size"].shape == (2,)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["mean_scene_spd_norm"].shape == (31,)
    assert case.payload["center_scene_spd_norm"].shape == (31,)


def test_run_python_case_supports_scene_from_rgb_vs_multispectral_stuffed_animals_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_from_rgb_vs_multispectral_stuffed_animals_small", asset_store=asset_store)

    assert case.payload["source_size"].shape == (2,)
    assert case.payload["source_wave"].shape == (31,)
    assert case.payload["source_illuminant_xy"].shape == (2,)
    assert case.payload["reconstructed_size"].shape == (2,)
    assert case.payload["reconstructed_wave"].shape == (101,)
    assert case.payload["reconstructed_illuminant_xy"].shape == (2,)
    assert case.payload["rgb_channel_corr"].shape == (3,)
    assert case.payload["xyz_channel_corr"].shape == (3,)


def test_run_python_case_supports_scene_rgb2radiance_displays_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_rgb2radiance_displays_small", asset_store=asset_store)

    assert case.payload["oled_wave"].shape == (101,)
    assert tuple(case.payload["oled_spd_shape"]) == (101, 4)
    assert case.payload["oled_white_xy"].shape == (2,)
    assert case.payload["oled_primary_xy"].shape == (4, 2)
    assert tuple(case.payload["oled_scene_size"]) == (64, 96)
    assert case.payload["oled_mean_scene_spd_norm"].shape == (101,)
    assert case.payload["oled_illuminant_energy_norm"].shape == (101,)
    assert case.payload["oled_rgb_stats"].shape == (4,)
    assert case.payload["oled_rgb_channel_means"].shape == (3,)
    assert tuple(case.payload["lcd_spd_shape"]) == (101, 3)
    assert case.payload["lcd_primary_xy"].shape == (3, 2)
    assert tuple(case.payload["lcd_scene_size"]) == (64, 96)
    assert tuple(case.payload["crt_spd_shape"]) == (101, 3)
    assert case.payload["crt_primary_xy"].shape == (3, 2)
    assert tuple(case.payload["crt_scene_size"]) == (64, 96)


def test_run_python_case_supports_scene_reflectance_samples_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_reflectance_samples_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (61,)
    assert case.payload["no_replacement_shape"].shape == (2,)
    assert tuple(case.payload["no_replacement_shape"]) == (61, 96)
    assert np.array_equal(case.payload["no_replacement_sample_sizes"], np.array([24, 24, 24, 24], dtype=int))
    assert np.array_equal(case.payload["no_replacement_unique_sizes"], case.payload["no_replacement_sample_sizes"])
    assert case.payload["explicit_shape"].shape == (2,)
    assert tuple(case.payload["explicit_shape"]) == (61, 120)
    assert np.array_equal(case.payload["explicit_sample_sizes"], np.array([60, 60], dtype=int))
    assert case.payload["explicit_sample_first_last"].shape == (2, 2)
    assert case.payload["explicit_mean_reflectance_norm"].shape == (61,)
    assert case.payload["explicit_singular_values_norm"].shape == (61,)


def test_run_python_case_supports_scene_reflectance_chart_basis_functions_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_reflectance_chart_basis_functions_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["scene_size"].shape == (2,)
    assert case.payload["reflectance_shape"].shape == (3,)
    assert int(case.payload["basis_count_999"]) >= int(case.payload["basis_count_95"])
    assert int(case.payload["basis_count_5"]) == 5
    assert case.payload["basis_projector_999"].shape == (31, 31)
    assert case.payload["basis_projector_95"].shape == (31, 31)
    assert case.payload["basis_projector_5"].shape == (31, 31)
    assert case.payload["coef_stats_999"].shape == (4,)
    assert case.payload["coef_stats_95"].shape == (4,)
    assert case.payload["coef_stats_5"].shape == (4,)


def test_run_python_case_supports_scene_roi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_roi_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["scene_size"].shape == (2,)
    assert case.payload["roi_rect"].shape == (4,)
    assert int(case.payload["roi_point_count"]) > 0
    assert case.payload["roi_photons_stats"].shape == (4,)
    assert case.payload["roi_mean_photons"].shape == (31,)
    assert case.payload["roi_energy_stats"].shape == (4,)
    assert case.payload["roi_mean_energy"].shape == (31,)
    assert case.payload["roi_illuminant_photons_stats"].shape == (4,)
    assert case.payload["roi_mean_illuminant_photons"].shape == (31,)
    assert case.payload["roi_reflectance_stats"].shape == (4,)
    assert case.payload["roi_reflectance_mean_manual"].shape == (31,)
    assert case.payload["roi_mean_reflectance_direct"].shape == (31,)
    assert float(case.payload["roi_reflectance_manual_vs_direct_max_abs"]) <= 1e-12


def test_run_python_case_supports_scene_rotate_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_rotate_small", asset_store=asset_store)

    assert case.payload["frame_angles_deg"].shape == (4,)
    assert case.payload["source_size"].shape == (2,)
    assert case.payload["rotated_sizes"].shape == (4, 2)
    assert case.payload["mean_luminance"].shape == (4,)
    assert case.payload["max_luminance"].shape == (4,)
    assert case.payload["center_luminance"].shape == (4,)
    assert case.payload["center_rows_norm"].shape == (4, 129)
    assert case.payload["center_cols_norm"].shape == (4, 129)


def test_run_python_case_supports_scene_wavelength_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_wavelength_small", asset_store=asset_store)

    assert case.payload["source_name"] == "macbethd65"
    assert case.payload["source_size"].shape == (2,)
    assert case.payload["source_wave"].shape == (31,)
    assert case.payload["source_mean_scene_spd_norm"].shape == (31,)
    assert case.payload["source_center_scene_spd_norm"].shape == (31,)
    assert case.payload["five_nm_name"] == "5nmspacing"
    assert case.payload["five_nm_size"].shape == (2,)
    assert case.payload["five_nm_wave"].shape == (61,)
    assert case.payload["five_nm_mean_scene_spd_norm"].shape == (61,)
    assert case.payload["five_nm_center_scene_spd_norm"].shape == (61,)
    assert case.payload["narrow_name"] == "2nmnarrowbandspacing"
    assert case.payload["narrow_size"].shape == (2,)
    assert case.payload["narrow_wave"].shape == (51,)
    assert case.payload["narrow_mean_scene_spd_norm"].shape == (51,)
    assert case.payload["narrow_center_scene_spd_norm"].shape == (51,)


def test_run_python_case_supports_scene_hc_compress_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_hc_compress_small", asset_store=asset_store)

    assert case.payload["source_name"] == "stuffedanimalstungstenhdrs"
    assert case.payload["source_size"].shape == (2,)
    assert case.payload["source_wave"].shape == (31,)
    assert int(case.payload["basis_count_99"]) >= int(case.payload["basis_count_95"])
    assert case.payload["scene95_size"].shape == (2,)
    assert case.payload["scene95_wave"].shape == (61,)
    assert case.payload["scene95_mean_scene_spd_norm"].shape == (61,)
    assert case.payload["scene95_center_scene_spd_norm"].shape == (61,)
    assert case.payload["scene99_size"].shape == (2,)
    assert case.payload["scene99_wave"].shape == (61,)
    assert case.payload["scene99_mean_scene_spd_norm"].shape == (61,)
    assert case.payload["scene99_center_scene_spd_norm"].shape == (61,)


def test_run_python_case_supports_scene_increase_size_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_increase_size_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert tuple(case.payload["source_size"]) == (64, 96)
    assert case.payload["source_mean_scene_spd_norm"].shape == (31,)
    assert tuple(case.payload["step1_size"]) == (128, 288)
    assert case.payload["step1_mean_scene_spd_norm"].shape == (31,)
    assert float(case.payload["step1_replay_max_abs"]) == 0.0
    assert tuple(case.payload["step2_size"]) == (128, 576)
    assert case.payload["step2_mean_scene_spd_norm"].shape == (31,)
    assert float(case.payload["step2_replay_max_abs"]) == 0.0
    assert tuple(case.payload["step3_size"]) == (384, 576)
    assert case.payload["step3_mean_scene_spd_norm"].shape == (31,)
    assert float(case.payload["step3_replay_max_abs"]) == 0.0
    assert np.isclose(float(case.payload["source_aspect_ratio"]), float(case.payload["final_aspect_ratio"]))


def test_scene_increase_size_script_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    source_photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    step1_photons = image_increase_image_rgb_size(source_photons, [2, 3])
    scene_step1 = scene_set(scene.clone(), "photons", step1_photons)
    step2_photons = image_increase_image_rgb_size(step1_photons, [1, 2])
    scene_step2 = scene_set(scene_step1.clone(), "photons", step2_photons)
    step3_photons = image_increase_image_rgb_size(step2_photons, [3, 1])
    scene_step3 = scene_set(scene_step2.clone(), "photons", step3_photons)

    assert source_photons.shape == (64, 96, 31)
    assert step1_photons.shape == (128, 288, 31)
    assert step2_photons.shape == (128, 576, 31)
    assert step3_photons.shape == (384, 576, 31)
    assert np.array_equal(step1_photons[::2, ::3, :], source_photons)
    assert np.array_equal(step2_photons[:, ::2, :], step1_photons)
    assert np.array_equal(step3_photons[::3, :, :], step2_photons)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), scene_get(scene_step1, "mean luminance", asset_store=asset_store))
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), scene_get(scene_step2, "mean luminance", asset_store=asset_store))
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), scene_get(scene_step3, "mean luminance", asset_store=asset_store))
    assert np.isclose(
        float(scene_get(scene, "cols")) / float(scene_get(scene, "rows")),
        float(scene_get(scene_step3, "cols")) / float(scene_get(scene_step3, "rows")),
    )


def test_run_python_case_supports_scene_render_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_render_small", asset_store=asset_store)

    assert tuple(case.payload["daylight_scene_size"]) == (506, 759)
    assert case.payload["daylight_wave"].shape == (31,)
    assert case.payload["daylight_illuminant_photons_norm"].shape == (31,)
    assert case.payload["daylight_srgb_stats"].shape == (4,)
    assert case.payload["daylight_srgb_channel_means"].shape == (3,)
    assert case.payload["daylight_srgb_center_rgb"].shape == (3,)
    assert case.payload["daylight_srgb_center_row_luma_norm"].shape == (129,)
    assert case.payload["hdr_scene_size"].shape == (2,)
    assert case.payload["hdr_wave"].ndim == 1
    assert case.payload["hdr_srgb_stats"].shape == (4,)
    assert case.payload["hdr_render_stats"].shape == (4,)
    assert case.payload["hdr_render_channel_means"].shape == (3,)
    assert case.payload["hdr_render_center_rgb"].shape == (3,)
    assert case.payload["hdr_render_center_row_luma_norm"].shape == (129,)
    assert float(case.payload["hdr_render_delta_mean_abs"]) > 0.0
    assert case.payload["standard_scene_size"].shape == (2,)
    assert case.payload["standard_wave"].ndim == 1
    assert case.payload["standard_srgb_stats"].shape == (4,)
    assert case.payload["standard_render_stats"].shape == (4,)
    assert case.payload["standard_render_channel_means"].shape == (3,)
    assert case.payload["standard_render_center_rgb"].shape == (3,)
    assert case.payload["standard_render_center_row_luma_norm"].shape == (129,)
    assert float(case.payload["standard_render_delta_mean_abs"]) > 0.0


def test_scene_render_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    stuffed_path = asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
    hdr_path = asset_store.resolve("data/images/multispectral/Feng_Office-hdrs.mat")

    daylight_scene = scene_from_file(stuffed_path, "multispectral", None, None, wave, asset_store=asset_store)
    daylight_scene = scene_adjust_illuminant(
        daylight_scene,
        ie_read_spectra("D75.mat", np.asarray(scene_get(daylight_scene, "wave"), dtype=float), asset_store=asset_store),
        asset_store=asset_store,
    )
    daylight_srgb = np.asarray(scene_show_image(daylight_scene, 0, asset_store=asset_store), dtype=float)

    hdr_scene = scene_from_file(hdr_path, "multispectral", asset_store=asset_store)
    hdr_srgb = np.asarray(scene_show_image(hdr_scene, 0, asset_store=asset_store), dtype=float)
    hdr_res = np.asarray(hdr_render(hdr_srgb), dtype=float)

    standard_scene = scene_from_file(stuffed_path, "multispectral", asset_store=asset_store)
    standard_srgb = np.asarray(scene_show_image(standard_scene, 0, asset_store=asset_store), dtype=float)
    standard_res = np.asarray(hdr_render(standard_srgb), dtype=float)

    assert daylight_srgb.shape == (506, 759, 3)
    assert np.all(daylight_srgb >= 0.0)
    assert np.all(daylight_srgb <= 1.0)
    assert hdr_srgb.shape == hdr_res.shape
    assert standard_srgb.shape == standard_res.shape
    assert np.all(hdr_res >= 0.0)
    assert np.all(hdr_res <= 1.0)
    assert np.all(standard_res >= 0.0)
    assert np.all(standard_res <= 1.0)
    assert float(np.mean(np.abs(hdr_res - hdr_srgb))) > 0.0
    assert float(np.mean(np.abs(standard_res - standard_srgb))) > 0.0
    assert tuple(scene_get(daylight_scene, "size")) == (506, 759)
    assert tuple(scene_get(standard_scene, "size")) == (506, 759)


def test_scene_vector_helpers_match_legacy_cube_layout() -> None:
    spectral = np.array([1.0, 2.0, 4.0], dtype=float)
    rows = 2
    cols = 4
    expected = np.broadcast_to(spectral.reshape(1, 1, -1), (rows, cols, spectral.size)).copy()

    radiance = sceneRadianceFromVector(spectral, rows, cols)
    photons = scenePhotonsFromVector(spectral, rows, cols)
    energy = sceneEnergyFromVector(spectral, rows, cols)

    assert radiance.shape == (rows, cols, 3)
    assert np.array_equal(radiance, expected)
    assert np.array_equal(photons, expected)
    assert np.array_equal(energy, expected)
    assert np.array_equal(radiance[0, 0], spectral)
    assert np.array_equal(radiance[-1, -1], spectral)


def test_run_python_case_supports_frequency_orientation_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_frequency_orientation_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_harmonic_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_harmonic_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_scene_sinusoid_workflow(asset_store) -> None:
    params = {
        "freq": np.array([1.0, 5.0], dtype=float),
        "contrast": np.array([0.2, 0.6], dtype=float),
        "ph": np.array([0.0, np.pi / 3.0], dtype=float),
        "ang": np.array([0.0, 0.0], dtype=float),
        "row": 64,
        "col": 64,
        "GaborFlag": 0.2,
    }
    scene = scene_create("sinusoid", params, asset_store=asset_store)
    harmonic_scene = scene_create("harmonic", params, asset_store=asset_store)

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.asarray(scene_get(scene, "photons"), dtype=float).shape == (64, 64, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.asarray(scene_get(harmonic_scene, "wave"), dtype=float))
    assert np.allclose(np.asarray(scene_get(scene, "photons"), dtype=float), np.asarray(scene_get(harmonic_scene, "photons"), dtype=float), atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_sinusoid_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_sinusoid_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_sweep_frequency_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_sweep_frequency_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_reflectance_chart_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_reflectance_chart_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (24, 24)
    assert np.array_equal(case.payload["chart_rowcol"], np.array([3, 3]))
    assert case.payload["chart_index_map"].shape == (24, 24)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_star_pattern_scene_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_star_pattern_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["scene"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["scene"].fields["wave"])
    assert case.payload["photons"].shape[:2] == (64, 64)
    assert case.payload["mean_luminance"] > 0.0


def test_run_python_case_supports_psf_default_oi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psf_default_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert case.context["oi"].fields["optics"]["model"] == "shiftinvariant"
    assert case.context["oi"].fields["optics"]["compute_method"] == "opticsotf"


def test_run_python_case_supports_custom_otf_flare_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_custom_otf_flare_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert case.payload["otf_abs550"].shape == (case.payload["fy"].size, case.payload["fx"].size)
    assert case.payload["interp_otf_abs550"].shape[0] > case.payload["photons"].shape[0]
    assert case.context["oi"].fields["optics"]["compute_method"] == "opticsotf"


def test_run_python_case_supports_optics_psf_to_otf_flare_parity_case(asset_store) -> None:
    case = run_python_case_with_context("optics_psf_to_otf_flare_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["fy"].ndim == 1
    assert case.payload["otf_abs550_row"].shape == case.payload["fx"].shape
    assert case.payload["otf_abs550_center"].shape == (33, 33)
    assert np.max(case.payload["otf_abs550_center"]) > 0.0


def test_run_python_case_supports_wvf_defocus_oi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_wvf_defocus_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert np.isclose(float(case.payload["defocus"]), 2.0)
    assert np.isclose(float(case.payload["vertical_astigmatism"]), 0.5)


def test_run_python_case_supports_wvf_script_defocus_oi_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_wvf_script_defocus_small", asset_store=asset_store)

    assert case.payload["photons"].shape == case.context["oi"].data["photons"].shape
    assert np.array_equal(case.payload["wave"], case.context["oi"].fields["wave"])
    assert np.isclose(float(case.payload["defocus_zcoeff"]), 1.5)
    assert np.isclose(float(case.payload["pupil_diameter_mm"]), 3.0)
    assert case.payload["f_number"] > 0.0


def test_optics_defocus_wvf_script_workflow_matches_explicit_and_oi_methods(asset_store) -> None:
    scene = scene_create("point array", np.array([512, 512], dtype=int), 128, asset_store=asset_store)
    scene = scene_set(scene, "fov", 1.5)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)

    def _psf_center_row_norm(current_oi) -> np.ndarray:
        psf_data = oi_get(current_oi, "psf data", 550.0, "um")
        psf = np.asarray(psf_data["psf"], dtype=float)
        center_row = psf[psf.shape[0] // 2, :]
        return center_row / max(float(np.max(np.abs(center_row))), 1e-12)

    def _oi_center_row_norm(current_oi) -> np.ndarray:
        photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
        oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
        wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
        center_row = photons[photons.shape[0] // 2, :, wave_index]
        return center_row / max(float(np.max(np.abs(center_row))), 1e-12)

    wvf0 = wvf_create(wave=wave)
    wvf0 = wvf_set(wvf0, "focal length", 8.0, "mm")
    wvf0 = wvf_set(wvf0, "pupil diameter", 3.0, "mm")
    wvf0 = wvf_compute(wvf0)
    oi0 = oi_compute(wvf_to_oi(wvf0), scene, crop=True)

    diopters = 1.5
    wvf1 = wvf_create(wave=wave)
    wvf1 = wvf_set(wvf1, "zcoeffs", diopters, "defocus")
    wvf1 = wvf_compute(wvf1)
    oi1 = oi_compute(wvf_to_oi(wvf1), scene, crop=True)

    oi = oi_compute(oi_create("wvf", wvf_create(wave=wave)), scene, crop=True)
    current_wvf = wvf_set(oi_get(oi, "optics wvf"), "zcoeffs", diopters, "defocus")
    oi = oi_compute(oi_set(oi, "optics wvf", current_wvf), scene, crop=True)

    diffraction_limited_psf = _psf_center_row_norm(oi0)
    explicit_defocus_psf = _psf_center_row_norm(oi1)
    explicit_defocus_oi = _oi_center_row_norm(oi1)
    oi_method_defocus_psf = _psf_center_row_norm(oi)
    oi_method_defocus_oi = _oi_center_row_norm(oi)

    assert np.isclose(float(oi_get(oi0, "fnumber")), 8.0 / 3.0)
    assert np.isclose(float(oi_get(oi1, "wvf", "zcoeffs", "defocus")), diopters)
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "defocus")), diopters)
    assert diffraction_limited_psf.shape == explicit_defocus_psf.shape
    assert explicit_defocus_oi.shape == oi_method_defocus_oi.shape
    assert np.max(np.abs(diffraction_limited_psf - explicit_defocus_psf)) > 0.05
    assert np.allclose(explicit_defocus_psf, oi_method_defocus_psf, atol=1e-8)
    assert np.allclose(explicit_defocus_oi, oi_method_defocus_oi, atol=1e-8)


def test_optics_defocus_workflow_supports_wvf_blur_astigmatism_and_pupil_reset(asset_store) -> None:
    scene = scene_create("disk array", 256, 32, np.array([2, 2], dtype=int), asset_store=asset_store)
    scene = scene_set(scene, "fov", 0.5)

    wvf = wvf_create(wave=scene_get(scene, "wave"))
    oi = oi_compute(oi_create("wvf", wvf), scene)

    initial_mean = float(np.mean(np.asarray(oi_get(oi, "photons"), dtype=float)))

    oi = oi_set(oi, "wvf zcoeffs", 2.5, "defocus")
    oi = oi_compute(oi, scene)
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "defocus")), 2.5)

    oi = oi_set(oi, "wvf zcoeffs", 1.0, "vertical_astigmatism")
    oi = oi_compute(oi, scene)
    assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism")), 1.0)

    oi = oi_set(oi, "wvf zcoeffs", 0.0, "vertical_astigmatism")
    oi = oi_compute(oi, scene)
    oi = oi_set(oi, "wvf zcoeffs", 0.0, "defocus")
    oi = oi_compute(oi, scene)
    ending_mean = float(np.mean(np.asarray(oi_get(oi, "photons"), dtype=float)))
    assert np.isclose(initial_mean / ending_mean, 1.0, atol=1e-6)

    current_wvf = oi_get(oi, "wvf")
    pupil_diameter_mm = float(wvf_get(current_wvf, "calc pupil diameter", "mm"))
    current_wvf = wvf_set(current_wvf, "calc pupil diameter", 2.0 * pupil_diameter_mm, "mm")
    oi = oi_set(oi, "optics wvf", current_wvf)
    oi = oi_compute(oi, scene)
    large_pupil_peak = float(np.max(np.asarray(oi_get(oi, "photons"), dtype=float)))

    restored_wvf = wvf_set(oi_get(oi, "wvf"), "calc pupil diameter", pupil_diameter_mm, "mm")
    oi = oi_set(oi, "optics wvf", restored_wvf)
    oi = oi_compute(oi, scene)
    final_mean = float(np.mean(np.asarray(oi_get(oi, "photons"), dtype=float)))
    final_peak = float(np.max(np.asarray(oi_get(oi, "photons"), dtype=float)))

    assert np.isclose(initial_mean / final_mean, 1.0, atol=1e-6)
    assert large_pupil_peak > final_peak


def test_run_python_case_supports_optics_defocus_parity_case(asset_store) -> None:
    case = run_python_case_with_context("optics_defocus_small", asset_store=asset_store)

    assert case.payload["wave"].ndim == 1
    assert np.isclose(float(case.payload["defocus_coeff"]), 2.5)
    assert np.isclose(float(case.payload["vertical_astigmatism_coeff"]), 1.0)
    assert np.isclose(float(case.payload["final_defocus_coeff"]), 0.0)
    assert np.isclose(float(case.payload["final_vertical_astigmatism_coeff"]), 0.0)
    assert np.isclose(float(case.payload["initial_reset_ratio"]), 1.0, atol=1e-6)
    assert np.isclose(float(case.payload["initial_final_ratio"]), 1.0, atol=1e-6)
    assert float(case.payload["doubled_pupil_diameter_mm"]) > float(case.payload["pupil_diameter_mm"])
    assert case.payload["base_center_row_550_norm"].ndim == 1
    assert case.payload["defocus_center_row_550_norm"].shape == case.payload["base_center_row_550_norm"].shape
    assert case.payload["large_pupil_center_row_550_norm"].shape == case.payload["base_center_row_550_norm"].shape


def test_wvf_astigmatism_workflow_supports_defocus_astigmatism_grid() -> None:
    max_um = 20.0
    wvf = wvf_create()
    wvf = wvf_set(wvf, "lcaMethod", "human")
    wvf = wvf_compute(wvf)

    z4 = np.arange(-0.5, 1.0, 0.5, dtype=float)
    z5 = np.arange(-0.5, 1.0, 0.5, dtype=float)
    z4_grid, z5_grid = np.meshgrid(z4, z5, indexing="xy")
    zvals = np.column_stack((z4_grid.reshape(-1, order="F"), z5_grid.reshape(-1, order="F")))

    profiles = []
    for pair in zvals:
        wvf = wvf_set(wvf, "zcoeffs", np.asarray(pair, dtype=float), ["defocus", "vertical_astigmatism"])
        wvf = wvf_set(wvf, "lcaMethod", "human")
        wvf = wvf_compute(wvf)
        udata, handle = wvf_plot(
            wvf,
            "psf normalized",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            max_um,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        profiles.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))
        assert handle is None
        assert np.isclose(float(np.max(psf)), 1.0)

    assert len(profiles) == 9
    assert profiles[0].shape == np.asarray(udata["x"], dtype=float).shape


def test_run_python_case_supports_wvf_astigmatism_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_astigmatism_small", asset_store=asset_store)

    assert case.payload["zvals"].shape == (9, 2)
    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_rows"].shape == (9, case.payload["x"].size)
    assert case.payload["psf_mid_cols"].shape == (9, case.payload["x"].size)
    assert case.payload["psf_centers"].shape == (9,)
    assert np.allclose(case.payload["psf_centers"], 1.0)


def test_run_python_case_supports_wvf_thibos_model_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_thibos_model_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["measured_pupil_mm"]), 4.5)
    assert np.isclose(float(case.payload["calc_pupil_mm"]), 3.0)
    assert np.isclose(float(case.payload["measured_wavelength_nm"]), 550.0)
    assert case.payload["calc_waves_nm"].shape == (3,)
    assert case.payload["example_coeffs"].shape[0] == 10
    assert case.payload["mean_subject_psf_mid_rows"].shape[0] == 3
    assert case.payload["mean_subject_psf_peaks"].shape == (3,)
    assert np.array_equal(case.payload["example_subject_indices"], np.array([1, 4, 7, 10], dtype=int))
    assert case.payload["example_subject_coeffs"].shape == (4, 13)
    assert case.payload["example_subject_psf_mid_rows_450"].shape[0] == 4
    assert case.payload["example_subject_psf_mid_rows_550"].shape[0] == 4
    assert case.payload["example_subject_psf_peaks_450"].shape == (4,)
    assert case.payload["example_subject_psf_peaks_550"].shape == (4,)


def test_wvf_zernike_set_workflow_supports_defocus_astigmatism_oi_bundle(asset_store) -> None:
    params = {
        "blockSize": 64,
        "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
    }
    scene = scene_create("frequency orientation", params, asset_store=asset_store)
    scene = scene_set(scene, "fov", 5.0)
    astigmatism_values = np.array([-1.0, 0.0, 1.0], dtype=float)
    defocus_microns = 2.0

    wvf = wvf_create(wave=scene_get(scene, "wave"))
    psf_rows = []
    oi_rows = []
    for astigmatism in astigmatism_values:
        wvf = wvf_set(wvf, "zcoeffs", np.array([defocus_microns, astigmatism], dtype=float), ["defocus", "vertical_astigmatism"])
        wvf = wvf_compute(wvf)
        udata, handle = wvf_plot(wvf, "psf", "unit", "um", "wave", 550.0, "plot range", 40.0, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        psf_row = psf[psf.shape[0] // 2, :]
        psf_rows.append(psf_row / max(float(np.max(np.abs(psf_row))), 1e-12))
        assert handle is None

        oi = oi_compute(wvf, scene)
        photons = np.asarray(oi_get(oi, "photons"), dtype=float)
        wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        wave_index = int(np.argmin(np.abs(wave - 550.0)))
        center_row = photons[photons.shape[0] // 2, :, wave_index]
        oi_rows.append(center_row / max(float(np.max(np.abs(center_row))), 1e-12))

        assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "defocus")), defocus_microns)
        assert np.isclose(float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism")), astigmatism)

    assert len(psf_rows) == 3
    assert len(oi_rows) == 3
    assert all(row.shape == psf_rows[0].shape for row in psf_rows)
    assert all(row.shape == oi_rows[0].shape for row in oi_rows)


def test_run_python_case_supports_wvf_zernike_set_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_zernike_set_small", asset_store=asset_store)

    assert case.payload["wave"].ndim == 1
    assert np.array_equal(case.payload["astigmatism_values"], np.array([-1.0, 0.0, 1.0], dtype=float))
    assert np.isclose(float(case.payload["defocus_microns"]), 2.0)
    assert case.payload["psf_support_um"].ndim == 1
    assert case.payload["psf_mid_rows"].shape[0] == 3
    assert np.allclose(case.payload["oi_defocus_coeffs"], np.full(3, 2.0))
    assert np.allclose(case.payload["oi_astigmatism_coeffs"], np.array([-1.0, 0.0, 1.0], dtype=float))
    assert case.payload["oi_center_rows_550"].shape[0] == 3
    assert case.payload["oi_peak_photons_550"].shape == (3,)


def test_wvf_wavefronts_workflow_supports_osa_index_sweep() -> None:
    indices = np.arange(1, 17, dtype=int)
    n_values, m_values = wvf_osa_index_to_zernike_nm(indices)
    row_profiles = []
    col_profiles = []
    peak_abs_values = []

    for index in indices:
        wvf = wvf_create()
        wvf = wvf_set(wvf, "npixels", 801)
        wvf = wvf_set(wvf, "measured pupil size", 2.0)
        wvf = wvf_set(wvf, "calc pupil size", 2.0)
        wvf = wvf_set(wvf, "zcoeff", 1.0, int(index))
        wvf = wvf_compute(wvf)
        udata, handle = wvf_plot(
            wvf,
            "image wavefront aberrations",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        wavefront = np.asarray(udata["z"], dtype=float)
        row_profiles.append(wavefront[wavefront.shape[0] // 2, :])
        col_profiles.append(wavefront[:, wavefront.shape[1] // 2])
        peak_abs_values.append(float(np.max(np.abs(wavefront))))

        assert handle is None
        assert np.isclose(float(wvf_get(wvf, "zcoeffs", int(index))), 1.0)

    assert np.array_equal(n_values, np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5], dtype=int))
    assert np.array_equal(m_values, np.array([-1, 1, -2, 0, 2, -3, -1, 1, 3, -4, -2, 0, 2, 4, -5, -3], dtype=int))
    assert len(row_profiles) == 16
    assert len(col_profiles) == 16
    assert all(profile.shape == row_profiles[0].shape for profile in row_profiles)
    assert all(profile.shape == col_profiles[0].shape for profile in col_profiles)
    assert np.all(np.asarray(peak_abs_values, dtype=float) > 0.0)


def test_run_python_case_supports_wvf_wavefronts_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_wavefronts_small", asset_store=asset_store)

    assert np.array_equal(case.payload["indices"], np.arange(1, 17, dtype=int))
    assert case.payload["x"].ndim == 1
    assert case.payload["wavefront_mid_rows_norm"].shape == (16, case.payload["x"].size)
    assert case.payload["wavefront_mid_cols_norm"].shape == (16, case.payload["x"].size)
    assert case.payload["wavefront_peak_abs"].shape == (16,)
    assert int(case.payload["npixels"]) == 801
    assert np.isclose(float(case.payload["measured_pupil_mm"]), 2.0)
    assert np.isclose(float(case.payload["calc_pupil_mm"]), 2.0)


def test_zernike_interpolation_workflow_supports_field_height_psf_comparison(asset_store) -> None:
    raw = asset_store.load_mat("data/optics/zernike_doubleGauss.mat")
    data = raw["data"]
    wavelengths = np.asarray(data.wavelengths, dtype=float).reshape(-1)
    image_heights = np.asarray(data.image_heights, dtype=float).reshape(-1)
    zcoeffs = data.zernikeCoefficients

    image_height_indices = np.arange(1, 22, 4, dtype=int)
    this_wave_index = 3
    test_index = 6
    image_heights_test = image_heights[image_height_indices - 1]
    coeff_matrix = np.vstack(
        [
            np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(index)}"), dtype=float).reshape(-1)
            for index in image_height_indices
        ]
    )
    test_height = float(image_heights[test_index - 1])
    zernike_interpolated = np.array(
        [np.interp(test_height, image_heights_test, coeff_matrix[:, column]) for column in range(coeff_matrix.shape[1])],
        dtype=float,
    )
    zernike_gt = np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{test_index}"), dtype=float).reshape(-1)
    validation = zernike_interpolated - zernike_gt

    case = run_python_case_with_context("zernike_interpolation_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["wavelength_nm"]), float(wavelengths[this_wave_index - 1]))
    assert np.array_equal(case.payload["image_height_indices"], image_height_indices)
    assert np.array_equal(case.payload["nearest_indices"], np.array([5, 9], dtype=int))
    assert np.allclose(case.payload["zernike_gt"], zernike_gt)
    assert np.allclose(case.payload["zernike_interpolated"], zernike_interpolated)
    assert np.allclose(case.payload["validation"], validation)
    assert case.payload["psf_interpolated_mid_row_norm"].shape == (512,)
    assert case.payload["psf_gt_mid_row_norm"].shape == (512,)
    assert case.payload["psf_interp_space_mid_row_norm"].shape == (512,)


def test_run_python_case_supports_zernike_interpolation_parity_case(asset_store) -> None:
    case = run_python_case_with_context("zernike_interpolation_small", asset_store=asset_store)

    assert int(case.payload["this_wave_index"]) == 3
    assert np.isclose(float(case.payload["wavelength_nm"]), 621.95)
    assert np.array_equal(case.payload["image_height_indices"], np.array([1, 5, 9, 13, 17, 21], dtype=int))
    assert case.payload["image_heights_test"].shape == (6,)
    assert int(case.payload["test_index"]) == 6
    assert np.array_equal(case.payload["nearest_indices"], np.array([5, 9], dtype=int))
    assert case.payload["zernike_gt"].shape == (15,)
    assert case.payload["zernike_interpolated"].shape == (15,)
    assert case.payload["validation"].shape == (15,)
    assert np.isfinite(float(case.payload["validation_rmse"]))
    assert case.payload["psf_interpolated_mid_row_norm"].shape == (512,)
    assert case.payload["psf_gt_mid_row_norm"].shape == (512,)
    assert case.payload["psf_interp_space_mid_row_norm"].shape == (512,)
    assert float(case.payload["psf_interpolated_peak"]) > 0.0
    assert float(case.payload["psf_gt_peak"]) > 0.0
    assert float(case.payload["psf_interp_space_peak"]) > 0.0


def test_wvf_plot_script_sequence_supports_wave_switch_and_mixed_units() -> None:
    wave_550 = 550.0
    wave_460 = 460.0

    wvf = wvf_create()
    wvf = wvf_set(wvf, "wave", wave_550)
    wvf = wvf_set(wvf, "spatial samples", 401)
    wvf = wvf_compute(wvf)

    udata_550_um, handle_550_um = wvf_plot(wvf, "1d psf", "unit", "um", "wave", wave_550, "window", False)
    udata_550_mm, handle_550_mm = wvf_plot(wvf, "1d psf", "unit", "mm", "wave", wave_550, "window", False)
    udata_550_norm, handle_550_norm = wvf_plot(wvf, "1d psf normalized", "unit", "mm", "wave", wave_550, "window", False)

    assert handle_550_um is None
    assert handle_550_mm is None
    assert handle_550_norm is None
    assert np.asarray(udata_550_um["x"], dtype=float).ndim == 1
    assert np.asarray(udata_550_mm["x"], dtype=float).ndim == 1
    assert np.isclose(float(np.max(np.asarray(udata_550_norm["y"], dtype=float))), 1.0)

    wvf = wvf_set(wvf, "wave", wave_460)
    wvf = wvf_compute(wvf)

    udata_460_angle, handle_460_angle = wvf_plot(
        wvf, "image psf angle", "unit", "min", "wave", wave_460, "plot range", 1.0, "window", False
    )
    udata_460_phase, handle_460_phase = wvf_plot(
        wvf, "image pupil phase", "unit", "mm", "wave", wave_460, "plot range", 2.0, "window", False
    )

    assert handle_460_angle is None
    assert handle_460_phase is None
    assert np.asarray(udata_460_angle["z"], dtype=float).ndim == 2
    assert np.asarray(udata_460_phase["z"], dtype=float).ndim == 2
    assert np.isclose(float(np.asarray(wvf_get(wvf, "wave"), dtype=float)[0]), wave_460)


def test_run_python_case_supports_wvf_plot_script_sequence_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_script_sequence_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["wave_550_nm"]), 550.0)
    assert np.isclose(float(case.payload["wave_460_nm"]), 460.0)
    assert case.payload["line_550_um_x"].ndim == 1
    assert case.payload["line_550_um_y_norm"].shape == case.payload["line_550_um_x"].shape
    assert case.payload["line_550_mm_x"].ndim == 1
    assert case.payload["line_550_mm_y_norm"].shape == case.payload["line_550_mm_x"].shape
    assert np.isclose(float(np.max(np.asarray(case.payload["line_550_mm_norm_y"], dtype=float))), 1.0)
    assert case.payload["psf_angle_460_x"].ndim == 1
    assert case.payload["psf_angle_460_mid_row_norm"].shape == case.payload["psf_angle_460_x"].shape
    assert float(case.payload["psf_angle_460_center"]) > 0.0
    assert case.payload["pupil_phase_460_x"].ndim == 1
    assert case.payload["pupil_phase_460_mid_row"].shape == case.payload["pupil_phase_460_x"].shape
    assert np.isfinite(float(case.payload["pupil_phase_460_center"]))


def test_wvf_diffraction_workflow_supports_script_sweeps() -> None:
    flength_mm = 6.0
    flength_m = flength_mm * 1e-3
    f_number = 3.0
    this_wave = 550.0

    wvf = wvf_create()
    wvf = wvf_set(wvf, "calc pupil diameter", flength_mm / f_number)
    wvf = wvf_set(wvf, "focal length", flength_m)
    wvf = wvf_compute(wvf)

    psf_udata, _ = wvf_plot(
        wvf,
        "psf",
        "unit",
        "um",
        "wave",
        this_wave,
        "plot range",
        10.0,
        "airy disk",
        True,
        "window",
        False,
    )
    oi = wvf_to_oi(wvf)
    oi_udata, _ = oi_plot(oi, "psfxaxis", None, this_wave, "um")

    assert np.isclose(float(wvf_get(wvf, "fnumber")), float(oi_get(oi, "optics fnumber")))
    assert float(psf_udata["airyDiskDiameter"]) == pytest.approx(float(airy_disk(this_wave, f_number, "units", "um", "diameter", True)))
    assert np.asarray(oi_udata["samp"], dtype=float).ndim == 1
    assert np.asarray(oi_udata["data"], dtype=float).shape == np.asarray(oi_udata["samp"], dtype=float).shape

    pupil_mm = np.linspace(1.5, 8.0, 4, dtype=float)
    pupil_airy = []
    for pupil in pupil_mm:
        wvf = wvf_set(wvf, "calc pupil diameter", float(pupil))
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf",
            "unit",
            "um",
            "wave",
            this_wave,
            "plot range",
            5.0,
            "airy disk",
            True,
            "window",
            False,
        )
        pupil_airy.append(float(udata["airyDiskDiameter"]))

    assert len(pupil_airy) == 4
    assert pupil_airy[0] > pupil_airy[-1]


def test_run_python_case_supports_wvf_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_diffraction_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["base_fnumber_ratio_oi_wvf"]), 1.0)
    assert float(case.payload["base_airy_diameter_um"]) > 0.0
    assert case.payload["base_oi_psfx_data"].ndim == 1
    assert case.payload["pupil_mm"].shape == (4,)
    assert case.payload["pupil_550_airy_diameter_um"].shape == (4,)
    assert case.payload["pupil_400_airy_diameter_um"].shape == (4,)
    assert case.payload["lca_wavelength_nm"].shape == (4,)
    assert case.payload["lca_airy_diameter_um"].shape == (4,)
    assert case.payload["lca_mid_rows"].shape == (4, 41)
    assert case.payload["focal_length_sweep_mm"].shape == (3,)
    assert case.payload["focal_length_um_per_degree"].shape == (3,)


def test_run_python_case_supports_wvf_spatial_sampling_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_spatial_sampling_small", asset_store=asset_store)

    assert int(case.payload["npixels"]) == 201
    assert int(case.payload["calc_nwave"]) == int(np.asarray(case.payload["wave"], dtype=float).size)
    assert case.payload["psf_xaxis_um"].shape == case.payload["psf_xaxis_data"].shape
    assert case.payload["pupil_positions_mm"].shape == (int(case.payload["npixels"]),)
    assert case.payload["pupil_amp_row"].shape == (int(case.payload["npixels"]),)
    assert case.payload["pupil_phase_row"].shape == (int(case.payload["npixels"]),)
    assert float(case.payload["psf_sample_spacing_arcmin"]) > 0.0


def test_run_python_case_supports_wvf_spatial_controls_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_spatial_controls_small", asset_store=asset_store)

    assert int(case.payload["base_npixels"]) == 201
    assert int(case.payload["reduced_pixels_npixels"]) == round(float(case.payload["base_npixels"]) / 4.0)
    assert np.isclose(
        float(case.payload["reduced_pixels_psf_sample_spacing_arcmin"]),
        float(case.payload["base_psf_sample_spacing_arcmin"]),
    )
    assert np.isclose(float(case.payload["reduced_pixels_um_per_degree"]), float(case.payload["base_um_per_degree"]))
    assert float(case.payload["pupil_plane_x4_psf_sample_spacing_arcmin"]) < float(case.payload["base_psf_sample_spacing_arcmin"])
    assert float(case.payload["pupil_plane_div4_psf_sample_spacing_arcmin"]) > float(case.payload["base_psf_sample_spacing_arcmin"])
    assert np.isclose(float(case.payload["pupil_plane_x4_um_per_degree"]), float(case.payload["base_um_per_degree"]))
    assert np.isclose(float(case.payload["pupil_plane_div4_um_per_degree"]), float(case.payload["base_um_per_degree"]))
    assert np.isclose(
        float(case.payload["focal_length_half_psf_sample_spacing_arcmin"]),
        float(case.payload["base_psf_sample_spacing_arcmin"]),
    )
    assert np.isclose(
        float(case.payload["focal_length_double_psf_sample_spacing_arcmin"]),
        float(case.payload["base_psf_sample_spacing_arcmin"]),
    )
    assert float(case.payload["focal_length_half_um_per_degree"]) < float(case.payload["base_um_per_degree"])
    assert float(case.payload["focal_length_double_um_per_degree"]) > float(case.payload["base_um_per_degree"])


def test_run_python_case_supports_wvf_compute_psf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_compute_psf_small", asset_store=asset_store)

    assert int(case.payload["npixels"]) == 101
    assert np.isclose(float(case.payload["psf_sum"]), 1.0)
    assert case.payload["psf_mid_row"].shape == (int(case.payload["npixels"]),)
    assert case.payload["pupil_amp_row"].shape == (int(case.payload["npixels"]),)
    assert case.payload["pupil_phase_row"].shape == (int(case.payload["npixels"]),)


def test_run_python_case_supports_wvf_plot_otf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_otf_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["otf_mid_row"].shape == case.payload["fx"].shape
    assert float(case.payload["otf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_otf_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_otf_normalized_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["otf_mid_row"].shape == case.payload["fx"].shape
    assert float(case.payload["otf_center"]) > 1.0


def test_run_python_case_supports_wvf_plot_1d_otf_angle_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_otf_angle_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["otf_mid_row"].shape == case.payload["fx"].shape
    assert float(case.payload["otf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_1d_otf_angle_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_otf_angle_normalized_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["otf_mid_row"].shape == case.payload["fx"].shape
    assert float(case.payload["otf_center"]) > 1.0


def test_run_python_case_supports_wvf_plot_1d_otf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_otf_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["otf_mid_row"].shape == case.payload["fx"].shape
    assert float(case.payload["otf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_1d_otf_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_otf_normalized_small", asset_store=asset_store)

    assert case.payload["fx"].ndim == 1
    assert case.payload["otf_mid_row"].shape == case.payload["fx"].shape
    assert float(case.payload["otf_center"]) > 1.0


def test_run_python_case_supports_wvf_plot_pupil_amp_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_pupil_amp_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["amp_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["amp_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_2d_pupil_amplitude_space_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_2d_pupil_amplitude_space_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["amp_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["amp_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_pupil_phase_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_pupil_phase_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["phase_mid_row"].shape == case.payload["x"].shape
    assert np.isfinite(float(case.payload["phase_center"]))


def test_run_python_case_supports_wvf_plot_2d_pupil_phase_space_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_2d_pupil_phase_space_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["phase_mid_row"].shape == case.payload["x"].shape
    assert np.isfinite(float(case.payload["phase_center"]))


def test_run_python_case_supports_wvf_plot_wavefront_aberrations_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_wavefront_aberrations_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["wavefront_mid_row"].shape == case.payload["x"].shape
    assert np.isfinite(float(case.payload["wavefront_center"]))


def test_run_python_case_supports_wvf_plot_2d_wavefront_aberrations_space_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_2d_wavefront_aberrations_space_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["wavefront_mid_row"].shape == case.payload["x"].shape
    assert np.isfinite(float(case.payload["wavefront_center"]))


def test_run_python_case_supports_wvf_plot_image_psf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_image_psf_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["psf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_image_psf_airy_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_image_psf_airy_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["psf_center"]) > 0.0
    assert float(case.payload["airy_disk_radius"]) > 0.0


def test_run_python_case_supports_wvf_plot_image_psf_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_image_psf_normalized_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert np.isclose(float(case.payload["psf_center"]), 1.0)


def test_run_python_case_supports_wvf_plot_psf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_psf_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["psf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_psf_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_psf_normalized_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert np.isclose(float(case.payload["psf_center"]), 1.0)


def test_run_python_case_supports_wvf_plot_image_psf_angle_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_image_psf_angle_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["psf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_image_psf_angle_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_image_psf_angle_normalized_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert np.isclose(float(case.payload["psf_center"]), 1.0)


def test_run_python_case_supports_wvf_plot_2d_psf_angle_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_2d_psf_angle_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert float(case.payload["psf_center"]) > 0.0


def test_run_python_case_supports_wvf_plot_2d_psf_angle_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_2d_psf_angle_normalized_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["psf_mid_row"].shape == case.payload["x"].shape
    assert np.isclose(float(case.payload["psf_center"]), 1.0)


def test_run_python_case_supports_wvf_plot_1d_psf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_psf_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["y"].shape == case.payload["x"].shape
    assert float(case.payload["peak"]) > 0.0


def test_run_python_case_supports_wvf_plot_1d_psf_space_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_psf_space_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["y"].shape == case.payload["x"].shape
    assert float(case.payload["peak"]) > 0.0


def test_run_python_case_supports_wvf_plot_1d_psf_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_psf_normalized_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["y"].shape == case.payload["x"].shape
    assert np.isclose(float(case.payload["peak"]), 1.0)


def test_run_python_case_supports_wvf_plot_1d_psf_angle_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_psf_angle_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["y"].shape == case.payload["x"].shape
    assert float(case.payload["peak"]) > 0.0


def test_run_python_case_supports_wvf_plot_1d_psf_angle_normalized_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_1d_psf_angle_normalized_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["y"].shape == case.payload["x"].shape
    assert np.isclose(float(case.payload["peak"]), 1.0)


def test_run_python_case_supports_wvf_plot_psf_xaxis_airy_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_psf_xaxis_airy_small", asset_store=asset_store)

    assert case.payload["samp"].ndim == 1
    assert case.payload["data"].shape == case.payload["samp"].shape
    assert float(case.payload["airy_disk_radius"]) > 0.0


def test_run_python_case_supports_wvf_plot_psfxaxis_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_psfxaxis_small", asset_store=asset_store)

    assert case.payload["samp"].ndim == 1
    assert case.payload["data"].shape == case.payload["samp"].shape


def test_run_python_case_supports_wvf_plot_psf_yaxis_airy_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_psf_yaxis_airy_small", asset_store=asset_store)

    assert case.payload["samp"].ndim == 1
    assert case.payload["data"].shape == case.payload["samp"].shape
    assert float(case.payload["airy_disk_radius"]) > 0.0


def test_run_python_case_supports_wvf_plot_psfyaxis_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_plot_psfyaxis_small", asset_store=asset_store)

    assert case.payload["samp"].ndim == 1
    assert case.payload["data"].shape == case.payload["samp"].shape


def test_wvf_wave_getter_supports_unit_and_index() -> None:
    wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))

    assert np.array_equal(np.asarray(wvf_get(wvf, "wave"), dtype=float), np.array([450.0, 550.0, 650.0], dtype=float))
    assert np.allclose(np.asarray(wvf_get(wvf, "wave", "um"), dtype=float), np.array([0.45, 0.55, 0.65], dtype=float))
    assert np.isclose(float(wvf_get(wvf, "wave", "um", 2)), 0.55)
    assert np.isclose(float(wvf_get(wvf, "measured wavelength", "um")), 0.55)


def test_wvf_wave_to_idx_matches_calc_wavelength_rounding() -> None:
    wvf = wvf_create(wave=np.array([400.0, 500.0, 600.0], dtype=float))

    idx = wvf_wave_to_idx(wvf, np.array([500.4, 599.6], dtype=float))

    assert np.array_equal(idx, np.array([2, 3], dtype=int))
    with pytest.raises(ValueError, match="No matching wavelength"):
        wvf_wave_to_idx(wvf, np.array([455.0], dtype=float))


def test_psf2zcoeff_error_is_small_for_matching_wvf_psf() -> None:
    wvf = wvf_create(wave=np.array([550.0], dtype=float))
    wvf = wvf_set(wvf, "zcoeffs", 0.2, "defocus")
    wvf = wvf_set(wvf, "zcoeffs", 0.0, "vertical_astigmatism")
    wvf = wvf_compute(wvf)

    this_wave_nm = float(wvf_get(wvf, "wave", 1))
    this_wave_um = float(wvf_get(wvf, "wave", "um", 1))
    psf_target = np.asarray(wvf_get(wvf, "psf", this_wave_nm), dtype=float)
    zcoeffs = np.asarray(wvf_get(wvf, "zcoeffs"), dtype=float)

    matching_error = psf_to_zcoeff_error(
        zcoeffs[:6],
        psf_target,
        wvf_get(wvf, "pupil size", "mm"),
        wvf_get(wvf, "z pupil diameter"),
        wvf_get(wvf, "pupil plane size", "mm", this_wave_nm),
        this_wave_um,
        wvf_get(wvf, "spatial samples"),
    )
    query_error = psf_to_zcoeff_error(
        np.array([0.0, 0.0, 0.0, 0.0, 0.15, 0.02], dtype=float),
        psf_target,
        wvf_get(wvf, "pupil size", "mm"),
        wvf_get(wvf, "z pupil diameter"),
        wvf_get(wvf, "pupil plane size", "mm", this_wave_nm),
        this_wave_um,
        wvf_get(wvf, "spatial samples"),
    )
    mismatched_error = psf_to_zcoeff_error(
        np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0], dtype=float),
        psf_target,
        wvf_get(wvf, "pupil size", "mm"),
        wvf_get(wvf, "z pupil diameter"),
        wvf_get(wvf, "pupil plane size", "mm", this_wave_nm),
        this_wave_um,
        wvf_get(wvf, "spatial samples"),
    )

    assert matching_error > 0.0
    assert query_error > 0.0
    assert mismatched_error > query_error


def test_run_python_case_supports_wvf_psf2zcoeff_error_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_psf2zcoeff_error_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["wave_um"]), 0.55)
    assert int(case.payload["n_pixels"]) == 201
    assert case.payload["query_zcoeffs"].shape == (6,)
    assert float(case.payload["error"]) >= 0.0


def test_run_python_case_supports_wvf_aperture_polygon_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_aperture_polygon_clean_small", asset_store=asset_store)

    assert int(case.payload["nsides"]) == 8
    assert case.payload["image"].shape == (101, 101)
    assert case.payload["mid_row"].shape == (101,)
    assert float(case.payload["image_sum"]) > 0.0


def test_run_python_case_supports_wvf_compute_aperture_polygon_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_compute_aperture_polygon_small", asset_store=asset_store)

    assert int(case.payload["nsides"]) == 8
    assert np.isclose(float(case.payload["psf_sum"]), 1.0)
    assert case.payload["psf_mid_row"].shape == (101,)
    assert case.payload["pupil_amp_row"].shape == (101,)


def test_run_python_case_supports_lswavelength_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_lswavelength_diffraction_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["wavelength"].ndim == 1
    assert case.payload["lsWave"].shape == (case.payload["wavelength"].size, case.payload["x"].size)
    assert np.all(case.payload["lsWave"] >= 0.0)


def test_run_python_case_supports_lswavelength_wvf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_lswavelength_wvf_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 1
    assert case.payload["wavelength"].ndim == 1
    assert case.payload["lsWave"].shape == (case.payload["wavelength"].size, case.payload["x"].size)
    assert np.all(case.payload["lsWave"] >= 0.0)


def test_run_python_case_supports_otfwavelength_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_otfwavelength_diffraction_small", asset_store=asset_store)

    assert case.payload["fSupport"].ndim == 1
    assert case.payload["wavelength"].ndim == 1
    assert case.payload["otf"].shape == (case.payload["fSupport"].size, case.payload["wavelength"].size)
    assert np.all(case.payload["otf"] >= 0.0)


def test_run_python_case_supports_otfwavelength_wvf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_otfwavelength_wvf_small", asset_store=asset_store)

    assert case.payload["fSupport"].ndim == 1
    assert case.payload["wavelength"].ndim == 1
    assert case.payload["otf"].shape == (case.payload["fSupport"].size, case.payload["wavelength"].size)
    assert np.all(case.payload["otf"] >= 0.0)


def test_run_python_case_supports_irradiance_hline_diffraction_lineep_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_irradiance_hline_diffraction_lineep_small", asset_store=asset_store)

    assert case.payload["roi_locs"].shape == (2,)
    assert case.payload["pos"].ndim == 1
    assert case.payload["wave"].ndim == 1
    assert case.payload["data"].shape == (case.payload["wave"].size, case.payload["pos"].size)
    assert np.all(case.payload["data"] >= 0.0)


def test_oi_cos4th_script_smoke(asset_store) -> None:
    scene = scene_create("uniform d65", 512, asset_store=asset_store)
    scene = scene_set(scene, "fov", 80)

    oi = oi_create("shift invariant", asset_store=asset_store)
    focal_length = float(oi_get(oi, "optics focal length"))
    oi = oi_compute(oi, scene)
    size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
    center_locator = np.array([1, size[1] / 2.0], dtype=float)
    default_line, _ = oi_plot(oi, "illuminance hline", center_locator)

    oi = oi_set(oi, "optics focal length", 4.0 * focal_length)
    oi = oi_compute(oi, scene)
    long_center_line, _ = oi_plot(oi, "illuminance hline", center_locator)
    long_edge_line, _ = oi_plot(oi, "illuminance hline", np.array([1, 20], dtype=float))

    assert default_line["pos"].ndim == 1
    assert default_line["data"].shape == default_line["pos"].shape
    assert long_center_line["data"].shape == long_center_line["pos"].shape
    assert long_edge_line["data"].shape == long_edge_line["pos"].shape
    assert np.isclose(float(oi_get(oi, "optics focal length")), 4.0 * focal_length)
    assert float(np.max(long_center_line["data"])) > 0.0
    assert not np.allclose(long_center_line["data"], long_edge_line["data"])


def test_run_python_case_supports_oi_cos4th_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_cos4th_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["focal_length_long_m"]), 4.0 * float(case.payload["focal_length_default_m"]))
    assert case.payload["size_default"].shape == (2,)
    assert case.payload["size_long"].shape == (2,)
    assert int(case.payload["center_row"]) > 0
    assert int(case.payload["edge_row"]) == 20
    assert case.payload["pos_default_um"].ndim == 1
    assert case.payload["center_line_default_lux"].shape == case.payload["pos_default_um"].shape
    assert case.payload["pos_long_um"].shape == case.payload["center_line_long_lux"].shape
    assert case.payload["edge_line_long_lux"].shape == case.payload["pos_long_um"].shape
    assert float(case.payload["mean_illuminance_default_lux"]) > 0.0
    assert float(case.payload["mean_illuminance_long_lux"]) > 0.0


def test_optics_diffraction_script_workflow(asset_store) -> None:
    scene = scene_create("point array", 128, 16, "d65", 1, asset_store=asset_store)
    scene = scene_set(scene, "h fov", 1.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_compute(oi, scene)

    assert np.isclose(float(scene_get(scene, "fov")), 1.0)
    assert np.isclose(float(oi_get(oi, "optics f number")), 4.0)
    assert np.array_equal(np.asarray(oi_get(oi, "size"), dtype=int), np.array([160, 160], dtype=int))

    oi = oi_set(oi, "optics fnumber", 12.0)
    oi = oi_compute(oi, scene)

    psf_udata, _ = oi_plot(oi, "psf 550")
    ls_udata, _ = oi_plot(oi, "ls wavelength")
    focal_length_mm = float(oi_get(oi, "optics focal length", "mm"))
    pupil_diameter_mm = float(oi_get(oi, "optics pupil diameter", "mm"))

    assert np.isclose(float(oi_get(oi, "optics f number")), 12.0)
    assert np.array_equal(np.asarray(oi_get(oi, "size"), dtype=int), np.array([160, 160], dtype=int))
    assert np.isclose(focal_length_mm / pupil_diameter_mm, 12.0)
    assert np.asarray(psf_udata["psf"]).shape == np.asarray(psf_udata["x"]).shape == np.asarray(psf_udata["y"]).shape
    assert np.asarray(ls_udata["x"]).shape == (41,)
    assert np.asarray(ls_udata["wavelength"]).shape == (31,)
    assert np.asarray(ls_udata["lsWave"]).shape == (31, 41)


def test_optics_flare_script_workflow(asset_store) -> None:
    point_scene = scene_create("point array", 384, 128, asset_store=asset_store)
    point_scene = scene_set(point_scene, "fov", 1.0)
    hdr_scene = scene_create("hdr", asset_store=asset_store)
    hdr_scene = scene_set(hdr_scene, "fov", 1.0)

    base_wvf = wvf_create()
    base_wvf = wvf_set(base_wvf, "calc pupil diameter", 3.0)
    base_wvf = wvf_set(base_wvf, "focal length", 7e-3)

    aperture_initial, params_initial = wvf_aperture(
        base_wvf,
        "nsides",
        3,
        "dot mean",
        20,
        "dot sd",
        3,
        "dot opacity",
        0.5,
        "line mean",
        20,
        "line sd",
        2,
        "line opacity",
        0.5,
        "image rotate",
        0,
        "seed",
        1,
    )
    wvf_initial = wvf_compute(base_wvf, "aperture", aperture_initial)
    psf_initial = np.asarray(wvf_get(wvf_initial, "psf", 550.0), dtype=float)
    oi_initial_point = oi_crop(oi_compute(wvf_initial, point_scene), "border")
    oi_initial_hdr = oi_compute(wvf_initial, hdr_scene)

    aperture_five, params_five = wvf_aperture(
        base_wvf,
        "nsides",
        5,
        "dot mean",
        20,
        "dot sd",
        3,
        "dot opacity",
        0.5,
        "line mean",
        20,
        "line sd",
        2,
        "line opacity",
        0.5,
        "image rotate",
        0,
        "seed",
        2,
    )
    wvf_five = wvf_compute(base_wvf, "aperture", aperture_five)
    psf_five = np.asarray(wvf_get(wvf_five, "psf", 550.0), dtype=float)
    oi_five_point = oi_crop(oi_compute(wvf_five, point_scene), "border")
    oi_five_hdr = oi_crop(oi_compute(wvf_five, hdr_scene), "border")

    defocus_wvf = wvf_set(wvf_five, "zcoeffs", 1.0, "defocus")
    aperture_defocus, params_defocus = wvf_aperture(
        defocus_wvf,
        "nsides",
        3,
        "dot mean",
        20,
        "dot sd",
        3,
        "dot opacity",
        0.5,
        "line mean",
        20,
        "line sd",
        2,
        "line opacity",
        0.5,
        "image rotate",
        0,
        "seed",
        3,
    )
    defocus_wvf = wvf_pupil_function(defocus_wvf, "aperture function", aperture_defocus)
    defocus_wvf = wvf_compute_psf(defocus_wvf, "compute pupil func", False)
    psf_defocus = np.asarray(wvf_get(defocus_wvf, "psf", 550.0), dtype=float)
    oi_defocus_hdr = oi_compute(defocus_wvf, hdr_scene)

    assert np.isclose(float(scene_get(point_scene, "fov")), 1.0)
    assert np.isclose(float(scene_get(hdr_scene, "fov")), 1.0)
    assert np.isclose(float(wvf_get(base_wvf, "fnumber")), 7.0 / 3.0)
    assert int(params_initial["nsides"]) == 3
    assert int(params_five["nsides"]) == 5
    assert int(params_defocus["nsides"]) == 3
    assert psf_initial.shape == psf_five.shape == psf_defocus.shape
    assert not np.allclose(psf_initial, psf_five)
    assert not np.allclose(psf_five, psf_defocus)
    assert np.asarray(oi_get(oi_initial_point, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_initial_hdr, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_five_point, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_five_hdr, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_defocus_hdr, "size"), dtype=int).shape == (2,)


def test_optics_flare2_script_workflow(asset_store) -> None:
    point_scene = scene_create("point array", 384, 128, asset_store=asset_store)
    point_scene = scene_set(point_scene, "fov", 1.0)
    hdr_scene = scene_create("hdr", asset_store=asset_store)
    hdr_scene = scene_set(hdr_scene, "fov", 3.0)

    base_wvf = wvf_create()
    base_wvf = wvf_set(base_wvf, "calc pupil diameter", 3.0)
    base_wvf = wvf_set(base_wvf, "focal length", 7e-3)

    aperture_initial, params_initial = wvf_aperture(
        base_wvf,
        "nsides",
        6,
        "dot mean",
        20,
        "dot sd",
        3,
        "dot opacity",
        0.5,
        "line mean",
        20,
        "line sd",
        2,
        "line opacity",
        0.5,
        "seed",
        4,
    )
    wvf_initial = wvf_pupil_function(base_wvf, "aperture function", aperture_initial)
    wvf_initial = wvf_compute(wvf_initial)
    psf_initial = np.asarray(wvf_get(wvf_initial, "psf", 550.0), dtype=float)
    oi_initial_point = oi_crop(oi_compute(wvf_initial, point_scene), "border")
    oi_initial_hdr = oi_compute(wvf_initial, hdr_scene)

    aperture_five, params_five = wvf_aperture(
        wvf_initial,
        "nsides",
        5,
        "dot mean",
        20,
        "dot sd",
        3,
        "dot opacity",
        0.5,
        "line mean",
        20,
        "line sd",
        2,
        "line opacity",
        0.5,
        "seed",
        5,
    )
    wvf_five = wvf_pupil_function(wvf_initial, "aperture function", aperture_five)
    wvf_five = wvf_compute_psf(wvf_five)
    psf_five = np.asarray(wvf_get(wvf_five, "psf", 550.0), dtype=float)
    oi_five_point = oi_crop(oi_compute(wvf_five, point_scene), "border")
    oi_five_hdr = oi_crop(oi_compute(wvf_five, hdr_scene), "border")

    defocus_wvf = wvf_set(wvf_five, "zcoeffs", 1.5, "defocus")
    aperture_defocus, params_defocus = wvf_aperture(
        defocus_wvf,
        "nsides",
        3,
        "dot mean",
        20,
        "dot sd",
        3,
        "dot opacity",
        0.5,
        "line mean",
        20,
        "line sd",
        2,
        "line opacity",
        0.5,
        "seed",
        6,
    )
    defocus_wvf = wvf_pupil_function(defocus_wvf, "aperture function", aperture_defocus)
    defocus_wvf = wvf_compute_psf(defocus_wvf)
    psf_defocus = np.asarray(wvf_get(defocus_wvf, "psf", 550.0), dtype=float)
    oi_defocus_hdr = oi_compute(defocus_wvf, hdr_scene)

    assert np.isclose(float(scene_get(point_scene, "fov")), 1.0)
    assert np.isclose(float(scene_get(hdr_scene, "fov")), 3.0)
    assert np.isclose(float(wvf_get(base_wvf, "fnumber")), 7.0 / 3.0)
    assert int(params_initial["nsides"]) == 6
    assert int(params_five["nsides"]) == 5
    assert int(params_defocus["nsides"]) == 3
    assert psf_initial.shape == psf_five.shape == psf_defocus.shape
    assert not np.allclose(psf_initial, psf_five)
    assert not np.allclose(psf_five, psf_defocus)
    assert np.asarray(oi_get(oi_initial_point, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_initial_hdr, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_five_point, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_five_hdr, "size"), dtype=int).shape == (2,)
    assert np.asarray(oi_get(oi_defocus_hdr, "size"), dtype=int).shape == (2,)


def test_run_python_case_supports_oi_pad_crop_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_pad_crop_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (128, 128)
    assert tuple(case.payload["oi_padded_size"]) == (160, 160)
    assert tuple(case.payload["crop_rect"]) == (17, 17, 127, 127)
    assert tuple(case.payload["oi_cropped_size"]) == (128, 128)
    assert float(case.payload["scene_fov_deg"]) == pytest.approx(10.0)
    assert float(case.payload["oi_cropped_fov_deg"]) > float(case.payload["scene_fov_deg"])
    assert float(case.payload["oi_cropped_fov_deg"]) < float(case.payload["oi_padded_fov_deg"])
    assert case.payload["sensor_scene_fov_pos_um"].shape == case.payload["sensor_scene_fov_padded_row"].shape
    assert case.payload["sensor_scene_fov_cropped_row"].shape == case.payload["sensor_scene_fov_pos_um"].shape
    assert float(case.payload["sensor_scene_fov_normalized_mae"]) < 0.02
    assert case.payload["sensor_padded_pos_um"].shape == case.payload["sensor_padded_row"].shape
    assert tuple(case.payload["sensor_padded_size"])[0] > tuple(case.payload["sensor_scene_fov_size"])[0]


def test_run_python_case_supports_psf550_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psf550_diffraction_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 2
    assert case.payload["y"].ndim == 2
    assert case.payload["psf"].shape == case.payload["x"].shape == case.payload["y"].shape
    assert np.all(case.payload["psf"] >= 0.0)


def test_run_python_case_supports_psfxaxis_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psfxaxis_diffraction_small", asset_store=asset_store)

    assert case.payload["samp"].ndim == 1
    assert case.payload["data"].shape == case.payload["samp"].shape
    assert np.isclose(float(case.payload["wave"]), 550.0)
    assert np.all(case.payload["data"] >= 0.0)


def test_run_python_case_supports_psfyaxis_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psfyaxis_diffraction_small", asset_store=asset_store)

    assert case.payload["samp"].ndim == 1
    assert case.payload["data"].shape == case.payload["samp"].shape
    assert np.isclose(float(case.payload["wave"]), 550.0)
    assert np.all(case.payload["data"] >= 0.0)


def test_run_python_case_supports_psf_plot_diffraction_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psf_plot_diffraction_small", asset_store=asset_store)

    assert case.payload["x"].shape == (200, 200)
    assert case.payload["y"].shape == (200, 200)
    assert case.payload["psf"].shape == (200, 200)
    assert np.all(case.payload["psf"] >= 0.0)
    assert float(case.payload["airy_disk_radius_um"]) == pytest.approx(8.784)


def test_run_python_case_supports_psfxaxis_wvf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psfxaxis_wvf_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["wave"]), 550.0)
    assert case.payload["oi_samp"].ndim == 1
    assert case.payload["oi_data"].shape == case.payload["oi_samp"].shape
    assert case.payload["wvf_samp"].shape == case.payload["oi_samp"].shape
    assert case.payload["wvf_data"].shape == case.payload["oi_samp"].shape
    assert np.all(case.payload["oi_data"] >= 0.0)
    assert np.all(case.payload["wvf_data"] >= 0.0)


def test_run_python_case_supports_psfyaxis_wvf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psfyaxis_wvf_small", asset_store=asset_store)

    assert np.isclose(float(case.payload["wave"]), 550.0)
    assert case.payload["oi_samp"].ndim == 1
    assert case.payload["oi_data"].shape == case.payload["oi_samp"].shape
    assert case.payload["wvf_samp"].shape == case.payload["oi_samp"].shape
    assert case.payload["wvf_data"].shape == case.payload["oi_samp"].shape
    assert np.all(case.payload["oi_data"] >= 0.0)
    assert np.all(case.payload["wvf_data"] >= 0.0)


def test_run_python_case_supports_psf550_wvf_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_psf550_wvf_small", asset_store=asset_store)

    assert case.payload["x"].ndim == 2
    assert case.payload["y"].ndim == 2
    assert case.payload["psf"].shape == case.payload["x"].shape
    assert case.payload["psf"].shape == case.payload["y"].shape
    assert np.all(case.payload["psf"] >= 0.0)


def test_run_python_case_supports_oi_wvf_otf_compare_parity_case(asset_store) -> None:
    case = run_python_case_with_context("oi_wvf_otf_compare_small", asset_store=asset_store)

    assert case.payload["oi_otf_abs"].ndim == 2
    assert case.payload["wvf_otf_abs_shifted"].shape == case.payload["oi_otf_abs"].shape
    assert np.all(case.payload["oi_otf_abs"] >= 0.0)
    assert np.all(case.payload["wvf_otf_abs_shifted"] >= 0.0)


def test_run_python_case_supports_unit_frequency_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_unit_frequency_list", asset_store=asset_store)

    assert np.allclose(case.payload["even"], np.asarray(case.payload["even"], dtype=float))
    assert np.allclose(case.payload["odd"], np.asarray(case.payload["odd"], dtype=float))
    assert case.payload["even"].shape == (50,)
    assert case.payload["odd"].shape == (51,)


def test_run_python_case_supports_energy_quanta_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_energy_quanta_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["energy"].shape == (31,)
    assert case.payload["photons"].shape == (31,)
    assert np.allclose(case.payload["energy_roundtrip"], case.payload["energy"])


def test_run_python_case_supports_energy_quanta_matrix_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_energy_quanta_matrix", asset_store=asset_store)

    assert case.payload["wave"].shape == (3,)
    assert case.payload["energy"].shape == (3, 2)
    assert case.payload["photons"].shape == (3, 2)
    assert np.allclose(case.payload["energy_roundtrip"], case.payload["energy"])


def test_run_python_case_supports_blackbody_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_blackbody_energy_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["temperatures"].shape == (2,)
    assert case.payload["energy"].shape == (31, 2)
    assert np.all(np.asarray(case.payload["energy"], dtype=float) > 0.0)


def test_run_python_case_supports_blackbody_quanta_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_blackbody_quanta_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["temperatures"].shape == (2,)
    assert case.payload["photons"].shape == (31, 2)
    assert np.all(np.asarray(case.payload["photons"], dtype=float) > 0.0)


def test_run_python_case_supports_ie_param_format_utility_parity_case(asset_store) -> None:
    case = run_python_case_with_context("utility_ie_param_format_string", asset_store=asset_store)

    assert case.payload["original"] == "Exposure Time"
    assert case.payload["formatted"] == "exposuretime"


def test_run_python_case_supports_xyz_from_energy_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_from_energy_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["energy"].shape == (31,)
    assert case.payload["xyz"].shape == (3,)
    assert np.all(np.asarray(case.payload["xyz"], dtype=float) > 0.0)


def test_run_python_case_supports_xyz_to_luv_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_to_luv_1d", asset_store=asset_store)

    assert case.payload["xyz"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)
    assert case.payload["luv"].shape == (3,)
    assert case.payload["luv"][0] > 0.0


def test_run_python_case_supports_xyz_to_lab_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_to_lab_1d", asset_store=asset_store)

    assert case.payload["xyz"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)
    assert case.payload["lab"].shape == (3,)
    assert case.payload["lab"][0] > 0.0


def test_run_python_case_supports_xyz_to_uv_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_xyz_to_uv_1d", asset_store=asset_store)

    assert case.payload["xyz"].shape == (3,)
    assert case.payload["uv"].shape == (2,)
    assert np.all(np.asarray(case.payload["uv"], dtype=float) > 0.0)


def test_run_python_case_supports_cct_from_uv_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_cct_from_uv_1d", asset_store=asset_store)

    assert case.payload["uv"].shape == (2,)
    assert float(case.payload["cct_k"]) > 0.0


def test_run_python_case_supports_delta_e_ab_metrics_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_delta_e_ab_1976_1d", asset_store=asset_store)

    assert case.payload["xyz1"].shape == (3,)
    assert case.payload["xyz2"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)
    assert float(case.payload["delta_e"]) > 0.0


def test_run_python_case_supports_metrics_spd_angle_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_angle_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (3,)
    assert case.payload["spd1"].shape == (3,)
    assert case.payload["spd2"].shape == (3,)
    assert np.isclose(float(case.payload["angle"]), 90.0)


def test_run_python_case_supports_metrics_spd_cielab_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_cielab_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["spd1"].shape == (31,)
    assert case.payload["spd2"].shape == (31,)
    assert float(case.payload["delta_e"]) > 0.0
    assert case.payload["xyz1"].shape == (3,)
    assert case.payload["xyz2"].shape == (3,)
    assert case.payload["lab1"].shape == (3,)
    assert case.payload["lab2"].shape == (3,)
    assert case.payload["white_point"].shape == (3,)


def test_run_python_case_supports_metrics_spd_mired_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_mired_1d", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["spd1"].shape == (31,)
    assert case.payload["spd2"].shape == (31,)
    assert float(case.payload["mired"]) > 0.0
    assert case.payload["uv"].shape == (2, 2)
    assert case.payload["cct_k"].shape == (2,)


def test_metrics_spd_script_daylight_sweep_workflow() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    ctemp = np.arange(4000.0, 7000.0 + 1.0, 500.0, dtype=float)

    standard_4000 = np.asarray(daylight(wave, 4000.0), dtype=float)
    d4000_angle = np.zeros(ctemp.size, dtype=float)
    d4000_delta_e = np.zeros(ctemp.size, dtype=float)
    d4000_mired = np.zeros(ctemp.size, dtype=float)
    for index, color_temperature in enumerate(ctemp):
        comparison = np.asarray(daylight(wave, float(color_temperature)), dtype=float)
        d4000_angle[index] = float(metrics_spd(standard_4000, comparison, metric="angle", wave=wave))
        d4000_delta_e[index] = float(metrics_spd(standard_4000, comparison, metric="cielab", wave=wave))
        d4000_mired[index] = float(metrics_spd(standard_4000, comparison, metric="mired", wave=wave))

    assert np.isclose(d4000_mired[-1], 114.3814, atol=1.0e-4)
    assert np.isclose(d4000_angle[-1], 25.0450, atol=1.0e-4)

    d65_white_point = np.array([94.9409, 100.0, 108.6656], dtype=float)
    standard_6500 = np.asarray(daylight(wave, 6500.0), dtype=float)
    d6500_angle = np.zeros(ctemp.size, dtype=float)
    d6500_delta_e = np.zeros(ctemp.size, dtype=float)
    d6500_mired = np.zeros(ctemp.size, dtype=float)
    for index, color_temperature in enumerate(ctemp):
        comparison = np.asarray(daylight(wave, float(color_temperature)), dtype=float)
        d6500_angle[index] = float(metrics_spd(standard_6500, comparison, metric="angle", wave=wave))
        d6500_delta_e[index] = float(
            metrics_spd(
                standard_6500,
                comparison,
                metric="cielab",
                wave=wave,
                white_point=d65_white_point,
            )
        )
        d6500_mired[index] = float(metrics_spd(standard_6500, comparison, metric="mired", wave=wave))

    assert np.isclose(d6500_mired[-1], 12.0726, atol=1.0e-4)
    assert np.isclose(d6500_angle[-1], 2.6800, atol=1.0e-4)
    assert np.all(d4000_delta_e >= 0.0)
    assert np.all(d6500_delta_e >= 0.0)


def test_run_python_case_supports_metrics_spd_daylight_sweep_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_spd_daylight_sweep_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["ctemp_k"].shape == (7,)
    assert case.payload["d65_white_point"].shape == (3,)
    assert case.payload["d4000_angle"].shape == (7,)
    assert case.payload["d4000_delta_e"].shape == (7,)
    assert case.payload["d4000_mired"].shape == (7,)
    assert case.payload["d6500_angle"].shape == (7,)
    assert case.payload["d6500_delta_e"].shape == (7,)
    assert case.payload["d6500_mired"].shape == (7,)
    assert np.isclose(float(case.payload["d4000_mired"][-1]), 114.3814, atol=1.0e-4)
    assert np.isclose(float(case.payload["d6500_mired"][-1]), 12.0726, atol=1.0e-4)


def test_metrics_vsnr_script_workflow(asset_store) -> None:
    levels = np.asarray(np.logspace(1.5, 3.0, 3), dtype=float)
    result = camera_vsnr(camera_create(asset_store=asset_store), levels, asset_store=asset_store)
    placeholder_result = camera_vsnr(camera_create(asset_store=asset_store), [], [], asset_store=asset_store)
    default_result = camera_vsnr(camera_create(asset_store=asset_store), asset_store=asset_store)

    valid = np.asarray(result.vSNR, dtype=float)[np.isfinite(result.vSNR)]
    assert result.lightLevels.shape == (3,)
    assert result.eTime.shape == (3,)
    assert result.rect.shape == (4,)
    assert len(result.ip) == 3
    assert np.all(result.eTime == 0.0)
    assert valid.size == 3
    assert np.all(valid > 0.0)
    assert np.all(np.diff(valid) > 0.0)
    np.testing.assert_allclose(placeholder_result.lightLevels, default_result.lightLevels, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_result.vSNR, default_result.vSNR, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_result.eTime, default_result.eTime, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(placeholder_result.rect, default_result.rect)


def test_camera_legacy_full_reference_and_vsnr_alias(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    camera = camera_set(camera, "sensor size", (48, 64))

    scene_path = asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
    reference = cameraFullReference(camera, [scene_path], [25.0], asset_store=asset_store)

    assert len(reference.sceneNames) == 1
    assert "StuffedAnimals_tungsten-hdrs" in reference.sceneNames[0]
    assert reference.meanLuminances.shape == (1,)
    assert reference.scielab.shape == (1, 1)
    assert reference.ssim.shape == (1, 1)
    assert np.isfinite(float(reference.scielab[0, 0]))
    assert 0.0 <= float(reference.ssim[0, 0]) <= 1.0

    placeholder_reference = cameraFullReference(camera, [], [], asset_store=asset_store)
    default_reference = cameraFullReference(camera, asset_store=asset_store)
    np.testing.assert_equal(placeholder_reference.sceneNames, default_reference.sceneNames)
    np.testing.assert_allclose(placeholder_reference.meanLuminances, default_reference.meanLuminances, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_reference.scielab, default_reference.scielab, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(placeholder_reference.ssim, default_reference.ssim, rtol=1e-10, atol=1e-12)

    levels = np.asarray([50.0, 100.0, 200.0], dtype=float)
    expected = camera_vsnr(camera, levels, asset_store=asset_store)
    legacy = cameraVSNR_SL(camera, levels, asset_store=asset_store)

    assert np.array_equal(legacy.lightLevels, expected.lightLevels)
    assert np.allclose(legacy.vSNR, expected.vSNR, equal_nan=True)
    assert np.array_equal(legacy.rect, expected.rect)
    assert np.allclose(legacy.eTime, expected.eTime)


def test_run_python_case_supports_metrics_vsnr_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_vsnr_small", asset_store=asset_store)

    assert case.payload["light_levels"].shape == (3,)
    assert case.payload["rect"].shape == (4,)
    assert case.payload["saturation_mask"].shape == (3,)
    assert case.payload["vsnr_norm"].shape == (3,)
    assert case.payload["delta_e_norm"].shape == (3,)
    assert case.payload["result_channel_means_norm"].shape == (3, 3)


def test_scielab_rgb_tutorial_workflow(asset_store) -> None:
    error_image, scene1, scene2, display = scielab_rgb(
        "hats.jpg",
        "hatsC.jpg",
        "LCD-Apple.mat",
        0.3,
        asset_store=asset_store,
    )

    scene1_size = np.asarray(scene_get(scene1, "size"), dtype=int)
    scene2_size = np.asarray(scene_get(scene2, "size"), dtype=int)
    white_point = np.asarray(display_get(display, "white point"), dtype=float)

    assert np.array_equal(scene1_size, np.array([128, 192], dtype=int))
    assert np.array_equal(scene1_size, scene2_size)
    assert error_image.ndim == 2
    assert abs(error_image.shape[0] - int(scene1_size[0])) <= 1
    assert abs(error_image.shape[1] - int(scene1_size[1])) <= 1
    assert np.isclose(float(scene_get(scene1, "fov")), float(scene_get(scene2, "fov")), atol=1e-12, rtol=1e-12)
    assert white_point.shape == (3,)
    assert np.all(white_point > 0.0)
    assert np.isclose(float(scene_get(scene1, "fov")), 9.679001581355232, atol=1e-12, rtol=1e-12)
    assert float(np.mean(error_image, dtype=float)) > 0.0
    assert float(np.max(error_image)) > float(np.mean(error_image, dtype=float))


def test_run_python_case_supports_metrics_scielab_rgb_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_rgb_small", asset_store=asset_store)

    assert tuple(case.payload["error_size"]) == (127, 191)
    assert tuple(case.payload["scene1_size"]) == (128, 192)
    assert tuple(case.payload["scene2_size"]) == (128, 192)
    assert np.isclose(float(case.payload["fov_deg"]), 9.679001581355232, atol=1e-12, rtol=1e-12)
    assert case.payload["display_white_point"].shape == (3,)
    assert case.payload["error_stats"].shape == (4,)
    assert case.payload["error_center_row_norm"].shape == (129,)


def test_metrics_rgb2scielab_script_workflow(asset_store) -> None:
    error_image, scene1, scene2, display = scielab_rgb(
        "hats.jpg",
        "hatsC.jpg",
        "crt.mat",
        0.3,
        asset_store=asset_store,
    )

    mask = np.asarray(error_image, dtype=float) > 2.0
    mean_above2 = float(np.mean(np.asarray(error_image, dtype=float)[mask], dtype=float))
    percent_above2 = float(np.count_nonzero(mask)) / float(np.asarray(error_image).size) * 100.0

    assert tuple(scene_get(scene1, "size")) == (128, 192)
    assert tuple(scene_get(scene2, "size")) == (128, 192)
    assert tuple(np.asarray(error_image).shape) == (127, 191)
    assert np.isclose(float(scene_get(scene1, "fov")), 8.783389963820218, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(np.mean(error_image, dtype=float)), 1.5116844348691976, atol=1e-12, rtol=1e-12)
    assert np.isclose(mean_above2, 2.8235651171190823, atol=1e-12, rtol=1e-12)
    assert np.isclose(percent_above2, 21.754545079770786, atol=1e-12, rtol=1e-12)
    assert np.all(np.asarray(display_get(display, "white point"), dtype=float) > 0.0)


def test_run_python_case_supports_metrics_rgb2scielab_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_rgb2scielab_small", asset_store=asset_store)

    assert tuple(case.payload["error_size"]) == (127, 191)
    assert tuple(case.payload["scene1_size"]) == (128, 192)
    assert tuple(case.payload["scene2_size"]) == (128, 192)
    assert np.isclose(float(case.payload["fov_deg"]), 8.783389963820218, atol=1e-12, rtol=1e-12)
    assert case.payload["display_white_point"].shape == (3,)
    assert np.isclose(float(case.payload["mean_delta_e"]), 1.5116844348691976, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(case.payload["mean_delta_e_above2"]), 2.8235651171190823, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(case.payload["percent_above2"]), 21.754545079770786, atol=1e-12, rtol=1e-12)
    assert case.payload["error_center_row_norm"].shape == (129,)


def test_metrics_scielab_example_script_workflow(asset_store) -> None:
    root = asset_store.ensure()
    hats = dac_to_rgb(iio.imread(root / "data" / "images" / "rgb" / "hats.jpg").astype(float) / 255.0)
    hats_c = dac_to_rgb(iio.imread(root / "data" / "images" / "rgb" / "hatsC.jpg").astype(float) / 255.0)
    dsp = display_create(str(root / "data" / "displays" / "crt.mat"), asset_store=asset_store)

    rgb2xyz = np.asarray(display_get(dsp, "rgb2xyz"), dtype=float)
    white_xyz = np.asarray(display_get(dsp, "white point"), dtype=float)
    img1_xyz = image_linear_transform(hats, rgb2xyz)
    img2_xyz = image_linear_transform(hats_c, rgb2xyz)

    img_width = hats.shape[1] * float(display_get(dsp, "meters per dot"))
    fov = float(np.rad2deg(2.0 * np.arctan2(img_width / 2.0, 0.3)))
    samp_per_deg = hats.shape[1] / fov
    params = {
        "deltaEversion": "2000",
        "sampPerDeg": samp_per_deg,
        "imageFormat": "xyz",
        "filterSize": samp_per_deg,
        "filters": [],
    }
    error_image, params_out, _, _ = scielab(img1_xyz, img2_xyz, white_xyz, params)

    mask = np.asarray(error_image, dtype=float) > 2.0
    mean_above2 = float(np.mean(np.asarray(error_image, dtype=float)[mask], dtype=float))
    percent_above2 = float(np.count_nonzero(mask)) / float(np.asarray(error_image).size) * 100.0

    assert hats.shape == (128, 192, 3)
    assert hats_c.shape == (128, 192, 3)
    assert rgb2xyz.shape == (3, 3)
    assert white_xyz.shape == (3,)
    assert np.isclose(fov, 8.783389963820218, atol=1e-12, rtol=1e-12)
    assert tuple(np.asarray(error_image).shape) == (127, 191)
    assert np.isclose(float(np.mean(error_image, dtype=float)), 1.8293812098678626, atol=1e-12, rtol=1e-12)
    assert np.isclose(mean_above2, 2.9118542384189685, atol=1e-12, rtol=1e-12)
    assert np.isclose(percent_above2, 35.96487611823391, atol=1e-12, rtol=1e-12)
    assert np.array_equal(np.asarray(params_out["filters"][0]).shape, np.array([21, 21], dtype=int))
    assert np.asarray(params_out["support"]).shape == (21,)


def test_run_python_case_supports_metrics_scielab_example_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_example_small", asset_store=asset_store)

    assert tuple(case.payload["scene1_size"]) == (128, 192)
    assert tuple(case.payload["scene2_size"]) == (128, 192)
    assert np.isclose(float(case.payload["fov_deg"]), 8.783389963820218, atol=1e-12, rtol=1e-12)
    assert case.payload["display_white_point"].shape == (3,)
    assert np.isclose(float(case.payload["scielab_rgb_mean_delta_e"]), 1.5116844348691976, atol=1e-12, rtol=1e-12)
    assert tuple(case.payload["explicit_error_size"]) == (127, 191)
    assert np.isclose(float(case.payload["explicit_mean_delta_e"]), 1.8293812098678626, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(case.payload["explicit_mean_delta_e_above2"]), 2.9118542384189685, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(case.payload["explicit_percent_above2"]), 35.96487611823391, atol=1e-12, rtol=1e-12)
    assert case.payload["filter_support"].shape == (21,)
    assert case.payload["filter_peaks"].shape == (3,)
    assert case.payload["filter_center_rows_norm"].shape == (3, 65)
    assert case.payload["explicit_error_center_row_norm"].shape == (129,)


def test_scielab_legacy_color_space_wrappers_match_current_helpers() -> None:
    image = np.arange(1, 13, dtype=float).reshape(2, 2, 3) / 12.0

    xyz_to_opp = cmatrix("xyz2opp", 10)
    transformed = changeColorSpace(image, xyz_to_opp)
    expected = image_linear_transform(image, colorTransformMatrix("xyz2opp", 10))

    assert np.allclose(transformed, expected)
    assert np.allclose(cmatrix("opp2xyz", 10), np.linalg.inv(cmatrix("xyz2opp", 10)))
    assert np.allclose(cmatrix("lms2xyz"), np.linalg.inv(cmatrix("xyz2lms")))


def test_scielab_legacy_filter_helpers_match_internal_support() -> None:
    kernel = gauss(2.0, 7)
    image = np.arange(1, 10, dtype=float).reshape(3, 3)
    fft_kernel = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)

    assert kernel.shape == (7,)
    assert np.isclose(float(np.sum(kernel)), 1.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(kernel, kernel[::-1])

    padded = pad4conv(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), 2)
    assert np.array_equal(
        padded,
        np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ],
            dtype=float,
        ),
    )

    fft_full = convolve2d(image, fft_kernel, mode="full")
    dy = fft_kernel.shape[0] // 2
    dx = fft_kernel.shape[1] // 2
    expected_same = fft_full[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    assert np.allclose(ieConv2FFT(image, fft_kernel, "same"), expected_same)

    k1, k2, k3 = separableFilters(224, 1, 7)
    assert k1.shape == k2.shape == k3.shape == (7,)
    assert np.isclose(float(np.sum(k1)), 1.0, atol=1e-12, rtol=1e-12)
    k1_sep, k2_sep, k3_sep = separableFilters(56, 3, 7)
    assert k1_sep.shape[0] == 3
    assert k2_sep.shape[0] == 2
    assert k3_sep.shape[0] == 2

    xkernels = np.array([[0.25, 0.5, 0.25]], dtype=float)
    expected = convolve2d(pad4conv(image, xkernels.shape[1], 2), xkernels, mode="full")
    expected = scResize(expected, image.shape)
    expected = convolve2d(pad4conv(expected, xkernels.shape[1], 1), xkernels.T, mode="full")
    expected = scResize(expected, image.shape)
    assert np.allclose(separableConv(image, xkernels), expected)

    opponent = np.dstack((image, image + 10.0, image + 20.0))
    identity_filters = [
        np.pad(np.array([[1.0]], dtype=float), ((1, 1), (1, 1)), mode="constant"),
        np.pad(np.array([[1.0]], dtype=float), ((1, 1), (1, 1)), mode="constant"),
        np.pad(np.array([[1.0]], dtype=float), ((1, 1), (1, 1)), mode="constant"),
    ]
    expected_filtered = sc_apply_filters(opponent, identity_filters)
    np.testing.assert_allclose(ApplyFilters(opponent, identity_filters), expected_filtered, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scApplyFilters(opponent, identity_filters), expected_filtered, rtol=1e-10, atol=1e-12)


def test_scielab_legacy_preprocess_and_plane_helpers() -> None:
    plane_image = np.arange(1, 19, dtype=float).reshape(2, 9)
    p1, p3 = getPlanes(plane_image, [1, 3])
    i1, i2, i3 = getPlanes(6)

    assert np.array_equal(p1, plane_image[:, :3])
    assert np.array_equal(p3, plane_image[:, 6:9])
    assert np.array_equal(i1, np.array([[1.0, 2.0]], dtype=float))
    assert np.array_equal(i2, np.array([[3.0, 4.0]], dtype=float))
    assert np.array_equal(i3, np.array([[5.0, 6.0]], dtype=float))

    resized = scResize(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), [4, 4], 0, -1.0)
    assert resized.shape == (4, 4)
    assert np.array_equal(resized[1:3, 1:3], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    assert np.all(resized[[0, -1], :] == -1.0)

    ref = np.array([[0.5, 1.0], [0.25, 0.75]], dtype=float)
    cmp = np.array([[2.0]], dtype=float)
    ref_out, cmp_out = preSCIELAB(ref, cmp)
    assert ref_out.shape == cmp_out.shape == (2, 6)
    assert np.isclose(float(np.max(ref_out)), 1.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(np.max(cmp_out)), 1.0, atol=1e-12, rtol=1e-12)

    assert np.isclose(visualAngle(20, 12, 72), np.rad2deg(np.arctan((20.0 / 72.0) / 12.0)))
    assert np.isclose(visualAngle(-1, 12, 72, 5), 72.0 * 12.0 * np.tan(np.deg2rad(5.0)))


def test_metrics_scielab_filters_script_workflow() -> None:
    filters_initial, support_initial, params_initial = sc_prepare_filters({"sampPerDeg": 101.0, "filterSize": 101.0})

    assert int(round(float(params_initial["filterSize"]))) == 101
    assert np.asarray(support_initial).shape == (101,)
    assert np.isclose(float(np.asarray(support_initial)[0]), -50.0 / 101.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(np.asarray(support_initial)[-1]), 50.0 / 101.0, atol=1e-12, rtol=1e-12)
    for kernel in filters_initial:
        array = np.asarray(kernel, dtype=float)
        assert array.shape == (101, 101)
        assert np.isclose(float(np.sum(array, dtype=float)), 1.0, atol=1e-12, rtol=1e-12)

    filters_mtf, _, params_mtf = sc_prepare_filters({"sampPerDeg": 512.0, "filterSize": 512.0})
    assert int(round(float(params_mtf["filterSize"]))) == 511
    mtf_peaks = []
    for kernel in filters_mtf:
        mtf = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.fftshift(np.asarray(kernel, dtype=float)))))
        mtf_peaks.append(float(np.max(mtf)))
    assert np.all(np.asarray(mtf_peaks, dtype=float) > 0.99)

    for version in ("distribution", "original", "hires"):
        version_filters, version_support, version_params = sc_prepare_filters(
            {"sampPerDeg": 350.0, "filterSize": 200.0, "filterversion": version}
        )
        assert int(round(float(version_params["filterSize"]))) == 199
        assert np.asarray(version_support).shape == (199,)
        for kernel in version_filters:
            array = np.asarray(kernel, dtype=float)
            assert array.shape == (199, 199)
            assert np.isclose(float(np.sum(array, dtype=float)), 1.0, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_metrics_scielab_filters_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_filters_small", asset_store=asset_store)

    assert int(case.payload["initial_filter_size"]) == 101
    assert case.payload["initial_support"].shape == (101,)
    assert case.payload["initial_filter_center_rows_norm"].shape == (3, 129)
    assert np.allclose(np.asarray(case.payload["initial_filter_sums"], dtype=float), np.ones(3, dtype=float), atol=1e-12, rtol=1e-12)
    assert int(case.payload["mtf_filter_size"]) == 511
    assert case.payload["mtf_filter_center_rows_norm"].shape == (3, 129)
    assert np.all(np.asarray(case.payload["mtf_filter_peaks"], dtype=float) > 0.99)
    assert np.array_equal(np.asarray(case.payload["version_filter_sizes"], dtype=int), np.array([199, 199, 199], dtype=int))
    assert case.payload["version_support"].shape == (199,)
    assert case.payload["version_filter_center_rows_norm"].shape == (3, 3, 129)
    assert case.payload["version_mtf_center_rows_norm"].shape == (3, 3, 129)


def test_metrics_scielab_mtf_script_workflow(asset_store) -> None:
    f_list = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)
    standard_params = {
        "freq": f_list[0],
        "contrast": 0.0,
        "ph": 0.0,
        "ang": 0.0,
        "row": 128,
        "col": 128,
        "GaborFlag": 0.0,
    }
    standard_scene = scene_create("harmonic", standard_params, asset_store=asset_store)
    standard_scene = scene_set(standard_scene, "fov", 1.0)

    white_xyz = np.asarray(scene_get(standard_scene, "illuminant xyz", asset_store=asset_store), dtype=float)
    illuminant_energy = np.asarray(scene_get(standard_scene, "illuminant energy", asset_store=asset_store), dtype=float)
    wave = np.asarray(scene_get(standard_scene, "wave"), dtype=float)

    delta_e = np.zeros(f_list.size, dtype=float)
    scielab_delta_e = np.zeros(f_list.size, dtype=float)
    for idx, frequency in enumerate(f_list):
        test_params = dict(standard_params)
        test_params["freq"] = float(frequency)
        test_params["contrast"] = 0.5
        test_scene = scene_create("harmonic", test_params, asset_store=asset_store)
        test_scene = scene_set(test_scene, "fov", 1.0)
        test_scene = scene_add(standard_scene, test_scene, "remove spatial mean")

        xyz1 = np.asarray(scene_get(standard_scene, "xyz", asset_store=asset_store), dtype=float)
        xyz2 = np.asarray(scene_get(test_scene, "xyz", asset_store=asset_store), dtype=float)
        delta_e[idx] = float(np.mean(delta_e_ab(xyz1, xyz2, white_xyz, "2000"), dtype=float))
        error_image, _, _, _ = scielab(xyz1, xyz2, white_xyz, sc_params())
        scielab_delta_e[idx] = float(np.mean(np.asarray(error_image, dtype=float), dtype=float))

    assert tuple(scene_get(standard_scene, "size")) == (128, 128)
    assert np.isclose(float(scene_get(standard_scene, "fov")), 1.0, atol=1e-12, rtol=1e-12)
    assert white_xyz.shape == (3,)
    assert illuminant_energy.shape == wave.shape
    assert np.all(delta_e > 0.0)
    assert np.all(scielab_delta_e > 0.0)
    assert np.array_equal(f_list, np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=float))


def test_run_python_case_supports_metrics_scielab_mtf_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_mtf_small", asset_store=asset_store)

    assert np.array_equal(case.payload["frequencies_cpd"], np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=float))
    assert tuple(case.payload["standard_scene_size"]) == (128, 128)
    assert np.isclose(float(case.payload["standard_fov_deg"]), 1.0, atol=1e-12, rtol=1e-12)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["white_xyz"].shape == (3,)
    assert case.payload["illuminant_energy_norm"].shape == (31,)
    assert case.payload["delta_e"].shape == (6,)
    assert case.payload["scielab_delta_e"].shape == (6,)
    assert case.payload["scielab_over_delta_e"].shape == (6,)
    assert np.all(np.asarray(case.payload["delta_e"], dtype=float) > 0.0)
    assert np.all(np.asarray(case.payload["scielab_delta_e"], dtype=float) > 0.0)


def test_metrics_scielab_patches_script_workflow(asset_store) -> None:
    u_standard = scene_create("uniform", asset_store=asset_store)

    white_xyz = np.asarray(scene_get(u_standard, "illuminant xyz", asset_store=asset_store), dtype=float)
    illuminant_energy = np.asarray(scene_get(u_standard, "illuminant energy", asset_store=asset_store), dtype=float)
    wave = np.asarray(scene_get(u_standard, "wave"), dtype=float)
    n_wave = int(scene_get(u_standard, "nwave"))
    lam = np.arange(1, n_wave + 1, dtype=float) / float(n_wave)

    w1_grid, w2_grid = np.meshgrid(
        np.arange(-0.3, 0.3 + 0.0001, 0.1, dtype=float),
        np.arange(-0.3, 0.3 + 0.0001, 0.1, dtype=float),
    )
    weights = np.column_stack((w1_grid.reshape(-1, order="F"), w2_grid.reshape(-1, order="F")))
    delta_e = np.ones(weights.shape[0], dtype=float)
    scielab_delta_e = np.ones(weights.shape[0], dtype=float)

    xyz1 = np.asarray(scene_get(u_standard, "xyz", asset_store=asset_store), dtype=float)
    for idx, (w1, w2) in enumerate(weights):
        e_adjust1 = float(w1) * np.sin(2.0 * np.pi * lam)
        e_adjust2 = float(w2) * np.cos(2.0 * np.pi * lam)
        new_illuminant = illuminant_energy * (float(w1) * e_adjust1 + float(w2) * e_adjust2 + 1.0)
        u_test = scene_adjust_illuminant(u_standard, new_illuminant)

        xyz2 = np.asarray(scene_get(u_test, "xyz", asset_store=asset_store), dtype=float)
        delta_e[idx] = float(np.mean(delta_e_ab(xyz1, xyz2, white_xyz, "2000"), dtype=float))
        error_image, _, _, _ = scielab(xyz1, xyz2, white_xyz, sc_params())
        scielab_delta_e[idx] = float(np.mean(np.asarray(error_image, dtype=float), dtype=float))

    quantized = 2.0 * np.round(scielab_delta_e / 2.0)

    assert tuple(scene_get(u_standard, "size")) == (32, 32)
    assert white_xyz.shape == (3,)
    assert illuminant_energy.shape == wave.shape == (31,)
    assert weights.shape == (49, 2)
    assert delta_e.shape == (49,)
    assert scielab_delta_e.shape == (49,)
    assert np.array_equal(quantized.shape, (49,))
    assert np.all(delta_e >= 0.0)
    assert np.all(scielab_delta_e >= 0.0)


def test_run_python_case_supports_metrics_scielab_patches_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_patches_small", asset_store=asset_store)

    assert case.payload["weights"].shape == (49, 2)
    assert tuple(case.payload["standard_scene_size"]) == (32, 32)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["white_xyz"].shape == (3,)
    assert case.payload["illuminant_energy_norm"].shape == (31,)
    assert case.payload["delta_e"].shape == (49,)
    assert case.payload["scielab_delta_e"].shape == (49,)
    assert case.payload["delta_gap"].shape == (49,)
    assert case.payload["delta_gap_stats"].shape == (2,)
    assert case.payload["quantized_scielab_delta_e"].shape == (49,)
    assert case.payload["quantized_scielab_delta_e_sorted"].shape == (49,)
    assert case.payload["quantized_scielab_levels"].shape == (6,)
    assert case.payload["quantized_scielab_counts"].shape == (6,)
    assert np.all(np.asarray(case.payload["delta_e"], dtype=float) >= 0.0)
    assert np.all(np.asarray(case.payload["scielab_delta_e"], dtype=float) >= 0.0)
    assert np.isclose(float(np.sum(np.asarray(case.payload["quantized_scielab_counts"], dtype=float))), 49.0)


def test_metrics_scielab_masking_script_workflow(asset_store) -> None:
    f_list = np.array([2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)
    t_list = np.arange(0.05, 0.2001, 0.05, dtype=float)
    mask_contrast = 0.8

    params = {"ph": 0.0, "ang": 0.0, "row": 128, "col": 128, "GaborFlag": 0.0, "freq": float(f_list[1]), "contrast": mask_contrast}
    mask = scene_create("harmonic", params, asset_store=asset_store)
    mask = scene_set(mask, "fov", 1.0)

    white_xyz = 2.0 * np.asarray(scene_get(mask, "illuminant xyz", asset_store=asset_store), dtype=float)
    delta_e = np.zeros(t_list.size, dtype=float)
    scielab_delta_e = np.zeros(t_list.size, dtype=float)

    xyz1 = np.maximum(np.asarray(scene_get(mask, "xyz", asset_store=asset_store), dtype=float), 0.0)
    for idx, contrast in enumerate(t_list):
        target_params = dict(params)
        target_params["contrast"] = float(contrast)
        target = scene_create("harmonic", target_params, asset_store=asset_store)
        target = scene_set(target, "fov", 1.0)
        combined = scene_add(mask, target, "remove spatial mean")

        xyz2 = np.maximum(np.asarray(scene_get(combined, "xyz", asset_store=asset_store), dtype=float), 0.0)
        delta_e[idx] = float(np.mean(delta_e_ab(xyz1, xyz2, white_xyz, "2000"), dtype=float))
        error_image, _, _, _ = scielab(xyz1, xyz2, white_xyz, sc_params())
        scielab_delta_e[idx] = float(np.mean(np.asarray(error_image, dtype=float), dtype=float))

    assert tuple(scene_get(mask, "size")) == (128, 128)
    assert np.isclose(float(scene_get(mask, "fov")), 1.0)
    assert white_xyz.shape == (3,)
    assert delta_e.shape == (4,)
    assert scielab_delta_e.shape == (4,)
    assert np.all(delta_e > 0.0)
    assert np.all(scielab_delta_e > delta_e)


def test_run_python_case_supports_metrics_scielab_masking_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_masking_small", asset_store=asset_store)

    assert case.payload["frequencies_cpd"].shape == (5,)
    assert np.isclose(float(case.payload["mask_frequency_cpd"]), 4.0)
    assert np.isclose(float(case.payload["mask_contrast"]), 0.8)
    assert case.payload["target_contrasts"].shape == (4,)
    assert tuple(case.payload["mask_scene_size"]) == (128, 128)
    assert np.isclose(float(case.payload["mask_fov_deg"]), 1.0)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["white_xyz"].shape == (3,)
    assert case.payload["illuminant_energy_norm"].shape == (31,)
    assert case.payload["delta_e"].shape == (4,)
    assert case.payload["scielab_delta_e"].shape == (4,)
    assert case.payload["scielab_over_delta_e"].shape == (4,)
    assert np.all(np.asarray(case.payload["delta_e"], dtype=float) > 0.0)
    assert np.all(np.asarray(case.payload["scielab_delta_e"], dtype=float) > np.asarray(case.payload["delta_e"], dtype=float))


def test_metrics_scielab_tutorial_script_workflow(asset_store) -> None:
    root = asset_store.ensure()
    scene_path = root / "data" / "images" / "multispectral" / "StuffedAnimals_tungsten-hdrs.mat"

    scene = scene_from_file(scene_path, "multispectral", asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)

    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_set_size_to_fov(sensor_create(asset_store=asset_store), 1.1 * float(scene_get(scene, "fov")), oi)
    sensor = sensor_compute(sensor, oi)

    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "gray world")
    ip = ip_compute(ip, sensor)

    srgb = np.asarray(ip_get(ip, "result"), dtype=float)
    img_xyz = srgb2xyz(srgb)
    white_xyz = srgb2xyz(np.ones((1, 1, 3), dtype=float)).reshape(3)

    params = sc_params()
    params["sampPerDeg"] = 50.0
    params["filterSize"] = 50.0

    img_opp = image_linear_transform(img_xyz, colorTransformMatrix("xyz2opp", 10))
    filters, support, params = sc_prepare_filters(params)
    img_filtered_xyz, img_filtered_opp = scOpponentFilter(img_xyz, params)
    result, white_pt = scComputeSCIELAB(img_xyz, white_xyz, params)

    assert tuple(scene_get(scene, "size")) == (506, 759)
    assert tuple(sensor_get(sensor, "size")) == (174, 212)
    assert tuple(srgb.shape) == (174, 212, 3)
    assert white_xyz.shape == (3,)
    assert np.all(white_xyz > 0.0)
    assert np.isclose(float(params["sampPerDeg"]), 50.0)
    assert np.isclose(float(params["filterSize"]), 49.0)
    assert img_opp.shape == img_xyz.shape
    assert tuple(np.asarray(support).shape) == (49,)
    assert len(filters) == 3
    assert tuple(np.asarray(img_filtered_xyz).shape) == (173, 211, 3)
    assert tuple(np.asarray(img_filtered_opp).shape) == (173, 211, 3)
    assert tuple(np.asarray(result).shape) == (173, 211, 3)
    assert np.all(np.asarray(white_pt, dtype=float) > 0.0)


def test_run_python_case_supports_metrics_scielab_tutorial_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_tutorial_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (506, 759)
    assert np.isclose(float(case.payload["scene_fov_deg"]), 8.0)
    assert tuple(case.payload["sensor_size"]) == (174, 212)
    assert tuple(case.payload["ip_result_size"]) == (174, 212, 3)
    assert case.payload["white_xyz"].shape == (3,)
    assert np.isclose(float(case.payload["samp_per_deg"]), 50.0)
    assert np.isclose(float(case.payload["filter_size"]), 50.0)
    assert np.isclose(float(case.payload["image_height_deg"]), 3.48)
    assert case.payload["original_render_mean_rgb_norm"].shape == (3,)
    assert case.payload["original_render_center_row_luma_norm"].shape == (129,)
    assert case.payload["img_opp_channel_means"].shape == (3,)
    assert case.payload["filter_support"].shape == (49,)
    assert case.payload["filter_peaks"].shape == (3,)
    assert case.payload["filter_center_rows_norm"].shape == (3, 65)
    assert tuple(case.payload["filtered_xyz_size"]) == (173, 211, 3)
    assert case.payload["filtered_xyz_delta_stats"].shape == (2,)
    assert case.payload["filtered_opp_channel_means"].shape == (3,)
    assert case.payload["filtered_render_mean_rgb_norm"].shape == (3,)
    assert case.payload["filtered_render_center_row_luma_norm"].shape == (129,)
    assert tuple(case.payload["result_size"]) == (173, 211, 3)
    assert case.payload["result_white_point"].shape == (3,)
    assert case.payload["result_lab_channel_means"].shape == (3,)
    assert case.payload["result_l_center_row_norm"].shape == (129,)


def test_xyz2vsnr_matches_scielab_variance_formula() -> None:
    rows, cols = 10, 12
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, cols), np.linspace(0.0, 1.0, rows), indexing="xy")
    roi_xyz = np.stack(
        [
            20.0 + 5.0 * xx,
            25.0 + 3.0 * yy + 0.5 * xx,
            18.0 + 2.0 * xx * yy,
        ],
        axis=2,
    )
    white_xyz = np.array([95.0, 100.0, 108.0], dtype=float)
    params = sc_params()

    s_lab, _ = scComputeSCIELAB(roi_xyz, white_xyz, params)
    mid = np.round(0.8 * np.array(s_lab.shape[:2], dtype=float)).astype(int)
    row0 = max((s_lab.shape[0] - int(mid[0])) // 2, 0)
    col0 = max((s_lab.shape[1] - int(mid[1])) // 2, 0)
    cropped = s_lab[row0 : row0 + int(mid[0]), col0 : col0 + int(mid[1]), :]
    expected = 1.0 / np.sqrt(
        float(np.var(cropped[:, :, 0], dtype=float))
        + float(np.var(cropped[:, :, 1], dtype=float))
        + float(np.var(cropped[:, :, 2], dtype=float))
    )

    assert np.isclose(float(xyz2vSNR(roi_xyz, white_xyz, params)), expected, atol=1e-10, rtol=1e-10)


def test_metrics_scielab_harmonic_experiments_script_workflow(asset_store) -> None:
    size = 512
    max_frequency = float(size) / 64.0

    scene = scene_create("sweep frequency", size, max_frequency, asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "diffuser method", "blur")
    oi = oi_set(oi, "diffuser blur", 1.5e-6)
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set_size_to_fov(sensor, 0.95 * float(scene_get(scene, "fov")), oi)
    sensor = sensor_compute(sensor, oi)

    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "gray world")
    ip = ip_compute(ip, sensor)

    img = np.asarray(ip_get(ip, "result"), dtype=float)
    img_xyz = srgb2xyz(img)
    img_opp = image_linear_transform(img_xyz, colorTransformMatrix("xyz2opp", 10))
    img_opp_xw, _, _, _ = rgb_to_xw_format(img_opp)
    opponent_means = np.mean(np.asarray(img_opp_xw, dtype=float), axis=0)

    params = sc_params()
    params["sampPerDeg"] = 100.0
    white_xyz = srgb2xyz(np.ones((1, 1, 3), dtype=float)).reshape(3)
    scale_factors = np.array([[1.0, 0.5, 1.0], [1.0, 1.0, 0.5], [0.75, 1.0, 1.0]], dtype=float)

    padded_img = np.pad(img, ((16, 16), (16, 16), (0, 0)), mode="constant")
    error_stats = np.zeros((scale_factors.shape[0], 4), dtype=float)
    for index, scale in enumerate(scale_factors):
        adjusted_opp = np.zeros_like(img_opp, dtype=float)
        for channel_index in range(3):
            adjusted_opp[:, :, channel_index] = (
                (img_opp[:, :, channel_index] - float(opponent_means[channel_index])) * float(scale[channel_index])
                + float(opponent_means[channel_index])
            )

        adjusted_rgb = xyz_to_srgb(image_linear_transform(adjusted_opp, colorTransformMatrix("opp2xyz", 10)))
        error_image, _, _, _ = scielab(
            padded_img,
            np.pad(adjusted_rgb, ((16, 16), (16, 16), (0, 0)), mode="constant"),
            white_xyz,
            params,
        )
        error_stats[index, :] = np.array(
            [
                float(np.mean(error_image)),
                float(np.std(error_image)),
                float(np.percentile(error_image, 5)),
                float(np.percentile(error_image, 95)),
            ],
            dtype=float,
        )

    assert tuple(scene_get(scene, "size")) == (512, 512)
    assert np.isclose(float(scene_get(scene, "fov")), 8.0)
    assert tuple(oi_get(oi, "size")) == (640, 640)
    assert np.isclose(float(oi_get(oi, "diffuser blur")), 1.5e-6)
    assert tuple(sensor_get(sensor, "size")) == (150, 184)
    assert tuple(img.shape) == (150, 184, 3)
    assert white_xyz.shape == (3,)
    assert img_opp.shape == img_xyz.shape
    assert opponent_means.shape == (3,)
    assert error_stats.shape == (3, 4)
    assert np.all(error_stats[:, 0] > 0.0)


def test_run_python_case_supports_metrics_scielab_harmonic_experiments_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_scielab_harmonic_experiments_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (512, 512)
    assert np.isclose(float(case.payload["scene_fov_deg"]), 8.0)
    assert np.isclose(float(case.payload["sweep_max_frequency_cpd"]), 8.0)
    assert tuple(case.payload["oi_size"]) == (640, 640)
    assert np.isclose(float(case.payload["oi_diffuser_blur_m"]), 1.5e-6)
    assert tuple(case.payload["sensor_size"]) == (150, 184)
    assert tuple(case.payload["ip_result_size"]) == (150, 184, 3)
    assert case.payload["white_xyz"].shape == (3,)
    assert np.isclose(float(case.payload["samp_per_deg"]), 100.0)
    assert case.payload["scale_factors"].shape == (3, 3)
    assert case.payload["original_render_mean_rgb_norm"].shape == (3,)
    assert case.payload["original_opp_channel_means"].shape == (3,)
    assert case.payload["altered_render_mean_rgb_norm"].shape == (3, 3)
    assert case.payload["altered_opp_channel_means"].shape == (3, 3)
    assert case.payload["error_stats"].shape == (3, 4)
    assert case.payload["error_center_row_norm"].shape == (3, 129)
    assert np.all(np.asarray(case.payload["error_stats"], dtype=float)[:, 0] > 0.0)


def test_metrics_edge2mtf_script_workflow(asset_store) -> None:
    scene = scene_create("slanted bar", 512, 7.0 / 3.0, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene = scene_set(scene, "distance", 1.0)
    scene = scene_set(scene, "fov", 5.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 2.8)
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "autoExposure", 1)
    sensor = sensor_compute(sensor, oi)

    ip = ip_compute(ip_create(asset_store=asset_store), sensor)
    rect = iso_find_slanted_bar(ip)
    result = np.asarray(ip_get(ip, "result"), dtype=float)
    col_min, row_min, width, height = np.asarray(rect, dtype=int).reshape(-1)
    bar_image = result[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
    mtf_data = edge_to_mtf(bar_image, channel=2, fixed_row=20)

    corr, lags = ie_cxcorr(np.arange(1.0, 11.0, dtype=float), np.roll(np.arange(1.0, 11.0, dtype=float), -3))

    assert tuple(scene_get(scene, "size")) == (513, 513)
    assert tuple(oi_get(oi, "size")) == (641, 641)
    assert tuple(sensor_get(sensor, "size")) == (72, 88)
    assert tuple(ip_get(ip, "size")) == (72, 88, 3)
    assert rect.shape == (4,)
    assert width > 5 and height > 5
    assert bar_image.shape == (height + 1, width + 1, 3)
    assert mtf_data["dimg"].shape == (height + 1, width)
    assert mtf_data["aligned"].shape == mtf_data["dimg"].shape
    assert mtf_data["lags"].shape == (height + 1,)
    assert mtf_data["lsf"].shape == (width,)
    assert mtf_data["mtf"].shape == (int(np.floor(width / 2.0 + 0.5)),)
    assert np.isclose(float(np.sum(mtf_data["lsf"])), 1.0, atol=1e-10, rtol=1e-10)
    assert np.isclose(float(mtf_data["mtf"][0]), 1.0, atol=1e-10, rtol=1e-10)
    assert np.all(np.asarray(mtf_data["mtf"], dtype=float) >= 0.0)
    assert int(np.argmax(corr)) == 3
    assert np.array_equal(lags, np.arange(10, dtype=int))


def test_run_python_case_supports_metrics_edge2mtf_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_edge2mtf_small", asset_store=asset_store)

    assert tuple(case.payload["sensor_size"]) == (72, 88)
    assert tuple(case.payload["ip_size"]) == (72, 88, 3)
    assert float(case.payload["roi_aspect_ratio"]) > 1.0
    assert 0.0 < float(case.payload["roi_fill_fraction"]) < 1.0
    assert case.payload["bar_green_mean_profile_norm"].shape == (65,)
    assert case.payload["lag_stats"].shape == (4,)
    assert case.payload["lsf_norm"].shape == (65,)
    assert case.payload["mtf_norm"].shape == (65,)
    assert np.isclose(float(case.payload["mtf_norm"][0]), 1.0, atol=1e-10, rtol=1e-10)


def test_metrics_mtf_slanted_bar_script_workflow(asset_store) -> None:
    scene = scene_create("slanted bar", 512, 7.0 / 3.0, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene = scene_set(scene, "distance", 1.0)
    scene = scene_set(scene, "fov", 5.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 2.0)
    oi = oi_compute(oi, scene)

    sensor_color = sensor_create(asset_store=asset_store)
    sensor_color = sensor_set(sensor_color, "autoExposure", 1)
    sensor_color = sensor_compute(sensor_color, oi)
    ip_color = ip_compute(ip_create(asset_store=asset_store), sensor_color)

    rect = iso_find_slanted_bar(ip_color)
    result_color = np.asarray(ip_get(ip_color, "result"), dtype=float)
    col_min, row_min, width, height = np.asarray(rect, dtype=int).reshape(-1)
    color_bar = result_color[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
    color_direct = iso12233(color_bar, float(sensor_get(sensor_color, "pixel width", "mm")), plot_options="none")
    color_ie = ie_iso12233(ip_color, sensor_color, plot_options="none", master_rect=rect)

    sensor_mono = sensor_create("monochrome", asset_store=asset_store)
    sensor_mono = sensor_set(sensor_mono, "autoExposure", 1)
    sensor_mono = sensor_compute(sensor_mono, oi)
    ip_mono = ip_compute(ip_create(asset_store=asset_store), sensor_mono)
    result_mono = np.asarray(ip_get(ip_mono, "result"), dtype=float)
    mono_bar = result_mono[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
    mono_direct = iso12233(mono_bar, float(sensor_get(sensor_mono, "pixel width", "mm")), plot_options="none")

    assert tuple(scene_get(scene, "size")) == (513, 513)
    assert tuple(oi_get(oi, "size")) == (641, 641)
    assert tuple(sensor_get(sensor_color, "size")) == (72, 88)
    assert tuple(sensor_get(sensor_mono, "size")) == (72, 88)
    assert np.array_equal(np.asarray(rect, dtype=int), np.array([32, 18, 24, 36], dtype=int))
    assert color_bar.shape == (37, 25, 3)
    assert mono_bar.shape == (37, 25, 3)
    assert color_direct.esf is not None and color_direct.esf.ndim == 2 and color_direct.esf.shape[1] == 4
    assert color_ie.esf is not None and color_ie.esf.ndim == 2 and color_ie.esf.shape[1] == 4
    assert mono_direct.esf is not None and mono_direct.esf.ndim == 2 and mono_direct.esf.shape[1] == 4
    assert color_direct.freq.shape[0] == color_direct.mtf.shape[0]
    assert color_ie.freq.shape[0] == color_ie.mtf.shape[0]
    assert mono_direct.freq.shape[0] == mono_direct.mtf.shape[0]
    assert np.isclose(float(color_direct.mtf[0, -1]), 1.0, atol=1e-10, rtol=1e-10)
    assert np.isclose(float(color_ie.mtf[0, -1]), 1.0, atol=1e-10, rtol=1e-10)
    assert np.isclose(float(mono_direct.mtf[0, -1]), 1.0, atol=1e-10, rtol=1e-10)
    assert np.isclose(float(color_direct.nyquistf), float(color_ie.nyquistf), rtol=1e-10, atol=1e-12)
    assert np.isfinite(float(color_ie.mtf50))
    assert np.isfinite(float(mono_direct.mtf50))


def test_metrics_iso12233_v1_wrappers_match_current_iso_surface(asset_store) -> None:
    scene = scene_create("slanted bar", 512, 7.0 / 3.0, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene = scene_set(scene, "distance", 1.0)
    scene = scene_set(scene, "fov", 5.0)

    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "autoExposure", 1)
    sensor = sensor_compute(sensor, oi)
    ip = ip_compute(ip_create(asset_store=asset_store), sensor)

    rect = iso_find_slanted_bar(ip)
    col_min, row_min, width, height = np.asarray(rect, dtype=int).reshape(-1)
    bar = np.asarray(ip_get(ip, "result"), dtype=float)[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
    dx = float(sensor_get(sensor, "pixel width", "mm"))

    current_direct = iso12233(bar, dx, plot_options="none")
    legacy_direct = ISO12233v1(bar, dx, plot_options="none")
    current_ie = ie_iso12233(ip, sensor, plot_options="none", master_rect=rect)
    legacy_ie = ieISO12233v1(ip, sensor, plot_options="none", master_rect=rect)

    assert np.allclose(np.asarray(legacy_direct.freq, dtype=float), np.asarray(current_direct.freq, dtype=float))
    assert np.allclose(np.asarray(legacy_direct.mtf, dtype=float), np.asarray(current_direct.mtf, dtype=float))
    assert np.allclose(np.asarray(legacy_direct.lsf, dtype=float), np.asarray(current_direct.lsf, dtype=float))
    assert np.isclose(float(legacy_direct.mtf50), float(current_direct.mtf50))
    assert np.isclose(float(legacy_direct.aliasingPercentage), float(current_direct.aliasingPercentage))

    assert np.array_equal(np.asarray(legacy_ie.rect, dtype=int), np.asarray(current_ie.rect, dtype=int))
    assert np.allclose(np.asarray(legacy_ie.freq, dtype=float), np.asarray(current_ie.freq, dtype=float))
    assert np.allclose(np.asarray(legacy_ie.mtf, dtype=float), np.asarray(current_ie.mtf, dtype=float))
    assert np.allclose(np.asarray(legacy_ie.lsf, dtype=float), np.asarray(current_ie.lsf, dtype=float))
    assert np.isclose(float(legacy_ie.mtf50), float(current_ie.mtf50))
    assert np.isclose(float(legacy_ie.aliasingPercentage), float(current_ie.aliasingPercentage))


def test_run_python_case_supports_metrics_mtf_slanted_bar_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_mtf_slanted_bar_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (513, 513)
    assert tuple(case.payload["oi_size"]) == (641, 641)
    assert tuple(case.payload["color_sensor_size"]) == (72, 88)
    assert tuple(case.payload["mono_sensor_size"]) == (72, 88)
    assert np.array_equal(case.payload["master_rect"], np.array([32, 18, 24, 36], dtype=int))
    assert np.isclose(float(case.payload["color_dx_mm"]), float(case.payload["mono_dx_mm"]), rtol=1e-10, atol=1e-12)
    assert case.payload["color_direct_esf_norm"].shape == (129,)
    assert case.payload["color_direct_lsf_norm"].shape == (129,)
    assert case.payload["color_direct_mtf_norm"].shape == (129,)
    assert case.payload["ie_color_esf_norm"].shape == (129,)
    assert case.payload["ie_color_lsf_norm"].shape == (129,)
    assert case.payload["ie_color_mtf_norm"].shape == (129,)
    assert case.payload["mono_direct_esf_norm"].shape == (129,)
    assert case.payload["mono_direct_lsf_norm"].shape == (129,)
    assert case.payload["mono_direct_mtf_norm"].shape == (129,)
    assert float(case.payload["color_direct_nyquistf"]) > 0.0
    assert float(case.payload["ie_color_nyquistf"]) > 0.0
    assert float(case.payload["mono_direct_nyquistf"]) > 0.0


def test_metrics_mtf_pixel_size_script_workflow(asset_store) -> None:
    scene = scene_create("slanted bar", 512, 7.0 / 3.0, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene = scene_set(scene, "distance", 1.0)
    scene = scene_set(scene, "fov", 5.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 4.0)
    oi = oi_compute(oi, scene)

    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "autoExposure", 1)
    ip = ip_create(asset_store=asset_store)
    master_rect = np.array([199, 168, 101, 167], dtype=int)

    sensor_sizes = []
    rects = []
    mtf50 = []
    for pixel_size_um in [2.0, 3.0, 5.0, 9.0]:
        sensor1 = sensor_set(
            sensor,
            "pixel size constant fill factor",
            np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6,
        )
        sensor1 = sensor_set(sensor1, "rows", round(512.0 / pixel_size_um))
        sensor1 = sensor_set(sensor1, "cols", round(512.0 / pixel_size_um))
        sensor1 = sensor_compute(sensor1, oi)
        ip1 = ip_compute(ip, sensor1)

        rect = np.floor((master_rect.astype(float) / pixel_size_um) + 0.5).astype(int)
        col_min, row_min, width, height = rect
        bar = np.asarray(ip_get(ip1, "result"), dtype=float)[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
        mtf_data = iso12233(bar, float(sensor_get(sensor1, "pixel width", "mm")), plot_options="none")

        sensor_sizes.append(tuple(sensor_get(sensor1, "size")))
        rects.append(rect)
        mtf50.append(float(mtf_data.mtf50))
        assert bar.ndim == 3 and bar.shape[2] == 3
        assert np.asarray(mtf_data.mtf, dtype=float).shape[1] == 4
        assert np.isclose(float(np.asarray(mtf_data.mtf, dtype=float)[0, -1]), 1.0, atol=1e-10, rtol=1e-10)

    assert tuple(scene_get(scene, "size")) == (513, 513)
    assert tuple(oi_get(oi, "size")) == (641, 641)
    assert sensor_sizes == [(256, 256), (171, 171), (102, 102), (57, 57)]
    assert np.array_equal(rects[0], np.array([100, 84, 51, 84], dtype=int))
    assert np.array_equal(rects[1], np.array([66, 56, 34, 56], dtype=int))
    assert np.array_equal(rects[2], np.array([40, 34, 20, 33], dtype=int))
    assert np.array_equal(rects[3], np.array([22, 19, 11, 19], dtype=int))
    assert mtf50[0] > mtf50[-1]


def test_run_python_case_supports_metrics_mtf_pixel_size_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_mtf_pixel_size_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (513, 513)
    assert tuple(case.payload["oi_size"]) == (641, 641)
    assert np.array_equal(case.payload["pixel_sizes_um"], np.array([2.0, 3.0, 5.0, 9.0], dtype=float))
    assert np.array_equal(case.payload["sensor_sizes"], np.array([[256, 256], [171, 171], [102, 102], [57, 57]], dtype=int))
    assert np.array_equal(case.payload["rects"][0], np.array([100, 84, 51, 84], dtype=int))
    assert np.array_equal(case.payload["bar_sizes"], np.array([[85, 52, 3], [57, 35, 3], [34, 21, 3], [20, 12, 3]], dtype=int))
    assert case.payload["mtf_profiles_norm"].shape == (4, 129)
    assert np.all(np.asarray(case.payload["nyquistf"], dtype=float) > 0.0)
    assert np.all(np.asarray(case.payload["mtf50"], dtype=float) > 0.0)


def test_metrics_snr_pixel_size_luxsec_script(asset_store) -> None:
    integration_time = 0.010
    pixel_sizes_um = np.array([2.0, 4.0, 6.0, 9.0, 10.0], dtype=float)
    read_noise_mv = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
    voltage_swing_v = np.array([0.7, 1.2, 1.5, 2.0, 3.0], dtype=float)
    dark_voltage_mv_per_sec = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "integration time", integration_time)

    luxsec_saturation = []
    volts_per_lux_sec = []
    terminal_snr = []
    for pixel_size_um, read_noise_mv_value, voltage_swing_value, dark_voltage_mv_value in zip(
        pixel_sizes_um,
        read_noise_mv,
        voltage_swing_v,
        dark_voltage_mv_per_sec,
        strict=False,
    ):
        sensor1 = sensor_set(
            sensor,
            "pixel size constant fill factor",
            np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6,
        )
        sensor1 = sensor_set(sensor1, "readNoiseSTDvolts", float(read_noise_mv_value) * 1.0e-3)
        sensor1 = sensor_set(sensor1, "voltageSwing", float(voltage_swing_value))
        sensor1 = sensor_set(sensor1, "darkVoltage", float(dark_voltage_mv_value) * 1.0e-3)

        snr, luxsec, snr_shot, snr_read, anti_luxsec = pixel_snr_luxsec(sensor1, asset_store=asset_store)
        vp_luxsec, sat_luxsec, mean_volts, vp_anti_luxsec, anti_sat_luxsec = pixel_v_per_lux_sec(
            sensor1,
            asset_store=asset_store,
        )

        assert snr.shape == (50,)
        assert luxsec.shape == (50, 1)
        assert snr_shot.shape == (50,)
        assert anti_luxsec.shape == (50, 1)
        assert np.all(np.diff(snr) > 0.0)
        assert np.all(np.diff(luxsec[:, 0]) > 0.0)
        assert float(vp_luxsec[0]) > 0.0
        assert float(sat_luxsec) > 0.0
        assert 0.0 < float(mean_volts[0]) < float(voltage_swing_value)
        assert np.isinf(float(vp_anti_luxsec[0])) or float(vp_anti_luxsec[0]) >= 0.0
        assert float(anti_sat_luxsec) >= 0.0
        if np.isscalar(snr_read):
            assert np.isfinite(float(snr_read)) or np.isinf(float(snr_read))
        else:
            assert np.asarray(snr_read, dtype=float).shape == (50,)

        luxsec_saturation.append(float(sat_luxsec))
        volts_per_lux_sec.append(float(vp_luxsec[0]))
        terminal_snr.append(float(snr[-1]))

    assert luxsec_saturation[0] > luxsec_saturation[-1]
    assert volts_per_lux_sec[0] < volts_per_lux_sec[-1]
    assert terminal_snr[0] < terminal_snr[-1]


def test_run_python_case_supports_metrics_snr_pixel_size_luxsec_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_snr_pixel_size_luxsec_small", asset_store=asset_store)

    assert np.array_equal(case.payload["pixel_sizes_um"], np.array([2.0, 4.0, 6.0, 9.0, 10.0], dtype=float))
    assert np.array_equal(case.payload["read_noise_mv"], np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float))
    assert case.payload["snr_db"].shape == (5, 50)
    assert case.payload["luxsec_curves"].shape == (5, 50)
    assert case.payload["snr_shot_db"].shape == (5, 50)
    assert case.payload["snr_read_db"].shape == (5, 50)
    assert np.all(np.diff(case.payload["snr_db"], axis=1) > 0.0)
    assert np.all(np.diff(case.payload["luxsec_curves"], axis=1) > 0.0)
    assert np.all(np.asarray(case.payload["volts_per_lux_sec"], dtype=float) > 0.0)
    assert np.all(np.asarray(case.payload["luxsec_saturation"], dtype=float) > 0.0)


def test_metrics_mtf_slanted_bar_infrared_script(asset_store) -> None:
    wave = np.arange(400.0, 1068.0 + 0.1, 4.0, dtype=float)
    scene = scene_create("slanted bar", 512, 7.0 / 3.0, 5.0, wave, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene = scene_set(scene, "distance", 1.0)
    scene = scene_set(scene, "fov", 5.0)

    oi = oi_create("diffraction limited", asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 4.0)
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    filter_spectra, filter_names, _ = ie_read_color_filter(wave, "NikonD200IR.mat", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", wave)
    sensor = sensor_set(sensor, "filterSpectra", filter_spectra)
    sensor = sensor_set(sensor, "filterNames", filter_names)
    sensor = sensor_set(sensor, "ir filter", np.ones_like(wave))
    sensor = sensor_set(sensor, "pixel spectral qe", np.ones_like(wave))
    sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)
    sensor = sensor_compute(sensor, oi)

    ip = ip_create(asset_store=asset_store)
    ip = ip_set(ip, "scale display", 1)
    ip = ip_set(ip, "render Gamma", 0.6)
    ip = ip_set(ip, "conversion method sensor ", "MCC Optimized")
    ip = ip_set(ip, "correction method illuminant ", "Gray World")
    ip = ip_set(ip, "internal CS", "XYZ")
    ip = ip_compute(ip, sensor)

    fixed_rect = np.array([39, 25, 51, 65], dtype=int)
    col_min, row_min, width, height = fixed_rect
    fixed_bar = np.asarray(ip_get(ip, "result"), dtype=float)[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
    fixed_mtf = iso12233(fixed_bar, float(sensor_get(sensor, "pixel width", "mm")), plot_options="none")

    ir_filter, ir_filter_names, _ = ie_read_color_filter(wave, "IRBlocking", asset_store=asset_store)
    blocked_sensor = sensor_set(sensor, "ir filter", np.asarray(ir_filter, dtype=float).reshape(-1))
    blocked_sensor = sensor_compute(blocked_sensor, oi)
    blocked_ip = ip_compute(ip, blocked_sensor)
    blocked_mtf = ie_iso12233(blocked_ip, blocked_sensor, "none")

    assert tuple(scene_get(scene, "size")) == (513, 513)
    assert tuple(oi_get(oi, "size")) == (641, 641)
    assert list(filter_names) == ["r_custom_", "g_custom_", "b_custom_"]
    assert list(ir_filter_names) == ["ir"]
    assert tuple(sensor_get(sensor, "size")) == (100, 120)
    assert int(sensor_get(sensor, "nfilters")) == 3
    assert fixed_bar.shape == (66, 52, 3)
    assert fixed_mtf.mtf50 > blocked_mtf.mtf50
    assert abs(float(blocked_mtf.mtf50) - 77.0) <= 3.0
    assert np.array_equal(np.asarray(blocked_mtf.rect, dtype=int), np.array([42, 23, 36, 54], dtype=int))


def test_run_python_case_supports_metrics_mtf_slanted_bar_infrared_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_mtf_slanted_bar_infrared_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (168,)
    assert tuple(case.payload["scene_size"]) == (513, 513)
    assert tuple(case.payload["oi_size"]) == (641, 641)
    assert tuple(case.payload["sensor_size"]) == (100, 120)
    assert list(case.payload["filter_names"]) == ["r_custom_", "g_custom_", "b_custom_"]
    assert list(case.payload["ir_filter_names"]) == ["ir"]
    assert np.array_equal(case.payload["fixed_rect"], np.array([39, 25, 51, 65], dtype=int))
    assert np.array_equal(case.payload["fixed_bar_size"], np.array([66, 52, 3], dtype=int))
    assert case.payload["fixed_esf_norm"].shape == (129,)
    assert case.payload["fixed_lsf_norm"].shape == (129,)
    assert case.payload["fixed_mtf_norm"].shape == (129,)
    assert case.payload["blocked_rect"].shape == (4,)
    assert case.payload["blocked_lsf_um"].shape == (129,)
    assert case.payload["blocked_lsf_norm"].shape == (129,)
    assert case.payload["blocked_mtf_norm"].shape == (129,)
    assert case.payload["fixed_mtf50"] > case.payload["blocked_mtf50"]


def test_metrics_acutance_script_workflow(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    camera = camera_set(camera, "sensor auto exposure", True)
    camera = camera_set(camera, "optics fnumber", 4.0)

    cmtf = camera_mtf(camera, asset_store=asset_store)
    oi = camera_get(camera, "oi")
    deg_per_mm = float(camera_get(camera, "sensor h deg per distance", "mm", None, oi))
    cpd = np.asarray(cmtf.freq, dtype=float) / deg_per_mm
    luminance_mtf = np.asarray(cmtf.mtf, dtype=float)[:, -1]
    cpiq = cpiq_csf(cpd)
    acutance = iso_acutance(cpd, luminance_mtf)

    assert np.asarray(camera_get(camera, "sensor size"), dtype=int).shape == (2,)
    assert np.asarray(ip_get(cmtf.vci, "size"), dtype=int).shape == (3,)
    assert np.asarray(cmtf.rect, dtype=int).shape == (4,)
    assert cmtf.freq.ndim == 1
    assert np.asarray(cmtf.mtf, dtype=float).ndim == 2
    assert np.asarray(cmtf.mtf, dtype=float).shape[0] == cmtf.freq.shape[0]
    assert np.asarray(cmtf.mtf, dtype=float).shape[1] == 4
    assert np.isclose(float(luminance_mtf[0]), 1.0, atol=1e-10, rtol=1e-10)
    assert cpiq.shape == cpd.shape
    assert np.isclose(float(np.max(cpiq)), 1.0, atol=1e-10, rtol=1e-10)
    assert deg_per_mm > 0.0
    assert np.isfinite(acutance)
    assert acutance > 0.0
    assert camera_acutance(camera, asset_store=asset_store) == pytest.approx(acutance, rel=1e-10, abs=1e-12)


def test_run_python_case_supports_metrics_acutance_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_acutance_small", asset_store=asset_store)

    assert case.payload["sensor_size"].shape == (2,)
    assert case.payload["ip_size"].shape == (3,)
    assert case.payload["rect"].shape == (4,)
    assert case.payload["cpd_stats"].shape == (4,)
    assert case.payload["cpiq_norm"].shape == (129,)
    assert case.payload["lum_mtf_norm"].shape == (129,)
    assert float(case.payload["deg_per_mm"]) > 0.0
    assert np.isfinite(float(case.payload["acutance"]))
    assert float(case.payload["acutance"]) > 0.0
    assert float(case.payload["camera_acutance"]) == pytest.approx(float(case.payload["acutance"]), rel=1e-10, abs=1e-12)


def test_metrics_color_accuracy_script_workflow(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    camera = camera_set(camera, "sensor auto exposure", True)
    color_accuracy, camera = camera_color_accuracy(camera, asset_store=asset_store)

    ip = camera_get(camera, "ip")
    embedded_rgb, compare_patch_srgb, patch_size = macbeth_compare_ideal(ip, asset_store=asset_store)
    ip_size = np.asarray(ip_get(ip, "size"), dtype=int)
    corner_points = np.asarray(ip_get(ip, "chart corner points"), dtype=float)

    assert np.asarray(camera_get(camera, "sensor size"), dtype=int).shape == (2,)
    assert ip_size.shape == (3,)
    assert np.array_equal(
        corner_points,
        np.array(
            [
                [1.0, float(ip_size[0])],
                [float(ip_size[1]), float(ip_size[0])],
                [float(ip_size[1]), 1.0],
                [1.0, 1.0],
            ],
            dtype=float,
        ),
    )
    assert np.asarray(color_accuracy["macbethLAB"], dtype=float).shape == (24, 3)
    assert np.asarray(color_accuracy["macbethXYZ"], dtype=float).shape == (24, 3)
    assert np.asarray(color_accuracy["deltaE"], dtype=float).shape == (24,)
    assert np.asarray(color_accuracy["whiteXYZ"], dtype=float).shape == (3,)
    assert np.asarray(color_accuracy["idealWhiteXYZ"], dtype=float).shape == (3,)
    assert np.all(np.asarray(color_accuracy["deltaE"], dtype=float) >= 0.0)
    assert np.isfinite(float(np.mean(np.asarray(color_accuracy["deltaE"], dtype=float), dtype=float)))
    assert compare_patch_srgb.shape == (4, 6, 3)
    assert embedded_rgb.ndim == 3 and embedded_rgb.shape[2] == 3
    assert np.asarray(patch_size, dtype=int).shape == (2,)
    assert np.all(np.asarray(compare_patch_srgb, dtype=float) >= 0.0)
    assert np.all(np.asarray(compare_patch_srgb, dtype=float) <= 1.0)
    assert np.all(np.asarray(embedded_rgb, dtype=float) >= 0.0)
    assert np.all(np.asarray(embedded_rgb, dtype=float) <= 1.0)


def test_run_python_case_supports_metrics_color_accuracy_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_color_accuracy_small", asset_store=asset_store)

    assert case.payload["sensor_size"].shape == (2,)
    assert case.payload["ip_size"].shape == (3,)
    assert case.payload["corner_points"].shape == (4, 2)
    assert case.payload["white_xyz_norm"].shape == (3,)
    assert case.payload["ideal_white_xyz_norm"].shape == (3,)
    assert case.payload["delta_e"].shape == (24,)
    assert case.payload["delta_e_stats"].shape == (3,)
    assert case.payload["macbeth_lab"].shape == (24, 3)
    assert case.payload["compare_patch_srgb"].shape == (4, 6, 3)
    assert case.payload["ideal_patch_srgb"].shape == (4, 6, 3)
    assert case.payload["embedded_channel_means"].shape == (3,)
    assert case.payload["patch_size"].shape == (2,)
    assert np.all(np.asarray(case.payload["delta_e"], dtype=float) >= 0.0)
    assert np.all(np.asarray(case.payload["compare_patch_srgb"], dtype=float) >= 0.0)
    assert np.all(np.asarray(case.payload["compare_patch_srgb"], dtype=float) <= 1.0)


def test_metrics_macbeth_delta_e_script_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 75.0, asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.64)
    scene = scene_set(scene, "distance", 10.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 4.0)
    oi = oi_set(oi, "optics focal length", 20.0e-3)
    oi = oi_set(oi, "optics off axis method", "skip")
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)
    sensor = sensor_compute(sensor, oi)

    corner_points = np.array([[1.0, 244.0], [328.0, 246.0], [329.0, 28.0], [2.0, 27.0]], dtype=float)
    sensor = sensor_set(sensor, "chart corner points", corner_points)
    ccm_matrix, sensor_locs = sensor_ccm(sensor, "macbeth", asset_store=asset_store)

    ip = ip_create(asset_store=asset_store)
    ip = ip_set(ip, "scale display", 1)
    ip = ip_set(ip, "conversion matrix sensor", ccm_matrix)
    ip = ip_set(ip, "correction matrix illuminant", np.array([], dtype=float))
    ip = ip_set(ip, "internal cs 2 display space", np.array([], dtype=float))
    ip = ip_set(ip, "conversion method sensor", "Current matrix")
    ip = ip_set(ip, "internalCS", "Sensor")
    ip = ip_compute(ip, sensor, asset_store=asset_store)

    point_loc = np.array([[4.0, 246.0], [328.0, 243.0], [327.0, 26.0], [3.0, 27.0]], dtype=float)
    macbeth_lab, macbeth_xyz, delta_e, _ = macbeth_color_error(ip, "D65", point_loc, asset_store=asset_store)

    result = np.asarray(ip_get(ip, "result"), dtype=float)
    assert tuple(scene_get(scene, "size")) == (64, 96)
    assert tuple(oi_get(oi, "size")) == (80, 120)
    assert tuple(sensor_get(sensor, "size")) == (270, 330)
    assert tuple(ip_get(ip, "size")) == (270, 330, 3)
    assert ccm_matrix.shape == (3, 3)
    assert np.array_equal(sensor_locs, corner_points)
    assert macbeth_lab.shape == (24, 3)
    assert macbeth_xyz.shape == (24, 3)
    assert delta_e.shape == (24,)
    assert np.all(delta_e >= 0.0)
    assert np.all(np.isfinite(delta_e))
    assert np.all(np.isfinite(macbeth_lab))
    assert result.shape == (270, 330, 3)
    assert np.all(result >= 0.0)


def test_run_python_case_supports_metrics_macbeth_delta_e_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_macbeth_delta_e_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 96)
    assert tuple(case.payload["oi_size"]) == (80, 120)
    assert tuple(case.payload["sensor_size"]) == (270, 330)
    assert tuple(case.payload["ip_size"]) == (270, 330, 3)
    assert case.payload["ccm_matrix"].shape == (3, 3)
    assert case.payload["sensor_locs"].shape == (4, 2)
    assert case.payload["point_loc"].shape == (4, 2)
    assert case.payload["white_xyz_norm"].shape == (3,)
    assert case.payload["delta_e"].shape == (24,)
    assert case.payload["delta_e_stats"].shape == (3,)
    assert case.payload["macbeth_lab"].shape == (24, 3)
    assert case.payload["result_channel_means_norm"].shape == (3,)
    assert case.payload["result_channel_p95_norm"].shape == (3,)
    assert np.all(np.asarray(case.payload["delta_e"], dtype=float) >= 0.0)


def test_run_python_case_supports_sensor_imx363_crop_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_imx363_crop_small", asset_store=asset_store)

    assert case.payload["name"] == "IMX363"
    assert np.array_equal(case.payload["size"], np.array([6, 8], dtype=int))
    assert np.array_equal(case.payload["metadata_crop"], np.array([3, 3, 7, 5], dtype=int))
    assert case.payload["digital_values"].shape == (6, 8)


def test_run_python_case_supports_sensor_plot_line_volts_space_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_plot_line_volts_space_small", asset_store=asset_store)

    assert case.payload["pixPos"].ndim == 1
    assert case.payload["pixData"].ndim == 1
    assert case.payload["pixPos"].size == case.payload["pixData"].size
    assert case.payload["pixPos"].size > 0


def test_signal_current_matches_sensor_compute_mean_electrons(asset_store) -> None:
    scene = scene_create("uniform ee", 64, asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)
    scene = scene_set(scene, "distance", 1.2)
    scene = scene_adjust_luminance(scene, 1.0, asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "exp time", 1.0)

    current = signal_current(oi, sensor)
    computed = sensor_compute(sensor, oi, seed=0)
    electrons = np.asarray(sensor_get(computed, "electrons"), dtype=float)

    start = (current.shape[0] - 40) // 2
    stop = start + 40
    current_center = np.asarray(current[start:stop, start:stop], dtype=float)
    electrons_center = electrons[start:stop, start:stop]
    coulombs_to_electrons = float(sensor_get(sensor, "integration time")) / 1.602176634e-19

    assert np.mean(current_center * coulombs_to_electrons) == pytest.approx(np.mean(electrons_center), rel=1e-5)


def _make_current_density_test_inputs(asset_store):
    scene = scene_create("uniform ee", 32, asset_store=asset_store)
    scene = scene_set(scene, "fov", 6.0)
    scene = scene_set(scene, "distance", 1.0)
    scene = scene_adjust_luminance(scene, 20.0, asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create("bayer", asset_store=asset_store)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "exp time", 0.05)
    return oi, sensor


def test_signal_current_density_matches_internal_current_density(asset_store) -> None:
    oi, sensor = _make_current_density_test_inputs(asset_store)

    current_density = SignalCurrentDensity(oi, sensor)
    internal_density = sensor_module._signal_current_density(oi, sensor.clone())

    assert current_density.ndim == 3
    np.testing.assert_allclose(current_density, internal_density, rtol=1e-10, atol=1e-14)


def test_spatial_integration_matches_signal_current_default_grid_spacing(asset_store) -> None:
    oi, sensor = _make_current_density_test_inputs(asset_store)

    current_density = SignalCurrentDensity(oi, sensor)
    integrated = spatialIntegration(current_density, oi, sensor)
    expected = signal_current(oi, sensor)

    np.testing.assert_allclose(integrated, expected, rtol=1e-10, atol=1e-14)


def test_spatial_integration_rejects_nondefault_grid_spacing(asset_store) -> None:
    oi, sensor = _make_current_density_test_inputs(asset_store)
    current_density = SignalCurrentDensity(oi, sensor)

    with pytest.raises(UnsupportedOptionError, match="gridSpacing != 1"):
        spatialIntegration(current_density, oi, sensor, grid_spacing=0.5)


def test_run_python_case_supports_sensor_signal_current_uniform_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_signal_current_uniform_small", asset_store=asset_store)

    assert case.payload["current_center"].shape == (40, 40)
    assert float(case.payload["mean_current"]) > 0.0


def test_run_python_case_supports_metrics_sqri_tutorial_parity_case(asset_store) -> None:
    case = run_python_case_with_context("metrics_sqri_tutorial_small", asset_store=asset_store)

    assert case.payload["sf"].shape == (5000,)
    assert np.array_equal(case.payload["widths"], np.array([0.5, 1.0, 2.3, 6.5, 60.0], dtype=float))
    assert case.payload["max_sqri_by_width"].shape == (5,)
    assert case.payload["csf_peak_by_width"].shape == (5,)
    assert case.payload["luminance_levels"].shape == (6,)
    assert case.payload["max_sqri_by_luminance"].shape == (6,)
    assert np.array_equal(case.payload["note3_distances_m"], np.array([0.2, 0.4, 0.8], dtype=float))
    assert case.payload["note3_dmtf_selected"].shape == (3, 5)
    assert float(case.payload["crt_dpi"]) > 0.0
    assert float(case.payload["crt_sqri"]) > 0.0
    assert float(case.payload["crt_csf_peak"]) > 0.0


def test_xyy2xyz_matches_closed_form() -> None:
    xyy = np.array(
        [
            [0.3127, 0.3290, 1.0],
            [0.25, 0.40, 12.0],
        ],
        dtype=float,
    )

    xyz = xyy2xyz(xyy)
    expected = np.array(
        [
            [(0.3127 / 0.3290) * 1.0, 1.0, ((1.0 - 0.3127 - 0.3290) / 0.3290) * 1.0],
            [(0.25 / 0.40) * 12.0, 12.0, ((1.0 - 0.25 - 0.40) / 0.40) * 12.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(xyz, expected, rtol=1e-10, atol=1e-12)


def test_ie_lab_to_xyz_roundtrips_metrics_xyz_to_lab() -> None:
    xyz = np.array(
        [
            [0.95047, 1.0, 1.08883],
            [0.31, 0.22, 0.08],
        ],
        dtype=float,
    )
    white = np.array([0.95047, 1.0, 1.08883], dtype=float)

    lab = metrics_module.xyz_to_lab(xyz, white)
    reconstructed = ieLAB2XYZ(lab, white)

    np.testing.assert_allclose(reconstructed, xyz, rtol=1e-6, atol=1e-8)


def test_xyz2lms_and_lms2xyz_roundtrip_for_row_vectors() -> None:
    xyz = np.array(
        [
            [0.30, 0.20, 0.10],
            [0.95, 1.00, 1.09],
            [0.15, 0.40, 0.25],
        ],
        dtype=float,
    )

    lms = xyz2lms(xyz)
    reconstructed = lms2xyz(lms)

    np.testing.assert_allclose(reconstructed, xyz, rtol=2e-2, atol=1.5e-2)


def test_xyz2lms_zero_fills_missing_cone_channel() -> None:
    xyz = np.array(
        [
            [[0.3, 0.2, 0.1], [0.6, 0.5, 0.2]],
            [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        ],
        dtype=float,
    )

    lms = xyz2lms(xyz, cb_type=-2, extrap_val=0.25)

    assert lms.shape == xyz.shape
    np.testing.assert_allclose(lms[:, :, 1], 0.25, rtol=0.0, atol=0.0)


def test_lms2srgb_matches_xyz2srgb_of_reconstructed_xyz() -> None:
    xyz = np.array(
        [
            [[0.20, 0.30, 0.10], [0.25, 0.35, 0.15]],
            [[0.40, 0.45, 0.30], [0.55, 0.60, 0.40]],
        ],
        dtype=float,
    )

    lms = xyz2lms(xyz)
    srgb = lms2srgb(lms)
    expected = xyz2srgb(lms2xyz(lms))

    np.testing.assert_allclose(srgb, expected, rtol=1e-10, atol=1e-12)


def test_ie_read_color_filter_handles_ovt_hdf_orientation(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)

    filters, names, _ = ie_read_color_filter(
        wave,
        "data/sensor/colorfilters/OVT/ovt-large.mat",
        asset_store=asset_store,
    )

    assert filters.shape == (wave.size, 3)
    assert names == ["r", "g", "b"]
    assert np.all(filters >= 0.0)


def test_sensor_create_supports_ovt_vendor_presets(asset_store) -> None:
    large = sensor_create("ovt-large", asset_store=asset_store)
    small = sensor_create("ovt-small", asset_store=asset_store)

    assert isinstance(large, list)
    assert len(large) == 2
    assert [sensor_get(sensor, "name") for sensor in large] == ["ovt-LPDLCG", "ovt-LPDHCG"]
    assert all(tuple(np.asarray(sensor_get(sensor, "size"), dtype=int)) == (968, 1288) for sensor in large)
    assert all(sensor_get(sensor, "filter names") == ["r", "g", "b"] for sensor in large)
    assert float(sensor_get(large[0], "analog gain")) == pytest.approx(1.0)
    assert float(sensor_get(large[1], "analog gain")) == pytest.approx(49.0 / 200.0)

    assert tuple(np.asarray(sensor_get(small, "size"), dtype=int)) == (968, 1288)
    assert sensor_get(small, "name") == "ovt-SPDLCG"
    assert sensor_get(small, "filter names") == ["r", "g", "b"]
    assert float(sensor_get(small, "pixel fill factor")) == pytest.approx(1e-2)


def test_sensor_create_supports_monochrome_array_dispatch(asset_store) -> None:
    default_array = sensor_create("monochrome array", asset_store=asset_store)
    explicit = sensor_create("monochrome array", None, 5, asset_store=asset_store)
    placeholder = sensor_create("monochrome array", [], 5, asset_store=asset_store)
    placeholder_default = sensor_create("monochrome array", [], [], asset_store=asset_store)
    overloaded = sensor_create("monochrome array", 4, asset_store=asset_store)

    assert isinstance(default_array, list)
    assert isinstance(explicit, list)
    assert isinstance(placeholder, list)
    assert isinstance(placeholder_default, list)
    assert isinstance(overloaded, list)
    assert len(default_array) == 3
    assert len(explicit) == 5
    assert len(placeholder) == 5
    assert len(placeholder_default) == 3
    assert len(overloaded) == 4
    assert explicit[0] is not explicit[1]
    assert placeholder[0] is not placeholder[1]
    assert placeholder_default[0] is not placeholder_default[1]

    for current in explicit + placeholder + placeholder_default + overloaded + default_array:
        assert sensor_get(current, "name") == "monochrome"
        assert sensor_get(current, "filter names") == ["w"]
        np.testing.assert_array_equal(
            np.asarray(sensor_get(current, "pattern"), dtype=int),
            np.array([[1]], dtype=int),
        )


def test_sensor_create_supports_dual_pixel_dispatch(asset_store) -> None:
    scene = scene_create("uniform d65", asset_store=asset_store)
    oi = oi_compute(oi_create(), scene)
    microlens_size = np.array([6, 7], dtype=int)

    dual = sensor_create("dual pixel", None, oi, microlens_size, asset_store=asset_store)
    overloaded = sensor_create("dual pixel", oi, microlens_size, asset_store=asset_store)
    sample_spacing = np.asarray(oi_get(oi, "sample spacing", "m"), dtype=float).reshape(-1)

    assert tuple(np.asarray(sensor_get(dual, "size"), dtype=int)) == (6, 14)
    assert tuple(np.asarray(sensor_get(overloaded, "size"), dtype=int)) == (6, 14)
    assert float(sensor_get(dual, "pixel height")) == pytest.approx(2.0 * float(sample_spacing[0]))
    assert float(sensor_get(dual, "pixel width")) == pytest.approx(float(sample_spacing[0]))
    np.testing.assert_array_equal(
        np.asarray(sensor_get(dual, "pattern"), dtype=int),
        np.array([[2, 2, 1, 1], [3, 3, 2, 2]], dtype=int),
    )
    np.testing.assert_array_equal(
        np.asarray(sensor_get(overloaded, "pattern"), dtype=int),
        np.array([[2, 2, 1, 1], [3, 3, 2, 2]], dtype=int),
    )


def test_sensor_compute_array_supports_ovt_saturated_flow(asset_store) -> None:
    scene = scene_create("uniform ee", np.array([32, 48], dtype=int), asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float).copy()
    levels = np.array([1.0, 10.0, 100.0, 1000.0], dtype=float)
    band_width = photons.shape[1] // levels.size
    for index, level in enumerate(levels):
        start = index * band_width
        stop = photons.shape[1] if index == (levels.size - 1) else (index + 1) * band_width
        photons[:, start:stop, :] *= level
    scene.data["photons"] = photons

    oi = oi_compute(oi_create("wvf", asset_store=asset_store), scene, "crop", True)
    sensor_array = sensor_create_array(
        "array type",
        "ovt",
        "exp time",
        0.1,
        "size",
        np.array([32, 48], dtype=int),
        "noise flag",
        0,
        asset_store=asset_store,
    )

    combined, captures = sensor_compute_array(sensor_array, oi, "method", "saturated")
    saturated = np.asarray(combined.metadata["saturated"], dtype=bool)

    assert len(captures) == 3
    assert tuple(sensor_get(combined, "size")) == (32, 48)
    assert np.asarray(sensor_get(combined, "volts"), dtype=float).shape == (32, 48)
    assert saturated.shape == (32, 48, 3)
    assert np.any(np.sum(saturated, axis=(0, 1)) > 0)


def test_sensor_create_array_supports_imx490_split_pixel_type(asset_store) -> None:
    sensor_array = sensor_create_array(
        "array type",
        "imx490",
        "exp time",
        0.01,
        "pixel size same fill factor",
        1.4e-6,
        "size",
        np.array([64, 96], dtype=int),
        asset_store=asset_store,
    )

    assert isinstance(sensor_array, list)
    assert len(sensor_array) == 4
    assert [sensor_get(current, "name") for current in sensor_array] == [
        "imx490-large",
        "imx490-large",
        "imx490-small",
        "imx490-small",
    ]
    assert all(tuple(np.asarray(sensor_get(current, "size"), dtype=int)) == (64, 96) for current in sensor_array)
    assert all(np.isclose(float(sensor_get(current, "exp time")), 0.01) for current in sensor_array)
    assert all(np.isclose(float(sensor_get(current, "pixel width")), 1.4e-6) for current in sensor_array)
    assert np.isclose(
        float(sensor_get(sensor_array[1], "pixel conversion gain")),
        4.0 * float(sensor_get(sensor_array[0], "pixel conversion gain")),
    )
    assert np.isclose(
        float(sensor_get(sensor_array[3], "pixel conversion gain")),
        4.0 * float(sensor_get(sensor_array[2], "pixel conversion gain")),
    )


def test_sensor_array_wrappers_treat_empty_array_type_as_ovt_default(asset_store) -> None:
    split_default = sensor_module.sensor_create_split_pixel(asset_store=asset_store)
    split_placeholder = sensor_module.sensor_create_split_pixel("array type", [], asset_store=asset_store)
    split_placeholder_settings = sensor_module.sensor_create_split_pixel("array type", [], "exp time", 0.01, asset_store=asset_store)

    assert isinstance(split_default, list)
    assert isinstance(split_placeholder, list)
    assert isinstance(split_placeholder_settings, list)
    assert [sensor_get(current, "name") for current in split_placeholder] == [sensor_get(current, "name") for current in split_default]
    assert [tuple(np.asarray(sensor_get(current, "size"), dtype=int)) for current in split_placeholder] == [
        tuple(np.asarray(sensor_get(current, "size"), dtype=int)) for current in split_default
    ]
    assert all(np.isclose(float(sensor_get(current, "exp time")), 0.01) for current in split_placeholder_settings)

    array_default = sensor_create_array(asset_store=asset_store)
    array_placeholder = sensor_create_array("array type", [], asset_store=asset_store)
    array_placeholder_settings = sensor_create_array("array type", [], "exp time", 0.02, asset_store=asset_store)

    assert isinstance(array_default, list)
    assert isinstance(array_placeholder, list)
    assert isinstance(array_placeholder_settings, list)
    assert [sensor_get(current, "name") for current in array_placeholder] == [sensor_get(current, "name") for current in array_default]
    assert [tuple(np.asarray(sensor_get(current, "size"), dtype=int)) for current in array_placeholder] == [
        tuple(np.asarray(sensor_get(current, "size"), dtype=int)) for current in array_default
    ]
    assert all(np.isclose(float(sensor_get(current, "exp time")), 0.02) for current in array_placeholder_settings)


def test_sensor_create_imx490_replays_upstream_capture_sequence(asset_store) -> None:
    scene = scene_create("uniform", 256, asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    resampled = oi_spatial_resample(oi_crop(oi, "border"), 3.0, "um")

    sensor_array = sensorCreateIMX490(
        resampled,
        "exp time",
        0.1,
        "noise flag",
        0,
        asset_store=asset_store,
    )

    assert isinstance(sensor_array, list)
    assert [str(sensor_get(sensor, "name")) for sensor in sensor_array] == [
        "large-HCG",
        "large-LCG",
        "small-HCG",
        "small-LCG",
    ]
    assert all(tuple(sensor_get(sensor, "size")) == tuple(oi_get(resampled, "size")) for sensor in sensor_array)
    assert all(np.isclose(float(sensor_get(sensor, "exp time")), 0.1) for sensor in sensor_array)
    assert all(np.asarray(sensor_get(sensor, "volts"), dtype=float).shape == tuple(oi_get(resampled, "size")) for sensor in sensor_array)
    np.testing.assert_allclose(
        [float(sensor_get(sensor, "pixel conversion gain")) for sensor in sensor_array],
        np.array([200e-6, 49e-6, 200e-6, 49e-6], dtype=float),
    )
    np.testing.assert_allclose(
        [float(sensor_get(sensor, "pixel read noise electrons")) for sensor in sensor_array],
        np.array([0.83, 3.05, 0.83, 2.96], dtype=float),
    )
    np.testing.assert_allclose(
        [float(np.mean(np.asarray(sensor_get(sensor, "pixel spectral qe"), dtype=float))) for sensor in sensor_array],
        np.array([1.0, 1.0, 0.1, 0.1], dtype=float),
    )


def test_sensor_create_array_supports_ideal_array_types(asset_store) -> None:
    xyz_array = sensor_create_array("array type", "xyz", "size", np.array([20, 24], dtype=int), asset_store=asset_store)

    assert isinstance(xyz_array, list)
    assert [sensor_get(current, "name") for current in xyz_array] == ["CIE-X-ideal", "CIE-Y-ideal", "CIE-Z-ideal"]
    assert [sensor_get(current, "filter names") for current in xyz_array] == [["rX"], ["gY"], ["bZ"]]
    assert all(tuple(np.asarray(sensor_get(current, "size"), dtype=int)) == (20, 24) for current in xyz_array)
    assert all(np.array_equal(np.asarray(sensor_get(current, "pattern"), dtype=int), np.array([[1]], dtype=int)) for current in xyz_array)
    assert all(np.isclose(float(sensor_get(current, "exp time")), 1.0) for current in xyz_array)
    assert all(sensor_get(current, "noise flag") == -1 for current in xyz_array)
    assert all(current.fields["mosaic"] is False for current in xyz_array)

    example = sensor_set(sensor_create("default", asset_store=asset_store), "exp time", 0.02)
    matched_xyz = sensor_create_array("array type", "matchxyz", "sensor example", example, "size", np.array([16, 18], dtype=int), asset_store=asset_store)

    assert isinstance(matched_xyz, list)
    assert [sensor_get(current, "name") for current in matched_xyz] == ["mono-rDefault", "mono-gDefault", "mono-bDefault"]
    assert [sensor_get(current, "filter names") for current in matched_xyz] == [["rX"], ["gY"], ["bZ"]]
    assert all(tuple(np.asarray(sensor_get(current, "size"), dtype=int)) == (16, 18) for current in matched_xyz)
    assert all(np.isclose(float(sensor_get(current, "exp time")), 0.02) for current in matched_xyz)

    monochrome = sensor_create_array("array type", "monochrome", "size", np.array([12, 14], dtype=int), asset_store=asset_store)

    assert isinstance(monochrome, sensor_module.Sensor)
    assert tuple(np.asarray(sensor_get(monochrome, "size"), dtype=int)) == (12, 14)
    assert sensor_get(monochrome, "filter names") == ["w"]
    assert np.array_equal(np.asarray(sensor_get(monochrome, "pattern"), dtype=int), np.array([[1]], dtype=int))


def test_sensor_create_array_accepts_empty_sensor_example_placeholder(asset_store) -> None:
    xyz_array = sensor_create_array(
        "array type",
        "xyz",
        "sensor example",
        [],
        "size",
        np.array([10, 12], dtype=int),
        asset_store=asset_store,
    )

    assert isinstance(xyz_array, list)
    assert [sensor_get(current, "name") for current in xyz_array] == ["CIE-X-ideal", "CIE-Y-ideal", "CIE-Z-ideal"]
    assert all(tuple(np.asarray(sensor_get(current, "size"), dtype=int)) == (10, 12) for current in xyz_array)

    monochrome = sensor_create_array(
        "array type",
        "monochrome",
        "sensor example",
        [],
        "size",
        np.array([8, 10], dtype=int),
        asset_store=asset_store,
    )

    assert isinstance(monochrome, sensor_module.Sensor)
    assert tuple(np.asarray(sensor_get(monochrome, "size"), dtype=int)) == (8, 10)
    assert sensor_get(monochrome, "name") == "Monochrome"

    with pytest.raises(ValueError, match=r"sensorCreateIdeal\('match xyz', \.\.\.\) requires a sensor example\."):
        sensor_create_array("array type", "matchxyz", "sensor example", [], asset_store=asset_store)


def test_sensor_create_array_monochrome_rejects_nonempty_sensor_example(asset_store) -> None:
    example = sensor_create("default", asset_store=asset_store)

    with pytest.raises(ValueError, match=r"Sensor example not used for monochrome case\."):
        sensor_create_array("array type", "monochrome", "sensor example", example, asset_store=asset_store)


def test_run_python_case_supports_sensor_split_pixel_ovt_saturated_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_split_pixel_ovt_saturated_small", asset_store=asset_store)

    assert case.payload["combined_volts"].shape == (32, 48)
    assert case.payload["sensor_max_volts"].shape == (3,)
    assert case.payload["saturated_counts"].shape == (3,)
    assert list(case.payload["sensor_names"]) == ["ovt-LPDLCG", "ovt-LPDHCG", "ovt-SPDLCG"]


def test_ie_read_spectra_resolves_sensor_color_filter_assets(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)

    foveon = ie_read_spectra("Foveon", wave, asset_store=asset_store)
    nikon = ie_read_spectra("NikonD1", wave, asset_store=asset_store)

    assert foveon.shape == (31, 3)
    assert nikon.shape == (31, 3)
    assert np.all(foveon >= 0.0)
    assert np.all(nikon >= 0.0)


def test_sensor_compute_supports_stacked_pixel_sensor_arrays(asset_store) -> None:
    scene = scene_create("macbeth d65", 32, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene = scene_set(scene, "hfov", 8.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 4.0)
    oi = oi_set(oi, "optics focal length", 3e-3)
    oi = oi_compute(oi, scene)

    wave = np.asarray(scene_get(scene, "wave"), dtype=float)
    foveon = np.asarray(ie_read_spectra("Foveon", wave, asset_store=asset_store), dtype=float)
    sensors = []
    for index in range(3):
        sensor = sensor_create("monochrome", asset_store=asset_store)
        sensor = sensor_set(sensor, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor = sensor_set(sensor, "exp time", 0.1)
        sensor = sensor_set(sensor, "filter spectra", foveon[:, index])
        sensor = sensor_set(sensor, "name", f"Channel-{index + 1}")
        sensor = sensor_set_size_to_fov(sensor, scene_get(scene, "fov"), oi)
        sensor = sensor_set(sensor, "wave", wave)
        sensors.append(sensor)

    computed = sensor_compute(sensors, oi)

    assert isinstance(computed, list)
    assert len(computed) == 3
    assert all(np.asarray(sensor_get(sensor, "volts"), dtype=float).shape == (245, 300) for sensor in computed)

    stacked = np.stack([np.asarray(sensor_get(sensor, "volts"), dtype=float) for sensor in computed], axis=2)
    sensor_foveon = sensor_create(asset_store=asset_store)
    sensor_foveon = sensor_set(sensor_foveon, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
    sensor_foveon = sensor_set(sensor_foveon, "autoexp", 1)
    sensor_foveon = sensor_set_size_to_fov(sensor_foveon, scene_get(scene, "fov"), oi)
    sensor_foveon = sensor_set(sensor_foveon, "wave", wave)
    sensor_foveon = sensor_set(sensor_foveon, "filter spectra", foveon)
    sensor_foveon = sensor_set(sensor_foveon, "pattern", np.array([[2]], dtype=int))
    sensor_foveon = sensor_set(sensor_foveon, "volts", stacked)

    ip = ip_compute(ip_create(asset_store=asset_store), sensor_foveon, asset_store=asset_store)
    udata, _ = ip_plot(ip, "horizontal line", np.array([1, 120], dtype=int))

    assert stacked.shape == (245, 300, 3)
    assert np.asarray(udata["values"], dtype=float).shape == (300, 3)


def test_run_python_case_supports_sensor_stacked_pixels_foveon_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_stacked_pixels_foveon_small", asset_store=asset_store)

    assert case.payload["stacked_center_patch_mean"].shape == (3,)
    assert case.payload["stacked_center_patch_std"].shape == (3,)
    assert case.payload["stacked_center_patch_p90"].shape == (3,)
    assert case.payload["stacked_mean_volts"].shape == (3,)
    assert case.payload["stacked_std_volts"].shape == (3,)
    assert float(case.payload["line_row"]) == 120.0
    assert case.payload["bayer_line_mean"].shape == (3,)
    assert case.payload["bayer_line_std"].shape == (3,)
    assert case.payload["bayer_line_p90"].shape == (3,)


def test_run_python_case_supports_sensor_microlens_etendue_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_microlens_etendue_small", asset_store=asset_store)

    assert case.payload["no_microlens_etendue"].ndim == 2
    assert case.payload["centered_etendue"].shape == case.payload["no_microlens_etendue"].shape
    assert case.payload["optimal_etendue"].shape == case.payload["no_microlens_etendue"].shape
    assert case.payload["ray_angles_deg"].shape == (10,)
    assert case.payload["optimal_offset_curve_um"].shape == (10,)
    assert case.payload["optimal_offsets_default_um"].shape == case.payload["no_microlens_etendue"].shape
    assert case.payload["optimal_offsets_half_fnumber_um"].shape == case.payload["no_microlens_etendue"].shape
    assert case.payload["optimal_offsets_source_f4_um"].shape == case.payload["no_microlens_etendue"].shape
    assert case.payload["optimal_offsets_source_f16_um"].shape == case.payload["no_microlens_etendue"].shape
    assert case.payload["radiance_midline_neg10"].shape == (255,)
    assert case.payload["radiance_midline_0"].shape == (255,)
    assert case.payload["radiance_midline_10"].shape == (255,)


def test_optics_microlens_workflow_supports_getters_and_radiance(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "fov", 30.0, oi)
    microlens = mlens_create(asset_store=asset_store)

    assert mlens_get(microlens, "name") == "default"
    assert mlens_get(microlens, "type") == "microlens"
    assert float(mlens_get(microlens, "source fnumber")) > 0.0
    assert float(mlens_get(microlens, "source diameter", "meters")) > 0.0
    assert float(mlens_get(microlens, "source diameter", "microns")) > 0.0
    assert float(mlens_get(microlens, "ml fnumber")) > 0.0
    assert float(mlens_get(microlens, "ml diameter", "meters")) > 0.0
    assert float(mlens_get(microlens, "ml diameter", "microns")) > 0.0
    assert float(sensor_get(sensor, "fov", oi)) == pytest.approx(30.0, abs=0.05)

    microlens = mlens_set(microlens, "chief ray angle", 10.0)
    assert float(mlens_get(microlens, "chief ray angle")) == pytest.approx(10.0)

    radiance_microlens = ml_radiance(mlens_create(asset_store=asset_store), asset_store=asset_store)
    source_irradiance = np.asarray(mlens_get(radiance_microlens, "source irradiance"), dtype=float)
    pixel_irradiance = np.asarray(mlens_get(radiance_microlens, "pixel irradiance"), dtype=float)
    x_coordinate = np.asarray(mlens_get(radiance_microlens, "x coordinate"), dtype=float)

    assert source_irradiance.shape == (255, 255)
    assert pixel_irradiance.shape == (255, 255)
    assert x_coordinate.shape == (255,)
    assert float(mlens_get(radiance_microlens, "etendue")) > 0.0


def test_microlens_compatibility_wrappers_cover_current_description_and_import(asset_store) -> None:
    session = session_create()
    oi = oi_create(asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store, session=session)
    microlens = mlens_create(sensor, oi, asset_store=asset_store)

    assert mlGetCurrent(session=session) is None

    updated_sensor = mlSetCurrent(microlens, session=session)
    current = mlGetCurrent(session=session)

    assert current is not None
    assert current["type"] == "microlens"
    assert sensor_get(updated_sensor, "ml")["type"] == "microlens"

    imported = mlImportParams(current, oi_get(oi, "optics"), sensor_get(updated_sensor, "pixel"))
    assert float(mlens_get(imported, "source fnumber")) == pytest.approx(float(oi_get(oi, "optics fnumber")))
    assert float(mlens_get(imported, "ml focal length")) == pytest.approx(float(sensor_get(updated_sensor, "pixel depth")))

    description = mlDescription(current, updated_sensor)
    assert "Pixel width (um)" in description
    assert "Optimal offset" in description

    printed = mlPrint(current)
    assert "Microlens properties:" in printed
    assert "Focal length (um):" in printed


def test_run_python_case_supports_optics_microlens_parity_case(asset_store) -> None:
    case = run_python_case_with_context("optics_microlens_small", asset_store=asset_store)

    assert case.payload["name"] == "default"
    assert case.payload["type"] == "microlens"
    assert float(case.payload["source_fnumber"]) > 0.0
    assert float(case.payload["source_diameter_m"]) > 0.0
    assert float(case.payload["source_diameter_um"]) > 0.0
    assert float(case.payload["ml_fnumber"]) > 0.0
    assert float(case.payload["ml_diameter_m"]) > 0.0
    assert float(case.payload["ml_diameter_um"]) > 0.0
    assert float(case.payload["chief_ray_angle_default_deg"]) == pytest.approx(0.0)
    assert float(case.payload["chief_ray_angle_set_deg"]) == pytest.approx(10.0)
    assert float(case.payload["sensor_fov_deg"]) == pytest.approx(30.0, abs=0.05)
    assert case.payload["x_coordinate_um"].shape == (255,)
    assert case.payload["source_center_row"].shape == (255,)
    assert case.payload["pixel_center_row"].shape == (255,)
    assert case.payload["source_irradiance_stats"].shape == (4,)
    assert case.payload["pixel_irradiance_stats"].shape == (4,)
    assert float(case.payload["etendue"]) > 0.0


def test_sensor_comparison_workflow_supports_mixed_sensor_types(asset_store) -> None:
    patch_size = 24
    scene_c = scene_create("macbeth d65", patch_size, asset_store=asset_store)
    macbeth_size = np.asarray(scene_get(scene_c, "size"), dtype=int)
    scene_c = scene_set(
        scene_c,
        "resize",
        np.rint(np.array([macbeth_size[0], macbeth_size[1] / 2.0], dtype=float)).astype(int),
    )
    scene_s = scene_create("sweep frequency", int(macbeth_size[0]), float(macbeth_size[0]) / 16.0, asset_store=asset_store)
    scene = scene_combine(scene_c, scene_s, "direction", "horizontal")
    scene = scene_set(scene, "fov", 20.0)

    assert tuple(scene_get(scene, "size")) == (96, 168)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 1.2)
    oi = oi_compute(oi, scene)

    scene_vfov = float(scene_get(scene, "vfov"))
    small_sizes = []
    large_sizes = []
    for sensor_type in ("imx363", "mt9v024", "cyym"):
        if sensor_type == "mt9v024":
            sensor = sensor_create(sensor_type, None, "rccc", asset_store=asset_store)
        else:
            sensor = sensor_create(sensor_type, asset_store=asset_store)

        sensor = sensor_set(sensor, "pixel size", 1.5e-6)
        sensor = sensor_set(sensor, "hfov", 20.0, oi)
        sensor = sensor_set(sensor, "vfov", scene_vfov)
        sensor = sensor_set(sensor, "auto exposure", True)
        sensor = sensor_compute(sensor, oi)
        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        small_sizes.append(tuple(sensor_get(sensor, "size")))
        assert volts.shape == tuple(sensor_get(sensor, "size"))
        assert np.all(np.isfinite(volts))

        if sensor_type == "imx363":
            ip = ip_compute(ip_create("imx363 RGB", sensor, asset_store=asset_store), sensor, asset_store=asset_store)
            assert np.asarray(ip_get(ip, "result"), dtype=float).shape == volts.shape + (3,)
        elif sensor_type == "mt9v024":
            ip = ip_create("mt9v024 RCCC", sensor, asset_store=asset_store)
            ip = ip_set(ip, "demosaic method", "analog rccc")
            ip = ip_compute(ip, sensor, asset_store=asset_store)
            assert np.asarray(ip_get(ip, "result"), dtype=float).shape == volts.shape + (3,)

        sensor = sensor_set(sensor, "pixel size constant fill factor", 6e-6)
        sensor = sensor_set(sensor, "hfov", 20.0, oi)
        sensor = sensor_set(sensor, "vfov", scene_vfov)
        sensor = sensor_set(sensor, "auto exposure", True)
        sensor = sensor_compute(sensor, oi)
        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        large_sizes.append(tuple(sensor_get(sensor, "size")))
        assert volts.shape == tuple(sensor_get(sensor, "size"))
        assert np.all(np.isfinite(volts))

    assert len(set(small_sizes)) == 1
    assert len(set(large_sizes)) == 1
    assert small_sizes[0][0] > large_sizes[0][0]
    assert small_sizes[0][1] > large_sizes[0][1]


def test_run_python_case_supports_sensor_comparison_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_comparison_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (96, 168)
    assert case.payload["oi_size"].shape == (2,)
    assert case.payload["small_sensor_sizes"].shape == (3, 2)
    assert case.payload["nonimx_small_sensor_mean_volts"].shape == (2,)
    assert case.payload["nonimx_small_sensor_p90_volts"].shape == (2,)
    assert case.payload["large_sensor_sizes"].shape == (3, 2)
    assert case.payload["nonimx_large_sensor_mean_volts"].shape == (2,)
    assert case.payload["nonimx_large_sensor_p90_volts"].shape == (2,)
    assert float(case.payload["imx363_mean_ratio_large_small"]) > 0.0
    assert float(case.payload["imx363_p90_ratio_large_small"]) > 0.0
    assert case.payload["small_ip_sizes"].shape == (2, 3)
    assert case.payload["large_ip_sizes"].shape == (2, 3)


def test_sensor_compute_samples_returns_multiple_noisy_captures(asset_store) -> None:
    scene = scene_create("slanted bar", 128, asset_store=asset_store)
    scene = scene_set(scene, "fov", 4.0)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "exp time", 0.05)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor_nf = sensor_compute(sensor, oi, seed=0)

    samples = np.asarray(sensor_compute_samples(sensor_nf, 8, 2, seed=7), dtype=float)
    repeated = np.asarray(sensor_compute_samples(sensor_nf, 8, 2, seed=7), dtype=float)

    assert samples.shape == np.asarray(sensor_get(sensor_nf, "volts"), dtype=float).shape + (8,)
    assert np.allclose(samples, repeated)
    assert not np.allclose(samples[:, :, 0], samples[:, :, 1])


def test_run_python_case_supports_sensor_noise_samples_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_noise_samples_small", asset_store=asset_store)

    assert case.payload["sample_shape"].shape == (3,)
    assert int(case.payload["sample_shape"][2]) == 64
    assert int(case.payload["sample_shape"][0]) > 0
    assert int(case.payload["sample_shape"][1]) > 0
    assert float(case.payload["noise_free_mean"]) > 0.0
    assert case.payload["noise_std_image_stats"].shape == (4,)
    assert case.payload["noise_distribution_stats"].shape == (4,)
    assert case.payload["mean_residual_stats"].shape == (2,)
    assert case.payload["pair_diff_stats"].shape == (4,)


def test_run_python_case_supports_sensor_mcc_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_mcc_small", asset_store=asset_store)

    assert tuple(case.payload["mosaic_size"]) == (600, 800)
    assert case.payload["volts_stats"].shape == (4,)
    assert case.payload["estimated_ccm"].shape == (3, 3)
    assert case.payload["uncorrected_mean_rgb_norm"].shape == (3,)
    assert case.payload["uncorrected_p95_rgb_norm"].shape == (3,)
    assert case.payload["corrected_mean_rgb_norm"].shape == (3,)
    assert case.payload["corrected_p95_rgb_norm"].shape == (3,)
    assert np.all(np.isfinite(case.payload["estimated_ccm"]))


def test_scene_rotate_and_oi_crop_support_centered_rolling_shutter_geometry(asset_store) -> None:
    scene = scene_create("star pattern", 48, "ee", 4, asset_store=asset_store)
    scene = scene_set(scene, "fov", 3.0)
    rotated = scene_rotate(scene, 12.0)

    original_size = np.asarray(scene_get(scene, "size"), dtype=int)
    rotated_size = np.asarray(scene_get(rotated, "size"), dtype=int)
    assert rotated_size[0] > original_size[0]
    assert rotated_size[1] > original_size[1]

    oi = oi_compute(oi_create(asset_store=asset_store), rotated)
    center_pixel = np.asarray(oi_get(oi, "center pixel"), dtype=int)
    crop_rows = 24
    crop_cols = 20
    rect = np.array(
        [
            int(np.rint(center_pixel[1] - (crop_cols - 1) / 2.0)),
            int(np.rint(center_pixel[0] - (crop_rows - 1) / 2.0)),
            crop_cols - 1,
            crop_rows - 1,
        ],
        dtype=int,
    )
    cropped = oi_crop(oi, rect)

    assert tuple(oi_get(cropped, "size")) == (crop_rows, crop_cols)
    assert np.allclose(oi_get(cropped, "distance per sample"), oi_get(oi, "distance per sample"), rtol=5e-3)
    assert np.asarray(oi_get(cropped, "illuminance"), dtype=float).shape == (crop_rows, crop_cols)


def test_run_python_case_supports_sensor_rolling_shutter_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_rolling_shutter_small", asset_store=asset_store)

    assert tuple(case.payload["sensor_size"]) == tuple(case.payload["crop_size"])
    assert int(case.payload["n_frames"]) > int(case.payload["sensor_size"][0])
    assert case.payload["first_crop_rect"].shape == (4,)
    assert case.payload["last_crop_rect"].shape == (4,)
    assert case.payload["temporal_mean_volts"].ndim == 1
    assert case.payload["center_pixel_trace"].shape == case.payload["temporal_mean_volts"].shape
    assert case.payload["final_stats"].shape == (4,)
    assert case.payload["sampled_rows"].shape == (3,)
    assert case.payload["sampled_cols"].shape == (3,)
    assert case.payload["sampled_row_stats"].shape == (3, 4)
    assert case.payload["result_mean_rgb_norm"].shape == (3,)
    assert case.payload["result_p95_rgb_norm"].shape == (3,)


def test_oi_crop_border_and_imx490_compute_support_uniform_hdr_workflow(asset_store) -> None:
    scene = scene_create("uniform", 256, asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    cropped = oi_crop(oi, "border")
    resampled = oi_spatial_resample(cropped, 3.0, "um")

    assert tuple(oi_get(cropped, "size"))[0] == tuple(oi_get(cropped, "size"))[1]
    assert np.isclose(float(np.asarray(oi_get(resampled, "distance per sample"), dtype=float)[0]), 3.0e-6, rtol=1e-4)

    sensor, metadata = imx490_compute(
        resampled,
        "method",
        "best snr",
        "exp time",
        0.1,
        "noise flag",
        0,
        asset_store=asset_store,
    )
    captures = metadata["sensorArray"]

    assert len(captures) == 4
    assert [str(sensor_get(capture, "name")) for capture in captures] == ["large-1x", "large-4x", "small-1x", "small-4x"]
    assert tuple(sensor_get(sensor, "size")) == tuple(oi_get(resampled, "size"))
    assert np.asarray(sensor_get(sensor, "volts"), dtype=float).shape == tuple(oi_get(resampled, "size"))
    assert np.asarray(sensor.metadata["bestPixel"], dtype=int).shape == tuple(oi_get(resampled, "size"))


def test_oi_pad_crop_supports_script_geometry_and_sensor_fov_flow(asset_store) -> None:
    scene = scene_create("sweep frequency", asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)

    padded_size = np.asarray(oi_get(oi, "size"), dtype=float)
    original_size = padded_size / 1.25
    offset = (padded_size - original_size) / 2.0
    rect = np.array([offset[1] + 1.0, offset[0] + 1.0, original_size[1] - 1.0, original_size[0] - 1.0], dtype=float)
    oi_cropped = oi_crop(oi, rect)

    assert tuple(np.rint(rect).astype(int)) == (17, 17, 127, 127)
    assert tuple(oi_get(oi_cropped, "size")) == tuple(scene_get(scene, "size"))
    assert float(oi_get(oi_cropped, "fov")) > float(scene_get(scene, "fov"))
    assert float(oi_get(oi_cropped, "fov")) < float(oi_get(oi, "fov"))

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "fov", scene_get(scene, "fov"), oi)
    sensor_from_padded = sensor_compute(sensor, oi, seed=0)
    sensor_from_cropped = sensor_compute(sensor, oi_cropped, seed=0)
    volts_padded = np.asarray(sensor_get(sensor_from_padded, "volts"), dtype=float)
    volts_cropped = np.asarray(sensor_get(sensor_from_cropped, "volts"), dtype=float)
    normalized_mae = float(np.mean(np.abs(volts_padded - volts_cropped))) / max(float(np.mean(np.abs(volts_cropped))), 1e-12)

    assert normalized_mae < 0.02

    sensor_padded = sensor_set_size_to_fov(sensor.clone(), oi_get(oi, "fov"), oi)
    sensor_padded = sensor_compute(sensor_padded, oi, seed=0)
    assert tuple(sensor_get(sensor_padded, "size"))[0] > tuple(sensor_get(sensor, "size"))[0]


def test_run_python_case_supports_sensor_imx490_uniform_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_imx490_uniform_small", asset_store=asset_store)

    assert tuple(case.payload["oi_size"])[0] == tuple(case.payload["oi_size"])[1]
    assert list(case.payload["capture_names"]) == ["large-1x", "large-4x", "small-1x", "small-4x"]
    assert case.payload["capture_mean_electrons"].shape == (4,)
    assert case.payload["capture_mean_volts"].shape == (4,)
    assert case.payload["capture_mean_dv"].shape == (4,)
    assert float(case.payload["large_gain_ratio"]) > 1.0
    assert float(case.payload["small_area_ratio"]) < 1.0
    assert case.payload["combined_volts_stats"].shape == (4,)
    assert case.payload["best_pixel_counts"].shape == (4,)
    assert int(np.sum(case.payload["best_pixel_counts"])) == int(np.prod(case.payload["oi_size"]))


def test_scene_from_file_multispectral_supports_hdr_pixel_size_workflow(asset_store) -> None:
    scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/Feng_Office-hdrs.mat"),
        "multispectral",
        200.0,
        asset_store=asset_store,
    )
    oi = oi_compute(oi_create(asset_store=asset_store), scene)

    pixel_sizes_um = np.array([1.0, 2.0, 4.0], dtype=float)
    dye_size_um = 512.0
    base_sensor = sensor_create("monochrome", asset_store=asset_store)
    base_sensor = sensor_set(base_sensor, "exp time", 0.003)

    volt_means: list[float] = []
    sensor_sizes: list[tuple[int, int]] = []
    result_shapes: list[tuple[int, int, int]] = []

    for pixel_size_um in pixel_sizes_um:
        sensor = sensor_set(base_sensor.clone(), "pixel size constant fill factor", np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6)
        sensor = sensor_set(sensor, "rows", int(np.rint(dye_size_um / pixel_size_um)))
        sensor = sensor_set(sensor, "cols", int(np.rint(dye_size_um / pixel_size_um)))
        sensor = sensor_compute(sensor, oi)
        ip = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

        sensor_sizes.append(tuple(int(value) for value in sensor_get(sensor, "size")))
        result_shapes.append(tuple(int(value) for value in np.asarray(ip_get(ip, "result"), dtype=float).shape))
        volt_means.append(float(np.mean(np.asarray(sensor_get(sensor, "volts"), dtype=float))))

    assert tuple(scene_get(scene, "size")) == (506, 759)
    assert np.asarray(scene_get(scene, "illuminant photons"), dtype=float).shape == (506, 759, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 200.0, rtol=5e-2)
    assert sensor_sizes == [(512, 512), (256, 256), (128, 128)]
    assert result_shapes == [(512, 512, 3), (256, 256, 3), (128, 128, 3)]
    assert volt_means[0] < volt_means[1] < volt_means[2]


def test_run_python_case_supports_sensor_hdr_pixel_size_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_hdr_pixel_size_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (506, 759)
    assert case.payload["wave"].shape == (31,)
    assert np.array_equal(case.payload["pixel_sizes_um"], np.array([1.0, 2.0, 4.0], dtype=float))
    assert case.payload["sensor_sizes"].shape == (3, 2)
    assert np.array_equal(case.payload["sensor_sizes"], np.array([[512, 512], [256, 256], [128, 128]], dtype=int))
    assert case.payload["mean_volts"].shape == (3,)
    assert case.payload["p95_volts"].shape == (3,)
    assert case.payload["mean_electrons"].shape == (3,)
    assert case.payload["result_sizes"].shape == (3, 3)
    assert case.payload["result_mean_gray"].shape == (3,)
    assert case.payload["result_p95_gray"].shape == (3,)


def test_sensor_log_ar0132at_workflow_matches_script_contract(asset_store) -> None:
    linear_scene = scene_create("linear intensity ramp", 64, 2**8, asset_store=asset_store)
    exp_scene = scene_create("exponential intensity ramp", 256, 2**16, asset_store=asset_store)
    exp_scene = scene_set(exp_scene, "fov", 60.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 2.8)
    oi = oi_compute(oi, exp_scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "response type", "log")
    sensor = sensor_set(sensor, "size", np.array([960, 1280], dtype=int))
    sensor = sensor_set(sensor, "pixel size same fill factor", 3.751e-6)

    wave = np.asarray(scene_get(exp_scene, "wave"), dtype=float)
    filter_spectra, filter_names, _ = ie_read_color_filter(wave, asset_store.resolve("data/sensor/colorfilters/auto/ar0132at.mat"))
    sensor = sensor_set(sensor, "filter spectra", filter_spectra)
    sensor = sensor_set(sensor, "filter names", filter_names)
    sensor = sensor_set(sensor, "pixel read noise volts", 1.0e-3)
    sensor = sensor_set(sensor, "pixel voltage swing", 2.8)
    sensor = sensor_set(sensor, "pixel dark voltage", 1.0e-3)
    sensor = sensor_set(sensor, "pixel conversion gain", 110.0e-6)
    sensor = sensor_set(sensor, "exp time", 0.003)
    sensor = sensor_compute(sensor, oi)

    volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
    assert tuple(scene_get(linear_scene, "size")) == (64, 64)
    assert tuple(scene_get(exp_scene, "size")) == (256, 256)
    assert tuple(sensor_get(sensor, "size")) == (960, 1280)
    assert volts.shape == (960, 1280)
    assert np.isclose(sensor_dr(sensor, 1.0), 34.243414090728336, atol=1e-9)
    assert np.all(np.isfinite(volts))
    assert float(np.max(volts)) <= 2.8 + 1e-12
    assert float(np.mean(volts[14, :])) < float(np.mean(volts[113, :]))


def test_run_python_case_supports_sensor_log_ar0132at_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_log_ar0132at_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (256, 256)
    assert tuple(case.payload["sensor_size"]) == (960, 1280)
    assert case.payload["wave"].shape == (31,)
    assert np.isclose(case.payload["dr_at_1s"], 34.243414090728336, atol=1e-9)
    assert case.payload["volts_stats"].shape == (4,)
    assert case.payload["sampled_cols"].shape == (33,)
    assert case.payload["row15_stats"].shape == (4,)
    assert case.payload["row114_stats"].shape == (4,)
    assert case.payload["row15_profile_norm"].shape == (33,)
    assert case.payload["row114_profile_norm"].shape == (33,)


def test_sensor_aliasing_workflow_matches_script_contract(asset_store) -> None:
    fov = 5.0
    sweep_scene = scene_create("sweep frequency", 768, 30.0, asset_store=asset_store)
    sweep_scene = scene_set(sweep_scene, "fov", fov)

    oi = oi_create("diffraction limited", asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 2.0)
    oi = oi_compute(oi, sweep_scene)

    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set_size_to_fov(sensor, fov, oi)
    sensor = sensor_set(sensor, "noise flag", 0)

    sensor_small = sensor_set(sensor.clone(), "pixel size constant fill factor", 2.0e-6)
    sensor_small = sensor_compute(sensor_small, oi)
    small_plot, _ = sensor_plot(sensor_small, "electrons hline", np.array([5, 1], dtype=int))

    sensor_large = sensor_set(sensor.clone(), "pixel size constant fill factor", 6.0e-6)
    sensor_large = sensor_set_size_to_fov(sensor_large, fov, oi)
    sensor_large = sensor_compute(sensor_large, oi)
    large_plot, _ = sensor_plot(sensor_large, "electrons hline", np.array([5, 1], dtype=int))

    oi_blur = oi_set(oi.clone(), "optics fnumber", 12.0)
    oi_blur = oi_compute(oi_blur, sweep_scene)
    sensor_blur = sensor_compute(sensor_large.clone(), oi_blur)
    blur_plot, _ = sensor_plot(sensor_blur, "electrons hline", np.array([5, 1], dtype=int))

    slanted_scene = scene_create("slanted bar", 1024, asset_store=asset_store)
    slanted_scene = scene_set(slanted_scene, "fov", fov)
    oi_sharp = oi_set(oi_blur.clone(), "optics fnumber", 2.0)
    oi_sharp = oi_compute(oi_sharp, slanted_scene)
    sensor_slanted = sensor_set(sensor_large.clone(), "pixel size constant fill factor", 6.0e-6)
    sensor_slanted = sensor_set_size_to_fov(sensor_slanted, fov, oi_sharp)
    sensor_slanted = sensor_compute(sensor_slanted, oi_sharp)
    oi_soft = oi_set(oi_sharp.clone(), "optics fnumber", 12.0)
    oi_soft = oi_compute(oi_soft, slanted_scene)
    sensor_slanted_blur = sensor_compute(sensor_slanted.clone(), oi_soft)

    assert tuple(sensor_get(sensor_small, "size")) == (99, 120)
    assert tuple(sensor_get(sensor_large, "size")) == (46, 56)
    assert small_plot["dataType"] == large_plot["dataType"] == blur_plot["dataType"] == "electrons"
    assert len(small_plot["data"]) == 1
    assert len(large_plot["data"]) == 1
    assert len(blur_plot["data"]) == 1
    assert np.asarray(small_plot["data"][0], dtype=float).shape == (120,)
    assert np.asarray(large_plot["data"][0], dtype=float).shape == (56,)
    assert np.asarray(blur_plot["data"][0], dtype=float).shape == (56,)
    assert np.std(np.asarray(blur_plot["data"][0], dtype=float)) < np.std(np.asarray(large_plot["data"][0], dtype=float))
    assert tuple(sensor_get(sensor_slanted, "size")) == (46, 56)
    assert np.asarray(sensor_get(sensor_slanted, "electrons"), dtype=float).shape == (46, 56)
    assert np.asarray(sensor_get(sensor_slanted_blur, "electrons"), dtype=float).shape == (46, 56)


def test_run_python_case_supports_sensor_aliasing_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_aliasing_small", asset_store=asset_store)

    assert tuple(case.payload["sweep_scene_size"]) == (768, 768)
    assert tuple(case.payload["small_sensor_size"]) == (99, 120)
    assert tuple(case.payload["large_sensor_size"]) == (46, 56)
    assert case.payload["small_line_pos"].shape == (120,)
    assert case.payload["small_line_data_norm"].shape == (120,)
    assert case.payload["large_line_pos"].shape == (56,)
    assert case.payload["large_line_data_norm"].shape == (56,)
    assert case.payload["blur_line_pos"].shape == (56,)
    assert case.payload["blur_line_data_norm"].shape == (56,)
    assert case.payload["small_line_stats"].shape == (4,)
    assert case.payload["large_line_stats"].shape == (4,)
    assert case.payload["blur_line_stats"].shape == (4,)
    assert tuple(case.payload["slanted_scene_size"]) == (1025, 1025)
    assert tuple(case.payload["slanted_sensor_size"]) == (46, 56)
    assert case.payload["slanted_sharp_center_row_norm"].shape == (129,)
    assert case.payload["slanted_sharp_center_col_norm"].shape == (129,)
    assert case.payload["slanted_blur_center_row_norm"].shape == (129,)
    assert case.payload["slanted_blur_center_col_norm"].shape == (129,)
    assert case.payload["slanted_sharp_stats"].shape == (4,)
    assert case.payload["slanted_blur_stats"].shape == (4,)


def test_sensor_external_analysis_workflow_matches_script_contract(asset_store) -> None:
    from scipy.io import loadmat

    dut = sensor_create(asset_store=asset_store)
    dut = sensor_set(dut, "name", "My Sensor")

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    dut = sensor_set(dut, "wave", wave)
    dut = sensor_set(dut, "colorFilters", ie_read_spectra("RGB.mat", wave, asset_store=asset_store))
    dut = sensor_set(dut, "irFilter", ie_read_spectra("infrared2.mat", wave, asset_store=asset_store))
    dut = sensor_set(dut, "cfapattern", np.array([[2, 1], [3, 2]], dtype=int))
    dut = sensor_set(dut, "size", [144, 176])
    dut = sensor_set(dut, "pixel name", "My Pixel")
    dut = sensor_set(dut, "pixel size constant fill factor", [2.0e-6, 2.0e-6])
    dut = sensor_set(dut, "pixel spectral qe", ie_read_spectra("photodetector.mat", wave, asset_store=asset_store))
    dut = sensor_set(dut, "pixel voltage swing", 1.5)

    volts = loadmat(asset_store.resolve("scripts/sensor/dutData.mat"), squeeze_me=True, struct_as_record=False)["volts"]
    dut = sensor_set(dut, "volts", volts)

    assert sensor_get(dut, "name") == "My Sensor"
    assert np.array_equal(sensor_get(dut, "wave"), wave)
    assert np.array_equal(sensor_get(dut, "cfapattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert tuple(sensor_get(dut, "size")) == (144, 176)
    assert sensor_get(dut, "pixel name") == "My Pixel"
    assert np.allclose(np.asarray(sensor_get(dut, "pixel size"), dtype=float), np.array([2.0e-6, 2.0e-6], dtype=float))
    assert np.isclose(sensor_get(dut, "pixel voltage swing"), 1.5)
    assert np.asarray(sensor_get(dut, "filter spectra"), dtype=float).shape == (31, 3)
    assert np.asarray(sensor_get(dut, "ir filter"), dtype=float).shape == (31,)
    assert np.asarray(sensor_get(dut, "pixel spectral qe"), dtype=float).shape == (31,)
    assert np.asarray(sensor_get(dut, "volts"), dtype=float).shape == (144, 176)
    assert np.isclose(float(np.max(np.asarray(sensor_get(dut, "volts"), dtype=float))), 0.9585338252565363)


def test_run_python_case_supports_sensor_external_analysis_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_external_analysis_small", asset_store=asset_store)

    assert case.payload["sensor_name"] == "My Sensor"
    assert case.payload["wave"].shape == (31,)
    assert case.payload["filter_spectra"].shape == (31, 3)
    assert case.payload["ir_filter"].shape == (31,)
    assert np.array_equal(case.payload["cfa_pattern"], np.array([[2, 1], [3, 2]], dtype=int))
    assert tuple(case.payload["sensor_size"]) == (144, 176)
    assert case.payload["pixel_name"] == "My Pixel"
    assert case.payload["pixel_size_m"].shape == (2,)
    assert case.payload["pixel_qe"].shape == (31,)
    assert np.isclose(case.payload["pixel_voltage_swing"], 1.5)
    assert case.payload["volts"].shape == (144, 176)
    assert case.payload["volts_stats"].shape == (4,)


def test_run_python_case_supports_sensor_filter_transmissivities_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_filter_transmissivities_small", asset_store=asset_store)

    assert case.payload["wave"].ndim == 1
    assert case.payload["filters"].shape[0] == case.payload["wave"].size
    assert case.payload["filters"].shape[1] == 3
    assert case.payload["spectral_qe"].shape == case.payload["filters"].shape


def test_ie_save_color_filter_roundtrips_gaussian_filters(tmp_path, asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    centers = np.arange(400.0, 701.0, 40.0, dtype=float)
    filters, returned_wave = sensor_color_filter("gaussian", wave, centers, np.full(centers.shape, 30.0, dtype=float))
    payload = {
        "wavelength": returned_wave,
        "data": filters,
        "filterNames": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "comment": "Gaussian filters created by unit test",
        "peakWavelengths": centers,
    }

    path = ie_save_color_filter(payload, tmp_path / "gFiltersDeleteMe")
    read_filters, read_names, file_data = ie_read_color_filter(wave, path, asset_store=asset_store)

    assert np.allclose(read_filters, filters)
    assert read_names == payload["filterNames"]
    assert np.allclose(np.asarray(file_data["peakWavelengths"], dtype=float).reshape(-1), centers)
    assert str(file_data["comment"]) == payload["comment"]


def test_run_python_case_supports_sensor_color_filter_gaussian_roundtrip_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_color_filter_gaussian_roundtrip_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["created_filters"].shape == (31, 8)
    assert np.allclose(case.payload["created_filters"], case.payload["read_filters"])
    assert list(case.payload["filter_names"]) == ["a", "b", "c", "d", "e", "f", "g", "h"]


def test_run_python_case_supports_sensor_color_filter_asset_nikond100_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_color_filter_asset_nikond100_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (601,)
    assert case.payload["filters"].shape == (601, 3)
    assert list(case.payload["filter_names"]) == ["r_custom_", "g_custom_", "b_custom_"]
    assert str(case.payload["comment"])


def test_sensor_create_supports_ycmy_and_cyym_variants(asset_store) -> None:
    ycmy = sensor_create("ycmy", asset_store=asset_store)
    cyym = sensor_create("cyym", asset_store=asset_store)

    assert np.array_equal(sensor_get(ycmy, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert np.array_equal(sensor_get(cyym, "pattern"), np.array([[1, 2], [2, 3]], dtype=int))
    assert sensor_get(ycmy, "cfaname") == "Bayer CMY"
    assert sensor_get(cyym, "cfaname") == "Bayer CMY"
    assert sensor_get(ycmy, "filtercolorletters") == "cym"


def test_run_python_case_supports_sensor_cfa_ycmy_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_cfa_ycmy_small", asset_store=asset_store)

    assert case.payload["pattern"].shape == (2, 2)
    assert tuple(case.payload["size"]) == (4, 4)
    assert case.payload["filter_spectra"].shape[1] == 3
    assert case.payload["rgb"].shape == (4, 4, 3)


def test_run_python_case_supports_sensor_cfa_pattern_and_size_rgb_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_cfa_pattern_and_size_rgb_small", asset_store=asset_store)

    assert case.payload["pattern"].shape == (3, 3)
    assert tuple(case.payload["size"]) == (6, 9)
    assert case.payload["rgb"].shape == (6, 9, 3)


def test_sensor_cfa_script_workflow(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    fov = 20.0
    pixel_size = np.array([1.4e-6, 1.4e-6], dtype=float)

    scene = scene_from_file("zebra.jpg", "rgb", 300, display_create(asset_store=asset_store), asset_store=asset_store)
    scene = scene_set(scene, "fov", fov)

    camera = camera_set(camera, "pixel size constant fill factor", pixel_size)
    camera = camera_compute(camera, scene, asset_store=asset_store)

    branches = [("default", camera_get(camera, "sensor").clone())]

    bayer = sensor_create(asset_store=asset_store)
    bayer = sensor_set(bayer, "fov", fov, camera_get(camera, "oi"))
    bayer = sensor_set(bayer, "name", "Bayer")
    camera = camera_set(camera, "sensor", bayer)
    camera = camera_set(camera, "pixel size constant fill factor", pixel_size)
    camera = camera_compute(camera, "oi", asset_store=asset_store)
    branches.append(("bayer", camera_get(camera, "sensor").clone()))

    ycmy = sensor_create("ycmy", asset_store=asset_store)
    ycmy = sensor_set(ycmy, "fov", fov, camera_get(camera, "oi"))
    ycmy = sensor_set(ycmy, "name", "cmy")
    camera = camera_set(camera, "sensor", ycmy)
    camera = camera_set(camera, "pixel size constant fill factor", pixel_size)
    camera = camera_compute(camera, "oi", asset_store=asset_store)
    branches.append(("ycmy", camera_get(camera, "sensor").clone()))

    rgb = sensor_create("rgb", asset_store=asset_store)
    rgb = sensor_set(rgb, "pattern and size", np.array([[2, 1, 2], [3, 2, 1], [2, 3, 2]], dtype=int))
    rgb = sensor_set(rgb, "fov", fov, camera_get(camera, "oi"))
    rgb = sensor_set(rgb, "name", "3x3 RGB")
    rgb = sensor_set(rgb, "pixel size constant fill factor", pixel_size)
    camera = camera_set(camera, "sensor", rgb)
    camera = camera_compute(camera, "oi", asset_store=asset_store)
    branches.append(("rgb", camera_get(camera, "sensor").clone()))

    rgbw = sensor_create("rgbw", asset_store=asset_store)
    rgbw = sensor_set(rgbw, "fov", fov, camera_get(camera, "oi"))
    rgbw = sensor_set(rgbw, "name", "rgbw")
    camera = camera_set(camera, "sensor", rgbw)
    camera = camera_set(camera, "pixel size constant fill factor", pixel_size)
    camera = camera_compute(camera, "oi", asset_store=asset_store)
    branches.append(("rgbw", camera_get(camera, "sensor").clone()))

    quad = sensor_create(asset_store=asset_store)
    quad = sensor_set(quad, "pattern", np.array([[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]], dtype=int))
    quad = sensor_set(quad, "fov", fov, camera_get(camera, "oi"))
    quad = sensor_set(quad, "name", "quad")
    camera = camera_set(camera, "sensor", quad)
    camera = camera_set(camera, "pixel size constant fill factor", pixel_size)
    camera = camera_compute(camera, scene, asset_store=asset_store)
    branches.append(("quad", camera_get(camera, "sensor").clone()))

    assert tuple(np.asarray(scene_get(scene, "size"), dtype=int)) == (391, 600)
    assert tuple(np.asarray(oi_get(camera_get(camera, "oi"), "size"), dtype=int)) == (489, 750)

    expected_sizes = {
        "default": (634, 974),
        "bayer": (398, 488),
        "ycmy": (398, 488),
        "rgb": (390, 489),
        "rgbw": (398, 488),
        "quad": (636, 976),
    }
    expected_patterns = {
        "default": np.array([[2, 1], [3, 2]], dtype=int),
        "bayer": np.array([[2, 1], [3, 2]], dtype=int),
        "ycmy": np.array([[2, 1], [3, 2]], dtype=int),
        "rgb": np.array([[2, 1, 2], [3, 2, 1], [2, 3, 2]], dtype=int),
        "rgbw": np.array([[1, 2], [3, 4]], dtype=int),
        "quad": np.array([[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]], dtype=int),
    }
    expected_cfa_names = {
        "default": "Bayer RGB",
        "bayer": "Bayer RGB",
        "ycmy": "Bayer CMY",
        "rgb": "Other",
        "rgbw": "RGBW",
        "quad": "Other",
    }
    expected_filter_letters = {
        "default": "rgb",
        "bayer": "rgb",
        "ycmy": "cym",
        "rgb": "rgb",
        "rgbw": "rgbw",
        "quad": "rgb",
    }

    for label, sensor in branches:
        rgb_image = np.asarray(sensor_get(sensor, "rgb"), dtype=float)
        assert tuple(np.asarray(sensor_get(sensor, "size"), dtype=int)) == expected_sizes[label]
        assert np.array_equal(np.asarray(sensor_get(sensor, "pattern"), dtype=int), expected_patterns[label])
        assert str(sensor_get(sensor, "cfaname")) == expected_cfa_names[label]
        assert str(sensor_get(sensor, "filter color letters")) == expected_filter_letters[label]
        assert rgb_image.shape == expected_sizes[label] + (3,)
        assert np.all(np.isfinite(rgb_image))
        assert float(np.mean(rgb_image)) > 0.0


def test_run_python_case_supports_sensor_cfa_script_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_cfa_script_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (391, 600)
    assert tuple(case.payload["oi_size"]) == (489, 750)
    assert list(case.payload["branch_labels"]) == ["default", "bayer", "ycmy", "rgb", "rgbw", "quad"]
    assert case.payload["branch_sizes"].shape == (6, 2)
    assert case.payload["branch_pattern_shapes"].shape == (6, 2)
    assert case.payload["branch_patterns_padded"].shape == (6, 4, 4)
    assert list(case.payload["branch_cfa_names"]) == ["Bayer RGB", "Bayer RGB", "Bayer CMY", "Other", "RGBW", "Other"]
    assert list(case.payload["branch_filter_letters"]) == ["rgb", "rgb", "cym", "rgb", "rgbw", "rgb"]
    assert case.payload["branch_mean_rgb_norm"].shape == (6, 3)
    assert case.payload["branch_center_rgb_norm"].shape == (6, 3)
    assert case.payload["branch_center_row_luma_norm"].shape == (6, 41)
    assert case.payload["branch_center_col_luma_norm"].shape == (6, 41)
    assert np.array_equal(case.payload["branch_sizes"][0], np.array([634, 974], dtype=int))
    assert np.array_equal(case.payload["branch_sizes"][1], np.array([398, 488], dtype=int))
    assert np.array_equal(case.payload["branch_sizes"][3], np.array([390, 489], dtype=int))
    assert np.array_equal(case.payload["branch_sizes"][5], np.array([400, 488], dtype=int))
    assert np.array_equal(case.payload["branch_pattern_shapes"][4], np.array([2, 2], dtype=int))
    assert np.array_equal(case.payload["branch_patterns_padded"][5], np.array([[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]], dtype=int))


def test_run_python_case_supports_sensor_snr_components_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_snr_components_small", asset_store=asset_store)

    assert case.payload["volts"].shape == (20,)
    assert case.payload["snr"].shape == (20,)
    assert case.payload["snr_shot"].shape == (20,)
    assert case.payload["snr_read"].shape == (20,)
    assert case.payload["snr_dsnu"].shape == (20,)
    assert case.payload["snr_prnu"].shape == (20,)
    assert np.all(np.diff(case.payload["volts"]) > 0.0)


def test_run_python_case_supports_sensor_counting_photons_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_counting_photons_small", asset_store=asset_store)

    assert case.payload["wave"].ndim == 1
    assert np.array_equal(case.payload["fnumbers"], np.arange(2.0, 17.0, dtype=float))
    assert case.payload["aperture_d"].shape == case.payload["fnumbers"].shape
    assert case.payload["spectral_irradiance"].shape == case.payload["wave"].shape
    assert case.payload["total_q"].shape == case.payload["fnumbers"].shape
    assert case.payload["snr"].shape == case.payload["fnumbers"].shape
    assert np.all(np.diff(case.payload["aperture_d"]) < 0.0)
    assert np.all(np.diff(case.payload["total_q"]) < 0.0)
    assert np.all(np.diff(case.payload["snr"]) < 0.0)


def test_run_python_case_supports_sensor_poisson_noise_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_poisson_noise_small", asset_store=asset_store)

    assert np.array_equal(case.payload["rect"], np.array([96, 156, 24, 28], dtype=int))
    assert float(case.payload["roi_mean_dv"]) > 0.0
    assert float(case.payload["roi_std_dv"]) > 0.0
    assert case.payload["roi_percentiles"].shape == (3,)
    assert np.all(np.diff(case.payload["roi_percentiles"]) > 0.0)
    assert float(case.payload["sqrt_mean_dv"]) == pytest.approx(np.sqrt(float(case.payload["roi_mean_dv"])))
    assert float(case.payload["sensor_mean_dv"]) > 0.0


def test_run_python_case_supports_sensor_exposure_color_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_exposure_color_small", asset_store=asset_store)

    assert float(case.payload["exposure_time"]) > 0.0
    assert case.payload["combined_transform"].shape == (3, 3)
    assert case.payload["mean_rgb"].shape == (3,)
    assert case.payload["white_patch_rgb"].shape == (3,)
    assert case.payload["result"].ndim == 3


def test_run_python_case_supports_sensor_dark_voltage_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_dark_voltage_small", asset_store=asset_store)

    assert case.payload["exp_times"].shape == (10,)
    assert case.payload["mean_volts"].shape == (10,)
    assert float(case.payload["dark_voltage_estimate"]) > 0.0
    assert float(case.payload["true_dark_voltage"]) > 0.0
    percent_error = abs(float(case.payload["dark_voltage_estimate"]) - float(case.payload["true_dark_voltage"])) / float(
        case.payload["true_dark_voltage"]
    )
    assert percent_error < 0.05


def test_run_python_case_supports_sensor_prnu_estimate_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_prnu_estimate_small", asset_store=asset_store)

    assert case.payload["exp_times"].shape == (33,)
    assert float(case.payload["prnu_estimate"]) > 0.0
    assert float(case.payload["slope_mean"]) == pytest.approx(1.0, rel=1e-6)
    assert float(case.payload["slope_std"]) > 0.0
    assert float(case.payload["offset_std"]) > 0.0
    assert case.payload["slope_sample"].shape == (8,)


def test_run_python_case_supports_sensor_dsnu_estimate_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_dsnu_estimate_small", asset_store=asset_store)

    assert float(case.payload["estimated_dsnu"]) > 0.0
    assert float(case.payload["mean_offset_mean"]) > 0.0
    assert float(case.payload["mean_offset_std"]) > 0.0
    assert case.payload["mean_offset_percentiles"].shape == (3,)


def test_sensor_size_resolution_script_workflow() -> None:
    pixel_size_um = np.arange(0.8, 3.0 + 1.0e-9, 0.2, dtype=float)
    pixel_size_m = pixel_size_um * 1.0e-6
    half_inch_size_m = np.asarray(sensor_formats("half inch"), dtype=float)
    quarter_inch_size_m = np.asarray(sensor_formats("quarter inch"), dtype=float)

    half_rows = half_inch_size_m[0] / pixel_size_m
    half_cols = half_inch_size_m[1] / pixel_size_m
    quarter_rows = quarter_inch_size_m[0] / pixel_size_m
    quarter_cols = quarter_inch_size_m[1] / pixel_size_m
    half_megapixels = np.asarray(ie_n_to_megapixel(half_rows * half_cols), dtype=float)
    quarter_megapixels = np.asarray(ie_n_to_megapixel(quarter_rows * quarter_cols), dtype=float)

    assert np.array_equal(np.asarray(sensor_formats("qcif"), dtype=float), np.array([144.0, 176.0], dtype=float))
    assert half_inch_size_m.shape == (2,)
    assert quarter_inch_size_m.shape == (2,)
    assert half_megapixels.shape == pixel_size_um.shape
    assert quarter_megapixels.shape == pixel_size_um.shape
    assert np.all(np.diff(half_megapixels) < 0.0)
    assert np.all(np.diff(quarter_megapixels) < 0.0)
    assert np.allclose(half_inch_size_m / quarter_inch_size_m, np.array([2.0, 2.0], dtype=float), atol=1e-12)
    assert np.allclose(half_rows / quarter_rows, np.full(pixel_size_um.shape, 2.0, dtype=float), atol=1e-12)
    assert np.allclose(half_cols / quarter_cols, np.full(pixel_size_um.shape, 2.0, dtype=float), atol=1e-12)


def test_run_python_case_supports_sensor_size_resolution_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_size_resolution_small", asset_store=asset_store)

    assert case.payload["pixel_size_um"].shape == (12,)
    assert np.array_equal(case.payload["half_inch_size_m"], np.array([0.0048, 0.0064], dtype=float))
    assert np.array_equal(case.payload["quarter_inch_size_m"], np.array([0.0024, 0.0032], dtype=float))
    assert case.payload["half_rows"].shape == (12,)
    assert case.payload["half_cols"].shape == (12,)
    assert case.payload["half_megapixels"].shape == (12,)
    assert case.payload["quarter_rows"].shape == (12,)
    assert case.payload["quarter_cols"].shape == (12,)
    assert case.payload["quarter_megapixels"].shape == (12,)
    assert np.all(np.diff(case.payload["half_megapixels"]) < 0.0)
    assert np.all(np.diff(case.payload["quarter_megapixels"]) < 0.0)


def test_sensor_cfa_point_spread_script_workflow(asset_store) -> None:
    scene = scene_create("point array", asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float)
    scene = scene_adjust_illuminant(scene, blackbody(wave, 8000.0), asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.0)

    oi = oi_create("diffraction limited", asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
    sensor = sensor_set(sensor, "auto exposure", True)

    rect = np.array([32, 24, 11, 11], dtype=int)
    crop_stack = []
    for ff in (2.0, 4.0, 8.0, 12.0):
        oi_ff = oi_set(oi, "optics fnumber", ff)
        oi_ff = oi_compute(oi_ff, scene)
        sensor_ff = sensor_compute(sensor, oi_ff)
        img = np.asarray(sensor_get(sensor_ff, "rgb"), dtype=float)
        crop = img[rect[1] - 1 : rect[1] + rect[3], rect[0] - 1 : rect[0] + rect[2], :]
        crop_stack.append(crop)

    crop_stack = np.stack(crop_stack, axis=3)
    x_um = np.arange(rect[2] + 1, dtype=float) * 1.4
    x_um = x_um - np.mean(x_um)

    assert crop_stack.shape == (12, 12, 3, 4)
    assert np.all(np.isfinite(crop_stack))
    assert np.allclose(x_um, -x_um[::-1], atol=1e-12)
    assert np.all(np.mean(crop_stack, axis=(0, 1, 3))[1] > 0.0)
    assert np.all(np.max(crop_stack, axis=(0, 1, 3)) > 0.0)


def test_run_python_case_supports_sensor_cfa_point_spread_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_cfa_point_spread_small", asset_store=asset_store)

    assert np.array_equal(case.payload["ff_numbers"], np.array([2.0, 4.0, 8.0, 12.0], dtype=float))
    assert np.array_equal(case.payload["pixel_size_um"], np.array([1.4, 1.4], dtype=float))
    assert np.array_equal(case.payload["rect"], np.array([32, 24, 11, 11], dtype=int))
    assert case.payload["x_um"].shape == (12,)
    assert case.payload["crop_mean_rgb"].shape == (4, 3)
    assert case.payload["crop_peak_rgb"].shape == (4, 3)
    assert case.payload["green_row_width_30_um"].shape == (4,)
    assert case.payload["green_row_width_50_um"].shape == (4,)
    assert case.payload["green_row_width_90_um"].shape == (4,)
    assert case.payload["red_center_cols_norm"].shape == (4, 12)
    assert np.all(np.isfinite(case.payload["crop_mean_rgb"]))


def test_run_python_case_supports_sensor_spatial_resolution_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_spatial_resolution_small", asset_store=asset_store)

    assert case.payload["coarse_pixPos"].ndim == 1
    assert case.payload["coarse_pixData"].ndim == 1
    assert case.payload["fine_pixPos"].ndim == 1
    assert case.payload["fine_pixData"].ndim == 1
    assert case.payload["oi_pos"].ndim == 1
    assert case.payload["oi_data"].ndim == 1
    assert case.payload["coarse_pixPos"].size == case.payload["coarse_pixData"].size
    assert case.payload["fine_pixPos"].size == case.payload["fine_pixData"].size
    coarse_spacing = float(np.mean(np.diff(case.payload["coarse_pixPos"])))
    fine_spacing = float(np.mean(np.diff(case.payload["fine_pixPos"])))
    assert abs(fine_spacing) < abs(coarse_spacing)
    assert case.payload["oi_pos"].size == case.payload["oi_data"].size


def test_run_python_case_supports_sensor_fpn_noise_modes_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_fpn_noise_modes_small", asset_store=asset_store)

    assert case.payload["pixPos"].ndim == 1
    assert case.payload["pixColor"].ndim == 1
    assert case.payload["noise0_pixData"].ndim == 1
    assert case.payload["pixPos"].size == case.payload["noise0_pixData"].size
    assert case.payload["pixColor"].size > 0
    for key in ("noiseM2_stats", "noise1_stats", "noise2_stats"):
        values = np.asarray(case.payload[key], dtype=float)
        assert values.shape == (4,)
        assert np.all(np.isfinite(values))


def test_run_python_case_supports_sensor_description_fpn_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_description_fpn_small", asset_store=asset_store)

    assert case.payload["title"] == "ISET Parameter Table for a Sensor"
    assert case.payload["handle_title"] == case.payload["title"]
    assert int(case.payload["row_count"]) > 0
    assert int(case.payload["col_count"]) == 3
    assert case.payload["read_noise_volts"] == "0.100"
    assert case.payload["analog_gain"] == "1"
    assert case.payload["exposure_time"] == "0"


def test_run_python_case_supports_sensor_dng_read_crop_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_dng_read_crop_small", asset_store=asset_store)

    assert tuple(case.payload["size"]) == (258, 258)
    assert case.payload["pattern"].shape == (2, 2)
    assert case.payload["digital_values"].shape == (258, 258)
    assert case.payload["result"].shape == (258, 258, 3)
    assert float(case.payload["black_level"]) > 0.0
    assert float(case.payload["exp_time"]) > 0.0
    assert float(case.payload["iso_speed"]) > 0.0


def test_sensor_exposure_color_tutorial_flow(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    scene = scene_create(asset_store=asset_store)
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set_size_to_fov(sensor, scene_get(scene, "fov"), oi)
    filters = np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float)
    filters[:, 0] = filters[:, 0] * 0.2
    filters[:, 2] = filters[:, 2] * 0.5
    sensor = sensor_set(sensor, "filter transmissivities", filters)
    sensor = sensor_set(sensor, "auto exposure", "on")

    sensor = sensor_compute(sensor, oi, seed=0)
    exposure_time = float(sensor_get(sensor, "exposure time"))
    assert exposure_time > 0.0

    ip = ip_create(asset_store=asset_store)
    ip = ip_compute(ip, sensor)
    combined_transform = np.asarray(ip_get(ip, "combined transform"), dtype=float)
    assert combined_transform.shape == (3, 3)

    ip = ip_set(ip, "transform method", "current")
    sensor = sensor_set(sensor, "auto exposure", "off")
    sensor = sensor_set(sensor, "exposure time", 3.0 * exposure_time)
    sensor = sensor_compute(sensor, oi, seed=0)
    ip = ip_compute(ip, sensor)

    rendered = np.asarray(ip_get(ip, "result"), dtype=float)
    assert rendered.ndim == 3 and rendered.shape[2] == 3
    assert float(np.max(rendered)) <= 1.0 + 1e-8


def test_ie_read_spectra_supports_sensor_estimation_sources(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)

    macbeth_chart = ie_read_spectra("macbethChart", wave, asset_store=asset_store)
    illuminant_d65 = ie_read_spectra("D65.mat", wave, asset_store=asset_store)
    sensors = ie_read_spectra("cMatch/camera", wave, asset_store=asset_store)
    cones = ie_read_spectra("SmithPokornyCones", wave, asset_store=asset_store)

    assert macbeth_chart.shape == (wave.size, 24)
    assert illuminant_d65.shape == (wave.size, 1)
    assert sensors.shape == (wave.size, 3)
    assert cones.shape == (wave.size, 3)


def test_run_python_case_supports_sensor_estimation_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_estimation_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["green_reflectance"].shape == (31,)
    assert case.payload["illuminant_d65"].shape == (31,)
    assert case.payload["sensors"].shape == (31, 3)
    assert case.payload["cones"].shape == (31, 3)
    assert case.payload["rgb_responses_gray"].shape == (3, 6)
    assert case.payload["estimate_full"].shape == (31, 3)
    assert case.payload["rgb_pred_full"].shape == (3, 24)
    assert case.payload["estimate_sparse"].shape == (31, 3)
    assert case.payload["rgb_pred_sparse"].shape == (3, 24)


def test_sensor_macbeth_daylight_estimate_script_contract(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    reflectance = ie_read_spectra("macbethChart", wave, asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    sensor_filters = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    day_basis_quanta = energy_to_quanta(ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=asset_store), wave)

    true_weights = np.array([1.0, 0.0, 0.0], dtype=float)
    illuminant_photons = day_basis_quanta @ true_weights
    camera_data = sensor_filters.T @ (illuminant_photons[:, None] * reflectance)

    x1 = sensor_filters.T @ (day_basis_quanta[:, [0]] * reflectance)
    x2 = sensor_filters.T @ (day_basis_quanta[:, [1]] * reflectance)
    x3 = sensor_filters.T @ (day_basis_quanta[:, [2]] * reflectance)
    design_matrix = np.column_stack(
        [
            x1.reshape(-1, order="F"),
            x2.reshape(-1, order="F"),
            x3.reshape(-1, order="F"),
        ]
    )
    camera_stacked = camera_data.reshape(-1, order="F")
    estimated_weights = np.linalg.solve(design_matrix.T @ design_matrix, design_matrix.T @ camera_stacked)
    estimated_weights = estimated_weights / estimated_weights[0]

    assert reflectance.shape == (31, 24)
    assert sensor_filters.shape == (31, 3)
    assert day_basis_quanta.shape == (31, 3)
    assert camera_data.shape == (3, 24)
    assert design_matrix.shape == (72, 3)
    assert np.allclose(estimated_weights, true_weights, atol=1e-10, rtol=1e-10)


def test_sensor_macbeth_daylight_estimate_wrapper_matches_linear_solution(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    reflectance = ie_read_spectra("macbethChart", wave, asset_store=asset_store)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", wave)
    sensor = sensor_set(sensor, "rows", 82)
    sensor = sensor_set(sensor, "cols", 122)

    sensor_filters = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    day_basis_quanta = energy_to_quanta(
        ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=asset_store),
        wave,
    )
    true_weights = np.array([1.0, 0.0, 0.0], dtype=float)
    illuminant_photons = day_basis_quanta @ true_weights
    camera_data = sensor_filters.T @ (illuminant_photons[:, None] * reflectance)

    corner_points = np.array(
        [
            [1.0, 82.0],
            [121.0, 82.0],
            [121.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    rects, _, _ = sensor_module._chart_rectangles(corner_points, 4, 6, 0.4)
    volts = np.zeros((82, 122, 3), dtype=float)
    conversion_gain = float(sensor.fields["pixel"]["conversion_gain_v_per_electron"])
    for index, rect in enumerate(np.asarray(rects, dtype=int)):
        col = int(rect[0]) - 1
        row = int(rect[1]) - 1
        width = int(rect[2]) + 1
        height = int(rect[3]) + 1
        volts[row : row + height, col : col + width, :] = (
            camera_data[:, index].reshape(1, 1, 3) * conversion_gain
        )
    sensor.data["volts"] = volts

    ill_photons, cp, wgts, c_stacked, design_matrix = sensorMacbethDaylightEstimate(
        sensor,
        "corner points",
        corner_points,
        asset_store=asset_store,
    )

    x1 = sensor_filters.T @ (day_basis_quanta[:, [0]] * reflectance)
    x2 = sensor_filters.T @ (day_basis_quanta[:, [1]] * reflectance)
    x3 = sensor_filters.T @ (day_basis_quanta[:, [2]] * reflectance)
    expected_matrix = np.column_stack(
        [
            x1.reshape(-1, order="F"),
            x2.reshape(-1, order="F"),
            x3.reshape(-1, order="F"),
        ]
    )
    expected_stacked = camera_data.reshape(-1, order="F")
    expected_weights = np.linalg.solve(expected_matrix.T @ expected_matrix, expected_matrix.T @ expected_stacked)
    expected_weights = expected_weights / expected_weights[0]

    assert np.allclose(cp, corner_points)
    assert np.allclose(c_stacked, expected_stacked)
    assert np.allclose(design_matrix, expected_matrix)
    assert np.allclose(wgts, expected_weights, atol=1e-10, rtol=1e-10)
    assert np.allclose(ill_photons, day_basis_quanta @ expected_weights, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(sensor_get(sensor, "chart rects"), dtype=float), np.asarray(rects, dtype=float))


def test_exposure_value_and_photometric_exposure_match_legacy_formulas(asset_store) -> None:
    oi = oi_create("diffraction limited", asset_store=asset_store)
    oi = oi_set(oi, "fnumber", 4.0)
    oi = oi_set(oi, "meanilluminance", 80.0)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "exposure time", 0.25)

    assert np.isclose(exposureValue(oi, sensor), np.log2((4.0**2) / 0.25))
    assert np.isclose(photometricExposure(oi, sensor), 20.0)


def test_chart_patch_compare_matches_legacy_template_layout() -> None:
    patch_height = 2
    patch_width = 2
    rows = 4
    cols = 6
    img_l = np.zeros((rows * patch_height, cols * patch_width, 3), dtype=float)
    img_s = np.zeros_like(img_l)
    rects = []

    count = 0
    for col in range(cols):
        for row in range(rows):
            color = np.array([count / 24.0, (count + 1) / 25.0, (count + 2) / 26.0], dtype=float)
            img_l[row * patch_height : (row + 1) * patch_height, col * patch_width : (col + 1) * patch_width, :] = color
            img_s[row * patch_height : (row + 1) * patch_height, col * patch_width : (col + 1) * patch_width, :] = color
            rects.append([col * patch_width, row * patch_height, patch_width - 1, patch_height - 1])
            count += 1

    img_s[0:patch_height, 0:patch_width, :] = np.array([0.9, 0.2, 0.1], dtype=float)

    chart_template, delta_map = chartPatchCompare(img_l, img_s, np.asarray(rects, dtype=float), np.asarray(rects, dtype=float), patch_size=4)

    assert chart_template.shape == (16, 24, 3)
    assert delta_map.shape == (16, 24)
    assert np.allclose(chart_template[0:2, 0:2, :], img_l[0, 0, :].reshape(1, 1, 3))
    assert np.allclose(chart_template[2:4, 2:4, :], img_s[0, 0, :].reshape(1, 1, 3))
    assert np.allclose(delta_map[4:8, 0:4], 0.0)
    assert np.allclose(delta_map[0:4, 0:4], delta_map[0, 0])
    assert float(delta_map[0, 0]) > 0.0


def _build_metrics_test_ip(name: str, result: np.ndarray, xyz: np.ndarray, white_point: np.ndarray):
    ip = ip_create()
    ip.name = name
    ip = ip_set(ip, "result", np.asarray(result, dtype=float))
    ip = ip_set(ip, "datawhitepoint", np.asarray(white_point, dtype=float))
    ip.data["xyz"] = np.asarray(xyz, dtype=float).copy()
    return ip


def test_metrics_compute_and_masked_error_helpers(tmp_path) -> None:
    white_point = np.array([0.95047, 1.0, 1.08883], dtype=float)
    white_point_2 = np.array([0.96422, 1.0, 0.82521], dtype=float)
    result1 = np.array(
        [
            [[0.10, 0.20, 0.30], [0.20, 0.30, 0.40]],
            [[0.30, 0.40, 0.50], [0.40, 0.50, 0.60]],
        ],
        dtype=float,
    )
    result2 = result1 + 0.05
    xyz1 = np.array(
        [
            [[0.20, 0.30, 0.10], [0.25, 0.35, 0.15]],
            [[0.30, 0.40, 0.20], [0.35, 0.45, 0.25]],
        ],
        dtype=float,
    )
    xyz2 = xyz1 + 0.02

    ip1 = _build_metrics_test_ip("first", result1, xyz1, white_point)
    ip2 = _build_metrics_test_ip("second", result2, xyz2, white_point_2)

    d_e_image, d_e_value = metricsCompute(ip1, ip2, "CIELAB (dE)")
    expected_d_e = delta_e_ab(xyz1, xyz2, (white_point, white_point_2))
    np.testing.assert_allclose(d_e_image, expected_d_e, rtol=1e-10, atol=1e-12)
    assert d_e_value is None

    d_uv_image, d_uv_value = metricsCompute(ip1, ip2, "CIELUV (dE)")
    expected_d_uv = deltaEuv(xyz1, xyz2, white_point)
    np.testing.assert_allclose(d_uv_image, expected_d_uv, rtol=1e-10, atol=1e-12)
    assert d_uv_value is None

    mse_image, mse_value = metricsCompute(ip1, ip2, "MSE")
    expected_mse = np.sum(np.square(result1 - result2), axis=-1)
    np.testing.assert_allclose(mse_image, expected_mse, rtol=1e-10, atol=1e-12)
    assert mse_value == pytest.approx(float(np.mean(expected_mse)), rel=1e-12, abs=1e-12)

    rmse_image, rmse_value = metricsCompute(ip1, ip2, "RMSE")
    expected_rmse = np.sqrt(np.sum(np.square(result1 - result2), axis=-1))
    np.testing.assert_allclose(rmse_image, expected_rmse, rtol=1e-10, atol=1e-12)
    assert rmse_value == pytest.approx(float(np.mean(expected_rmse)), rel=1e-12, abs=1e-12)

    psnr_image, psnr_value = metricsCompute(ip1, ip2, "PSNR")
    assert psnr_image is None
    assert psnr_value == pytest.approx(
        metrics_module.peak_signal_to_noise_ratio(result1, result2),
        rel=1e-12,
        abs=1e-12,
    )

    alpha = metricsMaskedError(result1, result2 - result1)
    expected_alpha = float(np.linalg.lstsq(result1.reshape(-1, 1), (result2 - result1).reshape(-1), rcond=None)[0][0])
    assert alpha == pytest.approx(expected_alpha, rel=1e-12, abs=1e-12)

    handles = {
        "vci1": ip1,
        "vci2": ip2,
        "image1_name": "first",
        "image2_name": "second",
        "metric_names": ["CIELAB (dE)", "CIELUV (dE)", "MSE", "RMSE", "PSNR"],
        "current_metric": "CIELAB (dE)",
        "metric_image": d_e_image,
        "gamma": 2.2,
    }

    selected1, selected2 = metricsGetVciPair(handles)
    assert selected1 is ip1
    assert selected2 is ip2
    assert metricsGet(handles, "image1name") == "first"
    assert metricsGet(handles, "currentmetric") == "CIELAB (dE)"
    assert metricsGet(handles, "metricnames") == ["CIELAB (dE)", "CIELUV (dE)", "MSE", "RMSE", "PSNR"]
    np.testing.assert_allclose(metricsGet(handles, "metricImageData"), d_e_image / 30.0, rtol=1e-10, atol=1e-12)

    updated = metricsSet(handles, "metricdata", mse_image)
    np.testing.assert_allclose(metricsGet(updated, "metricdata"), mse_image, rtol=1e-10, atol=1e-12)

    description = metricsDescription(handles)
    assert "Image 1 (first):" in description
    assert "Image 2 (second):" in description
    assert "White (X,Y,Z):" in description

    shown_images = metricsShowImage(handles)
    np.testing.assert_allclose(shown_images["image1"], result1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(shown_images["image2"], result2, rtol=1e-10, atol=1e-12)
    assert shown_images["gamma"] == pytest.approx(2.2, rel=0.0, abs=0.0)

    shown_metric = metricsShowMetric(handles)
    assert shown_metric["metric"] == "CIELAB (dE)"
    np.testing.assert_allclose(shown_metric["image"], d_e_image / 30.0, rtol=1e-10, atol=1e-12)

    refreshed = metricsRefresh({**handles, "images": {"first": ip1, "second": ip2}})
    assert refreshed["popImageList1"] == ["first", "second"]
    assert refreshed["popImageList2"] == ["first", "second"]
    assert "description_text" in refreshed
    np.testing.assert_allclose(refreshed["shown_images"]["image1"], result1, rtol=1e-10, atol=1e-12)

    computed = metricsKeyPress({**handles, "current_metric": "MSE"}, 16)
    np.testing.assert_allclose(computed["metric_image"], mse_image, rtol=1e-10, atol=1e-12)
    assert computed["metric_value"] == pytest.approx(float(np.mean(expected_mse)), rel=1e-12, abs=1e-12)
    assert computed["last_action"] == "compute"

    refreshed_via_key = metricsKeyPress({**handles, "images": {"first": ip1, "second": ip2}}, 18)
    assert refreshed_via_key["last_action"] == "refresh"
    assert refreshed_via_key["popImageList1"] == ["first", "second"]

    help_state = metricsKeyPress(handles, 8)
    assert help_state["last_action"] == "help"

    closed = metricsClose({"metricsWindow": object(), "figure1": object(), **handles})
    assert closed["closed"] is True
    assert "metricsWindow" not in closed
    assert "figure1" not in closed

    image_path, metric_name = metricsSaveImage(handles, tmp_path / "metric_output")
    data_path, saved_metric_name = metricsSaveData(handles, tmp_path / "metric_payload")
    assert Path(image_path).suffix == ".tiff"
    assert Path(data_path).suffix == ".mat"
    assert Path(image_path).exists()
    assert Path(data_path).exists()
    assert metric_name == "CIELAB (dE)"
    assert saved_metric_name == "CIELAB (dE)"


def test_metrics_roi_compare_and_sqri_helpers() -> None:
    white_point = np.array([0.95047, 1.0, 1.08883], dtype=float)
    white_point_2 = np.array([0.96422, 1.0, 0.82521], dtype=float)
    result1 = np.array(
        [
            [[0.10, 0.20, 0.30], [0.20, 0.30, 0.40]],
            [[0.30, 0.40, 0.50], [0.40, 0.50, 0.60]],
        ],
        dtype=float,
    )
    result2 = result1 + 0.05
    xyz1 = np.array(
        [
            [[0.20, 0.30, 0.10], [0.25, 0.35, 0.15]],
            [[0.30, 0.40, 0.20], [0.35, 0.45, 0.25]],
        ],
        dtype=float,
    )
    xyz2 = xyz1 + 0.02
    ip1 = _build_metrics_test_ip("first", result1, xyz1, white_point)
    ip2 = _build_metrics_test_ip("second", result2, xyz2, white_point_2)
    handles = {
        "vci1": ip1,
        "vci2": ip2,
        "img1_rect": [1, 1, 0, 1],
        "img2_rect": [1, 1, 1, 0],
        "metric_rect": [2, 1, 0, 1],
    }

    roi_locs = metricsROI(handles, "img1")
    expected_locs = np.array([[1, 1], [2, 1]], dtype=int)
    np.testing.assert_array_equal(roi_locs, expected_locs)
    np.testing.assert_array_equal(metricsROI(handles, "img2"), np.array([[1, 1], [1, 2]], dtype=int))
    np.testing.assert_array_equal(metricsROI(handles, "metricImage"), np.array([[1, 2], [2, 2]], dtype=int))

    d_e_roi, returned_locs = metricsCompareROI(handles)
    expected_d_e = delta_e_ab(
        ip_get(ip1, "roixyz", expected_locs),
        ip_get(ip2, "roixyz", expected_locs),
        (white_point, white_point_2),
    )
    np.testing.assert_array_equal(returned_locs, expected_locs)
    np.testing.assert_allclose(d_e_roi, expected_d_e, rtol=1e-10, atol=1e-12)

    sf = np.logspace(-1.5, 1.5, 30)
    d_mtf = np.ones_like(sf)
    luminance = 340.0 / np.pi
    sqri, h_csf = ieSQRI(sf, d_mtf, luminance, "width", 6.5)

    a = 540.0 * (1.0 + (0.7 / luminance)) ** (-0.2) / (1.0 + (12.0 / (6.5 * (1.0 + (sf / 3.0)) ** 2)))
    b = 0.3 * (1.0 + (100.0 / luminance)) ** 0.15
    c = 0.06
    expected_h_csf = (a * sf) * np.exp(-b * sf) * np.sqrt(1.0 + c * np.exp(b * sf))
    du = np.diff(sf)
    u = sf[1:]
    dm = 0.5 * (d_mtf[:-1] + d_mtf[1:])
    dh = 0.5 * (expected_h_csf[:-1] + expected_h_csf[1:])
    expected_sqri = (1.0 / np.log(2.0)) * np.sum(np.sqrt(dm * dh) * (du / u))

    np.testing.assert_allclose(h_csf, expected_h_csf, rtol=1e-10, atol=1e-12)
    assert sqri == pytest.approx(float(expected_sqri), rel=1e-12, abs=1e-12)


def test_metrics_camera_gateway_matches_existing_wrappers(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    camera = camera_set(camera, "sensor auto exposure", True)

    mtf = metricsCamera(camera, "slantededge", asset_store=asset_store)
    assert np.asarray(mtf.freq, dtype=float).ndim == 1
    assert np.asarray(mtf.mtf, dtype=float).ndim == 2
    assert float(mtf.mtf50) > 0.0

    color_metric = metricsCamera(camera, "mcccolor", asset_store=asset_store)
    assert np.asarray(color_metric["deltaE"], dtype=float).shape == (24,)
    assert "vci" not in color_metric

    full_reference = metricsCamera(camera, "fullreference", asset_store=asset_store)
    expected_full_reference = cameraFullReference(camera, asset_store=asset_store)
    assert full_reference.sceneNames == expected_full_reference.sceneNames
    np.testing.assert_allclose(full_reference.meanLuminances, expected_full_reference.meanLuminances, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(full_reference.scielab, expected_full_reference.scielab, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(full_reference.ssim, expected_full_reference.ssim, rtol=1e-10, atol=1e-12)

    moire_metric = metricsCamera(camera, "moire", asset_store=asset_store)
    expected_moire = cameraMoire(camera, asset_store=asset_store)
    np.testing.assert_allclose(moire_metric.data_abIdeal_mean, expected_moire.data_abIdeal_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(moire_metric.data_abim_mean, expected_moire.data_abim_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(moire_metric.cpd_mean, expected_moire.cpd_mean, rtol=1e-10, atol=1e-12)
    assert np.all(np.diff(np.asarray(moire_metric.cpd_mean, dtype=float)) >= 0.0)

    vsnr_metric = metricsCamera(camera, "visible snr", asset_store=asset_store)
    expected_vsnr = camera_vsnr(camera, asset_store=asset_store)
    np.testing.assert_allclose(vsnr_metric.vSNR, expected_vsnr.vSNR, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vsnr_metric.lightLevels, expected_vsnr.lightLevels, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vsnr_metric.eTime, expected_vsnr.eTime, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(vsnr_metric.rect, expected_vsnr.rect)

    computed_metric = camera_get(camera, "metric", "mcccolor")
    np.testing.assert_allclose(np.asarray(computed_metric["deltaE"], dtype=float), np.asarray(color_metric["deltaE"], dtype=float), rtol=1e-10, atol=1e-12)
    assert "vci" not in computed_metric

    computed_vsnr_metric = camera_get(camera, "metric", "visible snr")
    np.testing.assert_allclose(computed_vsnr_metric.vSNR, expected_vsnr.vSNR, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(computed_vsnr_metric.lightLevels, expected_vsnr.lightLevels, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(computed_vsnr_metric.eTime, expected_vsnr.eTime, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(computed_vsnr_metric.rect, expected_vsnr.rect)

    camera = camera_set(camera, "metric", expected_vsnr, "visible snr")
    stored_metrics = camera_get(camera, "metrics")
    assert "vsnr" in stored_metrics
    assert "visiblesnr" not in stored_metrics
    stored_vsnr_short = camera_get(camera, "metric", "vsnr")
    stored_vsnr_alias = camera_get(camera, "metric", "visible snr")
    np.testing.assert_allclose(stored_vsnr_short.vSNR, expected_vsnr.vSNR, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(stored_vsnr_alias.vSNR, expected_vsnr.vSNR, rtol=1e-10, atol=1e-12)

    camera = camera_set(camera, "metric", color_metric, "mcccolor")
    stored_metrics = camera_get(camera, "metrics")
    assert "mcccolor" in stored_metrics
    stored_metric = camera_get(camera, "metric", "mcccolor")
    np.testing.assert_allclose(np.asarray(stored_metric["deltaE"], dtype=float), np.asarray(color_metric["deltaE"], dtype=float), rtol=1e-10, atol=1e-12)


def test_scene_cct_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 721.0, 5.0, dtype=float)
    single_temperatures = np.array([3500.0, 6500.0, 8500.0], dtype=float)
    estimated_single = np.array(
        [
            float(
                spd_to_cct(
                    wave,
                    np.asarray(blackbody(wave, temperature_k, kind="energy"), dtype=float).reshape(-1),
                    asset_store=asset_store,
                )
            )
            for temperature_k in single_temperatures
        ],
        dtype=float,
    )
    multi_temperatures = np.arange(4500.0, 8501.0, 1000.0, dtype=float)
    spd_multi = np.asarray(blackbody(wave, multi_temperatures, kind="energy"), dtype=float)
    estimated_multi = np.asarray(spd_to_cct(wave, spd_multi, asset_store=asset_store), dtype=float).reshape(-1)

    assert spd_multi.shape == (wave.size, multi_temperatures.size)
    assert np.allclose(estimated_single, single_temperatures, atol=25.0, rtol=5e-3)
    assert np.allclose(estimated_multi, multi_temperatures, atol=25.0, rtol=5e-3)


def test_scene_daylight_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 771.0, 1.0, dtype=float)
    cct = np.arange(4000.0, 10001.0, 1000.0, dtype=float)
    photons = np.asarray(daylight(wave, cct, "photons", asset_store=asset_store), dtype=float)
    lum_photons = np.asarray(luminance_from_photons(photons.T, wave, asset_store=asset_store), dtype=float).reshape(-1)
    photons_scaled = photons * (100.0 / np.maximum(lum_photons, 1e-12)).reshape(1, -1)

    energy = np.asarray(daylight(wave, cct, "energy", asset_store=asset_store), dtype=float)
    lum_energy = np.asarray(luminance_from_energy(energy.T, wave, asset_store=asset_store), dtype=float).reshape(-1)
    energy_scaled = energy * (100.0 / np.maximum(lum_energy, 1e-12)).reshape(1, -1)

    day_basis = np.asarray(ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=asset_store), dtype=float)
    basis_weights = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]], dtype=float)
    basis_examples = day_basis @ basis_weights

    assert photons.shape == (wave.size, cct.size)
    assert energy.shape == (wave.size, cct.size)
    assert np.isclose(lum_photons[0], 100.0, atol=1e-6, rtol=1e-6)
    assert np.isclose(lum_energy[0], 100.0, atol=1e-6, rtol=1e-6)
    assert np.allclose(
        np.asarray(luminance_from_photons(photons_scaled.T, wave, asset_store=asset_store), dtype=float).reshape(-1),
        np.full(cct.size, 100.0, dtype=float),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(
        np.asarray(luminance_from_energy(energy_scaled.T, wave, asset_store=asset_store), dtype=float).reshape(-1),
        np.full(cct.size, 100.0, dtype=float),
        atol=1e-6,
        rtol=1e-6,
    )
    assert day_basis.shape == (wave.size, 3)
    assert basis_examples.shape == (wave.size, 3)


def test_color_temperature_helper_wrappers_match_existing_surfaces(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    cct = np.array([4000.0, 6500.0, 9000.0], dtype=float)

    assert np.allclose(
        np.asarray(cct2sun(wave, cct, "energy", asset_store=asset_store), dtype=float),
        np.asarray(daylight(wave, cct, "energy", asset_store=asset_store), dtype=float),
    )

    srgb = np.asarray(ieCTemp2SRGB(3000.0, wave=wave, asset_store=asset_store), dtype=float).reshape(-1)
    energy = np.asarray(blackbody(wave, 3000.0, kind="energy"), dtype=float).reshape(1, -1)
    xyz = np.asarray(xyz_from_energy(energy, wave, asset_store=asset_store), dtype=float).reshape(1, 1, 3)
    expected = np.asarray(xyz_to_srgb(xyz), dtype=float).reshape(-1)
    assert np.allclose(srgb, expected, atol=1e-10, rtol=1e-10)


def test_basic_color_geometry_and_gamma_helpers_match_legacy_contract() -> None:
    x, y = ieCirclePoints(2.0 * np.pi / 4.0)
    expected_circle = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    assert np.allclose(np.column_stack([x, y]), expected_circle, atol=1e-12, rtol=0.0)

    gamma = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.2],
            [0.3, 0.7],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    def _manual_mk_inv_gamma_table(g_table: np.ndarray, entries: int) -> np.ndarray:
        result = np.zeros((entries, g_table.shape[1]), dtype=float)
        target = np.arange(entries, dtype=float) / max(entries - 1, 1)
        for column in range(g_table.shape[1]):
            this_table = np.asarray(g_table[:, column], dtype=float).reshape(-1)
            if np.any(np.diff(this_table) <= 0.0):
                this_table = np.sort(this_table)
                positive_locs = np.where(np.diff(this_table) > 0.0)[0] + 1
                pos_locs = np.concatenate(([0], positive_locs)).astype(float)
                monotone_table = this_table[pos_locs.astype(int)]
            else:
                monotone_table = this_table
                pos_locs = np.arange(this_table.size, dtype=float)
            result[:, column] = np.interp(target, monotone_table, pos_locs)
        return result

    expected = _manual_mk_inv_gamma_table(gamma, 9)
    assert np.allclose(mkInvGammaTable(gamma, 9), expected, atol=1e-12, rtol=0.0)


def test_additional_color_wrappers_match_legacy_contract(asset_store) -> None:
    raw = {"spectrum": {"wave": np.array([1.0])}, "wave": np.array([1.0])}
    updated = initDefaultSpectrum(raw, "custom", [450.0, 550.0, 650.0])
    assert np.array_equal(np.asarray(updated["spectrum"]["wave"], dtype=float), np.array([450.0, 550.0, 650.0], dtype=float))
    assert np.array_equal(np.asarray(updated["wave"], dtype=float), np.array([450.0, 550.0, 650.0], dtype=float))

    monochrome_scene = initDefaultSpectrum(scene_create("default", asset_store=asset_store), "monochrome")
    assert np.array_equal(np.asarray(scene_get(monochrome_scene, "wave"), dtype=float), np.array([550.0], dtype=float))

    xy_data = np.array(
        [
            [68.5, 533.0],
            [39.5, 618.0],
            [79.0, 655.0],
            [58.0, 546.0],
            [35.0, 635.0],
            [62.0, 720.0],
            [92.0, 529.0],
        ],
        dtype=float,
    )
    e_vec, handle, payload = ieCovEllipsoid(xy_data, 2.0)
    center = np.mean(xy_data, axis=0)
    cov_inv = np.linalg.pinv(np.cov(xy_data, rowvar=False))
    ellipse_radius = np.einsum("ij,jk,ik->i", e_vec - center, cov_inv, e_vec - center)
    assert handle is None
    assert np.allclose(ellipse_radius, np.full(e_vec.shape[0], 4.0), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(payload["covariance"], dtype=float), np.cov(xy_data, rowvar=False))

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    spectrum_e = np.asarray(blackbody(wave, 6500.0, kind="energy"), dtype=float).reshape(-1)
    spectra_s, xyz, xyz0, s_basis = ieSpectraSphere(
        wave,
        spectrum_e,
        4,
        "cieDaylightBasis.mat",
        0.05,
        asset_store=asset_store,
    )
    assert spectra_s.shape == (wave.size, 25)
    assert xyz.shape == (25, 3)
    assert xyz0.shape == (3,)
    assert s_basis.shape[0] == wave.size
    assert np.allclose(xyz, xyz_from_energy(spectra_s.T, wave, asset_store=asset_store), atol=1e-8, rtol=1e-8)
    assert np.allclose(
        xyz0,
        xyz_from_energy(spectrum_e.reshape(1, -1), wave, asset_store=asset_store).reshape(-1),
        atol=1e-8,
        rtol=1e-8,
    )


def test_color_helper_wrappers_match_closed_form_math(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    photons = np.asarray(daylight(wave, 6500.0, "photons", asset_store=asset_store), dtype=float).reshape(-1)
    energy = np.asarray(quanta_to_energy(photons, wave), dtype=float).reshape(-1)

    xyz_from_wrapper = np.asarray(ie_xyz_from_photons(photons, wave, asset_store=asset_store), dtype=float).reshape(-1)
    xyz_expected = np.asarray(xyz_from_energy(energy, wave, asset_store=asset_store), dtype=float).reshape(-1)
    assert np.allclose(xyz_from_wrapper, xyz_expected, atol=1e-10, rtol=1e-10)

    radiance_energy, radiance_wave = ie_luminance_to_radiance(125.0, 520.0, sd=12.0, wave=wave, asset_store=asset_store)
    assert np.array_equal(np.asarray(radiance_wave, dtype=float), wave)
    assert np.isclose(float(luminance_from_energy(radiance_energy, radiance_wave, asset_store=asset_store)), 125.0, atol=1e-8, rtol=1e-8)
    assert int(np.argmax(np.asarray(radiance_energy, dtype=float))) == int(np.argmin(np.abs(wave - 520.0)))

    y_values = np.array([0.5, 10.0, 100.0], dtype=float)
    expected_lstar = np.array([903.3 * (0.5 / 100.0), 116.0 * (10.0 / 100.0) ** (1.0 / 3.0) - 16.0, 100.0], dtype=float)
    assert np.allclose(y_to_lstar(y_values, 100.0), expected_lstar, atol=1e-8, rtol=1e-8)

    srgb = np.array([[0.0, 0.04045, 0.5], [1.0, 0.25, 0.75]], dtype=float)
    linear = np.asarray(srgb_to_lrgb(srgb), dtype=float)
    assert np.allclose(lrgb_to_srgb(linear), srgb, atol=5e-8, rtol=1e-8)

    responsivity_energy = np.column_stack(
        [
            np.linspace(0.2, 0.8, wave.size, dtype=float),
            np.linspace(0.6, 0.1, wave.size, dtype=float),
            np.linspace(0.1, 0.5, wave.size, dtype=float),
        ],
    )
    responsivity_quanta, scale_e2q = ie_responsivity_convert(responsivity_energy, wave, "e2q")
    assert responsivity_quanta.shape == responsivity_energy.shape
    assert scale_e2q.shape == (wave.size,)
    assert np.isclose(np.max(responsivity_quanta), np.max(responsivity_energy), atol=1e-12, rtol=0.0)
    expected_quanta = scale_e2q[:, np.newaxis] * responsivity_energy
    expected_quanta *= np.max(responsivity_energy) / np.max(expected_quanta)
    assert np.allclose(responsivity_quanta, expected_quanta, atol=1e-12, rtol=1e-12)

    rods = np.asarray(ie_read_spectra("rods.mat", wave, asset_store=asset_store), dtype=float).reshape(-1)
    scotopic = np.asarray(ie_scotopic_luminance_from_energy(energy, wave, asset_store=asset_store), dtype=float).reshape(-1)
    expected_scotopic = np.array([1745.0 * np.sum(energy * rods * np.mean(np.diff(wave)))], dtype=float)
    assert np.allclose(scotopic, expected_scotopic, atol=1e-8, rtol=1e-8)


def test_scene_illuminant_script_workflow(asset_store) -> None:
    default_blackbody = illuminant_create("blackbody", asset_store=asset_store)
    blackbody_3000 = illuminant_create("blackbody", np.arange(400.0, 701.0, 1.0, dtype=float), 3000.0, asset_store=asset_store)
    d65_200 = illuminant_create("d65", None, 200.0, asset_store=asset_store)
    equal_energy = illuminant_create("equal energy", None, 200.0, asset_store=asset_store)
    equal_photons = illuminant_create("equal photons", None, 200.0, asset_store=asset_store)
    illuminant_c = illuminant_create("illuminant C", None, 200.0, asset_store=asset_store)
    mono_555 = illuminant_create("555 nm", None, 200.0, asset_store=asset_store)
    d65_sparse = illuminant_create("d65", np.arange(400.0, 601.0, 2.0, dtype=float), 200.0, asset_store=asset_store)
    d65_resampled = illuminant_set(d65_sparse, "wave", np.arange(400.0, 701.0, 5.0, dtype=float), asset_store=asset_store)
    fluorescent = illuminant_create("fluorescent", np.arange(400.0, 701.0, 5.0, dtype=float), 10.0, asset_store=asset_store)
    tungsten = illuminant_create("tungsten", None, 300.0, asset_store=asset_store)

    default_wave = np.asarray(illuminant_get(default_blackbody, "wave"), dtype=float).reshape(-1)
    mono_wave = np.asarray(illuminant_get(mono_555, "wave"), dtype=float).reshape(-1)
    mono_photons = np.asarray(illuminant_get(mono_555, "photons"), dtype=float).reshape(-1)

    assert default_wave.shape == (31,)
    assert np.isclose(float(illuminant_get(default_blackbody, "luminance", asset_store=asset_store)), 100.0, atol=1e-6, rtol=1e-6)
    assert np.asarray(illuminant_get(blackbody_3000, "photons"), dtype=float).shape == (301,)
    assert np.isclose(float(illuminant_get(d65_200, "luminance", asset_store=asset_store)), 200.0, atol=1e-6, rtol=1e-6)
    assert np.std(np.asarray(illuminant_get(equal_energy, "energy"), dtype=float)) < 1e-12
    equal_photons_values = np.asarray(illuminant_get(equal_photons, "photons"), dtype=float).reshape(-1)
    assert np.allclose(equal_photons_values, equal_photons_values[0], atol=1e-6, rtol=1e-12)
    assert np.asarray(illuminant_get(illuminant_c, "photons"), dtype=float).shape == (31,)
    assert int(np.count_nonzero(mono_photons > 0.0)) == 1
    assert np.isclose(mono_wave[int(np.argmax(mono_photons))], 550.0, atol=5.0)
    assert np.asarray(illuminant_get(d65_sparse, "energy"), dtype=float).shape == (101,)
    assert np.asarray(illuminant_get(d65_resampled, "energy"), dtype=float).shape == (61,)
    assert np.asarray(illuminant_get(fluorescent, "photons"), dtype=float).shape == (61,)
    assert np.asarray(illuminant_get(tungsten, "photons"), dtype=float).shape == (31,)


def test_illuminant_create_gaussian_accepts_key_value_params(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 5.0, dtype=float)
    params = {"center": 510.0, "sd": 5.0, "peakEnergy": 10.0}

    from_struct = illuminant_create("gaussian", wave, params, asset_store=asset_store)
    from_pairs = illuminant_create(
        "gaussian",
        wave,
        "center",
        510.0,
        "sd",
        5.0,
        "peak energy",
        10.0,
        asset_store=asset_store,
    )

    assert illuminant_get(from_pairs, "name") == "Gaussian 510"
    np.testing.assert_allclose(
        np.asarray(illuminant_get(from_pairs, "energy"), dtype=float).reshape(-1),
        np.asarray(illuminant_get(from_struct, "energy"), dtype=float).reshape(-1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_illuminant_legacy_wrappers_match_current_creation_paths(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    energy = np.linspace(0.25, 1.25, wave.size, dtype=float)
    legacy = {
        "data": {"photons": energy.copy()},
        "wavelength": wave.copy(),
        "comment": "legacy payload",
    }

    modern = illuminantModernize(legacy, asset_store=asset_store)

    assert np.array_equal(np.asarray(illuminant_get(modern, "wave"), dtype=float).reshape(-1), wave)
    assert np.allclose(np.asarray(illuminant_get(modern, "energy"), dtype=float).reshape(-1), energy)
    assert illuminant_get(modern, "comment") == "legacy payload"
    assert illuminant_get(modern, "name").startswith("CCT ")

    default_direct = illuminant_create(asset_store=asset_store)
    placeholder_direct = illuminant_create([], [], asset_store=asset_store)
    blank_direct = illuminant_create("", [], asset_store=asset_store)
    placeholder_read_energy, placeholder_read_wave = illuminantRead([], [], [], [], asset_store=asset_store)
    default_direct_energy = np.asarray(illuminant_get(default_direct, "energy"), dtype=float).reshape(-1)
    default_direct_wave = np.asarray(illuminant_get(default_direct, "wave"), dtype=float).reshape(-1)

    assert np.array_equal(np.asarray(illuminant_get(placeholder_direct, "wave"), dtype=float).reshape(-1), default_direct_wave)
    assert np.allclose(np.asarray(illuminant_get(placeholder_direct, "energy"), dtype=float).reshape(-1), default_direct_energy)
    assert np.array_equal(np.asarray(illuminant_get(blank_direct, "wave"), dtype=float).reshape(-1), default_direct_wave)
    assert np.allclose(np.asarray(illuminant_get(blank_direct, "energy"), dtype=float).reshape(-1), default_direct_energy)
    assert np.array_equal(np.asarray(placeholder_read_wave, dtype=float).reshape(-1), default_direct_wave)
    assert np.allclose(np.asarray(placeholder_read_energy, dtype=float).reshape(-1), default_direct_energy)

    read_energy, read_wave = illuminantRead(None, "d65", wave, 150.0, asset_store=asset_store)
    direct = illuminant_create("d65", wave, 150.0, asset_store=asset_store)
    assert np.array_equal(np.asarray(read_wave, dtype=float).reshape(-1), wave)
    assert np.allclose(np.asarray(read_energy, dtype=float).reshape(-1), np.asarray(illuminant_get(direct, "energy"), dtype=float).reshape(-1))

    blackbody_params = {
        "name": "blackbody",
        "temperature": 3200.0,
        "luminance": 75.0,
        "spectrum": {"wave": wave.copy()},
    }
    blackbody_energy, blackbody_wave = illuminantRead(blackbody_params, asset_store=asset_store)
    direct_blackbody = illuminant_create("blackbody", wave, 3200.0, 75.0, asset_store=asset_store)

    assert np.array_equal(np.asarray(blackbody_wave, dtype=float).reshape(-1), wave)
    assert np.allclose(
        np.asarray(blackbody_energy, dtype=float).reshape(-1),
        np.asarray(illuminant_get(direct_blackbody, "energy"), dtype=float).reshape(-1),
    )


def test_scene_illuminant_mixtures_script_workflow(asset_store) -> None:
    tungsten_scene = scene_illuminant_ss(scene_create("macbeth tungsten", asset_store=asset_store))
    daylight_scene = scene_illuminant_ss(scene_create(asset_store=asset_store))
    tungsten_energy = np.asarray(scene_get(tungsten_scene, "illuminant energy"), dtype=float)
    daylight_energy = np.asarray(scene_get(daylight_scene, "illuminant energy"), dtype=float)

    rows, cols = scene_get(tungsten_scene, "size")
    split_row = int(np.rint(rows / 2.0))
    mixed_energy = tungsten_energy.copy()
    mixed_energy[:split_row, :, :] = daylight_energy[:split_row, :, :]

    mixed_scene = scene_adjust_illuminant(tungsten_scene.clone(), mixed_energy, asset_store=asset_store)
    mixed_scene = scene_set(mixed_scene, "name", "Mixed illuminant")

    band_rows = max(1, rows // 4)
    top_slice = slice(0, band_rows)
    bottom_slice = slice(rows - band_rows, rows)
    mixed_illuminant = np.asarray(scene_get(mixed_scene, "illuminant energy"), dtype=float)
    source_reflectance = np.asarray(scene_get(tungsten_scene, "reflectance"), dtype=float)
    mixed_reflectance = np.asarray(scene_get(mixed_scene, "reflectance"), dtype=float)
    top_mixed = np.mean(mixed_illuminant[top_slice, :, :], axis=(0, 1))
    bottom_mixed = np.mean(mixed_illuminant[bottom_slice, :, :], axis=(0, 1))
    top_d65 = np.mean(daylight_energy[top_slice, :, :], axis=(0, 1))
    bottom_tungsten = np.mean(tungsten_energy[bottom_slice, :, :], axis=(0, 1))

    assert tuple(scene_get(mixed_scene, "size")) == (rows, cols)
    assert tuple(scene_get(daylight_scene, "size")) == (rows, cols)
    assert scene_get(mixed_scene, "illuminant format") == "spatial spectral"
    assert mixed_illuminant.shape == np.asarray(scene_get(mixed_scene, "photons"), dtype=float).shape
    assert np.allclose(
        top_mixed / np.max(top_mixed),
        top_d65 / np.max(top_d65),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(
        bottom_mixed / np.max(bottom_mixed),
        bottom_tungsten / np.max(bottom_tungsten),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(
        np.mean(mixed_reflectance[top_slice, :, :], axis=(0, 1)),
        np.mean(source_reflectance[top_slice, :, :], axis=(0, 1)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(
        np.mean(mixed_reflectance[bottom_slice, :, :], axis=(0, 1)),
        np.mean(source_reflectance[bottom_slice, :, :], axis=(0, 1)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.isclose(
        float(scene_get(mixed_scene, "mean luminance", asset_store=asset_store)),
        float(scene_get(tungsten_scene, "mean luminance", asset_store=asset_store)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.isfinite(float(scene_get(mixed_scene, "mean luminance", asset_store=asset_store)))


def test_scene_illuminant_space_script_workflow(asset_store) -> None:
    scene = scene_create("frequency orientation", asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    illuminant_photons_1d = np.asarray(scene_get(scene, "illuminant photons"), dtype=float).reshape(-1)
    scene = scene_illuminant_ss(scene)

    illuminant_photons = np.asarray(scene_get(scene, "illuminant photons"), dtype=float)
    rows, cols, nwave = illuminant_photons.shape
    c_temp = np.linspace(6500.0, 3000.0, rows, dtype=float)
    spd = np.asarray(blackbody(wave, c_temp, kind="quanta"), dtype=float)
    row_ratio = (spd.T / np.maximum(illuminant_photons_1d.reshape(1, nwave), 1e-12)).reshape(rows, 1, nwave)
    row_illuminant = illuminant_photons * row_ratio
    source_reflectance = np.asarray(scene_get(scene, "reflectance"), dtype=float)

    row_scene = scene.clone()
    row_scene = scene_set(row_scene, "photons", source_reflectance * row_illuminant)
    row_scene = scene_set(row_scene, "illuminant photons", row_illuminant)
    row_reflectance = np.asarray(scene_get(row_scene, "reflectance"), dtype=float)

    col_indices = np.arange(1.0, cols + 1.0, dtype=float)
    col_scale = 1.0 + 0.5 * np.sin(2.0 * np.pi * (col_indices / cols))
    col_illuminant = np.asarray(scene_get(row_scene, "illuminant photons"), dtype=float) * col_scale.reshape(1, cols, 1)
    col_scene = row_scene.clone()
    col_scene = scene_set(col_scene, "photons", row_reflectance * col_illuminant)
    col_scene = scene_set(col_scene, "illuminant photons", col_illuminant)
    col_energy = np.asarray(scene_get(col_scene, "illuminant energy"), dtype=float)
    col_reflectance = np.asarray(scene_get(col_scene, "reflectance"), dtype=float)

    row_indices = np.arange(1.0, rows + 1.0, dtype=float)
    row_scale = 1.0 + 0.5 * np.sin(2.0 * np.pi * (row_indices / rows))
    row_bug_scale = float(row_scale[cols - 1])
    final_illuminant = np.asarray(scene_get(col_scene, "illuminant photons"), dtype=float) * row_bug_scale
    final_scene = col_scene.clone()
    final_scene = scene_set(final_scene, "illuminant photons", final_illuminant)
    final_scene = scene_set(final_scene, "photons", col_reflectance * final_illuminant)
    final_energy = np.asarray(scene_get(final_scene, "illuminant energy"), dtype=float)

    center_wave_idx = int(np.argmin(np.abs(wave - 550.0)))
    col_profile = np.mean(col_energy[:, :, center_wave_idx], axis=0)
    col_profile_norm = col_profile / max(float(np.max(col_profile)), 1e-12)
    col_scale_norm = col_scale / max(float(np.max(col_scale)), 1e-12)
    final_profile = np.mean(final_energy[:, :, center_wave_idx], axis=0)
    final_profile_norm = final_profile / max(float(np.max(final_profile)), 1e-12)

    assert tuple(scene_get(scene, "size")) == (rows, cols)
    assert scene_get(scene, "illuminant format") == "spatial spectral"
    assert np.allclose(row_reflectance, source_reflectance, atol=1e-6, rtol=1e-6)
    assert np.allclose(col_reflectance, source_reflectance, atol=1e-6, rtol=1e-6)
    assert np.allclose(col_profile_norm, col_scale_norm, atol=1e-6, rtol=1e-6)
    assert np.isclose(row_bug_scale, 1.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(final_profile_norm, col_profile_norm, atol=1e-6, rtol=1e-6)
    assert np.isfinite(float(scene_get(final_scene, "mean luminance", asset_store=asset_store)))


def test_scene_xyz_illuminant_transforms_script_workflow(asset_store) -> None:
    scene = scene_create("reflectance chart", asset_store=asset_store)
    scene_d65 = scene_adjust_illuminant(scene.clone(), "D65.mat", asset_store=asset_store)
    scene_tungsten = scene_adjust_illuminant(scene.clone(), "Tungsten.mat", asset_store=asset_store)

    xyz_d65 = np.asarray(scene_get(scene_d65, "xyz", asset_store=asset_store), dtype=float)
    xyz_tungsten = np.asarray(scene_get(scene_tungsten, "xyz", asset_store=asset_store), dtype=float)
    xyz_d65_xw, rows, cols, channels = rgb_to_xw_format(xyz_d65)
    xyz_tungsten_xw, rows_t, cols_t, channels_t = rgb_to_xw_format(xyz_tungsten)

    full_transform, _, _, _ = np.linalg.lstsq(xyz_tungsten_xw, xyz_d65_xw, rcond=None)
    diagonal_transform = np.zeros((3, 3), dtype=float)
    for channel in range(3):
        diagonal_transform[channel, channel] = np.linalg.lstsq(
            xyz_tungsten_xw[:, [channel]],
            xyz_d65_xw[:, channel],
            rcond=None,
        )[0][0]

    predicted_full = xyz_tungsten_xw @ full_transform
    predicted_diagonal = xyz_tungsten_xw @ diagonal_transform
    reconstructed_xyz = xw_to_rgb_format(xyz_d65_xw, rows, cols)

    assert xyz_d65.shape[-1] == 3
    assert xyz_tungsten.shape == xyz_d65.shape
    assert (rows, cols, channels) == xyz_d65.shape
    assert (rows_t, cols_t, channels_t) == xyz_tungsten.shape
    assert reconstructed_xyz.shape == xyz_d65.shape
    assert np.allclose(reconstructed_xyz, xyz_d65, atol=1e-10, rtol=1e-10)
    assert full_transform.shape == (3, 3)
    assert diagonal_transform.shape == (3, 3)
    assert np.all(np.sqrt(np.mean(np.square(predicted_full - xyz_d65_xw), axis=0)) < np.sqrt(np.mean(np.square(predicted_diagonal - xyz_d65_xw), axis=0)))


def test_color_illuminant_transforms_script_workflow(asset_store) -> None:
    def _unit_length(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        return vector / max(float(np.linalg.norm(vector)), 1.0e-12)

    scene = scene_create("reflectance chart", asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    bb_range = np.arange(3500.0, 8000.0 + 0.1, 500.0, dtype=float)
    nbb = bb_range.size

    transform_list = np.zeros((9, nbb * nbb), dtype=float)
    column = 0
    for source_temp in bb_range:
        source_scene = scene_adjust_illuminant(scene.clone(), blackbody(wave, source_temp), asset_store=asset_store)
        xyz_source = np.asarray(scene_get(source_scene, "xyz", asset_store=asset_store), dtype=float)
        xyz_source_xw, rows, cols, channels = rgb_to_xw_format(xyz_source)
        for target_temp in bb_range:
            target_scene = scene_adjust_illuminant(scene.clone(), blackbody(wave, target_temp), asset_store=asset_store)
            xyz_target = np.asarray(scene_get(target_scene, "xyz", asset_store=asset_store), dtype=float)
            xyz_target_xw, rows_t, cols_t, channels_t = rgb_to_xw_format(xyz_target)
            transform, _, _, _ = np.linalg.lstsq(xyz_source_xw, xyz_target_xw, rcond=None)
            transform_list[:, column] = _unit_length(transform)
            column += 1

    buddha = _unit_length(
        np.array(
            [
                [0.9245, 0.0241, -0.0649],
                [0.2679, 0.9485, 0.1341],
                [-0.1693, 0.0306, 0.9078],
            ],
            dtype=float,
        )
    )
    flower = _unit_length(
        np.array(
            [
                [0.9570, -0.0727, -0.0347],
                [0.0588, 0.9682, -0.1848],
                [0.0423, 0.1489, 1.2323],
            ],
            dtype=float,
        )
    )
    buddha_similarity = (transform_list.T @ buddha).reshape(nbb, nbb, order="F")
    flower_similarity = (transform_list.T @ flower).reshape(nbb, nbb, order="F")
    transform_diagonal_terms = transform_list[[0, 4, 8], :]

    assert bb_range.shape == (10,)
    assert transform_list.shape == (9, 100)
    assert transform_diagonal_terms.shape == (3, 100)
    assert np.allclose(np.sqrt(np.sum(np.square(transform_list), axis=0)), np.ones(100), atol=1e-10, rtol=1e-10)
    assert buddha_similarity.shape == (10, 10)
    assert flower_similarity.shape == (10, 10)
    assert (rows, cols, channels) == (rows_t, cols_t, channels_t)
    assert np.max(buddha_similarity) <= 1.0 + 1e-12
    assert np.max(flower_similarity) <= 1.0 + 1e-12


def test_run_python_case_supports_color_illuminant_transforms_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("color_illuminant_transforms_small", asset_store=asset_store)

    assert np.array_equal(case.payload["bb_range"], np.arange(3500.0, 8000.0 + 0.1, 500.0, dtype=float))
    assert tuple(case.payload["scene_size"]) == (240, 264)
    assert case.payload["transform_diagonal_terms"].shape == (3, 100)
    assert case.payload["buddha_similarity"].shape == (10, 10)
    assert case.payload["flower_similarity"].shape == (10, 10)


def test_chromatic_spatial_chart_script_workflow(asset_store) -> None:
    n_rows = 256
    n_cols = 3 * n_rows
    max_freq = 30.0
    c_weights = np.array([0.3, 0.7, 1.0], dtype=float)
    c_freq = np.array([1.0, 1.5, 2.0], dtype=float) * 10.0

    r_samples = np.arange(n_rows, dtype=float)
    x = np.arange(1.0, n_cols + 1.0, dtype=float) / n_cols
    freq = (x**2) * max_freq
    img_row = np.sin(2.0 * np.pi * (freq * x))
    img_row = (img_row - float(np.min(img_row))) / max(float(np.max(img_row) - np.min(img_row)), 1e-12)
    img_row = img_row * 255.0 + 1.0
    img_row = img_row / max(float(np.max(img_row)), 1e-12) + 2.0

    channel_rows = np.stack(
        [c_weights[idx] * np.cos(2.0 * np.pi * c_freq[idx] * r_samples / n_rows) + 2.0 for idx in range(3)],
        axis=0,
    )

    rgb = np.zeros((n_rows, n_cols, 3), dtype=float)
    for idx in range(3):
        rgb[:, :, idx] = channel_rows[idx, :, None] * img_row[None, :]
    rgb /= np.max(rgb)

    border = np.zeros((n_rows // 4, n_cols, 3), dtype=float)
    border_template = np.full((n_rows // 4, 1), 0.5, dtype=float) @ img_row[None, :]
    border_template /= np.max(border_template)
    for idx in range(3):
        border[:, :, idx] = border_template
    rgb = np.concatenate([border, rgb, border], axis=0)

    center_row = rgb.shape[0] // 2
    center_col = rgb.shape[1] // 2
    scene = scene_from_file(rgb, "rgb", 100.0, "LCD-Apple.mat", asset_store=asset_store)
    scene_wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    scene_luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    scene_photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert rgb.shape == (384, 768, 3)
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)
    assert tuple(scene_get(scene, "size")) == (384, 768)
    assert scene_wave.shape == (101,)
    assert np.isclose(float(scene_get(scene, "mean luminance", asset_store=asset_store)), 100.0, atol=1e-6, rtol=1e-6)
    assert scene_luminance.shape == (384, 768)
    assert scene_photons.shape == (384, 768, 101)
    assert np.max(scene_luminance[center_row, :]) > 0.0
    assert np.max(scene_luminance[:, center_col]) > 0.0


def test_run_python_case_supports_chromatic_spatial_chart_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("chromatic_spatial_chart_small", asset_store=asset_store)

    assert tuple(case.payload["source_rgb_size"]) == (384, 768)
    assert case.payload["source_channel_means"].shape == (3,)
    assert case.payload["source_center_row_rgb"].shape == (768, 3)
    assert case.payload["source_center_col_rgb"].shape == (384, 3)
    assert tuple(case.payload["scene_size"]) == (384, 768)
    assert case.payload["scene_wave"].shape == (101,)
    assert np.isclose(float(case.payload["scene_mean_luminance"]), 100.0, atol=1e-6, rtol=1e-6)
    assert case.payload["scene_mean_photons_norm"].shape == (101,)
    assert case.payload["scene_center_row_luminance_norm"].shape == (768,)
    assert case.payload["scene_center_col_luminance_norm"].shape == (384,)


def test_scene_imgtarget_wrappers_match_existing_pattern_generators(asset_store) -> None:
    wave = np.array([550.0], dtype=float)

    def normalized_photons(scene) -> np.ndarray:
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)[:, :, 0]
        return photons / max(float(np.max(photons)), 1e-12)

    disk = imgDiskArray(64, 5, [2, 3])
    disk_scene = scene_create("diskarray", 64, 5, np.array([2, 3], dtype=int), wave, asset_store=asset_store)
    assert np.allclose(disk / max(float(np.max(disk)), 1e-12), normalized_photons(disk_scene))

    square = imgSquareArray(64, 6, [2, 2])
    square_scene = scene_create("squarearray", 64, 6, np.array([2, 2], dtype=int), wave, asset_store=asset_store)
    assert np.allclose(square / max(float(np.max(square)), 1e-12), normalized_photons(square_scene))

    mackay = imgMackay(10, 63)
    mackay_scene = scene_create("mackay", 10, 63, wave, asset_store=asset_store)
    assert np.allclose(mackay / max(float(np.max(mackay)), 1e-12), normalized_photons(mackay_scene))

    sweep = imgSweep([32, 48], 6.0)
    sweep_scene = scene_create("sweep", np.array([32, 48], dtype=int), 6.0, wave, asset_store=asset_store)
    assert np.allclose(sweep / max(float(np.max(sweep)), 1e-12), normalized_photons(sweep_scene))

    ramp = imgRamp([32, 48], 32.0)
    ramp_scene = scene_create("linear intensity ramp", np.array([32, 48], dtype=int), 32.0, wave, asset_store=asset_store)
    assert np.allclose(ramp / max(float(np.max(ramp)), 1e-12), normalized_photons(ramp_scene))

    zone = imgZonePlate([31, 47])
    zone_scene = scene_create("zoneplate", np.array([31, 47], dtype=int), wave, asset_store=asset_store)
    zone_scene_expected = np.clip(zone, 1.0e-4, 1.0)
    assert np.allclose(zone_scene_expected / max(float(np.max(zone_scene_expected)), 1e-12), normalized_photons(zone_scene))


def test_scene_imgtarget_wrappers_cover_radial_zone_and_deadleaves_contracts() -> None:
    centered = imgRadialRamp([5, 7])
    cornered = imgRadialRamp([5, 7], 2.0, [1, 1])
    zone = imgZonePlate([5, 7], 2.0, np.pi / 2.0)
    expected_zone = 2.0 * np.cos((np.pi / 7.0) * imgRadialRamp([5, 7], 2.0) + (np.pi / 2.0)) + 1.0
    leaves_options = {
        "rmin": 0.25,
        "rmax": 0.25,
        "nbr_iter": 2,
        "shape": "square",
        "random_samples": np.array(
            [
                [0.0, 0.25, 0.25, 0.2],
                [0.0, 0.75, 0.75, 0.8],
            ],
            dtype=float,
        ),
    }
    leaves = imgDeadleaves([8, 8], 2.0, leaves_options)

    assert centered.shape == (5, 7)
    assert np.isclose(centered[2, 3], 0.0)
    assert np.isclose(cornered[0, 0], 0.0)
    assert cornered[-1, -1] > cornered[0, 0]
    assert np.allclose(zone, expected_zone)
    assert leaves.shape == (8, 8)
    assert np.all(np.isfinite(leaves))
    assert np.all((leaves >= 0.0) & (leaves <= 1.0))
    assert np.any(np.isclose(leaves, 0.2))
    assert np.any(np.isclose(leaves, 0.8))


def test_color_constancy_script_workflow(asset_store) -> None:
    def normalize(values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        return vector / max(float(np.max(np.abs(vector))), 1e-12)

    c_temps = np.flip(1.0 / np.linspace(1.0 / 7000.0, 1.0 / 3000.0, 15, dtype=float))

    stuffed_scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
        "spectral",
        asset_store=asset_store,
    )
    stuffed_wave = np.asarray(scene_get(stuffed_scene, "wave"), dtype=float).reshape(-1)
    stuffed_means = np.zeros((c_temps.size, 3), dtype=float)
    stuffed_centers = np.zeros((c_temps.size, 3), dtype=float)

    for index, c_temp in enumerate(c_temps):
        stuffed_scene = scene_adjust_illuminant(stuffed_scene, blackbody(stuffed_wave, c_temp, kind="energy"), asset_store=asset_store)
        rgb = np.asarray(scene_get(stuffed_scene, "rgb", asset_store=asset_store), dtype=float)
        center_row = rgb.shape[0] // 2
        center_col = rgb.shape[1] // 2
        stuffed_means[index, :] = normalize(np.mean(rgb, axis=(0, 1), dtype=float))
        stuffed_centers[index, :] = normalize(rgb[center_row, center_col, :])

    uniform_scene = scene_create("uniform d65", 512, asset_store=asset_store)
    uniform_wave = np.asarray(scene_get(uniform_scene, "wave"), dtype=float).reshape(-1)
    uniform_means = np.zeros((c_temps.size, 3), dtype=float)
    uniform_centers = np.zeros((c_temps.size, 3), dtype=float)

    for index, c_temp in enumerate(c_temps):
        uniform_scene = scene_adjust_illuminant(uniform_scene, blackbody(uniform_wave, c_temp, kind="energy"), asset_store=asset_store)
        rgb = np.asarray(scene_get(uniform_scene, "rgb", asset_store=asset_store), dtype=float)
        center_row = rgb.shape[0] // 2
        center_col = rgb.shape[1] // 2
        uniform_means[index, :] = normalize(np.mean(rgb, axis=(0, 1), dtype=float))
        uniform_centers[index, :] = normalize(rgb[center_row, center_col, :])

    assert np.all(np.diff(c_temps) > 0.0)
    assert tuple(scene_get(stuffed_scene, "size")) == (506, 759)
    assert stuffed_wave.shape == (31,)
    assert stuffed_means.shape == (15, 3)
    assert stuffed_centers.shape == (15, 3)
    assert tuple(scene_get(uniform_scene, "size")) == (512, 512)
    assert uniform_wave.shape == (31,)
    assert uniform_means.shape == (15, 3)
    assert uniform_centers.shape == (15, 3)


def test_run_python_case_supports_color_constancy_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("color_constancy_small", asset_store=asset_store)

    assert case.payload["c_temps"].shape == (15,)
    assert np.all(np.diff(np.asarray(case.payload["c_temps"], dtype=float)) > 0.0)
    assert tuple(case.payload["stuffed_scene_size"]) == (506, 759)
    assert case.payload["stuffed_wave"].shape == (31,)
    assert case.payload["stuffed_mean_luminance"].shape == (15,)
    assert case.payload["stuffed_mean_rgb_norm"].shape == (15, 3)
    assert case.payload["stuffed_center_rgb_norm"].shape == (15, 3)
    assert tuple(case.payload["uniform_scene_size"]) == (512, 512)
    assert case.payload["uniform_wave"].shape == (31,)
    assert case.payload["uniform_mean_luminance"].shape == (15,)
    assert case.payload["uniform_mean_rgb_norm"].shape == (15, 3)
    assert case.payload["uniform_center_rgb_norm"].shape == (15, 3)


def test_rgb_color_temperature_script_workflow(asset_store) -> None:
    def estimate(scene_name: str) -> tuple[tuple[int, int], tuple[int, int, int], float, np.ndarray, np.ndarray]:
        scene = scene_create(scene_name, asset_store=asset_store)
        oi = oi_compute(oi_create(asset_store=asset_store), scene)
        sensor = sensor_create(asset_store=asset_store)
        sensor = sensor_set(sensor, "fov", scene_get(scene, "fov"), oi)
        sensor = sensor_compute(sensor, oi)
        ip = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)
        srgb = np.asarray(ip_get(ip, "srgb"), dtype=float)
        c_temp, c_table = srgb_to_color_temp(srgb, return_table=True, asset_store=asset_store)
        return tuple(scene_get(scene, "size")), tuple(ip_get(ip, "size")), float(c_temp), np.asarray(c_table, dtype=float), srgb

    tungsten_scene_size, tungsten_ip_size, tungsten_c_temp, tungsten_table, tungsten_srgb = estimate("macbeth tungsten")
    d65_scene_size, d65_ip_size, d65_c_temp, d65_table, d65_srgb = estimate("macbeth d65")

    assert tungsten_scene_size == (64, 96)
    assert d65_scene_size == (64, 96)
    assert tungsten_ip_size == d65_ip_size
    assert tungsten_ip_size[2] == 3
    assert tungsten_ip_size[0] > tungsten_scene_size[0]
    assert tungsten_ip_size[1] > tungsten_scene_size[1]
    assert tungsten_table.shape == (17, 3)
    assert np.array_equal(tungsten_table[:, 0], np.arange(2500.0, 10501.0, 500.0, dtype=float))
    assert np.allclose(d65_table, tungsten_table, atol=1e-10, rtol=1e-10)
    assert 2500.0 <= tungsten_c_temp <= 10500.0
    assert 2500.0 <= d65_c_temp <= 10500.0
    assert d65_c_temp > tungsten_c_temp
    assert tungsten_srgb.shape == tungsten_ip_size
    assert d65_srgb.shape == d65_ip_size


def test_run_python_case_supports_rgb_color_temperature_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("rgb_color_temperature_small", asset_store=asset_store)

    assert tuple(case.payload["tungsten_scene_size"]) == (64, 96)
    assert tuple(case.payload["tungsten_ip_size"]) == tuple(case.payload["d65_ip_size"])
    assert int(case.payload["tungsten_ip_size"][2]) == 3
    assert int(case.payload["tungsten_ip_size"][0]) > 64
    assert int(case.payload["tungsten_ip_size"][1]) > 96
    assert float(case.payload["tungsten_c_temp"]) >= 2500.0
    assert case.payload["tungsten_srgb_mean_norm"].shape == (3,)
    assert tuple(case.payload["d65_scene_size"]) == (64, 96)
    assert float(case.payload["d65_c_temp"]) > float(case.payload["tungsten_c_temp"])
    assert case.payload["d65_srgb_mean_norm"].shape == (3,)
    assert np.array_equal(case.payload["c_table_temps"], np.arange(2500.0, 10501.0, 500.0, dtype=float))
    assert case.payload["c_table_xy"].shape == (17, 2)


def test_srgb_gamut_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    srgb_xy = np.asarray(srgb_parameters("chromaticity"), dtype=float)
    adobergb_xy = np.asarray(adobergb_parameters("chromaticity"), dtype=float)

    natural_files = [
        "Nature_Vhrel.mat",
        "Objects_Vhrel.mat",
        "Food_Vhrel.mat",
        "Clothes_Vhrel.mat",
        "Hair_Vhrel.mat",
    ]
    natural_samples = [
        np.arange(1, 80, dtype=int),
        np.arange(1, 171, dtype=int),
        np.arange(1, 28, dtype=int),
        np.arange(1, 42, dtype=int),
        np.arange(1, 8, dtype=int),
    ]
    synthetic_files = [
        "DupontPaintChip_Vhrel.mat",
        "MunsellSamples_Vhrel.mat",
        "esserChart.mat",
        "gretagDigitalColorSG.mat",
    ]
    synthetic_samples = [
        np.arange(1, 121, dtype=int),
        np.arange(1, 65, dtype=int),
        np.arange(1, 114, dtype=int),
        np.arange(1, 141, dtype=int),
    ]

    natural_scene, natural_sample_list, natural_reflectances, natural_rc_size = scene_reflectance_chart(
        natural_files,
        natural_samples,
        32,
        wave,
        True,
        asset_store=asset_store,
    )
    natural_d65_scene = scene_adjust_illuminant(natural_scene.clone(), "D65.mat", asset_store=asset_store)
    natural_d65_light = np.asarray(scene_get(natural_d65_scene, "illuminant energy"), dtype=float).reshape(-1)
    natural_d65_xy = np.asarray(
        chromaticity_xy(
            xyz_from_energy((natural_d65_light.reshape(-1, 1) * natural_reflectances).T, wave, asset_store=asset_store)
        ),
        dtype=float,
    )

    natural_yellow_scene = scene_adjust_illuminant(
        natural_scene.clone(),
        blackbody(wave, 3000.0, kind="energy"),
        asset_store=asset_store,
    )
    natural_yellow_light = np.asarray(scene_get(natural_yellow_scene, "illuminant energy"), dtype=float).reshape(-1)
    natural_yellow_xy = np.asarray(
        chromaticity_xy(
            xyz_from_energy((natural_yellow_light.reshape(-1, 1) * natural_reflectances).T, wave, asset_store=asset_store)
        ),
        dtype=float,
    )

    synthetic_scene, synthetic_sample_list, synthetic_reflectances, synthetic_rc_size = scene_reflectance_chart(
        synthetic_files,
        synthetic_samples,
        32,
        wave,
        True,
        asset_store=asset_store,
    )
    synthetic_d65_scene = scene_adjust_illuminant(synthetic_scene.clone(), "D65.mat", asset_store=asset_store)
    synthetic_d65_light = np.asarray(scene_get(synthetic_d65_scene, "illuminant energy"), dtype=float).reshape(-1)
    synthetic_d65_xy = np.asarray(
        chromaticity_xy(
            xyz_from_energy((synthetic_d65_light.reshape(-1, 1) * synthetic_reflectances).T, wave, asset_store=asset_store)
        ),
        dtype=float,
    )

    assert srgb_xy.shape == (2, 3)
    assert adobergb_xy.shape == (2, 3)
    assert np.allclose(srgb_parameters("xyYwhite"), np.array([0.3127, 0.3290, 1.0]))
    assert np.allclose(adobergb_parameters("xyzblack"), np.array([0.5282, 0.5557, 0.6052]))
    assert tuple(scene_get(natural_scene, "size")) == (576, 608)
    assert tuple(natural_rc_size) == (18, 19)
    assert natural_reflectances.shape == (31, 342)
    assert [len(sample_list) for sample_list in natural_sample_list] == [79, 170, 27, 41, 7]
    assert natural_d65_xy.shape == (342, 2)
    assert natural_yellow_xy.shape == (342, 2)
    assert float(np.mean(natural_yellow_xy[:, 0])) > float(np.mean(natural_d65_xy[:, 0]))
    assert tuple(scene_get(synthetic_scene, "size")) == (672, 704)
    assert tuple(synthetic_rc_size) == (21, 22)
    assert synthetic_reflectances.shape == (31, 458)
    assert [len(sample_list) for sample_list in synthetic_sample_list] == [120, 64, 113, 140]
    assert synthetic_d65_xy.shape == (458, 2)
    assert np.all(np.isfinite(natural_d65_xy))
    assert np.all(np.isfinite(natural_yellow_xy))
    assert np.all(np.isfinite(synthetic_d65_xy))


def test_run_python_case_supports_srgb_gamut_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("srgb_gamut_small", asset_store=asset_store)

    assert np.array_equal(case.payload["wave"], np.arange(400.0, 701.0, 10.0, dtype=float))
    assert case.payload["srgb_xy_loop"].shape == (2, 4)
    assert case.payload["adobergb_xy_loop"].shape == (2, 4)
    assert tuple(case.payload["natural_scene_size"]) == (576, 608)
    assert tuple(case.payload["natural_rc_size"]) == (18, 19)
    assert np.array_equal(case.payload["natural_sample_counts"], np.array([79, 170, 27, 41, 7], dtype=int))
    assert tuple(case.payload["natural_reflectance_size"]) == (31, 342)
    assert case.payload["natural_d65_xy"].shape == (342, 2)
    assert case.payload["natural_yellow_xy"].shape == (342, 2)
    assert tuple(case.payload["synthetic_scene_size"]) == (672, 704)
    assert tuple(case.payload["synthetic_rc_size"]) == (21, 22)
    assert np.array_equal(case.payload["synthetic_sample_counts"], np.array([120, 64, 113, 140], dtype=int))
    assert tuple(case.payload["synthetic_reflectance_size"]) == (31, 458)
    assert case.payload["synthetic_d65_xy"].shape == (458, 2)


def test_scene_reflectance_charts_script_workflow(asset_store) -> None:
    default_scene = scene_create("reflectance chart", asset_store=asset_store)
    default_chart = scene_get(default_scene, "chart parameters")

    s_files = [
        "MunsellSamples_Vhrel.mat",
        "Food_Vhrel.mat",
        "DupontPaintChip_Vhrel.mat",
        "HyspexSkinReflectance.mat",
    ]
    s_samples = [12, 12, 24, 24]
    p_size = 24

    custom_scene = scene_create("reflectance chart", p_size, s_samples, s_files, None, False, "no replacement", asset_store=asset_store)
    custom_scene_kv = scene_create(
        "reflectance chart",
        "p size",
        p_size,
        "s samples",
        s_samples,
        "s files",
        s_files,
        "gray flag",
        False,
        "sampling",
        "no replacement",
        asset_store=asset_store,
    )
    custom_chart = scene_get(custom_scene, "chart parameters")
    custom_chart_kv = scene_get(custom_scene_kv, "chart parameters")
    wave = np.asarray(scene_get(custom_scene, "wave"), dtype=float).reshape(-1)

    d65_scene = scene_adjust_illuminant(custom_scene.clone(), "D65", asset_store=asset_store)
    d65_illuminant = np.asarray(scene_get(d65_scene, "illuminant energy"), dtype=float).reshape(-1)

    gray_scene, _, gray_reflectances, gray_rc = scene_reflectance_chart(
        s_files,
        s_samples,
        p_size,
        wave,
        True,
        asset_store=asset_store,
    )
    gray_chart = scene_get(gray_scene, "chart parameters")
    gray_col = int(gray_chart["rowcol"][1]) - 1
    gray_patch = np.asarray(scene_get(gray_scene, "photons"), dtype=float)[:, gray_col * p_size : (gray_col + 1) * p_size, :]
    gray_mask = np.asarray(gray_chart["rIdxMap"], dtype=int)[:, gray_col * p_size : (gray_col + 1) * p_size] > 0
    gray_mean_spd = np.mean(gray_patch[gray_mask], axis=0, dtype=float)

    original_scene, stored_samples, _, _ = scene_reflectance_chart(s_files, s_samples, p_size, asset_store=asset_store)
    replica_scene, replica_samples, _, _ = scene_reflectance_chart(s_files, stored_samples, p_size, asset_store=asset_store)
    original_photons = np.asarray(scene_get(original_scene, "photons"), dtype=float)
    replica_photons = np.asarray(scene_get(replica_scene, "photons"), dtype=float)

    assert tuple(scene_get(default_scene, "size")) == (240, 264)
    assert tuple(default_chart["rowcol"]) == (10, 11)
    assert [len(item) for item in default_chart["sSamples"]] == [50, 40, 10]
    assert np.isclose(scene_get(default_scene, "mean luminance", asset_store=asset_store), 100.0, rtol=1e-8, atol=1e-8)
    assert tuple(scene_get(custom_scene, "size")) == (216, 192)
    assert tuple(scene_get(custom_scene_kv, "size")) == (216, 192)
    assert tuple(custom_chart["rowcol"]) == (9, 8)
    assert tuple(custom_chart_kv["rowcol"]) == (9, 8)
    assert [len(item) for item in custom_chart["sSamples"]] == [12, 12, 24, 24]
    assert [len(item) for item in custom_chart_kv["sSamples"]] == [12, 12, 24, 24]
    assert np.array_equal(np.unique(custom_chart["rIdxMap"]), np.arange(1, 73, dtype=int))
    assert np.array_equal(np.asarray(scene_get(custom_scene_kv, "photons"), dtype=float), np.asarray(scene_get(custom_scene, "photons"), dtype=float))
    assert np.isclose(scene_get(d65_scene, "mean luminance", asset_store=asset_store), 100.0, rtol=1e-8, atol=1e-8)
    assert d65_illuminant.shape == (31,)
    assert tuple(scene_get(gray_scene, "size")) == (216, 216)
    assert tuple(gray_rc) == (9, 9)
    assert gray_reflectances.shape == (31, 81)
    assert gray_mean_spd.shape == (31,)
    assert [len(item) for item in stored_samples] == [12, 12, 24, 24]
    assert np.array_equal(
        np.concatenate([np.asarray(item, dtype=int).reshape(-1) for item in stored_samples]),
        np.concatenate([np.asarray(item, dtype=int).reshape(-1) for item in replica_samples]),
    )
    assert np.allclose(replica_photons, original_photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_reflectance_charts_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_reflectance_charts_small", asset_store=asset_store)

    assert tuple(case.payload["default_scene_size"]) == (240, 264)
    assert tuple(case.payload["default_chart_rowcol"]) == (10, 11)
    assert np.array_equal(case.payload["default_sample_counts"], np.array([50, 40, 10], dtype=int))
    assert np.isclose(float(case.payload["default_mean_luminance"]), 100.0, rtol=1e-8, atol=1e-8)
    assert tuple(case.payload["custom_scene_size"]) == (216, 192)
    assert tuple(case.payload["custom_chart_rowcol"]) == (9, 8)
    assert np.array_equal(case.payload["custom_sample_counts"], np.array([12, 12, 24, 24], dtype=int))
    assert tuple(case.payload["custom_reflectance_shape"]) == (31, 72)
    assert np.array_equal(case.payload["custom_idx_map_unique"], np.arange(1, 73, dtype=int))
    assert case.payload["d65_illuminant_norm"].shape == (31,)
    assert np.isclose(float(case.payload["d65_mean_luminance"]), 100.0, rtol=1e-8, atol=1e-8)
    assert tuple(case.payload["gray_scene_size"]) == (216, 216)
    assert tuple(case.payload["gray_chart_rowcol"]) == (9, 9)
    assert tuple(case.payload["gray_reflectance_shape"]) == (31, 81)
    assert case.payload["gray_mean_spd_norm"].shape == (31,)
    assert np.array_equal(case.payload["stored_sample_counts"], np.array([12, 12, 24, 24], dtype=int))
    assert np.isclose(float(case.payload["replica_photons_nmae"]), 0.0, atol=1e-12, rtol=1e-12)


def test_scene_change_illuminant_script_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    default_illuminant = np.asarray(scene_get(scene, "illuminant photons"), dtype=float).reshape(-1)

    tungsten_energy = np.asarray(ie_read_spectra("Tungsten.mat", wave, asset_store=asset_store), dtype=float).reshape(-1)
    tungsten_scene = scene_adjust_illuminant(scene.clone(), tungsten_energy, asset_store=asset_store)
    tungsten_scene = scene_set(tungsten_scene, "illuminant comment", "Tungsten illuminant")
    tungsten_illuminant = np.asarray(scene_get(tungsten_scene, "illuminant photons"), dtype=float).reshape(-1)

    stuffed_scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
        "multispectral",
        asset_store=asset_store,
    )
    stuffed_illuminant = np.asarray(scene_get(stuffed_scene, "illuminant energy"), dtype=float).reshape(-1)

    equal_energy_scene = scene_adjust_illuminant(stuffed_scene.clone(), "equalEnergy.mat", asset_store=asset_store)
    equal_energy_illuminant = np.asarray(scene_get(equal_energy_scene, "illuminant energy"), dtype=float).reshape(-1)
    equal_energy_rgb = np.asarray(scene_get(equal_energy_scene, "rgb", asset_store=asset_store), dtype=float)

    horizon_scene = scene_adjust_illuminant(stuffed_scene.clone(), "illHorizon-20180220.mat", asset_store=asset_store)
    horizon_illuminant = np.asarray(scene_get(horizon_scene, "illuminant energy"), dtype=float).reshape(-1)
    horizon_rgb = np.asarray(scene_get(horizon_scene, "rgb", asset_store=asset_store), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 96)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert default_illuminant.shape == (31,)
    assert np.isclose(scene_get(tungsten_scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert scene_get(tungsten_scene, "illuminant comment") == "Tungsten illuminant"
    assert tungsten_illuminant.shape == (31,)
    assert tuple(scene_get(stuffed_scene, "size")) == (506, 759)
    assert stuffed_illuminant.shape == (31,)
    assert np.isclose(scene_get(equal_energy_scene, "mean luminance", asset_store=asset_store), scene_get(stuffed_scene, "mean luminance", asset_store=asset_store), atol=1e-8, rtol=1e-8)
    assert equal_energy_illuminant.shape == (31,)
    assert equal_energy_rgb.shape == (506, 759, 3)
    assert np.isclose(scene_get(horizon_scene, "mean luminance", asset_store=asset_store), scene_get(stuffed_scene, "mean luminance", asset_store=asset_store), atol=1e-8, rtol=1e-8)
    assert horizon_illuminant.shape == (31,)
    assert horizon_rgb.shape == (506, 759, 3)
    assert not np.allclose(equal_energy_illuminant, horizon_illuminant)
    assert not np.allclose(np.mean(equal_energy_rgb, axis=(0, 1)), np.mean(horizon_rgb, axis=(0, 1)))


def test_run_python_case_supports_scene_change_illuminant_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_change_illuminant_small", asset_store=asset_store)

    assert tuple(case.payload["default_scene_size"]) == (64, 96)
    assert np.isclose(float(case.payload["default_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["default_illuminant_photons_norm"].shape == (31,)
    assert np.isclose(float(case.payload["tungsten_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["tungsten_comment"] == "Tungsten illuminant"
    assert case.payload["tungsten_illuminant_photons_norm"].shape == (31,)
    assert tuple(case.payload["stuffed_scene_size"]) == (506, 759)
    assert case.payload["stuffed_illuminant_energy_norm"].shape == (31,)
    assert case.payload["equal_energy_illuminant_norm"].shape == (31,)
    assert case.payload["equal_energy_mean_rgb_norm"].shape == (3,)
    assert case.payload["horizon_illuminant_norm"].shape == (31,)
    assert case.payload["horizon_mean_rgb_norm"].shape == (3,)


def test_scene_data_extraction_and_plotting_script_workflow(asset_store) -> None:
    scene = scene_create("macbethd65", asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    center_row = int(round(float(scene_get(scene, "rows")) / 2.0))

    line_data, _ = scene_plot(scene, "luminance hline", [1, center_row], asset_store=asset_store)
    illuminant_data, _ = scene_plot(scene, "illuminant energy", asset_store=asset_store)

    rect = np.array([51, 35, 10, 11], dtype=int)
    roi_locs = ie_rect2_locs(rect)
    energy_plot, _ = scene_plot(scene, "radiance energy roi", roi_locs, asset_store=asset_store)
    photons_plot, _ = scene_plot(scene, "radiance photons roi", roi_locs, asset_store=asset_store)
    reflectance_plot, _ = scene_plot(scene, "reflectance", roi_locs, asset_store=asset_store)

    photons_manual = np.mean(np.asarray(vc_get_roi_data(scene, roi_locs, "photons"), dtype=float), axis=0)
    energy_manual = np.mean(np.asarray(vc_get_roi_data(scene, roi_locs, "energy"), dtype=float), axis=0)

    assert tuple(scene_get(scene, "size")) == (64, 96)
    assert np.array_equal(wave, np.arange(400.0, 701.0, 10.0, dtype=float))
    assert center_row == 32
    assert line_data["unit"] == "mm"
    assert np.asarray(line_data["pos"]).shape == (96,)
    assert np.asarray(line_data["data"]).shape == (96,)
    assert np.all(np.diff(np.asarray(line_data["pos"], dtype=float)) > 0.0)
    assert illuminant_data["comment"] == "D65.mat"
    assert np.asarray(illuminant_data["energy"]).shape == (31,)
    assert np.array_equal(rect, np.array([51, 35, 10, 11], dtype=int))
    assert roi_locs.shape == (132, 2)
    assert np.asarray(energy_plot["energy"]).shape == (31,)
    assert np.asarray(photons_plot["photons"]).shape == (31,)
    assert np.asarray(reflectance_plot["reflectance"]).shape == (31,)
    assert np.allclose(np.asarray(photons_plot["photons"]), photons_manual, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.asarray(energy_plot["energy"]), energy_manual, atol=1e-12, rtol=1e-12)
    assert float(np.min(np.asarray(reflectance_plot["reflectance"], dtype=float))) > 0.0
    assert float(np.max(np.asarray(reflectance_plot["reflectance"], dtype=float))) < 1.0


def test_run_python_case_supports_scene_data_extraction_plotting_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_data_extraction_plotting_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 96)
    assert np.array_equal(case.payload["wave"], np.arange(400.0, 701.0, 10.0, dtype=float))
    assert int(case.payload["center_row"]) == 32
    assert case.payload["luminance_hline_pos_mm"].shape == (96,)
    assert case.payload["luminance_hline_norm"].shape == (96,)
    assert case.payload["illuminant_energy_norm"].shape == (31,)
    assert np.array_equal(case.payload["roi_rect"], np.array([51, 35, 10, 11], dtype=int))
    assert int(case.payload["roi_count"]) == 132
    assert np.isfinite(float(case.payload["roi_energy_mean"]))
    assert case.payload["roi_energy_norm"].shape == (31,)
    assert case.payload["roi_energy_manual_norm"].shape == (31,)
    assert np.isclose(float(case.payload["roi_energy_plot_manual_max_abs"]), 0.0, atol=1e-12, rtol=1e-12)
    assert np.isfinite(float(case.payload["roi_photons_mean"]))
    assert case.payload["roi_photons_norm"].shape == (31,)
    assert case.payload["roi_photons_manual_norm"].shape == (31,)
    assert np.isclose(float(case.payload["roi_photons_plot_manual_max_abs"]), 0.0, atol=1e-12, rtol=1e-12)
    assert float(case.payload["roi_reflectance_mean"]) > 0.0
    assert case.payload["roi_reflectance_norm"].shape == (31,)


def test_scene_monochrome_script_workflow(asset_store) -> None:
    display = display_create("crt", asset_store=asset_store)
    display_wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)

    scene = scene_from_file("cameraman.tif", "monochrome", 100.0, "crt", asset_store=asset_store)
    scene_wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    scene_illuminant = np.asarray(scene_get(scene, "illuminant energy"), dtype=float).reshape(-1)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    center_row = photons.shape[0] // 2
    center_col = photons.shape[1] // 2
    source_center_spd = photons[center_row, center_col, :]

    adjusted_scene = scene_adjust_illuminant(
        scene.clone(),
        blackbody(scene_wave, 6500.0, kind="energy"),
        asset_store=asset_store,
    )
    adjusted_illuminant = np.asarray(scene_get(adjusted_scene, "illuminant energy"), dtype=float).reshape(-1)
    adjusted_rgb = np.asarray(scene_get(adjusted_scene, "rgb", asset_store=asset_store), dtype=float)

    assert display_wave.shape == (101,)
    assert white_spd.shape == (101,)
    assert tuple(scene_get(scene, "size")) == (256, 256)
    assert np.array_equal(scene_wave, display_wave)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(
        scene_illuminant / np.max(scene_illuminant),
        white_spd / np.max(white_spd),
        atol=1e-12,
        rtol=1e-12,
    )
    assert photons.shape == (256, 256, 101)
    assert source_center_spd.shape == (101,)
    assert np.isclose(scene_get(adjusted_scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert adjusted_illuminant.shape == (101,)
    assert adjusted_rgb.shape == (256, 256, 3)
    assert not np.allclose(adjusted_illuminant, scene_illuminant)
    assert np.all(np.mean(adjusted_rgb, axis=(0, 1)) > 0.0)


def test_run_python_case_supports_scene_monochrome_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_monochrome_small", asset_store=asset_store)

    assert case.payload["display_wave"].shape == (101,)
    assert case.payload["display_white_spd_norm"].shape == (101,)
    assert tuple(case.payload["scene_size"]) == (256, 256)
    assert case.payload["scene_wave"].shape == (101,)
    assert np.isclose(float(case.payload["scene_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["scene_illuminant_energy_norm"].shape == (101,)
    assert case.payload["source_mean_spd_norm"].shape == (101,)
    assert case.payload["source_center_spd_norm"].shape == (101,)
    assert np.isclose(float(case.payload["adjusted_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["adjusted_illuminant_energy_norm"].shape == (101,)
    assert case.payload["adjusted_mean_spd_norm"].shape == (101,)
    assert case.payload["adjusted_center_spd_norm"].shape == (101,)
    assert case.payload["adjusted_mean_rgb_norm"].shape == (3,)


def test_scene_slanted_bar_script_workflow(asset_store) -> None:
    scene = scene_create("slantedBar", 256, 2.6, 2.0, asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    illuminant_roi, _ = scene_plot(scene, "illuminant energy roi", asset_store=asset_store)

    d65_scene = scene_adjust_illuminant(scene.clone(), "D65.mat", asset_store=asset_store)
    d65_illuminant_roi, _ = scene_plot(d65_scene, "illuminant energy roi", asset_store=asset_store)

    alt_scene = scene_create("slantedBar", 128, 3.6, 0.5, asset_store=asset_store)
    alt_luminance = np.asarray(scene_get(alt_scene, "luminance", asset_store=asset_store), dtype=float)

    assert tuple(scene_get(scene, "size")) == (257, 257)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.arange(400.0, 701.0, 10.0, dtype=float))
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert illuminant_roi["energy"].shape == (31,)
    assert illuminant_roi["comment"] is None
    assert np.allclose(illuminant_roi["energy"], illuminant_roi["energy"][0], atol=1e-12, rtol=1e-12)
    assert luminance.shape == (257, 257)
    assert float(np.max(luminance)) > 100.0
    assert float(np.min(luminance)) < 1e-2

    assert np.isclose(scene_get(d65_scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert d65_illuminant_roi["energy"].shape == (31,)
    assert d65_illuminant_roi["comment"] == "D65.mat"
    assert not np.allclose(d65_illuminant_roi["energy"], illuminant_roi["energy"])

    assert tuple(scene_get(alt_scene, "size")) == (129, 129)
    assert np.isclose(scene_get(alt_scene, "fov"), 0.5, atol=1e-12, rtol=1e-12)
    assert np.isclose(scene_get(alt_scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert alt_luminance.shape == (129, 129)


def test_run_python_case_supports_scene_slanted_bar_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_slanted_bar_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (257, 257)
    assert case.payload["wave"].shape == (31,)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["illuminant_energy_roi_norm"].shape == (31,)
    assert case.payload["center_row_luminance_norm"].shape == (257,)
    assert case.payload["center_col_luminance_norm"].shape == (257,)
    assert np.isclose(float(case.payload["d65_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["d65_illuminant_energy_roi_norm"].shape == (31,)
    assert tuple(case.payload["alt_scene_size"]) == (129, 129)
    assert np.isclose(float(case.payload["alt_fov_deg"]), 0.5, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(case.payload["alt_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["alt_center_row_luminance_norm"].shape == (129,)
    assert case.payload["alt_center_col_luminance_norm"].shape == (129,)


def test_scene_harmonics_script_workflow(asset_store) -> None:
    cases = [
        {
            "freq": 1.0,
            "contrast": 1.0,
            "ph": 0.0,
            "ang": 0.0,
            "row": 128,
            "col": 128,
            "GaborFlag": 0.0,
        },
        {
            "freq": np.array([1.0, 5.0], dtype=float),
            "contrast": np.array([0.2, 0.6], dtype=float),
            "ang": np.array([0.0, 0.0], dtype=float),
            "ph": np.array([0.0, np.pi / 3.0], dtype=float),
            "row": 128,
            "col": 128,
            "GaborFlag": 0.0,
        },
        {
            "freq": np.array([2.0, 5.0], dtype=float),
            "contrast": np.array([0.6, 0.6], dtype=float),
            "ang": np.array([np.pi / 4.0, -np.pi / 4.0], dtype=float),
            "ph": np.array([0.0, 0.0], dtype=float),
            "row": 128,
            "col": 128,
            "GaborFlag": 0.0,
        },
        {
            "freq": np.array([5.0, 5.0], dtype=float),
            "contrast": np.array([0.6, 0.6], dtype=float),
            "ang": np.array([np.pi / 4.0, -np.pi / 4.0], dtype=float),
            "ph": np.array([0.0, 0.0], dtype=float),
            "row": 128,
            "col": 128,
            "GaborFlag": 0.0,
        },
    ]

    row_profiles = []
    col_profiles = []
    for params in cases:
        scene = scene_create("harmonic", params, asset_store=asset_store)
        luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)

        assert tuple(scene_get(scene, "size")) == (128, 128)
        assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.arange(400.0, 701.0, 10.0, dtype=float))
        assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
        assert luminance.shape == (128, 128)

        row_profiles.append(luminance[luminance.shape[0] // 2, :])
        col_profiles.append(luminance[:, luminance.shape[1] // 2])

    assert not np.allclose(row_profiles[0], row_profiles[1])
    assert np.allclose(row_profiles[2], col_profiles[2], atol=1e-12, rtol=1e-12)
    assert np.allclose(row_profiles[3], col_profiles[3], atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_harmonics_script_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_harmonics_script_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["scene_sizes"].shape == (4, 2)
    assert np.array_equal(case.payload["scene_sizes"], np.full((4, 2), 128, dtype=int))
    assert case.payload["mean_luminance"].shape == (4,)
    assert np.allclose(case.payload["mean_luminance"], 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["center_row_luminance_norm"].shape == (4, 128)
    assert case.payload["center_col_luminance_norm"].shape == (4, 128)


def test_scene_zone_plate_script_workflow(asset_store) -> None:
    scene = scene_create("zone plate", 96, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    center_row = luminance[luminance.shape[0] // 2, :]
    center_col = luminance[:, luminance.shape[1] // 2]

    assert tuple(scene_get(scene, "size")) == (96, 96)
    assert np.isclose(scene_get(scene, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert photons.shape == (96, 96, 31)
    assert wave.shape == (31,)
    assert float(np.min(photons)) > 0.0
    assert float(np.max(photons)) > float(np.min(photons))
    assert np.isclose(
        scene_get(scene, "mean luminance", asset_store=asset_store),
        100.0,
        atol=1e-8,
        rtol=1e-8,
    )
    assert np.allclose(center_row, center_col, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_zone_plate_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_zone_plate_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (96, 96)
    assert np.isclose(float(case.payload["scene_fov_deg"]), 4.0, atol=1e-12, rtol=1e-12)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (96, 96, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert float(np.min(case.payload["photons"])) > 0.0
    assert float(np.max(case.payload["photons"])) > float(np.min(case.payload["photons"]))


def test_scene_dead_leaves_workflow(asset_store) -> None:
    scene = scene_create("dead leaves", 96, 3.0, {"seed": 12345, "nbr_iter": 1500}, asset_store=asset_store)
    display = display_create("OLED-Sony.mat", asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)

    assert tuple(scene_get(scene, "size")) == (96, 96)
    assert np.isclose(scene_get(scene, "fov"), 10.0, atol=1e-12, rtol=1e-12)
    assert np.array_equal(wave, np.asarray(display_get(display, "wave"), dtype=float).reshape(-1))
    assert photons.shape == (96, 96, wave.size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert float(np.min(photons)) >= 0.0
    assert float(np.max(photons)) > float(np.min(photons))
    assert 0.0 < float(np.mean(luminance > np.mean(luminance))) < 1.0


def test_run_python_case_supports_scene_dead_leaves_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_dead_leaves_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (96, 96)
    assert np.isclose(float(case.payload["scene_fov_deg"]), 10.0, atol=1e-12, rtol=1e-12)
    assert case.payload["wave"].ndim == 1
    assert case.payload["photons"].shape[:2] == (96, 96)
    assert case.payload["photons"].shape[2] == case.payload["wave"].size
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert float(np.min(case.payload["photons"])) >= 0.0
    assert float(np.max(case.payload["photons"])) > float(np.min(case.payload["photons"]))


def test_scene_bar_workflow(asset_store) -> None:
    scene = scene_create("bar", 64, 3, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    center_row = photons.shape[0] // 2

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[center_row, 31:34, :] > photons[center_row, 0:1, :])
    assert np.all(photons[center_row, 31:34, :] > photons[center_row, -1:, :])
    assert np.allclose(luminance[center_row, :31], luminance[center_row, 0], atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[center_row, 34:], luminance[center_row, -1], atol=1e-12, rtol=1e-12)
    assert np.all(luminance[center_row, 31:34] > luminance[center_row, 0])


def test_run_python_case_supports_scene_bar_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_bar_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    center_row = case.payload["photons"].shape[0] // 2
    assert np.all(case.payload["photons"][center_row, 31:34, :] > case.payload["photons"][center_row, 0:1, :])


def test_scene_line_ee_workflow(asset_store) -> None:
    scene = scene_create("line ee", 64, 2, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 + 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])


def test_run_python_case_supports_scene_line_ee_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_line_ee_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 + 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_line_ep_workflow(asset_store) -> None:
    scene = scene_create("line ep", 64, 2, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 + 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])


def test_run_python_case_supports_scene_line_ep_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_line_ep_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 + 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_lineequalphoton_workflow(asset_store) -> None:
    scene = scene_create("lineequalphoton", 64, 2, asset_store=asset_store)
    ep_scene = scene_create("line ep", 64, 2, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    ep_photons = np.asarray(scene_get(ep_scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 + 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])
    assert np.allclose(ep_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_lineequalphoton_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_lineequalphoton_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 + 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_lineee_workflow(asset_store) -> None:
    scene = scene_create("lineee", 64, 2, asset_store=asset_store)
    ee_scene = scene_create("line ee", 64, 2, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    ee_photons = np.asarray(scene_get(ee_scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 + 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])
    assert np.allclose(ee_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_lineee_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_lineee_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 + 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_line_d65_workflow(asset_store) -> None:
    scene = scene_create("lined65", 64, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 - 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])


def test_run_python_case_supports_scene_line_d65_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_line_d65_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 - 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_line_workflow(asset_store) -> None:
    scene = scene_create("line", 64, asset_store=asset_store)
    d65_scene = scene_create("lined65", 64, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    d65_photons = np.asarray(scene_get(d65_scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 - 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])
    assert np.allclose(d65_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_line_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_line_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 - 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_impulse1dd65_workflow(asset_store) -> None:
    scene = scene_create("impulse1dd65", 64, asset_store=asset_store)
    d65_scene = scene_create("lined65", 64, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    d65_photons = np.asarray(scene_get(d65_scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    line_col = photons.shape[1] // 2 - 1

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert photons.shape == (64, 64, 31)
    assert np.all(photons[:, line_col, :] > photons[:, 0, :])
    assert np.allclose(luminance[:, :line_col], np.repeat(luminance[:, 0:1], line_col, axis=1), atol=1e-12, rtol=1e-12)
    assert np.allclose(luminance[:, line_col + 1 :], np.repeat(luminance[:, 0:1], photons.shape[1] - line_col - 1, axis=1), atol=1e-12, rtol=1e-12)
    assert np.all(luminance[:, line_col] > luminance[:, 0])
    assert np.allclose(d65_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_impulse1dd65_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_impulse1dd65_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    line_col = case.payload["photons"].shape[1] // 2 - 1
    assert np.all(case.payload["photons"][:, line_col, :] > case.payload["photons"][:, 0, :])


def test_scene_uniform_monochromatic_workflow(asset_store) -> None:
    scene = scene_create("uniform monochromatic", 550, 12, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (12, 12)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.array([550.0], dtype=float))
    assert photons.shape == (12, 12, 1)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniform_monochromatic_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_monochromatic_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (12, 12)
    assert np.array_equal(np.asarray(case.payload["wave"], dtype=float), np.array([550.0], dtype=float))
    assert case.payload["photons"].shape == (12, 12, 1)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)


def test_scene_uniform_d65_workflow(asset_store) -> None:
    scene = scene_create("uniform d65", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)


def test_run_python_case_supports_scene_uniform_d65_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_d65_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_macbeth_tungsten_workflow(asset_store) -> None:
    scene = scene_create("macbethtungsten", asset_store=asset_store)
    default_scene = scene_create(asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    illuminant_energy = np.asarray(scene_get(scene, "illuminant energy"), dtype=float)
    default_illuminant_energy = np.asarray(scene_get(default_scene, "illuminant energy"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 96)
    assert photons.shape == (64, 96, 31)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.arange(400.0, 701.0, 10.0, dtype=float))
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert scene_get(scene, "illuminant comment") == "tungsten"
    assert illuminant_energy.shape == (31,)
    assert not np.allclose(illuminant_energy, illuminant_energy[0], atol=1e-8, rtol=1e-12)
    assert not np.allclose(illuminant_energy, default_illuminant_energy, atol=1e-8, rtol=1e-8)


def test_run_python_case_supports_scene_macbeth_tungsten_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_macbeth_tungsten", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 96, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)


def test_scene_empty_workflow(asset_store) -> None:
    scene = scene_create("empty", asset_store=asset_store)
    custom_wave = np.array([500.0, 600.0, 700.0], dtype=float)
    shorthand = scene_create("empty", custom_wave, asset_store=asset_store)
    custom = scene_create("empty", None, custom_wave, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    illuminant_energy = np.asarray(scene_get(scene, "illuminant energy"), dtype=float)
    shorthand_photons = np.asarray(scene_get(shorthand, "photons"), dtype=float)
    custom_photons = np.asarray(scene_get(custom, "photons"), dtype=float)
    shorthand_illuminant_energy = np.asarray(scene_get(shorthand, "illuminant energy"), dtype=float)
    custom_illuminant_energy = np.asarray(scene_get(custom, "illuminant energy"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 96)
    assert photons.shape == (64, 96, 31)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.arange(400.0, 701.0, 10.0, dtype=float))
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(photons, 0.0, atol=0.0, rtol=0.0)
    assert illuminant_energy.shape == (31,)
    assert np.all(illuminant_energy > 0.0)
    assert tuple(scene_get(shorthand, "size")) == (64, 96)
    assert shorthand_photons.shape == (64, 96, 3)
    assert np.array_equal(np.asarray(scene_get(shorthand, "wave"), dtype=float), custom_wave)
    assert np.isclose(scene_get(shorthand, "mean luminance", asset_store=asset_store), 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(shorthand_photons, custom_photons, atol=0.0, rtol=0.0)
    assert np.allclose(shorthand_illuminant_energy, custom_illuminant_energy, atol=0.0, rtol=0.0)
    assert tuple(scene_get(custom, "size")) == (64, 96)
    assert custom_photons.shape == (64, 96, 3)
    assert np.array_equal(np.asarray(scene_get(custom, "wave"), dtype=float), custom_wave)
    assert np.isclose(scene_get(custom, "mean luminance", asset_store=asset_store), 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(custom_photons, 0.0, atol=0.0, rtol=0.0)
    assert custom_illuminant_energy.shape == (3,)
    assert np.all(custom_illuminant_energy > 0.0)


def test_run_python_case_supports_scene_empty_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_empty_small", asset_store=asset_store)

    assert np.array_equal(np.asarray(case.payload["wave"], dtype=float), np.arange(400.0, 701.0, 10.0, dtype=float))
    assert np.isclose(float(case.payload["mean_luminance"]), 0.0, atol=1e-12, rtol=1e-12)
    assert case.payload["illuminant_energy"].shape == (31,)
    assert np.all(case.payload["illuminant_energy"] > 0.0)
    assert np.isclose(float(case.payload["photon_sum"]), 0.0, atol=0.0, rtol=0.0)
    assert np.isclose(float(case.payload["photon_max"]), 0.0, atol=0.0, rtol=0.0)


def test_scene_lstar_workflow(asset_store) -> None:
    scene = scene_create("lstar", np.array([80, 10], dtype=int), 20, 1, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    bar_means = np.array([np.mean(luminance[:, start : start + 10]) for start in range(0, 200, 10)], dtype=float)

    assert tuple(scene_get(scene, "size")) == (80, 200)
    assert photons.shape == (80, 200, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.all(np.diff(bar_means) > 0.0)
    assert np.allclose(photons[:, :10, :], photons[:, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(photons[:, :10, :], photons[:, -10:, :], atol=1e-8, rtol=1e-12)


def test_run_python_case_supports_scene_lstar_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_lstar_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (80, 200)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["bar_means_norm"].shape == (20,)
    assert case.payload["center_row_norm"].shape == (200,)
    assert np.all(np.diff(case.payload["bar_means_norm"]) > 0.0)
    assert np.isclose(float(case.payload["bar_means_norm"][-1]), 1.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(float(np.max(case.payload["center_row_norm"])), 1.0, atol=1e-12, rtol=1e-12)


def test_scene_hdr_workflow(asset_store) -> None:
    scene = scene_create("hdr", asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)

    assert tuple(scene_get(scene, "size")) == (384, 384)
    assert photons.shape == (384, 384, 31)
    assert np.asarray(scene_get(scene, "wave"), dtype=float).shape == (31,)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.isclose(float(np.min(luminance)), 1.0e-5, atol=1e-10, rtol=1e-8)
    assert float(np.max(luminance)) > float(np.mean(luminance))


def test_run_python_case_supports_scene_hdr_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_hdr_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (384, 384)
    assert case.payload["wave"].shape == (31,)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["mean_spd_norm"].shape == (31,)
    assert case.payload["row_profile_summaries"].shape == (3, 6)
    assert np.isclose(float(np.max(case.payload["mean_spd_norm"])), 1.0, atol=1e-12, rtol=1e-12)
    assert np.all(case.payload["row_profile_summaries"][:, 2] > 0.0)
    assert np.all(case.payload["row_profile_summaries"][:, 3] <= case.payload["row_profile_summaries"][:, 4])
    assert np.all(case.payload["row_profile_summaries"][:, 4] <= case.payload["row_profile_summaries"][:, 5])


def test_scene_uniform_ep_workflow(asset_store) -> None:
    scene = scene_create("uniform ep", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)


def test_run_python_case_supports_scene_uniform_ep_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_ep_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniformephoton_workflow(asset_store) -> None:
    scene = scene_create("uniformephoton", 24, asset_store=asset_store)
    ep_scene = scene_create("uniform ep", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    ep_photons = np.asarray(scene_get(ep_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(ep_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniformephoton_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniformephoton_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniformep_workflow(asset_store) -> None:
    scene = scene_create("uniformep", 24, asset_store=asset_store)
    ep_scene = scene_create("uniform ep", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    ep_photons = np.asarray(scene_get(ep_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(ep_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniformep_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniformep_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniformequalphoton_workflow(asset_store) -> None:
    scene = scene_create("uniformequalphoton", 24, asset_store=asset_store)
    ep_scene = scene_create("uniform ep", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    ep_photons = np.asarray(scene_get(ep_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(ep_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniformequalphoton_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniformequalphoton_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniformequalphotons_workflow(asset_store) -> None:
    scene = scene_create("uniformequalphotons", 24, asset_store=asset_store)
    ep_scene = scene_create("uniform ep", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    ep_photons = np.asarray(scene_get(ep_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(ep_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniformequalphotons_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniformequalphotons_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniformbb_workflow(asset_store) -> None:
    scene = scene_create("uniformbb", 16, 4500, asset_store=asset_store)
    bb_scene = scene_create("uniform bb", 16, 4500, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    bb_photons = np.asarray(scene_get(bb_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (16, 16)
    assert photons.shape == (16, 16, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(bb_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniformbb_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniformbb_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (16, 16)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (16, 16, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniformblackbody_workflow(asset_store) -> None:
    scene = scene_create("uniformblackbody", 16, 4500, asset_store=asset_store)
    bb_scene = scene_create("uniform bb", 16, 4500, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    bb_photons = np.asarray(scene_get(bb_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (16, 16)
    assert photons.shape == (16, 16, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(bb_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniformblackbody_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniformblackbody_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (16, 16)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (16, 16, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniform_ee_workflow(asset_store) -> None:
    scene = scene_create("uniform", 24, asset_store=asset_store)
    alias_scene = scene_create("uniformEE", 24, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    alias_photons = np.asarray(scene_get(alias_scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (24, 24)
    assert photons.shape == (24, 24, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(photons[0, 0, :], photons[0, 0, 0], atol=1e-8, rtol=1e-12)
    assert np.allclose(alias_photons, photons, atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniform_ee_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_ee_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (24, 24)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (24, 24, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)
    assert not np.allclose(case.payload["photons"][0, 0, :], case.payload["photons"][0, 0, 0], atol=1e-8, rtol=1e-12)


def test_scene_uniform_ee_specify_workflow(asset_store) -> None:
    wave = np.arange(380.0, 721.0, 10.0, dtype=float)
    scene = scene_create("uniformEESpecify", 128, wave, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (128, 128)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), wave)
    assert photons.shape == (128, 128, wave.size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons, photons[0:1, 0:1, :], atol=1e-12, rtol=1e-12)


def test_run_python_case_supports_scene_uniform_ee_specify_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_uniform_ee_specify_small", asset_store=asset_store)

    wave = np.arange(380.0, 721.0, 10.0, dtype=float)
    assert tuple(case.payload["scene_size"]) == (128, 128)
    assert np.array_equal(np.asarray(case.payload["wave"], dtype=float), wave)
    assert case.payload["photons"].shape == (128, 128, wave.size)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"], case.payload["photons"][0:1, 0:1, :], atol=1e-12, rtol=1e-12)


def test_scene_exponential_intensity_ramp_workflow(asset_store) -> None:
    scene = scene_create("exponential intensity ramp", 64, 256, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert photons.shape == (64, 64, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(photons[0, :, :], photons[-1, :, :], atol=1e-12, rtol=1e-12)
    assert np.all(np.diff(photons[0, :, 0]) > 0.0)
    assert np.isclose(photons[0, -1, 0] / photons[0, 0, 0], 256.0, atol=1e-10, rtol=1e-10)


def test_run_python_case_supports_scene_exponential_intensity_ramp_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_exponential_intensity_ramp_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(case.payload["photons"][0, :, :], case.payload["photons"][-1, :, :], atol=1e-12, rtol=1e-12)
    assert np.all(np.diff(case.payload["photons"][0, :, 0]) > 0.0)
    assert np.isclose(case.payload["photons"][0, -1, 0] / case.payload["photons"][0, 0, 0], 256.0, atol=1e-10, rtol=1e-10)


def test_scene_linear_intensity_ramp_workflow(asset_store) -> None:
    scene = scene_create("linear intensity ramp", 64, 256, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert photons.shape == (64, 64, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.all(np.diff(photons[:, :, 0], axis=1) > 0.0)
    assert np.isclose(photons[0, -1, 0] / photons[0, 0, 0], 256.0, atol=1e-10, rtol=1e-10)
    assert np.all(np.diff(photons[:, -1, 0]) < 0.0)
    assert np.all(np.diff(photons[:, 0, 0]) > 0.0)


def test_run_python_case_supports_scene_linear_intensity_ramp_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_linear_intensity_ramp_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.all(np.diff(case.payload["photons"][:, :, 0], axis=1) > 0.0)
    assert np.isclose(case.payload["photons"][0, -1, 0] / case.payload["photons"][0, 0, 0], 256.0, atol=1e-10, rtol=1e-10)
    assert np.all(np.diff(case.payload["photons"][:, -1, 0]) < 0.0)
    assert np.all(np.diff(case.payload["photons"][:, 0, 0]) > 0.0)


def test_scene_rings_rays_workflow(asset_store) -> None:
    scene = scene_create("rings rays", 8, 64, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    plane = photons[:, :, 0]
    center = plane.shape[0] // 2

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert photons.shape == (64, 64, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(plane[center, :], plane[:, center], atol=1e-12, rtol=1e-12)
    assert np.min(plane) > 0.0
    assert np.max(plane) > np.min(plane)


def test_run_python_case_supports_scene_rings_rays_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_rings_rays_small", asset_store=asset_store)
    plane = case.payload["photons"][:, :, 0]
    center = plane.shape[0] // 2

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(plane[center, :], plane[:, center], atol=1e-12, rtol=1e-12)
    assert np.min(plane) > 0.0
    assert np.max(plane) > np.min(plane)


def test_scene_point_array_workflow(asset_store) -> None:
    scene = scene_create("point array", 64, 16, "ep", 1, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    plane = photons[:, :, 0]
    coords = np.argwhere(np.isclose(plane, plane.max()))
    unique_x = np.unique(coords[:, 1])
    unique_y = np.unique(coords[:, 0])

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert photons.shape == (64, 64, 31)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.count_nonzero(plane) == 16
    assert np.array_equal(unique_x, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(unique_y, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(np.unique(np.diff(unique_x)), np.array([16], dtype=int))
    assert np.array_equal(np.unique(np.diff(unique_y)), np.array([16], dtype=int))


def test_run_python_case_supports_scene_point_array_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_point_array_small", asset_store=asset_store)
    plane = case.payload["photons"][:, :, 0]
    coords = np.argwhere(np.isclose(plane, plane.max()))
    unique_x = np.unique(coords[:, 1])
    unique_y = np.unique(coords[:, 0])

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.count_nonzero(plane) == 16
    assert np.array_equal(unique_x, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(unique_y, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(np.unique(np.diff(unique_x)), np.array([16], dtype=int))
    assert np.array_equal(np.unique(np.diff(unique_y)), np.array([16], dtype=int))


def test_scene_grid_lines_workflow(asset_store) -> None:
    scene = scene_create("grid lines", 64, 16, "ep", 1, asset_store=asset_store)
    plane = np.asarray(scene_get(scene, "photons"), dtype=float)[:, :, 0]
    full_hi_rows = np.where(np.all(np.isclose(plane, plane.max()), axis=1))[0]
    full_hi_cols = np.where(np.all(np.isclose(plane, plane.max()), axis=0))[0]

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert plane.shape == (64, 64)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, atol=1e-8, rtol=1e-8)
    assert np.unique(plane).size == 2
    assert np.count_nonzero(np.isclose(plane, plane.max())) == 496
    assert np.array_equal(full_hi_rows, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(full_hi_cols, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(np.unique(np.diff(full_hi_rows)), np.array([16], dtype=int))
    assert np.array_equal(np.unique(np.diff(full_hi_cols)), np.array([16], dtype=int))


def test_run_python_case_supports_scene_grid_lines_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_grid_lines_small", asset_store=asset_store)
    plane = case.payload["photons"][:, :, 0]
    full_hi_rows = np.where(np.all(np.isclose(plane, plane.max()), axis=1))[0]
    full_hi_cols = np.where(np.all(np.isclose(plane, plane.max()), axis=0))[0]

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert np.isclose(float(case.payload["mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.unique(plane).size == 2
    assert np.count_nonzero(np.isclose(plane, plane.max())) == 496
    assert np.array_equal(full_hi_rows, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(full_hi_cols, np.array([7, 23, 39, 55], dtype=int))
    assert np.array_equal(np.unique(np.diff(full_hi_rows)), np.array([16], dtype=int))
    assert np.array_equal(np.unique(np.diff(full_hi_cols)), np.array([16], dtype=int))


def test_scene_white_noise_workflow(asset_store) -> None:
    scene = scene_create("white noise", 128, 20, asset_store=asset_store)
    scene_repeat = scene_create("white noise", 128, 20, asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    photons_repeat = np.asarray(scene_get(scene_repeat, "photons"), dtype=float)
    plane = photons[:, :, 0]
    plane_mean = float(np.mean(plane))
    normalized_plane = plane / plane_mean

    assert tuple(scene_get(scene, "size")) == (128, 128)
    assert photons.shape == (128, 128, 31)
    assert np.array_equal(photons, photons_repeat)
    assert float(scene_get(scene, "fov")) == pytest.approx(1.0, abs=1e-12)
    assert float(scene_get(scene, "mean luminance", asset_store=asset_store)) == pytest.approx(100.11593267709446)
    assert normalized_plane.std() == pytest.approx(0.19899359211447812)
    assert np.percentile(normalized_plane, 50.0) == pytest.approx(0.9977061714414758)


def test_run_python_case_supports_scene_white_noise_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_white_noise_small", asset_store=asset_store)

    assert tuple(case.payload["scene_size"]) == (128, 128)
    assert case.payload["wave"].shape == (31,)
    assert tuple(case.payload["photons_shape"]) == (128, 128, 31)
    assert float(case.payload["fov_deg"]) == pytest.approx(1.0, abs=1e-12)
    assert float(case.payload["mean_luminance"]) == pytest.approx(100.11593267709446)
    assert case.payload["pattern_stats_norm"].shape == (3,)
    assert case.payload["pattern_stats_norm"][1] == pytest.approx(0.19899359211447812)
    assert case.payload["pattern_percentiles_norm"].shape == (7,)
    assert case.payload["pattern_percentiles_norm"][3] == pytest.approx(0.9977061714414758)
    assert case.payload["mean_spectrum_norm"].shape == (31,)
    assert np.isclose(np.max(case.payload["mean_spectrum_norm"]), 1.0, atol=1e-12, rtol=0.0)


def test_scene_disk_array_workflow(asset_store) -> None:
    scene = scene_create("disk array", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)
    plane = np.asarray(scene_get(scene, "photons"), dtype=float)[:, :, 0]
    labels, n_components = ndimage.label(plane > 0.0)
    centroids = np.asarray(ndimage.center_of_mass(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)
    areas = np.asarray(ndimage.sum(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert plane.shape == (64, 64)
    assert float(scene_get(scene, "mean luminance", asset_store=asset_store)) == pytest.approx(100.0, abs=1e-8)
    assert np.unique(plane).size == 2
    assert np.count_nonzero(plane) == 772
    assert n_components == 4
    assert np.array_equal(np.round(centroids).astype(int), np.array([[20, 20], [20, 41], [41, 20], [41, 41]], dtype=int))
    assert np.array_equal(areas.astype(int), np.array([193, 193, 193, 193], dtype=int))


def test_run_python_case_supports_scene_disk_array_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_disk_array_small", asset_store=asset_store)
    plane = case.payload["photons"][:, :, 0]
    labels, n_components = ndimage.label(plane > 0.0)
    centroids = np.asarray(ndimage.center_of_mass(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)
    areas = np.asarray(ndimage.sum(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert float(case.payload["mean_luminance"]) == pytest.approx(100.0, abs=1e-8)
    assert np.unique(plane).size == 2
    assert np.count_nonzero(plane) == 772
    assert n_components == 4
    assert np.array_equal(np.round(centroids).astype(int), np.array([[20, 20], [20, 41], [41, 20], [41, 41]], dtype=int))
    assert np.array_equal(areas.astype(int), np.array([193, 193, 193, 193], dtype=int))


def test_scene_square_array_workflow(asset_store) -> None:
    scene = scene_create("square array", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)
    plane = np.asarray(scene_get(scene, "photons"), dtype=float)[:, :, 0]
    labels, n_components = ndimage.label(plane > 0.0)
    centroids = np.asarray(ndimage.center_of_mass(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)
    areas = np.asarray(ndimage.sum(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)

    assert tuple(scene_get(scene, "size")) == (64, 64)
    assert plane.shape == (64, 64)
    assert float(scene_get(scene, "mean luminance", asset_store=asset_store)) == pytest.approx(100.0, abs=1e-8)
    assert np.unique(plane).size == 2
    assert np.count_nonzero(plane) == 256
    assert n_components == 4
    assert np.allclose(centroids, np.array([[19.5, 19.5], [19.5, 40.5], [40.5, 19.5], [40.5, 40.5]], dtype=float))
    assert np.array_equal(areas.astype(int), np.array([64, 64, 64, 64], dtype=int))


def test_run_python_case_supports_scene_square_array_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_square_array_small", asset_store=asset_store)
    plane = case.payload["photons"][:, :, 0]
    labels, n_components = ndimage.label(plane > 0.0)
    centroids = np.asarray(ndimage.center_of_mass(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)
    areas = np.asarray(ndimage.sum(plane > 0.0, labels, range(1, n_components + 1)), dtype=float)

    assert tuple(case.payload["scene_size"]) == (64, 64)
    assert case.payload["wave"].shape == (31,)
    assert case.payload["photons"].shape == (64, 64, 31)
    assert float(case.payload["mean_luminance"]) == pytest.approx(100.0, abs=1e-8)
    assert np.unique(plane).size == 2
    assert np.count_nonzero(plane) == 256
    assert n_components == 4
    assert np.allclose(centroids, np.array([[19.5, 19.5], [19.5, 40.5], [40.5, 19.5], [40.5, 40.5]], dtype=float))
    assert np.array_equal(areas.astype(int), np.array([64, 64, 64, 64], dtype=int))


def test_surface_munsell_script_workflow(asset_store) -> None:
    munsell = asset_store.load_mat("data/surfaces/charts/munsell.mat")["munsell"]
    xyz = np.asarray(munsell.XYZ, dtype=float)
    lab = np.asarray(munsell.LAB, dtype=float)
    wavelength = np.asarray(munsell.wavelength, dtype=float).reshape(-1)
    illuminant = np.asarray(munsell.illuminant, dtype=float).reshape(-1)
    hues = np.asarray(munsell.hue, dtype=object).reshape(-1)
    values = np.asarray(munsell.value, dtype=float).reshape(-1)
    angles = np.asarray(munsell.angle, dtype=float).reshape(-1)

    xyz_image = xyz.reshape(261, 9, 3)
    srgb = np.asarray(xyz_to_srgb(xyz_image), dtype=float)
    xy = np.asarray(chromaticity_xy(xyz), dtype=float)

    assert xyz.shape == (2349, 3)
    assert lab.shape == (2349, 3)
    assert wavelength.shape == (47,)
    assert illuminant.shape == (47,)
    assert srgb.shape == (261, 9, 3)
    assert xy.shape == (2349, 2)
    assert np.all(np.isfinite(srgb))
    assert np.all(np.isfinite(xy))
    assert len(hues[:45]) == 45
    assert values[:45].shape == (45,)
    assert angles[:45].shape == (45,)
    assert hues[0] == ".00R"
    assert np.isclose(values[0], 1.0)
    assert np.isclose(angles[0], 0.0)


def test_run_python_case_supports_surface_munsell_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("surface_munsell_small", asset_store=asset_store)

    assert tuple(case.payload["xyz_shape"]) == (2349, 3)
    assert tuple(case.payload["lab_shape"]) == (2349, 3)
    assert case.payload["wavelength"].shape == (47,)
    assert case.payload["illuminant_norm"].shape == (47,)
    assert tuple(case.payload["srgb_grid_shape"]) == (261, 9, 3)
    assert case.payload["srgb_mean_rgb"].shape == (3,)
    assert case.payload["srgb_selected_rgb"].shape == (4, 3)
    assert case.payload["xy_mean"].shape == (2,)
    assert case.payload["xy_bounds"].shape == (2, 2)
    assert case.payload["lab_mean"].shape == (3,)
    assert case.payload["lab_bounds"].shape == (2, 3)
    assert len(case.payload["first45_hues"]) == 45
    assert case.payload["first45_values"].shape == (45,)
    assert case.payload["first45_angles"].shape == (45,)


def test_scene_demo_script_workflow(asset_store) -> None:
    scene_macbeth = scene_create("macbethd65", asset_store=asset_store)
    macbeth_wave = np.asarray(scene_get(scene_macbeth, "wave"), dtype=float).reshape(-1)
    macbeth_luminance = np.asarray(scene_get(scene_macbeth, "luminance", asset_store=asset_store), dtype=float)
    macbeth_photons = np.asarray(scene_get(scene_macbeth, "photons"), dtype=float)
    macbeth_mean_photons = np.mean(macbeth_photons, axis=(0, 1), dtype=float)
    scene_macbeth_fov20 = scene_set(scene_macbeth.clone(), "fov", 20.0)

    assert tuple(scene_get(scene_macbeth, "size")) == macbeth_luminance.shape
    assert macbeth_photons.shape[:2] == macbeth_luminance.shape
    assert macbeth_photons.shape[2] == macbeth_wave.size
    assert np.isclose(np.max(macbeth_photons), 1.3119e16, rtol=1e-4, atol=1e-8)
    assert np.isclose(np.mean(macbeth_mean_photons), 3.7624e15, rtol=1e-4, atol=1e-8)
    assert float(scene_get(scene_macbeth_fov20, "fov")) == pytest.approx(20.0, abs=1e-12)

    scene_test = scene_create("freq orient pattern", asset_store=asset_store)
    scene_test_size = np.asarray(scene_get(scene_test, "size"), dtype=int)
    scene_test_luminance = np.asarray(scene_get(scene_test, "luminance", asset_store=asset_store), dtype=float)
    scene_test_support = dict(scene_get(scene_test, "spatial support linear", "mm"))
    luminance_line, _ = scene_plot(scene_test, "luminance hline", scene_test_size, asset_store=asset_store)
    rows_half = int(round(float(scene_get(scene_test, "rows")) / 2.0))
    radiance_line, _ = scene_plot(scene_test, "radiance hline", [1, rows_half], asset_store=asset_store)

    assert tuple(scene_test_size) == scene_test_luminance.shape
    assert rows_half == 128
    assert scene_test_support["x"].shape == (scene_test_size[1],)
    assert np.allclose(luminance_line["data"], scene_test_luminance[-1, :], atol=1e-12, rtol=1e-12)
    assert np.asarray(radiance_line["data"], dtype=float).shape[-1] == scene_test_size[1]


def test_scene_support_wrappers_match_scene_get(asset_store, tmp_path: Path) -> None:
    scene = scene_create("freq orient pattern", asset_store=asset_store)

    spatial = sceneSpatialSupport(scene, "mm")
    expected_spatial = scene_get(scene, "spatial support linear", "mm")
    assert np.allclose(np.asarray(spatial["x"], dtype=float), np.asarray(expected_spatial["x"], dtype=float))
    assert np.allclose(np.asarray(spatial["y"], dtype=float), np.asarray(expected_spatial["y"], dtype=float))

    frequency = sceneFrequencySupport(scene, "cpd")
    expected_frequency = scene_get(scene, "frequency resolution", "cpd")
    assert np.allclose(np.asarray(frequency["fx"], dtype=float), np.asarray(expected_frequency["fx"], dtype=float))
    assert np.allclose(np.asarray(frequency["fy"], dtype=float), np.asarray(expected_frequency["fy"], dtype=float))

    no_fov = scene.clone()
    no_fov.fields.pop("fov_deg", None)
    initialized = sceneInitSpatial(no_fov)
    assert np.isclose(float(scene_get(initialized, "fov")), 10.0, atol=1e-12, rtol=0.0)

    output = Path(sceneSaveImage(scene, tmp_path / "scene_support_demo"))
    assert output.name == "scene_support_demo.png"
    saved = iio.imread(output)
    assert saved.shape[:2] == tuple(np.asarray(scene_get(scene, "size"), dtype=int))
    assert saved.shape[2] == 3


def test_scene_description_list_and_thumbnail_wrappers(asset_store, tmp_path: Path) -> None:
    scene = scene_create("uniform ee", 8, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    scene.name = "thumbnail-scene"

    description = sceneDescription(scene, asset_store=asset_store)
    listing = sceneList()
    thumbnail = Path(
        sceneThumbnail(
            scene,
            "row size",
            24,
            "force square",
            True,
            "output file name",
            tmp_path / "scene_thumb",
            asset_store=asset_store,
        )
    )

    assert "Row,Col:" in description
    assert "Wave:" in description
    assert "uniform equal energy" in listing.lower()
    assert "scene from file" in listing.lower()
    assert thumbnail.name == "scene_thumb.png"
    written = np.asarray(iio.imread(thumbnail), dtype=float)
    assert written.shape[:2] == (24, 24)
    assert written.shape[2] == 3


def test_scene_helper_wrappers_match_legacy_contract(asset_store, tmp_path: Path) -> None:
    scene = scene_create("uniform ee", 4, np.array([500.0, 600.0, 700.0], dtype=float), asset_store=asset_store)
    photons = np.arange(1, 4 * 5 * 3 + 1, dtype=float).reshape(4, 5, 3)
    scene = scene_set(scene, "photons", photons)
    scene = scene_set(scene, "depth map", np.arange(1, 21, dtype=float).reshape(4, 5))
    illuminant = np.arange(1, 4 * 5 * 3 + 1, dtype=float).reshape(4, 5, 3) / 10.0
    scene = scene_set(scene, "illuminant photons", illuminant)

    cropped, rect = sceneCrop(scene, [2, 2, 2, 1], asset_store=asset_store)
    assert np.array_equal(rect, np.array([2, 2, 2, 1], dtype=int))
    assert np.array_equal(np.asarray(scene_get(cropped, "photons"), dtype=float), photons[1:3, 1:4, :])
    assert np.array_equal(np.asarray(scene_get(cropped, "depth map"), dtype=float), np.arange(1, 21, dtype=float).reshape(4, 5)[1:3, 1:4])
    assert np.array_equal(np.asarray(scene_get(cropped, "illuminant photons"), dtype=float), illuminant[1:3, 1:4, :])
    assert tuple(scene_get(cropped, "size")) == (2, 3)
    assert np.array_equal(np.asarray(cropped.metadata["rect"], dtype=int), rect)
    assert np.asarray(scene_get(cropped, "luminance", asset_store=asset_store), dtype=float).shape == (2, 3)

    extracted = sceneExtractWaveband(scene, np.array([550.0, 650.0], dtype=float), asset_store=asset_store)
    expected_photons = np.stack(
        [
            np.interp(np.array([550.0, 650.0], dtype=float), np.array([500.0, 600.0, 700.0], dtype=float), photons[row, col, :])
            for row in range(photons.shape[0])
            for col in range(photons.shape[1])
        ],
        axis=0,
    ).reshape(photons.shape[0], photons.shape[1], 2)
    expected_illuminant = np.stack(
        [
            np.interp(np.array([550.0, 650.0], dtype=float), np.array([500.0, 600.0, 700.0], dtype=float), illuminant[row, col, :])
            for row in range(illuminant.shape[0])
            for col in range(illuminant.shape[1])
        ],
        axis=0,
    ).reshape(illuminant.shape[0], illuminant.shape[1], 2)
    assert np.array_equal(np.asarray(scene_get(extracted, "wave"), dtype=float), np.array([550.0, 650.0], dtype=float))
    assert np.allclose(np.asarray(scene_get(extracted, "photons"), dtype=float), expected_photons)
    assert np.allclose(np.asarray(scene_get(extracted, "illuminant photons"), dtype=float), expected_illuminant)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float), np.array([500.0, 600.0, 700.0], dtype=float))

    scene_for_shift = scene_set(scene.clone(), "fov", 5.0)
    translated = sceneTranslate(scene_for_shift, [1.0, 0.0], 0.5)
    translated_photons = np.asarray(scene_get(translated, "photons"), dtype=float)
    expected_translated = np.full_like(photons, 0.5)
    expected_translated[:, 1:, :] = photons[:, :-1, :]
    assert np.allclose(translated_photons, expected_translated)
    assert np.array_equal(np.asarray(scene_get(scene_for_shift, "photons"), dtype=float), photons)


def test_scene_legacy_arithmetic_wrappers(asset_store) -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 4, wave, asset_store=asset_store)
    oi = oi_create()

    sample_spacing = np.asarray(scene_get(scene, "sample spacing", "m"), dtype=float).reshape(-1)
    adjusted, new_distance = sceneAdjustPixelSize(scene, oi, sample_spacing[0] * 2.0)
    assert np.isclose(float(scene_get(adjusted, "distance")), new_distance)
    assert np.isclose(new_distance, float(scene_get(scene, "distance")) * 2.0)

    base_energy = np.asarray(scene_get(scene, "energy"), dtype=float)
    scaled_scene, returned = sceneSPDScale(scene, np.array([1.0, 2.0, 4.0], dtype=float), "*", True, asset_store=asset_store)
    assert isinstance(returned, np.ndarray)
    assert np.allclose(
        np.asarray(scene_get(scaled_scene, "energy"), dtype=float),
        base_energy * np.array([1.0, 2.0, 4.0], dtype=float).reshape(1, 1, -1),
    )
    assert np.allclose(
        np.asarray(scene_get(scaled_scene, "illuminant energy"), dtype=float),
        np.asarray(scene_get(scene, "illuminant energy"), dtype=float),
    )

    reflectance = np.array([0.25, 0.5, 0.75], dtype=float)
    reflected = sceneAdjustReflectance(scene, reflectance)
    expected_reflected = np.broadcast_to(
        (reflectance * np.asarray(scene_get(scene, "illuminant photons"), dtype=float)).reshape(1, 1, -1),
        np.asarray(scene_get(scene, "photons"), dtype=float).shape,
    )
    assert np.allclose(np.asarray(scene_get(reflected, "photons"), dtype=float), expected_reflected)

    illuminant = np.asarray(scene_get(scene, "illuminant photons"), dtype=float).reshape(-1)
    dimmed = scene_set(
        scene.clone(),
        "photons",
        np.broadcast_to((0.45 * illuminant).reshape(1, 1, -1), np.asarray(scene_get(scene, "photons"), dtype=float).shape).copy(),
    )
    scaled_illuminant = sceneIlluminantScale(dimmed)
    assert np.allclose(np.asarray(scene_get(scaled_illuminant, "illuminant photons"), dtype=float), 0.5 * illuminant)
    assert np.isclose(float(np.asarray(scene_get(scaled_illuminant, "known reflectance"), dtype=float)[0]), 0.9)

    grid_source = scene_create("uniform ee", 8, wave, asset_store=asset_store)
    gridded = sceneAddGrid(grid_source, [4, 4], 1)
    gridded_photons = np.asarray(scene_get(gridded, "photons"), dtype=float)
    assert np.allclose(gridded_photons[0, :, :], 0.0)
    assert np.allclose(gridded_photons[-1, :, :], 0.0)
    assert np.allclose(gridded_photons[:, 0, :], 0.0)
    assert np.allclose(gridded_photons[:, -1, :], 0.0)
    assert np.allclose(gridded_photons[3, :, :], 0.0)
    assert np.allclose(gridded_photons[:, 3, :], 0.0)


def test_scene_pattern_legacy_wrappers(asset_store) -> None:
    fot = FOTParams()
    assert np.allclose(np.asarray(fot["angles"], dtype=float), np.linspace(0.0, np.pi / 2.0, 8))
    assert np.array_equal(np.asarray(fot["freqs"], dtype=float), np.arange(1.0, 9.0, dtype=float))
    assert fot["blockSize"] == 32
    assert np.isclose(float(fot["contrast"]), 1.0)

    gabor = gaborP(orientation=0.25, spread=7)
    assert np.isclose(float(gabor["orientation"]), 0.25)
    assert np.isclose(float(gabor["phase"]), np.pi / 2.0)
    assert int(gabor["imagesize"]) == 65
    assert np.isclose(float(gabor["spread"]), 7.0)

    checker = ieCheckerboard(2, 2)
    expected_checker = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    assert np.array_equal(checker, expected_checker)

    moire = MOTarget("squareim", {"sceneSize": 8, "f": 1.0 / 80.0})
    assert moire.shape == (8, 8, 3)
    assert np.array_equal(moire[:, :, 0], moire[:, :, 1])
    assert np.array_equal(moire[:, :, 1], moire[:, :, 2])
    assert set(np.unique(moire[:, :, 0]).tolist()).issubset({0.0, 1.0})

    base = scene_create("uniform ee", 4, np.array([500.0, 600.0], dtype=float), asset_store=asset_store)
    base = scene_set(base, "fov", 7.0)
    ramp = sceneRamp(base, 16, 64.0, asset_store=asset_store)
    ramp_direct = scene_ramp(base, 16, 64.0, asset_store=asset_store)
    ramp_from_create = scene_create("ramp", 16, 64.0, np.array([500.0, 600.0], dtype=float), asset_store=asset_store)

    assert ramp.name == "ramp DR 64.0"
    assert tuple(scene_get(ramp, "size")) == (16, 16)
    assert np.isclose(float(scene_get(ramp, "fov")), 7.0)
    assert np.array_equal(np.asarray(scene_get(ramp, "wave"), dtype=float), np.asarray(scene_get(base, "wave"), dtype=float))
    assert np.array_equal(np.asarray(scene_get(ramp, "photons"), dtype=float), np.asarray(scene_get(ramp_direct, "photons"), dtype=float))
    assert tuple(scene_get(ramp_from_create, "size")) == (16, 16)
    assert ramp_from_create.name == "ramp DR 64.0"


def test_scene_geometry_resample_and_noise_wrappers(asset_store) -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 4, wave, asset_store=asset_store)
    photons = np.arange(1, 4 * 5 * wave.size + 1, dtype=float).reshape(4, 5, wave.size)
    scene = scene_set(scene, "photons", photons)
    scene = scene_set(scene, "fov", 5.0)

    no_distance = scene.clone()
    no_distance.fields.pop("distance_m", None)
    initialized = sceneInitGeometry(no_distance)
    assert np.isclose(float(scene_get(initialized, "distance")), 1.2, atol=1e-12, rtol=0.0)
    assert tuple(scene_get(initialized, "size")) == (4, 5)

    support = scene_get(scene, "spatial support linear", "m")
    x_support = np.asarray(support["x"], dtype=float)
    y_support = np.asarray(support["y"], dtype=float)
    target_dx = float(np.mean(np.diff(x_support))) / 2.0
    resampled = sceneSpatialResample(scene, target_dx, "m")
    resampled_support = scene_get(resampled, "spatial support linear", "m")
    expected_cols = int(np.floor((x_support[-1] - x_support[0]) / target_dx + 1e-12)) + 1
    expected_rows = int(np.floor((y_support[-1] - y_support[0]) / target_dx + 1e-12)) + 1

    assert tuple(scene_get(resampled, "size")) == (expected_rows, expected_cols)
    assert np.isclose(np.mean(np.diff(np.asarray(resampled_support["x"], dtype=float))), target_dx, atol=1e-12, rtol=0.0)
    assert np.isclose(np.mean(np.diff(np.asarray(resampled_support["y"], dtype=float))), target_dx, atol=1e-12, rtol=0.0)
    assert resampled.name.endswith("-linear")
    assert np.array_equal(np.asarray(scene_get(scene, "photons"), dtype=float), photons)

    noise_scene = scene_create("uniform ee", 2, np.array([500.0, 600.0], dtype=float), asset_store=asset_store)
    noise_photons = np.array(
        [
            [[1.0, 20.0], [14.0, 30.0]],
            [[5.0, 40.0], [16.0, 8.0]],
        ],
        dtype=float,
    )
    noise_scene = scene_set(noise_scene, "photons", noise_photons)

    noisy_a, noise_a = scenePhotonNoise(noise_scene, seed=7)
    noisy_b, noise_b = scenePhotonNoise(noise_scene, seed=7)
    noisy_roi, noise_roi = scenePhotonNoise(noise_scene, [1, 1, 1, 1], seed=7)

    assert noisy_a.shape == noise_photons.shape
    assert noise_a.shape == noise_photons.shape
    assert np.array_equal(noisy_a, noisy_b)
    assert np.array_equal(noise_a, noise_b)
    assert np.allclose(noisy_a, np.rint(noisy_a))
    assert noisy_roi.shape == (4, 2)
    assert noise_roi.shape == (4, 2)
    assert np.array_equal(np.asarray(scene_get(noise_scene, "photons"), dtype=float), noise_photons)

    low_signal = noise_photons < 15.0
    assert np.allclose(noise_a[low_signal], noisy_a[low_signal])


def test_run_python_case_supports_scene_demo_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_demo_small", asset_store=asset_store)

    assert tuple(case.payload["macbeth_scene_size"]) == (64, 96)
    assert case.payload["macbeth_wave"].shape == (31,)
    assert np.isclose(case.payload["macbeth_mean_luminance"], 100.0, atol=1e-8, rtol=1e-8)
    assert case.payload["macbeth_luminance_bounds"].shape == (2,)
    assert case.payload["macbeth_center_row_luminance_norm"].shape == (96,)
    assert tuple(case.payload["macbeth_photons_shape"]) == (64, 96, 31)
    assert np.isclose(case.payload["macbeth_fov_after"], 20.0, atol=1e-12, rtol=0.0)
    assert tuple(case.payload["freq_scene_size"]) == (256, 256)
    assert case.payload["freq_scene_wave"].shape == (31,)
    assert np.isclose(case.payload["freq_scene_mean_luminance"], 100.0, atol=1e-8, rtol=1e-8)
    assert int(case.payload["freq_scene_rows_half"]) == 128
    assert case.payload["freq_scene_support_x_mm"].shape == (256,)
    assert case.payload["freq_scene_bottom_row_luminance_norm"].shape == (256,)
    assert case.payload["freq_scene_radiance_hline_550_norm"].shape == (256,)


def test_run_python_case_supports_scene_introduction_tutorial_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_introduction_tutorial_small", asset_store=asset_store)

    assert case.payload["scene_name"] == "Macbeth (D65)"
    assert case.payload["scene_type"] == "scene"
    assert tuple(case.payload["macbeth_size"]) == (64, 96)
    assert np.isclose(float(case.payload["hfov_after"]), 0.5, atol=1e-12, rtol=0.0)
    assert np.isclose(float(case.payload["distance_mm"]), 1200.0, atol=1e-9, rtol=0.0)
    assert case.payload["spacing_before_mm"].shape == (2,)
    assert np.all(case.payload["spacing_before_mm"] > 0.0)
    assert np.all(case.payload["spacing_after_distance_mm"] < case.payload["spacing_before_mm"])
    assert np.all(case.payload["spacing_after_hfov_mm"] < case.payload["spacing_after_distance_mm"])
    assert case.payload["spacing_after_hfov_um"].shape == (2,)
    assert case.payload["spacing_after_hfov_m"].shape == (2,)
    assert tuple(case.payload["stuffed_scene_size"]) == (506, 759)
    assert case.payload["stuffed_wave"].shape == (31,)
    assert float(case.payload["stuffed_mean_luminance"]) > 0.0
    assert np.isclose(
        float(case.payload["adjusted_mean_luminance"]),
        float(case.payload["stuffed_mean_luminance"]),
        atol=1e-10,
        rtol=1e-10,
    )
    assert case.payload["illuminant_energy_before_norm"].shape == (31,)
    assert case.payload["blackbody_energy_norm"].shape == (31,)
    assert case.payload["illuminant_energy_after_norm"].shape == (31,)
    assert case.payload["illuminant_photons_after_norm"].shape == (31,)
    assert not np.allclose(case.payload["illuminant_energy_before_norm"], case.payload["blackbody_energy_norm"])
    assert np.allclose(case.payload["illuminant_energy_after_norm"], case.payload["blackbody_energy_norm"])


def test_scene_examples_script_workflow(asset_store) -> None:
    rings = scene_create("rings rays", asset_store=asset_store)
    freq_orient = scene_create(
        "frequency orientation",
        {
            "angles": np.linspace(0.0, np.pi / 2.0, 5),
            "freqs": np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
            "blockSize": 64,
            "contrast": 0.8,
        },
        asset_store=asset_store,
    )
    harmonic_a = scene_create(
        "harmonic",
        {"freq": 1.0, "contrast": 1.0, "ph": 0.0, "ang": 0.0, "row": 64, "col": 64, "GaborFlag": 0.0},
        asset_store=asset_store,
    )
    harmonic_b = scene_create(
        "harmonic",
        {"freq": 1.0, "contrast": 1.0, "ph": 0.0, "ang": 0.0, "row": 64, "col": 64, "GaborFlag": 0.0},
        asset_store=asset_store,
    )
    checkerboard = scene_create("checkerboard", 16, 8, "ep", asset_store=asset_store)
    line = scene_create("lined65", 128, asset_store=asset_store)
    slanted = scene_create("slantedBar", 128, 1.3, asset_store=asset_store)
    grid = scene_create("grid lines", 128, 16, asset_store=asset_store)
    point = scene_create("point array", 256, 32, asset_store=asset_store)
    macbeth_a = scene_create("macbeth tungsten", 16, np.arange(380.0, 721.0, 5.0, dtype=float), asset_store=asset_store)
    reflectance_chart = scene_create("reflectance chart", asset_store=asset_store)
    macbeth_b = scene_create("macbeth tungsten", 16, np.arange(380.0, 721.0, 5.0, dtype=float), asset_store=asset_store)
    uniform_specify = scene_create("uniformEESpecify", 128, np.arange(380.0, 721.0, 10.0, dtype=float), asset_store=asset_store)
    lstar = scene_create("lstar", np.array([80, 10], dtype=int), 20, 1, asset_store=asset_store)
    exp_ramp = scene_create("exponential intensity ramp", 256, 1024, asset_store=asset_store)

    assert tuple(scene_get(rings, "size")) == (256, 256)
    assert tuple(scene_get(freq_orient, "size")) == (320, 320)
    assert tuple(scene_get(harmonic_a, "size")) == (64, 64)
    assert np.array_equal(np.asarray(scene_get(harmonic_a, "photons"), dtype=float), np.asarray(scene_get(harmonic_b, "photons"), dtype=float))
    assert tuple(scene_get(checkerboard, "size")) == (256, 256)
    assert tuple(scene_get(line, "size")) == (128, 128)
    assert tuple(scene_get(grid, "size")) == (128, 128)
    assert tuple(scene_get(point, "size")) == (256, 256)
    assert tuple(scene_get(macbeth_a, "size")) == (64, 96)
    assert tuple(scene_get(macbeth_b, "size")) == (64, 96)
    assert np.array_equal(np.asarray(scene_get(macbeth_a, "wave"), dtype=float), np.arange(380.0, 721.0, 5.0, dtype=float))
    assert tuple(scene_get(uniform_specify, "size")) == (128, 128)
    assert np.array_equal(np.asarray(scene_get(uniform_specify, "wave"), dtype=float), np.arange(380.0, 721.0, 10.0, dtype=float))
    assert tuple(scene_get(lstar, "size")) == (80, 200)
    assert tuple(scene_get(exp_ramp, "size")) == (256, 256)
    assert tuple(scene_get(slanted, "size"))[0] == tuple(scene_get(slanted, "size"))[1]
    assert tuple(scene_get(reflectance_chart, "size"))[0] > 0
    assert tuple(scene_get(reflectance_chart, "size"))[1] > 0

    bar_means = np.array(
        [
            np.mean(np.asarray(scene_get(lstar, "luminance", asset_store=asset_store), dtype=float)[:, start : start + 10])
            for start in range(0, 200, 10)
        ],
        dtype=float,
    )
    assert np.all(np.diff(bar_means) > 0.0)
    for scene in [rings, freq_orient, harmonic_a, checkerboard, slanted, macbeth_a, macbeth_b, uniform_specify, lstar, exp_ramp]:
        assert np.isclose(float(scene_get(scene, "mean luminance", asset_store=asset_store)), 100.0, atol=1e-8, rtol=1e-8)
    for scene in [line, grid, point, reflectance_chart]:
        assert np.isfinite(float(scene_get(scene, "mean luminance", asset_store=asset_store)))


def test_run_python_case_supports_scene_examples_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_examples_small", asset_store=asset_store)

    assert len(case.payload["scene_labels"]) == 14
    assert case.payload["scene_sizes"].shape == (14, 2)
    assert case.payload["scene_wave_counts"].shape == (14,)
    assert case.payload["scene_mean_luminance_stable"].shape == (10,)
    assert case.payload["scene_fov_deg_stable"].shape == (12,)
    assert case.payload["scene_luminance_bounds_stable"].shape == (10, 2)
    assert case.payload["scene_center_row_luminance_norm"].shape == (14, 41)
    assert case.payload["scene_center_col_luminance_norm"].shape == (14, 41)
    assert tuple(case.payload["scene_sizes"][0]) == (256, 256)
    assert tuple(case.payload["scene_sizes"][1]) == (320, 320)
    assert tuple(case.payload["scene_sizes"][2]) == (64, 64)
    assert tuple(case.payload["scene_sizes"][8]) == (256, 256)
    assert tuple(case.payload["scene_sizes"][9]) == (64, 96)
    assert tuple(case.payload["scene_sizes"][10]) == (64, 96)
    assert tuple(case.payload["scene_sizes"][11]) == (128, 128)
    assert tuple(case.payload["scene_sizes"][12]) == (80, 200)
    assert tuple(case.payload["scene_sizes"][13]) == (256, 256)
    assert np.array_equal(case.payload["scene_wave_counts"], np.array([31, 31, 31, 31, 31, 31, 31, 31, 31, 69, 69, 35, 31, 31], dtype=int))
    assert np.array_equal(case.payload["scene_fov_deg_stable"], np.array([10.0, 10.0, 10.0, 10.0, 2.0, 40.0, 40.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float))
    assert np.all(np.isfinite(case.payload["scene_mean_luminance_stable"]))
    assert np.allclose(case.payload["scene_mean_luminance_stable"], 100.0, atol=1e-8, rtol=1e-8)
    assert np.all(np.isfinite(case.payload["scene_luminance_bounds_stable"]))
    assert tuple(case.payload["reflectance_chart_size"].shape) == (2,)
    assert int(case.payload["reflectance_chart_wave_count"]) == 31
    assert np.isclose(float(case.payload["reflectance_chart_mean_luminance"]), 100.0, atol=1e-8, rtol=1e-8)


def test_scene_from_rgb_script_workflow(asset_store) -> None:
    display = display_create("LCD-Apple.mat", asset_store=asset_store)
    wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    spd = np.asarray(display_get(display, "spd"), dtype=float)
    white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)
    white_xy = np.asarray(
        chromaticity_xy(xyz_from_energy(white_spd, wave, asset_store=asset_store)),
        dtype=float,
    ).reshape(-1)

    scene = scene_from_file(
        asset_store.resolve("data/images/rgb/eagle.jpg"),
        "rgb",
        None,
        "LCD-Apple.mat",
        asset_store=asset_store,
    )
    initial_mean_luminance = float(scene_get(scene, "mean luminance", asset_store=asset_store))
    mean_photons = np.mean(np.asarray(scene_get(scene, "photons"), dtype=float), axis=(0, 1))

    adjusted_scene = scene_adjust_illuminant(
        scene.clone(),
        blackbody(scene_get(scene, "wave"), 6500.0, kind="energy"),
        asset_store=asset_store,
    )
    adjusted_mean_luminance = float(scene_get(adjusted_scene, "mean luminance", asset_store=asset_store))
    roi_mean_reflectance = np.asarray(
        scene_get(adjusted_scene, "roi mean reflectance", [144, 198, 27, 18], asset_store=asset_store),
        dtype=float,
    ).reshape(-1)

    assert spd.shape == (101, 3)
    assert white_spd.shape == (101,)
    assert white_xy.shape == (2,)
    assert tuple(scene_get(scene, "size")) == (336, 512)
    assert tuple(np.asarray(scene_get(scene, "photons"), dtype=float).shape) == (336, 512, 101)
    assert scene.fields["display_name"] == "LCD-Apple"
    assert scene.fields["illuminant_comment"] == "LCD-Apple"
    assert np.isclose(initial_mean_luminance, adjusted_mean_luminance, atol=1e-8, rtol=1e-8)
    assert np.all(mean_photons > 0.0)
    assert roi_mean_reflectance.shape == (101,)
    assert float(np.max(roi_mean_reflectance)) < 0.5
    assert float(np.min(roi_mean_reflectance)) > 0.0


def test_scene_rgb2radiance_tutorial_workflow(asset_store) -> None:
    display_names = ["OLED-Sony.mat", "LCD-Apple.mat", "CRT-Dell.mat"]
    payloads: dict[str, dict[str, np.ndarray | float | tuple[int, int]]] = {}

    for name in display_names:
        display = display_create(name, asset_store=asset_store)
        wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
        spd = np.asarray(display_get(display, "spd"), dtype=float)
        white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)
        white_xy = np.asarray(chromaticity_xy(xyz_from_energy(white_spd, wave, asset_store=asset_store)), dtype=float).reshape(-1)
        primary_xy = np.vstack(
            [
                np.asarray(chromaticity_xy(xyz_from_energy(spd[:, idx], wave, asset_store=asset_store)), dtype=float).reshape(-1)
                for idx in range(spd.shape[1])
            ]
        )
        scene = scene_from_file("macbeth.tif", "rgb", None, name, asset_store=asset_store)
        rendered_rgb = np.asarray(scene_get(scene, "rgb", asset_store=asset_store), dtype=float)

        payloads[name] = {
            "spd_shape": tuple(spd.shape),
            "white_xy": white_xy,
            "primary_xy": primary_xy,
            "scene_size": tuple(scene_get(scene, "size")),
            "mean_luminance": float(scene_get(scene, "mean luminance", asset_store=asset_store)),
            "rgb_channel_means": np.mean(rendered_rgb, axis=(0, 1), dtype=float).reshape(-1),
        }

        assert tuple(scene_get(scene, "size")) == (64, 96)
        assert rendered_rgb.shape == (64, 96, 3)
        assert np.all(rendered_rgb >= 0.0)
        assert np.all(rendered_rgb <= 1.0)
        assert np.all(np.isfinite(white_xy))
        assert np.all(np.isfinite(primary_xy))
        assert scene.fields["display_name"] == display.name

    assert payloads["OLED-Sony.mat"]["spd_shape"] == (101, 4)
    assert payloads["LCD-Apple.mat"]["spd_shape"] == (101, 3)
    assert payloads["CRT-Dell.mat"]["spd_shape"] == (101, 3)
    assert payloads["OLED-Sony.mat"]["primary_xy"].shape == (4, 2)
    assert payloads["LCD-Apple.mat"]["primary_xy"].shape == (3, 2)
    assert payloads["CRT-Dell.mat"]["primary_xy"].shape == (3, 2)
    assert payloads["OLED-Sony.mat"]["mean_luminance"] > payloads["LCD-Apple.mat"]["mean_luminance"] > payloads["CRT-Dell.mat"]["mean_luminance"]
    assert not np.allclose(payloads["OLED-Sony.mat"]["rgb_channel_means"], payloads["LCD-Apple.mat"]["rgb_channel_means"])
    assert not np.allclose(payloads["LCD-Apple.mat"]["rgb_channel_means"], payloads["CRT-Dell.mat"]["rgb_channel_means"])


def test_color_chromaticity_tutorial_workflow(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    spd = np.asarray(display_get(display, "spd"), dtype=float)
    white_xyz = np.asarray(display_get(display, "white xyz"), dtype=float).reshape(-1)
    white_xy = np.asarray(display_get(display, "white xy"), dtype=float).reshape(-1)

    primary_xyz = np.asarray(
        [
            xyz_from_energy(spd[:, idx], wave, asset_store=asset_store)
            for idx in range(spd.shape[1])
        ],
        dtype=float,
    ).reshape(spd.shape[1], 3)
    primary_xy = np.asarray(chromaticity_xy(primary_xyz), dtype=float).reshape(spd.shape[1], 2)

    scene = scene_create(asset_store=asset_store)
    scene_xyz = np.asarray(scene_get(scene, "xyz", asset_store=asset_store), dtype=float)
    scene_xyz_xw, rows, cols, channels = rgb_to_xw_format(scene_xyz)
    scene_xy = np.asarray(chromaticity_xy(scene_xyz_xw), dtype=float).reshape(rows * cols, 2)

    assert spd.shape == (101, 3)
    assert white_xyz.shape == (3,)
    assert white_xy.shape == (2,)
    assert primary_xyz.shape == (3, 3)
    assert primary_xy.shape == (3, 2)
    assert channels == 3
    assert scene_xy.shape == (rows * cols, 2)
    assert np.allclose(
        white_xy,
        np.asarray(chromaticity_xy(white_xyz), dtype=float).reshape(-1),
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.all(np.isfinite(primary_xy))
    assert np.all(np.isfinite(scene_xy))
    assert np.all(primary_xy >= 0.0)
    assert np.all(primary_xy <= 1.0)
    assert np.all(scene_xy >= 0.0)
    assert np.all(scene_xy <= 1.0)


def test_color_energy_quanta_tutorial_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 5.0, dtype=float)
    photons = np.ones(wave.size, dtype=float)
    energy = np.asarray(quanta_to_energy(photons, wave), dtype=float).reshape(-1)

    d65_energy = np.asarray(ie_read_spectra("D65.mat", wave, asset_store=asset_store), dtype=float).reshape(-1)
    d65_xyz = np.asarray(xyz_from_energy(d65_energy, wave, asset_store=asset_store), dtype=float).reshape(-1)
    d65_energy = d65_energy * 100.0 / max(float(d65_xyz[1]), 1.0e-12)
    d65_photons = np.asarray(energy_to_quanta(d65_energy, wave), dtype=float).reshape(-1)

    xyz_energy = np.asarray(ie_read_spectra("XYZ", wave, asset_store=asset_store), dtype=float)
    xyz_photons = np.asarray(ie_read_spectra("XYZQuanta", wave, asset_store=asset_store), dtype=float)

    xyz_from_energy_wrapper = np.asarray(xyz_from_energy(d65_energy, wave, asset_store=asset_store), dtype=float).reshape(-1)
    xyz_from_photons_wrapper = np.asarray(ie_xyz_from_photons(d65_photons, wave, asset_store=asset_store), dtype=float).reshape(-1)

    dnm = float(wave[1] - wave[0])
    xyz_from_energy_manual = dnm * 683.0 * (xyz_energy.T @ d65_energy)
    xyz_from_photons_manual = dnm * 683.0 * (xyz_photons.T @ d65_photons)

    assert energy.shape == wave.shape
    assert np.all(np.diff(energy) < 0.0)
    assert d65_photons.shape == wave.shape
    assert xyz_energy.shape == (wave.size, 3)
    assert xyz_photons.shape == (wave.size, 3)
    assert np.isclose(float(xyz_from_energy_wrapper[1]), 100.0, atol=1e-8, rtol=1e-8)
    assert np.allclose(xyz_from_energy_wrapper, xyz_from_photons_wrapper, atol=1e-8, rtol=1e-8)
    assert np.allclose(xyz_from_energy_manual, xyz_from_photons_manual, atol=1e-8, rtol=1e-8)
    assert np.allclose(xyz_from_energy_manual, xyz_from_energy_wrapper, atol=1e-8, rtol=1e-8)


def test_code_rendering_tutorial_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)

    warm_spd = np.asarray(blackbody(wave, 3000.0, kind="energy"), dtype=float).reshape(1, -1)
    warm_xyz = np.asarray(xyz_from_energy(warm_spd, wave, asset_store=asset_store), dtype=float).reshape(1, 1, 3)
    warm_srgb = np.asarray(xyz_to_srgb(warm_xyz), dtype=float)
    warm_cube = np.asarray(xw_to_rgb_format(warm_spd, 1, 1), dtype=float)
    warm_big = np.asarray(image_increase_image_rgb_size(warm_cube, 120), dtype=float)

    cool_spd = np.asarray(blackbody(wave, 8000.0, kind="energy"), dtype=float).reshape(1, -1)
    cool_xyz = np.asarray(xyz_from_energy(cool_spd, wave, asset_store=asset_store), dtype=float).reshape(1, 1, 3)
    cool_srgb = np.asarray(xyz_to_srgb(cool_xyz), dtype=float)
    cool_cube = np.asarray(xw_to_rgb_format(cool_spd, 1, 1), dtype=float)
    cool_big = np.asarray(image_increase_image_rgb_size(cool_cube, 120), dtype=float)

    assert warm_srgb.shape == (1, 1, 3)
    assert cool_srgb.shape == (1, 1, 3)
    assert warm_big.shape == (120, 120, wave.size)
    assert cool_big.shape == (120, 120, wave.size)
    assert np.allclose(warm_big[0, 0, :], warm_spd.reshape(-1), atol=1e-12, rtol=1e-12)
    assert np.allclose(cool_big[-1, -1, :], cool_spd.reshape(-1), atol=1e-12, rtol=1e-12)
    assert float(warm_srgb[0, 0, 0]) > float(warm_srgb[0, 0, 2])
    assert float(cool_srgb[0, 0, 2]) > float(cool_srgb[0, 0, 0])
    assert np.all(warm_srgb >= 0.0)
    assert np.all(warm_srgb <= 1.0)
    assert np.all(cool_srgb >= 0.0)
    assert np.all(cool_srgb <= 1.0)


def test_camera_introduction_tutorial_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)

    camera = camera_create(asset_store=asset_store)
    camera = camera_compute(camera, scene, asset_store=asset_store)
    ip = camera_get(camera, "ip")
    srgb_default = np.asarray(camera_get(camera, "ip data srgb"), dtype=float)

    camera = camera_set(camera, "optics fnumber", 16.0)
    camera = camera_compute(camera, scene, asset_store=asset_store)
    srgb_stopped = np.asarray(camera_get(camera, "ip data srgb"), dtype=float)

    camera = camera_set(camera, "ip illuminant correction method", "gray world")
    camera = camera_compute(camera, scene, asset_store=asset_store)
    srgb_gray = np.asarray(camera_get(camera, "ip data srgb"), dtype=float)

    camera_from_oi = camera_compute(camera, "oi", asset_store=asset_store)
    srgb_from_oi = np.asarray(camera_get(camera_from_oi, "ip data srgb"), dtype=float)
    camera_from_sensor = camera_compute(camera_from_oi, "sensor", asset_store=asset_store)
    srgb_from_sensor = np.asarray(camera_get(camera_from_sensor, "ip data srgb"), dtype=float)

    assert srgb_default.shape == srgb_stopped.shape == srgb_gray.shape
    assert srgb_default.shape[2] == 3
    assert np.allclose(np.asarray(ip_get(ip, "data srgb"), dtype=float), srgb_default, atol=1e-12, rtol=1e-12)
    assert not np.allclose(srgb_default, srgb_stopped)
    assert not np.allclose(srgb_stopped, srgb_gray)
    assert np.allclose(srgb_gray, srgb_from_oi, atol=1e-12, rtol=1e-12)
    assert np.allclose(srgb_gray, srgb_from_sensor, atol=1e-12, rtol=1e-12)


def test_camera_compute_supports_reference_lrgb_scaling(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)

    camera = camera_create(asset_store=asset_store)
    camera = camera_compute(camera, scene, asset_store=asset_store)
    result = np.asarray(camera_get(camera, "ip result"), dtype=float)
    reference = result * 0.25

    scaled_camera = camera_compute(camera, "sensor", reference, asset_store=asset_store)
    scaled_result = np.asarray(camera_get(scaled_camera, "ip result"), dtype=float)

    assert scaled_result.shape == reference.shape
    assert not np.allclose(scaled_result, result)
    np.testing.assert_allclose(
        np.mean(scaled_result[10:-10, 10:-10, :]),
        np.mean(reference[10:-10, 10:-10, :]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_camera_compute_ideal_xyz_mode_reuses_oi_path(asset_store) -> None:
    scene = scene_create("macbeth d65", 8, asset_store=asset_store)
    scene = scene_set(scene, "fov", 8.0)

    camera = camera_create(asset_store=asset_store)
    ideal_camera = camera_compute(camera, scene, "ideal xyz", asset_store=asset_store)

    xyz = np.asarray(camera_get(ideal_camera, "ip data xyz"), dtype=float)
    srgb = np.asarray(camera_get(ideal_camera, "ip data srgb"), dtype=float)
    result = np.asarray(camera_get(ideal_camera, "ip result"), dtype=float)

    assert xyz.shape == srgb.shape == result.shape
    assert xyz.shape[2] == 3
    assert np.all(srgb >= 0.0)
    assert np.all(srgb <= 1.0)
    assert camera_get(ideal_camera, "ip internal cs") == "XYZ"
    assert camera_get(ideal_camera, "sensor noise flag") == -1
    np.testing.assert_allclose(
        np.asarray(camera_get(ideal_camera, "sensor wave"), dtype=float),
        np.asarray(camera_get(ideal_camera, "oi wave"), dtype=float),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(ideal_camera, "image"), dtype=float),
        result,
        atol=1e-12,
        rtol=1e-12,
    )

    ideal_from_oi = camera_compute(ideal_camera, "oi", "ideal xyz", asset_store=asset_store)
    np.testing.assert_allclose(
        np.asarray(camera_get(ideal_from_oi, "ip data xyz"), dtype=float),
        xyz,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(camera_get(ideal_from_oi, "ip data srgb"), dtype=float),
        srgb,
        atol=1e-12,
        rtol=1e-12,
    )


def test_camera_antialias_and_noise_tutorial_workflows(asset_store) -> None:
    aa_scene = scene_create("freq orient", 512, asset_store=asset_store)
    aa_scene = scene_set(aa_scene, "fov", 6.0)

    oi_base = oi_create(asset_store=asset_store)
    oi_base = oi_set(oi_base, "optics fnumber", 2.0)
    oi_base = oi_compute(oi_base, aa_scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "pixel size constant fill factor", [1.5e-6, 1.5e-6])
    sensor = sensor_set_size_to_fov(sensor, 5.0, oi_base)
    pixel_size = float(np.asarray(sensor_get(sensor, "pixel size"), dtype=float).reshape(-1)[0])

    def _ip_gray_from_oi(current_oi):
        current_sensor = sensor_compute(sensor.clone(), current_oi, seed=0)
        current_ip = ip_compute(ip_create(asset_store=asset_store), current_sensor)
        current_rgb = np.asarray(ip_get(current_ip, "result"), dtype=float)
        current_gray = np.mean(current_rgb, axis=2)
        return current_gray

    base_gray = _ip_gray_from_oi(oi_base)

    oi_blur = oi_set(oi_base.clone(), "diffuser method", "blur")
    oi_blur = oi_set(oi_blur, "diffuser blur", pixel_size)
    oi_blur = oi_compute(oi_blur, aa_scene)
    blur_gray = _ip_gray_from_oi(oi_blur)

    oi_biref = oi_set(oi_base.clone(), "diffuser method", "birefringent")
    oi_biref = oi_compute(oi_biref, aa_scene)
    biref_gray = _ip_gray_from_oi(oi_biref)

    base_grad = float(np.mean(np.abs(np.diff(base_gray, axis=1))))
    blur_grad = float(np.mean(np.abs(np.diff(blur_gray, axis=1))))
    biref_grad = float(np.mean(np.abs(np.diff(biref_gray, axis=1))))

    assert base_gray.shape == blur_gray.shape == biref_gray.shape
    assert blur_grad < base_grad
    assert biref_grad < base_grad
    assert not np.allclose(base_gray, blur_gray)
    assert not np.allclose(base_gray, biref_gray)

    display = display_create(asset_store=asset_store)
    noise_scene = scene_from_file("faceMale.jpg", "rgb", None, display, asset_store=asset_store)
    low_noise_scene = scene_adjust_luminance(noise_scene.clone(), 5.0, asset_store=asset_store)
    bright_noise_scene = scene_adjust_luminance(noise_scene.clone(), 100.0, asset_store=asset_store)

    base_camera = camera_create(asset_store=asset_store)
    base_camera = camera_set(base_camera, "optics fnumber", 2.0)
    base_camera = camera_set(base_camera, "sensor exp time", 0.01)

    camera_noiseless = camera_set(base_camera.clone(), "sensor noise flag", 0)
    camera_noiseless = camera_compute(camera_noiseless, bright_noise_scene, asset_store=asset_store)
    noiseless_rgb = np.asarray(camera_get(camera_noiseless, "ip data srgb"), dtype=float)

    camera_photon = camera_set(base_camera.clone(), "sensor noise flag", 1)
    camera_photon = camera_set(camera_photon, "sensor noise seed", 7)
    camera_photon = camera_compute(camera_photon, low_noise_scene, asset_store=asset_store)
    photon_rgb = np.asarray(camera_get(camera_photon, "ip data srgb"), dtype=float)

    camera_all = camera_set(base_camera.clone(), "sensor noise flag", 2)
    camera_all = camera_set(camera_all, "sensor noise seed", 7)
    camera_all = camera_set(camera_all, "pixel read noise volts", 0.0)
    camera_all = camera_set(camera_all, "pixel dark voltage", 0.0)
    camera_all = camera_set(camera_all, "sensor dsnu level", 0.0)
    camera_all = camera_set(camera_all, "sensor prnu level", 0.0)
    camera_all = camera_compute(camera_all, bright_noise_scene, asset_store=asset_store)
    all_noise_rgb = np.asarray(camera_get(camera_all, "ip data srgb"), dtype=float)

    assert noiseless_rgb.shape == photon_rgb.shape == all_noise_rgb.shape
    assert float(np.mean(photon_rgb)) < float(np.mean(noiseless_rgb))
    assert float(np.mean(np.abs(all_noise_rgb - noiseless_rgb))) > 1e-4
    assert float(np.mean(np.abs(photon_rgb - noiseless_rgb))) > 1e-4


def test_printing_halftone_tutorial_workflow() -> None:
    half_tone_cell_4 = np.array(
        [
            [15, 5, 12, 14],
            [10, 3, 2, 8],
            [7, 1, 4, 11],
            [13, 9, 6, 16],
        ],
        dtype=float,
    )
    half_tone_cell_b4 = np.array(
        [
            [1, 17, 5, 21, 2, 18, 6, 22],
            [25, 9, 29, 13, 26, 10, 30, 14],
            [7, 23, 3, 19, 8, 24, 4, 20],
            [31, 15, 27, 11, 32, 16, 28, 12],
            [2, 18, 6, 22, 1, 17, 5, 21],
            [26, 10, 30, 14, 25, 9, 29, 13],
            [8, 24, 4, 20, 7, 23, 3, 19],
            [32, 16, 28, 12, 31, 15, 27, 11],
        ],
        dtype=float,
    )
    gray_strips = 8
    sweep_size = 128
    monitor_gamma = 2.0
    fs = np.array([[0.0, 0.0, 7.0], [3.0, 5.0, 1.0]], dtype=float) / 16.0

    x, y = np.meshgrid(np.arange(sweep_size, dtype=float), np.arange(sweep_size, dtype=float))
    y = np.fix(y * gray_strips / sweep_size) / gray_strips
    x = x / (sweep_size * gray_strips)
    sweep_8 = np.ones((sweep_size, sweep_size), dtype=float) - (x + y)
    sweep_8_linear = sweep_8**monitor_gamma

    ht_sweep_4 = HalfToneImage(half_tone_cell_4, sweep_8_linear)
    ht_sweep_b4 = HalfToneImage(half_tone_cell_b4, sweep_8_linear)
    fs_sweep = FloydSteinberg(fs, sweep_8_linear)

    def _block_mean(image: np.ndarray, block_size: int) -> np.ndarray:
        rows = image.shape[0] // block_size
        cols = image.shape[1] // block_size
        return (
            np.asarray(image[: rows * block_size, : cols * block_size], dtype=float)
            .reshape(rows, block_size, cols, block_size)
            .mean(axis=(1, 3))
        )

    source4 = _block_mean(sweep_8_linear, 4)
    source8 = _block_mean(sweep_8_linear, 8)
    ht4 = _block_mean(ht_sweep_4.astype(float), 4)
    ht8 = _block_mean(ht_sweep_b4.astype(float), 8)

    assert ht_sweep_4.shape == sweep_8_linear.shape
    assert ht_sweep_b4.shape == sweep_8_linear.shape
    assert fs_sweep.shape == sweep_8_linear.shape
    assert set(np.unique(ht_sweep_4)).issubset({False, True})
    assert set(np.unique(ht_sweep_b4)).issubset({False, True})
    assert set(np.unique(fs_sweep)).issubset({0.0, 1.0})
    assert float(np.mean(np.abs(ht4 - source4))) < 0.12
    assert float(np.mean(np.abs(ht8 - source8))) < 0.08
    assert abs(float(np.mean(fs_sweep)) - float(np.mean(sweep_8_linear))) < 0.02
    assert np.kron(ht_sweep_b4.astype(float), np.ones((3, 3), dtype=float)).shape == (384, 384)


def test_data_scratch_initialization_workflow(asset_store) -> None:
    scene = scene_create("slanted bar", asset_store=asset_store)
    oi = oi_create(asset_store=asset_store)
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_compute(sensor, oi)

    ip = ip_create(asset_store=asset_store)
    ip = ip_compute(ip, sensor)

    oi_photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    sensor_volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
    ip_rgb = np.asarray(ip_get(ip, "result"), dtype=float)

    assert scene.name
    assert oi.name == scene.name
    assert oi_photons.ndim == 3 and oi_photons.size > 0
    assert sensor_volts.ndim == 2 and sensor_volts.size > 0
    assert sensor_volts.shape[0] <= oi_photons.shape[0]
    assert sensor_volts.shape[1] <= oi_photons.shape[1]
    assert ip_rgb.shape[:2] == sensor_volts.shape
    assert ip_rgb.shape[2] == 3
    assert float(np.mean(oi_photons)) > 0.0
    assert float(np.mean(sensor_volts)) > 0.0
    assert np.all(ip_rgb >= 0.0)
    assert np.all(ip_rgb <= 1.0)


def test_display_helper_compatibility_surface(asset_store) -> None:
    all_names = displayList(show=False, asset_store=asset_store)
    lcd_names = displayList(type="LCD", show=False, asset_store=asset_store)
    display = display_create("LCD-Apple.mat", asset_store=asset_store)
    rgb = np.linspace(0.0, 1.0, 27, dtype=float).reshape(3, 3, 3)
    display.fields["image"] = rgb

    description = displayDescription(display)
    rendered = np.asarray(displayShowImage(display, asset_store=asset_store), dtype=float)
    expected = np.asarray(
        scene_get(scene_from_file(rgb, "rgb", None, display, asset_store=asset_store), "rgb", asset_store=asset_store),
        dtype=float,
    )
    current_white = np.asarray(display_get(display, "white point"), dtype=float).reshape(3)
    scaled = displaySetMaxLuminance(display, current_white[1] * 1.25)
    rewhite = displaySetWhitePoint(display, np.array([0.31, 0.33], dtype=float))

    assert "LCD-Apple.mat" in all_names
    assert "LCD-Apple.mat" in lcd_names
    assert all(name.startswith("LCD") for name in lcd_names)
    assert displayDescription(None) == "No display structure"
    assert str(display_get(display, "name")) in description
    assert "Image width: 3" in description
    assert "Height: 3" in description
    assert np.isclose(mperdot2dpi(254.0), 100.0, atol=1e-12, rtol=1e-12)
    dpi, pitch = ieCalculateMonitorDPI(25.4, 25.4, 1000, 1000)
    assert np.allclose(dpi, np.full(2, 100.0, dtype=float), atol=1e-12, rtol=1e-12)
    assert np.allclose(pitch, np.full(2, 0.254, dtype=float), atol=1e-12, rtol=1e-12)
    assert np.isclose(
        displayMaxContrast(np.array([0.5, -0.25, 0.1], dtype=float), np.array([0.5, 0.5, 0.2], dtype=float)),
        1.0,
        atol=1e-12,
        rtol=1e-12,
    )
    assert rendered.shape == rgb.shape
    assert np.allclose(rendered, expected, atol=1e-12, rtol=1e-12)
    assert np.isclose(display_get(scaled, "max luminance"), current_white[1] * 1.25, atol=1e-10, rtol=1e-10)
    assert np.isclose(display_get(display, "max luminance"), current_white[1], atol=1e-10, rtol=1e-10)
    assert np.isclose(display_get(rewhite, "white point")[1], current_white[1], atol=1e-10, rtol=1e-10)
    assert np.allclose(
        np.asarray(display_get(rewhite, "white xy"), dtype=float).reshape(2),
        np.array([0.31, 0.33], dtype=float),
        atol=1e-8,
        rtol=1e-8,
    )


def test_macbeth_helper_compatibility_surface(asset_store) -> None:
    scene = scene_create("macbeth d65", 16, asset_store=asset_store)
    corner_points = np.array(
        [
            [64, 1],
            [64, 96],
            [1, 96],
            [1, 1],
        ],
        dtype=float,
    )

    patch_locs, delta, patch_size = macbethRectangles(corner_points)
    roi_locs, rect = macbethROIs(patch_locs[:, 0], delta)
    patch_data, patch_std = macbethPatchData(scene, patch_locs, delta, data_type="photons")
    full_patch_data, empty_std = macbethPatchData(scene, patch_locs, delta, full_data=True, data_type="photons")

    manual_patch = np.asarray(vc_get_roi_data(scene, roi_locs, "photons"), dtype=float)
    ideal_xyz = np.asarray(macbethIdealColor("D65", "XYZ", asset_store=asset_store), dtype=float)
    ideal_lab = np.asarray(macbethIdealColor("tungsten", "LAB", asset_store=asset_store), dtype=float)
    ideal_lrgb = np.asarray(macbethIdealColor("D65", "lRGB", asset_store=asset_store), dtype=float)
    ideal_srgb = np.asarray(macbethIdealColor("D65", "sRGB", asset_store=asset_store), dtype=float)

    assert patch_locs.shape == (2, 24)
    assert delta == 5
    assert patch_size == 11
    assert rect.shape == (4,)
    assert np.array_equal(rect[2:], np.array([5, 5], dtype=int))
    assert 1 <= rect[0] <= 96
    assert 1 <= rect[1] <= 64
    assert roi_locs.shape == (36, 2)
    assert patch_data.shape == (24, 31)
    assert patch_std.shape == (24, 31)
    assert len(full_patch_data) == 24
    assert full_patch_data[0].shape == (36, 31)
    assert empty_std.size == 0
    assert np.allclose(full_patch_data[0], manual_patch, atol=1e-12, rtol=1e-12)
    assert np.allclose(patch_data[0], np.mean(manual_patch, axis=0), atol=1e-12, rtol=1e-12)
    patch_means = np.mean(patch_data, axis=1)
    assert float(np.max(patch_means)) > float(np.min(patch_means))

    assert ideal_xyz.shape == (24, 3)
    assert ideal_lab.shape == (24, 3)
    assert ideal_lrgb.shape == (24, 3)
    assert ideal_srgb.shape == (24, 3)
    assert np.isclose(float(np.max(ideal_xyz[:, 1])), 100.0, atol=1e-8, rtol=1e-8)
    assert ideal_lab[3, 0] > ideal_lab[23, 0]
    assert np.all(ideal_lrgb >= 0.0)
    assert np.all(ideal_lrgb <= 1.0)
    assert np.all(ideal_srgb >= 0.0)
    assert np.all(ideal_srgb <= 1.0)
    assert not np.allclose(ideal_lrgb, ideal_srgb)


def test_macbeth_wrapper_compatibility_surface(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    reflectance = macbeth_read_reflectance(wave, [1, 4, 24], asset_store=asset_store)
    chart = macbethChartCreate(4, [1, 4, 24], wave, None, False, asset_store=asset_store)
    chart_data = np.asarray(chart.data["data"], dtype=float)

    assert chart.name == "Macbeth Chart"
    assert chart.type == "scene"
    assert chart_data.shape == (4, 12, 31)
    assert np.allclose(chart_data[0:4, 0:4, :], reflectance[:, 0].reshape(1, 1, -1), atol=1e-12, rtol=1e-12)
    assert np.allclose(chart_data[0:4, 4:8, :], reflectance[:, 1].reshape(1, 1, -1), atol=1e-12, rtol=1e-12)
    assert np.allclose(chart_data[0:4, 8:12, :], reflectance[:, 2].reshape(1, 1, -1), atol=1e-12, rtol=1e-12)

    scene = scene_create("macbeth d65", 16, asset_store=asset_store)
    corner_points = np.array([[64.0, 1.0], [64.0, 96.0], [1.0, 96.0], [1.0, 1.0]], dtype=float)
    scene = scene_set(scene, "chart corner points", corner_points)
    rect_payload = macbethDrawRects(scene, "on")
    patch_data, m_locs, p_size, resolved_corners, patch_std = macbethSelect(scene, False, False, corner_points)

    assert rect_payload["rects"].shape == (24, 4)
    assert np.array_equal(np.asarray(scene_get(scene, "chart corner points"), dtype=float), corner_points)
    assert np.array_equal(np.asarray(scene_get(scene, "chart rects"), dtype=int), rect_payload["rects"])
    assert np.array_equal(np.asarray(scene_get(scene, "current rect"), dtype=int), rect_payload["rects"][0])
    assert patch_data.shape == (24, 31)
    assert patch_std.shape == (24, 31)
    assert m_locs.shape == (2, 24)
    assert p_size == 11
    assert np.array_equal(resolved_corners, corner_points)

    ideal_rgb = np.asarray(macbethIdealColor("D65", "lRGB", asset_store=asset_store), dtype=float)
    eval_payload = macbethEvaluationGraphs(np.eye(3), ideal_rgb, ideal_rgb, "sensor", asset_store=asset_store)

    assert eval_payload["rgbL"].shape == (24, 3)
    assert eval_payload["idealXYZ"].shape == (24, 3)
    assert eval_payload["deltaEab"].shape == (24,)
    assert np.isclose(eval_payload["meanDeltaEab"], 0.0, atol=1e-10, rtol=1e-10)


def test_scene_reflectance_wrapper_surface(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip = ip_set(ip, "result", np.arange(1.0, 6.0 * 6.0 * 3.0 + 1.0, dtype=float).reshape(6, 6, 3))
    m_locs = np.array([[2.0, 4.0], [2.0, 4.0]], dtype=float)

    patch_means = np.asarray(chartPatchData(ip, m_locs, 1, False), dtype=float)
    full_patch_data = chartPatchData(ip, m_locs, 1, True)
    roi_a = ie_rect2_locs(np.array([1, 1, 1, 1], dtype=int))
    roi_b = ie_rect2_locs(np.array([3, 3, 1, 1], dtype=int))
    manual_a = np.asarray(ip_get(ip, "roi data", roi_a), dtype=float)
    manual_b = np.asarray(ip_get(ip, "roi data", roi_b), dtype=float)

    assert patch_means.shape == (2, 3)
    assert len(full_patch_data) == 2
    assert np.allclose(full_patch_data[0], manual_a, atol=1e-12, rtol=1e-12)
    assert np.allclose(full_patch_data[1], manual_b, atol=1e-12, rtol=1e-12)
    assert np.allclose(patch_means[0], np.mean(manual_a, axis=0), atol=1e-12, rtol=1e-12)
    assert np.allclose(patch_means[1], np.mean(manual_b, axis=0), atol=1e-12, rtol=1e-12)

    sensor = sensor_create(asset_store=asset_store)
    with pytest.raises(ValueError, match="Use fullData = 1"):
        chartPatchData(sensor, m_locs, 1, False, "volts")


def test_ie_cook_torrance_wrapper_matches_manual_formula() -> None:
    surface_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    view_direction = np.array([0.1, -0.2, 1.0], dtype=float)
    light_direction = np.array([0.3, 0.4, 1.0], dtype=float)
    base_reflectance = np.array([0.04, 0.05, 0.06], dtype=float)
    roughness = 0.35

    result = np.asarray(ieCookTorrance(surface_normal, view_direction, light_direction, base_reflectance, roughness), dtype=float)

    surface = surface_normal / np.linalg.norm(surface_normal)
    view = view_direction / np.linalg.norm(view_direction)
    light = light_direction / np.linalg.norm(light_direction)
    half_vector = view + light
    half_vector = half_vector / np.linalg.norm(half_vector)

    n_dot_v = max(float(np.dot(surface, view)), 1.0e-5)
    n_dot_l = max(float(np.dot(surface, light)), 1.0e-5)
    n_dot_h = max(float(np.dot(surface, half_vector)), 1.0e-5)
    v_dot_h = max(float(np.dot(view, half_vector)), 1.0e-5)
    alpha = roughness**2
    alpha_sq = alpha**2
    distribution = alpha_sq / (np.pi * (n_dot_h**2 * (alpha_sq - 1.0) + 1.0) ** 2)
    fresnel = base_reflectance + (1.0 - base_reflectance) * (1.0 - v_dot_h) ** 5

    def g1(direction: np.ndarray) -> float:
        n_dot_w = float(np.dot(surface, direction))
        return (2.0 * n_dot_w) / (n_dot_w + np.sqrt(alpha_sq + (1.0 - alpha_sq) * n_dot_w**2))

    expected = (distribution * fresnel * (g1(view) * g1(light))) / (4.0 * n_dot_l * n_dot_v)
    scalar_result = ieCookTorrance(surface_normal, view_direction, light_direction, 0.04, roughness)

    assert result.shape == (3,)
    assert np.allclose(result, expected, atol=1e-12, rtol=1e-12)
    assert isinstance(scalar_result, float)
    assert np.isclose(scalar_result, expected[0], atol=1e-12, rtol=1e-12)


def test_macbeth_sensor_and_gretag_wrappers(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_adjust_luminance(scene, 75.0, asset_store=asset_store)
    scene = scene_set(scene, "fov", 2.64)
    scene = scene_set(scene, "distance", 10.0)

    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "optics fnumber", 4.0)
    oi = oi_set(oi, "optics focal length", 20.0e-3)
    oi = oi_set(oi, "optics off axis method", "skip")
    oi = oi_compute(oi, scene)

    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)
    sensor = sensor_compute(sensor, oi)

    sensor_corner_points = np.array([[1.0, 244.0], [328.0, 246.0], [329.0, 28.0], [2.0, 27.0]], dtype=float)
    sensor_img, sensor_sd, resolved_corner_points = macbethSensorValues(sensor, False, sensor_corner_points)

    assert sensor_img.shape == (24, 3)
    assert sensor_sd.shape == (24, 3)
    assert np.array_equal(resolved_corner_points, sensor_corner_points)
    assert np.all(np.isfinite(sensor_img))

    ip = ip_create(asset_store=asset_store)
    ip = ip_set(ip, "scale display", 1)
    ip = ip_compute(ip, sensor, asset_store=asset_store)
    ip_corner_points = np.array([[4.0, 246.0], [328.0, 243.0], [327.0, 26.0], [3.0, 27.0]], dtype=float)
    ip, y_noise, m_rgb = macbethLuminanceNoise(ip, ip_corner_points, asset_store=asset_store)

    assert np.array_equal(np.asarray(ip_get(ip, "chart corner points"), dtype=float), ip_corner_points)
    assert y_noise.shape == (6,)
    assert np.all(np.isfinite(y_noise))
    assert len(m_rgb) == 24
    assert m_rgb[0].shape[1] == 3

    gretag = macbethGretagSGCreate(asset_store=asset_store)
    assert gretag.name == "GretagSC"
    assert tuple(scene_get(gretag, "size")) == (305, 425)
    assert np.isclose(float(scene_get(gretag, "mean luminance", asset_store=asset_store)), 100.0, atol=1e-6, rtol=1e-6)


def test_scene_pattern_wrapper_surface(asset_store) -> None:
    hdr_lights = scene_create("hdr lights", asset_store=asset_store)
    hdr_lights_empty = scene_create("hdr lights", [], asset_store=asset_store)
    hdr_lights_kv = scene_create(
        "hdr lights",
        "n circles",
        4,
        "radius",
        [0.01, 0.01, 0.01, 0.01],
        "circle colors",
        ["white"],
        asset_store=asset_store,
    )

    assert tuple(scene_get(hdr_lights, "size")) == (384, 384)
    assert tuple(scene_get(hdr_lights_empty, "size")) == (384, 384)
    assert tuple(scene_get(hdr_lights_kv, "size")) == (384, 384)
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_lights_empty, "photons"), dtype=float),
        np.asarray(scene_get(hdr_lights, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.asarray(scene_get(hdr_lights_kv, "luminance", asset_store=asset_store), dtype=float)) > np.min(
        np.asarray(scene_get(hdr_lights_kv, "luminance", asset_store=asset_store), dtype=float)
    )

    hdr_chart = sceneHDRChart(1.0e3, 5, 4, 50.0, asset_store=asset_store)
    hdr_chart_alias = scene_create("hdr chart", 1.0e3, 5, 4, 50.0, asset_store=asset_store)
    hdr_chart_kv = scene_create(
        "hdr chart",
        "d range",
        1.0e3,
        "n levels",
        5,
        "cols per level",
        4,
        asset_store=asset_store,
    )
    hdr_chart_default = scene_create("hdr chart", asset_store=asset_store)
    hdr_chart_default_explicit = scene_create(
        "hdr chart",
        "d range",
        10**3.5,
        "n levels",
        16,
        "cols per level",
        12,
        asset_store=asset_store,
    )
    hdr_chart_placeholder = scene_create("hdr chart", [], [], [], [], [], asset_store=asset_store)
    hdr_chart_placeholder_kv = scene_create(
        "hdr chart",
        "d range",
        [],
        "n levels",
        [],
        "cols per level",
        [],
        "max l",
        [],
        "illuminant",
        [],
        asset_store=asset_store,
    )
    hdr_chart_placeholder_explicit = sceneHDRChart(1.0e4, 12, 8, None, None, asset_store=asset_store)

    hdr_luminance = np.asarray(scene_get(hdr_chart, "luminance", asset_store=asset_store), dtype=float)
    assert tuple(scene_get(hdr_chart, "size")) == (20, 20)
    assert np.isclose(float(np.max(hdr_luminance)), 50.0, atol=1e-6, rtol=1e-6)
    assert tuple(scene_get(hdr_chart_alias, "size")) == (20, 20)
    assert tuple(scene_get(hdr_chart_kv, "size")) == (20, 20)
    assert tuple(scene_get(hdr_chart_default, "size")) == (192, 192)
    hdr_chart_params = scene_get(hdr_chart_kv, "chart parameters")
    assert int(hdr_chart_params["nLevels"]) == 5
    assert int(hdr_chart_params["colsPerLevel"]) == 4
    assert float(hdr_chart_params["dRange"]) == pytest.approx(1.0e3)
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_chart_default, "photons"), dtype=float),
        np.asarray(scene_get(hdr_chart_default_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_chart_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(hdr_chart_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_chart_placeholder_kv, "photons"), dtype=float),
        np.asarray(scene_get(hdr_chart_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )

    hdr_scene, hdr_background = sceneHDRImage(
        3,
        background="",
        image_size=[96, 96],
        dynamic_range=2.0,
        patch_shape="circle",
        patch_size=8,
        row=40,
        asset_store=asset_store,
    )
    hdr_scene_alias = scene_create(
        "hdr image",
        3,
        {"background": "", "image size": [96, 96], "dynamic range": 2.0, "patch shape": "circle", "patch size": 8, "row": 40},
        asset_store=asset_store,
    )
    hdr_scene_kv = scene_create(
        "hdr image",
        "n patches",
        3,
        "background",
        "",
        "image size",
        [96, 96],
        "dynamic range",
        2.0,
        "patch shape",
        "circle",
        "patch size",
        8,
        "row",
        40,
        asset_store=asset_store,
    )
    hdr_scene_default = scene_create("hdr image", asset_store=asset_store)
    hdr_scene_default_explicit = scene_create("hdr image", "n patches", 8, asset_store=asset_store)
    hdr_scene_placeholder = scene_create(
        "hdr image",
        [],
        {"background": "", "image size": [96, 96]},
        asset_store=asset_store,
    )
    hdr_scene_placeholder_kv = scene_create(
        "hdr image",
        "n patches",
        [],
        "background",
        "",
        "image size",
        [96, 96],
        asset_store=asset_store,
    )
    hdr_scene_placeholder_explicit = scene_create(
        "hdr image",
        8,
        {"background": "", "image size": [96, 96]},
        asset_store=asset_store,
    )

    assert tuple(scene_get(hdr_scene, "size")) == (96, 96)
    assert tuple(scene_get(hdr_background, "size")) == (96, 96)
    assert tuple(scene_get(hdr_scene_alias, "size")) == (96, 96)
    assert tuple(scene_get(hdr_scene_kv, "size")) == (96, 96)
    assert np.max(np.asarray(scene_get(hdr_scene, "luminance", asset_store=asset_store), dtype=float)) > 50.0
    assert np.allclose(np.asarray(scene_get(hdr_background, "photons"), dtype=float), 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_scene_kv, "photons"), dtype=float),
        np.asarray(scene_get(hdr_scene_alias, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert int(scene_get(hdr_scene_default, "chart parameters")["nPatches"]) == 8
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_scene_default, "photons"), dtype=float),
        np.asarray(scene_get(hdr_scene_default_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_scene_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(hdr_scene_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(hdr_scene_placeholder_kv, "photons"), dtype=float),
        np.asarray(scene_get(hdr_scene_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    radiance = np.linspace(1.0e12, 5.0e12, wave.size * 4, dtype=float).reshape(wave.size, 4, order="F")
    radiance_scene, rc_size = sceneRadianceChart(wave, radiance, rowcol=[2, 2], patch_size=6, gray_fill=True, asset_store=asset_store)
    radiance_alias = scene_create("radiance chart", wave, radiance, {"rowcol": [2, 2], "patch size": 6, "gray fill": True}, asset_store=asset_store)
    radiance_kv = scene_create(
        "radiance chart",
        wave,
        radiance,
        "rowcol",
        [2, 2],
        "patch size",
        6,
        "gray fill",
        True,
        asset_store=asset_store,
    )
    radiance_positional = scene_create(
        "radiance chart",
        wave,
        radiance,
        [2, 2],
        6,
        True,
        asset_store=asset_store,
    )
    radiance_placeholder = scene_create(
        "radiance chart",
        wave,
        radiance,
        "rowcol",
        [],
        "patch size",
        [],
        "gray fill",
        [],
        asset_store=asset_store,
    )
    radiance_placeholder_struct = scene_create(
        "radiance chart",
        wave,
        radiance,
        {"rowcol": [], "patch size": [], "gray fill": []},
        asset_store=asset_store,
    )
    radiance_placeholder_positional = scene_create(
        "radiance chart",
        wave,
        radiance,
        [],
        [],
        [],
        asset_store=asset_store,
    )
    radiance_placeholder_explicit = scene_create(
        "radiance chart",
        wave,
        radiance,
        {"patch size": 10, "gray fill": True},
        asset_store=asset_store,
    )

    chart_parameters = scene_get(radiance_scene, "chart parameters")
    assert tuple(rc_size) == (2, 3)
    assert tuple(scene_get(radiance_scene, "size")) == (12, 18)
    assert tuple(scene_get(radiance_alias, "size")) == (12, 18)
    assert tuple(scene_get(radiance_kv, "size")) == (12, 18)
    assert tuple(np.asarray(chart_parameters["rowcol"], dtype=int)) == (2, 3)
    assert np.asarray(scene_get(radiance_scene, "photons"), dtype=float).shape == (12, 18, 31)
    np.testing.assert_allclose(
        np.asarray(scene_get(radiance_kv, "photons"), dtype=float),
        np.asarray(scene_get(radiance_alias, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(radiance_positional, "photons"), dtype=float),
        np.asarray(scene_get(radiance_alias, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(radiance_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(radiance_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(radiance_placeholder_struct, "photons"), dtype=float),
        np.asarray(scene_get(radiance_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(radiance_placeholder_positional, "photons"), dtype=float),
        np.asarray(scene_get(radiance_placeholder_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_scene_vernier_wrapper_surface(asset_store) -> None:
    display_scene = sceneVernier(
        scene_create("empty", asset_store=asset_store),
        "display",
        {"sceneSz": 64, "barWidth": 3, "offset": 2, "barLength": 20, "barColor": [1.0, 0.5, 0.5], "bgColor": [0.0, 0.0, 0.0], "meanLum": 10.0},
        asset_store=asset_store,
    )
    object_scene = sceneVernier(
        scene_create("empty", asset_store=asset_store),
        "object",
        {"sceneSz": 64, "barWidth": 3, "offset": 2, "barReflect": 0.7, "bgReflect": 0.2, "meanLum": 12.0},
        asset_store=asset_store,
    )
    object_alias = scene_create(
        "vernier",
        scene_create("empty", asset_store=asset_store),
        "object",
        {"sceneSz": 64, "barWidth": 3, "offset": 2, "barReflect": 0.7, "bgReflect": 0.2, "meanLum": 12.0},
        asset_store=asset_store,
    )
    shorthand_alias = scene_create(
        "vernier",
        "object",
        {"sceneSz": 64, "barWidth": 3, "offset": 2, "barReflect": 0.7, "bgReflect": 0.2, "meanLum": 12.0},
        asset_store=asset_store,
    )
    legacy_reference = sceneVernier(
        scene_create("empty", asset_store=asset_store),
        "object",
        {"sceneSz": 64, "barWidth": 3, "offset": 2, "barReflect": 0.7, "bgReflect": 0.2},
        asset_store=asset_store,
    )
    legacy_alias = scene_create("vernier", 64, 3, 2, 0.7, 0.2, asset_store=asset_store)
    placeholder_reference = scene_create("vernier", 65, 0, 1, 0.6, 0.3, asset_store=asset_store)
    placeholder_alias = scene_create("vernier", 65, [], [], [], [], asset_store=asset_store)
    fully_placeholder_reference = scene_create("vernier", 64, 0, 1, 0.6, 0.3, asset_store=asset_store)
    fully_placeholder_alias = scene_create("vernier", [], [], [], [], [], asset_store=asset_store)
    default_reference = sceneVernier(
        scene_create("empty", asset_store=asset_store),
        "object",
        {"sceneSz": 65, "barWidth": 3, "offset": 3, "barReflect": 0.6, "bgReflect": 0.3},
        asset_store=asset_store,
    )
    default_alias = scene_create("vernier", asset_store=asset_store)

    assert tuple(scene_get(display_scene, "size")) == (64, 64)
    assert np.isclose(float(scene_get(display_scene, "mean luminance", asset_store=asset_store)), 10.0, atol=1e-6, rtol=1e-6)
    assert tuple(scene_get(object_scene, "size")) == (64, 65)
    assert tuple(scene_get(object_alias, "size")) == (64, 65)
    assert tuple(scene_get(shorthand_alias, "size")) == (64, 65)
    assert tuple(scene_get(legacy_alias, "size")) == (64, 65)
    assert tuple(scene_get(placeholder_alias, "size")) == (65, 65)
    assert tuple(scene_get(fully_placeholder_alias, "size")) == (64, 65)
    assert tuple(scene_get(default_alias, "size")) == (65, 65)
    assert object_scene.name == "vernier-2"
    assert shorthand_alias.name == "vernier-2"
    assert legacy_alias.name == "vernier-2"
    assert placeholder_alias.name == "vernier-1"
    assert fully_placeholder_alias.name == "vernier-1"
    assert default_alias.name == "vernier-3"
    assert np.isclose(float(scene_get(object_scene, "mean luminance", asset_store=asset_store)), 12.0, atol=1e-6, rtol=1e-6)
    assert np.max(np.asarray(scene_get(object_scene, "luminance", asset_store=asset_store), dtype=float)) > np.min(
        np.asarray(scene_get(object_scene, "luminance", asset_store=asset_store), dtype=float)
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(shorthand_alias, "photons"), dtype=float),
        np.asarray(scene_get(object_alias, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(legacy_alias, "photons"), dtype=float),
        np.asarray(scene_get(legacy_reference, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_alias, "photons"), dtype=float),
        np.asarray(scene_get(placeholder_reference, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(fully_placeholder_alias, "photons"), dtype=float),
        np.asarray(scene_get(fully_placeholder_reference, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(default_alias, "photons"), dtype=float),
        np.asarray(scene_get(default_reference, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_remaining_utility_image_wrapper_surface() -> None:
    field_heights = np.array([0.1, 0.3, 0.7], dtype=float)
    leaves_options = {
        "rmin": 0.25,
        "rmax": 0.25,
        "nbr_iter": 2,
        "shape": "square",
        "rseed": 17,
        "random_samples": np.array(
            [
                [0.0, 0.25, 0.25, 0.2],
                [0.0, 0.75, 0.75, 0.8],
            ],
            dtype=float,
        ),
    }

    leaves, rseed = imageDeadLeaves([8, 8], 2.0, leaves_options)
    vernier, params = imageVernier({"sceneSz": 8, "barWidth": 2, "offset": 1, "gap": 2, "barColor": 1.0, "bgColor": 0.25})

    assert ieFieldHeight2Index(field_heights, 0.6) == ie_field_height_to_index(field_heights, 0.6) == 3
    assert rseed == 17
    assert np.allclose(leaves, imgDeadleaves([8, 8], 2.0, leaves_options))
    assert vernier.shape == (8, 8, 3)
    assert tuple(np.asarray(params["sceneSz"], dtype=int)) == (8, 8)
    assert params["barLength"] == 8
    assert params["gap"] == 2
    assert np.any(np.isclose(vernier, 1.0))
    assert np.any(np.isclose(vernier, 0.25))


def test_scene_export_wrapper_surface(asset_store, tmp_path: Path) -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    base = scene_create("uniform ee", np.array([5, 6], dtype=int), wave, asset_store=asset_store)
    base_photons = np.arange(1, 5 * 6 * wave.size + 1, dtype=float).reshape(5, 6, wave.size)
    base = scene_set(base, "photons", base_photons)

    inset = scene_create("uniform ee", np.array([2, 3], dtype=int), wave, asset_store=asset_store)
    inset_photons = np.full((2, 3, wave.size), 999.0, dtype=float)
    inset = scene_set(inset, "photons", inset_photons)

    inserted = sceneInsert(base, inset, [2, 3])
    expected = base_photons.copy()
    expected[1:3, 2:5, :] = inset_photons

    assert np.array_equal(np.asarray(scene_get(inserted, "photons"), dtype=float), expected)
    assert np.array_equal(np.asarray(scene_get(base, "photons"), dtype=float), base_photons)

    raw_path = Path(tmp_path / "scene_export")
    raw_var, raw_bases = sceneToFile(raw_path, inserted)
    raw_output = raw_path.with_suffix(".mat")
    raw_reload = scene_from_file(raw_output, "multispectral", asset_store=asset_store)

    assert raw_output.exists()
    assert np.isclose(raw_var, 1.0, atol=1e-12, rtol=0.0)
    assert raw_bases == wave.size
    assert np.array_equal(np.asarray(scene_get(raw_reload, "wave"), dtype=float), wave)
    assert np.allclose(np.asarray(scene_get(raw_reload, "photons"), dtype=float), expected, atol=1e-12, rtol=1e-12)

    basis_path = Path(tmp_path / "scene_basis_export")
    basis_var, basis_bases = sceneToFile(basis_path, inserted, 3, "mean svd")
    basis_output = basis_path.with_suffix(".mat")
    basis_payload = asset_store.load_mat(basis_output)
    basis_reload = sceneFromBasis(basis_payload, asset_store=asset_store)

    assert basis_output.exists()
    assert basis_bases == 3
    assert basis_var >= 0.999999999
    assert np.array_equal(np.asarray(scene_get(basis_reload, "wave"), dtype=float), wave)
    assert np.allclose(np.asarray(scene_get(basis_reload, "photons"), dtype=float), expected, atol=1e-10, rtol=1e-10)


def test_scene_waveband_and_video_wrapper_surface(asset_store, tmp_path: Path) -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    scene = scene_create("uniform ee", 4, wave, asset_store=asset_store)
    photons = np.arange(1, 4 * 4 * wave.size + 1, dtype=float).reshape(4, 4, wave.size)
    scene = scene_set(scene, "photons", photons)
    scene.name = "wb-scene"

    work_dir = Path(sceneWBCreate(scene, tmp_path / "wb-scenes"))
    saved_files = sorted(work_dir.glob("scene*.mat"))
    loaded_band, loaded_path = vcLoadObject("scene", saved_files[1])

    assert work_dir.exists()
    assert len(saved_files) == wave.size
    assert Path(loaded_path) == saved_files[1]
    assert np.array_equal(np.asarray(scene_get(loaded_band, "wave"), dtype=float), np.array([wave[1]], dtype=float))
    assert np.allclose(np.asarray(scene_get(loaded_band, "photons"), dtype=float), photons[:, :, 1], atol=1e-12, rtol=1e-12)

    brighter = scene_adjust_luminance(scene.clone(), 150.0, asset_store=asset_store)
    video_path = Path(sceneMakeVideo([scene, brighter], tmp_path / "scene-demo", asset_store=asset_store))
    frames = list(iio.imiter(video_path))

    assert video_path.suffix == ".gif"
    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert len(frames) == 2


def test_scene_from_multispectral_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
        "multispectral",
        None,
        None,
        wave,
        asset_store=asset_store,
    )
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    illuminant_energy = np.asarray(scene_get(scene, "illuminant energy"), dtype=float).reshape(-1)
    reflectance = np.asarray(scene_get(scene, "reflectance"), dtype=float)
    center_row = (photons.shape[0] - 1) // 2
    center_col = (photons.shape[1] - 1) // 2

    assert tuple(scene_get(scene, "size")) == (506, 759)
    assert np.array_equal(np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1), wave)
    assert photons.shape == (506, 759, 31)
    assert illuminant_energy.shape == (31,)
    assert scene_get(scene, "illuminant format") == "spectral"
    assert np.isclose(float(scene_get(scene, "mean luminance", asset_store=asset_store)), 30.047072285059308, atol=1e-6, rtol=1e-6)
    assert np.all(np.mean(photons, axis=(0, 1)) > 0.0)
    assert np.max(illuminant_energy) > np.min(illuminant_energy)
    assert np.all(np.mean(reflectance, axis=(0, 1)) > 0.19)
    assert np.all(np.mean(reflectance, axis=(0, 1)) < 0.31)
    assert np.all(np.isfinite(reflectance[center_row, center_col, :]))
    assert np.all(reflectance[center_row, center_col, :] >= 0.0)
    assert float(np.max(reflectance[center_row, center_col, :])) > 0.0


def test_scene_from_rgb_vs_multispectral_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
        "multispectral",
        None,
        None,
        wave,
        asset_store=asset_store,
    )
    scene = scene_adjust_illuminant(
        scene,
        blackbody(np.asarray(scene_get(scene, "wave"), dtype=float), 6500.0, kind="energy"),
        asset_store=asset_store,
    )
    source_rgb = np.asarray(scene_get(scene, "rgb", asset_store=asset_store), dtype=float)
    source_xyz = np.asarray(scene_get(scene, "xyz", asset_store=asset_store), dtype=float)
    mean_luminance = float(scene_get(scene, "mean luminance", asset_store=asset_store))

    display = display_create("LCD-Apple.mat", asset_store=asset_store)
    reconstructed = scene_from_file(source_rgb, "rgb", mean_luminance, display, asset_store=asset_store)
    reconstructed = scene_adjust_illuminant(
        reconstructed,
        blackbody(np.asarray(scene_get(reconstructed, "wave"), dtype=float), 6500.0, kind="energy"),
        asset_store=asset_store,
    )
    reconstructed = scene_adjust_luminance(reconstructed, mean_luminance, asset_store=asset_store)
    reconstructed_rgb = np.asarray(scene_get(reconstructed, "rgb", asset_store=asset_store), dtype=float)
    reconstructed_xyz = np.asarray(scene_get(reconstructed, "xyz", asset_store=asset_store), dtype=float)

    source_mean_xyz = np.mean(source_xyz, axis=(0, 1))
    reconstructed_mean_xyz = np.mean(reconstructed_xyz, axis=(0, 1))
    rgb_rmse = np.sqrt(np.mean(np.square(reconstructed_rgb - source_rgb), axis=(0, 1)))
    xyz_rmse_ratio = np.sqrt(np.mean(np.square(reconstructed_xyz - source_xyz), axis=(0, 1))) / np.maximum(source_mean_xyz, 1e-12)

    assert tuple(scene_get(scene, "size")) == (506, 759)
    assert source_rgb.shape == (506, 759, 3)
    assert np.all(source_rgb >= 0.0)
    assert np.all(source_rgb <= 1.0)
    assert np.isclose(mean_luminance, 30.047072285059308, atol=1e-6, rtol=1e-6)
    assert tuple(scene_get(reconstructed, "size")) == (506, 759)
    assert np.asarray(scene_get(reconstructed, "wave"), dtype=float).reshape(-1).shape == (101,)
    assert reconstructed_rgb.shape == (506, 759, 3)
    assert np.isclose(float(scene_get(reconstructed, "mean luminance", asset_store=asset_store)), mean_luminance, atol=1e-8, rtol=1e-8)
    assert np.all(rgb_rmse < np.array([0.05, 0.04, 0.04], dtype=float))
    assert np.all(np.abs((reconstructed_mean_xyz / np.sum(reconstructed_mean_xyz)) - (source_mean_xyz / np.sum(source_mean_xyz))) < 7e-3)
    assert np.all(xyz_rmse_ratio < np.array([0.06, 0.07, 0.13], dtype=float))


def test_scene_surface_models_tutorial_workflow(asset_store) -> None:
    def _render_surface_model(
        xyz: np.ndarray,
        illuminant: np.ndarray,
        basis: np.ndarray,
        weights: np.ndarray,
        n_dims: int | None,
    ) -> np.ndarray:
        current_basis = basis if n_dims is None else basis[:, :n_dims]
        current_weights = weights if n_dims is None else weights[:n_dims, :]
        mcc_xyz = xyz.T @ (illuminant.reshape(-1, 1) * current_basis) @ current_weights
        mcc_xyz = 100.0 * (mcc_xyz / max(float(np.max(mcc_xyz[1, :])), 1e-12))
        rendered = xyz_to_srgb(xw_to_rgb_format(mcc_xyz.T, 4, 6))
        return image_flip(image_flip(rendered, "updown"), "leftright")

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    reflectance = macbeth_read_reflectance(wave, asset_store=asset_store)
    subset = macbeth_read_reflectance(wave, [1, 24], asset_store=asset_store)
    xyz = np.asarray(ie_read_spectra("XYZ", wave, asset_store=asset_store), dtype=float)
    d65 = np.asarray(ie_read_spectra("D65", wave, asset_store=asset_store), dtype=float).reshape(-1)
    u, singular_values, vh = np.linalg.svd(reflectance, full_matrices=False)
    weights = np.diag(singular_values) @ vh

    render_1 = _render_surface_model(xyz, d65, u, weights, 1)
    render_4 = _render_surface_model(xyz, d65, u, weights, 4)
    render_full = _render_surface_model(xyz, d65, u, weights, None)
    approx_rmse = np.array(
        [
            float(np.sqrt(np.mean(np.square(u[:, :n_dims] @ weights[:n_dims, :] - reflectance), dtype=float)))
            for n_dims in range(1, 5)
        ],
        dtype=float,
    )

    probe = np.arange(24, dtype=float).reshape(2, 3, 4)

    assert reflectance.shape == (31, 24)
    assert np.allclose(subset, reflectance[:, [0, 23]], atol=1e-12, rtol=1e-12)
    assert xyz.shape == (31, 3)
    assert d65.shape == (31,)
    assert u[:, :4].shape == (31, 4)
    assert singular_values.shape == (24,)
    assert np.all(singular_values[:-1] >= singular_values[1:] - 1e-12)
    assert approx_rmse.shape == (4,)
    assert np.all(approx_rmse[:-1] >= approx_rmse[1:] - 1e-12)
    assert np.array_equal(image_flip(probe, "updown"), probe[::-1, :, :])
    assert np.array_equal(image_flip(probe, "leftright"), probe[:, ::-1, :])

    for rendered in (render_1, render_4, render_full):
        assert rendered.shape == (4, 6, 3)
        assert np.all(np.isfinite(rendered))
        assert np.all(rendered >= 0.0)
        assert np.all(rendered <= 1.0)

    assert float(np.mean(np.abs(render_full - render_1))) > 0.01
    assert float(np.mean(np.abs(render_full - render_4))) > 1e-4


def test_color_reflectance_basis_script_workflow(asset_store) -> None:
    snapshot_root = asset_store.ensure()
    reflectance_dirs = [
        snapshot_root / "data/surfaces/reflectances",
        snapshot_root / "data/surfaces/charts/esser/reflectance",
    ]
    filenames = []
    for directory in reflectance_dirs:
        filenames.extend(sorted(path.name for path in directory.glob("*.mat")))

    wave = np.arange(400.0, 701.0, 5.0, dtype=float)
    selected = [filenames[index - 1] for index in (5, 12)]
    reflectances = np.empty((wave.size, 0), dtype=float)
    for filename in selected:
        current = np.asarray(ie_read_spectra(filename, wave, asset_store=asset_store), dtype=float)
        reflectances = np.concatenate((current, reflectances), axis=1)

    u, singular_values, vh = np.linalg.svd(reflectances, full_matrices=False)
    basis = u[:, :8]
    weights = (np.diag(singular_values) @ vh)[:8, :]
    approx = basis @ weights
    approx_rmse = float(np.sqrt(np.mean(np.square(approx - reflectances), dtype=float)))

    assert len(filenames) >= 12
    assert selected == ["MiniatureMacbethChart.mat", "munsell_matte.mat"]
    assert reflectances.shape == (61, 1293)
    assert basis.shape == (61, 8)
    assert singular_values.shape[0] == min(reflectances.shape)
    assert np.all(singular_values[:-1] >= singular_values[1:] - 1e-12)
    assert approx.shape == reflectances.shape
    assert approx_rmse < 0.05


def test_run_python_case_supports_color_reflectance_basis_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("color_reflectance_basis_small", asset_store=asset_store)

    assert int(case.payload["file_count"]) >= 12
    assert np.array_equal(case.payload["selected_indices"], np.array([5, 12], dtype=int))
    assert list(np.asarray(case.payload["selected_filenames"], dtype=object).reshape(-1)) == [
        "MiniatureMacbethChart.mat",
        "munsell_matte.mat",
    ]
    assert case.payload["wave"].shape == (61,)
    assert np.array_equal(case.payload["reflectance_shape"], np.array([61, 1293], dtype=int))
    assert case.payload["singular_values_first8"].shape == (8,)
    assert case.payload["basis_first4"].shape == (61, 4)
    assert case.payload["basis_projector_8"].shape == (61, 61)
    assert np.isscalar(case.payload["approx_rmse"])
    assert case.payload["approx_stats"].shape == (4,)


def test_scene_reflectance_samples_script_workflow(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 5.0, dtype=float)
    random_sources = [
        "MunsellSamples_Vhrel.mat",
        "Food_Vhrel.mat",
        "DupontPaintChip_Vhrel.mat",
        "skin/HyspexSkinReflectance.mat",
    ]
    random_counts = np.array([24, 24, 24, 24], dtype=int)
    reflectances, sampled_lists, sampled_wave = ie_reflectance_samples(
        random_sources,
        random_counts,
        wave,
        "no replacement",
        asset_store=asset_store,
    )
    replay_reflectances, replay_lists, replay_wave = ie_reflectance_samples(
        random_sources,
        sampled_lists,
        wave,
        "no replacement",
        asset_store=asset_store,
    )

    assert reflectances.shape == (61, 96)
    assert np.array_equal(sampled_wave, wave)
    assert np.array_equal(replay_wave, wave)
    assert [sample.size for sample in sampled_lists] == [24, 24, 24, 24]
    assert [np.unique(sample).size for sample in sampled_lists] == [24, 24, 24, 24]
    assert np.array_equal(reflectances, replay_reflectances)
    assert all(np.array_equal(first, second) for first, second in zip(sampled_lists, replay_lists, strict=True))
    assert float(np.min(reflectances)) >= 0.0
    assert float(np.max(reflectances)) <= 1.0

    norms = np.sqrt(np.maximum(np.sum(np.square(reflectances), axis=0, dtype=float), 1e-12))
    normalized = reflectances / norms.reshape(1, -1)
    singular_values = np.linalg.svd(normalized - np.mean(normalized, axis=1, keepdims=True), compute_uv=False)
    assert singular_values.shape == (61,)
    assert np.all(singular_values[:-1] >= singular_values[1:] - 1e-12)

    explicit_sources = [
        "MunsellSamples_Vhrel.mat",
        "DupontPaintChip_Vhrel.mat",
    ]
    explicit_lists = [
        np.arange(1, 61, dtype=int),
        np.arange(1, 61, dtype=int),
    ]
    explicit_reflectances, stored_lists, _ = ie_reflectance_samples(
        explicit_sources,
        explicit_lists,
        wave,
        asset_store=asset_store,
    )
    explicit_replay, _, _ = ie_reflectance_samples(
        explicit_sources,
        stored_lists,
        wave,
        asset_store=asset_store,
    )

    assert explicit_reflectances.shape == (61, 120)
    assert all(np.array_equal(expected, current) for expected, current in zip(explicit_lists, stored_lists, strict=True))
    assert np.array_equal(explicit_reflectances, explicit_replay)


def test_scene_reflectance_chart_basis_functions_script_workflow(asset_store) -> None:
    scene = scene_create("reflectance chart", asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    reflectance = np.asarray(scene_get(scene, "reflectance"), dtype=float)

    img_mean_999, basis_999, coef_999, var_999 = hc_basis(reflectance, 0.999, "canonical")
    img_mean_95, basis_95, coef_95, var_95 = hc_basis(reflectance, 0.95, "canonical")
    img_mean_5, basis_5, coef_5, var_5 = hc_basis(reflectance, 5, "canonical")

    assert tuple(scene_get(scene, "size")) == tuple(reflectance.shape[:2])
    assert np.array_equal(wave, np.arange(400.0, 701.0, 10.0, dtype=float))
    assert reflectance.shape[2] == wave.size
    assert img_mean_999.size == 0
    assert img_mean_95.size == 0
    assert img_mean_5.size == 0
    assert basis_999.shape[0] == wave.size
    assert basis_95.shape[0] == wave.size
    assert basis_5.shape == (wave.size, 5)
    assert coef_999.shape[:2] == reflectance.shape[:2]
    assert coef_95.shape[:2] == reflectance.shape[:2]
    assert coef_5.shape == (reflectance.shape[0], reflectance.shape[1], 5)
    assert basis_999.shape[1] >= basis_95.shape[1]
    assert var_999 >= 0.999
    assert var_95 >= 0.95
    assert var_999 >= var_5 >= var_95
    assert np.allclose(basis_5.T @ basis_5, np.eye(5), atol=1e-8, rtol=1e-8)

    reflectance_xw, rows, cols, _ = rgb_to_xw_format(reflectance)
    reconstructed_5 = rgb_to_xw_format(coef_5)[0] @ basis_5.T
    relative_rmse_5 = np.sqrt(np.mean(np.square(reconstructed_5 - reflectance_xw), axis=0)) / np.maximum(
        np.mean(np.abs(reflectance_xw), axis=0),
        1e-12,
    )

    assert rows == reflectance.shape[0]
    assert cols == reflectance.shape[1]
    assert float(np.mean(relative_rmse_5)) < 0.07
    assert float(np.max(relative_rmse_5)) < 0.21


def test_run_python_case_supports_scene_surface_models_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("scene_surface_models_small", asset_store=asset_store)
    payload = case.payload

    assert np.array_equal(payload["wave"], np.arange(400.0, 701.0, 10.0, dtype=float))
    assert tuple(payload["reflectance_shape"]) == (31, 24)
    assert payload["basis_first4"].shape == (31, 4)
    assert payload["singular_values_first6"].shape == (6,)
    assert payload["approx_rmse_1to4"].shape == (4,)
    assert np.all(payload["approx_rmse_1to4"][:-1] >= payload["approx_rmse_1to4"][1:] - 1e-12)
    assert payload["render_full_center_rgb"].shape == (3,)
    assert np.all(payload["render_full_center_rgb"] >= 0.0)
    assert np.all(payload["render_full_center_rgb"] <= 1.0)


def test_scene_roi_script_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    scene_size = np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1)
    roi = np.rint(np.array([scene_size[0] / 2.0, scene_size[1], 10.0, 10.0], dtype=float)).astype(int)

    photons = np.asarray(scene_get(scene, "roi photons", roi), dtype=float)
    mean_photons = np.asarray(scene_get(scene, "roi mean photons", roi), dtype=float).reshape(-1)
    energy = np.asarray(scene_get(scene, "roi energy", roi), dtype=float)
    mean_energy = np.asarray(scene_get(scene, "roi mean energy", roi), dtype=float).reshape(-1)
    illuminant_photons = np.asarray(scene_get(scene, "roi illuminant photons", roi), dtype=float)
    mean_illuminant_photons = np.asarray(
        scene_get(scene, "roi mean illuminant photons", roi),
        dtype=float,
    ).reshape(-1)
    reflectance_manual = np.divide(
        photons,
        illuminant_photons,
        out=np.zeros_like(photons),
        where=illuminant_photons > 0.0,
    )
    reflectance_direct = np.asarray(scene_get(scene, "roi reflectance", roi), dtype=float)
    mean_reflectance_direct = np.asarray(scene_get(scene, "roi mean reflectance", roi), dtype=float).reshape(-1)

    assert wave.shape == (31,)
    assert scene_size.shape == (2,)
    assert roi.shape == (4,)
    assert photons.ndim == 2
    assert photons.shape[1] == wave.size
    assert photons.shape == energy.shape == illuminant_photons.shape == reflectance_direct.shape
    assert photons.shape[0] == 121
    assert mean_photons.shape == (wave.size,)
    assert mean_energy.shape == (wave.size,)
    assert mean_illuminant_photons.shape == (wave.size,)
    assert mean_reflectance_direct.shape == (wave.size,)
    assert np.allclose(mean_photons, np.mean(photons, axis=0), atol=1e-12, rtol=1e-12)
    assert np.allclose(mean_energy, np.mean(energy, axis=0), atol=1e-12, rtol=1e-12)
    assert np.allclose(mean_illuminant_photons, np.mean(illuminant_photons, axis=0), atol=1e-12, rtol=1e-12)
    assert np.allclose(reflectance_direct, reflectance_manual, atol=1e-12, rtol=1e-12)
    assert np.allclose(mean_reflectance_direct, np.mean(reflectance_direct, axis=0), atol=1e-12, rtol=1e-12)


def test_scene_rotate_script_workflow(asset_store) -> None:
    scene = scene_create("star pattern", asset_store=asset_store)
    original_size = np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1)
    rate = 1.0
    n_frames = 50

    rotated_sizes = np.zeros((n_frames, 2), dtype=int)
    mean_luminance = np.zeros(n_frames, dtype=float)
    max_luminance = np.zeros(n_frames, dtype=float)
    center_luminance = np.zeros(n_frames, dtype=float)

    for frame_index in range(n_frames):
        rotated = scene_rotate(scene, (frame_index + 1) * rate)
        luminance = np.asarray(scene_get(rotated, "luminance", asset_store=asset_store), dtype=float)
        center_row = luminance.shape[0] // 2
        center_col = luminance.shape[1] // 2
        rotated_sizes[frame_index, :] = np.asarray(scene_get(rotated, "size"), dtype=int).reshape(-1)
        mean_luminance[frame_index] = float(np.mean(luminance))
        max_luminance[frame_index] = float(np.max(luminance))
        center_luminance[frame_index] = float(luminance[center_row, center_col])

    assert tuple(original_size) == (256, 256)
    assert np.all(rotated_sizes >= original_size.reshape(1, 2))
    assert np.array_equal(rotated_sizes[:, 0], rotated_sizes[:, 1])
    assert np.all(np.diff(rotated_sizes[:40, 0]) >= 0)
    assert int(np.max(rotated_sizes[:, 0])) == 362
    assert int(rotated_sizes[-1, 0]) == 362
    assert np.all(mean_luminance > 0.0)
    assert mean_luminance[-1] < mean_luminance[0]
    assert np.allclose(max_luminance, center_luminance, atol=1e-9, rtol=1e-9)
    assert np.allclose(max_luminance, max_luminance[0], atol=1e-9, rtol=1e-9)


def test_scene_wavelength_script_workflow(asset_store) -> None:
    source_scene = scene_create(asset_store=asset_store)
    source_wave = np.asarray(scene_get(source_scene, "wave"), dtype=float).reshape(-1)
    source_mean_luminance = float(scene_get(source_scene, "mean luminance", asset_store=asset_store))

    fine_scene = scene_set(source_scene.clone(), "wave", np.arange(400.0, 701.0, 5.0, dtype=float))
    fine_scene = scene_set(fine_scene, "name", "5 nm spacing")
    fine_wave = np.asarray(scene_get(fine_scene, "wave"), dtype=float).reshape(-1)
    fine_mean_luminance = float(scene_get(fine_scene, "mean luminance", asset_store=asset_store))

    narrow_scene = scene_set(fine_scene.clone(), "wave", np.arange(500.0, 601.0, 2.0, dtype=float))
    narrow_scene = scene_set(narrow_scene, "name", "2 nm narrow band spacing")
    narrow_wave = np.asarray(scene_get(narrow_scene, "wave"), dtype=float).reshape(-1)
    narrow_mean_luminance = float(scene_get(narrow_scene, "mean luminance", asset_store=asset_store))

    source_photons = np.asarray(scene_get(source_scene, "photons"), dtype=float)
    fine_photons = np.asarray(scene_get(fine_scene, "photons"), dtype=float)
    narrow_photons = np.asarray(scene_get(narrow_scene, "photons"), dtype=float)

    assert source_scene.name == "Macbeth D65"
    assert tuple(scene_get(source_scene, "size")) == (64, 96)
    assert np.array_equal(source_wave, np.arange(400.0, 701.0, 10.0, dtype=float))
    assert fine_scene.name == "5 nm spacing"
    assert tuple(scene_get(fine_scene, "size")) == (64, 96)
    assert np.array_equal(fine_wave, np.arange(400.0, 701.0, 5.0, dtype=float))
    assert narrow_scene.name == "2 nm narrow band spacing"
    assert tuple(scene_get(narrow_scene, "size")) == (64, 96)
    assert np.array_equal(narrow_wave, np.arange(500.0, 601.0, 2.0, dtype=float))
    assert source_photons.shape == (64, 96, 31)
    assert fine_photons.shape == (64, 96, 61)
    assert narrow_photons.shape == (64, 96, 51)
    assert source_mean_luminance == pytest.approx(100.0, rel=1e-10, abs=1e-10)
    assert fine_mean_luminance == pytest.approx(source_mean_luminance, rel=1e-10, abs=1e-10)
    assert narrow_mean_luminance == pytest.approx(source_mean_luminance, rel=1e-10, abs=1e-10)
    assert np.all(np.mean(source_photons, axis=(0, 1)) > 0.0)
    assert np.all(np.mean(fine_photons, axis=(0, 1)) > 0.0)
    assert np.all(np.mean(narrow_photons, axis=(0, 1)) > 0.0)


def test_scene_hc_compress_script_workflow(asset_store, tmp_path) -> None:
    source_path = asset_store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
    scene = scene_from_file(source_path, "multispectral", asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    illuminant = scene_get(scene, "illuminant")
    output_path = tmp_path / "hc_compress.mat"

    img_mean_95, img_basis_95, coef_95, var_95 = hc_basis(photons, 0.95)
    ie_save_multispectral_image(
        output_path,
        coef_95,
        {"basis": img_basis_95, "wave": wave},
        "Compressed using hcBasis with imgMean)",
        img_mean_95,
        illuminant,
        float(scene_get(scene, "fov")),
        float(scene_get(scene, "distance")),
        "hcCompress95",
    )
    scene_95 = scene_from_file(output_path, "multispectral", None, None, np.arange(400.0, 701.0, 5.0, dtype=float), asset_store=asset_store)

    img_mean_99, img_basis_99, coef_99, var_99 = hc_basis(photons, 0.99)
    ie_save_multispectral_image(
        output_path,
        coef_99,
        {"basis": img_basis_99, "wave": wave},
        "Compressed using hcBasis with imgMean)",
        img_mean_99,
        illuminant,
        float(scene_get(scene, "fov")),
        float(scene_get(scene, "distance")),
        "hcCompress99",
    )
    scene_99 = scene_from_file(output_path, "multispectral", None, None, np.arange(400.0, 701.0, 5.0, dtype=float), asset_store=asset_store)

    photons_95 = np.asarray(scene_get(scene_95, "photons"), dtype=float)
    photons_99 = np.asarray(scene_get(scene_99, "photons"), dtype=float)

    assert tuple(scene_get(scene, "size")) == (506, 759)
    assert np.array_equal(wave, np.arange(400.0, 701.0, 10.0, dtype=float))
    assert scene_95.name == "hcCompress95"
    assert scene_99.name == "hcCompress99"
    assert tuple(scene_get(scene_95, "size")) == tuple(scene_get(scene, "size"))
    assert tuple(scene_get(scene_99, "size")) == tuple(scene_get(scene, "size"))
    assert np.array_equal(np.asarray(scene_get(scene_95, "wave"), dtype=float).reshape(-1), np.arange(400.0, 701.0, 5.0, dtype=float))
    assert np.array_equal(np.asarray(scene_get(scene_99, "wave"), dtype=float).reshape(-1), np.arange(400.0, 701.0, 5.0, dtype=float))
    assert photons_95.shape == (506, 759, 61)
    assert photons_99.shape == (506, 759, 61)
    assert float(var_99) >= float(var_95) >= 0.95
    assert img_basis_99.shape[1] >= img_basis_95.shape[1]
    assert float(scene_get(scene_95, "mean luminance", asset_store=asset_store)) > 0.0
    assert float(scene_get(scene_99, "mean luminance", asset_store=asset_store)) > 0.0
    mean_95 = np.mean(photons_95, axis=(0, 1))
    mean_99 = np.mean(photons_99, axis=(0, 1))
    assert np.all(mean_95 > 0.0)
    assert np.all(mean_99 > 0.0)
    assert float(np.mean(np.abs(mean_99 - mean_95) / np.maximum(np.abs(mean_99), 1e-12))) > 0.0


def test_run_python_case_supports_sensor_macbeth_daylight_estimate_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_macbeth_daylight_estimate_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["reflectance"].shape == (31, 24)
    assert case.payload["sensor_filters"].shape == (31, 3)
    assert case.payload["day_basis_quanta"].shape == (31, 3)
    assert case.payload["true_weights"].shape == (3,)
    assert case.payload["illuminant_photons"].shape == (31,)
    assert case.payload["camera_data"].shape == (3, 24)
    assert case.payload["design_matrix"].shape == (72, 3)
    assert case.payload["camera_stacked"].shape == (72,)
    assert case.payload["normal_matrix"].shape == (3, 3)
    assert case.payload["rhs"].shape == (3,)
    assert case.payload["estimated_weights"].shape == (3,)
    assert case.payload["estimated_illuminant"].shape == (31,)


def test_sensor_spectral_radiometer_script_contract(asset_store) -> None:
    scene = scene_create("uniform d65", asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)

    wave = np.arange(400.0, 701.0, 1.0, dtype=float)
    filter_spectra, filter_names, _ = ie_read_color_filter(wave, "radiometer", asset_store=asset_store)
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", wave)
    sensor = sensor_set(sensor, "filter spectra", filter_spectra)
    sensor = sensor_set(sensor, "filter names", filter_names)
    sensor = sensor_set(sensor, "pattern", np.arange(1, filter_spectra.shape[1] + 1, dtype=int).reshape(1, -1))
    sensor = sensor_set(sensor, "size", [10, filter_spectra.shape[1]])
    sensor = sensor_set(sensor, "pixel fill factor", 1.0)
    sensor = sensor_set(sensor, "pixel size same fill factor", np.array([1.5e-6, 1.5e-6], dtype=float))
    sensor = sensor_set(sensor, "exposure time", 1.0 / 100.0)

    sensor_noisy = sensor_set(sensor.clone(), "noise flag", -2)
    sensor_noisy = sensor_compute(sensor_noisy, oi, seed=0)
    noisy_electrons = np.asarray(sensor_get(sensor_noisy, "electrons"), dtype=float)

    sensor_noise_free = sensor_set(sensor.clone(), "noise flag", -1)
    sensor_noise_free = sensor_compute(sensor_noise_free, oi)
    noise_free_electrons = np.asarray(sensor_get(sensor_noise_free, "electrons"), dtype=float)

    assert noisy_electrons.shape == (10, filter_spectra.shape[1])
    assert noise_free_electrons.shape == (10, filter_spectra.shape[1])
    assert np.array_equal(np.asarray(sensor_get(sensor_noise_free, "pattern"), dtype=int), np.arange(1, filter_spectra.shape[1] + 1, dtype=int).reshape(1, -1))
    assert float(np.mean(noise_free_electrons)) > 0.0
    assert np.all(np.sqrt(np.maximum(noise_free_electrons[4, :], 0.0)) >= 0.0)
    assert abs(float(np.mean(noisy_electrons[4, :])) - float(np.mean(noise_free_electrons[4, :]))) / float(np.mean(noise_free_electrons[4, :])) < 0.15


def test_run_python_case_supports_sensor_spectral_radiometer_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_spectral_radiometer_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (301,)
    assert case.payload["w_samples"].shape == (76,)
    assert case.payload["filter_pattern"].shape == (1, 76)
    assert tuple(case.payload["sensor_size"]) == (10, 76)
    assert case.payload["filter_spectra"].shape == (301, 76)
    assert case.payload["noise_free_line"].shape == (76,)
    assert case.payload["shot_sd_line"].shape == (76,)
    assert case.payload["noisy_line_stats"].shape == (4,)
    assert case.payload["noisy_full_stats"].shape == (4,)
    assert case.payload["noise_free_full_stats"].shape == (4,)


def test_run_python_case_supports_sensor_spectral_estimation_small_parity_case(asset_store) -> None:
    case = run_python_case_with_context("sensor_spectral_estimation_small", asset_store=asset_store)

    assert case.payload["wave"].shape == (31,)
    assert case.payload["centers"].shape == (7,)
    assert case.payload["spd"].shape == (31, 7)
    assert case.payload["exposure_times"].shape == (7,)
    assert case.payload["responsivity"].shape == (3, 7)
    assert case.payload["weights"].shape == (3, 7)
    assert case.payload["estimated_filters"].shape == (31, 3)
    assert case.payload["sensor_filters"].shape == (31, 3)
    assert float(np.max(case.payload["estimated_filters"])) <= 1.0 + 1e-8
    assert float(np.max(case.payload["sensor_filters"])) <= 1.0 + 1e-8


def test_sensor_read_raw_tutorial_flow(asset_store) -> None:
    dng_path = asset_store.resolve("data/images/rawcamera/MCC-centered.dng")

    sensor, info = sensor_dng_read(
        dng_path,
        "full info",
        False,
        "crop",
        [500, 1000, 256, 256],
        asset_store=asset_store,
    )
    ip = ip_create(asset_store=asset_store)
    ip = ip_compute(ip, sensor)

    assert tuple(sensor_get(sensor, "size")) == (258, 258)
    assert np.array_equal(sensor_get(sensor, "pattern"), np.array([[2, 1], [3, 2]], dtype=int))
    assert float(sensor_get(sensor, "black level")) > 0.0
    assert float(info["isoSpeed"]) > 0.0

    rendered = np.asarray(ip_get(ip, "result"), dtype=float)
    assert rendered.shape == (258, 258, 3)
    assert np.isfinite(rendered).all()


def test_wvf_osa_index_helpers_round_trip() -> None:
    indices = np.array([0, 1, 2, 5, 15, 20, 35], dtype=int)

    n, m = wvf_osa_index_to_zernike_nm(indices)
    roundtrip = wvf_zernike_nm_to_osa_index(n, m)

    assert np.array_equal(n, np.array([0, 1, 1, 2, 5, 5, 7], dtype=int))
    assert np.array_equal(m, np.array([0, -1, 1, 2, -5, 5, 7], dtype=int))
    assert np.array_equal(roundtrip, indices)
    assert wvf_osa_index_to_zernike_nm(15) == (5, -5)
    assert wvf_zernike_nm_to_osa_index(5, -5) == 15


def test_wvf_osa_index_to_vector_index_supports_named_and_numeric_inputs() -> None:
    vector_index, j_index = wvf_osa_index_to_vector_index(np.array([0, 4, 12], dtype=int))
    named_vector_index, named_j_index = wvf_osa_index_to_vector_index(["defocus", "spherical"])

    assert np.array_equal(vector_index, np.array([1, 5, 13], dtype=int))
    assert np.array_equal(j_index, np.array([0, 4, 12], dtype=int))
    assert np.array_equal(named_vector_index, np.array([5, 13], dtype=int))
    assert np.array_equal(named_j_index, np.array([4, 12], dtype=int))
    assert wvf_osa_index_to_vector_index("defocus") == (5, 4)


def test_wvf_osa_index_to_name_supports_scalar_vector_and_invalid_inputs() -> None:
    assert wvfOSAIndexToName(0) == "piston"
    assert wvfOSAIndexToName([4, 12, 14]) == ["defocus", "spherical", "vertical_quadrafoil"]

    with pytest.raises(ValueError, match="Unknown index 99"):
        wvfOSAIndexToName(99)


def test_sce_create_and_wvf_sce_helpers_round_trip() -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    sce = sceCreate(wave, "berendschot_data", "applegate")
    assert np.array_equal(np.asarray(sceGet(sce, "wavelengths"), dtype=float), wave)
    assert np.allclose(np.asarray(sceGet(sce, "rho"), dtype=float), np.array([0.05775, 0.0410, 0.0490], dtype=float))
    assert np.isclose(float(sceGet(sce, "rho", 550.0)), 0.0410)
    assert np.isclose(float(sceGet(sce, "xo")), 0.51)
    assert np.isclose(float(sceGet(sce, "yo")), 0.20)

    model = sceCreate(np.array([550.0], dtype=float), "berendschot_model", "centered")
    assert np.isclose(float(sceGet(model, "rho", 550.0)), 0.0410, atol=1e-6)
    assert np.isclose(float(sceGet(model, "xo")), 0.0)
    assert np.isclose(float(sceGet(model, "yo")), 0.0)

    wvf = wvf_set(wvf_create(wave=wave), "sce params", sce)
    assert np.array_equal(np.asarray(wvf_get(wvf, "sce wavelengths"), dtype=float), wave)
    assert np.allclose(np.asarray(wvf_get(wvf, "sce rho"), dtype=float), np.asarray(sceGet(sce, "rho"), dtype=float))
    assert np.isclose(float(wvf_get(wvf, "sce rho", 550.0)), 0.0410)
    assert np.isclose(float(wvf_get(wvf, "scex0")), 0.51)
    assert np.isclose(float(wvf_get(wvf, "scey0")), 0.20)
    assert np.array_equal(np.asarray(sceGet(wvf_get(wvf, "sce"), "wave"), dtype=float), wave)


def test_wvf_cone_psf_and_criterion_radius_wrappers(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)
    wvf = wvf_create(wave=wave)
    wvf = wvf_set(wvf, "spatial samples", 81)
    wvf = wvf_set(wvf, "compute sce", True)
    wvf = wvf_compute(wvf)

    cone_psf, cone_sce_fraction = wvfComputeConePSF(wvf, asset_store=asset_store)
    assert cone_psf.shape == (81, 81, 3)
    assert np.allclose(np.sum(cone_psf, axis=(0, 1)), np.ones(3, dtype=float))

    info = wvf_get(wvf, "calc cone psf info")
    sensitivities = np.asarray(
        optics_module.interp_spectra(
            np.asarray(info["wavelengths"], dtype=float),
            np.asarray(info["spectral_sensitivities"], dtype=float),
            wave,
        ),
        dtype=float,
    )
    spd_weighting = np.asarray(
        optics_module.interp_spectra(
            np.asarray(info["wavelengths"], dtype=float),
            np.asarray(info["spectral_weighting"], dtype=float),
            wave,
        ),
        dtype=float,
    ).reshape(-1)
    spd_weighting = spd_weighting / np.sum(spd_weighting)
    cone_weights = sensitivities.T * spd_weighting[None, :]
    cone_weights = cone_weights / np.sum(cone_weights, axis=1, keepdims=True)

    psf = np.asarray(wvf_get(wvf, "psf"), dtype=float)
    expected_cone_psf = np.asarray(np.tensordot(psf, cone_weights.T, axes=([2], [0])), dtype=float)
    expected_cone_psf = expected_cone_psf / np.sum(expected_cone_psf, axis=(0, 1), keepdims=True)
    sce_fraction = np.asarray(wvf_get(wvf, "sce fraction", wave), dtype=float).reshape(-1)
    expected_cone_sce_fraction = np.asarray(cone_weights @ sce_fraction, dtype=float).reshape(-1)

    assert np.allclose(cone_psf, expected_cone_psf)
    assert np.allclose(cone_sce_fraction, expected_cone_sce_fraction)
    assert np.allclose(np.asarray(wvf_get(wvf, "cone psf"), dtype=float), cone_psf)
    assert np.allclose(np.asarray(wvf_get(wvf, "cone sce fraction"), dtype=float), cone_sce_fraction)

    avg_radius, cone_radii, updated = wvfComputeConeAverageCriterionRadius(wvf, 0.5, 0.5, asset_store=asset_store)
    expected_defocus = float(
        np.asarray(
            wvf_defocus_diopters_to_microns(0.5, wvf_get(wvf, "measured pupil size", "mm")),
            dtype=float,
        ).reshape(-1)[0]
    )
    assert np.isclose(float(wvf_get(updated, "zcoeffs", "defocus")), expected_defocus)

    updated_cone_psf, _ = wvfComputeConePSF(updated, asset_store=asset_store)
    expected_cone_radii = np.asarray(
        [psfFindCriterionRadius(updated_cone_psf[:, :, index], 0.5) for index in range(updated_cone_psf.shape[2])],
        dtype=float,
    )
    cone_weighting = np.asarray(wvf_get(updated, "calc cone psf info")["cone_weighting"], dtype=float).reshape(-1)
    expected_avg_radius = float(np.sum(cone_weighting * expected_cone_radii))

    assert np.allclose(cone_radii, expected_cone_radii)
    assert np.isclose(avg_radius, expected_avg_radius)


def test_wvf_pupil_function_compatibility_wrappers_match_explicit_compute() -> None:
    def custom_lca(measured_nm: float, this_nm: float) -> float:
        return 5e-4 * (float(this_nm) - float(measured_nm))

    wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
    wvf = wvf_set(wvf, "spatial samples", 61)
    wvf = wvf_set(wvf, "lca method", custom_lca)

    assert wvf_get(wvf, "lca method") is custom_lca

    custom = wvfComputePupilFunctionCustomLCA(wvf)
    manual = optics_module._wvf_compute_pupil_function_explicit(wvf)
    assert np.allclose(np.asarray(custom["pupil_function"]), np.asarray(manual["pupil_function"]))
    assert np.allclose(np.asarray(custom["wavefront_aberrations_um"]), np.asarray(manual["wavefront_aberrations_um"]))
    assert np.allclose(np.asarray(custom["areapix"]), np.asarray(manual["areapix"]))
    assert np.allclose(np.asarray(custom["areapixapod"]), np.asarray(manual["areapixapod"]))

    no_lca = wvfComputePupilFunctionCustomLCA(wvf, False, "no lca", True)
    manual_no_lca = optics_module._wvf_compute_pupil_function_explicit(wvf_set(wvf, "lca method", "none"))
    assert np.allclose(np.asarray(no_lca["pupil_function"]), np.asarray(manual_no_lca["pupil_function"]))
    assert np.allclose(np.asarray(no_lca["wavefront_aberrations_um"]), np.asarray(manual_no_lca["wavefront_aberrations_um"]))

    from_master = wvfComputePupilFunctionFromMaster(wvf)
    custom_from_master = wvfComputePupilFunctionCustomLCAFromMaster(wvf)
    assert np.allclose(np.asarray(from_master["pupil_function"]), np.asarray(manual["pupil_function"]))
    assert np.allclose(np.asarray(custom_from_master["pupil_function"]), np.asarray(manual["pupil_function"]))


def test_wvf_root_path_and_summary_helpers_match_headless_contract() -> None:
    root = Path(wvfRootPath())
    expected = Path("/Users/seongcheoljeong/Documents/CameraE2E/.cache/upstream/isetcam/412b9f9bdb3262f2552b96f0e769b5ad6cdff821/opticalimage/wavefront")
    assert root == expected
    assert root.name == "wavefront"

    wvf = wvf_create(wave=np.array([500.0, 600.0], dtype=float))
    wvf = wvf_set(wvf, "name", "summary-demo")
    summary = wvfSummarize(wvf)
    printed = wvfPrint(wvf)

    assert isinstance(printed, dict)
    assert printed["name"] == "summary-demo"
    assert "wavefront struct name: summary-demo" in summary
    assert "Summarizing for wave 500 nm." in summary
    assert "f number" in summary
    assert "calc pupil diam" in summary
    assert "zCoeffs:" in summary
    assert "Max OTF freq" in summary
    assert "Max PSF support" in summary


def test_wvf_key_synonyms_normalizes_strings_and_key_value_pairs() -> None:
    assert wvfKeySynonyms("measured wavelength") == "measuredwl"
    assert wvfKeySynonyms("zcoef") == "zcoeffs"

    canonical = wvfKeySynonyms(["wave", np.array([500.0, 600.0], dtype=float), "stiles crawford", "demo"])
    assert canonical[0] == "calcwavelengths"
    assert np.array_equal(np.asarray(canonical[1], dtype=float), np.array([500.0, 600.0], dtype=float))
    assert canonical[2] == "sceparams"
    assert canonical[3] == "demo"

    tuple_form = wvfKeySynonyms(("defocus diopters", 1.5))
    assert isinstance(tuple_form, tuple)
    assert tuple_form[0] == "calcobserverfocuscorrection"
    assert tuple_form[1] == 1.5


def test_wvf2sipsf_returns_shift_invariant_psf_data() -> None:
    wvf = wvf_create(wave=np.array([500.0, 600.0], dtype=float))
    wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")

    alias_data, alias_computed = wvf2SiPsf(wvf)
    direct_data, direct_computed = wvf2PSF(wvf, False)
    custom_data, _ = wvf2SiPsf(wvf, "nPSFSamples", 33, "umPerSample", 0.5)

    assert alias_computed["computed"] is True
    assert direct_computed["computed"] is True
    assert np.allclose(np.asarray(alias_data["psf"], dtype=float), np.asarray(direct_data["psf"], dtype=float))
    assert np.allclose(np.asarray(alias_data["wave"], dtype=float), np.array([500.0, 600.0], dtype=float))
    assert np.allclose(np.asarray(alias_data["umPerSamp"], dtype=float), np.array([0.25, 0.25], dtype=float))
    assert np.asarray(custom_data["psf"], dtype=float).shape == (33, 33, 2)
    assert np.allclose(np.asarray(custom_data["umPerSamp"], dtype=float), np.array([0.5, 0.5], dtype=float))


def test_wvf_apply_matches_direct_wvf_to_oi_compute_path(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    wvf = wvf_create(wave=np.asarray(scene_get(scene, "wave"), dtype=float))
    wvf = wvf_set(wvf, "zcoeffs", np.array([1.0], dtype=float), "defocus")

    applied = wvfApply(scene, wvf)
    direct = oi_compute(wvf_to_oi(wvf_compute_psf(wvf, "computepupilfunc", True)), scene)

    assert np.allclose(np.asarray(applied.data["photons"], dtype=float), np.asarray(direct.data["photons"], dtype=float))
    assert applied.fields["optics"]["wavefront"]["computed"] is True


def test_run_python_case_supports_wvf_osa_index_conversion_parity_case(asset_store) -> None:
    case = run_python_case_with_context("wvf_osa_index_conversion_small", asset_store=asset_store)

    assert np.array_equal(case.payload["indices"], np.array([0, 1, 2, 5, 15, 20, 35], dtype=int))
    assert np.array_equal(case.payload["roundtrip_indices"], case.payload["indices"])
    assert int(case.payload["scalar_index"]) == 15
    assert int(case.payload["scalar_n"]) == 5
    assert int(case.payload["scalar_m"]) == -5
    assert int(case.payload["scalar_roundtrip_index"]) == 15
