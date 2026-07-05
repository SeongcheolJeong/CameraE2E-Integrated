from __future__ import annotations

import base64
import hashlib
import importlib
import math

import numpy as np
import pytest
import pyisetcam.color as color_module
import pyisetcam.sensor as sensor_module
from scipy.signal import convolve2d

from pyisetcam import (
    FloydSteinberg,
    HalfToneImage,
    apply_channelwise_gaussian as top_level_apply_channelwise_gaussian,
    appendStruct,
    array_percentile as top_level_array_percentile,
    biNormal,
    cellDelete,
    cellMerge,
    chromaticity as top_level_chromaticity,
    chromaticityXY as top_level_chromaticityXY,
    chromaticity_xy as top_level_chromaticity_xy,
    compareFields,
    compStruct,
    convolvecirc,
    dpi2mperdot,
    Energy2Quanta,
    Quanta2Energy,
    example_spd_pair as top_level_example_spd_pair,
    expRand,
    ffndgrid,
    gammaPDF,
    gaussian_sigma_pixels as top_level_gaussian_sigma_pixels,
    getMiddleMatrix,
    getGaussian,
    gatherStruct,
    hcBasis,
    hcBlur,
    hcIlluminantScale,
    hcReadHyspex,
    hcReadHyspexImginfo,
    hcViewer,
    hcimage,
    hcimageCrop,
    hcimageRotateClip,
    ieCmap,
    ieClip,
    ieCompressData,
    ieContains,
    ieHash,
    ieCropRect,
    ieDataList,
    ieDpi2Mperdot,
    ieExprnd,
    ieFindFiles,
    ieFindWaveIndex,
    ieFractalDrawgrid,
    ieFractaldim,
    ieHwhm2SD,
    ieLineAlign,
    ieLightList,
    ieLUTDigital,
    ieLUTInvert,
    ieLUTLinear,
    ieMvnrnd,
    ieNormpdf,
    ieOneOverF,
    ieParameterOtype,
    iePoisson,
    iePrcomp,
    iePrctile,
    ieRadialMatrix,
    ieReflectanceList,
    ieScale,
    ieScaleColumns,
    ieSpace2Amp,
    ieTone,
    ieUnitScaleFactor,
    ieUncompressData,
    ieWave2Index,
    ieXYZFromPhotons,
    ieStructCompare,
    ieStructRemoveEmptyField,
    imageHparams,
    imageSPD,
    imageSPD2RGB,
    imageInterpolate,
    imageGabor,
    imageMakeMontage,
    imageMontage,
    imageSlantedEdge,
    imageTranslate,
    imageTranspose,
    listStruct,
    imagehc2rgb,
    imageBoundingBox,
    imageCentroid,
    imageCircular,
    imageContrast,
    interp_spectra as top_level_interp_spectra,
    invert_gamma_table as top_level_invert_gamma_table,
    internal_to_display_matrix as top_level_internal_to_display_matrix,
    imagescM,
    imagescOPP,
    imagescRGB,
    isodd,
    least_squares_matrix as top_level_least_squares_matrix,
    linear_to_srgb as top_level_linear_to_srgb,
    rgb2dac,
    rotationMatrix3d,
    sample2space,
    sceneCreate,
    sceneGet,
    space2sample,
    spectral_angle as top_level_spectral_angle,
    spectral_step as top_level_spectral_step,
    energy_to_quanta as top_level_energy_to_quanta,
    quanta_to_energy as top_level_quanta_to_energy,
    unitFrequencyList,
    unit_frequency_list as top_level_unit_frequency_list,
    qinterp2,
    resample_cube as top_level_resample_cube,
    unpadarray,
    upperQuad2FullMatrix,
    vectorLength,
    xyz_color_matching as top_level_xyz_color_matching,
    xyz_to_linear_srgb as top_level_xyz_to_linear_srgb,
    xyz2srgb,
    lorentzSum,
    max2,
    min2,
    replaceNaN,
    scComputeDifference,
    scGaussianParameters,
    sc_compute_difference as top_level_sc_compute_difference,
    sc_gaussian_parameters as top_level_sc_gaussian_parameters,
    sensor_to_target_matrix as top_level_sensor_to_target_matrix,
    sensor_to_xyz_matrix as top_level_sensor_to_xyz_matrix,
    struct2pairs,
    tile_pattern as top_level_tile_pattern,
    zernfun,
    zernfun2,
    zernpol,
)
from pyisetcam.utils import (
    apply_channelwise_gaussian,
    append_struct,
    blackbody,
    bi_normal,
    cell_delete,
    cell_merge,
    checkfields,
    compare_fields,
    comp_struct,
    convolve_circ,
    energy_to_quanta,
    exp_rand,
    ffndgrid as ffndgrid_fn,
    floyd_steinberg,
    gaussian_sigma_pixels,
    gather_struct,
    gamma_pdf,
    get_middle_matrix,
    get_gaussian,
    hc_basis,
    hc_blur,
    hc_illuminant_scale,
    hc_image,
    hc_image_crop,
    hc_image_rotate_clip,
    hc_read_hyspex,
    hc_read_hyspex_imginfo,
    hc_viewer,
    half_tone_image,
    ie_hash,
    ie_contains,
    ie_cmap,
    ie_clip,
    ie_compress_data,
    ie_crop_rect,
    ie_data_list,
    ie_dpi2_mperdot,
    ie_exprnd,
    ie_find_files,
    ie_fit_line,
    ie_find_wave_index,
    ie_fractal_dim,
    ie_fractal_drawgrid,
    ie_hwhm_to_sd,
    ie_line_align,
    ie_light_list,
    ie_lut_digital,
    ie_lut_invert,
    ie_lut_linear,
    ie_parameter_otype,
    ie_mvnrnd,
    ie_normpdf,
    ie_one_over_f,
    ie_poisson,
    ie_prcomp,
    ie_prctile,
    ie_radial_matrix,
    ie_reflectance_list,
    ie_scale,
    ie_scale_columns,
    ie_space_to_amp,
    ie_struct_compare,
    ie_struct_remove_empty_field,
    ie_tone,
    ie_unit_scale_factor,
    ie_uncompress_data,
    ie_wave2_index,
    image_hparams,
    image_hc2rgb,
    image_interpolate,
    image_gabor,
    image_make_montage,
    image_montage,
    image_slanted_edge,
    image_spd,
    image_spd2rgb,
    list_struct,
    image_translate,
    image_transpose,
    imagesc_m,
    imagesc_opp,
    imagesc_rgb,
    interp_spectra,
    image_bounding_box,
    image_centroid,
    image_circular,
    image_contrast,
    array_percentile,
    invert_gamma_table,
    least_squares_matrix,
    linear_to_srgb,
    param_format,
    qinterp2 as qinterp2_fn,
    quanta_to_energy,
    replace_nan,
    resample_cube,
    rgb_to_dac,
    rgb_to_xw_format,
    rotation_matrix_3d,
    sample2space as sample2space_fn,
    spectral_step,
    space2sample as space2sample_fn,
    struct2pairs as struct2pairs_fn,
    tile_pattern,
    dpi2mperdot as dpi2mperdot_fn,
    ie_tikhonov,
    lorentz_sum,
    max2 as max2_fn,
    min2 as min2_fn,
    unpadarray as unpadarray_fn,
    upper_quad_to_full_matrix,
    vector_length,
    unit_frequency_list,
    xyz_to_linear_srgb,
    zernfun as zernfun_fn,
    zernfun2 as zernfun2_fn,
    zernpol as zernpol_fn,
)
from pyisetcam.metrics import spectral_angle
from pyisetcam.color import (
    internal_to_display_matrix,
    sensor_to_target_matrix,
    sensor_to_xyz_matrix,
    xyz_color_matching,
)
from pyisetcam.metrics import example_spd_pair
from pyisetcam.scielab import sc_compute_difference, sc_gaussian_parameters

scielab_module = importlib.import_module("pyisetcam.scielab")
metrics_module = importlib.import_module("pyisetcam.metrics")


def test_color_module_energy_and_units_matlab_aliases() -> None:
    assert color_module.Energy2Quanta is color_module.energy_to_quanta
    assert color_module.Quanta2Energy is color_module.quanta_to_energy
    assert color_module.ieUnitScaleFactor is color_module.ie_unit_scale_factor


def test_color_module_photometry_matlab_aliases() -> None:
    assert color_module.ieXYZFromPhotons is color_module.ie_xyz_from_photons
    assert color_module.ieLuminance2Radiance is color_module.ie_luminance_to_radiance
    assert color_module.ieScotopicLuminanceFromEnergy is color_module.ie_scotopic_luminance_from_energy
    assert color_module.ieResponsivityConvert is color_module.ie_responsivity_convert
    assert color_module.ieLuminanceFromEnergy is color_module.luminance_from_energy
    assert color_module.ieLuminanceFromPhotons is color_module.luminance_from_photons
    assert color_module.xyz2srgb is color_module.xyz_to_srgb
    assert color_module.xyy2xyz is color_module.xyy_to_xyz


def test_color_module_color_space_matlab_aliases() -> None:
    assert color_module.srgbParameters is color_module.srgb_parameters
    assert color_module.adobergbParameters is color_module.adobergb_parameters
    assert color_module.srgb2lrgb is color_module.srgb_to_lrgb
    assert color_module.lrgb2srgb is color_module.lrgb_to_srgb
    assert color_module.Y2Lstar is color_module.y_to_lstar
    assert color_module.ieLAB2XYZ is color_module.ie_lab_to_xyz
    assert color_module.lms2srgb is color_module.lms_to_srgb
    assert color_module.lms2xyz is color_module.lms_to_xyz
    assert color_module.xyz2lms is color_module.xyz_to_lms


def test_color_module_cross_module_matlab_aliases() -> None:
    assert color_module.RGB2XWFormat is color_module.rgb_to_xw_format
    assert color_module.chromaticity is color_module.chromaticity_xy
    assert color_module.chromaticityXY is color_module.chromaticity_xy
    assert color_module.XW2RGBFormat is color_module.xw_to_rgb_format
    assert color_module.colorBlockMatrix is color_module.color_block_matrix
    assert color_module.srgb2xyz is color_module.srgb_to_xyz
    assert color_module.ieXYZFromEnergy is color_module.xyz_from_energy
    assert color_module.ieXYZ2LAB is color_module.ie_xyz_to_lab
    assert color_module.xyz2luv is color_module.xyz_to_luv
    assert color_module.xyz2uv is color_module.xyz_to_uv
    assert color_module.spd2cct is color_module.spd_to_cct
    assert color_module.srgb2colortemp is color_module.srgb_to_color_temp
    assert color_module.colorTransformMatrix is color_module.color_transform_matrix

    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array([1.0, 0.5, 0.25])
    xyz = color_module.ieXYZFromEnergy(energy, wave)
    assert np.allclose(xyz, metrics_module.xyz_from_energy(energy, wave))
    block_wave = np.arange(410.0, 691.0, 20.0, dtype=float)
    assert np.allclose(
        color_module.colorBlockMatrix(block_wave, 0.2),
        sensor_module.color_block_matrix(block_wave, 0.2),
    )
    assert np.allclose(
        color_module.colorTransformMatrix("xyz2opp"),
        scielab_module.color_transform_matrix("xyz2opp"),
    )
    xyz_sample = np.array([20.0, 30.0, 15.0], dtype=float)
    assert np.allclose(color_module.chromaticity(xyz_sample), metrics_module.chromaticity_xy(xyz_sample))
    assert top_level_chromaticity is top_level_chromaticity_xy
    assert top_level_chromaticityXY is top_level_chromaticity_xy


def test_scielab_module_native_matlab_aliases() -> None:
    assert scielab_module.getPlanes is scielab_module.get_planes
    assert scielab_module.scParams is scielab_module.sc_params
    assert scielab_module.separableConv is scielab_module.separable_conv
    assert scielab_module.separableFilters is scielab_module.separable_filters
    assert scielab_module.scielabRGB is scielab_module.scielab_rgb


def test_param_format_string_and_key_value_list() -> None:
    assert param_format("Exposure Time") == "exposuretime"
    assert param_format(["Exposure Time", 1, "Some Flag", True]) == ["exposuretime", 1, "someflag", True]


def test_energy_quanta_round_trip() -> None:
    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array([0.2, 0.5, 0.8])
    quanta = energy_to_quanta(energy, wave)
    restored = quanta_to_energy(quanta, wave)
    assert np.allclose(restored, energy)


def test_energy_quanta_round_trip_supports_wave_first_matrices() -> None:
    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array(
        [
            [0.2, 0.4],
            [0.5, 0.7],
            [0.8, 1.0],
        ],
        dtype=float,
    )
    quanta = energy_to_quanta(energy, wave)
    restored = quanta_to_energy(quanta, wave)
    assert np.allclose(restored, energy)


def test_energy_quanta_helpers_are_exposed_through_package_root() -> None:
    wave = np.array([400.0, 500.0, 600.0])
    energy = np.array([0.2, 0.5, 0.8])
    quanta = energy_to_quanta(energy, wave)

    assert np.allclose(top_level_energy_to_quanta(energy, wave), quanta)
    assert np.allclose(Energy2Quanta(energy, wave), quanta)
    assert np.allclose(top_level_quanta_to_energy(quanta, wave), energy)
    assert np.allclose(Quanta2Energy(quanta, wave), energy)


def test_spectral_helpers_are_exposed_through_package_root() -> None:
    source_wave = np.array([700.0, 600.0, 500.0, 400.0], dtype=float)
    values = np.array(
        [
            [0.7, 0.2],
            [0.6, 0.3],
            [0.5, 0.4],
            [0.4, 0.5],
        ],
        dtype=float,
    )
    target_wave = np.array([400.0, 500.0, 600.0, 700.0], dtype=float)
    first = np.array([0.1, 0.3, 0.6], dtype=float)
    second = np.array([0.1, 0.4, 0.5], dtype=float)
    percentile_data = np.array([[1.0, 2.0], [3.0, 7.0]], dtype=float)

    np.testing.assert_allclose(
        top_level_interp_spectra(source_wave, values, target_wave),
        interp_spectra(source_wave, values, target_wave),
    )
    assert np.isclose(top_level_spectral_step(target_wave), spectral_step(target_wave))
    assert np.isclose(top_level_array_percentile(percentile_data, 75.0), array_percentile(percentile_data, 75.0))
    assert np.isclose(top_level_spectral_angle(first, second), spectral_angle(first, second))


def test_srgb_helpers_are_exposed_through_package_root() -> None:
    xyz = np.array([[[0.25, 0.40, 0.10], [0.50, 0.50, 0.50]]], dtype=float)
    linear_rgb = np.array([[[0.001, 0.25, 0.75], [0.05, 0.5, 1.0]]], dtype=float)
    gamma_table = np.column_stack(
        [
            np.linspace(0.0, 1.0, 8, dtype=float) ** 2.2,
            np.linspace(0.0, 1.0, 8, dtype=float) ** 2.0,
            np.linspace(0.0, 1.0, 8, dtype=float) ** 1.8,
        ]
    )

    np.testing.assert_allclose(top_level_xyz_to_linear_srgb(xyz), xyz_to_linear_srgb(xyz))
    np.testing.assert_allclose(top_level_linear_to_srgb(linear_rgb), linear_to_srgb(linear_rgb))
    np.testing.assert_allclose(
        top_level_invert_gamma_table(linear_rgb, gamma_table),
        invert_gamma_table(linear_rgb, gamma_table),
    )


def test_image_numeric_helpers_are_exposed_through_package_root() -> None:
    cube = np.arange(12.0, dtype=float).reshape(2, 3, 2)
    resized = resample_cube(cube, (4, 6))
    sigmas = np.array([0.0, 1.0], dtype=float)
    source = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    transform = np.array(
        [
            [2.0, 1.0],
            [0.5, 3.0],
        ],
        dtype=float,
    )
    target = source @ transform
    pattern = np.array([[1, 2], [3, 4]], dtype=int)

    np.testing.assert_allclose(top_level_resample_cube(cube, (4, 6)), resized)
    np.testing.assert_allclose(
        top_level_apply_channelwise_gaussian(cube, sigmas, mode="nearest"),
        apply_channelwise_gaussian(cube, sigmas, mode="nearest"),
    )
    assert np.isclose(
        top_level_gaussian_sigma_pixels(4.0, 550.0, 2.0e-6, extra_blur_pixels=0.25),
        gaussian_sigma_pixels(4.0, 550.0, 2.0e-6, extra_blur_pixels=0.25),
    )
    np.testing.assert_allclose(top_level_least_squares_matrix(source, target), least_squares_matrix(source, target))
    np.testing.assert_array_equal(top_level_tile_pattern(pattern, 3, 5), tile_pattern(pattern, 3, 5))


def test_color_and_scielab_helpers_are_exposed_through_package_root() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    xyz_energy = xyz_color_matching(wave, energy=True)
    xyz_quanta = xyz_color_matching(wave, quanta=True)
    xyz1 = np.array([[[0.25, 0.35, 0.15]]], dtype=float)
    xyz2 = np.array([[[0.30, 0.30, 0.20]]], dtype=float)
    white = np.array([0.95, 1.0, 1.09], dtype=float)

    np.testing.assert_allclose(top_level_xyz_color_matching(wave, energy=True), xyz_energy)
    np.testing.assert_allclose(top_level_xyz_color_matching(wave, quanta=True), xyz_quanta)
    np.testing.assert_allclose(
        top_level_sensor_to_target_matrix(wave, xyz_quanta, target_space="xyz"),
        sensor_to_target_matrix(wave, xyz_quanta, target_space="xyz"),
    )
    np.testing.assert_allclose(top_level_sensor_to_xyz_matrix(wave, xyz_quanta), sensor_to_xyz_matrix(wave, xyz_quanta))
    np.testing.assert_allclose(
        top_level_internal_to_display_matrix(wave, xyz_quanta),
        internal_to_display_matrix(wave, xyz_quanta),
    )
    root_x1, root_x2, root_x3 = top_level_sc_gaussian_parameters(30.0, {"filterversion": "distribution"})
    mod_x1, mod_x2, mod_x3 = sc_gaussian_parameters(30.0, {"filterversion": "distribution"})
    np.testing.assert_allclose(root_x1, mod_x1)
    np.testing.assert_allclose(root_x2, mod_x2)
    np.testing.assert_allclose(root_x3, mod_x3)
    np.testing.assert_allclose(
        top_level_sc_compute_difference(xyz1, xyz2, white, "2000"),
        sc_compute_difference(xyz1, xyz2, white, "2000"),
    )
    np.testing.assert_allclose(scGaussianParameters(30.0, {"filterversion": "distribution"})[0], mod_x1)
    np.testing.assert_allclose(scGaussianParameters(30.0, {"filterversion": "distribution"})[1], mod_x2)
    np.testing.assert_allclose(scGaussianParameters(30.0, {"filterversion": "distribution"})[2], mod_x3)
    np.testing.assert_allclose(scComputeDifference(xyz1, xyz2, white, "2000"), sc_compute_difference(xyz1, xyz2, white, "2000"))


def test_example_spd_pair_is_exposed_through_package_root() -> None:
    root_wave, root_first, root_second = top_level_example_spd_pair()
    mod_wave, mod_first, mod_second = example_spd_pair()

    np.testing.assert_allclose(root_wave, mod_wave)
    np.testing.assert_allclose(root_first, mod_first)
    np.testing.assert_allclose(root_second, mod_second)


def test_interp_spectra_supports_descending_source_waves() -> None:
    source_wave = np.array([700.0, 600.0, 500.0, 400.0], dtype=float)
    values = np.array(
        [
            [0.7, 0.2],
            [0.6, 0.3],
            [0.5, 0.4],
            [0.4, 0.5],
        ],
        dtype=float,
    )
    target_wave = np.array([400.0, 500.0, 600.0, 700.0], dtype=float)
    interpolated = interp_spectra(source_wave, values, target_wave)
    expected = values[::-1, :]
    assert np.allclose(interpolated, expected)


def test_blackbody_matlab_scaling() -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    spectra = blackbody(wave, np.array([3000.0, 5000.0]))
    assert spectra.shape == (wave.size, 2)
    assert np.all(spectra > 0.0)
    eq_index = int(np.argmin(np.abs(wave - 550.0)))
    assert np.isclose(spectra[eq_index, 0], spectra[eq_index, 1])


def test_half_tone_image_scales_integer_cells_and_alias_matches() -> None:
    cell = np.array([[1, 2], [3, 4]], dtype=float)
    image = np.array(
        [
            [0.10, 0.50, 0.90],
            [0.20, 0.40, 0.60],
            [0.80, 1.00, 0.10],
        ],
        dtype=float,
    )
    scaled_cell = np.array([[0.125, 0.375], [0.625, 0.875]], dtype=float)
    tiled = np.tile(scaled_cell, (2, 2))[: image.shape[0], : image.shape[1]]
    expected = tiled < image

    result = half_tone_image(cell, image)
    alias = HalfToneImage(cell, image)

    assert result.dtype == np.bool_
    assert np.array_equal(result, expected)
    assert np.array_equal(alias, expected)


def test_floyd_steinberg_matches_legacy_error_diffusion_and_alias() -> None:
    fs = np.array([[0.0, 0.0, 7.0], [3.0, 5.0, 1.0]], dtype=float) / 16.0
    image = np.array(
        [
            [0.20, 0.40, 0.60, 0.80],
            [0.30, 0.70, 0.50, 0.10],
            [0.90, 0.20, 0.80, 0.40],
        ],
        dtype=float,
    )

    def _reference(fs_kernel: np.ndarray, image_data: np.ndarray) -> np.ndarray:
        rows, cols = image_data.shape
        fs_rows, fs_cols = fs_kernel.shape
        radius = fs_cols // 2
        temp = np.zeros((rows + fs_rows, cols + 2 * radius), dtype=float)
        temp[:rows, radius : radius + cols] = image_data

        def _round_scalar(value: float) -> float:
            return float(np.sign(value) * np.floor(abs(value) + 0.5))

        for row in range(rows):
            for col in range(radius, cols):
                error = float(temp[row, col])
                temp[row, col] = _round_scalar(error)
                error -= float(temp[row, col])
                for dr in range(fs_rows):
                    for dc in range(fs_cols):
                        temp[row + dr, col - radius + dc] += error * float(fs_kernel[dr, dc])

            temp[row : row + fs_rows, cols : cols + radius] += temp[row + 1 : row + fs_rows + 1, :radius]

            for col in range(cols, cols + radius):
                error = float(temp[row, col])
                temp[row, col] = _round_scalar(error)
                error -= float(temp[row, col])
                for dr in range(fs_rows):
                    for dc in range(fs_cols):
                        temp[row + dr, col - radius + dc] += error * float(fs_kernel[dr, dc])

            temp[row + 1 : row + fs_rows + 1, radius : 2 * radius] += temp[
                row : row + fs_rows, cols + radius : cols + 2 * radius
            ]
            temp[:, :radius] = 0.0
            temp[:, cols + radius : cols + 2 * radius] = 0.0

        return temp[:rows, radius : radius + cols]

    expected = _reference(fs, image)
    result = floyd_steinberg(fs, image)
    alias = FloydSteinberg(fs, image)

    assert np.array_equal(result, expected)
    assert np.array_equal(alias, expected)
    assert set(np.unique(result)).issubset({0.0, 1.0})


def test_unit_frequency_list_matches_matlab_even_and_odd() -> None:
    assert np.allclose(unit_frequency_list(1), np.array([0.0]))
    assert np.allclose(unit_frequency_list(4), np.array([-1.0, -0.5, 0.0, 0.5]))
    assert np.allclose(unit_frequency_list(5), np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_unit_frequency_list_is_exposed_through_package_root() -> None:
    assert np.allclose(top_level_unit_frequency_list(4), unit_frequency_list(4))
    assert np.allclose(unitFrequencyList(5), unit_frequency_list(5))


def test_ie_unit_scale_factor_and_dpi2mperdot_match_matlab_aliases() -> None:
    assert ie_unit_scale_factor("um") == pytest.approx(1.0e6)
    assert ieUnitScaleFactor("deg") == pytest.approx(180.0 / np.pi)
    assert dpi2mperdot_fn(100.0) == pytest.approx(254.0)
    assert dpi2mperdot(100.0) == pytest.approx(254.0)
    assert ie_dpi2_mperdot(100.0, "mm") == pytest.approx(0.254)
    assert ieDpi2Mperdot(100.0, "mm") == pytest.approx(0.254)


def test_ie_space_to_amp_matches_matlab_frequency_truncation_alias() -> None:
    pos = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    data = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    freq, f_data = ie_space_to_amp(pos, data)
    alias_freq, alias_f_data = ieSpace2Amp(pos, data, True)

    assert np.allclose(freq, np.array([0.0, 1.0 / 3.0], dtype=float))
    assert np.allclose(f_data, np.array([4.0, 0.0], dtype=float))
    assert np.allclose(alias_freq, freq)
    assert np.allclose(alias_f_data, np.array([1.0, 0.0], dtype=float))


def test_sample2space_and_space2sample_match_centered_zero_based_aliases() -> None:
    row_support, col_support = sample2space_fn(np.arange(1, 5), np.arange(1, 5), 5.0, 10.0)
    alias_row_support, alias_col_support = sample2space(np.arange(1, 5), np.arange(1, 5), 5.0, 10.0)

    assert np.allclose(row_support, np.array([-7.5, -2.5, 2.5, 7.5], dtype=float))
    assert np.allclose(col_support, np.array([-15.0, -5.0, 5.0, 15.0], dtype=float))
    assert np.allclose(alias_row_support, row_support)
    assert np.allclose(alias_col_support, col_support)

    row_samples, col_samples = space2sample_fn(row_support, col_support, 5.0, 10.0)
    alias_row_samples, alias_col_samples = space2sample(row_support, col_support, 5.0, 10.0)

    assert np.allclose(row_samples, np.array([0.0, 1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(col_samples, np.array([0.0, 1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(alias_row_samples, row_samples)
    assert np.allclose(alias_col_samples, col_samples)


def test_ie_light_and_reflectance_lists_and_dispatch_match_aliases() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)

    light_names, light_data, light_samples = ie_light_list(wave=wave)
    alias_light_names, alias_light_data, alias_light_samples = ieLightList("wave", wave)

    assert "D65.mat" in light_names
    d65_index = light_names.index("D65.mat")
    assert light_data[d65_index].shape[0] == wave.size
    assert light_samples[d65_index] == light_data[d65_index].shape[1]
    assert np.min(light_data[d65_index]) > 0.0
    assert alias_light_names == light_names
    assert np.array_equal(alias_light_samples, light_samples)
    assert np.allclose(alias_light_data[d65_index], light_data[d65_index])

    reflectance_names, reflectance_data, reflectance_samples = ie_reflectance_list(wave=wave)
    alias_reflectance_names, alias_reflectance_data, alias_reflectance_samples = ieReflectanceList("wave", wave)

    assert "reflectanceBasis.mat" not in reflectance_names
    assert "macbethChart.mat" in reflectance_names
    macbeth_index = reflectance_names.index("macbethChart.mat")
    assert reflectance_data[macbeth_index].shape[0] == wave.size
    assert reflectance_samples[macbeth_index] == reflectance_data[macbeth_index].shape[1]
    assert float(np.max(reflectance_data[macbeth_index])) <= 1.0
    assert alias_reflectance_names == reflectance_names
    assert np.array_equal(alias_reflectance_samples, reflectance_samples)
    assert np.allclose(alias_reflectance_data[macbeth_index], reflectance_data[macbeth_index])

    data_names, data_values, data_samples = ie_data_list("refl", "wave", wave)
    alias_names, alias_values, alias_samples = ieDataList("light", "wave", wave)

    assert data_names == reflectance_names
    assert np.array_equal(data_samples, reflectance_samples)
    assert np.allclose(data_values[macbeth_index], reflectance_data[macbeth_index])
    assert alias_names == light_names
    assert np.array_equal(alias_samples, light_samples)
    assert np.allclose(alias_values[d65_index], light_data[d65_index])

    with pytest.raises(ValueError, match="Unsupported ieDataList type"):
        ie_data_list("sensorqe")


def test_numerical_helper_wrappers_match_matlab_scaling_and_clipping_aliases() -> None:
    clipped_default = ieClip(np.array([-1.0, 0.5, 2.0], dtype=float))
    clipped_symmetric = ie_clip(np.array([-2.0, -0.5, 0.5, 2.0], dtype=float), 1.5)
    clipped_upper_only = ie_clip(np.array([-2.0, 0.5, 2.0], dtype=float), None, 1.0)

    assert np.allclose(clipped_default, np.array([0.0, 0.5, 1.0], dtype=float))
    assert np.allclose(clipped_symmetric, np.array([-1.5, -0.5, 0.5, 1.5], dtype=float))
    assert np.allclose(clipped_upper_only, np.array([-2.0, 0.5, 1.0], dtype=float))

    data = np.array([-10.0, 20.0, 50.0], dtype=float)
    scaled, mn, mx = ie_scale(data, 20.0, 90.0)
    peak_scaled, _, _ = ieScale(data, 1.0)
    column_scaled = ie_scale_columns(np.column_stack((data, 2.0 * data)), 0.0, 1.0)
    alias_columns = ieScaleColumns(np.column_stack((data, 2.0 * data)), 0.0, 1.0)

    assert mn == pytest.approx(-10.0)
    assert mx == pytest.approx(50.0)
    assert np.allclose(scaled, np.array([20.0, 55.0, 90.0], dtype=float))
    assert np.allclose(peak_scaled, np.array([-0.2, 0.4, 1.0], dtype=float))
    assert np.allclose(column_scaled[:, 0], np.array([0.0, 0.5, 1.0], dtype=float))
    assert np.allclose(column_scaled[:, 1], np.array([0.0, 0.5, 1.0], dtype=float))
    assert np.allclose(alias_columns, column_scaled)


def test_numerical_helper_wrappers_match_matlab_geometry_and_norm_aliases() -> None:
    matrix = np.arange(1.0, 82.0, dtype=float).reshape((9, 9), order="F")
    middle, center = get_middle_matrix(matrix, 3)
    alias_middle, alias_center = getMiddleMatrix(matrix, 3)

    assert np.array_equal(center[:2], np.array([5, 5], dtype=int))
    assert np.array_equal(alias_center[:2], center[:2])
    assert np.array_equal(middle, matrix[2:7, 2:7])
    assert np.array_equal(alias_middle, middle)

    padded = np.arange(1.0, 26.0, dtype=float).reshape(5, 5)
    expected_unpadded = np.array([[7.0, 8.0, 9.0], [12.0, 13.0, 14.0], [17.0, 18.0, 19.0]], dtype=float)
    assert np.array_equal(unpadarray_fn(padded, [1, 1]), expected_unpadded)
    assert np.array_equal(unpadarray(padded, [1, 1]), expected_unpadded)

    assert vector_length(np.array([1.0, 1.0], dtype=float)) == pytest.approx(np.sqrt(2.0))
    assert np.array_equal(vectorLength(np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=float), 1), np.array([1.0, 0.0]))


def test_numerical_helper_wrappers_match_matlab_rotation_quadrant_and_oddness_aliases() -> None:
    rot = rotation_matrix_3d([0.0, 0.0, np.pi / 2.0])
    alias_rot = rotationMatrix3d([0.0, 0.0, np.pi / 2.0], 2.0)
    expected_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    assert np.allclose(rot, expected_rot, atol=1e-12)
    assert np.allclose(alias_rot, 2.0 * expected_rot, atol=1e-12)

    upper_right = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=float)
    expected_full = np.array(
        [
            [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            [6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
            [9.0, 8.0, 7.0, 7.0, 8.0, 9.0],
            [6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
            [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    assert np.array_equal(upper_quad_to_full_matrix(upper_right, 5, 6), expected_full)
    assert np.array_equal(upperQuad2FullMatrix(upper_right, 5, 6), expected_full)
    assert isodd(3) is True
    assert np.array_equal(isodd(np.array([1, 2, 3], dtype=int)), np.array([True, False, True]))


def test_ie_hwhm_to_sd_matches_matlab_formulas_alias() -> None:
    assert ie_hwhm_to_sd(10.0, 1) == pytest.approx(10.0 / (2.0 * np.sqrt(np.log(2.0))))
    assert ieHwhm2SD(10.0, 2) == pytest.approx(10.0 / np.sqrt(2.0 * np.log(2.0)))


def test_ffndgrid_matches_average_and_axis_orientation() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    f = np.array([1.0, 2.0, 3.0, 4.0, 6.0], dtype=float)

    grid, axes = ffndgrid_fn(x, f, [-2, -2], None, 1)
    alias_grid, alias_axes = ffndgrid(x, f, [-2, -2], None, 1)

    expected = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=float)
    assert np.array_equal(grid, expected)
    assert np.array_equal(alias_grid, expected)
    assert np.array_equal(axes[0], np.array([0.0, 1.0], dtype=float))
    assert np.array_equal(axes[1], np.array([0.0, 1.0], dtype=float))
    assert np.array_equal(alias_axes[0], axes[0])
    assert np.array_equal(alias_axes[1], axes[1])


def test_ie_compress_data_matches_uint_quantization_alias() -> None:
    data = np.array([[[0.0], [0.5]], [[1.0], [0.25]]], dtype=float)

    compressed16, mn16, mx16 = ie_compress_data(data, 16)
    alias16, alias_mn16, alias_mx16 = ieCompressData(data, 16)
    compressed32, _, _ = ie_compress_data(data, 32)

    expected16 = np.array([[[0], [32768]], [[65535], [16384]]], dtype=np.uint16)
    expected32 = np.array([[[0], [2147483648]], [[4294967295], [1073741824]]], dtype=np.uint32)
    assert mn16 == pytest.approx(0.0)
    assert mx16 == pytest.approx(1.0)
    assert alias_mn16 == pytest.approx(mn16)
    assert alias_mx16 == pytest.approx(mx16)
    assert np.array_equal(compressed16, expected16)
    assert np.array_equal(alias16, expected16)
    assert np.array_equal(compressed32, expected32)


def test_ie_uncompress_data_inverts_quantization_alias() -> None:
    data = np.array([[[0.0], [0.5]], [[1.0], [0.25]]], dtype=float)
    compressed16, mn16, mx16 = ie_compress_data(data, 16)

    restored = ie_uncompress_data(compressed16, mn16, mx16, 16)
    alias_restored = ieUncompressData(compressed16, mn16, mx16, 16)

    assert np.allclose(restored, data, atol=1.0 / 65535.0)
    assert np.allclose(alias_restored, restored, atol=0.0)


def test_ie_line_align_recovers_shift_scale_alias() -> None:
    def func(values: np.ndarray) -> np.ndarray:
        return 0.3 * values**2 + 1.2 * values - 0.5

    d1 = {"x": np.linspace(-1.0, 1.0, 81, dtype=float)}
    d1["y"] = func(d1["x"])
    true_scale = 0.8
    true_shift = 0.1
    d2 = {"x": np.linspace(-2.0, 2.0, 161, dtype=float)}
    d2["y"] = func(true_scale * (d2["x"] - true_shift))

    estimated, est_y = ie_line_align(d1, d2)
    alias_estimated, alias_est_y = ieLineAlign(d1, d2)

    valid = ~np.isnan(est_y)
    assert estimated[0] == pytest.approx(true_scale, rel=1e-2, abs=1e-2)
    assert estimated[1] == pytest.approx(true_shift, rel=1e-2, abs=1e-2)
    assert np.allclose(est_y[valid], d1["y"][valid], atol=1e-4)
    assert np.allclose(alias_estimated, estimated, atol=1e-6)
    assert np.allclose(alias_est_y[valid], est_y[valid], atol=1e-6)


def test_ie_tikhonov_matches_closed_form_and_lstsq() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    vector = np.array([1.0, 2.0, 2.5, 3.0], dtype=float)
    x, x_ols = ie_tikhonov(matrix, vector, "minnorm", 0.5, "smoothness", 0.25)

    d2 = np.diff(np.eye(matrix.shape[1], dtype=float), 2, axis=0)
    expected = np.linalg.solve(
        (matrix.T @ matrix) + (0.5 * np.eye(matrix.shape[1], dtype=float)) + (0.25 * (d2.T @ d2)),
        matrix.T @ vector,
    )
    expected_ols = np.linalg.lstsq(matrix, vector, rcond=None)[0]

    assert np.allclose(x, expected)
    assert np.allclose(x_ols, expected_ols)


def test_ie_find_files_recurses_and_normalizes_extension_alias(tmp_path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    first = tmp_path / "root.mat"
    second = nested / "child.mat"
    third = nested / "ignore.txt"
    first.write_text("root", encoding="utf-8")
    second.write_text("child", encoding="utf-8")
    third.write_text("ignore", encoding="utf-8")

    found = ie_find_files(tmp_path, "mat")
    alias_found = ieFindFiles(tmp_path, ".mat")

    expected = [str(second), str(first)]
    assert found == expected
    assert alias_found == expected


def test_ie_tone_synthesizes_waveform_and_alias() -> None:
    params, waveform = ie_tone("Frequency", 512, "Amplitude", 0.5, "Duration", 0.1)
    alias_params, alias_waveform = ieTone({"Frequency": 512, "Amplitude": 0.5, "Duration": 0.1})

    assert params == {"Amplitude": 0.5, "Duration": 0.1, "Frequency": 512.0}
    assert alias_params == params
    assert waveform.shape == (820,)
    assert waveform[0] == pytest.approx(0.0)
    assert float(np.max(np.abs(waveform))) <= 0.5 + 1e-12
    assert np.allclose(alias_waveform, waveform)


def test_qinterp2_matches_nearest_triangle_and_bilinear_alias() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 1.0, 2.0], dtype=float)
    X, Y = np.meshgrid(x, y)
    Z = X + (10.0 * Y)

    query_x = np.array([0.5, 1.5, 3.0], dtype=float)
    query_y = np.array([0.5, 1.5, 0.5], dtype=float)

    nearest = qinterp2_fn(X, Y, Z, query_x, query_y, 0)
    triangle = qinterp2_fn(X, Y, Z, query_x, query_y, 1)
    bilinear = qinterp2_fn(X, Y, Z, query_x, query_y, 2)

    assert np.allclose(nearest[:2], np.array([11.0, 22.0], dtype=float))
    assert np.isnan(nearest[2])
    assert np.allclose(triangle[:2], np.array([5.5, 16.5], dtype=float))
    assert np.isnan(triangle[2])
    assert np.allclose(bilinear[:2], np.array([5.5, 16.5], dtype=float))
    assert np.isnan(bilinear[2])
    assert np.allclose(qinterp2(X, Y, Z, query_x, query_y, 2), bilinear, equal_nan=True)


def test_ie_fit_line_matches_matlab_one_line_and_multiple_lines() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    y = 2.0 * x + 0.5
    slope, offset = ie_fit_line(x, y)
    assert slope == pytest.approx(2.0)
    assert offset == pytest.approx(0.5)

    x_multi = x.reshape(-1, 1)
    y_multi = np.column_stack((2.0 * x + 0.5, -1.0 * x + 3.0))
    slopes, offsets = ie_fit_line(x_multi, y_multi, "multipleLines")
    assert np.allclose(slopes, np.array([2.0, -1.0], dtype=float))
    assert np.allclose(offsets, np.array([0.5, 3.0], dtype=float))


def test_ie_find_wave_index_exact_and_nearest_match_alias() -> None:
    wave = np.array([400.0, 500.0, 600.0, 700.0], dtype=float)

    exact = ie_find_wave_index(wave, [500.0, 700.0])
    nearest = ieFindWaveIndex(wave, [520.0, 610.0], perfect=False)

    assert np.array_equal(exact, np.array([False, True, False, True]))
    assert np.array_equal(nearest, np.array([False, True, True, False]))


def test_ie_wave2_index_matches_matlab_one_based_and_bounding_pair_alias() -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)

    assert ie_wave2_index(wave, 503.0) == 2
    assert ieWave2Index(wave, 487.0, bounding=True) == (1, 2)
    assert ieWave2Index(wave, 600.0, bounding=True) == (3, 3)


def test_ie_radial_matrix_matches_centered_distance_grid_alias() -> None:
    expected = np.array(
        [
            [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
            [1.0, 0.0, 1.0],
            [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
        ],
        dtype=float,
    )

    result = ie_radial_matrix(3, 3, 2, 2)
    alias = ieRadialMatrix(3, 3, 2, 2)

    assert np.allclose(result, expected)
    assert np.allclose(alias, expected)


def test_image_bounding_box_and_centroid_match_matlab_pixel_geometry_aliases() -> None:
    support = np.zeros((5, 5), dtype=float)
    support[1:4, 1:4] = 1.0
    weighted = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    box = image_bounding_box(support)
    alias_box = imageBoundingBox(support)
    centroid = image_centroid(weighted)
    alias_centroid = imageCentroid(weighted)

    assert np.allclose(box, np.array([2.0, 2.0, 2.0, 2.0], dtype=float))
    assert np.allclose(alias_box, box)
    assert centroid == (3, 2)
    assert alias_centroid == centroid


def test_image_circular_and_contrast_match_legacy_contract_aliases() -> None:
    image = np.ones((3, 3), dtype=float)
    contrast_input = np.array(
        [
            [[1.0, 2.0], [3.0, 6.0]],
            [[5.0, 10.0], [7.0, 14.0]],
        ],
        dtype=float,
    )
    expected_circular = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    expected_contrast = np.array(
        [
            [[-0.75, -0.75], [-0.25, -0.25]],
            [[0.25, 0.25], [0.75, 0.75]],
        ],
        dtype=float,
    )

    circular = image_circular(image)
    alias_circular = imageCircular(image)
    contrast = image_contrast(contrast_input)
    alias_contrast = imageContrast(contrast_input)

    assert np.array_equal(circular, expected_circular)
    assert np.array_equal(alias_circular, expected_circular)
    assert np.allclose(contrast, expected_contrast)
    assert np.allclose(alias_contrast, expected_contrast)


def test_ie_cmap_matches_rg_bw_and_alias() -> None:
    rg = ie_cmap("rg", 4)
    bw = ieCmap("bw", 3, 2.0)

    assert np.allclose(
        rg,
        np.array(
            [
                [0.0, 1.0, 0.5],
                [1.0 / 3.0, 2.0 / 3.0, 0.5],
                [2.0 / 3.0, 1.0 / 3.0, 0.5],
                [1.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
    )
    assert np.allclose(
        bw,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_ie_crop_rect_matches_matlab_fov_geometry_and_alias() -> None:
    oi = {"optics": {"raytrace": {"objectDistance": 2.0}}}
    scenesize = np.array([100, 200], dtype=float)

    crop_rect = ie_crop_rect(oi, scenesize, 20.0, 10.0)
    alias = ieCropRect(oi, scenesize, 20.0, 10.0)

    assert np.array_equal(crop_rect, np.array([52, 26, 98, 49]))
    assert np.array_equal(alias, crop_rect)


def test_ie_lut_digital_invert_linear_match_legacy_tables_and_aliases() -> None:
    dac = np.array([[0, 1], [2, 3]], dtype=float)
    gamma = np.array([0.0, 0.25, 0.5, 1.0], dtype=float)
    rgb = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=float)
    inverse_gamma = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)

    digital = ie_lut_digital(dac, gamma)
    digital_alias = ieLUTDigital(dac, gamma)
    invert = ie_lut_invert(gamma, 4)
    invert_alias = ieLUTInvert(gamma, 4)
    linear = ie_lut_linear(rgb, inverse_gamma)
    linear_alias = ieLUTLinear(rgb, inverse_gamma)

    assert np.allclose(digital, np.array([[0.0, 0.25], [0.5, 1.0]], dtype=float))
    assert np.allclose(digital_alias, digital)
    assert np.allclose(invert, np.array([[1.0], [2.0], [3.0], [3.5]], dtype=float))
    assert np.allclose(invert_alias, invert)
    assert np.allclose(linear, np.array([[0.0, 10.0], [20.0, 30.0]], dtype=float))
    assert np.allclose(linear_alias, linear)


def test_rgb2dac_matches_single_and_three_channel_tables_and_alias() -> None:
    rgb = np.array(
        [
            [[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]],
        ],
        dtype=float,
    )
    inv_gamma = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ],
        dtype=float,
    )

    result = rgb_to_dac(rgb, inv_gamma)
    alias = rgb2dac(rgb, inv_gamma)

    expected = np.array(
        [
            [[0.0, 50.0, 90.0], [10.0, 50.0, 60.0]],
        ],
        dtype=float,
    )
    assert np.array_equal(result, expected)
    assert np.array_equal(alias, expected)


def test_image_transpose_translate_and_interpolate_match_headless_numeric_contracts() -> None:
    cube = np.arange(1.0, 13.0, dtype=float).reshape(2, 3, 2)
    image = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    transposed = image_transpose(cube)
    transposed_alias = imageTranspose(cube)
    translated = image_translate(image, [1, 1])
    translated_alias = imageTranslate(image, [1, 1])
    interpolated = image_interpolate(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), 4, 4)
    interpolated_alias = imageInterpolate(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), 4, 4)

    assert np.array_equal(transposed[:, :, 0], np.array([[1.0, 7.0], [3.0, 9.0], [5.0, 11.0]], dtype=float))
    assert np.array_equal(transposed_alias, transposed)
    assert np.array_equal(
        translated,
        np.array(
            [
                [5.0, 6.0, 0.0],
                [8.0, 9.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(translated_alias, translated)
    assert interpolated.shape == (4, 4)
    assert interpolated[0, 0] == pytest.approx(1.0)
    assert interpolated[-1, -1] == pytest.approx(4.0)
    assert np.allclose(interpolated_alias, interpolated)


def test_image_hparams_matches_legacy_defaults_and_alias() -> None:
    params = image_hparams()
    alias = imageHparams()

    expected = {
        "freq": 2,
        "contrast": 1,
        "ang": 0,
        "ph": 1.5708,
        "row": 128,
        "col": 128,
        "GaborFlag": 0,
    }
    assert params == expected
    assert alias == expected


def test_image_gabor_matches_legacy_range_shape_and_alias() -> None:
    gabor = image_gabor(frequency=4, phase=0.0, spread=0.25, orientation=np.pi / 4.0, imagesize=6, contrast=1.0)
    alias = imageGabor(frequency=4, phase=0.0, spread=0.25, orientation=np.pi / 4.0, imagesize=6, contrast=1.0)

    assert gabor.shape == (7, 7)
    assert float(np.min(gabor)) >= 0.0
    assert float(np.max(gabor)) <= 1.0
    assert gabor[3, 3] == pytest.approx(1.0)
    assert np.allclose(alias, gabor)


def test_image_make_montage_and_image_montage_match_legacy_tile_layout_aliases() -> None:
    cube = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        ),
        axis=2,
    )

    montage, coords = image_make_montage(cube, [1, 3], n_cols=2, back_val=-1.0)
    fig_h, montage_alias, cb_h = imageMontage(cube, [1, 3], 2)
    headless = image_montage(cube, [1, 3], 2)

    expected = np.array(
        [
            [1.0, 2.0, 9.0, 10.0],
            [3.0, 4.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    assert np.array_equal(montage, expected)
    assert np.array_equal(coords, np.array([[1, 1], [3, 1]], dtype=int))
    assert fig_h is None and cb_h is None
    assert np.array_equal(montage_alias, expected)
    assert headless[0] is None and headless[2] is None
    assert np.array_equal(headless[1], expected)


def test_ie_parameter_otype_handles_direct_prefix_and_unique_params() -> None:
    assert ie_parameter_otype("scene") == ("scene", None)
    assert ie_parameter_otype("oi size") == ("oi", "size")
    assert ie_parameter_otype("pixel/size") == ("pixel", "size")
    assert ie_parameter_otype("pixel width and height") == ("pixel", "widthandheight")
    assert ie_parameter_otype("pixel pd width and height") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("width and height") == ("pixel", "widthandheight")
    assert ie_parameter_otype("widthandheight") == ("pixel", "widthandheight")
    assert ie_parameter_otype("widthheight") == ("pixel", "widthheight")
    assert ie_parameter_otype("pd width and height") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("pdwidthandheight") == ("pixel", "pdwidthandheight")
    assert ie_parameter_otype("size same fill factor") == ("pixel", "sizesamefillfactor")
    assert ie_parameter_otype("size constant fill factor") == ("pixel", "sizeconstantfillfactor")
    assert ie_parameter_otype("size keep fill factor") == ("pixel", "sizekeepfillfactor")
    assert ie_parameter_otype("sizesamefillfactor") == ("pixel", "sizesamefillfactor")
    assert ie_parameter_otype("sizeconstantfillfactor") == ("pixel", "sizeconstantfillfactor")
    assert ie_parameter_otype("sizekeepfillfactor") == ("pixel", "sizekeepfillfactor")
    assert ie_parameter_otype("dark voltage per pixel per sec") == ("pixel", "darkvoltageperpixelpersec")
    assert ie_parameter_otype("wvf_zcoeffs") == ("wvf", "zcoeffs")
    assert ie_parameter_otype("display gamma") == ("display", "gamma")
    assert ie_parameter_otype("match oi") == ("sensor", "matchoi")
    assert ie_parameter_otype("dsnu sigma") == ("sensor", "dsnusigma")
    assert ie_parameter_otype("dsnusigma") == ("sensor", "dsnusigma")
    assert ie_parameter_otype("dsnulevel") == ("sensor", "dsnulevel")
    assert ie_parameter_otype("sigma offset fpn") == ("sensor", "sigmaoffsetfpn")
    assert ie_parameter_otype("sigmaoffsetfpn") == ("sensor", "sigmaoffsetfpn")
    assert ie_parameter_otype("offset fpn") == ("sensor", "offsetfpn")
    assert ie_parameter_otype("offsetfpn") == ("sensor", "offsetfpn")
    assert ie_parameter_otype("offset") == ("sensor", "offset")
    assert ie_parameter_otype("offsetsd") == ("sensor", "offsetsd")
    assert ie_parameter_otype("offset sd") == ("sensor", "offsetsd")
    assert ie_parameter_otype("offsetnoisevalue") == ("sensor", "offsetnoisevalue")
    assert ie_parameter_otype("offset noise value") == ("sensor", "offsetnoisevalue")
    assert ie_parameter_otype("sigmadsnu") == ("sensor", "sigmadsnu")
    assert ie_parameter_otype("sigma dsnu") == ("sensor", "sigmadsnu")
    assert ie_parameter_otype("sigmagainfpn") == ("sensor", "sigmagainfpn")
    assert ie_parameter_otype("sigma gain fpn") == ("sensor", "sigmagainfpn")
    assert ie_parameter_otype("gainfpn") == ("sensor", "gainfpn")
    assert ie_parameter_otype("gain fpn") == ("sensor", "gainfpn")
    assert ie_parameter_otype("gain") == ("sensor", "gain")
    assert ie_parameter_otype("gainsd") == ("sensor", "gainsd")
    assert ie_parameter_otype("gain sd") == ("sensor", "gainsd")
    assert ie_parameter_otype("gainnoisevalue") == ("sensor", "gainnoisevalue")
    assert ie_parameter_otype("gain noise value") == ("sensor", "gainnoisevalue")
    assert ie_parameter_otype("prnusigma") == ("sensor", "prnusigma")
    assert ie_parameter_otype("prnulevel") == ("sensor", "prnulevel")
    assert ie_parameter_otype("sigmaprnu") == ("sensor", "sigmaprnu")
    assert ie_parameter_otype("sigma prnu") == ("sensor", "sigmaprnu")
    assert ie_parameter_otype("fpn") == ("sensor", "fpn")
    assert ie_parameter_otype("fpnparameters") == ("sensor", "fpnparameters")
    assert ie_parameter_otype("fpnoffsetgain") == ("sensor", "fpnoffsetgain")
    assert ie_parameter_otype("fpnoffsetandgain") == ("sensor", "fpnoffsetandgain")
    assert ie_parameter_otype("fpn offset gain") == ("sensor", "fpnoffsetgain")
    assert ie_parameter_otype("dsnuimage") == ("sensor", "dsnuimage")
    assert ie_parameter_otype("offset fpn image") == ("sensor", "offsetfpnimage")
    assert ie_parameter_otype("offsetfpnimage") == ("sensor", "offsetfpnimage")
    assert ie_parameter_otype("prnuimage") == ("sensor", "prnuimage")
    assert ie_parameter_otype("gain fpn image") == ("sensor", "gainfpnimage")
    assert ie_parameter_otype("gainfpnimage") == ("sensor", "gainfpnimage")
    assert ie_parameter_otype("column fpn parameters") == ("sensor", "columnfpnparameters")
    assert ie_parameter_otype("columnfpnparameters") == ("sensor", "columnfpnparameters")
    assert ie_parameter_otype("columnfpn") == ("sensor", "columnfpn")
    assert ie_parameter_otype("column fixed pattern noise") == ("sensor", "columnfixedpatternnoise")
    assert ie_parameter_otype("columnfixedpatternnoise") == ("sensor", "columnfixedpatternnoise")
    assert ie_parameter_otype("col fpn") == ("sensor", "colfpn")
    assert ie_parameter_otype("colfpn") == ("sensor", "colfpn")
    assert ie_parameter_otype("column fpn offset") == ("sensor", "columnfpnoffset")
    assert ie_parameter_otype("columnfpnoffset") == ("sensor", "columnfpnoffset")
    assert ie_parameter_otype("col fpn offset") == ("sensor", "colfpnoffset")
    assert ie_parameter_otype("colfpnoffset") == ("sensor", "colfpnoffset")
    assert ie_parameter_otype("columndsnu") == ("sensor", "columndsnu")
    assert ie_parameter_otype("col dsnu") == ("sensor", "coldsnu")
    assert ie_parameter_otype("coldsnu") == ("sensor", "coldsnu")
    assert ie_parameter_otype("column fpn gain") == ("sensor", "columnfpngain")
    assert ie_parameter_otype("columnfpngain") == ("sensor", "columnfpngain")
    assert ie_parameter_otype("col fpn gain") == ("sensor", "colfpngain")
    assert ie_parameter_otype("colfpngain") == ("sensor", "colfpngain")
    assert ie_parameter_otype("columnprnu") == ("sensor", "columnprnu")
    assert ie_parameter_otype("col prnu") == ("sensor", "colprnu")
    assert ie_parameter_otype("colprnu") == ("sensor", "colprnu")
    assert ie_parameter_otype("coloffsetfpnvector") == ("sensor", "coloffsetfpnvector")
    assert ie_parameter_otype("col offset fpn") == ("sensor", "coloffsetfpn")
    assert ie_parameter_otype("col offset") == ("sensor", "coloffset")
    assert ie_parameter_otype("colgainfpnvector") == ("sensor", "colgainfpnvector")
    assert ie_parameter_otype("col gain fpn") == ("sensor", "colgainfpn")
    assert ie_parameter_otype("col gain") == ("sensor", "colgain")
    assert ie_parameter_otype("responsedr") == ("sensor", "responsedr")
    assert ie_parameter_otype("drdb20") == ("sensor", "drdb20")
    assert ie_parameter_otype("analoggain") == ("sensor", "analoggain")
    assert ie_parameter_otype("analogoffset") == ("sensor", "analogoffset")
    assert ie_parameter_otype("dynamicrange") == ("sensor", "dynamicrange")
    assert ie_parameter_otype("diffusionmtf") == ("sensor", "diffusionmtf")
    assert ie_parameter_otype("sensordynamicrange") == ("sensor", "sensordynamicrange")
    assert ie_parameter_otype("maxdigitalvalue") == ("sensor", "maxdigitalvalue")
    assert ie_parameter_otype("noiseflag") == ("sensor", "noiseflag")
    assert ie_parameter_otype("integrationtime") == ("sensor", "integrationtime")
    assert ie_parameter_otype("integrationtimes") == ("sensor", "integrationtimes")
    assert ie_parameter_otype("nsamplesperpixel") == ("sensor", "nsamplesperpixel")
    assert ie_parameter_otype("sensorconsistency") == ("sensor", "sensorconsistency")
    assert ie_parameter_otype("sensorcompute") == ("sensor", "sensorcompute")
    assert ie_parameter_otype("sensorcomputemethod") == ("sensor", "sensorcomputemethod")
    assert ie_parameter_otype("nfilters") == ("sensor", "nfilters")
    assert ie_parameter_otype("filternames") == ("sensor", "filternames")
    assert ie_parameter_otype("filtercolorletters") == ("sensor", "filtercolorletters")
    assert ie_parameter_otype("filtercolorletterscell") == ("sensor", "filtercolorletterscell")
    assert ie_parameter_otype("filterplotcolors") == ("sensor", "filterplotcolors")
    assert ie_parameter_otype("filterspectra") == ("sensor", "filterspectra")
    assert ie_parameter_otype("colorfilters") == ("sensor", "colorfilters")
    assert ie_parameter_otype("infraredfilter") == ("sensor", "infraredfilter")
    assert ie_parameter_otype("irfilter") == ("sensor", "irfilter")
    assert ie_parameter_otype("unitblockrows") == ("sensor", "unitblockrows")
    assert ie_parameter_otype("unitblockcols") == ("sensor", "unitblockcols")
    assert ie_parameter_otype("cfasize") == ("sensor", "cfasize")
    assert ie_parameter_otype("spectralqe") == ("sensor", "spectralqe")
    assert ie_parameter_otype("sensorspectralsr") == ("sensor", "sensorspectralsr")
    assert ie_parameter_otype("patterncolors") == ("sensor", "patterncolors")
    assert ie_parameter_otype("pixelspectralqe") == ("pixel", "pixelspectralqe")
    assert ie_parameter_otype("pdspectralqe") == ("pixel", "pdspectralqe")
    assert ie_parameter_otype("pixelqe") == ("pixel", "pixelqe")
    assert ie_parameter_otype("spectralsr") == ("pixel", "spectralsr")
    assert ie_parameter_otype("pdspectralsr") == ("pixel", "pdspectralsr")
    assert ie_parameter_otype("pixelspectralsr") == ("pixel", "pixelspectralsr")
    assert ie_parameter_otype("sr") == ("pixel", "sr")
    assert ie_parameter_otype("dynamic range") == ("sensor", "dynamicrange")
    assert ie_parameter_otype("shotnoiseflag") == ("sensor", "shotnoiseflag")
    assert ie_parameter_otype("shot noise flag") == ("sensor", "shotnoiseflag")
    assert ie_parameter_otype("blacklevel") == ("sensor", "blacklevel")
    assert ie_parameter_otype("black level") == ("sensor", "blacklevel")
    assert ie_parameter_otype("zerolevel") == ("sensor", "zerolevel")
    assert ie_parameter_otype("zero level") == ("sensor", "zerolevel")
    assert ie_parameter_otype("bits") == ("sensor", "bits")
    assert ie_parameter_otype("nbits") == ("sensor", "nbits")
    assert ie_parameter_otype("quantization") == ("sensor", "quantization")
    assert ie_parameter_otype("quantizationmethod") == ("sensor", "quantizationmethod")
    assert ie_parameter_otype("maxdigital") == ("sensor", "maxdigital")
    assert ie_parameter_otype("maxoutput") == ("sensor", "maxoutput")
    assert ie_parameter_otype("roi") == ("sensor", "roi")
    assert ie_parameter_otype("roilocs") == ("sensor", "roilocs")
    assert ie_parameter_otype("roirect") == ("sensor", "roirect")
    assert ie_parameter_otype("voltage") == ("sensor", "voltage")
    assert ie_parameter_otype("scene_name") == ("sensor", "scene_name")
    assert ie_parameter_otype("scenename") == ("sensor", "scenename")
    assert ie_parameter_otype("scene name") == ("sensor", "scenename")
    assert ie_parameter_otype("metadata scene name") == ("sensor", "metadatascenename")
    assert ie_parameter_otype("metadatascenename") == ("sensor", "metadatascenename")
    assert ie_parameter_otype("lens") == ("sensor", "lens")
    assert ie_parameter_otype("metadata lens") == ("sensor", "metadatalens")
    assert ie_parameter_otype("metadata lensname") == ("sensor", "metadatalensname")
    assert ie_parameter_otype("metadataopticsname") == ("sensor", "metadataopticsname")
    assert ie_parameter_otype("metadata optics name") == ("sensor", "metadataopticsname")
    assert ie_parameter_otype("metadatasensorname") == ("sensor", "metadatasensorname")
    assert ie_parameter_otype("metadata sensor name") == ("sensor", "metadatasensorname")
    assert ie_parameter_otype("metadatacrop") == ("sensor", "metadatacrop")
    assert ie_parameter_otype("metadata crop") == ("sensor", "metadatacrop")
    assert ie_parameter_otype("sensoretendue") == ("sensor", "sensoretendue")
    assert ie_parameter_otype("microlens") == ("sensor", "microlens")
    assert ie_parameter_otype("ml") == ("sensor", "ml")
    assert ie_parameter_otype("mlens") == ("sensor", "mlens")
    assert ie_parameter_otype("ulens") == ("sensor", "ulens")
    assert ie_parameter_otype("microlensoffset") == ("sensor", "microlensoffset")
    assert ie_parameter_otype("microlens offset") == ("sensor", "microlensoffset")
    assert ie_parameter_otype("mloffset") == ("sensor", "mloffset")
    assert ie_parameter_otype("microlensoffsetmicrons") == ("sensor", "microlensoffsetmicrons")
    assert ie_parameter_otype("microlens offset microns") == ("sensor", "microlensoffsetmicrons")
    assert ie_parameter_otype("vignetting") == ("sensor", "vignetting")
    assert ie_parameter_otype("vignettingflag") == ("sensor", "vignettingflag")
    assert ie_parameter_otype("vignetting flag") == ("sensor", "vignettingflag")
    assert ie_parameter_otype("pixelvignetting") == ("sensor", "pixelvignetting")
    assert ie_parameter_otype("pixel vignetting") == ("sensor", "pixelvignetting")
    assert ie_parameter_otype("sensorvignetting") == ("sensor", "sensorvignetting")
    assert ie_parameter_otype("sensor vignetting") == ("sensor", "vignetting")
    assert ie_parameter_otype("bare etendue") == ("sensor", "bareetendue")
    assert ie_parameter_otype("sensorbareetendue") == ("sensor", "sensorbareetendue")
    assert ie_parameter_otype("sensor bare etendue") == ("sensor", "bareetendue")
    assert ie_parameter_otype("nomicrolensetendue") == ("sensor", "nomicrolensetendue")
    assert ie_parameter_otype("no microlens etendue") == ("sensor", "nomicrolensetendue")
    assert ie_parameter_otype("vignettingname") == ("sensor", "vignettingname")
    assert ie_parameter_otype("vignetting name") == ("sensor", "vignettingname")
    assert ie_parameter_otype("ngridsamples") == ("sensor", "ngridsamples")
    assert ie_parameter_otype("color") == ("sensor", "color")
    assert ie_parameter_otype("digitalvalue") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("digital value") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("digitalvalues") == ("sensor", "digitalvalues")
    assert ie_parameter_otype("digital values") == ("sensor", "digitalvalues")
    assert ie_parameter_otype("electron") == ("sensor", "electron")
    assert ie_parameter_otype("electrons per area") == ("sensor", "electronsperarea")
    assert ie_parameter_otype("filternamescellarray") == ("sensor", "filternamescellarray")
    assert ie_parameter_otype("filter names cell array") == ("sensor", "filternamescellarray")
    assert ie_parameter_otype("filtercolornamescellarray") == ("sensor", "filtercolornamescellarray")
    assert ie_parameter_otype("filter color names cell array") == ("sensor", "filtercolornamescellarray")
    assert ie_parameter_otype("filternamescell") == ("sensor", "filternamescell")
    assert ie_parameter_otype("filter names cell") == ("sensor", "filternamescell")
    assert ie_parameter_otype("lut") == ("sensor", "lut")
    assert ie_parameter_otype("quantizatonlut") == ("sensor", "quantizatonlut")
    assert ie_parameter_otype("quantizationlut") == ("sensor", "quantizationlut")
    assert ie_parameter_otype("quantization lut") == ("sensor", "quantizationlut")
    assert ie_parameter_otype("qmethod") == ("sensor", "qmethod")
    assert ie_parameter_otype("qMethod") == ("sensor", "qmethod")
    assert ie_parameter_otype("quantizationstructure") == ("sensor", "quantizationstructure")
    assert ie_parameter_otype("quantization structure") == ("sensor", "quantizationstructure")
    assert ie_parameter_otype("cfapattern") == ("sensor", "cfapattern")
    assert ie_parameter_otype("pattern") == ("sensor", "pattern")
    assert ie_parameter_otype("cfaname") == ("sensor", "cfaname")
    assert ie_parameter_otype("unitblockconfig") == ("sensor", "unitblockconfig")
    assert ie_parameter_otype("sensorspectrum") == ("sensor", "sensorspectrum")
    assert ie_parameter_otype("wavelengthresolution") == ("sensor", "wavelengthresolution")
    assert ie_parameter_otype("binwidth") == ("sensor", "binwidth")
    assert ie_parameter_otype("numberofwavelengthsamples") == ("sensor", "numberofwavelengthsamples")
    assert ie_parameter_otype("dvorvolts") == ("sensor", "dvorvolts")
    assert ie_parameter_otype("dv or volts") == ("sensor", "dvorvolts")
    assert ie_parameter_otype("digitalorvolts") == ("sensor", "digitalorvolts")
    assert ie_parameter_otype("digital or volts") == ("sensor", "digitalorvolts")
    assert ie_parameter_otype("voltimages") == ("sensor", "voltimages")
    assert ie_parameter_otype("volt images") == ("sensor", "voltimages")
    assert ie_parameter_otype("sensor rows") == ("sensor", "rows")
    assert ie_parameter_otype("sensor cols") == ("sensor", "cols")
    assert ie_parameter_otype("sensor size") == ("sensor", "size")
    assert ie_parameter_otype("sensor dimension") == ("sensor", "dimension")
    assert ie_parameter_otype("sensor arraywidth") == ("sensor", "arraywidth")
    assert ie_parameter_otype("sensor arrayheight") == ("sensor", "arrayheight")
    assert ie_parameter_otype("sensor wspatialresolution") == ("sensor", "wspatialresolution")
    assert ie_parameter_otype("sensor hspatialresolution") == ("sensor", "hspatialresolution")
    assert ie_parameter_otype("sensor deltax") == ("sensor", "deltax")
    assert ie_parameter_otype("sensor deltay") == ("sensor", "deltay")
    assert ie_parameter_otype("sensor volts") == ("sensor", "volts")
    assert ie_parameter_otype("sensor dv") == ("sensor", "dv")
    assert ie_parameter_otype("sensor electrons") == ("sensor", "electrons")
    assert ie_parameter_otype("sensor voltage") == ("sensor", "voltage")
    assert ie_parameter_otype("sensor digitalvalue") == ("sensor", "digitalvalue")
    assert ie_parameter_otype("roivolts") == ("sensor", "roivolts")
    assert ie_parameter_otype("roidata") == ("sensor", "roidata")
    assert ie_parameter_otype("roidatav") == ("sensor", "roidatav")
    assert ie_parameter_otype("roidatavolts") == ("sensor", "roidatavolts")
    assert ie_parameter_otype("roielectrons") == ("sensor", "roielectrons")
    assert ie_parameter_otype("roidatae") == ("sensor", "roidatae")
    assert ie_parameter_otype("roidataelectrons") == ("sensor", "roidataelectrons")
    assert ie_parameter_otype("roidv") == ("sensor", "roidv")
    assert ie_parameter_otype("roidigitalcount") == ("sensor", "roidigitalcount")
    assert ie_parameter_otype("roivoltsmean") == ("sensor", "roivoltsmean")
    assert ie_parameter_otype("roielectronsmean") == ("sensor", "roielectronsmean")
    assert ie_parameter_otype("responseratio") == ("sensor", "responseratio")
    assert ie_parameter_otype("volts2maxratio") == ("sensor", "volts2maxratio")
    assert ie_parameter_otype("sensor chromaticity") == ("sensor", "chromaticity")
    assert ie_parameter_otype("sensor roichromaticitymean") == ("sensor", "roichromaticitymean")
    assert ie_parameter_otype("sensor hlinevolts") == ("sensor", "hlinevolts")
    assert ie_parameter_otype("sensor hlineelectrons") == ("sensor", "hlineelectrons")
    assert ie_parameter_otype("sensor hlinedv") == ("sensor", "hlinedv")
    assert ie_parameter_otype("sensor vlinevolts") == ("sensor", "vlinevolts")
    assert ie_parameter_otype("sensor vlineelectrons") == ("sensor", "vlineelectrons")
    assert ie_parameter_otype("sensor vlinedv") == ("sensor", "vlinedv")
    assert ie_parameter_otype("fill factor") == ("pixel", "fillfactor")
    assert ie_parameter_otype("pixelsize") == ("pixel", "pixelsize")
    assert ie_parameter_otype("pixelwidth") == ("pixel", "pixelwidth")
    assert ie_parameter_otype("pixelheight") == ("pixel", "pixelheight")
    assert ie_parameter_otype("pixelwidthmeters") == ("pixel", "pixelwidthmeters")
    assert ie_parameter_otype("pixelheightmeters") == ("pixel", "pixelheightmeters")
    assert ie_parameter_otype("conversion gain") == ("pixel", "conversiongain")
    assert ie_parameter_otype("conversiongainvpelectron") == ("pixel", "conversiongainvpelectron")
    assert ie_parameter_otype("conversion gain v per electron") == ("pixel", "conversiongainvperelectron")
    assert ie_parameter_otype("vswing") == ("pixel", "vswing")
    assert ie_parameter_otype("wellcapacity") == ("pixel", "wellcapacity")
    assert ie_parameter_otype("well capacity") == ("pixel", "wellcapacity")
    assert ie_parameter_otype("voltsperelectron") == ("pixel", "voltsperelectron")
    assert ie_parameter_otype("volts per electron") == ("pixel", "voltsperelectron")
    assert ie_parameter_otype("saturation voltage") == ("pixel", "saturationvoltage")
    assert ie_parameter_otype("maxvoltage") == ("pixel", "maxvoltage")
    assert ie_parameter_otype("max voltage") == ("pixel", "maxvoltage")
    assert ie_parameter_otype("darkcurrentdensity") == ("pixel", "darkcurrentdensity")
    assert ie_parameter_otype("dark current density") == ("pixel", "darkcurrentdensity")
    assert ie_parameter_otype("dark current per pixel") == ("pixel", "darkcurrentperpixel")
    assert ie_parameter_otype("darkvolt") == ("pixel", "darkvolt")
    assert ie_parameter_otype("darkvolts") == ("pixel", "darkvolts")
    assert ie_parameter_otype("darkvoltageperpixelpersec") == ("pixel", "darkvoltageperpixelpersec")
    assert ie_parameter_otype("darkvoltageperpixel") == ("pixel", "darkvoltageperpixel")
    assert ie_parameter_otype("dark voltage per pixel") == ("pixel", "darkvoltageperpixel")
    assert ie_parameter_otype("darkelectrons") == ("pixel", "darkelectrons")
    assert ie_parameter_otype("voltspersecond") == ("pixel", "voltspersecond")
    assert ie_parameter_otype("volts per second") == ("pixel", "voltspersecond")
    assert ie_parameter_otype("read noise") == ("pixel", "readnoise")
    assert ie_parameter_otype("readnoiseelectrons") == ("pixel", "readnoiseelectrons")
    assert ie_parameter_otype("readstandarddeviationelectrons") == ("pixel", "readstandarddeviationelectrons")
    assert ie_parameter_otype("readnoisevolts") == ("pixel", "readnoisevolts")
    assert ie_parameter_otype("read noise volts") == ("pixel", "readnoisevolts")
    assert ie_parameter_otype("readstandarddeviationvolts") == ("pixel", "readstandarddeviationvolts")
    assert ie_parameter_otype("read standard deviation volts") == ("pixel", "readstandarddeviationvolts")
    assert ie_parameter_otype("read standard deviation electrons") == ("pixel", "readstandarddeviationelectrons")
    assert ie_parameter_otype("readnoisestdvolts") == ("pixel", "readnoisestdvolts")
    assert ie_parameter_otype("read noise std volts") == ("pixel", "readnoisestdvolts")
    assert ie_parameter_otype("readnoisemillivolts") == ("pixel", "readnoisemillivolts")
    assert ie_parameter_otype("read noise millivolts") == ("pixel", "readnoisemillivolts")
    assert ie_parameter_otype("widthgap") == ("pixel", "widthgap")
    assert ie_parameter_otype("width gap") == ("pixel", "widthgap")
    assert ie_parameter_otype("widthbetweenpixels") == ("pixel", "widthbetweenpixels")
    assert ie_parameter_otype("width between pixels") == ("pixel", "widthbetweenpixels")
    assert ie_parameter_otype("heightgap") == ("pixel", "heightgap")
    assert ie_parameter_otype("heightbetweenpixels") == ("pixel", "heightbetweenpixels")
    assert ie_parameter_otype("height between pixels") == ("pixel", "heightbetweenpixels")
    assert ie_parameter_otype("xyspacing") == ("pixel", "xyspacing")
    assert ie_parameter_otype("pixel width meters") == ("pixel", "widthmeters")
    assert ie_parameter_otype("pixel height meters") == ("pixel", "heightmeters")
    assert ie_parameter_otype("pdarea") == ("pixel", "pdarea")
    assert ie_parameter_otype("pd width") == ("pixel", "pdwidth")
    assert ie_parameter_otype("pd dimension") == ("pixel", "pddimension")
    assert ie_parameter_otype("pddimension") == ("pixel", "pddimension")
    assert ie_parameter_otype("pixeldepth") == ("pixel", "pixeldepth")
    assert ie_parameter_otype("layer thicknesses") == ("pixel", "layerthicknesses")
    assert ie_parameter_otype("layerthicknesses") == ("pixel", "layerthicknesses")
    assert ie_parameter_otype("refractive index") == ("pixel", "refractiveindex")
    assert ie_parameter_otype("refractive indices") == ("pixel", "refractiveindices")
    assert ie_parameter_otype("refractiveindices") == ("pixel", "refractiveindices")
    assert ie_parameter_otype("n") == ("pixel", "n")
    assert ie_parameter_otype("stack height") == ("pixel", "stackheight")
    assert ie_parameter_otype("stackheight") == ("pixel", "stackheight")
    assert ie_parameter_otype("pixel depth meters") == ("pixel", "depthmeters")
    assert ie_parameter_otype("pixeldepthmeters") == ("pixel", "pixeldepthmeters")
    assert ie_parameter_otype("pd xpos") == ("pixel", "pdxpos")
    assert ie_parameter_otype("pdxpos") == ("pixel", "pdxpos")
    assert ie_parameter_otype("pdypos") == ("pixel", "pdypos")
    assert ie_parameter_otype("photodetector x position") == ("pixel", "photodetectorxposition")
    assert ie_parameter_otype("photodetectorxposition") == ("pixel", "photodetectorxposition")
    assert ie_parameter_otype("photodetectoryposition") == ("pixel", "photodetectoryposition")
    assert ie_parameter_otype("pd position") == ("pixel", "pdposition")
    assert ie_parameter_otype("pdposition") == ("pixel", "pdposition")
    assert ie_parameter_otype("photodetectorwidth") == ("pixel", "photodetectorwidth")
    assert ie_parameter_otype("photodetectorheight") == ("pixel", "photodetectorheight")
    assert ie_parameter_otype("photodetectorsize") == ("pixel", "photodetectorsize")
    assert ie_parameter_otype("pixelspectrum") == ("pixel", "pixelspectrum")
    assert ie_parameter_otype("pixelwavelength") == ("pixel", "pixelwavelength")
    assert ie_parameter_otype("pixelwavelengthsamples") == ("pixel", "pixelwavelengthsamples")
    assert ie_parameter_otype("pixelbinwidth") == ("pixel", "pixelbinwidth")
    assert ie_parameter_otype("pixelnwave") == ("pixel", "pixelnwave")
    assert ie_parameter_otype("quantum efficiency") == ("pixel", "quantumefficiency")
    assert ie_parameter_otype("pixel quantum efficiency") == ("pixel", "quantumefficiency")
    assert ie_parameter_otype("photodetector quantum efficiency") == ("pixel", "photodetectorquantumefficiency")
    assert ie_parameter_otype("photodetector spectral quantum efficiency") == ("pixel", "photodetectorspectralquantumefficiency")
    assert ie_parameter_otype("pixeldynamicrange") == ("pixel", "pixeldynamicrange")
    assert ie_parameter_otype("diffusion MTF") == ("sensor", "diffusionmtf")
    assert ie_parameter_otype("ag") == ("sensor", "ag")
    assert ie_parameter_otype("ao") == ("sensor", "ao")
    assert ie_parameter_otype("human cone densities") == ("sensor", "humanconedensities")
    assert ie_parameter_otype("human cone seed") == ("sensor", "humanconeseed")
    assert ie_parameter_otype("mcccornerpoints") == ("sensor", "mcccornerpoints")
    assert ie_parameter_otype("mcc corner points") == ("sensor", "mcccornerpoints")
    assert ie_parameter_otype("mccrecthandles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("mcc rect handles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("consistency") == ("sensor", "consistency")
    assert ie_parameter_otype("sensor consistency") == ("sensor", "consistency")
    assert ie_parameter_otype("sensor compute") == ("sensor", "compute")
    assert ie_parameter_otype("sensor compute method") == ("sensor", "computemethod")
    assert ie_parameter_otype("integration time") == ("sensor", "integrationtime")
    assert ie_parameter_otype("exptime") == ("sensor", "exptime")
    assert ie_parameter_otype("exptimes") == ("sensor", "exptimes")
    assert ie_parameter_otype("exposuretimes") == ("sensor", "exposuretimes")
    assert ie_parameter_otype("exposuretime") == ("sensor", "exposuretime")
    assert ie_parameter_otype("expduration") == ("sensor", "expduration")
    assert ie_parameter_otype("exposureduration") == ("sensor", "exposureduration")
    assert ie_parameter_otype("exposuredurations") == ("sensor", "exposuredurations")
    assert ie_parameter_otype("uniqueintegrationtimes") == ("sensor", "uniqueintegrationtimes")
    assert ie_parameter_otype("uniqueexptime") == ("sensor", "uniqueexptime")
    assert ie_parameter_otype("unique exptimes") == ("sensor", "uniqueexptimes")
    assert ie_parameter_otype("uniqueexptimes") == ("sensor", "uniqueexptimes")
    assert ie_parameter_otype("geometricmeanexposuretime") == ("sensor", "geometricmeanexposuretime")
    assert ie_parameter_otype("central exposure") == ("sensor", "centralexposure")
    assert ie_parameter_otype("centralexposure") == ("sensor", "centralexposure")
    assert ie_parameter_otype("expmethod") == ("sensor", "expmethod")
    assert ie_parameter_otype("exposuremethod") == ("sensor", "exposuremethod")
    assert ie_parameter_otype("exposure method") == ("sensor", "exposuremethod")
    assert ie_parameter_otype("nexposures") == ("sensor", "nexposures")
    assert ie_parameter_otype("n exposures") == ("sensor", "nexposures")
    assert ie_parameter_otype("exposureplane") == ("sensor", "exposureplane")
    assert ie_parameter_otype("exposure plane") == ("sensor", "exposureplane")
    assert ie_parameter_otype("cds") == ("sensor", "cds")
    assert ie_parameter_otype("correlateddoublesampling") == ("sensor", "correlateddoublesampling")
    assert ie_parameter_otype("correlated double sampling") == ("sensor", "correlateddoublesampling")
    assert ie_parameter_otype("autoexp") == ("sensor", "autoexp")
    assert ie_parameter_otype("autoexposure") == ("sensor", "autoexposure")
    assert ie_parameter_otype("auto exposure") == ("sensor", "autoexposure")
    assert ie_parameter_otype("automaticexposure") == ("sensor", "automaticexposure")
    assert ie_parameter_otype("automatic exposure") == ("sensor", "automaticexposure")
    assert ie_parameter_otype("pixelsamples") == ("sensor", "pixelsamples")
    assert ie_parameter_otype("pixel samples") == ("sensor", "pixelsamples")
    assert ie_parameter_otype("npixelsamplesforcomputing") == ("sensor", "npixelsamplesforcomputing")
    assert ie_parameter_otype("n pixel samples for computing") == ("sensor", "npixelsamplesforcomputing")
    assert ie_parameter_otype("spatialsamplesperpixel") == ("sensor", "spatialsamplesperpixel")
    assert ie_parameter_otype("spatial samples per pixel") == ("sensor", "spatialsamplesperpixel")
    assert ie_parameter_otype("reusenoise") == ("sensor", "reusenoise")
    assert ie_parameter_otype("noiseseed") == ("sensor", "noiseseed")
    assert ie_parameter_otype("responsetype") == ("sensor", "responsetype")
    assert ie_parameter_otype("response type") == ("sensor", "responsetype")
    assert ie_parameter_otype("eye movement") == ("sensor", "eyemovement")
    assert ie_parameter_otype("eyemovement") == ("sensor", "eyemovement")
    assert ie_parameter_otype("sensormovement") == ("sensor", "sensormovement")
    assert ie_parameter_otype("movementpositions") == ("sensor", "movementpositions")
    assert ie_parameter_otype("movement positions") == ("sensor", "movementpositions")
    assert ie_parameter_otype("sensorpositions") == ("sensor", "sensorpositions")
    assert ie_parameter_otype("sensor positions") == ("sensor", "positions")
    assert ie_parameter_otype("sensorpositionsx") == ("sensor", "sensorpositionsx")
    assert ie_parameter_otype("sensor positions x") == ("sensor", "positionsx")
    assert ie_parameter_otype("sensorpositionsy") == ("sensor", "sensorpositionsy")
    assert ie_parameter_otype("sensor positions y") == ("sensor", "positionsy")
    assert ie_parameter_otype("framesperposition") == ("sensor", "framesperposition")
    assert ie_parameter_otype("framesperpositions") == ("sensor", "framesperpositions")
    assert ie_parameter_otype("frames per position") == ("sensor", "framesperposition")
    assert ie_parameter_otype("exposuretimesperposition") == ("sensor", "exposuretimesperposition")
    assert ie_parameter_otype("exposure times per position") == ("sensor", "exposuretimesperposition")
    assert ie_parameter_otype("etimeperpos") == ("sensor", "etimeperpos")
    assert ie_parameter_otype("etime per pos") == ("sensor", "etimeperpos")
    assert ie_parameter_otype("human") == ("sensor", "human")
    assert ie_parameter_otype("cone type") == ("sensor", "conetype")
    assert ie_parameter_otype("conetype") == ("sensor", "conetype")
    assert ie_parameter_otype("humanconetype") == ("sensor", "humanconetype")
    assert ie_parameter_otype("human cone type") == ("sensor", "humanconetype")
    assert ie_parameter_otype("densities") == ("sensor", "densities")
    assert ie_parameter_otype("humanconedensities") == ("sensor", "humanconedensities")
    assert ie_parameter_otype("cone xy") == ("sensor", "conexy")
    assert ie_parameter_otype("conexy") == ("sensor", "conexy")
    assert ie_parameter_otype("cone locs") == ("sensor", "conelocs")
    assert ie_parameter_otype("conelocs") == ("sensor", "conelocs")
    assert ie_parameter_otype("xy") == ("sensor", "xy")
    assert ie_parameter_otype("humanconelocs") == ("sensor", "humanconelocs")
    assert ie_parameter_otype("human cone locs") == ("sensor", "humanconelocs")
    assert ie_parameter_otype("rseed") == ("sensor", "rseed")
    assert ie_parameter_otype("human rseed") == ("sensor", "humanrseed")
    assert ie_parameter_otype("humanrseed") == ("sensor", "humanrseed")
    assert ie_parameter_otype("humanconeseed") == ("sensor", "humanconeseed")
    assert ie_parameter_otype("chart parameters") == ("sensor", "chartparameters")
    assert ie_parameter_otype("chartparameters") == ("sensor", "chartparameters")
    assert ie_parameter_otype("corner points") == ("sensor", "cornerpoints")
    assert ie_parameter_otype("cornerpoints") == ("sensor", "cornerpoints")
    assert ie_parameter_otype("chart corner points") == ("sensor", "chartcornerpoints")
    assert ie_parameter_otype("chartcornerpoints") == ("sensor", "chartcornerpoints")
    assert ie_parameter_otype("chart corners") == ("sensor", "chartcorners")
    assert ie_parameter_otype("chartcorners") == ("sensor", "chartcorners")
    assert ie_parameter_otype("chart rects") == ("sensor", "chartrects")
    assert ie_parameter_otype("chartrects") == ("sensor", "chartrects")
    assert ie_parameter_otype("chart rectangles") == ("sensor", "chartrectangles")
    assert ie_parameter_otype("chartrectangles") == ("sensor", "chartrectangles")
    assert ie_parameter_otype("current rect") == ("sensor", "currentrect")
    assert ie_parameter_otype("currentrect") == ("sensor", "currentrect")
    assert ie_parameter_otype("chart current rect") == ("sensor", "chartcurrentrect")
    assert ie_parameter_otype("chartcurrentrect") == ("sensor", "chartcurrentrect")
    assert ie_parameter_otype("mccrecthandles") == ("sensor", "mccrecthandles")
    assert ie_parameter_otype("scene name") == ("sensor", "scenename")
    assert ie_parameter_otype("metadata scene name") == ("sensor", "metadatascenename")
    assert ie_parameter_otype("metadatalensname") == ("sensor", "metadatalensname")
    assert ie_parameter_otype("metadata lens name") == ("sensor", "metadatalensname")
    assert ie_parameter_otype("metadatalens") == ("sensor", "metadatalens")
    assert ie_parameter_otype("metadata lens") == ("sensor", "metadatalens")
    assert ie_parameter_otype("metadata optics name") == ("sensor", "metadataopticsname")
    assert ie_parameter_otype("metadata sensor name") == ("sensor", "metadatasensorname")
    assert ie_parameter_otype("metadata crop") == ("sensor", "metadatacrop")
    assert ie_parameter_otype("fnumber") == ("optics", "fnumber")
    assert ie_parameter_otype("asset light") == ("asset", "assetlight")
    assert ieParameterOtype("ip display") == ("ip", "display")


def test_ie_parameter_otype_returns_empty_type_for_ambiguous_or_unknown_params() -> None:
    assert ie_parameter_otype("size") == ("", "size")
    assert ie_parameter_otype("mystery parameter") == ("", "mysteryparameter")


def test_convolvecirc_and_image_slanted_edge_match_legacy_aliases() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    kernel = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)

    expected_conv = np.array([[9.0, 7.0], [13.0, 11.0]], dtype=float)
    assert np.allclose(convolve_circ(image, kernel), expected_conv)
    assert np.allclose(convolvecirc(image, kernel), expected_conv)

    edge = image_slanted_edge([4, 6], 1.0, 0.25)
    alias = imageSlantedEdge([4, 6], 1.0, 0.25)
    assert edge.shape == (5, 7)
    assert np.allclose(alias, edge)
    assert float(edge[0, -1]) == pytest.approx(0.25)
    assert float(edge[-1, 0]) == pytest.approx(1.0)


def test_imagesc_helpers_match_headless_scaling_and_aliases() -> None:
    rgb = np.array(
        [
            [[-0.5, 0.5, 1.0], [0.25, 0.75, 0.5]],
            [[0.0, 0.1, 0.2], [0.8, 0.6, 0.4]],
        ],
        dtype=float,
    )
    handle, scaled_rgb = imagesc_rgb(rgb, 0.5)
    alias_handle, alias_scaled = imagescRGB(rgb, 0.5)
    assert handle is None
    assert alias_handle is None
    assert scaled_rgb.shape == rgb.shape
    assert np.allclose(alias_scaled, scaled_rgb)
    assert np.all((scaled_rgb >= 0.0) & (scaled_rgb <= 1.0))

    xw, rows, cols, _ = rgb_to_xw_format(np.clip(rgb, 0.0, None))
    _, scaled_xw = imagescRGB(xw, rows, cols, 1.0)
    assert scaled_xw.shape == rgb.shape

    opp = np.dstack(
        (
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float),
            np.array([[-1.0, 0.0], [1.0, 0.5]], dtype=float),
            np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float),
        )
    )
    res, cmap = imagesc_opp(opp, 0.4, 8)
    alias_res, alias_cmap = imagescOPP(opp, 0.4, 8)
    assert res.shape == opp.shape
    assert cmap.shape == (8, 3, 3)
    assert np.allclose(alias_res, res)
    assert np.allclose(alias_cmap, cmap)

    mono = np.arange(4.0, dtype=float).reshape(2, 2)
    payload = imagesc_m(mono, bar_dir="eastoutside")
    alias_payload = imagescM(mono, bar_dir="eastoutside")
    assert payload is not None
    assert alias_payload is not None
    assert payload["scaled"] is True
    assert payload["colorbar"]["direction"] == "eastoutside"
    assert np.allclose(alias_payload["image"], mono)


def test_image_spd_visible_gray_and_spd2rgb_match_aliases() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    spd = np.zeros((2, 2, wave.size), dtype=float)
    spd[:, :, 10] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    spd[:, :, 20] = np.array([[0.5, 0.25], [0.75, 1.5]], dtype=float)

    visible = image_spd(spd, wave, 1.0, display_flag=-1)
    alias_visible = imageSPD(spd, wave, 1.0, 2, 2, -1)
    expected_xyz = np.asarray(ieXYZFromPhotons(spd, wave), dtype=float)
    expected_xyz = expected_xyz / float(np.max(expected_xyz))
    expected_visible = xyz2srgb(expected_xyz)
    assert np.allclose(visible, expected_visible)
    assert np.allclose(alias_visible, expected_visible)

    gray = image_spd(spd, wave, 1.0, 2, 2, -2)
    expected_gray = np.mean(spd, axis=2)
    expected_gray = expected_gray / float(np.max(expected_gray))
    expected_gray = np.repeat(expected_gray[:, :, np.newaxis], 3, axis=2)
    assert np.allclose(gray, expected_gray)

    spd_xw, _, _, _ = rgb_to_xw_format(spd)
    rgb_xw = image_spd2rgb(spd_xw, wave, 1.0)
    alias_rgb_xw = imageSPD2RGB(spd_xw, wave, 1.0)
    visible_xw, _, _, _ = rgb_to_xw_format(expected_visible)
    assert np.allclose(rgb_xw, visible_xw)
    assert np.allclose(alias_rgb_xw, visible_xw)


def test_image_hc2rgb_returns_waveband_stack_and_overlay(asset_store) -> None:
    scene = sceneCreate("slanted bar", 64, asset_store=asset_store)
    rows = int(sceneGet(scene, "rows"))
    cols = int(sceneGet(scene, "cols"))

    rgb_images, overlay = image_hc2rgb(scene, 3, [10, 10])
    alias_images, alias_overlay = imagehc2rgb(scene, 3, [10, 10])

    assert rgb_images.shape == (rows, cols, 3, 3)
    assert overlay.shape[0] > rows
    assert overlay.shape[1] > cols
    assert np.all((rgb_images >= 0.0) & (rgb_images <= 1.0))
    assert np.all((overlay >= 0.0) & (overlay <= 1.0))
    assert np.allclose(alias_images, rgb_images)
    assert np.allclose(alias_overlay, overlay)


def test_statistics_gaussian_helpers_match_legacy_aliases() -> None:
    gaussian = bi_normal(1.0, 2.0, 0.0, 5)
    alias_gaussian = biNormal(1.0, 2.0, 0.0, 5)
    assert gaussian.shape == (5, 5)
    assert np.allclose(alias_gaussian, gaussian)
    assert float(np.sum(gaussian)) == pytest.approx(1.0)

    t = np.arange(5.0, dtype=float)
    gamma_values = gamma_pdf(t, 2, 1.0)
    alias_gamma = gammaPDF(t, 2, 1.0)
    expected_gamma = (t**1) * np.exp(-t) / math.factorial(1)
    expected_gamma = expected_gamma / float(np.sum(expected_gamma))
    assert np.allclose(gamma_values, expected_gamma)
    assert np.allclose(alias_gamma, expected_gamma)

    rf_support = {"X": np.array([-1.0, 0.0, 1.0]), "Y": np.array([-1.0, 0.0, 1.0])}
    gaussian_rf = get_gaussian(np.eye(2, dtype=float), rf_support)
    alias_rf = getGaussian(np.eye(2, dtype=float), rf_support)
    assert gaussian_rf.shape == (3, 3)
    assert np.allclose(alias_rf, gaussian_rf)
    assert float(np.sum(gaussian_rf)) == pytest.approx(1.0, rel=1e-6)

    norm_values = ie_normpdf(np.array([-1.0, 0.0, 1.0], dtype=float))
    alias_norm = ieNormpdf(np.array([-1.0, 0.0, 1.0], dtype=float))
    expected_norm = np.exp(-0.5 * np.array([1.0, 0.0, 1.0])) / np.sqrt(2.0 * np.pi)
    assert np.allclose(norm_values, expected_norm)
    assert np.allclose(alias_norm, expected_norm)


def test_statistics_random_helpers_match_legacy_aliases() -> None:
    expected = np.array([[1.0, 2.0, 3.0]], dtype=float)
    uniform = np.exp(-expected / 2.0)
    exp_values = exp_rand(2.0, [1, 3], uniform_samples=uniform)
    alias_exp = expRand(2.0, [1, 3], uniform_samples=uniform)
    assert np.allclose(exp_values, expected)
    assert np.allclose(alias_exp, expected)

    exprnd_values = ie_exprnd(2.0, 1, 3, uniform_samples=uniform)
    alias_exprnd = ieExprnd(2.0, 1, 3, uniform_samples=uniform)
    assert np.allclose(exprnd_values, expected)
    assert np.allclose(alias_exprnd, expected)

    z_samples = np.array([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    mvn = ie_mvnrnd([1.0, 2.0], [[1.0, 0.0], [0.0, 4.0]], standard_normal_samples=z_samples)
    alias_mvn = ieMvnrnd([1.0, 2.0], [[1.0, 0.0], [0.0, 4.0]], standard_normal_samples=z_samples)
    expected_mvn = np.array([[1.0, 4.0], [2.0, 0.0]], dtype=float)
    assert np.allclose(mvn, expected_mvn)
    assert np.allclose(alias_mvn, expected_mvn)

    lam = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=float)
    poisson_values, poisson_seed = ie_poisson(lam, "noiseFlag", "frozen", "seed", 17)
    alias_values, alias_seed = iePoisson(lam, "noiseFlag", "frozen", "seed", 17)
    repeat_values, repeat_seed = ie_poisson(lam, "noiseFlag", "frozen", "seed", 17)
    assert poisson_seed == 17
    assert alias_seed == 17
    assert repeat_seed == 17
    assert np.array_equal(poisson_values, repeat_values)
    assert np.array_equal(alias_values, repeat_values)


def test_statistics_spectral_and_percentile_helpers_match_legacy_aliases() -> None:
    rgb = np.dstack(
        (
            np.array([[0.1, 0.5], [0.7, 0.9]], dtype=float),
            np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
            np.array([[0.3, 0.1], [0.5, 0.7]], dtype=float),
        )
    )
    frequencies, amplitudes = ie_one_over_f(rgb, gamma=1.0)
    alias_frequencies, alias_amplitudes = ieOneOverF(rgb, gamma=1.0)
    assert frequencies.shape == amplitudes.shape
    assert np.allclose(alias_frequencies, frequencies)
    assert np.allclose(alias_amplitudes, amplitudes)
    assert np.all(frequencies > 0.0)
    assert np.all(amplitudes >= 0.0)

    data = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]], dtype=float)
    pcs, mean = ie_prcomp(data, "remove mean", 1)
    alias_pcs, alias_mean = iePrcomp(data, "remove mean", 1)
    assert pcs.shape == (2, 1)
    assert np.allclose(alias_pcs, pcs)
    assert np.allclose(mean, np.array([3.0, 14.0 / 3.0], dtype=float))
    assert np.allclose(alias_mean, mean)
    assert float(np.linalg.norm(pcs[:, 0])) == pytest.approx(1.0)

    percentiles = ie_prctile(np.array([1.0, 2.0, 3.0, 4.0], dtype=float), [0.0, 50.0, 100.0])
    alias_percentiles = iePrctile(np.array([1.0, 2.0, 3.0, 4.0], dtype=float), [0.0, 50.0, 100.0])
    assert np.allclose(percentiles, np.array([1.0, 2.5, 4.0], dtype=float))
    assert np.allclose(alias_percentiles, percentiles)

    x = np.array([0.0, 1.0, 2.0], dtype=float)
    params = np.array([[2.0, 3.0, 1.0], [1.0, 1.0, 2.0]], dtype=float)
    lorentz = lorentz_sum(x, params)
    alias_lorentz = lorentzSum(x, params)
    expected_lorentz = 3.0 / (1.0 + (x / 2.0) ** 2) + 1.0 / (1.0 + x**2) ** 2
    assert np.allclose(lorentz, expected_lorentz)
    assert np.allclose(alias_lorentz, expected_lorentz)


def test_statistics_fractal_helpers_match_legacy_aliases() -> None:
    image = np.full((4, 4), 255, dtype=np.uint8)
    image[:2, :2] = 0

    grid = ie_fractal_drawgrid(image, 2)
    alias_grid = ieFractalDrawgrid(image, 2)
    assert grid.shape == (4, 4, 3)
    assert np.array_equal(alias_grid, grid)
    assert np.all(grid[0, :, 0] == 255)
    assert np.all(grid[:, 0, 2] == 255)
    assert np.all(grid[::2, :, 1] == 0)
    assert np.all(grid[:, ::2, 1] == 0)

    fractal_dimension = ie_fractal_dim(image, 1, 2, 1)
    alias_fractal_dimension = ieFractaldim(image, 1, 2, 1)
    assert fractal_dimension == pytest.approx(2.0)
    assert alias_fractal_dimension == pytest.approx(2.0)


def test_hypercube_blur_and_illuminant_scale_match_aliases() -> None:
    cube = np.zeros((3, 3, 2), dtype=float)
    cube[1, 1, 0] = 1.0
    cube[:, :, 1] = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    blurred, kernel = hc_blur(cube, 3)
    alias_blurred, alias_kernel = hcBlur(cube, 3)
    coords = np.arange(3, dtype=float) - 1.0
    x_grid, y_grid = np.meshgrid(coords, coords)
    expected_kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * 0.5**2))
    expected_kernel /= float(np.sum(expected_kernel))
    assert np.allclose(kernel, expected_kernel)
    assert np.allclose(alias_kernel, expected_kernel)
    assert np.allclose(alias_blurred, blurred)
    assert np.allclose(blurred[:, :, 0], convolve2d(cube[:, :, 0], expected_kernel, mode="same"))

    spd = np.array([2.0, 4.0, 8.0], dtype=float)
    weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    illuminant_cube = weights[:, :, np.newaxis] * spd[np.newaxis, np.newaxis, :]
    scale, mean_spd = hc_illuminant_scale(illuminant_cube)
    alias_scale, alias_mean_spd = hcIlluminantScale(illuminant_cube)
    expected_scale = weights / float(np.max(weights))
    assert np.allclose(scale, expected_scale)
    assert np.allclose(alias_scale, expected_scale)
    assert np.allclose(mean_spd, np.max(weights) * spd)
    assert np.allclose(alias_mean_spd, mean_spd)


def test_hypercube_envi_readers_match_aliases(tmp_path) -> None:
    cube = np.arange(1, 1 + 2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    image_path = tmp_path / "sample.img"
    header_path = tmp_path / "sample.hdr"

    bil_payload = cube.transpose(0, 2, 1).reshape(-1)
    bil_payload.tofile(image_path)
    header_path.write_text(
        "\n".join(
            [
                "ENVI",
                "samples = 3",
                "lines = 2",
                "bands = 4",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 12",
                "interleave = bil",
                "byte order = 0",
                "default bands = {4, 2, 1}",
            ]
        ),
        encoding="utf-8",
    )

    info = hc_read_hyspex_imginfo(image_path)
    alias_info = hcReadHyspexImginfo(image_path)
    assert info["lines"] == 2
    assert info["samples"] == 3
    assert info["bands"] == 4
    assert info["byte_order"] == "ieee-le"
    assert np.array_equal(info["default_bands"], np.array([4, 2, 1]))
    assert alias_info["interleave"] == "bil"

    loaded, loaded_info = hc_read_hyspex(image_path)
    alias_loaded, alias_loaded_info = hcReadHyspex(image_path)
    assert np.array_equal(loaded, cube)
    assert np.array_equal(alias_loaded, cube)
    assert loaded_info["interleave"] == "bil"
    assert alias_loaded_info["interleave"] == "bil"

    subset, _ = hc_read_hyspex(image_path, [2], [1, 3], "default")
    expected_subset = cube[np.ix_(np.array([1]), np.array([0, 2]), np.array([3, 1, 0]))]
    assert np.array_equal(subset, expected_subset)


def test_hypercube_display_helpers_match_aliases() -> None:
    cube = np.arange(1, 1 + 3 * 4 * 3, dtype=float).reshape(3, 4, 3)

    mean_gray = hc_image(cube, "mean gray")
    alias_mean_gray = hcimage(cube, "mean gray")
    assert np.allclose(mean_gray, np.mean(cube, axis=2))
    assert np.allclose(alias_mean_gray, mean_gray)

    montage = hc_image(cube, "image montage", [1, 3])
    alias_montage = hcimage(cube, "image montage", [1, 3])
    assert montage[0] is None
    assert montage[2] is None
    assert montage[1].ndim == 2
    assert np.allclose(alias_montage[1], montage[1])

    movie = hc_image(cube, "movie")
    assert movie["type"] == "movie"
    assert movie["frames"].shape == cube.shape
    assert movie["title"] == "Hypercube wavebands: 3"

    cropped, rect = hc_image_crop(cube, [2, 1, 1, 1])
    alias_cropped, alias_rect = hcimageCrop(cube, [2, 1, 1, 1])
    assert cropped.shape == (2, 2, 3)
    assert np.array_equal(rect, np.array([2, 1, 1, 1]))
    assert np.array_equal(alias_rect, rect)
    assert np.array_equal(alias_cropped, cropped)

    rotated, clipped = hc_image_rotate_clip(cube, 50.0, 1)
    alias_rotated, alias_clipped = hcimageRotateClip(cube, 50.0, 1)
    assert rotated.shape == (4, 3, 3)
    assert clipped.shape == (4, 3)
    assert np.array_equal(alias_rotated, rotated)
    assert np.array_equal(alias_clipped, clipped)

    viewer = hc_viewer(cube, [500.0, 600.0, 700.0])
    alias_viewer = hcViewer(cube, [500.0, 600.0, 700.0])
    assert viewer["current_slice"] == 1
    assert viewer["label"] == "Slice: 500"
    assert np.array_equal(viewer["image"], cube[:, :, 0])
    assert np.array_equal(alias_viewer["slice_map"], np.array([500.0, 600.0, 700.0]))


def test_max2_supports_restricted_search() -> None:
    matrix = np.array(
        [
            [16.0, 2.0, 3.0, 13.0],
            [5.0, 11.0, 10.0, 8.0],
            [9.0, 7.0, 6.0, 12.0],
            [4.0, 14.0, 15.0, 1.0],
        ]
    )
    value, ij = max2_fn(matrix, [1, 2, 3], [2, 3])
    alias_value, alias_ij = max2(matrix, [1, 2, 3], [2, 3])

    assert value == 11.0
    assert np.array_equal(ij, np.array([2, 2]))
    assert alias_value == value
    assert np.array_equal(alias_ij, ij)


def test_min2_supports_restricted_search() -> None:
    matrix = np.array(
        [
            [16.0, 2.0, 3.0, 13.0],
            [5.0, 11.0, 10.0, 8.0],
            [9.0, 7.0, 6.0, 12.0],
            [4.0, 14.0, 15.0, 1.0],
        ]
    )
    value, ij = min2_fn(matrix, [1, 2, 3], [2, 3])
    alias_value, alias_ij = min2(matrix, [1, 2, 3], [2, 3])

    assert value == 2.0
    assert np.array_equal(ij, np.array([1, 2]))
    assert alias_value == value
    assert np.array_equal(alias_ij, ij)


def test_comp_struct_replays_nested_common_and_differences() -> None:
    struct_a = {
        "a": 1,
        "b": {
            "c": np.array([1.0, 2.0], dtype=float),
            "d": "left",
        },
    }
    struct_b = {
        "a": 1,
        "b": {
            "c": np.array([1.0, 2.0005], dtype=float),
            "e": "right",
        },
    }

    common, d1, d2 = comp_struct(struct_a, struct_b, tol=1e-3)
    alias_common, alias_d1, alias_d2 = compStruct(struct_a, struct_b, tol=1e-3)

    assert common["a"] == 1
    assert np.allclose(common["b"]["c"], np.array([1.0, 2.0]))
    assert d1 == {"b": {"d": "left"}}
    assert d2 == {"b": {"e": "right"}}
    assert alias_common["a"] == common["a"]
    assert np.allclose(alias_common["b"]["c"], common["b"]["c"])
    assert alias_d1 == d1
    assert alias_d2 == d2


def test_list_struct_returns_headless_field_listing() -> None:
    fields = list_struct([{"a": 1}, {"a": 2}], 0, "sample")
    alias_fields = listStruct({"nested": {"value": "x"}}, 1, "root")

    assert fields == ["Field:\tsample(1).a", "Field:\tsample(2).a"]
    assert alias_fields == ["Field:\troot.nested.value = x"]


def test_zernike_wrappers_match_known_polynomials_and_single_index_mapping() -> None:
    radius = np.array([0.0, 0.5, 1.0], dtype=float)
    theta = np.array([0.0, np.pi / 2.0, np.pi], dtype=float)

    radial = zernpol_fn([2, 4], [0, 2], radius)
    alias_radial = zernpol([2, 4], [0, 2], radius)
    expected_radial = np.column_stack(
        [
            2.0 * radius**2 - 1.0,
            4.0 * radius**4 - 3.0 * radius**2,
        ]
    )
    assert np.allclose(radial, expected_radial)
    assert np.allclose(alias_radial, expected_radial)

    functions = zernfun_fn([1, 1], [-1, 1], radius, theta)
    alias_functions = zernfun([1, 1], [-1, 1], radius, theta)
    expected_functions = np.column_stack(
        [
            radius * np.sin(theta),
            radius * np.cos(theta),
        ]
    )
    assert np.allclose(functions, expected_functions)
    assert np.allclose(alias_functions, expected_functions)

    expected_single_index = zernfun_fn([0, 1, 1, 2], [0, -1, 1, -2], radius, theta)
    single_index = zernfun2_fn([0, 1, 2, 3], radius, theta)
    alias_single_index = zernfun2([0, 1, 2, 3], radius, theta)
    assert np.allclose(single_index, expected_single_index)
    assert np.allclose(alias_single_index, expected_single_index)


def test_programming_helper_wrappers_cover_struct_and_cell_utilities() -> None:
    merged = cell_merge(["a", "b"], ("c",), [])
    alias_merged = cellMerge(["a"], ["b", "c"])
    deleted = cell_delete(["a", "b", "c", "d"], [1, 3])
    alias_deleted = cellDelete(["a", "b", "c", "d"], [2, 4])

    assert merged == ["a", "b", "c"]
    assert alias_merged == ["a", "b", "c"]
    assert deleted == ["b", "d"]
    assert alias_deleted == ["a", "c"]

    appended = append_struct({"a": 1, "b": 2}, {"b": 3, "c": 4})
    alias_appended = appendStruct({"x": 1}, {"x": 2, "y": 3})
    assert appended == {"a": 1, "b": 3, "c": 4}
    assert alias_appended == {"x": 2, "y": 3}

    assert checkfields({"pixel": {"OP": {"pd": {"type": "default"}}}}, "pixel", "OP", "pd", "type")
    assert not checkfields({"pixel": {"OP": {}}}, "pixel", "OP", "pd", "type")
    assert compare_fields({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not compareFields({"a": 1}, {"a": 2})


def test_programming_helper_wrappers_cover_string_and_struct_utilities() -> None:
    contains_vector = ie_contains(["help", "he", 4], "he")
    alias_contains = ieContains("help", "lp")
    replaced = replace_nan(np.array([1.0, np.nan, 3.0]), 5.0)
    alias_replaced = replaceNaN(np.array([np.nan, 2.0]), 7.0)
    pairs = struct2pairs_fn({"a": 1, "b": 2})
    alias_pairs = struct2pairs({"x": 3, "y": 4})
    gathered = gather_struct({"a": [1, {"b": 2}]})
    alias_gathered = gatherStruct({"a": {"b": 2}})

    assert np.array_equal(contains_vector, np.array([True, True, False]))
    assert alias_contains is True
    assert np.array_equal(replaced, np.array([1.0, 5.0, 3.0]))
    assert np.array_equal(alias_replaced, np.array([7.0, 2.0]))
    assert pairs == ["a", 1, "b", 2]
    assert alias_pairs == ["x", 3, "y", 4]
    assert gathered == {"a": [1, {"b": 2}]}
    assert alias_gathered == {"a": {"b": 2}}


def test_programming_helper_wrappers_cover_struct_compare_and_empty_field_cleanup() -> None:
    left = {"root": {"value": 1, "cell": [{"inner": 1}, {"inner": 2}]}, "keep": 1}
    right = {"root": {"value": 2, "cell": [{"inner": 1}, {"inner": 3}]}, "keep": 1}

    differences, common, d1, d2 = ie_struct_compare(left, right, "rootStruct")
    alias_differences, alias_common, alias_d1, alias_d2 = ieStructCompare(left, right, "rootStruct")
    cleaned = ie_struct_remove_empty_field({"a": 1, "b": None, "c": "", "d": [], "e": np.array([])})
    alias_cleaned = ieStructRemoveEmptyField({"keep": "x", "drop": []})

    assert differences == [
        "Path: rootStruct.root.cell{2}.inner | Difference: Values are not equal (Value A: 2, Value B: 3)",
        "Path: rootStruct.root.value | Difference: Values are not equal (Value A: 1, Value B: 2)",
    ]
    assert alias_differences == differences
    assert common == {"root": {"cell": [{"inner": 1}, None]}, "keep": 1}
    assert alias_common == common
    assert d1 == {"root": {"cell": [None, {"inner": 2}], "value": 1}}
    assert d2 == {"root": {"cell": [None, {"inner": 3}], "value": 2}}
    assert alias_d1 == d1
    assert alias_d2 == d2
    assert cleaned == {"a": 1}
    assert alias_cleaned == {"keep": "x"}


def test_iehash_matches_upstream_examples_and_output_formats(tmp_path) -> None:
    version = ie_hash()
    empty_double = np.empty((0, 0), dtype=np.float64)
    binary_vector = np.arange(1.0, 9.0, dtype=np.float64)
    md5_digest = hashlib.md5(b"abc").digest()
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"camera-e2e")

    assert version["HashVersion"] == 4
    assert version["Date"] == [2018, 5, 19]
    assert "MD5" in version["HashMethod"]
    assert ie_hash(empty_double) == "5b302b7b2099a97ba2a276640a192485"
    assert ieHash(binary_vector, "SHA-1", "bin") == "826cf9d3a5d74bbe415e97d4cecf03f445f69225"
    assert ie_hash("abc", {"Input": "ascii", "Method": "SHA-256", "OutFormat": "hex"}) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert np.array_equal(
        ie_hash(np.frombuffer(b"abc", dtype=np.uint8), "MD5", "bin", "uint8"),
        np.frombuffer(md5_digest, dtype=np.uint8),
    )
    assert np.array_equal(
        ieHash(np.frombuffer(b"abc", dtype=np.uint8), "MD5", "bin", "double"),
        np.frombuffer(md5_digest, dtype=np.uint8).astype(float),
    )
    assert ie_hash(np.frombuffer(b"abc", dtype=np.uint8), "MD5", "bin", "base64") == base64.b64encode(md5_digest).decode(
        "ascii"
    )
    assert ie_hash(np.frombuffer(b"abc", dtype=np.uint8), "MD5", "bin", "short") == base64.b64encode(md5_digest).decode(
        "ascii"
    ).rstrip("=")
    assert ie_hash(payload_path, "file", "MD5") == hashlib.md5(b"camera-e2e").hexdigest()
