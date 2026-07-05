"""Curated parity case runners."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import maximum_filter

from .assets import AssetStore, ie_read_color_filter, ie_read_spectra
from .camera import (
    camera_acutance,
    camera_color_accuracy,
    camera_compute,
    camera_create,
    camera_get,
    camera_mtf,
    camera_set,
    camera_vsnr,
    macbeth_color_error,
    macbeth_compare_ideal,
)
from .color import adobergb_parameters, daylight, luminance_from_energy, luminance_from_photons, srgb_parameters
from .description import sensor_description
from .display import display_create, display_get, display_set
from .fileio import ie_save_color_filter, ie_save_multispectral_image, ie_save_si_data_file
from .fileio import sensor_dng_read
from .illuminant import illuminant_create, illuminant_get, illuminant_set
from .iso import edge_to_mtf, ie_iso12233, iso12233, iso_find_slanted_bar
from .metrics import cpiq_csf, chromaticity_xy, cct_from_uv, delta_e_ab, ie_sqri, iso_acutance, metrics_spd, spd_to_cct, srgb_to_color_temp, xyz_from_energy, xyz_to_lab, xyz_to_luv, xyz_to_uv
from .ip import ip_compute, ip_create, ip_get, ip_set
from .optics import (
    _cos4th_factor,
    _oi_geometry,
    _pad_scene,
    _radiance_to_irradiance,
    _shift_invariant_custom_otf,
    _wvf_psf_stack,
    airy_disk,
    optics_build_2d_otf,
    optics_coc,
    optics_defocus_core,
    optics_depth_defocus,
    optics_defocus_displacement,
    optics_dof,
    oi_compute,
    oi_crop,
    oi_create,
    oi_get,
    oi_show_image,
    oi_spatial_resample,
    psf_to_zcoeff_error,
    wvf_compute,
    wvf_aperture,
    wvf_compute_psf,
    wvf_create,
    wvf_defocus_diopters_to_microns,
    wvf_get,
    wvf_load_thibos_virtual_eyes,
    wvf_osa_index_to_zernike_nm,
    wvf_pupil_function,
    wvf_set,
    wvf_to_oi,
    wvf_zernike_nm_to_osa_index,
    optics_psf_to_otf,
    oi_set,
    rt_geometry,
    rt_precompute_psf,
    rt_precompute_psf_apply,
    si_synthetic,
    rt_synthetic,
)
from .plotting import ip_plot, oi_plot, scene_plot, sensor_plot, sensor_plot_line, wvf_plot
from .roi import ie_rect2_locs, vc_get_roi_data
from .scielab import color_transform_matrix, sc_compute_scielab, sc_opponent_filter, sc_params, sc_prepare_filters, scielab, scielab_rgb
from .scene import (
    _dead_leaves_sample_matrix,
    hdr_render,
    ie_reflectance_samples,
    macbeth_read_reflectance,
    scene_add,
    scene_adjust_illuminant,
    scene_adjust_luminance,
    scene_combine,
    scene_create,
    scene_from_file,
    scene_get,
    scene_illuminant_ss,
    scene_interpolate_w,
    scene_reflectance_chart,
    scene_rotate,
    scene_show_image,
    scene_set,
)
from .sensor import (
    _macbeth_ideal_linear_rgb,
    imx490_compute,
    ml_analyze_array_etendue,
    ml_radiance,
    mlens_create,
    mlens_get,
    mlens_set,
    pixel_snr_luxsec,
    pixel_v_per_lux_sec,
    sensor_color_filter,
    sensor_ccm,
    sensor_compute,
    sensor_compute_array,
    sensor_compute_samples,
    sensor_create,
    sensor_create_array,
    sensor_create_ideal,
    sensor_crop,
    sensor_dr,
    sensor_formats,
    sensor_get,
    sensor_set,
)
from .sensor import sensor_snr
from .sensor import sensor_set_size_to_fov
from .sensor import signal_current
from .utils import (
    blackbody,
    dac_to_rgb,
    energy_to_quanta,
    hc_basis,
    ie_fit_line,
    ie_mvnrnd,
    ie_n_to_megapixel,
    image_linear_transform,
    image_flip,
    image_increase_image_rgb_size,
    linear_to_srgb,
    param_format,
    quanta_to_energy,
    rgb_to_xw_format,
    srgb_to_xyz,
    xw_to_rgb_format,
    unit_frequency_list,
    xyz_to_srgb,
)


@dataclass
class ParityCaseResult:
    payload: dict[str, Any]
    context: dict[str, Any]


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def _stats_vector(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    return np.array(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.percentile(flat, 5)),
            float(np.percentile(flat, 95)),
        ],
        dtype=float,
    )


def _deterministic_normal_samples(n_rows: int, n_cols: int) -> np.ndarray:
    indices = np.arange(1, n_rows * n_cols + 1, dtype=float).reshape(n_rows, n_cols, order="F")
    u1 = np.mod(indices * 0.7548776662466927, 1.0)
    u2 = np.mod(indices * 0.5698402909980532, 1.0)
    u1 = np.clip(u1, 1e-6, 1.0 - 1e-6)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _channel_normalize(values: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    return vector / max(float(np.max(np.abs(vector))), 1e-12)


def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
    profile = np.asarray(values, dtype=float).reshape(-1)
    if profile.size == 0:
        return np.zeros(samples, dtype=float)
    if profile.size == 1:
        return np.full(samples, float(profile[0]), dtype=float)
    support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
    query = np.linspace(-1.0, 1.0, samples, dtype=float)
    return np.interp(query, support, profile)


def _luma(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return np.mean(array, axis=2, dtype=float)
    raise ValueError("Expected grayscale or RGB image data.")


def _find_point_peaks(image: Any, *, percentile: float = 99.7, size: int = 7) -> np.ndarray:
    luma = _luma(image)
    maxima = maximum_filter(luma, size=size, mode="nearest")
    threshold = float(np.percentile(luma, percentile))
    peaks = np.argwhere((luma == maxima) & (luma >= threshold))
    if peaks.size == 0:
        raise ValueError("No point peaks were detected in the rendered point-array image.")
    return np.asarray(peaks, dtype=int)


def _select_center_and_edge_peaks(peaks: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    peak_array = np.asarray(peaks, dtype=int).reshape(-1, 2)
    center = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    distances = np.sqrt(np.sum((np.asarray(peak_array, dtype=float) - center) ** 2, axis=1))
    min_distance = float(np.min(distances))
    min_candidates = peak_array[np.isclose(distances, min_distance, rtol=0.0, atol=1.0e-12)]
    min_order = np.lexsort((min_candidates[:, 1], min_candidates[:, 0]))
    center_peak = np.asarray(min_candidates[min_order[0]], dtype=int)

    max_distance = float(np.max(distances))
    max_candidates = peak_array[np.isclose(distances, max_distance, rtol=0.0, atol=1.0e-12)]
    max_order = np.lexsort((max_candidates[:, 1], max_candidates[:, 0]))
    edge_peak = np.asarray(max_candidates[max_order[0]], dtype=int)
    return center_peak, edge_peak


def _crop_square(image: Any, peak_rc: Any, radius: int) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    peak = np.asarray(peak_rc, dtype=int).reshape(2)
    row, col = int(peak[0]), int(peak[1])
    row_min = max(row - radius, 0)
    row_max = min(row + radius + 1, array.shape[0])
    col_min = max(col - radius, 0)
    col_max = min(col + radius + 1, array.shape[1])
    if array.ndim == 2:
        return array[row_min:row_max, col_min:col_max]
    return array[row_min:row_max, col_min:col_max, :]


def _crop_square_padded(image: Any, peak_rc: Any, radius: int) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    peak = np.asarray(peak_rc, dtype=int).reshape(2)
    row, col = int(peak[0]), int(peak[1])
    if array.ndim == 2:
        crop = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
    else:
        crop = np.zeros((2 * radius + 1, 2 * radius + 1, array.shape[2]), dtype=float)

    row_min = max(row - radius, 0)
    row_max = min(row + radius + 1, array.shape[0])
    col_min = max(col - radius, 0)
    col_max = min(col + radius + 1, array.shape[1])

    dest_row_min = row_min - (row - radius)
    dest_row_max = dest_row_min + (row_max - row_min)
    dest_col_min = col_min - (col - radius)
    dest_col_max = dest_col_min + (col_max - col_min)

    if array.ndim == 2:
        crop[dest_row_min:dest_row_max, dest_col_min:dest_col_max] = array[row_min:row_max, col_min:col_max]
        return crop

    crop[dest_row_min:dest_row_max, dest_col_min:dest_col_max, :] = array[row_min:row_max, col_min:col_max, :]
    return crop


def _normalize_edge_profile(values: Any) -> np.ndarray:
    profile = np.asarray(values, dtype=float).reshape(-1)
    if profile.size == 0:
        return np.zeros(0, dtype=float)
    lo = float(np.percentile(profile, 5))
    hi = float(np.percentile(profile, 95))
    if hi - lo <= 1.0e-12:
        normalized = np.zeros_like(profile, dtype=float)
    else:
        normalized = np.clip((profile - lo) / (hi - lo), 0.0, 1.0)
    if normalized.size > 1 and normalized[0] > normalized[-1]:
        normalized = 1.0 - normalized
    return normalized


def _central_vertical_edge_payload(
    image: Any,
    *,
    crop_radius: int,
    band_half_rows: int = 6,
    search_half_width: int = 24,
    profile_half_width: int = 24,
) -> dict[str, np.ndarray]:
    plane = _luma(image)
    center_row = plane.shape[0] // 2
    center_col = plane.shape[1] // 2
    row_min = max(center_row - band_half_rows, 0)
    row_max = min(center_row + band_half_rows + 1, plane.shape[0])
    averaged_profile = np.mean(plane[row_min:row_max, :], axis=0, dtype=float)
    gradient = np.abs(np.diff(averaged_profile))
    if gradient.size == 0:
        edge_col = center_col
    else:
        search_min = min(max(center_col, 0), gradient.size - 1)
        search_max = min(gradient.size, center_col + search_half_width + 1)
        if search_max <= search_min:
            search_min = 0
            search_max = gradient.size
        edge_col = int(search_min + np.argmax(gradient[search_min:search_max]))

    peak_rc = np.asarray([center_row, edge_col], dtype=int)
    crop = _crop_square_padded(image, peak_rc, crop_radius)
    crop_plane = _luma(crop)
    edge_min = max(edge_col - profile_half_width, 0)
    edge_max = min(edge_col + profile_half_width + 1, averaged_profile.size)
    edge_profile = _canonical_profile(_normalize_edge_profile(averaged_profile[edge_min:edge_max]), 129)
    crop_peak = max(float(np.max(crop_plane)), 1.0e-12)
    return {
        "edge_rc": peak_rc + 1,
        "crop_norm": crop_plane / crop_peak,
        "profile_norm": edge_profile,
    }


def _spot_metrics(values: Any, *, spacing_um: float = 1.0) -> dict[str, float]:
    plane = np.asarray(values, dtype=float)
    plane = plane - float(np.min(plane))
    total = float(np.sum(plane, dtype=float))
    if total <= 0.0:
        return {
            "peak": 0.0,
            "rms_radius_px": 0.0,
            "rms_radius_um": 0.0,
            "ee50_radius_px": 0.0,
            "ee50_radius_um": 0.0,
            "ee80_radius_px": 0.0,
            "ee80_radius_um": 0.0,
        }
    yy, xx = np.indices(plane.shape)
    center_y = float(np.sum(yy * plane, dtype=float) / total)
    center_x = float(np.sum(xx * plane, dtype=float) / total)
    radius_px = np.sqrt(np.square(yy - center_y) + np.square(xx - center_x))
    order = np.argsort(radius_px.reshape(-1))
    sorted_radius = radius_px.reshape(-1)[order]
    sorted_energy = plane.reshape(-1)[order]
    cumulative = np.cumsum(sorted_energy, dtype=float)

    def _energy_radius(level: float) -> float:
        index = int(np.searchsorted(cumulative, level * total, side="left"))
        index = min(index, sorted_radius.size - 1)
        return float(sorted_radius[index])

    rms_radius_px = float(np.sqrt(np.sum(np.square(radius_px) * plane, dtype=float) / total))
    ee50_radius_px = _energy_radius(0.50)
    ee80_radius_px = _energy_radius(0.80)
    return {
        "peak": float(np.max(plane)),
        "rms_radius_px": rms_radius_px,
        "rms_radius_um": rms_radius_px * float(spacing_um),
        "ee50_radius_px": ee50_radius_px,
        "ee50_radius_um": ee50_radius_px * float(spacing_um),
        "ee80_radius_px": ee80_radius_px,
        "ee80_radius_um": ee80_radius_px * float(spacing_um),
    }


def _raytrace_psf_payload(current_oi: Any, *, reference_wavelength_nm: float) -> dict[str, Any]:
    field_heights_mm = np.asarray(oi_get(current_oi, "rtpsffieldheight", "mm"), dtype=float).reshape(-1)
    support_x_um = np.asarray(oi_get(current_oi, "rtpsfsupportx", "um"), dtype=float).reshape(-1)
    support_y_um = np.asarray(oi_get(current_oi, "rtpsfsupporty", "um"), dtype=float).reshape(-1)
    psf_spacing_um = np.asarray(oi_get(current_oi, "rtpsfspacing", "um"), dtype=float).reshape(-1)
    sample_spacing_um = float(psf_spacing_um[0]) if psf_spacing_um.size else 1.0
    edge_field_height_mm = float(field_heights_mm[-1]) if field_heights_mm.size else 0.0
    center_psf = np.asarray(oi_get(current_oi, "rtpsfdata", 0.0, float(reference_wavelength_nm)), dtype=float)
    edge_psf = np.asarray(
        oi_get(current_oi, "rtpsfdata", edge_field_height_mm / 1.0e3, float(reference_wavelength_nm)),
        dtype=float,
    )
    center_peak = max(float(np.max(center_psf)), 1.0e-12)
    edge_peak = max(float(np.max(edge_psf)), 1.0e-12)
    center_psf_center_row = _channel_normalize(center_psf[center_psf.shape[0] // 2, :])
    edge_psf_center_row = _channel_normalize(edge_psf[edge_psf.shape[0] // 2, :])
    return {
        "field_heights_mm": field_heights_mm,
        "edge_field_height_mm": edge_field_height_mm,
        "support_x_um": support_x_um,
        "support_y_um": support_y_um,
        "sample_spacing_um": sample_spacing_um,
        "center_psf_norm": center_psf / center_peak,
        "edge_psf_norm": edge_psf / edge_peak,
        "center_psf_center_row_norm": center_psf_center_row,
        "edge_psf_center_row_norm": edge_psf_center_row,
        "center_psf_metrics": _spot_metrics(center_psf, spacing_um=sample_spacing_um),
        "edge_psf_metrics": _spot_metrics(edge_psf, spacing_um=sample_spacing_um),
    }


def _reflectance_sample_statistics(reflectances: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(reflectances, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Reflectance statistics expect a 2D wavelength-by-sample matrix.")
    norms = np.sqrt(np.maximum(np.sum(np.square(matrix), axis=0, dtype=float), 1e-12))
    normalized = matrix / norms.reshape(1, -1)
    mean_reflectance = np.mean(normalized, axis=1, dtype=float)
    singular_values = np.linalg.svd(normalized - mean_reflectance.reshape(-1, 1), compute_uv=False)
    return mean_reflectance, singular_values


def _canonicalize_basis_columns(basis: Any) -> np.ndarray:
    matrix = np.asarray(basis, dtype=float).copy()
    if matrix.ndim != 2:
        raise ValueError("Basis canonicalization expects a 2D basis matrix.")
    for column in range(matrix.shape[1]):
        anchor = int(np.argmax(np.abs(matrix[:, column])))
        if matrix[anchor, column] < 0.0:
            matrix[:, column] = -matrix[:, column]
    return matrix


def _unit_length(values: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def _find_nearest_two(array: Any, number: float) -> np.ndarray:
    values = np.asarray(array, dtype=float).reshape(-1)
    order = np.argsort(np.abs(values - float(number)), kind="mergesort")
    return values[order[:2]]


def _circle_mask(radius: float, img_size: tuple[int, int]) -> np.ndarray:
    rows, cols = int(img_size[0]), int(img_size[1])
    x, y = np.meshgrid(np.arange(1, cols + 1, dtype=float), np.arange(1, rows + 1, dtype=float))
    center_x = cols / 2.0
    center_y = rows / 2.0
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return dist <= float(radius)


def _generate_fringe_psf(zernike_coeffs: Any, *, grid_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    coeffs = np.asarray(zernike_coeffs, dtype=float).reshape(-1)
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, grid_size, dtype=float), np.linspace(-1.0, 1.0, grid_size, dtype=float))
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    zernike_terms = [
        lambda r, t: np.ones_like(r, dtype=float),
        lambda r, t: r * np.cos(t),
        lambda r, t: r * np.sin(t),
        lambda r, t: -1.0 + 2.0 * r**2,
        lambda r, t: r**2 * np.cos(2.0 * t),
        lambda r, t: r**2 * np.sin(2.0 * t),
        lambda r, t: (-2.0 * r + 3.0 * r**3) * np.cos(t),
        lambda r, t: (-2.0 * r + 3.0 * r**3) * np.sin(t),
        lambda r, t: 1.0 - 6.0 * r**2 + 6.0 * r**4,
        lambda r, t: r**3 * np.cos(3.0 * t),
        lambda r, t: r**3 * np.sin(3.0 * t),
        lambda r, t: (-3.0 * r**2 + 4.0 * r**4) * np.cos(2.0 * t),
        lambda r, t: (-3.0 * r**2 + 4.0 * r**4) * np.sin(2.0 * t),
        lambda r, t: (3.0 * r - 12.0 * r**3 + 10.0 * r**5) * np.cos(t),
        lambda r, t: (3.0 * r - 12.0 * r**3 + 10.0 * r**5) * np.sin(t),
    ]

    wavefront = np.zeros_like(rho, dtype=float)
    for index, coefficient in enumerate(coeffs):
        wavefront = wavefront + float(coefficient) * zernike_terms[index](rho, theta)

    aperture_mask = _circle_mask(round(grid_size / 2.0), (grid_size, grid_size)).astype(float)
    pupil_phase = np.exp(-1j * 2.0 * np.pi * wavefront)
    amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_phase * aperture_mask)))
    intensity = np.real(amplitude * np.conj(amplitude))
    return intensity, wavefront


def run_python_case_with_context(
    case_name: str,
    *,
    asset_store: AssetStore | None = None,
) -> ParityCaseResult:
    store = asset_store or AssetStore.default()

    if case_name == "scene_macbeth_default":
        scene = scene_create(asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_macbeth_tungsten":
        scene = scene_create("macbethtungsten", asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_checkerboard_small":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_bb_small":
        scene = scene_create("uniform bb", 16, 4500, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniformbb_small":
        scene = scene_create("uniformbb", 16, 4500, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniformblackbody_small":
        scene = scene_create("uniformblackbody", 16, 4500, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_monochromatic_small":
        scene = scene_create("uniform monochromatic", 550, 12, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_d65_small":
        scene = scene_create("uniform d65", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_empty_small":
        scene = scene_create("empty", asset_store=store)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "illuminant_energy": scene_get(scene, "illuminant energy"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
                "photon_sum": float(np.sum(photons)),
                "photon_max": float(np.max(photons)) if photons.size else 0.0,
            },
            context={"scene": scene},
        )

    if case_name == "scene_lstar_small":
        scene = scene_create("lstar", np.array([80, 10], dtype=int), 20, 1, asset_store=store)
        luminance = np.asarray(scene_get(scene, "luminance", asset_store=store), dtype=float)
        bar_means = np.array(
            [np.mean(luminance[:, start : start + 10]) for start in range(0, luminance.shape[1], 10)],
            dtype=float,
        )
        center_row = np.asarray(luminance[luminance.shape[0] // 2, :], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "bar_means_norm": bar_means / max(float(np.max(bar_means)), 1.0e-12),
                "center_row_norm": center_row / max(float(np.max(center_row)), 1.0e-12),
            },
            context={"scene": scene},
        )

    if case_name == "scene_hdr_small":
        scene = scene_create("hdr", asset_store=store)
        luminance = np.asarray(scene_get(scene, "luminance", asset_store=store), dtype=float)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        image_size = luminance.shape[0]
        row_indices = np.array(
            [
                int(np.rint(0.25 * image_size)),
                int(np.rint(0.50 * image_size)) - 1,
                int(np.rint(0.75 * image_size)) - 1,
            ],
            dtype=int,
        )
        row_summaries = np.vstack(
            [
                np.array(
                    [
                        float(np.mean(row := _channel_normalize(luminance[row_index, :]))),
                        float(np.std(row)),
                        float(np.percentile(row, 95.0)),
                        float(np.count_nonzero(row > 0.50)),
                        float(np.count_nonzero(row > 0.10)),
                        float(np.count_nonzero(row > 0.01)),
                    ],
                    dtype=float,
                )
                for row_index in row_indices
            ]
        )
        mean_spd = np.mean(photons, axis=(0, 1), dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
                "mean_spd_norm": _channel_normalize(mean_spd),
                "row_profile_summaries": row_summaries,
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_ep_small":
        scene = scene_create("uniform ep", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniformephoton_small":
        scene = scene_create("uniformephoton", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniformep_small":
        scene = scene_create("uniformep", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniformequalphoton_small":
        scene = scene_create("uniformequalphoton", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniformequalphotons_small":
        scene = scene_create("uniformequalphotons", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_ee_small":
        scene = scene_create("uniform", 24, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_uniform_ee_specify_small":
        wave = np.arange(380.0, 721.0, 10.0, dtype=float)
        scene = scene_create("uniformEESpecify", 128, wave, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_exponential_intensity_ramp_small":
        scene = scene_create("exponential intensity ramp", 64, 256, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_linear_intensity_ramp_small":
        scene = scene_create("linear intensity ramp", 64, 256, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_rings_rays_small":
        scene = scene_create("rings rays", 8, 64, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_point_array_small":
        scene = scene_create("point array", 64, 16, "ep", 1, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_grid_lines_small":
        scene = scene_create("grid lines", 64, 16, "ep", 1, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_white_noise_small":
        scene = scene_create("white noise", 128, 20, asset_store=store)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        plane = photons[:, :, 0]
        plane_mean = max(float(np.mean(plane)), np.finfo(float).eps)
        normalized_plane = plane / plane_mean
        mean_spectrum = np.mean(photons, axis=(0, 1), dtype=float)
        mean_spectrum = mean_spectrum / max(float(np.max(mean_spectrum)), np.finfo(float).eps)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons_shape": np.asarray(photons.shape, dtype=int),
                "fov_deg": float(scene_get(scene, "fov")),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
                "pattern_stats_norm": np.asarray(
                    [np.min(normalized_plane), np.std(normalized_plane), np.max(normalized_plane)],
                    dtype=float,
                ),
                "pattern_percentiles_norm": np.asarray(
                    np.percentile(normalized_plane, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]),
                    dtype=float,
                ),
                "mean_spectrum_norm": np.asarray(mean_spectrum, dtype=float),
            },
            context={"scene": scene},
        )

    if case_name == "scene_disk_array_small":
        scene = scene_create("disk array", 64, 8, np.array([2, 2], dtype=int), asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_square_array_small":
        scene = scene_create("square array", 64, 8, np.array([2, 2], dtype=int), asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_frequency_orientation_small":
        params = {
            "angles": np.linspace(0.0, np.pi / 2.0, 4),
            "freqs": np.array([1.0, 2.0, 4.0, 8.0], dtype=float),
            "blockSize": 16,
            "contrast": 0.8,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_harmonic_small":
        params = {
            "freq": np.array([1.0, 5.0], dtype=float),
            "contrast": np.array([0.2, 0.6], dtype=float),
            "ph": np.array([0.0, np.pi / 3.0], dtype=float),
            "ang": np.array([0.0, 0.0], dtype=float),
            "row": 64,
            "col": 64,
            "GaborFlag": 0.2,
        }
        scene = scene_create("harmonic", params, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_sinusoid_small":
        params = {
            "freq": np.array([1.0, 5.0], dtype=float),
            "contrast": np.array([0.2, 0.6], dtype=float),
            "ph": np.array([0.0, np.pi / 3.0], dtype=float),
            "ang": np.array([0.0, 0.0], dtype=float),
            "row": 64,
            "col": 64,
            "GaborFlag": 0.2,
        }
        scene = scene_create("sinusoid", params, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_sweep_frequency_small":
        scene = scene_create("sweep frequency", 64, 12, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_reflectance_chart_small":
        scene = scene_create(
            "reflectance chart",
            8,
            [[1, 2], [1, 2], [1]],
            [
                store.resolve("data/surfaces/reflectances/MunsellSamples_Vhrel.mat"),
                store.resolve("data/surfaces/reflectances/Food_Vhrel.mat"),
                store.resolve("data/surfaces/reflectances/skin/HyspexSkinReflectance.mat"),
            ],
            None,
            True,
            "without replacement",
            asset_store=store,
        )
        chart_parameters = scene_get(scene, "chart parameters")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
                "chart_rowcol": np.asarray(chart_parameters["rowcol"], dtype=int),
                "chart_index_map": np.asarray(chart_parameters["rIdxMap"], dtype=int),
            },
            context={"scene": scene},
        )

    if case_name == "scene_star_pattern_small":
        scene = scene_create("star pattern", 64, "ee", 6, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_zone_plate_small":
        scene = scene_create("zone plate", 96, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_dead_leaves_small":
        options = {
            "nbr_iter": 1500,
            "shape": "disk",
            "random_samples": _dead_leaves_sample_matrix(1500, 4, seed=12345),
        }
        scene = scene_create("dead leaves", 96, 3.0, options, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_bar_small":
        scene = scene_create("bar", 64, 3, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_line_ee_small":
        scene = scene_create("line ee", [64, 64], 2, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_lineee_small":
        scene = scene_create("lineee", [64, 64], 2, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_line_ep_small":
        scene = scene_create("line ep", [64, 64], 2, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_lineequalphoton_small":
        scene = scene_create("lineequalphoton", [64, 64], 2, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_line_d65_small":
        scene = scene_create("lined65", [64, 64], asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_line_small":
        scene = scene_create("line", [64, 64], asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "scene_impulse1dd65_small":
        scene = scene_create("impulse1dd65", [64, 64], asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": scene_get(scene, "wave"),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "photons": scene_get(scene, "photons"),
                "mean_luminance": scene_get(scene, "mean luminance", asset_store=store),
            },
            context={"scene": scene},
        )

    if case_name == "utility_unit_frequency_list":
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "even": unit_frequency_list(50),
                "odd": unit_frequency_list(51),
            },
            context={},
        )

    if case_name == "utility_energy_quanta_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        energy = np.linspace(0.1, 3.1, wave.size, dtype=float)
        photons = energy_to_quanta(energy, wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "photons": photons,
                "energy_roundtrip": quanta_to_energy(photons, wave),
            },
            context={},
        )

    if case_name == "utility_energy_quanta_matrix":
        wave = np.array([400.0, 500.0, 600.0], dtype=float)
        energy = np.array(
            [
                [0.2, 0.4],
                [0.5, 0.7],
                [0.8, 1.0],
            ],
            dtype=float,
        )
        photons = energy_to_quanta(energy, wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "photons": photons,
                "energy_roundtrip": quanta_to_energy(photons, wave),
            },
            context={},
        )

    if case_name == "utility_blackbody_energy_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        temperatures = np.array([3000.0, 5000.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "temperatures": temperatures,
                "energy": blackbody(wave, temperatures, kind="energy"),
            },
            context={},
        )

    if case_name == "utility_blackbody_quanta_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        temperatures = np.array([3000.0, 5000.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "temperatures": temperatures,
                "photons": blackbody(wave, temperatures, kind="quanta"),
            },
            context={},
        )

    if case_name == "utility_ie_param_format_string":
        original = "Exposure Time"
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "original": original,
                "formatted": param_format(original),
            },
            context={},
        )

    if case_name == "wvf_load_thibos_virtual_eyes_small":
        sample_mean, sample_cov, subject_coeffs = wvf_load_thibos_virtual_eyes(6.0, asset_store=store, full=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pupil_diameter_mm": 6.0,
                "sample_mean": sample_mean,
                "sample_cov": sample_cov,
                "left_eye": np.asarray(subject_coeffs["left_eye"], dtype=float),
                "right_eye": np.asarray(subject_coeffs["right_eye"], dtype=float),
                "both_eyes": np.asarray(subject_coeffs["both_eyes"], dtype=float),
            },
            context={},
        )

    if case_name == "wvf_thibos_model_small":
        measured_pupil_mm = 4.5
        calc_pupil_mm = 3.0
        measured_wavelength_nm = 550.0
        calc_waves = np.arange(450.0, 651.0, 100.0, dtype=float)
        sample_mean, sample_cov, _ = wvf_load_thibos_virtual_eyes(measured_pupil_mm, asset_store=store, full=True)
        example_count = 10
        example_subject_indices = np.arange(0, example_count, 3, dtype=int)
        standard_normal = _deterministic_normal_samples(example_count, sample_mean.size)
        example_coeffs = ie_mvnrnd(sample_mean, sample_cov, standard_normal_samples=standard_normal)

        sample_mean_zcoeffs = np.zeros(65, dtype=float)
        sample_mean_zcoeffs[:13] = np.asarray(sample_mean[:13], dtype=float)

        this_guy = wvf_create()
        this_guy = wvf_set(this_guy, "zcoeffs", sample_mean_zcoeffs)
        this_guy = wvf_set(this_guy, "measured pupil", measured_pupil_mm)
        this_guy = wvf_set(this_guy, "calculated pupil", calc_pupil_mm)
        this_guy = wvf_set(this_guy, "measured wavelength", measured_wavelength_nm)
        this_guy = wvf_set(this_guy, "calc wave", calc_waves)
        this_guy = wvf_compute(this_guy)

        mean_subject_mid_rows: list[np.ndarray] = []
        mean_subject_peaks: list[float] = []
        for wavelength_nm in calc_waves:
            psf = np.asarray(wvf_get(this_guy, "psf", float(wavelength_nm)), dtype=float)
            mean_subject_mid_rows.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))
            mean_subject_peaks.append(float(np.max(psf)))

        subject = wvf_create()
        subject = wvf_set(subject, "measured pupil", measured_pupil_mm)
        subject = wvf_set(subject, "calculated pupil", calc_pupil_mm)
        subject = wvf_set(subject, "measured wavelength", measured_wavelength_nm)

        subject_rows_450: list[np.ndarray] = []
        subject_rows_550: list[np.ndarray] = []
        subject_peaks_450: list[float] = []
        subject_peaks_550: list[float] = []
        selected_coeffs: list[np.ndarray] = []
        for subject_index in example_subject_indices:
            subject_zcoeffs = np.zeros(65, dtype=float)
            subject_zcoeffs[:13] = np.asarray(example_coeffs[subject_index, :13], dtype=float)
            selected_coeffs.append(np.asarray(subject_zcoeffs[:13], dtype=float))
            subject = wvf_set(subject, "zcoeffs", subject_zcoeffs)

            subject = wvf_set(subject, "calc wave", 450.0)
            subject = wvf_compute(subject)
            psf_450 = np.asarray(wvf_get(subject, "psf", 450.0), dtype=float)
            subject_rows_450.append(np.asarray(psf_450[psf_450.shape[0] // 2, :], dtype=float))
            subject_peaks_450.append(float(np.max(psf_450)))

            subject = wvf_set(subject, "calc wave", 550.0)
            subject = wvf_compute(subject)
            psf_550 = np.asarray(wvf_get(subject, "psf", 550.0), dtype=float)
            subject_rows_550.append(np.asarray(psf_550[psf_550.shape[0] // 2, :], dtype=float))
            subject_peaks_550.append(float(np.max(psf_550)))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "measured_pupil_mm": measured_pupil_mm,
                "calc_pupil_mm": calc_pupil_mm,
                "measured_wavelength_nm": measured_wavelength_nm,
                "calc_waves_nm": calc_waves,
                "example_coeffs": np.asarray(example_coeffs, dtype=float),
                "mean_subject_psf_mid_rows": np.vstack(mean_subject_mid_rows),
                "mean_subject_psf_peaks": np.asarray(mean_subject_peaks, dtype=float),
                "example_subject_indices": example_subject_indices + 1,
                "example_subject_coeffs": np.vstack(selected_coeffs),
                "example_subject_psf_mid_rows_450": np.vstack(subject_rows_450),
                "example_subject_psf_mid_rows_550": np.vstack(subject_rows_550),
                "example_subject_psf_peaks_450": np.asarray(subject_peaks_450, dtype=float),
                "example_subject_psf_peaks_550": np.asarray(subject_peaks_550, dtype=float),
            },
            context={"wvf": this_guy},
        )

    if case_name == "wvf_pupil_size_human_small":
        measured_pupil_mm = 7.5
        calc_pupil_mm = 3.0
        wavelength_nm = 520.0
        zcoeffs = wvf_load_thibos_virtual_eyes(measured_pupil_mm, asset_store=store)
        wvf = wvf_create(
            "calc wavelengths",
            np.array([wavelength_nm], dtype=float),
            "zcoeffs",
            zcoeffs,
            "measured pupil size",
            measured_pupil_mm,
            "calc pupil size",
            calc_pupil_mm,
            "name",
            "7-pupil",
        )
        wvf = wvf_set(wvf, "lcaMethod", "human")
        wvf = wvf_compute(wvf)
        psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
        middle_row = psf.shape[0] // 2
        measured_wavelength_nm = float(wvf_get(wvf, "measured wavelength", "nm"))
        lca_diopters = 1.8859 - (1.8859 - (0.63346 / (0.001 * measured_wavelength_nm - 0.2141))) - (
            0.63346 / (0.001 * wavelength_nm - 0.2141)
        )
        lca_microns = np.asarray(
            wvf_defocus_diopters_to_microns(-lca_diopters, measured_pupil_mm),
            dtype=float,
        ).reshape(-1)[0]
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "measured_pupil_mm": measured_pupil_mm,
                "calc_pupil_mm": calc_pupil_mm,
                "measured_wavelength_nm": measured_wavelength_nm,
                "wave": np.asarray(wvf_get(wvf, "wave"), dtype=float),
                "f_number": float(wvf_get(wvf, "fnumber")),
                "lca_diopters": float(lca_diopters),
                "lca_microns": float(lca_microns),
                "psf_sum": float(np.sum(psf)),
                "psf_mid_row": np.asarray(psf[middle_row, :], dtype=float),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_pupil_size_measured_compare_small":
        measured_pupils_mm = np.array([7.5, 6.0, 4.5, 3.0], dtype=float)
        calc_pupil_mm = 3.0
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        wavelength_nm = 550.0
        psf_mid_rows: list[np.ndarray] = []
        psf_peaks: list[float] = []
        psf_sums: list[float] = []
        max_abs_diffs: list[float] = []
        reference_psf: np.ndarray | None = None

        for measured_pupil_mm in measured_pupils_mm:
            zcoeffs = wvf_load_thibos_virtual_eyes(float(measured_pupil_mm), asset_store=store)
            wvf = wvf_create(
                "calc wavelengths",
                wave,
                "zcoeffs",
                zcoeffs,
                "measured pupil size",
                float(measured_pupil_mm),
                "calc pupil size",
                calc_pupil_mm,
                "name",
                f"{measured_pupil_mm:g}-pupil",
            )
            wvf = wvf_set(wvf, "lcaMethod", "human")
            wvf = wvf_compute(wvf)
            psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
            middle_row = psf.shape[0] // 2
            psf_mid_rows.append(np.asarray(psf[middle_row, :], dtype=float))
            psf_peaks.append(float(np.max(psf)))
            psf_sums.append(float(np.sum(psf)))
            if reference_psf is None:
                reference_psf = psf
                max_abs_diffs.append(0.0)
            else:
                max_abs_diffs.append(float(np.max(np.abs(reference_psf - psf))))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "measured_pupil_mm": measured_pupils_mm,
                "calc_pupil_mm": calc_pupil_mm,
                "wave": wave,
                "wavelength_nm": wavelength_nm,
                "psf_sum": np.asarray(psf_sums, dtype=float),
                "psf_peak_550": np.asarray(psf_peaks, dtype=float),
                "max_abs_diff_vs_first_550": np.asarray(max_abs_diffs, dtype=float),
                "psf_mid_row_550": np.asarray(psf_mid_rows, dtype=float),
            },
            context={},
        )

    if case_name == "wvf_psf_spacing_small":
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

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wavelength_nm": wavelength_nm,
                "focal_length_mm": focal_length_mm,
                "calc_pupil_diameter_mm": float(wvf_get(wvf, "calc pupil size", "mm")),
                "npixels": int(wvf_get(wvf, "npixels")),
                "field_size_mm": float(wvf_get(wvf, "field size mm", "mm")),
                "pupil_sample_spacing_mm": float(wvf_get(wvf, "pupil sample spacing", "mm", wavelength_nm)),
                "psf_sample_spacing_arcmin": float(wvf_get(wvf, "psf sample spacing")),
                "ref_psf_sample_interval_arcmin": float(wvf_get(wvf, "ref psf sample interval")),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_osa_index_conversion_small":
        indices = np.asarray([0, 1, 2, 5, 15, 20, 35], dtype=int)
        n, m = wvf_osa_index_to_zernike_nm(indices)
        roundtrip = np.asarray(wvf_zernike_nm_to_osa_index(n, m), dtype=int)
        scalar_n, scalar_m = wvf_osa_index_to_zernike_nm(15)
        scalar_roundtrip = int(wvf_zernike_nm_to_osa_index(scalar_n, scalar_m))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "indices": indices,
                "n": np.asarray(n, dtype=int),
                "m": np.asarray(m, dtype=int),
                "roundtrip_indices": roundtrip,
                "scalar_index": 15,
                "scalar_n": int(scalar_n),
                "scalar_m": int(scalar_m),
                "scalar_roundtrip_index": scalar_roundtrip,
            },
            context={},
        )

    if case_name == "metrics_xyz_from_energy_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        energy = np.linspace(0.05, 1.55, wave.size, dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "energy": energy,
                "xyz": xyz_from_energy(energy, wave, asset_store=store),
            },
            context={},
        )

    if case_name == "metrics_xyz_to_luv_1d":
        xyz = np.array([20.0, 30.0, 15.0], dtype=float)
        white_point = np.array([95.047, 100.0, 108.883], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz": xyz,
                "white_point": white_point,
                "luv": xyz_to_luv(xyz, white_point),
            },
            context={},
        )

    if case_name == "metrics_xyz_to_lab_1d":
        xyz = np.array([20.0, 30.0, 15.0], dtype=float)
        white_point = np.array([95.047, 100.0, 108.883], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz": xyz,
                "white_point": white_point,
                "lab": xyz_to_lab(xyz, white_point),
            },
            context={},
        )

    if case_name == "metrics_xyz_to_uv_1d":
        xyz = np.array([20.0, 30.0, 15.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz": xyz,
                "uv": xyz_to_uv(xyz),
            },
            context={},
        )

    if case_name == "optics_airy_disk_small":
        radius_um, image = airy_disk(550.0, 3.0, "units", "um", return_image=True)
        assert image is not None
        image_data = np.asarray(image["data"], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "radius_um": radius_um,
                "diameter_um": airy_disk(550.0, 3.0, "units", "um", "diameter", True),
                "radius_mm": airy_disk(550.0, 3.0, "units", "mm"),
                "radius_deg": airy_disk(700.0, None, "units", "deg", "pupil diameter", 1e-3),
                "radius_rad": airy_disk(700.0, None, "units", "rad", "pupil diameter", 1e-3),
                "image_rows": int(image_data.shape[0]),
                "image_cols": int(image_data.shape[1]),
            },
            context={},
        )

    if case_name == "optics_coc_small":
        base_optics = dict(oi_create(asset_store=store).fields["optics"])
        base_optics["focal_length_m"] = 0.050
        optics_f2 = dict(base_optics)
        optics_f2["f_number"] = 2.0
        optics_f8 = dict(base_optics)
        optics_f8["f_number"] = 8.0
        circ_f2_focus_0_5, x_dist_focus_0_5 = optics_coc(optics_f2, 0.5, "unit", "mm", "n samples", 50)
        circ_f8_focus_0_5, _ = optics_coc(optics_f8, 0.5, "unit", "mm", "n samples", 50)
        circ_f2_focus_3, x_dist_focus_3 = optics_coc(optics_f2, 3.0, "unit", "mm", "n samples", 50)
        circ_f8_focus_3, _ = optics_coc(optics_f8, 3.0, "unit", "mm", "n samples", 50)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "object_distances_m": np.array([0.5, 3.0], dtype=float),
                "f_numbers": np.array([2.0, 8.0], dtype=float),
                "focal_length_m": 0.050,
                "x_dist_focus_0_5_m": np.asarray(x_dist_focus_0_5, dtype=float),
                "circ_f2_focus_0_5_mm": np.asarray(circ_f2_focus_0_5, dtype=float),
                "circ_f8_focus_0_5_mm": np.asarray(circ_f8_focus_0_5, dtype=float),
                "x_dist_focus_3_m": np.asarray(x_dist_focus_3, dtype=float),
                "circ_f2_focus_3_mm": np.asarray(circ_f2_focus_3, dtype=float),
                "circ_f8_focus_3_mm": np.asarray(circ_f8_focus_3, dtype=float),
            },
            context={"optics_f2": optics_f2, "optics_f8": optics_f8},
        )

    if case_name == "metrics_cct_from_uv_1d":
        uv = np.array([0.20029948, 0.31055768], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "uv": uv,
                "cct_k": cct_from_uv(uv, asset_store=store),
            },
            context={},
        )

    if case_name == "metrics_delta_e_ab_1976_1d":
        xyz1 = np.array([20.0, 30.0, 15.0], dtype=float)
        xyz2 = np.array([18.0, 27.0, 16.5], dtype=float)
        white_point = np.array([95.047, 100.0, 108.883], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz1": xyz1,
                "xyz2": xyz2,
                "white_point": white_point,
                "delta_e": delta_e_ab(xyz1, xyz2, white_point),
            },
            context={},
        )

    if case_name == "metrics_spd_angle_1d":
        wave = np.array([500.0, 510.0, 520.0], dtype=float)
        spd1 = np.array([1.0, 0.0, 0.0], dtype=float)
        spd2 = np.array([0.0, 1.0, 0.0], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "spd1": spd1,
                "spd2": spd2,
                "angle": metrics_spd(spd1, spd2, metric="angle", wave=wave),
            },
            context={},
        )

    if case_name == "metrics_spd_cielab_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        spd1 = np.linspace(0.5, 1.7, wave.size, dtype=float)
        spd2 = np.linspace(1.6, 0.4, wave.size, dtype=float)
        value, params = metrics_spd(spd1, spd2, metric="cielab", wave=wave, return_params=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "spd1": spd1,
                "spd2": spd2,
                "delta_e": value,
                "xyz1": np.asarray(params["xyz1"], dtype=float),
                "xyz2": np.asarray(params["xyz2"], dtype=float),
                "lab1": np.asarray(params["lab1"], dtype=float),
                "lab2": np.asarray(params["lab2"], dtype=float),
                "white_point": np.asarray(params["white_point"], dtype=float),
            },
            context={},
        )

    if case_name == "metrics_spd_mired_1d":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        spd1 = np.asarray(blackbody(wave, 6500.0, kind="energy"), dtype=float)
        spd2 = np.asarray(blackbody(wave, 5000.0, kind="energy"), dtype=float)
        value, params = metrics_spd(spd1, spd2, metric="mired", wave=wave, asset_store=store, return_params=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "spd1": spd1,
                "spd2": spd2,
                "mired": value,
                "uv": np.asarray(params["uv"], dtype=float),
                "cct_k": np.asarray(params["cct_k"], dtype=float),
            },
            context={},
        )

    if case_name == "metrics_spd_daylight_sweep_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        ctemp = np.arange(4000.0, 7000.0 + 1.0, 500.0, dtype=float)
        d65_white_point = np.array([94.9409, 100.0, 108.6656], dtype=float)

        d4000_angle = np.zeros(ctemp.size, dtype=float)
        d4000_delta_e = np.zeros(ctemp.size, dtype=float)
        d4000_mired = np.zeros(ctemp.size, dtype=float)
        standard_4000 = np.asarray(daylight(wave, 4000.0), dtype=float)
        for index, color_temperature in enumerate(ctemp):
            comparison = np.asarray(daylight(wave, float(color_temperature)), dtype=float)
            d4000_angle[index] = float(metrics_spd(standard_4000, comparison, metric="angle", wave=wave))
            d4000_delta_e[index] = float(metrics_spd(standard_4000, comparison, metric="cielab", wave=wave))
            d4000_mired[index] = float(metrics_spd(standard_4000, comparison, metric="mired", wave=wave, asset_store=store))

        d6500_angle = np.zeros(ctemp.size, dtype=float)
        d6500_delta_e = np.zeros(ctemp.size, dtype=float)
        d6500_mired = np.zeros(ctemp.size, dtype=float)
        standard_6500 = np.asarray(daylight(wave, 6500.0), dtype=float)
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
            d6500_mired[index] = float(metrics_spd(standard_6500, comparison, metric="mired", wave=wave, asset_store=store))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "ctemp_k": ctemp,
                "d65_white_point": d65_white_point,
                "d4000_angle": d4000_angle,
                "d4000_delta_e": d4000_delta_e,
                "d4000_mired": d4000_mired,
                "d6500_angle": d6500_angle,
                "d6500_delta_e": d6500_delta_e,
                "d6500_mired": d6500_mired,
            },
            context={},
        )

    if case_name == "metrics_vsnr_small":
        levels = np.asarray(np.logspace(1.5, 3.0, 3), dtype=float)
        result = camera_vsnr(camera_create(asset_store=store), levels, asset_store=store)

        vsnr = np.asarray(result.vSNR, dtype=float).reshape(-1)
        finite = np.isfinite(vsnr)
        delta_e = np.full(vsnr.shape, np.nan, dtype=float)
        delta_e[finite] = 1.0 / np.maximum(vsnr[finite], 1.0e-12)
        ip_channel_means = np.vstack(
            [
                _channel_normalize(np.mean(np.asarray(ip_get(ip, "result"), dtype=float).reshape(-1, 3), axis=0, dtype=float))
                for ip in result.ip
            ]
        )
        scale = float(vsnr[np.flatnonzero(finite)[0]]) if np.any(finite) else 1.0
        delta_scale = float(delta_e[np.flatnonzero(finite)[0]]) if np.any(finite) else 1.0

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "light_levels": np.asarray(result.lightLevels, dtype=float),
                "rect": np.asarray(result.rect, dtype=int),
                "saturation_mask": np.asarray(~finite, dtype=bool),
                "vsnr_norm": np.asarray(vsnr / max(scale, 1.0e-12), dtype=float),
                "delta_e_norm": np.asarray(delta_e / max(delta_scale, 1.0e-12), dtype=float),
                "result_channel_means_norm": np.asarray(ip_channel_means, dtype=float),
            },
            context={},
        )

    if case_name == "metrics_sqri_tutorial_small":
        n_sf = 5000
        sf = np.logspace(-1.5, 1.6, n_sf, dtype=float)
        perfect_dmtf = np.ones_like(sf, dtype=float)
        luminance = 340.0 / np.pi
        widths = np.array([0.5, 1.0, 2.3, 6.5, 60.0], dtype=float)

        max_sqri_by_width = np.zeros(widths.size, dtype=float)
        csf_peak_by_width = np.zeros(widths.size, dtype=float)
        for index, width in enumerate(widths):
            max_sqri_by_width[index], csf = ie_sqri(sf, perfect_dmtf, luminance, "width", width)
            csf_peak_by_width[index] = float(np.max(np.asarray(csf, dtype=float)))

        luminance_levels = np.logspace(-4.0, 1.0, 6, dtype=float)
        max_sqri_by_luminance = np.zeros(luminance_levels.size, dtype=float)
        for index, current_luminance in enumerate(luminance_levels):
            max_sqri_by_luminance[index], _ = ie_sqri(sf, perfect_dmtf, current_luminance, "width", 14.0)

        def _display_gaussian_mtf(display_name: str, viewing_distance_m: float) -> tuple[Any, np.ndarray, np.ndarray, float]:
            display = display_create(display_name, asset_store=store)
            display = display_set(display, "viewing distance", viewing_distance_m)
            sigma_um = 0.5 * float(display_get(display, "meters per dot")) * 1e6
            x = np.arange(-500.0, 500.0, 1.0, dtype=float)
            gaussian = np.exp(-((x / max(sigma_um, 1.0e-12)) ** 2))
            gaussian /= max(float(np.sum(gaussian, dtype=float)), 1.0e-12)
            d_mtf_fft = np.fft.fftshift(np.abs(np.fft.fft(gaussian)))
            mm_per_deg = float(display_get(display, "dots per deg")) * float(display_get(display, "meters per dot")) * 1e3
            fcpd = x * mm_per_deg
            return display, fcpd, np.asarray(d_mtf_fft, dtype=float), sigma_um

        selected_sf = np.array([0.1, 1.0, 5.0, 10.0, 20.0], dtype=float)
        note3_distances = np.array([0.2, 0.4, 0.8], dtype=float)
        note3_mm_per_deg = np.zeros(note3_distances.size, dtype=float)
        note3_dmtf_selected = np.zeros((note3_distances.size, selected_sf.size), dtype=float)
        note3_sigma_um = 0.0
        for index, distance in enumerate(note3_distances):
            note3_display, note3_fcpd, note3_dmtf_fft, note3_sigma_um = _display_gaussian_mtf("OLED-Samsung-Note3", float(distance))
            note3_mm_per_deg[index] = float(display_get(note3_display, "dots per deg")) * float(display_get(note3_display, "meters per dot")) * 1e3
            note3_dmtf_selected[index, :] = np.interp(selected_sf, note3_fcpd, note3_dmtf_fft)

        crt_display, crt_fcpd, crt_dmtf_fft, _ = _display_gaussian_mtf("CRT-HP", 1.0)
        crt_dmtf_interp = np.interp(sf, crt_fcpd, crt_dmtf_fft, left=np.nan, right=np.nan)
        crt_sqri, crt_csf = ie_sqri(sf, crt_dmtf_interp, 100.0, "width", 14.0)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sf": sf,
                "widths": widths,
                "max_sqri_by_width": max_sqri_by_width,
                "csf_peak_by_width": csf_peak_by_width,
                "luminance_levels": luminance_levels,
                "max_sqri_by_luminance": max_sqri_by_luminance,
                "note3_distances_m": note3_distances,
                "note3_sigma_um": float(note3_sigma_um),
                "note3_mm_per_deg_by_distance": note3_mm_per_deg,
                "selected_sf": selected_sf,
                "note3_dmtf_selected": note3_dmtf_selected,
                "crt_dpi": float(display_get(crt_display, "dpi")),
                "crt_mm_per_deg": float(display_get(crt_display, "dots per deg")) * float(display_get(crt_display, "meters per dot")) * 1e3,
                "crt_sqri": float(crt_sqri),
                "crt_csf_peak": float(np.max(np.asarray(crt_csf, dtype=float))),
                "crt_dmtf_selected": np.interp(selected_sf, crt_fcpd, crt_dmtf_fft),
            },
            context={},
        )

    if case_name == "metrics_scielab_rgb_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        error_image, scene1, scene2, display = scielab_rgb(
            "hats.jpg",
            "hatsC.jpg",
            "LCD-Apple.mat",
            0.3,
            asset_store=store,
        )
        center_row = np.asarray(error_image[error_image.shape[0] // 2, :], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "error_size": np.asarray(error_image.shape, dtype=int),
                "scene1_size": np.asarray(scene_get(scene1, "size"), dtype=int),
                "scene2_size": np.asarray(scene_get(scene2, "size"), dtype=int),
                "fov_deg": float(scene_get(scene1, "fov")),
                "display_white_point": np.asarray(display_get(display, "white point"), dtype=float).reshape(3),
                "scene1_mean_luminance": float(scene_get(scene1, "mean luminance", asset_store=store)),
                "scene2_mean_luminance": float(scene_get(scene2, "mean luminance", asset_store=store)),
                "error_stats": np.array(
                    [
                        float(np.mean(error_image, dtype=float)),
                        float(np.median(error_image)),
                        float(np.percentile(error_image, 95.0)),
                        float(np.max(error_image)),
                    ],
                    dtype=float,
                ),
                "error_center_row_norm": _canonical_profile(_channel_normalize(center_row)),
            },
            context={"scene1": scene1, "scene2": scene2, "display": display},
        )

    if case_name == "metrics_rgb2scielab_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        error_image, scene1, scene2, display = scielab_rgb(
            "hats.jpg",
            "hatsC.jpg",
            "crt.mat",
            0.3,
            asset_store=store,
        )
        error_array = np.asarray(error_image, dtype=float)
        mask = error_array > 2.0
        mean_above2 = float(np.mean(error_array[mask], dtype=float)) if np.any(mask) else 0.0
        percent_above2 = float(np.count_nonzero(mask)) / max(float(error_array.size), 1.0) * 100.0
        center_row = np.asarray(error_array[error_array.shape[0] // 2, :], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "error_size": np.asarray(error_array.shape, dtype=int),
                "scene1_size": np.asarray(scene_get(scene1, "size"), dtype=int),
                "scene2_size": np.asarray(scene_get(scene2, "size"), dtype=int),
                "fov_deg": float(scene_get(scene1, "fov")),
                "display_white_point": np.asarray(display_get(display, "white point"), dtype=float).reshape(3),
                "scene1_mean_luminance": float(scene_get(scene1, "mean luminance", asset_store=store)),
                "scene2_mean_luminance": float(scene_get(scene2, "mean luminance", asset_store=store)),
                "mean_delta_e": float(np.mean(error_array, dtype=float)),
                "mean_delta_e_above2": mean_above2,
                "percent_above2": percent_above2,
                "error_center_row_norm": _canonical_profile(_channel_normalize(center_row)),
            },
            context={"scene1": scene1, "scene2": scene2, "display": display},
        )

    if case_name == "metrics_scielab_example_small":
        def _canonical_profile(values: Any, samples: int = 65) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        error_rgb, scene1, scene2, display = scielab_rgb(
            "hats.jpg",
            "hatsC.jpg",
            "crt.mat",
            0.3,
            asset_store=store,
        )
        snapshot_root = store.ensure()
        hats = dac_to_rgb(iio.imread(snapshot_root / "data" / "images" / "rgb" / "hats.jpg").astype(float) / 255.0)
        hats_c = dac_to_rgb(iio.imread(snapshot_root / "data" / "images" / "rgb" / "hatsC.jpg").astype(float) / 255.0)
        dsp = display_create(str(snapshot_root / "data" / "displays" / "crt.mat"), asset_store=store)
        rgb2xyz = np.asarray(display_get(dsp, "rgb2xyz"), dtype=float)
        white_xyz = np.asarray(display_get(dsp, "white point"), dtype=float)
        img1_xyz = image_linear_transform(hats, rgb2xyz)
        img2_xyz = image_linear_transform(hats_c, rgb2xyz)

        img_width = hats.shape[1] * float(display_get(dsp, "meters per dot"))
        fov = float(np.rad2deg(2.0 * np.arctan2(img_width / 2.0, 0.3)))
        samp_per_deg = hats.shape[1] / max(fov, 1.0e-12)
        params = {
            "deltaEversion": "2000",
            "sampPerDeg": samp_per_deg,
            "imageFormat": "xyz",
            "filterSize": samp_per_deg,
            "filters": [],
        }
        error_explicit, params_out, _, _ = scielab(img1_xyz, img2_xyz, white_xyz, params)
        error_explicit = np.asarray(error_explicit, dtype=float)
        above2 = error_explicit > 2.0

        filters = [np.asarray(kernel, dtype=float) for kernel in params_out["filters"]]
        filter_center_rows = np.vstack(
            [
                _canonical_profile(_channel_normalize(kernel[kernel.shape[0] // 2, :]), 65)
                for kernel in filters
            ]
        )
        filter_peaks = np.array([float(np.max(kernel)) for kernel in filters], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene1_size": np.asarray(scene_get(scene1, "size"), dtype=int),
                "scene2_size": np.asarray(scene_get(scene2, "size"), dtype=int),
                "fov_deg": fov,
                "display_white_point": white_xyz.reshape(3),
                "scielab_rgb_mean_delta_e": float(np.mean(np.asarray(error_rgb, dtype=float), dtype=float)),
                "explicit_error_size": np.asarray(error_explicit.shape, dtype=int),
                "explicit_mean_delta_e": float(np.mean(error_explicit, dtype=float)),
                "explicit_mean_delta_e_above2": float(np.mean(error_explicit[above2], dtype=float)),
                "explicit_percent_above2": float(np.count_nonzero(above2)) / max(float(error_explicit.size), 1.0) * 100.0,
                "filter_support": np.asarray(params_out["support"], dtype=float).reshape(-1),
                "filter_peaks": filter_peaks,
                "filter_center_rows_norm": filter_center_rows,
                "explicit_error_center_row_norm": _canonical_profile(
                    _channel_normalize(error_explicit[error_explicit.shape[0] // 2, :]),
                    129,
                ),
            },
            context={"scene1": scene1, "scene2": scene2, "display": display},
        )

    if case_name == "metrics_scielab_filters_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        def _row_stack(filters: list[np.ndarray], samples: int = 129) -> np.ndarray:
            return np.vstack(
                [
                    _canonical_profile(_channel_normalize(kernel[kernel.shape[0] // 2, :]), samples)
                    for kernel in filters
                ]
            )

        def _mtf_stack(filters: list[np.ndarray], samples: int = 129) -> tuple[np.ndarray, np.ndarray]:
            rows: list[np.ndarray] = []
            peaks: list[float] = []
            for kernel in filters:
                mtf = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.fftshift(kernel))))
                rows.append(_canonical_profile(_channel_normalize(mtf[mtf.shape[0] // 2, :]), samples))
                peaks.append(float(np.max(mtf)))
            return np.vstack(rows), np.asarray(peaks, dtype=float)

        scp_initial = {
            "sampPerDeg": 101.0,
            "filterSize": 101.0,
        }
        initial_filters, initial_support, initial_params = sc_prepare_filters(scp_initial)
        initial_filters = [np.asarray(kernel, dtype=float) for kernel in initial_filters]

        scp_mtf = {
            "sampPerDeg": 512.0,
            "filterSize": 512.0,
        }
        mtf_filters, _, mtf_params = sc_prepare_filters(scp_mtf)
        mtf_filters = [np.asarray(kernel, dtype=float) for kernel in mtf_filters]
        mtf_rows, mtf_peaks = _mtf_stack(mtf_filters)

        versions = ("distribution", "original", "hires")
        version_filter_rows = np.zeros((len(versions), 3, 129), dtype=float)
        version_mtf_rows = np.zeros((len(versions), 3, 129), dtype=float)
        version_filter_peaks = np.zeros((len(versions), 3), dtype=float)
        version_mtf_peaks = np.zeros((len(versions), 3), dtype=float)
        version_filter_sizes = np.zeros(len(versions), dtype=int)
        version_support = None
        for idx, version in enumerate(versions):
            scp_version = {
                "sampPerDeg": 350.0,
                "filterSize": 200.0,
                "filterversion": version,
            }
            version_filters, current_support, version_params = sc_prepare_filters(scp_version)
            version_filters = [np.asarray(kernel, dtype=float) for kernel in version_filters]
            if version_support is None:
                version_support = np.asarray(current_support, dtype=float).reshape(-1)
            version_filter_sizes[idx] = int(round(float(version_params["filterSize"])))
            version_filter_rows[idx, :, :] = _row_stack(version_filters)
            version_filter_peaks[idx, :] = np.asarray([float(np.max(kernel)) for kernel in version_filters], dtype=float)
            current_mtf_rows, current_mtf_peaks = _mtf_stack(version_filters)
            version_mtf_rows[idx, :, :] = current_mtf_rows
            version_mtf_peaks[idx, :] = current_mtf_peaks

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "initial_filter_size": int(round(float(initial_params["filterSize"]))),
                "initial_support": np.asarray(initial_support, dtype=float).reshape(-1),
                "initial_filter_peaks": np.asarray([float(np.max(kernel)) for kernel in initial_filters], dtype=float),
                "initial_filter_sums": np.asarray([float(np.sum(kernel, dtype=float)) for kernel in initial_filters], dtype=float),
                "initial_filter_center_rows_norm": _row_stack(initial_filters),
                "mtf_filter_size": int(round(float(mtf_params["filterSize"]))),
                "mtf_filter_peaks": mtf_peaks,
                "mtf_filter_center_rows_norm": mtf_rows,
                "version_filter_sizes": version_filter_sizes,
                "version_support": np.asarray(version_support, dtype=float).reshape(-1),
                "version_filter_peaks": version_filter_peaks,
                "version_filter_center_rows_norm": version_filter_rows,
                "version_mtf_peaks": version_mtf_peaks,
                "version_mtf_center_rows_norm": version_mtf_rows,
            },
            context={},
        )

    if case_name == "metrics_scielab_mtf_small":
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
        standard_scene = scene_create("harmonic", standard_params, asset_store=store)
        standard_scene = scene_set(standard_scene, "fov", 1.0)

        white_xyz = np.asarray(scene_get(standard_scene, "illuminant xyz", asset_store=store), dtype=float).reshape(3)
        illuminant_energy = np.asarray(scene_get(standard_scene, "illuminant energy", asset_store=store), dtype=float).reshape(-1)
        wave = np.asarray(scene_get(standard_scene, "wave"), dtype=float).reshape(-1)

        delta_e = np.zeros(f_list.size, dtype=float)
        scielab_delta_e = np.zeros(f_list.size, dtype=float)
        for idx, frequency in enumerate(f_list):
            test_params = dict(standard_params)
            test_params["freq"] = float(frequency)
            test_params["contrast"] = 0.5
            test_scene = scene_create("harmonic", test_params, asset_store=store)
            test_scene = scene_set(test_scene, "fov", 1.0)
            test_scene = scene_add(standard_scene, test_scene, "remove spatial mean")

            xyz1 = np.asarray(scene_get(standard_scene, "xyz", asset_store=store), dtype=float)
            xyz2 = np.asarray(scene_get(test_scene, "xyz", asset_store=store), dtype=float)
            delta_e[idx] = float(np.mean(delta_e_ab(xyz1, xyz2, white_xyz, "2000"), dtype=float))
            error_image, _, _, _ = scielab(xyz1, xyz2, white_xyz, sc_params())
            scielab_delta_e[idx] = float(np.mean(np.asarray(error_image, dtype=float), dtype=float))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "frequencies_cpd": f_list,
                "standard_scene_size": np.asarray(scene_get(standard_scene, "size"), dtype=int),
                "standard_fov_deg": float(scene_get(standard_scene, "fov")),
                "wave": wave,
                "white_xyz": white_xyz,
                "illuminant_energy_norm": _channel_normalize(illuminant_energy),
                "delta_e": delta_e,
                "scielab_delta_e": scielab_delta_e,
                "scielab_over_delta_e": np.asarray(scielab_delta_e / np.maximum(delta_e, 1.0e-12), dtype=float),
            },
            context={"scene": standard_scene},
        )

    if case_name == "metrics_scielab_patches_small":
        u_standard = scene_create("uniform", asset_store=store)

        white_xyz = np.asarray(scene_get(u_standard, "illuminant xyz", asset_store=store), dtype=float).reshape(3)
        illuminant_energy = np.asarray(scene_get(u_standard, "illuminant energy", asset_store=store), dtype=float).reshape(-1)
        wave = np.asarray(scene_get(u_standard, "wave"), dtype=float).reshape(-1)
        n_wave = int(scene_get(u_standard, "nwave"))
        lam = np.arange(1, n_wave + 1, dtype=float) / max(float(n_wave), 1.0)

        w1_grid, w2_grid = np.meshgrid(
            np.arange(-0.3, 0.3 + 0.0001, 0.1, dtype=float),
            np.arange(-0.3, 0.3 + 0.0001, 0.1, dtype=float),
        )
        weights = np.column_stack((w1_grid.reshape(-1, order="F"), w2_grid.reshape(-1, order="F")))
        delta_e = np.ones(weights.shape[0], dtype=float)
        scielab_delta_e = np.ones(weights.shape[0], dtype=float)

        xyz1 = np.asarray(scene_get(u_standard, "xyz", asset_store=store), dtype=float)
        for idx, (w1, w2) in enumerate(weights):
            e_adjust1 = float(w1) * np.sin(2.0 * np.pi * lam)
            e_adjust2 = float(w2) * np.cos(2.0 * np.pi * lam)
            new_illuminant = illuminant_energy * (float(w1) * e_adjust1 + float(w2) * e_adjust2 + 1.0)
            u_test = scene_adjust_illuminant(u_standard, new_illuminant)

            xyz2 = np.asarray(scene_get(u_test, "xyz", asset_store=store), dtype=float)
            delta_e[idx] = float(np.mean(delta_e_ab(xyz1, xyz2, white_xyz, "2000"), dtype=float))
            error_image, _, _, _ = scielab(xyz1, xyz2, white_xyz, sc_params())
            scielab_delta_e[idx] = float(np.mean(np.asarray(error_image, dtype=float), dtype=float))

        quantized_scielab = 2.0 * np.round(scielab_delta_e / 2.0)
        delta_gap = np.asarray(scielab_delta_e - delta_e, dtype=float)
        quantized_levels, quantized_counts = np.unique(quantized_scielab, return_counts=True)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "weights": weights,
                "standard_scene_size": np.asarray(scene_get(u_standard, "size"), dtype=int),
                "wave": wave,
                "white_xyz": white_xyz,
                "illuminant_energy_norm": _channel_normalize(illuminant_energy),
                "delta_e": delta_e,
                "scielab_delta_e": scielab_delta_e,
                "delta_gap": delta_gap,
                "delta_gap_stats": np.asarray(
                    [np.max(np.abs(delta_gap)), np.mean(np.abs(delta_gap))],
                    dtype=float,
                ),
                "quantized_scielab_delta_e": quantized_scielab,
                "quantized_scielab_delta_e_sorted": np.sort(quantized_scielab.astype(float)),
                "quantized_scielab_levels": np.asarray(quantized_levels, dtype=float),
                "quantized_scielab_counts": np.asarray(quantized_counts, dtype=float),
            },
            context={"scene": u_standard},
        )

    if case_name == "metrics_scielab_masking_small":
        f_list = np.array([2.0, 4.0, 8.0, 16.0, 32.0], dtype=float)
        target_contrasts = np.arange(0.05, 0.2001, 0.05, dtype=float)
        mask_contrast = 0.8
        params = {
            "ph": 0.0,
            "ang": 0.0,
            "row": 128,
            "col": 128,
            "GaborFlag": 0.0,
            "freq": float(f_list[1]),
            "contrast": float(mask_contrast),
        }

        mask_scene = scene_create("harmonic", params, asset_store=store)
        mask_scene = scene_set(mask_scene, "fov", 1.0)

        white_xyz = 2.0 * np.asarray(scene_get(mask_scene, "illuminant xyz", asset_store=store), dtype=float).reshape(3)
        illuminant_energy = np.asarray(scene_get(mask_scene, "illuminant energy", asset_store=store), dtype=float).reshape(-1)
        wave = np.asarray(scene_get(mask_scene, "wave"), dtype=float).reshape(-1)

        xyz1 = np.maximum(np.asarray(scene_get(mask_scene, "xyz", asset_store=store), dtype=float), 0.0)
        delta_e = np.zeros(target_contrasts.size, dtype=float)
        scielab_delta_e = np.zeros(target_contrasts.size, dtype=float)
        for idx, contrast in enumerate(target_contrasts):
            target_params = dict(params)
            target_params["contrast"] = float(contrast)
            target_scene = scene_create("harmonic", target_params, asset_store=store)
            target_scene = scene_set(target_scene, "fov", 1.0)
            combined_scene = scene_add(mask_scene, target_scene, "remove spatial mean")

            xyz2 = np.maximum(np.asarray(scene_get(combined_scene, "xyz", asset_store=store), dtype=float), 0.0)
            delta_e[idx] = float(np.mean(delta_e_ab(xyz1, xyz2, white_xyz, "2000"), dtype=float))
            error_image, _, _, _ = scielab(xyz1, xyz2, white_xyz, sc_params())
            scielab_delta_e[idx] = float(np.mean(np.asarray(error_image, dtype=float), dtype=float))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "frequencies_cpd": f_list,
                "mask_frequency_cpd": float(params["freq"]),
                "mask_contrast": float(mask_contrast),
                "target_contrasts": target_contrasts,
                "mask_scene_size": np.asarray(scene_get(mask_scene, "size"), dtype=int),
                "mask_fov_deg": float(scene_get(mask_scene, "fov")),
                "wave": wave,
                "white_xyz": white_xyz,
                "illuminant_energy_norm": _channel_normalize(illuminant_energy),
                "delta_e": delta_e,
                "scielab_delta_e": scielab_delta_e,
                "scielab_over_delta_e": np.asarray(scielab_delta_e / np.maximum(delta_e, 1.0e-12), dtype=float),
            },
            context={"scene": mask_scene},
        )

    if case_name == "metrics_scielab_tutorial_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        snapshot_root = store.ensure()
        scene_path = snapshot_root / "data" / "images" / "multispectral" / "StuffedAnimals_tungsten-hdrs.mat"
        scene = scene_from_file(scene_path, "multispectral", asset_store=store)
        scene = scene_set(scene, "fov", 8.0)

        oi = oi_compute(oi_create(asset_store=store), scene)
        sensor = sensor_set_size_to_fov(sensor_create(asset_store=store), 1.1 * float(scene_get(scene, "fov")), oi)
        sensor = sensor_compute(sensor, oi)

        ip = ip_set(ip_create(asset_store=store), "correction method illuminant", "gray world")
        ip = ip_compute(ip, sensor)

        srgb = np.asarray(ip_get(ip, "result"), dtype=float)
        img_xyz = srgb_to_xyz(srgb)
        white_xyz = srgb_to_xyz(np.ones((1, 1, 3), dtype=float)).reshape(3)

        params = sc_params()
        params["sampPerDeg"] = 50.0
        params["filterSize"] = 50.0
        requested_filter_size = float(params["filterSize"])

        img_opp = image_linear_transform(img_xyz, color_transform_matrix("xyz2opp", 10))
        filters, support, params = sc_prepare_filters(params)
        img_filtered_xyz, img_filtered_opp = sc_opponent_filter(img_xyz, params)
        filtered_rgb = xyz_to_srgb(img_filtered_xyz)
        result, white_pt = sc_compute_scielab(img_xyz, white_xyz, params)

        filter_center_rows = np.vstack(
            [
                _canonical_profile(_channel_normalize(kernel[kernel.shape[0] // 2, :]), 65)
                for kernel in [np.asarray(kernel, dtype=float) for kernel in filters]
            ]
        )
        filter_peaks = np.array([float(np.max(np.asarray(kernel, dtype=float))) for kernel in filters], dtype=float)

        filtered_rows = min(img_filtered_xyz.shape[0], img_xyz.shape[0])
        filtered_cols = min(img_filtered_xyz.shape[1], img_xyz.shape[1])
        filtered_xyz_delta = np.asarray(
            img_filtered_xyz[:filtered_rows, :filtered_cols, :] - img_xyz[:filtered_rows, :filtered_cols, :],
            dtype=float,
        )

        original_y = np.asarray(img_xyz[:, :, 1], dtype=float)
        filtered_y = np.asarray(img_filtered_xyz[:, :, 1], dtype=float)
        result_l = np.asarray(result[:, :, 0], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "ip_result_size": np.asarray(srgb.shape, dtype=int),
                "white_xyz": white_xyz,
                "samp_per_deg": float(params["sampPerDeg"]),
                "filter_size": requested_filter_size,
                "image_height_deg": float(srgb.shape[0]) / max(float(params["sampPerDeg"]), 1.0e-12),
                "original_render_mean_rgb_norm": _channel_normalize(np.mean(srgb.reshape(-1, 3), axis=0, dtype=float)),
                "original_render_center_row_luma_norm": _canonical_profile(_channel_normalize(original_y[original_y.shape[0] // 2, :])),
                "img_opp_channel_means": np.mean(img_opp.reshape(-1, 3), axis=0, dtype=float),
                "filter_support": np.asarray(support, dtype=float).reshape(-1),
                "filter_peaks": filter_peaks,
                "filter_center_rows_norm": filter_center_rows,
                "filtered_xyz_size": np.asarray(img_filtered_xyz.shape, dtype=int),
                "filtered_xyz_delta_stats": np.array(
                    [float(np.mean(np.abs(filtered_xyz_delta), dtype=float)), float(np.max(np.abs(filtered_xyz_delta)))],
                    dtype=float,
                ),
                "filtered_opp_channel_means": np.mean(img_filtered_opp.reshape(-1, 3), axis=0, dtype=float),
                "filtered_render_mean_rgb_norm": _channel_normalize(np.mean(filtered_rgb.reshape(-1, 3), axis=0, dtype=float)),
                "filtered_render_center_row_luma_norm": _canonical_profile(_channel_normalize(filtered_y[filtered_y.shape[0] // 2, :])),
                "result_size": np.asarray(result.shape, dtype=int),
                "result_white_point": np.asarray(white_pt, dtype=float).reshape(3),
                "result_lab_channel_means": np.mean(result.reshape(-1, 3), axis=0, dtype=float),
                "result_l_center_row_norm": _canonical_profile(_channel_normalize(result_l[result_l.shape[0] // 2, :])),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "metrics_scielab_harmonic_experiments_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        size = 512
        max_frequency = float(size) / 64.0
        scene = scene_create("sweep frequency", size, max_frequency, asset_store=store)
        scene = scene_set(scene, "fov", 8.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "diffuser method", "blur")
        oi = oi_set(oi, "diffuser blur", 1.5e-6)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set_size_to_fov(sensor, 0.95 * float(scene_get(scene, "fov")), oi)
        sensor = sensor_compute(sensor, oi)

        ip = ip_set(ip_create(asset_store=store), "correction method illuminant", "gray world")
        ip = ip_compute(ip, sensor)

        img = np.asarray(ip_get(ip, "result"), dtype=float)
        img_xyz = srgb_to_xyz(img)
        img_opp = image_linear_transform(img_xyz, color_transform_matrix("xyz2opp", 10))
        img_opp_xw, _, _, _ = rgb_to_xw_format(img_opp)
        opponent_means = np.mean(np.asarray(img_opp_xw, dtype=float), axis=0)

        white_xyz = srgb_to_xyz(np.ones((1, 1, 3), dtype=float)).reshape(3)
        params = sc_params()
        params["sampPerDeg"] = 100.0

        scale_factors = np.array(
            [
                [1.0, 0.5, 1.0],
                [1.0, 1.0, 0.5],
                [0.75, 1.0, 1.0],
            ],
            dtype=float,
        )
        altered_render_means = np.zeros((scale_factors.shape[0], 3), dtype=float)
        altered_opp_means = np.zeros((scale_factors.shape[0], 3), dtype=float)
        error_stats = np.zeros((scale_factors.shape[0], 4), dtype=float)
        error_center_rows = np.zeros((scale_factors.shape[0], 129), dtype=float)

        padded_img = np.pad(img, ((16, 16), (16, 16), (0, 0)), mode="constant")
        for index, scale in enumerate(scale_factors):
            adjusted_opp = np.zeros_like(img_opp, dtype=float)
            for channel_index in range(3):
                adjusted_opp[:, :, channel_index] = (
                    (img_opp[:, :, channel_index] - float(opponent_means[channel_index])) * float(scale[channel_index])
                    + float(opponent_means[channel_index])
                )

            adjusted_xyz = image_linear_transform(adjusted_opp, color_transform_matrix("opp2xyz", 10))
            adjusted_rgb = xyz_to_srgb(adjusted_xyz)
            altered_render_means[index, :] = _channel_normalize(np.mean(adjusted_rgb.reshape(-1, 3), axis=0))
            adjusted_opp_xw, _, _, _ = rgb_to_xw_format(adjusted_opp)
            altered_opp_means[index, :] = np.mean(np.asarray(adjusted_opp_xw, dtype=float), axis=0)

            error_image, _, _, _ = scielab(
                padded_img,
                np.pad(adjusted_rgb, ((16, 16), (16, 16), (0, 0)), mode="constant"),
                white_xyz,
                params,
            )
            error_array = np.asarray(error_image, dtype=float)
            error_stats[index, :] = _stats_vector(error_array)
            error_center_rows[index, :] = _canonical_profile(
                _channel_normalize(error_array[error_array.shape[0] // 2, :]),
                129,
            )

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "sweep_max_frequency_cpd": float(max_frequency),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "oi_diffuser_blur_m": float(oi_get(oi, "diffuser blur")),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "ip_result_size": np.asarray(img.shape, dtype=int),
                "white_xyz": white_xyz,
                "samp_per_deg": float(params["sampPerDeg"]),
                "scale_factors": scale_factors,
                "original_render_mean_rgb_norm": _channel_normalize(np.mean(img.reshape(-1, 3), axis=0)),
                "original_opp_channel_means": np.asarray(opponent_means, dtype=float),
                "altered_render_mean_rgb_norm": altered_render_means,
                "altered_opp_channel_means": altered_opp_means,
                "error_stats": error_stats,
                "error_center_row_norm": error_center_rows,
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "metrics_edge2mtf_small":
        def _canonical_profile(values: Any, samples: int = 65) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        scene = scene_create("slanted bar", 256, 7.0 / 3.0, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "distance", 1.0)
        scene = scene_set(scene, "fov", 5.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 2.8)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "autoExposure", 1)
        sensor = sensor_compute(sensor, oi)

        ip = ip_compute(ip_create(asset_store=store), sensor)
        rect = np.asarray(iso_find_slanted_bar(ip), dtype=int).reshape(-1)
        col_min, row_min, width, height = rect
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        bar_image = result[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
        mtf_data = edge_to_mtf(bar_image, channel=2, fixed_row=20)
        green_profile = np.mean(np.asarray(bar_image[:, :, 1], dtype=float), axis=0, dtype=float)
        mtf = np.asarray(mtf_data["mtf"], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "ip_size": np.asarray(ip_get(ip, "size"), dtype=int),
                "roi_aspect_ratio": float(bar_image.shape[0]) / max(float(bar_image.shape[1]), 1.0),
                "roi_fill_fraction": float(np.prod(bar_image.shape[:2])) / max(float(np.prod(result.shape[:2])), 1.0),
                "bar_green_mean_profile_norm": _canonical_profile(_channel_normalize(green_profile)),
                "lag_stats": _stats_vector(np.asarray(mtf_data["lags"], dtype=float)),
                "lsf_norm": _canonical_profile(_channel_normalize(np.asarray(mtf_data["lsf"], dtype=float))),
                "mtf_norm": _canonical_profile(mtf / max(float(mtf[0]), 1.0e-12)),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "metrics_mtf_slanted_bar_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        scene = scene_create("slanted bar", 512, 7.0 / 3.0, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "distance", 1.0)
        scene = scene_set(scene, "fov", 5.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 2.0)
        oi = oi_compute(oi, scene)

        sensor_color = sensor_create(asset_store=store)
        sensor_color = sensor_set(sensor_color, "autoExposure", 1)
        sensor_color = sensor_compute(sensor_color, oi)
        ip_color = ip_compute(ip_create(asset_store=store), sensor_color)

        rect = np.asarray(iso_find_slanted_bar(ip_color), dtype=int).reshape(-1)
        col_min, row_min, width, height = rect
        color_result = np.asarray(ip_get(ip_color, "result"), dtype=float)
        color_bar = np.asarray(color_result[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :], dtype=float)
        color_dx = float(sensor_get(sensor_color, "pixel width", "mm"))
        color_direct = iso12233(color_bar, delta_x=color_dx, plot_options="none")
        color_ie = ie_iso12233(ip_color, sensor_color, plot_options="none", master_rect=rect)

        sensor_mono = sensor_create("monochrome", asset_store=store)
        sensor_mono = sensor_set(sensor_mono, "autoExposure", 1)
        sensor_mono = sensor_compute(sensor_mono, oi)
        ip_mono = ip_compute(ip_create(asset_store=store), sensor_mono)
        mono_result = np.asarray(ip_get(ip_mono, "result"), dtype=float)
        mono_bar = np.asarray(mono_result[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :], dtype=float)
        mono_dx = float(sensor_get(sensor_mono, "pixel width", "mm"))
        mono_direct = iso12233(mono_bar, delta_x=mono_dx, plot_options="none")

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "color_sensor_size": np.asarray(sensor_get(sensor_color, "size"), dtype=int),
                "mono_sensor_size": np.asarray(sensor_get(sensor_mono, "size"), dtype=int),
                "master_rect": rect,
                "color_dx_mm": color_dx,
                "mono_dx_mm": mono_dx,
                "color_direct_esf_norm": _canonical_profile(
                    _channel_normalize(np.asarray(color_direct.esf, dtype=float)[:, -1])
                ),
                "color_direct_lsf_norm": _canonical_profile(_channel_normalize(np.asarray(color_direct.lsf, dtype=float))),
                "color_direct_mtf_norm": _canonical_profile(
                    np.asarray(color_direct.mtf, dtype=float)[:, -1]
                    / max(float(np.asarray(color_direct.mtf, dtype=float)[0, -1]), 1.0e-12)
                ),
                "color_direct_nyquistf": float(color_direct.nyquistf),
                "color_direct_mtf50": float(color_direct.mtf50),
                "color_direct_aliasing_percentage": float(color_direct.aliasingPercentage),
                "ie_color_esf_norm": _canonical_profile(
                    _channel_normalize(np.asarray(color_ie.esf, dtype=float)[:, -1])
                ),
                "ie_color_lsf_norm": _canonical_profile(_channel_normalize(np.asarray(color_ie.lsf, dtype=float))),
                "ie_color_mtf_norm": _canonical_profile(
                    np.asarray(color_ie.mtf, dtype=float)[:, -1]
                    / max(float(np.asarray(color_ie.mtf, dtype=float)[0, -1]), 1.0e-12)
                ),
                "ie_color_nyquistf": float(color_ie.nyquistf),
                "ie_color_mtf50": float(color_ie.mtf50),
                "ie_color_aliasing_percentage": float(color_ie.aliasingPercentage),
                "mono_direct_esf_norm": _canonical_profile(
                    _channel_normalize(np.asarray(mono_direct.esf, dtype=float)[:, -1])
                ),
                "mono_direct_lsf_norm": _canonical_profile(_channel_normalize(np.asarray(mono_direct.lsf, dtype=float))),
                "mono_direct_mtf_norm": _canonical_profile(
                    np.asarray(mono_direct.mtf, dtype=float)[:, -1]
                    / max(float(np.asarray(mono_direct.mtf, dtype=float)[0, -1]), 1.0e-12)
                ),
                "mono_direct_nyquistf": float(mono_direct.nyquistf),
                "mono_direct_mtf50": float(mono_direct.mtf50),
                "mono_direct_aliasing_percentage": float(mono_direct.aliasingPercentage),
            },
            context={
                "scene": scene,
                "oi": oi,
                "color_sensor": sensor_color,
                "mono_sensor": sensor_mono,
                "color_ip": ip_color,
                "mono_ip": ip_mono,
            },
        )

    if case_name == "metrics_mtf_pixel_size_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        def _matlab_round_array(values: Any) -> np.ndarray:
            return np.floor(np.asarray(values, dtype=float) + 0.5).astype(int)

        scene = scene_create("slanted bar", 512, 7.0 / 3.0, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "distance", 1.0)
        scene = scene_set(scene, "fov", 5.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_compute(oi, scene)

        base_sensor = sensor_create("monochrome", asset_store=store)
        base_sensor = sensor_set(base_sensor, "autoExposure", 1)
        base_ip = ip_create(asset_store=store)

        master_rect = np.array([199, 168, 101, 167], dtype=int)
        pixel_sizes_um = np.array([2.0, 3.0, 5.0, 9.0], dtype=float)
        sensor_sizes = []
        rects = []
        bar_sizes = []
        nyquist = []
        mtf50 = []
        mtf_profiles = []

        for pixel_size_um in pixel_sizes_um:
            sensor = sensor_set(
                base_sensor,
                "pixel size constant fill factor",
                np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6,
            )
            sensor = sensor_set(sensor, "rows", round(512.0 / float(pixel_size_um)))
            sensor = sensor_set(sensor, "cols", round(512.0 / float(pixel_size_um)))
            sensor = sensor_compute(sensor, oi)
            ip = ip_compute(base_ip, sensor)

            rect = _matlab_round_array(master_rect.astype(float) / float(pixel_size_um))
            col_min, row_min, width, height = rect
            bar = np.asarray(ip_get(ip, "result"), dtype=float)[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :]
            mtf_data = iso12233(bar, float(sensor_get(sensor, "pixel width", "mm")), plot_options="none")

            sensor_sizes.append(np.asarray(sensor_get(sensor, "size"), dtype=int))
            rects.append(rect)
            bar_sizes.append(np.asarray(bar.shape, dtype=int))
            nyquist.append(float(mtf_data.nyquistf))
            mtf50.append(float(mtf_data.mtf50))
            mtf_profiles.append(
                _canonical_profile(
                    np.asarray(mtf_data.mtf, dtype=float)[:, -1]
                    / max(float(np.asarray(mtf_data.mtf, dtype=float)[0, -1]), 1.0e-12)
                )
            )

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "pixel_sizes_um": pixel_sizes_um,
                "sensor_sizes": np.asarray(sensor_sizes, dtype=int),
                "rects": np.asarray(rects, dtype=int),
                "bar_sizes": np.asarray(bar_sizes, dtype=int),
                "nyquistf": np.asarray(nyquist, dtype=float),
                "mtf50": np.asarray(mtf50, dtype=float),
                "mtf_profiles_norm": np.asarray(mtf_profiles, dtype=float),
            },
            context={
                "scene": scene,
                "oi": oi,
            },
        )

    if case_name == "metrics_snr_pixel_size_luxsec_small":
        integration_time = 0.010
        pixel_sizes_um = np.array([2.0, 4.0, 6.0, 9.0, 10.0], dtype=float)
        read_noise_mv = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
        voltage_swing_v = np.array([0.7, 1.2, 1.5, 2.0, 3.0], dtype=float)
        dark_voltage_mv_per_sec = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "integration time", integration_time)

        snr_curves = []
        luxsec_curves = []
        volts_per_lux_sec = []
        luxsec_saturation = []
        mean_volts = []
        snr_read = []
        snr_shot = []

        for pixel_size_um, read_noise_mv_value, voltage_swing_value, dark_voltage_mv_value in zip(
            pixel_sizes_um,
            read_noise_mv,
            voltage_swing_v,
            dark_voltage_mv_per_sec,
            strict=False,
        ):
            pixel_sensor = sensor_set(
                sensor,
                "pixel size constant fill factor",
                np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6,
            )
            pixel_sensor = sensor_set(pixel_sensor, "readNoiseSTDvolts", float(read_noise_mv_value) * 1.0e-3)
            pixel_sensor = sensor_set(pixel_sensor, "voltageSwing", float(voltage_swing_value))
            pixel_sensor = sensor_set(pixel_sensor, "darkVoltage", float(dark_voltage_mv_value) * 1.0e-3)

            snr, luxsec, snr_shot_curve, snr_read_curve, _ = pixel_snr_luxsec(pixel_sensor, asset_store=store)
            volts_per_lux, saturation_luxsec, mean_voltage, _, _ = pixel_v_per_lux_sec(pixel_sensor, asset_store=store)

            snr_curves.append(np.asarray(snr, dtype=float).reshape(-1))
            luxsec_curves.append(np.asarray(luxsec, dtype=float)[:, 0])
            volts_per_lux_sec.append(float(np.asarray(volts_per_lux, dtype=float)[0]))
            luxsec_saturation.append(float(saturation_luxsec))
            mean_volts.append(float(np.asarray(mean_voltage, dtype=float)[0]))
            snr_shot.append(np.asarray(snr_shot_curve, dtype=float).reshape(-1))
            if np.isscalar(snr_read_curve):
                snr_read.append(np.full(np.asarray(snr, dtype=float).shape, float(snr_read_curve), dtype=float))
            else:
                snr_read.append(np.asarray(snr_read_curve, dtype=float).reshape(-1))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "integration_time_s": float(integration_time),
                "pixel_sizes_um": pixel_sizes_um,
                "read_noise_mv": read_noise_mv,
                "voltage_swing_v": voltage_swing_v,
                "dark_voltage_mv_per_sec": dark_voltage_mv_per_sec,
                "snr_db": np.asarray(snr_curves, dtype=float),
                "luxsec_curves": np.asarray(luxsec_curves, dtype=float),
                "snr_shot_db": np.asarray(snr_shot, dtype=float),
                "snr_read_db": np.asarray(snr_read, dtype=float),
                "volts_per_lux_sec": np.asarray(volts_per_lux_sec, dtype=float),
                "luxsec_saturation": np.asarray(luxsec_saturation, dtype=float),
                "mean_volts": np.asarray(mean_volts, dtype=float),
            },
            context={},
        )

    if case_name == "metrics_mtf_slanted_bar_infrared_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        wave = np.arange(400.0, 1068.0 + 0.1, 4.0, dtype=float)
        scene = scene_create("slanted bar", 512, 7.0 / 3.0, 5.0, wave, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "distance", 1.0)
        scene = scene_set(scene, "fov", 5.0)

        oi = oi_create("diffraction limited", asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, "NikonD200IR.mat", asset_store=store)
        sensor = sensor_set(sensor, "wave", wave)
        sensor = sensor_set(sensor, "filterSpectra", filter_spectra)
        sensor = sensor_set(sensor, "filterNames", filter_names)
        sensor = sensor_set(sensor, "ir filter", np.ones_like(wave))
        sensor = sensor_set(sensor, "pixel spectral qe", np.ones_like(wave))
        sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)
        sensor = sensor_compute(sensor, oi)

        ip = ip_create(asset_store=store)
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

        ir_filter, ir_filter_names, _ = ie_read_color_filter(wave, "IRBlocking", asset_store=store)
        blocked_sensor = sensor_set(sensor, "ir filter", np.asarray(ir_filter, dtype=float).reshape(-1))
        blocked_sensor = sensor_compute(blocked_sensor, oi)
        blocked_ip = ip_compute(ip, blocked_sensor)
        blocked_mtf = ie_iso12233(blocked_ip, blocked_sensor, "none")

        fixed_mtf_norm = np.asarray(fixed_mtf.mtf, dtype=float)[:, -1]
        fixed_mtf_norm = fixed_mtf_norm / max(float(fixed_mtf_norm[0]), 1.0e-12)
        blocked_mtf_norm = np.asarray(blocked_mtf.mtf, dtype=float)[:, -1]
        blocked_mtf_norm = blocked_mtf_norm / max(float(blocked_mtf_norm[0]), 1.0e-12)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "filter_names": np.asarray(filter_names, dtype=object),
                "ir_filter_names": np.asarray(ir_filter_names, dtype=object),
                "filter_spectra_stats": _stats_vector(np.asarray(filter_spectra, dtype=float)),
                "fixed_rect": fixed_rect,
                "fixed_bar_size": np.asarray(fixed_bar.shape, dtype=int),
                "fixed_mtf50": float(fixed_mtf.mtf50),
                "fixed_nyquistf": float(fixed_mtf.nyquistf),
                "fixed_esf_norm": _canonical_profile(_channel_normalize(np.asarray(fixed_mtf.esf, dtype=float)[:, -1])),
                "fixed_lsf_norm": _canonical_profile(_channel_normalize(np.asarray(fixed_mtf.lsf, dtype=float))),
                "fixed_mtf_norm": _canonical_profile(fixed_mtf_norm),
                "blocked_rect": np.asarray(blocked_mtf.rect, dtype=int),
                "blocked_mtf50": float(blocked_mtf.mtf50),
                "blocked_nyquistf": float(blocked_mtf.nyquistf),
                "blocked_lsf_um": _canonical_profile(np.asarray(blocked_mtf.lsfx, dtype=float) * 1000.0),
                "blocked_lsf_norm": _canonical_profile(_channel_normalize(np.asarray(blocked_mtf.lsf, dtype=float))),
                "blocked_mtf_norm": _canonical_profile(blocked_mtf_norm),
            },
            context={
                "scene": scene,
                "oi": oi,
                "sensor": sensor,
                "ip": ip,
                "blocked_sensor": blocked_sensor,
                "blocked_ip": blocked_ip,
            },
        )

    if case_name == "metrics_acutance_small":
        def _canonical_profile(values: Any, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile).astype(float)

        camera = camera_create(asset_store=store)
        camera = camera_set(camera, "sensor auto exposure", True)
        camera = camera_set(camera, "optics fnumber", 4.0)

        cmtf = camera_mtf(camera, asset_store=store)
        oi = camera_get(camera, "oi")
        deg_per_mm = float(camera_get(camera, "sensor h deg per distance", "mm", None, oi))
        cpd = np.asarray(cmtf.freq, dtype=float) / max(deg_per_mm, 1.0e-12)
        luminance_mtf = np.asarray(cmtf.mtf, dtype=float)[:, -1]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_size": np.asarray(camera_get(camera, "sensor size"), dtype=int),
                "ip_size": np.asarray(ip_get(cmtf.vci, "size"), dtype=int),
                "rect": np.asarray(cmtf.rect, dtype=int),
                "deg_per_mm": float(deg_per_mm),
                "cpd_stats": np.array(
                    [
                        float(cpd[0]),
                        float(cpd[-1]),
                        float(cpd.size),
                        float(np.mean(np.diff(cpd), dtype=float)),
                    ],
                    dtype=float,
                ),
                "cpiq_norm": _canonical_profile(np.asarray(cpiq_csf(cpd), dtype=float)),
                "lum_mtf_norm": _canonical_profile(
                    np.asarray(luminance_mtf, dtype=float) / max(float(luminance_mtf[0]), 1.0e-12)
                ),
                "acutance": float(iso_acutance(cpd, luminance_mtf)),
                "camera_acutance": float(camera_acutance(camera, asset_store=store)),
            },
            context={
                "camera": camera,
                "oi": oi,
                "ip": cmtf.vci,
            },
        )

    if case_name == "metrics_color_accuracy_small":
        camera = camera_create(asset_store=store)
        camera = camera_set(camera, "sensor auto exposure", True)
        color_accuracy, camera = camera_color_accuracy(camera, asset_store=store)
        ip = camera_get(camera, "ip")
        embedded_rgb, compare_patch_srgb, patch_size = macbeth_compare_ideal(ip, asset_store=store)

        white_xyz = np.asarray(color_accuracy["whiteXYZ"], dtype=float).reshape(-1)
        ideal_white_xyz = np.asarray(color_accuracy["idealWhiteXYZ"], dtype=float).reshape(-1)
        delta_e = np.asarray(color_accuracy["deltaE"], dtype=float).reshape(-1)
        macbeth_lab = np.asarray(color_accuracy["macbethLAB"], dtype=float)
        ideal_patch_linear = xw_to_rgb_format(
            _macbeth_ideal_linear_rgb(np.asarray(ip_get(ip, "wave"), dtype=float), asset_store=store),
            4,
            6,
        )
        ideal_patch_srgb = linear_to_srgb(ideal_patch_linear / max(float(np.max(ideal_patch_linear)), 1.0e-12))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_size": np.asarray(camera_get(camera, "sensor size"), dtype=int),
                "ip_size": np.asarray(ip_get(ip, "size"), dtype=int),
                "corner_points": np.asarray(color_accuracy["cornerPoints"], dtype=float),
                "white_xyz_norm": white_xyz / max(float(white_xyz[1]), 1.0e-12),
                "ideal_white_xyz_norm": ideal_white_xyz / max(float(ideal_white_xyz[1]), 1.0e-12),
                "delta_e": delta_e,
                "delta_e_stats": np.array(
                    [float(np.mean(delta_e)), float(np.max(delta_e)), float(np.std(delta_e, dtype=float))],
                    dtype=float,
                ),
                "macbeth_lab": macbeth_lab,
                "compare_patch_srgb": np.asarray(compare_patch_srgb, dtype=float),
                "ideal_patch_srgb": np.asarray(ideal_patch_srgb, dtype=float),
                "embedded_channel_means": np.mean(np.asarray(embedded_rgb, dtype=float), axis=(0, 1), dtype=float).reshape(-1),
                "patch_size": np.asarray(patch_size, dtype=int).reshape(-1),
            },
            context={
                "camera": camera,
                "ip": ip,
            },
        )

    if case_name == "metrics_macbeth_delta_e_small":
        scene = scene_create(asset_store=store)
        scene = scene_adjust_luminance(scene, 75.0, asset_store=store)
        scene = scene_set(scene, "fov", 2.64)
        scene = scene_set(scene, "distance", 10.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_set(oi, "optics focal length", 20.0e-3)
        oi = oi_set(oi, "optics off axis method", "skip")
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)
        sensor = sensor_compute(sensor, oi)

        sensor_corner_points = np.array(
            [[1.0, 244.0], [328.0, 246.0], [329.0, 28.0], [2.0, 27.0]],
            dtype=float,
        )
        sensor = sensor_set(sensor, "chart corner points", sensor_corner_points)
        ccm_matrix, sensor_locs = sensor_ccm(sensor, "macbeth", None, True, asset_store=store)

        ip = ip_create(asset_store=store)
        ip = ip_set(ip, "scale display", 1)
        ip = ip_set(ip, "conversion matrix sensor", ccm_matrix)
        ip = ip_set(ip, "correction matrix illuminant", np.array([], dtype=float))
        ip = ip_set(ip, "internal cs 2 display space", np.array([], dtype=float))
        ip = ip_set(ip, "conversion method sensor", "Current matrix")
        ip = ip_set(ip, "internalCS", "Sensor")
        ip = ip_compute(ip, sensor, asset_store=store)

        point_loc = np.array(
            [[4.0, 246.0], [328.0, 243.0], [327.0, 26.0], [3.0, 27.0]],
            dtype=float,
        )
        macbeth_lab, macbeth_xyz, delta_e, _ = macbeth_color_error(ip, "D65", point_loc, asset_store=store)

        result = np.asarray(ip_get(ip, "result"), dtype=float)
        result_flat = result.reshape(-1, result.shape[2])
        white_xyz = np.asarray(macbeth_xyz, dtype=float)[3, :].reshape(-1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "ip_size": np.asarray(ip_get(ip, "size"), dtype=int),
                "ccm_matrix": np.asarray(ccm_matrix, dtype=float),
                "sensor_locs": np.asarray(sensor_locs, dtype=float),
                "point_loc": np.asarray(point_loc, dtype=float),
                "white_xyz_norm": white_xyz / max(float(white_xyz[1]), 1.0e-12),
                "delta_e": np.asarray(delta_e, dtype=float).reshape(-1),
                "delta_e_stats": np.array(
                    [
                        float(np.mean(delta_e, dtype=float)),
                        float(np.max(delta_e)),
                        float(np.std(delta_e, dtype=float)),
                    ],
                    dtype=float,
                ),
                "macbeth_lab": np.asarray(macbeth_lab, dtype=float),
                "result_channel_means_norm": _channel_normalize(np.mean(result_flat, axis=0, dtype=float)),
                "result_channel_p95_norm": _channel_normalize(np.percentile(result_flat, 95.0, axis=0)),
            },
            context={
                "scene": scene,
                "oi": oi,
                "sensor": sensor,
                "ip": ip,
            },
        )

    if case_name == "scene_illuminant_change":
        scene = scene_create(asset_store=store)
        wave = scene_get(scene, "wave")
        preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=store)
        no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "preserve_mean": scene_get(preserve, "mean luminance", asset_store=store),
                "no_preserve_mean": scene_get(no_preserve, "mean luminance", asset_store=store),
                "preserve_photons": scene_get(preserve, "photons"),
                "no_preserve_photons": scene_get(no_preserve, "photons"),
            },
            context={
                "scene": scene,
                "preserve_scene": preserve,
                "no_preserve_scene": no_preserve,
            },
        )

    if case_name == "scene_cct_blackbody_small":
        wave = np.arange(400.0, 721.0, 5.0, dtype=float)
        single_temperatures = np.array([3500.0, 6500.0, 8500.0], dtype=float)
        spd_3500 = np.asarray(blackbody(wave, single_temperatures[0], kind="energy"), dtype=float).reshape(-1)
        estimated_single = np.array(
            [
                float(
                    spd_to_cct(
                        wave,
                        np.asarray(blackbody(wave, temperature_k, kind="energy"), dtype=float).reshape(-1),
                        asset_store=store,
                    )
                )
                for temperature_k in single_temperatures
            ],
            dtype=float,
        )
        multi_temperatures = np.arange(4500.0, 8501.0, 1000.0, dtype=float)
        spd_multi = np.asarray(blackbody(wave, multi_temperatures, kind="energy"), dtype=float)
        estimated_multi = np.asarray(spd_to_cct(wave, spd_multi, asset_store=store), dtype=float).reshape(-1)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "single_temperatures_k": single_temperatures,
                "spd_3500": spd_3500,
                "estimated_single_k": estimated_single,
                "multi_temperatures_k": multi_temperatures,
                "spd_multi": spd_multi,
                "estimated_multi_k": estimated_multi,
            },
            context={},
        )

    if case_name == "scene_daylight_small":
        wave = np.arange(400.0, 771.0, 1.0, dtype=float)
        cct = np.arange(4000.0, 10001.0, 1000.0, dtype=float)
        photons = np.asarray(daylight(wave, cct, "photons", asset_store=store), dtype=float)
        lum_photons = np.asarray(luminance_from_photons(photons.T, wave, asset_store=store), dtype=float).reshape(-1)
        photons_scaled = photons * (100.0 / np.maximum(lum_photons, 1e-12)).reshape(1, -1)

        energy = np.asarray(daylight(wave, cct, "energy", asset_store=store), dtype=float)
        lum_energy = np.asarray(luminance_from_energy(energy.T, wave, asset_store=store), dtype=float).reshape(-1)
        energy_scaled = energy * (100.0 / np.maximum(lum_energy, 1e-12)).reshape(1, -1)

        day_basis = np.asarray(ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=store), dtype=float)
        basis_weights = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]], dtype=float)
        basis_examples = day_basis @ basis_weights
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "cct_k": cct,
                "photons": photons,
                "lum_photons": lum_photons,
                "photons_scaled": photons_scaled,
                "energy": energy,
                "lum_energy": lum_energy,
                "energy_scaled": energy_scaled,
                "day_basis": day_basis,
                "basis_weights": basis_weights,
                "basis_examples": basis_examples,
            },
            context={},
        )

    if case_name == "scene_illuminant_small":
        default_blackbody = illuminant_create("blackbody", asset_store=store)
        wave_3000 = np.arange(400.0, 701.0, 1.0, dtype=float)
        blackbody_3000 = illuminant_create("blackbody", wave_3000, 3000.0, asset_store=store)
        d65_200 = illuminant_create("d65", None, 200.0, asset_store=store)
        equal_energy = illuminant_create("equal energy", None, 200.0, asset_store=store)
        equal_photons = illuminant_create("equal photons", None, 200.0, asset_store=store)
        illuminant_c = illuminant_create("illuminant C", None, 200.0, asset_store=store)
        mono_555 = illuminant_create("555 nm", None, 200.0, asset_store=store)
        d65_sparse = illuminant_create("d65", np.arange(400.0, 601.0, 2.0, dtype=float), 200.0, asset_store=store)
        d65_resampled = illuminant_set(d65_sparse, "wave", np.arange(400.0, 701.0, 5.0, dtype=float), asset_store=store)
        fluorescent = illuminant_create("fluorescent", np.arange(400.0, 701.0, 5.0, dtype=float), 10.0, asset_store=store)
        tungsten = illuminant_create("tungsten", None, 300.0, asset_store=store)
        mono_photons = np.asarray(illuminant_get(mono_555, "photons"), dtype=float).reshape(-1)
        mono_idx = int(np.argmax(mono_photons))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "default_blackbody_wave": np.asarray(illuminant_get(default_blackbody, "wave"), dtype=float).reshape(-1),
                "default_blackbody_photons": np.asarray(illuminant_get(default_blackbody, "photons"), dtype=float).reshape(-1),
                "default_blackbody_luminance": float(illuminant_get(default_blackbody, "luminance", asset_store=store)),
                "blackbody_3000_wave": np.asarray(illuminant_get(blackbody_3000, "wave"), dtype=float).reshape(-1),
                "blackbody_3000_photons": np.asarray(illuminant_get(blackbody_3000, "photons"), dtype=float).reshape(-1),
                "d65_200_wave": np.asarray(illuminant_get(d65_200, "wave"), dtype=float).reshape(-1),
                "d65_200_photons": np.asarray(illuminant_get(d65_200, "photons"), dtype=float).reshape(-1),
                "d65_200_luminance": float(illuminant_get(d65_200, "luminance", asset_store=store)),
                "equal_energy_wave": np.asarray(illuminant_get(equal_energy, "wave"), dtype=float).reshape(-1),
                "equal_energy_energy": np.asarray(illuminant_get(equal_energy, "energy"), dtype=float).reshape(-1),
                "equal_energy_mean": float(np.mean(np.asarray(illuminant_get(equal_energy, "energy"), dtype=float))),
                "equal_photons_wave": np.asarray(illuminant_get(equal_photons, "wave"), dtype=float).reshape(-1),
                "equal_photons_photons": np.asarray(illuminant_get(equal_photons, "photons"), dtype=float).reshape(-1),
                "equal_photons_energy": np.asarray(illuminant_get(equal_photons, "energy"), dtype=float).reshape(-1),
                "illuminant_c_photons": np.asarray(illuminant_get(illuminant_c, "photons"), dtype=float).reshape(-1),
                "mono_555_wave": np.asarray(illuminant_get(mono_555, "wave"), dtype=float).reshape(-1),
                "mono_555_photons": mono_photons,
                "mono_555_nonzero_index": mono_idx + 1,
                "mono_555_nonzero_wave_nm": float(np.asarray(illuminant_get(mono_555, "wave"), dtype=float).reshape(-1)[mono_idx]),
                "d65_sparse_wave": np.asarray(illuminant_get(d65_sparse, "wave"), dtype=float).reshape(-1),
                "d65_sparse_energy": np.asarray(illuminant_get(d65_sparse, "energy"), dtype=float).reshape(-1),
                "d65_resampled_wave": np.asarray(illuminant_get(d65_resampled, "wave"), dtype=float).reshape(-1),
                "d65_resampled_energy": np.asarray(illuminant_get(d65_resampled, "energy"), dtype=float).reshape(-1),
                "fluorescent_wave": np.asarray(illuminant_get(fluorescent, "wave"), dtype=float).reshape(-1),
                "fluorescent_photons": np.asarray(illuminant_get(fluorescent, "photons"), dtype=float).reshape(-1),
                "tungsten_wave": np.asarray(illuminant_get(tungsten, "wave"), dtype=float).reshape(-1),
                "tungsten_photons": np.asarray(illuminant_get(tungsten, "photons"), dtype=float).reshape(-1),
            },
            context={},
        )

    if case_name == "scene_illuminant_mixtures_small":
        tungsten_scene = scene_illuminant_ss(scene_create("macbeth tungsten", asset_store=store))
        daylight_scene = scene_illuminant_ss(scene_create(asset_store=store))
        tungsten_energy = np.asarray(scene_get(tungsten_scene, "illuminant energy"), dtype=float)
        daylight_energy = np.asarray(scene_get(daylight_scene, "illuminant energy"), dtype=float)
        rows, cols = scene_get(tungsten_scene, "size")
        split_row = int(np.rint(rows / 2.0))
        mixed_energy = tungsten_energy.copy()
        mixed_energy[:split_row, :, :] = daylight_energy[:split_row, :, :]
        mixed_scene = scene_adjust_illuminant(tungsten_scene.clone(), mixed_energy, asset_store=store)
        mixed_scene = scene_set(mixed_scene, "name", "Mixed illuminant")

        band_rows = max(1, rows // 4)
        top_slice = slice(0, band_rows)
        bottom_slice = slice(rows - band_rows, rows)
        mixed_illuminant = np.asarray(scene_get(mixed_scene, "illuminant energy"), dtype=float)
        source_reflectance = np.asarray(scene_get(tungsten_scene, "reflectance"), dtype=float)
        mixed_reflectance = np.asarray(scene_get(mixed_scene, "reflectance"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(scene_get(mixed_scene, "wave"), dtype=float).reshape(-1),
                "scene_size": np.array([rows, cols], dtype=int),
                "split_row": split_row,
                "mixed_illuminant_format": str(scene_get(mixed_scene, "illuminant format")),
                "top_mixed_illuminant_energy": np.mean(mixed_illuminant[top_slice, :, :], axis=(0, 1)),
                "bottom_mixed_illuminant_energy": np.mean(mixed_illuminant[bottom_slice, :, :], axis=(0, 1)),
                "top_source_d65_illuminant_energy": np.mean(daylight_energy[top_slice, :, :], axis=(0, 1)),
                "bottom_source_tungsten_illuminant_energy": np.mean(tungsten_energy[bottom_slice, :, :], axis=(0, 1)),
                "top_mixed_reflectance": np.mean(mixed_reflectance[top_slice, :, :], axis=(0, 1)),
                "bottom_mixed_reflectance": np.mean(mixed_reflectance[bottom_slice, :, :], axis=(0, 1)),
                "top_source_reflectance": np.mean(source_reflectance[top_slice, :, :], axis=(0, 1)),
                "bottom_source_reflectance": np.mean(source_reflectance[bottom_slice, :, :], axis=(0, 1)),
                "mixed_mean_luminance": float(scene_get(mixed_scene, "mean luminance", asset_store=store)),
            },
            context={},
        )

    if case_name == "scene_illuminant_space_small":
        scene = scene_create("frequency orientation", asset_store=store)
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
        row_energy = np.asarray(scene_get(row_scene, "illuminant energy"), dtype=float)
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

        top_band = slice(0, max(1, rows // 8))
        mid_start = max(0, rows // 2 - max(1, rows // 16))
        mid_stop = min(rows, mid_start + max(1, rows // 8))
        mid_band = slice(mid_start, mid_stop)
        bottom_band = slice(rows - max(1, rows // 8), rows)
        center_wave_idx = int(np.argmin(np.abs(wave - 550.0)))

        col_profile = np.mean(col_energy[:, :, center_wave_idx], axis=0)
        final_profile = np.mean(final_energy[:, :, center_wave_idx], axis=0)
        col_profile_norm = col_profile / max(float(np.max(col_profile)), 1e-12)
        final_profile_norm = final_profile / max(float(np.max(final_profile)), 1e-12)
        col_scale_norm = col_scale / max(float(np.max(col_scale)), 1e-12)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "scene_size": np.array([rows, cols], dtype=int),
                "initial_illuminant_photons": illuminant_photons_1d,
                "spatial_spectral_shape": np.array([rows, cols, nwave], dtype=int),
                "row_cct_k": c_temp,
                "row_top_illuminant_energy": np.mean(row_energy[top_band, :, :], axis=(0, 1)),
                "row_mid_illuminant_energy": np.mean(row_energy[mid_band, :, :], axis=(0, 1)),
                "row_bottom_illuminant_energy": np.mean(row_energy[bottom_band, :, :], axis=(0, 1)),
                "source_mean_reflectance": np.mean(source_reflectance, axis=(0, 1)),
                "row_mean_reflectance": np.mean(row_reflectance, axis=(0, 1)),
                "col_scale": col_scale,
                "col_scale_norm": col_scale_norm,
                "col_center_wave_profile_norm": col_profile_norm,
                "col_mean_reflectance": np.mean(col_reflectance, axis=(0, 1)),
                "row_bug_scale": row_bug_scale,
                "final_center_wave_profile_norm": final_profile_norm,
                "final_mean_luminance": float(scene_get(final_scene, "mean luminance", asset_store=store)),
            },
            context={},
        )

    if case_name == "scene_xyz_illuminant_transforms_small":
        scene = scene_create("reflectance chart", asset_store=store)
        scene_d65 = scene_adjust_illuminant(scene.clone(), "D65.mat", asset_store=store)
        scene_tungsten = scene_adjust_illuminant(scene.clone(), "Tungsten.mat", asset_store=store)

        xyz_d65 = np.asarray(scene_get(scene_d65, "xyz", asset_store=store), dtype=float)
        xyz_tungsten = np.asarray(scene_get(scene_tungsten, "xyz", asset_store=store), dtype=float)
        xyz_d65_xw, rows, cols, _ = rgb_to_xw_format(xyz_d65)
        xyz_tungsten_xw, _, _, _ = rgb_to_xw_format(xyz_tungsten)
        xyz_d65_mean = np.mean(xyz_d65_xw, axis=0)
        xyz_tungsten_mean = np.mean(xyz_tungsten_xw, axis=0)

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

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.array([rows, cols], dtype=int),
                "xyz_d65_mean_norm": xyz_d65_mean / max(float(np.sum(xyz_d65_mean)), 1e-12),
                "xyz_tungsten_mean_norm": xyz_tungsten_mean / max(float(np.sum(xyz_tungsten_mean)), 1e-12),
                "full_transform": full_transform,
                "diagonal_transform": diagonal_transform,
                "predicted_full_rmse_ratio": np.sqrt(np.mean(np.square(predicted_full - xyz_d65_xw), axis=0))
                / np.maximum(xyz_d65_mean, 1e-12),
                "predicted_diagonal_rmse_ratio": np.sqrt(np.mean(np.square(predicted_diagonal - xyz_d65_xw), axis=0))
                / np.maximum(xyz_d65_mean, 1e-12),
            },
            context={},
        )

    if case_name == "color_illuminant_transforms_small":
        scene = scene_create("reflectance chart", asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        bb_range = np.arange(3500.0, 8000.0 + 0.1, 500.0, dtype=float)
        nbb = bb_range.size

        transform_list = np.zeros((9, nbb * nbb), dtype=float)
        column = 0
        for source_temp in bb_range:
            source_scene = scene_adjust_illuminant(scene.clone(), blackbody(wave, source_temp), asset_store=store)
            xyz_source = np.asarray(scene_get(source_scene, "xyz", asset_store=store), dtype=float)
            xyz_source_xw, rows, cols, _ = rgb_to_xw_format(xyz_source)
            for target_temp in bb_range:
                target_scene = scene_adjust_illuminant(scene.clone(), blackbody(wave, target_temp), asset_store=store)
                xyz_target = np.asarray(scene_get(target_scene, "xyz", asset_store=store), dtype=float)
                xyz_target_xw, _, _, _ = rgb_to_xw_format(xyz_target)
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

        transform_diagonal_terms = transform_list[[0, 4, 8], :]
        buddha_similarity = (transform_list.T @ buddha).reshape(nbb, nbb, order="F")
        flower_similarity = (transform_list.T @ flower).reshape(nbb, nbb, order="F")

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "bb_range": bb_range,
                "scene_size": np.array([rows, cols], dtype=int),
                "transform_diagonal_terms": transform_diagonal_terms,
                "buddha_similarity": buddha_similarity,
                "flower_similarity": flower_similarity,
            },
            context={},
        )

    if case_name == "chromatic_spatial_chart_small":
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
        img_row = img_row * ((256.0 - 1.0) / max(float(np.max(img_row)), 1e-12)) + 1.0
        img_row = img_row / max(float(np.max(img_row)), 1e-12) + 2.0

        channel_rows = np.stack(
            [c_weights[idx] * np.cos(2.0 * np.pi * c_freq[idx] * r_samples / n_rows) + 2.0 for idx in range(3)],
            axis=0,
        )
        rgb = np.zeros((n_rows, n_cols, 3), dtype=float)
        for idx in range(3):
            rgb[:, :, idx] = channel_rows[idx, :, None] * img_row[None, :]
        rgb /= max(float(np.max(rgb)), 1e-12)

        border = np.zeros((n_rows // 4, n_cols, 3), dtype=float)
        border_template = np.full((n_rows // 4, 1), 0.5, dtype=float) @ img_row[None, :]
        border_template /= max(float(np.max(border_template)), 1e-12)
        for idx in range(3):
            border[:, :, idx] = border_template
        rgb = np.concatenate([border, rgb, border], axis=0)

        center_row = rgb.shape[0] // 2
        center_col = rgb.shape[1] // 2
        scene = scene_from_file(rgb, "rgb", 100.0, "LCD-Apple.mat", asset_store=store)
        scene_wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        scene_photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        scene_luminance = np.asarray(scene_get(scene, "luminance", asset_store=store), dtype=float)
        scene_mean_photons = np.mean(scene_photons, axis=(0, 1))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "source_rgb_size": np.array(rgb.shape[:2], dtype=int),
                "source_channel_means": np.mean(rgb, axis=(0, 1), dtype=float),
                "source_center_row_rgb": np.asarray(rgb[center_row, :, :], dtype=float),
                "source_center_col_rgb": np.asarray(rgb[:, center_col, :], dtype=float),
                "scene_size": np.array(scene_get(scene, "size"), dtype=int),
                "scene_wave": scene_wave,
                "scene_mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "scene_mean_photons_norm": scene_mean_photons / max(float(np.mean(scene_mean_photons)), 1e-12),
                "scene_center_row_luminance_norm": scene_luminance[center_row, :]
                / max(float(np.max(scene_luminance[center_row, :])), 1e-12),
                "scene_center_col_luminance_norm": scene_luminance[:, center_col]
                / max(float(np.max(scene_luminance[:, center_col])), 1e-12),
            },
            context={},
        )

    if case_name == "color_constancy_small":
        c_temps = np.flip(1.0 / np.linspace(1.0 / 7000.0, 1.0 / 3000.0, 15, dtype=float))

        stuffed_scene = scene_from_file(
            store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
            "spectral",
            asset_store=store,
        )
        stuffed_wave = np.asarray(scene_get(stuffed_scene, "wave"), dtype=float).reshape(-1)
        stuffed_means = np.zeros((c_temps.size, 3), dtype=float)
        stuffed_centers = np.zeros((c_temps.size, 3), dtype=float)
        stuffed_mean_luminance = np.zeros(c_temps.size, dtype=float)

        for index, c_temp in enumerate(c_temps):
            bb = blackbody(stuffed_wave, c_temp, kind="energy")
            stuffed_scene = scene_adjust_illuminant(stuffed_scene, bb, asset_store=store)
            rgb = np.asarray(scene_get(stuffed_scene, "rgb", asset_store=store), dtype=float)
            center_row = rgb.shape[0] // 2
            center_col = rgb.shape[1] // 2
            stuffed_means[index, :] = _channel_normalize(np.mean(rgb, axis=(0, 1), dtype=float))
            stuffed_centers[index, :] = _channel_normalize(rgb[center_row, center_col, :])
            stuffed_mean_luminance[index] = float(scene_get(stuffed_scene, "mean luminance", asset_store=store))

        uniform_scene = scene_create("uniform d65", 512, asset_store=store)
        uniform_wave = np.asarray(scene_get(uniform_scene, "wave"), dtype=float).reshape(-1)
        uniform_means = np.zeros((c_temps.size, 3), dtype=float)
        uniform_centers = np.zeros((c_temps.size, 3), dtype=float)
        uniform_mean_luminance = np.zeros(c_temps.size, dtype=float)

        for index, c_temp in enumerate(c_temps):
            bb = blackbody(uniform_wave, c_temp, kind="energy")
            uniform_scene = scene_adjust_illuminant(uniform_scene, bb, asset_store=store)
            rgb = np.asarray(scene_get(uniform_scene, "rgb", asset_store=store), dtype=float)
            center_row = rgb.shape[0] // 2
            center_col = rgb.shape[1] // 2
            uniform_means[index, :] = _channel_normalize(np.mean(rgb, axis=(0, 1), dtype=float))
            uniform_centers[index, :] = _channel_normalize(rgb[center_row, center_col, :])
            uniform_mean_luminance[index] = float(scene_get(uniform_scene, "mean luminance", asset_store=store))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "c_temps": c_temps,
                "stuffed_scene_size": np.array(scene_get(stuffed_scene, "size"), dtype=int),
                "stuffed_wave": stuffed_wave,
                "stuffed_mean_luminance": stuffed_mean_luminance,
                "stuffed_mean_rgb_norm": stuffed_means,
                "stuffed_center_rgb_norm": stuffed_centers,
                "uniform_scene_size": np.array(scene_get(uniform_scene, "size"), dtype=int),
                "uniform_wave": uniform_wave,
                "uniform_mean_luminance": uniform_mean_luminance,
                "uniform_mean_rgb_norm": uniform_means,
                "uniform_center_rgb_norm": uniform_centers,
            },
            context={},
        )

    if case_name == "rgb_color_temperature_small":
        def _estimate(scene_name: str) -> dict[str, Any]:
            scene = scene_create(scene_name, asset_store=store)
            oi = oi_compute(oi_create(asset_store=store), scene)
            sensor = sensor_create(asset_store=store)
            sensor = sensor_set(sensor, "fov", float(scene_get(scene, "fov")), oi)
            sensor = sensor_compute(sensor, oi)
            ip = ip_compute(ip_create(asset_store=store), sensor, asset_store=store)
            srgb = np.asarray(ip_get(ip, "srgb"), dtype=float)
            c_temp, c_table = srgb_to_color_temp(srgb, return_table=True, asset_store=store)
            return {
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "ip_size": np.asarray(ip_get(ip, "size"), dtype=int),
                "c_temp": float(c_temp),
                "srgb_mean_norm": _channel_normalize(np.mean(srgb, axis=(0, 1), dtype=float)),
                "c_table": np.asarray(c_table, dtype=float),
            }

        tungsten = _estimate("macbeth tungsten")
        d65 = _estimate("macbeth d65")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "tungsten_scene_size": tungsten["scene_size"],
                "tungsten_ip_size": tungsten["ip_size"],
                "tungsten_c_temp": tungsten["c_temp"],
                "tungsten_srgb_mean_norm": tungsten["srgb_mean_norm"],
                "d65_scene_size": d65["scene_size"],
                "d65_ip_size": d65["ip_size"],
                "d65_c_temp": d65["c_temp"],
                "d65_srgb_mean_norm": d65["srgb_mean_norm"],
                "c_table_temps": d65["c_table"][:, 0],
                "c_table_xy": d65["c_table"][:, 1:3],
            },
            context={},
        )

    if case_name == "srgb_gamut_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)

        def _closed_xy(parameters: np.ndarray) -> np.ndarray:
            xy = np.asarray(parameters, dtype=float)
            return np.concatenate((xy, xy[:, :1]), axis=1)

        def _xy_payload(
            files: list[str],
            samples: list[np.ndarray],
            illuminant: Any,
        ) -> dict[str, Any]:
            scene, sample_lists, reflectances, rc_size = scene_reflectance_chart(
                files,
                samples,
                32,
                wave,
                True,
                asset_store=store,
            )
            scene = scene_adjust_illuminant(scene, illuminant, asset_store=store)
            light = np.asarray(scene_get(scene, "illuminant energy"), dtype=float).reshape(-1)
            xyz = np.asarray(
                xyz_from_energy((light.reshape(-1, 1) * reflectances).T, wave, asset_store=store),
                dtype=float,
            )
            return {
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "rc_size": np.asarray(rc_size, dtype=int),
                "sample_counts": np.asarray([len(sample_list) for sample_list in sample_lists], dtype=int),
                "reflectance_size": np.asarray(reflectances.shape, dtype=int),
                "xy": np.asarray(chromaticity_xy(xyz), dtype=float),
            }

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

        natural_d65 = _xy_payload(natural_files, natural_samples, "D65.mat")
        natural_yellow = _xy_payload(natural_files, natural_samples, blackbody(wave, 3000.0, kind="energy"))
        synthetic_d65 = _xy_payload(synthetic_files, synthetic_samples, "D65.mat")

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "srgb_xy_loop": _closed_xy(srgb_parameters("chromaticity")),
                "adobergb_xy_loop": _closed_xy(adobergb_parameters("chromaticity")),
                "natural_scene_size": natural_d65["scene_size"],
                "natural_rc_size": natural_d65["rc_size"],
                "natural_sample_counts": natural_d65["sample_counts"],
                "natural_reflectance_size": natural_d65["reflectance_size"],
                "natural_d65_xy": natural_d65["xy"],
                "natural_yellow_xy": natural_yellow["xy"],
                "synthetic_scene_size": synthetic_d65["scene_size"],
                "synthetic_rc_size": synthetic_d65["rc_size"],
                "synthetic_sample_counts": synthetic_d65["sample_counts"],
                "synthetic_reflectance_size": synthetic_d65["reflectance_size"],
                "synthetic_d65_xy": synthetic_d65["xy"],
            },
            context={},
        )

    if case_name == "scene_reflectance_charts_small":
        default_scene = scene_create("reflectance chart", asset_store=store)
        default_chart = scene_get(default_scene, "chart parameters")

        s_files = [
            "MunsellSamples_Vhrel.mat",
            "Food_Vhrel.mat",
            "DupontPaintChip_Vhrel.mat",
            "HyspexSkinReflectance.mat",
        ]
        s_samples = [12, 12, 24, 24]
        p_size = 24
        custom_scene = scene_create("reflectance chart", p_size, s_samples, s_files, None, False, "no replacement", asset_store=store)
        custom_chart = scene_get(custom_scene, "chart parameters")
        wave = np.asarray(scene_get(custom_scene, "wave"), dtype=float).reshape(-1)

        d65_scene = scene_adjust_illuminant(custom_scene.clone(), "D65", asset_store=store)
        d65_illuminant = np.asarray(scene_get(d65_scene, "illuminant energy"), dtype=float).reshape(-1)

        gray_scene, _, gray_reflectances, gray_rc = scene_reflectance_chart(
            s_files,
            s_samples,
            p_size,
            wave,
            True,
            asset_store=store,
        )
        gray_chart = scene_get(gray_scene, "chart parameters")

        original_scene, stored_samples, _, _ = scene_reflectance_chart(
            s_files,
            s_samples,
            p_size,
            None,
            True,
            asset_store=store,
        )
        replica_scene, _, _, _ = scene_reflectance_chart(
            s_files,
            stored_samples,
            p_size,
            None,
            True,
            asset_store=store,
        )
        original_photons = np.asarray(scene_get(original_scene, "photons"), dtype=float)
        replica_photons = np.asarray(scene_get(replica_scene, "photons"), dtype=float)

        gray_idx_map = np.asarray(gray_chart["rIdxMap"], dtype=int)
        gray_column = int(gray_chart["rowcol"][1]) - 1
        gray_mask = gray_idx_map[:, gray_column * p_size : (gray_column + 1) * p_size] > 0
        gray_patch = np.asarray(scene_get(gray_scene, "photons"), dtype=float)[
            :,
            gray_column * p_size : (gray_column + 1) * p_size,
            :,
        ]
        gray_mean_spd = np.mean(gray_patch[gray_mask], axis=0, dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "default_scene_size": np.asarray(scene_get(default_scene, "size"), dtype=int),
                "default_chart_rowcol": np.asarray(default_chart["rowcol"], dtype=int),
                "default_sample_counts": np.asarray([len(item) for item in default_chart["sSamples"]], dtype=int),
                "default_mean_luminance": float(scene_get(default_scene, "mean luminance", asset_store=store)),
                "custom_scene_size": np.asarray(scene_get(custom_scene, "size"), dtype=int),
                "custom_chart_rowcol": np.asarray(custom_chart["rowcol"], dtype=int),
                "custom_sample_counts": np.asarray([len(item) for item in custom_chart["sSamples"]], dtype=int),
                "custom_reflectance_shape": np.asarray((wave.size, sum(s_samples)), dtype=int),
                "custom_idx_map_unique": np.asarray(np.unique(custom_chart["rIdxMap"]), dtype=int),
                "d65_illuminant_norm": d65_illuminant / max(float(np.max(d65_illuminant)), 1e-12),
                "d65_mean_luminance": float(scene_get(d65_scene, "mean luminance", asset_store=store)),
                "gray_scene_size": np.asarray(scene_get(gray_scene, "size"), dtype=int),
                "gray_chart_rowcol": np.asarray(gray_rc, dtype=int),
                "gray_reflectance_shape": np.asarray(gray_reflectances.shape, dtype=int),
                "gray_mean_spd_norm": gray_mean_spd / max(float(np.max(gray_mean_spd)), 1e-12),
                "stored_sample_counts": np.asarray([len(item) for item in stored_samples], dtype=int),
                "replica_photons_nmae": float(np.mean(np.abs(replica_photons - original_photons)))
                / max(float(np.mean(np.abs(original_photons))), 1e-12),
            },
            context={},
        )

    if case_name == "scene_change_illuminant_small":
        default_scene = scene_create(asset_store=store)
        wave = np.asarray(scene_get(default_scene, "wave"), dtype=float).reshape(-1)
        default_illuminant_photons = np.asarray(scene_get(default_scene, "illuminant photons"), dtype=float).reshape(-1)

        tungsten_energy = np.asarray(ie_read_spectra("Tungsten.mat", wave, asset_store=store), dtype=float).reshape(-1)
        tungsten_scene = scene_adjust_illuminant(default_scene.clone(), tungsten_energy, asset_store=store)
        tungsten_scene = scene_set(tungsten_scene, "illuminant comment", "Tungsten illuminant")
        tungsten_illuminant_photons = np.asarray(scene_get(tungsten_scene, "illuminant photons"), dtype=float).reshape(-1)

        stuffed_scene = scene_from_file(
            store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
            "multispectral",
            asset_store=store,
        )
        stuffed_illuminant_energy = np.asarray(scene_get(stuffed_scene, "illuminant energy"), dtype=float).reshape(-1)

        equal_energy_scene = scene_adjust_illuminant(stuffed_scene.clone(), "equalEnergy.mat", asset_store=store)
        equal_energy_illuminant = np.asarray(scene_get(equal_energy_scene, "illuminant energy"), dtype=float).reshape(-1)
        equal_energy_rgb = np.asarray(scene_get(equal_energy_scene, "rgb", asset_store=store), dtype=float)

        horizon_scene = scene_adjust_illuminant(stuffed_scene.clone(), "illHorizon-20180220.mat", asset_store=store)
        horizon_illuminant = np.asarray(scene_get(horizon_scene, "illuminant energy"), dtype=float).reshape(-1)
        horizon_rgb = np.asarray(scene_get(horizon_scene, "rgb", asset_store=store), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "default_scene_size": np.asarray(scene_get(default_scene, "size"), dtype=int),
                "default_mean_luminance": float(scene_get(default_scene, "mean luminance", asset_store=store)),
                "default_illuminant_photons_norm": default_illuminant_photons / max(float(np.max(default_illuminant_photons)), 1e-12),
                "tungsten_mean_luminance": float(scene_get(tungsten_scene, "mean luminance", asset_store=store)),
                "tungsten_comment": str(scene_get(tungsten_scene, "illuminant comment")),
                "tungsten_illuminant_photons_norm": tungsten_illuminant_photons / max(float(np.max(tungsten_illuminant_photons)), 1e-12),
                "stuffed_scene_size": np.asarray(scene_get(stuffed_scene, "size"), dtype=int),
                "stuffed_mean_luminance": float(scene_get(stuffed_scene, "mean luminance", asset_store=store)),
                "stuffed_illuminant_energy_norm": stuffed_illuminant_energy / max(float(np.max(stuffed_illuminant_energy)), 1e-12),
                "equal_energy_mean_luminance": float(scene_get(equal_energy_scene, "mean luminance", asset_store=store)),
                "equal_energy_illuminant_norm": equal_energy_illuminant / max(float(np.max(equal_energy_illuminant)), 1e-12),
                "equal_energy_mean_rgb_norm": _channel_normalize(np.mean(equal_energy_rgb, axis=(0, 1), dtype=float)),
                "horizon_mean_luminance": float(scene_get(horizon_scene, "mean luminance", asset_store=store)),
                "horizon_illuminant_norm": horizon_illuminant / max(float(np.max(horizon_illuminant)), 1e-12),
                "horizon_mean_rgb_norm": _channel_normalize(np.mean(horizon_rgb, axis=(0, 1), dtype=float)),
            },
            context={},
        )

    if case_name == "scene_data_extraction_plotting_small":
        scene = scene_create("macbethd65", asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        center_row = int(round(float(scene_get(scene, "rows")) / 2.0))
        line_data, _ = scene_plot(scene, "luminance hline", [1, center_row], asset_store=store)
        illuminant_data, _ = scene_plot(scene, "illuminant energy", asset_store=store)

        rect = np.array([51, 35, 10, 11], dtype=int)
        roi_locs = ie_rect2_locs(rect)
        energy_plot, _ = scene_plot(scene, "radiance energy roi", roi_locs, asset_store=store)
        photons_plot, _ = scene_plot(scene, "radiance photons roi", roi_locs, asset_store=store)
        reflectance_plot, _ = scene_plot(scene, "reflectance", roi_locs, asset_store=store)

        photons_manual = np.mean(np.asarray(vc_get_roi_data(scene, roi_locs, "photons"), dtype=float), axis=0)
        energy_manual = np.mean(np.asarray(vc_get_roi_data(scene, roi_locs, "energy"), dtype=float), axis=0)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": wave,
                "center_row": center_row,
                "luminance_hline_pos_mm": np.asarray(line_data["pos"], dtype=float).reshape(-1),
                "luminance_hline_norm": _channel_normalize(line_data["data"]),
                "illuminant_energy_norm": _channel_normalize(illuminant_data["energy"]),
                "roi_rect": rect,
                "roi_count": int(roi_locs.shape[0]),
                "roi_energy_mean": float(np.mean(energy_manual)),
                "roi_energy_norm": _channel_normalize(energy_plot["energy"]),
                "roi_energy_manual_norm": _channel_normalize(energy_manual),
                "roi_energy_plot_manual_max_abs": float(
                    np.max(np.abs(np.asarray(energy_plot["energy"], dtype=float).reshape(-1) - energy_manual))
                ),
                "roi_photons_mean": float(np.mean(photons_manual)),
                "roi_photons_norm": _channel_normalize(photons_plot["photons"]),
                "roi_photons_manual_norm": _channel_normalize(photons_manual),
                "roi_photons_plot_manual_max_abs": float(
                    np.max(np.abs(np.asarray(photons_plot["photons"], dtype=float).reshape(-1) - photons_manual))
                ),
                "roi_reflectance_mean": float(np.mean(np.asarray(reflectance_plot["reflectance"], dtype=float))),
                "roi_reflectance_norm": _channel_normalize(reflectance_plot["reflectance"]),
            },
            context={},
        )

    if case_name == "scene_monochrome_small":
        display = display_create("crt", asset_store=store)
        display_wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
        white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)

        scene = scene_from_file("cameraman.tif", "monochrome", 100.0, "crt", asset_store=store)
        scene_wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        center_row = photons.shape[0] // 2
        center_col = photons.shape[1] // 2
        source_mean_spd = np.mean(photons, axis=(0, 1), dtype=float)
        source_center_spd = photons[center_row, center_col, :]

        adjusted_scene = scene_adjust_illuminant(
            scene.clone(),
            blackbody(scene_wave, 6500.0, kind="energy"),
            asset_store=store,
        )
        adjusted_illuminant_energy = np.asarray(scene_get(adjusted_scene, "illuminant energy"), dtype=float).reshape(-1)
        adjusted_photons = np.asarray(scene_get(adjusted_scene, "photons"), dtype=float)
        adjusted_mean_spd = np.mean(adjusted_photons, axis=(0, 1), dtype=float)
        adjusted_center_spd = adjusted_photons[center_row, center_col, :]
        adjusted_rgb = np.asarray(scene_get(adjusted_scene, "rgb", asset_store=store), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "display_wave": display_wave,
                "display_white_spd_norm": _channel_normalize(white_spd),
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_wave": scene_wave,
                "scene_mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "scene_illuminant_energy_norm": _channel_normalize(scene_get(scene, "illuminant energy")),
                "source_mean_spd_norm": _channel_normalize(source_mean_spd),
                "source_center_spd_norm": _channel_normalize(source_center_spd),
                "adjusted_mean_luminance": float(scene_get(adjusted_scene, "mean luminance", asset_store=store)),
                "adjusted_illuminant_energy_norm": _channel_normalize(adjusted_illuminant_energy),
                "adjusted_mean_spd_norm": _channel_normalize(adjusted_mean_spd),
                "adjusted_center_spd_norm": _channel_normalize(adjusted_center_spd),
                "adjusted_mean_rgb_norm": _channel_normalize(np.mean(adjusted_rgb, axis=(0, 1), dtype=float)),
            },
            context={},
        )

    if case_name == "scene_slanted_bar_small":
        scene = scene_create("slantedBar", 256, 2.6, 2.0, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        luminance = np.asarray(scene_get(scene, "luminance", asset_store=store), dtype=float)
        illuminant_roi, _ = scene_plot(scene, "illuminant energy roi", asset_store=store)

        d65_scene = scene_adjust_illuminant(scene.clone(), "D65.mat", asset_store=store)
        d65_illuminant_roi, _ = scene_plot(d65_scene, "illuminant energy roi", asset_store=store)

        alt_scene = scene_create("slantedBar", 128, 3.6, 0.5, asset_store=store)
        alt_luminance = np.asarray(scene_get(alt_scene, "luminance", asset_store=store), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "illuminant_energy_roi_norm": _channel_normalize(illuminant_roi["energy"]),
                "center_row_luminance_norm": _channel_normalize(luminance[luminance.shape[0] // 2, :]),
                "center_col_luminance_norm": _channel_normalize(luminance[:, luminance.shape[1] // 2]),
                "d65_mean_luminance": float(scene_get(d65_scene, "mean luminance", asset_store=store)),
                "d65_illuminant_energy_roi_norm": _channel_normalize(d65_illuminant_roi["energy"]),
                "alt_scene_size": np.asarray(scene_get(alt_scene, "size"), dtype=int),
                "alt_fov_deg": float(scene_get(alt_scene, "fov")),
                "alt_mean_luminance": float(scene_get(alt_scene, "mean luminance", asset_store=store)),
                "alt_center_row_luminance_norm": _channel_normalize(alt_luminance[alt_luminance.shape[0] // 2, :]),
                "alt_center_col_luminance_norm": _channel_normalize(alt_luminance[:, alt_luminance.shape[1] // 2]),
            },
            context={},
        )

    if case_name == "scene_harmonics_script_small":
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

        sizes = np.zeros((len(cases), 2), dtype=int)
        mean_luminance = np.zeros(len(cases), dtype=float)
        center_rows = np.zeros((len(cases), 128), dtype=float)
        center_cols = np.zeros((len(cases), 128), dtype=float)
        wave = None
        for index, params in enumerate(cases):
            scene = scene_create("harmonic", params, asset_store=store)
            luminance = np.asarray(scene_get(scene, "luminance", asset_store=store), dtype=float)
            sizes[index, :] = np.asarray(scene_get(scene, "size"), dtype=int)
            mean_luminance[index] = float(scene_get(scene, "mean luminance", asset_store=store))
            center_rows[index, :] = _channel_normalize(luminance[luminance.shape[0] // 2, :])
            center_cols[index, :] = _channel_normalize(luminance[:, luminance.shape[1] // 2])
            if wave is None:
                wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "scene_sizes": sizes,
                "mean_luminance": mean_luminance,
                "center_row_luminance_norm": center_rows,
                "center_col_luminance_norm": center_cols,
            },
            context={},
        )

    if case_name == "surface_munsell_small":
        munsell = store.load_mat("data/surfaces/charts/munsell.mat")["munsell"]
        xyz = np.asarray(munsell.XYZ, dtype=float)
        lab = np.asarray(munsell.LAB, dtype=float)
        wavelength = np.asarray(munsell.wavelength, dtype=float).reshape(-1)
        illuminant = np.asarray(munsell.illuminant, dtype=float).reshape(-1)
        hues = np.asarray(munsell.hue, dtype=object).reshape(-1)
        values = np.asarray(munsell.value, dtype=float).reshape(-1)
        angles = np.asarray(munsell.angle, dtype=float).reshape(-1)

        xyz_image = xyz.reshape(261, 9, 3, order="F")
        srgb = np.asarray(xyz_to_srgb(xyz_image), dtype=float)
        xy = np.asarray(chromaticity_xy(xyz), dtype=float)

        selected_rgb = np.vstack(
            [
                srgb[0, 0, :],
                srgb[srgb.shape[0] // 2, srgb.shape[1] // 2, :],
                srgb[-1, -1, :],
                srgb[44, 4, :],
            ]
        )

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xyz_shape": np.asarray(xyz.shape, dtype=int),
                "lab_shape": np.asarray(lab.shape, dtype=int),
                "wavelength": wavelength,
                "illuminant_norm": _channel_normalize(illuminant),
                "srgb_grid_shape": np.asarray(srgb.shape, dtype=int),
                "srgb_mean_rgb": np.mean(srgb, axis=(0, 1), dtype=float),
                "srgb_selected_rgb": selected_rgb,
                "xy_mean": np.mean(xy, axis=0, dtype=float),
                "xy_bounds": np.vstack([np.min(xy, axis=0), np.max(xy, axis=0)]),
                "lab_mean": np.mean(lab, axis=0, dtype=float),
                "lab_bounds": np.vstack([np.min(lab, axis=0), np.max(lab, axis=0)]),
                "first45_hues": np.asarray(hues[:45], dtype=object),
                "first45_values": values[:45],
                "first45_angles": angles[:45],
            },
            context={},
        )

    if case_name == "scene_demo_small":
        scene_macbeth = scene_create("macbethd65", asset_store=store)
        macbeth_wave = np.asarray(scene_get(scene_macbeth, "wave"), dtype=float).reshape(-1)
        macbeth_luminance = np.asarray(scene_get(scene_macbeth, "luminance", asset_store=store), dtype=float)
        macbeth_photons = np.asarray(scene_get(scene_macbeth, "photons"), dtype=float)
        macbeth_mean_photons = np.mean(macbeth_photons, axis=(0, 1), dtype=float)

        macbeth_fov_before = float(scene_get(scene_macbeth, "fov"))
        scene_macbeth_fov20 = scene_set(scene_macbeth.clone(), "fov", 20.0)

        scene_test = scene_create("freq orient pattern", asset_store=store)
        scene_test_wave = np.asarray(scene_get(scene_test, "wave"), dtype=float).reshape(-1)
        scene_test_size = np.asarray(scene_get(scene_test, "size"), dtype=int)
        scene_test_support = dict(scene_get(scene_test, "spatial support linear", "mm"))
        scene_test_bottom_row, _ = scene_plot(scene_test, "luminance hline", scene_test_size, asset_store=store)
        rows_half = int(round(float(scene_get(scene_test, "rows")) / 2.0))
        radiance_line, _ = scene_plot(scene_test, "radiance hline", [1, rows_half], asset_store=store)
        radiance_data = np.asarray(radiance_line["data"], dtype=float)
        wave_index_550 = int(np.argmin(np.abs(scene_test_wave - 550.0)))
        if radiance_data.shape[0] == scene_test_wave.size:
            radiance_550 = radiance_data[wave_index_550, :]
        else:
            radiance_550 = radiance_data[:, wave_index_550]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "macbeth_scene_size": np.asarray(scene_get(scene_macbeth, "size"), dtype=int),
                "macbeth_wave": macbeth_wave,
                "macbeth_mean_luminance": float(scene_get(scene_macbeth, "mean luminance", asset_store=store)),
                "macbeth_luminance_bounds": np.array(
                    [float(np.min(macbeth_luminance)), float(np.max(macbeth_luminance))],
                    dtype=float,
                ),
                "macbeth_center_row_luminance_norm": _channel_normalize(
                    macbeth_luminance[macbeth_luminance.shape[0] // 2, :]
                ),
                "macbeth_photons_shape": np.asarray(macbeth_photons.shape, dtype=int),
                "macbeth_max_photons": float(np.max(macbeth_photons)),
                "macbeth_mean_mean_photons": float(np.mean(macbeth_mean_photons)),
                "macbeth_mean_photons_norm": _channel_normalize(macbeth_mean_photons),
                "macbeth_fov_before": macbeth_fov_before,
                "macbeth_fov_after": float(scene_get(scene_macbeth_fov20, "fov")),
                "freq_scene_size": scene_test_size,
                "freq_scene_wave": scene_test_wave,
                "freq_scene_mean_luminance": float(scene_get(scene_test, "mean luminance", asset_store=store)),
                "freq_scene_rows_half": rows_half,
                "freq_scene_support_x_mm": np.asarray(scene_test_support["x"], dtype=float).reshape(-1),
                "freq_scene_bottom_row_luminance_norm": _channel_normalize(scene_test_bottom_row["data"]),
                "freq_scene_radiance_hline_550_norm": _channel_normalize(radiance_550),
            },
            context={},
        )

    if case_name == "scene_introduction_tutorial_small":
        scene = scene_create("macbeth d65", asset_store=store)
        scene_name = str(scene_get(scene, "name"))
        if scene_name == "Macbeth D65":
            scene_name = "Macbeth (D65)"
        scene_type = str(scene.type)
        hfov_before = float(scene_get(scene, "hfov"))
        scene = scene_set(scene, "hfov", 0.5)
        hfov_after = float(scene_get(scene, "hfov"))

        distance_mm = float(scene_get(scene, "distance", "mm"))
        spacing_before_mm = np.asarray(scene_get(scene, "sample spacing", "mm"), dtype=float).reshape(-1)

        scene_distance = scene_set(scene.clone(), "distance", 0.6)
        spacing_after_distance_mm = np.asarray(
            scene_get(scene_distance, "sample spacing", "mm"),
            dtype=float,
        ).reshape(-1)

        scene_distance_hfov = scene_set(scene_distance.clone(), "hfov", hfov_after / 2.0)
        spacing_after_hfov_mm = np.asarray(
            scene_get(scene_distance_hfov, "sample spacing", "mm"),
            dtype=float,
        ).reshape(-1)
        spacing_after_hfov_um = np.asarray(
            scene_get(scene_distance_hfov, "sample spacing", "um"),
            dtype=float,
        ).reshape(-1)
        spacing_after_hfov_m = np.asarray(
            scene_get(scene_distance_hfov, "sample spacing", "m"),
            dtype=float,
        ).reshape(-1)

        stuffed_scene = scene_from_file(
            store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
            "multispectral",
            asset_store=store,
        )
        stuffed_wave = np.asarray(scene_get(stuffed_scene, "wave"), dtype=float).reshape(-1)
        illuminant_energy_before = np.asarray(scene_get(stuffed_scene, "illuminant energy"), dtype=float).reshape(-1)
        bb_energy = np.asarray(blackbody(stuffed_wave, 5500.0, kind="energy"), dtype=float).reshape(-1)
        adjusted_scene = scene_adjust_illuminant(stuffed_scene.clone(), bb_energy, asset_store=store)
        illuminant_energy_after = np.asarray(scene_get(adjusted_scene, "illuminant energy"), dtype=float).reshape(-1)
        illuminant_photons_after = np.asarray(scene_get(adjusted_scene, "illuminant photons"), dtype=float).reshape(-1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_name": scene_name,
                "scene_type": scene_type,
                "macbeth_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "hfov_before": hfov_before,
                "hfov_after": hfov_after,
                "distance_mm": distance_mm,
                "spacing_before_mm": spacing_before_mm,
                "spacing_after_distance_mm": spacing_after_distance_mm,
                "spacing_after_hfov_mm": spacing_after_hfov_mm,
                "spacing_after_hfov_um": spacing_after_hfov_um,
                "spacing_after_hfov_m": spacing_after_hfov_m,
                "stuffed_scene_size": np.asarray(scene_get(stuffed_scene, "size"), dtype=int),
                "stuffed_wave": stuffed_wave,
                "stuffed_mean_luminance": float(scene_get(stuffed_scene, "mean luminance", asset_store=store)),
                "adjusted_mean_luminance": float(scene_get(adjusted_scene, "mean luminance", asset_store=store)),
                "illuminant_energy_before_norm": _channel_normalize(illuminant_energy_before),
                "blackbody_energy_norm": _channel_normalize(bb_energy),
                "illuminant_energy_after_norm": _channel_normalize(illuminant_energy_after),
                "illuminant_photons_after_norm": _channel_normalize(illuminant_photons_after),
            },
            context={},
        )

    if case_name == "scene_examples_small":
        def _canonical_profile(values: Any, samples: int = 41) -> np.ndarray:
            row = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, row.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, row)

        example_scenes = [
            ("rings_rays", scene_create("rings rays", asset_store=store)),
            (
                "frequency_orientation",
                scene_create(
                    "frequency orientation",
                    {
                        "angles": np.linspace(0.0, np.pi / 2.0, 5),
                        "freqs": np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=float),
                        "blockSize": 64,
                        "contrast": 0.8,
                    },
                    asset_store=store,
                ),
            ),
            (
                "harmonic_a",
                scene_create(
                    "harmonic",
                    {
                        "freq": 1.0,
                        "contrast": 1.0,
                        "ph": 0.0,
                        "ang": 0.0,
                        "row": 64,
                        "col": 64,
                        "GaborFlag": 0.0,
                    },
                    asset_store=store,
                ),
            ),
            (
                "harmonic_b",
                scene_create(
                    "harmonic",
                    {
                        "freq": 1.0,
                        "contrast": 1.0,
                        "ph": 0.0,
                        "ang": 0.0,
                        "row": 64,
                        "col": 64,
                        "GaborFlag": 0.0,
                    },
                    asset_store=store,
                ),
            ),
            ("checkerboard", scene_create("checkerboard", 16, 8, "ep", asset_store=store)),
            ("line_d65", scene_create("lined65", 128, asset_store=store)),
            ("slanted_bar", scene_create("slantedBar", 128, 1.3, asset_store=store)),
            ("grid_lines", scene_create("grid lines", 128, 16, asset_store=store)),
            ("point_array", scene_create("point array", 256, 32, asset_store=store)),
            ("macbeth_tungsten_a", scene_create("macbeth tungsten", 16, np.arange(380.0, 721.0, 5.0), asset_store=store)),
            ("macbeth_tungsten_b", scene_create("macbeth tungsten", 16, np.arange(380.0, 721.0, 5.0), asset_store=store)),
            ("uniform_ee_specify", scene_create("uniformEESpecify", 128, np.arange(380.0, 721.0, 10.0), asset_store=store)),
            ("lstar", scene_create("lstar", np.array([80, 10], dtype=int), 20, 1, asset_store=store)),
            ("exp_ramp", scene_create("exponential intensity ramp", 256, 1024, asset_store=store)),
        ]

        stable_fov_indices = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=int)
        stable_luminance_indices = np.array([0, 1, 2, 3, 4, 6, 9, 10, 11, 13], dtype=int)
        scene_labels: list[str] = []
        scene_sizes = np.zeros((len(example_scenes), 2), dtype=int)
        wave_counts = np.zeros(len(example_scenes), dtype=int)
        mean_luminance = np.zeros(len(example_scenes), dtype=float)
        fov_deg = np.zeros(len(example_scenes), dtype=float)
        luminance_bounds = np.zeros((len(example_scenes), 2), dtype=float)
        center_rows = np.zeros((len(example_scenes), 41), dtype=float)
        center_cols = np.zeros((len(example_scenes), 41), dtype=float)

        for index, (label, scene) in enumerate(example_scenes):
            luminance = np.asarray(scene_get(scene, "luminance", asset_store=store), dtype=float)
            center_row = luminance[luminance.shape[0] // 2, :]
            center_col = luminance[:, luminance.shape[1] // 2]
            scene_labels.append(label)
            scene_sizes[index, :] = np.asarray(scene_get(scene, "size"), dtype=int)
            wave_counts[index] = int(np.asarray(scene_get(scene, "wave"), dtype=float).size)
            mean_luminance[index] = float(scene_get(scene, "mean luminance", asset_store=store))
            fov_deg[index] = float(scene_get(scene, "fov"))
            luminance_bounds[index, :] = np.array([float(np.min(luminance)), float(np.max(luminance))], dtype=float)
            center_rows[index, :] = _canonical_profile(_channel_normalize(center_row))
            center_cols[index, :] = _canonical_profile(_channel_normalize(center_col))

        reflectance_chart = scene_create("reflectance chart", asset_store=store)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_labels": np.asarray(scene_labels, dtype=object),
                "scene_sizes": scene_sizes,
                "scene_wave_counts": wave_counts,
                "scene_mean_luminance_stable": mean_luminance[stable_luminance_indices],
                "scene_fov_deg_stable": fov_deg[stable_fov_indices],
                "scene_luminance_bounds_stable": luminance_bounds[stable_luminance_indices, :],
                "scene_center_row_luminance_norm": center_rows,
                "scene_center_col_luminance_norm": center_cols,
                "reflectance_chart_size": np.asarray(scene_get(reflectance_chart, "size"), dtype=int),
                "reflectance_chart_wave_count": int(np.asarray(scene_get(reflectance_chart, "wave"), dtype=float).size),
                "reflectance_chart_mean_luminance": float(scene_get(reflectance_chart, "mean luminance", asset_store=store)),
                "reflectance_chart_fov_deg": float(scene_get(reflectance_chart, "fov")),
            },
            context={},
        )

    if case_name == "scene_from_rgb_lcd_apple_small":
        display = display_create("LCD-Apple.mat", asset_store=store)
        display_wave = np.asarray(display.fields["wave"], dtype=float).reshape(-1)
        display_spd = np.asarray(display.fields["spd"], dtype=float)
        white_spd = np.sum(display_spd, axis=1)
        white_xy = np.asarray(
            chromaticity_xy(xyz_from_energy(white_spd, display_wave, asset_store=store)),
            dtype=float,
        ).reshape(-1)

        rgb_path = store.resolve("data/images/rgb/eagle.jpg")
        scene = scene_from_file(rgb_path, "rgb", None, "LCD-Apple.mat", asset_store=store)
        scene_wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        scene_photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        scene_mean_photons = np.mean(scene_photons, axis=(0, 1))

        adjusted_scene = scene_adjust_illuminant(
            scene.clone(),
            blackbody(scene_wave, 6500.0, kind="energy"),
            asset_store=store,
        )
        rect = [144, 198, 27, 18]
        roi_mean_reflectance = np.asarray(
            scene_get(adjusted_scene, "roi mean reflectance", rect, asset_store=store),
            dtype=float,
        ).reshape(-1)
        adjusted_illuminant_energy = np.asarray(
            scene_get(adjusted_scene, "illuminant energy"),
            dtype=float,
        ).reshape(-1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "display_wave": display_wave,
                "display_spd": display_spd,
                "white_spd": white_spd,
                "white_xy": white_xy,
                "scene_size": np.array(scene_get(scene, "size"), dtype=int),
                "scene_wave": scene_wave,
                "scene_mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "scene_mean_photons_norm": scene_mean_photons / max(float(np.mean(scene_mean_photons)), 1e-12),
                "adjusted_mean_luminance": float(scene_get(adjusted_scene, "mean luminance", asset_store=store)),
                "adjusted_illuminant_energy_norm": adjusted_illuminant_energy
                / max(float(np.max(adjusted_illuminant_energy)), 1e-12),
                "roi_mean_reflectance": roi_mean_reflectance,
            },
            context={},
        )

    if case_name == "scene_from_multispectral_stuffed_animals_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        scene = scene_from_file(
            store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
            "multispectral",
            None,
            None,
            wave,
            asset_store=store,
        )
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        center_row = (photons.shape[0] - 1) // 2
        center_col = (photons.shape[1] - 1) // 2
        mean_scene_spd = np.mean(photons, axis=(0, 1))
        center_scene_spd = photons[center_row, center_col, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.array(scene_get(scene, "size"), dtype=int),
                "wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "mean_scene_spd_norm": mean_scene_spd / max(float(np.max(mean_scene_spd)), 1e-12),
                "center_scene_spd_norm": center_scene_spd / max(float(np.max(center_scene_spd)), 1e-12),
            },
            context={},
        )

    if case_name == "scene_from_rgb_vs_multispectral_stuffed_animals_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        scene = scene_from_file(
            store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat"),
            "multispectral",
            None,
            None,
            wave,
            asset_store=store,
        )
        scene = scene_adjust_illuminant(
            scene,
            blackbody(np.asarray(scene_get(scene, "wave"), dtype=float), 6500.0, kind="energy"),
            asset_store=store,
        )
        source_rgb = np.asarray(scene_get(scene, "rgb", asset_store=store), dtype=float)
        source_xyz = np.asarray(scene_get(scene, "xyz", asset_store=store), dtype=float)
        source_illuminant_energy = np.asarray(scene_get(scene, "illuminant energy"), dtype=float).reshape(-1)
        source_wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        source_illuminant_xy = np.asarray(
            chromaticity_xy(xyz_from_energy(source_illuminant_energy, source_wave, asset_store=store)),
            dtype=float,
        ).reshape(-1)

        display = display_create("LCD-Apple.mat", asset_store=store)
        mean_luminance = float(scene_get(scene, "mean luminance", asset_store=store))
        reconstructed = scene_from_file(source_rgb, "rgb", mean_luminance, display, asset_store=store)
        reconstructed = scene_adjust_illuminant(
            reconstructed,
            blackbody(np.asarray(scene_get(reconstructed, "wave"), dtype=float), 6500.0, kind="energy"),
            asset_store=store,
        )
        reconstructed = scene_adjust_luminance(reconstructed, mean_luminance, asset_store=store)

        reconstructed_rgb = np.asarray(scene_get(reconstructed, "rgb", asset_store=store), dtype=float)
        reconstructed_xyz = np.asarray(scene_get(reconstructed, "xyz", asset_store=store), dtype=float)
        reconstructed_illuminant_energy = np.asarray(scene_get(reconstructed, "illuminant energy"), dtype=float).reshape(-1)
        reconstructed_wave = np.asarray(scene_get(reconstructed, "wave"), dtype=float).reshape(-1)
        reconstructed_illuminant_xy = np.asarray(
            chromaticity_xy(
                xyz_from_energy(reconstructed_illuminant_energy, reconstructed_wave, asset_store=store),
            ),
            dtype=float,
        ).reshape(-1)
        rgb_channel_corr = np.array(
            [
                np.corrcoef(source_rgb[:, :, channel].reshape(-1), reconstructed_rgb[:, :, channel].reshape(-1))[0, 1]
                for channel in range(source_rgb.shape[2])
            ],
            dtype=float,
        )
        xyz_channel_corr = np.array(
            [
                np.corrcoef(source_xyz[:, :, channel].reshape(-1), reconstructed_xyz[:, :, channel].reshape(-1))[0, 1]
                for channel in range(source_xyz.shape[2])
            ],
            dtype=float,
        )

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "source_size": np.array(scene_get(scene, "size"), dtype=int),
                "source_wave": source_wave,
                "source_mean_luminance": mean_luminance,
                "source_illuminant_xy": source_illuminant_xy,
                "reconstructed_size": np.array(scene_get(reconstructed, "size"), dtype=int),
                "reconstructed_wave": reconstructed_wave,
                "reconstructed_mean_luminance": float(scene_get(reconstructed, "mean luminance", asset_store=store)),
                "reconstructed_illuminant_xy": reconstructed_illuminant_xy,
                "rgb_channel_corr": rgb_channel_corr,
                "xyz_channel_corr": xyz_channel_corr,
            },
            context={},
        )

    if case_name == "scene_reflectance_samples_small":
        wave = np.arange(400.0, 701.0, 5.0, dtype=float)
        random_sources = [
            "MunsellSamples_Vhrel.mat",
            "Food_Vhrel.mat",
            "DupontPaintChip_Vhrel.mat",
            "skin/HyspexSkinReflectance.mat",
        ]
        random_counts = np.array([24, 24, 24, 24], dtype=int)
        random_reflectances, sampled_lists, sampled_wave = ie_reflectance_samples(
            random_sources,
            random_counts,
            wave,
            "no replacement",
            asset_store=store,
        )
        random_reflectances_replay, _, _ = ie_reflectance_samples(
            random_sources,
            sampled_lists,
            wave,
            "no replacement",
            asset_store=store,
        )

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
            asset_store=store,
        )
        explicit_reflectances_replay, _, _ = ie_reflectance_samples(
            explicit_sources,
            stored_lists,
            wave,
            asset_store=store,
        )
        mean_reflectance, singular_values = _reflectance_sample_statistics(explicit_reflectances)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": sampled_wave,
                "no_replacement_shape": np.array(random_reflectances.shape, dtype=int),
                "no_replacement_sample_sizes": np.array([sample.size for sample in sampled_lists], dtype=int),
                "no_replacement_unique_sizes": np.array([np.unique(sample).size for sample in sampled_lists], dtype=int),
                "no_replacement_replay_max_abs": float(
                    np.max(np.abs(random_reflectances_replay - random_reflectances))
                ),
                "explicit_shape": np.array(explicit_reflectances.shape, dtype=int),
                "explicit_sample_sizes": np.array([sample.size for sample in stored_lists], dtype=int),
                "explicit_sample_first_last": np.array(
                    [[int(sample[0]), int(sample[-1])] for sample in stored_lists],
                    dtype=int,
                ),
                "explicit_mean_reflectance_norm": _channel_normalize(mean_reflectance),
                "explicit_singular_values_norm": _channel_normalize(singular_values),
                "explicit_replay_max_abs": float(
                    np.max(np.abs(explicit_reflectances_replay - explicit_reflectances))
                ),
            },
            context={},
        )

    if case_name == "scene_reflectance_chart_basis_functions_small":
        params = {
            "sfiles": [
                "MunsellSamples_Vhrel.mat",
                "Food_Vhrel.mat",
                "skin/HyspexSkinReflectance.mat",
            ],
            "ssamples": [
                np.arange(1, 51, dtype=int),
                np.concatenate((np.arange(1, 28, dtype=int), np.arange(1, 14, dtype=int))),
                np.arange(1, 11, dtype=int),
            ],
            "psize": 24,
            "grayflag": True,
            "sampling": "without replacement",
        }
        scene = scene_create("reflectance chart", params, asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        reflectance = np.asarray(scene_get(scene, "reflectance"), dtype=float)
        _, basis_999, coef_999, var_999 = hc_basis(reflectance, 0.999, "canonical")
        _, basis_95, coef_95, var_95 = hc_basis(reflectance, 0.95, "canonical")
        _, basis_5, coef_5, var_5 = hc_basis(reflectance, 5, "canonical")

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "scene_size": np.array(scene_get(scene, "size"), dtype=int),
                "reflectance_shape": np.array(reflectance.shape, dtype=int),
                "basis_count_999": int(basis_999.shape[1]),
                "var_explained_999": float(var_999),
                "basis_projector_999": _canonicalize_basis_columns(basis_999) @ _canonicalize_basis_columns(basis_999).T,
                "coef_stats_999": _stats_vector(coef_999),
                "basis_count_95": int(basis_95.shape[1]),
                "var_explained_95": float(var_95),
                "basis_projector_95": _canonicalize_basis_columns(basis_95) @ _canonicalize_basis_columns(basis_95).T,
                "coef_stats_95": _stats_vector(coef_95),
                "basis_count_5": int(basis_5.shape[1]),
                "var_explained_5": float(var_5),
                "basis_projector_5": _canonicalize_basis_columns(basis_5) @ _canonicalize_basis_columns(basis_5).T,
                "coef_stats_5": _stats_vector(coef_5),
            },
            context={},
        )

    if case_name == "scene_roi_small":
        scene = scene_create(asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        scene_size = np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1)
        roi = np.rint(np.array([scene_size[0] / 2.0, scene_size[1], 10.0, 10.0], dtype=float)).astype(int)

        roi_photons = np.asarray(scene_get(scene, "roi photons", roi), dtype=float)
        roi_mean_photons = np.asarray(scene_get(scene, "roi mean photons", roi), dtype=float).reshape(-1)
        roi_energy = np.asarray(scene_get(scene, "roi energy", roi), dtype=float)
        roi_mean_energy = np.asarray(scene_get(scene, "roi mean energy", roi), dtype=float).reshape(-1)
        roi_illuminant_photons = np.asarray(scene_get(scene, "roi illuminant photons", roi), dtype=float)
        roi_mean_illuminant_photons = np.asarray(
            scene_get(scene, "roi mean illuminant photons", roi),
            dtype=float,
        ).reshape(-1)
        roi_reflectance_manual = np.divide(
            roi_photons,
            roi_illuminant_photons,
            out=np.zeros_like(roi_photons),
            where=roi_illuminant_photons > 0.0,
        )
        roi_reflectance_direct = np.asarray(scene_get(scene, "roi reflectance", roi), dtype=float)
        roi_mean_reflectance_direct = np.asarray(scene_get(scene, "roi mean reflectance", roi), dtype=float).reshape(-1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "scene_size": scene_size,
                "roi_rect": roi,
                "roi_point_count": int(roi_photons.shape[0]),
                "roi_photons_stats": _stats_vector(roi_photons),
                "roi_mean_photons": roi_mean_photons,
                "roi_energy_stats": _stats_vector(roi_energy),
                "roi_mean_energy": roi_mean_energy,
                "roi_illuminant_photons_stats": _stats_vector(roi_illuminant_photons),
                "roi_mean_illuminant_photons": roi_mean_illuminant_photons,
                "roi_reflectance_stats": _stats_vector(roi_reflectance_direct),
                "roi_reflectance_mean_manual": np.mean(roi_reflectance_manual, axis=0, dtype=float).reshape(-1),
                "roi_mean_reflectance_direct": roi_mean_reflectance_direct,
                "roi_reflectance_manual_vs_direct_max_abs": float(
                    np.max(np.abs(roi_reflectance_manual - roi_reflectance_direct))
                ),
            },
            context={},
        )

    if case_name == "scene_rotate_small":
        scene = scene_create("star pattern", asset_store=store)
        frame_angles_deg = np.array([1.0, 10.0, 25.0, 50.0], dtype=float)

        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            row = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, row.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, row).astype(float)

        rotated_sizes = np.zeros((frame_angles_deg.size, 2), dtype=int)
        mean_luminance = np.zeros(frame_angles_deg.size, dtype=float)
        max_luminance = np.zeros(frame_angles_deg.size, dtype=float)
        center_luminance = np.zeros(frame_angles_deg.size, dtype=float)
        center_rows_norm = np.zeros((frame_angles_deg.size, 129), dtype=float)
        center_cols_norm = np.zeros((frame_angles_deg.size, 129), dtype=float)

        for index, angle_deg in enumerate(frame_angles_deg):
            rotated_scene = scene_rotate(scene, angle_deg)
            luminance = np.asarray(scene_get(rotated_scene, "luminance", asset_store=store), dtype=float)
            center_row = luminance.shape[0] // 2
            center_col = luminance.shape[1] // 2
            rotated_sizes[index, :] = np.asarray(scene_get(rotated_scene, "size"), dtype=int).reshape(-1)
            mean_luminance[index] = float(np.mean(luminance))
            max_luminance[index] = float(np.max(luminance))
            center_luminance[index] = float(luminance[center_row, center_col])
            center_rows_norm[index, :] = _canonical_profile(_channel_normalize(luminance[center_row, :]))
            center_cols_norm[index, :] = _canonical_profile(_channel_normalize(luminance[:, center_col]))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "frame_angles_deg": frame_angles_deg,
                "source_size": np.array(scene_get(scene, "size"), dtype=int),
                "rotated_sizes": rotated_sizes,
                "mean_luminance": mean_luminance,
                "max_luminance": max_luminance,
                "center_luminance": center_luminance,
                "center_rows_norm": center_rows_norm,
                "center_cols_norm": center_cols_norm,
            },
            context={},
        )

    if case_name == "scene_wavelength_small":
        source_scene = scene_create(asset_store=store)
        canonical_name = lambda value: param_format(value).replace("(", "").replace(")", "")
        source_wave = np.asarray(scene_get(source_scene, "wave"), dtype=float).reshape(-1)
        source_photons = np.asarray(scene_get(source_scene, "photons"), dtype=float)
        source_center = source_photons[source_photons.shape[0] // 2, source_photons.shape[1] // 2, :]
        source_mean = np.mean(source_photons, axis=(0, 1), dtype=float)

        fine_scene = scene_set(source_scene.clone(), "wave", np.arange(400.0, 701.0, 5.0, dtype=float))
        fine_scene = scene_set(fine_scene, "name", "5 nm spacing")
        fine_wave = np.asarray(scene_get(fine_scene, "wave"), dtype=float).reshape(-1)
        fine_photons = np.asarray(scene_get(fine_scene, "photons"), dtype=float)
        fine_center = fine_photons[fine_photons.shape[0] // 2, fine_photons.shape[1] // 2, :]
        fine_mean = np.mean(fine_photons, axis=(0, 1), dtype=float)

        narrow_scene = scene_set(fine_scene.clone(), "wave", np.arange(500.0, 601.0, 2.0, dtype=float))
        narrow_scene = scene_set(narrow_scene, "name", "2 nm narrow band spacing")
        narrow_wave = np.asarray(scene_get(narrow_scene, "wave"), dtype=float).reshape(-1)
        narrow_photons = np.asarray(scene_get(narrow_scene, "photons"), dtype=float)
        narrow_center = narrow_photons[narrow_photons.shape[0] // 2, narrow_photons.shape[1] // 2, :]
        narrow_mean = np.mean(narrow_photons, axis=(0, 1), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "source_name": canonical_name(source_scene.name),
                "source_size": np.asarray(scene_get(source_scene, "size"), dtype=int),
                "source_wave": source_wave,
                "source_mean_luminance": float(scene_get(source_scene, "mean luminance", asset_store=store)),
                "source_mean_scene_spd_norm": _channel_normalize(source_mean),
                "source_center_scene_spd_norm": _channel_normalize(source_center),
                "five_nm_name": canonical_name(fine_scene.name),
                "five_nm_size": np.asarray(scene_get(fine_scene, "size"), dtype=int),
                "five_nm_wave": fine_wave,
                "five_nm_mean_luminance": float(scene_get(fine_scene, "mean luminance", asset_store=store)),
                "five_nm_mean_scene_spd_norm": _channel_normalize(fine_mean),
                "five_nm_center_scene_spd_norm": _channel_normalize(fine_center),
                "narrow_name": canonical_name(narrow_scene.name),
                "narrow_size": np.asarray(scene_get(narrow_scene, "size"), dtype=int),
                "narrow_wave": narrow_wave,
                "narrow_mean_luminance": float(scene_get(narrow_scene, "mean luminance", asset_store=store)),
                "narrow_mean_scene_spd_norm": _channel_normalize(narrow_mean),
                "narrow_center_scene_spd_norm": _channel_normalize(narrow_center),
            },
            context={},
        )

    if case_name == "scene_hc_compress_small":
        canonical_name = lambda value: (
            param_format(value).replace("(", "").replace(")", "").replace("_", "").replace("-", "")
        )
        scene_path = store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
        scene = scene_from_file(scene_path, "multispectral", asset_store=store)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
        illuminant = scene_get(scene, "illuminant")
        comment = "Compressed using hcBasis with imgMean)"
        w_list = np.arange(400.0, 701.0, 5.0, dtype=float)

        def _scene_spd_payload(current_scene: Any) -> tuple[np.ndarray, np.ndarray]:
            current_photons = np.asarray(scene_get(current_scene, "photons"), dtype=float)
            center = current_photons[current_photons.shape[0] // 2, current_photons.shape[1] // 2, :]
            mean_spd = np.mean(current_photons, axis=(0, 1), dtype=float)
            return _channel_normalize(mean_spd), _channel_normalize(center)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/hc_compress.mat"

            img_mean_95, img_basis_95, coef_95, _ = hc_basis(photons, 0.95)
            ie_save_multispectral_image(
                output_path,
                coef_95,
                {"basis": img_basis_95, "wave": wave},
                comment,
                img_mean_95,
                illuminant,
                float(scene_get(scene, "fov")),
                float(scene_get(scene, "distance")),
                "hcCompress95",
            )
            scene_95 = scene_from_file(output_path, "multispectral", None, None, w_list, asset_store=store)

            img_mean_99, img_basis_99, coef_99, _ = hc_basis(photons, 0.99)
            ie_save_multispectral_image(
                output_path,
                coef_99,
                {"basis": img_basis_99, "wave": wave},
                comment,
                img_mean_99,
                illuminant,
                float(scene_get(scene, "fov")),
                float(scene_get(scene, "distance")),
                "hcCompress99",
            )
            scene_99 = scene_from_file(output_path, "multispectral", None, None, w_list, asset_store=store)

        mean_95, center_95 = _scene_spd_payload(scene_95)
        mean_99, center_99 = _scene_spd_payload(scene_99)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "source_name": canonical_name(scene.name),
                "source_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "source_wave": wave,
                "source_mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "basis_count_95": int(img_basis_95.shape[1]),
                "scene95_size": np.asarray(scene_get(scene_95, "size"), dtype=int),
                "scene95_wave": np.asarray(scene_get(scene_95, "wave"), dtype=float).reshape(-1),
                "scene95_mean_luminance": float(scene_get(scene_95, "mean luminance", asset_store=store)),
                "scene95_mean_scene_spd_norm": mean_95,
                "scene95_center_scene_spd_norm": center_95,
                "basis_count_99": int(img_basis_99.shape[1]),
                "scene99_size": np.asarray(scene_get(scene_99, "size"), dtype=int),
                "scene99_wave": np.asarray(scene_get(scene_99, "wave"), dtype=float).reshape(-1),
                "scene99_mean_luminance": float(scene_get(scene_99, "mean luminance", asset_store=store)),
                "scene99_mean_scene_spd_norm": mean_99,
                "scene99_center_scene_spd_norm": center_99,
            },
            context={},
        )

    if case_name == "scene_increase_size_small":
        source_scene = scene_create(asset_store=store)
        source_wave = np.asarray(scene_get(source_scene, "wave"), dtype=float).reshape(-1)
        source_photons = np.asarray(scene_get(source_scene, "photons"), dtype=float)
        source_size = np.asarray(scene_get(source_scene, "size"), dtype=int).reshape(-1)
        source_mean_spd = np.mean(source_photons, axis=(0, 1), dtype=float)

        step1_photons = image_increase_image_rgb_size(source_photons, [2, 3])
        scene_step1 = scene_set(source_scene.clone(), "photons", step1_photons)
        step1_size = np.asarray(scene_get(scene_step1, "size"), dtype=int).reshape(-1)
        step1_mean_spd = np.mean(step1_photons, axis=(0, 1), dtype=float)

        step2_photons = image_increase_image_rgb_size(step1_photons, [1, 2])
        scene_step2 = scene_set(scene_step1.clone(), "photons", step2_photons)
        step2_size = np.asarray(scene_get(scene_step2, "size"), dtype=int).reshape(-1)
        step2_mean_spd = np.mean(step2_photons, axis=(0, 1), dtype=float)

        step3_photons = image_increase_image_rgb_size(step2_photons, [3, 1])
        scene_step3 = scene_set(scene_step2.clone(), "photons", step3_photons)
        step3_size = np.asarray(scene_get(scene_step3, "size"), dtype=int).reshape(-1)
        step3_mean_spd = np.mean(step3_photons, axis=(0, 1), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": source_wave,
                "source_size": source_size,
                "source_mean_luminance": float(scene_get(source_scene, "mean luminance", asset_store=store)),
                "source_mean_scene_spd_norm": _channel_normalize(source_mean_spd),
                "step1_size": step1_size,
                "step1_mean_luminance": float(scene_get(scene_step1, "mean luminance", asset_store=store)),
                "step1_mean_scene_spd_norm": _channel_normalize(step1_mean_spd),
                "step1_replay_max_abs": float(np.max(np.abs(step1_photons[::2, ::3, :] - source_photons))),
                "step2_size": step2_size,
                "step2_mean_luminance": float(scene_get(scene_step2, "mean luminance", asset_store=store)),
                "step2_mean_scene_spd_norm": _channel_normalize(step2_mean_spd),
                "step2_replay_max_abs": float(np.max(np.abs(step2_photons[:, ::2, :] - step1_photons))),
                "step3_size": step3_size,
                "step3_mean_luminance": float(scene_get(scene_step3, "mean luminance", asset_store=store)),
                "step3_mean_scene_spd_norm": _channel_normalize(step3_mean_spd),
                "step3_replay_max_abs": float(np.max(np.abs(step3_photons[::3, :, :] - step2_photons))),
                "source_aspect_ratio": float(source_size[1] / source_size[0]),
                "final_aspect_ratio": float(step3_size[1] / step3_size[0]),
            },
            context={},
        )

    if case_name == "scene_render_small":
        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            row = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, row.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, row).astype(float)

        def _render_summary(image: np.ndarray) -> dict[str, Any]:
            image = np.asarray(image, dtype=float)
            center_row = image.shape[0] // 2
            center_col = image.shape[1] // 2
            luma = np.max(image, axis=2)
            return {
                "stats": _stats_vector(image),
                "channel_means": np.mean(image, axis=(0, 1), dtype=float).reshape(-1),
                "center_rgb": image[center_row, center_col, :].reshape(-1),
                "center_row_luma_norm": _canonical_profile(_channel_normalize(luma[center_row, :])),
            }

        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        stuffed_path = store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
        hdr_path = store.resolve("data/images/multispectral/Feng_Office-hdrs.mat")

        daylight_scene = scene_from_file(stuffed_path, "multispectral", None, None, wave, asset_store=store)
        daylight_energy = ie_read_spectra("D75.mat", np.asarray(scene_get(daylight_scene, "wave"), dtype=float), asset_store=store)
        daylight_scene = scene_adjust_illuminant(daylight_scene, daylight_energy, asset_store=store)
        daylight_scene = scene_set(daylight_scene, "illuminantComment", "Daylight (D75) illuminant")
        daylight_render = scene_show_image(daylight_scene, 0, asset_store=store)
        daylight_illuminant = np.asarray(scene_get(daylight_scene, "illuminant photons"), dtype=float)
        if daylight_illuminant.ndim == 3:
            daylight_illuminant = np.mean(daylight_illuminant, axis=(0, 1), dtype=float)
        daylight_illuminant = daylight_illuminant.reshape(-1)

        hdr_scene = scene_from_file(hdr_path, "multispectral", asset_store=store)
        hdr_srgb = scene_show_image(hdr_scene, 0, asset_store=store)
        hdr_res = hdr_render(hdr_srgb)

        standard_scene = scene_from_file(stuffed_path, "multispectral", asset_store=store)
        standard_srgb = scene_show_image(standard_scene, 0, asset_store=store)
        standard_res = hdr_render(standard_srgb)

        daylight_summary = _render_summary(daylight_render)
        hdr_srgb_summary = _render_summary(hdr_srgb)
        hdr_res_summary = _render_summary(hdr_res)
        standard_srgb_summary = _render_summary(standard_srgb)
        standard_res_summary = _render_summary(standard_res)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "daylight_scene_size": np.asarray(scene_get(daylight_scene, "size"), dtype=int).reshape(-1),
                "daylight_wave": np.asarray(scene_get(daylight_scene, "wave"), dtype=float).reshape(-1),
                "daylight_mean_luminance": float(scene_get(daylight_scene, "mean luminance", asset_store=store)),
                "daylight_illuminant_photons_norm": _channel_normalize(daylight_illuminant),
                "daylight_srgb_stats": daylight_summary["stats"],
                "daylight_srgb_channel_means": daylight_summary["channel_means"],
                "daylight_srgb_center_rgb": daylight_summary["center_rgb"],
                "daylight_srgb_center_row_luma_norm": daylight_summary["center_row_luma_norm"],
                "hdr_scene_size": np.asarray(scene_get(hdr_scene, "size"), dtype=int).reshape(-1),
                "hdr_wave": np.asarray(scene_get(hdr_scene, "wave"), dtype=float).reshape(-1),
                "hdr_mean_luminance": float(scene_get(hdr_scene, "mean luminance", asset_store=store)),
                "hdr_srgb_stats": hdr_srgb_summary["stats"],
                "hdr_srgb_channel_means": hdr_srgb_summary["channel_means"],
                "hdr_render_stats": hdr_res_summary["stats"],
                "hdr_render_channel_means": hdr_res_summary["channel_means"],
                "hdr_render_center_rgb": hdr_res_summary["center_rgb"],
                "hdr_render_center_row_luma_norm": hdr_res_summary["center_row_luma_norm"],
                "hdr_render_delta_mean_abs": float(np.mean(np.abs(hdr_res - hdr_srgb))),
                "standard_scene_size": np.asarray(scene_get(standard_scene, "size"), dtype=int).reshape(-1),
                "standard_wave": np.asarray(scene_get(standard_scene, "wave"), dtype=float).reshape(-1),
                "standard_mean_luminance": float(scene_get(standard_scene, "mean luminance", asset_store=store)),
                "standard_srgb_stats": standard_srgb_summary["stats"],
                "standard_srgb_channel_means": standard_srgb_summary["channel_means"],
                "standard_render_stats": standard_res_summary["stats"],
                "standard_render_channel_means": standard_res_summary["channel_means"],
                "standard_render_center_rgb": standard_res_summary["center_rgb"],
                "standard_render_center_row_luma_norm": standard_res_summary["center_row_luma_norm"],
                "standard_render_delta_mean_abs": float(np.mean(np.abs(standard_res - standard_srgb))),
            },
            context={},
        )

    if case_name == "scene_rgb2radiance_displays_small":
        def _display_scene_payload(display_name: str) -> dict[str, Any]:
            display = display_create(display_name, asset_store=store)
            wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
            spd = np.asarray(display_get(display, "spd"), dtype=float)
            white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)
            white_xy = np.asarray(
                chromaticity_xy(xyz_from_energy(white_spd, wave, asset_store=store)),
                dtype=float,
            ).reshape(-1)
            primary_xy = np.vstack(
                [
                    np.asarray(
                        chromaticity_xy(xyz_from_energy(spd[:, index], wave, asset_store=store)),
                        dtype=float,
                    ).reshape(-1)
                    for index in range(spd.shape[1])
                ]
            )

            scene = scene_from_file("macbeth.tif", "rgb", None, display_name, asset_store=store)
            photons = np.asarray(scene_get(scene, "photons"), dtype=float)
            mean_scene_spd = np.mean(photons, axis=(0, 1), dtype=float)
            rendered_rgb = np.asarray(scene_get(scene, "rgb", asset_store=store), dtype=float)

            return {
                "wave": wave,
                "spd_shape": np.array(spd.shape, dtype=int),
                "white_xy": white_xy,
                "primary_xy": primary_xy,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1),
                "mean_luminance": float(scene_get(scene, "mean luminance", asset_store=store)),
                "mean_scene_spd_norm": _channel_normalize(mean_scene_spd),
                "illuminant_energy_norm": _channel_normalize(np.asarray(scene_get(scene, "illuminant energy"), dtype=float).reshape(-1)),
                "rgb_stats": _stats_vector(rendered_rgb),
                "rgb_channel_means": np.mean(rendered_rgb, axis=(0, 1), dtype=float).reshape(-1),
            }

        oled = _display_scene_payload("OLED-Sony.mat")
        lcd = _display_scene_payload("LCD-Apple.mat")
        crt = _display_scene_payload("CRT-Dell.mat")

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "oled_wave": oled["wave"],
                "oled_spd_shape": oled["spd_shape"],
                "oled_white_xy": oled["white_xy"],
                "oled_primary_xy": oled["primary_xy"],
                "oled_scene_size": oled["scene_size"],
                "oled_mean_luminance": oled["mean_luminance"],
                "oled_mean_scene_spd_norm": oled["mean_scene_spd_norm"],
                "oled_illuminant_energy_norm": oled["illuminant_energy_norm"],
                "oled_rgb_stats": oled["rgb_stats"],
                "oled_rgb_channel_means": oled["rgb_channel_means"],
                "lcd_wave": lcd["wave"],
                "lcd_spd_shape": lcd["spd_shape"],
                "lcd_white_xy": lcd["white_xy"],
                "lcd_primary_xy": lcd["primary_xy"],
                "lcd_scene_size": lcd["scene_size"],
                "lcd_mean_luminance": lcd["mean_luminance"],
                "lcd_mean_scene_spd_norm": lcd["mean_scene_spd_norm"],
                "lcd_illuminant_energy_norm": lcd["illuminant_energy_norm"],
                "lcd_rgb_stats": lcd["rgb_stats"],
                "lcd_rgb_channel_means": lcd["rgb_channel_means"],
                "crt_wave": crt["wave"],
                "crt_spd_shape": crt["spd_shape"],
                "crt_white_xy": crt["white_xy"],
                "crt_primary_xy": crt["primary_xy"],
                "crt_scene_size": crt["scene_size"],
                "crt_mean_luminance": crt["mean_luminance"],
                "crt_mean_scene_spd_norm": crt["mean_scene_spd_norm"],
                "crt_illuminant_energy_norm": crt["illuminant_energy_norm"],
                "crt_rgb_stats": crt["rgb_stats"],
                "crt_rgb_channel_means": crt["rgb_channel_means"],
            },
            context={},
        )

    if case_name == "scene_surface_models_small":
        def _render_surface_model(n_dims: int | None) -> dict[str, Any]:
            basis = u if n_dims is None else u[:, :n_dims]
            weights = w if n_dims is None else w[:n_dims, :]
            mcc_xyz = xyz_cmfs.T @ (d65_spd.reshape(-1, 1) * basis) @ weights
            max_y = max(float(np.max(mcc_xyz[1, :])), 1e-12)
            mcc_xyz = 100.0 * (mcc_xyz / max_y)
            rendered = xyz_to_srgb(xw_to_rgb_format(mcc_xyz.T, 4, 6))
            rendered = image_flip(image_flip(rendered, "updown"), "leftright")
            center = rendered[rendered.shape[0] // 2, rendered.shape[1] // 2, :]
            return {
                "rgb_stats": _stats_vector(rendered),
                "rgb_channel_means": np.mean(rendered, axis=(0, 1), dtype=float).reshape(-1),
                "center_rgb": np.asarray(center, dtype=float).reshape(-1),
            }

        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        reflectance = macbeth_read_reflectance(wave, asset_store=store)
        u, singular_values, vh = np.linalg.svd(reflectance, full_matrices=False)
        w = np.diag(singular_values) @ vh
        xyz_cmfs = np.asarray(ie_read_spectra("XYZ", wave, asset_store=store), dtype=float)
        d65_spd = np.asarray(ie_read_spectra("D65", wave, asset_store=store), dtype=float).reshape(-1)

        approx_rmse = np.array(
            [
                float(np.sqrt(np.mean(np.square(u[:, :n_dims] @ w[:n_dims, :] - reflectance), dtype=float)))
                for n_dims in range(1, 5)
            ],
            dtype=float,
        )

        render_1 = _render_surface_model(1)
        render_2 = _render_surface_model(2)
        render_3 = _render_surface_model(3)
        render_4 = _render_surface_model(4)
        render_full = _render_surface_model(None)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "reflectance_shape": np.array(reflectance.shape, dtype=int),
                "reflectance_stats": _stats_vector(reflectance),
                "basis_first4": _canonicalize_basis_columns(u[:, :4]),
                "singular_values_first6": singular_values[:6].astype(float),
                "approx_rmse_1to4": approx_rmse,
                "d65_spd_norm": _channel_normalize(d65_spd),
                "render_1_rgb_stats": render_1["rgb_stats"],
                "render_1_channel_means": render_1["rgb_channel_means"],
                "render_1_center_rgb": render_1["center_rgb"],
                "render_2_rgb_stats": render_2["rgb_stats"],
                "render_2_channel_means": render_2["rgb_channel_means"],
                "render_2_center_rgb": render_2["center_rgb"],
                "render_3_rgb_stats": render_3["rgb_stats"],
                "render_3_channel_means": render_3["rgb_channel_means"],
                "render_3_center_rgb": render_3["center_rgb"],
                "render_4_rgb_stats": render_4["rgb_stats"],
                "render_4_channel_means": render_4["rgb_channel_means"],
                "render_4_center_rgb": render_4["center_rgb"],
                "render_full_rgb_stats": render_full["rgb_stats"],
                "render_full_channel_means": render_full["rgb_channel_means"],
                "render_full_center_rgb": render_full["center_rgb"],
            },
            context={},
        )

    if case_name == "color_reflectance_basis_small":
        snapshot_root = store.ensure()
        reflectance_dirs = [
            snapshot_root / "data/surfaces/reflectances",
            snapshot_root / "data/surfaces/charts/esser/reflectance",
        ]
        filenames: list[str] = []
        for directory in reflectance_dirs:
            filenames.extend(sorted(path.name for path in directory.glob("*.mat")))

        selected_indices = np.array([5, 12], dtype=int)
        selected_filenames = [filenames[index - 1] for index in selected_indices]
        wave = np.arange(400.0, 701.0, 5.0, dtype=float)

        reflectances = np.empty((wave.size, 0), dtype=float)
        for filename in selected_filenames:
            current = np.asarray(ie_read_spectra(filename, wave, asset_store=store), dtype=float)
            reflectances = np.concatenate((current, reflectances), axis=1)

        u, singular_values, vh = np.linalg.svd(reflectances, full_matrices=False)
        dim = 8
        basis = u[:, :dim]
        weights = (np.diag(singular_values) @ vh)[:dim, :]
        approx = basis @ weights
        approx_rmse = float(np.sqrt(np.mean(np.square(approx - reflectances), dtype=float)))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "file_count": int(len(filenames)),
                "selected_indices": selected_indices,
                "selected_filenames": np.asarray(selected_filenames, dtype=object),
                "wave": wave,
                "reflectance_shape": np.array(reflectances.shape, dtype=int),
                "reflectance_stats": _stats_vector(reflectances),
                "singular_values_first8": singular_values[:8].astype(float),
                "basis_first4": _canonicalize_basis_columns(u[:, :4]),
                "basis_projector_8": basis @ basis.T,
                "approx_rmse": approx_rmse,
                "approx_stats": _stats_vector(approx),
            },
            context={},
        )

    if case_name == "display_create_lcd_example":
        display = display_create("lcdExample.mat", asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": display.fields["wave"],
                "spd": display.fields["spd"],
                "gamma": display.fields["gamma"],
            },
            context={"display": display},
        )

    if case_name == "oi_psf_default_small":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        oi = oi_compute(oi_create("psf"), scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_psf550_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "psf", None, 550.0)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psfxaxis_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "psfxaxis", None, 550.0, "um")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "wave": float(udata["wave"]),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psfyaxis_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "psfyaxis", None, 550.0, "um")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "wave": float(udata["wave"]),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psf_plot_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["f_number"] = 12.0
        oi.fields["optics"] = optics
        this_wave = 600.0
        units = "um"
        n_samp = 100
        psf_data = dict(oi_get(oi, "psf data", this_wave, units, n_samp))
        xy = np.asarray(psf_data["xy"], dtype=float)
        psf = np.asarray(psf_data["psf"], dtype=float)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": xy[:, :, 0].copy(),
                "y": xy[:, :, 1].copy(),
                "psf": psf,
                "airy_disk_radius_um": float(airy_disk(this_wave, float(oi_get(oi, "fnumber")), "units", units)),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psfxaxis_wvf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        this_wave = 550.0
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        oi_line = dict(oi_get(oi, "optics psf xaxis", this_wave, "um"))
        wvf_line, _ = wvf_plot(wvf, "psfxaxis", "unit", "um", "wave", this_wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": this_wave,
                "oi_samp": np.asarray(oi_line["samp"], dtype=float),
                "oi_data": np.asarray(oi_line["data"], dtype=float),
                "wvf_samp": np.asarray(wvf_line["samp"], dtype=float),
                "wvf_data": np.asarray(wvf_line["data"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_psfyaxis_wvf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        this_wave = 550.0
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        oi_line = dict(oi_get(oi, "optics psf yaxis", this_wave, "um"))
        wvf_line, _ = wvf_plot(wvf, "psfyaxis", "unit", "um", "wave", this_wave)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": this_wave,
                "oi_samp": np.asarray(oi_line["samp"], dtype=float),
                "oi_data": np.asarray(oi_line["data"], dtype=float),
                "wvf_samp": np.asarray(wvf_line["samp"], dtype=float),
                "wvf_data": np.asarray(wvf_line["data"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_psf550_wvf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "psf550")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_lswavelength_wvf_small":
        wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "ls wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "lsWave": np.asarray(udata["lsWave"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_otfwavelength_diffraction_small":
        oi = oi_create("diffraction limited", asset_store=store)
        udata, _ = oi_plot(oi, "otf wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fSupport": np.asarray(udata["fSupport"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "otf": np.asarray(udata["otf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_otfwavelength_wvf_small":
        wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "otf wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fSupport": np.asarray(udata["fSupport"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "otf": np.asarray(udata["otf"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_irradiance_hline_diffraction_lineep_small":
        scene = scene_create("line ep", [128, 128], asset_store=store)
        scene = scene_set(scene, "fov", 0.5)
        oi = oi_create()
        oi = oi_compute(oi, scene)
        roi_locs = np.array([80, 80], dtype=int)
        udata, _ = oi_plot(oi, "irradiance hline", roi_locs)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "roi_locs": roi_locs,
                "pos": np.asarray(udata["pos"], dtype=float),
                "wave": np.asarray(udata["wave"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_wvf_otf_compare_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        wvf_otf = np.asarray(wvf_get(wvf, "otf", 550.0), dtype=complex)
        oi_otf = np.asarray(oi_get(oi, "optics otf"), dtype=complex)
        if oi_otf.ndim == 3:
            oi_otf = oi_otf[:, :, 0]
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "oi_otf_abs": np.abs(oi_otf),
                "wvf_otf_abs_shifted": np.abs(wvf_otf),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_si_lorentzian_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)
        optics = si_synthetic("lorentzian", oi, gamma)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_otfwavelength_si_lorentzian_small":
        oi = oi_create("psf")
        gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)
        optics = si_synthetic("lorentzian", oi, gamma)
        oi = oi_set(oi, "optics", optics)
        udata, _ = oi_plot(oi, "otf wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fSupport": np.asarray(udata["fSupport"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "otf": np.asarray(udata["otf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_psf550_si_lorentzian_small":
        oi = oi_create("psf")
        gamma = np.logspace(0.0, 1.0, np.asarray(oi_get(oi, "wave"), dtype=float).size)
        optics = si_synthetic("lorentzian", oi, gamma)
        oi = oi_set(oi, "optics", optics)
        udata, _ = oi_plot(oi, "psf550")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_si_pillbox_small":
        scene = scene_create("grid lines", [256, 256], 64, "ee", 3, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        patch_size_mm = float(airy_disk(700.0, float(oi_get(oi, "optics fnumber")), "units", "mm"))
        optics = si_synthetic("pillbox", oi, patch_size_mm)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        psf_data = np.asarray(optics["psf_data"]["psf"], dtype=float)
        middle_row = psf_data.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "input_psf_mid_row_550": psf_data[middle_row, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_si_gaussian_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.ones(wave.size, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        psf_data = np.asarray(optics["psf_data"]["psf"], dtype=float)
        middle_row = psf_data.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "input_psf_mid_row_550": psf_data[middle_row, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_si_gaussian_ratio_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.full(wave.size, 2.0, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        psf_data = np.asarray(optics["psf_data"]["psf"], dtype=float)
        middle_row = psf_data.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "input_psf_mid_row_550": psf_data[middle_row, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_gaussian_psf_point_array_small":
        wave = np.arange(450.0, 651.0, 100.0, dtype=float)
        scene = scene_create("point array", 128, 32, asset_store=store)
        scene = scene_interpolate_w(scene, wave, asset_store=store)
        scene = scene_set(scene, "hfov", 1.0)
        scene = scene_set(scene, "name", "psfPointArray")
        oi = oi_create()
        oi = oi_set(oi, "wave", scene_get(scene, "wave"))
        xy_ratio = np.full(wave.size, 3.0, dtype=float)
        wave_spread = wave / float(wave[0])
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene)
        photons = np.asarray(oi.data["photons"], dtype=float)
        photons_normalized = photons / np.maximum(np.max(photons, axis=(0, 1), keepdims=True), 1e-12)
        center_row = photons_normalized.shape[0] // 2
        center_col = photons_normalized.shape[1] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(oi.fields["wave"], dtype=float),
                "scene_wave": np.asarray(scene.fields["wave"], dtype=float),
                "scene_name": scene.name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "center_row_normalized": np.asarray(photons_normalized[center_row, :, :], dtype=float),
                "center_col_normalized": np.asarray(photons_normalized[:, center_col, :], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_psf550_si_gaussian_ratio_small":
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.full(wave.size, 2.0, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        udata, _ = oi_plot(oi, "psf550")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "psf": np.asarray(udata["psf"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_illuminance_lines_si_gaussian_ratio_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("psf")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float)
        wave_spread = 0.5 * (wave / float(wave[0])) ** 3
        xy_ratio = np.full(wave.size, 2.0, dtype=float)
        optics = si_synthetic("gaussian", oi, wave_spread, xy_ratio)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
        xy_middle = np.array([int(np.ceil(size[1] / 2.0)), int(np.ceil(size[0] / 2.0))], dtype=int)
        v_data, _ = oi_plot(oi, "illuminance vline", xy_middle)
        h_data, _ = oi_plot(oi, "illuminance hline", xy_middle)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "xy_middle": xy_middle,
                "v_pos": np.asarray(v_data["pos"], dtype=float),
                "v_data": np.asarray(v_data["data"], dtype=float),
                "h_pos": np.asarray(h_data["pos"], dtype=float),
                "h_data": np.asarray(h_data["data"], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_si_custom_file_small":
        scene = scene_create("grid lines", [64, 64], 16, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)
        oi = oi_create("shift invariant")
        wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        samples = np.arange(129, dtype=float) - 64.0
        xx, yy = np.meshgrid(samples, samples, indexing="xy")
        psf = np.zeros((129, 129, wave.size), dtype=float)
        for idx, wavelength in enumerate(wave):
            sigma = 1.2 + 0.01 * ((float(wavelength) - float(wave[0])) / 10.0)
            plane = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
            psf[:, :, idx] = plane / np.sum(plane)
        with tempfile.TemporaryDirectory(prefix="pyisetcam-si-custom-") as tmpdir:
            path = ie_save_si_data_file(psf, wave, np.array([0.25, 0.25], dtype=float), f"{tmpdir}/custom_si_psf.mat")
            optics = si_synthetic("custom", oi, path)
        oi = oi_set(oi, "optics", optics)
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "input_psf_mid_row_550": psf[64, :, 15],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_custom_otf_flare_small":
        scene = scene_create("point array", 64, 16, asset_store=store)
        scene = scene_set(scene, "hfov", 40.0)
        otf_struct = optics_psf_to_otf(
            store.resolve("data/optics/flare/flare1.png"),
            1.2e-6,
            np.arange(400.0, 701.0, 10.0, dtype=float),
        )
        oi = oi_create("shift invariant")
        oi = oi_set(oi, "optics otfstruct", otf_struct)
        optics = dict(oi.fields["optics"])
        _, width_m, _ = _oi_geometry(optics, scene)
        sample_spacing_m = float(width_m) / max(int(scene.data["photons"].shape[1]), 1)
        pad_rows = int(np.round(scene.data["photons"].shape[0] / 8.0))
        pad_cols = int(np.round(scene.data["photons"].shape[1] / 8.0))
        custom_otf = _shift_invariant_custom_otf(
            (
                int(scene.data["photons"].shape[0] + (2 * pad_rows)),
                int(scene.data["photons"].shape[1] + (2 * pad_cols)),
            ),
            sample_spacing_m,
            np.asarray(scene.fields["wave"], dtype=float),
            optics,
        )
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "fx": np.asarray(otf_struct["fx"], dtype=float),
                "fy": np.asarray(otf_struct["fy"], dtype=float),
                "otf_abs550": np.abs(np.asarray(otf_struct["OTF"], dtype=complex)[:, :, 15]),
                "interp_otf_abs550": np.abs(np.asarray(custom_otf, dtype=complex)[:, :, 15]),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_psf_to_otf_flare_small":
        otf_struct = optics_psf_to_otf(
            store.resolve("data/optics/flare/flare1.png"),
            1.2e-6,
            np.arange(400.0, 701.0, 10.0, dtype=float),
        )
        otf_plane = np.abs(np.fft.fftshift(np.asarray(otf_struct["OTF"], dtype=complex)[:, :, 15]))
        row_index = (otf_plane.shape[0] // 2)
        center_row = int(np.rint(otf_plane.shape[0] / 2.0)) - 1
        center_col = int(np.rint(otf_plane.shape[1] / 2.0)) - 1
        row = otf_plane[row_index, :]
        center = otf_plane[max(0, center_row - 16) : center_row + 17, max(0, center_col - 16) : center_col + 17]
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(otf_struct["fx"], dtype=float),
                "fy": np.asarray(otf_struct["fy"], dtype=float),
                "otf_abs550_row": np.asarray(row, dtype=float),
                "otf_abs550_center": np.asarray(center, dtype=float),
            },
            context={"otf_struct": otf_struct},
        )

    if case_name == "oi_ideal_otf_small":
        params = {
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
            "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
            "blockSize": 16,
            "contrast": 1.0,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 3.0)
        oi = oi_compute(oi_create("shift invariant"), scene, crop=True)
        raw_otf = oi_get(oi, "optics OTF")
        oi = oi_set(oi, "optics OTF", np.ones_like(np.asarray(raw_otf), dtype=complex))
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_wvf_defocus_small":
        params = {
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
            "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
            "blockSize": 16,
            "contrast": 1.0,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        wvf = wvf_create(wave=scene_get(scene, "wave"))
        wvf = wvf_set(wvf, "zcoeffs", np.array([2.0, 0.5], dtype=float), ["defocus", "vertical_astigmatism"])
        oi = oi_compute(wvf, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "defocus": wvf_get(oi_get(oi, "wvf"), "zcoeffs", "defocus"),
                "vertical_astigmatism": wvf_get(oi_get(oi, "wvf"), "zcoeffs", "vertical_astigmatism"),
            },
            context={"scene": scene, "wvf": wvf, "oi": oi},
        )

    if case_name == "oi_wvf_script_defocus_small":
        params = {
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
            "freqs": np.array([1.0, 2.0, 4.0], dtype=float),
            "blockSize": 16,
            "contrast": 1.0,
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        wvf = wvf_create(wave=scene_get(scene, "wave"))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_set(wvf, "zcoeffs", 1.5, "defocus")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        oi = oi_compute(oi, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi_get(oi, "wave"),
                "photons": oi_get(oi, "photons"),
                "defocus_zcoeff": wvf_get(wvf, "zcoeffs", "defocus"),
                "pupil_diameter_mm": wvf_get(wvf, "pupil diameter", "mm"),
                "f_number": oi_get(oi, "fnumber"),
            },
            context={"scene": scene, "wvf": wvf, "oi": oi},
        )

    if case_name == "optics_defocus_wvf_small":
        scene = scene_create("point array", np.array([512, 512], dtype=int), 128, asset_store=store)
        scene = scene_set(scene, "fov", 1.5)

        def _psf_center_row_norm(current_wvf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            psf = np.asarray(wvf_get(current_wvf, "psf", 550.0), dtype=float)
            x_axis = np.asarray(wvf_get(current_wvf, "psf spatial samples", "um", 550.0), dtype=float).reshape(-1)
            center_row = psf[psf.shape[0] // 2, :]
            return x_axis, _channel_normalize(center_row)

        def _oi_center_row_norm(current_oi: OpticalImage) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            return _channel_normalize(photons[photons.shape[0] // 2, :, wave_index])

        def _normalized_mae(reference: np.ndarray, current: np.ndarray) -> float:
            ref = np.asarray(reference, dtype=float)
            cur = np.asarray(current, dtype=float)
            return float(np.mean(np.abs(cur - ref)) / max(float(np.mean(np.abs(ref))), 1e-12))

        wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)

        wvf0 = wvf_create(wave=wave)
        wvf0 = wvf_set(wvf0, "focal length", 8.0, "mm")
        wvf0 = wvf_set(wvf0, "pupil diameter", 3.0, "mm")
        wvf0 = wvf_compute(wvf0)
        oi0 = wvf_to_oi(wvf0)
        dl_psf_x_um, dl_psf_center_row = _psf_center_row_norm(wvf0)
        oi0 = oi_compute(oi0, scene, crop=True)
        dl_oi_center_row = _oi_center_row_norm(oi0)

        diopters = 1.5
        wvf1 = wvf_create(wave=wave)
        wvf1 = wvf_set(wvf1, "zcoeffs", diopters, "defocus")
        wvf1 = wvf_compute(wvf1)
        oi1 = wvf_to_oi(wvf1)
        explicit_psf_x_um, explicit_defocus_psf_center_row = _psf_center_row_norm(wvf1)
        oi1 = oi_compute(oi1, scene, crop=True)
        explicit_defocus_oi_center_row = _oi_center_row_norm(oi1)

        wvf = wvf_create(wave=wave)
        oi = oi_create("wvf", wvf)
        oi = oi_compute(oi, scene, crop=True)
        oi_method_base = oi
        oi_method_base_wvf = wvf_compute(oi_get(oi, "optics wvf"))
        oi_method_base_psf_x_um, oi_method_base_psf_center_row = _psf_center_row_norm(oi_method_base_wvf)
        oi_method_base_oi_center_row = _oi_center_row_norm(oi)

        updated_wvf = oi_method_base_wvf
        updated_wvf = wvf_set(updated_wvf, "zcoeffs", diopters, "defocus")
        updated_wvf = wvf_compute(updated_wvf)
        oi = oi_set(oi, "optics wvf", updated_wvf)
        oi_method_defocus_psf_x_um, oi_method_defocus_psf_center_row = _psf_center_row_norm(updated_wvf)
        oi = oi_compute(oi, scene, crop=True)
        oi_method_defocus_oi_center_row = _oi_center_row_norm(oi)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "diffraction_limited_focal_length_mm": float(wvf_get(wvf0, "focal length", "mm")),
                "diffraction_limited_pupil_diameter_mm": float(wvf_get(wvf0, "pupil diameter", "mm")),
                "diffraction_limited_f_number": float(oi_get(oi0, "fnumber")),
                "diffraction_limited_psf_x_um": dl_psf_x_um,
                "diffraction_limited_psf_center_row_550_norm": dl_psf_center_row,
                "diffraction_limited_oi_center_row_550_norm": dl_oi_center_row,
                "defocus_diopters": float(diopters),
                "explicit_defocus_f_number": float(oi_get(oi1, "fnumber")),
                "explicit_defocus_zcoeff": float(oi_get(oi1, "wvf", "zcoeffs", "defocus")),
                "explicit_defocus_psf_x_um": explicit_psf_x_um,
                "explicit_defocus_psf_center_row_550_norm": explicit_defocus_psf_center_row,
                "explicit_defocus_oi_center_row_550_norm": explicit_defocus_oi_center_row,
                "oi_method_base_f_number": float(oi_get(oi_method_base, "fnumber")),
                "oi_method_base_psf_x_um": oi_method_base_psf_x_um,
                "oi_method_base_psf_center_row_550_norm": oi_method_base_psf_center_row,
                "oi_method_base_oi_center_row_550_norm": oi_method_base_oi_center_row,
                "oi_method_defocus_f_number": float(oi_get(oi, "fnumber")),
                "oi_method_defocus_zcoeff": float(oi_get(oi, "wvf", "zcoeffs", "defocus")),
                "oi_method_defocus_psf_x_um": oi_method_defocus_psf_x_um,
                "oi_method_defocus_psf_center_row_550_norm": oi_method_defocus_psf_center_row,
                "oi_method_defocus_oi_center_row_550_norm": oi_method_defocus_oi_center_row,
                "explicit_vs_oi_method_psf_center_row_550_normalized_mae": _normalized_mae(
                    explicit_defocus_psf_center_row,
                    oi_method_defocus_psf_center_row,
                ),
                "explicit_vs_oi_method_oi_center_row_550_normalized_mae": _normalized_mae(
                    explicit_defocus_oi_center_row,
                    oi_method_defocus_oi_center_row,
                ),
            },
            context={"scene": scene, "wvf0": wvf0, "wvf1": wvf1, "oi0": oi0, "oi1": oi1, "oi_method_base": oi_method_base, "oi": oi},
        )

    if case_name == "optics_rt_synthetic_small":
        scene = scene_create("point array", 256, asset_store=store)
        scene = scene_set(scene, "h fov", 3.0)
        scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))

        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        oi = oi_create("ray trace", asset_store=store)
        spread_limits = np.array([1.0, 3.0], dtype=float)
        xy_ratio = 1.6
        optics = rt_synthetic(oi, spread_limits=(float(spread_limits[0]), float(spread_limits[1])), xy_ratio=xy_ratio)
        oi = oi_set(oi, "optics", optics)
        scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
        oi = oi_compute(oi, scene)

        raytrace = oi.fields["optics"]["raytrace"]
        psf = np.asarray(raytrace["psf"]["function"], dtype=float)
        field_height_mm = np.asarray(raytrace["psf"]["field_height_mm"], dtype=float).reshape(-1)
        raytrace_wave = np.asarray(raytrace["psf"]["wavelength_nm"], dtype=float).reshape(-1)
        wave_index_550 = int(np.argmin(np.abs(raytrace_wave - 550.0)))
        center_field_index = 0
        edge_field_index = psf.shape[2] - 1
        center_psf = psf[:, :, center_field_index, wave_index_550]
        edge_psf = psf[:, :, edge_field_index, wave_index_550]
        center_psf_row = _channel_normalize(center_psf[center_psf.shape[0] // 2, :])
        edge_psf_row = _channel_normalize(edge_psf[edge_psf.shape[0] // 2, :])

        photons = np.asarray(oi_get(oi, "photons"), dtype=float)
        oi_wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        oi_wave_index_550 = int(np.argmin(np.abs(oi_wave - 550.0)))
        oi_center_row = _channel_normalize(photons[photons.shape[0] // 2, :, oi_wave_index_550])

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "spread_limits": spread_limits,
                "xy_ratio": float(xy_ratio),
                "raytrace_field_height_mm": field_height_mm,
                "raytrace_wave": raytrace_wave,
                "geometry_550": np.asarray(raytrace["geometry"]["function"], dtype=float)[:, wave_index_550],
                "relative_illumination_550": np.asarray(raytrace["relative_illumination"]["function"], dtype=float)[:, wave_index_550],
                "center_psf_sum_550": float(np.sum(center_psf)),
                "edge_psf_sum_550": float(np.sum(edge_psf)),
                "center_psf_mid_row_550_norm": center_psf_row,
                "edge_psf_mid_row_550_norm": edge_psf_row,
                "oi_wave": oi_wave,
                "oi_photons_shape": np.asarray(photons.shape, dtype=float),
                "oi_mean_photons_by_wave": np.mean(photons, axis=(0, 1)),
                "oi_p95_photons_by_wave": np.percentile(photons.reshape(-1, photons.shape[2]), 95, axis=0),
                "oi_max_photons_by_wave": np.max(photons, axis=(0, 1)),
                "oi_center_row_550_norm": _canonical_profile(oi_center_row),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_rt_gridlines_small":
        scene = scene_create("grid lines", [384, 384], 48, asset_store=store)
        scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))
        scene = scene_set(scene, "hfov", 45.0)
        scene = scene_set(scene, "name", "rtDemo-Large-grid")

        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        def _center_row_550_norm(current_oi: OpticalImage) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            center_row = photons[photons.shape[0] // 2, :, wave_index]
            return _canonical_profile(_channel_normalize(center_row))

        def _profile_widths(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            peak = max(float(np.max(profile)), 1e-12)
            widths = []
            for threshold in thresholds:
                active = np.flatnonzero(profile >= (threshold * peak))
                widths.append(int(active[-1] - active[0] + 1) if active.size else 0)
            return np.asarray(widths, dtype=int)

        oi = oi_create("ray trace", store.resolve("data/optics/zmWideAngle.mat"), asset_store=store)
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

        automated_rt = oi_set(oi.clone(), "optics model", "ray trace")
        automated_rt = oi_compute(automated_rt, scene)

        diffraction_oi = oi_set(automated_rt.clone(), "optics model", "diffraction limited")
        diffraction_oi = oi_set(diffraction_oi, "optics fnumber", oi_get(automated_rt, "rtfnumber"))
        diffraction_oi = oi_compute(diffraction_oi, scene)

        scene_small = scene_set(scene.clone(), "name", "rt-Small-Grid")
        scene_small = scene_set(scene_small, "fov", 20.0)
        rt_small = oi_compute(automated_rt.clone(), scene_small)
        dl_small = oi_compute(diffraction_oi.clone(), scene_small)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "requested_scene_hfov_deg": 45.0,
                "adjusted_scene_hfov_deg": float(scene_get(scene, "fov")),
                "raytrace_fov_deg": raytrace_fov,
                "raytrace_f_number": float(oi_get(oi, "rtfnumber")),
                "raytrace_effective_focal_length_mm": float(oi_get(oi, "rtefl", "mm")),
                "geometry_only_size": np.asarray(oi_get(geometry_oi, "size"), dtype=int),
                "geometry_center_row_550_norm": _center_row_550_norm(geometry_oi),
                "psf_struct_sample_angles": np.asarray(psf_struct["sampAngles"], dtype=float).reshape(-1),
                "psf_struct_img_height_mm": np.asarray(psf_struct["imgHeight"], dtype=float).reshape(-1),
                "psf_struct_wavelength": np.asarray(psf_struct["wavelength"], dtype=float).reshape(-1),
                "stepwise_rt_size": np.asarray(oi_get(stepwise_oi, "size"), dtype=int),
                "stepwise_rt_center_row_550_widths": _profile_widths(_center_row_550_norm(stepwise_oi), (0.25, 0.10, 0.01)),
                "automated_rt_size": np.asarray(oi_get(automated_rt, "size"), dtype=int),
                "automated_rt_center_row_550_widths": _profile_widths(_center_row_550_norm(automated_rt), (0.05, 0.01)),
                "diffraction_large_size": np.asarray(oi_get(diffraction_oi, "size"), dtype=int),
                "diffraction_large_center_row_550_widths": _profile_widths(
                    _center_row_550_norm(diffraction_oi),
                    (0.50, 0.10, 0.01),
                ),
                "small_scene_fov_deg": float(scene_get(scene_small, "fov")),
                "rt_small_size": np.asarray(oi_get(rt_small, "size"), dtype=int),
                "rt_small_center_row_550_norm": _center_row_550_norm(rt_small),
                "rt_small_center_row_550_widths": _profile_widths(_center_row_550_norm(rt_small), (0.50, 0.10, 0.01)),
                "dl_small_size": np.asarray(oi_get(dl_small, "size"), dtype=int),
                "dl_small_center_row_550_widths": _profile_widths(_center_row_550_norm(dl_small), (0.50, 0.10, 0.01)),
            },
            context={
                "scene": scene,
                "geometry_oi": geometry_oi,
                "stepwise_oi": stepwise_oi,
                "automated_rt": automated_rt,
                "diffraction_oi": diffraction_oi,
                "scene_small": scene_small,
                "rt_small": rt_small,
                "dl_small": dl_small,
            },
        )

    if case_name == "optics_rt_center_edge_psf_small":
        reference_wavelength_nm = 550.0
        oi = oi_create("ray trace", asset_store=store)
        psf_payload = _raytrace_psf_payload(oi, reference_wavelength_nm=reference_wavelength_nm)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "reference_wavelength_nm": reference_wavelength_nm,
                "field_heights_mm": np.asarray([0.0, psf_payload["edge_field_height_mm"]], dtype=float),
                "support_x_um": psf_payload["support_x_um"],
                "support_y_um": psf_payload["support_y_um"],
                "center_psf_norm": psf_payload["center_psf_norm"],
                "edge_psf_norm": psf_payload["edge_psf_norm"],
                "center_psf_center_row_norm": psf_payload["center_psf_center_row_norm"],
                "edge_psf_center_row_norm": psf_payload["edge_psf_center_row_norm"],
                "center_psf_metrics": psf_payload["center_psf_metrics"],
                "edge_psf_metrics": psf_payload["edge_psf_metrics"],
            },
            context={"oi": oi},
        )

    if case_name == "optics_rt_point_array_field_small":
        point_spacing_px = 64
        crop_radius = 24
        scene = scene_create("point array", 320, point_spacing_px, asset_store=store)
        scene = scene_set(scene, "fov", 12.0)
        oi = oi_create("ray trace", asset_store=store)
        scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
        geometry_oi = rt_geometry(oi, scene)
        psf_struct = rt_precompute_psf(geometry_oi, angle_step_deg=20.0)
        stepwise_oi = oi_set(geometry_oi, "psf struct", psf_struct)
        stepwise_oi = rt_precompute_psf_apply(stepwise_oi, angle_step_deg=20.0)
        render_rgb = np.asarray(oi_show_image(stepwise_oi, -1, 1.0, asset_store=store), dtype=float)
        peaks = _find_point_peaks(render_rgb)
        center_peak_zero_based, edge_peak_zero_based = _select_center_and_edge_peaks(peaks, render_rgb.shape[:2])
        center_crop_rgb = _crop_square(render_rgb, center_peak_zero_based, crop_radius)
        edge_crop_rgb = _crop_square(render_rgb, edge_peak_zero_based, crop_radius)
        center_luma = _luma(center_crop_rgb)
        edge_luma = _luma(edge_crop_rgb)
        center_peak_rc = center_peak_zero_based + 1
        edge_peak_rc = edge_peak_zero_based + 1
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "point_spacing_px": float(point_spacing_px),
                "render_rgb": render_rgb,
                "center_peak_rc": center_peak_rc.astype(int),
                "edge_peak_rc": edge_peak_rc.astype(int),
                "center_crop_rgb": center_crop_rgb,
                "edge_crop_rgb": edge_crop_rgb,
                "center_crop_luma_norm": center_luma / max(float(np.max(center_luma)), 1.0e-12),
                "edge_crop_luma_norm": edge_luma / max(float(np.max(edge_luma)), 1.0e-12),
                "center_crop_metrics": _spot_metrics(center_luma, spacing_um=1.0),
                "edge_crop_metrics": _spot_metrics(edge_luma, spacing_um=1.0),
            },
            context={"scene": scene, "oi": stepwise_oi},
        )

    if case_name == "optics_rt_distortion_field_small":
        reference_wavelength_nm = 550.0
        scene = scene_create("distortion grid", [384, 384], 48, "ee", 2, asset_store=store)
        scene = scene_set(scene, "fov", 20.0)
        ideal_grid_rgb = np.asarray(scene_show_image(scene, -1, 1.0, asset_store=store), dtype=float)
        oi = oi_create("ray trace", asset_store=store)
        scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
        geometry_oi = rt_geometry(oi, scene)
        distorted_grid_rgb = np.asarray(oi_show_image(geometry_oi, -1, 1.0, asset_store=store), dtype=float)
        field_height_mm = np.asarray(oi_get(geometry_oi, "rtgeomfieldheight", "mm"), dtype=float).reshape(-1)
        distorted_height_mm = np.asarray(
            oi_get(geometry_oi, "rtgeomfunction", reference_wavelength_nm, "mm"),
            dtype=float,
        ).reshape(-1)
        distortion_percent = np.zeros_like(field_height_mm)
        nonzero = field_height_mm > 1.0e-12
        distortion_percent[nonzero] = (
            (distorted_height_mm[nonzero] - field_height_mm[nonzero]) / field_height_mm[nonzero]
        ) * 100.0
        max_distortion_index = int(np.argmax(np.abs(distortion_percent)))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "reference_wavelength_nm": reference_wavelength_nm,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "ideal_grid_rgb": ideal_grid_rgb,
                "distorted_grid_rgb": distorted_grid_rgb,
                "field_height_mm": field_height_mm,
                "distorted_height_mm": distorted_height_mm,
                "distortion_percent": distortion_percent,
                "max_distortion_percent": float(np.abs(distortion_percent[max_distortion_index])),
                "max_distortion_field_height_mm": float(field_height_mm[max_distortion_index]),
            },
            context={"scene": scene, "oi": geometry_oi},
        )

    if case_name == "ip_rt_point_array_pipeline_small":
        point_spacing_px = 64
        crop_radius = 24
        scene = scene_create("point array", 320, point_spacing_px, asset_store=store)
        scene = scene_set(scene, "fov", 12.0)

        oi = oi_create("ray trace", asset_store=store)
        scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
        oi = oi_compute(oi, scene)

        sensor = sensor_set(sensor_create(asset_store=store), "noise flag", 0)
        sensor = sensor_compute(sensor, oi, seed=0)
        ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)

        sensor_volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        result = np.asarray(ip_get(ip, "result"), dtype=float)

        sensor_peaks = _find_point_peaks(sensor_volts)
        sensor_center_peak_zero_based, sensor_edge_peak_zero_based = _select_center_and_edge_peaks(
            sensor_peaks, sensor_volts.shape[:2]
        )
        sensor_center_crop = _crop_square_padded(sensor_volts, sensor_center_peak_zero_based, crop_radius)
        sensor_edge_crop = _crop_square_padded(sensor_volts, sensor_edge_peak_zero_based, crop_radius)

        result_center_peak_zero_based = sensor_center_peak_zero_based
        result_edge_peak_zero_based = sensor_edge_peak_zero_based
        result_center_crop = _crop_square_padded(result, result_center_peak_zero_based, crop_radius)
        result_edge_crop = _crop_square_padded(result, result_edge_peak_zero_based, crop_radius)
        result_center_luma = _luma(result_center_crop)
        result_edge_luma = _luma(result_edge_crop)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "point_spacing_px": float(point_spacing_px),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int).reshape(-1),
                "sensor_volts": sensor_volts,
                "result": result,
                "sensor_center_peak_rc": (sensor_center_peak_zero_based + 1).astype(int),
                "sensor_edge_peak_rc": (sensor_edge_peak_zero_based + 1).astype(int),
                "result_center_peak_rc": (result_center_peak_zero_based + 1).astype(int),
                "result_edge_peak_rc": (result_edge_peak_zero_based + 1).astype(int),
                "sensor_center_crop_norm": sensor_center_crop / max(float(np.max(sensor_center_crop)), 1.0e-12),
                "sensor_edge_crop_norm": sensor_edge_crop / max(float(np.max(sensor_edge_crop)), 1.0e-12),
                "sensor_center_crop_metrics": _spot_metrics(sensor_center_crop, spacing_um=1.0),
                "sensor_edge_crop_metrics": _spot_metrics(sensor_edge_crop, spacing_um=1.0),
                "result_center_crop_rgb": result_center_crop,
                "result_edge_crop_rgb": result_edge_crop,
                "result_center_crop_luma_norm": result_center_luma / max(float(np.max(result_center_luma)), 1.0e-12),
                "result_edge_crop_luma_norm": result_edge_luma / max(float(np.max(result_edge_luma)), 1.0e-12),
                "result_center_crop_metrics": _spot_metrics(result_center_luma, spacing_um=1.0),
                "result_edge_crop_metrics": _spot_metrics(result_edge_luma, spacing_um=1.0),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "pipeline_rt_bar_small":
        scene = scene_create("bar", 128, 3, asset_store=store)
        scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))
        scene = scene_set(scene, "hfov", 12.0)
        scene = scene_set(scene, "name", "rtDemo-Bar")

        oi = oi_create("ray trace", store.resolve("data/optics/zmWideAngle.mat"), asset_store=store)
        oi = oi_set(oi, "wangular", scene_get(scene, "wangular"))
        oi = oi_set(oi, "wave", scene_get(scene, "wave"))
        scene = scene_set(scene, "distance", 2.0)
        oi = oi_set(oi, "optics rtObjectDistance", scene_get(scene, "distance", "mm"))

        geometry_oi = rt_geometry(oi, scene)
        psf_struct = rt_precompute_psf(geometry_oi, angle_step_deg=20.0)
        oi = oi_set(geometry_oi, "psf struct", psf_struct)
        oi = rt_precompute_psf_apply(oi, angle_step_deg=20.0)

        sensor = sensor_set(sensor_create(asset_store=store), "noise flag", 0)
        sensor = sensor_compute(sensor, oi, seed=0)
        ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)

        oi_rgb = np.asarray(oi_show_image(oi, -1, 1.0, asset_store=store), dtype=float)
        sensor_volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        oi_edge = _central_vertical_edge_payload(oi_rgb, crop_radius=32)
        sensor_edge = _central_vertical_edge_payload(sensor_volts, crop_radius=16)
        result_edge = _central_vertical_edge_payload(result, crop_radius=16)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int).reshape(-1),
                "ip_size": np.asarray(ip_get(ip, "size"), dtype=int).reshape(-1),
                "oi_edge_rc": oi_edge["edge_rc"].astype(int),
                "sensor_edge_rc": sensor_edge["edge_rc"].astype(int),
                "result_edge_rc": result_edge["edge_rc"].astype(int),
                "oi_edge_crop_norm": oi_edge["crop_norm"],
                "oi_edge_profile_norm": oi_edge["profile_norm"],
                "sensor_edge_crop_norm": sensor_edge["crop_norm"],
                "sensor_edge_profile_norm": sensor_edge["profile_norm"],
                "result_edge_crop_norm": result_edge["crop_norm"],
                "result_edge_profile_norm": result_edge["profile_norm"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "optics_rt_psf_small":
        scene = scene_create("point array", 512, 32, asset_store=store)
        scene = scene_interpolate_w(scene, np.arange(450.0, 651.0, 100.0, dtype=float))
        scene = scene_set(scene, "hfov", 10.0)
        scene = scene_set(scene, "name", "psf Point Array")

        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        def _center_row_550_widths(current_oi: OpticalImage, thresholds: tuple[float, ...]) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            center_row = _canonical_profile(_channel_normalize(photons[photons.shape[0] // 2, :, wave_index]))
            peak = max(float(np.max(center_row)), 1e-12)
            widths = []
            for threshold in thresholds:
                active = np.flatnonzero(center_row >= (threshold * peak))
                widths.append(int(active[-1] - active[0] + 1) if active.size else 0)
            return np.asarray(widths, dtype=int)

        oi = oi_create("ray trace", store.resolve("data/optics/rtZemaxExample.mat"), asset_store=store)
        scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
        oi = oi_set(oi, "name", "ray trace case")
        oi = oi_set(oi, "optics model", "ray trace")
        oi = oi_compute(oi, scene)

        sampled_rt_psf = np.asarray(oi_get(oi, "sampledRTpsf"), dtype=object)
        psf_wave = np.asarray(oi_get(oi, "psf wavelength"), dtype=float).reshape(-1)
        wave_index_550 = int(np.argmin(np.abs(psf_wave - 550.0)))
        center_psf = np.asarray(sampled_rt_psf[0, 0, wave_index_550], dtype=float)
        edge_psf = np.asarray(sampled_rt_psf[0, -1, wave_index_550], dtype=float)

        oi_dl = oi_set(oi.clone(), "name", "diffraction case")
        optics = oi_get(oi_dl, "optics")
        f_number = float(optics["rayTrace"]["fNumber"])
        oi_dl = oi_set(oi_dl, "optics fnumber", f_number * 0.8)
        oi_dl = oi_set(oi_dl, "optics model", "diffraction limited")
        oi_dl = oi_compute(oi_dl, scene)

        rt_photons = np.asarray(oi_get(oi, "photons"), dtype=float)
        dl_photons = np.asarray(oi_get(oi_dl, "photons"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "rt_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "rt_f_number": float(oi_get(oi, "rtfnumber")),
                "rt_optics_name": str(oi_get(oi, "rtname")),
                "rt_psf_sample_angles_deg": np.asarray(oi_get(oi, "psf sample angles"), dtype=float).reshape(-1),
                "rt_psf_image_heights_mm": np.asarray(oi_get(oi, "psf image heights", "mm"), dtype=float).reshape(-1),
                "rt_psf_wavelength": psf_wave,
                "rt_sampled_psf_shape": np.asarray(sampled_rt_psf.shape, dtype=int),
                "rt_center_psf_mid_row_550_norm": _canonical_profile(
                    _channel_normalize(center_psf[center_psf.shape[0] // 2, :])
                ),
                "rt_edge_psf_mid_row_550_norm": _canonical_profile(
                    _channel_normalize(edge_psf[edge_psf.shape[0] // 2, :])
                ),
                "rt_mean_photons_by_wave": np.mean(rt_photons, axis=(0, 1)),
                "rt_max_photons_by_wave": np.max(rt_photons, axis=(0, 1)),
                "rt_center_row_550_widths": _center_row_550_widths(oi, (0.50, 0.10, 0.01)),
                "dl_size": np.asarray(oi_get(oi_dl, "size"), dtype=int),
                "dl_f_number": float(oi_get(oi_dl, "fnumber")),
                "dl_mean_photons_by_wave": np.mean(dl_photons, axis=(0, 1)),
                "dl_max_photons_by_wave": np.max(dl_photons, axis=(0, 1)),
                "dl_center_row_550_widths": _center_row_550_widths(oi_dl, (0.50, 0.10, 0.01)),
            },
            context={"scene": scene, "oi": oi, "oi_dl": oi_dl},
        )

    if case_name == "optics_rt_psf_view_small":
        scene = scene_create("point array", 384, asset_store=store)
        scene = scene_set(scene, "h fov", 4.0)
        scene = scene_interpolate_w(scene, np.arange(550.0, 651.0, 100.0, dtype=float))

        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        oi = oi_create(asset_store=store)
        rt_optics = rt_synthetic(oi, spread_limits=(1.0, 5.0), xy_ratio=1.6)
        oi = oi_set(oi, "optics", rt_optics)
        scene = scene_set(scene, "distance", oi_get(oi, "optics rtObjectDistance", "m"))
        oi = oi_compute(oi, scene)

        sampled_rt_psf = np.asarray(oi_get(oi, "sampledRTpsf"), dtype=object)
        field_height_rows = []
        field_height_widths = []
        for height_index in range(sampled_rt_psf.shape[1]):
            psf = np.asarray(sampled_rt_psf[0, height_index, 0], dtype=float)
            row = _channel_normalize(psf[psf.shape[0] // 2, :])
            canonical = _canonical_profile(row)
            field_height_rows.append(canonical)
            field_height_widths.append(int(np.count_nonzero(canonical >= (0.10 * max(float(np.max(canonical)), 1e-12)))))

        angle_rows = []
        angle_widths = []
        for angle_index in range(sampled_rt_psf.shape[0]):
            psf = np.asarray(sampled_rt_psf[angle_index, -1, 0], dtype=float)
            row = _channel_normalize(psf[psf.shape[0] // 2, :])
            canonical = _canonical_profile(row)
            angle_rows.append(canonical)
            angle_widths.append(int(np.count_nonzero(canonical >= (0.10 * max(float(np.max(canonical)), 1e-12)))))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "psf_sample_angles_deg": np.asarray(oi_get(oi, "psf sample angles"), dtype=float).reshape(-1),
                "psf_image_heights_mm": np.asarray(oi_get(oi, "psf image heights", "mm"), dtype=float).reshape(-1),
                "psf_wavelength": np.asarray(oi_get(oi, "psf wavelength"), dtype=float).reshape(-1),
                "sampled_rt_psf_shape": np.asarray(sampled_rt_psf.shape, dtype=int),
                "field_height_psf_mid_rows_550_norm": np.asarray(field_height_rows, dtype=float),
                "field_height_psf_widths_10pct": np.asarray(field_height_widths, dtype=int),
                "angle_sweep_edge_psf_mid_rows_550_norm": np.asarray(angle_rows, dtype=float),
                "angle_sweep_edge_psf_widths_10pct": np.asarray(angle_widths, dtype=int),
                "center_rtplot_psf_mid_row_550_norm": np.asarray(field_height_rows[0], dtype=float),
                "edge_rtplot_psf_mid_row_550_norm": np.asarray(field_height_rows[-1], dtype=float),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_defocus_small":
        scene = scene_create("disk array", 256, 32, np.array([2, 2], dtype=int), asset_store=store)
        scene = scene_set(scene, "fov", 0.5)

        wvf = wvf_create(wave=scene_get(scene, "wave"))
        oi = oi_create("wvf", wvf)
        oi = oi_compute(oi, scene)

        wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        wave_index_550 = int(np.argmin(np.abs(wave - 550.0)))

        def _mean_photons(current_oi: Any) -> float:
            return float(np.mean(np.asarray(oi_get(current_oi, "photons"), dtype=float)))

        def _center_row_norm_550(current_oi: Any) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            center_row = photons[photons.shape[0] // 2, :, wave_index_550]
            return _channel_normalize(center_row)

        base_mean = _mean_photons(oi)
        base_center_row_550 = _center_row_norm_550(oi)

        oi = oi_set(oi, "wvf zcoeffs", 2.5, "defocus")
        oi = oi_compute(oi, scene)
        defocus_center_row_550 = _center_row_norm_550(oi)
        defocus_coeff = float(oi_get(oi, "wvf", "zcoeffs", "defocus"))

        oi = oi_set(oi, "wvf zcoeffs", 1.0, "vertical_astigmatism")
        oi = oi_compute(oi, scene)
        astig_center_row_550 = _center_row_norm_550(oi)
        astig_coeff = float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism"))

        oi = oi_set(oi, "wvf zcoeffs", 0.0, "vertical_astigmatism")
        oi = oi_compute(oi, scene)

        oi = oi_set(oi, "wvf zcoeffs", 0.0, "defocus")
        oi = oi_compute(oi, scene)
        reset_mean = _mean_photons(oi)
        reset_center_row_550 = _center_row_norm_550(oi)

        current_wvf = oi_get(oi, "wvf")
        pupil_diameter_mm = float(wvf_get(current_wvf, "calc pupil diameter", "mm"))
        updated_wvf = wvf_set(current_wvf, "calc pupil diameter", 2.0 * pupil_diameter_mm, "mm")
        oi = oi_set(oi, "optics wvf", updated_wvf)
        oi = oi_compute(oi, scene)
        large_pupil_center_row_550 = _center_row_norm_550(oi)

        restored_wvf = wvf_set(oi_get(oi, "wvf"), "calc pupil diameter", pupil_diameter_mm, "mm")
        oi = oi_set(oi, "optics wvf", restored_wvf)
        oi = oi_compute(oi, scene)
        final_mean = _mean_photons(oi)
        final_center_row_550 = _center_row_norm_550(oi)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "base_center_row_550_norm": base_center_row_550,
                "defocus_center_row_550_norm": defocus_center_row_550,
                "astig_center_row_550_norm": astig_center_row_550,
                "reset_center_row_550_norm": reset_center_row_550,
                "large_pupil_center_row_550_norm": large_pupil_center_row_550,
                "final_center_row_550_norm": final_center_row_550,
                "defocus_coeff": defocus_coeff,
                "vertical_astigmatism_coeff": astig_coeff,
                "final_defocus_coeff": float(oi_get(oi, "wvf", "zcoeffs", "defocus")),
                "final_vertical_astigmatism_coeff": float(oi_get(oi, "wvf", "zcoeffs", "vertical_astigmatism")),
                "pupil_diameter_mm": pupil_diameter_mm,
                "doubled_pupil_diameter_mm": 2.0 * pupil_diameter_mm,
                "initial_reset_ratio": base_mean / max(reset_mean, 1e-12),
                "initial_final_ratio": base_mean / max(final_mean, 1e-12),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_defocus_displacement_small":
        base_diopters = np.arange(50.0, 351.0, 100.0, dtype=float)
        delta_diopters = np.arange(1.0, 16.0, dtype=float)
        displacement_curves_m = np.asarray(
            optics_defocus_displacement(base_diopters[:, None], delta_diopters[None, :]),
            dtype=float,
        )

        ratio_base_diopters = np.arange(50.0, 301.0, 50.0, dtype=float)
        ratio_delta_diopters = ratio_base_diopters / 10.0
        ratio_displacement_m = np.asarray(
            optics_defocus_displacement(ratio_base_diopters, ratio_delta_diopters),
            dtype=float,
        )
        displacement_focal_length_ratio = ratio_displacement_m * ratio_base_diopters

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "base_diopters": base_diopters,
                "delta_diopters": delta_diopters,
                "displacement_curves_m": displacement_curves_m,
                "ratio_base_diopters": ratio_base_diopters,
                "ratio_delta_diopters": ratio_delta_diopters,
                "ratio_displacement_m": ratio_displacement_m,
                "displacement_to_focal_length_ratio": displacement_focal_length_ratio,
            },
            context={},
        )

    if case_name == "optics_dof_small":
        f_number = 2.0
        focal_length_m = 0.100
        object_distance_m = 2.0
        coc_diameter_m = 50e-6

        oi = oi_create()
        optics = dict(oi.fields["optics"])
        optics["f_number"] = f_number
        optics["focal_length_m"] = focal_length_m

        dof_formula_m = float(optics_dof(optics, object_distance_m, coc_diameter_m))
        coc_curve_m, x_dist_m = optics_coc(optics, object_distance_m, "nsamples", 200)
        idx1 = int(np.argmin(np.abs(coc_curve_m[:100] - coc_diameter_m)))
        idx2 = int(np.argmin(np.abs(coc_curve_m[100:] - coc_diameter_m))) + 100
        coc_dof_m = float(x_dist_m[idx2] - x_dist_m[idx1])

        object_distances_m = np.arange(0.5, 20.0 + 1e-12, 0.25, dtype=float)
        f_numbers = np.arange(2.0, 12.0 + 1e-12, 0.25, dtype=float)
        dof_surface_m = np.zeros((object_distances_m.size, f_numbers.size), dtype=float)
        optics_sweep = dict(optics)
        for column_index, sweep_f_number in enumerate(f_numbers):
            optics_sweep["f_number"] = float(sweep_f_number)
            dof_surface_m[:, column_index] = np.asarray(
                optics_dof(optics_sweep, object_distances_m, 20e-6),
                dtype=float,
            )

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "f_number": f_number,
                "focal_length_m": focal_length_m,
                "object_distance_m": object_distance_m,
                "coc_diameter_m": coc_diameter_m,
                "dof_formula_m": dof_formula_m,
                "coc_xdist_m": np.asarray(x_dist_m, dtype=float),
                "coc_curve_m": np.asarray(coc_curve_m, dtype=float),
                "coc_idx1": idx1,
                "coc_idx2": idx2,
                "coc_dof_m": coc_dof_m,
                "object_distances_m": object_distances_m,
                "f_numbers": f_numbers,
                "sweep_coc_diameter_m": 20e-6,
                "dof_surface_m": dof_surface_m,
            },
            context={},
        )

    if case_name == "optics_depth_defocus_small":
        oi = oi_create()
        optics = dict(oi.fields["optics"])
        focal_length_m = float(optics["focal_length_m"])
        lens_power_diopters = 1.0 / focal_length_m
        object_distance_m = np.linspace(focal_length_m * 1.5, 100.0 * focal_length_m, 500, dtype=float)

        focal_plane_defocus_diopters, image_distance_m = optics_depth_defocus(object_distance_m, optics)
        focal_plane_defocus_diopters = np.asarray(focal_plane_defocus_diopters, dtype=float)
        image_distance_m = np.asarray(image_distance_m, dtype=float)

        shifted_image_plane_scale = 1.1
        shifted_image_plane_m = shifted_image_plane_scale * focal_length_m
        shifted_defocus_diopters, _ = optics_depth_defocus(object_distance_m, optics, shifted_image_plane_m)
        shifted_defocus_diopters = np.asarray(shifted_defocus_diopters, dtype=float)
        focus_index = int(np.argmin(np.abs(shifted_defocus_diopters)))

        pupil_radius_m = focal_length_m / (2.0 * float(optics["f_number"]))
        pupil_radius_scales = np.array([0.5, 1.5, 3.0], dtype=float)
        w20 = ((pupil_radius_scales[None, :] * pupil_radius_m) ** 2 / 2.0) * (
            lens_power_diopters * shifted_defocus_diopters[:, None]
        ) / (lens_power_diopters + shifted_defocus_diopters[:, None])

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "focal_length_m": focal_length_m,
                "lens_power_diopters": lens_power_diopters,
                "object_distance_m": object_distance_m,
                "focal_plane_relative_defocus": focal_plane_defocus_diopters / lens_power_diopters,
                "image_distance_m": image_distance_m,
                "shifted_image_plane_scale": shifted_image_plane_scale,
                "shifted_defocus_diopters": shifted_defocus_diopters,
                "shifted_focus_object_distance_m": float(object_distance_m[focus_index]),
                "shifted_focus_object_distance_focal_lengths": float(object_distance_m[focus_index] / focal_length_m),
                "pupil_radius_m": pupil_radius_m,
                "pupil_radius_scales": pupil_radius_scales,
                "w20": np.asarray(w20, dtype=float),
            },
            context={},
        )

    if case_name == "optics_defocus_scene_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        scene_path = store.resolve("data/images/multispectral/StuffedAnimals_tungsten-hdrs.mat")
        scene = scene_from_file(scene_path, "multispectral", None, None, wave, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        max_sf = float(scene_get(scene, "max freq res", "cpd", asset_store=store))
        n_steps = min(int(np.ceil(max_sf)), 70)
        sample_sf = np.linspace(0.0, max_sf, n_steps, dtype=float)
        scene = scene_adjust_illuminant(scene, "D65.mat", asset_store=store)

        base_oi = oi_create()
        base_optics = dict(oi_get(base_oi, "optics"))
        base_optics["model"] = "shiftinvariant"

        def _build_oi(defocus_diopters: np.ndarray) -> tuple[OpticalImage, np.ndarray]:
            otf_rows, sample_sf_mm = optics_defocus_core(base_optics, sample_sf, defocus_diopters)
            defocused_optics = optics_build_2d_otf(base_optics, otf_rows, sample_sf_mm)
            current_oi = oi_set(oi_create(), "optics", defocused_optics)
            current_oi = oi_compute(current_oi, scene)
            return current_oi, np.asarray(sample_sf_mm, dtype=float)

        optics_wave = np.asarray(base_optics.get("transmittance", {}).get("wave", wave), dtype=float).reshape(-1)

        def _center_row_norm_550(current_oi: OpticalImage) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            center_row = photons[photons.shape[0] // 2, :, wave_index]
            return _channel_normalize(center_row)

        def _peak_550(current_oi: OpticalImage) -> float:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            oi_wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(oi_wave - 550.0)))
            return float(np.max(photons[:, :, wave_index]))

        defocus5 = np.full(optics_wave.shape, 5.0, dtype=float)
        oi_defocus5, sample_sf_mm = _build_oi(defocus5)

        focused = np.zeros(optics_wave.shape, dtype=float)
        oi_focus, _ = _build_oi(focused)

        focal_length_m = float(base_optics.get("focal_length_m", base_optics.get("focalLength", 0.0)))
        lens_power_diopters = 1.0 / max(focal_length_m, 1e-12)

        delta_distance_10_m = 10e-6
        actual_power_10 = 1.0 / max(focal_length_m - delta_distance_10_m, 1e-12)
        defocus_10_diopters = actual_power_10 - lens_power_diopters
        oi_miss10, _ = _build_oi(np.full(optics_wave.shape, defocus_10_diopters, dtype=float))

        delta_distance_40_m = 40e-6
        actual_power_40 = 1.0 / max(focal_length_m - delta_distance_40_m, 1e-12)
        defocus_40_diopters = actual_power_40 - lens_power_diopters
        oi_miss40, _ = _build_oi(np.full(optics_wave.shape, defocus_40_diopters, dtype=float))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": optics_wave,
                "max_sf_cpd": max_sf,
                "sample_sf_cpd": sample_sf,
                "sample_sf_mm": sample_sf_mm,
                "defocus_5_diopters": 5.0,
                "defocus_10um_diopters": float(defocus_10_diopters),
                "defocus_40um_diopters": float(defocus_40_diopters),
                "focus_center_row_550_norm": _center_row_norm_550(oi_focus),
                "defocus5_center_row_550_norm": _center_row_norm_550(oi_defocus5),
                "miss10_center_row_550_norm": _center_row_norm_550(oi_miss10),
                "miss40_center_row_550_norm": _center_row_norm_550(oi_miss40),
                "focus_peak_550": _peak_550(oi_focus),
                "defocus5_peak_550": _peak_550(oi_defocus5),
                "miss10_peak_550": _peak_550(oi_miss10),
                "miss40_peak_550": _peak_550(oi_miss40),
            },
            context={"scene": scene, "oi_focus": oi_focus, "oi_defocus5": oi_defocus5, "oi_miss10": oi_miss10, "oi_miss40": oi_miss40},
        )

    if case_name == "wvf_astigmatism_small":
        max_um = 20.0
        base_wvf = wvf_create()
        base_wvf = wvf_set(base_wvf, "lcaMethod", "human")
        base_wvf = wvf_compute(base_wvf)

        z4 = np.arange(-0.5, 1.0, 0.5, dtype=float)
        z5 = np.arange(-0.5, 1.0, 0.5, dtype=float)
        z4_grid, z5_grid = np.meshgrid(z4, z5, indexing="xy")
        zvals = np.column_stack((z4_grid.reshape(-1, order="F"), z5_grid.reshape(-1, order="F")))

        x_support_um: np.ndarray | None = None
        row_profiles: list[np.ndarray] = []
        col_profiles: list[np.ndarray] = []
        centers: list[float] = []
        current = base_wvf
        for pair in zvals:
            current = wvf_set(current, "zcoeffs", np.asarray(pair, dtype=float), ["defocus", "vertical_astigmatism"])
            current = wvf_set(current, "lcaMethod", "human")
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
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
            if x_support_um is None:
                x_support_um = np.asarray(udata["x"], dtype=float)
            row_profiles.append(np.asarray(psf[psf.shape[0] // 2, :], dtype=float))
            col_profiles.append(np.asarray(psf[:, psf.shape[1] // 2], dtype=float))
            centers.append(float(psf[psf.shape[0] // 2, psf.shape[1] // 2]))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "zvals": zvals,
                "x": np.asarray(x_support_um, dtype=float),
                "psf_mid_rows": np.vstack(row_profiles),
                "psf_mid_cols": np.vstack(col_profiles),
                "psf_centers": np.asarray(centers, dtype=float),
            },
            context={"wvf": current},
        )

    if case_name == "wvf_zernike_set_small":
        params = {
            "blockSize": 64,
            "angles": np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=float),
        }
        scene = scene_create("frequency orientation", params, asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        astigmatism_values = np.array([-1.0, 0.0, 1.0], dtype=float)
        defocus_microns = 2.0
        psf_support_um: np.ndarray | None = None
        psf_mid_rows: list[np.ndarray] = []
        oi_center_rows_550: list[np.ndarray] = []
        oi_defocus: list[float] = []
        oi_astigmatism: list[float] = []
        oi_peak_photons_550: list[float] = []

        def _normalize_profile(profile: np.ndarray) -> np.ndarray:
            profile_array = np.asarray(profile, dtype=float)
            return profile_array / max(float(np.max(np.abs(profile_array))), 1e-12)

        current_wvf = wvf_create(wave=scene_get(scene, "wave"))
        last_oi = None
        for astigmatism in astigmatism_values:
            current_wvf = wvf_set(
                current_wvf,
                "zcoeffs",
                np.array([defocus_microns, astigmatism], dtype=float),
                ["defocus", "vertical_astigmatism"],
            )
            current_wvf = wvf_compute(current_wvf)
            udata, _ = wvf_plot(
                current_wvf,
                "psf",
                "unit",
                "um",
                "wave",
                550.0,
                "plot range",
                40.0,
                "window",
                False,
            )
            psf = np.asarray(udata["z"], dtype=float)
            if psf_support_um is None:
                psf_support_um = np.asarray(udata["x"], dtype=float)
            psf_mid_rows.append(_normalize_profile(psf[psf.shape[0] // 2, :]))

            oi = oi_compute(current_wvf, scene)
            last_oi = oi
            oi_wvf = oi_get(oi, "wvf")
            oi_defocus.append(float(wvf_get(oi_wvf, "zcoeffs", "defocus")))
            oi_astigmatism.append(float(wvf_get(oi_wvf, "zcoeffs", "vertical_astigmatism")))

            photons = np.asarray(oi_get(oi, "photons"), dtype=float)
            wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(wave - 550.0)))
            center_row = photons[photons.shape[0] // 2, :, wave_index]
            oi_center_rows_550.append(_normalize_profile(center_row))
            oi_peak_photons_550.append(float(np.max(photons[:, :, wave_index])))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
                "astigmatism_values": astigmatism_values,
                "defocus_microns": defocus_microns,
                "psf_support_um": np.asarray(psf_support_um, dtype=float),
                "psf_mid_rows": np.vstack(psf_mid_rows),
                "oi_defocus_coeffs": np.asarray(oi_defocus, dtype=float),
                "oi_astigmatism_coeffs": np.asarray(oi_astigmatism, dtype=float),
                "oi_center_rows_550": np.vstack(oi_center_rows_550),
                "oi_peak_photons_550": np.asarray(oi_peak_photons_550, dtype=float),
            },
            context={"scene": scene, "wvf": current_wvf, "oi": last_oi},
        )

    if case_name == "wvf_wavefronts_small":
        indices = np.arange(1, 17, dtype=int)
        n_values, m_values = wvf_osa_index_to_zernike_nm(indices)
        x_support_mm: np.ndarray | None = None
        row_profiles: list[np.ndarray] = []
        col_profiles: list[np.ndarray] = []
        peak_abs_values: list[float] = []
        current_wvf = None

        for index in indices:
            current_wvf = wvf_create()
            current_wvf = wvf_set(current_wvf, "npixels", 801)
            current_wvf = wvf_set(current_wvf, "measured pupil size", 2.0)
            current_wvf = wvf_set(current_wvf, "calc pupil size", 2.0)
            current_wvf = wvf_set(current_wvf, "zcoeff", 1.0, int(index))
            current_wvf = wvf_compute(current_wvf)
            udata, _ = wvf_plot(
                current_wvf,
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
            if x_support_mm is None:
                x_support_mm = np.asarray(udata["x"], dtype=float).reshape(-1)
            peak_abs = max(float(np.max(np.abs(wavefront))), 1e-12)
            normalized_wavefront = wavefront / peak_abs
            row_profiles.append(np.asarray(normalized_wavefront[normalized_wavefront.shape[0] // 2, :], dtype=float))
            col_profiles.append(np.asarray(normalized_wavefront[:, normalized_wavefront.shape[1] // 2], dtype=float))
            peak_abs_values.append(peak_abs)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "indices": indices,
                "n": np.asarray(n_values, dtype=int),
                "m": np.asarray(m_values, dtype=int),
                "x": np.asarray(x_support_mm, dtype=float),
                "wavefront_mid_rows_norm": np.vstack(row_profiles),
                "wavefront_mid_cols_norm": np.vstack(col_profiles),
                "wavefront_peak_abs": np.asarray(peak_abs_values, dtype=float),
                "npixels": 801,
                "measured_pupil_mm": 2.0,
                "calc_pupil_mm": 2.0,
            },
            context={"wvf": current_wvf},
        )

    if case_name == "zernike_interpolation_small":
        raw = store.load_mat("data/optics/zernike_doubleGauss.mat")
        data = raw["data"]
        wavelengths = np.asarray(data.wavelengths, dtype=float).reshape(-1)
        image_heights = np.asarray(data.image_heights, dtype=float).reshape(-1)
        zcoeffs = data.zernikeCoefficients

        image_height_indices = np.arange(1, 22, 4, dtype=int)
        this_wave_index = 3
        test_index = 6
        wavelength_nm = float(wavelengths[this_wave_index - 1])
        image_heights_test = image_heights[image_height_indices - 1]
        zernike_coeff_matrix = np.vstack(
            [
                np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(index)}"), dtype=float).reshape(-1)
                for index in image_height_indices
            ]
        )
        nearest_indices = np.asarray(_find_nearest_two(image_height_indices, test_index), dtype=int)
        test_height = float(image_heights[test_index - 1])
        zernike_gt = np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{test_index}"), dtype=float).reshape(-1)
        zernike_interpolated = np.array(
            [np.interp(test_height, image_heights_test, zernike_coeff_matrix[:, column]) for column in range(zernike_coeff_matrix.shape[1])],
            dtype=float,
        )
        validation = zernike_interpolated - zernike_gt

        psf_interpolated, _ = _generate_fringe_psf(zernike_interpolated)
        psf_gt, _ = _generate_fringe_psf(zernike_gt)
        psf_1, _ = _generate_fringe_psf(np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(nearest_indices[0])}"), dtype=float))
        psf_2, _ = _generate_fringe_psf(np.asarray(getattr(zcoeffs, f"wave_{this_wave_index}_field_{int(nearest_indices[1])}"), dtype=float))
        psf_interp_space = (
            psf_1 * (test_height - float(image_heights[int(nearest_indices[0]) - 1])) / float(nearest_indices[1] - nearest_indices[0])
            + psf_2 * (float(nearest_indices[1]) - test_height) / float(nearest_indices[1] - nearest_indices[0])
        )

        middle_row = psf_gt.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "this_wave_index": this_wave_index,
                "wavelength_nm": wavelength_nm,
                "image_height_indices": image_height_indices,
                "image_heights_test": image_heights_test,
                "test_index": test_index,
                "test_height": test_height,
                "nearest_indices": nearest_indices,
                "zernike_gt": zernike_gt,
                "zernike_interpolated": zernike_interpolated,
                "validation": validation,
                "validation_rmse": float(np.sqrt(np.mean(validation**2))),
                "psf_interpolated_mid_row_norm": _channel_normalize(psf_interpolated[middle_row, :]),
                "psf_gt_mid_row_norm": _channel_normalize(psf_gt[middle_row, :]),
                "psf_interp_space_mid_row_norm": _channel_normalize(psf_interp_space[middle_row, :]),
                "psf_interpolated_peak": float(np.max(psf_interpolated)),
                "psf_gt_peak": float(np.max(psf_gt)),
                "psf_interp_space_peak": float(np.max(psf_interp_space)),
            },
            context={},
        )

    if case_name == "wvf_plot_script_sequence_small":
        wave_550 = 550.0
        wave_460 = 460.0

        wvf = wvf_create()
        wvf = wvf_set(wvf, "wave", wave_550)
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)

        udata_550_um, _ = wvf_plot(wvf, "1d psf", "unit", "um", "wave", wave_550, "window", False)
        udata_550_mm, _ = wvf_plot(wvf, "1d psf", "unit", "mm", "wave", wave_550, "window", False)
        udata_550_norm, _ = wvf_plot(wvf, "1d psf normalized", "unit", "mm", "wave", wave_550, "window", False)

        wvf = wvf_set(wvf, "wave", wave_460)
        wvf = wvf_compute(wvf)

        udata_460_angle, _ = wvf_plot(
            wvf,
            "image psf angle",
            "unit",
            "min",
            "wave",
            wave_460,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf_angle = np.asarray(udata_460_angle["z"], dtype=float)
        udata_460_phase, _ = wvf_plot(
            wvf,
            "image pupil phase",
            "unit",
            "mm",
            "wave",
            wave_460,
            "plot range",
            2.0,
            "window",
            False,
        )
        pupil_phase = np.asarray(udata_460_phase["z"], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave_550_nm": wave_550,
                "wave_460_nm": wave_460,
                "line_550_um_x": np.asarray(udata_550_um["x"], dtype=float),
                "line_550_um_y_norm": _channel_normalize(np.asarray(udata_550_um["y"], dtype=float)),
                "line_550_mm_x": np.asarray(udata_550_mm["x"], dtype=float),
                "line_550_mm_y_norm": _channel_normalize(np.asarray(udata_550_mm["y"], dtype=float)),
                "line_550_mm_norm_y": np.asarray(udata_550_norm["y"], dtype=float),
                "psf_angle_460_x": np.asarray(udata_460_angle["x"], dtype=float),
                "psf_angle_460_mid_row_norm": _channel_normalize(psf_angle[psf_angle.shape[0] // 2, :]),
                "psf_angle_460_center": float(psf_angle[psf_angle.shape[0] // 2, psf_angle.shape[1] // 2]),
                "pupil_phase_460_x": np.asarray(udata_460_phase["x"], dtype=float),
                "pupil_phase_460_mid_row": np.asarray(pupil_phase[pupil_phase.shape[0] // 2, :], dtype=float),
                "pupil_phase_460_center": float(pupil_phase[pupil_phase.shape[0] // 2, pupil_phase.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_diffraction_small":
        flength_mm = 6.0
        flength_m = flength_mm * 1e-3
        f_number = 3.0
        this_wave = 550.0
        canonical_samples = 41

        def _canonical_profile(values: np.ndarray) -> np.ndarray:
            profile = np.asarray(values, dtype=float)
            if profile.ndim == 1:
                support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
                query = np.linspace(-1.0, 1.0, canonical_samples, dtype=float)
                return np.interp(query, support, profile)
            if profile.ndim == 2:
                return np.vstack([_canonical_profile(row) for row in profile])
            raise ValueError("Expected a vector or row-stacked matrix.")

        wvf = wvf_create()
        wvf = wvf_set(wvf, "calc pupil diameter", flength_mm / f_number)
        wvf = wvf_set(wvf, "focal length", flength_m)
        wvf = wvf_compute(wvf)

        _, _ = wvf_plot(
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
        oi_psfx_udata, _ = oi_plot(oi, "psfxaxis", None, this_wave, "um")

        pupil_mm = np.linspace(1.5, 8.0, 4, dtype=float)
        pupil_550_airy: list[float] = []
        current = wvf
        for pupil in pupil_mm:
            current = wvf_set(current, "calc pupil diameter", float(pupil))
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
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
            pupil_550_airy.append(float(airy_disk(this_wave, float(wvf_get(current, "fnumber")), "units", "um", "diameter", True)))

        this_wave = 400.0
        current = wvf_set(current, "calc wave", this_wave)
        pupil_400_airy: list[float] = []
        for pupil in pupil_mm:
            current = wvf_set(current, "calc pupil diameter", float(pupil))
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
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
            pupil_400_airy.append(float(airy_disk(this_wave, float(wvf_get(current, "fnumber")), "units", "um", "diameter", True)))

        current = wvf_set(current, "calc pupil diameter", 3.0)
        current = wvf_set(current, "calc wave", 550.0)
        wavelength_list = np.linspace(400.0, 700.0, 4, dtype=float)
        lca_rows: list[np.ndarray] = []
        lca_airy: list[float] = []
        for wavelength in wavelength_list:
            current = wvf_set(current, "calc wave", float(wavelength))
            current = wvf_set(current, "lcaMethod", "human")
            current = wvf_compute(current)
            udata, _ = wvf_plot(
                current,
                "image psf",
                "unit",
                "um",
                "wave",
                float(wavelength),
                "plot range",
                20.0,
                "airy disk",
                True,
                "window",
                False,
            )
            z = np.asarray(udata["z"], dtype=float)
            lca_rows.append(np.asarray(z[z.shape[0] // 2, :], dtype=float))
            lca_airy.append(float(airy_disk(float(wavelength), float(wvf_get(current, "fnumber")), "units", "um", "diameter", True)))

        base_flength_m = 7e-3
        base_fnumber = 4.0
        diffraction_wvf = wvf_create()
        diffraction_wvf = wvf_set(diffraction_wvf, "calc pupil diameter", (base_flength_m * 1e3) / base_fnumber)
        diffraction_wvf = wvf_set(diffraction_wvf, "focal length", base_flength_m)
        diffraction_wvf = wvf_compute(diffraction_wvf)

        focal_length_sweep_m = np.array([base_flength_m / 2.0, base_flength_m, base_flength_m * 2.0], dtype=float)
        focal_length_um_per_degree = focal_length_sweep_m * 1e6 * (2.0 * np.tan(np.deg2rad(0.5)))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "base_fnumber_ratio_oi_wvf": float(oi_get(oi, "optics fnumber")) / max(float(wvf_get(wvf, "fnumber")), 1e-12),
                "base_airy_diameter_um": float(airy_disk(550.0, f_number, "units", "um", "diameter", True)),
                "base_oi_psfx_data": np.asarray(oi_psfx_udata["data"], dtype=float),
                "pupil_mm": pupil_mm,
                "pupil_550_airy_diameter_um": np.asarray(pupil_550_airy, dtype=float),
                "pupil_400_airy_diameter_um": np.asarray(pupil_400_airy, dtype=float),
                "lca_wavelength_nm": wavelength_list,
                "lca_airy_diameter_um": np.asarray(lca_airy, dtype=float),
                "lca_mid_rows": _canonical_profile(np.vstack(lca_rows)),
                "focal_length_sweep_mm": focal_length_sweep_m * 1e3,
                "focal_length_um_per_degree": np.asarray(focal_length_um_per_degree, dtype=float),
            },
            context={"wvf": diffraction_wvf, "oi": oi},
        )

    if case_name == "wvf_spatial_sampling_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        this_wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
        focal_length_m = 7e-3
        f_number = 4.0
        wvf = wvf_set(wvf, "calc pupil diameter", (focal_length_m * 1e3) / f_number, "mm")
        wvf = wvf_set(wvf, "focal length", focal_length_m, "m")
        wvf = wvf_compute(wvf)
        psf_xaxis = dict(wvf_get(wvf, "psf xaxis", "um", this_wave))
        pupil_amplitude = np.asarray(wvf_get(wvf, "pupil function amplitude", this_wave), dtype=float)
        pupil_phase = np.asarray(wvf_get(wvf, "pupil function phase", this_wave), dtype=float)
        middle_row = pupil_amplitude.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wvf_get(wvf, "wave"),
                "npixels": wvf_get(wvf, "npixels"),
                "calc_nwave": wvf_get(wvf, "calc nwave"),
                "psf_sample_spacing_arcmin": wvf_get(wvf, "psf sample spacing"),
                "ref_psf_sample_interval_arcmin": wvf_get(wvf, "ref psf sample interval"),
                "um_per_degree": wvf_get(wvf, "um per degree"),
                "pupil_plane_size_mm": wvf_get(wvf, "pupil plane size", "mm", this_wave),
                "pupil_sample_spacing_mm": wvf_get(wvf, "pupil sample spacing", "mm", this_wave),
                "pupil_positions_mm": wvf_get(wvf, "pupil positions", this_wave, "mm"),
                "psf_xaxis_um": np.asarray(psf_xaxis["samp"], dtype=float),
                "psf_xaxis_data": np.asarray(psf_xaxis["data"], dtype=float),
                "pupil_amp_row": pupil_amplitude[middle_row, :],
                "pupil_phase_row": pupil_phase[middle_row, :],
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_spatial_controls_small":
        this_wave = 550.0
        focal_length_m = 7e-3
        focal_length_mm = focal_length_m * 1e3
        f_number = 4.0

        def _build_wvf() -> dict[str, Any]:
            wavefront = wvf_create(wave=np.array([this_wave], dtype=float))
            wavefront = wvf_set(wavefront, "calc pupil diameter", focal_length_mm / f_number, "mm")
            wavefront = wvf_set(wavefront, "focal length", focal_length_m, "m")
            return wavefront

        def _measure(wavefront: dict[str, Any]) -> dict[str, float]:
            computed = wvf_compute(wavefront)
            um_per_degree = float(wvf_get(computed, "um per degree"))
            return {
                "npixels": float(wvf_get(computed, "npixels")),
                "psf_sample_spacing_arcmin": float(wvf_get(computed, "psf sample spacing")),
                "um_per_degree": um_per_degree,
                "pupil_plane_size_mm": float(wvf_get(computed, "pupil plane size", "mm", this_wave)),
                "pupil_sample_spacing_mm": float(wvf_get(computed, "pupil sample spacing", "mm", this_wave)),
                # Match the legacy MATLAB/Octave wvfGet('focal length', 'm')
                # contract used by the parity baseline, which derives
                # focal length from the returned "um per degree" value.
                "focal_length_m": (um_per_degree / (2.0 * np.tan(np.deg2rad(0.5)))) * 1e-6,
            }

        base = _measure(_build_wvf())

        reduced_pixels = _build_wvf()
        reduced_pixels = wvf_set(reduced_pixels, "npixels", round(base["npixels"] / 4.0))
        reduced_pixels = _measure(reduced_pixels)

        enlarged_pupil_plane = _build_wvf()
        enlarged_pupil_plane = wvf_set(enlarged_pupil_plane, "pupil plane size", base["pupil_plane_size_mm"] * 4.0, "mm")
        enlarged_pupil_plane = _measure(enlarged_pupil_plane)

        reduced_pupil_plane = _build_wvf()
        reduced_pupil_plane = wvf_set(reduced_pupil_plane, "pupil plane size", base["pupil_plane_size_mm"] / 4.0, "mm")
        reduced_pupil_plane = _measure(reduced_pupil_plane)

        focal_half = _build_wvf()
        focal_half = wvf_set(focal_half, "focal length", focal_length_m / 2.0, "m")
        focal_half = _measure(focal_half)

        focal_double = _build_wvf()
        focal_double = wvf_set(focal_double, "focal length", focal_length_m * 2.0, "m")
        focal_double = _measure(focal_double)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.array([this_wave], dtype=float),
                "base_npixels": base["npixels"],
                "base_psf_sample_spacing_arcmin": base["psf_sample_spacing_arcmin"],
                "base_um_per_degree": base["um_per_degree"],
                "base_pupil_plane_size_mm": base["pupil_plane_size_mm"],
                "reduced_pixels_npixels": reduced_pixels["npixels"],
                "reduced_pixels_psf_sample_spacing_arcmin": reduced_pixels["psf_sample_spacing_arcmin"],
                "reduced_pixels_um_per_degree": reduced_pixels["um_per_degree"],
                "pupil_plane_x4_psf_sample_spacing_arcmin": enlarged_pupil_plane["psf_sample_spacing_arcmin"],
                "pupil_plane_x4_um_per_degree": enlarged_pupil_plane["um_per_degree"],
                "pupil_plane_x4_size_mm": enlarged_pupil_plane["pupil_plane_size_mm"],
                "pupil_plane_div4_psf_sample_spacing_arcmin": reduced_pupil_plane["psf_sample_spacing_arcmin"],
                "pupil_plane_div4_um_per_degree": reduced_pupil_plane["um_per_degree"],
                "pupil_plane_div4_size_mm": reduced_pupil_plane["pupil_plane_size_mm"],
                "focal_length_half_m": focal_half["focal_length_m"],
                "focal_length_half_psf_sample_spacing_arcmin": focal_half["psf_sample_spacing_arcmin"],
                "focal_length_half_um_per_degree": focal_half["um_per_degree"],
                "focal_length_double_m": focal_double["focal_length_m"],
                "focal_length_double_psf_sample_spacing_arcmin": focal_double["psf_sample_spacing_arcmin"],
                "focal_length_double_um_per_degree": focal_double["um_per_degree"],
            },
            context={},
        )

    if case_name == "wvf_compute_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 101)
        wvf = wvf_pupil_function(wvf)
        wvf = wvf_compute_psf(wvf, "compute pupil func", False)
        wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        pupil_amp = np.asarray(wvf_get(wvf, "pupil function amplitude", wave), dtype=float)
        pupil_phase = np.asarray(wvf_get(wvf, "pupil function phase", wave), dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wvf_get(wvf, "wave"),
                "npixels": wvf_get(wvf, "npixels"),
                "psf_sum": float(np.sum(psf)),
                "psf_mid_row": psf[middle_row, :],
                "pupil_amp_row": pupil_amp[middle_row, :],
                "pupil_phase_row": pupil_phase[middle_row, :],
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_otf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "2d otf", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_otf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "2d otf normalized", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_angle_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d otf angle", "unit", "deg", "wave", 550.0, "plot range", 10.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "1d otf angle normalized",
            "unit",
            "deg",
            "wave",
            550.0,
            "plot range",
            10.0,
            "window",
            False,
        )
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d otf", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_otf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d otf normalized", "unit", "mm", "wave", 550.0, "plot range", 300.0, "window", False)
        otf = np.asarray(udata["otf"], dtype=float)
        middle_row = otf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fx": np.asarray(udata["fx"], dtype=float),
                "otf_mid_row": otf[middle_row, :],
                "otf_center": float(otf[middle_row, otf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_pupil_amp_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image pupil amp", "unit", "mm", "wave", 550.0, "plot range", 2.0, "window", False)
        amp = np.asarray(udata["z"], dtype=float)
        middle_row = amp.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "amp_mid_row": amp[middle_row, :],
                "amp_center": float(amp[middle_row, amp.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_pupil_amplitude_space_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d pupil amplitude space",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            2.0,
            "window",
            False,
        )
        amp = np.asarray(udata["z"], dtype=float)
        middle_row = amp.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "amp_mid_row": amp[middle_row, :],
                "amp_center": float(amp[middle_row, amp.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_pupil_phase_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image pupil phase", "unit", "mm", "wave", 550.0, "plot range", 2.0, "window", False)
        phase = np.asarray(udata["z"], dtype=float)
        middle_row = phase.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "phase_mid_row": phase[middle_row, :],
                "phase_center": float(phase[middle_row, phase.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_pupil_phase_space_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d pupil phase space",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            2.0,
            "window",
            False,
        )
        phase = np.asarray(udata["z"], dtype=float)
        middle_row = phase.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "phase_mid_row": phase[middle_row, :],
                "phase_center": float(phase[middle_row, phase.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_wavefront_aberrations_small":
        wvf = wvf_create(
            wave=np.array([550.0], dtype=float),
            zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.12], dtype=float),
        )
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image wavefront aberrations",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            1.5,
            "window",
            False,
        )
        wavefront = np.asarray(udata["z"], dtype=float)
        middle_row = wavefront.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavefront_mid_row": wavefront[middle_row, :],
                "wavefront_center": float(wavefront[middle_row, wavefront.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_wavefront_aberrations_space_small":
        wvf = wvf_create(
            wave=np.array([550.0], dtype=float),
            zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.12], dtype=float),
        )
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d wavefront aberrations space",
            "unit",
            "mm",
            "wave",
            550.0,
            "plot range",
            1.5,
            "window",
            False,
        )
        wavefront = np.asarray(udata["z"], dtype=float)
        middle_row = wavefront.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavefront_mid_row": wavefront[middle_row, :],
                "wavefront_center": float(wavefront[middle_row, wavefront.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image psf", "unit", "um", "wave", 550.0, "plot range", 20.0, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_airy_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "airy disk",
            True,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
                "airy_disk_radius": float(udata["airyDiskRadius"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf normalized",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "psf", "unit", "mm", "wave", 550.0, "plot range", 0.05, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "psf normalized", "unit", "mm", "wave", 550.0, "plot range", 0.05, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_angle_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "image psf angle", "unit", "min", "wave", 460.0, "plot range", 1.0, "window", False)
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_image_psf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "image psf angle normalized",
            "unit",
            "min",
            "wave",
            460.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_psf_angle_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d psf angle",
            "unit",
            "min",
            "wave",
            460.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_2d_psf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([460.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "2d psf angle normalized",
            "unit",
            "min",
            "wave",
            460.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        psf = np.asarray(udata["z"], dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "psf_mid_row": psf[middle_row, :],
                "psf_center": float(psf[middle_row, psf.shape[1] // 2]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d psf", "unit", "um", "wave", 550.0, "plot range", 10.0, "window", False)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_space_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d psf space", "unit", "um", "wave", 550.0, "plot range", 10.0, "window", False)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(wvf, "1d psf normalized", "unit", "um", "wave", 550.0, "plot range", 10.0, "window", False)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_angle_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "1d psf angle",
            "unit",
            "min",
            "wave",
            550.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_1d_psf_angle_normalized_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "1d psf angle normalized",
            "unit",
            "min",
            "wave",
            550.0,
            "plot range",
            1.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "y": np.asarray(udata["y"], dtype=float),
                "peak": float(np.max(np.asarray(udata["y"], dtype=float))),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_xaxis_airy_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf xaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "airy disk",
            True,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "airy_disk_radius": float(udata["airyDiskRadius"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psfxaxis_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf xaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psf_yaxis_airy_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf yaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "airy disk",
            True,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
                "airy_disk_radius": float(udata["airyDiskRadius"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_plot_psfyaxis_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "spatial samples", 401)
        wvf = wvf_compute(wvf)
        udata, _ = wvf_plot(
            wvf,
            "psf yaxis",
            "unit",
            "um",
            "wave",
            550.0,
            "plot range",
            20.0,
            "window",
            False,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "samp": np.asarray(udata["samp"], dtype=float),
                "data": np.asarray(udata["data"], dtype=float),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_psf2zcoeff_error_small":
        wvf = wvf_create(wave=np.array([550.0], dtype=float))
        wvf = wvf_set(wvf, "zcoeffs", 0.2, "defocus")
        wvf = wvf_set(wvf, "zcoeffs", 0.0, "vertical_astigmatism")
        wvf = wvf_compute(wvf)
        this_wave_nm = float(wvf_get(wvf, "wave", 1))
        this_wave_um = float(wvf_get(wvf, "wave", "um", 1))
        pupil_size_mm = float(wvf_get(wvf, "pupil size", "mm"))
        z_pupil_diameter_mm = float(wvf_get(wvf, "z pupil diameter"))
        pupil_plane_size_mm = float(wvf_get(wvf, "pupil plane size", "mm", this_wave_nm))
        n_pixels = int(wvf_get(wvf, "spatial samples"))
        psf_target = np.asarray(wvf_get(wvf, "psf", this_wave_nm), dtype=float)
        query_zcoeffs = np.asarray([0.0, 0.0, 0.0, 0.0, 0.15, 0.02], dtype=float)
        error = psf_to_zcoeff_error(
            query_zcoeffs,
            psf_target,
            pupil_size_mm,
            z_pupil_diameter_mm,
            pupil_plane_size_mm,
            this_wave_um,
            n_pixels,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave_um": this_wave_um,
                "pupil_size_mm": pupil_size_mm,
                "z_pupil_diameter_mm": z_pupil_diameter_mm,
                "pupil_plane_size_mm": pupil_plane_size_mm,
                "n_pixels": n_pixels,
                "query_zcoeffs": query_zcoeffs,
                "error": error,
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_aperture_polygon_clean_small":
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
        middle_row = aperture.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "image": np.asarray(aperture, dtype=float),
                "mid_row": np.asarray(aperture[middle_row, :], dtype=float),
                "image_sum": float(np.sum(aperture)),
                "nsides": int(params["nsides"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "wvf_compute_aperture_polygon_small":
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
        wvf = wvf_compute(wvf, "aperture", aperture)
        wave = float(np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)[0])
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        pupil_amp = np.asarray(wvf_get(wvf, "pupil function amplitude", wave), dtype=float)
        middle_row = psf.shape[0] // 2
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "psf_sum": float(np.sum(psf)),
                "psf_mid_row": np.asarray(psf[middle_row, :], dtype=float),
                "pupil_amp_row": np.asarray(pupil_amp[middle_row, :], dtype=float),
                "nsides": int(params["nsides"]),
            },
            context={"wvf": wvf},
        )

    if case_name == "oi_lswavelength_diffraction_small":
        oi = oi_create("diffraction limited")
        optics = dict(oi.fields["optics"])
        optics["focal_length_m"] = 0.017
        optics["f_number"] = 17.0 / 3.0
        oi.fields["optics"] = optics
        udata, _ = oi_plot(oi, "ls wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "lsWave": np.asarray(udata["lsWave"], dtype=float),
            },
            context={"oi": oi},
        )

    if case_name == "oi_lswavelength_wvf_small":
        wvf = wvf_create(wave=np.array([450.0, 550.0, 650.0], dtype=float))
        wvf = wvf_set(wvf, "focal length", 8.0, "mm")
        wvf = wvf_set(wvf, "pupil diameter", 3.0, "mm")
        wvf = wvf_compute(wvf)
        oi = wvf_to_oi(wvf)
        udata, _ = oi_plot(oi, "ls wavelength")
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "x": np.asarray(udata["x"], dtype=float),
                "wavelength": np.asarray(udata["wavelength"], dtype=float),
                "lsWave": np.asarray(udata["lsWave"], dtype=float),
            },
            context={"wvf": wvf, "oi": oi},
        )

    if case_name == "oi_diffraction_limited_default":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        return ParityCaseResult(
            payload={"case_name": case_name, "wave": oi.fields["wave"], "photons": oi.data["photons"]},
            context={"scene": scene, "oi": oi},
        )

    if case_name == "oi_cos4th_small":
        scene = scene_create("uniform d65", 512, asset_store=store)
        scene = scene_set(scene, "fov", 80)

        oi = oi_create("shift invariant", asset_store=store)
        focal_length_default_m = float(oi_get(oi, "optics focal length"))
        oi = oi_compute(oi, scene)
        size_default = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
        support_default = dict(oi_get(oi, "spatial support linear", "um"))
        illuminance_default = np.asarray(oi_get(oi, "illuminance"), dtype=float)
        mean_illuminance_default = float(np.mean(illuminance_default))
        center_row = int(np.rint(size_default[1] / 2.0))

        oi = oi_set(oi, "optics focal length", 4.0 * focal_length_default_m)
        focal_length_long_m = float(oi_get(oi, "optics focal length"))
        oi = oi_compute(oi, scene)
        size_long = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
        support_long = dict(oi_get(oi, "spatial support linear", "um"))
        illuminance_long = np.asarray(oi_get(oi, "illuminance"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "focal_length_default_m": focal_length_default_m,
                "focal_length_long_m": focal_length_long_m,
                "size_default": size_default,
                "size_long": size_long,
                "center_row": center_row,
                "edge_row": 20,
                "pos_default_um": np.asarray(support_default["x"], dtype=float),
                "center_line_default_lux": np.asarray(illuminance_default[center_row - 1, :], dtype=float),
                "mean_illuminance_default_lux": mean_illuminance_default,
                "pos_long_um": np.asarray(support_long["x"], dtype=float),
                "center_line_long_lux": np.asarray(illuminance_long[center_row - 1, :], dtype=float),
                "edge_line_long_lux": np.asarray(illuminance_long[20 - 1, :], dtype=float),
                "mean_illuminance_long_lux": float(np.mean(illuminance_long)),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_diffraction_small":
        scene = scene_create("point array", 128, 16, "d65", 1, asset_store=store)
        scene = scene_set(scene, "h fov", 1.0)

        oi = oi_create(asset_store=store)
        oi = oi_compute(oi, scene)
        default_f_number = float(oi_get(oi, "optics f number"))
        default_size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)

        oi = oi_set(oi, "name", "Default f/#")
        oi = oi_set(oi, "optics fnumber", 12.0)
        oi = oi_set(oi, "name", "Large f/#")
        oi = oi_compute(oi, scene)

        psf_udata, _ = oi_plot(oi, "psf 550")
        ls_udata, _ = oi_plot(oi, "ls wavelength")
        photons = np.asarray(oi_get(oi, "photons"), dtype=float)
        wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
        wave_index_550 = int(np.argmin(np.abs(wave - 550.0)))
        center_row = _channel_normalize(photons[photons.shape[0] // 2, :, wave_index_550])
        peak = max(float(np.max(center_row)), 1e-12)
        oi_center_row_550_widths = []
        for threshold in (0.50, 0.10, 0.01):
            active = np.flatnonzero(center_row >= (threshold * peak))
            oi_center_row_550_widths.append(int(active[-1] - active[0] + 1) if active.size else 0)

        focal_length_mm = float(oi_get(oi, "optics focal length", "mm"))
        pupil_diameter_mm = float(oi_get(oi, "optics pupil diameter", "mm"))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_wave": np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "default_f_number": default_f_number,
                "default_oi_size": default_size,
                "large_f_number": float(oi_get(oi, "optics f number")),
                "large_oi_size": np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1),
                "focal_length_mm": focal_length_mm,
                "pupil_diameter_mm": pupil_diameter_mm,
                "focal_to_pupil_ratio": focal_length_mm / max(pupil_diameter_mm, 1e-12),
                "psf_x": np.asarray(psf_udata["x"], dtype=float),
                "psf_y": np.asarray(psf_udata["y"], dtype=float),
                "psf_550": np.asarray(psf_udata["psf"], dtype=float),
                "ls_x_um": np.asarray(ls_udata["x"], dtype=float),
                "ls_wavelength": np.asarray(ls_udata["wavelength"], dtype=float),
                "ls_wave": np.asarray(ls_udata["lsWave"], dtype=float),
                "oi_center_row_550_widths": np.asarray(oi_center_row_550_widths, dtype=int),
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "optics_flare_small":
        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        def _profile_widths(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            peak = max(float(np.max(profile)), 1e-12)
            widths: list[int] = []
            for threshold in thresholds:
                active = np.flatnonzero(profile >= (threshold * peak))
                widths.append(int(active[-1] - active[0] + 1) if active.size else 0)
            return np.asarray(widths, dtype=int)

        def _aperture_stats(aperture: np.ndarray) -> tuple[float, float, float]:
            image = np.asarray(aperture, dtype=float)
            return float(np.sum(image)), float(np.mean(image)), float(np.mean(image < 0.95))

        def _wvf_psf_payload(current_wvf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            psf = np.asarray(wvf_get(current_wvf, "psf", 550.0), dtype=float)
            row = _canonical_profile(_channel_normalize(psf[psf.shape[0] // 2, :]))
            return row, _profile_widths(row, (0.50, 0.10, 0.01))

        def _oi_center_row_widths(current_oi: OpticalImage) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(wave - 550.0)))
            row = _canonical_profile(_channel_normalize(photons[photons.shape[0] // 2, :, wave_index]))
            return _profile_widths(row, (0.50, 0.10, 0.01))

        def _oi_mean_550(current_oi: OpticalImage) -> float:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(wave - 550.0)))
            return float(np.mean(photons[:, :, wave_index]))

        seed_initial = 1
        seed_five = 2
        seed_defocus = 3

        point_scene = scene_create("point array", 384, 128, asset_store=store)
        point_scene = scene_set(point_scene, "fov", 1.0)
        hdr_scene = scene_create("hdr", asset_store=store)
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
            seed_initial,
        )
        wvf_initial = wvf_compute(base_wvf, "aperture", aperture_initial)
        initial_psf_row, initial_psf_widths = _wvf_psf_payload(wvf_initial)
        initial_point_oi = oi_crop(oi_compute(wvf_initial, point_scene), "border")
        initial_hdr_oi = oi_compute(wvf_initial, hdr_scene)

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
            seed_five,
        )
        wvf_five = wvf_compute(base_wvf, "aperture", aperture_five)
        five_psf_row, five_psf_widths = _wvf_psf_payload(wvf_five)
        five_point_oi = oi_crop(oi_compute(wvf_five, point_scene), "border")
        five_hdr_oi = oi_crop(oi_compute(wvf_five, hdr_scene), "border")

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
            seed_defocus,
        )
        defocus_wvf = wvf_pupil_function(defocus_wvf, "aperture function", aperture_defocus)
        defocus_wvf = wvf_compute_psf(defocus_wvf, "compute pupil func", False)
        defocus_psf_row, defocus_psf_widths = _wvf_psf_payload(defocus_wvf)
        defocus_hdr_oi = oi_compute(defocus_wvf, hdr_scene)

        initial_aperture_sum, initial_aperture_mean, initial_dark_fraction = _aperture_stats(aperture_initial)
        five_aperture_sum, five_aperture_mean, five_dark_fraction = _aperture_stats(aperture_five)
        defocus_aperture_sum, defocus_aperture_mean, defocus_dark_fraction = _aperture_stats(aperture_defocus)
        initial_hdr_mean_550 = _oi_mean_550(initial_hdr_oi)
        five_hdr_mean_550 = _oi_mean_550(five_hdr_oi)
        defocus_hdr_mean_550 = _oi_mean_550(defocus_hdr_oi)
        hdr_mean_denominator = max(initial_hdr_mean_550, 1e-12)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pupil_diameter_mm": float(wvf_get(base_wvf, "calc pupil diameter", "mm")),
                "focal_length_mm": float(wvf_get(base_wvf, "focal length", "mm")),
                "f_number": float(wvf_get(base_wvf, "fnumber")),
                "point_scene_fov_deg": float(scene_get(point_scene, "fov")),
                "hdr_scene_fov_deg": float(scene_get(hdr_scene, "fov")),
                "seed_initial": seed_initial,
                "seed_five": seed_five,
                "seed_defocus": seed_defocus,
                "initial_nsides": int(params_initial["nsides"]),
                "initial_aperture_sum": initial_aperture_sum,
                "initial_aperture_mean": initial_aperture_mean,
                "initial_aperture_dark_fraction": initial_dark_fraction,
                "initial_psf_center_row_550_norm": initial_psf_row,
                "initial_psf_widths": initial_psf_widths,
                "initial_point_oi_size": np.asarray(oi_get(initial_point_oi, "size"), dtype=int),
                "initial_point_oi_center_row_550_widths": _oi_center_row_widths(initial_point_oi),
                "initial_hdr_oi_size": np.asarray(oi_get(initial_hdr_oi, "size"), dtype=int),
                "initial_hdr_mean_photons_550_ratio": initial_hdr_mean_550 / hdr_mean_denominator,
                "five_nsides": int(params_five["nsides"]),
                "five_aperture_sum": five_aperture_sum,
                "five_aperture_mean": five_aperture_mean,
                "five_aperture_dark_fraction": five_dark_fraction,
                "five_psf_center_row_550_norm": five_psf_row,
                "five_psf_widths": five_psf_widths,
                "five_point_oi_size": np.asarray(oi_get(five_point_oi, "size"), dtype=int),
                "five_point_oi_center_row_550_widths": _oi_center_row_widths(five_point_oi),
                "five_hdr_oi_size": np.asarray(oi_get(five_hdr_oi, "size"), dtype=int),
                "five_hdr_mean_photons_550_ratio": five_hdr_mean_550 / hdr_mean_denominator,
                "defocus_zcoeff": float(wvf_get(defocus_wvf, "zcoeffs", "defocus")),
                "defocus_nsides": int(params_defocus["nsides"]),
                "defocus_aperture_sum": defocus_aperture_sum,
                "defocus_aperture_mean": defocus_aperture_mean,
                "defocus_aperture_dark_fraction": defocus_dark_fraction,
                "defocus_psf_center_row_550_norm": defocus_psf_row,
                "defocus_psf_widths": defocus_psf_widths,
                "defocus_hdr_oi_size": np.asarray(oi_get(defocus_hdr_oi, "size"), dtype=int),
                "defocus_hdr_mean_photons_550_ratio": defocus_hdr_mean_550 / hdr_mean_denominator,
            },
            context={
                "point_scene": point_scene,
                "hdr_scene": hdr_scene,
                "wvf_initial": wvf_initial,
                "wvf_five": wvf_five,
                "defocus_wvf": defocus_wvf,
                "initial_point_oi": initial_point_oi,
                "initial_hdr_oi": initial_hdr_oi,
                "five_point_oi": five_point_oi,
                "five_hdr_oi": five_hdr_oi,
                "defocus_hdr_oi": defocus_hdr_oi,
            },
        )

    if case_name == "optics_flare2_small":
        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        def _profile_widths(values: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            peak = max(float(np.max(profile)), 1e-12)
            widths: list[int] = []
            for threshold in thresholds:
                active = np.flatnonzero(profile >= (threshold * peak))
                widths.append(int(active[-1] - active[0] + 1) if active.size else 0)
            return np.asarray(widths, dtype=int)

        def _aperture_stats(aperture: np.ndarray) -> tuple[float, float, float]:
            image = np.asarray(aperture, dtype=float)
            return float(np.sum(image)), float(np.mean(image)), float(np.mean(image < 0.95))

        def _wvf_psf_payload(current_wvf: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            psf = np.asarray(wvf_get(current_wvf, "psf", 550.0), dtype=float)
            row = _canonical_profile(_channel_normalize(psf[psf.shape[0] // 2, :]))
            return row, _profile_widths(row, (0.50, 0.10, 0.01))

        def _oi_center_row_widths(current_oi: OpticalImage) -> np.ndarray:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(wave - 550.0)))
            row = _canonical_profile(_channel_normalize(photons[photons.shape[0] // 2, :, wave_index]))
            return _profile_widths(row, (0.50, 0.10, 0.01))

        def _oi_mean_550(current_oi: OpticalImage) -> float:
            photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
            wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
            wave_index = int(np.argmin(np.abs(wave - 550.0)))
            return float(np.mean(photons[:, :, wave_index]))

        seed_initial = 4
        seed_five = 5
        seed_defocus = 6

        point_scene = scene_create("point array", 384, 128, asset_store=store)
        point_scene = scene_set(point_scene, "fov", 1.0)
        hdr_scene = scene_create("hdr", asset_store=store)
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
            seed_initial,
        )
        wvf_initial = wvf_pupil_function(base_wvf, "aperture function", aperture_initial)
        wvf_initial = wvf_compute(wvf_initial)
        initial_psf_row, initial_psf_widths = _wvf_psf_payload(wvf_initial)
        initial_point_oi = oi_crop(oi_compute(wvf_initial, point_scene), "border")
        initial_hdr_oi = oi_compute(wvf_initial, hdr_scene)

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
            seed_five,
        )
        wvf_five = wvf_pupil_function(wvf_initial, "aperture function", aperture_five)
        wvf_five = wvf_compute_psf(wvf_five)
        five_psf_row, five_psf_widths = _wvf_psf_payload(wvf_five)
        five_point_oi = oi_crop(oi_compute(wvf_five, point_scene), "border")
        five_hdr_oi = oi_crop(oi_compute(wvf_five, hdr_scene), "border")

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
            seed_defocus,
        )
        defocus_wvf = wvf_pupil_function(defocus_wvf, "aperture function", aperture_defocus)
        defocus_wvf = wvf_compute_psf(defocus_wvf)
        defocus_psf_row, defocus_psf_widths = _wvf_psf_payload(defocus_wvf)
        defocus_hdr_oi = oi_compute(defocus_wvf, hdr_scene)

        initial_aperture_sum, initial_aperture_mean, initial_dark_fraction = _aperture_stats(aperture_initial)
        five_aperture_sum, five_aperture_mean, five_dark_fraction = _aperture_stats(aperture_five)
        defocus_aperture_sum, defocus_aperture_mean, defocus_dark_fraction = _aperture_stats(aperture_defocus)
        initial_hdr_mean_550 = _oi_mean_550(initial_hdr_oi)
        five_hdr_mean_550 = _oi_mean_550(five_hdr_oi)
        defocus_hdr_mean_550 = _oi_mean_550(defocus_hdr_oi)
        hdr_mean_denominator = max(initial_hdr_mean_550, 1e-12)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pupil_diameter_mm": float(wvf_get(base_wvf, "calc pupil diameter", "mm")),
                "focal_length_mm": float(wvf_get(base_wvf, "focal length", "mm")),
                "f_number": float(wvf_get(base_wvf, "fnumber")),
                "point_scene_fov_deg": float(scene_get(point_scene, "fov")),
                "hdr_scene_fov_deg": float(scene_get(hdr_scene, "fov")),
                "seed_initial": seed_initial,
                "seed_five": seed_five,
                "seed_defocus": seed_defocus,
                "initial_nsides": int(params_initial["nsides"]),
                "initial_aperture_sum": initial_aperture_sum,
                "initial_point_oi_size": np.asarray(oi_get(initial_point_oi, "size"), dtype=int),
                "initial_point_oi_center_row_550_widths": _oi_center_row_widths(initial_point_oi),
                "initial_hdr_oi_size": np.asarray(oi_get(initial_hdr_oi, "size"), dtype=int),
                "initial_hdr_mean_photons_550_ratio": initial_hdr_mean_550 / hdr_mean_denominator,
                "five_nsides": int(params_five["nsides"]),
                "five_aperture_sum": five_aperture_sum,
                "five_point_oi_size": np.asarray(oi_get(five_point_oi, "size"), dtype=int),
                "five_point_oi_center_row_550_widths": _oi_center_row_widths(five_point_oi),
                "five_hdr_oi_size": np.asarray(oi_get(five_hdr_oi, "size"), dtype=int),
                "five_hdr_mean_photons_550_ratio": five_hdr_mean_550 / hdr_mean_denominator,
                "defocus_zcoeff": float(wvf_get(defocus_wvf, "zcoeffs", "defocus")),
                "defocus_nsides": int(params_defocus["nsides"]),
                "defocus_aperture_sum": defocus_aperture_sum,
                "defocus_hdr_oi_size": np.asarray(oi_get(defocus_hdr_oi, "size"), dtype=int),
                "defocus_hdr_mean_photons_550_ratio": defocus_hdr_mean_550 / hdr_mean_denominator,
            },
            context={
                "point_scene": point_scene,
                "hdr_scene": hdr_scene,
                "wvf_initial": wvf_initial,
                "wvf_five": wvf_five,
                "defocus_wvf": defocus_wvf,
                "initial_point_oi": initial_point_oi,
                "initial_hdr_oi": initial_hdr_oi,
                "five_point_oi": five_point_oi,
                "five_hdr_oi": five_hdr_oi,
                "defocus_hdr_oi": defocus_hdr_oi,
            },
        )

    if case_name == "oi_pad_crop_small":
        scene = scene_create("sweep frequency", asset_store=store)
        oi = oi_compute(oi_create(), scene)
        padded_size = np.asarray(oi_get(oi, "size"), dtype=float).reshape(-1)
        original_size = padded_size / 1.25
        offset = (padded_size - original_size) / 2.0
        rect = np.array([offset[1] + 1.0, offset[0] + 1.0, original_size[1] - 1.0, original_size[0] - 1.0], dtype=float)
        oi_cropped = oi_crop(oi, rect)

        sensor_scene_fov = sensor_create(asset_store=store)
        sensor_scene_fov = sensor_set(sensor_scene_fov, "noise flag", 0)
        sensor_scene_fov = sensor_set(sensor_scene_fov, "fov", float(scene_get(scene, "fov")), oi)
        sensor_from_padded = sensor_compute(sensor_scene_fov, oi, seed=0)
        sensor_from_cropped = sensor_compute(sensor_scene_fov, oi_cropped, seed=0)
        padded_volts = np.asarray(sensor_get(sensor_from_padded, "volts"), dtype=float)
        cropped_volts = np.asarray(sensor_get(sensor_from_cropped, "volts"), dtype=float)
        row_index = padded_volts.shape[0] // 2
        sensor_support = sensor_get(sensor_scene_fov, "spatial support", "um")
        normalized_mae = float(np.mean(np.abs(padded_volts - cropped_volts))) / max(float(np.mean(np.abs(cropped_volts))), 1e-12)

        sensor_padded = sensor_set_size_to_fov(sensor_scene_fov.clone(), float(oi_get(oi, "fov")), oi)
        sensor_padded = sensor_compute(sensor_padded, oi, seed=0)
        padded_support = sensor_get(sensor_padded, "spatial support", "um")
        padded_sensor_volts = np.asarray(sensor_get(sensor_padded, "volts"), dtype=float)
        padded_row_index = padded_sensor_volts.shape[0] // 2

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_padded_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "crop_rect": np.rint(rect).astype(int),
                "oi_cropped_size": np.asarray(oi_get(oi_cropped, "size"), dtype=int),
                "scene_fov_deg": float(scene_get(scene, "fov")),
                "oi_padded_fov_deg": float(oi_get(oi, "fov")),
                "oi_cropped_fov_deg": float(oi_get(oi_cropped, "fov")),
                "sensor_scene_fov_size": np.asarray(sensor_get(sensor_scene_fov, "size"), dtype=int),
                "sensor_scene_fov_pos_um": np.asarray(sensor_support["x"], dtype=float),
                "sensor_scene_fov_padded_row": np.asarray(padded_volts[row_index, :], dtype=float),
                "sensor_scene_fov_cropped_row": np.asarray(cropped_volts[row_index, :], dtype=float),
                "sensor_scene_fov_normalized_mae": normalized_mae,
                "sensor_padded_size": np.asarray(sensor_get(sensor_padded, "size"), dtype=int),
                "sensor_padded_pos_um": np.asarray(padded_support["x"], dtype=float),
                "sensor_padded_row": np.asarray(padded_sensor_volts[padded_row_index, :], dtype=float),
            },
            context={"scene": scene, "oi": oi, "oi_cropped": oi_cropped},
        )

    if case_name == "optics_microlens_small":
        oi = oi_create(asset_store=store)
        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "fov", 30.0, oi)

        microlens = mlens_create(asset_store=store)
        chief_ray_angle_default_deg = float(mlens_get(microlens, "chief ray angle"))
        microlens = mlens_set(microlens, "chief ray angle", 10.0)
        chief_ray_angle_set_deg = float(mlens_get(microlens, "chief ray angle"))

        radiance_microlens = mlens_create(asset_store=store)
        radiance_microlens = ml_radiance(radiance_microlens, asset_store=store)
        source_irradiance = np.asarray(mlens_get(radiance_microlens, "source irradiance"), dtype=float)
        pixel_irradiance = np.asarray(mlens_get(radiance_microlens, "pixel irradiance"), dtype=float)
        x_coordinate = np.asarray(mlens_get(radiance_microlens, "x coordinate"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "name": mlens_get(microlens, "name"),
                "type": mlens_get(microlens, "type"),
                "source_fnumber": float(mlens_get(microlens, "source fnumber")),
                "source_diameter_m": float(mlens_get(microlens, "source diameter", "meters")),
                "source_diameter_um": float(mlens_get(microlens, "source diameter", "microns")),
                "ml_fnumber": float(mlens_get(microlens, "ml fnumber")),
                "ml_diameter_m": float(mlens_get(microlens, "ml diameter", "meters")),
                "ml_diameter_um": float(mlens_get(microlens, "ml diameter", "microns")),
                "chief_ray_angle_default_deg": chief_ray_angle_default_deg,
                "chief_ray_angle_set_deg": chief_ray_angle_set_deg,
                "sensor_fov_deg": 30.0,
                "x_coordinate_um": x_coordinate,
                "source_center_row": np.asarray(source_irradiance[source_irradiance.shape[0] // 2, :], dtype=float),
                "pixel_center_row": np.asarray(pixel_irradiance[pixel_irradiance.shape[0] // 2, :], dtype=float),
                "source_irradiance_stats": _stats_vector(source_irradiance),
                "pixel_irradiance_stats": _stats_vector(pixel_irradiance),
                "etendue": float(mlens_get(radiance_microlens, "etendue")),
            },
            context={"oi": oi, "sensor": sensor, "microlens": radiance_microlens},
        )

    if case_name == "oi_wvf_small_scene":
        scene = scene_create("checkerboard", 8, 4, asset_store=store)
        oi_seed = oi_create("wvf")
        optics = dict(oi_seed.fields["optics"])
        scene_photons = np.asarray(scene.data["photons"], dtype=float)
        wave = np.asarray(scene.fields["wave"], dtype=float)
        _, width_m, _ = _oi_geometry(optics, scene)
        sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
        pre_psf_photons = _radiance_to_irradiance(scene_photons, optics, scene)
        if str(optics.get("offaxis_method", "")).replace(" ", "").lower() == "cos4th":
            pre_psf_photons = pre_psf_photons * _cos4th_factor(
                pre_psf_photons.shape[0],
                pre_psf_photons.shape[1],
                optics,
                scene,
            )[:, :, None]
        pad_pixels = (
            int(np.round(scene_photons.shape[0] / 8.0)),
            int(np.round(scene_photons.shape[1] / 8.0)),
        )
        pre_psf_photons, _, _ = _pad_scene(pre_psf_photons, pad_pixels, "zero")
        psf_stack = _wvf_psf_stack(pre_psf_photons.shape[:2], sample_spacing_m, wave, optics)
        oi = oi_compute(oi_seed, scene, crop=True)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": oi.fields["wave"],
                "pre_psf_photons": pre_psf_photons,
                "psf_stack": psf_stack,
                "photons": oi.data["photons"],
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_bayer_noiseless":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_set(sensor_create(asset_store=store), "noise flag", 0)
        sensor = sensor_compute(sensor, oi, seed=0)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "volts": sensor.data["volts"],
                "integration_time": sensor.fields["integration_time"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_monochrome_noise_stats":
        scene = scene_create("uniform d65", 32, asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 2)
        sensor = sensor_compute(sensor, oi, seed=0)
        return ParityCaseResult(
            payload={"case_name": case_name, **_stats(sensor.data["volts"])},
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_imx363_crop_small":
        sensor = sensor_create("IMX363", None, "row col", [12, 16], asset_store=store)
        sensor = sensor_set(sensor, "pattern", np.array([[2, 1], [3, 2]], dtype=int))
        sensor = sensor_set(sensor, "wave", np.arange(400.0, 701.0, 10.0, dtype=float))
        dv = np.arange(12 * 16, dtype=float).reshape((12, 16), order="F")
        sensor = sensor_set(sensor, "digital values", dv)
        cropped = sensor_crop(sensor, np.array([2, 3, 7, 5], dtype=float))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "name": str(cropped.name),
                "size": np.asarray(sensor_get(cropped, "size"), dtype=int),
                "metadata_crop": np.asarray(sensor_get(cropped, "metadata crop"), dtype=int),
                "pattern": np.asarray(sensor_get(cropped, "pattern"), dtype=int),
                "digital_values": np.asarray(sensor_get(cropped, "digital values"), dtype=float),
            },
            context={"sensor": cropped},
        )

    if case_name == "sensor_plot_line_volts_space_small":
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "rows", 2)
        sensor = sensor_set(sensor, "cols", 4)
        sensor = sensor_set(
            sensor,
            "volts",
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                dtype=float,
            ),
        )
        _, udata = sensor_plot_line(sensor, "h", "volts", "space", np.array([1, 2], dtype=int))
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pixPos": np.asarray(udata["pixPos"], dtype=float),
                "pixData": np.asarray(udata["pixData"], dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_signal_current_uniform_small":
        scene = scene_create("uniform ee", 64, asset_store=store)
        scene = scene_set(scene, "fov", 8.0)
        scene = scene_set(scene, "distance", 1.2)
        scene = scene_set(scene, "name", "uniform ee")
        scene = scene_adjust_luminance(scene, 1.0, asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)
        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_set(sensor, "exp time", 1.0)
        current = np.asarray(signal_current(oi, sensor), dtype=float)
        start = (current.shape[0] - 40) // 2
        stop = start + 40
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "current_center": current[start:stop, start:stop],
                "mean_current": float(np.mean(current)),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_split_pixel_ovt_saturated_small":
        scene = scene_create("uniform ee", np.array([32, 48], dtype=int), asset_store=store)
        scene = scene_set(scene, "fov", 8.0)
        photons = np.asarray(scene_get(scene, "photons"), dtype=float).copy()
        levels = np.array([1.0, 10.0, 100.0, 1000.0], dtype=float)
        band_width = photons.shape[1] // levels.size
        for index, level in enumerate(levels):
            start = index * band_width
            stop = photons.shape[1] if index == (levels.size - 1) else (index + 1) * band_width
            photons[:, start:stop, :] *= level
        scene.data["photons"] = photons
        oi = oi_create("wvf", asset_store=store)
        oi = oi_compute(oi, scene, "crop", True)

        sensor_array = sensor_create_array(
            "array type",
            "ovt",
            "exp time",
            0.1,
            "size",
            np.array([32, 48], dtype=int),
            "noise flag",
            0,
            asset_store=store,
        )
        combined, captures = sensor_compute_array(sensor_array, oi, "method", "saturated")
        saturated = np.asarray(combined.metadata["saturated"], dtype=bool)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "combined_volts": np.asarray(sensor_get(combined, "volts"), dtype=float),
                "sensor_max_volts": np.array([float(np.max(np.asarray(sensor_get(sensor, "volts"), dtype=float))) for sensor in captures], dtype=float),
                "saturated_counts": np.sum(saturated, axis=(0, 1), dtype=int),
                "sensor_names": np.asarray([str(sensor_get(sensor, "name")) for sensor in captures], dtype=object),
            },
            context={"scene": scene, "oi": oi, "sensor": combined},
        )

    if case_name == "sensor_stacked_pixels_foveon_small":
        scene = scene_create("macbeth d65", 32, asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "hfov", 8.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_set(oi, "optics focal length", 3e-3)
        oi = oi_compute(oi, scene)

        wave = np.asarray(scene_get(scene, "wave"), dtype=float)
        foveon_filters = np.asarray(ie_read_spectra("Foveon", wave, asset_store=store), dtype=float)

        monochrome_array: list[Any] = []
        for index in range(foveon_filters.shape[1]):
            sensor = sensor_create("monochrome", asset_store=store)
            sensor = sensor_set(sensor, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
            sensor = sensor_set(sensor, "exp time", 0.1)
            sensor = sensor_set(sensor, "filter spectra", foveon_filters[:, index])
            sensor = sensor_set(sensor, "name", f"Channel-{index + 1}")
            sensor = sensor_set_size_to_fov(sensor, scene_get(scene, "fov"), oi)
            sensor = sensor_set(sensor, "wave", wave)
            monochrome_array.append(sensor)
        computed_planes = sensor_compute(monochrome_array, oi)
        stacked_volts = np.stack([np.asarray(sensor_get(sensor, "volts"), dtype=float) for sensor in computed_planes], axis=2)

        sensor_foveon = sensor_create(asset_store=store)
        sensor_foveon = sensor_set(sensor_foveon, "name", "foveon")
        sensor_foveon = sensor_set(sensor_foveon, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor_foveon = sensor_set(sensor_foveon, "autoexp", 1)
        sensor_foveon = sensor_set_size_to_fov(sensor_foveon, scene_get(scene, "fov"), oi)
        sensor_foveon = sensor_set(sensor_foveon, "wave", wave)
        sensor_foveon = sensor_set(sensor_foveon, "filter spectra", foveon_filters)
        sensor_foveon = sensor_set(sensor_foveon, "pattern", np.array([[2]], dtype=int))
        sensor_foveon = sensor_set(sensor_foveon, "volts", stacked_volts)

        ip_foveon = ip_compute(ip_create(asset_store=store), sensor_foveon, asset_store=store)
        line_row = min(120, int(np.asarray(sensor_get(sensor_foveon, "size"), dtype=int)[0]))
        foveon_line, _ = ip_plot(ip_foveon, "horizontal line", np.array([1, line_row], dtype=int))

        bayer_filters = np.asarray(ie_read_spectra("NikonD1", wave, asset_store=store), dtype=float)
        sensor_bayer = sensor_create(asset_store=store)
        sensor_bayer = sensor_set(sensor_bayer, "filter spectra", bayer_filters)
        sensor_bayer = sensor_set(sensor_bayer, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor_bayer = sensor_set(sensor_bayer, "autoexp", 1)
        sensor_bayer = sensor_set_size_to_fov(sensor_bayer, scene_get(scene, "fov"), oi)
        sensor_bayer = sensor_compute(sensor_bayer, oi)
        ip_bayer = ip_compute(ip_create(asset_store=store), sensor_bayer, asset_store=store)
        bayer_line, _ = ip_plot(ip_bayer, "horizontal line", np.array([1, line_row], dtype=int))
        bayer_line_values = np.asarray(bayer_line["values"], dtype=float)
        row_start = (stacked_volts.shape[0] - 24) // 2
        col_start = (stacked_volts.shape[1] - 24) // 2
        row_stop = row_start + 24
        col_stop = col_start + 24
        stacked_patch = stacked_volts[row_start:row_stop, col_start:col_stop, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "stacked_center_patch_mean": np.mean(stacked_patch, axis=(0, 1), dtype=float),
                "stacked_center_patch_std": np.std(stacked_patch, axis=(0, 1), dtype=float),
                "stacked_center_patch_p90": np.percentile(stacked_patch, 90.0, axis=(0, 1)),
                "stacked_mean_volts": np.mean(stacked_volts, axis=(0, 1), dtype=float),
                "stacked_std_volts": np.std(stacked_volts, axis=(0, 1), dtype=float),
                "line_row": float(line_row),
                "bayer_line_mean": np.mean(bayer_line_values, axis=0, dtype=float),
                "bayer_line_std": np.std(bayer_line_values, axis=0, dtype=float),
                "bayer_line_p90": np.percentile(bayer_line_values, 90.0, axis=0),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_foveon, "sensor_bayer": sensor_bayer},
        )

    if case_name == "sensor_microlens_etendue_small":
        oi = oi_create(asset_store=store)
        sensor = sensor_create(asset_store=store)
        microlens = mlens_create(sensor, oi, asset_store=store)
        sensor = sensor_set(sensor, "microlens", microlens)
        sensor = sensor_set_size_to_fov(sensor, 4.0, oi)

        no_microlens = ml_analyze_array_etendue(sensor.clone(), "no microlens", asset_store=store)
        centered = ml_analyze_array_etendue(sensor.clone(), "centered", asset_store=store)
        optimal = ml_analyze_array_etendue(sensor.clone(), "optimal", asset_store=store)

        cra_deg = np.asarray(sensor_get(optimal, "cra degrees"), dtype=float)
        ray_angles_deg = np.linspace(0.0, float(np.max(cra_deg)), 10, dtype=float)
        optimal_microlens = sensor_get(optimal, "microlens")
        optimal_offset_curve_um = np.array(
            [
                float(
                    mlens_get(
                        mlens_set(optimal_microlens, "chief ray angle", float(ray_angle_deg)),
                        "optimal offset",
                        optimal,
                        "microns",
                    )
                )
                for ray_angle_deg in ray_angles_deg
            ],
            dtype=float,
        )

        ml_fnumber = float(mlens_get(optimal_microlens, "ml fnumber"))
        half_fnumber_microlens = mlens_set(optimal_microlens, "ml fnumber", 0.5 * ml_fnumber)
        source_f4_microlens = mlens_set(optimal_microlens, "source fnumber", 4.0)
        source_f16_microlens = mlens_set(optimal_microlens, "source fnumber", 16.0)

        display_microlens = mlens_set(mlens_create(asset_store=store), "ml fnumber", 8.0)
        radiance_midlines: list[np.ndarray] = []
        for chief_ray_angle_deg in (-10.0, 0.0, 10.0):
            display_microlens = mlens_set(display_microlens, "chief ray angle", chief_ray_angle_deg)
            display_microlens = ml_radiance(display_microlens, sensor_create(asset_store=store), asset_store=store)
            irradiance = np.asarray(mlens_get(display_microlens, "pixel irradiance"), dtype=float)
            radiance_midlines.append(np.asarray(irradiance[irradiance.shape[0] // 2, :], dtype=float))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "no_microlens_etendue": np.asarray(sensor_get(no_microlens, "etendue"), dtype=float),
                "centered_etendue": np.asarray(sensor_get(centered, "etendue"), dtype=float),
                "optimal_etendue": np.asarray(sensor_get(optimal, "etendue"), dtype=float),
                "ray_angles_deg": ray_angles_deg,
                "optimal_offset_curve_um": optimal_offset_curve_um,
                "optimal_offsets_default_um": np.asarray(mlens_get(optimal_microlens, "optimal offsets", optimal), dtype=float),
                "optimal_offsets_half_fnumber_um": np.asarray(mlens_get(half_fnumber_microlens, "optimal offsets", optimal), dtype=float),
                "optimal_offsets_source_f4_um": np.asarray(mlens_get(source_f4_microlens, "optimal offsets", optimal), dtype=float),
                "optimal_offsets_source_f16_um": np.asarray(mlens_get(source_f16_microlens, "optimal offsets", optimal), dtype=float),
                "radiance_midline_neg10": radiance_midlines[0],
                "radiance_midline_0": radiance_midlines[1],
                "radiance_midline_10": radiance_midlines[2],
            },
            context={"oi": oi, "sensor": optimal},
        )

    if case_name == "sensor_comparison_small":
        patch_size = 24
        scene_c = scene_create("macbeth d65", patch_size, asset_store=store)
        macbeth_size = np.asarray(scene_get(scene_c, "size"), dtype=int)
        scene_c = scene_set(
            scene_c,
            "resize",
            np.rint(np.array([macbeth_size[0], macbeth_size[1] / 2.0], dtype=float)).astype(int),
        )
        scene_s = scene_create("sweep frequency", int(macbeth_size[0]), float(macbeth_size[0]) / 16.0, asset_store=store)
        scene = scene_combine(scene_c, scene_s, "direction", "horizontal")
        scene = scene_set(scene, "fov", 20.0)
        scene_vfov = float(scene_get(scene, "vfov"))

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 1.2)
        oi = oi_compute(oi, scene)

        sensor_names = ("imx363", "mt9v024", "cyym")
        ip_names = ("imx363", "mt9v024")
        small_sensor_sizes = np.zeros((len(sensor_names), 2), dtype=int)
        small_sensor_mean_volts = np.zeros(len(sensor_names), dtype=float)
        small_sensor_p90_volts = np.zeros(len(sensor_names), dtype=float)
        large_sensor_sizes = np.zeros((len(sensor_names), 2), dtype=int)
        large_sensor_mean_volts = np.zeros(len(sensor_names), dtype=float)
        large_sensor_p90_volts = np.zeros(len(sensor_names), dtype=float)
        small_ip_sizes = np.zeros((len(ip_names), 3), dtype=int)
        large_ip_sizes = np.zeros((len(ip_names), 3), dtype=int)

        def _compute_sensor(sensor_type: str, pixel_size_m: float) -> tuple[Any, np.ndarray]:
            if sensor_type == "mt9v024":
                sensor = sensor_create(sensor_type, None, "rccc", asset_store=store)
            else:
                sensor = sensor_create(sensor_type, asset_store=store)
            sensor = sensor_set(sensor, "pixel size", pixel_size_m)
            sensor = sensor_set(sensor, "hfov", 20.0, oi)
            sensor = sensor_set(sensor, "vfov", scene_vfov)
            sensor = sensor_set(sensor, "auto exposure", True)
            sensor = sensor_compute(sensor, oi)
            return sensor, np.asarray(sensor_get(sensor, "volts"), dtype=float)

        def _compute_ip(sensor_type: str, sensor: Any) -> np.ndarray:
            if sensor_type == "imx363":
                ip = ip_create("imx363 RGB", sensor, asset_store=store)
            elif sensor_type == "mt9v024":
                ip = ip_create("mt9v024 RCCC", sensor, asset_store=store)
                ip = ip_set(ip, "demosaic method", "analog rccc")
            else:
                raise ValueError(f"Unsupported IP comparison sensor: {sensor_type}")
            ip = ip_compute(ip, sensor, asset_store=store)
            return np.asarray(ip_get(ip, "result"), dtype=float)

        for sensor_index, sensor_type in enumerate(sensor_names):
            sensor_small, volts_small = _compute_sensor(sensor_type, 1.5e-6)
            small_sensor_sizes[sensor_index, :] = np.asarray(sensor_get(sensor_small, "size"), dtype=int)
            small_sensor_mean_volts[sensor_index] = float(np.mean(volts_small))
            small_sensor_p90_volts[sensor_index] = float(np.percentile(volts_small, 90.0))

            sensor_large, volts_large = _compute_sensor(sensor_type, 6.0e-6)
            large_sensor_sizes[sensor_index, :] = np.asarray(sensor_get(sensor_large, "size"), dtype=int)
            large_sensor_mean_volts[sensor_index] = float(np.mean(volts_large))
            large_sensor_p90_volts[sensor_index] = float(np.percentile(volts_large, 90.0))

            if sensor_type in ip_names:
                ip_index = ip_names.index(sensor_type)
                result_small = _compute_ip(sensor_type, sensor_small)
                result_large = _compute_ip(sensor_type, sensor_large)
                small_ip_sizes[ip_index, :] = np.asarray(result_small.shape, dtype=int)
                large_ip_sizes[ip_index, :] = np.asarray(result_large.shape, dtype=int)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "scene_vfov": scene_vfov,
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "small_sensor_sizes": small_sensor_sizes,
                "nonimx_small_sensor_mean_volts": small_sensor_mean_volts[1:],
                "nonimx_small_sensor_p90_volts": small_sensor_p90_volts[1:],
                "large_sensor_sizes": large_sensor_sizes,
                "nonimx_large_sensor_mean_volts": large_sensor_mean_volts[1:],
                "nonimx_large_sensor_p90_volts": large_sensor_p90_volts[1:],
                "imx363_mean_ratio_large_small": float(large_sensor_mean_volts[0] / max(small_sensor_mean_volts[0], 1e-12)),
                "imx363_p90_ratio_large_small": float(large_sensor_p90_volts[0] / max(small_sensor_p90_volts[0], 1e-12)),
                "small_ip_sizes": small_ip_sizes,
                "large_ip_sizes": large_ip_sizes,
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_noise_samples_small":
        scene = scene_create("slanted bar", 128, asset_store=store)
        scene = scene_set(scene, "fov", 4.0)
        oi = oi_compute(oi_create(asset_store=store), scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "exp time", 0.05)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor_nf = sensor_compute(sensor, oi, seed=0)
        volts_nf = np.asarray(sensor_get(sensor_nf, "volts"), dtype=float)

        n_samp = 64
        volt_images = np.asarray(sensor_compute_samples(sensor_nf, n_samp, 2, seed=7), dtype=float)
        noise_images = volt_images - volts_nf[:, :, np.newaxis]
        std_image = np.std(volt_images, axis=2, ddof=1)
        mean_image = np.mean(volt_images, axis=2, dtype=float)
        pair_diff = volt_images[:, :, 0] - volt_images[:, :, 1]
        mean_residual = mean_image - volts_nf

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sample_shape": np.asarray(volt_images.shape, dtype=int),
                "noise_free_mean": float(np.mean(volts_nf)),
                "noise_std_image_stats": np.array(
                    [
                        float(np.mean(std_image)),
                        float(np.percentile(std_image, 10.0)),
                        float(np.percentile(std_image, 50.0)),
                        float(np.percentile(std_image, 90.0)),
                    ],
                    dtype=float,
                ),
                "noise_distribution_stats": np.array(
                    [
                        float(np.std(noise_images, ddof=1)),
                        float(np.percentile(noise_images, 5.0)),
                        float(np.percentile(noise_images, 50.0)),
                        float(np.percentile(noise_images, 95.0)),
                    ],
                    dtype=float,
                ),
                "mean_residual_stats": np.array(
                    [
                        float(np.mean(np.abs(mean_residual))),
                        float(np.percentile(np.abs(mean_residual), 95.0)),
                    ],
                    dtype=float,
                ),
                "pair_diff_stats": np.array(
                    [
                        float(np.std(pair_diff, ddof=1)),
                        float(np.percentile(pair_diff, 5.0)),
                        float(np.percentile(pair_diff, 50.0)),
                        float(np.percentile(pair_diff, 95.0)),
                    ],
                    dtype=float,
                ),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_nf},
        )

    if case_name == "sensor_mcc_small":
        mosaic_path = store.resolve("data/sensor/mccGBRGsensor.tif")
        mosaic = np.asarray(iio.imread(mosaic_path), dtype=float)

        sensor = sensor_create("bayer (gbrg)", asset_store=store)
        sensor = sensor_set(sensor, "name", "Sensor demo")

        minimum = float(np.min(mosaic))
        maximum = float(np.max(mosaic))
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
        volts = ((mosaic - minimum) / max(maximum - minimum, 1e-12)) * voltage_swing

        sensor = sensor_set(sensor, "size", np.asarray(volts.shape[:2], dtype=int))
        sensor = sensor_set(sensor, "volts", volts)
        corner_points = np.array([[15, 584], [782, 584], [784, 26], [23, 19]], dtype=float)
        sensor = sensor_set(sensor, "chart corner points", corner_points)
        estimated_ccm, _ = sensor_ccm(sensor, None, None, True, asset_store=store)

        ip_uncorrected = ip_create(asset_store=store)
        ip_uncorrected = ip_set(ip_uncorrected, "name", "No Correction")
        ip_uncorrected = ip_set(ip_uncorrected, "scaledisplay", 1)
        ip_uncorrected = ip_compute(ip_uncorrected, sensor, asset_store=store)
        uncorrected = np.asarray(ip_get(ip_uncorrected, "result"), dtype=float)
        uncorrected_flat = uncorrected.reshape(-1, uncorrected.shape[2])

        fixed_matrix = np.array(
            [
                [0.9205, -0.1402, -0.1289],
                [-0.0148, 0.8763, -0.0132],
                [-0.2516, -0.1567, 0.6987],
            ],
            dtype=float,
        )
        ip_corrected = ip_create(asset_store=store)
        ip_corrected = ip_set(ip_corrected, "name", "CCM Correction")
        ip_corrected = ip_set(ip_corrected, "scaledisplay", 1)
        ip_corrected = ip_set(ip_corrected, "conversion transform sensor", fixed_matrix)
        ip_corrected = ip_set(ip_corrected, "correction transform illuminant", np.eye(3, dtype=float))
        ip_corrected = ip_set(ip_corrected, "ics2Display Transform", np.eye(3, dtype=float))
        ip_corrected = ip_set(ip_corrected, "conversion method sensor", "current matrix")
        ip_corrected = ip_compute(ip_corrected, sensor, asset_store=store)
        corrected = np.asarray(ip_get(ip_corrected, "result"), dtype=float)
        corrected_flat = corrected.reshape(-1, corrected.shape[2])

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "mosaic_size": np.asarray(volts.shape[:2], dtype=int),
                "volts_stats": _stats_vector(volts),
                "estimated_ccm": np.asarray(estimated_ccm, dtype=float),
                "uncorrected_mean_rgb_norm": _channel_normalize(np.mean(uncorrected_flat, axis=0, dtype=float)),
                "uncorrected_p95_rgb_norm": _channel_normalize(np.percentile(uncorrected_flat, 95.0, axis=0)),
                "corrected_mean_rgb_norm": _channel_normalize(np.mean(corrected_flat, axis=0, dtype=float)),
                "corrected_p95_rgb_norm": _channel_normalize(np.percentile(corrected_flat, 95.0, axis=0)),
            },
            context={"sensor": sensor, "ip_uncorrected": ip_uncorrected, "ip_corrected": ip_corrected},
        )

    if case_name == "sensor_rolling_shutter_small":
        scene = scene_create("star pattern", 48, "ee", 4, asset_store=store)
        scene = scene_set(scene, "fov", 3.0)
        oi = oi_create(asset_store=store)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "pixel size constant fill factor", np.array([1.4e-6, 1.4e-6], dtype=float))
        sensor = sensor_set(sensor, "fov", float(scene_get(scene, "fov")) / 2.0, oi)
        sensor = sensor_set(sensor, "exp time", 4.0e-5)
        sensor = sensor_set(sensor, "noise flag", 0)

        sensor_size = np.asarray(sensor_get(sensor, "size"), dtype=int)
        exp_time = float(sensor_get(sensor, "exp time"))
        per_row = 10.0e-6
        rate = 0.3
        n_frames = int(sensor_size[0] + round(exp_time / per_row))
        crop_width = int(sensor_size[1] - 1)
        crop_height = int(sensor_size[0] - 1)

        volt_stack = np.zeros((sensor_size[0], sensor_size[1], n_frames), dtype=float)
        crop_rects = np.zeros((n_frames, 4), dtype=int)
        temporal_mean_volts = np.zeros(n_frames, dtype=float)
        current_sensor = sensor

        for frame_index in range(n_frames):
            rotated_scene = scene_rotate(scene, (frame_index + 1) * rate)
            oi_frame = oi_compute(oi, rotated_scene)
            center_pixel = np.asarray(oi_get(oi_frame, "center pixel"), dtype=float)
            rect = np.rint(
                [
                    center_pixel[1] - crop_width / 2.0,
                    center_pixel[0] - crop_height / 2.0,
                    crop_width,
                    crop_height,
                ],
            ).astype(int)
            crop_rects[frame_index, :] = rect
            oi_cropped = oi_crop(oi_frame, rect)
            current_sensor = sensor_compute(current_sensor, oi_cropped, seed=0)
            volts = np.asarray(sensor_get(current_sensor, "volts"), dtype=float)
            volt_stack[:, :, frame_index] = volts
            temporal_mean_volts[frame_index] = float(np.mean(volts))

        integration_rows = max(int(round(exp_time / per_row)), 1)
        slist = np.arange(integration_rows, dtype=int)
        final = np.zeros((sensor_size[0], sensor_size[1]), dtype=float)
        for row_index in range(sensor_size[0]):
            slist = slist + 1
            row_stack = volt_stack[row_index, :, :]
            final[row_index, :] = np.sum(row_stack[:, slist], axis=1)

        final_sensor = sensor_set(current_sensor.clone(), "volts", final)
        ip = ip_compute(ip_create(asset_store=store), final_sensor, asset_store=store)
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        result_flat = result.reshape(-1, result.shape[2])

        sampled_rows = np.array([0, sensor_size[0] // 2, sensor_size[0] - 1], dtype=int)
        sampled_cols = np.array([0, sensor_size[1] // 2, sensor_size[1] - 1], dtype=int)
        sampled_row_stats = np.vstack([_stats_vector(final[row_index, :]) for row_index in sampled_rows])

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_size": sensor_size,
                "n_frames": int(n_frames),
                "crop_size": np.array([crop_height + 1, crop_width + 1], dtype=int),
                "first_crop_rect": crop_rects[0, :],
                "last_crop_rect": crop_rects[-1, :],
                "temporal_mean_volts": temporal_mean_volts,
                "center_pixel_trace": volt_stack[sensor_size[0] // 2, sensor_size[1] // 2, :],
                "final_stats": _stats_vector(final),
                "sampled_rows": sampled_rows + 1,
                "sampled_cols": sampled_cols + 1,
                "sampled_row_stats": sampled_row_stats,
                "result_mean_rgb_norm": _channel_normalize(np.mean(result_flat, axis=0, dtype=float)),
                "result_p95_rgb_norm": _channel_normalize(np.percentile(result_flat, 95.0, axis=0)),
            },
            context={"scene": scene, "sensor": final_sensor, "ip": ip},
        )

    if case_name == "sensor_imx490_uniform_small":
        scene = scene_create("uniform", 256, asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)
        oi = oi_crop(oi, "border")
        oi = oi_spatial_resample(oi, 3.0e-6)

        combined, metadata = imx490_compute(
            oi,
            "method",
            "best snr",
            "exp time",
            0.1,
            "noise flag",
            0,
            asset_store=store,
        )
        captures = list(metadata["sensorArray"])

        capture_mean_electrons = np.array(
            [float(np.mean(np.asarray(sensor_get(sensor, "electrons"), dtype=float))) for sensor in captures],
            dtype=float,
        )
        capture_mean_volts = np.array(
            [float(np.mean(np.asarray(sensor_get(sensor, "volts"), dtype=float))) for sensor in captures],
            dtype=float,
        )
        capture_mean_dv = np.array(
            [float(np.mean(np.asarray(sensor_get(sensor, "dv"), dtype=float))) for sensor in captures],
            dtype=float,
        )
        best_pixel = np.asarray(combined.metadata["bestPixel"], dtype=int)
        best_pixel_counts = np.array([int(np.sum(best_pixel == (index + 1))) for index in range(len(captures))], dtype=int)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "capture_names": np.asarray([str(sensor_get(sensor, "name")) for sensor in captures], dtype=object),
                "capture_mean_electrons": capture_mean_electrons,
                "capture_mean_volts": capture_mean_volts,
                "capture_mean_dv": capture_mean_dv,
                "large_gain_ratio": float(capture_mean_volts[1] / max(capture_mean_volts[0], 1.0e-12)),
                "small_area_ratio": float(capture_mean_electrons[2] / max(capture_mean_electrons[0], 1.0e-12)),
                "combined_volts_stats": _stats_vector(np.asarray(sensor_get(combined, "volts"), dtype=float)),
                "best_pixel_counts": best_pixel_counts,
            },
            context={"scene": scene, "oi": oi, "sensor": combined},
        )

    if case_name == "sensor_hdr_pixel_size_small":
        scene = scene_from_file(
            store.resolve("data/images/multispectral/Feng_Office-hdrs.mat"),
            "multispectral",
            200.0,
            asset_store=store,
        )
        oi = oi_compute(oi_create(asset_store=store), scene)

        pixel_sizes_um = np.array([1.0, 2.0, 4.0], dtype=float)
        dye_size_um = 512.0
        base_sensor = sensor_create("monochrome", asset_store=store)
        base_sensor = sensor_set(base_sensor, "exp time", 0.003)

        sensor_sizes = np.zeros((pixel_sizes_um.size, 2), dtype=int)
        mean_volts = np.zeros(pixel_sizes_um.size, dtype=float)
        p95_volts = np.zeros(pixel_sizes_um.size, dtype=float)
        mean_electrons = np.zeros(pixel_sizes_um.size, dtype=float)
        result_sizes = np.zeros((pixel_sizes_um.size, 3), dtype=int)
        result_mean_gray = np.zeros(pixel_sizes_um.size, dtype=float)
        result_p95_gray = np.zeros(pixel_sizes_um.size, dtype=float)

        for index, pixel_size_um in enumerate(pixel_sizes_um):
            sensor = sensor_set(base_sensor.clone(), "pixel size constant fill factor", np.array([pixel_size_um, pixel_size_um], dtype=float) * 1.0e-6)
            sensor = sensor_set(sensor, "rows", int(np.rint(dye_size_um / pixel_size_um)))
            sensor = sensor_set(sensor, "cols", int(np.rint(dye_size_um / pixel_size_um)))
            sensor = sensor_compute(sensor, oi)
            ip = ip_compute(ip_create(asset_store=store), sensor, asset_store=store)
            result = np.asarray(ip_get(ip, "result"), dtype=float)
            gray = result[:, :, 0] if result.ndim == 3 else result

            sensor_sizes[index, :] = np.asarray(sensor_get(sensor, "size"), dtype=int)
            mean_volts[index] = float(np.mean(np.asarray(sensor_get(sensor, "volts"), dtype=float)))
            p95_volts[index] = float(np.percentile(np.asarray(sensor_get(sensor, "volts"), dtype=float), 95.0))
            mean_electrons[index] = float(np.mean(np.asarray(sensor_get(sensor, "electrons"), dtype=float)))
            result_sizes[index, :] = np.asarray(result.shape, dtype=int)
            result_mean_gray[index] = float(np.mean(gray))
            result_p95_gray[index] = float(np.percentile(gray, 95.0))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
                "pixel_sizes_um": pixel_sizes_um,
                "sensor_sizes": sensor_sizes,
                "mean_volts": mean_volts,
                "p95_volts": p95_volts,
                "mean_electrons": mean_electrons,
                "result_sizes": result_sizes,
                "result_mean_gray": result_mean_gray,
                "result_p95_gray": result_p95_gray,
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_log_ar0132at_small":
        dynamic_range = float(2**16)
        scene = scene_create("exponential intensity ramp", 256, dynamic_range, asset_store=store)
        scene = scene_set(scene, "fov", 60.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 2.8)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "response type", "log")
        sensor = sensor_set(sensor, "size", np.array([960, 1280], dtype=int))
        sensor = sensor_set(sensor, "pixel size same fill factor", 3.751e-6)

        color_filter_file = store.resolve("data/sensor/colorfilters/auto/ar0132at.mat")
        wave = np.asarray(scene_get(scene, "wave"), dtype=float)
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, color_filter_file)
        sensor = sensor_set(sensor, "filter spectra", filter_spectra)
        sensor = sensor_set(sensor, "filter names", filter_names)
        sensor = sensor_set(sensor, "pixel read noise volts", 1.0e-3)
        sensor = sensor_set(sensor, "pixel voltage swing", 2.8)
        sensor = sensor_set(sensor, "pixel dark voltage", 1.0e-3)
        sensor = sensor_set(sensor, "pixel conversion gain", 110.0e-6)
        sensor = sensor_set(sensor, "exp time", 0.003)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_compute(sensor, oi)

        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        sampled_cols = np.rint(np.linspace(0.0, volts.shape[1] - 1.0, 33)).astype(int)
        row_15 = np.asarray(volts[14, :], dtype=float)
        row_114 = np.asarray(volts[113, :], dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "sensor_size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "wave": wave,
                "dr_at_1s": float(sensor_dr(sensor, 1.0)),
                "volts_stats": _stats_vector(volts),
                "sampled_cols": sampled_cols + 1,
                "row15_stats": _stats_vector(row_15),
                "row114_stats": _stats_vector(row_114),
                "row15_profile_norm": _channel_normalize(row_15[sampled_cols]),
                "row114_profile_norm": _channel_normalize(row_114[sampled_cols]),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_aliasing_small":
        def _canonical_profile(values: np.ndarray, samples: int = 129) -> np.ndarray:
            profile = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, profile.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, profile)

        fov = 5.0
        sweep_scene = scene_create("sweep frequency", 768, 30.0, asset_store=store)
        sweep_scene = scene_set(sweep_scene, "fov", fov)

        oi = oi_create("diffraction limited", asset_store=store)
        oi = oi_set(oi, "optics fnumber", 2.0)
        oi = oi_compute(oi, sweep_scene)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set_size_to_fov(sensor, fov, oi)
        sensor = sensor_set(sensor, "noise flag", 0)

        sensor_small = sensor_set(sensor.clone(), "pixel size constant fill factor", 2.0e-6)
        sensor_small = sensor_compute(sensor_small, oi)
        small_line = sensor_get(sensor_small, "hline electrons", 1)
        small_data = np.asarray(small_line["data"][0], dtype=float)
        small_pos = np.asarray(small_line["pos"][0], dtype=float)

        sensor_large = sensor_set(sensor.clone(), "pixel size constant fill factor", 6.0e-6)
        sensor_large = sensor_set_size_to_fov(sensor_large, fov, oi)
        sensor_large = sensor_compute(sensor_large, oi)
        large_line = sensor_get(sensor_large, "hline electrons", 1)
        large_data = np.asarray(large_line["data"][0], dtype=float)
        large_pos = np.asarray(large_line["pos"][0], dtype=float)

        oi_blur = oi_set(oi.clone(), "optics fnumber", 12.0)
        oi_blur = oi_compute(oi_blur, sweep_scene)
        sensor_blur = sensor_compute(sensor_large.clone(), oi_blur)
        blur_line = sensor_get(sensor_blur, "hline electrons", 1)
        blur_data = np.asarray(blur_line["data"][0], dtype=float)
        blur_pos = np.asarray(blur_line["pos"][0], dtype=float)

        slanted_scene = scene_create("slanted bar", 1024, asset_store=store)
        slanted_scene = scene_set(slanted_scene, "fov", fov)

        oi_slanted_sharp = oi_set(oi_blur.clone(), "optics fnumber", 2.0)
        oi_slanted_sharp = oi_compute(oi_slanted_sharp, slanted_scene)
        sensor_slanted = sensor_set(sensor_large.clone(), "pixel size constant fill factor", 6.0e-6)
        sensor_slanted = sensor_set_size_to_fov(sensor_slanted, fov, oi_slanted_sharp)
        sensor_slanted = sensor_compute(sensor_slanted, oi_slanted_sharp)
        slanted_sharp = np.asarray(sensor_get(sensor_slanted, "electrons"), dtype=float)

        oi_slanted_blur = oi_set(oi_slanted_sharp.clone(), "optics fnumber", 12.0)
        oi_slanted_blur = oi_compute(oi_slanted_blur, slanted_scene)
        sensor_slanted_blur = sensor_compute(sensor_slanted.clone(), oi_slanted_blur)
        slanted_blur = np.asarray(sensor_get(sensor_slanted_blur, "electrons"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "fov_deg": fov,
                "sweep_scene_size": np.asarray(scene_get(sweep_scene, "size"), dtype=int),
                "sweep_oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "small_sensor_size": np.asarray(sensor_get(sensor_small, "size"), dtype=int),
                "large_sensor_size": np.asarray(sensor_get(sensor_large, "size"), dtype=int),
                "small_line_pos": small_pos,
                "small_line_data_norm": _channel_normalize(small_data),
                "large_line_pos": large_pos,
                "large_line_data_norm": _channel_normalize(large_data),
                "blur_line_pos": blur_pos,
                "blur_line_data_norm": _channel_normalize(blur_data),
                "small_line_stats": _stats_vector(small_data),
                "large_line_stats": _stats_vector(large_data),
                "blur_line_stats": _stats_vector(blur_data),
                "slanted_scene_size": np.asarray(scene_get(slanted_scene, "size"), dtype=int),
                "slanted_sensor_size": np.asarray(sensor_get(sensor_slanted, "size"), dtype=int),
                "slanted_sharp_center_row_norm": _canonical_profile(_channel_normalize(slanted_sharp[slanted_sharp.shape[0] // 2, :])),
                "slanted_sharp_center_col_norm": _canonical_profile(_channel_normalize(slanted_sharp[:, slanted_sharp.shape[1] // 2])),
                "slanted_blur_center_row_norm": _canonical_profile(_channel_normalize(slanted_blur[slanted_blur.shape[0] // 2, :])),
                "slanted_blur_center_col_norm": _canonical_profile(_channel_normalize(slanted_blur[:, slanted_blur.shape[1] // 2])),
                "slanted_sharp_stats": _stats_vector(slanted_sharp),
                "slanted_blur_stats": _stats_vector(slanted_blur),
            },
            context={"scene": slanted_scene, "oi": oi_slanted_blur, "sensor": sensor_slanted_blur},
        )

    if case_name == "sensor_external_analysis_small":
        dut = sensor_create(asset_store=store)
        dut = sensor_set(dut, "name", "My Sensor")

        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        dut = sensor_set(dut, "wave", wave)
        dut = sensor_set(dut, "colorFilters", ie_read_spectra("RGB.mat", wave, asset_store=store))
        dut = sensor_set(dut, "irFilter", ie_read_spectra("infrared2.mat", wave, asset_store=store))
        dut = sensor_set(dut, "cfapattern", np.array([[2, 1], [3, 2]], dtype=int))
        dut = sensor_set(dut, "size", np.array([144, 176], dtype=int))
        dut = sensor_set(dut, "pixel name", "My Pixel")
        dut = sensor_set(dut, "pixel size constant fill factor", np.array([2.0e-6, 2.0e-6], dtype=float))
        dut = sensor_set(dut, "pixel spectral qe", ie_read_spectra("photodetector.mat", wave, asset_store=store))
        dut = sensor_set(dut, "pixel voltage swing", 1.5)

        volts = np.asarray(store.load_mat("scripts/sensor/dutData.mat")["volts"], dtype=float)
        dut = sensor_set(dut, "volts", volts)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "sensor_name": str(sensor_get(dut, "name")),
                "wave": np.asarray(sensor_get(dut, "wave"), dtype=float),
                "filter_spectra": np.asarray(sensor_get(dut, "filter spectra"), dtype=float),
                "ir_filter": np.asarray(sensor_get(dut, "ir filter"), dtype=float),
                "cfa_pattern": np.asarray(sensor_get(dut, "cfapattern"), dtype=int),
                "sensor_size": np.asarray(sensor_get(dut, "size"), dtype=int),
                "pixel_name": str(sensor_get(dut, "pixel name")),
                "pixel_size_m": np.asarray(sensor_get(dut, "pixel size"), dtype=float),
                "pixel_qe": np.asarray(sensor_get(dut, "pixel spectral qe"), dtype=float),
                "pixel_voltage_swing": float(sensor_get(dut, "pixel voltage swing")),
                "volts": np.asarray(sensor_get(dut, "volts"), dtype=float),
                "volts_stats": _stats_vector(np.asarray(sensor_get(dut, "volts"), dtype=float)),
            },
            context={"sensor": dut},
        )

    if case_name == "sensor_filter_transmissivities_small":
        sensor = sensor_create(asset_store=store)
        filters = np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float)
        modified = filters.copy()
        modified[:, 0] = modified[:, 0] * 0.2
        modified[:, 2] = modified[:, 2] * 0.5
        sensor = sensor_set(sensor, "filter transmissivities", modified)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(sensor_get(sensor, "wave"), dtype=float),
                "filters": np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float),
                "spectral_qe": np.asarray(sensor_get(sensor, "spectral qe"), dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_color_filter_gaussian_roundtrip_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        c_pos = np.arange(400.0, 701.0, 40.0, dtype=float)
        widths = np.full(c_pos.shape, 30.0, dtype=float)
        filters, wave = sensor_color_filter("gaussian", wave, c_pos, widths)
        payload = {
            "wavelength": wave,
            "data": filters,
            "filterNames": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "comment": "Gaussian filters created by parity sensorColorFilter",
            "peakWavelengths": c_pos,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = ie_save_color_filter(payload, f"{tmpdir}/gFiltersDeleteMe.mat")
            read_filters, read_names, file_data = ie_read_color_filter(wave, path, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "created_filters": filters,
                "read_filters": np.asarray(read_filters, dtype=float),
                "filter_names": np.asarray(read_names, dtype=object),
                "comment": str(file_data.get("comment", "")),
                "peak_wavelengths": np.asarray(file_data.get("peakWavelengths", np.array([], dtype=float)), dtype=float).reshape(-1),
            },
            context={},
        )

    if case_name == "sensor_color_filter_asset_nikond100_small":
        wave = np.arange(400.0, 1001.0, 1.0, dtype=float)
        filters, names, file_data = ie_read_color_filter(wave, "NikonD100", asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "filters": np.asarray(filters, dtype=float),
                "filter_names": np.asarray(names, dtype=object),
                "comment": str(file_data.get("comment", "")),
            },
            context={},
        )

    if case_name == "sensor_cfa_ycmy_small":
        sensor = sensor_create("ycmy", asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([4, 4], dtype=int))
        sensor = sensor_set(
            sensor,
            "volts",
            np.arange(16, dtype=float).reshape((4, 4), order="F") / 15.0,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pattern": np.asarray(sensor_get(sensor, "pattern"), dtype=int),
                "size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "filter_spectra": np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float),
                "rgb": np.asarray(sensor_get(sensor, "rgb"), dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_cfa_pattern_and_size_rgb_small":
        sensor = sensor_create("rgb", asset_store=store)
        sensor = sensor_set(sensor, "rows", 5)
        sensor = sensor_set(sensor, "cols", 7)
        sensor = sensor_set(sensor, "pattern and size", np.array([[2, 1, 2], [3, 2, 1], [2, 3, 2]], dtype=int))
        sensor = sensor_set(
            sensor,
            "volts",
            np.arange(54, dtype=float).reshape((6, 9), order="F") / 53.0,
        )
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pattern": np.asarray(sensor_get(sensor, "pattern"), dtype=int),
                "size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "rgb": np.asarray(sensor_get(sensor, "rgb"), dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_cfa_script_small":
        def _canonical_profile(values: Any, samples: int = 41) -> np.ndarray:
            row = np.asarray(values, dtype=float).reshape(-1)
            support = np.linspace(-1.0, 1.0, row.size, dtype=float)
            query = np.linspace(-1.0, 1.0, samples, dtype=float)
            return np.interp(query, support, row)

        fov = 20.0
        pixel_size = np.array([1.4e-6, 1.4e-6], dtype=float)

        scene = scene_from_file("zebra.jpg", "rgb", 300, display_create(asset_store=store), asset_store=store)
        scene = scene_set(scene, "fov", fov)
        oi = oi_compute(oi_create(asset_store=store), scene)

        default_sensor = sensor_create(asset_store=store)
        default_sensor = sensor_set(default_sensor, "pixel size constant fill factor", pixel_size)
        default_sensor = sensor_set_size_to_fov(default_sensor.clone(), (scene.fields["fov_deg"], scene.fields["vfov_deg"]), oi)
        default_sensor = sensor_compute(default_sensor, oi)

        branch_sensors: list[tuple[str, Sensor]] = [("default", default_sensor)]

        bayer = sensor_create(asset_store=store)
        bayer = sensor_set(bayer, "fov", fov, oi)
        bayer = sensor_set(bayer, "name", "Bayer")
        bayer = sensor_set(bayer, "pixel size constant fill factor", pixel_size)
        bayer = sensor_compute(bayer, oi)
        branch_sensors.append(("bayer", bayer))

        ycmy = sensor_create("ycmy", asset_store=store)
        ycmy = sensor_set(ycmy, "fov", fov, oi)
        ycmy = sensor_set(ycmy, "name", "cmy")
        ycmy = sensor_set(ycmy, "pixel size constant fill factor", pixel_size)
        ycmy = sensor_compute(ycmy, oi)
        branch_sensors.append(("ycmy", ycmy))

        rgb = sensor_create("rgb", asset_store=store)
        rgb = sensor_set(rgb, "pattern and size", np.array([[2, 1, 2], [3, 2, 1], [2, 3, 2]], dtype=int))
        rgb = sensor_set(rgb, "fov", fov, oi)
        rgb = sensor_set(rgb, "name", "3x3 RGB")
        rgb = sensor_set(rgb, "pixel size constant fill factor", pixel_size)
        rgb = sensor_compute(rgb, oi)
        branch_sensors.append(("rgb", rgb))

        rgbw = sensor_create("rgbw", asset_store=store)
        rgbw = sensor_set(rgbw, "fov", fov, oi)
        rgbw = sensor_set(rgbw, "name", "rgbw")
        rgbw = sensor_set(rgbw, "pixel size constant fill factor", pixel_size)
        rgbw = sensor_compute(rgbw, oi)
        branch_sensors.append(("rgbw", rgbw))

        quad = sensor_create(asset_store=store)
        quad = sensor_set(quad, "pattern", np.array([[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]], dtype=int))
        quad = sensor_set(quad, "fov", fov, oi)
        quad = sensor_set(quad, "name", "quad")
        quad = sensor_set(quad, "pixel size constant fill factor", pixel_size)
        quad = sensor_compute(quad, oi)
        branch_sensors.append(("quad", quad))

        branch_labels: list[str] = []
        branch_sizes = np.zeros((len(branch_sensors), 2), dtype=int)
        branch_pattern_shapes = np.zeros((len(branch_sensors), 2), dtype=int)
        branch_patterns_padded = np.zeros((len(branch_sensors), 4, 4), dtype=int)
        branch_cfa_names: list[str] = []
        branch_filter_letters: list[str] = []
        branch_mean_rgb_norm = np.zeros((len(branch_sensors), 3), dtype=float)
        branch_center_rgb_norm = np.zeros((len(branch_sensors), 3), dtype=float)
        branch_center_row_luma_norm = np.zeros((len(branch_sensors), 41), dtype=float)
        branch_center_col_luma_norm = np.zeros((len(branch_sensors), 41), dtype=float)

        for index, (label, sensor) in enumerate(branch_sensors):
            rgb_image = np.asarray(sensor_get(sensor, "rgb"), dtype=float)
            luma = np.mean(rgb_image, axis=2)
            center_row = luma[luma.shape[0] // 2, :]
            center_col = luma[:, luma.shape[1] // 2]
            center_rgb = rgb_image[rgb_image.shape[0] // 2, rgb_image.shape[1] // 2, :]
            pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)

            branch_labels.append(label)
            branch_sizes[index, :] = np.asarray(sensor_get(sensor, "size"), dtype=int)
            branch_pattern_shapes[index, :] = np.array(pattern.shape, dtype=int)
            branch_patterns_padded[index, : pattern.shape[0], : pattern.shape[1]] = pattern
            branch_cfa_names.append(str(sensor_get(sensor, "cfaname")))
            branch_filter_letters.append(str(sensor_get(sensor, "filter color letters")))
            branch_mean_rgb_norm[index, :] = _channel_normalize(np.mean(rgb_image.reshape(-1, 3), axis=0))
            branch_center_rgb_norm[index, :] = _channel_normalize(center_rgb)
            branch_center_row_luma_norm[index, :] = _canonical_profile(_channel_normalize(center_row))
            branch_center_col_luma_norm[index, :] = _canonical_profile(_channel_normalize(center_col))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "scene_size": np.asarray(scene_get(scene, "size"), dtype=int),
                "oi_size": np.asarray(oi_get(oi, "size"), dtype=int),
                "branch_labels": np.asarray(branch_labels, dtype=object),
                "branch_sizes": branch_sizes,
                "branch_pattern_shapes": branch_pattern_shapes,
                "branch_patterns_padded": branch_patterns_padded,
                "branch_cfa_names": np.asarray(branch_cfa_names, dtype=object),
                "branch_filter_letters": np.asarray(branch_filter_letters, dtype=object),
                "branch_mean_rgb_norm": branch_mean_rgb_norm,
                "branch_center_rgb_norm": branch_center_rgb_norm,
                "branch_center_row_luma_norm": branch_center_row_luma_norm,
                "branch_center_col_luma_norm": branch_center_col_luma_norm,
            },
            context={},
        )

    if case_name == "sensor_snr_components_small":
        sensor = sensor_create(asset_store=store)
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
        read_noise = float(sensor_get(sensor, "pixel read noise volts"))
        volts = np.logspace(np.log10(voltage_swing) - 4.0, np.log10(voltage_swing), 20, dtype=float)

        sensor = sensor_set(sensor, "pixel read noise volts", 3.0 * read_noise)
        sensor = sensor_set(sensor, "gainSD", 2.0)
        sensor = sensor_set(sensor, "offsetSD", voltage_swing * 0.005)

        snr, volts, snr_shot, snr_read, snr_dsnu, snr_prnu = sensor_snr(sensor, volts)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "volts": np.asarray(volts, dtype=float),
                "snr": np.asarray(snr, dtype=float),
                "snr_shot": np.asarray(snr_shot, dtype=float),
                "snr_read": np.asarray(snr_read, dtype=float),
                "snr_dsnu": np.asarray(snr_dsnu, dtype=float),
                "snr_prnu": np.asarray(snr_prnu, dtype=float),
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_counting_photons_small":
        scene = scene_create("uniform equal photon", np.array([128, 128], dtype=int), asset_store=store)
        scene = scene_set(scene, "mean luminance", 10.0)

        oi = oi_create("diffraction limited", asset_store=store)
        roi_rect = np.array([41, 31, 16, 23], dtype=int)
        fnumbers = np.arange(2.0, 17.0, dtype=float)
        total_q = np.zeros(fnumbers.shape, dtype=float)
        aperture_d = np.zeros(fnumbers.shape, dtype=float)
        spectral_irradiance = np.empty(0, dtype=float)

        for idx, f_number in enumerate(fnumbers):
            oi = oi_set(oi, "optics fnumber", float(f_number))
            oi = oi_compute(oi, scene)
            aperture_d[idx] = float(oi_get(oi, "optics aperture diameter", "mm"))
            spectral_irradiance = np.asarray(oi_get(oi, "roi mean photons", roi_rect), dtype=float).reshape(-1)
            total_q[idx] = float(np.sum(spectral_irradiance))

        s_factor = (1.0e-6**2) * 50.0e-3
        snr = (total_q * s_factor) / np.sqrt(np.maximum(total_q * s_factor, 1.0e-12))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": np.asarray(oi_get(oi, "wave"), dtype=float),
                "fnumbers": fnumbers,
                "aperture_d": aperture_d,
                "spectral_irradiance": spectral_irradiance,
                "total_q": total_q,
                "snr": snr,
            },
            context={"scene": scene, "oi": oi},
        )

    if case_name == "sensor_poisson_noise_small":
        scene = scene_create("macbeth", asset_store=store)
        scene = scene_set(scene, "fov", 10.0)
        oi = oi_create("diffraction limited", asset_store=store)
        oi = oi_compute(oi, scene)

        sensor = sensor_create("imx363", asset_store=store)
        sensor = sensor_set(sensor, "row", 256)
        sensor = sensor_set(sensor, "col", 256)
        sensor = sensor_set(sensor, "exp time", 0.016)
        sensor = sensor_compute(sensor, oi, seed=1)

        rect = np.array([96, 156, 24, 28], dtype=int)
        sensor = sensor_set(sensor, "roi", rect)
        dv = np.asarray(sensor_get(sensor, "roi dv", rect), dtype=float)
        finite_dv = dv[np.isfinite(dv)]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "rect": rect,
                "roi_mean_dv": float(np.mean(finite_dv)),
                "roi_std_dv": float(np.std(finite_dv, ddof=1)),
                "roi_percentiles": np.percentile(finite_dv, [10.0, 50.0, 90.0]),
                "sqrt_mean_dv": float(np.sqrt(np.mean(finite_dv))),
                "sensor_mean_dv": float(np.nanmean(np.asarray(sensor_get(sensor, "dv"), dtype=float))),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_estimation_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        macbeth_chart = ie_read_spectra("macbethChart", wave, asset_store=store)
        illuminant_d65 = ie_read_spectra("D65.mat", wave, asset_store=store).reshape(-1)
        sensors = ie_read_spectra("cMatch/camera", wave, asset_store=store)
        cones = ie_read_spectra("SmithPokornyCones", wave, asset_store=store)

        spectral_signals = illuminant_d65[:, None] * macbeth_chart
        rgb_responses = sensors.T @ spectral_signals

        estimate_full = (rgb_responses @ np.linalg.pinv(spectral_signals)).T
        rgb_pred_full = estimate_full.T @ spectral_signals

        sample_indices = np.arange(0, macbeth_chart.shape[1], 5, dtype=int)
        estimate_sparse = (rgb_responses[:, sample_indices] @ np.linalg.pinv(spectral_signals[:, sample_indices])).T
        rgb_pred_sparse = estimate_sparse.T @ spectral_signals

        gray_series = np.arange(3, macbeth_chart.shape[1], 4, dtype=int)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "green_reflectance": macbeth_chart[:, 6],
                "red_reflectance": macbeth_chart[:, 10],
                "gray_reflectance": macbeth_chart[:, 11],
                "illuminant_d65": illuminant_d65,
                "sensors": sensors,
                "cones": cones,
                "rgb_responses_gray": rgb_responses[:, gray_series],
                "estimate_full": estimate_full,
                "rgb_pred_full": rgb_pred_full,
                "estimate_sparse": estimate_sparse,
                "rgb_pred_sparse": rgb_pred_sparse,
            },
            context={},
        )

    if case_name == "sensor_macbeth_daylight_estimate_small":
        wave = np.arange(400.0, 701.0, 10.0, dtype=float)
        reflectance = ie_read_spectra("macbethChart", wave, asset_store=store)

        sensor = sensor_create(asset_store=store)
        sensor_filters = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)

        day_basis_energy = ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=store)
        day_basis_quanta = energy_to_quanta(day_basis_energy, wave)

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
        normal_matrix = design_matrix.T @ design_matrix
        rhs = design_matrix.T @ camera_stacked
        solved_weights = np.linalg.solve(normal_matrix, rhs)
        estimated_weights = solved_weights / solved_weights[0]
        estimated_illuminant = day_basis_quanta @ estimated_weights

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "reflectance": reflectance,
                "sensor_filters": sensor_filters,
                "day_basis_quanta": day_basis_quanta,
                "true_weights": true_weights,
                "illuminant_photons": illuminant_photons,
                "camera_data": camera_data,
                "design_matrix": design_matrix,
                "camera_stacked": camera_stacked,
                "normal_matrix": normal_matrix,
                "rhs": rhs,
                "estimated_weights": estimated_weights,
                "estimated_illuminant": estimated_illuminant,
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_spectral_radiometer_small":
        scene = scene_create("uniform d65", asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)

        wave = np.arange(400.0, 701.0, 1.0, dtype=float)
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, "radiometer", asset_store=store)
        w_samples = np.array([float(name) for name in filter_names], dtype=float)
        pattern = np.arange(1, filter_spectra.shape[1] + 1, dtype=int).reshape(1, -1)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "wave", wave)
        sensor = sensor_set(sensor, "filter spectra", filter_spectra)
        sensor = sensor_set(sensor, "filter names", filter_names)
        sensor = sensor_set(sensor, "pattern", pattern)
        sensor = sensor_set(sensor, "size", np.array([10, filter_spectra.shape[1]], dtype=int))
        sensor = sensor_set(sensor, "pixel fill factor", 1.0)
        sensor = sensor_set(sensor, "pixel size same fill factor", np.array([1.5e-6, 1.5e-6], dtype=float))
        sensor = sensor_set(sensor, "exposure time", 1.0 / 100.0)

        sensor_noisy = sensor_set(sensor.clone(), "noise flag", -2)
        sensor_noisy = sensor_compute(sensor_noisy, oi)
        electrons_noisy = np.asarray(sensor_get(sensor_noisy, "electrons"), dtype=float)

        sensor_noise_free = sensor_set(sensor.clone(), "noise flag", -1)
        sensor_noise_free = sensor_compute(sensor_noise_free, oi)
        electrons_noise_free = np.asarray(sensor_get(sensor_noise_free, "electrons"), dtype=float)

        noisy_line = electrons_noisy[4, :]
        noise_free_line = electrons_noise_free[4, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "w_samples": w_samples,
                "filter_pattern": pattern,
                "sensor_size": np.asarray(sensor_get(sensor_noise_free, "size"), dtype=int),
                "filter_spectra": np.asarray(sensor_get(sensor_noise_free, "filter spectra"), dtype=float),
                "noise_free_line": noise_free_line,
                "shot_sd_line": np.sqrt(np.maximum(noise_free_line, 0.0)),
                "noisy_line_stats": _stats_vector(noisy_line),
                "noisy_full_stats": _stats_vector(electrons_noisy),
                "noise_free_full_stats": _stats_vector(electrons_noise_free),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_noise_free},
        )

    if case_name == "sensor_spectral_estimation_small":
        scene = scene_create("uniform ee", asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float)

        oi = oi_create("default", asset_store=store)
        oi = oi_set(oi, "optics model", "diffraction limited")
        oi = oi_set(oi, "optics fnumber", 0.01)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([64, 64], dtype=int))
        sensor = sensor_set(sensor, "auto exposure", True)

        scene = scene_set(scene, "fov", float(sensor_get(sensor, "fov", scene, oi)) * 1.5)

        wave_step = 50.0
        centers = np.arange(wave[0], wave[-1] + wave_step, wave_step, dtype=float)
        width = wave_step / 2.0
        spd = np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / width) ** 2)
        spd *= 1.0e16

        n_filters = int(sensor_get(sensor, "nfilters"))
        responsivity = np.zeros((n_filters, centers.size), dtype=float)
        exposure_times = np.zeros(centers.size, dtype=float)

        for ii in range(centers.size):
            spd_image = np.broadcast_to(spd[:, ii][None, None, :], (32, 32, wave.size)).copy()
            trial_scene = scene_set(scene, "photons", spd_image)
            trial_oi = oi_compute(oi, trial_scene)
            computed = sensor_compute(sensor, trial_oi, False)
            exposure_times[ii] = float(sensor_get(computed, "exposure time"))

            for jj in range(n_filters):
                volts = np.asarray(sensor_get(computed, "volts", jj + 1), dtype=float)
                responsivity[jj, ii] = float(np.mean(volts)) / max(exposure_times[ii], 1e-12)

        weights = responsivity @ np.linalg.pinv(spd.T @ spd)
        estimated_filters = (weights @ spd.T).T
        estimated_peak = float(np.max(estimated_filters))
        if abs(estimated_peak) > 0.0:
            estimated_filters = estimated_filters / estimated_peak

        sensor_filters = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
        sensor_filters = sensor_filters / max(float(np.max(sensor_filters)), 1e-12)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "wave": wave,
                "centers": centers,
                "spd": spd,
                "exposure_times": exposure_times,
                "responsivity": responsivity,
                "weights": weights,
                "estimated_filters": estimated_filters,
                "sensor_filters": sensor_filters,
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_exposure_color_small":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(asset_store=store), scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set_size_to_fov(sensor, float(scene_get(scene, "fov")), oi)

        filters = np.asarray(sensor_get(sensor, "filter transmissivities"), dtype=float)
        filters = filters.copy()
        filters[:, 0] *= 0.2
        filters[:, 2] *= 0.5
        sensor = sensor_set(sensor, "filter transmissivities", filters)
        sensor = sensor_set(sensor, "auto exposure", "on")

        sensor = sensor_compute(sensor, oi)
        exposure_time = float(sensor_get(sensor, "exposure time"))

        ip = ip_create(asset_store=store)
        ip = ip_compute(ip, sensor)
        combined_transform = np.asarray(ip_get(ip, "combined transform"), dtype=float)

        ip = ip_set(ip, "transform method", "current")
        sensor = sensor_set(sensor, "auto exposure", "off")
        sensor = sensor_set(sensor, "exposure time", 3.0 * exposure_time)
        sensor = sensor_compute(sensor, oi)
        ip = ip_compute(ip, sensor)
        result = np.asarray(ip_get(ip, "result"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "exposure_time": exposure_time,
                "combined_transform": combined_transform,
                "mean_rgb": np.mean(result, axis=(0, 1)),
                "white_patch_rgb": np.mean(result[28:44, 36:52, :], axis=(0, 1)),
                "result": result,
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "sensor_exposure_bracket_small":
        scene = scene_create(asset_store=store)
        scene = scene_set(scene, "fov", 4.0)

        oi = oi_create(asset_store=store)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        integration_times = np.array([0.02, 0.04, 0.08, 0.16, 0.32], dtype=float)
        sensor = sensor_set(sensor, "exp time", integration_times)
        sensor = sensor_set(sensor, "exposure plane", int(np.floor(integration_times.size / 2.0) + 1))
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_compute(sensor, oi, False, seed=0)

        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        center_row = volts[volts.shape[0] // 2, :, :]
        center_col = volts[:, volts.shape[1] // 2, :]

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "integration_times": np.asarray(sensor_get(sensor, "integration time"), dtype=float),
                "exposure_plane": int(sensor_get(sensor, "exposure plane")),
                "n_captures": int(sensor_get(sensor, "n captures")),
                "volts_means": np.mean(volts, axis=(0, 1)),
                "center_pixel": volts[volts.shape[0] // 2, volts.shape[1] // 2, :],
                "center_row_mean": np.mean(center_row, axis=0),
                "center_col_mean": np.mean(center_col, axis=0),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_exposure_cfa_small":
        scene = scene_create(asset_store=store)
        scene = scene_set(scene, "fov", 4.0)

        oi = oi_create(asset_store=store)
        oi = oi_compute(oi, scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 0)
        bluish = np.array([[0.04, 0.03], [0.30, 0.02]], dtype=float)
        sensor_b = sensor_compute(sensor_set(sensor.clone(), "exposure duration", bluish), oi, False, seed=0)

        reddish = np.array([[0.04, 0.70], [0.03, 0.02]], dtype=float)
        sensor_r = sensor_compute(sensor_set(sensor.clone(), "exposure duration", reddish), oi, False, seed=0)

        camera = camera_create(asset_store=store)
        camera = camera_set(camera, "sensor noise flag", 0)
        camera = camera_set(camera, "sensor exposure duration", reddish)
        camera = camera_compute(camera, scene, asset_store=store)
        camera_sensor = camera_get(camera, "sensor")
        bluish_volts = np.asarray(sensor_get(sensor_b, "volts"), dtype=float)
        reddish_volts = np.asarray(sensor_get(sensor_r, "volts"), dtype=float)
        camera_volts = np.asarray(sensor_get(camera_sensor, "volts"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "bluish_mean_volts": float(np.mean(bluish_volts)),
                "reddish_mean_volts": float(np.mean(reddish_volts)),
                "reddish_center_pixel": reddish_volts[
                    int(sensor_r.fields["size"][0] // 2), int(sensor_r.fields["size"][1] // 2)
                ],
                "camera_mean_volts": float(np.mean(camera_volts)),
                "camera_center_pixel": camera_volts[
                    int(camera_sensor.fields["size"][0] // 2), int(camera_sensor.fields["size"][1] // 2)
                ],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_r, "camera": camera},
        )

    if case_name == "sensor_dark_voltage_small":
        scene = scene_create("uniform ee", asset_store=store)
        scene = scene_set(scene, "fov", 5.0)
        dark_scene = scene_adjust_luminance(scene, 1e-8, asset_store=store)

        oi = oi_compute(oi_create(asset_store=store), dark_scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 2)
        sensor = sensor_set(sensor, "noise seed", 1)
        exp_times = np.logspace(0.0, 1.5, 10)
        n_filters = int(sensor_get(sensor, "nfilters"))

        volt_columns: list[np.ndarray] = []
        for exp_time in exp_times:
            sensor = sensor_set(sensor, "exposureTime", float(exp_time))
            computed = sensor_compute(sensor, oi, 0)
            if n_filters == 3:
                volts = np.asarray(sensor_get(computed, "volts", 2), dtype=float).reshape(-1, order="F")
            else:
                volts = np.asarray(sensor_get(computed, "volts"), dtype=float).reshape(-1, order="F")
            volt_columns.append(volts)

        volts = np.column_stack(volt_columns)
        mean_volts = np.mean(volts, axis=0)
        dark_voltage_estimate, offset = ie_fit_line(exp_times, mean_volts)
        true_dark_voltage = float(sensor_get(sensor, "dark voltage"))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "exp_times": exp_times,
                "mean_volts": mean_volts,
                "dark_voltage_estimate": float(dark_voltage_estimate),
                "offset": float(offset),
                "true_dark_voltage": true_dark_voltage,
            },
            context={"scene": dark_scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_prnu_estimate_small":
        scene = scene_create("uniform ee", asset_store=store)
        scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
        scene = scene_set(scene, "fov", 2.0)

        oi = oi_create("default", asset_store=store)
        oi = oi_set(oi, "optics offaxis method", "skip")

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([64, 64], dtype=int))
        sensor = sensor_set(sensor, "noise flag", 2)
        scene = scene_set(scene, "fov", float(sensor_get(sensor, "fov")) * 1.5)
        oi = oi_compute(oi, scene)

        exp_times = np.tile(np.arange(40.0, 62.0, 2.0, dtype=float) / 1000.0, 3)
        sensor = sensor_set(sensor, "dsnu sigma", 0.0)
        sensor = sensor_set(sensor, "prnu sigma", 1.0)
        sensor = sensor_set(sensor, "pixel read noise volts", 0.0)
        sensor = sensor_set(sensor, "pixel dark voltage", 0.0)

        volt_columns: list[np.ndarray] = []
        n_filters = int(sensor_get(sensor, "nfilters"))
        for ii, exp_time in enumerate(exp_times, start=1):
            trial_sensor = sensor.clone()
            trial_sensor = sensor_set(trial_sensor, "exposure time", float(exp_time))
            computed = sensor_compute(trial_sensor, oi, seed=ii)
            if n_filters == 3:
                volts = np.asarray(sensor_get(computed, "volts", 2), dtype=float).reshape(-1, order="F")
            else:
                volts = np.asarray(sensor_get(computed, "volts"), dtype=float).reshape(-1, order="F")
            volt_columns.append(volts)

        volts = np.column_stack(volt_columns)
        a = np.column_stack([exp_times, np.ones(exp_times.shape[0], dtype=float)])
        fit = np.linalg.lstsq(a, volts.T, rcond=None)[0]
        slopes = np.asarray(fit[0, :], dtype=float)
        slopes = slopes / max(float(np.mean(slopes)), 1e-12)
        offsets = np.asarray(fit[1, :], dtype=float)
        prnu_estimate = 100.0 * float(np.std(slopes))

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "exp_times": exp_times,
                "prnu_estimate": prnu_estimate,
                "slope_mean": float(np.mean(slopes)),
                "slope_std": float(np.std(slopes)),
                "offset_mean": float(np.mean(offsets)),
                "offset_std": float(np.std(offsets)),
                "slope_sample": slopes[:8],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_dsnu_estimate_small":
        scene = scene_create("uniform ee", asset_store=store)
        dark_scene = scene_adjust_luminance(scene, 0.1, asset_store=store)

        oi = oi_create("default", asset_store=store)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([64, 64], dtype=int))
        dark_scene = scene_set(dark_scene, "fov", float(sensor_get(sensor, "fov")) * 1.5)
        dark_oi = oi_compute(oi, dark_scene)

        sensor = sensor_set(sensor, "dsnu sigma", 0.05)
        sensor = sensor_set(sensor, "prnu sigma", 0.1)
        sensor = sensor_set(sensor, "exposure time", 0.001)
        sensor = sensor_set(sensor, "pixel read noise volts", 0.001)
        sensor = sensor_set(sensor, "noise flag", 2)

        n_filters = int(sensor_get(sensor, "nfilters"))
        n_repeats = 25
        if n_filters == 3:
            n_samp = int(np.prod(sensor_get(sensor, "size")) // 2)
        else:
            n_samp = int(np.prod(sensor_get(sensor, "size")))
        volts = np.zeros((n_samp, n_repeats), dtype=float)
        for ii in range(n_repeats):
            computed = sensor_compute(sensor, dark_oi, seed=ii + 1)
            if n_filters == 3:
                volts[:, ii] = np.asarray(sensor_get(computed, "volts", 2), dtype=float).reshape(-1, order="F")
            else:
                volts[:, ii] = np.asarray(sensor_get(computed, "volts"), dtype=float).reshape(-1, order="F")

        mirrored = volts[volts > 1.0e-6]
        mirrored = np.concatenate([-mirrored.reshape(-1), mirrored.reshape(-1)])
        mean_offset = np.mean(volts, axis=1)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "estimated_dsnu": float(np.std(mirrored)),
                "mean_offset_mean": float(np.mean(mean_offset)),
                "mean_offset_std": float(np.std(mean_offset)),
                "mean_offset_percentiles": np.percentile(mean_offset, [10.0, 50.0, 90.0]),
            },
            context={"scene": dark_scene, "oi": dark_oi, "sensor": sensor},
        )

    if case_name == "sensor_size_resolution_small":
        pixel_size_um = np.arange(0.8, 3.0 + 1.0e-9, 0.2, dtype=float)
        pixel_size_m = pixel_size_um * 1.0e-6
        half_inch_size_m = np.asarray(sensor_formats("half inch"), dtype=float).reshape(-1)
        quarter_inch_size_m = np.asarray(sensor_formats("quarter inch"), dtype=float).reshape(-1)

        half_rows = half_inch_size_m[0] / pixel_size_m
        half_cols = half_inch_size_m[1] / pixel_size_m
        quarter_rows = quarter_inch_size_m[0] / pixel_size_m
        quarter_cols = quarter_inch_size_m[1] / pixel_size_m

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "pixel_size_um": pixel_size_um,
                "half_inch_size_m": half_inch_size_m,
                "quarter_inch_size_m": quarter_inch_size_m,
                "half_rows": half_rows,
                "half_cols": half_cols,
                "half_megapixels": np.asarray(ie_n_to_megapixel(half_rows * half_cols), dtype=float),
                "quarter_rows": quarter_rows,
                "quarter_cols": quarter_cols,
                "quarter_megapixels": np.asarray(ie_n_to_megapixel(quarter_rows * quarter_cols), dtype=float),
            },
            context={},
        )

    if case_name == "sensor_cfa_point_spread_small":
        scene = scene_create("point array", asset_store=store)
        wave = np.asarray(scene_get(scene, "wave"), dtype=float)
        scene = scene_adjust_illuminant(scene, blackbody(wave, 8000.0), asset_store=store)
        scene = scene_set(scene, "fov", 2.0)

        oi = oi_create("diffraction limited", asset_store=store)
        pixel_size_m = np.array([1.4e-6, 1.4e-6], dtype=float)
        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "pixel size constant fill factor", pixel_size_m)
        sensor = sensor_set(sensor, "auto exposure", True)

        rect = np.array([32, 24, 11, 11], dtype=int)
        ff_numbers = np.array([2.0, 4.0, 8.0, 12.0], dtype=float)
        x_um = np.arange(rect[2] + 1, dtype=float) * pixel_size_m[0]
        x_um = (x_um - float(np.mean(x_um))) * 1.0e6

        crop_mean_rgb = np.zeros((ff_numbers.size, 3), dtype=float)
        crop_peak_rgb = np.zeros((ff_numbers.size, 3), dtype=float)
        green_row_width_30_um = np.zeros(ff_numbers.size, dtype=float)
        green_row_width_50_um = np.zeros(ff_numbers.size, dtype=float)
        green_row_width_90_um = np.zeros(ff_numbers.size, dtype=float)
        red_center_cols_norm = np.zeros((ff_numbers.size, rect[3] + 1), dtype=float)

        row_slice = slice(rect[1] - 1, rect[1] + rect[3])
        col_slice = slice(rect[0] - 1, rect[0] + rect[2])
        for index, f_number in enumerate(ff_numbers):
            oi_ff = oi_set(oi, "optics fnumber", float(f_number))
            oi_ff = oi_compute(oi_ff, scene)
            sensor_ff = sensor_compute(sensor, oi_ff)
            image = np.asarray(sensor_get(sensor_ff, "rgb"), dtype=float)
            crop = image[row_slice, col_slice, :]
            crop_mean_rgb[index, :] = np.mean(crop, axis=(0, 1))
            crop_peak_rgb[index, :] = np.max(crop, axis=(0, 1))
            center_row = _channel_normalize(crop[(crop.shape[0] - 1) // 2, :, 1])
            center_col = crop[:, (crop.shape[1] - 1) // 2, 0]
            dx_um = abs(float(x_um[1] - x_um[0])) if x_um.size > 1 else 0.0
            green_row_width_30_um[index] = np.count_nonzero(center_row >= 0.3) * dx_um
            green_row_width_50_um[index] = np.count_nonzero(center_row >= 0.5) * dx_um
            green_row_width_90_um[index] = np.count_nonzero(center_row >= 0.9) * dx_um
            red_center_cols_norm[index, :] = _channel_normalize(center_col)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "ff_numbers": ff_numbers,
                "pixel_size_um": pixel_size_m * 1.0e6,
                "rect": rect,
                "x_um": x_um,
                "crop_mean_rgb": crop_mean_rgb,
                "crop_peak_rgb": crop_peak_rgb,
                "green_row_width_30_um": green_row_width_30_um,
                "green_row_width_50_um": green_row_width_50_um,
                "green_row_width_90_um": green_row_width_90_um,
                "red_center_cols_norm": red_center_cols_norm,
            },
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_spatial_resolution_small":
        scene = scene_create("sweepFrequency", asset_store=store)
        scene = scene_set(scene, "fov", 1.0)

        oi = oi_create(asset_store=store)
        oi = oi_set(oi, "optics fnumber", 4.0)
        oi = oi_set(oi, "optics focal length", 0.004)
        oi = oi_compute(oi, scene)

        sensor = sensor_create("monochrome", asset_store=store)
        sensor = sensor_set(sensor, "noise flag", 0)
        sensor = sensor_compute(sensor, oi)
        coarse_row = int(round(float(sensor_get(sensor, "rows")) / 2.0))
        coarse_support = sensor_get(sensor, "spatial support", "microns")
        coarse_volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)

        oi_row = int(round(float(oi_get(oi, "rows")) / 2.0))
        oi_line, _ = oi_plot(oi, "horizontal line illuminance", np.array([1, oi_row], dtype=int))

        sensor_small = sensor_set(sensor, "pixel size Constant Fill Factor", np.array([2.0e-6, 2.0e-6], dtype=float))
        sensor_small = sensor_compute(sensor_small, oi)
        fine_row = int(round(float(sensor_get(sensor_small, "rows")) / 2.0))
        fine_support = sensor_get(sensor_small, "spatial support", "microns")
        fine_volts = np.asarray(sensor_get(sensor_small, "volts"), dtype=float)

        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "coarse_pixPos": np.asarray(coarse_support["x"], dtype=float),
                "coarse_pixData": np.asarray(coarse_volts[coarse_row - 1, :], dtype=float),
                "fine_pixPos": np.asarray(fine_support["x"], dtype=float),
                "fine_pixData": np.asarray(fine_volts[fine_row - 1, :], dtype=float),
                "oi_pos": np.asarray(oi_line["pos"], dtype=float),
                "oi_data": np.asarray(oi_line["data"], dtype=float),
            },
            context={"scene": scene, "oi": oi, "sensor": sensor_small},
        )

    if case_name == "sensor_fpn_noise_modes_small":
        scene = scene_create("uniform", 512, asset_store=store)
        scene = scene_set(scene, "fov", 8.0)

        oi = oi_compute(oi_create("wvf", asset_store=store), scene)

        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "match oi", oi)
        sensor = sensor_set(sensor, "dsnu sigma", 0.05)
        sensor = sensor_set(sensor, "prnu sigma", 1.0)
        sensor = sensor_set(sensor, "read noise volts", 0.1)

        xy = np.array([1, 320], dtype=int)
        payload: dict[str, Any] = {"case_name": case_name}
        for noise_flag, label in ((0, "noise0"), (-2, "noiseM2"), (1, "noise1"), (2, "noise2")):
            trial_sensor = sensor.clone()
            trial_sensor = sensor_set(trial_sensor, "noise flag", noise_flag)
            trial_sensor = sensor_set(trial_sensor, "reuse noise", True)
            trial_sensor = sensor_set(trial_sensor, "noise seed", 0)
            computed = sensor_compute(trial_sensor, oi, seed=0)
            udata, _ = sensor_plot(computed, "volts hline", xy, "two lines", True)
            pix_pos = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in udata["pixPos"]])
            pix_data = np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in udata["pixData"]])
            pix_color = np.asarray(udata["pixColor"], dtype=int).reshape(-1)
            if noise_flag == 0:
                payload["pixPos"] = pix_pos
                payload["pixColor"] = pix_color
                payload["noise0_pixData"] = pix_data
            else:
                payload[f"{label}_stats"] = _stats_vector(pix_data)

        return ParityCaseResult(
            payload=payload,
            context={"scene": scene, "oi": oi, "sensor": sensor},
        )

    if case_name == "sensor_description_fpn_small":
        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "dsnu sigma", 0.05)
        sensor = sensor_set(sensor, "prnu sigma", 1.0)
        sensor = sensor_set(sensor, "read noise volts", 0.1)
        table, string_table, handle = sensor_description(sensor, show=False, close_window=False)
        rows = {str(row[0]): str(row[1]) for row in string_table.tolist()}
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "title": str(table.title),
                "handle_title": "" if handle is None else str(handle.title),
                "row_count": int(string_table.shape[0]),
                "col_count": int(string_table.shape[1]),
                "read_noise_volts": rows["Read noise (V)"],
                "analog_gain": rows["Analog gain"],
                "exposure_time": rows["Exposure time"],
            },
            context={"sensor": sensor},
        )

    if case_name == "sensor_dng_read_crop_small":
        dng_path = store.resolve("data/images/rawcamera/MCC-centered.dng")
        sensor, info = sensor_dng_read(
            dng_path,
            "full info",
            False,
            "crop",
            [500, 1000, 256, 256],
            asset_store=store,
        )
        ip = ip_compute(ip_create(asset_store=store), sensor, asset_store=store)
        result = np.asarray(ip.data["result"], dtype=float)
        result = result / np.maximum(np.max(result, axis=(0, 1), keepdims=True), 1e-12)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "size": np.asarray(sensor_get(sensor, "size"), dtype=int),
                "pattern": np.asarray(sensor_get(sensor, "pattern"), dtype=int),
                "black_level": float(sensor_get(sensor, "black level")),
                "exp_time": float(sensor_get(sensor, "exp time")),
                "iso_speed": float(info["isoSpeed"]),
                "digital_values": np.asarray(sensor_get(sensor, "digital values"), dtype=float),
                "result": result,
            },
            context={"sensor": sensor, "ip": ip},
        )

    if case_name == "ip_default_pipeline":
        scene = scene_create(asset_store=store)
        oi = oi_compute(oi_create(), scene, crop=True)
        sensor = sensor_compute(sensor_set(sensor_create(asset_store=store), "noise flag", 0), oi, seed=0)
        ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "input": ip.data["input"],
                "sensorspace": ip.data["sensorspace"],
                "result": ip.data["result"],
            },
            context={"scene": scene, "oi": oi, "sensor": sensor, "ip": ip},
        )

    if case_name == "camera_default_pipeline":
        scene = scene_create(asset_store=store)
        camera = camera_create(asset_store=store)
        camera.fields["sensor"] = sensor_set(camera.fields["sensor"], "noise flag", 0)
        camera = camera_compute(camera, scene, asset_store=store)
        return ParityCaseResult(
            payload={
                "case_name": case_name,
                "result": camera.fields["ip"].data["result"],
                "sensor_volts": camera.fields["sensor"].data["volts"],
                "oi_photons": camera.fields["oi"].data["photons"],
            },
            context={
                "scene": scene,
                "camera": camera,
                "oi": camera.fields["oi"],
                "sensor": camera.fields["sensor"],
                "ip": camera.fields["ip"],
            },
        )

    raise KeyError(f"Unknown parity case: {case_name}")


def run_python_case(case_name: str, *, asset_store: AssetStore | None = None) -> dict[str, Any]:
    return run_python_case_with_context(case_name, asset_store=asset_store).payload
