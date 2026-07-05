"""Spectral and color helpers."""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .assets import AssetStore, ie_read_color_filter
from .exceptions import UnsupportedOptionError
from .utils import (
    blackbody,
    energy_to_quanta,
    ie_unit_scale_factor,
    interp_spectra,
    param_format,
    quanta_to_energy,
    rgb_to_xw_format,
    srgb_to_xyz,
    spectral_step,
    xw_to_rgb_format,
    xyz_to_srgb,
)


def xyz_color_matching(
    wave_nm: NDArray[np.float64],
    *,
    energy: bool = False,
    quanta: bool = False,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    store = asset_store or AssetStore.default()
    wave = np.asarray(wave_nm, dtype=float)
    if energy and quanta:
        raise ValueError("xyz_color_matching cannot request both energy and quanta data.")
    if quanta:
        _, xyz = store.load_xyz_quanta(wave_nm=wave)
    else:
        _, xyz = store.load_xyz(wave_nm=wave, energy=energy)
    return np.asarray(xyz, dtype=float)


def luminance_from_photons(
    photons: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    energy = quanta_to_energy(np.asarray(photons, dtype=float), np.asarray(wave_nm, dtype=float))
    return luminance_from_energy(energy, wave_nm, asset_store=asset_store)


def luminance_from_energy(
    energy: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    xyz_energy = xyz_color_matching(wave_nm, energy=True, asset_store=asset_store)
    y_bar = xyz_energy[:, 1]
    return 683.0 * np.tensordot(
        np.asarray(energy, dtype=float),
        y_bar * spectral_step(np.asarray(wave_nm, dtype=float)),
        axes=([-1], [0]),
    )


def ie_xyz_from_photons(
    photons: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert photon spectra to XYZ using MATLAB ieXYZFromPhotons() semantics."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    energy = np.asarray(quanta_to_energy(np.asarray(photons, dtype=float), wave), dtype=float)
    xyz_energy = xyz_color_matching(wave, energy=True, asset_store=asset_store)
    return 683.0 * np.tensordot(energy, xyz_energy * spectral_step(wave), axes=([-1], [0]))


def ie_luminance_to_radiance(
    luminance: float,
    this_wave: float,
    *,
    sd: float = 10.0,
    wave: NDArray[np.float64] | None = None,
    asset_store: AssetStore | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Model monochromatic LED radiance from luminance using a Gaussian SPD."""

    center_wave = float(this_wave)
    if center_wave < 350.0 or center_wave > 720.0:
        raise ValueError("this_wave must be between 350 and 720 nm.")
    wave_array = np.asarray(np.arange(300.0, 771.0, 1.0, dtype=float) if wave is None else wave, dtype=float).reshape(-1)
    if wave_array.size == 0:
        raise ValueError("wave must not be empty.")
    sigma = float(sd)
    if sigma <= 0.0:
        raise ValueError("sd must be positive.")

    energy = np.exp(-0.5 * ((wave_array - center_wave) / sigma) ** 2)
    scale = float(luminance) / max(float(luminance_from_energy(energy, wave_array, asset_store=asset_store)), 1e-12)
    return np.asarray(energy * scale, dtype=float).reshape(-1), wave_array


def ie_scotopic_luminance_from_energy(
    energy: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Compute rod-weighted luminance from energy using MATLAB semantics."""

    store = asset_store or AssetStore.default()
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    _, rods = store.load_spectra("rods.mat", wave_nm=wave)
    v_prime = np.asarray(rods, dtype=float).reshape(-1)
    return 1745.0 * np.tensordot(
        np.asarray(energy, dtype=float),
        v_prime * spectral_step(wave),
        axes=([-1], [0]),
    )


def ie_responsivity_convert(
    responsivity: NDArray[np.float64],
    wave_nm: NDArray[np.float64],
    method: str = "e2q",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert responsivities between energy and quanta conventions."""

    response = np.asarray(responsivity, dtype=float)
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    if response.shape[0] != wave.size:
        raise ValueError("Responsivity rows must match the wavelength vector length.")

    peak = float(np.max(response)) if response.size else 0.0
    normalized_method = param_format(method)
    if normalized_method in {"e2q", "energy2quanta", "e2p", "energy2photons"}:
        scale = np.asarray(quanta_to_energy(np.ones((wave.size,), dtype=float), wave), dtype=float).reshape(-1)
        converted = scale[:, np.newaxis] * response
    elif normalized_method in {"q2e", "quanta2energy", "p2e", "photons2energy"}:
        scale = np.asarray(energy_to_quanta(np.ones((wave.size,), dtype=float), wave), dtype=float).reshape(-1)
        converted = scale[:, np.newaxis] * response
    else:
        raise UnsupportedOptionError("ieResponsivityConvert", method)

    if converted.size and peak > 0.0:
        converted_peak = float(np.max(converted))
        if converted_peak > 0.0:
            converted = converted * (peak / converted_peak)
    return np.asarray(converted, dtype=float), np.asarray(scale, dtype=float).reshape(-1)


def y_to_lstar(y_value: NDArray[np.float64] | float, white_y: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Convert luminance Y to CIELAB L* using MATLAB Y2Lstar() semantics."""

    ratio = np.asarray(y_value, dtype=float) / np.maximum(np.asarray(white_y, dtype=float), 1e-12)
    lstar = 116.0 * np.cbrt(ratio) - 16.0
    low = ratio < 0.008856
    if np.any(low):
        lstar = np.asarray(lstar, dtype=float)
        lstar[low] = 903.3 * ratio[low]
    return np.asarray(lstar, dtype=float)


def srgb_to_lrgb(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert nonlinear sRGB values to linear sRGB values."""

    values = np.asarray(rgb, dtype=float)
    if values.size and float(np.max(values)) > 1.0:
        warnings.warn("srgb appears to be outside the (0,1) range", RuntimeWarning, stacklevel=2)
    linear = values.copy()
    high = linear > 0.04045
    linear[~high] = linear[~high] / 12.92
    linear[high] = ((linear[high] + 0.055) / 1.055) ** 2.4
    return linear


def lrgb_to_srgb(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert linear sRGB values to nonlinear framebuffer sRGB values."""

    values = np.asarray(rgb, dtype=float)
    if values.size and (float(np.max(values)) > 1.0 or float(np.min(values)) < 0.0):
        raise ValueError("Linear rgb values must be between 0 and 1.")
    srgb = values.copy()
    high = srgb > 0.0031308
    srgb[~high] = srgb[~high] * 12.92
    srgb[high] = 1.055 * (srgb[high] ** (1.0 / 2.4)) - 0.055
    return srgb


def _xyy_to_xyz(xyy: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(xyy, dtype=float)
    reshaped = values.reshape(-1, 3)
    x = reshaped[:, 0]
    y = reshaped[:, 1]
    big_y = reshaped[:, 2]
    denominator = np.maximum(y, 1e-12)
    xyz = np.column_stack(
        [
            (x * big_y) / denominator,
            big_y,
            ((1.0 - x - y) * big_y) / denominator,
        ]
    )
    return xyz.reshape(values.shape)


def xyy_to_xyz(xyy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIE xyY values to CIE XYZ values."""

    values = np.asarray(xyy, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("xyy must have a trailing dimension of size 3.")
    return np.asarray(_xyy_to_xyz(values), dtype=float)


def ie_lab_to_xyz(lab: NDArray[np.float64], white_point: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIELAB values to XYZ using MATLAB ieLAB2XYZ() semantics."""

    lab_array = np.asarray(lab, dtype=float)
    white = np.asarray(white_point, dtype=float)
    if lab_array.shape[-1] != 3:
        raise ValueError("lab must have a trailing dimension of size 3.")
    if white.shape[-1] != 3:
        raise ValueError("white_point must have a trailing dimension of size 3.")

    delta = 6.0 / 29.0
    fy = (lab_array[..., 0] + 16.0) / 116.0
    fx = (lab_array[..., 1] / 500.0) + fy
    fz = fy - (lab_array[..., 2] / 200.0)

    def _inverse_f(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(values > delta, values**3, 3.0 * delta**2 * (values - (4.0 / 29.0)))

    xyz = np.empty_like(lab_array, dtype=float)
    xyz[..., 0] = _inverse_f(fx) * white[..., 0]
    xyz[..., 1] = _inverse_f(fy) * white[..., 1]
    xyz[..., 2] = _inverse_f(fz) * white[..., 2]
    return xyz


def _stockman_xyz_matrices(
    *,
    asset_store: AssetStore | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    store = asset_store or AssetStore.default()
    wave = np.arange(400.0, 701.0, 5.0, dtype=float)
    xyz = np.asarray(xyz_color_matching(wave, asset_store=store), dtype=float)
    _, lms = store.load_spectra("stockman.mat", wave_nm=wave)
    lms_array = np.asarray(lms, dtype=float)
    xyz_to_lms_matrix, _, _, _ = np.linalg.lstsq(xyz, lms_array, rcond=None)
    lms_to_xyz_matrix, _, _, _ = np.linalg.lstsq(lms_array, xyz, rcond=None)
    return np.asarray(xyz_to_lms_matrix, dtype=float), np.asarray(lms_to_xyz_matrix, dtype=float)


def _apply_tristimulus_transform(values: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    transform = np.asarray(matrix, dtype=float)
    if array.ndim >= 1 and array.shape[-1] == 3:
        reshaped = array.reshape(-1, 3)
        return np.asarray(reshaped @ transform, dtype=float).reshape(array.shape)
    if array.ndim == 2 and array.shape[0] == 3 and array.shape[1] != 3:
        return np.asarray(transform.T @ array, dtype=float)
    raise ValueError("Input must have a trailing dimension of size 3 or be a 3xN array.")


def xyz_to_lms(
    xyz: NDArray[np.float64],
    cb_type: int = 0,
    extrap_val: Any = 0.0,
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert XYZ values to Stockman LMS, including MATLAB's Brettel dichromat path."""

    cb = int(cb_type)
    store = asset_store or AssetStore.default()
    xyz_to_lms_matrix, _ = _stockman_xyz_matrices(asset_store=store)
    lms = np.asarray(_apply_tristimulus_transform(np.asarray(xyz, dtype=float), xyz_to_lms_matrix), dtype=float)
    if cb == 0:
        return lms
    if cb > 0:
        white_xyz = np.asarray(extrap_val, dtype=float).reshape(-1)
        if white_xyz.size < 3:
            raise ValueError("xyz2lms requires whiteXYZ when cbType > 0.")
        anchor_e = np.asarray(white_xyz[:3] @ xyz_to_lms_matrix, dtype=float).reshape(3)
        _, anchor_values = store.load_spectra(
            "stockman.mat",
            wave_nm=np.array([475.0, 485.0, 575.0, 660.0], dtype=float),
        )
        anchors = np.asarray(anchor_values, dtype=float).reshape(4, 3)
        anchor_475, anchor_485, anchor_575, anchor_660 = anchors

        if lms.ndim >= 1 and lms.shape[-1] == 3:
            L = np.asarray(lms[..., 0], dtype=float).copy()
            M = np.asarray(lms[..., 1], dtype=float).copy()
            S = np.asarray(lms[..., 2], dtype=float).copy()
            layout = "last"
        elif lms.ndim == 2 and lms.shape[0] == 3 and lms.shape[1] != 3:
            L = np.asarray(lms[0, :], dtype=float).copy()
            M = np.asarray(lms[1, :], dtype=float).copy()
            S = np.asarray(lms[2, :], dtype=float).copy()
            layout = "first"
        else:
            raise ValueError("xyz2lms expects an RGB-format image or a 3xN array.")

        def _safe_ratio(numerator: NDArray[np.float64], denominator: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.inf, dtype=float),
                where=np.abs(denominator) > np.finfo(float).eps,
            )

        if cb in {1, 2}:
            a1 = anchor_e[1] * anchor_575[2] - anchor_e[2] * anchor_575[1]
            b1 = anchor_e[2] * anchor_575[0] - anchor_e[0] * anchor_575[2]
            c1 = anchor_e[0] * anchor_575[1] - anchor_e[1] * anchor_575[0]
            a2 = anchor_e[1] * anchor_475[2] - anchor_e[2] * anchor_475[1]
            b2 = anchor_e[2] * anchor_475[0] - anchor_e[0] * anchor_475[2]
            c2 = anchor_e[0] * anchor_475[1] - anchor_e[1] * anchor_475[0]
            if cb == 1:
                inflection = anchor_e[2] / anchor_e[1]
                lst = _safe_ratio(S, M) < inflection
                L[lst] = -(b1 * M[lst] + c1 * S[lst]) / a1
                L[~lst] = -(b2 * M[~lst] + c2 * S[~lst]) / a2
            else:
                inflection = anchor_e[2] / anchor_e[0]
                lst = _safe_ratio(S, L) < inflection
                M[lst] = -(a1 * L[lst] + c1 * S[lst]) / b1
                M[~lst] = -(a2 * L[~lst] + c2 * S[~lst]) / b2
        elif cb == 3:
            a1 = anchor_e[1] * anchor_660[2] - anchor_e[2] * anchor_660[1]
            b1 = anchor_e[2] * anchor_660[0] - anchor_e[0] * anchor_660[2]
            c1 = anchor_e[0] * anchor_660[1] - anchor_e[1] * anchor_660[0]
            a2 = anchor_e[1] * anchor_485[2] - anchor_e[2] * anchor_485[1]
            b2 = anchor_e[2] * anchor_485[0] - anchor_e[0] * anchor_485[2]
            c2 = anchor_e[0] * anchor_485[1] - anchor_e[1] * anchor_485[0]
            inflection = anchor_e[1] / anchor_e[0]
            lst = _safe_ratio(M, L) < inflection
            S[lst] = -(a1 * L[lst] + b1 * M[lst]) / c1
            S[~lst] = -(a2 * L[~lst] + b2 * M[~lst]) / c2
        else:
            raise UnsupportedOptionError("xyz2lms", f"cbType={cb_type}")

        updated = np.asarray(lms, dtype=float).copy()
        if layout == "last":
            updated[..., 0] = L
            updated[..., 1] = M
            updated[..., 2] = S
        else:
            updated[0, :] = L
            updated[1, :] = M
            updated[2, :] = S
        return updated
    channel_index = abs(cb) - 1
    if channel_index not in {0, 1, 2}:
        raise UnsupportedOptionError("xyz2lms", f"cbType={cb_type}")
    lms = np.asarray(lms, dtype=float).copy()
    if lms.ndim >= 1 and lms.shape[-1] == 3:
        lms[..., channel_index] = float(extrap_val)
        return lms
    lms[channel_index, :] = float(extrap_val)
    return lms


def lms_to_xyz(
    lms: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert Stockman LMS values to XYZ using the direct MATLAB lms2xyz() path."""

    _, lms_to_xyz_matrix = _stockman_xyz_matrices(asset_store=asset_store)
    return np.asarray(_apply_tristimulus_transform(np.asarray(lms, dtype=float), lms_to_xyz_matrix), dtype=float)


def lms_to_srgb(
    lms: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert LMS image data to sRGB for visualization."""

    from .utils import xyz_to_srgb

    xyz = np.asarray(lms_to_xyz(lms, asset_store=asset_store), dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("lms2srgb expects an RGB-format LMS image.")
    return np.asarray(xyz_to_srgb(xyz), dtype=float)


def srgb_parameters(value: str = "all") -> NDArray[np.float64]:
    """Return sRGB display parameters using MATLAB srgbParameters() semantics."""

    params = np.array(
        [
            [0.6400, 0.3000, 0.1500, 0.3127],
            [0.3300, 0.6000, 0.0600, 0.3290],
            [0.2126, 0.7152, 0.0722, 1.0000],
        ],
        dtype=float,
    )
    key = param_format(value)
    if key == "all":
        return params.copy()
    if key == "chromaticity":
        return params[:2, :3].copy()
    if key == "luminance":
        return params[2, :3].copy()
    if key == "xyywhite":
        return params[:, 3].copy()
    if key == "xyzwhite":
        return _xyy_to_xyz(params[:, 3]).reshape(3)
    raise UnsupportedOptionError("srgbParameters", value)


def adobergb_parameters(value: str = "all") -> NDArray[np.float64]:
    """Return Adobe RGB display parameters using MATLAB adobergbParameters() semantics."""

    params = np.array(
        [
            [0.64, 0.21, 0.15, 0.3127],
            [0.33, 0.71, 0.06, 0.3290],
            [47.5744, 100.3776, 12.0320, 160.0],
        ],
        dtype=float,
    )
    key = param_format(value)
    if key == "all":
        return params.copy()
    if key == "chromaticity":
        return params[:2, :3].copy()
    if key == "luminance":
        return params[2, :3].copy()
    if key == "xyywhite":
        return params[:, 3].copy()
    if key == "xyzwhite":
        return _xyy_to_xyz(params[:, 3]).reshape(3)
    if key == "xyzblack":
        return np.array([0.5282, 0.5557, 0.6052], dtype=float)
    raise UnsupportedOptionError("adobergbParameters", value)


def daylight(
    wave_nm: NDArray[np.float64],
    cct_k: float | NDArray[np.float64] = 6500.0,
    units: str = "energy",
    *,
    return_xyz: bool = False,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate CIE daylight spectra from correlated color temperature."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    cct = np.asarray(cct_k, dtype=float).reshape(-1)
    if wave.size == 0:
        raise ValueError("wave_nm must not be empty.")
    if cct.size == 0:
        raise ValueError("cct_k must not be empty.")
    if np.any((cct < 4000.0) | (cct >= 30000.0)):
        raise ValueError("daylight supports 4000 K <= cct_k < 30000 K.")

    lower_mask = (cct >= 4000.0) & (cct < 7000.0)
    upper_mask = cct >= 7000.0
    xdt = np.empty((2, cct.size), dtype=float)
    xdt[0, :] = (-4.6070e9 / cct**3) + (2.9678e6 / cct**2) + (0.09911e3 / cct) + 0.244063
    xdt[1, :] = (-2.0064e9 / cct**3) + (1.9018e6 / cct**2) + (0.24748e3 / cct) + 0.237040
    xd = lower_mask.astype(float) * xdt[0, :] + upper_mask.astype(float) * xdt[1, :]
    yd = (-3.0 * xd**2) + (2.87 * xd) - 0.275

    denominator = 0.0241 + (0.2562 * xd) - (0.7341 * yd)
    weights = np.empty((2, cct.size), dtype=float)
    weights[0, :] = (-1.3515 - (1.7703 * xd) + (5.9114 * yd)) / denominator
    weights[1, :] = (0.03 - (31.4424 * xd) + (30.0717 * yd)) / denominator

    store = asset_store or AssetStore.default()
    _, day_basis = store.load_spectra("cieDaylightBasis.mat", wave_nm=wave)
    basis = np.asarray(day_basis, dtype=float)
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    energy = basis[:, [0]] + basis[:, 1:3] @ weights

    normalized_units = param_format(units)
    if normalized_units in {"photons", "quanta"}:
        spectra = np.asarray(energy_to_quanta(energy, wave), dtype=float)
        first_luminance = float(luminance_from_photons(spectra[:, 0], wave, asset_store=store))
    elif normalized_units in {"energy", "watts"}:
        spectra = energy
        first_luminance = float(luminance_from_energy(spectra[:, 0], wave, asset_store=store))
    else:
        raise UnsupportedOptionError("daylight", units)
    spectra = (spectra / max(first_luminance, 1e-12)) * 100.0

    if not return_xyz:
        return spectra[:, 0] if cct.size == 1 else spectra

    xyz_energy = xyz_color_matching(wave, energy=True, asset_store=store)
    xyz = 683.0 * np.tensordot(
        np.asarray(spectra, dtype=float).T,
        xyz_energy * spectral_step(wave),
        axes=([-1], [0]),
    )
    if cct.size == 1:
        return spectra[:, 0], np.asarray(xyz, dtype=float).reshape(3)
    return spectra, np.asarray(xyz, dtype=float)


def cct_to_sun(
    wave_nm: NDArray[np.float64] | None,
    cct_k: float | NDArray[np.float64],
    units: str = "energy",
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Legacy MATLAB ``cct2sun`` compatibility wrapper."""

    wave = np.arange(400.0, 701.0, 1.0, dtype=float) if wave_nm is None else np.asarray(wave_nm, dtype=float).reshape(-1)
    return np.asarray(daylight(wave, cct_k, units, asset_store=asset_store), dtype=float)


def ie_ctemp_to_srgb(
    c_temp: float,
    *,
    wave: NDArray[np.float64] | None = None,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Convert a blackbody color temperature into a headless sRGB triplet."""

    wave_array = np.asarray(np.arange(400.0, 701.0, 10.0, dtype=float) if wave is None else wave, dtype=float).reshape(-1)
    energy = np.asarray(blackbody(wave_array, float(c_temp), kind="energy"), dtype=float).reshape(1, -1)
    from .metrics import xyz_from_energy

    xyz = np.asarray(xyz_from_energy(energy, wave_array, asset_store=asset_store), dtype=float).reshape(1, 1, 3)
    return np.asarray(xyz_to_srgb(xyz), dtype=float).reshape(3)


def ie_circle_points(rad_spacing: float = 2.0 * np.pi / 60.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return samples on the unit circle using the MATLAB ``ieCirclePoints`` contract."""

    spacing = float(rad_spacing)
    if spacing <= 0.0:
        raise ValueError("rad_spacing must be positive.")
    theta = np.arange(0.0, (2.0 * np.pi) + (spacing * 0.5), spacing, dtype=float)
    return np.cos(theta), np.sin(theta)


def mk_inv_gamma_table(g_table: NDArray[np.float64], num_entries: int | None = None) -> NDArray[np.float64]:
    """Compute a MATLAB-style inverse gamma lookup table."""

    gamma_table = np.asarray(g_table, dtype=float)
    if gamma_table.ndim == 1:
        gamma_table = gamma_table.reshape(-1, 1)
    if gamma_table.ndim != 2 or gamma_table.shape[0] == 0:
        raise ValueError("g_table must be a non-empty 1D or 2D gamma table.")

    entry_count = int(4 * gamma_table.shape[0] if num_entries is None else num_entries)
    if entry_count <= 0:
        raise ValueError("num_entries must be positive.")

    result = np.zeros((entry_count, gamma_table.shape[1]), dtype=float)
    target_axis = np.arange(entry_count, dtype=float) / max(entry_count - 1, 1)

    for column in range(gamma_table.shape[1]):
        this_table = np.asarray(gamma_table[:, column], dtype=float).reshape(-1)
        if np.any(np.diff(this_table) <= 0.0):
            this_table = np.sort(this_table)
            positive_locs = np.where(np.diff(this_table) > 0.0)[0] + 1
            pos_locs = np.concatenate(([0], positive_locs)).astype(float)
            monotone_table = this_table[pos_locs.astype(int)]
        else:
            monotone_table = this_table
            pos_locs = np.arange(this_table.size, dtype=float)
        result[:, column] = np.interp(target_axis, monotone_table, pos_locs)

    return result


def _copy_iset_like_object(object_in: Any) -> Any:
    if hasattr(object_in, "clone"):
        return object_in.clone()
    return copy.deepcopy(object_in)


def _object_fields(object_in: Any) -> dict[str, Any]:
    if hasattr(object_in, "fields"):
        return object_in.fields
    if isinstance(object_in, dict):
        return object_in
    raise TypeError("initDefaultSpectrum requires an ISET object with `.fields` or a dictionary.")


def init_default_spectrum(object_in: Any, spectral_type: str = "hyperspectral", wave: Any | None = None) -> Any:
    """Attach a default wavelength spectrum to an ISET-style object."""

    if object_in is None:
        raise ValueError("Object required.")
    object_out = _copy_iset_like_object(object_in)
    fields = _object_fields(object_out)
    spectrum = dict(fields.get("spectrum", {}))
    normalized = param_format(spectral_type or "hyperspectral")

    if normalized in {"spectral", "multispectral", "hyperspectral"}:
        wave_values = np.arange(400.0, 701.0, 10.0, dtype=float)
    elif normalized == "monochrome":
        wave_values = np.array([550.0], dtype=float)
    elif normalized == "custom":
        if wave is None:
            raise ValueError("wave required for custom spectrum")
        wave_values = np.asarray(wave, dtype=float).reshape(-1)
    else:
        raise UnsupportedOptionError("initDefaultSpectrum", spectral_type)

    spectrum["wave"] = np.asarray(wave_values, dtype=float).reshape(-1)
    fields["spectrum"] = spectrum
    fields["wave"] = np.asarray(wave_values, dtype=float).reshape(-1)
    return object_out


def ie_cov_ellipsoid(
    xy_data: Any,
    n_sd: float = 1.0,
    h: Any | None = None,
    n_samp: int = 20,
) -> tuple[NDArray[np.float64], None, dict[str, NDArray[np.float64]]]:
    """Calculate a covariance ellipse or ellipsoid without opening a figure."""

    del h
    data = np.asarray(xy_data, dtype=float)
    if data.ndim != 2 or data.shape[1] not in {2, 3}:
        raise ValueError("ieCovEllipsoid requires an Nx2 or Nx3 data matrix.")
    dimensionality = int(data.shape[1])
    if dimensionality == 2:
        x_circle, y_circle = ie_circle_points(2.0 * np.pi * 0.01)
        u_vec = np.column_stack((x_circle, y_circle))
    else:
        phi = np.linspace(0.0, np.pi, int(n_samp) + 1, dtype=float)
        theta = np.linspace(0.0, 2.0 * np.pi, int(n_samp) + 1, dtype=float)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x_sphere = np.cos(theta_grid) * np.sin(phi_grid)
        y_sphere = np.sin(theta_grid) * np.sin(phi_grid)
        z_sphere = np.cos(phi_grid)
        u_vec = np.column_stack((x_sphere.reshape(-1), y_sphere.reshape(-1), z_sphere.reshape(-1)))

    covariance = np.asarray(np.cov(data, rowvar=False), dtype=float)
    covariance_inv = np.linalg.pinv(covariance)
    mn = np.asarray(np.mean(data, axis=0), dtype=float).reshape(1, dimensionality)
    lengths = np.einsum("ij,jk,ik->i", u_vec, covariance_inv, u_vec)
    lengths = np.maximum(lengths, 1.0e-12)
    e_vec = float(n_sd) * (u_vec / np.sqrt(lengths)[:, np.newaxis]) + mn

    payload: dict[str, NDArray[np.float64]] = {
        "pts": np.asarray(data, dtype=float),
        "covariance": covariance,
    }
    if dimensionality == 2:
        payload["crv"] = np.asarray(e_vec, dtype=float)
    else:
        payload["surf"] = np.asarray(e_vec, dtype=float)
    return np.asarray(e_vec, dtype=float), None, payload


def ie_spectra_sphere(
    wave: Any | None = None,
    spectrum_e: Any | None = None,
    n: int = 8,
    s_basis: Any | None = None,
    s_factor: float = 0.05,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate spectra whose XYZ values lie on a sphere around a base spectrum."""

    store = asset_store or AssetStore.default()
    wave_array = (
        np.arange(400.0, 701.0, 10.0, dtype=float)
        if wave is None
        else np.asarray(wave, dtype=float).reshape(-1)
    )
    spectrum = (
        np.zeros(wave_array.shape, dtype=float)
        if spectrum_e is None
        else np.asarray(spectrum_e, dtype=float).reshape(-1)
    )
    if spectrum.size != wave_array.size:
        raise ValueError("spectrumE must match the wavelength vector length.")
    if s_basis is None:
        _, basis_array = store.load_spectra("cieDaylightBasis.mat", wave_nm=wave_array)
    elif isinstance(s_basis, str):
        _, basis_array = store.load_spectra(s_basis, wave_nm=wave_array)
    else:
        basis_array = np.asarray(s_basis, dtype=float)
    basis_array = np.asarray(basis_array, dtype=float)
    if basis_array.shape[0] != wave_array.size:
        raise ValueError("sBasis rows must match the wavelength vector length.")

    theta = np.linspace(0.0, 2.0 * np.pi, int(n) + 1, dtype=float)
    phi = np.linspace(0.0, np.pi, int(n) + 1, dtype=float)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    d_xyz = np.column_stack(
        (
            (np.cos(theta_grid) * np.sin(phi_grid)).reshape(-1),
            (np.sin(theta_grid) * np.sin(phi_grid)).reshape(-1),
            np.cos(phi_grid).reshape(-1),
        )
    )

    cie_xyz = xyz_color_matching(wave_array, energy=True, asset_store=store)
    transform = np.linalg.lstsq(np.asarray(cie_xyz.T @ basis_array, dtype=float), d_xyz.T, rcond=None)[0]
    spectra_s = np.asarray(basis_array @ transform, dtype=float)
    basis_norm = float(np.linalg.norm(spectra_s[:, 0])) if spectra_s.shape[1] else 0.0
    target_norm = float(np.linalg.norm(spectrum))
    scale = 0.0 if basis_norm <= 0.0 else float(s_factor) * target_norm / basis_norm
    spectra_s = spectra_s * scale + spectrum[:, np.newaxis]

    xyz_energy = np.asarray(cie_xyz * spectral_step(wave_array), dtype=float)
    xyz = 683.0 * np.asarray(spectra_s.T @ xyz_energy, dtype=float)
    xyz0 = 683.0 * np.asarray(spectrum.reshape(1, -1) @ xyz_energy, dtype=float).reshape(-1)
    return np.asarray(spectra_s, dtype=float), np.asarray(xyz, dtype=float), xyz0, basis_array


def xyz_to_vsnr(
    roi_xyz: Any,
    white_pt_xyz: Any,
    params: Any | None = None,
) -> float:
    """Calculate visual SNR from an XYZ image via the SCIELAB path."""

    from .scielab import sc_compute_scielab, sc_params

    roi = np.asarray(roi_xyz, dtype=float)
    if roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError("xyz2vSNR requires an XYZ image with shape rows x cols x 3.")
    white = np.asarray(white_pt_xyz, dtype=float).reshape(-1)
    if white.size != 3:
        raise ValueError("xyz2vSNR requires a 3-element white point.")
    params_dict = sc_params() if params is None else params
    s_lab, _ = sc_compute_scielab(roi, white, params_dict)

    rows, cols, _ = s_lab.shape
    mid_rows = max(int(np.round(0.8 * rows)), 1)
    mid_cols = max(int(np.round(0.8 * cols)), 1)
    row0 = max((rows - mid_rows) // 2, 0)
    col0 = max((cols - mid_cols) // 2, 0)
    cropped = s_lab[row0 : row0 + mid_rows, col0 : col0 + mid_cols, :]

    l_var = float(np.var(cropped[:, :, 0], dtype=float))
    a_var = float(np.var(cropped[:, :, 1], dtype=float))
    b_var = float(np.var(cropped[:, :, 2], dtype=float))
    return float(1.0 / np.sqrt(max(l_var + a_var + b_var, 1.0e-12)))


def _surface_reflectances(
    surfaces: str,
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore,
) -> NDArray[np.float64]:
    normalized = param_format(surfaces)
    if normalized in {"mcc", "mccoptimized"}:
        _, reflectances = asset_store.load_reflectances("macbethChart.mat", wave_nm=np.asarray(wave_nm, dtype=float))
        return np.asarray(reflectances, dtype=float)
    if normalized in {"esser", "esseroptimized"}:
        data = asset_store.load_mat("data/surfaces/charts/esser/reflectance/esserChart.mat")
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        reflectances = np.asarray(data["data"], dtype=float)
        return np.asarray(
            interp_spectra(wavelengths, reflectances, np.asarray(wave_nm, dtype=float)),
            dtype=float,
        )
    raise UnsupportedOptionError("ieColorTransform", surfaces)


def _target_qe(
    target_space: str,
    wave_nm: NDArray[np.float64],
    *,
    asset_store: AssetStore,
) -> NDArray[np.float64]:
    normalized = param_format(target_space)
    if normalized == "xyz":
        return xyz_color_matching(wave_nm, quanta=True, asset_store=asset_store)
    if normalized == "stockman":
        return np.asarray(
            ie_read_color_filter(
                np.asarray(wave_nm, dtype=float),
                "data/human/stockmanQuanta.mat",
                asset_store=asset_store,
            )[0],
            dtype=float,
        )
    raise UnsupportedOptionError("ieColorTransform", target_space)


def ie_color_transform(
    sensor: Any,
    target_space: str = "XYZ",
    illuminant: str | NDArray[np.float64] = "D65",
    surface: str | NDArray[np.float64] = "multisurface",
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Mirror MATLAB ieColorTransform() for the supported headless target spaces."""

    from .ip import image_sensor_transform
    from .sensor import sensor_get

    store = asset_store or AssetStore.default()
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    if sensor_qe.ndim == 1:
        sensor_qe = sensor_qe.reshape(-1, 1)

    normalized_target = param_format(target_space)
    if normalized_target == "sensor":
        return np.eye(sensor_qe.shape[1], dtype=float)

    target_qe = _target_qe(normalized_target, wave, asset_store=store)
    return np.asarray(
        image_sensor_transform(
            sensor_qe,
            target_qe,
            illuminant,
            wave,
            surface,
            asset_store=store,
        ),
        dtype=float,
    )


def sensor_to_target_matrix(
    wave_nm: NDArray[np.float64],
    filter_spectra: NDArray[np.float64],
    *,
    target_space: str = "xyz",
    illuminant: str = "D65",
    surfaces: str = "mcc",
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    store = asset_store or AssetStore.default()
    wave = np.asarray(wave_nm, dtype=float)
    sensor_qe = np.asarray(filter_spectra, dtype=float)
    target_qe = _target_qe(target_space, wave, asset_store=store)
    _, illuminant_energy = store.load_illuminant(illuminant, wave_nm=wave)
    illuminant_quanta = energy_to_quanta(np.asarray(illuminant_energy, dtype=float), wave)
    reflectances = _surface_reflectances(surfaces, wave, asset_store=store)
    weighted_surfaces = reflectances * illuminant_quanta.reshape(-1, 1)
    sensor_response = weighted_surfaces.T @ sensor_qe
    target_response = weighted_surfaces.T @ target_qe
    matrix, _, _, _ = np.linalg.lstsq(sensor_response, target_response, rcond=None)
    return matrix


def sensor_to_xyz_matrix(
    wave_nm: NDArray[np.float64],
    filter_spectra: NDArray[np.float64],
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    return sensor_to_target_matrix(
        wave_nm,
        filter_spectra,
        target_space="xyz",
        illuminant="D65",
        surfaces="mcc",
        asset_store=asset_store,
    )


def internal_to_display_matrix(
    wave_nm: NDArray[np.float64],
    display_spd: NDArray[np.float64],
    *,
    internal_cs: str = "xyz",
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    normalized = param_format(internal_cs)
    if normalized != "xyz":
        raise UnsupportedOptionError("displayRender", internal_cs)
    internal_cmf = xyz_color_matching(np.asarray(wave_nm, dtype=float), energy=False, asset_store=asset_store)
    return np.linalg.inv(np.asarray(display_spd, dtype=float).T @ internal_cmf)


def xyz_from_energy(
    energy: Any,
    wave_nm: Any,
    *,
    asset_store: AssetStore | None = None,
) -> NDArray[np.float64]:
    """Lazy xyz_from_energy() wrapper to avoid importing metrics at module load time."""

    from .metrics import xyz_from_energy as _xyz_from_energy

    return _xyz_from_energy(energy, wave_nm, asset_store=asset_store)


def ie_xyz_to_lab(xyz: Any, white_point: Any) -> NDArray[np.float64]:
    """Lazy ieXYZ2LAB() wrapper to avoid importing metrics at module load time."""

    from .metrics import xyz_to_lab as _xyz_to_lab

    return _xyz_to_lab(xyz, white_point)


def xyz_to_luv(xyz: Any, white_point: Any) -> NDArray[np.float64]:
    """Lazy xyz2luv() wrapper to avoid importing metrics at module load time."""

    from .metrics import xyz_to_luv as _xyz_to_luv

    return _xyz_to_luv(xyz, white_point)


def xyz_to_uv(xyz: Any) -> NDArray[np.float64]:
    """Lazy xyz2uv() wrapper to avoid importing metrics at module load time."""

    from .metrics import xyz_to_uv as _xyz_to_uv

    return _xyz_to_uv(xyz)


def spd_to_cct(
    wave_nm: Any,
    spd: Any,
    *,
    asset_store: AssetStore | None = None,
) -> float | NDArray[np.float64]:
    """Lazy spd2cct() wrapper to avoid importing metrics at module load time."""

    from .metrics import spd_to_cct as _spd_to_cct

    return _spd_to_cct(wave_nm, spd, asset_store=asset_store)


def srgb_to_color_temp(
    rgb: Any,
    method: str = "bright",
    *args: Any,
    return_table: bool = False,
    asset_store: AssetStore | None = None,
) -> float | tuple[float, NDArray[np.float64]]:
    """Lazy srgb2colortemp() wrapper to avoid importing metrics at module load time."""

    from .metrics import srgb_to_color_temp as _srgb_to_color_temp

    return _srgb_to_color_temp(
        rgb,
        method,
        *args,
        return_table=return_table,
        asset_store=asset_store,
    )


def chromaticity_xy(xyz: Any) -> NDArray[np.float64]:
    """Lazy chromaticity() wrapper to avoid importing metrics at module load time."""

    from .metrics import chromaticity_xy as _chromaticity_xy

    return _chromaticity_xy(xyz)


def color_transform_matrix(matrix_type: str, space_type: int = 10) -> NDArray[np.float64]:
    """Lazy colorTransformMatrix() wrapper to avoid importing scielab at module load time."""

    from .scielab import color_transform_matrix as _color_transform_matrix

    return _color_transform_matrix(matrix_type, space_type)


def color_block_matrix(wave_nm: Any, extrap_val: float = 0.0) -> NDArray[np.float64]:
    """Lazy colorBlockMatrix() wrapper to avoid importing sensor at module load time."""

    from .sensor import color_block_matrix as _color_block_matrix

    return _color_block_matrix(wave_nm, extrap_val)


adobergbParameters = adobergb_parameters
cct2sun = cct_to_sun
Energy2Quanta = energy_to_quanta
ieCTemp2SRGB = ie_ctemp_to_srgb
ieCirclePoints = ie_circle_points
ieCovEllipsoid = ie_cov_ellipsoid
ieLAB2XYZ = ie_lab_to_xyz
ieLuminance2Radiance = ie_luminance_to_radiance
ieLuminanceFromEnergy = luminance_from_energy
ieLuminanceFromPhotons = luminance_from_photons
ieResponsivityConvert = ie_responsivity_convert
ieScotopicLuminanceFromEnergy = ie_scotopic_luminance_from_energy
ieSpectraSphere = ie_spectra_sphere
ieUnitScaleFactor = ie_unit_scale_factor
ieXYZ2LAB = ie_xyz_to_lab
ieXYZFromEnergy = xyz_from_energy
ieXYZFromPhotons = ie_xyz_from_photons
initDefaultSpectrum = init_default_spectrum
lms2srgb = lms_to_srgb
lms2xyz = lms_to_xyz
lrgb2srgb = lrgb_to_srgb
mkInvGammaTable = mk_inv_gamma_table
Quanta2Energy = quanta_to_energy
RGB2XWFormat = rgb_to_xw_format
chromaticity = chromaticity_xy
chromaticityXY = chromaticity_xy
colorBlockMatrix = color_block_matrix
colorTransformMatrix = color_transform_matrix
spd2cct = spd_to_cct
srgb2colortemp = srgb_to_color_temp
srgb2lrgb = srgb_to_lrgb
srgb2xyz = srgb_to_xyz
srgbParameters = srgb_parameters
XW2RGBFormat = xw_to_rgb_format
Y2Lstar = y_to_lstar
xyy2xyz = xyy_to_xyz
xyz2lms = xyz_to_lms
xyz2luv = xyz_to_luv
xyz2srgb = xyz_to_srgb
xyz2uv = xyz_to_uv
xyz2vSNR = xyz_to_vsnr
ieColorTransform = ie_color_transform
