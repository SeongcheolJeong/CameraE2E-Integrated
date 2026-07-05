"""Optical image creation and computation."""

from __future__ import annotations

from math import factorial
from pathlib import Path
import re
from typing import Any, Mapping

import imageio.v3 as iio
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import map_coordinates, rotate, uniform_filter
from scipy.signal import fftconvolve
from scipy.special import jv
from skimage.draw import polygon2mask

from .assets import AssetStore, ensure_upstream_snapshot
from .exceptions import UnsupportedOptionError
from .illuminant import illuminant_get
from .metrics import chromaticity_xy, xyz_from_energy
from .scene import scene_get
from .session import track_session_object
from .types import BaseISETObject, OpticalImage, Scene, SessionContext
from .utils import (
    DEFAULT_WAVE,
    apply_channelwise_gaussian,
    gaussian_sigma_pixels,
    interp_spectra,
    param_format,
    quanta_to_energy,
    spectral_step,
    split_prefixed_parameter,
    unit_frequency_list,
)

DEFAULT_FOCAL_LENGTH_M = 0.003862755099228
DEFAULT_WVF_FOCAL_LENGTH_M = 0.0171883
DEFAULT_WVF_MEASURED_PUPIL_MM = 8.0
DEFAULT_WVF_MEASURED_WAVELENGTH_NM = 550.0
DEFAULT_WVF_SPATIAL_SAMPLES = 201
DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM = 16.212
DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM = 3.0
DEFAULT_CAMERA_WVF_CALC_PUPIL_DIAMETER_MM = 9.6569e-01
DEFAULT_RAYTRACE_ANGLE_STEP_DEG = 10.0
DEFAULT_WVF_APERTURE_PARAMS = {
    "shape": "polygon",
    "nsides": 5,
    "aspectratio": np.array([1.0, 1.0], dtype=float),
    "dotmean": 10.0,
    "dotsd": 5.0,
    "dotopacity": 0.5,
    "dotradius": 5.0,
    "linemean": 10.0,
    "linesd": 5.0,
    "lineopacity": 0.5,
    "linewidth": 2.0,
    "segmentlength": 600.0,
    "texfile": None,
    "imagerotate": None,
    "seed": None,
}
_SPATIAL_UNIT_SCALE = {
    "meters": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "millimeters": 1e3,
    "millimeter": 1e3,
    "mm": 1e3,
    "microns": 1e6,
    "micron": 1e6,
    "um": 1e6,
}


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _is_empty_dispatch_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip() == ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value).size == 0
    return False


def _mat_to_native(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "_fieldnames"):
        return {str(name): _mat_to_native(getattr(value, name)) for name in getattr(value, "_fieldnames", [])}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return _mat_to_native(value.item())
            flattened = [_mat_to_native(item) for item in value.reshape(-1)]
            return np.asarray(flattened, dtype=object).reshape(value.shape)
        return np.asarray(value)
    return value


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def airy_disk(
    this_wave: float,
    f_number: float | None,
    *args: Any,
    return_image: bool = False,
) -> float | tuple[float, dict[str, np.ndarray] | None]:
    """Return the Airy disk radius or diameter in the requested units.

    This follows the upstream MATLAB ``airyDisk(...)`` semantics:
    wavelengths may be provided in nm or meters, spatial units default to
    meters, angular units use the pupil-diameter formula, and the optional
    image payload is produced from the diffraction-limited WVF PSF path.
    """

    units = "m"
    diameter = False
    pupil_diameter_m = 3e-3
    if len(args) % 2 != 0:
        raise ValueError("airy_disk optional arguments must be key/value pairs.")
    for index in range(0, len(args), 2):
        key = param_format(args[index])
        value = args[index + 1]
        if key == "units":
            units = str(value)
        elif key == "diameter":
            diameter = bool(value)
        elif key == "pupildiameter":
            pupil_diameter_m = float(np.asarray(value, dtype=float).reshape(-1)[0])
        else:
            raise UnsupportedOptionError("airyDisk", str(args[index]))

    normalized_unit = param_format(units)
    wave_m = float(this_wave)
    if wave_m > 200.0:
        wave_m *= 1e-9

    f_number_array = np.asarray(f_number if f_number is not None else [], dtype=float)
    f_number_is_empty = f_number is None or f_number_array.size == 0
    if f_number_is_empty and normalized_unit not in {"deg", "rad"}:
        raise ValueError("Airy disk spatial units require a finite f-number.")

    if normalized_unit in {"m", "meter", "meters", "mm", "millimeter", "millimeters", "um", "micron", "microns"}:
        radius = 1.22 * float(f_number_array.reshape(-1)[0]) * wave_m
        radius *= _spatial_unit_scale(normalized_unit)
    elif normalized_unit == "deg":
        radius = float(np.degrees(np.arcsin(1.22 * wave_m / max(pupil_diameter_m, 1e-12))))
    elif normalized_unit == "rad":
        radius = float(np.arcsin(1.22 * wave_m / max(pupil_diameter_m, 1e-12)))
    else:
        raise UnsupportedOptionError("airyDisk", f"units {units}")

    if diameter:
        radius *= 2.0

    if not return_image:
        return float(radius)

    if f_number_is_empty:
        return float(radius), None

    wave_nm = wave_m * 1e9
    wvf = wvf_create(wave=np.array([wave_nm], dtype=float))
    focal_length_mm = float(wvf_get(wvf, "focal length", "mm"))
    wvf = wvf_set(wvf, "calc pupil diameter", focal_length_mm / float(f_number_array.reshape(-1)[0]), "mm")
    wvf = wvf_compute(wvf)
    psf = np.asarray(wvf_get(wvf, "psf", wave_nm), dtype=float)
    axis = np.asarray(wvf_get(wvf, "psf spatial samples", "um", wave_nm), dtype=float)
    return float(radius), {"data": psf, "x": axis.copy(), "y": axis.copy()}


def _coerce_optics_struct(optics: OpticalImage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(optics, OpticalImage):
        return dict(optics.fields.get("optics", {}))
    if isinstance(optics, dict):
        return dict(optics)
    raise TypeError("optics must be an OpticalImage or optics dictionary.")


def optics_coc(
    optics: OpticalImage | dict[str, Any],
    o_dist: float,
    *args: Any,
    unit: str = "m",
    xdist: Any | None = None,
    nsamples: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the thin-lens circle of confusion for an in-focus object distance."""

    options = {
        "unit": unit,
        "xdist": xdist,
        "nsamples": nsamples,
    }
    if len(args) % 2 != 0:
        raise ValueError("optics_coc optional arguments must be key/value pairs.")
    for index in range(0, len(args), 2):
        key = param_format(args[index]).replace("_", "").replace("-", "")
        value = args[index + 1]
        if key == "unit":
            options["unit"] = value
        elif key == "xdist":
            options["xdist"] = value
        elif key in {"nsamples", "nsample"}:
            options["nsamples"] = value
        else:
            raise UnsupportedOptionError("opticsCoC", str(args[index]))

    current = _coerce_optics_struct(optics)
    normalized_unit = param_format(options["unit"])
    if normalized_unit not in _SPATIAL_UNIT_SCALE:
        raise UnsupportedOptionError("opticsCoC", f"unit {options['unit']}")

    o_dist_m = float(np.asarray(o_dist, dtype=float).reshape(-1)[0])
    if o_dist_m <= 0.0:
        raise ValueError("o_dist must be positive.")

    focal_length_m = float(np.asarray(current.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M), dtype=float).reshape(-1)[0])
    if focal_length_m <= 0.0:
        raise ValueError("Optics focal length must be positive.")

    f_number = float(np.asarray(current.get("f_number", 4.0), dtype=float).reshape(-1)[0])
    if f_number <= 0.0:
        raise ValueError("Optics f-number must be positive.")

    raw_xdist = options["xdist"]
    if raw_xdist is None:
        x_dist_m = 10.0 ** (np.log10(o_dist_m) + np.linspace(-0.5, 0.5, int(np.asarray(options["nsamples"], dtype=int).reshape(-1)[0])))
    else:
        x_dist_m = np.asarray(raw_xdist, dtype=float).reshape(-1)
    x_dist_m = np.asarray(x_dist_m[x_dist_m > focal_length_m], dtype=float)

    lensmaker = lambda dist, focal_length: 1.0 / ((1.0 / focal_length) - (1.0 / dist))
    aperture_m = focal_length_m / f_number
    focus_distance_m = lensmaker(o_dist_m, focal_length_m)
    image_distance_m = lensmaker(x_dist_m, focal_length_m)
    image_distance_m = np.maximum(image_distance_m, 0.0)
    circ = aperture_m * np.abs(image_distance_m - focus_distance_m) / image_distance_m
    return np.asarray(circ * _spatial_unit_scale(normalized_unit), dtype=float), x_dist_m


def optics_dof(
    optics: OpticalImage | dict[str, Any],
    o_dist: Any,
    coc_diam: float = 10e-6,
) -> float | np.ndarray:
    """Return the thin-lens depth of field in meters."""

    current = _coerce_optics_struct(optics)
    f_number = float(np.asarray(current.get("f_number", 4.0), dtype=float).reshape(-1)[0])
    focal_length_m = float(np.asarray(current.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M), dtype=float).reshape(-1)[0])
    if f_number <= 0.0 or focal_length_m <= 0.0:
        raise ValueError("Optics f-number and focal length must be positive.")

    object_distance_m = np.asarray(o_dist, dtype=float)
    dof = (2.0 * f_number * float(coc_diam) * np.square(object_distance_m)) / max(focal_length_m**2, 1e-30)
    if dof.ndim == 0:
        return float(dof)
    return np.asarray(dof, dtype=float)


def optics_depth_defocus(
    obj_dist: Any,
    optics: OpticalImage | dict[str, Any],
    img_plane_dist: Any | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return thin-lens defocus in diopters and the in-focus image distance."""

    current = _coerce_optics_struct(optics)
    focal_length_m = _optics_scalar_value(
        current,
        "focal_length_m",
        "nominal_focal_length_m",
        "focalLength",
        "nominalFocalLength",
        default=DEFAULT_FOCAL_LENGTH_M,
    )
    if focal_length_m <= 0.0:
        raise ValueError("Optics focal length must be positive.")

    object_distance_m = np.asarray(obj_dist, dtype=float)
    if np.any(object_distance_m <= focal_length_m):
        raise ValueError("obj_dist must be greater than the focal length.")

    if img_plane_dist is None:
        image_plane_distance_m = np.asarray(focal_length_m, dtype=float)
    else:
        image_plane_distance_m = np.asarray(img_plane_dist, dtype=float)
    if np.any(image_plane_distance_m < focal_length_m):
        raise ValueError("img_plane_dist must be at least the focal length.")

    image_distance_m = 1.0 / ((1.0 / focal_length_m) - (1.0 / object_distance_m))
    defocus_diopters = (1.0 / image_distance_m) - (1.0 / image_plane_distance_m)

    if defocus_diopters.ndim == 0:
        return float(defocus_diopters), float(image_distance_m)
    return np.asarray(defocus_diopters, dtype=float), np.asarray(image_distance_m, dtype=float)


def optics_defocus_depth(
    defocus_diopters: Any,
    optics: OpticalImage | dict[str, Any],
    img_plane_dist: Any | None = None,
) -> float | np.ndarray:
    """Return object distances in meters that produce the requested defocus."""

    current = _coerce_optics_struct(optics)
    focal_length_m = _optics_scalar_value(
        current,
        "focal_length_m",
        "nominal_focal_length_m",
        "focalLength",
        "nominalFocalLength",
        default=DEFAULT_FOCAL_LENGTH_M,
    )
    if focal_length_m <= 0.0:
        raise ValueError("Optics focal length must be positive.")

    if img_plane_dist is None:
        image_plane_distance_m = np.asarray(focal_length_m, dtype=float)
    else:
        image_plane_distance_m = np.asarray(img_plane_dist, dtype=float)
    if np.any(image_plane_distance_m < focal_length_m):
        raise ValueError("img_plane_dist must be at least the focal length.")

    defocus = np.asarray(defocus_diopters, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        image_distance_m = 1.0 / ((1.0 / image_plane_distance_m) + defocus)
        object_distance_m = (image_distance_m * focal_length_m) / np.maximum(image_distance_m - focal_length_m, 1e-30)

    infinite_mask = np.isclose(image_distance_m, focal_length_m)
    if np.any(infinite_mask):
        object_distance_m = np.asarray(object_distance_m, dtype=float)
        object_distance_m[infinite_mask] = np.inf

    if np.asarray(object_distance_m).ndim == 0:
        return float(np.asarray(object_distance_m, dtype=float))
    return np.asarray(object_distance_m, dtype=float)


def optics_defocus_displacement(base_power_diopters: Any, delta_power_diopters: Any) -> float | np.ndarray:
    """Return the image-plane displacement in meters for a change in lens power."""

    base_power = np.asarray(base_power_diopters, dtype=float)
    delta_power = np.asarray(delta_power_diopters, dtype=float)
    if np.any(base_power <= 0.0):
        raise ValueError("base_power_diopters must be positive.")
    if np.any(base_power + delta_power <= 0.0):
        raise ValueError("base_power_diopters + delta_power_diopters must be positive.")

    displacement = (1.0 / base_power) - (1.0 / (base_power + delta_power))
    if displacement.ndim == 0:
        return float(displacement)
    return np.asarray(displacement, dtype=float)


def _optics_wave_values(optics: dict[str, Any]) -> np.ndarray:
    transmittance = optics.get("transmittance")
    if isinstance(transmittance, dict) and transmittance.get("wave") is not None:
        wave = np.asarray(transmittance.get("wave"), dtype=float).reshape(-1)
        if wave.size > 0:
            return wave
    wave = optics.get("otf_wave")
    if wave is not None:
        wave_values = np.asarray(wave, dtype=float).reshape(-1)
        if wave_values.size > 0:
            return wave_values
    return np.asarray(DEFAULT_WAVE, dtype=float).reshape(-1)


def _optics_scalar_value(optics: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = optics.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size > 0:
            return float(array[0])
    return float(default)


def _optics_defocused_mtf(reduced_spatial_frequency: Any, alpha: Any) -> np.ndarray:
    reduced = np.asarray(reduced_spatial_frequency, dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    nf = np.abs(reduced) / 2.0
    beta = np.sqrt(np.clip(1.0 - np.square(nf), 0.0, None))

    otf = np.zeros_like(nf, dtype=float)
    focused = np.isclose(alpha_array, 0.0)
    if np.any(focused):
        otf[focused] = (2.0 / np.pi) * (np.arccos(np.clip(nf[focused], -1.0, 1.0)) - (nf[focused] * beta[focused]))

    defocused = ~focused
    if np.any(defocused):
        alpha_values = alpha_array[defocused]
        beta_values = beta[defocused]
        nf_values = nf[defocused]
        h1 = (
            beta_values * jv(1, alpha_values)
            + 0.5 * np.sin(2.0 * beta_values) * (jv(1, alpha_values) - jv(3, alpha_values))
            - 0.25 * np.sin(4.0 * beta_values) * (jv(3, alpha_values) - jv(5, alpha_values))
        )
        h2 = (
            np.sin(beta_values) * (jv(0, alpha_values) - jv(2, alpha_values))
            + (1.0 / 3.0) * np.sin(3.0 * beta_values) * (jv(2, alpha_values) - jv(4, alpha_values))
            - (1.0 / 5.0) * np.sin(5.0 * beta_values) * (jv(4, alpha_values) - jv(6, alpha_values))
        )
        scale = 4.0 / (np.pi * alpha_values)
        otf[defocused] = (scale * np.cos(alpha_values * nf_values) * h1) - (scale * np.sin(alpha_values * nf_values) * h2)

    otf[nf > 1.0] = 0.0
    dc = float(np.ravel(otf)[0]) if otf.size else 1.0
    if not np.isclose(dc, 0.0):
        otf = otf / dc
    return np.asarray(otf, dtype=float)


def optics_defocus_core(
    optics: OpticalImage | dict[str, Any],
    sample_sf_cpd: Any,
    defocus_diopters: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the wavelength-by-frequency defocus OTF rows and support in cyc/mm."""

    current = _coerce_optics_struct(optics)
    sample_sf = np.asarray(sample_sf_cpd, dtype=float).reshape(-1)
    if sample_sf.size == 0:
        raise ValueError("sample_sf_cpd must contain at least one sample.")

    wave = _optics_wave_values(current)
    defocus = np.asarray(defocus_diopters, dtype=float).reshape(-1)
    if defocus.size == 1:
        defocus = np.full(wave.size, float(defocus[0]), dtype=float)
    if defocus.size != wave.size:
        raise ValueError("defocus_diopters must be scalar or match the optics wavelength count.")

    focal_length_m = _optics_scalar_value(
        current,
        "focal_length_m",
        "nominal_focal_length_m",
        "focalLength",
        "nominalFocalLength",
        default=DEFAULT_FOCAL_LENGTH_M,
    )
    f_number = _optics_scalar_value(current, "f_number", "fNumber", default=4.0)
    if focal_length_m <= 0.0 or f_number <= 0.0:
        raise ValueError("Optics focal length and f-number must be positive.")

    diopters = 1.0 / focal_length_m
    pupil_radius_m = focal_length_m / (2.0 * f_number)
    w20 = ((pupil_radius_m**2) / 2.0) * (diopters * defocus) / np.maximum(diopters + defocus, 1e-12)
    deg_per_meter = diopters / np.tan(np.deg2rad(1.0))
    cycles_per_meter = sample_sf * deg_per_meter
    if np.any(np.isclose(cycles_per_meter, 0.0)):
        nonzero = cycles_per_meter[~np.isclose(cycles_per_meter, 0.0)]
        if nonzero.size > 0:
            cycles_per_meter = cycles_per_meter.copy()
            cycles_per_meter[np.isclose(cycles_per_meter, 0.0)] = np.min(np.abs(nonzero)) * 1e-12

    wavelengths_m = wave * 1e-9
    otf = np.zeros((wave.size, sample_sf.size), dtype=float)
    for band_index, wavelength_m in enumerate(wavelengths_m):
        reduced_sf = (wavelength_m / max(diopters * pupil_radius_m, 1e-12)) * cycles_per_meter
        alpha = (4.0 * np.pi / max(wavelength_m, 1e-12)) * w20[band_index] * np.abs(reduced_sf)
        otf[band_index, :] = _optics_defocused_mtf(reduced_sf, np.abs(alpha))

    sample_sf_mm = sample_sf * (deg_per_meter / 1000.0)
    return np.asarray(otf, dtype=float), np.asarray(sample_sf_mm, dtype=float)


def optics_build_2d_otf(
    optics: OpticalImage | dict[str, Any],
    otf: Any,
    sample_sf_mm: Any,
) -> dict[str, Any]:
    """Build and store a circularly symmetric 2-D OTF bundle on an optics struct."""

    current = _coerce_optics_struct(optics)
    sample_support = np.asarray(sample_sf_mm, dtype=float).reshape(-1)
    if sample_support.size == 0:
        raise ValueError("sample_sf_mm must contain at least one sample.")

    otf_rows = np.asarray(otf, dtype=float)
    if otf_rows.ndim == 1:
        otf_rows = otf_rows.reshape(1, -1)
    if otf_rows.shape[1] != sample_support.size:
        raise ValueError("OTF rows must have the same frequency count as sample_sf_mm.")

    wave = _optics_wave_values(current)
    if otf_rows.shape[0] == 1 and wave.size > 1:
        otf_rows = np.repeat(otf_rows, wave.size, axis=0)
    if otf_rows.shape[0] != wave.size:
        raise ValueError("OTF row count must match the optics wavelength count.")

    max_sample = float(np.max(np.abs(sample_support)))
    max_frequency = max(int(np.ceil(np.sqrt(max_sample**2 + max_sample**2))), 1)
    f_support = unit_frequency_list(max_frequency) * max_frequency
    fx, fy = np.meshgrid(f_support, f_support, indexing="xy")
    effective_sf = np.sqrt(np.square(fx) + np.square(fy))
    outside = effective_sf > max_frequency

    otf_data = np.zeros((f_support.size, f_support.size, wave.size), dtype=complex)
    for band_index in range(wave.size):
        plane = np.interp(
            effective_sf.reshape(-1),
            sample_support,
            np.abs(otf_rows[band_index, :]),
            left=0.0,
            right=0.0,
        ).reshape(effective_sf.shape)
        plane[outside] = 0.0
        shifted = np.fft.ifftshift(plane)
        dc = shifted[0, 0]
        if not np.isclose(dc, 0.0):
            shifted = shifted / dc
        otf_data[:, :, band_index] = np.asarray(shifted, dtype=complex)

    focal_length_m = _optics_scalar_value(
        current,
        "focal_length_m",
        "nominal_focal_length_m",
        "focalLength",
        "nominalFocalLength",
        default=DEFAULT_FOCAL_LENGTH_M,
    )
    nominal_focal_length_m = _optics_scalar_value(
        current,
        "nominal_focal_length_m",
        "focalLength",
        "nominalFocalLength",
        "focal_length_m",
        default=focal_length_m,
    )
    f_number = _optics_scalar_value(current, "f_number", "fNumber", default=4.0)
    transmittance = dict(current.get("transmittance", {}))
    transmittance_wave = np.asarray(transmittance.get("wave", wave), dtype=float).reshape(-1)
    transmittance_scale = np.asarray(
        transmittance.get("scale", np.ones(transmittance_wave.size, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if transmittance_scale.size == 1 and transmittance_wave.size > 1:
        transmittance_scale = np.full(transmittance_wave.size, float(transmittance_scale[0]), dtype=float)
    if transmittance_scale.size != transmittance_wave.size:
        transmittance_scale = np.ones(transmittance_wave.size, dtype=float)

    updated = {
        "name": str(current.get("name", "")),
        "model": "shiftinvariant",
        "f_number": f_number,
        "focal_length_m": focal_length_m,
        "nominal_focal_length_m": nominal_focal_length_m,
        "compute_method": "opticsotf",
        "aberration_scale": float(current.get("aberration_scale", current.get("aberrationScale", 0.0))),
        "offaxis_method": str(
            current.get("offaxis_method", current.get("offaxisMethod", current.get("offaxis", "skip")))
        ),
        "transmittance": {
            "wave": transmittance_wave.copy(),
            "scale": transmittance_scale.copy(),
        },
        "otf_data": otf_data,
        "otf_fx": np.asarray(f_support, dtype=float),
        "otf_fy": np.asarray(f_support, dtype=float),
        "otf_wave": wave.copy(),
        "otf_function": "custom",
    }
    if "wavefront" in current:
        updated["wavefront"] = dict(current["wavefront"])
    return updated


def _normalize_shift_invariant_psf_data(value: Any) -> dict[str, Any]:
    current = dict(value)
    psf = np.asarray(current.get("psf"), dtype=float)
    if psf.ndim == 2:
        psf = psf[:, :, None]
    if psf.ndim != 3:
        raise ValueError("Shift-invariant PSF data must be a 2-D or 3-D array.")

    wave_raw = current.get("wave", current.get("wavelength", current.get("wavelength_nm")))
    if wave_raw is None:
        if psf.shape[2] == 1:
            wave = np.array([550.0], dtype=float)
        elif psf.shape[2] <= DEFAULT_WAVE.size:
            wave = np.asarray(DEFAULT_WAVE[: psf.shape[2]], dtype=float)
        else:
            wave = np.linspace(400.0, 700.0, psf.shape[2], dtype=float)
    else:
        wave = np.asarray(wave_raw, dtype=float).reshape(-1)
    if psf.shape[2] != wave.size and psf.shape[2] != 1:
        raise ValueError("Shift-invariant PSF wavelength samples must match the PSF stack depth.")

    spacing_raw = current.get(
        "sample_spacing_m",
        current.get(
            "mmPerSamp",
            current.get(
                "sample_spacing_mm",
                current.get(
                    "sample_spacing_um",
                    current.get("umPerSamp"),
                ),
            ),
        ),
    )
    if spacing_raw is None:
        sample_spacing_m = 0.25e-6
    else:
        spacing = np.asarray(spacing_raw, dtype=float).reshape(-1)
        if spacing.size == 0:
            sample_spacing_m = 0.25e-6
        else:
            sample_spacing_m = float(np.mean(spacing))
            if current.get("sample_spacing_m") is None:
                if current.get("mmPerSamp") is not None or current.get("sample_spacing_mm") is not None:
                    sample_spacing_m *= 1e-3
                elif current.get("sample_spacing_um") is not None or current.get("umPerSamp") is not None:
                    sample_spacing_m *= 1e-6

    normalized = {
        "psf": psf.copy(),
        "wave": wave.copy(),
        "sample_spacing_m": float(sample_spacing_m),
    }
    normalized["umPerSamp"] = np.full(2, float(sample_spacing_m) * 1e6, dtype=float)
    return normalized


def _export_shift_invariant_psf_data(value: dict[str, Any]) -> dict[str, Any]:
    current = _normalize_shift_invariant_psf_data(value)
    return {
        "psf": np.asarray(current["psf"], dtype=float).copy(),
        "wave": np.asarray(current["wave"], dtype=float).copy(),
        "umPerSamp": np.asarray(current["umPerSamp"], dtype=float).copy(),
        "sample_spacing_m": float(current["sample_spacing_m"]),
    }


def _normalize_shift_invariant_otf_struct(
    value: Any,
    *,
    target_wave: np.ndarray | None = None,
) -> dict[str, Any]:
    current = dict(value)
    otf = np.asarray(current.get("OTF", current.get("otf")), dtype=complex)
    if otf.ndim == 2:
        otf = otf[:, :, None]
    if otf.ndim != 3:
        raise ValueError("Shift-invariant OTF data must be a 2-D or 3-D array.")

    wave_default = DEFAULT_WAVE if target_wave is None else np.asarray(target_wave, dtype=float).reshape(-1)
    wave = np.asarray(current.get("wave", current.get("otf_wave", wave_default)), dtype=float).reshape(-1)
    if wave.size == 0:
        raise ValueError("Shift-invariant OTF data must include wavelength samples.")
    if otf.shape[2] == 1 and wave.size > 1:
        otf = np.repeat(otf, wave.size, axis=2)
    elif otf.shape[2] != wave.size:
        raise ValueError("Shift-invariant OTF wavelength dimension must match the wavelength vector.")

    fx = np.asarray(current.get("fx", current.get("otf_fx")), dtype=float).reshape(-1)
    fy = np.asarray(current.get("fy", current.get("otf_fy")), dtype=float).reshape(-1)
    if fx.size != otf.shape[1]:
        raise ValueError("OTF fx support must match OTF column count.")
    if fy.size != otf.shape[0]:
        raise ValueError("OTF fy support must match OTF row count.")

    return {
        "function": str(current.get("function", current.get("otf_function", "custom"))),
        "OTF": otf.copy(),
        "fx": fx.copy(),
        "fy": fy.copy(),
        "wave": wave.copy(),
    }


def _export_shift_invariant_otf_struct(value: dict[str, Any]) -> dict[str, Any]:
    current = _normalize_shift_invariant_otf_struct(value)
    return {
        "function": current["function"],
        "OTF": current["OTF"].copy(),
        "fx": current["fx"].copy(),
        "fy": current["fy"].copy(),
        "wave": current["wave"].copy(),
    }


def optics_psf_to_otf(
    image_source: Any,
    pix_size_m: float = 1.2e-6,
    wave: np.ndarray | None = None,
) -> dict[str, Any]:
    wave_values = np.asarray(DEFAULT_WAVE if wave is None else wave, dtype=float).reshape(-1)
    if isinstance(image_source, (str, Path)):
        image = np.asarray(iio.imread(Path(image_source)), dtype=float)
    else:
        image = np.asarray(image_source, dtype=float)

    if image.ndim == 3:
        channel_index = 1 if image.shape[2] > 1 else 0
        psf = np.asarray(image[:, :, channel_index], dtype=float)
    elif image.ndim == 2:
        psf = np.asarray(image, dtype=float)
    else:
        raise ValueError("PSF image source must be a 2-D grayscale image or a 3-D image array.")

    total = float(np.sum(psf))
    if total <= 0.0:
        raise ValueError("PSF image must contain positive energy.")
    psf = psf / total

    rows, cols = psf.shape
    otf_plane = np.fft.fft2(np.fft.fftshift(psf))
    img_size_mm = float(pix_size_m) * cols * 1e3
    fx = np.arange(-(cols / 2.0), cols / 2.0, 1.0, dtype=float) * (1.0 / max(img_size_mm, 1e-12))
    fy = np.arange(-(rows / 2.0), rows / 2.0, 1.0, dtype=float) * (1.0 / max(img_size_mm, 1e-12))

    return _normalize_shift_invariant_otf_struct(
        {
            "function": "custom",
            "OTF": np.repeat(otf_plane[:, :, None], wave_values.size, axis=2),
            "fx": fx,
            "fy": fy,
            "wave": wave_values,
        }
    )


def make_combined_otf(otf: Any, sample_sf: Any) -> np.ndarray:
    """Apply the Williams/Brainard chromatic-aberration correction factor."""

    otf_array = np.asarray(otf, dtype=float)
    sample_sf_array = np.asarray(sample_sf, dtype=float).reshape(-1)
    if otf_array.ndim != 2:
        raise ValueError("makeCombinedOtf expects a 2-D OTF array.")
    if otf_array.shape[1] != sample_sf_array.size:
        raise ValueError("sampleSf length must match the OTF spatial-frequency dimension.")

    a = 0.1212
    w1 = 0.3481
    w2 = 0.6519
    williams_factor = w1 + w2 * np.exp(-a * sample_sf_array)
    return np.asarray(otf_array * williams_factor.reshape(1, -1), dtype=float)


def make_cmatrix(otf: Any, receptor: Any, monitor_spd: Any) -> np.ndarray:
    """Build per-frequency 3x3 calibration matrices for the chromatic-aberration model."""

    otf_array = np.asarray(otf, dtype=float)
    receptor_array = np.asarray(receptor, dtype=float)
    monitor_array = np.asarray(monitor_spd, dtype=float)
    if otf_array.ndim != 2:
        raise ValueError("makeCmatrix expects a 2-D OTF array.")
    if receptor_array.ndim != 2 or monitor_array.ndim != 2:
        raise ValueError("makeCmatrix expects 2-D receptor and monitor SPD arrays.")
    if otf_array.shape[0] != receptor_array.shape[0] or otf_array.shape[0] != monitor_array.shape[0]:
        raise ValueError("OTF, receptor, and monitor SPD must agree on the wavelength dimension.")

    n_receptors = receptor_array.shape[1]
    n_primaries = monitor_array.shape[1]
    matrices = np.zeros((n_receptors * n_primaries, otf_array.shape[1]), dtype=float)
    for frequency_index in range(otf_array.shape[1]):
        current = receptor_array.T @ (otf_array[:, frequency_index][:, None] * monitor_array)
        matrices[:, frequency_index] = np.reshape(current, -1, order="F")
    return matrices


def retinal_image(image: Any, c_matrices: Any) -> np.ndarray:
    """Apply the 1-D chromatic-aberration frequency-domain transform to an RGB image strip."""

    image_array = np.asarray(image, dtype=float)
    matrices_array = np.asarray(c_matrices, dtype=float)
    if image_array.ndim != 2 or image_array.shape[1] != 3:
        raise ValueError("retinalImage expects an N x 3 image array.")
    if matrices_array.ndim != 2 or matrices_array.shape[0] != 9:
        raise ValueError("retinalImage expects Cmatrices with shape 9 x M.")

    image_t = image_array.T
    max_sf = int(matrices_array.shape[1] - 1)
    image_size = int(image_t.shape[1])
    if max_sf <= 0:
        raise ValueError("retinalImage requires at least two Cmatrix frequency samples.")
    if (image_size / 2.0) % max_sf != 0:
        raise ValueError("The image is not matched to the OTF (too large).")

    image_fft = np.fft.fft(image_t, axis=1)
    cone_response_fft = np.zeros_like(image_fft, dtype=np.complex128)

    for frequency_index in range(max_sf + 1):
        cmatrix = matrices_array[:, frequency_index].reshape(3, 3, order="F")
        cone_response_fft[:, frequency_index] = cmatrix @ image_fft[:, frequency_index]

    upper_limit = min(2 * max_sf, image_fft.shape[1] - 1)
    for frequency_index in range(max_sf + 1, upper_limit + 1):
        mirrored_index = 2 * max_sf + 1 - frequency_index
        cmatrix = matrices_array[:, mirrored_index - 1].reshape(3, 3, order="F")
        cone_response_fft[:, frequency_index] = cmatrix @ image_fft[:, frequency_index]

    image_out = np.real(np.fft.ifft(cone_response_fft, axis=1))
    return np.asarray(image_out.T, dtype=float)


makeCombinedOtf = make_combined_otf
makeCmatrix = make_cmatrix
retinalImage = retinal_image


def _synthetic_shift_invariant_gaussian_psf_data(
    wave: np.ndarray,
    f_number: float,
    *,
    samples: int = 129,
    sample_spacing_um: float = 0.25,
) -> dict[str, Any]:
    wave_values = np.asarray(wave, dtype=float).reshape(-1)
    airy_radius_um = 1.22 * float(f_number) * (float(np.max(wave_values)) * 1e-3)
    sigma_um = 2.0 * airy_radius_um
    sigma_pixels = sigma_um / max(float(sample_spacing_um), 1e-12)
    kernel_1d = _gaussian_kernel_1d(int(samples), float(sigma_pixels))
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / max(float(np.sum(kernel_2d)), 1e-12)
    psf = np.repeat(kernel_2d[:, :, None], wave_values.size, axis=2)
    return _normalize_shift_invariant_psf_data(
        {
            "psf": psf,
            "wave": wave_values,
            "umPerSamp": np.array([sample_spacing_um, sample_spacing_um], dtype=float),
        }
    )


def _synthetic_shift_invariant_gaussian_psf_data_from_spreads(
    wave: np.ndarray,
    x_spread_um: Any,
    xy_ratio: Any,
    *,
    samples: int = 129,
    sample_spacing_um: float = 0.25,
) -> dict[str, Any]:
    wave_values = np.asarray(wave, dtype=float).reshape(-1)
    n_wave = wave_values.size

    x_spread = np.asarray(x_spread_um, dtype=float).reshape(-1)
    if x_spread.size == 1:
        x_spread = np.full(n_wave, float(x_spread[0]), dtype=float)
    if x_spread.size != n_wave:
        raise ValueError("Gaussian waveSpread must be scalar or match the wavelength count.")

    xy_ratio_values = np.asarray(xy_ratio, dtype=float).reshape(-1)
    if xy_ratio_values.size == 1:
        xy_ratio_values = np.full(n_wave, float(xy_ratio_values[0]), dtype=float)
    if xy_ratio_values.size != n_wave:
        raise ValueError("Gaussian xyRatio must be scalar or match the wavelength count.")

    center = (int(samples) - 1) / 2.0
    positions = np.arange(int(samples), dtype=float) - center
    xx, yy = np.meshgrid(positions, positions, indexing="xy")
    psf = np.empty((int(samples), int(samples), n_wave), dtype=float)

    for wave_index in range(n_wave):
        sigma_x = max(float(x_spread[wave_index]) / max(float(sample_spacing_um), 1e-12), 1e-12)
        sigma_y = max(float(x_spread[wave_index] * xy_ratio_values[wave_index]) / max(float(sample_spacing_um), 1e-12), 1e-12)
        plane = np.exp(-0.5 * (((xx / sigma_x) ** 2) + ((yy / sigma_y) ** 2)))
        plane = plane / max(float(np.sum(plane)), 1e-12)
        # MATLAB siSynthetic rotates the Gaussian PSF before OTF conversion,
        # which swaps the displayed horizontal/vertical spread in the
        # stored PSF data returned by opticsGet(..., 'psf data').
        psf[:, :, wave_index] = np.rot90(plane)

    return _normalize_shift_invariant_psf_data(
        {
            "psf": psf,
            "wave": wave_values,
            "umPerSamp": np.array([sample_spacing_um, sample_spacing_um], dtype=float),
        }
    )


def _synthetic_shift_invariant_lorentzian_psf_data(
    wave: np.ndarray,
    g_parameter: Any,
    *,
    samples: int = 129,
    sample_spacing_um: float = 0.25,
) -> dict[str, Any]:
    wave_values = np.asarray(wave, dtype=float).reshape(-1)
    n_wave = wave_values.size

    gamma = np.asarray(g_parameter, dtype=float).reshape(-1)
    if gamma.size == 1:
        gamma = np.full(n_wave, float(gamma[0]), dtype=float)
    if gamma.size != n_wave:
        raise ValueError("Lorentzian gParameter must be scalar or match the wavelength count.")

    center = (int(samples) - 1) / 2.0
    positions = np.arange(int(samples), dtype=float) - center
    xx, yy = np.meshgrid(positions, positions, indexing="xy")
    radius = np.sqrt(xx**2 + yy**2)
    psf = np.empty((int(samples), int(samples), n_wave), dtype=float)

    for wave_index in range(n_wave):
        plane = 1.0 / (1.0 + (radius / max(float(gamma[wave_index]), 1e-12)) ** 2)
        plane = plane / max(float(np.sum(plane)), 1e-12)
        psf[:, :, wave_index] = plane

    return _normalize_shift_invariant_psf_data(
        {
            "psf": psf,
            "wave": wave_values,
            "umPerSamp": np.array([sample_spacing_um, sample_spacing_um], dtype=float),
        }
    )


def _synthetic_shift_invariant_pillbox_psf_data(
    wave: np.ndarray,
    patch_size_mm: float,
    *,
    samples: int = 129,
    sample_spacing_um: float = 0.25,
) -> dict[str, Any]:
    wave_values = np.asarray(wave, dtype=float).reshape(-1)
    sample_spacing_mm = float(sample_spacing_um) * 1e-3
    patch_samples = int(np.ceil(float(patch_size_mm) / max(sample_spacing_mm, 1e-12)))
    center = (int(samples) - 1) / 2.0
    positions = np.arange(int(samples), dtype=float) - center
    xx, yy = np.meshgrid(positions, positions, indexing="xy")
    plane = ((np.abs(xx) <= patch_samples) & (np.abs(yy) <= patch_samples)).astype(float)
    plane = plane / max(float(np.sum(plane)), 1e-12)
    psf = np.repeat(plane[:, :, None], wave_values.size, axis=2)
    return _normalize_shift_invariant_psf_data(
        {
            "psf": psf,
            "wave": wave_values,
            "umPerSamp": np.array([sample_spacing_um, sample_spacing_um], dtype=float),
        }
    )


def _synthetic_shift_invariant_gaussian_otf_bundle(
    wave: np.ndarray,
    f_number: float,
    *,
    samples: int = 129,
    sample_spacing_um: float = 0.25,
) -> dict[str, Any]:
    wave_values = np.asarray(wave, dtype=float).reshape(-1)
    airy_radius_um = 1.22 * float(f_number) * (float(np.max(wave_values)) * 1e-3)
    sigma_um = 2.0 * airy_radius_um
    sigma_pixels = sigma_um / max(float(sample_spacing_um), 1e-12)
    kernel_1d = _gaussian_kernel_1d(int(samples), float(sigma_pixels))
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / max(float(np.sum(kernel_2d)), 1e-12)
    shifted_psf = np.fft.ifftshift(np.rot90(kernel_2d))
    otf_plane = np.fft.fft2(shifted_psf)
    dx_mm = float(sample_spacing_um) * 1e-3
    nyquist_frequency = 1.0 / max(2.0 * dx_mm, 1e-12)
    support = unit_frequency_list(int(samples)) * nyquist_frequency
    return {
        "otf_data": np.repeat(otf_plane[:, :, None], wave_values.size, axis=2),
        "otf_fx": support.copy(),
        "otf_fy": support.copy(),
        "otf_wave": wave_values.copy(),
    }


def _custom_shift_invariant_otf_bundle(
    psf_data: dict[str, Any],
    *,
    samples: int = 129,
    sample_spacing_um: float = 0.25,
) -> dict[str, Any]:
    normalized = _normalize_shift_invariant_psf_data(psf_data)
    source_psf = np.asarray(normalized["psf"], dtype=float)
    source_wave = np.asarray(normalized["wave"], dtype=float).reshape(-1)
    source_spacing_mm = float(normalized["sample_spacing_m"]) * 1e3
    target_spacing_mm = float(sample_spacing_um) * 1e-3
    source_x = _centered_support_axis(source_psf.shape[1], source_spacing_mm)
    source_y = _centered_support_axis(source_psf.shape[0], source_spacing_mm)
    target_support = _centered_support_axis(int(samples), target_spacing_mm)
    otf_data = np.empty((int(samples), int(samples), source_wave.size), dtype=complex)

    for band_index in range(source_wave.size):
        plane = source_psf[:, :, 0 if source_psf.shape[2] == 1 else band_index]
        if plane.shape != (int(samples), int(samples)) or not np.isclose(source_spacing_mm, target_spacing_mm):
            plane = _resample_plane_on_support(
                plane,
                source_x,
                source_y,
                target_support,
                target_support,
                method="linear",
            )
            yy, xx = np.meshgrid(target_support, target_support, indexing="ij")
            outside = (
                (xx < float(np.min(source_x)))
                | (xx > float(np.max(source_x)))
                | (yy < float(np.min(source_y)))
                | (yy > float(np.max(source_y)))
            )
            plane[outside] = 0.0
        plane = np.clip(np.asarray(plane, dtype=float), 0.0, None)
        plane = plane / max(float(np.sum(plane)), 1e-12)
        otf_data[:, :, band_index] = np.fft.fft2(np.fft.fftshift(plane))

    nyquist_frequency = 1.0 / max(2.0 * target_spacing_mm, 1e-12)
    support = unit_frequency_list(int(samples)) * nyquist_frequency
    return {
        "otf_data": otf_data,
        "otf_fx": support.copy(),
        "otf_fy": support.copy(),
        "otf_wave": source_wave.copy(),
    }


def _shift_invariant_otf_bundle_from_psf_data(
    psf_data: dict[str, Any],
    *,
    center_shift: str = "fft",
) -> dict[str, Any]:
    normalized = _normalize_shift_invariant_psf_data(psf_data)
    source_psf = np.asarray(normalized["psf"], dtype=float)
    source_wave = np.asarray(normalized["wave"], dtype=float).reshape(-1)
    sample_spacing_mm = float(normalized["sample_spacing_m"]) * 1e3
    rows, cols = source_psf.shape[:2]
    otf_data = np.empty((rows, cols, source_wave.size), dtype=complex)

    for band_index in range(source_wave.size):
        plane = np.asarray(source_psf[:, :, 0 if source_psf.shape[2] == 1 else band_index], dtype=float)
        plane = np.clip(plane, 0.0, None)
        plane = plane / max(float(np.sum(plane)), 1e-12)
        if center_shift == "ifft":
            shifted = np.fft.ifftshift(plane)
        elif center_shift == "fft":
            shifted = np.fft.fftshift(plane)
        else:
            raise ValueError("center_shift must be 'fft' or 'ifft'.")
        otf_data[:, :, band_index] = np.fft.fft2(shifted)

    nyquist_frequency = 1.0 / max(2.0 * sample_spacing_mm, 1e-12)
    fx = unit_frequency_list(cols) * nyquist_frequency
    fy = unit_frequency_list(rows) * nyquist_frequency
    return {
        "otf_data": otf_data,
        "otf_fx": fx.copy(),
        "otf_fy": fy.copy(),
        "otf_wave": source_wave.copy(),
    }


def _shift_invariant_otf_support(
    rows: int,
    cols: int,
    sample_spacing_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    spacing_mm = float(sample_spacing_m) * 1e3 if sample_spacing_m is not None else 0.0
    if spacing_mm > 0.0:
        fx = unit_frequency_list(int(cols)) * (1.0 / max(2.0 * spacing_mm, 1e-12))
        fy = unit_frequency_list(int(rows)) * (1.0 / max(2.0 * spacing_mm, 1e-12))
    else:
        fx = unit_frequency_list(int(cols))
        fy = unit_frequency_list(int(rows))
    return np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)


def _synthesized_shift_invariant_otf_bundle(oi: OpticalImage) -> dict[str, Any] | None:
    optics = dict(oi.fields.get("optics", {}))
    otf_data = optics.get("otf_data")
    if otf_data is not None:
        source = np.asarray(otf_data, dtype=complex)
        rows, cols = source.shape[:2]
        fx = optics.get("otf_fx")
        fy = optics.get("otf_fy")
        if fx is None or fy is None:
            support_fx, support_fy = _shift_invariant_otf_support(
                rows,
                cols,
                oi.fields.get("sample_spacing_m"),
            )
            fx = support_fx
            fy = support_fy
        return {
            "OTF": source.copy(),
            "fx": np.asarray(fx, dtype=float).reshape(-1).copy(),
            "fy": np.asarray(fy, dtype=float).reshape(-1).copy(),
            "wave": np.asarray(optics.get("otf_wave", oi.fields.get("wave", DEFAULT_WAVE)), dtype=float).reshape(-1).copy(),
        }

    if param_format(optics.get("model", "")) != "shiftinvariant":
        return None

    wavefront = optics.get("wavefront")
    if isinstance(wavefront, dict):
        current_wvf = dict(wavefront if wavefront.get("computed") else wvf_compute(wavefront))
        wave = np.asarray(current_wvf.get("wave", oi.fields.get("wave", DEFAULT_WAVE)), dtype=float).reshape(-1)
        if wave.size == 0:
            wave = DEFAULT_WAVE.copy()
        first_wave = float(wave[0])
        first_plane = np.fft.fftshift(np.asarray(wvf_get(current_wvf, "otf", first_wave), dtype=complex))
        rows, cols = first_plane.shape[:2]
        otf_stack = np.empty((rows, cols, wave.size), dtype=complex)
        otf_stack[:, :, 0] = first_plane
        for band_index in range(1, wave.size):
            otf_stack[:, :, band_index] = np.fft.fftshift(
                np.asarray(
                    wvf_get(current_wvf, "otf", float(wave[band_index])),
                    dtype=complex,
                )
            )
        support = np.asarray(wvf_get(current_wvf, "otf support", "mm", first_wave), dtype=float).reshape(-1)
        return {
            "OTF": otf_stack,
            "fx": support.copy(),
            "fy": support.copy(),
            "wave": wave.copy(),
        }

    rows, cols = _oi_shape(oi)
    sample_spacing_m = oi.fields.get("sample_spacing_m")
    wave = np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    if rows <= 0 or cols <= 0 or sample_spacing_m is None or float(sample_spacing_m) <= 0.0 or wave.size == 0:
        return None

    psf_stack = _shift_invariant_psf_stack((rows, cols), float(sample_spacing_m), wave, optics)
    otf_stack = np.empty((rows, cols, wave.size), dtype=complex)
    for band_index in range(wave.size):
        plane = np.asarray(psf_stack[:, :, band_index], dtype=float)
        plane = plane / max(float(np.sum(plane)), 1e-12)
        otf_stack[:, :, band_index] = np.fft.fft2(np.fft.ifftshift(plane))
    fx, fy = _shift_invariant_otf_support(rows, cols, float(sample_spacing_m))
    return {
        "OTF": otf_stack,
        "fx": fx,
        "fy": fy,
        "wave": wave.copy(),
    }


def _default_plot_wavelength(wave: Any) -> float:
    wavelengths = np.asarray(wave, dtype=float).reshape(-1)
    if wavelengths.size == 0:
        return 550.0
    if wavelengths.size == 1:
        return float(wavelengths[0])
    return 550.0


def _support_grid_from_axes(x_axis_m: np.ndarray, y_axis_m: np.ndarray, units: Any | None) -> np.ndarray:
    scale = _spatial_unit_scale(units)
    x_axis = np.asarray(x_axis_m, dtype=float) * scale
    y_axis = np.asarray(y_axis_m, dtype=float) * scale
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    return np.stack((xx, yy), axis=2)


def _stack_plane_at_wavelength(stack: np.ndarray, wave: np.ndarray, wavelength_nm: float) -> np.ndarray:
    source = np.asarray(stack)
    wavelengths = np.asarray(wave, dtype=float).reshape(-1)
    if source.ndim == 2:
        return source.copy()
    if source.shape[2] == 1 or wavelengths.size <= 1:
        return np.asarray(source[:, :, 0]).copy()

    query = float(wavelength_nm)
    if query <= float(wavelengths[0]):
        return np.asarray(source[:, :, 0]).copy()
    if query >= float(wavelengths[-1]):
        return np.asarray(source[:, :, -1]).copy()

    upper_index = int(np.searchsorted(wavelengths, query, side="right"))
    lower_index = max(upper_index - 1, 0)
    upper_index = min(upper_index, wavelengths.size - 1)
    lower_wave = float(wavelengths[lower_index])
    upper_wave = float(wavelengths[upper_index])
    if np.isclose(lower_wave, upper_wave):
        return np.asarray(source[:, :, lower_index]).copy()
    weight = (query - lower_wave) / (upper_wave - lower_wave)
    return (
        (1.0 - weight) * np.asarray(source[:, :, lower_index])
        + weight * np.asarray(source[:, :, upper_index])
    )


def _plane_at_wavelength(stack: np.ndarray, wave: np.ndarray, wavelength_nm: float) -> np.ndarray:
    return np.asarray(_stack_plane_at_wavelength(stack, wave, wavelength_nm), dtype=float)


def _spatial_axis_from_frequency_support(f_support: np.ndarray, *, support_unit_to_m: float) -> np.ndarray:
    support = np.asarray(f_support, dtype=float).reshape(-1)
    if support.size == 0:
        return np.zeros(0, dtype=float)
    peak_frequency = float(np.max(np.abs(support)))
    if peak_frequency <= 0.0:
        return np.zeros(support.size, dtype=float)
    sample_spacing_m = (1.0 / max(2.0 * peak_frequency, 1e-12)) * float(support_unit_to_m)
    return _centered_support_axis(support.size, sample_spacing_m)


def _diffraction_limited_plot_otf(
    oi: OpticalImage,
    wavelength_nm: float,
    *,
    n_samp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    optics = dict(oi.fields.get("optics", {}))
    rows = cols = max(2 * int(n_samp), 2)
    wavelength_m = float(wavelength_nm) * 1e-9
    f_number = float(optics.get("f_number", 4.0))
    cutoff_frequency = 1.0 / max(wavelength_m * max(f_number, 1e-12), 1e-12)
    # Match MATLAB opticsGet(optics, 'dl fsupport matrix', wave, units, nSamp):
    # fSamp = (-nSamp:(nSamp-1)) / nSamp, then the PSF plotting path expands
    # the support by a factor of 4 before calling dlMTF().
    f_samp = np.arange(-int(n_samp), int(n_samp), dtype=float) / max(float(n_samp), 1.0)
    fx = f_samp * cutoff_frequency * 4.0
    fy = f_samp * cutoff_frequency * 4.0
    rho = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    normalized = rho / max(cutoff_frequency, 1e-12)
    clipped = np.clip(normalized, 0.0, 1.0)
    otf = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
    otf[normalized >= 1.0] = 0.0
    return np.fft.ifftshift(otf), fx, fy


def _oi_psf_data(
    oi: OpticalImage,
    wavelength_nm: Any | None = None,
    units: Any | None = "um",
    n_samp: int = 25,
) -> dict[str, Any]:
    optics = dict(oi.fields.get("optics", {}))
    model = param_format(optics.get("model", ""))
    wave = np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    this_wave = _default_plot_wavelength(wave) if wavelength_nm is None else float(np.asarray(wavelength_nm, dtype=float).reshape(-1)[0])

    if model == "diffractionlimited":
        otf, fx, fy = _diffraction_limited_plot_otf(oi, this_wave, n_samp=n_samp)
        psf = np.abs(np.fft.fftshift(np.fft.ifft2(otf)))
        peak_frequency = float(np.max(fx)) if fx.size > 0 else 0.0
        delta_space_m = 1.0 / max(2.0 * peak_frequency, 1e-12) if peak_frequency > 0.0 else 0.0
        samples = np.arange(-int(n_samp), int(n_samp), dtype=float)
        x_axis_m = samples * delta_space_m
        y_axis_m = samples * delta_space_m
        return {"psf": psf, "xy": _support_grid_from_axes(x_axis_m, y_axis_m, units)}

    if model == "shiftinvariant":
        psf_data = optics.get("psf_data")
        if isinstance(psf_data, dict):
            normalized = _normalize_shift_invariant_psf_data(psf_data)
            psf = _plane_at_wavelength(normalized["psf"], normalized["wave"], this_wave)
            sample_spacing_m = float(normalized["sample_spacing_m"])
            x_axis_m = _centered_support_axis(psf.shape[1], sample_spacing_m)
            y_axis_m = _centered_support_axis(psf.shape[0], sample_spacing_m)
            return {"psf": np.asarray(psf, dtype=float), "xy": _support_grid_from_axes(x_axis_m, y_axis_m, units)}

        wavefront = optics.get("wavefront")
        if isinstance(wavefront, dict) and wavefront:
            psf = np.asarray(wvf_get(wavefront, "psf", this_wave), dtype=float)
            x_axis_m = np.asarray(wvf_get(wavefront, "psf spatial samples", "m", this_wave), dtype=float).reshape(-1)
            y_axis_m = x_axis_m.copy()
            return {"psf": psf, "xy": _support_grid_from_axes(x_axis_m, y_axis_m, units)}

        otf_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        if otf_bundle is None:
            raise ValueError("Optical image has no shift-invariant PSF or OTF data available.")
        otf = _interpolate_shift_invariant_otf_wavelength(otf_bundle["OTF"], otf_bundle["wave"], this_wave)
        psf = np.abs(np.fft.fftshift(np.fft.ifft2(otf)))
        x_axis_m = _spatial_axis_from_frequency_support(np.asarray(otf_bundle["fx"], dtype=float), support_unit_to_m=1e-3)
        y_axis_m = _spatial_axis_from_frequency_support(np.asarray(otf_bundle["fy"], dtype=float), support_unit_to_m=1e-3)
        return {"psf": np.asarray(psf, dtype=float), "xy": _support_grid_from_axes(x_axis_m, y_axis_m, units)}

    raise UnsupportedOptionError("oiGet", "optics psf data")


def _oi_psf_axis(
    oi: OpticalImage,
    axis: str,
    wavelength_nm: Any | None = None,
    units: Any | None = "um",
    n_samp: int = 25,
) -> dict[str, Any]:
    psf_data = _oi_psf_data(oi, wavelength_nm, units, n_samp=n_samp)
    psf = np.asarray(psf_data["psf"], dtype=float)
    xy = np.asarray(psf_data["xy"], dtype=float)
    x_axis = np.asarray(xy[0, :, 0], dtype=float)
    y_axis = np.asarray(xy[:, 0, 1], dtype=float)

    if axis == "x":
        row_coord = float(np.interp(0.0, y_axis, np.arange(y_axis.size, dtype=float)))
        data = map_coordinates(
            psf,
            [np.full(x_axis.size, row_coord, dtype=float), np.arange(x_axis.size, dtype=float)],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        return {"samp": x_axis.copy(), "data": np.asarray(data, dtype=float)}

    col_coord = float(np.interp(0.0, x_axis, np.arange(x_axis.size, dtype=float)))
    data = map_coordinates(
        psf,
        [np.arange(y_axis.size, dtype=float), np.full(y_axis.size, col_coord, dtype=float)],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return {"samp": y_axis.copy(), "data": np.asarray(data, dtype=float)}


def _normalize_raytrace_table(raw: dict[str, Any] | None) -> dict[str, Any]:
    current = {} if raw is None else dict(raw)
    return {
        "field_height_mm": np.asarray(
            current.get("field_height_mm", current.get("fieldHeight", np.empty(0, dtype=float))),
            dtype=float,
        ).reshape(-1),
        "wavelength_nm": np.asarray(
            current.get("wavelength_nm", current.get("wavelength", np.empty(0, dtype=float))),
            dtype=float,
        ).reshape(-1),
        "function": np.asarray(current.get("function", np.empty(0, dtype=float)), dtype=float),
    }


def _normalize_raytrace_psf(raw: dict[str, Any] | None) -> dict[str, Any]:
    current = _normalize_raytrace_table(raw)
    source = {} if raw is None else dict(raw)
    current["sample_spacing_mm"] = np.asarray(
        source.get("sample_spacing_mm", source.get("sampleSpacing", np.array([0.0, 0.0], dtype=float))),
        dtype=float,
    ).reshape(-1)
    return current


def _normalize_raytrace_optics(raw: dict[str, Any]) -> dict[str, Any]:
    if "raytrace" in raw and "geometry" in raw["raytrace"]:
        normalized = dict(raw)
        transmittance = dict(normalized.get("transmittance", {}))
        wave = np.asarray(transmittance.get("wave", DEFAULT_WAVE.copy()), dtype=float).reshape(-1)
        scale = np.asarray(transmittance.get("scale", np.ones(wave.size, dtype=float)), dtype=float).reshape(-1)
        if scale.size == 1 and wave.size > 1:
            scale = np.full(wave.size, float(scale[0]), dtype=float)
        if scale.size != wave.size:
            scale = np.ones(wave.size, dtype=float)
        normalized["transmittance"] = {
            "wave": wave.copy(),
            "scale": scale.copy(),
        }
        normalized["raytrace"] = dict(normalized["raytrace"])
        normalized["raytrace"]["geometry"] = _normalize_raytrace_table(normalized["raytrace"].get("geometry"))
        normalized["raytrace"]["relative_illumination"] = _normalize_raytrace_table(
            normalized["raytrace"].get("relative_illumination")
        )
        normalized["raytrace"]["psf"] = _normalize_raytrace_psf(normalized["raytrace"].get("psf"))
        computation = dict(normalized["raytrace"].get("computation", {}))
        psf_spacing_m = computation.get("psf_spacing_m", computation.get("psfSpacing"))
        normalized["raytrace"]["computation"] = {
            "psf_spacing_m": None if psf_spacing_m is None else float(np.asarray(psf_spacing_m).reshape(-1)[0]),
        }
        effective_focal_length_m = _scalar(
            normalized.get("focal_length_m", normalized["raytrace"].get("effective_focal_length_m")),
            DEFAULT_FOCAL_LENGTH_M,
        )
        normalized["model"] = "raytrace"
        normalized["name"] = str(normalized.get("name", normalized["raytrace"].get("name", "raytrace")))
        normalized["f_number"] = _scalar(
            normalized.get("f_number", normalized.get("fNumber", normalized["raytrace"].get("f_number"))),
            4.0,
        )
        normalized["focal_length_m"] = effective_focal_length_m
        normalized["nominal_focal_length_m"] = _scalar(
            normalized.get(
                "nominal_focal_length_m",
                normalized.get("nominalFocalLength", normalized.get("focalLength", effective_focal_length_m)),
            ),
            effective_focal_length_m,
        )
        normalized["compute_method"] = str(
            normalized.get("compute_method", normalized.get("computeMethod", ""))
        )
        normalized["aberration_scale"] = float(
            normalized.get("aberration_scale", normalized.get("aberrationScale", 0.0))
        )
        normalized["offaxis_method"] = str(
            normalized.get("offaxis_method", normalized.get("offaxisMethod", normalized.get("offaxis", "skip")))
        )
        return normalized

    raytrace = dict(raw.get("rayTrace", raw.get("raytrace", {})))
    transmittance = dict(raw.get("transmittance", {}))
    transmittance_wave = np.asarray(transmittance.get("wave", DEFAULT_WAVE.copy()), dtype=float).reshape(-1)
    transmittance_scale = np.asarray(
        transmittance.get("scale", np.ones(transmittance_wave.size, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if transmittance_scale.size == 1 and transmittance_wave.size > 1:
        transmittance_scale = np.full(transmittance_wave.size, float(transmittance_scale[0]), dtype=float)
    if transmittance_scale.size != transmittance_wave.size:
        transmittance_scale = np.ones(transmittance_wave.size, dtype=float)

    effective_focal_length_raw = raytrace.get("effectiveFocalLength")
    if effective_focal_length_raw is None:
        effective_focal_length_m = _scalar(
            raytrace.get("effective_focal_length_m"),
            DEFAULT_FOCAL_LENGTH_M,
        )
    else:
        effective_focal_length_m = _scalar(effective_focal_length_raw, DEFAULT_FOCAL_LENGTH_M * 1e3) / 1e3
    nominal_f_number = _scalar(
        raytrace.get("fNumber", raytrace.get("f_number", raw.get("fNumber", raw.get("f_number")))),
        4.0,
    )
    reference_wavelength_nm = _scalar(
        raytrace.get("referenceWavelength", raytrace.get("reference_wavelength_nm")),
        DEFAULT_WVF_MEASURED_WAVELENGTH_NM,
    )
    object_distance_raw = raytrace.get("objectDistance")
    if object_distance_raw is None:
        object_distance_m = _scalar(raytrace.get("object_distance_m"), np.inf)
    else:
        object_distance_m = _scalar(object_distance_raw, np.inf) / 1e3
    magnification = _scalar(raytrace.get("mag", raytrace.get("magnification")), 0.0)
    effective_f_number = _scalar(
        raytrace.get("effectiveFNumber", raytrace.get("effective_f_number")),
        nominal_f_number,
    )
    max_fov_deg = _scalar(
        raytrace.get("maxfov", raytrace.get("fov", raytrace.get("max_fov_deg"))),
        np.inf,
    )
    computation = dict(raytrace.get("computation", {}))
    psf_spacing_m = computation.get("psfSpacing", computation.get("psf_spacing_m"))
    return {
        "model": "raytrace",
        "name": str(raw.get("name", raytrace.get("name", "raytrace"))),
        "f_number": nominal_f_number,
        "focal_length_m": effective_focal_length_m,
        "nominal_focal_length_m": _scalar(
            raw.get("nominal_focal_length_m", raw.get("nominalFocalLength", raw.get("focalLength"))),
            DEFAULT_FOCAL_LENGTH_M,
        ),
        "compute_method": "",
        "aberration_scale": 0.0,
        "offaxis_method": "skip",
        "transmittance": {
            "wave": transmittance_wave.copy(),
            "scale": transmittance_scale.copy(),
        },
        "raytrace": {
            "program": str(raytrace.get("program", "")),
            "lens_file": str(raytrace.get("lensFile", raytrace.get("lens_file", ""))),
            "reference_wavelength_nm": reference_wavelength_nm,
            "object_distance_m": object_distance_m,
            "magnification": magnification,
            "f_number": nominal_f_number,
            "effective_focal_length_m": effective_focal_length_m,
            "effective_f_number": effective_f_number,
            "max_fov_deg": max_fov_deg,
            "geometry": _normalize_raytrace_table(raytrace.get("geometry")),
            "relative_illumination": _normalize_raytrace_table(raytrace.get("relIllum", raytrace.get("relative_illumination"))),
            "psf": _normalize_raytrace_psf(raytrace.get("psf")),
            "computation": {
                "psf_spacing_m": None if psf_spacing_m is None else float(np.asarray(psf_spacing_m).reshape(-1)[0]),
            },
            "blocks_per_field_height": int(
                raytrace.get("blocksPerFieldHeight", raytrace.get("blocks_per_field_height", 4))
            ),
            "name": str(raytrace.get("name", raw.get("name", "raytrace"))),
        },
    }


def _raytrace_struct_uses_normalized_keys(raytrace: dict[str, Any]) -> bool:
    if any(
        key in raytrace
        for key in (
            "f_number",
            "lens_file",
            "magnification",
            "reference_wavelength_nm",
            "object_distance_m",
            "effective_focal_length_m",
            "effective_f_number",
            "max_fov_deg",
            "blocks_per_field_height",
            "relative_illumination",
        )
    ):
        return True
    psf = raytrace.get("psf")
    if isinstance(psf, dict) and any(key in psf for key in ("field_height_mm", "wavelength_nm", "sample_spacing_mm")):
        return True
    geometry = raytrace.get("geometry")
    if isinstance(geometry, dict) and any(key in geometry for key in ("field_height_mm", "wavelength_nm")):
        return True
    relative_illumination = raytrace.get("relative_illumination")
    if isinstance(relative_illumination, dict) and any(
        key in relative_illumination for key in ("field_height_mm", "wavelength_nm")
    ):
        return True
    computation = raytrace.get("computation")
    if isinstance(computation, dict) and "psf_spacing_m" in computation:
        return True
    return False


def _merge_mapping(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _normalize_transmittance_update(source: dict[str, Any], current: dict[str, Any] | None) -> dict[str, Any]:
    current_transmittance = dict(current or {})
    current_wave = np.asarray(current_transmittance.get("wave", DEFAULT_WAVE.copy()), dtype=float).reshape(-1)
    current_scale = np.asarray(
        current_transmittance.get("scale", np.ones(current_wave.size, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if current_scale.size == 1 and current_wave.size > 1:
        current_scale = np.full(current_wave.size, float(current_scale[0]), dtype=float)
    elif current_scale.size != current_wave.size:
        current_scale = np.ones(current_wave.size, dtype=float)

    wave = np.asarray(source.get("wave", current_wave), dtype=float).reshape(-1)
    if "scale" in source:
        source_scale = np.asarray(source["scale"], dtype=float).reshape(-1)
        unchanged_exported_scale = (
            source_scale.size == current_scale.size
            and np.allclose(source_scale, current_scale)
        )
        if unchanged_exported_scale:
            scale = np.interp(wave, current_wave, current_scale) if current_wave.size > 0 else np.ones(wave.size, dtype=float)
        else:
            scale = source_scale
            if scale.size != wave.size:
                raise ValueError("Transmittance must match wave dimension.")
            if np.any((scale < 0.0) | (scale > 1.0)):
                raise ValueError("Transmittance should be in [0, 1].")
    elif current_wave.size > 0 and current_scale.size == current_wave.size:
        scale = np.interp(wave, current_wave, current_scale)
    else:
        scale = np.ones(wave.size, dtype=float)
    return {
        "wave": wave.copy(),
        "scale": scale.copy(),
    }


def _normalize_optics_update(value: Any, current_optics: dict[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    model = param_format(raw.get("model", current_optics.get("model", "")))
    is_raytrace = (
        model == "raytrace"
        or "rayTrace" in raw
        or ("raytrace" in raw and isinstance(raw["raytrace"], dict))
    )
    if not is_raytrace:
        normalized = dict(raw)
        psf_data = normalized.get("psf_data", normalized.get("psfData"))
        if psf_data is not None:
            normalized["psf_data"] = _normalize_shift_invariant_psf_data(psf_data)
            normalized.update(_custom_shift_invariant_otf_bundle(normalized["psf_data"]))
            normalized.pop("psfData", None)
            normalized.setdefault("model", current_optics.get("model", "shiftinvariant"))
            normalized.setdefault(
                "compute_method",
                raw.get(
                    "compute_method",
                    raw.get("computeMethod", current_optics.get("compute_method", "opticsotf")),
                ),
            )
        return normalized

    source = dict(raw)
    source.setdefault("model", "raytrace")
    source.setdefault("name", current_optics.get("name", current_optics.get("raytrace", {}).get("name", "raytrace")))
    if "fNumber" not in source and "f_number" not in source:
        source["fNumber"] = current_optics.get("f_number", 4.0)
    if (
        "focalLength" not in source
        and "nominal_focal_length_m" not in source
        and "focal_length_m" not in source
    ):
        source["focalLength"] = current_optics.get(
            "nominal_focal_length_m",
            current_optics.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M),
        )
    if "transmittance" not in source and "transmittance" in current_optics:
        source["transmittance"] = dict(current_optics["transmittance"])
    elif isinstance(source.get("transmittance"), dict):
        source["transmittance"] = _normalize_transmittance_update(
            dict(source["transmittance"]),
            dict(current_optics.get("transmittance", {})),
        )

    nested_raw_raytrace = source.get("rayTrace")
    if isinstance(nested_raw_raytrace, dict):
        current_exported_raytrace = _export_raytrace(current_optics.get("raytrace", {}))
        merged_raw_raytrace = _merge_mapping(current_exported_raytrace, nested_raw_raytrace)
        top_level_f_number = source.get("f_number", source.get("fNumber"))
        nested_fnumber = nested_raw_raytrace.get("fNumber")
        current_nested_fnumber = current_exported_raytrace.get("fNumber")
        if top_level_f_number is not None and (
            "fNumber" not in nested_raw_raytrace
            or (
                nested_fnumber is not None
                and current_nested_fnumber is not None
                and np.isclose(float(nested_fnumber), float(current_nested_fnumber))
            )
        ):
            merged_raw_raytrace["fNumber"] = float(top_level_f_number)
        top_level_effective_focal_length = source.get("focal_length_m", source.get("focalLength"))
        nested_effective_focal_length = nested_raw_raytrace.get("effectiveFocalLength")
        current_nested_effective_focal_length = current_exported_raytrace.get("effectiveFocalLength")
        if top_level_effective_focal_length is not None and (
            "effectiveFocalLength" not in nested_raw_raytrace
            or (
                nested_effective_focal_length is not None
                and current_nested_effective_focal_length is not None
                and np.isclose(
                    float(nested_effective_focal_length),
                    float(current_nested_effective_focal_length),
                )
            )
        ):
            merged_raw_raytrace["effectiveFocalLength"] = float(top_level_effective_focal_length) * 1e3
        source["rayTrace"] = merged_raw_raytrace

    nested_raytrace = source.get("raytrace")
    if isinstance(nested_raytrace, dict):
        if _raytrace_struct_uses_normalized_keys(nested_raytrace):
            source["raytrace"] = _merge_mapping(dict(current_optics.get("raytrace", {})), nested_raytrace)
        else:
            source["rayTrace"] = dict(nested_raytrace)
            source.pop("raytrace", None)

    normalized = _normalize_raytrace_optics(source)
    normalized["compute_method"] = str(
        raw.get("compute_method", raw.get("computeMethod", current_optics.get("compute_method", normalized.get("compute_method", ""))))
    )
    normalized["aberration_scale"] = float(
        raw.get(
            "aberration_scale",
            raw.get("aberrationScale", current_optics.get("aberration_scale", normalized.get("aberration_scale", 0.0))),
        )
    )
    normalized["offaxis_method"] = str(
        raw.get(
            "offaxis_method",
            raw.get(
                "offaxisMethod",
                raw.get("offaxis", current_optics.get("offaxis_method", normalized.get("offaxis_method", "skip"))),
            ),
        )
    )
    return normalized


def _normalize_raytrace_update(value: Any, current_optics: dict[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    if "raytrace" in raw or "rayTrace" in raw or "model" in raw:
        return _normalize_optics_update(raw, current_optics)

    source: dict[str, Any] = {
        "model": "raytrace",
        "name": current_optics.get("name", current_optics.get("raytrace", {}).get("name", "raytrace")),
        "fNumber": current_optics.get("f_number", 4.0),
        "focalLength": current_optics.get(
            "nominal_focal_length_m",
            current_optics.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M),
        ),
        "transmittance": dict(current_optics.get("transmittance", {})),
        "compute_method": current_optics.get("compute_method", ""),
        "aberration_scale": current_optics.get("aberration_scale", 0.0),
        "offaxis_method": current_optics.get("offaxis_method", "skip"),
    }
    if "name" in raw:
        source["name"] = raw["name"]
    if "fNumber" in raw:
        source["fNumber"] = raw["fNumber"]
    if _raytrace_struct_uses_normalized_keys(raw):
        source["raytrace"] = raw
    else:
        source["rayTrace"] = raw
    return _normalize_optics_update(source, current_optics)


def _export_raytrace_table(table: dict[str, Any], *, include_sample_spacing: bool = False) -> dict[str, Any]:
    current = dict(table)
    exported: dict[str, Any] = {
        "fieldHeight": np.asarray(current.get("field_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy(),
        "wavelength": np.asarray(current.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy(),
        "function": np.asarray(current.get("function", np.empty(0, dtype=float)), dtype=float).copy(),
    }
    if include_sample_spacing:
        exported["sampleSpacing"] = np.asarray(
            current.get("sample_spacing_mm", np.empty(0, dtype=float)),
            dtype=float,
        ).reshape(-1).copy()
    return exported


def _export_raytrace(raytrace: dict[str, Any]) -> dict[str, Any]:
    current = dict(raytrace)
    exported: dict[str, Any] = {
        "name": str(current.get("name", "")),
        "program": str(current.get("program", "")),
        "lensFile": str(current.get("lens_file", "")),
        "referenceWavelength": float(current.get("reference_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM)),
        "objectDistance": float(current.get("object_distance_m", np.inf)) * 1e3,
        "mag": float(current.get("magnification", 0.0)),
        "fNumber": float(current.get("f_number", np.nan)),
        "effectiveFocalLength": float(current.get("effective_focal_length_m", np.nan)) * 1e3,
        "effectiveFNumber": float(current.get("effective_f_number", np.nan)),
        "fov": float(current.get("max_fov_deg", np.inf)),
        "geometry": _export_raytrace_table(current.get("geometry", {})),
        "relIllum": _export_raytrace_table(current.get("relative_illumination", {})),
        "psf": _export_raytrace_table(current.get("psf", {}), include_sample_spacing=True),
    }
    computation = dict(current.get("computation", {}))
    if "psf_spacing_m" in computation:
        exported["computation"] = {"psfSpacing": computation.get("psf_spacing_m")}
    if "blocks_per_field_height" in current:
        exported["blocksPerFieldHeight"] = int(current.get("blocks_per_field_height", 4))
    return exported


def _export_optics(optics: dict[str, Any]) -> dict[str, Any]:
    current = dict(optics)
    raytrace = dict(current.get("raytrace", {}))
    export_f_number = float(current.get("f_number", np.nan))
    export_focal_length = float(current.get("focal_length_m", np.nan))
    if param_format(current.get("model", "")) == "raytrace":
        export_f_number = float(raytrace.get("f_number", export_f_number))
        export_focal_length = float(raytrace.get("effective_focal_length_m", export_focal_length))
    exported: dict[str, Any] = {
        "model": str(current.get("model", "")),
        "name": str(current.get("name", "")),
        "fNumber": export_f_number,
        "focalLength": export_focal_length,
        "computeMethod": str(current.get("compute_method", "")),
        "aberrationScale": float(current.get("aberration_scale", 0.0)),
        "offaxis": str(current.get("offaxis_method", "skip")),
    }
    if "nominal_focal_length_m" in current:
        exported["nominalFocalLength"] = float(current.get("nominal_focal_length_m", np.nan))
    transmittance = current.get("transmittance")
    if isinstance(transmittance, dict):
        exported["transmittance"] = {
            "wave": np.asarray(transmittance.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy(),
            "scale": np.asarray(transmittance.get("scale", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy(),
        }
    if isinstance(current.get("wavefront"), dict):
        exported["wavefront"] = dict(current.get("wavefront", {}))
    if isinstance(current.get("raytrace"), dict):
        exported["rayTrace"] = _export_raytrace(current.get("raytrace", {}))
    return exported


def _export_psf_cell_array(psf_stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(psf_stack, dtype=float)
    if stack.ndim != 5:
        return np.empty((0, 0, 0), dtype=object)
    exported = np.empty(stack.shape[:3], dtype=object)
    for index in np.ndindex(exported.shape):
        exported[index] = np.asarray(stack[index], dtype=float).copy()
    return exported


def _export_psf_struct(psf_struct: dict[str, Any]) -> dict[str, Any]:
    current = dict(psf_struct)
    exported: dict[str, Any] = {}
    if "psf" in current:
        exported["psf"] = _export_psf_cell_array(np.asarray(current.get("psf"), dtype=float))
    if "sample_angles_deg" in current:
        exported["sampAngles"] = np.asarray(current.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy()
    if "img_height_mm" in current:
        exported["imgHeight"] = np.asarray(current.get("img_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy() / 1e3
    if "wavelength_nm" in current:
        exported["wavelength"] = np.asarray(current.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1).copy()
    if "optics_name" in current:
        exported["opticsName"] = str(current.get("optics_name", ""))
    return exported


def _load_raytrace_optics(source: Any, *, asset_store: AssetStore) -> dict[str, Any]:
    if source is None:
        raw = asset_store.load_mat("data/optics/rtZemaxExample.mat")["optics"]
        return _normalize_raytrace_optics(_mat_to_native(raw))
    if isinstance(source, dict):
        return _normalize_raytrace_optics(source)

    path = Path(source)
    if path.is_dir():
        try:
            params_path = _resolve_isetparams_path(path)
        except ValueError as error:
            raise UnsupportedOptionError("oiCreate", f"ray trace optics {source}") from error
        imported, _ = rt_import_data(p_file_full=params_path)
        return imported
    if path.is_absolute() or path.exists():
        if path.suffix.lower() == ".txt":
            imported, _ = rt_import_data(p_file_full=path)
            return imported
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)["optics"]
        return _normalize_raytrace_optics(_mat_to_native(raw))

    candidates = [Path(str(source))]
    if not str(source).lower().endswith(".mat"):
        candidates.append(Path("data/optics") / f"{source}.mat")
    candidates.append(Path("data/optics") / str(source))
    for candidate in candidates:
        try:
            raw = asset_store.load_mat(candidate)["optics"]
            return _normalize_raytrace_optics(_mat_to_native(raw))
        except Exception:
            continue
    raise UnsupportedOptionError("oiCreate", f"ray trace optics {source}")


def _parse_matlab_range(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) == 2:
        start = float(parts[0])
        step = 1.0
        stop = float(parts[1])
    elif len(parts) == 3:
        start = float(parts[0])
        step = float(parts[1])
        stop = float(parts[2])
    else:
        raise ValueError(f"Unsupported MATLAB range syntax: {text}")
    if np.isclose(step, 0.0):
        raise ValueError("MATLAB range step must be non-zero.")
    limit = stop + (0.5 * step)
    return np.arange(start, limit, step, dtype=float)


def _parse_matlab_numeric_sequence(body: str) -> Any:
    stripped = body.strip()
    if ":" in stripped and "," not in stripped and ";" not in stripped:
        return _parse_matlab_range(stripped)
    array = np.fromstring(
        stripped.replace(",", " ").replace(";", " ").replace("\n", " ").replace("\t", " "),
        sep=" ",
        dtype=float,
    )
    if array.size == 1:
        return float(array[0])
    return array


def _parse_matlab_value(text: str) -> Any:
    value = text.strip().rstrip(";").strip()
    if not value:
        return None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.endswith("'") and not value.startswith("'"):
        transposed = value[:-1].strip()
        if (transposed.startswith("[") and transposed.endswith("]")) or (
            transposed.startswith("(") and transposed.endswith(")")
        ):
            return _parse_matlab_numeric_sequence(transposed[1:-1])
    if value.startswith("[") and value.endswith("]"):
        return _parse_matlab_numeric_sequence(value[1:-1])
    if value.startswith("(") and value.endswith(")"):
        return _parse_matlab_numeric_sequence(value[1:-1])
    if ":" in value and all(ch not in value for ch in "[]'"):
        return _parse_matlab_range(value)
    try:
        return float(value)
    except ValueError:
        return value


def _read_isetparams(path: str | Path) -> dict[str, Any]:
    ascii_text = Path(path).read_bytes().decode("latin1", errors="ignore")
    ascii_text = "".join(ch for ch in ascii_text if 0 < ord(ch) < 128)
    params: dict[str, Any] = {}
    for statement in _iter_matlab_assignments(ascii_text):
        match = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+?)\s*;?$", statement, flags=re.S)
        if match is None:
            continue
        params[match.group(1)] = _parse_matlab_value(match.group(2))
    return params


def _strip_matlab_comment(line: str) -> str:
    in_string = False
    result: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "'":
            result.append(char)
            if in_string and index + 1 < len(line) and line[index + 1] == "'":
                result.append("'")
                index += 2
                continue
            if in_string:
                in_string = False
            elif _matlab_apostrophe_starts_string(line, index):
                in_string = True
            index += 1
            continue
        if char == "%" and not in_string:
            break
        result.append(char)
        index += 1
    return "".join(result)


def _matlab_apostrophe_starts_string(text: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    if previous < 0:
        return True
    return text[previous] in "=([{,:;"


def _matlab_line_starts_assignment(line: str) -> bool:
    return re.match(r"^[A-Za-z_]\w*\s*=", line) is not None


def _iter_matlab_assignments(text: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    in_string = False
    continuing_statement = False

    for raw_line in text.splitlines():
        line = _strip_matlab_comment(raw_line).rstrip()
        if not line:
            continue
        if (
            current
            and not continuing_statement
            and bracket_depth == 0
            and not in_string
            and _matlab_line_starts_assignment(line)
        ):
            statements.append("".join(current).strip())
            current = []
        continued = line.endswith("...")
        if continued:
            line = line[:-3].rstrip()
        current.append(line)
        if continued:
            current.append(" ")
            continuing_statement = True
            continue

        for index, char in enumerate(line):
            if char == "'":
                if in_string and index + 1 < len(line) and line[index + 1] == "'":
                    continue
                if in_string:
                    in_string = False
                elif _matlab_apostrophe_starts_string(line, index):
                    in_string = True
            elif not in_string:
                if char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth = max(0, bracket_depth - 1)

        joined = "".join(current).strip()
        if joined and bracket_depth == 0 and not in_string and joined.endswith(";"):
            statements.append(joined)
            current = []
        continuing_statement = False

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _raytrace_lens_basename(lens_file: str | Path) -> str:
    text = str(lens_file).strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/").rstrip("/")
    base = normalized.rsplit("/", 1)[-1]
    stem = Path(base).stem
    return stem or base


def _resolve_isetparams_path(source: str | Path) -> Path:
    path = Path(source)
    if path.is_dir():
        matches = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".txt" and "isetpar" in candidate.name.lower()
        )
        if not matches:
            raise ValueError(f"Unable to locate an ISETPARAMS file in {path}")
        return matches[0]
    return path


def rt_file_names(
    lens_file: str | Path,
    wave: np.ndarray | list[float] | tuple[float, ...],
    img_height: np.ndarray | list[float] | tuple[float, ...],
    *,
    directory: str | Path | None = None,
) -> tuple[str, str, np.ndarray, str]:
    base = _raytrace_lens_basename(lens_file)
    prefix = Path(directory) if directory is not None else None

    def _resolve(name: str) -> str:
        return str((prefix / name) if prefix is not None else name)

    wave_samples = np.asarray(wave, dtype=float).reshape(-1)
    height_samples = np.asarray(img_height, dtype=float).reshape(-1)
    di_name = _resolve(f"{base}_DI_.dat")
    ri_name = _resolve(f"{base}_RI_.dat")
    cra_name = _resolve(f"{base}_CRA_.dat")
    psf_name_list = np.empty((height_samples.size, wave_samples.size), dtype=object)
    for height_index in range(height_samples.size):
        for wave_index in range(wave_samples.size):
            psf_name_list[height_index, wave_index] = _resolve(
                f"{base}_2D_PSF_Fld{height_index + 1:d}_Wave{wave_index + 1:d}.dat"
            )
    return di_name, ri_name, psf_name_list, cra_name


def rt_root_path() -> str:
    return str(ensure_upstream_snapshot() / "opticalimage" / "raytrace")


def rt_image_rotate(isource: Any, theta: float) -> np.ndarray:
    source = np.asarray(isource, dtype=float)
    if source.ndim != 2:
        raise ValueError("rtImageRotate expects a 2D source image.")

    rows, cols = source.shape
    row_center = (rows + 1.0) / 2.0
    col_center = (cols + 1.0) / 2.0
    theta_rad = np.deg2rad(float(theta))
    cos_theta = float(np.cos(theta_rad))
    sin_theta = float(np.sin(theta_rad))

    rotated_image = np.zeros((rows, cols), dtype=float)
    for row_index in range(1, rows + 1):
        for col_index in range(1, cols + 1):
            source_row = cos_theta * (row_index - row_center) + sin_theta * (col_index - col_center) + row_center
            source_col = -sin_theta * (row_index - row_center) + cos_theta * (col_index - col_center) + col_center
            if 1.0 < source_row < rows and 1.0 < source_col < cols:
                row_floor = int(np.floor(source_row))
                col_floor = int(np.floor(source_col))
                row_alpha = source_row - row_floor
                col_alpha = source_col - col_floor
                top = source[row_floor - 1, col_floor - 1] * (1.0 - col_alpha) + source[row_floor - 1, col_floor] * col_alpha
                bottom = source[row_floor, col_floor - 1] * (1.0 - col_alpha) + source[row_floor, col_floor] * col_alpha
                rotated_image[row_index - 1, col_index - 1] = (1.0 - row_alpha) * top + row_alpha * bottom
    return rotated_image


def zemax_read_header(fname: str | Path) -> tuple[float, float]:
    text = Path(fname).read_bytes().decode("latin1", errors="ignore")
    ascii_text = "".join(ch for ch in text if 0 < ord(ch) < 128)
    spacing_match = re.search(r"spacing is\s+([-+0-9.eE]+)", ascii_text)
    area_match = re.search(r"area is\s+([-+0-9.eE]+)", ascii_text)
    if spacing_match is None or area_match is None:
        raise ValueError(f"Unable to parse Zemax header in {fname}")
    return float(spacing_match.group(1)), float(area_match.group(1))


def zemax_load(f_name: str | Path, psf_size: int) -> np.ndarray:
    text = Path(f_name).read_bytes().decode("latin1", errors="ignore")
    ascii_text = "".join(ch for ch in text if 0 < ord(ch) < 128)
    marker = "normalized."
    start = ascii_text.lower().find(marker)
    if start < 0:
        raise ValueError(f"Unable to locate normalized PSF data in {f_name}")
    data_text = ascii_text[start + len(marker) :]
    values = np.fromstring(data_text, sep=" ", dtype=float)
    expected = int(psf_size) * int(psf_size)
    if values.size < expected:
        raise ValueError(f"Expected {expected} PSF values in {f_name}, found {values.size}")
    data = values[:expected].reshape(int(psf_size), int(psf_size))
    return np.rot90(data)


def rt_import_data(
    optics: OpticalImage | dict[str, Any] | None = None,
    rt_program: str = "zemax",
    p_file_full: str | Path | None = None,
) -> tuple[dict[str, Any], None]:
    if param_format(rt_program) != "zemax":
        raise UnsupportedOptionError("rtImportData", rt_program)
    if p_file_full is None:
        raise ValueError("p_file_full is required for rtImportData.")
    p_file_full = _resolve_isetparams_path(p_file_full)

    params = _read_isetparams(p_file_full)
    required = (
        "lensFile",
        "wave",
        "imgHeightNum",
        "imgHeightMax",
        "objDist",
        "mag",
        "refWave",
        "fov",
        "efl",
        "fnumber_eff",
        "fnumber",
    )
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing Zemax parameter(s): {', '.join(missing)}")
    if "psfSize" in params:
        parameter_psf_size = int(round(float(params["psfSize"])))
        if parameter_psf_size % 2:
            raise ValueError("PSF size must be even.")

    wave = np.asarray(params["wave"], dtype=float).reshape(-1)
    n_height = int(round(float(params["imgHeightNum"])))
    img_height = np.linspace(0.0, float(params["imgHeightMax"]), n_height, dtype=float)
    directory = Path(p_file_full).resolve().parent
    base_name = _raytrace_lens_basename(params.get("baseLensFileName", params["lensFile"]))
    di_name, ri_name, psf_name_list, _ = rt_file_names(base_name, wave, img_height, directory=directory)

    geometry_values = np.fromstring(Path(di_name).read_text(encoding="latin1", errors="ignore"), sep=" ", dtype=float)
    rel_illum_values = np.fromstring(Path(ri_name).read_text(encoding="latin1", errors="ignore"), sep=" ", dtype=float)
    expected_table = n_height * wave.size
    if geometry_values.size != expected_table:
        raise ValueError(f"Expected {expected_table} geometry values in {di_name}, found {geometry_values.size}")
    if rel_illum_values.size != expected_table:
        raise ValueError(f"Expected {expected_table} relative-illumination values in {ri_name}, found {rel_illum_values.size}")

    psf_spacing_um, psf_area_um = zemax_read_header(psf_name_list[0, 0])
    psf_size = int(round(psf_area_um / max(psf_spacing_um, 1e-12)))
    if not np.isclose(psf_size * psf_spacing_um, psf_area_um):
        raise ValueError("Zemax PSF header spacing and area imply a non-integer PSF size.")

    psf_function = np.zeros((psf_size, psf_size, n_height, wave.size), dtype=float)
    for height_index in range(n_height):
        for wave_index in range(wave.size):
            file_name = psf_name_list[height_index, wave_index]
            spacing_um, area_um = zemax_read_header(file_name)
            if not (np.isclose(spacing_um, psf_spacing_um) and np.isclose(area_um, psf_area_um)):
                raise ValueError(f"Inconsistent PSF header in {file_name}")
            kernel = zemax_load(file_name, psf_size)
            kernel_sum = float(np.sum(kernel))
            if kernel_sum > 0.0 and not np.isclose(kernel_sum, 1.0):
                kernel = kernel / kernel_sum
            psf_function[:, :, height_index, wave_index] = kernel

    current: dict[str, Any] = {}
    if optics is not None:
        current = dict(optics.fields.get("optics", {})) if isinstance(optics, OpticalImage) else dict(optics)
    current_raytrace = dict(current.get("raytrace", current.get("rayTrace", {})))
    current_computation = dict(current_raytrace.get("computation", {}))
    current_psf_spacing_m = current_computation.get("psf_spacing_m", current_computation.get("psfSpacing"))
    current_name = current.get("name", current_raytrace.get("name", base_name))
    current_raytrace_name = current_raytrace.get("name", current.get("name", base_name))
    current_blocks_per_field_height = current_raytrace.get(
        "blocks_per_field_height",
        current_raytrace.get("blocksPerFieldHeight"),
    )
    current_compute_method = current.get("compute_method", current.get("computeMethod", ""))
    current_aberration_scale = current.get("aberration_scale", current.get("aberrationScale"))
    current_offaxis_method = current.get("offaxis_method", current.get("offaxisMethod", current.get("offaxis")))
    effective_focal_length_m = float(params["efl"]) / 1e3
    effective_f_number = float(params["fnumber_eff"])

    raw_optics = {
        "name": str(current_name),
        "model": "raytrace",
        "transmittance": current.get(
            "transmittance",
            {
                "wave": wave.copy(),
                "scale": np.ones(wave.size, dtype=float),
            },
        ),
        "rayTrace": {
            "program": str(rt_program),
            "lensFile": str(params["lensFile"]),
            "referenceWavelength": float(params["refWave"]),
            "objectDistance": float(params["objDist"]),
            "mag": -abs(float(params["mag"])),
            "fNumber": float(params["fnumber"]),
            "effectiveFocalLength": float(params["efl"]),
            "effectiveFNumber": float(params["fnumber_eff"]),
            "maxfov": float(params["fov"]) * 2.0,
            "name": str(current_raytrace_name),
            "geometry": {
                "function": geometry_values.reshape(wave.size, n_height).T,
                "fieldHeight": img_height.copy(),
                "wavelength": wave.copy(),
            },
            "relIllum": {
                "function": rel_illum_values.reshape(wave.size, n_height).T,
                "fieldHeight": img_height.copy(),
                "wavelength": wave.copy(),
            },
            "psf": {
                "function": psf_function,
                "fieldHeight": img_height.copy(),
                "sampleSpacing": np.array([psf_spacing_um, psf_spacing_um], dtype=float) / 1e3,
                "wavelength": wave.copy(),
            },
            "computation": {
                "psfSpacing": (
                    float(np.asarray(params["psfSpacing"]).reshape(-1)[0]) / 1e3
                    if params.get("psfSpacing") is not None
                    else (
                        None
                        if current_psf_spacing_m is None
                        else float(np.asarray(current_psf_spacing_m).reshape(-1)[0])
                    )
                ),
            },
        },
    }
    normalized = _normalize_raytrace_optics(raw_optics)
    normalized["name"] = str(current_name if current_name is not None else normalized.get("name", base_name))
    normalized["focal_length_m"] = effective_focal_length_m
    normalized["f_number"] = effective_f_number
    normalized["compute_method"] = str(current_compute_method or normalized.get("compute_method", ""))
    normalized["aberration_scale"] = float(
        normalized.get("aberration_scale", 0.0) if current_aberration_scale is None else current_aberration_scale
    )
    normalized["offaxis_method"] = str(current_offaxis_method or normalized.get("offaxis_method", "skip"))
    if current_blocks_per_field_height is not None:
        normalized["raytrace"]["blocks_per_field_height"] = int(current_blocks_per_field_height)
    if "transmittance" in current:
        normalized["transmittance"] = dict(current["transmittance"])
    return normalized, None


def _synthetic_raytrace_general(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = {} if raw is None else dict(raw)
    normalized_keys = {
        "program": "program",
        "lens_file": "lensFile",
        "reference_wavelength_nm": "referenceWavelength",
        "object_distance_m": "objectDistance",
        "magnification": "mag",
        "f_number": "fNumber",
        "effective_focal_length_m": "effectiveFocalLength",
        "effective_f_number": "effectiveFNumber",
        "max_fov_deg": "maxfov",
        "name": "name",
    }
    converted = dict(source)
    for old_key, new_key in normalized_keys.items():
        if old_key in converted and new_key not in converted:
            value = converted.pop(old_key)
            if old_key in {"object_distance_m", "effective_focal_length_m"}:
                value = float(value) * 1e3
            converted[new_key] = value

    general = {
        "program": "Zemax",
        "lensFile": "Synthetic Gaussian",
        "referenceWavelength": 500.0,
        "objectDistance": 10.0,
        "mag": 0.10,
        "fNumber": 4.8,
        "effectiveFocalLength": 3.0,
        "effectiveFNumber": 4.2,
        "maxfov": 30.0,
        "name": "Synthetic Gaussian",
    }
    general.update(converted)
    return general


def _synthetic_binormal(x_spread: float, y_spread: float, samples: int = 128) -> np.ndarray:
    x_kernel = _gaussian_kernel_1d(int(samples), float(x_spread))
    y_kernel = _gaussian_kernel_1d(int(samples), float(y_spread))
    kernel = y_kernel[:, None] * x_kernel[None, :]
    kernel_sum = float(np.sum(kernel))
    if kernel_sum > 0.0:
        kernel = kernel / kernel_sum
    return kernel


def rt_synthetic(
    oi: OpticalImage | None = None,
    ray_trace: dict[str, Any] | None = None,
    spread_limits: tuple[float, float] = (1.0, 4.0),
    xy_ratio: float = 1.0,
) -> dict[str, Any]:
    if len(spread_limits) != 2:
        raise ValueError("spread_limits must contain [min_spread, max_spread].")
    del oi
    current_wave = np.array([450.0, 550.0, 650.0], dtype=float)

    general = _synthetic_raytrace_general(ray_trace)
    field_height_mm = np.arange(0.0, 1.000001, 0.05, dtype=float)

    d = field_height_mm[1] * (field_height_mm / field_height_mm[1]) ** 0.85
    geometry = {
        "function": np.repeat(d[:, None], current_wave.size, axis=1),
        "fieldHeight": field_height_mm.copy(),
        "wavelength": current_wave.copy(),
    }
    rel_illum = {
        "function": np.repeat((1.0 - (field_height_mm / (10.0 * field_height_mm[-1])) ** 0.85)[:, None], current_wave.size, axis=1),
        "fieldHeight": field_height_mm.copy(),
        "wavelength": current_wave.copy(),
    }

    samples = 128
    spread = float(spread_limits[1]) - float(spread_limits[0])
    norm_fh = field_height_mm / max(field_height_mm[-1], 1e-12)
    x_spread = 4.0 * (float(spread_limits[0]) + norm_fh * spread)
    y_spread = float(xy_ratio) * (x_spread * (1.0 + norm_fh))

    psf_function = np.zeros((samples, samples, field_height_mm.size, current_wave.size), dtype=float)
    for height_index in range(field_height_mm.size):
        kernel = _synthetic_binormal(float(x_spread[height_index]), float(y_spread[height_index]), samples=samples)
        for wave_index in range(current_wave.size):
            psf_function[:, :, height_index, wave_index] = kernel

    raw_optics = {
        "name": str(general.get("name", "Synthetic Gaussian")),
        "model": "raytrace",
        "transmittance": {
            "wave": current_wave.copy(),
            "scale": np.ones(current_wave.size, dtype=float),
        },
        "rayTrace": {
            **general,
            "geometry": geometry,
            "relIllum": rel_illum,
            "psf": {
                "function": psf_function,
                "fieldHeight": field_height_mm.copy(),
                "sampleSpacing": np.array([2.5e-4, 2.5e-4], dtype=float),
                "wavelength": current_wave.copy(),
            },
        },
    }
    normalized = _normalize_raytrace_optics(raw_optics)
    normalized["raytrace"]["blocks_per_field_height"] = int(
        normalized["raytrace"].get("blocks_per_field_height", 4)
    )
    return normalized


def _wvf_default_state() -> dict[str, Any]:
    calc_pupil = float(DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM)
    focal_length = float(DEFAULT_WVF_FOCAL_LENGTH_M)
    wave_values = np.asarray(DEFAULT_WAVE, dtype=float)
    return {
        "type": "wvf",
        "wave": wave_values.copy(),
        "focal_length_m": focal_length,
        "f_number": (focal_length * 1e3) / max(calc_pupil, 1e-12),
        "aberration_scale": 0.0,
        "measured_pupil_diameter_mm": float(DEFAULT_WVF_MEASURED_PUPIL_MM),
        "measured_wavelength_nm": float(DEFAULT_WVF_MEASURED_WAVELENGTH_NM),
        "sample_interval_domain": "psf",
        "spatial_samples": DEFAULT_WVF_SPATIAL_SAMPLES,
        "ref_pupil_plane_size_mm": DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM,
        "calc_pupil_diameter_mm": calc_pupil,
        "zcoeffs": np.asarray([0.0], dtype=float),
        "lca_method": "none",
        "flip_psf_upside_down": False,
        "rotate_psf_90_degs": False,
        "compute_sce": False,
        "sce_params": _normalize_sce_params(wave_values, None),
        "calc_cone_psf_info": None,
    }


def wvf_create(*args: Any, **kwargs: Any) -> dict[str, Any]:
    wvf = _wvf_default_state()
    updates: list[tuple[Any, Any]] = []
    if args:
        if len(args) % 2 != 0:
            raise ValueError("wvfCreate expects key/value arguments.")
        for index in range(0, len(args), 2):
            key = args[index]
            if not isinstance(key, str):
                raise ValueError("wvfCreate expects string keys in key/value arguments.")
            updates.append((key, args[index + 1]))
    updates.extend(kwargs.items())

    for key, value in updates:
        normalized = param_format(key).replace("_", "").replace("-", "")
        if normalized in {"wave", "wavelength", "wavelengths", "calcwave", "calcwavelengths", "wls"}:
            wvf = wvf_set(wvf, "wave", value)
        elif normalized in {"focallengthm"}:
            wvf = wvf_set(wvf, "focal length", value, "m")
        elif normalized in {"focallength"}:
            # MATLAB wvfCreate('focal length', ...) uses millimeters.
            wvf = wvf_set(wvf, "focal length", value, "mm")
        elif normalized in {"fnumber", "f"}:
            wvf = wvf_set(wvf, "fnumber", value)
        elif normalized in {"aberrationscale"}:
            wvf["aberration_scale"] = float(np.asarray(value, dtype=float).reshape(-1)[0])
        elif normalized in {"measuredpupildiametermm", "measuredpupildiameter", "measuredpupilsize", "measuredpupil"}:
            wvf = wvf_set(wvf, "measured pupil size", value, "mm")
        elif normalized in {"measuredwavelengthnm", "measuredwavelength", "measuredwl", "measuredwave"}:
            wvf = wvf_set(wvf, "measured wavelength", value)
        elif normalized in {"calcpupildiametermm", "calcpupildiameter", "calcpupilsize", "calculatedpupil", "calculatedpupildiameter"}:
            wvf = wvf_set(wvf, "calc pupil size", value, "mm")
        elif normalized in {"zcoeffs", "zcoeff", "zcoef"}:
            wvf = wvf_set(wvf, "zcoeffs", value)
        elif normalized in {"lcamethod"}:
            wvf = wvf_set(wvf, "lca method", value)
        elif normalized in {"flippsfupsidedown"}:
            wvf = wvf_set(wvf, "flipPSFUpsideDown", value)
        elif normalized in {"rotatepsf90degs"}:
            wvf = wvf_set(wvf, "rotatePSF90degs", value)
        elif normalized in {"computesce"}:
            wvf = wvf_set(wvf, "compute_sce", value)
        elif normalized in {"sceparams"}:
            wvf = wvf_set(wvf, "sce params", value)
        elif normalized in {"sampleintervaldomain"}:
            wvf = wvf_set(wvf, "sample interval domain", value)
        elif normalized in {"spatialsamples", "numberspatialsamples", "npixels", "fieldsizepixels"}:
            wvf = wvf_set(wvf, "spatial samples", value)
        elif normalized in {"refpupilplanesize", "refpupilplanesizemm", "fieldsizemm"}:
            wvf = wvf_set(wvf, "ref pupil plane size", value, "mm")
        elif normalized in {"name", "type"}:
            wvf = wvf_set(wvf, str(key), value)
        else:
            wvf = wvf_set(wvf, str(key), value)
    return wvf


_WVF_ABERRATION_NAME_TO_OSA_INDEX = {
    "piston": 0,
    "verticaltilt": 1,
    "horizontaltilt": 2,
    "obliqueastigmatism": 3,
    "defocus": 4,
    "verticalastigmatism": 5,
    "verticaltrefoil": 6,
    "verticalcoma": 7,
    "horizontalcoma": 8,
    "obliquetrefoil": 9,
    "obliquequadrafoil": 10,
    "obliquesecondaryastigmatism": 11,
    "primaryspherical": 12,
    "spherical": 12,
    "verticalsecondaryastigmatism": 13,
    "verticalquadrafoil": 14,
}


def _normalize_wvf_aberration_name(name: Any) -> str:
    return param_format(name).replace("_", "").replace("-", "")


def _coerce_wvf_zcoeff_indices(indices: Any) -> np.ndarray:
    if isinstance(indices, str):
        normalized = _normalize_wvf_aberration_name(indices)
        if normalized not in _WVF_ABERRATION_NAME_TO_OSA_INDEX:
            raise ValueError(f"Unsupported wavefront aberration name: {indices}")
        return np.asarray([_WVF_ABERRATION_NAME_TO_OSA_INDEX[normalized]], dtype=int)

    if isinstance(indices, (list, tuple)):
        result: list[int] = []
        for item in indices:
            result.extend(_coerce_wvf_zcoeff_indices(item).tolist())
        return np.asarray(result, dtype=int)

    vector = np.asarray(indices, dtype=object).reshape(-1)
    if vector.size == 0:
        return np.empty(0, dtype=int)
    if all(isinstance(item, str) for item in vector.tolist()):
        return _coerce_wvf_zcoeff_indices(vector.tolist())
    return np.asarray(vector, dtype=int).reshape(-1)


def wvf_defocus_diopters_to_microns(diopters: Any, pupil_size_mm: Any) -> np.ndarray:
    diopters_array = np.asarray(diopters, dtype=float)
    pupil_size = float(pupil_size_mm)
    return diopters_array * (pupil_size**2) / (16.0 * np.sqrt(3.0))


def wvf_defocus_microns_to_diopters(microns: Any, pupil_size_mm: Any) -> np.ndarray:
    microns_array = np.asarray(microns, dtype=float)
    pupil_size = float(pupil_size_mm)
    return (16.0 * np.sqrt(3.0)) * microns_array / max(pupil_size**2, 1e-12)


def wvf_osa_index_to_zernike_nm(j: Any) -> tuple[Any, Any]:
    indices = np.asarray(j, dtype=float)
    n = np.ceil((-3.0 + np.sqrt(9.0 + 8.0 * indices)) / 2.0).astype(int)
    m = (2.0 * indices - n * (n + 2.0)).astype(int)
    if np.isscalar(j):
        return int(n.reshape(())), int(m.reshape(()))
    return n, m


def wvf_zernike_nm_to_osa_index(n: Any, m: Any) -> Any:
    radial = np.asarray(n, dtype=float)
    angular = np.asarray(m, dtype=float)
    indices = ((radial * (radial + 2.0)) + angular) / 2.0
    indices = indices.astype(int)
    if np.isscalar(n) and np.isscalar(m):
        return int(indices.reshape(()))
    return indices


def wvf_osa_index_to_vector_index(j_index: Any) -> tuple[Any, Any]:
    indices = _coerce_wvf_zcoeff_indices(j_index)
    vector_index = indices + 1
    if np.isscalar(j_index) or (isinstance(j_index, str) and indices.size == 1):
        return int(vector_index.reshape(())), int(indices.reshape(()))
    return vector_index, indices


def wvf_osa_index_to_name(idx: Any) -> str | list[str]:
    """Return the legacy aberration names for OSA j indices."""

    names_by_index = {
        0: "piston",
        1: "vertical_tilt",
        2: "horizontal_tilt",
        3: "oblique_astigmatism",
        4: "defocus",
        5: "vertical_astigmatism",
        6: "vertical_trefoil",
        7: "vertical_coma",
        8: "horizontal_coma",
        9: "oblique_trefoil",
        10: "oblique_quadrafoil",
        11: "oblique_secondary_astigmatism",
        12: "spherical",
        13: "vertical_secondary_astigmatism",
        14: "vertical_quadrafoil",
    }
    indices = np.asarray(idx, dtype=int).reshape(-1)
    names: list[str] = []
    for value in indices.tolist():
        if value not in names_by_index:
            raise ValueError(f"Unknown index {value}")
        names.append(names_by_index[value])
    if np.isscalar(idx):
        return names[0]
    return names


def wvf_wave_to_idx(wvf: dict[str, Any], w_list: Any) -> np.ndarray:
    wave = np.asarray(wvf_get(wvf, "calc wavelengths"), dtype=float).reshape(-1)
    rounded_wave = np.round(wave).astype(int)
    rounded_target = np.round(np.asarray(w_list, dtype=float).reshape(-1)).astype(int)
    idx = np.flatnonzero(np.isin(rounded_wave, rounded_target)) + 1
    if idx.size == 0:
        raise ValueError("wvfWave2idx: No matching wavelength in list")
    return idx.astype(int)


def wvf_root_path() -> str:
    """Return the root path of the vendored upstream wavefront toolbox snapshot."""

    return str(ensure_upstream_snapshot() / "opticalimage" / "wavefront")


def _wvf_summary_text(wvf: dict[str, Any]) -> str:
    wave = np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)
    summary_wave = float(wave[0]) if wave.size else float(DEFAULT_WVF_MEASURED_WAVELENGTH_NM)
    zcoeffs = np.asarray(wvf_get(wvf, "zcoeffs"), dtype=float).reshape(-1)
    otf_support = np.asarray(wvf_get(wvf, "otf support", "mm", summary_wave), dtype=float).reshape(-1)
    psf_support = np.asarray(wvf_get(wvf, "psf support", "um", summary_wave), dtype=float).reshape(-1)

    lines = [
        "",
        f"wavefront struct name: {wvf.get('name', 'wvf')}",
    ]
    if wave.size > 1:
        lines.append(f"Summarizing for wave {int(round(summary_wave))} nm.")
    lines.extend(
        [
            "-------------------",
            f"f number\t {float(wvf_get(wvf, 'fnumber')):.6f}",
            f"f length\t {float(wvf_get(wvf, 'focal length', 'mm')):.6f}\t mm",
            f"um per deg\t {float(wvf_get(wvf, 'um per degree')):.6f}\t um",
            f"calc pupil diam\t {float(wvf_get(wvf, 'calc pupil diameter', 'mm')):.6f}\t mm",
            "",
            "Reference",
            "------",
            f"n samples\t {int(wvf_get(wvf, 'spatial samples'))}",
            f"ref pupil plane\t {float(wvf_get(wvf, 'pupil plane size', 'mm', summary_wave)):.6f}\t mm",
            f"ref pupil dx\t {float(wvf_get(wvf, 'pupil sample spacing', 'um', summary_wave)):.6f}\t um",
            "",
            "Measured",
            "------",
            "zCoeffs:\t " + " ".join(f"{value:.2f}" for value in zcoeffs),
            f"zDiameter:\t {float(wvf_get(wvf, 'z pupil diameter', 'mm')):.6f}\t mm",
            f"Max OTF freq\t {float(np.max(otf_support)) if otf_support.size else 0.0:.6f}\t cyc/mm",
            f"OTF df\t\t {float(otf_support[1] - otf_support[0]) if otf_support.size >= 2 else 0.0:.6f}\t cyc/mm",
            f"Max PSF support\t {float(np.max(psf_support)) if psf_support.size else 0.0:.6f}\t um",
            f"PSF dx\t\t {float(psf_support[1] - psf_support[0]) if psf_support.size >= 2 else 0.0:.6f}\t um",
            "-------------------",
        ]
    )
    return "\n".join(lines)


def wvf_summarize(wvf: dict[str, Any], *, show: bool = False) -> str:
    """Return a headless MATLAB-style WVF summary string."""

    text = _wvf_summary_text(wvf)
    if show:
        print(text)
    return text


def wvf_print(wvf: dict[str, Any], *args: Any, show: bool = False) -> dict[str, Any]:
    """Headless MATLAB-style WVF print helper."""

    if args:
        _parse_key_value_options(args, "wvfPrint")
    text = _wvf_summary_text(wvf)
    if show:
        print(text)
    return dict(wvf)


wvfOSAIndexToZernikeNM = wvf_osa_index_to_zernike_nm
wvfZernikeNMToOSAIndex = wvf_zernike_nm_to_osa_index
wvfOSAIndexToVectorIndex = wvf_osa_index_to_vector_index
wvfOSAIndexToName = wvf_osa_index_to_name
wvfWave2idx = wvf_wave_to_idx
wvfRootPath = wvf_root_path
wvfSummarize = wvf_summarize
wvfPrint = wvf_print


_WVF_KEY_SYNONYMS = {
    "name": "name",
    "type": "type",
    "umperdegree": "umperdegree",
    "zcoeffs": "zcoeffs",
    "zcoeff": "zcoeffs",
    "zcoef": "zcoeffs",
    "wavefrontaberrations": "wavefrontaberrations",
    "pupilfunction": "pupilfunction",
    "pupilfunc": "pupilfunction",
    "pupfun": "pupilfunction",
    "measuredpupilsize": "measuredpupil",
    "measuredpupil": "measuredpupil",
    "measuredpupilmm": "measuredpupil",
    "measuredpupildiameter": "measuredpupil",
    "measuredwave": "measuredwl",
    "measuredwl": "measuredwl",
    "measuredwavelength": "measuredwl",
    "measuredopticalaxis": "measuredopticalaxis",
    "measuredopticalaxisdeg": "measuredopticalaxis",
    "measuredobserveraccommodation": "measuredobserveraccommodation",
    "measuredobserveraccommodationdiopters": "measuredobserveraccommodation",
    "measuredobserverfocuscorrection": "measuredobserverfocuscorrection",
    "measuredobserverfocuscorrectiondiopters": "measuredobserverfocuscorrection",
    "sampleintervaldomain": "sampleintervaldomain",
    "numberspatialsamples": "spatialsamples",
    "spatialsamples": "spatialsamples",
    "npixels": "spatialsamples",
    "fieldsizepixels": "spatialsamples",
    "refpupilplanesize": "refpupilplanesize",
    "refpupilplanesizemm": "refpupilplanesize",
    "fieldsizemm": "refpupilplanesize",
    "refpupilplanesampleinterval": "refpupilplanesampleinterval",
    "fieldsamplesize": "refpupilplanesampleinterval",
    "refpupilplanesampleintervalmm": "refpupilplanesampleinterval",
    "fieldsamplesizemmperpixel": "refpupilplanesampleinterval",
    "refpsfsampleinterval": "refpsfsampleinterval",
    "refpsfarcminpersample": "refpsfsampleinterval",
    "refpsfarcminperpixel": "refpsfsampleinterval",
    "calcpupilsize": "calcpupilsize",
    "calcpupildiameter": "calcpupilsize",
    "calculatedpupil": "calcpupilsize",
    "calculatedpupildiameter": "calcpupilsize",
    "calcopticalaxis": "calcopticalaxis",
    "calcobserveraccommodation": "calcobserveraccommodation",
    "calcobserverfocuscorrection": "calcobserverfocuscorrection",
    "defocusdiopters": "calcobserverfocuscorrection",
    "calcwave": "calcwavelengths",
    "calcwavelengths": "calcwavelengths",
    "wavelengths": "calcwavelengths",
    "wavelength": "calcwavelengths",
    "wls": "calcwavelengths",
    "wave": "calcwavelengths",
    "calcconepsfinfo": "calcconepsfinfo",
    "sceparams": "sceparams",
    "stilescrawford": "sceparams",
}


def _wvf_canonical_key(key: str) -> str:
    normalized = param_format(key).replace("_", "").replace("-", "")
    return _WVF_KEY_SYNONYMS.get(normalized, normalized)


def wvf_key_synonyms(key_values: Any) -> Any:
    """Convert a MATLAB-style WVF key or key/value list to canonical form."""

    if isinstance(key_values, str):
        return _wvf_canonical_key(key_values)

    if not isinstance(key_values, (list, tuple)):
        raise ValueError("wvfKeySynonyms expects a string or a list/tuple of key/value pairs.")

    canonical = list(key_values)
    for index in range(0, len(canonical), 2):
        key = canonical[index]
        if not isinstance(key, str):
            raise ValueError("wvfKeySynonyms expects string keys in odd positions.")
        canonical[index] = _wvf_canonical_key(key)

    if isinstance(key_values, tuple):
        return tuple(canonical)
    return canonical


def _human_wave_defocus(wave_nm: Any) -> np.ndarray:
    wave = np.asarray(wave_nm, dtype=float)
    q1 = 1.7312
    q2 = 0.63346
    q3 = 0.21410
    return q1 - (q2 / (wave * 1e-3 - q3))


def human_wave_defocus(wave_nm: Any) -> np.ndarray:
    """Legacy MATLAB humanWaveDefocus() compatibility wrapper."""

    return _human_wave_defocus(wave_nm)


def human_achromatic_otf(
    sample_sf: Any | None = None,
    model: str = "exp",
    pupil_d: float | None = None,
) -> np.ndarray:
    """Legacy MATLAB humanAchromaticOTF() compatibility wrapper."""

    spatial_frequency = np.arange(0.0, 51.0, 1.0, dtype=float) if sample_sf is None else np.asarray(sample_sf, dtype=float).reshape(-1)
    normalized_model = param_format(model)

    if normalized_model in {"exp", "exponential"}:
        a = 0.1212
        w1 = 0.3481
        w2 = 0.6519
        mtf = w1 * np.ones_like(spatial_frequency, dtype=float) + w2 * np.exp(-a * spatial_frequency)
        return np.asarray(mtf, dtype=float)

    if normalized_model in {"dl", "diffractionlimited"}:
        if pupil_d is None:
            raise ValueError("humanAchromaticOTF('dl') requires pupil diameter in mm.")
        wavelength_nm = 555.0
        u0 = float(pupil_d) * np.pi * 1e6 / wavelength_nm / 180.0
        u_hat = spatial_frequency / max(u0, 1.0e-12)
        inside = np.clip(1.0 - np.square(u_hat), 0.0, None)
        mtf = (2.0 / np.pi) * (np.arccos(np.clip(u_hat, -1.0, 1.0)) - u_hat * np.sqrt(inside))
        mtf[u_hat >= 1.0] = 0.0
        return np.asarray(mtf, dtype=float)

    if normalized_model == "watson":
        if pupil_d is None:
            raise ValueError("humanAchromaticOTF('watson') requires pupil diameter in mm.")
        if float(pupil_d) > 6.0 or float(pupil_d) < 2.0:
            pass
        u1 = 21.95 - 5.512 * float(pupil_d) + 0.3922 * float(pupil_d) ** 2
        mtf_dl = human_achromatic_otf(spatial_frequency, "dl", pupil_d)
        mtf = np.power(1.0 + np.square(spatial_frequency / u1), -0.62) * np.sqrt(np.clip(mtf_dl, 0.0, None))
        return np.asarray(mtf, dtype=float)

    raise UnsupportedOptionError("humanAchromaticOTF", model)


def human_core(
    wave: Any,
    sample_sf: Any,
    p: float,
    D0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy MATLAB humanCore() compatibility wrapper."""

    wave_nm = np.asarray(wave, dtype=float).reshape(-1)
    sample_sf_cpd = np.asarray(sample_sf, dtype=float).reshape(-1)
    if wave_nm.size == 0 or sample_sf_cpd.size == 0:
        raise ValueError("humanCore requires non-empty wavelength and sample spatial-frequency inputs.")

    defocus = _human_wave_defocus(wave_nm)
    w20 = (float(p) ** 2 / 2.0) * (float(D0) * defocus) / np.maximum(float(D0) + defocus, 1.0e-12)
    deg_per_meter = 1.0 / (np.tan(np.deg2rad(1.0)) * (1.0 / float(D0)))
    ach_otf = human_achromatic_otf(sample_sf_cpd)

    otf = np.zeros((wave_nm.size, sample_sf_cpd.size), dtype=float)
    wavelengths_m = wave_nm * 1e-9
    for index, wavelength_m in enumerate(wavelengths_m):
        reduced_sf = ((deg_per_meter * wavelength_m) / max(float(D0) * float(p), 1.0e-12)) * sample_sf_cpd
        alpha = ((4.0 * np.pi) / max(wavelength_m, 1.0e-12)) * w20[index] * reduced_sf
        otf[index, :] = _optics_defocused_mtf(reduced_sf, np.abs(alpha)) * ach_otf

    return np.asarray(otf, dtype=float), np.asarray(ach_otf, dtype=float)


def _human_otf_support(
    f_support: Any | None = None,
) -> tuple[np.ndarray, float]:
    if f_support is None:
        max_frequency = 60.0
        f_list = unit_frequency_list(int(max_frequency)) * max_frequency
        x_grid, y_grid = np.meshgrid(f_list, f_list, indexing="xy")
        support = np.dstack((x_grid, y_grid))
        return np.asarray(support, dtype=float), float(max_frequency)

    support = np.asarray(f_support, dtype=float)
    if support.ndim != 3 or support.shape[2] != 2:
        raise ValueError("humanOTF expects f_support with shape (rows, cols, 2).")
    max_f1 = float(np.max(support[:, :, 0]))
    max_f2 = float(np.max(support[:, :, 1]))
    return support, min(max_f1, max_f2)


def _human_otf_stack(
    p_radius: float | None = None,
    D0: float | None = None,
    f_support: Any | None = None,
    wave: Any | None = None,
    *,
    storage: str = "legacy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pupil_radius_m = 0.0015 if p_radius is None else float(p_radius)
    dioptric_power = 1.0 / 0.017 if D0 is None else float(D0)
    wave_nm = np.arange(400.0, 701.0, 1.0, dtype=float) if wave is None else np.asarray(wave, dtype=float).reshape(-1)
    support, max_frequency = _human_otf_support(f_support)

    dist = np.sqrt(np.square(support[:, :, 0]) + np.square(support[:, :, 1]))
    sample_sf = (np.arange(40, dtype=float) / 39.0) * max_frequency
    otf_rows, _ = human_core(wave_nm, sample_sf, pupil_radius_m, dioptric_power)

    otf_2d = np.zeros(support.shape[:2] + (wave_nm.size,), dtype=complex)
    outside = dist > max_frequency
    for index in range(wave_nm.size):
        interpolator = interp1d(sample_sf, otf_rows[index, :], kind="cubic", bounds_error=False, fill_value=0.0, assume_sorted=True)
        plane = np.abs(np.asarray(interpolator(dist), dtype=float))
        plane[outside] = 0.0
        if param_format(storage) == "ibio":
            otf_2d[:, :, index] = np.fft.ifftshift(plane)
        else:
            otf_2d[:, :, index] = np.fft.fftshift(plane)

    return np.asarray(otf_2d, dtype=complex), np.asarray(support, dtype=float), np.asarray(wave_nm, dtype=float)


def human_otf(
    p_radius: float | None = None,
    D0: float | None = None,
    f_support: Any | None = None,
    wave: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy MATLAB humanOTF() compatibility wrapper."""

    return _human_otf_stack(p_radius, D0, f_support, wave, storage="legacy")


def human_otf_ibio(
    p_radius: float | None = None,
    D0: float | None = None,
    f_support: Any | None = None,
    wave: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy MATLAB humanOTF_ibio() compatibility wrapper."""

    return _human_otf_stack(p_radius, D0, f_support, wave, storage="ibio")


def human_lsf(
    pupil_radius: float | None = None,
    dioptric_power: float | None = None,
    unit: str = "mm",
    wave: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy MATLAB humanLSF() compatibility wrapper."""

    combined_otf, sample_sf, wavelengths = human_otf(pupil_radius, dioptric_power, None, wave)
    n_wave = wavelengths.size
    n_samples = combined_otf.shape[0]
    line_spread = np.zeros((n_wave, n_samples), dtype=float)

    for index in range(n_wave):
        center_line = np.asarray(combined_otf[:, :, index], dtype=complex)[:, 0]
        line_spread[index, :] = np.fft.fftshift(np.abs(np.fft.ifft(center_line)))

    delta_space = 1.0 / (2.0 * max(float(np.max(sample_sf)), 1.0e-12))
    spatial_extent_deg = delta_space * n_samples
    x_dim = unit_frequency_list(n_samples) * spatial_extent_deg

    mm_per_deg = 0.330
    normalized_unit = param_format(unit)
    if normalized_unit in {"mm", "default"}:
        x_dim = x_dim * mm_per_deg
    elif normalized_unit == "um":
        x_dim = x_dim * mm_per_deg * 1e3
    else:
        raise UnsupportedOptionError("humanLSF", unit)

    return np.asarray(line_spread, dtype=float), np.asarray(x_dim, dtype=float), np.asarray(wavelengths, dtype=float)


def _macular_profile(
    density: float,
    wave_nm: np.ndarray,
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, np.ndarray]:
    _, profile = _store(asset_store).load_spectra("macularPigment.mat", wave_nm=np.asarray(wave_nm, dtype=float).reshape(-1))
    base_density = np.asarray(profile, dtype=float).reshape(-1)
    unit_density = base_density / 0.3521
    actual_density = unit_density * float(density)
    transmittance = np.power(10.0, -actual_density)
    absorption = 1.0 - transmittance
    return {
        "wave": np.asarray(wave_nm, dtype=float).reshape(-1).copy(),
        "unitDensity": np.asarray(unit_density, dtype=float),
        "density": np.asarray(actual_density, dtype=float),
        "transmittance": np.asarray(transmittance, dtype=float),
        "absorption": np.asarray(absorption, dtype=float),
    }


def human_macular_transmittance(
    oi: OpticalImage | None = None,
    dens: float = 0.35,
    *,
    asset_store: AssetStore | None = None,
) -> OpticalImage:
    """Legacy MATLAB humanMacularTransmittance() compatibility wrapper."""

    current = oi_create(asset_store=_store(asset_store)) if oi is None else oi.clone()
    wave_nm = np.asarray(oi_get(current, "wave"), dtype=float).reshape(-1)
    profile = _macular_profile(float(dens), wave_nm, asset_store=asset_store)
    current = oi_set(current, "transmittance wave", profile["wave"])
    current = oi_set(current, "transmittance", profile["transmittance"])
    return current


def human_oi(
    scene: Scene,
    oi: OpticalImage | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> OpticalImage:
    """Legacy MATLAB humanOI() compatibility wrapper."""

    if scene is None:
        raise ValueError("Scene required.")

    current = oi_create("shift invariant", asset_store=_store(asset_store)) if oi is None else oi.clone()
    wave_nm = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    otf, support, otf_wave = human_otf_ibio(wave=wave_nm)
    otf_struct = {
        "OTF": np.asarray(otf, dtype=complex),
        "fx": np.asarray(support[0, :, 0], dtype=float).reshape(-1),
        "fy": np.asarray(support[:, 0, 1], dtype=float).reshape(-1),
        "wave": np.asarray(otf_wave, dtype=float).reshape(-1),
        "function": "humanOTF_ibio",
    }

    current = oi_set(current, "wave", wave_nm)
    current = oi_set(current, "fov", float(scene_get(scene, "wangular")))
    current = oi_set(current, "otfstruct", otf_struct)
    current = oi_set(current, "compute method", "humanmw")
    current.fields["optics"]["name"] = "human"
    current.fields["optics"]["otf_method"] = "human"
    return oi_compute(current, scene)


humanWaveDefocus = human_wave_defocus
humanAchromaticOTF = human_achromatic_otf
humanCore = human_core
humanOTF = human_otf
humanOTF_ibio = human_otf_ibio
humanLSF = human_lsf
humanMacularTransmittance = human_macular_transmittance
humanOI = human_oi


def ijspeert(
    age: float,
    p: float,
    m: float,
    q: Any,
    phi: Any | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Legacy MATLAB ijspeert() compatibility wrapper."""

    sample_sf = np.asarray(q, dtype=float).reshape(-1)
    D = 70.0
    age_factor = 1.0 + (float(age) / D) ** 4
    c_sa = 1.0 / (1.0 + age_factor / (1.0 / float(m) - 1.0))
    c_la = 1.0 / (1.0 + (1.0 / float(m) - 1.0) / age_factor)

    b = 9000.0 - 936.0 * np.sqrt(age_factor)
    d = 3.2
    e = np.sqrt(age_factor) / 2000.0

    c = np.zeros(4, dtype=float)
    beta = np.zeros(4, dtype=float)
    c[0] = c_sa / (1.0 + (float(p) / d) ** 2)
    c[1] = c_sa / (1.0 + (d / float(p)) ** 2)
    beta[0] = (1.0 + (float(p) / d) ** 2) / (b * float(p))
    beta[1] = (1.0 + (d / float(p)) ** 2) * (e - 1.0 / (b * float(p)))

    c[2] = c_la / ((1.0 + 25.0 * float(m)) * (1.0 + 1.0 / age_factor))
    c[3] = c_la - c[2]
    beta[2] = 1.0 / (10.0 + 60.0 * float(m) - 5.0 / age_factor)
    beta[3] = 1.0

    m_beta = np.exp(-360.0 * beta[:, None] * sample_sf[None, :])
    mtf = np.sum(c[:, None] * m_beta, axis=0)

    psf: np.ndarray | None = None
    lsf: np.ndarray | None = None
    if phi is not None:
        angles = np.asarray(phi, dtype=float).reshape(-1)
        sinphi2 = np.square(np.sin(angles))
        cosphi2 = np.square(np.cos(angles))
        beta2 = np.square(beta)

        f_beta = np.zeros((4, angles.size), dtype=float)
        for index in range(4):
            f_beta[index, :] = beta[index] / (2.0 * np.pi * np.power(sinphi2 + beta2[index] * cosphi2, 1.5))
        psf = np.sum(c[:, None] * f_beta, axis=0)

        l_beta = np.zeros((4, angles.size), dtype=float)
        for index in range(4):
            l_beta[index, :] = beta[index] / (np.pi * (sinphi2 + beta2[index] * cosphi2))
        lsf = np.sum(c[:, None] * l_beta, axis=0)

    return np.asarray(mtf, dtype=float), None if psf is None else np.asarray(psf, dtype=float), None if lsf is None else np.asarray(lsf, dtype=float)


def _wvf_lca_from_wavelength_difference(
    wl1_nm: Any,
    wl2_nm: Any,
    which_calc: str = "hoferCode",
) -> np.ndarray:
    first = np.asarray(wl1_nm, dtype=float)
    second = np.asarray(wl2_nm, dtype=float)
    mode = param_format(which_calc)
    if mode in {"hofercode", ""}:
        constant = 1.8859 - (0.63346 / (0.001 * first - 0.2141))
        return 1.8859 - constant - (0.63346 / (0.001 * second - 0.2141))
    if mode == "thibospaper":
        r_m = 5.55e-3
        n_d = 1.333
        a = 1.320535
        b = 0.004685
        c = 0.214102
        wl1_um = first * 1e-3
        wl2_um = second * 1e-3
        n1 = a + b / (wl1_um - c)
        n2 = a + b / (wl2_um - c)
        return (n1 - n2) / (n_d * r_m)
    if mode == "iset":
        if np.any(first != 580):
            raise ValueError("'iset' LCA mode assumes the first wavelength is 580 nm.")
        return _human_wave_defocus(second)
    raise UnsupportedOptionError("wvfLCAFromWavelengthDifference", which_calc)


def _wvf_lca_microns(
    wvf: dict[str, Any],
    wavelength_nm: float,
) -> float:
    raw_method = wvf.get("lca_method", "none")
    method = param_format(raw_method)
    if method in {"none", ""}:
        return 0.0
    measured_wavelength_nm = float(wvf.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    measured_pupil_mm = float(wvf.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM))
    if method == "human":
        lca_diopters = _wvf_lca_from_wavelength_difference(measured_wavelength_nm, wavelength_nm)
    elif callable(raw_method):
        lca_diopters = raw_method(measured_wavelength_nm, wavelength_nm)
    else:
        raise UnsupportedOptionError("wvfCompute", f"wvf lca method {raw_method}")
    return float(
        np.asarray(
            wvf_defocus_diopters_to_microns(-np.asarray(lca_diopters, dtype=float), measured_pupil_mm),
            dtype=float,
        ).reshape(-1)[0]
    )


def _wvf_zcoeffs_with_lca(zcoeffs: np.ndarray, lca_microns: float) -> np.ndarray:
    coefficients = np.asarray(zcoeffs, dtype=float).reshape(-1).copy()
    if coefficients.size < 5:
        coefficients = np.pad(coefficients, (0, 5 - coefficients.size), constant_values=0.0)
    coefficients[4] += float(lca_microns)
    return coefficients


def _default_cone_psf_info(*, asset_store: AssetStore | None = None) -> dict[str, Any]:
    wavelengths, sensitivities = _store(asset_store).load_spectra("stockman.mat")
    sensitivities_array = np.asarray(sensitivities, dtype=float)
    if sensitivities_array.ndim == 1:
        sensitivities_array = sensitivities_array.reshape(-1, 1)
    n_cones = int(sensitivities_array.shape[1])
    return {
        "wavelengths": np.asarray(wavelengths, dtype=float).reshape(-1).copy(),
        "spectral_sensitivities": sensitivities_array.copy(),
        "spectral_weighting": np.ones(int(np.asarray(wavelengths).size), dtype=float),
        "cone_weighting": np.full(n_cones, 1.0 / max(n_cones, 1), dtype=float),
    }


def _normalize_cone_psf_info(
    value: Any | None,
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    if value is None:
        return _default_cone_psf_info(asset_store=asset_store)

    current = dict(_mat_to_native(value)) if not isinstance(value, dict) else dict(value)
    wavelengths = np.asarray(
        current.get("wavelengths", current.get("wave", current.get("S", []))),
        dtype=float,
    ).reshape(-1)
    sensitivities = np.asarray(
        current.get(
            "spectral_sensitivities",
            current.get("spectralSensitivities", current.get("T", current.get("data", []))),
        ),
        dtype=float,
    )
    if wavelengths.size == 0 or sensitivities.size == 0:
        return _default_cone_psf_info(asset_store=asset_store)
    if sensitivities.ndim == 1:
        sensitivities = sensitivities.reshape(-1, 1)
    elif sensitivities.shape[0] != wavelengths.size and sensitivities.shape[1] == wavelengths.size:
        sensitivities = sensitivities.T
    if sensitivities.shape[0] != wavelengths.size:
        raise ValueError("Cone PSF info sensitivities must match the wavelength dimension.")

    spectral_weighting = np.asarray(
        current.get(
            "spectral_weighting",
            current.get("spectralWeighting", current.get("spdWeighting", np.ones(wavelengths.size, dtype=float))),
        ),
        dtype=float,
    ).reshape(-1)
    if spectral_weighting.size != wavelengths.size:
        raise ValueError("Cone PSF info spectral weighting must match the wavelength dimension.")

    cone_weighting = np.asarray(
        current.get(
            "cone_weighting",
            current.get("coneWeighting", np.full(sensitivities.shape[1], 1.0 / max(sensitivities.shape[1], 1), dtype=float)),
        ),
        dtype=float,
    ).reshape(-1)
    if cone_weighting.size != sensitivities.shape[1]:
        raise ValueError("Cone PSF info cone weighting must match the number of cone classes.")
    cone_weighting_sum = float(np.sum(cone_weighting))
    if cone_weighting_sum > 0.0:
        cone_weighting = cone_weighting / cone_weighting_sum

    return {
        "wavelengths": wavelengths.copy(),
        "spectral_sensitivities": sensitivities.copy(),
        "spectral_weighting": spectral_weighting.copy(),
        "cone_weighting": cone_weighting.copy(),
    }


def _export_cone_psf_info(value: Any | None, *, asset_store: AssetStore | None = None) -> dict[str, Any]:
    current = _normalize_cone_psf_info(value, asset_store=asset_store)
    return {
        "wave": current["wavelengths"].copy(),
        "wavelengths": current["wavelengths"].copy(),
        "spectral_sensitivities": current["spectral_sensitivities"].copy(),
        "spectralSensitivities": current["spectral_sensitivities"].copy(),
        "spectral_weighting": current["spectral_weighting"].copy(),
        "spectralWeighting": current["spectral_weighting"].copy(),
        "cone_weighting": current["cone_weighting"].copy(),
        "coneWeighting": current["cone_weighting"].copy(),
    }


def _wvf_cone_psf_weights(
    cone_psf_info: dict[str, Any],
    wave_nm: np.ndarray,
) -> np.ndarray:
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    info = _normalize_cone_psf_info(cone_psf_info)
    sensitivities = np.asarray(interp_spectra(info["wavelengths"], info["spectral_sensitivities"], wave), dtype=float)
    weighting = np.asarray(interp_spectra(info["wavelengths"], info["spectral_weighting"], wave), dtype=float).reshape(-1)
    weighting_sum = float(np.sum(weighting))
    if weighting_sum > 0.0:
        weighting = weighting / weighting_sum
    weights = sensitivities.T * weighting[None, :]
    normalizer = np.sum(weights, axis=1, keepdims=True)
    normalizer[normalizer == 0.0] = 1.0
    return np.asarray(weights / normalizer, dtype=float)


def wvf_load_thibos_virtual_eyes(
    pupil_diameter_mm: float = 6.0,
    *,
    asset_store: AssetStore | None = None,
    full: bool = False,
) -> Any:
    sample_mean, sample_cov, subject_coeffs = _store(asset_store).load_thibos_virtual_eyes(pupil_diameter_mm)
    if full:
        return sample_mean, sample_cov, subject_coeffs
    return sample_mean.copy()


def _wvf_middle_row(wvf: dict[str, Any]) -> float:
    return np.floor(int(wvf.get("spatial_samples", DEFAULT_WVF_SPATIAL_SAMPLES)) / 2.0) + 1.0


def _wvf_wave_values(wvf: dict[str, Any]) -> np.ndarray:
    return np.asarray(wvf.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)


def _wvf_parse_wave_and_unit_args(
    wvf: dict[str, Any],
    args: tuple[Any, ...],
    default_unit: str,
) -> tuple[float, str]:
    wavelength_nm = _default_plot_wavelength(_wvf_wave_values(wvf))
    unit = str(default_unit)
    if args:
        first = args[0]
        if isinstance(first, str):
            unit = str(first)
        else:
            wavelength_nm = float(np.asarray(first, dtype=float).reshape(-1)[0])
    if len(args) > 1:
        second = args[1]
        if isinstance(second, str):
            unit = str(second)
        else:
            wavelength_nm = float(np.asarray(second, dtype=float).reshape(-1)[0])
    return wavelength_nm, unit


def _parse_key_value_options(args: tuple[Any, ...], context: str) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError(f"{context} expects key/value arguments.")
    options: dict[str, Any] = {}
    for index in range(0, len(args), 2):
        key = args[index]
        if not isinstance(key, str):
            raise ValueError(f"{context} expects string keys in key/value arguments.")
        options[param_format(key)] = args[index + 1]
    return options


def _wvf_base_amplitude_mask(n_pixels: int, aperture: np.ndarray | None = None) -> np.ndarray:
    if aperture is not None:
        current = np.asarray(aperture, dtype=float)
        if current.shape != (n_pixels, n_pixels):
            current = _resize_image(current, (n_pixels, n_pixels), method="linear")
        return np.asarray(current, dtype=float)

    sample_positions = (np.arange(n_pixels, dtype=float) + 1.0) - (np.floor(n_pixels / 2.0) + 1.0)
    xpos, ypos = np.meshgrid(sample_positions, sample_positions)
    radius = np.sqrt(xpos**2 + ypos**2) / max((n_pixels - 1) / 2.0, 1e-12)
    return (radius <= 1.0).astype(float)


def _wvf_compute_pupil_function_explicit(
    wvf: dict[str, Any],
    *,
    aperture: np.ndarray | None = None,
) -> dict[str, Any]:
    updated = dict(wvf)
    wave = np.asarray(updated.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    spatial_samples = int(updated.get("spatial_samples", DEFAULT_WVF_SPATIAL_SAMPLES))
    n_pixels = max(spatial_samples, 3)
    zcoeffs = np.asarray(updated.get("zcoeffs", np.array([0.0], dtype=float)), dtype=float).reshape(-1)
    pupil_diameter_mm = float(updated.get("calc_pupil_diameter_mm", DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM))
    base_amplitude = _wvf_base_amplitude_mask(n_pixels, aperture=aperture)
    pupil_function = np.zeros((n_pixels, n_pixels, wave.size), dtype=np.complex128)
    wavefront_stack = np.zeros((n_pixels, n_pixels, wave.size), dtype=float)
    areapix = np.zeros(wave.size, dtype=float)
    areapixapod = np.zeros(wave.size, dtype=float)

    for band_index, wavelength_nm in enumerate(wave):
        wave_um = float(wavelength_nm) * 1e-3
        pupil_plane_size_mm = _wvf_pupil_plane_size_m(updated, float(wavelength_nm)) * 1e3
        pupil_pos = ((np.arange(n_pixels, dtype=float) + 1.0) - (np.floor(n_pixels / 2.0) + 1.0))
        pupil_pos = pupil_pos * (pupil_plane_size_mm / max(n_pixels, 1))
        xpos, ypos = np.meshgrid(pupil_pos, pupil_pos)
        ypos = -ypos
        norm_radius = np.sqrt(xpos**2 + ypos**2) / max(pupil_diameter_mm / 2.0, 1e-12)
        theta = np.arctan2(ypos, xpos)
        norm_radius_index = norm_radius <= 1.0

        bounding_box = _image_bounding_box(norm_radius_index)
        target_shape = (
            max(int(round(bounding_box[3])), 1),
            max(int(round(bounding_box[2])), 1),
        )
        amplitude_band = _resize_image(base_amplitude, target_shape, method="linear")
        pad = int(round((n_pixels - bounding_box[2]) / 2.0))
        if pad > 0:
            amplitude_band = np.pad(amplitude_band, ((pad, pad), (pad, pad)), mode="constant")
        amplitude_band = _resize_image(amplitude_band, (n_pixels, n_pixels), method="linear")
        amplitude_band = np.clip(amplitude_band, 0.0, 1.0)

        lca_microns = _wvf_lca_microns(updated, float(wavelength_nm))
        wavefront_aberrations_um = _zernike_surface_osa(_wvf_zcoeffs_with_lca(zcoeffs, lca_microns), norm_radius, theta)
        pupil_phase = np.exp(-1j * 2.0 * np.pi * wavefront_aberrations_um / max(wave_um, 1e-12))
        pupil_phase[norm_radius > 0.5] = 1.0
        pupil = amplitude_band * pupil_phase

        wavefront_stack[:, :, band_index] = wavefront_aberrations_um
        pupil_function[:, :, band_index] = pupil
        areapix[band_index] = float(np.sum(np.abs(pupil_phase)))
        areapixapod[band_index] = float(np.sum(np.abs(pupil)))

    updated["computed"] = True
    updated["aperture_function"] = np.asarray(aperture, dtype=float).copy() if aperture is not None else updated.get("aperture_function")
    updated["pupil_function"] = pupil_function
    updated["pupil_amplitude"] = np.abs(pupil_function)
    updated["pupil_phase"] = np.angle(pupil_function)
    updated["wavefront_aberrations_um"] = wavefront_stack
    updated["pupil_support"] = ((np.arange(n_pixels, dtype=float) + 1.0) - (np.floor(n_pixels / 2.0) + 1.0))
    updated["areapix"] = areapix
    updated["areapixapod"] = areapixapod
    return updated


def _wvf_compute_psf_from_pupil_function(
    wvf: dict[str, Any],
    pupil_function: np.ndarray,
) -> dict[str, Any]:
    updated = dict(wvf)
    pupil_stack = np.asarray(pupil_function, dtype=np.complex128)
    if pupil_stack.ndim == 2:
        pupil_stack = pupil_stack[:, :, None]
    psf_stack = np.zeros_like(np.abs(pupil_stack), dtype=float)

    for band_index in range(pupil_stack.shape[2]):
        amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_stack[:, :, band_index])))
        intensity = np.real(amp * np.conjugate(amp))
        intensity_sum = float(np.sum(intensity))
        if intensity_sum > 0.0:
            intensity = intensity / intensity_sum
        if bool(updated.get("flip_psf_upside_down", False)):
            intensity = np.flipud(intensity)
        if bool(updated.get("rotate_psf_90_degs", False)):
            intensity = np.rot90(intensity)
        psf_stack[:, :, band_index] = intensity

    updated["computed"] = True
    updated["psf"] = psf_stack
    return updated


def _wvf_wave_for_query(wvf: dict[str, Any], args: tuple[Any, ...], default_unit: str) -> tuple[str, float]:
    unit = default_unit
    wavelength_nm = _default_plot_wavelength(_wvf_wave_values(wvf))
    if args:
        unit = str(args[0])
    if len(args) > 1:
        wavelength_nm = float(np.asarray(args[1], dtype=float).reshape(-1)[0])
    return unit, wavelength_nm


def _wvf_ref_pupil_plane_size_m(wvf: dict[str, Any]) -> float:
    return float(wvf.get("ref_pupil_plane_size_mm", DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM)) * 1e-3


def _wvf_pupil_plane_size_m(wvf: dict[str, Any], wavelength_nm: float) -> float:
    ref_size_m = _wvf_ref_pupil_plane_size_m(wvf)
    measured_wavelength_nm = float(wvf.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    which_domain = param_format(wvf.get("sample_interval_domain", "psf"))
    if which_domain == "psf":
        return ref_size_m * (float(wavelength_nm) / max(measured_wavelength_nm, 1e-12))
    if which_domain == "pupil":
        return ref_size_m
    raise UnsupportedOptionError("wvfGet", f"sample interval domain {wvf.get('sample_interval_domain')}")


def _wave_unit_scale(unit: Any) -> float:
    normalized = param_format(unit if unit is not None else "nm")
    if normalized in {"nm", "nanometer", "nanometers"}:
        return 1.0
    if normalized in {"um", "micron", "microns"}:
        return 1e-3
    if normalized in {"mm", "millimeter", "millimeters"}:
        return 1e-6
    if normalized in {"m", "meter", "meters"}:
        return 1e-9
    raise ValueError(f"Unsupported wavelength unit: {unit}")


def _wvf_um_per_degree(wvf: dict[str, Any]) -> float:
    focal_length_m = float(wvf.get("focal_length_m", DEFAULT_WVF_FOCAL_LENGTH_M))
    return focal_length_m * (2.0 * np.tan(np.deg2rad(0.5))) * 1e6


def _wvf_psf_angle_per_sample_deg(wvf: dict[str, Any], wavelength_nm: float) -> float:
    ref_size_m = _wvf_ref_pupil_plane_size_m(wvf)
    measured_wavelength_nm = float(wvf.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    which_domain = param_format(wvf.get("sample_interval_domain", "psf"))
    if which_domain == "psf":
        radians_per_sample = (measured_wavelength_nm * 1e-9) / max(ref_size_m, 1e-12)
    elif which_domain == "pupil":
        radians_per_sample = (float(wavelength_nm) * 1e-9) / max(ref_size_m, 1e-12)
    else:
        raise UnsupportedOptionError("wvfGet", f"sample interval domain {wvf.get('sample_interval_domain')}")
    return float(np.degrees(radians_per_sample))


def _wvf_angle_samples_for_unit(samples_deg: np.ndarray, unit: Any) -> np.ndarray:
    normalized = param_format(unit or "deg")
    if normalized in {"deg", "degree", "degrees"}:
        return samples_deg
    if normalized in {"min", "arcmin", "minute", "minutes"}:
        return samples_deg * 60.0
    if normalized in {"sec", "arcsec", "second", "seconds"}:
        return samples_deg * 3600.0
    raise UnsupportedOptionError("wvfGet", f"angle unit {unit}")


def _wvf_angle_scalar_to_radians(value: Any, unit: Any | None = None) -> float:
    normalized = param_format(unit or "min")
    scalar = float(np.asarray(value, dtype=float).reshape(-1)[0])
    if normalized in {"deg", "degree", "degrees"}:
        return float(np.deg2rad(scalar))
    if normalized in {"min", "arcmin", "minute", "minutes"}:
        return float(np.deg2rad(scalar / 60.0))
    if normalized in {"sec", "arcsec", "second", "seconds"}:
        return float(np.deg2rad(scalar / 3600.0))
    if normalized in {"rad", "radian", "radians"}:
        return scalar
    raise UnsupportedOptionError("wvfSet", f"angle unit {unit}")


def _wvf_psf_angular_samples(wvf: dict[str, Any], unit: Any, wavelength_nm: float) -> np.ndarray:
    n_pixels = int(wvf.get("spatial_samples", DEFAULT_WVF_SPATIAL_SAMPLES))
    angle_per_sample_deg = _wvf_psf_angle_per_sample_deg(wvf, wavelength_nm)
    samples_deg = angle_per_sample_deg * ((np.arange(n_pixels, dtype=float) + 1.0) - _wvf_middle_row(wvf))
    return _wvf_angle_samples_for_unit(samples_deg, unit)


def _wvf_psf_spatial_samples(wvf: dict[str, Any], unit: Any, wavelength_nm: float) -> np.ndarray:
    normalized = param_format(unit or "deg")
    angle_support_deg = _wvf_psf_angular_samples(wvf, "deg", wavelength_nm)
    if normalized in {"deg", "degree", "degrees", "min", "arcmin", "minute", "minutes", "sec", "arcsec", "second", "seconds"}:
        return _wvf_angle_samples_for_unit(angle_support_deg, normalized)
    support_m = angle_support_deg * (_wvf_um_per_degree(wvf) * 1e-6)
    return support_m * _spatial_unit_scale(normalized)


def _wvf_pupil_spatial_samples(wvf: dict[str, Any], unit: Any, wavelength_nm: float) -> np.ndarray:
    n_pixels = int(wvf.get("spatial_samples", DEFAULT_WVF_SPATIAL_SAMPLES))
    spacing_m = _wvf_pupil_plane_size_m(wvf, wavelength_nm) / max(n_pixels, 1)
    samples_m = spacing_m * ((np.arange(n_pixels, dtype=float) + 1.0) - _wvf_middle_row(wvf))
    return samples_m * _spatial_unit_scale(unit or "m")


def _wvf_otf_support(wvf: dict[str, Any], unit: Any, wavelength_nm: float) -> np.ndarray:
    support = _wvf_psf_spatial_samples(wvf, unit, wavelength_nm)
    if support.size < 2:
        return np.zeros_like(support, dtype=float)
    dx = float(support[1] - support[0])
    nyquist = 1.0 / max(2.0 * dx, 1e-12)
    return unit_frequency_list(support.size) * nyquist


def _logical_scalar(value: Any) -> bool:
    array = np.asarray(value)
    if array.size == 0:
        return False
    current = array.reshape(-1)[0]
    if isinstance(current, bytes):
        current = current.decode("utf-8")
    if isinstance(current, str):
        normalized = param_format(current)
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return bool(current)


def wvf_compute(
    wvf: dict[str, Any],
    *args: Any,
    compute_pupil_function: bool | None = None,
    compute_psf: bool | None = None,
    aperture: np.ndarray | None = None,
    compute_sce: bool | None = None,
) -> dict[str, Any]:
    options = _parse_key_value_options(args, "wvfCompute") if args else {}
    if "computepupilfunction" in options:
        compute_pupil_function = _logical_scalar(options.pop("computepupilfunction"))
    if "computepupilfunc" in options:
        compute_pupil_function = _logical_scalar(options.pop("computepupilfunc"))
    if "computepsf" in options:
        compute_psf = _logical_scalar(options.pop("computepsf"))
    if "aperture" in options:
        aperture = np.asarray(options.pop("aperture"), dtype=float)
    if "computesce" in options:
        compute_sce = _logical_scalar(options.pop("computesce"))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfCompute parameter: {unsupported}")

    if compute_pupil_function is None:
        compute_pupil_function = True
    if compute_psf is None:
        compute_psf = True

    updated = dict(wvf)
    if not compute_pupil_function and not compute_psf:
        updated["computed"] = False
        return updated
    if aperture is None and updated.get("aperture_function") is not None:
        aperture = np.asarray(updated.get("aperture_function"), dtype=float)

    wave = np.asarray(updated.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    spatial_samples = int(updated.get("spatial_samples", DEFAULT_WVF_SPATIAL_SAMPLES))
    n_pixels = max(spatial_samples, 3)
    ref_pupil_size_mm = float(updated.get("ref_pupil_plane_size_mm", DEFAULT_WVF_REF_PUPIL_PLANE_SIZE_MM))
    calc_pupil_mm = float(updated.get("calc_pupil_diameter_mm", DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM))
    measured_pupil_mm = float(updated.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM))
    measured_wavelength_nm = float(updated.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    focal_length_mm = float(updated.get("focal_length_m", DEFAULT_WVF_FOCAL_LENGTH_M)) * 1e3
    calc_radius = max(calc_pupil_mm / max(measured_pupil_mm, 1e-12), 1e-12)
    local_compute_sce = bool(updated.get("compute_sce", False) if compute_sce is None else compute_sce)
    sce_params = _normalize_sce_params(wave, updated.get("sce_params"))
    middle_row = np.floor(n_pixels / 2.0) + 1.0
    sample_positions = (np.arange(n_pixels, dtype=float) + 1.0) - middle_row
    pupil_function = np.zeros((n_pixels, n_pixels, wave.size), dtype=np.complex128)
    psf_stack = np.zeros((n_pixels, n_pixels, wave.size), dtype=float)
    wavefront_stack = np.zeros((n_pixels, n_pixels, wave.size), dtype=float)
    areapix = np.zeros(wave.size, dtype=float)
    areapixapod = np.zeros(wave.size, dtype=float)

    zcoeffs = np.asarray(updated.get("zcoeffs", np.array([0.0], dtype=float)), dtype=float).reshape(-1)
    for band_index, wavelength_nm in enumerate(wave):
        pupil_plane_size_mm = ref_pupil_size_mm * (float(wavelength_nm) / max(measured_wavelength_nm, 1e-12))
        pupil_sample_spacing_mm = pupil_plane_size_mm / max(n_pixels, 1)
        pupil_pos = sample_positions * pupil_sample_spacing_mm
        xpos, ypos = np.meshgrid(pupil_pos, -pupil_pos)
        norm_radius = np.sqrt(xpos**2 + ypos**2) / max(measured_pupil_mm / 2.0, 1e-12)
        theta = np.arctan2(ypos, xpos)
        calc_radius_index = norm_radius <= calc_radius
        aperture_mask = _wvf_aperture_mask(n_pixels, calc_radius_index, aperture=aperture)
        base_aperture_mask = np.asarray(aperture_mask, dtype=float).copy()
        if local_compute_sce:
            rho = _sce_rho_for_wave(sce_params, float(wavelength_nm))
            xo_mm = float(sce_params.get("xo_mm", 0.0))
            yo_mm = float(sce_params.get("yo_mm", 0.0))
            aperture_mask = aperture_mask * np.power(10.0, -rho * ((xpos - xo_mm) ** 2 + (ypos - yo_mm) ** 2))

        lca_microns = _wvf_lca_microns(updated, float(wavelength_nm))
        wavefront_aberrations_um = _zernike_surface_osa(_wvf_zcoeffs_with_lca(zcoeffs, lca_microns), norm_radius, theta)
        wavefront_aberrations_um[norm_radius > calc_radius] = 0.0
        wavefront_stack[:, :, band_index] = wavefront_aberrations_um
        pupil_phase = np.exp(-1j * 2.0 * np.pi * wavefront_aberrations_um / max(float(wavelength_nm) * 1e-3, 1e-12))
        local_pupil = aperture_mask * pupil_phase
        pupil_function[:, :, band_index] = local_pupil
        areapix[band_index] = float(np.sum(np.abs(base_aperture_mask)))
        areapixapod[band_index] = float(np.sum(np.abs(local_pupil)))
        if compute_psf:
            psf = np.abs(np.fft.fftshift(np.fft.fft2(local_pupil))) ** 2
            psf_sum = float(np.sum(psf))
            if psf_sum > 0.0:
                psf = psf / psf_sum
            if bool(updated.get("flip_psf_upside_down", False)):
                psf = np.flipud(psf)
            if bool(updated.get("rotate_psf_90_degs", False)):
                psf = np.rot90(psf)
            psf_stack[:, :, band_index] = psf

    updated["computed"] = True
    updated["sce_params"] = sce_params
    updated["pupil_function"] = pupil_function if compute_pupil_function else None
    updated["pupil_amplitude"] = np.abs(pupil_function) if compute_pupil_function else None
    updated["pupil_phase"] = np.angle(pupil_function) if compute_pupil_function else None
    updated["wavefront_aberrations_um"] = wavefront_stack if compute_pupil_function else None
    updated["psf"] = psf_stack if compute_psf else None
    updated["pupil_support"] = sample_positions.copy()
    updated["areapix"] = areapix
    updated["areapixapod"] = areapixapod
    return updated


def wvf_pupil_function(
    wvf: dict[str, Any],
    *args: Any,
) -> dict[str, Any]:
    options = _parse_key_value_options(args, "wvfPupilFunction")
    aperture = options.pop("aperturefunction", None)
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfPupilFunction parameter: {unsupported}")

    updated = dict(wvf)
    if aperture is not None:
        updated["aperture_function"] = np.asarray(aperture, dtype=float).copy()
    return _wvf_compute_pupil_function_explicit(updated, aperture=aperture)


def wvf_compute_psf(
    wvf: dict[str, Any],
    *args: Any,
) -> dict[str, Any]:
    options = _parse_key_value_options(args, "wvfComputePSF")
    compute_pupil_func = bool(options.pop("computepupilfunc", False))
    lca = bool(options.pop("lca", False))
    if lca:
        raise UnsupportedOptionError("wvfComputePSF", "lca=true")
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfComputePSF parameter: {unsupported}")

    current = dict(wvf)
    pupil_available = current.get("pupil_function") is not None
    if compute_pupil_func or not pupil_available:
        current = _wvf_compute_pupil_function_explicit(current)
    return _wvf_compute_psf_from_pupil_function(current, current.get("pupil_function"))


def _wvf_parse_pupil_function_options(show_bar: Any | None, args: tuple[Any, ...], context: str) -> dict[str, Any]:
    key_args = args
    if isinstance(show_bar, str):
        key_args = (show_bar, *args)
    elif show_bar is not None:
        _logical_scalar(show_bar)
    return _parse_key_value_options(key_args, context) if key_args else {}


def wvf_compute_cone_psf(
    wvf: dict[str, Any],
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    current = dict(wvf if wvf.get("psf") is not None else wvf_compute(wvf))
    psf = np.asarray(current.get("psf"), dtype=float)
    if psf.ndim == 2:
        psf = psf[:, :, np.newaxis]
    wave = _wvf_wave_values(current)
    cone_psf_info = _normalize_cone_psf_info(current.get("calc_cone_psf_info"), asset_store=asset_store)
    cone_weights = _wvf_cone_psf_weights(cone_psf_info, wave)
    cone_psf = np.asarray(np.tensordot(psf, cone_weights.T, axes=([2], [0])), dtype=float)
    cone_psf_sum = np.sum(cone_psf, axis=(0, 1), keepdims=True)
    cone_psf_sum[cone_psf_sum == 0.0] = 1.0
    cone_psf = cone_psf / cone_psf_sum
    sce_fraction = np.asarray(wvf_get(current, "sce fraction", wave), dtype=float).reshape(-1)
    cone_sce_fraction = np.asarray(cone_weights @ sce_fraction, dtype=float).reshape(-1)
    return cone_psf, cone_sce_fraction


def wvf_compute_cone_average_criterion_radius(
    wvf: dict[str, Any],
    defocus_diopters: Any,
    criterion_fraction: float,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    pupil_diameter_mm = float(wvf_get(wvf, "measured pupil size", "mm"))
    defocus_microns = np.asarray(
        wvf_defocus_diopters_to_microns(defocus_diopters, pupil_diameter_mm),
        dtype=float,
    ).reshape(-1)
    updated = wvf_set(dict(wvf), "zcoeffs", defocus_microns, "defocus")
    updated = wvf_compute(updated)
    cone_psf, _ = wvf_compute_cone_psf(updated, asset_store=asset_store)
    cone_weighting = np.asarray(
        wvf_get(updated, "calc cone psf info")["cone_weighting"],
        dtype=float,
    ).reshape(-1)
    cone_criterion_radii = np.asarray(
        [psf_find_criterion_radius(cone_psf[:, :, index], criterion_fraction) for index in range(cone_psf.shape[2])],
        dtype=float,
    )
    cone_avg_criterion_radius = float(np.sum(cone_weighting * cone_criterion_radii))
    return cone_avg_criterion_radius, cone_criterion_radii, updated


def wvf_compute_pupil_function_custom_lca(
    wvf: dict[str, Any],
    show_bar: Any | None = None,
    *args: Any,
) -> dict[str, Any]:
    options = _wvf_parse_pupil_function_options(show_bar, args, "wvfComputePupilFunctionCustomLCA")
    no_lca = _logical_scalar(options.pop("nolca", False))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfComputePupilFunctionCustomLCA parameter: {unsupported}")
    current = dict(wvf)
    if no_lca:
        current = wvf_set(current, "lca method", "none")
    return _wvf_compute_pupil_function_explicit(current)


def wvf_compute_pupil_function_custom_lca_from_master(
    wvf: dict[str, Any],
    show_bar: Any | None = None,
    *args: Any,
) -> dict[str, Any]:
    options = _wvf_parse_pupil_function_options(show_bar, args, "wvfComputePupilFunctionCustomLCAFromMaster")
    no_lca = _logical_scalar(options.pop("nolca", False))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfComputePupilFunctionCustomLCAFromMaster parameter: {unsupported}")
    current = dict(wvf)
    if no_lca:
        current = wvf_set(current, "lca method", "none")
    return _wvf_compute_pupil_function_explicit(current)


def wvf_compute_pupil_function_from_master(
    wvf: dict[str, Any],
    show_bar: Any | None = None,
    *args: Any,
) -> dict[str, Any]:
    options = _wvf_parse_pupil_function_options(show_bar, args, "wvfComputePupilFunctionFromMaster")
    no_lca = _logical_scalar(options.pop("nolca", False))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfComputePupilFunctionFromMaster parameter: {unsupported}")
    current = dict(wvf)
    if no_lca:
        current = wvf_set(current, "lca method", "none")
    return _wvf_compute_pupil_function_explicit(current)


wvfComputeConePSF = wvf_compute_cone_psf
wvfComputeConeAverageCriterionRadius = wvf_compute_cone_average_criterion_radius
wvfComputePupilFunctionCustomLCA = wvf_compute_pupil_function_custom_lca
wvfComputePupilFunctionCustomLCAFromMaster = wvf_compute_pupil_function_custom_lca_from_master
wvfComputePupilFunctionFromMaster = wvf_compute_pupil_function_from_master


def wvf_clear_data(wvf: dict[str, Any]) -> dict[str, Any]:
    updated = dict(wvf)
    updated["psf"] = None
    updated["wavefront_aberrations_um"] = None
    updated["pupil_function"] = None
    updated["pupil_amplitude"] = None
    updated["pupil_phase"] = None
    updated["areapix"] = None
    updated["areapixapod"] = None
    updated["computed"] = False
    return updated


def psf_find_peak(input_psf: Any) -> tuple[int, int]:
    """Return the 1-based row/column location of the PSF peak."""

    psf = np.asarray(input_psf, dtype=float)
    if psf.ndim != 2:
        raise ValueError("psfFindPeak expects a 2-D PSF array.")
    peak_index = int(np.argmax(psf))
    peak_row, peak_col = np.unravel_index(peak_index, psf.shape)
    return int(peak_row + 1), int(peak_col + 1)


def psf_volume(psf: Any, x_samples: Any, y_samples: Any) -> tuple[float, np.ndarray]:
    """Compute PSF volume and the corresponding unit-volume normalized PSF."""

    psf_array = np.asarray(psf, dtype=float)
    x = np.asarray(x_samples, dtype=float)
    y = np.asarray(y_samples, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        if x.size < 2 or y.size < 2:
            raise ValueError("psfVolume requires at least two x and y samples.")
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
    elif x.ndim == 2 and y.ndim == 2:
        if x.shape[1] < 2 or y.shape[0] < 2:
            raise ValueError("psfVolume matrix support requires at least 2x2 samples.")
        dx = float(x[0, 1] - x[0, 0])
        dy = float(y[1, 0] - y[0, 0])
    else:
        raise ValueError("Unexpected X,Y format. Both should be vectors or matrices.")

    volume = float(np.sum(psf_array) * dx * dy)
    normalized = np.asarray(psf_array / volume if volume != 0.0 else psf_array.copy(), dtype=float)
    return volume, normalized


def psf_center(input_psf: Any) -> tuple[np.ndarray, int, int]:
    """Shift the PSF peak to the center of the grid using linear interpolation."""

    in_psf = np.asarray(input_psf, dtype=float)
    if in_psf.ndim != 2:
        raise ValueError("psfCenter expects a 2-D PSF array.")

    peak_row, peak_col = psf_find_peak(in_psf)
    rows, cols = in_psf.shape
    x_in = np.arange(1, cols + 1, dtype=float) - float(peak_col)
    y_in = np.arange(1, rows + 1, dtype=float) - float(peak_row)
    x_out = np.arange(1, cols + 1, dtype=float) - float(np.floor(cols / 2.0) + 1.0)
    y_out = np.arange(1, rows + 1, dtype=float) - float(np.floor(rows / 2.0) + 1.0)

    interpolator = RegularGridInterpolator((y_in, x_in), in_psf, bounds_error=False, fill_value=0.0)
    xx, yy = np.meshgrid(x_out, y_out)
    centered = np.asarray(interpolator(np.stack((yy, xx), axis=-1)), dtype=float)

    input_sum = float(np.sum(in_psf))
    centered_sum = float(np.sum(centered))
    if centered_sum > 0.0:
        centered = centered * (input_sum / centered_sum)
    return centered, peak_row, peak_col


def psf_find_criterion_radius(input_psf: Any, criterion: float) -> float:
    """Find the radius, in pixels, containing the requested PSF mass fraction."""

    psf = np.asarray(input_psf, dtype=float)
    if psf.ndim != 2 or psf.shape[0] != psf.shape[1]:
        raise ValueError("psfFindCriterionRadius expects a square 2-D PSF array.")
    if not (0.0 <= float(criterion) <= 1.0):
        raise ValueError("criterion must be between 0 and 1.")

    normalized = psf / max(float(np.sum(psf)), 1e-12)
    centered, _, _ = psf_center(normalized)
    peak_row, peak_col = psf_find_peak(centered)
    yy, xx = np.indices(centered.shape, dtype=float)
    radius_mat = np.sqrt((yy - (peak_row - 1)) ** 2 + (xx - (peak_col - 1)) ** 2)
    max_radius = float(np.max(radius_mat))
    radius = max_radius
    previous_mass = 0.0
    for sample_radius in range(1, int(np.floor(max_radius)) + 1):
        mass = float(np.sum(centered[radius_mat <= float(sample_radius)]))
        if mass > float(criterion):
            if mass <= previous_mass:
                return float(sample_radius)
            lam = (float(criterion) - previous_mass) / (mass - previous_mass)
            return (1.0 - lam) * float(sample_radius - 1) + lam * float(sample_radius)
        previous_mass = mass
    return radius


psfFindPeak = psf_find_peak
psfVolume = psf_volume
psfCenter = psf_center
psfFindCriterionRadius = psf_find_criterion_radius


def psf_to_lsf(psf: Any, *args: Any) -> np.ndarray:
    """Derive a horizontal or vertical line-spread function from a PSF stack."""

    options = _parse_key_value_options(args, "psf2lsf")
    direction = param_format(options.pop("direction", "horizontal"))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported psf2lsf parameter: {unsupported}")
    if direction not in {"horizontal", "vertical"}:
        raise ValueError("psf2lsf direction must be 'horizontal' or 'vertical'.")

    psf_array = np.asarray(psf, dtype=float)
    if psf_array.ndim == 2:
        psf_array = psf_array[:, :, np.newaxis]
    elif psf_array.ndim != 3:
        raise ValueError("psf2lsf expects a 2-D PSF or a 3-D PSF stack.")

    if direction == "horizontal":
        length = psf_array.shape[0]
        samples = np.zeros((length, psf_array.shape[2]), dtype=float)
        for band in range(psf_array.shape[2]):
            psf_fft = np.fft.fft2(psf_array[:, :, band])
            samples[:, band] = np.asarray(np.real_if_close(np.fft.ifft(psf_fft[:, 0])), dtype=float)
    else:
        length = psf_array.shape[1]
        samples = np.zeros((length, psf_array.shape[2]), dtype=float)
        for band in range(psf_array.shape[2]):
            psf_fft = np.fft.fft2(psf_array[:, :, band])
            samples[:, band] = np.asarray(np.real_if_close(np.fft.ifft(psf_fft[0, :])), dtype=float)

    if samples.shape[1] == 1:
        return samples[:, 0]
    return samples


def lsf_to_circular_psf(lsf: Any) -> np.ndarray:
    """Convert a symmetric line-spread function into a circularly symmetric PSF."""

    lsf_array = np.asarray(lsf, dtype=float).reshape(-1)
    n = int(lsf_array.size)
    if n == 0:
        raise ValueError("lsf2circularpsf expects a non-empty LSF.")

    center = (n // 2) + 1 if n % 2 else (n // 2) + 1
    lsf_norm = lsf_array / max(float(np.sum(lsf_array)), 1e-12)
    otf_1d = np.asarray(
        np.real_if_close(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(lsf_norm)))),
        dtype=float,
    )
    otf_1d = np.abs(otf_1d)

    u, v = np.meshgrid(np.arange(1, n + 1, dtype=float), np.arange(1, n + 1, dtype=float), indexing="xy")
    r_uv = np.sqrt((u - float(center)) ** 2 + (v - float(center)) ** 2)
    if n % 2 == 0:
        freq = np.arange(0, n // 2, dtype=float)
    else:
        freq = np.arange(0, (n // 2) + 1, dtype=float)
    radial_otf = np.asarray(otf_1d[center - 1 :], dtype=float)
    otf_2d = np.interp(r_uv.reshape(-1), freq, radial_otf, left=float(radial_otf[0]), right=0.0).reshape(n, n)

    psf_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf_2d)))
    psf = np.abs(np.asarray(np.real_if_close(psf_complex), dtype=float))
    psf_sum = float(np.sum(psf))
    if psf_sum > 0.0:
        psf = psf / psf_sum
    return psf


def psf_circularly_average(input_psf: Any) -> np.ndarray:
    """Circularly average the provided PSF while preserving total volume."""

    in_psf = np.asarray(input_psf, dtype=float)
    if in_psf.ndim != 2 or in_psf.shape[0] != in_psf.shape[1]:
        raise ValueError("psfCircularlyAverage expects a square 2-D PSF array.")

    n_linear_pixels = int(in_psf.shape[0])
    peak_row, peak_col = psf_find_peak(in_psf)
    yy, xx = np.meshgrid(
        np.arange(1, n_linear_pixels + 1, dtype=float),
        np.arange(1, n_linear_pixels + 1, dtype=float),
        indexing="ij",
    )
    radius_mat = np.sqrt((xx - float(peak_col)) ** 2 + (yy - float(peak_row)) ** 2)
    out_psf = np.zeros_like(in_psf)
    radii = np.linspace(0.0, 0.75 * float(n_linear_pixels), int(round(n_linear_pixels)))

    for index in range(len(radii) - 1):
        mask = np.logical_and(radius_mat >= radii[index], radius_mat < radii[index + 1])
        if np.any(mask):
            out_psf[mask] = float(np.mean(in_psf[mask]))

    out_sum = float(np.sum(out_psf))
    if out_sum > 0.0:
        out_psf = float(np.sum(in_psf)) * out_psf / out_sum
    return out_psf


def psf_average_multiple(input_psfs: Any, check_in_sf_domain: Any | None = None) -> np.ndarray:
    """Average a 3-D PSF stack over the third dimension."""

    del check_in_sf_domain
    psf_stack = np.asarray(input_psfs, dtype=float)
    if psf_stack.ndim == 2:
        return psf_stack.copy()
    if psf_stack.ndim != 3:
        raise ValueError("psfAverageMultiple expects a 2-D PSF or a 3-D PSF stack.")
    return np.mean(psf_stack, axis=2)


psf2lsf = psf_to_lsf
lsf2circularpsf = lsf_to_circular_psf
psfCircularlyAverage = psf_circularly_average
psfAverageMultiple = psf_average_multiple


def psf_to_zcoeff_error(
    zcoeffs: Any,
    psf_target: Any,
    pupil_size_mm: Any,
    z_pupil_diameter_mm: Any,
    pupil_plane_size_mm: Any,
    wave_um: Any,
    n_pixels: Any,
) -> float:
    coefficients = np.asarray(zcoeffs, dtype=float).reshape(-1).copy()
    if coefficients.size == 0:
        coefficients = np.zeros(1, dtype=float)
    coefficients[0] = 0.0
    if coefficients.size < 5:
        coefficients = np.pad(coefficients, (0, 5 - coefficients.size), constant_values=0.0)

    pixels = int(n_pixels)
    psf_reference = np.asarray(psf_target, dtype=float)
    if psf_reference.shape != (pixels, pixels):
        raise ValueError("psf_target shape must match n_pixels x n_pixels.")

    pupil_pos = (np.arange(pixels, dtype=float) + 1.0) - (np.floor(pixels / 2.0) + 1.0)
    pupil_pos = pupil_pos * (float(pupil_plane_size_mm) / max(pixels, 1))
    xpos, ypos = np.meshgrid(pupil_pos, pupil_pos, indexing="xy")
    ypos = -ypos

    z_pupil_diameter_mm = float(z_pupil_diameter_mm)
    pupil_size_mm = float(pupil_size_mm)
    norm_radius = np.sqrt(xpos**2 + ypos**2) / max(z_pupil_diameter_mm / 2.0, 1e-12)
    theta = np.arctan2(ypos, xpos)

    wavefront_aberrations_um = _zernike_surface_osa(coefficients, norm_radius, theta)
    pupilfuncphase = np.exp(-1j * 2.0 * np.pi * wavefront_aberrations_um / max(float(wave_um), 1e-12))
    pupilfuncphase[norm_radius > (pupil_size_mm / max(z_pupil_diameter_mm, 1e-12))] = 0.0

    amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupilfuncphase)))
    intensity = np.real(amp * np.conj(amp))
    psf = intensity / max(float(np.sum(intensity)), 1e-12)
    diff = psf_reference - psf
    return float(np.sqrt(np.mean(np.square(diff))))


psf2zcoeff = psf_to_zcoeff_error


def wvf_set(wvf: dict[str, Any], parameter: str, value: Any, *args: Any) -> dict[str, Any]:
    key = param_format(parameter)
    updated = dict(wvf)

    if key in {"name", "type", "sampleintervaldomain"}:
        mapped_key = {
            "name": "name",
            "type": "type",
            "sampleintervaldomain": "sample_interval_domain",
        }[key]
        updated[mapped_key] = str(value)
        return updated

    if key == "lcamethod":
        updated["lca_method"] = "none" if value is None else (value if callable(value) else str(value))
        return updated

    if key in {"wave", "wavelength", "wavelengths", "calcwave", "calcwavelengths", "wls"}:
        updated["wave"] = np.asarray(value, dtype=float).reshape(-1)
        if "sce_params" in updated:
            updated["sce_params"] = _normalize_sce_params(updated["wave"], updated.get("sce_params"))
        return updated

    if key in {"zcoeffs", "zcoeff", "zcoef"}:
        if not args:
            updated["zcoeffs"] = np.asarray(value, dtype=float).reshape(-1).copy()
            return updated
        indices = _coerce_wvf_zcoeff_indices(args[0])
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size != indices.size:
            raise ValueError("Wavefront coefficient values must match the number of requested indices.")
        zcoeffs = np.asarray(updated.get("zcoeffs", np.array([0.0], dtype=float)), dtype=float).reshape(-1).copy()
        max_index = int(np.max(indices)) if indices.size > 0 else -1
        if max_index >= zcoeffs.size:
            zcoeffs = np.pad(zcoeffs, (0, max_index + 1 - zcoeffs.size), constant_values=0.0)
        zcoeffs[indices] = values
        updated["zcoeffs"] = zcoeffs
        return updated

    if key in {"calcpupildiameter", "calcpupilsize", "calculatedpupil", "calculatedpupildiameter"}:
        updated["calc_pupil_diameter_mm"] = float(value) / _spatial_unit_scale(args[0] if args else "mm") * 1e3
        updated["f_number"] = (float(updated.get("focal_length_m", DEFAULT_WVF_FOCAL_LENGTH_M)) * 1e3) / max(
            float(updated["calc_pupil_diameter_mm"]), 1e-12
        )
        return updated

    if key in {"measuredpupil", "measuredpupilsize", "measuredpupildiameter", "measuredpupilmm", "pupildiameter", "pupilsize"}:
        updated["measured_pupil_diameter_mm"] = float(value) / _spatial_unit_scale(args[0] if args else "mm") * 1e3
        return updated

    if key in {"measuredwl", "measuredwave", "measuredwavelength"}:
        updated["measured_wavelength_nm"] = float(value)
        return updated

    if key in {"spatialsamples", "numberspatialsamples", "npixels", "fieldsizepixels"}:
        updated["spatial_samples"] = int(value)
        return updated

    if key in {"refpupilplanesize", "pupilplanesize", "refpupilplanesizemm", "fieldsizemm", "fieldsizemm"}:
        updated["ref_pupil_plane_size_mm"] = float(value) / _spatial_unit_scale(args[0] if args else "mm") * 1e3
        return updated

    if key in {"psfsampleinterval", "refpsfsampleinterval", "refpsfarcminpersample", "refpsfarcminperpixel"}:
        radians_per_pixel = _wvf_angle_scalar_to_radians(value, args[0] if args else "min")
        measured_wavelength_mm = float(wvf_get(updated, "measured wavelength", "mm"))
        field_size_mm = measured_wavelength_mm / max(radians_per_pixel, 1e-12)
        updated = wvf_set(updated, "field size mm", field_size_mm)
        updated["computed"] = False
        return updated

    if key in {"psfsamplespacing", "psfdx"}:
        psf_spacing_mm = float(value) / _spatial_unit_scale(args[0] if args else "mm") * 1e3
        lambda_mm = float(np.asarray(wvf_get(updated, "wave", "mm"), dtype=float).reshape(-1)[0])
        focal_length_mm = float(wvf_get(updated, "focal length", "mm"))
        n_pixels = int(wvf_get(updated, "npixels"))
        pupil_spacing_mm = lambda_mm * focal_length_mm / max(psf_spacing_mm * n_pixels, 1e-12)
        updated = wvf_set(updated, "field size mm", pupil_spacing_mm * n_pixels)
        updated["computed"] = False
        return updated

    if key in {"focallength"}:
        updated["focal_length_m"] = float(value) / _spatial_unit_scale(args[0] if args else "m")
        if float(updated.get("calc_pupil_diameter_mm", 0.0)) > 0.0:
            updated["f_number"] = (updated["focal_length_m"] * 1e3) / float(updated["calc_pupil_diameter_mm"])
        return updated

    if key in {"fnumber", "f#"}:
        updated["f_number"] = float(value)
        focal_length_mm = float(updated.get("focal_length_m", DEFAULT_WVF_FOCAL_LENGTH_M)) * 1e3
        updated["calc_pupil_diameter_mm"] = focal_length_mm / max(float(updated["f_number"]), 1e-12)
        return updated

    if key in {"defocusdiopters", "calcobserverfocuscorrection"}:
        defocus_microns = np.asarray(
            wvf_defocus_diopters_to_microns(value, updated.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM)),
            dtype=float,
        ).reshape(-1)
        return wvf_set(updated, "zcoeffs", defocus_microns, "defocus")

    if key in {"compute_sce", "computesce"}:
        updated["compute_sce"] = bool(value)
        return updated

    if key in {"sceparams", "stilescrawford"}:
        updated["sce_params"] = _normalize_sce_params(np.asarray(updated.get("wave", DEFAULT_WAVE), dtype=float), value)
        return updated

    if key in {"calcconepsfinfo"}:
        updated["calc_cone_psf_info"] = _normalize_cone_psf_info(value)
        return updated

    if key in {"aperturefunction", "aperturefunc"}:
        updated["aperture_function"] = np.asarray(value, dtype=float).copy()
        return updated

    if key in {"flippsfupsidedown"}:
        updated["flip_psf_upside_down"] = bool(value)
        return updated

    if key in {"rotatepsf90degs"}:
        updated["rotate_psf_90_degs"] = bool(value)
        return updated

    raise KeyError(f"Unsupported wvfSet parameter: {parameter}")


def wvf_get(wvf: dict[str, Any], parameter: str, *args: Any) -> Any:
    key = param_format(parameter)

    if key in {"name", "type"}:
        return wvf.get(key)

    if key in {"wave", "wavelength", "wavelengths", "calcwave", "calcwavelengths", "wls"}:
        values = np.asarray(wvf.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1).copy()
        unit = None
        index = None
        if args:
            if isinstance(args[0], str):
                unit = args[0]
                if len(args) > 1:
                    index = int(np.asarray(args[1], dtype=int).reshape(-1)[0])
            else:
                index = int(np.asarray(args[0], dtype=int).reshape(-1)[0])
        if unit is not None:
            values = values * _wave_unit_scale(unit)
        if index is not None:
            return float(values[index - 1])
        return values

    if key in {"calcnwave", "nwave", "numbercalcwavelengths", "nwavelengths"}:
        return int(_wvf_wave_values(wvf).size)

    if key in {"sampleintervaldomain"}:
        return str(wvf.get("sample_interval_domain", "psf"))

    if key in {"lcamethod"}:
        raw_method = wvf.get("lca_method", "none")
        return "none" if raw_method is None else raw_method

    if key in {"calcconepsfinfo"}:
        return _export_cone_psf_info(wvf.get("calc_cone_psf_info"))

    if key in {"umperdegree", "umperdeg"}:
        # MATLAB/Octave wvfGet('um per degree') reports the mm-per-degree
        # scale for this legacy getter path.
        return _wvf_um_per_degree(wvf) / 1e3

    if key in {"zcoeffs", "zcoeff", "zcoef"}:
        zcoeffs = np.asarray(wvf.get("zcoeffs", np.array([0.0], dtype=float)), dtype=float).reshape(-1)
        if not args:
            return zcoeffs.copy()
        indices = _coerce_wvf_zcoeff_indices(args[0])
        values = zcoeffs[indices]
        if values.size == 1:
            return float(values[0])
        return values

    if key in {"calcpupildiameter", "calcpupilsize", "calculatedpupil", "calculatedpupildiameter"}:
        return (float(wvf.get("calc_pupil_diameter_mm", DEFAULT_WVF_CALC_PUPIL_DIAMETER_MM)) / 1e3) * _spatial_unit_scale(
            args[0] if args else "mm"
        )

    if key in {"zpupildiameter", "zpupilsize", "measuredpupil", "measuredpupilsize", "measuredpupildiameter", "measuredpupilmm", "pupildiameter", "pupilsize"}:
        return (float(wvf.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM)) / 1e3) * _spatial_unit_scale(
            args[0] if args else "mm"
        )

    if key in {"measuredwl", "measuredwave", "measuredwavelength"}:
        value_nm = float(wvf.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
        if args:
            return value_nm * _wave_unit_scale(args[0])
        return value_nm

    if key in {"spatialsamples", "numberspatialsamples", "npixels", "fieldsizepixels"}:
        return int(wvf.get("spatial_samples", DEFAULT_WVF_SPATIAL_SAMPLES))

    if key in {"middlerow"}:
        return int(_wvf_middle_row(wvf))

    if key in {"refpupilplanesize", "pupilplanesize", "refpupilplanesizemm", "fieldsizemm"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "mm")
        if key in {"refpupilplanesize", "refpupilplanesizemm", "fieldsizemm"}:
            size_m = _wvf_ref_pupil_plane_size_m(wvf)
        else:
            size_m = _wvf_pupil_plane_size_m(wvf, wavelength_nm)
        return size_m * _spatial_unit_scale(unit)

    if key in {"focallength"}:
        return float(wvf.get("focal_length_m", DEFAULT_WVF_FOCAL_LENGTH_M)) * _spatial_unit_scale(args[0] if args else "m")

    if key in {"fnumber", "f#"}:
        return float(wvf.get("f_number", 4.0))

    if key in {"defocusdiopters", "calcobserverfocuscorrection"}:
        defocus_microns = float(wvf_get(wvf, "zcoeffs", "defocus"))
        return float(
            np.asarray(
                wvf_defocus_microns_to_diopters(
                    defocus_microns, wvf.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM)
                ),
                dtype=float,
            ).reshape(-1)[0]
        )

    if key == "sce":
        sce_params = wvf.get("sce_params", {})
        return _sce_export(
            np.asarray(sce_params.get("wave", _wvf_wave_values(wvf)), dtype=float),
            np.asarray(sce_params.get("rho", np.zeros(_wvf_wave_values(wvf).size, dtype=float)), dtype=float),
            xo_mm=float(sce_params.get("xo_mm", 0.0)),
            yo_mm=float(sce_params.get("yo_mm", 0.0)),
        )

    if key in {"scex0", "scexo"}:
        return float(wvf.get("sce_params", {}).get("xo_mm", 0.0))

    if key in {"scey0", "sceyo"}:
        return float(wvf.get("sce_params", {}).get("yo_mm", 0.0))

    if key in {"scewavelength", "scewavelengths", "scewave"}:
        values = np.asarray(wvf.get("sce_params", {}).get("wave", _wvf_wave_values(wvf)), dtype=float).reshape(-1).copy()
        if args:
            values = values * _wave_unit_scale(args[0])
        return values

    if key == "scerho":
        sce_params = wvf.get("sce_params", {})
        exported = _sce_export(
            np.asarray(sce_params.get("wave", _wvf_wave_values(wvf)), dtype=float),
            np.asarray(sce_params.get("rho", np.zeros(_wvf_wave_values(wvf).size, dtype=float)), dtype=float),
            xo_mm=float(sce_params.get("xo_mm", 0.0)),
            yo_mm=float(sce_params.get("yo_mm", 0.0)),
        )
        if not args:
            return np.asarray(exported["rho"], dtype=float).copy()
        return sce_get(exported, "rho", args[0])

    if key in {"scefraction", "scefrac", "stilescrawfordeffectfraction"}:
        computed = wvf if wvf.get("areapix") is not None and wvf.get("areapixapod") is not None else wvf_compute(wvf, compute_psf=False)
        base = np.asarray(computed.get("areapix"), dtype=float).reshape(-1)
        apodized = np.asarray(computed.get("areapixapod"), dtype=float).reshape(-1)
        values = np.divide(apodized, base, out=np.zeros_like(apodized, dtype=float), where=base != 0.0)
        if not args:
            return values
        idx = wvf_wave_to_idx(computed, args[0]) - 1
        selected = values[idx]
        if selected.size == 1:
            return float(selected[0])
        return selected.copy()

    if key in {"areapix"}:
        computed = wvf if wvf.get("areapix") is not None else wvf_compute(wvf, compute_psf=False)
        values = np.asarray(computed.get("areapix"), dtype=float).reshape(-1)
        if not args:
            return values.copy()
        idx = wvf_wave_to_idx(computed, args[0]) - 1
        selected = values[idx]
        if selected.size == 1:
            return float(selected[0])
        return selected.copy()

    if key in {"areapixapod"}:
        computed = wvf if wvf.get("areapixapod") is not None else wvf_compute(wvf, compute_psf=False)
        values = np.asarray(computed.get("areapixapod"), dtype=float).reshape(-1)
        if not args:
            return values.copy()
        idx = wvf_wave_to_idx(computed, args[0]) - 1
        selected = values[idx]
        if selected.size == 1:
            return float(selected[0])
        return selected.copy()

    if key in {"conepsf"}:
        return wvf_compute_cone_psf(wvf)[0]

    if key in {"sceconesfraction", "conescefraction"}:
        return wvf_compute_cone_psf(wvf)[1]

    if key in {"sceparams", "stilescrawford"}:
        sce_params = wvf.get("sce_params", {})
        return _sce_export(
            np.asarray(sce_params.get("wave", _wvf_wave_values(wvf)), dtype=float),
            np.asarray(sce_params.get("rho", np.zeros(_wvf_wave_values(wvf).size, dtype=float)), dtype=float),
            xo_mm=float(sce_params.get("xo_mm", 0.0)),
            yo_mm=float(sce_params.get("yo_mm", 0.0)),
        )

    if key in {"aperturefunction", "aperturefunc"}:
        aperture = wvf.get("aperture_function")
        if aperture is None:
            raise KeyError("Wavefront aperture function has not been set.")
        return np.asarray(aperture, dtype=float).copy()

    if key in {"flippsfupsidedown"}:
        return bool(wvf.get("flip_psf_upside_down", False))

    if key in {"rotatepsf90degs"}:
        return bool(wvf.get("rotate_psf_90_degs", False))

    if key in {"psfanglepersample", "psfangularsample", "angleperpixel", "angperpix"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "min")
        return float(_wvf_angle_samples_for_unit(np.array([_wvf_psf_angle_per_sample_deg(wvf, wavelength_nm)], dtype=float), unit)[0])

    if key in {"psfsamplespacing", "psfsampleinterval", "refpsfsampleinterval", "refpsfarcminpersample", "refpsfarcminperpixel"}:
        unit = str(args[0]) if args else "min"
        measured_wavelength_nm = float(wvf.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
        spacing_deg = _wvf_psf_angle_per_sample_deg(wvf, measured_wavelength_nm)
        return float(_wvf_angle_samples_for_unit(np.array([spacing_deg], dtype=float), unit)[0])

    if key in {"psfangularsamples"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "min")
        return _wvf_psf_angular_samples(wvf, unit, wavelength_nm)

    if key in {"psfsupport", "psfspatialsample", "psfspatialsamples", "samplesspace", "supportspace", "spatialsupport"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "deg")
        return _wvf_psf_spatial_samples(wvf, unit, wavelength_nm)

    if key in {"pupilpositions", "pupilpos", "pupilsamples"}:
        wavelength_nm, unit = _wvf_parse_wave_and_unit_args(wvf, args, "mm")
        spacing = float(wvf_get(wvf, "pupil sample spacing", unit, wavelength_nm))
        n_pixels = int(wvf_get(wvf, "npixels"))
        positions = ((np.arange(n_pixels, dtype=float) + 1.0) - _wvf_middle_row(wvf)) * spacing
        return positions

    if key in {"pupilsamplespacing", "pupilsampleinterval"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "mm")
        return float(_wvf_pupil_plane_size_m(wvf, wavelength_nm) / max(int(wvf_get(wvf, "npixels")), 1) * _spatial_unit_scale(unit))

    if key in {"refpupilplanesampleinterval", "fieldsamplesizemmperpixel", "refpupilplanesampleintervalmm", "fieldsamplesize"}:
        unit = str(args[0]) if args else "mm"
        return float(_wvf_ref_pupil_plane_size_m(wvf) / max(int(wvf_get(wvf, "npixels")), 1) * _spatial_unit_scale(unit))

    if key in {"pupilsupport", "pupilspatialsamples"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "mm")
        return _wvf_pupil_spatial_samples(wvf, unit, wavelength_nm)

    if key in {"1dpsf"}:
        wavelength_nm = float(np.asarray(args[0], dtype=float).reshape(-1)[0]) if args else _default_plot_wavelength(_wvf_wave_values(wvf))
        which_row = int(args[1]) if len(args) > 1 else int(_wvf_middle_row(wvf))
        psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
        return np.asarray(psf[which_row - 1, :], dtype=float).copy()

    if key in {"psfxaxis"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "mm")
        psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
        # MATLAB/Octave wvfGet('psf xaxis','um',...) returns the underlying
        # mm-scaled support for this path. Keep that quirk for parity.
        axis_unit = "mm" if unit == "um" else unit
        samp = _wvf_psf_spatial_samples(wvf, axis_unit, wavelength_nm)
        return {"samp": samp.copy(), "data": np.interp(samp, samp, psf[int(_wvf_middle_row(wvf)) - 1, :]).copy()}

    if key in {"psfyaxis"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "mm")
        psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
        # MATLAB/Octave wvfGet('psf yaxis','um',...) returns the underlying
        # mm-scaled support for this path. Keep that quirk for parity.
        axis_unit = "mm" if unit == "um" else unit
        samp = _wvf_psf_spatial_samples(wvf, axis_unit, wavelength_nm)
        return {"samp": samp.copy(), "data": np.interp(samp, samp, psf[:, int(_wvf_middle_row(wvf)) - 1]).copy()}

    if key in {"otfsupport"}:
        unit, wavelength_nm = _wvf_wave_for_query(wvf, args, "mm")
        return _wvf_otf_support(wvf, unit, wavelength_nm)

    if key in {"otf"}:
        wavelength_nm = float(np.asarray(args[0], dtype=float).reshape(-1)[0]) if args else _default_plot_wavelength(_wvf_wave_values(wvf))
        psf = np.asarray(wvf_get(wvf, "psf", wavelength_nm), dtype=float)
        return np.asarray(np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(psf))), dtype=np.complex128)

    if key in {"wavefrontaberrations", "wavefrontaberration"}:
        computed = wvf if wvf.get("wavefront_aberrations_um") is not None else wvf_compute(wvf, compute_psf=False)
        wavefront = computed.get("wavefront_aberrations_um")
        if wavefront is None:
            raise ValueError("Wavefront aberrations are unavailable.")
        if args:
            return np.asarray(_stack_plane_at_wavelength(np.asarray(wavefront), _wvf_wave_values(computed), float(np.asarray(args[0], dtype=float).reshape(-1)[0])), dtype=float)
        return np.asarray(wavefront, dtype=float).copy()

    if key in {"pupilfunction", "pupilfunc", "pupfun"}:
        computed = wvf if wvf.get("pupil_function") is not None else wvf_compute(wvf, compute_psf=False)
        pupil_function = computed.get("pupil_function")
        if pupil_function is None:
            raise ValueError("Pupil function is unavailable.")
        if args:
            return np.asarray(
                _stack_plane_at_wavelength(np.asarray(pupil_function), _wvf_wave_values(computed), float(np.asarray(args[0], dtype=float).reshape(-1)[0])),
                dtype=np.complex128,
            )
        return np.asarray(pupil_function, dtype=np.complex128).copy()

    if key in {"pupilfunctionamplitude", "pupilamplitude", "pupilamp", "aperture"}:
        return np.abs(np.asarray(wvf_get(wvf, "pupil function", *args), dtype=np.complex128))

    if key in {"pupilfunctionphase", "pupilphase"}:
        return np.angle(np.asarray(wvf_get(wvf, "pupil function", *args), dtype=np.complex128))

    if key in {"psf"}:
        computed = wvf if wvf.get("psf") is not None else wvf_compute(wvf)
        psf = computed.get("psf")
        if psf is None:
            raise ValueError("PSF is unavailable.")
        if args:
            return np.asarray(
                _stack_plane_at_wavelength(np.asarray(psf), _wvf_wave_values(computed), float(np.asarray(args[0], dtype=float).reshape(-1)[0])),
                dtype=float,
            )
        return np.asarray(psf, dtype=float).copy()

    raise KeyError(f"Unsupported wvfGet parameter: {parameter}")


def wvf_to_oi(wvf: dict[str, Any]) -> OpticalImage:
    current = dict(wvf if wvf.get("computed") else wvf_compute(wvf))
    return oi_create("wvf", current)


def _wvf_to_shift_invariant_psf_data(
    wvf: dict[str, Any],
    *,
    n_psf_samples: int = 128,
    um_per_sample: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = dict(wvf if wvf.get("psf") is not None else wvf_compute(wvf))
    wave = np.asarray(wvf_get(current, "wave"), dtype=float).reshape(-1)

    n_pix = int(n_psf_samples)
    if n_pix <= 0:
        raise ValueError("nPSFSamples must be positive.")
    sample_spacing_um = float(um_per_sample)
    if sample_spacing_um <= 0.0:
        raise ValueError("umPerSample must be positive.")

    out_samp = ((np.arange(1, n_pix + 1, dtype=float) - (np.floor(n_pix / 2.0) + 1.0)) * sample_spacing_um).reshape(-1)
    target_x, target_y = np.meshgrid(out_samp, out_samp, indexing="xy")
    target_points = np.column_stack((target_y.reshape(-1), target_x.reshape(-1)))
    psf = np.zeros((n_pix, n_pix, wave.size), dtype=float)

    for band_index, wavelength_nm in enumerate(wave):
        plane = np.asarray(wvf_get(current, "psf", float(wavelength_nm)), dtype=float)
        support = np.asarray(wvf_get(current, "psf spatial samples", "um", float(wavelength_nm)), dtype=float).reshape(-1)
        if support.shape == out_samp.shape and np.max(np.abs(support - out_samp)) < 1e-10:
            psf[:, :, band_index] = plane
            continue
        interpolator = RegularGridInterpolator((support, support), plane, bounds_error=False, fill_value=0.0)
        psf[:, :, band_index] = interpolator(target_points).reshape(n_pix, n_pix)

    um_per_samp = np.array([sample_spacing_um, sample_spacing_um], dtype=float)
    return _export_shift_invariant_psf_data({"psf": psf, "wave": wave.copy(), "umPerSamp": um_per_samp}), current


def wvf_to_si_psf(
    wvf: dict[str, Any],
    *args: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a WVF structure to MATLAB-style shift-invariant PSF data."""

    options = _parse_key_value_options(args, "wvf2SiPsf") if args else {}
    show_bar = _logical_scalar(options.pop("showbar", False)) if "showbar" in options else False
    n_psf_samples = int(options.pop("npsfsamples", 128))
    um_per_sample = float(options.pop("umpersample", 0.25))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvf2SiPsf parameter: {unsupported}")
    del show_bar
    return _wvf_to_shift_invariant_psf_data(
        wvf,
        n_psf_samples=n_psf_samples,
        um_per_sample=um_per_sample,
    )


def wvf_to_psf(
    wvf: dict[str, Any],
    show_bar: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del show_bar
    return _wvf_to_shift_invariant_psf_data(wvf, n_psf_samples=128, um_per_sample=0.25)


def wvf_apply(
    scene: Scene,
    wvf: dict[str, Any],
    *args: Any,
) -> OpticalImage:
    """Headless compatibility wrapper for the deprecated MATLAB `wvfApply` helper."""

    options = _parse_key_value_options(args, "wvfApply") if args else {}
    lca = _logical_scalar(options.pop("lca", False)) if "lca" in options else False
    compute_pupil_func = True
    if "computepupilfunc" in options:
        compute_pupil_func = _logical_scalar(options.pop("computepupilfunc"))
    if "computepupilfunction" in options:
        compute_pupil_func = _logical_scalar(options.pop("computepupilfunction"))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported wvfApply parameter: {unsupported}")

    computed_wvf = wvf_compute_psf(
        wvf,
        "lca",
        lca,
        "computepupilfunc",
        compute_pupil_func,
    )
    return oi_compute(wvf_to_oi(computed_wvf), scene)


def wvf_to_optics(wvf: dict[str, Any]) -> dict[str, Any]:
    si_data, current = wvf_to_psf(wvf, show_bar=False)
    oi = wvf_to_oi(current)
    optics = dict(oi.fields["optics"])
    normalized_psf = _normalize_shift_invariant_psf_data(si_data)
    optics["name"] = "wvf"
    optics["compute_method"] = "opticsotf"
    optics["wavefront"] = current
    optics["psf_data"] = normalized_psf
    optics.update(_custom_shift_invariant_otf_bundle(normalized_psf, samples=128, sample_spacing_um=0.25))
    return optics


wvfKeySynonyms = wvf_key_synonyms
wvf2SiPsf = wvf_to_si_psf
wvfApply = wvf_apply


def _normalize_public_optics(optics: dict[str, Any]) -> dict[str, Any]:
    current = dict(oi_create().fields["optics"])
    raw = dict(optics)
    mapped = dict(raw)
    remap = {
        "fNumber": "f_number",
        "focalLength": "focal_length_m",
        "computeMethod": "compute_method",
        "aberrationScale": "aberration_scale",
        "offaxis": "offaxis_method",
        "offaxisMethod": "offaxis_method",
        "nominalFocalLength": "nominal_focal_length_m",
        "psfData": "psf_data",
    }
    for source_key, target_key in remap.items():
        if source_key in mapped and target_key not in mapped:
            mapped[target_key] = mapped.pop(source_key)
    if isinstance(mapped.get("transmittance"), dict):
        mapped["transmittance"] = _normalize_transmittance_update(
            dict(mapped["transmittance"]),
            dict(current.get("transmittance", {})),
        )
    return _normalize_optics_update(mapped, current)


def _wave_for_optics_context(optics: dict[str, Any]) -> np.ndarray:
    current = dict(optics)
    transmittance = current.get("transmittance")
    if isinstance(transmittance, dict):
        wave = np.asarray(transmittance.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
        if wave.size > 0:
            return wave
    if isinstance(current.get("wavefront"), dict):
        wave = np.asarray(wvf_get(current["wavefront"], "wave"), dtype=float).reshape(-1)
        if wave.size > 0:
            return wave
    otf_wave = np.asarray(current.get("otf_wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if otf_wave.size > 0:
        return otf_wave
    psf_data = current.get("psf_data")
    if isinstance(psf_data, dict):
        wave = np.asarray(psf_data.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
        if wave.size > 0:
            return wave
    return DEFAULT_WAVE.copy()


def _oi_for_optics_context(optics: dict[str, Any]) -> OpticalImage:
    current = _normalize_public_optics(optics)
    oi = oi_create()
    oi = oi_set(oi, "wave", _wave_for_optics_context(current))
    oi.fields["optics"] = current
    oi.fields["compute_method"] = str(current.get("compute_method", oi.fields.get("compute_method", "")))
    _sync_oi_geometry_fields(oi)
    return oi


def optics_create(
    optics_type: str = "default",
    *args: Any,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    """Create a headless optics struct using the currently supported OI models."""

    normalized = param_format("default" if _is_empty_dispatch_placeholder(optics_type) else optics_type)
    valid_types = [
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
    if normalized.startswith("valid"):
        return valid_types.copy()
    if normalized == "empty":
        return {
            "type": "optics",
            "name": "empty",
        }
    if normalized in {"human", "humanmw"}:
        oi = oi_create("human mw", *args, asset_store=asset_store)
        optics = dict(oi.fields["optics"])
        optics.setdefault("type", "optics")
        optics.pop("wavefront", None)
        optics.pop("wvf", None)
        optics.pop("transmittance", None)
        optics.setdefault("vignetting", 0)
        return optics
    elif normalized in {"wvfhuman", "humanwvf"}:
        oi = oi_create("human", *args, asset_store=asset_store)
        optics = dict(oi.fields["optics"])
        optics.setdefault("type", "optics")
        optics["name"] = "humanwvf"
        if isinstance(optics.get("wavefront"), dict):
            optics.setdefault("wvf", dict(optics["wavefront"]))
        optics.pop("transmittance", None)
        optics.setdefault("vignetting", 0)
        return optics
    else:
        oi = oi_create(optics_type, *args, asset_store=asset_store)
    optics = dict(oi.fields["optics"])
    optics.setdefault("type", "optics")
    if normalized in {"default", "diffractionlimited", "diffraction"}:
        optics.setdefault("name", "standard (1/4-inch)")
    elif normalized in {"wvf", "shiftinvariant"}:
        optics.setdefault("name", "wvf")
        if isinstance(optics.get("wavefront"), dict):
            optics.setdefault("wvf", dict(optics["wavefront"]))
    optics.setdefault("vignetting", 0)
    return optics


def optics_get(optics: dict[str, Any], parameter: str, *args: Any) -> Any:
    """Get optics parameters through a temporary headless OI context."""

    return oi_get(_oi_for_optics_context(optics), parameter, *args)


def optics_set(optics: dict[str, Any], parameter: str, value: Any, *args: Any) -> dict[str, Any]:
    """Set optics parameters through a temporary headless OI context."""

    updated = oi_set(_oi_for_optics_context(optics), parameter, value, *args)
    return dict(updated.fields["optics"])


def optics_clear_data(optics: dict[str, Any]) -> dict[str, Any]:
    """Clear cached optics payloads without losing the core model fields."""

    current = dict(_normalize_public_optics(optics))
    for key in ("otf_data", "otf_fx", "otf_fy", "otf_wave", "cos4th_data", "cos4th_value"):
        current.pop(key, None)
    return current


def optics_description(optics: dict[str, Any] | None = None) -> str:
    """Return a short headless text summary for an optics struct."""

    current = optics_create("default") if optics is None else _normalize_public_optics(optics)
    name = str(current.get("name", "No name") or "No name")
    f_number = float(current.get("f_number", np.nan))
    focal_length = float(current.get("focal_length_m", np.nan))
    diameter = focal_length / max(f_number, 1.0e-12) if np.isfinite(focal_length) and np.isfinite(f_number) else np.nan
    numerical_aperture = 1.0 / (2.0 * f_number) if np.isfinite(f_number) and f_number > 0.0 else np.nan
    aperture_area = np.pi * (diameter / 2.0) ** 2 if np.isfinite(diameter) else np.nan
    return (
        f"Optics: {name}\n"
        f" NA       : \t{numerical_aperture:0.2e}\n"
        f" Aper Area: \t{aperture_area:0.2e} m^2\n"
        f" Aper Diam: \t{diameter:0.2e} m\n"
    )


def lens_list(
    star: str = "*.json",
    quiet: bool = False,
    *,
    asset_store: AssetStore | None = None,
) -> list[dict[str, str]]:
    """List the pinned upstream lens JSON descriptors."""

    lens_root = _store(asset_store).snapshot_root / "data" / "lens"
    files = [
        {"name": path.name, "path": str(path)}
        for path in sorted(lens_root.glob(star))
        if path.is_file()
    ]
    if not quiet:
        for index, item in enumerate(files, start=1):
            print(f"{index} - {item['name']}")
    return files


def optics_to_wvf(optics: dict[str, Any]) -> dict[str, Any]:
    """Create a lightweight WVF description from the current optics parameters."""

    current = _normalize_public_optics(optics)
    stored_wvf = current.get("wavefront")
    if isinstance(stored_wvf, dict):
        return dict(stored_wvf)
    focal_length_m = float(current.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M))
    f_number = float(current.get("f_number", 4.0))
    wvf = wvf_create(
        wave=_wave_for_optics_context(current),
        focal_length_m=focal_length_m,
        f_number=f_number,
    )
    return wvf_set(wvf, "calc pupil diameter", (focal_length_m * 1e3) / max(f_number, 1.0e-12), "mm")


opticsCreate = optics_create
opticsGet = optics_get
opticsSet = optics_set
opticsClearData = optics_clear_data
opticsDescription = optics_description
lensList = lens_list
optics2wvf = optics_to_wvf


def dl_core(rho: Any, in_cut_freq: Any) -> np.ndarray:
    """Compute the diffraction-limited OTF from radial support and cutoff frequency."""

    rho_array = np.asarray(rho, dtype=float)
    if rho_array.ndim != 2:
        raise ValueError("dlCore rho must be a 2-D support grid.")

    cutoff = np.asarray(in_cut_freq, dtype=float).reshape(-1)
    if cutoff.size == 0:
        raise ValueError("dlCore requires at least one cutoff frequency.")

    otf = np.zeros(rho_array.shape + (cutoff.size,), dtype=float)
    for band_index, cutoff_frequency in enumerate(cutoff):
        normalized = rho_array / max(float(cutoff_frequency), 1.0e-12)
        clipped = np.clip(normalized, 0.0, 1.0)
        plane = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
        plane[normalized >= 1.0] = 0.0
        otf[:, :, band_index] = np.fft.ifftshift(np.asarray(plane, dtype=float))

    if cutoff.size == 1:
        return np.asarray(otf[:, :, 0], dtype=float)
    return np.asarray(otf, dtype=float)


def dl_mtf(
    oi_or_optics: OpticalImage | dict[str, Any],
    f_support: Any | None = None,
    wavelength: Any | None = None,
    units: str = "cyclesPerDegree",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Headless `dlMTF` compatibility wrapper for OI or optics inputs."""

    current_oi = oi_or_optics if isinstance(oi_or_optics, OpticalImage) else None
    current_optics = _normalize_public_optics(_coerce_optics_struct(oi_or_optics))
    normalized_unit = param_format(units)

    if current_oi is None and normalized_unit in {"cyclesperdegree", "cycperdeg"}:
        raise ValueError("dlMTF with an optics struct requires non-angular frequency units.")

    if f_support is None:
        if current_oi is None:
            raise ValueError("dlMTF requires f_support when the first argument is an optics struct.")
        support = np.asarray(oi_get(current_oi, "frequency support", units), dtype=float)
    else:
        support = np.asarray(f_support, dtype=float)
    if support.ndim != 3 or support.shape[2] != 2:
        raise ValueError("dlMTF frequency support must be a rows x cols x 2 grid.")

    if wavelength is None:
        if current_oi is None:
            raise ValueError("dlMTF requires wavelength when the first argument is an optics struct.")
        wave_nm = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
    else:
        wave_nm = np.asarray(wavelength, dtype=float).reshape(-1)
    if wave_nm.size == 0:
        raise ValueError("dlMTF requires at least one wavelength sample.")

    aperture_diameter = _optics_scalar_value(
        current_optics,
        "aperture_diameter_m",
        "apertureDiameter",
        default=_optics_scalar_value(current_optics, "focal_length_m", "focalLength", default=DEFAULT_FOCAL_LENGTH_M)
        / max(_optics_scalar_value(current_optics, "f_number", "fNumber", default=4.0), 1.0e-12),
    )
    focal_plane_distance = _optics_scalar_value(
        current_optics,
        "focal_plane_distance_m",
        "focalPlaneDistance",
        "focal_length_m",
        "focalLength",
        default=DEFAULT_FOCAL_LENGTH_M,
    )
    cutoff = (aperture_diameter / max(focal_plane_distance, 1.0e-12)) / np.maximum(wave_nm * 1.0e-9, 1.0e-12)
    if normalized_unit in {"cyclesperdegree", "cycperdeg"}:
        cutoff = cutoff * float(oi_get(current_oi, "distance per degree", "meters"))
    else:
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1.0e3, "mm": 1.0e3, "microns": 1.0e6, "um": 1.0e6}
        if normalized_unit not in unit_scale:
            raise KeyError(f"Unknown dlMTF units: {units}")
        cutoff = cutoff / unit_scale[normalized_unit]

    rho = np.sqrt(np.asarray(support[:, :, 0], dtype=float) ** 2 + np.asarray(support[:, :, 1], dtype=float) ** 2)
    otf = dl_core(rho, cutoff)
    return np.asarray(otf, dtype=float), support.copy(), np.asarray(cutoff, dtype=float)


def optics_plot_defocus(
    defocus: Any | None = None,
    sample_sf: Any | None = None,
    wave: Any = 550.0,
    optics: OpticalImage | dict[str, Any] | None = None,
) -> np.ndarray:
    """Return the legacy defocus-by-spatial-frequency OTF surface without plotting."""

    defocus_values = np.asarray(np.arange(-1.0, 1.0001, 0.05) if defocus is None else defocus, dtype=float).reshape(-1)
    sample_sf_values = np.asarray(np.arange(0.0, 65.0, 1.0) if sample_sf is None else sample_sf, dtype=float).reshape(-1)
    if defocus_values.size == 0 or sample_sf_values.size == 0:
        raise ValueError("opticsPlotDefocus requires non-empty defocus and sample_sf inputs.")

    current = _normalize_public_optics(_coerce_optics_struct(optics_create() if optics is None else optics))
    focal_length_m = _optics_scalar_value(current, "focal_length_m", "focalLength", default=DEFAULT_FOCAL_LENGTH_M)
    f_number = _optics_scalar_value(current, "f_number", "fNumber", default=4.0)
    if focal_length_m <= 0.0 or f_number <= 0.0:
        raise ValueError("opticsPlotDefocus requires positive focal length and f-number.")

    wave_nm = float(np.asarray(wave, dtype=float).reshape(-1)[0])
    diopters = 1.0 / focal_length_m
    pupil_radius_m = focal_length_m / (2.0 * f_number)
    deg_per_meter = diopters / np.tan(np.deg2rad(1.0))
    reduced_sf = (deg_per_meter * wave_nm * 1.0e-9 / max(diopters * pupil_radius_m, 1.0e-12)) * sample_sf_values

    otf = np.zeros((defocus_values.size, sample_sf_values.size), dtype=float)
    for index, defocus_diopters in enumerate(defocus_values):
        w20 = ((pupil_radius_m**2) / 2.0) * (diopters * defocus_diopters) / max(diopters + defocus_diopters, 1.0e-12)
        alpha = (4.0 * np.pi / max(wave_nm * 1.0e-9, 1.0e-12)) * w20 * np.abs(reduced_sf)
        otf[index, :] = _optics_defocused_mtf(reduced_sf, np.abs(alpha))
    return otf


def optics_plot_off_axis(oi: OpticalImage, this_w: Any | None = None) -> OpticalImage:
    """Return an OI with cached cos4th off-axis falloff payload, without plotting."""

    del this_w
    current = oi.clone()
    stored = current.fields.get("optics", {}).get("cos4th_data")
    if stored is None:
        rows, cols = _oi_shape(current)
        surrogate_scene = Scene(
            name=current.name,
            type="scene",
            fields={
                "distance_m": _oi_depth_distance_m(current) or np.inf,
                "fov_deg": float(oi_get(current, "fov")),
                "vfov_deg": float(oi_get(current, "vfov")),
                "wave": np.asarray(oi_get(current, "wave"), dtype=float).reshape(-1).copy(),
            },
            data={},
        )
        data = _cos4th_factor(rows, cols, current.fields["optics"], surrogate_scene)
        current.fields["optics"]["cos4th_data"] = np.asarray(data, dtype=float).copy()
        current.fields["optics"]["cos4th_value"] = np.asarray(data, dtype=float).copy()
    else:
        current.fields["optics"]["cos4th_data"] = np.asarray(stored, dtype=float).copy()
    return current


def si_convert_rt_data(
    in_name: str | Path | dict[str, Any],
    field_height: float = 0.0,
    out_name: str | Path | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Convert a single-field ray-trace PSF bundle into shift-invariant custom OTF optics."""

    input_path: Path | None = None
    if isinstance(in_name, (str, Path)):
        from .fileio import _deserialize_value, _reconstruct_object

        input_path = Path(in_name).expanduser()
        payload = loadmat(input_path, squeeze_me=True, struct_as_record=False)
        raw = payload.get("optics")
        if raw is None:
            raise ValueError(f"siConvertRTdata expected an optics payload in {input_path}.")
        reconstructed = _reconstruct_object(_deserialize_value(raw))
        if isinstance(reconstructed, OpticalImage):
            current = dict(reconstructed.fields.get("optics", {}))
        elif isinstance(reconstructed, dict):
            current = reconstructed
        else:
            raise ValueError(f"siConvertRTdata could not reconstruct optics from {input_path}.")
    else:
        current = dict(in_name)

    rt_optics = _normalize_public_optics(current)
    rt_wave = np.asarray(optics_get(rt_optics, "rtpsfwavelength"), dtype=float).reshape(-1)
    dx = np.asarray(optics_get(rt_optics, "rtpsfspacing", "m"), dtype=float).reshape(-1)
    if rt_wave.size == 0 or dx.size != 2:
        raise ValueError("siConvertRTdata requires ray-trace PSF wavelengths and 2-D sample spacing.")

    sample_psf = np.asarray(optics_get(rt_optics, "rtpsfdata", float(field_height), float(rt_wave[0])), dtype=float)
    if sample_psf.ndim != 2:
        raise ValueError("siConvertRTdata requires 2-D ray-trace PSF planes.")

    n_samples = int(sample_psf.shape[0])
    nyquist_f = 1.0 / np.maximum(2.0 * dx, 1.0e-12)
    otf = np.zeros((n_samples, n_samples, rt_wave.size), dtype=complex)
    for band_index, wavelength_nm in enumerate(rt_wave):
        psf = np.asarray(optics_get(rt_optics, "rtpsfdata", float(field_height), float(wavelength_nm)), dtype=float)
        psf = psf / max(float(np.sum(psf)), 1.0e-12)
        otf[:, :, band_index] = np.fft.fftshift(np.fft.fft2(psf))

    fx = unit_frequency_list(n_samples) * float(nyquist_f[-1])
    fy = unit_frequency_list(n_samples) * float(nyquist_f[0])

    converted = optics_create()
    converted = optics_set(
        converted,
        "otfstruct",
        {
            "function": "custom",
            "OTF": otf,
            "fx": fx,
            "fy": fy,
            "wave": rt_wave,
        },
    )

    output_path: Path | None = None
    if out_name is not None:
        output_path = Path(out_name).expanduser()
        if output_path.suffix.lower() != ".mat":
            output_path = output_path.with_suffix(".mat")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(output_path, {"optics": _export_optics(converted)}, do_compression=True)

    return converted, (str(input_path) if input_path is not None else None), (str(output_path) if output_path is not None else None)


dlCore = dl_core
dlMTF = dl_mtf
opticsPlotDefocus = optics_plot_defocus
opticsPlotOffAxis = optics_plot_off_axis
siConvertRTdata = si_convert_rt_data


def wvf_pupil_amplitude(
    wvf: dict[str, Any],
    *args: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    return wvf_aperture(wvf, *args)


def si_synthetic(
    psf_type: str = "gaussian",
    oi: OpticalImage | None = None,
    *args: Any,
) -> dict[str, Any]:
    current_oi = oi_create("shift invariant") if oi is None else oi
    wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
    if wave.size == 0:
        wave = DEFAULT_WAVE.copy()

    normalized = param_format(psf_type)
    psf_data: dict[str, Any]
    optics_name = "siSynthetic"

    if normalized == "gaussian":
        if len(args) < 2:
            raise ValueError("siSynthetic('gaussian', ...) requires waveSpread and xyRatio.")
        psf_data = _synthetic_shift_invariant_gaussian_psf_data_from_spreads(wave, args[0], args[1])
        otf_bundle = _shift_invariant_otf_bundle_from_psf_data(psf_data, center_shift="ifft")
    elif normalized == "lorentzian":
        gamma = args[0] if args else 1.0
        psf_data = _synthetic_shift_invariant_lorentzian_psf_data(wave, gamma)
        otf_bundle = _shift_invariant_otf_bundle_from_psf_data(psf_data, center_shift="ifft")
    elif normalized == "pillbox":
        if args:
            patch_size_mm = float(args[0])
        else:
            patch_size_mm = 1.22 * float(oi_get(current_oi, "fnumber")) * (float(np.max(wave)) * 1e-6)
        psf_data = _synthetic_shift_invariant_pillbox_psf_data(wave, patch_size_mm)
        otf_bundle = _shift_invariant_otf_bundle_from_psf_data(psf_data, center_shift="ifft")
    elif normalized == "custom":
        if not args:
            raise ValueError("siSynthetic('custom', ...) requires a PSF struct or .mat file path.")
        source = args[0]
        if isinstance(source, (str, Path)):
            raw = loadmat(Path(source), squeeze_me=True, struct_as_record=False)
            if "psf" not in raw or "wave" not in raw or "umPerSamp" not in raw:
                raise ValueError("Custom shift-invariant PSF files must contain psf, wave, and umPerSamp.")
            psf_data = _normalize_shift_invariant_psf_data(
                {
                    "psf": np.asarray(raw["psf"], dtype=float),
                    "wave": np.asarray(raw["wave"], dtype=float).reshape(-1),
                    "umPerSamp": np.asarray(raw["umPerSamp"], dtype=float).reshape(-1),
                }
            )
            optics_name = str(Path(source).stem)
        else:
            psf_data = _normalize_shift_invariant_psf_data(source)
        otf_bundle = _custom_shift_invariant_otf_bundle(psf_data)
    else:
        raise UnsupportedOptionError("siSynthetic", psf_type)

    return {
        "name": optics_name,
        "model": "shiftinvariant",
        "f_number": float(oi_get(current_oi, "fnumber")),
        "focal_length_m": float(oi_get(current_oi, "focal length")),
        "compute_method": "opticsotf",
        "aberration_scale": float(current_oi.fields.get("optics", {}).get("aberration_scale", 0.0)),
        "offaxis_method": str(current_oi.fields.get("optics", {}).get("offaxis_method", "cos4th")),
        "psf_data": psf_data,
        "otf_function": "custom",
        **otf_bundle,
    }


def _rebuild_oi_from_wvf(oi: OpticalImage, wvf: dict[str, Any]) -> OpticalImage:
    diffuser_method = oi.fields.get("diffuser_method", "skip")
    diffuser_blur_m = float(oi.fields.get("diffuser_blur_m", 0.0))
    compute_method = str(oi.fields.get("compute_method", oi.fields.get("optics", {}).get("compute_method", "opticspsf")))
    optics_model = str(oi.fields.get("optics", {}).get("model", "shiftinvariant"))
    metadata = dict(oi.metadata)
    depth_map = oi.fields.get("depth_map_m")
    wangular = float(oi.fields.get("fov_deg", 10.0))

    rebuilt = wvf_to_oi(wvf)
    rebuilt.fields["diffuser_method"] = diffuser_method
    rebuilt.fields["diffuser_blur_m"] = diffuser_blur_m
    rebuilt.fields["compute_method"] = compute_method
    rebuilt.fields["optics"]["compute_method"] = compute_method
    rebuilt.fields["optics"]["model"] = optics_model
    rebuilt.metadata = metadata
    rebuilt = oi_set(rebuilt, "wangular", wangular)
    if depth_map is not None:
        rebuilt.fields["depth_map_m"] = np.asarray(depth_map, dtype=float).copy()
    rebuilt.data.clear()
    return rebuilt


def oi_create(
    oi_type: str = "diffraction limited",
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> OpticalImage | list[str]:
    """Create a supported optical image object."""

    if _is_empty_dispatch_placeholder(oi_type):
        oi_type = "diffraction limited"
    normalized = param_format(oi_type)
    store = _store(asset_store)
    valid_types = [
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
    if normalized.startswith("valid"):
        return valid_types.copy()
    oi = OpticalImage(name="opticalimage")
    optics: dict[str, Any]
    psf_data: dict[str, Any] | None = None

    if normalized in {"default", "diffractionlimited", "diffraction"}:
        optics = {
            "name": "standard (1/4-inch)",
            "model": "diffractionlimited",
            "f_number": 4.0,
            "focal_length_m": DEFAULT_FOCAL_LENGTH_M,
            "compute_method": "opticsotf",
            "aberration_scale": 0.0,
            "offaxis_method": "cos4th",
        }
    elif normalized in {"wvf", "shiftinvariant"}:
        wavefront = (
            args[0]
            if args and isinstance(args[0], dict)
            else wvf_create(
                wave=DEFAULT_WAVE.copy(),
                focal_length_m=DEFAULT_FOCAL_LENGTH_M,
                f_number=4.0,
                calc_pupil_diameter_mm=DEFAULT_CAMERA_WVF_CALC_PUPIL_DIAMETER_MM,
            )
        )
        optics = {
            "name": "wvf",
            "model": "shiftinvariant",
            "f_number": float(wavefront.get("f_number", 4.0)),
            "focal_length_m": float(wavefront.get("focal_length_m", DEFAULT_FOCAL_LENGTH_M)),
            "compute_method": "opticspsf",
            "aberration_scale": float(wavefront.get("aberration_scale", 0.0)),
            "wavefront": wavefront,
            "offaxis_method": "cos4th",
        }
    elif normalized == "psf":
        if args and isinstance(args[0], dict):
            psf_data = _normalize_shift_invariant_psf_data(args[0])
            otf_bundle = _custom_shift_invariant_otf_bundle(psf_data)
        else:
            default_wave = DEFAULT_WAVE.copy()
            psf_data = _synthetic_shift_invariant_gaussian_psf_data(default_wave, 4.0)
            otf_bundle = _synthetic_shift_invariant_gaussian_otf_bundle(default_wave, 4.0)
        optics = {
            "model": "shiftinvariant",
            "name": "siSynthetic",
            "f_number": 4.0,
            "focal_length_m": DEFAULT_FOCAL_LENGTH_M,
            "compute_method": "opticsotf",
            "aberration_scale": 0.0,
            "offaxis_method": "cos4th",
            "psf_data": psf_data,
            **otf_bundle,
        }
    elif normalized == "raytrace":
        source = args[0] if args else None
        optics = _load_raytrace_optics(source, asset_store=_store(asset_store))
    elif normalized == "pinhole":
        optics = {
            "name": "pinhole",
            "model": "skip",
            "f_number": 1e-3,
            "focal_length_m": 1e-2,
            "compute_method": "skip",
            "aberration_scale": 0.0,
            "offaxis_method": "skip",
        }
    elif normalized == "empty":
        optics = optics_create("empty", asset_store=store)
    elif normalized == "humanmw":
        pupil_radius_m = 0.0015 if len(args) == 0 or _is_empty_dispatch_placeholder(args[0]) else float(np.asarray(args[0], dtype=float).reshape(-1)[0])
        focal_length_m = DEFAULT_WVF_FOCAL_LENGTH_M if len(args) < 2 or _is_empty_dispatch_placeholder(args[1]) else float(np.asarray(args[1], dtype=float).reshape(-1)[0])
        wave = DEFAULT_WAVE.copy()
        otf, support, otf_wave = human_otf(pupil_radius_m, 1.0 / max(focal_length_m, 1.0e-12), None, wave)
        human_mw = oi_create("shift invariant", asset_store=store)
        human_mw = oi_set(
            human_mw,
            "otfstruct",
            {
                "OTF": np.asarray(otf, dtype=complex),
                "fx": np.asarray(support[0, :, 0], dtype=float).reshape(-1),
                "fy": np.asarray(support[:, 0, 1], dtype=float).reshape(-1),
                "wave": np.asarray(otf_wave, dtype=float).reshape(-1),
                "function": "humanOTF",
            },
        )
        human_mw = oi_set(human_mw, "compute method", "humanmw")
        human_mw.name = "human-MW"
        human_mw.fields["optics"]["name"] = "human-MW"
        human_mw.fields["optics"]["focal_length_m"] = float(focal_length_m)
        human_mw.fields["optics"]["f_number"] = float(focal_length_m / max(2.0 * pupil_radius_m, 1.0e-12))
        human_mw.fields["optics"]["otf_method"] = "human"
        human_mw.fields["optics"]["lens"] = _human_lens_payload(
            np.asarray(human_mw.fields.get("wave", wave), dtype=float),
            asset_store=store,
        )
        human_mw.fields["optics"].pop("transmittance", None)
        _sync_oi_identity_fields(human_mw)
        return track_session_object(session, human_mw)
    elif normalized in {"human", "wvfhuman", "humanwvf"}:
        pupil_diameter_mm = 3.0 if len(args) == 0 or _is_empty_dispatch_placeholder(args[0]) else float(np.asarray(args[0], dtype=float).reshape(-1)[0])
        if len(args) > 1 and not _is_empty_dispatch_placeholder(args[1]):
            zcoeffs = np.asarray(args[1], dtype=float).reshape(-1)
            measured_pupil_mm = pupil_diameter_mm
        else:
            supported_pupils = np.array([3.0, 4.5, 6.0, 7.5], dtype=float)
            larger_or_equal = supported_pupils[supported_pupils >= pupil_diameter_mm]
            measured_pupil_mm = float(larger_or_equal[0] if larger_or_equal.size > 0 else supported_pupils[-1])
            zcoeffs = np.asarray(wvf_load_thibos_virtual_eyes(measured_pupil_mm, asset_store=store), dtype=float).reshape(-1)
        wave = DEFAULT_WAVE.copy() if len(args) < 3 or _is_empty_dispatch_placeholder(args[2]) else np.asarray(args[2], dtype=float).reshape(-1)
        focal_length_m = DEFAULT_WVF_FOCAL_LENGTH_M
        if len(args) > 3 and not _is_empty_dispatch_placeholder(args[3]):
            um_per_degree = float(np.asarray(args[3], dtype=float).reshape(-1)[0])
            focal_length_m = um_per_degree * 1.0e-6 / (2.0 * np.tan(np.deg2rad(0.5)))
        lca_method = "human" if len(args) < 5 or _is_empty_dispatch_placeholder(args[4]) else args[4]
        wavefront = wvf_create(
            wave=wave,
            focal_length_m=focal_length_m,
            calc_pupil_diameter_mm=pupil_diameter_mm,
            measured_pupil=measured_pupil_mm,
            zcoeffs=zcoeffs,
            lcamethod=lca_method,
        )
        human_wvf = wvf_to_oi(wavefront)
        human_wvf = oi_set(human_wvf, "compute method", "opticspsf")
        human_wvf.name = "human-WVF"
        human_wvf.fields["optics"]["name"] = "humanwvf"
        human_wvf.fields["optics"]["otf_method"] = "human"
        human_wvf.fields["optics"]["lens"] = _human_lens_payload(
            np.asarray(human_wvf.fields.get("wave", wave), dtype=float),
            asset_store=store,
        )
        human_wvf.fields["optics"].pop("transmittance", None)
        _sync_oi_identity_fields(human_wvf)
        return track_session_object(session, human_wvf)
    elif normalized in {"uniformd65", "uniformee", "uniformeespecify"}:
        from .scene import scene_create, scene_set

        size = 32 if len(args) == 0 or _is_empty_dispatch_placeholder(args[0]) else args[0]
        wave = DEFAULT_WAVE.copy() if len(args) < 2 or _is_empty_dispatch_placeholder(args[1]) else np.asarray(args[1], dtype=float)
        scene_name = "uniform d65" if normalized == "uniformd65" else "uniform ee"
        source_scene = scene_create(scene_name, size, wave, asset_store=store)
        source_scene = scene_set(source_scene, "hfov", 120.0)

        base = oi_create("default", asset_store=store, session=session)
        base = oi_set(base, "optics fnumber", 1e-3)
        base = oi_set(base, "optics offaxis method", "skip")
        computed = oi_compute(base, source_scene, session=session)
        computed = oi_set(computed, "compute method", "")
        _sync_oi_identity_fields(computed)
        return computed
    elif normalized == "black":
        size_value = 32 if len(args) == 0 or _is_empty_dispatch_placeholder(args[0]) else int(np.asarray(args[0], dtype=int).reshape(-1)[0])
        wave = DEFAULT_WAVE.copy() if len(args) < 2 or _is_empty_dispatch_placeholder(args[1]) else np.asarray(args[1], dtype=float)
        black_oi = oi_create("shift invariant", asset_store=store, session=session)
        black_oi = oi_set(black_oi, "wave", wave)
        black_oi = oi_set(black_oi, "photons", np.zeros((size_value, size_value, int(np.asarray(wave, dtype=float).size)), dtype=float))
        black_oi = oi_set(black_oi, "fov", 100.0)
        _sync_oi_identity_fields(black_oi)
        return black_oi
    else:
        raise UnsupportedOptionError("oiCreate", oi_type)

    wave_index = 1 if normalized in {"wvf", "shiftinvariant", "raytrace"} else 0
    if normalized == "psf":
        wave = np.asarray(psf_data["wave"], dtype=float).copy() if psf_data is not None else DEFAULT_WAVE.copy()
    elif normalized == "empty":
        wave = (
            np.empty(0, dtype=float)
            if len(args) == 0 or _is_empty_dispatch_placeholder(args[0])
            else np.asarray(args[0], dtype=float).reshape(-1)
        )
    elif normalized in {"default", "diffractionlimited", "diffraction", "pinhole"}:
        wave = (
            DEFAULT_WAVE.copy()
            if len(args) <= wave_index or _is_empty_dispatch_placeholder(args[wave_index])
            else np.asarray(args[wave_index], dtype=float)
        )
    elif normalized in {"wvf", "shiftinvariant"} and args and isinstance(args[0], dict) and len(args) <= wave_index:
        wave = np.asarray(optics.get("wavefront", {}).get("wave", DEFAULT_WAVE.copy()), dtype=float)
    else:
        wave = np.asarray(args[wave_index], dtype=float) if len(args) > wave_index else DEFAULT_WAVE.copy()
    if normalized != "empty":
        optics.setdefault(
            "transmittance",
            {
                "wave": np.asarray(wave, dtype=float).reshape(-1).copy(),
                "scale": np.ones(np.asarray(wave, dtype=float).size, dtype=float),
            },
        )
    oi.fields["optics"] = optics
    oi.fields["wave"] = wave
    oi.fields["compute_method"] = optics.get("compute_method", "")
    oi.fields["diffuser_method"] = "skip"
    oi.fields["diffuser_blur_m"] = 2e-6
    if normalized == "pinhole":
        oi.name = "pinhole"
    oi.fields["psf_angle_step_deg"] = DEFAULT_RAYTRACE_ANGLE_STEP_DEG
    oi.fields["psf_sample_angles_deg"] = None
    oi.fields["psf_image_heights_m"] = None
    oi.fields["psf_wavelength_nm"] = None
    oi.fields["psf_optics_name"] = None
    oi.fields["psf_struct"] = None
    oi.fields["sample_spacing_m"] = None
    oi.data["photons"] = np.empty((0, 0, 0), dtype=float)
    _sync_oi_identity_fields(oi)
    return track_session_object(session, oi)


def _scene_sample_spacing(scene: Scene) -> float:
    width = float(scene.fields["width_m"])
    cols = int(scene.fields["cols"])
    return width / max(cols, 1)


def _image_distance_m(optics: dict[str, Any], scene: Scene) -> float:
    focal_length = float(optics["focal_length_m"])
    model = param_format(optics.get("model", ""))
    if model == "skip":
        return focal_length
    if model == "raytrace":
        raytrace = optics.get("raytrace", {})
        return float(raytrace.get("effective_focal_length_m", focal_length))
    scene_distance = float(scene.fields.get("distance_m", np.inf))
    if not np.isfinite(scene_distance) or scene_distance <= focal_length:
        return focal_length
    return 1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12)


def _magnification(optics: dict[str, Any], scene: Scene) -> float:
    model = param_format(optics.get("model", ""))
    if model == "skip":
        return -1.0
    if model == "raytrace":
        return float(optics.get("raytrace", {}).get("magnification", 0.0))
    scene_distance = float(scene.fields.get("distance_m", np.inf))
    if not np.isfinite(scene_distance) or scene_distance <= 0.0:
        return 0.0
    return -_image_distance_m(optics, scene) / scene_distance


def _oi_geometry(optics: dict[str, Any], scene: Scene) -> tuple[float, float, float]:
    image_distance = _image_distance_m(optics, scene)
    hfov_deg = float(scene.fields.get("fov_deg", 10.0))
    vfov_deg = float(scene.fields.get("vfov_deg", hfov_deg))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(hfov_deg) / 2.0)
    height_m = 2.0 * image_distance * np.tan(np.deg2rad(vfov_deg) / 2.0)
    return image_distance, width_m, height_m


def _radiance_to_irradiance(scene_cube: np.ndarray, optics: dict[str, Any], scene: Scene) -> np.ndarray:
    wave = np.asarray(scene.fields["wave"], dtype=float)
    transmittance = _optics_transmittance_scale(optics, wave)
    raytrace = optics.get("raytrace", {})
    if param_format(optics.get("model", "")) == "raytrace":
        f_number = float(raytrace.get("effective_f_number", optics["f_number"]))
        magnification = float(raytrace.get("magnification", 0.0))
    else:
        f_number = float(optics["f_number"])
        magnification = _magnification(optics, scene)
    scale = np.pi / (1.0 + 4.0 * (f_number**2) * ((1.0 + abs(magnification)) ** 2))
    irradiance = np.asarray(scene_cube, dtype=float)
    if np.any(transmittance != 1.0):
        irradiance = irradiance * transmittance.reshape(1, 1, -1)
    return irradiance * scale


def _cos4th_factor(rows: int, cols: int, optics: dict[str, Any], scene: Scene) -> np.ndarray:
    _, width_m, height_m = _oi_geometry(optics, scene)
    # ISETCam's cos4th.m uses the OI spatial support that reflects the
    # scene-focused image size, but then calls opticsGet('imageDistance')
    # without a scene-distance argument. That getter falls back to the focal
    # length, so we mirror that behavior here.
    image_distance = float(optics["focal_length_m"])
    x = np.linspace(-width_m / 2.0 + width_m / (2.0 * cols), width_m / 2.0 - width_m / (2.0 * cols), cols)
    y = np.linspace(-height_m / 2.0 + height_m / (2.0 * rows), height_m / 2.0 - height_m / (2.0 * rows), rows)
    xx, yy = np.meshgrid(x, y)
    s_factor = np.sqrt(image_distance**2 + xx**2 + yy**2)
    image_diagonal = np.sqrt(width_m**2 + height_m**2)
    if image_distance > 10.0 * image_diagonal:
        return (image_distance / np.maximum(s_factor, 1e-12)) ** 4

    magnification = _magnification(optics, scene)
    cos_phi = image_distance / np.maximum(s_factor, 1e-12)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, None))
    tan_phi = sin_phi / np.maximum(cos_phi, 1e-12)

    f_number = float(optics["f_number"])
    sin_theta = 1.0 / (1.0 + 4.0 * (f_number * (1.0 - magnification)) ** 2)
    cos_theta = np.sqrt(max(1e-12, 1.0 - sin_theta**2))
    tan_theta = sin_theta / cos_theta
    numerator = 1.0 - tan_theta**2 + tan_phi**2
    denominator = np.sqrt(tan_phi**4 + 2.0 * tan_phi**2 * (1.0 - tan_theta**2) + 1.0 / (cos_theta**4))
    spatial_fall = (np.pi / 2.0) * (1.0 - numerator / np.maximum(denominator, 1e-12))
    return spatial_fall / max(np.pi * (sin_theta**2), 1e-12)


def _pad_scene(
    scene_cube: np.ndarray,
    pad_pixels: tuple[int, int],
    pad_value: str,
) -> tuple[np.ndarray, str, float]:
    pad_rows, pad_cols = int(pad_pixels[0]), int(pad_pixels[1])
    if pad_rows <= 0 and pad_cols <= 0:
        return scene_cube, "nearest", 0.0
    mode = param_format(pad_value)
    if mode == "zero":
        return np.pad(scene_cube, ((pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0))), "constant", 0.0
    if mode == "border":
        padded = np.empty(
            (scene_cube.shape[0] + 2 * pad_rows, scene_cube.shape[1] + 2 * pad_cols, scene_cube.shape[2]),
            dtype=float,
        )
        band_corner = scene_cube[0, 0, :]
        row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
        col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
        for band_index, corner_value in enumerate(band_corner):
            padded[:, :, band_index] = corner_value
            padded[row_slice, col_slice, band_index] = scene_cube[:, :, band_index]
        return padded, "constant", float(np.mean(band_corner))
    if mode == "mean":
        padded = np.empty(
            (scene_cube.shape[0] + 2 * pad_rows, scene_cube.shape[1] + 2 * pad_cols, scene_cube.shape[2]),
            dtype=float,
        )
        band_means = scene_cube.mean(axis=(0, 1))
        row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
        col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
        for band_index, mean_value in enumerate(band_means):
            padded[:, :, band_index] = mean_value
            padded[row_slice, col_slice, band_index] = scene_cube[:, :, band_index]
        return padded, "constant", float(np.mean(band_means))
    raise UnsupportedOptionError("oiCompute", pad_value)


def _pad_depth_map(scene: Scene, pad_pixels: tuple[int, int]) -> np.ndarray:
    depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
    pad_rows, pad_cols = int(pad_pixels[0]), int(pad_pixels[1])
    if pad_rows <= 0 and pad_cols <= 0:
        return depth_map.copy()
    return np.pad(depth_map, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="constant", constant_values=0.0)


def _diffraction_otf(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
    scene: Scene,
) -> np.ndarray:
    rows, cols = shape
    image_distance = _image_distance_m(optics, scene)
    width_m = float(cols) * float(sample_spacing_m)
    height_m = float(rows) * float(sample_spacing_m)
    fov_width = float(np.rad2deg(2.0 * np.arctan2(width_m / 2.0, image_distance)))
    fov_height = float(np.rad2deg(2.0 * np.arctan2(height_m / 2.0, image_distance)))
    distance_per_degree = width_m / max(fov_width, 1e-12)
    deg_per_dist = 1.0 / max(distance_per_degree, 1e-12)
    fx = unit_frequency_list(cols) * ((cols / 2.0) / max(fov_width, 1e-12) * deg_per_dist)
    fy = unit_frequency_list(rows) * ((rows / 2.0) / max(fov_height, 1e-12) * deg_per_dist)
    rho = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    aperture_diameter = float(optics["focal_length_m"]) / max(float(optics["f_number"]), 1e-12)
    # ISETCam's dlMTF() uses opticsGet(optics, 'focalPlaneDistance') without
    # passing the scene distance, which resolves to the focal length rather
    # than the thin-lens image distance for finite scene depth.
    focal_plane_distance = float(optics["focal_length_m"])
    wavelengths_m = np.asarray(wave, dtype=float) * 1e-9
    cutoff = (aperture_diameter / max(focal_plane_distance, 1e-12)) / np.maximum(wavelengths_m, 1e-12)

    otf = np.zeros((rows, cols, wavelengths_m.size), dtype=float)
    for index, cutoff_frequency in enumerate(cutoff):
        normalized = rho / max(float(cutoff_frequency), 1e-12)
        clipped = np.clip(normalized, 0.0, 1.0)
        current = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
        current[normalized >= 1.0] = 0.0
        otf[:, :, index] = np.fft.ifftshift(current)
    return otf


def _apply_otf(cube: np.ndarray, otf: np.ndarray) -> np.ndarray:
    result = np.empty_like(cube, dtype=float)
    for band_index in range(cube.shape[2]):
        filtered = np.fft.ifft2(np.fft.fft2(cube[:, :, band_index]) * otf[:, :, band_index])
        result[:, :, band_index] = np.abs(filtered)
    return result


def _resize_image(image: np.ndarray, output_shape: tuple[int, int], *, method: str) -> np.ndarray:
    out_rows = max(int(round(output_shape[0])), 1)
    out_cols = max(int(round(output_shape[1])), 1)
    image = np.asarray(image, dtype=float)
    in_rows, in_cols = image.shape
    if (in_rows, in_cols) == (out_rows, out_cols):
        return image.copy()

    row_positions = np.linspace(0.0, max(in_rows - 1, 0), out_rows)
    col_positions = np.linspace(0.0, max(in_cols - 1, 0), out_cols)
    row_grid, col_grid = np.meshgrid(row_positions, col_positions, indexing="ij")
    if method == "nearest":
        row_index = np.clip(np.rint(row_grid).astype(int), 0, max(in_rows - 1, 0))
        col_index = np.clip(np.rint(col_grid).astype(int), 0, max(in_cols - 1, 0))
        return image[row_index, col_index]
    if method == "linear":
        return map_coordinates(image, [row_grid, col_grid], order=1, mode="nearest", prefilter=False)
    raise ValueError(f"Unsupported resize method: {method}")


def _resize_nearest(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    return _resize_image(image, output_shape, method="nearest")


def _support_resample_positions(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("Resample step must be positive.")
    if stop <= start:
        return np.array([float(start)], dtype=float)
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return float(start) + float(step) * np.arange(max(count, 1), dtype=float)


def _resample_plane_on_support(
    plane: np.ndarray,
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    if x_support.size <= 1:
        col_coords = np.zeros_like(x_query, dtype=float)
    else:
        col_coords = (np.asarray(x_query, dtype=float) - float(x_support[0])) / float(x_support[1] - x_support[0])
    if y_support.size <= 1:
        row_coords = np.zeros_like(y_query, dtype=float)
    else:
        row_coords = (np.asarray(y_query, dtype=float) - float(y_support[0])) / float(y_support[1] - y_support[0])
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    if method == "linear":
        order = 1
    elif method == "nearest":
        order = 0
    else:
        raise ValueError(f"Unsupported resample method: {method}")
    return map_coordinates(plane, [row_grid, col_grid], order=order, mode="nearest", prefilter=False)


def _oi_spatial_resample(oi: OpticalImage, sample_spacing_m: float, *, method: str = "linear") -> OpticalImage:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    if photons.size == 0:
        return oi.clone()

    support = oi_get(oi, "spatial support linear", "m")
    x_support = np.asarray(support["x"], dtype=float)
    y_support = np.asarray(support["y"], dtype=float)
    x_query = _support_resample_positions(float(x_support[0]), float(x_support[-1]), float(sample_spacing_m))
    y_query = _support_resample_positions(float(y_support[0]), float(y_support[-1]), float(sample_spacing_m))

    resampled_cube = np.empty((y_query.size, x_query.size, photons.shape[2]), dtype=float)
    for band_index in range(photons.shape[2]):
        resampled_cube[:, :, band_index] = _resample_plane_on_support(
            photons[:, :, band_index],
            x_support,
            y_support,
            x_query,
            y_query,
            method=method,
        )

    resampled = oi.clone()
    original_fov = float(oi_get(oi, "fov"))
    resampled.data["photons"] = resampled_cube
    depth_map = oi.fields.get("depth_map_m")
    if depth_map is not None:
        resampled.fields["depth_map_m"] = _resample_plane_on_support(
            np.asarray(depth_map, dtype=float),
            x_support,
            y_support,
            x_query,
            y_query,
            method="nearest",
        )
    _sync_oi_geometry_fields(resampled)
    current_spacing = float(oi_get(resampled, "wspatialresolution"))
    if current_spacing > 0.0:
        resampled = oi_set(resampled, "fov", original_fov * float(sample_spacing_m) / current_spacing)
    return resampled


def _image_bounding_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or cols.size == 0:
        return (0.0, 0.0, 1.0, 1.0)

    min_row = float(rows.min() + 1)
    max_row = float(rows.max() + 1)
    min_col = float(cols.min() + 1)
    max_col = float(cols.max() + 1)
    center_x = (min_col + max_col) / 2.0
    center_y = (min_row + max_row) / 2.0
    max_diff = max(abs(max_row - center_y), abs(max_col - center_x))
    return (center_x - max_diff, center_y - max_diff, 2.0 * max_diff, 2.0 * max_diff)


def _ensure_optics_transmittance(optics: dict[str, Any], wave: np.ndarray | None = None) -> dict[str, np.ndarray]:
    current = optics.get("transmittance")
    if isinstance(current, dict) and "wave" in current and "scale" in current:
        wave_values = np.asarray(current["wave"], dtype=float).reshape(-1)
        scale_values = np.asarray(current["scale"], dtype=float).reshape(-1)
        if wave_values.size == scale_values.size and wave_values.size > 0:
            current["wave"] = wave_values
            current["scale"] = scale_values
            return current

    base_wave = np.asarray(DEFAULT_WAVE if wave is None else wave, dtype=float).reshape(-1)
    optics["transmittance"] = {
        "wave": base_wave.copy(),
        "scale": np.ones(base_wave.size, dtype=float),
    }
    return optics["transmittance"]


def _optics_transmittance_scale(optics: dict[str, Any], wave: np.ndarray) -> np.ndarray:
    target_wave = np.asarray(wave, dtype=float).reshape(-1)
    transmittance = _optics_transmittance_payload(optics, target_wave)
    source_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
    source_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
    if source_wave.size == 0 or source_scale.size == 0:
        return np.ones(target_wave.size, dtype=float)
    if source_wave.size == target_wave.size and np.array_equal(source_wave, target_wave):
        return source_scale.copy()
    fill_value = 1.0 if "transmittance" in optics else float(source_scale[-1])
    return np.interp(target_wave, source_wave, source_scale, left=fill_value, right=fill_value)


def _normalize_sce_params(wave: np.ndarray, sce_params: dict[str, Any] | None = None) -> dict[str, Any]:
    target_wave = np.asarray(wave, dtype=float).reshape(-1)
    current = {} if sce_params is None else dict(sce_params)
    wave_nm = np.asarray(current.get("wave", current.get("wavelengths", target_wave)), dtype=float).reshape(-1)
    rho = np.asarray(current.get("rho", np.zeros(wave_nm.size, dtype=float)), dtype=float).reshape(-1)
    if rho.size == 1 and wave_nm.size > 1:
        rho = np.full(wave_nm.size, float(rho[0]), dtype=float)
    if rho.size != wave_nm.size:
        raise ValueError("SCE rho must match the SCE wavelength sampling.")
    return {
        "wave": wave_nm.copy(),
        "rho": rho.copy(),
        "xo_mm": float(current.get("xo_mm", current.get("xo", 0.0))),
        "yo_mm": float(current.get("yo_mm", current.get("yo", 0.0))),
    }


def _sce_sampling_to_wavelengths(start_nm: float, step_nm: float, count: int) -> np.ndarray:
    last_nm = float(start_nm) + (int(count) - 1) * float(step_nm)
    return np.arange(float(start_nm), last_nm + (float(step_nm) * 0.5), float(step_nm), dtype=float)


def _wvf_data_root() -> Path:
    return ensure_upstream_snapshot() / "opticalimage" / "wavefront" / "data"


def _load_sce_berendschot_model() -> tuple[np.ndarray, np.ndarray]:
    data_path = _wvf_data_root() / "BerendschotEtAl2001_Figure2BoldSmooth.txt"
    raw_lines = data_path.read_text(encoding="latin1").replace("\r", "\n").splitlines()
    rows: list[tuple[float, float]] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("wavelength"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError(f"Unable to parse SCE model data from {data_path}")

    raw = np.asarray(rows, dtype=float)
    init_wave = _sce_sampling_to_wavelengths(400.0, 5.0, 71)
    rho = np.interp(init_wave, raw[:, 0], raw[:, 1], left=raw[0, 1], right=raw[-1, 1])
    reference_index = int(np.argmin(np.abs(init_wave - 550.0)))
    rho = rho - rho[reference_index] + 0.041
    return init_wave, rho


def _sce_export(wave_nm: np.ndarray, rho: np.ndarray, *, xo_mm: float, yo_mm: float) -> dict[str, Any]:
    wave_values = np.asarray(wave_nm, dtype=float).reshape(-1).copy()
    rho_values = np.asarray(rho, dtype=float).reshape(-1).copy()
    return {
        "wave": wave_values,
        "wavelengths": wave_values.copy(),
        "rho": rho_values,
        "xo": float(xo_mm),
        "yo": float(yo_mm),
        "xo_mm": float(xo_mm),
        "yo_mm": float(yo_mm),
    }


def sce_create(
    wave: Any | None = None,
    rho_source: str | None = None,
    position_source: str | None = None,
) -> dict[str, Any]:
    wave_nm = np.asarray(_sce_sampling_to_wavelengths(400.0, 10.0, 31) if wave is None else wave, dtype=float).reshape(-1)
    rho_key = str(param_format("none" if rho_source is None else rho_source)).replace("_", "")
    position_key = str(param_format("centered" if position_source is None else position_source)).replace("_", "")

    if rho_key == "berendschotdata":
        init_wave = _sce_sampling_to_wavelengths(400.0, 10.0, 31)
        rho0 = np.asarray(
            [
                0.0565,
                0.0585,
                0.0605,
                0.0600,
                0.05875,
                0.05775,
                0.0565,
                0.0545,
                0.0525,
                0.0510,
                0.04925,
                0.04675,
                0.0440,
                0.0410,
                0.0400,
                0.0410,
                0.0410,
                0.0415,
                0.0425,
                0.04275,
                0.04325,
                0.0450,
                0.0470,
                0.0480,
                0.0485,
                0.0490,
                0.04975,
                0.0500,
                0.04975,
                0.04925,
                0.0490,
            ],
            dtype=float,
        )
    elif rho_key == "berendschotmodel":
        init_wave, rho0 = _load_sce_berendschot_model()
    elif rho_key == "none":
        init_wave = _sce_sampling_to_wavelengths(400.0, 10.0, 31)
        rho0 = np.zeros(init_wave.size, dtype=float)
    else:
        raise ValueError(f"Unsupported SCE rho source: {rho_source}")

    if position_key == "centered":
        xo_mm = 0.0
        yo_mm = 0.0
    elif position_key == "applegate":
        xo_mm = 0.51
        yo_mm = 0.20
    else:
        raise ValueError(f"Unsupported SCE position source: {position_source}")

    rho = np.interp(wave_nm, init_wave, rho0, left=rho0[0], right=rho0[-1])
    return _sce_export(wave_nm, rho, xo_mm=xo_mm, yo_mm=yo_mm)


def sce_get(sce_params: dict[str, Any], parameter: str, *args: Any) -> Any:
    wave_nm = np.asarray(sce_params.get("wave", sce_params.get("wavelengths", np.empty(0, dtype=float))), dtype=float).reshape(-1)
    current = _normalize_sce_params(wave_nm, sce_params)
    key = param_format(parameter)

    if key == "xo":
        return float(current.get("xo_mm", 0.0))
    if key == "yo":
        return float(current.get("yo_mm", 0.0))
    if key in {"wave", "wavelengths"}:
        return np.asarray(current.get("wave", np.empty(0, dtype=float)), dtype=float).copy()
    if key == "rho":
        rho = np.asarray(current.get("rho", np.empty(0, dtype=float)), dtype=float).reshape(-1)
        if not args:
            return rho.copy()
        requested_wave = np.asarray(args[0], dtype=float).reshape(-1)
        values = np.interp(requested_wave, wave_nm, rho, left=rho[0], right=rho[-1])
        if values.size == 1:
            return float(values[0])
        return values
    raise KeyError(f"Unknown SCE parameter: {parameter}")


def _sce_rho_for_wave(sce_params: dict[str, Any], wavelength_nm: float) -> float:
    wave_nm = np.asarray(sce_params.get("wave", np.array([], dtype=float)), dtype=float).reshape(-1)
    rho = np.asarray(sce_params.get("rho", np.array([], dtype=float)), dtype=float).reshape(-1)
    if wave_nm.size == 0 or rho.size == 0:
        return 0.0
    if wave_nm.size == 1:
        return float(rho[0])
    return float(np.interp(float(wavelength_nm), wave_nm, rho, left=float(rho[0]), right=float(rho[-1])))


def _osa_index_to_nm(index: int) -> tuple[int, int]:
    normalized_index = int(index)
    if normalized_index < 0:
        raise ValueError("OSA index must be non-negative.")
    order = 0
    offset = 0
    while True:
        ms = list(range(-order, order + 1, 2))
        next_offset = offset + len(ms)
        if normalized_index < next_offset:
            return order, ms[normalized_index - offset]
        offset = next_offset
        order += 1


def _zernike_radial(n: int, m_abs: int, radius: np.ndarray) -> np.ndarray:
    radial = np.zeros_like(radius, dtype=float)
    half_sum = (n + m_abs) // 2
    half_diff = (n - m_abs) // 2
    for s in range(half_diff + 1):
        coefficient = (
            ((-1) ** s)
            * factorial(n - s)
            / (
                factorial(s)
                * factorial(half_sum - s)
                * factorial(half_diff - s)
            )
        )
        radial = radial + (coefficient * np.power(radius, n - (2 * s)))
    return radial


def _zernike_surface_osa(zcoeffs: np.ndarray, norm_radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(zcoeffs, dtype=float).reshape(-1)
    if coefficients.size == 0 or np.allclose(coefficients, 0.0):
        return np.zeros_like(norm_radius, dtype=float)

    valid = norm_radius <= 1.0
    surface = np.zeros_like(norm_radius, dtype=float)
    radius_valid = norm_radius[valid]
    theta_valid = theta[valid]

    for osa_index, coefficient in enumerate(coefficients):
        if np.isclose(coefficient, 0.0):
            continue
        n, m = _osa_index_to_nm(osa_index)
        radial = _zernike_radial(n, abs(m), radius_valid)
        if m == 0:
            angular = np.ones_like(theta_valid, dtype=float)
            normalization = np.sqrt(n + 1.0)
        elif m > 0:
            angular = np.cos(m * theta_valid)
            normalization = np.sqrt(2.0 * (n + 1.0))
        else:
            angular = np.sin(abs(m) * theta_valid)
            normalization = np.sqrt(2.0 * (n + 1.0))
        term = normalization * radial * angular
        surface[valid] = surface[valid] + (float(coefficient) * term)

    return surface


def _wvf_aperture_mask(
    n_pixels: int,
    calc_radius_index: np.ndarray,
    aperture: np.ndarray | None = None,
) -> np.ndarray:
    if aperture is None:
        return calc_radius_index.astype(float)

    current = np.asarray(aperture, dtype=float)
    if current.shape != (n_pixels, n_pixels):
        current = _resize_image(current, (n_pixels, n_pixels), method="linear")

    bounding_box = _image_bounding_box(calc_radius_index)
    target_shape = (
        max(int(round(bounding_box[3])), 1),
        max(int(round(bounding_box[2])), 1),
    )
    current = _resize_nearest(current, target_shape)

    pad = int(round((n_pixels - bounding_box[2]) / 2.0) - 2.0)
    if pad > 0:
        current = np.pad(current, ((pad, pad), (pad, pad)), mode="constant")

    current = _resize_nearest(current, (n_pixels, n_pixels))
    current = np.clip(current, 0.0, 1.0)
    current[~calc_radius_index] = 0.0
    return current


def _wvf_aperture_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, value in DEFAULT_WVF_APERTURE_PARAMS.items():
        if isinstance(value, np.ndarray):
            defaults[key] = np.asarray(value, dtype=float).copy()
        else:
            defaults[key] = value
    return defaults


def _normalize_wvf_aperture_options(args: tuple[Any, ...]) -> dict[str, Any]:
    options = _wvf_aperture_defaults()
    if len(args) == 1 and isinstance(args[0], dict):
        incoming = {param_format(key): value for key, value in dict(args[0]).items()}
    elif args:
        incoming = _parse_key_value_options(args, "wvfAperture")
    else:
        incoming = {}

    for key, value in incoming.items():
        if key in {"shape"}:
            options["shape"] = str(value)
        elif key in {"nsides", "nside"}:
            options["nsides"] = int(np.asarray(value, dtype=float).reshape(-1)[0])
        elif key in {"aspectratio"}:
            aspect_ratio = np.asarray(value, dtype=float).reshape(-1)
            if aspect_ratio.size != 2:
                raise ValueError("wvfAperture aspect ratio must have two entries.")
            options["aspectratio"] = aspect_ratio.copy()
        elif key in {
            "dotmean",
            "dotsd",
            "dotopacity",
            "dotradius",
            "linemean",
            "linesd",
            "lineopacity",
            "linewidth",
            "segmentlength",
        }:
            options[key] = float(np.asarray(value, dtype=float).reshape(-1)[0])
        elif key in {"texfile"}:
            options["texfile"] = value
        elif key in {"imagerotate"}:
            array = np.asarray(value, dtype=float).reshape(-1)
            options["imagerotate"] = None if array.size == 0 else float(array[0])
        elif key in {"seed", "randomseed"}:
            array = np.asarray(value, dtype=float).reshape(-1)
            options["seed"] = None if array.size == 0 else int(array[0])
        else:
            raise KeyError(f"Unsupported wvfAperture parameter: {key}")
    return options


def _random_points_in_unit_circle(num_points: int, rng: np.random.Generator) -> np.ndarray:
    radius = rng.random(int(num_points), dtype=np.float32)
    theta = rng.random(int(num_points), dtype=np.float32) * (2.0 * np.pi)
    return np.vstack((radius * np.cos(theta), radius * np.sin(theta)))


def _apply_filled_circle(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    color: float,
    blend: bool,
) -> None:
    if radius <= 0:
        return
    rows, cols = image.shape
    x_min = max(int(np.floor(center_x - radius)), 0)
    x_max = min(int(np.ceil(center_x + radius)), cols - 1)
    y_min = max(int(np.floor(center_y - radius)), 0)
    y_max = min(int(np.ceil(center_y + radius)), rows - 1)
    if x_min > x_max or y_min > y_max:
        return
    yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= (radius**2)
    current = image[y_min : y_max + 1, x_min : x_max + 1]
    if blend:
        current[mask] = current[mask] * max(1.0 - float(color), 0.0)
    else:
        current[mask] = float(color)


def _apply_polyline(
    image: np.ndarray,
    vertices_xy: np.ndarray,
    *,
    width: float,
    color: float,
) -> None:
    radius = max(float(width) / 2.0, 0.5)
    for segment_index in range(vertices_xy.shape[1] - 1):
        start_xy = vertices_xy[:, segment_index]
        stop_xy = vertices_xy[:, segment_index + 1]
        length = float(np.linalg.norm(stop_xy - start_xy))
        n_samples = max(int(np.ceil(length * 2.0)), 1)
        samples = np.linspace(0.0, 1.0, n_samples + 1, dtype=float)
        for alpha in samples:
            point = ((1.0 - alpha) * start_xy) + (alpha * stop_xy)
            _apply_filled_circle(
                image,
                float(point[0]),
                float(point[1]),
                radius,
                color=float(color),
                blend=False,
            )


def _polygon_mask_from_vertices(image_size: int, vertices_xy: np.ndarray) -> np.ndarray:
    vertices_rc = np.column_stack((np.floor(vertices_xy[:, 1]) - 1.0, np.floor(vertices_xy[:, 0]) - 1.0))
    return polygon2mask((image_size, image_size), vertices_rc).astype(float)


def wvf_aperture_params() -> dict[str, Any]:
    return _wvf_aperture_defaults()


def wvf_aperture(
    wvf: dict[str, Any],
    *args: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    options = _normalize_wvf_aperture_options(args)
    image_size = int(wvf_get(wvf, "spatial samples"))
    image = np.ones((image_size, image_size), dtype=float)
    seed = options.get("seed")
    rng = np.random.default_rng(None if seed is None else int(seed))

    shape = param_format(options["shape"])
    aspect_ratio = np.asarray(options["aspectratio"], dtype=float).reshape(2)
    n_sides = int(options["nsides"])
    rotate_deg = options["imagerotate"]
    tex_file = options["texfile"]

    dot_mean = float(options["dotmean"])
    dot_sd = float(options["dotsd"])
    dot_opacity = float(options["dotopacity"])
    dot_radius = float(options["dotradius"])
    line_mean = float(options["linemean"])
    line_sd = float(options["linesd"])
    line_opacity = float(options["lineopacity"])
    line_width = float(options["linewidth"])
    segment_length = float(options["segmentlength"])

    if tex_file is None:
        if dot_radius <= 0.0:
            dot_radius = float(max(int(round(image_size / 200.0)), 0))
        num_dots = int(np.round(rng.normal(dot_mean, dot_sd)))
        max_radius = dot_radius * 5.0
        for _ in range(max(num_dots, 0)):
            radius = float(max(np.round(dot_radius + (rng.random() * 5.0)), 0.0))
            radius = min(radius, max_radius)
            center_x = float(rng.random() * image_size)
            center_y = float(rng.random() * image_size)
            opacity = min(dot_opacity + (float(rng.random()) * 0.5), 1.0)
            _apply_filled_circle(
                image,
                center_x,
                center_y,
                radius,
                color=opacity,
                blend=True,
            )

        num_lines = int(np.round(rng.normal(line_mean, line_sd)))
        for _ in range(max(num_lines, 0)):
            num_segments = int(rng.integers(1, 17))
            this_segment_length = float(rng.random() * segment_length)
            start_xy = rng.random(2) * image_size
            segments_xy = _random_points_in_unit_circle(num_segments, rng) * this_segment_length
            vertices_xy = np.cumsum(np.column_stack((start_xy, segments_xy)), axis=1)
            width = int(np.round(max(1.0, line_width + rng.normal(0.0, line_width / 2.0))))
            opacity = line_opacity + (float(rng.random()) * 0.5)
            _apply_polyline(image, vertices_xy, width=width, color=float(opacity))
    else:
        texture = np.asarray(iio.imread(Path(tex_file)), dtype=float)
        if texture.ndim == 3:
            texture = np.mean(texture[:, :, :3], axis=2)
        texture = np.asarray(texture, dtype=float)
        max_value = float(np.max(texture)) if texture.size else 0.0
        if max_value > 0.0:
            texture = texture / max_value
        if float(np.mean(texture)) < 0.2:
            texture = 1.0 - texture
        dark = texture < 0.95
        if np.any(dark):
            texture[dark] = np.maximum(texture[dark] - float(rng.random()), 0.0)
        image = _resize_image(texture, (image_size, image_size), method="linear")

    center_x = (image_size / 2.0) + 1.0
    center_y = (image_size / 2.0) + 1.0
    radius = (image_size - 1.0) / 2.0

    if shape == "polygon":
        if n_sides > 0:
            theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False, dtype=float)
            vertices_x = center_x + (radius * np.cos(theta))
            vertices_y = center_y + (radius * np.sin(theta))
            polygon_mask = _polygon_mask_from_vertices(
                image_size,
                np.column_stack((vertices_x, vertices_y)),
            )
            image = image * polygon_mask
    elif shape == "rectangle":
        mx = max(float(np.max(aspect_ratio)), 1e-12)
        rect_sides = np.round((aspect_ratio / (1.1 * mx)) * image_size)
        ll = np.array([image_size / 2.0 - rect_sides[1] / 2.0, image_size / 2.0 - rect_sides[0] / 2.0], dtype=float)
        ul = np.array([image_size / 2.0 - rect_sides[1] / 2.0, image_size / 2.0 + rect_sides[0] / 2.0], dtype=float)
        ur = np.array([image_size / 2.0 + rect_sides[1] / 2.0, image_size / 2.0 + rect_sides[0] / 2.0], dtype=float)
        lr = np.array([image_size / 2.0 + rect_sides[1] / 2.0, image_size / 2.0 - rect_sides[0] / 2.0], dtype=float)
        corners_xy = np.vstack((ll, ul, ur, lr))
        rectangle_mask = _polygon_mask_from_vertices(image_size, corners_xy)
        image = image * rectangle_mask
    else:
        raise UnsupportedOptionError("wvfAperture", f"shape {options['shape']}")

    if shape == "polygon":
        xx, yy = np.meshgrid((np.arange(1, image_size + 1, dtype=float) - center_x), (np.arange(1, image_size + 1, dtype=float) - center_y))
        image[np.sqrt(xx**2 + yy**2) > radius] = 0.0
        rotation = int(rng.integers(1, 31)) if rotate_deg is None else float(rotate_deg)
        if not np.isclose(rotation, 0.0):
            image = rotate(image, rotation, reshape=True, order=1, mode="constant", cval=0.0, prefilter=False)
    elif shape == "rectangle" and rotate_deg is not None and not np.isclose(float(rotate_deg), 0.0):
        image = rotate(image, float(rotate_deg), reshape=True, order=1, mode="constant", cval=0.0, prefilter=False)

    params = {
        "dotMean": dot_mean,
        "dotSD": dot_sd,
        "dotOpacity": dot_opacity,
        "dotRadius": dot_radius,
        "lineMean": line_mean,
        "lineSD": line_sd,
        "lineOpacity": line_opacity,
        "lineWidth": line_width,
        "segmentLength": segment_length,
        "nsides": n_sides,
    }
    return np.asarray(image, dtype=float), params


def _centered_support_axis(size: int, sample_spacing_m: float) -> np.ndarray:
    center = (int(size) - 1) / 2.0
    return (np.arange(int(size), dtype=float) - center) * float(sample_spacing_m)


def _interpolate_shift_invariant_otf_wavelength(
    otf_data: np.ndarray,
    source_wave: np.ndarray,
    wavelength_nm: float,
) -> np.ndarray:
    source = np.asarray(otf_data, dtype=complex)
    wavelengths = np.asarray(source_wave, dtype=float).reshape(-1)
    if source.shape[2] == 1 or wavelengths.size <= 1:
        return np.asarray(source[:, :, 0], dtype=complex)

    query = float(wavelength_nm)
    if query <= float(wavelengths[0]):
        return np.asarray(source[:, :, 0], dtype=complex)
    if query >= float(wavelengths[-1]):
        return np.asarray(source[:, :, -1], dtype=complex)

    upper_index = int(np.searchsorted(wavelengths, query, side="left"))
    lower_index = max(upper_index - 1, 0)
    lower_wave = float(wavelengths[lower_index])
    upper_wave = float(wavelengths[upper_index])
    if np.isclose(upper_wave, lower_wave):
        return np.asarray(source[:, :, lower_index], dtype=complex)

    weight = (query - lower_wave) / (upper_wave - lower_wave)
    return ((1.0 - weight) * np.asarray(source[:, :, lower_index], dtype=complex)) + (
        weight * np.asarray(source[:, :, upper_index], dtype=complex)
    )


def _resample_complex_otf_on_support(
    plane: np.ndarray,
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    shifted = np.fft.fftshift(np.asarray(plane, dtype=complex))
    real = _resample_plane_on_support(
        np.real(shifted),
        x_support,
        y_support,
        x_query,
        y_query,
        method="linear",
    )
    imag = _resample_plane_on_support(
        np.imag(shifted),
        x_support,
        y_support,
        x_query,
        y_query,
        method="linear",
    )
    yy, xx = np.meshgrid(y_query, x_query, indexing="ij")
    outside = (
        (xx < float(np.min(x_support)))
        | (xx > float(np.max(x_support)))
        | (yy < float(np.min(y_support)))
        | (yy > float(np.max(y_support)))
    )
    result = np.asarray(real, dtype=float) + (1j * np.asarray(imag, dtype=float))
    result[outside] = 0.0
    return np.fft.ifftshift(result)


def _shift_invariant_custom_otf(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
) -> np.ndarray | None:
    otf_data = optics.get("otf_data")
    if otf_data is None:
        return None

    source_otf = np.asarray(otf_data, dtype=complex)
    source_fx = np.asarray(optics.get("otf_fx", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    source_fy = np.asarray(optics.get("otf_fy", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    source_wave = np.asarray(optics.get("otf_wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if source_fx.size == 0 or source_fy.size == 0:
        return None

    rows, cols = shape
    spacing_mm = float(sample_spacing_m) * 1e3
    target_fx = unit_frequency_list(int(cols)) * (1.0 / max(2.0 * spacing_mm, 1e-12))
    target_fy = unit_frequency_list(int(rows)) * (1.0 / max(2.0 * spacing_mm, 1e-12))
    target_wave = np.asarray(wave, dtype=float).reshape(-1)
    otf_stack = np.empty((rows, cols, target_wave.size), dtype=complex)
    for band_index, wavelength_nm in enumerate(target_wave):
        plane = _interpolate_shift_invariant_otf_wavelength(source_otf, source_wave, float(wavelength_nm))
        otf_stack[:, :, band_index] = _resample_complex_otf_on_support(
            plane,
            source_fx,
            source_fy,
            target_fx,
            target_fy,
        )
    return otf_stack


def _shift_invariant_psf_stack(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
    *,
    aperture: np.ndarray | None = None,
) -> np.ndarray:
    psf_data = optics.get("psf_data")
    if not isinstance(psf_data, dict):
        return _wvf_psf_stack(shape, sample_spacing_m, wave, optics, aperture=aperture)

    del aperture
    normalized = _normalize_shift_invariant_psf_data(psf_data)
    source_psf = np.asarray(normalized["psf"], dtype=float)
    source_wave = np.asarray(normalized["wave"], dtype=float).reshape(-1)
    source_spacing_m = float(normalized["sample_spacing_m"])
    source_x = _centered_support_axis(source_psf.shape[1], source_spacing_m)
    source_y = _centered_support_axis(source_psf.shape[0], source_spacing_m)

    n_pixels = int(max(shape[0], shape[1]))
    target_support = _centered_support_axis(n_pixels, sample_spacing_m)
    target_wave = np.asarray(wave, dtype=float).reshape(-1)
    stack = np.empty((n_pixels, n_pixels, target_wave.size), dtype=float)
    for band_index, wavelength_nm in enumerate(target_wave):
        if source_psf.shape[2] == 1:
            plane = source_psf[:, :, 0]
        else:
            source_index = int(np.argmin(np.abs(source_wave - float(wavelength_nm))))
            plane = source_psf[:, :, source_index]
        resampled = _resample_plane_on_support(
            plane,
            source_x,
            source_y,
            target_support,
            target_support,
            method="linear",
        )
        resampled = np.clip(np.asarray(resampled, dtype=float), 0.0, None)
        stack[:, :, band_index] = resampled / max(float(np.sum(resampled)), 1e-12)
    return stack


def _wvf_psf_stack(
    shape: tuple[int, int],
    sample_spacing_m: float,
    wave: np.ndarray,
    optics: dict[str, Any],
    aperture: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = shape
    n_pixels = int(max(rows, cols))
    wavefront = dict(optics.get("wavefront", {}))
    measured_pupil_mm = float(wavefront.get("measured_pupil_diameter_mm", DEFAULT_WVF_MEASURED_PUPIL_MM))
    measured_wavelength_nm = float(wavefront.get("measured_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    compute_sce = bool(wavefront.get("compute_sce", False))
    sce_params = _normalize_sce_params(np.asarray(wave, dtype=float), wavefront.get("sce_params"))
    zcoeffs = np.asarray(wavefront.get("zcoeffs", np.array([0.0], dtype=float)), dtype=float).reshape(-1)
    focal_length_mm = float(optics["focal_length_m"]) * 1e3
    calc_pupil_mm = float(
        wavefront.get(
            "calc_pupil_diameter_mm",
            focal_length_mm / max(float(optics["f_number"]), 1e-12),
        )
    )
    psf_spacing_mm = float(sample_spacing_m) * 1e3
    ref_field_size_mm = measured_wavelength_nm * 1e-6 * focal_length_mm / max(psf_spacing_mm, 1e-12)

    middle_row = np.floor(n_pixels / 2.0) + 1.0
    sample_positions = (np.arange(n_pixels, dtype=float) + 1.0) - middle_row
    calc_radius = calc_pupil_mm / max(measured_pupil_mm, 1e-12)

    psf_stack = np.empty((n_pixels, n_pixels, len(wave)), dtype=float)
    for band_index, wavelength_nm in enumerate(np.asarray(wave, dtype=float).reshape(-1)):
        pupil_plane_size_mm = ref_field_size_mm * (float(wavelength_nm) / max(measured_wavelength_nm, 1e-12))
        pupil_sample_spacing_mm = pupil_plane_size_mm / max(n_pixels, 1)
        pupil_pos = sample_positions * pupil_sample_spacing_mm
        xpos, ypos = np.meshgrid(pupil_pos, -pupil_pos)
        norm_radius = np.sqrt(xpos**2 + ypos**2) / max(measured_pupil_mm / 2.0, 1e-12)
        theta = np.arctan2(ypos, xpos)
        calc_radius_index = norm_radius <= calc_radius
        aperture_mask = _wvf_aperture_mask(n_pixels, calc_radius_index, aperture=aperture)
        if compute_sce:
            rho = _sce_rho_for_wave(sce_params, float(wavelength_nm))
            xo_mm = float(sce_params.get("xo_mm", 0.0))
            yo_mm = float(sce_params.get("yo_mm", 0.0))
            aperture_mask = aperture_mask * np.power(10.0, -rho * ((xpos - xo_mm) ** 2 + (ypos - yo_mm) ** 2))
        lca_microns = _wvf_lca_microns(wavefront, float(wavelength_nm))
        wavefront_aberrations_um = _zernike_surface_osa(_wvf_zcoeffs_with_lca(zcoeffs, lca_microns), norm_radius, theta)
        pupil_phase = np.exp(-1j * 2.0 * np.pi * wavefront_aberrations_um / max(float(wavelength_nm) * 1e-3, 1e-12))
        pupil_phase[norm_radius > calc_radius] = 0.0
        pupil_function = aperture_mask * pupil_phase

        amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))
        intensity = np.real(amp * np.conj(amp))
        psf = intensity / max(float(np.sum(intensity)), 1e-12)
        if bool(wavefront.get("flip_psf_upside_down", False)):
            psf = np.flipud(psf)
        if bool(wavefront.get("rotate_psf_90_degs", False)):
            psf = np.rot90(psf)
        psf_stack[:, :, band_index] = psf

    return psf_stack


def _apply_psf(cube: np.ndarray, psf_stack: np.ndarray) -> np.ndarray:
    rows, cols = cube.shape[:2]
    result = np.empty_like(cube, dtype=float)

    for band_index in range(cube.shape[2]):
        plane = cube[:, :, band_index]
        psf = psf_stack[:, :, band_index]

        if rows != cols:
            delta = abs(cols - rows)
            pre = delta // 2
            post = delta - pre
            if cols < rows:
                padded_plane = np.pad(plane, ((0, 0), (pre, post)))
                filtered = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(padded_plane) * np.fft.fft2(psf)))
                plane_result = np.real(filtered[:, pre : pre + cols])
            else:
                padded_plane = np.pad(plane, ((pre, post), (0, 0)))
                filtered = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(padded_plane) * np.fft.fft2(psf)))
                plane_result = np.real(filtered[pre : pre + rows, :])
        else:
            filtered = np.fft.ifft2(np.fft.fft2(plane) * np.fft.fft2(np.fft.ifftshift(psf)))
            plane_result = np.real(filtered)

        result[:, :, band_index] = plane_result

    return result


def _scene_diagonal_fov_deg(scene: Scene) -> float:
    hfov = float(scene.fields.get("fov_deg", 10.0))
    vfov = float(scene.fields.get("vfov_deg", hfov))
    tangent = np.sqrt(np.tan(np.deg2rad(hfov) / 2.0) ** 2 + np.tan(np.deg2rad(vfov) / 2.0) ** 2)
    return float(np.rad2deg(2.0 * np.arctan(tangent)))


def _raytrace_curve(table: dict[str, Any], wavelength_nm: float) -> np.ndarray:
    values = np.asarray(table.get("function", np.empty(0, dtype=float)), dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    wavelengths = np.asarray(table.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if values.ndim == 1 or wavelengths.size <= 1:
        return np.asarray(values, dtype=float).reshape(-1)
    wave_index = int(np.argmin(np.abs(wavelengths - float(wavelength_nm))))
    return np.asarray(values[:, wave_index], dtype=float).reshape(-1)


def _raytrace_field_height_index_pair(field_height_list: np.ndarray, height: float) -> tuple[int, int]:
    heights = np.asarray(field_height_list, dtype=float).reshape(-1)
    if heights.size == 0:
        raise ValueError("No field-height samples are available.")
    idx1 = int(np.argmin(np.abs(heights - float(height))))
    if heights[idx1] > float(height):
        idx2 = max(0, idx1 - 1)
        idx1, idx2 = idx2, idx1
    else:
        idx2 = min(heights.size - 1, idx1 + 1)
    return idx1, idx2


def ie_field_height_to_index(
    field_height_list: np.ndarray,
    height: float,
    *,
    bounding: bool = False,
) -> int | tuple[int, int]:
    heights = np.asarray(field_height_list, dtype=float).reshape(-1)
    if heights.size == 0:
        raise ValueError("No field-height samples are available.")
    if not bounding:
        return int(np.argmin(np.abs(heights - float(height)))) + 1
    idx1, idx2 = _raytrace_field_height_index_pair(heights, float(height))
    return idx1 + 1, idx2 + 1


def _coerce_optics_for_raytrace(value: OpticalImage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, OpticalImage):
        optics = dict(value.fields.get("optics", {}))
    else:
        optics = dict(value)
    if param_format(optics.get("model", "")) == "raytrace" and isinstance(optics.get("raytrace"), dict):
        return optics
    if "raytrace" in optics or "rayTrace" in optics:
        normalized = _normalize_raytrace_optics(optics)
        if isinstance(normalized.get("raytrace"), dict):
            return normalized
    raise ValueError("Ray-trace optics data are required.")


def rt_psf_interp(
    optics_or_oi: OpticalImage | dict[str, Any],
    field_height_m: float = 0.0,
    field_angle_deg: float = 0.0,
    wavelength_nm: float = 550.0,
    x_grid_m: np.ndarray | None = None,
    y_grid_m: np.ndarray | None = None,
) -> np.ndarray:
    optics = _coerce_optics_for_raytrace(optics_or_oi)
    geometry = dict(optics.get("raytrace", {}).get("geometry", {}))
    distorted_height_m = _raytrace_curve(geometry, float(wavelength_nm)) / 1e3
    if distorted_height_m.size == 0:
        return np.empty((0, 0), dtype=float)

    idx1, idx2 = _raytrace_field_height_index_pair(distorted_height_m, float(field_height_m))
    psf1 = np.asarray(
        _raw_raytrace_psf_function_from_optics(optics, float(distorted_height_m[idx1]), float(wavelength_nm)),
        dtype=float,
    )
    psf2 = np.asarray(
        _raw_raytrace_psf_function_from_optics(optics, float(distorted_height_m[idx2]), float(wavelength_nm)),
        dtype=float,
    )
    if psf1.size == 0:
        return np.empty((0, 0), dtype=float)

    denom = float(distorted_height_m[idx2] - distorted_height_m[idx1])
    if abs(denom) > 0.0:
        height_weight = (float(field_height_m) - float(distorted_height_m[idx1])) / denom
        psf = (1.0 - height_weight) * psf1 + height_weight * psf2
    else:
        psf = psf1

    if not np.isclose(float(field_angle_deg), 0.0):
        psf = rotate(psf, float(field_angle_deg), reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)

    if x_grid_m is None or y_grid_m is None:
        return np.asarray(psf, dtype=float)

    x_grid = np.asarray(x_grid_m, dtype=float)
    y_grid = np.asarray(y_grid_m, dtype=float)
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid_m and y_grid_m must have matching shapes.")

    source_x_mm, source_y_mm = _raytrace_psf_support_axes(optics.get("raytrace", {}).get("psf", {}))
    source_x_m = np.asarray(source_x_mm, dtype=float) / 1e3
    source_y_m = np.asarray(source_y_mm, dtype=float) / 1e3
    if source_x_m.size <= 1:
        col_coords = np.zeros_like(x_grid, dtype=float)
    else:
        col_coords = (x_grid - float(source_x_m[0])) / float(source_x_m[1] - source_x_m[0])
    if source_y_m.size <= 1:
        row_coords = np.zeros_like(y_grid, dtype=float)
    else:
        row_coords = (y_grid - float(source_y_m[0])) / float(source_y_m[1] - source_y_m[0])
    return map_coordinates(psf, [row_coords, col_coords], order=1, mode="constant", cval=0.0, prefilter=False)


def rt_psf_edit(
    optics_or_oi: OpticalImage | dict[str, Any],
    cntr: bool | int = False,
    rot: int = 0,
    visualize_flag: bool | int = False,
) -> dict[str, Any]:
    del visualize_flag
    edited = _normalize_raytrace_optics(_coerce_optics_for_raytrace(optics_or_oi))
    psf = np.asarray(edited.get("raytrace", {}).get("psf", {}).get("function", np.empty(0, dtype=float)), dtype=float)
    if psf.ndim != 4:
        raise ValueError("rtPSFEdit requires 4D ray-trace PSF data.")

    centered = bool(cntr)
    rotation = int(rot) % 4
    if not centered and rotation == 0:
        return edited

    result = psf.copy()
    for height_index in range(result.shape[2]):
        for wave_index in range(result.shape[3]):
            kernel = result[:, :, height_index, wave_index]
            if centered:
                kernel = 0.5 * (kernel + np.flipud(kernel))
                kernel = 0.5 * (kernel + np.fliplr(kernel))
            if rotation:
                kernel = np.rot90(kernel, rotation)
            result[:, :, height_index, wave_index] = kernel

    edited["raytrace"] = dict(edited.get("raytrace", {}))
    edited["raytrace"]["psf"] = dict(edited["raytrace"].get("psf", {}))
    edited["raytrace"]["psf"]["function"] = result
    return edited


def rt_di_interp(
    optics_or_oi: OpticalImage | dict[str, Any],
    wavelength_nm: float,
) -> np.ndarray:
    optics = _coerce_optics_for_raytrace(optics_or_oi)
    return _raytrace_curve(dict(optics.get("raytrace", {}).get("geometry", {})), float(wavelength_nm))


def rt_ri_interp(
    optics_or_oi: OpticalImage | dict[str, Any],
    wavelength_nm: float,
) -> np.ndarray:
    optics = _coerce_optics_for_raytrace(optics_or_oi)
    return _raytrace_curve(dict(optics.get("raytrace", {}).get("relative_illumination", {})), float(wavelength_nm))


def rt_sample_heights(all_heights: np.ndarray, data_height: np.ndarray) -> tuple[np.ndarray, float]:
    img_height = _raytrace_sample_heights(np.asarray(all_heights, dtype=float), np.asarray(data_height, dtype=float))
    max_data_height = float(np.max(np.asarray(data_height, dtype=float))) if np.asarray(data_height).size > 0 else 0.0
    return img_height, max_data_height


def rt_block_center(r_block: int, c_block: int, block_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(block_samples, dtype=float).reshape(-1)
    if samples.size != 2:
        raise ValueError("block_samples must contain row and column counts.")
    return np.array(
        [
            samples[0] * (float(r_block) - 0.5),
            samples[1] * (float(c_block) - 0.5),
        ],
        dtype=float,
    )


def rt_extract_block(
    irrad_pad: np.ndarray,
    block_samples: np.ndarray,
    r_block: int,
    c_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(irrad_pad, dtype=float)
    samples = np.asarray(block_samples, dtype=int).reshape(-1)
    if data.ndim != 2:
        raise ValueError("rtExtractBlock expects a 2D irradiance plane.")
    if samples.size != 2:
        raise ValueError("block_samples must contain row and column counts.")

    r_start = (int(r_block) - 1) * int(samples[0])
    c_start = (int(c_block) - 1) * int(samples[1])
    r_end = int(r_block) * int(samples[0])
    c_end = int(c_block) * int(samples[1])
    if r_end > data.shape[0] or c_end > data.shape[1]:
        raise ValueError("Block outside of data range")

    r_list = np.arange(r_start + 1, r_end + 1, dtype=int)
    c_list = np.arange(c_start + 1, c_end + 1, dtype=int)
    block_data = data[r_start:r_end, c_start:c_end]
    return block_data, r_list, c_list


def rt_insert_block(
    img: np.ndarray,
    filtered_data: np.ndarray,
    block_samples: np.ndarray,
    block_padding: np.ndarray,
    r_block: int,
    c_block: int,
) -> np.ndarray:
    del block_padding
    image = np.asarray(img, dtype=float).copy()
    data = np.asarray(filtered_data, dtype=float)
    samples = np.asarray(block_samples, dtype=int).reshape(-1)
    if image.ndim != 2 or data.ndim != 2:
        raise ValueError("rtInsertBlock expects 2D image data.")
    if samples.size != 2:
        raise ValueError("block_samples must contain row and column counts.")

    if int(r_block) == 1:
        r_start = 0
    else:
        r_start = (int(r_block) - 1) * int(samples[0])
    if int(c_block) == 1:
        c_start = 0
    else:
        c_start = (int(c_block) - 1) * int(samples[1])

    r_end = r_start + data.shape[0]
    c_end = c_start + data.shape[1]
    if r_end > image.shape[0] or c_end > image.shape[1]:
        raise ValueError("Filtered block outside of image range")
    image[r_start:r_end, c_start:c_end] = image[r_start:r_end, c_start:c_end] + data
    return image


def rt_choose_block_size(
    scene: Scene,
    oi: OpticalImage,
    optics: OpticalImage | dict[str, Any] | None = None,
    steps_fh: int = 4,
) -> tuple[int, np.ndarray, np.ndarray]:
    current_optics = _coerce_optics_for_raytrace(oi if optics is None else optics)
    rows = int(scene.fields.get("rows", np.asarray(scene.data.get("photons", np.empty((0, 0, 0)))).shape[0]))
    cols = int(scene.fields.get("cols", np.asarray(scene.data.get("photons", np.empty((0, 0, 0)))).shape[1]))

    diagonal_mm = (float(oi_get(oi, "diagonal")) * 1e3) / 2.0
    field_heights = np.asarray(current_optics.get("raytrace", {}).get("geometry", {}).get("field_height_mm", np.empty(0)), dtype=float)
    if field_heights.size == 0:
        raise ValueError("Ray-trace geometry field heights are required.")
    n_heights = int(ie_field_height_to_index(field_heights, diagonal_mm))
    n_blocks = int(steps_fh) * n_heights + 1

    row_samples = max(1, int(2 ** np.ceil(np.log2(max(rows / max(n_blocks, 1), 1.0)))))
    col_samples = max(1, int(2 ** np.ceil(np.log2(max(cols / max(n_blocks, 1), 1.0)))))
    block_samples = np.array([row_samples, col_samples], dtype=int)
    irrad_padding = np.ceil((np.array([n_blocks * row_samples - rows, n_blocks * col_samples - cols], dtype=float)) / 2.0).astype(int)
    return n_blocks, block_samples, irrad_padding


def rt_filtered_block_support(
    oi: OpticalImage,
    block_samples: np.ndarray,
    block_padding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mm_row = float(oi_get(oi, "hspatialresolution")) * 1e3
    mm_col = float(oi_get(oi, "wspatialresolution")) * 1e3
    row_f = int(block_samples[0]) + 2 * int(block_padding[0])
    col_f = int(block_samples[1]) + 2 * int(block_padding[1])

    block_x = np.arange(1, col_f + 1, dtype=float) * mm_col
    block_x = block_x - block_x[int(np.floor(col_f / 2.0))]
    block_y = np.arange(1, row_f + 1, dtype=float) * mm_row
    block_y = block_y - block_y[int(np.floor(row_f / 2.0))]
    return block_x.reshape(-1), block_y.reshape(-1), mm_row, mm_col


def rt_otf(
    scene: Scene,
    oi: OpticalImage,
    steps_fh: int | None = None,
) -> np.ndarray:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    if photons.ndim != 3 or photons.size == 0:
        raise ValueError("rtOTF requires an optical image with irradiance photons.")

    optics = _coerce_optics_for_raytrace(oi)
    if steps_fh is None:
        steps_fh = int(
            oi.fields.get(
                "rt_blocks_per_field_height",
                optics.get("raytrace", {}).get("blocks_per_field_height", 4),
            )
        )

    wave = np.asarray(oi.fields.get("wave", scene.fields.get("wave", np.empty(0, dtype=float))), dtype=float).reshape(-1)
    if wave.size != photons.shape[2]:
        raise ValueError("Optical image wavelength sampling must match photon planes for rtOTF.")

    n_blocks, block_samples, irrad_padding = rt_choose_block_size(scene, oi, optics=optics, steps_fh=int(steps_fh))
    block_padding = (block_samples // 2).astype(int)
    row_p = int(n_blocks * block_samples[0])
    col_p = int(n_blocks * block_samples[1])
    row_o = int(row_p + 2 * block_padding[0])
    col_o = int(col_p + 2 * block_padding[1])
    output = np.zeros((row_o, col_o, wave.size), dtype=float)

    block_x_mm, block_y_mm, mm_row, mm_col = rt_filtered_block_support(oi, block_samples, block_padding)
    block_x_grid_mm, block_y_grid_mm = np.meshgrid(block_x_mm, block_y_mm)
    image_center = np.array([np.floor(row_p / 2.0) + 1.0, np.floor(col_p / 2.0) + 1.0], dtype=float)

    for wave_index, wavelength_nm in enumerate(wave):
        irradiance = photons[:, :, wave_index]
        irradiance_padded = np.pad(
            irradiance,
            ((int(irrad_padding[0]), int(irrad_padding[0])), (int(irrad_padding[1]), int(irrad_padding[1]))),
            mode="constant",
        )
        for r_block in range(1, n_blocks + 1):
            for c_block in range(1, n_blocks + 1):
                block_data, _, _ = rt_extract_block(irradiance_padded, block_samples, r_block, c_block)
                center_delta = rt_block_center(r_block, c_block, block_samples) - image_center
                field_angle_deg = float(np.rad2deg(np.arctan2(center_delta[1] * mm_col, center_delta[0] * mm_row)))
                field_height_m = float(np.hypot(center_delta[0] * mm_row, center_delta[1] * mm_col) / 1e3)
                psf = rt_psf_interp(
                    optics,
                    field_height_m=field_height_m,
                    field_angle_deg=field_angle_deg,
                    wavelength_nm=float(wavelength_nm),
                    x_grid_m=block_x_grid_mm / 1e3,
                    y_grid_m=block_y_grid_mm / 1e3,
                )
                if psf.size == 0:
                    filtered_data = np.pad(
                        block_data,
                        ((int(block_padding[0]), int(block_padding[0])), (int(block_padding[1]), int(block_padding[1]))),
                        mode="constant",
                    )
                else:
                    psf = np.asarray(psf, dtype=float)
                    psf[np.isnan(psf)] = 0.0
                    psf_sum = float(np.sum(psf))
                    if psf_sum > 0.0:
                        psf = psf / psf_sum
                    filtered_data = np.pad(
                        block_data,
                        ((int(block_padding[0]), int(block_padding[0])), (int(block_padding[1]), int(block_padding[1]))),
                        mode="constant",
                    )
                    if np.max(psf) < 0.98:
                        filtered_data = fftconvolve(filtered_data, psf, mode="same")
                output[:, :, wave_index] = rt_insert_block(
                    output[:, :, wave_index],
                    filtered_data,
                    block_samples,
                    block_padding,
                    r_block,
                    c_block,
                )

    return output


def rt_psf_grid(
    oi: OpticalImage,
    units: str = "m",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = _spatial_unit_scale(units)
    sample_spacing = np.asarray(
        [
            float(oi_get(oi, "wspatialresolution", units)),
            float(oi_get(oi, "hspatialresolution", units)),
        ],
        dtype=float,
    )
    optics = _coerce_optics_for_raytrace(oi)
    psf_data = dict(optics.get("raytrace", {}).get("psf", {}))
    source_x_mm, source_y_mm = _raytrace_psf_support_axes(psf_data)
    source_x = np.asarray(source_x_mm, dtype=float) / 1e3 * scale
    source_y = np.asarray(source_y_mm, dtype=float) / 1e3 * scale

    if source_x.size == 0 or source_y.size == 0:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty, sample_spacing

    x_positive = np.arange(0.0, float(source_x[-1]) + float(sample_spacing[0]) * 0.5, float(sample_spacing[0]))
    x_negative = -np.flip(np.arange(0.0, abs(float(source_x[0])) + float(sample_spacing[0]) * 0.5, float(sample_spacing[0])))
    y_positive = np.arange(0.0, float(source_y[-1]) + float(sample_spacing[1]) * 0.5, float(sample_spacing[1]))
    y_negative = -np.flip(np.arange(0.0, abs(float(source_y[0])) + float(sample_spacing[1]) * 0.5, float(sample_spacing[1])))
    x_grid = np.concatenate((x_negative[:-1], x_positive))
    y_grid = np.concatenate((y_negative[:-1], y_positive))
    xx, yy = np.meshgrid(x_grid, y_grid)
    return xx, yy, sample_spacing


def rt_angle_lut(psf_struct: dict[str, Any] | OpticalImage) -> np.ndarray:
    if isinstance(psf_struct, OpticalImage):
        current = oi_get(psf_struct, "psf struct")
        if current is None:
            raise ValueError("Precomputed ray-trace psfStruct is required.")
        psf_struct = current
    normalized = _normalize_psf_struct(psf_struct)
    if not isinstance(normalized, dict):
        raise ValueError("Precomputed ray-trace psfStruct is required.")
    sample_angles = np.asarray(normalized.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if sample_angles.size < 2:
        raise ValueError("Precomputed ray-trace psfStruct must include sample angles.")
    lower_index, weight = _raytrace_angle_lut(sample_angles)
    return np.column_stack((lower_index + 1, weight))


def _prepare_raytrace_scene(oi: OpticalImage, scene: Scene) -> tuple[dict[str, Any], Scene]:
    optics = _coerce_optics_for_raytrace(oi)
    object_distance = float(optics.get("raytrace", {}).get("object_distance_m", scene.fields.get("distance_m", np.inf)))
    if np.isclose(float(scene.fields.get("distance_m", np.inf)), object_distance):
        return optics, scene
    compute_scene = scene.clone()
    compute_scene.fields["distance_m"] = object_distance
    return optics, compute_scene


def rt_geometry(
    oi: OpticalImage,
    scene: Scene,
    p_num: int = 8,
) -> OpticalImage:
    optics, compute_scene = _prepare_raytrace_scene(oi, scene)
    scene_photons = np.asarray(compute_scene.data["photons"], dtype=float)
    wave = np.asarray(compute_scene.fields["wave"], dtype=float)
    image_distance_m, width_m, height_m = _oi_geometry(optics, compute_scene)
    sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
    photons = _radiance_to_irradiance(scene_photons, optics, compute_scene)
    result = _raytrace_geometry(photons, wave, optics, compute_scene)

    computed = oi.clone()
    computed.name = compute_scene.name
    computed.fields["wave"] = wave
    computed.fields["compute_method"] = optics.get("compute_method", computed.fields.get("compute_method", ""))
    computed.fields["padding_pixels"] = (0, 0)
    computed.fields["sample_spacing_m"] = float(sample_spacing_m)
    computed.fields["image_distance_m"] = float(image_distance_m)
    computed.fields["depth_map_m"] = np.asarray(scene_get(compute_scene, "depth map"), dtype=float).copy()
    computed.fields["width_m"] = float(width_m)
    computed.fields["height_m"] = float(height_m)
    computed.fields["fov_deg"] = float(compute_scene.fields.get("fov_deg", 10.0))
    computed.fields["vfov_deg"] = float(compute_scene.fields.get("vfov_deg", computed.fields["fov_deg"]))
    computed.data["photons"] = result
    return computed


def rt_precompute_psf(
    oi: OpticalImage,
    angle_step_deg: float | None = None,
) -> dict[str, Any]:
    optics = _coerce_optics_for_raytrace(oi)
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    if photons.ndim != 3 or photons.size == 0:
        raise ValueError("Ray-trace precompute requires an optical image with photons.")
    sample_spacing_m = _oi_sample_size_m(oi)
    if sample_spacing_m is None:
        raise ValueError("Ray-trace precompute requires optical image sample spacing.")
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if wave.size == 0:
        raise ValueError("Ray-trace precompute requires optical image wavelengths.")
    if angle_step_deg is None:
        angle_step_deg = float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
    sample_angles_deg = _raytrace_requested_sample_angles(float(angle_step_deg), oi.fields.get("psf_sample_angles_deg"))
    psf_struct = _raytrace_precompute_psf(
        optics,
        wave,
        int(photons.shape[0]),
        int(photons.shape[1]),
        float(sample_spacing_m),
        sample_angles_deg=sample_angles_deg,
    )
    return _export_psf_struct(psf_struct)


def rt_precompute_psf_apply(
    oi: OpticalImage,
    angle_step_deg: float | None = None,
) -> OpticalImage:
    optics = _coerce_optics_for_raytrace(oi)
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    if photons.ndim != 3 or photons.size == 0:
        raise ValueError("Ray-trace PSF application requires an optical image with photons.")
    sample_spacing_m = _oi_sample_size_m(oi)
    if sample_spacing_m is None:
        raise ValueError("Ray-trace PSF application requires optical image sample spacing.")
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if wave.size == 0:
        raise ValueError("Ray-trace PSF application requires optical image wavelengths.")
    if angle_step_deg is None:
        angle_step_deg = float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
    sample_angles_deg = _raytrace_requested_sample_angles(float(angle_step_deg), oi.fields.get("psf_sample_angles_deg"))
    psf_struct = _raytrace_finalize_psf_struct(
        oi.fields.get("psf_struct"),
        optics,
        wave,
        int(photons.shape[0]),
        int(photons.shape[1]),
        float(sample_spacing_m),
        sample_angles_deg,
    )
    if not _raytrace_psf_struct_matches(
        psf_struct,
        optics,
        wave,
        int(photons.shape[0]),
        int(photons.shape[1]),
        float(sample_spacing_m),
        sample_angles_deg,
    ):
        psf_struct = _raytrace_precompute_psf(
            optics,
            wave,
            int(photons.shape[0]),
            int(photons.shape[1]),
            float(sample_spacing_m),
            sample_angles_deg=sample_angles_deg,
        )
    assert psf_struct is not None

    pad_pixels = _raytrace_padding_pixels(psf_struct)
    result = _raytrace_apply_psf(photons, psf_struct, float(sample_spacing_m), pad_pixels=pad_pixels)
    image_distance_m = _oi_image_distance_m(oi)
    output_width_m = float(result.shape[1] * float(sample_spacing_m))
    output_height_m = float(result.shape[0] * float(sample_spacing_m))
    output_fov_deg = float(np.rad2deg(2.0 * np.arctan2(output_width_m / 2.0, image_distance_m)))
    output_vfov_deg = float(np.rad2deg(2.0 * np.arctan2(output_height_m / 2.0, image_distance_m)))

    computed = oi.clone()
    computed.fields["padding_pixels"] = pad_pixels
    computed.fields["sample_spacing_m"] = float(sample_spacing_m)
    computed.fields["image_distance_m"] = float(image_distance_m)
    computed.fields["width_m"] = output_width_m
    computed.fields["height_m"] = output_height_m
    computed.fields["fov_deg"] = output_fov_deg
    computed.fields["vfov_deg"] = output_vfov_deg
    computed.fields["psf_angle_step_deg"] = float(angle_step_deg)
    computed.fields["psf_struct"] = psf_struct
    depth_map = oi.fields.get("depth_map_m")
    if depth_map is not None:
        computed.fields["depth_map_m"] = np.pad(
            np.asarray(depth_map, dtype=float),
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1])),
            mode="constant",
            constant_values=0.0,
        )
    _sync_psf_metadata_fields(computed)
    computed.data["photons"] = result
    return computed


def rt_psf_apply(
    oi: OpticalImage,
    angle_step_deg: float | None = None,
) -> OpticalImage:
    return rt_precompute_psf_apply(oi, angle_step_deg=angle_step_deg)


def _oi_illuminance(oi: OpticalImage) -> np.ndarray:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if photons.ndim != 3 or photons.size == 0 or wave.size == 0:
        if photons.ndim >= 2:
            return np.empty(photons.shape[:2], dtype=float)
        return np.empty((0, 0), dtype=float)
    store = AssetStore.default()
    _, luminosity = store.load_luminosity(wave_nm=wave)
    energy = quanta_to_energy(photons, wave)
    return 683.0 * spectral_step(wave) * np.tensordot(energy, luminosity, axes=([2], [0]))


def oi_calculate_illuminance(oi: OpticalImage) -> tuple[np.ndarray, float, float]:
    illuminance = _oi_illuminance(oi)
    mean_illuminance = 0.0 if illuminance.size == 0 else float(np.mean(illuminance))

    mean_comp_illuminance = 0.0
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if photons.ndim == 3 and photons.size > 0 and wave.size > 0 and np.any(wave > 750.0):
        energy = quanta_to_energy(photons, wave)
        comp_illuminance = 683.0 * spectral_step(wave) * np.sum(energy, axis=2)
        mean_comp_illuminance = float(np.mean(comp_illuminance))

    oi.fields["illuminance"] = illuminance
    oi.fields["mean_illuminance"] = mean_illuminance
    oi.fields["mean_comp_illuminance"] = mean_comp_illuminance
    return illuminance, mean_illuminance, mean_comp_illuminance


def oi_calculate_irradiance(scene: Scene, optics: Any) -> np.ndarray:
    """Convert scene radiance into optical-image irradiance."""

    optics_struct = dict(optics.fields.get("optics", {})) if isinstance(optics, OpticalImage) else dict(optics)
    scene_photons = np.asarray(scene.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    return _radiance_to_irradiance(scene_photons, optics_struct, scene)


def oi_adjust_illuminance(oi: OpticalImage, new_level: float, stat: str = "mean") -> OpticalImage:
    """Scale an OI to the requested mean or peak illuminance."""

    adjusted = oi.clone()
    illuminance = np.asarray(oi_get(adjusted, "illuminance"), dtype=float)
    if illuminance.size == 0:
        return adjusted

    mode = param_format(stat)
    if mode == "mean":
        current_level = float(np.mean(illuminance))
    elif mode in {"max", "peak"}:
        current_level = float(np.max(illuminance))
    else:
        raise ValueError(f"Unknown oiAdjustIlluminance statistic: {stat}")
    if current_level <= 0.0:
        return adjusted

    scale = float(new_level) / current_level
    photons = np.asarray(oi_get(adjusted, "photons"), dtype=float)
    adjusted = oi_set(adjusted, "photons", photons * scale)
    oi_calculate_illuminance(adjusted)
    return adjusted


def _oi_resample_wave_last(values: np.ndarray, source_wave_nm: np.ndarray, target_wave_nm: np.ndarray) -> np.ndarray:
    wave_first = np.moveaxis(np.asarray(values, dtype=float), -1, 0)
    resampled = interp_spectra(np.asarray(source_wave_nm, dtype=float), wave_first, np.asarray(target_wave_nm, dtype=float))
    return np.moveaxis(np.asarray(resampled, dtype=float), 0, -1)


def oi_interpolate_w(oi: OpticalImage, new_wave: Any) -> OpticalImage:
    """Interpolate the wavelength dimension of an optical image."""

    current = oi.clone()
    source_wave = np.asarray(oi_get(current, "wave"), dtype=float).reshape(-1)
    target_wave = np.asarray(new_wave, dtype=float).reshape(-1)
    if target_wave.size == 0:
        raise ValueError("oiInterpolateW target wavelength samples must not be empty.")
    if source_wave.size == 0 or np.array_equal(source_wave, target_wave):
        return oi_set(current, "wave", target_wave)
    if float(np.min(target_wave)) < float(np.min(source_wave)) or float(np.max(target_wave)) > float(np.max(source_wave)):
        raise ValueError("oiInterpolateW does not support extrapolation outside the current wavelength support.")

    photons = np.asarray(oi_get(current, "photons"), dtype=float)
    if photons.ndim != 3 or photons.shape[2] != source_wave.size:
        raise ValueError("oiInterpolateW requires a wave-last OI photon cube.")
    original_mean = float(oi_get(current, "mean illuminance"))
    interpolated = _oi_resample_wave_last(photons, source_wave, target_wave)
    current = oi_set(current, "wave", target_wave)
    current = oi_set(current, "photons", interpolated)
    oi_calculate_illuminance(current)
    return oi_adjust_illuminance(current, original_mean, "mean")


def oi_extract_waveband(oi: OpticalImage, wave_list: Any, illuminance_flag: Any = 0) -> OpticalImage:
    """Extract the requested OI wavelength bands, interpolating when needed."""

    current = oi.clone()
    target_wave = np.asarray(wave_list, dtype=float).reshape(-1)
    if target_wave.size == 0:
        raise ValueError("oiExtractWaveband requires a non-empty wavelength list.")

    source_wave = np.asarray(oi_get(current, "wave"), dtype=float).reshape(-1)
    photons = np.asarray(oi_get(current, "photons"), dtype=float)
    if photons.ndim != 3 or photons.shape[2] != source_wave.size:
        raise ValueError("oiExtractWaveband requires a wave-last OI photon cube.")

    extracted = _oi_resample_wave_last(photons, source_wave, target_wave)
    current = oi_set(current, "wave", target_wave)
    current = oi_set(current, "photons", extracted)
    if bool(illuminance_flag):
        oi_calculate_illuminance(current)
    else:
        current.fields.pop("illuminance", None)
        current.fields.pop("mean_illuminance", None)
        current.fields.pop("mean_comp_illuminance", None)
    return current


def oi_add(in1: Any, in2: Any, add_flag: str = "add") -> OpticalImage:
    """Combine matched optical images using the legacy MATLAB oiAdd contract."""

    flag = param_format(add_flag)
    if flag not in {"add", "removespatialmean"}:
        raise ValueError(f"Unknown oiAdd addFlag: {add_flag}")

    def _prepare_component(component: OpticalImage, *, remove_spatial_mean: bool) -> np.ndarray:
        photons = np.asarray(oi_get(component, "photons"), dtype=float)
        if remove_spatial_mean:
            photons = photons - np.mean(photons, axis=(0, 1), keepdims=True)
        return photons

    if isinstance(in1, (list, tuple)):
        ois = list(in1)
        weights = np.asarray(in2, dtype=float).reshape(-1)
        if len(ois) == 0:
            raise ValueError("oiAdd requires at least one optical image.")
        if weights.size != len(ois):
            raise ValueError("oiAdd weights must match the number of optical images.")
        reference_shape = np.asarray(oi_get(ois[0], "photons"), dtype=float).shape
        reference_wave = np.asarray(oi_get(ois[0], "wave"), dtype=float)
        combined = weights[0] * np.asarray(oi_get(ois[0], "photons"), dtype=float)
        for index, component in enumerate(ois[1:], start=1):
            photons = np.asarray(oi_get(component, "photons"), dtype=float)
            if photons.shape != reference_shape or not np.array_equal(np.asarray(oi_get(component, "wave"), dtype=float), reference_wave):
                raise ValueError("oiAdd requires matched OI geometry and wavelength support.")
            combined = combined + weights[index] * _prepare_component(component, remove_spatial_mean=(flag == "removespatialmean"))
        output = ois[0].clone()
    else:
        reference_shape = np.asarray(oi_get(in1, "photons"), dtype=float).shape
        other = np.asarray(oi_get(in2, "photons"), dtype=float)
        if other.shape != reference_shape or not np.array_equal(np.asarray(oi_get(in1, "wave"), dtype=float), np.asarray(oi_get(in2, "wave"), dtype=float)):
            raise ValueError("oiAdd requires matched OI geometry and wavelength support.")
        combined = np.asarray(oi_get(in1, "photons"), dtype=float) + _prepare_component(in2, remove_spatial_mean=(flag == "removespatialmean"))
        output = in1.clone()

    output = oi_set(output, "photons", combined)
    oi_calculate_illuminance(output)
    return output


def _normalize_oi_pad_size(pad_size: Any) -> tuple[int, int, int]:
    values = np.asarray(pad_size, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("oiPadValue requires a non-empty pad size.")
    if values.size == 1:
        rows = cols = int(np.rint(values[0]))
        waves = 0
    elif values.size == 2:
        rows = int(np.rint(values[0]))
        cols = int(np.rint(values[1]))
        waves = 0
    else:
        rows = int(np.rint(values[0]))
        cols = int(np.rint(values[1]))
        waves = int(np.rint(values[2]))
    if rows < 0 or cols < 0 or waves < 0:
        raise ValueError("oiPadValue pad sizes must be non-negative.")
    return rows, cols, waves


def _oi_pad_width_scale(oi: OpticalImage, pad_cols: int) -> float:
    if pad_cols <= 0:
        return float(oi_get(oi, "hfov"))
    width_m = float(oi_get(oi, "width"))
    cols = int(oi_get(oi, "cols"))
    image_distance = float(oi_get(oi, "image distance"))
    if width_m <= 0.0 or cols <= 0 or image_distance <= 0.0:
        return float(oi_get(oi, "hfov"))
    new_width_m = width_m * (1.0 + (float(pad_cols) / float(cols)))
    return float(np.rad2deg(2.0 * np.arctan2(new_width_m / 2.0, image_distance)))


def _oi_pad_bandwise(
    photons: np.ndarray,
    row_pad: tuple[int, int],
    col_pad: tuple[int, int],
    band_values: np.ndarray,
) -> np.ndarray:
    padded_bands: list[np.ndarray] = []
    for band_index in range(photons.shape[2]):
        padded_bands.append(
            np.pad(
                photons[:, :, band_index],
                (row_pad, col_pad),
                mode="constant",
                constant_values=float(band_values[band_index]),
            )
        )
    return np.stack(padded_bands, axis=2)


def _oi_pad_wave_constant(
    photons: np.ndarray,
    wave_pad: tuple[int, int],
    pad_value: float,
) -> np.ndarray:
    if wave_pad == (0, 0):
        return photons
    return np.pad(
        photons,
        ((0, 0), (0, 0), wave_pad),
        mode="constant",
        constant_values=float(pad_value),
    )


def oi_pad_value(
    oi: OpticalImage,
    pad_size: Any,
    pad_type: Any = "mean photons",
    s_dist: Any | None = None,
    direction: str = "both",
) -> OpticalImage:
    """Pad an optical image using the legacy MATLAB oiPadValue contract."""

    del s_dist
    padded = oi.clone()
    photons = np.asarray(oi_get(padded, "photons"), dtype=float)
    if photons.ndim != 3:
        raise ValueError("oiPadValue requires a wave-last OI photon cube.")

    pad_rows, pad_cols, pad_waves = _normalize_oi_pad_size(pad_size)
    mode = param_format(direction)
    if mode == "both":
        row_pad = (pad_rows, pad_rows)
        col_pad = (pad_cols, pad_cols)
        wave_pad = (pad_waves, pad_waves)
        total_cols = 2 * pad_cols
    elif mode == "pre":
        row_pad = (pad_rows, 0)
        col_pad = (pad_cols, 0)
        wave_pad = (pad_waves, 0)
        total_cols = pad_cols
    elif mode == "post":
        row_pad = (0, pad_rows)
        col_pad = (0, pad_cols)
        wave_pad = (0, pad_waves)
        total_cols = pad_cols
    else:
        raise ValueError(f"Unknown oiPadValue direction: {direction}")

    band_values: np.ndarray
    wave_value: float
    if isinstance(pad_type, str):
        normalized = param_format(pad_type)
        if normalized in {"zerophotons", "zero", "zeros"}:
            band_values = np.zeros(photons.shape[2], dtype=float)
            wave_value = 0.0
            stored_pad_value: Any = "zero photons"
        elif normalized in {"meanphotons", "mean"}:
            band_values = np.mean(photons, axis=(0, 1), dtype=float).reshape(-1)
            wave_value = float(np.mean(band_values)) if band_values.size else 0.0
            stored_pad_value = "mean photons"
        elif normalized in {"borderphotons", "border"}:
            band_values = np.asarray(photons[0, 0, :], dtype=float).reshape(-1)
            wave_value = float(band_values[0]) if band_values.size else 0.0
            stored_pad_value = "border photons"
        else:
            raise ValueError(f"Unknown oiPadValue padType: {pad_type}")
    else:
        wave_value = float(np.asarray(pad_type, dtype=float).reshape(-1)[0])
        band_values = np.full(photons.shape[2], wave_value, dtype=float)
        stored_pad_value = wave_value

    padded_photons = _oi_pad_bandwise(photons, row_pad, col_pad, band_values)
    padded_photons = _oi_pad_wave_constant(padded_photons, wave_pad, wave_value)

    padded = oi_set(padded, "photons", padded_photons)
    padded = oi_set(padded, "hfov", _oi_pad_width_scale(oi, total_cols))
    padded.fields["pad_value"] = stored_pad_value
    padded.fields["padding_pixels"] = (int(pad_rows), int(pad_cols))
    padded.fields["depth_map_m"] = None
    oi_calculate_illuminance(padded)
    return padded


def oi_pad(
    oi: OpticalImage,
    pad_size: Any,
    s_dist: Any | None = None,
    direction: str = "both",
) -> OpticalImage:
    """Pad an optical image using the deprecated near-zero MATLAB contract."""

    photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    data_max = float(np.max(photons)) if photons.size else 0.0
    return oi_pad_value(oi, pad_size, data_max * 1.0e-9, s_dist=s_dist, direction=direction)


def oi_make_even_row_col(
    oi: OpticalImage,
    s_dist: Any | None = None,
) -> OpticalImage:
    """Pad odd OI dimensions to even sizes using the legacy MATLAB contract."""

    rows, cols = np.asarray(oi_get(oi, "size"), dtype=int).reshape(2)
    pad_rows = int(rows % 2)
    pad_cols = int(cols % 2)
    if pad_rows == 0 and pad_cols == 0:
        return oi.clone()
    return oi_pad(oi, [pad_rows, pad_cols, 0], s_dist=s_dist, direction="post")


def oi_pad_depth_map(scene: Scene, invert: Any = 0, *args: Any) -> np.ndarray:
    """Pad a scene depth map into OI coordinates using the legacy MATLAB contract."""

    if bool(invert):
        raise UnsupportedOptionError("oiPadDepthMap", "invert=True")
    pad_pixels = tuple(np.rint(np.asarray(scene_get(scene, "size"), dtype=float).reshape(2) / 8.0).astype(int))
    return _pad_depth_map(scene, pad_pixels)


def oi_depth_segment_map(oi_dmap: Any, depth_edges: Any) -> np.ndarray:
    """Assign each OI depth-map sample to the closest depth plane."""

    depth_map = np.asarray(oi_dmap, dtype=float)
    edges = np.asarray(depth_edges, dtype=float).reshape(-1)
    if depth_map.ndim != 2:
        raise ValueError("oiDepthSegmentMap requires a 2-D depth map.")
    if edges.size == 0:
        raise ValueError("oiDepthSegmentMap requires at least one depth edge.")
    distance = np.abs(depth_map[:, :, np.newaxis] - edges.reshape(1, 1, -1))
    return np.argmin(distance, axis=2).astype(int) + 1


def _validate_depth_oi_stack(oi_depths: list[OpticalImage]) -> tuple[tuple[int, int, int], np.ndarray]:
    if not oi_depths:
        raise ValueError("At least one optical image is required.")
    reference_photons = np.asarray(oi_get(oi_depths[0], "photons"), dtype=float)
    reference_wave = np.asarray(oi_get(oi_depths[0], "wave"), dtype=float).reshape(-1)
    if reference_photons.ndim != 3:
        raise ValueError("Depth-combine helpers require wave-last OI photon cubes.")
    for current in oi_depths[1:]:
        photons = np.asarray(oi_get(current, "photons"), dtype=float)
        wave = np.asarray(oi_get(current, "wave"), dtype=float).reshape(-1)
        if photons.shape != reference_photons.shape:
            raise ValueError("All optical images in the depth stack must share the same photon-cube shape.")
        if not np.array_equal(wave, reference_wave):
            raise ValueError("All optical images in the depth stack must share the same wavelength sampling.")
    return reference_photons.shape, reference_wave


def oi_depth_combine(oi_depths: list[OpticalImage] | tuple[OpticalImage, ...], scene: Scene, depth_edges: Any) -> OpticalImage:
    """Combine defocused optical images using the scene depth-map segmentation."""

    current_depths = list(oi_depths)
    shape, _ = _validate_depth_oi_stack(current_depths)
    edges = np.asarray(depth_edges, dtype=float).reshape(-1)
    if edges.size != len(current_depths):
        raise ValueError("oiDepthCombine requires one depth edge per optical image.")

    depth_map = oi_pad_depth_map(scene)
    if depth_map.shape != shape[:2]:
        raise ValueError("The padded scene depth map must match the optical image size.")
    idx = oi_depth_segment_map(depth_map, edges) - 1

    photons_stack = np.stack([np.asarray(oi_get(oi, "photons"), dtype=float) for oi in current_depths], axis=3)
    selected = np.take_along_axis(photons_stack, idx[:, :, np.newaxis, np.newaxis], axis=3).squeeze(axis=3)

    combined = current_depths[0].clone()
    combined = oi_set(combined, "photons", selected)
    combined = oi_set(combined, "depth map", depth_map)
    oi_calculate_illuminance(combined)
    combined = oi_set(combined, "name", "Combined")
    return combined


def oi_combine_depths(oi_depths: list[OpticalImage] | tuple[OpticalImage, ...]) -> OpticalImage:
    """Combine an ordered depth stack from back to front using legacy MATLAB semantics."""

    current_depths = list(oi_depths)
    shape, _ = _validate_depth_oi_stack(current_depths)
    photons = np.sum(np.stack([np.asarray(oi_get(oi, "photons"), dtype=float) for oi in current_depths], axis=0), axis=0)

    depth_maps = np.stack(
        [np.asarray(oi_get(oi, "depth map"), dtype=float) for oi in current_depths],
        axis=2,
    )
    if depth_maps.shape[:2] != shape[:2]:
        raise ValueError("All depth maps in the OI stack must match the optical image size.")
    positive = depth_maps[depth_maps > 0.0]
    if positive.size == 0:
        depth_map = np.zeros(shape[:2], dtype=float)
    else:
        far_fill = 2.0 * float(np.max(positive))
        adjusted = depth_maps.copy()
        adjusted[adjusted <= 0.0] = far_fill
        depth_map = np.min(adjusted, axis=2)
        depth_map[depth_map == far_fill] = 0.0

    combined = current_depths[0].clone()
    combined = oi_set(combined, "photons", photons)
    combined = oi_set(combined, "depth map", depth_map)
    oi_calculate_illuminance(combined)
    return combined


def _scene_depth_range(scene: Scene, depth_edges: Any) -> tuple[Scene, np.ndarray]:
    """Restrict a scene to the requested depth slab using MATLAB sceneDepthRange semantics."""

    edges = np.asarray(depth_edges, dtype=float).reshape(-1)
    if edges.size != 2:
        raise ValueError("sceneDepthRange requires a two-element depth range.")

    depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
    depth_plane = (float(edges[0]) <= depth_map) & (depth_map < float(edges[1]))

    current = scene.clone()
    photons = np.asarray(scene_get(current, "photons"), dtype=float).copy()
    photons[~depth_plane, :] = 0.0

    from .scene import scene_set

    current = scene_set(current, "photons", photons)
    current = scene_set(current, "depth map", depth_map * depth_plane.astype(float))
    return current, depth_plane


def oi_depth_edges(
    oi: OpticalImage,
    defocus: Any,
    in_focus_depth: Any,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Determine depth edges that achieve the requested defocus values."""

    optics = oi_get(oi, "optics")
    focal_length = float(optics_get(optics, "focal length", "m"))
    defocus_values = np.asarray(defocus, dtype=float).reshape(-1).copy()
    defocus_values[defocus_values >= 0.0] = -0.01

    depth_edges = np.asarray(optics_defocus_depth(defocus_values, optics, focal_length), dtype=float).reshape(-1)
    target_depth = float(np.asarray(in_focus_depth, dtype=float).reshape(-1)[0])
    nearest_index = int(np.argmin(np.abs(target_depth - depth_edges)))
    object_distance = float(depth_edges[nearest_index])
    _, image_distance = optics_depth_defocus(object_distance, optics, focal_length)
    object_defocus, _ = optics_depth_defocus(depth_edges, optics, image_distance)
    return depth_edges, float(image_distance), np.asarray(object_defocus, dtype=float).reshape(-1)


def s3d_render_depth_defocus(
    scene: Scene,
    oi: OpticalImage,
    img_plane_dist: Any | None = None,
    depth_edges: Any | None = None,
    c_aberration: Any | None = None,
) -> tuple[OpticalImage, list[OpticalImage], np.ndarray]:
    """Compute a defocused OI stack and combined OI across scene depth slabs."""

    if scene is None:
        raise ValueError("Scene required")
    if oi is None:
        raise ValueError("oi required")

    optics = oi_get(oi, "optics")
    if img_plane_dist is None:
        img_plane_distance = float(optics_get(optics, "focal length", "m"))
    else:
        img_plane_distance = float(np.asarray(img_plane_dist, dtype=float).reshape(-1)[0])

    depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
    if depth_edges is None:
        edges = np.array([float(np.min(depth_map)), float(np.max(depth_map))], dtype=float)
    else:
        edges = np.asarray(depth_edges, dtype=float).reshape(-1)
        if edges.size == 1:
            edges = np.array([float(np.min(depth_map)), float(edges[0]), float(np.max(depth_map))], dtype=float)
    if edges.size < 2:
        raise ValueError("s3dRenderDepthDefocus requires at least two depth edges.")

    depth_centers = edges[:-1] + (np.diff(edges) / 2.0)
    wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    if c_aberration is None:
        chromatic_aberration = np.zeros(wave.size, dtype=float)
    else:
        chromatic_aberration = np.asarray(c_aberration, dtype=float).reshape(-1)
        if chromatic_aberration.size == 1:
            chromatic_aberration = np.full(wave.size, float(chromatic_aberration[0]), dtype=float)
        if chromatic_aberration.size != wave.size:
            raise ValueError("cAberration must be scalar or match the scene wavelength support.")

    defocus_diopters, _ = optics_depth_defocus(depth_centers, optics, img_plane_distance)
    defocus_diopters = np.asarray(defocus_diopters, dtype=float).reshape(-1)

    max_sf = float(scene_get(scene, "maxfreqres", "cpd"))
    n_steps = int(min(max(np.ceil(max_sf), 1.0), 70.0))
    sample_sf = np.linspace(0.0, max_sf, n_steps, dtype=float)

    oi_depths: list[OpticalImage] = [oi.clone() for _ in range(depth_centers.size)]
    for depth_index in range(depth_centers.size - 1, -1, -1):
        defocus = chromatic_aberration + defocus_diopters[depth_index]
        otf_rows, sample_sf_mm = optics_defocus_core(optics, sample_sf, defocus)
        current_optics = optics_build_2d_otf(optics, otf_rows, sample_sf_mm)
        current_oi = oi_set(oi.clone(), "optics", current_optics)

        current_range = np.array([edges[depth_index], edges[depth_index + 1]], dtype=float)
        if np.isclose(current_range[0], current_range[1]):
            scene_depth = scene.clone()
        else:
            scene_depth, _ = _scene_depth_range(scene, current_range)
        oi_depths[depth_index] = oi_compute(current_oi, scene_depth)

    combined = oi_combine_depths(oi_depths) if len(oi_depths) > 1 else oi_depths[0]
    oi_calculate_illuminance(combined)
    return combined, oi_depths, defocus_diopters


def oi_depth_compute(
    oi: OpticalImage,
    scene: Scene,
    image_dist: Any | None = None,
    depth_edges: Any | None = None,
    c_aberration: Any | None = None,
    display_flag: Any = 1,
) -> tuple[list[OpticalImage], np.ndarray]:
    """Compute one defocused OI per requested scene depth."""

    del display_flag
    if oi is None:
        raise ValueError("oi required")
    if scene is None:
        raise ValueError("scene required")
    if depth_edges is None:
        raise ValueError("depthEdges required")

    if image_dist is None:
        image_distance = float(optics_get(oi_get(oi, "optics"), "focal length", "m"))
    else:
        image_distance = float(np.asarray(image_dist, dtype=float).reshape(-1)[0])

    original_depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
    oi_depths: list[OpticalImage] = []
    last_defocus = np.empty(0, dtype=float)

    from .scene import scene_set

    for edge in np.asarray(depth_edges, dtype=float).reshape(-1):
        current_scene = scene.clone()
        current_scene = scene_set(current_scene, "depth map", np.ones_like(original_depth_map, dtype=float) * float(edge))
        current_oi, _, current_defocus = s3d_render_depth_defocus(current_scene, oi, image_distance, None, c_aberration)
        oi_depths.append(oi_set(current_oi, "name", f"Defocus {float(np.asarray(current_defocus).reshape(-1)[0]):.2f}"))
        last_defocus = np.asarray(current_defocus, dtype=float).reshape(-1)

    return oi_depths, last_defocus


oiDepthEdges = oi_depth_edges
s3dRenderDepthDefocus = s3d_render_depth_defocus
oiDepthCompute = oi_depth_compute


def optics_dl_compute(
    scene: Scene,
    oi: OpticalImage | None = None,
    *args: Any,
    session: SessionContext | None = None,
    **kwargs: Any,
) -> OpticalImage:
    """Run the legacy diffraction-limited optics entry point through ``oi_compute``."""

    current = oi_create("diffraction limited", session=session) if oi is None else oi
    model = param_format(current.fields.get("optics", {}).get("model", ""))
    if model not in {"diffractionlimited", "skip"}:
        raise ValueError("opticsDLCompute requires a diffraction-limited or skip optics model.")
    return oi_compute(current, scene, *args, session=session, **kwargs)


def optics_si_compute(
    scene: Scene,
    oi: OpticalImage | None = None,
    *args: Any,
    session: SessionContext | None = None,
    **kwargs: Any,
) -> OpticalImage:
    """Run the legacy shift-invariant optics entry point through ``oi_compute``."""

    current = oi_create("shift invariant", session=session) if oi is None else oi
    model = param_format(current.fields.get("optics", {}).get("model", ""))
    if model != "shiftinvariant":
        raise ValueError("opticsSICompute requires a shift-invariant optics model.")
    return oi_compute(current, scene, *args, session=session, **kwargs)


def optics_plot_transmittance(oi: OpticalImage, this_w: Any | None = None) -> dict[str, Any]:
    """Return MATLAB-style optics transmittance plot payload without opening a figure."""

    del this_w
    wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    if wave.size == 0:
        return {"wave": np.empty(0, dtype=float), "transmittance": np.empty(0, dtype=float)}
    return {
        "wave": wave.copy(),
        "transmittance": np.asarray(oi_get(oi, "transmittance", wave), dtype=float).reshape(-1).copy(),
    }


def oi_illuminant_ss(oi: OpticalImage, pattern: Any | None = None) -> OpticalImage:
    """Convert an OI illuminant to spatial-spectral format using the MATLAB contract."""

    illuminant_format = _oi_illuminant_format(oi)
    if not illuminant_format:
        raise ValueError("No OI illuminant present.")

    current = oi.clone()
    if param_format(illuminant_format) == "spectral":
        rows, cols = _oi_shape(current)
        illuminant = np.asarray(oi_get(current, "illuminant photons"), dtype=float).reshape(1, 1, -1)
        current = oi_set(
            current,
            "illuminant photons",
            np.broadcast_to(illuminant, (rows, cols, illuminant.shape[2])).copy(),
        )
    elif param_format(illuminant_format) != "spatialspectral":
        raise ValueError(f"Unknown illuminant format: {illuminant_format}")

    if pattern is None or np.asarray(pattern).size == 0:
        return current
    return oi_illuminant_pattern(current, pattern)


def oi_illuminant_pattern(oi: OpticalImage, pattern: Any) -> OpticalImage:
    """Apply a spatial illuminant pattern to both OI photons and illuminant photons."""

    current = oi.clone() if param_format(_oi_illuminant_format(oi)) == "spatialspectral" else oi_illuminant_ss(oi)
    rows, cols = _oi_shape(current)
    pattern_array = _resize_oi_pattern(pattern, (rows, cols))
    photons = np.asarray(oi_get(current, "photons"), dtype=float) * pattern_array[:, :, None]
    illuminant = np.asarray(oi_get(current, "illuminant photons"), dtype=float) * pattern_array[:, :, None]
    current = oi_set(current, "photons", photons)
    current = oi_set(current, "illuminant photons", illuminant)
    return current


def oi_photon_noise(oi: OpticalImage | Any, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Add MATLAB-style photon noise to an OI photon cube or a raw array."""

    if isinstance(oi, OpticalImage):
        photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    else:
        photons = np.asarray(oi, dtype=float)

    rng = np.random.default_rng(None if seed is None else int(seed))
    the_noise = np.sqrt(np.clip(photons, 0.0, None)) * rng.standard_normal(photons.shape)
    noisy_photons = np.rint(photons + the_noise)

    poisson_mask = photons < 15.0
    if np.any(poisson_mask):
        poisson_samples = rng.poisson(np.clip(photons[poisson_mask], 0.0, None))
        the_noise = np.asarray(the_noise, dtype=float)
        the_noise[poisson_mask] = np.asarray(poisson_samples, dtype=float)
        noisy_photons[poisson_mask] = np.asarray(poisson_samples, dtype=float)

    return np.asarray(noisy_photons, dtype=float), np.asarray(the_noise, dtype=float)


oiCalculateIrradiance = oi_calculate_irradiance
oiAdjustIlluminance = oi_adjust_illuminance
oiInterpolateW = oi_interpolate_w
oiExtractWaveband = oi_extract_waveband
oiAdd = oi_add
opticsDLCompute = optics_dl_compute
opticsSICompute = optics_si_compute
opticsPlotTransmittance = optics_plot_transmittance
oiIlluminantSS = oi_illuminant_ss
oiIlluminantPattern = oi_illuminant_pattern
oiPhotonNoise = oi_photon_noise
oiPadValue = oi_pad_value
oiPad = oi_pad
oiMakeEvenRowCol = oi_make_even_row_col
opticsDefocusDepth = optics_defocus_depth


def _oi_photon_cube(oi: OpticalImage) -> tuple[np.ndarray | None, np.ndarray]:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if photons.ndim == 2 and wave.size == 1:
        photons = photons[:, :, np.newaxis]
    if photons.ndim != 3 or photons.size == 0 or wave.size == 0:
        return None, wave
    return photons, wave


def _oi_rgb_render(oi: OpticalImage, *, asset_store: AssetStore | None = None) -> np.ndarray | None:
    store = _store(asset_store)
    photons, wave = _oi_photon_cube(oi)
    if photons is None:
        return None
    energy = quanta_to_energy(photons, wave)
    xyz = xyz_from_energy(energy, wave, asset_store=store)
    from .utils import xyz_to_srgb

    return np.asarray(xyz_to_srgb(xyz), dtype=float)


def oi_show_image(
    oi: OpticalImage,
    render_flag: int = 1,
    gam: float = 1.0,
    oi_w: Any | None = None,
    title_string: str | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray | None:
    del oi_w, title_string
    method = abs(int(render_flag))
    clip_level = 90.0 if method == 5 else 99.5
    if method == 5:
        method = 4

    if method in {0, 1}:
        rgb = _oi_rgb_render(oi, asset_store=asset_store)
    elif method == 2:
        photons, _ = _oi_photon_cube(oi)
        if photons is None:
            return None
        gray = np.mean(photons, axis=2, dtype=float)
        gray_min = float(np.min(gray))
        gray_max = float(np.max(gray))
        if gray_max > gray_min:
            gray = (gray - gray_min) / (gray_max - gray_min)
        else:
            gray = np.zeros_like(gray, dtype=float)
        rgb = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    elif method == 3:
        from .scene import hdr_render

        base = _oi_rgb_render(oi, asset_store=asset_store)
        rgb = None if base is None else np.asarray(hdr_render(base), dtype=float)
    elif method == 4:
        from .scene import hdr_render
        from .utils import xyz_to_srgb

        photons, wave = _oi_photon_cube(oi)
        if photons is None:
            return None
        energy = quanta_to_energy(photons, wave)
        xyz = np.asarray(xyz_from_energy(energy, wave, asset_store=_store(asset_store)), dtype=float)
        y_channel = xyz[:, :, 1]
        y_clip = float(np.percentile(y_channel, clip_level))
        rgb = np.asarray(hdr_render(xyz_to_srgb(np.clip(xyz, 0.0, y_clip))), dtype=float)
    else:
        raise UnsupportedOptionError(f"oiShowImage renderFlag={render_flag} is not supported.")

    if rgb is None:
        return None
    if float(gam) != 1.0:
        rgb = np.power(np.clip(np.asarray(rgb, dtype=float), 0.0, None), float(gam))
    return np.asarray(rgb, dtype=float)


def oi_save_image(
    oi: OpticalImage,
    f_name: str | Path,
    *,
    asset_store: AssetStore | None = None,
) -> str:
    """Save the current optical-image rendering to an 8-bit PNG file."""

    rgb = oi_show_image(oi, -1, 1.0, asset_store=asset_store)
    if rgb is None:
        raise ValueError("Optical image has no computed photon data to save.")

    output_path = Path(f_name).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.clip(np.round(np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
    iio.imwrite(output_path, payload)
    return str(output_path)


def _normalize_waveband_scene(scene: Scene) -> Scene:
    current = scene.clone()
    current.fields["wave"] = np.asarray(scene_get(current, "wave"), dtype=float).reshape(-1)

    photons = np.asarray(scene_get(current, "photons"), dtype=float)
    if photons.ndim == 2:
        current.data["photons"] = photons[:, :, np.newaxis]

    for field_name in ("illuminant_photons", "illuminant_energy"):
        if field_name not in current.fields:
            continue
        field_value = np.asarray(current.fields[field_name], dtype=float)
        if field_value.ndim == 0:
            current.fields[field_name] = field_value.reshape(1)
        elif field_value.ndim == 2:
            current.fields[field_name] = field_value[:, :, np.newaxis]
    return current


def oi_wb_compute(
    work_dir: str | Path,
    oi: OpticalImage | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> str:
    """Convert a directory of `sceneXXX.mat` files into `oiXXX.mat` files."""

    from .fileio import vc_load_object, vc_save_object

    directory = Path(work_dir).expanduser()
    if not directory.exists():
        raise ValueError(f"Scene waveband directory does not exist: {directory}")

    scene_files = sorted(directory.glob("scene*.mat"))
    if not scene_files:
        raise ValueError(f"No scene waveband files found in {directory}.")

    current_oi = oi_create(asset_store=_store(asset_store)) if oi is None else oi
    for scene_file in scene_files:
        loaded_scene, _ = vc_load_object("scene", scene_file)
        if not isinstance(loaded_scene, Scene):
            raise ValueError(f"{scene_file} does not contain a scene object.")
        current_scene = _normalize_waveband_scene(loaded_scene)
        computed = oi_compute(current_oi.clone(), current_scene)
        wavelength = float(np.asarray(scene_get(current_scene, "wave"), dtype=float).reshape(-1)[0])
        vc_save_object(computed, directory / f"oi{int(np.rint(wavelength))}.mat")
    return str(directory)


def oi_calculate_otf(
    oi: OpticalImage,
    wave: Any | None = None,
    unit: str = "cyclesPerDegree",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current shift-invariant OTF on the OI frequency support."""

    normalized_model = param_format(oi_get(oi, "model"))
    normalized_unit = param_format(unit)
    support = np.asarray(oi_get(oi, "frequency support", unit), dtype=float)
    target_wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1) if wave is None else np.asarray(wave, dtype=float).reshape(-1)
    if target_wave.size == 0:
        target_wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)

    if normalized_model == "raytrace":
        raise UnsupportedOptionError("oiCalculateOTF", "raytrace")

    if normalized_model in {"diffractionlimited", "dlmtf"}:
        optics = dict(oi.fields.get("optics", {}))
        fx = np.asarray(support[:, :, 0], dtype=float)
        fy = np.asarray(support[:, :, 1], dtype=float)
        rho = np.sqrt(fx**2 + fy**2)
        aperture_diameter = float(optics["focal_length_m"]) / max(float(optics["f_number"]), 1.0e-12)
        focal_plane_distance = float(optics["focal_length_m"])
        cutoff = (aperture_diameter / max(focal_plane_distance, 1.0e-12)) / np.maximum(target_wave * 1.0e-9, 1.0e-12)
        if normalized_unit in {"cyclesperdegree", "cycperdeg"}:
            cutoff = cutoff * float(oi_get(oi, "distance per degree", "meters"))
        else:
            unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1.0e3, "mm": 1.0e3, "microns": 1.0e6, "um": 1.0e6}
            if normalized_unit not in unit_scale:
                raise KeyError(f"Unsupported oiCalculateOTF unit: {unit}")
            cutoff = cutoff / unit_scale[normalized_unit]

        otf = np.empty((support.shape[0], support.shape[1], target_wave.size), dtype=complex)
        for band_index, cutoff_frequency in enumerate(cutoff):
            normalized_rho = rho / max(float(cutoff_frequency), 1.0e-12)
            clipped = np.clip(normalized_rho, 0.0, 1.0)
            plane = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
            plane[normalized_rho >= 1.0] = 0.0
            otf[:, :, band_index] = np.fft.ifftshift(np.asarray(plane, dtype=float))
    else:
        otf_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        if otf_bundle is None:
            raise ValueError("Optical image does not provide a shift-invariant OTF.")

        source_otf = np.asarray(otf_bundle["OTF"], dtype=complex)
        source_wave = np.asarray(otf_bundle["wave"], dtype=float).reshape(-1)
        source_fx = np.asarray(otf_bundle["fx"], dtype=float).reshape(-1)
        source_fy = np.asarray(otf_bundle["fy"], dtype=float).reshape(-1)
        query_fx = np.asarray(support[0, :, 0], dtype=float).reshape(-1)
        query_fy = np.asarray(support[:, 0, 1], dtype=float).reshape(-1)
        if normalized_unit in {"cyclesperdegree", "cycperdeg"}:
            deg_per_mm = float(oi_get(oi, "degrees per distance", "mm"))
            query_fx_mm = query_fx * deg_per_mm
            query_fy_mm = query_fy * deg_per_mm
        elif normalized_unit in {"meters", "m"}:
            query_fx_mm = query_fx / 1.0e3
            query_fy_mm = query_fy / 1.0e3
        elif normalized_unit in {"millimeters", "mm"}:
            query_fx_mm = query_fx
            query_fy_mm = query_fy
        elif normalized_unit in {"microns", "um"}:
            query_fx_mm = query_fx * 1.0e3
            query_fy_mm = query_fy * 1.0e3
        else:
            raise KeyError(f"Unsupported oiCalculateOTF unit: {unit}")

        otf = np.empty((support.shape[0], support.shape[1], target_wave.size), dtype=complex)
        for band_index, wavelength_nm in enumerate(target_wave):
            plane = _stack_plane_at_wavelength(source_otf, source_wave, float(wavelength_nm))
            otf[:, :, band_index] = _resample_complex_otf_on_support(
                plane,
                source_fx,
                source_fy,
                query_fx_mm,
                query_fy_mm,
            )

    if target_wave.size == 1:
        otf = np.asarray(otf[:, :, 0], dtype=complex)

    return otf, support


def oi_custom_compute(oi: OpticalImage | None = None) -> tuple[bool, str | None]:
    """Return whether a non-standard custom OI compute method is selected."""

    current = oi_create() if oi is None else oi
    method = oi.fields.get("custom_compute_method", oi_get(current, "compute method"))
    method_text = "" if method is None else str(method)
    normalized = param_format(method_text)
    is_custom = normalized not in {"", "opticspsf", "opticsotf", "humanmw", "skip"}
    return bool(is_custom), (method_text if is_custom else None)


def _load_oi_preview_sequence(input_oi_path: Any) -> list[OpticalImage]:
    from .fileio import _deserialize_value, _reconstruct_object, vc_load_object

    if isinstance(input_oi_path, (list, tuple)):
        sequence = [item for item in input_oi_path if isinstance(item, OpticalImage)]
        if not sequence:
            raise ValueError("oiPreviewVideo input list must contain optical images.")
        return sequence

    path = Path(input_oi_path).expanduser()
    if path.is_dir():
        files = sorted(path.glob("oi*.mat"))
        if not files:
            raise ValueError(f"No oi*.mat files found in {path}.")
        return [vc_load_object("oi", file_path)[0] for file_path in files]

    try:
        loaded, _ = vc_load_object("oi", path)
        if isinstance(loaded, OpticalImage):
            return [loaded]
    except Exception:
        pass

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    scenes_to_save = payload.get("scenesToSave")
    if scenes_to_save is None:
        raise ValueError(f"Unable to load an OI preview sequence from {path}.")

    rebuilt = _reconstruct_object(_deserialize_value(scenes_to_save))
    if isinstance(rebuilt, OpticalImage):
        return [rebuilt]
    if isinstance(rebuilt, list):
        sequence = [item for item in rebuilt if isinstance(item, OpticalImage)]
        if sequence:
            return sequence
    raise ValueError(f"{path} does not contain a supported optical-image preview sequence.")


def oi_preview_video(
    input_oi_path: Any,
    output_name: str | Path | None = None,
    render_flag: int = -1,
    gam: float = 1.0,
    fps: float = 3.0,
    *,
    asset_store: AssetStore | None = None,
) -> str:
    """Render a headless preview animation from an OI file, folder, or sequence."""

    oi_sequence = _load_oi_preview_sequence(input_oi_path)
    if output_name is None:
        if isinstance(input_oi_path, (str, Path)):
            input_path = Path(input_oi_path).expanduser()
            base = input_path / "oi-preview.gif" if input_path.is_dir() else input_path.with_suffix(".gif")
        else:
            base = Path.cwd() / "oi-preview.gif"
        output_path = base
    else:
        output_path = Path(output_name).expanduser()

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".gif")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_frames: list[np.ndarray] = []
    for current_oi in oi_sequence:
        rendered = oi_show_image(current_oi, render_flag, gam, asset_store=asset_store)
        if rendered is None:
            raise ValueError("oiPreviewVideo encountered an OI without renderable photon data.")
        rendered_frames.append(np.asarray(rendered, dtype=float))

    max_value = max(float(np.max(frame)) for frame in rendered_frames) if rendered_frames else 0.0
    scale = 1.0 if max_value <= 0.0 else max_value
    frames = [
        np.clip(np.rint(np.clip(frame / scale, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
        for frame in rendered_frames
    ]

    stack = np.stack(frames, axis=0)
    if output_path.suffix.lower() == ".gif":
        iio.imwrite(output_path, stack, duration=1.0 / max(float(fps), 1.0e-12), loop=0)
    else:
        iio.imwrite(output_path, stack, fps=float(fps))
    return str(output_path)


def oi_extract_bright(oi: OpticalImage) -> OpticalImage:
    """Extract the brightest optical-image patch as a small skip-optics OI."""

    illuminance = oi_get(oi, "illuminance")
    if illuminance is None:
        oi_calculate_illuminance(oi)
        illuminance = oi_get(oi, "illuminance")
    illuminance_array = np.asarray(illuminance, dtype=float)
    row_index, col_index = np.unravel_index(int(np.argmax(illuminance_array)), illuminance_array.shape)
    extracted = oi_crop(oi, np.array([col_index + 1, row_index + 1, 1, 1], dtype=int))
    extracted = oi_set(extracted, "compute method", "skip")
    extracted = oi_set(extracted, "off axis method", "skip")
    return extracted


def oi_from_file(
    image_data: Any,
    image_type: str,
    mean_luminance: Any | None = None,
    disp_cal: Any | None = None,
    w_list: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> OpticalImage:
    """Create a ray-trace optical image from the existing scene-from-file path."""

    from .display import display_create
    from .scene import scene_adjust_illuminant, scene_from_file

    store = _store(asset_store)
    display = display_create("LCD-Apple", asset_store=store) if disp_cal is None else disp_cal
    wave = DEFAULT_WAVE.copy() if w_list is None else np.asarray(w_list, dtype=float).reshape(-1)
    scene = scene_from_file(image_data, image_type, mean_luminance, display, wave, asset_store=store)
    scene = scene_adjust_illuminant(scene, "D65", asset_store=store)

    oi = oi_create("ray trace", asset_store=store)
    oi = oi_set(oi, "wave", np.asarray(scene_get(scene, "wave"), dtype=float))
    oi = oi_set(oi, "fov", float(scene_get(scene, "fov")))
    return oi_set(oi, "photons", np.asarray(scene_get(scene, "photons"), dtype=float) / np.pi)


oiExtractBright = oi_extract_bright
oiFromFile = oi_from_file


def oi_frequency_resolution(
    oi: OpticalImage,
    units: str = "cyclesPerDegree",
) -> dict[str, np.ndarray]:
    support = oi_get(oi, "frequency resolution", units)
    return {
        "fx": np.asarray(support["fx"], dtype=float).copy(),
        "fy": np.asarray(support["fy"], dtype=float).copy(),
    }


def oi_spatial_support(
    oi: OpticalImage,
    units: str = "meters",
) -> dict[str, np.ndarray]:
    support = oi_get(oi, "spatial support linear", units)
    return {
        "x": np.asarray(support["x"], dtype=float).copy(),
        "y": np.asarray(support["y"], dtype=float).copy(),
    }


def oi_space(
    oi: OpticalImage,
    s_pos: Any,
    unit: str = "mm",
) -> np.ndarray:
    sample_position = np.asarray(s_pos, dtype=float).reshape(-1)
    if sample_position.size != 2:
        raise ValueError("oi_space expects [row, col] sample positions.")
    distance_per_sample = np.asarray(oi_get(oi, "distance per sample", unit), dtype=float).reshape(-1)
    size = np.asarray(oi_get(oi, "size"), dtype=float).reshape(-1)
    middle = size / 2.0
    return np.array(
        [
            (middle[0] - sample_position[0]) * distance_per_sample[0],
            (sample_position[1] - middle[1]) * distance_per_sample[1],
        ],
        dtype=float,
    )


def _oi_roi_xyz(oi: OpticalImage, roi_locs: Any | None = None, *, asset_store: AssetStore | None = None) -> np.ndarray:
    store = _store(asset_store)
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if roi_locs is None:
        photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
        if photons.ndim != 3 or wave.size == 0:
            return np.empty((0, 3), dtype=float)
        photons = photons.reshape(-1, wave.size)
    else:
        photons = np.asarray(oi_get(oi, "roi photons", roi_locs), dtype=float)
    energy = quanta_to_energy(photons, wave)
    return xyz_from_energy(energy, wave, asset_store=store)


def _oi_line_index(line_arg: Any, orientation: str) -> int:
    values = np.asarray(line_arg, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Line location must be a scalar or [col, row] locator.")
    if values.size == 1:
        return int(np.rint(values[0]))
    return int(np.rint(values[1 if orientation == "h" else 0]))


def _oi_line_profile(
    oi: OpticalImage,
    data_type: str,
    orientation: str,
    line_arg: Any,
    *,
    unit: Any | None = "um",
) -> dict[str, np.ndarray]:
    wave = np.asarray(oi.fields.get("wave", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    support = oi_get(oi, "spatial support linear", unit)
    line_index = _oi_line_index(line_arg, orientation)
    if data_type == "photons":
        data = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    elif data_type == "energy":
        data = quanta_to_energy(np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float), wave)
    elif data_type == "illuminance":
        data = np.asarray(oi_get(oi, "illuminance"), dtype=float)
    else:
        raise KeyError(f"Unsupported oi line profile data type: {data_type}")

    if orientation == "h":
        if line_index < 1 or line_index > data.shape[0]:
            raise IndexError("Horizontal optical-image line index is out of range.")
        pos = np.asarray(support["x"], dtype=float)
        line = np.asarray(data[line_index - 1, ...], dtype=float)
    else:
        if line_index < 1 or line_index > data.shape[1]:
            raise IndexError("Vertical optical-image line index is out of range.")
        pos = np.asarray(support["y"], dtype=float)
        line = np.asarray(data[:, line_index - 1, ...], dtype=float)

    if line.ndim == 1:
        return {"pos": pos.copy(), "data": line.copy(), "unit": str(unit or "m")}
    return {"pos": pos.copy(), "wave": wave.copy(), "data": line.T.copy(), "unit": str(unit or "m")}


def _gaussian_kernel_1d(size: int, sigma_pixels: float) -> np.ndarray:
    if size <= 1 or sigma_pixels <= 0.0:
        return np.ones((1,), dtype=float)
    center = (size - 1) / 2.0
    coords = np.arange(size, dtype=float) - center
    kernel = np.exp(-0.5 * (coords / max(sigma_pixels, 1e-12)) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.ones((1,), dtype=float)
    return kernel / kernel_sum


def _diffuser_kernel_shape(sigma_pixels: float, limit: int) -> int:
    if sigma_pixels <= 0.0:
        return 1
    size = int(np.ceil(8.0 * sigma_pixels))
    if limit > 0 and size >= limit:
        size = int(limit)
    if size <= 0:
        size = 1
    if size % 2 == 0:
        size += 1
    return size


def oi_diffuser(
    oi: OpticalImage,
    sd_um: float | np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> tuple[OpticalImage, float | np.ndarray, np.ndarray]:
    w_spatial_res_um = float(oi_get(oi, "wspatialresolution")) * _spatial_unit_scale("um")
    if sd_um is None:
        sd_array = np.array([w_spatial_res_um * (1.4427 / 2.0)], dtype=float)
    else:
        sd_array = np.asarray(sd_um, dtype=float).reshape(-1)
    if sd_array.size == 0 or sd_array.size > 2:
        raise ValueError("oiDiffuser expects a scalar or a 2-element blur standard deviation in microns.")

    sigma_pixels = sd_array / max(w_spatial_res_um, 1e-12)
    rows, cols = _oi_shape(oi)
    if sigma_pixels.size == 1:
        size = _diffuser_kernel_shape(float(sigma_pixels[0]), rows)
        kernel_1d = _gaussian_kernel_1d(size, float(sigma_pixels[0]))
        blur_filter = np.outer(kernel_1d, kernel_1d)
    else:
        row_size = _diffuser_kernel_shape(float(sigma_pixels[0]), rows)
        col_size = _diffuser_kernel_shape(float(sigma_pixels[1]), cols)
        row_kernel = _gaussian_kernel_1d(row_size, float(sigma_pixels[0]))
        col_kernel = _gaussian_kernel_1d(col_size, float(sigma_pixels[1]))
        blur_filter = row_kernel[:, None] * col_kernel[None, :]
    blur_sum = float(np.sum(blur_filter))
    if blur_sum > 0.0:
        blur_filter = blur_filter / blur_sum

    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0), dtype=float)), dtype=float)
    if photons.ndim != 3 or photons.size == 0:
        oi.fields["illuminance"] = np.empty(photons.shape[:2], dtype=float) if photons.ndim >= 2 else np.empty((0, 0), dtype=float)
        oi.fields["mean_illuminance"] = 0.0
        oi.fields["mean_comp_illuminance"] = 0.0
        returned_sd: np.ndarray | float = float(sd_array[0]) if sd_array.size == 1 else sd_array.copy()
        return oi, returned_sd, blur_filter

    filtered = np.empty_like(photons, dtype=float)
    for wave_index in range(photons.shape[2]):
        filtered[:, :, wave_index] = fftconvolve(photons[:, :, wave_index], blur_filter, mode="same")
    oi.data["photons"] = filtered
    oi_calculate_illuminance(oi)
    returned_sd = float(sd_array[0]) if sd_array.size == 1 else sd_array.copy()
    return oi, returned_sd, blur_filter


def oi_birefringent_diffuser(
    oi: OpticalImage | None = None,
    um_disp: Any | None = None,
) -> tuple[OpticalImage, float]:
    """Apply the legacy birefringent anti-alias filter to an OI photon cube."""

    current = oi.clone() if oi is not None else oi_create()
    photons, _ = _oi_photon_cube(current)

    if um_disp is None:
        default_disp_um = float(oi_get(current, "wspatialresolution")) * _spatial_unit_scale("um") / 2.0
        displacement_um = 1.0 if default_disp_um <= 0.0 else default_disp_um
    else:
        displacement_um = float(np.asarray(um_disp, dtype=float).reshape(-1)[0])

    if photons is None:
        rows, cols = _oi_shape(current)
        current.fields["illuminance"] = np.empty((rows, cols), dtype=float)
        current.fields["mean_illuminance"] = 0.0
        current.fields["mean_comp_illuminance"] = 0.0
        return current, displacement_um

    support = oi_get(current, "spatial support linear", "um")
    x_coords = np.asarray(support["x"], dtype=float).reshape(-1)
    y_coords = np.asarray(support["y"], dtype=float).reshape(-1)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    offsets = (
        (-displacement_um, -displacement_um),
        (-displacement_um, displacement_um),
        (displacement_um, -displacement_um),
        (displacement_um, displacement_um),
    )

    filtered = np.empty_like(photons, dtype=float)
    for wave_index in range(photons.shape[2]):
        interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            np.asarray(photons[:, :, wave_index], dtype=float),
            bounds_error=False,
            fill_value=0.0,
        )
        plane = np.zeros(photons.shape[:2], dtype=float)
        for dy_um, dx_um in offsets:
            sample_points = np.stack((yy + dy_um, xx + dx_um), axis=-1)
            plane += np.asarray(interpolator(sample_points), dtype=float)
        filtered[:, :, wave_index] = plane / 4.0

    current.data["photons"] = filtered
    oi_calculate_illuminance(current)
    return current, displacement_um


def _normalize_camera_motion_amounts(value: Any) -> list[np.ndarray]:
    amount_array = np.asarray(value, dtype=float)
    if amount_array.size == 0:
        return []
    if amount_array.ndim == 1:
        if amount_array.size != 2:
            raise ValueError("oiCameraMotion expects a 2-element shift or an array of [row, col] shifts.")
        amount_array = amount_array.reshape(1, 2)
    if amount_array.shape[-1] != 2:
        raise ValueError("oiCameraMotion amount entries must contain [row, col] motion pairs.")
    return [np.asarray(pair, dtype=float).reshape(2) for pair in amount_array.reshape(-1, 2)]


def oi_camera_motion(
    oi: OpticalImage | None = None,
    options: dict[str, Any] | None = None,
    *,
    amount: Any | None = None,
    focal_length: float | None = None,
) -> OpticalImage:
    """Replay the legacy depth-map camera-motion burst helper on an OI."""

    current = oi.clone() if oi is not None else oi_create()
    photons, _ = _oi_photon_cube(current)
    if photons is None:
        return current

    params: dict[str, Any] = {
        "amount": ((0.0, 0.1), (0.0, 0.2)),
        "focallength": 0.004,
    }
    if options:
        for key, value in options.items():
            normalized = param_format(key)
            if normalized == "amount":
                params["amount"] = value
            elif normalized == "focallength":
                params["focallength"] = float(np.asarray(value, dtype=float).reshape(-1)[0])
    if amount is not None:
        params["amount"] = amount
    if focal_length is not None:
        params["focallength"] = float(focal_length)

    shifts = _normalize_camera_motion_amounts(params["amount"])
    if not shifts:
        return current

    row_spacing_m = float(oi_get(current, "hspatialresolution"))
    col_spacing_m = float(oi_get(current, "wspatialresolution"))
    if row_spacing_m <= 0.0 or col_spacing_m <= 0.0:
        raise ValueError("oiCameraMotion requires an optical image with positive spatial sampling.")

    base_illuminance = current.fields.get("illuminance")
    if base_illuminance is None:
        base_illuminance = oi_get(current, "illuminance")
    base_illuminance_array = np.asarray(base_illuminance, dtype=float)
    if base_illuminance_array.shape != photons.shape[:2]:
        base_illuminance_array = _oi_illuminance(current)

    depth_map = np.asarray(oi_get(current, "depth map"), dtype=float)
    if depth_map.shape != photons.shape[:2]:
        raise ValueError("oiCameraMotion requires a depth map matching the OI size.")

    focal_length_m = float(params["focallength"])
    rows, cols = photons.shape[:2]
    photon_frames = [np.asarray(photons, dtype=float).copy()]
    illuminance_frames = [base_illuminance_array.copy()]

    for shift in shifts:
        row_offset_m = np.divide(
            float(shift[0]) * focal_length_m,
            depth_map,
            out=np.zeros_like(depth_map, dtype=float),
            where=np.isfinite(depth_map) & (depth_map != 0.0),
        )
        col_offset_m = np.divide(
            float(shift[1]) * focal_length_m,
            depth_map,
            out=np.zeros_like(depth_map, dtype=float),
            where=np.isfinite(depth_map) & (depth_map != 0.0),
        )
        row_offset_px = row_offset_m / max(row_spacing_m, 1.0e-30)
        col_offset_px = col_offset_m / max(col_spacing_m, 1.0e-30)
        row_offset_px[~np.isfinite(row_offset_px)] = 0.0
        col_offset_px[~np.isfinite(col_offset_px)] = 0.0

        shifted_photons = np.zeros_like(photons, dtype=float)
        shifted_illuminance = np.zeros_like(base_illuminance_array, dtype=float)
        for row in range(rows):
            for col in range(cols):
                new_row = int(np.floor(row + row_offset_px[row, col]))
                new_col = int(np.floor(col + col_offset_px[row, col]))
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    shifted_photons[new_row, new_col, :] = photons[row, col, :]
                    shifted_illuminance[new_row, new_col] = base_illuminance_array[row, col]

        photon_frames.append(shifted_photons)
        illuminance_frames.append(shifted_illuminance)

    current.data["photons"] = np.stack(photon_frames, axis=3)
    current.fields["illuminance"] = np.stack(illuminance_frames, axis=2)
    current.fields.pop("mean_illuminance", None)
    current.fields.pop("mean_comp_illuminance", None)
    return current


def optics_ray_trace(
    scene: Scene,
    oi: OpticalImage,
    angle_step_deg: float | None = None,
) -> OpticalImage:
    if param_format(oi.fields.get("optics", {}).get("model", "")) != "raytrace":
        raise ValueError("opticsRayTrace requires a ray-trace optical image.")

    _, compute_scene = _prepare_raytrace_scene(oi, scene)
    staged = oi.clone()
    staged.name = compute_scene.name
    staged.fields["wave"] = np.asarray(compute_scene.fields["wave"], dtype=float).copy()
    staged.fields["fov_deg"] = float(compute_scene.fields.get("fov_deg", staged.fields.get("fov_deg", 10.0)))
    staged.fields["vfov_deg"] = float(
        compute_scene.fields.get("vfov_deg", staged.fields.get("vfov_deg", staged.fields["fov_deg"]))
    )
    if angle_step_deg is not None:
        staged = oi_set(staged, "psf angle step", float(angle_step_deg))

    computed = rt_precompute_psf_apply(rt_geometry(staged, compute_scene), angle_step_deg=angle_step_deg)
    diffuser_method = param_format(computed.fields.get("diffuser_method", "skip"))
    if diffuser_method == "blur":
        blur_m = float(computed.fields.get("diffuser_blur_m", 0.0))
        if blur_m > 0.0:
            computed, _, _ = oi_diffuser(computed, blur_m * 1e6)
    elif diffuser_method == "birefringent":
        raise UnsupportedOptionError("opticsRayTrace", "birefringent")
    elif diffuser_method != "skip":
        raise UnsupportedOptionError("opticsRayTrace", computed.fields.get("diffuser_method", diffuser_method))

    oi_calculate_illuminance(computed)
    return computed


def _raytrace_geometry(cube: np.ndarray, wave: np.ndarray, optics: dict[str, Any], scene: Scene) -> np.ndarray:
    raytrace = dict(optics.get("raytrace", {}))
    max_fov = float(raytrace.get("max_fov_deg", np.inf))
    if np.isfinite(max_fov) and (_scene_diagonal_fov_deg(scene) > max_fov + 1e-9):
        raise ValueError("Scene field of view exceeds the loaded ray-trace analysis.")

    rows, cols = cube.shape[:2]
    width_mm = _oi_geometry(optics, scene)[1] * 1e3
    dx_mm = width_mm / max(cols, 1)
    zeropad = 16
    padded_rows = rows + 2 * zeropad
    padded_cols = cols + 2 * zeropad
    row_center = (padded_rows + 1.0) / 2.0
    col_center = (padded_cols + 1.0) / 2.0

    r = (np.arange(padded_rows, dtype=float) + 1.0) - row_center
    c = (np.arange(padded_cols, dtype=float) + 1.0) - col_center
    cc, rr = np.meshgrid(c, r)
    pixdist_units = np.sqrt(cc**2 + rr**2) * dx_mm
    pixang = np.arctan2(cc, rr)

    field_heights = np.asarray(raytrace.get("geometry", {}).get("field_height_mm", np.empty(0, dtype=float)), dtype=float)
    if field_heights.size < 2:
        return np.asarray(cube, dtype=float).copy()

    degree = min(8, max(field_heights.size - 2, 0))
    result = np.empty_like(cube, dtype=float)
    for band_index, wavelength_nm in enumerate(np.asarray(wave, dtype=float).reshape(-1)):
        distorted_height = rt_di_interp(optics, float(wavelength_nm))
        relative_illumination = rt_ri_interp(optics, float(wavelength_nm))
        if distorted_height.size != field_heights.size or relative_illumination.size != field_heights.size:
            result[:, :, band_index] = cube[:, :, band_index]
            continue

        distortion_poly = np.polyfit(distorted_height, field_heights, degree)
        ri_degree = min(2, max(distorted_height.size - 1, 0))
        illumination_poly = np.polyfit(distorted_height, relative_illumination, ri_degree)

        ideal_height = np.polyval(distortion_poly, pixdist_units)
        rel_illum = np.clip(np.polyval(illumination_poly, pixdist_units), 0.0, None)
        pad_r = np.clip(row_center + ideal_height * np.cos(pixang) / max(dx_mm, 1e-12), 1.0, float(padded_rows)) - 1.0
        pad_c = np.clip(col_center + ideal_height * np.sin(pixang) / max(dx_mm, 1e-12), 1.0, float(padded_cols)) - 1.0

        padded = np.pad(np.asarray(cube[:, :, band_index], dtype=float), ((zeropad, zeropad), (zeropad, zeropad)))
        distorted = map_coordinates(padded, [pad_r, pad_c], order=1, mode="nearest", prefilter=False)
        result[:, :, band_index] = distorted[zeropad:-zeropad, zeropad:-zeropad] * rel_illum[zeropad:-zeropad, zeropad:-zeropad]

    return result


def _raytrace_psf_support_axes(psf_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    psf_function = np.asarray(psf_data.get("function", np.empty(0, dtype=float)), dtype=float)
    if psf_function.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    sample_spacing = np.asarray(psf_data.get("sample_spacing_mm", np.array([0.0, 0.0], dtype=float)), dtype=float).reshape(-1)
    if sample_spacing.size == 1:
        sample_spacing = np.repeat(sample_spacing, 2)
    rows, cols = psf_function.shape[:2]
    y_support = np.arange((-rows / 2.0) + 1.0, (rows / 2.0) + 1.0, dtype=float) * float(sample_spacing[0])
    x_support = np.arange((-cols / 2.0) + 1.0, (cols / 2.0) + 1.0, dtype=float) * float(sample_spacing[1])
    return x_support, y_support


def _raytrace_sampling_axes(rows: int, cols: int, sample_spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(
        -(cols * sample_spacing_mm) / 2.0 + sample_spacing_mm / 2.0,
        (cols * sample_spacing_mm) / 2.0 - sample_spacing_mm / 2.0,
        cols,
    )
    y = np.linspace(
        -(rows * sample_spacing_mm) / 2.0 + sample_spacing_mm / 2.0,
        (rows * sample_spacing_mm) / 2.0 - sample_spacing_mm / 2.0,
        rows,
    )
    return x, y


def _raytrace_data_angle_height(rows: int, cols: int, sample_spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    x_support, y_support = _raytrace_sampling_axes(rows, cols, sample_spacing_mm)
    xx, yy = np.meshgrid(x_support, y_support[::-1])
    data_angle = np.rint(np.rad2deg(np.arctan2(yy, xx) + np.pi)).astype(int)
    data_angle[data_angle == 0] = 1
    data_angle[data_angle > 360] = 360
    data_height = np.sqrt(xx**2 + yy**2)
    return data_angle, data_height


def _raytrace_sample_heights(field_heights_mm: np.ndarray, data_height_mm: np.ndarray) -> np.ndarray:
    heights = np.asarray(field_heights_mm, dtype=float).reshape(-1)
    if heights.size == 0:
        return heights
    max_height = float(np.max(np.asarray(data_height_mm, dtype=float)))
    keep = np.zeros(heights.size, dtype=bool)
    for index, value in enumerate(heights):
        keep[index] = True
        if float(value) > max_height:
            break
    return heights[keep]


def _raytrace_psf_grid(sample_spacing_mm: float, psf_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source_x, source_y = _raytrace_psf_support_axes(psf_data)
    if source_x.size == 0 or source_y.size == 0:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)
    x_positive = np.arange(0.0, float(source_x[-1]) + sample_spacing_mm * 0.5, sample_spacing_mm)
    x_negative = -np.flip(np.arange(0.0, abs(float(source_x[0])) + sample_spacing_mm * 0.5, sample_spacing_mm))
    y_positive = np.arange(0.0, float(source_y[-1]) + sample_spacing_mm * 0.5, sample_spacing_mm)
    y_negative = -np.flip(np.arange(0.0, abs(float(source_y[0])) + sample_spacing_mm * 0.5, sample_spacing_mm))
    x_grid = np.concatenate((x_negative[:-1], x_positive))
    y_grid = np.concatenate((y_negative[:-1], y_positive))
    return x_grid, y_grid


def _raytrace_resample_psf(
    psf: np.ndarray,
    source_x_mm: np.ndarray,
    source_y_mm: np.ndarray,
    target_x_mm: np.ndarray,
    target_y_mm: np.ndarray,
) -> np.ndarray:
    kernel = _resample_plane_on_support(
        np.asarray(psf, dtype=float),
        np.asarray(source_x_mm, dtype=float),
        np.asarray(source_y_mm, dtype=float),
        np.asarray(target_x_mm, dtype=float),
        np.asarray(target_y_mm, dtype=float),
        method="linear",
    )
    kernel = np.clip(np.asarray(kernel, dtype=float), 0.0, None)
    total = float(np.sum(kernel))
    if total <= 0.0:
        kernel = np.zeros_like(kernel, dtype=float)
        kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
        return kernel
    return kernel / total


def _raytrace_angle_lut(sample_angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_angles = np.asarray(sample_angles_deg, dtype=float).reshape(-1)
    if sample_angles.size < 2:
        return np.zeros(360, dtype=int), np.ones(360, dtype=float)
    angle_step = float(sample_angles[1] - sample_angles[0])
    lower_index = np.zeros(360, dtype=int)
    weight = np.zeros(360, dtype=float)
    for degree in range(1, 361):
        nearest = int(np.argmin(np.abs(float(degree) - sample_angles)))
        value = float(abs(float(degree) - sample_angles[nearest]))
        if float(degree) <= sample_angles[nearest]:
            nearest = max(nearest - 1, 0)
            value = angle_step - value
        lower_index[degree - 1] = nearest
        weight[degree - 1] = value / max(angle_step, 1e-12)
    return lower_index, weight


def _raytrace_psf_struct_matches(
    psf_struct: dict[str, Any] | None,
    optics: dict[str, Any],
    wave: np.ndarray,
    rows: int,
    cols: int,
    sample_spacing_m: float,
    sample_angles_deg: np.ndarray,
) -> bool:
    if not isinstance(psf_struct, dict):
        return False
    kernel_shape = _raytrace_psf_kernel_shape(psf_struct)
    if kernel_shape == (0, 0):
        return False
    expected_kernel_shape = _raytrace_expected_kernel_shape(optics, sample_spacing_m)
    if expected_kernel_shape != (0, 0) and kernel_shape != expected_kernel_shape:
        return False
    stored_shape = tuple(psf_struct.get("cube_shape", ()))
    if stored_shape and stored_shape != (rows, cols):
        return False
    stored_spacing = psf_struct.get("sample_spacing_m")
    if stored_spacing is not None and not np.isclose(float(stored_spacing), float(sample_spacing_m)):
        return False
    wavelengths = np.asarray(psf_struct.get("wavelength_nm", np.empty(0)), dtype=float).reshape(-1)
    if wavelengths.size > 0 and not np.array_equal(wavelengths, np.asarray(wave, dtype=float).reshape(-1)):
        return False
    current_angles = np.asarray(psf_struct.get("sample_angles_deg", np.empty(0)), dtype=float).reshape(-1)
    if not np.array_equal(current_angles, np.asarray(sample_angles_deg, dtype=float).reshape(-1)):
        return False
    raytrace = optics.get("raytrace", {})
    optics_name = str(psf_struct.get("optics_name", ""))
    if optics_name and optics_name != str(raytrace.get("name", optics.get("name", ""))):
        return False
    return True


def _raytrace_default_sample_angles(angle_step_deg: float) -> np.ndarray:
    step = float(angle_step_deg)
    if step <= 0.0:
        raise ValueError("PSF angle step must be positive.")
    return np.arange(0.0, 360.0 + step, step, dtype=float)


def _raytrace_requested_sample_angles(
    angle_step_deg: float,
    sample_angles_deg: Any | None,
) -> np.ndarray:
    if sample_angles_deg is None:
        return _raytrace_default_sample_angles(angle_step_deg)
    sample_angles = np.asarray(sample_angles_deg, dtype=float).reshape(-1)
    if sample_angles.size < 2:
        raise ValueError("Ray-trace PSF sample angles must contain at least two entries.")
    if not np.isclose(sample_angles[0], 0.0):
        raise ValueError("Ray-trace PSF sample angles must start at 0 degrees.")
    if sample_angles[-1] < 360.0 and not np.isclose(sample_angles[-1], 360.0):
        raise ValueError("Ray-trace PSF sample angles must include 360 degrees.")
    diffs = np.diff(sample_angles)
    if np.any(diffs <= 0.0):
        raise ValueError("Ray-trace PSF sample angles must be strictly increasing.")
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("Ray-trace PSF sample angles must be uniformly spaced.")
    return sample_angles


def _coerce_psf_stack(value: Any) -> np.ndarray:
    psf = np.asarray(value)
    if psf.dtype != object:
        numeric = np.asarray(value, dtype=float)
        if numeric.ndim != 5:
            raise ValueError("PSF stack must be a 5D array or a 3D object array of 2D kernels.")
        return numeric

    if psf.ndim != 3:
        raise ValueError("MATLAB-style PSF cell arrays must be a 3D object array of 2D kernels.")

    first_kernel: np.ndarray | None = None
    for index in np.ndindex(psf.shape):
        candidate = np.asarray(psf[index], dtype=float)
        if candidate.size == 0:
            continue
        if candidate.ndim != 2:
            raise ValueError("Each PSF kernel must be 2D.")
        first_kernel = candidate
        break
    if first_kernel is None:
        return np.empty(psf.shape + (0, 0), dtype=float)

    stack = np.empty(psf.shape + first_kernel.shape, dtype=float)
    for index in np.ndindex(psf.shape):
        kernel = np.asarray(psf[index], dtype=float)
        if kernel.shape != first_kernel.shape:
            raise ValueError("All PSF kernels must share the same shape.")
        stack[index] = kernel
    return stack


def _coerce_psf_sample_angles(value: Any) -> np.ndarray:
    sample_angles = np.asarray(value, dtype=float).reshape(-1)
    if sample_angles.size < 2:
        raise ValueError("PSF sample angles must contain at least two entries.")
    diffs = np.diff(sample_angles)
    if np.any(diffs <= 0.0):
        raise ValueError("PSF sample angles must be strictly increasing.")
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("PSF sample angles must be uniformly spaced.")
    return sample_angles


def _normalize_psf_struct(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    current = dict(value)
    normalized = dict(current)
    if "psf" in current:
        normalized["psf"] = _coerce_psf_stack(current["psf"])
    sample_angles = current.get("sample_angles_deg", current.get("sampleAngles", current.get("sampAngles")))
    if sample_angles is not None:
        normalized["sample_angles_deg"] = _coerce_psf_sample_angles(sample_angles)
        if normalized["sample_angles_deg"].size >= 2:
            normalized["angle_step_deg"] = float(normalized["sample_angles_deg"][1] - normalized["sample_angles_deg"][0])
    if "img_height_mm" in current:
        normalized["img_height_mm"] = np.asarray(current.get("img_height_mm"), dtype=float).reshape(-1)
    elif "imgHeight" in current:
        # MATLAB stores psfStruct.imgHeight in meters.
        normalized["img_height_mm"] = np.asarray(current.get("imgHeight"), dtype=float).reshape(-1) * 1e3
    wavelength = current.get("wavelength_nm", current.get("wavelength"))
    if wavelength is not None:
        normalized["wavelength_nm"] = np.asarray(wavelength, dtype=float).reshape(-1)
    optics_name = current.get("optics_name", current.get("opticsName"))
    if optics_name is not None:
        normalized["optics_name"] = str(optics_name)
    return normalized


def _raytrace_psf_kernel_shape(psf_struct: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(psf_struct, dict):
        return (0, 0)
    psf = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)), dtype=float)
    if psf.ndim != 5 or psf.shape[3] <= 0 or psf.shape[4] <= 0:
        return (0, 0)
    return int(psf.shape[3]), int(psf.shape[4])


def _raytrace_expected_kernel_shape(optics: dict[str, Any], sample_spacing_m: float) -> tuple[int, int]:
    psf_data = dict(optics.get("raytrace", {}).get("psf", {}))
    source_x_mm, source_y_mm = _raytrace_psf_support_axes(psf_data)
    if source_x_mm.size == 0 or source_y_mm.size == 0:
        return (0, 0)
    target_x_mm, target_y_mm = _raytrace_psf_grid(float(sample_spacing_m) * 1e3, psf_data)
    return int(target_y_mm.size), int(target_x_mm.size)


def _raytrace_finalize_psf_struct(
    psf_struct: dict[str, Any] | None,
    optics: dict[str, Any],
    wave: np.ndarray,
    rows: int,
    cols: int,
    sample_spacing_m: float,
    sample_angles_deg: np.ndarray,
) -> dict[str, Any] | None:
    finalized = _normalize_psf_struct(psf_struct)
    if not isinstance(finalized, dict):
        return None

    sample_angles = np.asarray(finalized.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if sample_angles.size == 0:
        sample_angles = np.asarray(sample_angles_deg, dtype=float).reshape(-1).copy()
        if sample_angles.size > 0:
            finalized["sample_angles_deg"] = sample_angles
    if sample_angles.size >= 2:
        finalized["angle_step_deg"] = float(sample_angles[1] - sample_angles[0])
        if "angle_lut_index" not in finalized or "angle_lut_weight" not in finalized:
            lower_index, angle_weight = _raytrace_angle_lut(sample_angles)
            finalized["angle_lut_index"] = lower_index
            finalized["angle_lut_weight"] = angle_weight

    wavelengths = np.asarray(finalized.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if wavelengths.size == 0:
        finalized["wavelength_nm"] = np.asarray(wave, dtype=float).reshape(-1).copy()

    optics_name = str(optics.get("raytrace", {}).get("name", optics.get("name", "raytrace")))
    if not str(finalized.get("optics_name", "")):
        finalized["optics_name"] = optics_name

    finalized["cube_shape"] = (int(rows), int(cols))
    finalized["sample_spacing_m"] = float(sample_spacing_m)
    return finalized


def _sync_psf_metadata_fields(oi: OpticalImage) -> None:
    psf_struct = oi.fields.get("psf_struct")
    if not isinstance(psf_struct, dict):
        return
    sample_angles = np.asarray(psf_struct.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if sample_angles.size > 0:
        oi.fields["psf_sample_angles_deg"] = sample_angles.copy()
        if sample_angles.size >= 2:
            oi.fields["psf_angle_step_deg"] = float(sample_angles[1] - sample_angles[0])
    img_height_mm = np.asarray(psf_struct.get("img_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if img_height_mm.size > 0:
        oi.fields["psf_image_heights_m"] = img_height_mm / 1e3
    wavelength_nm = np.asarray(psf_struct.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if wavelength_nm.size > 0:
        oi.fields["psf_wavelength_nm"] = wavelength_nm.copy()
    optics_name = psf_struct.get("optics_name")
    if optics_name is not None:
        oi.fields["psf_optics_name"] = str(optics_name)


def _clear_precomputed_psf_state(oi: OpticalImage) -> None:
    oi.fields["psf_struct"] = None
    oi.fields["psf_sample_angles_deg"] = None
    oi.fields["psf_image_heights_m"] = None
    oi.fields["psf_wavelength_nm"] = None
    oi.fields["psf_optics_name"] = None


def _spatial_unit_scale(unit: Any | None) -> float:
    if unit is None:
        return 1.0
    return _SPATIAL_UNIT_SCALE.get(param_format(unit), 1.0)


def _raw_raytrace_psf_data(oi: OpticalImage) -> dict[str, Any]:
    return dict(oi.fields["optics"].get("raytrace", {}).get("psf", {}))


def _raw_raytrace_table(oi: OpticalImage, name: str) -> dict[str, Any]:
    raytrace = dict(oi.fields["optics"].get("raytrace", {}))
    if name == "psf":
        return dict(raytrace.get("psf", {}))
    if name == "geometry":
        return dict(raytrace.get("geometry", {}))
    if name in {"relative_illumination", "relillum"}:
        return dict(raytrace.get("relative_illumination", {}))
    return {}


def _nearest_wave_index(wavelengths_nm: np.ndarray, wavelength_nm: float) -> int:
    if wavelengths_nm.size == 0:
        raise ValueError("No wavelength samples are available.")
    return int(np.argmin(np.abs(np.asarray(wavelengths_nm, dtype=float).reshape(-1) - float(wavelength_nm))))


def _nearest_field_height_index(field_heights_m: np.ndarray, field_height_m: float) -> int:
    heights = np.asarray(field_heights_m, dtype=float).reshape(-1)
    if heights.size == 0:
        raise ValueError("No field-height samples are available.")
    return int(np.argmin(np.abs(heights - float(field_height_m))))


def _raw_raytrace_field_height(table: dict[str, Any], unit: Any | None = None) -> np.ndarray:
    field_height_mm = np.asarray(table.get("field_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    return (field_height_mm / 1e3) * _spatial_unit_scale(unit)


def _raw_raytrace_psf_size(oi: OpticalImage) -> tuple[int, int]:
    psf_function = np.asarray(_raw_raytrace_psf_data(oi).get("function", np.empty(0, dtype=float)), dtype=float)
    if psf_function.ndim < 2:
        return (0, 0)
    return int(psf_function.shape[0]), int(psf_function.shape[1])


def _raw_raytrace_psf_dimensions(oi: OpticalImage) -> tuple[int, ...]:
    psf_function = np.asarray(_raw_raytrace_psf_data(oi).get("function", np.empty(0, dtype=float)), dtype=float)
    if psf_function.ndim == 0:
        return tuple()
    return tuple(int(size) for size in psf_function.shape)


def _raw_raytrace_psf_spacing(oi: OpticalImage, unit: Any | None = None) -> np.ndarray:
    spacing_mm = np.asarray(_raw_raytrace_psf_data(oi).get("sample_spacing_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if spacing_mm.size == 0:
        return np.empty(0, dtype=float)
    return (spacing_mm / 1e3) * _spatial_unit_scale(unit)


def _raw_raytrace_support_axis(length: int, spacing: float) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=float)
    return np.arange((-length / 2.0) + 1.0, (length / 2.0) + 1.0, dtype=float) * float(spacing)


def _raw_raytrace_psf_support_axes(oi: OpticalImage, unit: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _raw_raytrace_psf_size(oi)
    spacing = _raw_raytrace_psf_spacing(oi, unit)
    if spacing.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    y_support = _raw_raytrace_support_axis(rows, float(spacing[0]))
    x_support = _raw_raytrace_support_axis(cols, float(spacing[1]))
    return x_support, y_support


def _raw_raytrace_frequency_axes(oi: OpticalImage, unit: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _raw_raytrace_psf_size(oi)
    spacing = _raw_raytrace_psf_spacing(oi, unit)
    if rows <= 0 or cols <= 0 or spacing.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    fy = _raw_raytrace_support_axis(rows, 1.0 / max(rows * float(spacing[0]), 1e-12))
    fx = _raw_raytrace_support_axis(cols, 1.0 / max(cols * float(spacing[1]), 1e-12))
    return fx, fy


def _raw_raytrace_psf_function(oi: OpticalImage, field_height_m: float | None = None, wavelength_nm: float | None = None) -> np.ndarray:
    table = _raw_raytrace_table(oi, "psf")
    return _raw_raytrace_psf_function_from_table(table, field_height_m, wavelength_nm)


def _raw_raytrace_psf_function_from_table(
    table: dict[str, Any],
    field_height_m: float | None = None,
    wavelength_nm: float | None = None,
) -> np.ndarray:
    function = np.asarray(table.get("function", np.empty(0, dtype=float)), dtype=float)
    if function.size == 0 or field_height_m is None or wavelength_nm is None:
        return function
    field_heights_m = _raw_raytrace_field_height(table)
    wavelengths_nm = np.asarray(table.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    if field_heights_m.size == 0 or wavelengths_nm.size == 0:
        return function
    field_index = int(np.argmin(np.abs(field_heights_m - float(field_height_m))))
    wave_index = _nearest_wave_index(wavelengths_nm, float(wavelength_nm))
    return function[:, :, field_index, wave_index]


def _raw_raytrace_psf_function_from_optics(
    optics: dict[str, Any],
    field_height_m: float | None = None,
    wavelength_nm: float | None = None,
) -> np.ndarray:
    table = dict(optics.get("raytrace", {}).get("psf", {}))
    return _raw_raytrace_psf_function_from_table(table, field_height_m, wavelength_nm)


def _raw_raytrace_geometry_function(
    oi: OpticalImage,
    wavelength_nm: float | None = None,
    unit: Any | None = None,
) -> np.ndarray:
    table = _raw_raytrace_table(oi, "geometry")
    function = np.asarray(table.get("function", np.empty(0, dtype=float)), dtype=float)
    if function.size == 0:
        return function
    if wavelength_nm is None and unit is None:
        return function
    if wavelength_nm is not None:
        wavelengths_nm = np.asarray(table.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
        if wavelengths_nm.size > 0:
            function = function[:, _nearest_wave_index(wavelengths_nm, float(wavelength_nm))]
    return (np.asarray(function, dtype=float) / 1e3) * _spatial_unit_scale(unit)


def _raytrace_padding_pixels(psf_struct: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(psf_struct, dict):
        return (0, 0)
    psf = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)))
    if psf.ndim != 5 or psf.shape[3] <= 0 or psf.shape[4] <= 0:
        return (0, 0)
    rows = int(psf.shape[3])
    cols = int(psf.shape[4])
    row_extent = int(np.floor(rows / 2.0))
    col_extent = int(np.floor(cols / 2.0))
    pad_rows = int(2 * np.ceil((np.ceil(row_extent) + 2.0) / 2.0))
    pad_cols = int(2 * np.ceil((np.ceil(col_extent) + 2.0) / 2.0))
    return pad_rows, pad_cols


def _raytrace_precompute_psf(
    optics: dict[str, Any],
    wave: np.ndarray,
    rows: int,
    cols: int,
    sample_spacing_m: float,
    *,
    sample_angles_deg: np.ndarray,
) -> dict[str, Any]:
    raytrace = dict(optics.get("raytrace", {}))
    psf_data = dict(raytrace.get("psf", {}))
    psf_function = np.asarray(psf_data.get("function", np.empty(0, dtype=float)), dtype=float)
    field_heights_all = np.asarray(psf_data.get("field_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    sample_angles_deg = np.asarray(sample_angles_deg, dtype=float).reshape(-1)
    angle_step_deg = float(sample_angles_deg[1] - sample_angles_deg[0]) if sample_angles_deg.size >= 2 else 0.0
    if psf_function.size == 0 or field_heights_all.size == 0:
        return {
            "optics_name": str(raytrace.get("name", optics.get("name", "raytrace"))),
            "psf": np.empty((0, 0, 0, 0, 0), dtype=np.float32),
            "sample_angles_deg": np.empty(0, dtype=float),
            "img_height_mm": np.empty(0, dtype=float),
            "wavelength_nm": np.asarray(wave, dtype=float).copy(),
            "angle_step_deg": float(angle_step_deg),
            "cube_shape": (rows, cols),
            "sample_spacing_m": float(sample_spacing_m),
        }

    sample_spacing_mm = float(sample_spacing_m) * 1e3
    _, data_height_mm = _raytrace_data_angle_height(rows, cols, sample_spacing_mm)
    img_height_mm = _raytrace_sample_heights(field_heights_all, data_height_mm)
    source_x_mm, source_y_mm = _raytrace_psf_support_axes(psf_data)
    target_x_mm, target_y_mm = _raytrace_psf_grid(sample_spacing_mm, psf_data)
    n_angles = int(sample_angles_deg.size)
    n_heights = int(img_height_mm.size)
    n_wave = int(np.asarray(wave, dtype=float).size)
    psf_stack = np.empty((n_angles, n_heights, n_wave, target_y_mm.size, target_x_mm.size), dtype=np.float32)
    psf_wavelengths = np.asarray(psf_data.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)

    for wave_index, wavelength_nm in enumerate(np.asarray(wave, dtype=float).reshape(-1)):
        source_wave_index = int(np.argmin(np.abs(psf_wavelengths - float(wavelength_nm)))) if psf_wavelengths.size > 0 else 0
        for height_index, height_mm in enumerate(img_height_mm):
            source_height_index = int(np.argmin(np.abs(field_heights_all - float(height_mm))))
            current = np.asarray(psf_function[:, :, source_height_index, source_wave_index], dtype=float)
            current = np.rot90(current, 2)
            for angle_index, sample_angle in enumerate(sample_angles_deg):
                rotation_deg = 0.0 if height_index < 2 else float(sample_angle)
                rotated = rotate(current, rotation_deg, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
                rotated = uniform_filter(rotated, size=3, mode="nearest")
                kernel = _raytrace_resample_psf(rotated, source_x_mm, source_y_mm, target_x_mm, target_y_mm)
                psf_stack[angle_index, height_index, wave_index, :, :] = kernel.astype(np.float32, copy=False)

    lower_index, angle_weight = _raytrace_angle_lut(sample_angles_deg)
    return {
        "optics_name": str(raytrace.get("name", optics.get("name", "raytrace"))),
        "psf": psf_stack,
        "sample_angles_deg": sample_angles_deg,
        "img_height_mm": img_height_mm,
        "wavelength_nm": np.asarray(wave, dtype=float).copy(),
        "angle_step_deg": float(angle_step_deg),
        "cube_shape": (rows, cols),
        "sample_spacing_m": float(sample_spacing_m),
        "target_x_mm": target_x_mm,
        "target_y_mm": target_y_mm,
        "angle_lut_index": lower_index,
        "angle_lut_weight": angle_weight,
    }


def _raytrace_apply_psf(
    cube: np.ndarray,
    psf_struct: dict[str, Any],
    sample_spacing_m: float,
    *,
    pad_pixels: tuple[int, int] = (0, 0),
) -> np.ndarray:
    psf_stack = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)), dtype=float)
    if psf_stack.size == 0:
        return np.asarray(cube, dtype=float).copy()

    rows, cols = cube.shape[:2]
    sample_spacing_mm = float(sample_spacing_m) * 1e3
    data_angle, data_height = _raytrace_data_angle_height(rows, cols, sample_spacing_mm)
    img_height_mm = np.asarray(psf_struct.get("img_height_mm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
    angle_lut_index = np.asarray(psf_struct.get("angle_lut_index", np.zeros(360, dtype=int)), dtype=int).reshape(-1)
    angle_lut_weight = np.asarray(psf_struct.get("angle_lut_weight", np.ones(360, dtype=float)), dtype=float).reshape(-1)
    if img_height_mm.size < 2:
        return np.asarray(cube, dtype=float).copy()

    pad_rows = int(pad_pixels[0])
    pad_cols = int(pad_pixels[1])
    lower_angle = angle_lut_index[np.clip(data_angle - 1, 0, 359)]
    lower_weight = angle_lut_weight[np.clip(data_angle - 1, 0, 359)]
    unique_angles = np.unique(lower_angle)
    if pad_rows > 0 or pad_cols > 0:
        padded_shape = (rows + 2 * pad_rows, cols + 2 * pad_cols)
    else:
        padded_shape = (rows, cols)
    result = np.zeros((padded_shape[0], padded_shape[1], cube.shape[2]), dtype=float)

    for wave_index in range(cube.shape[2]):
        plane = np.asarray(cube[:, :, wave_index], dtype=float)
        plane_result = np.zeros(padded_shape, dtype=float)
        for height_index in range(1, img_height_mm.size):
            band_mask = (data_height >= img_height_mm[height_index - 1]) & (data_height < img_height_mm[height_index])
            if not np.any(band_mask):
                continue
            inner_weight = 1.0 - (
                np.abs(data_height - img_height_mm[height_index - 1])
                / max(img_height_mm[height_index] - img_height_mm[height_index - 1], 1e-12)
            )
            for angle_index in unique_angles:
                mask = band_mask & (lower_angle == angle_index)
                if not np.any(mask):
                    continue
                masked_plane = plane * mask
                radial_inner = inner_weight * mask
                angular_lower = lower_weight * mask
                if pad_rows > 0 or pad_cols > 0:
                    masked_plane = np.pad(masked_plane, ((pad_rows, pad_rows), (pad_cols, pad_cols)))
                    radial_inner = np.pad(radial_inner, ((pad_rows, pad_rows), (pad_cols, pad_cols)))
                    angular_lower = np.pad(angular_lower, ((pad_rows, pad_rows), (pad_cols, pad_cols)))
                plane_result = plane_result + fftconvolve(
                    masked_plane * radial_inner * angular_lower,
                    psf_stack[angle_index, height_index - 1, wave_index, :, :],
                    mode="same",
                )
                plane_result = plane_result + fftconvolve(
                    masked_plane * (1.0 - radial_inner) * angular_lower,
                    psf_stack[angle_index, height_index, wave_index, :, :],
                    mode="same",
                )
                plane_result = plane_result + fftconvolve(
                    masked_plane * radial_inner * (1.0 - angular_lower),
                    psf_stack[angle_index + 1, height_index - 1, wave_index, :, :],
                    mode="same",
                )
                plane_result = plane_result + fftconvolve(
                    masked_plane * (1.0 - radial_inner) * (1.0 - angular_lower),
                    psf_stack[angle_index + 1, height_index, wave_index, :, :],
                    mode="same",
                )
        result[:, :, wave_index] = plane_result

    return result


def oi_compute(
    oi: OpticalImage | dict[str, Any],
    scene: Scene,
    *args: Any,
    pad_value: str = "zero",
    crop: bool = False,
    pixel_size: float | None = None,
    aperture: np.ndarray | None = None,
    session: SessionContext | None = None,
) -> OpticalImage:
    """Compute a supported optical image from a scene."""

    del args
    if isinstance(oi, dict) and param_format(oi.get("type", "")) == "wvf":
        oi = wvf_to_oi(oi)
    optics = dict(oi.fields["optics"])
    compute_scene = scene
    if pixel_size is not None:
        from .scene import scene_set

        compute_scene = scene.clone()
        scene_cols = int(compute_scene.fields["cols"])
        focal_length_m = float(optics["focal_length_m"])
        w_angular = float(np.rad2deg(2.0 * np.arctan((float(pixel_size) * scene_cols / 2.0) / focal_length_m)))
        compute_scene = scene_set(compute_scene, "wAngular", w_angular)
    if param_format(optics.get("model", "")) == "raytrace":
        object_distance = float(optics.get("raytrace", {}).get("object_distance_m", compute_scene.fields.get("distance_m", np.inf)))
        if not np.isclose(float(compute_scene.fields.get("distance_m", np.inf)), object_distance):
            compute_scene = compute_scene.clone()
            compute_scene.fields["distance_m"] = object_distance
    scene_photons = np.asarray(compute_scene.data["photons"], dtype=float)
    wave = np.asarray(compute_scene.fields["wave"], dtype=float)
    image_distance_m, width_m, height_m = _oi_geometry(optics, compute_scene)
    sample_spacing_m = width_m / max(scene_photons.shape[1], 1)
    photons = _radiance_to_irradiance(scene_photons, optics, compute_scene)
    if param_format(optics.get("offaxis_method", "cos4th")) == "cos4th":
        photons = photons * _cos4th_factor(photons.shape[0], photons.shape[1], optics, compute_scene)[:, :, None]
    extra_blur = float(optics.get("aberration_scale", 0.0))

    pad_pixels = (
        int(np.round(scene_photons.shape[0] / 8.0)),
        int(np.round(scene_photons.shape[1] / 8.0)),
    )
    model = param_format(optics.get("model", ""))
    psf_struct: dict[str, Any] | None = None
    if model == "raytrace":
        angle_step_deg = float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
        sample_angles_deg = _raytrace_requested_sample_angles(
            angle_step_deg,
            oi.fields.get("psf_sample_angles_deg"),
        )
        current_psf_struct = _raytrace_finalize_psf_struct(
            oi.fields.get("psf_struct"),
            optics,
            wave,
            scene_photons.shape[0],
            scene_photons.shape[1],
            sample_spacing_m,
            sample_angles_deg,
        )
        if _raytrace_psf_struct_matches(
            current_psf_struct,
            optics,
            wave,
            scene_photons.shape[0],
            scene_photons.shape[1],
            sample_spacing_m,
            sample_angles_deg,
        ):
            psf_struct = dict(current_psf_struct)
        else:
            psf_struct = _raytrace_precompute_psf(
                optics,
                wave,
                scene_photons.shape[0],
                scene_photons.shape[1],
                sample_spacing_m,
                sample_angles_deg=sample_angles_deg,
            )
        pad_pixels = _raytrace_padding_pixels(psf_struct)
        depth_map = _pad_depth_map(compute_scene, pad_pixels)
        result = _raytrace_apply_psf(
            _raytrace_geometry(photons, wave, optics, compute_scene),
            psf_struct,
            sample_spacing_m,
            pad_pixels=pad_pixels,
        )
        if crop and (pad_pixels[0] > 0 or pad_pixels[1] > 0):
            row_slice = slice(pad_pixels[0], None if pad_pixels[0] == 0 else -pad_pixels[0])
            col_slice = slice(pad_pixels[1], None if pad_pixels[1] == 0 else -pad_pixels[1])
            result = result[row_slice, col_slice, :]
            depth_map = depth_map[row_slice, col_slice]
    else:
        padded, blur_mode, blur_cval = _pad_scene(photons, pad_pixels, pad_value)
        if model == "diffractionlimited":
            otf = _diffraction_otf(padded.shape[:2], sample_spacing_m, wave, optics, compute_scene)
            blurred = _apply_otf(padded, otf)
        elif model == "skip":
            blurred = padded
        elif model == "shiftinvariant":
            compute_method = param_format(optics.get("compute_method", oi.fields.get("compute_method", "opticspsf")))
            if compute_method in {"opticsotf", "humanmw"}:
                otf = _shift_invariant_custom_otf(
                    padded.shape[:2],
                    sample_spacing_m,
                    wave,
                    optics,
                )
                if otf is None:
                    psf_stack = _shift_invariant_psf_stack(
                        padded.shape[:2],
                        sample_spacing_m,
                        wave,
                        optics,
                        aperture=aperture,
                    )
                    blurred = _apply_psf(padded, psf_stack)
                else:
                    blurred = _apply_otf(padded, otf)
            else:
                psf_stack = _shift_invariant_psf_stack(
                    padded.shape[:2],
                    sample_spacing_m,
                    wave,
                    optics,
                    aperture=aperture,
                )
                blurred = _apply_psf(padded, psf_stack)
            if extra_blur > 0.0:
                blurred = apply_channelwise_gaussian(
                    blurred,
                    np.full(wave.shape, extra_blur, dtype=float),
                    mode=blur_mode,
                    cval=blur_cval,
                )
        else:
            sigmas = np.array(
                [
                    gaussian_sigma_pixels(
                        float(optics["f_number"]),
                        float(wavelength),
                        sample_spacing_m,
                        extra_blur_pixels=extra_blur,
                    )
                    for wavelength in wave
                ],
                dtype=float,
            )
            blurred = apply_channelwise_gaussian(padded, sigmas, mode=blur_mode, cval=blur_cval)

        pad_rows, pad_cols = pad_pixels
        depth_map = _pad_depth_map(compute_scene, pad_pixels)
        if crop and (pad_rows > 0 or pad_cols > 0):
            row_slice = slice(pad_rows, None if pad_rows == 0 else -pad_rows)
            col_slice = slice(pad_cols, None if pad_cols == 0 else -pad_cols)
            result = blurred[row_slice, col_slice, :]
            depth_map = depth_map[row_slice, col_slice]
        else:
            result = blurred

    output_sample_spacing_m = float(sample_spacing_m)
    output_width_m = float(result.shape[1] * output_sample_spacing_m)
    output_height_m = float(result.shape[0] * output_sample_spacing_m)
    output_fov_deg = float(np.rad2deg(2.0 * np.arctan2(output_width_m / 2.0, image_distance_m)))
    output_vfov_deg = float(np.rad2deg(2.0 * np.arctan2(output_height_m / 2.0, image_distance_m)))

    if crop:
        focal_length_m = float(optics["focal_length_m"])
        output_fov_deg = float(
            np.rad2deg(2.0 * np.arctan2((result.shape[1] * sample_spacing_m) / 2.0, focal_length_m))
        )
        output_width_m = float(2.0 * image_distance_m * np.tan(np.deg2rad(output_fov_deg) / 2.0))
        output_sample_spacing_m = output_width_m / max(result.shape[1], 1)
        output_height_m = float(result.shape[0] * output_sample_spacing_m)
        output_vfov_deg = float(np.rad2deg(2.0 * np.arctan2(output_height_m / 2.0, image_distance_m)))

    computed = oi.clone()
    computed.name = compute_scene.name
    computed.fields["wave"] = wave
    computed.fields["compute_method"] = optics.get("compute_method", computed.fields.get("compute_method", "opticsotf"))
    computed.fields["pad_value"] = pad_value
    computed.fields["crop"] = bool(crop)
    computed.fields["padding_pixels"] = pad_pixels
    computed.fields["sample_spacing_m"] = output_sample_spacing_m
    computed.fields["image_distance_m"] = image_distance_m
    computed.fields["depth_map_m"] = depth_map
    computed.fields["width_m"] = output_width_m
    computed.fields["height_m"] = output_height_m
    computed.fields["fov_deg"] = output_fov_deg
    computed.fields["vfov_deg"] = output_vfov_deg
    if model == "raytrace":
        computed.fields["psf_angle_step_deg"] = float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
        computed.fields["psf_struct"] = psf_struct
        _sync_psf_metadata_fields(computed)
    computed.data["photons"] = result
    if model != "raytrace":
        diffuser_method = param_format(computed.fields.get("diffuser_method", "skip"))
        if diffuser_method == "blur":
            blur_m = float(computed.fields.get("diffuser_blur_m", 0.0))
            if blur_m > 0.0:
                computed, _, _ = oi_diffuser(computed, blur_m * 1e6)
        elif diffuser_method == "birefringent":
            displacement_um = float(computed.fields.get("sample_spacing_m", output_sample_spacing_m)) * 1e6
            computed, _ = oi_birefringent_diffuser(computed, displacement_um)
        elif diffuser_method != "skip":
            raise UnsupportedOptionError("oiCompute", computed.fields.get("diffuser_method", diffuser_method))
    if pixel_size is not None:
        computed = _oi_spatial_resample(computed, float(pixel_size), method="linear")
        computed.fields["sample_spacing_m"] = float(pixel_size)
        computed.fields["requested_pixel_size_m"] = float(pixel_size)
    return track_session_object(session, computed)


def oi_clear_data(oi: OpticalImage) -> OpticalImage:
    """Clear computed data from an optical image while preserving configuration."""

    cleared = oi.clone()
    cleared.data.clear()
    cleared.fields.pop("illuminance", None)
    cleared.fields.pop("mean_illuminance", None)
    cleared.fields.pop("mean_comp_illuminance", None)
    cleared.fields["depth_map_m"] = None
    _clear_precomputed_psf_state(cleared)

    optics = dict(cleared.fields.get("optics", {}))
    wavefront = optics.get("wavefront")
    if isinstance(wavefront, dict):
        optics["wavefront"] = wvf_clear_data(wavefront)
    cleared.fields["optics"] = optics
    return cleared


def oi_crop(oi: OpticalImage, rect: Any) -> OpticalImage:
    if isinstance(rect, str):
        normalized = param_format(rect)
        if normalized != "border":
            raise UnsupportedOptionError("oiCrop", rect)
        size = np.asarray(oi_get(oi, "size"), dtype=float).reshape(-1)
        if size.size != 2:
            raise ValueError("oiCrop(..., 'border') requires a 2-D optical image.")
        rect_array = np.ceil(
            [
                size[1] * 0.1 + 1.0,
                size[0] * 0.1 + 1.0,
                size[1] * 0.8 - 1.0,
                size[0] * 0.8 - 1.0,
            ],
        ).astype(int)
        if ((rect_array[2] - rect_array[0]) % 2) != 0:
            rect_array[2] -= 1
        if ((rect_array[3] - rect_array[1]) % 2) != 0:
            rect_array[3] -= 1
    else:
        rect_array = np.rint(np.asarray(rect, dtype=float).reshape(-1)).astype(int)
    if rect_array.size != 4:
        raise ValueError("oi_crop expects [col, row, width, height].")

    from .roi import ie_rect2_locs, vc_get_roi_data

    roi_locs = ie_rect2_locs(rect_array)
    cropped_rows = int(rect_array[3]) + 1
    cropped_cols = int(rect_array[2]) + 1
    sample_spacing = np.asarray(oi_get(oi, "distance per sample"), dtype=float)

    photons = np.asarray(vc_get_roi_data(oi, roi_locs, "photons"), dtype=float)
    photons = photons.reshape(cropped_rows, cropped_cols, -1)

    cropped = oi.clone()
    cropped = oi_set(cropped, "photons", photons)

    depth_map = oi.fields.get("depth_map_m")
    if depth_map is not None:
        depth = np.asarray(depth_map, dtype=float)
        row_index = np.clip(roi_locs[:, 0] - 1, 0, depth.shape[0] - 1)
        col_index = np.clip(roi_locs[:, 1] - 1, 0, depth.shape[1] - 1)
        cropped.fields["depth_map_m"] = depth[row_index, col_index].reshape(cropped_rows, cropped_cols)

    oi_calculate_illuminance(cropped)

    focal_length = float(oi_get(cropped, "optics focal length"))
    new_wangular = float(np.rad2deg(2.0 * np.arctan2((cropped_cols * sample_spacing[1]) / 2.0, focal_length)))
    cropped = oi_set(cropped, "wangular", new_wangular)
    return cropped


def oi_spatial_resample(
    oi: OpticalImage,
    sample_spacing: float,
    units: str | None = "m",
    *,
    method: str = "linear",
) -> OpticalImage:
    sample_spacing_m = float(sample_spacing) / max(_spatial_unit_scale(units), 1.0e-30)
    if sample_spacing_m <= 0.0:
        raise ValueError("oi_spatial_resample requires a positive sample spacing.")
    return _oi_spatial_resample(oi, sample_spacing_m, method=method)


def oi_psf(oi: OpticalImage, param: str, *args: Any) -> float:
    """Summarize a point-spread-function OI using the legacy MATLAB oiPSF contract."""

    options = _parse_key_value_options(args, "oiPSF") if args else {}
    units = str(options.pop("units", "m"))
    threshold = float(options.pop("threshold", 0.1))
    wavelength_nm = options.pop("wave", options.pop("wavelength", None))
    if options:
        unsupported = next(iter(options))
        raise KeyError(f"Unsupported oiPSF parameter: {unsupported}")

    normalized = param_format(param)
    if normalized not in {"area", "diameter"}:
        raise UnsupportedOptionError("oiPSF", param)
    if threshold < 0.0:
        raise ValueError("oiPSF threshold must be non-negative.")

    psf_data = _oi_psf_data(oi, wavelength_nm=wavelength_nm, units=units)
    psf = np.asarray(psf_data["psf"], dtype=float)
    xy = np.asarray(psf_data["xy"], dtype=float)
    if psf.ndim != 2 or xy.ndim != 3:
        raise ValueError("oiPSF requires a 2-D PSF plane and a 2-D support grid.")

    x_axis = np.asarray(xy[0, :, 0], dtype=float).reshape(-1)
    y_axis = np.asarray(xy[:, 0, 1], dtype=float).reshape(-1)
    if x_axis.size < 2 or y_axis.size < 2:
        raise ValueError("oiPSF requires at least two spatial samples per axis.")

    peak = float(np.max(psf)) if psf.size else 0.0
    mask = np.asarray(psf >= (threshold * peak), dtype=float) if peak > 0.0 else np.zeros_like(psf, dtype=float)
    dx = float(abs(x_axis[1] - x_axis[0]))
    dy = float(abs(y_axis[1] - y_axis[0]))
    area = float(np.sum(mask) * dx * dy)
    if normalized == "area":
        return area
    return float(2.0 * np.sqrt(area / np.pi))


oiPSF = oi_psf


def _oi_shape(oi: OpticalImage) -> tuple[int, int]:
    photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    if photons.ndim >= 2:
        return int(photons.shape[0]), int(photons.shape[1])
    return int(oi.fields.get("rows", 0)), int(oi.fields.get("cols", 0))


def _oi_depth_distance_m(oi: OpticalImage) -> float | None:
    depth_map = oi.fields.get("depth_map_m")
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=float)
    positive = depth[depth > 0.0]
    if positive.size == 0:
        return None
    if depth.shape[0] < 20 or depth.shape[1] < 20:
        return float(np.mean(positive))
    center_half = 10
    row_center = depth.shape[0] // 2
    col_center = depth.shape[1] // 2
    center = depth[
        max(row_center - center_half, 0) : min(row_center + center_half, depth.shape[0]),
        max(col_center - center_half, 0) : min(col_center + center_half, depth.shape[1]),
    ]
    center_positive = center[center > 0.0]
    if center_positive.size > 0:
        return float(np.mean(center_positive))
    return float(np.mean(positive))


def _oi_image_distance_m(oi: OpticalImage) -> float:
    if oi.fields.get("image_distance_m") is not None:
        return float(oi.fields["image_distance_m"])
    scene_distance = _oi_depth_distance_m(oi)
    focal_length = float(oi.fields["optics"].get("focal_length_m", DEFAULT_FOCAL_LENGTH_M))
    if scene_distance is not None:
        if param_format(oi.fields["optics"].get("model", "")) == "skip":
            return focal_length
        if scene_distance <= focal_length:
            return focal_length
        return float(1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12))
    return focal_length


def _oi_width_m(oi: OpticalImage) -> float:
    rows, cols = _oi_shape(oi)
    if oi.fields.get("width_m") is not None:
        return float(oi.fields["width_m"])
    image_distance = _oi_image_distance_m(oi)
    fov_deg = float(oi.fields.get("fov_deg", 10.0))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(fov_deg) / 2.0)
    if cols > 0 and rows > 0:
        return float(width_m)
    return float(width_m)


def _oi_height_m(oi: OpticalImage) -> float:
    rows, cols = _oi_shape(oi)
    if oi.fields.get("height_m") is not None:
        return float(oi.fields["height_m"])
    width_m = _oi_width_m(oi)
    if rows > 0 and cols > 0:
        return float(width_m * rows / max(cols, 1))
    image_distance = _oi_image_distance_m(oi)
    vfov_deg = float(oi.fields.get("vfov_deg", oi.fields.get("fov_deg", 10.0)))
    return float(2.0 * image_distance * np.tan(np.deg2rad(vfov_deg) / 2.0))


def _oi_sample_size_m(oi: OpticalImage) -> float | None:
    sample_spacing = oi.fields.get("sample_spacing_m")
    if sample_spacing is not None:
        return float(sample_spacing)
    _, cols = _oi_shape(oi)
    if cols <= 0:
        return None
    return float(_oi_width_m(oi) / cols)


def _oi_distance_per_degree_m(oi: OpticalImage) -> float:
    return _oi_width_m(oi) / max(float(oi.fields.get("fov_deg", 10.0)), 1e-12)


def _oi_spatial_support_linear(oi: OpticalImage) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _oi_shape(oi)
    spatial = np.asarray(oi_get(oi, "distancepersample"), dtype=float)
    if rows <= 0 or cols <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    y = np.linspace(-(rows * spatial[0]) / 2.0 + spatial[0] / 2.0, (rows * spatial[0]) / 2.0 - spatial[0] / 2.0, rows)
    x = np.linspace(-(cols * spatial[1]) / 2.0 + spatial[1] / 2.0, (cols * spatial[1]) / 2.0 - spatial[1] / 2.0, cols)
    return x, y


def _oi_angular_support_linear_deg(oi: OpticalImage) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _oi_shape(oi)
    angular = np.asarray(oi_get(oi, "angularresolution"), dtype=float)
    if rows <= 0 or cols <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    x = angular[1] * (np.arange(cols, dtype=float) + 1.0)
    y = angular[0] * (np.arange(rows, dtype=float) + 1.0)
    return x - np.mean(x), y - np.mean(y)


def _oi_frequency_support_1d(oi: OpticalImage, unit: str = "cyclesperdegree") -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _oi_shape(oi)
    fov_width = float(oi.fields.get("fov_deg", 10.0))
    fov_height = float(oi.fields.get("vfov_deg", oi.fields.get("fov_deg", 10.0)))
    max_frequency_cpd = np.array(
        [
            (cols / 2.0) / max(fov_width, 1e-12),
            (rows / 2.0) / max(fov_height, 1e-12),
        ],
        dtype=float,
    )
    normalized_unit = param_format(unit)
    if normalized_unit in {"cyclesperdegree", "cycperdeg"}:
        max_frequency = max_frequency_cpd
    elif normalized_unit in {"meters", "m", "millimeters", "mm", "microns", "um"}:
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1e3, "mm": 1e3, "microns": 1e6, "um": 1e6}
        deg_per_dist = 1.0 / max(_oi_distance_per_degree_m(oi) * unit_scale[normalized_unit], 1e-12)
        max_frequency = max_frequency_cpd * deg_per_dist
    else:
        raise KeyError(f"Unsupported oiGet frequency unit: {unit}")
    fx = unit_frequency_list(cols) * max_frequency[0]
    fy = unit_frequency_list(rows) * max_frequency[1]
    return fx, fy


def _oi_illuminant_photons(oi: OpticalImage) -> np.ndarray:
    stored = oi.fields.get("illuminant_photons")
    if stored is None:
        return np.empty(0, dtype=float)
    return np.asarray(stored, dtype=float)


def _oi_illuminant_format(oi: OpticalImage) -> str:
    illuminant = _oi_illuminant_photons(oi)
    wave = np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    if illuminant.ndim == 1 and illuminant.size == wave.size:
        return "spectral"
    if illuminant.ndim == 3 and illuminant.shape[2] == wave.size:
        return "spatial spectral"
    return ""


def _set_oi_illuminant_photons(oi: OpticalImage, value: Any) -> OpticalImage:
    illuminant = np.asarray(value, dtype=float)
    wave = np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    rows, cols = _oi_shape(oi)
    if illuminant.ndim == 1:
        if illuminant.size != wave.size:
            raise ValueError("Spectral illuminant photons must match the OI wavelength support.")
        oi.fields["illuminant_photons"] = illuminant.reshape(-1).copy()
        return oi
    if illuminant.ndim == 3:
        if illuminant.shape[2] != wave.size:
            raise ValueError("Spatial-spectral illuminant photons must match the OI wavelength support.")
        if rows > 0 and cols > 0 and illuminant.shape[:2] != (rows, cols):
            raise ValueError("Spatial-spectral illuminant photons must match the OI image size.")
        oi.fields["illuminant_photons"] = illuminant.copy()
        return oi
    raise ValueError("OI illuminant photons must be either a 1-D spectrum or a 3-D spatial-spectral cube.")


def _set_oi_illuminant(oi: OpticalImage, illuminant: Any) -> OpticalImage:
    wave = np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    if isinstance(illuminant, BaseISETObject):
        photons = np.asarray(illuminant_get(illuminant, "photons", wave), dtype=float)
        return _set_oi_illuminant_photons(oi, photons)
    if isinstance(illuminant, dict) and "photons" in illuminant:
        photons = np.asarray(illuminant["photons"], dtype=float)
        source_wave = illuminant.get("wave")
        if source_wave is not None and photons.ndim == 1:
            photons = np.asarray(interp_spectra(np.asarray(source_wave, dtype=float).reshape(-1), photons.reshape(-1), wave), dtype=float)
        return _set_oi_illuminant_photons(oi, photons)
    raise ValueError("oiSet(..., 'illuminant', value) expects an illuminant object or a dict with photons.")


def _resize_oi_pattern(pattern: Any, size: tuple[int, int]) -> np.ndarray:
    array = np.asarray(pattern, dtype=float)
    if array.ndim != 2:
        raise ValueError("OI illuminant patterns must be 2-D arrays.")
    if array.shape == tuple(size):
        return array.copy()
    return _resize_image(array, size, method="linear")


def _sync_oi_geometry_fields(oi: OpticalImage) -> None:
    rows, cols = _oi_shape(oi)
    oi.fields["rows"] = rows
    oi.fields["cols"] = cols
    if rows <= 0 or cols <= 0:
        return

    image_distance = _oi_image_distance_m(oi)
    fov_deg = float(oi.fields.get("fov_deg", 10.0))
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(fov_deg) / 2.0)
    sample_spacing_m = width_m / cols
    height_m = sample_spacing_m * rows
    vfov_deg = float(np.rad2deg(2.0 * np.arctan2(height_m / 2.0, image_distance)))

    oi.fields["image_distance_m"] = image_distance
    oi.fields["width_m"] = float(width_m)
    oi.fields["height_m"] = float(height_m)
    oi.fields["sample_spacing_m"] = float(sample_spacing_m)
    oi.fields["vfov_deg"] = vfov_deg


def _sync_oi_identity_fields(oi: OpticalImage) -> None:
    oi.fields["name"] = str(oi.name)
    oi.fields["type"] = str(oi.type)


def _human_lens_payload(wave: np.ndarray, *, asset_store: AssetStore | None = None) -> dict[str, Any]:
    sampled_wave = np.asarray(wave, dtype=float).reshape(-1)
    _, density = _store(asset_store).load_spectra("lensDensity.mat", wave_nm=sampled_wave)
    density_profile = np.asarray(density, dtype=float).reshape(-1)
    return {
        "type": "lens",
        "name": "human lens",
        "wave": sampled_wave.copy(),
        "density": density_profile.copy(),
        "transmittance": np.power(10.0, -density_profile),
    }


def _normalize_lens_payload(value: Any, *, default_wave: np.ndarray) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Lens payload must be a mapping.")
    payload = dict(value)
    wave = np.asarray(payload.get("wave", default_wave), dtype=float).reshape(-1)
    if payload.get("transmittance") is not None:
        transmittance = np.asarray(payload.get("transmittance"), dtype=float).reshape(-1)
        if transmittance.size != wave.size:
            raise ValueError("Lens transmittance must match wave dimension.")
        density = payload.get("density")
        density_array = None if density is None else np.asarray(density, dtype=float).reshape(-1)
        if density_array is not None and density_array.size != wave.size:
            raise ValueError("Lens density must match wave dimension.")
    else:
        density = payload.get("density")
        if density is None:
            transmittance = np.ones(wave.size, dtype=float)
            density_array = np.zeros(wave.size, dtype=float)
        else:
            density_array = np.asarray(density, dtype=float).reshape(-1)
            if density_array.size != wave.size:
                raise ValueError("Lens density must match wave dimension.")
            transmittance = np.power(10.0, -density_array)
    return {
        "type": str(payload.get("type", "lens")),
        "name": str(payload.get("name", "lens")),
        "wave": wave.copy(),
        "transmittance": np.asarray(transmittance, dtype=float).reshape(-1).copy(),
        **({"density": density_array.copy()} if density_array is not None else {}),
    }


def _optics_lens_payload(optics: dict[str, Any]) -> dict[str, Any] | None:
    lens = optics.get("lens")
    if not isinstance(lens, dict):
        return None
    if "wave" not in lens or "transmittance" not in lens:
        return None
    return {
        "type": str(lens.get("type", "lens")),
        "name": str(lens.get("name", "lens")),
        "wave": np.asarray(lens.get("wave"), dtype=float).reshape(-1).copy(),
        "transmittance": np.asarray(lens.get("transmittance"), dtype=float).reshape(-1).copy(),
        **({"density": np.asarray(lens.get("density"), dtype=float).reshape(-1).copy()} if lens.get("density") is not None else {}),
    }


def _optics_transmittance_payload(optics: dict[str, Any], wave: np.ndarray) -> dict[str, np.ndarray]:
    transmittance = optics.get("transmittance")
    if isinstance(transmittance, dict) and transmittance.get("wave") is not None and transmittance.get("scale") is not None:
        return {
            "wave": np.asarray(transmittance.get("wave"), dtype=float).reshape(-1).copy(),
            "scale": np.asarray(transmittance.get("scale"), dtype=float).reshape(-1).copy(),
        }
    lens = _optics_lens_payload(optics)
    if lens is not None:
        return {
            "wave": np.asarray(lens["wave"], dtype=float).reshape(-1).copy(),
            "scale": np.asarray(lens["transmittance"], dtype=float).reshape(-1).copy(),
        }
    sampled_wave = np.asarray(wave, dtype=float).reshape(-1)
    return {
        "wave": sampled_wave.copy(),
        "scale": np.ones(sampled_wave.size, dtype=float),
    }


def _set_optics_transmittance_payload(optics: dict[str, Any], wave: np.ndarray, scale: np.ndarray) -> None:
    wave_array = np.asarray(wave, dtype=float).reshape(-1)
    scale_array = np.asarray(scale, dtype=float).reshape(-1)
    if wave_array.size != scale_array.size:
        raise ValueError("Transmittance must match wave dimension.")
    if "transmittance" in optics:
        optics["transmittance"] = {
            "wave": wave_array.copy(),
            "scale": scale_array.copy(),
        }
        return
    if isinstance(optics.get("lens"), dict):
        updated_lens = dict(optics["lens"])
        updated_lens["wave"] = wave_array.copy()
        updated_lens["transmittance"] = scale_array.copy()
        updated_lens.pop("density", None)
        optics["lens"] = updated_lens
        return
    optics["transmittance"] = {
        "wave": wave_array.copy(),
        "scale": scale_array.copy(),
    }


def oi_get(oi: OpticalImage, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    prefix, remainder = split_prefixed_parameter(parameter, ("optics", "wvf"))
    if prefix == "optics" and remainder:
        if remainder in {"rtpsfsize", "rtpsfdimensions"}:
            return _raw_raytrace_psf_dimensions(oi)
        return oi_get(oi, remainder, *args)
    if prefix == "wvf" and remainder:
        wavefront = dict(oi.fields["optics"].get("wavefront", {}))
        return wvf_get(wavefront, remainder, *args)
    if key == "type":
        return oi.type
    if key == "name":
        return oi.name
    if key == "metadata":
        return oi.metadata
    if key == "data":
        return oi.data
    if key == "lens":
        lens = _optics_lens_payload(oi.fields["optics"])
        return None if lens is None else lens
    if key in {"wave", "wavelength"}:
        return np.asarray(oi.fields["wave"], dtype=float)
    if key == "illuminantformat":
        return _oi_illuminant_format(oi)
    if key == "illuminant":
        return {
            "wave": np.asarray(oi.fields["wave"], dtype=float).reshape(-1),
            "photons": _oi_illuminant_photons(oi),
            "comment": oi.fields.get("illuminant_comment"),
        }
    if key == "illuminantphotons":
        return _oi_illuminant_photons(oi)
    if key == "photons":
        return np.asarray(oi.data["photons"], dtype=float)
    if key in {"irradiancehline", "hline", "hlineirradiance"}:
        if not args:
            raise ValueError("Line location required for oiGet(..., 'irradiance hline').")
        unit = args[1] if len(args) >= 2 else "um"
        return _oi_line_profile(oi, "photons", "h", args[0], unit=unit)
    if key in {"irradiancevline", "vline", "vlineirradiance"}:
        if not args:
            raise ValueError("Line location required for oiGet(..., 'irradiance vline').")
        unit = args[1] if len(args) >= 2 else "um"
        return _oi_line_profile(oi, "photons", "v", args[0], unit=unit)
    if key in {"irradianceenergyhline", "hlineenergy", "hlineirradianceenergy"}:
        if not args:
            raise ValueError("Line location required for oiGet(..., 'irradiance energy hline').")
        unit = args[1] if len(args) >= 2 else "um"
        return _oi_line_profile(oi, "energy", "h", args[0], unit=unit)
    if key in {"irradianceenergyvline", "vlineenergy", "vlineirradianceenergy"}:
        if not args:
            raise ValueError("Line location required for oiGet(..., 'irradiance energy vline').")
        unit = args[1] if len(args) >= 2 else "um"
        return _oi_line_profile(oi, "energy", "v", args[0], unit=unit)
    if key in {"illuminancehline", "horizontallineilluminance", "hlineilluminance"}:
        if not args:
            raise ValueError("Line location required for oiGet(..., 'illuminance hline').")
        unit = args[1] if len(args) >= 2 else "um"
        return _oi_line_profile(oi, "illuminance", "h", args[0], unit=unit)
    if key in {"illuminancevline", "vlineilluminance"}:
        if not args:
            raise ValueError("Line location required for oiGet(..., 'illuminance vline').")
        unit = args[1] if len(args) >= 2 else "um"
        return _oi_line_profile(oi, "illuminance", "v", args[0], unit=unit)
    if key == "roiphotons":
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi photons').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(oi, args[0], "photons")
    if key == "roimeanphotons":
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi mean photons').")
        return np.mean(np.asarray(oi_get(oi, "roi photons", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roienergy":
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi energy').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(oi, args[0], "energy")
    if key == "roimeanenergy":
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi mean energy').")
        return np.mean(np.asarray(oi_get(oi, "roi energy", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roiilluminance":
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi illuminance').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(oi, args[0], "illuminance")
    if key == "roimeanilluminance":
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi mean illuminance').")
        return float(np.mean(np.asarray(oi_get(oi, "roi illuminance", args[0]), dtype=float)))
    if key in {"chromaticity", "roichromaticity"}:
        roi_locs = args[0] if args else None
        return chromaticity_xy(_oi_roi_xyz(oi, roi_locs))
    if key in {"roichromaticitymean", "roimeanchromaticity"}:
        if not args:
            raise ValueError("ROI required for oiGet(..., 'roi chromaticity mean').")
        return np.mean(np.asarray(oi_get(oi, "chromaticity", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "illuminance":
        stored = oi.fields.get("illuminance")
        if stored is not None:
            return np.asarray(stored, dtype=float)
        return _oi_illuminance(oi)
    if key == "meanilluminance":
        stored = oi.fields.get("mean_illuminance")
        if stored is not None:
            return float(stored)
        illuminance = np.asarray(oi_get(oi, "illuminance"), dtype=float)
        return 0.0 if illuminance.size == 0 else float(np.mean(illuminance))
    if key == "meancompilluminance":
        return float(oi.fields.get("mean_comp_illuminance", 0.0))
    if key == "depthmap":
        depth_map = oi.fields.get("depth_map_m")
        if depth_map is None:
            rows, cols = _oi_shape(oi)
            if rows <= 0 or cols <= 0:
                depth_map = np.empty((0, 0), dtype=float)
            else:
                scene_distance = _oi_depth_distance_m(oi)
                if scene_distance is None:
                    scene_distance = 1.2
                depth_map = np.full((rows, cols), float(scene_distance), dtype=float)
        scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]), 1.0) if args else 1.0
        return np.asarray(depth_map, dtype=float) * scale
    if key == "depthrange":
        depth_map = np.asarray(oi_get(oi, "depth map"), dtype=float)
        positive = depth_map[depth_map > 0.0]
        if positive.size == 0:
            return np.empty(0, dtype=float)
        scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]), 1.0) if args else 1.0
        return np.array([positive.min(), positive.max()], dtype=float) * scale
    if key == "rows":
        return _oi_shape(oi)[0]
    if key == "cols":
        return _oi_shape(oi)[1]
    if key == "size":
        return _oi_shape(oi)
    if key in {"centerpixel", "centerpoint"}:
        return np.floor(np.array([oi_get(oi, "rows"), oi_get(oi, "cols")], dtype=float) / 2.0).astype(int) + 1
    if key in {"imagedistance", "focalplanedistance", "distance"}:
        return _oi_image_distance_m(oi)
    if key in {"wangular", "widthangular", "hfov", "horizontalfieldofview", "fov"}:
        return float(oi.fields.get("fov_deg", 10.0))
    if key in {"hangular", "heightangular", "vfov", "verticalfieldofview"}:
        return float(oi.fields.get("vfov_deg", oi.fields.get("fov_deg", 10.0)))
    if key == "width":
        return _oi_width_m(oi)
    if key == "height":
        return _oi_height_m(oi)
    if key in {"diagonal", "diagonalsize"}:
        return float(np.hypot(_oi_height_m(oi), _oi_width_m(oi)))
    if key in {"heightwidth", "heightandwidth"}:
        return np.array([_oi_height_m(oi), _oi_width_m(oi)], dtype=float)
    if key in {"samplespacing", "sample spacing"}:
        rows, cols = _oi_shape(oi)
        width_spacing = 0.0 if cols <= 0 else _oi_width_m(oi) / cols
        height_spacing = 0.0 if rows <= 0 else _oi_height_m(oi) / rows
        return np.array([width_spacing, height_spacing], dtype=float)
    if key in {"samplesize"}:
        return _oi_sample_size_m(oi)
    if key in {"hspatialresolution", "heightspatialresolution", "hres"}:
        rows, _ = _oi_shape(oi)
        return 0.0 if rows <= 0 else float(_oi_height_m(oi) / rows)
    if key in {"wspatialresolution", "widthspatialresolution", "wres"}:
        _, cols = _oi_shape(oi)
        return 0.0 if cols <= 0 else float(_oi_width_m(oi) / cols)
    if key in {"spatialresolution", "distancepersample", "distpersamp"}:
        return np.array(
            [
                oi_get(oi, "hspatialresolution"),
                oi_get(oi, "wspatialresolution"),
            ],
            dtype=float,
        )
    if key in {"distanceperdegree", "distperdeg"}:
        unit = param_format(args[0]) if args else "m"
        val = _oi_distance_per_degree_m(oi)
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1e3, "mm": 1e3, "microns": 1e6, "um": 1e6}
        return val * unit_scale.get(unit, 1.0)
    if key in {"degreesperdistance", "degperdist"}:
        unit = args[0] if args else "m"
        return 1.0 / max(float(oi_get(oi, "distanceperdegree", unit)), 1e-12)
    if key in {"hangularresolution", "heightangularresolution"}:
        spatial = float(oi_get(oi, "hspatialresolution"))
        image_distance = _oi_image_distance_m(oi)
        return float(np.rad2deg(2.0 * np.arctan((spatial / image_distance) / 2.0)))
    if key in {"wangularresolution", "widthangularresolution"}:
        spatial = float(oi_get(oi, "wspatialresolution"))
        image_distance = _oi_image_distance_m(oi)
        return float(np.rad2deg(2.0 * np.arctan((spatial / image_distance) / 2.0)))
    if key in {"angularresolution", "degperpixel", "degpersample", "degreepersample", "degreeperpixel"}:
        return np.array(
            [
                oi_get(oi, "hangularresolution"),
                oi_get(oi, "wangularresolution"),
            ],
            dtype=float,
        )
    if key in {"spatialsupportlinear"}:
        x, y = _oi_spatial_support_linear(oi)
        unit = param_format(args[0]) if args else "m"
        unit_scale = {"meters": 1.0, "m": 1.0, "millimeters": 1e3, "mm": 1e3, "microns": 1e6, "um": 1e6}
        scale = unit_scale.get(unit, 1.0)
        return {"x": x * scale, "y": y * scale}
    if key in {"spatialsupportmesh", "spatialsupport"}:
        support = oi_get(oi, "spatialsupportlinear", *(args[:1] if args else ()))
        xx, yy = np.meshgrid(support["x"], support["y"])
        return np.stack((xx, yy), axis=2)
    if key in {"angularsupport", "angularsamplingpositions"}:
        x, y = _oi_angular_support_linear_deg(oi)
        unit = param_format(args[0]) if args else "deg"
        if unit == "deg":
            scale = 1.0
        elif unit == "min":
            scale = 60.0
        elif unit == "sec":
            scale = 3600.0
        elif unit == "radians":
            scale = np.pi / 180.0
        else:
            raise KeyError(f"Unsupported oiGet angular support unit: {args[0]}")
        xx, yy = np.meshgrid(x * scale, y * scale)
        return np.stack((xx, yy), axis=2)
    if key in {"frequencyresolution", "freqres"}:
        unit = args[0] if args else "cyclesperdegree"
        fx, fy = _oi_frequency_support_1d(oi, str(unit))
        return {"fx": fx, "fy": fy}
    if key in {"maxfrequencyresolution", "maxfreqres"}:
        unit = args[0] if args else "cyclesperdegree"
        support = oi_get(oi, "frequencyresolution", unit)
        return float(max(np.max(support["fx"]), np.max(support["fy"])))
    if key in {"frequencysupport", "fsupportxy", "fsupport2d", "fsupport"}:
        unit = args[0] if args else "cyclesperdegree"
        fx, fy = _oi_frequency_support_1d(oi, str(unit))
        xx, yy = np.meshgrid(fx, fy)
        return np.stack((xx, yy), axis=2)
    if key in {"frequencysupportcol", "fsupportx"}:
        unit = args[0] if args else "cyclesperdegree"
        fx, _ = _oi_frequency_support_1d(oi, str(unit))
        return fx
    if key in {"frequencysupportrow", "fsupporty"}:
        unit = args[0] if args else "cyclesperdegree"
        _, fy = _oi_frequency_support_1d(oi, str(unit))
        return fy
    if key == "optics":
        return _export_optics(oi.fields["optics"])
    if key in {"focallength", "opticsfocallength"}:
        return float(oi.fields["optics"]["focal_length_m"]) * _spatial_unit_scale(args[0] if args else None)
    if key in {"fnumber", "opticsfnumber"}:
        if param_format(oi.fields["optics"].get("model", "")) == "raytrace":
            return float(oi.fields["optics"].get("raytrace", {}).get("f_number", oi.fields["optics"]["f_number"]))
        return float(oi.fields["optics"]["f_number"])
    if key in {"pupildiameter", "pupilsize", "pdiameter", "opticspupildiameter", "opticspupilsize", "opticspdiameter"}:
        optics = oi.fields["optics"]
        f_number = float(oi_get(oi, "fnumber"))
        pupil_m = float(optics["focal_length_m"]) / max(f_number, 1.0e-12)
        return pupil_m * _spatial_unit_scale(args[0] if args else None)
    if key in {"aperturediameter", "opticsaperturediameter"}:
        optics = oi.fields["optics"]
        f_number = float(oi_get(oi, "fnumber"))
        aperture_m = float(optics["focal_length_m"]) / max(f_number, 1.0e-12)
        return aperture_m * _spatial_unit_scale(args[0] if args else None)
    if key in {"opticsmodel", "model"}:
        return oi.fields["optics"]["model"]
    if key in {"computemethod"}:
        return oi.fields["optics"].get("compute_method", oi.fields.get("compute_method"))
    if key in {"customcompute"}:
        method = oi.fields.get("custom_compute_method", oi.fields["optics"].get("compute_method", oi.fields.get("compute_method", "")))
        normalized = param_format(method)
        return normalized not in {"", "opticspsf", "opticsotf", "humanmw", "skip"}
    if key in {"customcomputemethod"}:
        if not bool(oi_get(oi, "custom compute")):
            return None
        return oi.fields.get("custom_compute_method", oi.fields["optics"].get("compute_method", oi.fields.get("compute_method")))
    if key in {"diffusermethod"}:
        return oi.fields.get("diffuser_method", "skip")
    if key in {"diffuserblur"}:
        return float(oi.fields.get("diffuser_blur_m", 0.0))
    if key in {"offaxismethod", "opticsoffaxismethod"}:
        return oi.fields["optics"].get("offaxis_method", "cos4th")
    if key in {"opticswvf", "wvf", "wavefront"}:
        wavefront = dict(oi.fields["optics"].get("wavefront", {}))
        if not args:
            return wavefront
        return wvf_get(wavefront, str(args[0]), *args[1:])
    if key in {"opticsraytrace", "raytrace", "rt"}:
        return _export_raytrace(oi.fields["optics"].get("raytrace", {}))
    if key in {"rtname"}:
        return oi.fields["optics"].get("raytrace", {}).get("name", oi.fields["optics"].get("name"))
    if key in {"opticsprogram", "rtopticsprogram"}:
        return oi.fields["optics"].get("raytrace", {}).get("program", "")
    if key in {"lensfile", "rtlensfile"}:
        return oi.fields["optics"].get("raytrace", {}).get("lens_file", "")
    if key in {"rteffectivefnumber", "rtefff#"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("effective_f_number", oi.fields["optics"].get("f_number", np.nan)))
    if key in {"rtfnumber"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("f_number", oi.fields["optics"].get("f_number", np.nan)))
    if key in {"rtmagnification", "rtmag"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("magnification", 0.0))
    if key in {"rtreferencewavelength", "rtrefwave"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("reference_wavelength_nm", DEFAULT_WVF_MEASURED_WAVELENGTH_NM))
    if key in {"rteffectivefocallength", "rtefl", "rteffectivefl"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("effective_focal_length_m", oi.fields["optics"].get("focal_length_m", np.nan))) * _spatial_unit_scale(args[0] if args else None)
    if key in {"raytraceopticsname", "psfopticsname"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict) and "optics_name" in psf_struct:
            return psf_struct["optics_name"]
        stored = oi.fields.get("psf_optics_name")
        if stored is not None:
            return str(stored)
        raytrace = oi.fields["optics"].get("raytrace", {})
        return raytrace.get("name", oi.fields["optics"].get("name"))
    if key in {"psfstruct", "shiftvariantstructure"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return _export_psf_struct(psf_struct)
        return None
    if key in {"svpsf", "sampledrtpsf", "shiftvariantpsf"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return _export_psf_cell_array(np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)), dtype=float))
        return None
    if key in {"psfsampleangles"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            angles = np.asarray(psf_struct.get("sample_angles_deg", np.empty(0, dtype=float)), dtype=float)
            if angles.size > 0:
                return angles
        stored = oi.fields.get("psf_sample_angles_deg")
        if stored is not None:
            return np.asarray(stored, dtype=float).reshape(-1)
        return _raytrace_default_sample_angles(float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG)))
    if key in {"psfanglestep"}:
        sample_angles = np.asarray(oi_get(oi, "psf sample angles"), dtype=float).reshape(-1)
        if sample_angles.size >= 2:
            return float(sample_angles[1] - sample_angles[0])
        return float(oi.fields.get("psf_angle_step_deg", DEFAULT_RAYTRACE_ANGLE_STEP_DEG))
    if key in {"psfimageheights"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]) if args else "m", 1.0)
            return np.asarray(psf_struct.get("img_height_mm", np.empty(0, dtype=float)), dtype=float) / 1e3 * scale
        stored = oi.fields.get("psf_image_heights_m")
        if stored is not None:
            scale = _SPATIAL_UNIT_SCALE.get(param_format(args[0]) if args else "m", 1.0)
            return np.asarray(stored, dtype=float).reshape(-1) * scale
        return np.empty(0, dtype=float)
    if key in {"psfwavelength"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return np.asarray(psf_struct.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float)
        stored = oi.fields.get("psf_wavelength_nm")
        if stored is not None:
            return np.asarray(stored, dtype=float).reshape(-1)
        return np.empty(0, dtype=float)
    if key in {"psfimageheightsn", "psfimageheightcount"}:
        return int(np.asarray(oi_get(oi, "psf image heights"), dtype=float).size)
    if key in {"psfwavelengthn", "psfwn"}:
        return int(np.asarray(oi_get(oi, "psf wavelength"), dtype=float).size)
    if key in {"psfspatialsupportx"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return np.asarray(psf_struct.get("target_x_mm", np.empty(0, dtype=float)), dtype=float)
        return np.empty(0, dtype=float)
    if key in {"psfspatialsupporty"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            return np.asarray(psf_struct.get("target_y_mm", np.empty(0, dtype=float)), dtype=float)
        return np.empty(0, dtype=float)
    if key in {"rtpsfsize", "rtpsfdimensions"}:
        psf_struct = oi.fields.get("psf_struct")
        if isinstance(psf_struct, dict):
            psf = np.asarray(psf_struct.get("psf", np.empty((0, 0, 0, 0, 0), dtype=float)), dtype=float)
            if psf.ndim == 5 and psf.shape[3] > 0 and psf.shape[4] > 0:
                return (int(psf.shape[3]), int(psf.shape[4]))
        return (0, 0)
    if key in {"rtobjectdistance", "rtobjdist", "rtrefobjdist", "rtreferenceobjectdistance"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("object_distance_m", np.inf)) * _spatial_unit_scale(
            args[0] if args else None
        )
    if key in {"rtblocksperfieldheight"}:
        return int(oi.fields.get("rt_blocks_per_field_height", oi.fields["optics"].get("raytrace", {}).get("blocks_per_field_height", 4)))
    if key in {"rtfieldofview", "rtfov", "rthorizontalfov", "rtmaximumfieldofview", "rtmaxfov"}:
        return float(oi.fields["optics"].get("raytrace", {}).get("max_fov_deg", np.inf))
    if key in {"rtcomputespacing"}:
        psf_spacing_m = oi.fields["optics"].get("raytrace", {}).get("computation", {}).get("psf_spacing_m")
        if psf_spacing_m is None:
            return None
        return float(psf_spacing_m) * _spatial_unit_scale(args[0] if args else None)
    if key in {"rtpsf"}:
        return _export_raytrace_table(_raw_raytrace_table(oi, "psf"), include_sample_spacing=True)
    if key in {"rtpsffunction", "rtpsfdata"}:
        field_height_m = float(args[0]) if len(args) >= 1 else None
        wavelength_nm = float(args[1]) if len(args) >= 2 else None
        return _raw_raytrace_psf_function(oi, field_height_m, wavelength_nm)
    if key in {"rtpsffieldheight"}:
        field_height_mm = np.asarray(_raw_raytrace_psf_data(oi).get("field_height_mm", np.empty(0, dtype=float)), dtype=float)
        return (field_height_mm / 1e3) * _spatial_unit_scale(args[0] if args else None)
    if key in {"rtpsfwavelength"}:
        return np.asarray(_raw_raytrace_psf_data(oi).get("wavelength_nm", np.empty(0, dtype=float)), dtype=float)
    if key in {"rtpsfsamplespacing", "rtpsfspacing"}:
        return _raw_raytrace_psf_spacing(oi, args[0] if args else None)
    if key in {"rtsupport", "rtpsfsupport"}:
        x_support, y_support = _raw_raytrace_psf_support_axes(oi, args[0] if args else None)
        if x_support.size == 0 or y_support.size == 0:
            return np.empty((0, 0, 2), dtype=float)
        xx, yy = np.meshgrid(x_support, y_support)
        return np.stack((xx, yy), axis=2)
    if key in {"rtpsfsupportrow", "rtpsfsupporty"}:
        _, y_support = _raw_raytrace_psf_support_axes(oi, args[0] if args else None)
        return y_support.reshape(-1, 1)
    if key in {"rtpsfsupportcol", "rtpsfsupportx"}:
        x_support, _ = _raw_raytrace_psf_support_axes(oi, args[0] if args else None)
        return x_support.reshape(-1)
    if key in {"rtfreqsupportcol", "rtfreqsupportx"}:
        fx, _ = _raw_raytrace_frequency_axes(oi, args[0] if args else None)
        return fx.reshape(-1)
    if key in {"rtfreqsupportrow", "rtfreqsupporty"}:
        _, fy = _raw_raytrace_frequency_axes(oi, args[0] if args else None)
        return fy.reshape(-1, 1)
    if key in {"rtfreqsupport"}:
        fx, fy = _raw_raytrace_frequency_axes(oi, args[0] if args else "mm")
        return {"fx": fx, "fy": fy}
    if key in {"rtrelillum"}:
        return _export_raytrace_table(_raw_raytrace_table(oi, "relative_illumination"))
    if key in {"rtrifunction", "rtrelativeilluminationfunction", "rtrelillumfunction"}:
        return np.asarray(_raw_raytrace_table(oi, "relative_illumination").get("function", np.empty(0, dtype=float)), dtype=float)
    if key in {"rtriwavelength", "rtrelativeilluminationwavelength"}:
        return np.asarray(_raw_raytrace_table(oi, "relative_illumination").get("wavelength_nm", np.empty(0, dtype=float)), dtype=float)
    if key in {"rtrifieldheight", "rtrelativeilluminationfieldheight"}:
        return _raw_raytrace_field_height(_raw_raytrace_table(oi, "relative_illumination"), args[0] if args else None)
    if key in {"rtgeometry"}:
        return _export_raytrace_table(_raw_raytrace_table(oi, "geometry"))
    if key in {"rtgeomfunction", "rtgeometryfunction", "rtdistortionfunction", "rtgeomdistortion"}:
        wavelength_nm = None
        unit = None
        if len(args) >= 1 and args[0] is not None:
            wavelength_nm = float(args[0])
        if len(args) >= 2:
            unit = args[1]
        elif len(args) == 1 and args[0] is None:
            unit = None
        return _raw_raytrace_geometry_function(oi, wavelength_nm, unit)
    if key in {"rtgeomwavelength", "rtgeometrywavelength"}:
        return np.asarray(_raw_raytrace_table(oi, "geometry").get("wavelength_nm", np.empty(0, dtype=float)), dtype=float)
    if key in {"rtgeomfieldheight", "rtgeometryfieldheight"}:
        return _raw_raytrace_field_height(_raw_raytrace_table(oi, "geometry"), args[0] if args else None)
    if key in {"rtgeommaxfieldheight", "rtmaximumfieldheight", "rtmaxfieldheight"}:
        field_height = np.asarray(oi_get(oi, "rtgeometryfieldheight"), dtype=float).reshape(-1)
        if field_height.size == 0:
            return 0.0
        return float(np.max(field_height)) * _spatial_unit_scale(args[0] if args else None)
    if key in {"transmittance", "transmittancescale", "lenstransmittance", "opticstransmittance", "opticstransmittancescale"}:
        target_wave = np.asarray(args[0], dtype=float).reshape(-1) if args else np.asarray(oi.fields["wave"], dtype=float)
        source = _optics_transmittance_payload(oi.fields["optics"], target_wave)
        source_wave = np.asarray(source["wave"], dtype=float).reshape(-1)
        source_scale = np.asarray(source["scale"], dtype=float).reshape(-1)
        if source_scale.size == 1 and source_wave.size > 1:
            source_scale = np.full(source_wave.size, float(source_scale[0]), dtype=float)
        if source_wave.size == 0:
            return np.empty(0, dtype=float)
        fill_value = 1.0 if "transmittance" in oi.fields["optics"] else float(source_scale[-1])
        return np.interp(target_wave, source_wave, source_scale, left=fill_value, right=fill_value)
    if key in {"transmittancewave", "opticstransmittancewave"}:
        transmittance = _optics_transmittance_payload(oi.fields["optics"], np.asarray(oi.fields["wave"], dtype=float))
        return np.asarray(transmittance["wave"], dtype=float).copy()
    if key in {"transmittancenwave", "opticstransmittancenwave"}:
        transmittance = _optics_transmittance_payload(oi.fields["optics"], np.asarray(oi.fields["wave"], dtype=float))
        return int(np.asarray(transmittance["wave"], dtype=float).size)
    if key in {"otf", "opticsotf", "otfstruct", "opticsotfstruct"}:
        otf_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        if otf_bundle is None:
            return None
        if key in {"otf", "opticsotf"}:
            return np.asarray(otf_bundle["OTF"], dtype=complex).copy()
        return _export_shift_invariant_otf_struct(
            {
                "function": oi.fields["optics"].get("otf_function", "custom"),
                "OTF": otf_bundle["OTF"],
                "fx": otf_bundle["fx"],
                "fy": otf_bundle["fy"],
                "wave": otf_bundle["wave"],
            }
        )
    if key in {"otfdata", "opticsotfdata"}:
        data = oi.fields["optics"].get("otf_data")
        return None if data is None else np.asarray(data).copy()
    if key in {"otffx", "opticsotffx"}:
        data = oi.fields["optics"].get("otf_fx")
        if data is not None:
            return np.asarray(data, dtype=float).copy()
        otf_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        return None if otf_bundle is None else np.asarray(otf_bundle["fx"], dtype=float).copy()
    if key in {"otffy", "opticsotffy"}:
        data = oi.fields["optics"].get("otf_fy")
        if data is not None:
            return np.asarray(data, dtype=float).copy()
        otf_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        return None if otf_bundle is None else np.asarray(otf_bundle["fy"], dtype=float).copy()
    if key in {"otfwave", "opticsotfwave"}:
        data = oi.fields["optics"].get("otf_wave")
        if data is not None:
            return np.asarray(data, dtype=float).copy()
        otf_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        return None if otf_bundle is None else np.asarray(otf_bundle["wave"], dtype=float).copy()
    if key in {"otffunction", "opticsotffunction"}:
        return oi.fields["optics"].get("otf_function")
    if key in {"psfxaxis"}:
        this_wave = args[0] if args else _default_plot_wavelength(oi.fields.get("wave", DEFAULT_WAVE))
        units = args[1] if len(args) >= 2 else "um"
        n_samp = int(args[2]) if len(args) >= 3 else 25
        return _oi_psf_axis(oi, "x", this_wave, units, n_samp=n_samp)
    if key in {"psfyaxis"}:
        this_wave = args[0] if args else _default_plot_wavelength(oi.fields.get("wave", DEFAULT_WAVE))
        units = args[1] if len(args) >= 2 else "um"
        n_samp = int(args[2]) if len(args) >= 3 else 25
        return _oi_psf_axis(oi, "y", this_wave, units, n_samp=n_samp)
    if key in {"psfdata", "opticspsfdata", "shiftinvariantpsfdata"}:
        if args:
            this_wave = args[0]
            units = args[1] if len(args) >= 2 else "um"
            n_samp = int(args[2]) if len(args) >= 3 else 25
            return _oi_psf_data(oi, this_wave, units, n_samp=n_samp)
        psf_data = oi.fields["optics"].get("psf_data")
        if isinstance(psf_data, dict):
            return _export_shift_invariant_psf_data(psf_data)
        return None
    if key in {"crop"}:
        return bool(oi.fields.get("crop", False))
    if key in {"padvalue"}:
        return oi.fields.get("pad_value", "zero")
    if key in {"paddingpixels"}:
        return tuple(oi.fields.get("padding_pixels", (0, 0)))
    raise KeyError(f"Unsupported oiGet parameter: {parameter}")


def oi_set(oi: OpticalImage, parameter: str, value: Any, *args: Any) -> OpticalImage:
    key = param_format(parameter)
    prefix, remainder = split_prefixed_parameter(parameter, ("optics", "wvf"))
    if prefix == "optics" and remainder:
        return oi_set(oi, remainder, value, *args)
    if prefix == "wvf" and remainder:
        wavefront = dict(oi.fields["optics"].get("wavefront", {}))
        updated_wvf = wvf_set(wavefront, remainder, value, *args)
        return _rebuild_oi_from_wvf(oi, updated_wvf)
    if key == "name":
        oi.name = str(value)
        _sync_oi_identity_fields(oi)
        return oi
    if key == "wave":
        oi.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        return oi
    if key == "illuminant":
        return _set_oi_illuminant(oi, value)
    if key == "illuminantphotons":
        return _set_oi_illuminant_photons(oi, value)
    if key == "illuminantcomment":
        oi.fields["illuminant_comment"] = str(value)
        return oi
    if key == "photons":
        oi.data["photons"] = np.asarray(value, dtype=float)
        _sync_oi_geometry_fields(oi)
        return oi
    if key == "illuminance":
        oi.fields["illuminance"] = np.asarray(value, dtype=float)
        return oi
    if key == "meanilluminance":
        oi.fields["mean_illuminance"] = float(value)
        return oi
    if key == "meancompilluminance":
        oi.fields["mean_comp_illuminance"] = float(value)
        return oi
    if key == "depthmap":
        depth_map = np.asarray(value, dtype=float)
        rows, cols = _oi_shape(oi)
        if rows > 0 and cols > 0 and depth_map.shape != (rows, cols):
            raise ValueError("Depth map must match the optical image size.")
        oi.fields["depth_map_m"] = depth_map
        return oi
    if key in {"wangular", "widthangular", "hfov", "horizontalfieldofview", "fov"}:
        oi.fields["fov_deg"] = float(value)
        _sync_oi_geometry_fields(oi)
        return oi
    if key in {"hangular", "heightangular", "vfov", "verticalfieldofview"}:
        oi.fields["vfov_deg"] = float(value)
        return oi
    if key in {"imagedistance", "focalplanedistance", "distance"}:
        oi.fields["image_distance_m"] = float(value)
        _sync_oi_geometry_fields(oi)
        return oi
    if key == "optics":
        oi.fields["optics"] = _normalize_optics_update(value, dict(oi.fields.get("optics", {})))
        oi.fields["compute_method"] = oi.fields["optics"].get("compute_method", oi.fields.get("compute_method"))
        _clear_precomputed_psf_state(oi)
        _sync_oi_geometry_fields(oi)
        return oi
    if key in {"focallength", "opticsfocallength"}:
        oi.fields["optics"]["focal_length_m"] = float(value)
        if oi.fields.get("image_distance_m") is None:
            oi.fields["image_distance_m"] = float(value)
        _sync_oi_geometry_fields(oi)
        return oi
    if key in {"fnumber", "opticsfnumber"}:
        oi.fields["optics"]["f_number"] = float(value)
        return oi
    if key in {"opticsmodel", "model"}:
        oi.fields["optics"]["model"] = str(value)
        return oi
    if key == "lens":
        oi.fields["optics"]["lens"] = _normalize_lens_payload(value, default_wave=np.asarray(oi.fields["wave"], dtype=float).reshape(-1))
        oi.fields["optics"].pop("transmittance", None)
        return oi
    if key in {"transmittance", "transmittancescale", "opticstransmittance", "opticstransmittancescale"}:
        scale = np.asarray(value, dtype=float).reshape(-1)
        transmittance = _optics_transmittance_payload(oi.fields["optics"], np.asarray(oi.fields["wave"], dtype=float))
        if scale.size != np.asarray(transmittance["wave"], dtype=float).size:
            raise ValueError("Transmittance must match wave dimension.")
        if np.any((scale < 0.0) | (scale > 1.0)):
            raise ValueError("Transmittance should be in [0, 1].")
        _set_optics_transmittance_payload(oi.fields["optics"], np.asarray(transmittance["wave"], dtype=float), scale)
        return oi
    if key in {"transmittancewave", "opticstransmittancewave"}:
        transmittance = _optics_transmittance_payload(oi.fields["optics"], np.asarray(oi.fields["wave"], dtype=float))
        old_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
        old_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
        new_wave = np.asarray(value, dtype=float).reshape(-1)
        _set_optics_transmittance_payload(oi.fields["optics"], new_wave, np.interp(new_wave, old_wave, old_scale))
        return oi
    if key in {"psfdata", "opticspsfdata", "shiftinvariantpsfdata"}:
        psf_data = _normalize_shift_invariant_psf_data(value)
        oi.fields["optics"]["psf_data"] = psf_data
        oi.fields["optics"].update(_custom_shift_invariant_otf_bundle(psf_data))
        oi.fields["optics"]["otf_function"] = "custom"
        oi.fields["optics"]["model"] = "shiftinvariant"
        oi.fields["optics"]["compute_method"] = "opticsotf"
        oi.fields["compute_method"] = "opticsotf"
        oi.fields["wave"] = np.asarray(psf_data["wave"], dtype=float).copy()
        transmittance = _optics_transmittance_payload(oi.fields["optics"], np.asarray(oi.fields["wave"], dtype=float))
        old_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
        old_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
        new_wave = np.asarray(oi.fields["wave"], dtype=float).reshape(-1)
        _set_optics_transmittance_payload(
            oi.fields["optics"],
            new_wave,
            np.interp(new_wave, old_wave, old_scale, left=1.0, right=1.0),
        )
        return oi
    if key in {"otf", "opticsotf"}:
        otf = np.asarray(value, dtype=complex)
        if otf.ndim == 2:
            otf = otf[:, :, None]
        if otf.ndim != 3:
            raise ValueError("Shift-invariant OTF data must be a 2-D or 3-D array.")

        wave = np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
        if otf.shape[2] == 1 and wave.size > 1:
            otf = np.repeat(otf, wave.size, axis=2)
        elif otf.shape[2] != wave.size:
            raise ValueError("Shift-invariant OTF wavelength dimension must match the wavelength vector.")

        current_bundle = _synthesized_shift_invariant_otf_bundle(oi)
        rows, cols = otf.shape[:2]
        if current_bundle is not None and np.asarray(current_bundle["OTF"]).shape[:2] == (rows, cols):
            fx = np.asarray(current_bundle["fx"], dtype=float).reshape(-1)
            fy = np.asarray(current_bundle["fy"], dtype=float).reshape(-1)
        else:
            fx, fy = _shift_invariant_otf_support(rows, cols, oi.fields.get("sample_spacing_m"))

        oi.fields["optics"]["otf_data"] = np.asarray(otf, dtype=complex)
        oi.fields["optics"]["otf_fx"] = fx.copy()
        oi.fields["optics"]["otf_fy"] = fy.copy()
        oi.fields["optics"]["otf_wave"] = wave.copy()
        oi.fields["optics"]["otf_function"] = "custom"
        oi.fields["optics"]["model"] = "shiftinvariant"
        oi.fields["optics"]["compute_method"] = "opticsotf"
        oi.fields["compute_method"] = "opticsotf"
        transmittance = _optics_transmittance_payload(oi.fields["optics"], wave)
        old_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
        old_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
        _set_optics_transmittance_payload(
            oi.fields["optics"],
            wave,
            np.interp(wave, old_wave, old_scale, left=1.0, right=1.0),
        )
        return oi
    if key in {"otfstruct", "opticsotfstruct"}:
        otf_struct = _normalize_shift_invariant_otf_struct(value, target_wave=np.asarray(oi.fields.get("wave", DEFAULT_WAVE), dtype=float))
        oi.fields["optics"]["otf_data"] = np.asarray(otf_struct["OTF"], dtype=complex)
        oi.fields["optics"]["otf_fx"] = np.asarray(otf_struct["fx"], dtype=float)
        oi.fields["optics"]["otf_fy"] = np.asarray(otf_struct["fy"], dtype=float)
        oi.fields["optics"]["otf_wave"] = np.asarray(otf_struct["wave"], dtype=float)
        oi.fields["optics"]["otf_function"] = str(otf_struct["function"])
        oi.fields["optics"]["model"] = "shiftinvariant"
        oi.fields["optics"]["compute_method"] = "opticsotf"
        oi.fields["compute_method"] = "opticsotf"
        oi.fields["wave"] = np.asarray(otf_struct["wave"], dtype=float).copy()
        transmittance = _optics_transmittance_payload(oi.fields["optics"], np.asarray(oi.fields["wave"], dtype=float))
        old_wave = np.asarray(transmittance["wave"], dtype=float).reshape(-1)
        old_scale = np.asarray(transmittance["scale"], dtype=float).reshape(-1)
        new_wave = np.asarray(oi.fields["wave"], dtype=float).reshape(-1)
        _set_optics_transmittance_payload(
            oi.fields["optics"],
            new_wave,
            np.interp(new_wave, old_wave, old_scale, left=1.0, right=1.0),
        )
        return oi
    if key in {"computemethod"}:
        oi.fields["compute_method"] = str(value)
        oi.fields["optics"]["compute_method"] = str(value)
        return oi
    if key in {"customcomputemethod"}:
        oi.fields["custom_compute_method"] = str(value)
        oi.fields["compute_method"] = str(value)
        oi.fields["optics"]["compute_method"] = str(value)
        return oi
    if key in {"customcompute"}:
        if bool(value):
            current = oi.fields.get("custom_compute_method") or oi.fields["optics"].get("compute_method") or "customCompute"
            oi.fields["custom_compute_method"] = str(current)
            oi.fields["compute_method"] = str(current)
            oi.fields["optics"]["compute_method"] = str(current)
        else:
            oi.fields.pop("custom_compute_method", None)
            default_method = oi.fields["optics"].get("compute_method", oi.fields.get("compute_method", "opticspsf"))
            if param_format(default_method) not in {"opticspsf", "opticsotf", "humanmw", "skip"}:
                default_method = "opticspsf"
            oi.fields["compute_method"] = str(default_method)
            oi.fields["optics"]["compute_method"] = str(default_method)
        return oi
    if key in {"diffusermethod"}:
        oi.fields["diffuser_method"] = str(value)
        return oi
    if key in {"diffuserblur"}:
        oi.fields["diffuser_blur_m"] = float(value)
        return oi
    if key in {"offaxismethod", "opticsoffaxismethod"}:
        oi.fields["optics"]["offaxis_method"] = str(value)
        return oi
    if key in {"opticswvf", "wvf", "wavefront"}:
        return _rebuild_oi_from_wvf(oi, dict(value))
    if key in {"psfstruct", "shiftvariantstructure"}:
        oi.fields["psf_struct"] = _normalize_psf_struct(value)
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"svpsf", "sampledrtpsf", "shiftvariantpsf"}:
        current = dict(oi.fields.get("psf_struct") or {})
        current["psf"] = _coerce_psf_stack(value)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"psfanglestep"}:
        oi.fields["psf_angle_step_deg"] = float(value)
        oi.fields["psf_sample_angles_deg"] = None
        oi.fields["psf_struct"] = None
        return oi
    if key in {"psfsampleangles"}:
        sample_angles = _raytrace_requested_sample_angles(DEFAULT_RAYTRACE_ANGLE_STEP_DEG, value)
        oi.fields["psf_angle_step_deg"] = float(sample_angles[1] - sample_angles[0])
        oi.fields["psf_sample_angles_deg"] = sample_angles
        oi.fields["psf_struct"] = None
        return oi
    if key in {"psfopticsname", "raytraceopticsname"}:
        oi.fields["psf_optics_name"] = str(value)
        current = dict(oi.fields.get("psf_struct") or {})
        current["optics_name"] = str(value)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"psfimageheights"}:
        heights_m = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_image_heights_m"] = heights_m.copy()
        current = dict(oi.fields.get("psf_struct") or {})
        current["img_height_mm"] = heights_m * 1e3
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"psfwavelength"}:
        oi.fields["psf_wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        current = dict(oi.fields.get("psf_struct") or {})
        current["wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        oi.fields["psf_struct"] = current
        _sync_psf_metadata_fields(oi)
        return oi
    if key in {"rtname"}:
        oi.fields["optics"].setdefault("raytrace", {})["name"] = str(value)
        oi.fields["optics"]["name"] = str(value)
        return oi
    if key in {"opticsprogram", "rtopticsprogram"}:
        oi.fields["optics"].setdefault("raytrace", {})["program"] = str(value)
        return oi
    if key in {"lensfile", "rtlensfile"}:
        oi.fields["optics"].setdefault("raytrace", {})["lens_file"] = str(value)
        return oi
    if key in {"rteffectivefnumber", "rtefff#"}:
        oi.fields["optics"].setdefault("raytrace", {})["effective_f_number"] = float(value)
        return oi
    if key in {"rtfnumber"}:
        oi.fields["optics"].setdefault("raytrace", {})["f_number"] = float(value)
        oi.fields["optics"]["f_number"] = float(value)
        return oi
    if key in {"rtmagnification", "rtmag"}:
        oi.fields["optics"].setdefault("raytrace", {})["magnification"] = float(value)
        return oi
    if key in {"rtreferencewavelength", "rtrefwave"}:
        oi.fields["optics"].setdefault("raytrace", {})["reference_wavelength_nm"] = float(value)
        return oi
    if key in {"rtobjectdistance", "rtobjdist", "rtrefobjdist", "rtreferenceobjectdistance"}:
        oi.fields["optics"].setdefault("raytrace", {})["object_distance_m"] = float(value)
        return oi
    if key in {"rtblocksperfieldheight"}:
        oi.fields["rt_blocks_per_field_height"] = int(value)
        oi.fields["optics"].setdefault("raytrace", {})["blocks_per_field_height"] = int(value)
        return oi
    if key in {"rtfieldofview", "rtfov", "rthorizontalfov", "rtmaximumfieldofview", "rtmaxfov"}:
        oi.fields["optics"].setdefault("raytrace", {})["max_fov_deg"] = float(value)
        return oi
    if key in {"rteffectivefocallength", "rtefl", "rteffectivefl"}:
        oi.fields["optics"].setdefault("raytrace", {})["effective_focal_length_m"] = float(value)
        oi.fields["optics"]["focal_length_m"] = float(value)
        return oi
    if key in {"rtcomputespacing"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("computation", {})["psf_spacing_m"] = float(value)
        return oi
    if key in {"rtpsfwavelength"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})["wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        _clear_precomputed_psf_state(oi)
        return oi
    if key in {"rtpsf"}:
        oi.fields["optics"].setdefault("raytrace", {})["psf"] = _normalize_raytrace_psf(value)
        _clear_precomputed_psf_state(oi)
        return oi
    if key in {"rtpsffunction", "rtpsfdata"}:
        psf = oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})
        if args:
            if len(args) < 2:
                raise ValueError("rtpsfdata indexed updates require field height and wavelength.")
            function = np.asarray(psf.get("function", np.empty(0, dtype=float)), dtype=float).copy()
            if function.ndim != 4:
                raise ValueError("rtpsfdata indexed updates require a 4-D raw PSF table.")
            field_heights_m = _raw_raytrace_field_height(psf)
            wavelengths_nm = np.asarray(psf.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
            field_index = _nearest_field_height_index(field_heights_m, float(args[0]))
            wave_index = _nearest_wave_index(wavelengths_nm, float(args[1]))
            function[:, :, field_index, wave_index] = np.asarray(value, dtype=float)
            psf["function"] = function
        else:
            psf["function"] = np.asarray(value, dtype=float)
        _clear_precomputed_psf_state(oi)
        return oi
    if key in {"rtpsffieldheight"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})["field_height_mm"] = np.asarray(value, dtype=float).reshape(-1)
        _clear_precomputed_psf_state(oi)
        return oi
    if key in {"rtpsfsamplespacing", "rtpsfspacing"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("psf", {})["sample_spacing_mm"] = np.asarray(value, dtype=float).reshape(-1)
        _clear_precomputed_psf_state(oi)
        return oi
    if key in {"rtrelillum"}:
        oi.fields["optics"].setdefault("raytrace", {})["relative_illumination"] = _normalize_raytrace_table(value)
        return oi
    if key in {"rtrifunction", "rtrelativeilluminationfunction", "rtrelillumfunction"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("relative_illumination", {})["function"] = np.asarray(value, dtype=float)
        return oi
    if key in {"rtriwavelength", "rtrelativeilluminationwavelength"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("relative_illumination", {})["wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        return oi
    if key in {"rtrifieldheight", "rtrelativeilluminationfieldheight"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("relative_illumination", {})["field_height_mm"] = np.asarray(value, dtype=float).reshape(-1)
        return oi
    if key in {"rtgeometry"}:
        oi.fields["optics"].setdefault("raytrace", {})["geometry"] = _normalize_raytrace_table(value)
        return oi
    if key in {"rtgeomfunction", "rtgeometryfunction", "rtdistortionfunction", "rtgeomdistortion"}:
        geometry = oi.fields["optics"].setdefault("raytrace", {}).setdefault("geometry", {})
        if args:
            function = np.asarray(geometry.get("function", np.empty(0, dtype=float)), dtype=float).copy()
            if function.ndim != 2:
                raise ValueError("rtgeomfunction indexed updates require a 2-D geometry table.")
            wavelengths_nm = np.asarray(geometry.get("wavelength_nm", np.empty(0, dtype=float)), dtype=float).reshape(-1)
            wave_index = _nearest_wave_index(wavelengths_nm, float(args[0]))
            function[:, wave_index] = np.asarray(value, dtype=float).reshape(-1)
            geometry["function"] = function
        else:
            geometry["function"] = np.asarray(value, dtype=float)
        return oi
    if key in {"rtgeomwavelength", "rtgeometrywavelength"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("geometry", {})["wavelength_nm"] = np.asarray(value, dtype=float).reshape(-1)
        return oi
    if key in {"rtgeomfieldheight", "rtgeometryfieldheight"}:
        oi.fields["optics"].setdefault("raytrace", {}).setdefault("geometry", {})["field_height_mm"] = np.asarray(value, dtype=float).reshape(-1)
        return oi
    if key in {"opticsraytrace", "raytrace", "rt"}:
        oi.fields["optics"] = _normalize_raytrace_update(value, dict(oi.fields.get("optics", {})))
        oi.fields["compute_method"] = oi.fields["optics"].get("compute_method", oi.fields.get("compute_method"))
        _clear_precomputed_psf_state(oi)
        _sync_oi_geometry_fields(oi)
        return oi
    raise KeyError(f"Unsupported oiSet parameter: {parameter}")


opticsBuild2Dotf = optics_build_2d_otf
opticsCoC = optics_coc
opticsDefocusCore = optics_defocus_core
opticsDefocusDisplacement = optics_defocus_displacement
opticsDepthDefocus = optics_depth_defocus
opticsDoF = optics_dof
opticsPSF2OTF = optics_psf_to_otf
opticsRayTrace = optics_ray_trace
oiBirefringentDiffuser = oi_birefringent_diffuser
oiCalculateIlluminance = oi_calculate_illuminance
oiCalculateOTF = oi_calculate_otf
oiCameraMotion = oi_camera_motion
oiClearData = oi_clear_data
oiCombineDepths = oi_combine_depths
oiCompute = oi_compute
oiCreate = oi_create
oiCrop = oi_crop
oiCustomCompute = oi_custom_compute
oiDepthCombine = oi_depth_combine
oiDepthSegmentMap = oi_depth_segment_map
oiDiffuser = oi_diffuser
oiFrequencyResolution = oi_frequency_resolution
oiGet = oi_get
oiPadDepthMap = oi_pad_depth_map
oiPreviewVideo = oi_preview_video
oiSaveImage = oi_save_image
oiSet = oi_set
oiShowImage = oi_show_image
oiSpace = oi_space
oiSpatialResample = oi_spatial_resample
oiSpatialSupport = oi_spatial_support
oiWBCompute = oi_wb_compute
wvf2oi = wvf_to_oi
wvf2optics = wvf_to_optics
wvf2PSF = wvf_to_psf
wvfAperture = wvf_aperture
wvfApertureP = wvf_aperture_params
wvfClearData = wvf_clear_data
wvfCompute = wvf_compute
wvfComputePSF = wvf_compute_psf
wvfCreate = wvf_create
wvfDefocusDioptersToMicrons = wvf_defocus_diopters_to_microns
wvfDefocusMicronsToDiopters = wvf_defocus_microns_to_diopters
wvfGet = wvf_get
wvfLoadThibosVirtualEyes = wvf_load_thibos_virtual_eyes
wvfPupilAmplitude = wvf_pupil_amplitude
wvfPupilFunction = wvf_pupil_function
wvfSet = wvf_set
rtAngleLUT = rt_angle_lut
rtBlockCenter = rt_block_center
rtChooseBlockSize = rt_choose_block_size
rtDIInterp = rt_di_interp
rtExtractBlock = rt_extract_block
rtFileNames = rt_file_names
rtFilteredBlockSupport = rt_filtered_block_support
rtGeometry = rt_geometry
rtImageRotate = rt_image_rotate
rtImportData = rt_import_data
rtInsertBlock = rt_insert_block
rtOTF = rt_otf
rtPSFApply = rt_psf_apply
rtPSFEdit = rt_psf_edit
rtPSFGrid = rt_psf_grid
rtPSFInterp = rt_psf_interp
rtPrecomputePSF = rt_precompute_psf
rtPrecomputePSFApply = rt_precompute_psf_apply
rtRIInterp = rt_ri_interp
rtRootPath = rt_root_path
rtSampleHeights = rt_sample_heights
rtSynthetic = rt_synthetic
airyDisk = airy_disk
ieFieldHeight2Index = ie_field_height_to_index
sceCreate = sce_create
sceGet = sce_get
siSynthetic = si_synthetic
zemaxLoad = zemax_load
zemaxReadHeader = zemax_read_header
