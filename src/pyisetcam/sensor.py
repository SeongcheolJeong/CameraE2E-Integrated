"""Sensor creation and computation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.io import savemat
from scipy.signal import convolve2d

from .assets import AssetStore, ie_read_color_filter, ie_read_spectra
from .color import luminance_from_photons, xyz_color_matching
from .exceptions import UnsupportedOptionError
from .fdtd_sensor import fdtd_sensor_apply_optical_response, fdtd_sensor_config, fdtd_sensor_qe_scale
from .tcad_sensor import tcad_sensor_apply_collection_response, tcad_sensor_config
from .fileio import vc_load_object
from .metrics import ie_cone_plot, xyz_from_energy
from .optics import DEFAULT_FOCAL_LENGTH_M
from .optics import oi_crop, oi_get
from .session import session_get_selected, session_replace_object, track_session_object
from .types import OpticalImage, Scene, Sensor, SessionContext
from .utils import DEFAULT_WAVE, blackbody, energy_to_quanta, ensure_multiple, ie_parameter_otype, linear_to_srgb, param_format, tile_pattern, xyz_to_linear_srgb
from .utils import image_increase_image_rgb_size

_DEFAULT_PIXEL = {
    "name": "aps",
    "type": "pixel",
    "size_m": np.array([2.8e-6, 2.8e-6], dtype=float),
    "fill_factor": 0.75,
    "layer_thickness_m": np.array([2.0e-6, 5.0e-6], dtype=float),
    "refractive_indices": np.array([1.0, 2.0, 1.46, 3.5], dtype=float),
    "spectrum": {},
    "conversion_gain_v_per_electron": 1.0e-4,
    "voltage_swing": 1.0,
    "dark_voltage_v_per_sec": 1.0e-3,
    "read_noise_v": 1.0e-3,
    "dsnu_sigma_v": 0.0,
    "prnu_sigma": 0.0,
}
_ELEMENTARY_CHARGE_C = 1.602177e-19
_PLANCK_CONSTANT_J_S = 6.62607015e-34
_LIGHT_SPEED_M_S = 2.99792458e8
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
_TIME_UNIT_SCALE = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e3,
    "msec": 1e3,
    "msecs": 1e3,
    "millisecond": 1e3,
    "milliseconds": 1e3,
    "us": 1e6,
    "usec": 1e6,
    "usecs": 1e6,
    "microsecond": 1e6,
    "microseconds": 1e6,
    "ns": 1e9,
    "nsec": 1e9,
    "nanosecond": 1e9,
    "nanoseconds": 1e9,
}
_SENSOR_FORMATS = {
    "qqcif": np.array([72.0, 88.0], dtype=float),
    "qcif": np.array([144.0, 176.0], dtype=float),
    "qqvga": np.array([120.0, 160.0], dtype=float),
    "qvga": np.array([240.0, 320.0], dtype=float),
    "cif": np.array([288.0, 352.0], dtype=float),
    "vga": np.array([480.0, 640.0], dtype=float),
    "svga": np.array([600.0, 800.0], dtype=float),
    "xvga": np.array([768.0, 1024.0], dtype=float),
    "uvga": np.array([1024.0, 1280.0], dtype=float),
    "uxvga": np.array([1200.0, 1600.0], dtype=float),
    "halfinch": np.array([0.0048, 0.0064], dtype=float),
    "quarterinch": np.array([0.0024, 0.0032], dtype=float),
    "sixteenthinch": np.array([0.0012, 0.0016], dtype=float),
}
_SENSOR_FORMAT_ALIASES = {
    "qqcif": "qqcif",
    "qcif": "qcif",
    "qqvga": "qqvga",
    "qvga": "qvga",
    "cif": "cif",
    "vga": "vga",
    "svga": "svga",
    "xvga": "xvga",
    "uvga": "uvga",
    "uxvga": "uxvga",
    "halfinch": "halfinch",
    "half": "halfinch",
    "quarterinch": "quarterinch",
    "quarter": "quarterinch",
    "sixteenthinch": "sixteenthinch",
    "sixteenth": "sixteenthinch",
}
_SENSOR_COLOR_ORDER = ("r", "g", "b", "c", "y", "m", "w", "i", "u", "x", "z", "o", "k")
_SENSOR_COLOR_MAP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.3, 0.3, 0.3],
        [0.4, 0.7, 0.3],
        [0.9, 0.6, 0.3],
        [0.2, 0.5, 0.8],
        [1.0, 0.6, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def sensor_formats(format_name: str | None = None) -> dict[str, np.ndarray] | np.ndarray:
    """Return MATLAB-style row/column sensor formats or physical sensor sizes."""

    if format_name is None or str(format_name).strip() == "":
        return {name: values.copy() for name, values in _SENSOR_FORMATS.items()}

    canonical = _SENSOR_FORMAT_ALIASES.get(param_format(format_name))
    if canonical is None:
        return {name: values.copy() for name, values in _SENSOR_FORMATS.items()}
    return _SENSOR_FORMATS[canonical].copy()


sensorFormats = sensor_formats


def sensor_color_filter(
    cf_type: str = "gaussian",
    wave: np.ndarray | list[float] | tuple[float, ...] | None = None,
    *args: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Create MATLAB-style sensor color filter curves."""

    normalized_type = param_format(cf_type or "gaussian")
    wave_array = (
        np.arange(400.0, 701.0, 1.0, dtype=float)
        if wave is None
        else np.asarray(wave, dtype=float).reshape(-1)
    )
    smooth = -1.0

    if normalized_type == "gaussian":
        c_pos = np.asarray(args[0], dtype=float).reshape(-1) if len(args) >= 1 else np.array([450.0, 550.0, 650.0], dtype=float)
        widths = np.asarray(args[1], dtype=float).reshape(-1) if len(args) >= 2 else np.full(c_pos.shape, 40.0, dtype=float)
        if widths.size == 1 and c_pos.size > 1:
            widths = np.full(c_pos.shape, float(widths[0]), dtype=float)
        if widths.size != c_pos.size:
            raise ValueError("Gaussian filter widths must match the number of center positions.")
        filters = np.zeros((wave_array.size, c_pos.size), dtype=float)
        for idx, (center, width) in enumerate(zip(c_pos, widths, strict=False)):
            filters[:, idx] = np.exp(-0.5 * ((wave_array - float(center)) / float(width)) ** 2)
    elif normalized_type == "block":
        c_pos = np.asarray(args[0], dtype=float).reshape(-1) if len(args) >= 1 else np.array([450.0, 550.0, 650.0], dtype=float)
        widths = np.asarray(args[1], dtype=float).reshape(-1) if len(args) >= 2 else np.full(c_pos.shape, 40.0, dtype=float)
        if widths.size == 1 and c_pos.size > 1:
            widths = np.full(c_pos.shape, float(widths[0]), dtype=float)
        if widths.size != c_pos.size:
            raise ValueError("Block filter widths must match the number of center positions.")
        filters = np.zeros((wave_array.size, c_pos.size), dtype=float)
        for idx, (center, width) in enumerate(zip(c_pos, widths, strict=False)):
            filters[:, idx] = np.abs(wave_array - float(center)) <= (float(width) / 2.0)
    elif normalized_type == "irfilter":
        cut = float(args[0]) if len(args) >= 1 else 700.0
        if len(args) >= 2:
            smooth = float(args[1])
        filters = np.ones((wave_array.size, 1), dtype=float)
        filters[wave_array > cut, 0] = 0.0
    elif normalized_type == "uvfilter":
        cut = float(args[0]) if len(args) >= 1 else 400.0
        if len(args) >= 2:
            smooth = float(args[1])
        filters = np.ones((wave_array.size, 1), dtype=float)
        filters[wave_array < cut, 0] = 0.0
    else:
        raise UnsupportedOptionError("sensorColorFilter", cf_type)

    if smooth > 0:
        filters = gaussian_filter(filters, sigma=float(smooth))

    return np.asarray(filters, dtype=float), wave_array


sensorColorFilter = sensor_color_filter


def _default_pixel(pixel: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_PIXEL)
    if pixel:
        merged.update(pixel)
    merged["size_m"] = np.asarray(merged["size_m"], dtype=float)
    merged["layer_thickness_m"] = np.asarray(merged.get("layer_thickness_m", np.array([], dtype=float)), dtype=float).reshape(-1)
    merged["refractive_indices"] = np.asarray(merged.get("refractive_indices", np.array([], dtype=float)), dtype=float).reshape(-1)
    stored_spectrum = merged.get("spectrum", {})
    if isinstance(stored_spectrum, dict):
        merged["spectrum"] = copy.deepcopy(stored_spectrum)
    elif stored_spectrum is None:
        merged["spectrum"] = {}
    else:
        merged["spectrum"] = copy.deepcopy(dict(vars(stored_spectrum)))
    return merged


def _is_empty_dispatch_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value).size == 0
    return False


def _empty_dispatch_placeholder_to_none(value: Any) -> Any:
    return None if _is_empty_dispatch_placeholder(value) else value


def _normalize_pixel_size_m(
    pixel_size_m: Any | None,
    *,
    default: tuple[float, float] | None = None,
) -> np.ndarray:
    if pixel_size_m is None or _is_empty_dispatch_placeholder(pixel_size_m):
        if default is None:
            raise ValueError("pixel size required")
        return np.asarray(default, dtype=float)

    size = np.asarray(pixel_size_m, dtype=float).reshape(-1)
    if size.size == 1:
        return np.array([float(size[0]), float(size[0])], dtype=float)
    return np.array([float(size[0]), float(size[1])], dtype=float)


def _sensor_base(
    name: str,
    wave: np.ndarray,
    size: tuple[int, int],
    pixel: dict[str, Any] | None,
) -> Sensor:
    sensor = Sensor(name=name)
    sensor.fields.update(
        {
            "wave": np.asarray(wave, dtype=float),
            "spectrum": {"wave": np.asarray(wave, dtype=float).copy()},
            "size": (int(size[0]), int(size[1])),
            "pixel": _default_pixel(pixel),
            "render": {"gamma": 1.0, "scale": False},
            "zero_level": 0.0,
            "analog_gain": 1.0,
            "analog_offset": 0.0,
            "nbits": 10,
            "noise_flag": 2,
            "reuse_noise": False,
            "noise_seed": 0,
            "response_type": "linear",
            "auto_exposure": True,
            "integration_time": 0.0,
            "quantization": "analog",
            "quantization_lut": None,
            "mosaic": True,
            "n_samples_per_pixel": 1,
            "vignetting": 0,
            "etendue": None,
            "fdtd_sensor": None,
            "tcad_sensor": None,
            "pixel_qe": np.ones(np.asarray(wave, dtype=float).size, dtype=float),
            "ir_filter": np.ones(np.asarray(wave, dtype=float).size, dtype=float),
        }
    )
    return sensor


def _spatial_unit_scale(unit: Any) -> float:
    if unit is None:
        return 1.0
    return _SPATIAL_UNIT_SCALE.get(param_format(unit), 1.0)


def _time_unit_scale(unit: Any) -> float:
    if unit is None:
        return 1.0
    return _TIME_UNIT_SCALE.get(param_format(unit), 1.0)


def _matlab_round_to_int(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    rounded = np.where(array >= 0.0, np.floor(array + 0.5), np.ceil(array - 0.5))
    return rounded.astype(int)


def _matlab_round_scalar(value: Any) -> int:
    return int(_matlab_round_to_int(np.asarray([value], dtype=float))[0])


def _pixel_size_m(pixel: dict[str, Any]) -> np.ndarray:
    size = np.asarray(pixel.get("size_m", np.array([0.0, 0.0], dtype=float)), dtype=float).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 2)
    if size.size < 2:
        raise ValueError("pixel size must have at least two elements.")
    return size[:2]


def _pixel_gaps_m(pixel: dict[str, Any]) -> np.ndarray:
    row_gap = pixel.get("height_gap_m", pixel.get("heightGapM", pixel.get("heightgap", pixel.get("height_gap", pixel.get("heightGap", 0.0)))))
    col_gap = pixel.get("width_gap_m", pixel.get("widthGapM", pixel.get("widthgap", pixel.get("width_gap", pixel.get("widthGap", 0.0)))))
    return np.array([float(row_gap), float(col_gap)], dtype=float)


def _sensor_spatial_resolution_m(sensor: Sensor) -> np.ndarray:
    pixel = sensor.fields["pixel"]
    return _pixel_size_m(pixel) + _pixel_gaps_m(pixel)


def _sensor_rows_cols(sensor: Sensor) -> tuple[int, int]:
    volts = sensor.data.get("volts")
    if volts is not None:
        shape = np.asarray(volts).shape
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    dv = sensor.data.get("dv")
    if dv is not None:
        shape = np.asarray(dv).shape
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    return int(sensor.fields["size"][0]), int(sensor.fields["size"][1])


def _sensor_unit_block(sensor: Sensor) -> tuple[int, int]:
    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    return int(pattern.shape[0]), int(pattern.shape[1])


def _sensor_filter_color_letters(sensor: Sensor) -> str:
    names = sensor.fields.get("filter_names", [])
    return "".join(str(name)[0].lower() if str(name) else "k" for name in names)


def _sensor_cfa_pattern(sensor: Sensor) -> np.ndarray:
    return np.asarray(sensor.fields["pattern"], dtype=int)


def _sensor_cfa_preset_pattern(value: Any) -> np.ndarray:
    normalized = param_format(str(value)).replace("_", "")
    presets = {
        "bayer": np.array([[1, 2], [2, 3]], dtype=int),
        "bayerrgb": np.array([[1, 2], [2, 3]], dtype=int),
        "bayerrggb": np.array([[1, 2], [2, 3]], dtype=int),
        "rggb": np.array([[1, 2], [2, 3]], dtype=int),
        "bayergrbg": np.array([[2, 1], [3, 2]], dtype=int),
        "grbg": np.array([[2, 1], [3, 2]], dtype=int),
        "bayerbggr": np.array([[3, 2], [2, 1]], dtype=int),
        "bggr": np.array([[3, 2], [2, 1]], dtype=int),
        "bayergbrg": np.array([[2, 3], [1, 2]], dtype=int),
        "gbrg": np.array([[2, 3], [1, 2]], dtype=int),
        "quadbayer": np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]],
            dtype=int,
        ),
        "quadbayerrgb": np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]],
            dtype=int,
        ),
        "quadbayerrggb": np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]],
            dtype=int,
        ),
        "quadbayerbggr": np.array(
            [[3, 3, 2, 2], [3, 3, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]],
            dtype=int,
        ),
        "quadbayergrbg": np.array(
            [[2, 2, 1, 1], [2, 2, 1, 1], [3, 3, 2, 2], [3, 3, 2, 2]],
            dtype=int,
        ),
        "quadbayergbrg": np.array(
            [[2, 2, 3, 3], [2, 2, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2]],
            dtype=int,
        ),
    }
    if normalized not in presets:
        raise ValueError(f"Unsupported CFA preset: {value!r}.")
    return presets[normalized].copy()


def _sensor_spectrum_struct(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("spectrum")
    if not isinstance(stored, dict):
        stored = {}
        sensor.fields["spectrum"] = stored
    spectrum = dict(stored)
    spectrum["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
    return spectrum


def _sensor_color_struct(sensor: Sensor) -> dict[str, Any]:
    return {
        "filterSpectra": np.asarray(sensor.fields["filter_spectra"], dtype=float).copy(),
        "filterNames": list(sensor.fields["filter_names"]),
        "irFilter": _sensor_ir_filter(sensor),
    }


def _sensor_chart_parameters(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("chartP")
    if not isinstance(stored, dict):
        stored = {}
        sensor.fields["chartP"] = stored
    chart = dict(stored)
    for key in ("cornerPoints", "rects", "currentRect"):
        if key in chart and chart[key] is not None:
            chart[key] = np.asarray(chart[key]).copy()
    return chart


def _sensor_integration_time_value(sensor: Sensor, unit: Any = None) -> Any:
    integration_time = sensor.fields.get("integration_time", 0.0)
    scale = _time_unit_scale(unit)
    if np.isscalar(integration_time):
        return float(integration_time) * scale
    return np.asarray(integration_time, dtype=float).copy() * scale


def _sensor_microlens(sensor: Sensor) -> dict[str, Any] | None:
    stored = sensor.fields.get("ml")
    if stored is None:
        return None
    return copy.deepcopy(stored)


def _sensor_movement(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("movement")
    if not isinstance(stored, dict):
        stored = {}
        sensor.fields["movement"] = stored
    movement = dict(stored)
    for key, entry in movement.items():
        if isinstance(entry, np.ndarray):
            movement[key] = entry.copy()
    return movement


def _sensor_human(sensor: Sensor) -> dict[str, Any] | None:
    stored = sensor.fields.get("human")
    if stored is None:
        return None
    return copy.deepcopy(stored)


def _sensor_column_fpn(sensor: Sensor) -> np.ndarray:
    stored = sensor.fields.get("column_fpn")
    if stored is None:
        return np.array([0.0, 0.0], dtype=float)
    return np.asarray(stored, dtype=float).reshape(-1).copy()


def _sensor_dynamic_range(sensor: Sensor, integration_time: Any = None) -> Any:
    if integration_time is None:
        integration_time = sensor_get(sensor, "integration time")
    integration_time_array = np.asarray(integration_time, dtype=float).reshape(-1)
    if integration_time_array.size == 0:
        return None
    if integration_time_array.size == 1 and np.isclose(integration_time_array[0], 0.0):
        return None
    integration_time_array = np.sort(integration_time_array)
    if integration_time_array.size > 1:
        integration_time_array = integration_time_array[[0, -1]]

    pixel = sensor.fields["pixel"]
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])
    conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-12)
    dsnu_sigma = float(pixel["dsnu_sigma_v"])
    voltage_swing = float(pixel["voltage_swing"])

    dark_variance = (dark_voltage * integration_time_array / conversion_gain) * (conversion_gain**2)
    read_variance = read_noise**2
    offset_variance = dsnu_sigma**2
    noise_sd = np.sqrt(dark_variance + read_variance + offset_variance)
    max_voltage = voltage_swing - (dark_voltage * integration_time_array)

    dr = np.full(noise_sd.shape, np.inf, dtype=float)
    positive = noise_sd > 0.0
    dr[positive] = 10.0 * np.log10(np.maximum(max_voltage[positive], 0.0) / noise_sd[positive])
    if dr.size == 1:
        return float(dr[0])
    return dr


def _apply_sensor_response_type(sensor: Sensor, volts: np.ndarray) -> np.ndarray:
    response_type = param_format(sensor.fields.get("response_type", "linear"))
    if response_type == "linear":
        return np.asarray(volts, dtype=float)
    if response_type != "log":
        raise UnsupportedOptionError("sensorCompute", f"response type {response_type}")

    read_noise = float(sensor.fields["pixel"]["read_noise_v"])
    if np.isclose(read_noise, 0.0):
        read_noise = float(sensor.fields["pixel"]["voltage_swing"]) / float(2**16)
    return np.log10(np.maximum(np.asarray(volts, dtype=float), 0.0) + read_noise) - np.log10(read_noise)


def _pixel_dynamic_range(sensor: Sensor, integration_time: Any = None) -> Any:
    if integration_time is None:
        integration_time = sensor_get(sensor, "integration time")
    integration_time_array = np.asarray(integration_time, dtype=float).reshape(-1)
    if integration_time_array.size == 0:
        return None
    if integration_time_array.size > 1:
        return None
    integration_time_value = float(integration_time_array[0])
    if np.isclose(integration_time_value, 0.0):
        return None

    pixel = sensor.fields["pixel"]
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])
    noise_sd = np.sqrt((dark_voltage * integration_time_value) + (read_noise**2))
    max_voltage = float(pixel["voltage_swing"]) - (dark_voltage * integration_time_value)
    if np.isclose(noise_sd, 0.0):
        return float(np.inf)
    return float(20.0 * np.log10(max(max_voltage, 0.0) / noise_sd))


def _sensor_plane_images(sensor: Sensor, data: np.ndarray | None, *, empty_value: float = np.nan) -> np.ndarray | None:
    if data is None:
        return None
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    rows, cols = array.shape[:2]
    pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), rows, cols)
    n_planes = int(sensor_get(sensor, "nfilters"))
    plane_stack = np.full((rows, cols, n_planes), float(empty_value), dtype=float)
    for index in range(1, n_planes + 1):
        mask = pattern == index
        plane_stack[:, :, index - 1][mask] = array[mask]
    return plane_stack


def _sensor_color_data(sensor: Sensor, data: np.ndarray | None, which_sensor: Any) -> np.ndarray | None:
    plane_stack = _sensor_plane_images(sensor, data)
    if plane_stack is None:
        return None
    sensor_index = int(np.rint(float(np.asarray(which_sensor).reshape(-1)[0])))
    if sensor_index < 1 or sensor_index > plane_stack.shape[2]:
        raise ValueError("Requested sensor channel is out of range.")
    channel = np.asarray(plane_stack[:, :, sensor_index - 1], dtype=float)
    return channel[~np.isnan(channel)]


def _pixel_pd_area_m2(sensor: Sensor) -> float:
    return float(np.prod(_pixel_pd_size_from_pixel(sensor.fields["pixel"])))


def _sensor_pd_size_m(sensor: Sensor) -> np.ndarray:
    return _pixel_pd_size_from_pixel(sensor.fields["pixel"])


def _pixel_spectrum_struct(sensor: Sensor) -> dict[str, Any]:
    pixel = sensor.fields["pixel"]
    stored = pixel.get("spectrum", {})
    if isinstance(stored, dict):
        spectrum = copy.deepcopy(stored)
    elif stored is None:
        spectrum = {}
    else:
        spectrum = copy.deepcopy(dict(vars(stored)))
    spectrum["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
    return spectrum


def _pixel_pd_position_from_pixel(pixel: dict[str, Any]) -> np.ndarray:
    stored = pixel.get("pd_position_m")
    if stored is not None:
        position = np.asarray(stored, dtype=float).reshape(-1)
        if position.size == 1:
            position = np.repeat(position, 2)
        return position[:2].copy()
    pixel_size = _pixel_size_m(pixel)
    pd_size = _pixel_pd_size_from_pixel(pixel)
    return ((pixel_size - pd_size) / 2.0).astype(float, copy=False)


def _pixel_pd_size_from_pixel(pixel: dict[str, Any]) -> np.ndarray:
    stored = pixel.get("pd_size_m")
    if stored is not None:
        pd_size = np.asarray(stored, dtype=float).reshape(-1)
        if pd_size.size == 1:
            pd_size = np.repeat(pd_size, 2)
        return pd_size[:2].copy()
    pixel_size = _pixel_size_m(pixel)
    return np.sqrt(float(pixel.get("fill_factor", 1.0))) * pixel_size[:2]


def _sync_pixel_pd_state(pixel: dict[str, Any]) -> None:
    pixel_size = _pixel_size_m(pixel)
    pd_size = _pixel_pd_size_from_pixel(pixel)
    pd_position = _pixel_pd_position_from_pixel(pixel)
    if np.any(pd_size < 0.0):
        raise ValueError("photodetector size must be nonnegative.")
    if np.any(pd_size - pixel_size > 1e-18):
        raise ValueError("photodetector size must not exceed the pixel size.")
    if np.any(pd_position < 0.0):
        raise ValueError("photodetector position must be nonnegative.")
    if np.any(pd_position + pd_size - pixel_size > 1e-18):
        raise ValueError("photodetector position must keep the photodetector inside the pixel.")
    pixel["pd_size_m"] = pd_size
    pixel["pd_position_m"] = pd_position
    pixel["fill_factor"] = float(np.prod(pd_size) / max(np.prod(pixel_size), 1e-30))


def _sensor_pixel_get(sensor: Sensor, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    pixel = sensor.fields["pixel"]
    spatial_scale = _spatial_unit_scale(args[0] if args else None)
    pixel_size = _pixel_size_m(pixel)
    pixel_gaps = _pixel_gaps_m(pixel)
    pixel_spacing = pixel_size + pixel_gaps
    pd_size = _sensor_pd_size_m(sensor)
    if key == "name":
        return str(pixel.get("name", ""))
    if key == "type":
        return str(pixel.get("type", "pixel"))
    if key in {"width", "pixelwidth", "pixelwidthmeters", "widthmeters"}:
        return float(pixel_size[1]) * spatial_scale
    if key in {"height", "pixelheight", "pixelheightmeters", "heightmeters"}:
        return float(pixel_size[0]) * spatial_scale
    if key in {"pixelwidthgap", "widthgap", "widthbetweenpixels"}:
        return float(pixel_gaps[1]) * spatial_scale
    if key in {"pixelheightgap", "heightgap", "heightbetweenpixels"}:
        return float(pixel_gaps[0]) * spatial_scale
    if key in {"wspatialresolution", "deltax"}:
        return float(pixel_spacing[1]) * spatial_scale
    if key in {"hspatialresolution", "deltay"}:
        return float(pixel_spacing[0]) * spatial_scale
    if key in {"xyspacing", "dimension"}:
        return np.array([pixel_spacing[1], pixel_spacing[0]], dtype=float) * spatial_scale
    if key in {"size", "pixelsize"}:
        return np.array([pixel_spacing[0], pixel_spacing[1]], dtype=float) * spatial_scale
    if key in {"pixelarea", "area"}:
        return float(np.prod(pixel_spacing) * (spatial_scale**2))
    if key == "fillfactor":
        return float(pixel["fill_factor"])
    if key in {"photodetectorwidth", "pdwidth"}:
        return float(pd_size[1]) * spatial_scale
    if key in {"photodetectorheight", "pdheight"}:
        return float(pd_size[0]) * spatial_scale
    if key in {"pdsize", "photodetectorsize"}:
        return pd_size * spatial_scale
    pd_position = _pixel_pd_position_from_pixel(pixel)
    if key in {"pdxpos", "photodetectorxposition"}:
        return float(pd_position[1]) * spatial_scale
    if key in {"pdypos", "photodetectoryposition"}:
        return float(pd_position[0]) * spatial_scale
    if key == "pdposition":
        return np.array([pd_position[1], pd_position[0]], dtype=float) * spatial_scale
    if key == "pddimension":
        return np.array([pd_size[1], pd_size[0]], dtype=float) * spatial_scale
    if key == "pdarea":
        return float(_pixel_pd_area_m2(sensor) * (spatial_scale**2))
    if key in {"layerthickness", "layerthicknesses"}:
        return np.asarray(pixel.get("layer_thickness_m", np.array([], dtype=float)), dtype=float).copy() * spatial_scale
    if key in {"pixeldepth", "depth", "pixeldepthmeters", "depthmeters", "stackheight"}:
        layer_thickness = np.asarray(pixel.get("layer_thickness_m", np.array([], dtype=float)), dtype=float).reshape(-1)
        return float(np.sum(layer_thickness)) * spatial_scale
    if key in {"refractiveindex", "refractiveindices", "n"}:
        return np.asarray(pixel.get("refractive_indices", np.array([], dtype=float)), dtype=float).copy()
    if key in {"spectrum", "pixelspectrum"}:
        return _pixel_spectrum_struct(sensor)
    if key in {"wave", "wavelength", "wavelengthsamples", "pixelwavelength", "pixelwavelengthsamples"}:
        return np.asarray(sensor.fields["wave"], dtype=float).copy()
    if key in {"binwidth", "wavelengthresolution", "pixelbinwidth"}:
        wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
        return float(wave[1] - wave[0]) if wave.size > 1 else 1.0
    if key in {"nwave", "nwaves", "numberofwavelengthsamples", "pixelnwave"}:
        return int(np.asarray(sensor.fields["wave"], dtype=float).size)
    if key in {
        "pdspectralqe",
        "spectralqe",
        "qe",
        "pixelspectralqe",
        "pixelqe",
        "quantumefficiency",
        "pixelquantumefficiency",
        "photodetectorquantumefficiency",
        "photodetectorspectralquantumefficiency",
    }:
        return _sensor_pixel_qe(sensor)
    if key in {"pdspectralsr", "spectralsr", "sr"}:
        return _pixel_spectral_sr(sensor)
    if key in {"pixeldr", "pixeldynamicrange", "dr", "dynamicrange"}:
        return _pixel_dynamic_range(sensor, args[0] if args else None)
    if key in {"conversiongain", "conversiongainvpelectron", "conversiongainvperelectron", "voltsperelectron"}:
        return float(pixel["conversion_gain_v_per_electron"])
    if key in {"voltageswing", "vswing", "saturationvoltage", "maxvoltage"}:
        return float(pixel["voltage_swing"])
    if key == "wellcapacity":
        conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-30)
        return float(pixel["voltage_swing"]) / conversion_gain
    if key in {"darkvolt", "darkvoltage", "darkvolts", "darkvoltageperpixelpersec", "darkvoltageperpixel", "voltspersecond"}:
        return float(pixel["dark_voltage_v_per_sec"])
    if key == "darkelectrons":
        conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-30)
        return float(pixel["dark_voltage_v_per_sec"]) / conversion_gain
    if key in {"darkcurrent", "darkcurrentperpixel"}:
        return float(_sensor_pixel_get(sensor, "darkelectrons")) * _ELEMENTARY_CHARGE_C
    if key == "darkcurrentdensity":
        return float(_sensor_pixel_get(sensor, "darkcurrent")) / max(_pixel_pd_area_m2(sensor), 1e-30)
    if key in {"readnoise", "readnoisevolts", "readstandarddeviationvolts", "readnoisestdvolts"}:
        return float(pixel["read_noise_v"])
    if key == "readnoisemillivolts":
        return float(pixel["read_noise_v"]) * 1e3
    if key in {"readnoiseelectrons", "readstandarddeviationelectrons"}:
        conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-30)
        return float(pixel["read_noise_v"]) / conversion_gain
    raise KeyError(f"Unsupported sensor pixel parameter: {parameter}")


def _sensor_pixel_set(sensor: Sensor, parameter: str, value: Any) -> Sensor:
    key = param_format(parameter)
    if key in {"pixel", "pixelfields"}:
        sensor.fields["pixel"] = _default_pixel(dict(value))
        sensor.fields["etendue"] = None
        return sensor
    pixel = sensor.fields["pixel"]
    pixel_size = _pixel_size_m(pixel)
    pixel_gaps = _pixel_gaps_m(pixel)
    pd_size = _sensor_pd_size_m(sensor)
    if key == "name":
        pixel["name"] = str(value)
        return sensor
    if key == "type":
        pixel["type"] = str(value)
        return sensor
    if key in {"width", "pixelwidth", "pixelwidthmeters"}:
        pixel["size_m"] = np.array([pixel_size[0], float(value)], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"height", "pixelheight", "pixelheightmeters"}:
        pixel["size_m"] = np.array([float(value), pixel_size[1]], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"pixelwidthgap", "widthgap", "widthbetweenpixels"}:
        pixel["width_gap_m"] = float(value)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"pixelheightgap", "heightgap", "heightbetweenpixels"}:
        pixel["height_gap_m"] = float(value)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"wspatialresolution", "deltax"}:
        width = float(value) - float(pixel_gaps[1])
        if width <= 0.0:
            raise ValueError("pixel width must stay positive after subtracting the width gap.")
        pixel["size_m"] = np.array([pixel_size[0], width], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"hspatialresolution", "deltay"}:
        height = float(value) - float(pixel_gaps[0])
        if height <= 0.0:
            raise ValueError("pixel height must stay positive after subtracting the height gap.")
        pixel["size_m"] = np.array([height, pixel_size[1]], dtype=float)
        if pixel.get("pd_size_m") is not None:
            _sync_pixel_pd_state(pixel)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {
        "sizeconstantfillfactor",
        "sizekeepfillfactor",
        "sizesamefillfactor",
        "pixelsizeconstantfillfactor",
        "pixelsizekeepfillfactor",
        "pixelsizesamefillfactor",
    }:
        size = np.asarray(value, dtype=float).reshape(-1)
        if size.size == 1:
            size = np.repeat(size, 2)
        current_size = np.array([pixel_size[0], pixel_size[1]], dtype=float)
        current_pd_size = _pixel_pd_size_from_pixel(sensor.fields["pixel"])
        stored_pd_position = sensor.fields["pixel"].get("pd_position_m")
        current_pd_position = (
            _pixel_pd_position_from_pixel(sensor.fields["pixel"])
            if stored_pd_position is not None
            else None
        )
        scale_factor = size[:2] / np.maximum(current_size, 1e-30)
        sensor.fields["pixel"]["size_m"] = size[:2].copy()
        sensor.fields["pixel"]["pd_size_m"] = current_pd_size * scale_factor
        if current_pd_position is not None:
            sensor.fields["pixel"]["pd_position_m"] = current_pd_position * scale_factor
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"widthheight", "widthandheight"}:
        size = np.asarray(value, dtype=float).reshape(-1)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor.fields["pixel"]["size_m"] = np.array([size[1], size[0]], dtype=float)
        if sensor.fields["pixel"].get("pd_size_m") is not None:
            _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"size", "pixelsize"}:
        size = np.asarray(value, dtype=float)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor.fields["pixel"]["size_m"] = size
        if sensor.fields["pixel"].get("pd_size_m") is not None:
            _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"photodetectorwidth", "pdwidth"}:
        sensor.fields["pixel"]["pd_size_m"] = np.array([pd_size[0], float(value)], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key in {"photodetectorheight", "pdheight"}:
        sensor.fields["pixel"]["pd_size_m"] = np.array([float(value), pd_size[1]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "fillfactor":
        sensor.fields["pixel"]["fill_factor"] = float(value)
        sensor.fields["pixel"].pop("pd_size_m", None)
        sensor.fields["pixel"].pop("pd_position_m", None)
        sensor.fields["etendue"] = None
        return sensor
    if key in {"pdsize", "photodetectorsize"}:
        pd_size = np.asarray(value, dtype=float).reshape(-1)
        if pd_size.size == 1:
            pd_size = np.repeat(pd_size, 2)
        sensor.fields["pixel"]["pd_size_m"] = pd_size[:2].copy()
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "pdwidthandheight":
        pd_size = np.asarray(value, dtype=float).reshape(-1)
        if pd_size.size == 1:
            pd_size = np.repeat(pd_size, 2)
        sensor.fields["pixel"]["pd_size_m"] = np.array([pd_size[1], pd_size[0]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key in {"pdxpos", "photodetectorxposition"}:
        pd_position = _pixel_pd_position_from_pixel(sensor.fields["pixel"])
        sensor.fields["pixel"]["pd_position_m"] = np.array([pd_position[0], float(value)], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key in {"pdypos", "photodetectoryposition"}:
        pd_position = _pixel_pd_position_from_pixel(sensor.fields["pixel"])
        sensor.fields["pixel"]["pd_position_m"] = np.array([float(value), pd_position[1]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "pdposition":
        pd_position = np.asarray(value, dtype=float).reshape(-1)
        if pd_position.size == 1:
            pd_position = np.repeat(pd_position, 2)
        sensor.fields["pixel"]["pd_position_m"] = np.array([pd_position[1], pd_position[0]], dtype=float)
        _sync_pixel_pd_state(sensor.fields["pixel"])
        sensor.fields["etendue"] = None
        return sensor
    if key == "pddimension":
        pd_dimension = np.asarray(value, dtype=float).reshape(-1)
        if pd_dimension.size == 1:
            pd_dimension = np.repeat(pd_dimension, 2)
        return _sensor_pixel_set(sensor, "pd size", np.array([pd_dimension[1], pd_dimension[0]], dtype=float))
    if key in {"layerthickness", "layerthicknesses"}:
        sensor.fields["pixel"]["layer_thickness_m"] = np.asarray(value, dtype=float).reshape(-1).copy()
        return sensor
    if key in {"refractiveindex", "refractiveindices", "n"}:
        sensor.fields["pixel"]["refractive_indices"] = np.asarray(value, dtype=float).reshape(-1).copy()
        return sensor
    if key in {"spectrum", "pixelspectrum"}:
        payload = dict(value) if isinstance(value, dict) else dict(vars(value))
        if "wave" in payload:
            sensor = _sensor_update_wave(sensor, np.asarray(payload["wave"], dtype=float).reshape(-1))
        payload["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
        sensor.fields["pixel"]["spectrum"] = copy.deepcopy(payload)
        return sensor
    if key in {"wave", "wavelength", "wavelengthsamples", "pixelwavelength", "pixelwavelengthsamples"}:
        sensor = _sensor_update_wave(sensor, np.asarray(value, dtype=float).reshape(-1))
        spectrum = _pixel_spectrum_struct(sensor)
        sensor.fields["pixel"]["spectrum"] = copy.deepcopy(spectrum)
        return sensor
    if key in {
        "pdspectralqe",
        "spectralqe",
        "qe",
        "pixelspectralqe",
        "pixelqe",
        "quantumefficiency",
        "pixelquantumefficiency",
        "photodetectorquantumefficiency",
        "photodetectorspectralquantumefficiency",
    }:
        qe = np.asarray(value, dtype=float).reshape(-1)
        if qe.size == 1:
            sensor.fields["pixel_qe"] = np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(qe[0]), dtype=float)
        elif qe.size == np.asarray(sensor.fields["wave"], dtype=float).size:
            sensor.fields["pixel_qe"] = qe
        else:
            raise ValueError("pixel spectral QE must match the sensor wavelength sampling.")
        return sensor
    if key in {"conversiongain", "conversiongainvpelectron", "conversiongainvperelectron", "voltsperelectron"}:
        sensor.fields["pixel"]["conversion_gain_v_per_electron"] = float(value)
        return sensor
    if key in {"voltageswing", "vswing", "saturationvoltage", "maxvoltage"}:
        sensor.fields["pixel"]["voltage_swing"] = float(value)
        return sensor
    if key in {"darkvolt", "darkvoltage", "darkvolts", "darkvoltageperpixelpersec", "darkvoltageperpixel", "voltspersecond"}:
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = float(value)
        return sensor
    if key in {"readnoise", "readnoiseelectrons", "readstandarddeviationelectrons"}:
        conversion_gain = max(float(sensor.fields["pixel"]["conversion_gain_v_per_electron"]), 1e-30)
        sensor.fields["pixel"]["read_noise_v"] = float(value) * conversion_gain
        return sensor
    if key in {"readnoisevolts", "readstandarddeviationvolts", "readnoisestdvolts"}:
        sensor.fields["pixel"]["read_noise_v"] = float(value)
        return sensor
    if key == "readnoisemillivolts":
        sensor.fields["pixel"]["read_noise_v"] = float(value) * 1e-3
        return sensor
    raise KeyError(f"Unsupported sensor pixel parameter: {parameter}")


def _copy_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _sensor_ocl_group_proxy_default() -> dict[str, Any]:
    return {
        "schema_version": "camerae2e_ocl_group_proxy_v1",
        "mode": "off",
        "shape": (1, 1),
        "equalization": 0.0,
        "readiness_tier": "proxy",
        "truth_boundary": (
            "Analytic shared-OCL aperture proxy. It equalizes spatial sampling "
            "within an OCL group per spectral/current plane before CFA selection; "
            "it is not an FDTD-calibrated OCL/DTI process-stack model."
        ),
    }


def _sensor_ocl_group_proxy(sensor: Sensor) -> dict[str, Any]:
    stored = sensor.fields.get("ocl_group_proxy")
    if not isinstance(stored, dict):
        return _sensor_ocl_group_proxy_default()
    proxy = dict(stored)
    proxy.setdefault("schema_version", "camerae2e_ocl_group_proxy_v1")
    proxy["shape"] = _sensor_ocl_group_shape_from_value(proxy.get("shape", (1, 1)))
    proxy["equalization"] = _sensor_ocl_group_equalization_from_value(
        proxy.get("equalization", 0.0)
    )
    normalized_mode = param_format(str(proxy.get("mode", "uniform")))
    if proxy["shape"] == (1, 1) or proxy["equalization"] <= 0.0:
        normalized_mode = "off"
    if normalized_mode not in {"off", "uniform"}:
        raise ValueError("OCL group proxy mode currently supports 'off' or 'uniform'.")
    proxy["mode"] = normalized_mode
    proxy.setdefault("readiness_tier", "proxy")
    proxy.setdefault(
        "truth_boundary",
        (
            "Analytic shared-OCL aperture proxy. It equalizes spatial sampling "
            "within an OCL group per spectral/current plane before CFA selection; "
            "it is not an FDTD-calibrated OCL/DTI process-stack model."
        ),
    )
    return proxy


def _sensor_ocl_group_proxy_from_value(value: Any) -> dict[str, Any]:
    if value is None:
        return _sensor_ocl_group_proxy_default()
    if isinstance(value, Mapping):
        shape_value = (
            value.get("shape")
            if "shape" in value
            else value.get("group_shape", value.get("ocl_group_shape", (1, 1)))
        )
        equalization_value = (
            value.get("equalization")
            if "equalization" in value
            else value.get("equalization_strength", value.get("strength", None))
        )
        shape = _sensor_ocl_group_shape_from_value(shape_value)
        equalization = (
            0.0
            if equalization_value is None and shape == (1, 1)
            else 1.0
            if equalization_value is None
            else _sensor_ocl_group_equalization_from_value(equalization_value)
        )
        mode = param_format(str(value.get("mode", "uniform" if equalization > 0.0 else "off")))
    else:
        shape = _sensor_ocl_group_shape_from_value(value)
        equalization = 0.0 if shape == (1, 1) else 1.0
        mode = "uniform" if equalization > 0.0 else "off"
    if mode not in {"off", "uniform"}:
        raise ValueError("OCL group proxy mode currently supports 'off' or 'uniform'.")
    return {
        "schema_version": "camerae2e_ocl_group_proxy_v1",
        "mode": mode,
        "shape": shape,
        "equalization": float(equalization),
        "readiness_tier": "proxy",
        "truth_boundary": (
            "Analytic shared-OCL aperture proxy. It equalizes spatial sampling "
            "within an OCL group per spectral/current plane before CFA selection; "
            "it is not an FDTD-calibrated OCL/DTI process-stack model."
        ),
    }


def _sensor_ocl_group_shape_from_value(value: Any) -> tuple[int, int]:
    if value is None:
        return (1, 1)
    if isinstance(value, str):
        normalized = param_format(value)
        if normalized in {"", "off", "none", "disabled", "false", "1", "1x1"}:
            return (1, 1)
        if normalized in {"quad", "quadbayer", "2", "2x2"}:
            return (2, 2)
        if "x" in normalized:
            parts = normalized.split("x", 1)
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                return (int(parts[0]), int(parts[1]))
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        rows = cols = int(np.rint(float(array[0])))
    elif array.size >= 2:
        rows = int(np.rint(float(array[0])))
        cols = int(np.rint(float(array[1])))
    else:
        rows = cols = 1
    if rows <= 0 or cols <= 0:
        raise ValueError("OCL group shape must contain positive integer dimensions.")
    return rows, cols


def _sensor_ocl_group_equalization_from_value(value: Any) -> float:
    equalization = float(np.asarray(value, dtype=float).reshape(-1)[0])
    if not np.isfinite(equalization) or not (0.0 <= equalization <= 1.0):
        raise ValueError("OCL group equalization must be a finite value in [0, 1].")
    return equalization


def _sensor_apply_ocl_group_proxy(
    sensor: Sensor, values: np.ndarray, *, sample_scale: int = 1
) -> np.ndarray:
    proxy = _sensor_ocl_group_proxy(sensor)
    if proxy["mode"] == "off":
        return np.asarray(values)
    equalization = float(proxy["equalization"])
    if equalization <= 0.0:
        return np.asarray(values)
    group_rows, group_cols = proxy["shape"]
    scale = max(int(sample_scale), 1)
    block_rows = max(int(group_rows) * scale, 1)
    block_cols = max(int(group_cols) * scale, 1)
    if block_rows <= 1 and block_cols <= 1:
        return np.asarray(values)

    source = np.asarray(values, dtype=float)
    if source.ndim < 2:
        return source
    result = source.copy()
    rows, cols = source.shape[:2]
    for row0 in range(0, rows, block_rows):
        row1 = min(row0 + block_rows, rows)
        for col0 in range(0, cols, block_cols):
            col1 = min(col0 + block_cols, cols)
            block = source[row0:row1, col0:col1, ...]
            block_mean = np.mean(block, axis=(0, 1), keepdims=True)
            result[row0:row1, col0:col1, ...] = (
                (1.0 - equalization) * block + equalization * block_mean
            )
    return result


def _spectrum_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        spectrum = dict(value)
    else:
        spectrum = {}
        if hasattr(value, "__dict__"):
            spectrum.update(vars(value))
    if "wave" not in spectrum:
        wave = getattr(value, "wave", None)
        if wave is None:
            raise ValueError("Spectrum value must include a wave field.")
        spectrum["wave"] = wave
    spectrum["wave"] = np.asarray(spectrum["wave"], dtype=float).reshape(-1)
    return spectrum


def _sensor_color_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        color = dict(value)
    else:
        color = {}
        if hasattr(value, "__dict__"):
            color.update(vars(value))
    return color


def _microlens_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    microlens = {}
    if hasattr(value, "__dict__"):
        microlens.update(vars(value))
    return copy.deepcopy(microlens)


def _human_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    human = {}
    if hasattr(value, "__dict__"):
        human.update(vars(value))
    return copy.deepcopy(human)


def _movement_struct_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        movement = dict(value)
    else:
        movement = {}
        if hasattr(value, "__dict__"):
            movement.update(vars(value))
    for key, entry in list(movement.items()):
        if isinstance(entry, np.ndarray):
            movement[key] = entry.copy()
    return movement


def _movement_positions_from_value(value: Any) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.size == 0:
        return positions.reshape(0, 2)
    if positions.ndim == 1:
        if positions.size != 2:
            raise ValueError("sensor movement positions must be Nx2.")
        return positions.reshape(1, 2)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("sensor movement positions must be Nx2.")
    return positions.copy()


def _sensor_unit_block_config(sensor: Sensor) -> np.ndarray:
    pixel = sensor.fields["pixel"]
    pixel_size = _pixel_size_m(pixel)
    pattern = _sensor_cfa_pattern(sensor)
    x, y = np.meshgrid(
        np.arange(pattern.shape[1], dtype=float) * pixel_size[1],
        np.arange(pattern.shape[0], dtype=float) * pixel_size[0],
    )
    return np.column_stack([x.ravel(), y.ravel()])


def _sensor_cfa_struct(sensor: Sensor) -> dict[str, Any]:
    pattern = _sensor_cfa_pattern(sensor)
    return {
        "pattern": pattern.copy(),
        "unitBlock": {
            "rows": int(pattern.shape[0]),
            "cols": int(pattern.shape[1]),
            "config": _sensor_unit_block_config(sensor),
        },
    }


def _sensor_cfa_name(sensor: Sensor) -> str:
    pattern = _sensor_cfa_pattern(sensor)
    filter_colors = "".join(sorted(_sensor_filter_color_letters(sensor)))
    if pattern.size == 1:
        return "Monochrome"
    if pattern.shape != (2, 2):
        return "Other"
    if filter_colors == "bgr":
        return "Bayer RGB"
    if filter_colors == "cmy":
        return "Bayer CMY"
    if filter_colors == "bgrw":
        return "RGBW"
    return "Other"


def _cfa_pattern_from_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "pattern" in value:
            return np.asarray(value["pattern"], dtype=int)
        if "cfapattern" in value:
            return np.asarray(value["cfapattern"], dtype=int)
    pattern = getattr(value, "pattern", None)
    if pattern is not None:
        return np.asarray(pattern, dtype=int)
    raise ValueError("CFA value must include a pattern.")


def _sensor_filter_display_colors(sensor: Sensor) -> np.ndarray:
    letters = list(_sensor_filter_color_letters(sensor))
    filter_spectra = _sensor_combined_qe(sensor, dtype=float)
    colors = np.zeros((filter_spectra.shape[1], 3), dtype=float)
    standard = {
        "r": np.array([1.0, 0.0, 0.0], dtype=float),
        "g": np.array([0.0, 1.0, 0.0], dtype=float),
        "b": np.array([0.0, 0.0, 1.0], dtype=float),
        "c": np.array([0.0, 1.0, 1.0], dtype=float),
        "m": np.array([1.0, 0.0, 1.0], dtype=float),
        "y": np.array([1.0, 1.0, 0.0], dtype=float),
        "w": np.array([1.0, 1.0, 1.0], dtype=float),
        "k": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    fallback_indices: list[int] = []
    for index in range(colors.shape[0]):
        letter = letters[index].lower() if index < len(letters) and letters[index] else ""
        mapped = standard.get(letter)
        if mapped is not None:
            colors[index] = mapped
        else:
            fallback_indices.append(index)

    if fallback_indices:
        wave = np.asarray(sensor.fields["wave"], dtype=float)
        xyz = np.asarray(
            filter_spectra[:, fallback_indices].T @ xyz_color_matching(wave, quanta=True),
            dtype=float,
        )
        linear_rgb = np.clip(xyz_to_linear_srgb(xyz), 0.0, None)
        row_max = np.max(linear_rgb, axis=1, keepdims=True)
        normalized = np.divide(
            linear_rgb,
            np.maximum(row_max, 1e-12),
            out=np.zeros_like(linear_rgb),
            where=row_max > 1e-12,
        )
        low_energy = np.ravel(row_max <= 1e-12)
        if np.any(low_energy):
            normalized[low_energy] = 1.0
        colors[fallback_indices] = linear_to_srgb(normalized)

    return np.clip(colors, 0.0, 1.0)


def _chart_roi(current_loc: Any, delta: Any) -> tuple[np.ndarray, np.ndarray]:
    from .roi import ie_rect2_locs

    current = np.asarray(current_loc, dtype=float).reshape(2)
    delta_value = float(np.asarray(delta, dtype=float).reshape(-1)[0])
    half_delta = _matlab_round_scalar(delta_value / 2.0)
    rect = np.array(
        [
            _matlab_round_scalar(current[1]) - half_delta,
            _matlab_round_scalar(current[0]) - half_delta,
            _matlab_round_scalar(delta_value),
            _matlab_round_scalar(delta_value),
        ],
        dtype=int,
    )
    return ie_rect2_locs(rect), rect


def _chart_rectangles(
    corner_points: Any,
    n_rows: int,
    n_cols: int,
    s_factor: float = 0.5,
    black_edge: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp = np.asarray(corner_points, dtype=float)
    if cp.shape != (4, 2):
        raise ValueError("Chart corner points must be a 4x2 array in [x, y] order.")

    chart_x = float(np.hypot(cp[3, 0] - cp[2, 0], cp[3, 1] - cp[2, 1]))
    chart_y = float(np.hypot(cp[3, 0] - cp[0, 0], cp[3, 1] - cp[0, 1]))
    p_size = _matlab_round_to_int(np.array([chart_y / float(n_rows), chart_x / float(n_cols)], dtype=float))

    m_locs = np.zeros((2, int(n_rows) * int(n_cols)), dtype=int)
    index = 0
    for col in range(1, int(n_cols) + 1):
        col_frac = float(col - 1) / float(n_cols)
        this_col = (cp[3, :] * (1.0 - col_frac)) + (cp[2, :] * col_frac) + 0.5
        this_col[0] = this_col[0] + (float(p_size[0]) / 2.0)
        for row in range(1, int(n_rows) + 1):
            row_frac = float(row - 1) / float(n_rows)
            this_row = (cp[3, :] * (1.0 - row_frac)) + (cp[0, :] * row_frac) + 0.5
            this_row[1] = this_row[1] + (float(p_size[1]) / 2.0)
            this_point = _matlab_round_to_int(this_col + this_row - cp[3, :])
            m_locs[:, index] = np.array([this_point[1], this_point[0]], dtype=int)
            index += 1

    if black_edge:
        delta = _matlab_round_to_int(p_size / 8.0)
        m_locs[0, :] = m_locs[0, :] - int(delta[0])
        m_locs[1, :] = m_locs[1, :] - int(delta[1])
        s_factor = float(s_factor) * 0.9

    rects = np.zeros((m_locs.shape[1], 4), dtype=int)
    rect_delta = float(p_size[0]) * float(s_factor)
    for index in range(m_locs.shape[1]):
        _, rect = _chart_roi(m_locs[:, index], rect_delta)
        rects[index, :] = rect

    return rects, m_locs, p_size


def _chart_rects_data(
    sensor: Sensor,
    m_locs: Any,
    delta: Any,
    *,
    full_data: bool = False,
    data_type: str = "volts",
) -> np.ndarray | list[np.ndarray]:
    from .roi import vc_get_roi_data

    locs = np.asarray(m_locs, dtype=float)
    if locs.ndim != 2 or locs.shape[0] != 2:
        raise ValueError("Chart midpoint locations must be a 2xN array in [row; col] order.")

    patch_data: list[np.ndarray] = []
    for index in range(locs.shape[1]):
        roi_locs, _ = _chart_roi(locs[:, index], delta)
        patch_data.append(np.asarray(vc_get_roi_data(sensor, roi_locs, data_type), dtype=float))

    if full_data:
        return patch_data

    n_filters = int(patch_data[0].shape[1]) if patch_data else int(sensor_get(sensor, "nfilters"))
    mean_rgb = np.zeros((locs.shape[1], n_filters), dtype=float)
    for index, data in enumerate(patch_data):
        for channel_index in range(n_filters):
            channel = np.asarray(data[:, channel_index], dtype=float)
            finite = np.isfinite(channel)
            mean_rgb[index, channel_index] = float(np.mean(channel[finite])) if np.any(finite) else np.nan
    return mean_rgb


def _macbeth_ideal_linear_rgb(
    wave: Any,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    wave_nm = np.asarray(wave, dtype=float).reshape(-1)
    _, reflectances = asset_store.load_reflectances("macbethChart.mat", wave_nm=wave_nm)
    _, illuminant_energy = asset_store.load_illuminant("D65.mat", wave_nm=wave_nm)
    color_signal = np.asarray(reflectances, dtype=float) * np.asarray(illuminant_energy, dtype=float).reshape(-1, 1)
    macbeth_xyz = xyz_from_energy(color_signal.T, wave_nm, asset_store=asset_store)
    macbeth_xyz = 100.0 * (macbeth_xyz / max(float(np.max(macbeth_xyz[:, 1])), 1e-12))
    return np.clip(xyz_to_linear_srgb(macbeth_xyz / 100.0), 0.0, 1.0)


def _color_block_matrix(wave_nm: np.ndarray, extrap_val: float = 0.0) -> np.ndarray:
    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
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
            matrix[:, index] = np.interp(
                wave,
                default_wave,
                default_matrix[:, index],
                left=float(extrap_val),
                right=float(extrap_val),
            )
    column_sums = np.sum(matrix, axis=0, keepdims=True)
    matrix = np.divide(
        matrix,
        np.maximum(column_sums, 1e-12),
        out=np.zeros_like(matrix),
        where=column_sums > 0.0,
    )
    white_spd = blackbody(wave, 6500.0, kind="quanta").reshape(-1)
    white_spd = white_spd / max(float(np.max(white_spd)), 1e-12)
    rgb = white_spd @ matrix
    matrix = matrix @ np.diag(1.0 / np.maximum(rgb, 1e-12))
    return matrix


def color_block_matrix(wave_nm: Any, extrap_val: float = 0.0) -> np.ndarray:
    """Return the legacy three-band display block matrix for the requested wave sampling."""

    return _color_block_matrix(np.asarray(wave_nm, dtype=float), float(extrap_val))


def _sensor_display_transform(sensor: Sensor) -> np.ndarray:
    block_matrix = _color_block_matrix(np.asarray(sensor.fields["wave"], dtype=float), extrap_val=0.2)
    filter_spectra = np.asarray(sensor_get(sensor, "filterspectra"), dtype=float)
    filter_rgb = block_matrix.T @ filter_spectra
    transform = filter_rgb.T
    scale = max(float(np.max(transform)), 1e-12)
    return transform / scale


def _image_linear_transform(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    rows, cols, channels = array.shape
    xw = array.reshape(rows * cols, channels)
    transformed = xw @ np.asarray(transform, dtype=float)
    return transformed.reshape(rows, cols, -1)


def _sensor_render_state(sensor: Sensor) -> dict[str, Any]:
    render = sensor.fields.get("render")
    if not isinstance(render, dict):
        render = {}
        sensor.fields["render"] = render
    render.setdefault("gamma", 1.0)
    render.setdefault("scale", False)
    return render


def _sensor_pixel_qe(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    stored = sensor.fields.get("pixel_qe")
    if stored is None:
        return np.ones(np.asarray(sensor.fields["wave"], dtype=float).size, dtype=dtype)
    qe = np.asarray(stored, dtype=dtype).reshape(-1)
    if qe.size == 1:
        return np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(qe[0]), dtype=dtype)
    return qe


def _sensor_ir_filter(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    stored = sensor.fields.get("ir_filter")
    if stored is None:
        return np.ones(np.asarray(sensor.fields["wave"], dtype=float).size, dtype=dtype)
    ir_filter = np.asarray(stored, dtype=dtype).reshape(-1)
    if ir_filter.size == 1:
        return np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(ir_filter[0]), dtype=dtype)
    return ir_filter


def _sensor_combined_qe(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    filter_spectra = np.asarray(sensor.fields["filter_spectra"], dtype=dtype)
    if filter_spectra.ndim == 1:
        filter_spectra = filter_spectra.reshape(-1, 1)
    pixel_qe = _sensor_pixel_qe(sensor, dtype=dtype).reshape(-1, 1)
    ir_filter = _sensor_ir_filter(sensor, dtype=dtype).reshape(-1, 1)
    return filter_spectra * pixel_qe * ir_filter


def _sensor_combined_sr(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    wave_m = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1, 1) * 1e-9
    conversion = (wave_m * _ELEMENTARY_CHARGE_C) / (_PLANCK_CONSTANT_J_S * _LIGHT_SPEED_M_S)
    return (conversion * _sensor_combined_qe(sensor, dtype=dtype)).astype(dtype, copy=False)


def _pixel_spectral_sr(sensor: Sensor, *, dtype: Any = float) -> np.ndarray:
    wave_m = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1) * 1e-9
    pixel_qe = _sensor_pixel_qe(sensor, dtype=dtype).reshape(-1)
    return ((wave_m * _ELEMENTARY_CHARGE_C) / (_PLANCK_CONSTANT_J_S * _LIGHT_SPEED_M_S) * pixel_qe).astype(dtype, copy=False)


def _sensor_rgb_source(sensor: Sensor, data_type: str) -> tuple[np.ndarray | None, str]:
    key = param_format(data_type)
    resolved = key
    if key in {"dvorvolts", "digitalorvolts"}:
        if sensor.data.get("dv") is not None:
            resolved = "dv"
            source = sensor.data.get("dv")
        else:
            resolved = "volts"
            source = sensor.data.get("volts")
    elif key in {"dv", "digitalvalue", "digitalvalues"}:
        source = sensor.data.get("dv")
    elif key == "electrons":
        source = _sensor_electrons(sensor)
    else:
        resolved = "volts"
        source = sensor.data.get("volts")

    if source is None:
        return None, resolved

    array = np.asarray(source, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    return np.asarray(array, dtype=float), resolved


def _sensor_display_scale(sensor: Sensor, data: np.ndarray, data_type: str, *, scale_max: bool) -> float:
    if scale_max:
        return float(max(np.max(np.asarray(data, dtype=float)), 1e-12))
    key = param_format(data_type)
    if key in {"dv", "digitalvalue", "digitalvalues"}:
        return float(max(sensor_get(sensor, "max digital value"), 1.0))
    return float(max(sensor_get(sensor, "max output"), 1e-12))


def _sensor_rgb_image(
    sensor: Sensor,
    data_type: str = "volts",
    gamma: float | None = None,
    scale_max: bool | None = None,
) -> np.ndarray | None:
    data, resolved_type = _sensor_rgb_source(sensor, data_type)
    if data is None:
        return None

    gamma_value = float(sensor_get(sensor, "gamma") if gamma is None else gamma)
    scale_max_value = bool(sensor_get(sensor, "scale max") if scale_max is None else scale_max)
    normalized = np.clip(
        np.asarray(data, dtype=float) / _sensor_display_scale(sensor, data, resolved_type, scale_max=scale_max_value),
        0.0,
        1.0,
    )

    if int(sensor_get(sensor, "nfilters")) <= 1:
        return np.power(normalized, gamma_value)

    linear = _sensor_plane_images(sensor, np.asarray(data, dtype=float), empty_value=0.0)
    if linear is None:
        return None
    filter_letters = str(sensor_get(sensor, "filtercolorletters")).lower()
    if filter_letters == "rgb":
        transformed = linear
    elif filter_letters == "wrgb":
        transformed = _image_linear_transform(
            linear,
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
        )
    elif filter_letters == "rgbw":
        transformed = _image_linear_transform(
            linear,
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=float,
            ),
        )
    else:
        transformed = _image_linear_transform(linear, _sensor_display_transform(sensor))
    transformed = np.clip(transformed / _sensor_display_scale(sensor, data, resolved_type, scale_max=scale_max_value), 0.0, 1.0)
    transformed = np.power(transformed, gamma_value)
    if transformed.shape[2] == 3:
        return linear_to_srgb(transformed)
    return transformed


def _interp_spectral_array(old_wave: np.ndarray, new_wave: np.ndarray, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(new_wave.shape, float(array), dtype=float)
    if array.ndim == 1:
        if array.size == 1:
            return np.full(new_wave.shape, float(array[0]), dtype=float)
        return np.interp(new_wave, old_wave, array, left=0.0, right=0.0)
    if array.shape[0] == 1:
        return np.repeat(array.astype(float), new_wave.size, axis=0)
    interpolated = np.empty((new_wave.size, array.shape[1]), dtype=float)
    for index in range(array.shape[1]):
        interpolated[:, index] = np.interp(new_wave, old_wave, array[:, index], left=0.0, right=0.0)
    return interpolated


def _sensor_update_wave(sensor: Sensor, new_wave: np.ndarray) -> Sensor:
    old_wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
    new_wave = np.asarray(new_wave, dtype=float).reshape(-1)

    if old_wave.size > 0 and not np.array_equal(old_wave, new_wave):
        filter_spectra = np.asarray(sensor.fields.get("filter_spectra"), dtype=float)
        if filter_spectra.ndim >= 1 and filter_spectra.shape[0] == old_wave.size:
            sensor.fields["filter_spectra"] = _interp_spectral_array(old_wave, new_wave, filter_spectra)

        pixel_qe = np.asarray(sensor.fields.get("pixel_qe"), dtype=float).reshape(-1)
        if pixel_qe.size == old_wave.size:
            sensor.fields["pixel_qe"] = _interp_spectral_array(old_wave, new_wave, pixel_qe)

        ir_filter = np.asarray(sensor.fields.get("ir_filter"), dtype=float).reshape(-1)
        if ir_filter.size == old_wave.size:
            sensor.fields["ir_filter"] = _interp_spectral_array(old_wave, new_wave, ir_filter)

    sensor.fields["wave"] = new_wave
    spectrum = _sensor_spectrum_struct(sensor)
    spectrum["wave"] = new_wave.copy()
    sensor.fields["spectrum"] = spectrum
    return sensor


def _sensor_aligned_dimension(value: Any, block_size: int) -> int:
    dimension = int(np.floor(float(value)))
    if block_size <= 1:
        return dimension
    return int(np.floor(dimension / block_size) * block_size)


def _sensor_clear_data(sensor: Sensor) -> None:
    sensor.data.clear()


def _sensor_electrons(sensor: Sensor) -> np.ndarray | None:
    volts = sensor.data.get("volts")
    if volts is None:
        return None
    analog_gain = float(sensor.fields["analog_gain"])
    analog_offset = float(sensor.fields["analog_offset"])
    conversion_gain = float(sensor.fields["pixel"]["conversion_gain_v_per_electron"])
    return np.clip((np.asarray(volts, dtype=float) * analog_gain) - analog_offset, 0.0, None) / max(
        conversion_gain, 1e-12
    )


def _filter_bundle(
    filter_name: str | list[str] | tuple[str, ...],
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(filter_name, (list, tuple)):
        spectra_parts: list[np.ndarray] = []
        names: list[str] = []
        for current_filter in filter_name:
            _, current_spectra, current_names = asset_store.load_color_filters(current_filter, wave_nm=wave)
            current_spectra = np.asarray(current_spectra, dtype=float)
            if current_spectra.ndim == 1:
                current_spectra = current_spectra.reshape(-1, 1)
            spectra_parts.append(current_spectra)
            names.extend(current_names)
        return np.concatenate(spectra_parts, axis=1), names
    _, spectra, names = asset_store.load_color_filters(filter_name, wave_nm=wave)
    return np.asarray(spectra, dtype=float), names


def _matlab_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in np.atleast_1d(value)]


def _sensor_from_upstream_model(
    relative_path: str,
    *,
    asset_store: AssetStore,
) -> Sensor:
    data = asset_store.load_mat(relative_path)
    model = data["sensor"]
    wave = np.asarray(model.spectrum.wave, dtype=float).reshape(-1)
    size = (int(model.rows), int(model.cols))
    pixel_width = float(model.pixel.width)
    pixel_height = float(model.pixel.height)
    pd_width = float(getattr(model.pixel, "pdWidth", pixel_width))
    pd_height = float(getattr(model.pixel, "pdHeight", pixel_height))
    fill_factor = float((pd_width * pd_height) / max(pixel_width * pixel_height, 1e-12))
    pixel = {
        "size_m": np.array([pixel_height, pixel_width], dtype=float),
        "height_gap_m": float(getattr(model.pixel, "heightGap", 0.0)),
        "width_gap_m": float(getattr(model.pixel, "widthGap", 0.0)),
        "fill_factor": fill_factor,
        "conversion_gain_v_per_electron": float(model.pixel.conversionGain),
        "voltage_swing": float(model.pixel.voltageSwing),
        "dark_voltage_v_per_sec": float(model.pixel.darkVoltage),
        "read_noise_v": float(model.pixel.readNoise),
    }
    sensor = _sensor_base(str(model.name), wave, size, pixel)
    pattern = np.asarray(model.cfa.pattern, dtype=int)
    if pattern.ndim == 0:
        pattern = pattern.reshape(1, 1)
    filter_spectra = np.asarray(model.color.filterSpectra, dtype=float)
    if filter_spectra.ndim == 1:
        filter_spectra = filter_spectra.reshape(-1, 1)
    sensor.fields["pattern"] = pattern
    sensor.fields["filter_spectra"] = filter_spectra
    sensor.fields["filter_names"] = _matlab_string_list(model.color.filterNames)
    sensor.fields["pixel_qe"] = np.asarray(getattr(model.pixel, "spectralQE", np.ones(wave.size)), dtype=float).reshape(-1)
    sensor.fields["ir_filter"] = np.asarray(getattr(model.color, "irFilter", np.ones(wave.size)), dtype=float).reshape(-1)
    sensor.fields["analog_gain"] = float(getattr(model, "analogGain", 1.0))
    sensor.fields["analog_offset"] = float(getattr(model, "analogOffset", 0.0))
    quantization = getattr(model, "quantization", "analog")
    if hasattr(quantization, "bits") and hasattr(quantization, "method"):
        bits_value = np.asarray(quantization.bits, dtype=float).reshape(-1)
        bits = int(np.rint(float(bits_value[0]))) if bits_value.size else 0
        method = str(quantization.method)
        if param_format(method) == "linear" and bits > 0:
            sensor.fields["quantization"] = f"{bits} bit"
        else:
            sensor.fields["quantization"] = method
        sensor.fields["nbits"] = bits
    else:
        sensor.fields["quantization"] = str(quantization)
    sensor.fields["integration_time"] = float(getattr(model, "integrationTime", 0.0))
    sensor.fields["auto_exposure"] = bool(getattr(model, "AE", False))
    sensor.fields["noise_flag"] = int(getattr(model, "noiseFlag", 0))
    sensor.fields["cds"] = bool(getattr(model, "CDS", False))
    if hasattr(model, "sigmaOffsetFPN"):
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(model.sigmaOffsetFPN)
    if hasattr(model, "sigmaGainFPN"):
        sensor.fields["pixel"]["prnu_sigma"] = float(model.sigmaGainFPN)
    if hasattr(model, "offsetFPNimage") and np.size(model.offsetFPNimage):
        sensor.fields["offset_fpn_image"] = np.asarray(model.offsetFPNimage, dtype=float).copy()
    if hasattr(model, "gainFPNimage") and np.size(model.gainFPNimage):
        sensor.fields["gain_fpn_image"] = np.asarray(model.gainFPNimage, dtype=float).copy()
    return sensor


def _sensor_variant_names(args: tuple[Any, ...], default: str) -> list[str]:
    if not args:
        return [default]
    variant = args[0]
    if isinstance(variant, (list, tuple, np.ndarray)):
        values = np.asarray(variant, dtype=object).reshape(-1)
        if values.size == 0:
            return [default]
        return [str(value) for value in values]
    return [str(variant)]


def _sensor_variant_name(args: tuple[Any, ...], default: str) -> str:
    return _sensor_variant_names(args, default)[0]


def _human_cone_mosaic(
    sz: Any,
    densities: Any | None = None,
    um_cone_width: float = 2.0,
    r_seed: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    size = np.asarray(sz, dtype=int).reshape(-1)
    if size.size != 2:
        raise ValueError("humanConeMosaic requires a [rows, cols] size.")
    rows, cols = int(size[0]), int(size[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("humanConeMosaic requires positive rows and cols.")

    density_array = np.asarray(
        [0.1, 0.55, 0.25, 0.1] if densities is None else densities,
        dtype=float,
    ).reshape(-1)
    if density_array.size == 3:
        density_sum = float(np.sum(density_array, dtype=float))
        if density_sum >= 1.0:
            density_array = np.concatenate(([0.0], density_array / max(density_sum, 1.0e-12)))
        else:
            density_array = np.concatenate(([1.0 - density_sum], density_array))
    elif density_array.size != 4:
        raise ValueError("humanConeMosaic requires three LMS or four K/L/M/S densities.")
    density_array = density_array / max(float(np.sum(density_array, dtype=float)), 1.0e-12)

    seed_value = 0 if r_seed is None else int(np.asarray(r_seed).reshape(-1)[0])
    rng = np.random.default_rng(seed_value)

    n_locs = rows * cols
    n_receptors = np.rint(density_array * float(n_locs)).astype(int)
    difference = int(n_locs - int(np.sum(n_receptors)))
    if difference != 0:
        max_index = int(np.argmax(n_receptors))
        n_receptors[max_index] += difference

    cone_type_vector = np.zeros(n_locs, dtype=int)
    start = 0
    for index, count in enumerate(n_receptors, start=1):
        stop = start + int(max(count, 0))
        cone_type_vector[start:stop] = index
        start = stop
    cone_type_vector = cone_type_vector[rng.permutation(n_locs)]
    cone_type = cone_type_vector.reshape(rows, cols)

    x = (np.arange(cols, dtype=float) + 1.0) * float(um_cone_width)
    x -= np.mean(x, dtype=float)
    y = (np.arange(rows, dtype=float) + 1.0) * float(um_cone_width)
    y -= np.mean(y, dtype=float)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

    return xy, cone_type, density_array, seed_value


def human_cone_mosaic(
    sz: Any,
    densities: Any | None = None,
    um_cone_width: Any | None = None,
    r_seed: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Create the standalone human cone-mosaic payload."""

    densities = _empty_dispatch_placeholder_to_none(densities)
    um_cone_width = _empty_dispatch_placeholder_to_none(um_cone_width)
    r_seed = _empty_dispatch_placeholder_to_none(r_seed)

    return _human_cone_mosaic(
        sz,
        densities=densities,
        um_cone_width=2.0 if um_cone_width is None else float(np.asarray(um_cone_width).reshape(-1)[0]),
        r_seed=r_seed,
    )


def sensor_cone_plot(
    sensor: Sensor,
    support: Any | None = None,
    spread: Any | None = None,
    delta: Any | None = None,
) -> dict[str, Any]:
    """Headless sensorConePlot() compatibility wrapper."""

    xy = sensor_get(sensor, "human cone locs")
    cone_type = sensor_get(sensor, "cone type")

    resolved_support = None if _is_empty_dispatch_placeholder(support) else support
    resolved_spread = None if _is_empty_dispatch_placeholder(spread) else float(np.asarray(spread, dtype=float).reshape(-1)[0])
    resolved_delta = None if _is_empty_dispatch_placeholder(delta) else float(np.asarray(delta, dtype=float).reshape(-1)[0])

    if xy is None or np.asarray(xy).size == 0:
        _, image = sensor_show_cfa(sensor)
        support_payload = None
        if resolved_support is not None:
            support_array = np.asarray(resolved_support, dtype=int).reshape(-1)
            support_payload = np.repeat(support_array, 2) if support_array.size == 1 else support_array
        return {
            "support": support_payload,
            "spread": resolved_spread,
            "delta": resolved_delta,
            "grid": None,
            "image": np.asarray(image, dtype=float),
        }

    return ie_cone_plot(
        xy,
        cone_type,
        support=resolved_support,
        spread=resolved_spread,
        delta=0.4 if resolved_delta is None else resolved_delta,
    )


def sensor_roi(sensor: Sensor, roi_type: Any | None = "center") -> np.ndarray:
    """Return a standard sensor ROI in MATLAB roiLocs form."""

    from .roi import ie_rect2_locs

    normalized = "center" if _is_empty_dispatch_placeholder(roi_type) else param_format(str(roi_type))
    if normalized != "center":
        raise UnsupportedOptionError("sensorROI", str(roi_type))

    size = np.asarray(sensor_get(sensor, "size"), dtype=float).reshape(-1)
    if size.size != 2:
        raise ValueError("sensorROI requires a 2-D sensor size.")

    rect = _matlab_round_to_int(
        np.array([size[1] / 4.0, size[0] / 4.0, size[1] / 2.0, size[0] / 2.0], dtype=float)
    )
    return ie_rect2_locs(rect)


def _sensor_vendor_mt9v024(variant: str, *, asset_store: AssetStore) -> Sensor:
    normalized = param_format(variant)
    mapping = {
        "mono": "data/sensor/auto/MT9V024SensorMono.mat",
        "monochrome": "data/sensor/auto/MT9V024SensorMono.mat",
        "rgb": "data/sensor/auto/MT9V024SensorRGB.mat",
        "rccc": "data/sensor/auto/MT9V024SensorRCCC.mat",
        "rgbw": "data/sensor/auto/MT9V024SensorRGBW.mat",
    }
    if normalized not in mapping:
        raise UnsupportedOptionError("sensorCreate", f"MT9V024/{variant}")
    return _sensor_from_upstream_model(mapping[normalized], asset_store=asset_store)


def _sensor_vendor_ar0132at(variant: str, *, asset_store: AssetStore) -> Sensor:
    normalized = param_format(variant)
    mapping = {
        "rgb": "data/sensor/auto/ar0132atSensorRGB.mat",
        "rccc": "data/sensor/auto/ar0132atSensorRCCC.mat",
        "rgbw": "data/sensor/auto/ar0132atSensorRGBW.mat",
    }
    if normalized not in mapping:
        raise UnsupportedOptionError("sensorCreate", f"ar0132at/{variant}")
    return _sensor_from_upstream_model(mapping[normalized], asset_store=asset_store)


def _sensor_vendor_imx363(*, asset_store: AssetStore) -> Sensor:
    return _sensor_from_upstream_model("data/sensor/sony/imx363.mat", asset_store=asset_store)


def _sensor_vendor_imx490(variant: str, *, asset_store: AssetStore) -> Sensor:
    normalized = param_format(variant)
    if normalized not in {"large", "small", "imx490large", "imx490small"}:
        raise UnsupportedOptionError("sensorCreate", f"IMX490/{variant}")

    is_large = "large" in normalized
    wave = np.arange(390.0, 711.0, 10.0, dtype=float)
    sensor = sensor_create("bayer-rggb", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", wave)
    sensor = sensor_set(sensor, "rows", 600)
    sensor = sensor_set(sensor, "cols", 800)
    sensor = sensor_set(sensor, "pixel size same fill factor", 3.0e-6)

    voltage_swing = 4096.0 * 0.25e-3
    well_capacity = 120000.0 if is_large else 60000.0
    fill_factor = 0.9 if is_large else 0.1

    sensor = sensor_set(sensor, "pixel conversion gain", voltage_swing / well_capacity)
    sensor = sensor_set(sensor, "pixel voltage swing", voltage_swing)
    sensor = sensor_set(sensor, "pixel dark voltage", 0.0)
    sensor = sensor_set(sensor, "pixel read noise electrons", 1.0)
    sensor = sensor_set(sensor, "pixel fill factor", fill_factor)
    sensor = sensor_set(sensor, "dsnu level", 0.0)
    sensor = sensor_set(sensor, "prnu level", 0.7)
    sensor = sensor_set(sensor, "analog gain", 1.0)
    sensor = sensor_set(sensor, "analog offset", 0.0)
    sensor = sensor_set(sensor, "exp time", 1.0 / 60.0)
    sensor = sensor_set(sensor, "black level", 0.0)
    sensor = sensor_set(sensor, "quantization method", "12 bit")
    sensor = sensor_set(sensor, "bits", 12)

    filter_spectra, filter_names, _ = ie_read_color_filter(
        wave,
        "data/sensor/colorfilters/auto/SONY/cf_imx490.mat",
        asset_store=asset_store,
    )
    sensor = sensor_set(sensor, "filter spectra", filter_spectra)
    sensor = sensor_set(sensor, "filter names", filter_names)

    ir_filter = ie_read_spectra(
        "data/sensor/irfilters/ircf_public.mat",
        wave,
        asset_store=asset_store,
    ).reshape(-1)
    sensor = sensor_set(sensor, "ir filter", ir_filter)
    sensor = sensor_set(sensor, "name", f"imx490-{'large' if is_large else 'small'}")
    return sensor


def _matlab_kv_pairs(args: tuple[Any, ...], *, function_name: str) -> list[tuple[str, Any]]:
    if len(args) % 2 != 0:
        raise ValueError(f"{function_name} expects MATLAB-style key/value pairs.")
    return [(param_format(str(args[index])), args[index + 1]) for index in range(0, len(args), 2)]


def _apply_sensor_settings(sensor: Sensor, settings: list[tuple[str, Any]]) -> Sensor:
    updated = sensor
    for parameter, value in settings:
        updated = sensor_set(updated, parameter, value)
    return updated


def _track_sensor_sequence(
    session: SessionContext | None,
    sensors: list[Sensor],
) -> list[Sensor]:
    if session is None:
        return sensors
    tracked: list[Sensor] = []
    last_index = len(sensors) - 1
    for index, sensor in enumerate(sensors):
        tracked.append(track_session_object(session, sensor, select=index == last_index))
    return tracked


def _sensor_create_ideal_match_exposure_time(sensor_example: Sensor) -> float:
    if bool(sensor_get(sensor_example, "auto exposure")):
        return 0.05
    exposure_time = np.asarray(sensor_get(sensor_example, "exp time"), dtype=float).reshape(-1)
    if exposure_time.size == 0:
        return 0.0
    return float(exposure_time[0])


def _sensor_create_ideal_match_sequence(
    sensor_example: Sensor,
    *,
    asset_store: AssetStore,
    use_xyz_filters: bool = False,
) -> list[Sensor]:
    source_names = list(sensor_get(sensor_example, "filter names"))
    source_filters = np.asarray(sensor_get(sensor_example, "filter spectra"), dtype=float)
    if source_filters.ndim == 1:
        source_filters = source_filters.reshape(-1, 1)
    exposure_time = _sensor_create_ideal_match_exposure_time(sensor_example)

    if use_xyz_filters:
        target_filters, target_names = _filter_bundle(
            "xyz",
            np.asarray(sensor_get(sensor_example, "wave"), dtype=float).reshape(-1),
            asset_store=asset_store,
        )
        target_filters = np.asarray(target_filters, dtype=float)
        if target_filters.ndim == 1:
            target_filters = target_filters.reshape(-1, 1)
        channel_count = min(3, int(target_filters.shape[1]))
    else:
        target_filters = source_filters
        target_names = source_names
        channel_count = int(source_filters.shape[1])

    sensors: list[Sensor] = []
    for index in range(channel_count):
        current = sensor_example.clone()
        source_name = str(source_names[index] if index < len(source_names) else f"Channel-{index + 1}")
        target_name = str(target_names[index] if index < len(target_names) else f"Channel-{index + 1}")
        current = sensor_set(current, "name", f"mono-{source_name}")
        current = sensor_set(current, "pattern", np.array([[1]], dtype=int))
        current = sensor_set(current, "filter spectra", target_filters[:, index].reshape(-1, 1))
        current = sensor_set(current, "filter names", [target_name])
        current = sensor_set(current, "integration time", exposure_time)
        current.fields["noise_flag"] = -1
        _sensor_clear_data(current)
        sensors.append(current)

    return sensors


def _sensor_create_ideal_xyz_sequence(
    *,
    asset_store: AssetStore,
    pixel_size_m: Any | None = None,
) -> list[Sensor]:
    wave = DEFAULT_WAVE.copy()
    pixel = _default_pixel(None)
    pixel["size_m"] = _normalize_pixel_size_m(pixel_size_m, default=(2.8e-6, 2.8e-6))
    pixel["fill_factor"] = 1.0
    pixel["dark_voltage_v_per_sec"] = 0.0
    pixel["read_noise_v"] = 0.0
    pixel["voltage_swing"] = 1e6

    filter_spectra, filter_names = _filter_bundle("xyz", wave, asset_store=asset_store)
    sensor_names = ["CIE-X-ideal", "CIE-Y-ideal", "CIE-Z-ideal"]
    sensors: list[Sensor] = []
    for index in range(3):
        current = _sensor_base(sensor_names[index], wave, (72, 88), pixel)
        current.fields["pattern"] = np.array([[1]], dtype=int)
        current.fields["filter_spectra"] = np.asarray(filter_spectra[:, index], dtype=float).reshape(-1, 1)
        current.fields["filter_names"] = [str(filter_names[index])]
        current.fields["noise_flag"] = -1
        current.fields["mosaic"] = False
        current.fields["integration_time"] = 1.0
        sensors.append(current)
    return sensors


def _sensor_custom_dispatch(
    filter_pattern: Any,
    filter_file: Any,
    *,
    pixel: dict[str, Any] | None = None,
    sensor_size: Any | None = None,
    wave: Any | None = None,
    name: str,
    asset_store: AssetStore,
) -> Sensor:
    pixel_dict = _default_pixel(pixel)
    current_wave = np.asarray(pixel_dict.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    if wave is not None and not _is_empty_dispatch_placeholder(wave):
        current_wave = np.asarray(wave, dtype=float).reshape(-1)
    current_size = tuple(pixel_dict.get("size", (72, 88)))
    current = _sensor_base(name, current_wave, current_size, pixel_dict)
    current = sensor_interleaved(current, filter_pattern, filter_file, asset_store=asset_store)
    current = sensor_set(current, "name", name)
    if sensor_size is not None and not _is_empty_dispatch_placeholder(sensor_size):
        current = sensor_set(current, "size", sensor_size)
    return current


def _sensor_create_ovt_large_pair(*, asset_store: AssetStore) -> tuple[Sensor, Sensor]:
    common_settings = [
        ("size", np.array([968, 1288], dtype=int)),
        ("pixel size same fill factor", 2.8e-6),
        ("pixel voltage swing", 22000.0 * 49e-6),
        ("pixel conversion gain", 49e-6),
        ("pixel fill factor", 1.0),
        ("pixel read noise electrons", 3.05),
        ("pixel dark voltage", 25.6 * 49e-6),
        ("analog gain", 1.0),
        ("quantization", "12 bit"),
        ("bits", 12),
        ("name", "ovt-LPDLCG"),
    ]
    filter_spectra, filter_names, _ = ie_read_color_filter(
        DEFAULT_WAVE,
        "data/sensor/colorfilters/OVT/ovt-large.mat",
        asset_store=asset_store,
    )
    common_settings.extend(
        [
            ("filter spectra", filter_spectra),
            ("filter names", filter_names),
        ]
    )
    lpd_lcg = _apply_sensor_settings(sensor_create(asset_store=asset_store), common_settings)
    lpd_hcg = _apply_sensor_settings(
        lpd_lcg.clone(),
        [
            ("pixel read noise electrons", 0.83),
            ("analog gain", 49.0 / 200.0),
            ("name", "ovt-LPDHCG"),
        ],
    )
    return lpd_lcg, lpd_hcg


def _sensor_create_ovt_small(*, asset_store: AssetStore) -> Sensor:
    filter_spectra, filter_names, _ = ie_read_color_filter(
        DEFAULT_WAVE,
        "data/sensor/colorfilters/OVT/ovt-large.mat",
        asset_store=asset_store,
    )
    settings = [
        ("size", np.array([968, 1288], dtype=int)),
        ("pixel size same fill factor", 2.8e-6),
        ("pixel voltage swing", 7900.0 * 49e-6),
        ("pixel conversion gain", 49e-6),
        ("pixel fill factor", 1e-2),
        ("pixel read noise electrons", 0.83),
        ("pixel dark voltage", 4.2 * 49e-6),
        ("quantization", "12 bit"),
        ("bits", 12),
        ("name", "ovt-SPDLCG"),
        ("filter spectra", filter_spectra),
        ("filter names", filter_names),
    ]
    return _apply_sensor_settings(sensor_create(asset_store=asset_store), settings)


def _sensor_create_imx490_quad(*, asset_store: AssetStore) -> tuple[Sensor, Sensor, Sensor, Sensor]:
    large_lcg = sensor_set(_sensor_vendor_imx490("large", asset_store=asset_store), "pixel size same fill factor", 2.8e-6)
    large_hcg = sensor_set(
        large_lcg.clone(),
        "pixel conversion gain",
        4.0 * float(sensor_get(large_lcg, "pixel conversion gain")),
    )
    small_lcg = sensor_set(_sensor_vendor_imx490("small", asset_store=asset_store), "pixel size same fill factor", 2.8e-6)
    small_hcg = sensor_set(
        small_lcg.clone(),
        "pixel conversion gain",
        4.0 * float(sensor_get(small_lcg, "pixel conversion gain")),
    )
    return large_lcg, large_hcg, small_lcg, small_hcg


def sensor_create_imx490(
    oi: OpticalImage,
    *args: Any,
    asset_store: AssetStore | None = None,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> list[Sensor]:
    """Mirror the legacy MATLAB sensorCreateIMX490() capture-array helper."""

    store = _store(asset_store)
    settings = _matlab_kv_pairs(args, function_name="sensorCreateIMX490")

    gains = np.array([1.0, 4.0, 1.0, 4.0], dtype=float)
    noise_flag = 2
    exp_time = 1.0 / 60.0
    for parameter, value in settings:
        if parameter == "gain":
            gains = np.asarray(value, dtype=float).reshape(-1)
            if gains.size != 4:
                raise ValueError("sensorCreateIMX490 gain must contain four multiplicative gains.")
        elif parameter == "noiseflag":
            noise_flag = int(value)
        elif parameter in {"exptime", "exposuretime", "integrationtime"}:
            exp_time = float(value)
        else:
            raise UnsupportedOptionError("sensorCreateIMX490", parameter)
    _ = gains

    large = sensor_create("imx490-large", asset_store=store)
    large = sensor_set(large, "match oi", oi)
    large = sensor_set(large, "noise flag", noise_flag)
    large = sensor_set(large, "exp time", exp_time)

    small = sensor_create("imx490-small", asset_store=store)
    small = sensor_set(small, "match oi", oi)
    small = sensor_set(small, "noise flag", noise_flag)
    small = sensor_set(small, "exp time", exp_time)

    oi_spacing_m = float(np.asarray(oi_get(oi, "spatial resolution"), dtype=float).reshape(-1)[0])
    sensor_spacing_m = float(np.asarray(sensor_get(large, "pixel size"), dtype=float).reshape(-1)[0])
    if abs(oi_spacing_m - sensor_spacing_m) > 1.0e-9:
        raise ValueError("sensorCreateIMX490 requires an optical image sampled at the IMX490 pixel size.")

    oi_size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
    if oi_size.size != 2:
        raise ValueError("sensorCreateIMX490 requires a 2-D optical image.")
    large = sensor_set(large, "size", oi_size)
    small = sensor_set(small, "size", oi_size)

    capture_specs = (
        (
            "large-HCG",
            large,
            [
                ("pixel conversion gain", 200e-6),
                ("pixel read noise electrons", 0.83),
                ("pixel dark voltage", 25.6 * 200e-6),
                ("pixel voltage swing", 22000.0 * 49e-6),
                ("pixel spectral qe", 1.0),
            ],
        ),
        (
            "large-LCG",
            large,
            [
                ("pixel conversion gain", 49e-6),
                ("pixel read noise electrons", 3.05),
                ("pixel dark voltage", 25.6 * 49e-6),
                ("pixel voltage swing", 22000.0 * 49e-6),
                ("pixel spectral qe", 1.0),
            ],
        ),
        (
            "small-HCG",
            small,
            [
                ("pixel conversion gain", 200e-6),
                ("pixel read noise electrons", 0.83),
                ("pixel dark voltage", 4.2 * 200e-6),
                ("pixel voltage swing", 7900.0 * 49e-6),
                ("pixel spectral qe", 0.1),
            ],
        ),
        (
            "small-LCG",
            small,
            [
                ("pixel conversion gain", 49e-6),
                ("pixel read noise electrons", 2.96),
                ("pixel dark voltage", 4.2 * 49e-6),
                ("pixel voltage swing", 7900.0 * 49e-6),
                ("pixel spectral qe", 0.1),
            ],
        ),
    )

    captures: list[Sensor] = []
    for index, (name, template, capture_settings) in enumerate(capture_specs):
        capture = template.clone()
        for parameter, value in capture_settings:
            capture = sensor_set(capture, parameter, value)
        capture = sensor_set(capture, "name", name)
        capture = sensor_compute(capture, oi, seed=None if seed is None else int(seed) + index)
        captures.append(capture)

    return _track_sensor_sequence(session, captures)


def sensor_create(
    sensor_type: str = "default",
    pixel: dict[str, Any] | None = None,
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor | list[Sensor]:
    """Create a supported sensor."""

    store = _store(asset_store)
    resolved_type = (
        "default"
        if _is_empty_dispatch_placeholder(sensor_type)
        or (isinstance(sensor_type, str) and sensor_type.strip() == "")
        else sensor_type
    )
    normalized = param_format(resolved_type)
    if normalized == "lightfield" and isinstance(pixel, OpticalImage):
        args = (pixel, *args)
        pixel = None
    if normalized == "dualpixel" and isinstance(pixel, OpticalImage):
        args = (pixel, *args)
        pixel = None
    if normalized == "monochromearray" and pixel is not None and not isinstance(pixel, dict) and not _is_empty_dispatch_placeholder(pixel):
        args = (pixel, *args)
        pixel = None
    if normalized == "imec44" and pixel is not None and not isinstance(pixel, dict) and not _is_empty_dispatch_placeholder(pixel):
        args = (pixel, *args)
        pixel = None
    if normalized in {"custom", "fourcolor"} and pixel is not None and not isinstance(pixel, dict) and not _is_empty_dispatch_placeholder(pixel):
        args = (pixel, *args)
        pixel = None
    if normalized in {"mt9v024", "ar0132at"} and pixel is not None and not isinstance(pixel, dict) and not _is_empty_dispatch_placeholder(pixel):
        args = (pixel, *args)
        pixel = None
    if normalized in {"imx363", "googlepixel4a"} and isinstance(pixel, str):
        args = (pixel, *args)
        pixel = None
    pixel_dict = _default_pixel(pixel)
    wave = np.asarray(pixel_dict.get("wave", DEFAULT_WAVE), dtype=float)
    size = tuple(pixel_dict.get("size", (72, 88)))

    if normalized in {"default", "color", "bayer", "rgb", "bayergrbg", "bayer-grbg", "bayer(grbg)"}:
        sensor = _sensor_base("bayer-grbg", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 1], [3, 2]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayergbrg", "bayer-gbrg", "bayer(gbrg)"}:
        sensor = _sensor_base("bayer-gbrg", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 3], [1, 2]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayerrggb", "bayer-rggb", "bayer(rggb)"}:
        sensor = _sensor_base("bayer-rggb", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayerbggr", "bayer-bggr", "bayer(bggr)"}:
        sensor = _sensor_base("bayer-bggr", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[3, 2], [2, 1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("RGB", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "monochrome":
        sensor = _sensor_base("monochrome", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("monochrome", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "monochromearray":
        count_source = args[0] if args and args[0] is not None else 3
        if _is_empty_dispatch_placeholder(count_source):
            count_source = 3
        count = int(np.rint(np.asarray(count_source, dtype=float).reshape(-1)[0]))
        if count < 1:
            raise ValueError("sensorCreate('monochrome array', ...) requires a positive sensor count.")
        template = sensor_create("monochrome", pixel, asset_store=store)
        sensors = [copy.deepcopy(template) for _ in range(count)]
        return _track_sensor_sequence(session, sensors)

    if normalized in {"rgbw", "interleaved"}:
        sensor = _sensor_base("rgbw", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [3, 4]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("interleavedrgbw", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "grbc":
        sensor = _sensor_base("grbc", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [3, 4]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("grbc", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "rccc":
        sensor = _sensor_base("rccc", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 2], [2, 1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle(["r", "w"], wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayer-ycmy", "ycmy", "bayer(ycmy)"}:
        sensor = _sensor_base("bayer-ycmy", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[2, 1], [3, 2]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("cym", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized in {"bayer-cyym", "cyym", "bayer(cyym)"}:
        sensor = _sensor_base("bayer-cyym", wave, size, pixel_dict)
        sensor.fields["pattern"] = np.array([[1, 2], [2, 3]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("cym", wave, asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "lightfield":
        if args:
            oi = args[0]
        elif isinstance(pixel, OpticalImage):
            oi = pixel
        else:
            raise ValueError("sensorCreate('light field', ...) requires an optical image.")
        if not isinstance(oi, OpticalImage):
            raise ValueError("sensorCreate('light field', ...) requires an optical image.")
        sensor = sensor_light_field(oi, asset_store=store)
        sensor = sensor_set(sensor, "name", str(oi_get(oi, "name")))
        return track_session_object(session, sensor)

    if normalized == "dualpixel":
        if len(args) < 2:
            raise ValueError("sensorCreate('dual pixel', ...) requires an optical image and microlens size.")
        oi = args[0]
        n_microlens = np.rint(np.asarray(args[1], dtype=float).reshape(-1)[:2]).astype(int)
        if not isinstance(oi, OpticalImage):
            raise ValueError("sensorCreate('dual pixel', ...) requires an optical image.")
        if n_microlens.size != 2:
            raise ValueError("sensorCreate('dual pixel', ...) requires a [rows cols] microlens size.")
        sample_spacing = np.asarray(oi_get(oi, "sample spacing", "m"), dtype=float).reshape(-1)
        sensor = sensor_create(asset_store=store)
        sensor = sensor_set(sensor, "pixel height", 2.0 * float(sample_spacing[0]))
        sensor = sensor_set(sensor, "pixel width", float(sample_spacing[0]))
        sensor = sensor_set(sensor, "size", (int(n_microlens[0]), int(2 * n_microlens[1])))
        sensor = sensor_set(sensor, "pattern", np.array([[2, 2, 1, 1], [3, 3, 2, 2]], dtype=int))
        return track_session_object(session, sensor)

    if normalized == "mt9v024":
        variants = _sensor_variant_names(args, "rgb")
        sensors = [_sensor_vendor_mt9v024(variant, asset_store=store) for variant in variants]
        if len(sensors) == 1:
            return track_session_object(session, sensors[0])
        return _track_sensor_sequence(session, sensors)

    if normalized == "ar0132at":
        variants = _sensor_variant_names(args, "rgb")
        sensors = [_sensor_vendor_ar0132at(variant, asset_store=store) for variant in variants]
        if len(sensors) == 1:
            return track_session_object(session, sensors[0])
        return _track_sensor_sequence(session, sensors)

    if normalized in {"imx363", "googlepixel4a"}:
        sensor = _sensor_vendor_imx363(asset_store=store)
        if args:
            if len(args) % 2 != 0:
                raise ValueError("sensorCreate('IMX363', ...) expects key/value pairs.")
            current_gain = float(sensor.fields["analog_gain"])
            current_offset = float(sensor.fields["analog_offset"])
            black_level_v = current_offset / max(current_gain, 1e-12)
            for index in range(0, len(args), 2):
                parameter = param_format(args[index])
                value = args[index + 1]
                if parameter in {"rowcol", "rowcolsize"}:
                    rows, cols = np.rint(np.asarray(value, dtype=float).reshape(-1)[:2]).astype(int)
                    sensor = sensor_set(sensor, "rows", int(rows))
                    sensor = sensor_set(sensor, "cols", int(cols))
                    continue
                if parameter in {"isospeed", "iso"}:
                    iso_speed = float(value)
                    analog_gain = 55.0 / max(iso_speed, 1e-12)
                    sensor = sensor_set(sensor, "analog gain", analog_gain)
                    sensor = sensor_set(sensor, "analog offset", black_level_v * analog_gain)
                    continue
                if parameter in {"exposuretime", "exptime", "integrationtime"}:
                    sensor = sensor_set(sensor, "integration time", value)
                    continue
                if parameter == "wave":
                    sensor = sensor_set(sensor, "wave", value)
                    continue
                sensor = sensor_set(sensor, parameter, value)
        return track_session_object(session, sensor)

    if normalized in {"imx490large", "imx490-large"}:
        return track_session_object(session, _sensor_vendor_imx490("large", asset_store=store))

    if normalized in {"imx490small", "imx490-small"}:
        return track_session_object(session, _sensor_vendor_imx490("small", asset_store=store))

    if normalized == "nikond100":
        sensor = _sensor_from_upstream_model("data/sensor/nikon/NikonD100Sensor.mat", asset_store=store)
        sensor = sensor_set(sensor, "size", np.array([72, 88], dtype=int) * 4)
        sensor = sensor_set(sensor, "name", "Nikon-D100")
        return track_session_object(session, sensor)

    if normalized in {"ovtlarge", "ovt-large"}:
        large_lcg, large_hcg = _sensor_create_ovt_large_pair(asset_store=store)
        return _track_sensor_sequence(session, [large_lcg, large_hcg])

    if normalized in {"ovtsmall", "ovt-small"}:
        return track_session_object(session, _sensor_create_ovt_small(asset_store=store))

    if normalized == "imec44":
        if args and not _is_empty_dispatch_placeholder(args[0]):
            return sensor_create_imec_ssm_4x4_vis("row col", args[0], asset_store=store, session=session)
        if args:
            return sensor_create_imec_ssm_4x4_vis(asset_store=store, session=session)
        return sensor_create_imec_ssm_4x4_vis("row col", np.array([400, 400], dtype=int), asset_store=store, session=session)

    if normalized == "human":
        if args and hasattr(args[0], "items"):
            params_source = args[0]
        elif pixel is not None and hasattr(pixel, "items"):
            params_source = pixel
        else:
            params_source = {}
        params = dict(params_source) if isinstance(params_source, dict) else dict(vars(params_source)) if hasattr(params_source, "__dict__") else {}
        wave_source = _empty_dispatch_placeholder_to_none(params.get("wave"))
        wave = np.asarray(np.arange(400.0, 701.0, 10.0) if wave_source is None else wave_source, dtype=float).reshape(-1)
        base = sensor_create(asset_store=store)
        base = sensor_set(base, "wave", wave)
        base = sensor_set(base, "pixel", pixel_create("human", wave))
        current, xy, cone_type, resolved_seed, corrected_densities = sensor_create_cone_mosaic(
            base,
            _empty_dispatch_placeholder_to_none(params.get("sz")),
            _empty_dispatch_placeholder_to_none(params.get("rgbDensities")),
            _empty_dispatch_placeholder_to_none(params.get("coneAperture")),
            _empty_dispatch_placeholder_to_none(params.get("rSeed")),
            "human",
            asset_store=store,
        )
        human_pixel = pixel_set(sensor_get(current, "pixel"), "voltage swing", 1.0)
        current = sensor_set(current, "pixel", human_pixel)
        current = sensor_set(current, "exposure time", 1.0)
        current = sensor_set(current, "cone locs", xy)
        current = sensor_set(current, "cone type", cone_type)
        current = sensor_set(current, "densities", corrected_densities)
        current = sensor_set(current, "rSeed", resolved_seed)
        return track_session_object(session, current)

    if normalized == "fourcolor":
        if len(args) < 2:
            raise ValueError("sensorCreate('fourcolor', ...) requires a filter pattern and filter file.")
        sensor = _sensor_custom_dispatch(args[0], args[1], pixel=pixel, name="fourcolor", asset_store=store)
        return track_session_object(session, sensor)

    if normalized == "custom":
        if len(args) < 2:
            raise ValueError("sensorCreate('custom', ...) requires a filter pattern and filter file.")
        filter_pattern = args[0]
        filter_file = args[1]
        sensor_size = (
            args[2]
            if len(args) > 2 and not _is_empty_dispatch_placeholder(args[2])
            else np.asarray(filter_pattern, dtype=int).shape
        )
        wave_override = args[3] if len(args) > 3 and not _is_empty_dispatch_placeholder(args[3]) else None
        sensor = _sensor_custom_dispatch(
            filter_pattern,
            filter_file,
            pixel=pixel,
            sensor_size=sensor_size,
            wave=wave_override,
            name="custom",
            asset_store=store,
        )
        return track_session_object(session, sensor)

    if normalized == "ideal":
        raise ValueError("sensorCreate('ideal') is deprecated. Use sensorCreateIdeal")

    raise UnsupportedOptionError("sensorCreate", sensor_type)


def sensor_create_ideal(
    ideal_type: str = "monochrome",
    sensor_example: Sensor | None = None,
    pixel_size_m: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor | list[Sensor]:
    """Create an ideal milestone-one sensor."""

    store = _store(asset_store)
    normalized = param_format(ideal_type)
    if sensor_example is not None and _is_empty_dispatch_placeholder(sensor_example):
        sensor_example = None
    pixel = _default_pixel(sensor_example.fields["pixel"] if sensor_example is not None else None)
    if pixel_size_m is not None:
        pixel["size_m"] = _normalize_pixel_size_m(pixel_size_m)
    pixel["fill_factor"] = 1.0
    size = sensor_example.fields["size"] if sensor_example is not None else (72, 88)
    wave = sensor_example.fields["wave"] if sensor_example is not None else DEFAULT_WAVE.copy()

    if normalized in {"monochrome"}:
        if sensor_example is not None:
            raise ValueError("Sensor example not used for monochrome case.")
        sensor = _sensor_base("Monochrome", np.asarray(wave, dtype=float), size, pixel)
        sensor.fields["pattern"] = np.array([[1]], dtype=int)
        sensor.fields["filter_spectra"], sensor.fields["filter_names"] = _filter_bundle("monochrome", np.asarray(wave, dtype=float), asset_store=store)
        sensor.fields["noise_flag"] = 0
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = 0.0
        sensor.fields["pixel"]["read_noise_v"] = 0.0
        sensor.fields["pixel"]["voltage_swing"] = 1e6
        return track_session_object(session, sensor)

    if normalized == "match":
        if sensor_example is None:
            raise ValueError("sensorCreateIdeal('match', ...) requires a sensor example.")
        return _track_sensor_sequence(session, _sensor_create_ideal_match_sequence(sensor_example, asset_store=store))

    if normalized == "matchxyz":
        if sensor_example is None:
            raise ValueError("sensorCreateIdeal('match xyz', ...) requires a sensor example.")
        return _track_sensor_sequence(
            session,
            _sensor_create_ideal_match_sequence(sensor_example, asset_store=store, use_xyz_filters=True),
        )

    if normalized == "xyz":
        sensors = _sensor_create_ideal_xyz_sequence(asset_store=store, pixel_size_m=pixel_size_m)
        return _track_sensor_sequence(session, sensors)

    raise UnsupportedOptionError("sensorCreateIdeal", ideal_type)


def sensor_create_cone_mosaic(
    sensor: Sensor | None = None,
    sz: Any | None = None,
    densities: Any | None = None,
    cone_aperture: Any | None = None,
    r_seed: Any | None = None,
    species: str = "human",
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[Sensor, np.ndarray, np.ndarray, int, np.ndarray]:
    """Create a MATLAB-style human cone-mosaic sensor."""

    store = _store(asset_store)
    normalized_species = param_format(species)
    if normalized_species != "human":
        raise UnsupportedOptionError("sensorCreateConeMosaic", species)

    current = sensor_create(asset_store=store, session=session) if sensor is None else sensor.clone()
    sz = _empty_dispatch_placeholder_to_none(sz)
    densities = _empty_dispatch_placeholder_to_none(densities)
    cone_aperture = _empty_dispatch_placeholder_to_none(cone_aperture)
    r_seed = _empty_dispatch_placeholder_to_none(r_seed)

    mosaic_size = np.asarray([72, 88] if sz is None else sz, dtype=int).reshape(-1)
    aperture = np.asarray([1.5e-6, 1.5e-6] if cone_aperture is None else cone_aperture, dtype=float).reshape(-1)
    if aperture.size == 1:
        aperture = np.repeat(aperture, 2)

    xy, cone_type, corrected_densities, resolved_seed = _human_cone_mosaic(
        mosaic_size,
        densities,
        float(aperture[0] * 1.0e6),
        r_seed,
    )
    current = sensor_set(current, "name", "human")
    current = sensor_set(current, "size", tuple(int(value) for value in cone_type.shape))
    current = sensor_set(current, "pattern", cone_type)
    current = sensor_set(
        current,
        "human",
        {
            "name": "human",
            "coneType": cone_type.copy(),
            "densities": corrected_densities.copy(),
            "xy": xy.copy(),
            "rSeed": resolved_seed,
            "species": "human",
        },
    )

    wave = np.asarray(sensor_get(current, "wave"), dtype=float).reshape(-1)
    stockman_quanta, filter_names, _ = ie_read_color_filter(wave, "data/human/stockmanQuanta.mat", asset_store=store)
    black = np.zeros((wave.size, 1), dtype=float)
    current = sensor_set(current, "filter spectra", np.hstack((black, np.asarray(stockman_quanta, dtype=float))))
    current = sensor_set(current, "filter names", ["kBlack", *filter_names])
    current = sensor_set(current, "pixel size same fill factor", aperture[:2])
    return track_session_object(session, current), xy, cone_type, resolved_seed, corrected_densities


def sensor_check_human(sensor: Sensor) -> bool:
    """Determine whether the sensor carries the legacy human-cone metadata."""

    if sensor is None:
        raise ValueError("sensor required")
    name = str(sensor_get(sensor, "name") or "")
    return ("human" in param_format(name)) or isinstance(sensor.fields.get("human"), dict)


def sensor_human_resize(sensor: Sensor, rows: Any, cols: Any) -> Sensor:
    """Add or remove rows/cols around a human cone mosaic."""

    current = sensor.clone()
    row_adjust = np.asarray(rows, dtype=int).reshape(-1)
    col_adjust = np.asarray(cols, dtype=int).reshape(-1)
    if row_adjust.size != 2 or col_adjust.size != 2:
        raise ValueError("sensorHumanResize expects [top, bottom] rows and [left, right] cols.")

    pattern = np.asarray(sensor_get(current, "pattern"), dtype=int)
    if col_adjust[0] > 0:
        pattern = np.hstack((np.ones((pattern.shape[0], int(col_adjust[0])), dtype=int), pattern))
    elif col_adjust[0] < 0:
        pattern = pattern[:, int(abs(col_adjust[0])) :]
    if col_adjust[1] > 0:
        pattern = np.hstack((pattern, np.ones((pattern.shape[0], int(col_adjust[1])), dtype=int)))
    elif col_adjust[1] < 0:
        pattern = pattern[:, : pattern.shape[1] - int(abs(col_adjust[1]))]

    if row_adjust[0] > 0:
        pattern = np.vstack((np.ones((int(row_adjust[0]), pattern.shape[1]), dtype=int), pattern))
    elif row_adjust[0] < 0:
        pattern = pattern[int(abs(row_adjust[0])) :, :]
    if row_adjust[1] > 0:
        pattern = np.vstack((pattern, np.ones((int(row_adjust[1]), pattern.shape[1]), dtype=int)))
    elif row_adjust[1] < 0:
        pattern = pattern[: pattern.shape[0] - int(abs(row_adjust[1])), :]

    current = sensor_set(current, "size", tuple(int(value) for value in pattern.shape))
    current = sensor_set(current, "pattern", pattern)
    if isinstance(current.fields.get("human"), dict):
        current.fields["human"]["coneType"] = pattern.copy()
        current.fields["human"]["xy"] = None
    return current


def sensor_light_field(
    oi: OpticalImage,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Create a light-field sensor matched to the optical-image geometry."""

    sensor = sensor_create(asset_store=_store(asset_store), session=session)
    sample_spacing = np.asarray(oi_get(oi, "sample spacing", "m"), dtype=float).reshape(-1)
    sensor = sensor_set(sensor, "pixel size same fill factor", float(sample_spacing[0]))
    sensor = sensor_set(sensor, "size", tuple(int(value) for value in np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)[:2]))
    sensor = sensor_set(sensor, "name", "lightfield")
    return track_session_object(session, sensor)


def sensor_interleaved(
    sensor: Sensor | None,
    filter_pattern: Any,
    filter_file: Any,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Create a MATLAB-style interleaved sensor with a custom filter set."""

    store = _store(asset_store)
    current = sensor_create(asset_store=store, session=session) if sensor is None else sensor.clone()
    current = sensor_set(current, "name", "interleaved")
    current = sensor_set(current, "cfapattern", np.asarray(filter_pattern, dtype=int))

    wave = np.asarray(sensor_get(current, "wave"), dtype=float).reshape(-1)
    if isinstance(filter_file, (str, Path)):
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, filter_file, asset_store=store)
    elif isinstance(filter_file, dict) or hasattr(filter_file, "data"):
        source_data = filter_file if isinstance(filter_file, dict) else vars(filter_file)
        filter_spectra = np.asarray(source_data["data"], dtype=float)
        filter_names = list(source_data["filterNames"])
        filter_wave = np.asarray(source_data["wavelength"], dtype=float).reshape(-1)
        if filter_spectra.ndim == 1:
            filter_spectra = filter_spectra.reshape(-1, 1)
        if filter_spectra.shape[0] != filter_wave.size and filter_spectra.shape[1] == filter_wave.size:
            filter_spectra = filter_spectra.T
        filter_spectra = np.vstack(
            [np.interp(wave, filter_wave, filter_spectra[:, index], left=0.0, right=0.0) for index in range(filter_spectra.shape[1])]
        ).T
    else:
        raise ValueError("Bad format for filterFile variable.")

    current = sensor_set(current, "filter spectra", filter_spectra)
    current = sensor_set(current, "filter names", filter_names)
    return track_session_object(session, current)


def sensor_mt9v024(
    sensor_or_color_type: Any | None = None,
    color_type: str | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Legacy MATLAB wrapper for the ON MT9V024 sensor family."""

    variant = color_type
    if variant is None and isinstance(sensor_or_color_type, str):
        variant = sensor_or_color_type
    if variant is None:
        variant = "rgb"
    return track_session_object(session, _sensor_vendor_mt9v024(str(variant), asset_store=_store(asset_store)))


def sensor_imx363_v2(
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Legacy MATLAB compatibility wrapper for the IMX363V2 constructor."""

    forwarded = args[1:] if args and args[0] is None else args
    return sensor_create("imx363", None, *forwarded, asset_store=_store(asset_store), session=session)


def sensor_create_imec_ssm_4x4_vis(
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Create the legacy IMEC SSM 4x4 visible multispectral sensor."""

    store = _store(asset_store)
    settings = dict(_matlab_kv_pairs(args, function_name="sensorCreateIMECSSM4x4vis"))
    row_col_source = settings.get("rowcol", [1016, 2040])
    if _is_empty_dispatch_placeholder(row_col_source):
        row_col_source = [1016, 2040]
    row_col = np.asarray(row_col_source, dtype=int).reshape(-1)
    pixel_size = float(settings.get("pixelsize", 5.5e-6))
    quantization = str(settings.get("quantization", "10 bit"))
    dsnu = float(settings.get("dsnu", 0.0))
    prnu = float(settings.get("prnu", 0.7))
    fill_factor = float(settings.get("fillfactor", 0.42))
    dark_current = float(settings.get("darkcurrent", 125.0))
    exposure_time = float(settings.get("exposuretime", 1.0 / 60.0))
    wave = np.asarray(settings.get("wave", np.arange(460.0, 621.0, 1.0)), dtype=float).reshape(-1)
    read_noise = float(settings.get("readnoise", 13.0))
    well_capacity = float(settings.get("wellcapacity", 13.5e3))
    voltage_swing = float(settings.get("voltageswing", 2.0))
    qe_filename = settings.get("qefilename", "data/sensor/imec/qe_IMEC.mat")

    sensor = sensor_create(asset_store=store, session=session)
    sensor = sensor_set(sensor, "pixel fill factor", fill_factor)
    sensor = sensor_set(sensor, "pixel size constant fill factor", pixel_size)
    sensor = sensor_set(sensor, "rows", int(row_col[0]))
    sensor = sensor_set(sensor, "cols", int(row_col[1]))
    sensor = sensor_set(sensor, "name", "IMEC SSM")
    sensor = sensor_set(sensor, "quantization", quantization)
    sensor = sensor_set(sensor, "exp time", exposure_time)
    sensor = sensor_set(sensor, "pixel voltage swing", voltage_swing)
    sensor = sensor_set(sensor, "pixel conversion gain", voltage_swing / max(well_capacity, 1.0e-12))
    sensor = sensor_set(sensor, "pixel dark voltage", (voltage_swing / max(well_capacity, 1.0e-12)) * dark_current)
    sensor = sensor_set(sensor, "pixel read noise electrons", read_noise)
    sensor = sensor_set(sensor, "DSNU level", dsnu)
    sensor = sensor_set(sensor, "PRNU level", prnu)
    sensor = sensor_set(sensor, "pattern", np.arange(1, 17, dtype=int).reshape(4, 4).T)
    sensor = sensor_set(sensor, "wave", wave)
    filters, filter_names, _ = ie_read_color_filter(wave, qe_filename, asset_store=store)
    sensor = sensor_set(sensor, "filter transmissivities", filters)
    sensor = sensor_set(sensor, "filter names", filter_names)
    return track_session_object(session, sensor)


def sensor_create_split_pixel(
    *args: Any,
    asset_store: AssetStore | None = None,
) -> list[Sensor]:
    """Create a MATLAB-style split-pixel sensor array."""

    store = _store(asset_store)
    settings = _matlab_kv_pairs(args, function_name="sensorCreateSplitPixel")
    array_type = "ovt"
    shared_settings: list[tuple[str, Any]] = []
    for parameter, value in settings:
        if parameter == "arraytype":
            if _is_empty_dispatch_placeholder(value):
                continue
            array_type = str(value)
            continue
        shared_settings.append((parameter, value))

    normalized_array_type = param_format(array_type)
    if normalized_array_type == "ovt":
        sensors = _sensor_create_ovt_large_pair(asset_store=store) + (_sensor_create_ovt_small(asset_store=store),)
    elif normalized_array_type == "imx490":
        sensors = _sensor_create_imx490_quad(asset_store=store)
    else:
        raise UnsupportedOptionError("sensorCreateSplitPixel", array_type)
    return [_apply_sensor_settings(sensor.clone(), shared_settings) for sensor in sensors]


def sensor_create_array(
    *args: Any,
    asset_store: AssetStore | None = None,
) -> Sensor | list[Sensor]:
    """Create a MATLAB-style coordinated sensor array."""

    store = _store(asset_store)
    settings = _matlab_kv_pairs(args, function_name="sensorCreateArray")
    array_type = "ovt"
    sensor_example: Sensor | None = None
    shared_settings: list[tuple[str, Any]] = []
    for parameter, value in settings:
        if parameter == "arraytype":
            if _is_empty_dispatch_placeholder(value):
                continue
            array_type = str(value)
            continue
        if parameter == "sensorexample":
            if _is_empty_dispatch_placeholder(value):
                sensor_example = None
                continue
            if not isinstance(value, Sensor):
                raise ValueError("sensorCreateArray('sensor example', ...) requires a Sensor.")
            sensor_example = value
            continue
        shared_settings.append((parameter, value))

    normalized_array_type = param_format(array_type)
    if normalized_array_type in {"ovt", "imx490"}:
        forward_args: list[Any] = ["arraytype", array_type]
        for parameter, value in shared_settings:
            forward_args.extend((parameter, value))
        return sensor_create_split_pixel(*forward_args, asset_store=store)
    if normalized_array_type == "monochrome":
        sensor = sensor_create_ideal("monochrome", sensor_example, asset_store=store)
        return _apply_sensor_settings(sensor, shared_settings)
    if normalized_array_type == "xyz":
        sensors = _sensor_create_ideal_xyz_sequence(asset_store=store)
        return [_apply_sensor_settings(sensor, shared_settings) for sensor in sensors]
    if normalized_array_type in {"match", "matchxyz"}:
        sensors = sensor_create_ideal(array_type, sensor_example, asset_store=store)
        if isinstance(sensors, Sensor):
            return _apply_sensor_settings(sensors, shared_settings)
        return [_apply_sensor_settings(sensor, shared_settings) for sensor in sensors]
    raise UnsupportedOptionError("sensorCreateArray", array_type)


def sensor_compute_array(
    sensor_array: list[Sensor] | tuple[Sensor, ...],
    oi: OpticalImage,
    *args: Any,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> tuple[Sensor, list[Sensor]]:
    """Compute a split-pixel sensor array and its combined response."""

    settings = _matlab_kv_pairs(args, function_name="sensorComputeArray")
    method = "saturated"
    saturated_fraction = 0.95
    for parameter, value in settings:
        if parameter == "method":
            method = str(value)
        elif parameter == "saturated":
            saturated_fraction = float(value)
        else:
            raise UnsupportedOptionError("sensorComputeArray", parameter)

    computed_sensors = [
        sensor_compute(sensor, oi, seed=None if seed is None else int(seed) + index)
        for index, sensor in enumerate(sensor_array)
    ]
    if not computed_sensors:
        raise ValueError("sensorComputeArray requires at least one sensor.")

    design_name = str(sensor_get(computed_sensors[0], "name"))
    if not param_format(design_name).startswith("ovt"):
        raise UnsupportedOptionError("sensorComputeArray", design_name)

    reference_shape = tuple(int(value) for value in sensor_get(computed_sensors[0], "size"))
    input_referred = np.zeros(reference_shape + (len(computed_sensors),), dtype=float)
    saturated_mask = np.zeros(reference_shape + (len(computed_sensors),), dtype=bool)
    voltage_swings = np.zeros(len(computed_sensors), dtype=float)

    for index, sensor in enumerate(computed_sensors):
        volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
        voltage_swings[index] = float(sensor_get(sensor, "pixel voltage swing"))
        saturated_mask[:, :, index] = volts >= (saturated_fraction * voltage_swings[index])
        if index == 0:
            input_referred[:, :, index] = volts
        elif index == 1:
            input_referred[:, :, index] = volts * float(sensor_get(sensor, "analog gain"))
        elif index == 2:
            input_referred[:, :, index] = volts / max(float(sensor.fields["pixel"]["fill_factor"]), 1e-12)
        else:
            raise UnsupportedOptionError("sensorComputeArray", "OVT arrays beyond 3 captures")

    combined = computed_sensors[0].clone()
    normalized_method = param_format(method)
    if normalized_method == "saturated":
        combined_volts = np.zeros(reference_shape, dtype=float)
        good_first = ~saturated_mask[:, :, 0]
        good_second = ~saturated_mask[:, :, 1]
        both_good = good_first & good_second
        combined_volts[both_good] = 0.5 * (input_referred[:, :, 0][both_good] + input_referred[:, :, 1][both_good])
        first_only = good_first & ~good_second
        combined_volts[first_only] = input_referred[:, :, 0][first_only]
        neither = ~good_first & ~good_second
        combined_volts[neither] = input_referred[:, :, 2][neither]

        target_swing = float(sensor_get(combined, "pixel voltage swing"))
        max_value = float(np.max(combined_volts))
        if max_value > 0.0:
            combined_volts = combined_volts * (target_swing / max_value)
        combined = sensor_set(combined, "quantization method", "analog")
        combined = sensor_set(combined, "volts", combined_volts)
        combined = sensor_set(combined, "analog gain", 1.0)
        combined = sensor_set(combined, "analog offset", 0.0)
        combined.metadata["saturated"] = saturated_mask
    elif normalized_method == "bestsnr":
        electrons = []
        for sensor in computed_sensors:
            signal = np.asarray(sensor_get(sensor, "electrons"), dtype=float)
            well_capacity = float(sensor_get(sensor, "pixel well capacity"))
            electrons.append(np.where(signal < well_capacity, signal, 0.0))
        electron_stack = np.stack(electrons, axis=2)
        best_pixel = np.argmax(electron_stack, axis=2) + 1
        best_signal = np.max(electron_stack, axis=2)
        combined_volts = best_signal * float(sensor_get(combined, "pixel conversion gain"))
        combined = sensor_set(combined, "quantization method", "analog")
        combined = sensor_set(combined, "volts", combined_volts)
        combined = sensor_set(combined, "analog gain", 1.0)
        combined = sensor_set(combined, "analog offset", 0.0)
        combined.metadata["bestPixel"] = best_pixel.astype(int)
    else:
        raise UnsupportedOptionError("sensorComputeArray", method)

    design_root = design_name.split("-", 1)[0]
    combined = sensor_set(combined, "name", f"{design_root}-{normalized_method}")
    return track_session_object(session, combined), computed_sensors


sensorCreateSplitPixel = sensor_create_split_pixel
sensorCreateArray = sensor_create_array
sensorComputeArray = sensor_compute_array


def imx490_compute(
    oi: OpticalImage,
    *args: Any,
    asset_store: AssetStore | None = None,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> tuple[Sensor, dict[str, Any]]:
    """Mirror the stable headless `imx490Compute(...)` workflow."""

    store = _store(asset_store)
    settings = _matlab_kv_pairs(args, function_name="imx490Compute")

    gains = np.array([1.0, 4.0, 1.0, 4.0], dtype=float)
    noise_flag = 2
    exp_time = 1.0 / 60.0
    method = "average"
    for parameter, value in settings:
        if parameter == "gain":
            gains = np.asarray(value, dtype=float).reshape(-1)
            if gains.size != 4:
                raise ValueError("imx490Compute gain must contain four multiplicative gains.")
        elif parameter == "noiseflag":
            noise_flag = int(value)
        elif parameter in {"exptime", "exposuretime", "integrationtime"}:
            exp_time = float(value)
        elif parameter == "method":
            method = str(value)
        else:
            raise UnsupportedOptionError("imx490Compute", parameter)

    large = sensor_create("imx490-large", asset_store=store)
    large = sensor_set(large, "match oi", oi)
    large = sensor_set(large, "noise flag", noise_flag)
    large = sensor_set(large, "exp time", exp_time)

    small = sensor_create("imx490-small", asset_store=store)
    small = sensor_set(small, "match oi", oi)
    small = sensor_set(small, "noise flag", noise_flag)
    small = sensor_set(small, "exp time", exp_time)

    oi_spacing_m = float(np.asarray(oi_get(oi, "spatial resolution"), dtype=float).reshape(-1)[0])
    sensor_spacing_m = float(np.asarray(sensor_get(large, "pixel size"), dtype=float).reshape(-1)[0])
    if abs(oi_spacing_m - sensor_spacing_m) > 1.0e-9:
        raise ValueError("imx490Compute requires an optical image sampled at the IMX490 pixel size.")

    oi_size = np.asarray(oi_get(oi, "size"), dtype=int).reshape(-1)
    if oi_size.size != 2:
        raise ValueError("imx490Compute requires a 2-D optical image.")
    large = sensor_set(large, "size", oi_size)
    small = sensor_set(small, "size", oi_size)

    iset_gains = 1.0 / np.maximum(gains, 1.0e-12)
    capture_specs = (
        ("large-1x", large, float(iset_gains[0])),
        ("large-4x", large, float(iset_gains[1])),
        ("small-1x", small, float(iset_gains[2])),
        ("small-4x", small, float(iset_gains[3])),
    )
    captures: list[Sensor] = []
    for index, (name, template, analog_gain) in enumerate(capture_specs):
        capture = sensor_set(template.clone(), "analog gain", analog_gain)
        capture = sensor_set(capture, "name", name)
        capture = sensor_compute(capture, oi, seed=None if seed is None else int(seed) + index)
        captures.append(capture)

    normalized_method = param_format(method)
    if normalized_method not in {"average", "bestsnr"}:
        raise UnsupportedOptionError("imx490Compute", method)

    combined = large.clone()
    if normalized_method == "average":
        v1 = np.asarray(sensor_get(captures[0], "volts"), dtype=float)
        v2 = np.asarray(sensor_get(captures[1], "volts"), dtype=float)
        v3 = np.asarray(sensor_get(captures[2], "volts"), dtype=float)
        v4 = np.asarray(sensor_get(captures[3], "volts"), dtype=float)

        v_swing_large = float(sensor_get(large, "pixel voltage swing"))
        v_swing_small = float(sensor_get(small, "pixel voltage swing"))
        idx1 = v1 < v_swing_large
        idx2 = v2 < v_swing_large
        idx3 = v3 < v_swing_small
        idx4 = v4 < v_swing_small
        contributing = idx1.astype(int) + idx2.astype(int) + idx3.astype(int) + idx4.astype(int)

        in1 = np.asarray(sensor_get(captures[0], "electrons per area", "um"), dtype=float)
        in2 = np.asarray(sensor_get(captures[1], "electrons per area", "um"), dtype=float)
        in3 = np.asarray(sensor_get(captures[2], "electrons per area", "um"), dtype=float)
        in4 = np.asarray(sensor_get(captures[3], "electrons per area", "um"), dtype=float)

        conversion_gain = float(sensor_get(large, "pixel conversion gain"))
        safe_contributing = np.maximum(contributing, 1)
        combined_volts = conversion_gain * ((in1 + in2 + in3 + in4) / safe_contributing)
        combined_volts[contributing == 0] = 1.0

        voltage_swing = float(sensor_get(large, "pixel voltage swing"))
        max_value = float(np.max(combined_volts))
        if max_value > 0.0:
            combined_volts = voltage_swing * (combined_volts / max_value)

        combined = sensor_set(combined, "quantization method", "analog")
        combined = sensor_set(combined, "volts", combined_volts)
        combined = sensor_set(combined, "analog gain", 1.0)
        combined = sensor_set(combined, "analog offset", 0.0)
        combined.metadata["npixels"] = contributing.astype(int)
    else:
        e1 = np.asarray(sensor_get(captures[0], "electrons"), dtype=float)
        e2 = np.asarray(sensor_get(captures[1], "electrons"), dtype=float)
        e3 = np.asarray(sensor_get(captures[2], "electrons"), dtype=float)
        e4 = np.asarray(sensor_get(captures[3], "electrons"), dtype=float)

        well_large = float(sensor_get(large, "pixel well capacity"))
        well_small = float(sensor_get(small, "pixel well capacity"))
        e1 = np.where(e1 < well_large, e1, 0.0)
        e2 = np.where(e2 < well_large, e2, 0.0)
        e3 = np.where(e3 < well_small, e3, 0.0)
        e4 = np.where(e4 < well_small, e4, 0.0)

        electron_stack = np.stack([e1, e2, e3, e4], axis=2)
        best_pixel = np.argmax(electron_stack, axis=2) + 1
        best_signal = np.max(electron_stack, axis=2)

        combined_volts = best_signal * float(sensor_get(large, "pixel conversion gain"))
        combined = sensor_set(combined, "quantization method", "analog")
        combined = sensor_set(combined, "volts", combined_volts)
        combined = sensor_set(combined, "analog gain", 1.0)
        combined = sensor_set(combined, "analog offset", 0.0)
        combined.metadata["bestPixel"] = best_pixel.astype(int)

    nbits = int(sensor_get(large, "nbits"))
    voltage_swing = max(float(sensor_get(combined, "pixel voltage swing")), 1.0e-12)
    dv = (2**nbits) * (np.asarray(sensor_get(combined, "volts"), dtype=float) / voltage_swing)
    combined = sensor_set(combined, "dv", dv)
    combined = sensor_set(combined, "name", f"Combined-{normalized_method}")

    metadata = {
        "sensorArray": captures,
        "method": normalized_method,
    }
    return track_session_object(session, combined), metadata


imx490Compute = imx490_compute


def _pixel_payload_from_value(pixel: Any) -> dict[str, Any]:
    if isinstance(pixel, dict):
        return copy.deepcopy(pixel)
    if pixel is None:
        raise ValueError("pixel must be defined.")
    if isinstance(pixel, Sensor):
        return _pixel_payload_from_sensor(pixel)
    return copy.deepcopy(dict(vars(pixel)))


def _pixel_wave_from_payload(pixel: dict[str, Any]) -> np.ndarray:
    wave = pixel.get("wave")
    if wave is None:
        spectrum = pixel.get("spectrum", {})
        if isinstance(spectrum, dict):
            wave = spectrum.get("wave")
        elif spectrum is not None:
            wave = getattr(spectrum, "wave", None)
    if wave is None:
        return DEFAULT_WAVE.copy()
    return np.asarray(wave, dtype=float).reshape(-1)


def _pixel_qe_from_payload(pixel: dict[str, Any], wave: np.ndarray) -> np.ndarray:
    for key in ("pd_spectral_qe", "pixel_qe", "spectral_qe", "spectralQE", "qe"):
        stored = pixel.get(key)
        if stored is None:
            continue
        qe = np.asarray(stored, dtype=float).reshape(-1)
        if qe.size == 1:
            return np.full(wave.size, float(qe[0]), dtype=float)
        return qe
    return np.ones(wave.size, dtype=float)


def _pixel_sensor_from_value(pixel: Any) -> Sensor:
    if isinstance(pixel, Sensor):
        return pixel.clone()
    payload = _pixel_payload_from_value(pixel)
    wave = _pixel_wave_from_payload(payload)
    pixel_qe = _pixel_qe_from_payload(payload, wave)
    for key in ("wave", "pd_spectral_qe", "pixel_qe", "spectral_qe", "spectralQE", "qe"):
        payload.pop(key, None)
    sensor = _sensor_base(str(payload.get("name", "pixel")), wave, (1, 1), payload)
    sensor.fields["pixel_qe"] = pixel_qe
    sensor.fields["pixel"]["spectrum"] = _pixel_spectrum_struct(sensor)
    return sensor


def _pixel_payload_from_sensor(sensor: Sensor) -> dict[str, Any]:
    payload = copy.deepcopy(sensor.fields["pixel"])
    payload["spectrum"] = _pixel_spectrum_struct(sensor)
    payload["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
    payload["pd_spectral_qe"] = _sensor_pixel_qe(sensor).copy()
    return payload


def pixel_create(
    pixel_type: str = "default",
    wave: Any | None = None,
    pixel_size_m: Any = 2.8e-6,
) -> dict[str, Any]:
    """Create a standalone MATLAB-style pixel payload."""

    resolved_wave = DEFAULT_WAVE.copy() if _is_empty_dispatch_placeholder(wave) else np.asarray(wave, dtype=float).reshape(-1)
    resolved_type = (
        "default"
        if _is_empty_dispatch_placeholder(pixel_type)
        or (isinstance(pixel_type, str) and pixel_type.strip() == "")
        else pixel_type
    )
    normalized = param_format(resolved_type)

    if normalized in {"default", "aps"}:
        payload = _default_pixel({"name": "aps", "type": "pixel"})
    elif normalized in {"human", "humancone"}:
        payload = _default_pixel(
            {
                "name": "humancone",
                "type": "pixel",
                "size_m": np.array([2.0e-6, 2.0e-6], dtype=float),
                "pd_size_m": np.array([2.0e-6, 2.0e-6], dtype=float),
                "fill_factor": 1.0,
                "conversion_gain_v_per_electron": 1.0e-5,
                "voltage_swing": 1.0,
                "dark_voltage_v_per_sec": 1.0e-3,
                "read_noise_v": 1.0e-3,
                "layer_thickness_m": np.array([0.5e-6, 4.5e-6], dtype=float),
                "refractive_indices": np.array([1.0, 2.0, 1.46, 3.5], dtype=float),
            }
        )
        _sync_pixel_pd_state(payload)
    elif normalized in {"mouse", "mousecone"}:
        payload = _default_pixel(
            {
                "name": "mousecone",
                "type": "pixel",
                "size_m": np.array([9.0e-6, 9.0e-6], dtype=float),
                "pd_size_m": np.array([2.0e-6, 2.0e-6], dtype=float),
                "conversion_gain_v_per_electron": 1.0e-5,
                "voltage_swing": 0.2,
                "dark_voltage_v_per_sec": 0.0,
                "read_noise_v": 0.0,
            }
        )
        _sync_pixel_pd_state(payload)
    elif normalized == "ideal":
        ideal_size_m = _normalize_pixel_size_m(pixel_size_m, default=(2.8e-6, 2.8e-6))
        payload = _default_pixel(
            {
                "name": "aps",
                "type": "pixel",
                "size_m": ideal_size_m.copy(),
                "pd_size_m": ideal_size_m.copy(),
                "fill_factor": 1.0,
            }
        )
        _sync_pixel_pd_state(payload)
        payload["wave"] = resolved_wave.copy()
        payload["spectrum"] = {"wave": resolved_wave.copy()}
        payload["pd_spectral_qe"] = np.ones(resolved_wave.size, dtype=float)
        return pixel_ideal(payload)
    else:
        raise UnsupportedOptionError("pixelCreate", pixel_type)

    payload["wave"] = resolved_wave.copy()
    payload["spectrum"] = {"wave": resolved_wave.copy()}
    payload["pd_spectral_qe"] = np.ones(resolved_wave.size, dtype=float)
    return payload


def pixel_get(pixel: Any, parameter: str, *args: Any) -> Any:
    """Return MATLAB-style standalone pixel properties."""

    return _sensor_pixel_get(_pixel_sensor_from_value(pixel), parameter, *args)


def pixel_set(pixel: Any, parameter: str, value: Any) -> dict[str, Any]:
    """Set MATLAB-style standalone pixel properties."""

    sensor = _pixel_sensor_from_value(pixel)
    sensor = _sensor_pixel_set(sensor, parameter, value)
    return _pixel_payload_from_sensor(sensor)


def pixel_ideal(pixel: Any | None = None) -> dict[str, Any]:
    """Create a matched standalone pixel payload without read or dark noise."""

    payload = pixel_create("default") if pixel is None else _pixel_payload_from_value(pixel)
    payload = pixel_set(payload, "readNoiseVolts", 0.0)
    payload = pixel_set(payload, "darkVoltage", 0.0)
    return pixel_set(payload, "voltage swing", 1.0e6)


def pixel_sr(pixel: Any) -> np.ndarray:
    """Return MATLAB-style pixel spectral responsivity."""

    return _pixel_spectral_sr(_pixel_sensor_from_value(pixel))


def pixel_dr(sensor: Sensor, integration_time: Any | None = None) -> Any:
    """Return MATLAB-style pixel dynamic range in dB."""

    if _is_empty_dispatch_placeholder(integration_time):
        integration_time = None
    return _pixel_dynamic_range(sensor, integration_time)


def pixel_position_pd(pixel: Any, place: str | None = None) -> dict[str, Any]:
    """Place the photodetector within a standalone MATLAB-style pixel."""

    payload = _pixel_payload_from_value(pixel)
    normalized_place = param_format(place or "center")
    pixel_size = _pixel_size_m(payload)
    pd_size = _pixel_pd_size_from_pixel(payload)

    if normalized_place == "center":
        payload["pd_position_m"] = ((pixel_size - pd_size) / 2.0).astype(float, copy=False)
    elif normalized_place == "corner":
        payload["pd_position_m"] = np.zeros(2, dtype=float)
    else:
        raise ValueError(f"Unknown photodetector placement principle: {place}")

    _sync_pixel_pd_state(payload)
    return payload


def pixel_center_fill_pd(sensor_or_pixel: Any, fillfactor: float | int | None = None) -> Any:
    """Center a photodetector with a specified fill factor inside a pixel."""

    fill_factor = 1.0 if fillfactor is None else float(fillfactor)
    if fill_factor < 0.0 or fill_factor > 1.0:
        raise ValueError(f"Fill factor must be between 0 and 1. Parameter value = {fill_factor}")

    if isinstance(sensor_or_pixel, Sensor):
        sensor = sensor_or_pixel.clone()
        pixel = _pixel_payload_from_sensor(sensor)
        pixel = pixel_set(pixel, "pd width", np.sqrt(fill_factor) * float(pixel_get(pixel, "deltax")))
        pixel = pixel_set(pixel, "pd height", np.sqrt(fill_factor) * float(pixel_get(pixel, "deltay")))
        pixel = pixel_position_pd(pixel, "center")
        return sensor_set(sensor, "pixel", pixel)

    pixel = _pixel_payload_from_value(sensor_or_pixel)
    pixel = pixel_set(pixel, "pd width", np.sqrt(fill_factor) * float(pixel_get(pixel, "deltax")))
    pixel = pixel_set(pixel, "pd height", np.sqrt(fill_factor) * float(pixel_get(pixel, "deltay")))
    return pixel_position_pd(pixel, "center")


def pv_full_overlap(optics: dict[str, Any], pixel: Any) -> tuple[np.ndarray, int, int]:
    """Compute the legacy full photodiode/spot overlap kernel for pixel vignetting."""

    from .optics import optics_get

    focal_length = float(optics_get(optics, "focal length"))
    diameter = float(optics_get(optics, "aperture diameter"))
    width = float(pixel_get(pixel, "width"))
    depth = float(pixel_get(pixel, "depth"))
    spot_diameter = diameter * (depth / max(focal_length, 1e-30))

    num_grid = int(np.floor(51.0 * width / max(spot_diameter, 1e-30)))
    num_grid = max(num_grid, 1)
    if num_grid % 2 == 0:
        num_grid += 1

    diode_grid = np.zeros((num_grid, num_grid), dtype=float)
    x, y = np.meshgrid(
        np.linspace(-width / 2.0, width / 2.0, num_grid, dtype=float),
        np.linspace(-width / 2.0, width / 2.0, num_grid, dtype=float),
    )
    diode_mask = (np.abs(x) < (width / 2.0)) & (np.abs(y) < (width / 2.0))
    diode_grid[diode_mask] = 1.0
    diode_grid /= max(float(np.count_nonzero(diode_mask)), 1.0)

    num_grid_spot = int(np.floor(spot_diameter / max(width / max(num_grid, 1), 1e-30)))
    num_grid_spot = max(num_grid_spot, 1)
    if num_grid_spot % 2 == 0:
        num_grid_spot += 1

    spot = np.zeros((num_grid_spot, num_grid_spot), dtype=float)
    x, y = np.meshgrid(
        np.linspace(-spot_diameter / 2.0, spot_diameter / 2.0, num_grid_spot, dtype=float),
        np.linspace(-spot_diameter / 2.0, spot_diameter / 2.0, num_grid_spot, dtype=float),
    )
    spot_mask = np.sqrt(x**2 + y**2) < (spot_diameter / 2.0)
    spot[spot_mask] = 1.0
    spot /= max(float(np.count_nonzero(spot_mask)), 1.0)

    overlap = convolve2d(diode_grid, spot, mode="full")
    return np.asarray(overlap, dtype=float), num_grid, num_grid_spot


def pv_reduction(
    overlap: np.ndarray,
    num_grid: int,
    num_grid_spot: int,
    diode_location: Any,
    outside_angle: Any,
    optics: dict[str, Any],
    pixel: Any,
) -> np.ndarray:
    """Compute the legacy pixel-vignetting efficiency reduction for a diode position."""

    from .optics import optics_get

    _ = int(num_grid_spot)
    focal_length = float(optics_get(optics, "focal length"))
    width = float(pixel_get(pixel, "width"))
    depth = float(pixel_get(pixel, "depth"))

    overlap_array = np.asarray(overlap, dtype=float)
    diode_center = np.asarray(diode_location, dtype=float)
    if diode_center.ndim == 1:
        diode_center = diode_center.reshape(1, -1)
    angle = np.asarray(outside_angle, dtype=float)
    if angle.ndim == 1:
        angle = angle.reshape(1, -1)

    diode_center_x = diode_center[:, 0]
    diode_center_y = diode_center[:, 1]
    spot_center_x = (focal_length - depth) * np.tan(angle[:, 0])
    spot_center_y = (focal_length - depth) * np.tan(angle[:, 1])
    offset_x = diode_center_x - spot_center_x
    offset_y = diode_center_y - spot_center_y

    index_offset_x = np.rint((offset_x / max(width, 1e-30)) * float(num_grid)).astype(int)
    index_offset_y = np.rint((offset_y / max(width, 1e-30)) * float(num_grid)).astype(int)
    index_center_x = ((overlap_array.shape[1] - 1) / 2.0) + 1.0
    index_center_y = ((overlap_array.shape[0] - 1) / 2.0) + 1.0
    support = np.arange(-(num_grid - 1) / 2.0, ((num_grid - 1) / 2.0) + 1.0, dtype=float)

    ratios = np.empty(diode_center.shape[0], dtype=float)
    for idx in range(diode_center.shape[0]):
        index_range_x = np.rint(index_center_x - index_offset_x[idx] + support).astype(int)
        index_range_y = np.rint(index_center_y - index_offset_y[idx] + support).astype(int)
        index_range_x[index_range_x <= 0] = 1
        index_range_y[index_range_y <= 0] = 1
        index_range_x[index_range_x > overlap_array.shape[1]] = 1
        index_range_y[index_range_y > overlap_array.shape[0]] = 1
        overlap_view = overlap_array[np.ix_(index_range_x - 1, index_range_y - 1)]
        ratios[idx] = float(np.sum(overlap_view))

    return ratios if ratios.size > 1 else np.asarray(ratios[0], dtype=float)


def _block_apply(array: np.ndarray, block_shape: tuple[int, int], fn: Any) -> np.ndarray:
    rows, cols = array.shape[:2]
    block_rows, block_cols = block_shape
    pieces: list[np.ndarray] = []
    for row_start in range(0, rows, block_rows):
        row_blocks: list[np.ndarray] = []
        for col_start in range(0, cols, block_cols):
            block = np.asarray(array[row_start : row_start + block_rows, col_start : col_start + block_cols], dtype=float)
            row_blocks.append(np.asarray(fn(block), dtype=float))
        pieces.append(np.concatenate(row_blocks, axis=1))
    return np.concatenate(pieces, axis=0)


def bin_pixel(sensor: Sensor, b_method: str | None = None) -> Sensor:
    """Apply the legacy first-stage pixel-binning transform to sensor image data."""

    current = sensor.clone()
    method = param_format(b_method or "kodak2008")

    if method == "kodak2008":
        volts = np.asarray(sensor_get(current, "volts"), dtype=float)
        dv = _block_apply(volts, (4, 4), lambda x: x @ np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))
        return sensor_set(current, "digitalValues", dv)
    if method == "addadjacentblocks":
        volts = np.asarray(sensor_get(current, "volts"), dtype=float)
        left = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=float)
        right = left.T
        dv = _block_apply(volts, (4, 4), lambda x: left @ x @ right)
        return sensor_set(current, "digitalValues", dv)
    if method == "averageadjacentdigitalblocks":
        return sensor_set(current, "digitalValues", 1)
    raise ValueError(f"Unknown binning method {b_method}")


def bin_pixel_post(sensor: Sensor, b_method: str) -> Sensor:
    """Apply the legacy second-stage digital pixel-binning transform."""

    current = sensor.clone()
    method = param_format(b_method)

    if method == "kodak2008":
        dv = np.asarray(sensor_get(current, "dv"), dtype=float)
        left = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=float)
        reduced = _block_apply(dv, (4, 2), lambda x: np.rint(0.5 * (left @ x)))
        return sensor_set(current, "digitalValues", reduced)
    if method == "averageadjacentdigitalblocks":
        dv = np.asarray(sensor_get(current, "dv"), dtype=float)
        left = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=float)
        reduced = _block_apply(dv, (4, 4), lambda x: 0.25 * (left @ x @ left.T))
        return sensor_set(current, "digitalValues", reduced)
    return current


def pixel_description(pixel: Any, sensor: Sensor | None = None) -> str:
    """Return a headless MATLAB-style pixel summary string."""

    payload = _pixel_payload_from_value(pixel)
    lines = [
        f"Pixel (H,W):\t({float(pixel_get(payload, 'deltay', 'um')):.1f},{float(pixel_get(payload, 'deltax', 'um')):.1f}) um",
        f"PD (H,W):\t({float(pixel_get(payload, 'pdheight', 'um')):.1f}, {float(pixel_get(payload, 'pdwidth', 'um')):.1f}) um",
        f"Fill percentage:\t{float(pixel_get(payload, 'fillfactor')) * 100:.0f}",
        f"Well capacity\t{float(pixel_get(payload, 'wellcapacity')):.0f} e-",
    ]

    if sensor is not None:
        dr = sensor_dr(sensor, 0.001)
        if dr is not None and np.asarray(dr).size > 0:
            lines.append(f"DR (1 ms):\t{float(np.asarray(dr, dtype=float).reshape(-1)[0]):.1f} dB")
        peak_voltage = float(pixel_get(payload, "voltageswing"))
        snr_peak = pixel_snr(sensor, peak_voltage)[0]
        lines.append(f"Peak SNR:\t{float(np.asarray(snr_peak, dtype=float).reshape(-1)[0]):.0f} dB")
        sensor_wave = np.asarray(sensor_get(sensor, "wavelength"), dtype=float).reshape(-1)
        pixel_wave = np.asarray(pixel_get(payload, "wavelength"), dtype=float).reshape(-1)
        if sensor_wave.size != pixel_wave.size or not np.array_equal(sensor_wave, pixel_wave):
            lines.append("Wave rep mismatch!!")

    return "\n".join(lines)


def _interp_linear_with_extrapolation(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    result = np.interp(x_new, x, y)
    if x.size >= 2:
        left_mask = x_new < x[0]
        if np.any(left_mask):
            left_slope = (y[1] - y[0]) / max(x[1] - x[0], 1e-30)
            result[left_mask] = y[0] + (x_new[left_mask] - x[0]) * left_slope
        right_mask = x_new > x[-1]
        if np.any(right_mask):
            right_slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-30)
            result[right_mask] = y[-1] + (x_new[right_mask] - x[-1]) * right_slope
    return result


def ie_pixel_well_capacity(
    pixel_size_um: Any | None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[float | np.ndarray | None, np.ndarray]:
    """Estimate full well capacity from pixel size in microns."""

    store = _store(asset_store)
    well_capacity = np.asarray(store.load_mat("data/sensor/wellCapacity.mat")["wellCapacity"], dtype=float)
    if pixel_size_um is None:
        return None, well_capacity
    sizes = np.asarray(pixel_size_um, dtype=float)
    if sizes.size == 0:
        return None, well_capacity

    flat_sizes = sizes.reshape(-1)
    electrons = _interp_linear_with_extrapolation(well_capacity[:, 0], well_capacity[:, 1], flat_sizes)
    reshaped = electrons.reshape(sizes.shape)
    if reshaped.ndim == 0:
        return float(reshaped), well_capacity
    if reshaped.size == 1:
        return float(reshaped.reshape(-1)[0]), well_capacity
    return reshaped, well_capacity


def _snr_voltage_levels(pixel: dict[str, Any], volts: Any | None) -> np.ndarray:
    if volts is None:
        return np.logspace(-4.0, 0.0, 50, dtype=float) * max(float(pixel["voltage_swing"]), 1e-12)
    return np.asarray(volts, dtype=float)


def _snr_db(signal_power: np.ndarray, noise_power: np.ndarray) -> np.ndarray:
    safe_noise = np.maximum(np.asarray(noise_power, dtype=float), 1e-30)
    return 10.0 * np.log10(np.asarray(signal_power, dtype=float) / safe_noise)


def pixel_snr(sensor: Sensor, volts: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray]:
    """Compute MATLAB-style pixel SNR curves in dB over voltage levels."""

    pixel = sensor.fields["pixel"]
    voltage_levels = _snr_voltage_levels(pixel, volts)
    conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-12)
    read_sd = float(pixel["read_noise_v"]) / conversion_gain

    electron_mean = np.maximum(np.asarray(voltage_levels, dtype=float) / conversion_gain, 0.0)
    shot_sd = np.sqrt(electron_mean)
    signal_power = np.square(np.asarray(voltage_levels, dtype=float) / conversion_gain)
    snr = _snr_db(signal_power, np.square(read_sd) + np.square(shot_sd))
    snr_shot = _snr_db(signal_power, np.square(shot_sd))
    snr_read: float | np.ndarray
    if np.isclose(read_sd, 0.0):
        snr_read = float(np.inf)
    else:
        snr_read = _snr_db(signal_power, np.square(read_sd))
    return snr, np.asarray(voltage_levels, dtype=float), snr_shot, snr_read


def pixel_v_per_lux_sec(
    sensor: Sensor,
    light_type: str = "ee",
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Compute MATLAB-style pixel photometric sensitivity in volts per lux-second."""

    from .optics import oi_calculate_illuminance, oi_compute, oi_create
    from .scene import scene_create

    store = _store(asset_store)
    normalized_light = param_format(light_type or "ee")
    scene_name = "uniform d65" if normalized_light == "d65" else "uniform"
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)

    scene = scene_create(scene_name, 32, wave, asset_store=store)
    oi = oi_create("pinhole", wave, asset_store=store)
    oi = oi_compute(oi, scene)
    _, lux, anti_lux = oi_calculate_illuminance(oi)

    saturation_exposure_time = float(_auto_exposure_default(sensor, oi))
    signal_sensor = sensor_set(sensor, "integration time", saturation_exposure_time)
    signal_sensor = sensor_compute(signal_sensor, oi, seed=0)

    n_colors = int(sensor_get(signal_sensor, "ncolors"))
    mean_volts = np.zeros(n_colors, dtype=float)
    for color_index in range(1, n_colors + 1):
        mean_volts[color_index - 1] = float(np.mean(np.asarray(sensor_get(signal_sensor, "volts", color_index), dtype=float)))

    luxsec = float(lux) * saturation_exposure_time
    anti_luxsec = float(anti_lux) * saturation_exposure_time
    volts_per_lux_sec = np.divide(
        mean_volts,
        luxsec,
        out=np.full_like(mean_volts, np.inf),
        where=not np.isclose(luxsec, 0.0),
    )
    volts_per_anti_lux_sec = np.divide(
        mean_volts,
        anti_luxsec,
        out=np.full_like(mean_volts, np.inf),
        where=not np.isclose(anti_luxsec, 0.0),
    )
    return volts_per_lux_sec, luxsec, mean_volts, volts_per_anti_lux_sec, anti_luxsec


def pixel_snr_luxsec(
    sensor: Sensor,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray, np.ndarray]:
    """Compute MATLAB-style pixel SNR curves in dB over lux-second levels."""

    snr, volts, snr_shot, snr_read = pixel_snr(sensor)
    volts_per_lux_sec, _, _, volts_per_anti_lux_sec, _ = pixel_v_per_lux_sec(sensor, asset_store=asset_store)

    volts_column = np.asarray(volts, dtype=float).reshape(-1, 1)
    luxsec = np.divide(
        volts_column,
        np.asarray(volts_per_lux_sec, dtype=float).reshape(1, -1),
        out=np.zeros((volts_column.shape[0], np.asarray(volts_per_lux_sec).size), dtype=float),
        where=~np.isclose(np.asarray(volts_per_lux_sec, dtype=float).reshape(1, -1), 0.0),
    )
    anti_luxsec = np.divide(
        volts_column,
        np.asarray(volts_per_anti_lux_sec, dtype=float).reshape(1, -1),
        out=np.zeros((volts_column.shape[0], np.asarray(volts_per_anti_lux_sec).size), dtype=float),
        where=np.isfinite(np.asarray(volts_per_anti_lux_sec, dtype=float).reshape(1, -1))
        & ~np.isclose(np.asarray(volts_per_anti_lux_sec, dtype=float).reshape(1, -1), 0.0),
    )
    return snr, luxsec, snr_shot, snr_read, anti_luxsec


def pt_snells_law(n: Any, theta_in: Any) -> np.ndarray:
    """Return the MATLAB-style Snell-law angle grid for a refractive-index stack."""

    indices = np.asarray(n, dtype=np.complex128).reshape(-1)
    incoming = np.asarray(theta_in, dtype=np.complex128).reshape(-1)
    reduced_theta = indices[0] * np.sin(incoming)
    reduced_grid, index_grid = np.meshgrid(reduced_theta, indices)
    return np.asarray(np.arcsin(reduced_grid / index_grid), dtype=np.complex128)[None, :, :]


def pt_reflection_and_transmission(
    n_in: Any,
    n_out: Any,
    theta_in: Any,
    polarization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the MATLAB-style Fresnel reflection/transmission coefficients."""

    n_in_value = np.asarray(n_in, dtype=np.complex128).reshape(-1)[0]
    n_out_value = np.asarray(n_out, dtype=np.complex128).reshape(-1)[0]
    theta_in_value = np.asarray(theta_in, dtype=np.complex128)
    theta_out = np.arcsin(n_in_value * np.sin(theta_in_value) / n_out_value)

    normalized = param_format(polarization)
    if normalized == "s":
        denominator = (n_in_value * np.cos(theta_in_value)) + (n_out_value * np.cos(theta_out))
        rho = ((n_in_value * np.cos(theta_in_value)) - (n_out_value * np.cos(theta_out))) / denominator
        tau = (2.0 * n_in_value * np.cos(theta_in_value)) / denominator
        return np.asarray(rho, dtype=np.complex128), np.asarray(tau, dtype=np.complex128)
    if normalized == "p":
        denominator = (n_out_value * np.cos(theta_in_value)) + (n_in_value * np.cos(theta_out))
        rho = ((n_out_value * np.cos(theta_in_value)) - (n_in_value * np.cos(theta_out))) / denominator
        tau = (2.0 * n_in_value * np.cos(theta_in_value)) / denominator
        return np.asarray(rho, dtype=np.complex128), np.asarray(tau, dtype=np.complex128)
    raise UnsupportedOptionError("ptReflectionAndTransmission", polarization)


def pt_interface_matrix(rho: Any, tau: Any) -> np.ndarray:
    """Return the MATLAB-style 2x2xN interface matrix stack."""

    rho_values = np.asarray(rho, dtype=np.complex128).reshape(-1)
    tau_values = np.asarray(tau, dtype=np.complex128).reshape(-1)
    if rho_values.size != tau_values.size:
        raise ValueError("rho and tau must have the same number of samples.")
    interface = np.zeros((2, 2, rho_values.size), dtype=np.complex128)
    interface[0, 0, :] = 1.0 / tau_values
    interface[0, 1, :] = rho_values / tau_values
    interface[1, 0, :] = rho_values / tau_values
    interface[1, 1, :] = 1.0 / tau_values
    return interface


def pt_propagation_matrix(n: Any, d: Any, theta: Any, lambda_value: Any) -> np.ndarray:
    """Return the MATLAB-style 2x2xN propagation matrix for one layer."""

    index_value = np.asarray(n, dtype=np.complex128).reshape(-1)[0]
    distance_value = float(np.asarray(d, dtype=float).reshape(-1)[0])
    theta_values = np.asarray(theta, dtype=np.complex128).reshape(-1)
    wavelength_m = float(np.asarray(lambda_value, dtype=float).reshape(-1)[0])
    wave_number = (2.0 * np.pi / max(wavelength_m, 1.0e-30)) * index_value
    phase = np.exp(1j * wave_number * distance_value * np.cos(theta_values))
    propagation = np.zeros((2, 2, theta_values.size), dtype=np.complex128)
    propagation[0, 0, :] = phase
    propagation[1, 1, :] = np.exp(-1j * wave_number * distance_value * np.cos(theta_values))
    return propagation


def pt_scattering_matrix(
    n: Any,
    d: Any,
    theta_in: Any,
    lambda_value: Any,
    polarization: str,
) -> np.ndarray:
    """Return the MATLAB-style multilayer scattering matrix over incidence angle."""

    indices = np.asarray(n, dtype=np.complex128).reshape(-1)
    thickness = np.asarray(d, dtype=float).reshape(-1)
    theta = pt_snells_law(indices, theta_in)

    interface_count = max(indices.size - 1, 0)
    interfaces = []
    for interface_index in range(interface_count):
        rho, tau = pt_reflection_and_transmission(
            indices[interface_index],
            indices[interface_index + 1],
            theta[0, interface_index, :],
            polarization,
        )
        interfaces.append(pt_interface_matrix(rho, tau))

    propagation_count = thickness.size
    propagations = []
    for propagation_index in range(propagation_count):
        propagations.append(
            pt_propagation_matrix(
                indices[propagation_index + 1],
                thickness[propagation_index],
                theta[0, propagation_index + 1, :],
                lambda_value,
            )
        )

    scattering = np.zeros((2, 2, theta.shape[2]), dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    for angle_index in range(theta.shape[2]):
        current = interfaces[0][:, :, angle_index] if interfaces else identity.copy()
        for interface_index in range(1, interface_count):
            layer_index = interface_index - 1
            if layer_index < propagation_count:
                current = current @ propagations[layer_index][:, :, angle_index]
            current = current @ interfaces[interface_index][:, :, angle_index]
        if propagation_count == interface_count and propagation_count > 0:
            current = current @ propagations[-1][:, :, angle_index]
        scattering[:, :, angle_index] = current
    return scattering


def pt_poynting_factor(n: Any, theta_in: Any) -> np.ndarray:
    """Return the MATLAB-style Poynting correction factor for the output medium."""

    indices = np.asarray(n, dtype=np.complex128).reshape(-1)
    theta = pt_snells_law(indices, theta_in)
    if np.isclose(np.imag(indices[-1]), 0.0):
        factor = (indices[-1] * np.cos(theta[0, -1, :])) / (indices[0] * np.cos(theta[0, 0, :]))
        return np.asarray(np.real_if_close(factor), dtype=np.complex128).reshape(-1)

    eta = np.abs(indices[-1] * np.cos(theta[0, -1, :]))
    beta = np.angle(indices[-1] * np.cos(theta[0, -1, :]))
    n_t = np.sqrt((indices[0] ** 2) * np.sin(theta[0, 0, :]) ** 2 + eta**2 * np.cos(beta) ** 2)
    cos_theta_t = np.sqrt(1.0 - ((indices[0] * np.sin(theta[0, 0, :])) / n_t) ** 2)
    factor = (np.real(indices[-1]) * np.real(cos_theta_t)) / (indices[0] * np.cos(theta[0, 0, :]))
    return np.asarray(np.real_if_close(factor), dtype=np.complex128).reshape(-1)


def pt_transmittance(n: Any, d: Any, lambda_nm: Any, theta: Any) -> dict[str, Any]:
    """Return the MATLAB-style pixel tunnel transmittance summary."""

    indices = np.asarray(n, dtype=np.complex128).reshape(-1)
    thickness = np.asarray(d, dtype=float).reshape(-1)
    wavelengths_nm = np.asarray(lambda_nm, dtype=float).reshape(-1)
    theta_values = np.asarray(theta, dtype=float).reshape(-1)
    wavelengths_m = wavelengths_nm * 1.0e-9

    transmission = np.empty((theta_values.size, wavelengths_nm.size), dtype=float)
    poynting = np.asarray(pt_poynting_factor(indices, theta_values), dtype=np.complex128).reshape(-1)
    for wave_index, wavelength_m in enumerate(wavelengths_m):
        scattering_s = pt_scattering_matrix(indices, thickness, theta_values, wavelength_m, "s")
        scattering_p = pt_scattering_matrix(indices, thickness, theta_values, wavelength_m, "p")
        band = (np.abs(1.0 / scattering_s[0, 0, :]) ** 2 + np.abs(1.0 / scattering_p[0, 0, :]) ** 2) / 2.0
        transmission[:, wave_index] = np.asarray(np.real_if_close(band * poynting), dtype=float)

    spectral = np.mean(transmission, axis=0, dtype=float)
    average = np.mean(transmission, axis=1, dtype=float)
    return {
        "transmission": {
            "average": np.asarray(average, dtype=float),
            "spectral": np.asarray(spectral, dtype=float),
            "spectra": np.asarray(spectral, dtype=float),
            "data": np.asarray(transmission, dtype=float),
        },
        "wave": wavelengths_nm.copy(),
        "theta": theta_values.copy(),
    }


def pixel_transmittance(
    optical_image: OpticalImage,
    pixel: Any,
    optics: dict[str, Any],
) -> tuple[OpticalImage, Any, dict[str, Any]]:
    """Apply the legacy pixel transmittance stack to an optical image."""

    current_oi = optical_image.clone()
    photons = np.asarray(oi_get(current_oi, "photons"), dtype=float)
    wave = np.asarray(oi_get(current_oi, "wave"), dtype=float).reshape(-1)
    if photons.ndim != 3 or photons.shape[2] != wave.size:
        raise ValueError("pixelTransmittance requires an optical image with spectral photons.")

    current_pixel = pixel.clone() if isinstance(pixel, Sensor) else _pixel_payload_from_value(pixel)
    current_optics = dict(optics)
    f_number = float(current_optics.get("f_number", current_optics.get("fNumber", 4.0)))
    incidence_angles = np.linspace(-np.arctan(1.0 / (2.0 * max(f_number, 1.0e-30))), np.arctan(1.0 / (2.0 * max(f_number, 1.0e-30))), 25)

    refractive_index = np.asarray(pixel_get(current_pixel, "refractiveindex"), dtype=np.complex128).reshape(-1)
    layer_thickness = np.asarray(pixel_get(current_pixel, "layerthickness"), dtype=float).reshape(-1)
    tunnel = pt_transmittance(refractive_index, layer_thickness, wave, incidence_angles)
    spectral = np.asarray(tunnel["transmission"]["spectral"], dtype=float).reshape(1, 1, -1)

    current_oi.data["photons"] = photons * spectral
    current_oi.fields["pixel_transmittance"] = copy.deepcopy(tunnel)
    current_oi.fields.pop("illuminance", None)
    return current_oi, current_pixel, current_optics


def sensor_snr(
    sensor: Sensor,
    volts: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Compute MATLAB-style sensor SNR curves in dB over voltage levels."""

    pixel = sensor.fields["pixel"]
    voltage_levels = _snr_voltage_levels(pixel, volts)
    conversion_gain = max(float(pixel["conversion_gain_v_per_electron"]), 1e-12)
    read_sd = float(pixel["read_noise_v"]) / conversion_gain
    dsnu_sd = float(pixel["dsnu_sigma_v"]) / conversion_gain
    prnu_gain_sd = float(pixel["prnu_sigma"])

    electron_mean = np.maximum(np.asarray(voltage_levels, dtype=float) / conversion_gain, 0.0)
    shot_sd = np.sqrt(electron_mean)
    prnu_sd = prnu_gain_sd * (np.asarray(voltage_levels, dtype=float) / conversion_gain)
    signal_power = np.square(np.asarray(voltage_levels, dtype=float) / conversion_gain)
    noise_power = np.square(shot_sd) + np.square(read_sd) + np.square(dsnu_sd) + np.square(prnu_sd)

    snr = _snr_db(signal_power, noise_power)
    snr_shot = _snr_db(signal_power, np.square(shot_sd))
    if np.isclose(read_sd, 0.0):
        snr_read: float | np.ndarray = float(np.inf)
    else:
        snr_read = _snr_db(signal_power, np.square(read_sd))
    if np.isclose(dsnu_sd, 0.0):
        snr_dsnu: float | np.ndarray = float(np.inf)
    else:
        snr_dsnu = _snr_db(signal_power, np.square(dsnu_sd))
    if np.isclose(prnu_gain_sd, 0.0):
        snr_prnu: float | np.ndarray = float(np.inf)
    else:
        snr_prnu = _snr_db(signal_power, np.square(prnu_sd))

    return snr, np.asarray(voltage_levels, dtype=float), snr_shot, snr_read, snr_dsnu, snr_prnu


def sensor_snr_luxsec(
    sensor: Sensor,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MATLAB-style sensor SNR curves in dB over lux-second levels."""

    snr, volts, *_ = sensor_snr(sensor)
    volts_per_lux_sec, _, _, _, _ = pixel_v_per_lux_sec(sensor, asset_store=asset_store)

    volts_column = np.asarray(volts, dtype=float).reshape(-1, 1)
    luxsec = np.divide(
        volts_column,
        np.asarray(volts_per_lux_sec, dtype=float).reshape(1, -1),
        out=np.zeros((volts_column.shape[0], np.asarray(volts_per_lux_sec).size), dtype=float),
        where=~np.isclose(np.asarray(volts_per_lux_sec, dtype=float).reshape(1, -1), 0.0),
    )
    return np.asarray(snr, dtype=float), luxsec


def sensor_mpe30(sensor: Sensor) -> float | np.ndarray:
    """Return the lux-second level where the sensor reaches 30 dB SNR."""

    snr, luxsec = sensor_snr_luxsec(sensor)
    snr_axis = np.asarray(snr, dtype=float).reshape(-1)
    luxsec_array = np.asarray(luxsec, dtype=float)
    if luxsec_array.ndim == 1 or luxsec_array.shape[1] == 1:
        return float(np.interp(30.0, snr_axis, luxsec_array.reshape(-1)))
    return np.asarray(
        [np.interp(30.0, snr_axis, luxsec_array[:, index]) for index in range(luxsec_array.shape[1])],
        dtype=float,
    )


def sensor_display_transform(sensor: Sensor) -> np.ndarray:
    return _sensor_display_transform(sensor)


def sensor_equate_transmittances(filters: Any) -> np.ndarray:
    array = np.asarray(filters, dtype=float)
    if array.ndim != 2:
        raise ValueError("sensorEquateTransmittances expects a 2D filter matrix.")
    filter_area = np.sum(array, axis=0, keepdims=True)
    balanced = np.divide(
        array,
        np.maximum(filter_area, 1e-12),
        out=np.zeros_like(array),
        where=filter_area > 0.0,
    )
    filter_peak = max(float(np.max(balanced)), 1e-12)
    return balanced / filter_peak


def sensor_filter_rgb(sensor: Sensor, saturation: float = 1.0) -> np.ndarray:
    block_matrix = _color_block_matrix(np.asarray(sensor_get(sensor, "wave"), dtype=float), extrap_val=0.2)
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    filter_spectra = np.asarray(sensor_get(sensor, "filterSpectra"), dtype=float)
    background = np.full(3, 0.94, dtype=float)
    saturation_value = float(saturation)
    filter_rgb = np.zeros(pattern.shape + (3,), dtype=float)

    for row in range(pattern.shape[0]):
        for col in range(pattern.shape[1]):
            color_filter = filter_spectra[:, int(pattern[row, col]) - 1]
            rgb = np.asarray(block_matrix.T @ color_filter, dtype=float).reshape(-1)
            rgb = rgb / max(float(np.max(rgb)), 1e-12)
            filter_rgb[row, col, :] = rgb * saturation_value + (1.0 - saturation_value) * background

    return filter_rgb


def _sensor_line_profile(sensor: Sensor, line_key: str, rc: Any) -> dict[str, list[np.ndarray]] | None:
    from .roi import _sensor_signal_cube

    normalized = param_format(line_key)
    if normalized not in {"hlinevolts", "hlineelectrons", "hlinedv", "vlinevolts", "vlineelectrons", "vlinedv"}:
        raise KeyError(f"Unsupported sensor line profile parameter: {line_key}")
    data_type = "electrons" if "electrons" in normalized else "dv" if "dv" in normalized else "volts"
    cube = _sensor_signal_cube(sensor, data_type)
    support = sensor_get(sensor, "spatial support")
    nfilters = int(sensor_get(sensor, "nfilters"))
    line_index = int(np.rint(float(rc)))
    if normalized.startswith("h"):
        if line_index < 1 or line_index > cube.shape[0]:
            raise IndexError("Horizontal sensor line index is out of range.")
        position = np.asarray(support["x"], dtype=float)
        oriented = cube[line_index - 1, :, :]
    else:
        if line_index < 1 or line_index > cube.shape[1]:
            raise IndexError("Vertical sensor line index is out of range.")
        position = np.asarray(support["y"], dtype=float)
        oriented = cube[:, line_index - 1, :]

    data: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for filter_index in range(nfilters):
        channel = np.asarray(oriented[:, filter_index], dtype=float)
        valid = ~np.isnan(channel)
        data.append(channel[valid].copy())
        positions.append(position[valid].copy())
    return {"data": data, "pos": positions, "pixPos": [values.copy() for values in positions]}


def _sensor_rect_for_chromaticity(rect_or_locs: Any) -> np.ndarray | None:
    if rect_or_locs is None:
        return None
    rect_array = np.asarray(rect_or_locs)
    if rect_array.size == 0:
        return None
    if rect_array.ndim == 1 and rect_array.size == 4:
        rect = np.rint(rect_array.astype(float)).astype(int)
    else:
        from .roi import ie_locs2_rect

        rect = ie_locs2_rect(rect_array)
    rect = rect.reshape(-1).astype(int)
    if rect.size != 4:
        raise ValueError("Chromaticity ROI must be [col, row, width, height] or Nx2 locations.")
    even = (rect % 2) == 0
    rect[even] -= 1
    rect[:2] = np.maximum(rect[:2], 1)
    rect[2:] = np.maximum(rect[2:], 0)
    return rect


def _sensor_crop_rect(data: np.ndarray | None, rect: np.ndarray | None) -> np.ndarray | None:
    if data is None or rect is None:
        return None if data is None else np.asarray(data).copy()
    array = np.asarray(data)
    row_start = max(int(rect[1]) - 1, 0)
    col_start = max(int(rect[0]) - 1, 0)
    row_end = min(int(rect[1] + rect[3]), array.shape[0])
    col_end = min(int(rect[0] + rect[2]), array.shape[1])
    return np.asarray(array[row_start:row_end, col_start:col_end, ...], dtype=float).copy()


def _sensor_cfa_crop_rect(sensor: Sensor, rect: Any) -> np.ndarray:
    rect_array = np.rint(np.asarray(rect, dtype=float).reshape(-1)[:4]).astype(int)
    if rect_array.size != 4:
        raise ValueError("Crop rect must be [x, y, width, height].")

    cfa_rows, cfa_cols = _sensor_unit_block(sensor)
    cfa_xy = np.array([max(int(cfa_cols), 1), max(int(cfa_rows), 1)], dtype=int)
    adjusted = rect_array.copy()
    for axis in range(2):
        start_index = axis
        size_index = axis + 2
        remainder = adjusted[start_index] % cfa_xy[axis]
        if remainder != 1:
            if remainder == 0:
                adjusted[start_index] += 1
            else:
                adjusted[start_index] += cfa_xy[axis] - remainder + 1
        count = adjusted[size_index] + 1
        remainder = count % cfa_xy[axis]
        if remainder != 0:
            adjusted[size_index] += cfa_xy[axis] - remainder

    adjusted[:2] = np.maximum(adjusted[:2], 1)
    adjusted[2:] = np.maximum(adjusted[2:], 0)
    return adjusted


def sensor_crop(sensor: Sensor, rect: Any) -> Sensor:
    """Crop a sensor while preserving CFA alignment."""

    adjusted_rect = _sensor_cfa_crop_rect(sensor, rect)
    cropped = sensor.clone()

    volts = sensor.data.get("volts")
    dv = sensor.data.get("dv")
    new_size: tuple[int, int] | None = None
    if volts is not None:
        new_volts = _sensor_crop_rect(volts, adjusted_rect)
        if new_volts is not None:
            cropped.data["volts"] = new_volts
            new_size = tuple(int(value) for value in new_volts.shape[:2])
    if dv is not None:
        new_dv = _sensor_crop_rect(dv, adjusted_rect)
        if new_dv is not None:
            cropped.data["dv"] = np.rint(new_dv).astype(np.asarray(dv).dtype, copy=False)
            if new_size is None:
                new_size = tuple(int(value) for value in new_dv.shape[:2])
    if new_size is not None:
        cropped.fields["size"] = new_size
    cropped.metadata["crop"] = adjusted_rect.copy()
    return cropped


def sensor_clear_data(sensor: Sensor) -> Sensor:
    """Clear the computed sensor payload while preserving configuration."""

    cleared = sensor.clone()
    _sensor_clear_data(cleared)
    return cleared


def sensor_jiggle(sensor: Sensor, pixels: Any) -> Sensor:
    """Return the current sensor unchanged, matching the upstream no-op helper."""

    del pixels
    return sensor.clone()


def sensor_vignetting(
    sensor: Sensor | None = None,
    pv_flag: Any | None = None,
    n_angles: Any = 5,
    *,
    asset_store: AssetStore | None = None,
) -> Sensor:
    """Gateway wrapper for the legacy pixel-vignetting/etendue computation."""

    current_sensor = sensor
    if current_sensor is None:
        current_sensor = sensor_create(asset_store=asset_store)

    flag_value = sensor_get(current_sensor, "vignetting") if _is_empty_dispatch_placeholder(pv_flag) else pv_flag
    if _is_empty_dispatch_placeholder(flag_value):
        flag_value = 0
    angle_count = 5 if _is_empty_dispatch_placeholder(n_angles) else int(np.asarray(n_angles).reshape(-1)[0])

    if isinstance(flag_value, str):
        normalized = param_format(flag_value)
    else:
        normalized = int(np.asarray(flag_value).reshape(-1)[0])

    if normalized in {0, "skip"}:
        size = tuple(np.asarray(sensor_get(current_sensor, "size"), dtype=int).reshape(-1))
        return sensor_set(current_sensor.clone(), "etendue", np.ones(size, dtype=float))
    if normalized in {1, "bare", "nomicrolens"}:
        return ml_analyze_array_etendue(current_sensor, "no microlens", angle_count, asset_store=asset_store)
    if normalized in {2, "centered"}:
        return ml_analyze_array_etendue(current_sensor, "centered", angle_count, asset_store=asset_store)
    if normalized in {3, "optimal", "optimized"}:
        return ml_analyze_array_etendue(current_sensor, "optimal", angle_count, asset_store=asset_store)
    raise UnsupportedOptionError("sensorVignetting", str(flag_value))


def sensor_no_noise(sensor: Sensor) -> Sensor:
    """Return a sensor copy with all non-photon-noise terms disabled."""

    noiseless = sensor.clone()
    noiseless = sensor_set(noiseless, "prnulevel", 0.0)
    noiseless = sensor_set(noiseless, "dsnulevel", 0.0)
    noiseless = sensor_set(noiseless, "quantizationmethod", "analog")
    noiseless = sensor_set(noiseless, "pixel read noise volts", 0.0)
    noiseless = sensor_set(noiseless, "pixel dark voltage", 0.0)
    return noiseless


def sensor_pd_array(sensor: Sensor, spacing: float) -> np.ndarray:
    """Measure photodetector coverage over the MATLAB-style ISA sampling grid."""

    spacing_value = float(spacing)
    if spacing_value > 1.0 or spacing_value <= 0.0:
        raise ValueError("The spacing parameter exceeds the limit. It must be between 0 and 1.")

    pixel = sensor_get(sensor, "pixel")
    pixel_size = np.asarray(pixel_get(pixel, "size"), dtype=float).reshape(-1)
    pd_position = np.asarray(pixel_get(pixel, "pd position"), dtype=float).reshape(-1)
    pd_size = np.asarray(pixel_get(pixel, "pd size"), dtype=float).reshape(-1)

    normalized_pd_min = pd_position / max(spacing_value, 1.0e-30) / pixel_size
    normalized_pd_max = (pd_size + pd_position) / max(spacing_value, 1.0e-30) / pixel_size

    grid_positions = np.arange(0.0, 1.0 + spacing_value, spacing_value, dtype=float) / max(spacing_value, 1.0e-30)
    n_squares = max(grid_positions.size - 1, 0)
    in_pd_rows = np.zeros(n_squares, dtype=float)
    in_pd_cols = np.zeros(n_squares, dtype=float)

    for index in range(n_squares):
        lower = max(grid_positions[index], normalized_pd_min[0])
        upper = min(grid_positions[index + 1], normalized_pd_max[0])
        in_pd_rows[index] = max(0.0, upper - lower)

        lower = max(grid_positions[index], normalized_pd_min[1])
        upper = min(grid_positions[index + 1], normalized_pd_max[1])
        in_pd_cols[index] = max(0.0, upper - lower)

    return in_pd_rows.reshape(-1, 1) @ in_pd_cols.reshape(1, -1)


def sensor_wb_compute(sensor: Sensor, work_dir: str | Path, display_flag: int | bool = 0) -> Sensor:
    """Replay MATLAB-style per-waveband OI files through the sensor pipeline."""

    del display_flag
    directory = Path(work_dir)
    oi_paths = sorted(directory.glob("oi*.mat"))
    if not oi_paths:
        raise ValueError("sensorWBCompute requires a directory containing oi*.mat files.")

    def _normalize_loaded_oi(optical_image: OpticalImage) -> OpticalImage:
        normalized = optical_image.clone()
        wave = np.asarray(normalized.fields["wave"], dtype=float).reshape(-1)
        photons = np.asarray(normalized.data["photons"], dtype=float)
        if photons.ndim == 2:
            if wave.size != 1:
                raise ValueError("Single-plane OI data must have exactly one wavelength sample.")
            photons = photons[:, :, np.newaxis]
        normalized.fields["wave"] = wave
        normalized.data["photons"] = photons
        return normalized

    noiseless_sensor = sensor_no_noise(sensor)
    accumulated_volts: np.ndarray | None = None
    last_oi: OpticalImage | None = None

    for oi_path in oi_paths[:-1]:
        optical_image, _ = vc_load_object("oi", oi_path)
        if not isinstance(optical_image, OpticalImage):
            raise ValueError(f"{oi_path} did not contain an optical image.")
        optical_image = _normalize_loaded_oi(optical_image)
        computed = sensor_compute(noiseless_sensor, optical_image, False)
        current_volts = np.asarray(sensor_get(computed, "volts"), dtype=float)
        accumulated_volts = current_volts if accumulated_volts is None else accumulated_volts + current_volts

    last_oi, _ = vc_load_object("oi", oi_paths[-1])
    if not isinstance(last_oi, OpticalImage):
        raise ValueError(f"{oi_paths[-1]} did not contain an optical image.")
    last_oi = _normalize_loaded_oi(last_oi)
    result = sensor_compute(sensor, last_oi, False)
    final_volts = np.asarray(sensor_get(result, "volts"), dtype=float)
    combined_volts = final_volts if accumulated_volts is None else accumulated_volts + final_volts
    result = sensor_set(result, "volts", combined_volts)

    if param_format(sensor_get(result, "quantizationmethod")) != "analog":
        digital_values, _ = analog_to_digital(result)
        result.data["dv"] = np.asarray(np.rint(digital_values), dtype=np.int32)

    result = sensor_set(result, "name", f"wb-{oi_get(last_oi, 'name')}")
    return result


def sensor_gain_offset(sensor: Sensor, ag: float, ao: float) -> Sensor:
    """Apply the legacy MATLAB analog gain/offset transform to sensor volts."""

    gain = float(ag)
    offset = float(ao)
    adjusted = sensor.clone()
    if np.isclose(gain, 1.0) and np.isclose(offset, 0.0):
        return adjusted
    if np.isclose(gain, 0.0):
        raise ValueError("sensorGainOffset requires a non-zero analog gain.")

    volts = sensor_get(adjusted, "volts")
    if volts is not None:
        adjusted = sensor_set(adjusted, "volts", (np.asarray(volts, dtype=float) + offset) / gain)
    adjusted = sensor_set(adjusted, "analog gain", gain)
    adjusted = sensor_set(adjusted, "analog offset", offset)
    return adjusted


def sensor_resample_wave(sensor: Sensor, new_wave_samples: Any) -> Sensor:
    """Resample sensor spectral data onto a new wavelength sampling."""

    resampled = sensor.clone()
    return sensor_set(resampled, "wavelengthSamples", np.asarray(new_wave_samples, dtype=float).reshape(-1))


def analog_to_digital(sensor: Sensor, method: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Quantize sensor volts using the legacy MATLAB analog2digital contract."""

    quantization_method = method if method is not None else str(sensor_get(sensor, "quantization method"))
    normalized_method = param_format(quantization_method)
    volts = sensor_get(sensor, "volts")
    if volts is None:
        raise ValueError("analog2digital requires sensor volts.")

    image = np.asarray(volts, dtype=float)
    if normalized_method == "analog":
        return image.copy(), np.zeros_like(image, dtype=float)

    if normalized_method in {"lin", "linear"} or normalized_method.endswith("bit"):
        n_bits = int(sensor_get(sensor, "nbits"))
        if n_bits <= 0:
            n_bits = 8
        quantization_step = float(sensor_get(sensor, "pixel voltage swing")) / float(2**n_bits)
        quantized = np.rint(image / max(quantization_step, 1e-30))
        quantization_error = image - (quantized * quantization_step)
        return np.asarray(quantized, dtype=float), np.asarray(quantization_error, dtype=float)

    raise UnsupportedOptionError("analog2digital", quantization_method)


def bin_noise_fpn(sensor: Sensor, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the legacy binning-path DSNU/PRNU helper to the current sensor image."""

    return noise_fpn(sensor, seed=seed)


def noise_fpn(sensor: Sensor, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply legacy DSNU/PRNU fixed-pattern noise to the current sensor volts."""

    volts = sensor_get(sensor, "volts")
    if volts is None:
        raise ValueError("noiseFPN requires sensor volts.")

    image = np.asarray(volts, dtype=float)
    rows, cols = image.shape[:2]
    rng = np.random.default_rng(None if seed is None else int(seed))

    stored_dsnu = sensor.fields.get("offset_fpn_image")
    if stored_dsnu is not None and np.asarray(stored_dsnu).shape == (rows, cols):
        dsnu_image = np.asarray(stored_dsnu, dtype=float).copy()
    else:
        dsnu_image = rng.normal(0.0, float(sensor_get(sensor, "dsnu level")), size=(rows, cols))

    stored_prnu = sensor.fields.get("gain_fpn_image")
    if stored_prnu is not None and np.asarray(stored_prnu).shape == (rows, cols):
        prnu_image = np.asarray(stored_prnu, dtype=float).copy()
    else:
        prnu_image = 1.0 + rng.normal(0.0, float(sensor_get(sensor, "prnu level")) / 100.0, size=(rows, cols))

    integration_time = np.asarray(sensor_get(sensor, "integration time"), dtype=float)
    auto_exposure = bool(sensor_get(sensor, "auto exposure"))
    if np.all(np.isclose(integration_time, 0.0)) and not auto_exposure:
        noisy_image = dsnu_image.copy()
    else:
        noisy_image = (image * _pixel_plane(image, prnu_image)) + _pixel_plane(image, dsnu_image)

    return np.asarray(noisy_image, dtype=float), dsnu_image, prnu_image


def bin_noise_column_fpn(sensor: Sensor, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the legacy binning-path column FPN helper to the current sensor image."""

    return noise_column_fpn(sensor, seed=seed)


def noise_column_fpn(sensor: Sensor, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply legacy column fixed-pattern noise to the current sensor volts."""

    volts = sensor_get(sensor, "volts")
    if volts is None:
        raise ValueError("noiseColumnFPN requires sensor volts.")

    image = np.asarray(volts, dtype=float)
    cols = int(sensor_get(sensor, "cols"))
    rng = np.random.default_rng(None if seed is None else int(seed))

    stored_offset = sensor.fields.get("column_offset_fpn")
    if stored_offset is not None and np.asarray(stored_offset).reshape(-1).size == cols:
        col_dsnu = np.asarray(stored_offset, dtype=float).reshape(-1).copy()
    else:
        offset_sigma = float(sensor_get(sensor, "column fpn offset"))
        col_dsnu = rng.normal(0.0, offset_sigma, size=cols) if not np.isclose(offset_sigma, 0.0) else np.zeros(cols, dtype=float)

    stored_gain = sensor.fields.get("column_gain_fpn")
    if stored_gain is not None and np.asarray(stored_gain).reshape(-1).size == cols:
        col_prnu = np.asarray(stored_gain, dtype=float).reshape(-1).copy()
    else:
        gain_sigma = float(sensor_get(sensor, "column fpn gain"))
        col_prnu = 1.0 + rng.normal(0.0, gain_sigma, size=cols) if not np.isclose(gain_sigma, 0.0) else np.ones(cols, dtype=float)

    if np.allclose(col_dsnu, 0.0) and np.allclose(col_prnu, 1.0):
        noisy_image = image.copy()
    else:
        dsnu_image = np.broadcast_to(col_dsnu.reshape(1, cols), image.shape[:2])
        prnu_image = np.broadcast_to(col_prnu.reshape(1, cols), image.shape[:2])
        noisy_image = (image * _pixel_plane(image, prnu_image)) + _pixel_plane(image, dsnu_image)

    return np.asarray(noisy_image, dtype=float), col_dsnu, col_prnu


def bin_noise_read(sensor: Sensor, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply legacy temporal read noise to the current binning-path sensor image."""

    source = sensor_get(sensor, "digital values")
    if source is None:
        source = sensor_get(sensor, "volts")
    if source is None:
        raise ValueError("binNoiseRead requires sensor digital values or volts.")

    image = np.asarray(source, dtype=float)
    sigma_read = float(pixel_get(sensor_get(sensor, "pixel"), "readNoiseVolts"))
    rng = np.random.default_rng(None if seed is None else int(seed))
    noise = rng.normal(0.0, sigma_read, size=image.shape)
    return np.asarray(image + noise, dtype=float), np.asarray(noise, dtype=float)


def sensor_compute_noise_free(
    sensor: Sensor,
    oi: OpticalImage,
    *,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Compute the mean sensor-voltage image without noise, clipping, or quantization."""

    working = sensor.clone()
    original_noise_flag = int(sensor_get(working, "noise flag"))
    original_quantization = str(sensor_get(working, "quantization method"))
    original_analog_gain = float(sensor_get(working, "analog gain"))
    original_analog_offset = float(sensor_get(working, "analog offset"))
    original_voltage_swing = float(sensor_get(working, "pixel voltage swing"))

    integration_time = np.asarray(sensor_get(working, "integration time"), dtype=float)
    if bool(sensor_get(working, "auto exposure")) or (
        integration_time.ndim == 0 and float(integration_time) <= 0.0
    ):
        exposure_time = _auto_exposure_default(working, oi)
        working = sensor_set(working, "integration time", exposure_time)

    working = sensor_set(working, "noise flag", 0)
    working = sensor_set(working, "quantization method", "analog")
    working = sensor_set(working, "analog gain", 1.0)
    working = sensor_set(working, "analog offset", 0.0)
    working.fields["pixel"] = dict(working.fields["pixel"])
    working.fields["pixel"]["voltage_swing"] = 1.0e6
    if _sensor_compute_method_is_binning(working.fields.get("sensor_compute_method")):
        working.fields["sensor_compute_method"] = None

    computed = sensor_compute(working, oi, seed=seed, session=session)
    computed.fields["pixel"] = dict(computed.fields["pixel"])
    computed.fields["pixel"]["voltage_swing"] = original_voltage_swing
    computed = sensor_set(computed, "noise flag", original_noise_flag)
    computed = sensor_set(computed, "quantization method", original_quantization)
    computed = sensor_set(computed, "analog gain", original_analog_gain)
    computed = sensor_set(computed, "analog offset", original_analog_offset)
    return computed


def sensor_add_noise(sensor: Sensor, *, seed: int | None = None) -> Sensor:
    """Add legacy photon/electrical/FPN noise to a precomputed mean-voltage sensor image."""

    updated = sensor.clone()
    base_volts = updated.data.get("volts")
    if base_volts is None:
        raise ValueError("sensorAddNoise requires sensor volts.")

    noise_flag = int(sensor_get(updated, "noise flag"))
    if noise_flag == 0:
        return updated
    if noise_flag not in {-2, 1, 2}:
        raise UnsupportedOptionError("sensorAddNoise", f"noise flag {noise_flag}")

    if seed is None:
        seed_value = int(updated.fields.get("noise_seed", 0))
    else:
        seed_value = int(seed)
    updated.fields["noise_seed"] = seed_value
    rng = np.random.default_rng(seed_value)

    pixel = dict(updated.fields["pixel"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])
    dark_voltage = float(pixel["dark_voltage_v_per_sec"])
    read_noise = float(pixel["read_noise_v"])

    integration_time = np.asarray(sensor_get(updated, "integration time"), dtype=float)
    if integration_time.ndim > 1:
        raise UnsupportedOptionError("sensorAddNoise", "matrix integration times")
    exposure_times = integration_time.reshape(-1)
    base = np.asarray(base_volts, dtype=float)
    if base.ndim == 2:
        captures = [base.copy()]
    elif base.ndim == 3:
        captures = [base[:, :, index].copy() for index in range(base.shape[2])]
    else:
        raise ValueError("sensorAddNoise expects 2-D or 3-D sensor volts.")

    dsnu_image = None if updated.fields.get("offset_fpn_image") is None else np.asarray(updated.fields["offset_fpn_image"], dtype=float).copy()
    prnu_image = None if updated.fields.get("gain_fpn_image") is None else np.asarray(updated.fields["gain_fpn_image"], dtype=float).copy()
    column_dsnu = None if updated.fields.get("column_offset_fpn") is None else np.asarray(updated.fields["column_offset_fpn"], dtype=float).reshape(-1).copy()
    column_prnu = None if updated.fields.get("column_gain_fpn") is None else np.asarray(updated.fields["column_gain_fpn"], dtype=float).reshape(-1).copy()

    noisy_captures: list[np.ndarray] = []
    for index, capture in enumerate(captures):
        exposure_time = float(exposure_times[index]) if exposure_times.size > 1 else float(exposure_times[0] if exposure_times.size == 1 else 0.0)
        volts = np.asarray(capture, dtype=float).copy()

        if noise_flag == 2:
            volts = volts + (dark_voltage * exposure_time)

        volts = _shot_noise_electrons(rng, volts / max(conversion_gain, 1e-12)) * conversion_gain

        if noise_flag == 2:
            volts = _apply_read_noise(rng, volts, read_noise)

        if noise_flag in {1, 2}:
            stage_sensor = updated.clone()
            stage_sensor.data["volts"] = volts
            if dsnu_image is not None:
                stage_sensor.fields["offset_fpn_image"] = dsnu_image.copy()
            if prnu_image is not None:
                stage_sensor.fields["gain_fpn_image"] = prnu_image.copy()
            volts, dsnu_image, prnu_image = noise_fpn(stage_sensor, seed=seed_value if dsnu_image is None or prnu_image is None else None)

            stage_sensor = updated.clone()
            stage_sensor.data["volts"] = volts
            if column_dsnu is not None:
                stage_sensor.fields["column_offset_fpn"] = column_dsnu.copy()
            if column_prnu is not None:
                stage_sensor.fields["column_gain_fpn"] = column_prnu.copy()
            volts, column_dsnu, column_prnu = noise_column_fpn(
                stage_sensor,
                seed=seed_value if column_dsnu is None or column_prnu is None else None,
            )

        noisy_captures.append(np.asarray(volts, dtype=float))

    if dsnu_image is not None:
        updated.fields["offset_fpn_image"] = dsnu_image.copy()
    if prnu_image is not None:
        updated.fields["gain_fpn_image"] = prnu_image.copy()
    if column_dsnu is not None:
        updated.fields["column_offset_fpn"] = column_dsnu.copy()
    if column_prnu is not None:
        updated.fields["column_gain_fpn"] = column_prnu.copy()

    updated.data["volts"] = noisy_captures[0] if len(noisy_captures) == 1 else np.stack(noisy_captures, axis=2)
    return updated


def sensor_compute_noise(
    sensor: Sensor,
    w_bar: Any | None = None,
    *,
    seed: int | None = None,
) -> Sensor:
    """Apply the legacy noise, gain/offset, clip, and quantization stage."""

    del w_bar

    updated = sensor_add_noise(sensor, seed=seed)
    analog_gain = float(sensor_get(updated, "analog gain"))
    analog_offset = float(sensor_get(updated, "analog offset"))
    voltage_swing = float(sensor_get(updated, "pixel voltage swing"))

    volts = np.asarray(sensor_get(updated, "volts"), dtype=float)
    volts = np.clip((volts + analog_offset) / max(analog_gain, 1e-12), 0.0, voltage_swing)
    updated = sensor_set(updated, "volts", volts)

    quantization_method = str(sensor_get(updated, "quantization method"))
    normalized_quantization = param_format(quantization_method)
    if normalized_quantization == "analog":
        updated.data["dv"] = None
    elif normalized_quantization in {"lin", "linear"} or normalized_quantization.endswith("bit"):
        quantized, _ = analog_to_digital(updated, "linear")
        updated = sensor_set(updated, "digital values", quantized)
    elif normalized_quantization in {"sqrt", "sqrtsq"}:
        quantized, _ = analog_to_digital(updated, "sqrt")
        updated = sensor_set(updated, "digital values", quantized)
    elif normalized_quantization in {"lut", "gamma"}:
        updated.data["dv"] = None
    else:
        quantized, _ = analog_to_digital(updated, "linear")
        updated = sensor_set(updated, "digital values", quantized)

    return updated


def sensor_compute_image(
    oi: OpticalImage,
    sensor: Sensor,
    w_bar: Any | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return the legacy MATLAB sensorComputeImage voltage image and FPN payloads."""

    del w_bar
    noise_free = sensor_compute_noise_free(sensor, oi, seed=seed)
    noisy = sensor_add_noise(noise_free, seed=seed)
    analog_gain = float(sensor_get(sensor, "analog gain"))
    analog_offset = float(sensor_get(sensor, "analog offset"))
    volt_image = (np.asarray(noisy.data["volts"], dtype=float) + analog_offset) / max(analog_gain, 1e-12)
    dsnu = noisy.fields.get("offset_fpn_image")
    prnu = noisy.fields.get("gain_fpn_image")
    return (
        np.asarray(volt_image, dtype=float),
        None if dsnu is None else np.asarray(dsnu, dtype=float).copy(),
        None if prnu is None else np.asarray(prnu, dtype=float).copy(),
    )


def bin_sensor_compute_image(
    oi: OpticalImage,
    sensor: Sensor,
    b_method: str | None = None,
    w_bar: Any | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return the legacy MATLAB binSensorComputeImage payloads."""

    del w_bar
    method = b_method or "kodak2008"
    integration_time = np.asarray(sensor_get(sensor, "integration time"), dtype=float).reshape(-1)
    if integration_time.size != 1:
        raise ValueError("Pixel binning only runs with a single integration time.")

    noise_flag = int(sensor_get(sensor, "noise flag"))
    noise_free = sensor_compute_noise_free(sensor, oi, seed=seed)
    volt_image = np.asarray(sensor_get(noise_free, "volts"), dtype=float).copy()
    binned_input = volt_image.copy()

    if noise_flag in {1, 2, -2}:
        if seed is None:
            seed_value = int(sensor.fields.get("noise_seed", 0))
        else:
            seed_value = int(seed)
        rng = np.random.default_rng(seed_value)
        pixel = sensor_get(sensor, "pixel")
        conversion_gain = float(pixel_get(pixel, "conversionGain"))
        if noise_flag == 2:
            binned_input = binned_input + (float(pixel_get(pixel, "darkVoltage")) * float(integration_time[0]))
        binned_input = _shot_noise_electrons(rng, binned_input / max(conversion_gain, 1e-12)) * conversion_gain

    stage = sensor_set(sensor.clone(), "volts", binned_input)
    stage = bin_pixel(stage, method)
    dv = np.asarray(sensor_get(stage, "dv"), dtype=float)
    dsnu = None
    prnu = None

    if noise_flag in {1, 2}:
        noisy_stage = sensor_set(sensor.clone(), "volts", dv)
        dv, dsnu, prnu = bin_noise_fpn(noisy_stage, seed=seed)
        noisy_stage = sensor_set(noisy_stage, "volts", dv)
        dv, _, _ = bin_noise_column_fpn(noisy_stage, seed=seed)

    if noise_flag == 2:
        read_stage = sensor_set(sensor.clone(), "digital values", dv)
        dv, _ = bin_noise_read(read_stage, seed=seed)

    return (
        np.asarray(dv, dtype=float),
        np.asarray(volt_image, dtype=float),
        None if dsnu is None else np.asarray(dsnu, dtype=float).copy(),
        None if prnu is None else np.asarray(prnu, dtype=float).copy(),
    )


def bin_sensor_compute(
    sensor: Sensor,
    optical_image: OpticalImage,
    b_method: str | None = None,
    show_wait_bar: Any = 1,
    *,
    seed: int | None = None,
) -> Sensor:
    """Compute the legacy MATLAB binSensorCompute sensor wrapper."""

    del show_wait_bar
    method = b_method or "kodak2008"
    current = sensor_clear_data(sensor)
    integration_time = np.asarray(sensor_get(current, "integration time"), dtype=float).reshape(-1)
    if integration_time.size != 1:
        raise ValueError("Pixel binning only runs with a single integration time.")
    if bool(sensor_get(current, "auto exposure")) or float(integration_time[0]) <= 0.0:
        exposure_sensor = current.clone()
        exposure_sensor.fields["sensor_compute_method"] = None
        exposure_time = _auto_exposure_default(exposure_sensor, optical_image)
        current = sensor_set(current, "integration time", exposure_time)

    dv_stage, volt_image, dsnu, prnu = bin_sensor_compute_image(
        optical_image,
        current,
        method,
        None,
        seed=seed,
    )

    dv_stage = np.asarray(dv_stage, dtype=float)
    if bool(sensor_get(current, "cds")):
        cds_sensor = sensor_set(current.clone(), "integration time", 0.0)
        cds_stage, _, _, _ = bin_sensor_compute_image(
            optical_image,
            cds_sensor,
            method,
            None,
            seed=seed,
        )
        dv_stage = np.clip(dv_stage - np.asarray(cds_stage, dtype=float), 0.0, None)

    analog_gain = float(sensor_get(current, "analog gain"))
    analog_offset = float(sensor_get(current, "analog offset"))
    voltage_swing = float(sensor_get(current, "pixel voltage swing"))
    dv_linear = np.clip((dv_stage + analog_offset) / max(analog_gain, 1e-12), 0.0, voltage_swing)

    quant_sensor = sensor_set(current.clone(), "volts", dv_linear)
    quantization_method = str(sensor_get(current, "quantization method"))
    if param_format(quantization_method) == "analog":
        quantized, _ = analog_to_digital(sensor_set(quant_sensor.clone(), "quantization method", "8 bit"), "linear")
    else:
        quantized, _ = analog_to_digital(quant_sensor, quantization_method)
    quant_sensor = sensor_set(quant_sensor, "digital values", quantized)
    quant_sensor = bin_pixel_post(quant_sensor, method)

    result = current.clone()
    result_vignetting = result.fields.get("vignetting", 0)
    normalized_vignetting = (
        param_format(result_vignetting)
        if isinstance(result_vignetting, str)
        else result_vignetting
    )
    if normalized_vignetting not in {0, "skip", "off", "none", "disabled", "false", "", None}:
        result.fields["etendue"] = _sensor_etendue(result).copy()
    method_payload = {"name": "binning", "method": method}
    original_method = current.fields.get("sensor_compute_method")
    if isinstance(original_method, dict):
        for key in ("factor", "domain", "stage"):
            if key in original_method:
                method_payload[key] = _copy_metadata_value(original_method[key])
    result.fields["sensor_compute_method"] = method_payload
    result.data["volts"] = np.asarray(dv_linear, dtype=float).copy()
    result.data["dv"] = np.asarray(sensor_get(quant_sensor, "dv"), dtype=float).copy()
    if dsnu is not None:
        result.fields["offset_fpn_image"] = np.asarray(dsnu, dtype=float).copy()
    if prnu is not None:
        result.fields["gain_fpn_image"] = np.asarray(prnu, dtype=float).copy()
    result.fields["pre_binning_volts"] = np.asarray(volt_image, dtype=float).copy()
    return result


def sensor_compute_full_array(
    sensor: Sensor,
    oi: OpticalImage,
    c_filters: Any | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-array volt and DV images for each supplied filter spectrum."""

    if c_filters is None:
        filter_spectra = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
    else:
        filter_spectra = np.asarray(c_filters, dtype=float)
    if filter_spectra.ndim == 1:
        filter_spectra = filter_spectra.reshape(-1, 1)

    rows, cols = map(int, sensor_get(sensor, "size"))
    num_channels = int(filter_spectra.shape[1])
    volts = np.zeros((rows, cols, num_channels), dtype=float)
    dvs = np.zeros((rows, cols, num_channels), dtype=float)

    for channel_index in range(num_channels):
        current = sensor.clone()
        current.fields["mosaic"] = False
        current.fields["pattern"] = np.array([[1]], dtype=int)
        current = sensor_set(current, "filter spectra", filter_spectra[:, channel_index].reshape(-1, 1))
        current = sensor_set(current, "filter names", [f"Channel-{channel_index + 1}"])
        current = sensor_compute(current, oi, seed=None if seed is None else int(seed) + channel_index)
        channel_volts = np.asarray(sensor_get(current, "volts"), dtype=float)
        if channel_volts.ndim == 3 and channel_volts.shape[2] == 1:
            channel_volts = channel_volts[:, :, 0]
        volts[:, :, channel_index] = channel_volts
        if param_format(sensor_get(current, "quantization method")) != "analog" and current.data.get("dv") is not None:
            channel_dv = np.asarray(sensor_get(current, "dv"), dtype=float)
            if channel_dv.ndim == 3 and channel_dv.shape[2] == 1:
                channel_dv = channel_dv[:, :, 0]
            dvs[:, :, channel_index] = channel_dv

    return volts, dvs


def sensor_show_image(
    sensor: Sensor,
    gam: float | None = None,
    scale_max: bool | None = None,
    app: Any | None = None,
) -> np.ndarray | None:
    """Return the headless sensor-window RGB rendering for the current sensor data."""

    del app
    return _sensor_rgb_image(sensor, "dv or volts", gam, scale_max)


def sensor_save_image(
    sensor: Sensor,
    full_name: str | Path,
    data_type: str = "volts",
    gam: float = 1.0,
    scale_max: bool = True,
) -> str:
    """Save the current sensor rendering to an 8-bit PNG file."""

    image = sensor_get(sensor, "rgb", data_type, gam, scale_max)
    if image is None:
        raise ValueError("Sensor has no computed image data to save.")

    rgb = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    max_value = float(np.max(rgb))
    if max_value > 0.0:
        rgb = rgb / max_value

    output_path = Path(full_name).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
    iio.imwrite(output_path, payload)
    return str(output_path)


def sensor_rescale(sensor: Sensor, rowcol: Any, sensor_height_width: Any) -> Sensor:
    """Rescale a sensor to a new row/column size and physical die dimensions."""

    if sensor_height_width is None:
        raise ValueError("sensorRescale requires sensor height/width.")

    target_size = np.asarray(rowcol, dtype=int).reshape(-1)
    if target_size.size != 2:
        raise ValueError("sensorRescale rowcol must contain [rows, cols].")
    sensor_size_m = np.asarray(sensor_height_width, dtype=float).reshape(-1)
    if sensor_size_m.size != 2:
        raise ValueError("sensorRescale sensorHeightWidth must contain [height, width].")

    rescaled = sensor.clone()
    pixel = sensor_get(rescaled, "pixel")
    pixel = pixel_set(pixel, "widthGap", 0.0)
    pixel = pixel_set(pixel, "heightGap", 0.0)
    pixel = pixel_set(pixel, "width", float(sensor_size_m[1]) / float(target_size[1]))
    pixel = pixel_set(pixel, "height", float(sensor_size_m[0]) / float(target_size[0]))
    pixel = pixel_set(pixel, "pd width", np.sqrt(0.5) * float(pixel_get(pixel, "width")))
    pixel = pixel_set(pixel, "pd height", np.sqrt(0.5) * float(pixel_get(pixel, "height")))
    pixel = pixel_position_pd(pixel, "center")
    rescaled = sensor_set(rescaled, "pixel", pixel)
    rescaled = sensor_set(rescaled, "size", target_size.astype(int))
    return sensor_clear_data(rescaled)


def sensor_cfa_save(sensor: Sensor, full_name: str | Path) -> str:
    """Save the MATLAB-style CFA/color/spectrum payload to a MAT file."""

    path = Path(full_name).expanduser()
    if path.suffix.lower() != ".mat":
        path = path.with_suffix(".mat")
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    color = dict(sensor_get(sensor, "color"))
    filter_names = list(color.get("filterNames", []))
    color["filterNames"] = [
        (name[:1].lower() + name[1:]) if isinstance(name, str) and name else name for name in filter_names
    ]
    cfa = sensor_get(sensor, "cfa")
    spectrum = sensor_get(sensor, "spectrum")
    savemat(path, {"cfa": cfa, "color": color, "spectrum": spectrum}, do_compression=True)
    return str(path)


def sensor_from_file(filename: str | Path) -> Sensor:
    """Load a saved MATLAB-style sensor struct from a MAT file."""

    loaded, _ = vc_load_object("sensor", filename)
    if not isinstance(loaded, Sensor):
        raise ValueError(f"sensorFromFile expected a sensor payload in {filename}.")
    return loaded


def load_raw_sensor_data(
    filename: str | Path,
    bpp: int = 8,
    byte_format: str = "little",
    row: int | None = None,
    col: int | None = None,
) -> np.ndarray:
    """Load raw sensor bytes using the legacy MATLAB LoadRawSensorData contract."""

    del row, col
    path = Path(filename).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Cannot open file {path}")

    bits_per_pixel = int(bpp)
    normalized_format = param_format(byte_format or "little")
    if normalized_format not in {"little", "big"}:
        raise ValueError("byteFormat must be 'little' or 'big'.")

    if bits_per_pixel == 8:
        return np.fromfile(path, dtype=np.uint8)
    if bits_per_pixel == 10:
        dtype = np.dtype("<u2") if normalized_format == "little" else np.dtype(">u2")
        return np.asarray(np.fromfile(path, dtype=dtype), dtype=np.uint16)
    raise ValueError(f"Bad bpp {bits_per_pixel}. Must be 8 or 10 bits per pixel.")


def sensor_show_cfa(
    sensor: Sensor,
    app: Any | None = None,
    sz: Any | None = None,
    s_scale: int | None = None,
) -> tuple[None, np.ndarray]:
    """Return the headless CFA-render image for the requested unit-block tiling."""

    del app
    render_sensor = sensor.clone()
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    if sz is not None and np.asarray(sz).size != 0:
        sz_array = np.asarray(sz, dtype=int).reshape(-1)
        if sz_array.size != 2:
            raise ValueError("sensorShowCFA sz must contain [rows, cols].")
        pattern = np.tile(pattern, (int(sz_array[0]), int(sz_array[1])))
    render_sensor.fields["pattern"] = pattern.copy()
    render_sensor.fields["size"] = tuple(int(value) for value in pattern.shape)
    render_sensor.data.clear()
    render_sensor.data["volts"] = np.full(pattern.shape, float(sensor_get(sensor, "voltage swing")), dtype=float)
    cfa_small = np.asarray(sensor_get(render_sensor, "rgb", "volts", 1.0, False), dtype=float)
    if s_scale is None:
        scale = 32 if cfa_small.shape[0] < 2 else 8 if cfa_small.shape[0] < 8 else 1
    else:
        scale = int(s_scale)
    cfa_img = image_increase_image_rgb_size(cfa_small, [scale, scale]) if scale > 1 else cfa_small.copy()
    return None, np.asarray(cfa_img, dtype=float)


def sensor_color_order(format: str = "cell") -> tuple[list[str] | str, np.ndarray]:
    """Return the legacy ISET CFA color-hint ordering and its RGB colormap."""

    normalized = param_format(format)
    ordering = list(_SENSOR_COLOR_ORDER)
    if normalized == "string":
        return "".join(ordering), _SENSOR_COLOR_MAP.copy()
    return ordering, _SENSOR_COLOR_MAP.copy()


def sensor_cfa_name_list() -> list[str]:
    """Return the legacy ISET popup list of built-in CFA names."""

    return ["Bayer RGB", "Bayer CMY", "RGBW", "Monochrome", "Other"]


def sensor_pixel_coord(
    sensor: Sensor,
    quadrant_type: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor pixel-center coordinates for the requested array view."""

    n_rows = int(sensor_get(sensor, "rows"))
    n_cols = int(sensor_get(sensor, "cols"))
    pitch_x = float(sensor_get(sensor, "deltax"))
    pitch_y = float(sensor_get(sensor, "deltay"))
    normalized = param_format(quadrant_type)

    def _upper_right(count: int, pitch: float) -> np.ndarray:
        if count % 2 == 0:
            return pitch * np.arange(count // 2, dtype=float) + pitch / 2.0
        return pitch * np.arange(count // 2 + 1, dtype=float)

    def _full(count: int, pitch: float) -> np.ndarray:
        positive = _upper_right(count, pitch)
        if count % 2 == 0:
            return np.concatenate((-np.flip(positive), positive))
        return np.concatenate((-np.flip(positive), positive[1:]))

    if normalized in {"upperright", "upper-right"}:
        return _upper_right(n_cols, pitch_x), _upper_right(n_rows, pitch_y)
    if normalized == "full":
        return _full(n_cols, pitch_x), _full(n_rows, pitch_y)
    raise UnsupportedOptionError("sensorPixelCoord", quadrant_type)


def sensor_determine_cfa(sensor: Sensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tiled CFA letters, CFA numbers, and the per-filter RGB map."""

    rows = int(sensor_get(sensor, "rows"))
    cols = int(sensor_get(sensor, "cols"))
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    cfa_numbers = tile_pattern(pattern, rows, cols)

    pattern_colors = np.asarray(sensor_get(sensor, "pattern colors"), dtype="<U1")
    row_tiles = int(np.ceil(rows / max(pattern_colors.shape[0], 1)))
    col_tiles = int(np.ceil(cols / max(pattern_colors.shape[1], 1)))
    cfa_letters = np.tile(pattern_colors, (row_tiles, col_tiles))[:rows, :cols]

    ordering_string, ordering_map = sensor_color_order("string")
    lookup = {letter: ordering_map[index] for index, letter in enumerate(ordering_string)}
    fallback = lookup["k"]
    filter_letters = str(sensor_get(sensor, "filter color letters"))
    if filter_letters:
        mp = np.vstack([lookup.get(letter.lower(), fallback) for letter in filter_letters])
    else:
        mp = np.zeros((0, 3), dtype=float)

    return cfa_letters, cfa_numbers, np.asarray(mp, dtype=float)


def sensor_image_color_array(cfa: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CFA letter array into MATLAB-style color-order indices and colormap."""

    if cfa is None:
        raise ValueError("cfa (letter array) required.")
    cfa_array = np.asarray(cfa)
    if cfa_array.ndim == 0:
        cfa_letters = np.array([[str(cfa_array.item())]], dtype="<U1")
    elif cfa_array.ndim == 1 and cfa_array.dtype.kind in {"U", "S", "O"} and all(
        isinstance(item, str) for item in cfa_array.tolist()
    ):
        cfa_letters = np.array([list(str(item)) for item in cfa_array.tolist()], dtype="<U1")
    else:
        cfa_letters = np.asarray(cfa_array, dtype="<U1")

    cfa_numbers = np.zeros(cfa_letters.shape, dtype=int)
    lowered = np.char.lower(cfa_letters)
    for index, letter in enumerate(_SENSOR_COLOR_ORDER, start=1):
        cfa_numbers[lowered == letter] = index
    return cfa_numbers, _SENSOR_COLOR_MAP.copy()


def sensor_rgb_to_plane(rgb_data: Any, cfa_pattern: Any) -> tuple[np.ndarray, Sensor]:
    """Convert multiband CFA-aligned data into a legacy MATLAB sensor plane."""

    if rgb_data is None:
        raise ValueError("rgb data required")
    if cfa_pattern is None:
        raise ValueError("cfaPattern required")

    rgb = np.asarray(rgb_data, dtype=float)
    if rgb.ndim != 3:
        raise ValueError("rgb data must be an (r, c, w) array.")

    pattern = np.asarray(cfa_pattern, dtype=int)
    if pattern.ndim != 2 or pattern.size == 0:
        raise ValueError("cfaPattern must be a non-empty 2D array.")

    n_bands = int(rgb.shape[2])
    if int(np.max(pattern)) > n_bands:
        raise ValueError("bad cfa pattern")

    block_rows, block_cols = pattern.shape
    rows = block_rows * (int(rgb.shape[0]) // block_rows)
    cols = block_cols * (int(rgb.shape[1]) // block_cols)
    rgb = rgb[:rows, :cols, :]

    sensor = sensor_create()
    filter_names, _ = sensor_color_order()
    sensor = sensor_set(sensor, "pattern", pattern.copy())
    sensor = sensor_set(sensor, "size", [rows, cols])
    sensor = sensor_set(sensor, "filter spectra", np.ones((int(sensor_get(sensor, "nwave")), n_bands), dtype=float))
    sensor = sensor_set(sensor, "filter names", list(filter_names[:n_bands]))

    tiled = tile_pattern(pattern, rows, cols)
    sensor_plane = np.zeros((rows, cols), dtype=float)
    for band_index in range(n_bands):
        selector = tiled == (band_index + 1)
        sensor_plane[selector] = rgb[:, :, band_index][selector]

    return sensor_plane, sensor


def sensor_check_array(sensor: Sensor, n: int = 64) -> np.ndarray:
    """Return a headless CFA visibility image following MATLAB sensorCheckArray()."""

    _, image = sensor_show_cfa(sensor, None, None, int(n))
    return np.asarray(image, dtype=float)


def sensor_stats(
    sensor: Sensor,
    stat_type: str = "basic",
    unit_type: str = "volts",
    roi: Any | None = None,
    quiet: bool = False,
) -> tuple[Any, Sensor]:
    """Return MATLAB-style ROI summary statistics for volts, electrons, or DV data."""

    del quiet
    updated = sensor.clone()
    if roi is not None and np.asarray(roi).size != 0:
        updated = sensor_set(updated, "roi", roi)
    elif sensor_get(updated, "roi") is None:
        rows = int(sensor_get(updated, "rows"))
        cols = int(sensor_get(updated, "cols"))
        updated = sensor_set(updated, "roi", np.array([1, 1, cols - 1, rows - 1], dtype=int))

    normalized_unit = param_format(unit_type or "volts")
    if normalized_unit == "volts":
        data = sensor_get(updated, "roi volts")
    elif normalized_unit == "electrons":
        data = sensor_get(updated, "roi electrons")
    elif normalized_unit in {"dv", "digitalcount"}:
        data = sensor_get(updated, "roi dv")
    else:
        raise ValueError("Unknown unit type")

    if data is None:
        raise ValueError("sensorStats requires ROI data.")

    array = np.asarray(data, dtype=float)
    if array.ndim <= 1:
        columns = [array.reshape(-1)]
    else:
        columns = [array[:, index].reshape(-1) for index in range(array.shape[1])]

    def _valid_stats(values: np.ndarray) -> tuple[float, float, float, int]:
        usable = values[np.isfinite(values)]
        count = int(usable.size)
        if count == 0:
            return float("nan"), float("nan"), float("nan"), 0
        mean = float(np.mean(usable, dtype=float))
        std = float(np.std(usable, ddof=1)) if count > 1 else 0.0
        sem = float(std / np.sqrt(max(count - 1, 1)))
        return mean, std, sem, count

    normalized_stat = param_format(stat_type or "basic")
    if normalized_stat == "mean":
        means = np.array([_valid_stats(column)[0] for column in columns], dtype=float)
        return (float(means[0]) if means.size == 1 else means), updated

    if normalized_stat == "basic":
        means = []
        stds = []
        sems = []
        counts = []
        for column in columns:
            mean, std, sem, count = _valid_stats(column)
            means.append(mean)
            stds.append(std)
            sems.append(sem)
            counts.append(count)
        if len(columns) == 1:
            return (
                {
                    "mean": float(means[0]),
                    "std": float(stds[0]),
                    "sem": float(sems[0]),
                    "N": int(counts[0]),
                },
                updated,
            )
        return (
            {
                "mean": np.asarray(means, dtype=float),
                "std": np.asarray(stds, dtype=float),
                "sem": np.asarray(sems, dtype=float),
                "N": int(counts[0]),
            },
            updated,
        )

    raise ValueError("Unknown statistic type.")


def _sensor_filter_selection(
    data: np.ndarray,
    names: list[str],
    which_column: int | None,
) -> tuple[np.ndarray, str]:
    spectra = np.asarray(data, dtype=float)
    if spectra.ndim == 1:
        spectra = spectra.reshape(-1, 1)
    if spectra.shape[1] == 0:
        raise ValueError("Filter spectra must contain at least one column.")
    column_index = 0 if which_column is None else int(which_column) - 1
    if column_index < 0 or column_index >= spectra.shape[1]:
        raise ValueError("Requested filter column is out of range.")
    resolved_names = list(names) if names else [f"f{index + 1}" for index in range(spectra.shape[1])]
    return spectra[:, column_index].copy(), str(resolved_names[column_index])


def sensor_read_color_filters(
    sensor: Sensor,
    filter_file: str,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return MATLAB-style filter spectra matched to the sensor wavelength sampling."""

    store = _store(asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    normalized = param_format(filter_file)

    if normalized in {"rgb", "monochrome", "cym", "grbc", "xyz"}:
        _, filter_spectra, filter_names = store.load_color_filters(normalized, wave_nm=wave)
        return np.asarray(filter_spectra, dtype=float), list(filter_names)

    if normalized == "stockmanabs":
        filter_spectra, filter_names, _ = ie_read_color_filter(wave, "data/human/stockman.mat", asset_store=store)
        return np.asarray(filter_spectra, dtype=float), list(filter_names)

    filter_spectra, filter_names, _ = ie_read_color_filter(wave, filter_file, asset_store=store)
    return np.asarray(filter_spectra, dtype=float), list(filter_names)


def sensor_read_filter(
    filter_type: str,
    sensor: Sensor,
    fname: str,
    *,
    asset_store: AssetStore | None = None,
) -> Sensor:
    """Read MATLAB-style sensor filter payloads into the requested slot."""

    updated = sensor.clone()
    normalized_type = param_format(filter_type or "cfa")
    if normalized_type in {"cfa", "colorfilters", "colorfilter"}:
        filter_spectra, filter_names = sensor_read_color_filters(updated, fname, asset_store=asset_store)
        pattern = np.minimum(np.asarray(sensor_get(updated, "pattern"), dtype=int), int(filter_spectra.shape[1]))
        updated = sensor_set(updated, "filter spectra", filter_spectra)
        updated = sensor_set(updated, "filter names", filter_names)
        updated = sensor_set(updated, "pattern", pattern)
        return updated

    wave = np.asarray(sensor_get(updated, "wave"), dtype=float).reshape(-1)
    filter_spectra, _, _ = ie_read_color_filter(wave, fname, asset_store=_store(asset_store))
    if normalized_type == "pdspectralqe":
        return sensor_set(updated, "pixel pd spectral qe", np.asarray(filter_spectra, dtype=float).reshape(-1))
    if normalized_type in {"infrared", "irfilter"}:
        return sensor_set(updated, "ir filter", np.asarray(filter_spectra, dtype=float).reshape(-1))
    raise UnsupportedOptionError("sensorReadFilter", filter_type)


def sensor_add_filter(
    sensor: Sensor,
    fname: str,
    *,
    which_column: int | None = None,
    asset_store: AssetStore | None = None,
) -> Sensor:
    """Add a MATLAB-style color filter column to the sensor."""

    updated = sensor.clone()
    filter_spectra = np.asarray(sensor_get(updated, "filter spectra"), dtype=float)
    filter_names = list(sensor_get(updated, "filter names"))
    loaded_data, loaded_names = sensor_read_color_filters(updated, fname, asset_store=asset_store)
    new_column, new_name = _sensor_filter_selection(loaded_data, loaded_names, which_column)
    updated = sensor_set(updated, "filter spectra", np.column_stack([filter_spectra, new_column]))
    updated = sensor_set(updated, "filter names", [*filter_names, new_name])
    return updated


def sensor_replace_filter(
    sensor: Sensor,
    which_filter: int,
    new_filter_file: str,
    *,
    which_column: int | None = None,
    new_filter_name: str | None = None,
    asset_store: AssetStore | None = None,
) -> Sensor:
    """Replace a MATLAB-style sensor filter without changing the CFA pattern."""

    updated = sensor.clone()
    filter_index = int(which_filter) - 1
    filter_spectra = np.asarray(sensor_get(updated, "filter spectra"), dtype=float)
    if filter_index < 0 or filter_index >= filter_spectra.shape[1]:
        raise ValueError("Requested filter index is out of range.")
    filter_names = list(sensor_get(updated, "filter names"))
    loaded_data, loaded_names = sensor_read_color_filters(updated, new_filter_file, asset_store=asset_store)
    new_column, inferred_name = _sensor_filter_selection(loaded_data, loaded_names, which_column)
    filter_spectra[:, filter_index] = new_column
    filter_names[filter_index] = str(new_filter_name) if new_filter_name is not None else inferred_name
    updated = sensor_set(updated, "filter spectra", filter_spectra)
    updated = sensor_set(updated, "filter names", filter_names)
    return updated


def sensor_delete_filter(sensor: Sensor, which_filter: int) -> Sensor:
    """Delete a MATLAB-style color filter and shift the CFA pattern down."""

    updated = sensor.clone()
    filter_index = int(which_filter) - 1
    filter_spectra = np.asarray(sensor_get(updated, "filter spectra"), dtype=float)
    if filter_index < 0 or filter_index >= filter_spectra.shape[1]:
        raise ValueError("Requested filter index is out of range.")
    keep_list = np.ones(filter_spectra.shape[1], dtype=bool)
    keep_list[filter_index] = False
    updated = sensor_set(updated, "filter spectra", filter_spectra[:, keep_list])
    updated = sensor_set(updated, "filter names", [name for index, name in enumerate(sensor_get(sensor, "filter names")) if keep_list[index]])
    pattern = np.asarray(sensor_get(updated, "pattern"), dtype=int)
    threshold = int(which_filter)
    pattern = pattern.copy()
    mask = pattern >= threshold
    pattern[mask] = np.maximum(1, pattern[mask] - 1)
    updated = sensor_set(updated, "pattern", pattern)
    return updated


def sensor_show_cfa_weights(
    wgts: Any,
    sensor: Sensor,
    c_pos: Any | None = None,
    *args: Any,
) -> np.ndarray:
    """Return a weighted CFA-color image on a small local CFA region."""

    weights = np.asarray(wgts, dtype=float)
    if weights.ndim != 2:
        raise ValueError("sensorShowCFAWeights expects a 2D weight matrix.")

    if c_pos is None:
        c_pos_array = np.ceil(np.asarray(weights.shape, dtype=float) / 2.0).astype(int)
    else:
        c_pos_array = np.rint(np.asarray(c_pos, dtype=float).reshape(-1)).astype(int)
    if c_pos_array.size != 2:
        raise ValueError("sensorShowCFAWeights cPos must contain [row, col].")

    img_scale = 32
    if args:
        if len(args) == 1 and not isinstance(args[0], str):
            img_scale = int(np.rint(float(args[0])))
        else:
            settings = _matlab_kv_pairs(args, function_name="sensorShowCFAWeights")
            for key, value in settings:
                if param_format(key) == "imgscale":
                    img_scale = int(np.rint(float(value)))
                else:
                    raise UnsupportedOptionError("sensorShowCFAWeights", key)

    unit_letters = np.asarray(sensor_get(sensor, "pattern colors"), dtype="<U1")
    block_rows, block_cols = unit_letters.shape
    center_row = int(c_pos_array[0]) - 1
    center_col = int(c_pos_array[1]) - 1
    row_offsets = np.arange(weights.shape[0], dtype=int) - (weights.shape[0] // 2)
    col_offsets = np.arange(weights.shape[1], dtype=int) - (weights.shape[1] // 2)
    row_indices = np.mod(center_row + row_offsets, block_rows)
    col_indices = np.mod(center_col + col_offsets, block_cols)
    cfa_letters = unit_letters[np.ix_(row_indices, col_indices)]

    cfa_numbers, cfa_map = sensor_image_color_array(cfa_letters)
    cfa_rgb = np.zeros(cfa_letters.shape + (3,), dtype=float)
    valid = cfa_numbers > 0
    if np.any(valid):
        cfa_rgb[valid] = cfa_map[cfa_numbers[valid] - 1]

    if np.allclose(np.max(weights), np.min(weights)):
        normalized_weights = np.ones_like(weights, dtype=float)
    else:
        min_weight = float(np.min(weights))
        max_weight = float(np.max(weights))
        normalized_weights = (weights - min_weight) / max(max_weight - min_weight, 1e-12)

    weighted = np.repeat(normalized_weights[:, :, np.newaxis], 3, axis=2) * cfa_rgb
    return image_increase_image_rgb_size(weighted, img_scale)


def _sensor_chromaticity(sensor: Sensor, rect_or_locs: Any = None, mode: str = "vec") -> np.ndarray | None:
    from .ip import _sensor_space

    volts = sensor.data.get("volts")
    if volts is None:
        return None

    rect = _sensor_rect_for_chromaticity(rect_or_locs)
    cropped = sensor.clone()
    cropped_volts = _sensor_crop_rect(volts, rect)
    if cropped_volts is None:
        return None
    cropped.data["volts"] = cropped_volts
    cropped_dv = _sensor_crop_rect(sensor.data.get("dv"), rect)
    if cropped_dv is not None:
        cropped.data["dv"] = cropped_dv
    elif "dv" in cropped.data:
        cropped.data.pop("dv", None)
    cropped.fields["size"] = tuple(int(value) for value in cropped_volts.shape[:2])

    sensor_space = np.asarray(_sensor_space(cropped), dtype=float)
    if sensor_space.ndim == 2:
        sensor_space = sensor_space[:, :, np.newaxis]
    nchannels = int(sensor_space.shape[2])
    output_shape = sensor_space.shape[:2] + (max(nchannels - 1, 0),)
    denominator = np.sum(sensor_space, axis=2, keepdims=True)
    chromaticity = np.divide(
        sensor_space[:, :, : max(nchannels - 1, 0)],
        denominator,
        out=np.full(output_shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )

    normalized_mode = param_format(mode)
    if normalized_mode == "matrix":
        return chromaticity
    return chromaticity.reshape(-1, output_shape[2])


def sensor_get(sensor: Sensor, parameter: str, *args: Any) -> Any:
    object_type, object_param = ie_parameter_otype(parameter)
    if object_type == "pixel":
        try:
            if object_param is None:
                return sensor.fields["pixel"]
            return _sensor_pixel_get(sensor, object_param, *args)
        except KeyError:
            pass
    key = param_format(parameter)
    if key == "type":
        return sensor.type
    if key == "name":
        return sensor.name
    if key == "human":
        return _sensor_human(sensor)
    if key in {"humanconetype", "conetype"}:
        human = sensor.fields.get("human", {})
        value = human.get("coneType")
        return None if value is None else np.asarray(value).copy()
    if key in {"humanconedensities", "densities"}:
        human = sensor.fields.get("human", {})
        value = human.get("densities")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"humanconelocs", "conexy", "conelocs", "xy"}:
        human = sensor.fields.get("human", {})
        value = human.get("xy")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"humanrseed", "humanconeseed", "rseed"}:
        human = sensor.fields.get("human", {})
        return _copy_metadata_value(human.get("rSeed"))
    if key in {"sensormovement", "eyemovement"}:
        return _sensor_movement(sensor)
    if key in {"movementpositions", "sensorpositions"}:
        movement = sensor.fields.get("movement", {})
        value = movement.get("pos")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key == "sensorpositionsx":
        positions = sensor_get(sensor, "sensor positions")
        return None if positions is None else np.asarray(positions, dtype=float)[:, 0].copy()
    if key == "sensorpositionsy":
        positions = sensor_get(sensor, "sensor positions")
        return None if positions is None else np.asarray(positions, dtype=float)[:, 1].copy()
    if key in {"framesperposition", "framesperpositions", "exposuretimesperposition", "etimeperpos"}:
        movement = sensor.fields.get("movement", {})
        return _copy_metadata_value(movement.get("framesPerPosition"))
    if key in {"wave", "wavelength", "wavelengthsamples"}:
        return np.asarray(sensor.fields["wave"], dtype=float)
    if key in {"chartparameters"}:
        return _sensor_chart_parameters(sensor)
    if key in {"cornerpoints", "chartcornerpoints", "chartcorners"}:
        chart = sensor.fields.get("chartP", {})
        value = chart.get("cornerPoints")
        return None if value is None else np.asarray(value).copy()
    if key == "mcccornerpoints":
        return sensor_get(sensor, "chart corner points")
    if key in {"chartrects", "chartrectangles"}:
        chart = sensor.fields.get("chartP", {})
        value = chart.get("rects")
        return None if value is None else np.asarray(value).copy()
    if key in {"currentrect", "chartcurrentrect"}:
        chart = sensor.fields.get("chartP", {})
        value = chart.get("currentRect")
        return None if value is None else np.asarray(value).copy()
    if key == "mccrecthandles":
        return _copy_metadata_value(sensor.fields.get("mccRectHandles"))
    if key in {"spectrum", "sensorspectrum"}:
        return _sensor_spectrum_struct(sensor)
    if key == "color":
        return _sensor_color_struct(sensor)
    if key in {"binwidth", "waveresolution", "wavelengthresolution"}:
        wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
        return float(wave[1] - wave[0]) if wave.size > 1 else 1.0
    if key in {"nwave", "nwaves", "numberofwavelengthsamples"}:
        return int(np.asarray(sensor.fields["wave"]).size)
    if key == "pattern":
        return np.asarray(sensor.fields["pattern"], dtype=int)
    if key in {"filterspectra", "colorfilters", "filtertransmissivities"}:
        return np.asarray(sensor.fields["filter_spectra"], dtype=float)
    if key in {"spectralqe", "sensorqe", "sensorspectralqe", "qe"}:
        return _sensor_combined_qe(sensor)
    if key in {"sensorspectralsr"}:
        return _sensor_combined_sr(sensor)
    if key in {"pixelspectralqe", "pdspectralqe", "pixelqe"}:
        return _sensor_pixel_qe(sensor)
    if key in {"spectralsr", "pdspectralsr", "pixelspectralsr", "sr"}:
        return _pixel_spectral_sr(sensor)
    if key in {"ir", "infraredfilter", "irfilter", "otherfilter"}:
        return _sensor_ir_filter(sensor)
    if key in {"consistency", "sensorconsistency"}:
        return bool(sensor.fields.get("consistency", False))
    if key in {"sensorcompute", "sensorcomputemethod"}:
        return _copy_metadata_value(sensor.fields.get("sensor_compute_method"))
    if key in {"fdtdsensor", "fdtdlut", "fdtdopticalresponse", "fdtdopticalresponselut"}:
        return _copy_metadata_value(sensor.fields.get("fdtd_sensor"))
    if key in {"tcadsensor", "tcadlut", "tcadcollection", "tcadcollectionlut", "tcadcollectionresponse"}:
        return _copy_metadata_value(sensor.fields.get("tcad_sensor"))
    if key in {"filternames", "filtername"}:
        return list(sensor.fields["filter_names"])
    if key in {"nfilters", "nfilter", "ncolors", "ncolor", "nsensors", "nsensor"}:
        return int(np.asarray(sensor.fields["filter_spectra"]).shape[1])
    if key == "size":
        return _sensor_rows_cols(sensor)
    if key in {"rows", "row"}:
        return _sensor_rows_cols(sensor)[0]
    if key in {"cols", "col"}:
        return _sensor_rows_cols(sensor)[1]
    if key in {"height", "arrayheight"}:
        height = float(sensor_get(sensor, "rows")) * float(sensor_get(sensor, "deltay"))
        return height * _spatial_unit_scale(args[0] if args else None)
    if key in {"width", "arraywidth"}:
        width = float(sensor_get(sensor, "cols")) * float(sensor_get(sensor, "deltax"))
        return width * _spatial_unit_scale(args[0] if args else None)
    if key == "dimension":
        dimension = np.array([sensor_get(sensor, "height"), sensor_get(sensor, "width")], dtype=float)
        return dimension * _spatial_unit_scale(args[0] if args else None)
    if key in {"pixelfields", "pixel"}:
        return sensor.fields["pixel"]
    if key in {"pixelsize", "pixelsizesamefillfactor"}:
        return np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    if key in {"wspatialresolution", "wres", "deltax", "widthspatialresolution"}:
        return float(_sensor_spatial_resolution_m(sensor)[1]) * _spatial_unit_scale(args[0] if args else None)
    if key in {"hspatialresolution", "hres", "deltay", "heightspatialresolultion"}:
        return float(_sensor_spatial_resolution_m(sensor)[0]) * _spatial_unit_scale(args[0] if args else None)
    if key in {"spatialsupport", "xyvaluesinmeters"}:
        rows, cols = _sensor_rows_cols(sensor)
        deltay = float(sensor_get(sensor, "deltay"))
        deltax = float(sensor_get(sensor, "deltax"))
        support = {
            "y": _sample_centers(rows, deltay),
            "x": _sample_centers(cols, deltax),
        }
        scale = _spatial_unit_scale(args[0] if args else None)
        if not np.isclose(scale, 1.0):
            support = {axis: values * scale for axis, values in support.items()}
        return support
    if key in {"chiefrayangle", "cra", "chiefrayangleradians", "craradians", "craradian", "chiefrayangleradian"}:
        support = sensor_get(sensor, "spatial support")
        x, y = np.meshgrid(np.asarray(support["x"], dtype=float), np.asarray(support["y"], dtype=float))
        if args:
            source_focal_length = float(args[0])
        else:
            microlens = _sensor_microlens(sensor)
            source_focal_length = (
                float(microlens.get("sourceFocalLength", DEFAULT_FOCAL_LENGTH_M))
                if isinstance(microlens, dict)
                else DEFAULT_FOCAL_LENGTH_M
            )
        return np.arctan(np.sqrt(np.square(x) + np.square(y)) / max(source_focal_length, 1e-12))
    if key in {"chiefrayangledegrees", "cradegrees", "cradegree", "chiefrayangledegree"}:
        if args:
            return np.rad2deg(np.asarray(sensor_get(sensor, "cra", args[0]), dtype=float))
        return np.rad2deg(np.asarray(sensor_get(sensor, "cra"), dtype=float))
    if key == "filtercolorletters":
        return _sensor_filter_color_letters(sensor)
    if key == "filtercolorletterscell":
        return list(_sensor_filter_color_letters(sensor))
    if key in {"filternamescellarray", "filtercolornamescellarray", "filternamescell"}:
        return [str(letter) for letter in _sensor_filter_color_letters(sensor)]
    if key in {"cfa", "colorfilterarray"}:
        return _sensor_cfa_struct(sensor)
    if key in {"cfapattern", "pattern"}:
        return _sensor_cfa_pattern(sensor).copy()
    if key == "cfaname":
        return _sensor_cfa_name(sensor)
    if key in {"oclgroupproxy", "oclgroup", "sharedoclaperture"}:
        return _sensor_ocl_group_proxy(sensor)
    if key in {"oclgroupshape", "sharedoclshape"}:
        return _sensor_ocl_group_proxy(sensor)["shape"]
    if key in {"oclgroupequalization", "oclsharingequalization"}:
        return float(_sensor_ocl_group_proxy(sensor)["equalization"])
    if key == "diffusionmtf":
        value = sensor.fields.get("diffusion_mtf")
        return None if value is None else copy.deepcopy(value)
    if key in {"filterplotcolor", "filterplotcolors"}:
        colors = "".join(letter if letter in "rgbcmyk" else "k" for letter in _sensor_filter_color_letters(sensor))
        if args:
            index = int(np.rint(float(args[0]))) - 1
            return colors[index]
        return colors
    if key == "unitblockrows":
        return _sensor_unit_block(sensor)[0]
    if key == "unitblockcols":
        return _sensor_unit_block(sensor)[1]
    if key in {"cfasize", "unitblocksize"}:
        return _sensor_unit_block(sensor)
    if key == "unitblockconfig":
        return _sensor_unit_block_config(sensor)
    if key in {"patterncolors", "pcolors", "blockcolors"}:
        letters = np.array(list(_sensor_filter_color_letters(sensor)), dtype="<U1")
        if letters.size == 0:
            return np.empty(np.asarray(sensor.fields["pattern"], dtype=int).shape, dtype="<U1")
        known = np.array(list("rgbcmykw"), dtype="<U1")
        unknown = ~np.isin(letters, known)
        letters[unknown] = "k"
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        return letters[np.clip(pattern - 1, 0, letters.size - 1)]
    if key in {
        "integrationtime",
        "integrationtimes",
        "exptime",
        "exptimes",
        "exposuretimes",
        "exposuretime",
        "expduration",
        "exposureduration",
        "exposuredurations",
    }:
        return _sensor_integration_time_value(sensor, args[0] if args else None)
    if key in {"uniqueintegrationtimes", "uniqueexptime", "uniqueexptimes"}:
        return np.unique(np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float).reshape(-1))
    if key in {"centralexposure", "geometricmeanexposuretime"}:
        exposure_times = np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float).reshape(-1)
        return float(np.prod(exposure_times) ** (1.0 / max(len(exposure_times), 1)))
    if key in {"exposuremethod", "expmethod"}:
        stored = sensor.fields.get("exposure_method")
        if stored is not None:
            return _copy_metadata_value(stored)
        exposure_times = np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float)
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        if exposure_times.size <= 1:
            return "singleExposure"
        if exposure_times.ndim == 1:
            return "bracketedExposure"
        if exposure_times.shape == pattern.shape:
            return "cfaExposure"
        return None
    if key == "nexposures":
        return int(np.asarray(sensor.fields.get("integration_time")).size)
    if key == "exposureplane":
        if "exposure_plane" in sensor.fields:
            return int(sensor.fields["exposure_plane"])
        return int(np.floor(sensor_get(sensor, "n exposures") / 2.0) + 1)
    if key in {"cds", "correlateddoublesampling"}:
        return bool(sensor.fields.get("cds", False))
    if key == "gamma":
        return float(_sensor_render_state(sensor)["gamma"])
    if key in {"maxbright", "scalemax", "scaleintensity"}:
        return bool(_sensor_render_state(sensor)["scale"])
    if key in {"autoexp", "autoexposure", "automaticexposure"}:
        return bool(sensor.fields["auto_exposure"])
    if key in {"analoggain", "ag"}:
        return float(sensor.fields["analog_gain"])
    if key in {"analogoffset", "ao"}:
        return float(sensor.fields["analog_offset"])
    if key in {"pixelpdarea", "pdarea"}:
        area = _pixel_pd_area_m2(sensor)
        if args:
            area *= _spatial_unit_scale(args[0]) ** 2
        return float(area)
    if key in {"noiseflag", "shotnoiseflag"}:
        return int(sensor.fields["noise_flag"])
    if key in {"dr", "drdb20", "dynamicrange", "sensordynamicrange"}:
        integration_time = args[0] if args else None
        return _sensor_dynamic_range(sensor, integration_time)
    if key in {"fpnparameters", "fpn", "fpnoffsetgain", "fpnoffsetandgain"}:
        return np.array([sensor_get(sensor, "dsnu sigma"), sensor_get(sensor, "prnu sigma")], dtype=float)
    if key in {"dsnulevel", "sigmaoffsetfpn", "offsetfpn", "offset", "offsetsd", "offsetnoisevalue", "dsnusigma", "sigmadsnu"}:
        return float(sensor.fields["pixel"]["dsnu_sigma_v"])
    if key in {"sigmagainfpn", "gainfpn", "gain", "gainsd", "gainnoisevalue", "prnusigma", "sigmaprnu", "prnulevel"}:
        return float(sensor.fields["pixel"]["prnu_sigma"]) * 100.0
    if key in {"dsnuimage", "offsetfpnimage"}:
        stored = sensor.fields.get("offset_fpn_image")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"prnuimage", "gainfpnimage"}:
        stored = sensor.fields.get("gain_fpn_image")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"columnfpn", "columnfixedpatternnoise", "colfpn"}:
        return _sensor_column_fpn(sensor)
    if key in {"columndsnu", "columnfpnoffset", "colfpnoffset", "coldsnu"}:
        column_fpn = _sensor_column_fpn(sensor)
        return float(column_fpn[0]) if column_fpn.size >= 1 else 0.0
    if key in {"columnprnu", "columnfpngain", "colfpngain", "colprnu"}:
        column_fpn = _sensor_column_fpn(sensor)
        return float(column_fpn[1]) if column_fpn.size >= 2 else 0.0
    if key in {"coloffsetfpnvector", "coloffsetfpn", "coloffset"}:
        stored = sensor.fields.get("column_offset_fpn")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key in {"colgainfpnvector", "colgainfpn", "colgain"}:
        stored = sensor.fields.get("column_gain_fpn")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key == "reusenoise":
        return bool(sensor.fields.get("reuse_noise", False))
    if key == "noiseseed":
        return _copy_metadata_value(sensor.fields.get("noise_seed", 0))
    if key == "responsedr":
        volts = sensor.data.get("volts")
        if volts is None:
            return None
        volts_array = np.asarray(volts, dtype=float)
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
        v_max = float(np.max(volts_array))
        v_min = max(float(np.min(volts_array)), voltage_swing / float(2**12))
        return v_max / v_min
    if key in {"nbits", "bits"}:
        return int(sensor.fields["nbits"])
    if key in {"lut", "quantizationlut", "quantizatonlut"}:
        stored = sensor.fields.get("quantization_lut")
        return None if stored is None else np.asarray(stored, dtype=float).copy()
    if key == "quantizationstructure":
        return {
            "bits": int(sensor.fields["nbits"]),
            "method": _copy_metadata_value(sensor.fields["quantization"]),
            "lut": sensor_get(sensor, "lut"),
        }
    if key in {
        "vignetting",
        "vignettingflag",
        "pixelvignetting",
        "sensorvignetting",
        "bareetendue",
        "sensorbareetendue",
        "nomicrolensetendue",
    }:
        return sensor.fields.get("vignetting", 0)
    if key == "vignettingname":
        vignetting = sensor.fields.get("vignetting", 0)
        if isinstance(vignetting, str):
            normalized = param_format(vignetting)
            if normalized in {"", "skip"}:
                return "skip"
            if normalized in {"bare"}:
                return "bare"
            if normalized in {"centered"}:
                return "centered"
            if normalized in {"optimal"}:
                return "optimal"
            return str(vignetting)
        if vignetting in {0, None}:
            return "skip"
        if int(vignetting) == 1:
            return "bare"
        if int(vignetting) == 2:
            return "centered"
        if int(vignetting) == 3:
            return "optimal"
        return str(vignetting)
    if key in {"microlens", "ulens", "mlens", "ml"}:
        return _sensor_microlens(sensor)
    if key in {"microlensoffset", "mloffset", "microlensoffsetmicrons"}:
        return _copy_metadata_value(sensor.fields.get("mlOffset"))
    if key in {"etendue", "sensoretendue", "imagesensorarrayetendue"}:
        stored = sensor.fields.get("etendue")
        if stored is None:
            return np.ones(sensor.fields["size"], dtype=float)
        return np.asarray(stored, dtype=float).copy()
    if key in {"ngridsamples", "pixelsamples", "nsamplesperpixel", "npixelsamplesforcomputing", "spatialsamplesperpixel"}:
        return int(sensor.fields.get("n_samples_per_pixel", 1))
    if key == "quantization":
        return sensor.fields["quantization"]
    if key in {"qmethod", "quantizationmethod"}:
        return sensor.fields["quantization"]
    if key == "responsetype":
        return str(sensor.fields.get("response_type", "linear"))
    if key in {"pixelvoltageswing", "voltageswing"}:
        return float(sensor.fields["pixel"]["voltage_swing"])
    if key in {"maxvoltage", "max", "maxoutput"}:
        return float(sensor_get(sensor, "pixel voltage swing"))
    if key in {"metadatascenename", "scenename", "scene_name"}:
        value = sensor.metadata.get("scenename", sensor.metadata.get("scene_name"))
        return _copy_metadata_value(value)
    if key in {"metadataopticsname", "metadatalensname", "metadatalens", "lens"}:
        value = sensor.metadata.get("opticsname", sensor.metadata.get("lens"))
        return _copy_metadata_value(value)
    if key == "metadatasensorname":
        return _copy_metadata_value(sensor.metadata.get("sensorname"))
    if key == "metadatacrop":
        return _copy_metadata_value(sensor.metadata.get("crop"))
    if key in {"blacklevel", "zerolevel", "zero"}:
        return float(sensor.fields.get("zero_level", 0.0))
    if key in {"maxdigital", "maxdigitalvalue"}:
        nbits = int(sensor_get(sensor, "nbits"))
        zero_level = float(sensor_get(sensor, "zero level"))
        return float((2**nbits) - zero_level)
    if key in {"roi", "roilocs"}:
        roi = sensor.fields.get("roi")
        if roi is None:
            return None
        roi_array = np.asarray(roi, dtype=int)
        if key == "roilocs" and roi_array.ndim == 1 and roi_array.size == 4:
            from .roi import ie_rect2_locs

            return ie_rect2_locs(roi_array)
        return roi_array.copy()
    if key == "roirect":
        roi = sensor.fields.get("roi")
        if roi is None:
            return None
        roi_array = np.asarray(roi, dtype=int)
        if roi_array.ndim == 1 and roi_array.size == 4:
            return roi_array.copy()
        from .roi import ie_locs2_rect

        return ie_locs2_rect(roi_array)
    if key in {"volts", "voltage"}:
        volts = sensor.data.get("volts")
        if args and volts is not None:
            return _sensor_color_data(sensor, volts, args[0])
        return volts
    if key == "voltimages":
        return _sensor_plane_images(sensor, sensor.data.get("volts"))
    if key in {"electrons", "electron"}:
        electrons = _sensor_electrons(sensor)
        if args and electrons is not None:
            return _sensor_color_data(sensor, electrons, args[0])
        return electrons
    if key == "electronsperarea":
        units = args[0] if args else "m"
        channel = args[1] if len(args) > 1 else None
        electrons = sensor_get(sensor, "electrons")
        if electrons is None:
            return None
        values = np.asarray(electrons, dtype=float) / max(_pixel_pd_area_m2(sensor), 1e-30)
        values = values / (_spatial_unit_scale(units) ** 2)
        if channel is not None:
            return _sensor_color_data(sensor, values, channel)
        return values
    if key in {"dv", "digitalvalue", "digitalvalues"}:
        dv = sensor.data.get("dv")
        if args and dv is not None:
            return _sensor_color_data(sensor, dv, args[0])
        return dv
    if key in {"ncaptures", "ncapture"}:
        volts = sensor.data.get("volts")
        if volts is not None and np.asarray(volts).ndim >= 3:
            return int(np.asarray(volts).shape[2])
        dv = sensor.data.get("dv")
        if dv is not None and np.asarray(dv).ndim >= 3:
            return int(np.asarray(dv).shape[2])
        integration_time = np.asarray(sensor.fields.get("integration_time"))
        if integration_time.ndim == 1 and integration_time.size > 1:
            return int(integration_time.size)
        return 1
    if key in {"dvorvolts", "digitalorvolts"}:
        return sensor.data.get("dv", sensor.data.get("volts"))
    if key in {"hlinevolts", "hlineelectrons", "hlinedv", "vlinevolts", "vlineelectrons", "vlinedv"}:
        if not args:
            raise ValueError("Specify row or col.")
        return _sensor_line_profile(sensor, key, args[0])
    if key in {"roivolts", "roidata", "roidatav", "roidatavolts"}:
        roi_locs = sensor_get(sensor, "roi locs")
        if roi_locs is None:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(sensor, roi_locs, "volts")
    if key in {"roielectrons", "roidatae", "roidataelectrons"}:
        roi_locs = sensor_get(sensor, "roi locs")
        if roi_locs is None:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(sensor, roi_locs, "electrons")
    if key == "chromaticity":
        rect = args[0] if args else None
        mode = args[1] if len(args) >= 2 else "vec"
        return _sensor_chromaticity(sensor, rect, str(mode))
    if key == "roichromaticitymean":
        if not args:
            return None
        chromaticity = sensor_get(sensor, "chromaticity", args[0], "vec")
        if chromaticity is None:
            return None
        return np.nanmean(np.asarray(chromaticity, dtype=float), axis=0)
    if key in {"roidv", "roidigitalcount"}:
        roi_locs = sensor_get(sensor, "roi locs")
        if roi_locs is None:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(sensor, roi_locs, "dv")
    if key == "roivoltsmean":
        roi_data = sensor_get(sensor, "roi volts")
        if roi_data is None:
            return None
        return np.nanmean(np.asarray(roi_data, dtype=float), axis=0)
    if key == "roielectronsmean":
        roi_data = sensor_get(sensor, "roi electrons")
        if roi_data is None:
            return None
        return np.nanmean(np.asarray(roi_data, dtype=float), axis=0)
    if key in {"responseratio", "volts2maxratio"}:
        volts = sensor.data.get("volts")
        if volts is not None:
            voltage_swing = float(sensor.fields["pixel"]["voltage_swing"])
            return float(np.max(np.asarray(volts, dtype=float)) / max(voltage_swing, 1e-12))
        dv = sensor.data.get("dv")
        if dv is not None:
            nbits = int(sensor.fields["nbits"])
            max_digital = float(2**nbits)
            return float(np.max(np.asarray(dv, dtype=float)) / max(max_digital, 1e-12))
        return 0.0
    if key == "rgb":
        data_type = str(args[0]) if args else "volts"
        gamma = float(args[1]) if len(args) > 1 else None
        scale_max = bool(args[2]) if len(args) > 2 else None
        return _sensor_rgb_image(sensor, data_type, gamma, scale_max)
    if key in {"fovhorizontal", "fov"}:
        scene_or_distance = args[0] if args else None
        oi = args[1] if len(args) >= 2 else args[0] if args and isinstance(args[0], OpticalImage) else None
        focal_length = _sensor_image_distance_m(scene_or_distance, oi)
        width = float(sensor_get(sensor, "width"))
        return float(np.rad2deg(2.0 * np.arctan2(width / 2.0, focal_length)))
    if key in {"fovvertical", "vfov"}:
        scene_or_distance = args[0] if args else None
        oi = args[1] if len(args) >= 2 else args[0] if args and isinstance(args[0], OpticalImage) else None
        focal_length = _sensor_image_distance_m(scene_or_distance, oi)
        height = float(sensor_get(sensor, "height"))
        return float(np.rad2deg(2.0 * np.arctan2(height / 2.0, focal_length)))
    if key in {"hdegperdistance", "degperdistance"}:
        unit = args[0] if args else "m"
        scene_or_distance = args[1] if len(args) >= 2 else None
        oi = args[2] if len(args) >= 3 else args[1] if len(args) >= 2 and isinstance(args[1], OpticalImage) else None
        width = float(sensor_get(sensor, "width", unit))
        fov = float(sensor_get(sensor, "fov", scene_or_distance, oi))
        return fov / max(width, 1.0e-12)
    if key == "vdegperdistance":
        unit = args[0] if args else "m"
        scene_or_distance = args[1] if len(args) >= 2 else None
        oi = args[2] if len(args) >= 3 else args[1] if len(args) >= 2 and isinstance(args[1], OpticalImage) else None
        height = float(sensor_get(sensor, "height", unit))
        fov = float(sensor_get(sensor, "vfov", scene_or_distance, oi))
        return fov / max(height, 1.0e-12)
    raise KeyError(f"Unsupported sensorGet parameter: {parameter}")


def sensor_set(sensor: Sensor, parameter: str, value: Any, *args: Any) -> Sensor:
    object_type, object_param = ie_parameter_otype(parameter)
    if object_type == "pixel":
        try:
            return _sensor_pixel_set(sensor, object_param or "pixel", value)
        except KeyError:
            pass
    key = param_format(parameter)
    if key == "name":
        sensor.name = str(value)
        return sensor
    if key == "human":
        sensor.fields["human"] = None if value is None else _human_struct_from_value(value)
        return sensor
    if key in {"humanconetype", "conetype"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["coneType"] = np.asarray(value).copy()
        sensor.fields["human"] = human
        return sensor
    if key in {"humanconedensities", "densities"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["densities"] = np.asarray(value, dtype=float).copy()
        sensor.fields["human"] = human
        return sensor
    if key in {"humanconelocs", "conexy", "conelocs", "xy"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["xy"] = np.asarray(value, dtype=float).copy()
        sensor.fields["human"] = human
        return sensor
    if key in {"humanrseed", "humanconeseed", "rseed"}:
        human = sensor.fields.get("human")
        if not isinstance(human, dict):
            human = {}
        human["rSeed"] = _copy_metadata_value(value)
        sensor.fields["human"] = human
        return sensor
    if key in {"sensormovement", "eyemovement"}:
        sensor.fields["movement"] = _movement_struct_from_value(value)
        return sensor
    if key in {"movementpositions", "sensorpositions"}:
        movement = _sensor_movement(sensor)
        movement["pos"] = _movement_positions_from_value(value)
        sensor.fields["movement"] = movement
        return sensor
    if key in {"framesperposition", "framesperpositions", "exposuretimesperposition", "etimeperpos"}:
        movement = _sensor_movement(sensor)
        movement["framesPerPosition"] = _copy_metadata_value(value)
        sensor.fields["movement"] = movement
        return sensor
    if key in {"spectrum", "sensorspectrum"}:
        spectrum = _spectrum_struct_from_value(value)
        sensor = _sensor_update_wave(sensor, np.asarray(spectrum["wave"], dtype=float).reshape(-1))
        sensor.fields["spectrum"] = spectrum
        sensor.fields["spectrum"]["wave"] = np.asarray(sensor.fields["wave"], dtype=float).copy()
        return sensor
    if key == "color":
        color = _sensor_color_struct_from_value(value)
        if "filterSpectra" in color:
            sensor.fields["filter_spectra"] = np.asarray(color["filterSpectra"], dtype=float)
        if "filterNames" in color:
            sensor.fields["filter_names"] = list(color["filterNames"])
        if "irFilter" in color:
            ir_filter = np.asarray(color["irFilter"], dtype=float).reshape(-1)
            if ir_filter.size == 1:
                sensor.fields["ir_filter"] = np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(ir_filter[0]), dtype=float)
            elif ir_filter.size == np.asarray(sensor.fields["wave"], dtype=float).size:
                sensor.fields["ir_filter"] = ir_filter
            else:
                raise ValueError("IR filter must match the sensor wavelength sampling.")
        return sensor
    if key in {"chartparameters"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"].update(dict(value))
        return sensor
    if key in {"chartcornerpoints", "cornerpoints", "chartcorners"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["cornerPoints"] = np.asarray(value).copy()
        return sensor
    if key == "mcccornerpoints":
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["cornerPoints"] = np.asarray(value).copy()
        return sensor
    if key in {"chartrects", "chartrectangles"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["rects"] = np.asarray(value).copy()
        return sensor
    if key in {"chartcurrentrect", "currentrect"}:
        sensor.fields["chartP"] = _sensor_chart_parameters(sensor)
        sensor.fields["chartP"]["currentRect"] = np.asarray(value).copy()
        return sensor
    if key == "mccrecthandles":
        sensor.fields["mccRectHandles"] = _copy_metadata_value(value)
        return sensor
    if key in {"wave", "wavelength", "wavelengthsamples"}:
        sensor = _sensor_update_wave(sensor, np.asarray(value, dtype=float).reshape(-1))
        return sensor
    if key == "size":
        sensor = sensor_set(sensor, "rows", value[0])
        sensor = sensor_set(sensor, "cols", value[1])
        return sensor
    if key == "matchoi":
        if not isinstance(value, OpticalImage):
            raise ValueError("match oi requires an optical image.")
        pixel_size = float(oi_get(value, "width spatial resolution", "m"))
        sensor = sensor_set(sensor, "pixel size same fill factor", pixel_size)
        oi_size = np.asarray(oi_get(value, "size"), dtype=int).reshape(-1)
        if oi_size.size != 2:
            raise ValueError("match oi requires an optical image with a 2D size.")
        return sensor_set(sensor, "size", oi_size)
    if key in {"fov", "hfov", "horizontalfieldofview"}:
        oi = args[0] if args else None
        if not isinstance(oi, OpticalImage):
            raise ValueError("sensorSet(..., 'hfov', ...) requires an optical image.")
        return sensor_set_size_to_fov(sensor, float(value), oi)
    if key in {"vfov", "verticalfieldofview"}:
        hfov = float(sensor_get(sensor, "fov"))
        if abs(hfov) <= 0.0:
            raise ValueError("sensor vfov requires a non-zero horizontal field of view.")
        size = np.asarray(sensor_get(sensor, "size"), dtype=float).reshape(-1)
        new_rows = int(round((size[1] / hfov) * float(value)))
        return sensor_set(sensor, "rows", new_rows)
    if key in {"rows", "row"}:
        block_rows, _ = _sensor_unit_block(sensor)
        rows = _sensor_aligned_dimension(value, block_rows)
        sensor.fields["size"] = (rows, int(sensor.fields["size"][1]))
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"cols", "col"}:
        _, block_cols = _sensor_unit_block(sensor)
        cols = _sensor_aligned_dimension(value, block_cols)
        sensor.fields["size"] = (int(sensor.fields["size"][0]), cols)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"cfapattern", "pattern"}:
        sensor.fields["pattern"] = np.asarray(value, dtype=int)
        return sensor
    if key in {"cfapreset", "cfapatternpreset", "cfatype", "cfalayout"}:
        return sensor_set(sensor, "pattern and size", _sensor_cfa_preset_pattern(value))
    if key in {"patternandsize", "patternsize", "cfapatternandsize"}:
        pattern = np.asarray(value, dtype=int)
        sensor.fields["pattern"] = pattern
        rows, cols = _sensor_rows_cols(sensor)
        block_rows, block_cols = pattern.shape
        new_rows = ensure_multiple(rows, block_rows)
        new_cols = ensure_multiple(cols, block_cols)
        if new_rows != rows or new_cols != cols:
            sensor.fields["size"] = (new_rows, new_cols)
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {"colorfilterarray", "cfa"}:
        sensor.fields["pattern"] = _cfa_pattern_from_value(value)
        return sensor
    if key in {"filterspectra", "colorfilters", "filtertransmissivities"}:
        sensor.fields["filter_spectra"] = np.asarray(value, dtype=float)
        return sensor
    if key in {"pixelspectralqe", "pdspectralqe", "pixelqe"}:
        current = _sensor_pixel_qe(sensor)
        qe = np.asarray(value, dtype=float).reshape(-1)
        if qe.size == 1:
            sensor.fields["pixel_qe"] = current * float(qe[0])
            return sensor
        if qe.size != np.asarray(sensor.fields["wave"], dtype=float).size:
            raise ValueError("pixel spectral QE must match the sensor wavelength sampling.")
        sensor.fields["pixel_qe"] = qe
        return sensor
    if key in {"ir", "infraredfilter", "irfilter", "otherfilter"}:
        ir_filter = np.asarray(value, dtype=float).reshape(-1)
        if ir_filter.size == 1:
            sensor.fields["ir_filter"] = np.full(np.asarray(sensor.fields["wave"], dtype=float).size, float(ir_filter[0]), dtype=float)
            return sensor
        if ir_filter.size != np.asarray(sensor.fields["wave"], dtype=float).size:
            raise ValueError("IR filter must match the sensor wavelength sampling.")
        sensor.fields["ir_filter"] = ir_filter
        return sensor
    if key in {"filternames", "filtername"}:
        sensor.fields["filter_names"] = list(value)
        return sensor
    if key in {"consistency", "sensorconsistency"}:
        sensor.fields["consistency"] = bool(value)
        return sensor
    if key in {"sensorcompute", "sensorcomputemethod"}:
        sensor.fields["sensor_compute_method"] = _copy_metadata_value(value)
        return sensor
    if key in {"oclgroupproxy", "oclgroup", "sharedoclaperture"}:
        sensor.fields["ocl_group_proxy"] = _sensor_ocl_group_proxy_from_value(value)
        _sensor_clear_data(sensor)
        return sensor
    if key in {"oclgroupshape", "sharedoclshape"}:
        proxy = _sensor_ocl_group_proxy(sensor)
        proxy["shape"] = _sensor_ocl_group_shape_from_value(value)
        if "equalization" not in proxy or float(proxy["equalization"]) <= 0.0:
            proxy["equalization"] = 0.0 if tuple(proxy["shape"]) == (1, 1) else 1.0
        proxy["mode"] = (
            "off"
            if tuple(proxy["shape"]) == (1, 1) or float(proxy["equalization"]) <= 0.0
            else "uniform"
        )
        sensor.fields["ocl_group_proxy"] = proxy
        _sensor_clear_data(sensor)
        return sensor
    if key in {"oclgroupequalization", "oclsharingequalization"}:
        proxy = _sensor_ocl_group_proxy(sensor)
        proxy["equalization"] = _sensor_ocl_group_equalization_from_value(value)
        proxy["mode"] = (
            "off"
            if tuple(proxy["shape"]) == (1, 1) or float(proxy["equalization"]) <= 0.0
            else "uniform"
        )
        sensor.fields["ocl_group_proxy"] = proxy
        _sensor_clear_data(sensor)
        return sensor
    if key in {"fdtdsensor", "fdtdlut", "fdtdopticalresponse", "fdtdopticalresponselut"}:
        sensor.fields["fdtd_sensor"] = None if value is None else fdtd_sensor_config(value)
        return sensor
    if key in {"tcadsensor", "tcadlut", "tcadcollection", "tcadcollectionlut", "tcadcollectionresponse"}:
        sensor.fields["tcad_sensor"] = None if value is None else tcad_sensor_config(value)
        return sensor
    if key in {"roi", "roilocs"}:
        roi = np.asarray(value, dtype=float)
        if roi.ndim == 1 and roi.size == 4:
            sensor.fields["roi"] = np.rint(roi).astype(int)
            return sensor
        if roi.ndim == 2 and roi.shape[1] == 2:
            sensor.fields["roi"] = np.rint(roi).astype(int)
            return sensor
        raise ValueError("sensor ROI must be a rect or an Nx2 location array.")
    if key == "roirect":
        rect = np.rint(np.asarray(value, dtype=float).reshape(-1)).astype(int)
        if rect.size != 4:
            raise ValueError("sensor ROI rect must contain [col, row, width, height].")
        sensor.fields["roi"] = rect
        return sensor
    if key in {"pixelsizesamefillfactor", "pixelsize"}:
        size_value = np.asarray(value, dtype=float)
        if size_value.size == 1:
            size_value = np.repeat(size_value, 2)
        sensor.fields["pixel"]["size_m"] = size_value
        sensor.fields["etendue"] = None
        _sensor_clear_data(sensor)
        return sensor
    if key in {
        "integrationtime",
        "integrationtimes",
        "exptime",
        "exptimes",
        "exposuretimes",
        "exposuretime",
        "expduration",
        "exposureduration",
        "exposuredurations",
    }:
        integration_time = np.asarray(value, dtype=float)
        if integration_time.ndim == 0:
            sensor.fields["integration_time"] = float(integration_time)
        else:
            sensor.fields["integration_time"] = integration_time.copy()
        sensor.fields["auto_exposure"] = False
        return sensor
    if key == "exposureplane":
        sensor.fields["exposure_plane"] = int(np.rint(float(value)))
        return sensor
    if key in {"exposuremethod", "expmethod"}:
        sensor.fields["exposure_method"] = _copy_metadata_value(value)
        return sensor
    if key in {"cds", "correlateddoublesampling"}:
        sensor.fields["cds"] = bool(value)
        return sensor
    if key == "diffusionmtf":
        sensor.fields["diffusion_mtf"] = None if value is None else copy.deepcopy(value)
        return sensor
    if key in {"blacklevel", "zerolevel", "zero"}:
        sensor.fields["zero_level"] = float(value)
        return sensor
    if key in {"metadatascenename", "scenename", "scene_name"}:
        copied = _copy_metadata_value(value)
        sensor.metadata["scenename"] = copied
        sensor.metadata["scene_name"] = _copy_metadata_value(copied)
        return sensor
    if key in {"metadataopticsname", "metadatalensname", "metadatalens", "lens"}:
        copied = _copy_metadata_value(value)
        sensor.metadata["opticsname"] = copied
        sensor.metadata["lens"] = _copy_metadata_value(copied)
        return sensor
    if key == "metadatasensorname":
        sensor.metadata["sensorname"] = _copy_metadata_value(value)
        return sensor
    if key == "metadatacrop":
        sensor.metadata["crop"] = _copy_metadata_value(value)
        return sensor
    if key == "gamma":
        _sensor_render_state(sensor)["gamma"] = float(value)
        return sensor
    if key in {"maxbright", "scalemax", "scaleintensity"}:
        _sensor_render_state(sensor)["scale"] = bool(value)
        return sensor
    if key in {"autoexp", "autoexposure", "automaticexposure"}:
        if isinstance(value, str):
            enabled = param_format(value) == "on"
        else:
            enabled = bool(value)
        sensor.fields["auto_exposure"] = enabled
        if enabled:
            integration_time = np.asarray(sensor.fields.get("integration_time", 0.0))
            pattern = np.asarray(sensor.fields["pattern"], dtype=int)
            if integration_time.ndim > 0 and integration_time.shape == pattern.shape:
                sensor.fields["integration_time"] = np.zeros(pattern.shape, dtype=float)
            else:
                sensor.fields["integration_time"] = 0.0
        return sensor
    if key in {"analoggain", "ag"}:
        sensor.fields["analog_gain"] = float(value)
        return sensor
    if key in {"analogoffset", "ao"}:
        sensor.fields["analog_offset"] = float(value)
        return sensor
    if key == "noiseflag":
        sensor.fields["noise_flag"] = int(value)
        return sensor
    if key in {"dsnulevel", "sigmaoffsetfpn", "offsetfpn", "offset", "offsetsd", "offsetnoisevalue", "dsnusigma", "sigmadsnu"}:
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(value)
        sensor.fields.pop("offset_fpn_image", None)
        return sensor
    if key in {"sigmagainfpn", "gainfpn", "gain", "gainsd", "gainnoisevalue", "prnusigma", "sigmaprnu", "prnulevel"}:
        sensor.fields["pixel"]["prnu_sigma"] = float(value) / 100.0
        sensor.fields.pop("gain_fpn_image", None)
        return sensor
    if key in {"dsnuimage", "offsetfpnimage"}:
        if value is None:
            sensor.fields.pop("offset_fpn_image", None)
            return sensor
        image = np.asarray(value, dtype=float)
        if image.shape != tuple(sensor.fields["size"]):
            raise ValueError("DSNU image must match the sensor size.")
        sensor.fields["offset_fpn_image"] = image
        return sensor
    if key in {"prnuimage", "gainfpnimage"}:
        if value is None:
            sensor.fields.pop("gain_fpn_image", None)
            return sensor
        image = np.asarray(value, dtype=float)
        if image.shape != tuple(sensor.fields["size"]):
            raise ValueError("PRNU image must match the sensor size.")
        sensor.fields["gain_fpn_image"] = image
        return sensor
    if key in {"columnfpnparameters", "columnfpn", "columnfixedpatternnoise", "colfpn"}:
        column_fpn = np.asarray(value, dtype=float).reshape(-1)
        if column_fpn.size == 0:
            sensor.fields.pop("column_fpn", None)
            return sensor
        if column_fpn.size != 2:
            raise ValueError("Column FPN must be in [offset, gain] format.")
        sensor.fields["column_fpn"] = column_fpn.copy()
        return sensor
    if key in {"colgainfpnvector", "columnprnu"}:
        column_gain = np.asarray(value, dtype=float).reshape(-1)
        if column_gain.size == 0:
            sensor.fields.pop("column_gain_fpn", None)
            return sensor
        if column_gain.size != int(sensor_get(sensor, "cols")):
            raise ValueError("Bad column gain data.")
        sensor.fields["column_gain_fpn"] = column_gain.copy()
        return sensor
    if key in {"coloffsetfpnvector", "columndsnu"}:
        column_offset = np.asarray(value, dtype=float).reshape(-1)
        if column_offset.size == 0:
            sensor.fields.pop("column_offset_fpn", None)
            return sensor
        if column_offset.size != int(sensor_get(sensor, "cols")):
            raise ValueError("Bad column offset data.")
        sensor.fields["column_offset_fpn"] = column_offset.copy()
        return sensor
    if key == "reusenoise":
        sensor.fields["reuse_noise"] = bool(value)
        return sensor
    if key == "noiseseed":
        sensor.fields["noise_seed"] = _copy_metadata_value(value)
        return sensor
    if key in {
        "vignetting",
        "vignettingflag",
        "pixelvignetting",
        "sensorvignetting",
        "bareetendue",
        "sensorbareetendue",
        "nomicrolensetendue",
    }:
        sensor.fields["vignetting"] = value
        sensor.fields["etendue"] = None
        return sensor
    if key in {"microlens", "ulens", "mlens", "ml"}:
        sensor.fields["ml"] = None if value is None else _microlens_struct_from_value(value)
        return sensor
    if key in {"microlensoffset", "mloffset", "microlensoffsetmicrons"}:
        sensor.fields["mlOffset"] = _copy_metadata_value(value)
        return sensor
    if key in {"etendue", "sensoretendue", "imagesensorarrayetendue"}:
        etendue = np.asarray(value, dtype=float)
        if etendue.shape != tuple(sensor.fields["size"]):
            raise ValueError("sensor etendue must match the sensor size.")
        sensor.fields["etendue"] = etendue
        return sensor
    if key in {"ngridsamples", "pixelsamples", "nsamplesperpixel", "npixelsamplesforcomputing", "spatialsamplesperpixel"}:
        sensor.fields["n_samples_per_pixel"] = int(value)
        return sensor
    if key in {"quantization", "qmethod", "quantizationmethod"}:
        sensor.fields["quantization"] = str(value)
        return sensor
    if key in {"nbits", "bits"}:
        sensor.fields["nbits"] = int(value)
        return sensor
    if key in {"lut", "quantizationlut", "quantizatonlut"}:
        sensor.fields["quantization_lut"] = None if value is None else np.asarray(value, dtype=float).copy()
        return sensor
    if key == "quantizationstructure":
        payload = dict(value) if isinstance(value, dict) else dict(vars(value))
        if "bits" in payload:
            sensor.fields["nbits"] = int(payload["bits"])
        if "method" in payload:
            sensor.fields["quantization"] = str(payload["method"])
        if "lut" in payload:
            sensor.fields["quantization_lut"] = None if payload["lut"] is None else np.asarray(payload["lut"], dtype=float).copy()
        return sensor
    if key == "responsetype":
        normalized = str(param_format(value))
        if normalized not in {"linear", "log"}:
            raise ValueError("response type must be 'linear' or 'log'.")
        sensor.fields["response_type"] = normalized
        return sensor
    if key in {"volts", "voltage"}:
        volts = np.asarray(value, dtype=float)
        sensor.data["volts"] = volts
        if param_format(sensor.fields.get("quantization", "analog")) == "analog":
            sensor.data.pop("dv", None)
        if volts.ndim >= 2:
            sensor.fields["size"] = (int(volts.shape[0]), int(volts.shape[1]))
        return sensor
    if key in {"dv", "digitalvalue", "digitalvalues"}:
        dv = np.asarray(value, dtype=float)
        sensor.data["dv"] = dv
        if dv.ndim >= 2:
            sensor.fields["size"] = (int(dv.shape[0]), int(dv.shape[1]))
        return sensor
    raise KeyError(f"Unsupported sensorSet parameter: {parameter}")


def sensor_set_size_to_fov(sensor: Sensor, fov: float | tuple[float, float], oi: OpticalImage) -> Sensor:
    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    pattern_rows, pattern_cols = pattern.shape
    focal_length = float(oi_get(oi, "focal length"))
    if isinstance(fov, (tuple, list, np.ndarray)):
        hfov = float(fov[0])
        vfov = float(fov[1] if len(fov) > 1 else fov[0])
        width = 2.0 * focal_length * np.tan(np.deg2rad(hfov) / 2.0)
        height = 2.0 * focal_length * np.tan(np.deg2rad(vfov) / 2.0)
        cols = max(pattern_cols, int(round(width / float(sensor_get(sensor, "deltax")))))
        rows = max(pattern_rows, int(round(height / float(sensor_get(sensor, "deltay")))))
    else:
        hfov = float(fov)
        width = 2.0 * focal_length * np.tan(np.deg2rad(hfov) / 2.0)
        current_width = float(sensor_get(sensor, "width"))
        scale = width / max(current_width, 1e-12)
        rows = max(pattern_rows, int(round(sensor_get(sensor, "rows") * scale)))
        cols = max(pattern_cols, int(round(sensor_get(sensor, "cols") * scale)))
    cols = ensure_multiple(cols, pattern_cols)
    rows = ensure_multiple(rows, pattern_rows)
    sensor = sensor_set(sensor, "rows", rows)
    sensor = sensor_set(sensor, "cols", cols)
    return sensor


def _mlens_sensor_and_unit(args: tuple[Any, ...]) -> tuple[Sensor | None, Any | None]:
    if not args:
        return None, None
    if isinstance(args[0], Sensor):
        return args[0], args[1] if len(args) > 1 else None
    return None, args[0]


def _mlens_spatial_value_to_meters(value: Any, unit: Any | None) -> float:
    scale = _spatial_unit_scale(unit)
    return float(value) / max(scale, 1e-30)


def _mlens_offset_to_microns(value: Any, unit: Any | None) -> float:
    if unit is None:
        return float(value)
    return float(value) * (1e6 / max(_spatial_unit_scale(unit), 1e-30))


def _ffndgrid_average_2d(
    x_coords: np.ndarray,
    u_coords: np.ndarray,
    values: np.ndarray,
    x_axis: np.ndarray,
    u_axis: np.ndarray,
) -> np.ndarray:
    x_axis = np.asarray(x_axis, dtype=float).reshape(-1)
    u_axis = np.asarray(u_axis, dtype=float).reshape(-1)
    x_coords = np.asarray(x_coords, dtype=float).reshape(-1)
    u_coords = np.asarray(u_coords, dtype=float).reshape(-1)
    samples = np.asarray(values, dtype=float).reshape(-1)

    if x_axis.size == 0 or u_axis.size == 0:
        return np.zeros((u_axis.size, x_axis.size), dtype=float)
    dx = 1.0 if x_axis.size == 1 else float((x_axis[-1] - x_axis[0]) / max(x_axis.size - 1, 1))
    du = 1.0 if u_axis.size == 1 else float((u_axis[-1] - u_axis[0]) / max(u_axis.size - 1, 1))

    x_index = _matlab_round_to_int((x_coords - float(x_axis[0])) / max(dx, 1e-30))
    u_index = _matlab_round_to_int((u_coords - float(u_axis[0])) / max(du, 1e-30))
    valid = (
        (x_index >= 0)
        & (x_index < x_axis.size)
        & (u_index >= 0)
        & (u_index < u_axis.size)
        & np.isfinite(samples)
    )

    binned = np.zeros((u_axis.size, x_axis.size), dtype=float)
    counts = np.zeros((u_axis.size, x_axis.size), dtype=float)
    if np.any(valid):
        np.add.at(binned, (u_index[valid], x_index[valid]), samples[valid])
        np.add.at(counts, (u_index[valid], x_index[valid]), 1.0)
    nonzero = counts > 0.0
    binned[nonzero] /= counts[nonzero]
    return binned


def _ml_coordinates(x1: float, x2: float, n: float, _lambda_um: float) -> tuple[np.ndarray, np.ndarray]:
    n_points = 255
    x = np.linspace(float(x1), float(x2), n_points, dtype=float)
    u = np.linspace(-float(n) * 0.99, float(n) * 0.99, n_points, dtype=float)
    return np.meshgrid(x, u)


def _ml_source(
    x1: float,
    x2: float,
    u1: float,
    u2: float,
    X: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    x = np.asarray(X[0, :], dtype=float)
    u = np.asarray(U[:, 0], dtype=float)
    dx = 0.5 * float(np.mean(np.diff(x))) if x.size > 1 else 0.0
    du = 0.5 * float(np.mean(np.diff(u))) if u.size > 1 else 0.0
    W = np.zeros_like(X, dtype=float)

    x_equal = np.isclose(x1, x2)
    u_equal = np.isclose(u1, u2)
    if x_equal:
        x_mask = np.abs(x - float(x1)) < max(dx, 1e-12)
    else:
        x_mask = (x > float(x1)) & (x < float(x2))
    if u_equal:
        u_mask = np.abs(u - float(u1)) < max(du, 1e-12)
    else:
        u_mask = (u > float(u1)) & (u < float(u2))
    if np.any(x_mask) and np.any(u_mask):
        W[np.ix_(u_mask, x_mask)] = 1.0
    return W


def _ml_lens(
    focal_length_um: float,
    W_in: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    *,
    propagation_type: str = "non-paraxial",
) -> np.ndarray:
    x = np.asarray(X[0, :], dtype=float)
    u = np.asarray(U[:, 0], dtype=float)
    if param_format(propagation_type) == "paraxial":
        new_u = U + (X / max(float(focal_length_um), 1e-30))
    else:
        new_u = U + np.sin(np.arctan2(X, max(float(focal_length_um), 1e-30)))
    return _ffndgrid_average_2d(X, new_u, W_in, x, u)


def _ml_displacement(
    displacement_um: float,
    W_in: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    x = np.asarray(X[0, :], dtype=float)
    u = np.asarray(U[:, 0], dtype=float)
    return _ffndgrid_average_2d(X + float(displacement_um), U, W_in, x, u)


def _ml_propagate(
    distance_um: float,
    refractive_index: float,
    W_in: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    *,
    propagation_type: str = "non-paraxial",
) -> np.ndarray:
    x = np.asarray(X[0, :], dtype=float)
    u = np.asarray(U[:, 0], dtype=float)
    if param_format(propagation_type) == "paraxial":
        new_x = X - (float(distance_um) / max(float(refractive_index), 1e-30)) * U
    else:
        argument = np.clip(U / max(float(refractive_index), 1e-30), -1.0, 1.0)
        new_x = X - float(distance_um) * np.tan(np.arcsin(argument))
    return _ffndgrid_average_2d(new_x, U, W_in, x, u)


def mlens_create(
    sensor: Sensor | None = None,
    oi: OpticalImage | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    if sensor is None:
        sensor = sensor_create(asset_store=asset_store)
    if oi is None:
        from .optics import oi_create

        oi = oi_create(asset_store=asset_store)

    focal_length_m = float(oi_get(oi, "optics focal length"))
    source_f_number = float(oi_get(oi, "optics fnumber"))
    pixel_depth_m = float(sensor_get(sensor, "pixel depth"))
    pixel_width_m = float(sensor_get(sensor, "pixel width"))
    microlens = {
        "name": "default",
        "type": "microlens",
        "rayAngle": 0.0,
        "wavelength": 500.0,
        "sourceFNumber": source_f_number,
        "sourceFocalLength": focal_length_m,
        "focalLength": pixel_depth_m,
        "fnumber": pixel_depth_m / max(pixel_width_m, 1e-30),
        "offset": 0.0,
        "refractiveIndex": 1.5,
    }
    return copy.deepcopy(microlens)


def mlens_set(mlens: dict[str, Any], parameter: str, value: Any, *args: Any) -> dict[str, Any]:
    microlens = _microlens_struct_from_value(mlens)
    key = param_format(parameter)
    if key in {"name", "title"}:
        microlens["name"] = str(value)
        return microlens
    if key in {"wavelength", "sourcewavelength"}:
        microlens["wavelength"] = float(value)
        return microlens
    if key in {"chiefrayangle", "rayangle", "chiefray", "chiefrayangledegrees"}:
        microlens["rayAngle"] = float(value)
        return microlens
    if key in {"sourcefnumber", "sfnumber"}:
        microlens["sourceFNumber"] = float(value)
        return microlens
    if key in {"sourcefocallength", "sourceflength"}:
        microlens["sourceFocalLength"] = _mlens_spatial_value_to_meters(value, args[0] if args else None)
        return microlens
    if key == "sourceirradiance":
        microlens["sourceIrradiance"] = np.asarray(value, dtype=float).copy()
        return microlens
    if key in {"mlfnumber", "fnumber", "microlensfnumber"}:
        microlens["fnumber"] = float(value)
        return microlens
    if key in {"mlfocallength", "mlflength", "microlensfocallength"}:
        microlens["focalLength"] = _mlens_spatial_value_to_meters(value, args[0] if args else None)
        return microlens
    if key in {"mloffset", "microlensoffset", "offset", "microlensoffsetmicrons"}:
        microlens["offset"] = _mlens_offset_to_microns(value, args[0] if args else None)
        return microlens
    if key in {"mlrefractiveindex", "microlensrefractiveindex", "mlrefindx"}:
        microlens["refractiveIndex"] = float(value)
        return microlens
    if key in {"xcoordinate", "spacecoordinate"}:
        microlens["x"] = np.asarray(value, dtype=float).copy()
        return microlens
    if key in {"anglecoordinate", "pcoordinate"}:
        microlens["p"] = np.asarray(value, dtype=float).copy()
        return microlens
    if key in {"pixelirradiance", "irradiance", "pirradiance"}:
        microlens["pixelIrradiance"] = np.asarray(value, dtype=float).copy()
        return microlens
    if key == "etendue":
        microlens["E"] = float(value)
        return microlens
    raise UnsupportedOptionError("mlensSet", parameter)


def mlens_get(mlens: dict[str, Any], parameter: str, *args: Any) -> Any:
    microlens = _microlens_struct_from_value(mlens)
    key = param_format(parameter)
    sensor_arg, unit = _mlens_sensor_and_unit(args)

    if key in {"name", "title"}:
        return str(microlens.get("name", ""))
    if key == "type":
        return str(microlens.get("type", "microlens"))
    if key == "wavelength":
        wavelength_nm = float(microlens.get("wavelength", 500.0))
        if unit is None:
            return wavelength_nm
        return wavelength_nm * 1e-9 * _spatial_unit_scale(unit)
    if key in {"chiefrayangle", "rayangle", "chiefray", "chiefrayangledegrees"}:
        return float(microlens.get("rayAngle", 0.0))
    if key == "chiefrayangleradians":
        return float(np.deg2rad(float(microlens.get("rayAngle", 0.0))))
    if key in {"mlfocallength", "microlensfocallength", "mlflength", "focallength"}:
        focal_length_m = float(microlens.get("focalLength", 0.0))
        if unit is None:
            return focal_length_m
        return focal_length_m * _spatial_unit_scale(unit)
    if key in {"mlfnumber", "fnumber", "microlensfnumber"}:
        return float(microlens.get("fnumber", 0.0))
    if key in {"mldiameter", "diameter"}:
        diameter_m = float(microlens.get("focalLength", 0.0)) / max(float(microlens.get("fnumber", 1.0)), 1e-30)
        if unit is None:
            return diameter_m
        return diameter_m * _spatial_unit_scale(unit)
    if key in {"microlensrefractiveindex", "mlrefindx", "mlrefractiveindex"}:
        return float(microlens.get("refractiveIndex", 1.5))
    if key in {"microlensoffset", "mloffset", "offset"}:
        value_um = float(microlens.get("offset", 0.0))
        if unit is None or param_format(unit) in {"micron", "microns", "um"}:
            return value_um
        return value_um / 1e6 * _spatial_unit_scale(unit)
    if key in {"sourcefocallength", "sourceflength"}:
        source_focal_length_m = float(microlens.get("sourceFocalLength", DEFAULT_FOCAL_LENGTH_M))
        if unit is None:
            return source_focal_length_m
        return source_focal_length_m * _spatial_unit_scale(unit)
    if key in {"sourcefnumber", "sfnumber"}:
        return float(microlens.get("sourceFNumber", 4.0))
    if key == "sourcediameter":
        diameter_m = float(microlens.get("sourceFocalLength", DEFAULT_FOCAL_LENGTH_M)) / max(
            float(microlens.get("sourceFNumber", 4.0)),
            1e-30,
        )
        if unit is None:
            return diameter_m
        return diameter_m * _spatial_unit_scale(unit)
    if key == "sourceirradiance":
        value = microlens.get("sourceIrradiance")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"optimaloffset", "microoptimaloffsetpixel", "microoptimaloffset"}:
        if sensor_arg is None:
            raise ValueError("mlensGet(..., 'optimal offset') requires an explicit sensor in the Python backend.")
        unit_name = unit or "microns"
        cra = float(mlens_get(microlens, "chief ray angle radians"))
        z_stack = np.asarray(sensor_get(sensor_arg, "pixel layer thicknesses", unit_name), dtype=float).reshape(-1)
        n_stack = np.asarray(sensor_get(sensor_arg, "pixel refractive indices"), dtype=float).reshape(-1)
        working_n = n_stack[1:-1] if n_stack.size >= 3 else n_stack
        if working_n.size == 0:
            return 0.0
        if working_n.size < z_stack.size:
            working_n = np.pad(working_n, (0, z_stack.size - working_n.size), mode="edge")
        value = 0.0
        for thickness, refractive_index in zip(z_stack, working_n[: z_stack.size], strict=False):
            argument = np.clip(np.sin(cra) / max(float(refractive_index), 1e-30), -1.0, 1.0)
            value += float(thickness) * np.tan(np.arcsin(argument))
        return float(value)
    if key in {"optimaloffsets", "microoptimaloffsetarray", "microoptimaloffsets"}:
        if sensor_arg is None:
            raise ValueError("mlensGet(..., 'optimal offsets') requires an explicit sensor in the Python backend.")
        sensor_for_offsets = sensor_arg.clone()
        pixel_width_m = float(mlens_get(microlens, "diameter", "meters"))
        sensor_for_offsets = sensor_set(sensor_for_offsets, "pixel width", pixel_width_m)
        sensor_for_offsets = sensor_set(sensor_for_offsets, "pixel height", pixel_width_m)
        support_um = sensor_get(sensor_for_offsets, "spatial support", "um")
        x, y = np.meshgrid(np.asarray(support_um["x"], dtype=float), np.asarray(support_um["y"], dtype=float))
        source_focal_length_um = float(mlens_get(microlens, "source focal length", "microns"))
        cra = np.arctan(np.sqrt(np.square(x) + np.square(y)) / max(source_focal_length_um, 1e-12))
        ml_focal_length_um = float(mlens_get(microlens, "ml focal length", "microns"))
        refractive_index = float(mlens_get(microlens, "ml refractive index"))
        argument = np.clip(np.sin(cra) / max(refractive_index, 1e-30), -1.0, 1.0)
        return ml_focal_length_um * np.tan(np.arcsin(argument))
    if key in {"pixelirradiance", "irradiance", "pirradiance"}:
        value = microlens.get("pixelIrradiance")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"xcoordinate", "spacecoordinate"}:
        value = microlens.get("x")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key in {"anglecoordinate", "pcoordinate"}:
        value = microlens.get("p")
        return None if value is None else np.asarray(value, dtype=float).copy()
    if key == "etendue":
        return float(microlens.get("E", 0.0))
    if key == "pixeldistance":
        value_m = float(microlens.get("sourceFocalLength", DEFAULT_FOCAL_LENGTH_M)) * np.tan(
            float(mlens_get(microlens, "chief ray angle radians"))
        )
        if unit is None:
            return value_m
        return value_m * _spatial_unit_scale(unit)
    if key in {"pixelposition", "pixelrowcol"}:
        if sensor_arg is None:
            raise ValueError("mlensGet(..., 'pixel position') requires an explicit sensor in the Python backend.")
        distance_um = float(mlens_get(microlens, "pixel distance", "um"))
        pixel_size_um = float(sensor_get(sensor_arg, "pixel width", "um"))
        return {
            "hPix": int(_matlab_round_to_int(distance_um / max(pixel_size_um, 1e-30)).reshape(-1)[0]),
            "dPix": int(_matlab_round_to_int(distance_um / max(pixel_size_um * np.sqrt(2.0), 1e-30)).reshape(-1)[0]),
        }
    raise UnsupportedOptionError("mlensGet", parameter)


def ml_radiance(
    mlens: dict[str, Any] | None = None,
    sensor: Sensor | None = None,
    ml_flag: int | bool = 1,
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    if sensor is None:
        sensor = sensor_create(asset_store=asset_store)
    if mlens is None:
        mlens = mlens_create(sensor, asset_store=asset_store)
    microlens = _microlens_struct_from_value(mlens)

    source_f_number = float(mlens_get(microlens, "source fnumber"))
    ml_aperture_um = float(mlens_get(microlens, "ml diameter", "micron"))
    pixel_width_um = float(sensor_get(sensor, "pixel width", "micron"))
    pd_width_um = float(sensor_get(sensor, "pixel photodetector width", "micron"))
    d_stack_um = np.asarray(sensor_get(sensor, "pixel layer thicknesses", "micron"), dtype=float).reshape(-1)
    n_stack = np.asarray(sensor_get(sensor, "pixel refractive indices"), dtype=float).reshape(-1)

    n_source = 1.0
    width_ps_um = 2.0 * ml_aperture_um
    r_stack = 1.52
    focal_length_um = float(mlens_get(microlens, "ml focal length", "micron")) / max(r_stack, 1e-30)
    lens_offset_um = float(mlens_get(microlens, "offset"))
    lambda_um = float(mlens_get(microlens, "wavelength", "um"))
    chief_ray_angle_deg = float(mlens_get(microlens, "chief ray"))

    X, P = _ml_coordinates(-width_ps_um, width_ps_um, n_source, lambda_um)
    x = np.asarray(X[0, :], dtype=float)
    p = np.asarray(P[:, 0], dtype=float)

    source_na = n_source * np.sin(np.arctan(1.0 / max(2.0 * source_f_number, 1e-30)))
    W_source = _ml_source(
        -ml_aperture_um / 2.0,
        ml_aperture_um / 2.0,
        np.sin(np.deg2rad(chief_ray_angle_deg)) - source_na,
        np.sin(np.deg2rad(chief_ray_angle_deg)) + source_na,
        X,
        P,
    )

    if bool(ml_flag):
        W_lens = _ml_lens(focal_length_um, W_source, X, P)
        W_lens_offset = _ml_displacement(lens_offset_um, W_lens, X, P)
    else:
        W_lens_offset = W_source

    W_stack = np.asarray(W_lens_offset, dtype=float)
    for index, distance_um in enumerate(d_stack_um):
        if n_stack.size >= index + 2:
            refractive_index = float(n_stack[index + 1])
        elif n_stack.size > 0:
            refractive_index = float(n_stack[min(index, n_stack.size - 1)])
        else:
            refractive_index = 1.0
        W_stack = _ml_propagate(float(distance_um), refractive_index, W_stack, X, P)

    W_detector = W_stack
    projected = np.sum(W_detector, axis=0)
    projected_max = float(np.max(projected))
    if projected_max > 0.0:
        projected = projected / projected_max
    else:
        projected = np.zeros_like(projected, dtype=float)
    pixel_irradiance = projected[:, None] * projected[None, :]

    irradiance_in = np.sum(W_source, axis=0)
    etendue_in = float(np.sum(irradiance_in[np.abs(x) < (pixel_width_um / 2.0)]))
    irradiance_out = np.sum(W_detector, axis=0)
    etendue_out = float(np.sum(irradiance_out[np.abs(x) < (pd_width_um / 2.0)]))
    etendue = etendue_out / max(etendue_in, 1e-30)

    microlens["pixelIrradiance"] = np.asarray(pixel_irradiance, dtype=float)
    microlens["E"] = float(etendue)
    microlens["sourceIrradiance"] = np.asarray(W_source, dtype=float)
    microlens["x"] = x.copy()
    microlens["p"] = p.copy()
    return microlens


def ml_analyze_array_etendue(
    sensor: Sensor,
    method: str = "centered",
    n_angles: int = 5,
    *,
    asset_store: AssetStore | None = None,
) -> Sensor:
    microlens = sensor_get(sensor, "ml")
    if microlens is None:
        microlens = mlens_create(sensor, asset_store=asset_store)

    source_focal_length = float(mlens_get(microlens, "source focal length"))
    sensor_cra_deg = np.asarray(sensor_get(sensor, "cra degrees", source_focal_length), dtype=float)
    sampled_angles = np.linspace(0.0, float(np.max(sensor_cra_deg)), int(n_angles) + 1, dtype=float)
    sampled_etendue = np.zeros(sampled_angles.shape, dtype=float)
    normalized_method = param_format(method)

    for index, ray_angle_deg in enumerate(sampled_angles):
        working_microlens = mlens_set(microlens, "chief ray angle", float(ray_angle_deg))
        if normalized_method == "centered":
            working_microlens = mlens_set(working_microlens, "offset", 0.0)
            working_microlens = ml_radiance(working_microlens, sensor, 1)
        elif normalized_method in {"optimized", "optimal"}:
            optimal_offset_um = float(mlens_get(working_microlens, "optimal offset", sensor, "microns"))
            working_microlens = mlens_set(working_microlens, "offset", optimal_offset_um)
            working_microlens = ml_radiance(working_microlens, sensor, 1)
        elif normalized_method in {"nomicrolens", "bare"}:
            working_microlens = ml_radiance(working_microlens, sensor, 0)
        else:
            raise UnsupportedOptionError("mlAnalyzeArrayEtendue", method)
        sampled_etendue[index] = float(mlens_get(working_microlens, "etendue"))
        microlens = working_microlens

    interpolated = np.interp(sensor_cra_deg.reshape(-1), sampled_angles, sampled_etendue).reshape(sensor_cra_deg.shape)
    analyzed = sensor_set(sensor.clone(), "etendue", interpolated)
    analyzed = sensor_set(analyzed, "ml", microlens)
    return analyzed


def ml_get_current(sensor: Sensor | None = None, *, session: SessionContext | None = None) -> dict[str, Any] | None:
    """Return the microlens attached to the explicit or selected sensor."""

    current_sensor = sensor
    if current_sensor is None and session is not None:
        selected = session_get_selected(session, "sensor")
        if isinstance(selected, Sensor):
            current_sensor = selected
    if current_sensor is None:
        return None
    microlens = sensor_get(current_sensor, "micro lens")
    return None if microlens is None else _microlens_struct_from_value(microlens)


def ml_set_current(
    ml: dict[str, Any],
    sensor: Sensor | None = None,
    *,
    session: SessionContext | None = None,
) -> Sensor:
    """Store a microlens on the explicit or selected sensor."""

    if ml is None:
        raise ValueError("mlSetCurrent requires a microlens payload.")

    current_sensor = sensor
    if current_sensor is None and session is not None:
        selected = session_get_selected(session, "sensor")
        if isinstance(selected, Sensor):
            current_sensor = selected
    if current_sensor is None:
        raise ValueError("mlSetCurrent requires an explicit sensor or a session with a selected sensor.")

    updated = sensor_set(current_sensor.clone(), "micro lens", ml)
    if session is not None:
        return session_replace_object(session, updated)
    return updated


def ml_import_params(ml: dict[str, Any], optics: dict[str, Any], pixel: Any) -> dict[str, Any]:
    """Import optics and pixel parameters into a microlens payload."""

    from .optics import optics_get

    imported = _microlens_struct_from_value(ml)
    imported["sourceFNumber"] = float(optics_get(optics, "f number"))
    imported["sourceFocalLength"] = float(optics_get(optics, "focal length"))
    imported["focalLength"] = float(pixel_get(pixel, "pixel depth"))
    diameter = float(pixel_get(pixel, "pixel width"))
    imported["fnumber"] = imported["focalLength"] / max(diameter, 1e-30)
    return imported


def ml_description(
    ml: dict[str, Any],
    sensor: Sensor | None = None,
    *,
    session: SessionContext | None = None,
    asset_store: AssetStore | None = None,
) -> str:
    """Return the headless text summary used by the legacy microlens window."""

    current_sensor = sensor
    if current_sensor is None and session is not None:
        selected = session_get_selected(session, "sensor")
        if isinstance(selected, Sensor):
            current_sensor = selected
    if current_sensor is None:
        current_sensor = sensor_create(asset_store=asset_store)

    cra = float(mlens_get(ml, "chief ray angle radians"))
    source_focal_length_mm = float(mlens_get(ml, "source focal length", "mm"))
    pixel_width_um = float(sensor_get(current_sensor, "pixel width", "um"))
    diameter_um = float(mlens_get(ml, "diameter", "um"))
    distance_from_center_mm = source_focal_length_mm * np.tan(cra)
    horiz_pix = int(np.round(distance_from_center_mm / max(pixel_width_um, 1e-30)))
    diag_pix = int(np.round(distance_from_center_mm / max(pixel_width_um * np.sqrt(2.0), 1e-30)))

    lines = [
        f"  Pixel width (um) {pixel_width_um:.2f}",
        f"  ML diameter (um) {diameter_um:.2f} {'(uLens too big)' if diameter_um > pixel_width_um else ' '}",
        "",
        f"  Distance from center (mm) {distance_from_center_mm:.2f}",
        f"  horiz pix ({horiz_pix:d}), diag pix ({diag_pix:d})",
    ]

    etendue = float(mlens_get(ml, "etendue"))
    if etendue > 0.0:
        lines.append(f"  Etendue: {etendue:.3f}")

    optimal_offset = float(mlens_get(ml, "optimal offset", current_sensor, "microns"))
    lines.append(f"  Optimal offset = {optimal_offset:.2f} (um)")
    return "\n".join(lines)


def ml_print(
    ml: dict[str, Any] | None = None,
    sensor: Sensor | None = None,
    *,
    session: SessionContext | None = None,
    show: bool = False,
) -> str:
    """Return the headless MATLAB-style microlens print summary."""

    current = ml
    if current is None:
        current = ml_get_current(sensor, session=session)
    if current is None:
        raise ValueError("mlPrint requires a microlens payload or a current sensor microlens.")

    text = "\n".join(
        [
            "",
            "Microlens properties:",
            "--------------------",
            f"Focal length (um): {float(mlens_get(current, 'mlFocalLength', 'um')):.2f}",
            f"F-number:          {float(mlens_get(current, 'mlFnumber')):.2f}",
            f"Diameter (um):     {float(mlens_get(current, 'mlDiameter', 'um')):.2f}",
            f"Refractive index:  {float(mlens_get(current, 'mlRefractiveIndex')):.2f}",
            "",
        ]
    )
    if show:
        print(text)
    return text


mlGetCurrent = ml_get_current
mlSetCurrent = ml_set_current
mlImportParams = ml_import_params
mlDescription = ml_description
mlPrint = ml_print
mlensCreate = mlens_create
mlensSet = mlens_set
mlensGet = mlens_get
mlRadiance = ml_radiance
mlAnalyzeArrayEtendue = ml_analyze_array_etendue


def _scene_distance_m(scene_or_distance: Scene | OpticalImage | float | None) -> float:
    if scene_or_distance is None:
        return np.inf
    if isinstance(scene_or_distance, OpticalImage):
        return np.inf
    if isinstance(scene_or_distance, Scene):
        return float(scene_or_distance.fields.get("distance_m", np.inf))
    return float(scene_or_distance)


def _sensor_image_distance_m(
    scene_or_distance: Scene | OpticalImage | float | None,
    oi: OpticalImage | None,
) -> float:
    if oi is None:
        return DEFAULT_FOCAL_LENGTH_M

    optics = oi.fields.get("optics", {})
    focal_length = float(optics.get("focal_length_m", oi_get(oi, "focal length")))
    if param_format(optics.get("model", "")) == "skip":
        return focal_length

    scene_distance = _scene_distance_m(scene_or_distance)
    if not np.isfinite(scene_distance) or scene_distance <= focal_length:
        return focal_length
    return 1.0 / max((1.0 / focal_length) - (1.0 / scene_distance), 1e-12)


def _shot_noise_electrons(rng: np.random.Generator, electrons: np.ndarray) -> np.ndarray:
    electrons = np.asarray(electrons, dtype=float)
    clipped = np.clip(electrons, 0.0, None)
    noisy = clipped + (np.sqrt(clipped) * rng.standard_normal(clipped.shape))
    low_count = clipped < 25.0
    if np.any(low_count):
        noisy[low_count] = rng.poisson(clipped[low_count])
    return np.rint(noisy)


def _pixel_plane(volume: np.ndarray, values: np.ndarray) -> np.ndarray:
    plane = np.asarray(values, dtype=float)
    if np.asarray(volume).ndim == 2:
        return plane
    return plane[:, :, np.newaxis]


def _apply_read_noise(rng: np.random.Generator, volts: np.ndarray, sigma_v: float) -> np.ndarray:
    if sigma_v <= 0.0:
        return volts
    return volts + _pixel_plane(volts, rng.normal(0.0, sigma_v, size=volts.shape[:2]))


def _apply_fixed_pattern_noise(
    rng: np.random.Generator,
    volts: np.ndarray,
    *,
    dsnu_sigma_v: float,
    prnu_sigma: float,
    integration_time: float | np.ndarray,
    auto_exposure: bool,
) -> np.ndarray:
    dsnu = _pixel_plane(volts, rng.normal(0.0, dsnu_sigma_v, size=volts.shape[:2]))
    integration_time_array = np.asarray(integration_time, dtype=float)
    if np.all(np.isclose(integration_time_array, 0.0)) and not auto_exposure:
        return dsnu
    prnu = _pixel_plane(volts, 1.0 + rng.normal(0.0, prnu_sigma, size=volts.shape[:2]))
    return (volts * prnu) + dsnu


def _sample_centers(count: int, spacing_m: float) -> np.ndarray:
    return ((np.arange(count, dtype=float) + 0.5) - (count / 2.0)) * float(spacing_m)


def _sample2space(samples: np.ndarray, spacing_m: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    return (samples - np.mean(samples)) * float(spacing_m)


def _gaussian_kernel(shape: tuple[int, int], sigma: float) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 1 and cols <= 1:
        return np.ones((1, 1), dtype=float)
    y = np.arange(rows, dtype=float) - ((rows - 1.0) / 2.0)
    x = np.arange(cols, dtype=float) - ((cols - 1.0) / 2.0)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / max(2.0 * sigma * sigma, 1e-12))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.ones((1, 1), dtype=float)
    return kernel / kernel_sum


def _sensor_etendue(sensor: Sensor) -> np.ndarray:
    stored = sensor.fields.get("etendue")
    if stored is not None:
        etendue = np.asarray(stored, dtype=float)
        if etendue.shape == tuple(sensor.fields["size"]):
            return etendue

    vignetting = sensor.fields.get("vignetting", 0)
    normalized = param_format(vignetting) if isinstance(vignetting, str) else vignetting
    if normalized in {0, "skip", "off", "none", "disabled", "false", "", None}:
        etendue = np.ones(sensor.fields["size"], dtype=float)
        sensor.fields["etendue"] = etendue
        return etendue

    if normalized in {1, 2, 3, "bare", "nomicrolens", "centered", "optimal", "optimized"}:
        updated = sensor_vignetting(sensor, vignetting)
        etendue = np.asarray(updated.fields.get("etendue"), dtype=float)
        if etendue.shape == tuple(sensor.fields["size"]):
            sensor.fields["etendue"] = etendue.copy()
            sensor.fields["ml"] = _copy_metadata_value(updated.fields.get("ml"))
            return etendue

    raise UnsupportedOptionError("sensorCompute", f"vignetting {vignetting}")


def _pixel_pd_size_m(pixel: dict[str, Any]) -> np.ndarray:
    return _pixel_pd_size_from_pixel(pixel)


def _sensor_pd_array(sensor: Sensor, spacing: float) -> np.ndarray:
    if spacing <= 0.0 or spacing > 1.0:
        raise ValueError("spacing must be within (0, 1].")
    pixel = sensor.fields["pixel"]
    pixel_size = np.asarray(pixel["size_m"], dtype=float)
    pd_size = _pixel_pd_size_m(pixel)
    pd_position = _pixel_pd_position_from_pixel(pixel)

    normalized_pd_min = pd_position / (spacing * pixel_size)
    normalized_pd_max = (pd_size + pd_position) / (spacing * pixel_size)
    grid_positions = np.arange(0.0, 1.0 + spacing, spacing) / spacing
    n_squares = max(len(grid_positions) - 1, 1)
    in_pd_rows = np.zeros(n_squares, dtype=float)
    in_pd_cols = np.zeros(n_squares, dtype=float)
    for index in range(n_squares):
        lower = max(grid_positions[index], normalized_pd_min[0])
        upper = min(grid_positions[index + 1], normalized_pd_max[0])
        in_pd_rows[index] = max(0.0, upper - lower)

        lower = max(grid_positions[index], normalized_pd_min[1])
        upper = min(grid_positions[index + 1], normalized_pd_max[1])
        in_pd_cols[index] = max(0.0, upper - lower)
    return np.outer(in_pd_rows, in_pd_cols)


def _interpolated_cfa(sensor: Sensor, spacing: float, row_count: int, col_count: int) -> np.ndarray:
    pattern = tile_pattern(np.asarray(sensor.fields["pattern"], dtype=int), sensor.fields["size"][0], sensor.fields["size"][1])
    if np.isclose(spacing, 1.0):
        return pattern
    row_coords = np.floor(spacing * np.arange(row_count, dtype=float)).astype(int)
    col_coords = np.floor(spacing * np.arange(col_count, dtype=float)).astype(int)
    row_coords = np.clip(row_coords, 0, pattern.shape[0] - 1)
    col_coords = np.clip(col_coords, 0, pattern.shape[1] - 1)
    return pattern[row_coords[:, None], col_coords[None, :]]


def _interp2_linear_constant_zero(
    plane: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
) -> np.ndarray:
    if source_cols.size <= 1:
        col_coords = np.zeros_like(target_cols, dtype=float)
    else:
        col_coords = (np.asarray(target_cols, dtype=float) - float(source_cols[0])) / float(source_cols[1] - source_cols[0])
    if source_rows.size <= 1:
        row_coords = np.zeros_like(target_rows, dtype=float)
    else:
        row_coords = (np.asarray(target_rows, dtype=float) - float(source_rows[0])) / float(source_rows[1] - source_rows[0])
    # Match MATLAB's endpoint-inclusive behavior for coordinates that only miss
    # the valid range by floating-point roundoff.
    eps = 1e-9
    max_row = float(np.asarray(plane).shape[0] - 1)
    max_col = float(np.asarray(plane).shape[1] - 1)
    row_coords = np.where(np.abs(row_coords) <= eps, 0.0, row_coords)
    col_coords = np.where(np.abs(col_coords) <= eps, 0.0, col_coords)
    row_coords = np.where(np.abs(row_coords - max_row) <= eps, max_row, row_coords)
    col_coords = np.where(np.abs(col_coords - max_col) <= eps, max_col, col_coords)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    return map_coordinates(
        np.asarray(plane, dtype=float),
        [row_grid, col_grid],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _sensor_qe_on_wave(sensor: Sensor, target_wave: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    target_wave = np.asarray(target_wave, dtype=float).reshape(-1)
    sensor_wave = np.asarray(sensor.fields["wave"], dtype=float).reshape(-1)
    spectral_qe = _sensor_combined_qe(sensor, dtype=dtype)
    if spectral_qe.ndim == 1:
        spectral_qe = spectral_qe.reshape(-1, 1)
    if np.array_equal(target_wave, sensor_wave):
        return _sensor_apply_fdtd_qe_scale(sensor, target_wave, spectral_qe)
    if sensor_wave.size <= 1:
        raise ValueError("Sensor and optical image wavelength samplings do not match.")
    interpolated = np.empty((target_wave.size, spectral_qe.shape[1]), dtype=spectral_qe.dtype)
    for index in range(spectral_qe.shape[1]):
        interpolated[:, index] = np.interp(target_wave, sensor_wave, spectral_qe[:, index], left=0.0, right=0.0)
    return _sensor_apply_fdtd_qe_scale(sensor, target_wave, interpolated)


def _sensor_apply_fdtd_qe_scale(sensor: Sensor, wave: np.ndarray, spectral_qe: np.ndarray) -> np.ndarray:
    config = sensor.fields.get("fdtd_sensor")
    if not isinstance(config, dict) or not bool(config.get("enabled", True)):
        return spectral_qe
    scale = fdtd_sensor_qe_scale(config, wave).astype(spectral_qe.dtype, copy=False)
    if scale.size == 0:
        return spectral_qe
    return spectral_qe * scale.reshape(-1, 1)


def _sensor_apply_fdtd_optical_response(sensor: Sensor, values: np.ndarray) -> np.ndarray:
    config = sensor.fields.get("fdtd_sensor")
    if not isinstance(config, dict) or not bool(config.get("enabled", True)):
        return np.asarray(values)
    return fdtd_sensor_apply_optical_response(values, config)


def _sensor_apply_tcad_collection_response(sensor: Sensor, values: np.ndarray) -> np.ndarray:
    config = sensor.fields.get("tcad_sensor")
    if not isinstance(config, dict) or not bool(config.get("enabled", True)):
        return np.asarray(values)
    return tcad_sensor_apply_collection_response(values, config)


def _sensor_apply_physics_response(sensor: Sensor, values: np.ndarray) -> np.ndarray:
    optical = _sensor_apply_fdtd_optical_response(sensor, values)
    return _sensor_apply_tcad_collection_response(sensor, optical)


def _signal_current_density(oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    irradiance = np.asarray(oi.data["photons"], dtype=np.float32)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    spectral_qe = _sensor_qe_on_wave(sensor, wave, dtype=np.float32)
    bin_width = np.float32(np.mean(np.diff(wave)) if wave.size > 1 else 1.0)
    weighted_qe = spectral_qe * bin_width
    return np.tensordot(irradiance, weighted_qe, axes=([2], [0])).astype(np.float32) * np.float32(_ELEMENTARY_CHARGE_C)


def _spatial_integrate_current_density(scdi: np.ndarray, oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    n_samples_per_pixel = int(sensor.fields.get("n_samples_per_pixel", 1))
    if n_samples_per_pixel <= 0:
        raise ValueError("n_samples_per_pixel must be positive.")
    spacing = 1.0 / float(n_samples_per_pixel)
    if n_samples_per_pixel % 2 == 0:
        raise NotImplementedError("sensorCompute only supports odd nSamplesPerPixel values.")

    oi_rows, oi_cols = scdi.shape[:2]
    sensor_rows, sensor_cols = sensor.fields["size"]
    oi_height_spacing = float(oi_get(oi, "hspatialresolution"))
    oi_width_spacing = float(oi_get(oi, "wspatialresolution"))
    sensor_height_spacing = float(np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)[0])
    sensor_width_spacing = float(np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)[1])

    source_rows = _sample2space(np.arange(oi_rows, dtype=float), oi_height_spacing)
    source_cols = _sample2space(np.arange(oi_cols, dtype=float), oi_width_spacing)
    target_row_samples = np.arange(0.0, sensor_rows, spacing, dtype=float) + (spacing / 2.0)
    target_col_samples = np.arange(0.0, sensor_cols, spacing, dtype=float) + (spacing / 2.0)
    target_rows = _sample2space(target_row_samples, sensor_height_spacing)
    target_cols = _sample2space(target_col_samples, sensor_width_spacing)

    interpolated_cfa = _interpolated_cfa(sensor, spacing, target_rows.size, target_cols.size)

    height_samples_per_pixel = max(1, int(np.ceil(sensor_height_spacing / max(oi_height_spacing, 1e-12))))
    width_samples_per_pixel = max(1, int(np.ceil(sensor_width_spacing / max(oi_width_spacing, 1e-12))))
    kernel = _gaussian_kernel((height_samples_per_pixel, width_samples_per_pixel), height_samples_per_pixel / 4.0)

    flat_scdi = np.zeros((target_rows.size, target_cols.size), dtype=float)
    for channel_index in range(scdi.shape[2]):
        plane = convolve2d(np.asarray(scdi[:, :, channel_index], dtype=float), kernel, mode="same")
        sampled = _interp2_linear_constant_zero(plane, source_rows, source_cols, target_rows, target_cols)
        sampled = _sensor_apply_ocl_group_proxy(
            sensor, sampled, sample_scale=n_samples_per_pixel
        )
        mask = interpolated_cfa == (channel_index + 1)
        flat_scdi = flat_scdi + (mask * sampled)

    pixel_area = float(np.prod(np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)))
    if n_samples_per_pixel == 1:
        return flat_scdi * float(sensor.fields["pixel"]["fill_factor"]) * pixel_area

    pd_array = _sensor_pd_array(sensor, spacing)
    photo_detector_array = np.tile(pd_array, sensor.fields["size"])
    signal_current_large = flat_scdi * photo_detector_array
    filt = pixel_area * (np.ones((n_samples_per_pixel, n_samples_per_pixel), dtype=float) / float(n_samples_per_pixel**2))
    blurred = convolve2d(signal_current_large, filt, mode="same")
    start = n_samples_per_pixel // 2
    return blurred[start::n_samples_per_pixel, start::n_samples_per_pixel]


def signal_current(oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    """Compute signal current in Amps/pixel using the MATLAB signalCurrent path."""

    working = sensor.clone()
    if working.fields["mosaic"]:
        current_density = _signal_current_density(oi, working)
        signal = _spatial_integrate_current_density(current_density, oi, working)
        return np.asarray(_sensor_apply_physics_response(working, signal), dtype=float)

    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    delta_nm = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    pixel = working.fields["pixel"]
    pixel_area = float(np.prod(np.asarray(pixel["size_m"], dtype=float)) * float(pixel["fill_factor"]))
    filter_spectra = _sensor_qe_on_wave(working, wave)
    electron_rate_density = np.tensordot(cube * delta_nm, filter_spectra, axes=([2], [0]))
    electron_rate = _regrid_electron_rate_density(electron_rate_density, oi, working) * pixel_area
    electron_rate = _sensor_apply_ocl_group_proxy(working, electron_rate)
    electron_rate = _sensor_apply_physics_response(working, electron_rate)
    return np.asarray(electron_rate * _ELEMENTARY_CHARGE_C, dtype=float)


def signal_current_density(oi: OpticalImage, sensor: Sensor) -> np.ndarray:
    """Return the MATLAB SignalCurrentDensity current-density cube."""

    return np.asarray(_signal_current_density(oi, sensor.clone()), dtype=float)


def spatial_integration(
    scdi: np.ndarray,
    oi: OpticalImage,
    sensor: Sensor,
    grid_spacing: float = 1.0,
) -> np.ndarray:
    """Integrate a current-density cube using the default MATLAB spatialIntegration path."""

    if not np.isclose(float(grid_spacing), 1.0):
        raise UnsupportedOptionError("spatialIntegration", "gridSpacing != 1")
    return np.asarray(
        _spatial_integrate_current_density(np.asarray(scdi, dtype=float), oi, sensor.clone()),
        dtype=float,
    )


def regrid_oi_to_isa(
    scdi: np.ndarray,
    oi: OpticalImage,
    sensor: Sensor,
    spacing: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid current density from OI coordinates into sensor coordinates."""

    spacing_value = float(spacing)
    if spacing_value <= 0.0:
        raise ValueError("spacing must be positive.")

    density = np.asarray(scdi, dtype=float)
    oi_rows, oi_cols = density.shape[:2]
    oi_height_spacing = float(oi_get(oi, "hspatialresolution"))
    oi_width_spacing = float(oi_get(oi, "wspatialresolution"))
    source_rows = _sample2space(np.arange(oi_rows, dtype=float), oi_height_spacing)
    source_cols = _sample2space(np.arange(oi_cols, dtype=float), oi_width_spacing)

    sensor_rows = int(sensor_get(sensor, "rows"))
    sensor_cols = int(sensor_get(sensor, "cols"))
    target_row_samples = np.arange(0.0, sensor_rows, spacing_value, dtype=float) + (spacing_value / 2.0)
    target_col_samples = np.arange(0.0, sensor_cols, spacing_value, dtype=float) + (spacing_value / 2.0)
    sensor_height_spacing = float(sensor_get(sensor, "hres"))
    sensor_width_spacing = float(sensor_get(sensor, "wres"))
    new_rows = _sample2space(target_row_samples, sensor_height_spacing)
    new_cols = _sample2space(target_col_samples, sensor_width_spacing)

    interpolated_cfa = _interpolated_cfa(sensor, spacing_value, new_rows.size, new_cols.size)
    height_samples_per_pixel = max(1, int(np.ceil(sensor_height_spacing / max(oi_height_spacing, 1e-12))))
    width_samples_per_pixel = max(1, int(np.ceil(sensor_width_spacing / max(oi_width_spacing, 1e-12))))
    kernel = _gaussian_kernel((height_samples_per_pixel, width_samples_per_pixel), height_samples_per_pixel / 4.0)

    flat_scdi = np.zeros((new_rows.size, new_cols.size), dtype=float)
    n_filters = int(sensor_get(sensor, "nfilters"))
    for index in range(n_filters):
        plane = convolve2d(np.asarray(density[:, :, index], dtype=float), kernel, mode="same")
        sampled = _interp2_linear_constant_zero(plane, source_rows, source_cols, new_rows, new_cols)
        mask = interpolated_cfa == (index + 1)
        if sampled.ndim == 1:
            sampled = np.reshape(sampled, mask.shape)
        flat_scdi = flat_scdi + (mask * sampled)

    flat_scdi = np.nan_to_num(flat_scdi, nan=0.0)
    return np.asarray(flat_scdi, dtype=float), np.asarray(new_rows, dtype=float), np.asarray(new_cols, dtype=float)


def _regrid_electron_rate_density(
    density_cube: np.ndarray,
    oi: OpticalImage,
    sensor: Sensor,
) -> np.ndarray:
    oi_rows, oi_cols = density_cube.shape[:2]
    sensor_rows, sensor_cols = sensor.fields["size"]
    if oi_rows == 1 and oi_cols == 1:
        return np.broadcast_to(
            np.asarray(density_cube[0, 0, :], dtype=float),
            (sensor_rows, sensor_cols, density_cube.shape[2]),
        ).copy()
    pixel_size = np.asarray(sensor.fields["pixel"]["size_m"], dtype=float)
    oi_spacing = float(oi.fields.get("sample_spacing_m") or (oi.fields["width_m"] / max(oi_cols, 1)))

    oi_y = _sample_centers(oi_rows, oi_spacing)
    oi_x = _sample_centers(oi_cols, oi_spacing)
    sensor_y = _sample_centers(sensor_rows, pixel_size[0])
    sensor_x = _sample_centers(sensor_cols, pixel_size[1])
    sensor_yy, sensor_xx = np.meshgrid(sensor_y, sensor_x, indexing="ij")

    row_samples_per_pixel = max(1.0, float(np.ceil(pixel_size[0] / max(oi_spacing, 1e-12))))
    col_samples_per_pixel = max(1.0, float(np.ceil(pixel_size[1] / max(oi_spacing, 1e-12))))
    kernel_shape = (int(row_samples_per_pixel), int(col_samples_per_pixel))
    kernel_sigma = row_samples_per_pixel / 4.0

    regridded = np.empty((sensor_rows, sensor_cols, density_cube.shape[2]), dtype=float)
    for channel_index in range(density_cube.shape[2]):
        plane = density_cube[:, :, channel_index]
        if row_samples_per_pixel > 1.0 or col_samples_per_pixel > 1.0:
            plane = convolve2d(plane, _gaussian_kernel(kernel_shape, kernel_sigma), mode="same")
        interpolator = RegularGridInterpolator((oi_y, oi_x), plane, bounds_error=False, fill_value=0.0)
        regridded[:, :, channel_index] = interpolator(np.stack([sensor_yy, sensor_xx], axis=-1))
    return regridded


def plane2mosaic(img: np.ndarray, sensor: Sensor, empty_val: float = np.nan) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sensor mosaic plane into per-filter image planes."""

    image = np.asarray(img, dtype=float)
    _, cfa_numbers, _ = sensor_determine_cfa(sensor)
    filter_letters = str(sensor_get(sensor, "filter color letters"))
    n_planes = len(filter_letters)
    rows, cols = image.shape[:2]
    rgb_format = np.zeros((rows, cols, n_planes), dtype=float)

    if cfa_numbers.shape != image.shape:
        cfa_numbers = cfa_numbers[:rows, :cols]

    for index in range(n_planes):
        plane = np.full((rows, cols), float(empty_val), dtype=float)
        locations = cfa_numbers == (index + 1)
        plane[locations] = image[locations]
        rgb_format[:, :, index] = plane

    return rgb_format, np.unique(cfa_numbers)


def sensor_compute_mev(
    sensor: Sensor,
    oi: OpticalImage,
    *,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    """Compute a multi-exposure stack and combine it into a single response."""

    computed = sensor_compute(sensor, oi, seed=seed, session=session)
    volts = np.asarray(sensor_get(computed, "volts"), dtype=float)
    if volts.ndim != 3 or volts.shape[2] <= 1:
        return computed

    voltage_swing = float(sensor_get(computed, "pixel voltage swing"))
    exposure_times = np.asarray(sensor_get(computed, "exp time"), dtype=float).reshape(-1)
    n_exposures = int(sensor_get(computed, "n exposures"))
    max_time = float(np.max(exposure_times))

    combined = volts[:, :, -1].copy()
    threshold = voltage_swing * 0.95
    for exposure_index in range(n_exposures - 2, -1, -1):
        saturated = combined > threshold
        if not np.any(saturated):
            break
        scale_factor = max_time / max(float(exposure_times[exposure_index]), 1e-30)
        scaled = volts[:, :, exposure_index] * scale_factor
        combined[saturated] = scaled[saturated]
        threshold *= scale_factor

    updated = computed.clone()
    updated = sensor_set(updated, "pixel voltage swing", float(np.max(combined)) / 0.95 if np.max(combined) > 0 else voltage_swing)
    updated = sensor_set(updated, "volts", combined)
    updated = sensor_set(updated, "name", "combined")
    updated = sensor_set(updated, "exp time", max_time)
    if session is not None:
        return track_session_object(session, updated)
    return updated


def sensor_compute_sv_filters(
    sensor: Sensor,
    oi: OpticalImage,
    filter_file: Any,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a voltage image with space-varying IR filters."""

    from .optics import optics_get

    store = _store(asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    ir_filters, _, ir_filter_all_data = ie_read_color_filter(wave, filter_file, asset_store=store)
    optics = oi_get(oi, "optics")
    sensor_deg_map = np.asarray(sensor_get(sensor, "chiefRayAngleDegrees", optics_get(optics, "focal length")), dtype=float)

    field_heights = np.asarray(ir_filter_all_data.get("fHeight", []), dtype=float).reshape(-1)
    if field_heights.size == 0:
        raise ValueError("sensorComputeSVFilters requires fHeight entries in the filter payload.")
    keep = field_heights <= float(np.max(sensor_deg_map))
    field_heights = field_heights[keep]
    ir_filters = np.asarray(ir_filters, dtype=float)[:, keep]
    n_filters = field_heights.size
    if n_filters == 0:
        raise ValueError("sensorComputeSVFilters requires at least one in-range filter.")

    size = tuple(int(v) for v in np.asarray(sensor_get(sensor, "size"), dtype=int))
    voltage_images = np.zeros((size[0], size[1], n_filters), dtype=float)
    for index in range(n_filters):
        current_sensor = sensor_set(sensor.clone(), "irFilter", ir_filters[:, index])
        current_sensor = sensor_compute(current_sensor, oi, session=None)
        voltage_images[:, :, index] = np.asarray(sensor_get(current_sensor, "volts"), dtype=float)

    combined = np.zeros(size, dtype=float)
    for index in range(1, n_filters):
        this_band = (field_heights[index - 1] < sensor_deg_map) & (sensor_deg_map < field_heights[index])
        inner_distance = np.abs(sensor_deg_map - field_heights[index - 1])
        inner_weight = 1.0 - (inner_distance / max(field_heights[index] - field_heights[index - 1], 1e-30))
        inner_weight[~this_band] = 0.0
        weighted = (inner_weight * voltage_images[:, :, index - 1]) + ((1.0 - inner_weight) * voltage_images[:, :, index])
        combined[this_band] = weighted[this_band]

    return np.asarray(combined, dtype=float), np.asarray(voltage_images, dtype=float), np.asarray(ir_filters, dtype=float)


def _auto_exposure_luminance(
    sensor: Sensor,
    oi: OpticalImage,
    *,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, OpticalImage]:
    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    voltage_swing = float(sensor.fields["pixel"]["voltage_swing"])

    illuminance = luminance_from_photons(cube, wave, asset_store=AssetStore.default())
    bright_row, bright_col = np.unravel_index(int(np.argmax(illuminance)), illuminance.shape)
    # MATLAB oiExtractBright/oiCrop uses a [x y width-1 height-1] rect
    # convention, so the "1x1" bright patch in auto exposure is a 2x2 crop.
    row_start = min(bright_row, max(cube.shape[0] - 2, 0))
    col_start = min(bright_col, max(cube.shape[1] - 2, 0))
    row_stop = min(row_start + 2, cube.shape[0])
    col_stop = min(col_start + 2, cube.shape[1])

    small_oi = oi.clone()
    small_oi.data["photons"] = cube[row_start:row_stop, col_start:col_stop, :].copy()
    small_oi.fields["optics"] = dict(small_oi.fields["optics"])
    small_oi.fields["optics"]["model"] = "skip"
    small_oi.fields["optics"]["compute_method"] = "skip"
    small_oi.fields["optics"]["offaxis_method"] = "skip"

    pattern = np.asarray(sensor.fields["pattern"], dtype=int)
    small_sensor = sensor.clone()
    small_sensor.fields["size"] = (8 * pattern.shape[0], 8 * pattern.shape[1])
    small_sensor.fields["integration_time"] = 1.0
    small_sensor.fields["auto_exposure"] = False
    small_sensor.fields["pixel"] = dict(small_sensor.fields["pixel"])
    small_sensor.fields["pixel"]["voltage_swing"] = 1e6
    small_sensor.data.clear()

    sensor_hfov = float(sensor_get(small_sensor, "fov", None, oi))
    sensor_vfov = float(sensor_get(small_sensor, "vfov", None, oi))
    image_distance = float(oi.fields.get("image_distance_m", oi_get(oi, "focal length")))
    target_hfov = 2.0 * sensor_hfov
    target_vfov = 2.0 * sensor_vfov
    width_m = 2.0 * image_distance * np.tan(np.deg2rad(target_hfov) / 2.0)
    height_m = 2.0 * image_distance * np.tan(np.deg2rad(target_vfov) / 2.0)
    small_oi.fields["width_m"] = width_m
    small_oi.fields["height_m"] = height_m
    small_oi.fields["fov_deg"] = target_hfov
    small_oi.fields["vfov_deg"] = target_vfov
    small_oi.fields["rows"] = int(small_oi.data["photons"].shape[0])
    small_oi.fields["cols"] = int(small_oi.data["photons"].shape[1])
    small_oi.fields["sample_spacing_m"] = width_m / max(int(small_oi.data["photons"].shape[1]), 1)

    signal_sensor = sensor_compute(small_sensor, small_oi, seed=seed)
    signal_voltage = np.asarray(signal_sensor.data["volts"], dtype=float)
    max_signal_voltage = float(np.max(signal_voltage))
    integration_time = (float(level) * voltage_swing) / max(max_signal_voltage, 1e-12)
    return float(integration_time), max_signal_voltage, small_oi


def _auto_exposure_default(sensor: Sensor, oi: OpticalImage) -> float:
    integration_time, _, _ = _auto_exposure_luminance(sensor, oi, level=0.95, seed=0)
    return float(integration_time)


def _auto_exposure_weighted(
    oi: OpticalImage,
    sensor: Sensor,
    level: float,
    center_rect: Any | None = None,
    *,
    seed: int | None = None,
) -> tuple[float, float]:
    rect = None if _is_empty_dispatch_placeholder(center_rect) else center_rect
    if rect is not None and hasattr(rect, "Position"):
        rect = getattr(rect, "Position")
    center_oi = oi if rect is None else oi_crop(oi, rect)

    small_sensor = sensor_set_size_to_fov(sensor.clone(), float(oi_get(center_oi, "fov")), oi)
    small_sensor = sensor_set(small_sensor, "noise flag", 0)
    small_sensor = sensor_set(small_sensor, "integration time", 1)
    small_sensor = sensor_clear_data(small_sensor)

    voltage_swing = float(sensor_get(small_sensor, "pixel voltage swing"))
    signal_voltage, _, _ = sensor_compute_image(center_oi, small_sensor, seed=seed)
    max_signal_voltage = float(np.max(signal_voltage))
    integration_time = (float(level) * voltage_swing) / max(max_signal_voltage, 1e-12)
    return float(integration_time), max_signal_voltage


def auto_exposure(
    oi: OpticalImage,
    sensor: Sensor,
    level: Any | None = None,
    ae_method: Any | None = None,
    *args: Any,
    seed: int | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray | float, np.ndarray | float | None, OpticalImage | None]:
    """Mirror MATLAB autoExposure() for the supported headless exposure methods."""

    resolved_level = 0.95 if _is_empty_dispatch_placeholder(level) else float(np.asarray(level, dtype=float).reshape(-1)[0])
    resolved_method = "default" if _is_empty_dispatch_placeholder(ae_method) else str(param_format(str(ae_method)))
    options = {param_format(str(key)): value for key, value in kwargs.items() if value is not None}
    options.update(dict(_matlab_kv_pairs(args, function_name="autoExposure")))

    if resolved_method in {"default", "luminance"}:
        return _auto_exposure_luminance(sensor, oi, level=resolved_level, seed=0 if seed is None else int(seed))

    if resolved_method == "full":
        working = sensor_set(sensor.clone(), "integration time", 1)
        voltage_swing = float(sensor_get(working, "pixel voltage swing"))
        signal_voltage, _, _ = sensor_compute_image(oi, working, seed=seed)
        max_signal_voltage = float(np.max(signal_voltage))
        integration_time = (resolved_level * voltage_swing) / max(max_signal_voltage, 1e-12)
        return float(integration_time), max_signal_voltage, None

    if resolved_method == "mean":
        working = sensor_set(sensor.clone(), "integration time", 1)
        working = sensor_set(working, "analog gain", 1)
        voltage_swing = float(sensor_get(working, "pixel voltage swing"))
        signal_voltage, _, _ = sensor_compute_image(oi, working, seed=seed)
        mean_voltage = float(np.mean(signal_voltage))
        integration_time = (resolved_level * voltage_swing) / max(mean_voltage, 1e-12)
        max_signal_voltage = float(integration_time * float(np.max(signal_voltage)))
        return float(integration_time), max_signal_voltage, None

    if resolved_method == "cfa":
        working = sensor_set(sensor.clone(), "integration time", 1)
        voltage_swing = float(sensor_get(working, "pixel voltage swing"))
        signal_voltage, _, _ = sensor_compute_image(oi, working, seed=seed)
        pattern = np.asarray(sensor_get(working, "pattern"), dtype=int)
        n_rows, n_cols = pattern.shape
        integration_time = np.zeros((n_rows, n_cols), dtype=float)
        max_signal_voltage = np.zeros((n_rows, n_cols), dtype=float)
        for row in range(n_rows):
            for col in range(n_cols):
                current = np.asarray(signal_voltage[row::n_rows, col::n_cols], dtype=float)
                max_signal_voltage[row, col] = float(np.max(current))
                integration_time[row, col] = (resolved_level * voltage_swing) / max(max_signal_voltage[row, col], 1e-12)
        return integration_time, max_signal_voltage, None

    if resolved_method == "specular":
        working = sensor_set(sensor.clone(), "noise flag", 0)
        working = sensor_set(working, "integration time", 1)
        voltage_swing = float(sensor_get(working, "pixel voltage swing"))
        signal_voltage, _, _ = sensor_compute_image(oi, working, seed=seed)
        target_voltage = float(np.percentile(np.asarray(signal_voltage, dtype=float), resolved_level * 100.0))
        max_signal_voltage = float(np.max(signal_voltage))
        integration_time = voltage_swing / max(target_voltage, 1e-12)
        return float(integration_time), max_signal_voltage, None

    if resolved_method in {"weighted", "center"}:
        integration_time, _ = _auto_exposure_weighted(
            oi,
            sensor,
            resolved_level,
            options.get("centerrect"),
            seed=seed,
        )
        return integration_time, None, None

    if resolved_method == "video":
        video_max = options.get("videomax", 1.0 / 60.0)
        if _is_empty_dispatch_placeholder(video_max):
            video_max = 1.0 / 60.0
        weighted_time, _ = _auto_exposure_weighted(
            oi,
            sensor,
            resolved_level,
            options.get("centerrect"),
            seed=seed,
        )
        return min(float(np.asarray(video_max, dtype=float).reshape(-1)[0]), weighted_time), None, None

    raise UnsupportedOptionError("autoExposure", str(ae_method))


def _sensor_compute_method_is_binning(value: Any) -> bool:
    if isinstance(value, dict):
        return param_format(str(value.get("name", ""))) == "binning"
    return param_format(str(value)) == "binning"


def _sensor_compute_binning_method(value: Any) -> str | None:
    if isinstance(value, dict):
        method = value.get("method", value.get("b_method", value.get("bMethod")))
        return None if method is None else str(method)
    return None


def sensor_compute(
    sensor: Sensor | list[Sensor] | tuple[Sensor, ...],
    oi: OpticalImage,
    show_bar: bool | None = None,
    *,
    seed: int | None = None,
    session: SessionContext | None = None,
) -> Sensor | list[Sensor]:
    """Compute sensor response from an optical image or sensor array."""

    if isinstance(sensor, (list, tuple)):
        computed_sensors = [
            sensor_compute(
                item,
                oi,
                show_bar,
                seed=None if seed is None else int(seed) + index,
                session=session,
            )
            for index, item in enumerate(sensor)
        ]
        return computed_sensors

    del show_bar
    computed = sensor.clone()
    compute_method = computed.fields.get("sensor_compute_method")
    if _sensor_compute_method_is_binning(compute_method):
        return bin_sensor_compute(
            computed,
            oi,
            _sensor_compute_binning_method(compute_method),
            seed=seed,
        )
    cube = np.asarray(oi.data["photons"], dtype=float)
    wave = np.asarray(oi.fields["wave"], dtype=float)
    pixel = computed.fields["pixel"]
    delta_nm = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    pixel_area = float(np.prod(np.asarray(pixel["size_m"], dtype=float)) * pixel["fill_factor"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])

    integration_time_value = np.asarray(computed.fields["integration_time"], dtype=float)
    integration_time_matrix: np.ndarray | None = None
    if integration_time_value.ndim > 1:
        pattern = np.asarray(computed.fields["pattern"], dtype=int)
        if not computed.fields["mosaic"]:
            raise UnsupportedOptionError("sensorCompute", "matrix integration times on nonmosaic sensors")
        if integration_time_value.shape != pattern.shape:
            raise UnsupportedOptionError("sensorCompute", "matrix integration times with non-pattern shape")
        if computed.fields["auto_exposure"]:
            raise UnsupportedOptionError("sensorCompute", "auto exposure with matrix integration times")
        if np.any(integration_time_value <= 0.0):
            raise UnsupportedOptionError("sensorCompute", "nonpositive integration times")
        integration_time_matrix = np.asarray(integration_time_value, dtype=float).copy()
        computed.fields["integration_time"] = integration_time_matrix.copy()
        integration_time_array = np.empty(0, dtype=float)
    else:
        integration_time_array = integration_time_value.reshape(-1)

    if integration_time_matrix is None and integration_time_array.size == 0:
        integration_time_array = np.array([0.0], dtype=float)

    if integration_time_matrix is None and integration_time_array.size == 1 and (
        computed.fields["auto_exposure"] or float(integration_time_array[0]) <= 0.0
    ):
        computed.fields["integration_time"] = _auto_exposure_default(computed, oi)
        integration_time_array = np.array([float(computed.fields["integration_time"])], dtype=float)
    elif integration_time_matrix is None and integration_time_array.size > 1:
        if computed.fields["auto_exposure"]:
            raise UnsupportedOptionError("sensorCompute", "auto exposure with multiple integration times")
        if np.any(integration_time_array <= 0.0):
            raise UnsupportedOptionError("sensorCompute", "nonpositive integration times")
        computed.fields["integration_time"] = integration_time_array.copy()
    elif integration_time_matrix is None:
        computed.fields["integration_time"] = float(integration_time_array[0])

    seed_value = sensor.fields.get("noise_seed", 0) if seed is None else seed
    rng = np.random.default_rng(seed_value)
    noise_flag = int(computed.fields["noise_flag"])

    if computed.fields["mosaic"]:
        current_density = _signal_current_density(oi, computed)
        signal_current = _spatial_integrate_current_density(current_density, oi, computed)
        signal_current = _sensor_apply_physics_response(computed, signal_current)
        exposure_map_m: np.ndarray | None = None
        if integration_time_matrix is not None:
            pattern_rows, pattern_cols = integration_time_matrix.shape
            row_tiles = int(np.ceil(signal_current.shape[0] / pattern_rows))
            col_tiles = int(np.ceil(signal_current.shape[1] / pattern_cols))
            exposure_map_m = np.tile(integration_time_matrix, (row_tiles, col_tiles))[
                : signal_current.shape[0], : signal_current.shape[1]
            ]

        def _base_volts(current_integration_time: float) -> tuple[np.ndarray, None]:
            volts = signal_current * (current_integration_time * conversion_gain / _ELEMENTARY_CHARGE_C)
            return np.asarray(volts, dtype=float).copy(), None

        def _base_volts_matrix() -> tuple[np.ndarray, None]:
            assert exposure_map_m is not None
            volts = signal_current * (exposure_map_m * conversion_gain / _ELEMENTARY_CHARGE_C)
            return np.asarray(volts, dtype=float).copy(), None

    else:
        filter_spectra = _sensor_qe_on_wave(computed, wave)
        electron_rate_density = np.tensordot(cube * delta_nm, filter_spectra, axes=([2], [0]))
        electron_rate = _regrid_electron_rate_density(electron_rate_density, oi, computed) * pixel_area
        electron_rate = _sensor_apply_ocl_group_proxy(computed, electron_rate)
        electron_rate = _sensor_apply_physics_response(computed, electron_rate)

        def _base_volts(current_integration_time: float) -> tuple[np.ndarray, np.ndarray]:
            electrons = electron_rate * current_integration_time
            volts_full = electrons * conversion_gain
            volts_full = np.asarray(volts_full, dtype=float)
            return volts_full.copy(), volts_full.copy()

    etendue = _sensor_etendue(computed)
    analog_gain = float(computed.fields["analog_gain"])
    analog_offset = float(computed.fields["analog_offset"])
    voltage_swing = float(pixel["voltage_swing"])
    volts_captures: list[np.ndarray] = []
    channel_volts_captures: list[np.ndarray] = []
    dv_captures: list[np.ndarray] = []

    if integration_time_matrix is not None:
        volts, channel_volts = _base_volts_matrix()
        if volts.ndim == 2:
            volts = volts * etendue
        else:
            volts = volts * etendue[:, :, None]

        if noise_flag in {1, 2, -2}:
            if noise_flag == 2:
                volts = volts + (float(pixel["dark_voltage_v_per_sec"]) * exposure_map_m)
            volts = _shot_noise_electrons(rng, volts / max(conversion_gain, 1e-12)) * conversion_gain
            if noise_flag == 2:
                volts = _apply_read_noise(rng, volts, float(pixel["read_noise_v"]))
            if noise_flag in {1, 2}:
                volts = _apply_fixed_pattern_noise(
                    rng,
                    volts,
                    dsnu_sigma_v=float(pixel["dsnu_sigma_v"]),
                    prnu_sigma=float(pixel["prnu_sigma"]),
                    integration_time=exposure_map_m,
                    auto_exposure=bool(computed.fields["auto_exposure"]),
                )
        elif noise_flag not in {0, -1}:
            raise UnsupportedOptionError("sensorCompute", f"noise flag {noise_flag}")

        volts = _apply_sensor_response_type(computed, volts)
        if channel_volts is not None:
            channel_volts = _apply_sensor_response_type(computed, channel_volts)
        volts = np.clip((volts + analog_offset) / max(analog_gain, 1e-12), 0.0, voltage_swing)
        computed.data["volts"] = np.asarray(volts, dtype=float).copy()
        computed.data["channel_volts"] = None if channel_volts is None else np.asarray(channel_volts, dtype=float).copy()

        if param_format(computed.fields["quantization"]) != "analog":
            nbits = int(computed.fields["nbits"])
            max_digital = (2**nbits) - 1
            computed.data["dv"] = np.round(volts / voltage_swing * max_digital).astype(np.int32)
        return track_session_object(session, computed)

    for integration_time in integration_time_array:
        volts, channel_volts = _base_volts(float(integration_time))
        if volts.ndim == 2:
            volts = volts * etendue
        else:
            volts = volts * etendue[:, :, None]

        if noise_flag in {1, 2, -2}:
            if noise_flag == 2:
                volts = volts + (float(pixel["dark_voltage_v_per_sec"]) * float(integration_time))
            volts = _shot_noise_electrons(rng, volts / max(conversion_gain, 1e-12)) * conversion_gain
            if noise_flag == 2:
                volts = _apply_read_noise(rng, volts, float(pixel["read_noise_v"]))
            if noise_flag in {1, 2}:
                volts = _apply_fixed_pattern_noise(
                    rng,
                    volts,
                    dsnu_sigma_v=float(pixel["dsnu_sigma_v"]),
                    prnu_sigma=float(pixel["prnu_sigma"]),
                    integration_time=float(integration_time),
                    auto_exposure=bool(computed.fields["auto_exposure"]),
                )
        elif noise_flag not in {0, -1}:
            raise UnsupportedOptionError("sensorCompute", f"noise flag {noise_flag}")

        volts = _apply_sensor_response_type(computed, volts)
        if channel_volts is not None:
            channel_volts = _apply_sensor_response_type(computed, channel_volts)
        volts = np.clip((volts + analog_offset) / max(analog_gain, 1e-12), 0.0, voltage_swing)
        volts_captures.append(np.asarray(volts, dtype=float).copy())
        if channel_volts is not None:
            channel_volts_captures.append(np.asarray(channel_volts, dtype=float).copy())

        if param_format(computed.fields["quantization"]) != "analog":
            nbits = int(computed.fields["nbits"])
            max_digital = (2**nbits) - 1
            dv_captures.append(np.round(volts / voltage_swing * max_digital).astype(np.int32))

    if len(volts_captures) == 1:
        computed.data["volts"] = volts_captures[0]
        computed.data["channel_volts"] = channel_volts_captures[0] if channel_volts_captures else None
        if dv_captures:
            computed.data["dv"] = dv_captures[0]
    else:
        computed.data["volts"] = np.stack(volts_captures, axis=2)
        computed.data["channel_volts"] = (
            np.stack(channel_volts_captures, axis=3) if channel_volts_captures else None
        )
        if dv_captures:
            computed.data["dv"] = np.stack(dv_captures, axis=2)
        exposure_plane = int(np.rint(float(computed.fields.get("exposure_plane", np.floor(len(volts_captures) / 2.0) + 1))))
        computed.fields["exposure_plane"] = int(np.clip(exposure_plane, 1, len(volts_captures)))

    return track_session_object(session, computed)


def sensor_compute_samples(
    sensor_nf: Sensor,
    n_samp: int = 10,
    noise_flag: int = 2,
    show_bar: bool | None = None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Compute repeated noisy voltage captures from a precomputed noise-free sensor."""

    del show_bar
    if int(n_samp) <= 0:
        raise ValueError("sensor_compute_samples requires a positive sample count.")

    base_volts = sensor_nf.data.get("volts")
    if base_volts is None:
        raise ValueError("sensor_compute_samples requires a sensor with precomputed volts.")
    base = np.asarray(base_volts, dtype=float)
    if base.ndim not in {2, 3}:
        raise ValueError("sensor_compute_samples supports 2-D or 3-D volts data.")

    pixel = dict(sensor_nf.fields["pixel"])
    voltage_swing = float(pixel["voltage_swing"])
    conversion_gain = float(pixel["conversion_gain_v_per_electron"])
    integration_time = np.asarray(sensor_nf.fields.get("integration_time", 0.0), dtype=float)
    auto_exposure = bool(sensor_nf.fields.get("auto_exposure", False))
    base_seed = sensor_nf.fields.get("noise_seed", 0) if seed is None else seed
    samples: list[np.ndarray] = []

    for sample_index in range(int(n_samp)):
        sample = np.asarray(base, dtype=float).copy()
        rng_seed = None if base_seed is None else int(base_seed) + sample_index
        rng = np.random.default_rng(rng_seed)

        if int(noise_flag) in {1, 2, -2}:
            if int(noise_flag) == 2:
                sample = sample + (float(pixel["dark_voltage_v_per_sec"]) * integration_time)
            sample = _shot_noise_electrons(rng, sample / max(conversion_gain, 1e-12)) * conversion_gain
            if int(noise_flag) == 2:
                sample = _apply_read_noise(rng, sample, float(pixel["read_noise_v"]))
            if int(noise_flag) in {1, 2}:
                sample = _apply_fixed_pattern_noise(
                    rng,
                    sample,
                    dsnu_sigma_v=float(pixel["dsnu_sigma_v"]),
                    prnu_sigma=float(pixel["prnu_sigma"]),
                    integration_time=integration_time,
                    auto_exposure=auto_exposure,
                )
        elif int(noise_flag) not in {0, -1}:
            raise UnsupportedOptionError("sensorComputeSamples", f"noise flag {noise_flag}")

        samples.append(np.clip(sample, 0.0, voltage_swing))

    return np.stack(samples, axis=base.ndim)


def sensor_dr(sensor: Sensor, integration_time: Any = None) -> Any:
    """Return the MATLAB-style sensor dynamic range in dB."""

    return _sensor_dynamic_range(sensor, integration_time)


def sensor_ccm(
    sensor: Sensor,
    ccm_method: str | None = None,
    point_loc: Any | None = None,
    show_selection: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a MATLAB-style sensor color conversion matrix from stored chart corners."""

    store = _store(asset_store)
    method = param_format(ccm_method or "macbeth")
    if method != "macbeth":
        raise UnsupportedOptionError("sensorCCM", ccm_method)

    if point_loc is None or np.asarray(point_loc).size == 0:
        corner_points = sensor_get(sensor, "chart corner points")
    else:
        corner_points = np.asarray(point_loc)
    if corner_points is None or np.asarray(corner_points).size == 0:
        raise ValueError("sensorCCM requires chart corner points in headless mode.")

    rects, m_locs, p_size = _chart_rectangles(corner_points, 4, 6, 0.5)
    delta = _matlab_round_scalar(float(p_size[0]) * 0.5)
    rgb = np.asarray(_chart_rects_data(sensor, m_locs, delta, full_data=False, data_type="volts"), dtype=float)
    ideal_rgb = _macbeth_ideal_linear_rgb(sensor_get(sensor, "wave"), asset_store=store)
    matrix, _, _, _ = np.linalg.lstsq(rgb, ideal_rgb, rcond=None)

    sensor.fields.setdefault("chartP", {})
    sensor.fields["chartP"]["cornerPoints"] = np.asarray(corner_points).copy()
    if show_selection:
        sensor.fields["chartP"]["rects"] = rects.copy()

    return np.asarray(matrix, dtype=float), np.asarray(corner_points).copy()


def sensor_macbeth_daylight_estimate(
    sensor: Sensor,
    *args: Any,
    asset_store: AssetStore | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a daylight illuminant from Macbeth-chart sensor data."""

    store = _store(asset_store)
    options = dict(_matlab_kv_pairs(args, function_name="sensorMacbethDaylightEstimate"))
    for key, value in kwargs.items():
        if value is None:
            continue
        options[param_format(str(key)).replace("_", "")] = value

    corner_points = options.get("cornerpoints")
    if _is_empty_dispatch_placeholder(corner_points):
        corner_points = None
    if corner_points is None:
        corner_points = sensor_get(sensor, "chart corner points")
    if corner_points is None or np.asarray(corner_points).size == 0:
        raise ValueError("sensorMacbethDaylightEstimate requires chart corner points in headless mode.")

    patch_size_fraction = options.get("patchsizefraction", 0.4)
    if _is_empty_dispatch_placeholder(patch_size_fraction):
        patch_size_fraction = 0.4
    patch_size_fraction = float(np.asarray(patch_size_fraction, dtype=float).reshape(-1)[0])
    if not (0.0 < patch_size_fraction < 1.0):
        raise ValueError("sensorMacbethDaylightEstimate patch size fraction must be in (0, 1).")

    rects, m_locs, p_size = _chart_rectangles(corner_points, 4, 6, patch_size_fraction)
    delta = _matlab_round_scalar(patch_size_fraction * float(np.asarray(p_size, dtype=float).reshape(-1)[0]))
    camera_data = np.asarray(
        _chart_rects_data(sensor, m_locs, delta, full_data=False, data_type="electrons"),
        dtype=float,
    ).T
    camera_stacked = camera_data.reshape(-1, order="F")

    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    day_basis = energy_to_quanta(
        ie_read_spectra("cieDaylightBasis.mat", wave, asset_store=store),
        wave,
    )
    reflectance = ie_read_spectra("macbethChart", wave, asset_store=store)
    sensor_filters = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)

    x1 = sensor_filters.T @ (day_basis[:, [0]] * reflectance)
    x2 = sensor_filters.T @ (day_basis[:, [1]] * reflectance)
    x3 = sensor_filters.T @ (day_basis[:, [2]] * reflectance)
    design_matrix = np.column_stack(
        [
            x1.reshape(-1, order="F"),
            x2.reshape(-1, order="F"),
            x3.reshape(-1, order="F"),
        ]
    )

    normal_matrix = design_matrix.T @ design_matrix
    rhs = design_matrix.T @ camera_stacked
    weights = np.linalg.solve(normal_matrix, rhs)
    weights = weights / max(float(weights[0]), 1e-12)
    illuminant_photons = day_basis @ weights

    sensor.fields.setdefault("chartP", {})
    sensor.fields["chartP"]["cornerPoints"] = np.asarray(corner_points).copy()
    sensor.fields["chartP"]["rects"] = rects.copy()

    return (
        np.asarray(illuminant_photons, dtype=float).reshape(-1),
        np.asarray(corner_points, dtype=float).copy(),
        np.asarray(weights, dtype=float).reshape(-1),
        np.asarray(camera_stacked, dtype=float).reshape(-1),
        np.asarray(design_matrix, dtype=float),
    )


def sensor_description(sensor: Sensor, *args: Any, **kwargs: Any) -> Any:
    """Lazy sensorDescription() wrapper to avoid import cycles at module load time."""

    from .description import sensor_description as _sensor_description

    return _sensor_description(sensor, *args, **kwargs)


def sensor_dng_read(
    fname: str | Path,
    *args: Any,
    asset_store: Any | None = None,
) -> Any:
    """Lazy sensorDNGRead() wrapper to avoid import cycles at module load time."""

    from .fileio import sensor_dng_read as _sensor_dng_read

    return _sensor_dng_read(fname, *args, asset_store=asset_store)


sensorComputeSamples = sensor_compute_samples
autoExposure = auto_exposure
sensorComputeNoiseFree = sensor_compute_noise_free
sensorAddNoise = sensor_add_noise
binSensorCompute = bin_sensor_compute
binSensorComputeImage = bin_sensor_compute_image
sensorComputeImage = sensor_compute_image
sensorComputeFullArray = sensor_compute_full_array
sensorComputeMEV = sensor_compute_mev
sensorDR = sensor_dr
sensorCCM = sensor_ccm
sensorMacbethDaylightEstimate = sensor_macbeth_daylight_estimate
sensorComputeSVFilters = sensor_compute_sv_filters
binNoiseColumnFPN = bin_noise_column_fpn
binNoiseFPN = bin_noise_fpn
binNoiseRead = bin_noise_read
sensorJiggle = sensor_jiggle
sensorMPE30 = sensor_mpe30
sensorPDArray = sensor_pd_array
sensorWBCompute = sensor_wb_compute
analog2digital = analog_to_digital
noiseFPN = noise_fpn
noiseColumnFPN = noise_column_fpn
regridOI2ISA = regrid_oi_to_isa
plane2rgb = plane2mosaic
sensorClearData = sensor_clear_data
sensorColorOrder = sensor_color_order
colorBlockMatrix = color_block_matrix
sensorDetermineCFA = sensor_determine_cfa
sensorDisplayTransform = sensor_display_transform
sensorEquateTransmittances = sensor_equate_transmittances
sensorFilterRGB = sensor_filter_rgb
sensorGainOffset = sensor_gain_offset
sensorCheckArray = sensor_check_array
sensorImageColorArray = sensor_image_color_array
sensorNoNoise = sensor_no_noise
sensorRGB2Plane = sensor_rgb_to_plane
sensorResampleWave = sensor_resample_wave
sensorSNRluxsec = sensor_snr_luxsec
sensorStats = sensor_stats
sensorShowCFA = sensor_show_cfa
sensorShowCFAWeights = sensor_show_cfa_weights
sensorShowImage = sensor_show_image
sensorSaveImage = sensor_save_image
sensorCheckHuman = sensor_check_human
humanConeMosaic = human_cone_mosaic
sensorConePlot = sensor_cone_plot
sensorROI = sensor_roi
sensorVignetting = sensor_vignetting
sensorCreateIMX490 = sensor_create_imx490
sensorCreate = sensor_create
sensorCreateIdeal = sensor_create_ideal
sensorCreateConeMosaic = sensor_create_cone_mosaic
sensorHumanResize = sensor_human_resize
sensorCreateIMECSSM4x4vis = sensor_create_imec_ssm_4x4_vis
sensorIMX363V2 = sensor_imx363_v2
sensorInterleaved = sensor_interleaved
sensorLightField = sensor_light_field
sensorMT9V024 = sensor_mt9v024
sensorAddFilter = sensor_add_filter
sensorCfaSave = sensor_cfa_save
sensorCFANameList = sensor_cfa_name_list
sensorCompute = sensor_compute
sensorGet = sensor_get
sensorCrop = sensor_crop
sensorDeleteFilter = sensor_delete_filter
sensorDescription = sensor_description
sensorDNGRead = sensor_dng_read
sensorFromFile = sensor_from_file
sensorPixelCoord = sensor_pixel_coord
sensorReadColorFilters = sensor_read_color_filters
sensorReadFilter = sensor_read_filter
sensorReplaceFilter = sensor_replace_filter
sensorRescale = sensor_rescale
sensorSet = sensor_set
sensorSetSizeToFOV = sensor_set_size_to_fov
sensorComputeNoise = sensor_compute_noise
sensorSNR = sensor_snr
pixelCreate = pixel_create
pixelCenterFillPD = pixel_center_fill_pd
pixelDescription = pixel_description
pixelGet = pixel_get
pixelIdeal = pixel_ideal
pixelPositionPD = pixel_position_pd
pixelSet = pixel_set
pixelSR = pixel_sr
pixelSNR = pixel_snr
pixelSNRluxsec = pixel_snr_luxsec
pixelVperLuxSec = pixel_v_per_lux_sec
iePixelWellCapacity = ie_pixel_well_capacity
SignalCurrentDensity = signal_current_density
signalCurrent = signal_current
LoadRawSensorData = load_raw_sensor_data
binPixel = bin_pixel
binPixelPost = bin_pixel_post
pvFullOverlap = pv_full_overlap
pvReduction = pv_reduction
pixelDR = pixel_dr
pixelTransmittance = pixel_transmittance
ptInterfaceMatrix = pt_interface_matrix
ptPoyntingFactor = pt_poynting_factor
ptPropagationMatrix = pt_propagation_matrix
spatialIntegration = spatial_integration
ptReflectionAndTransmission = pt_reflection_and_transmission
ptScatteringMatrix = pt_scattering_matrix
ptSnellsLaw = pt_snells_law
ptTransmittance = pt_transmittance
