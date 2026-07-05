"""Image processing pipeline."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
import tempfile
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import convolve2d

from .assets import AssetStore, ie_read_spectra
from .color import internal_to_display_matrix, sensor_to_target_matrix, xyz_color_matching
from .display import Display, display_create, display_get, display_set
from .exceptions import UnsupportedOptionError
from .metrics import chromaticity_xy, xyz_from_energy
from .fileio import vc_load_object
from .optics import oi_compute, oi_create, oi_get
from .scene import scene_create
from .sensor import (
    ie_pixel_well_capacity,
    sensor_compute,
    sensor_create,
    sensor_determine_cfa,
    sensor_get,
    sensor_set,
)
from .session import track_ip_session_state, track_session_object
from .types import ImageProcessor, OpticalImage, Scene, Sensor, SessionContext
from .utils import (
    _normalize_legacy_kwargs,
    energy_to_quanta,
    image_linear_transform,
    invert_gamma_table,
    linear_to_srgb,
    param_format,
    rgb_to_xw_format,
    srgb_to_linear,
    split_prefixed_parameter,
    tile_pattern,
    xw_to_rgb_format,
    xyz_to_srgb,
)


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


def _copy_metadata_value(value: Any) -> Any:
    return copy.deepcopy(value)


def _l3_get(l3: Any, *names: str) -> Any:
    aliases = {param_format(name) for name in names}
    if hasattr(l3, "items"):
        for key, value in l3.items():
            if param_format(str(key)) in aliases:
                return value
        return None
    if hasattr(l3, "__dict__"):
        for key, value in vars(l3).items():
            if param_format(str(key)) in aliases:
                return value
        return None
    raise ValueError("L3 payload must be mapping-like.")


def _ip_chart_parameters(ip: ImageProcessor) -> dict[str, Any]:
    chart = ip.fields.get("chartP")
    if not isinstance(chart, dict):
        chart = {}
        ip.fields["chartP"] = chart
    return chart


def _identity_transform() -> np.ndarray:
    return np.eye(3, dtype=float)


def _as_channel_image(data: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(data, dtype=float)
    if array.ndim == 2:
        return array[:, :, np.newaxis], True
    return array, False


def _restore_channel_image(data: np.ndarray, squeeze_channel: bool) -> np.ndarray:
    if squeeze_channel and data.ndim == 3 and data.shape[2] == 1:
        return data[:, :, 0]
    return data


def _faulty_pixel_colors(list_array: np.ndarray) -> np.ndarray:
    x = np.asarray(list_array[:, 0], dtype=int)
    y = np.asarray(list_array[:, 1], dtype=int)
    colors = np.full(x.shape, 2, dtype=int)
    odd_x = (x % 2) == 1
    odd_y = (y % 2) == 1
    colors[(~odd_x) & odd_y] = 1
    colors[odd_x & (~odd_y)] = 3
    return colors


def _bayer_reflect(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    rows, cols = array.shape[:2]
    extended = np.concatenate((array[:, 2:4, :], array, array[:, (cols - 4) : (cols - 2), :]), axis=1)
    extended = np.concatenate((extended[2:4, :, :], extended, extended[(rows - 4) : (rows - 2), :, :]), axis=0)
    return extended


def _crop_border(img: np.ndarray, threshold: float = 0.06) -> np.ndarray:
    rgb = np.asarray(img, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    gray = np.mean(rgb, axis=2)
    mask = gray > float(threshold)
    if not np.any(mask):
        return rgb
    rows, cols = np.where(mask)
    row1, row2 = int(rows.min()), int(rows.max())
    col1, col2 = int(cols.min()), int(cols.max())
    if row2 <= row1 or col2 <= col1:
        return rgb
    return rgb[row1 : row2 + 1, col1 : col2 + 1, :]


def faulty_list(
    row: int,
    col: int,
    n_bad_pixels: int | None = None,
    min_separation: float = 2.0,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate MATLAB-style faulty-pixel `[x, y]` locations."""

    rows = int(row)
    cols = int(col)
    if rows <= 0 or cols <= 0:
        raise ValueError("faultyList requires positive row and col sizes.")
    count = int(round(rows * cols * 0.01)) if n_bad_pixels is None else int(n_bad_pixels)
    if count <= 0:
        return np.zeros((0, 2), dtype=int)
    if count * float(min_separation) * 4.0 > rows * cols:
        raise ValueError("Separation parameter and size are poorly chosen.")

    generator = np.random.default_rng() if rng is None else rng
    current = np.zeros((0, 2), dtype=int)
    while current.shape[0] != count:
        xlist = generator.integers(1, cols + 1, size=count, endpoint=False)
        ylist = generator.integers(1, rows + 1, size=count, endpoint=False)
        current = np.unique(np.column_stack((xlist, ylist)).astype(int), axis=0)

    if count == 1:
        return current

    min_distance = float(min_separation)
    index = 0
    while index < count:
        deltas = current.astype(float) - current[index].astype(float)
        distances = np.sort(np.sqrt(np.sum(deltas * deltas, axis=1)))
        if distances[1] < min_distance:
            while True:
                replacement = np.array(
                    [
                        generator.integers(1, cols + 1, endpoint=False),
                        generator.integers(1, rows + 1, endpoint=False),
                    ],
                    dtype=int,
                )
                current[index] = replacement
                current = np.unique(current, axis=0)
                if current.shape[0] == count:
                    break
            index = max(index - 1, 0)
            continue
        index += 1
    return current


def faulty_insert(list_array: Any, img: Any, val: Any = 0) -> np.ndarray:
    """Insert MATLAB-style faulty pixels into an RGB or plane-stack image."""

    faulty = np.asarray(list_array, dtype=int)
    image = np.asarray(img, dtype=float).copy()
    if faulty.ndim != 2 or faulty.shape[1] != 2:
        raise ValueError("faultyInsert requires an Nx2 faulty-pixel list.")
    if image.ndim != 3:
        raise ValueError("faultyInsert requires an image with shape rows x cols x channels.")
    fill = np.asarray(val, dtype=float)
    for x, y in faulty:
        image[int(y) - 1, int(x) - 1, :] = fill
    return image


def faulty_nearest_neighbor(list_array: Any, bayer_in: Any) -> np.ndarray:
    """Replace faulty Bayer samples using the legacy nearest-neighbor rule."""

    faulty = np.asarray(list_array, dtype=int)
    bayer = np.asarray(bayer_in, dtype=float)
    if faulty.ndim != 2 or faulty.shape[1] != 2:
        raise ValueError("FaultyNearestNeighbor requires an Nx2 faulty-pixel list.")
    if bayer.ndim != 3 or bayer.shape[2] < 3:
        raise ValueError("FaultyNearestNeighbor requires a Bayer plane stack with at least three channels.")

    extended = _bayer_reflect(bayer)
    colors = _faulty_pixel_colors(faulty)
    output = bayer.copy()
    shifted = faulty + 1
    for index, (x, y) in enumerate(shifted):
        channel = int(colors[index]) - 1
        if channel in {0, 2}:
            missing = extended[int(y) + 2, int(x), channel]
        else:
            missing = extended[int(y) + 1, int(x) + 1, channel]
        output[int(faulty[index, 1]) - 1, int(faulty[index, 0]) - 1, channel] = float(missing)
    return output


def faulty_bilinear(list_array: Any, bayer_in: Any) -> np.ndarray:
    """Replace faulty Bayer samples using the legacy bilinear rule."""

    faulty = np.asarray(list_array, dtype=int)
    bayer = np.asarray(bayer_in, dtype=float)
    if faulty.ndim != 2 or faulty.shape[1] != 2:
        raise ValueError("FaultyBilinear requires an Nx2 faulty-pixel list.")
    if bayer.ndim != 3 or bayer.shape[2] < 3:
        raise ValueError("FaultyBilinear requires a Bayer plane stack with at least three channels.")

    extended = _bayer_reflect(bayer)
    colors = _faulty_pixel_colors(faulty)
    output = bayer.copy()
    shifted = faulty + 1
    for index, (x, y) in enumerate(shifted):
        channel = int(colors[index]) - 1
        if channel in {0, 2}:
            missing = 0.25 * (
                extended[int(y) - 2, int(x), channel]
                + extended[int(y), int(x) - 2, channel]
                + extended[int(y) + 2, int(x), channel]
                + extended[int(y), int(x) + 2, channel]
            )
        else:
            missing = 0.25 * (
                extended[int(y) - 1, int(x) - 1, channel]
                + extended[int(y) - 1, int(x) + 1, channel]
                + extended[int(y) + 1, int(x) - 1, channel]
                + extended[int(y) + 1, int(x) + 1, channel]
            )
        output[int(faulty[index, 1]) - 1, int(faulty[index, 0]) - 1, channel] = float(missing)
    return output


def _ensure_ip_state(ip: ImageProcessor) -> ImageProcessor:
    wave = np.asarray(
        ip.fields.get("wave", np.arange(400.0, 701.0, 10.0, dtype=float)), dtype=float
    )
    ip.fields["wave"] = wave
    ip.fields.setdefault("spectrum", {"wave": wave.copy()})
    ip.fields["spectrum"]["wave"] = wave.copy()
    ip.fields.setdefault("display", display_create("default"))
    ip.fields.setdefault("transform_method", "adaptive")
    ip.fields.setdefault("internal_cs", "xyz")
    ip.fields.setdefault("conversion_method_sensor", "mcc optimized")
    ip.fields.setdefault("illuminant_correction_method", "none")
    ip.fields.setdefault("demosaic_method", "bilinear")
    ip.fields.setdefault("render", {"renderflag": 1, "scale": True, "whitept": False})
    ip.fields["render"].setdefault("renderflag", 1)
    ip.fields["render"].setdefault("scale", True)
    ip.fields["render"].setdefault("whitept", False)
    ip.fields.setdefault("demosaic", {"method": ip.fields["demosaic_method"]})
    ip.fields.setdefault("sensor_correction", {"method": ip.fields["conversion_method_sensor"]})
    ip.fields.setdefault(
        "illuminant_correction",
        {"method": ip.fields["illuminant_correction_method"]},
    )
    ip.fields["demosaic"]["method"] = ip.fields.get(
        "demosaic_method", ip.fields["demosaic"].get("method", "bilinear")
    )
    ip.fields["sensor_correction"]["method"] = ip.fields.get(
        "conversion_method_sensor",
        ip.fields["sensor_correction"].get("method", "mcc optimized"),
    )
    ip.fields["illuminant_correction"]["method"] = ip.fields.get(
        "illuminant_correction_method",
        ip.fields["illuminant_correction"].get("method", "none"),
    )
    transforms = list(ip.data.get("transforms", [None, None, None]))
    while len(transforms) < 3:
        transforms.append(None)
    ip.data["transforms"] = transforms[:3]
    return ip


def _ip_transform(ip: ImageProcessor, index: int) -> np.ndarray:
    _ensure_ip_state(ip)
    transform = ip.data["transforms"][index]
    if transform is None:
        return _identity_transform()
    return np.asarray(transform, dtype=float)


def ip_create(
    ip_name: str = "default",
    sensor: Sensor | None = None,
    display: Display | str | None = None,
    l3: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> ImageProcessor:
    """Create an image processor."""

    store = _store(asset_store)
    resolved_name = "default" if _is_empty_dispatch_placeholder(ip_name) else ip_name
    ip = ImageProcessor(name=str(resolved_name))
    if sensor is not None:
        ip.fields["wave"] = np.asarray(sensor.fields["wave"], dtype=float)
    else:
        ip.fields["wave"] = np.arange(400.0, 701.0, 10.0, dtype=float)
    ip.fields["spectrum"] = {"wave": np.asarray(ip.fields["wave"], dtype=float).copy()}
    if display is None:
        ip.fields["display"] = display_create(
            "lcdExample.mat", wave=ip.fields["wave"], asset_store=store, session=session
        )
    elif isinstance(display, str):
        ip.fields["display"] = display_create(
            display, wave=ip.fields["wave"], asset_store=store, session=session
        )
    else:
        ip.fields["display"] = track_session_object(session, display)
    ip.fields.update(
        {
            "transform_method": "adaptive",
            "demosaic_method": "bilinear",
            "illuminant_correction_method": "none",
            "internal_cs": "xyz",
            "conversion_method_sensor": "mcc optimized",
            "demosaic": {"method": "bilinear"},
            "sensor_correction": {"method": "mcc optimized"},
            "illuminant_correction": {"method": "none"},
            "render": {"renderflag": 1, "scale": True, "whitept": False},
        }
    )
    ip.data["input"] = None if sensor is None else sensor.data.get("dv", sensor.data.get("volts"))
    ip.fields["datamax"] = (
        None if sensor is None else float(sensor.fields["pixel"]["voltage_swing"])
    )
    if l3 is not None:
        ip.fields["l3"] = copy.deepcopy(l3)
    ip.data["transforms"] = [None, None, None]
    return track_ip_session_state(session, _ensure_ip_state(ip))


def vcimage_srgb(
    scene_name: str = "macbethD65",
    *,
    oi: OpticalImage | None = None,
    sensor: Sensor | None = None,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> ImageProcessor:
    """Create a MATLAB-style sRGB IP object for a scene using default OI and sensor settings."""

    store = _store(asset_store)
    scene = scene_create(scene_name, asset_store=store, session=session)
    working_oi = oi_create(asset_store=store, session=session) if oi is None else oi.clone()
    working_oi = oi_compute(working_oi, scene, session=session)

    working_sensor = sensor_create(asset_store=store, session=session) if sensor is None else sensor.clone()
    working_sensor = sensor_set(working_sensor, "size", [256, 256])
    working_sensor = sensor_set(working_sensor, "pixel size", np.array([3.0e-6, 3.0e-6], dtype=float))
    wave = np.asarray(sensor_get(working_sensor, "wave"), dtype=float).reshape(-1)
    working_sensor = sensor_set(
        working_sensor,
        "color filters",
        np.asarray(ie_read_spectra("XYZ", wave, asset_store=store), dtype=float),
    )
    working_sensor = sensor_set(working_sensor, "filter names", ["x", "y", "z"])
    working_sensor = sensor_compute(working_sensor, working_oi)

    ip = ip_create(asset_store=store, session=session)
    ip = ip_set(ip, "demosaicMethod", "Adaptive Laplacian", session=session)
    ip = ip_set(ip, "colorBalanceMethod", "Gray World", session=session)
    ip = ip_set(ip, "internalCS", "XYZ", session=session)
    ip = ip_set(ip, "colorconversionmethod", "MCC Optimized", session=session)
    return ip_compute(ip, working_sensor, asset_store=store, session=session)


def _ie_radiance_to_ip_oi(
    radiance: Scene | OpticalImage,
    pixel_size_um: float | None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> OpticalImage:
    store = _store(asset_store)
    normalized_type = param_format(getattr(radiance, "type", ""))
    if isinstance(radiance, OpticalImage) or normalized_type == "opticalimage":
        return radiance.clone()
    if isinstance(radiance, Scene) or normalized_type == "scene":
        scene = radiance.clone()
        wave = np.asarray(scene.fields["wave"], dtype=float).reshape(-1)
        oi = oi_create("pinhole", wave, asset_store=store, session=session)
        compute_kwargs: dict[str, Any] = {}
        if pixel_size_um is not None:
            compute_kwargs["pixel_size"] = float(pixel_size_um) * 1e-6
        return oi_compute(oi, scene, session=session, **compute_kwargs)
    raise ValueError("ieRadiance2IP requires a scene or optical image input.")


def _ie_radiance_to_ip_default_sensor(
    oi: OpticalImage,
    pixel_size_um: float | None,
    analog_gain: float,
    quantization: str,
    conversion_gain: float | None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Sensor:
    store = _store(asset_store)
    sensor = sensor_create(asset_store=store, session=session)
    effective_pixel_size_um = (
        float(pixel_size_um)
        if pixel_size_um is not None
        else float(oi_get(oi, "width spatial resolution")) * 1e6
    )
    well_capacity, _ = ie_pixel_well_capacity(effective_pixel_size_um, asset_store=store)
    sensor = sensor_set(sensor, "pixel read noise volts", 2.0e-3)
    sensor = sensor_set(sensor, "pixel voltage swing", 1.0)
    sensor = sensor_set(sensor, "pixel dark voltage", 2.0e-3)
    gain_value = float(conversion_gain) if conversion_gain is not None else 1.0 / max(float(well_capacity), 1.0e-12)
    sensor = sensor_set(sensor, "pixel conversion gain", gain_value)
    sensor = sensor_set(sensor, "quantization method", str(quantization))
    sensor = sensor_set(sensor, "analog gain", float(analog_gain))
    if np.isfinite(effective_pixel_size_um) and effective_pixel_size_um > 0.0:
        sensor = sensor_set(sensor, "pixel size same fill factor", effective_pixel_size_um * 1e-6)
    return sensor


def ie_radiance_to_ip(
    radiance: Scene | OpticalImage,
    *args: Any,
    sensor: Sensor | str | Path | None = None,
    pixel_size: float | None = None,
    film_diagonal: float = 5.0,
    etime: float | None = None,
    noise_flag: int = 2,
    conversion_gain: float | None = None,
    analog_gain: float = 1.0,
    quantization: str = "12 bit",
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[ImageProcessor | None, Sensor]:
    """Convert a scene or optical image into a headless IP/sensor pair."""

    del film_diagonal
    store = _store(asset_store)
    options = _normalize_legacy_kwargs(
        args,
        {
            "sensor": sensor,
            "pixel size": pixel_size,
            "etime": etime,
            "noise flag": noise_flag,
            "conversion gain": conversion_gain,
            "analog gain": analog_gain,
            "quantization": quantization,
        },
    )
    pixel_size_um = (
        None
        if options.get("pixelsize") is None
        else float(np.asarray(options["pixelsize"], dtype=float).reshape(-1)[0])
    )
    oi = _ie_radiance_to_ip_oi(radiance, pixel_size_um, asset_store=store, session=session)

    sensor_spec = options.get("sensor")
    if sensor_spec is None or (isinstance(sensor_spec, str) and sensor_spec == ""):
        working_sensor = _ie_radiance_to_ip_default_sensor(
            oi,
            pixel_size_um,
            float(options.get("analoggain", 1.0)),
            str(options.get("quantization", "12 bit")),
            None if options.get("conversiongain") is None else float(options["conversiongain"]),
            asset_store=store,
            session=session,
        )
    elif isinstance(sensor_spec, Sensor) or param_format(getattr(sensor_spec, "type", "")) == "sensor":
        working_sensor = sensor_spec.clone()
    else:
        loaded_sensor, _ = vc_load_object("sensor", sensor_spec, session=None)
        if not isinstance(loaded_sensor, Sensor):
            raise ValueError("ieRadiance2IP sensor input must resolve to a sensor object.")
        working_sensor = loaded_sensor

    working_sensor = sensor_set(working_sensor, "match oi", oi)
    if options.get("etime") is None:
        working_sensor = sensor_set(working_sensor, "auto exposure", True)
    else:
        working_sensor = sensor_set(working_sensor, "exp time", float(options["etime"]))
    working_sensor = sensor_set(working_sensor, "noise flag", int(options.get("noiseflag", 2)))
    working_sensor = sensor_compute(working_sensor, oi, session=session)

    filter_names = list(sensor_get(working_sensor, "filter names"))
    if len(filter_names) > 3:
        return None, working_sensor

    ip = ip_create(sensor=working_sensor, asset_store=store, session=session)
    ip = ip_set(ip, "conversion method sensor", "MCC Optimized", session=session)
    ip = ip_set(ip, "illuminant correction method", "gray world", session=session)
    ip = ip_set(ip, "demosaic method", "Adaptive Laplacian", session=session)
    ip = ip_compute(ip, working_sensor, asset_store=store, session=session)

    if working_sensor.metadata:
        ip.metadata = _copy_metadata_value(working_sensor.metadata)
    integration_time = np.asarray(sensor_get(working_sensor, "integration time"), dtype=float)
    ip.metadata["eT"] = (
        float(integration_time.reshape(-1)[0])
        if integration_time.size == 1
        else integration_time.copy()
    )
    return ip, working_sensor


def vcimage_iso_mtf(
    camera: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> ImageProcessor:
    """Legacy MATLAB wrapper that returns the slanted-edge ISO 12233 IP result."""

    from .camera import camera_create, camera_mtf

    working_camera = camera_create(asset_store=asset_store, session=session) if camera is None else camera
    result = camera_mtf(working_camera, asset_store=asset_store, session=session)
    return ip_set(result.vci.clone(), "name", "iso12233", session=session)


def vcimage_vsnr(
    ip: ImageProcessor,
    dpi: float | None = None,
    dist: float | None = None,
    rect: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[float, np.ndarray]:
    """Legacy MATLAB wrapper for VSNR on an image-processor ROI."""

    from .camera import _default_vsnr_rect, _ip_vsnr

    if dpi is None:
        dpi = float(ip_get(ip, "display dpi"))
    if dist is None:
        dist = float(ip_get(ip, "display viewing distance"))
    del dpi, dist

    rect_array = (
        _default_vsnr_rect(ip)
        if rect is None
        else np.asarray(rect, dtype=int).reshape(-1)
    )
    return float(_ip_vsnr(ip, rect_array, asset_store=_store(asset_store))), rect_array.copy()


def ip_mcc_xyz(
    ip: ImageProcessor,
    corner_points: Any | None = None,
    method: str = "sRGB",
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate Macbeth patch XYZ values and white point from an image processor."""

    from .camera import _chart_rectangles, _chart_rects_data, _linear_srgb_to_xyz, _whole_chart_corner_points

    result_size = np.asarray(ip_get(ip, "size"), dtype=int).reshape(-1)
    if result_size.size < 2:
        raise ValueError("ipMCCXYZ requires a computed IP image.")

    if corner_points is None or (
        isinstance(corner_points, str) and param_format(corner_points) == "wholechart"
    ):
        corners = _whole_chart_corner_points(int(result_size[0]), int(result_size[1]))
    else:
        corners = np.asarray(corner_points, dtype=float).reshape(4, 2)

    _, m_locs, p_size = _chart_rectangles(corners, 4, 6, 0.3)
    rgb_data = np.asarray(
        _chart_rects_data(ip, m_locs, float(np.asarray(p_size, dtype=float).reshape(-1)[0]), full_data=False, data_type="result"),
        dtype=float,
    )
    rgb_image = xw_to_rgb_format(rgb_data, 4, 6)

    normalized_method = param_format(method)
    if normalized_method == "srgb":
        macbeth_xyz = np.asarray(_linear_srgb_to_xyz(rgb_image), dtype=float)
    elif normalized_method == "custom":
        macbeth_xyz = np.asarray(image_rgb_to_xyz(ip, rgb_image, asset_store=asset_store), dtype=float)
    else:
        raise UnsupportedOptionError("ipMCCXYZ", method)

    macbeth_xyz_xw, _, _, _ = rgb_to_xw_format(macbeth_xyz)
    white_xyz = np.asarray(macbeth_xyz_xw[3, :], dtype=float).reshape(-1)
    return macbeth_xyz_xw, white_xyz, np.asarray(corners, dtype=float)


def vcimage_mcc_xyz(
    ip: ImageProcessor,
    corner_points: Any | None = None,
    method: str = "sRGB",
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy alias for `ipMCCXYZ`."""

    return ip_mcc_xyz(ip, corner_points, method, asset_store=asset_store)


def _is_empty_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) == 0
    return False


def lf_default_val(var: Any, default_val: Any) -> Any:
    """Return a default when the provided Python light-field value is empty."""

    return default_val if _is_empty_like(var) else var


def lf_default_field(parent_struct: dict[str, Any] | None, field_name: str, default_val: Any) -> dict[str, Any]:
    """Apply a default field value to a Python dictionary."""

    current = {} if parent_struct is None else dict(parent_struct)
    if field_name not in current:
        current[str(field_name)] = default_val
    return current


def lf_convert_to_float(lf: Any, precision: str = "single") -> np.ndarray:
    """Convert light-field data to floating-point, normalizing integer inputs."""

    array = np.asarray(lf)
    dtype_name = param_format(precision)
    if dtype_name in {"single", "float32"}:
        dtype = np.float32
    elif dtype_name in {"double", "float64"}:
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported LF precision: {precision}")

    converted = array.astype(dtype, copy=True)
    if np.issubdtype(array.dtype, np.integer):
        converted = converted / np.array(np.iinfo(array.dtype).max, dtype=dtype)
    return converted


def lf_buffer_to_image(lfbuffer: Any) -> np.ndarray:
    """Unshuffle a 5-D light-field buffer into a 2-D image."""

    buffer_array = np.asarray(lfbuffer, dtype=float)
    if buffer_array.ndim != 5:
        raise ValueError("LFbuffer2image requires a 5-D light-field buffer.")
    t_dim, s_dim, v_dim, u_dim, channels = buffer_array.shape
    image = np.zeros((t_dim * v_dim, s_dim * u_dim, channels), dtype=buffer_array.dtype)
    for t_index in range(t_dim):
        for s_index in range(s_dim):
            image[
                t_index * v_dim : (t_index + 1) * v_dim,
                s_index * u_dim : (s_index + 1) * u_dim,
                :,
            ] = buffer_array[t_index, s_index, :, :, :]
    return image


def lf_image_to_buffer(img: Any, ydim: int, xdim: int) -> np.ndarray:
    """Shuffle a 2-D image into the legacy light-field buffer layout."""

    image = np.asarray(img, dtype=float)
    if image.ndim != 3:
        raise ValueError("LFImage2buffer requires an image with shape rows x cols x channels.")
    y_res, x_res, channels = image.shape
    t_dim = int(xdim)
    s_dim = int(ydim)
    if t_dim <= 0 or s_dim <= 0:
        raise ValueError("LFImage2buffer requires positive microlens dimensions.")
    if y_res % t_dim != 0 or x_res % s_dim != 0:
        raise ValueError("LFImage2buffer requires image dimensions divisible by the microlens dimensions.")
    v_dim = y_res // t_dim
    u_dim = x_res // s_dim

    buffer_array = np.zeros((t_dim, s_dim, v_dim, u_dim, channels), dtype=image.dtype)
    for v_index in range(v_dim):
        for u_index in range(u_dim):
            buffer_array[:, :, v_index, u_index, :] = image[v_index::v_dim, u_index::u_dim, :]
    return buffer_array


def lf_buffer_to_sub_aperture_views(image4d: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lay out sub-aperture views from a 5-D light-field buffer into a 2-D array."""

    lightfield = np.asarray(image4d, dtype=float)
    if lightfield.ndim != 5:
        raise ValueError("LFbuffer2SubApertureViews requires a 5-D light-field buffer.")
    m_dim, n_dim, v_dim, u_dim, channels = lightfield.shape
    image2d = np.zeros((m_dim * v_dim, n_dim * u_dim, channels), dtype=lightfield.dtype)
    corners = np.zeros((u_dim, v_dim, 2), dtype=float)
    for u_index in range(u_dim):
        for v_index in range(v_dim):
            corners[u_index, v_index, :] = np.array([v_index * m_dim + 1, u_index * n_dim + 1], dtype=float)
            image2d[
                v_index * m_dim : (v_index + 1) * m_dim,
                u_index * n_dim : (u_index + 1) * n_dim,
                :,
            ] = lightfield[:, :, v_index, u_index, :]
    return image2d, corners


def lf_toolbox_version() -> str:
    """Return the vendored Light Field Toolbox version string."""

    return "v0.4 released 12-Feb-2015"


def ip_to_lightfield(ip: ImageProcessor, *args: Any, **kwargs: Any) -> np.ndarray:
    """Convert IP result data into the legacy light-field array layout."""

    pinholes = kwargs.pop("pinholes", kwargs.pop("nPinholes", None))
    colorspace = kwargs.pop("colorspace", "linear")
    if args:
        if pinholes is None:
            pinholes = args[0]
            args = args[1:]
        if args:
            colorspace = args[0]
            args = args[1:]
    if args or kwargs:
        raise TypeError("ip2lightfield accepts only `pinholes` and `colorspace`.")
    if pinholes is None:
        raise ValueError("ip2lightfield requires a `pinholes` vector.")

    rgb = np.asarray(ip_get(ip, "result"), dtype=float)
    if rgb.ndim != 3:
        raise ValueError("ip2lightfield requires image-processor result data with shape rows x cols x channels.")
    pinholes_array = np.asarray(pinholes, dtype=int).reshape(-1)
    if pinholes_array.size != 2 or np.any(pinholes_array <= 0):
        raise ValueError("ip2lightfield requires two positive pinhole counts.")

    color_mode = param_format(colorspace)
    if color_mode in {"linear", "lrgb"}:
        output_rgb = rgb
    elif color_mode == "srgb":
        output_rgb = linear_to_srgb(np.clip(rgb, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown color space {colorspace}")

    super_pixel_h = int(output_rgb.shape[0] // pinholes_array[0])
    super_pixel_w = int(output_rgb.shape[1] // pinholes_array[1])
    if super_pixel_h * int(pinholes_array[0]) != output_rgb.shape[0] or super_pixel_w * int(pinholes_array[1]) != output_rgb.shape[1]:
        raise ValueError("ip2lightfield requires image dimensions divisible by the pinhole counts.")

    lightfield = np.zeros(
        (super_pixel_h, super_pixel_w, int(pinholes_array[0]), int(pinholes_array[1]), output_rgb.shape[2]),
        dtype=output_rgb.dtype,
    )
    for i_index in range(int(pinholes_array[1])):
        for j_index in range(int(pinholes_array[0])):
            lightfield[:, :, j_index, i_index, :] = output_rgb[
                j_index * super_pixel_h : (j_index + 1) * super_pixel_h,
                i_index * super_pixel_w : (i_index + 1) * super_pixel_w,
                :,
            ]
    return lightfield


def _lf_interp_order(interp_method: str) -> int:
    normalized = param_format(interp_method)
    if normalized in {"nearest"}:
        return 0
    if normalized in {"linear"}:
        return 1
    if normalized in {"cubic", "spline"}:
        return 3
    raise UnsupportedOptionError("LFFiltShiftSum", interp_method)


def demosaic_rccc(mosaic_image: Any) -> np.ndarray:
    """Convert RCCC sensor planes into a monochrome image."""

    mosaic = np.asarray(mosaic_image, dtype=float)
    if mosaic.ndim != 3 or mosaic.shape[2] < 2:
        raise ValueError("demosaicRCCC requires an RCCC plane stack with at least two channels.")
    rows, cols = mosaic.shape[:2]
    if rows < 2 or cols < 2:
        raise ValueError("demosaicRCCC requires at least a 2x2 input image.")

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

    r_ex = mosaic_ex[:, :, 0]
    c_ex = mosaic_ex[:, :, 1]
    r_mask = (r_ex != 0).astype(float)
    r_conv = convolve2d(r_ex + c_ex, kernel, mode="same") * r_mask
    return c_ex[1:-1, 1:-1] + r_conv[1:-1, 1:-1]


def lf_filt_shift_sum(
    lf: Any,
    slope: float,
    filt_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Shift light-field slices to a common depth and flatten them."""

    options = lf_default_field(filt_options, "Precision", "single")
    options = lf_default_field(options, "Normalize", True)
    precision_name = str(options["Precision"])
    precision_array = np.array(0.0, dtype=lf_convert_to_float(np.array([0]), precision_name).dtype)
    options = lf_default_field(options, "MinWeight", 10.0 * float(np.finfo(precision_array.dtype).eps))
    options = lf_default_field(options, "Aspect4D", 1.0)
    options = lf_default_field(options, "FlattenMethod", "sum")
    options = lf_default_field(options, "InterpMethod", "linear")
    options = lf_default_field(options, "ExtrapVal", 0.0)

    aspect4d = np.asarray(options["Aspect4D"], dtype=float).reshape(-1)
    if aspect4d.size == 1:
        aspect4d = np.repeat(aspect4d, 4)
    if aspect4d.size != 4:
        raise ValueError("LFFiltShiftSum requires `Aspect4D` to be scalar or length four.")
    options["Aspect4D"] = aspect4d

    working_lf = lf_convert_to_float(lf, precision_name)
    if working_lf.ndim != 5:
        raise ValueError("LFFiltShiftSum requires a 5-D light field.")

    lf_size = working_lf.shape
    n_col_chans = lf_size[4]
    has_weight = n_col_chans in {2, 4}
    n_color_chans = n_col_chans - 1 if has_weight else n_col_chans
    normalize = bool(options["Normalize"])
    if normalize:
        if has_weight:
            working_lf[:, :, :, :, :n_color_chans] *= working_lf[:, :, :, :, -1:]
        else:
            weight = np.ones(working_lf.shape[:4] + (1,), dtype=working_lf.dtype)
            working_lf = np.concatenate((working_lf, weight), axis=4)

    lf_size = working_lf.shape
    t_dim, s_dim, v_dim, u_dim, _ = lf_size
    tv_slope = float(slope) * aspect4d[2] / aspect4d[0]
    su_slope = float(slope) * aspect4d[3] / aspect4d[1]
    v_vec = np.linspace(-0.5, 0.5, t_dim, dtype=float) * tv_slope * t_dim
    u_vec = np.linspace(-0.5, 0.5, s_dim, dtype=float) * su_slope * s_dim

    interp_order = _lf_interp_order(str(options["InterpMethod"]))
    extrap_val = float(options["ExtrapVal"])
    vv, uu = np.meshgrid(np.arange(v_dim, dtype=float), np.arange(u_dim, dtype=float), indexing="ij")
    shifted_lf = working_lf.copy()
    for t_index, v_offset in enumerate(v_vec):
        for s_index, u_offset in enumerate(u_vec):
            coords = np.array([vv + float(v_offset), uu + float(u_offset)], dtype=float)
            for channel_index in range(lf_size[4]):
                shifted_lf[t_index, s_index, :, :, channel_index] = map_coordinates(
                    shifted_lf[t_index, s_index, :, :, channel_index],
                    coords,
                    order=interp_order,
                    mode="constant",
                    cval=extrap_val,
                    prefilter=interp_order > 1,
                )

    flatten_method = param_format(str(options["FlattenMethod"]))
    if flatten_method == "sum":
        img_out = np.sum(shifted_lf, axis=(0, 1))
    elif flatten_method == "max":
        img_out = np.max(shifted_lf, axis=(0, 1))
    elif flatten_method == "median":
        img_out = np.median(shifted_lf.reshape(t_dim * s_dim, v_dim, u_dim, lf_size[4]), axis=0)
    else:
        raise UnsupportedOptionError("LFFiltShiftSum", options["FlattenMethod"])

    if normalize:
        weight_chan = np.asarray(img_out[:, :, -1], dtype=float)
        invalid = weight_chan < float(options["MinWeight"])
        safe_weight = np.where(invalid, 1.0, weight_chan)
        for channel_index in range(n_color_chans):
            img_out[:, :, channel_index] = np.asarray(img_out[:, :, channel_index], dtype=float) / safe_weight
            img_out[:, :, channel_index][invalid] = 0.0

    options["FilterInfo"] = {
        "mfilename": "LFFiltShiftSum",
        "time": datetime.now(UTC).strftime("%d%b%Y_%H%M%S"),
        "VersionStr": lf_toolbox_version(),
    }
    return np.asarray(img_out, dtype=shifted_lf.dtype), options, shifted_lf


def lf_autofocus(
    lightfield: Any,
    rect: Any | None = None,
    slope_range: Any | None = None,
    slope_step: float | None = None,
    fast_mode: bool | None = None,
) -> np.ndarray:
    """Focus a light-field ROI by maximizing the shifted-image variance."""

    lf = np.asarray(lightfield, dtype=float)
    if lf.ndim != 5 or lf.shape[4] < 3:
        raise ValueError("LFAutofocus requires a 5-D RGB light field.")

    if rect is None:
        round_rect = np.array([1, 1, lf.shape[3] - 1, lf.shape[2] - 1], dtype=int)
    else:
        round_rect = np.round(np.asarray(rect, dtype=float).reshape(-1)[:4]).astype(int)
    if round_rect.size != 4:
        raise ValueError("LFAutofocus requires a MATLAB-style `[x, y, width, height]` rect.")

    if slope_range is None:
        slope_bounds = np.array([-0.7, 0.5], dtype=float)
    else:
        slope_bounds = np.asarray(slope_range, dtype=float).reshape(-1)
    if slope_bounds.size != 2:
        raise ValueError("LFAutofocus requires `slope_range` to contain two values.")
    step = 0.05 if slope_step is None else float(slope_step)
    if step <= 0:
        raise ValueError("LFAutofocus requires a positive `slope_step`.")
    del fast_mode

    x0 = max(int(round_rect[0]) - 1, 0)
    y0 = max(int(round_rect[1]) - 1, 0)
    x1 = min(x0 + max(int(round_rect[2]), 0) + 1, lf.shape[3])
    y1 = min(y0 + max(int(round_rect[3]), 0) + 1, lf.shape[2])
    roi_lf = lf[:, :, y0:y1, x0:x1, :]

    slope_vec = np.arange(
        float(slope_bounds[0]),
        float(slope_bounds[1]) + 0.5 * step,
        step,
        dtype=float,
    )
    variance_vec = np.zeros_like(slope_vec)
    for index, slope_value in enumerate(slope_vec):
        shift_img, _, _ = lf_filt_shift_sum(roi_lf, float(slope_value))
        tmp_img = np.asarray(shift_img[:, :, :3], dtype=float)
        variance_vec[index] = float(np.var(tmp_img, axis=0).sum())

    best_slope = float(slope_vec[int(np.argmax(variance_vec))])
    shift_img, _, _ = lf_filt_shift_sum(lf, best_slope)
    return np.asarray(shift_img[:, :, :3], dtype=float)


def _ie_bilinear(planes: np.ndarray, cfa_pattern: np.ndarray) -> np.ndarray:
    rows, cols, nplanes = planes.shape
    extended = np.pad(planes, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    rgb = np.zeros((rows, cols, nplanes), dtype=float)
    for channel_index in range(nplanes):
        plane = extended[:, :, channel_index]
        mask = cfa_pattern == (channel_index + 1)
        if (mask[0, 0] and mask[-1, -1]) or (mask[0, -1] and mask[-1, 0]):
            kernel = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], dtype=float)
            rgb[:, :, channel_index] = convolve2d(plane, kernel, mode="valid")
        else:
            horizontal = convolve2d(plane, np.array([[0.5, 1.0, 0.5]], dtype=float), mode="valid")
            rgb[:, :, channel_index] = convolve2d(
                horizontal, np.array([[0.5], [1.0], [0.5]], dtype=float), mode="valid"
            )
    return rgb


def _bayer_pattern_name(cfa_pattern: np.ndarray) -> str | None:
    pattern = np.asarray(cfa_pattern, dtype=int)
    if pattern.shape != (2, 2):
        return None
    if np.array_equal(pattern, np.array([[2, 1], [3, 2]], dtype=int)):
        return "grbg"
    if np.array_equal(pattern, np.array([[1, 2], [2, 3]], dtype=int)):
        return "rggb"
    if np.array_equal(pattern, np.array([[2, 3], [1, 2]], dtype=int)):
        return "gbrg"
    if np.array_equal(pattern, np.array([[3, 2], [2, 1]], dtype=int)):
        return "bggr"
    return None


_BAYER_PATTERN_OFFSETS: dict[str, tuple[int, int]] = {
    "grbg": (0, 0),
    "rggb": (0, 1),
    "bggr": (1, 0),
    "gbrg": (1, 1),
}


def _shift_bayer_planes(data: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    rows, cols = map(int, data.shape[:2])
    row_index = np.clip(np.arange(rows, dtype=int) + int(row_shift), 0, rows - 1)
    col_index = np.clip(np.arange(cols, dtype=int) + int(col_shift), 0, cols - 1)
    return np.asarray(data[row_index[:, None], col_index[None, :], :], dtype=float)


def _mosaic_converter(
    bayer_in: np.ndarray, in_bayer_pattern: str, out_bayer_pattern: str = "grbg"
) -> tuple[np.ndarray, str]:
    bayer_in = np.asarray(bayer_in, dtype=float)
    in_pattern = param_format(in_bayer_pattern)
    out_pattern = param_format(out_bayer_pattern)
    if in_pattern == out_pattern:
        return bayer_in.copy(), out_pattern
    if in_pattern not in _BAYER_PATTERN_OFFSETS:
        raise ValueError(f"Unsupported Bayer RGB pattern: {in_bayer_pattern}")
    if out_pattern not in _BAYER_PATTERN_OFFSETS:
        raise ValueError(f"Unsupported Bayer RGB pattern: {out_bayer_pattern}")

    src_row, src_col = _BAYER_PATTERN_OFFSETS[in_pattern]
    dst_row, dst_col = _BAYER_PATTERN_OFFSETS[out_pattern]
    row_shift = src_row - dst_row
    col_shift = src_col - dst_col
    return _shift_bayer_planes(bayer_in, row_shift, col_shift), out_pattern


def _bayer_pattern_array(b_pattern: str) -> np.ndarray:
    pattern = param_format(b_pattern)
    mapping = {
        "grbg": np.array([[2, 1], [3, 2]], dtype=int),
        "rggb": np.array([[1, 2], [2, 3]], dtype=int),
        "gbrg": np.array([[2, 3], [1, 2]], dtype=int),
        "bggr": np.array([[3, 2], [2, 1]], dtype=int),
    }
    if pattern not in mapping:
        raise ValueError(f"Unsupported Bayer pattern: {b_pattern}")
    return mapping[pattern].copy()


def _resolve_bayer_pattern_name(b_pattern: str | np.ndarray) -> str:
    if isinstance(b_pattern, str):
        pattern = param_format(b_pattern)
    else:
        pattern = _bayer_pattern_name(np.asarray(b_pattern, dtype=int))
    if pattern not in _BAYER_PATTERN_OFFSETS:
        raise ValueError(f"Unsupported Bayer pattern: {b_pattern}")
    return str(pattern)


def _bayer_indices(b_pattern: str, size: tuple[int, int], clip: int = 0) -> tuple[np.ndarray, ...]:
    pattern = param_format(b_pattern)
    rows, cols = map(int, size)
    if pattern == "grbg":
        g1x = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        g1y = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        rx = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        ry = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        bx = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        by = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        g2x = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        g2y = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        return rx, ry, bx, by, g1x, g1y, g2x, g2y
    if pattern == "rggb":
        g1x = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        g1y = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        rx = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        ry = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        bx = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        by = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        g2x = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        g2y = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        return rx, ry, bx, by, g1x, g1y, g2x, g2y
    if pattern == "gbrg":
        g1x = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        g1y = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        rx = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        ry = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        bx = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        by = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        g2x = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        g2y = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        return rx, ry, bx, by, g1x, g1y, g2x, g2y
    if pattern == "bggr":
        g1x = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        g1y = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        rx = np.arange(2 + clip, cols - clip + 1, 2, dtype=int)
        ry = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        bx = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        by = np.arange(1 + clip, rows - clip + 1, 2, dtype=int)
        g2x = np.arange(1 + clip, cols - clip + 1, 2, dtype=int)
        g2y = np.arange(2 + clip, rows - clip + 1, 2, dtype=int)
        return rx, ry, bx, by, g1x, g1y, g2x, g2y
    raise ValueError(f"Unsupported Bayer pattern: {b_pattern}")


def _indexed_plane(
    data: np.ndarray, ys: np.ndarray, xs: np.ndarray, channel_index: int
) -> np.ndarray:
    return np.asarray(
        data[np.ix_(ys.astype(int) - 1, xs.astype(int) - 1, [int(channel_index)])], dtype=float
    )[:, :, 0]


def _assign_indexed_plane(
    data: np.ndarray, ys: np.ndarray, xs: np.ndarray, channel_index: int, values: np.ndarray
) -> None:
    data[np.ix_(ys.astype(int) - 1, xs.astype(int) - 1, [int(channel_index)])] = np.asarray(
        values, dtype=float
    )[:, :, None]


def _bayer_extend(data: np.ndarray) -> np.ndarray:
    rows, cols = data.shape[:2]
    extended = np.concatenate((data[:, 2:4, :], data, data[:, (cols - 4) : (cols - 2), :]), axis=1)
    extended = np.concatenate(
        (extended[2:4, :, :], extended, extended[(rows - 4) : (rows - 2), :, :]), axis=0
    )
    return np.asarray(extended, dtype=float)


def _bayer_laplacian_core(bayer_in: np.ndarray, *, adaptive: bool) -> np.ndarray:
    rows, cols = map(int, bayer_in.shape[:2])
    bayer_ex = _bayer_extend(bayer_in)
    rows_ex, cols_ex = rows + 4, cols + 4
    rgb = np.zeros_like(np.asarray(bayer_in, dtype=float), dtype=float)
    rgb[:, :, 0] = bayer_in[:, :, 0]
    rgb[:, :, 1] = bayer_in[:, :, 1]
    rgb[:, :, 2] = bayer_in[:, :, 2]

    rx, ry, bx, by, g1x, g1y, g2x, g2y = _bayer_indices("grbg", (rows_ex, cols_ex), 2)

    gs_h = _indexed_plane(bayer_ex, g1y, g1x, 1) + _indexed_plane(bayer_ex, g1y, g1x + 2, 1)
    gs_v = _indexed_plane(bayer_ex, g2y, g2x, 1) + _indexed_plane(bayer_ex, g2y - 2, g2x, 1)
    r_h = (
        2.0 * _indexed_plane(bayer_ex, ry, rx, 0)
        - _indexed_plane(bayer_ex, ry, rx - 2, 0)
        - _indexed_plane(bayer_ex, ry, rx + 2, 0)
    )
    r_v = (
        2.0 * _indexed_plane(bayer_ex, ry, rx, 0)
        - _indexed_plane(bayer_ex, ry - 2, rx, 0)
        - _indexed_plane(bayer_ex, ry + 2, rx, 0)
    )
    if adaptive:
        gd_h = _indexed_plane(bayer_ex, g1y, g1x, 1) - _indexed_plane(bayer_ex, g1y, g1x + 2, 1)
        gd_v = _indexed_plane(bayer_ex, g2y, g2x, 1) - _indexed_plane(bayer_ex, g2y - 2, g2x, 1)
        delta_h = np.abs(gd_h) + np.abs(r_h)
        delta_v = np.abs(gd_v) + np.abs(r_v)
        green_on_red = (
            (delta_h < delta_v) * (0.50 * gs_h + 0.25 * r_h)
            + (delta_h > delta_v) * (0.50 * gs_v + 0.25 * r_v)
            + (delta_h == delta_v) * (0.25 * (gs_h + gs_v) + 0.125 * (r_h + r_v))
        )
    else:
        green_on_red = 0.25 * (gs_h + gs_v) + 0.125 * (r_h + r_v)
    _assign_indexed_plane(rgb, ry - 2, rx - 2, 1, green_on_red)

    gs_h = _indexed_plane(bayer_ex, g2y, g2x, 1) + _indexed_plane(bayer_ex, g2y, g2x - 2, 1)
    gs_v = _indexed_plane(bayer_ex, g1y, g1x, 1) + _indexed_plane(bayer_ex, g1y + 2, g1x, 1)
    b_h = (
        2.0 * _indexed_plane(bayer_ex, by, bx, 2)
        - _indexed_plane(bayer_ex, by, bx - 2, 2)
        - _indexed_plane(bayer_ex, by, bx + 2, 2)
    )
    b_v = (
        2.0 * _indexed_plane(bayer_ex, by, bx, 2)
        - _indexed_plane(bayer_ex, by - 2, bx, 2)
        - _indexed_plane(bayer_ex, by + 2, bx, 2)
    )
    if adaptive:
        gd_h = _indexed_plane(bayer_ex, g2y, g2x, 1) - _indexed_plane(bayer_ex, g2y, g2x - 2, 1)
        gd_v = _indexed_plane(bayer_ex, g1y, g1x, 1) - _indexed_plane(bayer_ex, g1y + 2, g1x, 1)
        delta_h = np.abs(gd_h) + np.abs(b_h)
        delta_v = np.abs(gd_v) + np.abs(b_v)
        green_on_blue = (
            (delta_h < delta_v) * (0.50 * gs_h + 0.25 * b_h)
            + (delta_h > delta_v) * (0.50 * gs_v + 0.25 * b_v)
            + (delta_h == delta_v) * (0.25 * (gs_h + gs_v) + 0.125 * (b_h + b_v))
        )
    else:
        green_on_blue = 0.25 * (gs_h + gs_v) + 0.125 * (b_h + b_v)
    _assign_indexed_plane(rgb, by - 2, bx - 2, 1, green_on_blue)

    grn = np.concatenate((rgb[:, 2:4, 1], rgb[:, :, 1], rgb[:, (cols - 4) : (cols - 2), 1]), axis=1)
    grn = np.concatenate((grn[2:4, :], grn, grn[(rows - 4) : (rows - 2), :]), axis=0)
    bayer_ex[:, :, 1] = grn

    red_on_g1 = 0.5 * (
        _indexed_plane(bayer_ex, ry, rx, 0) + _indexed_plane(bayer_ex, ry, rx - 2, 0)
    ) + 0.25 * (
        2.0 * _indexed_plane(bayer_ex, g1y, g1x, 1)
        - _indexed_plane(bayer_ex, ry, rx - 2, 1)
        - _indexed_plane(bayer_ex, ry, rx, 1)
    )
    _assign_indexed_plane(rgb, g1y - 2, g1x - 2, 0, red_on_g1)
    red_on_g2 = 0.5 * (
        _indexed_plane(bayer_ex, ry, rx, 0) + _indexed_plane(bayer_ex, ry + 2, rx, 0)
    ) + 0.25 * (
        2.0 * _indexed_plane(bayer_ex, g2y, g2x, 1)
        - _indexed_plane(bayer_ex, ry + 2, rx, 1)
        - _indexed_plane(bayer_ex, ry, rx, 1)
    )
    _assign_indexed_plane(rgb, g2y - 2, g2x - 2, 0, red_on_g2)

    blue_on_g2 = 0.5 * (
        _indexed_plane(bayer_ex, by, bx, 2) + _indexed_plane(bayer_ex, by, bx + 2, 2)
    ) + 0.25 * (
        2.0 * _indexed_plane(bayer_ex, g2y, g2x, 1)
        - _indexed_plane(bayer_ex, by, bx + 2, 1)
        - _indexed_plane(bayer_ex, by, bx, 1)
    )
    _assign_indexed_plane(rgb, g2y - 2, g2x - 2, 2, blue_on_g2)
    blue_on_g1 = 0.5 * (
        _indexed_plane(bayer_ex, by, bx, 2) + _indexed_plane(bayer_ex, by - 2, bx, 2)
    ) + 0.25 * (
        2.0 * _indexed_plane(bayer_ex, g1y, g1x, 1)
        - _indexed_plane(bayer_ex, by - 2, bx, 1)
        - _indexed_plane(bayer_ex, by, bx, 1)
    )
    _assign_indexed_plane(rgb, g1y - 2, g1x - 2, 2, blue_on_g1)

    rs_n = _indexed_plane(bayer_ex, ry, rx - 2, 0) + _indexed_plane(bayer_ex, ry + 2, rx, 0)
    rs_p = _indexed_plane(bayer_ex, ry, rx, 0) + _indexed_plane(bayer_ex, ry + 2, rx - 2, 0)
    g_n = (
        2.0 * _indexed_plane(bayer_ex, by, bx, 1)
        - _indexed_plane(bayer_ex, ry, rx - 2, 1)
        - _indexed_plane(bayer_ex, ry + 2, rx, 1)
    )
    g_p = (
        2.0 * _indexed_plane(bayer_ex, by, bx, 1)
        - _indexed_plane(bayer_ex, ry, rx, 1)
        - _indexed_plane(bayer_ex, ry + 2, rx - 2, 1)
    )
    if adaptive:
        rd_n = _indexed_plane(bayer_ex, ry, rx - 2, 0) - _indexed_plane(bayer_ex, ry + 2, rx, 0)
        rd_p = _indexed_plane(bayer_ex, ry, rx, 0) - _indexed_plane(bayer_ex, ry + 2, rx - 2, 0)
        delta_n = np.abs(rd_n) + np.abs(g_n)
        delta_p = np.abs(rd_p) + np.abs(g_p)
        red_on_blue = (
            (delta_n < delta_p) * (0.50 * rs_n + 0.25 * g_n)
            + (delta_n > delta_p) * (0.50 * rs_p + 0.25 * g_p)
            + (delta_n == delta_p) * (0.25 * (rs_n + rs_p) + 0.125 * (g_n + g_p))
        )
    else:
        red_on_blue = 0.25 * (rs_n + rs_p) + 0.125 * (g_n + g_p)
    _assign_indexed_plane(rgb, by - 2, bx - 2, 0, red_on_blue)

    bs_n = _indexed_plane(bayer_ex, by - 2, bx, 2) + _indexed_plane(bayer_ex, by, bx + 2, 2)
    bs_p = _indexed_plane(bayer_ex, by, bx, 2) + _indexed_plane(bayer_ex, by - 2, bx + 2, 2)
    g_n = (
        2.0 * _indexed_plane(bayer_ex, ry, rx, 1)
        - _indexed_plane(bayer_ex, by - 2, bx, 1)
        - _indexed_plane(bayer_ex, by, bx + 2, 1)
    )
    g_p = (
        2.0 * _indexed_plane(bayer_ex, ry, rx, 1)
        - _indexed_plane(bayer_ex, by, bx, 1)
        - _indexed_plane(bayer_ex, by - 2, bx + 2, 1)
    )
    if adaptive:
        bd_n = _indexed_plane(bayer_ex, by - 2, bx, 2) - _indexed_plane(bayer_ex, by, bx + 2, 2)
        bd_p = _indexed_plane(bayer_ex, by, bx, 2) - _indexed_plane(bayer_ex, by - 2, bx + 2, 2)
        delta_n = np.abs(bd_n) + np.abs(g_n)
        delta_p = np.abs(bd_p) + np.abs(g_p)
        blue_on_red = (
            (delta_n < delta_p) * (0.50 * bs_n + 0.25 * g_n)
            + (delta_n > delta_p) * (0.50 * bs_p + 0.25 * g_p)
            + (delta_n == delta_p) * (0.25 * (bs_n + bs_p) + 0.125 * (g_n + g_p))
        )
    else:
        blue_on_red = 0.25 * (bs_n + bs_p) + 0.125 * (g_n + g_p)
    _assign_indexed_plane(rgb, ry - 2, rx - 2, 2, blue_on_red)

    return rgb


def _laplacian_demosaic(planes: np.ndarray, b_pattern: str) -> np.ndarray:
    converted, _ = _mosaic_converter(planes, b_pattern, "grbg")
    demosaiced = _bayer_laplacian_core(converted, adaptive=False)
    restored, _ = _mosaic_converter(demosaiced, "grbg", b_pattern)
    return restored


def _adaptive_laplacian_demosaic(planes: np.ndarray, b_pattern: str) -> np.ndarray:
    converted, _ = _mosaic_converter(planes, b_pattern, "grbg")
    demosaiced = _bayer_laplacian_core(converted, adaptive=True)
    restored, _ = _mosaic_converter(demosaiced, "grbg", b_pattern)
    return restored


def _nearest_neighbor_demosaic(planes: np.ndarray, b_pattern: str) -> np.ndarray:
    rows, cols = map(int, planes.shape[:2])
    rgb = np.asarray(planes, dtype=float).copy()
    rx, ry, bx, by, g1x, g1y, g2x, g2y = _bayer_indices(b_pattern, (rows, cols))
    red = _indexed_plane(planes, ry, rx, 0)
    green1 = _indexed_plane(planes, g1y, g1x, 1)
    green2 = _indexed_plane(planes, g2y, g2x, 1)
    blue = _indexed_plane(planes, by, bx, 2)

    dy = 1 if int(ry[0]) % 2 == 1 else -1
    dx = 1 if int(rx[0]) % 2 == 1 else -1
    _assign_indexed_plane(rgb, ry, rx + dx, 0, red)
    _assign_indexed_plane(rgb, ry + dy, rx, 0, red)
    _assign_indexed_plane(rgb, ry + dy, rx + dx, 0, red)

    dx = 1 if int(g1x[0]) % 2 == 1 else -1
    _assign_indexed_plane(rgb, g1y, g1x + dx, 1, green1)
    _assign_indexed_plane(rgb, g2y, g2x - dx, 1, green2)

    dy = 1 if int(by[0]) % 2 == 1 else -1
    dx = 1 if int(bx[0]) % 2 == 1 else -1
    _assign_indexed_plane(rgb, by, bx + dx, 2, blue)
    _assign_indexed_plane(rgb, by + dy, bx, 2, blue)
    _assign_indexed_plane(rgb, by + dy, bx + dx, 2, blue)
    return rgb


def _demosaic_rgb_planes(
    planes: np.ndarray, cfa_pattern: np.ndarray, demosaic_method: str
) -> np.ndarray:
    method = param_format(demosaic_method)
    pattern_name = _bayer_pattern_name(cfa_pattern)
    if method in {"nearestneighbor"} and pattern_name is not None:
        return _nearest_neighbor_demosaic(planes, pattern_name)
    if method in {"laplacian"} and pattern_name is not None:
        return _laplacian_demosaic(planes, pattern_name)
    if method in {"adaptivelaplacian"} and pattern_name in {"grbg", "rggb", "bggr"}:
        return _adaptive_laplacian_demosaic(planes, pattern_name)
    return _ie_bilinear(planes, cfa_pattern)


def _sensor_space_from_data(
    sensor: Sensor, sensor_data: np.ndarray, demosaic_method: str = "bilinear"
) -> np.ndarray:
    sensor_data = np.asarray(sensor_data, dtype=float)
    if sensor_data.ndim == 3 and not sensor.fields["mosaic"]:
        return sensor_data
    if sensor_data.ndim == 3:
        return sensor_data
    if sensor.fields["mosaic"]:
        pattern = np.asarray(sensor.fields["pattern"], dtype=int)
        rows, cols = sensor_data.shape
        nfilters = int(np.asarray(sensor.fields["filter_spectra"]).shape[1])
        tiled = tile_pattern(pattern, rows, cols)
        planes = np.zeros((rows, cols, nfilters), dtype=float)
        for channel_index in range(nfilters):
            mask = tiled == (channel_index + 1)
            planes[:, :, channel_index][mask] = sensor_data[mask]
        if nfilters == 1:
            return planes
        if nfilters == 3 and tuple(pattern.shape) == (2, 2):
            return _demosaic_rgb_planes(planes, pattern, demosaic_method)
        return _ie_bilinear(planes, pattern)
    return np.repeat(sensor_data[..., None], 3, axis=2)


def _sensor_space(sensor: Sensor, demosaic_method: str = "bilinear") -> np.ndarray:
    sensor_data = sensor.data.get("volts")
    if sensor_data is None:
        sensor_data = sensor.data.get("dv")
    if sensor_data is None:
        raise ValueError("Sensor has no computed volts.")
    return _sensor_space_from_data(sensor, sensor_data, demosaic_method)


def demosaic(ip: ImageProcessor, sensor: Sensor) -> np.ndarray:
    """Return the demosaiced sensor-space image for the current IP method."""

    method = str(
        ip.fields.get(
            "demosaic_method",
            ip.fields.get("demosaic", {}).get("method", "bilinear"),
        )
    )
    sensor_data = ip.data.get("input")
    if sensor_data is None:
        sensor_data = sensor.data.get("volts")
    if sensor_data is None:
        sensor_data = sensor.data.get("dv")
    if sensor_data is None:
        raise ValueError("Sensor has no computed volts.")
    return _sensor_space_from_data(sensor, sensor_data, method)


def _local_stat_fill(
    values: np.ndarray, sample_mask: np.ndarray, statistic: str, radius: int = 1
) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    valid = np.asarray(sample_mask, dtype=bool)
    if not np.any(valid):
        return output

    if statistic == "mean":
        kernel = np.ones((2 * int(radius) + 1, 2 * int(radius) + 1), dtype=float)
        weighted = convolve2d(output * valid.astype(float), kernel, mode="same", boundary="symm")
        counts = convolve2d(valid.astype(float), kernel, mode="same", boundary="symm")
        fill = np.divide(weighted, counts, out=np.zeros_like(weighted), where=counts > 0)
        output[~valid] = fill[~valid]
        return output

    pad = int(radius)
    padded = np.pad(output, pad, mode="symmetric")
    padded_mask = np.pad(valid, pad, mode="symmetric")
    default = float(np.median(output[valid]))
    missing_rows, missing_cols = np.where(~valid)
    for row, col in zip(missing_rows.tolist(), missing_cols.tolist(), strict=False):
        window = padded[row : row + 2 * pad + 1, col : col + 2 * pad + 1]
        window_mask = padded_mask[row : row + 2 * pad + 1, col : col + 2 * pad + 1]
        selected = window[window_mask]
        output[row, col] = default if selected.size == 0 else float(np.median(selected))
    return output


def demosaic_multichannel(
    rgb_format: Any,
    sensor: Sensor,
    method: str = "mean",
) -> np.ndarray:
    """Fill sparse per-filter CFA planes using the legacy multichannel helper contract."""

    if sensor is None:
        raise ValueError("demosaicMultichannel requires a sensor.")

    sparse = np.asarray(rgb_format, dtype=float)
    if sparse.ndim != 3:
        raise ValueError("demosaicMultichannel requires a rows x cols x bands array.")

    _, cfa_numbers, _ = sensor_determine_cfa(sensor)
    rows, cols, n_bands = sparse.shape
    cfa_numbers = np.asarray(cfa_numbers[:rows, :cols], dtype=int)
    normalized = param_format(method or "mean")
    if normalized in {"multichannel"}:
        normalized = "interpolate"
    if normalized == "kernelregression":
        normalized = "interpolate"
    if normalized not in {"interpolate", "mean", "median"}:
        raise UnsupportedOptionError("demosaicMultichannel", method)

    grid_x, grid_y = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    filled = sparse.copy()
    for band in range(n_bands):
        sample_mask = cfa_numbers == (band + 1)
        band_values = np.asarray(sparse[:, :, band], dtype=float)
        if not np.any(sample_mask):
            continue

        if normalized == "interpolate":
            sample_rows, sample_cols = np.where(sample_mask)
            samples = band_values[sample_mask]
            if samples.size == 1:
                interpolated = np.full((rows, cols), float(samples[0]), dtype=float)
            else:
                points = np.column_stack((sample_cols.astype(float), sample_rows.astype(float)))
                interp_method = "linear" if samples.size >= 3 else "nearest"
                interpolated = griddata(points, samples, (grid_x, grid_y), method=interp_method)
                if interpolated is None:
                    interpolated = np.full((rows, cols), np.nan, dtype=float)
                if np.any(~np.isfinite(interpolated)):
                    nearest = griddata(points, samples, (grid_x, grid_y), method="nearest")
                    interpolated = np.where(np.isfinite(interpolated), interpolated, nearest)
            filled[:, :, band] = np.asarray(interpolated, dtype=float)
        else:
            filled[:, :, band] = _local_stat_fill(band_values, sample_mask, normalized)

        filled[:, :, band][sample_mask] = band_values[sample_mask]

    return np.clip(np.asarray(filled, dtype=float), 0.0, None)


def pocs(bayer_in: Any, b_pattern: str | np.ndarray, iterN: int = 10) -> np.ndarray:
    """Return a sample-preserving RGB demosaic for the legacy POCS surface."""

    del iterN
    planes = np.asarray(bayer_in, dtype=float)
    if planes.ndim != 3 or planes.shape[2] < 3:
        raise ValueError("Pocs requires a Bayer plane stack with at least three channels.")

    pattern_name = _resolve_bayer_pattern_name(b_pattern)
    pattern = _bayer_pattern_array(pattern_name)
    rgb = _adaptive_laplacian_demosaic(np.asarray(planes[:, :, :3], dtype=float), pattern_name)
    cfa_numbers = tile_pattern(pattern, rgb.shape[0], rgb.shape[1])
    for channel_index in range(3):
        observed = cfa_numbers == (channel_index + 1)
        rgb[:, :, channel_index][observed] = planes[:, :, channel_index][observed]
    return np.clip(np.asarray(rgb, dtype=float), 0.0, None)


def _resolve_transform_reflectances(
    surfaces: np.ndarray | str,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    if isinstance(surfaces, str):
        normalized = param_format(surfaces)
        if normalized in {"mcc", "mccoptimized"}:
            _, reflectances = asset_store.load_reflectances("macbethChart.mat", wave_nm=wave)
            return np.asarray(reflectances, dtype=float)
        if normalized in {"esser", "esseroptimized"}:
            data = asset_store.load_mat("data/surfaces/charts/esser/reflectance/esserChart.mat")
            wavelengths = np.asarray(data["wavelength"], dtype=float)
            reflectances = np.asarray(data["data"], dtype=float)
            from .utils import interp_spectra

            return np.asarray(interp_spectra(wavelengths, reflectances, wave), dtype=float)
        if normalized == "multisurface":
            from .scene import ie_reflectance_samples

            return np.asarray(
                ie_reflectance_samples(None, None, wave, asset_store=asset_store)[0],
                dtype=float,
            )
        raise UnsupportedOptionError("imageSensorTransform", surfaces)
    return np.asarray(surfaces, dtype=float)


def _resolve_transform_illuminant_quanta(
    illuminant: str | np.ndarray,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    if isinstance(illuminant, str):
        _, illuminant_energy = asset_store.load_illuminant(illuminant, wave_nm=wave)
        illuminant_energy_array = np.asarray(illuminant_energy, dtype=float).reshape(-1)
    else:
        illuminant_energy_array = np.asarray(illuminant, dtype=float).reshape(-1)
    return np.asarray(energy_to_quanta(illuminant_energy_array, wave), dtype=float)


def image_sensor_transform(
    sensor_qe: np.ndarray,
    target_qe: np.ndarray,
    illuminant: str | np.ndarray = "D65",
    wave: np.ndarray | None = None,
    surfaces: np.ndarray | str = "multisurface",
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Calculate the linear transform from sensor space to a target QE space."""

    wave_array = (
        np.arange(400.0, 701.0, 10.0, dtype=float)
        if wave is None
        else np.asarray(wave, dtype=float).reshape(-1)
    )
    store = _store(asset_store)
    reflectances = _resolve_transform_reflectances(surfaces, wave_array, asset_store=store)
    illuminant_quanta = _resolve_transform_illuminant_quanta(
        illuminant,
        wave_array,
        asset_store=store,
    )
    weighted_surfaces = np.asarray(reflectances, dtype=float) * illuminant_quanta.reshape(-1, 1)
    sensor_response = weighted_surfaces.T @ np.asarray(sensor_qe, dtype=float)
    target_response = weighted_surfaces.T @ np.asarray(target_qe, dtype=float)
    matrix, _, _, _ = np.linalg.lstsq(sensor_response, target_response, rcond=None)
    return np.asarray(matrix, dtype=float)


def image_esser_transform(
    sensor_qe: np.ndarray,
    target_qe: np.ndarray,
    illuminant: str | np.ndarray = "D65",
    wave: np.ndarray | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Calculate the Esser-optimized linear transform from sensor to target space."""

    return image_sensor_transform(
        sensor_qe,
        target_qe,
        illuminant,
        wave,
        "esser",
        asset_store=asset_store,
    )


def image_sensor_conversion(
    sensor: Sensor,
    cmf: np.ndarray | None = None,
    surfaces: np.ndarray | str | None = None,
    illuminant: np.ndarray | str | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the linear transform from sensor catch to the desired CMF space."""

    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    store = _store(asset_store)

    if cmf is None:
        cmf_array = np.asarray(ie_read_spectra("XYZ.mat", wave, asset_store=store), dtype=float)
    else:
        cmf_array = np.asarray(cmf, dtype=float)
    if surfaces is None:
        raise ValueError("Surface reflectances are required.")
    if illuminant is None:
        raise ValueError("Illuminant data are required.")

    if isinstance(surfaces, str):
        _, reflectances = store.load_reflectances(surfaces, wave_nm=wave)
        surfaces_array = np.asarray(reflectances, dtype=float)
    else:
        surfaces_array = np.asarray(surfaces, dtype=float)
    if isinstance(illuminant, str):
        _, illuminant_energy = store.load_illuminant(illuminant, wave_nm=wave)
        illuminant_vector = np.asarray(illuminant_energy, dtype=float).reshape(-1)
    else:
        illuminant_vector = np.asarray(illuminant, dtype=float).reshape(-1)

    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    weighted_surfaces = illuminant_vector.reshape(-1, 1) * surfaces_array
    actual = spectral_qe.T @ weighted_surfaces
    desired = cmf_array.T @ weighted_surfaces
    transform = desired @ np.linalg.pinv(actual)
    white_cmf = cmf_array.T @ illuminant_vector
    return transform, actual, desired, white_cmf


def image_sensor_correction(
    img: np.ndarray,
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, ImageProcessor, np.ndarray]:
    """Convert sensor-space image data into the image processor internal space."""

    corrected_ip = _ensure_ip_state(ip.clone())
    sensor_space, squeeze_channel = _as_channel_image(np.asarray(img, dtype=float))
    internal, sensor_transform = _sensor_to_internal(
        sensor_space, corrected_ip, sensor, asset_store=_store(asset_store)
    )
    corrected_ip.data["sensorspace"] = np.asarray(sensor_space, dtype=float)
    corrected_ip.data["xyz"] = np.asarray(internal, dtype=float)
    corrected_ip.data["ics"] = np.asarray(internal, dtype=float)
    corrected_ip.data["transforms"][0] = np.asarray(sensor_transform, dtype=float)
    corrected_ip.fields["sensor_conversion_matrix"] = np.asarray(sensor_transform, dtype=float)
    return _restore_channel_image(internal, squeeze_channel), corrected_ip, sensor_transform


def image_illuminant_correction(
    img: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, ImageProcessor, np.ndarray]:
    """Apply the IP illuminant-correction stage to internal-color-space data."""

    corrected_ip = _ensure_ip_state(ip.clone())
    internal_image, squeeze_channel = _as_channel_image(np.asarray(img, dtype=float))
    corrected, illuminant_transform = _illuminant_correct_internal(
        internal_image, corrected_ip, asset_store=_store(asset_store)
    )
    corrected_ip.data["ics"] = np.asarray(corrected, dtype=float)
    corrected_ip.data["transforms"][1] = np.asarray(illuminant_transform, dtype=float)
    corrected_ip.fields["illuminant_correction_matrix"] = np.asarray(
        illuminant_transform, dtype=float
    )
    return _restore_channel_image(corrected, squeeze_channel), corrected_ip, illuminant_transform


def image_color_balance(
    img: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, ImageProcessor, np.ndarray]:
    """Deprecated MATLAB alias for `imageIlluminantCorrection`."""

    return image_illuminant_correction(img, ip, asset_store=asset_store)


def image_rgb_to_xyz(
    ip: ImageProcessor,
    rgb: np.ndarray,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Convert linear display RGB values into XYZ, preserving RGB or XW format."""

    rgb_array = np.asarray(rgb, dtype=float)
    rgb_flag = rgb_array.ndim != 2
    if rgb_flag:
        rgb_xw, rows, cols, _ = rgb_to_xw_format(rgb_array)
    else:
        rgb_xw = rgb_array
        rows = cols = 0

    spd = np.asarray(display_get(ip.fields["display"], "rgb spd"), dtype=float)
    wave = np.asarray(display_get(ip.fields["display"], "wave"), dtype=float)
    energy = rgb_xw @ spd.T
    xyz = xyz_from_energy(energy, wave, asset_store=_store(asset_store))

    if rgb_flag:
        return xw_to_rgb_format(np.asarray(xyz, dtype=float), rows, cols)
    return np.asarray(xyz, dtype=float)


def ie_internal_to_display(ip: ImageProcessor) -> np.ndarray:
    """Return the linear transform from IP internal color space to display RGB."""

    internal_cmf = ip_get(ip, "internal cmf")
    if internal_cmf is None:
        raise UnsupportedOptionError("ieInternal2Display", ip_get(ip, "internal cs"))
    display_spd = np.asarray(ip_get(ip, "display rgb spd"), dtype=float)
    return np.asarray(
        np.linalg.inv(display_spd.T @ np.asarray(internal_cmf, dtype=float)),
        dtype=float,
    )


def ip_hdr_white(
    ip: ImageProcessor,
    *args: Any,
    saturation: float | None = None,
    hdr_level: float = 0.95,
    wgt_blur: float = 1.0,
) -> tuple[ImageProcessor, np.ndarray]:
    """Apply the legacy HDR whitening step to the current IP result image."""

    if len(args) % 2 != 0:
        raise ValueError("ipHDRWhite expects MATLAB-style key/value pairs.")
    for index in range(0, len(args), 2):
        key = param_format(str(args[index]))
        value = args[index + 1]
        if key == "saturation":
            saturation = float(value)
        elif key == "hdrlevel":
            hdr_level = float(value)
        elif key == "wgtblur":
            wgt_blur = float(value)
        else:
            raise UnsupportedOptionError("ipHDRWhite", args[index])

    updated = _ensure_ip_state(ip.clone())
    input_data = updated.data.get("input")
    if input_data is None:
        raise ValueError("IP has no input data for ipHDRWhite.")
    result = updated.data.get("result")
    if result is None:
        raise ValueError("IP has no result data for ipHDRWhite.")

    input_array = np.asarray(input_data, dtype=float)
    result_array = np.asarray(result, dtype=float)
    saturation_value = float(np.max(input_array)) if saturation is None else float(saturation)
    if saturation_value <= 0.0:
        raise ValueError("ipHDRWhite saturation must be positive.")

    weights = (input_array / saturation_value - float(hdr_level)) / max(1e-6, 1.0 - float(hdr_level))
    weights = np.clip(weights, 0.0, 1.0)
    sigma = max(float(wgt_blur), 0.0)
    if sigma > 0.0:
        weights = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0, truncate=2.0)

    whitened = np.ones_like(result_array, dtype=float)
    updated.data["result"] = whitened * weights[:, :, np.newaxis] + result_array * (1.0 - weights[:, :, np.newaxis])
    return updated, np.asarray(weights, dtype=float)


def image_distort(
    img: np.ndarray,
    d_method: str = "Gaussian Noise",
    *args: Any,
) -> np.ndarray:
    """Apply a simple legacy image-distortion method."""

    if len(args) > 1:
        raise ValueError("imageDistort accepts at most one method parameter.")

    image = np.asarray(img)
    method = param_format(d_method or "Gaussian Noise")

    if method == "gaussiannoise":
        if args:
            noise_scale = float(args[0])
        else:
            noise_scale = 0.05 * float(np.max(image)) if image.size else 0.0
        noise = noise_scale * np.random.randn(*image.shape)
        if np.issubdtype(image.dtype, np.integer) and float(np.max(image, initial=0.0)) < 256.0:
            distorted = np.clip(np.asarray(image, dtype=float) + noise, 0.0, 255.0)
            return np.asarray(np.rint(distorted), dtype=np.uint8)
        return np.asarray(image, dtype=float) + noise

    if method == "jpegcompress":
        quality = int(np.rint(float(args[0]))) if args else 75
        quality = min(max(quality, 1), 100)
        if np.issubdtype(image.dtype, np.integer):
            payload = np.asarray(image)
        else:
            payload = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
            payload = np.clip(np.round(payload * 255.0), 0.0, 255.0).astype(np.uint8)
        temp_path: Path | None = None
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            iio.imwrite(temp_path, payload, extension=".jpg", quality=quality)
            return np.asarray(iio.imread(temp_path))
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    if method == "scalecontrast":
        scale = float(args[0]) if args else 0.1
        return np.asarray(image, dtype=float) * (1.0 + scale)

    raise UnsupportedOptionError("imageDistort", d_method)


def display_render(
    ics_image: np.ndarray,
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, ImageProcessor, np.ndarray]:
    """Convert internal-color-space data into linear display RGB."""

    rendered_ip = _ensure_ip_state(ip.clone())
    display_linear, display_transform = _display_render(
        np.asarray(ics_image, dtype=float), rendered_ip, sensor, asset_store=_store(asset_store)
    )
    rendered_ip.data["result"] = np.asarray(display_linear, dtype=float)
    rendered_ip.data["transforms"][2] = np.asarray(display_transform, dtype=float)
    rendered_ip.fields["ics2display"] = np.asarray(display_transform, dtype=float)
    return (
        np.asarray(display_linear, dtype=float),
        rendered_ip,
        np.asarray(display_transform, dtype=float),
    )


def ip_clear_data(ip: ImageProcessor) -> ImageProcessor:
    """Clear the computed data payload from an image processor."""

    cleared = ip.clone()
    cleared.data = {}
    return _ensure_ip_state(cleared)


def vcimage_clear_data(ip: ImageProcessor) -> ImageProcessor:
    """Legacy alias for clearing the computed data payload from an image processor."""

    return ip_clear_data(ip)


def ip_save_image(
    ip: ImageProcessor,
    f_name: str | Path,
    show_image_flag: bool = False,
    true_size_flag: bool = False,
    *,
    cropborder: bool = False,
) -> str:
    """Save the current IP image to an 8-bit PNG file."""

    del show_image_flag, true_size_flag
    ip = _ensure_ip_state(ip)
    image = ip.data.get("srgb")
    if image is None:
        result = ip.data.get("result")
        if result is None:
            raise ValueError("IP has no computed image data to save.")
        image = linear_to_srgb(np.clip(np.asarray(result, dtype=float), 0.0, 1.0))

    srgb = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    if cropborder:
        srgb = _crop_border(srgb)

    output_path = Path(f_name).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.clip(np.round(srgb * 255.0), 0.0, 255.0).astype(np.uint8)
    iio.imwrite(output_path, payload)
    return str(output_path)


def image_show_image(
    ip: ImageProcessor,
    gam: float | None = None,
    true_size_flag: bool = False,
    app: Any = 0,
) -> np.ndarray:
    """Calculate the current IP image and return the headless display RGB."""

    del true_size_flag, app
    ip = _ensure_ip_state(ip)
    if gam is None:
        gam = float(ip.fields.get("render", {}).get("gamma", 1.0) or 1.0)

    base = ip.data.get("srgb")
    if base is None:
        result = ip.data.get("result")
        if result is not None:
            base = linear_to_srgb(np.clip(np.asarray(result, dtype=float), 0.0, 1.0))
        else:
            xyz = image_data_xyz(ip)
            if xyz is None:
                raise ValueError("IP has no image data to render.")
            base = xyz_to_srgb(np.asarray(xyz, dtype=float))

    img = np.asarray(base, dtype=float)
    render_flag = ip_get(ip, "render flag")
    if render_flag in {2, "hdr"}:
        from .scene import hdr_render

        img = np.asarray(hdr_render(img), dtype=float)
    elif render_flag in {3, "gray", "monochrome"}:
        gray = np.mean(img, axis=2, keepdims=True)
        img = np.repeat(gray, 3, axis=2)
    elif render_flag not in {1, "rgb"}:
        raise ValueError(f"Unsupported render flag {render_flag!r}.")
    elif bool(ip_get(ip, "scale display")):
        linear = srgb_to_linear(np.clip(img, 0.0, 1.0))
        maximum = float(np.max(linear))
        if maximum > 0.0:
            img = linear_to_srgb(linear / maximum)

    img = np.clip(np.asarray(img, dtype=float), 0.0, 1.0)
    if float(gam) != 1.0:
        img = np.power(img, float(gam))
    return np.asarray(img, dtype=float)


def image_mcc_transform(
    sensor_qe: np.ndarray,
    target_qe: np.ndarray,
    illuminant: str | np.ndarray = "D65",
    wave: np.ndarray | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Calculate the Macbeth-optimized linear transform from sensor to target space."""

    if wave is None:
        wave_array = np.arange(400.0, 701.0, 10.0, dtype=float)
    else:
        wave_array = np.asarray(wave, dtype=float).reshape(-1)
    store = _store(asset_store)
    _, reflectances = store.load_reflectances("macbethChart.mat", wave_nm=wave_array)
    if isinstance(illuminant, str):
        _, illuminant_energy = store.load_illuminant(illuminant, wave_nm=wave_array)
        illuminant_energy_array = np.asarray(illuminant_energy, dtype=float).reshape(-1)
    else:
        illuminant_energy_array = np.asarray(illuminant, dtype=float).reshape(-1)
    illuminant_quanta = np.asarray(
        energy_to_quanta(illuminant_energy_array, wave_array), dtype=float
    )
    weighted_surfaces = np.asarray(reflectances, dtype=float) * illuminant_quanta.reshape(-1, 1)
    sensor_macbeth = weighted_surfaces.T @ np.asarray(sensor_qe, dtype=float)
    target_macbeth = weighted_surfaces.T @ np.asarray(target_qe, dtype=float)
    return np.asarray(np.linalg.pinv(sensor_macbeth) @ target_macbeth, dtype=float)


def _sensor_to_internal(
    sensor_space: np.ndarray,
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, np.ndarray]:
    conversion_method = param_format(ip.fields.get("conversion_method_sensor", "mcc optimized"))
    filter_spectra = np.asarray(sensor.fields["filter_spectra"], dtype=float)
    wave = np.asarray(sensor.fields["wave"], dtype=float)

    if conversion_method in {"none", "sensor"}:
        transform = np.eye(sensor_space.shape[2], dtype=float)
        internal = sensor_space.copy()
    elif conversion_method in {"current", "currentmatrix", "manualmatrixentry"}:
        transform = _ip_transform(ip, 0)
        internal = sensor_space @ transform
    elif conversion_method in {"mccoptimized", "mcc", "esseroptimized", "esser"}:
        surfaces = "esser" if "esser" in conversion_method else "mcc"
        transform = sensor_to_target_matrix(
            wave,
            filter_spectra,
            target_space="xyz",
            illuminant="D65",
            surfaces=surfaces,
            asset_store=asset_store,
        )
        internal = sensor_space @ transform
    else:
        raise UnsupportedOptionError("ipCompute", conversion_method)

    internal = np.clip(internal, 0.0, None)
    internal_max = float(np.max(internal))
    if internal_max > 0.0:
        internal = internal / internal_max
    return internal, transform


def _illuminant_white_ratio(
    ip: ImageProcessor,
    channels: int,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    internal_cmf = ip_get(ip, "internal cmf")
    if internal_cmf is None:
        return np.ones(int(channels), dtype=float)

    wave = np.asarray(ip_get(ip, "wave"), dtype=float).reshape(-1)
    target = np.asarray(ie_read_spectra("D65", wave, asset_store=asset_store), dtype=float).reshape(
        -1
    )
    white_ratio = np.asarray(internal_cmf, dtype=float).T @ target
    max_value = float(np.max(white_ratio))
    if max_value <= 0.0:
        return np.ones(int(channels), dtype=float)
    return np.asarray(white_ratio / max_value, dtype=float).reshape(-1)


def _gray_world_transform(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    data = np.nan_to_num(np.asarray(internal_image, dtype=float), nan=0.0)
    channels = int(data.shape[2])
    averages = np.mean(data, axis=(0, 1))
    white_ratio = _illuminant_white_ratio(ip, channels, asset_store=asset_store)
    reference = max(float(white_ratio[0]), np.finfo(float).tiny)
    white_ratio = white_ratio / reference
    base_average = max(float(averages[0]), np.finfo(float).tiny)
    scale = np.zeros(channels, dtype=float)
    for channel_index in range(channels):
        average = max(float(averages[channel_index]), np.finfo(float).tiny)
        scale[channel_index] = float(white_ratio[channel_index]) * (base_average / average)
    return np.diag(scale)


def _white_world_transform(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    data = np.nan_to_num(np.asarray(internal_image, dtype=float), nan=0.0)
    channels = int(data.shape[2])
    maxima = np.max(data, axis=(0, 1))
    brightest_channel = int(np.argmax(maxima))
    brightest_plane = data[:, :, brightest_channel]
    max_brightness = max(float(maxima[brightest_channel]), np.finfo(float).tiny)
    criterion = 0.7
    white_ratio = _illuminant_white_ratio(ip, channels, asset_store=asset_store)
    reference = max(float(white_ratio[0]), np.finfo(float).tiny)
    white_ratio = white_ratio / reference

    bright_values = np.zeros(channels, dtype=float)
    mask = brightest_plane >= (criterion * max_brightness)
    for channel_index in range(channels):
        channel_data = data[:, :, channel_index][mask]
        if channel_data.size == 0:
            bright_values[channel_index] = np.finfo(float).tiny
        else:
            bright_values[channel_index] = max(float(np.mean(channel_data)), np.finfo(float).tiny)

    base_value = max(float(bright_values[0]), np.finfo(float).tiny)
    scale = np.zeros(channels, dtype=float)
    for channel_index in range(channels):
        scale[channel_index] = float(white_ratio[channel_index]) * (
            base_value / float(bright_values[channel_index])
        )
    return np.diag(scale)


def _illuminant_correct_internal(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, np.ndarray]:
    method = param_format(ip.fields.get("illuminant_correction_method", "none"))
    channels = int(np.asarray(internal_image, dtype=float).shape[2])

    if method in {"none"}:
        transform = np.eye(channels, dtype=float)
        return np.asarray(internal_image, dtype=float), transform
    if method in {"grayworld"}:
        transform = _gray_world_transform(internal_image, ip, asset_store=asset_store)
        return image_linear_transform(internal_image, transform), transform
    if method in {"whiteworld"}:
        transform = _white_world_transform(internal_image, ip, asset_store=asset_store)
        return image_linear_transform(internal_image, transform), transform
    if method in {"manualmatrixentry", "manual"}:
        transform = _ip_transform(ip, 1)
        return image_linear_transform(internal_image, transform), transform

    raise UnsupportedOptionError("imageIlluminantCorrection", method)


def _display_render(
    internal_image: np.ndarray,
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, np.ndarray]:
    internal_cs = str(ip.fields.get("internal_cs", "xyz"))
    display = ip.fields["display"]
    display_spd = np.asarray(display.fields["spd"], dtype=float)
    conversion_method = param_format(ip.fields.get("conversion_method_sensor", "mcc optimized"))

    if param_format(internal_cs) == "xyz":
        transform = internal_to_display_matrix(
            np.asarray(ip.fields["wave"], dtype=float),
            display_spd,
            internal_cs=internal_cs,
            asset_store=asset_store,
        )
    elif param_format(internal_cs) == "sensor":
        sensor_qe = np.asarray(sensor.fields["filter_spectra"], dtype=float)
        transform = np.linalg.pinv(sensor_qe.T @ display_spd).T
    else:
        raise UnsupportedOptionError("displayRender", internal_cs)

    if conversion_method in {"current", "currentmatrix", "manualmatrixentry", "none"}:
        display_linear = internal_image.copy()
    elif conversion_method in {"sensor", "mccoptimized", "mcc", "esseroptimized", "esser"}:
        display_linear = internal_image @ transform
    else:
        raise UnsupportedOptionError("displayRender", conversion_method)

    display_max = float(np.max(display_linear))
    if bool(ip.fields["render"].get("scale", True)) and display_max > 0.0:
        display_linear = display_linear / display_max * float(sensor_get(sensor, "response ratio"))
    display_linear = np.maximum(display_linear, 0.0)
    return display_linear, transform


def ip_compute(
    ip: ImageProcessor,
    sensor: Sensor,
    *,
    hdr_white: bool = False,
    hdr_level: float = 0.95,
    wgt_blur: float = 2.0,
    network_demosaic: str | None = None,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> ImageProcessor:
    """Compute the default image processing pipeline."""

    del wgt_blur, network_demosaic
    store = _store(asset_store)
    computed = _ensure_ip_state(ip.clone())
    computed.data["input"] = sensor.data.get("dv", sensor.data.get("volts"))
    sensor_space = _sensor_space(sensor, computed.fields.get("demosaic_method", "bilinear"))
    internal_image, sensor_transform = _sensor_to_internal(
        sensor_space, computed, sensor, asset_store=store
    )
    corrected_internal, illuminant_transform = _illuminant_correct_internal(
        internal_image, computed, asset_store=store
    )
    display_linear, display_transform = _display_render(
        corrected_internal, computed, sensor, asset_store=store
    )

    if hdr_white:
        max_channel = np.max(display_linear, axis=2, keepdims=True)
        blend = np.clip((max_channel - hdr_level) / max(1e-6, 1.0 - hdr_level), 0.0, 1.0)
        display_linear = display_linear * (1.0 - blend) + blend

    display = computed.fields["display"]
    clamped_display = np.clip(display_linear, 0.0, 1.0)
    display_rgb = invert_gamma_table(
        clamped_display, np.asarray(display.fields["gamma"], dtype=float)
    )
    srgb = linear_to_srgb(clamped_display)

    computed.fields["sensor_conversion_matrix"] = sensor_transform
    computed.fields["illuminant_correction_matrix"] = illuminant_transform
    computed.fields["ics2display"] = display_transform
    computed.data["transforms"] = [
        np.asarray(sensor_transform, dtype=float),
        np.asarray(computed.fields["illuminant_correction_matrix"], dtype=float),
        np.asarray(display_transform, dtype=float),
    ]
    computed.data["sensorspace"] = sensor_space
    computed.data["xyz"] = internal_image
    computed.data["ics"] = corrected_internal
    computed.data["display_rgb"] = display_rgb
    computed.data["srgb"] = srgb
    computed.data["result"] = display_linear
    return track_ip_session_state(session, computed)


def image_data_xyz(
    ip: ImageProcessor,
    roi_locs: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray | None:
    """Convert display-linear image-processor data to XYZ."""

    ip = _ensure_ip_state(ip)
    if roi_locs is None:
        data = ip_get(ip, "result")
    else:
        from .roi import vc_get_roi_data

        data = vc_get_roi_data(ip, roi_locs, "result")
    if data is None:
        return None
    return image_rgb_to_xyz(ip, np.asarray(data, dtype=float), asset_store=asset_store)


def ip_get(ip: ImageProcessor, parameter: str, *args: Any) -> Any:
    ip = _ensure_ip_state(ip)
    key = param_format(parameter)
    if key in {"result", "displaylinearrgb", "datadisplay", "displaydata"}:
        return ip.data.get("result")
    if key in {"dataintensitiesmax", "resultmax"}:
        result = ip.data.get("result")
        return None if result is None else float(np.max(np.asarray(result, dtype=float)))
    if key in {"maxsensor", "maximumsensorvalue", "maximumsensorvoltageswing"}:
        if "max" in ip.data and ip.data["max"] is not None:
            return float(np.asarray(ip.data["max"], dtype=float).reshape(-1)[0])
        sensor_input = ip.data.get("input")
        return None if sensor_input is None else float(np.max(np.asarray(sensor_input, dtype=float)))
    if key in {"displayviewingdistance"}:
        return display_get(ip.fields["display"], "viewing distance")
    if key in {"displaydpi"}:
        return display_get(ip.fields["display"], "dpi")

    prefix, remainder = split_prefixed_parameter(parameter, ("display", "l3"))
    if prefix == "display":
        if not remainder:
            return ip.fields["display"]
        return display_get(ip.fields["display"], remainder, *args)
    if prefix == "l3":
        if not remainder:
            return ip.fields.get("l3")
        l3 = ip.fields.get("l3")
        if l3 is None:
            return None
        return _l3_get(l3, remainder)

    if key == "type":
        return ip.type
    if key == "name":
        return ip.name
    if key in {"spectrum", "spectrumstructure"}:
        return ip.fields["spectrum"]
    if key in {"wave", "wavelength"}:
        return np.asarray(ip.fields["wave"], dtype=float)
    if key in {"binwidth", "waveresolution"}:
        wave = np.asarray(ip.fields["wave"], dtype=float)
        if wave.size < 2:
            return 1.0
        return float(wave[1] - wave[0])
    if key in {"nwave", "nwaves"}:
        return int(np.asarray(ip.fields["wave"], dtype=float).size)
    if key in {"row", "rows"}:
        input_data = ip.data.get("input")
        return None if input_data is None else int(np.asarray(input_data).shape[0])
    if key in {"col", "cols"}:
        input_data = ip.data.get("input")
        return None if input_data is None else int(np.asarray(input_data).shape[1])
    if key == "inputsize":
        input_data = ip.data.get("input")
        return None if input_data is None else tuple(np.asarray(input_data).shape)
    if key in {"rgbsize", "resultsize", "displaysize", "size"}:
        result = ip.data.get("result")
        return None if result is None else tuple(np.asarray(result).shape)
    if key in {"imagecenter", "center"}:
        size = ip_get(ip, "size")
        if size is None:
            return None
        size_array = np.asarray(size, dtype=float).reshape(-1)
        return (size_array[:2] + 1.0) / 2.0
    if key in {"internalcs", "internalcolorspace"}:
        return ip.fields["internal_cs"]
    if key in {"internalcmf", "internalcolormatchingfunction"}:
        if param_format(ip.fields["internal_cs"]) == "sensor":
            return None
        return xyz_color_matching(
            np.asarray(ip.fields["wave"], dtype=float), asset_store=_store(None)
        )
    if key in {"illuminantcorrection"}:
        return ip.fields["illuminant_correction"]
    if key in {"illuminantcorrectionmethod"}:
        return ip.fields["illuminant_correction"].get("method", "none")
    if key in {
        "illuminantcorrectionmatrix",
        "illuminantcorrectiontransform",
        "correctiontransformilluminant",
        "correctionmatrixilluminant",
    }:
        return _ip_transform(ip, 1)
    if key in {"demosaic", "demosaicstructure"}:
        return ip.fields["demosaic"]
    if key in {"demosaicmethod"}:
        return ip.fields["demosaic"].get("method", "none")
    if key == "chartparameters":
        return copy.deepcopy(_ip_chart_parameters(ip))
    if key in {"cornerpoints", "chartcornerpoints", "chartcorners"}:
        value = _ip_chart_parameters(ip).get("cornerPoints")
        return None if value is None else np.asarray(value).copy()
    if key == "mcccornerpoints":
        return ip_get(ip, "chart corner points")
    if key in {"chartrects", "chartrectangles"}:
        value = _ip_chart_parameters(ip).get("rects")
        return None if value is None else np.asarray(value).copy()
    if key in {"currentrect", "chartcurrentrect"}:
        value = _ip_chart_parameters(ip).get("currentRect")
        return None if value is None else np.asarray(value).copy()
    if key == "mccrecthandles":
        return _copy_metadata_value(ip.fields.get("mccRectHandles"))
    if key in {"sensorconversion", "conversionsensor"}:
        return ip.fields["sensor_correction"]
    if key in {"sensorconversionmethod", "conversionmethodsensor"}:
        return ip.fields["sensor_correction"].get("method", "none")
    if key in {"sensorconversionmatrix", "conversiontransformsensor", "correctionmatrixsensor"}:
        return _ip_transform(ip, 0)
    if key in {"transformcellarray", "transforms"}:
        return list(ip.data["transforms"])
    if key in {"transformlist", "eachtransform"}:
        return [_ip_transform(ip, 0), _ip_transform(ip, 1), _ip_transform(ip, 2)]
    if key == "transformmethod":
        return ip.fields["transform_method"]
    if key in {
        "ics2display",
        "ics2displaymatrix",
        "ics2displaytransform",
        "internalcs2displayspace",
    }:
        return _ip_transform(ip, 2)
    if key in {"transformcombined", "combinedtransform", "prodt"}:
        return _ip_transform(ip, 0) @ _ip_transform(ip, 1) @ _ip_transform(ip, 2)
    if key in {"render", "renderstructure"}:
        return ip.fields["render"]
    if key in {"renderflag", "displaymode"}:
        return ip.fields["render"].get("renderflag", 1)
    if key in {"combineexposures", "combinationmethod"}:
        return ip.fields.get("combination_method", "longest")
    if key in {"renderscale", "scaledisplay", "scaledisplayoutput"}:
        return bool(ip.fields["render"].get("scale", True))
    if key == "renderwhitept":
        return bool(ip.fields["render"].get("whitept", False))
    if key in {"gammadisplay", "rendergamma", "gamma"}:
        return float(ip.fields["render"].get("gamma", 1.0) or 1.0)
    if key in {"data", "datastructure"}:
        return ip.data
    if key in {"roidata", "dataroi", "roiresult"}:
        if not args:
            return None
        from .roi import vc_get_roi_data

        return vc_get_roi_data(ip, args[0], "result")
    if key in {"roixyz", "xyzroi"}:
        return image_data_xyz(ip, args[0] if args else None)
    if key in {"chromaticity", "roichromaticity"}:
        xyz = image_data_xyz(ip, args[0] if args else None)
        return None if xyz is None else chromaticity_xy(np.asarray(xyz, dtype=float))
    if key in {"roichromaticitymean", "roimeanchromaticity"}:
        if not args:
            raise ValueError("ROI required for ipGet(..., 'roi chromaticity mean').")
        chromaticity = ip_get(ip, "chromaticity", args[0])
        return (
            None
            if chromaticity is None
            else np.mean(np.asarray(chromaticity, dtype=float), axis=0).reshape(-1)
        )
    if key == "imagegrid":
        size = ip_get(ip, "size")
        if size is None:
            return None
        center = np.asarray(ip_get(ip, "center"), dtype=float).reshape(-1)
        size_array = np.asarray(size, dtype=int).reshape(-1)
        x_coords, y_coords = np.meshgrid(
            np.arange(1, int(size_array[1]) + 1, dtype=float),
            np.arange(1, int(size_array[0]) + 1, dtype=float),
        )
        return [x_coords - float(center[1]), y_coords - float(center[0])]
    if key == "distance2center":
        grid = ip_get(ip, "imagegrid")
        if grid is None:
            return None
        return np.sqrt(np.square(np.asarray(grid[0], dtype=float)) + np.square(np.asarray(grid[1], dtype=float)))
    if key == "angle":
        grid = ip_get(ip, "imagegrid")
        if grid is None:
            return None
        return np.arctan2(np.asarray(grid[1], dtype=float), np.asarray(grid[0], dtype=float))
    if key in {"input", "sensorinput", "sensormosaic"}:
        return ip.data.get("input")
    if key == "quantization":
        return copy.deepcopy(ip.data.get("quantization"))
    if key == "quantizationmethod":
        quantization = ip.data.get("quantization")
        if isinstance(quantization, dict):
            return quantization.get("method")
        return quantization
    if key in {"quantizationnbits", "nbits", "bits"}:
        quantization = ip.data.get("quantization")
        if not isinstance(quantization, dict):
            return None
        bits = quantization.get("bits")
        return None if bits is None else int(bits)
    if key == "maxdigitalvalue":
        nbits = ip_get(ip, "nbits")
        return 1.0 if nbits is None else float(2 ** int(nbits))
    if key in {"datasensor", "sensordata", "sensorchannels", "sensorspace", "demosaicsensor"}:
        return ip.data.get("sensorspace")
    if key in {"ninputfilters", "numbersensorchannels", "nsensorinputs", "nsensorchannels"}:
        sensor_space = ip.data.get("sensorspace")
        if sensor_space is None:
            return None
        sensor_space_array = np.asarray(sensor_space)
        return 1 if sensor_space_array.ndim < 3 else int(sensor_space_array.shape[2])
    if key in {"sensormax", "rgbmax", "datamax"}:
        return ip.fields.get("datamax")
    if key in {"datasrgb", "srgb"}:
        return ip.data.get("srgb")
    if key in {"dataintensitiesscaled", "scaledresult", "resultscaledtomax", "resultscaled"}:
        scaled_ip = ip_set(ip.clone(), "scale display output", True)
        return image_show_image(scaled_ip)
    if key in {"dataintensitiesdv", "dataintensitiesdigitalvalues"}:
        result = ip.data.get("result")
        if result is None:
            return None
        return np.asarray(result, dtype=float) * float(ip_get(ip, "max digital value"))
    if key in {"dataintensity", "resultprimary", "resultprimaryn"}:
        if not args:
            raise ValueError("Primary number required for ipGet(..., 'result primary').")
        result = ip.data.get("result")
        if result is None:
            raise ValueError("IP has no result data for ipGet(..., 'result primary').")
        primary_index = int(args[0]) - 1
        result_array = np.asarray(result, dtype=float)
        if result_array.ndim < 3 or primary_index < 0 or primary_index >= int(result_array.shape[2]):
            raise ValueError("No such display primary.")
        return result_array[:, :, primary_index]
    if key in {"dataxyz", "xyz"}:
        return ip.data.get("xyz")
    if key in {"dataics", "ics"}:
        return ip.data.get("ics", ip.data.get("xyz"))
    if key == "dataicsilluminantcorrected":
        return ip.data.get("ics", ip.data.get("xyz"))
    if key == "dataluminance":
        xyz = ip.data.get("xyz")
        return None if xyz is None else np.asarray(xyz, dtype=float)[..., 1]
    if key in {"whitepoint", "wp", "datawhitepoint", "datawp", "imagewhitepoint", "imagewp"}:
        return ip.data.get("wp")
    if key in {"dataordisplaywhitepoint", "dataormonitorwhitepoint"}:
        data_white_point = ip.data.get("wp")
        if data_white_point is not None:
            return data_white_point
        return display_get(ip.fields["display"], "white point")
    raise KeyError(f"Unsupported ipGet parameter: {parameter}")


def ip_set(
    ip: ImageProcessor,
    parameter: str,
    value: Any,
    *args: Any,
    session: SessionContext | None = None,
) -> ImageProcessor:
    ip = _ensure_ip_state(ip)
    key = param_format(parameter)
    if key == "displayviewingdistance":
        ip.fields["display"] = display_set(ip.fields["display"], "viewing distance", value)
        return track_ip_session_state(session, ip)
    if key == "displaydpi":
        ip.fields["display"] = display_set(ip.fields["display"], "dpi", value)
        return track_ip_session_state(session, ip)

    prefix, remainder = split_prefixed_parameter(parameter, ("display", "l3"))
    if prefix == "display":
        if not remainder:
            ip.fields["display"] = track_session_object(session, value)
        else:
            ip.fields["display"] = display_set(ip.fields["display"], remainder, value, *args)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if prefix == "l3":
        ip.fields["l3"] = value
        return track_ip_session_state(session, ip)

    if key == "type":
        ip.type = str(value)
        return track_ip_session_state(session, ip)
    if key == "name":
        ip.name = str(value)
        return track_ip_session_state(session, ip)
    if key == "chartparameters":
        chart = _ip_chart_parameters(ip)
        chart.update(dict(value))
        return track_ip_session_state(session, ip)
    if key in {"chartcornerpoints", "cornerpoints", "chartcorners"}:
        chart = _ip_chart_parameters(ip)
        chart["cornerPoints"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key == "mcccornerpoints":
        chart = _ip_chart_parameters(ip)
        chart["cornerPoints"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key in {"chartrects", "chartrectangles"}:
        chart = _ip_chart_parameters(ip)
        chart["rects"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key in {"currentrect", "chartcurrentrect"}:
        chart = _ip_chart_parameters(ip)
        chart["currentRect"] = np.asarray(value).copy()
        return track_ip_session_state(session, ip)
    if key == "mccrecthandles":
        ip.fields["mccRectHandles"] = _copy_metadata_value(value)
        return track_ip_session_state(session, ip)
    if key in {"spectrum"}:
        ip.fields["spectrum"] = dict(value)
        if "wave" in ip.fields["spectrum"]:
            ip.fields["wave"] = np.asarray(ip.fields["spectrum"]["wave"], dtype=float).reshape(-1)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"wave", "wavelength"}:
        ip.fields["wave"] = np.asarray(value, dtype=float).reshape(-1)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"internalcs", "internalcolorspace"}:
        ip.fields["internal_cs"] = str(value)
        return track_ip_session_state(session, ip)
    if key in {"ics2display", "ics2displaytransform", "internalcs2displayspace"}:
        ip.data["transforms"][2] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"demosaicstructure", "demosaic"}:
        ip.fields["demosaic"] = dict(value)
        ip.fields["demosaic_method"] = str(ip.fields["demosaic"].get("method", "none"))
        return track_ip_session_state(session, ip)
    if key == "demosaicmethod":
        method = "none" if value in {None, ""} else str(value).lower()
        ip.fields["demosaic_method"] = method
        ip.fields["demosaic"]["method"] = method
        return track_ip_session_state(session, ip)
    if key in {"sensorconversion", "conversionsensor"}:
        ip.fields["sensor_correction"] = dict(value)
        ip.fields["conversion_method_sensor"] = str(
            ip.fields["sensor_correction"].get("method", "none")
        )
        return track_ip_session_state(session, ip)
    if key in {"sensorconversionmethod", "conversionmethodsensor", "colorconversionmethod"}:
        method = "none" if value in {None, ""} else str(value)
        ip.fields["conversion_method_sensor"] = method
        ip.fields["sensor_correction"]["method"] = method
        return track_ip_session_state(session, ip)
    if key in {"sensorconversionmatrix", "conversiontransformsensor", "conversionmatrixsensor"}:
        ip.data["transforms"][0] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"illuminantcorrection", "correctionilluminant"}:
        ip.fields["illuminant_correction"] = dict(value)
        ip.fields["illuminant_correction_method"] = str(
            ip.fields["illuminant_correction"].get("method", "none")
        )
        return track_ip_session_state(session, ip)
    if key in {"illuminantcorrectionmethod", "correctionmethodilluminant", "colorbalancemethod"}:
        method = "none" if value in {None, ""} else str(value).lower()
        ip.fields["illuminant_correction_method"] = method
        ip.fields["illuminant_correction"]["method"] = method
        return track_ip_session_state(session, ip)
    if key in {
        "correctionmatrixilluminant",
        "illuminantcorrectionmatrix",
        "illuminantcorrectiontransform",
        "correctiontransformilluminant",
    }:
        ip.data["transforms"][1] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"display", "displaystructure"}:
        ip.fields["display"] = track_session_object(session, value)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"data", "datastructure"}:
        ip.data = dict(value)
        return track_ip_session_state(session, _ensure_ip_state(ip))
    if key in {"input", "sensorinput"}:
        ip.data["input"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"result", "displaylinearrgb"}:
        ip.data["result"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key in {"datawhitepoint", "datawp"}:
        ip.data["wp"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key == "sensorspace":
        ip.data["sensorspace"] = np.asarray(value, dtype=float)
        return track_ip_session_state(session, ip)
    if key == "quantization":
        ip.data["quantization"] = value
        return track_ip_session_state(session, ip)
    if key in {"nbits", "quantizationnbits"}:
        ip.data.setdefault("quantization", {})
        if not isinstance(ip.data["quantization"], dict):
            ip.data["quantization"] = {"method": ip.data["quantization"]}
        ip.data["quantization"]["bits"] = int(value)
        return track_ip_session_state(session, ip)
    if key == "transforms":
        if args:
            index = int(args[0]) - 1
            ip.data["transforms"][index] = np.asarray(value, dtype=float)
        else:
            transforms = list(value)
            while len(transforms) < 3:
                transforms.append(None)
            ip.data["transforms"] = transforms[:3]
        return track_ip_session_state(session, ip)
    if key == "transformmethod":
        ip.fields["transform_method"] = str(value).lower()
        return track_ip_session_state(session, ip)
    if key in {"datamax", "rgbmax", "sensormax", "maximumsensorvalue", "maximumsensorvoltageswing"}:
        ip.fields["datamax"] = float(value)
        return track_ip_session_state(session, ip)
    if key in {"render", "renderstructure"}:
        ip.fields["render"] = dict(value)
        ip.fields["render"].setdefault("renderflag", 1)
        ip.fields["render"].setdefault("scale", True)
        ip.fields["render"].setdefault("whitept", False)
        return track_ip_session_state(session, ip)
    if key in {"renderflag", "displaymode"}:
        normalized = param_format(value)
        mapping = {"rgb": 1, "hdr": 2, "gray": 3}
        ip.fields["render"]["renderflag"] = mapping.get(
            normalized, int(value) if isinstance(value, (int, np.integer)) else 1
        )
        return track_ip_session_state(session, ip)
    if key in {"combineexposures", "combinationmethod"}:
        method = str(value)
        ip.fields["combine_exposures"] = method
        ip.fields["combination_method"] = method
        return track_ip_session_state(session, ip)
    if key in {"renderscale", "scaledisplay", "scaledisplayoutput"}:
        ip.fields["render"]["scale"] = bool(value)
        return track_ip_session_state(session, ip)
    if key == "renderwhitept":
        if isinstance(value, (bool, np.bool_)) and not bool(value):
            ip.fields["render"]["whitept"] = False
            return track_ip_session_state(session, ip)
        illuminant = np.asarray(value, dtype=float).reshape(-1)
        if not args:
            raise ValueError("ipSet(..., 'renderwhitept', illuminant, sensor_qe) requires sensor data.")
        sensor_arg = args[0]
        if isinstance(sensor_arg, Sensor):
            sensor_qe = np.asarray(sensor_get(sensor_arg, "spectral qe"), dtype=float)
        else:
            sensor_qe = np.asarray(sensor_arg, dtype=float)
        if sensor_qe.ndim != 2 or sensor_qe.shape[0] != illuminant.size:
            raise ValueError("sensor_qe must be wave x channels and match the illuminant length.")
        sensor_light = illuminant.reshape(1, -1) @ sensor_qe
        sensor_light = np.asarray(sensor_light, dtype=float).reshape(-1)
        sensor_light /= max(float(np.max(sensor_light)), 1.0e-12)
        transform = np.asarray(ip_get(ip, "sensor conversion matrix"), dtype=float)
        sensor_white = sensor_light @ transform
        transform = transform @ np.diag(1.0 / np.maximum(sensor_white, 1.0e-12))
        ip = ip_set(ip, "sensor conversion matrix", transform, session=session)
        ip = ip_set(ip, "transform method", "current", session=session)
        ip.fields["render"]["whitept"] = True
        return track_ip_session_state(session, ip)
    if key in {"gammadisplay", "rendergamma", "gamma"}:
        ip.fields["render"]["gamma"] = value
        return track_ip_session_state(session, ip)
    if key == "renderdemosaiconly" and bool(value):
        ip = ip_set(ip, "internal cs", "Sensor", session=session)
        ip = ip_set(ip, "conversion method sensor", "None", session=session)
        ip = ip_set(ip, "correction method illuminant", "None", session=session)
        ip = ip_set(ip, "transform method", "current", session=session)
        ip = ip_set(ip, "ics2display transform", _identity_transform(), session=session)
        return track_ip_session_state(session, ip)
    raise KeyError(f"Unsupported ipSet parameter: {parameter}")


imageDataXYZ = image_data_xyz  # noqa: N816
imageRGB2XYZ = image_rgb_to_xyz  # noqa: N816
displayRender = display_render  # noqa: N816
ipCreate = ip_create  # noqa: N816
ipCompute = ip_compute  # noqa: N816
ipGet = ip_get  # noqa: N816
ipSet = ip_set  # noqa: N816
ipClearData = ip_clear_data  # noqa: N816
vcimageClearData = vcimage_clear_data  # noqa: N816
ipSaveImage = ip_save_image  # noqa: N816
imageShowImage = image_show_image  # noqa: N816
imageMCCTransform = image_mcc_transform  # noqa: N816
imageSensorTransform = image_sensor_transform  # noqa: N816
imageEsserTransform = image_esser_transform  # noqa: N816
ieInternal2Display = ie_internal_to_display  # noqa: N816
ipHDRWhite = ip_hdr_white  # noqa: N816
imageDistort = image_distort  # noqa: N816
Demosaic = demosaic  # noqa: N816
demosaicMultichannel = demosaic_multichannel  # noqa: N816
Pocs = pocs  # noqa: N816
faultyList = faulty_list  # noqa: N816
faultyInsert = faulty_insert  # noqa: N816
FaultyNearestNeighbor = faulty_nearest_neighbor  # noqa: N816
FaultyBilinear = faulty_bilinear  # noqa: N816
LFDefaultVal = lf_default_val  # noqa: N816
LFDefaultField = lf_default_field  # noqa: N816
LFConvertToFloat = lf_convert_to_float  # noqa: N816
LFbuffer2image = lf_buffer_to_image  # noqa: N816
LFImage2buffer = lf_image_to_buffer  # noqa: N816
LFbuffer2SubApertureViews = lf_buffer_to_sub_aperture_views  # noqa: N816
LFToolboxVersion = lf_toolbox_version  # noqa: N816
ip2lightfield = ip_to_lightfield  # noqa: N816
imageSensorConversion = image_sensor_conversion  # noqa: N816
imageSensorCorrection = image_sensor_correction  # noqa: N816
imageIlluminantCorrection = image_illuminant_correction  # noqa: N816
imageColorBalance = image_color_balance  # noqa: N816
ieRadiance2IP = ie_radiance_to_ip  # noqa: N816
vcimageSRGB = vcimage_srgb  # noqa: N816
vcimageISOMTF = vcimage_iso_mtf  # noqa: N816
vcimageVSNR = vcimage_vsnr  # noqa: N816
ipMCCXYZ = ip_mcc_xyz  # noqa: N816
vcimageMCCXYZ = vcimage_mcc_xyz  # noqa: N816
demosaicRCCC = demosaic_rccc  # noqa: N816
LFFiltShiftSum = lf_filt_shift_sum  # noqa: N816
LFAutofocus = lf_autofocus  # noqa: N816
