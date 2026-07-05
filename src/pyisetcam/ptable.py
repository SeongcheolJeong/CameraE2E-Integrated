"""Headless parameter-table helpers modeled on MATLAB iePTable."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from .camera import camera_get
from .display import display_get
from .ip import ip_get
from .optics import oi_create, oi_get
from .scene import scene_get
from .sensor import sensor_get
from .types import Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor
from .utils import param_format


@dataclass
class IEPTable:
    """Headless representation of an iePTable result."""

    title: str
    columns: tuple[str, ...]
    data: list[tuple[str, ...]]
    format: str
    font_size: int = 14
    tag: str = "IEPTable Table"


def _column_length(column: Any) -> int:
    if isinstance(column, (str, bytes, bytearray)):
        raise TypeError("Table columns must be sequence-like, not scalar strings.")
    return len(column)


def _column_value(column: Any, index: int) -> Any:
    if isinstance(column, np.ndarray):
        return column[index]
    return column[index]


def _slice_column(column: Any, indices: list[int]) -> Any:
    if isinstance(column, np.ndarray):
        return np.asarray(column)[indices]
    if isinstance(column, tuple):
        return tuple(column[index] for index in indices)
    if isinstance(column, list):
        return [column[index] for index in indices]
    return [column[index] for index in indices]


def _value_matches(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    if np.isscalar(left) and np.isscalar(right):
        return bool(left == right)
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _normalize_table(
    table: Any,
) -> tuple[list[dict[str, Any]], Callable[[list[int]], Any], dict[str, str]]:
    if isinstance(table, Mapping):
        normalized_names = {param_format(key): str(key) for key in table}
        lengths = {_column_length(column) for column in table.values()}
        if len(lengths) > 1:
            raise ValueError("All table columns must have the same length.")
        count = lengths.pop() if lengths else 0
        rows = [
            {
                normalized: _column_value(table[original], index)
                for normalized, original in normalized_names.items()
            }
            for index in range(count)
        ]

        def subset(indices: list[int]) -> dict[str, Any]:
            return {
                original: _slice_column(table[original], indices)
                for original in table
            }

        return rows, subset, normalized_names

    if isinstance(table, Sequence) and not isinstance(table, (str, bytes, bytearray)):
        rows: list[dict[str, Any]] = []
        normalized_names: dict[str, str] = {}
        for row in table:
            if not isinstance(row, Mapping):
                raise TypeError("Sequence-based tables must contain mapping rows.")
            normalized_row = {param_format(key): value for key, value in row.items()}
            rows.append(normalized_row)
            for key in row:
                normalized_names.setdefault(param_format(key), str(key))

        def subset(indices: list[int]) -> list[Any]:
            return [table[index] for index in indices]

        return rows, subset, normalized_names

    raise TypeError("Unsupported table type. Use a mapping of columns or a sequence of row mappings.")


def _format_scalar(value: Any, precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    if np.isscalar(value):
        numeric = float(value)
        if np.isclose(numeric, round(numeric)):
            return str(int(round(numeric)))
        return f"{numeric:.{precision}f}"
    array = np.asarray(value)
    if array.size == 0:
        return ""
    return np.array2string(array, precision=precision, separator=" ", suppress_small=False)


def ie_table_get(
    table: Any,
    *args: Any,
    operator: str = "and",
    return_type: str = "table rows",
) -> tuple[list[Any], Any]:
    """Return rows/files matching MATLAB-style metadata-table conditions."""

    if len(args) % 2 != 0:
        raise ValueError("ieTableGet expects key/value pairs.")

    rows, subset, normalized_names = _normalize_table(table)
    pairs = param_format(list(args))
    op = param_format(operator)
    result_kind = param_format(return_type)
    conditions: list[tuple[str, Any]] = []

    for index in range(0, len(pairs), 2):
        key = pairs[index]
        value = args[index + 1]
        if key == "operator":
            op = param_format(value)
            continue
        if key == "return":
            result_kind = param_format(value)
            continue
        if key not in normalized_names:
            raise ValueError(f"{key} is not a table variable name.")
        conditions.append((key, value))

    if op not in {"and", "or"}:
        raise ValueError(f"Unknown operator {operator}")
    if result_kind not in {"tablerows", "rows", "table", "file", "files"}:
        raise ValueError(f"Unknown return type {return_type}")

    selected: list[int] | None = None
    for field, value in conditions:
        matches = [index for index, row in enumerate(rows) if _value_matches(row.get(field), value)]
        if op == "and":
            selected = matches if selected is None else [index for index in selected if index in matches]
        else:
            selected = sorted(set([] if selected is None else selected).union(matches))

    final_indices = [] if selected is None else selected
    filtered_rows = subset(final_indices)
    files = [rows[index].get("file") for index in final_indices] if "file" in normalized_names else []
    return files, filtered_rows


def _wave_summary(wave: Any) -> str:
    array = np.asarray(wave, dtype=float).reshape(-1)
    if array.size == 0:
        return ""
    if array.size == 1:
        return _format_scalar(array[0], 0)
    step = array[1] - array[0]
    return f"{_format_scalar(array[0], 0)} {_format_scalar(array[-1], 0)} {_format_scalar(step, 0)}"


def _dynamic_range(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    positive = array[array > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.max(positive) / max(np.min(positive), 1e-12))


def _diagonal_fov_deg(width_m: float, height_m: float, distance_m: float) -> float:
    half_diag = float(np.hypot(width_m / 2.0, height_m / 2.0))
    return float(np.rad2deg(2.0 * np.arctan2(half_diag, max(distance_m, 1e-12))))


def _window_row(label: str, value: Any, units: str = "", precision: int = 3) -> tuple[str, str, str]:
    return (label, _format_scalar(value, precision), units)


def _embed_row(label: str, value: Any, precision: int = 3) -> tuple[str, str]:
    return (label, _format_scalar(value, precision))


def _combine_rows(*groups: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for group in groups:
        rows.extend(group)
    return rows


def _infer_table_type(obj: Any) -> str:
    if isinstance(obj, Scene):
        return "scene"
    if isinstance(obj, OpticalImage):
        return "oi"
    if isinstance(obj, Sensor):
        return "sensor"
    if isinstance(obj, ImageProcessor):
        return "ip"
    if isinstance(obj, Display):
        return "display"
    if isinstance(obj, Camera):
        return "camera"
    if isinstance(obj, dict):
        if {"focal_length_m", "f_number"} <= set(obj):
            return "optics"
        if {"size_m", "fill_factor", "voltage_swing"} <= set(obj):
            return "pixel"
    raise ValueError(f"Unsupported iePTable object type: {type(obj).__name__}")


def _table_scene(scene: Scene, table_format: str) -> tuple[str, list[tuple[str, ...]]]:
    width_m = float(scene_get(scene, "width"))
    height_m = float(scene_get(scene, "height"))
    distance_m = float(scene_get(scene, "distance"))
    hfov = float(scene_get(scene, "hfov"))
    diag_fov = _diagonal_fov_deg(width_m, height_m, distance_m)
    rows, cols = scene_get(scene, "size")
    sample_spacing_mm = np.array([height_m / max(rows, 1), width_m / max(cols, 1)]) * 1e3
    angular_resolution = hfov / max(cols, 1)
    illuminant_name = scene.fields.get("illuminant_name", scene.fields.get("illuminant_format", ""))

    if table_format == "window":
        rows_data = [
            _window_row("Name", scene_get(scene, "name")),
            _window_row("Hor, Dia fov", f"{hfov:.4f} {diag_fov:.4f}", "deg"),
            _window_row("Rows & columns", scene_get(scene, "size"), "samples"),
            _window_row("Height & Width", np.array([height_m, width_m]) * 1e3, "mm", 4),
            _window_row("Distance", distance_m, "meters", 4),
            _window_row("Angular resolution", angular_resolution, "deg/samp", 4),
            _window_row("Sample spacing", sample_spacing_mm, "mm/sample", 4),
            _window_row("Wave - min,max,delta (nm)", _wave_summary(scene_get(scene, "wave")), "nm"),
            _window_row("Mean luminance", scene_get(scene, "mean luminance"), "cd/m^2 (nits)", 4),
            _window_row("Dynamic range (linear)", _dynamic_range(scene_get(scene, "luminance")), "linear", 2),
            _window_row("Illuminant name", illuminant_name),
            _window_row("Depth range", scene_get(scene, "depth range"), "m", 3),
        ]
    else:
        rows_data = [
            _embed_row("Hor, Dia fov", f"{hfov:.2f} {diag_fov:.2f}"),
            _embed_row("Rows/cols", scene_get(scene, "size")),
            _embed_row("Hght/Width (mm)", np.array([height_m, width_m]) * 1e3, 2),
            _embed_row("Distance (m)", distance_m, 2),
            _embed_row("Angular res (deg/samp)", angular_resolution, 2),
            _embed_row("Samp space (mm/sample)", sample_spacing_mm, 2),
            _embed_row("Wave (nm)", _wave_summary(scene_get(scene, "wave"))),
            _embed_row("Mean luminance (cd/m^2)", scene_get(scene, "mean luminance"), 2),
            _embed_row("Dynamic range (linear)", _dynamic_range(scene_get(scene, "luminance")), 2),
            _embed_row("Illuminant name", illuminant_name),
        ]
    return " for a Scene", rows_data


def _table_optics(optics: dict[str, Any], table_format: str) -> list[tuple[str, ...]]:
    focal_length_mm = float(optics.get("focal_length_m", 0.0)) * 1e3
    f_number = float(optics.get("f_number", 0.0))
    aperture_mm = focal_length_mm / max(f_number, 1e-12)
    model = str(optics.get("model", ""))
    name = str(optics.get("name", ""))

    if table_format == "window":
        return [
            _window_row("Optics model", model),
            _window_row("Optics name", name),
            _window_row("Focal length", focal_length_mm, "mm", 3),
            _window_row("F-number", f_number, "dimensionless", 1),
            _window_row("Aperture diameter", aperture_mm, "mm", 3),
        ]
    return [
        _embed_row("Optics model", f"{name}-{model}" if name else model),
        _embed_row("Focal length (mm)", focal_length_mm, 1),
        _embed_row("F-number", f_number, 1),
        _embed_row("Aperture diameter (mm)", aperture_mm, 3),
    ]


def _table_oi(oi: OpticalImage, table_format: str) -> tuple[str, list[tuple[str, ...]]]:
    width_m = float(oi_get(oi, "width"))
    height_m = float(oi_get(oi, "height"))
    image_distance = float(oi_get(oi, "image distance"))
    hfov = float(oi_get(oi, "hfov"))
    diag_fov = _diagonal_fov_deg(width_m, height_m, image_distance)
    optics_rows = _table_optics(dict(oi.fields["optics"]), table_format)

    if table_format == "window":
        oi_rows = [
            _window_row("Optical Image name", oi_get(oi, "name")),
            _window_row("Compute method", oi_get(oi, "compute method")),
            _window_row("Rows & cols", oi_get(oi, "size"), "samples"),
            _window_row("Horiz, Diag FOV", f"{hfov:.1f} {diag_fov:.1f}", "deg"),
            _window_row("Wave (nm)", _wave_summary(oi_get(oi, "wave")), "nm"),
            _window_row("Hght,Wdth", np.array([height_m, width_m]) * 1e3, "mm", 3),
            _window_row("Resolution", oi_get(oi, "spatial resolution") * 1e6, "um/sample", 3),
            _window_row("Mean illuminance", oi_get(oi, "mean illuminance"), "lux", 3),
            _window_row("Dynamic range", _dynamic_range(oi_get(oi, "photons")), "linear", 3),
        ]
    else:
        oi_rows = [
            _embed_row("Compute method", oi_get(oi, "compute method")),
            _embed_row("Rows & columns", oi_get(oi, "size")),
            _embed_row("Hor, Dia, FOV (deg)", f"{hfov:.1f} {diag_fov:.1f}"),
            _embed_row("Wave (nm)", _wave_summary(oi_get(oi, "wave"))),
            _embed_row("Resolution (um/sample)", oi_get(oi, "spatial resolution") * 1e6, 1),
            _embed_row("Hght,Wdth (mm)", np.array([height_m, width_m]) * 1e3, 1),
            _embed_row("Mean illuminance (lux)", oi_get(oi, "mean illuminance"), 1),
            _embed_row("Dynamic range", _dynamic_range(oi_get(oi, "photons")), 1),
        ]
    return " for an Optical Image", _combine_rows(optics_rows, oi_rows)


def _table_pixel(pixel: dict[str, Any], table_format: str) -> list[tuple[str, ...]]:
    size_um = np.asarray(pixel.get("size_m", np.zeros(2)), dtype=float) * 1e6
    if table_format == "window":
        return [
            _window_row("-------------------", "------- Pixel -------", "-----------"),
            _window_row("Width & height", size_um, "um", 3),
            _window_row("Fill factor", pixel.get("fill_factor", 0.0), "", 3),
            _window_row("Dark voltage (V/sec)", pixel.get("dark_voltage_v_per_sec", 0.0), "V/sec", 3),
            _window_row("Read noise (V)", pixel.get("read_noise_v", 0.0), "V", 3),
            _window_row("Conversion Gain (V/e-)", pixel.get("conversion_gain_v_per_electron", 0.0), "V/e-", 3),
            _window_row("Voltage Swing (V)", pixel.get("voltage_swing", 0.0), "V", 3),
            _window_row("Well Capacity (e-)", pixel.get("well_capacity_electrons", 0.0), "e-", 3),
        ]
    return [
        _embed_row("Width/height (um)", size_um, 3),
        _embed_row("Fill factor", pixel.get("fill_factor", 0.0), 3),
        _embed_row("Read noise (V)", pixel.get("read_noise_v", 0.0), 3),
        _embed_row("Conversion Gain (V/e-)", pixel.get("conversion_gain_v_per_electron", 0.0), 3),
        _embed_row("Voltage Swing (V)", pixel.get("voltage_swing", 0.0), 3),
        _embed_row("Well Capacity (e-)", pixel.get("well_capacity_electrons", 0.0), 3),
    ]


def _table_sensor(sensor: Sensor, table_format: str, reference_oi: OpticalImage | None = None) -> tuple[str, list[tuple[str, ...]]]:
    oi = reference_oi if reference_oi is not None else oi_create()
    rows_data: list[tuple[str, ...]]
    nbits_value = sensor_get(sensor, "nbits")
    nbits = "analog" if param_format(sensor_get(sensor, "quantization")) == "analog" else str(nbits_value)
    pixel_rows = _table_pixel(dict(sensor.fields["pixel"]), table_format)

    if table_format == "window":
        rows_data = [
            _window_row("-------------------", "------- Sensor -------", "-----------"),
            _window_row("Name", sensor_get(sensor, "name")),
            _window_row("Size", sensor_get(sensor, "dimension", "mm"), "mm", 3),
            _window_row("Rows and Columns", sensor_get(sensor, "size")),
            _window_row("Horizontal FOV", sensor_get(sensor, "fov", 1e6, oi), "deg", 3),
            _window_row("Horizontal Res / distance", sensor_get(sensor, "wspatial resolution", "um"), "um", 3),
            _window_row("Exposure time", sensor_get(sensor, "exp time"), "s", 3),
            _window_row("Bits", nbits, "bits"),
            _window_row("Analog gain", sensor_get(sensor, "analog gain"), "(raw + ao)/ag", 3),
            _window_row("Analog offset", sensor_get(sensor, "analog offset"), "V", 3),
        ]
        rows_data = _combine_rows(rows_data, pixel_rows)
    else:
        rows_data = [
            _embed_row("Size (mm)", sensor_get(sensor, "dimension", "mm"), 3),
            _embed_row("Hor FOV (deg)", sensor_get(sensor, "fov", 1e6, oi), 3),
            _embed_row("Res (um)", sensor_get(sensor, "wspatial resolution", "um"), 3),
            _embed_row("Bits", nbits),
            _embed_row("Analog gain", sensor_get(sensor, "analog gain"), 3),
            _embed_row("Analog offset", sensor_get(sensor, "analog offset"), 3),
        ]
    return " for a Sensor", rows_data


def _table_ip(ip: ImageProcessor, table_format: str) -> tuple[str, list[tuple[str, ...]]]:
    result_size = ip_get(ip, "result size")
    display = ip.fields["display"]
    if table_format == "window":
        rows_data = [
            _window_row("Name", ip_get(ip, "name")),
            _window_row("Rows, Columns, Primaries", result_size),
            _window_row("Demosaic", ip_get(ip, "demosaic method")),
            _window_row("Sensor conversion", ip_get(ip, "sensor conversion method")),
            _window_row("Internal color space", ip_get(ip, "internal color space")),
            _window_row("Illuminant correction", ip_get(ip, "illuminant correction method")),
            _window_row("--------------------", "---------------------", "-------------------"),
            _window_row("Display name", display_get(display, "name")),
            _window_row("Display dpi", display_get(display, "dpi"), "dots per inch"),
            _window_row("Display bits", display_get(display, "bits"), "bits"),
        ]
    else:
        rows_data = [
            _embed_row("Name", ip_get(ip, "name")),
            _embed_row("Row, col, primaries", result_size),
            _embed_row("Demosaic", ip_get(ip, "demosaic method")),
            _embed_row("Sensor conversion", ip_get(ip, "sensor conversion method")),
            _embed_row("Internal color space", ip_get(ip, "internal color space")),
            _embed_row("Illuminant correction", ip_get(ip, "illuminant correction method")),
            _embed_row("--------------------", "---------------------"),
            _embed_row("Display name", display_get(display, "name")),
            _embed_row("Display dpi", display_get(display, "dpi")),
            _embed_row("Display bits", display_get(display, "bits")),
        ]
    return " for an Imaging Pipeline", rows_data


def _table_display(display: Display, table_format: str) -> tuple[str, list[tuple[str, ...]]]:
    if table_format == "window":
        rows_data = [
            _window_row("Name", display_get(display, "name")),
            _window_row("DPI", display_get(display, "dpi")),
            _window_row("DAC size", display_get(display, "dac size")),
        ]
    else:
        rows_data = [
            _embed_row("Name", display_get(display, "name")),
            _embed_row("DPI", display_get(display, "dpi")),
            _embed_row("DAC size", display_get(display, "dac size")),
        ]
    return " for a Display", rows_data


def _table_camera(camera: Camera, table_format: str) -> tuple[str, list[tuple[str, ...]]]:
    oi_rows = _table_oi(camera_get(camera, "oi"), table_format)[1]
    sensor_rows = _table_sensor(camera_get(camera, "sensor"), table_format, reference_oi=camera_get(camera, "oi"))[1]
    ip_rows = _table_ip(camera_get(camera, "ip"), table_format)[1]
    return " for a Camera", _combine_rows(ip_rows, sensor_rows, oi_rows)


def ie_p_table(
    obj: Any,
    *,
    format: str = "window",
    fontsize: int = 14,
    uitable: Any | None = None,
    reference_oi: OpticalImage | None = None,
) -> tuple[IEPTable, None]:
    """Create a headless parameter table for a supported ISET object."""

    del uitable
    table_format = param_format(format)
    if table_format not in {"window", "embed"}:
        raise ValueError(f"Unsupported iePTable format: {format}")

    object_type = _infer_table_type(obj)
    if table_format == "window":
        columns: tuple[str, ...] = ("Property", "Value", "Units")
    else:
        columns = ("Property", "Value")

    if object_type == "scene":
        title_suffix, rows = _table_scene(obj, table_format)
    elif object_type == "oi":
        title_suffix, rows = _table_oi(obj, table_format)
    elif object_type == "optics":
        title_suffix, rows = " for an Optics", _table_optics(obj, table_format)
    elif object_type == "sensor":
        title_suffix, rows = _table_sensor(obj, table_format, reference_oi=reference_oi)
    elif object_type == "pixel":
        title_suffix, rows = " for a Pixel", _table_pixel(obj, table_format)
    elif object_type == "ip":
        title_suffix, rows = _table_ip(obj, table_format)
    elif object_type == "display":
        title_suffix, rows = _table_display(obj, table_format)
    elif object_type == "camera":
        title_suffix, rows = _table_camera(obj, table_format)
    else:
        raise ValueError(f"Unsupported iePTable object type: {object_type}")

    table = IEPTable(
        title=f"ISET Parameter Table{title_suffix}",
        columns=columns,
        data=rows,
        format=table_format,
        font_size=int(fontsize),
    )
    return table, None


# MATLAB-style alias.
iePTable = ie_p_table
ieTableGet = ie_table_get
