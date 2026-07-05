"""Headless MATLAB-style ROI extraction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import UnsupportedOptionError
from .ip import image_data_xyz, ip_get
from .optics import oi_get
from .scene import scene_get
from .sensor import sensor_get
from .types import ImageProcessor, OpticalImage, Scene, Sensor
from .utils import param_format, quanta_to_energy, tile_pattern


def _rect_to_locs(rect: Any) -> np.ndarray:
    rect_array = np.rint(np.asarray(rect, dtype=float).reshape(-1)).astype(int)
    if rect_array.size != 4:
        raise ValueError("ROI rect must contain [col, row, width, height].")
    col_min, row_min, width, height = rect_array
    col_max = col_min + width
    row_max = row_min + height
    cols, rows = np.meshgrid(np.arange(col_min, col_max + 1), np.arange(row_min, row_max + 1))
    return np.column_stack((rows.reshape(-1), cols.reshape(-1)))


def ie_rect2_locs(rect: Any) -> np.ndarray:
    """Convert an ISET rect `[col, row, width, height]` to ROI locations."""

    return _rect_to_locs(rect)


def vc_rect2_locs(rect: Any) -> np.ndarray:
    """Obsolete MATLAB alias for `ieRect2Locs`."""

    return ie_rect2_locs(rect)


def ie_roi2_locs(rect: Any) -> np.ndarray:
    """Deprecated MATLAB alias for `ieRect2Locs`."""

    return ie_rect2_locs(rect)


def ie_locs2_rect(roi_locs: Any) -> np.ndarray:
    """Convert ROI locations to `[colMin, rowMin, width, height]` rect form."""

    locs = np.asarray(roi_locs, dtype=float)
    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError("Expecting roiLocs as Nx2.")
    rounded = np.rint(locs).astype(int)
    rect = np.zeros(4, dtype=int)
    rect[0] = int(np.min(rounded[:, 1]))
    rect[1] = int(np.min(rounded[:, 0]))
    rect[2] = int(np.max(rounded[:, 1]) - rect[0])
    rect[3] = int(np.max(rounded[:, 0]) - rect[1])
    return rect


def ie_rect2_vertices(rect: Any, close_flag: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Convert an ISET rect to x/y vertex vectors."""

    rect_array = np.rint(np.asarray(rect, dtype=float).reshape(-1)).astype(int)
    if rect_array.size != 4:
        raise ValueError("ROI rect must contain [col, row, width, height].")
    xv = np.array(
        [rect_array[0], rect_array[0], rect_array[0] + rect_array[2], rect_array[0] + rect_array[2]],
        dtype=int,
    )
    yv = np.array(
        [rect_array[1], rect_array[1] + rect_array[3], rect_array[1] + rect_array[3], rect_array[1]],
        dtype=int,
    )
    if close_flag:
        xv = np.concatenate([xv, xv[:1]])
        yv = np.concatenate([yv, yv[:1]])
    return xv, yv


def _normalize_roi_locs(roi_locs: Any) -> np.ndarray:
    locs = np.asarray(roi_locs, dtype=float)
    if locs.ndim == 1:
        if locs.size == 4:
            return _rect_to_locs(locs)
        if locs.size == 2:
            return np.rint(locs.reshape(1, 2)).astype(int)
    if locs.ndim == 2:
        if locs.shape == (1, 4):
            return _rect_to_locs(locs.reshape(-1))
        if locs.shape[1] == 2:
            return np.rint(locs).astype(int)
    raise ValueError("ROI locations must be an Nx2 array or [col, row, width, height] rect.")


def _clip_roi_locs(roi_locs: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError("ROI source data must have positive rows and columns.")
    clipped = np.asarray(roi_locs, dtype=int).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 1, rows)
    clipped[:, 1] = np.clip(clipped[:, 1], 1, cols)
    return clipped


def _extract_roi_rows(data: np.ndarray, roi_locs: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError("ROI source data must be a 2D image or 3D image cube.")
    if roi_locs.size == 0:
        return np.empty((0, array.shape[2]), dtype=array.dtype)
    row_index = roi_locs[:, 0] - 1
    col_index = roi_locs[:, 1] - 1
    return np.asarray(array[row_index, col_index, :])


def _broadcast_spectrum(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return np.broadcast_to(array.reshape(1, 1, -1), (rows, cols, array.size)).copy()
    return array


def _scene_roi_data(scene: Scene, roi_locs: np.ndarray, data_type: str | None) -> np.ndarray:
    key = param_format(data_type or "photons")
    rows, cols = scene_get(scene, "size")
    wave = np.asarray(scene_get(scene, "wave"), dtype=float)

    if key in {"photons", "radiancephotons"}:
        data = np.asarray(scene_get(scene, "photons"), dtype=float)
    elif key in {"energy", "radianceenergy"}:
        data = quanta_to_energy(np.asarray(scene_get(scene, "photons"), dtype=float), wave)
    elif key == "illuminantphotons":
        data = _broadcast_spectrum(np.asarray(scene_get(scene, "illuminant photons"), dtype=float), rows, cols)
    elif key == "illuminantenergy":
        data = _broadcast_spectrum(np.asarray(scene_get(scene, "illuminant energy"), dtype=float), rows, cols)
    elif key in {"luminance", "luminanceroi"}:
        data = np.asarray(scene_get(scene, "luminance"), dtype=float)
    elif key == "reflectance":
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        illuminant = _broadcast_spectrum(np.asarray(scene_get(scene, "illuminant photons"), dtype=float), rows, cols)
        data = np.divide(photons, illuminant, out=np.zeros_like(photons), where=illuminant > 0.0)
    else:
        raise UnsupportedOptionError("vcGetROIData", f"scene/{data_type}")

    clipped = _clip_roi_locs(roi_locs, rows, cols)
    return _extract_roi_rows(data, clipped)


def _oi_roi_data(oi: OpticalImage, roi_locs: np.ndarray, data_type: str | None) -> np.ndarray:
    key = param_format(data_type or "photons")
    photons = np.asarray(oi_get(oi, "photons"), dtype=float)
    rows, cols = photons.shape[:2]
    if key == "photons":
        data = photons
    elif key == "energy":
        data = quanta_to_energy(photons, np.asarray(oi_get(oi, "wave"), dtype=float))
    elif key == "illuminance":
        data = np.asarray(oi_get(oi, "illuminance"), dtype=float)
    else:
        raise UnsupportedOptionError("vcGetROIData", f"oi/{data_type}")

    clipped = _clip_roi_locs(roi_locs, rows, cols)
    return _extract_roi_rows(data, clipped)


def _sensor_signal_cube(sensor: Sensor, data_type: str | None) -> np.ndarray:
    key = param_format(data_type or "volts")
    if key in {"volts", "dvorvolts"}:
        data = sensor_get(sensor, "volts")
    elif key in {"dv", "digitalvalues"}:
        data = sensor_get(sensor, "dv")
    elif key == "electrons":
        data = sensor_get(sensor, "electrons")
        if data is None:
            raise ValueError("Sensor has no computed volts for ROI extraction.")
    else:
        raise UnsupportedOptionError("vcGetROIData", f"sensor/{data_type}")

    if data is None:
        raise ValueError(f"Sensor has no {data_type or 'volts'} data for ROI extraction.")
    array = np.asarray(data, dtype=float)
    if array.ndim == 3:
        return array
    if array.ndim != 2:
        raise ValueError("Sensor ROI extraction expects 2D or 3D sensor data.")

    nfilters = int(sensor_get(sensor, "nfilters"))
    if nfilters <= 1:
        return array[:, :, np.newaxis]

    pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), array.shape[0], array.shape[1])
    cube = np.full((array.shape[0], array.shape[1], nfilters), np.nan, dtype=float)
    for filter_index in range(nfilters):
        mask = pattern == (filter_index + 1)
        cube[:, :, filter_index][mask] = array[mask]
    return cube


def _sensor_roi_data(sensor: Sensor, roi_locs: np.ndarray, data_type: str | None) -> np.ndarray:
    cube = _sensor_signal_cube(sensor, data_type)
    clipped = _clip_roi_locs(roi_locs, cube.shape[0], cube.shape[1])
    return _extract_roi_rows(cube, clipped)


def _ip_roi_data(ip: ImageProcessor, roi_locs: np.ndarray, data_type: str | None) -> np.ndarray:
    key = param_format(data_type or "results")
    if key in {"result", "results"}:
        data = ip_get(ip, "result")
    elif key == "input":
        data = ip_get(ip, "input")
    elif key == "xyz":
        data = image_data_xyz(ip)
    else:
        raise UnsupportedOptionError("vcGetROIData", f"ip/{data_type}")
    if data is None:
        raise ValueError(f"IP has no {data_type or 'results'} data for ROI extraction.")
    array = np.asarray(data, dtype=float)
    clipped = _clip_roi_locs(roi_locs, array.shape[0], array.shape[1])
    return _extract_roi_rows(array, clipped)


def vc_get_roi_data(
    obj: Scene | OpticalImage | Sensor | ImageProcessor,
    roi_locs: Any,
    data_type: str | None = None,
) -> np.ndarray:
    """Extract ROI data in MATLAB-style XW row format."""

    normalized_locs = _normalize_roi_locs(roi_locs)
    if isinstance(obj, Scene):
        return _scene_roi_data(obj, normalized_locs, data_type)
    if isinstance(obj, OpticalImage):
        return _oi_roi_data(obj, normalized_locs, data_type)
    if isinstance(obj, Sensor):
        return _sensor_roi_data(obj, normalized_locs, data_type)
    if isinstance(obj, ImageProcessor):
        return _ip_roi_data(obj, normalized_locs, data_type)
    raise UnsupportedOptionError("vcGetROIData", getattr(obj, "type", type(obj).__name__))


ieRect2Locs = ie_rect2_locs
vcRect2Locs = vc_rect2_locs
ieRoi2Locs = ie_roi2_locs
ieLocs2Rect = ie_locs2_rect
ieRect2Vertices = ie_rect2_vertices
imageDataXYZ = image_data_xyz
ipGet = ip_get
oiGet = oi_get
Quanta2Energy = quanta_to_energy
sceneGet = scene_get
sensorGet = sensor_get
vcGetROIData = vc_get_roi_data
