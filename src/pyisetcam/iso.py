"""ISO 12233-oriented helpers used by metrics tutorials."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d

from .ip import ip_get
from .sensor import sensor_get
from .types import ImageProcessor, Sensor


@dataclass
class ISO12233Result:
    """Headless result payload for the slanted-edge ISO 12233 workflow."""

    freq: NDArray[np.float64]
    mtf: NDArray[np.float64]
    nyquistf: float
    lsf: NDArray[np.float64]
    lsfx: NDArray[np.float64]
    mtf50: float
    aliasingPercentage: float
    esf: NDArray[np.float64] | None = None
    fitme: NDArray[np.float64] | None = None
    rect: NDArray[np.int_] | None = None
    win: None = None


def ie_cxcorr(a: Any, b: Any) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Circular cross correlation using MATLAB ieCXcorr() semantics."""

    first = np.asarray(a, dtype=float).reshape(-1)
    second = np.asarray(b, dtype=float).reshape(-1)
    if first.size != second.size:
        raise ValueError("Vector length mismatch.")

    norm_first = max(float(np.linalg.norm(first)), 1.0e-12)
    norm_second = max(float(np.linalg.norm(second)), 1.0e-12)
    first = first / norm_first
    second = second / norm_second

    xc = np.empty(first.size, dtype=float)
    shifted = second.copy()
    for index in range(first.size):
        xc[index] = float(np.dot(first, shifted))
        shifted = np.roll(shifted, 1)

    return xc, np.arange(first.size, dtype=int)


def _matlab_round_scalar(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    radius = (int(size) - 1) / 2.0
    sample = np.arange(-radius, radius + 1.0, dtype=float)
    xx, yy = np.meshgrid(sample, sample, indexing="xy")
    kernel = np.exp(-((xx**2 + yy**2) / max(2.0 * float(sigma) ** 2, 1.0e-12)))
    return kernel / max(float(np.sum(kernel, dtype=float)), 1.0e-12)


def _ahamming(length: int, midpoint: float) -> NDArray[np.float64]:
    midpoint = float(midpoint) + 0.5
    width = max(midpoint - 1.0, float(length) - midpoint)
    support = np.arange(1, int(length) + 1, dtype=float)
    data = np.cos(np.pi * (support - midpoint) / max(width, 1.0e-12))
    return np.asarray(0.54 + (0.46 * data), dtype=float)


def _rect_to_slices(rect: Any) -> tuple[slice, slice]:
    rect_array = np.asarray(rect, dtype=int).reshape(-1)
    if rect_array.size != 4:
        raise ValueError("Rect must contain [col, row, width, height].")
    col_min, row_min, width, height = rect_array
    row_slice = slice(max(int(row_min) - 1, 0), max(int(row_min + height), 0))
    col_slice = slice(max(int(col_min) - 1, 0), max(int(col_min + width), 0))
    return row_slice, col_slice


def _bar_image_from_cube(cube: Any, rect: Any) -> NDArray[np.float64]:
    array = np.asarray(cube, dtype=float)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError("Expected a 2D image or 3D image cube.")
    row_slice, col_slice = _rect_to_slices(rect)
    return np.asarray(array[row_slice, col_slice, :], dtype=float)


def _weighted_luminance(image: NDArray[np.float64], weight: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(image, dtype=float)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError("Expected a 2D image or 3D image cube.")
    if array.shape[2] == 1:
        return array[:, :, 0]
    usable = min(array.shape[2], weight.size)
    return np.tensordot(array[:, :, :usable], weight[:usable], axes=([2], [0]))


def _pb_centroid(values: Any) -> float:
    vector = np.asarray(values, dtype=float).reshape(-1)
    support = np.arange(1, vector.size + 1, dtype=float)
    total = float(np.sum(vector, dtype=float))
    if abs(total) <= 1.0e-12:
        return float((vector.size + 1) / 2.0)
    return float(np.sum(support * vector, dtype=float) / total)


def _cent(values: Any, center: int) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    count = vector.size
    shifted = np.zeros(count, dtype=float)
    midpoint = _matlab_round_scalar((count + 1) / 2.0)
    delta = _matlab_round_scalar(float(center) - float(midpoint))
    if delta > 0:
        shifted[: count - delta] = vector[delta:]
    else:
        shifted[-delta:] = vector[: count + delta]
    return shifted


def _deriv1(data: Any, n_rows: int, n_cols: int, kernel: Any) -> NDArray[np.float64]:
    array = np.asarray(data, dtype=float).reshape(int(n_rows), int(n_cols))
    coeffs = np.asarray(kernel, dtype=float).reshape(-1)
    derivative = np.zeros((int(n_rows), int(n_cols)), dtype=float)
    for row_index in range(int(n_rows)):
        derivative[row_index, :] = np.convolve(array[row_index, :], coeffs, mode="same")
        if int(n_cols) > 1:
            derivative[row_index, 0] = derivative[row_index, 1]
            derivative[row_index, int(n_cols) - 1] = derivative[row_index, int(n_cols) - 2]
    return derivative


def _polyfit_convert(coeffs: Any, x_values: Any) -> NDArray[np.float64]:
    scaled = np.asarray(coeffs, dtype=float).reshape(-1)
    support = np.asarray(x_values, dtype=float).reshape(-1)
    degree = scaled.size - 1
    mean = float(np.mean(support, dtype=float))
    std = float(np.std(support, ddof=1)) if support.size > 1 else 1.0
    if std <= 1.0e-12:
        std = 1.0
    unscaled = np.zeros_like(scaled, dtype=float)
    for i in range(degree + 1):
        for j in range(i + 1):
            unscaled[degree - j] += (
                scaled[degree - i] * comb(i, j) * ((-mean) ** (i - j)) / (std**i)
            )
    return unscaled


def _findedge2(centroids: Any, n_rows: int, degree: int = 1) -> NDArray[np.float64]:
    support = np.arange(int(n_rows), dtype=float)
    mean = float(np.mean(support, dtype=float))
    std = float(np.std(support, ddof=1)) if support.size > 1 else 1.0
    if std <= 1.0e-12:
        std = 1.0
    scaled_support = (support - mean) / std
    scaled_fit = np.polyfit(scaled_support, np.asarray(centroids, dtype=float).reshape(-1), deg=int(degree))
    return _polyfit_convert(scaled_fit, support)


def _project2(data: Any, fit: Any, oversample: int = 4) -> NDArray[np.float64]:
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError("Projection expects a 2D channel image.")
    n_rows, n_cols = image.shape
    fit_vector = np.asarray(fit, dtype=float).reshape(-1)
    nn = int(np.floor(float(n_cols) * float(oversample)))
    slope = 1.0 / max(float(fit_vector[-2]), 1.0e-12)
    offset = _matlab_round_scalar(float(oversample) * (0.0 - ((n_rows - 1) / slope)))
    offset_abs = abs(offset)
    if offset > 0:
        offset = 0
    bwidth = int(nn + offset_abs + 150)
    bins = np.zeros((2, bwidth), dtype=float)

    offsets = np.zeros(n_rows, dtype=float)
    for row_index in range(n_rows):
        y_value = float(row_index)
        offsets[row_index] = float(np.polyval(fit_vector, y_value) - fit_vector[-1])

    for col_index in range(n_cols):
        x_value = float(col_index)
        for row_index in range(n_rows):
            bin_index = int(np.ceil((x_value - offsets[row_index]) * float(oversample))) + 1 - int(offset)
            bin_index = min(max(bin_index, 1), bwidth)
            bins[0, bin_index - 1] += 1.0
            bins[1, bin_index - 1] += image[row_index, col_index]

    start = 1 + _matlab_round_scalar(0.5 * float(offset_abs))
    for index in range(start, start + nn):
        if bins[0, index - 1] != 0.0:
            continue
        if index == 1:
            bins[:, index - 1] = bins[:, index]
        elif index == (start + nn - 1):
            bins[:, index - 1] = bins[:, index - 2]
        else:
            bins[:, index - 1] = 0.5 * (bins[:, index - 2] + bins[:, index])

    point = np.zeros(nn, dtype=float)
    for index in range(nn):
        point[index] = bins[1, index + start - 1] / max(bins[0, index + start - 1], 1.0e-12)
    return point


def _rotatev2(data: Any) -> tuple[NDArray[np.float64], int, int, int]:
    array = np.asarray(data, dtype=float)
    n_rows = int(array.shape[0])
    n_cols = int(array.shape[1])
    channel_index = 1 if array.ndim == 3 and array.shape[2] > 1 else 0
    band = array[:, :, channel_index] if array.ndim == 3 else array
    top_index = min(2, max(n_rows - 1, 0))
    bottom_index = max(n_rows - 4, 0)
    left_index = min(2, max(n_cols - 1, 0))
    right_index = max(n_cols - 4, 0)
    test_vertical = abs(float(np.mean(band[bottom_index, :], dtype=float) - np.mean(band[top_index, :], dtype=float)))
    test_horizontal = abs(float(np.mean(band[:, right_index], dtype=float) - np.mean(band[:, left_index], dtype=float)))

    rotated = 0
    if test_vertical > test_horizontal:
        array = np.rot90(array, k=1, axes=(0, 1)).copy()
        n_rows, n_cols = n_cols, n_rows
        rotated = 1
    return np.asarray(array, dtype=float), int(n_rows), int(n_cols), rotated


def iso_find_slanted_bar(ip: ImageProcessor, blur_flag: bool = False) -> NDArray[np.int_]:
    """Return a good slanted-bar ROI rectangle following MATLAB ISOFindSlantedBar()."""

    image = np.asarray(ip_get(ip, "data display"), dtype=float)
    if image.ndim != 3:
        raise ValueError("ISOFindSlantedBar expects an RGB-format image-processor result.")

    monochrome = np.sum(image, axis=2, dtype=float)
    monochrome = monochrome / max(float(np.max(monochrome)), 1.0e-12)

    black_border = True
    if blur_flag:
        size = monochrome.shape
        kernel_size = max(_matlab_round_scalar(min(size) / 10.0), 1)
        kernel = _gaussian_kernel(kernel_size, max(float(np.rint(kernel_size * 0.7)), 1.0))
        monochrome = convolve2d(monochrome, kernel, mode="same")
        black_border = False

    y_sums = np.sum(monochrome, axis=1, dtype=float)
    dy_sums = np.diff(y_sums)
    below_average = y_sums < (float(np.mean(y_sums)) / 5.0)
    remove = below_average[:-1] & below_average[1:]
    dy_sums = dy_sums.copy()
    dy_sums[remove] = 0.0

    if float(np.max(np.abs(dy_sums))) > 5.0 and black_border:
        row_max = int(np.argmin(dy_sums)) + 1
        row_min = int(np.argmax(dy_sums[: max(row_max - 4, 1)])) + 1
    else:
        rows = int(ip_get(ip, "row"))
        skip = _matlab_round_scalar(rows * 0.05)
        row_min = max(skip, 1)
        row_max = max(rows - skip, row_min + 1)

    dx_row_min = np.diff(np.sum(monochrome[row_min + 1 : row_min + 5, :], axis=0, dtype=float))
    dx_row_max = np.diff(np.sum(monochrome[row_max - 5 : row_max - 1, :], axis=0, dtype=float))

    upper_left_x = int(np.argmin(dx_row_min)) + 1
    lower_right_x = int(np.argmin(dx_row_max)) + 1
    upper_left = np.array([upper_left_x, row_min], dtype=float)
    lower_right = np.array([lower_right_x, row_max], dtype=float)
    midpoint = (lower_right + upper_left) / 2.0

    width = int(np.rint(abs(lower_right[0] - upper_left[0])))
    height = _matlab_round_scalar(1.5 * float(width))
    col_min = _matlab_round_scalar(midpoint[0] - (float(width) / 2.0))
    row_min_rect = _matlab_round_scalar(midpoint[1] - (float(height) / 2.0))

    if width < 5 or height < 5:
        return np.array([], dtype=int)
    return np.array([col_min, row_min_rect, width, height], dtype=int)


def iso12233(
    bar_image: Any,
    delta_x: float | None = None,
    weight: Any | None = None,
    plot_options: str | None = None,
) -> ISO12233Result:
    """Compute a headless slanted-edge ISO 12233 estimate."""

    del plot_options
    if isinstance(bar_image, ImageProcessor):
        ip = bar_image
        rect = iso_find_slanted_bar(ip)
        if rect.size == 0:
            raise ValueError("Could not determine a slanted-edge rect from the IP.")
        image = _bar_image_from_cube(ip_get(ip, "result"), rect)
    else:
        rect = None
        image = np.asarray(bar_image, dtype=float)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError("ISO12233 expects a 2D image or 3D bar-image cube.")

    if delta_x is None:
        delta_x = 0.002
    delta_x = float(delta_x)
    if abs(delta_x - 1.0) <= 1.0e-12:
        funit = "cy/pixel"
    elif delta_x > 1.0:
        delta_x = 25.4 / delta_x
        funit = "cy/deg at 1m distance"
    else:
        funit = "cy/mm on sensor"

    luminance_weight = np.asarray(weight if weight is not None else [0.213, 0.715, 0.072], dtype=float).reshape(-1)
    if image.shape[2] == 3:
        image = np.dstack((image, _weighted_luminance(image, luminance_weight)))

    image, n_rows, n_cols, _ = _rotatev2(image)
    n_wave = int(image.shape[2])
    loc = np.zeros((n_wave, n_rows), dtype=float)

    fil1 = np.array([0.5, -0.5], dtype=float)
    fil2 = np.array([0.5, 0.0, -0.5], dtype=float)
    col_window = min(5, n_cols)
    left_sum = float(np.sum(image[:, :col_window, 0], dtype=float))
    right_sum = float(np.sum(image[:, n_cols - col_window :, 0], dtype=float))
    if left_sum > right_sum:
        fil1 = np.array([-0.5, 0.5], dtype=float)
        fil2 = np.array([-0.5, 0.0, 0.5], dtype=float)

    fit = np.zeros((n_wave, 2), dtype=float)
    win1 = _ahamming(n_cols, (n_cols + 1) / 2.0)
    for color_index in range(n_wave):
        derivative = _deriv1(image[:, :, color_index], n_rows, n_cols, fil1)
        for row_index in range(n_rows):
            loc[color_index, row_index] = _pb_centroid(derivative[row_index, :] * win1) - 0.5
        fit[color_index, :] = _findedge2(loc[color_index, :], n_rows, 1)
        for row_index in range(n_rows):
            place = float(np.polyval(fit[color_index, :], row_index + 1.0))
            win2 = _ahamming(n_cols, place)
            loc[color_index, row_index] = _pb_centroid(derivative[row_index, :] * win2) - 0.5
        fit[color_index, :] = _findedge2(loc[color_index, :], n_rows, 1)

    oversample = 4
    nn = int(np.floor(n_cols * oversample))
    nn2 = (nn // 2) + 1
    freq = (oversample * np.arange(nn, dtype=float)) / (delta_x * nn)
    nn2out = _matlab_round_scalar(nn2 / 2.0)
    nyquistf = float(1.0 / (2.0 * delta_x))
    if funit.startswith("cy/deg"):
        mm_per_degree = 2.0 * np.arctan(np.deg2rad(0.5)) * 1.0e3
        freq = freq * mm_per_degree
        nyquistf = nyquistf * mm_per_degree

    mtf = np.zeros((nn2, n_wave), dtype=float)
    esf = np.zeros((nn, n_wave), dtype=float)
    win = _ahamming(nn, (nn + 1) / 2.0)
    final_lsf = np.zeros(nn, dtype=float)
    for color_index in range(n_wave):
        point = _project2(image[:, :, color_index], fit[color_index, :], oversample)
        esf[:, color_index] = point
        lsf = _deriv1(point.reshape(1, -1), 1, nn, fil2).reshape(-1)
        lsf = _cent(lsf, _matlab_round_scalar(_pb_centroid(lsf)))
        lsf = win * lsf
        spectrum = np.abs(np.fft.fft(lsf, n=nn))
        mtf[:, color_index] = spectrum[:nn2] / max(float(spectrum[0]), 1.0e-12)
        final_lsf = lsf

    freq_out = np.asarray(freq[:nn2out], dtype=float)
    mtf_out = np.asarray(mtf[:nn2out, :], dtype=float)
    lsf_out = np.asarray(final_lsf / max(float(np.max(final_lsf)), 1.0e-12), dtype=float)
    lsfx = np.arange(1, lsf_out.size + 1, dtype=float)
    lsfx = lsfx - float(np.mean(lsfx, dtype=float))
    lsfx = lsfx * 0.5 * (1.0 / max(float(np.max(freq_out)), 1.0e-12))

    lum_mtf = mtf_out[:, -1]
    below_nyquist = freq_out < nyquistf
    if np.any(below_nyquist):
        interp_freq = np.arange(0.0, float(nyquistf) + 0.2, 0.2, dtype=float)
        interp_mtf = np.interp(interp_freq, freq_out[below_nyquist], lum_mtf[below_nyquist])
        mtf50 = float(interp_freq[int(np.argmin(np.abs(interp_mtf - 0.5)))])
    else:
        mtf50 = float("nan")

    alias_channel = 1 if n_wave == 4 else 0
    aliasing = 100.0 * float(np.sum(mtf_out[~below_nyquist, alias_channel], dtype=float)) / max(
        float(np.sum(mtf_out[:, alias_channel], dtype=float)),
        1.0e-12,
    )

    return ISO12233Result(
        freq=freq_out,
        mtf=mtf_out[:, 0] if mtf_out.shape[1] == 1 else mtf_out,
        nyquistf=nyquistf,
        lsf=lsf_out,
        lsfx=np.asarray(lsfx, dtype=float),
        mtf50=mtf50,
        aliasingPercentage=aliasing,
        esf=esf[:, 0] if esf.shape[1] == 1 else esf,
        fitme=np.asarray(fit, dtype=float),
        rect=None if rect is None else np.asarray(rect, dtype=int),
        win=None,
    )


def ie_iso12233(
    ip: ImageProcessor,
    sensor: Sensor,
    plot_options: str | None = None,
    master_rect: Any | None = None,
    *,
    weight: Any | None = None,
    npoly: int = 1,
) -> ISO12233Result:
    """Headless MATLAB-style wrapper around `iso12233` for IP/sensor inputs."""

    del npoly
    rect = np.asarray(master_rect if master_rect is not None else iso_find_slanted_bar(ip), dtype=int).reshape(-1)
    if rect.size == 0:
        raise ValueError("Could not determine a slanted-edge rect from the IP.")
    bar_image = _bar_image_from_cube(ip_get(ip, "sensor space"), rect)
    delta_x = float(sensor_get(sensor, "pixel width", "mm"))
    result = iso12233(bar_image, delta_x=delta_x, weight=weight, plot_options=plot_options)
    result.rect = rect
    return result


def iso12233_v1(
    bar_image: Any,
    delta_x: float | None = None,
    weight: Any | None = None,
    plot_options: str | None = None,
) -> ISO12233Result:
    """Compatibility wrapper for the legacy MATLAB ISO12233v1 file surface."""

    return iso12233(bar_image, delta_x=delta_x, weight=weight, plot_options=plot_options)


def ie_iso12233_v1(
    ip: ImageProcessor,
    sensor: Sensor,
    plot_options: str | None = None,
    master_rect: Any | None = None,
    *,
    weight: Any | None = None,
    npoly: int = 1,
) -> ISO12233Result:
    """Compatibility wrapper for the legacy MATLAB ieISO12233v1 file surface."""

    return ie_iso12233(
        ip,
        sensor,
        plot_options=plot_options,
        master_rect=master_rect,
        weight=weight,
        npoly=npoly,
    )


def edge_to_mtf(
    bar_image: Any,
    *,
    channel: int = 2,
    fixed_row: int = 20,
) -> dict[str, NDArray[np.float64] | NDArray[np.int_]]:
    """Estimate an edge-derived MTF using the workflow from s_metricsEdge2MTF.m."""

    image = np.asarray(bar_image, dtype=float)
    if image.ndim != 3:
        raise ValueError("edge_to_mtf expects an RGB-format bar image.")
    channel_index = int(channel) - 1
    if channel_index < 0 or channel_index >= image.shape[2]:
        raise ValueError("Requested channel is out of range.")

    mono = image[:, :, channel_index]
    dimg = np.abs(np.diff(mono, axis=1))
    row_count, col_count = dimg.shape
    aligned = np.zeros_like(dimg, dtype=float)
    lags = np.zeros(row_count, dtype=int)
    fixed = dimg[min(max(int(fixed_row) - 1, 0), row_count - 1), :]

    for row_index in range(row_count):
        corr, row_lags = ie_cxcorr(fixed, dimg[row_index, :])
        best_index = int(np.argmax(corr))
        shift = int(row_lags[best_index])
        aligned[row_index, :] = np.roll(dimg[row_index, :], shift)
        lags[row_index] = shift

    lsf = np.mean(aligned, axis=0, dtype=float)
    lsf = lsf / max(float(np.sum(lsf, dtype=float)), 1.0e-12)
    mtf = np.abs(np.fft.fft(lsf))
    freq = np.arange(_matlab_round_scalar(col_count / 2.0), dtype=float)

    return {
        "dimg": np.asarray(dimg, dtype=float),
        "aligned": np.asarray(aligned, dtype=float),
        "lsf": np.asarray(lsf, dtype=float),
        "mtf": np.asarray(mtf[: freq.size], dtype=float),
        "freq": np.asarray(freq, dtype=float),
        "lags": np.asarray(lags, dtype=int),
    }


ISOFindSlantedBar = iso_find_slanted_bar
ISO12233 = iso12233
ISO12233v1 = iso12233_v1
ieCXcorr = ie_cxcorr
ieISO12233 = ie_iso12233
ieISO12233v1 = ie_iso12233_v1
