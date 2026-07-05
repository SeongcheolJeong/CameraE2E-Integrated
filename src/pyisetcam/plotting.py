"""Headless plot-data wrappers for selected MATLAB plotting APIs."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import griddata as scipy_griddata
from scipy.signal import convolve2d
from scipy.spatial import Voronoi

from .color import xyz_color_matching
from .display import display_get
from .exceptions import UnsupportedOptionError
from .ip import demosaic, ip_create, ip_get
from .metrics import chromaticity_xy, metrics_get, metrics_roi, xyz_from_energy, xyz_to_lab, xyz_to_luv
from .optics import airy_disk, oi_get, wvf_get
from .scene import scene_get
from .sensor import ml_get_current, mlens_get, pixel_snr, sensor_get, sensor_snr
from .types import ImageProcessor, OpticalImage, Scene, Sensor
from .utils import blackbody, linear_to_srgb, param_format, tile_pattern, xw_to_rgb_format, xyz_to_linear_srgb, xyz_to_srgb


def _roi_required(function_name: str, plot_type: str, roi_locs: Any | None) -> Any:
    if roi_locs is None:
        raise ValueError(f"ROI required for {function_name}(..., '{plot_type}').")
    return roi_locs


def _roi_payload(roi_locs: Any) -> dict[str, Any]:
    from .roi import ie_locs2_rect

    roi = np.asarray(roi_locs, dtype=int)
    payload: dict[str, Any] = {"roiLocs": roi.copy()}
    if roi.ndim == 1 and roi.size == 4:
        payload["rect"] = roi.copy()
    elif roi.ndim == 2 and roi.shape[1] == 2:
        payload["rect"] = ie_locs2_rect(roi)
    return payload


def _line_index(function_name: str, plot_type: str, xy: Any | None, orientation: str) -> tuple[int, np.ndarray]:
    if xy is None:
        raise ValueError(f"Line selector required for {function_name}(..., '{plot_type}').")
    xy_array = np.rint(np.asarray(xy, dtype=float)).astype(int).reshape(-1)
    if xy_array.size == 1:
        return int(xy_array[0]), xy_array.copy()
    if xy_array.size != 2:
        raise ValueError("Line selector must be a scalar index or [col, row].")
    index = int(xy_array[1] if orientation == "h" else xy_array[0])
    return index, xy_array.copy()


def _plot_option(args: tuple[Any, ...], key: str, default: Any = None) -> Any:
    if len(args) % 2 != 0:
        raise ValueError("Optional plotting arguments must be key/value pairs.")
    normalized_key = param_format(key)
    for index in range(0, len(args), 2):
        if param_format(args[index]) == normalized_key:
            return args[index + 1]
    return default


def _axis_limits(ax: Any | None, key: str, default: tuple[float, float]) -> np.ndarray:
    if ax is None:
        return np.asarray(default, dtype=float)
    if isinstance(ax, dict):
        value = ax.get(key, default)
        return np.asarray(value, dtype=float).reshape(2)
    if hasattr(ax, key):
        return np.asarray(getattr(ax, key), dtype=float).reshape(2)
    try:
        value = ax[key]
    except Exception:
        value = default
    return np.asarray(value, dtype=float).reshape(2)


def _axis_scale(ax: Any | None, key: str, default: str) -> str:
    if ax is None:
        return default
    if isinstance(ax, dict):
        return str(ax.get(key, default))
    if hasattr(ax, key):
        return str(getattr(ax, key))
    try:
        return str(ax[key])
    except Exception:
        return default


def _hot_colormap(count: int = 64) -> np.ndarray:
    a = np.linspace(0.0, 1.0, int(count), dtype=float)
    return np.column_stack(
        (
            np.clip(3.0 * a, 0.0, 1.0),
            np.clip(3.0 * a - 1.0, 0.0, 1.0),
            np.clip(3.0 * a - 2.0, 0.0, 1.0),
        )
    )


def identity_line(ax: Any | None = None, three_d: bool = False) -> dict[str, Any]:
    """Return MATLAB-style `identityLine` line data without opening a figure."""

    xlim = _axis_limits(ax, "xlim", (0.0, 1.0))
    ylim = _axis_limits(ax, "ylim", (0.0, 1.0))
    if three_d:
        zlim = _axis_limits(ax, "zlim", (0.0, 1.0))
        minimum = float(min(xlim[0], ylim[0], zlim[0]))
        maximum = float(max(xlim[1], ylim[1], zlim[1]))
        return {
            "x": np.asarray([minimum, maximum], dtype=float),
            "y": np.asarray([minimum, maximum], dtype=float),
            "z": np.asarray([minimum, maximum], dtype=float),
            "color": np.asarray([0.5, 0.5, 0.5], dtype=float),
            "linestyle": "--",
            "linewidth": 2.0,
            "grid": True,
        }

    minimum = float(min(xlim[0], ylim[0]))
    maximum = float(max(xlim[1], ylim[1]))
    return {
        "x": np.asarray([minimum, maximum], dtype=float),
        "y": np.asarray([minimum, maximum], dtype=float),
        "color": np.asarray([0.5, 0.5, 0.5], dtype=float),
        "linestyle": "--",
        "linewidth": 2.0,
        "grid": True,
    }


def xaxis_line(ax: Any | None = None, yval: float = 0.0, linestyle: str = "--") -> dict[str, Any]:
    """Return MATLAB-style `xaxisLine` line data without opening a figure."""

    xlim = _axis_limits(ax, "xlim", (0.0, 1.0))
    return {
        "x": np.asarray(xlim, dtype=float).copy(),
        "y": np.asarray([float(yval), float(yval)], dtype=float),
        "color": np.asarray([0.3, 0.3, 0.3], dtype=float),
        "linestyle": str(linestyle),
        "linewidth": 2.0,
        "grid": True,
    }


def yaxis_line(ax: Any | None = None, val: float = 0.0) -> dict[str, Any]:
    """Return MATLAB-style `yaxisLine` line data without opening a figure."""

    ylim = _axis_limits(ax, "ylim", (0.0, 1.0))
    return {
        "x": np.asarray([float(val), float(val)], dtype=float),
        "y": np.asarray(ylim, dtype=float).copy(),
        "color": np.asarray([0.3, 0.3, 0.3], dtype=float),
        "linestyle": "--",
        "linewidth": 2.0,
        "grid": True,
    }


def plot_text_string(
    text: str,
    position: str = "ur",
    delta: Any = (0.2, 0.2),
    font_size: float = 12.0,
    ax: Any | None = None,
) -> dict[str, Any]:
    """Return MATLAB-style `plotTextString` placement data without opening a figure."""

    xlim = _axis_limits(ax, "xlim", (0.0, 1.0))
    ylim = _axis_limits(ax, "ylim", (0.0, 1.0))
    xscale = _axis_scale(ax, "xscale", "linear")
    yscale = _axis_scale(ax, "yscale", "linear")
    delta_array = np.asarray(delta, dtype=float).reshape(-1)
    if delta_array.size == 1:
        delta_array = np.repeat(delta_array, 2)
    if delta_array.size != 2:
        raise ValueError("plotTextString delta must be a scalar or length-2 sequence.")

    pos_key = param_format(position)
    if pos_key == "ul":
        x = xlim[0] + (xlim[1] - xlim[0]) * delta_array[0]
        y = ylim[1] - (ylim[1] - ylim[0]) * delta_array[1]
    elif pos_key == "ll":
        x = xlim[0] + (xlim[1] - xlim[0]) * delta_array[0]
        y = ylim[0] + (ylim[1] - ylim[0]) * delta_array[1]
    elif pos_key == "ur":
        x = xlim[1] - (xlim[1] - xlim[0]) * delta_array[0]
        y = ylim[1] - (ylim[1] - ylim[0]) * delta_array[1]
    elif pos_key == "lr":
        x = xlim[1] - (xlim[1] - xlim[0]) * delta_array[0]
        y = ylim[0] + (ylim[1] - ylim[0]) * delta_array[1]
    else:
        raise ValueError("Unknown position")

    return {
        "text": str(text),
        "position": str(position),
        "x": float(x),
        "y": float(y),
        "fontSize": float(font_size),
        "background": "w",
        "delta": delta_array.astype(float).copy(),
        "xscale": xscale,
        "yscale": yscale,
    }


def plot_gaussian_spectrum(
    wavelength: Any,
    mu: float,
    sig: float,
    *,
    asset_store: Any | None = None,
) -> dict[str, Any]:
    """Return MATLAB-style `plotGaussianSpectrum` payload without opening a figure."""

    wave = np.asarray(wavelength, dtype=float).reshape(-1)
    sigma = float(sig)
    tran = np.exp(-((wave - float(mu)) ** 2) / max(2.0 * sigma**2, 1.0e-30))
    xyz = np.asarray(xyz_color_matching(wave, energy=True, asset_store=asset_store), dtype=float)
    if xyz.size:
        scaled_xyz = xyz.copy()
        max_y = float(np.max(scaled_xyz[:, 1]))
        if max_y > 1.0:
            scaled_xyz = scaled_xyz / max_y
        if float(np.min(scaled_xyz)) < 0.0:
            scaled_xyz = np.clip(scaled_xyz, 0.0, 1.0)
        colors = linear_to_srgb(np.clip(xyz_to_linear_srgb(scaled_xyz), 0.0, 1.0))
    else:
        colors = np.zeros((wave.size, 3), dtype=float)
    invisible = np.all(colors <= 0.0, axis=1)
    if np.any(invisible):
        colors[invisible] = 0.3
    return {
        "wavelength": wave.copy(),
        "transmittance": tran.copy(),
        "supportRGB": colors.copy(),
        "xTick": np.array([], dtype=float),
        "yTick": np.array([], dtype=float),
    }


def plot_spectrum_locus(*, asset_store: Any | None = None) -> dict[str, Any]:
    """Return MATLAB-style `plotSpectrumLocus` payload without opening a figure."""

    wave = np.arange(370.0, 731.0, dtype=float)
    xyz = np.asarray(xyz_color_matching(wave, energy=True, asset_store=asset_store), dtype=float)
    xy = np.asarray(chromaticity_xy(xyz), dtype=float)
    closing = np.vstack((xy[0], xy[-1]))
    return {
        "wave": wave.copy(),
        "xy": xy.copy(),
        "closingLine": closing.copy(),
        "axisEqual": True,
        "grid": True,
        "linestyle": "--",
    }


def plot_contrast_histogram(data: Any, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-style `plotContrastHistogram` histogram outputs."""

    values = np.asarray(data, dtype=float)
    mean_value = float(np.mean(values))
    contrast = (values.reshape(-1) - mean_value) / mean_value
    counts, edges = np.histogram(contrast, bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), np.asarray(centers, dtype=float)


def plot_etendue_ratio(
    sensor: Sensor,
    optimal: Any,
    bare: Any,
    z_label: str = "Etendue improvement (%)",
) -> dict[str, Any]:
    """Return MATLAB-style `plotEtendueRatio` payload without opening a figure."""

    support = sensor_get(sensor, "spatial support", "microns")
    optimal_array = np.asarray(optimal, dtype=float)
    bare_array = np.asarray(bare, dtype=float)
    ratio = (optimal_array / bare_array - 1.0) * 100.0
    return {
        "support": {
            "x": np.asarray(support["x"], dtype=float).copy(),
            "y": np.asarray(support["y"], dtype=float).copy(),
        },
        "Ratio": ratio.copy(),
        "zLabel": str(z_label),
    }


def _coerce_spectral_lines(wavelength: Any, values: Any) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(wavelength, dtype=float).reshape(-1)
    array = np.asarray(values, dtype=float)
    if wave.size == array.shape[0]:
        return wave.copy(), array.copy()
    if array.ndim >= 2 and wave.size == array.shape[1]:
        return wave.copy(), np.swapaxes(array, 0, 1).copy()
    raise ValueError("Wavelength support must match one dimension of the data array.")


def ie_plane_from_vectors(
    A: Any,
    xylimits: tuple[float, float] | list[float] | np.ndarray = (-1.0, 1.0),
    npoints: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MATLAB-style plane points from two 3-D vectors."""

    basis = np.asarray(A, dtype=float)
    if basis.shape != (3, 2):
        raise ValueError("iePlaneFromVectors expects a 3x2 input matrix.")

    limits = np.asarray(xylimits, dtype=float).reshape(-1)
    if limits.size != 2:
        raise ValueError("xylimits must contain exactly two values.")

    point1 = basis[:, 0]
    point2 = basis[:, 1]
    normal_vector = np.cross(point1, point2)
    a, b, c = normal_vector
    if abs(float(c)) < 1.0e-12:
        raise ValueError("The input vectors do not define a plane with finite z support.")

    x, y = np.meshgrid(
        np.linspace(limits[0], limits[1], 10, dtype=float),
        np.linspace(limits[0], limits[1], int(npoints), dtype=float),
    )
    z = -(a * x + b * y) / c
    return x, y, np.asarray(z, dtype=float)


def ie_plot_jitter(
    x: Any,
    y: Any,
    f: float | None = None,
    fig: int = -1,
    p_symbol: str = ".k",
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Return MATLAB-style jittered point payload without opening a figure."""

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same number of elements.")

    if f is None:
        mx = max(float(np.max(x_values) - np.min(x_values)), float(np.max(y_values) - np.min(y_values)))
        f = mx / 200.0
    generator = rng if rng is not None else np.random.default_rng(0)
    jx = x_values + generator.random(x_values.size) * float(f)
    jy = y_values + generator.random(y_values.size) * float(f)
    return {
        "jx": jx.astype(float).copy(),
        "jy": jy.astype(float).copy(),
        "f": float(f),
        "figure": int(fig),
        "pSymbol": str(p_symbol),
    }


def plot_normal(
    m: Any,
    s: Any,
    *,
    color: list[str] | tuple[str, ...] | None = None,
    sample_count: int = 200,
) -> dict[str, Any]:
    """Return MATLAB-style normal-distribution plot payload without opening a figure."""

    means = np.asarray(m, dtype=float).reshape(-1)
    sigmas = np.asarray(s, dtype=float).reshape(-1)
    if means.size != sigmas.size:
        raise ValueError("Mean and sigma vectors must have the same length.")
    if np.any(sigmas <= 0.0):
        raise ValueError("All standard deviations must be positive.")

    x = np.linspace(
        float(np.min(means) - 4.0 * np.max(sigmas)),
        float(np.max(means) + 4.0 * np.max(sigmas)),
        int(sample_count),
        dtype=float,
    )
    pdf = np.empty((means.size, x.size), dtype=float)
    for index, (mean_value, sigma_value) in enumerate(zip(means, sigmas, strict=False)):
        pdf[index] = (1.0 / (sigma_value * np.sqrt(2.0 * np.pi))) * np.exp(
            -((x - mean_value) ** 2) / (2.0 * sigma_value**2)
        )

    curve_colors = list(color) if color is not None else []
    return {
        "x": x.copy(),
        "pdf": pdf.copy(),
        "mean": means.copy(),
        "sigma": sigmas.copy(),
        "color": curve_colors,
        "xlabel": "x",
        "ylabel": "Probability Density",
        "grid": True,
        "axisTight": True,
    }


def plot_radiance(
    wavelength: Any,
    radiance: Any,
    *,
    title: str = "Spectral radiance",
    color: Any = "",
    linewidth: float = 2.0,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotRadiance` payload without opening a figure."""

    wave, lines = _coerce_spectral_lines(wavelength, radiance)
    return {
        "wavelength": wave.copy(),
        "radiance": np.asarray(lines, dtype=float).copy(),
        "title": str(title),
        "color": color,
        "linewidth": float(linewidth),
        "xlabel": "Wavelength (nm)",
        "ylabel": "Radiance (watts/sr/nm/m^2)",
        "grid": True,
    }, None


def plot_reflectance(
    wavelength: Any,
    reflectance: Any,
    *,
    title: str = "Spectral reflectance",
    color: Any = None,
    linewidth: float = 2.0,
    linestyle: str = "-",
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotReflectance` payload without opening a figure."""

    wave, lines = _coerce_spectral_lines(wavelength, reflectance)
    return {
        "wavelength": wave.copy(),
        "reflectance": np.asarray(lines, dtype=float).copy(),
        "title": str(title),
        "color": color,
        "linewidth": float(linewidth),
        "linestyle": str(linestyle),
        "xlabel": "Wavelength (nm)",
        "ylabel": "Reflectance",
        "grid": True,
    }, None


def _numeric_plot_vector(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    return np.asarray(array, dtype=float).reshape(-1)


def ie_plot(*args: Any) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `iePlot` series payload without opening a figure."""

    if not args:
        raise ValueError("iePlot requires at least one data vector.")

    series: list[dict[str, Any]] = []
    index = 0
    while index < len(args):
        first = _numeric_plot_vector(args[index])
        if first is None:
            raise ValueError("iePlot expects numeric plot vectors and optional style strings.")

        if index + 1 < len(args):
            second = _numeric_plot_vector(args[index + 1])
        else:
            second = None

        if second is None:
            y = first
            x = np.arange(1.0, float(y.size) + 1.0, dtype=float)
            index += 1
        else:
            x = first
            y = second
            index += 2
        if x.size != y.size:
            raise ValueError("iePlot x/y vectors must have the same number of samples.")

        style = "-"
        if index < len(args) and isinstance(args[index], str):
            style = str(args[index])
            index += 1

        series.append(
            {
                "x": x.copy(),
                "y": y.copy(),
                "style": style,
                "lineWidth": 0.5,
            }
        )

    return {"series": series, "figureFactory": "ieFigure"}, None


def ie_plot_set(ax: dict[str, Any], *args: Any) -> dict[str, Any]:
    """Return MATLAB-style `iePlotSet` payload updates without mutating figures."""

    if len(args) % 2 != 0:
        raise ValueError("iePlotSet optional arguments must be key/value pairs.")

    updated = dict(ax)
    for index in range(0, len(args), 2):
        key = param_format(args[index])
        value = args[index + 1]
        if key != "linewidth":
            raise ValueError(f"Unknown parameter {args[index]}")
        if "series" in updated:
            updated["series"] = [
                {**series, "lineWidth": float(value)} for series in updated.get("series", [])
            ]
        if "lines" in updated:
            updated["lines"] = [{**line, "lineWidth": float(value)} for line in updated.get("lines", [])]
    return updated


def ie_plot_shade_background(ax: dict[str, Any], *args: Any) -> dict[str, Any]:
    """Return MATLAB-style shaded-axis background payload without opening a figure."""

    vertex_colors = np.asarray(
        _plot_option(
            args,
            "vertexcolors",
            np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float),
        ),
        dtype=float,
    )
    grid_state = str(_plot_option(args, "grid", "on")).lower()
    xlim = _axis_limits(ax, "xlim", (0.0, 1.0))
    ylim = _axis_limits(ax, "ylim", (0.0, 1.0))
    vertices = np.array(
        [[xlim[0], ylim[0]], [xlim[1], ylim[0]], [xlim[1], ylim[1]], [xlim[0], ylim[1]]],
        dtype=float,
    )
    updated = dict(ax)
    updated["background"] = {
        "vertices": vertices,
        "faceVertexCData": vertex_colors.copy(),
        "faceColor": "interp",
        "edgeColor": "none",
        "stack": "bottom",
    }
    if grid_state == "on":
        updated["layer"] = "top"
        updated["grid"] = True
    else:
        updated["grid"] = False
    return updated


def ie_shape(type: str = "circle", n_samp: int = 200, *args: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MATLAB-style `ieShape` samples for simple analytic shapes."""

    shape_key = param_format(type)
    count = int(n_samp)
    x = np.zeros(count, dtype=float)
    y = np.zeros(count, dtype=float)
    z = np.zeros(count, dtype=float)
    if shape_key in {"circle", "circ"}:
        radius = 1.0 if not args else float(args[0])
        theta = (2.0 * np.pi * np.arange(1, count + 1, dtype=float)) / float(max(count, 1))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y, z
    raise ValueError(f"Unknown type: {type}")


def ie_figure_format(
    fig: dict[str, Any] | None = None,
    fontname: str = "Helvetica",
    fontsize: Any = (18.0, 14.0),
    figsize: Any = (6.5, 6.5),
    border: Any = (1.0, 0.5),
) -> dict[str, Any]:
    """Return MATLAB-style formatted-figure properties without opening a figure."""

    formatted = {} if fig is None else dict(fig)
    fontsize_array = np.asarray(fontsize, dtype=float).reshape(-1)
    if fontsize_array.size == 1:
        fontsize_array = np.repeat(fontsize_array, 2)
    figure_size = np.asarray(figsize, dtype=float).reshape(-1)
    if figure_size.size == 1:
        figure_size = np.repeat(figure_size, 2)
    border_array = np.asarray(border, dtype=float).reshape(-1)
    if border_array.size == 2:
        border_array = np.array([border_array[0], border_array[0], border_array[1], border_array[1]], dtype=float)
    if border_array.size != 4:
        raise ValueError("border must have length 2 or 4.")

    prior_position = np.asarray(formatted.get("position", [0.0, 0.0, 0.0, 0.0]), dtype=float).reshape(-1)
    if param_format(formatted.get("units", "")) == "inches" and prior_position.size >= 2:
        origin = prior_position[:2]
    else:
        origin = np.array([0.0, 0.0], dtype=float)

    formatted["units"] = "inches"
    formatted["position"] = np.array([origin[0], origin[1], figure_size[0], figure_size[1]], dtype=float)
    formatted["paperPosition"] = np.array(
        [4.25 - figure_size[0] / 2.0, 5.5 - figure_size[1] / 2.0, figure_size[0], figure_size[1]],
        dtype=float,
    )
    formatted["color"] = np.array([1.0, 1.0, 1.0], dtype=float)
    formatted["axes"] = {
        **dict(formatted.get("axes", {})),
        "fontName": str(fontname),
        "titleFontSize": float(fontsize_array[0]),
        "labelFontSize": float(fontsize_array[0]),
        "tickFontSize": float(fontsize_array[1]),
        "units": "inches",
        "position": np.array(
            [
                border_array[0],
                border_array[1],
                figure_size[0] - border_array[0] - border_array[2],
                figure_size[1] - border_array[1] - border_array[3],
            ],
            dtype=float,
        ),
    }
    return formatted


def ie_figure_resize(
    fig_or_ax: dict[str, Any] | None = None,
    figpos: Any = (0.0035, 0.4125, 0.3266, 0.4972),
    units: str = "normalize",
) -> dict[str, Any]:
    """Return MATLAB-style figure-resize properties without opening a figure."""

    resized = {} if fig_or_ax is None else dict(fig_or_ax)
    resized["windowStyle"] = "normal"
    resized["units"] = str(units)
    resized["resize"] = "off"
    resized["position"] = np.asarray(figpos, dtype=float).reshape(-1)
    return resized


def fise_plot_defaults() -> dict[str, Any]:
    """Return the root graphics defaults from `fise_plotDefaults.m`."""

    return {
        "DefaultAxesFontName": "Georgia",
        "DefaultAxesFontSize": 16.0,
        "DefaultAxesBox": "off",
        "DefaultAxesTickDir": "out",
        "DefaultAxesLineWidth": 1.2,
        "DefaultAxesXColor": np.array([0.3, 0.3, 0.3], dtype=float),
        "DefaultAxesYColor": np.array([0.3, 0.3, 0.3], dtype=float),
        "DefaultTextFontName": "Georgia",
        "DefaultTextFontSize": 12.0,
        "DefaultTextColor": np.array([0.3, 0.3, 0.3], dtype=float),
        "DefaultLegendFontName": "Georgia",
        "DefaultLegendFontSize": 11.0,
        "DefaultLegendTextColor": np.array([0.2, 0.2, 0.2], dtype=float),
        "DefaultLegendBox": "off",
    }


def _gray_colormap(count: int = 256) -> np.ndarray:
    levels = np.linspace(0.0, 1.0, int(count), dtype=float)
    return np.repeat(levels[:, None], 3, axis=1)


def _hist2d_prepare_input(data: Any) -> np.ndarray:
    array = np.asarray(data)
    if np.iscomplexobj(array):
        return np.column_stack((np.real(array).reshape(-1), np.imag(array).reshape(-1)))
    if array.ndim != 2:
        raise ValueError("hist2d input must be a 2-column/2-row matrix or complex vector.")
    if array.shape[0] < array.shape[1] and array.shape[0] > 1:
        array = array.T
    if array.shape[1] != 2:
        raise ValueError("The input data matrix must have 2 rows or 2 columns.")
    return np.asarray(array, dtype=float)


def _polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def hist2d(
    D: Any,
    Xn: int = 20,
    Yn: int = 20,
    Xrange: Any | None = None,
    Yrange: Any | None = None,
) -> np.ndarray:
    """Return MATLAB-style 2-D histogram counts using nearest support bins."""

    samples = _hist2d_prepare_input(D)
    x_range = np.asarray(Xrange if Xrange is not None else [np.min(samples[:, 0]), np.max(samples[:, 0])], dtype=float)
    y_range = np.asarray(Yrange if Yrange is not None else [np.min(samples[:, 1]), np.max(samples[:, 1])], dtype=float)
    x_support = np.linspace(x_range[0], x_range[1], int(Xn), dtype=float)
    y_support = np.linspace(y_range[0], y_range[1], int(Yn), dtype=float)
    x_indices = np.argmin(np.abs(x_support[:, None] - samples[:, 0][None, :]), axis=0)
    y_indices = np.argmin(np.abs(y_support[:, None] - samples[:, 1][None, :]), axis=0)
    counts = np.zeros((int(Yn), int(Xn)), dtype=float)
    np.add.at(counts, (y_indices, x_indices), 1.0)
    return counts


def _scatter_groups(x: np.ndarray, y: np.ndarray, density: np.ndarray, marker_size: float, colormap: np.ndarray) -> list[dict[str, Any]]:
    color_map = np.asarray(colormap, dtype=float)
    if color_map.ndim != 2 or color_map.shape[1] != 3:
        raise ValueError("scatplot colormap must be an Nx3 array.")
    finite_density = np.asarray(density, dtype=float)
    minimum = float(np.nanmin(finite_density))
    maximum = float(np.nanmax(finite_density))
    if maximum <= minimum:
        indices = np.zeros(finite_density.shape, dtype=int)
    else:
        scaled = (finite_density - minimum) / (maximum - minimum)
        indices = np.clip(np.floor(scaled * (color_map.shape[0] - 1)).astype(int), 0, color_map.shape[0] - 1)
    groups: list[dict[str, Any]] = []
    for color_index in range(color_map.shape[0]):
        mask = indices == color_index
        if not np.any(mask):
            continue
        groups.append(
            {
                "x": np.asarray(x[mask], dtype=float).copy(),
                "y": np.asarray(y[mask], dtype=float).copy(),
                "color": color_map[color_index].copy(),
                "marker": ".",
                "markerSize": float(marker_size),
            }
        )
    return groups


def _griddata_linear_then_nearest(points: np.ndarray, values: np.ndarray, xi: Any) -> np.ndarray:
    try:
        return np.asarray(scipy_griddata(points, values, xi, method="linear", fill_value=np.nan), dtype=float)
    except Exception:
        return np.asarray(scipy_griddata(points, values, xi, method="nearest", fill_value=np.nan), dtype=float)


def _scatplot_density(x: np.ndarray, y: np.ndarray, method: str, radius: float) -> np.ndarray:
    mode = param_format(method)[:2]
    count = x.size
    density = np.zeros(count, dtype=float)
    if mode == "sq":
        for index in range(count):
            mask = (x > (x[index] - radius)) & (x < (x[index] + radius)) & (y > (y[index] - radius)) & (y < (y[index] + radius))
            density[index] = float(np.sum(mask))
        return density / max((2.0 * radius) ** 2, 1.0e-30)
    if mode == "ci":
        for index in range(count):
            mask = np.sqrt((x - x[index]) ** 2 + (y - y[index]) ** 2) < radius
            density[index] = float(np.sum(mask))
        return density / max(np.pi * radius**2, 1.0e-30)
    if mode == "vo":
        if count < 4:
            return density
        try:
            voronoi = Voronoi(np.column_stack((x, y)))
        except Exception:
            return density
        for index, region_index in enumerate(voronoi.point_region):
            region = voronoi.regions[region_index]
            if not region or any(vertex < 0 for vertex in region):
                continue
            vertices = np.asarray(voronoi.vertices[region], dtype=float)
            area = _polygon_area(vertices)
            if area > 0.0:
                density[index] = 1.0 / area
        return density
    raise ValueError(f"Unknown scatplot method: {method}")


def scatplot(
    x: Any,
    y: Any,
    method: Any = "vo",
    radius: float | None = None,
    N: int = 100,
    n: Any = 5,
    po: int = 1,
    ms: float = 4.0,
    colormap: Any | None = None,
) -> dict[str, Any]:
    """Return MATLAB-style scatter-density payload without opening a figure."""

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same number of elements.")
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    clean_x = x_values[finite_mask]
    clean_y = y_values[finite_mask]
    if clean_x.size == 0:
        raise ValueError("scatplot requires at least one finite point.")

    color_map = _gray_colormap(256) if colormap is None else np.asarray(colormap, dtype=float)
    if np.issubdtype(np.asarray(method).dtype, np.number):
        density = np.asarray(method, dtype=float).reshape(-1)
        groups = _scatter_groups(clean_x, clean_y, density[finite_mask], 2.0, color_map)
        return {"dd": density.copy(), "hs": groups}

    method_key = str(method)
    filter_shape = np.asarray(n, dtype=int).reshape(-1)
    if filter_shape.size == 0:
        filter_shape = np.array([5], dtype=int)
    if radius is None:
        radius = float(np.sqrt((np.ptp(clean_x) / 30.0) ** 2 + (np.ptp(clean_y) / 30.0) ** 2))

    dd_clean = _scatplot_density(clean_x, clean_y, method_key, float(radius))
    xi = np.tile(np.linspace(np.min(clean_x), np.max(clean_x), int(N), dtype=float), (int(N), 1))
    yi = np.tile(np.linspace(np.min(clean_y), np.max(clean_y), int(N), dtype=float).reshape(-1, 1), (1, int(N)))
    sample_points = np.column_stack((clean_x, clean_y))
    zi = _griddata_linear_then_nearest(sample_points, dd_clean, (xi, yi))
    zi[np.isnan(zi)] = 0.0
    coef = np.ones((int(filter_shape[0]),), dtype=float) / max(int(filter_shape[0]), 1)
    kernel = np.outer(coef, coef)
    zif = convolve2d(zi, kernel, mode="same")
    if filter_shape.size > 1:
        for _ in range(int(filter_shape[1])):
            zif = convolve2d(zif, kernel, mode="same")

    ddf_clean = _griddata_linear_then_nearest(
        np.column_stack((xi.reshape(-1), yi.reshape(-1))),
        zif.reshape(-1),
        np.column_stack((clean_x, clean_y)),
    ).reshape(-1)

    dd = np.full(x_values.shape, np.nan, dtype=float)
    ddf = np.full(x_values.shape, np.nan, dtype=float)
    dd[finite_mask] = dd_clean
    ddf[finite_mask] = ddf_clean

    if int(po) in {1, 2}:
        plot_density = ddf_clean
        contour_source = zif
    else:
        plot_density = dd_clean
        contour_source = zi

    payload: dict[str, Any] = {
        "dd": dd.copy(),
        "ddf": ddf.copy(),
        "radius": float(radius),
        "xi": xi.copy(),
        "yi": yi.copy(),
        "zi": zi.copy(),
        "zif": zif.copy(),
        "hs": _scatter_groups(clean_x, clean_y, plot_density, float(ms), color_map),
        "colorbar": True,
    }
    if int(po) in {2, 4}:
        payload["contour"] = {"xi": xi.copy(), "yi": yi.copy(), "z": contour_source.copy()}
    return payload


def ie_hist_image(X: Any, *args: Any) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray, Any]:
    """Return MATLAB-style histogram-image payloads without opening a figure."""

    samples = np.asarray(X, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("ieHistImage expects an Nx2 matrix.")

    plot_flag = bool(_plot_option(args, "plotflag", True))
    fhandle = _plot_option(args, "fhandle", None)
    edges = np.asarray(_plot_option(args, "edges", [32, 32]), dtype=int).reshape(-1)
    if edges.size != 2:
        raise ValueError("ieHistImage edges must have length 2.")
    hist_type = param_format(_plot_option(args, "histtype", "scatplot"))

    if hist_type == "histcn":
        x = samples[:, 0]
        y = samples[:, 1]
        x_edges = np.linspace(np.min(x), np.max(x), int(edges[0]) + 1, dtype=float)
        y_edges = np.linspace(np.min(y), np.max(y), int(edges[1]) + 1, dtype=float)
        histogram, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        img = histogram.T.astype(float)
        xy = (
            0.5 * (x_edges[:-1] + x_edges[1:]),
            0.5 * (y_edges[:-1] + y_edges[1:]),
        )
    elif hist_type == "scatplot":
        x = samples[:, 0]
        y = samples[:, 1]
        method = _plot_option(args, "scatmethod", "voronoi")
        radius = _plot_option(args, "scatradius", float(np.sqrt((np.ptp(x) / 30.0) ** 2 + (np.ptp(y) / 30.0) ** 2)))
        scat_struct = scatplot(x, y, method, float(radius), 100, 5, 1, 4, _gray_colormap(256))
        img = np.asarray(scat_struct["zif"], dtype=float)
        xy = (
            np.asarray(scat_struct["xi"][0, :], dtype=float).copy(),
            np.asarray(scat_struct["yi"][:, 0], dtype=float).copy(),
        )
    else:
        raise ValueError(f"Unknown histogram method {hist_type}.")

    cmap = 0.4 + 0.6 * _gray_colormap(256)
    returned_handle = fhandle if plot_flag else fhandle
    return img, xy, cmap, returned_handle


def plot_set_up_window(fig_num: Any | None = None) -> dict[str, Any]:
    """Return MATLAB-style graph-window defaults without opening a figure."""

    if fig_num is None:
        return {
            "figureId": "GRAPHWIN",
            "units": "Normalized",
            "position": np.array([0.5769, 0.0308, 0.4200, 0.4200], dtype=float),
            "name": "ISET GraphWin",
            "numberTitle": "off",
            "colormap": "default",
            "clear": True,
        }
    return {
        "figureId": fig_num,
        "colormap": "default",
        "clear": True,
    }


def _airy_disk_circle(radius: float, *, sample_count: int = 200) -> dict[str, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, int(sample_count), endpoint=False, dtype=float)
    return {
        "x": float(radius) * np.cos(theta),
        "y": float(radius) * np.sin(theta),
    }


def _middle_samples_1d(values: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1:
        raise ValueError("Expected a 1-D vector for middle-sample extraction.")
    half = int(np.rint(float(size) / 2.0))
    center = int(np.rint(vector.size / 2.0))
    start = max(0, center - half - 1)
    stop = min(vector.size, center + half)
    return np.asarray(vector[start:stop], dtype=vector.dtype)


def _oi_otf_wavelength_payload(
    oi: OpticalImage,
    *,
    units: str = "um",
    n_samp: int = 100,
) -> dict[str, Any]:
    optics = dict(oi.fields.get("optics", {}))
    model = param_format(optics.get("model", ""))
    wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    if wave.size == 0:
        raise ValueError("Optical image has no wavelength support.")

    if model == "diffractionlimited":
        f_number = float(optics.get("f_number", 4.0))
        wave_um = wave * 1e-3
        in_cutoff = 1.0 / np.maximum(wave_um * max(f_number, 1e-12), 1e-12)
        peak_f = 3.0 * float(np.max(in_cutoff))
        f_samp = np.arange(-int(n_samp), int(n_samp), dtype=float) / float(max(int(n_samp), 1))
        fx = f_samp * peak_f
        otf_wave = np.empty((fx.size, wave.size), dtype=float)
        for wave_index, wavelength_um in enumerate(wave_um):
            cutoff = 1.0 / max(float(wavelength_um) * max(f_number, 1e-12), 1e-12)
            normalized = np.abs(fx) / max(cutoff, 1e-12)
            clipped = np.clip(normalized, 0.0, 1.0)
            line = (2.0 / np.pi) * (np.arccos(clipped) - clipped * np.sqrt(1.0 - clipped**2))
            line[normalized >= 1.0] = 0.0
            otf_wave[:, wave_index] = np.asarray(line, dtype=float)
        return {"fSupport": fx, "wavelength": wave.copy(), "otf": otf_wave}

    if model == "shiftinvariant":
        otf_struct = oi_get(oi, "optics otfstruct")
        if otf_struct is None:
            raise ValueError("Shift-invariant OTF data are missing or inconsistent.")
        otf = np.asarray(otf_struct["OTF"], dtype=complex)
        fx_mm = np.asarray(otf_struct["fx"], dtype=float).reshape(-1)
        if otf.ndim != 3 or fx_mm.size != otf.shape[1]:
            raise ValueError("Shift-invariant OTF data are missing or inconsistent.")
        fx = fx_mm
        otf_wave = np.empty((fx.size, wave.size), dtype=float)
        for wave_index in range(wave.size):
            otf_wave[:, wave_index] = np.abs(np.fft.fftshift(otf[0, :, wave_index]))
        return {"fSupport": fx, "wavelength": wave.copy(), "otf": otf_wave}

    raise UnsupportedOptionError("oiPlot", "otf wavelength")


def _oi_lswavelength_payload(
    oi: OpticalImage,
    *,
    units: str = "um",
    middle_samps: int = 40,
    n_samp: int = 100,
) -> dict[str, Any]:
    optics = dict(oi.fields.get("optics", {}))
    model = param_format(optics.get("model", optics.get("compute_method", "diffractionlimited")))
    wave = np.asarray(oi_get(oi, "wavelength"), dtype=float).reshape(-1)

    if model == "shiftinvariant":
        otf_struct = oi_get(oi, "optics otfstruct")
        if otf_struct is None:
            raise ValueError("Shift-invariant OTF data are missing or inconsistent.")
        otf = np.asarray(otf_struct["OTF"], dtype=complex)
        fx_mm = np.asarray(otf_struct["fx"], dtype=float).reshape(-1)
        if otf.ndim != 3 or fx_mm.size != otf.shape[1]:
            raise ValueError("Shift-invariant OTF data are missing or inconsistent.")
        scale = 1e-3 if param_format(units) == "um" else 1.0
        fx = fx_mm * scale
        peak_f = float(np.max(np.abs(fx))) if fx.size > 0 else 0.0
        delta_space = 1.0 / max(2.0 * peak_f, 1e-12) if peak_f > 0.0 else 0.0

        ls_wave = np.empty((_middle_samples_1d(np.arange(2 * int(n_samp), dtype=float), middle_samps).size, wave.size), dtype=float)
        for wave_index in range(wave.size):
            tmp = np.asarray(otf[0, :, wave_index], dtype=complex)
            lsf = np.fft.fftshift(np.fft.ifft(tmp))
            lsf_vector = np.asarray(lsf).reshape(-1)
            half = int(np.rint(float(middle_samps) / 2.0))
            center = int(np.rint(lsf_vector.size / 2.0))
            start = max(0, center - half)
            stop = min(lsf_vector.size, center + half + 1)
            ls_wave[:, wave_index] = np.abs(np.asarray(lsf_vector[start:stop], dtype=lsf_vector.dtype))

        x = np.arange(-int(n_samp), int(n_samp), dtype=float) * float(delta_space)
        x = np.asarray(_middle_samples_1d(x, middle_samps), dtype=float)
        return {"x": x, "wavelength": wave.copy(), "lsWave": ls_wave.T}

    otf_payload = _oi_otf_wavelength_payload(oi, units=units, n_samp=n_samp)
    fx = np.asarray(otf_payload["fSupport"], dtype=float).reshape(-1)
    otf_wave = np.asarray(otf_payload["otf"], dtype=float)
    peak_f = float(np.max(np.abs(fx))) if fx.size > 0 else 0.0
    delta_space = 1.0 / max(2.0 * peak_f, 1e-12) if peak_f > 0.0 else 0.0

    ls_wave = np.empty((_middle_samples_1d(np.arange(2 * int(n_samp), dtype=float), middle_samps).size, wave.size), dtype=float)
    for wave_index in range(wave.size):
        tmp = np.asarray(otf_wave[:, wave_index], dtype=complex)
        lsf = np.fft.fftshift(np.fft.ifft(tmp))
        ls_wave[:, wave_index] = np.abs(_middle_samples_1d(np.asarray(lsf).reshape(-1), middle_samps))

    x = np.arange(-int(n_samp), int(n_samp), dtype=float) * float(delta_space)
    x = np.asarray(_middle_samples_1d(x, middle_samps), dtype=float)
    return {"x": x, "wavelength": wave.copy(), "lsWave": ls_wave.T}


def _sensor_plot_line_data(sensor: Sensor, line_key: str, xy: Any) -> dict[str, Any]:
    key = param_format(line_key)
    orientation = "h" if "hline" in key else "v"
    data_type = "electrons" if "electrons" in key else "dv" if "dv" in key else "volts"
    line_index, xy_array = _line_index("plotSensor", line_key, xy, orientation)
    profile = sensor_get(sensor, f"{orientation}line {data_type}", line_index)
    if profile is None:
        raise ValueError(f"Sensor has no {data_type} data for {line_key}.")
    pix_color = np.array(
        [
            color_index
            for color_index, values in enumerate(profile["data"], start=1)
            if np.asarray(values, dtype=float).size > 0
        ],
        dtype=int,
    )
    return {
        "xy": xy_array,
        "ori": orientation,
        "dataType": data_type,
        "data": [np.asarray(values, dtype=float).copy() for values in profile["data"]],
        "pos": [1e6 * np.asarray(values, dtype=float).copy() for values in profile["pos"]],
        "pixPos": [1e6 * np.asarray(values, dtype=float).copy() for values in profile["pixPos"]],
        "pixColor": pix_color,
        "filterPlotColors": sensor_get(sensor, "filter plot colors"),
        "xLabel": "Position (um)",
        "yLabel": "digital value" if data_type == "dv" else data_type,
        "titleString": f"{'Horizontal' if orientation == 'h' else 'Vertical'} line {line_index}",
    }


def _sensor_plot_two_lines(sensor: Sensor, line_key: str, xy: Any) -> dict[str, Any]:
    key = param_format(line_key)
    orientation = "h" if "hline" in key else "v"
    data_type = "electrons" if "electrons" in key else "dv" if "dv" in key else "volts"
    line_index, xy_array = _line_index("plotSensor", line_key, xy, orientation)
    second_xy = xy_array.copy()
    if second_xy.size == 1:
        second_xy[0] = line_index + 1
    elif orientation == "h":
        second_xy[1] = line_index + 1
    else:
        second_xy[0] = line_index + 1

    max_index = int(sensor_get(sensor, "rows" if orientation == "h" else "cols"))
    if line_index >= max_index:
        raise IndexError("Two-line sensor plot requires an adjacent line within the sensor bounds.")

    first_line = _sensor_plot_line_data(sensor, line_key, xy_array)
    second_line = _sensor_plot_line_data(sensor, line_key, second_xy)
    first_profile = sensor_get(sensor, f"{orientation}line {data_type}", line_index)
    second_profile = sensor_get(sensor, f"{orientation}line {data_type}", line_index + 1)

    pix_pos: list[np.ndarray] = []
    pix_data: list[np.ndarray] = []
    pix_color: list[int] = []
    for line, profile in ((first_line, first_profile), (second_line, second_profile)):
        for color_index, (positions, values, raw_positions) in enumerate(
            zip(line["pos"], line["data"], profile["pos"]), start=1
        ):
            if np.asarray(values).size == 0:
                continue
            pix_pos.append(np.asarray(raw_positions, dtype=float).copy())
            pix_data.append(np.asarray(values, dtype=float).copy())
            pix_color.append(color_index)

    return {
        "xy": xy_array.copy(),
        "xy2": second_xy.copy(),
        "ori": orientation,
        "dataType": data_type,
        "pixPos": pix_pos,
        "pixData": pix_data,
        "pixColor": np.asarray(pix_color, dtype=int),
        "filterPlotColors": sensor_get(sensor, "filter plot colors"),
        "xLabel": "Position (um)",
        "yLabel": "digital value" if data_type == "dv" else data_type,
        "titleString": f"{'Horizontal' if orientation == 'h' else 'Vertical'} line {line_index}",
    }


def _sensor_plot_histogram(sensor: Sensor, data_type: str, roi_locs: Any) -> dict[str, Any]:
    from .roi import vc_get_roi_data

    roi = np.asarray(roi_locs, dtype=int)
    data = np.asarray(vc_get_roi_data(sensor, roi, data_type), dtype=float)
    payload = _roi_payload(roi)
    payload["data"] = data
    payload["unitType"] = data_type
    payload["filterPlotColors"] = sensor_get(sensor, "filter plot colors")
    payload["xLabel"] = {
        "volts": "Volts",
        "electrons": "Electrons",
        "dv": "Digital value",
    }.get(param_format(data_type), str(data_type))
    payload["yLabel"] = "Count"
    return payload


def _sensor_plot_chromaticity(sensor: Sensor, roi_locs: Any | None) -> dict[str, Any]:
    roi = roi_locs if roi_locs is not None else sensor_get(sensor, "roi")
    roi = _roi_required("plotSensor", "chromaticity", roi)
    if int(sensor_get(sensor, "nfilters")) < 2:
        raise UnsupportedOptionError("plotSensor", "chromaticity")
    rg = np.asarray(sensor_get(sensor, "chromaticity", roi), dtype=float)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    sums = np.sum(spectral_qe, axis=1, keepdims=True)
    spectrum_locus = np.divide(
        spectral_qe[:, :2],
        sums,
        out=np.full((spectral_qe.shape[0], 2), np.nan, dtype=float),
        where=sums > 0.0,
    )
    payload = _roi_payload(roi)
    payload["rg"] = rg.copy()
    payload["spectrumlocus"] = spectrum_locus.copy()
    payload["xLabel"] = "r-chromaticity"
    payload["yLabel"] = "g-chromaticity"
    payload["titleString"] = "rg sensor chromaticity"
    return payload


def _sensor_plot_spectra(sensor: Sensor, data_type: str) -> dict[str, Any]:
    key = param_format(data_type)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    if key == "colorfilters":
        data = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
        names = list(sensor_get(sensor, "filter color letters cell"))
        y_label = "Transmittance"
    elif key == "irfilter":
        data = np.asarray(sensor_get(sensor, "ir filter"), dtype=float).reshape(-1, 1)
        names = ["o"]
        y_label = "Transmittance"
    elif key in {"pdspectralqe", "pixelspectralqe"}:
        data = np.asarray(sensor_get(sensor, "pixel spectral qe"), dtype=float).reshape(-1, 1)
        names = ["k"]
        y_label = "QE"
    elif key in {"spectralsr", "sr", "pdspectralsr", "pixelspectralsr"}:
        data = np.asarray(sensor_get(sensor, "pixel spectral sr"), dtype=float).reshape(-1, 1)
        names = ["k"]
        y_label = "Responsivity:  Volts/Watt"
    elif key in {"spectralqe", "sensorspectralqe"}:
        data = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
        names = list(sensor_get(sensor, "filter color letters cell"))
        y_label = "Quantum efficiency"
    elif key in {"sensorspectralsr"}:
        data = np.asarray(sensor_get(sensor, "sensor spectral sr"), dtype=float)
        names = list(sensor_get(sensor, "filter color letters cell"))
        y_label = "Responsivity:  Volts/Watt"
    else:
        raise UnsupportedOptionError("plotSensor", data_type)
    return {
        "x": wave.copy(),
        "y": data.copy(),
        "filterNames": names,
        "dataType": key,
        "xLabel": "Wavelength (nm)",
        "yLabel": y_label,
        "nameString": f"ISET: {key}",
    }


def _cfa_scale_factor(rows: int) -> int:
    if rows < 2:
        return 32
    if rows < 8:
        return 8
    return 1


def _sensor_pattern_letters(sensor: Sensor, pattern: np.ndarray) -> np.ndarray:
    letters = np.array(list(sensor_get(sensor, "filter color letters")), dtype="<U16")
    if letters.size == 0:
        return np.empty(np.asarray(pattern, dtype=int).shape, dtype="<U16")
    pattern_array = np.asarray(pattern, dtype=int)
    return letters[np.clip(pattern_array - 1, 0, letters.size - 1)]


def _sensor_filter_display_colors(sensor: Sensor) -> np.ndarray:
    letters = list(sensor_get(sensor, "filter color letters"))
    filter_spectra = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
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
        wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
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


def _sensor_name_string(sensor: Sensor) -> str:
    return str(sensor_get(sensor, "name"))


def _sensor_plot_cfa(sensor: Sensor, *, full_array: bool) -> dict[str, Any]:
    unit_pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    if full_array:
        rows, cols = sensor_get(sensor, "size")
        pattern = tile_pattern(unit_pattern, int(rows), int(cols))
        mode = "full"
    else:
        pattern = unit_pattern.copy()
        mode = "block"
    filter_colors = _sensor_filter_display_colors(sensor)
    small_img = filter_colors[np.clip(pattern - 1, 0, filter_colors.shape[0] - 1)]
    scale = _cfa_scale_factor(int(small_img.shape[0]))
    if scale > 1:
        img = np.repeat(np.repeat(small_img, scale, axis=0), scale, axis=1)
    else:
        img = small_img.copy()
    return {
        "img": img.copy(),
        "imgSmall": small_img.copy(),
        "pattern": pattern.copy(),
        "unitPattern": unit_pattern.copy(),
        "patternColors": _sensor_pattern_letters(sensor, pattern),
        "unitPatternColors": _sensor_pattern_letters(sensor, unit_pattern),
        "filterNames": list(sensor_get(sensor, "filter color letters cell")),
        "filterColors": filter_colors.copy(),
        "scale": scale,
        "mode": mode,
        "nameString": _sensor_name_string(sensor),
    }


def _sensor_plot_data(sensor: Sensor, *, cfa_constant: bool = False) -> tuple[np.ndarray, str]:
    data_type = "dv" if sensor_get(sensor, "dv") is not None else "volts"
    data = sensor_get(sensor, data_type)
    if data is None:
        rows, cols = sensor_get(sensor, "size")
        fill = 1.0 if cfa_constant else 0.0
        return np.full((int(rows), int(cols)), fill, dtype=float), data_type
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    if cfa_constant:
        array = np.ones(array.shape[:2], dtype=float)
    return np.asarray(array, dtype=float), data_type


def _sensor_plot_true_size(sensor: Sensor) -> dict[str, Any]:
    data, data_type = _sensor_plot_data(sensor)
    gamma = float(sensor_get(sensor, "gamma"))
    scale_max = bool(sensor_get(sensor, "scale max"))
    render_sensor = sensor.clone()
    render_sensor.data.clear()
    render_sensor.data[data_type] = np.asarray(data, dtype=float)
    return {
        "img": np.asarray(sensor_get(render_sensor, "rgb", data_type, gamma, scale_max), dtype=float),
        "dataType": data_type,
        "nameString": _sensor_name_string(sensor),
        "gamma": gamma,
        "scaleMax": scale_max,
    }


def _sensor_plot_cfa_image(sensor: Sensor) -> dict[str, Any]:
    data, _data_type = _sensor_plot_data(sensor, cfa_constant=True)
    gamma = 1.0
    scale_max = False
    render_sensor = sensor.clone()
    render_sensor.data.clear()
    render_sensor.data["volts"] = np.asarray(data, dtype=float)
    return {
        "img": np.asarray(sensor_get(render_sensor, "rgb", "volts", gamma, scale_max), dtype=float),
        "dataType": "volts",
        "nameString": _sensor_name_string(sensor),
        "gamma": gamma,
        "scaleMax": scale_max,
    }


def _sensor_plot_channels(sensor: Sensor) -> dict[str, Any]:
    data, data_type = _sensor_plot_data(sensor)
    rows, cols = data.shape[:2]
    pattern = tile_pattern(np.asarray(sensor_get(sensor, "pattern"), dtype=int), rows, cols)
    filter_colors = _sensor_filter_display_colors(sensor)
    scale = float(sensor_get(sensor, "max digital value" if param_format(data_type) == "dv" else "max output"))
    normalized = np.clip(
        np.asarray(data, dtype=float) / max(scale, 1e-12),
        0.0,
        1.0,
    )
    channel_data: list[np.ndarray] = []
    channel_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index, color in enumerate(filter_colors, start=1):
        mask = pattern == index
        plane = np.full((rows, cols), np.nan, dtype=float)
        plane[mask] = data[mask]
        channel_data.append(plane)
        masks.append(mask.copy())
        tinted = np.zeros((rows, cols, 3), dtype=float)
        if np.any(mask):
            tinted[mask] = normalized[mask, None] * color.reshape(1, 3)
        channel_images.append(linear_to_srgb(tinted))
    return {
        "channelData": channel_data,
        "channelImages": channel_images,
        "masks": masks,
        "pattern": pattern.copy(),
        "filterNames": list(sensor_get(sensor, "filter color letters cell")),
        "dataType": data_type,
        "nameString": _sensor_name_string(sensor),
    }


def _sensor_plot_etendue(sensor: Sensor) -> dict[str, Any]:
    return {
        "support": sensor_get(sensor, "spatial support", "um"),
        "sensorEtendue": np.asarray(sensor_get(sensor, "etendue"), dtype=float),
        "xLabel": "Position (um)",
        "yLabel": "Position (um)",
        "zLabel": "Relative illumination",
        "nameString": f"ISET: Etendue ({sensor_get(sensor, 'vignetting name')})",
    }


def _sensor_fft_payload(sensor: Sensor, orientation: str, data_type: str, xy: Any) -> dict[str, Any]:
    if int(sensor_get(sensor, "nfilters")) > 1:
        raise UnsupportedOptionError("plotSensorFFT", "color sensors")
    line_index, xy_array = _line_index("plotSensorFFT", orientation, xy, orientation)
    data = sensor_get(sensor, data_type)
    if data is None:
        raise ValueError(f"Sensor has no {data_type} data for plotSensorFFT.")
    array = np.asarray(data, dtype=float)
    if array.ndim >= 3:
        array = np.asarray(array[:, :, 0], dtype=float)
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal sensor FFT line index is out of range.")
        line = np.asarray(array[line_index - 1, :], dtype=float)
        title_string = f"ISET:  Horizontal fft {line_index}"
        x_label = "Cycles/deg (col)"
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical sensor FFT line index is out of range.")
        line = np.asarray(array[:, line_index - 1], dtype=float)
        title_string = f"ISET:  Vertical fft {line_index}"
        x_label = "Cycles/deg (row)"
    fov = float(sensor_get(sensor, "fov"))
    cpd = np.arange(0, round((line.size - 1) / 2) + 1, dtype=float) / max(fov, 1e-12)
    n_freq = int(cpd.size)
    mean_value = float(np.mean(line))
    amp = np.abs(np.fft.fft(line - mean_value)) / max(float(n_freq), 1.0)
    peak_contrast = float(np.max(amp) / mean_value) if not np.isclose(mean_value, 0.0) else float(np.inf)
    return {
        "xy": xy_array.copy(),
        "ori": orientation,
        "dataType": data_type,
        "cpd": cpd.copy(),
        "amp": np.asarray(amp, dtype=float).copy(),
        "ampPlot": np.asarray(amp[:n_freq], dtype=float).copy(),
        "mean": mean_value,
        "peakContrast": peak_contrast,
        "titleString": title_string,
        "xLabel": x_label,
        "yLabel": "Abs(fft(data))",
    }


def _sensor_noise_title(the_noise: np.ndarray, voltage_swing: float) -> str:
    noise = np.asarray(the_noise, dtype=float)
    return f"Max/min: [{float(np.max(noise)):.2E},{float(np.min(noise)):.2E}] on voltage swing {float(voltage_swing):.2f}"


def _sensor_plot_shot_noise(sensor: Sensor) -> dict[str, Any]:
    electrons = sensor_get(sensor, "electrons")
    if electrons is None:
        electrons = np.zeros(sensor_get(sensor, "size"), dtype=float)
    electron_image = np.clip(np.asarray(electrons, dtype=float), 0.0, None)
    rng = np.random.default_rng(0)
    electron_noise = np.sqrt(electron_image) * rng.standard_normal(electron_image.shape)
    low_count = electron_image < 25.0
    if np.any(low_count):
        poisson_counts = rng.poisson(electron_image[low_count])
        electron_noise[low_count] = poisson_counts - electron_image[low_count]
    conversion_gain = float(sensor.fields["pixel"]["conversion_gain_v_per_electron"])
    the_noise = conversion_gain * electron_noise
    noisy_image = conversion_gain * np.rint(electron_image + electron_noise)
    return {
        "noiseType": "shotnoise",
        "nameString": "ISET:  Shot noise",
        "titleString": _sensor_noise_title(the_noise, float(sensor_get(sensor, "voltage swing"))),
        "signal": electron_image.copy(),
        "noisyImage": np.asarray(noisy_image, dtype=float).copy(),
        "theNoise": np.asarray(the_noise, dtype=float).copy(),
    }


def _sensor_plot_fixed_pattern_noise(sensor: Sensor, noise_type: str) -> dict[str, Any]:
    rows, cols = sensor_get(sensor, "size")
    rng = np.random.default_rng(0)
    normalized = param_format(noise_type)
    voltage_swing = float(sensor_get(sensor, "voltage swing"))
    if normalized == "dsnu":
        the_noise = sensor_get(sensor, "dsnu image")
        if the_noise is None:
            sigma = float(sensor_get(sensor, "dsnu sigma"))
            the_noise = rng.normal(0.0, sigma, size=(int(rows), int(cols)))
        the_noise = np.asarray(the_noise, dtype=float)
        noisy_image = the_noise.copy()
        name_string = "ISET:  DSNU"
        title_string = _sensor_noise_title(the_noise, voltage_swing)
    elif normalized == "prnu":
        the_noise = sensor_get(sensor, "prnu image")
        if the_noise is None:
            sigma = float(sensor.fields["pixel"]["prnu_sigma"])
            the_noise = 1.0 + rng.normal(0.0, sigma, size=(int(rows), int(cols)))
        the_noise = np.asarray(the_noise, dtype=float)
        noisy_image = the_noise.copy()
        name_string = "ISET:  PRNU"
        title_string = f"Max/min: [{float(np.max(the_noise)):.2E},{float(np.min(the_noise)):.2E}] slope"
    else:
        raise UnsupportedOptionError("plotSensor", noise_type)
    return {
        "noiseType": normalized,
        "nameString": name_string,
        "titleString": title_string,
        "noisyImage": noisy_image.copy(),
        "theNoise": the_noise.copy(),
    }


def _ip_line_data(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("ipPlot", f"{orientation}line", xy, orientation)
    data = ip_get(ip, "result")
    if data is None:
        raise ValueError("IP has no result data for line plotting.")
    array = np.asarray(data, dtype=float)
    if array.ndim != 3:
        raise ValueError("IP result data must be an RGB image for line plotting.")
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal IP line index is out of range.")
        values = np.asarray(array[line_index - 1, :, :], dtype=float)
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical IP line index is out of range.")
        values = np.asarray(array[:, line_index - 1, :], dtype=float)
    return {
        "xy": xy_array,
        "ori": orientation,
        "pos": np.arange(1, values.shape[0] + 1, dtype=float),
        "values": values.copy(),
    }


def _ip_luminance_line_data(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("ipPlot", f"{orientation}line luminance", xy, orientation)
    data = ip_get(ip, "data luminance")
    if data is None:
        raise ValueError("IP has no XYZ data for luminance line plotting.")
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("IP luminance data must be a 2D image.")
    if orientation == "h":
        if line_index < 1 or line_index > array.shape[0]:
            raise IndexError("Horizontal IP line index is out of range.")
        line = np.asarray(array[line_index - 1, :], dtype=float)
    else:
        if line_index < 1 or line_index > array.shape[1]:
            raise IndexError("Vertical IP line index is out of range.")
        line = np.asarray(array[:, line_index - 1], dtype=float)
    return {
        "xy": xy_array,
        "ori": orientation,
        "pos": np.arange(1, line.shape[0] + 1, dtype=float),
        "data": line.copy(),
    }


def _ip_plot_color_data(ip: ImageProcessor, roi_locs: Any) -> tuple[np.ndarray, np.ndarray]:
    from .roi import vc_get_roi_data

    roi = np.asarray(roi_locs, dtype=int)
    rgb = np.asarray(vc_get_roi_data(ip, roi, "result"), dtype=float)
    xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    return rgb, xyz


def _ip_plot_white_point(ip: ImageProcessor, roi_xyz: np.ndarray) -> np.ndarray:
    white_point = ip_get(ip, "data white point")
    if white_point is not None:
        return np.asarray(white_point, dtype=float).reshape(3)
    return np.mean(np.asarray(roi_xyz, dtype=float), axis=0).reshape(3)


def scene_plot(
    scene: Scene,
    p_type: str = "hlineluminance",
    roi_locs: Any | None = None,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotScene` user-data without opening a figure."""

    key = param_format(p_type)

    def _mean_illuminant(field: str) -> np.ndarray:
        illuminant = np.asarray(scene_get(scene, field, asset_store=asset_store), dtype=float)
        if illuminant.ndim == 1:
            return illuminant.reshape(-1).copy()
        return np.mean(illuminant.reshape(-1, illuminant.shape[-1]), axis=0, dtype=float).reshape(-1)

    if key == "radianceenergyroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        energy = np.mean(np.asarray(scene_get(scene, "roi energy", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "energy": energy}, None
    if key == "radiancephotonsroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        photons = np.mean(np.asarray(scene_get(scene, "roi photons", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "photons": photons}, None
    if key in {"radiancehline", "hlineradiance", "radiancevline", "vlineradiance", "luminancehline", "hlineluminance", "luminancevline", "vlineluminance"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        udata = dict(scene_get(scene, p_type, roi, *(args[:1] if args else ()), asset_store=asset_store))
        udata["roiLocs"] = np.asarray(roi, dtype=int).copy()
        return udata, None
    if key in {"reflectanceroi", "reflectance"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        reflectance = np.mean(np.asarray(scene_get(scene, "roi reflectance", roi, asset_store=asset_store), dtype=float), axis=0).reshape(-1)
        return {"wave": np.asarray(scene_get(scene, "wave"), dtype=float), "reflectance": reflectance}, None
    if key == "luminanceroi":
        roi = _roi_required("scenePlot", p_type, roi_locs)
        return {"lum": np.asarray(scene_get(scene, "roi luminance", roi, asset_store=asset_store), dtype=float), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {"chromaticityroi", "chromaticity"}:
        roi = _roi_required("scenePlot", p_type, roi_locs)
        data = np.asarray(scene_get(scene, "chromaticity", roi, asset_store=asset_store), dtype=float)
        return {"x": data[:, 0].copy(), "y": data[:, 1].copy(), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {"illuminantenergyroi", "illuminantenergy"}:
        if "roi" in key:
            if roi_locs is None:
                energy = _mean_illuminant("illuminant energy")
            else:
                roi = _roi_required("scenePlot", p_type, roi_locs)
                energy = np.asarray(scene_get(scene, "roi mean illuminant energy", roi, asset_store=asset_store), dtype=float).reshape(-1)
        else:
            energy = _mean_illuminant("illuminant energy")
        return {
            "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
            "energy": energy,
            "comment": scene_get(scene, "illuminant comment", asset_store=asset_store),
        }, None
    if key in {"illuminantphotonsroi", "illuminantphotons"}:
        if "roi" in key:
            if roi_locs is None:
                photons = _mean_illuminant("illuminant photons")
            else:
                roi = _roi_required("scenePlot", p_type, roi_locs)
                photons = np.asarray(scene_get(scene, "roi mean illuminant photons", roi, asset_store=asset_store), dtype=float).reshape(-1)
        else:
            photons = _mean_illuminant("illuminant photons")
        return {
            "wave": np.asarray(scene_get(scene, "wave"), dtype=float),
            "photons": photons,
            "comment": scene_get(scene, "illuminant comment", asset_store=asset_store),
        }, None
    if key == "illuminantimage":
        wave = np.asarray(scene_get(scene, "wave"), dtype=float)
        rows, cols = map(int, scene_get(scene, "size"))
        energy = np.asarray(scene_get(scene, "illuminant energy", asset_store=asset_store), dtype=float)
        if energy.size == 0:
            raise ValueError("No illuminant data.")
        if param_format(scene_get(scene, "illuminant format", asset_store=asset_store)) == "spectral":
            energy = np.broadcast_to(energy.reshape(1, 1, -1), (rows, cols, energy.size)).copy()
        elif energy.ndim == 2:
            energy = xw_to_rgb_format(energy, rows, cols)
        elif energy.ndim != 3:
            raise ValueError("Scene illuminant energy must be spectral or spatial-spectral.")
        srgb = xyz_to_srgb(np.asarray(xyz_from_energy(energy, wave, asset_store=asset_store), dtype=float))
        return {"srgb": srgb}, None
    raise UnsupportedOptionError("scenePlot", p_type)


def oi_plot(
    oi: OpticalImage,
    p_type: str = "illuminance hline",
    roi_locs: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `oiPlot` user-data without opening a figure."""

    key = param_format(p_type)
    if key in {"psf", "psf550", "psfxaxis", "psfyaxis"}:
        default_wave = 550.0 if key == "psf550" else float(np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)[0]) if np.asarray(oi_get(oi, "wave"), dtype=float).size == 1 else 550.0
        remaining = list(args)
        this_wave = default_wave
        units = "um"
        if remaining:
            candidate = remaining[0]
            candidate_array = np.asarray(candidate) if not isinstance(candidate, str) else None
            if candidate_array is not None and candidate_array.size == 1:
                this_wave = float(candidate_array.reshape(-1)[0])
                remaining.pop(0)
        if remaining and isinstance(remaining[0], str) and param_format(remaining[0]) not in {"airydisk", "airydisk"}:
            units = str(remaining.pop(0))
        if key in {"psf", "psf550"}:
            psf_data = dict(oi_get(oi, "psf data", this_wave, units))
            xy = np.asarray(psf_data.pop("xy"), dtype=float)
            udata = {"x": xy[:, :, 0].copy(), "y": xy[:, :, 1].copy(), **psf_data}
            udata["wave"] = float(this_wave)
            udata["units"] = units
            airy_disk_flag = bool(_plot_option(tuple(remaining), "airydisk", True)) if remaining else True
            udata["airyDisk"] = airy_disk_flag
            if airy_disk_flag:
                radius = float(airy_disk(this_wave, float(oi_get(oi, "fnumber")), "units", units))
                udata["airyDiskRadius"] = radius
                udata["airyDiskDiameter"] = radius * 2.0
                udata["airyDiskCircle"] = _airy_disk_circle(radius)
            return udata, None
        udata = dict(oi_get(oi, "psf xaxis" if key == "psfxaxis" else "psf yaxis", this_wave, units))
        udata["wave"] = float(this_wave)
        udata["units"] = units
        airy_disk_flag = bool(_plot_option(tuple(remaining), "airydisk", False))
        udata["airyDisk"] = airy_disk_flag
        if airy_disk_flag:
            radius = float(airy_disk(this_wave, float(oi_get(oi, "fnumber")), "units", units))
            udata["airyDiskRadius"] = radius
            udata["airyDiskDiameter"] = radius * 2.0
        return udata, None
    if key in {"lswavelength", "lsfwavelength"}:
        middle_samps = int(args[0]) if args else 40
        udata = _oi_lswavelength_payload(oi, units="um", middle_samps=middle_samps, n_samp=100)
        return udata, None
    if key in {"otfwavelength", "mtfwavelength"}:
        udata = _oi_otf_wavelength_payload(oi, units="um", n_samp=100)
        return udata, None
    if key == "irradiancephotonsroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        irradiance = np.mean(np.asarray(oi_get(oi, "roi photons", roi), dtype=float), axis=0).reshape(-1)
        return {"x": np.asarray(oi_get(oi, "wave"), dtype=float), "y": irradiance, "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key == "irradianceenergyroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        irradiance = np.mean(np.asarray(oi_get(oi, "roi energy", roi), dtype=float), axis=0).reshape(-1)
        return {"x": np.asarray(oi_get(oi, "wave"), dtype=float), "y": irradiance, "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key in {
        "irradiancehline",
        "hline",
        "hlineirradiance",
        "irradiancevline",
        "vline",
        "vlineirradiance",
        "irradianceenergyhline",
        "hlineenergy",
        "hlineirradianceenergy",
        "irradianceenergyvline",
        "vlineenergy",
        "vlineirradianceenergy",
        "illuminancehline",
        "horizontallineilluminance",
        "hlineilluminance",
        "illuminancevline",
        "vlineilluminance",
    }:
        roi = _roi_required("oiPlot", p_type, roi_locs)
        udata = dict(oi_get(oi, p_type, roi, *(args[:1] if args else ())))
        udata["roiLocs"] = np.asarray(roi, dtype=int).copy()
        return udata, None
    if key == "illuminanceroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        return {"illum": np.asarray(oi_get(oi, "roi illuminance", roi), dtype=float), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    if key == "chromaticityroi":
        roi = _roi_required("oiPlot", p_type, roi_locs)
        data = np.asarray(oi_get(oi, "chromaticity", roi), dtype=float)
        return {"x": data[:, 0].copy(), "y": data[:, 1].copy(), "roiLocs": np.asarray(roi, dtype=int).copy()}, None
    raise UnsupportedOptionError("oiPlot", p_type)


def _wvf_default_wave(wvf: dict[str, Any]) -> float:
    wave = np.asarray(wvf_get(wvf, "wave"), dtype=float).reshape(-1)
    if wave.size == 0:
        return 550.0
    if wave.size == 1:
        return float(wave[0])
    return 550.0


def _wvf_crop_axis_and_plane(axis: np.ndarray, plane: np.ndarray, plot_range: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(float(plot_range)):
        return axis.copy(), plane.copy()
    index = np.abs(axis) < float(plot_range)
    cropped_axis = np.asarray(axis[index], dtype=float)
    if plane.ndim == 2:
        return cropped_axis, np.asarray(plane[np.ix_(index, index)], dtype=float)
    return cropped_axis, np.asarray(plane[index], dtype=float)


def _wvf_line_payload(wvf: dict[str, Any], key: str, wave: float, unit: str, plot_range: float) -> dict[str, Any]:
    samp = np.asarray(wvf_get(wvf, "psf spatial samples", unit, wave), dtype=float)
    psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
    middle = psf.shape[0] // 2
    if key == "psfxaxis":
        data = np.asarray(psf[middle, :], dtype=float)
    else:
        data = np.asarray(psf[:, middle], dtype=float)
    return {"samp": samp, "data": data, "wave": float(wave), "unit": unit}


def _wvf_centered_otf(psf: np.ndarray) -> np.ndarray:
    return np.asarray(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.asarray(psf, dtype=float)))), dtype=np.complex128)


def wvf_plot(
    wvf: dict[str, Any],
    p_type: str = "psf",
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `wvfPlot` user-data without opening a figure."""

    key = param_format(p_type)
    unit = str(_plot_option(args, "unit", "mm"))
    wave = float(_plot_option(args, "wave", _wvf_default_wave(wvf)))
    plot_range = float(_plot_option(args, "plotrange", _plot_option(args, "plot range", np.inf)))
    show_airy_disk = bool(_plot_option(args, "airydisk", False))

    if key in {"psf", "psfnormalized", "imagepsf", "imagepsfnormalized"}:
        samp = np.asarray(wvf_get(wvf, "psf spatial samples", unit, wave), dtype=float)
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        if "normalized" in key:
            psf = psf / max(float(np.max(psf)), 1e-12)
        samp, psf = _wvf_crop_axis_and_plane(samp, psf, plot_range)
        udata = {
            "x": samp,
            "y": samp.copy(),
            "z": psf,
            "wave": float(wave),
            "unit": unit,
            "airyDisk": show_airy_disk,
        }
        if show_airy_disk:
            radius = float(airy_disk(wave, float(wvf_get(wvf, "fnumber")), "units", unit))
            udata["airyDiskRadius"] = radius
            udata["airyDiskDiameter"] = radius * 2.0
            udata["airyDiskCircle"] = _airy_disk_circle(radius)
        return udata, None

    if key in {"psfangle", "2dpsfangle", "2dpsfanglenormalized", "imagepsfangle", "imagepsfanglenormalized"}:
        samp = np.asarray(wvf_get(wvf, "psf angular samples", unit, wave), dtype=float)
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        if "normalized" in key:
            psf = psf / max(float(np.max(psf)), 1e-12)
        samp, psf = _wvf_crop_axis_and_plane(samp, psf, plot_range)
        return {"x": samp, "y": samp.copy(), "z": psf, "wave": float(wave), "unit": unit}, None

    if key in {"psfxaxis", "psfyaxis"}:
        udata = _wvf_line_payload(wvf, key, wave, unit, plot_range)
        udata["airyDisk"] = show_airy_disk
        if show_airy_disk:
            radius = float(airy_disk(wave, float(wvf_get(wvf, "fnumber")), "units", unit))
            udata["airyDiskRadius"] = radius
            udata["airyDiskDiameter"] = radius * 2.0
        return udata, None

    if key in {"1dpsf", "1dpsfspace", "1dpsfnormalized"}:
        samp = np.asarray(wvf_get(wvf, "psf spatial samples", unit, wave), dtype=float)
        line = np.asarray(wvf_get(wvf, "1d psf", wave), dtype=float)
        if "normalized" in key:
            line = line / max(float(np.max(line)), 1e-12)
        samp, line = _wvf_crop_axis_and_plane(samp, line, plot_range)
        udata = {"x": samp, "y": line, "wave": float(wave), "unit": unit, "airyDisk": show_airy_disk}
        if show_airy_disk:
            radius = float(airy_disk(wave, float(wvf_get(wvf, "fnumber")), "units", unit))
            udata["airyDiskRadius"] = radius
            udata["airyDiskDiameter"] = radius * 2.0
        return udata, None

    if key in {"1dpsfangle", "1dpsfanglenormalized"}:
        samp = np.asarray(wvf_get(wvf, "psf angular samples", unit, wave), dtype=float)
        line = np.asarray(wvf_get(wvf, "1d psf", wave), dtype=float)
        if "normalized" in key:
            line = line / max(float(np.max(line)), 1e-12)
        samp, line = _wvf_crop_axis_and_plane(samp, line, plot_range)
        return {"x": samp, "y": line, "wave": float(wave), "unit": unit}, None

    if key in {"imagepupilamp", "imagepupilamplitude", "imagepupilampspace", "2dpupilamplitudespace"}:
        samp = np.asarray(wvf_get(wvf, "pupil spatial samples", unit, wave), dtype=float)
        pupil = np.asarray(wvf_get(wvf, "pupil function", wave), dtype=np.complex128)
        amp = np.abs(pupil)
        samp, amp = _wvf_crop_axis_and_plane(samp, amp, plot_range)
        return {"x": samp, "y": samp.copy(), "z": amp, "wave": float(wave), "unit": unit}, None

    if key in {"imagepupilphase", "2dpupilphasespace"}:
        samp = np.asarray(wvf_get(wvf, "pupil spatial samples", unit, wave), dtype=float)
        phase = np.asarray(wvf_get(wvf, "pupil phase", wave), dtype=float)
        samp, phase = _wvf_crop_axis_and_plane(samp, phase, plot_range)
        return {"x": samp, "y": samp.copy(), "z": phase, "wave": float(wave), "unit": unit}, None

    if key in {"imagewavefrontaberrations", "2dwavefrontaberrationsspace"}:
        samp = np.asarray(wvf_get(wvf, "pupil spatial samples", unit, wave), dtype=float)
        wavefront = np.asarray(wvf_get(wvf, "wavefront aberrations", wave), dtype=float)
        samp, wavefront = _wvf_crop_axis_and_plane(samp, wavefront, plot_range)
        return {"x": samp, "y": samp.copy(), "z": wavefront, "wave": float(wave), "unit": unit}, None

    if key in {"2dotf", "otfspace", "otf", "2dotfnormalized", "otfspacenormalized", "otfnormalized"}:
        freq = np.asarray(wvf_get(wvf, "otf support", unit, wave), dtype=float)
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        if "normalized" in key:
            psf = psf / max(float(np.max(psf)), 1e-12)
        otf = np.abs(_wvf_centered_otf(psf))
        freq, otf = _wvf_crop_axis_and_plane(freq, otf, plot_range)
        return {"fx": freq, "fy": freq.copy(), "otf": otf, "wave": float(wave), "unit": unit}, None

    if key in {
        "1dotf",
        "1dotfspace",
        "1dotfangle",
        "1dotfnormalized",
        "1dotfspacenormalized",
        "1dotfanglenormalized",
    }:
        freq = np.asarray(wvf_get(wvf, "otf support", unit, wave), dtype=float)
        psf = np.asarray(wvf_get(wvf, "psf", wave), dtype=float)
        if "normalized" in key:
            psf = psf / max(float(np.max(psf)), 1e-12)
        otf = np.abs(_wvf_centered_otf(psf))
        freq, otf = _wvf_crop_axis_and_plane(freq, otf, plot_range)
        return {
            "fx": freq,
            "fy": freq.copy(),
            "otf": otf,
            "wave": float(wave),
            "unit": "deg" if "angle" in key else unit,
        }, None

    raise UnsupportedOptionError("wvfPlot", p_type)


def _sensor_plot_select_capture(sensor: Sensor, capture: Any) -> Sensor:
    capture_index = int(np.rint(float(capture)))
    if capture_index < 1:
        raise IndexError("Sensor capture index must be positive and 1-based.")

    n_captures = int(sensor_get(sensor, "n captures"))
    if capture_index > n_captures:
        raise IndexError(f"Requested sensor capture {capture_index} exceeds available captures ({n_captures}).")
    if n_captures <= 1:
        return sensor

    selected = sensor.clone()
    capture_zero_based = capture_index - 1
    for key in ("volts", "dv"):
        data = selected.data.get(key)
        if data is None:
            continue
        array = np.asarray(data, dtype=float)
        if array.ndim >= 3:
            selected.data[key] = np.asarray(array[:, :, capture_zero_based], dtype=float).copy()
    integration_time = np.asarray(selected.fields.get("integration_time"))
    if integration_time.ndim > 0 and integration_time.size > 1:
        selected.fields["integration_time"] = float(integration_time.reshape(-1)[capture_zero_based])
    return selected


def sensor_plot(
    sensor: Sensor,
    p_type: str = "volts hline",
    roi_locs: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensor` user-data without opening a figure."""

    two_lines = bool(_plot_option(args, "twolines", False))
    sensor = _sensor_plot_select_capture(sensor, _plot_option(args, "capture", 1))
    key = param_format(p_type)
    if key in {"electronshline", "hlineelectrons", "electronsvline", "vlineelectrons", "voltshline", "hlinevolts", "voltsvline", "vlinevolts", "dvhline", "hlinedv", "dvvline", "vlinedv"}:
        xy = _roi_required("plotSensor", p_type, roi_locs)
        if two_lines:
            return _sensor_plot_two_lines(sensor, key, xy), None
        return _sensor_plot_line_data(sensor, key, xy), None
    if key == "chromaticity":
        return _sensor_plot_chromaticity(sensor, roi_locs), None
    if key == "shotnoise":
        return _sensor_plot_shot_noise(sensor), None
    if key in {"dsnu", "prnu"}:
        return _sensor_plot_fixed_pattern_noise(sensor, key), None
    if key == "etendue":
        return _sensor_plot_etendue(sensor), None
    if key == "channels":
        return _sensor_plot_channels(sensor), None
    if key == "truesize":
        return _sensor_plot_true_size(sensor), None
    if key in {"cfa", "cfablock"}:
        return _sensor_plot_cfa(sensor, full_array=False), None
    if key == "cfaimage":
        return _sensor_plot_cfa_image(sensor), None
    if key == "cfafull":
        return _sensor_plot_cfa(sensor, full_array=True), None
    if key == "pixelsnr":
        snr, volts, snr_shot, snr_read = pixel_snr(sensor)
        return {
            "volts": volts,
            "snr": snr,
            "snrShot": snr_shot,
            "snrRead": snr_read,
            "xLabel": "Signal (V)",
            "yLabel": "SNR (db)",
            "titleString": "Pixel SNR over response range",
            "legend": ["Total pixel SNR", "Shot noise SNR", "Read noise SNR"],
        }, None
    if key in {"sensorsnr", "snr"}:
        snr, volts, snr_shot, snr_read, snr_dsnu, snr_prnu = sensor_snr(sensor)
        legend = ["Total", "Shot"]
        if np.all(np.isfinite(np.asarray(snr_read, dtype=float))):
            legend.append("Read")
        if np.all(np.isfinite(np.asarray(snr_dsnu, dtype=float))):
            legend.append("DSNU")
        if np.all(np.isfinite(np.asarray(snr_prnu, dtype=float))):
            legend.append("PRNU")
        return {
            "volts": volts,
            "snr": snr,
            "snrShot": snr_shot,
            "snrRead": snr_read,
            "snrDSNU": snr_dsnu,
            "snrPRNU": snr_prnu,
            "xLabel": "Signal (V)",
            "yLabel": "SNR (db)",
            "titleString": "Sensor SNR over response range",
            "legend": legend,
        }, None
    if key == "colorfilters":
        return _sensor_plot_spectra(sensor, key), None
    if key == "irfilter":
        return _sensor_plot_spectra(sensor, key), None
    if key in {"pdspectralqe", "pixelspectralqe"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"spectralsr", "sr", "pdspectralsr", "pixelspectralsr"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"spectralqe", "sensorspectralqe"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"sensorspectralsr"}:
        return _sensor_plot_spectra(sensor, key), None
    if key in {"voltshistogram", "voltshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "volts", roi), None
    if key in {"electronshistogram", "electronshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "electrons", roi), None
    if key in {"dvhistogram", "dvhist", "digitalcountshistogram", "digitalcountshist"}:
        roi = _roi_required("plotSensor", p_type, roi_locs)
        return _sensor_plot_histogram(sensor, "dv", roi), None
    raise UnsupportedOptionError("plotSensor", p_type)


def sensor_plot_fft(
    sensor: Sensor,
    ori: str = "h",
    data_type: str = "volts",
    xy: Any | None = None,
    *args: Any,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensorFFT` user-data without opening a figure."""

    sensor = _sensor_plot_select_capture(sensor, _plot_option(args, "capture", 1))
    orientation_key = param_format(ori)
    if orientation_key in {"h", "horizontal"}:
        orientation = "h"
    elif orientation_key in {"v", "vertical"}:
        orientation = "v"
    else:
        raise UnsupportedOptionError("plotSensorFFT", ori)
    selector = _roi_required("plotSensorFFT", f"{ori} {data_type}", xy)
    return _sensor_fft_payload(sensor, orientation, param_format(data_type), selector), None


def sensor_plot_line(
    sensor: Sensor,
    ori: str = "h",
    data_type: str = "dv",
    s_or_t: str = "space",
    xy: Any | None = None,
) -> tuple[None, dict[str, Any]]:
    """Return MATLAB-style `sensorPlotLine` user-data without opening a figure."""

    orientation_key = param_format(ori)
    if orientation_key in {"h", "horizontal"}:
        orientation = "h"
    elif orientation_key in {"v", "vertical"}:
        orientation = "v"
    else:
        raise UnsupportedOptionError("sensorPlotLine", ori)

    data_key = param_format(data_type)
    if data_key not in {"volts", "electrons", "dv"}:
        raise UnsupportedOptionError("sensorPlotLine", data_type)

    domain_key = param_format(s_or_t)
    if domain_key in {"spatial", "space", "spacedomain"}:
        selector = _roi_required("sensorPlotLine", f"{ori} {data_type}", xy)
        line_key = f"{data_key} {'hline' if orientation == 'h' else 'vline'}"
        payload = _sensor_plot_line_data(sensor, line_key, selector)
        if int(sensor_get(sensor, "nfilters")) > 1:
            return None, {
                "pos": [np.asarray(values, dtype=float).copy() for values in payload["pos"]],
                "data": [np.asarray(values, dtype=float).copy() for values in payload["data"]],
                "pixColor": [int(color) for color in np.asarray(payload["pixColor"], dtype=int).reshape(-1)],
            }
        return None, {
            "pixPos": np.asarray(payload["pos"][0], dtype=float).copy(),
            "pixData": np.asarray(payload["data"][0], dtype=float).copy(),
        }

    if domain_key in {"transform", "fourier", "fourierdomain", "fft"}:
        selector = _roi_required("sensorPlotLine", f"{ori} {data_type}", xy)
        payload = _sensor_fft_payload(sensor, orientation, data_key, selector)
        return None, {
            "freq": np.asarray(payload["cpd"], dtype=float).copy(),
            "amp": np.asarray(payload["ampPlot"], dtype=float).copy(),
            "mean": float(payload["mean"]),
            "peakContrast": float(payload["peakContrast"]),
        }

    raise UnsupportedOptionError("sensorPlotLine", s_or_t)


def sensor_plot_hist(
    sensor: Sensor,
    unit_type: str = "electrons",
    roi_locs: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `sensorPlotHist` user-data without opening a figure."""

    roi = _roi_required("sensorPlotHist", unit_type, roi_locs)
    key = param_format(unit_type)
    if key in {"v", "volts"}:
        data_type = "volts"
    elif key in {"e", "electrons"}:
        data_type = "electrons"
    elif key in {"dv", "digitalvalues"}:
        data_type = "dv"
    else:
        raise UnsupportedOptionError("sensorPlotHist", unit_type)
    return _sensor_plot_histogram(sensor, data_type, roi), None


def plot_pixel_snr(sensor: Sensor) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotPixelSNR` payload without opening a figure."""

    return sensor_plot(sensor, "pixel snr")


def plot_sensor_etendue(sensor: Sensor) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensorEtendue` payload without opening a figure."""

    return sensor_plot(sensor, "etendue")


def plot_sensor_snr(sensor: Sensor) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotSensorSNR` payload without opening a figure."""

    return sensor_plot(sensor, "sensor snr")


def plot_ml(
    ml: dict[str, Any] | None = None,
    p_type: str = "offsets",
    sensor: Sensor | None = None,
    *,
    session: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotML` payloads without opening a figure."""

    current_ml = ml if ml is not None else ml_get_current(sensor, session=session)
    if current_ml is None:
        raise ValueError("plotML requires a microlens payload.")

    current_sensor = sensor
    if current_sensor is None and session is not None:
        from .session import session_get_selected

        selected = session_get_selected(session, "sensor")
        if isinstance(selected, Sensor):
            current_sensor = selected

    key = param_format(p_type)
    if key == "offsets":
        if current_sensor is None:
            raise ValueError("plotML(..., 'offsets') requires an explicit sensor or a session with a selected sensor.")
        support = sensor_get(current_sensor, "spatial support", "mm")
        optimal_offsets = np.asarray(mlens_get(current_ml, "optimal offsets", current_sensor), dtype=float)
        return {
            "support": {
                "x": np.asarray(support["x"], dtype=float).copy(),
                "y": np.asarray(support["y"], dtype=float).copy(),
            },
            "optimalOffsets": optimal_offsets.copy(),
            "xLabel": "Position (mm)",
            "yLabel": "Position (mm)",
            "zLabel": "Optimal offset (um) toward center",
            "command": "mesh(support.y, support.x, optimalOffsets)",
        }, None

    if key in {"meshpixelirradiance", "pixelirradiance"}:
        irradiance = np.asarray(mlens_get(current_ml, "pixel irradiance"), dtype=float)
        x = np.asarray(mlens_get(current_ml, "x coordinate"), dtype=float).reshape(-1)
        return {
            "x": x.copy(),
            "y": x.copy(),
            "pixelIrradiance": irradiance.copy(),
            "xLabel": "Position (um)",
            "yLabel": "Position (um)",
            "zLabel": "Relative irradiance",
            "colormap": _hot_colormap(256)[29:220].copy(),
        }, None

    if key == "imagepixelirradiance":
        irradiance = np.asarray(mlens_get(current_ml, "pixel irradiance"), dtype=float)
        x = np.asarray(mlens_get(current_ml, "x coordinate"), dtype=float).reshape(-1)
        pixel_width_um = float(mlens_get(current_ml, "diameter", "microns"))
        image = irradiance.copy()
        positive_index = int(np.argmin(np.abs(x - (pixel_width_um / 2.0))))
        negative_index = int(np.argmin(np.abs((-x) - (pixel_width_um / 2.0))))
        for index in {positive_index, negative_index}:
            image[index, :] = 1.0
            image[:, index] = 1.0
        return {
            "x": x.copy(),
            "y": x.copy(),
            "pixelIrradiance": irradiance.copy(),
            "image": image,
            "boundaryIndices": np.asarray([negative_index + 1, positive_index + 1], dtype=int),
            "colormap": _hot_colormap(64),
            "axis": "image",
            "xLabel": "Position (um)",
            "titleString": "Pixel efficiency (normalized)",
            "colorbarTicks": np.arange(0.0, 1.01, 0.25, dtype=float),
        }, None

    raise UnsupportedOptionError("plotML", p_type)


def plot_metrics(
    handles: Any,
    plot_type: str | None = None,
    rect: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotMetrics` histogram payload without opening a figure."""

    metric_name = str(metrics_get(handles, "currentmetric") if plot_type is None else plot_type)
    image1_name = str(metrics_get(handles, "image1name"))
    image2_name = str(metrics_get(handles, "image2name"))
    roi_locs = np.asarray(metrics_roi(handles, "img1", rect), dtype=int)
    metric_image = np.asarray(metrics_get(handles, "metricdata"), dtype=float)
    if metric_image.ndim != 2:
        metric_image = np.asarray(np.squeeze(metric_image), dtype=float)
    if metric_image.ndim != 2:
        raise ValueError("plotMetrics expects a 2D metric image.")

    row_index = np.clip(roi_locs[:, 0] - 1, 0, metric_image.shape[0] - 1)
    col_index = np.clip(roi_locs[:, 1] - 1, 0, metric_image.shape[1] - 1)
    data = np.asarray(metric_image[row_index, col_index], dtype=float).reshape(-1)
    n_bins = max(10, int(np.ceil(data.size / 10.0)))
    counts, edges = np.histogram(data, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    stats = {
        "mean": float(np.mean(data)) if data.size else 0.0,
        "median": float(np.median(data)) if data.size else 0.0,
        "std": float(np.std(data)) if data.size else 0.0,
        "min": float(np.min(data)) if data.size else 0.0,
        "max": float(np.max(data)) if data.size else 0.0,
    }
    annotation = plot_text_string(
        (
            f"Mean   {stats['mean']:.02f}\n"
            f"Median {stats['median']:.02f}\n"
            f"SD     {stats['std']:.02f}\n"
            f"Min    {stats['min']:.02f}\n"
            f"Max   {stats['max']:.02f}"
        ),
        "ur",
        ax={
            "xlim": (
                float(edges[0]) if edges.size else 0.0,
                float(edges[-1]) if edges.size else 1.0,
            ),
            "ylim": (
                0.0,
                float(np.max(counts)) if counts.size and float(np.max(counts)) > 0.0 else 1.0,
            ),
        },
    )

    payload = _roi_payload(roi_locs)
    payload.update(
        {
            "metricName": metric_name,
            "image1Name": image1_name,
            "image2Name": image2_name,
            "data": data.copy(),
            "nBins": int(n_bins),
            "histogram": counts.astype(float).copy(),
            "binEdges": edges.astype(float).copy(),
            "binCenters": centers.astype(float).copy(),
            "xLabel": metric_name,
            "yLabel": "Count",
            "titleString": f"ROI: {image1_name} and {image2_name} ",
            "annotation": annotation,
            "stats": stats,
            "grid": True,
        }
    )
    return payload, None


def sensor_plot_color(sensor: Sensor, plot_type: str = "rg") -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `sensorPlotColor` payload without opening a figure."""

    key = param_format(plot_type)
    if key == "rg":
        indices = (0, 1)
    elif key == "rb":
        indices = (0, 2)
    else:
        raise UnsupportedOptionError("sensorPlotColor", plot_type)

    ip = ip_create(sensor=sensor)
    demosaiced = np.asarray(demosaic(ip, sensor), dtype=float)
    if demosaiced.ndim != 3 or demosaiced.shape[2] < 3:
        raise ValueError("sensorPlotColor requires a demosaiced RGB sensor result.")

    labels = ["Red sensor", "Green sensor", "Blue sensor"]

    d1 = np.asarray(demosaiced[:, :, indices[0]], dtype=float)
    d2 = np.asarray(demosaiced[:, :, indices[1]], dtype=float)
    d = float(np.max(np.sqrt(d1 * d1 + d2 * d2))) if d1.size else 0.0
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float).reshape(-1)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    ctemps = np.asarray([2500, 3000, 3500, 4000, 4500, 5500, 6500, 8000, 10500], dtype=float)
    reference_points: list[dict[str, Any]] = []
    for ctemp in ctemps:
        spec = np.asarray(blackbody(wave, ctemp, kind="quanta"), dtype=float).reshape(-1)
        rgb = np.asarray(spectral_qe.T @ spec, dtype=float).reshape(-1)
        denom = float(np.linalg.norm(rgb[list(indices)]))
        if denom > 0.0 and d > 0.0:
            rgb = 0.9 * d * (rgb / denom)
        else:
            rgb = np.zeros_like(rgb, dtype=float)
        reference_points.append(
            {
                "temperatureK": float(ctemp),
                "point": np.asarray([rgb[indices[0]], rgb[indices[1]]], dtype=float),
                "label": f"{round(ctemp / 100.0) / 10.0:.1f}K",
            }
        )

    return {
        "name": "sensorColorPlot",
        "type": str(plot_type),
        "x": d1.reshape(-1).copy(),
        "y": d2.reshape(-1).copy(),
        "d1": d1.copy(),
        "d2": d2.copy(),
        "labels": [labels[indices[0]], labels[indices[1]]],
        "referencePoints": reference_points,
        "xlim": np.asarray([0.0, d], dtype=float),
        "ylim": np.asarray([0.0, d], dtype=float),
        "axisEqual": True,
        "grid": True,
        "titleString": "Sensor Color Balance",
    }, None


def ip_plot(
    ip: ImageProcessor,
    p_type: str = "horizontal line",
    roi_locs: Any | None = None,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `ipPlot` user-data without opening a figure."""

    del args, asset_store
    key = param_format(p_type)
    if key in {"horizontalline", "hline"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_line_data(ip, "h", xy), None
    if key in {"verticalline", "vline"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_line_data(ip, "v", xy), None
    if key in {"horizontallineluminance", "hlineluminance"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_luminance_line_data(ip, "h", xy), None
    if key in {"verticallineluminance", "vlineluminance"}:
        xy = _roi_required("ipPlot", p_type, roi_locs)
        return _ip_luminance_line_data(ip, "v", xy), None
    if key == "chromaticity":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        data = np.asarray(ip_get(ip, "chromaticity", roi), dtype=float)
        xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
        payload = _roi_payload(roi)
        payload["x"] = data[:, 0].copy()
        payload["y"] = data[:, 1].copy()
        payload["XYZ"] = xyz.copy()
        return payload, None
    if key in {"rgbhistogram", "rgb"}:
        roi = _roi_required("ipPlot", p_type, roi_locs)
        rgb, _ = _ip_plot_color_data(ip, roi)
        payload = _roi_payload(roi)
        payload["RGB"] = rgb.copy()
        payload["meanRGB"] = np.mean(rgb, axis=0).reshape(-1)
        return payload, None
    if key == "rgb3d":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        rgb, _ = _ip_plot_color_data(ip, roi)
        payload = _roi_payload(roi)
        payload["RGB"] = rgb.copy()
        return payload, None
    if key == "luminance":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        luminance = np.asarray(xyz[:, 1], dtype=float).reshape(-1)
        payload = _roi_payload(roi)
        payload["luminance"] = luminance.copy()
        payload["meanL"] = float(np.mean(luminance))
        payload["stdLum"] = float(np.std(luminance))
        return payload, None
    if key == "cielab":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        white_point = _ip_plot_white_point(ip, xyz)
        lab = np.asarray(xyz_to_lab(xyz, white_point), dtype=float)
        payload = _roi_payload(roi)
        payload["LAB"] = lab.copy()
        payload["whitePoint"] = np.asarray(white_point, dtype=float).copy()
        payload["meanLAB"] = np.mean(lab, axis=0).reshape(-1)
        return payload, None
    if key == "cieluv":
        roi = _roi_required("ipPlot", p_type, roi_locs)
        _, xyz = _ip_plot_color_data(ip, roi)
        white_point = _ip_plot_white_point(ip, xyz)
        luv = np.asarray(xyz_to_luv(xyz, white_point), dtype=float)
        payload = _roi_payload(roi)
        payload["LUV"] = luv.copy()
        payload["whitePoint"] = np.asarray(white_point, dtype=float).copy()
        payload["meanLUV"] = np.mean(luv, axis=0).reshape(-1)
        return payload, None
    raise UnsupportedOptionError("ipPlot", p_type)


def _display_line_payload(ip: ImageProcessor, orientation: str, xy: Any) -> dict[str, Any]:
    line_index, xy_array = _line_index("plotDisplayLine", orientation, xy, orientation)
    result = ip_get(ip, "result")
    if result is None:
        raise ValueError("Results not computed in display window.")

    values = np.asarray(result, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Display line plotting requires RGB result data.")

    quantization = ip.data.get("quantization")
    method = "analog"
    n_bits = 8
    if isinstance(quantization, dict):
        method = param_format(quantization.get("method", "analog"))
        n_bits = int(quantization.get("bits", n_bits))
    elif quantization is not None:
        method = param_format(quantization)
    if method != "analog":
        values = values * float(2 ** int(n_bits))
        data_type = "digital"
    else:
        data_type = "analog"

    if orientation == "h":
        if line_index < 1 or line_index > values.shape[0]:
            raise IndexError("Horizontal display line index is out of range.")
        line_values = np.asarray(values[line_index - 1, :, :], dtype=float)
    else:
        if line_index < 1 or line_index > values.shape[1]:
            raise IndexError("Vertical display line index is out of range.")
        line_values = np.asarray(values[:, line_index - 1, :], dtype=float)

    return {
        "xy": xy_array.copy(),
        "ori": orientation,
        "pos": np.arange(1, line_values.shape[0] + 1, dtype=float),
        "values": line_values.copy(),
        "dataType": data_type,
    }


def plot_display_spd(ip: ImageProcessor) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotDisplaySPD` user-data without opening a figure."""

    display = ip_get(ip, "display")
    wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    spd = np.asarray(display_get(display, "spd"), dtype=float)
    return {"wave": wave.copy(), "spd": spd.copy()}, None


def plot_display_line(
    ip: ImageProcessor,
    ori: str = "h",
    xy: Any | None = None,
) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotDisplayLine` user-data without opening a figure."""

    orientation_key = param_format(ori)
    if orientation_key in {"h", "horizontal"}:
        orientation = "h"
    elif orientation_key in {"v", "vertical"}:
        orientation = "v"
    else:
        raise UnsupportedOptionError("plotDisplayLine", ori)
    selector = _roi_required("plotDisplayLine", ori, xy)
    return _display_line_payload(ip, orientation, selector), None


def plot_display_color(
    ip: ImageProcessor,
    data_type: str = "rgb histogram",
    roi_locs: Any | None = None,
) -> tuple[Any, None]:
    """Return MATLAB-style `plotDisplayColor` user-data without opening a figure."""

    key = param_format(data_type)
    roi = _roi_required("plotDisplayColor", data_type, roi_locs)
    if key in {"rgb", "rgbhistogram"}:
        return ip_plot(ip, "rgb histogram", roi)
    if key == "rgb3d":
        return ip_plot(ip, "rgb3d", roi)
    if key in {"xy", "chromaticity"}:
        payload, _ = ip_plot(ip, "chromaticity", roi)
        payload["xy"] = np.column_stack((np.asarray(payload["x"], dtype=float), np.asarray(payload["y"], dtype=float)))
        return payload, None
    if key == "luminance":
        return ip_plot(ip, "luminance", roi)
    if key == "cielab":
        payload, _ = ip_plot(ip, "cielab", roi)
        return np.asarray(payload["LAB"], dtype=float).copy(), None
    if key == "cieluv":
        payload, _ = ip_plot(ip, "cieluv", roi)
        return np.asarray(payload["LUV"], dtype=float).copy(), None
    raise UnsupportedOptionError("plotDisplayColor", data_type)


def plot_display_gamut(ip: ImageProcessor) -> tuple[dict[str, Any], None]:
    """Return MATLAB-style `plotDisplayGamut` user-data without opening a figure."""

    display = ip_get(ip, "display")
    rgb2xyz = np.asarray(display_get(display, "rgb2xyz"), dtype=float)
    xy = np.asarray(chromaticity_xy(rgb2xyz), dtype=float)
    return {
        "xy": xy.copy(),
        "peakLuminance": float(display_get(display, "max luminance")),
    }, None


plotScene = scene_plot
scenePlot = scene_plot
plotOI = oi_plot
oiPlot = oi_plot
plotSensor = sensor_plot
sensorPlot = sensor_plot
plotSensorFFT = sensor_plot_fft
plotSensorHist = sensor_plot_hist
sensorPlotHist = sensor_plot_hist
sensorPlotLine = sensor_plot_line
ipPlot = ip_plot
identityLine = identity_line
fisePlotDefaults = fise_plot_defaults
ieFigureFormat = ie_figure_format
ieFigureResize = ie_figure_resize
ieFormatFigure = ie_figure_format
iePlaneFromVectors = ie_plane_from_vectors
ieHistImage = ie_hist_image
iePlot = ie_plot
iePlotJitter = ie_plot_jitter
iePlotSet = ie_plot_set
iePlotShadeBackground = ie_plot_shade_background
ieShape = ie_shape
plotContrastHistogram = plot_contrast_histogram
plotEtendueRatio = plot_etendue_ratio
plotGaussianSpectrum = plot_gaussian_spectrum
plotDisplaySPD = plot_display_spd
plotDisplayLine = plot_display_line
plotDisplayColor = plot_display_color
plotDisplayGamut = plot_display_gamut
plotML = plot_ml
plotMetrics = plot_metrics
plotNormal = plot_normal
plotPixelSNR = plot_pixel_snr
plotRadiance = plot_radiance
plotReflectance = plot_reflectance
plotSensorEtendue = plot_sensor_etendue
plotSensorSNR = plot_sensor_snr
plotSetUpWindow = plot_set_up_window
plotSpectrumLocus = plot_spectrum_locus
plotTextString = plot_text_string
sensorPlotColor = sensor_plot_color
wvfPlot = wvf_plot
xaxisLine = xaxis_line
yaxisLine = yaxis_line
