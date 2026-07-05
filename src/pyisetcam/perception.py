"""High-level human-perception wrappers for image-quality evidence.

This module intentionally reuses the lower-level ``metrics`` and ``scielab``
implementations.  It provides viewing-condition aware summaries instead of a
new image-processing algorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .metrics import comparison_metrics, delta_e_ab, ie_sqri, iso_acutance
from .scielab import scielab
from .utils import srgb_to_linear, srgb_to_xyz


ColorSpace = Literal["srgb", "xyz"]
VisibleDifferenceMethod = Literal["scielab", "delta_e"]


@dataclass(frozen=True)
class PerceptionViewingConfig:
    """Viewing assumptions required to interpret perceptual metrics."""

    viewing_distance_m: float = 0.50
    pixel_pitch_m: float = 0.00025
    display_luminance_cd_m2: float = 100.0
    display_black_luminance_cd_m2: float = 0.10
    horizontal_fov_deg: float | None = None
    reference_white_xyz: tuple[float, float, float] = (0.95047, 1.0, 1.08883)
    jnd_delta_e: float = 1.0
    visible_delta_e: float = 2.3


@dataclass(frozen=True)
class PerceptionMetricResult:
    """Scalar perceptual metric plus small supporting metadata."""

    name: str
    value: float
    units: str = ""
    lower_is_better: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "units": self.units,
            "lower_is_better": bool(self.lower_is_better),
            "details": _json_ready(self.details),
        }


def perception_config(**overrides: Any) -> PerceptionViewingConfig:
    """Create a validated viewing configuration for perception helpers."""

    config = PerceptionViewingConfig(**overrides)
    if config.viewing_distance_m <= 0.0:
        raise ValueError("viewing_distance_m must be positive.")
    if config.pixel_pitch_m <= 0.0:
        raise ValueError("pixel_pitch_m must be positive.")
    if config.display_luminance_cd_m2 <= 0.0:
        raise ValueError("display_luminance_cd_m2 must be positive.")
    if config.display_black_luminance_cd_m2 < 0.0:
        raise ValueError("display_black_luminance_cd_m2 must be nonnegative.")
    if config.display_black_luminance_cd_m2 >= config.display_luminance_cd_m2:
        raise ValueError("display_black_luminance_cd_m2 must be below display_luminance_cd_m2.")
    if config.horizontal_fov_deg is not None and config.horizontal_fov_deg <= 0.0:
        raise ValueError("horizontal_fov_deg must be positive when provided.")
    if config.jnd_delta_e <= 0.0:
        raise ValueError("jnd_delta_e must be positive.")
    if config.visible_delta_e <= 0.0:
        raise ValueError("visible_delta_e must be positive.")
    if len(config.reference_white_xyz) != 3:
        raise ValueError("reference_white_xyz must contain three XYZ values.")
    return config


def pixels_per_degree(
    config: PerceptionViewingConfig | None = None,
    *,
    image_width_px: int | None = None,
    horizontal_fov_deg: float | None = None,
) -> float:
    """Return display pixels per visual degree for the configured setup."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    fov = cfg.horizontal_fov_deg if horizontal_fov_deg is None else float(horizontal_fov_deg)
    if fov is not None:
        if image_width_px is None:
            raise ValueError("image_width_px is required when horizontal_fov_deg is used.")
        if int(image_width_px) <= 0:
            raise ValueError("image_width_px must be positive.")
        return float(image_width_px) / float(fov)

    degrees_per_pixel = np.degrees(2.0 * np.arctan2(cfg.pixel_pitch_m / 2.0, cfg.viewing_distance_m))
    return float(1.0 / max(float(degrees_per_pixel), 1.0e-12))


def image_to_luminance(image: Any, config: PerceptionViewingConfig | None = None) -> NDArray[np.float64]:
    """Convert grayscale or sRGB image values to display luminance cd/m^2."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    array = _normalized_image(image, name="image")
    if array.ndim >= 3 and array.shape[-1] == 3:
        linear = srgb_to_linear(array)
        relative = np.tensordot(linear, np.array([0.2126, 0.7152, 0.0722], dtype=float), axes=([-1], [0]))
    else:
        relative = np.asarray(array, dtype=float)
    relative = np.clip(relative, 0.0, 1.0)
    scale = cfg.display_luminance_cd_m2 - cfg.display_black_luminance_cd_m2
    return np.asarray(cfg.display_black_luminance_cd_m2 + (relative * scale), dtype=float)


def perception_image_metrics(
    reference: Any,
    test: Any,
    config: PerceptionViewingConfig | None = None,
    *,
    data_range: float | None = None,
) -> dict[str, float]:
    """Return basic numeric and luminance-domain comparison metrics."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    reference_array, test_array = _paired_normalized_images(reference, test)
    if data_range is None:
        data_range = _data_range(reference_array, test_array)
    metrics = dict(comparison_metrics(reference_array, test_array, data_range=data_range))

    reference_luma = image_to_luminance(reference_array, cfg)
    test_luma = image_to_luminance(test_array, cfg)
    luma_error = test_luma - reference_luma
    metrics.update(
        {
            "reference_mean_luminance_cd_m2": float(np.mean(reference_luma, dtype=float)),
            "test_mean_luminance_cd_m2": float(np.mean(test_luma, dtype=float)),
            "mean_luminance_error_cd_m2": float(np.mean(luma_error, dtype=float)),
            "rms_luminance_error_cd_m2": float(np.sqrt(np.mean(np.square(luma_error), dtype=float))),
            "max_luminance_error_cd_m2": float(np.max(np.abs(luma_error))),
        }
    )
    return metrics


def perception_color_metrics(
    reference: Any,
    test: Any,
    config: PerceptionViewingConfig | None = None,
    *,
    color_space: ColorSpace = "srgb",
    white_point: Any | None = None,
    delta_e_version: str = "2000",
) -> dict[str, Any]:
    """Compute Delta E map and compact color-perception summary."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    reference_xyz = _as_xyz_image(reference, color_space=color_space)
    test_xyz = _as_xyz_image(test, color_space=color_space)
    if reference_xyz.shape != test_xyz.shape:
        raise ValueError("reference and test must have the same shape.")

    white = np.asarray(cfg.reference_white_xyz if white_point is None else white_point, dtype=float).reshape(3)
    delta_e = np.asarray(delta_e_ab(reference_xyz, test_xyz, white, delta_e_version), dtype=float)
    summary = _perceptual_error_summary(delta_e, cfg)
    return {
        "delta_e": delta_e,
        "summary": summary,
        "white_point": white.copy(),
        "delta_e_version": str(delta_e_version),
        "color_space": color_space,
    }


def perception_visible_difference_map(
    reference: Any,
    test: Any,
    config: PerceptionViewingConfig | None = None,
    *,
    color_space: ColorSpace = "srgb",
    method: VisibleDifferenceMethod = "scielab",
    white_point: Any | None = None,
) -> dict[str, Any]:
    """Return a perceptual difference map in approximate JND units."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    reference_xyz = _as_xyz_image(reference, color_space=color_space)
    test_xyz = _as_xyz_image(test, color_space=color_space)
    if reference_xyz.shape != test_xyz.shape:
        raise ValueError("reference and test must have the same shape.")

    white = np.asarray(cfg.reference_white_xyz if white_point is None else white_point, dtype=float).reshape(3)
    if method == "scielab":
        ppd = pixels_per_degree(cfg, image_width_px=reference_xyz.shape[1]) if cfg.horizontal_fov_deg else pixels_per_degree(cfg)
        params = {
            "deltaEversion": "2000",
            "sampPerDeg": float(ppd),
            "imageFormat": "xyz",
            "filterSize": max(1.0, float(ppd)),
            "filters": [],
            "filterversion": "distribution",
        }
        delta_e, params_out, _, _ = scielab(reference_xyz, test_xyz, white, params)
        method_details: dict[str, Any] = {
            "method": method,
            "pixels_per_degree": float(ppd),
            "filterversion": str(params_out.get("filterversion", "distribution")),
        }
    elif method == "delta_e":
        delta_e = np.asarray(delta_e_ab(reference_xyz, test_xyz, white, "2000"), dtype=float)
        method_details = {"method": method}
    else:
        raise ValueError(f"Unsupported visible-difference method {method!r}.")

    delta_e = np.asarray(delta_e, dtype=float)
    jnd_map = delta_e / cfg.jnd_delta_e
    return {
        "delta_e": delta_e,
        "jnd_map": jnd_map,
        "summary": _perceptual_error_summary(delta_e, cfg),
        "white_point": white.copy(),
        "details": method_details,
    }


def perception_sharpness_metrics(
    frequency_cpd: Any,
    luminance_mtf: Any,
    config: PerceptionViewingConfig | None = None,
) -> dict[str, Any]:
    """Compute perception-weighted sharpness metrics from MTF samples."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    frequency = np.asarray(frequency_cpd, dtype=float).reshape(-1)
    mtf = np.asarray(luminance_mtf, dtype=float).reshape(-1)
    if frequency.size != mtf.size:
        raise ValueError("frequency_cpd and luminance_mtf must have the same length.")
    if frequency.size < 2:
        raise ValueError("frequency_cpd must contain at least two samples.")
    if np.any(frequency < 0.0):
        raise ValueError("frequency_cpd must be nonnegative.")

    clipped_mtf = np.clip(mtf, 0.0, 1.0)
    acutance = float(iso_acutance(frequency, clipped_mtf))
    sqri, csf = ie_sqri(frequency, clipped_mtf, cfg.display_luminance_cd_m2)
    high_frequency_mask = frequency >= (0.5 * float(np.max(frequency)))
    high_frequency_mean = float(np.mean(clipped_mtf[high_frequency_mask], dtype=float)) if np.any(high_frequency_mask) else 0.0
    return {
        "iso_acutance": acutance,
        "sqri": float(sqri),
        "high_frequency_mtf_mean": high_frequency_mean,
        "frequency_cpd": frequency.copy(),
        "luminance_mtf": clipped_mtf.copy(),
        "csf": np.asarray(csf, dtype=float).copy(),
    }


def perception_artifact_metrics(
    reference: Any,
    test: Any,
    config: PerceptionViewingConfig | None = None,
) -> dict[str, float]:
    """Return simple artifact visibility proxies for a reference/test pair."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    reference_array, test_array = _paired_normalized_images(reference, test)
    reference_luma = image_to_luminance(reference_array, cfg)
    test_luma = image_to_luminance(test_array, cfg)
    error = test_luma - reference_luma

    gy, gx = np.gradient(error)
    gradient_energy = np.sqrt(np.square(gx) + np.square(gy))
    high_freq_fraction = _high_frequency_energy_fraction(error)

    chroma_error = 0.0
    if reference_array.ndim >= 3 and reference_array.shape[-1] == 3:
        reference_chroma = reference_array - np.mean(reference_array, axis=-1, keepdims=True)
        test_chroma = test_array - np.mean(test_array, axis=-1, keepdims=True)
        chroma_error = float(np.mean(np.abs(test_chroma - reference_chroma), dtype=float))

    return {
        "luma_error_mae_cd_m2": float(np.mean(np.abs(error), dtype=float)),
        "luma_error_rmse_cd_m2": float(np.sqrt(np.mean(np.square(error), dtype=float))),
        "edge_artifact_energy": float(np.mean(gradient_energy, dtype=float)),
        "high_frequency_error_fraction": float(high_freq_fraction),
        "chroma_error_mae": chroma_error,
    }


def perception_compare(
    reference: Any,
    test: Any,
    config: PerceptionViewingConfig | None = None,
    *,
    color_space: ColorSpace = "srgb",
    visible_method: VisibleDifferenceMethod = "scielab",
    include_maps: bool = False,
) -> dict[str, Any]:
    """Run the default perception comparison bundle."""

    cfg = perception_config() if config is None else perception_config(**asdict(config))
    image_metrics = perception_image_metrics(reference, test, cfg)
    color_metrics = perception_color_metrics(reference, test, cfg, color_space=color_space)
    visible_difference = perception_visible_difference_map(reference, test, cfg, color_space=color_space, method=visible_method)
    artifact_metrics = perception_artifact_metrics(reference, test, cfg)

    payload: dict[str, Any] = {
        "config": asdict(cfg),
        "image_metrics": image_metrics,
        "color_metrics": color_metrics["summary"],
        "visible_difference": visible_difference["summary"],
        "artifact_metrics": artifact_metrics,
    }
    if include_maps:
        payload["maps"] = {
            "delta_e": visible_difference["delta_e"],
            "jnd": visible_difference["jnd_map"],
        }
    return payload


def perception_report(
    reference: Any,
    test: Any,
    config: PerceptionViewingConfig | None = None,
    *,
    color_space: ColorSpace = "srgb",
    include_maps: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper for the default perception comparison bundle."""

    return perception_compare(reference, test, config, color_space=color_space, include_maps=include_maps)


def _normalized_image(values: Any, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if array.ndim < 2:
        raise ValueError(f"{name} must be at least two-dimensional.")
    if array.ndim >= 3 and array.shape[-1] not in {1, 3}:
        raise ValueError(f"{name} must be grayscale or have a trailing RGB dimension.")
    normalized = array.copy()
    max_value = float(np.max(normalized))
    if max_value > 1.0:
        normalized = normalized / (255.0 if max_value <= 255.0 else max_value)
    return np.clip(normalized, 0.0, 1.0)


def _paired_normalized_images(reference: Any, test: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference_array = _normalized_image(reference, name="reference")
    test_array = _normalized_image(test, name="test")
    if reference_array.shape != test_array.shape:
        raise ValueError("reference and test must have the same shape.")
    return reference_array, test_array


def _as_xyz_image(values: Any, *, color_space: ColorSpace) -> NDArray[np.float64]:
    if color_space == "srgb":
        image = _normalized_image(values, name="image")
        if image.ndim < 3 or image.shape[-1] != 3:
            raise ValueError("sRGB perception metrics require an RGB image.")
        return np.asarray(srgb_to_xyz(image), dtype=float)
    if color_space == "xyz":
        xyz = np.asarray(values, dtype=float)
        if xyz.ndim < 3 or xyz.shape[-1] != 3:
            raise ValueError("XYZ perception metrics require a trailing dimension of size 3.")
        if not np.all(np.isfinite(xyz)):
            raise ValueError("XYZ image must contain only finite values.")
        if float(np.min(xyz)) < 0.0:
            raise ValueError("XYZ image must be nonnegative.")
        return np.asarray(xyz, dtype=float)
    raise ValueError(f"Unsupported color_space {color_space!r}.")


def _perceptual_error_summary(delta_e: NDArray[np.float64], config: PerceptionViewingConfig) -> dict[str, float]:
    values = np.asarray(delta_e, dtype=float)
    return {
        "mean_delta_e": float(np.mean(values, dtype=float)),
        "median_delta_e": float(np.median(values)),
        "p95_delta_e": float(np.percentile(values, 95.0)),
        "max_delta_e": float(np.max(values)),
        "mean_jnd": float(np.mean(values / config.jnd_delta_e, dtype=float)),
        "max_jnd": float(np.max(values / config.jnd_delta_e)),
        "visible_fraction": float(np.mean(values >= config.visible_delta_e, dtype=float)),
    }


def _data_range(reference: NDArray[np.float64], test: NDArray[np.float64]) -> float:
    lower = min(float(np.min(reference)), float(np.min(test)))
    upper = max(float(np.max(reference)), float(np.max(test)))
    return max(upper - lower, 1.0)


def _high_frequency_energy_fraction(image: NDArray[np.float64]) -> float:
    values = np.asarray(image, dtype=float)
    if values.ndim > 2:
        values = np.mean(values, axis=-1)
    centered = values - float(np.mean(values, dtype=float))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    energy = np.square(spectrum)
    total = float(np.sum(energy, dtype=float))
    if total <= 1.0e-18:
        return 0.0

    rows, cols = values.shape
    yy, xx = np.ogrid[:rows, :cols]
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = np.sqrt(np.square((yy - cy) / max(cy, 1.0)) + np.square((xx - cx) / max(cx, 1.0)))
    high_frequency = radius >= 0.50
    return float(np.sum(energy[high_frequency], dtype=float) / total)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


perceptionConfig = perception_config
pixelsPerDegree = pixels_per_degree
imageToLuminance = image_to_luminance
perceptionImageMetrics = perception_image_metrics
perceptionColorMetrics = perception_color_metrics
perceptionVisibleDifferenceMap = perception_visible_difference_map
perceptionSharpnessMetrics = perception_sharpness_metrics
perceptionArtifactMetrics = perception_artifact_metrics
perceptionCompare = perception_compare
perceptionReport = perception_report


__all__ = [
    "ColorSpace",
    "PerceptionMetricResult",
    "PerceptionViewingConfig",
    "VisibleDifferenceMethod",
    "imageToLuminance",
    "image_to_luminance",
    "perceptionArtifactMetrics",
    "perceptionColorMetrics",
    "perceptionCompare",
    "perceptionConfig",
    "perceptionImageMetrics",
    "perceptionReport",
    "perceptionSharpnessMetrics",
    "perceptionVisibleDifferenceMap",
    "perception_artifact_metrics",
    "perception_color_metrics",
    "perception_compare",
    "perception_config",
    "perception_image_metrics",
    "perception_report",
    "perception_sharpness_metrics",
    "perception_visible_difference_map",
    "pixelsPerDegree",
    "pixels_per_degree",
]
