from __future__ import annotations

import numpy as np
from pathlib import Path

import pyisetcam.perception as perception_module
from pyisetcam import (
    PerceptionViewingConfig,
    imageToLuminance,
    perceptionColorMetrics,
    perceptionCompare,
    perceptionConfig,
    perceptionVisibleDifferenceMap,
    pixelsPerDegree,
)
from tools import render_perception_report


def _sample_pair(size: int = 40) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(float)
    reference = np.zeros((size, size, 3), dtype=float)
    reference[..., 0] = x / max(size - 1, 1)
    reference[..., 1] = y / max(size - 1, 1)
    reference[..., 2] = 0.25

    test = reference.copy()
    test[12:28, 12:28, 0] = np.clip(test[12:28, 12:28, 0] + 0.12, 0.0, 1.0)
    test[..., 2] = np.clip(test[..., 2] * 0.92, 0.0, 1.0)
    return reference, test


def test_perception_config_and_pixels_per_degree() -> None:
    config = perceptionConfig(viewing_distance_m=0.5, pixel_pitch_m=0.00025)

    assert isinstance(config, PerceptionViewingConfig)
    assert pixelsPerDegree(config) > 0.0
    assert pixelsPerDegree(perceptionConfig(horizontal_fov_deg=20.0), image_width_px=200) == 10.0


def test_perception_module_matlab_aliases() -> None:
    assert perception_module.perceptionConfig is perception_module.perception_config
    assert perception_module.pixelsPerDegree is perception_module.pixels_per_degree
    assert perception_module.imageToLuminance is perception_module.image_to_luminance
    assert perception_module.perceptionCompare is perception_module.perception_compare


def test_perception_root_exports_aliases() -> None:
    assert perceptionConfig is perception_module.perception_config
    assert pixelsPerDegree is perception_module.pixels_per_degree
    assert imageToLuminance is perception_module.image_to_luminance
    assert perceptionColorMetrics is perception_module.perception_color_metrics
    assert perceptionVisibleDifferenceMap is perception_module.perception_visible_difference_map
    assert perceptionCompare is perception_module.perception_compare


def test_image_to_luminance_returns_display_domain_values() -> None:
    config = perceptionConfig(display_luminance_cd_m2=80.0, display_black_luminance_cd_m2=0.2)
    image = np.ones((8, 8, 3), dtype=float)

    luminance = imageToLuminance(image, config)

    assert luminance.shape == (8, 8)
    assert np.allclose(luminance, 80.0)


def test_visible_difference_is_zero_for_identical_images() -> None:
    reference, _ = _sample_pair()
    result = perceptionVisibleDifferenceMap(reference, reference, method="delta_e")

    assert result["delta_e"].shape == reference.shape[:2]
    assert result["summary"]["max_delta_e"] == 0.0
    assert result["summary"]["visible_fraction"] == 0.0


def test_perception_compare_reports_color_and_visible_differences() -> None:
    reference, test = _sample_pair()

    summary = perceptionCompare(reference, test, visible_method="delta_e")

    assert summary["image_metrics"]["mae"] > 0.0
    assert summary["color_metrics"]["mean_delta_e"] > 0.0
    assert summary["visible_difference"]["max_jnd"] > 0.0
    assert 0.0 <= summary["artifact_metrics"]["high_frequency_error_fraction"] <= 1.0


def test_perception_sharpness_metrics_are_finite() -> None:
    frequency = np.linspace(0.0, 50.0, 16)
    mtf = np.exp(-frequency / 30.0)

    metrics = perception_module.perception_sharpness_metrics(frequency, mtf)

    assert np.isfinite(metrics["iso_acutance"])
    assert np.isfinite(metrics["sqri"])
    assert metrics["frequency_cpd"].shape == frequency.shape
    assert metrics["csf"].shape == frequency.shape


def test_render_perception_report_creates_artifacts(tmp_path) -> None:
    result = render_perception_report.build_report(tmp_path)

    assert result["html"].exists()
    assert result["summary"].exists()
    for figure_path in result["figures"].values():
        assert Path(figure_path).exists()
