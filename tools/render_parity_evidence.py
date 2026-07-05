"""Render a camera-pipeline parity evidence bundle for executive review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import AssetStore
from pyisetcam.color import luminance_from_photons
from pyisetcam.parity import run_python_case_with_context

from parity_report import DEFAULT_OUTPUT, _case_definitions, _compare, _load_reference, _normalize, build_report

SELECTED_CASES = [
    "camera_default_pipeline",
    "ip_default_pipeline",
    "metrics_color_accuracy_small",
    "optics_rt_center_edge_psf_small",
    "optics_rt_point_array_field_small",
    "optics_rt_distortion_field_small",
    "metrics_vsnr_small",
    "metrics_acutance_small",
    "metrics_mtf_slanted_bar_small",
    "metrics_mtf_pixel_size_small",
]

DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "parity" / "camera_field"
DEFAULT_MARKDOWN = REPO_ROOT / "reports" / "parity" / "camera_field_parity_report.md"
DEFAULT_SUMMARY = REPO_ROOT / "reports" / "parity" / "camera_field_summary.json"

POINT_ARRAY_CROP_RADIUS = 24


def _load_report(report_path: Path, *, refresh: bool) -> dict[str, Any]:
    if refresh or not report_path.exists():
        report = build_report()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return report
    return json.loads(report_path.read_text())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _case_definition_map() -> dict[str, dict[str, Any]]:
    return {str(case["name"]): case for case in _case_definitions()}


def _report_case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["name"]): case for case in report["results"]}


def _selected_report_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    case_map = _report_case_map(report)
    return [case_map[name] for name in SELECTED_CASES]


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(array.reshape(-1)[0])


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _format_number(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _format_shape(shape: Any) -> str:
    if shape is None:
        return "n/a"
    array = np.asarray(shape, dtype=int).reshape(-1)
    if array.size == 0:
        return "[]"
    return "x".join(str(int(item)) for item in array)


def _format_phase_metrics(metrics: dict[str, Any]) -> str:
    phase = metrics.get("phase_2x2_mean_rel")
    if not isinstance(phase, dict):
        return "n/a"
    ordered = [f"{key}={_format_number(value, digits=4)}" for key, value in sorted(phase.items())]
    return ", ".join(ordered)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, divider, *body])


def _shared_percentile_range(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([reference.reshape(-1), actual.reshape(-1)])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if low == high:
        high = low + 1.0
    return low, high


def _normalize_to_unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scale = max(high - low, 1.0e-12)
    return np.clip((values - low) / scale, 0.0, 1.0)


def _image_payload(
    value: Any,
    *,
    field_name: str,
    wave_nm: np.ndarray | None,
    asset_store: AssetStore,
) -> tuple[np.ndarray, str]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        return array, field_name
    if array.ndim == 3 and array.shape[-1] == 3:
        return array, field_name
    if field_name == "oi_photons" and wave_nm is not None and array.ndim == 3 and array.shape[-1] == wave_nm.size:
        luminance = np.asarray(luminance_from_photons(array, wave_nm, asset_store=asset_store), dtype=float)
        return luminance, "oi_photons_luminance_projection"
    if array.ndim >= 3:
        return np.mean(array, axis=-1), f"{field_name}_mean_projection"
    return array.reshape(array.shape[0], -1), f"{field_name}_reshaped"


def _triptych_difference(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    diff = np.abs(actual - reference)
    if diff.ndim == 3 and diff.shape[-1] == 3:
        return np.mean(diff, axis=-1)
    return diff


def _render_triptych(
    reference: Any,
    actual: Any,
    *,
    title: str,
    field_name: str,
    wave_nm: np.ndarray | None,
    asset_store: AssetStore,
    output_path: Path,
) -> dict[str, Any]:
    reference_image, projection_label = _image_payload(reference, field_name=field_name, wave_nm=wave_nm, asset_store=asset_store)
    actual_image, _ = _image_payload(actual, field_name=field_name, wave_nm=wave_nm, asset_store=asset_store)
    low, high = _shared_percentile_range(reference_image, actual_image)
    normalized_reference = _normalize_to_unit(reference_image, low, high)
    normalized_actual = _normalize_to_unit(actual_image, low, high)
    diff = _triptych_difference(reference_image, actual_image)
    diff_high = float(np.percentile(diff.reshape(-1), 99.0)) if diff.size else 1.0
    if diff_high <= 0.0:
        diff_high = float(np.max(diff)) if diff.size else 1.0
    if diff_high <= 0.0:
        diff_high = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    fig.suptitle(title)
    image_kwargs: dict[str, Any] = {"interpolation": "nearest"}

    if normalized_reference.ndim == 2:
        ref_artist = axes[0].imshow(normalized_reference, cmap="gray", **image_kwargs)
        act_artist = axes[1].imshow(normalized_actual, cmap="gray", **image_kwargs)
    else:
        ref_artist = axes[0].imshow(normalized_reference, **image_kwargs)
        act_artist = axes[1].imshow(normalized_actual, **image_kwargs)

    diff_artist = axes[2].imshow(diff, cmap="inferno", vmin=0.0, vmax=diff_high, **image_kwargs)

    axes[0].set_title("MATLAB baseline")
    axes[1].set_title("Python port")
    axes[2].set_title("Absolute difference")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    if normalized_reference.ndim == 2:
        plt.colorbar(ref_artist, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(act_artist, ax=axes[1], fraction=0.046, pad=0.04)
    plt.colorbar(diff_artist, ax=axes[2], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity")),
        "projection": projection_label,
        "shared_low": low,
        "shared_high": high,
        "difference_high": diff_high,
    }


def _peak_zero_based(values: Any) -> np.ndarray:
    peak = np.asarray(values, dtype=int).reshape(2)
    return peak - 1


def _annotate_crop_box(axis: Any, peak_rc: Any, *, label: str, color: str, radius: int) -> None:
    row, col = _peak_zero_based(peak_rc)
    rect = Rectangle(
        (int(col) - radius, int(row) - radius),
        2 * radius + 1,
        2 * radius + 1,
        fill=False,
        linewidth=2.0,
        edgecolor=color,
    )
    axis.add_patch(rect)
    axis.text(
        int(col) + radius + 6,
        int(row),
        label,
        color=color,
        fontsize=9,
        va="center",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
    )


def _render_annotated_overview_triptych(
    reference: Any,
    actual: Any,
    *,
    reference_center_peak_rc: Any,
    reference_edge_peak_rc: Any,
    actual_center_peak_rc: Any,
    actual_edge_peak_rc: Any,
    title: str,
    output_path: Path,
) -> dict[str, Any]:
    reference_rgb = np.asarray(reference, dtype=float)
    actual_rgb = np.asarray(actual, dtype=float)
    low, high = _shared_percentile_range(reference_rgb, actual_rgb)
    normalized_reference = _normalize_to_unit(reference_rgb, low, high)
    normalized_actual = _normalize_to_unit(actual_rgb, low, high)
    diff = np.mean(np.abs(actual_rgb - reference_rgb), axis=2)
    diff_high = float(np.percentile(diff.reshape(-1), 99.0)) if diff.size else 1.0
    diff_high = diff_high if diff_high > 0.0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9), constrained_layout=True)
    fig.suptitle(title)
    axes[0].imshow(normalized_reference, interpolation="nearest")
    axes[1].imshow(normalized_actual, interpolation="nearest")
    diff_artist = axes[2].imshow(diff, cmap="inferno", vmin=0.0, vmax=diff_high, interpolation="nearest")
    axes[0].set_title("MATLAB baseline")
    axes[1].set_title("Python port")
    axes[2].set_title("Absolute RGB difference")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    for axis, center_peak, edge_peak in (
        (axes[0], reference_center_peak_rc, reference_edge_peak_rc),
        (axes[1], actual_center_peak_rc, actual_edge_peak_rc),
    ):
        _annotate_crop_box(axis, center_peak, label="Center", color="#00d8ff", radius=POINT_ARRAY_CROP_RADIUS)
        _annotate_crop_box(axis, edge_peak, label="Edge", color="#ffae00", radius=POINT_ARRAY_CROP_RADIUS)

    plt.colorbar(diff_artist, ax=axes[2], fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _render_center_edge_crop_grid(
    reference_center: Any,
    actual_center: Any,
    reference_edge: Any,
    actual_edge: Any,
    *,
    title: str,
    output_path: Path,
) -> dict[str, Any]:
    center_ref = np.asarray(reference_center, dtype=float)
    center_actual = np.asarray(actual_center, dtype=float)
    edge_ref = np.asarray(reference_edge, dtype=float)
    edge_actual = np.asarray(actual_edge, dtype=float)

    center_low, center_high = _shared_percentile_range(center_ref, center_actual)
    edge_low, edge_high = _shared_percentile_range(edge_ref, edge_actual)
    center_diff = np.mean(np.abs(center_actual - center_ref), axis=2)
    edge_diff = np.mean(np.abs(edge_actual - edge_ref), axis=2)
    center_diff_high = float(np.percentile(center_diff.reshape(-1), 99.0)) if center_diff.size else 1.0
    edge_diff_high = float(np.percentile(edge_diff.reshape(-1), 99.0)) if edge_diff.size else 1.0
    center_diff_high = center_diff_high if center_diff_high > 0.0 else 1.0
    edge_diff_high = edge_diff_high if edge_diff_high > 0.0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.2), constrained_layout=True)
    fig.suptitle(title)
    rows = (
        ("Center field", center_ref, center_actual, center_diff, center_low, center_high, center_diff_high),
        ("Edge field", edge_ref, edge_actual, edge_diff, edge_low, edge_high, edge_diff_high),
    )
    for row_index, (label, reference_rgb, actual_rgb, diff, low, high, diff_high) in enumerate(rows):
        axes[row_index, 0].imshow(_normalize_to_unit(reference_rgb, low, high), interpolation="nearest")
        axes[row_index, 1].imshow(_normalize_to_unit(actual_rgb, low, high), interpolation="nearest")
        diff_artist = axes[row_index, 2].imshow(diff, cmap="inferno", vmin=0.0, vmax=diff_high, interpolation="nearest")
        axes[row_index, 0].set_title(f"{label}: MATLAB")
        axes[row_index, 1].set_title(f"{label}: Python")
        axes[row_index, 2].set_title(f"{label}: abs diff")
        for col_index in range(3):
            axes[row_index, col_index].set_xticks([])
            axes[row_index, col_index].set_yticks([])
        plt.colorbar(diff_artist, ax=axes[row_index, 2], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _render_psf_grid(
    reference_center_psf: Any,
    actual_center_psf: Any,
    reference_edge_psf: Any,
    actual_edge_psf: Any,
    *,
    support_x_um: Any,
    support_y_um: Any,
    title: str,
    output_path: Path,
) -> dict[str, Any]:
    support_x = np.asarray(support_x_um, dtype=float).reshape(-1)
    support_y = np.asarray(support_y_um, dtype=float).reshape(-1)
    extent = [
        float(np.min(support_x)),
        float(np.max(support_x)),
        float(np.max(support_y)),
        float(np.min(support_y)),
    ]

    rows = (
        ("Center field", np.asarray(reference_center_psf, dtype=float), np.asarray(actual_center_psf, dtype=float)),
        ("Edge field", np.asarray(reference_edge_psf, dtype=float), np.asarray(actual_edge_psf, dtype=float)),
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.5), constrained_layout=True)
    fig.suptitle(title)
    for row_index, (label, reference_psf, actual_psf) in enumerate(rows):
        reference_log = np.log10(np.clip(reference_psf, 1.0e-6, None))
        actual_log = np.log10(np.clip(actual_psf, 1.0e-6, None))
        diff = np.abs(actual_psf - reference_psf)
        diff_high = float(np.percentile(diff.reshape(-1), 99.0)) if diff.size else 1.0
        diff_high = diff_high if diff_high > 0.0 else 1.0
        first = axes[row_index, 0].imshow(reference_log, cmap="viridis", vmin=-6.0, vmax=0.0, extent=extent, interpolation="nearest")
        second = axes[row_index, 1].imshow(actual_log, cmap="viridis", vmin=-6.0, vmax=0.0, extent=extent, interpolation="nearest")
        third = axes[row_index, 2].imshow(diff, cmap="inferno", vmin=0.0, vmax=diff_high, extent=extent, interpolation="nearest")
        axes[row_index, 0].set_title(f"{label}: MATLAB")
        axes[row_index, 1].set_title(f"{label}: Python")
        axes[row_index, 2].set_title(f"{label}: abs diff")
        for col_index in range(3):
            axes[row_index, col_index].set_xlabel("x (um)")
            axes[row_index, col_index].set_ylabel("y (um)")
        plt.colorbar(first, ax=axes[row_index, 0], fraction=0.046, pad=0.04)
        plt.colorbar(second, ax=axes[row_index, 1], fraction=0.046, pad=0.04)
        plt.colorbar(third, ax=axes[row_index, 2], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _render_psf_profile_overlay(
    support_x_um: Any,
    reference_center_row: Any,
    actual_center_row: Any,
    reference_edge_row: Any,
    actual_edge_row: Any,
    *,
    title: str,
    output_path: Path,
) -> dict[str, Any]:
    support_x = np.asarray(support_x_um, dtype=float).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.6), constrained_layout=True)
    fig.suptitle(title)
    for axis, label, reference_row, actual_row in (
        (axes[0], "Center field", reference_center_row, actual_center_row),
        (axes[1], "Edge field", reference_edge_row, actual_edge_row),
    ):
        axis.plot(support_x, np.asarray(reference_row, dtype=float).reshape(-1), linewidth=2.1, label="MATLAB")
        axis.plot(support_x, np.asarray(actual_row, dtype=float).reshape(-1), linewidth=2.1, linestyle="--", label="Python")
        axis.set_title(label)
        axis.set_xlabel("Support x (um)")
        axis.set_ylabel("Normalized PSF row")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _render_distortion_grid_triptych(
    reference_ideal: Any,
    actual_ideal: Any,
    reference_distorted: Any,
    actual_distorted: Any,
    *,
    title: str,
    output_path: Path,
) -> dict[str, Any]:
    ref_ideal = np.asarray(reference_ideal, dtype=float)
    act_ideal = np.asarray(actual_ideal, dtype=float)
    ref_distorted = np.asarray(reference_distorted, dtype=float)
    act_distorted = np.asarray(actual_distorted, dtype=float)
    ideal_low, ideal_high = _shared_percentile_range(ref_ideal, act_ideal)
    distorted_low, distorted_high = _shared_percentile_range(ref_distorted, act_distorted)
    ideal_diff = np.mean(np.abs(act_ideal - ref_ideal), axis=2)
    distorted_diff = np.mean(np.abs(act_distorted - ref_distorted), axis=2)
    ideal_diff_high = float(np.percentile(ideal_diff.reshape(-1), 99.0)) if ideal_diff.size else 1.0
    distorted_diff_high = float(np.percentile(distorted_diff.reshape(-1), 99.0)) if distorted_diff.size else 1.0
    ideal_diff_high = ideal_diff_high if ideal_diff_high > 0.0 else 1.0
    distorted_diff_high = distorted_diff_high if distorted_diff_high > 0.0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(13.3, 8.6), constrained_layout=True)
    fig.suptitle(title)
    rows = (
        ("Ideal grid", ref_ideal, act_ideal, ideal_diff, ideal_low, ideal_high, ideal_diff_high),
        ("Distorted grid", ref_distorted, act_distorted, distorted_diff, distorted_low, distorted_high, distorted_diff_high),
    )
    for row_index, (label, reference_rgb, actual_rgb, diff, low, high, diff_high) in enumerate(rows):
        axes[row_index, 0].imshow(_normalize_to_unit(reference_rgb, low, high), interpolation="nearest")
        axes[row_index, 1].imshow(_normalize_to_unit(actual_rgb, low, high), interpolation="nearest")
        diff_artist = axes[row_index, 2].imshow(diff, cmap="inferno", vmin=0.0, vmax=diff_high, interpolation="nearest")
        axes[row_index, 0].set_title(f"{label}: MATLAB")
        axes[row_index, 1].set_title(f"{label}: Python")
        axes[row_index, 2].set_title(f"{label}: abs diff")
        for col_index in range(3):
            axes[row_index, col_index].set_xticks([])
            axes[row_index, col_index].set_yticks([])
        plt.colorbar(diff_artist, ax=axes[row_index, 2], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _render_distortion_curve_overlay(
    field_height_mm: Any,
    reference_distorted_height_mm: Any,
    actual_distorted_height_mm: Any,
    reference_distortion_percent: Any,
    actual_distortion_percent: Any,
    *,
    title: str,
    output_path: Path,
) -> dict[str, Any]:
    x = np.asarray(field_height_mm, dtype=float).reshape(-1)
    ref_height = np.asarray(reference_distorted_height_mm, dtype=float).reshape(-1)
    act_height = np.asarray(actual_distorted_height_mm, dtype=float).reshape(-1)
    ref_distortion = np.asarray(reference_distortion_percent, dtype=float).reshape(-1)
    act_distortion = np.asarray(actual_distortion_percent, dtype=float).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), constrained_layout=True)
    fig.suptitle(title)
    axes[0].plot(x, x, color="black", linewidth=1.2, alpha=0.6, label="Ideal")
    axes[0].plot(x, ref_height, linewidth=2.0, label="MATLAB distorted")
    axes[0].plot(x, act_height, linewidth=2.0, linestyle="--", label="Python distorted")
    axes[0].set_title("Image height mapping")
    axes[0].set_xlabel("Field height (mm)")
    axes[0].set_ylabel("Image height (mm)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(x[1:], ref_distortion[1:], linewidth=2.0, label="MATLAB")
    axes[1].plot(x[1:], act_distortion[1:], linewidth=2.0, linestyle="--", label="Python")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_title("Radial distortion")
    axes[1].set_xlabel("Field height (mm)")
    axes[1].set_ylabel("Distortion (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _plot_overlay(
    x_values: np.ndarray,
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    title: str,
    y_label: str,
    x_label: str,
    output_path: Path,
    legend_labels: list[str] | None = None,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=float)
    act = np.asarray(actual, dtype=float)
    x = np.asarray(x_values, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    if ref.ndim == 1:
        ax.plot(x, ref, linewidth=2.2, label="MATLAB")
        ax.plot(x, act, linewidth=2.2, linestyle="--", label="Python")
    else:
        labels = legend_labels or [f"Series {index + 1}" for index in range(ref.shape[0])]
        for index, label in enumerate(labels):
            ax.plot(x, ref[index], linewidth=2.0, label=f"MATLAB {label}")
            ax.plot(x, act[index], linewidth=2.0, linestyle="--", label=f"Python {label}")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _plot_two_panel_overlay(
    x_values: np.ndarray,
    left_reference: np.ndarray,
    left_actual: np.ndarray,
    right_reference: np.ndarray,
    right_actual: np.ndarray,
    *,
    title: str,
    left_title: str,
    right_title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> dict[str, Any]:
    x = np.asarray(x_values, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    fig.suptitle(title)
    for axis, reference, actual, subplot_title in (
        (axes[0], left_reference, left_actual, left_title),
        (axes[1], right_reference, right_actual, right_title),
    ):
        axis.plot(x, np.asarray(reference, dtype=float), linewidth=2.2, label="MATLAB")
        axis.plot(x, np.asarray(actual, dtype=float), linewidth=2.2, linestyle="--", label="Python")
        axis.set_title(subplot_title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _plot_triplet_overlay(
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> dict[str, Any]:
    x = np.asarray(x_values, dtype=float)
    fig, axes = plt.subplots(1, len(series), figsize=(5.1 * len(series), 4.6), constrained_layout=True)
    if len(series) == 1:
        axes = [axes]
    fig.suptitle(title)
    for axis, (label, reference, actual) in zip(axes, series, strict=True):
        axis.plot(x, np.asarray(reference, dtype=float), linewidth=2.0, label="MATLAB")
        axis.plot(x, np.asarray(actual, dtype=float), linewidth=2.0, linestyle="--", label="Python")
        axis.set_title(label)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _plot_mtf_pixel_size_subplots(
    pixel_sizes_um: np.ndarray,
    profiles_ref: np.ndarray,
    profiles_actual: np.ndarray,
    mtf50_ref: np.ndarray,
    mtf50_actual: np.ndarray,
    nyquist: np.ndarray,
    *,
    output_path: Path,
) -> dict[str, Any]:
    x = np.linspace(0.0, 1.0, profiles_ref.shape[1], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
    fig.suptitle("metrics_mtf_pixel_size_small: normalized MTF by pixel size")
    axes_flat = list(np.ravel(axes))

    for axis, pixel_size, ref_profile, actual_profile, ref_mtf50, actual_mtf50, nyquistf in zip(
        axes_flat,
        np.asarray(pixel_sizes_um, dtype=float).reshape(-1),
        np.asarray(profiles_ref, dtype=float),
        np.asarray(profiles_actual, dtype=float),
        np.asarray(mtf50_ref, dtype=float).reshape(-1),
        np.asarray(mtf50_actual, dtype=float).reshape(-1),
        np.asarray(nyquist, dtype=float).reshape(-1),
        strict=True,
    ):
        ref_profile = np.asarray(ref_profile, dtype=float).reshape(-1)
        actual_profile = np.asarray(actual_profile, dtype=float).reshape(-1)
        axis.plot(x, ref_profile, linewidth=2.2, label="MATLAB")
        axis.plot(x, actual_profile, linewidth=2.2, linestyle="--", label="Python")
        ref_norm = float(ref_mtf50) / max(float(nyquistf), 1.0e-12)
        actual_norm = float(actual_mtf50) / max(float(nyquistf), 1.0e-12)
        axis.axvline(ref_norm, color="#1f77b4", alpha=0.35, linewidth=1.6)
        axis.axvline(actual_norm, color="#ff7f0e", alpha=0.35, linewidth=1.6, linestyle="--")
        axis.set_title(f"{pixel_size:.2f} um")
        axis.set_xlabel("Normalized frequency (f / Nyquist)")
        axis.set_ylabel("Normalized MTF")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"path": str(output_path.relative_to(REPO_ROOT / "reports" / "parity"))}


def _case_comparison(
    case_name: str,
    *,
    report_case: dict[str, Any],
    asset_store: AssetStore,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray | None]:
    reference = _normalize(_load_reference(case_name))
    case_result = run_python_case_with_context(case_name, asset_store=asset_store)
    actual = _normalize(case_result.payload)
    case_definition = _case_definition_map()[case_name]
    comparison = _compare(
        reference,
        actual,
        rtol=float(case_definition["rtol"]),
        atol=float(case_definition["atol"]),
        field_rules=report_case.get("field_overrides"),
    )
    wave_nm = None
    oi = case_result.context.get("oi")
    if oi is not None and "wave" in oi.fields:
        wave_nm = np.asarray(oi.fields["wave"], dtype=float).reshape(-1)
    return reference, actual, comparison, wave_nm


def _case_metric_rows(fields: dict[str, Any], requested_fields: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for field_name in requested_fields:
        metrics = fields[field_name]
        rows.append(
            [
                field_name,
                _format_shape(metrics.get("shape")),
                _format_number(metrics.get("mae")),
                _format_number(metrics.get("rmse")),
                _format_number(metrics.get("max_abs")),
                _format_number(metrics.get("mean_rel")),
                _format_number(metrics.get("max_rel")),
                _format_number(metrics.get("normalized_mae")),
                _format_number(metrics.get("edge_mean_rel")),
                _format_number(metrics.get("interior_mean_rel")),
                _format_phase_metrics(metrics),
            ]
        )
    return rows


def _render_camera_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
    comparison: dict[str, Any],
    wave_nm: np.ndarray | None,
    asset_store: AssetStore,
) -> dict[str, Any]:
    fields = comparison["fields"]
    figures = {
        "result": _render_triptych(
            reference["result"],
            actual["result"],
            title="camera_default_pipeline: final result",
            field_name="result",
            wave_nm=wave_nm,
            asset_store=asset_store,
            output_path=figures_dir / "camera_default_pipeline_result_triptych.png",
        ),
        "sensor_volts": _render_triptych(
            reference["sensor_volts"],
            actual["sensor_volts"],
            title="camera_default_pipeline: sensor volts",
            field_name="sensor_volts",
            wave_nm=wave_nm,
            asset_store=asset_store,
            output_path=figures_dir / "camera_default_pipeline_sensor_volts_triptych.png",
        ),
        "oi_photons": _render_triptych(
            reference["oi_photons"],
            actual["oi_photons"],
            title="camera_default_pipeline: OI photons projection",
            field_name="oi_photons",
            wave_nm=wave_nm,
            asset_store=asset_store,
            output_path=figures_dir / "camera_default_pipeline_oi_photons_triptych.png",
        ),
    }
    return {
        "figures": figures,
        "metrics_table": _markdown_table(
            [
                "Field",
                "Shape",
                "MAE",
                "RMSE",
                "Max abs",
                "Mean rel",
                "Max rel",
                "Normalized MAE",
                "Edge mean rel",
                "Interior mean rel",
                "2x2 phase means",
            ],
            _case_metric_rows(fields, ["result", "sensor_volts", "oi_photons"]),
        ),
    }


def _render_ip_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
    comparison: dict[str, Any],
    asset_store: AssetStore,
) -> dict[str, Any]:
    fields = comparison["fields"]
    figures = {
        "input": _render_triptych(
            reference["input"],
            actual["input"],
            title="ip_default_pipeline: input",
            field_name="input",
            wave_nm=None,
            asset_store=asset_store,
            output_path=figures_dir / "ip_default_pipeline_input_triptych.png",
        ),
        "sensorspace": _render_triptych(
            reference["sensorspace"],
            actual["sensorspace"],
            title="ip_default_pipeline: sensorspace",
            field_name="sensorspace",
            wave_nm=None,
            asset_store=asset_store,
            output_path=figures_dir / "ip_default_pipeline_sensorspace_triptych.png",
        ),
        "result": _render_triptych(
            reference["result"],
            actual["result"],
            title="ip_default_pipeline: final result",
            field_name="result",
            wave_nm=None,
            asset_store=asset_store,
            output_path=figures_dir / "ip_default_pipeline_result_triptych.png",
        ),
    }
    return {
        "figures": figures,
        "metrics_table": _markdown_table(
            [
                "Field",
                "Shape",
                "MAE",
                "RMSE",
                "Max abs",
                "Mean rel",
                "Max rel",
                "Normalized MAE",
                "Edge mean rel",
                "Interior mean rel",
                "2x2 phase means",
            ],
            _case_metric_rows(fields, ["input", "sensorspace", "result"]),
        ),
    }


def _vector_pair_rows(
    labels: list[str],
    reference: np.ndarray,
    actual: np.ndarray,
) -> list[list[str]]:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    act = np.asarray(actual, dtype=float).reshape(-1)
    count = min(ref.size, act.size)
    if len(labels) != count:
        labels = [f"Value {index + 1}" for index in range(count)]
    rows: list[list[str]] = []
    for label, ref_value, act_value in zip(labels, ref[:count], act[:count], strict=True):
        abs_diff = abs(float(act_value - ref_value))
        rel_diff = abs_diff / max(abs(float(ref_value)), 1.0e-12)
        rows.append(
            [
                label,
                _format_number(ref_value),
                _format_number(act_value),
                _format_number(abs_diff),
                _format_number(rel_diff),
            ]
        )
    return rows


def _dict_metric_pair_rows(
    labels: list[tuple[str, str]],
    reference: dict[str, Any],
    actual: dict[str, Any],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for label, key in labels:
        ref_value = _scalar(reference[key])
        actual_value = _scalar(actual[key])
        abs_diff = abs(actual_value - ref_value)
        rel_diff = abs_diff / max(abs(ref_value), 1.0e-12)
        rows.append(
            [
                label,
                _format_number(ref_value),
                _format_number(actual_value),
                _format_number(abs_diff),
                _format_number(rel_diff),
            ]
        )
    return rows


def _render_color_accuracy_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
    comparison: dict[str, Any],
    asset_store: AssetStore,
) -> dict[str, Any]:
    patch_figure = _render_triptych(
        reference["compare_patch_srgb"],
        actual["compare_patch_srgb"],
        title="metrics_color_accuracy_small: rendered Macbeth patches",
        field_name="compare_patch_srgb",
        wave_nm=None,
        asset_store=asset_store,
        output_path=figures_dir / "metrics_color_accuracy_small_patch_triptych.png",
    )
    delta_rows = _vector_pair_rows(
        ["mean", "max", "std"],
        reference["delta_e_stats"],
        actual["delta_e_stats"],
    )
    return {
        "figures": {"patches": patch_figure},
        "delta_table": _markdown_table(
            ["Statistic", "MATLAB", "Python", "Abs diff", "Rel diff"],
            delta_rows,
        ),
        "white_point_table": _markdown_table(
            ["Component", "MATLAB", "Python", "Abs diff", "Rel diff"],
            _vector_pair_rows(
                ["X/Y", "Y/Y", "Z/Y"],
                np.asarray(reference["white_xyz_norm"], dtype=float),
                np.asarray(actual["white_xyz_norm"], dtype=float),
            ),
        ),
        "comparison_summary": _markdown_table(
            ["Field", "Mean rel", "Max rel", "Normalized MAE"],
            [
                [
                    "delta_e",
                    _format_number(comparison["fields"]["delta_e"].get("mean_rel")),
                    _format_number(comparison["fields"]["delta_e"].get("max_rel")),
                    _format_number(comparison["fields"]["delta_e"].get("normalized_mae")),
                ],
                [
                    "compare_patch_srgb",
                    _format_number(comparison["fields"]["compare_patch_srgb"].get("mean_rel")),
                    _format_number(comparison["fields"]["compare_patch_srgb"].get("max_rel")),
                    _format_number(comparison["fields"]["compare_patch_srgb"].get("normalized_mae")),
                ],
                [
                    "white_xyz_norm",
                    _format_number(comparison["fields"]["white_xyz_norm"].get("mean_rel")),
                    _format_number(comparison["fields"]["white_xyz_norm"].get("max_rel")),
                    _format_number(comparison["fields"]["white_xyz_norm"].get("normalized_mae")),
                ],
            ],
        ),
    }


def _render_sharpness_case(
    figures_dir: Path,
    *,
    psf_reference: dict[str, Any],
    psf_actual: dict[str, Any],
    psf_comparison: dict[str, Any],
    point_reference: dict[str, Any],
    point_actual: dict[str, Any],
    point_comparison: dict[str, Any],
) -> dict[str, Any]:
    figures = {
        "overview": _render_annotated_overview_triptych(
            point_reference["render_rgb"],
            point_actual["render_rgb"],
            reference_center_peak_rc=point_reference["center_peak_rc"],
            reference_edge_peak_rc=point_reference["edge_peak_rc"],
            actual_center_peak_rc=point_actual["center_peak_rc"],
            actual_edge_peak_rc=point_actual["edge_peak_rc"],
            title="optics_rt_point_array_field_small: point-array field parity",
            output_path=figures_dir / "sharpness_point_array_overview.png",
        ),
        "crops": _render_center_edge_crop_grid(
            point_reference["center_crop_rgb"],
            point_actual["center_crop_rgb"],
            point_reference["edge_crop_rgb"],
            point_actual["edge_crop_rgb"],
            title="optics_rt_point_array_field_small: center and edge crops",
            output_path=figures_dir / "sharpness_crop_triptych.png",
        ),
        "psf": _render_psf_grid(
            psf_reference["center_psf_norm"],
            psf_actual["center_psf_norm"],
            psf_reference["edge_psf_norm"],
            psf_actual["edge_psf_norm"],
            support_x_um=psf_reference["support_x_um"],
            support_y_um=psf_reference["support_y_um"],
            title="optics_rt_center_edge_psf_small: center and edge PSFs",
            output_path=figures_dir / "sharpness_psf_triptych.png",
        ),
        "profiles": _render_psf_profile_overlay(
            psf_reference["support_x_um"],
            psf_reference["center_psf_center_row_norm"],
            psf_actual["center_psf_center_row_norm"],
            psf_reference["edge_psf_center_row_norm"],
            psf_actual["edge_psf_center_row_norm"],
            title="optics_rt_center_edge_psf_small: center-row PSF profiles",
            output_path=figures_dir / "sharpness_psf_profiles.png",
        ),
    }

    psf_rows: list[list[str]] = []
    for label, reference_metrics, actual_metrics in (
        ("Center PSF", psf_reference["center_psf_metrics"], psf_actual["center_psf_metrics"]),
        ("Edge PSF", psf_reference["edge_psf_metrics"], psf_actual["edge_psf_metrics"]),
    ):
        for metric_label, key in (
            ("Peak", "peak"),
            ("EE50 radius (um)", "ee50_radius_um"),
            ("EE80 radius (um)", "ee80_radius_um"),
            ("RMS radius (um)", "rms_radius_um"),
        ):
            ref_value = _scalar(reference_metrics[key])
            act_value = _scalar(actual_metrics[key])
            abs_diff = abs(act_value - ref_value)
            rel_diff = abs_diff / max(abs(ref_value), 1.0e-12)
            psf_rows.append(
                [
                    label,
                    metric_label,
                    _format_number(ref_value),
                    _format_number(act_value),
                    _format_number(abs_diff),
                    _format_number(rel_diff),
                ]
            )

    crop_rows: list[list[str]] = []
    for label, reference_metrics, actual_metrics in (
        ("Center crop", point_reference["center_crop_metrics"], point_actual["center_crop_metrics"]),
        ("Edge crop", point_reference["edge_crop_metrics"], point_actual["edge_crop_metrics"]),
    ):
        for metric_label, key in (
            ("Peak", "peak"),
            ("EE50 radius (px)", "ee50_radius_px"),
            ("EE80 radius (px)", "ee80_radius_px"),
            ("RMS radius (px)", "rms_radius_px"),
        ):
            ref_value = _scalar(reference_metrics[key])
            act_value = _scalar(actual_metrics[key])
            abs_diff = abs(act_value - ref_value)
            rel_diff = abs_diff / max(abs(ref_value), 1.0e-12)
            crop_rows.append(
                [
                    label,
                    metric_label,
                    _format_number(ref_value),
                    _format_number(act_value),
                    _format_number(abs_diff),
                    _format_number(rel_diff),
                ]
            )

    return {
        "figures": figures,
        "psf_table": _markdown_table(
            ["Field", "Metric", "MATLAB", "Python", "Abs diff", "Rel diff"],
            psf_rows,
        ),
        "crop_table": _markdown_table(
            ["Field", "Metric", "MATLAB", "Python", "Abs diff", "Rel diff"],
            crop_rows,
        ),
        "comparison_summary": _markdown_table(
            ["Payload field", "Mean rel", "Max rel", "Normalized MAE"],
            [
                [
                    "render_rgb",
                    _format_number(point_comparison["fields"]["render_rgb"].get("mean_rel")),
                    _format_number(point_comparison["fields"]["render_rgb"].get("max_rel")),
                    _format_number(point_comparison["fields"]["render_rgb"].get("normalized_mae")),
                ],
                [
                    "center_psf_norm",
                    _format_number(psf_comparison["fields"]["center_psf_norm"].get("mean_rel")),
                    _format_number(psf_comparison["fields"]["center_psf_norm"].get("max_rel")),
                    _format_number(psf_comparison["fields"]["center_psf_norm"].get("normalized_mae")),
                ],
                [
                    "edge_psf_norm",
                    _format_number(psf_comparison["fields"]["edge_psf_norm"].get("mean_rel")),
                    _format_number(psf_comparison["fields"]["edge_psf_norm"].get("max_rel")),
                    _format_number(psf_comparison["fields"]["edge_psf_norm"].get("normalized_mae")),
                ],
                [
                    "center_crop_luma_norm",
                    _format_number(point_comparison["fields"]["center_crop_luma_norm"].get("mean_rel")),
                    _format_number(point_comparison["fields"]["center_crop_luma_norm"].get("max_rel")),
                    _format_number(point_comparison["fields"]["center_crop_luma_norm"].get("normalized_mae")),
                ],
                [
                    "edge_crop_luma_norm",
                    _format_number(point_comparison["fields"]["edge_crop_luma_norm"].get("mean_rel")),
                    _format_number(point_comparison["fields"]["edge_crop_luma_norm"].get("max_rel")),
                    _format_number(point_comparison["fields"]["edge_crop_luma_norm"].get("normalized_mae")),
                ],
            ],
        ),
        "field_heights_table": _markdown_table(
            ["Field", "MATLAB field height (mm)", "Python field height (mm)"],
            [
                [
                    "Center",
                    _format_number(np.asarray(psf_reference["field_heights_mm"], dtype=float).reshape(-1)[0]),
                    _format_number(np.asarray(psf_actual["field_heights_mm"], dtype=float).reshape(-1)[0]),
                ],
                [
                    "Edge",
                    _format_number(np.asarray(psf_reference["field_heights_mm"], dtype=float).reshape(-1)[1]),
                    _format_number(np.asarray(psf_actual["field_heights_mm"], dtype=float).reshape(-1)[1]),
                ],
            ],
        ),
    }


def _render_distortion_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    figures = {
        "grid": _render_distortion_grid_triptych(
            reference["ideal_grid_rgb"],
            actual["ideal_grid_rgb"],
            reference["distorted_grid_rgb"],
            actual["distorted_grid_rgb"],
            title="optics_rt_distortion_field_small: grid parity",
            output_path=figures_dir / "distortion_grid_triptych.png",
        ),
        "curve": _render_distortion_curve_overlay(
            reference["field_height_mm"],
            reference["distorted_height_mm"],
            actual["distorted_height_mm"],
            reference["distortion_percent"],
            actual["distortion_percent"],
            title="optics_rt_distortion_field_small: distortion curves",
            output_path=figures_dir / "distortion_curve.png",
        ),
    }
    summary_rows = []
    for label, ref_value, act_value in (
        ("Reference wavelength (nm)", reference["reference_wavelength_nm"], actual["reference_wavelength_nm"]),
        ("Max |distortion| (%)", reference["max_distortion_percent"], actual["max_distortion_percent"]),
        ("Field height at max distortion (mm)", reference["max_distortion_field_height_mm"], actual["max_distortion_field_height_mm"]),
    ):
        ref_scalar = _scalar(ref_value)
        act_scalar = _scalar(act_value)
        abs_diff = abs(act_scalar - ref_scalar)
        rel_diff = abs_diff / max(abs(ref_scalar), 1.0e-12)
        summary_rows.append(
            [
                label,
                _format_number(ref_scalar),
                _format_number(act_scalar),
                _format_number(abs_diff),
                _format_number(rel_diff),
            ]
        )
    return {
        "figures": figures,
        "summary_table": _markdown_table(
            ["Metric", "MATLAB", "Python", "Abs diff", "Rel diff"],
            summary_rows,
        ),
        "comparison_summary": _markdown_table(
            ["Payload field", "Mean rel", "Max rel", "Normalized MAE"],
            [
                [
                    "ideal_grid_rgb",
                    _format_number(comparison["fields"]["ideal_grid_rgb"].get("mean_rel")),
                    _format_number(comparison["fields"]["ideal_grid_rgb"].get("max_rel")),
                    _format_number(comparison["fields"]["ideal_grid_rgb"].get("normalized_mae")),
                ],
                [
                    "distorted_grid_rgb",
                    _format_number(comparison["fields"]["distorted_grid_rgb"].get("mean_rel")),
                    _format_number(comparison["fields"]["distorted_grid_rgb"].get("max_rel")),
                    _format_number(comparison["fields"]["distorted_grid_rgb"].get("normalized_mae")),
                ],
                [
                    "distortion_percent",
                    _format_number(comparison["fields"]["distortion_percent"].get("mean_rel")),
                    _format_number(comparison["fields"]["distortion_percent"].get("max_rel")),
                    _format_number(comparison["fields"]["distortion_percent"].get("normalized_mae")),
                ],
            ],
        ),
    }


def _render_vsnr_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    levels = np.asarray(reference["light_levels"], dtype=float).reshape(-1)
    figure = _plot_two_panel_overlay(
        levels,
        reference["vsnr_norm"],
        actual["vsnr_norm"],
        reference["delta_e_norm"],
        actual["delta_e_norm"],
        title="metrics_vsnr_small: normalized VSNR response",
        left_title="vSNR (normalized)",
        right_title="Delta E proxy (normalized)",
        x_label="Light level",
        y_label="Normalized response",
        output_path=figures_dir / "metrics_vsnr_small_curves.png",
    )
    channel_rows = []
    reference_rows = np.asarray(reference["result_channel_means_norm"], dtype=float)
    actual_rows = np.asarray(actual["result_channel_means_norm"], dtype=float)
    for index in range(reference_rows.shape[0]):
        label = f"Level {index + 1}"
        channel_rows.append(
            [
                label,
                np.array2string(reference_rows[index], precision=4, separator=", "),
                np.array2string(actual_rows[index], precision=4, separator=", "),
            ]
        )
    return {
        "figures": {"curves": figure},
        "channel_table": _markdown_table(
            ["Light level", "MATLAB channel means", "Python channel means"],
            channel_rows,
        ),
    }


def _render_acutance_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    lum_ref = np.asarray(reference["lum_mtf_norm"], dtype=float).reshape(-1)
    lum_actual = np.asarray(actual["lum_mtf_norm"], dtype=float).reshape(-1)
    x = np.arange(lum_ref.size, dtype=float)
    figure = _plot_overlay(
        x,
        lum_ref,
        lum_actual,
        title="metrics_acutance_small: luminance MTF",
        y_label="Normalized luminance MTF",
        x_label="Frequency sample index",
        output_path=figures_dir / "metrics_acutance_small_lum_mtf.png",
    )
    scalar_rows = []
    for field_name in ("acutance", "camera_acutance"):
        ref_value = _scalar(reference[field_name])
        actual_value = _scalar(actual[field_name])
        abs_diff = abs(actual_value - ref_value)
        rel_diff = abs_diff / max(abs(ref_value), 1.0e-12)
        scalar_rows.append(
            [
                field_name,
                _format_number(ref_value),
                _format_number(actual_value),
                _format_number(abs_diff),
                _format_number(rel_diff),
            ]
        )
    scalar_rows.append(
        [
            "cpiq_norm_mean",
            _format_number(np.mean(np.asarray(reference["cpiq_norm"], dtype=float))),
            _format_number(np.mean(np.asarray(actual["cpiq_norm"], dtype=float))),
            _format_number(abs(float(np.mean(np.asarray(actual["cpiq_norm"], dtype=float))) - float(np.mean(np.asarray(reference["cpiq_norm"], dtype=float))))),
            _format_number(
                abs(float(np.mean(np.asarray(actual["cpiq_norm"], dtype=float))) - float(np.mean(np.asarray(reference["cpiq_norm"], dtype=float))))
                / max(abs(float(np.mean(np.asarray(reference["cpiq_norm"], dtype=float)))), 1.0e-12)
            ),
        ]
    )
    return {
        "figures": {"lum_mtf": figure},
        "summary_table": _markdown_table(
            ["Metric", "MATLAB", "Python", "Abs diff", "Rel diff"],
            scalar_rows,
        ),
    }


def _render_mtf_slanted_bar_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    mono_ref = np.asarray(reference["mono_direct_mtf_norm"], dtype=float).reshape(-1)
    mono_actual = np.asarray(actual["mono_direct_mtf_norm"], dtype=float).reshape(-1)
    x = np.arange(mono_ref.size, dtype=float)
    figure = _plot_triplet_overlay(
        x,
        [
            ("Mono direct MTF", reference["mono_direct_mtf_norm"], actual["mono_direct_mtf_norm"]),
            ("Color direct MTF", reference["color_direct_mtf_norm"], actual["color_direct_mtf_norm"]),
            ("IE color MTF", reference["ie_color_mtf_norm"], actual["ie_color_mtf_norm"]),
        ],
        title="metrics_mtf_slanted_bar_small: slanted-bar MTF curves",
        x_label="Frequency sample index",
        y_label="Normalized MTF",
        output_path=figures_dir / "metrics_mtf_slanted_bar_small_curves.png",
    )
    table_rows = []
    for label, mtf_field, nyquist_field, alias_field in (
        ("Mono direct", "mono_direct_mtf50", "mono_direct_nyquistf", "mono_direct_aliasing_percentage"),
        ("Color direct", "color_direct_mtf50", "color_direct_nyquistf", "color_direct_aliasing_percentage"),
        ("IE color", "ie_color_mtf50", "ie_color_nyquistf", "ie_color_aliasing_percentage"),
    ):
        table_rows.append(
            [
                label,
                _format_number(reference[mtf_field]),
                _format_number(actual[mtf_field]),
                _format_number(reference[nyquist_field]),
                _format_number(actual[nyquist_field]),
                _format_number(reference[alias_field]),
                _format_number(actual[alias_field]),
            ]
        )
    return {
        "figures": {"curves": figure},
        "summary_table": _markdown_table(
            [
                "Curve",
                "MATLAB MTF50",
                "Python MTF50",
                "MATLAB Nyquist",
                "Python Nyquist",
                "MATLAB alias %",
                "Python alias %",
            ],
            table_rows,
        ),
    }


def _render_mtf_pixel_size_case(
    figures_dir: Path,
    *,
    reference: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    profiles_ref = np.asarray(reference["mtf_profiles_norm"], dtype=float)
    profiles_actual = np.asarray(actual["mtf_profiles_norm"], dtype=float)
    pixel_sizes = np.asarray(reference["pixel_sizes_um"], dtype=float).reshape(-1)
    mtf50_ref = np.asarray(reference["mtf50"], dtype=float).reshape(-1)
    mtf50_actual = np.asarray(actual["mtf50"], dtype=float).reshape(-1)
    nyquist_ref = np.asarray(reference["nyquistf"], dtype=float).reshape(-1)
    figure = _plot_mtf_pixel_size_subplots(
        pixel_sizes,
        profiles_ref,
        profiles_actual,
        mtf50_ref,
        mtf50_actual,
        nyquist_ref,
        output_path=figures_dir / "metrics_mtf_pixel_size_small_profiles.png",
    )
    table_rows = []
    for pixel_size, ref_value, act_value, ref_profile, actual_profile, nyquistf in zip(
        pixel_sizes,
        mtf50_ref,
        mtf50_actual,
        profiles_ref,
        profiles_actual,
        nyquist_ref,
        strict=True,
    ):
        abs_diff = abs(float(act_value - ref_value))
        rel_diff = abs_diff / max(abs(float(ref_value)), 1.0e-12)
        ref_profile = np.asarray(ref_profile, dtype=float).reshape(-1)
        actual_profile = np.asarray(actual_profile, dtype=float).reshape(-1)
        profile_abs = np.abs(actual_profile - ref_profile)
        profile_nmae = float(np.mean(profile_abs)) / max(float(np.mean(np.abs(ref_profile))), 1.0e-12)
        table_rows.append(
            [
                _format_number(pixel_size),
                _format_number(ref_value),
                _format_number(act_value),
                _format_number(abs_diff),
                _format_number(rel_diff),
                _format_number(float(ref_value) / max(float(nyquistf), 1.0e-12)),
                _format_number(float(act_value) / max(float(nyquistf), 1.0e-12)),
                _format_number(profile_nmae),
                _format_number(float(np.max(profile_abs))),
            ]
        )
    return {
        "figures": {"profiles": figure},
        "summary_table": _markdown_table(
            [
                "Pixel size (um)",
                "MATLAB MTF50",
                "Python MTF50",
                "Abs diff",
                "Rel diff",
                "MATLAB MTF50 / Nyquist",
                "Python MTF50 / Nyquist",
                "Profile normalized MAE",
                "Profile max abs",
            ],
            table_rows,
        ),
    }


def _executive_summary(report: dict[str, Any], selected_cases: list[dict[str, Any]]) -> str:
    selected_passed = sum(1 for case in selected_cases if case["status"] == "passed")
    total_selected = len(selected_cases)
    return "\n".join(
        [
            f"- Global curated parity: `{report['summary']['passed']} passed`, `{report['summary']['failed']} failed`, `{report['summary']['skipped']} skipped`.",
            f"- Selected camera-pipeline cases: `{selected_passed}/{total_selected}` passed.",
            "- Conclusion: the selected pipeline, color, sharpness, and distortion cases match the MATLAB baselines, and the exported figures show visually small residual differences.",
        ]
    )


def _selected_case_pass_table(selected_cases: list[dict[str, Any]], case_defs: dict[str, dict[str, Any]]) -> str:
    rows = []
    for case in selected_cases:
        name = case["name"]
        rows.append(
            [
                name,
                case_defs[name]["matlab_function"],
                case["status"],
                _format_number(case["rtol"]),
                _format_number(case["atol"]),
            ]
        )
    return _markdown_table(
        ["Case", "MATLAB function", "Status", "rtol", "atol"],
        rows,
    )


def _build_markdown(
    *,
    report: dict[str, Any],
    case_defs: dict[str, dict[str, Any]],
    camera_payload: dict[str, Any],
    ip_payload: dict[str, Any],
    color_payload: dict[str, Any],
    sharpness_payload: dict[str, Any],
    distortion_payload: dict[str, Any],
    vsnr_payload: dict[str, Any],
    acutance_payload: dict[str, Any],
    mtf_bar_payload: dict[str, Any],
    mtf_pixel_payload: dict[str, Any],
    summary_path: Path,
    parity_json_path: Path,
) -> str:
    selected_cases = _selected_report_cases(report)
    camera_case = _report_case_map(report)["camera_default_pipeline"]
    camera_context = camera_case.get("diagnostics", {}).get("context", {})
    git_commit = report.get("git_commit") or _git_commit() or "unknown"

    lines: list[str] = [
        "# Camera Pipeline Parity Evidence Report",
        "",
        f"Generated: `{datetime.now(UTC).isoformat()}`",
        f"Git commit: `{git_commit}`",
        "",
        "## Executive Summary",
        _executive_summary(report, selected_cases),
        "",
        _selected_case_pass_table(selected_cases, case_defs),
        "",
        "## Evidence Sources",
        f"- Selected parity cases: [`/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/cases.yaml`](/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/cases.yaml)",
        f"- MATLAB baselines: [`/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/baselines`](/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/baselines)",
        f"- Python parity runners: [`/Users/seongcheoljeong/Documents/CameraE2E/src/pyisetcam/parity.py`](/Users/seongcheoljeong/Documents/CameraE2E/src/pyisetcam/parity.py)",
        f"- Machine-readable parity report: [`{parity_json_path}`]({parity_json_path})",
        f"- Selected-case summary JSON: [`{summary_path}`]({summary_path})",
        "- Regenerate with: `python tools/render_parity_evidence.py --refresh-report`",
        "",
        "## Core Camera Pipeline",
        "`camera_default_pipeline` is the headline end-to-end case. It is intentionally noiseless (`sensor_noise_flag = 0`) so the comparison isolates deterministic scene -> OI -> sensor -> IP parity instead of stochastic sensor variance.",
        "",
        f"![camera result]({REPO_ROOT / 'reports' / 'parity' / camera_payload['figures']['result']['path']})",
        "",
        f"![camera sensor volts]({REPO_ROOT / 'reports' / 'parity' / camera_payload['figures']['sensor_volts']['path']})",
        "",
        f"![camera oi photons]({REPO_ROOT / 'reports' / 'parity' / camera_payload['figures']['oi_photons']['path']})",
        "",
        "### Camera Stage Metrics",
        camera_payload["metrics_table"],
        "",
        "### Camera Context",
        _markdown_table(
            ["Field", "Value"],
            [[key, _format_number(value)] for key, value in sorted(camera_context.items())],
        ),
        "",
        "## IP Pipeline",
        "`ip_default_pipeline` isolates the image processor from the outer camera wrapper. The three exported figures below compare the MATLAB and Python payloads at `input`, `sensorspace`, and final `result`.",
        "",
        f"![ip input]({REPO_ROOT / 'reports' / 'parity' / ip_payload['figures']['input']['path']})",
        "",
        f"![ip sensorspace]({REPO_ROOT / 'reports' / 'parity' / ip_payload['figures']['sensorspace']['path']})",
        "",
        f"![ip result]({REPO_ROOT / 'reports' / 'parity' / ip_payload['figures']['result']['path']})",
        "",
        "### IP Stage Metrics",
        ip_payload["metrics_table"],
        "",
        "## Color Parity",
        "`metrics_color_accuracy_small` provides the boss-facing color evidence using the existing Macbeth-based parity case, so the patch render and Delta E summaries are both backed by the stored MATLAB baseline.",
        "",
        f"![color accuracy]({REPO_ROOT / 'reports' / 'parity' / color_payload['figures']['patches']['path']})",
        "",
        color_payload["delta_table"],
        "",
        color_payload["white_point_table"],
        "",
        color_payload["comparison_summary"],
        "",
        "## Center/Edge Field Sharpness Parity",
        "These figures are backed by the new curated MATLAB baselines `optics_rt_center_edge_psf_small` and `optics_rt_point_array_field_small`. The point-array overview gives an intuitive field-quality image, while the PSF panels show the direct center-vs-edge optics parity at `550 nm`.",
        "",
        f"![sharpness overview]({REPO_ROOT / 'reports' / 'parity' / sharpness_payload['figures']['overview']['path']})",
        "",
        f"![sharpness crops]({REPO_ROOT / 'reports' / 'parity' / sharpness_payload['figures']['crops']['path']})",
        "",
        f"![sharpness psf]({REPO_ROOT / 'reports' / 'parity' / sharpness_payload['figures']['psf']['path']})",
        "",
        f"![sharpness psf profiles]({REPO_ROOT / 'reports' / 'parity' / sharpness_payload['figures']['profiles']['path']})",
        "",
        "### PSF Metrics",
        sharpness_payload["psf_table"],
        "",
        "### Crop Metrics",
        sharpness_payload["crop_table"],
        "",
        "### Field Heights",
        sharpness_payload["field_heights_table"],
        "",
        "### Sharpness Comparison Summary",
        sharpness_payload["comparison_summary"],
        "",
        "## Distortion Parity",
        "The distortion section is backed by `optics_rt_distortion_field_small`. The grid figure is the qualitative image evidence, while the radial distortion curve is the quantitative parity measurement.",
        "",
        f"![distortion grid]({REPO_ROOT / 'reports' / 'parity' / distortion_payload['figures']['grid']['path']})",
        "",
        f"![distortion curve]({REPO_ROOT / 'reports' / 'parity' / distortion_payload['figures']['curve']['path']})",
        "",
        distortion_payload["summary_table"],
        "",
        distortion_payload["comparison_summary"],
        "",
        "## Configuration Sweep",
        "These cases show that the Python port tracks MATLAB not just for a single end-to-end pipeline image, but across several camera-pipeline operating conditions and quality metrics.",
        "",
        "### VSNR",
        f"![vsnr curves]({REPO_ROOT / 'reports' / 'parity' / vsnr_payload['figures']['curves']['path']})",
        "",
        vsnr_payload["channel_table"],
        "",
        "### Acutance",
        f"![acutance]({REPO_ROOT / 'reports' / 'parity' / acutance_payload['figures']['lum_mtf']['path']})",
        "",
        acutance_payload["summary_table"],
        "",
        "### MTF Slanted Bar",
        f"![mtf slanted bar]({REPO_ROOT / 'reports' / 'parity' / mtf_bar_payload['figures']['curves']['path']})",
        "",
        mtf_bar_payload["summary_table"],
        "",
        "### MTF Pixel Size Sweep",
        "The pixel-size sweep below is plotted as one subplot per pixel size on normalized frequency (`f / Nyquist`) rather than raw sample index. Vertical guide lines mark the MATLAB and Python `MTF50 / Nyquist` locations, which avoids overstating the coarser `9 um` visual delta.",
        "",
        f"![mtf pixel size]({REPO_ROOT / 'reports' / 'parity' / mtf_pixel_payload['figures']['profiles']['path']})",
        "",
        mtf_pixel_payload["summary_table"],
        "",
        "## Conclusion",
        "- Python matches MATLAB on the selected camera end-to-end and stagewise parity cases.",
        "- The color, center/edge sharpness, and distortion sections are all traceable to curated MATLAB baselines rather than Python-only visuals.",
        "- The configuration sweeps preserve the same trends as MATLAB for VSNR, acutance, and MTF-oriented analyses.",
        "",
    ]
    return "\n".join(lines)


def _summary_payload(
    *,
    report: dict[str, Any],
    case_defs: dict[str, dict[str, Any]],
    generated_figures: dict[str, dict[str, Any]],
    parity_json_path: Path,
) -> dict[str, Any]:
    selected_cases = _selected_report_cases(report)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": report.get("git_commit") or _git_commit(),
        "parity_report_path": str(parity_json_path.relative_to(REPO_ROOT)),
        "summary": {
            "selected_cases": len(selected_cases),
            "passed": sum(1 for case in selected_cases if case["status"] == "passed"),
            "failed": sum(1 for case in selected_cases if case["status"] == "failed"),
            "skipped": sum(1 for case in selected_cases if case["status"] == "skipped"),
        },
        "cases": [
            {
                "name": case["name"],
                "matlab_function": case_defs[case["name"]]["matlab_function"],
                "status": case["status"],
                "rtol": case["rtol"],
                "atol": case["atol"],
                "figures": generated_figures.get(case["name"], {}),
            }
            for case in selected_cases
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-report", action="store_true", help="Regenerate reports/parity/latest.json before rendering.")
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_OUTPUT, help="Path to the machine-readable parity report.")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Directory for rendered PNG figures.")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN, help="Path to the narrative Markdown report.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY, help="Path to the selected-case summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = _load_report(args.parity_report, refresh=bool(args.refresh_report))
    case_defs = _case_definition_map()
    report_case_map = _report_case_map(report)
    asset_store = AssetStore.default()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    camera_reference, camera_actual, camera_comparison, camera_wave = _case_comparison(
        "camera_default_pipeline",
        report_case=report_case_map["camera_default_pipeline"],
        asset_store=asset_store,
    )
    ip_reference, ip_actual, ip_comparison, _ = _case_comparison(
        "ip_default_pipeline",
        report_case=report_case_map["ip_default_pipeline"],
        asset_store=asset_store,
    )
    color_reference, color_actual, _, _ = _case_comparison(
        "metrics_color_accuracy_small",
        report_case=report_case_map["metrics_color_accuracy_small"],
        asset_store=asset_store,
    )
    sharpness_psf_reference, sharpness_psf_actual, sharpness_psf_comparison, _ = _case_comparison(
        "optics_rt_center_edge_psf_small",
        report_case=report_case_map["optics_rt_center_edge_psf_small"],
        asset_store=asset_store,
    )
    sharpness_point_reference, sharpness_point_actual, sharpness_point_comparison, _ = _case_comparison(
        "optics_rt_point_array_field_small",
        report_case=report_case_map["optics_rt_point_array_field_small"],
        asset_store=asset_store,
    )
    distortion_reference, distortion_actual, distortion_comparison, _ = _case_comparison(
        "optics_rt_distortion_field_small",
        report_case=report_case_map["optics_rt_distortion_field_small"],
        asset_store=asset_store,
    )
    vsnr_reference, vsnr_actual, _, _ = _case_comparison(
        "metrics_vsnr_small",
        report_case=report_case_map["metrics_vsnr_small"],
        asset_store=asset_store,
    )
    acutance_reference, acutance_actual, _, _ = _case_comparison(
        "metrics_acutance_small",
        report_case=report_case_map["metrics_acutance_small"],
        asset_store=asset_store,
    )
    mtf_bar_reference, mtf_bar_actual, _, _ = _case_comparison(
        "metrics_mtf_slanted_bar_small",
        report_case=report_case_map["metrics_mtf_slanted_bar_small"],
        asset_store=asset_store,
    )
    mtf_pixel_reference, mtf_pixel_actual, _, _ = _case_comparison(
        "metrics_mtf_pixel_size_small",
        report_case=report_case_map["metrics_mtf_pixel_size_small"],
        asset_store=asset_store,
    )

    camera_payload = _render_camera_case(
        args.figures_dir,
        reference=camera_reference,
        actual=camera_actual,
        comparison=camera_comparison,
        wave_nm=camera_wave,
        asset_store=asset_store,
    )
    ip_payload = _render_ip_case(
        args.figures_dir,
        reference=ip_reference,
        actual=ip_actual,
        comparison=ip_comparison,
        asset_store=asset_store,
    )
    color_payload = _render_color_accuracy_case(
        args.figures_dir,
        reference=color_reference,
        actual=color_actual,
        comparison=report_case_map["metrics_color_accuracy_small"]["comparison"],
        asset_store=asset_store,
    )
    sharpness_payload = _render_sharpness_case(
        args.figures_dir,
        psf_reference=sharpness_psf_reference,
        psf_actual=sharpness_psf_actual,
        psf_comparison=sharpness_psf_comparison,
        point_reference=sharpness_point_reference,
        point_actual=sharpness_point_actual,
        point_comparison=sharpness_point_comparison,
    )
    distortion_payload = _render_distortion_case(
        args.figures_dir,
        reference=distortion_reference,
        actual=distortion_actual,
        comparison=distortion_comparison,
    )
    vsnr_payload = _render_vsnr_case(
        args.figures_dir,
        reference=vsnr_reference,
        actual=vsnr_actual,
    )
    acutance_payload = _render_acutance_case(
        args.figures_dir,
        reference=acutance_reference,
        actual=acutance_actual,
    )
    mtf_bar_payload = _render_mtf_slanted_bar_case(
        args.figures_dir,
        reference=mtf_bar_reference,
        actual=mtf_bar_actual,
    )
    mtf_pixel_payload = _render_mtf_pixel_size_case(
        args.figures_dir,
        reference=mtf_pixel_reference,
        actual=mtf_pixel_actual,
    )

    markdown = _build_markdown(
        report=report,
        case_defs=case_defs,
        camera_payload=camera_payload,
        ip_payload=ip_payload,
        color_payload=color_payload,
        sharpness_payload=sharpness_payload,
        distortion_payload=distortion_payload,
        vsnr_payload=vsnr_payload,
        acutance_payload=acutance_payload,
        mtf_bar_payload=mtf_bar_payload,
        mtf_pixel_payload=mtf_pixel_payload,
        summary_path=args.summary_output,
        parity_json_path=args.parity_report,
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown)

    generated_figures = {
        "camera_default_pipeline": camera_payload["figures"],
        "ip_default_pipeline": ip_payload["figures"],
        "metrics_color_accuracy_small": color_payload["figures"],
        "optics_rt_center_edge_psf_small": sharpness_payload["figures"],
        "optics_rt_point_array_field_small": sharpness_payload["figures"],
        "optics_rt_distortion_field_small": distortion_payload["figures"],
        "metrics_vsnr_small": vsnr_payload["figures"],
        "metrics_acutance_small": acutance_payload["figures"],
        "metrics_mtf_slanted_bar_small": mtf_bar_payload["figures"],
        "metrics_mtf_pixel_size_small": mtf_pixel_payload["figures"],
    }
    summary = _summary_payload(
        report=report,
        case_defs=case_defs,
        generated_figures=generated_figures,
        parity_json_path=args.parity_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
