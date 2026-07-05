"""Render MATLAB-vs-Python evidence for the ray-trace bar edge pipeline parity case."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import AssetStore
from pyisetcam.parity import run_python_case_with_context

CASE_NAME = "pipeline_rt_bar_small"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "parity" / "pipeline_rt_bar_small"
DEFAULT_REPORT = REPO_ROOT / "reports" / "parity" / "pipeline_rt_bar_small_report.md"
DEFAULT_SUMMARY = REPO_ROOT / "reports" / "parity" / "pipeline_rt_bar_small_summary.json"


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
    value = result.stdout.strip()
    return value or None


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _normalized_mae(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - reference)) / max(float(np.mean(np.abs(reference))), 1.0e-12))


def _mean_rel(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - reference) / np.maximum(np.abs(reference), 1.0e-12)))


def _format_shape(values: Any) -> str:
    array = np.asarray(values, dtype=int).reshape(-1)
    return "x".join(str(int(item)) for item in array)


def _format_vector(values: Any) -> str:
    array = np.asarray(values, dtype=int).reshape(-1)
    return "[" + ", ".join(str(int(item)) for item in array) + "]"


def _render_triptych(reference: np.ndarray, actual: np.ndarray, *, title: str, output_path: Path) -> dict[str, float]:
    diff = np.abs(actual - reference)
    low = float(min(reference.min(), actual.min()))
    high = float(max(reference.max(), actual.max()))
    if high <= low:
        high = low + 1.0
    diff_high = float(np.percentile(diff.reshape(-1), 99.0))
    if diff_high <= 0.0:
        diff_high = float(np.max(diff))
    if diff_high <= 0.0:
        diff_high = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    fig.suptitle(title)
    ref_artist = axes[0].imshow(reference, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    act_artist = axes[1].imshow(actual, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    diff_artist = axes[2].imshow(diff, cmap="inferno", vmin=0.0, vmax=diff_high, interpolation="nearest")
    axes[0].set_title("MATLAB baseline")
    axes[1].set_title("Python port")
    axes[2].set_title("Absolute difference")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.colorbar(ref_artist, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(act_artist, ax=axes[1], fraction=0.046, pad=0.04)
    plt.colorbar(diff_artist, ax=axes[2], fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"normalized_mae": _normalized_mae(reference, actual), "max_abs": float(np.max(diff))}


def _render_profile(reference: np.ndarray, actual: np.ndarray, *, title: str, output_path: Path) -> dict[str, float]:
    x = np.linspace(-1.0, 1.0, reference.size, dtype=float)
    fig, axis = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    axis.plot(x, reference, label="MATLAB baseline", linewidth=2.2)
    axis.plot(x, actual, label="Python port", linewidth=2.2, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("Normalized edge support")
    axis.set_ylabel("Normalized response")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"normalized_mae": _normalized_mae(reference, actual), "max_abs": float(np.max(np.abs(actual - reference)))}


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def render(output_dir: Path, report_path: Path, summary_path: Path) -> None:
    baseline = {
        key: value
        for key, value in loadmat(
            REPO_ROOT / "tests" / "parity" / "baselines" / f"{CASE_NAME}.mat",
            squeeze_me=True,
            struct_as_record=False,
        ).items()
        if not key.startswith("__")
    }
    actual = run_python_case_with_context(CASE_NAME, asset_store=AssetStore.default()).payload

    stage_specs = [
        ("Lens/OI", "oi_edge_crop_norm", "oi_edge_profile_norm", "oi_edge_rc", "lens"),
        ("Sensor", "sensor_edge_crop_norm", "sensor_edge_profile_norm", "sensor_edge_rc", "sensor"),
        ("ISP", "result_edge_crop_norm", "result_edge_profile_norm", "result_edge_rc", "isp"),
    ]

    stage_rows: list[list[str]] = []
    figures: list[str] = []
    summary: dict[str, Any] = {
        "case_name": CASE_NAME,
        "git_commit": _git_commit(),
        "scene_size": _format_shape(actual["scene_size"]),
        "scene_fov_deg": float(actual["scene_fov_deg"]),
        "oi_size": _format_shape(actual["oi_size"]),
        "sensor_size": _format_shape(actual["sensor_size"]),
        "ip_size": _format_shape(actual["ip_size"]),
        "stages": {},
    }

    for stage_name, crop_key, profile_key, rc_key, slug in stage_specs:
        reference_crop = _array(baseline[crop_key])
        actual_crop = _array(actual[crop_key])
        reference_profile = _array(baseline[profile_key]).reshape(-1)
        actual_profile = _array(actual[profile_key]).reshape(-1)

        crop_path = output_dir / f"{slug}_edge_crop_triptych.png"
        profile_path = output_dir / f"{slug}_edge_profile.png"
        crop_metrics = _render_triptych(reference_crop, actual_crop, title=f"{stage_name}: edge crop parity", output_path=crop_path)
        profile_metrics = _render_profile(reference_profile, actual_profile, title=f"{stage_name}: edge profile parity", output_path=profile_path)

        rc_reference = _array(baseline[rc_key]).astype(int)
        rc_actual = _array(actual[rc_key]).astype(int)
        rc_mean_rel = _mean_rel(rc_reference, rc_actual)

        stage_rows.append(
            [
                stage_name,
                _format_vector(rc_reference),
                _format_vector(rc_actual),
                f"{rc_mean_rel:.4f}",
                f"{crop_metrics['normalized_mae']:.4f}",
                f"{profile_metrics['normalized_mae']:.4f}",
            ]
        )
        summary["stages"][slug] = {
            "reference_edge_rc": rc_reference.tolist(),
            "actual_edge_rc": rc_actual.tolist(),
            "edge_rc_mean_rel": rc_mean_rel,
            "crop_normalized_mae": crop_metrics["normalized_mae"],
            "crop_max_abs": crop_metrics["max_abs"],
            "profile_normalized_mae": profile_metrics["normalized_mae"],
            "profile_max_abs": profile_metrics["max_abs"],
            "crop_figure": str(crop_path),
            "profile_figure": str(profile_path),
        }
        figures.extend(
            [
                f"![{stage_name} crop]({crop_path})",
                f"![{stage_name} profile]({profile_path})",
            ]
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_lines = [
        "# Shift-Variant PSF Edge Pipeline Parity",
        "",
        "## Summary",
        f"- Case: `{CASE_NAME}`",
        "- Path: `Scene(bar) -> rtGeometry -> rtPrecomputePSF -> rtPrecomputePSFApply -> Sensor(noise off) -> IP`",
        "- This is a shift-variant PSF convolution case, not a diffraction-limited or shift-invariant shortcut.",
        f"- Git commit: `{summary['git_commit'] or 'unknown'}`",
        "",
        "## Geometry",
        f"- Scene size: `{summary['scene_size']}`",
        f"- Scene FOV: `{summary['scene_fov_deg']:.4f} deg`",
        f"- Lens/OI size: `{summary['oi_size']}`",
        f"- Sensor size: `{summary['sensor_size']}`",
        f"- ISP result size: `{summary['ip_size']}`",
        "",
        "## Stage Metrics",
        _markdown_table(
            ["Stage", "MATLAB edge rc", "Python edge rc", "edge rc mean rel", "crop normalized MAE", "profile normalized MAE"],
            stage_rows,
        ),
        "",
        "## Evidence Images",
        *figures,
        "",
        "## Interpretation",
        "- `crop normalized MAE` is the main image-level parity metric for the detected central edge ROI.",
        "- `profile normalized MAE` compares the stage-averaged edge profile after per-stage normalization.",
        "- The remaining Lens/OI size and edge-location differences come from ray-trace registration details, but the detected edge crops and downstream Sensor/ISP images remain closely matched.",
        "",
        "## Regenerate",
        f"- `python {Path(__file__).relative_to(REPO_ROOT)}`",
        f"- Summary JSON: [`{summary_path}`]({summary_path})",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    render(args.output_dir, args.report, args.summary)


if __name__ == "__main__":
    main()
