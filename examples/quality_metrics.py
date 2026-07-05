from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from pyisetcam import AssetStore, camera_acutance, camera_compute, camera_create, comparison_metrics, scene_create


def _default_output_dir() -> Path:
    return Path("reports/tutorial/quality_metrics")


def _to_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot save an empty image.")
    clipped = np.clip(array, 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def main(output_dir: Path | None = None, asset_store: AssetStore | None = None) -> dict[str, Any]:
    store = asset_store or AssetStore.default()
    destination = (output_dir if output_dir is not None else _default_output_dir()).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    camera = camera_create(asset_store=store)
    scene = scene_create(asset_store=store)
    rendered_camera = camera_compute(camera.clone(), scene, asset_store=store)
    reference = np.asarray(rendered_camera.fields["ip"].data["srgb"], dtype=float)
    degraded = np.asarray(gaussian_filter(reference, sigma=(1.2, 1.2, 0.0)), dtype=float)

    metrics = comparison_metrics(reference, degraded, data_range=1.0)
    acutance = float(camera_acutance(camera.clone(), asset_store=store))

    difference = np.mean(np.abs(reference - degraded), axis=2, dtype=float)
    figure_path = destination / "quality_summary.png"
    reference_path = destination / "reference.png"
    degraded_path = destination / "degraded.png"

    iio.imwrite(reference_path, _to_uint8(reference))
    iio.imwrite(degraded_path, _to_uint8(degraded))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    fig.suptitle("Camera quality metrics")

    axes[0, 0].imshow(np.clip(reference, 0.0, 1.0), interpolation="nearest")
    axes[0, 0].set_title("Reference render")
    axes[0, 1].imshow(np.clip(degraded, 0.0, 1.0), interpolation="nearest")
    axes[0, 1].set_title("Blurred comparison")
    diff_artist = axes[1, 0].imshow(difference, cmap="inferno", interpolation="nearest")
    axes[1, 0].set_title("Absolute difference")
    plt.colorbar(diff_artist, ax=axes[1, 0], fraction=0.046, pad=0.04)

    metric_names = ["mae", "rmse", "max_abs"]
    metric_values = [float(metrics[name]) for name in metric_names]
    axes[1, 1].bar(metric_names, metric_values, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1, 1].set_title("Comparison metrics")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Acutance: {acutance:.4f}",
                f"Mean rel: {float(metrics['mean_rel']):.4f}",
                f"PSNR: {float(metrics['psnr']):.2f} dB",
            ]
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    for axis in axes[:2, :2].reshape(-1)[:3]:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    summary = {
        "figure_path": figure_path,
        "reference_path": reference_path,
        "degraded_path": degraded_path,
        "camera_acutance": acutance,
        "comparison_metrics": {key: float(value) for key, value in metrics.items()},
    }

    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    main()
