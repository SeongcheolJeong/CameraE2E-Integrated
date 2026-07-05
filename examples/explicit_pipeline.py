from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pyisetcam import (
    AssetStore,
    ip_compute,
    ip_create,
    ip_get,
    oi_compute,
    oi_create,
    oi_show_image,
    scene_create,
    scene_get,
    sensor_compute,
    sensor_create,
    sensor_get,
)


def _default_output_dir() -> Path:
    return Path("reports/tutorial/explicit_pipeline")


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

    scene = scene_create(asset_store=store)
    oi = oi_compute(oi_create(asset_store=store), scene, crop=True)
    sensor = sensor_compute(sensor_create(asset_store=store), oi)
    ip = ip_compute(ip_create(sensor=sensor, asset_store=store), sensor, asset_store=store)

    scene_rgb = np.asarray(scene_get(scene, "rgb", asset_store=store), dtype=float)
    oi_rgb = np.asarray(oi_show_image(oi, -1, 1.0, asset_store=store), dtype=float)
    sensor_volts = np.asarray(sensor_get(sensor, "volts"), dtype=float)
    ip_srgb = np.asarray(ip_get(ip, "srgb"), dtype=float)

    scene_path = destination / "scene_rgb.png"
    oi_path = destination / "oi_rgb.png"
    sensor_path = destination / "sensor_volts.png"
    ip_path = destination / "ip_srgb.png"

    iio.imwrite(scene_path, _to_uint8(scene_rgb))
    iio.imwrite(oi_path, _to_uint8(oi_rgb))
    iio.imwrite(ip_path, _to_uint8(ip_srgb))

    fig, axis = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    image = axis.imshow(sensor_volts, cmap="viridis", interpolation="nearest")
    axis.set_title("Sensor volts")
    axis.set_xticks([])
    axis.set_yticks([])
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(sensor_path, dpi=160)
    plt.close(fig)

    summary = {
        "scene_rgb_path": scene_path,
        "oi_rgb_path": oi_path,
        "sensor_volts_path": sensor_path,
        "ip_srgb_path": ip_path,
        "wave_length_count": int(np.asarray(scene_get(scene, "nwave")).reshape(-1)[0]),
        "oi_shape": tuple(int(item) for item in np.asarray(oi.data["photons"]).shape),
        "sensor_size": tuple(int(item) for item in np.asarray(sensor_get(sensor, "size")).reshape(-1)),
        "result_shape": tuple(int(item) for item in np.asarray(ip_srgb).shape),
    }

    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    main()
