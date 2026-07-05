from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pyisetcam import AssetStore, camera_compute, camera_create, scene_from_file


def _default_output_dir() -> Path:
    return Path("reports/tutorial/scene_from_file")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def main(output_dir: Path | None = None, asset_store: AssetStore | None = None) -> dict[str, Path]:
    store = asset_store or AssetStore.default()
    rows, cols = 96, 144
    rr, cc = np.indices((rows, cols))
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.round(255.0 * cc / max(cols - 1, 1)).astype(np.uint8)
    rgb[:, :, 1] = np.round(255.0 * rr / max(rows - 1, 1)).astype(np.uint8)
    rgb[:, :, 2] = np.round(255.0 * (1.0 - cc / max(cols - 1, 1))).astype(np.uint8)

    scene = scene_from_file(rgb, "rgb", 100.0, "lcdExample.mat", asset_store=store)
    camera = camera_compute(camera_create(asset_store=store), scene, asset_store=store)

    destination = (output_dir if output_dir is not None else _default_output_dir()).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / "input.png"
    result_path = destination / "result.png"
    iio.imwrite(input_path, rgb)
    iio.imwrite(result_path, _to_uint8(np.asarray(camera.fields["ip"].data["srgb"], dtype=float)))
    print(input_path)
    print(result_path)
    return {"input": input_path, "result": result_path}


if __name__ == "__main__":
    main()
