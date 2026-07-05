from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pyisetcam import AssetStore, camera_compute, camera_create, scene_create


def _default_output_dir() -> Path:
    return Path("reports/tutorial/end_to_end")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def main(output_dir: Path | None = None, asset_store: AssetStore | None = None) -> Path:
    store = asset_store or AssetStore.default()
    scene = scene_create(asset_store=store)
    camera = camera_compute(camera_create(asset_store=store), scene, asset_store=store)
    image = np.asarray(camera.fields["ip"].data["srgb"], dtype=float)
    destination = (output_dir if output_dir is not None else _default_output_dir()).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "end_to_end.png"
    iio.imwrite(output_path, _to_uint8(image))
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
