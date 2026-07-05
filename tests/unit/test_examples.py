from __future__ import annotations

import imageio.v3 as iio
import numpy as np

from examples import end_to_end, explicit_pipeline, quality_metrics, scene_from_file


def _assert_nonempty_image(path) -> None:
    image = np.asarray(iio.imread(path))
    assert image.size > 0
    assert image.shape[0] > 0
    assert image.shape[1] > 0


def test_end_to_end_example_smoke(tmp_path, asset_store) -> None:
    output_path = end_to_end.main(output_dir=tmp_path / "end_to_end", asset_store=asset_store)

    assert output_path.exists()
    _assert_nonempty_image(output_path)


def test_scene_from_file_example_smoke(tmp_path, asset_store) -> None:
    outputs = scene_from_file.main(output_dir=tmp_path / "scene_from_file", asset_store=asset_store)

    assert outputs["input"].exists()
    assert outputs["result"].exists()
    _assert_nonempty_image(outputs["input"])
    _assert_nonempty_image(outputs["result"])


def test_explicit_pipeline_example_smoke(tmp_path, asset_store) -> None:
    outputs = explicit_pipeline.main(output_dir=tmp_path / "explicit_pipeline", asset_store=asset_store)

    for key in ("scene_rgb_path", "oi_rgb_path", "sensor_volts_path", "ip_srgb_path"):
        assert outputs[key].exists()
        _assert_nonempty_image(outputs[key])

    assert outputs["wave_length_count"] > 0
    assert all(int(value) > 0 for value in outputs["oi_shape"])
    assert all(int(value) > 0 for value in outputs["sensor_size"])
    assert all(int(value) > 0 for value in outputs["result_shape"])


def test_quality_metrics_example_smoke(tmp_path, asset_store) -> None:
    outputs = quality_metrics.main(output_dir=tmp_path / "quality_metrics", asset_store=asset_store)

    assert outputs["figure_path"].exists()
    assert outputs["reference_path"].exists()
    assert outputs["degraded_path"].exists()
    _assert_nonempty_image(outputs["figure_path"])
    _assert_nonempty_image(outputs["reference_path"])
    _assert_nonempty_image(outputs["degraded_path"])

    assert np.isfinite(outputs["camera_acutance"])
    for value in outputs["comparison_metrics"].values():
        assert np.isfinite(value)
