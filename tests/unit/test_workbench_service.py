from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBENCH_ROOT = REPO_ROOT / "camerae2e-workbench"
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from backend.app.service import WorkbenchService  # noqa: E402

from pyisetcam import TaskBoundingBox  # noqa: E402


class _FixtureDetector:
    def detect(self, image, config=None):
        rows, cols = image.shape[:2]
        return [
            TaskBoundingBox(
                (0.42 * cols, 0.55 * rows, 0.66 * cols, 0.82 * rows),
                label="Car",
                score=0.95,
            ),
            TaskBoundingBox(
                (0.18 * cols, 0.50 * rows, 0.31 * cols, 0.66 * rows),
                label="Car",
                score=0.9,
            ),
            TaskBoundingBox(
                (0.72 * cols, 0.49 * rows, 0.77 * cols, 0.73 * rows),
                label="Pedestrian",
                score=0.88,
            ),
        ]


def test_workbench_simulate_allows_ocl_off_camera_preset() -> None:
    service = WorkbenchService()
    settings = service.default_settings()
    settings.update(
        {
            "cameraPreset": "narrow_fov_adas_demo",
            "fovDeg": 61.2,
            "pixelSizeUm": 2.8,
            "fNumber": 2.4,
            "oclMode": "off",
        }
    )

    result = service.simulate({"settings": settings})

    assert result["preview_png"].startswith("data:image/png;base64,")
    assert any(
        metric["id"] == "rgb_mean" and metric["value"] is not None
        for metric in result["metrics"]
    )


def test_workbench_default_optimization_includes_ccm_method_axis() -> None:
    service = WorkbenchService()
    settings = service.default_settings()

    scenario = service._scenario_from_settings(settings)
    axes = service._optimization_axes(settings)

    assert settings["sensorConversionMethod"] == "mcc_optimized"
    assert scenario["parameters"]["ip.sensor_conversion_method"] == "mcc_optimized"
    assert axes["ip.sensor_conversion_method"] == ["mcc_optimized", "esser_optimized"]


def test_workbench_perception_optimization_requires_yolo_model(monkeypatch) -> None:
    monkeypatch.delenv("CAMERAE2E_YOLO_MODEL", raising=False)
    monkeypatch.setenv("CAMERAE2E_DISABLE_YOLO_AUTODISCOVER", "1")
    service = WorkbenchService()
    settings = service.default_settings()
    settings["yoloModelPath"] = ""

    result = service.optimize({"settings": settings, "maxCandidates": 1})

    assert result["ok"] is False
    assert result["target_status"] == "detector_not_configured"
    assert result["top_cases"] == []
    assert "proxy score" in result["truth_boundary"].lower()


def test_workbench_perception_optimization_requires_labels(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "kitti-yolo.pt"
    model_path.write_bytes(b"fixture")
    monkeypatch.setenv("CAMERAE2E_YOLO_MODEL", str(model_path))
    service = WorkbenchService()
    settings = service.default_settings()
    settings.update({"yoloModelPath": str(model_path), "labelSource": "none"})

    result = service.optimize({"settings": settings, "maxCandidates": 1})

    assert result["ok"] is False
    assert result["target_status"] == "labels_missing"
    assert result["best_case"] is None


def test_workbench_perception_optimization_scores_fake_detector(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "kitti-yolo.pt"
    model_path.write_bytes(b"fixture")
    monkeypatch.setenv("CAMERAE2E_YOLO_MODEL", str(model_path))
    service = WorkbenchService()
    monkeypatch.setattr(
        service,
        "_perception_readiness",
        lambda settings: {"ok": True, "status": "available", "detail": "", "required_assets": []},
    )
    monkeypatch.setattr(service, "_make_yolo_detector", lambda settings: _FixtureDetector())
    settings = service.default_settings()
    settings.update(
        {
            "yoloModelPath": str(model_path),
            "labelSource": "synthetic_kitti",
            "kittiImagePath": "",
            "kittiLabelPath": "",
            "maxCandidates": 1,
            "robustnessCases": 1,
            "fdtdEnabled": False,
            "tcadEnabled": False,
        }
    )

    result = service.optimize({"settings": settings, "maxCandidates": 1})

    assert result["ok"] is True
    assert result["target_profile"] == "adas_yolo_perception"
    assert result["best_case"] is not None
    assert result["best_case"]["target_score"] > 0.8
    assert result["best_case"]["objective_values"]["perception.map50_95"] > 0.8
    assert result["best_case"]["objective_values"]["perception.recall50"] == 1.0


def test_workbench_scene_preview_accepts_yolo_label_file(tmp_path) -> None:
    image_path = tmp_path / "000001.png"
    label_path = tmp_path / "000001.txt"
    import base64
    import io

    import imageio.v3 as iio
    import numpy as np

    image = np.zeros((48, 96, 3), dtype=np.uint8)
    image[18:38, 30:60] = np.array([120, 140, 160], dtype=np.uint8)
    iio.imwrite(image_path, image)
    label_path.write_text("0 0.46875 0.583333 0.3125 0.416667\n", encoding="utf-8")

    service = WorkbenchService()
    settings = service.default_settings()
    settings.update(
        {
            "kittiImagePath": str(image_path),
            "kittiLabelPath": str(label_path),
            "labelSource": "local_kitti_yolo",
        }
    )

    result = service.scene_preview({"settings": settings})

    assert result["source"] == "kitti_file"
    assert result["object_count"] == 1
    assert result["scene_preview_png"].startswith("data:image/png;base64,")
    encoded = result["scene_preview_png"].split(",", 1)[1]
    preview = iio.imread(io.BytesIO(base64.b64decode(encoded)))
    border_pixel = np.asarray(preview[18, 30, :3], dtype=int)
    expected_original_preview = np.array([191, 223, 255])
    removed_overlay_color = np.array([255, 210, 64])
    assert np.linalg.norm(border_pixel - expected_original_preview) < 3
    assert np.linalg.norm(border_pixel - removed_overlay_color) > 100


def test_workbench_simulate_uses_configured_kitti_scene(tmp_path) -> None:
    image_path = tmp_path / "000003.png"
    label_path = tmp_path / "000003.txt"
    import imageio.v3 as iio
    import numpy as np

    image = np.zeros((48, 96, 3), dtype=np.uint8)
    image[:, :48] = np.array([40, 80, 160], dtype=np.uint8)
    image[:, 48:] = np.array([180, 120, 40], dtype=np.uint8)
    iio.imwrite(image_path, image)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    service = WorkbenchService()
    settings = service.default_settings()
    settings.update(
        {
            "kittiImagePath": str(image_path),
            "kittiLabelPath": str(label_path),
            "labelSource": "local_kitti_yolo",
            "fdtdEnabled": False,
            "tcadEnabled": False,
            "hwIspEnabled": False,
        }
    )

    result = service.simulate({"settings": settings})

    assert result["scene_source"]["source"] == "kitti_file"
    assert result["scene_source"]["source_path"] == str(image_path)
    assert result["preview_png"].startswith("data:image/png;base64,")
