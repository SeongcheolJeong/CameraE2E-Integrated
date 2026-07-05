from __future__ import annotations

from pathlib import Path

import numpy as np

import pyisetcam.task_perception as task_module
from pyisetcam import (
    TaskBoundingBox,
    TaskModelConfig,
    TaskSegmentationMask,
    bboxIoU,
    detectionMetrics,
    meanAveragePrecision,
    segmentationMetrics,
    taskModelConfigFromProfile,
    taskModelFromConfig,
    taskModelProfileNames,
    taskPerceptionSweep,
)
from tools import render_task_perception_report


def _mask(shape=(40, 40), box=(10, 10, 25, 25)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _sample_image() -> tuple[np.ndarray, list[TaskBoundingBox], list[TaskSegmentationMask]]:
    image = np.zeros((48, 64, 3), dtype=float)
    image[..., :] = 0.10
    image[10:26, 8:26, 0] = 0.90
    image[10:26, 8:26, 1:] = 0.05
    image[18:38, 38:56, 1] = 0.82
    image[18:38, 38:56, [0, 2]] = 0.05
    red_mask = np.zeros(image.shape[:2], dtype=bool)
    red_mask[10:26, 8:26] = True
    green_mask = np.zeros(image.shape[:2], dtype=bool)
    green_mask[18:38, 38:56] = True
    boxes = [
        TaskBoundingBox((8, 10, 26, 26), label="red", score=1.0),
        TaskBoundingBox((38, 18, 56, 38), label="green", score=1.0),
    ]
    masks = [TaskSegmentationMask(red_mask, label="red"), TaskSegmentationMask(green_mask, label="green")]
    return image, boxes, masks


def _detector(image: np.ndarray) -> list[TaskBoundingBox]:
    boxes = []
    for label_name, channel in (("red", 0), ("green", 1)):
        other = np.max(np.delete(image, channel, axis=2), axis=2)
        mask = (image[..., channel] > 0.25) & ((image[..., channel] - other) > 0.08)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            continue
        score = float(np.clip(np.mean(image[..., channel][mask] - other[mask]) * 1.7, 0.0, 1.0))
        boxes.append(TaskBoundingBox((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1), label=label_name, score=score))
    return boxes


def _segmenter(image: np.ndarray) -> list[TaskSegmentationMask]:
    masks = []
    for label_name, channel in (("red", 0), ("green", 1)):
        other = np.max(np.delete(image, channel, axis=2), axis=2)
        mask = (image[..., channel] > 0.25) & ((image[..., channel] - other) > 0.08)
        if np.count_nonzero(mask) == 0:
            continue
        score = float(np.clip(np.mean(image[..., channel][mask] - other[mask]) * 1.7, 0.0, 1.0))
        masks.append(TaskSegmentationMask(mask, label=label_name, score=score))
    return masks


def test_task_perception_module_aliases() -> None:
    assert task_module.taskPerceptionConfig is task_module.task_perception_config
    assert task_module.taskModelConfig is task_module.task_model_config
    assert task_module.taskModelFromConfig is task_module.task_model_from_config
    assert task_module.bboxIoU is task_module.bbox_iou
    assert task_module.detectionMetrics is task_module.detection_metrics
    assert task_module.taskPerceptionSweep is task_module.task_perception_sweep


def test_task_perception_root_exports_aliases() -> None:
    assert bboxIoU is task_module.bbox_iou
    assert taskModelConfigFromProfile is task_module.task_model_config_from_profile
    assert taskModelFromConfig is task_module.task_model_from_config
    assert detectionMetrics is task_module.detection_metrics
    assert meanAveragePrecision is task_module.mean_average_precision
    assert segmentationMetrics is task_module.segmentation_metrics
    assert taskPerceptionSweep is task_module.task_perception_sweep


def test_task_model_profiles_are_available() -> None:
    names = taskModelProfileNames()

    assert "custom_callable" in names
    assert "ultralytics_yolo11n_detection" in names
    assert "ultralytics_yolo11n_segmentation" in names
    assert "ultralytics_yolo11n_classification" in names
    assert "ultralytics_yolo11n_pose" in names
    assert "ultralytics_yolo11n_obb" in names
    assert "ultralytics_yolo11n_bytetrack" in names
    assert "torchvision_fasterrcnn_resnet50_fpn_v2_coco" in names
    assert "transformers_detr_resnet50_coco" in names
    assert "sam_vit_b_automatic" in names


def test_task_model_config_from_profile_can_be_overridden() -> None:
    config = taskModelConfigFromProfile("ultralytics_yolo11n_detection", device="cuda:0", score_threshold=0.4)

    assert isinstance(config, TaskModelConfig)
    assert config.backend == "ultralytics_yolo"
    assert config.device == "cuda:0"
    assert config.score_threshold == 0.4


def test_callable_model_adapter_runs_detector_and_segmenter() -> None:
    image, boxes, masks = _sample_image()
    adapter = taskModelFromConfig(
        {"name": "unit_callable", "backend": "callable", "task": "detection_segmentation", "score_threshold": 0.1},
        detector=_detector,
        segmenter=_segmenter,
    )

    detections = adapter.detect(image)
    segmentations = adapter.segment(image)

    assert len(detections) == len(boxes)
    assert len(segmentations) == len(masks)


def test_callable_model_adapter_supports_extended_tasks() -> None:
    image, _, _ = _sample_image()
    keypoints = np.array([[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]], dtype=float)
    adapter = taskModelFromConfig(
        {"name": "extended_callable", "backend": "callable", "task": "detection_segmentation", "score_threshold": 0.1},
        detector=_detector,
        segmenter=_segmenter,
    )
    adapter.classifier = lambda _: [{"label": "target", "score": 0.95, "class_id": 2}]
    adapter.pose_estimator = lambda _: [{"keypoints_xyc": keypoints, "score": 0.91}]
    adapter.oriented_detector = lambda _: [
        {
            "xywhr": [10, 10, 8, 4, 0.2],
            "corners_xy": [[6, 8], [14, 8], [14, 12], [6, 12]],
            "label": "obb",
            "score": 0.88,
        }
    ]
    adapter.tracker = lambda _: [{"track_id": 7, "box": {"xyxy": [8, 10, 26, 26], "label": "red", "score": 0.93}}]

    assert adapter.classify(image)[0].label == "target"
    assert adapter.estimate_pose(image)[0].keypoints_xyc.shape == (2, 3)
    assert adapter.oriented_detect(image)[0].label == "obb"
    assert adapter.track(image)[0].track_id == 7


def test_non_callable_backend_reports_missing_optional_dependency(monkeypatch) -> None:
    def fake_optional_import(module_name: str, install_hint: str):
        raise ImportError(f"Task perception backend requires optional dependency {module_name!r}. Install it with {install_hint}.")

    monkeypatch.setattr(task_module, "_optional_import", fake_optional_import)
    config = task_module.task_model_config_from_profile("ultralytics_yolo11n_detection", model_id="not-loaded.pt")
    try:
        task_module.task_model_from_config(config)
    except ImportError as exc:
        assert "optional dependency" in str(exc)
    else:
        raise AssertionError("Expected missing optional dependency error.")


def test_bbox_iou_matches_expected_overlap() -> None:
    first = TaskBoundingBox((0, 0, 10, 10))
    second = TaskBoundingBox((5, 5, 15, 15))

    assert task_module.bbox_iou(first, second) == 25.0 / 175.0


def test_detection_metrics_and_map_for_matched_boxes() -> None:
    gt = [TaskBoundingBox((0, 0, 10, 10), label="target")]
    pred = [TaskBoundingBox((1, 1, 9, 9), label="target", score=0.9)]

    metrics = task_module.detection_metrics(pred, gt)
    map_metrics = task_module.mean_average_precision(pred, gt)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["recall"] == 1.0
    assert map_metrics["ap50"] == 1.0


def test_segmentation_metrics_include_boundary_f1() -> None:
    gt = [TaskSegmentationMask(_mask(), label="target")]
    pred = [TaskSegmentationMask(_mask(box=(11, 10, 26, 25)), label="target")]

    metrics = task_module.segmentation_metrics(pred, gt)

    assert metrics["true_positive"] == 1
    assert 0.0 < metrics["mean_iou"] < 1.0
    assert 0.0 <= metrics["mean_boundary_f1"] <= 1.0


def test_annotations_to_bboxes_supports_xywh_and_yolo() -> None:
    xywh = task_module.annotations_to_bboxes([{"bbox": [2, 3, 10, 12], "label": "a"}], bbox_format="xywh")
    yolo = task_module.annotations_to_bboxes([{"bbox": [0.5, 0.5, 0.5, 0.5], "label": "b"}], bbox_format="yolo", image_size=(100, 200))

    assert xywh[0].xyxy == (2.0, 3.0, 12.0, 15.0)
    assert yolo[0].xyxy == (50.0, 25.0, 150.0, 75.0)


def test_task_perception_sweep_reports_degradation() -> None:
    image, boxes, masks = _sample_image()
    sweep = task_module.task_perception_sweep(
        image,
        _detector,
        boxes,
        segmenter=_segmenter,
        ground_truth_masks=masks,
        perturbations=(
            {"name": "baseline", "kind": "identity", "amount": 0.0},
            {"name": "low_light", "kind": "brightness", "amount": 0.20},
        ),
    )

    assert len(sweep["cases"]) == 2
    assert sweep["cases"][0]["detection_metrics"]["recall"] == 1.0
    assert sweep["cases"][1]["detection_metrics"]["recall"] < 1.0
    assert sweep["degradation"]["cases"][1]["recall_drop"] > 0.0


def test_task_score_by_stage_runs_detector_and_segmenter() -> None:
    image, boxes, masks = _sample_image()
    stage_result = task_module.task_score_by_stage(
        {"clean": image},
        _detector,
        boxes,
        segmenter=_segmenter,
        ground_truth_masks=masks,
    )

    assert stage_result["stages"]["clean"]["detection_metrics"]["recall"] == 1.0
    assert stage_result["stages"]["clean"]["segmentation_metrics"]["mean_iou"] == 1.0


def test_render_overlays_return_rgb_images() -> None:
    image, boxes, masks = _sample_image()

    detection_overlay = task_module.render_detection_overlay(image, boxes, boxes)
    segmentation_overlay = task_module.render_segmentation_overlay(image, masks)

    assert detection_overlay.shape == image.shape
    assert segmentation_overlay.shape == image.shape
    assert np.max(detection_overlay) <= 1.0
    assert np.max(segmentation_overlay) <= 1.0


def test_render_task_perception_report_creates_artifacts(tmp_path) -> None:
    result = render_task_perception_report.build_report(tmp_path)

    assert result["html"].exists()
    assert result["summary"].exists()
    for path in result["figures"].values():
        assert Path(path).exists()
