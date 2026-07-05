"""Task-level perception helpers for detection and segmentation evaluation.

This module is intentionally model-agnostic.  It evaluates outputs from
callable detectors/segmenters without making YOLO, SAM, Detectron, or Torch a
core dependency of pyisetcam.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter


DetectorCallable = Callable[[NDArray[np.float64]], Any]
SegmenterCallable = Callable[[NDArray[np.float64]], Any]
ClassifierCallable = Callable[[NDArray[np.float64]], Any]
PoseCallable = Callable[[NDArray[np.float64]], Any]
OrientedDetectorCallable = Callable[[NDArray[np.float64]], Any]
TrackerCallable = Callable[[NDArray[np.float64]], Any]


@dataclass(frozen=True)
class TaskModelConfig:
    """Configuration for an optional task-perception model backend."""

    name: str = "callable"
    backend: str = "callable"
    task: str = "detection"
    model_id: str | None = None
    checkpoint_path: str | None = None
    device: str = "cpu"
    score_threshold: float = 0.25
    labels: Mapping[int | str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TaskModelAdapter:
    """Runtime adapter that exposes normalized detection/segmentation callables."""

    config: TaskModelConfig
    detector: DetectorCallable | None = None
    segmenter: SegmenterCallable | None = None
    classifier: ClassifierCallable | None = None
    pose_estimator: PoseCallable | None = None
    oriented_detector: OrientedDetectorCallable | None = None
    tracker: TrackerCallable | None = None

    def detect(self, image: Any, config: TaskPerceptionConfig | None = None) -> list[TaskBoundingBox]:
        if self.detector is None:
            raise ValueError(f"Model profile {self.config.name!r} does not provide detection.")
        return run_task_detector(self.detector, image, config or task_perception_config(score_threshold=self.config.score_threshold))

    def segment(self, image: Any, config: TaskPerceptionConfig | None = None) -> list[TaskSegmentationMask]:
        if self.segmenter is None:
            raise ValueError(f"Model profile {self.config.name!r} does not provide segmentation.")
        return run_task_segmenter(self.segmenter, image, config or task_perception_config(score_threshold=self.config.score_threshold))

    def classify(self, image: Any) -> list["TaskClassificationResult"]:
        if self.classifier is None:
            raise ValueError(f"Model profile {self.config.name!r} does not provide classification.")
        return _prepare_classifications(self.classifier(_rgb_image(image)), self.config)

    def estimate_pose(self, image: Any) -> list["TaskPoseResult"]:
        if self.pose_estimator is None:
            raise ValueError(f"Model profile {self.config.name!r} does not provide pose estimation.")
        return _prepare_poses(self.pose_estimator(_rgb_image(image)), self.config)

    def oriented_detect(self, image: Any) -> list["TaskOrientedBoundingBox"]:
        if self.oriented_detector is None:
            raise ValueError(f"Model profile {self.config.name!r} does not provide oriented detection.")
        return _prepare_oriented_boxes(self.oriented_detector(_rgb_image(image)), self.config)

    def track(self, image: Any) -> list["TaskTrackResult"]:
        if self.tracker is None:
            raise ValueError(f"Model profile {self.config.name!r} does not provide tracking.")
        return _prepare_tracks(self.tracker(_rgb_image(image)), self.config)


@dataclass(frozen=True)
class TaskPerceptionConfig:
    """Configuration shared by task-level perception metrics."""

    iou_threshold: float = 0.50
    score_threshold: float = 0.0
    map_iou_thresholds: tuple[float, ...] = (0.50, 0.75)
    mask_threshold: float = 0.50
    boundary_tolerance_px: int = 1
    max_detections: int | None = None
    label_agnostic: bool = False


@dataclass(frozen=True)
class TaskBoundingBox:
    """Detection box in continuous ``xyxy`` coordinates."""

    xyxy: tuple[float, float, float, float]
    label: str = "object"
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = tuple(float(value) for value in self.xyxy)
        if len(coords) != 4:
            raise ValueError("xyxy must contain four coordinates.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("xyxy coordinates must be finite.")
        if coords[2] < coords[0] or coords[3] < coords[1]:
            raise ValueError("xyxy must be ordered as x1, y1, x2, y2.")
        if not np.isfinite(float(self.score)):
            raise ValueError("score must be finite.")
        object.__setattr__(self, "xyxy", coords)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "score", float(self.score))

    @property
    def area(self) -> float:
        return float(max(self.xyxy[2] - self.xyxy[0], 0.0) * max(self.xyxy[3] - self.xyxy[1], 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyxy": list(self.xyxy),
            "label": self.label,
            "score": float(self.score),
            "area": self.area,
            "metadata": _json_ready(self.metadata),
        }


@dataclass
class TaskSegmentationMask:
    """Segmentation mask with label and confidence score."""

    mask: Any
    label: str = "object"
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask_array = np.asarray(self.mask)
        if mask_array.ndim != 2:
            raise ValueError("mask must be a 2-D array.")
        if mask_array.size == 0:
            raise ValueError("mask must not be empty.")
        self.mask = np.asarray(mask_array > 0, dtype=bool)
        self.label = str(self.label)
        self.score = float(self.score)
        if not np.isfinite(self.score):
            raise ValueError("score must be finite.")

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def bbox(self) -> TaskBoundingBox:
        rows, cols = np.nonzero(self.mask)
        if rows.size == 0:
            return TaskBoundingBox((0.0, 0.0, 0.0, 0.0), label=self.label, score=self.score)
        return TaskBoundingBox(
            (float(np.min(cols)), float(np.min(rows)), float(np.max(cols) + 1), float(np.max(rows) + 1)),
            label=self.label,
            score=self.score,
        )

    def to_dict(self, *, include_mask: bool = False) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "score": float(self.score),
            "shape": list(self.mask.shape),
            "area": self.area,
            "bbox": self.bbox.to_dict(),
            "metadata": _json_ready(self.metadata),
        }
        if include_mask:
            payload["mask"] = self.mask.astype(int).tolist()
        return payload


@dataclass(frozen=True)
class TaskClassificationResult:
    """Classification result with label, class id, and confidence."""

    label: str
    score: float
    class_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": float(self.score),
            "class_id": self.class_id,
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class TaskPoseResult:
    """Pose-estimation result with keypoints in ``x, y, confidence`` form."""

    keypoints_xyc: Any
    bbox: TaskBoundingBox | None = None
    label: str = "person"
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_xyc, dtype=float)
        if keypoints.ndim != 2 or keypoints.shape[1] not in {2, 3}:
            raise ValueError("keypoints_xyc must have shape (N, 2) or (N, 3).")
        if keypoints.shape[1] == 2:
            keypoints = np.column_stack([keypoints, np.ones(keypoints.shape[0], dtype=float)])
        object.__setattr__(self, "keypoints_xyc", keypoints)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "score", float(self.score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": float(self.score),
            "bbox": None if self.bbox is None else self.bbox.to_dict(),
            "keypoints_xyc": np.asarray(self.keypoints_xyc, dtype=float).tolist(),
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class TaskOrientedBoundingBox:
    """Oriented detection box represented by center-size-rotation and corners."""

    xywhr: tuple[float, float, float, float, float]
    corners_xy: Any
    label: str = "object"
    score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xywhr = tuple(float(value) for value in self.xywhr)
        corners = np.asarray(self.corners_xy, dtype=float)
        if len(xywhr) != 5:
            raise ValueError("xywhr must contain cx, cy, width, height, rotation.")
        if corners.shape != (4, 2):
            raise ValueError("corners_xy must have shape (4, 2).")
        object.__setattr__(self, "xywhr", xywhr)
        object.__setattr__(self, "corners_xy", corners)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "score", float(self.score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "xywhr": list(self.xywhr),
            "corners_xy": np.asarray(self.corners_xy, dtype=float).tolist(),
            "label": self.label,
            "score": float(self.score),
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class TaskTrackResult:
    """Tracking result with track id and current detection box."""

    track_id: int
    box: TaskBoundingBox
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "box": self.box.to_dict(),
            "metadata": _json_ready(self.metadata),
        }


def task_perception_config(**overrides: Any) -> TaskPerceptionConfig:
    """Create a validated task-perception configuration."""

    config = TaskPerceptionConfig(**overrides)
    if not 0.0 <= config.iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0, 1].")
    if config.score_threshold < 0.0:
        raise ValueError("score_threshold must be nonnegative.")
    if not 0.0 <= config.mask_threshold <= 1.0:
        raise ValueError("mask_threshold must lie in [0, 1].")
    if config.boundary_tolerance_px < 0:
        raise ValueError("boundary_tolerance_px must be nonnegative.")
    if config.max_detections is not None and config.max_detections <= 0:
        raise ValueError("max_detections must be positive when provided.")
    thresholds = tuple(float(value) for value in config.map_iou_thresholds)
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("map_iou_thresholds must contain values in [0, 1].")
    return TaskPerceptionConfig(
        iou_threshold=float(config.iou_threshold),
        score_threshold=float(config.score_threshold),
        map_iou_thresholds=thresholds,
        mask_threshold=float(config.mask_threshold),
        boundary_tolerance_px=int(config.boundary_tolerance_px),
        max_detections=config.max_detections,
        label_agnostic=bool(config.label_agnostic),
    )


def task_model_config(source: Mapping[str, Any] | TaskModelConfig | None = None, **overrides: Any) -> TaskModelConfig:
    """Create a model-backend config from a mapping/profile plus overrides."""

    payload: dict[str, Any]
    if source is None:
        payload = {}
    elif isinstance(source, TaskModelConfig):
        payload = asdict(source)
    else:
        payload = dict(source)
    payload.update(overrides)

    labels = payload.get("labels", {})
    normalized_labels = {int(key) if str(key).lstrip("-").isdigit() else str(key): str(value) for key, value in dict(labels).items()}
    options = dict(payload.get("options", {}))
    config = TaskModelConfig(
        name=str(payload.get("name", payload.get("model_id", payload.get("backend", "callable")))),
        backend=str(payload.get("backend", "callable")).lower().replace("-", "_"),
        task=str(payload.get("task", "detection")).lower().replace("-", "_"),
        model_id=None if payload.get("model_id") is None else str(payload.get("model_id")),
        checkpoint_path=None if payload.get("checkpoint_path") in {None, ""} else str(payload.get("checkpoint_path")),
        device=str(payload.get("device", "cpu")),
        score_threshold=float(payload.get("score_threshold", 0.25)),
        labels=normalized_labels,
        options=options,
    )
    if config.score_threshold < 0.0:
        raise ValueError("score_threshold must be nonnegative.")
    if config.backend not in _supported_model_backends():
        raise ValueError(f"Unsupported task perception model backend {config.backend!r}.")
    if config.task not in {"detection", "segmentation", "detection_segmentation", "classification", "pose", "oriented_detection", "tracking"}:
        raise ValueError("task must be detection, segmentation, detection_segmentation, classification, pose, oriented_detection, or tracking.")
    return config


def task_model_profile_names() -> list[str]:
    """Return bundled task-perception model profile names."""

    return sorted(_task_model_profiles().keys())


def task_model_profile(name: str) -> dict[str, Any]:
    """Return one bundled task-perception model profile."""

    profiles = _task_model_profiles()
    key = str(name)
    if key not in profiles:
        raise KeyError(f"Unknown task perception model profile {name!r}.")
    return json.loads(json.dumps(profiles[key]))


def task_model_config_from_profile(name: str, **overrides: Any) -> TaskModelConfig:
    """Create a model config from a bundled profile."""

    return task_model_config(task_model_profile(name), **overrides)


def task_model_from_config(
    config: Mapping[str, Any] | TaskModelConfig | None = None,
    *,
    detector: DetectorCallable | None = None,
    segmenter: SegmenterCallable | None = None,
) -> TaskModelAdapter:
    """Instantiate a model adapter from config without adding hard dependencies."""

    cfg = task_model_config(config)
    if cfg.backend == "callable":
        return TaskModelAdapter(config=cfg, detector=detector, segmenter=segmenter)
    if cfg.backend == "ultralytics_yolo":
        return _ultralytics_yolo_adapter(cfg)
    if cfg.backend == "torchvision_detection":
        return _torchvision_detection_adapter(cfg)
    if cfg.backend == "transformers_object_detection":
        return _transformers_object_detection_adapter(cfg)
    if cfg.backend == "transformers_segmentation":
        return _transformers_segmentation_adapter(cfg)
    if cfg.backend == "sam_automatic":
        return _sam_automatic_adapter(cfg)
    raise ValueError(f"Unsupported task perception model backend {cfg.backend!r}.")


def task_detector_from_config(config: Mapping[str, Any] | TaskModelConfig, **kwargs: Any) -> DetectorCallable:
    """Create a detector callable from a model config."""

    adapter = task_model_from_config(config, **kwargs)
    if adapter.detector is None:
        raise ValueError(f"Model config {adapter.config.name!r} does not provide detection.")
    return adapter.detector


def task_segmenter_from_config(config: Mapping[str, Any] | TaskModelConfig, **kwargs: Any) -> SegmenterCallable:
    """Create a segmenter callable from a model config."""

    adapter = task_model_from_config(config, **kwargs)
    if adapter.segmenter is None:
        raise ValueError(f"Model config {adapter.config.name!r} does not provide segmentation.")
    return adapter.segmenter


def bbox_iou(box_a: Any, box_b: Any) -> float:
    """Compute intersection-over-union for two detection boxes."""

    first = _as_box(box_a)
    second = _as_box(box_b)
    x1 = max(first.xyxy[0], second.xyxy[0])
    y1 = max(first.xyxy[1], second.xyxy[1])
    x2 = min(first.xyxy[2], second.xyxy[2])
    y2 = min(first.xyxy[3], second.xyxy[3])
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = first.area + second.area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def mask_iou(mask_a: Any, mask_b: Any) -> float:
    """Compute intersection-over-union for two binary masks."""

    first = _as_bool_mask(mask_a)
    second = _as_bool_mask(mask_b)
    if first.shape != second.shape:
        raise ValueError("mask_a and mask_b must have the same shape.")
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def boundary_f1_score(pred_mask: Any, gt_mask: Any, *, tolerance_px: int = 1) -> float:
    """Compute boundary F1 with a pixel tolerance."""

    predicted = _mask_boundary(_as_bool_mask(pred_mask))
    truth = _mask_boundary(_as_bool_mask(gt_mask))
    if predicted.shape != truth.shape:
        raise ValueError("pred_mask and gt_mask must have the same shape.")
    if not np.any(predicted) and not np.any(truth):
        return 1.0
    if not np.any(predicted) or not np.any(truth):
        return 0.0

    iterations = max(int(tolerance_px), 0)
    truth_dilated = binary_dilation(truth, iterations=iterations) if iterations else truth
    pred_dilated = binary_dilation(predicted, iterations=iterations) if iterations else predicted
    precision = float(np.count_nonzero(predicted & truth_dilated) / max(np.count_nonzero(predicted), 1))
    recall = float(np.count_nonzero(truth & pred_dilated) / max(np.count_nonzero(truth), 1))
    return _f1(precision, recall)


def detection_metrics(
    predictions: Iterable[Any],
    ground_truth: Iterable[Any],
    config: TaskPerceptionConfig | None = None,
    *,
    iou_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute detection precision/recall/F1/AP for predicted boxes."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    threshold = cfg.iou_threshold if iou_threshold is None else float(iou_threshold)
    pred_boxes = _prepare_predictions(predictions, cfg)
    gt_boxes = [_as_box(item) for item in ground_truth]
    matches, unmatched_pred, unmatched_gt = _match_boxes(pred_boxes, gt_boxes, threshold, cfg)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    labels = _labels(pred_boxes, gt_boxes, cfg)
    per_label = {
        label: _detection_metrics_for_label(pred_boxes, gt_boxes, threshold, cfg, label)
        for label in labels
        if not cfg.label_agnostic
    }
    return {
        "iou_threshold": float(threshold),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "mean_matched_iou": float(np.mean([match["iou"] for match in matches], dtype=float)) if matches else 0.0,
        "average_precision": _average_precision(pred_boxes, gt_boxes, threshold, cfg),
        "matches": matches,
        "per_label": per_label,
    }


def mean_average_precision(
    predictions: Iterable[Any],
    ground_truth: Iterable[Any],
    config: TaskPerceptionConfig | None = None,
    *,
    iou_thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute mAP across labels and IoU thresholds."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    thresholds = tuple(float(value) for value in (cfg.map_iou_thresholds if iou_thresholds is None else iou_thresholds))
    pred_boxes = _prepare_predictions(predictions, cfg)
    gt_boxes = [_as_box(item) for item in ground_truth]
    labels = _labels(pred_boxes, gt_boxes, cfg)
    if cfg.label_agnostic:
        labels = ("object",)

    by_threshold: dict[str, Any] = {}
    threshold_values: list[float] = []
    for threshold in thresholds:
        aps = []
        by_label = {}
        for label in labels:
            label_preds = pred_boxes if cfg.label_agnostic else [box for box in pred_boxes if box.label == label]
            label_gts = gt_boxes if cfg.label_agnostic else [box for box in gt_boxes if box.label == label]
            ap = _average_precision(label_preds, label_gts, threshold, cfg)
            by_label[label] = float(ap)
            aps.append(float(ap))
        value = float(np.mean(aps, dtype=float)) if aps else 0.0
        by_threshold[f"ap{int(round(threshold * 100.0)):02d}"] = {"value": value, "per_label": by_label}
        threshold_values.append(value)

    payload = {
        "map": float(np.mean(threshold_values, dtype=float)) if threshold_values else 0.0,
        "thresholds": [float(value) for value in thresholds],
        "by_threshold": by_threshold,
    }
    for key, value in by_threshold.items():
        payload[key] = value["value"]
    return payload


def detection_recall_at_iou(
    predictions: Iterable[Any],
    ground_truth: Iterable[Any],
    iou_threshold: float,
    config: TaskPerceptionConfig | None = None,
) -> float:
    """Return detection recall at a specific IoU threshold."""

    return float(detection_metrics(predictions, ground_truth, config, iou_threshold=iou_threshold)["recall"])


def segmentation_metrics(
    predictions: Iterable[Any],
    ground_truth: Iterable[Any],
    config: TaskPerceptionConfig | None = None,
) -> dict[str, Any]:
    """Compute segmentation precision/recall/F1/mIoU/boundary-F1."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    pred_masks = _prepare_masks(predictions, cfg)
    gt_masks = [_as_segmentation_mask(item, cfg) for item in ground_truth]
    matches, unmatched_pred, unmatched_gt = _match_masks(pred_masks, gt_masks, cfg)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "iou_threshold": float(cfg.iou_threshold),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "mean_iou": float(np.mean([match["iou"] for match in matches], dtype=float)) if matches else 0.0,
        "mean_boundary_f1": float(np.mean([match["boundary_f1"] for match in matches], dtype=float)) if matches else 0.0,
        "matches": matches,
    }


def mean_iou(
    predictions: Iterable[Any],
    ground_truth: Iterable[Any],
    config: TaskPerceptionConfig | None = None,
) -> float:
    """Return matched-mask mean IoU."""

    return float(segmentation_metrics(predictions, ground_truth, config)["mean_iou"])


def run_task_detector(detector: DetectorCallable, image: Any, config: TaskPerceptionConfig | None = None, **kwargs: Any) -> list[TaskBoundingBox]:
    """Run a callable detector and normalize its output boxes."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    image_array = _rgb_image(image)
    raw = detector(image_array, **kwargs)
    return _prepare_predictions(raw, cfg)


def run_task_segmenter(
    segmenter: SegmenterCallable,
    image: Any,
    config: TaskPerceptionConfig | None = None,
    **kwargs: Any,
) -> list[TaskSegmentationMask]:
    """Run a callable segmenter and normalize its output masks."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    image_array = _rgb_image(image)
    raw = segmenter(image_array, **kwargs)
    return _prepare_masks(raw, cfg)


def task_score_by_stage(
    stage_images: Mapping[str, Any],
    detector: DetectorCallable | None = None,
    ground_truth_boxes: Iterable[Any] | None = None,
    *,
    segmenter: SegmenterCallable | None = None,
    ground_truth_masks: Iterable[Any] | None = None,
    config: TaskPerceptionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate detection/segmentation results for multiple pipeline stages."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    gt_boxes = list(ground_truth_boxes or [])
    gt_masks = list(ground_truth_masks or [])
    stages: dict[str, Any] = {}
    for stage_name, image in stage_images.items():
        stage_payload: dict[str, Any] = {"stage": str(stage_name)}
        if detector is not None:
            boxes = run_task_detector(detector, image, cfg)
            stage_payload["detections"] = [box.to_dict() for box in boxes]
            if gt_boxes:
                stage_payload["detection_metrics"] = detection_metrics(boxes, gt_boxes, cfg)
                stage_payload["map"] = mean_average_precision(boxes, gt_boxes, cfg)
        if segmenter is not None:
            masks = run_task_segmenter(segmenter, image, cfg)
            stage_payload["segmentations"] = [mask.to_dict() for mask in masks]
            if gt_masks:
                stage_payload["segmentation_metrics"] = segmentation_metrics(masks, gt_masks, cfg)
        stages[str(stage_name)] = stage_payload
    return {"config": asdict(cfg), "stages": stages}


def task_perception_sweep(
    image: Any,
    detector: DetectorCallable | None = None,
    ground_truth_boxes: Iterable[Any] | None = None,
    *,
    segmenter: SegmenterCallable | None = None,
    ground_truth_masks: Iterable[Any] | None = None,
    perturbations: Iterable[Mapping[str, Any]] | None = None,
    config: TaskPerceptionConfig | None = None,
) -> dict[str, Any]:
    """Run a deterministic task-perception robustness sweep."""

    cfg = task_perception_config() if config is None else task_perception_config(**asdict(config))
    base_image = _rgb_image(image)
    cases = list(_default_perturbations() if perturbations is None else perturbations)
    gt_boxes = list(ground_truth_boxes or [])
    gt_masks = list(ground_truth_masks or [])

    results = []
    for case in cases:
        case_name = str(case.get("name", case.get("kind", "case")))
        transformed = task_perception_perturb_image(base_image, case)
        payload: dict[str, Any] = {
            "name": case_name,
            "kind": str(case.get("kind", "identity")),
            "amount": _json_ready(case.get("amount", 0.0)),
        }
        if detector is not None:
            boxes = run_task_detector(detector, transformed, cfg)
            payload["detection_count"] = len(boxes)
            payload["mean_detection_score"] = float(np.mean([box.score for box in boxes], dtype=float)) if boxes else 0.0
            payload["detections"] = [box.to_dict() for box in boxes]
            if gt_boxes:
                payload["detection_metrics"] = detection_metrics(boxes, gt_boxes, cfg)
                payload["map"] = mean_average_precision(boxes, gt_boxes, cfg)
        if segmenter is not None:
            masks = run_task_segmenter(segmenter, transformed, cfg)
            payload["segmentation_count"] = len(masks)
            payload["segmentations"] = [mask.to_dict() for mask in masks]
            if gt_masks:
                payload["segmentation_metrics"] = segmentation_metrics(masks, gt_masks, cfg)
        results.append(payload)

    return {
        "config": asdict(cfg),
        "cases": results,
        "degradation": task_degradation_report(results),
    }


def task_degradation_report(case_results: Sequence[Mapping[str, Any]], *, baseline_name: str | None = None) -> dict[str, Any]:
    """Summarize metric drops from the baseline case in a sweep."""

    if not case_results:
        return {"baseline": None, "cases": []}
    baseline = next((case for case in case_results if baseline_name is None or case.get("name") == baseline_name), case_results[0])
    baseline_detection = dict(baseline.get("detection_metrics", {}))
    baseline_map = dict(baseline.get("map", {}))
    baseline_segmentation = dict(baseline.get("segmentation_metrics", {}))

    rows = []
    for case in case_results:
        det = dict(case.get("detection_metrics", {}))
        seg = dict(case.get("segmentation_metrics", {}))
        m_ap = dict(case.get("map", {}))
        rows.append(
            {
                "name": str(case.get("name", "")),
                "precision_drop": float(baseline_detection.get("precision", 0.0) - det.get("precision", 0.0)),
                "recall_drop": float(baseline_detection.get("recall", 0.0) - det.get("recall", 0.0)),
                "map_drop": float(baseline_map.get("map", 0.0) - m_ap.get("map", 0.0)),
                "mean_iou_drop": float(baseline_segmentation.get("mean_iou", 0.0) - seg.get("mean_iou", 0.0)),
                "mean_score_drop": float(baseline.get("mean_detection_score", 0.0) - case.get("mean_detection_score", 0.0)),
            }
        )
    return {"baseline": str(baseline.get("name", "")), "cases": rows}


def task_perception_perturb_image(image: Any, case: Mapping[str, Any]) -> NDArray[np.float64]:
    """Apply a deterministic perturbation used by task-perception sweeps."""

    values = _rgb_image(image)
    kind = str(case.get("kind", "identity")).lower().replace("-", "_")
    amount = case.get("amount", 0.0)
    if kind in {"identity", "none", "baseline"}:
        return values.copy()
    if kind in {"brightness", "low_light"}:
        factor = float(amount)
        return np.clip(values * factor, 0.0, 1.0)
    if kind in {"contrast"}:
        factor = float(amount)
        return np.clip(0.5 + (values - 0.5) * factor, 0.0, 1.0)
    if kind in {"gamma"}:
        gamma = max(float(amount), 1.0e-6)
        return np.clip(np.power(values, gamma), 0.0, 1.0)
    if kind in {"gaussian_blur", "blur"}:
        sigma = float(amount)
        return np.clip(gaussian_filter(values, sigma=(sigma, sigma, 0.0)), 0.0, 1.0)
    if kind in {"noise", "gaussian_noise"}:
        sigma = float(amount)
        seed = int(case.get("seed", 0))
        rng = np.random.default_rng(seed)
        return np.clip(values + rng.normal(0.0, sigma, size=values.shape), 0.0, 1.0)
    if kind in {"blur_noise", "blur_plus_noise"}:
        sigma = float(case.get("sigma", amount if np.isscalar(amount) else 1.0))
        noise = float(case.get("noise", 0.04))
        seed = int(case.get("seed", 0))
        blurred = gaussian_filter(values, sigma=(sigma, sigma, 0.0))
        rng = np.random.default_rng(seed)
        return np.clip(blurred + rng.normal(0.0, noise, size=values.shape), 0.0, 1.0)
    raise ValueError(f"Unsupported perturbation kind {case.get('kind')!r}.")


def render_detection_overlay(
    image: Any,
    predictions: Iterable[Any] | None = None,
    ground_truth: Iterable[Any] | None = None,
) -> NDArray[np.float64]:
    """Return an RGB image with predicted and ground-truth boxes overlaid."""

    output = _rgb_image(image).copy()
    thickness = max(1, min(output.shape[0], output.shape[1]) // 96)
    for box in [_as_box(item) for item in (ground_truth or [])]:
        _draw_box(output, box, color=np.array([0.1, 0.85, 0.25]), thickness=thickness)
    for box in [_as_box(item) for item in (predictions or [])]:
        _draw_box(output, box, color=np.array([1.0, 0.76, 0.05]), thickness=thickness)
    return np.clip(output, 0.0, 1.0)


def render_segmentation_overlay(
    image: Any,
    masks: Iterable[Any],
    *,
    alpha: float = 0.38,
) -> NDArray[np.float64]:
    """Return an RGB image with segmentation masks overlaid."""

    output = _rgb_image(image).copy()
    palette = _palette()
    for index, mask_value in enumerate(masks):
        mask = _as_segmentation_mask(mask_value, task_perception_config())
        color = palette[index % len(palette)]
        output[mask.mask] = (1.0 - float(alpha)) * output[mask.mask] + float(alpha) * color
    return np.clip(output, 0.0, 1.0)


def annotations_to_bboxes(
    annotations: Iterable[Mapping[str, Any]],
    *,
    bbox_format: str = "xyxy",
    image_size: tuple[int, int] | None = None,
    label_key: str = "label",
) -> list[TaskBoundingBox]:
    """Convert simple COCO/YOLO/xyxy-like annotations into boxes."""

    boxes = []
    fmt = bbox_format.lower()
    for ann in annotations:
        label = str(ann.get(label_key, ann.get("category_id", "object")))
        score = float(ann.get("score", 1.0))
        bbox = ann.get("bbox", ann.get("xyxy"))
        if bbox is None:
            bbox = (ann["x1"], ann["y1"], ann["x2"], ann["y2"])
        values = tuple(float(value) for value in bbox)
        if fmt == "xyxy":
            xyxy = values
        elif fmt == "xywh":
            x, y, w, h = values
            xyxy = (x, y, x + w, y + h)
        elif fmt == "yolo":
            if image_size is None:
                raise ValueError("image_size is required for YOLO annotations.")
            height, width = image_size
            cx, cy, w, h = values
            bw = w * width
            bh = h * height
            xyxy = ((cx * width) - (bw / 2.0), (cy * height) - (bh / 2.0), (cx * width) + (bw / 2.0), (cy * height) + (bh / 2.0))
        else:
            raise ValueError(f"Unsupported bbox_format {bbox_format!r}.")
        boxes.append(TaskBoundingBox(tuple(xyxy), label=label, score=score, metadata={key: value for key, value in ann.items() if key not in {"bbox", "xyxy"}}))
    return boxes


def annotations_to_masks(
    annotations: Iterable[Mapping[str, Any]],
    *,
    image_shape: tuple[int, int],
    bbox_format: str = "xyxy",
    label_key: str = "label",
) -> list[TaskSegmentationMask]:
    """Convert annotations with masks or boxes into binary masks."""

    masks = []
    for ann in annotations:
        label = str(ann.get(label_key, ann.get("category_id", "object")))
        score = float(ann.get("score", 1.0))
        if "mask" in ann:
            mask = np.asarray(ann["mask"]) > 0
        else:
            box = annotations_to_bboxes([ann], bbox_format=bbox_format, image_size=image_shape, label_key=label_key)[0]
            mask = np.zeros(image_shape, dtype=bool)
            x1, y1, x2, y2 = _int_box(box, image_shape)
            mask[y1:y2, x1:x2] = True
        masks.append(TaskSegmentationMask(mask, label=label, score=score))
    return masks


def _supported_model_backends() -> set[str]:
    return {
        "callable",
        "ultralytics_yolo",
        "torchvision_detection",
        "transformers_object_detection",
        "transformers_segmentation",
        "sam_automatic",
    }


def _task_model_cache_dir(config: TaskModelConfig | None = None) -> Path:
    if config is not None and config.options.get("cache_dir"):
        return Path(str(config.options["cache_dir"])).expanduser()
    return Path(os.environ.get("PYISETCAM_TASK_MODEL_CACHE", Path.home() / ".cache" / "pyisetcam" / "task_perception" / "yolo")).expanduser()


def _resolve_cached_model_id(model_id: str, config: TaskModelConfig) -> str:
    path = Path(model_id).expanduser()
    if path.exists() or path.is_absolute():
        return str(path)
    cached = _task_model_cache_dir(config) / model_id
    return str(cached) if cached.exists() else model_id


def _task_model_profiles() -> dict[str, dict[str, Any]]:
    try:
        path = resources.files("pyisetcam").joinpath("data/task_perception/model_profiles.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError):
        return {}
    profiles = payload.get("profiles", [])
    return {str(profile["name"]): dict(profile) for profile in profiles}


def _optional_import(module_name: str, install_hint: str) -> Any:
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Task perception backend requires optional dependency {module_name!r}. "
            f"Install it with {install_hint} or pass a custom callable detector/segmenter."
        ) from exc


def _ultralytics_yolo_adapter(config: TaskModelConfig) -> TaskModelAdapter:
    ultralytics = _optional_import("ultralytics", "pip install ultralytics")
    model_id = _resolve_cached_model_id(config.model_id or "yolov8n.pt", config)
    model = ultralytics.YOLO(model_id)

    def _predict(image: NDArray[np.float64]) -> Any:
        return _first_result(
            model(_uint8_image(image), conf=config.score_threshold, device=config.device, **dict(config.options.get("inference", {})))
        )

    def detector(image: NDArray[np.float64], **_: Any) -> list[TaskBoundingBox]:
        result = _predict(image)
        boxes = []
        names = getattr(result, "names", {}) or {}
        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is None:
            return boxes
        xyxy = _to_numpy(raw_boxes.xyxy)
        conf = _to_numpy(raw_boxes.conf) if getattr(raw_boxes, "conf", None) is not None else np.ones(xyxy.shape[0])
        cls = _to_numpy(raw_boxes.cls).astype(int) if getattr(raw_boxes, "cls", None) is not None else np.zeros(xyxy.shape[0], dtype=int)
        for coords, score, class_id in zip(xyxy, conf, cls, strict=False):
            label = _label_for_id(config, int(class_id), names)
            boxes.append(TaskBoundingBox(tuple(float(value) for value in coords), label=label, score=float(score), metadata={"class_id": int(class_id), "backend": config.backend}))
        return boxes

    def segmenter(image: NDArray[np.float64], **_: Any) -> list[TaskSegmentationMask]:
        result = _predict(image)
        raw_masks = getattr(result, "masks", None)
        raw_boxes = getattr(result, "boxes", None)
        if raw_masks is None:
            return []
        names = getattr(result, "names", {}) or {}
        masks = _to_numpy(raw_masks.data) > 0.5
        cls = _to_numpy(raw_boxes.cls).astype(int) if raw_boxes is not None and getattr(raw_boxes, "cls", None) is not None else np.zeros(masks.shape[0], dtype=int)
        conf = _to_numpy(raw_boxes.conf) if raw_boxes is not None and getattr(raw_boxes, "conf", None) is not None else np.ones(masks.shape[0])
        return [
            TaskSegmentationMask(mask, label=_label_for_id(config, int(class_id), names), score=float(score), metadata={"class_id": int(class_id), "backend": config.backend})
            for mask, class_id, score in zip(masks, cls, conf, strict=False)
        ]

    def classifier(image: NDArray[np.float64], **_: Any) -> list[TaskClassificationResult]:
        result = _predict(image)
        probs = getattr(result, "probs", None)
        if probs is None:
            return []
        names = getattr(result, "names", {}) or {}
        if getattr(probs, "top5", None) is not None:
            class_ids = [int(value) for value in probs.top5]
            scores = [float(value) for value in _to_numpy(probs.top5conf)]
        else:
            data = _to_numpy(getattr(probs, "data", []))
            order = np.argsort(data)[::-1][:5]
            class_ids = [int(value) for value in order]
            scores = [float(data[value]) for value in order]
        return [
            TaskClassificationResult(label=_label_for_id(config, class_id, names), score=score, class_id=class_id, metadata={"backend": config.backend})
            for class_id, score in zip(class_ids, scores, strict=False)
            if score >= config.score_threshold
        ]

    def pose_estimator(image: NDArray[np.float64], **_: Any) -> list[TaskPoseResult]:
        result = _predict(image)
        keypoints = getattr(result, "keypoints", None)
        raw_boxes = getattr(result, "boxes", None)
        if keypoints is None:
            return []
        xy = _to_numpy(keypoints.xy)
        conf = _to_numpy(keypoints.conf) if getattr(keypoints, "conf", None) is not None else np.ones(xy.shape[:2], dtype=float)
        boxes = detector(image)
        poses = []
        for index in range(xy.shape[0]):
            keypoints_xyc = np.column_stack([xy[index], conf[index]])
            bbox = boxes[index] if index < len(boxes) else None
            score = bbox.score if bbox is not None else 1.0
            if raw_boxes is not None and getattr(raw_boxes, "conf", None) is not None and index < len(_to_numpy(raw_boxes.conf)):
                score = float(_to_numpy(raw_boxes.conf)[index])
            poses.append(TaskPoseResult(keypoints_xyc, bbox=bbox, score=score, metadata={"backend": config.backend}))
        return poses

    def oriented_detector(image: NDArray[np.float64], **_: Any) -> list[TaskOrientedBoundingBox]:
        result = _predict(image)
        obb = getattr(result, "obb", None)
        if obb is None:
            return []
        names = getattr(result, "names", {}) or {}
        xywhr = _to_numpy(obb.xywhr)
        corners = _to_numpy(obb.xyxyxyxy)
        conf = _to_numpy(obb.conf) if getattr(obb, "conf", None) is not None else np.ones(xywhr.shape[0])
        cls = _to_numpy(obb.cls).astype(int) if getattr(obb, "cls", None) is not None else np.zeros(xywhr.shape[0], dtype=int)
        return [
            TaskOrientedBoundingBox(
                tuple(float(value) for value in row),
                np.asarray(corner, dtype=float).reshape(4, 2),
                label=_label_for_id(config, int(class_id), names),
                score=float(score),
                metadata={"class_id": int(class_id), "backend": config.backend},
            )
            for row, corner, class_id, score in zip(xywhr, corners, cls, conf, strict=False)
            if float(score) >= config.score_threshold
        ]

    def tracker(image: NDArray[np.float64], **_: Any) -> list[TaskTrackResult]:
        tracker_name = str(config.options.get("tracker", "bytetrack.yaml"))
        result = _first_result(
            model.track(_uint8_image(image), conf=config.score_threshold, device=config.device, tracker=tracker_name, persist=True, **dict(config.options.get("track", {})))
        )
        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is None:
            return []
        xyxy = _to_numpy(raw_boxes.xyxy)
        conf = _to_numpy(raw_boxes.conf) if getattr(raw_boxes, "conf", None) is not None else np.ones(xyxy.shape[0])
        cls = _to_numpy(raw_boxes.cls).astype(int) if getattr(raw_boxes, "cls", None) is not None else np.zeros(xyxy.shape[0], dtype=int)
        ids = _to_numpy(raw_boxes.id).astype(int) if getattr(raw_boxes, "id", None) is not None else np.arange(xyxy.shape[0], dtype=int)
        names = getattr(result, "names", {}) or {}
        return [
            TaskTrackResult(
                int(track_id),
                TaskBoundingBox(tuple(float(value) for value in coords), label=_label_for_id(config, int(class_id), names), score=float(score), metadata={"class_id": int(class_id), "backend": config.backend}),
            )
            for coords, score, class_id, track_id in zip(xyxy, conf, cls, ids, strict=False)
        ]

    return TaskModelAdapter(
        config=config,
        detector=detector,
        segmenter=segmenter,
        classifier=classifier,
        pose_estimator=pose_estimator,
        oriented_detector=oriented_detector,
        tracker=tracker,
    )


def _torchvision_detection_adapter(config: TaskModelConfig) -> TaskModelAdapter:
    torch = _optional_import("torch", "pip install torch torchvision")
    detection_models = _optional_import("torchvision.models.detection", "pip install torch torchvision")
    model_name = config.model_id or "fasterrcnn_resnet50_fpn_v2"
    if not hasattr(detection_models, model_name):
        raise ValueError(f"torchvision.models.detection has no model {model_name!r}.")
    weights = config.options.get("weights", "DEFAULT")
    kwargs = {"weights": weights} if weights not in {None, "none", "None"} else {"weights": None}
    model = getattr(detection_models, model_name)(**kwargs)
    model.eval()
    if config.device:
        model.to(config.device)

    def _predict(image: NDArray[np.float64]) -> Mapping[str, Any]:
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)).to(config.device)
        with torch.no_grad():
            return model([tensor])[0]

    def detector(image: NDArray[np.float64], **_: Any) -> list[TaskBoundingBox]:
        output = _predict(image)
        boxes = _to_numpy(output.get("boxes", []))
        scores = _to_numpy(output.get("scores", np.ones(len(boxes))))
        labels = _to_numpy(output.get("labels", np.zeros(len(boxes)))).astype(int)
        return [
            TaskBoundingBox(tuple(float(value) for value in coords), label=_label_for_id(config, int(label), {}), score=float(score), metadata={"class_id": int(label), "backend": config.backend})
            for coords, score, label in zip(boxes, scores, labels, strict=False)
            if float(score) >= config.score_threshold
        ]

    segmenter: SegmenterCallable | None = None
    if "mask" in model_name.lower():

        def segmenter(image: NDArray[np.float64], **_: Any) -> list[TaskSegmentationMask]:
            output = _predict(image)
            masks = _to_numpy(output.get("masks", []))
            scores = _to_numpy(output.get("scores", np.ones(len(masks))))
            labels = _to_numpy(output.get("labels", np.zeros(len(masks)))).astype(int)
            if masks.ndim == 4:
                masks = masks[:, 0, :, :]
            return [
                TaskSegmentationMask(mask >= 0.5, label=_label_for_id(config, int(label), {}), score=float(score), metadata={"class_id": int(label), "backend": config.backend})
                for mask, score, label in zip(masks, scores, labels, strict=False)
                if float(score) >= config.score_threshold
            ]

    return TaskModelAdapter(config=config, detector=detector, segmenter=segmenter)


def _transformers_object_detection_adapter(config: TaskModelConfig) -> TaskModelAdapter:
    transformers = _optional_import("transformers", "pip install transformers torch pillow")
    pipeline = transformers.pipeline(
        "object-detection",
        model=config.model_id,
        device=-1 if config.device == "cpu" else config.device,
        **dict(config.options.get("pipeline", {})),
    )

    def detector(image: NDArray[np.float64], **_: Any) -> list[TaskBoundingBox]:
        outputs = pipeline(_uint8_image(image), threshold=config.score_threshold, **dict(config.options.get("inference", {})))
        boxes = []
        for item in outputs:
            box = dict(item.get("box", {}))
            coords = (box.get("xmin", 0.0), box.get("ymin", 0.0), box.get("xmax", 0.0), box.get("ymax", 0.0))
            boxes.append(TaskBoundingBox(tuple(float(value) for value in coords), label=str(item.get("label", "object")), score=float(item.get("score", 1.0)), metadata={"backend": config.backend}))
        return boxes

    return TaskModelAdapter(config=config, detector=detector)


def _transformers_segmentation_adapter(config: TaskModelConfig) -> TaskModelAdapter:
    transformers = _optional_import("transformers", "pip install transformers torch pillow")
    pipeline = transformers.pipeline(
        "image-segmentation",
        model=config.model_id,
        device=-1 if config.device == "cpu" else config.device,
        **dict(config.options.get("pipeline", {})),
    )

    def segmenter(image: NDArray[np.float64], **_: Any) -> list[TaskSegmentationMask]:
        outputs = pipeline(_uint8_image(image), **dict(config.options.get("inference", {})))
        masks = []
        for item in outputs:
            mask = np.asarray(item.get("mask"), dtype=float)
            if mask.ndim == 3:
                mask = mask[..., 0]
            masks.append(TaskSegmentationMask(mask > 0, label=str(item.get("label", "object")), score=float(item.get("score", 1.0)), metadata={"backend": config.backend}))
        return masks

    return TaskModelAdapter(config=config, segmenter=segmenter)


def _sam_automatic_adapter(config: TaskModelConfig) -> TaskModelAdapter:
    sam = _optional_import("segment_anything", "pip install segment-anything torch")
    if not config.checkpoint_path:
        raise ValueError("sam_automatic requires checkpoint_path.")
    model_type = str(config.options.get("model_type", "vit_b"))
    model = sam.sam_model_registry[model_type](checkpoint=config.checkpoint_path)
    if config.device:
        model.to(device=config.device)
    generator = sam.SamAutomaticMaskGenerator(model, **dict(config.options.get("generator", {})))

    def segmenter(image: NDArray[np.float64], **_: Any) -> list[TaskSegmentationMask]:
        outputs = generator.generate(_uint8_image(image))
        masks = []
        for index, item in enumerate(outputs):
            score = float(item.get("predicted_iou", item.get("stability_score", 1.0)))
            masks.append(TaskSegmentationMask(item["segmentation"], label=str(item.get("label", f"sam_mask_{index}")), score=score, metadata={"backend": config.backend, "area": int(item.get("area", 0))}))
        return masks

    return TaskModelAdapter(config=config, segmenter=segmenter)


def _detection_metrics_for_label(
    predictions: list[TaskBoundingBox],
    ground_truth: list[TaskBoundingBox],
    threshold: float,
    config: TaskPerceptionConfig,
    label: str,
) -> dict[str, float]:
    pred = [box for box in predictions if box.label == label]
    gt = [box for box in ground_truth if box.label == label]
    matches, unmatched_pred, unmatched_gt = _match_boxes(pred, gt, threshold, config)
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "average_precision": _average_precision(pred, gt, threshold, config),
    }


def _match_boxes(
    predictions: list[TaskBoundingBox],
    ground_truth: list[TaskBoundingBox],
    threshold: float,
    config: TaskPerceptionConfig,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    matched_gt: set[int] = set()
    matches = []
    for pred_index, pred in enumerate(predictions):
        best_gt = None
        best_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            if not config.label_agnostic and pred.label != gt.label:
                continue
            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_gt = gt_index
                best_iou = iou
        if best_gt is not None and best_iou >= threshold:
            matched_gt.add(best_gt)
            matches.append(
                {
                    "prediction_index": int(pred_index),
                    "ground_truth_index": int(best_gt),
                    "label": ground_truth[best_gt].label,
                    "score": float(pred.score),
                    "iou": float(best_iou),
                }
            )
    unmatched_pred = [index for index in range(len(predictions)) if index not in {match["prediction_index"] for match in matches}]
    unmatched_gt = [index for index in range(len(ground_truth)) if index not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def _match_masks(
    predictions: list[TaskSegmentationMask],
    ground_truth: list[TaskSegmentationMask],
    config: TaskPerceptionConfig,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    matched_gt: set[int] = set()
    matches = []
    for pred_index, pred in enumerate(predictions):
        best_gt = None
        best_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            if not config.label_agnostic and pred.label != gt.label:
                continue
            iou = mask_iou(pred.mask, gt.mask)
            if iou > best_iou:
                best_gt = gt_index
                best_iou = iou
        if best_gt is not None and best_iou >= config.iou_threshold:
            matched_gt.add(best_gt)
            matches.append(
                {
                    "prediction_index": int(pred_index),
                    "ground_truth_index": int(best_gt),
                    "label": ground_truth[best_gt].label,
                    "score": float(pred.score),
                    "iou": float(best_iou),
                    "boundary_f1": boundary_f1_score(pred.mask, ground_truth[best_gt].mask, tolerance_px=config.boundary_tolerance_px),
                }
            )
    unmatched_pred = [index for index in range(len(predictions)) if index not in {match["prediction_index"] for match in matches}]
    unmatched_gt = [index for index in range(len(ground_truth)) if index not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def _average_precision(
    predictions: list[TaskBoundingBox],
    ground_truth: list[TaskBoundingBox],
    threshold: float,
    config: TaskPerceptionConfig,
) -> float:
    if not ground_truth:
        return 1.0 if not predictions else 0.0
    matched_gt: set[int] = set()
    tp = np.zeros(len(predictions), dtype=float)
    fp = np.zeros(len(predictions), dtype=float)
    for index, pred in enumerate(predictions):
        best_gt = None
        best_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            if not config.label_agnostic and pred.label != gt.label:
                continue
            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_gt = gt_index
                best_iou = iou
        if best_gt is not None and best_iou >= threshold:
            matched_gt.add(best_gt)
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / max(len(ground_truth), 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1], dtype=float))


def _prepare_predictions(values: Iterable[Any], config: TaskPerceptionConfig) -> list[TaskBoundingBox]:
    boxes = [_as_box(item) for item in values]
    boxes = [box for box in boxes if box.score >= config.score_threshold]
    boxes = sorted(boxes, key=lambda box: box.score, reverse=True)
    if config.max_detections is not None:
        boxes = boxes[: int(config.max_detections)]
    return boxes


def _prepare_masks(values: Iterable[Any], config: TaskPerceptionConfig) -> list[TaskSegmentationMask]:
    masks = [_as_segmentation_mask(item, config) for item in values]
    masks = [mask for mask in masks if mask.score >= config.score_threshold]
    masks = sorted(masks, key=lambda mask: mask.score, reverse=True)
    if config.max_detections is not None:
        masks = masks[: int(config.max_detections)]
    return masks


def _prepare_classifications(values: Iterable[Any], config: TaskModelConfig) -> list[TaskClassificationResult]:
    results = [_as_classification(item) for item in values]
    return [item for item in sorted(results, key=lambda result: result.score, reverse=True) if item.score >= config.score_threshold]


def _prepare_poses(values: Iterable[Any], config: TaskModelConfig) -> list[TaskPoseResult]:
    results = [_as_pose(item) for item in values]
    return [item for item in sorted(results, key=lambda result: result.score, reverse=True) if item.score >= config.score_threshold]


def _prepare_oriented_boxes(values: Iterable[Any], config: TaskModelConfig) -> list[TaskOrientedBoundingBox]:
    results = [_as_oriented_box(item) for item in values]
    return [item for item in sorted(results, key=lambda result: result.score, reverse=True) if item.score >= config.score_threshold]


def _prepare_tracks(values: Iterable[Any], config: TaskModelConfig) -> list[TaskTrackResult]:
    results = [_as_track(item) for item in values]
    return [item for item in sorted(results, key=lambda result: result.box.score, reverse=True) if item.box.score >= config.score_threshold]


def _as_box(value: Any) -> TaskBoundingBox:
    if isinstance(value, TaskBoundingBox):
        return value
    if isinstance(value, Mapping):
        label = str(value.get("label", value.get("category_id", "object")))
        score = float(value.get("score", 1.0))
        if "xyxy" in value:
            coords = tuple(float(item) for item in value["xyxy"])
        elif "bbox" in value:
            coords = tuple(float(item) for item in value["bbox"])
            if str(value.get("bbox_format", "xyxy")).lower() == "xywh":
                x, y, w, h = coords
                coords = (x, y, x + w, y + h)
        else:
            coords = (float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
        return TaskBoundingBox(coords, label=label, score=score, metadata={key: item for key, item in value.items() if key not in {"xyxy", "bbox", "x1", "y1", "x2", "y2"}})
    sequence = list(value)
    if len(sequence) < 4:
        raise ValueError("box sequence must contain at least four coordinates.")
    label = str(sequence[5]) if len(sequence) > 5 else "object"
    score = float(sequence[4]) if len(sequence) > 4 and np.isscalar(sequence[4]) else 1.0
    return TaskBoundingBox(tuple(float(item) for item in sequence[:4]), label=label, score=score)


def _as_classification(value: Any) -> TaskClassificationResult:
    if isinstance(value, TaskClassificationResult):
        return value
    if isinstance(value, Mapping):
        return TaskClassificationResult(
            label=str(value.get("label", value.get("class_id", "class"))),
            score=float(value.get("score", 1.0)),
            class_id=None if value.get("class_id") is None else int(value.get("class_id")),
            metadata={key: item for key, item in value.items() if key not in {"label", "score", "class_id"}},
        )
    sequence = list(value)
    if len(sequence) < 2:
        raise ValueError("classification sequence must contain label and score.")
    return TaskClassificationResult(label=str(sequence[0]), score=float(sequence[1]), class_id=int(sequence[2]) if len(sequence) > 2 else None)


def _as_pose(value: Any) -> TaskPoseResult:
    if isinstance(value, TaskPoseResult):
        return value
    if isinstance(value, Mapping):
        bbox = None if value.get("bbox") is None else _as_box(value["bbox"])
        return TaskPoseResult(
            value["keypoints_xyc"],
            bbox=bbox,
            label=str(value.get("label", "person")),
            score=float(value.get("score", 1.0)),
            metadata={key: item for key, item in value.items() if key not in {"keypoints_xyc", "bbox", "label", "score"}},
        )
    return TaskPoseResult(value)


def _as_oriented_box(value: Any) -> TaskOrientedBoundingBox:
    if isinstance(value, TaskOrientedBoundingBox):
        return value
    if isinstance(value, Mapping):
        return TaskOrientedBoundingBox(
            tuple(float(item) for item in value["xywhr"]),
            value["corners_xy"],
            label=str(value.get("label", "object")),
            score=float(value.get("score", 1.0)),
            metadata={key: item for key, item in value.items() if key not in {"xywhr", "corners_xy", "label", "score"}},
        )
    raise ValueError("oriented box must be TaskOrientedBoundingBox or mapping.")


def _as_track(value: Any) -> TaskTrackResult:
    if isinstance(value, TaskTrackResult):
        return value
    if isinstance(value, Mapping):
        return TaskTrackResult(int(value["track_id"]), _as_box(value["box"]), metadata={key: item for key, item in value.items() if key not in {"track_id", "box"}})
    raise ValueError("track result must be TaskTrackResult or mapping.")


def _as_segmentation_mask(value: Any, config: TaskPerceptionConfig) -> TaskSegmentationMask:
    if isinstance(value, TaskSegmentationMask):
        return value
    if isinstance(value, Mapping):
        label = str(value.get("label", value.get("category_id", "object")))
        score = float(value.get("score", 1.0))
        if "mask" not in value:
            raise ValueError("segmentation mapping must include a mask field.")
        return TaskSegmentationMask(np.asarray(value["mask"], dtype=float) >= config.mask_threshold, label=label, score=score)
    return TaskSegmentationMask(np.asarray(value, dtype=float) >= config.mask_threshold)


def _as_bool_mask(value: Any) -> NDArray[np.bool_]:
    if isinstance(value, TaskSegmentationMask):
        return np.asarray(value.mask, dtype=bool)
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("mask must be a 2-D array.")
    return np.asarray(array > 0, dtype=bool)


def _mask_boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return np.asarray(mask & ~eroded, dtype=bool)


def _labels(predictions: list[TaskBoundingBox], ground_truth: list[TaskBoundingBox], config: TaskPerceptionConfig) -> tuple[str, ...]:
    if config.label_agnostic:
        return ("object",)
    return tuple(sorted({box.label for box in predictions} | {box.label for box in ground_truth}))


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _rgb_image(image: Any) -> NDArray[np.float64]:
    values = np.asarray(image, dtype=float)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("image must be grayscale or RGB.")
    if not np.all(np.isfinite(values)):
        raise ValueError("image must contain only finite values.")
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value > 1.0:
        values = values / (255.0 if max_value <= 255.0 else max_value)
    return np.clip(values, 0.0, 1.0)


def _uint8_image(image: Any) -> NDArray[np.uint8]:
    return np.asarray(np.round(_rgb_image(image) * 255.0), dtype=np.uint8)


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _first_result(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _label_for_id(config: TaskModelConfig, class_id: int, names: Mapping[Any, Any]) -> str:
    labels = dict(config.labels)
    if class_id in labels:
        return str(labels[class_id])
    if str(class_id) in labels:
        return str(labels[str(class_id)])
    if class_id in names:
        return str(names[class_id])
    if str(class_id) in names:
        return str(names[str(class_id)])
    return str(class_id)


def _default_perturbations() -> tuple[Mapping[str, Any], ...]:
    return (
        {"name": "baseline", "kind": "identity", "amount": 0.0},
        {"name": "low_light_50pct", "kind": "brightness", "amount": 0.50},
        {"name": "blur_sigma_1p5", "kind": "gaussian_blur", "amount": 1.50},
        {"name": "noise_sigma_0p05", "kind": "gaussian_noise", "amount": 0.05, "seed": 7},
        {"name": "blur_noise", "kind": "blur_noise", "sigma": 1.20, "noise": 0.04, "seed": 9},
    )


def _draw_box(image: NDArray[np.float64], box: TaskBoundingBox, *, color: NDArray[np.float64], thickness: int) -> None:
    x1, y1, x2, y2 = _int_box(box, image.shape[:2])
    if x2 <= x1 or y2 <= y1:
        return
    image[y1 : min(y1 + thickness, y2), x1:x2, :] = color
    image[max(y2 - thickness, y1) : y2, x1:x2, :] = color
    image[y1:y2, x1 : min(x1 + thickness, x2), :] = color
    image[y1:y2, max(x2 - thickness, x1) : x2, :] = color


def _int_box(box: TaskBoundingBox, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1 = int(np.clip(np.floor(box.xyxy[0]), 0, width))
    y1 = int(np.clip(np.floor(box.xyxy[1]), 0, height))
    x2 = int(np.clip(np.ceil(box.xyxy[2]), 0, width))
    y2 = int(np.clip(np.ceil(box.xyxy[3]), 0, height))
    return x1, y1, x2, y2


def _palette() -> tuple[NDArray[np.float64], ...]:
    return (
        np.array([0.95, 0.22, 0.18], dtype=float),
        np.array([0.15, 0.70, 0.28], dtype=float),
        np.array([0.18, 0.36, 0.95], dtype=float),
        np.array([0.92, 0.64, 0.16], dtype=float),
        np.array([0.60, 0.30, 0.85], dtype=float),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


taskPerceptionConfig = task_perception_config
taskModelConfig = task_model_config
taskModelProfileNames = task_model_profile_names
taskModelProfile = task_model_profile
taskModelConfigFromProfile = task_model_config_from_profile
taskModelFromConfig = task_model_from_config
taskDetectorFromConfig = task_detector_from_config
taskSegmenterFromConfig = task_segmenter_from_config
bboxIoU = bbox_iou
maskIoU = mask_iou
boundaryF1Score = boundary_f1_score
detectionMetrics = detection_metrics
meanAveragePrecision = mean_average_precision
detectionRecallAtIoU = detection_recall_at_iou
segmentationMetrics = segmentation_metrics
meanIoU = mean_iou
runTaskDetector = run_task_detector
runTaskSegmenter = run_task_segmenter
taskScoreByStage = task_score_by_stage
taskPerceptionSweep = task_perception_sweep
taskDegradationReport = task_degradation_report
taskPerceptionPerturbImage = task_perception_perturb_image
renderDetectionOverlay = render_detection_overlay
renderSegmentationOverlay = render_segmentation_overlay
annotationsToBboxes = annotations_to_bboxes
annotationsToMasks = annotations_to_masks


__all__ = [
    "ClassifierCallable",
    "DetectorCallable",
    "OrientedDetectorCallable",
    "PoseCallable",
    "SegmenterCallable",
    "TaskBoundingBox",
    "TaskClassificationResult",
    "TaskModelAdapter",
    "TaskModelConfig",
    "TaskOrientedBoundingBox",
    "TaskPerceptionConfig",
    "TaskPoseResult",
    "TaskSegmentationMask",
    "TaskTrackResult",
    "TrackerCallable",
    "annotationsToBboxes",
    "annotationsToMasks",
    "annotations_to_bboxes",
    "annotations_to_masks",
    "bboxIoU",
    "bbox_iou",
    "boundaryF1Score",
    "boundary_f1_score",
    "detectionMetrics",
    "detectionRecallAtIoU",
    "detection_metrics",
    "detection_recall_at_iou",
    "maskIoU",
    "mask_iou",
    "meanAveragePrecision",
    "meanIoU",
    "mean_average_precision",
    "mean_iou",
    "renderDetectionOverlay",
    "renderSegmentationOverlay",
    "render_detection_overlay",
    "render_segmentation_overlay",
    "runTaskDetector",
    "runTaskSegmenter",
    "run_task_detector",
    "run_task_segmenter",
    "segmentationMetrics",
    "segmentation_metrics",
    "taskDegradationReport",
    "taskDetectorFromConfig",
    "taskModelConfig",
    "taskModelConfigFromProfile",
    "taskModelFromConfig",
    "taskModelProfile",
    "taskModelProfileNames",
    "taskPerceptionConfig",
    "taskPerceptionPerturbImage",
    "taskPerceptionSweep",
    "taskSegmenterFromConfig",
    "taskScoreByStage",
    "task_degradation_report",
    "task_detector_from_config",
    "task_model_config",
    "task_model_config_from_profile",
    "task_model_from_config",
    "task_model_profile",
    "task_model_profile_names",
    "task_perception_config",
    "task_perception_perturb_image",
    "task_perception_sweep",
    "task_segmenter_from_config",
    "task_score_by_stage",
]
