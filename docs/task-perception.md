# Task Perception

`pyisetcam.task_perception` is the downstream computer-vision perception layer for object detection, segmentation, and robustness evidence. It is separate from `pyisetcam.perception`, which covers human-visible image-quality metrics.

## Scope

This layer answers questions such as:

- Does object detection still work after lens, sensor, ISP, noise, blur, or low-light degradation?
- Does segmentation mask quality degrade at edges or thin structures?
- Which camera/ISP stage causes task-level mAP, recall, or mIoU loss?
- How stable are model confidence scores across robustness sweeps?

## Design Rule

Do not make heavy AI frameworks a core dependency. The module accepts detector and segmenter callables, then handles:

- output normalization
- detection and segmentation metrics
- overlay rendering
- annotation conversion
- robustness sweeps
- report evidence

YOLO, Detectron, SAM, or other models should be optional adapters that return the common box/mask payloads.

## Configurable Model Backends

Use bundled model profiles when you want pyisetcam to instantiate an optional model adapter:

- `custom_callable`: user-supplied detector and/or segmenter callables.
- `ultralytics_yolo11n_detection`: YOLO11n object detection profile using `yolo11n.pt`.
- `ultralytics_yolo11n_segmentation`: YOLO11n instance segmentation profile using `yolo11n-seg.pt`.
- `ultralytics_yolo11n_classification`: YOLO11n classification profile using `yolo11n-cls.pt`.
- `ultralytics_yolo11n_pose`: YOLO11n pose-estimation profile using `yolo11n-pose.pt`.
- `ultralytics_yolo11n_obb`: YOLO11n oriented detection profile using `yolo11n-obb.pt`.
- `ultralytics_yolo11n_bytetrack`: YOLO11n detection profile wired to ByteTrack using `yolo11n.pt`.
- `torchvision_fasterrcnn_resnet50_fpn_v2_coco`: TorchVision Faster R-CNN profile.
- `torchvision_maskrcnn_resnet50_fpn_v2_coco`: TorchVision Mask R-CNN profile.
- `transformers_detr_resnet50_coco`: Hugging Face Transformers DETR object-detection profile.
- `transformers_mask2former_ade20k`: Hugging Face Transformers image-segmentation profile.
- `sam_vit_b_automatic`: Segment Anything automatic-mask profile. This requires a local checkpoint path.

The optional packages are not installed by default. Install only the backend you need:

```bash
python -m pip install ultralytics lap
python -m pip install torch torchvision
python -m pip install transformers torch pillow
python -m pip install segment-anything torch
```

Equivalent optional extras are also available:

```bash
python -m pip install -e ".[yolo]"
python -m pip install -e ".[torch-vision]"
python -m pip install -e ".[transformers]"
python -m pip install -e ".[sam]"
```

Example:

```python
from pyisetcam import task_model_config_from_profile, task_model_from_config

config = task_model_config_from_profile(
    "ultralytics_yolo11n_detection",
    device="cpu",
    score_threshold=0.35,
)
adapter = task_model_from_config(config)

detections = adapter.detect(image)
```

Download the YOLO11n family into the local pyisetcam cache:

```bash
python tools/download_yolo_models.py
```

The default cache location is:

```text
~/.cache/pyisetcam/task_perception/yolo
```

For custom models:

```python
from pyisetcam import task_model_config, task_model_from_config

def my_detector(image):
    return [{"xyxy": [10, 10, 80, 80], "label": "target", "score": 0.9}]

config = task_model_config(name="my_detector", backend="callable", task="detection")
adapter = task_model_from_config(config, detector=my_detector)
boxes = adapter.detect(image)
```

## Main Types

- `TaskBoundingBox`: detection box in `xyxy` coordinates with `label`, `score`, and metadata.
- `TaskSegmentationMask`: binary mask with `label`, `score`, and metadata.
- `TaskPerceptionConfig`: thresholds for IoU, score filtering, mask binarization, mAP thresholds, and boundary tolerance.
- `TaskModelConfig`: optional model backend, model ID, device, score threshold, labels, and backend options.
- `TaskModelAdapter`: normalized `detect(...)` and `segment(...)` wrapper around optional model backends.

## Phase 1 Functions

- `task_perception_config(...)`: create a validated task-perception config.
- `task_model_profile_names()`: list bundled model profile names.
- `task_model_profile(name)`: read one model profile.
- `task_model_config_from_profile(name, ...)`: create a configurable backend profile.
- `task_model_from_config(config, ...)`: instantiate an optional model adapter.
- `task_detector_from_config(config)`: create a detector callable.
- `task_segmenter_from_config(config)`: create a segmenter callable.
- `bbox_iou(box_a, box_b)`: bounding-box IoU.
- `mask_iou(mask_a, mask_b)`: binary-mask IoU.
- `boundary_f1_score(pred_mask, gt_mask)`: segmentation boundary F1.
- `detection_metrics(predictions, ground_truth)`: precision, recall, F1, AP, and matches.
- `mean_average_precision(predictions, ground_truth)`: mAP across configured IoU thresholds.
- `segmentation_metrics(predictions, ground_truth)`: precision, recall, F1, mIoU, boundary F1.
- `run_task_detector(detector, image)`: run and normalize a callable detector.
- `run_task_segmenter(segmenter, image)`: run and normalize a callable segmenter.

## Phase 2 Functions

- `task_score_by_stage(stage_images, detector, ground_truth_boxes, ...)`: evaluate stage-wise perception scores.
- `task_perception_sweep(image, detector, ground_truth_boxes, ...)`: run low-light, blur, noise, and blur+noise robustness sweeps.
- `task_degradation_report(case_results)`: summarize metric drops from baseline.
- `task_perception_perturb_image(image, case)`: deterministic perturbation helper.
- `render_detection_overlay(...)`: draw predicted and ground-truth boxes.
- `render_segmentation_overlay(...)`: render mask overlays.
- `annotations_to_bboxes(...)`: convert simple COCO/YOLO/xyxy annotations to boxes.
- `annotations_to_masks(...)`: convert masks or box annotations to binary masks.

## Example

```python
from pyisetcam import TaskBoundingBox, bbox_iou, detection_metrics

ground_truth = [TaskBoundingBox((10, 10, 50, 50), label="car")]
predictions = [TaskBoundingBox((12, 12, 48, 48), label="car", score=0.91)]

print(bbox_iou(predictions[0], ground_truth[0]))
print(detection_metrics(predictions, ground_truth))
```

## HTML Report

Generate the task perception evidence report with:

```bash
python tools/render_task_perception_report.py
```

Outputs are written to:

- `reports/perception/task_perception/task_perception_report.html`
- `reports/perception/task_perception/task_perception_summary.json`
- `reports/perception/task_perception/task_reference.png`
- `reports/perception/task_perception/task_detection_overlay.png`
- `reports/perception/task_perception/task_segmentation_overlay.png`
- `reports/perception/task_perception/task_robustness_sweep.png`
- `reports/perception/task_perception/task_degraded_overlay.png`
