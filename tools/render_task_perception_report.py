"""Render a task-level perception report for detection/segmentation evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam.task_perception import (  # noqa: E402
    TaskBoundingBox,
    TaskSegmentationMask,
    detection_metrics,
    mean_average_precision,
    render_detection_overlay,
    render_segmentation_overlay,
    segmentation_metrics,
    task_model_profile,
    task_model_profile_names,
    task_perception_config,
    task_perception_perturb_image,
    task_perception_sweep,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "perception" / "task_perception"
DEFAULT_YOLO_CACHE_SUMMARY = Path.home() / ".cache" / "pyisetcam" / "task_perception" / "yolo" / "download_summary.json"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _scene() -> tuple[np.ndarray, list[TaskBoundingBox], list[TaskSegmentationMask]]:
    height, width = 224, 256
    y, x = np.mgrid[0:height, 0:width].astype(float)
    image = np.zeros((height, width, 3), dtype=float)
    image[..., 0] = 0.10 + 0.08 * (x / max(width - 1, 1))
    image[..., 1] = 0.12 + 0.08 * (y / max(height - 1, 1))
    image[..., 2] = 0.14

    masks: list[TaskSegmentationMask] = []

    red_mask = np.zeros((height, width), dtype=bool)
    red_mask[34:92, 30:92] = True
    image[red_mask] = np.array([0.92, 0.10, 0.08])
    masks.append(TaskSegmentationMask(red_mask, label="red_box", score=1.0))

    green_mask = (x - 172.0) ** 2 + (y - 66.0) ** 2 <= 31.0**2
    image[green_mask] = np.array([0.10, 0.78, 0.20])
    masks.append(TaskSegmentationMask(green_mask, label="green_circle", score=1.0))

    blue_mask = np.zeros((height, width), dtype=bool)
    blue_mask[130:194, 58:194] = ((np.floor((x[130:194, 58:194] - 58.0) / 8.0) % 2.0) == 0)
    image[blue_mask] = np.array([0.08, 0.18, 0.88])
    masks.append(TaskSegmentationMask(blue_mask, label="blue_stripes", score=1.0))

    boxes = [mask.bbox for mask in masks]
    return np.clip(image, 0.0, 1.0), boxes, masks


def _color_detector(image: np.ndarray) -> list[TaskBoundingBox]:
    specs = (
        ("red_box", 0, 0.24),
        ("green_circle", 1, 0.22),
        ("blue_stripes", 2, 0.20),
    )
    boxes = []
    for label_name, channel, threshold in specs:
        other = np.max(np.delete(image, channel, axis=2), axis=2)
        dominance = image[..., channel] - other
        mask = (image[..., channel] > threshold) & (dominance > 0.08)
        if int(np.count_nonzero(mask)) < 16:
            continue
        rows, cols = np.nonzero(mask)
        score = float(np.clip(np.mean(dominance[mask], dtype=float) * 1.7, 0.0, 1.0))
        boxes.append(
            TaskBoundingBox(
                (float(np.min(cols)), float(np.min(rows)), float(np.max(cols) + 1), float(np.max(rows) + 1)),
                label=label_name,
                score=score,
            )
        )
    return boxes


def _color_segmenter(image: np.ndarray) -> list[TaskSegmentationMask]:
    specs = (
        ("red_box", 0, 0.24),
        ("green_circle", 1, 0.22),
        ("blue_stripes", 2, 0.20),
    )
    masks = []
    for label_name, channel, threshold in specs:
        other = np.max(np.delete(image, channel, axis=2), axis=2)
        dominance = image[..., channel] - other
        mask = (image[..., channel] > threshold) & (dominance > 0.08)
        if int(np.count_nonzero(mask)) < 16:
            continue
        score = float(np.clip(np.mean(dominance[mask], dtype=float) * 1.7, 0.0, 1.0))
        masks.append(TaskSegmentationMask(mask, label=label_name, score=score))
    return masks


def _save_image(path: Path, image: np.ndarray, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 4.8))
    plt.imshow(np.clip(image, 0.0, 1.0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=150)
    plt.close()


def _save_sweep_plot(path: Path, sweep: dict[str, Any]) -> None:
    cases = sweep["cases"]
    names = [case["name"] for case in cases]
    recall = [case.get("detection_metrics", {}).get("recall", 0.0) for case in cases]
    map_value = [case.get("map", {}).get("map", 0.0) for case in cases]
    miou = [case.get("segmentation_metrics", {}).get("mean_iou", 0.0) for case in cases]
    score = [case.get("mean_detection_score", 0.0) for case in cases]

    x = np.arange(len(names))
    plt.figure(figsize=(9.2, 4.8))
    plt.plot(x, recall, marker="o", linewidth=2.2, label="Detection recall")
    plt.plot(x, map_value, marker="o", linewidth=2.2, label="mAP")
    plt.plot(x, miou, marker="o", linewidth=2.2, label="Segmentation mIoU")
    plt.plot(x, score, marker="o", linewidth=2.2, label="Mean confidence")
    plt.xticks(x, names, rotation=22, ha="right")
    plt.ylim(-0.04, 1.04)
    plt.grid(alpha=0.35)
    plt.title("Task Perception Robustness Sweep")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    header_html = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    row_html = "\n".join(
        "<tr>" + "".join(f"<td>{escape(_format(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #15212d;
      --muted: #627184;
      --line: #d2dce6;
      --card: #ffffff;
      --accent: #8b4d22;
      --green: #2f7d54;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top right, rgb(139 77 34 / 13%), transparent 34rem),
        linear-gradient(135deg, #f7f0df 0%, #eff7f6 100%);
      color: var(--ink);
      font: 15px/1.58 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 38px 24px 58px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 38px;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 34px 0 14px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }}
    .lead {{
      max-width: 900px;
      color: var(--muted);
      font-size: 17px;
    }}
    .cards, .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .card, .panel, table {{
      background: rgb(255 255 255 / 95%);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 28px rgb(21 33 45 / 7%);
    }}
    .card, .panel {{
      padding: 17px 18px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 760;
    }}
    img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
      margin: 12px 0 20px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f6f8fa;
    }}
    code {{
      background: #edf3f5;
      border-radius: 6px;
      padding: 2px 5px;
    }}
    .note {{
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: rgb(139 77 34 / 8%);
      border-radius: 12px;
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>
"""


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _downloaded_yolo_rows() -> list[list[Any]]:
    if not DEFAULT_YOLO_CACHE_SUMMARY.exists():
        return []
    payload = json.loads(DEFAULT_YOLO_CACHE_SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for model in payload.get("models", []):
        rows.append(
            [
                model.get("model_id", ""),
                model.get("profile", ""),
                round(float(model.get("size_bytes", 0)) / (1024.0 * 1024.0), 3),
                model.get("path", ""),
            ]
        )
    return rows


def build_report(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = task_perception_config(iou_threshold=0.50, score_threshold=0.05, map_iou_thresholds=(0.50, 0.75))

    image, gt_boxes, gt_masks = _scene()
    predictions = _color_detector(image)
    segmentations = _color_segmenter(image)
    det_metrics = detection_metrics(predictions, gt_boxes, config)
    map_metrics = mean_average_precision(predictions, gt_boxes, config)
    seg_metrics = segmentation_metrics(segmentations, gt_masks, config)

    perturbations = (
        {"name": "baseline", "kind": "identity", "amount": 0.0},
        {"name": "low_light_45pct", "kind": "brightness", "amount": 0.45},
        {"name": "blur_sigma_2p0", "kind": "gaussian_blur", "amount": 2.0},
        {"name": "noise_sigma_0p07", "kind": "gaussian_noise", "amount": 0.07, "seed": 11},
        {"name": "blur_noise", "kind": "blur_noise", "sigma": 1.55, "noise": 0.055, "seed": 13},
    )
    sweep = task_perception_sweep(
        image,
        _color_detector,
        gt_boxes,
        segmenter=_color_segmenter,
        ground_truth_masks=gt_masks,
        perturbations=perturbations,
        config=config,
    )

    reference_path = output_dir / "task_reference.png"
    detection_path = output_dir / "task_detection_overlay.png"
    segmentation_path = output_dir / "task_segmentation_overlay.png"
    sweep_path = output_dir / "task_robustness_sweep.png"
    degraded_path = output_dir / "task_degraded_overlay.png"
    html_path = output_dir / "task_perception_report.html"
    json_path = output_dir / "task_perception_summary.json"

    degraded_image = task_perception_perturb_image(image, perturbations[-1])
    degraded_boxes = _color_detector(degraded_image)
    _save_image(reference_path, image, "Synthetic task scene")
    _save_image(detection_path, render_detection_overlay(image, predictions, gt_boxes), "Detection overlay")
    _save_image(segmentation_path, render_segmentation_overlay(image, segmentations), "Segmentation overlay")
    _save_image(degraded_path, render_detection_overlay(degraded_image, degraded_boxes, gt_boxes), "Blur + noise detection overlay")
    _save_sweep_plot(sweep_path, sweep)

    generated_at = datetime.now().isoformat(timespec="seconds")
    git_commit = _git_commit()
    profile_rows = []
    for profile_name in task_model_profile_names():
        profile = task_model_profile(profile_name)
        profile_rows.append(
            [
                profile["name"],
                profile["backend"],
                profile["task"],
                profile.get("model_id", ""),
                profile.get("options", {}).get("install", "custom callable"),
            ]
        )
    downloaded_rows = _downloaded_yolo_rows()
    sweep_rows = []
    for case in sweep["cases"]:
        sweep_rows.append(
            [
                case["name"],
                case.get("detection_count", 0),
                case.get("mean_detection_score", 0.0),
                case.get("detection_metrics", {}).get("recall", 0.0),
                case.get("map", {}).get("map", 0.0),
                case.get("segmentation_metrics", {}).get("mean_iou", 0.0),
            ]
        )

    body = f"""
<h1>Task Perception Detection And Segmentation Report</h1>
<p class="lead">This report is the downstream perception layer requested for object detection and segmentation. It is separate from the human-visible image-quality perception report and focuses on task decisions after camera/ISP degradation.</p>
<div class="cards">
  <div class="card"><div class="label">Git commit</div><div class="value">{escape(git_commit)}</div></div>
  <div class="card"><div class="label">Generated</div><div class="value">{escape(generated_at)}</div></div>
  <div class="card"><div class="label">Baseline mAP</div><div class="value">{map_metrics["map"]:.3f}</div></div>
  <div class="card"><div class="label">Baseline mIoU</div><div class="value">{seg_metrics["mean_iou"]:.3f}</div></div>
</div>

<h2>Architecture</h2>
<div class="panel">
  <p><code>task_perception.py</code> is model-agnostic. A detector or segmenter is any callable that accepts an RGB image and returns boxes or masks. The package then normalizes outputs, computes metrics, renders overlays, and runs robustness sweeps.</p>
  <p class="note">Heavy models such as YOLO, Detectron, or SAM are intentionally not core dependencies. They should be optional adapters on top of this metric/report layer.</p>
</div>

<h2>Configurable Model Profiles</h2>
<p>These profiles can instantiate optional model adapters when the corresponding third-party package and weights are available. The core report below still uses a deterministic color-threshold detector so unit tests stay fast and dependency-free.</p>
{_table(["Profile", "Backend", "Task", "Model ID", "Install"], profile_rows)}
{("<h2>Downloaded YOLO11 Weights</h2>" + _table(["Model", "Profile", "Size MB", "Path"], downloaded_rows)) if downloaded_rows else ""}

<h2>Phase 1: Detection And Segmentation Metrics</h2>
{_table(["Metric", "Value"], [
    ["Detection precision", det_metrics["precision"]],
    ["Detection recall", det_metrics["recall"]],
    ["Detection F1", det_metrics["f1"]],
    ["Detection AP@0.50", map_metrics.get("ap50", 0.0)],
    ["Detection AP@0.75", map_metrics.get("ap75", 0.0)],
    ["Detection mAP", map_metrics["map"]],
    ["Segmentation precision", seg_metrics["precision"]],
    ["Segmentation recall", seg_metrics["recall"]],
    ["Segmentation mean IoU", seg_metrics["mean_iou"]],
    ["Segmentation boundary F1", seg_metrics["mean_boundary_f1"]],
])}

<div class="figures">
  <figure><img src="{reference_path.name}" alt="Synthetic task scene"><figcaption>Synthetic task scene with object-level ground truth.</figcaption></figure>
  <figure><img src="{detection_path.name}" alt="Detection overlay"><figcaption>Green boxes are ground truth; yellow boxes are model predictions.</figcaption></figure>
  <figure><img src="{segmentation_path.name}" alt="Segmentation overlay"><figcaption>Segmentation masks rendered over the scene.</figcaption></figure>
</div>

<h2>Phase 2: Robustness Sweep</h2>
<p>The sweep perturbs the same input image with low light, blur, noise, and blur+noise. This is the camera/ISP-facing evidence: task scores should be tracked across image-quality changes, not only on a clean input.</p>
{_table(["Case", "Detections", "Mean score", "Recall", "mAP", "Segmentation mIoU"], sweep_rows)}
<div class="figures">
  <figure><img src="{sweep_path.name}" alt="Robustness sweep plot"><figcaption>Detection recall, mAP, segmentation mIoU, and confidence across perturbations.</figcaption></figure>
  <figure><img src="{degraded_path.name}" alt="Degraded detection overlay"><figcaption>Detection overlay under blur+noise, showing task degradation evidence.</figcaption></figure>
</div>

<h2>Implemented Public API</h2>
<div class="panel">
  <p>Core functions: <code>bbox_iou</code>, <code>mask_iou</code>, <code>detection_metrics</code>, <code>mean_average_precision</code>, <code>segmentation_metrics</code>, <code>boundary_f1_score</code>, <code>run_task_detector</code>, <code>run_task_segmenter</code>, <code>task_score_by_stage</code>, and <code>task_perception_sweep</code>.</p>
  <p>Overlay and annotation helpers: <code>render_detection_overlay</code>, <code>render_segmentation_overlay</code>, <code>annotations_to_bboxes</code>, and <code>annotations_to_masks</code>.</p>
</div>
"""

    summary = {
        "generated_at": generated_at,
        "git_commit": git_commit,
        "config": _json_ready(config.__dict__),
        "figures": {
            "reference": str(reference_path),
            "detection_overlay": str(detection_path),
            "segmentation_overlay": str(segmentation_path),
            "robustness_sweep": str(sweep_path),
            "degraded_overlay": str(degraded_path),
        },
        "model_profiles": profile_rows,
        "downloaded_yolo_models": downloaded_rows,
        "baseline": {
            "detections": [box.to_dict() for box in predictions],
            "segmentations": [mask.to_dict() for mask in segmentations],
            "detection_metrics": _json_ready(det_metrics),
            "map": _json_ready(map_metrics),
            "segmentation_metrics": _json_ready(seg_metrics),
        },
        "sweep": _json_ready(sweep),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_html_page("Task Perception Detection And Segmentation Report", body), encoding="utf-8")
    return {"html": html_path, "summary": json_path, "figures": summary["figures"], "metrics": summary}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = build_report(args.output_dir)
    print(result["html"])
    return Path(result["html"])


if __name__ == "__main__":
    main()
