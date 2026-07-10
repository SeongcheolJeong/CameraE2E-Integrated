# ADAS Benchmark Readiness

This file records measured behavior of the current integrated demo. It is
research evidence, not camera-module or vehicle product sign-off.

## Verified on 2026-07-11

### Detector and geometry

- KITTI inventory: 5,985 train image/label pairs.
- Quick preflight: 50 stratified scenes, 185 ADAS boxes.
- Active model: locally fine-tuned KITTI YOLO11n epoch-4 checkpoint.
- Source mAP50: 0.5069, gate 0.50.
- Source mAP50-95: 0.3376, gate 0.30.
- Source class-balanced recall: 0.5407, gate 0.50.
- Geometry-only recall retention: 1.0066, gate 0.90.
- Geometry-only SSIM: 0.9990, gate 0.85.
- Bounding-box transform maximum error: 0 px.
- Camera output: 374 x 1242 after explicit Bayer alignment from a requested
  375 x 1242 array.
- HFOV error: below 1e-9 degree.

### Fidelity sanity

Preflight also executes three camera frames at the selected search fidelity.
These gates establish that the objective remains rankable; they are not camera
quality acceptance limits. The gates are detector-recall retention >= 0.20,
relative channel-gain imbalance <= 0.25, and ideal/output SSIM >= 0.45.

| Search fidelity | Recall retention | Channel mismatch | SSIM | Decision |
|---|---:|---:|---:|---|
| L0 analytic | 0.3095 | 0.1806 | 0.4933 | rankable |
| L1 bundled FDTD LUT | 0.0000 | 0.4389 | 0.4877 | blocked |

The L1 asset remains available as proxy evidence, but it is not allowed to
drive optimization until its color/response behavior is corrected or
calibrated. The active study therefore uses L0 analytic search.

### Real workflow smoke

A two-candidate exposure study completed real successive halving over 1, 2,
and 3 scenes plus five camera-pipeline robustness reruns. Both candidates were
feasible. Only the three-scene finalist was eligible for `best_case` and the
Pareto front. A three-scene dataset export then produced three float32 RAW NPZ
files, RGB previews, transformed labels, metadata, and a manifest with zero
validation issues. The HTML decision report was generated from the same
persisted evidence.

## Metric semantics

`rgb_clip_fraction` remains as a backward-compatible combined black/highlight
diagnostic. Feasibility uses `rgb_high_clip_fraction` only. Black-level pixels
are reported separately as `rgb_low_clip_fraction`; underexposure is handled by
signal-level gates. This prevents normal dark road regions from being mistaken
for highlight saturation.

## Remaining measured-evidence boundary

No measured QE/PTC/full-well, lens MTF/PSF, HW ISP trace, or complete TCAD
process-deck evidence is attached. L3 calibrated and product sign-off states
remain blocked by design. The trained detector checkpoint is a local runtime
artifact and is not silently replaced by a generic COCO score.
