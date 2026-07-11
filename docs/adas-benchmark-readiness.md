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
| L1 bundled FDTD LUT | 0.3095 | 0.1803 | 0.4981 | rankable proxy |

The previous L1 failure was a runtime normalization defect. The LUT field
response had been normalized to the first sweep row, which could be a different
wavelength, but the runtime consumed that column as a same-wavelength center
ratio. The runtime now computes `response(field, wavelength) / response(center,
wavelength)` when absolute response rows are available. This removes the
artificial spectral cast without changing the gates. L1 is now rankable, but
remains proxy evidence rather than calibrated sensor response.

### Real workflow smoke

A two-candidate exposure study completed real successive halving over 1, 2,
and 3 scenes plus five camera-pipeline robustness reruns. Both candidates were
feasible. Only the three-scene finalist was eligible for `best_case` and the
Pareto front. A three-scene dataset export then produced three float32 RAW NPZ
files, RGB previews, transformed labels, metadata, and a manifest with zero
validation issues. The HTML decision report was generated from the same
persisted evidence.

## Reproducibility and decision confidence

Every preflight, benchmark, optimization, and dataset export now carries a
`camerae2e_benchmark_manifest_v1`. It fixes selected frame and label hashes,
detector hash, ADAS class mapping, metric version, thresholds, robustness cases,
seed, fidelity policy, and code revision without using absolute paths as
identity.

Finalist scores include deterministic scene-bootstrap confidence intervals.
The top two finalists are compared with a paired bootstrap over the same scene
IDs and a minimum meaningful score delta. When that evidence does not separate
them, the decision is `indistinguishable`. These intervals describe benchmark
scene sampling uncertainty, not sensor-lot or hardware repeatability.

## Calibration and dataset gates

Calibration fits now require sample-count, normalized-RMSE, and R-squared gates.
Evidence is aggregated as separate sensor (`QE`, `PTC`, angular response), optics
(`MTF`), ISP (`color`), and HW ISP (`latency`) scopes. A scoped fit cannot promote
unrelated stages or create a product sign-off claim.

RAW validation now opens every NPZ and checks float32 dtype, shape, finite
values, RGB alignment, label bounds, group and source-hash split leakage, sample
checksums, metadata checksum, and manifest hash. The manifest records camera
configuration hashes, optimization evidence, benchmark manifest, fidelity,
model, seed, and code revision.

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
