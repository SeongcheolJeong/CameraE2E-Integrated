# CameraE2E v2 Architecture

CameraE2E v2 is a local Camera Design Decision Platform for camera-system
researchers. Its primary workflow is:

```text
Requirements -> Baseline -> Sensitivity -> Optimization
             -> Candidate Validation -> RAW Dataset -> Decision Report
```

For ADAS studies, sensitivity and optimization are gated by a benchmark
preflight:

```text
KITTI inventory -> source YOLO baseline -> ideal geometry re-capture
                -> coordinate gates -> selected-fidelity camera sanity
                -> camera optimization
```

The system remains research-grade and calibration-ready. It does not promote a
solver result, a public sensor record, or an optimized proxy model to product
sign-off without measured evidence.

## Runtime Architecture

CameraE2E v2 is a modular monolith. React, the FastAPI v2 router, and the CLI
all use the same `CameraE2EService` application boundary.

```text
React / CLI
    |
Typed v2 API
    |
Persistent Job Orchestrator ---- project.db
    |                                  |
Study Evaluator                 Artifact Registry
    |                                  |
Fidelity Router -------- content-addressed files
    |
pyisetcam analytic/LUT engine
    |
RayOptics / FDTD / TCAD adapters
```

The Workbench can switch between persisted studies without mixing their jobs,
preflight status, candidates, datasets, or reports.

The Workbench server exposes only `/api/v2` resources. The v1 Python modules
remain available for migration and parity tests but are not instantiated by the
runtime API.

## Project Contract

Each project is portable and self-describing:

```text
project/
  project.toml
  project.db
  artifacts/       # SHA-256 content-addressed files
  runs/
  exports/
  reports/
```

`project.db` stores immutable study revisions, persistent jobs, artifact
metadata, and job-to-artifact relationships. A backend restart changes a
running job to `interrupted`; it does not silently lose or recreate the result.

Every artifact records its content hash, type, media type, fidelity level,
readiness tier, source, dependencies, validation result, and creation time.
Missing files, hash mismatches, and missing dependency hashes are reported by
`camerae2e doctor`.

Canonical camera assets are typed records above the blob layer. Supported kinds
include lens design, sensor product, pixel stack, optical LUT, material n/k,
ISP profile, calibration, perception model, and camera preset. One asset can
reference one or more immutable artifact hashes and records its valid parameter
domain separately from file storage.

## Fidelity Contract

| Level | Backend | Intended use |
|---|---|---|
| `L0_analytic` | pyisetcam analytic models | broad sensitivity and candidate search |
| `L1_lut` | FDTD/TCAD/RayOptics-derived LUT | improved candidate ranking |
| `L2_solver` | explicit local solver validation job | shortlisted candidate evidence |
| `L3_calibrated` | measured-to-simulated fit | scoped quantitative prediction |

An unavailable L1 asset can fall back to L0 only when `allow_proxy=true`; the
fallback is recorded. L2 cannot be selected as a normal camera evaluation until
a solver-generated artifact is attached. L3 requires measured evidence.

RayOptics PSFs remain geometric ray histograms. FDTD requires convergence,
measured stack geometry, and material evidence. TCAD must share lineage with its
FDTD generation map and remains calibration-required until device measurements
are attached.

## Study Contract

A study snapshot contains:

- mission requirements with explicit units;
- one baseline camera module;
- one or more scene cases and their physical/proxy classification;
- design variables with units and readiness tiers;
- objectives, hard constraints, random seed, and search budget;
- fidelity policy and optional perception model path.

The current ADAS target uses a real YOLO model and KITTI/YOLO labels. It does not
substitute a RAW-quality proxy when the detector or labels are missing. Both
original KITTI object rows and Ultralytics YOLO-normalized KITTI rows are
accepted and normalized to the same ADAS class groups.

Dataset and model discovery is portable: `CAMERAE2E_KITTI_ROOT` and
`CAMERAE2E_YOLO_MODEL` override repository-local runtime locations. Trained
weights and KITTI data are not committed as source assets.

RGB scene files are always marked `display_rgb_proxy` unless the caller attaches
stronger provenance. CameraE2E applies the camera pipeline but does not claim to
recover lost source spectra or the original KITTI sensor RAW values.

### Geometry and color contracts

`GeometryTransform` records source, requested sensor, CFA-aligned active sensor,
readout, and ISP output dimensions. CameraE2E v2 disables the legacy automatic
sensor resize, sets the scene HFOV explicitly, and transforms every label with
the same coordinate mapping used by the camera output. Odd sensor dimensions
are reported as CFA-alignment warnings instead of being silently changed.

QE profiles are executable sensor inputs. `isetcam_default_rgb`,
`sony_imx363`, and `onsemi_ar0132at` are supported; the latter two reuse public
ISETCam response assets and remain proxy evidence unless measured QE is
attached. A custom NPZ profile must contain validated spectral arrays.

Free 3x3 CCM search is not enabled without color evidence. The color
calibration path fits a regularized 3x3 matrix, constrains neutral preservation
and coefficient magnitude, and requires an independent holdout improvement
before marking the scoped CCM fit calibrated.

## ADAS Benchmark Contract

The default KITTI suite uses 50 stratified scenes for quick decisions and 200
for final candidate evidence. Candidate search uses successive halving:

```text
24 candidates x 8 scenes
12 candidates x 20 scenes
 6 candidates x 50 scenes
 3 candidates x 200 scenes
```

Optimization is disabled until source-model preflight meets all thresholds:

- mAP50 >= 0.50;
- mAP50-95 >= 0.30;
- class-balanced recall >= 0.50;
- ideal re-capture recall retention >= 0.90;
- ideal re-capture SSIM >= 0.85;
- label transform error <= 0.5 px.

Preflight then executes a small camera-output subset at the selected search
fidelity. It requires detector-signal retention >= 0.20, relative channel-gain
imbalance <= 0.25, and ideal/output SSIM >= 0.45. These are objective
rankability gates, not final camera performance targets. A poor but rankable
baseline can be optimized; a path that produces no detections or a severe
channel cast cannot.

`Misc` and other non-ADAS KITTI classes are excluded from ground truth. If all
camera candidates produce zero detector signal, the job fails with
`objective_degenerate`; image-quality support cannot create a best perception
candidate by itself.

Only candidates evaluated at the final successive-halving scene budget are
eligible for `best_case` and the Pareto front. Early-stage scores remain visible
for audit, but are not compared as if they had the same evidence budget.

The exact benchmark selection is content-addressed by
`camerae2e_benchmark_manifest_v1`. It records image/label hashes, detector hash,
metric version, class mapping, thresholds, perturbations, fidelity policy, seed,
and code revision. Finalists are compared by paired scene bootstrap and a
minimum practical score delta. `decision_status=indistinguishable` prevents a
small or uncertain score difference from being presented as a resolved winner.

Robustness is computed from real camera reruns for exposure -1/+1 EV, PSF
defocus, OCL/CRA mismatch, and low illumination. It is not the previous nominal
`min(mAP, recall)` placeholder.

Detector-training jobs record epoch-level progress and validation metrics.
Cancellation is honored after the current native epoch rather than leaving a
job permanently in the running state.

## Requirement Gates

Every evaluated candidate records gates for HFOV consistency, projected object
pixels at range, sensor diagonal, diffraction sampling, pixel bandwidth/FPS,
rolling-shutter time, total latency, clipping, and available SNR/full-well
evidence. Missing read-noise/full-well measurements are reported as
`not_evaluable` and prevent calibrated sensor claims; they are not silently
invented.

Highlight saturation and black-level occupancy are separate metrics. The 1%
hard clipping limit applies to `rgb_high_clip_fraction`; low/zero pixels are a
diagnostic and underexposure is evaluated through signal-level gates.

## RAW Dataset Contract

The v2 dataset factory reruns the selected camera candidate over the requested
benchmark scenes and writes:

```text
manifest.json
metadata.jsonl
raw/*.npz
rgb/*.png
labels/*.json
optional raw_tiff/*.tiff and stages/*.npz
```

The camera-aware v3 export adds a KITTI source adapter, source split preservation,
optional P2 calibration, camera/exposure/noise recipes, and resumable sample contracts.
When P2 is available, the RGB proxy and labels share one source-to-target pinhole
transform. `source_bounded` is the default resolution policy; `target_readout_proxy`
records `upsampled_scene_proxy` whenever the target exceeds source information.

Each RAW NPZ records the floating RAW response, sensor digital response, CFA, bit depth,
black level, and white level. These research arrays are not standards-compliant DNGs.

RAW NPZ arrays are normalized to float32 before compression so a 200-scene
export does not retain unnecessary float64 storage.

Metadata includes source hashes, camera configuration, geometry transform,
fidelity, seed, metrics, and truth boundary. Splits are deterministic by scene
group so variants of the same source frame cannot leak across train,
validation, and test. DNG remains excluded until compliant tag writing is
implemented.

Export validation reads every RAW NPZ and RGB/label artifact. It verifies RAW
dtype, shape and finite values, RGB alignment, bounding-box bounds, source-hash
and group split isolation, per-file checksums, metadata checksum, and the
content-addressed manifest. A failed validation remains attached to the artifact
and is shown as failed in the Workbench.

## Solver Jobs

RayOptics, FDTD, and TCAD implement a common lifecycle:

```text
prepare -> submit -> collect -> validate -> publish artifact
```

By default, candidate validation checks existing assets and produces evidence
manifests without launching expensive solvers. Actual solver commands run only
when `execute=true` and `acknowledge_expensive=true`. Commands execute in their
registered local environments and preserve stdout, stderr, return code, source
stage, and manifest validation.

## Calibration

The v2 calibration job ingests paired measured and simulated JSON, CSV, NPY, or
NPZ values. It fits an affine response and records RMSE, normalized RMSE, MAE,
R-squared, a 95% residual interval, and the valid input domain. Sample-count,
normalized-RMSE, and R-squared gates determine whether that scoped artifact is
`calibrated` or remains `calibration_required`.

Calibration promotion is scoped. A calibrated QE fit does not promote lens PSF,
TCAD collection, ISP latency, or the complete camera module.

`camerae2e_calibration_pack_v1` aggregates required evidence by sensor, optics,
ISP, and HW ISP stage. Even a complete pack is not by itself product sign-off;
requirements, manufacturing variation, and hardware validation remain separate
evidence.

## Interfaces

Install the package and use the local CLI:

```bash
pip install -e .
camerae2e init ./my-camera-study --name "ADAS Camera Study"
camerae2e doctor ./my-camera-study
camerae2e run ./my-camera-study STUDY_ID evaluate
camerae2e run ./my-camera-study STUDY_ID optimize
camerae2e run ./my-camera-study STUDY_ID dataset_export
camerae2e run ./my-camera-study STUDY_ID report
```

Primary REST resources are under `/api/v2/projects`, `/studies`, `/jobs`, and
`/artifacts`. Convenience endpoints submit evaluate, explore, optimize,
candidate-validation, dataset, calibration, and report jobs. Job submission
returns HTTP 202 and a poll URL.

ADAS-specific resources are:

```text
GET  .../benchmark/status
GET  .../benchmark/manifest
POST .../benchmark/preflight
POST .../benchmark/run
POST .../benchmark/train-detector
POST .../requirements/evaluate
POST .../candidates/{case_id}/promote
```

`benchmark/run` accepts `compare_fidelity=true` to attach a same-scene L0/L1
rankability comparison. `GET .../calibrations/status` returns the current scoped
calibration pack.

An executed FDTD validation can automatically attach a consumable camera LUT
and rerun the promoted candidate on 200 scenes. RayOptics manifests remain
geometric evidence and TCAD lineage mismatches remain stale/calibration-required;
neither is treated as a directly consumable wave-optics camera LUT.

Migrate a v1 Workbench settings JSON without inventing missing detector scores:

```bash
camerae2e migrate ./my-camera-study ./legacy-settings.json
```

Import a legacy final-result DB as immutable proxy/calibration-required
artifacts:

```bash
camerae2e migrate ./my-camera-study ./camerae2e_db --db
```
