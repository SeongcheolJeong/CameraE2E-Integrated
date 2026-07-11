# CameraE2E Integrated User Manual

## 1. Intended Use

CameraE2E is a local decision workbench for camera-system researchers. It supports
requirement definition, camera simulation, sensitivity studies, parameter optimization,
candidate validation, virtual RAW generation, and evidence reports.

Use it for relative trade studies and reproducible research. Do not interpret an
analytic, public-derived, geometric, or uncalibrated solver result as product sign-off.

## 2. Install and Launch

Required software:

- Git
- Python 3.12
- Node.js 20 or newer and npm

Recommended one-command setup:

```bash
git clone https://github.com/SeongcheolJeong/CameraE2E-Integrated.git
cd CameraE2E-Integrated
./tools/workbench.sh bootstrap
```

The launcher writes only local runtime state under `.camerae2e/`. Projects are stored
under `camerae2e-workbench/projects/` by default. Both paths are ignored by Git.
Set `CAMERAE2E_PYTHON=/path/to/python3.12` when that interpreter is not on `PATH`.

For manual setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[workbench,yolo,dev]"
cd camerae2e-workbench
npm ci
```

Then run these in separate terminals from `camerae2e-workbench/`:

```bash
npm run backend
npm run dev
```

## 3. First Run Without External Data

1. Open the Workbench and use the default ADAS or general project.
2. Select a study and inspect its mission requirements and baseline camera.
3. Run baseline evaluation with analytic fidelity.
4. Review geometry, clipping, signal, optics, and fidelity gates.
5. Run sensitivity or raw-quality optimization only when the selected target is ready.
6. Export a small RAW dataset from a validated candidate.
7. Generate the decision report and inspect provenance and warnings.

Without a detector and labels, YOLO perception optimization is disabled. This is
intentional: CameraE2E never substitutes an image-quality proxy for detector mAP.

## 4. Workbench Workflow

### Requirements and Configure

Define the camera mission before searching parameters. The baseline includes HFOV,
pixel pitch, sensor geometry, CFA/QE profile, exposure, OCL/CRA behavior, lens PSF,
ISP settings, and fidelity policy. Advanced settings expose binning, HW ISP timing,
TCAD collection mode, CCM evidence, and custom search axes.

Important controls:

| Control | Interpretation |
|---|---|
| HFOV | Scene coverage and object pixel density |
| Pixel pitch | Sampling, photon area, noise/full-well trade-off |
| CFA and QE | Spectral sampling and demosaic behavior |
| OCL/CRA | Pixel-stack angular response and field mismatch |
| PSF radius | Optical blur support used by the selected model |
| Exposure | Signal, motion risk, and clipping trade-off |
| CCM | Color transform; free fitting requires calibration evidence |
| Fidelity | Analytic search, LUT-backed ranking, solver evidence, or calibration |

### Baseline Evaluation

`Run Simulation` evaluates one configured camera. It produces stage outputs, scalar
metrics, requirement gates, fidelity badges, random seed, and artifact lineage. It does
not search alternative configurations.

### Sensitivity and Optimization

Sensitivity changes one or more declared axes to identify influential parameters.
Optimization evaluates multiple candidates under objectives and hard constraints.
Only candidates evaluated at the final evidence budget can become a winner.

For ADAS YOLO optimization, the platform runs source-model and geometry preflight,
camera-output rankability gates, successive-halving search, perturbation robustness,
and paired bootstrap comparison. `indistinguishable` means the evidence does not
support a meaningful winner.

### Candidate Validation

Use solver or higher-fidelity assets on shortlisted candidates rather than across the
entire broad search. A missing asset may fall back to L0 only when proxy fallback is
explicitly allowed and recorded.

### RAW Dataset Factory

The dataset exporter reruns the selected candidate rather than copying a preview. Its
portable output is:

```text
manifest.json
metadata.jsonl
raw/*.npz
rgb/*.png
labels/*.json
optional raw_tiff/*.tiff
optional stages/*.npz
```

The manifest records configuration, scene selection, seed, source hashes, code revision,
fidelity, and validation status. Labels are transformed with the same geometry mapping
as the camera output. CameraE2E does not invent labels for unknown objects.

### Decision Report

The report summarizes requirements, baseline, search space, gates, ranked candidates,
uncertainty, fidelity boundaries, selected artifacts, and dataset outputs. Treat warning
and `not_evaluable` fields as part of the result, not presentation noise.

## 5. KITTI and YOLO Setup

Set absolute local paths before starting the backend:

```bash
export CAMERAE2E_KITTI_ROOT=/data/kitti-yolo
export CAMERAE2E_YOLO_MODEL=/models/kitti-yolo.pt
./tools/workbench.sh restart
```

Expected minimum layout:

```text
kitti-yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Labels may be KITTI object rows or Ultralytics YOLO-normalized rows supported by the
current adapter. Keep image/label stems aligned. The model, labels, and class mapping
must describe the same task. Use the Workbench asset/readiness status before starting
optimization.

## 6. Projects, CLI, and Reproducibility

Create and inspect a portable project:

```bash
.venv/bin/camerae2e init /path/to/my-project --preset adas
.venv/bin/camerae2e doctor /path/to/my-project
.venv/bin/camerae2e artifacts /path/to/my-project
```

Each project contains `project.toml`, `project.db`, content-addressed artifacts, runs,
exports, and reports. For a reproducible comparison, preserve the project, benchmark
manifest, model hash, source hashes, seed, configuration revision, and Git commit.

Useful path overrides:

| Variable | Purpose |
|---|---|
| `CAMERAE2E_PROJECTS_ROOT` | Persistent Workbench project location |
| `CAMERAE2E_PYTHON` | Explicit Python 3.12 interpreter used during setup |
| `CAMERAE2E_KITTI_ROOT` | KITTI/YOLO image and label root |
| `CAMERAE2E_YOLO_MODEL` | Detector checkpoint |
| `PYISETCAM_CACHE_ROOT` | Upstream ISETCam asset cache |
| `PYISETCAM_UPSTREAM_ROOT` | Existing upstream asset snapshot |

## 7. Verification Before Sharing Results

```bash
./tools/workbench.sh check
.venv/bin/python -m pytest
cd camerae2e-workbench && npm run build
```

Before publishing a result, also verify that the project doctor passes, all external
asset licenses permit the intended sharing, and report wording matches the fidelity
actually used.
