# CameraE2E Integrated

CameraE2E Integrated is a local, research-grade camera design decision platform. It
combines the `pyisetcam` imaging pipeline, camera and lens databases, analytic models,
FDTD/TCAD assets, RayOptics geometric PSFs, a persistent experiment service, and a
React Workbench in one repository.

The platform is intended for camera-system trade studies, reproducible configuration
optimization, and virtual RAW dataset generation. It is **not** a product sign-off
tool unless measured calibration evidence is attached for the relevant stage.

## What You Can Run

| Available inputs | Supported workflow | Important boundary |
|---|---|---|
| Repository only | Synthetic scenes, analytic/LUT camera simulation, RAW export, reports | Useful for workflow validation and relative comparisons |
| RGB scenes without labels | Camera re-capture and image-quality studies | Input is a display-RGB scene proxy, not recovered source RAW or spectra |
| KITTI images/labels + YOLO model | ADAS detector preflight and camera optimization | Model and benchmark validity determine the perception claim |
| Measured sensor/lens/ISP evidence | Scoped calibration and L3 comparison | Only validated stages may be described as calibrated |

No KITTI dataset, trained detector weights, vendor measurements, or proprietary solver
decks are downloaded or committed automatically.

## Architecture

```text
Requirements -> Baseline -> Sensitivity -> Optimization
             -> Candidate Validation -> RAW Dataset -> Decision Report

React Workbench / CLI
          |
CameraE2E v2 service + persistent projects
          |
Scene -> Optics -> Sensor -> ISP -> Metrics / Perception
          |
Analytic models | Camera DB | RayOptics | FDTD LUT | TCAD artifacts
```

Fidelity is always reported as `L0_analytic`, `L1_lut`, `L2_solver`, or
`L3_calibrated`. A higher label is not inferred from the presence of a file alone.

## Quick Start

Requirements: Git, Python 3.12, and Node.js 20 or newer.

```bash
git clone https://github.com/SeongcheolJeong/CameraE2E-Integrated.git
cd CameraE2E-Integrated
./tools/workbench.sh bootstrap
```

`bootstrap` creates `.venv`, installs the Python Workbench dependencies, runs
`npm ci`, validates the integrated repository, and starts both local servers.
If Python 3.12 is not on `PATH`, set `CAMERAE2E_PYTHON=/path/to/python3.12`.

- Workbench: http://127.0.0.1:5175
- API documentation: http://127.0.0.1:8010/docs
- Logs and PID files: `.camerae2e/`

Later sessions only require:

```bash
./tools/workbench.sh start
./tools/workbench.sh status
./tools/workbench.sh stop
```

Manual installation and server commands are documented in the
[User Manual](docs/user-manual.md).

## Optional ADAS Assets

Perception optimization intentionally remains disabled until a real model and matching
labels are available:

```bash
export CAMERAE2E_KITTI_ROOT=/absolute/path/to/kitti-yolo
export CAMERAE2E_YOLO_MODEL=/absolute/path/to/kitti-trained.pt
./tools/workbench.sh restart
```

The expected dataset layout is described in the user manual. A generic COCO YOLO model
can be used for engineering experiments, but it must not be reported as a KITTI-trained
ADAS benchmark.

## Documentation

- [User Manual](docs/user-manual.md)
- [한국어 사용자 매뉴얼](docs/user-manual.ko.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Research and Fidelity Boundaries](docs/research-boundaries.md)
- [CameraE2E v2 Architecture](docs/camerae2e-v2-architecture.md)
- [Integrated Repository Layout](INTEGRATED_REPOSITORY.md)
- [Contributing](CONTRIBUTING.md)
- [pyisetcam Development Status Archive](docs/pyisetcam-development-status.md)

## Verification

```bash
./tools/workbench.sh check
.venv/bin/python -m pytest
cd camerae2e-workbench && npm run build
```

The repository self-check verifies required integrated assets, local path portability,
GitHub file-size limits, and the declared physics-asset layout.

## License

Source code in this repository is available under the [MIT License](LICENSE). External
datasets, model weights, patents, upstream ISETCam assets, and solver outputs may have
separate terms; verify them before redistribution.
