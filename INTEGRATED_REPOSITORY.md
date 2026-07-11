# CameraE2E Integrated Repository

This repository is the self-contained CameraE2E management checkout. It is
intended to be cloned once and used as the local source of truth for:

- CameraE2E / pyisetcam source code, tests, examples, reports, and tools
- selected FDTD/TCAD scripts and compact handoff fixtures
- selected RayOptics UI/backend code and lens DB final-result assets
- packaged CameraE2E camera DB artifacts under `camerae2e_db`
- a local React/FastAPI CameraE2E Workbench for simulation, optimization,
  RAW dataset export, and report generation

The repository is deliberately not a product sign-off package. FDTD, TCAD, and
RayOptics assets here remain research/proxy inputs unless measured calibration,
vendor traces, and strict lineage gates are attached.

## CameraE2E v2

The primary application architecture is now the local Camera Design Decision
Platform described in `docs/camerae2e-v2-architecture.md`. It adds typed study
contracts, SQLite project/job persistence, content-addressed artifacts, fidelity
routing, solver adapters, scoped calibration, RAW dataset export, and automatic
decision reports under the `camerae2e_v2` Python namespace and `/api/v2` REST
surface.

The React Workbench and FastAPI runtime use v2 projects rather than
backend-memory `latest_*` results. Existing v1 Python modules remain for
migration and parity tests, but the server exposes only `/api/v2` resources.

## Layout

- `src/pyisetcam`: CameraE2E runtime code and public APIs
- `simulations/fdtd_tcad`: FDTD/TCAD scripts, configs, and small fixtures
- `simulations/rayoptics`: RayOptics app/backend and lens-package inputs
- `camerae2e-workbench`: local CameraE2E engineering Workbench UI/backend
- `camerae2e_db`: final-result lens and image-sensor DB artifacts
- `tools`: packaging, validation, reporting, and integration scripts
- `docs`, `reports`, `outputs`: project documentation and generated evidence
- `tests`: unit and parity tests used by the CameraE2E codebase

## Clone-And-Run Contract

After cloning, the default runtime lookup order is repository-local:

1. `camerae2e_db`
2. `simulations/fdtd_tcad`
3. `simulations/rayoptics`
4. optional sibling `CameraE2E-DB`
5. optional environment-variable overrides

External solver workspaces may still be attached when needed:

- `PYISETCAM_FDTD_ROOT`
- `PYISETCAM_RAYOPTICS_ROOT`
- `PYISETCAM_CAMERA_DB_ROOT`
- `PYISETCAM_EXTERNAL_FDTD_ROOT`
- `PYISETCAM_EXTERNAL_RAYOPTICS_ROOT`

## Workbench

Run the local CameraE2E Workbench from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[workbench,yolo]"
cd camerae2e-workbench
npm install
npm run backend
npm run dev
```

The backend launcher uses `python3` by default and honors the `PYTHON`
environment variable. It contains no developer-machine interpreter path.

KITTI data and trained detector weights are runtime evidence rather than source
assets. Point the Workbench at them with `CAMERAE2E_KITTI_ROOT` and
`CAMERAE2E_YOLO_MODEL`; no user-specific absolute path is embedded in runtime
discovery.

The Workbench calls the real CameraE2E FACA, optimization, DB/LUT status,
dataset export, and report APIs. It exposes analytic, FDTD LUT-backed,
RayOptics geometric, and TCAD calibration-required fidelity boundaries in the
UI instead of treating proxy assets as product sign-off evidence.

## Validation

Run the integrated repository check from the repository root:

```bash
python3 tools/check_integrated_repository.py --write-manifest
```

The command writes `integrated_repository_manifest.json` and verifies:

- required code, simulation, and DB paths exist
- runtime helper defaults resolve to repository-local assets
- no file exceeds GitHub's 100 MB regular-file limit
- no nested `.git` directories were copied
- core physics simulation manifest generation works without external paths

Manifest size and GitHub file-limit checks use tracked plus non-ignored files,
so local caches, projects, datasets, and training runs are reported separately
from the clone payload.

## Known Boundaries

- Historical manifests and reports may preserve absolute source paths as
  provenance strings. Those strings are not runtime dependencies.
- Full product-grade FDTD/TCAD decks, measured calibration, and vendor HW traces
  are not implied by this repository.
- DNG compliance, measured ISP latency sign-off, and wave-optics sign-off remain
  separate milestones.
