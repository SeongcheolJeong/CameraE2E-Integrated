# CameraE2E Lens/Sensor DB

This repository contains final-result-only data artifacts used by CameraE2E for
camera information lookup and runtime simulation inputs.

## Contents

- `lens_db/CameraE2E_Lens_DB_v9_20260627/data/lens_patents/`: RayOptics-derived lens
  patent SQLite DB, CameraE2E manifest, inventory, and geometric PSF NPZ assets.
- `fdtd_tcad/sensor_db/`: image sensor catalog and generated stack configs.
- `fdtd_tcad/image_sensor_db/`: enriched sensor catalog and TCAD candidate report.
- `fdtd_tcad/fixtures/`: small FDTD/TCAD handoff fixtures with known proxy lineage.
- `fdtd_tcad/runs/`: small compatibility aliases so existing CameraE2E default
  path discovery works without copying full solver runs.

## CameraE2E setup

```bash
export PYISETCAM_CAMERA_DB_ROOT=/absolute/path/to/CameraE2E-DB
export PYISETCAM_LENS_DB_ROOT=$PYISETCAM_CAMERA_DB_ROOT/lens_db/CameraE2E_Lens_DB_v9_20260627
export PYISETCAM_FDTD_ROOT=$PYISETCAM_CAMERA_DB_ROOT/fdtd_tcad
```

`PYISETCAM_CAMERA_DB_ROOT` is enough for recent CameraE2E code.  The explicit
`PYISETCAM_LENS_DB_ROOT` and `PYISETCAM_FDTD_ROOT` exports are kept for older
tools and scripts.

## Truth Boundary

This is not a product sign-off database.  RayOptics PSFs are geometric
ray-histogram PSFs, not diffraction/wave-optics validation.  FDTD/TCAD assets
are lookup/proxy handoff artifacts unless measured calibration and strict
lineage gates are attached.  The current small TCAD fixture intentionally keeps
the known FDTD-LUT/generation-map lineage mismatch visible.
