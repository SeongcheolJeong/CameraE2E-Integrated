# CameraE2E Simulation Workspaces

This directory is the monorepo home for CameraE2E-adjacent simulation source.

Included:

- `fdtd_tcad/`: FDTD, Meep, TCAD/DEVSIM, sensor-stack, and handoff source/config files.
- `rayoptics/`: RayOptics workbench source/config files.

Final Lens/Sensor DB artifacts are kept out of this source tree.  They are
packaged as final-result-only runtime inputs in:

- `../camerae2e_db/` inside this repository when a small vendored mirror is useful.
- `/Users/seongcheoljeong/Documents/CameraE2E-DB` as the standalone DB repository.

Regenerate both with:

```bash
python3 tools/package_camerae2e_db_repository.py --target ../CameraE2E-DB --clean
python3 tools/package_camerae2e_db_repository.py --target camerae2e_db --clean
```

Excluded by policy:

- expensive/generated solver runs such as `runs/`
- Python virtual environments and JS `node_modules/`
- zipped packages, screenshots, RTF notes, cache directories, and generated web builds

CameraE2E integration code should import or reference these files through
`pyisetcam.physics_simulation`, `pyisetcam.physics_pipeline`, `pyisetcam.fdtd_sensor`,
`pyisetcam.tcad_sensor`, and `pyisetcam.lens_patents`. Large regenerated outputs should
remain reproducible artifacts, not silent source files.

For runtime DB discovery, set `PYISETCAM_CAMERA_DB_ROOT` to the standalone DB
repo, or rely on the in-repository `camerae2e_db` mirror.  Older tools can use
`PYISETCAM_LENS_DB_ROOT=$PYISETCAM_CAMERA_DB_ROOT/lens_db/CameraE2E_Lens_DB_v9_20260627`
and `PYISETCAM_FDTD_ROOT=$PYISETCAM_CAMERA_DB_ROOT/fdtd_tcad`.

Refresh from the current local workspaces with:

```bash
python tools/import_camerae2e_simulation_workspaces.py
python tools/render_camerae2e_physics_simulation_manifest.py
```

The manifest report is written to:

- `reports/camerae2e_goal/physics_simulation_manifest.json`
- `reports/camerae2e_goal/physics_simulation_manifest.html`
