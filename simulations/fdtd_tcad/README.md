# Image Sensor Pixel Workbench

Open-source research workbench for image-sensor pixel design studies. The
current stack combines CAD-template geometry, Meep optical simulations,
Gmsh/OpenCASCADE mesh generation, DEVSIM diagnostics, and a React UI for
pattern/OCL/CRA/crosstalk/PDAF-oriented analysis.

This repository is intended for practical design screening and workflow
validation. It is not product-LUT accurate until measured stack geometry,
measured optical n,k, calibrated implant/device parameters, and quantitative
convergence gates all pass.

## What This Repo Contains

- `pixel_workbench_server.py`: local backend that serves the UI and launches
  solver/test-suite jobs.
- `pixel_workbench_ui/`: React/Vite UI for design, simulation, and analysis.
- `pixel_cad_template_library.py`: parametric CAD template generator for common
  pixel/OCL/PDAF topologies.
- `meep_*.py`: optical FDTD runners for CRA, crosstalk, supercell, and
  microlens studies.
- `devsim_*.py` and `tcad_*.py`: DEVSIM/Gmsh diagnostics, coupling checks, and
  TCAD-readiness gates.
- `configs/`, `materials/`, `measured_profiles/`: reference proxy inputs and
  measured-data schema examples.
- `PIXEL_WORKBENCH_RUNBOOK.md`: detailed operating guide.

Generated outputs are intentionally ignored by Git under `runs/`. Local solver
environments such as `.meep-env/`, `.tcad-env/`, and UI dependencies under
`node_modules/` are also ignored.

## Quick Start

Generated assets under `runs/` are not committed. On a fresh checkout, build the
local CAD catalog and reference UI first:

```bash
cd /Users/seongcheoljeong/FDTD
python3 pixel_workbench_bootstrap.py
```

Then start the local backend:

```bash
cd /Users/seongcheoljeong/FDTD
python3 pixel_workbench_server.py --port 8766
```

Open:

```text
http://127.0.0.1:8766/runs/image_sensor_pixel_studio_reference/index.html
```

Use the `8766` backend URL for real solver execution. A static file server can
render the UI, but it cannot launch Meep/DEVSIM jobs.

If Gmsh is not available yet, run UI-only bootstrap with `--skip-cad`; the UI
will render, but CAD-template workflows and CAD-first simulation presets will
be incomplete until `runs/pixel_cad_template_library_reference` is generated.
Add `--cad-mesh` when you also need coarse `model.msh` CAD review meshes.

## Regenerate The Reference UI

When the React UI or studio payload changes:

```bash
cd /Users/seongcheoljeong/FDTD
python3 pixel_workbench_bootstrap.py
```

Equivalent manual commands:

```bash
cd /Users/seongcheoljeong/FDTD/pixel_workbench_ui
npm run build

cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python pixel_cad_template_library.py \
  --output-dir runs/pixel_cad_template_library_reference \
  --append

python3 image_sensor_pixel_studio.py \
  --config configs/image_sensor_pixel_studio_reference.json \
  --output-dir runs/image_sensor_pixel_studio_reference
```

Add `--mesh` to the CAD command when coarse 3D CAD review meshes are needed.

## Validate

Run syntax/build checks:

```bash
cd /Users/seongcheoljeong/FDTD
python3 -m py_compile \
  pixel_workbench_server.py \
  image_sensor_pixel_studio.py \
  pixel_cad_template_library.py \
  cad_template_variant_create.py

cd /Users/seongcheoljeong/FDTD/pixel_workbench_ui
npm run build
```

Run the browser functional test against the backend URL:

```bash
cd /Users/seongcheoljeong/FDTD/pixel_workbench_ui
npm run test:functional -- \
  --url http://127.0.0.1:8766/runs/image_sensor_pixel_studio_reference/index.html \
  --out ../runs/image_sensor_pixel_studio_reference/ux_functional_test_report.json \
  --screenshot ../runs/image_sensor_pixel_studio_reference/ux_functional_test.png
```

For solver-backed browser validation, add:

```bash
--solver --solver-timeout-ms 180000
```

## Typical Design Flow

1. Choose or create a CAD template in `Project > Template`.
2. Keep topology changes as new base templates. This includes `nx/nz`, OCL
   grouping, CFA pattern, split mode, and shield mode.
3. Use scalar variants for fixed-topology parameter changes such as pixel pitch,
   lens height, CFA thickness, passivation thickness, DTI width/depth, and PD
   margins. Regenerate CAD/mesh/FDTD/TCAD artifacts before comparing results.
   Pixel pitch changes are mixed-scale changes: x/z lattice-derived geometry
   follows pitch, while process thicknesses, gaps, DTI, and PD depth/margin
   parameters remain absolute unless explicitly overridden.
4. Run `Fast Preview`, `FDTD Detail`, or `Test Suite` depending on whether the
   task is a UI smoke check, a single optical case, or a matrix study.
5. Review KPI cards, convergence gates, crosstalk/field/PDAF charts, and linked
   JSON/CSV artifacts from the completed job.

For Quad Bayer 2x2 OCL, `pitch_um = 1.4um` is the subpixel pitch and the OCL
group pitch is `2.8um x 2.8um`. Crosstalk kernel studies need neighboring OCL
groups around the target group. Use the 3x3 OCL-group neighborhood template for
minimum central-to-8-neighbor checks, and use the 5x5 OCL-group crosstalk domain
for high-CRA or long-range leakage truncation checks instead of the compact 2x2
OCL layout-review template.

## Accuracy Position

The current practical decision path is:

```text
FDTD Optical -> 3D G*W surrogate -> DEVSIM diagnostic checks
```

DEVSIM DD is useful for mesh/contact/electrical sanity checks, but it is not yet
a calibrated pinned-PD/TG/FD/readout model. Product LUT use remains blocked
until measured geometry/material/device calibration and quantitative convergence
reports pass.

## More Documentation

- `PIXEL_WORKBENCH_RUNBOOK.md`: operation, suite artifacts, replay, and
  convergence commands.
- `CAD_TEMPLATE_LIBRARY_README.md`: CAD template library and FreeCAD workflow.
- `TCAD_README.md`: DEVSIM/TCAD assumptions and calibration track.
- `CAMERA_LUT_README.md` and `CAMERA_LUT_SCHEMA.md`: camera-system export
  package and schema.
- `MEASURED_TCAD_SCHEMA.md`: expected measured device/profile inputs.
