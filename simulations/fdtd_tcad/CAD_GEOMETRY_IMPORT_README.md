# CAD/GDS Geometry Import

This workflow converts 2D GDS mask polygons into `pixel_geometry_import_v1` JSON that the Meep LUT and crosstalk solvers can load with `@file.json`.

It is meant for practical geometry injection into the open-source workbench. It is not mask signoff, and it does not recover OCL 3D sag/profilometry, measured n,k, implants, TG/FD, traps, or electrical calibration.

For common structures that are not yet backed by measured GDS/CAD, use [CAD_TEMPLATE_LIBRARY_README.md](/Users/seongcheoljeong/FDTD/CAD_TEMPLATE_LIBRARY_README.md) to generate explicit `STEP/BREP` parametric templates instead of embedding ad hoc assumptions in solver commands.

## Install

Use the local Meep environment:

```bash
.meep-env/bin/python -m pip install -r requirements-geometry-import.txt
```

## Reference Conversion

Create a small reference GDS:

```bash
.meep-env/bin/python gds_pixel_geometry_import.py \
  --write-reference-gds runs/ui_solver_tests/gds_geometry_import/reference_pixel_masks.gds
```

Convert it to solver geometry JSON:

```bash
.meep-env/bin/python gds_pixel_geometry_import.py \
  runs/ui_solver_tests/gds_geometry_import/reference_pixel_masks.gds \
  --map-config configs/gds_pixel_geometry_import_map_reference.json \
  --output-json runs/ui_solver_tests/gds_geometry_import/pixel_geometry_from_gds.json \
  --report-json runs/ui_solver_tests/gds_geometry_import/gds_import_report.json \
  --preview-svg runs/ui_solver_tests/gds_geometry_import/gds_import_preview.svg
```

Check `gds_import_report.json` before using the converted geometry. It reports layer polygon counts, matched OCL/CFA counts, global bbox, warnings, and `PASS / CHECK / FAIL` validation status. `gds_import_preview.svg` is a visual sanity check of the mapped OCL/CFA footprints.

Run a solver smoke using the converted GDS geometry:

```bash
.meep-env/bin/python meep_supercell_lut.py \
  --mode ocl-layout \
  --layout-nx 2 \
  --layout-nz 2 \
  --ocl-layout imported_qpd:0:0:2:2 \
  --target-lens-id imported_qpd \
  --collection-mode split-pd \
  --split-mode quad \
  --ocl-polygons @runs/ui_solver_tests/gds_geometry_import/pixel_geometry_from_gds.json \
  --cfa-polygons @runs/ui_solver_tests/gds_geometry_import/pixel_geometry_from_gds.json \
  --ocl-layout-name gds_imported_geometry_qpd_smoke \
  --wavelengths-nm 550 \
  --color-channel green \
  --cfa-pattern quad \
  --cases center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0 \
  --resolution 8 \
  --after-source-time 0.5 \
  --pml-um 0.45 \
  --grid-snap-y nearest \
  --output-dir runs/ui_solver_tests/gds_geometry_import/solver_smoke
```

Smoke resolution is intentionally low and should report a grid gate `CHECK` or `FAIL`. For trend or quantitative work, increase resolution and require convergence report pass.

## Workbench Test Suite Pipeline

The workbench backend also has an end-to-end suite case that runs:

1. reference GDS generation,
2. GDS layer-map conversion to `pixel_geometry_import_v1`,
3. GDS import report and preview generation,
4. Gmsh TCAD bridge mesh generation from the GDS bbox,
5. Meep LUT execution using the converted JSON.

Run only that case through the local backend:

```bash
curl -sS -X POST http://127.0.0.1:8766/api/simulation/run-suite \
  -H 'Content-Type: application/json' \
  --data '{"suite_id":"ocl_mixed_boundary","tier":"smoke","case_ids":["gds_imported_geometry_lut_pipeline"]}'
```

The suite result reports `gds_import_case_count`, `gmsh_bridge_case_count`, `imported_geometry_case_count`, `cfa_polygon_case_count`, and `split_collection_case_count` so the CAD path is visible in the KPI summary.

Artifacts include:

- `gds_import_report.json`: layer-map validation report.
- `gds_import_preview.svg`: quick visual preview of the imported footprints.
- `gmsh_bridge_report.json`: bridge metadata and the explicit warning that this is not a native polygon-preserving mesh.
- `gmsh_mesh/mesh_metadata.json` and `gmsh_mesh/split_pixel_2d.msh`: Gmsh TCAD mesh artifacts.

## Layer Map

The reference map is [configs/gds_pixel_geometry_import_map_reference.json](/Users/seongcheoljeong/FDTD/configs/gds_pixel_geometry_import_map_reference.json).

Key fields:

- `ocl_layers`: maps GDS layer/datatype to OCL lens ids used by `--ocl-layout`.
- `cfa_layers`: maps GDS layer/datatype to `red`, `green`, or `blue` aperture polygons.
- `localize`: converts global GDS coordinates into local polygon coordinates. Default is `bbox-center`.
- `cfa_cell_pitch_um`: optional. Set this when a GDS contains multiple CFA cells per color and you need cell-specific `ix/iz` polygons.

GDS x/y coordinates are mapped to solver x/z coordinates in microns.

## Practical Limit

This import preserves 2D polygon footprints only. For accuracy-oriented image sensor LUTs, combine it with measured stack geometry, measured n,k tables, measured OCL surface maps, electrical calibration, and convergence-gated solver runs.

Gmsh is used here as a mesh generator for TCAD/DEVSIM bridge artifacts. It is not the CAD source of truth. The current bridge mesh is GDS-bbox-informed proxy geometry; it does not preserve arbitrary GDS mask polygon connectivity as the electrical solver mesh.
