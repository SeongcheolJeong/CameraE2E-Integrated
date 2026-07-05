# Pixel CAD Template Library

This library creates common image-sensor pixel structures as explicit parametric CAD sources instead of scattering hidden proxy assumptions through solver commands.

It does not require FreeCAD to generate files. It uses Gmsh/OpenCASCADE headlessly and writes `STEP` and `BREP` files that FreeCAD can open for 3D inspection. The reference library also includes coarse `model.msh` files for 3D CAD mesh review.

## Why This Exists

The workbench should not keep stacking assumptions on top of assumptions. For common structures, use an explicit parameter source:

1. edit or extend a CAD template,
2. generate `model.step` / `model.brep`,
3. review the geometry in FreeCAD or another CAD viewer,
4. use `geometry_import.json` as the FDTD footprint source,
5. use `model.msh` for CAD mesh review,
6. use `tcad_mesh_from_cad_template.py` for a parameter-derived 2D TCAD bridge mesh,
7. use calibrated Gmsh/DEVSIM electrical meshes only after device geometry and process data are defined.

The templates still are not measured process geometry. They are controlled parametric references until measured mask, OCL profilometry, stack thickness, n,k, implant, and trap data are available.

## Generate Templates

Use the TCAD environment because it already has Gmsh:

```bash
.tcad-env/bin/python pixel_cad_template_library.py \
  --output-dir runs/pixel_cad_template_library_reference
```

Generate one template:

```bash
.tcad-env/bin/python pixel_cad_template_library.py \
  --output-dir runs/pixel_cad_template_library_reference \
  --template qpd_split_pd_2x2
```

Add or refresh selected templates while preserving existing UI/FCStd variants:

```bash
.tcad-env/bin/python pixel_cad_template_library.py \
  --output-dir runs/pixel_cad_template_library_reference \
  --append \
  --template dual_pd_x_1x1 \
  --template dual_pd_z_1x1
```

Write a coarse 3D mesh preview:

```bash
.tcad-env/bin/python pixel_cad_template_library.py \
  --output-dir runs/pixel_cad_template_library_reference \
  --template qpd_split_pd_2x2 \
  --mesh
```

## Included Templates

- `bayer_1x1_3x3`: Bayer 1x1 OCL, 3x3 neighborhood.
- `quad_2x2_ocl`: Quad Bayer 2x2 OCL supercell.
- `quad_2x2_ocl_3x3_neighborhood`: Quad Bayer 2x2 OCL, 3x3 OCL-group neighborhood for central-group crosstalk checks.
- `quad_2x2_ocl_5x5_crosstalk`: Quad Bayer 2x2 OCL, 5x5 OCL-group practical crosstalk domain.
- `nona_3x3_ocl`: Nona 3x3 OCL supercell.
- `dual_pd_x_1x1`: Green 1x1 dual-PD left/right split pixel.
- `dual_pd_z_1x1`: Green 1x1 dual-PD top/bottom split pixel.
- `pdaf_dual_x_shield_pair`: Paired dual-PD x-split PDAF pixels with left/right metal shield blockers.
- `qpd_split_pd_2x2`: Green QPD 2x2 split photodiode pixel with paired shield proxy.
- `qpd_split_pd_no_shield_2x2`: QPD control structure without metal shield.
- `mixed_1x1_2x2_3x3_boundary`: Mixed 1x1 / 2x2 / 3x3 OCL transition.

The reference catalog may also contain generated variants, for example lens
height, DTI width, or FCStd round-trip variants. Those are preserved by
`--append` and should remain traceable through `variant_source.json`.

`quad_2x2_ocl` is a compact 4x4-subpixel layout-review template with only
2x2 OCL groups. For minimum crosstalk kernels, use
`quad_2x2_ocl_3x3_neighborhood`: it contains 6x6 subpixels, i.e. a central 2x2
OCL group plus the surrounding 8 neighboring 2x2 OCL groups. For high-CRA,
long-range leakage, or camera-system kernel export checks, use
`quad_2x2_ocl_5x5_crosstalk`; it contains 10x10 subpixels, i.e. 5x5 OCL groups.

Each template directory contains:

- `template_parameters.json`: the editable source-of-truth parameters.
- `assumption_ledger.json`: tracked parametric assumptions, solver mapping, and measured-data blockers.
- `model.step`: FreeCAD-openable 3D CAD.
- `model.brep`: OpenCASCADE-native CAD.
- `model.msh`: coarse 3D Gmsh CAD review mesh with physical volume groups.
- `geometry_import.json`: FDTD-compatible OCL/CFA footprint import.
- `footprint_preview.svg`: fast 2D footprint review.

`uniform_green` in a template is intentionally normalized by the backend to
`--cfa-pattern uniform --color-channel green`. The raw value remains in the
template parameters so the source intent is still traceable.

The assumption ledger is the first file to inspect when deciding whether a
template is good enough for a simulation question. It should make remaining
proxy choices explicit instead of hiding them behind a CAD artifact.

## TCAD Bridge Scope

`tcad_mesh_from_cad_template.py` currently creates a parameter-derived 2D
electrical cross-section. It can generate either an x-depth or z-depth section
with `--section-axis auto|x|z`; `auto` uses z for `dual-z` templates and x for
the other common templates. This bridge can verify mesh/import wiring and
single-axis split current trends, but it is not a native 3D CAD electrical
solve.

Capability gates are therefore explicit:

- `dual_pd_x_1x1`: bridge/DD smoke can report left/right x-axis phase.
- `dual_pd_z_1x1`: bridge/DD smoke can report top/bottom z-axis phase through
  an auto-selected z-depth section. The DEVSIM contact names remain
  `cathode_left/right` internally, but the summary adds z-axis aliases.
- `qpd_split_pd_2x2` and QPD variants: bridge/DD smoke reports x-axis
  projection by default. A forced z-section can report z-axis projection, but
  full Q1-Q4 balance still needs a coupled 3D solve.

Do not interpret a numeric `photo_split_phase_x_proxy` as proof that every
split direction in the CAD template was solved.

For QPD templates, the workbench also provides a 3D quadrant terminal
weighting-potential smoke:

```bash
curl -s -X POST http://127.0.0.1:8766/api/cad/run-tcad-qpd-weighting-3d \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_2x2"}'
```

This creates `tcad_qpd_weighting_3d/summary.json`,
`qpd_weighting_3d.msh`, and `qpd_weighting_3d.svg`. It has four bottom
quadrant contacts and reports normalized Q00/Q10/Q01/Q11 weighting,
quadrant uniformity, and x/z phase from a 3D Laplace solve. Its
`full_q1q4_weighting_gate` can pass, but `full_q1q4_dd_gate` remains `CHECK`
because calibrated 3D drift-diffusion collection is still not implemented.
Metal shield optical effects are not included in this pure electrical
weighting solve.

After the 3D weighting solve and a 3D FDTD generation volume exist, the
workbench can compute a QPD `G*W` response surrogate:

```bash
curl -s -X POST http://127.0.0.1:8766/api/cad/run-qpd-gw-3d \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_2x2","integration_grid":"generation"}'
```

This creates `tcad_qpd_gw_3d/summary.json`,
`qpd_gw_3d_response.csv`, and `qpd_gw_3d_response.svg`. It interpolates the
3D terminal weighting potentials onto the FDTD generation grid and reports
normalized Q00/Q10/Q01/Q11 response, x/z phase, quadrant uniformity, and a
`full_q1q4_gw_gate`. This is a better QPD response proxy than averaging
terminal weights because the optical generation distribution is included.
It still leaves `full_q1q4_dd_gate` as `CHECK`; calibrated 3D drift-diffusion,
measured geometry/materials, and convergence are still required before using
the result as product LUT data.

## Practical CAD Strategy

Use the CAD template library as the geometry control point for repeated design
work:

1. keep common topologies as named templates,
2. keep numeric dimensions in `template_parameters.json`,
3. review the generated `STEP/BREP/FCStd` in FreeCAD,
4. create variants only through recorded scalar overrides or new named base
   templates,
5. pass the same template's `geometry_import.json` and derived mesh artifacts
   into FDTD/TCAD.

This avoids changing a drawing in one place while the solver silently uses a
different assumed geometry. FreeCAD is the review/edit surface; the controlled
parametric template remains the source of truth unless a measured CAD/GDS or
profilometry source replaces it.

When a CAD template is active in the Pixel Workbench, geometry-bearing stack
overrides are protected. Direct UI/API overrides for `geometry_um.*` and
`shield.mode` are ignored by the backend and reported in
`cad_template.ignored_stack_override_keys`; change those dimensions by creating
a CAD variant or by an FCStd round-trip variant. Non-geometry solver/material
overrides, such as a lens `n,k` sweep, can still be passed through the request.
The canonical run evidence is written to `solver_case.json`,
`kpi_summary.json`, and `workbench_job_summary.json` in the run output folder.

The UI now keeps common presets CAD-first by default. Bayer 1x1, Quad 2x2,
Nona 3x3, QPD 2x2, and sparse PDAF presets automatically select their matching
starter template before an active-design solver run. Presets without an exact
template are intentionally left without a CAD source until a named base
template or recorded variant exists.

For now the practical base set is:

- 1x1 Bayer image pixel neighborhood,
- Quad 2x2 OCL,
- Nona 3x3 OCL,
- mixed 1x1/2x2/3x3 OCL boundary,
- dual-PD x and z split pixels,
- paired PDAF shield pixel,
- QPD 2x2 split photodiode with and without shield.

This is a better foundation than ad hoc assumptions, but it is still not a
measured product stack. Lens sag, CFA rounding/taper, DTI/BDTI details,
pinned-PD/TG/FD implants, traps, and calibrated material/device parameters
must still come from measured or process-owned data before the result can be
called product-accurate.

## Review In FreeCAD

FreeCAD is optional for generation but useful for design review. On this
workstation FreeCAD 1.1.1 is installed at:

```text
/Users/seongcheoljeong/Applications/FreeCAD.app
```

The helper first looks for `~/Applications/FreeCAD.app`, then
`/Applications/FreeCAD.app`; otherwise it falls back to the default macOS
`open` behavior.

Check CAD viewer availability:

```bash
python3 cad_template_review.py --check-tools
```

List review-ready templates:

```bash
python3 cad_template_review.py --list
```

Print the command that would open a template:

```bash
python3 cad_template_review.py \
  --template qpd_split_pd_2x2 \
  --artifact step \
  --print-command
```

Open a generated template in FreeCAD:

```bash
python3 cad_template_review.py \
  --template qpd_split_pd_2x2 \
  --artifact step
```

Validate all generated STEP/BREP files through headless FreeCAD and write
native `.FCStd` review files:

```bash
python3 cad_template_review.py \
  --validate-freecad \
  --write-fcstd
```

This writes:

- `freecad_validation_report.json`
- `model.FCStd` in each template directory

The validation imports both `model.step` and `model.brep` with FreeCAD,
checks shape validity, solid count, volume, and compares the CAD bounding box
against `template_parameters.json`. On the current workstation the full
reference library passes: 15 / 15 templates, including base templates and
registered variants.

Refresh the CAD/template design-rule report without regenerating STEP/BREP or
overwriting UI-created variants:

```bash
.tcad-env/bin/python pixel_cad_template_library.py \
  --validate-only \
  --mesh
```

This keeps `template_library_manifest.json` intact and rewrites
`cad_template_validation_report.json`. The current design-rule checks verify:

- `geometry_import.json` schema, units, and template id match `template_parameters.json`.
- CFA cells cover the full pixel grid exactly once and match the requested Bayer/Quad/Nona/uniform pattern.
- OCL blocks stay inside the supercell, cover each pixel exactly once, and have aperture dimensions derived from pitch and edge gap.
- CAD volume counts match expected CFA/OCL/DTI/PD/shield topology, including QPD split-PD quadrant counts.

`model.FCStd` is a FreeCAD-native review package generated from the imported
shape. It contains:

- `ImportedTemplateShape`: the STEP-imported CAD shape.
- `TemplateParameters`: a spreadsheet copy of `template_parameters.json`.
- `ValidationSummary`: FreeCAD/design-rule status and rule names.

This makes the review file self-describing when opened outside the workbench.
It is still not a fully parametric FreeCAD design model. The parametric source
of truth remains `template_parameters.json` plus the generator.

The Pixel Workbench backend exposes the same local review path:

```bash
curl -s http://127.0.0.1:8766/api/cad/tools | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8766/api/cad/open \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_2x2","artifact":"step","prefer_freecad":true}'
curl -s -X POST http://127.0.0.1:8766/api/cad/validate-freecad \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_2x2","write_fcstd":true}'
```

In the UI, open the `Template` view and use `Open STEP in FreeCAD`,
`Open BREP`, `Open FCStd`, `Validate FreeCAD`, or `Open Assumptions` before
running the CAD template simulation. This keeps the geometry review,
assumption ledger, and solver run tied to the same template id.

## FCStd Parameter Round Trip

For scalar design edits, the `.FCStd` review package can be used as an edit
surface. Do not edit the base `model.FCStd` in-place for design exploration;
create a working copy first so the controlled template remains unchanged:

1. In the UI, use `Make FCStd Working Copy`, or call
   `/api/cad/create-fcstd-working-copy`.
2. Open the copied `.FCStd` file in FreeCAD.
3. Edit scalar values in the `TemplateParameters` spreadsheet, such as
   `lens_height_um`, `cfa_thickness_um`, `dti_width_um`, or
   `pd_depth_max_um`.
4. Save the edited working copy.
5. In the UI, keep the copied path in `FCStd import path`.
6. Use `Read FCStd Parameters` to preview scalar overrides.
7. Use `Create Variant From FCStd` to generate a registered CAD variant.

The backend also exposes this flow:

```bash
curl -s -X POST http://127.0.0.1:8766/api/cad/extract-fcstd-parameters \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_2x2"}'

curl -s -X POST http://127.0.0.1:8766/api/cad/create-fcstd-working-copy \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_2x2"}'

curl -s -X POST http://127.0.0.1:8766/api/cad/create-variant-from-fcstd \
  -H 'Content-Type: application/json' \
  -d '{
    "template_id": "qpd_split_pd_2x2",
    "fcstd_path": "runs/tmp_fcstd_roundtrip/qpd_lens666.FCStd",
    "id": "qpd_split_pd_fcstd_roundtrip_lens_666nm",
    "label": "QPD 2x2 FCStd round-trip lens 666 nm"
  }'
```

Only scalar `TemplateSpec` fields are imported. Topology edits such as
`ocl_blocks` are intentionally blocked, because those should become new base
templates rather than hidden spreadsheet edits.

## Create Variants From The UI

The backend also exposes controlled scalar-override variant creation:

```bash
curl -s -X POST http://127.0.0.1:8766/api/cad/create-variant \
  -H 'Content-Type: application/json' \
  -d '{
    "base_template": "qpd_split_pd_2x2",
    "id": "qpd_split_pd_ui_smoke_lens_690nm",
    "label": "QPD 2x2 UI smoke lens 690 nm",
    "overrides": {"lens_height_um": "0.690"}
  }'
```

In the UI, use `Template -> Create CAD Variant`. The form intentionally accepts
only scalar template parameters such as lens height, CFA thickness, passivation,
DTI width/depth, and PD margins. It does not allow OCL topology edits; those
must be added as new base templates so geometry assumptions stay explicit.

After creation, the variant is registered in the CAD catalog, gets its own
`variant_source.json` and `assumption_ledger.json`, and appears in the
`CAD Variant Comparison` test suite. A smoke suite remains a wiring/trend check:
`CHECK` status or grid-gate failure must not be treated as product accuracy.

When TCAD DD smoke is available for both the base and variant, the
`CAD Variant Comparison` suite reports both optical FDTD response delta and
electrical DD split-phase delta in the same variant table. This helps separate
optical-only changes, such as microlens height, from geometry/device changes
that should affect split-PD electrical balance.

For case-local coupled smoke, the suite feeds each FDTD
`tcad_generation_map_2d.npz` into the template-derived DEVSIM mesh. The default
coupled smoke uses `generation_map_scale=1.0e-3` and
`dd_relative_error=1.0e-5` so convergence tests the CAD/FDTD/DD data path
rather than claiming calibrated absolute photocurrent. Quantitative electrical
comparison must rerun without smoke scaling and with calibrated optical flux,
measured material/process data, and convergence gates.

## Generate TCAD Bridge From A Template

After reviewing or creating a template, generate a 2D DEVSIM-oriented bridge
mesh and run a DEVSIM import/potential smoke:

```bash
curl -s -X POST http://127.0.0.1:8766/api/cad/generate-tcad-bridge \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_ui_smoke_lens_690nm"}'
```

In the UI, use `Template -> Generate TCAD Bridge`. The generated files appear
under the template directory:

- `tcad_bridge_2d/split_pixel_2d.msh`
- `tcad_bridge_2d/derived_tcad_config.json`
- `tcad_bridge_2d/tcad_bridge_report.json`
- `tcad_bridge_2d/devsim_import_smoke/gmsh_pixel_2d_import_summary.json`

This verifies that DEVSIM can import and solve a potential-only smoke problem
on the parameter-derived mesh. It is still not a calibrated product TCAD mesh.

## Run TCAD Drift-Diffusion Smoke

After the bridge exists, run a proxy drift-diffusion smoke on the same Gmsh
mesh:

```bash
curl -s -X POST http://127.0.0.1:8766/api/cad/run-tcad-dd-smoke \
  -H 'Content-Type: application/json' \
  -d '{"template_id":"qpd_split_pd_ui_smoke_lens_690nm"}'
```

In the UI, use `Template -> Run TCAD DD Smoke`. This writes:

- `tcad_bridge_2d/devsim_smoke/summary.json`
- `tcad_bridge_2d/devsim_smoke/split_currents.csv`
- `tcad_bridge_2d/devsim_smoke/split_currents.png`
- `tcad_bridge_2d/devsim_smoke/node_maps.png`

The DD smoke verifies that the template-derived mesh can support the current
proxy split-PD electrical solve. It still uses proxy implants, traps, mobility,
and recombination unless measured inputs are loaded.

Inspect the assumptions before treating a CAD artifact as solver input:

```bash
python3 cad_template_review.py \
  --template qpd_split_pd_2x2 \
  --show-ledger
```

## Create A CAD Variant

For repeated design changes, do not manually copy STEP files. Generate a new
variant from a base template and keep the override record:

```bash
.tcad-env/bin/python cad_template_variant_create.py \
  --base-template qpd_split_pd_2x2 \
  --id qpd_split_pd_lens_high_8pct \
  --label "QPD 2x2 lens height +8%" \
  --set lens_height_um=0.710 \
  --set lens_edge_gap_um=0.070
```

Each generated variant includes:

- `variant_source.json`: base template id and parameter overrides.
- `assumption_ledger.json`: updated assumption ledger with override history.
- `model.step` / `model.brep`: FreeCAD-openable CAD.
- `model.msh`: coarse CAD review mesh when mesh generation is enabled.
- `geometry_import.json`: FDTD footprint import source.

## Use A Template In FDTD

Example for QPD:

```bash
.meep-env/bin/python meep_supercell_lut.py \
  --mode ocl-layout \
  --layout-nx 2 \
  --layout-nz 2 \
  --ocl-layout qpd_2x2_ocl:0:0:2:2 \
  --target-lens-id qpd_2x2_ocl \
  --collection-mode split-pd \
  --split-mode quad \
  --ocl-polygons @runs/pixel_cad_template_library_reference/qpd_split_pd_2x2/geometry_import.json \
  --cfa-polygons @runs/pixel_cad_template_library_reference/qpd_split_pd_2x2/geometry_import.json \
  --ocl-layout-name cad_template_qpd_2x2_field_anchor_smoke \
  --wavelengths-nm 550 \
  --color-channel green \
  --cfa-pattern quad \
  --cases center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0,edge20z:0:20:0:1:0:0,diag20:14.1421356:14.1421356:1:1:0:0 \
  --resolution 8 \
  --after-source-time 0.5 \
  --pml-um 0.45 \
  --grid-snap-y nearest \
  --output-dir runs/pixel_cad_template_library_reference/qpd_split_pd_2x2/fdtd_smoke
```

Smoke runs are only wiring checks. For trend/quantitative use, raise resolution and require convergence gates.

## Resolve A Workbench Request Without Running

The backend can resolve a UI/API request into the canonical solver case without
starting Meep. This is the fastest way to check whether a CAD template is the
active geometry authority and whether protected overrides were ignored:

```bash
curl -s -X POST http://127.0.0.1:8766/api/simulation/resolve-request \
  -H 'Content-Type: application/json' \
  -d '{
    "simulation_request": {
      "design": {"preset_label": "CAD authority check", "cad_template": {"template_id": "qpd_split_pd_2x2"}},
      "condition": {"wavelength_nm": 550, "color_channel": "green"},
      "solver": {
        "cad_template_id": "qpd_split_pd_2x2",
        "stack_overrides": {
          "geometry_um.lens_height": 9.99,
          "materials.lens": {"n": 1.61, "k": 0, "measured": false, "source": "local sweep"}
        }
      }
    }
  }'
```

The resolved `solver_case` should keep the CAD-template lens height while
listing `geometry_um.lens_height` under ignored protected keys, and should keep
the non-geometry material override.

## Generate A 2D TCAD Bridge Mesh

For QPD/split-PD electrical smoke checks:

```bash
.tcad-env/bin/python tcad_mesh_from_cad_template.py \
  --template-id qpd_split_pd_2x2 \
  --output-dir runs/pixel_cad_template_library_reference/qpd_split_pd_2x2/tcad_bridge_2d \
  --include-dti-oxide \
  --include-fd-contact \
  --include-tg-contact
```

This writes:

- `split_pixel_2d.msh`: DEVSIM-oriented 2D Gmsh mesh.
- `derived_tcad_config.json`: explicit derivation from `template_parameters.json`.
- `tcad_bridge_report.json`: mesh/contact/region evidence and limitations.

Potential-only import smoke:

```bash
.tcad-env/bin/python devsim_gmsh_pixel_import.py \
  --mesh runs/pixel_cad_template_library_reference/qpd_split_pd_2x2/tcad_bridge_2d/split_pixel_2d.msh \
  --dimension 2 \
  --output-dir runs/pixel_cad_template_library_reference/qpd_split_pd_2x2/tcad_bridge_2d/devsim_import_smoke
```

The current bridge verifies DEVSIM mesh import and potential solve. Drift-diffusion product accuracy still needs calibrated contacts, implant, trap, TG/FD, DTI/BDTI, mobility, and recombination data.

## Tool Roles

- FreeCAD: inspect/edit generated `STEP/BREP` 3D CAD manually.
- Gmsh/OpenCASCADE: generate headless CAD and coarse CAD review mesh artifacts.
- KLayout/GDS: remain the better source of truth for mask layout and hierarchy.
- Meep: optical FDTD using explicit footprint imports.
- DEVSIM: electrical simulation after calibrated device geometry/material data is available; current CAD-template bridge is import/potential-smoke level.

## Important Limit

These templates reduce uncontrolled assumptions, but they do not make the simulation product-accurate. The included `model.msh` files are CAD review meshes, not calibrated DEVSIM electrical meshes. The `tcad_bridge_2d` meshes are parameter-derived electrical smoke meshes, not full 3D product TCAD. Accuracy still requires measured geometry, measured material tables, calibrated electrical parameters, and convergence pass.
