# Image Sensor Pixel Studio UX Goal

This document translates public Ansys Lumerical/CHARGE image-sensor workflow
ideas into an open-source implementation target for this workspace. It is not a
copy of the commercial UI; it is a practical structure we can implement with
Meep, Gmsh, DEVSIM, static HTML viewers, CSV/JSON/NPZ artifacts, and future CAD
or layout import.

## Public Workflow Signals Used

- Ansys Lumerical FDTD modern UI:
  https://optics.ansys.com/hc/en-us/articles/36952912384403-Ansys-Lumerical-FDTD-Modern-User-Interface
  - Relevant structure: tabbed tool areas, viewport, object tree, result view,
    optimization/sweep window, script/editor workspace, materials, sources,
    monitors, mesh, resources, and run controls.
- Lumerical FDTD simulation object model:
  https://optics.ansys.com/hc/en-us/articles/52523416098835-Getting-Started-with-Lumerical-FDTD-for-Photonic-Integrated-Circuits-Part-1
  - Relevant structure: hierarchical object tree, property editing, material
    library, sources, monitors, geometry objects, mesh controls, resource
    configuration, and run status.
- Results Manager and Visualizer:
  https://optics.ansys.com/hc/en-us/articles/360034394974-Browsing-data-with-the-Results-Manager
  https://optics.ansys.com/hc/en-us/articles/360037222234-Using-the-data-visualizer-and-figure-windows
  - Relevant structure: selected objects expose available results, datasets can
    be opened in viewers, and result inspection should reduce ad hoc scripting.
- Lumerical datasets:
  https://optics.ansys.com/hc/en-us/articles/360034409554-Introduction-to-Lumerical-datasets
  - Relevant structure: monitor data should be structured by parameters and
    attributes rather than treated only as loose images or screenshots.
- CMOS image sensor angular response:
  https://optics.ansys.com/hc/en-us/articles/360042358574-CMOS-image-sensor-Angular-response-3D
  - Relevant structure: initial manual simulation, angle/polarization sweep,
    microlens-shift sweep, electrical weighting function, and G*W integration
    for QE/crosstalk.
- CMOS Sensor Camera characterization:
  https://optics.ansys.com/hc/en-us/articles/360062131614-CMOS-Sensor-Camera-Sensor-Characterization
  - Relevant structure: 3D electrical weighting, broadband optical simulation,
    CRA/MRA inputs, EQE vs angle/wavelength/pixel type, and camera-system
    export.
- CMOS optical simulation methodology:
  https://optics.ansys.com/hc/en-us/articles/360042851793-CMOS-Optical-simulation-methodology
  - Relevant structure: wavelength-scale pixel optics should be modeled with
    wave optics and spatial generation maps, not ray-only attribution.
- CMOS electrical simulation methodology:
  https://optics.ansys.com/hc/en-us/articles/360042358674-CMOS-Electrical-simulation-methodology
  - Relevant structure: pinned photodiode, transfer gate, floating diffusion,
    implants, and carrier transport are solver objects/equations, not just
    labels in a drawing.
- Parameter sweep utility:
  https://optics.ansys.com/hc/en-us/articles/360034922873-Parameter-sweep-utility
  - Relevant structure: explicit sweep parameters, result collection, and
    repeatable run tasks.
- CMOS optical simulation tips:
  https://optics.ansys.com/hc/en-us/articles/360042851673-CMOS-Optical-simulation-tips
  - Relevant structure: start coarse, then run convergence; do not treat a fast
    smoke mesh as accuracy evidence.

## Lumerical UX Findings

The public Ansys examples point to a UX pattern more important than the exact
window layout:

1. A project is an object graph.
   Geometry, materials, sources, monitors, solvers, analysis groups, sweeps,
   and results are selectable objects with properties and generated datasets.

2. Configuration and execution are separated.
   The user edits object properties first, then runs a single simulation,
   sweep, optimization, or analysis script. The run knows which parameters and
   result columns are part of the task.

3. Results are dataset-first.
   A selected object exposes available results with dimensions/values. The user
   can open 1D plots, 2D images/slices, 3D/vector plots, or export figures/data
   without writing ad hoc scripts for every inspection.

4. The CIS workflow is staged.
   Optical FDTD produces spatial field/transmission/generation data. Electrical
   CHARGE produces a collection weighting function. The analysis combines
   generation `G(r, lambda, theta, phi)` and weighting `W(r)` to produce
   IQE/EQE/crosstalk and camera-system exports.

5. A fast setup is only a setup.
   Coarse mesh smoke runs are encouraged for debugging, but convergence and
   calibration must pass before a result is treated as accuracy evidence.

## Target UX Structure

The local Studio should converge toward these panels:

1. Object Tree
   - Project
   - Process stack
   - Pixel geometry
   - Optical sources and CRA cases
   - Monitors and result exports
   - Electrical features, implants, TG, FD, DTI/BDTI, interfaces
   - Solver tasks and runbook commands
   - Design parameters and variants
   - Every node must identify its config owner, output artifacts, and whether
     edits are solver-wired or metadata-only.

2. Properties
   - Shows selected object metadata.
   - Must expose whether the parameter is measured, proxy, wired to solver, or
     metadata-only.
   - Must expose which solver stages become stale when the parameter changes.
   - Editable properties should write the relevant JSON/profile/stack config
     through a command builder or a controlled save action.

3. Viewers
   - 2D cross-section viewer for doping, potential, optical generation, and
     split current summaries.
   - 3D geometry viewer for process boxes and mesh/data exports.
   - Dataset visualizer for 1D spectra, angle response curves, 2D field/charge
     maps, arbitrary slices, and 3D/vector fields where the data supports it.
   - G*W and Camera LUT report viewers.
   - Future: direct CAD/layer viewer from GDS/STEP/gmsh physical groups.

4. Sweeps
   - Coarse smoke sweep for quick feedback.
   - Convergence sweep for result credibility.
   - Design variant sweep for changes such as microlens height, split gap,
     DTI width, fixed charge, and mobility.
   - Each sweep should record parameter values and collected result columns.
   - The sweep table should show run status, runtime hint, input hash,
     convergence state, and whether the result is eligible for camera LUT use.

5. Results Manager
   - Shows every file used by or generated by the project.
   - Distinguishes solver-native mesh/data from visualization-only derived
     artifacts.
   - Marks product-LUT readiness separately from framework readiness.
   - Should group outputs by object/result name, not just by filename.

6. Accuracy Gate
   - Blocks product LUT export unless measured stack geometry/n,k, measured or
     calibrated electrical profile, convergence, and calibration targets pass.
   - Should show which failures are physics accuracy blockers and which are only
     framework/plumbing blockers.

## Current Workspace Position

Implemented:

- Meep optical generation exports for center/edge CRA smoke cases.
- Gmsh-native 2D split-PD mesh path.
- DEVSIM profile-PPD split current extraction for center and edge20x.
- G*W reduction with analytic W_proxy and FEM Laplace W_mesh.
- DEVSIM-native pure-Laplace terminal weighting export
  `W_devsim_laplace` through `devsim_weighting_potential_2d.py`.
- Camera-system LUT export in CSV/JSON/NPZ.
- Static Studio shell with Object Tree, Properties, 2D/3D viewers, G*W, Camera
  LUT, native split runs, accuracy gate, runbook, and results manager.
- Design parameter registry and candidate variants in
  `configs/image_sensor_design_space_reference.json`.
- Variant materialization through `image_sensor_variant_builder.py`, which
  writes isolated stack/profile/project JSON files plus stage-by-stage run
  plans under `runs/image_sensor_design_variants_reference/`.
- Variant comparison through `image_sensor_variant_compare.py`, which records
  completed candidate runs against the baseline in CSV/JSON/HTML and surfaces
  the results in the Studio Design Space tab.
- Run and dataset management through `image_sensor_run_manager.py`, which
  records expected stage outputs, completed/missing/blocked stage state, and a
  structured dataset catalog for solver-native and derived files. It also marks
  completed stages as fresh or stale by comparing tracked input and output
  timestamps, so design changes can be routed to the stages that need rerun.
- Studio object-result browsing through `result_groups`, which groups the
  dataset catalog by solver/object and result role, exposes dataset count,
  native/derived state, viewer availability, primary open action, and product
  readiness in the Results Manager. Result groups are also attached under the
  Dataset Catalog object tree node so selecting a group exposes its dataset
  metadata in the Properties pane.
- Local variant orchestration through `image_sensor_variant_orchestrator.py`,
  which previews or executes materialized run-plan stages, writes per-command
  logs, refreshes comparison/run-manager/Studio artifacts, and exposes the last
  run plus history in the Studio Results Manager. The Studio Overview also
  translates the latest run-manager/orchestrator state into a next-action
  advisor with copyable preview/execute commands.
- Ad-hoc variant creation through `image_sensor_design_variant_create.py`, which
  validates design-space parameter edits, reports required rerun stages, writes
  isolated variant configs/run plans on request, and can refresh the Studio
  management views without launching solver stages. The Studio Design Space tab
  includes a command builder for this path, including range and metadata-only
  warnings.
- Variant builder writes are content-stable, so regenerating plans no longer
  makes completed stages stale when the rendered JSON/text is unchanged. Variant
  stack files also rewrite material `n,k` table paths to absolute source paths,
  so an isolated `inputs/stack_config.json` does not break relative material
  references.
- Orchestrator static preflight now checks material `n,k` tables referenced by
  stack configs in addition to executable/env/script/input/output checks.
- G*W LUTs now include `native_devsim`, `gw_proxy_ref_scaled`,
  `gw_mesh_ref_scaled`, and `gw_devsim_laplace_ref_scaled` methods when the
  DEVSIM weighting CSV is present.

Current executed reference snapshot:

- `lens_height_plus_8pct` completed `meep_fdtd`, `design_viewer`, `gw_lut`, and
  `studio`.
- `split_gap_plus_50pct` completed `gmsh_mesh`, `devsim_electrical`,
  `design_viewer`, `gw_lut`, and `studio`.
- `front_fixed_charge_stress` completed `devsim_electrical`, `design_viewer`,
  `gw_lut`, and `studio`.
- Run Manager summary: 14 stage rows, 14 complete, 14 fresh, 0 stale, 0 missing,
  0 failed, 209 existing datasets.
- The snapshot is useful for UX and pipeline validation only. `accuracy_ready`
  and `product_lut_ready` remain false by design.

Still missing for accuracy:

- Measured stack geometry and measured n,k for the target sensor.
- Calibrated drift-diffusion adjoint collection probability, not just W_proxy,
  independent Laplace W_mesh, or DEVSIM-native pure-Laplace W_devsim_laplace.
- TG/FD/interface trap/DTI/BDTI equations calibrated to measured targets.
- Mobility/recombination calibration.
- Broad wavelength/angle/polarization convergence.
- Real job queue/HPC control and pass/fail acceptance against calibrated
  measured targets.
- Interactive CAD/layout editing. Current 2D/3D viewers are useful for design
  review, but they are not yet a solver-native CAD editor.

## Orchestration Target

The current open-source implementation now has a local runner, not a full
commercial solver job manager:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_variant_orchestrator.py --all --next-needed
```

This previews the next missing or stale stage per candidate variant without
running heavy FDTD. Actual execution is opt-in:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_variant_orchestrator.py \
  --variant split_gap_plus_50pct \
  --stage studio \
  --rerun-complete \
  --execute
```

Heavy Meep/convergence stages remain blocked unless `--include-heavy` is passed.
That guard is intentional: a Studio button or clean run plan should not
accidentally spend hours on a coarse or unreviewed simulation.

The Run Manager freshness check is a design-input freshness check, not a physics
validation check. It catches cases where a tracked input file is newer than a
completed output. It does not prove convergence, calibration, or product
accuracy. The orchestrator consumes that signal through `--next-needed` and
`--next-stale`, so a design edit can flow from stale detection into a rerun plan.

The design-edit command path is:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_design_variant_create.py \
  --id custom_split_gap_050um \
  --label "Custom split gap 0.050um" \
  --param split_gap_um=0.05 \
  --materialize \
  --overwrite \
  --refresh
```

This produces design files and run plans, not calibrated results. Solver
execution remains explicit through the orchestrator.

The Studio Overview reads the latest orchestrator result and reports the
current next action. If a heavy Meep/convergence stage is blocked, it shows a
decision state and a preview command with `--include-heavy`; if a heavy stage
has already been dry-run with `--include-heavy`, it keeps that flag in the
explicit execute command. The same plan row includes static preflight evidence:
command parsing, executable/env/script checks, input-path checks, output-parent
checks, material `n,k` table checks for stack configs, and lightweight runtime
hints. This preflight helps prevent a bad local launch, but it is not
convergence, calibration, or product-LUT evidence.

## Next Implementation Goal

The next useful goal is not to copy Lumerical's window layout more closely. A
clean clone of the chrome would be misleading if properties do not drive solver
configs or if pretty viewers hide proxy-only physics. The right target is a
credible open-source pixel-design workbench with the same workflow contracts:

1. Object/property editing must write solver-owned config, not just metadata.
2. Every edit must declare stale solver stages before execution.
3. Every run must preserve input hashes, preflight, logs, outputs, and pass/fail
   gates.
4. Viewers must label solver-native mesh/data separately from visualization-only
   derived geometry.
5. Camera-system LUT exports must remain blocked until measured inputs,
   convergence, and calibration pass.

Near-term implementation goal:

1. Extend the object-result browser into a true dataset visualizer.
   The first grouped Result View exists. The next step is to let a selected
   dataset open through a typed viewer/action contract rather than only opening
   its primary linked file.

2. Add controlled property editing.
   The Studio should generate and optionally apply config edits for wired
   parameters, then immediately show the stale-stage plan before any solver
   launch.

3. Add the dataset visualizer layer.
   CSV/NPZ/VTK/VTU outputs should be opened through a common viewer contract:
   plot type, axes, slice dimensions, units, source object, and export path.

4. Add a run queue facade.
   The current orchestrator is enough as the backend. The UX should present
   dry-run plan, heavy-stage guard, preflight evidence, logs, freshness, and
   rerun actions as one job-management panel.

5. Add a CAD/layout import target, not a custom CAD editor first.
   GDS/OASIS for lateral masks plus STEP/STL/Gmsh physical groups for 3D/process
   geometry is a better next step than trying to build a full CAD tool inside
   the browser.

The highest-value solver feature is still a calibrated drift-diffusion adjoint
collection probability `W(x,y,z,bias,region)` from DEVSIM or an equivalent open
solver. `W_devsim_laplace` is a useful solver-native terminal weighting dataset,
but it intentionally does not include recombination, mobility, trap occupancy,
TG transfer, or FD transient behavior. Without calibration, CRA/split/OCL trends
can be explored, but the output should not be called an accuracy LUT.

## Implementation Rule

If a parameter is not wired to the solver, the UI must say so. A clean-looking
viewer is not enough evidence that the simulation result has changed.
