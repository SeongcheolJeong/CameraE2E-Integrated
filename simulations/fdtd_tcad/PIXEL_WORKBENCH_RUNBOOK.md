# Pixel Workbench Runbook

## Start the local solver backend

The repo does not commit generated files under `runs/`. On a fresh checkout, or
after deleting generated assets, bootstrap the reference CAD catalog and studio
before opening the backend URL:

```bash
cd /Users/seongcheoljeong/FDTD
python3 pixel_workbench_bootstrap.py
```

Use `python3 pixel_workbench_bootstrap.py --skip-cad` only for UI-only review
when the TCAD/Gmsh environment is not available yet. CAD-template workflows and
CAD-first solver presets require the generated
`runs/pixel_cad_template_library_reference` catalog.

Use `python3 pixel_workbench_bootstrap.py --cad-mesh` when you also need coarse
`model.msh` CAD review meshes; it is slower and produces much more Gmsh log
output.

```bash
cd /Users/seongcheoljeong/FDTD
python3 pixel_workbench_server.py --port 8766
```

Open:

```text
http://127.0.0.1:8766/runs/image_sensor_pixel_studio_reference/index.html
```

The static `8765` page can render the UI, but the `8766` backend URL is required
when the UI should launch Meep jobs and poll KPI results.

## Run from the UX

1. Open `FDTD Detail` or press `Run FDTD Detail`.
2. Choose one of the solver-backed examples:
   - `Quad Bayer + 2x2 OCL smoke`
   - `Nona 3x3 + 3x3 OCL smoke`
   - `Split-PD / QPD smoke`
3. Wait for the status to become `completed`.
4. Review `KPI Status`, `Grid Gate`, response/focal maps, and the run log.

By default, common presets are CAD-first:

- `Bayer + 1x1 OCL` uses `bayer_1x1_3x3`.
- `Quad Bayer + 2x2 OCL` uses `quad_2x2_ocl`.
- `Quad Bayer + QPD` uses `qpd_split_pd_2x2`.
- `Nona 3x3 + 3x3 OCL` uses `nona_3x3_ocl`.
- `Sparse Half-shield PDAF` uses `pdaf_dual_x_shield_pair`.

For Quad Bayer 2x2 OCL crosstalk, do not use the compact `quad_2x2_ocl`
layout-review template as the kernel domain. The `Crosstalk Kernel Practical`
suite runs `quad_2x2_ocl_5x5_crosstalk`, which is a 10x10-subpixel,
5x5-OCL-group CAD domain centered on `quad_4_4`. The OCL geometry comes from
CAD `geometry_import.json`; CFA is procedural `quad` pattern for this large
domain to avoid oversized per-cell CFA polygon imports in the Meep material
function.

Presets without an exact CAD source, such as custom supercell drafts, are left
without an active CAD template until a named base template or CAD variant is
created. This is intentional: avoid silently using a near-match geometry when
the design topology is not represented.

For CAD-template runs, also open:

- `solver_case.json`: canonical solver input after backend authority gates.
- `kpi_summary.json`: persisted parsed KPI, including CAD template metadata.
- `workbench_job_summary.json`: run-level pointers to request, solver case, KPI,
  and CAD source.

If a CAD template is active, the backend treats that template as the geometry
authority. Protected geometry overrides such as `geometry_um.*` and
`shield.mode` are ignored; change those dimensions by making a CAD variant or an
FCStd round-trip variant. Material/sweep overrides that are not geometry source
fields can still pass through the request.

Smoke examples are actual Meep runs, but intentionally low resolution for UX
testing. They should prove job execution, artifact generation, KPI parsing, and
viewer rendering. They should not be treated as quantitative sensor LUTs when
`Grid Gate` is `CHECK`.

## Validate the UX

```bash
cd /Users/seongcheoljeong/FDTD/pixel_workbench_ui
npm run test:functional -- \
  --url http://127.0.0.1:8766/runs/image_sensor_pixel_studio_reference/index.html \
  --out ../runs/image_sensor_pixel_studio_reference/ux_functional_test_report.json \
  --screenshot ../runs/image_sensor_pixel_studio_reference/ux_functional_test.png
```

Run the solver-backed browser test:

```bash
npm run test:functional -- \
  --url http://127.0.0.1:8766/runs/image_sensor_pixel_studio_reference/index.html \
  --out ../runs/image_sensor_pixel_studio_reference/ux_solver_test_report.json \
  --screenshot ../runs/image_sensor_pixel_studio_reference/ux_solver_test.png \
  --solver \
  --solver-timeout-ms 180000
```

The solver-backed browser test now checks that:

- active CAD-template designs run through Meep,
- `kpi_summary.json` is persisted and linked in the UI,
- CAD authority metadata survives in the persisted KPI,
- direct geometry stack overrides are ignored by the backend resolve gate.
- suite runs persist `suite_result.json` and `workbench_suite_summary.json`,
  including case-level artifact pointers.

## Test Suite Output Files

When a suite is run from the `Test Suite` view, the backend writes:

- `suite_result.json`: full suite KPI, charts, gates, and case table.
- `workbench_suite_summary.json`: compact run-level index with suite id, status,
  case count, gate count, output/log URLs, and per-case artifact pointers.
- `camera_system_export/`: camera-system research/trend package generated from
  the suite result. It includes field/PDAF/crosstalk/gate CSV files plus
  `camera_system_suite_export.json`.
- `camera_system_export/consumer_validation/`: downstream ingest validation and
  compact indexes. It includes `camera_system_suite_export_validation.json`,
  `camera_system_suite_export_field_query.csv`,
  `camera_system_suite_export_crosstalk_index.csv`, and
  `camera_system_suite_export_gate_summary.csv`.
- each case subdirectory:
  - `case_input.json`: selected suite case spec after tier overrides.
  - `case_command.json`: exact command and working directory for executable
    runners.
  - `solver_case.json`: canonical Meep solver input where the runner has one,
    including CAD-template geometry authority metadata.
  - `case_result.json`: case KPI/charts/artifact index.
  - solver output, logs, and case-specific KPI artifacts where that runner
    produces them.

The UI links both suite files after completion. For practical design comparison,
use `suite_result.json` for detailed chart/table data,
`workbench_suite_summary.json` as the stable index for later review, and a
case's `case_result.json` when investigating one design point without reopening
the full suite payload.

To regenerate the camera-system export from an existing suite result:

```bash
curl -s -X POST http://127.0.0.1:8766/api/simulation/export-camera-package \
  -H 'Content-Type: application/json' \
  -d '{
    "suite_result": "runs/ui_solver_tests/<suite>/suite_result.json"
  }'
```

To validate an existing camera-system export for downstream camera simulation:

```bash
curl -s -X POST http://127.0.0.1:8766/api/simulation/validate-camera-package \
  -H 'Content-Type: application/json' \
  -d '{
    "export_json": "runs/ui_solver_tests/<suite>/camera_system_export/camera_system_suite_export.json",
    "field_x": "0,0.5,1",
    "field_z": "0"
  }'
```

The validation pass means the package is structurally safe to ingest as
research/trend data. It does not make the data product-accurate; `product_lut_ready`
must remain false until measured stack/material/device calibration and
quantitative convergence gates pass.

## Quantitative Evidence Manifest

Use this when you need the current answer to “which evidence is actually good
enough, and what still blocks product LUT use?”:

```bash
python3 camera_system_quantitative_evidence.py \
  --config configs/image_sensor_pixel_studio_reference.json \
  --output-dir runs/camera_system_quantitative_evidence_reference
```

Or regenerate it through the local backend:

```bash
curl -s -X POST http://127.0.0.1:8766/api/simulation/quantitative-evidence \
  -H 'Content-Type: application/json' \
  -d '{
    "config": "configs/image_sensor_pixel_studio_reference.json",
    "output_dir": "runs/camera_system_quantitative_evidence_reference"
  }'
```

The manifest writes `camera_system_quantitative_evidence.json`,
`camera_system_quantitative_evidence.csv`,
`camera_system_quantitative_blockers.csv`, and a Markdown report. It indexes
existing optical convergence, crosstalk convergence, DEVSIM convergence, field
LUT validation, spectral coverage, research LUT artifacts, and the TCAD
accuracy gate. A `RESEARCH_READY_NOT_PRODUCT` result means the research flow is
usable for trend studies, while product LUT use is still blocked.

## Replay A Suite Case

Use `pixel_workbench_replay.py` to rerun an executable suite case from its
persisted `case_command.json`. By default, the tool replaces the original
`--output-dir` with a new `runs/replay/...` folder so the prior run is not
overwritten.

Dry-run the replay command:

```bash
python3 pixel_workbench_replay.py \
  runs/ui_solver_tests/<suite>/<case>/case_command.json \
  --dry-run
```

Replay into an explicit output folder:

```bash
python3 pixel_workbench_replay.py \
  runs/ui_solver_tests/<suite>/<case>/case_command.json \
  --output-dir runs/replay/<case>_rerun \
  --compare-source \
  --timeout-sec 180
```

Replay through the local backend API. Prefer the async job endpoint for UI or
longer trend/quantitative reruns because the browser request returns
immediately and normal job polling reports completion:

```bash
curl -s -X POST http://127.0.0.1:8766/api/simulation/replay-case-job \
  -H 'Content-Type: application/json' \
  -d '{
    "case_command": "runs/ui_solver_tests/<suite>/<case>/case_command.json",
    "compare_source": true,
    "timeout_sec": 180
  }'
```

Poll the returned job id through:

```bash
curl -s http://127.0.0.1:8766/api/simulation/jobs/<replay_job_id>
```

The completed replay job links `replay_manifest.json`,
`replay_comparison.json`, `workbench_replay_summary.json`, and the output
folder. The older synchronous endpoint is still available for short CLI/API
checks:

```bash
curl -s -X POST http://127.0.0.1:8766/api/simulation/replay-case \
  -H 'Content-Type: application/json' \
  -d '{
    "case_command": "runs/ui_solver_tests/<suite>/<case>/case_command.json",
    "output_dir": "runs/replay/<case>_rerun",
    "compare_source": true,
    "timeout_sec": 180
  }'
```

Replay from the UI:

1. Open `Test Suite`.
2. Run a suite case that produces `case_command.json`.
3. In the completed case table, press `Replay + Compare`; the UI starts an
   async replay job and polls it without replacing the suite result screen.
4. Open the returned `replay_manifest.json` and `replay_comparison.json` links.

The replay output folder contains `replay_manifest.json` with the source
`case_command.json`, resolved command, return code, elapsed time, and stdout
tail. With `--compare-source`, it also writes `replay_comparison.json`, comparing
common CSV/JSON outputs such as `camera_lut_summary.csv`,
`crosstalk_kernel_summary.csv`, crosstalk kernel CSVs, and convergence JSONs
against the source case directory. This is the preferred way to verify that a
saved suite case is actually reproducible before making design decisions from
it.

## Quantitative convergence path

For LUT-quality numerical checks, run convergence instead of smoke:

```bash
cd /Users/seongcheoljeong/FDTD
.meep-env/bin/python run_convergence_sweep.py \
  --mode ocl-2x2 \
  --wavelengths-nm 450,550,650 \
  --cases center:0:0:0:0:0:0,cra10x:10:0:0.5:0:0:0,edge20x:20:0:1:0:0:0 \
  --sparse-setting 90:8:0.45 \
  --sparse-setting 100:8:0.45 \
  --sparse-setting 90:12:0.45 \
  --sparse-setting 90:8:0.60 \
  --grid-snap-y nearest \
  --min-feature-pixels 2 \
  --min-si-wavelength-pixels 8 \
  --relative-tolerance 0.05 \
  --output-dir runs/manual_ocl2x2_convergence
```

Use `convergence_report.md` to decide whether numerical convergence passed.
Measured stack/material/device calibration is still required before calling the
result a product-accurate sensor LUT.
