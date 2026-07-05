# Camera LUT Simulation Notes

This workspace contains Meep-based optical proxy simulations for image-sensor
camera-system LUT generation.

## Main Script

```bash
micromamba run -p /Users/seongcheoljeong/FDTD/.meep-env \
  python /Users/seongcheoljeong/FDTD/meep_supercell_lut.py --mode split-pd-1x1
```

Supported modes:

- `split-pd-1x1`: 1x1 supercell split photodiode model.
- `ocl-2x2`: 2x2 on-chip-lens/pixel supercell.
- `ocl-3x3`: 3x3 on-chip-lens/pixel supercell.

## Optional Optical Shield

The default imaging-pixel stack uses `shield.mode=off`; `metal_edge_width` stays
as a geometric mask dimension but does not create metal in the optical stack by
itself. Use explicit shield modes only for PDAF or aperture-mask variants:

```bash
--shield-mode off        # normal imaging pixel, no metal optical shield
--shield-mode pdaf_left  # left half masked
--shield-mode pdaf_right # right half masked
--shield-mode pdaf_pair  # alternating left/right mask over x-indexed pixels
--shield-mode edge       # legacy edge aperture mask
```

`camera_lut_long.csv`, `camera_lut_summary.csv`, `camera_lut.json`, and the main
NPZ exports include `shield_mode` / `shield_mask_edge_width_um` so camera-system
LUT consumers can reject mixed optical-stack assumptions.

## Sweep Axes

Use comma-separated wavelength and case lists:

```bash
--wavelengths-nm 450,550,650
--cases center:0:0:0:0:0:0,edge20:20:0:1:0:-0.18:0
```

Case format:

```text
name:cra_x_deg:cra_z_deg:field_x_norm:field_z_norm:lens_shift_x_um:lens_shift_z_um[:aperture_shift_x_um[:aperture_shift_z_um]]
```

The lens/aperture shift is explicit geometry in microns. Replace default values
with the actual OCL/CFA/PD shift rule for the sensor.

## Exported Files

Each run writes:

- `camera_lut_long.csv`: one row per sweep point and collection region.
- `camera_lut_summary.csv`: one row per sweep point.
- `camera_lut.json`: schema, geometry, cases, regions, and summaries.
- `camera_lut.npz`: dense response tensor for camera-system ingestion.
- `camera_lut_pupil_rays.csv`: per-ray diagnostics when finite-pupil integration is enabled.
- `tcad_generation_profile_1d.csv`: FDTD-derived depth profile for DEVSIM 1D import.
- `tcad_generation_profile_1d.npz`: dense version of the same TCAD profile.
- `tcad_generation_map_2d.npz`: FDTD-derived `G(x, depth)` map for DEVSIM 2D import.
- `tcad_generation_volume_3d.npz`: FDTD-derived `G(x, depth, z)` volume for future 3D TCAD import.
- `response_maps.png`: region response map previews.
- `focal_maps.png`: Si-top focal-plane intensity previews.

Primary camera response column:

```text
response = flux_calibrated_integral(Im(epsilon_Si) * |E|^2 dV over collection region)
```

The absolute scale is calibrated to the full Si top/bottom flux absorption
estimate. Regional top/bottom flux differences remain diagnostic columns only
because small monitor regions can include lateral power flow and may be unstable
at low resolution.

The TCAD profile export uses:

```text
generation_cm3_s = incident_photon_flux_cm2_s * absorption_fraction_per_cm
```

Set `--incident-photon-flux-cm2-s` to match the illumination condition. The
default is a smoke-test scale, not a calibrated camera exposure.

The 2D TCAD map export preserves lateral/depth variation:

```text
tcad_generation_map_2d.npz:
  generation_cm3_s[case, x, depth]
  x_um[x]
  depth_um_from_si_top[depth]
```

This is the preferred import for 2D split-PD collection simulation because it
does not use analytic lateral gaussian shaping.

## Stack And Material Inputs

The default runnable stack config is:

```text
/Users/seongcheoljeong/FDTD/configs/sensor_stack_proxy_1p4um.json
```

Use `--stack-config` to point to a product-specific stack file and
`--color-channel red|green|blue` to select the CFA material table. The proxy
CFA/OCL/passivation tables now include public-source anchors where available,
but they are still not target-product measurements. See
`/Users/seongcheoljeong/FDTD/PUBLIC_SENSOR_STACK_SOURCES.md`.

## CRA Cone / Pupil Integration

By default the script runs one chief ray per case. To integrate over a finite
pupil cone:

```bash
--f-number 2.8 --pupil-samples 3
```

This samples a uniform disk in direction-cosine space and averages the region
responses. Increase `--pupil-samples` only after single-ray convergence is under
control.

## Convergence Automation

Run resolution/time/PML sweeps with:

```bash
micromamba run -p /Users/seongcheoljeong/FDTD/.meep-env \
  python /Users/seongcheoljeong/FDTD/run_convergence_sweep.py \
  --mode split-pd-1x1 \
  --cases center:0:0:0:0:0:0,edge20:20:0:1:0:0:0 \
  --resolutions 16,24,32 \
  --after-source-times 25,40,60 \
  --pml-um 0.45,0.60,0.80
```

The report flags response drift versus the highest setting and reports negative
signed flux diagnostics. The Si absorption diagnostic uses the +Y flux-plane
convention as bottom-plane flux minus top-plane flux, normalized by incident
power. Passing this report is numerical evidence only, not a guarantee that
material or process parameters are accurate.

For more robust Fourier-transform convergence, enable Meep field-decay
termination:

```bash
--decay-by 1e-4 --decay-check-time 50
```

## Quantitative Use

The smoke outputs in `runs/supercell_lut_*_smoke` are format/flow checks. For
quantitative LUTs, rerun with convergence sweeps, for example:

```bash
--resolution 16
--resolution 24
--resolution 32
```

and increase `--after-source-time` until total and regional responses settle.

The fixed camera-system schema is documented in
`/Users/seongcheoljeong/FDTD/CAMERA_LUT_SCHEMA.md`.

## CameraE2E Multi-Sensor Package

The current multi-sensor CameraE2E package is generated under:

```text
/Users/seongcheoljeong/FDTD/runs/camera_e2e_sensor_lut_package
```

Refresh the non-HPC handoff artifacts without rerunning long FDTD/TCAD solver
jobs:

```bash
cd /Users/seongcheoljeong/FDTD
python3 run_camera_e2e_package_pipeline.py --include-failed --skip-rebuild
```

The package is research/trend/load-test data, not product-calibrated sensor
data. CameraE2E loaders should preserve these gates and read order:

1. `camera_e2e_sensor_deliverable_summary/`
2. `camera_e2e_usage_policy/`
3. `camera_e2e_lut_source_integrity/`
4. `camera_e2e_flat_sensor_bundle/` for one JSON per sensor, or
   `camera_e2e_consumer_bundle/` for table-oriented loading
5. `camera_e2e_import_contract/` to verify every requested CameraE2E item
   resolves through the selected loader path
6. `camera_e2e_canonical_payload/` for adapter-ready per-sensor payload JSONs
7. `camera_e2e_runtime_bundle/` and domain LUTs only after source/gate checks

The first per-sensor review table is:

```text
runs/camera_e2e_sensor_lut_package/camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.csv
```

It lists each sensor's recommended loader path, row counts, use scope,
uncertainty bands, source-integrity coverage, and product blockers.

For one-file-per-sensor loading, each JSON under
`camera_e2e_flat_sensor_bundle/sensors/` includes
`objective_fulfillment.requirement_rows`: a 21-row map from each requested
CameraE2E item to the in-file `flat_json_pointer`, source class, calculation
method, uncertainty band, and product blocker.

For table-oriented loading, each JSON under
`camera_e2e_consumer_bundle/sensors/` now carries the same
`objective_fulfillment.requirement_rows` map plus `source_integrity` rows,
`source_tables`, and `join_keys`. This is the safer entrypoint when CameraE2E
will keep LUTs in separate CSV tables instead of loading a flat per-sensor JSON.

For adapter-side import validation, use:

```text
runs/camera_e2e_sensor_lut_package/camera_e2e_import_contract/camera_e2e_import_contract.json
```

It resolves all per-sensor `flat_json_pointer` entries and emits a 147-row
`camera_e2e_import_contract_by_requirement.csv` table. Current expected state is
147 resolved rows, 147 research-loadable rows, and 0 product-loadable rows.

For the most direct CameraE2E adapter payload, use:

```text
runs/camera_e2e_sensor_lut_package/camera_e2e_canonical_payload/camera_e2e_canonical_payload.json
```

It points to one JSON per sensor under `camera_e2e_canonical_payload/sensors/`.
Those JSON files reorganize the flat model into `optical_color`,
`pixel_electrical`, `readout_raw`, and `module_coupling`, while preserving
source integrity, method provenance, uncertainty, and product blockers.

The most compact requirement-level audit table is:

```text
runs/camera_e2e_sensor_lut_package/camera_e2e_lut_source_integrity/camera_e2e_lut_source_integrity_matrix.csv
```

It joins each `slug + requirement_id` with the source class, calculation method,
solver/external/proxy dependency, primary uncertainty band, research/product
gates, and next action. Typical current source classes include
`solver_anchor_plus_cfa_si_proxy`, `cfa_si_proxy_response_prior`,
`low_res_solver_or_compact_surrogate`, `electrical_prior_seed`,
`readout_raw_prior_seed`, and `module_coupling_prior`.

For a shorter 21-row review by requested objective item, use:

```text
runs/camera_e2e_sensor_lut_package/camera_e2e_objective_trace/camera_e2e_objective_trace_by_requirement_summary.csv
```

This summarizes each CameraE2E requirement across sensors with loader pointers,
source-class distribution, calculation-method distribution, uncertainty range,
and product blockers.

The expected current validation status is:

```text
RESEARCH_VALID_PRODUCT_BLOCKED
```

That means the package is coherent enough for CameraE2E plumbing, sensitivity,
and trend studies, while product-mode queries intentionally return zero allowed
rows until measured stack/material/CRA, electrical/readout/noise/module
calibration, and high-resolution convergence gates pass.

## TCAD Collection Coupling

The optical LUT script can export a DEVSIM generation profile:

```text
tcad_generation_profile_1d.csv
tcad_generation_profile_1d.npz
```

The 1D import path is a file-level smoke test. For split-PD, PD pixel, 2x2 OCL,
and 3x3 OCL response, use the 2D map import path as the current
collection-side prototype:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_split_pd_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case center \
  --generation-profile-wavelength-nm 550 \
  --electrical-model proxy-pinned-split-pd \
  --output-dir runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke
```

This produces left/right cathode photo-current deltas and a
`photo_split_phase_x_proxy`. It is useful for checking sign, direction, and data
plumbing into camera-system simulation.

Do not treat the current 2D TCAD result as a product-accurate LUT. The optical
input now comes from Meep `G(x, depth)`, but the electrical model is still an
analytic proxy for pinning, split collection columns, center isolation, and side
DTI rather than a calibrated pinned photodiode or OCL pixel array.

The measured-profile, Gmsh import, and calibration framework is documented in:

```text
/Users/seongcheoljeong/FDTD/MEASURED_TCAD_SCHEMA.md
```

For camera-system use, run the TCAD accuracy gate before treating any TCAD
collection output as a LUT input:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_accuracy_gate.py \
  --profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --split-summary runs/devsim_split_pd_2d_reference_profile_center_gmsh_native/summary.json \
  --split-summary runs/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native/summary.json \
  --gmsh-summary runs/devsim_gmsh_reference_import_2d/gmsh_pixel_2d_import_summary.json \
  --gmsh-summary runs/devsim_gmsh_reference_import_3d/gmsh_pixel_3d_import_summary.json \
  --convergence-report runs/convergence_public_anchor_smoke/convergence_report.json \
  --calibration-result runs/tcad_calibration_reference_profile/calibration_result.json \
  --targets-csv measured_profiles/reference_cmos_ppd_1p4um/calibration_targets_synthetic.csv \
  --calibration-target-report runs/tcad_calibration_target_report_reference/calibration_target_report.json \
  --transport-sensitivity-report runs/tcad_transport_sensitivity_reference/transport_sensitivity_report.json \
  --transport-calibration-target-report runs/tcad_calibration_transport_target_report_reference/calibration_target_report.json \
  --optical-stack-summary runs/optical_stack_evidence_reference/optical_stack_summary.json \
  --weighting-summary runs/devsim_weighting_potential_2d_reference/weighting_potential_2d_summary.json \
  --weighting-csv runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.csv \
  --gw-manifest runs/tcad_gw_coupling_reference/gw_coupling_manifest.json \
  --output-dir runs/tcad_accuracy_gate_reference_profile
```

The current reference profile reports `framework_ready=true` and
`accuracy_ready=false`. Use it for sign, direction, and pipeline development;
do not use it as a product sensor LUT.
