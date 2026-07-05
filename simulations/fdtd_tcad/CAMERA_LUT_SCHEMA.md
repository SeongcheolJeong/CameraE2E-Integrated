# Camera LUT Schema

Schema id:

```text
camera_supercell_optical_lut_v2
```

Primary response model:

```text
si_volume_absorption_flux_calibrated_v1
```

## Files

- `camera_lut_long.csv`: canonical table for camera simulator ingestion.
- `camera_lut_summary.csv`: sweep-level totals and split-PD phase proxies.
- `camera_lut.npz`: dense arrays for fast loading.
- `camera_lut.json`: metadata, sensor stack, regions, cases, and notes.
- `camera_lut_pupil_rays.csv`: ray-level diagnostics for finite-pupil runs.
- `tcad_generation_profile_1d.csv`: depth-only TCAD generation profile.
- `tcad_generation_profile_1d.npz`: dense depth-only TCAD generation profile.
- `tcad_generation_map_2d.npz`: `G(x, depth)` map for 2D DEVSIM import.
- `tcad_generation_volume_3d.npz`: `G(x, depth, z)` volume for future 3D TCAD import.

The finite-array crosstalk runner writes a separate schema:

```text
camera_crosstalk_full_array_fdtd_v1
```

Files:

- `crosstalk_output_kernel.csv`: binned output-cell crosstalk kernel.
- `crosstalk_raw_pd_kernel.csv`: raw physical-PD absorption kernel.
- `crosstalk_kernel_summary.csv`: one row per mode/case/wavelength/resolution.
- `crosstalk_kernel.json`: manifest with stack/material/source/DTI metadata.
- `crosstalk_convergence.json`: truncation and resolution convergence checks.
- `crosstalk_kernel_heatmap.png`: visual diagnostic only.

## Canonical Axes

Each row is identified by:

- `mode`: `split-pd-1x1`, `ocl-2x2`, or `ocl-3x3`
- `color_channel`: `red`, `green`, `blue`, or `clear`
- `wavelength_nm`
- `field_x_norm`, `field_z_norm`
- `cra_x_deg`, `cra_z_deg`
- `case`
- `region_id`

`camera_lut.npz` stores:

- `response`: shape `(sweep_index, region_index)`
- `case`: string array for `sweep_index`
- `wavelength_nm`
- `field_x_norm`, `field_z_norm`
- `cra_x_deg`, `cra_z_deg`
- `region_id`
- `color_channel`

When TCAD export is enabled by the standard runner, it also writes:

- `tcad_generation_profile_1d.csv`
- `tcad_generation_profile_1d.npz`
- `tcad_generation_map_2d.npz`
- `tcad_generation_volume_3d.npz`

The TCAD profile schema is:

```text
tcad_generation_profile_1d_v1
```

Key columns:

- `depth_um_from_si_top`
- `absorption_fraction_per_um`
- `absorption_fraction_per_cm`
- `generation_cm3_s`
- `incident_photon_flux_cm2_s`

This profile is a 1D collapse of the Si volume absorption over `x/z`; it is
intended for DEVSIM 1D coupling smoke.

The 2D map schema is:

```text
tcad_generation_map_2d_x_depth_v1
```

Key arrays:

- `generation_cm3_s`: shape `(sweep_index, x_index, depth_index)`
- `absorption_fraction_per_um2`: shape `(sweep_index, x_index, depth_index)`
- `x_um`
- `depth_um_from_si_top`
- `case`
- `wavelength_nm`
- `cra_x_deg`, `cra_z_deg`

This is the preferred input for the current 2D split-PD DEVSIM path because it
preserves lateral optical asymmetry without analytic gaussian shaping.

The 3D volume schema is:

```text
tcad_generation_volume_3d_v1
```

Key arrays:

- `generation_cm3_s`: shape `(sweep_index, x_index, depth_index, z_index)`
- `absorption_fraction_per_um3`: shape `(sweep_index, x_index, depth_index, z_index)`
- `x_um`
- `depth_um_from_si_top`
- `z_um`

This is exported for future 3D electrical import; the current DEVSIM script
consumes the 2D map.

## Camera-System Uncertainty LUT

The native-DEVSIM research LUT can be wrapped with completed stress-variant
bounds for camera-system simulation before measured calibration data exists:

```text
camera_system_uncertainty_lut_v1
```

Files:

- `camera_system_uncertainty_lut.json`: nominal/min/max response rows plus
  source stress variants and remaining accuracy blockers.
- `camera_system_uncertainty_lut.csv`: flat ingest table for camera simulation.
- `camera_system_uncertainty_lut.html`: review report.
- `camera_system_field_lut.json`: dense `field_x_norm` interpolation table for
  camera simulation.
- `camera_system_field_lut.csv`: flat dense field table.
- `camera_system_field_lut.html`: review report for the dense field table.
- `camera_system_field_lut.npz`: compressed array dataset for direct camera
  simulation ingestion. It stores the same dense rows as typed arrays and can be
  loaded with `numpy.load(..., allow_pickle=False)`.
- `field_lut_query.json` / `field_lut_query.csv`: consumer-side validation and
  arbitrary field-position query output from `camera_system_field_lut_query.py`.

Important row columns:

- `case`, `wavelength_nm`, `cra_x_deg`, `field_x_norm`
- `nominal_total_response_a_per_cm`
- `min_total_response_a_per_cm`, `max_total_response_a_per_cm`
- `nominal_split_phase_x`, `min_split_phase_x`, `max_split_phase_x`
- `nominal_left_response_a_per_cm`, `nominal_right_response_a_per_cm`
- `min_left_response_a_per_cm`, `max_left_response_a_per_cm`
- `min_right_response_a_per_cm`, `max_right_response_a_per_cm`
- `stress_source`, `stress_row_count`, `bound_method`
- `product_lut_ready`

When `bound_method=independent_total_split_stress_envelope_v1`, total response
and split phase are independent stress bounds. Left-channel min/max are computed
from the worst left-channel combinations, not by pairing the same min/min and
max/max split rows:

```text
left_min  = 0.5 * total_min * (1 - split_max)
left_max  = 0.5 * total_max * (1 - split_min)
right_min = 0.5 * total_min * (1 + split_min)
right_max = 0.5 * total_max * (1 + split_max)
```

The `min_*` and `max_*` columns include the nominal value as a valid point in
the envelope. This avoids the common failure mode where all stress variants move
one direction and the nominal row falls outside the reported uncertainty band.

The dense field LUT has schema:

```text
camera_system_field_lut_v1
```

It interpolates the uncertainty anchor rows over `field_x_norm`; the current
reference data has `field_z_norm=0` only. The default interpolation is
`piecewise_linear_3_anchor` from `center`, `cra10x`, and `edge20x`. Piecewise
linear interpolation is used deliberately to avoid polynomial overshoot before
measured field-response calibration exists.

The NPZ export includes:

- axis arrays: `field_x_norm`, `field_z_norm`, `cra_x_deg`, `cra_z_deg`,
  `wavelength_nm`
- nominal response arrays: `nominal_total_response_a_per_cm`,
  `nominal_split_phase_x`, `nominal_left_response_a_per_cm`,
  `nominal_right_response_a_per_cm`
- min/max response arrays for total, split, left, and right
- metadata arrays: `schema`, `artifact_role`, `product_lut_ready`,
  `numeric_columns`, `columns`, `interpolation_method`, `anchor_cases`,
  `bound_method`, and `accuracy_blockers`

The query/validator output has schema:

```text
camera_system_field_lut_query_v1
```

It validates schema, finite numeric values, sorted/unique field axes,
nonnegative total response, split bounds inside `[-1, 1]`, nominal values inside
min/max envelopes, nominal left+right equal to total, and JSON/NPZ numeric
consistency. Query rows recompute left/right responses from interpolated total
and split bounds so the consumed rows remain algebraically consistent.

This artifact is suitable for risk-range camera-system simulation. It is not a
product accuracy LUT unless the accuracy gate also passes with measured stack,
measured n,k, measured implant/profile sources, calibrated transport, and
measured calibration targets.

## CameraE2E Sensor Ingest Export

The TechInsights-derived multi-sensor package can be converted into a
CameraE2E-ingestable research package:

```text
camera_e2e_ingest_manifest_v1
camera_e2e_sensor_field_response_lut_v1
camera_e2e_crosstalk_status_lut_v1
camera_e2e_ingest_lut_query_v1
camera_e2e_compact_crosstalk_lut_v1
camera_e2e_combined_lut_query_v1
camera_e2e_lut_readiness_audit_v1
camera_e2e_runtime_bundle_v1
camera_e2e_runtime_query_v1
camera_e2e_prior_seed_models_export_v1
camera_e2e_prior_seed_model_v1
camera_e2e_electrical_readout_tables_export_v1
camera_e2e_sensor_models_export_v1
camera_e2e_sensor_model_v1
camera_e2e_color_response_export_v1
camera_e2e_color_response_model_v1
camera_e2e_material_tables_export_v1
camera_e2e_cfa_provenance_audit_v1
camera_e2e_cfa_db_tables_v1
camera_e2e_module_coupling_export_v1
camera_e2e_coverage_matrix_export_v1
camera_e2e_consumer_bundle_v1
camera_e2e_consumer_sensor_manifest_v1
camera_e2e_flat_sensor_bundle_v1
camera_e2e_flat_sensor_model_v1
camera_e2e_import_contract_v1
camera_e2e_sensor_import_contract_v1
camera_e2e_canonical_payload_v1
camera_e2e_canonical_sensor_payload_v1
camera_e2e_flat_sensor_query_v1
camera_e2e_analysis_report_v1
camera_e2e_consumer_query_v1
camera_e2e_use_scope_summary_v1
camera_e2e_mesh_confidence_audit_v1
camera_e2e_field_execution_pack_v1
camera_e2e_crosstalk_support_audit_v1
camera_e2e_crosstalk_batch_priority_v1
camera_e2e_crosstalk_execution_pack_v1
camera_e2e_capability_profile_v1
camera_e2e_lut_trust_assessment_v1
camera_e2e_uncertainty_budget_v1
camera_e2e_response_trace_v1
camera_e2e_response_example_v1
camera_e2e_method_provenance_v1
camera_e2e_lut_source_integrity_v1
camera_e2e_sensor_deliverable_summary_v1
camera_e2e_handoff_manifest_v1
camera_e2e_handoff_loader_validation_v1
camera_e2e_objective_acceptance_audit_v1
camera_e2e_sensor_probe_v1
camera_e2e_closure_plan_v1
camera_e2e_product_closure_summary_v1
camera_e2e_usage_policy_v1
camera_e2e_adapter_examples_v1
camera_e2e_adapter_smoke_v1
camera_e2e_objective_trace_v1
camera_e2e_pipeline_run_v1
```

Files:

- `camera_e2e_ingest_manifest.json` / `.csv`: package manifest with research and
  production ingest flags.
- `camera_e2e_field_response_lut.json` / `.csv`: multi-sensor field-response
  rows keyed by `slug`, field position, CRA, microlens shift, wavelength, and
  row-level evidence. The canonical grid is generated from the field design
  cases; additional quantitative Meep color/wavelength points are exported as
  supplementary rows instead of being discarded.
- `camera_e2e_field_response_lut.npz`: compressed typed-array version of the
  field-response LUT for high-throughput CameraE2E ingestion.
- `camera_e2e_quantitative_execution_plan.csv`: full-sweep point count and runtime
  estimate by sensor/solver. The estimate includes `color_channels`,
  `fdtd_cell_volume_um3`, `estimated_volume_factor`, and
  `fdtd_domain_factor` so large-pitch sensors and finite-array crosstalk domains
  are not misrepresented as short local jobs.
- `camera_e2e_quantitative_point_queue.csv`: point-sized solver queue. RGB sensors
  use `red;green;blue`; monochrome/clear sensors use `clear` and therefore require
  fewer field/wavelength points. Each row carries `fdtd_cell_volume_um3`,
  `estimated_volume_factor`, and `fdtd_domain_factor` alongside the concrete
  command. For crosstalk, `fdtd_domain_factor` reflects the required finite
  raw-pixel simulation domain, for example a 1x1 Bayer 3x3 output kernel needs a
  5x5 raw-pixel simulation window after guard cells.
- `camera_e2e_crosstalk_status_lut.csv`: crosstalk execution/status rows. This
  is a status artifact unless converged crosstalk fractions are present.
- `camera_e2e_compact_crosstalk_lut/camera_e2e_compact_crosstalk_kernel_lut.csv`:
  CHECK-gated compact output crosstalk kernels generated from field-FDTD focal
  spot metrics plus TCAD/DTI geometry priors. This is intended for CameraE2E
  research/trend sensitivity runs while finite-array crosstalk FDTD is
  resource-limited.
- `camera_e2e_ingest_sensor_summary.csv`: per-sensor row counts and gates.
- `index.html`: human review report.
- `camera_e2e_ingest_query.json` / `.csv`: consumer-side validation and
  arbitrary field-position query output from `query_camera_e2e_ingest_lut.py`.
- `camera_e2e_combined_query*/camera_e2e_combined_query.json` / `.csv`:
  field-response query rows joined with nearest compact crosstalk kernels from
  `query_camera_e2e_combined_lut.py`.
- `camera_e2e_readiness_audit/`: strict readiness report from
  `audit_camera_e2e_lut_readiness.py`. This is the artifact to check before
  allowing any CameraE2E ingest. It separates `research_ingest_gate` from
  `production_lut_gate` per sensor and emits blocker rows.
- `camera_e2e_runtime_bundle/`: consolidated runtime package from
  `export_camera_e2e_runtime_bundle.py`. This joins field response, compact
  crosstalk kernels, readiness gates, and uncertainty bounds into one manifest,
  CSV pair, NPZ file, and HTML report for CameraE2E integration.
- `camera_e2e_runtime_query*/`: safe runtime lookups from
  `query_camera_e2e_runtime_bundle.py`. The query tool interpolates runtime
  response rows, blends crosstalk kernels, and enforces research/product mode
  gates.
- `camera_e2e_prior_seed_models/`: research-only electrical/readout/module
  seed values from `build_camera_e2e_prior_seed_models.py`. These provide
  configurable CameraE2E placeholders for conversion gain, FWC, dark current,
  DSNU/PRNU, temporal noise, ADC/black level, FPN, defects, binning/remosaic,
  alignment, and simple vignetting. They are always
  `evidence_level=prior_seed_not_measured` and must not raise production gates.
- `camera_e2e_electrical_readout_tables/`: row-based electrical/readout export
  from `export_camera_e2e_electrical_readout_tables.py`. It expands prior seed
  JSONs into `camera_e2e_electrical_noise_lut.csv` over temperature, exposure,
  and signal level; `camera_e2e_readout_gain_lut.csv` over analog gain, digital
  gain, and ADC bit depth; `camera_e2e_binning_remosaic_lut.csv` for binned
  mode signal/noise/crosstalk redefinition; and a per-sensor summary CSV.
  The electrical table now includes a conservative geometry-scaled diffusion
  prior for `charge_collection_electrical_crosstalk_gate`,
  `electrical_collection_efficiency_prior`,
  `electrical_crosstalk_fraction_prior`, min/max crosstalk bounds, and
  `electrical_diffusion_length_um`. These rows are CameraE2E research
  placeholders only. Product use requires measured conversion gain, FWC, dark
  current, DSNU/PRNU, read/reset/SF/ADC noise, FPN, defect maps, readout timing,
  mode/register calibration, and calibrated DEVSIM/TCAD charge collection.
- `camera_e2e_sensor_models/`: per-sensor CameraE2E model manifests from
  `export_camera_e2e_sensor_models.py`. This merges stack/CFA proxy n,k,
  optical QE DB evidence, runtime spectral response rows, crosstalk kernels,
  TCAD structure/profile context, module CRA validation, and explicit
  electrical/readout missing-data gates into one JSON per sensor plus summary
  and requirement-matrix CSV files. The current handoff also links the
  `camera_e2e_color_response/` spectral table and RGB-to-XYZ seed, carries
  lens-vs-sensor CRA mismatch tolerance summaries, and points to the module
  coupling LUT artifacts. Final per-sensor JSONs include
  `optical_color.cfa_provenance` plus summary columns
  `cfa_provenance_class`, `cfa_assumption_gate`, and `color_accuracy_gate`, so a
  CameraE2E consumer that loads only the sensor model still sees whether color
  rows are sensor-specific proxy, monochrome/clear proxy, or generic RGB
  fallback.
- `camera_e2e_color_response/`: color-response export from
  `export_camera_e2e_color_response.py`. It emits a 400-700 nm, 25 nm
  `camera_e2e_spectral_response.csv` table and a rough
  `camera_e2e_color_matrix_seed.csv` RGB-to-XYZ seed. The rows are derived from
  CFA proxy transmission and runtime center anchors where available. Monochrome
  sensors export `clear` spectral rows and mark RGB CCM as not applicable.
  Both CSVs carry `cfa_provenance_class`, `cfa_assumption_gate`,
  `generic_rgb_fallback_detected`, and matrix rows also carry
  `color_accuracy_gate`. `gate=CHECK` in these files means the row is loadable
  for research plumbing; `color_accuracy_gate=MISSING` or
  `cfa_assumption_gate=MISSING` must block color-accuracy use.
- `camera_e2e_material_tables/`: explicit material input export from
  `export_camera_e2e_material_tables.py`. It emits
  `camera_e2e_material_nk_lut.csv` and `camera_e2e_material_summary.csv`.
  Rows include CFA transmission proxy rows from
  `image_sensor_db/optical_qe_db`, FDTD material n,k rows for CFA/OCL,
  passivation, and silicon from generated stack configs, thickness/source
  provenance, and research/product gates. Unknown CFA patterns use a generic
  CFA-library fallback and remain product-blocked. The material rows and summary
  also carry `cfa_provenance_class` and `cfa_assumption_gate` so FDTD setup code
  can distinguish sensor-specific CFA proxy input from generic RGB fallback.
- `camera_e2e_cfa_provenance/`: CFA source/fallback audit from
  `audit_camera_e2e_cfa_provenance.py`. It emits
  `camera_e2e_cfa_provenance.json`,
  `camera_e2e_cfa_provenance_by_sensor.csv`,
  `camera_e2e_cfa_provenance_checks.csv`, and an HTML review page. The audit
  joins `image_sensor_db/optical_qe_db`, material summary, spectral response,
  and color-matrix rows to classify each sensor as
  `SENSOR_SPECIFIC_RGB_PROXY`, `RGB_PROXY_DEFAULT_THICKNESS`,
  `MONO_CLEAR_PROXY`, `GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN`, or another
  explicit CFA state. `GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN` means the RGB rows
  are CameraE2E plumbing priors only; they are not sensor-confirmed color
  response evidence and should be rejected by any color-accuracy workflow.
- `camera_e2e_cfa_db_tables/`: dedicated CFA DB lookup export from
  `export_camera_e2e_cfa_db_tables.py`. It emits
  `camera_e2e_cfa_db_tables.json`,
  `camera_e2e_cfa_db_by_sensor.csv`,
  `camera_e2e_cfa_db_transmission_lut.csv`, and an HTML review page. The sensor
  table exposes the `image_sensor_db/optical_qe_db` CFA pattern, thickness,
  source kind, confidence, proxy library, channel set, and blockers. The
  transmission LUT is keyed by `slug + color_channel + wavelength_nm` and
  carries CFA n,k plus absorption-only transmission. This is the direct
  CameraE2E lookup layer for CFA information. It preserves
  `GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN` and `cfa_assumption_gate=MISSING` for
  sensors whose CFA pattern is unavailable, and it keeps product gates blocked
  until measured CFA material and spectral response are imported/calibrated.
- `camera_e2e_module_coupling/`: module-facing field LUT from
  `export_camera_e2e_module_coupling.py`. It expands each sensor's field design
  cases into wavelength rows and carries field position, CRA vector, ML/OCL
  shift, lens-vs-sensor CRA mismatch gate, sensor decenter/tilt priors, cos^4
  relative illumination, pupil data gate, wavelength-dependent pupil prior
  columns, `research_use_gate`, and `product_lut_gate`. The pupil prior columns
  are `pupil_relative_transmission`, `pupil_cra_shift_x_deg`,
  `pupil_cra_shift_z_deg`, `pupil_cra_shift_uncertainty_deg`, and
  `pupil_model`. Current rows are research-only because no imported lens
  raytrace/measured CRA map or calibrated wavelength-dependent pupil table is
  present. The research gate can be `CHECK` while the product gate remains
  `MISSING`; CameraE2E consumers must preserve that distinction.
- `camera_e2e_coverage_matrix/`: requirement-to-artifact coverage export from
  `export_camera_e2e_coverage_matrix.py`. It emits
  `camera_e2e_coverage_matrix.csv`, `camera_e2e_coverage_summary.csv`,
  `camera_e2e_coverage_matrix.json`, and an HTML review page. The matrix has
  one row per sensor per CameraE2E requirement, including Optical/Color,
  Pixel/Electrical, Readout/RAW, and Module Coupling. Each row carries the
  concrete source artifacts, row count, research gate, product gate, and primary
  blocker. This is the integration-facing checklist for deciding which rows can
  be loaded for research and which values must still be replaced by measured or
  calibrated product data.
- `camera_e2e_consumer_bundle/`: recommended CameraE2E load contract from
  `export_camera_e2e_consumer_bundle.py`. It emits
  `camera_e2e_consumer_bundle.json`,
  `camera_e2e_consumer_sensor_index.csv`,
  `camera_e2e_consumer_artifacts.csv`, one per-sensor JSON under
  `camera_e2e_consumer_bundle/sensors/`, and an HTML review page. Each
  per-sensor manifest carries source table paths, join keys, row counts,
  coverage rows, mesh-confidence classification, CFA provenance class, dedicated
  CFA DB lookup links, finite-array crosstalk support/truncation rows, LUT trust
  assessment rows, scalar probe summaries, source-integrity rows, a
  requirement-level `objective_fulfillment.requirement_rows` load map, and
  product blockers. The top-level bundle also exposes `source_tables`,
  `join_keys`, and `requirement_load_map`, so table-oriented CameraE2E loaders
  can discover every requested Optical/Color, Pixel/Electrical, Readout/RAW,
  and Module Coupling item without hard-coded file knowledge. The
  per-sensor JSON must include `lut_trust`, `lut_trust_by_sensor`,
  `lut_trust_by_domain`, `lut_trust_by_requirement`, and matching join keys so
  CameraE2E cannot load a sensor without also loading the trust/mesh usage
  guard. It also carries `crosstalk_support`, `crosstalk_support_by_sensor`, and
  `crosstalk_product_candidates` join keys so compact crosstalk kernels are
  never loaded without their finite-array support-risk guard. The consumer
  sensor index also exposes aggregate finite-array support hints:
  `crosstalk_support_summary`,
  `crosstalk_support_max_required_neighborhood`,
  `crosstalk_support_min_truncation_fraction`, and
  `crosstalk_support_max_truncation_fraction`. These are support-sizing and
  scheduling hints only; product crosstalk use still requires product-resolution
  mesh/convergence PASS rows. It also carries
  `cfa_db_by_sensor` and `cfa_db_transmission_lut` join keys so CameraE2E can
  load CFA pattern/thickness/transmission rows without reverse-engineering the
  material table. This is the
  easiest entrypoint
  for CameraE2E code that
  wants to load the whole research package without knowing the internal exporter
  order.
- `camera_e2e_flat_sensor_bundle/`: self-contained per-sensor CameraE2E load
  model from `export_camera_e2e_flat_sensor_bundle.py`. It emits
  `camera_e2e_flat_sensor_bundle.json`,
  `camera_e2e_flat_sensor_index.csv`, one JSON per sensor under
  `camera_e2e_flat_sensor_bundle/sensors/`, and an HTML review page. This is
  the most direct entrypoint when a CameraE2E runtime wants to load one sensor
  without reverse-engineering multiple CSV joins. Each per-sensor JSON embeds
  the filtered rows in four domain groups:
  `optical_color`, `pixel_electrical`, `readout_raw`, and `module_coupling`.
  It also embeds `uncertainty_budget`, `response_trace`,
  `response_example`, `method_provenance`, and `source_integrity` sections so a
  one-file-per-sensor loader can still distinguish solver rows from
  external-DB, topology-derived, proxy, and prior rows.
  It now also embeds `objective_fulfillment.requirement_rows`, a 21-row
  per-sensor map from each active CameraE2E requirement to the in-file
  `flat_json_pointer`, primary loader table, source class, calculation method,
  uncertainty band, recommended use, and product blocker. The flat bundle fails
  validation if the objective fulfillment row count does not match both coverage
  and source-integrity rows for the sensor.
- `camera_e2e_import_contract/`: downstream adapter import contract from
  `export_camera_e2e_import_contract.py`. It opens every flat per-sensor JSON,
  resolves every `objective_fulfillment.requirement_rows[*].flat_json_pointer`,
  and emits `camera_e2e_import_contract.json`,
  `camera_e2e_import_contract_by_sensor.csv`,
  `camera_e2e_import_contract_by_requirement.csv`,
  `camera_e2e_import_contract_checks.csv`, one per-sensor import contract JSON
  under `camera_e2e_import_contract/sensors/`, and an HTML review page. This is
  the adapter-facing proof that every requested Optical/Color,
  Pixel/Electrical, Readout/RAW, and Module Coupling item can be loaded in
  research mode while product mode remains fail-closed. It does not certify
  product physical accuracy.
- `camera_e2e_canonical_payload/`: adapter-ready payload export from
  `export_camera_e2e_canonical_payload.py`. It consumes the flat per-sensor
  model plus import contract, then emits `camera_e2e_canonical_payload.json`,
  `camera_e2e_canonical_payload_by_sensor.csv`,
  `camera_e2e_canonical_payload_checks.csv`, one per-sensor JSON under
  `camera_e2e_canonical_payload/sensors/`, and an HTML review page. Each
  per-sensor payload is organized as `camera_e2e_payload.optical_color`,
  `camera_e2e_payload.pixel_electrical`, `camera_e2e_payload.readout_raw`, and
  `camera_e2e_payload.module_coupling`, with source integrity, method
  provenance, uncertainty, import-contract rows, and product blockers preserved
  under `routing_and_evidence`.
  The optical/color group includes CFA DB rows, material n,k rows, spectral
  response rows, runtime CRA/field response rows, compact optical crosstalk
  kernels, finite-array crosstalk support rows, and solver-priority rows. The
  flat sensor index also exposes finite-array crosstalk support summary columns:
  `crosstalk_support_row_count`, `crosstalk_support_gate`,
  `crosstalk_support_summary`,
  `crosstalk_support_max_required_neighborhood`,
  `crosstalk_support_min_truncation_fraction`,
  `crosstalk_support_max_truncation_fraction`,
  `crosstalk_product_candidate_row_count`, and
  `crosstalk_batch_priority_row_count`. These columns are load-routing and
  scheduling hints only. They do not convert low-resolution support pilots into
  product crosstalk evidence.
  electrical/readout/module groups embed the corresponding prior LUTs and CRA
  coupling rows. This artifact improves loader ergonomics only; it does not
  promote sparse mesh, proxy material, or prior electrical/readout rows to
  product accuracy. Product use remains blocked unless the embedded gates and
  upstream product gates pass.
- `camera_e2e_flat_sensor_query/`: direct flat-bundle query output from
  `query_camera_e2e_flat_sensor_bundle.py`. It starts from
  `camera_e2e_flat_sensor_bundle.json`, opens the embedded per-sensor JSON
  files, performs the runtime optical lookup from embedded rows, joins embedded
  spectral/material/electrical/readout/module rows, and emits
  `camera_e2e_flat_sensor_query.csv`,
  `camera_e2e_flat_sensor_query_kernel.csv`,
  `camera_e2e_flat_sensor_query_summary.csv`, JSON, and HTML. The rows include
  response/QE proxy, spectral response, material n,k row counts, crosstalk
  kernel summary, CRA and lens shift, electrical/noise priors, readout/RAW
  priors, module CRA/vignetting priors, raw DN and SNR scalar probes, and
  product blockers. The paired
  `camera_e2e_flat_sensor_query_product_probe/` run verifies that product mode
  returns zero allowed rows until product gates are actually closed.
- `camera_e2e_analysis_report/`: design-facing summary from
  `export_camera_e2e_analysis_report.py`. It consumes the flat query outputs and
  emits `camera_e2e_analysis_by_sensor.csv`,
  `camera_e2e_analysis_by_channel.csv`,
  `camera_e2e_analysis_actions.csv`, JSON, and HTML. The sensor table gives a
  compact CameraE2E usability view per image sensor: query row counts, summary
  PASS/CHECK counts, field-coverage min/mean, edge-to-center response range,
  crosstalk maxima, signal/raw/SNR summary, CFA provenance, trust class, mesh
  confidence class, field mesh pass fraction, finite-array crosstalk mesh pass
  fraction, product blockers, and recommended use. The channel table is keyed by
  `slug + color_channel + wavelength_nm` and preserves `summary_gate` and
  `summary_notes`, so partial field coverage or suspicious near-zero edge
  response is visible before CameraE2E uses the row for CRA/color-shading
  studies. The actions table lists follow-up solver or measured-input work for
  CHECK rows. This is a design review artifact and remains product-blocked.
- `camera_e2e_use_scope_summary/`: consumer routing table from
  `export_camera_e2e_use_scope_summary.py`. It emits
  `camera_e2e_use_scope_summary.json`,
  `camera_e2e_use_scope_by_sensor.csv`,
  `camera_e2e_use_scope_by_domain.csv`,
  `camera_e2e_use_scope_next_actions.csv`, and an HTML review page. This is the
  first table a CameraE2E consumer should load before using runtime values. It
  classifies each sensor as `CAMERA_E2E_RESEARCH_TREND_ONLY`,
  `CAMERA_E2E_SINGLE_ANCHOR_OR_SMOKE_ONLY`,
  `CAMERA_E2E_SCHEMA_PRIOR_ONLY`, or
  `CAMERA_E2E_SCHEMA_PRIOR_ONLY_CFA_UNKNOWN`, and it classifies each domain
  separately as optical trend, optical proxy, electrical prior, readout prior,
  or module prior. It also carries the first crosstalk batch command and the
  measured-input requirements needed before product use. Current
  `product_ready_count` is zero by design.
- `camera_e2e_consumer_query/`: consumer-bundle query output from
  `query_camera_e2e_consumer_bundle.py`. It starts from
  `camera_e2e_consumer_bundle.json`, runs the runtime optical interpolation,
  blends crosstalk kernels, and joins spectral response, material n,k,
  electrical/noise, readout, binning/remosaic, module coupling, coverage rows,
  mesh-confidence rows, and CFA provenance rows into
  `camera_e2e_consumer_query.csv`. Query rows carry
  `runtime_confidence_class`, `mesh_confidence_class`,
  `cfa_provenance_class`, `cfa_assumption_gate`, `lut_trust_class`,
  `lut_trust_allowed_use`, `lut_trust_evidence_score_0_100`,
  `lut_trust_product_score_0_100`,
  `lut_trust_field_mesh_pass_fraction`, and
  `lut_trust_crosstalk_mesh_pass_fraction`, plus
  `crosstalk_support_gate`,
  `crosstalk_support_best_neighborhood`,
  `crosstalk_support_best_truncation_fraction`,
  `crosstalk_support_summary`,
  `crosstalk_support_max_required_neighborhood`,
  `crosstalk_support_min_truncation_fraction`,
  `crosstalk_support_max_truncation_fraction`,
  `crosstalk_support_threshold`, and
  `crosstalk_support_recommendation` so a CameraE2E consumer can
  reject structural-prior rows, partial-trend rows, or generic RGB fallback rows
  when a stricter workflow is required. `crosstalk_product_candidate_count` is
  the number of generated candidate commands for that exact query condition.
  `crosstalk_product_candidate_lowest_available_neighborhood` is only the
  smallest generated support size, while
  `crosstalk_product_candidate_min_neighborhood` is the recommended minimum
  support size when support evidence exists. For example, SC550XS green 550 nm
  center has generated candidates starting at 3x3, but the recommended minimum
  after the low-resolution support audit is 15x15. Center-field support pilots
  are not applied to non-center field queries or to other color channels. The
  pipeline also writes
  `camera_e2e_consumer_query_product_probe/` to prove product-mode queries fail
  closed while product gates remain blocked.
- `camera_e2e_mesh_confidence/`: mesh-resolution and convergence confidence
  audit from `audit_camera_e2e_mesh_confidence.py`. It emits
  `camera_e2e_mesh_confidence.json`,
  `camera_e2e_mesh_confidence_by_sensor.csv`,
  `camera_e2e_mesh_confidence_by_domain.csv`,
  `camera_e2e_mesh_confidence_checks.csv`, and an HTML review page. The audit
  compares quantitative queue requirements against completed PASS points,
  grid-resolution gates, signed-flux diagnostics, crosstalk resource limits,
  and runtime evidence gates. This is the recommended way for CameraE2E
  consumers to distinguish structural priors, sparse local anchors, partial
  field-trend evidence, and product candidates. Current output is research-only
  and product-blocked.
- `camera_e2e_field_execution_pack/`: runnable field/QE execution handoff from
  `export_camera_e2e_field_execution_pack.py`. It emits
  `camera_e2e_field_execution_pack.json`,
  `camera_e2e_field_execution_jobs.csv`,
  `camera_e2e_field_execution_scripts.csv`, an HTML review page, and executable
  shell scripts: `run_center_spectral_color_anchors.sh`,
  `run_green_cra_field_anchors.sh`, `run_failed_or_stale_field_reruns.sh`,
  `run_all_field_quantitative_remaining.sh`, and
  `refresh_after_field_jobs.sh`. The job CSV ranks remaining or stale Meep
  field points by practical impact: failed/stale reruns, center 550 nm color
  anchors, center spectral anchors, green 550 nm CRA field anchors, and full RGB
  x wavelength x field closure. This artifact is scheduling guidance only;
  `product_use_gate` stays `FAIL` until selected solver outputs pass
  mesh/convergence gates and measured stack/material calibration is available.
- `camera_e2e_crosstalk_support_audit/`: finite-array crosstalk support audit
  from `audit_camera_e2e_crosstalk_support.py`. It emits
  `camera_e2e_crosstalk_support_audit.json`,
  `camera_e2e_crosstalk_support_by_sensor.csv`,
  `camera_e2e_crosstalk_support_pilots.csv`,
  `camera_e2e_crosstalk_product_candidates.csv`, and an HTML review page. The
  audit compares low-resolution finite-array support pilots such as 3x3, 5x5,
  7x7, ... output kernels and records truncation, grid, convergence, and
  product-use gates. The product-candidate CSV is expanded from the full
  quantitative crosstalk queue by `slug`, color/clear channel, wavelength, field
  case, and candidate support size, so missing red/blue/clear or edge-field
  crosstalk jobs remain explicit batch/HPC work items. This artifact is a
  support-risk guard only; it must not be treated as product crosstalk evidence
  unless a product-resolution finite-array run also passes the normal
  mesh/convergence gates.
- `camera_e2e_crosstalk_batch_priority/`: support-aware crosstalk batch
  scheduler from `export_camera_e2e_crosstalk_batch_priority.py`. It emits
  `camera_e2e_crosstalk_batch_priority.json`,
  `camera_e2e_crosstalk_batch_priority.csv`, and an HTML review page. The CSV
  collapses the full crosstalk candidate table into one actionable row per
  crosstalk condition: `product_resolution_crosstalk_primary` where low-res
  finite-array support evidence already exists, or
  `low_resolution_support_discovery` where support evidence is still missing.
  The current first P0 row is the SC550XS green/center/550 nm 15x15
  product-resolution run, which is marked as HPC/domain-decomposition work. This
  artifact is scheduling guidance only; `product_use_gate` stays `FAIL` until
  the selected product-resolution jobs pass mesh/convergence gates and measured
  stack/material calibration is available.
- `camera_e2e_crosstalk_execution_pack/`: runnable crosstalk execution handoff
  from `export_camera_e2e_crosstalk_execution_pack.py`. It emits
  `camera_e2e_crosstalk_execution_pack.json`,
  `camera_e2e_crosstalk_execution_jobs.csv`,
  `camera_e2e_crosstalk_execution_scripts.csv`,
  `camera_e2e_crosstalk_local_probe_evidence.csv`, an HTML review page, and
  executable shell scripts:
  `run_product_primary_hpc.sh`,
  `run_support_discovery_local_candidates.sh`,
  `run_support_discovery_batch_or_reformulation.sh`, and
  `refresh_after_solver_jobs.sh`. The package separates P0 product-primary
  finite-array crosstalk jobs from support-discovery jobs and includes local
  probe evidence showing that even low-resolution n15/res20 finite-array setup
  can be expensive on a laptop. These scripts are execution handoff artifacts
  only. They do not make crosstalk product-ready; product use remains blocked
  until completed solver outputs pass product mesh/convergence gates and
  measured stack/material calibration blockers are removed.
- `camera_e2e_capability_profile/`: per-sensor CameraE2E use-scope profile from
  `export_camera_e2e_capability_profile.py`. It emits
  `camera_e2e_capability_profile.json`,
  `camera_e2e_capability_by_sensor.csv`, `camera_e2e_capability_checks.csv`,
  and an HTML review page. The profile collapses coverage, mesh confidence, CFA
  provenance, material, electrical/readout, and module-coupling gates into
  practical scope labels such as `PARTIAL_RESEARCH_TREND`,
  `SINGLE_ANCHOR_RESEARCH_CHECK`,
  `RESEARCH_PRIOR_SCHEMA_PLUMBING`,
  `PLUMBING_PLUS_PRIORS_CFA_UNKNOWN`, and
  `COMPACT_SURROGATE_ONLY_NO_FINITE_ARRAY_PASS`. CameraE2E consumers should use
  this table as the first-pass routing layer before deciding whether a sensor is
  suitable for plumbing tests, research trend studies, or stricter accuracy
  workflows.
- `camera_e2e_lut_trust_assessment/`: explicit trust and usage-guard export
  from `export_camera_e2e_lut_trust_assessment.py`. It emits
  `camera_e2e_lut_trust_assessment.json`,
  `camera_e2e_lut_trust_by_sensor.csv`,
  `camera_e2e_lut_trust_by_domain.csv`,
  `camera_e2e_lut_trust_by_requirement.csv`, checks, and an HTML page. The
  scores intentionally separate `research_usability_score_0_100`,
  `evidence_confidence_score_0_100`, and
  `product_calibration_score_0_100`. These values are CameraE2E routing guards,
  not physical sensor accuracy percentages. Current product calibration scores
  remain zero until measured stack/material/CRA, quantitative FDTD coverage,
  finite-array crosstalk, and measured electrical/readout/module calibration
  pass.
- `camera_e2e_uncertainty_budget/`: engineering uncertainty-band export from
  `export_camera_e2e_uncertainty_budget.py`. It emits
  `camera_e2e_uncertainty_budget.csv`,
  `camera_e2e_uncertainty_by_sensor.csv`, JSON, checks, and an HTML page. The
  sensor summary carries uncertainty ranges for material RI, CFA
  k/transmission, QE, CRA edge response, optical crosstalk, conversion
  gain/FWC, temporal noise, dark current, DSNU/PRNU, readout/RAW, and module
  coupling. These bands are for CameraE2E sensitivity studies only; the
  `uncertainty_product_gate` remains `FAIL` until measured inputs and
  calibration data replace the priors.
- `camera_e2e_response_trace/`: row-level response-construction trace from
  `export_camera_e2e_response_trace.py`. It joins runtime response rows with
  CFA/OCL/passivation/Si material rows and a simple CFA-transmission times
  silicon-absorption sanity calculation. This explains how proxy response rows
  were formed, but the sanity calculation is not a replacement for converged
  Meep FDTD.
- `camera_e2e_response_example/`: compact human-readable response examples from
  `export_camera_e2e_response_example.py`. It emits representative center-field
  R/G/B or clear rows per sensor showing CFA transmission, simple silicon
  absorption, runtime normalization, QE proxy, direct response, and neighbor
  leakage response. These examples are for review and debugging, not product
  QE evidence.
- `camera_e2e_method_provenance/`: per-requirement source-method matrix from
  `export_camera_e2e_method_provenance.py`. It emits
  `camera_e2e_method_provenance_matrix.csv` and
  `camera_e2e_method_provenance_by_sensor.csv`. Each row classifies a
  CameraE2E requirement as solver-anchor, external/local DB plus proxy
  material, topology/structure-derived, compact surrogate, electrical prior,
  readout prior, or module prior, and records the source priority and product
  blocker.
- `camera_e2e_lut_source_integrity/`: joined source/method/uncertainty matrix
  from `export_camera_e2e_lut_source_integrity.py`. This is the recommended
  first audit table when reviewing a LUT row for CameraE2E use. It emits
  `camera_e2e_lut_source_integrity_matrix.csv`,
  `camera_e2e_lut_source_integrity_by_sensor.csv`, JSON, checks, and HTML. The
  matrix joins `coverage`, `method_provenance`, and `uncertainty_budget` by
  `slug + requirement_id`, so every requirement row carries source class,
  calculation method, solver/external/proxy dependencies, primary uncertainty
  band, research/product gates, and the next measured-data or solver action.
  `source_integrity_gate=CHECK` means the row is loadable for research/trend
  only; it is not product evidence.
- `camera_e2e_sensor_deliverable_summary/`: one-row-per-sensor CameraE2E
  deliverable selector from `export_camera_e2e_sensor_deliverable_summary.py`.
  It joins the flat bundle index, consumer manifest paths, source-integrity
  summary, analysis rows, uncertainty bands, and objective acceptance evidence.
  Use this as the first human-facing table to decide whether a sensor should be
  loaded as research/trend data, schema/prior-only data, or product-blocked
  data. `deliverable_gate=PASS` means the package is coherent; it does not mean
  `product_ready=true`.
- `camera_e2e_handoff_manifest/`: integration load index from
  `export_camera_e2e_handoff_manifest.py`. This is the recommended entrypoint
  for CameraE2E consumers. It emits an artifact load index, a per-sensor load
  index, and a compact HTML page. The recommended load order starts with
  `camera_e2e_sensor_deliverable_summary/`, then `camera_e2e_usage_policy/`,
  then `camera_e2e_lut_source_integrity/`, then either the flat sensor bundle
  for direct one-file-per-sensor loading or the consumer bundle and runtime
  values for table-oriented loading.
  It verifies that the required runtime, consumer bundle, use-scope, coverage,
  crosstalk, color, material, CFA provenance, electrical/readout, prior-seed,
  module-coupling, readiness, trust assessment, and probe
  artifacts exist, while
  keeping product gates blocked when the source data is not
  measured/calibrated.
- `camera_e2e_objective_trace/camera_e2e_objective_trace_by_requirement_summary.csv`:
  21-row objective summary for CameraE2E reviewers. Each row aggregates one
  requested requirement across all sensors and lists the flat JSON pointer,
  primary loader table, source-class distribution, calculation-method
  distribution, uncertainty range, source artifacts, and product blockers. Use
  it to decide which requirements are solver/structure-supported versus
  proxy/prior-only before loading runtime values. The objective trace validation
  also resolves every `flat_json_pointer` against every per-sensor flat JSON,
  including table-column pointers, so broken loader paths fail the pipeline.
- `camera_e2e_handoff_loader_validation/`: consumer-path loader validation from
  `validate_camera_e2e_handoff_loader.py`. It reads the handoff manifest,
  loads referenced artifacts, validates per-sensor model/prior JSON schemas,
  checks `slug` joins across coverage, runtime, color, material, CFA provenance,
  electrical, module, trust, and probe rows, checks per-sensor consumer
  manifests for required trust source tables and join keys, and checks
  `runtime_id` joins plus crosstalk-kernel normalization. This proves research
  package loadability, not product accuracy.
- `camera_e2e_objective_acceptance/`: active-objective acceptance audit from
  `audit_camera_e2e_objective_acceptance.py`. It verifies that every packaged
  sensor has complete research requirement coverage, a loadable consumer
  bundle, at least one allowed research consumer query, zero allowed product
  consumer queries, a loadable flat per-sensor bundle/query, design analysis
  rows, mesh-confidence rows, and explicit product blockers. The per-sensor CSV
  carries `field_mesh_pass_fraction` and `crosstalk_mesh_pass_fraction` so a
  green research acceptance result cannot be mistaken for full numerical
  convergence. A PASS means the current research package is coherent enough for
  CameraE2E sensitivity/prototyping handoff. It does not mean the LUT is
  product-accurate.
- `camera_e2e_sensor_probe/`: scalar CameraE2E smoke/probe output from
  `simulate_camera_e2e_sensor_probe.py`. It queries the runtime bundle, applies
  prior-seed FWC/noise/ADC/black-level terms, and emits `signal_e`,
  `noise_e_rms`, `raw_dn`, and per-kernel `crosstalk_e` rows for a selected
  sensor/field/wavelength/exposure condition. This is a consumer-path test for
  research integration, not a product calibration.
- `camera_e2e_sensor_probe_all_sensors/`: same probe path run for all selected
  sensors over a small field/wavelength matrix. The summary CSV groups by
  sensor/color/wavelength and reports row counts, mean/min/max signal, SNR
  bounds, edge-to-center signal ratio where available, and max crosstalk
  fractions.
- `camera_e2e_closure_plan*/`: product-gate closure plans from
  `plan_camera_e2e_closure.py`. These convert readiness blockers into measured
  input tasks and exact queue-runner commands for the next quantitative solver
  batches. The closure plan now consumes
  `camera_e2e_crosstalk_batch_priority.csv` directly when available, so
  finite-array crosstalk work uses the support-aware primary/support-discovery
  commands rather than older generic resource-limit rows. The summary carries
  `crosstalk_priority_solver_row_count`,
  `crosstalk_product_primary_solver_row_count`, and
  `crosstalk_support_discovery_solver_row_count`.
- `camera_e2e_product_closure_summary/`: compact per-sensor product-readiness
  worklist from `export_camera_e2e_product_closure_summary.py`. It emits
  `camera_e2e_product_closure_summary.json`,
  `camera_e2e_product_closure_by_sensor.csv`,
  `camera_e2e_product_closure_by_domain.csv`,
  `camera_e2e_product_closure_checks.csv`, and an HTML review page. The sensor
  CSV joins use scope, coverage, closure plan, and crosstalk execution-pack
  rows so reviewers can see current CameraE2E use scope, mesh pass counts,
  measured-input blocker counts, calibration blocker counts, field solver rows,
  crosstalk product-primary/HPC job counts, support-discovery counts, and the
  first required closure action. This is a routing/worklist artifact only.
  Product use remains blocked until the underlying measured-data and
  product-resolution solver gates pass.
- `camera_e2e_usage_policy/`: CameraE2E ingest policy from
  `export_camera_e2e_usage_policy.py`. It emits
  `camera_e2e_usage_policy.json`,
  `camera_e2e_usage_policy_by_sensor.csv`,
  `camera_e2e_usage_policy_by_domain.csv`,
  `camera_e2e_usage_policy_runtime_filters.csv`,
  `camera_e2e_usage_policy_checks.csv`, and an HTML review page. This artifact
  is the loader-facing contract for choosing research versus product filters.
  `research_runtime_rows` may be used for plumbing and trend studies with
  evidence gates attached; `strict_product_runtime_rows` is the only acceptable
  product-mode filter and is expected to have zero rows in the current package.
  The policy also records per-sensor allowed modes, blocked modes, recommended
  bundles, and the first product-closure action.
- `camera_e2e_adapter_examples/`: concrete downstream load recipes from
  `export_camera_e2e_adapter_examples.py`. It emits
  `camera_e2e_adapter_examples.json`,
  `camera_e2e_adapter_examples_by_sensor.csv`,
  `camera_e2e_adapter_examples_checks.csv`, per-sensor JSON files under
  `camera_e2e_adapter_examples/sensors/`, and an HTML review page. Each
  per-sensor JSON points to the usage policy, flat sensor JSON, recommended
  runtime filter, CameraE2E domain sections,
  `objective_fulfillment.requirement_rows`, representative research query
  summaries, and a fail-closed product probe command. The product probe is
  expected to expose zero allowed rows while product gates remain blocked. The
  adapter examples fail validation if any per-sensor flat JSON lacks the
  requirement-level objective map.
- `camera_e2e_adapter_smoke/`: executable adapter smoke output from
  `run_camera_e2e_adapter_smoke.py`. It runs
  `query_camera_e2e_flat_sensor_bundle.py` per sensor in research mode and
  product-probe mode, then emits `camera_e2e_adapter_smoke.json`,
  `camera_e2e_adapter_smoke_by_sensor.csv`,
  `camera_e2e_adapter_smoke_checks.csv`, per-sensor query output folders under
  `camera_e2e_adapter_smoke/sensors/`, and an HTML review page. A PASS proves
  the adapter recipes execute, research mode returns allowed rows, and product
  mode remains fail-closed with zero allowed rows. It does not certify product
  sensor accuracy.
- `camera_e2e_objective_trace/`: requirement-to-loader trace from
  `export_camera_e2e_objective_trace.py`. It emits
  `camera_e2e_objective_trace.json`,
  `camera_e2e_objective_trace_by_requirement.csv`,
  `camera_e2e_objective_trace_by_sensor.csv`,
  `camera_e2e_objective_trace_checks.csv`, and an HTML review page. It maps
  each objective requirement to a flat sensor JSON section, JSON pointer,
  primary loader table, source artifacts, usage-policy profile, and adapter
  smoke evidence. A PASS means every objective row is loadable for research and
  product usage remains blocked.
- `camera_e2e_pipeline_validation/`: end-to-end rebuild and gate-validation
  report from `run_camera_e2e_package_pipeline.py`.

Important field-response row columns:

- `slug`, `code`, `manufacturer`, `device_name`, `pixel_pitch_um`
- `field_case`, `field_x_norm`, `field_z_norm`
- `cra_x_deg`, `cra_z_deg`
- `lens_cra_x_deg`, `lens_cra_z_deg`
- `sensor_cra_x_deg`, `sensor_cra_z_deg`
- `cra_mismatch_x_deg`, `cra_mismatch_z_deg`,
  `cra_mismatch_total_deg`, `cra_mismatch_gate`
- `cra_mismatch_tolerance_profile`,
  `cra_mismatch_pass_tolerance_deg`,
  `cra_mismatch_check_tolerance_deg`
- `lens_shift_x_um`, `lens_shift_z_um`
- `lens_shift_model`, `cra_measurement_gate`, `cra_input_gate`, `cra_source`
- `edge_cra_assumption_deg`, `focus_target_depth_um`, `lens_shift_cap_um`
- `wavelength_nm`, `color_channel`
- `relative_qe_proxy`, `relative_qe_min`, `relative_qe_max`
- `split_phase_x_proxy`, `split_phase_z_proxy`
- `response_model`, `evidence_level`, `evidence_gate`, `source`
- `product_lut_ready`

Important compact crosstalk row columns:

- `slug`, `field_case`, `field_x_norm`, `field_z_norm`
- `wavelength_nm`, `color_channel`
- `kernel_scope`, `neighborhood`, `dx`, `dz`
- `response_fraction`, `color_relation`
- `center_fraction`, `output_crosstalk_fraction`,
  `strongest_neighbor_fraction`
- `sigma_x_um`, `sigma_z_um`, `centroid_x_um`, `centroid_z_um`
- `dti_width_um`, `dti_depth_um`, `dti_barrier_factor`
- `model`, `source`, `evidence_level`, `evidence_gate`

The compact crosstalk model is a separable Gaussian focal-spot integration with
DTI attenuation. It must remain `evidence_gate=CHECK` until calibrated against
finite-array FDTD or measurement. It is useful for CameraE2E development because
it provides a normalized kernel for every exported field/wavelength row, but it
is not product crosstalk evidence.

`evidence_level` separates direct solver evidence from fallback rows:

- `fdtd_quantitative_pass`: direct quantitative Meep FDTD row with PASS gate.
- `fdtd_quantitative_fail_numeric_reference`: failed quantitative row retained
  only as a non-production numeric reference.
- `tcad_lateral_proxy_scaled`: FDTD center/default spectral prior scaled by the
  DEVSIM lateral-generation proxy.
- `design_prior_spectral_rolloff`: generated prior used when neither FDTD nor
  usable TCAD proxy exists.

The ingest exporter deliberately sets `production_ingest_allowed=false` and
`product_lut_ready=false` until measured CRA/ML shift input, measured stack/n,k,
complete quantitative field coverage, and converged crosstalk all pass. CameraE2E
research runs may ingest these rows only if they preserve `evidence_level` and
do not mix prior rows with solver-pass rows as equal-confidence data.

The query output bilinearly interpolates within each `slug` and `wavelength_nm`
field grid, clamps query points to the exported field axes, and carries forward
the weakest source evidence gate from the neighboring anchors. Exact anchor
queries use the exact source row so direct `fdtd_quantitative_pass` anchors stay
PASS, while mixed-source interpolation normally becomes CHECK.

Supplementary quantitative color/wavelength rows are grouped by
`slug/wavelength/color_channel`. Sparse supplementary groups are not spread
across the full field map: if a group has only one x or z coordinate, the
combined query emits only matching requested coordinates for that axis. This
prevents a single center-only spectral point from being misrepresented as full
field coverage.

Per-sensor model manifests deliberately keep requested CameraE2E categories as
separate evidence items:

- Optical/Color: CFA/OCL/passivation/Si material inputs, spectral response/QE,
  color response matrix rows, optical crosstalk kernel, CRA response, and
  microlens/OCL shift map.
- Pixel/Electrical: conversion gain, FWC/saturation/nonlinearity, dark current,
  DSNU, PRNU, temporal noise, and charge-collection/electrical crosstalk.
- Readout/RAW: gain tables, black level, ADC, row/column FPN, rolling shutter,
  defects/hot pixels, and binning/remosaic behavior.
- Module Coupling: lens raytrace CRA map, sensor tilt/decenter, vignetting, and
  wavelength-dependent chief ray/pupil behavior.

Items that are not measured or calibrated remain `gate=MISSING` or `CHECK`.
This is intentional: the model export is a CameraE2E handoff manifest and an
evidence index. It must not be interpreted as product accuracy unless every
section gate and the sensor-level `production_lut_gate` pass.

The dedicated coverage matrix uses these columns:

- `slug`, `code`, `manufacturer`, `device_name`
- `domain`: `Optical / Color`, `Pixel / Electrical`, `Readout / RAW`, or
  `Module Coupling`
- `requirement_id`, `requirement`, `camera_e2e_use`
- `research_status`, `research_gate`, `product_gate`
- `row_count`: number of concrete evidence rows backing this requirement
- `source_artifacts`: semicolon-separated load paths for CameraE2E consumers
- `evidence_summary`: compact JSON summary of counts, grids, and gate details
- `primary_blocker`: measured-data or solver-calibration gap that blocks product
  use
- `notes`

`research_gate=CHECK` means the row is loadable for trend/system testing.
`product_gate=FAIL` or `MISSING` means it must not be used as calibrated sensor
characterization. Monochrome sensors use `N/A` for RGB color-matrix rows while
keeping clear-channel spectral response rows.

The combined query output keeps separate gates:

- `field_evidence_gate`: field-response source evidence after interpolation.
- `cra_input_gate`: CRA / microlens-shift input provenance. This is PASS only
  when all contributing field anchors are marked `MEASURED`, `CALIBRATED`, or
  `RAYTRACE_VALIDATED` and their lens-vs-sensor CRA mismatch gate is PASS.
- `crosstalk_evidence_gate`: compact-kernel source evidence.
- `combined_evidence_gate`: weakest of the three. With the current compact
  crosstalk surrogate and assumed CRA priors this is normally CHECK even when
  the field anchor itself is PASS.

The NPZ export stores:

- scalar metadata: `schema`, `artifact_role`, `product_lut_ready`, `row_count`
- arrays: all numeric columns such as `field_x_norm`, `cra_x_deg`,
  `lens_shift_x_um`, `relative_qe_proxy`, `relative_qe_min`, and
  `relative_qe_max`
- string arrays: `slug`, `field_case`, `response_model`, `evidence_level`,
  `evidence_gate`, `cra_measurement_gate`, `cra_source`, and other
  source/provenance fields
- `row_product_lut_ready`, which must remain all false in the current
  research/trend export

`export_camera_e2e_ingest_luts.py` validates NPZ arrays against the JSON rows
after writing. Any numeric/string mismatch makes the NPZ gate fail.

The readiness audit is deliberately stricter than the ingest manifest:

- `research_ingest_gate`: whether the package is internally usable for
  non-production trend experiments while preserving row-level gates.
- `production_lut_gate`: PASS only when measured/calibrated CRA/ML-shift,
  measured stack/n,k, complete quantitative FDTD field coverage, and
  finite-array crosstalk convergence all pass.
- `camera_e2e_lut_readiness_issues.csv`: actionable blockers and warnings to
  drive the next simulation/calibration work.

For field FDTD coverage, the product audit uses the quantitative point queue
when available. That means the stricter product denominator is the full
field/color/wavelength queue, not only the smaller canonical display grid.

The runtime bundle is the preferred integration entry point for CameraE2E trend
experiments:

- `camera_e2e_runtime_bundle.json`: manifest, source paths, validation result,
  gate counts, and use policy.
- `camera_e2e_runtime_lut.csv`: one row per sensor/field/wavelength/color query,
  including nominal response, response min/max, CRA and lens shift, crosstalk
  summary, readiness gates, and confidence class.
- `camera_e2e_runtime_crosstalk_kernel.csv`: normalized crosstalk kernel rows
  keyed by `runtime_id`.
- `camera_e2e_runtime_bundle.npz`: compressed typed arrays for the same runtime
  and kernel rows.

Current runtime bundles remain `product_lut_ready=false`. Their
`confidence_class` values distinguish partial solver-pass trend rows from
proxy/prior trend rows, and their crosstalk uncertainty bounds are intentionally
wide because the compact kernels are not finite-array calibrated.

Runtime query behavior:

- `--mode research` returns rows when `research_ingest_allowed=true`; current
  rows are normally `query_gate=CHECK`.
- `--mode product` blocks current rows because `product_lut_ready=false`,
  `production_lut_gate=FAIL`, and `production_ingest_allowed=false`.
- `--strict` exits nonzero if any queried row is blocked, which lets automated
  CameraE2E jobs fail closed instead of silently using research rows as product
  data.
- Query kernel rows are normalized per `runtime_query_id`; validation fails if
  any kernel sum differs from one beyond tolerance.

The closure plan is the operational handoff from research LUT to product-gate
closure:

- `camera_e2e_closure_plan.json`: manifest with row counts, track counts,
  selected solver-hour estimate, output paths, validation status, plan rows,
  and generated batch rows.
- `camera_e2e_closure_plan.csv`: one row per measured optical input blocker,
  measured electrical/readout/module calibration blocker, or runnable
  quantitative solver point. Electrical/readout/module rows use
  `track=measured_calibration_input` and are generated directly from
  product-blocked `Pixel / Electrical`, `Readout / RAW`, and `Module Coupling`
  coverage-matrix requirements. Resource-limited finite-array crosstalk rows
  are emitted under `track=solver_resource_limited_batch` and keep the direct
  `--max-local-voxels 0` command rather than being wrapped back into the local
  queue runner.
- `camera_e2e_closure_batches.csv`: grouped commands for
  `run_camera_e2e_quantitative_queue.py`, including comma-separated `queue_ids`,
  plus direct resource-limited crosstalk commands when local execution was
  skipped by voxel limits.
- `camera_e2e_closure_checks.csv`: validation checks proving measured optical
  input blockers, measured calibration blockers, resource-limited crosstalk
  rows, runnable commands, and generated batches are present.
- `index.html`: human-readable review page for design, optics, TCAD, and camera
  system stakeholders.

Measured-input rows are deliberately not runnable. They require external data
such as `image_sensor_db/camera_module_field_map.csv` from module raytrace or
measurement, measured stack geometry, wavelength-dependent measured `n,k`,
measured electrical/noise/readout calibration, and module coupling calibration.
Solver-batch rows are runnable and should be followed by merge, ingest export,
readiness audit, runtime bundle export, and runtime query validation. Estimated
solver hours in the closure plan are scheduling guidance only; they are not
physics evidence and do not raise any product gate by themselves.

The pipeline runner is the repeatable no-long-solver command for refreshing the
CameraE2E integration package:

```bash
python3 run_camera_e2e_package_pipeline.py --include-failed
```

It rebuilds the package, merges any existing quantitative point evidence,
exports ingest LUTs, builds compact crosstalk kernels, runs the combined query,
reruns the readiness audit, exports the runtime bundle, executes a product-mode
strict probe, executes a research-mode runtime query, and regenerates the
closure plan. The expected current status is
`RESEARCH_VALID_PRODUCT_BLOCKED`: the research runtime bundle validates, while
the product strict query returns no allowed rows because product gates are still
open. This is the correct fail-closed behavior until measured CRA/ML shift,
measured stack/n,k, full quantitative FDTD coverage, and finite-array crosstalk
all pass.

Pipeline files:

- `camera_e2e_pipeline_validation.json`: full step log, gate summary, and
  validation result.
- `camera_e2e_pipeline_steps.csv`: command, return code, duration, and stdout /
  stderr tails for each pipeline step.
- `index.html`: compact stakeholder review page.

## Workbench Suite Export Package

The Pixel Workbench test-suite backend writes a suite-level camera-system
package for practical design comparison:

```text
camera_system_suite_export_v1
```

Files:

- `camera_system_suite_export.json`: canonical suite-level package.
- `workbench_camera_system_export_summary.json`: compact package summary.
- `camera_system_field_response.csv`: field/CRA response rows when the suite
  generated field-response data.
- `camera_system_pdaf_response.csv`: split/PDAF response rows when available.
- `camera_system_crosstalk_summary.csv`: crosstalk KPI rows.
- `camera_system_crosstalk_cells.csv`: output-kernel cell rows.
- `camera_system_gate_report.csv`: per-case numerical/product-readiness gates.

The consumer validation tool writes:

```text
camera_system_suite_export_validation_v1
```

Files:

- `camera_system_suite_export_validation.json`: structural ingest validation.
- `camera_system_suite_export_validation.md`: human-readable validation card.
- `camera_system_suite_export_field_query.csv`: lightweight field interpolation
  query over suite field rows, when available.
- `camera_system_suite_export_crosstalk_index.csv`: compact crosstalk kernel
  index grouped by suite case/source field.
- `camera_system_suite_export_gate_summary.csv`: downstream gate state per case.

This validation is a consumer-safety check, not an accuracy certification. It
checks schema, row counts, finite/nonnegative response values, crosstalk fraction
ranges, center-cell coverage for kernel groups, artifact targets, and gate rows.
The package remains `product_lut_ready=false` unless the separate measured
accuracy and quantitative convergence gates pass.

## Quantitative Evidence Manifest

The current evidence index schema is:

```text
camera_system_quantitative_evidence_v1
```

Files:

- `camera_system_quantitative_evidence.json`: manifest with status,
  readiness booleans, evidence rows, blockers, and output pointers.
- `camera_system_quantitative_evidence.csv`: flat evidence table.
- `camera_system_quantitative_blockers.csv`: flat blocker table.
- `camera_system_quantitative_evidence.md`: human-readable report.

The manifest gathers existing FDTD convergence reports, 3D/2D crosstalk
convergence, DEVSIM convergence, field-LUT consumer validation, spectral
coverage, camera-system LUT artifacts, and the TCAD accuracy gate. Its status is
an evidence index. It must not be interpreted as product accuracy unless
`product_lut_ready=true`, which requires measured stack/material/device
calibration and full quantitative convergence.

## Full-Array Crosstalk Kernel

The crosstalk kernel is not the periodic supercell LUT. It simulates a finite,
guarded OCL/pixel neighborhood, illuminates the center output cell, and
integrates silicon absorption into:

- binned output cells, e.g. 3x3 output cells for a 3x3 camera crosstalk kernel
- raw physical PD cells, e.g. 9x9 raw PD cells for a 3x3 OCL neighborhood

Key columns in `crosstalk_kernel_summary.csv`:

- `mode`: `split-pd-1x1`, `ocl-2x2`, or `ocl-3x3`
- `neighborhood`: exported output-kernel size
- `simulation_neighborhood`: simulated output-cell neighborhood including guard cells
- `guard_cells`
- `raw_pd_kernel_shape`
- `center_response_fraction`
- `output_crosstalk_fraction`
- `outside_output_kernel_fraction`
- `truncation_response_fraction`: response outside the exported kernel support
- `support_edge_response_fraction`: response on the exported kernel border
- `si_internal_wavelength_pixels`: grid samples per wavelength inside silicon
- `minimum_critical_feature_pixels`: smallest configured critical optical
  feature in grid pixels, currently including DTI width, passivation thickness,
  and lens edge gap
- `grid_resolution_gate_pass`
- `convergence_status` in the manifest

For camera-system use, `truncation_response_fraction` must pass the configured
threshold and at least two resolution points at the largest exported support
must pass convergence. `support_edge_response_fraction` is a diagnostic showing
whether still-larger support is advisable; it is not itself counted as
truncated energy because border pixels are included in the exported kernel.
The grid gate must also pass: the default requires at least 8 grid pixels per
silicon internal wavelength and at least 2 pixels across critical optical
features. A low-resolution smoke run can be useful for plumbing, but it must not
be labeled an accuracy LUT when these gates fail.
If `crosstalk_convergence.json` is `FAIL` or `WARN`, treat the kernel as a
diagnostic simulation result, not an accuracy LUT.

## High-Resolution 2D Crosstalk X-Section

The high-resolution x-section runner writes:

```text
camera_crosstalk_xsection_fdtd_v1
```

Files:

- `crosstalk_xsection_output_kernel.csv`: binned output-cell x-line kernel.
- `crosstalk_xsection_raw_pd_kernel.csv`: raw physical-PD x-line absorption kernel.
- `crosstalk_xsection_summary.csv`: one row per mode/case/wavelength/resolution.
- `crosstalk_xsection_kernel.json`: manifest.
- `crosstalk_xsection_convergence.json`: convergence report.
- `crosstalk_xsection_kernel_lines.png`: visual diagnostic.

This runner is the practical local path for resolving the numerical grid blocker.
At 550 nm with the reference Si n,k, `84 px/um` gives about `11.33` grid pixels
per internal Si wavelength and `3.36` pixels on the 40 nm lens-edge feature.
The reference runs pass for:

- `split-pd-1x1`, 9-cell x-line support.
- `ocl-2x2`, 5-cell x-line support.
- `ocl-3x3`, 5-cell x-line support.

This is not a replacement for full 3D OCL footprint convergence. Use it for
CRA-x lateral crosstalk and DTI/passivation/grid convergence. Use the 3D
finite-array runner, or a calibrated 3D reduction, for final OCL footprint
coupling.

## Required CSV Columns

- `schema`
- `mode`
- `color_channel`
- `wavelength_nm`
- `case`
- `field_x_norm`, `field_z_norm`
- `cra_x_deg`, `cra_z_deg`
- `lens_shift_x_um`, `lens_shift_z_um`
- `aperture_shift_x_um`, `aperture_shift_z_um`
- `region_id`, `region_kind`, `region_ix`, `region_iz`
- `region_x_um`, `region_z_um`, `region_sx_um`, `region_sz_um`
- `response_model`
- `response`

## Diagnostic Columns

- `total_si_absorption_fraction_estimate`
- `signed_flux_si_absorption_fraction_diagnostic`: Si slab flux-balance
  diagnostic using the +Y flux-plane convention, computed as bottom-plane flux
  minus top-plane flux and normalized by incident power.
- `volume_absorption_region_fraction`
- `volume_absorption_region_raw`
- `volume_absorption_total_raw`
- `volume_absorption_scale_to_flux`
- `focal_region_fraction`
- `regional_flux_response_diagnostic`
- `pupil_integrated`, `pupil_ray_count`
- `ray_cra_x_deg`, `ray_cra_z_deg`

## Current Limits

The camera LUT schema is optical only. TCAD charge-collection efficiency is
computed in separate DEVSIM outputs for now. When it is promoted to a camera LUT
correction, add a separate correction term rather than changing the meaning of
`response`:

```text
electrical_response = integral(G_optical(x,y,z) * eta_collection(x,y,z) dV)
```

Until then, `response` should be treated as an optical absorption response, not
as final electron collection probability.

The finite-array crosstalk kernel is also optical absorption only. It includes
configured optical DTI geometry when a TCAD profile supplies it, but it does not
replace measured stack geometry, measured wavelength-dependent `n,k`, calibrated
electrical collection efficiency, or camera-system validation targets.

Measured geometry, implant/profile import, Gmsh mesh import, and calibration
schemas are documented separately:

```text
/Users/seongcheoljeong/FDTD/MEASURED_TCAD_SCHEMA.md
```
