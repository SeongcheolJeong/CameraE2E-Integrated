# Measured TCAD Profile And Calibration Schema

This workspace uses a plain JSON/CSV schema for process-aware TCAD inputs:

```text
measured_tcad_profile_v1
```

The schema is intentionally independent of a proprietary TCAD deck. It lets the
open-source flow import measured geometry, implant/doping tables, interface
metadata, and calibration targets.

## What Open Source Can Do

With the current Gmsh + DEVSIM flow, open source can:

- generate 2D/3D meshes from supplied geometry dimensions
- preserve Gmsh physical groups for silicon regions and contacts
- import the mesh into DEVSIM
- interpolate measured or proxy donor/acceptor profiles onto DEVSIM nodes
- run a Poisson potential smoke solve on imported 2D/3D meshes
- fit simulator parameters to measured or synthetic targets with SciPy

Public documentation basis:

- DEVSIM supports user-defined PDEs and 1D/2D/3D simulation:
  https://devsim.org/introduction.html
- DEVSIM supports Gmsh mesh import through `create_gmsh_mesh`,
  `add_gmsh_region`, `add_gmsh_contact`, and `add_gmsh_interface`:
  https://devsim.net/CommandReference.html
- Gmsh is an open-source 3D mesh generator with CAD and physical groups:
  https://gmsh.info/

## What Open Source Does Not Provide

Open-source software does not provide these product-specific facts by itself:

- measured implant profiles
- transfer-gate and floating-diffusion process geometry
- real DTI/BDTI geometry
- measured interface trap density and capture cross sections
- calibrated mobility/recombination/lifetime model parameters
- measured optical stack `n,k`
- measured calibration targets for QE, split response, dark current, lag, or PRNU

If these are not supplied and calibrated, the output must remain labeled as a
proxy or framework smoke result. It should not be called an accuracy LUT.

## Profile JSON Fields

Example:

```text
/Users/seongcheoljeong/FDTD/measured_profiles/example_proxy/profile.json
```

Top-level fields:

- `schema`: must be `measured_tcad_profile_v1`
- `profile_name`: human-readable profile id
- `units`: expected `um` length and `cm^-3` doping
- `geometry`: pixel width/depth/z-width, split gap, and optional TG/FD geometry
- `background`: uniform donor/acceptor background
- `implants`: one or more doping sources
- `electrical_features`: executable or metadata electrical additions such as
  TG barrier boxes, FD doping boxes, and fixed-charge sheets
- `interfaces`: interface-trap density and capture-cross-section inputs
- `mobility_recombination`: carrier mobility, field saturation, and SRH
  lifetime model labels, measured/calibrated flags, and model parameters
- `calibration_status`: whether the values are measured or proxy

Supported implant entries:

- `csv_scattered`: reads `x_um,depth_um,z_um,donor_cm3,acceptor_cm3`
- `analytic_box`: rectangular donor/acceptor block for proxy tests
- `analytic_smooth_box`: donor/acceptor block with configurable tanh edge
  rolloff and optional Gaussian `x_peak_um/x_sigma_um` or
  `depth_peak_um/depth_sigma_um`; this is still a reference model unless
  explicitly calibrated to measured implant data

Supported executable electrical feature entries:

- `doping_box`: adds donor/acceptor terms to DEVSIM `NetDoping`
- `fixed_charge_sheet`: converts sheet charge in `cm^-2` to an equivalent
  signed volume doping over `sheet_thickness_um`
- `metadata_only`: records data that should not affect the solve yet
- `mobility_region`: reserved for explicit region-dependent transport overrides;
  the current DEVSIM path consumes the top-level `mobility_recombination` block
  as doping-dependent low-field mobility nodes, field-dependent effective edge
  mobility, and SRH lifetime node models

The loader function `electrical_terms_from_profile()` returns:

```text
Donors, Acceptors, FixedChargeDoping, feature_summary
```

`devsim_split_pd_2d.py --electrical-model profile-ppd` and
`devsim_gmsh_pixel_import.py --measured-profile ...` both use:

```text
NetDoping = Donors - Acceptors + FixedChargeDoping
```

This means TG/FD/fixed-charge proxy terms are executable in DEVSIM. In
`devsim_split_pd_2d.py`, Dit interface entries are also converted into a
potential-dependent trap-charge term and an SRH sheet-recombination proxy. It
does not by itself mean gate transient transfer or FD readout capacitance are
calibrated. For framework evidence, `devsim_tg_fd_transient_2d.py
--tg-drive-mode resolved_gate` exercises a resolved Si/oxide TG boundary, and
`tcad_gmsh_pixel_mesh.py --include-dti-oxide` exports resolved side DTI/BDTI
oxide regions that have been smoke-tested through DEVSIM. Product accuracy
still requires measured TG/FD/DTI geometry, implant profiles, interface
parameters, and target calibration.

For `csv_scattered`, the loader supports these interpolation modes:

- `linear_nearest` (default): linear scattered interpolation inside the measured
  convex hull, with nearest-neighbor fallback outside the hull so DEVSIM nodes
  remain finite.
- `linear`: linear scattered interpolation only; it fails if any queried DEVSIM
  node falls outside the measured convex hull.
- `nearest`: deterministic nearest-neighbor interpolation, retained for smoke
  comparisons and sparse public examples.
- `idw`: inverse-distance weighting with optional `idw_power`, default `2.0`.

The DEVSIM summaries record per-implant interpolation metadata, including point
count, query count, interpolation dimension, SciPy availability, and
outside-hull fallback count. A production calibration flow should still document
dose-preservation checks against the measured profile's native export.

Run the interpolation smoke check:

```bash
cd /Users/seongcheoljeong/FDTD
python3 measured_profile_interpolation_check.py \
  --output-dir runs/measured_profile_interpolation_check
```

Output:

```text
runs/measured_profile_interpolation_check/interpolation_check.json
```

## Generated Artifacts

Generate 2D and 3D meshes using geometry from the profile:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_gmsh_pixel_mesh.py \
  --dimension both \
  --measured-profile measured_profiles/example_proxy/profile.json \
  --output-dir runs/gmsh_pixel_mesh
```

Outputs:

```text
runs/gmsh_pixel_mesh/split_pixel_2d.msh
runs/gmsh_pixel_mesh/split_pixel_3d.msh
runs/gmsh_pixel_mesh/mesh_metadata.json
```

Import and solve in DEVSIM:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_gmsh_pixel_import.py \
  --mesh runs/gmsh_pixel_mesh/split_pixel_2d.msh \
  --dimension 2 \
  --measured-profile measured_profiles/example_proxy/profile.json \
  --output-dir runs/devsim_gmsh_pixel_import_2d
```

For 3D, use `--dimension 3` and `split_pixel_3d.msh`.

## Public-Reference Profile

The practical non-measured profile is:

```text
/Users/seongcheoljeong/FDTD/measured_profiles/reference_cmos_ppd_1p4um/profile.json
```

It is intentionally marked:

```text
reference_mode = true
calibration_status.is_measured = false
```

The file stores public anchors for a 1.4 um-class dual/split PPD exploration
deck, including vertical PPD depth and literature doping values, P+ pinning
layer estimates, positive dielectric charge impact, and silicon mobility
references. Those anchors are not used blindly as executable values. The
active defaults are lower, stable values because the high-dose values are too
aggressive for the current coarse open-source drift-diffusion deck.

Run the reference profile smoke:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_split_pd_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case center \
  --generation-profile-wavelength-nm 550 \
  --electrical-model profile-ppd \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --width-um 1.4 \
  --depth-um 3.0 \
  --output-dir runs/devsim_split_pd_2d_reference_profile_center_smoke
```

The current verified values are:

```text
center:
  left_photo_delta_a_per_cm  = 1.0668213005415525e-06
  right_photo_delta_a_per_cm = 1.061680826392776e-06
  photo_split_phase_x_proxy  = -0.0024150664844203363

edge20x:
  left_photo_delta_a_per_cm  = 2.282685426203794e-06
  right_photo_delta_a_per_cm = 2.2867886928868222e-06
  photo_split_phase_x_proxy  = 0.0008979735033152538
```

The edge split response is intentionally modest with this stable reference
profile. If the camera-system model requires a stronger PDAF phase response,
that should be calibrated against measured center/edge split targets rather
than forced by arbitrary doping.

## TechInsights SIMS Seed Profiles

Local TechInsights DEP reports with extracted SIMS/SPM doping tables can be
converted into executable seed profiles:

```bash
cd /Users/seongcheoljeong/FDTD
python3 build_tcad_candidate_report.py --include-pdf
python3 build_tcad_sims_seed_profiles.py
```

Generated outputs:

```text
/Users/seongcheoljeong/FDTD/measured_profiles/techinsights_sims_seed/index.html
/Users/seongcheoljeong/FDTD/measured_profiles/techinsights_sims_seed/manifest.json
/Users/seongcheoljeong/FDTD/measured_profiles/techinsights_sims_seed/<code>*/profile.json
```

These profiles are different from the proxy profiles under
`image_sensor_db/generated_tcad_profiles/`:

- SIMS/SPM table concentrations and depth anchors are carried into the profile.
- 2D lateral placement, smoothing, overlap handling, and contact placement are
  inferred so the open-source solver can run.
- Periphery rows and non-silicon/interface rows such as DTI poly-Si fill or
  backside passivation are retained as `metadata_only` unless a resolved
  material/interface model exists.
- `calibration_status.is_measured` remains `false`; measured table anchors do
  not make the generated profile a calibrated process deck.

Use one as a stronger TCAD starting point, for example:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_gmsh_pixel_mesh.py \
  --dimension 2 \
  --measured-profile measured_profiles/techinsights_sims_seed/dep_2511_801_samsung_hp5/profile.json \
  --output-dir runs/gmsh_hp5_sims_seed
```

## Image Sensor TCAD Structure DB

For broader image-sensor coverage, generate the TCAD structure database:

```bash
cd /Users/seongcheoljeong/FDTD
python3 build_tcad_candidate_report.py --include-pdf
python3 build_tcad_sims_seed_profiles.py
python3 build_tcad_structure_db.py
```

Generated local outputs:

```text
/Users/seongcheoljeong/FDTD/image_sensor_db/tcad_structure_db/index.html
/Users/seongcheoljeong/FDTD/image_sensor_db/tcad_structure_db/structure_summary.csv
/Users/seongcheoljeong/FDTD/image_sensor_db/tcad_structure_db/manifest.json
/Users/seongcheoljeong/FDTD/image_sensor_db/tcad_structure_db/validation.json
/Users/seongcheoljeong/FDTD/image_sensor_db/tcad_structure_db/models/*.json
/Users/seongcheoljeong/FDTD/image_sensor_db/tcad_structure_db/starter_profiles/*/profile.json
```

This DB makes the candidate/structure information usable by assigning each
image sensor a TCAD-oriented structure model:

- single-pixel cell dimensions, recommended mesh size, and DTI side bounds
- active silicon, DTI/STI, photodiode, P-well, pinning, TG, FD/S-D, and optical
  stack fields
- SIMS seed profile links where available
- estimated doping archetype anchors where measured tables are absent
- a recommended TCAD profile path for every record: SIMS seed first, existing
  TechInsights proxy second, and estimated structure starter profile last
- `source_kind`, `confidence`, and `method` for every important value
- readiness level such as `sims_seed_ready`, `structure_ready_high`,
  `structure_ready_medium`, and `proxy_structure_low`

This is the best broad-coverage database for selecting sensors and starting
TCAD sensitivity sweeps. It still must not be treated as a calibrated product
deck: most records rely on empirical/rule-based inference for missing DTI,
photodiode, TG/FD, and doping details.

## Image Sensor Optical/CFA/QE DB

Optical-stack, CFA, microlens, grid, spectral, and QE evidence is kept in a
separate database:

```bash
cd /Users/seongcheoljeong/FDTD
python3 build_optical_qe_db.py
```

Generated local outputs:

```text
/Users/seongcheoljeong/FDTD/cfa_proxy_nk_library.json
/Users/seongcheoljeong/FDTD/image_sensor_db/optical_qe_db/index.html
/Users/seongcheoljeong/FDTD/image_sensor_db/optical_qe_db/optical_qe_summary.csv
/Users/seongcheoljeong/FDTD/image_sensor_db/optical_qe_db/manifest.json
/Users/seongcheoljeong/FDTD/image_sensor_db/optical_qe_db/validation.json
/Users/seongcheoljeong/FDTD/image_sensor_db/optical_qe_db/models/*.json
```

Each per-sensor model uses schema `image_sensor_optical_qe_model_v1` and keeps:

- CFA pattern, representative thickness, thickness range, and color-filter pitch
- microlens/OCL type and pitch
- optical stack height, pixel pitch, active silicon thickness, grid material,
  and grid pitch
- NIR/modality/illumination context
- reported QE points when explicit `% at wavelength` text exists
- qualitative QE/PDE/spectral-response mentions when no numeric point is found
- common proxy CFA `n,k(lambda)` and thickness-scaled absorption-only
  transmission when product-specific CFA optical constants are unavailable
- `source_kind`, `confidence`, `method`, and local source snippets for
  extracted versus inferred fields

Important limitation: the QE fields are not measured QE curves. The current
local source set exposes only reported point claims for one image sensor and
qualitative QE/spectral-response mentions for a small set of others. Treat
`inferred_rule` pitch values as optical setup defaults, not measured CFA/OCL
geometry. Treat `inferred_proxy_nk` CFA optical constants as common color-resist
setup proxies, not product-specific measured `n,k`; they are suitable for FDTD
setup and sensitivity sweeps, but not calibrated colorimetry.

## Calibration Config

Calibration loop schema:

```text
tcad_calibration_config_v1
```

Example:

```text
/Users/seongcheoljeong/FDTD/configs/tcad_calibration_example.json
```

Run:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_calibration_loop.py \
  --config configs/tcad_calibration_example.json \
  --output-dir runs/tcad_calibration_example \
  --max-evals 4
```

The example target file is synthetic:

```text
/Users/seongcheoljeong/FDTD/measured_profiles/example_proxy/calibration_targets.csv
```

A real accuracy loop should replace it with measured targets and fit physically
meaningful parameters, not just a scalar generation scale.

Reference-profile calibration smoke:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_calibration_loop.py \
  --config configs/tcad_calibration_reference_profile.json \
  --output-dir runs/tcad_calibration_reference_profile \
  --max-evals 4
```

This uses:

```text
/Users/seongcheoljeong/FDTD/measured_profiles/reference_cmos_ppd_1p4um/calibration_targets_synthetic.csv
```

The current result fits the synthetic smoke targets with:

```text
generation_map_scale = 0.9999993920036888
cost                 = 3.696772138547732e-13
```

Validate the best-fit row against per-target tolerances:

```bash
cd /Users/seongcheoljeong/FDTD
python3 tcad_calibration_target_report.py \
  --calibration-result runs/tcad_calibration_reference_profile/calibration_result.json \
  --targets-csv measured_profiles/reference_cmos_ppd_1p4um/calibration_targets_synthetic.csv \
  --output-dir runs/tcad_calibration_target_report_reference
```

This writes:

```text
runs/tcad_calibration_target_report_reference/calibration_target_report.json
runs/tcad_calibration_target_report_reference/calibration_target_report.csv
runs/tcad_calibration_target_report_reference/calibration_target_metric_residuals.csv
```

Default tolerances are 5% relative error for total current and 0.005 absolute
error for split phase unless tolerance columns are provided in the target CSV.
Because the current target file is synthetic, residuals can pass while
`product_accuracy_ready` remains false.

Additional measured target metrics can be added as numeric `target_*` columns.
For nested DEVSIM summary values, use double underscores, for example:

```text
target_transport_summary__tau_n_min_s
target_interface_trap_summary__recombination_coeff_max_s1
target_doping_summary__fixed_charge_doping_max_cm3
```

Multi-parameter transport/interface calibration smoke:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_calibration_loop.py \
  --config configs/tcad_calibration_transport_reference.json \
  --output-dir runs/tcad_calibration_transport_reference \
  --max-evals 8

python3 tcad_calibration_target_report.py \
  --calibration-result runs/tcad_calibration_transport_reference/calibration_result.json \
  --targets-csv measured_profiles/reference_cmos_ppd_1p4um/calibration_targets_transport_synthetic.csv \
  --output-dir runs/tcad_calibration_transport_target_report_reference \
  --max-abs-normalized-residual 0.01
```

This is still synthetic, but it verifies that measured dark-current,
transport, fixed-charge, and interface-trap targets can drive the same
least-squares loop.

Validate that runtime transport/interface calibration controls are executable:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_transport_parameter_sensitivity.py \
  --output-dir runs/tcad_transport_sensitivity_reference
```

This writes:

```text
runs/tcad_transport_sensitivity_reference/transport_sensitivity_report.json
runs/tcad_transport_sensitivity_reference/transport_sensitivity_report.csv
```

This proves runtime control wiring for mobility, lifetime, fixed charge, and
interface-trap scales. It does not make the transport model measured or
calibrated.

## Accuracy Gate

Generate the automated gate report:

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

Current result:

```text
framework_ready = true
accuracy_ready  = false
accuracy_blocking_failure_count = 6
```

Before using any result as a sensor-accuracy LUT, require all of the following:

- `calibration_status.is_measured` is true for geometry and implant sources.
- Measured optical stack geometry and measured `n,k` are used.
- FDTD convergence report passes for the exact stack, wavelengths, CRA cases,
  and pupil settings.
- Signed flux diagnostic is not treated as pass when the report flags it as
  failed.
- Gmsh mesh quality and mesh-refinement sensitivity are documented.
- DEVSIM terminal current balance and mesh convergence pass.
- Interface traps, TG/FD, DTI/BDTI, mobility, recombination, and lifetime terms
  are implemented in the electrical model and calibrated to measured targets,
  not only present as profile records. The reference profile now exercises
  resolved TG oxide transient and resolved DTI/BDTI oxide mesh smoke paths plus
  doping- and field-dependent transport models, but all remain unmeasured
  reference parameters.
- Calibration residuals pass against measured center/edge response, split
  response, and any dark/lag targets required by the camera-system model.

Until those gates pass, this is a calibration-capable open-source TCAD
framework, not a calibrated product TCAD deck.
