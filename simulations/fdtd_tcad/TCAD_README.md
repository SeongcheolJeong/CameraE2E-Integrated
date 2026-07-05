# Open-Source TCAD Notes

This workspace now has a DEVSIM-based TCAD smoke path plus a Gmsh import and
calibration framework.

## Environment

DEVSIM, Gmsh, and SciPy are installed in:

```text
/Users/seongcheoljeong/FDTD/.tcad-env
```

Installed package:

```bash
/Users/seongcheoljeong/FDTD/.tcad-env/bin/pip show devsim
```

The tested versions are:

```text
devsim 2.10.0
gmsh 4.15.2
scipy 1.13.1
```

## 1D PN Photodiode Smoke

Run:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_pn_photodiode_1d.py
```

Outputs:

```text
/Users/seongcheoljeong/FDTD/runs/devsim_pn_photodiode_1d/summary.json
/Users/seongcheoljeong/FDTD/runs/devsim_pn_photodiode_1d/iv.csv
/Users/seongcheoljeong/FDTD/runs/devsim_pn_photodiode_1d/final_node_profile.csv
/Users/seongcheoljeong/FDTD/runs/devsim_pn_photodiode_1d/iv_curve.png
/Users/seongcheoljeong/FDTD/runs/devsim_pn_photodiode_1d/final_profile.png
```

The current smoke model is:

- 1D p+/n Si diode
- 2.8 um Si thickness
- 0.35 um junction depth
- reverse-bias sweep
- dark and illuminated runs
- Gaussian optical generation near the junction

This is not an image-sensor pixel model. It only proves that DEVSIM can solve a
drift-diffusion PN junction and that an optical-generation term can drive a
photocurrent.

## FDTD To TCAD Coupling Direction

Current optical LUT response is:

```text
response_optical = integral(Im(epsilon_Si) * |E|^2 dV over region)
```

For TCAD coupling, the desired physical flow is:

```text
FDTD absorption density -> G_optical(x,y,z,lambda,CRA)
DEVSIM drift-diffusion -> eta_collection(x,y,z,bias,region)
camera response -> integral(G_optical * eta_collection dV)
```

The 1D smoke script currently uses:

```text
OpticalGenerationRate = PhotoG0 * exp(-((x-JunctionX)^2)/(2*PhotoSigma^2))
```

The Meep supercell runner now exports an imported FDTD-derived generation table.
For the 1D approximation, it collapses the 3D FDTD absorption volume into a 1D
depth profile:

```text
G_1d(y) = sum_xz G_3d(x,y,z) / active_area
```

Then `devsim_pn_photodiode_1d.py` interpolates `G_1d` onto the DEVSIM mesh as
`OpticalGenerationRate`. For split-PD and OCL crosstalk, move to a 2D
cross-section or imported 3D tetrahedral mesh.

## End-To-End FDTD To DEVSIM Smoke

1. Export a TCAD generation profile from Meep:

```bash
cd /Users/seongcheoljeong/FDTD
/Users/seongcheoljeong/.local/bin/micromamba run \
  -p /Users/seongcheoljeong/FDTD/.meep-env \
  python meep_supercell_lut.py \
  --mode split-pd-1x1 \
  --split-mode dual-x \
  --wavelengths-nm 550 \
  --cases center:0:0:0:0:0:0 \
  --resolution 6 \
  --after-source-time 2 \
  --incident-photon-flux-cm2-s 1e20 \
  --output-dir runs/fdtd_to_tcad_generation_smoke
```

This writes:

```text
runs/fdtd_to_tcad_generation_smoke/tcad_generation_profile_1d.csv
runs/fdtd_to_tcad_generation_smoke/tcad_generation_profile_1d.npz
```

2. Import the FDTD profile into DEVSIM:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_pn_photodiode_1d.py \
  --generation-profile-csv runs/fdtd_to_tcad_generation_smoke/tcad_generation_profile_1d.csv \
  --generation-profile-case center \
  --generation-profile-wavelength-nm 550 \
  --output-dir runs/fdtd_to_tcad_devsim_1d \
  --reverse-bias-stop-v -1.0 \
  --reverse-bias-step-v -0.25
```

Smoke output:

```text
runs/fdtd_to_tcad_devsim_1d/summary.json
runs/fdtd_to_tcad_devsim_1d/iv.csv
runs/fdtd_to_tcad_devsim_1d/final_node_profile.csv
runs/fdtd_to_tcad_devsim_1d/iv_curve.png
runs/fdtd_to_tcad_devsim_1d/final_profile.png
```

The tested smoke run produced a `generation_source` of `imported_profile` and a
nonzero photocurrent delta at `-1 V`. This validates the file-level coupling
path, not the final sensor accuracy.

## 2D Split-PD Smoke

For split pixels, a 1D photodiode is the wrong abstraction because it cannot
represent lateral collection imbalance. The current 2D smoke script adds a
proxy pinned split-PD cross-section with one top anode, two bottom cathodes,
graded analytic top pinning, split collection columns, center isolation, and
side-DTI doping:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_split_pd_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case center \
  --generation-profile-wavelength-nm 550 \
  --electrical-model proxy-pinned-split-pd \
  --output-dir runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke
```

Outputs:

```text
runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json
runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/split_currents.csv
runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/node_profile_2d.csv
runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/split_currents.png
runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/node_maps.png
```

The center imported-map smoke run produced nearly equal left/right photo
current, with residual asymmetry from low-resolution FDTD and TCAD mesh
interpolation:

```text
generation_source          = imported_2d_map
electrical_model           = proxy-pinned-split-pd
left_photo_delta_a_per_cm  = 1.546946525071209e-06
right_photo_delta_a_per_cm = 1.5108225164672566e-06
photo_split_phase_x_proxy  = -0.011813844706131667
terminal_balance_illum     = 1.8730824962625344e-15 A/cm
```

To test CRA lateral response, use the same imported Meep `G(x, depth)` map for
an off-axis case:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_split_pd_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case edge20x \
  --generation-profile-wavelength-nm 550 \
  --electrical-model proxy-pinned-split-pd \
  --output-dir runs/devsim_split_pd_2d_fdtd_map_proxy_edge20x_smoke
```

The edge20x smoke run produced a right-biased collected-current split response:

```text
generation_source          = imported_2d_map
electrical_model           = proxy-pinned-split-pd
left_photo_delta_a_per_cm  = 3.4580149544765933e-06
right_photo_delta_a_per_cm = 3.654325098789889e-06
photo_split_phase_x_proxy  = 0.02760134398005003
terminal_balance_illum     = 1.8539679272583202e-15 A/cm
```

This validates left/right current extraction and a phase-like split metric. It
does not validate product pixel accuracy.

### 2D Split-PD Limits

- The default optical input is now imported Meep `G(x, depth)`, not analytic
  lateral gaussian shaping.
- Meep also exports `tcad_generation_volume_3d.npz` with `G(x, depth, z)` for a
  future 3D electrical import path.
- The electrical structure is still an analytic proxy. It includes pinning,
  split collection columns, center isolation, and side-DTI doping terms, but it
  is not a calibrated pinned photodiode with transfer gate, FD, realistic STI/DTI,
  surface pinning, or measured implants.
- DEVSIM 2D mesh contacts need adjacent dummy contact regions; the script uses
  those only to make contact ownership explicit. They are not physical layers.
- Currents are reported as A/cm because this is a 2D cross-section with
  implicit out-of-plane depth.
- The current result is suitable for software plumbing and sign/direction
  checks only. Do not use it as a sensor-accuracy LUT.

## Gmsh Measured-Profile Import Smoke

The open-source path now includes Gmsh mesh generation from the
`measured_tcad_profile_v1` geometry block and DEVSIM import of measured or proxy
doping profiles:

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

The verified mesh metadata uses the profile JSON as the geometry source and
creates:

```text
2D: 239 nodes, silicon region, anode/cathode_left/cathode_right contacts
3D: 1449 nodes, 3 silicon volumes, anode/cathode_left/cathode_right surfaces
```

Import the 2D mesh into DEVSIM and run a potential-only solve:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_gmsh_pixel_import.py \
  --mesh runs/gmsh_pixel_mesh/split_pixel_2d.msh \
  --dimension 2 \
  --measured-profile measured_profiles/example_proxy/profile.json \
  --output-dir runs/devsim_gmsh_pixel_import_2d
```

Import the 3D mesh:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_gmsh_pixel_import.py \
  --mesh runs/gmsh_pixel_mesh/split_pixel_3d.msh \
  --dimension 3 \
  --measured-profile measured_profiles/example_proxy/profile.json \
  --output-dir runs/devsim_gmsh_pixel_import_3d
```

Verified smoke outputs:

```text
runs/devsim_gmsh_pixel_import_2d/gmsh_pixel_2d_import_summary.json
runs/devsim_gmsh_pixel_import_3d/gmsh_pixel_3d_import_summary.json
```

Both runs imported the silicon region and three contacts, interpolated the
profile donor/acceptor fields onto DEVSIM nodes, and converged the potential
equation. This validates the mesh/profile plumbing. It is not yet a calibrated
charge-collection simulation.

## Reference PPD Profile Smoke

The non-measured but public-reference-anchored profile is:

```text
/Users/seongcheoljeong/FDTD/measured_profiles/reference_cmos_ppd_1p4um/profile.json
```

Use it with the executable profile model:

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

Run the off-axis CRA case:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_split_pd_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case edge20x \
  --generation-profile-wavelength-nm 550 \
  --electrical-model profile-ppd \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --width-um 1.4 \
  --depth-um 3.0 \
  --output-dir runs/devsim_split_pd_2d_reference_profile_edge20x_smoke
```

Current verified output:

```text
center photo_split_phase_x_proxy = -0.03853408422755443
edge20x photo_split_phase_x_proxy = -0.03958349063048183
terminal balance = < 1e-15 A/cm
```

The profile model applies TG barrier, FD doping, DTI liner, split isolation,
BDTI liner, and fixed-charge sheet terms to DEVSIM `NetDoping`. Dit interface
entries are also converted into potential-dependent trap charge and SRH
sheet-recombination proxy terms. A separate quasi-static TG/FD diagnostic can
generate a 2D Gmsh mesh with a real `floating_diffusion` DEVSIM terminal and
sweep the transfer-gate barrier scale:

```bash
.tcad-env/bin/python devsim_tg_fd_transfer_sweep_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --output-dir runs/devsim_tg_fd_transfer_sweep_2d_reference
```

This proves TG/FD solver wiring and terminal-current extraction. It is still
not a time-domain pinned-photodiode transfer/lag simulation.

A native transient diagnostic is also available. The preferred open-source
implementation is now the resolved-gate mode: the Gmsh mesh contains separate
silicon and oxide regions, a `silicon_oxide_interface`, a metal
`transfer_gate` boundary on the oxide, and a real `floating_diffusion` terminal
on silicon. DEVSIM solves the silicon drift-diffusion equations, the oxide
potential equation, and the Si/oxide potential-continuity interface equation
through the same transient run:

```bash
.tcad-env/bin/python devsim_tg_fd_transient_2d.py \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --tg-drive-mode resolved_gate \
  --output-dir runs/devsim_resolved_gate_transient_center_auto
```

This replaces the previous TG/FD-only proxy status with a real transient
drift-diffusion diagnostic with resolved gate oxide electrostatics. The
diagnostic runs paired photo-minus-dark sequences, exports Gmsh-triangle carrier
inventory, and fails the gate if FD current has not settled by the end of the
transfer pulse. In `resolved_gate` mode, the default open TG bias is 1.5 V and
an automatic 100-step open-gate hold is appended unless
`--transfer-open-hold-steps` is explicitly set. The default resolved-gate
transient tolerance is `--dd-relative-error 2e-5`, matching the observed
long-dark-transient residual floor on the reference mesh. The verified
reference smoke:

```text
runs/devsim_resolved_gate_transient_center_auto/tg_fd_transient_report.json
method                  = native_devsim_transient_bdf1_resolved_si_oxide_tg_bias_ramp_with_fd_terminal_photo_minus_dark
FD abs electrons/cm     = 6.8247970478645e9
FD fraction             = 0.9968265692723401
PD remaining fraction   = 0.40258696827233525
last_abs_current/peak   = 0.06935464086095092
settling gate           = PASS (< 0.10)
```

The mesh adds a small 5 nm oxide edge guard so silicon contacts and Si/oxide
interfaces do not share endpoint nodes. The optional `--fd-terminal-mode
circuit` path still exists, but the current reference geometry does not yet
produce a useful FD voltage response in that mode; the verified path therefore
uses the ohmic FD terminal current diagnostic.

Resolved DTI/BDTI oxide geometry is also executable. This is different from the
older side-wall doping proxy: Gmsh cuts side oxide trench regions out of the
silicon mesh, exports an `oxide` region plus `silicon_oxide_interface`, and
DEVSIM solves the oxide potential together with the silicon drift-diffusion
deck:

```bash
.tcad-env/bin/python tcad_gmsh_pixel_mesh.py \
  --dimension 2 \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --mesh-um 0.10 \
  --fine-mesh-um 0.025 \
  --include-dti-oxide \
  --output-dir runs/gmsh_split_pd_2d_resolved_dti_reference

.tcad-env/bin/python devsim_split_pd_2d.py \
  --mesh-source gmsh \
  --gmsh-mesh runs/gmsh_split_pd_2d_resolved_dti_reference/split_pixel_2d.msh \
  --width-um 1.4 \
  --depth-um 3.0 \
  --split-gap-um 0.04 \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case center \
  --generation-profile-wavelength-nm 550 \
  --electrical-model profile-ppd \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --reverse-bias-v -1 \
  --dd-max-iterations 220 \
  --output-dir runs/devsim_split_pd_2d_resolved_dti_center
```

Verified reference output:

```text
resolved_dti_oxide_solver_path = PASS
generation_integral_rel_error  = 2.04e-16
terminal_balance               = -8.26e-16 A/cm
node_count                     = 714
photo_split_phase_x_proxy      = -0.07984115994890183
```

This removes the "DTI only as metadata/proxy geometry" blocker for the open
framework path. It still does not make the DTI/BDTI product-accurate: the trench
width, depth, taper, liner charge/dose, oxide/fill material, and process
calibration are reference values until measured data is supplied.

Two experimental TG drive modes are available for model development:

```bash
# Direct semiconductor-surface TG potential contact. This is intentionally
# not the default because it is numerically stiff without a resolved oxide.
.tcad-env/bin/python devsim_tg_fd_transient_2d.py \
  --cases center \
  --tg-drive-mode gate_contact \
  --open-tg-barrier-scale 1.0 \
  --transfer-gate-closed-bias-v 0.0 \
  --transfer-gate-open-bias-v 0.2 \
  --output-dir runs/devsim_tg_contact_transient_smallbias_smoke

# Oxide-capacitance proxy: adds a Cox*(Vg-Psi_s) sheet-charge term under TG.
.tcad-env/bin/python devsim_tg_fd_transient_2d.py \
  --cases center \
  --tg-drive-mode gate_capacitance \
  --open-tg-barrier-scale 1.0 \
  --transfer-gate-closed-bias-v 0.0 \
  --transfer-gate-open-bias-v 1.0 \
  --transfer-gate-coupling-sign 1 \
  --output-dir runs/devsim_tg_cap_transient_open1_signp_center
```

Current evidence: `gate_contact` fails the reverse-bias solve on the reference
deck, confirming that directly clamping the silicon surface is not a good
replacement for oxide/poly TG physics. `gate_capacitance` converges and can
increase FD collection in short smokes, but it still fails transfer settling:
the center full-schedule run has `last_abs_current_to_peak = 1.0`. These modes
remain development references; use `resolved_gate` for the current executable
TG/FD path. Measured process calibration is still required before
accuracy-LUT use.

## Calibration Loop Smoke

The calibration loop runs a simulator command, reads each `summary.json`, and
fits configured parameters to target current/phase values:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_calibration_loop.py \
  --config configs/tcad_calibration_example.json \
  --output-dir runs/tcad_calibration_example \
  --max-evals 4
```

Outputs:

```text
runs/tcad_calibration_example/calibration_history.csv
runs/tcad_calibration_example/calibration_result.json
```

The current target file is synthetic and was generated from the proxy model:

```text
measured_profiles/example_proxy/calibration_targets.csv
```

The smoke run starts from `generation_map_scale = 0.8` and converges to:

```text
generation_map_scale = 0.9999995115237852
cost                 = 2.3860974394764713e-13
```

This proves the optimizer and file plumbing. It does not prove sensor accuracy.

For the reference profile:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_calibration_loop.py \
  --config configs/tcad_calibration_reference_profile.json \
  --output-dir runs/tcad_calibration_reference_profile \
  --max-evals 4
```

Current result:

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

Outputs:

```text
runs/tcad_calibration_target_report_reference/calibration_target_report.json
runs/tcad_calibration_target_report_reference/calibration_target_report.csv
runs/tcad_calibration_target_report_reference/calibration_target_metric_residuals.csv
```

This target file is synthetic, so the residual check can pass while
`product_accuracy_ready` remains false. Replace it with measured center/edge QE
and split-response targets before using the result for a camera-system accuracy
LUT.

Target CSVs can include additional numeric `target_*` columns. The calibration
loop maps known output names such as `target_dark_total_cathode_current_abs_a_per_cm`
and nested summary paths such as
`target_transport_summary__electron_mobility_min_cm2_v_s` into residual terms.
Use `weight_<metric>` columns to tune their influence when needed.

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

Current synthetic smoke result recovers the reference settings with:

```text
best_residual_norm = 4.520161417545061e-06
```

The profile-PPD solver also exposes runtime calibration controls for transport
and interface terms:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_transport_parameter_sensitivity.py \
  --output-dir runs/tcad_transport_sensitivity_reference
```

Outputs:

```text
runs/tcad_transport_sensitivity_reference/transport_sensitivity_report.json
runs/tcad_transport_sensitivity_reference/transport_sensitivity_report.csv
```

This verifies that `--lifetime-scale`, `--electron-mobility-scale`,
`--hole-mobility-scale`, `--fixed-charge-scale`,
`--interface-trap-density-scale`, and
`--interface-trap-recombination-scale` are wired into DEVSIM summaries and
solver responses. It is control wiring evidence, not measured transport
calibration.

## Accuracy Gate Report

Run:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_accuracy_gate.py \
  --profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --split-summary runs/devsim_native_response_sweep_2d_reference/cases/center_wl550nm/summary.json \
  --split-summary runs/devsim_native_response_sweep_2d_reference/cases/edge20x_wl550nm/summary.json \
  --gmsh-summary runs/devsim_gmsh_reference_import_2d/gmsh_pixel_2d_import_summary.json \
  --gmsh-summary runs/devsim_gmsh_reference_import_3d/gmsh_pixel_3d_import_summary.json \
  --resolved-dti-mesh-metadata runs/gmsh_split_pd_2d_resolved_dti_reference/mesh_metadata.json \
  --resolved-dti-split-summary runs/devsim_split_pd_2d_resolved_dti_center/summary.json \
  --convergence-report runs/convergence_public_anchor_smoke/convergence_report.json \
  --native-response-convergence-report runs/devsim_native_response_convergence_2d_reference/native_response_convergence_report.json \
  --tg-fd-report runs/devsim_tg_fd_transfer_sweep_2d_reference/tg_fd_transfer_sweep_report.json \
  --tg-fd-transient-report runs/devsim_tg_fd_transient_2d_reference/tg_fd_transient_report.json \
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
framework_blocking_failure_count = 0
accuracy_blocking_failure_count = 6
```

The framework gate now passes after the resolved-gate TG/FD transient update:
`tg_fd_transfer_settling` reports `last_abs_current_to_peak` of about 0.069 for
center and 0.071 for edge20x, below the 0.10 pass threshold. The resolved
DTI/BDTI oxide mesh/run also passes as framework evidence. Accuracy still fails
because the profile is reference/proxy, implants are not measured, mobility and
recombination use a reference doping- and field-dependent transport model that
is not measured or target-calibrated, the calibration targets are synthetic, and
the supplied optical-stack evidence is public-reference/proxy rather than
measured target-stack geometry and n,k.

Measured-profile and calibration schema details are documented in:

```text
/Users/seongcheoljeong/FDTD/MEASURED_TCAD_SCHEMA.md
```

## Design Viewer And VTK Export

For interactive design review, export the current DEVSIM/Gmsh and split-PD
results to ParaView-compatible ASCII VTK/VTU files and generate local HTML
viewers:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_design_viewer.py
```

Default outputs are written to:

```text
/Users/seongcheoljeong/FDTD/runs/tcad_design_viewer_reference
```

Important files:

```text
runs/tcad_design_viewer_reference/vtk/devsim_split_pd_2d_reference_profile_center_gmsh_native_center_native_split2d.vtu
runs/tcad_design_viewer_reference/vtk/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native_edge20x_native_split2d.vtu
runs/tcad_design_viewer_reference/vtk/gmsh_reference_2d_devsim.vtu
runs/tcad_design_viewer_reference/vtk/gmsh_reference_3d_devsim.vtu
runs/tcad_design_viewer_reference/viewers/cross_section_2d.html
runs/tcad_design_viewer_reference/viewers/geometry_3d.html
runs/tcad_design_viewer_reference/reports/parameter_sweep_comparison.md
runs/tcad_design_viewer_reference/reports/parameter_sweep_comparison.csv
runs/tcad_design_viewer_reference/reports/parameter_sweep_comparison.png
```

The 2D viewer overlays doping, potential, optical generation, device geometry,
and split current metrics for center and edge CRA cases. The 3D viewer shows the
microlens/CFA/passivation/Si/DTI/TG/FD proxy geometry with an optical generation
slice. The report compares center/edge CRA response, split phase, and terminal
balance.

To regenerate the solver-native split-PD path, first create a Gmsh mesh:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_gmsh_pixel_mesh.py \
  --dimension 2 \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --mesh-um 0.10 \
  --fine-mesh-um 0.025 \
  --output-dir runs/gmsh_split_pd_2d_reference_native
```

Then run center and edge CRA split-PD solves on that imported mesh:

```bash
.tcad-env/bin/python devsim_split_pd_2d.py \
  --mesh-source gmsh \
  --gmsh-mesh runs/gmsh_split_pd_2d_reference_native/split_pixel_2d.msh \
  --width-um 1.4 \
  --depth-um 3.0 \
  --split-gap-um 0.04 \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case center \
  --generation-profile-wavelength-nm 550 \
  --electrical-model profile-ppd \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --output-dir runs/devsim_split_pd_2d_reference_profile_center_gmsh_native

.tcad-env/bin/python devsim_split_pd_2d.py \
  --mesh-source gmsh \
  --gmsh-mesh runs/gmsh_split_pd_2d_reference_native/split_pixel_2d.msh \
  --width-um 1.4 \
  --depth-um 3.0 \
  --split-gap-um 0.04 \
  --generation-map-npz runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz \
  --generation-profile-case edge20x \
  --generation-profile-wavelength-nm 550 \
  --electrical-model profile-ppd \
  --measured-profile measured_profiles/reference_cmos_ppd_1p4um/profile.json \
  --output-dir runs/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native
```

These runs write `split_pd_2d_device.dat` from DEVSIM. `tcad_design_viewer.py`
uses that Tecplot file when present, so the split-PD VTK/VTU preserves solver
mesh connectivity. Older CSV-only split-PD summaries still fall back to a
Delaunay visualization mesh and should not be treated as authoritative solver
meshes.

## G*W Coupling Report

Run:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python tcad_gw_coupling.py
```

Default outputs are written to:

```text
/Users/seongcheoljeong/FDTD/runs/tcad_gw_coupling_reference
```

Important files:

```text
runs/tcad_gw_coupling_reference/gw_coupling_manifest.json
runs/tcad_gw_coupling_reference/gw_coupling_summary.csv
runs/tcad_gw_coupling_reference/gw_coupling_report.html
runs/tcad_gw_coupling_reference/gw_coupling_response.png
runs/tcad_gw_coupling_reference/gw_coupling_maps.png
runs/tcad_gw_coupling_reference/gw_coupling_nodes_center.csv
runs/tcad_gw_coupling_reference/gw_coupling_nodes_edge20x.csv
runs/tcad_gw_coupling_reference/camera_system_lut_summary.csv
runs/tcad_gw_coupling_reference/camera_system_lut_long.csv
runs/tcad_gw_coupling_reference/camera_system_lut.json
runs/tcad_gw_coupling_reference/camera_system_lut_report.html
```

The report integrates Meep `G(x,depth)` with two electrical weighting variants
on the Gmsh-native triangle mesh:

```text
W_proxy = analytic geometry/doping collection proxy
W_mesh  = FEM Laplace terminal weighting potential with Gmsh contact boundaries
q * integral G(x,depth) * W(x,depth) dA -> A/cm
```

Current normalized reference result:

```text
native_devsim center total response  = 1.236151874e-05 A/cm
native_devsim edge20x total response = 7.061945217e-06 A/cm
native_devsim edge20x / center       = 0.571284594

W_mesh edge20x total error after center reference scale = 0.0269363
W_devsim_dd_probe edge20x total error                  = -0.0120168
W_devsim_dd_probe split-phase error                    = 0.0681011  # gate fail
```

`W_proxy` split trend is fit to the native DEVSIM cathode electron-current
signal deltas with `split_transition_um=3.0` and `split_center_offset_um=0.07`.
Cathode hole-current and total-current deltas are retained as diagnostics, not
as the split-PD signal. `W_mesh` uses the
actual Gmsh mesh connectivity and physical contact boundaries, so it is a better
step toward a real weighting solve, but it is still a Laplace terminal weighting
potential, not a calibrated drift-diffusion collection probability.
`W_devsim_dd_probe`, when enabled, is interpolated from direct DEVSIM
local-generation drift-diffusion probes, so it includes the configured
transport, SRH/trap, and bias point, but it is sparse and uncalibrated. Do not
call these outputs a product LUT or a true DEVSIM adjoint weighting function.
For split-PD camera-system response, the primary accuracy-oriented open-source
path is now `native_devsim`: the full FDTD `G(x,depth)` map is imported into
DEVSIM and the left/right cathode electron-current deltas are measured directly.
Imported 2D generation maps are normalized to preserve the rectilinear FDTD
`integral G(x,depth) dA` on each Gmsh mesh. This removed a large mesh-sampling
dependence in the direct-response path; the native response convergence report
now compares `0.10/0.025 um` and `0.08/0.020 um` mesh levels and passes the
current 5% total-response / 0.05 split-phase gates.
The G*W rows are retained as speed-oriented surrogate diagnostics and must pass
their error gates against `native_devsim` before being promoted.

The camera-system diagnostic export contains these methods:

```text
native_devsim        : DEVSIM cathode electron-current signal delta per split PD
gw_proxy_ref_scaled  : G*W_proxy response, center-scaled to native DEVSIM
gw_mesh_ref_scaled   : G*W_mesh response, center-scaled to native DEVSIM
gw_devsim_laplace_ref_scaled : G*W_devsim_laplace response, center-scaled
gw_devsim_dd_probe_ref_scaled: sparse DD-probe response, center-scaled
```

`camera_system_research_lut.json` and `camera_system_research_lut.npz` are the
primary camera-system ingestion artifacts at the current reference/proxy
calibration level. They contain only the direct `native_devsim` response tensor,
with axes `[case, region_id]`, so the camera simulation does not need to choose
between proxy and surrogate methods. `camera_system_native_devsim_response.json`
and `camera_system_native_devsim_response.npz` contain the same direct-solver
response in a lower-level diagnostic form.

The expanded CRA reference now uses a five-point chief-ray sweep
`0/5/10/15/20 deg` at `550 nm` and exports the camera-system research LUT from
the r20 FDTD generation map plus native DEVSIM terminal-current solves:

```text
runs/tcad_gw_coupling_cra5_r20_gridsnap_smoke/camera_system_research_lut.json
runs/tcad_gw_coupling_cra5_r20_gridsnap_smoke/camera_system_research_lut.npz
runs/tcad_gw_coupling_cra5_r20_gridsnap_smoke/camera_system_research_lut_report.html
```

The tensor shape is `[5, 2]` for cases by split-PD region. The r15/r20 optical
convergence report is embedded in the LUT metadata and currently reports
`SMOKE_CONVERGENCE_FAIL`; the maximum total-response delta is still about 56.8%.
The current run uses `--grid-snap-y nearest`, which adjusts only bottom-air
padding so the Meep y cell length lands exactly on the FDTD grid. This removes
the previous cell-rounding artifact while preserving the active microlens/CFA/
passivation/Si layer thicknesses. Use this artifact for camera-system trend
development, not product accuracy.
`camera_system_diagnostic.json` and `camera_system_diagnostic.npz` contain all
native and surrogate methods for comparison. `camera_system_lut.json` is a
blocked product-LUT marker until measured stack/n,k, calibrated electrical
targets, and convergence gates all pass.
`product_lut_ready` is intentionally `false`.

The practical fix for the optical side is now encoded in the runners rather than
only documented as a warning. `meep_supercell_lut.py` writes grid-resolution
metadata for every summary row, and `run_convergence_sweep.py` fails quantitative
reports when the Si internal wavelength or critical optical features are
under-resolved. For the current 550 nm green stack, the computed minimum 3D FDTD
resolution is `60 px/um`; r20/r25 are rejected by this gate:

```text
runs/convergence_center_r20_r25_gridsnap_gate_smoke/convergence_report.json
passed: false
max total-response delta: 93.8%
min Si internal-wavelength pixels: 2.70 / required 8
recommended minimum resolution: 60 px/um
```

A grid-qualified r60 center/cra10x/edge set is kept as the imported baseline for
heavier 3D convergence runs:

```text
runs/supercell_center_r60_quant_smoke/camera_lut_summary.csv
runs/supercell_center_r60_quant_smoke/tcad_generation_map_2d.npz
runs/supercell_center_r60_quant_smoke/tcad_generation_volume_3d.npz
runs/supercell_cra10x_r60_quant_smoke/camera_lut_summary.csv
runs/supercell_cra10x_r60_quant_smoke/tcad_generation_map_2d.npz
runs/supercell_cra10x_r60_quant_smoke/tcad_generation_volume_3d.npz
runs/supercell_edge20x_r60_quant_smoke/camera_lut_summary.csv
runs/supercell_edge20x_r60_quant_smoke/tcad_generation_map_2d.npz
runs/supercell_edge20x_r60_quant_smoke/tcad_generation_volume_3d.npz
```

Those r60 rows are not the default LUT source anymore. The current default
optical map is still the r80 half of the integer-grid r60/r70/r80 CRA3
convergence run:

```text
runs/convergence_cra3_r60_r70_r80_gridsnap_quant/convergence_report.json
runs/convergence_cra3_r60_r70_r80_gridsnap_quant/r80_t8_pml0p45/camera_lut_summary.csv
runs/convergence_cra3_r60_r70_r80_gridsnap_quant/r80_t8_pml0p45/tcad_generation_map_2d.npz
runs/convergence_cra3_r60_r70_r80_gridsnap_quant/r80_t8_pml0p45/tcad_generation_volume_3d.npz
```

The r60/r70/r80 spatial report passes the resolution sweep at 5% tolerance with
`max_total_response_rel_delta_to_reference=0.0480`, zero negative signed-flux
diagnostics, and no lateral-period or axis-rounding grid issues. An r72 probe
also had a small response delta, but it is rejected for the 1.4 um periodic cell
because `1.4 * 72 = 100.8` grid pixels. Quantitative 3D periodic-cell
convergence should use integer-safe resolutions for this pitch, for example
multiples of 5 such as r60/r70/r80.

The camera-system research LUT now uses the combined full-axis convergence
report:

```text
runs/convergence_cra3_full_axes_quant/convergence_report.json
runs/convergence_cra3_full_axes_quant/r80_t12_pml0p45/camera_lut_summary.csv
runs/convergence_cra3_full_axes_quant/r80_t8_pml0p60/camera_lut_summary.csv
```

That report passes all configured optical numerical axes at 5% tolerance:
`spatial_convergence_pass=true`, `time_convergence_pass=true`,
`pml_convergence_pass=true`, and `full_numerical_convergence_pass=true`.
The maximum total-response deltas are 0.0480 for resolution, 0.0115 for
after-source time, and 1.13e-5 for PML. Negative signed-flux diagnostics are
zero.

The r80 optical map is imported into native DEVSIM terminal-current solves and
exported as the Studio's default camera-system research LUT:

```text
runs/native_devsim_research_lut_cra3_r80_quant/camera_system_research_lut.json
runs/native_devsim_research_lut_cra3_r80_quant/camera_system_research_lut.npz
runs/native_devsim_research_lut_cra3_r80_quant/camera_system_research_lut_report.html
```

The tensor shape is `[3, 2]` for `center/cra10x/edge20x` by `pd_left/pd_right`.
The native-DEVSIM response tensor is:

```text
center  : left 5.8015e-4 A/cm, right 5.8940e-4 A/cm, split 0.0079
cra10x  : left 2.8240e-4 A/cm, right 9.0716e-4 A/cm, split 0.5252, total/reference 1.0171
edge20x : left 2.9709e-4 A/cm, right 9.3550e-4 A/cm, split 0.5179, total/reference 1.0539
```

Because the remaining inputs are still unmeasured, the workflow also exports a
camera-system uncertainty envelope. It applies completed stress-variant response
deltas to the current native-DEVSIM nominal LUT and is intended for system
simulation risk ranges, not product certification:

```text
runs/camera_system_uncertainty_lut_reference/camera_system_uncertainty_lut.json
runs/camera_system_uncertainty_lut_reference/camera_system_uncertainty_lut.csv
runs/camera_system_uncertainty_lut_reference/camera_system_uncertainty_lut.html
runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.json
runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.csv
runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.html
runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.npz
runs/camera_system_field_lut_query_reference/field_lut_query.json
runs/camera_system_field_lut_query_reference/field_lut_query.csv
```

The first envelope is deliberately conservative: it uses completed lens-height,
split-gap, and front-fixed-charge stress variants for all three CRA cases
(`center`, `cra10x`, and `edge20x`). The lens-height stress map is regenerated
with the same r80 CRA3 FDTD settings as the nominal research LUT, so the
envelope no longer mixes r6 smoke optical deltas with the r80 nominal LUT. This
makes the camera-system consumer see a nominal response and a stress-derived
min/max response while the measured stack, measured n,k, implant, and
calibrated transport blockers remain open.

For split-PD channels the envelope treats total response and split phase as
independent stress bounds. Therefore left-channel minimum uses the minimum total
response with maximum split phase, left-channel maximum uses the maximum total
response with minimum split phase, and right-channel bounds use the matching
minimum/minimum and maximum/maximum combinations. Rows record
`bound_method=independent_total_split_stress_envelope_v1`.

The envelope also includes the nominal value inside each min/max band. This
prevents a stress set that moves only one direction from reporting a range that
excludes the baseline response. The dense `camera_system_field_lut` then
interpolates these bands over `field_x_norm` with
`piecewise_linear_3_anchor`; polynomial interpolation is intentionally avoided
until measured field-response calibration exists.

For camera-system integration, the dense field LUT is also exported as
`camera_system_field_lut.npz` with typed arrays that can be loaded with
`numpy.load(..., allow_pickle=False)`. `camera_system_field_lut_query.py`
performs the consumer-side contract check and arbitrary field query: it verifies
the JSON/NPZ match, validates response bounds, and recomputes left/right channels
from total and split so the queried rows remain internally consistent.

This replaces both the legacy r20 CRA5 smoke LUT and the earlier r60 CRA3 LUT as
the Studio's default research LUT. The r70 CRA3 LUT remains a retained
intermediate reference, but Studio now points at the r80 export with full
resolution/time/PML optical convergence evidence. It is still not a product LUT:
the remaining blockers are measured stack geometry, measured optical n,k,
measured implant/profile sources, calibrated mobility/recombination, and
measured calibration targets.

Optical n,k input now has a dedicated interpolation and evidence check:

```text
runs/optical_nk_interpolation_check/optical_nk_interpolation_check.json
```

The FDTD material loader sorts wavelength rows before interpolation, rejects
duplicate wavelengths, rejects `n <= 0` and `k < 0`, and fails when a requested
wavelength is outside the table range. `optical_stack_evidence.py` records the
same checks per material, including required wavelength coverage for 0.45, 0.55,
and 0.65 um in the reference runbook.

For design-iteration crosstalk, the high-resolution 2D x-section path is already
the usable reference:

```text
runs/crosstalk_xsection_2d_reference/crosstalk_xsection_convergence.json
runs/crosstalk_xsection_2d_reference/crosstalk_xsection_kernel.json
runs/crosstalk_xsection_2d_reference/crosstalk_xsection_output_kernel.csv
```

That merged x-section reference passes r72/r84 convergence for split-pd-1x1,
2x2 OCL, and 3x3 OCL center/edge20x cases. The TCAD accuracy gate now consumes
this convergence report directly as `fdtd_crosstalk_xsection_convergence`, so
the solved crosstalk piece is tracked as PASS rather than hidden behind the
legacy r20 3D smoke LUT.

## Image Sensor Pixel Studio

The Lumerical-style UX target is not a pixel-accurate clone of a commercial
tool. The practical open-source target is a local design workbench with the same
main workflow structure: object tree, property inspector, 2D/3D viewers, result
manager, accuracy gate, and reproducible runbook.

Run:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_pixel_studio.py
```

Default outputs are written to:

```text
/Users/seongcheoljeong/FDTD/runs/image_sensor_pixel_studio_reference
```

Open:

```text
runs/image_sensor_pixel_studio_reference/index.html
```

The Studio reads:

```text
configs/image_sensor_pixel_studio_reference.json
runs/tcad_design_viewer_reference/manifest.json
runs/tcad_accuracy_gate_reference_profile/tcad_accuracy_gate.json
runs/tcad_gw_coupling_reference/gw_coupling_manifest.json
configs/image_sensor_design_space_reference.json
runs/devsim_split_pd_2d_reference_profile_center_gmsh_native/summary.json
runs/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native/summary.json
```

The current Studio v0 provides:

```text
Next Action      : Overview advisor translating orchestrator/run-manager state into copyable preview/execute commands
Object Tree       : project, process stack, pixel geometry, implants, TG/FD/interface entries, solvers
Properties        : selected object JSON plus proxy/reference and solver-wired/metadata-only markers
Design Space      : parameter registry, command builder, rerun invalidation tags, candidate variants, and public UX/workflow source links
Variant Compare   : baseline-relative table for completed candidate runs, including total response and split-phase deltas
Run Manager       : stage-by-stage status for materialized variants, inferred from expected output files
Dataset Catalog   : structured table of monitor-like arrays, meshes, reports, LUTs, and project inputs
2D/3D Views       : embedded cross-section and geometry viewers
Native Coupling  : embedded native-DEVSIM response report plus optional G*W surrogate diagnostics
Camera LUT        : embedded camera-system LUT table and CSV/JSON/NPZ outputs
Results Manager  : VTU/VTK/HTML/CSV/PNG/JSON outputs with native/derived labels
Accuracy Gate     : blocking items that prevent product-LUT use
Runbook           : exact commands for native mesh, center/cra10x/edge DEVSIM, viewer, native research LUT, and Studio regeneration
```

Design variants can be materialized into isolated input JSON files and run
plans:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_variant_builder.py
```

This writes:

```text
runs/image_sensor_design_variants_reference/variant_run_manifest.json
runs/image_sensor_design_variants_reference/<variant_id>/inputs/stack_config.json
runs/image_sensor_design_variants_reference/<variant_id>/inputs/tcad_profile.json
runs/image_sensor_design_variants_reference/<variant_id>/inputs/studio_project.json
runs/image_sensor_design_variants_reference/<variant_id>/run_plan.sh
runs/image_sensor_design_variants_reference/<variant_id>/variant_manifest.json
```

The generated `run_plan.sh` files are execution plans, not completed
simulation evidence. They are useful because they preserve the baseline inputs
and make each candidate's required rerun stages explicit before any heavy
FDTD/DEVSIM work is launched.

The builder uses content-stable writes, so re-materializing the same variant no
longer makes downstream stages stale solely from JSON/text timestamp churn. It
also rewrites stack material `n,k` table references to absolute paths inside
variant `inputs/stack_config.json` files, because relative paths such as
`../materials/...` are otherwise wrong after the stack file is copied into a
variant input directory.

Completed variants can be compared against the baseline:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_variant_compare.py
```

This writes:

```text
runs/image_sensor_design_variants_reference/variant_comparison.csv
runs/image_sensor_design_variants_reference/variant_comparison.json
runs/image_sensor_design_variants_reference/variant_comparison_report.html
```

Missing variant outputs stay visible as `planned_only` or `partial`; they are
not treated as zeros. Completed rows are trend evidence only and keep
`product_lut_ready=false`.

Run status and dataset catalog artifacts can be generated without launching
solver jobs:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_run_manager.py
```

This writes:

```text
runs/image_sensor_design_variants_reference/run_manager_status.csv
runs/image_sensor_design_variants_reference/run_manager_status.json
runs/image_sensor_design_variants_reference/run_manager_status.html
runs/image_sensor_design_variants_reference/dataset_catalog.csv
runs/image_sensor_design_variants_reference/dataset_catalog.json
runs/image_sensor_design_variants_reference/dataset_catalog.html
```

The status is file-inferred, not a real scheduler. It is meant to answer
which stage outputs exist, which stages are still missing, and which missing
stages are blocked by upstream missing outputs. The dataset catalog includes
the main Studio artifacts plus existing files under materialized variant
directories, so completed candidate runs become visible as additional
DEVSIM/Gmsh/G*W/Camera-LUT datasets.

Important limit: this is currently a design/result shell, not a full
solver-running CAD environment. It helps inspect and compare the open-source
Meep/Gmsh/DEVSIM outputs, but it does not make the reference/proxy simulation
product-accurate. Parameters marked `metadata-only` are visible design records,
not proof that the current solver equations use them. `G*W` coupling now
includes analytic `W_proxy`, independent mesh Laplace `W_mesh`, DEVSIM-native
pure-Laplace `W_devsim_laplace`, and sparse direct-solve DD probe
`W_devsim_dd_probe`. The primary response path for split-PD rows is direct
`native_devsim`, because sparse DD-probe superposition does not currently pass
the split-phase surrogate gate. The remaining product-accuracy gap is measured
stack/n,k, measured/calibrated process and transport targets, broader native
angle/wavelength sweeps, and passing convergence/calibration gates.

The UX target and rationale are recorded in:

```text
LUMERICAL_UX_GOAL.md
```

## Variant Orchestrator

The workspace now includes a local sequential runner for the materialized design
variant run plans:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_variant_orchestrator.py --all --next-needed
```

The default mode is dry-run. It previews which stage would run next, including
missing or stale stages, and keeps heavy Meep/convergence stages skipped unless
explicitly allowed. To execute a safe already-complete Studio refresh stage:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_variant_orchestrator.py \
  --variant split_gap_plus_50pct \
  --stage studio \
  --rerun-complete \
  --execute
```

Outputs:

```text
runs/image_sensor_design_variants_reference/orchestrator_last_run.json
runs/image_sensor_design_variants_reference/orchestrator_history.jsonl
runs/image_sensor_design_variants_reference/orchestrator_logs/
```

After a successful executed run, the orchestrator refreshes variant comparison,
run-manager, dataset-catalog, and Studio artifacts. This is still not a product
accuracy gate: `accuracy_ready` and `product_lut_ready` remain false until the
electrical model and measured calibration targets pass.

The Studio Overview now includes a next-action advisor built from the latest
orchestrator and run-manager data. It distinguishes ordinary runnable stages
from heavy Meep/convergence stages. Heavy execute commands keep
`--include-heavy` in the copied command, so the UI does not accidentally suggest
a command that the orchestrator would skip.

Each orchestrator plan row also carries a static preflight summary. The
preflight parses the command without launching it, verifies the executable,
micromamba prefix when present, Python script, input paths, and output target
parent directories. For stack configs, it also checks referenced material `n,k`
tables. The preflight records lightweight runtime hints such as heavy-stage
status, FDTD resolution, wavelength count, case count, and after-source time.
This is an execution-safety check, not a runtime estimate or physics-quality
gate.

Current reference run snapshot:

```text
Run Manager stages : 14 complete / 14 fresh
Missing stages     : 0
Stale stages       : 0
Failed stages      : 0
Existing datasets  : 206
Executed variants  : lens_height_plus_8pct, split_gap_plus_50pct,
                     front_fixed_charge_stress
```

This snapshot means the local workflow is clean and reproducible. It does not
change the accuracy position: `accuracy_ready=false` and
`product_lut_ready=false`.

The run manager also reports stage freshness:

```text
fresh   = tracked input files are older than the expected output files
stale   = at least one tracked input file is newer than a completed stage output
missing = expected stage outputs are not complete
```

This is useful for Lumerical-style design iteration because a geometry/profile
edit can immediately show which stages need rerun. It is not a convergence or
accuracy pass; a fresh proxy output can still be unsuitable for product LUT use.
The orchestrator uses this freshness signal when `--next-needed` or
`--next-stale` is selected.

## Ad-Hoc Design Variant Creation

For Lumerical-style design iteration, use the design variant creator to turn a
parameter edit into isolated variant configs and a run plan:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python image_sensor_design_variant_create.py \
  --id custom_split_gap_050um \
  --label "Custom split gap 0.050um" \
  --param split_gap_um=0.05
```

The default mode is preview-only. It validates the parameter id/path, shows the
old and new values, checks the design-space range, reports metadata-only
warnings, and lists required rerun stages. To write the isolated variant files
and refresh the Studio management views:

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

This still does not execute solver stages. Use the orchestrator after
materialization to preview or run the required stage sequence.

The same flow is exposed in the Studio Design Space tab through the Design Edit
Command Builder. It generates preview/materialize commands, shows the tracked
profile/stack path, lists required rerun stages, and warns when a parameter is
metadata-only or outside its recommended range.

## DEVSIM Weighting Potential Export

The current G*W pipeline now has a solver-native terminal weighting-potential
export:

```bash
cd /Users/seongcheoljeong/FDTD
.tcad-env/bin/python devsim_weighting_potential_2d.py \
  --mesh runs/gmsh_split_pd_2d_reference_native/split_pixel_2d.msh \
  --output-dir runs/devsim_weighting_potential_2d_reference
```

Outputs:

```text
runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.csv
runs/devsim_weighting_potential_2d_reference/weighting_potential_2d_summary.json
runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.dat
runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.png
```

The baseline run solved pure-Laplace terminal weighting potentials for anode,
left cathode, and right cathode on the Gmsh-native 2D mesh. The all-contact
weighting sum was numerically consistent with 1.0:

```text
node_count = 712
sum_all_contacts_max_abs_error_to_one = 1.1102230246251565e-16
```

`tcad_gw_coupling.py` consumes this CSV through:

```bash
.tcad-env/bin/python tcad_gw_coupling.py \
  --devsim-weighting-csv runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.csv
```

Camera-system diagnostic exports include these methods when the DEVSIM
weighting and DD-probe CSVs are present:

```text
native_devsim
gw_proxy_ref_scaled
gw_mesh_ref_scaled
gw_devsim_laplace_ref_scaled
gw_devsim_dd_probe_ref_scaled
```

Important limit: `W_devsim_laplace` is solver-native, but it is still a pure
Laplace terminal weighting potential. It does not include the drift-diffusion
carrier transport response from recombination, mobility, trap occupancy, TG
transfer, or FD transient behavior. It is not a calibrated drift-diffusion
adjoint collection probability. `W_devsim_dd_probe` is closer to the DD physics
because each probe is a direct DEVSIM drift-diffusion solve, but the current
reference map is sparse, interpolated, 2D, and uncalibrated.

The DD-probe response is retained as a diagnostic only. Dense 7x7 and bilinear
interpolation experiments did not pass the split-phase gate, so the open-source
camera-system response path remains full-map `native_devsim`, not local-probe
superposition.

## Next Accuracy Step

For CRA/split/OCL camera-system LUTs, the next meaningful accuracy step is:

```text
Meep 2D/3D absorption map G(x,y,z,lambda,CRA)
-> DEVSIM 2D/3D collection model eta_collection(x,y,z,bias)
-> collected charge per PD region
-> camera LUT columns for center/edge/split/OCL response
```

The current implementation has the optical `G(x, depth)` and `G(x, depth, z)`
exports, the 2D proxy collection simulation, the Gmsh 2D/3D mesh import smoke,
the `G*W` coupling report with `W_proxy`, `W_mesh`, `W_devsim_laplace`, and
sparse `W_devsim_dd_probe`, Dit
trap-charge/SRH sheet terms, configurable BDTI proxy liners, doping-dependent
low-field mobility, field-dependent velocity-saturation edge mobility, SRH
lifetime node models, and the calibration loop. The missing accuracy step is
the calibrated electrical model on measured geometry: transfer gate, FD,
DTI/BDTI geometry and doses, mobility, recombination, lifetime terms, and a
true drift-diffusion adjoint collection probability must be fit to measured
targets.

If measured process information is unavailable, keep the result labeled as a
proxy model. The first measured inputs to prioritize are Si stack geometry,
implant/contact geometry, measured n,k tables, and at least one electrical or
optical calibration target.

## Public Sources

- DEVSIM install and platform support:
  https://devsim.net/gettingstarted.html
- DEVSIM introduction and feature list:
  https://devsim.org/introduction.html
- DEVSIM source:
  https://github.com/devsim/devsim
- DEVSIM meshing example:
  https://devsim.net/meshing.html
- DEVSIM Gmsh command reference:
  https://devsim.net/CommandReference.html
- DEVSIM diode example documentation:
  https://devsim.net/examples_diode.html
- Gmsh:
  https://gmsh.info/
- Gmsh reference manual:
  https://gmsh.info/doc/texinfo/

## Limits To Keep Explicit

- This is not Synopsys/Silvaco-equivalent image-sensor TCAD.
- The example measured profile is intentionally marked `is_measured: false`.
- Real measured implant tables can be imported, but the included implants are
  graded analytic reference presets, not measured implant profiles.
- Transfer-gate and floating-diffusion geometry can be represented in the
  profile schema; the 2D profile model applies TG/FD proxy doping terms and the
  optional FD-contact mesh exposes a `floating_diffusion` DEVSIM terminal.
  `devsim_tg_fd_transient_2d.py --tg-drive-mode resolved_gate` now solves a
  2D silicon/oxide TG transient with a metal gate boundary and passes the
  current-settling diagnostic on the reference center case. It is still not a
  foundry-calibrated PPD/TG/FD deck.
- Interface-trap density is solver-coupled in the 2D split-PD profile model as a
  potential-dependent trap-charge and SRH sheet-recombination proxy, but it is
  not calibrated to measured Dit, dark current, lag, or PRNU targets.
- The reference profile includes DTI liners and configurable BDTI proxy liner
  doping boxes, but the current mesh is still a simple split-pixel silicon proxy,
  not a measured trench/oxide/fill topology.
- 1D cannot model lateral split-PD collection or OCL crosstalk; use
  `tcad_generation_map_2d.npz` with `devsim_split_pd_2d.py` for the current
  lateral smoke path.
- The Gmsh 3D import path is implemented only as a potential solve smoke. Full
  3D drift-diffusion collection from `G(x, depth, z)` is not implemented yet.
- Product accuracy requires process geometry, doping, measured/calibrated
  mobility/recombination, interface-trap model, and measured optical generation.
