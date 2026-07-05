# FDTD/TCAD-Informed Sensor Block

This feature connects the `/Users/seongcheoljeong/FDTD` optical simulation repo to the `pyisetcam` sensor block through lookup-table layers.

It is optional. Existing `sensor_compute(...)` behavior is unchanged unless an FDTD or TCAD config is attached to the sensor.

## What It Models

The current integration uses FDTD camera LUTs to model:

- wavelength-dependent optical response correction
- field / chief-ray-angle response rolloff
- center-to-edge optical shading
- 3x3 regional-response crosstalk proxy
- downstream impact on raw sensor volts and ISP output

The TCAD/DEVSIM collection layer is separate and uses:

- `tcad_generation_map_2d.npz` for FDTD-derived `G(x, depth)` generation maps
- DEVSIM `summary.json` files for left/right cathode photo-current deltas
- `tcad_accuracy_gate.json` to distinguish framework readiness from product accuracy readiness
- optional collection-efficiency scaling when explicitly attached through `sensor_attach_tcad_lut(...)` or `sensor_attach_physics_lut(...)`

The default source path is:

```bash
/Users/seongcheoljeong/FDTD/runs/supercell_lut_ocl_3x3_smoke/camera_lut.json
```

You can override the FDTD repo root with:

```bash
export PYISETCAM_FDTD_ROOT=/path/to/FDTD
```

## Important Physics Boundary

The current FDTD DB is an optical absorption / regional response proxy. TCAD/DEVSIM artifacts now add a carrier-collection framework, but the available profile is still not product calibrated.

Do not present this as a full sensor-electrical model. The correct claim is:

> FDTD-informed optical sensor response layer for microlens/CFA/Si-stack response, field rolloff, and regional-response crosstalk proxy.

For TCAD/DEVSIM, the correct claim is:

> FDTD-to-TCAD framework evidence for generation-map ingestion, split-PD collection-current plumbing, split-phase proxy, and terminal-current balance. Treat current outputs as proxy simulation until the TCAD accuracy gate passes with measured inputs.

Also note that the available `*_smoke` LUTs are flow checks. Product-grade quantitative use needs convergence sweeps for resolution, runtime after source, wavelength grid, CRA grid, and field grid.

## API

```python
from pyisetcam import (
    fdtd_sensor_lut_load,
    fdtd_sensor_lut_validate,
    sensor_attach_fdtd_lut,
    sensor_attach_physics_lut,
    tcad_sensor_db_load,
    tcad_sensor_validate,
    sensor_compute,
)

lut = fdtd_sensor_lut_load()
print(fdtd_sensor_lut_validate(lut))
tcad_db = tcad_sensor_db_load()
print(tcad_sensor_validate(tcad_db))

sensor = sensor_attach_fdtd_lut(
    sensor,
    lut,
    mode="field+crosstalk",
    crosstalk_strength=0.25,
)
sensor = sensor_compute(sensor, oi)

sensor = sensor_attach_physics_lut(
    sensor,
    fdtd_lut=lut,
    tcad_db=tcad_db,
    fdtd_kwargs={"mode": "field+crosstalk", "crosstalk_strength": 0.25},
    tcad_kwargs={"case": "edge20x", "mode": "collection"},
)
```

Supported mode tokens:

- `qe`
- `field`
- `crosstalk`
- combinations such as `qe+field+crosstalk`

## Report

Generate the verification report with:

```bash
python tools/render_fdtd_sensor_report.py
python tools/render_fdtd_sensor_physics_report.py
python tools/render_fdtd_tcad_sensor_report.py
```

Outputs:

- `reports/fdtd_sensor/fdtd_sensor_report.html`
- `reports/fdtd_sensor/fdtd_sensor_summary.json`
- `reports/fdtd_sensor/physics_validation_report.html`
- `reports/fdtd_sensor/physics_validation_summary.json`
- `reports/fdtd_sensor/fdtd_tcad_sensor_report.html`
- `reports/fdtd_sensor/fdtd_tcad_sensor_summary.json`
- `reports/fdtd_sensor/fdtd_response_rolloff.png`
- `reports/fdtd_sensor/fdtd_crosstalk_kernel.png`
- `reports/fdtd_sensor/sensor_volts_triptych.png`
- `reports/fdtd_sensor/center_edge_crops.png`
- `reports/fdtd_sensor/isp_output_triptych.png`
- `reports/fdtd_sensor/ip_channel_means.png`
- `reports/fdtd_sensor/physics_ri_cos4.png`
- `reports/fdtd_sensor/physics_ocl_shift.png`
- `reports/fdtd_sensor/physics_wavelength_absorption.png`
- `reports/fdtd_sensor/physics_energy_budget.png`
- `reports/fdtd_sensor/physics_kernel_locality.png`
- `reports/fdtd_sensor/fdtd_tcad_fdtd_response.png`
- `reports/fdtd_sensor/fdtd_tcad_generation_maps.png`
- `reports/fdtd_sensor/fdtd_tcad_devsim_currents.png`
- `reports/fdtd_sensor/fdtd_tcad_split_balance.png`

The report covers the Phase 1-5 evidence path: LUT loading, validation, field/CRA response, crosstalk proxy, raw sensor volts, center/edge crops, and ISP output impact.

The physics report covers whether the result is physically plausible:

- relative illumination compared with the first-order `cos^4(CRA)` curve
- energy-budget sanity checks
- OCL/compensated edge response compared with uncompensated response
- wavelength trend from the available Si absorption sweep
- signed-CRA symmetry coverage
- FDTD convergence metadata coverage
- crosstalk-kernel locality

The integrated FDTD + TCAD report covers:

- FDTD optical response across wavelength / CRA / field cases
- `G(x, depth)` generation maps used as DEVSIM input
- DEVSIM left/right photo-current deltas
- split-PD phase proxy
- terminal-current balance
- TCAD accuracy-gate pass/fail table

Current known issue:

- the available `ocl-3x3` smoke LUT has a near-uniform 3x3 regional-response kernel and its `edge20x_ocl` case is lower than `edge20x_uncomp`
- the separate `cra_response_lut` shows the expected compensated improvement
- treat this as a smoke-LUT limitation or shift-rule issue until a product-grade 3x3 LUT is regenerated

## Recommended Next DB Work

For better sensor-block fidelity, generate product-grade FDTD LUTs rather than relying on smoke outputs:

- wavelength sweep: at least R/G/B representative bands, ideally aligned to sensor wave sampling
- CRA sweep: center to max chief-ray angle
- field sweep: center, mid-field, edge, corner
- OCL shift sweep: uncompensated and product shift rule
- localized optical crosstalk source or tagged source-region simulation, not only uniform regional response
- convergence sweeps for FDTD resolution and after-source time

If full photodiode charge collection is required, pair this optical LUT with TCAD or a calibrated compact electrical collection model.

For product accuracy, the TCAD DB must be regenerated with measured optical stack `n,k`, measured implant/profile data, real DTI/BDTI geometry, calibrated mobility/recombination/interface terms, and measured QE/split/dark/lag calibration targets. Until then, reports and metadata intentionally label the result as proxy simulation.
