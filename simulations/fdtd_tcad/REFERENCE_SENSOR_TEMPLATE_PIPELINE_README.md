# Reference Sensor Template Pipeline

This pipeline turns local `sensor_db` records into sensor-specific CAD templates,
solver provenance, optional Meep smoke simulations, and an analysis database.

It is intended for benchmark and trend exploration of known image sensors. It is
not a product-accuracy process deck generator.

## Input

Default input:

```bash
sensor_db/sensor_catalog.json
```

The `sensor_db` directory is local and ignored by git because it may be derived
from licensed/reference material.

## Generate Main Sensor Templates

Use the TCAD environment because CAD generation depends on Gmsh/OpenCASCADE:

```bash
.tcad-env/bin/python reference_sensor_template_pipeline.py \
  --max-sensors 8 \
  --output-dir runs/reference_sensor_template_analysis
```

Generated outputs:

- `runs/reference_sensor_template_analysis/analysis_catalog.json`
- `runs/reference_sensor_template_analysis/analysis_catalog.csv`
- `runs/reference_sensor_template_analysis/index.html`
- `runs/reference_sensor_template_analysis/templates/<sensor_id>/model.step`
- `runs/reference_sensor_template_analysis/templates/<sensor_id>/model.brep`
- `runs/reference_sensor_template_analysis/templates/<sensor_id>/geometry_import.json`
- `runs/reference_sensor_template_analysis/templates/<sensor_id>/analysis_record.json`

## Run Smoke Simulations

To run Meep smoke cases for the first selected sensors:

```bash
.tcad-env/bin/python reference_sensor_template_pipeline.py \
  --max-sensors 8 \
  --run-smoke \
  --smoke-count 2 \
  --resolution 4 \
  --after-source-time 0.3 \
  --output-dir runs/reference_sensor_template_analysis
```

The script invokes `.meep-env/bin/python meep_supercell_lut.py` for each smoke
case and stores command, logs, artifact index, and parsed JSON schema metadata
inside the analysis catalog.

## Current Reference Run

The current generated DB contains eight high-score main image sensors from
`sensor_db`, including Samsung HP5/HP1/HP3-class records, Sony LYT/IMX records,
and OmniVision OV52A/L3A records.

Two smoke runs were executed:

- `DEP-2511-801 Samsung HP5`
- `DEF-2604-801 Sony LYT-901`

Both smoke commands completed with exit code `0` and generated
`camera_lut.json`, response maps, focal maps, and TCAD generation exports.

Important: these were low-resolution smoke runs. The generated `camera_lut.json`
records show `grid_resolution_gate_pass: false`, so they prove wiring and
artifact generation, not quantitative accuracy.

## Topology Mapping

The script maps sensor records to reusable template topologies:

- `super_qpd` or `qpd` -> QPD split-PD no-shield template.
- `nona` -> 3x3 OCL/Nona template.
- `quad`, `tetracell`, `four_shared`, or `eight_shared` -> Quad 2x2 OCL template.
- Otherwise -> Bayer 1x1 OCL 3x3 neighborhood template.

Template dimensions are derived from:

- `pixel_pitch_um`
- `active_si_thickness_um`
- `cfa_thickness_um`
- `dti_depth_um`
- `dti_width_um`
- generated proxy stack geometry when available

## Accuracy Boundary

The generated STEP/BREP files are CAD review artifacts from extracted metadata
and proxy stack values. They are not measured product CAD.

Do not treat smoke results as camera-system LUT data. For quantitative use,
each sensor still needs:

- measured mask or SEM-derived geometry
- measured OCL surface/profilometry
- measured CFA/passivation/Si thickness and material n,k
- calibrated implant, trap, mobility, and recombination parameters
- convergence pass at the requested wavelength/CRA/field conditions
