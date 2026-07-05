# CameraE2E DB Catalog

This document explains how CameraE2E data sources are organized and how to use them as runtime parameters.

The goal is not to copy every DB into one physical directory. The goal is to expose one searchable catalog over bundled fallback assets and external high-fidelity DBs.

## Core API

Use these functions from `pyisetcam`:

```python
from pyisetcam import (
    camerae2e_db_catalog,
    camerae2e_db_search,
    camerae2e_db_get,
    camerae2e_db_parameters,
    camerae2e_db_summary,
)
```

- `camerae2e_db_catalog()`: returns all known DB/LUT/model-profile entries.
- `camerae2e_db_search(query, family=..., tags=...)`: searches by text, family, role, status, or tags.
- `camerae2e_db_get(name)`: returns one catalog entry as a JSON-friendly dictionary.
- `camerae2e_db_parameters(name)`: returns paths and values intended for direct API use.
- `camerae2e_db_summary()`: returns family/status counts and active entries.

## Families

### Lens

Main entries:

- `lens_patents_active`: the active Lens DB selected by runtime discovery.
- `rayoptics_lens_db_v9`: external RayOptics v9 package when present.
- `lens_patents_bundled_v6`: bundled fallback Lens DB.

Parameter use:

```python
from pyisetcam import camerae2e_db_parameters, lens_patent_search, lens_patent_raytrace_optics

params = camerae2e_db_parameters("lens_patents_active")
rows = lens_patent_search(db_path=params["db_path"], require_camerae2e=True, limit=5)
optics = lens_patent_raytrace_optics(
    rows[0]["simulation_id"],
    psf_dir=params["highres_psf_dir"],
    target_psf_size=64,
)
```

Relevant env vars:

- `PYISETCAM_LENS_PATENT_DB`
- `PYISETCAM_LENS_DB_ROOT`
- `PYISETCAM_LENS_PATENT_PSF_DIR`

Important limitation: RayOptics PSF assets are geometric ray-histogram PSFs, not diffraction wave-optics PSFs.

### Sensor FDTD

Main entries:

- `fdtd_sensor_lut_active`: active FDTD optical-response LUT.
- `fdtd_sensor_stack_catalog`: external sensor stack catalog from the FDTD workspace.

Image-sensor selector APIs:

```python
from pyisetcam import (
    image_sensor_db_records,
    image_sensor_db_parameters,
    image_sensor_db_config,
    image_sensor_db_optimize_camera_parameters,
)

records = image_sensor_db_records("sony", limit=10)
params = image_sensor_db_parameters(records[0]["sensor_id"])
config = image_sensor_db_config(records[0]["sensor_id"], strategy="hybrid")
optimization = image_sensor_db_optimize_camera_parameters(
    records[0]["sensor_id"],
    strategy="analytic_only",
    preset="exposure",
    parameter_space={"sensor.integration_time": [0.001, 0.004]},
)
```

The selector parameters include:

- `catalog_path`
- `stack_config_path`
- `tcad_profile_path`
- `lut_path`
- `generation_map_path`
- `collection_summary_paths`
- `accuracy_gate_path`

`image_sensor_db_config(...)` returns a CameraE2E scenario fragment plus a
policy manifest. Use `strategy="hybrid"` when DB/LUT paths should be attached
and analytic proxy defaults should fill only metadata-derived axes. Use
`strategy="analytic_only"` for broad free-configuration optimization without
attaching the external FDTD/TCAD paths. Use `strategy="lut_only"` when the DB/LUT
side should be inspected without analytic OCL group fallback.
`image_sensor_db_optimize_camera_parameters(...)` runs the standard FACA
optimizer from that selected sensor DB record and adds `source_image_sensor_db`
lineage so downstream RAW exports can keep the sensor provenance.

Parameter use:

```python
from pyisetcam import camerae2e_db_parameters, fdtd_sensor_config, sensor_attach_fdtd_lut

params = camerae2e_db_parameters("fdtd_sensor_lut_active")
config = fdtd_sensor_config(params["lut_path"])
sensor = sensor_attach_fdtd_lut(sensor, config)
```

Relevant env vars:

- `PYISETCAM_FDTD_LUT_PATH`
- `PYISETCAM_FDTD_ROOT`

### Sensor TCAD / DEVSIM

Main entry:

- `tcad_sensor_db_active`: active TCAD/DEVSIM collection DB rooted in the FDTD workspace.

Parameter use:

```python
from pyisetcam import camerae2e_db_parameters, tcad_sensor_db_load, sensor_attach_tcad_lut

params = camerae2e_db_parameters("tcad_sensor_db_active")
db = tcad_sensor_db_load(
    generation_map_path=params["generation_map_path"],
    collection_summary_paths=params["collection_summary_paths"],
    accuracy_gate_path=params["accuracy_gate_path"],
)
sensor = sensor_attach_tcad_lut(sensor, db)
```

Relevant env var:

- `PYISETCAM_FDTD_ROOT`

### HW ISP

Main entry:

- `hwisp_parameter_profiles`: timing, transport, and 3A profile DB.

Parameter use:

```python
from pyisetcam import camerae2e_db_parameters, hw_isp_config_from_profile

params = camerae2e_db_parameters("hwisp_parameter_profiles")
config = hw_isp_config_from_profile(params["default_profile"])
```

Relevant env var:

- `PYISETCAM_HWISP_DB`

### Perception

Main entry:

- `task_perception_model_profiles`: model profile catalog for YOLO, tracking, detection, segmentation, pose, OBB, and optional adapters.

Parameter use:

```python
from pyisetcam import camerae2e_db_parameters, task_model_config_from_profile, task_model_from_config

params = camerae2e_db_parameters("task_perception_model_profiles")
config = task_model_config_from_profile(params["profile_names"][0])
model = task_model_from_config(config)
```

Relevant env var:

- `PYISETCAM_TASK_MODEL_CACHE`

### Upstream ISETCam Assets

Main entry:

- `isetcam_upstream_snapshot`: pinned upstream ISETCam asset snapshot used by `AssetStore`.

Parameter use:

```python
from pyisetcam import AssetStore

store = AssetStore.default()
```

Relevant env vars:

- `PYISETCAM_CACHE_ROOT`
- `PYISETCAM_UPSTREAM_ROOT`

### MATLAB Parity Baselines

Main entry:

- `matlab_parity_baselines`: curated `.mat` baselines and `cases.yaml`.

Parameter use:

```bash
python tools/parity_report.py
PYISETCAM_RUN_PARITY=1 pytest -q tests/parity
```

## HTML And JSON Report

Regenerate the catalog report:

```bash
python tools/render_camerae2e_db_catalog_report.py
```

Outputs:

- `reports/db_catalog/camerae2e_db_catalog.html`
- `reports/db_catalog/camerae2e_db_catalog.json`

Regenerate the image-sensor selector report:

```bash
python tools/render_sensor_db_overview.py
```

Outputs:

- `reports/sensor_db/sensor_db_overview.html`
- `reports/sensor_db/sensor_db_summary.json`
- `reports/sensor_db/images/*.png`

The sensor selector contains six panels per selected sensor:

- sensor stack structure
- OCL / pixel optical response
- relative illumination / QE-like response
- TCAD / charge-collection proxy
- CameraE2E impact
- parameter bundle

Important: OCL, field-response, and CameraE2E-impact panels use the active FDTD reference LUT unless a matching per-sensor FDTD LUT has been generated.

Use the JSON file when another script needs a machine-readable inventory.

## Recommended Usage Pattern

1. Search by block or data type.
2. Select an entry by `name`.
3. Call `camerae2e_db_parameters(name)`.
4. Pass returned values into the block-specific CameraE2E API.

Example:

```python
from pyisetcam import camerae2e_db_search, camerae2e_db_parameters

sensor_sources = camerae2e_db_search(family="sensor", include_missing=False)
fdtd_params = camerae2e_db_parameters("fdtd_sensor_lut_active")
```
