# Image Sensor DB

Generated at `2026-06-23T18:46:36.797271+00:00` from local TechInsights export folders under:

`/Users/seongcheoljeong/Sensor_DB_TechInsight`

This database is for FDTD/TCAD setup exploration. It intentionally stores extracted
metadata and local source paths, not full report bodies.

## Contents

- `sensor_catalog.json`: normalized records, extracted specs, evidence snippets, and source file paths.
- `sensor_catalog.csv`: flat table for spreadsheet filtering.
- `index.html`: local browser table with links to source HTML and generated configs.
- `generated_stack_configs/`: runnable `image_sensor_stack_v1` proxy stack configs.
- `generated_tcad_profiles/`: runnable `measured_tcad_profile_v1` proxy profiles.
- `validation.json`: extraction counts and coverage.

## Coverage

- Records: 224
- Source records before image-sensor filter: 543
- Parsed PDF files: 420
- Generated FDTD stack configs: 216
- Generated TCAD proxy profiles: 216
- Records with pixel pitch: 216
- Records with active Si thickness: 155
- Records with optical stack height: 178
- Records with DTI type: 154
- Records with transfer-gate type: 42

## Important Limitations

- These generated stack/profile files are not measured process decks.
- The catalog is filtered to image-sensor/CIS records; camera modules, packaging-only reports, ToF/LiDAR/SPAD/fingerprint/thermal records are excluded by default.
- Pixel pitch, active Si thickness, optical stack height, DTI, CFA/OCL, transfer-gate, and similar values are extracted from local report metadata, HTML tables, and PDF text snippets when present.
- CFA, microlens, passivation split and material n,k are inherited proxy assumptions unless the source explicitly exposed more detail.
- Doping profiles are scaled from the existing reference proxy TCAD profile, not extracted from TechInsights implant data.
- Treat all generated configs as starting points for simulation setup, then run convergence and accuracy gates before quantitative use.
