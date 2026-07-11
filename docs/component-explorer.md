# Lens/Sensor Component Explorer

## Purpose

The Component Explorer turns the bundled Lens and Sensor databases into an engineering
selection workflow. It is designed for requirement-driven module screening, not for
browsing a catalog in isolation and not for declaring product qualification.

The workflow is:

```text
Mission requirements
  -> Lens/Sensor search
  -> capability and geometry gates
  -> Pareto shortlist
  -> same-scene CameraE2E rerun
  -> versioned baseline + artifacts
  -> decision report
```

## Data Contract

Source databases are read-only. Search responses normalize source records without
inventing unknown specifications. The selected project stores a small module descriptor
containing source IDs, derived geometry, compatibility gates, fidelity/readiness, and
provenance. It does not copy or modify the source DB.

Lens records expose patent/configuration identity, focal length, F-number, image height,
field domain, prescription complexity, and geometric PSF availability. Sensor records
expose source identity, pixel pitch, native or inferred geometry, CFA, modality, structural
features, and whether the current frame-RAW engine can represent the record.

## Compatibility Model

Each candidate is classified as:

- `compatible`: every executable hard gate passed.
- `conditional`: no gate failed, but at least one required value is not evaluable.
- `incompatible`: at least one hard gate failed.

The current gates are:

| Gate | Meaning |
|---|---|
| `frame_sensor_model` | Current engine supports the source sensor modality |
| `cfa_model` | Source CFA maps to an implemented CFA acquisition model |
| `lens_image_circle` | Sensor diagonal fits the lens image circle |
| `lens_field_domain` | Sensor diagonal FOV stays inside the patent/RayOptics analyzed field |
| `hfov` | Derived module HFOV meets the active mission target |
| `object_pixels_at_range` | Reference object has enough vertical samples at range |
| `diffraction_sampling` | Analytic Airy support does not exceed the current sampling limit |
| `pixel_bandwidth` | Native pixel count and frame rate meet the study bandwidth limit |
| `rolling_shutter` | Native rows and study row-time meet the motion timing limit |

Pareto membership uses object pixels, relative pixel-area/F-number support, evidence
completeness, HFOV error, and prescription complexity. It intentionally produces no
single hidden score and no automatic winner.

## Native and Simulation Geometry

`native_rows` and `native_cols` preserve the physical/source geometry and drive module
width, diagonal, bandwidth, and rolling-shutter calculations. `rows` and `cols` are the
research execution readout, CFA-aligned and capped at 640 by 360. This avoids allocating
full 20-100 MP arrays during broad local studies while keeping the approximation visible.
The simulation pitch is expanded to preserve full sensor extent, while simulation fill
factor is reduced so its photosensitive area equals one native pixel. This is a sparse
sampling proxy, not sensor binning; native resolution and CFA phase are not reproduced.

If only megapixels are available, geometry is inferred as 16:9 and tagged
`resolution_mp_16_9_proxy`. Such a record is not source-resolution evidence.

## Real Evaluation Path

`Compare on Scene` accepts two to four candidates. Each candidate is rebuilt and sent
through the actual `Scene -> Optics -> Sensor -> ISP -> Metrics` path using the same
scene and seed. Available selected RayOptics assets are loaded as geometric PSFs. Missing
assets are not rendered as fake results. Preview images, metrics, compatibility, lineage,
and the comparison artifact are persisted and included in the decision report.

## API

```text
GET  /api/v2/catalog/lenses
GET  /api/v2/catalog/lenses/{simulation_id}
GET  /api/v2/catalog/sensors
GET  /api/v2/catalog/sensors/{sensor_id}
POST /api/v2/catalog/modules/evaluate
POST /api/v2/projects/{project_id}/studies/{study_id}/module
POST /api/v2/projects/{project_id}/studies/{study_id}/modules/compare
```

Open `http://127.0.0.1:8010/docs` for the generated request/response schema.

## Fidelity Boundaries

- Event, NIR, and SWIR records need dedicated acquisition models and are blocked from
  the frame-RAW path.
- Unknown CFA is not replaced with Bayer.
- Sensor-specific measured QE, PTC, noise, full well, and timing are not inferred.
- Low-light support is a relative pixel-area/F-number quantity, not measured SNR.
- RayOptics PSFs are geometric ray histograms and do not establish diffraction, flare,
  tolerance, or measured MTF performance.
- Patent- and source-derived records remain `proxy` until calibration evidence promotes
  the specific contributing stage.
