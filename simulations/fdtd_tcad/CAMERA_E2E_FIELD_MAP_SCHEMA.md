# CameraE2E CRA / Microlens Shift Field Map

This file defines the optional input contract for replacing generated CRA and
microlens-shift priors with camera-module field data.

Default input path:

```bash
image_sensor_db/camera_module_field_map.csv
```

The LUT builder also accepts an explicit path:

```bash
python3 build_camera_e2e_sensor_luts.py --major-only --field-map-csv path/to/camera_module_field_map.csv
```

## Required Columns

| column | meaning |
| --- | --- |
| `slug` | Sensor slug matching `image_sensor_db/generated_stack_configs/<slug>.json`. |
| `code` | Optional TechInsights/report code. |
| `field_case` | Stable field case id, such as `center`, `x_plus_edge`, `diag_plus_plus`. |
| `field_x_norm` | Normalized image field coordinate in x, usually `-1..1`. |
| `field_z_norm` | Normalized image field coordinate in z, usually `-1..1`. |
| `cra_x_deg` | Chief ray angle component in x at the sensor. |
| `cra_z_deg` | Chief ray angle component in z at the sensor. |
| `lens_shift_x_um` | Microlens/OCL center shift from PD center in x. |
| `lens_shift_z_um` | Microlens/OCL center shift from PD center in z. |
| `wavelength_set_nm` | Semicolon-separated wavelengths to sweep, for example `450;550;620`. |
| `measurement_gate` | Data quality gate. Product-usable values: `MEASURED`, `CALIBRATED`, `RAYTRACE_VALIDATED`. |
| `source` | Data provenance, for example camera module raytrace version, lab measurement id, or calibration run id. |

## Optional CRA Mismatch Columns

These columns are optional for backwards compatibility, but they are required
before the field map can become a product CameraE2E LUT input.

| column | meaning |
| --- | --- |
| `lens_cra_x_deg` | Lens raytrace chief-ray x component. If blank, `cra_x_deg` is treated as the lens CRA for diagnostics only. |
| `lens_cra_z_deg` | Lens raytrace chief-ray z component. If blank, `cra_z_deg` is treated as the lens CRA for diagnostics only. |
| `sensor_cra_x_deg` | Sensor/ML/OCL optimized chief-ray acceptance x component. |
| `sensor_cra_z_deg` | Sensor/ML/OCL optimized chief-ray acceptance z component. |
| `cra_mismatch_tolerance_profile` | Optional tolerance label, for example `rgb_small_pixel_or_high_cra`, `pdaf_split_strict`, or `custom_field_map_tolerance`. |
| `cra_mismatch_pass_tolerance_deg` | Optional custom PASS threshold for total lens-vs-sensor CRA mismatch. |
| `cra_mismatch_check_tolerance_deg` | Optional custom CHECK threshold. Above this value the row is FAIL. |

The builder computes:

- `cra_mismatch_x_deg = lens_cra_x_deg - sensor_cra_x_deg`
- `cra_mismatch_z_deg = lens_cra_z_deg - sensor_cra_z_deg`
- `cra_mismatch_total_deg = hypot(dx, dz)`
- `cra_mismatch_gate = PASS / CHECK / FAIL / MISSING`

Default inferred tolerance profiles:

| profile | PASS | CHECK | intended use |
| --- | ---: | ---: | --- |
| `pdaf_split_strict` | <= 2 deg | <= 4 deg | PDAF, split pixel, QPD-sensitive cases. |
| `rgb_small_pixel_or_high_cra` | <= 3 deg | <= 5 deg | RGB small-pixel or high-CRA camera modules. |
| `rgb_mid_cra` | <= 5 deg | <= 7 deg | RGB larger-pixel or mid-CRA modules. |
| `mono_relaxed` | <= 8 deg | <= 12 deg | Monochrome sensors where color shading is not a constraint. |

## Gate Meaning

`MEASURED`, `CALIBRATED`, and `RAYTRACE_VALIDATED` are the only values that make
the measurement provenance pass. Product CRA input additionally requires
finite sensor CRA reference columns and `cra_mismatch_gate=PASS`. Any other
value keeps the generated LUT in research or trend mode.

Passing the CRA input gate is not enough for CameraE2E product use. The Meep
field sweep and crosstalk sweep still need convergence PASS, and stack geometry
plus n,k values still need measured/calibrated data.

## Practical Notes

- CRA is a camera-module property, not a standalone sensor teardown property.
- TechInsights-derived stack metadata can seed geometry, but it should not be
  treated as a measured CRA or microlens-shift map.
- If `lens_shift_x_um` or `lens_shift_z_um` is blank, the builder derives a
  clipped first-order value from `tan(CRA) * focus_depth / n_eff`. This derived
  value should not be used for product LUTs unless separately calibrated.
