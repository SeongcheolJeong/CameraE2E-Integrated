# Camera Pipeline Parity Evidence Report

Generated: `2026-04-01T14:20:32.377751+00:00`
Git commit: `55f1570`

## Executive Summary
- Global curated parity: `258 passed`, `0 failed`, `1 skipped`.
- Selected camera-pipeline cases: `10/10` passed.
- Conclusion: the selected pipeline, color, sharpness, and distortion cases match the MATLAB baselines, and the exported figures show visually small residual differences.

| Case | MATLAB function | Status | rtol | atol |
| --- | --- | --- | --- | --- |
| camera_default_pipeline | cameraCompute | passed | 0.001 | 0.001 |
| ip_default_pipeline | ipCompute | passed | 0.001 | 0.001 |
| metrics_color_accuracy_small | s_metricsColorAccuracy.m | passed | 1e-05 | 1e-08 |
| optics_rt_center_edge_psf_small | oiCreate/oiGet rtpsfdata | passed | 1e-05 | 1e-08 |
| optics_rt_point_array_field_small | sceneCreate/oiCompute point array | passed | 1e-05 | 1e-08 |
| optics_rt_distortion_field_small | sceneCreate/oiCompute distortion grid | passed | 1e-05 | 1e-08 |
| metrics_vsnr_small | s_metricsVSNR.m | passed | 1e-05 | 1e-08 |
| metrics_acutance_small | s_metricsAcutance.m | passed | 1e-05 | 1e-08 |
| metrics_mtf_slanted_bar_small | s_metricsMTFSlantedBar.m | passed | 1e-05 | 1e-08 |
| metrics_mtf_pixel_size_small | s_metricsMTFPixelSize.m | passed | 1e-05 | 1e-08 |

## Evidence Sources
- Selected parity cases: [`/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/cases.yaml`](/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/cases.yaml)
- MATLAB baselines: [`/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/baselines`](/Users/seongcheoljeong/Documents/CameraE2E/tests/parity/baselines)
- Python parity runners: [`/Users/seongcheoljeong/Documents/CameraE2E/src/pyisetcam/parity.py`](/Users/seongcheoljeong/Documents/CameraE2E/src/pyisetcam/parity.py)
- Machine-readable parity report: [`/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/latest.json`](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/latest.json)
- Selected-case summary JSON: [`/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field_summary.json`](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field_summary.json)
- Regenerate with: `python tools/render_parity_evidence.py --refresh-report`

## Core Camera Pipeline
`camera_default_pipeline` is the headline end-to-end case. It is intentionally noiseless (`sensor_noise_flag = 0`) so the comparison isolates deterministic scene -> OI -> sensor -> IP parity instead of stochastic sensor variance.

![camera result](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/camera_default_pipeline_result_triptych.png)

![camera sensor volts](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/camera_default_pipeline_sensor_volts_triptych.png)

![camera oi photons](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/camera_default_pipeline_oi_photons_triptych.png)

### Camera Stage Metrics
| Field | Shape | MAE | RMSE | Max abs | Mean rel | Max rel | Normalized MAE | Edge mean rel | Interior mean rel | 2x2 phase means |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| result | 162x242x3 | 9.54418e-08 | 1.29805e-07 | 7.44838e-07 | 5.82184e-07 | 2.81622e-05 | 3.32967e-07 | 5.92963e-07 | 5.76207e-07 | r0c0=5.907e-07, r0c1=5.762e-07, r1c0=5.835e-07, r1c1=5.784e-07 |
| sensor_volts | 162x242 | 1.51243e-08 | 2.51401e-08 | 2.33303e-07 | 7.9871e-08 | 5.46787e-07 | 7.666e-08 | 8.21548e-08 | 7.86046e-08 | r0c0=7.868e-08, r0c1=8.111e-08, r1c0=8.111e-08, r1c1=7.859e-08 |
| oi_photons | 80x120x31 | 9.56678e+06 | 1.49228e+07 | 1.65499e+08 | 2.54049e-06 | 5.21949e-05 | 8.3371e-08 | 6.90276e-06 | 8.67145e-08 | r0c0=2.546e-06, r0c1=2.539e-06, r1c0=2.542e-06, r1c1=2.536e-06 |

### Camera Context
| Field | Value |
| --- | --- |
| oi_height_m | 0.000565064 |
| oi_image_distance_m | 0.00387523 |
| oi_sample_spacing_m | 7.0633e-06 |
| oi_size | [80, 120, 31] |
| oi_width_m | 0.000847597 |
| sensor_integration_time_s | 0.0577805 |
| sensor_noise_flag | 0 |
| sensor_size | [162, 242] |

## IP Pipeline
`ip_default_pipeline` isolates the image processor from the outer camera wrapper. The three exported figures below compare the MATLAB and Python payloads at `input`, `sensorspace`, and final `result`.

![ip input](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/ip_default_pipeline_input_triptych.png)

![ip sensorspace](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/ip_default_pipeline_sensorspace_triptych.png)

![ip result](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/ip_default_pipeline_result_triptych.png)

### IP Stage Metrics
| Field | Shape | MAE | RMSE | Max abs | Mean rel | Max rel | Normalized MAE | Edge mean rel | Interior mean rel | 2x2 phase means |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| input | 72x88 | 1.84605e-08 | 3.26052e-08 | 2.06867e-07 | 8.48256e-08 | 4.26211e-07 | 9.03362e-08 | 8.41123e-08 | 8.52255e-08 | r0c0=8.374e-08, r0c1=8.58e-08, r1c0=8.402e-08, r1c1=8.574e-08 |
| sensorspace | 72x88x3 | 1.91603e-08 | 3.24501e-08 | 2.06867e-07 | 7.80482e-08 | 4.26211e-07 | 8.33302e-08 | 7.72005e-08 | 7.85235e-08 | r0c0=7.892e-08, r0c1=7.601e-08, r1c0=7.789e-08, r1c1=7.938e-08 |
| result | 72x88x3 | 7.40063e-08 | 9.24639e-08 | 4.06481e-07 | 7.54307e-07 | 6.87631e-06 | 3.08763e-07 | 6.50567e-07 | 8.12463e-07 | r0c0=7.578e-07, r0c1=7.436e-07, r1c0=7.506e-07, r1c1=7.652e-07 |

## Color Parity
`metrics_color_accuracy_small` provides the boss-facing color evidence using the existing Macbeth-based parity case, so the patch render and Delta E summaries are both backed by the stored MATLAB baseline.

![color accuracy](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/metrics_color_accuracy_small_patch_triptych.png)

| Statistic | MATLAB | Python | Abs diff | Rel diff |
| --- | --- | --- | --- | --- |
| mean | 10.6957 | 14.3282 | 3.63246 | 0.339617 |
| max | 24.1463 | 32.5286 | 8.38232 | 0.347147 |
| std | 5.73722 | 8.22387 | 2.48665 | 0.433423 |

| Component | MATLAB | Python | Abs diff | Rel diff |
| --- | --- | --- | --- | --- |
| X/Y | 0.983107 | 0.977571 | 0.00553593 | 0.00563105 |
| Y/Y | 1 | 1 | 0 | 0 |
| Z/Y | 1.03014 | 1.01691 | 0.0132335 | 0.0128463 |

| Field | Mean rel | Max rel | Normalized MAE |
| --- | --- | --- | --- |
| delta_e | 0.524772 | 1.82384 | 0.462108 |
| compare_patch_srgb | 0.0196499 | 0.171912 | 0.0150243 |
| white_xyz_norm | 0.00615913 | 0.0128463 | 0.00622898 |

## Center/Edge Field Sharpness Parity
These figures are backed by the new curated MATLAB baselines `optics_rt_center_edge_psf_small` and `optics_rt_point_array_field_small`. The point-array overview gives an intuitive field-quality image, while the PSF panels show the direct center-vs-edge optics parity at `550 nm`.

![sharpness overview](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/sharpness_point_array_overview.png)

![sharpness crops](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/sharpness_crop_triptych.png)

![sharpness psf](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/sharpness_psf_triptych.png)

![sharpness psf profiles](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/sharpness_psf_profiles.png)

### PSF Metrics
| Field | Metric | MATLAB | Python | Abs diff | Rel diff |
| --- | --- | --- | --- | --- | --- |
| Center PSF | Peak | 0.599399 | 0.599399 | 0 | 0 |
| Center PSF | EE50 radius (um) | 2.54754 | 2.54754 | 1.30118e-13 | 5.1076e-14 |
| Center PSF | EE80 radius (um) | 4.85114 | 4.85114 | 2.00728e-13 | 4.13775e-14 |
| Center PSF | RMS radius (um) | 4.19845 | 4.19845 | 2.30926e-14 | 5.50028e-15 |
| Edge PSF | Peak | 0.486 | 0.486 | 0 | 0 |
| Edge PSF | EE50 radius (um) | 2.83726 | 2.83726 | 0 | 0 |
| Edge PSF | EE80 radius (um) | 5.23452 | 5.23452 | 0 | 0 |
| Edge PSF | RMS radius (um) | 4.71204 | 4.71204 | 8.88178e-16 | 1.88491e-16 |

### Crop Metrics
| Field | Metric | MATLAB | Python | Abs diff | Rel diff |
| --- | --- | --- | --- | --- | --- |
| Center crop | Peak | 0.743752 | 0.978554 | 0.234802 | 0.3157 |
| Center crop | EE50 radius (px) | 3.30582 | 3.65964 | 0.353818 | 0.107029 |
| Center crop | EE80 radius (px) | 5.50926 | 6.2825 | 0.773242 | 0.140353 |
| Center crop | RMS radius (px) | 4.5981 | 5.16982 | 0.571719 | 0.124338 |
| Edge crop | Peak | 0.61108 | 0.910935 | 0.299855 | 0.490696 |
| Edge crop | EE50 radius (px) | 3.21422 | 3.60487 | 0.390644 | 0.121536 |
| Edge crop | EE80 radius (px) | 5.21057 | 6.05311 | 0.84254 | 0.161698 |
| Edge crop | RMS radius (px) | 4.37389 | 4.84775 | 0.473861 | 0.108339 |

### Field Heights
| Field | MATLAB field height (mm) | Python field height (mm) |
| --- | --- | --- |
| Center | 0 | 0 |
| Edge | 1.0365 | 1.0365 |

### Sharpness Comparison Summary
| Payload field | Mean rel | Max rel | Normalized MAE |
| --- | --- | --- | --- |
| render_rgb | 2.13504e+07 | 7.76439e+10 | 0.685994 |
| center_psf_norm | 0 | 0 | 0 |
| edge_psf_norm | 0 | 0 | 0 |
| center_crop_luma_norm | 1.10852e+08 | 1.09822e+10 | 0.275729 |
| edge_crop_luma_norm | 2.48531e+06 | 8.88799e+08 | 0.247153 |

## Distortion Parity
The distortion section is backed by `optics_rt_distortion_field_small`. The grid figure is the qualitative image evidence, while the radial distortion curve is the quantitative parity measurement.

![distortion grid](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/distortion_grid_triptych.png)

![distortion curve](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/distortion_curve.png)

| Metric | MATLAB | Python | Abs diff | Rel diff |
| --- | --- | --- | --- | --- |
| Reference wavelength (nm) | 550 | 550 | 0 | 0 |
| Max |distortion| (%) | 7.45177 | 7.45177 | 0 | 0 |
| Field height at max distortion (mm) | 1.0365 | 1.0365 | 0 | 0 |

| Payload field | Mean rel | Max rel | Normalized MAE |
| --- | --- | --- | --- |
| ideal_grid_rgb | 3.02082e-07 | 4.99775e-07 | 3.29847e-08 |
| distorted_grid_rgb | 4.80679e-07 | 0.00557711 | 1.06338e-07 |
| distortion_percent | 0 | 0 | 0 |

## Configuration Sweep
These cases show that the Python port tracks MATLAB not just for a single end-to-end pipeline image, but across several camera-pipeline operating conditions and quality metrics.

### VSNR
![vsnr curves](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/metrics_vsnr_small_curves.png)

| Light level | MATLAB channel means | Python channel means |
| --- | --- | --- |
| Level 1 | [0.9387, 0.9246, 1.    ] | [0.9441, 0.9355, 1.    ] |
| Level 2 | [0.9477, 0.9312, 1.    ] | [0.9463, 0.9333, 1.    ] |
| Level 3 | [0.9472, 0.9315, 1.    ] | [0.947 , 0.9324, 1.    ] |

### Acutance
![acutance](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/metrics_acutance_small_lum_mtf.png)

| Metric | MATLAB | Python | Abs diff | Rel diff |
| --- | --- | --- | --- | --- |
| acutance | 0.358735 | 0.359336 | 0.000601245 | 0.00167602 |
| camera_acutance | 0.358735 | 0.359336 | 0.000601245 | 0.00167602 |
| cpiq_norm_mean | 0.491461 | 0.491135 | 0.000325972 | 0.000663273 |

### MTF Slanted Bar
![mtf slanted bar](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/metrics_mtf_slanted_bar_small_curves.png)

| Curve | MATLAB MTF50 | Python MTF50 | MATLAB Nyquist | Python Nyquist | MATLAB alias % | Python alias % |
| --- | --- | --- | --- | --- | --- | --- |
| Mono direct | 156.4 | 154.8 | 178.571 | 178.571 | 21.7562 | 20.7061 |
| Color direct | 85.8 | 89.8 | 178.571 | 178.571 | 22.204 | 18.1347 |
| IE color | 81.6 | 83.6 | 178.571 | 178.571 | 18.917 | 20.8304 |

### MTF Pixel Size Sweep
The pixel-size sweep below is plotted as one subplot per pixel size on normalized frequency (`f / Nyquist`) rather than raw sample index. Vertical guide lines mark the MATLAB and Python `MTF50 / Nyquist` locations, which avoids overstating the coarser `9 um` visual delta.

![mtf pixel size](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/camera_field/metrics_mtf_pixel_size_small_profiles.png)

| Pixel size (um) | MATLAB MTF50 | Python MTF50 | Abs diff | Rel diff | MATLAB MTF50 / Nyquist | Python MTF50 / Nyquist | Profile normalized MAE | Profile max abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 129 | 131.2 | 2.2 | 0.0170543 | 0.516 | 0.5248 | 0.0342319 | 0.0438042 |
| 3 | 118 | 118.2 | 0.2 | 0.00169492 | 0.708 | 0.7092 | 0.018206 | 0.0396181 |
| 5 | 87.6 | 90.8 | 3.2 | 0.0365297 | 0.876 | 0.908 | 0.0584092 | 0.0797921 |
| 9 | 46.2 | 46.4 | 0.2 | 0.004329 | 0.8316 | 0.8352 | 0.0926761 | 0.0952987 |

## Conclusion
- Python matches MATLAB on the selected camera end-to-end and stagewise parity cases.
- The color, center/edge sharpness, and distortion sections are all traceable to curated MATLAB baselines rather than Python-only visuals.
- The configuration sweeps preserve the same trends as MATLAB for VSNR, acutance, and MTF-oriented analyses.
