# Getting Started Tutorial

## Mental Model

`pyisetcam` follows the core ISETCam imaging pipeline with explicit Python objects:

`Scene -> OpticalImage -> Sensor -> ImageProcessor -> Camera`

The main user workflow is:

1. create or load a `Scene`
2. compute an `OpticalImage`
3. compute a `Sensor` response
4. render an `ImageProcessor` result
5. optionally wrap the full sequence with a `Camera`

The main prose below uses snake_case names such as `scene_create` and `camera_compute`. MATLAB-style camelCase aliases still exist for compatibility; see [MATLAB To Python Mapping](./migration.md).

## Workflow 1: Default Camera Pipeline

Goal: run the default end-to-end pipeline with one scene and one camera.

Functions used:

- `AssetStore.default`
- `scene_create`
- `camera_create`
- `camera_compute`

Run:

```bash
python examples/end_to_end.py
```

Expected output:

- `reports/tutorial/end_to_end/end_to_end.png`

What to inspect:

Open the generated PNG and confirm that the default scene renders all the way through the camera pipeline without needing to manage intermediate objects yourself. This is the fastest way to confirm that scenes, optics, sensor, and IP are all working together.

## Workflow 2: Explicit Scene To OI To Sensor To IP Pipeline

Goal: inspect each major stage of the pipeline separately instead of relying on the `Camera` wrapper.

Functions used:

- `scene_create`
- `scene_get`
- `oi_create`
- `oi_compute`
- `oi_show_image`
- `sensor_create`
- `sensor_compute`
- `sensor_get`
- `ip_create`
- `ip_compute`
- `ip_get`

Run:

```bash
python examples/explicit_pipeline.py
```

Expected output:

- `reports/tutorial/explicit_pipeline/scene_rgb.png`
- `reports/tutorial/explicit_pipeline/oi_rgb.png`
- `reports/tutorial/explicit_pipeline/sensor_volts.png`
- `reports/tutorial/explicit_pipeline/ip_srgb.png`

What to inspect:

Compare the four saved images in order. The scene preview shows the input target, the OI image shows the optics stage, the sensor volts view shows the raw measurement domain, and the IP image shows the rendered result. The script also prints wave-count, OI size, sensor size, and final result shape so you can connect the images back to the underlying object state.

## Workflow 3: Build A Scene From RGB Input

Goal: convert a Python RGB array into a calibrated scene and then push it through the camera pipeline.

Functions used:

- `scene_from_file`
- `camera_create`
- `camera_compute`

Run:

```bash
python examples/scene_from_file.py
```

Expected output:

- `reports/tutorial/scene_from_file/input.png`
- `reports/tutorial/scene_from_file/result.png`

What to inspect:

Compare the synthetic RGB input against the rendered output. This workflow is useful when you want to begin from an RGB image instead of a built-in scene constructor, while still using display calibration and the downstream camera pipeline.

## Workflow 4: Measure Output Quality

Goal: compute a small set of output-quality metrics without dropping into the full parity toolchain.

Functions used:

- `camera_create`
- `camera_compute`
- `camera_acutance`
- `comparison_metrics`

Run:

```bash
python examples/quality_metrics.py
```

Expected output:

- `reports/tutorial/quality_metrics/quality_summary.png`
- printed scalar summary including `camera_acutance` and `comparison_metrics`

What to inspect:

Look at the summary figure and the printed metrics together. The example intentionally compares a rendered result against a blurred version of itself so you can see how `comparison_metrics` reports MAE, RMSE, relative error, and PSNR, while `camera_acutance` gives you a camera-centric sharpness scalar.

## How To Navigate The Function Surface

The package root re-exports a very large compatibility surface. For onboarding, treat it as a set of module families rather than a flat list of names.

### `scene`, `optics`, `sensor`, `ip`, `camera`

- `scene`: scene construction and spectral/spatial scene queries
  - start with `scene_create`, `scene_from_file`, `scene_get`, `scene_set`, `scene_adjust_illuminant`
- `optics`: optical-image creation, optics computation, and optics visualization
  - start with `oi_create`, `oi_compute`, `oi_get`, `oi_set`, `oi_show_image`
- `sensor`: sensor models, raw response computation, and sensor-state inspection
  - start with `sensor_create`, `sensor_create_ideal`, `sensor_compute`, `sensor_get`, `sensor_set`
- `ip`: image-processor creation, rendering, and display-facing outputs
  - start with `ip_create`, `ip_compute`, `ip_get`, `ip_set`, `image_show_image`
- `camera`: end-to-end orchestration across the full pipeline
  - start with `camera_create`, `camera_compute`, `camera_get`, `camera_set`, `camera_compute_sequence`

### `display`, `illuminant`, `color`

- `display`: calibrated display models and display-derived rendering helpers
  - start with `display_create`, `display_get`, `display_set`, `display_plot`
- `illuminant`: illuminant creation and illuminant data access
  - start with `illuminant_create`, `illuminant_get`, `illuminant_set`, `illuminant_read`, `daylight`
- `color`: color transforms, luminance conversions, and colorimetry helpers
  - start with `luminance_from_energy`, `luminance_from_photons`, `ie_color_transform`, `srgb_parameters`, `adobergb_parameters`

### `metrics`, `iso`, `scielab`, `perception`, `roi`

- `metrics`: summary metrics and camera-quality workflows
  - start with `comparison_metrics`, `camera_acutance`, `camera_mtf`, `camera_color_accuracy`, `metrics_spd`
- `iso`: ISO 12233 slanted-edge and MTF helpers
  - start with `iso12233`, `iso_find_slanted_bar`, `edge_to_mtf`
- `scielab`: perceptual image-difference workflows
  - start with `scielab`, `scielab_rgb`, `sc_prepare_filters`
- `perception`: viewing-condition aware summaries for human-visible image quality
  - start with `perception_config`, `perception_compare`, `perception_visible_difference_map`, `perception_color_metrics`, `perception_sharpness_metrics`
- `task_perception`: object-detection and segmentation metrics for downstream perception decisions
  - start with `TaskBoundingBox`, `TaskSegmentationMask`, `detection_metrics`, `mean_average_precision`, `segmentation_metrics`, `task_perception_sweep`
- `roi`: region-of-interest geometry and data extraction
  - start with `vc_get_roi_data`, `ie_rect2_locs`, `ie_locs2_rect`

### `fileio`, `assets`, `web`

- `fileio`: direct read/write helpers for images, spectra, DNG, and saved objects
  - start with `vc_read_image`, `vc_read_spectra`, `ie_dng_read`, `vc_export_object`, `vc_import_object`
- `assets`: pinned upstream snapshot management and asset loading
  - start with `AssetStore`, `ensure_upstream_snapshot`, `ie_read_spectra`, `ie_read_color_filter`
- `web`: network-backed data helpers
  - start with `webData`, `webFlickr`, `webPixabay`, `webCreateThumbnails`

### `utils`, `session`, `plotting`

- `utils`: shared numeric and spectral utility helpers
  - start with `blackbody`, `energy_to_quanta`, `quanta_to_energy`, `param_format`, `unit_frequency_list`
- `session`: optional MATLAB-style object/session helpers
  - start with `session_create`, `session_get_objects`, `ie_get_object`, `ie_add_object`, `iset_path`
- `plotting`: plotting wrappers for scenes, OI, sensors, WVF, and metrics
  - start with `scene_plot`, `oi_plot`, `sensor_plot`, `plot_metrics`, `wvf_plot`

## Where To Go Next

- See [MATLAB To Python Mapping](./migration.md) if you are porting existing MATLAB code.
- See [Perception Metrics](./perception.md) when you need viewing-condition aware color, sharpness, and visible-difference evidence.
- See [Task Perception](./task-perception.md) when you need object-detection, segmentation, or task-robustness evidence.
- Use the README for parity-report and migration-audit commands when you move from user workflows into developer verification.
- Explore advanced areas such as human optics, ray-trace optics, SCIELAB workflows, and broader metrics once you are comfortable with the basic explicit-object pipeline.
