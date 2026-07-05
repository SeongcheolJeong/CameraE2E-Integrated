# MATLAB To Python Mapping

This milestone keeps MATLAB semantics close to the source while exposing Pythonic snake_case names.

| MATLAB | Python |
| --- | --- |
| `sceneCreate` | `scene_create` |
| `sceneFromFile` | `scene_from_file` |
| `sceneGet` | `scene_get` |
| `sceneSet` | `scene_set` |
| `sceneAdjustIlluminant` | `scene_adjust_illuminant` |
| `displayCreate` | `display_create` |
| `oiCreate` | `oi_create` |
| `oiCompute` | `oi_compute` |
| `sensorCreate` | `sensor_create` |
| `sensorCreateIdeal` | `sensor_create_ideal` |
| `sensorCompute` | `sensor_compute` |
| `ipCreate` | `ip_create` |
| `ipCompute` | `ip_compute` |
| `cameraCreate` | `camera_create` |
| `cameraCompute` | `camera_compute` |
| `metricsSPD` | `metrics_spd` |
| `iePSNR` | `peak_signal_to_noise_ratio` |

## Object Model

MATLAB `struct` objects are represented as dataclasses with mutable `fields` and `data` dictionaries:

- `Scene`
- `OpticalImage`
- `Sensor`
- `ImageProcessor`
- `Display`
- `Camera`

## Intentional Differences

- All compute functions require explicit object arguments.
- Unsupported milestone-one features raise `NotImplementedError`.
- Upstream assets are fetched into `.cache/upstream/isetcam/<sha>/` instead of being vendored into the package.
