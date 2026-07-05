# Shift-Variant PSF Edge Pipeline Parity

## Summary
- Case: `pipeline_rt_bar_small`
- Path: `Scene(bar) -> rtGeometry -> rtPrecomputePSF -> rtPrecomputePSFApply -> Sensor(noise off) -> IP`
- This is a shift-variant PSF convolution case, not a diffraction-limited or shift-invariant shortcut.
- Git commit: `5c283cd`

## Geometry
- Scene size: `128x128`
- Scene FOV: `12.0000 deg`
- Lens/OI size: `188x188`
- Sensor size: `72x88`
- ISP result size: `72x88x3`

## Stage Metrics
| Stage | MATLAB edge rc | Python edge rc | edge rc mean rel | crop normalized MAE | profile normalized MAE |
| --- | --- | --- | --- | --- | --- |
| Lens/OI | [93, 94] | [95, 96] | 0.0214 | 0.0187 | 0.1076 |
| Sensor | [37, 45] | [37, 45] | 0.0000 | 0.0115 | 0.1307 |
| ISP | [37, 45] | [37, 45] | 0.0000 | 0.0100 | 0.0758 |

## Evidence Images
![Lens/OI crop](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small/lens_edge_crop_triptych.png)
![Lens/OI profile](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small/lens_edge_profile.png)
![Sensor crop](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small/sensor_edge_crop_triptych.png)
![Sensor profile](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small/sensor_edge_profile.png)
![ISP crop](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small/isp_edge_crop_triptych.png)
![ISP profile](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small/isp_edge_profile.png)

## Interpretation
- `crop normalized MAE` is the main image-level parity metric for the detected central edge ROI.
- `profile normalized MAE` compares the stage-averaged edge profile after per-stage normalization.
- The remaining Lens/OI size and edge-location differences come from ray-trace registration details, but the detected edge crops and downstream Sensor/ISP images remain closely matched.

## Regenerate
- `python tools/render_pipeline_rt_bar_parity.py`
- Summary JSON: [`/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small_summary.json`](/Users/seongcheoljeong/Documents/CameraE2E/reports/parity/pipeline_rt_bar_small_summary.json)
