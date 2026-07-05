# HW ISP Simulation

`pyisetcam` includes a hardware ISP simulation layer for system verification. The normal `Scene -> OpticalImage -> Sensor -> ImageProcessor` computation still produces the image, and the HW ISP layer adds deterministic timing metadata plus optional delayed AE/AWB control feedback on top.

This is useful when image parity is not enough. System verification often needs to know when a frame was exposed, when rolling-shutter readout finished, when ISP stages completed, which AE/AWB controls were active, and when the frame became visible to the app.

## Quick Start

```python
from pyisetcam import camera_create, scene_create
from pyisetcam import hw_isp_config_from_profile, hw_isp_profile_names
from pyisetcam.hwisp import hw_isp_config, hw_isp_simulate_sequence, hw_isp_latency_summary

scene = scene_create("uniform ee", 8)
camera = camera_create()

config = hw_isp_config(
    fps=30.0,
    line_time_us=15.2,
    exposure_time_us=8000.0,
    transport={"request_queue_depth": 2, "max_buffers": 3},
)

sequence = hw_isp_simulate_sequence(camera, scene, config, nframes=8)
print(hw_isp_latency_summary(sequence))
```

The computed image values are still produced by `camera_compute`, `sensor_compute`, and `ip_compute`. Timing data is attached under:

```python
frame = sequence.frames[0]
frame.camera.metadata["hw_isp"]
frame.ip.metadata["hw_isp"]
```

## Parameter DB

The simulator can load named HW ISP parameter profiles:

```python
print(hw_isp_profile_names())

config = hw_isp_config_from_profile(
    "rpi_vc4_imx219_public_seed",
    control_path={"apply_to_image": True},
)
```

Built-in profiles live under `src/pyisetcam/data/hwisp/`:

- `generic_1080p_30fps`: synthetic engineering seed profile.
- `rpi_vc4_imx219_public_seed`: public Raspberry Pi/libcamera IMX219 seed profile with a small normalized calibration subset.

These are not sign-off vendor databases. They are seed inputs. Replace timing
fields with BSP, kernel trace, hardware counter, or measured values before using
latency numbers as product evidence.

To load product-specific or collected profiles:

```bash
export PYISETCAM_HWISP_DB=/path/to/hwisp/profiles
```

To collect normalized profiles from local libcamera tuning files:

```bash
python tools/collect_hwisp_parameter_db.py /usr/share/libcamera/ipa/rpi/vc4 --output-dir configs/hwisp/collected
export PYISETCAM_HWISP_DB=configs/hwisp/collected
```

The collector extracts public tuning/calibration summaries such as black level,
lux reference, noise, GEQ, and AGC target seed. Raw libcamera tuning files do not
normally contain complete block latency, queue, or DMA timing, so the collector
fills missing timing from the generic profile and marks the profile confidence as
`collected_tuning_with_seed_timing`.

## Timing Model

The default model records these timestamps in microseconds:

- `request`
- `exposure_start`
- `exposure_mid`
- `readout_start`
- `readout_end`
- `stats_ready`
- `isp_start`
- `isp_done`
- `dma_done`
- `app_visible`

Rolling-shutter row timing follows:

```text
t_row_start(row) = t_readout_start + (hidden_lines_top + row) * line_time_us
```

ISP stages support `stream`, `line`, and `frame` buffering. Streaming and line-buffer stages can start before full-frame readout completes. Frame-buffer stages wait for the previous stage to complete the full frame.

## Latency Factors

Use `global_latency_factor` to scale ISP and transport delays together:

```python
slow = hw_isp_config(global_latency_factor=2.0)
```

Use `HWIspStage.latency_factor` to scale one block:

```python
from pyisetcam.hwisp import HWIspStage, hw_isp_config

config = hw_isp_config(
    stages=[
        HWIspStage("blc", "bayer", "stream", stage_latency_cycles=24),
        HWIspStage("demosaic", "bayer_to_rgb", "line", window_lines=5, stage_latency_cycles=220, latency_factor=2.0),
    ]
)
```

## 3A And Queue Delay

By default, AE/AWB are modeled as delayed control metadata only, so existing image values are unchanged. If `ae_apply_delay_frames=2`, statistics from frame `N` become active on frame `N + 2`. Early frames are marked as warmup.

Enable image-affecting AE/AWB with:

```python
config = hw_isp_config(
    control_path={
        "apply_to_image": True,
        "ae_apply_delay_frames": 2,
        "awb_apply_delay_frames": 2,
        "target_luma": 0.18,
    }
)
```

When enabled, AE applies delayed `exposure_time_us` and `analog_gain` to the sensor. AWB applies delayed gray-world RGB gains through a manual IP illuminant-correction matrix. AF is not part of v1.

The default 3A stats model is H3A-like:

- AE uses an `8 x 8` stats grid with center-weighted metering by default.
- AE records tile luma, weighted luma, clipped-pixel fraction, target error, and EV error.
- Highlight clipping can force the requested exposure/gain product downward even when the full-frame mean is below target.
- AWB uses `ip.data["sensorspace"]` with a valid-luma ROI by default, excluding dark and clipped pixels before gray-world gain estimation.
- Sequence summaries include AE settle frame, AWB settle frame, final EV error, final RGB imbalance, clipping reduction, clamp compliance, and warmup-delay verdicts.

For compatibility/debug, switch AE back to full-frame mean metering:

```python
config = hw_isp_config(control_path={"ae_metering": "mean"})
```

The queue model uses `request_queue_depth` and `max_buffers`. When the pipeline is slower than the frame interval, the simulator records `queue_stall_us`. By default it stalls rather than dropping frames.

## Report

Generate the default timeline report:

```bash
python tools/render_hwisp_timeline_report.py
```

Use a DB profile:

```bash
python tools/render_hwisp_timeline_report.py --profile rpi_vc4_imx219_public_seed
```

Generate the implementation and verification report with architecture diagram:

```bash
python tools/render_hwisp_implementation_report.py --profile rpi_vc4_imx219_public_seed --run-tests
```

Generate a self-contained copy with PNG figures embedded directly in the HTML:

```bash
python tools/render_hwisp_implementation_report.py --profile rpi_vc4_imx219_public_seed --embed-images
```

Generate the detailed technical report:

```bash
python tools/render_hwisp_technical_report.py --profile rpi_vc4_imx219_public_seed
```

Generate the HW ISP parameter DB requirements report:

```bash
python tools/render_hwisp_parameter_requirements_report.py --profile rpi_vc4_imx219_public_seed
```

Outputs are written under `reports/hwisp/`:

- `index.html`
- `hwisp_technical_report.html`
- `hwisp_technical_summary.json`
- `hwisp_parameter_requirements_report.html`
- `hwisp_parameter_requirements_summary.json`
- `implementation_verification_report.html`
- `implementation_verification_report_integrated.html`
- `implementation_verification_summary.json`
- `implementation_verification_integrated_summary.json`
- `frame_details.html`
- `timeline_report.md`
- `timeline_summary.json`
- `three_a_summary.json`
- `frame_timeline.png`
- `stage_latency.png`
- `e2e_latency.png`
- `ae_convergence.png`
- `awb_convergence.png`
- `three_a_thumbnails.png`

Open `reports/hwisp/index.html` for the browser-facing dashboard. Use `frame_details.html` when you need the full per-frame and per-stage timestamp tables. The dashboard includes a deterministic 12-frame Macbeth-style 3A scenario with a brightness step and a warm-illuminant step.

The dashboard also includes a validation verdict table:

- AE settled when absolute EV error stays below `0.25 EV` for two consecutive frames.
- AWB settled when corrected RGB imbalance stays below `0.20` for two consecutive frames.
- Clamp compliance verifies exposure, analog gain, and WB gains remain within configured limits.
- Warmup delay mapping verifies frame-based control delay behavior.
- Clip reduction verifies the highlight patch clip fraction decreases after delayed AE response.

## Relationship To Parity

The HW ISP simulation is not MATLAB parity. It is a new system timing layer for verification. MATLAB parity remains responsible for image behavior, while this layer explains timing, control delay, and pipeline visibility.
