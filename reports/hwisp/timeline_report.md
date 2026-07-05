# HW ISP Timeline Report

## Summary
- Frames: `8`
- Mean E2E latency: `12.984 ms`
- Max E2E latency: `12.984 ms`
- Total queue stall: `83.565 ms`
- HW ISP profile: `rpi_vc4_imx219_public_seed`
- HTML dashboard: [/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/index.html](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/index.html)
- HTML details: [/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/frame_details.html](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/frame_details.html)
- 3A summary JSON: [/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/three_a_summary.json](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/three_a_summary.json)

## 3A Validation
- AE settle frame: `8`
- AE final error: `-0.153 EV`
- AWB settle frame: `10`
- AWB final RGB imbalance: `0.156`
- Max clip fraction: `0.1638`

## Figures
![frame timeline](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/frame_timeline.png)
![stage latency](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/stage_latency.png)
![e2e latency](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/e2e_latency.png)
![ae convergence](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/ae_convergence.png)
![awb convergence](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/awb_convergence.png)
![3a thumbnails](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/three_a_thumbnails.png)

## Regenerate
- `python tools/render_hwisp_timeline_report.py`
- JSON: [/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/timeline_summary.json](/Users/seongcheoljeong/Documents/CameraE2E/reports/hwisp/timeline_summary.json)
