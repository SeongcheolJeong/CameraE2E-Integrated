# Perception Metrics

`pyisetcam.perception` is a high-level evidence layer for human-visible image quality. It does not replace the existing camera, metrics, color, or SCIELAB implementations. Instead, it combines them under explicit viewing assumptions.

## Why This Layer Exists

Numeric image differences are not always human-visible, and human-visible differences depend on viewing conditions. A report that says only `RMSE` or `PSNR` can hide important color, sharpness, artifact, and temporal failure modes.

The perception layer requires a `PerceptionViewingConfig` so results are tied to:

- viewing distance
- display pixel pitch
- display peak and black luminance
- white point
- JND and visible-difference thresholds

## Main Functions

- `perception_config(...)`: create a validated viewing configuration.
- `pixels_per_degree(...)`: convert display geometry into visual sampling.
- `image_to_luminance(...)`: convert grayscale or sRGB data into display luminance.
- `perception_image_metrics(...)`: compute numeric and luminance-domain comparison metrics.
- `perception_color_metrics(...)`: compute Delta E map and summary.
- `perception_visible_difference_map(...)`: compute S-CIELAB or Delta E visible-difference evidence.
- `perception_sharpness_metrics(...)`: compute ISO acutance and SQRI from MTF samples.
- `perception_artifact_metrics(...)`: compute simple artifact visibility proxies.
- `perception_compare(...)`: run the default combined perception metric bundle.
- `perception_report(...)`: compatibility wrapper for the default comparison bundle.

MATLAB-style aliases are also exposed, for example `perceptionConfig`, `pixelsPerDegree`, and `perceptionVisibleDifferenceMap`.

## Example

```python
import numpy as np
from pyisetcam import perception_compare, perception_config

reference = np.zeros((64, 64, 3), dtype=float)
test = reference.copy()
test[24:40, 24:40, 0] = 0.12

config = perception_config(viewing_distance_m=0.5, pixel_pitch_m=0.00025)
summary = perception_compare(reference, test, config)

print(summary["visible_difference"])
```

## HTML Report

Generate the implementation and verification report with:

```bash
python tools/render_perception_report.py
```

Outputs are written to:

- `reports/perception/perception_report.html`
- `reports/perception/perception_summary.json`
- `reports/perception/perception_reference.png`
- `reports/perception/perception_test.png`
- `reports/perception/perception_visible_difference_map.png`
- `reports/perception/perception_sharpness_mtf.png`

## Design Constraint

Do not collapse perception into one global score by default. Color error, visible-difference maps, sharpness, noise/artifact proxies, and temporal artifacts are different failure modes. Keep them separate in reports, then add weighted summaries only when a specific product requirement defines the weights.
