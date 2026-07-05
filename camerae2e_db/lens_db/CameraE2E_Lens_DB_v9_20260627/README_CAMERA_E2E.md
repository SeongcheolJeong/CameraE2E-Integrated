# CameraE2E Lens DB Package

This package is the CameraE2E-oriented v9 export of `/Users/seongcheoljeong/Lens_Patent_DB`.

Main entry point:

```text
data/lens_patents/lens_patent_simulation_v9.sqlite
```

Use `data/lens_patents/camerae2e_manifest.json` and the DB table `camerae2e_quality` to choose rows. The safest filter is:

```sql
SELECT *
FROM camerae2e_quality
WHERE recommended_for_camerae2e = 1;
```

Package summary:

- 822 simulation rows
- 438 rows with generated RayOptics PSF assets
- 5 high-resolution sample PSF grids
- 146 proxy-only rows
- 235 partial rows

PSF assets:

- `data/lens_patents/raytrace_psf`: debug grids, 64 x 64, 3 fields, 550 nm.
- `data/lens_patents/raytrace_psf_highres`: production-preset samples, 512 x 512, 5 fields, 450/550/650 nm.

Use from CameraE2E:

```python
from pathlib import Path
from pyisetcam import lens_patent_raytrace_optics

root = Path("/Users/seongcheoljeong/RayOptics/CameraE2E_Lens_DB_v9_20260627/data/lens_patents")
optics = lens_patent_raytrace_optics(
    "p0014:intermediate",
    psf_dir=root / "raytrace_psf_highres",
    target_psf_size=128,
)
```

Caveat: current PSFs are geometric ray histograms, not diffraction/wave-optics PSFs. For final image-quality simulation, use this package as the data foundation and add diffraction/wavefront PSF generation later.
