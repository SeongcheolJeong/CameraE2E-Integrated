# Lens Patent High-Resolution Raytrace PSF Samples

This directory contains production-preset RayOptics geometric PSF samples for
Lens_Patent_DB rows. It is intentionally separate from `../raytrace_psf`, which
contains the low-resolution debug/validation batch.

Current sample:

- simulation: `p0014:intermediate`
- company: Canon
- preset: `production`
- PSF grid: 512 x 512
- fields: 5 image-height samples
- wavelengths: 450, 550, 650 nm
- pupil sampling: 65 x 65 disk samples

Regenerate this high-resolution sample:

```bash
/Users/seongcheoljeong/RayOptics/rayoptics-env/bin/python3.12 \
  tools/build_lens_patent_raytrace_psf_grid.py \
  --preset production \
  --simulation-id 'p0014:intermediate' \
  --out-dir src/pyisetcam/data/lens_patents/raytrace_psf_highres \
  --overwrite \
  --reset-manifest
```

Load the native high-resolution grid:

```python
from pyisetcam import lens_patent_raytrace_optics

optics = lens_patent_raytrace_optics(
    "p0014:intermediate",
    psf_dir="src/pyisetcam/data/lens_patents/raytrace_psf_highres",
)
```

Load a downsampled grid while preserving each PSF slice sum:

```python
optics_128 = lens_patent_raytrace_optics(
    "p0014:intermediate",
    psf_dir="src/pyisetcam/data/lens_patents/raytrace_psf_highres",
    target_psf_size=128,
)
```

Important limitation: these arrays are still RayOptics geometric ray histograms,
not diffraction or wave-optics PSFs.
