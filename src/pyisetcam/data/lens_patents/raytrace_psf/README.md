# Lens Patent Raytrace PSF Grids

This directory contains RayOptics-generated geometric PSF grid samples for
Lens_Patent_DB rows. The files are optional add-ons to
`lens_patent_simulation_v6.sqlite`.

Available presets:

- `debug`: 64 x 64 PSF, 3 fields, 550 nm, 13 x 13 pupil disk samples
- `standard`: 384 x 384 PSF, 5 fields, 450/550/650 nm, 49 x 49 pupil disk samples
- `production`: 512 x 512 PSF, 5 fields, 450/550/650 nm, 65 x 65 pupil disk samples
- `golden`: 768 x 768 PSF, 7 fields, 450/500/550/600/650 nm, 97 x 97 pupil disk samples

Regenerate a small debug batch:

```bash
/Users/seongcheoljeong/RayOptics/rayoptics-env/bin/python3.12 \
  tools/build_lens_patent_raytrace_psf_grid.py \
  --preset debug \
  --company Canon \
  --limit 5 \
  --overwrite
```

Regenerate all currently CameraE2E-ready rows as a production master set:

```bash
/Users/seongcheoljeong/RayOptics/rayoptics-env/bin/python3.12 \
  tools/build_lens_patent_raytrace_psf_grid.py \
  --preset production \
  --limit 10000 \
  --overwrite
```

That production command is substantially slower and larger than the debug
batch. Generate representative samples first before launching the full batch.

Load one grid:

```python
from pyisetcam.lens_patents import lens_patent_raytrace_optics

optics = lens_patent_raytrace_optics("p0014:intermediate")
```

The PSF arrays follow CameraE2E's raytrace optics shape:
`(psf_y, psf_x, field_height, wavelength)`.

Current batch summary:

- total attempted CameraE2E-ready rows: 547
- generated PSF grids: 408
- failed RayOptics traces: 139
- generated grid setting: `debug` preset, 64 x 64 PSF, 3 field heights, 550 nm, 13 x 13 pupil sampling disk

A representative 512 x 512 `production` preset sample is stored in
`../raytrace_psf_highres`.

Current failure summary:

- Canon: 99 generated / 45 failed
- Konica Minolta: 22 generated / 2 failed
- Largan: 0 generated / 67 failed
- Nikon: 76 generated / 3 failed
- Olympus/OM System: 53 generated / 4 failed
- Panasonic: 126 generated / 12 failed
- Samsung: 26 generated / 0 failed
- Seiko Epson: 6 generated / 6 failed

The dominant current failure modes are `TraceTIRError`,
`TraceMissedSurfaceError`, and insufficient traced-ray success fraction. Largan
requires additional prescription/trace interpretation work before its rows
should be considered usable as RayOptics PSF grids.

Important limitation: these PSFs are geometric ray histograms from RayOptics
trace intersections. They do not include diffraction or diffraction-aberration
wavefront propagation. Treat them as a raytrace PSF-grid starting point, not as
a final Zemax-quality physical optics result.
