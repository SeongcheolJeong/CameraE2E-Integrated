# Lens Patent Simulation DB

Generated from Lens_Patent_DB `expanded_v6` exports.

Files:

- `lens_patent_simulation_v6.sqlite`: CameraE2E-ready prescription and simulation result DB.
- `summary.json`: generation summary and status counts.
- `companies/`: company-specific SQLite DB subsets and manifest.
- `raytrace_psf/`: optional low-resolution RayOptics geometric PSF grid `.npz` files and manifest.
- `raytrace_psf_highres/`: representative 512 x 512 production-preset PSF sample.

Regenerate:

```bash
python tools/build_lens_patent_simulation_db.py
python tools/export_lens_patent_company_sets.py --overwrite
```

Use:

```python
from pyisetcam import oi_create, oi_set
from pyisetcam.lens_patents import lens_patent_search, lens_patent_optics

row = lens_patent_search(company='Canon', require_camerae2e=True, limit=1)[0]
optics = lens_patent_optics(row['simulation_id'])
oi = oi_set(oi_create(), 'optics', optics)
```

Company-specific DB:

```python
from pyisetcam.lens_patents import lens_patent_company_db_path, lens_patent_search

canon_db = lens_patent_company_db_path('Canon')
canon_rows = lens_patent_search(db_path=canon_db, require_camerae2e=True)
```

Raytrace PSF grids:

```bash
/Users/seongcheoljeong/RayOptics/rayoptics-env/bin/python3.12 \
  tools/build_lens_patent_raytrace_psf_grid.py \
  --preset production \
  --limit 10000 \
  --overwrite
```

Generated PSF grids are stored under `raytrace_psf/` as compressed `.npz` files with a `manifest.json`. Load them with `lens_patent_raytrace_optics(simulation_id)`. Pass `target_psf_size` to downsample a high-resolution grid while preserving each PSF slice sum:

```python
from pyisetcam import lens_patent_raytrace_optics

optics = lens_patent_raytrace_optics(
    "p0014:intermediate",
    psf_dir="src/pyisetcam/data/lens_patents/raytrace_psf_highres",
    target_psf_size=128,
)
```

The PSF manifest records both generated and failed rows. The bundled
`raytrace_psf/` batch is currently the `debug` preset; use the `production`
preset for a 512 x 512 master set.

Main API:

- `lens_patent_db_summary()`
- `lens_patent_companies()`
- `lens_patent_company_sets_manifest()`
- `lens_patent_company_db_path(company)`
- `lens_patent_search(company=None, readiness=None, require_camerae2e=False, limit=None)`
- `lens_patent_get(simulation_id)`
- `lens_patent_surfaces(lens_id, configuration=None)`
- `lens_patent_optics(simulation_id, default_f_number=None)`
- `lens_patent_raytrace_psf_search(company=None, status=None)`
- `lens_patent_raytrace_optics(simulation_id, psf_dir=None, target_psf_size=None)`
- `lens_patent_downsample_psf(psf_function, target_psf_size)`

Caveat: these are patent-disclosed examples, not confirmed production lenses. Rows marked `paraxial_proxy` or with `proxy:` sources should be verified before being treated as ray-trace-accurate designs. The PSF grids are RayOptics geometric ray histograms, not diffraction or wave-optics PSFs. This DB is a CameraE2E optics/proxy catalog plus generated raytrace assets, not a full Zemax/OpticStudio parity result set.

Summary: `{"camerae2e_ready_by_company": {"Canon": 144, "Konica Minolta": 24, "Largan": 67, "Nikon": 79, "Olympus/OM System": 57, "Panasonic": 138, "Samsung": 26, "Seiko Epson": 12}, "companies": 14, "lenses": 414, "readiness_counts": {"needs_variable_distances": 47, "ready_configured": 168, "ready_staging": 199}, "simulation_results": 793, "status_counts": {"camerae2e_ready": 547, "metadata_only": 4, "partial": 242}, "surfaces": 20693}`
