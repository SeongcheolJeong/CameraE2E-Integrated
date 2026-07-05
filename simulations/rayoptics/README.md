# RayOptics Web Workbench

Local web workbench and CameraE2E-oriented lens patent database package built around RayOptics.

## Contents

- `src/`: React/Vite frontend.
- `backend/`: local Python backend for RayOptics model and analysis workflows.
- `CameraE2E_Lens_DB_v9_20260627/`: CameraE2E-ready v9 lens patent DB package, including quality tiers and RayOptics PSF assets.

## CameraE2E DB

Main DB:

```text
CameraE2E_Lens_DB_v9_20260627/data/lens_patents/lens_patent_simulation_v9.sqlite
```

Use `camerae2e_quality` to choose rows:

```sql
SELECT simulation_id, company, publication_number, use_tier, focal_length_mm, f_number
FROM camerae2e_quality
WHERE recommended_for_camerae2e = 1
ORDER BY company, simulation_id;
```

Current counts:

- `highres_psf`: 5
- `raytrace_psf`: 433
- `proxy_only`: 146
- `partial`: 235
- `metadata_only`: 3

Important caveat: PSF assets are RayOptics geometric ray-histogram PSFs. They are not diffraction or full wave-optics PSFs.

## Local Development

```bash
npm install
npm run dev
```

The RayOptics Python environment is intentionally not committed. Recreate/install it locally before running backend optical calculations.
