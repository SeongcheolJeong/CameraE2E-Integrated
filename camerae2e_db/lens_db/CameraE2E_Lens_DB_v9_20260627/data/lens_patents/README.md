# CameraE2E Lens Patent DB v9

Generated from `/Users/seongcheoljeong/Lens_Patent_DB` `expanded_v9`.

## Files

- `lens_patent_simulation_v9.sqlite`: master CameraE2E lens catalog.
- `camerae2e_quality`: SQLite table inside the master DB with `use_tier` and PSF status.
- `camerae2e_psf_assets`: SQLite table inside the master DB with debug/highres PSF manifest rows.
- `camerae2e_inventory.csv`: flat inventory for quick filtering.
- `camerae2e_manifest.json`: package-level summary.
- `psf_grid_spec.json`: PSF grid policy and recommended future production settings.
- `companies/`: company-specific SQLite subsets.
- `raytrace_psf/`: debug RayOptics PSF grids, 64 x 64, 3 fields, 550 nm.
- `raytrace_psf_highres/`: production-preset sample PSF grids, 512 x 512, 5 fields, 450/550/650 nm.

## Current Counts

- Total simulation rows: 822
- `highres_psf`: 5
- `raytrace_psf`: 433
- `proxy_only`: 146
- `partial`: 235
- `metadata_only`: 3
- Recommended for direct CameraE2E raytrace PSF use: 438

## Use In CameraE2E

```python
from pathlib import Path
from pyisetcam import lens_patent_raytrace_optics, lens_patent_search

root = Path("/Users/seongcheoljeong/RayOptics/CameraE2E_Lens_DB_v9_20260627/data/lens_patents")
db_path = root / "lens_patent_simulation_v9.sqlite"

rows = lens_patent_search(db_path=db_path, require_camerae2e=True, limit=10)

optics = lens_patent_raytrace_optics(
    "p0014:intermediate",
    psf_dir=root / "raytrace_psf_highres",
    target_psf_size=128,
)
```

For a broad set, query `camerae2e_quality` and prefer `use_tier in ('highres_psf', 'raytrace_psf')`.

```sql
SELECT simulation_id, company, publication_number, use_tier, focal_length_mm, f_number
FROM camerae2e_quality
WHERE recommended_for_camerae2e = 1
ORDER BY company, simulation_id;
```

## High-Resolution Samples

- `p0014:intermediate`: Canon
- `p0320:telephoto`: Panasonic
- `p0410:base`: Largan
- `p0137:middle`: Ricoh/Pentax
- `p0102:base`: Samsung

## Important Caveats

Raytrace PSF assets here are RayOptics geometric ray-histogram PSFs. They include field and wavelength axes where generated, but they do not include diffraction or full wave-optics PSF computation.

Rows marked `caption_proxy` use patent-caption `f`, `Fno`, and `HFOV` values when the normalized variable-distance table was empty. This fixed several Largan rows where the previous paraxial/proxy fallback produced unrealistic f-numbers, but most Largan rows still fail RayOptics PSF generation and remain `proxy_only`.
