# Lens Patent Company DB Sets

Each subdirectory contains a company-specific SQLite subset generated from `lens_patent_simulation_v6.sqlite`.

Regenerate:

```bash
python tools/export_lens_patent_company_sets.py --overwrite
```

Use a company DB directly:

```python
from pyisetcam.lens_patents import lens_patent_search, lens_patent_optics

db_path = 'src/pyisetcam/data/lens_patents/companies/canon/lens_patent_simulation_v6_canon.sqlite'
row = lens_patent_search(db_path=db_path, require_camerae2e=True, limit=1)[0]
optics = lens_patent_optics(row['simulation_id'], db_path=db_path)
```

Summary: `{"camerae2e_ready": 547, "companies": 14, "lenses": 414, "metadata_only": 4, "partial": 242, "simulation_results": 793, "surfaces": 20693}`
