"""Render CameraE2E database catalog HTML and JSON reports."""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import camerae2e_db_catalog, camerae2e_db_summary  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "db_catalog"


def main(output_dir: str | Path | None = None) -> dict[str, Path]:
    out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    entries = [entry.to_dict() for entry in camerae2e_db_catalog()]
    summary = camerae2e_db_summary()
    payload = {"summary": summary, "entries": entries}

    json_path = out / "camerae2e_db_catalog.json"
    html_path = out / "camerae2e_db_catalog.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_html(payload, html_path), encoding="utf-8")
    return {"html": html_path, "json": json_path}


def _render_html(payload: dict[str, Any], html_path: Path) -> str:
    entries = payload["entries"]
    summary = payload["summary"]
    cards = "\n".join(_entry_card(entry) for entry in entries)
    family_rows = "\n".join(
        f"<tr><td>{escape(family)}</td><td>{_status_cells(counts)}</td></tr>"
        for family, counts in sorted(summary.get("families", {}).items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CameraE2E DB Catalog</title>
  <style>
    body {{ margin: 32px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }}
    h1, h2, h3 {{ color: #102a43; }}
    code, pre {{ background: #edf2f7; border-radius: 6px; }}
    code {{ padding: 2px 5px; }}
    pre {{ padding: 14px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px; vertical-align: top; }}
    th {{ background: #eef4f8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 12px; padding: 16px; background: #fbfcfd; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 2px 9px; font-size: 12px; font-weight: 700; }}
    .active {{ background: #d8f3dc; color: #1b4332; }}
    .available {{ background: #dbeafe; color: #1e3a8a; }}
    .fallback {{ background: #fef3c7; color: #92400e; }}
    .missing {{ background: #fee2e2; color: #991b1b; }}
    .hint {{ color: #52606d; }}
    .path {{ word-break: break-all; }}
  </style>
</head>
<body>
  <h1>CameraE2E DB Catalog</h1>
  <p>This catalog indexes the DB/LUT/model-profile assets that CameraE2E can use as block parameters. It separates active external DBs from bundled fallback data and gives direct parameter bundles for runtime APIs.</p>

  <h2>Summary</h2>
  <table>
    <tr><th>Total entries</th><td>{summary.get("total", 0)}</td></tr>
    <tr><th>Active entries</th><td>{escape(", ".join(summary.get("active", [])))}</td></tr>
  </table>
  <table>
    <tr><th>Family</th><th>Status counts</th></tr>
    {family_rows}
  </table>

  <h2>Search And Parameter Use</h2>
  <pre><code>from pyisetcam import camerae2e_db_search, camerae2e_db_parameters

# Find all Lens DB entries.
lens_rows = camerae2e_db_search("lens", include_missing=False)

# Get direct parameters for the active Lens DB.
lens_params = camerae2e_db_parameters("lens_patents_active")
# lens_params["db_path"], lens_params["psf_dir"], lens_params["highres_psf_dir"]

# Use the parameters in CameraE2E APIs.
from pyisetcam import lens_patent_search, lens_patent_raytrace_optics
rows = lens_patent_search(db_path=lens_params["db_path"], require_camerae2e=True, limit=5)
optics = lens_patent_raytrace_optics(rows[0]["simulation_id"], psf_dir=lens_params["highres_psf_dir"])</code></pre>

  <h2>Catalog Entries</h2>
  <div class="grid">
    {cards}
  </div>

  <p class="hint">JSON catalog: <code>{escape((html_path.parent / "camerae2e_db_catalog.json").name)}</code></p>
</body>
</html>
"""


def _entry_card(entry: dict[str, Any]) -> str:
    params = entry.get("parameters", {})
    metadata = entry.get("metadata", {})
    param_rows = "\n".join(
        f"<tr><td>{escape(str(key))}</td><td class='path'><code>{escape(_short_value(value))}</code></td></tr>"
        for key, value in sorted(params.items())
    )
    metadata_text = escape(json.dumps(_compact_metadata(metadata), indent=2, sort_keys=True))
    tags = ", ".join(entry.get("tags", []))
    status = entry.get("status", "missing")
    return f"""<section class="card">
  <h3>{escape(entry["name"])} <span class="status {escape(status)}">{escape(status)}</span></h3>
  <p><b>Family:</b> {escape(entry.get("family", ""))} / <b>Role:</b> {escape(entry.get("role", ""))}</p>
  <p>{escape(entry.get("description", ""))}</p>
  <p class="path"><b>Path:</b> <code>{escape(str(entry.get("path")))}</code></p>
  <p><b>Parameter use:</b> {escape(entry.get("parameter_hint", ""))}</p>
  <p><b>Env vars:</b> {escape(", ".join(entry.get("env_vars", [])) or "-")}</p>
  <p><b>Tags:</b> {escape(tags)}</p>
  <table><tr><th>Parameter</th><th>Value</th></tr>{param_rows}</table>
  <details><summary>Metadata</summary><pre><code>{metadata_text}</code></pre></details>
</section>"""


def _status_cells(counts: dict[str, int]) -> str:
    return " ".join(
        f"<span class='status {escape(status)}'>{escape(status)}: {count}</span>"
        for status, count in sorted(counts.items())
    )


def _short_value(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else str(value)
    return text if len(text) <= 240 else text[:237] + "..."


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if key.endswith("_manifest") and isinstance(value, dict):
            result[key] = {subkey: value.get(subkey) for subkey in ("built_at", "files", "caveats") if subkey in value}
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    result = main()
    print(result["html"])
