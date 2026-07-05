#!/usr/bin/env python3
"""Build an FDTD-oriented sensor database from local TechInsights exports.

The extractor uses locally licensed TechInsights exports to derive factual
sensor parameters for reference CAD/FDTD/TCAD setup. By default, generated
outputs redact local source paths and evidence text. Use ``--private-output``
only for a local-only database that should not be committed or redistributed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from html import escape, unescape
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
except Exception as exc:  # pragma: no cover - import guard for clean CLI errors
    BeautifulSoup = None  # type: ignore[assignment]
    BS4_IMPORT_ERROR = exc
else:
    BS4_IMPORT_ERROR = None

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover - optional PDF support
    PdfReader = None  # type: ignore[assignment]
    PYPDF_IMPORT_ERROR = exc
else:
    PYPDF_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parent
DEFAULT_TECHINSIGHTS_ROOT = Path.home() / "Sensor_DB_TechInsight"
DEFAULT_OUTPUT_DIR = ROOT / "sensor_db"
BASE_STACK_CONFIG = ROOT / "configs" / "sensor_stack_proxy_1p4um.json"
BASE_TCAD_PROFILE = ROOT / "measured_profiles" / "reference_cmos_ppd_1p4um" / "profile.json"
REDACTED_SOURCE_ROOT = "redacted: local licensed source root"
REDACTED_SOURCE_PATH = "redacted: run with --private-output for local source paths"
REDACTED_EVIDENCE_TEXT = "redacted: run with --private-output for local evidence text"

CODE_RE = re.compile(r"([A-Z]{2,4}-\d{4}-\d{3})")
UM_UNIT = r"(?:um|\u00b5m|\u03bcm)"
NUM_RE = r"(\d+(?:\.\d+)?)"

METADATA_FIELDS = {
    "Product Code": "code",
    "Title": "title",
    "Analysis Type": "report_type",
    "Release Date": "release_date",
    "Analysis End Date": "analysis_end_date",
    "Manufacturer": "manufacturer",
    "Device Name": "device_name",
    "Device Type": "device_type",
    "Authors": "authors",
    "Images Count": "images_count",
    "Documents Count": "documents_count",
    "Report ID": "report_id",
    "Base Asset Group": "base_asset_group",
}

CSV_FIELDS = [
    "code",
    "report_type",
    "release_date",
    "analysis_year",
    "manufacturer",
    "device_name",
    "device_type",
    "title",
    "content_title",
    "sensor_modality",
    "pixel_pitch_um",
    "resolution_mp",
    "resolution_x",
    "resolution_y",
    "optical_format",
    "active_si_thickness_um",
    "optical_stack_height_um",
    "cfa_thickness_um",
    "cfa_thickness_min_um",
    "cfa_thickness_max_um",
    "ocl_pitch_um",
    "color_filter_pitch_um",
    "grid_pitch_um",
    "die_area_mm2",
    "die_size_text",
    "package_dimensions_text",
    "cis_foundry",
    "cis_process_nm",
    "isp_process_nm",
    "pixel_beol_metal_pitch_nm",
    "pixel_sharing",
    "pixel_architecture",
    "transistors_per_pixel",
    "subpixels_per_pixel",
    "illumination",
    "shutter",
    "transfer_gate_type",
    "is_stacked",
    "has_dbi",
    "has_dti",
    "dti_type",
    "dti_depth_um",
    "dti_width_um",
    "dti_width_nm",
    "dti_aspect_ratio",
    "dti_fill_material",
    "dti_liner_material",
    "dti_biasing",
    "cfa_pattern",
    "microlens_type",
    "grid_material",
    "metal_stack",
    "has_hdr",
    "has_lofic",
    "has_pdaf",
    "source_html",
    "source_pdf",
    "stack_config",
    "tcad_profile",
]

PDF_KEYWORDS = (
    "pixel pitch",
    "pixel size",
    "pixel generation",
    "resolution",
    "optical format",
    "active si",
    "silicon thickness",
    "optical stack",
    "deep trench",
    "dti",
    "b-dti",
    "f-dti",
    "cfa",
    "color filter",
    "bayer",
    "microlens",
    "ocl",
    "grid",
    "light shield",
    "transfer gate",
    "vtg",
    "dbi",
    "hybrid bond",
    "pixel sharing",
    "transistor",
    "photodiode",
    "sub-pixel",
    "subpixel",
    "lofic",
    "hdr",
    "conversion gain",
    "metal pitch",
    "design rules",
    "foundry",
    "die size",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relpath_for_output(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path, output_dir).replace(os.sep, "/")


def repo_relative(path: Path) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def truncate_text(value: str, limit: int = 360) -> str:
    value = normalize_space(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def safe_slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def extract_code(path_or_name: str) -> str | None:
    match = CODE_RE.search(path_or_name)
    return match.group(1) if match else None


def float_or_none(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def first_number(value: str) -> float | None:
    match = re.search(NUM_RE, value)
    return float_or_none(match.group(1)) if match else None


def first_um(value: str) -> float | None:
    match = re.search(NUM_RE + r"\s*" + UM_UNIT, value, flags=re.IGNORECASE)
    return float_or_none(match.group(1)) if match else None


def first_nm(value: str) -> float | None:
    match = re.search(NUM_RE + r"\s*nm\b", value, flags=re.IGNORECASE)
    return float_or_none(match.group(1)) if match else None


def all_nm(value: str) -> list[float]:
    numbers = []
    for match in re.finditer(NUM_RE + r"\s*nm\b", value, flags=re.IGNORECASE):
        number = float_or_none(match.group(1))
        if number is not None:
            numbers.append(number)
    return numbers


def html_fragment_text(fragment: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", fragment)
    text = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</tr>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return normalize_space(unescape(text))


def parse_html_fallback(path: Path) -> dict[str, Any]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    h1_match = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    title = html_fragment_text(title_match.group(1)) if title_match else ""
    h1 = html_fragment_text(h1_match.group(1)) if h1_match else ""
    tables: list[dict[str, Any]] = []
    metadata: dict[str, str] = {}
    source_rows: list[dict[str, str]] = []
    for table_index, table_html in enumerate(re.findall(r"(?is)<table[^>]*>(.*?)</table>", html)):
        rows = []
        table_heading = ""
        for tr_html in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", table_html):
            cells = [
                html_fragment_text(cell_match.group(2))
                for cell_match in re.finditer(r"(?is)<(th|td)[^>]*>(.*?)</\1>", tr_html)
            ]
            if len(cells) < 2:
                continue
            label, value = cells[0], cells[1]
            if not label:
                continue
            if not value and not table_heading:
                table_heading = label
            rows.append({"label": label, "value": value})
            source_rows.append(
                {
                    "label": label,
                    "value": value,
                    "table_heading": table_heading,
                    "source_file": str(path),
                    "source_kind": f"html_table_{table_index}",
                }
            )
            if label in METADATA_FIELDS:
                metadata[METADATA_FIELDS[label]] = value
        if rows:
            tables.append({"index": table_index, "rows": rows})
    description = ""
    content_title = ""
    for heading_html, paragraph_html in re.findall(r"(?is)<h3[^>]*>(.*?)</h3>\s*<p[^>]*>(.*?)</p>", html):
        heading = html_fragment_text(heading_html).lower()
        text = html_fragment_text(paragraph_html)
        if heading == "description":
            description = text
        elif heading == "content title":
            content_title = text
    body_text = html_fragment_text(html)
    return {
        "title_tag": title,
        "h1": h1,
        "metadata": metadata,
        "description": description,
        "content_title": content_title,
        "tables": tables,
        "source_rows": source_rows,
        "body_text": body_text,
    }


def nm_to_um(value_nm: float | None) -> float | None:
    if value_nm is None:
        return None
    return round(value_nm / 1000.0, 6)


def all_um(value: str) -> list[float]:
    numbers = []
    for match in re.finditer(NUM_RE + r"\s*" + UM_UNIT, value, flags=re.IGNORECASE):
        number = float_or_none(match.group(1))
        if number is not None:
            numbers.append(number)
    return numbers


def parse_html(path: Path) -> dict[str, Any]:
    if BeautifulSoup is None:
        return parse_html_fallback(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")
    h1 = normalize_space(soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
    tables: list[dict[str, Any]] = []
    metadata: dict[str, str] = {}
    source_rows: list[dict[str, str]] = []

    for table_index, table in enumerate(soup.find_all("table")):
        rows = []
        table_heading = ""
        for tr in table.find_all("tr"):
            cells = [
                normalize_space(cell.get_text(" ", strip=True))
                for cell in tr.find_all(["th", "td"])
            ]
            if len(cells) < 2:
                continue
            label, value = cells[0], cells[1]
            if not label:
                continue
            if not value and not table_heading:
                table_heading = label
            rows.append({"label": label, "value": value})
            source_rows.append(
                {
                    "label": label,
                    "value": value,
                    "table_heading": table_heading,
                    "source_file": str(path),
                    "source_kind": f"html_table_{table_index}",
                }
            )
            if label in METADATA_FIELDS:
                metadata[METADATA_FIELDS[label]] = value
        if rows:
            tables.append({"index": table_index, "rows": rows})

    description = ""
    content_title = ""
    for h3 in soup.find_all("h3"):
        heading = normalize_space(h3.get_text(" ", strip=True)).lower()
        next_p = h3.find_next_sibling("p")
        if next_p is None:
            continue
        text = normalize_space(next_p.get_text(" ", strip=True))
        if heading == "description":
            description = text
        elif heading == "content title":
            content_title = text

    body_text = normalize_space(soup.get_text(" ", strip=True))
    return {
        "title_tag": title,
        "h1": h1,
        "metadata": metadata,
        "description": description,
        "content_title": content_title,
        "tables": tables,
        "source_rows": source_rows,
        "body_text": body_text,
    }


def iter_export_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("techinsights_html_download*") if path.is_dir())


def iter_pdf_files(root: Path) -> list[Path]:
    return sorted(root.glob("techinsights_html_download*/html/pdfs/**/*.pdf"))


def new_record(code: str) -> dict[str, Any]:
    return {
        "code": code,
        "source_files": {"search_results": [], "details": [], "content": [], "html": [], "pdf": []},
        "source_dirs": [],
        "metadata": {"code": code},
        "content_title": "",
        "description_excerpt": "",
        "table_evidence": [],
        "derived_specs": {},
        "derived_evidence": {},
        "generated_files": {},
        "_text_sources": [],
        "_rows": [],
    }


def get_record(records: dict[str, dict[str, Any]], code: str) -> dict[str, Any]:
    if code not in records:
        records[code] = new_record(code)
    return records[code]


def merge_scalar(metadata: dict[str, Any], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    if key not in metadata or metadata[key] in {"", None}:
        metadata[key] = value


def merge_search_doc(record: dict[str, Any], doc: dict[str, Any]) -> None:
    mapping = {
        "code": "code",
        "reportType": "report_type",
        "analysisYear": "analysis_year",
        "releaseDate": "release_date",
        "publishDate": "publish_date",
        "deviceManufacturer": "manufacturer",
        "deviceName": "device_name",
        "deviceType": "device_type",
        "name": "title",
        "reportHandle": "report_handle",
        "authorsC": "authors",
        "imagesCount": "images_count",
        "documentsCount": "documents_count",
        "coverageAreaTagName": "coverage_area",
        "categoryTagName": "category_tags",
        "technologyTagName": "technology_tags",
        "purposeTagName": "purpose_tags",
        "workCodeCategory": "work_code_category",
        "primaryItemCodeCommonName": "primary_item",
    }
    for source_key, target_key in mapping.items():
        merge_scalar(record["metadata"], target_key, doc.get(source_key))
    if doc.get("description"):
        record["_text_sources"].append(
            {
                "kind": "search_description",
                "source_file": record["source_files"]["search_results"][-1]
                if record["source_files"]["search_results"]
                else "",
                "text": normalize_space(doc.get("description")),
            }
        )


def merge_detail(record: dict[str, Any], detail: dict[str, Any], source_file: Path) -> None:
    report = detail.get("report", {}) if isinstance(detail, dict) else {}
    mapping = {
        "code": "code",
        "type": "report_type",
        "analysisYear": "analysis_year",
        "releaseDate": "release_date",
        "deviceManufacturer": "manufacturer",
        "deviceName": "device_name",
        "devicePartNumber": "device_part_number",
        "deviceType": "device_type",
        "name": "title",
        "id": "report_id",
        "imagesCount": "images_count",
        "documentsCount": "documents_count",
        "baseAssetGroup": "base_asset_group",
        "subscriptionName": "subscription",
        "analysisState": "analysis_state",
    }
    for source_key, target_key in mapping.items():
        merge_scalar(record["metadata"], target_key, report.get(source_key))
    if report.get("description"):
        record["_text_sources"].append(
            {
                "kind": "detail_description",
                "source_file": str(source_file),
                "text": normalize_space(report.get("description")),
            }
        )
    authors = detail.get("authors", [])
    if authors and not record["metadata"].get("authors"):
        names = [normalize_space(author.get("name")) for author in authors if isinstance(author, dict)]
        merge_scalar(record["metadata"], "authors", ", ".join(name for name in names if name))


def merge_content(record: dict[str, Any], content: dict[str, Any], source_file: Path) -> None:
    title = normalize_space(content.get("title"))
    if title:
        record["content_title"] = record["content_title"] or title
        record["_text_sources"].append(
            {"kind": "content_title", "source_file": str(source_file), "text": title}
        )
    inner = content.get("content", {})
    if isinstance(inner, dict):
        inner_title = normalize_space(inner.get("title"))
        sub_title = normalize_space(inner.get("subTitle"))
        for kind, text in (("content_inner_title", inner_title), ("content_subtitle", sub_title)):
            if text:
                record["_text_sources"].append({"kind": kind, "source_file": str(source_file), "text": text})


def merge_html(record: dict[str, Any], parsed: dict[str, Any], source_file: Path) -> None:
    for key, value in parsed["metadata"].items():
        merge_scalar(record["metadata"], key, value)
    for kind in ("h1", "title_tag", "content_title", "description"):
        text = normalize_space(parsed.get(kind))
        if text:
            record["_text_sources"].append({"kind": f"html_{kind}", "source_file": str(source_file), "text": text})
    if parsed.get("description") and not record["description_excerpt"]:
        record["description_excerpt"] = truncate_text(parsed["description"])
    if parsed.get("content_title") and not record["content_title"]:
        record["content_title"] = parsed["content_title"]
    record["_rows"].extend(parsed["source_rows"])
    for snippet in pdf_snippets_from_text(parsed.get("body_text", ""), limit=120):
        record["_text_sources"].append(
            {
                "kind": "html_design_snippet",
                "source_file": str(source_file),
                "text": snippet,
            }
        )


def record_search_text(record: dict[str, Any]) -> str:
    parts = []
    for key in ("title", "report_handle", "device_name", "device_type", "manufacturer", "category_tags"):
        parts.append(normalize_space(record.get("metadata", {}).get(key)))
    parts.append(normalize_space(record.get("content_title")))
    parts.append(normalize_space(record.get("description_excerpt")))
    parts.extend(normalize_space(source.get("text")) for source in record.get("_text_sources", []))
    return " ".join(part for part in parts if part)


def should_parse_pdf_for_record(record: dict[str, Any]) -> bool:
    report_type = normalize_space(record.get("metadata", {}).get("report_type", "")).upper()
    if report_type and report_type not in {"DEF", "DEP", "EXR", "FCT"}:
        return False
    text = record_search_text(record).lower()
    return bool(
        "image sensor" in text
        or "cmos image sensor" in text
        or re.search(r"\bcis\b", text)
        or "back-illuminated" in text
        or "back illuminated" in text
    )


def is_image_sensor_record(record: dict[str, Any]) -> bool:
    report_type = normalize_space(record.get("metadata", {}).get("report_type", "")).upper()
    if report_type not in {"DEF", "DEP", "EXR"}:
        return False
    device_type = normalize_space(record.get("metadata", {}).get("device_type", "")).lower()
    text = record_search_text(record).lower()
    if "camera module" in device_type:
        return False
    reject_terms = (
        "time-of-flight",
        "tof sensor",
        "lidar",
        "spad",
        "single-photon avalanche",
        "fingerprint",
        "thermal",
        "microbolometer",
        "ambient light",
        "proximity sensor",
    )
    if any(term in text for term in reject_terms):
        return False
    return bool(
        device_type == "cmos image sensor"
        or "cmos image sensor" in text
        or re.search(r"\bcis\b", text)
        or "image sensor" in text
    )


def pdf_snippets_from_text(text: str, limit: int = 90) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    chunks = re.split(r"[\r\n]+|(?<=\.)\s+(?=[A-Z0-9])", text)
    for chunk in chunks:
        snippet = truncate_text(chunk, 420)
        if len(snippet) < 24:
            continue
        lower = snippet.lower()
        if not any(keyword in lower for keyword in PDF_KEYWORDS):
            continue
        key = lower[:220]
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def extract_pdf_snippets(path: Path, max_pages: int = 0) -> tuple[list[str], int]:
    if PdfReader is None:
        raise RuntimeError(f"pypdf is required for PDF extraction: {PYPDF_IMPORT_ERROR}")
    reader = PdfReader(str(path))
    pages_total = len(reader.pages)
    pages_to_read = pages_total if max_pages <= 0 else min(max_pages, pages_total)
    texts: list[str] = []
    for index in range(pages_to_read):
        try:
            texts.append(reader.pages[index].extract_text() or "")
        except Exception:
            continue
    return pdf_snippets_from_text("\n".join(texts)), pages_to_read


def attach_pdf_sources(
    records: dict[str, dict[str, Any]],
    techinsights_root: Path,
    source_summary: dict[str, Any],
    pdf_max_pages: int = 0,
) -> None:
    source_summary.setdefault("pdf_files", 0)
    source_summary.setdefault("pdf_files_parsed", 0)
    source_summary.setdefault("pdf_files_skipped", 0)
    source_summary.setdefault("pdf_pages_read", 0)
    for pdf_path in iter_pdf_files(techinsights_root):
        code = extract_code(pdf_path.name) or extract_code(str(pdf_path.parent))
        if not code:
            continue
        record = get_record(records, code)
        record["source_files"]["pdf"].append(str(pdf_path))
        record["source_dirs"].append(str(pdf_path.parents[3]) if len(pdf_path.parents) > 3 else str(pdf_path.parent))
        source_summary["pdf_files"] += 1
        if not should_parse_pdf_for_record(record):
            source_summary["pdf_files_skipped"] += 1
            continue
        if PdfReader is None:
            source_summary["pdf_files_skipped"] += 1
            source_summary["pdf_parse_unavailable"] = str(PYPDF_IMPORT_ERROR)
            continue
        try:
            snippets, pages_read = extract_pdf_snippets(pdf_path, max_pages=pdf_max_pages)
            source_summary["pdf_files_parsed"] += 1
            source_summary["pdf_pages_read"] += pages_read
            for snippet in snippets:
                record["_text_sources"].append(
                    {
                        "kind": "pdf_design_snippet",
                        "source_file": str(pdf_path),
                        "text": snippet,
                    }
                )
        except Exception as exc:
            source_summary.setdefault("parse_failures", []).append({"file": str(pdf_path), "error": str(exc)})


def collect_records(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    source_summary = {
        "export_dirs": [],
        "search_documents": 0,
        "detail_files": 0,
        "content_files": 0,
        "html_files": 0,
        "pdf_files": 0,
        "pdf_files_parsed": 0,
        "pdf_files_skipped": 0,
        "pdf_pages_read": 0,
        "parse_failures": [],
    }

    for export_dir in iter_export_dirs(root):
        data_dir = export_dir / "data"
        html_dir = export_dir / "html" / "reports"
        source_summary["export_dirs"].append(str(export_dir))
        search_path = data_dir / "search_results.json"
        if search_path.exists():
            try:
                payload = read_json(search_path)
                docs = payload.get("documents", []) if isinstance(payload, dict) else payload
                if isinstance(docs, list):
                    for doc in docs:
                        if not isinstance(doc, dict) or not doc.get("code"):
                            continue
                        code = normalize_space(doc["code"])
                        record = get_record(records, code)
                        record["source_files"]["search_results"].append(str(search_path))
                        record["source_dirs"].append(str(export_dir))
                        merge_search_doc(record, doc)
                        source_summary["search_documents"] += 1
            except Exception as exc:
                source_summary["parse_failures"].append({"file": str(search_path), "error": str(exc)})

        for detail_path in sorted((data_dir / "details").glob("*.json")):
            code = extract_code(detail_path.name)
            if not code:
                continue
            try:
                detail = read_json(detail_path)
                record = get_record(records, code)
                record["source_files"]["details"].append(str(detail_path))
                record["source_dirs"].append(str(export_dir))
                merge_detail(record, detail, detail_path)
                source_summary["detail_files"] += 1
            except Exception as exc:
                source_summary["parse_failures"].append({"file": str(detail_path), "error": str(exc)})

        for content_path in sorted((data_dir / "content").glob("*.json")):
            code = extract_code(content_path.name)
            if not code:
                continue
            try:
                content = read_json(content_path)
                record = get_record(records, code)
                record["source_files"]["content"].append(str(content_path))
                record["source_dirs"].append(str(export_dir))
                if isinstance(content, dict):
                    merge_content(record, content, content_path)
                source_summary["content_files"] += 1
            except Exception as exc:
                source_summary["parse_failures"].append({"file": str(content_path), "error": str(exc)})

        for html_path in sorted(html_dir.glob("*.html")):
            code = extract_code(html_path.name)
            if not code:
                continue
            try:
                parsed = parse_html(html_path)
                record = get_record(records, code)
                record["source_files"]["html"].append(str(html_path))
                record["source_dirs"].append(str(export_dir))
                merge_html(record, parsed, html_path)
                source_summary["html_files"] += 1
            except Exception as exc:
                source_summary["parse_failures"].append({"file": str(html_path), "error": str(exc)})

    for record in records.values():
        record["source_dirs"] = sorted(set(record["source_dirs"]))
        for key, values in record["source_files"].items():
            record["source_files"][key] = sorted(set(values))
    return records, source_summary


def add_spec(record: dict[str, Any], key: str, value: Any, evidence: dict[str, str]) -> None:
    if value in ("", None):
        return
    if isinstance(value, float):
        value = round(value, 6)
    if key not in record["derived_specs"] or record["derived_specs"][key] in ("", None):
        record["derived_specs"][key] = value
        record["derived_evidence"][key] = [evidence]


def all_text_sources(record: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for key in ("title", "device_name", "device_type", "manufacturer"):
        value = normalize_space(record["metadata"].get(key))
        if value:
            sources.append({"kind": f"metadata_{key}", "source_file": "", "text": value})
    sources.extend(record["_text_sources"])
    for row in record["_rows"]:
        row_label = row["label"]
        if row.get("table_heading") and row.get("table_heading") != row_label:
            row_label = f"{row['table_heading']} / {row_label}"
        sources.append(
            {
                "kind": f"table:{row_label}",
                "source_file": row["source_file"],
                "text": f"{row['label']}: {row['value']}",
            }
        )
    return sources


def find_pixel_pitch(source: dict[str, str]) -> float | None:
    text = source["text"]
    label = source["kind"].lower()
    if "optical format" in label and "pixel pitch" in label:
        values = [float_or_none(match.group(1)) for match in re.finditer(NUM_RE + r"\s*" + UM_UNIT, text, re.I)]
        values = [value for value in values if value is not None and 0.1 <= value <= 20.0]
        return values[-1] if values else None
    patterns = [
        NUM_RE + r"\s*" + UM_UNIT + r"\s*(?:pixel\s*)?(?:pitch|size|generation|pixels?\b)",
        r"(?:pixel\s*(?:pitch|size|generation)|pixel\s*resolution).*?" + NUM_RE + r"\s*" + UM_UNIT,
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = float_or_none(match.group(1))
            if value is not None and 0.1 <= value <= 20.0:
                return value
    title_like = "title" in label
    has_resolution_context = bool(re.search(r"\b(?:mp|megapixel)\b", text, flags=re.I))
    if title_like and has_resolution_context:
        for match in re.finditer(r"(?:^|,)\s*" + NUM_RE + r"\s*" + UM_UNIT + r"\s*(?:,|$)", text, re.I):
            value = float_or_none(match.group(1))
            if value is not None and 0.1 <= value <= 20.0:
                return value
    return None


def find_resolution_mp(source: dict[str, str]) -> float | None:
    text = source["text"]
    label = source["kind"].lower()
    if "metadata_title" not in label and "description" not in label and "resolution" not in label and "optical format" not in label:
        return None
    match = re.search(NUM_RE + r"\s*(?:mp|megapixel)\b", text, flags=re.I)
    return float_or_none(match.group(1)) if match else None


def find_resolution_xy(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{2,5})\s*[x\u00d7]\s*(\d{2,5})", text, flags=re.I)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def find_optical_format(text: str) -> str:
    match = re.search(r"\b(\d+\s*/\s*\d+(?:\.\d+)?)\s*[\"”]", text)
    if match:
        return match.group(1).replace(" ", "") + '"'
    if re.search(r"\b1\s*[\"”]\s*(?:format)?", text, flags=re.I):
        return '1"'
    if re.search(r"\bAPS-C\b|Advanced Photo System Type-C", text, flags=re.I):
        return "APS-C"
    return ""


def value_after_keywords_um(text: str, keyword_pattern: str) -> float | None:
    match = re.search(keyword_pattern + r"[^.\n;:]{0,120}?" + NUM_RE + r"\s*" + UM_UNIT, text, flags=re.I)
    if match:
        return float_or_none(match.group(match.lastindex or 1))
    match = re.search(NUM_RE + r"\s*" + UM_UNIT + r"[^.\n;:]{0,80}?" + keyword_pattern, text, flags=re.I)
    if match:
        return float_or_none(match.group(1))
    return None


def value_after_keywords_nm(text: str, keyword_pattern: str) -> float | None:
    match = re.search(keyword_pattern + r"[^.\n;:]{0,120}?" + NUM_RE + r"\s*nm\b", text, flags=re.I)
    if match:
        return float_or_none(match.group(match.lastindex or 1))
    match = re.search(NUM_RE + r"\s*nm\b[^.\n;:]{0,80}?" + keyword_pattern, text, flags=re.I)
    if match:
        return float_or_none(match.group(1))
    return None


def aspect_ratio_value(text: str) -> float | None:
    match = re.search(r"aspect\s+ratio(?:\s+of)?\s+(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)", text, flags=re.I)
    return float_or_none(match.group(1)) if match else None


def active_si_thickness_value(text: str) -> float | None:
    patterns = [
        r"(?:back[- ]illuminated\s+)?(?:cis\s+)?active\s+si\s+thickness[^.\n;:]{0,60}?" + NUM_RE + r"\s*" + UM_UNIT,
        r"(?:cis\s+)?active\s+si\s+(?:measures|is|of|has|with)[^.\n;:]{0,60}?" + NUM_RE + r"\s*" + UM_UNIT,
        NUM_RE + r"\s*" + UM_UNIT + r"(?:[- ]?thick| thick)?\s+(?:cis\s+)?active\s+si\b",
        NUM_RE + r"\s*" + UM_UNIT + r"\s+(?:cis\s+)?active\s+si\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return float_or_none(match.group(match.lastindex or 1))
    return None


def dti_depth_value(text: str) -> float | None:
    dti = r"(?:[fb]-\s*dti|[fb]-dti|dti|deep\s+trench\s+isolation|deep\s+trench)"
    patterns = [
        dti + r"[^;\n:]{0,80}?(?:depth|thickness|thick)[^;\n:]{0,50}?" + NUM_RE + r"\s*" + UM_UNIT,
        dti + r"[^;\n:]{0,80}?(?:is|measures)[^;\n:]{0,20}?" + NUM_RE + r"\s*" + UM_UNIT + r"\s*(?:deep|thick)",
        NUM_RE + r"\s*" + UM_UNIT + r"\s*(?:deep|thick)\s+(?:partial[- ]depth\s+|full[- ]depth\s+)?(?:[fb]-\s*dti|[fb]-dti|dti)",
        NUM_RE + r"\s*" + UM_UNIT + r"\s+deep\s+partial[- ]depth\s+b[- ]dti",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = float_or_none(match.group(match.lastindex or 1))
            if value is not None and 0.1 <= value <= 20.0:
                return value
    return None


def dti_width_nm_value(text: str) -> float | None:
    dti = r"(?:[fb]-\s*dti|[fb]-dti|dti|deep\s+trench\s+isolation|deep\s+trench)"
    patterns = [
        dti + r"[^;\n:]{0,100}?(?:average\s+)?width(?:\s+of|\s+is|\s+measures)?[^;\n:]{0,30}?" + NUM_RE + r"\s*nm\b",
        dti + r"[^;\n:]{0,80}?trench\s+is\s+" + NUM_RE + r"\s*nm\s+wide\b",
        dti + r"[^;\n:]{0,80}?" + NUM_RE + r"\s*nm\s+wide\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = float_or_none(match.group(match.lastindex or 1))
            if value is not None and 10.0 <= value <= 1000.0:
                return value
    return None


def parse_die_size(value: str) -> tuple[str, float | None]:
    text = normalize_space(value)
    area = None
    area_match = re.search(NUM_RE + r"\s*mm\s*(?:\^?\s*2|\u00b2)", text, flags=re.I)
    if area_match:
        area = float_or_none(area_match.group(1))
    return text, area


def extract_range_to_um(text: str) -> tuple[float | None, float | None]:
    um_values = all_um(text)
    if len(um_values) >= 2:
        return min(um_values), max(um_values)
    nm_values = all_nm(text)
    if len(nm_values) >= 2:
        return nm_to_um(min(nm_values)), nm_to_um(max(nm_values))
    return None, None


def source_contains(source: dict[str, str], *needles: str) -> bool:
    text = source["text"].lower()
    return any(needle.lower() in text for needle in needles)


def classify_sensor_modality(text: str) -> str:
    lower = text.lower()
    if "swir" in lower or "short-wave infrared" in lower:
        return "swir_image_sensor"
    if "event-based" in lower or "event based" in lower or "event vision" in lower:
        return "event_image_sensor"
    if "global shutter" in lower:
        return "global_shutter_cis"
    if "nir" in lower or "near infrared" in lower or "near-infrared" in lower:
        return "nir_cis"
    if "cmos image sensor" in lower or re.search(r"\bcis\b", lower):
        return "cmos_image_sensor"
    if "image sensor" in lower:
        return "image_sensor"
    return ""


def classify_dti(text: str) -> dict[str, Any]:
    lower = text.lower()
    dti_type = ""
    if re.search(r"full[- ]depth[^.]{0,60}(?:front )?(?:deep trench isolation|f[- ]dti|dti)", lower) or "f-dti" in lower:
        dti_type = "full_depth_front_dti"
    elif re.search(r"partial[^.]{0,50}(?:back )?(?:deep trench isolation|b[- ]dti|dti)", lower):
        dti_type = "partial_depth_back_dti"
    elif "b-dti" in lower or "back deep trench" in lower:
        dti_type = "back_dti"
    elif "front deep trench" in lower:
        dti_type = "front_dti"
    elif "dti" in lower or "deep trench" in lower:
        dti_type = "dti_present"

    fill = ""
    if "polysilicon-filled" in lower or "poly-filled" in lower or "polysilicon filled" in lower:
        fill = "polysilicon"
    elif "oxide-filled" in lower or "oxide filled" in lower:
        fill = "oxide"
    elif "metal-filled" in lower or "metal filled" in lower:
        fill = "metal"
    elif "new material" in lower and ("dti" in lower or "deep trench" in lower):
        fill = "new_material_unspecified"

    liner = ""
    if "tin liner" in lower or "ti n liner" in lower:
        liner = "tin"
    elif "dielectric-lined" in lower or "dielectric lined" in lower:
        liner = "dielectric"
    elif "oxide liner" in lower:
        liner = "oxide"

    biasing = ""
    if "biased" in lower and ("dti" in lower or "deep trench" in lower):
        biasing = "biased"

    return {
        "dti_type": dti_type,
        "dti_fill_material": fill,
        "dti_liner_material": liner,
        "dti_biasing": biasing,
    }


def classify_pixel_architecture(text: str) -> dict[str, Any]:
    lower = text.lower()
    architecture = ""
    if "super qpd" in lower:
        architecture = "super_qpd"
    elif "quad" in lower and ("pixel" in lower or "pd" in lower):
        architecture = "quad_pixel"
    elif "dual pixel" in lower:
        architecture = "dual_pixel"
    elif "eight-shared" in lower or "8-shared" in lower:
        architecture = "eight_shared"
    elif "four-shared" in lower or "4-shared" in lower or "4×2" in lower or "4x2" in lower:
        architecture = "four_shared"
    elif "two-shared" in lower or "2-shared" in lower:
        architecture = "two_shared"
    elif "standalone" in lower or "stand-alone" in lower or "non-shared" in lower:
        architecture = "standalone"

    transfer_gate = ""
    if "dual vertical transfer gate" in lower or "d-vtg" in lower:
        transfer_gate = "dual_vertical_transfer_gate"
    elif "vertical transfer gate" in lower or re.search(r"\bvtg\b", lower):
        transfer_gate = "vertical_transfer_gate"
    elif "planar transfer gate" in lower:
        transfer_gate = "planar_transfer_gate"

    transistors = None
    match = re.search(NUM_RE + r"\s*t\s+effective\s+per\s+pixel", text, flags=re.I)
    if match:
        transistors = float_or_none(match.group(1))
    elif "transistor" in lower:
        match = re.search(r"\b(\d{1,2})\s+transistors?\b", text, flags=re.I)
        if match:
            transistors = float_or_none(match.group(1))

    subpixels = None
    match = re.search(r"\b(?:four|4)\s+sub[- ]?pixels?", text, flags=re.I)
    if match:
        subpixels = 4
    elif re.search(r"\b(?:two|2)\s+sub[- ]?pixels?", text, flags=re.I):
        subpixels = 2

    return {
        "pixel_architecture": architecture,
        "transfer_gate_type": transfer_gate,
        "transistors_per_pixel": transistors,
        "subpixels_per_pixel": subpixels,
    }


def classify_optical_features(text: str) -> dict[str, Any]:
    lower = text.lower()
    cfa_pattern = ""
    if "tetracell" in lower:
        cfa_pattern = "tetracell_bayer"
    elif "quad bayer" in lower:
        cfa_pattern = "quad_bayer"
    elif "rggb" in lower:
        cfa_pattern = "rggb_bayer"
    elif "bayer" in lower:
        cfa_pattern = "bayer"
    elif "monochrome" in lower:
        cfa_pattern = "monochrome"
    microlens = ""
    if "high-p" in lower or "high-precision microlens" in lower:
        microlens = "high_precision"
    elif "gapless" in lower and ("microlens" in lower or "ocl" in lower):
        microlens = "gapless_ocl"
    elif "on-chip microlens" in lower or "ocl" in lower:
        microlens = "ocl"
    grid_material = ""
    if "w light shield" in lower or "tungsten" in lower:
        grid_material = "w"
    elif "tin" in lower and "grid" in lower:
        grid_material = "tin"
    elif "low refractive index" in lower or "lri" in lower:
        grid_material = "lri"
    return {
        "cfa_pattern": cfa_pattern,
        "microlens_type": microlens,
        "grid_material": grid_material,
        "has_pdaf": bool("pdaf" in lower or "phase detection" in lower or "super qpd" in lower),
    }


def classify_features(text: str) -> dict[str, Any]:
    lower = text.lower()
    illumination = ""
    if "front-illuminated" in lower or "front illuminated" in lower:
        illumination = "front_illuminated"
    elif "back-illuminated" in lower or "back illuminated" in lower or re.search(r"\bbi\b", lower):
        illumination = "back_illuminated"
    shutter = ""
    if "global shutter" in lower:
        shutter = "global"
    elif "rolling shutter" in lower:
        shutter = "rolling"
    features = {
        "illumination": illumination,
        "shutter": shutter,
        "sensor_modality": classify_sensor_modality(text),
        "is_stacked": bool("stacked" in lower or "3-layer" in lower or "three-layer" in lower),
        "has_dbi": bool("dbi" in lower or "hybrid bond" in lower or "hybrid bonding" in lower),
        "has_dti": bool("dti" in lower or "deep trench" in lower),
        "is_tof": bool("time-of-flight" in lower or "tof" in lower),
        "is_spad": bool("spad" in lower or "single-photon avalanche" in lower),
        "is_nir": bool(" nir " in f" {lower} " or "near infrared" in lower or "near-infrared" in lower),
        "has_hdr": bool(" hdr" in f" {lower}" or "high dynamic range" in lower),
        "has_lofic": bool("lofic" in lower),
    }
    features.update({key: value for key, value in classify_dti(text).items() if value})
    features.update({key: value for key, value in classify_pixel_architecture(text).items() if value not in ("", None)})
    features.update({key: value for key, value in classify_optical_features(text).items() if value not in ("", None, False)})
    return features


def derive_specs(record: dict[str, Any]) -> None:
    sources = all_text_sources(record)
    combined_text = " ".join(source["text"] for source in sources)
    combined_features = classify_features(combined_text)
    for key, value in combined_features.items():
        if isinstance(value, bool):
            if value:
                add_spec(record, key, value, {"source": "combined_text", "text": key})
        elif value:
            add_spec(record, key, value, {"source": "combined_text", "text": value})

    for source in sources:
        evidence = {"source_file": source.get("source_file", ""), "source": source["kind"], "text": truncate_text(source["text"], 220)}
        text = source["text"]
        if "pixel_pitch_um" not in record["derived_specs"]:
            pitch = find_pixel_pitch(source)
            add_spec(record, "pixel_pitch_um", pitch, evidence)
        if "resolution_mp" not in record["derived_specs"]:
            resolution = find_resolution_mp(source)
            add_spec(record, "resolution_mp", resolution, evidence)
        if "resolution_x" not in record["derived_specs"]:
            xy = find_resolution_xy(text)
            if xy:
                add_spec(record, "resolution_x", xy[0], evidence)
                add_spec(record, "resolution_y", xy[1], evidence)
        if "optical_format" not in record["derived_specs"]:
            optical_format = find_optical_format(text)
            add_spec(record, "optical_format", optical_format, evidence)
        if "active_si_thickness_um" not in record["derived_specs"]:
            add_spec(record, "active_si_thickness_um", active_si_thickness_value(text), evidence)
        if "optical_stack_height_um" not in record["derived_specs"]:
            add_spec(record, "optical_stack_height_um", value_after_keywords_um(text, r"optical\s+stack\s+height"), evidence)
        if "dti_depth_um" not in record["derived_specs"] and source_contains(source, "dti", "deep trench"):
            add_spec(record, "dti_depth_um", dti_depth_value(text), evidence)
        if "dti_width_nm" not in record["derived_specs"] and source_contains(source, "dti", "deep trench"):
            width_nm = dti_width_nm_value(text)
            add_spec(record, "dti_width_nm", width_nm, evidence)
            add_spec(record, "dti_width_um", nm_to_um(width_nm), evidence)
        if "dti_aspect_ratio" not in record["derived_specs"] and source_contains(source, "aspect ratio"):
            add_spec(record, "dti_aspect_ratio", aspect_ratio_value(text), evidence)
        if "cfa_thickness_um" not in record["derived_specs"] and source_contains(source, "cfa", "color filter"):
            add_spec(record, "cfa_thickness_um", value_after_keywords_um(text, r"(?:cfa|color\s+filter)[^.\n;:]{0,80}?thickness"), evidence)
        if "cfa_thickness_min_um" not in record["derived_specs"] and source_contains(source, "cfa", "color filter"):
            min_um, max_um = extract_range_to_um(text)
            if min_um is not None and max_um is not None and "thickness" in text.lower():
                add_spec(record, "cfa_thickness_min_um", min_um, evidence)
                add_spec(record, "cfa_thickness_max_um", max_um, evidence)
        if "ocl_pitch_um" not in record["derived_specs"]:
            add_spec(record, "ocl_pitch_um", value_after_keywords_um(text, r"(?:ocl|microlens)[^.\n;:]{0,80}?pitch"), evidence)
        if "color_filter_pitch_um" not in record["derived_specs"]:
            add_spec(record, "color_filter_pitch_um", value_after_keywords_um(text, r"color\s+filter[^.\n;:]{0,80}?pitch"), evidence)
        if "grid_pitch_um" not in record["derived_specs"]:
            add_spec(record, "grid_pitch_um", value_after_keywords_um(text, r"grid[^.\n;:]{0,80}?pitch"), evidence)

    for row in record["_rows"]:
        label = row["label"].lower()
        heading = row.get("table_heading", "").lower()
        value = row["value"]
        evidence = {"source_file": row["source_file"], "source": f"table:{row['label']}", "text": truncate_text(value, 220)}
        if "active si thickness" in label:
            add_spec(record, "active_si_thickness_um", first_um(value), evidence)
        elif "optical stack height" in label:
            add_spec(record, "optical_stack_height_um", first_um(value), evidence)
        elif "pixel beol metal pitch" in label:
            nm_values = all_nm(value)
            if nm_values:
                add_spec(record, "pixel_beol_metal_pitch_nm", nm_values[0], evidence)
            if "design" in label and len(nm_values) > 1:
                add_spec(record, "cis_process_nm", nm_values[-1], evidence)
        elif "pixel beol design rules" in label:
            add_spec(record, "cis_process_nm", first_nm(value), evidence)
        elif "logic design rules" in label:
            target_key = "isp_process_nm" if "isp" in heading or "image signal processor" in heading else "logic_process_nm"
            add_spec(record, target_key, first_nm(value), evidence)
        elif "pixel sharing" in label:
            add_spec(record, "pixel_sharing", value, evidence)
            for key, parsed in classify_pixel_architecture(value).items():
                add_spec(record, key, parsed, evidence)
        elif "number, type of metals" in label:
            add_spec(record, "metal_stack", value, evidence)
        elif "die markings" in label and "foundry" in label:
            if "isp" in heading or "image signal processor" in heading:
                continue
            pieces = [normalize_space(piece) for piece in value.split(",") if normalize_space(piece)]
            if pieces:
                add_spec(record, "cis_foundry", pieces[-1], evidence)
        elif "die size" in label or "die stack size" in label:
            die_text, area = parse_die_size(value)
            if "die_size_text" not in record["derived_specs"]:
                add_spec(record, "die_size_text", die_text, evidence)
            if "die_area_mm2" not in record["derived_specs"]:
                add_spec(record, "die_area_mm2", area, evidence)
        elif "package dimensions" in label:
            add_spec(record, "package_dimensions_text", value, evidence)

    if not record["description_excerpt"]:
        for source in record["_text_sources"]:
            if "description" in source["kind"]:
                record["description_excerpt"] = truncate_text(source["text"])
                break


def update_material_paths(stack: dict[str, Any], stack_path: Path) -> None:
    base_dir = BASE_STACK_CONFIG.parent
    for material in stack.get("materials", {}).values():
        if not isinstance(material, dict) or "nk_table" not in material:
            continue
        raw_path = Path(str(material["nk_table"]))
        source_path = raw_path if raw_path.is_absolute() else (base_dir / raw_path).resolve()
        material["nk_table"] = os.path.relpath(source_path, stack_path.parent).replace(os.sep, "/")


def derived_depth_um(record: dict[str, Any], pitch_um: float) -> tuple[float, str]:
    specs = record["derived_specs"]
    active_si = specs.get("active_si_thickness_um")
    if isinstance(active_si, (int, float)) and 0.5 <= float(active_si) <= 20.0:
        return float(active_si), "active_si_thickness_um"
    return round(max(1.2, min(8.0, pitch_um * 2.0)), 3), "pitch_scaled_proxy"


def set_lumped_optical_stack(geometry: dict[str, Any], optical_height_um: float | None, pitch_um: float) -> str:
    if optical_height_um is None or not (0.4 <= optical_height_um <= 8.0):
        geometry["lens_height"] = round(max(0.25, min(0.9, pitch_um * 0.35)), 3)
        geometry["cfa_thickness"] = round(max(0.35, min(1.2, pitch_um * 0.45)), 3)
        geometry["passivation_thickness"] = 0.08
        return "pitch_scaled_proxy"
    lens = max(0.25, min(optical_height_um * 0.34, pitch_um * 0.40, 1.2))
    cfa = max(0.30, min(optical_height_um * 0.42, pitch_um * 0.55, 1.4))
    passivation = max(0.05, optical_height_um - lens - cfa)
    geometry["lens_height"] = round(lens, 3)
    geometry["cfa_thickness"] = round(cfa, 3)
    geometry["passivation_thickness"] = round(passivation, 3)
    return "lumped_from_report_optical_stack_height"


def extracted_cfa_thickness_um(record: dict[str, Any]) -> tuple[float | None, str]:
    specs = record.get("derived_specs", {})
    direct = specs.get("cfa_thickness_um")
    if isinstance(direct, (int, float)) and 0.05 <= float(direct) <= 3.0:
        return float(direct), "cfa_thickness_um"
    low = specs.get("cfa_thickness_min_um")
    high = specs.get("cfa_thickness_max_um")
    if (
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and 0.05 <= float(low) <= float(high) <= 3.0
    ):
        return (float(low) + float(high)) / 2.0, "midpoint(cfa_thickness_min_um,cfa_thickness_max_um)"
    return None, ""


def apply_extracted_cfa_thickness(
    geometry: dict[str, Any],
    record: dict[str, Any],
    optical_height_um: float | None,
) -> str | None:
    value, source = extracted_cfa_thickness_um(record)
    if value is None:
        return None
    old_cfa = geometry.get("cfa_thickness")
    geometry["cfa_thickness"] = round(value, 4)
    if isinstance(optical_height_um, (int, float)) and optical_height_um > 0:
        lens = float(geometry.get("lens_height", 0.25))
        geometry["passivation_thickness"] = round(max(0.05, float(optical_height_um) - lens - value), 4)
    return f"{source}; old_proxy_cfa={old_cfa}; adjusted_passivation_to_preserve_optical_stack={bool(optical_height_um)}"


def source_file_counts(record: dict[str, Any]) -> dict[str, int]:
    return {
        key: len(values) if isinstance(values, list) else 0
        for key, values in record.get("source_files", {}).items()
    }


def source_list(record: dict[str, Any], key: str) -> list[str]:
    files = record.get("source_files", {})
    values = files.get(key, []) if isinstance(files, dict) else []
    return values if isinstance(values, list) else []


def techinsights_source_payload(
    record: dict[str, Any],
    *,
    include_source_paths: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": record["code"],
        "title": record["metadata"].get("title", ""),
        "content_title": record.get("content_title", ""),
        "source_file_counts": source_file_counts(record),
        "source_paths_redacted": not include_source_paths,
        "derived_specs": record["derived_specs"],
    }
    if include_source_paths:
        payload["source_html"] = source_list(record, "html")[:3]
    return payload


def build_stack_config(
    record: dict[str, Any],
    output_dir: Path,
    slug: str,
    *,
    include_source_paths: bool,
) -> Path | None:
    pitch = record["derived_specs"].get("pixel_pitch_um")
    if not isinstance(pitch, (int, float)):
        return None
    pitch_um = float(pitch)
    if not (0.1 <= pitch_um <= 20.0):
        return None
    stack_path = output_dir / "generated_stack_configs" / f"{slug}.json"
    stack = read_json(BASE_STACK_CONFIG)
    stack["name"] = f"{record['code']}_{slug}_techinsights_proxy_stack"
    stack["description"] = (
        "FDTD runnable proxy stack derived from TechInsights local metadata. "
        "Use as a starting geometry, not as a calibrated product deck."
    )
    stack["calibration_status"] = {
        "is_measured": False,
        "geometry_measured": False,
        "mode": "techinsights_metadata_proxy",
        "source_report_code": record["code"],
        "note": "Only public/report metadata fields were extracted; layer split and n,k remain proxy values.",
    }
    geometry = stack["geometry_um"]
    depth_um, depth_source = derived_depth_um(record, pitch_um)
    geometry["pitch"] = round(pitch_um, 4)
    geometry["si_thickness"] = round(depth_um, 4)
    geometry["metal_edge_width"] = round(max(0.03, min(0.18, pitch_um * 0.085)), 4)
    geometry["lens_edge_gap"] = round(max(0.015, min(0.08, pitch_um * 0.03)), 4)
    geometry["bottom_air"] = 0.25
    geometry["pml"] = 0.45 if pitch_um <= 3.0 else 0.60
    optical_height = record["derived_specs"].get("optical_stack_height_um")
    optical_source = set_lumped_optical_stack(
        geometry,
        float(optical_height) if isinstance(optical_height, (int, float)) else None,
        pitch_um,
    )
    cfa_override_source = apply_extracted_cfa_thickness(
        geometry,
        record,
        float(optical_height) if isinstance(optical_height, (int, float)) else None,
    )
    stack["shield"] = {
        "mode": "off",
        "mask_edge_width_um": geometry["metal_edge_width"],
        "pdaf_axis": "x",
        "notes": "Generated database keeps optical masks off unless a measured PDAF/aperture mask is modeled separately.",
    }
    stack["techinsights_source"] = {
        **techinsights_source_payload(record, include_source_paths=include_source_paths),
        "derived_geometry_sources": {
            "pitch": "pixel_pitch_um",
            "si_thickness": depth_source,
            "lumped_optical_stack": optical_source,
            "cfa_thickness": cfa_override_source or "lumped_optical_stack_proxy",
        },
    }
    stack.setdefault("accuracy_notes", [])
    stack["accuracy_notes"] = [
        "Generated from report metadata/tables, not from a process deck.",
        "CFA/OCL/passivation split is a lumped proxy when only optical stack height is known.",
        "Material n,k values are inherited from proxy reference files.",
        "Run optical/TCAD convergence gates before using this quantitatively.",
    ] + list(stack["accuracy_notes"])
    update_material_paths(stack, stack_path)
    write_json(stack_path, stack)
    return stack_path


X_BOUND_KEYS = {"x_min_um", "x_max_um"}
DEPTH_KEYS = {"depth_min_um", "depth_max_um", "depth_peak_um", "depth_sigma_um", "depth_rolloff_um"}
WIDTH_KEYS = {"x_rolloff_um", "liner_width_um", "oxide_thickness_um"}


def scale_profile_geometry(value: Any, scale_x: float, scale_depth: float) -> Any:
    if isinstance(value, list):
        return [scale_profile_geometry(item, scale_x, scale_depth) for item in value]
    if not isinstance(value, dict):
        return value
    scaled = {}
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            scaled[key] = scale_profile_geometry(item, scale_x, scale_depth)
        elif isinstance(item, (int, float)):
            if key in X_BOUND_KEYS or key in WIDTH_KEYS:
                scaled[key] = round(float(item) * scale_x, 6)
            elif key in DEPTH_KEYS:
                scaled[key] = round(float(item) * scale_depth, 6)
            else:
                scaled[key] = item
        else:
            scaled[key] = item
    return scaled


def build_tcad_profile(
    record: dict[str, Any],
    output_dir: Path,
    slug: str,
    *,
    include_source_paths: bool,
) -> Path | None:
    pitch = record["derived_specs"].get("pixel_pitch_um")
    if not isinstance(pitch, (int, float)):
        return None
    pitch_um = float(pitch)
    depth_um, depth_source = derived_depth_um(record, pitch_um)
    profile_dir = output_dir / "generated_tcad_profiles" / slug
    profile_path = profile_dir / "profile.json"
    base = read_json(BASE_TCAD_PROFILE)
    scale_x = pitch_um / 1.4
    scale_depth = depth_um / 3.0
    profile = scale_profile_geometry(base, scale_x, scale_depth)
    geometry = profile["geometry"]
    half = pitch_um / 2.0
    dti_width = max(0.035, min(0.22, pitch_um * 0.045))
    split_gap = max(0.02, min(0.12, pitch_um * 0.03))
    geometry["width_um"] = round(pitch_um, 6)
    geometry["z_width_um"] = round(pitch_um, 6)
    geometry["depth_um"] = round(depth_um, 6)
    geometry["split_gap_um"] = round(split_gap, 6)
    geometry["pinning_depth_um"] = round(max(0.04, min(0.20, depth_um * 0.027)), 6)
    geometry["dti_width_um"] = round(dti_width, 6)
    if isinstance(geometry.get("bdti"), dict):
        geometry["bdti"]["x_left_min_um"] = round(-half, 6)
        geometry["bdti"]["x_left_max_um"] = round(-half + dti_width, 6)
        geometry["bdti"]["x_right_min_um"] = round(half - dti_width, 6)
        geometry["bdti"]["x_right_max_um"] = round(half, 6)
        geometry["bdti"]["depth_max_um"] = round(min(depth_um, max(0.8, depth_um * 0.4)), 6)
        geometry["bdti"]["note"] = (
            "Proxy BDTI geometry scaled from TechInsights-derived pixel pitch. "
            "Replace with measured trench geometry for product work."
        )
    profile["profile_name"] = f"{record['code']}_{slug}_techinsights_proxy_profile"
    profile["reference_mode"] = True
    profile["reference_notes"] = [
        "Generated from TechInsights metadata for FDTD/TCAD setup exploration.",
        "This is not a measured implant profile or calibrated process deck.",
        f"Pixel width came from derived pixel_pitch_um; depth source: {depth_source}.",
    ]
    profile["calibration_status"] = {
        "is_measured": False,
        "geometry_measured": False,
        "mode": "techinsights_metadata_proxy",
        "source_report_code": record["code"],
        "note": "Doping terms are scaled from the reference proxy profile and are not product measured.",
    }
    profile["techinsights_source"] = techinsights_source_payload(
        record,
        include_source_paths=include_source_paths,
    )
    write_json(profile_path, profile)
    return profile_path


def html_link(path: str, *, base_dir: Path | None = None) -> str:
    if not path:
        return ""
    path_obj = Path(path)
    if not path_obj.is_absolute() and base_dir is not None:
        path_obj = base_dir / path_obj
    return f"file://{path_obj.resolve()}"


def table_link(path: str, label: str) -> str:
    if not path:
        return ""
    return f'<a href="{escape(path)}">{escape(label)}</a>'


def record_slug(record: dict[str, Any]) -> str:
    pieces = [
        record["code"],
        normalize_space(record["metadata"].get("manufacturer", "")),
        normalize_space(record["metadata"].get("device_name", "")),
    ]
    return safe_slug("_".join(piece for piece in pieces if piece))


def clean_record_for_output(
    record: dict[str, Any],
    *,
    include_source_paths: bool,
    include_evidence_text: bool,
) -> dict[str, Any]:
    cleaned = {key: value for key, value in record.items() if not key.startswith("_")}
    evidence = cleaned.get("derived_evidence", {})
    cleaned["derived_evidence"] = {
        key: [
            {
                "source_file": item.get("source_file", "") if include_source_paths else REDACTED_SOURCE_PATH,
                "source": item.get("source", ""),
                "text": truncate_text(item.get("text", ""), 220)
                if include_evidence_text
                else REDACTED_EVIDENCE_TEXT,
            }
            for item in values[:3]
        ]
        for key, values in evidence.items()
    }
    cleaned["table_evidence"] = []
    cleaned["source_file_counts"] = source_file_counts(record)
    cleaned["source_paths_redacted"] = not include_source_paths
    cleaned["evidence_text_redacted"] = not include_evidence_text
    if not include_source_paths:
        cleaned["source_files"] = {}
        cleaned["source_dirs"] = []
    return cleaned


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    specs = record["derived_specs"]
    source_html = source_list(record, "html")
    source_pdf = source_list(record, "pdf")
    row = {
        "code": record["code"],
        "report_type": metadata.get("report_type", ""),
        "release_date": metadata.get("release_date", ""),
        "analysis_year": metadata.get("analysis_year", ""),
        "manufacturer": metadata.get("manufacturer", ""),
        "device_name": metadata.get("device_name", ""),
        "device_type": metadata.get("device_type", ""),
        "title": metadata.get("title", ""),
        "content_title": record.get("content_title", ""),
        "source_html": source_html[0] if source_html else "",
        "source_pdf": source_pdf[0] if source_pdf else "",
        "stack_config": record["generated_files"].get("stack_config", ""),
        "tcad_profile": record["generated_files"].get("tcad_profile", ""),
    }
    for key in CSV_FIELDS:
        if key not in row:
            value = specs.get(key, "")
            if isinstance(value, bool):
                value = "true" if value else "false"
            row[key] = value
    return row


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_row(record))


def write_index_html(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        row = csv_row(record)
        html_path = html_link(row["source_html"])
        pdf_path = html_link(row["source_pdf"])
        stack_path = html_link(row["stack_config"], base_dir=path.parent)
        profile_path = html_link(row["tcad_profile"], base_dir=path.parent)
        html_cell = table_link(html_path, "HTML")
        pdf_cell = table_link(pdf_path, "PDF")
        stack_cell = table_link(stack_path, "stack")
        profile_cell = table_link(profile_path, "profile")
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['code']))}</td>"
            f"<td>{escape(str(row['report_type']))}</td>"
            f"<td>{escape(str(row['release_date'])[:10])}</td>"
            f"<td>{escape(str(row['manufacturer']))}</td>"
            f"<td>{escape(str(row['device_name']))}</td>"
            f"<td>{escape(str(row['pixel_pitch_um']))}</td>"
            f"<td>{escape(str(row['resolution_mp']))}</td>"
            f"<td>{escape(str(row['active_si_thickness_um']))}</td>"
            f"<td>{escape(str(row['optical_stack_height_um']))}</td>"
            f"<td>{escape(str(row['dti_type']))}</td>"
            f"<td>{escape(str(row['dti_depth_um']))}</td>"
            f"<td>{escape(str(row['transfer_gate_type']))}</td>"
            f"<td>{escape(str(row['pixel_architecture']))}</td>"
            f"<td>{escape(str(row['cfa_pattern']))}</td>"
            f"<td>{escape(str(row['illumination']))}</td>"
            f"<td>{escape(str(row['shutter']))}</td>"
            f"<td>{html_cell}</td>"
            f"<td>{pdf_cell}</td>"
            f"<td>{stack_cell}</td>"
            f"<td>{profile_cell}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Sensor DB</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 24px; color: #1f2933; background: #f7f9fb; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ margin: 0 0 16px; color: #52606d; }}
    input {{ width: min(680px, 100%); padding: 10px 12px; border: 1px solid #bcccdc; border-radius: 6px; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: white; }}
    th, td {{ border-bottom: 1px solid #e4e7eb; padding: 8px 10px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #edf2f7; z-index: 1; }}
    a {{ color: #0967d2; }}
  </style>
</head>
<body>
  <h1>Image Sensor DB</h1>
  <p>{len(records)} image-sensor records. Extracted factual fields only; source report HTML/PDF stays in the local TechInsights export folders. Source links appear only when generated with private output enabled.</p>
  <input id="q" placeholder="Filter by code, manufacturer, device, pitch, feature">
  <table id="records">
    <thead><tr>
      <th>Code</th><th>Type</th><th>Release</th><th>Manufacturer</th><th>Device</th>
      <th>Pitch (&micro;m)</th><th>MP</th><th>Si (&micro;m)</th><th>Optical (&micro;m)</th>
      <th>DTI</th><th>DTI depth</th><th>TG</th><th>Pixel arch</th><th>CFA</th>
      <th>Illumination</th><th>Shutter</th><th>HTML</th><th>PDF</th><th>Stack</th><th>TCAD</th>
    </tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <script>
    const q = document.getElementById('q');
    const rows = Array.from(document.querySelectorAll('#records tbody tr'));
    q.addEventListener('input', () => {{
      const needle = q.value.toLowerCase();
      for (const row of rows) {{
        row.style.display = row.textContent.toLowerCase().includes(needle) ? '' : 'none';
      }}
    }});
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_readme(path: Path, catalog: dict[str, Any]) -> None:
    records = catalog["records"]
    generated_stacks = sum(1 for record in records if record.get("generated_files", {}).get("stack_config"))
    generated_profiles = sum(1 for record in records if record.get("generated_files", {}).get("tcad_profile"))
    text = f"""# Image Sensor DB

Generated at `{catalog['generated_at']}` from local TechInsights export folders.

`{catalog['source_root']}`

This database is for FDTD/TCAD setup exploration. It intentionally stores extracted
metadata and proxy simulation inputs. Default output redacts local source paths and
evidence text. If this database was generated with `--private-output`, treat it as
local-only and do not commit or redistribute it.

## Contents

- `sensor_catalog.json`: normalized records, extracted specs, redacted evidence metadata by default.
- `sensor_catalog.csv`: flat table for spreadsheet filtering.
- `index.html`: local browser table with generated stack/profile links; source links only in private output.
- `generated_stack_configs/`: runnable `image_sensor_stack_v1` proxy stack configs.
- `generated_tcad_profiles/`: runnable `measured_tcad_profile_v1` proxy profiles.
- `validation.json`: extraction counts and coverage.

## Coverage

- Records: {len(records)}
- Source records before image-sensor filter: {catalog['validation'].get('records_before_filter', '')}
- Parsed PDF files: {catalog['validation']['source_summary'].get('pdf_files_parsed', 0)}
- Generated FDTD stack configs: {generated_stacks}
- Generated TCAD proxy profiles: {generated_profiles}
- Records with pixel pitch: {catalog['validation']['records_with_pixel_pitch']}
- Records with active Si thickness: {catalog['validation']['records_with_active_si']}
- Records with optical stack height: {catalog['validation']['records_with_optical_stack_height']}
- Records with DTI type: {catalog['validation'].get('records_with_dti_type', 0)}
- Records with transfer-gate type: {catalog['validation'].get('records_with_transfer_gate_type', 0)}

## Important Limitations

- These generated stack/profile files are not measured process decks.
- The catalog is filtered to image-sensor/CIS records; camera modules, packaging-only reports, ToF/LiDAR/SPAD/fingerprint/thermal records are excluded by default.
- Pixel pitch, active Si thickness, optical stack height, DTI, CFA/OCL, transfer-gate, and similar values are extracted from local report metadata, HTML tables, and PDF text snippets when present.
- CFA, microlens, passivation split and material n,k are inherited proxy assumptions unless the source explicitly exposed more detail.
- Doping profiles are scaled from the existing reference proxy TCAD profile, not extracted from TechInsights implant data.
- Treat all generated configs as starting points for simulation setup, then run convergence and accuracy gates before quantitative use.
- Do not commit generated database directories; they are local research artifacts derived from licensed/reference sources.
"""
    path.write_text(text, encoding="utf-8")


def sanitized_source_summary(source_summary: dict[str, Any], *, include_source_paths: bool) -> dict[str, Any]:
    if include_source_paths:
        return deepcopy(source_summary)
    sanitized: dict[str, Any] = {}
    for key, value in source_summary.items():
        if key in {"export_dirs", "parse_failures"}:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
    sanitized["export_dir_count"] = len(source_summary.get("export_dirs", []))
    sanitized["parse_failure_count"] = len(source_summary.get("parse_failures", []))
    sanitized["source_paths_redacted"] = True
    return sanitized


def validation_summary(records: list[dict[str, Any]], source_summary: dict[str, Any]) -> dict[str, Any]:
    def has_spec(key: str) -> int:
        return sum(1 for record in records if key in record.get("derived_specs", {}))

    def has_source_file_kind(key: str) -> int:
        count = 0
        for record in records:
            source_files = record.get("source_files", {})
            if isinstance(source_files, dict) and source_files.get(key):
                count += 1
                continue
            source_counts = record.get("source_file_counts", {})
            if isinstance(source_counts, dict) and source_counts.get(key, 0):
                count += 1
        return count

    report_types: dict[str, int] = {}
    manufacturers: dict[str, int] = {}
    for record in records:
        report_type = normalize_space(record.get("metadata", {}).get("report_type", "")) or "unknown"
        manufacturer = normalize_space(record.get("metadata", {}).get("manufacturer", "")) or "unknown"
        report_types[report_type] = report_types.get(report_type, 0) + 1
        manufacturers[manufacturer] = manufacturers.get(manufacturer, 0) + 1

    return {
        "source_summary": source_summary,
        "record_count": len(records),
        "records_with_pixel_pitch": has_spec("pixel_pitch_um"),
        "records_with_resolution_mp": has_spec("resolution_mp"),
        "records_with_active_si": has_spec("active_si_thickness_um"),
        "records_with_optical_stack_height": has_spec("optical_stack_height_um"),
        "records_with_dti_type": has_spec("dti_type"),
        "records_with_dti_depth": has_spec("dti_depth_um"),
        "records_with_transfer_gate_type": has_spec("transfer_gate_type"),
        "records_with_pixel_architecture": has_spec("pixel_architecture"),
        "records_with_cfa_pattern": has_spec("cfa_pattern"),
        "records_with_source_pdf": has_source_file_kind("pdf"),
        "records_with_generated_stack": sum(1 for record in records if record.get("generated_files", {}).get("stack_config")),
        "records_with_generated_tcad_profile": sum(1 for record in records if record.get("generated_files", {}).get("tcad_profile")),
        "report_types": dict(sorted(report_types.items())),
        "top_manufacturers": dict(sorted(manufacturers.items(), key=lambda item: (-item[1], item[0]))[:25]),
        "parse_failure_count": source_summary.get(
            "parse_failure_count",
            len(source_summary.get("parse_failures", [])),
        ),
    }


def dedupe_sources(records: dict[str, dict[str, Any]]) -> None:
    for record in records.values():
        record["source_dirs"] = sorted(set(record["source_dirs"]))
        for key, values in record["source_files"].items():
            record["source_files"][key] = sorted(set(values))


def build_database(
    techinsights_root: Path,
    output_dir: Path,
    pdf_max_pages: int = 0,
    *,
    include_source_paths: bool = False,
    include_evidence_text: bool = False,
) -> dict[str, Any]:
    records_map, source_summary = collect_records(techinsights_root)
    attach_pdf_sources(records_map, techinsights_root, source_summary, pdf_max_pages=pdf_max_pages)
    dedupe_sources(records_map)
    for record in records_map.values():
        derive_specs(record)

    records_before_filter = len(records_map)
    records_map = {code: record for code, record in records_map.items() if is_image_sensor_record(record)}
    source_summary["records_before_filter"] = records_before_filter
    source_summary["records_after_image_sensor_filter"] = len(records_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    for generated_dir in (output_dir / "generated_stack_configs", output_dir / "generated_tcad_profiles"):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    output_records: list[dict[str, Any]] = []
    for code in sorted(records_map):
        record = records_map[code]
        slug = record_slug(record)
        stack_path = build_stack_config(
            record,
            output_dir,
            slug,
            include_source_paths=include_source_paths,
        )
        profile_path = build_tcad_profile(
            record,
            output_dir,
            slug,
            include_source_paths=include_source_paths,
        )
        if stack_path:
            record["generated_files"]["stack_config"] = relpath_for_output(stack_path, output_dir)
        if profile_path:
            record["generated_files"]["tcad_profile"] = relpath_for_output(profile_path, output_dir)
        output_records.append(
            clean_record_for_output(
                record,
                include_source_paths=include_source_paths,
                include_evidence_text=include_evidence_text,
            )
        )

    source_summary_for_output = sanitized_source_summary(
        source_summary,
        include_source_paths=include_source_paths,
    )
    validation = validation_summary(output_records, source_summary_for_output)
    validation["records_before_filter"] = records_before_filter
    validation["records_excluded_by_image_sensor_filter"] = records_before_filter - len(output_records)
    catalog = {
        "schema": "image_sensor_db_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(techinsights_root) if include_source_paths else REDACTED_SOURCE_ROOT,
        "source_policy": (
            "Private local output includes source paths/evidence text; do not commit or redistribute."
            if include_source_paths or include_evidence_text
            else "Default sanitized output: extracted facts only; source paths and evidence text are redacted."
        ),
        "private_output": include_source_paths or include_evidence_text,
        "source_paths_redacted": not include_source_paths,
        "evidence_text_redacted": not include_evidence_text,
        "base_stack_config": repo_relative(BASE_STACK_CONFIG),
        "base_tcad_profile": repo_relative(BASE_TCAD_PROFILE),
        "validation": validation,
        "records": output_records,
    }
    write_json(output_dir / "sensor_catalog.json", catalog)
    write_json(output_dir / "validation.json", validation)
    write_csv(output_dir / "sensor_catalog.csv", output_records)
    write_index_html(output_dir / "index.html", output_records)
    write_readme(output_dir / "README.md", catalog)
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--techinsights-root", type=Path, default=DEFAULT_TECHINSIGHTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pdf-max-pages",
        type=int,
        default=0,
        help="Maximum pages to extract per PDF; 0 means all pages.",
    )
    parser.add_argument(
        "--private-source-links",
        action="store_true",
        help="Include local source HTML/PDF paths in outputs. Local-only; do not commit generated output.",
    )
    parser.add_argument(
        "--private-evidence-text",
        action="store_true",
        help="Include short extracted evidence text in outputs. Local-only; do not commit generated output.",
    )
    parser.add_argument(
        "--private-output",
        action="store_true",
        help="Shortcut for --private-source-links --private-evidence-text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.techinsights_root.exists():
        raise SystemExit(f"TechInsights root not found: {args.techinsights_root}")
    if PdfReader is None:
        print(f"Warning: pypdf is unavailable; PDF snippet extraction will be skipped: {PYPDF_IMPORT_ERROR}")
    include_source_paths = bool(args.private_output or args.private_source_links)
    include_evidence_text = bool(args.private_output or args.private_evidence_text)
    catalog = build_database(
        args.techinsights_root.resolve(),
        args.output_dir.resolve(),
        pdf_max_pages=args.pdf_max_pages,
        include_source_paths=include_source_paths,
        include_evidence_text=include_evidence_text,
    )
    validation = catalog["validation"]
    print(f"Wrote {validation['record_count']} records to {args.output_dir.resolve()}")
    print(f"Source records before image-sensor filter: {validation['records_before_filter']}")
    print(f"Excluded by image-sensor filter: {validation['records_excluded_by_image_sensor_filter']}")
    print(f"Parsed PDF files: {validation['source_summary'].get('pdf_files_parsed', 0)}")
    print(f"Pixel pitch coverage: {validation['records_with_pixel_pitch']} records")
    print(f"DTI type coverage: {validation.get('records_with_dti_type', 0)} records")
    print(f"Generated stack configs: {validation['records_with_generated_stack']}")
    print(f"Generated TCAD profiles: {validation['records_with_generated_tcad_profile']}")
    print(
        "Output policy: "
        + ("PRIVATE local output; do not commit" if catalog["private_output"] else "sanitized output")
    )
    if validation["parse_failure_count"]:
        print(f"Parse failures: {validation['parse_failure_count']} (see validation.json)")


if __name__ == "__main__":
    main()
