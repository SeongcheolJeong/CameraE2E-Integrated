#!/usr/bin/env python3
"""Build a local-only TCAD candidate report from the image sensor database.

This script does not create a calibrated process deck. It audits the local
TechInsights-derived sensor database for fields that can seed TCAD work:

- geometry and mesh anchors from the extracted image-sensor DB
- HTML tables/snippets that look like SIMS/SRP/SCM/SMIM doping evidence
- optional PDF keyword snippets when pypdf is available

The output is intended for private, local review only.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore[assignment]
    PYPDF_IMPORT_ERROR = exc
else:
    PYPDF_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "image_sensor_db" / "sensor_catalog.json"
DEFAULT_OUTPUT_DIR = ROOT / "image_sensor_db"

TCAD_KEYWORDS = (
    "SIMS",
    "SRP",
    "SCM",
    "SMIM",
    "sMIM",
    "doping",
    "dopant",
    "dopants",
    "P-well",
    "N-well",
    "P-type",
    "N-type",
    "P+",
    "N+",
    "photocathode",
    "photodiode",
    "floating diffusion",
    "FD",
    "transfer gate",
    "VTG",
    "D-VTG",
    "gate dielectric",
    "gate oxide",
    "oxide thickness",
    "pinning",
    "backside passivation",
    "fixed charge",
    "STI",
    "DTI",
    "deep trench",
    "trench",
    "isolation",
)

DOPING_TABLE_TERMS = (
    "doping concentration",
    "doping type",
    "dopant",
    "cm-3",
    "cm^-3",
    "sims",
    "srp",
)

GEOMETRY_FIELDS = (
    "pixel_pitch_um",
    "active_si_thickness_um",
    "dti_type",
    "dti_depth_um",
    "dti_width_nm",
    "transfer_gate_type",
    "pixel_architecture",
    "cis_process_nm",
    "pixel_beol_metal_pitch_nm",
    "optical_stack_height_um",
)

COMMON_NOISE = (
    "TechInsights calibrates length measurements",
    "Statement of Scope Variation",
    "Secondary ion mass spectrometry (SIMS) data may be calibrated",
)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def short_text(value: str, limit: int = 320) -> str:
    value = normalize_space(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rel_to(path: str | Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        try:
            return Path(path).resolve().as_posix()
        except Exception:
            return str(path)


@dataclass
class ParsedTable:
    caption: str = ""
    rows: list[list[str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        row_text = " ".join(" | ".join(row) for row in self.rows)
        return normalize_space(f"{self.caption} {row_text}")


class ReportHTMLParser(HTMLParser):
    """Small table/text extractor that works without BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.last_caption = ""
        self._in_figcaption = False
        self._figcaption_parts: list[str] = []
        self._in_table = False
        self._table: ParsedTable | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self.tables: list[ParsedTable] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"p", "h1", "h2", "h3", "h4", "tr", "li", "figcaption"}:
            self.text_parts.append("\n")
        if tag == "figcaption":
            self._in_figcaption = True
            self._figcaption_parts = []
        elif tag == "table":
            self._in_table = True
            self._table = ParsedTable(caption=self.last_caption)
        elif self._in_table and tag == "tr":
            self._row = []
        elif self._in_table and tag in {"td", "th"}:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if not data:
            return
        self.text_parts.append(data)
        if self._in_figcaption:
            self._figcaption_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"p", "h1", "h2", "h3", "h4", "tr", "li", "figcaption"}:
            self.text_parts.append("\n")
        if tag == "figcaption":
            self._in_figcaption = False
            caption = normalize_space(" ".join(self._figcaption_parts))
            if caption:
                self.last_caption = caption
        elif self._in_table and tag in {"td", "th"}:
            cell = normalize_space(" ".join(self._cell_parts or []))
            if self._row is not None:
                self._row.append(cell)
            self._cell_parts = None
        elif self._in_table and tag == "tr":
            if self._table is not None and self._row:
                self._table.rows.append(self._row)
            self._row = None
        elif tag == "table":
            if self._table is not None:
                self.tables.append(self._table)
            self._table = None
            self._in_table = False


def parse_html(path: Path) -> tuple[str, list[ParsedTable]]:
    parser = ReportHTMLParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return normalize_space(" ".join(parser.text_parts)), parser.tables


def snippet_around(text: str, pattern: re.Pattern[str], limit: int = 360) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    start = max(0, match.start() - limit // 2)
    end = min(len(text), match.end() + limit // 2)
    return short_text(text[start:end], limit)


def keyword_snippets(text: str, limit: int = 8) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    for keyword in TCAD_KEYWORDS:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - 170)
            end = min(len(text), match.end() + 190)
            snippet = short_text(text[start:end], 420)
            if any(noise in snippet for noise in COMMON_NOISE):
                continue
            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            snippets.append(snippet)
            if len(snippets) >= limit:
                return snippets
    return snippets


def table_is_doping(table: ParsedTable) -> bool:
    lower = table.text.lower()
    header = " ".join(table.rows[0]).lower() if table.rows else ""
    asset_header_terms = ("asset", "classification", "mime", "file size", "file_size", "contentid")
    if any(term in header for term in asset_header_terms):
        return False
    doping_context = (
        "doping" in lower
        or "dopant" in lower
        or "sims" in lower
        or "srp" in lower
        or "p- and n-type" in lower
        or "p-type" in lower
        or "n-type" in lower
        or "p-well" in lower
        or "n-well" in lower
        or "photocathode" in lower
    )
    if not doping_context:
        return False
    if "doping concentration" in lower:
        return True
    if "doping levels" in lower and ("p-type" in lower or "n-type" in lower):
        return True
    if ("concentration" in header or "dopant" in header or "doping type" in header) and (
        "dopant" in header or "doping type" in header or "cm-3" in lower or "cm^-3" in lower
    ):
        return True
    return False


def table_is_tcad_relevant(table: ParsedTable) -> bool:
    lower = table.text.lower()
    header = " ".join(table.rows[0]).lower() if table.rows else ""
    asset_header_terms = ("asset", "classification", "mime", "file size", "file_size", "contentid")
    if any(term in header for term in asset_header_terms):
        return False
    terms = (
        "gate dielectric",
        "front dielectrics",
        "back layers",
        "doping",
        "sims",
        "srp",
        "photocathode",
        "p-well",
        "n-well",
        "dti",
        "sti",
        "metal interconnect",
        "thickness",
    )
    return any(term in lower for term in terms)


def normalize_concentration(value: str) -> list[float]:
    """Return approximate concentration values from strings like 2 x 10^20."""
    values: list[float] = []
    cleaned = value.replace("\u00d7", "x").replace("×", "x")
    # Convert HTML-ish exponent markers like 10^20^ or 10^20.
    patterns = [
        re.compile(r"([<>~]?\s*\d+(?:\.\d+)?)\s*x\s*10\^?\s*([+-]?\d+)\^?", re.I),
        re.compile(r"([<>~]?\s*\d+(?:\.\d+)?)\s*e\s*([+-]?\d+)", re.I),
    ]
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            coeff_text = match.group(1).replace("<", "").replace(">", "").replace("~", "").strip()
            try:
                coeff = float(coeff_text)
                exp = int(match.group(2))
            except ValueError:
                continue
            number = coeff * (10**exp)
            if math.isfinite(number):
                values.append(number)
    return values


def parse_float(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_depth_um(value: str) -> float | None:
    number = parse_float(value)
    if number is None:
        return None
    lower = value.lower()
    if "nm" in lower and not any(unit in lower for unit in ("um", "µm", "μm")):
        return number / 1000.0
    return number


def normalize_doping_rows(table: ParsedTable, max_rows: int = 16) -> list[dict[str, Any]]:
    rows = table.rows
    if not rows:
        return []
    header = [normalize_space(cell).lower() for cell in rows[0]]
    out: list[dict[str, Any]] = []
    for raw_row in rows[1 : max_rows + 1]:
        cells = raw_row + [""] * max(0, len(header) - len(raw_row))
        item: dict[str, Any] = {"raw": raw_row}
        for idx, name in enumerate(header):
            value = cells[idx] if idx < len(cells) else ""
            if "feature" in name or "layer" == name:
                item["feature"] = value
            elif "type" in name or "dopant" in name:
                item["dopant"] = value
            elif "concentration" in name or "cm" in name:
                item["concentration_text"] = value
                parsed = normalize_concentration(value)
                if parsed:
                    item["concentration_cm3_values"] = parsed
            elif "depth" in name:
                item["depth_text"] = value
                depth = parse_depth_um(value)
                if depth is not None:
                    item["depth_um"] = depth
            elif "thickness" in name:
                item["thickness_text"] = value
        if len(item) > 1:
            out.append(item)
    return out


def extract_region_depths(text: str) -> list[dict[str, Any]]:
    unit = r"(um|µm|μm|nm)"
    number = r"(\d+(?:\.\d+)?)"
    connector = r"(?:is|are|was|were|measures|measure|of|about|confirmed\s+(?:at|as|to\s+be))"
    feature_patterns: list[tuple[str, list[re.Pattern[str]], str]] = [
        (
            "p_well_depth_um",
            [
                re.compile(rf"P[- ]?well\s+{connector}\s*{number}\s*{unit}\s*[- ]?(?:deep|thick)", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?(?:deep|thick)\s+P[- ]?well", re.I),
            ],
            "um",
        ),
        (
            "n_well_depth_um",
            [
                re.compile(rf"N[- ]?well\s+{connector}\s*{number}\s*{unit}\s*[- ]?(?:deep|thick)", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?(?:deep|thick)\s+N[- ]?well", re.I),
            ],
            "um",
        ),
        (
            "photocathode_depth_um",
            [
                re.compile(rf"(?:N[- ]?type\s+)?photocathode\s+{connector}\s*{number}\s*{unit}\s*[- ]?(?:deep|thick)", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?(?:deep|thick)\s+(?:N[- ]?type\s+)?photocathode", re.I),
            ],
            "um",
        ),
        (
            "fd_depth_um",
            [
                re.compile(rf"(?:N\^?\+?\^?\s*)?FD\s+{connector}\s*{number}\s*{unit}\s*[- ]?deep", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?deep\s+(?:N\^?\+?\^?\s*)?FD", re.I),
            ],
            "um",
        ),
        (
            "sd_depth_um",
            [
                re.compile(rf"S/D\s+(?:region\s+)?{connector}\s*{number}\s*{unit}\s*[- ]?deep", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?deep\s+(?:N\^?\+?\^?\s*)?S/D", re.I),
            ],
            "um",
        ),
        (
            "p_plus_contact_depth_um",
            [
                re.compile(rf"P\^?\+?\^?[^.{{}}]{{0,40}}?contact\s+{connector}\s*{number}\s*{unit}\s*[- ]?deep", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?deep\s+P\^?\+?\^?[^.{{}}]{{0,40}}?contact", re.I),
            ],
            "um",
        ),
        (
            "sti_depth_um",
            [
                re.compile(rf"STI(?:\s+trench)?\s+{connector}\s*{number}\s*{unit}\s*[- ]?(?:deep|thick)", re.I),
                re.compile(rf"{number}\s*{unit}\s*[- ]?(?:deep|thick)\s+STIs?", re.I),
            ],
            "um",
        ),
        (
            "gate_dielectric_nm",
            [re.compile(rf"gate dielectric[^.]{{0,100}}?{number}\s*(nm)", re.I)],
            "nm",
        ),
        (
            "gate_oxide_nm",
            [re.compile(rf"gate oxide[^.]{{0,100}}?{number}\s*(nm)", re.I)],
            "nm",
        ),
    ]
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    for label, patterns, target_unit in feature_patterns:
        for pattern in patterns:
            for match in pattern.finditer(text):
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                source_unit = match.group(2).lower() if len(match.groups()) >= 2 else target_unit
                if target_unit == "um" and source_unit == "nm":
                    value = value / 1000.0
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 100)
                context = short_text(text[start:end], 260)
                key = (label, value, context.lower())
                if key in seen:
                    continue
                seen.add(key)
                hits.append({"field": label, "value": value, "context": context})
                if len(hits) >= 20:
                    return hits
    return hits


def extract_pdf_snippets(path: Path, max_pages: int | None, snippet_limit: int = 4) -> tuple[list[str], int, str | None]:
    if PdfReader is None:
        return [], 0, f"pypdf unavailable: {PYPDF_IMPORT_ERROR}"
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return [], 0, f"open failed: {exc}"
    snippets: list[str] = []
    pages_read = 0
    pattern = re.compile("|".join(re.escape(k) for k in TCAD_KEYWORDS), re.I)
    for page_index, page in enumerate(reader.pages):
        if max_pages is not None and page_index >= max_pages:
            break
        pages_read += 1
        try:
            text = normalize_space(page.extract_text() or "")
        except Exception:
            continue
        snippet = snippet_around(text, pattern)
        if snippet and not any(noise in snippet for noise in COMMON_NOISE):
            snippets.append(f"p{page_index + 1}: {snippet}")
            if len(snippets) >= snippet_limit:
                break
    return snippets, pages_read, None


def geometry_score(specs: dict[str, Any]) -> tuple[int, list[str]]:
    fields = [field for field in GEOMETRY_FIELDS if specs.get(field) not in (None, "", [])]
    return len(fields), fields


def classify_candidate(
    geometry_points: int,
    doping_rows: int,
    region_depths: int,
    html_snippets: int,
    pdf_snippets: int,
) -> str:
    if doping_rows:
        return "measured_doping_table"
    if region_depths and (html_snippets or pdf_snippets):
        return "region_depths_and_process_evidence"
    if geometry_points >= 5 and (html_snippets or pdf_snippets):
        return "geometry_plus_process_evidence"
    if geometry_points >= 5:
        return "geometry_only"
    if html_snippets or pdf_snippets:
        return "process_snippet_only"
    return "insufficient_for_tcad"


def build_record_report(
    record: dict[str, Any],
    base_dir: Path,
    include_pdf: bool,
    max_pdf_pages: int | None,
) -> dict[str, Any]:
    specs = record.get("derived_specs", {})
    metadata = record.get("metadata", {})
    source_files = record.get("source_files", {})
    geometry_points, geometry_fields = geometry_score(specs)

    html_snips: list[str] = []
    region_depths: list[dict[str, Any]] = []
    doping_tables: list[dict[str, Any]] = []
    tcad_tables: list[dict[str, Any]] = []

    for html_path_text in source_files.get("html", []):
        path = Path(html_path_text)
        if not path.exists():
            continue
        text, tables = parse_html(path)
        html_snips.extend(keyword_snippets(text, limit=8 - len(html_snips)))
        region_depths.extend(extract_region_depths(text))
        for table in tables:
            if table_is_doping(table):
                rows = normalize_doping_rows(table)
                if not rows:
                    continue
                doping_tables.append(
                    {
                        "source": rel_to(path, base_dir),
                        "caption": table.caption,
                        "row_count": max(0, len(table.rows) - 1),
                        "rows": rows,
                    }
                )
            elif table_is_tcad_relevant(table) and len(tcad_tables) < 6:
                tcad_tables.append(
                    {
                        "source": rel_to(path, base_dir),
                        "caption": table.caption,
                        "row_count": max(0, len(table.rows) - 1),
                        "preview_rows": table.rows[:5],
                    }
                )

    pdf_snips: list[str] = []
    pdf_pages_read = 0
    pdf_errors: list[str] = []
    if include_pdf:
        for pdf_path_text in source_files.get("pdf", []):
            path = Path(pdf_path_text)
            if not path.exists():
                continue
            snippets, pages_read, error = extract_pdf_snippets(path, max_pages=max_pdf_pages)
            pdf_pages_read += pages_read
            if error:
                pdf_errors.append(f"{rel_to(path, base_dir)}: {error}")
            for snippet in snippets:
                if len(pdf_snips) >= 8:
                    break
                pdf_snips.append(f"{rel_to(path, base_dir)} {snippet}")

    # De-duplicate extracted depth entries.
    unique_depths: list[dict[str, Any]] = []
    depth_seen: set[tuple[str, float, str]] = set()
    for hit in region_depths:
        key = (hit.get("field", ""), hit.get("value", 0.0), normalize_space(hit.get("context", "")).lower())
        if key in depth_seen:
            continue
        depth_seen.add(key)
        unique_depths.append(hit)
        if len(unique_depths) >= 20:
            break

    doping_row_count = sum(len(table.get("rows", [])) for table in doping_tables)
    candidate_level = classify_candidate(
        geometry_points=geometry_points,
        doping_rows=doping_row_count,
        region_depths=len(unique_depths),
        html_snippets=len(html_snips),
        pdf_snippets=len(pdf_snips),
    )

    return {
        "code": record.get("code"),
        "candidate_level": candidate_level,
        "tcad_score": geometry_points + min(10, doping_row_count * 2) + min(5, len(unique_depths)) + min(3, len(html_snips) + len(pdf_snips)),
        "geometry_score": geometry_points,
        "geometry_fields": geometry_fields,
        "manufacturer": metadata.get("manufacturer"),
        "device_name": metadata.get("device_name"),
        "title": metadata.get("title"),
        "report_type": metadata.get("report_type"),
        "release_date": metadata.get("release_date"),
        "key_geometry": {field: specs.get(field) for field in GEOMETRY_FIELDS if specs.get(field) not in (None, "", [])},
        "doping_table_count": len(doping_tables),
        "doping_row_count": doping_row_count,
        "doping_tables": doping_tables[:6],
        "region_depths": unique_depths,
        "html_snippets": html_snips[:8],
        "pdf_snippets": pdf_snips[:8],
        "pdf_pages_read": pdf_pages_read,
        "pdf_errors": pdf_errors[:6],
        "source_html": [rel_to(path, base_dir) for path in source_files.get("html", [])],
        "source_pdf": [rel_to(path, base_dir) for path in source_files.get("pdf", [])],
        "generated_tcad_profile": record.get("generated_files", {}).get("tcad_profile"),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "code",
        "candidate_level",
        "tcad_score",
        "geometry_score",
        "manufacturer",
        "device_name",
        "report_type",
        "pixel_pitch_um",
        "active_si_thickness_um",
        "dti_type",
        "dti_depth_um",
        "dti_width_nm",
        "transfer_gate_type",
        "pixel_architecture",
        "doping_table_count",
        "doping_row_count",
        "region_depth_count",
        "html_snippet_count",
        "pdf_snippet_count",
        "source_html",
        "source_pdf",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, rec in enumerate(records, start=1):
            geom = rec.get("key_geometry", {})
            writer.writerow(
                {
                    "rank": index,
                    "code": rec.get("code"),
                    "candidate_level": rec.get("candidate_level"),
                    "tcad_score": rec.get("tcad_score"),
                    "geometry_score": rec.get("geometry_score"),
                    "manufacturer": rec.get("manufacturer"),
                    "device_name": rec.get("device_name"),
                    "report_type": rec.get("report_type"),
                    "pixel_pitch_um": geom.get("pixel_pitch_um"),
                    "active_si_thickness_um": geom.get("active_si_thickness_um"),
                    "dti_type": geom.get("dti_type"),
                    "dti_depth_um": geom.get("dti_depth_um"),
                    "dti_width_nm": geom.get("dti_width_nm"),
                    "transfer_gate_type": geom.get("transfer_gate_type"),
                    "pixel_architecture": geom.get("pixel_architecture"),
                    "doping_table_count": rec.get("doping_table_count"),
                    "doping_row_count": rec.get("doping_row_count"),
                    "region_depth_count": len(rec.get("region_depths", [])),
                    "html_snippet_count": len(rec.get("html_snippets", [])),
                    "pdf_snippet_count": len(rec.get("pdf_snippets", [])),
                    "source_html": "; ".join(rec.get("source_html", [])),
                    "source_pdf": "; ".join(rec.get("source_pdf", [])),
                }
            )


def html_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_html(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    rows: list[str] = []
    for rec in records:
        geom = rec.get("key_geometry", {})
        evidence_bits = []
        if rec.get("doping_row_count"):
            evidence_bits.append(f"{rec['doping_row_count']} doping rows")
        if rec.get("region_depths"):
            evidence_bits.append(f"{len(rec['region_depths'])} region/depth hits")
        if rec.get("html_snippets"):
            evidence_bits.append(f"{len(rec['html_snippets'])} HTML snippets")
        if rec.get("pdf_snippets"):
            evidence_bits.append(f"{len(rec['pdf_snippets'])} PDF snippets")
        first_table = rec.get("doping_tables", [{}])[0] if rec.get("doping_tables") else {}
        first_rows = first_table.get("rows", [])[:4]
        doping_preview = "<br>".join(
            html_escape(
                f"{row.get('feature', row.get('raw', [''])[0] if row.get('raw') else '')}: "
                f"{row.get('dopant', '')}; {row.get('concentration_text', '')}; depth {row.get('depth_text', '')}"
            )
            for row in first_rows
        )
        if not doping_preview:
            doping_preview = html_escape((rec.get("html_snippets") or rec.get("pdf_snippets") or [""])[0])
        source_link = rec.get("source_html", [""])[0] if rec.get("source_html") else ""
        source_cell = html_escape(source_link)
        if source_link:
            source_cell = f'<a href="{html_escape(source_link)}">{html_escape(Path(source_link).name)}</a>'
        rows.append(
            "<tr>"
            f"<td>{html_escape(rec.get('code'))}</td>"
            f"<td>{html_escape(rec.get('candidate_level'))}<br><small>score {html_escape(rec.get('tcad_score'))}</small></td>"
            f"<td>{html_escape(rec.get('manufacturer'))}<br><small>{html_escape(rec.get('device_name'))}</small></td>"
            f"<td>pitch {html_escape(geom.get('pixel_pitch_um'))} um<br>"
            f"active Si {html_escape(geom.get('active_si_thickness_um'))} um<br>"
            f"{html_escape(geom.get('dti_type'))} {html_escape(geom.get('dti_depth_um'))} um<br>"
            f"{html_escape(geom.get('transfer_gate_type'))}</td>"
            f"<td>{html_escape(', '.join(evidence_bits))}</td>"
            f"<td>{doping_preview}</td>"
            f"<td>{source_cell}</td>"
            "</tr>"
        )
    summary_items = "".join(
        f"<li><strong>{html_escape(k)}</strong>: {html_escape(v)}</li>" for k, v in summary.items()
    )
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TCAD Candidate Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }
    h1 { font-size: 24px; margin-bottom: 8px; }
    p, li { line-height: 1.45; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #d6dbe1; padding: 8px; vertical-align: top; }
    th { background: #edf2f7; text-align: left; position: sticky; top: 0; }
    tr:nth-child(even) { background: #f8fafc; }
    small { color: #667085; }
    code { background: #eef2f6; padding: 1px 4px; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>TCAD Candidate Report</h1>
  <p>Local-only audit of image-sensor records. Proxy generated profiles are not measured process decks.</p>
  <ul>
"""
        + summary_items
        + """
  </ul>
  <table>
    <thead>
      <tr>
        <th>Code</th>
        <th>Level</th>
        <th>Device</th>
        <th>Geometry</th>
        <th>Evidence</th>
        <th>Doping / Snippet Preview</th>
        <th>Source</th>
      </tr>
    </thead>
    <tbody>
"""
        + "\n".join(rows)
        + """
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-pdf", action="store_true", help="scan source PDFs with pypdf when available")
    parser.add_argument("--max-pdf-pages-per-file", type=int, default=None)
    args = parser.parse_args()

    catalog = read_json(args.catalog)
    records = catalog.get("records", [])
    base_dir = args.catalog.parent
    report_records = [
        build_record_report(
            record=record,
            base_dir=base_dir,
            include_pdf=args.include_pdf,
            max_pdf_pages=args.max_pdf_pages_per_file,
        )
        for record in records
    ]
    report_records.sort(key=lambda item: (item.get("tcad_score", 0), item.get("doping_row_count", 0)), reverse=True)

    level_counts: dict[str, int] = {}
    for item in report_records:
        level = item.get("candidate_level", "unknown")
        level_counts[level] = level_counts.get(level, 0) + 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(report_records),
        "include_pdf": args.include_pdf,
        "pdf_dependency": "available" if PdfReader is not None else f"unavailable: {PYPDF_IMPORT_ERROR}",
        "pdf_pages_read": sum(item.get("pdf_pages_read", 0) for item in report_records),
        "records_with_measured_doping_table": level_counts.get("measured_doping_table", 0),
        "records_with_region_depths_and_process_evidence": level_counts.get("region_depths_and_process_evidence", 0),
        "records_with_geometry_plus_process_evidence": level_counts.get("geometry_plus_process_evidence", 0),
        "records_geometry_only": level_counts.get("geometry_only", 0),
        "records_insufficient_for_tcad": level_counts.get("insufficient_for_tcad", 0),
    }

    payload = {
        "schema": "tcad_candidate_report_v1",
        "summary": summary,
        "level_counts": level_counts,
        "records": report_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "tcad_candidate_report.json", payload)
    write_csv(args.output_dir / "tcad_candidate_report.csv", report_records)
    render_html(args.output_dir / "tcad_candidate_report.html", summary, report_records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir / 'tcad_candidate_report.json'}")
    print(f"Wrote {args.output_dir / 'tcad_candidate_report.csv'}")
    print(f"Wrote {args.output_dir / 'tcad_candidate_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
