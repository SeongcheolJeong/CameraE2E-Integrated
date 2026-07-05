#!/usr/bin/env python3
"""Build a local optical/CFA/QE evidence database for image sensor records.

This is intentionally separate from the electrical TCAD structure database.
Most TechInsights image-sensor reports expose CFA/OCL geometry and qualitative
optical-design evidence, while QE is usually a product claim or descriptive
sentence rather than a measured curve. The output keeps extracted, inferred,
and unavailable values visibly separate.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "image_sensor_db" / "sensor_catalog.json"
DEFAULT_OUTPUT_DIR = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_PROXY_NK_LIBRARY = ROOT / "cfa_proxy_nk_library.json"


NUMERIC_UNITS = {
    "pixel_pitch_um": "um",
    "optical_stack_height_um": "um",
    "cfa_thickness_um": "um",
    "cfa_thickness_min_um": "um",
    "cfa_thickness_max_um": "um",
    "color_filter_pitch_um": "um",
    "ocl_pitch_um": "um",
    "grid_pitch_um": "um",
    "active_si_thickness_um": "um",
}

OPTICAL_SPEC_FIELDS = [
    "illumination",
    "sensor_modality",
    "is_nir",
    "pixel_pitch_um",
    "optical_format",
    "optical_stack_height_um",
    "cfa_pattern",
    "cfa_thickness_um",
    "cfa_thickness_min_um",
    "cfa_thickness_max_um",
    "color_filter_pitch_um",
    "microlens_type",
    "ocl_pitch_um",
    "grid_material",
    "grid_pitch_um",
    "active_si_thickness_um",
]

SOURCE_TEXT_CATEGORIES = ("details", "html", "content")

QE_TERM_RX = re.compile(
    r"\b(?:quantum efficiency|external quantum efficiency|optical quantum efficiency|\bQE\b|"
    r"photon detection efficiency|\bPDE\b|spectral response|spectral sensitivity|responsivity)\b",
    re.I,
)
QE_POINT_PERCENT_AT_WAVELENGTH_RX = re.compile(
    r"(?P<qe>\d+(?:\.\d+)?)\s*%\s*(?:at|@)\s*"
    r"(?P<wavelength>\d+(?:\.\d+)?)\s*(?P<unit>nm|um|\u00b5m|micron|microns)\b",
    re.I,
)
QE_POINT_WAVELENGTH_AT_PERCENT_RX = re.compile(
    r"(?P<wavelength>\d+(?:\.\d+)?)\s*(?P<unit>nm|um|\u00b5m|micron|microns)\b"
    r".{0,80}?(?P<qe>\d+(?:\.\d+)?)\s*%",
    re.I,
)
SPECTRAL_RANGE_RX = re.compile(
    r"(?:covering|covers|range|wavelengths?)[^.;:]{0,80}?"
    r"(?P<low>\d+(?:\.\d+)?)\s*(?P<low_unit>nm|um|\u00b5m|micron|microns)\s*"
    r"(?:to|-|through)\s*"
    r"(?P<high>\d+(?:\.\d+)?)\s*(?P<high_unit>nm|um|\u00b5m|micron|microns)",
    re.I,
)
FILTER_WAVELENGTH_RX = re.compile(
    r"(?P<filter_count>\d+)\s+types?\s+of\s+filters?.{0,80}?"
    r"(?P<wavelength_count>\d+)\s+wavelengths?",
    re.I,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def safe_slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "unknown"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def field(
    value: Any,
    *,
    unit: str | None = None,
    source_kind: str,
    confidence: float,
    method: str,
    evidence: str | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    out = {
        "value": value,
        "source_kind": source_kind,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        "method": method,
    }
    if unit:
        out["unit"] = unit
    if evidence:
        out["evidence"] = evidence
    if source_file:
        out["source_file"] = source_file
    return out


def field_value(item: dict[str, Any] | None, default: Any = None) -> Any:
    if not isinstance(item, dict):
        return default
    value = item.get("value", default)
    return default if value is None else value


def load_proxy_nk_library(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"proxy n,k library not found: {path}")
    return read_json(path)


def evidence_entries(record: dict[str, Any], spec_name: str, *, limit: int = 4) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in record.get("derived_evidence", {}).get(spec_name, [])[:limit]:
        text = normalize_space(item.get("text"))
        if not text:
            continue
        out.append(
            {
                "source": normalize_space(item.get("source")),
                "source_file": normalize_space(item.get("source_file")),
                "text": text[:420],
            }
        )
    return out


def evidence_summary(record: dict[str, Any], spec_name: str) -> tuple[str | None, str | None]:
    entries = evidence_entries(record, spec_name, limit=1)
    if not entries:
        return None, None
    entry = entries[0]
    source = entry.get("source")
    text = entry.get("text")
    evidence = f"{source}: {text[:220]}" if source and text else text
    return evidence, entry.get("source_file") or None


def spec_field(record: dict[str, Any], name: str, *, confidence: float = 0.88) -> dict[str, Any]:
    specs = record.get("derived_specs", {})
    value = specs.get(name)
    unit = NUMERIC_UNITS.get(name)
    if value in (None, "", []):
        return field(
            None,
            unit=unit,
            source_kind="unavailable",
            confidence=0.0,
            method=f"sensor_catalog.derived_specs.{name} missing",
        )
    evidence, source_file = evidence_summary(record, name)
    return field(
        value,
        unit=unit,
        source_kind="extracted",
        confidence=confidence,
        method=f"sensor_catalog.derived_specs.{name}",
        evidence=evidence,
        source_file=source_file,
    )


def representative_cfa_thickness(record: dict[str, Any]) -> dict[str, Any]:
    direct = spec_field(record, "cfa_thickness_um", confidence=0.84)
    if field_value(direct) is not None:
        return direct
    low = as_float(record.get("derived_specs", {}).get("cfa_thickness_min_um"))
    high = as_float(record.get("derived_specs", {}).get("cfa_thickness_max_um"))
    if low is not None and high is not None:
        evidence, source_file = evidence_summary(record, "cfa_thickness_min_um")
        return field(
            round((low + high) / 2.0, 6),
            unit="um",
            source_kind="derived_from_extracted_range",
            confidence=0.76,
            method="midpoint(cfa_thickness_min_um,cfa_thickness_max_um)",
            evidence=evidence,
            source_file=source_file,
        )
    return direct


def inferred_pitch_field(
    record: dict[str, Any],
    direct_name: str,
    *,
    target_name: str,
    confidence: float,
    multiplier: float = 1.0,
    reason: str,
) -> dict[str, Any]:
    direct = spec_field(record, direct_name, confidence=0.84)
    if field_value(direct) is not None:
        return direct
    pitch = as_float(record.get("derived_specs", {}).get("pixel_pitch_um"))
    if pitch is None:
        return direct
    return field(
        round(pitch * multiplier, 6),
        unit="um",
        source_kind="inferred_rule",
        confidence=confidence,
        method=f"{target_name} inferred from pixel_pitch_um; {reason}",
    )


def cfa_filter_pitch(record: dict[str, Any]) -> dict[str, Any]:
    pattern = normalize_space(record.get("derived_specs", {}).get("cfa_pattern")).lower()
    multiplier = 2.0 if pattern in {"quad_bayer", "tetracell_bayer"} else 1.0
    reason = "2x pitch for grouped CFA pattern" if multiplier == 2.0 else "1x pitch fallback"
    return inferred_pitch_field(
        record,
        "color_filter_pitch_um",
        target_name="color_filter_pitch_um",
        confidence=0.42 if multiplier == 2.0 else 0.38,
        multiplier=multiplier,
        reason=reason,
    )


def ocl_pitch(record: dict[str, Any]) -> dict[str, Any]:
    return inferred_pitch_field(
        record,
        "ocl_pitch_um",
        target_name="ocl_pitch_um",
        confidence=0.40,
        multiplier=1.0,
        reason="OCL pitch usually follows pixel pitch when not reported",
    )


def grid_pitch(record: dict[str, Any]) -> dict[str, Any]:
    return inferred_pitch_field(
        record,
        "grid_pitch_um",
        target_name="grid_pitch_um",
        confidence=0.35,
        multiplier=1.0,
        reason="grid pitch approximated as pixel pitch",
    )


def proxy_channels_for_pattern(pattern: Any) -> tuple[str, list[str]]:
    normalized = normalize_space(pattern).lower()
    if not normalized:
        return "unknown_cfa_pattern", []
    if normalized == "monochrome":
        return "clear_or_monochrome", ["clear"]
    if normalized in {"bayer", "rggb_bayer", "quad_bayer", "tetracell_bayer"}:
        return "rgb_color_filter_proxy", ["red", "green", "blue"]
    if "ryyb" in normalized or "yellow" in normalized:
        return "ryyb_color_filter_proxy", ["red", "yellow", "blue"]
    return "unsupported_cfa_pattern", []


def proxy_thickness_field(cfa: dict[str, Any], library: dict[str, Any]) -> dict[str, Any]:
    thickness = as_float(field_value(cfa["representative_thickness_um"]))
    if thickness is not None:
        source = cfa["representative_thickness_um"].get("source_kind", "extracted")
        return field(
            round(thickness, 6),
            unit="um",
            source_kind=source,
            confidence=cfa["representative_thickness_um"].get("confidence", 0.75),
            method="sensor-specific CFA thickness for proxy n,k transmission scaling",
            evidence=cfa["representative_thickness_um"].get("evidence"),
            source_file=cfa["representative_thickness_um"].get("source_file"),
        )
    default_thickness = as_float(library.get("reference_thickness_um")) or 0.65
    return field(
        default_thickness,
        unit="um",
        source_kind="inferred_proxy_default",
        confidence=0.30,
        method="proxy library reference_thickness_um used because sensor-specific CFA thickness is unavailable",
    )


def rescaled_proxy_nk_data(channel_data: list[dict[str, Any]], thickness_um: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in channel_data:
        wavelength_nm = float(item["wavelength_nm"])
        wavelength_um = wavelength_nm / 1000.0
        k = max(0.0, float(item["k"]))
        transmission = math.exp(-4.0 * math.pi * k * thickness_um / wavelength_um)
        rows.append(
            {
                "wavelength_nm": round(wavelength_nm, 6),
                "n": round(float(item["n"]), 6),
                "k": round(k, 6),
                "transmission_absorption_only": round(max(0.0, min(1.0, transmission)), 6),
            }
        )
    return rows


def build_proxy_cfa_nk(optical: dict[str, Any], proxy_library: dict[str, Any] | None) -> dict[str, Any]:
    if not proxy_library:
        return {
            "enabled": False,
            "source_kind": "unavailable",
            "method": "proxy n,k library not provided",
            "channels": {},
        }
    pattern = field_value(optical["cfa"]["pattern"])
    applicability, channel_names = proxy_channels_for_pattern(pattern)
    thickness = proxy_thickness_field(optical["cfa"], proxy_library)
    thickness_value = float(field_value(thickness, proxy_library.get("reference_thickness_um", 0.65)))
    channels: dict[str, Any] = {}
    library_channels = proxy_library.get("channels", {})
    for channel in channel_names:
        source = library_channels.get(channel)
        if not source:
            continue
        channels[channel] = {
            "description": source.get("description", ""),
            "source_kind": proxy_library.get("source_kind", "inferred_proxy_nk"),
            "confidence": 0.38 if thickness.get("source_kind") == "inferred_proxy_default" else 0.48,
            "method": "fixed common CFA n,k proxy with sensor-specific thickness scaling",
            "data": rescaled_proxy_nk_data(source.get("data", []), thickness_value),
        }
    return {
        "enabled": bool(channels),
        "library_id": proxy_library.get("library_id"),
        "source_kind": proxy_library.get("source_kind", "inferred_proxy_nk"),
        "applicability": applicability,
        "reference_thickness_um": proxy_library.get("reference_thickness_um"),
        "thickness_um": thickness,
        "channels": channels,
        "model_limitations": [
            "Absorption-only CFA layer proxy; does not include full optical-stack interference, microlens focusing, grid diffraction, or silicon absorption.",
            "Common R/G/B n,k proxy, not product-specific measured optical constants.",
        ],
    }


def strip_markup(raw: str) -> str:
    raw = re.sub(r"(?is)<script\b.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style\b.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    return normalize_space(html.unescape(raw))


def iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(iter_strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(iter_strings(child))
        return out
    if value is None:
        return []
    return [str(value)]


def read_source_text(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
            raw = " ".join(iter_strings(data))
        except json.JSONDecodeError:
            pass
    return strip_markup(raw)


def source_texts(record: dict[str, Any]) -> list[tuple[str, str]]:
    source_files = record.get("source_files", {})
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for category in SOURCE_TEXT_CATEGORIES:
        for path in source_files.get(category, []) or []:
            if path in seen:
                continue
            seen.add(path)
            text = read_source_text(path)
            if text:
                out.append((path, text))
    fallback = strip_markup(json.dumps(record, ensure_ascii=False))
    if fallback:
        out.append(("sensor_catalog_record", fallback))
    return out


def context_window(text: str, start: int, end: int, *, radius: int = 360) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return normalize_space(text[left:right])


def unit_to_nm(value: float, unit: str) -> float:
    unit = unit.lower().replace("\u00b5", "u")
    if unit in {"um", "micron", "microns"}:
        return value * 1000.0
    return value


def extract_qe_points_from_context(context: str, source_file: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for rx in (QE_POINT_PERCENT_AT_WAVELENGTH_RX, QE_POINT_WAVELENGTH_AT_PERCENT_RX):
        for match in rx.finditer(context):
            if rx is QE_POINT_WAVELENGTH_AT_PERCENT_RX and not re.search(
                r"\b(?:qe|quantum|efficiency|responsivity)\b", match.group(0), re.I
            ):
                continue
            qe = float(match.group("qe"))
            wavelength_nm = unit_to_nm(float(match.group("wavelength")), match.group("unit"))
            if not (0.0 < qe <= 100.0 and 150.0 <= wavelength_nm <= 20000.0):
                continue
            points.append(
                {
                    "wavelength_nm": round(wavelength_nm, 3),
                    "qe_percent": round(qe, 3),
                    "source_kind": "reported_performance_point",
                    "confidence": 0.78,
                    "method": "regex near QE/quantum-efficiency context",
                    "source_file": source_file,
                    "context": context[:620],
                }
            )
    return points


def extract_spectral_ranges(text: str, source_file: str, *, limit: int = 8) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for match in SPECTRAL_RANGE_RX.finditer(text):
        low = unit_to_nm(float(match.group("low")), match.group("low_unit"))
        high = unit_to_nm(float(match.group("high")), match.group("high_unit"))
        if low > high:
            low, high = high, low
        key = (round(low, 3), round(high, 3), source_file)
        if key in seen or not (100.0 <= low <= 20000.0 and low < high <= 20000.0):
            continue
        seen.add(key)
        ranges.append(
            {
                "low_nm": round(low, 3),
                "high_nm": round(high, 3),
                "source_kind": "reported_spectral_range",
                "confidence": 0.72,
                "source_file": source_file,
                "context": context_window(text, match.start(), match.end(), radius=180)[:460],
            }
        )
        if len(ranges) >= limit:
            break
    return ranges


def extract_filter_wavelength_counts(text: str, source_file: str, *, limit: int = 6) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for match in FILTER_WAVELENGTH_RX.finditer(text):
        filter_count = int(match.group("filter_count"))
        wavelength_count = int(match.group("wavelength_count"))
        key = (filter_count, wavelength_count, source_file)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "filter_count": filter_count,
                "wavelength_count": wavelength_count,
                "source_kind": "reported_filter_set",
                "confidence": 0.74,
                "source_file": source_file,
                "context": context_window(text, match.start(), match.end(), radius=180)[:460],
            }
        )
        if len(hits) >= limit:
            break
    return hits


def extract_qe_and_spectral(record: dict[str, Any]) -> dict[str, Any]:
    quantitative: list[dict[str, Any]] = []
    qualitative: list[dict[str, Any]] = []
    spectral_ranges: list[dict[str, Any]] = []
    filter_sets: list[dict[str, Any]] = []
    seen_points: set[tuple[float, float]] = set()
    seen_mentions: set[str] = set()

    for source_file, text in source_texts(record):
        spectral_ranges.extend(extract_spectral_ranges(text, source_file))
        filter_sets.extend(extract_filter_wavelength_counts(text, source_file))
        for match in QE_TERM_RX.finditer(text):
            context = context_window(text, match.start(), match.end())
            mention_key = context[:240].lower()
            if mention_key not in seen_mentions and len(qualitative) < 12:
                seen_mentions.add(mention_key)
                qualitative.append(
                    {
                        "term": match.group(0),
                        "source_kind": "text_mention",
                        "confidence": 0.62,
                        "source_file": source_file,
                        "context": context[:620],
                    }
                )
            for point in extract_qe_points_from_context(context, source_file):
                key = (point["wavelength_nm"], point["qe_percent"])
                if key in seen_points:
                    continue
                seen_points.add(key)
                quantitative.append(point)

    if quantitative:
        status = "reported_qe_points"
    elif qualitative:
        status = "qualitative_only"
    else:
        status = "not_found"

    spectral_ranges = dedupe_dicts(spectral_ranges, ("low_nm", "high_nm"))
    filter_sets = dedupe_dicts(filter_sets, ("filter_count", "wavelength_count"))

    return {
        "measurement_data_status": status,
        "quantitative_points": sorted(quantitative, key=lambda item: item["wavelength_nm"]),
        "qualitative_mentions": qualitative,
        "spectral_ranges": spectral_ranges,
        "filter_sets": filter_sets,
    }


def dedupe_dicts(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        key = tuple(item.get(name) for name in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def collect_evidence_bundle(record: dict[str, Any], fields: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for name in fields:
        for entry in evidence_entries(record, name, limit=3):
            key = (entry.get("source_file", ""), entry.get("text", "")[:120])
            if key in seen:
                continue
            seen.add(key)
            entry = dict(entry)
            entry["field"] = name
            out.append(entry)
    return out


def optical_readiness(optical: dict[str, Any], qe: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    checks = [
        ("cfa_pattern", optical["cfa"]["pattern"], 12),
        ("microlens_type", optical["microlens"]["type"], 10),
        ("optical_stack_height", optical["geometry"]["optical_stack_height_um"], 16),
        ("cfa_thickness", optical["cfa"]["representative_thickness_um"], 16),
        ("color_filter_pitch", optical["cfa"]["color_filter_pitch_um"], 10),
        ("ocl_pitch", optical["microlens"]["ocl_pitch_um"], 10),
        ("grid_pitch", optical["grid"]["pitch_um"], 8),
        ("grid_material", optical["grid"]["material"], 6),
    ]
    for name, item, weight in checks:
        if field_value(item) is not None:
            score += weight
            reasons.append(name)

    if qe["quantitative_points"]:
        score += 12
        reasons.append("qe_points")
    elif qe["qualitative_mentions"]:
        score += 4
        reasons.append("qe_qualitative")

    if score >= 72:
        level = "optical_ready_high"
    elif score >= 48:
        level = "optical_ready_medium"
    elif score >= 24:
        level = "optical_context_only"
    else:
        level = "insufficient_optical_evidence"
    return {"level": level, "score": score, "reasons": reasons}


def source_context(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata", {})
    html_files = record.get("source_files", {}).get("html", []) or []
    pdf_files = record.get("source_files", {}).get("pdf", []) or []
    return {
        "code": record.get("code"),
        "manufacturer": meta.get("manufacturer"),
        "device_name": meta.get("device_name"),
        "title": meta.get("title") or record.get("content_title"),
        "report_type": meta.get("report_type"),
        "release_date": meta.get("release_date"),
        "html_files": html_files,
        "pdf_files": pdf_files,
    }


def build_model(record: dict[str, Any], proxy_library: dict[str, Any] | None) -> dict[str, Any]:
    qe = extract_qe_and_spectral(record)
    optical = {
        "geometry": {
            "pixel_pitch_um": spec_field(record, "pixel_pitch_um", confidence=0.90),
            "optical_format": spec_field(record, "optical_format", confidence=0.74),
            "optical_stack_height_um": spec_field(record, "optical_stack_height_um", confidence=0.86),
            "active_si_thickness_um": spec_field(record, "active_si_thickness_um", confidence=0.84),
        },
        "cfa": {
            "pattern": spec_field(record, "cfa_pattern", confidence=0.82),
            "representative_thickness_um": representative_cfa_thickness(record),
            "thickness_min_um": spec_field(record, "cfa_thickness_min_um", confidence=0.82),
            "thickness_max_um": spec_field(record, "cfa_thickness_max_um", confidence=0.82),
            "color_filter_pitch_um": cfa_filter_pitch(record),
            "evidence": collect_evidence_bundle(
                record,
                [
                    "cfa_pattern",
                    "cfa_thickness_um",
                    "cfa_thickness_min_um",
                    "cfa_thickness_max_um",
                    "color_filter_pitch_um",
                ],
            ),
        },
        "microlens": {
            "type": spec_field(record, "microlens_type", confidence=0.82),
            "ocl_pitch_um": ocl_pitch(record),
            "evidence": collect_evidence_bundle(record, ["microlens_type", "ocl_pitch_um", "optical_stack_height_um"]),
        },
        "grid": {
            "material": spec_field(record, "grid_material", confidence=0.76),
            "pitch_um": grid_pitch(record),
            "evidence": collect_evidence_bundle(record, ["grid_material", "grid_pitch_um"]),
        },
        "sensor_context": {
            "illumination": spec_field(record, "illumination", confidence=0.78),
            "sensor_modality": spec_field(record, "sensor_modality", confidence=0.78),
            "is_nir": spec_field(record, "is_nir", confidence=0.72),
        },
    }
    proxy_cfa_nk = build_proxy_cfa_nk(optical, proxy_library)
    readiness = optical_readiness(optical, qe)
    return {
        "schema": "image_sensor_optical_qe_model_v1",
        "code": record.get("code"),
        "source": source_context(record),
        "optical": optical,
        "cfa_proxy_nk": proxy_cfa_nk,
        "qe": qe,
        "readiness": readiness,
        "notes": [
            "QE quantitative_points are reported text points, not reconstructed curves.",
            "inferred_rule values are setup defaults and must not be treated as measurements.",
            "cfa_proxy_nk is a common color-resist n,k proxy for setup/sensitivity work, not measured product-specific optical constants.",
        ],
    }


def qe_summary(points: list[dict[str, Any]]) -> str:
    if not points:
        return ""
    return "; ".join(f"{p['wavelength_nm']:g}nm:{p['qe_percent']:g}%" for p in points[:8])


def proxy_channel_transmission(proxy: dict[str, Any], channel: str, wavelength_nm: int) -> str:
    data = proxy.get("channels", {}).get(channel, {}).get("data", [])
    for item in data:
        if int(round(float(item.get("wavelength_nm", -1)))) == wavelength_nm:
            return f"{float(item.get('transmission_absorption_only', 0.0)):.4f}"
    return ""


def row_for_model(model: dict[str, Any], rel_model_path: str) -> dict[str, Any]:
    source = model["source"]
    optical = model["optical"]
    cfa = optical["cfa"]
    microlens = optical["microlens"]
    grid = optical["grid"]
    geometry = optical["geometry"]
    qe = model["qe"]
    proxy = model.get("cfa_proxy_nk", {})
    html_files = source.get("html_files") or []
    return {
        "code": model["code"],
        "manufacturer": source.get("manufacturer") or "",
        "device_name": source.get("device_name") or "",
        "report_type": source.get("report_type") or "",
        "optical_readiness": model["readiness"]["level"],
        "optical_score": model["readiness"]["score"],
        "pixel_pitch_um": field_value(geometry["pixel_pitch_um"], ""),
        "optical_stack_height_um": field_value(geometry["optical_stack_height_um"], ""),
        "cfa_pattern": field_value(cfa["pattern"], ""),
        "cfa_thickness_um": field_value(cfa["representative_thickness_um"], ""),
        "cfa_thickness_min_um": field_value(cfa["thickness_min_um"], ""),
        "cfa_thickness_max_um": field_value(cfa["thickness_max_um"], ""),
        "color_filter_pitch_um": field_value(cfa["color_filter_pitch_um"], ""),
        "color_filter_pitch_source": cfa["color_filter_pitch_um"].get("source_kind"),
        "microlens_type": field_value(microlens["type"], ""),
        "ocl_pitch_um": field_value(microlens["ocl_pitch_um"], ""),
        "ocl_pitch_source": microlens["ocl_pitch_um"].get("source_kind"),
        "grid_material": field_value(grid["material"], ""),
        "grid_pitch_um": field_value(grid["pitch_um"], ""),
        "grid_pitch_source": grid["pitch_um"].get("source_kind"),
        "cfa_proxy_nk_enabled": proxy.get("enabled", False),
        "cfa_proxy_applicability": proxy.get("applicability", ""),
        "cfa_proxy_library_id": proxy.get("library_id", ""),
        "cfa_proxy_thickness_um": field_value(proxy.get("thickness_um"), ""),
        "cfa_proxy_thickness_source": proxy.get("thickness_um", {}).get("source_kind", ""),
        "proxy_t_blue_450": proxy_channel_transmission(proxy, "blue", 450),
        "proxy_t_green_550": proxy_channel_transmission(proxy, "green", 550),
        "proxy_t_red_650": proxy_channel_transmission(proxy, "red", 650),
        "proxy_t_clear_550": proxy_channel_transmission(proxy, "clear", 550),
        "qe_status": qe["measurement_data_status"],
        "qe_point_count": len(qe["quantitative_points"]),
        "qe_points": qe_summary(qe["quantitative_points"]),
        "qe_qualitative_count": len(qe["qualitative_mentions"]),
        "spectral_range_count": len(qe["spectral_ranges"]),
        "filter_set_count": len(qe["filter_sets"]),
        "source_html": html_files[0] if html_files else "",
        "model_json": rel_model_path,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    summary_items = "".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in summary.items()
    )
    table_rows = []
    for row in rows:
        model_link = html.escape(row["model_json"])
        source_html = row.get("source_html") or ""
        source_link = (
            f'<a href="file://{html.escape(source_html)}">HTML</a>'
            if source_html
            else ""
        )
        table_rows.append(
            "<tr>"
            f"<td><a href=\"{model_link}\">{html.escape(str(row['code']))}</a></td>"
            f"<td>{html.escape(str(row['manufacturer']))}<br><small>{html.escape(str(row['device_name']))}</small></td>"
            f"<td>{html.escape(str(row['optical_readiness']))}<br><small>score {html.escape(str(row['optical_score']))}</small></td>"
            f"<td>{html.escape(str(row['cfa_pattern']))}<br><small>{html.escape(str(row['cfa_thickness_um']))} um</small></td>"
            f"<td>{html.escape(str(row['microlens_type']))}<br><small>OCL {html.escape(str(row['ocl_pitch_um']))} um</small></td>"
            f"<td>{html.escape(str(row['optical_stack_height_um']))}</td>"
            f"<td>{html.escape(str(row['grid_material']))}<br><small>{html.escape(str(row['grid_pitch_um']))} um</small></td>"
            f"<td>{html.escape(str(row['cfa_proxy_applicability']))}<br><small>B450 {html.escape(str(row['proxy_t_blue_450']))} / G550 {html.escape(str(row['proxy_t_green_550']))} / R650 {html.escape(str(row['proxy_t_red_650']))}</small></td>"
            f"<td>{html.escape(str(row['qe_status']))}<br><small>{html.escape(str(row['qe_points']))}</small></td>"
            f"<td>{source_link}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Sensor Optical/CFA/QE DB</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ max-width: 1080px; color: #52616f; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px 24px; padding: 16px; border: 1px solid #d7dee8; border-radius: 8px; background: #f7f9fc; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #fff; z-index: 1; }}
    small {{ color: #66788a; }}
    a {{ color: #155e9f; }}
  </style>
</head>
<body>
  <h1>Image Sensor Optical/CFA/QE DB</h1>
  <p class="note">Extracted optical-stack, CFA, microlens, grid, spectral, and QE evidence from the local image-sensor catalog. Reported QE points are text claims, not reconstructed QE curves.</p>
  <ul class="summary">{summary_items}</ul>
  <table>
    <thead>
      <tr>
        <th>Code</th><th>Sensor</th><th>Readiness</th><th>CFA</th><th>Microlens</th><th>Optical stack um</th><th>Grid</th><th>Proxy CFA T</th><th>QE</th><th>Source</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    readiness = Counter(row["optical_readiness"] for row in rows)
    qe_status = Counter(row["qe_status"] for row in rows)
    cfa = Counter(row["cfa_pattern"] or "missing" for row in rows)
    color_filter_pitch_source = Counter(row["color_filter_pitch_source"] for row in rows)
    ocl_pitch_source = Counter(row["ocl_pitch_source"] for row in rows)
    grid_pitch_source = Counter(row["grid_pitch_source"] for row in rows)
    proxy_applicability = Counter(row["cfa_proxy_applicability"] or "missing" for row in rows)
    proxy_thickness_source = Counter(row["cfa_proxy_thickness_source"] or "missing" for row in rows)
    return {
        "schema": "image_sensor_optical_qe_db_v1",
        "record_count": len(rows),
        "readiness": dict(readiness),
        "qe_status": dict(qe_status),
        "cfa_pattern": dict(cfa),
        "color_filter_pitch_source": dict(color_filter_pitch_source),
        "ocl_pitch_source": dict(ocl_pitch_source),
        "grid_pitch_source": dict(grid_pitch_source),
        "cfa_proxy_applicability": dict(proxy_applicability),
        "cfa_proxy_thickness_source": dict(proxy_thickness_source),
        "cfa_proxy_nk_records": sum(1 for row in rows if str(row["cfa_proxy_nk_enabled"]) == "True"),
        "reported_qe_point_records": sum(1 for row in rows if int(row["qe_point_count"]) > 0),
        "reported_qe_points": sum(int(row["qe_point_count"]) for row in rows),
        "cfa_thickness_records": sum(1 for row in rows if row["cfa_thickness_um"] != ""),
        "optical_stack_height_records": sum(1 for row in rows if row["optical_stack_height_um"] != ""),
        "color_filter_pitch_records": sum(1 for row in rows if row["color_filter_pitch_um"] != ""),
        "ocl_pitch_records": sum(1 for row in rows if row["ocl_pitch_um"] != ""),
    }


def validate(output_dir: Path, rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if len(rows) != len(models):
        issues.append({"level": "error", "message": "row/model count mismatch"})
    for row in rows:
        model_path = output_dir / row["model_json"]
        if not model_path.exists():
            issues.append({"level": "error", "code": row["code"], "message": "model_json missing"})
        if row["qe_status"] == "reported_qe_points" and int(row["qe_point_count"]) == 0:
            issues.append({"level": "error", "code": row["code"], "message": "QE status/point count mismatch"})
        if row["cfa_thickness_min_um"] and row["cfa_thickness_max_um"]:
            if float(row["cfa_thickness_min_um"]) > float(row["cfa_thickness_max_um"]):
                issues.append({"level": "error", "code": row["code"], "message": "CFA thickness min > max"})
        for key in ("proxy_t_blue_450", "proxy_t_green_550", "proxy_t_red_650", "proxy_t_clear_550"):
            if not row[key]:
                continue
            value = float(row[key])
            if not 0.0 <= value <= 1.0:
                issues.append({"level": "error", "code": row["code"], "message": f"{key} outside 0..1"})
    for model in models:
        proxy = model.get("cfa_proxy_nk", {})
        for channel, channel_data in proxy.get("channels", {}).items():
            if not channel_data.get("data"):
                issues.append({"level": "error", "code": model["code"], "message": f"empty proxy channel {channel}"})
    return {
        "schema": "image_sensor_optical_qe_db_validation_v1",
        "pass": not any(issue["level"] == "error" for issue in issues),
        "model_count": len(models),
        "csv_row_count": len(rows),
        "issue_count": len(issues),
        "issues": issues,
    }


def build(catalog_path: Path, output_dir: Path, proxy_nk_library_path: Path | None) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    proxy_library = load_proxy_nk_library(proxy_nk_library_path)
    records = catalog["records"]
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        model = build_model(record, proxy_library)
        source = model["source"]
        slug = safe_slug(
            " ".join(
                str(part)
                for part in [model.get("code"), source.get("manufacturer"), source.get("device_name")]
                if part
            )
        )
        rel_model_path = f"models/{slug}.json"
        write_json(output_dir / rel_model_path, model)
        models.append(model)
        rows.append(row_for_model(model, rel_model_path))

    rows.sort(key=lambda item: (item["code"], item["manufacturer"], item["device_name"]))
    summary = build_summary(rows)
    manifest = {
        "schema": "image_sensor_optical_qe_db_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_catalog": str(catalog_path),
        "proxy_nk_library": str(proxy_nk_library_path) if proxy_nk_library_path else None,
        "output_dir": str(output_dir),
        "summary": summary,
        "files": {
            "index_html": str(output_dir / "index.html"),
            "summary_csv": str(output_dir / "optical_qe_summary.csv"),
            "manifest_json": str(output_dir / "manifest.json"),
            "validation_json": str(output_dir / "validation.json"),
            "proxy_nk_library": str(proxy_nk_library_path) if proxy_nk_library_path else "",
            "models_dir": str(models_dir),
        },
    }
    write_csv(output_dir / "optical_qe_summary.csv", rows)
    render_html(output_dir / "index.html", rows, summary)
    write_json(output_dir / "manifest.json", manifest)
    validation = validate(output_dir, rows, models)
    write_json(output_dir / "validation.json", validation)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--proxy-nk-library", type=Path, default=DEFAULT_PROXY_NK_LIBRARY)
    args = parser.parse_args()
    manifest = build(args.catalog, args.output_dir, args.proxy_nk_library)
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    print(json.dumps(manifest["files"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
