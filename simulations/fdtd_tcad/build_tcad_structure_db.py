#!/usr/bin/env python3
"""Build a TCAD-usable structure database for local image-sensor records.

The output is a per-sensor structure model, not a calibrated process deck. It
combines extracted geometry, measured SIMS seed availability, empirical
imputation, and conservative image-sensor TCAD defaults. Every important value
is wrapped with source kind, method, and confidence so measured and inferred
data remain visibly separate.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "image_sensor_db" / "sensor_catalog.json"
DEFAULT_CANDIDATE_REPORT = ROOT / "image_sensor_db" / "tcad_candidate_report.json"
DEFAULT_SIMS_MANIFEST = ROOT / "measured_profiles" / "techinsights_sims_seed" / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "image_sensor_db" / "tcad_structure_db"


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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    return None


def median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.median(clean))


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    index = int(clamp(round((len(clean) - 1) * q), 0, len(clean) - 1))
    return clean[index]


def field(
    value: Any,
    *,
    unit: str | None = None,
    source_kind: str,
    confidence: float,
    method: str,
    evidence: str | None = None,
) -> dict[str, Any]:
    out = {
        "value": value,
        "source_kind": source_kind,
        "confidence": round(clamp(float(confidence), 0.0, 1.0), 3),
        "method": method,
    }
    if unit:
        out["unit"] = unit
    if evidence:
        out["evidence"] = evidence
    return out


def field_value(item: dict[str, Any] | None, default: Any = None) -> Any:
    if not isinstance(item, dict):
        return default
    return item.get("value", default)


def combine_confidence(*items: dict[str, Any]) -> float:
    values = [float(item.get("confidence", 0.0)) for item in items if isinstance(item, dict)]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def pitch_bin(pitch_um: float | None) -> str:
    if pitch_um is None:
        return "unknown"
    if pitch_um <= 0.7:
        return "le_0p7"
    if pitch_um <= 1.0:
        return "0p7_to_1p0"
    if pitch_um <= 1.5:
        return "1p0_to_1p5"
    if pitch_um <= 2.5:
        return "1p5_to_2p5"
    if pitch_um <= 4.0:
        return "2p5_to_4p0"
    return "gt_4p0"


class EmpiricalStats:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.global_values: dict[str, list[float]] = defaultdict(list)
        self.by_manufacturer: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.by_dti_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.by_arch: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.by_pitch_bin: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.by_transfer: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.ratios: dict[str, list[float]] = defaultdict(list)
        self._build()

    def _build(self) -> None:
        numeric_fields = [
            "pixel_pitch_um",
            "active_si_thickness_um",
            "dti_depth_um",
            "dti_width_nm",
            "optical_stack_height_um",
            "cfa_thickness_um",
            "grid_pitch_um",
            "cis_process_nm",
            "pixel_beol_metal_pitch_nm",
        ]
        for record in self.records:
            specs = record.get("derived_specs", {})
            meta = record.get("metadata", {})
            manufacturer = normalize_space(meta.get("manufacturer")) or "unknown"
            dti_type = normalize_space(specs.get("dti_type")) or "unknown"
            arch = normalize_space(specs.get("pixel_architecture")) or "unknown"
            transfer = normalize_space(specs.get("transfer_gate_type")) or "unknown"
            pitch = as_float(specs.get("pixel_pitch_um"))
            pbin = pitch_bin(pitch)
            for name in numeric_fields:
                value = as_float(specs.get(name))
                if value is None:
                    continue
                self.global_values[name].append(value)
                self.by_manufacturer[manufacturer][name].append(value)
                self.by_dti_type[dti_type][name].append(value)
                self.by_arch[arch][name].append(value)
                self.by_transfer[transfer][name].append(value)
                self.by_pitch_bin[pbin][name].append(value)
            active = as_float(specs.get("active_si_thickness_um"))
            dti_depth = as_float(specs.get("dti_depth_um"))
            dti_width_nm = as_float(specs.get("dti_width_nm"))
            if pitch and active:
                self.ratios["active_over_pitch"].append(active / pitch)
            if pitch and dti_width_nm:
                self.ratios["dti_width_um_over_pitch"].append((dti_width_nm / 1000.0) / pitch)
            if active and dti_depth:
                self.ratios["dti_depth_over_active"].append(dti_depth / active)

    def lookup(
        self,
        field_name: str,
        *,
        manufacturer: str | None = None,
        dti_type: str | None = None,
        arch: str | None = None,
        transfer: str | None = None,
        pitch_um: float | None = None,
        min_count: int = 3,
    ) -> tuple[float | None, str, float]:
        candidates: list[tuple[str, list[float], float]] = []
        if manufacturer:
            candidates.append((f"manufacturer_median:{manufacturer}", self.by_manufacturer[manufacturer][field_name], 0.68))
        if dti_type:
            candidates.append((f"dti_type_median:{dti_type}", self.by_dti_type[dti_type][field_name], 0.62))
        if arch:
            candidates.append((f"pixel_architecture_median:{arch}", self.by_arch[arch][field_name], 0.58))
        if transfer:
            candidates.append((f"transfer_gate_median:{transfer}", self.by_transfer[transfer][field_name], 0.56))
        pbin = pitch_bin(pitch_um)
        candidates.append((f"pitch_bin_median:{pbin}", self.by_pitch_bin[pbin][field_name], 0.55))
        candidates.append(("global_median", self.global_values[field_name], 0.40))
        for method, values, confidence in candidates:
            if len(values) >= min_count:
                value = median(values)
                if value is not None:
                    return value, f"{method}; n={len(values)}", confidence
        return None, "no_empirical_value", 0.0

    def ratio(self, name: str, fallback: float) -> tuple[float, str]:
        value = median(self.ratios[name])
        if value is None:
            return fallback, f"fallback_ratio:{fallback}"
        p20 = percentile(self.ratios[name], 0.2)
        p80 = percentile(self.ratios[name], 0.8)
        return value, f"empirical_median_ratio:{name}; p20={p20:.3g}; p80={p80:.3g}; n={len(self.ratios[name])}"


def record_evidence(record: dict[str, Any], spec_name: str) -> str | None:
    evidence = record.get("derived_evidence", {}).get(spec_name, [])
    if not evidence:
        return None
    first = evidence[0]
    text = normalize_space(first.get("text"))
    source = normalize_space(first.get("source"))
    if text and source:
        return f"{source}: {text[:180]}"
    return text[:180] if text else None


def spec_field(
    record: dict[str, Any],
    name: str,
    *,
    unit: str | None = None,
    confidence: float = 0.92,
) -> dict[str, Any] | None:
    value = record.get("derived_specs", {}).get(name)
    if value in (None, "", []):
        return None
    return field(
        value,
        unit=unit,
        source_kind="extracted",
        confidence=confidence,
        method=f"sensor_catalog.derived_specs.{name}",
        evidence=record_evidence(record, name),
    )


def candidate_region_depth(candidate: dict[str, Any], names: set[str]) -> dict[str, Any] | None:
    for hit in candidate.get("region_depths", []):
        if hit.get("field") in names and is_number(hit.get("value")):
            return field(
                round(float(hit["value"]), 6),
                unit="um" if not str(hit["field"]).endswith("_nm") else "nm",
                source_kind="extracted_region_text",
                confidence=0.80,
                method=f"tcad_candidate_report.region_depths.{hit.get('field')}",
                evidence=normalize_space(hit.get("context", ""))[:220],
            )
    return None


def source_context(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata", {})
    return {
        "code": record.get("code"),
        "manufacturer": meta.get("manufacturer") or candidate.get("manufacturer"),
        "device_name": meta.get("device_name") or candidate.get("device_name"),
        "title": meta.get("title") or candidate.get("title"),
        "report_type": meta.get("report_type") or candidate.get("report_type"),
        "analysis_year": meta.get("analysis_year"),
        "release_date": meta.get("release_date") or candidate.get("release_date"),
        "source_html": record.get("source_files", {}).get("html", []) or candidate.get("source_html", []),
        "source_pdf": record.get("source_files", {}).get("pdf", []) or candidate.get("source_pdf", []),
    }


def infer_pixel_pitch(record: dict[str, Any], stats: EmpiricalStats) -> dict[str, Any]:
    direct = spec_field(record, "pixel_pitch_um", unit="um", confidence=0.96)
    if direct:
        return direct
    meta = record.get("metadata", {})
    value, method, confidence = stats.lookup(
        "pixel_pitch_um",
        manufacturer=meta.get("manufacturer"),
        arch=record.get("derived_specs", {}).get("pixel_architecture"),
        min_count=3,
    )
    if value is None:
        value = 1.4
        method = "fallback_common_cmos_pixel_pitch"
        confidence = 0.18
    return field(round(value, 6), unit="um", source_kind="inferred_empirical", confidence=confidence, method=method)


def infer_active_si(record: dict[str, Any], stats: EmpiricalStats, pitch: dict[str, Any]) -> dict[str, Any]:
    direct = spec_field(record, "active_si_thickness_um", unit="um", confidence=0.94)
    if direct:
        return direct
    specs = record.get("derived_specs", {})
    meta = record.get("metadata", {})
    pitch_value = as_float(field_value(pitch))
    value, method, confidence = stats.lookup(
        "active_si_thickness_um",
        manufacturer=meta.get("manufacturer"),
        dti_type=specs.get("dti_type"),
        arch=specs.get("pixel_architecture"),
        pitch_um=pitch_value,
    )
    if value is None and pitch_value:
        ratio, ratio_method = stats.ratio("active_over_pitch", 2.3)
        value = clamp(pitch_value * ratio, 2.0, 8.0)
        method = ratio_method
        confidence = 0.42
    if value is None:
        value = 3.2
        method = "fallback_median_active_si"
        confidence = 0.25
    return field(round(value, 6), unit="um", source_kind="inferred_empirical", confidence=confidence, method=method)


def infer_dti_type(record: dict[str, Any], pitch: dict[str, Any], active_si: dict[str, Any]) -> dict[str, Any]:
    direct = spec_field(record, "dti_type", confidence=0.90)
    if direct:
        return direct
    specs = record.get("derived_specs", {})
    meta = record.get("metadata", {})
    year = meta.get("analysis_year") or 0
    manufacturer = normalize_space(meta.get("manufacturer")).lower()
    pitch_value = as_float(field_value(pitch))
    has_dti = bool(specs.get("has_dti"))
    if has_dti:
        return field("dti_present_unknown_type", source_kind="inferred_rule", confidence=0.45, method="has_dti true but no extracted subtype")
    if pitch_value and pitch_value <= 1.2 and year >= 2018 and manufacturer in {"samsung", "sony", "sk hynix"}:
        return field("probable_full_depth_front_dti", source_kind="inferred_rule", confidence=0.38, method="small-pitch recent stacked/BI CIS heuristic")
    if pitch_value and pitch_value <= 2.5 and year >= 2018:
        return field("probable_back_or_front_dti", source_kind="inferred_rule", confidence=0.30, method="modern BI CIS isolation heuristic")
    return field("unknown", source_kind="unavailable", confidence=0.10, method="no DTI evidence")


def infer_dti_depth(record: dict[str, Any], stats: EmpiricalStats, active_si: dict[str, Any], dti_type: dict[str, Any], pitch: dict[str, Any]) -> dict[str, Any] | None:
    direct = spec_field(record, "dti_depth_um", unit="um", confidence=0.90)
    if direct:
        return direct
    dtype = normalize_space(field_value(dti_type, "")).lower()
    if dtype in {"unknown", ""}:
        return None
    active = as_float(field_value(active_si))
    pitch_value = as_float(field_value(pitch))
    specs = record.get("derived_specs", {})
    meta = record.get("metadata", {})
    value, method, confidence = stats.lookup(
        "dti_depth_um",
        manufacturer=meta.get("manufacturer"),
        dti_type=specs.get("dti_type") or dtype,
        arch=specs.get("pixel_architecture"),
        pitch_um=pitch_value,
        min_count=3,
    )
    if active and ("full_depth_front" in dtype or "probable_full_depth" in dtype):
        value = clamp(active * 0.98, 0.1, active * 1.05)
        method = "active_si_scaled_full_depth_dti; depth=0.98*active_si"
        confidence = 0.62 * float(active_si.get("confidence", 0.5))
    elif active and ("partial" in dtype or "back_dti" in dtype or "probable_back" in dtype):
        empirical = value if value is not None else active * 0.65
        value = clamp(empirical, 0.1, active)
        method = method if method != "no_empirical_value" else "partial/back_dti_empirical_or_0.65*active_si"
        confidence = max(confidence, 0.45)
    if value is None:
        return None
    return field(round(value, 6), unit="um", source_kind="inferred_empirical", confidence=confidence, method=method)


def infer_dti_width(record: dict[str, Any], stats: EmpiricalStats, pitch: dict[str, Any], dti_type: dict[str, Any]) -> dict[str, Any] | None:
    direct_nm = spec_field(record, "dti_width_nm", unit="nm", confidence=0.90)
    if direct_nm:
        return field(
            round(float(direct_nm["value"]) / 1000.0, 6),
            unit="um",
            source_kind="extracted",
            confidence=direct_nm["confidence"],
            method="converted from extracted dti_width_nm",
            evidence=direct_nm.get("evidence"),
        )
    dtype = normalize_space(field_value(dti_type, "")).lower()
    if dtype in {"unknown", ""}:
        return None
    pitch_value = as_float(field_value(pitch))
    specs = record.get("derived_specs", {})
    meta = record.get("metadata", {})
    value_nm, method, confidence = stats.lookup(
        "dti_width_nm",
        manufacturer=meta.get("manufacturer"),
        dti_type=specs.get("dti_type") or dtype,
        arch=specs.get("pixel_architecture"),
        pitch_um=pitch_value,
        min_count=3,
    )
    if value_nm is not None:
        value_um = value_nm / 1000.0
    elif pitch_value:
        ratio, ratio_method = stats.ratio("dti_width_um_over_pitch", 0.09)
        value_um = clamp(pitch_value * ratio, 0.035, min(0.24, pitch_value * 0.25))
        method = ratio_method
        confidence = 0.42
    else:
        value_um = 0.09
        method = "fallback_dti_width"
        confidence = 0.22
    return field(round(value_um, 6), unit="um", source_kind="inferred_empirical", confidence=confidence, method=method)


def infer_optical_stack(record: dict[str, Any], stats: EmpiricalStats, pitch: dict[str, Any]) -> dict[str, Any]:
    direct = spec_field(record, "optical_stack_height_um", unit="um", confidence=0.86)
    if direct:
        return direct
    pitch_value = as_float(field_value(pitch))
    meta = record.get("metadata", {})
    specs = record.get("derived_specs", {})
    value, method, confidence = stats.lookup(
        "optical_stack_height_um",
        manufacturer=meta.get("manufacturer"),
        arch=specs.get("pixel_architecture"),
        pitch_um=pitch_value,
    )
    if value is None and pitch_value:
        if pitch_value <= 0.8:
            value = 1.5
        elif pitch_value <= 1.5:
            value = 1.9
        elif pitch_value <= 2.5:
            value = 2.4
        else:
            value = clamp(pitch_value * 0.8, 2.5, 5.5)
        method = "pitch_class_optical_stack_rule"
        confidence = 0.35
    if value is None:
        value = 2.4
        method = "fallback_optical_stack"
        confidence = 0.22
    return field(round(value, 6), unit="um", source_kind="inferred_empirical", confidence=confidence, method=method)


def parse_sims_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in candidate.get("doping_tables", []):
        caption = normalize_space(table.get("caption"))
        if "periphery" in caption.lower():
            continue
        for row in table.get("rows", []):
            raw = [normalize_space(item).lower() for item in row.get("raw", [])]
            if ("p- and n-type" in caption.lower() or "doping levels" in caption.lower()) and raw and "pixel" not in raw[0]:
                continue
            row_copy = dict(row)
            row_copy["_source_caption"] = caption
            rows.append(row_copy)
    return rows


def row_role(row: dict[str, Any]) -> str:
    text = normalize_space(row.get("feature", "")).lower()
    if "pinning" in text:
        return "p_pinning"
    if "vss" in text or ("contact" in text and ("p+" in text or "p^+^" in text)):
        return "p_plus_contact"
    if "fd" in text and "s/d" in text:
        return "fd_sd"
    if "fd" in text:
        return "floating_diffusion"
    if "s/d" in text:
        return "source_drain"
    if "vtg" in text or "transfer" in text:
        return "transfer_gate"
    if "p-well" in text or "p well" in text:
        return "p_well"
    if "n-well" in text or "n well" in text:
        return "n_well"
    if "photocathode" in text or "cathode" in text:
        return "photocathode"
    if "channel" in text:
        return "channel"
    if "isolation" in text:
        return "pixel_isolation"
    if "dti" in text:
        return "dti_metadata"
    return "generic"


def role_depth_from_rows(candidate: dict[str, Any], role: str) -> dict[str, Any] | None:
    for row in parse_sims_rows(candidate):
        if row_role(row) == role and is_number(row.get("depth_um")):
            return field(
                round(float(row["depth_um"]), 6),
                unit="um",
                source_kind="measured_sims_table",
                confidence=0.86,
                method=f"SIMS/SPM table role {role}",
                evidence=f"{row.get('_source_caption', '')}: {row.get('feature')} depth {row.get('depth_text')}",
            )
    return None


def infer_photodiode(candidate: dict[str, Any], active_si: dict[str, Any]) -> dict[str, Any]:
    active = float(field_value(active_si, 3.0))
    photocathode = role_depth_from_rows(candidate, "photocathode")
    if not photocathode:
        photocathode = field(
            round(clamp(active * 0.88, 0.6, active), 6),
            unit="um",
            source_kind="inferred_rule",
            confidence=0.48 * float(active_si.get("confidence", 0.5)),
            method="photocathode_depth=0.88*active_si",
        )
    p_well = role_depth_from_rows(candidate, "p_well")
    if not p_well:
        p_well = field(
            round(clamp(0.16 * active, 0.35, 0.8), 6),
            unit="um",
            source_kind="inferred_rule",
            confidence=0.42,
            method="p_well_depth=clamp(0.16*active_si,0.35,0.8)",
        )
    pinning = role_depth_from_rows(candidate, "p_pinning")
    if not pinning:
        pinning = field(0.10, unit="um", source_kind="inferred_rule", confidence=0.35, method="typical CIS P+ pinning depth seed")
    return {
        "photocathode_depth_um": photocathode,
        "p_well_depth_um": p_well,
        "pinning_depth_um": pinning,
        "collection_model": field(
            "pinned_photodiode" if field_value(photocathode) else "generic_photodiode",
            source_kind="inferred_rule",
            confidence=0.55,
            method="image-sensor CIS default collection region model",
        ),
    }


def infer_transfer_gate(record: dict[str, Any], candidate: dict[str, Any], pitch: dict[str, Any], active_si: dict[str, Any]) -> dict[str, Any]:
    direct_type = spec_field(record, "transfer_gate_type", confidence=0.88)
    if not direct_type:
        direct_type = field("unknown", source_kind="unavailable", confidence=0.12, method="no transfer gate type evidence")
    pitch_value = float(field_value(pitch, 1.4))
    tg_depth = role_depth_from_rows(candidate, "transfer_gate")
    if not tg_depth:
        if field_value(direct_type) in {"vertical_transfer_gate", "dual_vertical_transfer_gate"}:
            tg_depth = field(0.44, unit="um", source_kind="inferred_rule", confidence=0.48, method="VTG/D-VTG median depth from measured SIMS candidates")
        else:
            tg_depth = field(0.0, unit="um", source_kind="inferred_rule", confidence=0.25, method="planar/unknown TG depth placeholder")
    gate_oxide = candidate_region_depth(candidate, {"gate_dielectric_nm", "gate_oxide_nm"})
    if not gate_oxide:
        oxide_nm = 5.5 if field_value(direct_type) in {"vertical_transfer_gate", "dual_vertical_transfer_gate"} else 4.5
        gate_oxide = field(oxide_nm, unit="nm", source_kind="inferred_rule", confidence=0.35, method="typical CIS TG nitrided oxide seed")
    tg_half_width = clamp(pitch_value * 0.18, 0.04, min(0.25, pitch_value * 0.45))
    return {
        "type": direct_type,
        "depth_um": tg_depth,
        "x_min_um": field(round(-tg_half_width, 6), unit="um", source_kind="inferred_layout", confidence=0.40, method="centered TG lateral placement"),
        "x_max_um": field(round(tg_half_width, 6), unit="um", source_kind="inferred_layout", confidence=0.40, method="centered TG lateral placement"),
        "gate_oxide_thickness_nm": gate_oxide,
    }


def infer_fd_sd(candidate: dict[str, Any], pitch: dict[str, Any], active_si: dict[str, Any]) -> dict[str, Any]:
    pitch_value = float(field_value(pitch, 1.4))
    fd_depth = role_depth_from_rows(candidate, "fd_sd") or role_depth_from_rows(candidate, "floating_diffusion")
    if not fd_depth:
        fd_depth = field(
            round(clamp(float(field_value(active_si, 3.0)) * 0.055, 0.12, 0.28), 6),
            unit="um",
            source_kind="inferred_rule",
            confidence=0.42,
            method="FD depth seed=clamp(0.055*active_si,0.12,0.28)",
        )
    sd_depth = role_depth_from_rows(candidate, "source_drain") or fd_depth
    width = clamp(pitch_value * 0.16, 0.05, 0.25)
    return {
        "floating_diffusion_depth_um": fd_depth,
        "source_drain_depth_um": sd_depth,
        "floating_diffusion_x_min_um": field(round(pitch_value * 0.28, 6), unit="um", source_kind="inferred_layout", confidence=0.35, method="right-side FD placement seed"),
        "floating_diffusion_x_max_um": field(round(min(pitch_value * 0.28 + width, pitch_value * 0.48), 6), unit="um", source_kind="inferred_layout", confidence=0.35, method="right-side FD placement seed"),
    }


def infer_sti(candidate: dict[str, Any], active_si: dict[str, Any]) -> dict[str, Any]:
    depth = candidate_region_depth(candidate, {"sti_depth_um"})
    if not depth:
        depth = field(0.24, unit="um", source_kind="inferred_rule", confidence=0.35, method="modern CIS STI depth seed")
    return {
        "depth_um": depth,
        "fill_material": field("oxide", source_kind="inferred_rule", confidence=0.45, method="standard STI fill assumption"),
        "liner_material": field("oxide_nitride", source_kind="inferred_rule", confidence=0.35, method="standard STI liner assumption"),
    }


def infer_dti_materials(dti_type: dict[str, Any], dti_width: dict[str, Any] | None) -> dict[str, Any]:
    dtype = normalize_space(field_value(dti_type, "")).lower()
    if "front" in dtype or "full_depth" in dtype:
        fill = "poly_si_with_oxide_or_dual_poly_fill"
        fill_conf = 0.45
    elif "back" in dtype or "partial" in dtype:
        fill = "oxide_or_dielectric_fill"
        fill_conf = 0.35
    else:
        fill = "unknown"
        fill_conf = 0.10
    width = as_float(field_value(dti_width)) if dti_width else None
    liner_nm = clamp((width or 0.09) * 1000.0 * 0.23, 8.0, 28.0)
    return {
        "fill_material": field(fill, source_kind="inferred_rule", confidence=fill_conf, method="DTI subtype material heuristic"),
        "sidewall_liner_thickness_nm": field(round(liner_nm, 3), unit="nm", source_kind="inferred_rule", confidence=0.28, method="liner~23% of DTI width, clamped 8-28 nm"),
    }


def measured_doping_archetype(candidate: dict[str, Any]) -> dict[str, Any]:
    rows = parse_sims_rows(candidate)
    by_role: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        role = row_role(row)
        for value in row.get("concentration_cm3_values", []) or []:
            if is_number(value) and value > 0:
                by_role[role].append(float(value))
    anchors: dict[str, dict[str, Any]] = {}
    for role, values in by_role.items():
        anchors[role] = field(
            float(statistics.median(values)),
            unit="cm^-3",
            source_kind="measured_sims_table",
            confidence=0.78,
            method=f"median SIMS/SPM concentration for role {role}; n={len(values)}",
        )
    return anchors


DEFAULT_DOPING_ANCHORS = {
    "p_pinning": 8.0e18,
    "p_plus_contact": 2.0e20,
    "fd_sd": 1.0e20,
    "floating_diffusion": 2.0e19,
    "source_drain": 2.0e19,
    "transfer_gate": 5.0e19,
    "p_well": 5.0e17,
    "photocathode": 1.5e16,
    "pixel_isolation": 7.5e16,
    "background_acceptor": 1.0e14,
}


def infer_doping_model(candidate: dict[str, Any], sims_profile: dict[str, Any] | None) -> dict[str, Any]:
    measured = measured_doping_archetype(candidate)
    anchors: dict[str, dict[str, Any]] = {}
    for role, default_value in DEFAULT_DOPING_ANCHORS.items():
        if role in measured:
            anchors[role] = measured[role]
        else:
            anchors[role] = field(
                default_value,
                unit="cm^-3",
                source_kind="inferred_archetype",
                confidence=0.28,
                method="global median from measured SIMS candidate roles or conservative CIS default",
            )
    if sims_profile:
        kind = "measured_sims_seed_available"
        conf = 0.76
        profile_path = sims_profile.get("profile_abs") or sims_profile.get("profile")
    elif measured:
        kind = "measured_table_anchors_no_profile"
        conf = 0.58
        profile_path = None
    else:
        kind = "estimated_archetype_only"
        conf = 0.25
        profile_path = None
    return {
        "model_kind": field(kind, source_kind="derived", confidence=conf, method="SIMS availability and measured anchor scan"),
        "sims_seed_profile": profile_path,
        "anchors": anchors,
        "solver_warning": "Use anchors as seed values only. They are not process-calibrated implant profiles and may need scaling for drift-diffusion convergence.",
    }


def tcad_profile_links(candidate: dict[str, Any], sims_profile: dict[str, Any] | None) -> dict[str, Any]:
    proxy = candidate.get("generated_tcad_profile")
    proxy_abs = ""
    if proxy:
        proxy_path = Path(str(proxy))
        if not proxy_path.is_absolute():
            proxy_path = ROOT / "image_sensor_db" / proxy_path
        proxy_abs = str(proxy_path)
    sims_abs = ""
    if sims_profile:
        sims_abs = str(sims_profile.get("profile_abs") or sims_profile.get("profile") or "")
    if sims_abs:
        recommended = sims_abs
        kind = "sims_seed_profile"
        confidence = 0.76
        method = "prefer SIMS table seed profile"
    elif proxy_abs:
        recommended = proxy_abs
        kind = "techinsights_proxy_profile"
        confidence = 0.35
        method = "fallback to existing proxy profile"
    else:
        recommended = ""
        kind = "none"
        confidence = 0.0
        method = "no generated profile available"
    return {
        "recommended_profile": field(recommended, source_kind="derived", confidence=confidence, method=method),
        "recommended_profile_kind": field(kind, source_kind="derived", confidence=confidence, method=method),
        "sims_seed_profile": sims_abs,
        "proxy_profile": proxy_abs,
        "warning": "Recommended profile is a TCAD starting point. SIMS seed is stronger than proxy, but neither is product-calibrated.",
    }


def infer_tcad_cell(pitch: dict[str, Any], active_si: dict[str, Any], dti_width: dict[str, Any] | None, dti_depth: dict[str, Any] | None) -> dict[str, Any]:
    pitch_value = float(field_value(pitch, 1.4))
    active_value = float(field_value(active_si, max(2.0, pitch_value * 2.3)))
    dti_width_value = as_float(field_value(dti_width)) if dti_width else clamp(pitch_value * 0.08, 0.035, 0.22)
    dti_depth_value = as_float(field_value(dti_depth)) if dti_depth else 0.0
    split_gap = clamp(pitch_value * 0.04, 0.02, 0.12)
    mesh_um = clamp(pitch_value / 8.0, 0.035, 0.18)
    fine_mesh_um = clamp(min(mesh_um / 2.5, dti_width_value / 2.0), 0.008, 0.06)
    half = pitch_value / 2.0
    return {
        "width_um": pitch,
        "depth_um": active_si,
        "z_width_um": field(round(pitch_value, 6), unit="um", source_kind="inferred_layout", confidence=float(pitch.get("confidence", 0.3)), method="single-pixel square cell z width = pitch"),
        "split_gap_um": field(round(split_gap, 6), unit="um", source_kind="inferred_layout", confidence=0.30, method="split gap seed=clamp(0.04*pitch,0.02,0.12)"),
        "x_min_um": field(round(-half, 6), unit="um", source_kind="derived_geometry", confidence=float(pitch.get("confidence", 0.3)), method="centered single pixel cell"),
        "x_max_um": field(round(half, 6), unit="um", source_kind="derived_geometry", confidence=float(pitch.get("confidence", 0.3)), method="centered single pixel cell"),
        "recommended_mesh_um": field(round(mesh_um, 6), unit="um", source_kind="derived_meshing", confidence=0.55, method="mesh seed from pixel pitch"),
        "recommended_fine_mesh_um": field(round(fine_mesh_um, 6), unit="um", source_kind="derived_meshing", confidence=0.50, method="fine mesh from pitch and DTI width"),
        "dti_left_x_min_um": field(round(-half, 6), unit="um", source_kind="derived_geometry", confidence=combine_confidence(pitch, dti_width or {}), method="side DTI left boundary"),
        "dti_left_x_max_um": field(round(-half + dti_width_value, 6), unit="um", source_kind="derived_geometry", confidence=combine_confidence(pitch, dti_width or {}), method="side DTI left boundary"),
        "dti_right_x_min_um": field(round(half - dti_width_value, 6), unit="um", source_kind="derived_geometry", confidence=combine_confidence(pitch, dti_width or {}), method="side DTI right boundary"),
        "dti_right_x_max_um": field(round(half, 6), unit="um", source_kind="derived_geometry", confidence=combine_confidence(pitch, dti_width or {}), method="side DTI right boundary"),
        "dti_depth_max_um": field(round(clamp(dti_depth_value or active_value, 0.0, active_value * 1.05), 6), unit="um", source_kind="derived_geometry", confidence=float((dti_depth or {}).get("confidence", 0.2)), method="DTI depth clipped to active silicon for mesh"),
    }


def readiness(
    structure: dict[str, Any],
    candidate: dict[str, Any],
    sims_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    score = 0.0
    weights = {
        ("tcad_cell", "width_um"): 14,
        ("tcad_cell", "depth_um"): 14,
        ("dti", "type"): 8,
        ("dti", "depth_um"): 10,
        ("dti", "width_um"): 8,
        ("transfer_gate", "type"): 6,
        ("photodiode", "photocathode_depth_um"): 10,
        ("photodiode", "p_well_depth_um"): 8,
        ("fd_sd", "floating_diffusion_depth_um"): 8,
        ("optical_stack", "height_um"): 5,
    }
    for (section, name), weight in weights.items():
        item = structure.get(section, {}).get(name)
        if isinstance(item, dict) and item.get("value") not in (None, "", "unknown"):
            score += weight * float(item.get("confidence", 0.0))
    if sims_profile:
        score += 12
    elif candidate.get("doping_row_count", 0):
        score += 8
    elif candidate.get("html_snippets") or candidate.get("pdf_snippets"):
        score += 3
    score = clamp(score, 0.0, 100.0)
    missing = []
    for section, name in [("tcad_cell", "width_um"), ("tcad_cell", "depth_um"), ("dti", "depth_um"), ("transfer_gate", "type"), ("photodiode", "photocathode_depth_um")]:
        item = structure.get(section, {}).get(name)
        if not isinstance(item, dict) or item.get("value") in (None, "", "unknown"):
            missing.append(f"{section}.{name}")
    if sims_profile:
        level = "sims_seed_ready"
    elif score >= 58:
        level = "structure_ready_high"
    elif score >= 42:
        level = "structure_ready_medium"
    elif score >= 25:
        level = "proxy_structure_low"
    else:
        level = "insufficient"
    return {
        "level": level,
        "score": round(score, 1),
        "candidate_level": candidate.get("candidate_level"),
        "tcad_candidate_score": candidate.get("tcad_score"),
        "missing_for_structure_model": missing,
        "missing_for_calibrated_deck": [
            "continuous implant profile or process simulation recipe",
            "interface trap/fixed-charge calibration",
            "measured dark current/full-well/lag/charge-transfer targets",
            "exact lateral implant masks and 3D sharing geometry",
        ],
        "recommended_use": (
            "Use for mesh/profile seeding and sensitivity sweeps. Do not use for product accuracy claims without calibration."
        ),
    }


def anchor_value(model: dict[str, Any], role: str, default: float) -> float:
    item = model.get("structure", {}).get("doping", {}).get("anchors", {}).get(role)
    value = field_value(item)
    if is_number(value) and float(value) > 0:
        return float(value)
    return default


def write_estimated_starter_profile(model: dict[str, Any], profile_dir: Path) -> Path:
    """Write a minimal measured_tcad_profile_v1 starter for records with no profile.

    This is intentionally lower-confidence than both SIMS seed and existing
    TechInsights proxy profiles. It exists so every structure model has a
    runnable starting point.
    """
    structure = model["structure"]
    cell = structure["tcad_cell"]
    dti = structure["dti"]
    tg = structure["transfer_gate"]
    fd = structure["fd_sd"]
    pd = structure["photodiode"]
    width = float(field_value(cell["width_um"], 1.4))
    depth = float(field_value(cell["depth_um"], 3.0))
    half = width / 2.0
    dti_width = as_float(field_value(dti.get("width_um"))) or clamp(width * 0.08, 0.035, 0.22)
    body_min = -half + min(dti_width, half * 0.35)
    body_max = half - min(dti_width, half * 0.35)
    pin_depth = clamp(float(field_value(pd["pinning_depth_um"], 0.10)), 0.02, depth)
    photocathode_depth = clamp(float(field_value(pd["photocathode_depth_um"], depth * 0.85)), pin_depth + 0.01, depth)
    p_well_depth = clamp(float(field_value(pd["p_well_depth_um"], 0.5)), 0.05, depth)
    fd_depth = clamp(float(field_value(fd["floating_diffusion_depth_um"], 0.18)), 0.02, depth)
    fd_x_min = clamp(float(field_value(fd["floating_diffusion_x_min_um"], width * 0.28)), -half, half)
    fd_x_max = clamp(float(field_value(fd["floating_diffusion_x_max_um"], width * 0.44)), fd_x_min + 0.01, half)
    tg_x_min = clamp(float(field_value(tg["x_min_um"], -width * 0.18)), -half, half)
    tg_x_max = clamp(float(field_value(tg["x_max_um"], width * 0.18)), tg_x_min + 0.01, half)
    tg_depth = clamp(float(field_value(tg["depth_um"], 0.0)), 0.0, depth)
    roll_x = round(clamp(width * 0.025, 0.008, 0.05), 6)
    roll_d = round(clamp(depth * 0.015, 0.008, 0.08), 6)

    implants: list[dict[str, Any]] = [
        {
            "name": "estimated_p_pinning",
            "type": "analytic_smooth_box",
            "role": "p_pinning",
            "x_min_um": round(body_min, 6),
            "x_max_um": round(body_max, 6),
            "depth_min_um": 0.0,
            "depth_max_um": round(pin_depth, 6),
            "x_rolloff_um": roll_x,
            "depth_rolloff_um": roll_d,
            "donor_cm3": 0.0,
            "acceptor_cm3": anchor_value(model, "p_pinning", DEFAULT_DOPING_ANCHORS["p_pinning"]),
            "measured": False,
            "mapping_inferred": True,
        },
        {
            "name": "estimated_photocathode",
            "type": "analytic_smooth_box",
            "role": "photocathode",
            "x_min_um": round(body_min, 6),
            "x_max_um": round(body_max, 6),
            "depth_min_um": round(pin_depth, 6),
            "depth_max_um": round(photocathode_depth, 6),
            "x_rolloff_um": roll_x,
            "depth_rolloff_um": roll_d,
            "donor_cm3": anchor_value(model, "photocathode", DEFAULT_DOPING_ANCHORS["photocathode"]),
            "acceptor_cm3": 0.0,
            "measured": False,
            "mapping_inferred": True,
        },
        {
            "name": "estimated_p_well",
            "type": "analytic_smooth_box",
            "role": "p_well",
            "x_min_um": round(body_min, 6),
            "x_max_um": round(body_max, 6),
            "depth_min_um": 0.0,
            "depth_max_um": round(p_well_depth, 6),
            "x_rolloff_um": roll_x,
            "depth_rolloff_um": roll_d,
            "donor_cm3": 0.0,
            "acceptor_cm3": anchor_value(model, "p_well", DEFAULT_DOPING_ANCHORS["p_well"]),
            "measured": False,
            "mapping_inferred": True,
        },
        {
            "name": "estimated_fd_sd",
            "type": "analytic_smooth_box",
            "role": "fd_sd",
            "x_min_um": round(fd_x_min, 6),
            "x_max_um": round(fd_x_max, 6),
            "depth_min_um": 0.0,
            "depth_max_um": round(fd_depth, 6),
            "x_rolloff_um": roll_x,
            "depth_rolloff_um": roll_d,
            "donor_cm3": anchor_value(model, "fd_sd", DEFAULT_DOPING_ANCHORS["fd_sd"]),
            "acceptor_cm3": 0.0,
            "measured": False,
            "mapping_inferred": True,
        },
    ]
    if tg_depth > 0.0 and field_value(tg["type"]) != "unknown":
        implants.append(
            {
                "name": "estimated_transfer_gate_channel",
                "type": "analytic_smooth_box",
                "role": "transfer_gate",
                "x_min_um": round(tg_x_min, 6),
                "x_max_um": round(tg_x_max, 6),
                "depth_min_um": 0.0,
                "depth_max_um": round(tg_depth, 6),
                "x_rolloff_um": roll_x,
                "depth_rolloff_um": roll_d,
                "donor_cm3": anchor_value(model, "transfer_gate", DEFAULT_DOPING_ANCHORS["transfer_gate"]),
                "acceptor_cm3": 0.0,
                "measured": False,
                "mapping_inferred": True,
            }
        )
    if field_value(dti["type"]) not in {None, "", "unknown"}:
        dti_depth = clamp(float(field_value(dti["depth_um"], depth)), 0.05, depth)
        side_depth_max = round(dti_depth, 6)
        side_width = clamp(dti_width, 0.02, half)
        for side, x_min, x_max in [
            ("left", -half, -half + side_width),
            ("right", half - side_width, half),
        ]:
            implants.append(
                {
                    "name": f"estimated_{side}_pixel_isolation",
                    "type": "analytic_smooth_box",
                    "role": "pixel_isolation",
                    "x_min_um": round(x_min, 6),
                    "x_max_um": round(x_max, 6),
                    "depth_min_um": 0.0,
                    "depth_max_um": side_depth_max,
                    "x_rolloff_um": roll_x,
                    "depth_rolloff_um": roll_d,
                    "donor_cm3": 0.0,
                    "acceptor_cm3": anchor_value(model, "pixel_isolation", DEFAULT_DOPING_ANCHORS["pixel_isolation"]),
                    "measured": False,
                    "mapping_inferred": True,
                }
            )

    profile = {
        "schema": "measured_tcad_profile_v1",
        "profile_name": f"{model['code']}_{safe_slug(str(model['source'].get('device_name') or 'device'))}_estimated_structure_starter",
        "units": {"length": "um", "doping": "cm^-3", "sheet_charge": "cm^-2"},
        "reference_mode": True,
        "reference_notes": [
            "Generated from TCAD structure DB because no SIMS seed or existing proxy profile was available.",
            "All implants are inferred starter terms. This is the lowest-confidence profile class.",
            "Use only for mesh/solver smoke and sensitivity exploration.",
        ],
        "geometry": {
            "width_um": round(width, 6),
            "depth_um": round(depth, 6),
            "z_width_um": round(float(field_value(cell["z_width_um"], width)), 6),
            "split_gap_um": round(float(field_value(cell["split_gap_um"], clamp(width * 0.04, 0.02, 0.12))), 6),
            "pinning_depth_um": round(pin_depth, 6),
            "dti_width_um": round(dti_width, 6),
            "transfer_gate": {
                "x_min_um": round(tg_x_min, 6),
                "x_max_um": round(tg_x_max, 6),
                "oxide_thickness_um": round(float(field_value(tg["gate_oxide_thickness_nm"], 5.0)) / 1000.0, 6),
            },
            "floating_diffusion": {
                "x_min_um": round(fd_x_min, 6),
                "x_max_um": round(fd_x_max, 6),
                "depth_min_um": 0.0,
                "depth_max_um": round(fd_depth, 6),
            },
        },
        "background": {
            "acceptor_cm3": anchor_value(model, "background_acceptor", DEFAULT_DOPING_ANCHORS["background_acceptor"]),
            "donor_cm3": 0.0,
        },
        "implants": implants,
        "electrical_features": [],
        "interfaces": [],
        "mobility_recombination": {
            "transport_model": "reference_defaults_for_estimated_structure_starter",
            "transport_model_measured": False,
            "transport_model_calibrated": False,
            "mu_n_cm2_v_s": 400.0,
            "mu_p_cm2_v_s": 200.0,
            "tau_n_s": 1.0e-6,
            "tau_p_s": 1.0e-6,
        },
        "calibration_status": {
            "is_measured": False,
            "geometry_measured": False,
            "doping_table_measured": False,
            "deck_calibrated": False,
            "mode": "estimated_structure_starter_profile",
            "source_report_code": model.get("code"),
            "note": "Lowest-confidence generated profile from structure DB estimates.",
        },
        "tcad_structure_source": {
            "code": model.get("code"),
            "readiness": model.get("readiness"),
        },
    }
    profile_path = profile_dir / "profile.json"
    write_json(profile_path, profile)
    return profile_path


def build_structure_record(
    record: dict[str, Any],
    candidate: dict[str, Any],
    stats: EmpiricalStats,
    sims_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    pitch = infer_pixel_pitch(record, stats)
    active_si = infer_active_si(record, stats, pitch)
    dti_type = infer_dti_type(record, pitch, active_si)
    dti_depth = infer_dti_depth(record, stats, active_si, dti_type, pitch)
    dti_width = infer_dti_width(record, stats, pitch, dti_type)
    optical_stack = infer_optical_stack(record, stats, pitch)
    tcad_cell = infer_tcad_cell(pitch, active_si, dti_width, dti_depth)
    photodiode = infer_photodiode(candidate, active_si)
    transfer_gate = infer_transfer_gate(record, candidate, pitch, active_si)
    fd_sd = infer_fd_sd(candidate, pitch, active_si)
    sti = infer_sti(candidate, active_si)
    dti_materials = infer_dti_materials(dti_type, dti_width)
    dti = {
        "type": dti_type,
        "depth_um": dti_depth or field(None, unit="um", source_kind="unavailable", confidence=0.0, method="no DTI depth evidence"),
        "width_um": dti_width or field(None, unit="um", source_kind="unavailable", confidence=0.0, method="no DTI width evidence"),
        **dti_materials,
    }
    optical = {
        "height_um": optical_stack,
        "cfa_thickness_um": spec_field(record, "cfa_thickness_um", unit="um", confidence=0.82)
        or field(None, unit="um", source_kind="unavailable", confidence=0.0, method="no CFA thickness evidence"),
        "grid_pitch_um": spec_field(record, "grid_pitch_um", unit="um", confidence=0.82)
        or field(field_value(pitch), unit="um", source_kind="inferred_rule", confidence=0.35, method="grid pitch approximated as pixel pitch"),
    }
    process = {
        "cis_process_nm": spec_field(record, "cis_process_nm", unit="nm", confidence=0.80)
        or field(None, unit="nm", source_kind="unavailable", confidence=0.0, method="no CIS process node evidence"),
        "pixel_beol_metal_pitch_nm": spec_field(record, "pixel_beol_metal_pitch_nm", unit="nm", confidence=0.80)
        or field(None, unit="nm", source_kind="unavailable", confidence=0.0, method="no BEOL metal pitch evidence"),
        "pixel_architecture": spec_field(record, "pixel_architecture", confidence=0.86)
        or field("unknown", source_kind="unavailable", confidence=0.0, method="no pixel architecture evidence"),
    }
    doping = infer_doping_model(candidate, sims_profile)
    structure = {
        "tcad_cell": tcad_cell,
        "dti": dti,
        "sti": sti,
        "photodiode": photodiode,
        "transfer_gate": transfer_gate,
        "fd_sd": fd_sd,
        "optical_stack": optical,
        "process": process,
        "doping": doping,
        "tcad_profiles": tcad_profile_links(candidate, sims_profile),
    }
    return {
        "schema": "image_sensor_tcad_structure_model_v1",
        "code": record.get("code"),
        "source": source_context(record, candidate),
        "structure": structure,
        "readiness": readiness(structure, candidate, sims_profile),
        "source_candidate_report": {
            "candidate_level": candidate.get("candidate_level"),
            "geometry_score": candidate.get("geometry_score"),
            "doping_row_count": candidate.get("doping_row_count"),
            "doping_table_count": candidate.get("doping_table_count"),
            "source_html": candidate.get("source_html", []),
            "source_pdf": candidate.get("source_pdf", []),
        },
    }


def flatten_for_csv(model: dict[str, Any], rel_model_path: str) -> dict[str, Any]:
    s = model["structure"]
    src = model["source"]
    ready = model["readiness"]
    return {
        "code": model.get("code"),
        "manufacturer": src.get("manufacturer"),
        "device_name": src.get("device_name"),
        "report_type": src.get("report_type"),
        "readiness_level": ready.get("level"),
        "readiness_score": ready.get("score"),
        "candidate_level": ready.get("candidate_level"),
        "pixel_pitch_um": field_value(s["tcad_cell"]["width_um"]),
        "pixel_pitch_source": s["tcad_cell"]["width_um"].get("source_kind"),
        "active_si_um": field_value(s["tcad_cell"]["depth_um"]),
        "active_si_source": s["tcad_cell"]["depth_um"].get("source_kind"),
        "dti_type": field_value(s["dti"]["type"]),
        "dti_depth_um": field_value(s["dti"]["depth_um"]),
        "dti_depth_source": s["dti"]["depth_um"].get("source_kind"),
        "dti_width_um": field_value(s["dti"]["width_um"]),
        "photocathode_depth_um": field_value(s["photodiode"]["photocathode_depth_um"]),
        "p_well_depth_um": field_value(s["photodiode"]["p_well_depth_um"]),
        "pinning_depth_um": field_value(s["photodiode"]["pinning_depth_um"]),
        "transfer_gate_type": field_value(s["transfer_gate"]["type"]),
        "transfer_gate_depth_um": field_value(s["transfer_gate"]["depth_um"]),
        "fd_depth_um": field_value(s["fd_sd"]["floating_diffusion_depth_um"]),
        "optical_stack_um": field_value(s["optical_stack"]["height_um"]),
        "doping_model_kind": field_value(s["doping"]["model_kind"]),
        "sims_seed_profile": s["doping"].get("sims_seed_profile") or "",
        "recommended_tcad_profile": field_value(s["tcad_profiles"]["recommended_profile"], ""),
        "recommended_profile_kind": field_value(s["tcad_profiles"]["recommended_profile_kind"], ""),
        "model_json": rel_model_path,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_html(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    summary_items = "".join(f"<li><strong>{html.escape(str(k))}</strong>: {html.escape(str(v))}</li>" for k, v in summary.items())
    table_rows = []
    for row in rows:
        model_link = html.escape(row["model_json"])
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['code']))}</td>"
            f"<td>{html.escape(str(row['manufacturer']))}<br><small>{html.escape(str(row['device_name']))}</small></td>"
            f"<td>{html.escape(str(row['readiness_level']))}<br><small>{html.escape(str(row['readiness_score']))}</small></td>"
            f"<td>pitch {html.escape(str(row['pixel_pitch_um']))} um<br>"
            f"active {html.escape(str(row['active_si_um']))} um<br>"
            f"<small>{html.escape(str(row['pixel_pitch_source']))}/{html.escape(str(row['active_si_source']))}</small></td>"
            f"<td>{html.escape(str(row['dti_type']))}<br>"
            f"{html.escape(str(row['dti_depth_um']))} um x {html.escape(str(row['dti_width_um']))} um<br>"
            f"<small>{html.escape(str(row['dti_depth_source']))}</small></td>"
            f"<td>PD {html.escape(str(row['photocathode_depth_um']))} um<br>"
            f"P-well {html.escape(str(row['p_well_depth_um']))} um<br>"
            f"pin {html.escape(str(row['pinning_depth_um']))} um</td>"
            f"<td>{html.escape(str(row['transfer_gate_type']))}<br>"
            f"TG {html.escape(str(row['transfer_gate_depth_um']))} um<br>"
            f"FD {html.escape(str(row['fd_depth_um']))} um</td>"
            f"<td>{html.escape(str(row['doping_model_kind']))}<br><small>{html.escape(str(row['sims_seed_profile']))}</small></td>"
            f"<td>{html.escape(str(row['recommended_profile_kind']))}<br><small>{html.escape(str(row['recommended_tcad_profile']))}</small></td>"
            f"<td><a href=\"{model_link}\">model</a></td>"
            "</tr>"
        )
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Sensor TCAD Structure DB</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }
    h1 { font-size: 24px; margin-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #d6dbe1; padding: 7px; vertical-align: top; }
    th { background: #edf2f7; text-align: left; position: sticky; top: 0; }
    tr:nth-child(even) { background: #f8fafc; }
    small { color: #667085; overflow-wrap: anywhere; }
    ul { line-height: 1.45; }
  </style>
</head>
<body>
  <h1>Image Sensor TCAD Structure DB</h1>
  <p>Every model separates extracted values from inferred values using source_kind and confidence. Use as a TCAD starting point, not a calibrated process deck.</p>
  <ul>
"""
        + summary_items
        + """
  </ul>
  <table>
    <thead>
      <tr>
        <th>Code</th><th>Device</th><th>Readiness</th><th>Cell</th><th>DTI</th><th>Photodiode</th><th>TG/FD</th><th>Doping</th><th>Profile</th><th>JSON</th>
      </tr>
    </thead>
    <tbody>
"""
        + "\n".join(table_rows)
        + """
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def validate_output(models: list[dict[str, Any]], csv_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    codes = [str(model.get("code")) for model in models]
    if len(codes) != len(set(codes)):
        issues.append({"level": "error", "message": "duplicate model codes detected"})
    if len(models) != len(csv_rows):
        issues.append({"level": "error", "message": "model/csv row count mismatch", "models": len(models), "csv_rows": len(csv_rows)})
    required_positive = [
        ("tcad_cell", "width_um"),
        ("tcad_cell", "depth_um"),
        ("tcad_cell", "recommended_mesh_um"),
        ("tcad_cell", "recommended_fine_mesh_um"),
    ]
    for model in models:
        structure = model.get("structure", {})
        for section, key in required_positive:
            item = structure.get(section, {}).get(key)
            value = field_value(item)
            if not is_number(value) or float(value) <= 0:
                issues.append(
                    {
                        "level": "error",
                        "code": model.get("code"),
                        "message": f"{section}.{key} must be positive",
                        "value": value,
                    }
                )
        ready = model.get("readiness", {})
        if not ready.get("level"):
            issues.append({"level": "error", "code": model.get("code"), "message": "missing readiness level"})
    for row in csv_rows:
        model_path = output_dir / str(row.get("model_json"))
        if not model_path.exists():
            issues.append({"level": "error", "code": row.get("code"), "message": "missing model JSON file", "path": str(model_path)})
        profile_path = row.get("recommended_tcad_profile")
        if not profile_path:
            issues.append({"level": "error", "code": row.get("code"), "message": "missing recommended TCAD profile"})
        elif not Path(str(profile_path)).exists():
            issues.append({"level": "error", "code": row.get("code"), "message": "recommended TCAD profile does not exist", "path": str(profile_path)})
    return {
        "schema": "image_sensor_tcad_structure_db_validation_v1",
        "pass": not any(issue["level"] == "error" for issue in issues),
        "model_count": len(models),
        "csv_row_count": len(csv_rows),
        "issue_count": len(issues),
        "issues": issues[:100],
    }


def load_sims_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    manifest = read_json(path)
    return {item["code"]: item for item in manifest.get("profiles", [])}


def build(args: argparse.Namespace) -> dict[str, Any]:
    catalog = read_json(args.catalog)
    records = catalog.get("records", [])
    candidate_report = read_json(args.candidate_report)
    candidates = {record["code"]: record for record in candidate_report.get("records", [])}
    sims_profiles = load_sims_profiles(args.sims_manifest)
    stats = EmpiricalStats(records)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        code = record.get("code")
        candidate = candidates.get(code, {"code": code, "candidate_level": "not_scanned", "key_geometry": {}})
        model = build_structure_record(record, candidate, stats, sims_profiles.get(code))
        slug = safe_slug(f"{code}_{model['source'].get('manufacturer')}_{model['source'].get('device_name')}")
        profile_links = model["structure"]["tcad_profiles"]
        if field_value(profile_links["recommended_profile_kind"]) == "none":
            starter_path = write_estimated_starter_profile(
                model,
                args.output_dir / "starter_profiles" / slug,
            )
            profile_links["starter_profile"] = str(starter_path)
            profile_links["recommended_profile"] = field(
                str(starter_path),
                source_kind="generated_estimated",
                confidence=0.18,
                method="fallback starter profile from TCAD structure DB estimates",
            )
            profile_links["recommended_profile_kind"] = field(
                "estimated_structure_starter_profile",
                source_kind="generated_estimated",
                confidence=0.18,
                method="fallback starter profile from TCAD structure DB estimates",
            )
        model_path = models_dir / f"{slug}.json"
        write_json(model_path, model)
        rel_model_path = os.path.relpath(model_path, args.output_dir).replace(os.sep, "/")
        models.append(model)
        csv_rows.append(flatten_for_csv(model, rel_model_path))

    csv_rows.sort(key=lambda row: (float(row.get("readiness_score") or 0), str(row.get("code"))), reverse=True)
    level_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for row in csv_rows:
        level_counts[str(row["readiness_level"])] += 1
        source_counts[str(row["doping_model_kind"])] += 1
    summary = {
        "schema": "image_sensor_tcad_structure_db_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(models),
        "model_count": len(models),
        "sims_seed_ready": level_counts.get("sims_seed_ready", 0),
        "structure_ready_high": level_counts.get("structure_ready_high", 0),
        "structure_ready_medium": level_counts.get("structure_ready_medium", 0),
        "proxy_structure_low": level_counts.get("proxy_structure_low", 0),
        "insufficient": level_counts.get("insufficient", 0),
        "doping_model_counts": dict(source_counts),
        "notes": [
            "Extracted values have source_kind=extracted or measured_sims_table.",
            "Inferred values carry source_kind=inferred_* and lower confidence.",
            "This DB is for TCAD setup, meshing, and sensitivity sweeps, not product accuracy claims.",
        ],
    }
    manifest = {
        **summary,
        "source_catalog": str(args.catalog),
        "source_candidate_report": str(args.candidate_report),
        "source_sims_manifest": str(args.sims_manifest),
        "models_dir": "models",
        "models": [
            {
                "code": row["code"],
                "manufacturer": row["manufacturer"],
                "device_name": row["device_name"],
                "readiness_level": row["readiness_level"],
                "readiness_score": row["readiness_score"],
                "model_json": row["model_json"],
            }
            for row in csv_rows
        ],
    }
    validation = validate_output(models, csv_rows, args.output_dir)
    manifest["validation"] = validation
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "validation.json", validation)
    write_csv(args.output_dir / "structure_summary.csv", csv_rows)
    render_html(args.output_dir / "index.html", csv_rows, summary)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--sims-manifest", type=Path, default=DEFAULT_SIMS_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "record_count": manifest["record_count"],
                "sims_seed_ready": manifest["sims_seed_ready"],
                "structure_ready_high": manifest["structure_ready_high"],
                "structure_ready_medium": manifest["structure_ready_medium"],
                "proxy_structure_low": manifest["proxy_structure_low"],
                "insufficient": manifest["insufficient"],
                "output_dir": str(args.output_dir),
                "index": str(args.output_dir / "index.html"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
