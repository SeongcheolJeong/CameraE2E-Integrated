#!/usr/bin/env python3
"""Generate executable TCAD seed profiles from extracted SIMS doping tables.

The generated profiles are useful starting points for mesh/solver setup, not
calibrated process decks. Concentrations and depth anchors come from extracted
SIMS/SPM tables, while lateral placement and smoothing are inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_REPORT = ROOT / "image_sensor_db" / "tcad_candidate_report.json"
DEFAULT_OUTPUT_DIR = ROOT / "measured_profiles" / "techinsights_sims_seed"


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


def finite_positive(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            out.append(number)
    return out


def geometric_mean(values: list[float]) -> float:
    positives = finite_positive(values)
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def normalize_depth_text(value: str) -> str:
    return (
        normalize_space(value)
        .replace("–", "-")
        .replace("—", "-")
        .replace("^a^", "")
        .replace("~", "")
    )


def parse_depth_range_um(value: str) -> tuple[float | None, float | None]:
    text = normalize_depth_text(value)
    if not text or text in {"-", "–"}:
        return None, None
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None, None
    lower = min(numbers)
    upper = max(numbers)
    if "nm" in text.lower() and not any(unit in text.lower() for unit in ("um", "µm", "μm")):
        lower /= 1000.0
        upper /= 1000.0
    return lower, upper


def concentration_seed(row: dict[str, Any]) -> tuple[float | None, str]:
    values = finite_positive(row.get("concentration_cm3_values", []))
    if not values:
        return None, "missing_concentration"
    text = normalize_space(row.get("concentration_text", "")).lower()
    feature = normalize_space(row.get("feature", "")).lower()
    if text.startswith("<"):
        return min(values) * 0.5, "half_of_reported_upper_bound"
    if any(token in feature for token in ("contact", "fd", "s/d", "vss", "vtg", "pinning", "n+")):
        return max(values), "max_component_for_local_high_dose_feature"
    if len(values) > 1:
        return geometric_mean(values), "geometric_mean_of_reported_components_or_range"
    return values[0], "single_reported_value"


def row_text(row: dict[str, Any]) -> str:
    return normalize_space(
        " ".join(str(row.get(key, "")) for key in ("feature", "dopant", "concentration_text"))
    ).lower()


def polarity(row: dict[str, Any]) -> str | None:
    text = row_text(row)
    if any(token in text for token in ("n-type", "n^+^", "n+", "31p", "75as", "phosphorus", "arsenic")):
        return "donor"
    if any(token in text for token in ("p-type", "p^+^", "p+", "11b", "10b", "boron")):
        return "acceptor"
    if any(token in text for token in ("photocathode", "cathode", "fd", "s/d", "vtg", "n-well")):
        return "donor"
    if any(token in text for token in ("p-well", "pinning", "isolation", "backside passivation")):
        return "acceptor"
    return None


def role_for_feature(feature_text: str) -> str:
    text = normalize_space(feature_text).lower()
    if "active si" in text:
        return "active_si_background"
    if "pinning" in text:
        return "p_pinning"
    if "vss" in text or ("contact" in text and ("p+" in text or "p^+^" in text)):
        return "p_plus_contact"
    if "fd" in text and "s/d" in text:
        return "fd_and_source_drain"
    if "fd" in text:
        return "floating_diffusion"
    if "s/d" in text:
        return "source_drain"
    if "vtg" in text or "transfer gate" in text:
        return "vertical_transfer_gate"
    if "p-well" in text or "p well" in text:
        return "p_well"
    if "n-well" in text or "n well" in text:
        return "n_well"
    if "photocathode" in text or "cathode" in text:
        return "photocathode"
    if "channel" in text:
        return "channel"
    if "buried" in text:
        return "buried_region"
    if "dti" in text and ("passivation" in text or "fill" in text or "polysi" in text or "poly" in text):
        return "dti_non_silicon_or_interface_metadata"
    if "backside passivation" in text:
        return "backside_passivation_metadata"
    if "isolation" in text:
        return "pixel_isolation"
    return "generic_silicon_region"


def executable_role(role: str) -> bool:
    return role not in {
        "active_si_background",
        "dti_non_silicon_or_interface_metadata",
        "backside_passivation_metadata",
    }


def is_pixel_array_row(row: dict[str, Any]) -> bool:
    caption = normalize_space(row.get("_source_table_caption", "")).lower()
    raw = [normalize_space(item).lower() for item in row.get("raw", [])]
    first_cell = raw[0] if raw else ""
    if "periphery" in caption:
        return False
    if "periphery" in first_cell or "peripheral" in first_cell:
        return False
    if "pixel array" in caption:
        return True
    if "pixel" in first_cell:
        return True
    if "p- and n-type regions" in caption or "doping levels" in caption:
        return "pixel" in first_cell
    return True


def estimate_background_acceptor(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    for row in rows:
        if role_for_feature(row.get("feature", "")) != "active_si_background":
            continue
        if polarity(row) != "acceptor":
            continue
        seed, method = concentration_seed(row)
        if seed is None:
            continue
        return clamp(seed, 1.0e13, 5.0e15), {"row": row, "seed_method": method}
    return 1.0e14, None


def geometry_from_record(record: dict[str, Any]) -> dict[str, Any]:
    geom = record.get("key_geometry", {})
    pitch = float(geom.get("pixel_pitch_um") or 1.4)
    active_depth = float(
        geom.get("active_si_thickness_um")
        or geom.get("dti_depth_um")
        or max(pitch * 2.5, 2.0)
    )
    dti_width_um = None
    if geom.get("dti_width_nm") not in (None, ""):
        dti_width_um = float(geom["dti_width_nm"]) / 1000.0
    if not dti_width_um:
        dti_width_um = clamp(pitch * 0.08, 0.035, 0.22)
    split_gap = clamp(pitch * 0.04, 0.02, 0.12)
    tg_half = clamp(pitch * 0.18, 0.04, min(0.25, pitch * 0.45))
    fd_width = clamp(pitch * 0.16, 0.05, 0.25)
    fd_depth = clamp(active_depth * 0.05, 0.08, 0.30)
    return {
        "width_um": round(pitch, 6),
        "depth_um": round(active_depth, 6),
        "z_width_um": round(pitch, 6),
        "split_gap_um": round(split_gap, 6),
        "pinning_depth_um": round(clamp(active_depth * 0.025, 0.05, 0.18), 6),
        "dti_width_um": round(dti_width_um, 6),
        "dti": {
            "type": geom.get("dti_type"),
            "depth_um": geom.get("dti_depth_um"),
            "width_um": dti_width_um,
            "source": "TechInsights-derived geometry table where available",
        },
        "transfer_gate": {
            "x_min_um": round(-tg_half, 6),
            "x_max_um": round(tg_half, 6),
            "oxide_thickness_um": 0.006,
            "type": geom.get("transfer_gate_type"),
        },
        "floating_diffusion": {
            "x_min_um": round(pitch * 0.28, 6),
            "x_max_um": round(min(pitch * 0.28 + fd_width, pitch * 0.48), 6),
            "depth_min_um": 0.0,
            "depth_max_um": round(fd_depth, 6),
        },
    }


def body_bounds(geometry: dict[str, Any]) -> tuple[float, float]:
    half = float(geometry["width_um"]) / 2.0
    dti_width = float(geometry.get("dti_width_um", 0.0) or 0.0)
    margin = min(max(dti_width, float(geometry["width_um"]) * 0.03), half * 0.35)
    return -half + margin, half - margin


def row_depth_or_default(
    row: dict[str, Any],
    role: str,
    geometry: dict[str, Any],
) -> tuple[float, float]:
    depth = float(geometry["depth_um"])
    lower, upper = parse_depth_range_um(row.get("depth_text", ""))
    if lower is not None and upper is not None:
        if role in {"photocathode", "active_si_background"} and math.isclose(lower, upper):
            return 0.0, clamp(upper, 0.01, depth)
        if role in {"pixel_isolation", "buried_region"} and upper > lower:
            return clamp(lower, 0.0, depth), clamp(upper, 0.01, depth)
        return 0.0, clamp(upper, 0.01, depth)
    defaults = {
        "p_pinning": min(0.12, depth),
        "p_plus_contact": min(0.12, depth),
        "fd_and_source_drain": min(0.22, depth),
        "floating_diffusion": min(0.22, depth),
        "source_drain": min(0.22, depth),
        "vertical_transfer_gate": min(0.45, depth),
        "p_well": min(0.60, depth),
        "n_well": min(0.60, depth),
        "channel": min(0.45, depth),
        "photocathode": depth,
        "buried_region": depth,
        "pixel_isolation": depth,
        "generic_silicon_region": min(depth, 0.8),
    }
    return 0.0, clamp(defaults.get(role, min(depth, 0.5)), 0.01, depth)


def implant_bounds(
    row: dict[str, Any],
    role: str,
    geometry: dict[str, Any],
) -> list[dict[str, float]]:
    width = float(geometry["width_um"])
    half = width / 2.0
    body_min, body_max = body_bounds(geometry)
    dti_width = float(geometry.get("dti_width_um", width * 0.06) or width * 0.06)
    d0, d1 = row_depth_or_default(row, role, geometry)
    if d1 <= d0:
        d1 = min(float(geometry["depth_um"]), d0 + 0.01)

    if role in {"p_pinning", "p_well", "n_well", "photocathode", "channel", "buried_region", "generic_silicon_region"}:
        return [{"x_min_um": body_min, "x_max_um": body_max, "depth_min_um": d0, "depth_max_um": d1}]
    if role == "vertical_transfer_gate":
        tg = geometry.get("transfer_gate", {})
        return [
            {
                "x_min_um": float(tg.get("x_min_um", -width * 0.15)),
                "x_max_um": float(tg.get("x_max_um", width * 0.15)),
                "depth_min_um": d0,
                "depth_max_um": d1,
            }
        ]
    if role in {"floating_diffusion", "fd_and_source_drain", "source_drain"}:
        fd = geometry.get("floating_diffusion", {})
        return [
            {
                "x_min_um": float(fd.get("x_min_um", width * 0.25)),
                "x_max_um": float(fd.get("x_max_um", width * 0.45)),
                "depth_min_um": d0,
                "depth_max_um": d1,
            }
        ]
    if role == "p_plus_contact":
        return [
            {
                "x_min_um": -half + dti_width,
                "x_max_um": -half + dti_width + clamp(width * 0.12, 0.04, 0.18),
                "depth_min_um": d0,
                "depth_max_um": d1,
            }
        ]
    if role == "pixel_isolation":
        side_width = clamp(max(dti_width, width * 0.06), 0.03, half * 0.5)
        return [
            {"x_min_um": -half, "x_max_um": -half + side_width, "depth_min_um": d0, "depth_max_um": d1},
            {"x_min_um": half - side_width, "x_max_um": half, "depth_min_um": d0, "depth_max_um": d1},
        ]
    return [{"x_min_um": body_min, "x_max_um": body_max, "depth_min_um": d0, "depth_max_um": d1}]


def make_implants_and_metadata(
    rows: list[dict[str, Any]],
    geometry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    implants: list[dict[str, Any]] = []
    metadata_features: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        feature = normalize_space(row.get("feature", f"row_{index}"))
        role = role_for_feature(feature)
        polarity_value = polarity(row)
        seed, method = concentration_seed(row)
        normalized = {
            "source_index": index,
            "feature": feature,
            "role": role,
            "polarity": polarity_value,
            "dopant": row.get("dopant"),
            "concentration_text": row.get("concentration_text"),
            "concentration_cm3_values": row.get("concentration_cm3_values", []),
            "seed_concentration_cm3": seed,
            "seed_method": method,
            "depth_text": row.get("depth_text"),
            "depth_um": row.get("depth_um"),
            "raw": row.get("raw", []),
            "source_table_caption": row.get("_source_table_caption", ""),
        }
        normalized_rows.append(normalized)
        pixel_array_row = is_pixel_array_row(row)
        if seed is None or polarity_value is None or not executable_role(role) or not pixel_array_row:
            reason = (
                "non-silicon/interface feature or missing polarity/concentration; "
                "do not apply to silicon NetDoping without a resolved material/interface model"
            )
            if not pixel_array_row:
                reason = "periphery/non-pixel-array table row retained as metadata; not applied to pixel-array NetDoping"
            metadata_features.append(
                {
                    "name": f"sims_row_{index:02d}_{safe_slug(feature)}",
                    "type": "metadata_only",
                    "role": role,
                    "measured": bool(seed is not None),
                    "source_feature": feature,
                    "dopant": row.get("dopant"),
                    "concentration_text": row.get("concentration_text"),
                    "seed_concentration_cm3": seed,
                    "source_table_caption": row.get("_source_table_caption", ""),
                    "reason_not_executable": reason,
                }
            )
            continue
        for copy_index, bounds in enumerate(implant_bounds(row, role, geometry), start=1):
            suffix = f"_{copy_index}" if len(implant_bounds(row, role, geometry)) > 1 else ""
            implant = {
                "name": f"sims_{index:02d}_{safe_slug(feature)}{suffix}",
                "type": "analytic_smooth_box",
                "role": role,
                "x_min_um": round(bounds["x_min_um"], 6),
                "x_max_um": round(bounds["x_max_um"], 6),
                "depth_min_um": round(bounds["depth_min_um"], 6),
                "depth_max_um": round(bounds["depth_max_um"], 6),
                "x_rolloff_um": round(clamp(float(geometry["width_um"]) * 0.025, 0.005, 0.04), 6),
                "depth_rolloff_um": round(clamp(float(geometry["depth_um"]) * 0.015, 0.005, 0.08), 6),
                "donor_cm3": seed if polarity_value == "donor" else 0.0,
                "acceptor_cm3": seed if polarity_value == "acceptor" else 0.0,
                "measured": True,
                "mapping_inferred": True,
                "source_feature": feature,
                "source_dopant": row.get("dopant"),
                "source_concentration_text": row.get("concentration_text"),
                "source_depth_text": row.get("depth_text"),
                "source_table_caption": row.get("_source_table_caption", ""),
                "source_table_source": row.get("_source_table_source", ""),
                "seed_method": method,
                "note": "SIMS/SPM table concentration with inferred lateral placement; not a calibrated implant process model.",
            }
            implants.append(implant)
    return implants, metadata_features, normalized_rows


def profile_for_record(record: dict[str, Any], output_dir: Path) -> tuple[Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in record.get("doping_tables", []):
        for row in table.get("rows", []):
            row_copy = dict(row)
            row_copy["_source_table_caption"] = table.get("caption", "")
            row_copy["_source_table_source"] = table.get("source", "")
            rows.append(row_copy)
    geometry = geometry_from_record(record)
    background_acceptor, background_source = estimate_background_acceptor(rows)
    implants, metadata_features, normalized_rows = make_implants_and_metadata(rows, geometry)
    slug = safe_slug(f"{record.get('code')}_{record.get('manufacturer')}_{record.get('device_name')}")
    profile_dir = output_dir / slug
    profile_path = profile_dir / "profile.json"
    source_tables = [
        {
            "caption": table.get("caption"),
            "source": table.get("source"),
            "row_count": len(table.get("rows", [])),
        }
        for table in record.get("doping_tables", [])
    ]
    profile = {
        "schema": "measured_tcad_profile_v1",
        "profile_name": f"{record.get('code')}_{slug}_techinsights_sims_seed_profile",
        "units": {"length": "um", "doping": "cm^-3", "sheet_charge": "cm^-2"},
        "reference_mode": False,
        "reference_notes": [
            "SIMS/SPM table rows provide measured concentration/depth anchors where present.",
            "Lateral placement, smoothing, and overlap handling are inferred for open-source TCAD setup.",
            "This is not a calibrated product process deck and should not be used as accuracy evidence without target calibration.",
            "DTI poly-Si fill and passivation rows are exported as metadata unless the current silicon mesh can resolve those regions/interfaces.",
        ],
        "geometry": geometry,
        "background": {"acceptor_cm3": background_acceptor, "donor_cm3": 0.0},
        "implants": implants,
        "electrical_features": metadata_features,
        "interfaces": [],
        "mobility_recombination": {
            "transport_model": "caughey_thomas_doping_and_field_dependent_reference_v1",
            "transport_model_measured": False,
            "transport_model_calibrated": False,
            "mu_n_model": "caughey_thomas_doping_dependent_reference",
            "mu_p_model": "caughey_thomas_doping_dependent_reference",
            "mu_n_cm2_v_s": 400.0,
            "mu_p_cm2_v_s": 200.0,
            "tau_n_s": 1.0e-6,
            "tau_p_s": 1.0e-6,
            "note": "Transport/lifetime settings are still reference defaults; only selected doping anchors come from local extracted tables.",
        },
        "calibration_status": {
            "is_measured": False,
            "geometry_measured": bool(record.get("geometry_score", 0) >= 6),
            "doping_table_measured": True,
            "deck_calibrated": False,
            "mode": "techinsights_sims_table_seed",
            "source_report_code": record.get("code"),
            "note": "Measured table values are present, but this executable profile uses inferred 2D placement and is not a calibrated TCAD process deck.",
        },
        "techinsights_source": {
            "code": record.get("code"),
            "manufacturer": record.get("manufacturer"),
            "device_name": record.get("device_name"),
            "title": record.get("title"),
            "report_type": record.get("report_type"),
            "release_date": record.get("release_date"),
            "source_html": record.get("source_html", []),
            "source_pdf": record.get("source_pdf", []),
            "key_geometry": record.get("key_geometry", {}),
            "doping_tables": source_tables,
        },
        "sims_seed_summary": {
            "doping_row_count": len(rows),
            "executable_implant_count": len(implants),
            "metadata_only_row_count": len(metadata_features),
            "background_acceptor_source": background_source,
            "normalized_rows_file": "sims_doping_rows.csv",
        },
    }
    write_json(profile_path, profile)
    write_rows_csv(profile_dir / "sims_doping_rows.csv", normalized_rows)
    return profile_path, profile


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_index",
        "feature",
        "role",
        "polarity",
        "dopant",
        "concentration_text",
        "seed_concentration_cm3",
        "seed_method",
        "depth_text",
        "depth_um",
        "source_table_caption",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_manifest_csv(path: Path, items: list[dict[str, Any]]) -> None:
    fields = [
        "code",
        "manufacturer",
        "device_name",
        "profile",
        "doping_row_count",
        "executable_implant_count",
        "metadata_only_row_count",
        "pixel_pitch_um",
        "active_si_thickness_um",
        "dti_type",
        "dti_depth_um",
        "dti_width_nm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field, "") for field in fields})


def render_index(path: Path, items: list[dict[str, Any]]) -> None:
    rows = []
    for item in items:
        profile = item["profile"]
        rows.append(
            "<tr>"
            f"<td>{item['code']}</td>"
            f"<td>{item['manufacturer']}<br><small>{item['device_name']}</small></td>"
            f"<td>{item['pixel_pitch_um']} um<br>{item['active_si_thickness_um']} um active Si</td>"
            f"<td>{item['dti_type']}<br>{item['dti_depth_um']} um / {item['dti_width_nm']} nm</td>"
            f"<td>{item['doping_row_count']} rows<br>{item['executable_implant_count']} executable<br>{item['metadata_only_row_count']} metadata</td>"
            f"<td><a href=\"{profile}\">profile.json</a></td>"
            "</tr>"
        )
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TechInsights SIMS TCAD Seed Profiles</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }
    h1 { font-size: 24px; margin-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #d6dbe1; padding: 8px; vertical-align: top; }
    th { background: #edf2f7; text-align: left; }
    tr:nth-child(even) { background: #f8fafc; }
    small { color: #667085; }
  </style>
</head>
<body>
  <h1>TechInsights SIMS TCAD Seed Profiles</h1>
  <p>These are executable seed profiles, not calibrated product decks. Concentrations/depth anchors come from extracted table rows; lateral placement is inferred.</p>
  <table>
    <thead><tr><th>Code</th><th>Device</th><th>Core Geometry</th><th>DTI</th><th>SIMS Mapping</th><th>Profile</th></tr></thead>
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(args.candidate_report)
    selected = [
        record
        for record in report.get("records", [])
        if record.get("candidate_level") == "measured_doping_table"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_items: list[dict[str, Any]] = []
    for record in selected:
        profile_path, profile = profile_for_record(record, args.output_dir)
        geom = record.get("key_geometry", {})
        manifest_items.append(
            {
                "code": record.get("code"),
                "manufacturer": record.get("manufacturer"),
                "device_name": record.get("device_name"),
                "profile": profile_path.relative_to(args.output_dir).as_posix(),
                "profile_abs": str(profile_path),
                "doping_row_count": profile["sims_seed_summary"]["doping_row_count"],
                "executable_implant_count": profile["sims_seed_summary"]["executable_implant_count"],
                "metadata_only_row_count": profile["sims_seed_summary"]["metadata_only_row_count"],
                "pixel_pitch_um": geom.get("pixel_pitch_um"),
                "active_si_thickness_um": geom.get("active_si_thickness_um"),
                "dti_type": geom.get("dti_type"),
                "dti_depth_um": geom.get("dti_depth_um"),
                "dti_width_nm": geom.get("dti_width_nm"),
            }
        )
    manifest = {
        "schema": "techinsights_sims_seed_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidate_report": str(args.candidate_report),
        "profile_count": len(manifest_items),
        "notes": [
            "Profiles are executable measured_tcad_profile_v1 seed inputs.",
            "SIMS/SPM concentration rows are measured/table-derived where present.",
            "2D lateral placement and smoothing are inferred and require calibration before product claims.",
            "Non-silicon DTI fill/passivation rows are retained as metadata only.",
        ],
        "profiles": manifest_items,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_manifest_csv(args.output_dir / "manifest.csv", manifest_items)
    render_index(args.output_dir / "index.html", manifest_items)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "profile_count": manifest["profile_count"],
                "output_dir": str(args.output_dir),
                "manifest": str(args.output_dir / "manifest.json"),
                "index": str(args.output_dir / "index.html"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
