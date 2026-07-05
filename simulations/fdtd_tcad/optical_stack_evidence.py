#!/usr/bin/env python3
"""Build optical-stack geometry/n,k evidence for the TCAD accuracy gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REQUIRED_WAVELENGTHS_UM = (0.45, 0.55, 0.65)


REQUIRED_GEOMETRY_KEYS = {
    "pitch",
    "lens_height",
    "cfa_thickness",
    "passivation_thickness",
    "si_thickness",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def classify_source(material: dict[str, Any], table_path: Path) -> str:
    if bool(material.get("measured", False)):
        return "measured"
    text = " ".join(
        str(material.get(key, ""))
        for key in ("source", "source_url", "usage", "note", "nk_table")
    ).lower()
    path_text = str(table_path).lower()
    if "proxy" in text or "proxy" in path_text:
        return "proxy"
    if material.get("source_url") or "doi" in text or "patent" in text:
        return "public_reference"
    if material.get("model"):
        return "model"
    return "unspecified"


def _split_nk_line(line: str) -> list[str]:
    return [part.strip() for part in line.replace(",", " ").split() if part.strip()]


def read_csv_nk(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    data_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        return rows
    first = _split_nk_line(data_lines[0])
    has_header = bool(first) and not _looks_float(first[0])
    if has_header:
        for row in csv.DictReader(data_lines):
            if not row:
                continue
            wavelength = row.get("wavelength_um") or row.get("wl") or row.get("lambda")
            n_value = row.get("n")
            k_value = row.get("k")
            if wavelength is None or n_value is None or k_value is None:
                continue
            rows.append(
                {
                    "wavelength_um": float(wavelength),
                    "n": float(n_value),
                    "k": float(k_value),
                }
            )
        return rows
    for line in data_lines:
        parts = _split_nk_line(line)
        if len(parts) < 3:
            continue
        rows.append(
            {
                "wavelength_um": float(parts[0]),
                "n": float(parts[1]),
                "k": float(parts[2]),
            }
        )
    return rows


def _looks_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_refractiveindexinfo_nk(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    in_data_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "data: |":
            in_data_block = True
            continue
        if not in_data_block or not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            if not line.startswith(" "):
                in_data_block = False
            continue
        try:
            rows.append(
                {
                    "wavelength_um": float(parts[0]),
                    "n": float(parts[1]),
                    "k": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def read_nk_rows(path: Path) -> list[dict[str, float]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_nk(path)
    if suffix in {".yml", ".yaml"}:
        return read_refractiveindexinfo_nk(path)
    return []


def summarize_nk_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "wavelength_min_um": None,
            "wavelength_max_um": None,
            "n_min": None,
            "n_max": None,
            "k_min": None,
            "k_max": None,
        }
    wavelengths = [row["wavelength_um"] for row in rows]
    ns = [row["n"] for row in rows]
    ks = [row["k"] for row in rows]
    return {
        "row_count": len(rows),
        "wavelength_min_um": min(wavelengths),
        "wavelength_max_um": max(wavelengths),
        "n_min": min(ns),
        "n_max": max(ns),
        "k_min": min(ks),
        "k_max": max(ks),
    }


def parse_required_wavelengths(raw: str | None) -> list[float]:
    if raw is None:
        return list(DEFAULT_REQUIRED_WAVELENGTHS_UM)
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values


def validate_nk_rows(
    rows: list[dict[str, float]],
    required_wavelengths_um: list[float],
) -> dict[str, Any]:
    issues: list[str] = []
    if not rows:
        return {
            "nk_table_valid": False,
            "issues": ["empty_or_unreadable_nk_table"],
            "input_monotonic_increasing": False,
            "duplicate_wavelength_count": 0,
            "nonpositive_n_count": 0,
            "negative_k_count": 0,
            "required_wavelengths_um": required_wavelengths_um,
            "covered_required_wavelengths_um": [],
            "missing_required_wavelengths_um": required_wavelengths_um,
            "sampled_nk": [],
        }
    wavelengths = [row["wavelength_um"] for row in rows]
    ns = [row["n"] for row in rows]
    ks = [row["k"] for row in rows]
    finite_count = sum(
        1
        for values in zip(wavelengths, ns, ks)
        if all(value == value and abs(value) != float("inf") for value in values)
    )
    if finite_count != len(rows):
        issues.append("nonfinite_wavelength_n_or_k")
    input_monotonic = all(
        right > left for left, right in zip(wavelengths, wavelengths[1:])
    )
    duplicate_count = len(wavelengths) - len(set(wavelengths))
    if duplicate_count:
        issues.append("duplicate_wavelength_rows")
    nonpositive_n_count = sum(1 for value in ns if value <= 0.0)
    if nonpositive_n_count:
        issues.append("nonpositive_refractive_index")
    negative_k_count = sum(1 for value in ks if value < 0.0)
    if negative_k_count:
        issues.append("negative_extinction_coefficient")

    sorted_rows = sorted(rows, key=lambda row: row["wavelength_um"])
    sorted_wavelengths = [row["wavelength_um"] for row in sorted_rows]
    wl_min = min(sorted_wavelengths)
    wl_max = max(sorted_wavelengths)
    covered = [
        wavelength
        for wavelength in required_wavelengths_um
        if wl_min <= wavelength <= wl_max
    ]
    missing = [
        wavelength
        for wavelength in required_wavelengths_um
        if wavelength < wl_min or wavelength > wl_max
    ]
    if missing:
        issues.append("required_wavelength_outside_table_range")
    sampled_nk = []
    if not duplicate_count and not nonpositive_n_count and not negative_k_count:
        sorted_n = [row["n"] for row in sorted_rows]
        sorted_k = [row["k"] for row in sorted_rows]
        for wavelength in covered:
            sampled_nk.append(
                {
                    "wavelength_um": wavelength,
                    "n": float(_interp_sorted(sorted_wavelengths, sorted_n, wavelength)),
                    "k": float(_interp_sorted(sorted_wavelengths, sorted_k, wavelength)),
                }
            )
    return {
        "nk_table_valid": not issues,
        "issues": issues,
        "input_monotonic_increasing": input_monotonic,
        "duplicate_wavelength_count": duplicate_count,
        "nonpositive_n_count": nonpositive_n_count,
        "negative_k_count": negative_k_count,
        "required_wavelengths_um": required_wavelengths_um,
        "covered_required_wavelengths_um": covered,
        "missing_required_wavelengths_um": missing,
        "sampled_nk": sampled_nk,
    }


def _interp_sorted(x_values: list[float], y_values: list[float], x_value: float) -> float:
    if x_value <= x_values[0]:
        return y_values[0]
    if x_value >= x_values[-1]:
        return y_values[-1]
    for left_index, right_index in zip(range(len(x_values) - 1), range(1, len(x_values))):
        x0 = x_values[left_index]
        x1 = x_values[right_index]
        if x0 <= x_value <= x1:
            t = (x_value - x0) / (x1 - x0)
            return y_values[left_index] * (1.0 - t) + y_values[right_index] * t
    return y_values[-1]


def build_evidence(
    stack_path: Path,
    required_wavelengths_um: list[float] | None = None,
) -> dict[str, Any]:
    required_wavelengths_um = list(
        DEFAULT_REQUIRED_WAVELENGTHS_UM if required_wavelengths_um is None else required_wavelengths_um
    )
    stack = read_json(stack_path)
    base_dir = stack_path.parent
    geometry = stack.get("geometry_um", {})
    missing_geometry = sorted(REQUIRED_GEOMETRY_KEYS - set(geometry))
    calibration = stack.get("calibration_status", {})
    geometry_measured = bool(
        calibration.get("geometry_measured", False)
        or stack.get("geometry_measured", False)
        or calibration.get("is_measured", False)
    )

    material_rows = []
    missing_tables = []
    invalid_tables = []
    coverage_failures = []
    non_measured_materials = []
    proxy_materials = []
    materials = stack.get("materials", {})
    for role, material in materials.items():
        table_value = material.get("nk_table")
        table_path = resolve_path(str(table_value), base_dir) if table_value else None
        table_exists = bool(table_path and table_path.exists())
        nk_rows = read_nk_rows(table_path) if table_path and table_exists else []
        nk_summary = summarize_nk_rows(nk_rows)
        nk_validation = (
            validate_nk_rows(nk_rows, required_wavelengths_um)
            if table_value
            else {
                "nk_table_valid": True,
                "issues": [],
                "required_wavelengths_um": required_wavelengths_um,
                "covered_required_wavelengths_um": [],
                "missing_required_wavelengths_um": [],
                "sampled_nk": [],
            }
        )
        source_type = classify_source(material, table_path or Path(""))
        measured = bool(material.get("measured", False))
        row = {
            "role": role,
            "nk_table": str(table_path) if table_path else "",
            "nk_table_exists": table_exists,
            "source_type": source_type,
            "measured": measured,
            "source": material.get("source", ""),
            "source_url": material.get("source_url", ""),
            "usage": material.get("usage", ""),
            **nk_summary,
            **nk_validation,
        }
        material_rows.append(row)
        if table_value and not table_exists:
            missing_tables.append(role)
        if table_value and table_exists and not nk_validation.get("nk_table_valid", False):
            invalid_tables.append(role)
        if table_value and nk_validation.get("missing_required_wavelengths_um"):
            coverage_failures.append(role)
        if source_type == "proxy":
            proxy_materials.append(role)
        if table_value and not measured:
            non_measured_materials.append(role)

    all_tables_exist = not missing_tables
    all_tables_valid = not invalid_tables
    all_required_wavelengths_covered = not coverage_failures
    all_materials_measured = bool(material_rows) and not non_measured_materials
    stack_measured = bool(calibration.get("is_measured", False) or stack.get("measured", False))
    is_measured = bool(stack_measured and geometry_measured and all_materials_measured)

    return {
        "schema": "image_sensor_optical_stack_evidence_v1",
        "stack_config": str(stack_path.resolve()),
        "stack_name": stack.get("name", ""),
        "description": stack.get("description", ""),
        "calibration_status": {
            "is_measured": is_measured,
            "stack_marked_measured": stack_measured,
            "geometry_measured": geometry_measured,
            "all_materials_measured": all_materials_measured,
            "mode": calibration.get("mode", "measured" if is_measured else "reference_proxy"),
            "note": calibration.get(
                "note",
                "Optical stack is not product measured unless geometry and every material n,k table are marked measured.",
            ),
        },
        "measured": is_measured,
        "geometry": {
            "values_um": geometry,
            "missing_required_keys": missing_geometry,
            "geometry_measured": geometry_measured,
        },
        "materials": material_rows,
        "summary": {
            "material_count": len(material_rows),
            "all_nk_tables_exist": all_tables_exist,
            "all_nk_tables_valid": all_tables_valid,
            "all_required_wavelengths_covered": all_required_wavelengths_covered,
            "required_wavelengths_um": required_wavelengths_um,
            "missing_nk_tables": missing_tables,
            "invalid_nk_tables": invalid_tables,
            "nk_coverage_failures": coverage_failures,
            "non_measured_materials": non_measured_materials,
            "proxy_materials": proxy_materials,
            "accuracy_ready": (
                is_measured
                and all_tables_exist
                and all_tables_valid
                and all_required_wavelengths_covered
                and not missing_geometry
            ),
        },
        "accuracy_notes": stack.get("accuracy_notes", []),
        "public_references": stack.get("public_references", []),
    }


def write_material_csv(path: Path, evidence: dict[str, Any]) -> None:
    rows = evidence.get("materials", [])
    fieldnames = [
        "role",
        "measured",
        "source_type",
        "nk_table_exists",
        "nk_table_valid",
        "row_count",
        "wavelength_min_um",
        "wavelength_max_um",
        "missing_required_wavelengths_um",
        "issues",
        "n_min",
        "n_max",
        "k_min",
        "k_max",
        "nk_table",
        "source_url",
        "usage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, evidence: dict[str, Any]) -> None:
    summary = evidence["summary"]
    lines = [
        "# Optical Stack Evidence",
        "",
        f"- Stack: `{evidence.get('stack_name', '')}`",
        f"- Measured accuracy-ready: `{evidence.get('measured', False)}`",
        f"- All n,k tables exist: `{summary.get('all_nk_tables_exist')}`",
        f"- All n,k tables valid: `{summary.get('all_nk_tables_valid')}`",
        f"- Required wavelengths covered: `{summary.get('all_required_wavelengths_covered')}`",
        f"- Non-measured materials: `{', '.join(summary.get('non_measured_materials', [])) or 'none'}`",
        f"- Proxy materials: `{', '.join(summary.get('proxy_materials', [])) or 'none'}`",
        "",
        "| Role | Measured | Source Type | Valid | Rows | Wavelength Range (um) | Missing Required wl (um) |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for material in evidence.get("materials", []):
        wl_min = material.get("wavelength_min_um")
        wl_max = material.get("wavelength_max_um")
        wl_range = "" if wl_min is None else f"{wl_min:g}-{wl_max:g}"
        missing = ",".join(str(value) for value in material.get("missing_required_wavelengths_um", []))
        lines.append(
            f"| {material.get('role')} | {material.get('measured')} | "
            f"{material.get('source_type')} | {material.get('nk_table_valid')} | "
            f"{material.get('row_count')} | {wl_range} | {missing} |"
        )
    lines.extend(
        [
            "",
            "This report is evidence for file plumbing and source classification. "
            "It is not a product accuracy claim unless geometry and every material "
            "n,k table are measured for the target sensor.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument(
        "--required-wavelengths-um",
        default=",".join(str(value) for value in DEFAULT_REQUIRED_WAVELENGTHS_UM),
        help="Comma-separated wavelengths that every nk_table must cover.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "optical_stack_evidence_reference")
    args = parser.parse_args()

    evidence = build_evidence(
        args.stack_config,
        required_wavelengths_um=parse_required_wavelengths(args.required_wavelengths_um),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "optical_stack_summary.json"
    material_csv_path = args.output_dir / "optical_stack_materials.csv"
    markdown_path = args.output_dir / "optical_stack_evidence.md"
    summary_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    write_material_csv(material_csv_path, evidence)
    write_markdown(markdown_path, evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
