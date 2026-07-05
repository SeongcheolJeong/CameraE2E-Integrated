#!/usr/bin/env python3
"""Convert GDS mask polygons into pixel_geometry_import_v1 JSON.

This is a geometry-injection bridge for the pixel workbench. It extracts 2D
mask footprints from selected GDS layers and emits the JSON format accepted by
meep_supercell_lut.py and meep_crosstalk_kernel.py through @file imports.

It does not preserve full mask signoff hierarchy, boolean intent, or 3D OCL
surface height. Use measured profilometry / process stack inputs for those.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CFA_COLORS = {"red", "green", "blue"}


@dataclass(frozen=True)
class LayerSpec:
    layer: int
    datatype: int | None
    role: str
    name: str


@dataclass(frozen=True)
class ExtractedPolygon:
    layer: int
    datatype: int
    points_um: tuple[tuple[float, float], ...]
    center_um: tuple[float, float]


def load_gdstk():
    try:
        import gdstk  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "gdstk is required for GDS import. Install it in the local environment with "
            "`.meep-env/bin/python -m pip install gdstk`."
        ) from error
    return gdstk


def parse_layer(value: Any, *, default_datatype: int | None = None) -> tuple[int, int | None]:
    if isinstance(value, int):
        return value, default_datatype
    if isinstance(value, dict):
        layer = int(value["layer"])
        datatype_value = value.get("datatype", default_datatype)
        datatype = None if datatype_value in {None, ""} else int(datatype_value)
        return layer, datatype
    text = str(value).strip()
    if not text:
        raise ValueError("Layer spec must not be empty")
    separator = "/" if "/" in text else ":" if ":" in text else None
    if separator:
        layer_text, datatype_text = text.split(separator, 1)
        return int(layer_text), int(datatype_text)
    return int(text), default_datatype


def load_map_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("--map-config must contain a JSON object")
    return payload


def layer_specs_from_config(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[LayerSpec], list[LayerSpec]]:
    ocl_specs: list[LayerSpec] = []
    cfa_specs: list[LayerSpec] = []
    for item in config.get("ocl_layers", []):
        if isinstance(item, dict):
            layer, datatype = parse_layer(item.get("layer", item), default_datatype=item.get("datatype"))
            lens_id = str(item.get("id", item.get("id_prefix", f"ocl_l{layer}")))
        else:
            layer, datatype = parse_layer(item)
            lens_id = f"ocl_l{layer}"
        ocl_specs.append(LayerSpec(layer=layer, datatype=datatype, role="ocl", name=lens_id))
    for item in config.get("cfa_layers", []):
        if not isinstance(item, dict):
            raise ValueError("cfa_layers entries must be objects with layer/datatype/color")
        layer, datatype = parse_layer(item.get("layer", item), default_datatype=item.get("datatype"))
        color = str(item.get("color", "")).lower()
        if color not in CFA_COLORS:
            raise ValueError(f"CFA layer color must be red, green, or blue: {item!r}")
        cfa_specs.append(LayerSpec(layer=layer, datatype=datatype, role="cfa", name=color))
    for value in args.ocl_layer:
        layer_text, lens_id = value.split("=", 1)
        layer, datatype = parse_layer(layer_text, default_datatype=0)
        ocl_specs.append(LayerSpec(layer=layer, datatype=datatype, role="ocl", name=lens_id))
    for value in args.cfa_layer:
        color, layer_text = value.split("=", 1)
        color = color.lower()
        if color not in CFA_COLORS:
            raise ValueError(f"CFA layer color must be red, green, or blue: {color!r}")
        layer, datatype = parse_layer(layer_text, default_datatype=0)
        cfa_specs.append(LayerSpec(layer=layer, datatype=datatype, role="cfa", name=color))
    return ocl_specs, cfa_specs


def matches_spec(polygon: Any, spec: LayerSpec) -> bool:
    return polygon.layer == spec.layer and (spec.datatype is None or polygon.datatype == spec.datatype)


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    area = 0.0
    for index, (x0, z0) in enumerate(points):
        x1, z1 = points[(index + 1) % len(points)]
        area += x0 * z1 - x1 * z0
    return 0.5 * area


def polygon_bbox(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    return min(xs), min(zs), max(xs), max(zs)


def merge_bboxes(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    return (
        min(item[0] for item in bboxes),
        min(item[1] for item in bboxes),
        max(item[2] for item in bboxes),
        max(item[3] for item in bboxes),
    )


def bbox_to_dict(bbox: tuple[float, float, float, float] | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    xmin, zmin, xmax, zmax = bbox
    return {
        "x_min": round(xmin, 9),
        "z_min": round(zmin, 9),
        "x_max": round(xmax, 9),
        "z_max": round(zmax, 9),
        "width": round(xmax - xmin, 9),
        "height": round(zmax - zmin, 9),
    }


def layer_key(layer: int, datatype: int | None) -> str:
    return f"{layer}/{datatype if datatype is not None else '*'}"


def polygon_record(item: ExtractedPolygon, *, role: str = "unmapped", name: str | None = None) -> dict[str, Any]:
    bbox = polygon_bbox(item.points_um)
    return {
        "role": role,
        "name": name,
        "layer": item.layer,
        "datatype": item.datatype,
        "layer_key": layer_key(item.layer, item.datatype),
        "point_count": len(item.points_um),
        "area_um2": round(abs(polygon_area(item.points_um)), 9),
        "center_um": {"x": round(item.center_um[0], 9), "z": round(item.center_um[1], 9)},
        "bbox_um": bbox_to_dict(bbox),
    }


def spec_matches(polygons: list[ExtractedPolygon], spec: LayerSpec) -> list[ExtractedPolygon]:
    return [item for item in polygons if matches_spec_proxy(item, spec)]


def build_layer_summary(polygons: list[ExtractedPolygon]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for item in polygons:
        key = layer_key(item.layer, item.datatype)
        record = summary.setdefault(
            key,
            {
                "layer": item.layer,
                "datatype": item.datatype,
                "polygon_count": 0,
                "total_area_um2": 0.0,
                "bbox_um": None,
            },
        )
        record["polygon_count"] += 1
        record["total_area_um2"] += abs(polygon_area(item.points_um))
        existing_bbox = record["bbox_um_raw"] if "bbox_um_raw" in record else None
        item_bbox = polygon_bbox(item.points_um)
        record["bbox_um_raw"] = item_bbox if existing_bbox is None else merge_bboxes([existing_bbox, item_bbox])
    for record in summary.values():
        record["total_area_um2"] = round(record["total_area_um2"], 9)
        record["bbox_um"] = bbox_to_dict(record.pop("bbox_um_raw", None))
    return dict(sorted(summary.items(), key=lambda pair: (pair[1]["layer"], pair[1]["datatype"])))


def build_validation_report(
    *,
    gds_path: Path,
    top_cell_name: str,
    polygons: list[ExtractedPolygon],
    ocl_specs: list[LayerSpec],
    cfa_specs: list[LayerSpec],
    payload: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    matched_ids: set[int] = set()
    matched_ocl: list[dict[str, Any]] = []
    matched_cfa: list[dict[str, Any]] = []
    configured_layers = [
        {
            "role": spec.role,
            "name": spec.name,
            "layer": spec.layer,
            "datatype": spec.datatype,
            "layer_key": layer_key(spec.layer, spec.datatype),
            "matched_polygon_count": 0,
        }
        for spec in [*ocl_specs, *cfa_specs]
    ]

    for index, spec in enumerate([*ocl_specs, *cfa_specs]):
        matches = spec_matches(polygons, spec)
        configured_layers[index]["matched_polygon_count"] = len(matches)
        if not matches:
            warnings.append(f"No polygons matched configured {spec.role} layer {layer_key(spec.layer, spec.datatype)}.")
            continue
        for item in matches:
            matched_ids.add(id(item))
            record = polygon_record(item, role=spec.role, name=spec.name)
            if spec.role == "ocl":
                matched_ocl.append(record)
            else:
                matched_cfa.append(record)

    if not ocl_specs:
        warnings.append("No OCL layer mapping was configured.")
    elif not matched_ocl:
        warnings.append("No OCL polygons were matched; optical lens aperture import will be missing.")
    if not cfa_specs:
        warnings.append("No CFA layer mapping was configured.")
    elif not matched_cfa:
        warnings.append("No CFA polygons were matched; color-filter aperture import will be missing.")
    if not matched_ocl and not matched_cfa:
        errors.append("No configured OCL/CFA layers matched any GDS polygon.")

    bbox = merge_bboxes([polygon_bbox(item.points_um) for item in polygons])
    ocl_bbox = merge_bboxes([polygon_bbox(item.points_um) for item in polygons if any(matches_spec_proxy(item, spec) for spec in ocl_specs)])
    cfa_bbox = merge_bboxes([polygon_bbox(item.points_um) for item in polygons if any(matches_spec_proxy(item, spec) for spec in cfa_specs)])
    if bbox:
        width_um = bbox[2] - bbox[0]
        height_um = bbox[3] - bbox[1]
        if width_um < 0.2 or height_um < 0.2:
            warnings.append("GDS bbox is very small for a pixel mask import; verify coordinate units and scale_to_um.")
        if width_um > 100.0 or height_um > 100.0:
            warnings.append("GDS bbox is very large for a single pixel/supercell import; verify coordinate units and top cell.")

    unmatched = [item for item in polygons if id(item) not in matched_ids]
    status = "FAIL" if errors else "CHECK" if warnings else "PASS"
    return {
        "schema": "gds_pixel_geometry_import_report_v1",
        "input_gds": str(gds_path),
        "top_cell": top_cell_name,
        "validation_status": status,
        "errors": errors,
        "warnings": warnings,
        "polygon_count": len(polygons),
        "matched_ocl_polygon_count": len(matched_ocl),
        "matched_cfa_polygon_count": len(matched_cfa),
        "unmapped_polygon_count": len(unmatched),
        "bbox_um": bbox_to_dict(bbox),
        "ocl_bbox_um": bbox_to_dict(ocl_bbox),
        "cfa_bbox_um": bbox_to_dict(cfa_bbox),
        "configured_layers": configured_layers,
        "layer_summary": build_layer_summary(polygons),
        "matched_ocl_polygons": matched_ocl,
        "matched_cfa_polygons": matched_cfa,
        "unmapped_polygons": [polygon_record(item) for item in unmatched],
        "converted_geometry_counts": {
            "ocl_polygon_count": len(payload.get("ocl_polygons", {})),
            "cfa_polygon_count": payload.get("gds_import", {}).get("matched_cfa_polygon_count", 0),
        },
        "notes": [
            "Preview and report validate 2D mask footprints only.",
            "This report does not prove optical/electrical accuracy without measured stack geometry and n,k calibration.",
        ],
    }


def svg_polygon_points(
    points: tuple[tuple[float, float], ...],
    *,
    bbox: tuple[float, float, float, float],
    scale: float,
    margin: float,
) -> str:
    xmin, _zmin, _xmax, zmax = bbox
    values = []
    for x_um, z_um in points:
        x = margin + (x_um - xmin) * scale
        y = margin + (zmax - z_um) * scale
        values.append(f"{x:.2f},{y:.2f}")
    return " ".join(values)


def preview_style(item: ExtractedPolygon, ocl_specs: list[LayerSpec], cfa_specs: list[LayerSpec]) -> tuple[str, str, str]:
    for spec in cfa_specs:
        if matches_spec_proxy(item, spec):
            fill = {"red": "#ef4444", "green": "#22c55e", "blue": "#3b82f6"}.get(spec.name, "#94a3b8")
            return spec.name.upper(), fill, "#e2e8f0"
    for spec in ocl_specs:
        if matches_spec_proxy(item, spec):
            return f"OCL {spec.name}", "#38bdf833", "#38bdf8"
    return "UNMAPPED", "#64748b22", "#64748b"


def write_preview_svg(
    path: Path,
    *,
    top_cell_name: str,
    polygons: list[ExtractedPolygon],
    ocl_specs: list[LayerSpec],
    cfa_specs: list[LayerSpec],
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas_w = 760.0
    canvas_h = 520.0
    margin = 42.0
    bbox = merge_bboxes([polygon_bbox(item.points_um) for item in polygons])
    if bbox is None:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="180" viewBox="0 0 760 180">'
            '<rect width="760" height="180" fill="#07131f"/>'
            '<text x="32" y="92" fill="#e2e8f0" font-family="Inter, Arial" font-size="18">No polygons found</text>'
            "</svg>\n",
            encoding="utf-8",
        )
        return
    width = max(bbox[2] - bbox[0], 1.0e-6)
    height = max(bbox[3] - bbox[1], 1.0e-6)
    scale = min((canvas_w - 2 * margin) / width, (canvas_h - 2 * margin) / height)
    status = str(report.get("validation_status", "CHECK"))
    status_color = {"PASS": "#22c55e", "CHECK": "#facc15", "FAIL": "#ef4444"}.get(status, "#facc15")
    title = html.escape(f"GDS import preview: {top_cell_name}")
    subtitle = html.escape(
        f"{report.get('polygon_count', 0)} polygons, "
        f"OCL {report.get('matched_ocl_polygon_count', 0)}, "
        f"CFA {report.get('matched_cfa_polygon_count', 0)}"
    )
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="520" viewBox="0 0 760 520">',
        '<rect width="760" height="520" rx="10" fill="#07131f"/>',
        f'<text x="28" y="34" fill="#e2e8f0" font-family="Inter, Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="28" y="58" fill="#94a3b8" font-family="Inter, Arial" font-size="13">{subtitle}</text>',
        f'<rect x="650" y="24" width="76" height="26" rx="6" fill="{status_color}22" stroke="{status_color}"/>',
        f'<text x="688" y="42" fill="{status_color}" text-anchor="middle" font-family="Inter, Arial" font-size="12" font-weight="700">{status}</text>',
        f'<rect x="{margin - 8:.2f}" y="{margin - 8:.2f}" width="{width * scale + 16:.2f}" height="{height * scale + 16:.2f}" fill="#0f2233" stroke="#31506a" rx="8"/>',
    ]
    for item in polygons:
        label, fill, stroke = preview_style(item, ocl_specs, cfa_specs)
        points = svg_polygon_points(item.points_um, bbox=bbox, scale=scale, margin=margin)
        elements.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="2">'
            f'<title>{html.escape(label)} layer {item.layer}/{item.datatype}</title></polygon>'
        )
    legend_y = 462
    for index, (label, color) in enumerate(
        [("OCL", "#38bdf8"), ("Red CFA", "#ef4444"), ("Green CFA", "#22c55e"), ("Blue CFA", "#3b82f6"), ("Unmapped", "#64748b")]
    ):
        x = 28 + index * 132
        elements.append(f'<rect x="{x}" y="{legend_y}" width="18" height="18" rx="3" fill="{color}"/>')
        elements.append(
            f'<text x="{x + 26}" y="{legend_y + 14}" fill="#cbd5e1" font-family="Inter, Arial" font-size="12">{html.escape(label)}</text>'
        )
    elements.append(
        f'<text x="28" y="500" fill="#94a3b8" font-family="Inter, Arial" font-size="12">'
        f'bbox {width:.3f} um x {height:.3f} um, x/y mapped to x/z</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def clean_points(
    raw_points: Any,
    *,
    scale_to_um: float,
    origin_x_um: float,
    origin_z_um: float,
    max_points: int,
) -> ExtractedPolygon:
    points: list[tuple[float, float]] = []
    for raw_x, raw_y in raw_points:
        x = (float(raw_x) - origin_x_um) * scale_to_um
        z = (float(raw_y) - origin_z_um) * scale_to_um
        if not math.isfinite(x) or not math.isfinite(z):
            raise ValueError("GDS polygon contains a non-finite coordinate")
        points.append((x, z))
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise ValueError("GDS polygon contains fewer than three unique points")
    if len(points) > max_points:
        raise ValueError(f"GDS polygon has {len(points)} points; max supported is {max_points}")
    polygon = tuple(points)
    if abs(polygon_area(polygon)) < 1e-9:
        raise ValueError("GDS polygon area is too small")
    xs = [point[0] for point in polygon]
    zs = [point[1] for point in polygon]
    center = ((min(xs) + max(xs)) * 0.5, (min(zs) + max(zs)) * 0.5)
    return ExtractedPolygon(layer=0, datatype=0, points_um=polygon, center_um=center)


def localize_polygon(
    polygon: ExtractedPolygon,
    *,
    localize: str,
    center_override: tuple[float, float] | None = None,
) -> list[list[float]]:
    if center_override is not None:
        cx, cz = center_override
    elif localize == "origin":
        cx, cz = 0.0, 0.0
    elif localize == "centroid":
        cx = sum(point[0] for point in polygon.points_um) / len(polygon.points_um)
        cz = sum(point[1] for point in polygon.points_um) / len(polygon.points_um)
    else:
        cx, cz = polygon.center_um
    return [[round(x - cx, 9), round(z - cz, 9)] for x, z in polygon.points_um]


def stable_polygon_sort_key(item: ExtractedPolygon) -> tuple[float, float, float]:
    return (item.center_um[1], item.center_um[0], abs(polygon_area(item.points_um)))


def read_polygons(
    gds_path: Path,
    *,
    top_cell: str | None,
    scale_to_um: float,
    origin_x_um: float,
    origin_z_um: float,
    max_points: int,
) -> tuple[str, list[ExtractedPolygon]]:
    gdstk = load_gdstk()
    library = gdstk.read_gds(gds_path)
    if top_cell:
        cells = [cell for cell in library.cells if cell.name == top_cell]
        if not cells:
            raise ValueError(f"Top cell {top_cell!r} not found in {gds_path}")
        cell = cells[0]
    else:
        top_cells = library.top_level()
        if not top_cells:
            raise ValueError(f"No top-level cell found in {gds_path}")
        cell = top_cells[0]
    extracted: list[ExtractedPolygon] = []
    for polygon in cell.get_polygons(apply_repetitions=True):
        clean = clean_points(
            polygon.points,
            scale_to_um=scale_to_um,
            origin_x_um=origin_x_um,
            origin_z_um=origin_z_um,
            max_points=max_points,
        )
        extracted.append(
            ExtractedPolygon(
                layer=int(polygon.layer),
                datatype=int(polygon.datatype),
                points_um=clean.points_um,
                center_um=clean.center_um,
            )
        )
    return cell.name, extracted


def build_geometry_payload(
    *,
    gds_path: Path,
    top_cell_name: str,
    polygons: list[ExtractedPolygon],
    ocl_specs: list[LayerSpec],
    cfa_specs: list[LayerSpec],
    cfa_background: str,
    localize: str,
    cfa_cell_pitch_um: float | None,
    cfa_grid_origin_x_um: float,
    cfa_grid_origin_z_um: float,
    map_config_path: Path | None,
    scale_to_um: float,
    origin_x_um: float,
    origin_z_um: float,
) -> dict[str, Any]:
    if localize not in {"bbox-center", "centroid", "origin"}:
        raise ValueError("--localize must be bbox-center, centroid, or origin")
    if cfa_background not in {"nearest", "passivation", "air"}:
        raise ValueError("--cfa-background must be nearest, passivation, or air")

    ocl_polygons: dict[str, list[list[float]]] = {}
    for spec in ocl_specs:
        matches = sorted([item for item in polygons if item.layer == spec.layer and (spec.datatype is None or item.datatype == spec.datatype)], key=stable_polygon_sort_key)
        for index, item in enumerate(matches):
            lens_id = spec.name if len(matches) == 1 else f"{spec.name}_{index}"
            ocl_polygons[lens_id] = localize_polygon(item, localize=localize)

    cfa_payload: dict[str, Any] = {"background": cfa_background}
    for spec in cfa_specs:
        matches = sorted([item for item in polygons if matches_spec_proxy(item, spec)], key=stable_polygon_sort_key)
        if not matches:
            continue
        if cfa_cell_pitch_um is None:
            if len(matches) > 1:
                raise ValueError(
                    f"CFA layer {spec.layer}/{spec.datatype} has {len(matches)} polygons. "
                    "Set cfa_cell_pitch_um in the map config to export cell-specific CFA polygons."
                )
            cfa_payload[spec.name] = {
                "points": localize_polygon(matches[0], localize=localize),
                "source": f"GDS {gds_path.name} layer {spec.layer}/{spec.datatype if spec.datatype is not None else '*'}",
            }
            continue
        cells = cfa_payload.setdefault("cells", [])
        for index, item in enumerate(matches):
            ix = int(round((item.center_um[0] - cfa_grid_origin_x_um) / cfa_cell_pitch_um))
            iz = int(round((item.center_um[1] - cfa_grid_origin_z_um) / cfa_cell_pitch_um))
            cell_center = (
                cfa_grid_origin_x_um + ix * cfa_cell_pitch_um,
                cfa_grid_origin_z_um + iz * cfa_cell_pitch_um,
            )
            cells.append(
                {
                    "id": f"{spec.name}_{ix}_{iz}_{index}",
                    "color": spec.name,
                    "ix": ix,
                    "iz": iz,
                    "points": localize_polygon(item, localize=localize, center_override=cell_center),
                    "source": f"GDS {gds_path.name} layer {spec.layer}/{spec.datatype if spec.datatype is not None else '*'}",
                }
            )

    if not ocl_polygons and len(cfa_payload) == 1:
        raise ValueError("No OCL or CFA polygons matched the configured layers")

    return {
        "schema": "pixel_geometry_import_v1",
        "units": "um",
        "source": f"GDS import from {gds_path.name}",
        "gds_import": {
            "input_gds": str(gds_path),
            "top_cell": top_cell_name,
            "map_config": str(map_config_path) if map_config_path else None,
            "scale_to_um": scale_to_um,
            "origin_um": {"x": origin_x_um, "z": origin_z_um},
            "localize": localize,
            "cfa_cell_pitch_um": cfa_cell_pitch_um,
            "cfa_grid_origin_um": {"x": cfa_grid_origin_x_um, "z": cfa_grid_origin_z_um},
            "matched_ocl_polygon_count": len(ocl_polygons),
            "matched_cfa_polygon_count": len(cfa_payload.get("cells", []))
            + sum(1 for color in CFA_COLORS if color in cfa_payload),
        },
        "notes": [
            "GDS coordinates are mapped as x/y -> x/z in microns.",
            "This export contains 2D mask footprints only; measured OCL surface maps and n,k tables remain separate inputs.",
        ],
        "ocl_polygons": ocl_polygons,
        "cfa_polygons": cfa_payload,
    }


def matches_spec_proxy(item: ExtractedPolygon, spec: LayerSpec) -> bool:
    return item.layer == spec.layer and (spec.datatype is None or item.datatype == spec.datatype)


def write_reference_gds(path: Path) -> None:
    gdstk = load_gdstk()
    path.parent.mkdir(parents=True, exist_ok=True)
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("PIXEL_IMPORT_REF")
    cell.add(
        gdstk.Polygon(
            [(-1.32, -1.30), (1.25, -1.14), (1.30, 1.25), (-1.18, 1.34)],
            layer=10,
            datatype=0,
        )
    )
    cell.add(
        gdstk.Polygon(
            [(-0.64, -0.60), (0.62, -0.48), (0.55, 0.62), (-0.61, 0.52)],
            layer=11,
            datatype=0,
        )
    )
    cell.add(gdstk.Polygon([(-0.58, -0.58), (0.555, -0.58), (0.58, 0.555), (-0.555, 0.58)], layer=20, datatype=0))
    cell.add(gdstk.Polygon([(-0.58, -0.58), (0.58, -0.58), (0.58, 0.58), (-0.58, 0.58)], layer=21, datatype=0))
    cell.add(gdstk.Polygon([(-0.555, -0.58), (0.58, -0.555), (0.555, 0.58), (-0.58, 0.555)], layer=22, datatype=0))
    library.write_gds(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_gds", nargs="?", type=Path, help="Input GDS file.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output pixel_geometry_import_v1 JSON path.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional validation report JSON path.")
    parser.add_argument("--preview-svg", type=Path, default=None, help="Optional SVG preview of mapped GDS polygons.")
    parser.add_argument("--map-config", type=Path, default=None, help="Layer mapping JSON config.")
    parser.add_argument("--top-cell", default=None, help="Top cell name. Defaults to the first top-level cell.")
    parser.add_argument("--scale-to-um", type=float, default=None, help="Coordinate scale applied after origin subtraction.")
    parser.add_argument("--origin-x-um", type=float, default=None)
    parser.add_argument("--origin-z-um", type=float, default=None)
    parser.add_argument("--localize", choices=("bbox-center", "centroid", "origin"), default=None)
    parser.add_argument("--cfa-background", choices=("nearest", "passivation", "air"), default=None)
    parser.add_argument("--cfa-cell-pitch-um", type=float, default=None)
    parser.add_argument("--cfa-grid-origin-x-um", type=float, default=None)
    parser.add_argument("--cfa-grid-origin-z-um", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--ocl-layer",
        action="append",
        default=[],
        help="Additional OCL layer mapping in layer/datatype=lens_id form. Can be repeated.",
    )
    parser.add_argument(
        "--cfa-layer",
        action="append",
        default=[],
        help="Additional CFA layer mapping in color=layer/datatype form. Can be repeated.",
    )
    parser.add_argument("--write-reference-gds", type=Path, default=None, help="Write a small reference GDS and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_reference_gds is not None:
        write_reference_gds(args.write_reference_gds)
        print(json.dumps({"wrote": str(args.write_reference_gds)}, indent=2))
        return
    if args.input_gds is None:
        raise SystemExit("input_gds is required unless --write-reference-gds is used")
    if args.output_json is None:
        raise SystemExit("--output-json is required")
    config = load_map_config(args.map_config)
    ocl_specs, cfa_specs = layer_specs_from_config(config, args)
    if not ocl_specs and not cfa_specs:
        raise SystemExit("No OCL/CFA layer mappings configured. Use --map-config or --ocl-layer/--cfa-layer.")

    top_cell = args.top_cell or config.get("top_cell")
    scale_to_um = float(args.scale_to_um if args.scale_to_um is not None else config.get("scale_to_um", 1.0))
    if not math.isfinite(scale_to_um) or scale_to_um <= 0:
        raise ValueError("--scale-to-um must be positive")
    origin = config.get("origin_um", {}) if isinstance(config.get("origin_um"), dict) else {}
    cfa_grid_origin = config.get("cfa_grid_origin_um", {}) if isinstance(config.get("cfa_grid_origin_um"), dict) else {}
    origin_x_um = float(args.origin_x_um if args.origin_x_um is not None else origin.get("x", 0.0))
    origin_z_um = float(args.origin_z_um if args.origin_z_um is not None else origin.get("z", 0.0))
    localize = str(args.localize or config.get("localize", "bbox-center"))
    cfa_background = str(args.cfa_background or config.get("cfa_background", "passivation"))
    cfa_cell_pitch_um = args.cfa_cell_pitch_um if args.cfa_cell_pitch_um is not None else config.get("cfa_cell_pitch_um")
    cfa_cell_pitch = None if cfa_cell_pitch_um in {None, ""} else float(cfa_cell_pitch_um)
    if cfa_cell_pitch is not None and cfa_cell_pitch <= 0:
        raise ValueError("cfa_cell_pitch_um must be positive")
    cfa_grid_origin_x_um = float(
        args.cfa_grid_origin_x_um if args.cfa_grid_origin_x_um is not None else cfa_grid_origin.get("x", 0.0)
    )
    cfa_grid_origin_z_um = float(
        args.cfa_grid_origin_z_um if args.cfa_grid_origin_z_um is not None else cfa_grid_origin.get("z", 0.0)
    )
    max_points = int(args.max_points if args.max_points is not None else config.get("max_points", 24))
    if max_points < 3 or max_points > 256:
        raise ValueError("--max-points must be between 3 and 256")

    top_cell_name, polygons = read_polygons(
        args.input_gds,
        top_cell=top_cell,
        scale_to_um=scale_to_um,
        origin_x_um=origin_x_um,
        origin_z_um=origin_z_um,
        max_points=max_points,
    )
    payload = build_geometry_payload(
        gds_path=args.input_gds,
        top_cell_name=top_cell_name,
        polygons=polygons,
        ocl_specs=ocl_specs,
        cfa_specs=cfa_specs,
        cfa_background=cfa_background,
        localize=localize,
        cfa_cell_pitch_um=cfa_cell_pitch,
        cfa_grid_origin_x_um=cfa_grid_origin_x_um,
        cfa_grid_origin_z_um=cfa_grid_origin_z_um,
        map_config_path=args.map_config,
        scale_to_um=scale_to_um,
        origin_x_um=origin_x_um,
        origin_z_um=origin_z_um,
    )
    report = build_validation_report(
        gds_path=args.input_gds,
        top_cell_name=top_cell_name,
        polygons=polygons,
        ocl_specs=ocl_specs,
        cfa_specs=cfa_specs,
        payload=payload,
    )
    payload["gds_import"]["validation_status"] = report["validation_status"]
    payload["gds_import"]["validation_warning_count"] = len(report["warnings"])
    payload["gds_import"]["validation_error_count"] = len(report["errors"])
    payload["gds_import"]["bbox_um"] = report["bbox_um"]
    payload["gds_import"]["ocl_bbox_um"] = report["ocl_bbox_um"]
    payload["gds_import"]["cfa_bbox_um"] = report["cfa_bbox_um"]
    payload["gds_import_validation"] = {
        "status": report["validation_status"],
        "warnings": report["warnings"],
        "errors": report["errors"],
        "polygon_count": report["polygon_count"],
        "layer_summary": report["layer_summary"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.preview_svg is not None:
        write_preview_svg(
            args.preview_svg,
            top_cell_name=top_cell_name,
            polygons=polygons,
            ocl_specs=ocl_specs,
            cfa_specs=cfa_specs,
            report=report,
        )
    print(
        json.dumps(
            {
                "schema": "gds_pixel_geometry_import_result_v1",
                "output_json": str(args.output_json),
                "report_json": str(args.report_json) if args.report_json else None,
                "preview_svg": str(args.preview_svg) if args.preview_svg else None,
                "top_cell": top_cell_name,
                "validation_status": report["validation_status"],
                "warning_count": len(report["warnings"]),
                "ocl_polygon_count": len(payload["ocl_polygons"]),
                "cfa_polygon_count": payload["gds_import"]["matched_cfa_polygon_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - CLI should report actionable conversion errors.
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
