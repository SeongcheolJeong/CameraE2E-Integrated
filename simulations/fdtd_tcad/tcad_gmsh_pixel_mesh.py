#!/usr/bin/env python3
"""Generate open-source Gmsh meshes for proxy image-sensor TCAD.

Coordinates are written in centimeters because the DEVSIM physics helpers use
cm-based semiconductor units.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gmsh

from measured_tcad_profile import load_measured_profile


@dataclass(frozen=True)
class PixelMeshConfig:
    width_um: float = 1.4
    depth_um: float = 2.8
    z_width_um: float = 1.4
    split_gap_um: float = 0.04
    mesh_um: float = 0.18
    fine_mesh_um: float = 0.06
    include_fd_contact: bool = False
    include_tg_contact: bool = False
    include_tg_oxide: bool = False
    include_dti_oxide: bool = False
    transfer_gate_x_min_um: float = -0.18
    transfer_gate_x_max_um: float = 0.18
    transfer_gate_oxide_thickness_um: float = 0.006
    floating_diffusion_x_min_um: float = 0.45
    floating_diffusion_x_max_um: float = 0.62
    contact_gap_um: float = 0.005
    dti_left_x_min_um: float = -0.70
    dti_left_x_max_um: float = -0.64
    dti_right_x_min_um: float = 0.64
    dti_right_x_max_um: float = 0.70
    dti_depth_min_um: float = 0.0
    dti_depth_max_um: float = 3.0

    @property
    def width_cm(self) -> float:
        return self.width_um * 1.0e-4

    @property
    def depth_cm(self) -> float:
        return self.depth_um * 1.0e-4

    @property
    def z_width_cm(self) -> float:
        return self.z_width_um * 1.0e-4

    @property
    def gap_half_cm(self) -> float:
        return 0.5 * self.split_gap_um * 1.0e-4

    @property
    def mesh_cm(self) -> float:
        return self.mesh_um * 1.0e-4

    @property
    def fine_mesh_cm(self) -> float:
        return self.fine_mesh_um * 1.0e-4

    @property
    def transfer_gate_x_min_cm(self) -> float:
        return self.transfer_gate_x_min_um * 1.0e-4

    @property
    def transfer_gate_x_max_cm(self) -> float:
        return self.transfer_gate_x_max_um * 1.0e-4

    @property
    def transfer_gate_oxide_thickness_cm(self) -> float:
        return self.transfer_gate_oxide_thickness_um * 1.0e-4

    @property
    def floating_diffusion_x_min_cm(self) -> float:
        return self.floating_diffusion_x_min_um * 1.0e-4

    @property
    def floating_diffusion_x_max_cm(self) -> float:
        return self.floating_diffusion_x_max_um * 1.0e-4

    @property
    def contact_gap_cm(self) -> float:
        return self.contact_gap_um * 1.0e-4

    @property
    def dti_left_x_min_cm(self) -> float:
        return self.dti_left_x_min_um * 1.0e-4

    @property
    def dti_left_x_max_cm(self) -> float:
        return self.dti_left_x_max_um * 1.0e-4

    @property
    def dti_right_x_min_cm(self) -> float:
        return self.dti_right_x_min_um * 1.0e-4

    @property
    def dti_right_x_max_cm(self) -> float:
        return self.dti_right_x_max_um * 1.0e-4

    @property
    def dti_depth_min_cm(self) -> float:
        return self.dti_depth_min_um * 1.0e-4

    @property
    def dti_depth_max_cm(self) -> float:
        return self.dti_depth_max_um * 1.0e-4


def _physical(dim: int, tags: list[int], name: str) -> int:
    group = gmsh.model.addPhysicalGroup(dim, tags)
    gmsh.model.setPhysicalName(dim, group, name)
    return group


def generate_2d(config: PixelMeshConfig, output: Path) -> dict:
    if config.include_dti_oxide:
        return generate_2d_resolved_oxide(config, output)
    if config.include_tg_oxide:
        return generate_2d_resolved_tg_oxide(config, output)

    gmsh.initialize()
    try:
        gmsh.model.add("split_pixel_2d")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        xmin = -0.5 * config.width_cm
        xmax = 0.5 * config.width_cm
        ymin = 0.0
        ymax = config.depth_cm
        gap = config.gap_half_cm
        lc = config.mesh_cm
        lf = config.fine_mesh_cm

        p_bl = gmsh.model.geo.addPoint(xmin, ymax, 0.0, lc)
        p_gl = gmsh.model.geo.addPoint(-gap, ymax, 0.0, lf)
        p_gr = gmsh.model.geo.addPoint(gap, ymax, 0.0, lf)
        p_br = gmsh.model.geo.addPoint(xmax, ymax, 0.0, lc)
        p_tr = gmsh.model.geo.addPoint(xmax, ymin, 0.0, lc)
        p_tl = gmsh.model.geo.addPoint(xmin, ymin, 0.0, lc)

        bottom_left = gmsh.model.geo.addLine(p_bl, p_gl)
        bottom_gap = gmsh.model.geo.addLine(p_gl, p_gr)
        bottom_right = gmsh.model.geo.addLine(p_gr, p_br)
        right = gmsh.model.geo.addLine(p_br, p_tr)
        top_lines: list[int] = []
        fd_lines: list[int] = []
        tg_lines: list[int] = []
        tg_insulated_lines: list[int] = []
        if config.include_fd_contact or config.include_tg_contact:
            tg_min = max(xmin, min(xmax, config.transfer_gate_x_min_cm))
            tg_max = max(xmin, min(xmax, config.transfer_gate_x_max_cm))
            fd_min = max(xmin, min(xmax, config.floating_diffusion_x_min_cm))
            fd_max = max(xmin, min(xmax, config.floating_diffusion_x_max_cm))
            tg_gap = min(config.contact_gap_cm, max((tg_max - tg_min) * 0.2, 0.0))
            tg_contact_min = tg_min + tg_gap
            tg_contact_max = tg_max - tg_gap
            fd_gap = min(config.contact_gap_cm, max((fd_max - fd_min) * 0.2, 0.0))
            fd_contact_min = fd_min + fd_gap
            fd_contact_max = fd_max - fd_gap
            split_points = sorted(
                {
                    xmin,
                    xmax,
                    tg_min,
                    tg_contact_min,
                    tg_contact_max,
                    tg_max,
                    fd_min,
                    fd_contact_min,
                    fd_contact_max,
                    fd_max,
                },
                reverse=True,
            )
            top_point_tags = []
            for x_value in split_points:
                if abs(x_value - xmax) <= 1.0e-18:
                    top_point_tags.append(p_tr)
                elif abs(x_value - xmin) <= 1.0e-18:
                    top_point_tags.append(p_tl)
                else:
                    top_point_tags.append(
                        gmsh.model.geo.addPoint(x_value, ymin, 0.0, lf if tg_min <= x_value <= fd_max else lc)
                    )
            top_for_loop = []
            for left_tag, right_tag, x_left, x_right in zip(
                top_point_tags,
                top_point_tags[1:],
                split_points,
                split_points[1:],
            ):
                line = gmsh.model.geo.addLine(left_tag, right_tag)
                top_for_loop.append(line)
                x_mid = 0.5 * (x_left + x_right)
                if config.include_fd_contact and fd_contact_min <= x_mid <= fd_contact_max:
                    fd_lines.append(line)
                elif config.include_tg_contact and tg_contact_min <= x_mid <= tg_contact_max:
                    tg_lines.append(line)
                elif tg_min <= x_mid <= tg_max:
                    tg_insulated_lines.append(line)
                elif fd_min <= x_mid <= fd_max:
                    tg_insulated_lines.append(line)
                else:
                    top_lines.append(line)
        else:
            top = gmsh.model.geo.addLine(p_tr, p_tl)
            top_lines = [top]
            top_for_loop = [top]
        left = gmsh.model.geo.addLine(p_tl, p_bl)
        loop = gmsh.model.geo.addCurveLoop(
            [bottom_left, bottom_gap, bottom_right, right, *top_for_loop, left]
        )
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        _physical(2, [surface], "silicon")
        _physical(1, top_lines, "anode")
        if tg_lines:
            _physical(1, tg_lines, "transfer_gate")
        if fd_lines:
            _physical(1, fd_lines, "floating_diffusion")
        _physical(1, [bottom_left], "cathode_left")
        _physical(1, [bottom_right], "cathode_right")
        _physical(1, [bottom_gap, left, right, *tg_insulated_lines], "insulated")

        gmsh.model.mesh.generate(2)
        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output))
        return {
            "dimension": 2,
            "mesh": str(output),
            "region_physical": "silicon",
            "contacts": ["anode", "cathode_left", "cathode_right"]
            + (["transfer_gate"] if tg_lines else [])
            + (["floating_diffusion"] if fd_lines else []),
            "config": asdict(config),
        }
    finally:
        gmsh.finalize()


def _overlaps(a_min: float, a_max: float, b_min: float, b_max: float, tol: float) -> bool:
    return max(a_min, b_min) <= min(a_max, b_max) + tol


def _contains(value: float, low: float, high: float, tol: float) -> bool:
    return low - tol <= value <= high + tol


def _valid_interval(low: float, high: float, tol: float) -> bool:
    return high > low + tol


def _clamp_interval(low: float, high: float, limit_low: float, limit_high: float) -> tuple[float, float]:
    return max(limit_low, min(limit_high, low)), max(limit_low, min(limit_high, high))


def generate_2d_resolved_oxide(config: PixelMeshConfig, output: Path) -> dict:
    """Generate a 2D mesh with resolved oxide DTI/BDTI trenches.

    The silicon region is cut by side oxide rectangles.  Optional TG oxide is
    also included above the silicon surface, sharing the same DEVSIM oxide
    region and silicon_oxide interface group.
    """

    gmsh.initialize()
    try:
        gmsh.model.add("split_pixel_2d_resolved_oxide")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", config.fine_mesh_cm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.mesh_cm)

        xmin = -0.5 * config.width_cm
        xmax = 0.5 * config.width_cm
        ymin = 0.0
        ymax = config.depth_cm
        gap = config.gap_half_cm
        tol = max(config.fine_mesh_cm * 0.25, 1.0e-12)

        oxide_specs: list[dict[str, float | str]] = []
        dti_specs: list[dict[str, float | str]] = []

        fd_min = max(xmin, min(xmax, config.floating_diffusion_x_min_cm))
        fd_max = max(xmin, min(xmax, config.floating_diffusion_x_max_cm))
        fd_gap = min(config.contact_gap_cm, max((fd_max - fd_min) * 0.2, 0.0))
        fd_contact_min = fd_min + fd_gap
        fd_contact_max = fd_max - fd_gap

        tg_min = max(xmin, min(xmax, config.transfer_gate_x_min_cm))
        tg_max = max(xmin, min(xmax, config.transfer_gate_x_max_cm))
        tg_gap = min(config.contact_gap_cm, max((tg_max - tg_min) * 0.2, 0.0))
        tg_contact_min = tg_min + tg_gap
        tg_contact_max = tg_max - tg_gap

        dti_y0, dti_y1 = _clamp_interval(config.dti_depth_min_cm, config.dti_depth_max_cm, ymin, ymax)
        for name, x0_raw, x1_raw in (
            ("left_dti_oxide", config.dti_left_x_min_cm, config.dti_left_x_max_cm),
            ("right_dti_oxide", config.dti_right_x_min_cm, config.dti_right_x_max_cm),
        ):
            x0, x1 = _clamp_interval(x0_raw, x1_raw, xmin, xmax)
            if not (_valid_interval(x0, x1, tol) and _valid_interval(dti_y0, dti_y1, tol)):
                continue
            dti_specs.append(
                {
                    "name": name,
                    "x_min": x0,
                    "x_max": x1,
                    "y_min": dti_y0,
                    "y_max": dti_y1,
                }
            )

        x_breaks = {
            xmin,
            xmax,
            -gap,
            gap,
            tg_min,
            tg_contact_min,
            tg_contact_max,
            tg_max,
            fd_min,
            fd_contact_min,
            fd_contact_max,
            fd_max,
        }
        for spec in dti_specs:
            x_breaks.add(float(spec["x_min"]))
            x_breaks.add(float(spec["x_max"]))
        x_values = sorted(value for value in x_breaks if xmin - tol <= value <= xmax + tol)
        silicon_seeds: list[int] = []
        for x0, x1 in zip(x_values, x_values[1:]):
            if _valid_interval(x0, x1, tol):
                silicon_seeds.append(gmsh.model.occ.addRectangle(x0, ymin, 0.0, x1 - x0, config.depth_cm))
        if not silicon_seeds:
            raise RuntimeError("resolved oxide mesh did not create silicon seed strips")

        dti_tools: list[int] = []
        for spec in dti_specs:
            tag = gmsh.model.occ.addRectangle(
                float(spec["x_min"]),
                float(spec["y_min"]),
                0.0,
                float(spec["x_max"]) - float(spec["x_min"]),
                float(spec["y_max"]) - float(spec["y_min"]),
            )
            spec["surface_tag"] = tag
            dti_tools.append(tag)
            oxide_specs.append(spec)

        if dti_tools:
            gmsh.model.occ.cut(
                [(2, tag) for tag in silicon_seeds],
                [(2, tag) for tag in dti_tools],
                removeObject=True,
                removeTool=False,
            )

        tg_spec: dict[str, float | str] | None = None
        if config.include_tg_oxide:
            if tg_max <= tg_min:
                raise ValueError("transfer_gate_x_max_um must be greater than transfer_gate_x_min_um")
            tg_guard = min(config.contact_gap_cm, max((tg_max - tg_min) * 0.1, 0.0))
            oxide_min = tg_min + tg_guard
            oxide_max = tg_max - tg_guard
            if oxide_max <= oxide_min:
                oxide_min = tg_min
                oxide_max = tg_max
            tox = max(config.transfer_gate_oxide_thickness_cm, 1.0e-8)
            tag = gmsh.model.occ.addRectangle(oxide_min, -tox, 0.0, oxide_max - oxide_min, tox)
            tg_spec = {
                "name": "transfer_gate_oxide",
                "x_min": oxide_min,
                "x_max": oxide_max,
                "y_min": -tox,
                "y_max": 0.0,
                "surface_tag": tag,
                "edge_guard_um": tg_guard * 1.0e4,
            }
            oxide_specs.append(tg_spec)

        gmsh.model.occ.synchronize()

        silicon_surfaces: list[int] = []
        oxide_surfaces: list[int] = []
        for _dim, tag in gmsh.model.getEntities(2):
            x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(2, tag)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            matched_oxide = False
            for spec in oxide_specs:
                if _contains(cx, float(spec["x_min"]), float(spec["x_max"]), tol) and _contains(
                    cy, float(spec["y_min"]), float(spec["y_max"]), tol
                ):
                    oxide_surfaces.append(tag)
                    matched_oxide = True
                    break
            if not matched_oxide and y0 >= ymin - tol and y1 <= ymax + tol:
                silicon_surfaces.append(tag)

        if not silicon_surfaces:
            raise RuntimeError("resolved oxide mesh did not create a silicon surface")
        _physical(2, silicon_surfaces, "silicon")
        if oxide_surfaces:
            _physical(2, oxide_surfaces, "oxide")

        curve_roles: dict[int, set[str]] = {}
        for role, surfaces_for_role in (("silicon", silicon_surfaces), ("oxide", oxide_surfaces)):
            for surface in surfaces_for_role:
                for dim, curve in gmsh.model.getBoundary(
                    [(2, surface)], oriented=False, recursive=False
                ):
                    if dim == 1:
                        curve_roles.setdefault(int(curve), set()).add(role)

        def in_any_dti(x_mid: float, y_mid: float) -> bool:
            for spec in oxide_specs:
                if not str(spec["name"]).endswith("dti_oxide"):
                    continue
                if _contains(x_mid, float(spec["x_min"]), float(spec["x_max"]), tol) and _contains(
                    y_mid, float(spec["y_min"]), float(spec["y_max"]), tol
                ):
                    return True
            return False

        interface_lines: list[int] = []
        transfer_gate_lines: list[int] = []
        anode_lines: list[int] = []
        fd_lines: list[int] = []
        tg_lines: list[int] = []
        cathode_left_lines: list[int] = []
        cathode_right_lines: list[int] = []
        classified: set[int] = set()

        for _dim, tag in gmsh.model.getEntities(1):
            x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(1, tag)
            x_mid = 0.5 * (x0 + x1)
            y_mid = 0.5 * (y0 + y1)
            horizontal = abs(y0 - y1) <= tol
            roles = curve_roles.get(int(tag), set())

            if tg_spec and horizontal and abs(y_mid - float(tg_spec["y_min"])) <= tol:
                if "oxide" in roles and _contains(
                    x_mid, float(tg_spec["x_min"]), float(tg_spec["x_max"]), tol
                ):
                    transfer_gate_lines.append(tag)
                    classified.add(tag)
                    continue
            if {"silicon", "oxide"}.issubset(roles):
                interface_lines.append(tag)
                classified.add(tag)
                continue
            if horizontal and abs(y_mid) <= tol and "silicon" in roles and "oxide" not in roles and not in_any_dti(x_mid, y_mid):
                if config.include_fd_contact and fd_contact_min <= x_mid <= fd_contact_max:
                    fd_lines.append(tag)
                    classified.add(tag)
                    continue
                if config.include_tg_contact and not config.include_tg_oxide and tg_contact_min <= x_mid <= tg_contact_max:
                    tg_lines.append(tag)
                    classified.add(tag)
                    continue
                anode_lines.append(tag)
                classified.add(tag)
                continue
            if horizontal and abs(y_mid - ymax) <= tol and "silicon" in roles and "oxide" not in roles and not in_any_dti(x_mid, y_mid):
                if x_mid < -gap:
                    cathode_left_lines.append(tag)
                    classified.add(tag)
                    continue
                if x_mid > gap:
                    cathode_right_lines.append(tag)
                    classified.add(tag)
                    continue

        if anode_lines:
            _physical(1, anode_lines, "anode")
        if tg_lines:
            _physical(1, tg_lines, "transfer_gate")
        if transfer_gate_lines:
            _physical(1, transfer_gate_lines, "transfer_gate")
        if fd_lines:
            _physical(1, fd_lines, "floating_diffusion")
        if cathode_left_lines:
            _physical(1, cathode_left_lines, "cathode_left")
        if cathode_right_lines:
            _physical(1, cathode_right_lines, "cathode_right")
        if interface_lines:
            _physical(1, interface_lines, "silicon_oxide_interface")

        insulated_lines = [tag for _dim, tag in gmsh.model.getEntities(1) if tag not in classified]
        if insulated_lines:
            _physical(1, insulated_lines, "insulated")

        gmsh.model.mesh.generate(2)
        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output))
        contacts = ["anode", "cathode_left", "cathode_right"]
        if tg_lines or transfer_gate_lines:
            contacts.append("transfer_gate")
        if fd_lines:
            contacts.append("floating_diffusion")
        return {
            "dimension": 2,
            "mesh": str(output),
            "region_physical": "silicon",
            "oxide_region_physical": "oxide" if oxide_surfaces else "",
            "contacts": contacts,
            "interfaces": ["silicon_oxide_interface"] if interface_lines else [],
            "config": asdict(config),
            "resolved_dti_oxide": [
                {
                    "name": str(spec["name"]),
                    "x_min_um": float(spec["x_min"]) * 1.0e4,
                    "x_max_um": float(spec["x_max"]) * 1.0e4,
                    "depth_min_um": float(spec["y_min"]) * 1.0e4,
                    "depth_max_um": float(spec["y_max"]) * 1.0e4,
                }
                for spec in oxide_specs
                if str(spec["name"]).endswith("dti_oxide")
            ],
            "resolved_tg_oxide": (
                {
                    "x_min_um": float(tg_spec["x_min"]) * 1.0e4,
                    "x_max_um": float(tg_spec["x_max"]) * 1.0e4,
                    "edge_guard_um": float(tg_spec["edge_guard_um"]),
                    "oxide_thickness_um": config.transfer_gate_oxide_thickness_um,
                }
                if tg_spec
                else None
            ),
            "surface_counts": {
                "silicon": len(silicon_surfaces),
                "oxide": len(oxide_surfaces),
            },
            "line_counts": {
                "anode": len(anode_lines),
                "cathode_left": len(cathode_left_lines),
                "cathode_right": len(cathode_right_lines),
                "transfer_gate": len(tg_lines) + len(transfer_gate_lines),
                "floating_diffusion": len(fd_lines),
                "silicon_oxide_interface": len(interface_lines),
                "insulated": len(insulated_lines),
            },
        }
    finally:
        gmsh.finalize()


def generate_2d_resolved_tg_oxide(config: PixelMeshConfig, output: Path) -> dict:
    gmsh.initialize()
    try:
        gmsh.model.add("split_pixel_2d_resolved_tg_oxide")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        xmin = -0.5 * config.width_cm
        xmax = 0.5 * config.width_cm
        ymin = 0.0
        ymax = config.depth_cm
        tox = max(config.transfer_gate_oxide_thickness_cm, 1.0e-8)
        gap = config.gap_half_cm
        lc = config.mesh_cm
        lf = min(config.fine_mesh_cm, max(tox * 0.5, 1.0e-8))
        tg_min = max(xmin, min(xmax, config.transfer_gate_x_min_cm))
        tg_max = max(xmin, min(xmax, config.transfer_gate_x_max_cm))
        if tg_max <= tg_min:
            raise ValueError("transfer_gate_x_max_um must be greater than transfer_gate_x_min_um")
        tg_guard = min(config.contact_gap_cm, max((tg_max - tg_min) * 0.1, 0.0))
        oxide_min = tg_min + tg_guard
        oxide_max = tg_max - tg_guard
        if oxide_max <= oxide_min:
            oxide_min = tg_min
            oxide_max = tg_max
        fd_min = max(xmin, min(xmax, config.floating_diffusion_x_min_cm))
        fd_max = max(xmin, min(xmax, config.floating_diffusion_x_max_cm))
        fd_gap = min(config.contact_gap_cm, max((fd_max - fd_min) * 0.2, 0.0))
        fd_contact_min = fd_min + fd_gap
        fd_contact_max = fd_max - fd_gap

        top_breaks = sorted(
            {
                xmin,
                xmax,
                tg_min,
                oxide_min,
                oxide_max,
                tg_max,
                fd_min,
                fd_contact_min,
                fd_contact_max,
                fd_max,
            }
        )
        top_points = [
            gmsh.model.geo.addPoint(x_value, ymin, 0.0, lf if tg_min <= x_value <= tg_max else lc)
            for x_value in top_breaks
        ]
        top_lines: list[int] = []
        anode_lines: list[int] = []
        fd_lines: list[int] = []
        si_oxide_lines: list[int] = []
        insulated_top_lines: list[int] = []
        for left_tag, right_tag, x_left, x_right in zip(
            top_points,
            top_points[1:],
            top_breaks,
            top_breaks[1:],
        ):
            line = gmsh.model.geo.addLine(left_tag, right_tag)
            top_lines.append(line)
            x_mid = 0.5 * (x_left + x_right)
            if oxide_min <= x_mid <= oxide_max:
                si_oxide_lines.append(line)
            elif tg_min <= x_mid <= tg_max:
                insulated_top_lines.append(line)
            elif config.include_fd_contact and fd_contact_min <= x_mid <= fd_contact_max:
                fd_lines.append(line)
            elif config.include_fd_contact and fd_min <= x_mid <= fd_max:
                insulated_top_lines.append(line)
            else:
                anode_lines.append(line)

        p_br = gmsh.model.geo.addPoint(xmax, ymax, 0.0, lc)
        p_gap_r = gmsh.model.geo.addPoint(gap, ymax, 0.0, lf)
        p_gap_l = gmsh.model.geo.addPoint(-gap, ymax, 0.0, lf)
        p_bl = gmsh.model.geo.addPoint(xmin, ymax, 0.0, lc)
        right = gmsh.model.geo.addLine(top_points[-1], p_br)
        bottom_right = gmsh.model.geo.addLine(p_br, p_gap_r)
        bottom_gap = gmsh.model.geo.addLine(p_gap_r, p_gap_l)
        bottom_left = gmsh.model.geo.addLine(p_gap_l, p_bl)
        left = gmsh.model.geo.addLine(p_bl, top_points[0])
        silicon_loop = gmsh.model.geo.addCurveLoop(
            [*top_lines, right, bottom_right, bottom_gap, bottom_left, left]
        )
        silicon_surface = gmsh.model.geo.addPlaneSurface([silicon_loop])

        p_ox_tl = gmsh.model.geo.addPoint(oxide_min, -tox, 0.0, lf)
        p_ox_tr = gmsh.model.geo.addPoint(oxide_max, -tox, 0.0, lf)
        oxide_top = gmsh.model.geo.addLine(p_ox_tl, p_ox_tr)
        oxide_right = gmsh.model.geo.addLine(p_ox_tr, top_points[top_breaks.index(oxide_max)])
        oxide_left = gmsh.model.geo.addLine(top_points[top_breaks.index(oxide_min)], p_ox_tl)
        oxide_loop = gmsh.model.geo.addCurveLoop(
            [oxide_top, oxide_right, *[-line for line in reversed(si_oxide_lines)], oxide_left]
        )
        oxide_surface = gmsh.model.geo.addPlaneSurface([oxide_loop])
        gmsh.model.geo.synchronize()

        _physical(2, [silicon_surface], "silicon")
        _physical(2, [oxide_surface], "oxide")
        if anode_lines:
            _physical(1, anode_lines, "anode")
        _physical(1, si_oxide_lines, "silicon_oxide_interface")
        _physical(1, [oxide_top], "transfer_gate")
        if fd_lines:
            _physical(1, fd_lines, "floating_diffusion")
        _physical(1, [bottom_left], "cathode_left")
        _physical(1, [bottom_right], "cathode_right")
        insulated_lines = [bottom_gap, left, right, oxide_left, oxide_right, *insulated_top_lines]
        _physical(1, insulated_lines, "insulated")

        gmsh.model.mesh.generate(2)
        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output))
        return {
            "dimension": 2,
            "mesh": str(output),
            "region_physical": "silicon",
            "oxide_region_physical": "oxide",
            "contacts": ["anode", "cathode_left", "cathode_right", "transfer_gate"]
            + (["floating_diffusion"] if fd_lines else []),
            "interfaces": ["silicon_oxide_interface"],
            "config": asdict(config),
            "resolved_tg_oxide": {
                "x_min_um": oxide_min * 1.0e4,
                "x_max_um": oxide_max * 1.0e4,
                "edge_guard_um": tg_guard * 1.0e4,
            },
        }
    finally:
        gmsh.finalize()


def _surface_tags_for_volumes(volumes: list[int]) -> list[int]:
    surfaces = set()
    for volume in volumes:
        for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False):
            if dim == 2:
                surfaces.add(tag)
    return sorted(surfaces)


def generate_3d(config: PixelMeshConfig, output: Path) -> dict:
    gmsh.initialize()
    try:
        gmsh.model.add("split_pixel_3d")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", config.fine_mesh_cm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.mesh_cm)

        xmin = -0.5 * config.width_cm
        xmax = 0.5 * config.width_cm
        zmin = -0.5 * config.z_width_cm
        zmax = 0.5 * config.z_width_cm
        gap = config.gap_half_cm
        depth = config.depth_cm

        left = gmsh.model.occ.addBox(xmin, 0.0, zmin, -gap - xmin, depth, zmax - zmin)
        mid = gmsh.model.occ.addBox(-gap, 0.0, zmin, 2.0 * gap, depth, zmax - zmin)
        right = gmsh.model.occ.addBox(gap, 0.0, zmin, xmax - gap, depth, zmax - zmin)
        gmsh.model.occ.fragment([(3, left), (3, mid), (3, right)], [])
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        surfaces = _surface_tags_for_volumes(volumes)
        top_surfaces: list[int] = []
        bottom_left_surfaces: list[int] = []
        bottom_right_surfaces: list[int] = []
        insulated_surfaces: list[int] = []
        tol = max(config.fine_mesh_cm, 1.0e-12)
        for surface in surfaces:
            xmin_s, ymin_s, zmin_s, xmax_s, ymax_s, zmax_s = gmsh.model.getBoundingBox(2, surface)
            x_center = 0.5 * (xmin_s + xmax_s)
            if abs(ymin_s) <= tol and abs(ymax_s) <= tol:
                top_surfaces.append(surface)
            elif abs(ymin_s - depth) <= tol and abs(ymax_s - depth) <= tol:
                if x_center < -gap:
                    bottom_left_surfaces.append(surface)
                elif x_center > gap:
                    bottom_right_surfaces.append(surface)
                else:
                    insulated_surfaces.append(surface)
            else:
                insulated_surfaces.append(surface)

        _physical(3, volumes, "silicon")
        _physical(2, top_surfaces, "anode")
        _physical(2, bottom_left_surfaces, "cathode_left")
        _physical(2, bottom_right_surfaces, "cathode_right")
        if insulated_surfaces:
            _physical(2, insulated_surfaces, "insulated")

        gmsh.model.mesh.generate(3)
        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output))
        return {
            "dimension": 3,
            "mesh": str(output),
            "region_physical": "silicon",
            "contacts": ["anode", "cathode_left", "cathode_right"],
            "config": asdict(config),
            "volume_count": len(volumes),
            "surface_counts": {
                "anode": len(top_surfaces),
                "cathode_left": len(bottom_left_surfaces),
                "cathode_right": len(bottom_right_surfaces),
                "insulated": len(insulated_surfaces),
            },
        }
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", choices=("2", "3", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gmsh_pixel_mesh"))
    parser.add_argument("--measured-profile", type=Path, default=None)
    parser.add_argument("--width-um", type=float, default=None)
    parser.add_argument("--depth-um", type=float, default=None)
    parser.add_argument("--z-width-um", type=float, default=None)
    parser.add_argument("--split-gap-um", type=float, default=None)
    parser.add_argument("--mesh-um", type=float, default=None)
    parser.add_argument("--fine-mesh-um", type=float, default=None)
    parser.add_argument("--contact-gap-um", type=float, default=None)
    parser.add_argument("--include-fd-contact", action="store_true")
    parser.add_argument("--include-tg-contact", action="store_true")
    parser.add_argument("--include-tg-oxide", action="store_true")
    parser.add_argument(
        "--include-dti-oxide",
        action="store_true",
        help="Resolve side DTI/BDTI as oxide regions instead of only doping features.",
    )
    parser.add_argument("--dti-width-um", type=float, default=None)
    parser.add_argument("--dti-depth-um", type=float, default=None)
    args = parser.parse_args()

    defaults = PixelMeshConfig()
    geometry = {}
    geometry_source = "built_in_proxy_defaults"
    if args.measured_profile:
        profile = load_measured_profile(args.measured_profile)
        geometry = profile.geometry
        geometry_source = str(profile.path)

    def pick(cli_value: float | None, geometry_key: str, default: float) -> float:
        if cli_value is not None:
            return cli_value
        if geometry_key in geometry:
            return float(geometry[geometry_key])
        return default

    width_um = pick(args.width_um, "width_um", defaults.width_um)
    depth_um = pick(args.depth_um, "depth_um", defaults.depth_um)
    z_width_um = pick(args.z_width_um, "z_width_um", defaults.z_width_um)
    split_gap_um = pick(args.split_gap_um, "split_gap_um", defaults.split_gap_um)
    dti_width_um = (
        args.dti_width_um
        if args.dti_width_um is not None
        else float(geometry.get("dti_width_um", defaults.dti_right_x_max_um - defaults.dti_right_x_min_um))
    )
    bdti_geometry = geometry.get("bdti", {}) if isinstance(geometry.get("bdti", {}), dict) else {}
    if bool(bdti_geometry.get("enabled", False)):
        dti_left_x_min_um = float(bdti_geometry.get("x_left_min_um", -0.5 * width_um))
        dti_left_x_max_um = float(bdti_geometry.get("x_left_max_um", -0.5 * width_um + dti_width_um))
        dti_right_x_min_um = float(bdti_geometry.get("x_right_min_um", 0.5 * width_um - dti_width_um))
        dti_right_x_max_um = float(bdti_geometry.get("x_right_max_um", 0.5 * width_um))
        dti_depth_min_um = float(bdti_geometry.get("depth_min_um", 0.0))
        dti_depth_max_um = float(bdti_geometry.get("depth_max_um", depth_um))
    else:
        dti_left_x_min_um = -0.5 * width_um
        dti_left_x_max_um = -0.5 * width_um + dti_width_um
        dti_right_x_min_um = 0.5 * width_um - dti_width_um
        dti_right_x_max_um = 0.5 * width_um
        dti_depth_min_um = 0.0
        dti_depth_max_um = depth_um
    if args.dti_depth_um is not None:
        dti_depth_max_um = args.dti_depth_um

    config = PixelMeshConfig(
        width_um=width_um,
        depth_um=depth_um,
        z_width_um=z_width_um,
        split_gap_um=split_gap_um,
        mesh_um=args.mesh_um if args.mesh_um is not None else defaults.mesh_um,
        fine_mesh_um=args.fine_mesh_um if args.fine_mesh_um is not None else defaults.fine_mesh_um,
        contact_gap_um=args.contact_gap_um
        if args.contact_gap_um is not None
        else defaults.contact_gap_um,
        include_fd_contact=args.include_fd_contact,
        include_tg_contact=args.include_tg_contact,
        include_tg_oxide=args.include_tg_oxide,
        include_dti_oxide=args.include_dti_oxide,
        transfer_gate_x_min_um=float(
            geometry.get("transfer_gate", {}).get("x_min_um", defaults.transfer_gate_x_min_um)
        ),
        transfer_gate_x_max_um=float(
            geometry.get("transfer_gate", {}).get("x_max_um", defaults.transfer_gate_x_max_um)
        ),
        transfer_gate_oxide_thickness_um=float(
            geometry.get("transfer_gate", {}).get(
                "oxide_thickness_um", defaults.transfer_gate_oxide_thickness_um
            )
        ),
        floating_diffusion_x_min_um=float(
            geometry.get("floating_diffusion", {}).get("x_min_um", defaults.floating_diffusion_x_min_um)
        ),
        floating_diffusion_x_max_um=float(
            geometry.get("floating_diffusion", {}).get("x_max_um", defaults.floating_diffusion_x_max_um)
        ),
        dti_left_x_min_um=dti_left_x_min_um,
        dti_left_x_max_um=dti_left_x_max_um,
        dti_right_x_min_um=dti_right_x_min_um,
        dti_right_x_max_um=dti_right_x_max_um,
        dti_depth_min_um=dti_depth_min_um,
        dti_depth_max_um=dti_depth_max_um,
    )
    metadata = []
    if args.dimension in ("2", "both"):
        metadata.append(generate_2d(config, args.output_dir / "split_pixel_2d.msh"))
    if args.dimension in ("3", "both"):
        metadata.append(generate_3d(config, args.output_dir / "split_pixel_3d.msh"))
    output = {
        "schema": "tcad_gmsh_pixel_mesh_v1",
        "geometry_source": geometry_source,
        "meshes": metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mesh_metadata.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
