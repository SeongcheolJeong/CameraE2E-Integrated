#!/usr/bin/env python3
"""Local API server for the Image Sensor Pixel Workbench.

The React UI is static, so it cannot launch Meep/DEVSIM directly. This server
keeps the deployment simple: serve the existing files and expose a local-only
API for solver smoke jobs and KPI parsing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from cad_template_variant_create import SCALAR_OVERRIDE_FIELDS, TOPOLOGY_OVERRIDE_FIELDS, sanitize_id
from cad_template_review import extract_fcstd_parameters, freecad_status, open_command, validate_freecad_library
from camera_system_quantitative_evidence import build_evidence as build_quantitative_evidence
from camera_system_suite_export_validate import parse_float_list as parse_export_float_list
from camera_system_suite_export_validate import validate_export_package
from pixel_cad_template_library import OclBlock, TemplateSpec, validate_library, write_template


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".meep-env" / "bin" / "python"
TCAD_PYTHON = ROOT / ".tcad-env" / "bin" / "python"
RUN_ROOT = ROOT / "runs" / "ui_solver_tests"
CAD_TEMPLATE_ROOT = ROOT / "runs" / "pixel_cad_template_library_reference"
CAD_TEMPLATE_MANIFEST = CAD_TEMPLATE_ROOT / "template_library_manifest.json"
CAD_TEMPLATE_VALIDATION_REPORT = CAD_TEMPLATE_ROOT / "cad_template_validation_report.json"
CAD_TEMPLATE_FREECAD_VALIDATION_REPORT = CAD_TEMPLATE_ROOT / "freecad_validation_report.json"
FCSTD_WORKING_COPY_ROOT = ROOT / "runs" / "fcstd_working_copies"
BASE_STACK_CONFIG = ROOT / "configs" / "sensor_stack_proxy_1p4um.json"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
SOLVER_MODES = {"split-pd-1x1", "ocl-2x2", "ocl-3x3", "ocl-layout"}
SPLIT_MODES = {"dual-x", "dual-z", "quad"}
COLLECTION_MODES = {"auto", "pixel", "split-pd"}
COLOR_CHANNELS = {"red", "green", "blue"}
UNIFORM_COLOR_CFA_PATTERNS = {f"uniform_{color}": color for color in COLOR_CHANNELS}
SHIELD_MODES = {"off", "edge", "pdaf_left", "pdaf_right", "pdaf_pair"}
CFA_PATTERNS = {"uniform", "bayer", "quad", "nona"}
STACK_OVERRIDE_ALLOWLIST = {
    "geometry_um.pitch",
    "geometry_um.lens_height",
    "geometry_um.lens_edge_gap",
    "geometry_um.cfa_thickness",
    "geometry_um.passivation_thickness",
    "geometry_um.si_thickness",
    "shield.mode",
    "shield.mask_edge_width_um",
    "materials.lens",
}
CAD_OPENABLE_ARTIFACTS = {
    "step",
    "brep",
    "fcstd",
    "mesh",
    "geometry_import",
    "parameters",
    "variant_source",
    "assumption_ledger",
    "footprint_preview",
    "tcad_bridge_report",
    "tcad_derived_config",
    "devsim_import_summary",
    "devsim_dd_summary",
    "devsim_split_currents",
    "devsim_split_currents_plot",
    "devsim_node_maps_plot",
    "freecad_validation_report",
}
FREECAD_PREFERRED_ARTIFACTS = {"step", "brep", "fcstd"}
CAD_VARIANT_QUICK_FIELDS = {
    "pitch_um",
    "lens_height_um",
    "lens_edge_gap_um",
    "cfa_thickness_um",
    "passivation_thickness_um",
    "si_thickness_um",
    "dti_width_um",
    "dti_depth_um",
    "pd_margin_um",
    "pd_depth_min_um",
    "pd_depth_max_um",
}
CAD_TOPOLOGY_CHANGE_FIELDS = set(TOPOLOGY_OVERRIDE_FIELDS) | {"ocl_blocks"}
CAD_METADATA_CHANGE_FIELDS = {"template_id", "label", "notes"}
CAD_BASE_TEMPLATE_TOPOLOGY_PRESETS = {
    "bayer_1x1_3x3": {
        "label": "Bayer 1x1 neighborhood",
        "nx": 3,
        "nz": 3,
        "cfa_pattern": "bayer",
        "split_mode": "none",
        "shield_mode": "off",
        "ocl": "unit",
    },
    "quad_2x2_ocl": {
        "label": "Quad Bayer 2x2 OCL",
        "nx": 4,
        "nz": 4,
        "cfa_pattern": "quad",
        "split_mode": "none",
        "shield_mode": "off",
        "ocl": "block_2x2",
    },
    "quad_2x2_ocl_3x3_neighborhood": {
        "label": "Quad Bayer 2x2 OCL 3x3 neighborhood",
        "nx": 6,
        "nz": 6,
        "cfa_pattern": "quad",
        "split_mode": "none",
        "shield_mode": "off",
        "ocl": "block_2x2",
    },
    "quad_2x2_ocl_5x5_crosstalk": {
        "label": "Quad Bayer 2x2 OCL 5x5 crosstalk",
        "nx": 10,
        "nz": 10,
        "cfa_pattern": "quad",
        "split_mode": "none",
        "shield_mode": "off",
        "ocl": "block_2x2",
    },
    "nona_3x3_ocl": {
        "label": "Nona 3x3 OCL",
        "nx": 6,
        "nz": 6,
        "cfa_pattern": "nona",
        "split_mode": "none",
        "shield_mode": "off",
        "ocl": "block_3x3",
    },
    "qpd_2x2": {
        "label": "QPD 2x2 split photodiode",
        "nx": 2,
        "nz": 2,
        "cfa_pattern": "uniform_green",
        "split_mode": "quad",
        "shield_mode": "pdaf_pair",
        "ocl": "single_full",
    },
    "dual_pd_x_1x1": {
        "label": "Dual-PD X 1x1",
        "nx": 1,
        "nz": 1,
        "cfa_pattern": "uniform_green",
        "split_mode": "dual-x",
        "shield_mode": "off",
        "ocl": "single_full",
    },
    "dual_pd_z_1x1": {
        "label": "Dual-PD Z 1x1",
        "nx": 1,
        "nz": 1,
        "cfa_pattern": "uniform_green",
        "split_mode": "dual-z",
        "shield_mode": "off",
        "ocl": "single_full",
    },
    "pdaf_dual_x_pair": {
        "label": "Dual-PD X PDAF pair",
        "nx": 2,
        "nz": 1,
        "cfa_pattern": "uniform_green",
        "split_mode": "dual-x",
        "shield_mode": "pdaf_pair",
        "ocl": "unit",
    },
    "mixed_1x1_2x2_3x3_boundary": {
        "label": "Mixed 1x1 / 2x2 / 3x3 OCL boundary",
        "nx": 5,
        "nz": 3,
        "cfa_pattern": "nona",
        "split_mode": "none",
        "shield_mode": "off",
        "ocl": "mixed_boundary",
    },
}
CAD_STARTER_TEMPLATE_SET = (
    {
        "template_id": "bayer_1x1_3x3",
        "role": "baseline_image_pixel",
        "label": "Bayer 1x1 image-pixel neighborhood",
    },
    {
        "template_id": "quad_2x2_ocl",
        "role": "quad_binning_ocl",
        "label": "Quad 2x2 OCL supercell",
    },
    {
        "template_id": "nona_3x3_ocl",
        "role": "nona_binning_ocl",
        "label": "Nona 3x3 OCL supercell",
    },
    {
        "template_id": "mixed_1x1_2x2_3x3_boundary",
        "role": "ocl_transition_boundary",
        "label": "Mixed 1x1/2x2/3x3 OCL boundary",
    },
    {
        "template_id": "dual_pd_x_1x1",
        "role": "dual_pd_cra_x",
        "label": "Dual-PD x-split pixel",
    },
    {
        "template_id": "dual_pd_z_1x1",
        "role": "dual_pd_cra_z",
        "label": "Dual-PD z-split pixel",
    },
    {
        "template_id": "pdaf_dual_x_shield_pair",
        "role": "pdaf_shield_pair",
        "label": "Dual-PD PDAF shield pair",
    },
    {
        "template_id": "qpd_split_pd_2x2",
        "role": "qpd_with_shield",
        "label": "QPD 2x2 split photodiode with shield",
    },
    {
        "template_id": "qpd_split_pd_no_shield_2x2",
        "role": "qpd_no_shield_control",
        "label": "QPD 2x2 no-shield control",
    },
)
CAD_STARTER_TEMPLATE_IDS = {item["template_id"] for item in CAD_STARTER_TEMPLATE_SET}
CAD_TEMPLATE_PROTECTED_STACK_OVERRIDE_KEYS = {"shield.mode"}
CAD_TEMPLATE_PROTECTED_STACK_OVERRIDE_PREFIXES = ("geometry_um.",)


def is_cad_template_protected_stack_override(key: Any) -> bool:
    text = str(key)
    return text in CAD_TEMPLATE_PROTECTED_STACK_OVERRIDE_KEYS or any(
        text.startswith(prefix) for prefix in CAD_TEMPLATE_PROTECTED_STACK_OVERRIDE_PREFIXES
    )


def split_cad_template_stack_overrides(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw is None or raw == "":
        return {}, {}
    if not isinstance(raw, dict):
        raise ValueError("solver.stack_overrides must be an object")
    allowed: dict[str, Any] = {}
    protected: dict[str, Any] = {}
    for key, value in raw.items():
        text = str(key)
        if is_cad_template_protected_stack_override(text):
            protected[text] = value
        else:
            allowed[text] = value
    return allowed, protected


EXAMPLES: dict[str, dict[str, Any]] = {
    "bayer1x1_smoke": {
        "id": "bayer1x1_smoke",
        "label": "Bayer + 1x1 split-PD smoke",
        "preset_hint": "bayer_1x1",
        "mode": "split-pd-1x1",
        "split_mode": "dual-x",
        "wavelengths_nm": "550",
        "cases": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
        "resolution": 18,
        "after_source_time": 2,
        "pml_um": 0.45,
        "description": "Fast 1x1 reference run for center/edge response and split-PD phase.",
    },
    "ocl2x2_smoke": {
        "id": "ocl2x2_smoke",
        "label": "Quad Bayer + 2x2 OCL smoke",
        "preset_hint": "quad_2x2",
        "mode": "ocl-2x2",
        "wavelengths_nm": "550",
        "cases": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
        "resolution": 18,
        "after_source_time": 2,
        "pml_um": 0.45,
        "description": "Fast Meep run for center/edge CRA response on a 2x2 OCL supercell.",
    },
    "ocl3x3_smoke": {
        "id": "ocl3x3_smoke",
        "label": "Nona 3x3 + 3x3 OCL smoke",
        "preset_hint": "nona_3x3",
        "mode": "ocl-3x3",
        "wavelengths_nm": "550",
        "cases": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
        "resolution": 18,
        "after_source_time": 2,
        "pml_um": 0.45,
        "description": "Fast Meep run for grouped 3x3 OCL binning response.",
    },
    "split_pd_quad_smoke": {
        "id": "split_pd_quad_smoke",
        "label": "Split-PD / QPD smoke",
        "preset_hint": "quad_qpd",
        "mode": "split-pd-1x1",
        "split_mode": "quad",
        "wavelengths_nm": "550",
        "cases": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
        "resolution": 18,
        "after_source_time": 2,
        "pml_um": 0.45,
        "description": "Fast Meep run for split-PD quadrant response and phase-balance KPI wiring.",
    },
    "split_pd_dualz_smoke": {
        "id": "split_pd_dualz_smoke",
        "label": "Split-PD dual-z smoke",
        "preset_hint": "sparse_pdaf",
        "mode": "split-pd-1x1",
        "split_mode": "dual-z",
        "wavelengths_nm": "550",
        "cases": "center:0:0:0:0:0:0,edge20z:0:20:0:1:0:0",
        "resolution": 18,
        "after_source_time": 2,
        "pml_um": 0.45,
        "description": "Fast split-PD run for top/bottom phase sensitivity versus CRA-z.",
    },
}


TEST_SUITES: dict[str, dict[str, Any]] = {
    "pattern_baseline": {
        "id": "pattern_baseline",
        "label": "Pattern Topology Baseline",
        "category": "Pattern",
        "priority": 1,
        "runtime_hint": "1-3 min smoke",
        "decision_goal": "Compare Bayer, 2x2 OCL, 3x3 OCL, and QPD response trends before deeper convergence.",
        "recommended_tier": "smoke",
        "tiers": ["smoke", "trend", "quantitative"],
        "cases": [
            {
                "id": "bayer_1x1",
                "label": "Bayer 1x1 reference",
                "runner": "example",
                "example_id": "bayer1x1_smoke",
                "tiers": ["smoke"],
                "design_factors": {"pattern": "Bayer", "ocl": "1x1", "readout": "full-resolution"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "convergence_report_card"],
                "decision_notes": ["Reference anchor for center/edge response and split signal sign."],
            },
            {
                "id": "quad_2x2",
                "label": "Quad Bayer + 2x2 OCL",
                "runner": "example",
                "example_id": "ocl2x2_smoke",
                "tiers": ["smoke"],
                "design_factors": {"pattern": "Quad Bayer", "ocl": "2x2", "readout": "2x2 binning/remosaic"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "ocl_focus_map"],
                "decision_notes": ["Checks same-color group uniformity and edge response trend."],
            },
            {
                "id": "nona_3x3",
                "label": "Nona 3x3 + 3x3 OCL",
                "runner": "example",
                "example_id": "ocl3x3_smoke",
                "tiers": ["smoke"],
                "design_factors": {"pattern": "Nona", "ocl": "3x3", "readout": "3x3 binning/remosaic"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "ocl_focus_map"],
                "decision_notes": ["Longer smoke run; useful for grouped-pixel corner response risk."],
            },
            {
                "id": "quad_qpd",
                "label": "Quad Bayer + QPD",
                "runner": "example",
                "example_id": "split_pd_quad_smoke",
                "tiers": ["smoke"],
                "design_factors": {"pattern": "Quad Bayer", "ocl": "2x2", "pdaf": "QPD"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "pdaf_balance"],
                "decision_notes": ["Checks Q1-Q4 balance and phase response direction."],
            },
        ],
    },
    "ocl_mixed_boundary": {
        "id": "ocl_mixed_boundary",
        "label": "Mixed OCL Boundary Risk",
        "category": "OCL / CRA",
        "priority": 2,
        "runtime_hint": "1-2 min smoke",
        "decision_goal": "Solve a smoke mixed-OCL supercell and compare it with 1x1/2x2 anchors.",
        "recommended_tier": "smoke",
        "tiers": ["smoke", "trend"],
        "cases": [
            {
                "id": "ocl_1x1_anchor",
                "label": "1x1 OCL anchor",
                "runner": "example",
                "example_id": "bayer1x1_smoke",
                "tiers": ["smoke"],
                "design_factors": {"ocl": "1x1", "role": "boundary anchor"},
                "charts": ["cra_response_curve", "subpixel_response_matrix"],
            },
            {
                "id": "ocl_2x2_anchor",
                "label": "2x2 OCL anchor",
                "runner": "example",
                "example_id": "ocl2x2_smoke",
                "tiers": ["smoke"],
                "design_factors": {"ocl": "2x2", "role": "boundary anchor"},
                "charts": ["cra_response_curve", "subpixel_response_matrix"],
            },
            {
                "id": "mixed_boundary_fdtd",
                "label": "Mixed 1x1/2x2/3x3 OCL boundary",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-layout",
                "layout_nx": 5,
                "layout_nz": 3,
                "ocl_layout": "nona_l:0:0:3:3,quad_r:3:0:2:2,bayer_r0:3:2:1:1,bayer_r1:4:2:1:1",
                "ocl_layout_name": "mixed_3x3_2x2_1x1_boundary_5x3",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "nona",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 10,
                "after_source_time": 1,
                "pml_um": 0.45,
                "design_factors": {"ocl": "1x1/2x2/3x3 mixed", "solver_status": "fdtd smoke primitive"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "ocl_focus_map"],
                "decision_notes": [
                    "This is a smoke mixed-OCL supercell; use higher resolution and measured stack for quantitative decisions.",
                    "The layout contains a 3x3 OCL block, a 2x2 OCL block, and two 1x1 boundary lenses in one FDTD cell.",
                ],
            },
            {
                "id": "custom_polygon_ocl_fdtd",
                "label": "Custom polygon OCL aperture",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-layout",
                "layout_nx": 2,
                "layout_nz": 2,
                "ocl_layout": "poly_quad:0:0:2:2",
                "ocl_polygons": "{\"poly_quad\":[[-1.32,-1.32],[1.24,-1.12],[1.30,1.26],[-1.16,1.34]]}",
                "ocl_layout_name": "custom_polygon_quad_ocl_aperture",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "quad",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 10,
                "after_source_time": 1,
                "pml_um": 0.45,
                "design_factors": {"ocl": "custom polygon", "solver_status": "polygon footprint FDTD primitive"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "ocl_focus_map"],
                "decision_notes": [
                    "This clips the OCL footprint with a local polygon and keeps the lens sag as an equivalent spherical cap.",
                    "Use CAD/GDS import for true freeform lens surface accuracy.",
                ],
            },
            {
                "id": "asphere_ocl_fdtd",
                "label": "Asphere OCL sag profile",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-layout",
                "layout_nx": 2,
                "layout_nz": 2,
                "ocl_layout": "asphere_quad:0:0:2:2",
                "ocl_sag": "{\"asphere_quad\":{\"type\":\"asphere\",\"conic_k\":-0.65,\"a4\":0.018,\"normalize_edge\":true}}",
                "ocl_layout_name": "asphere_quad_ocl_aperture",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "quad",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 10,
                "after_source_time": 1,
                "pml_um": 0.45,
                "design_factors": {"ocl": "asphere sag", "solver_status": "conic/asphere FDTD primitive"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "ocl_focus_map"],
                "decision_notes": [
                    "This changes the OCL surface sag profile while preserving center height and edge height.",
                    "Coefficients are configurable proxy values; measured lens profilometry is required for product accuracy.",
                ],
            },
            {
                "id": "surface_map_ocl_fdtd",
                "label": "Freeform surface-map OCL",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-layout",
                "layout_nx": 2,
                "layout_nz": 2,
                "ocl_layout": "surface_quad:0:0:2:2",
                "ocl_surface_map": "{\"surface_quad\":{\"source\":\"inline_freeform_smoke\",\"x_um\":[-1.32,0,1.32],\"z_um\":[-1.32,0,1.32],\"height_um\":[[0,0.20,0],[0.18,0.657,0.22],[0,0.21,0]]}}",
                "ocl_layout_name": "surface_map_quad_ocl_aperture",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "quad",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 10,
                "after_source_time": 1,
                "pml_um": 0.45,
                "design_factors": {"ocl": "freeform surface map", "solver_status": "bilinear height-map FDTD primitive"},
                "charts": ["cra_response_curve", "subpixel_response_matrix", "ocl_focus_map"],
                "decision_notes": [
                    "This uses a local measured/freeform height grid with bilinear interpolation in the FDTD material function.",
                    "For real accuracy, replace the inline grid with measured OCL profilometry and keep geometry lens_height consistent.",
                ],
            },
            {
                "id": "gds_imported_geometry_lut_pipeline",
                "label": "GDS import -> OCL/CFA LUT pipeline",
                "runner": "gds_import_lut",
                "tiers": ["smoke"],
                "write_reference_gds": True,
                "gds_filename": "reference_pixel_masks.gds",
                "gds_map_config": "configs/gds_pixel_geometry_import_map_reference.json",
                "geometry_json_filename": "pixel_geometry_from_gds.json",
                "gds_report_filename": "gds_import_report.json",
                "gds_preview_filename": "gds_import_preview.svg",
                "generate_gmsh_mesh": True,
                "gmsh_mesh_dir": "gmsh_mesh",
                "gmsh_dimension": "2",
                "gmsh_depth_um": 2.8,
                "gmsh_mesh_um": 0.18,
                "gmsh_fine_mesh_um": 0.06,
                "gmsh_include_dti_oxide": True,
                "mode": "ocl-layout",
                "layout_nx": 2,
                "layout_nz": 2,
                "ocl_layout": "imported_qpd:0:0:2:2",
                "target_lens_id": "imported_qpd",
                "collection_mode": "split-pd",
                "split_mode": "quad",
                "ocl_layout_name": "gds_imported_geometry_qpd_smoke",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "quad",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {
                    "geometry_source": "reference GDS -> pixel_geometry_import_v1",
                    "ocl": "GDS-derived polygon aperture",
                    "cfa": "GDS-derived aperture polygons",
                    "tcad_mesh": "GDS-bbox-informed proxy Gmsh mesh",
                    "collection_mode": "split-pd",
                    "solver_status": "GDS conversion and Meep LUT pipeline smoke",
                },
                "decision_notes": [
                    "This regression runs the CAD path end-to-end: reference GDS, layer-map conversion, then Meep LUT import.",
                    "The Gmsh mesh artifact is a TCAD bridge mesh informed by the GDS bbox, not a native mask-polygon mesh.",
                    "The reference GDS is synthetic; real design use should replace it with CAD-exported GDS and a product layer map.",
                ],
                "charts": ["cra_response_curve", "subpixel_response_matrix", "convergence_report_card"],
            },
        ],
    },
    "cad_template_solver_smoke": {
        "id": "cad_template_solver_smoke",
        "label": "CAD Template Solver Smoke",
        "category": "CAD Geometry",
        "priority": 2.5,
        "runtime_hint": "single case 20-60 sec smoke",
        "decision_goal": "Verify that each FreeCAD-openable CAD template can drive the FDTD footprint path with traceable geometry provenance.",
        "recommended_tier": "smoke",
        "tiers": ["smoke", "trend"],
        "cases": [
            {
                "id": "cad_bayer_1x1_3x3",
                "label": "CAD Bayer 1x1 3x3 template",
                "runner": "cad_template_lut",
                "tiers": ["smoke"],
                "cad_template_id": "bayer_1x1_3x3",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cases_arg": "center:0:0:0:0:0:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {"geometry_source": "CAD template geometry_import.json", "template": "bayer_1x1_3x3"},
                "charts": ["subpixel_response_matrix", "convergence_report_card"],
            },
            {
                "id": "cad_quad_2x2_ocl",
                "label": "CAD Quad 2x2 OCL template",
                "runner": "cad_template_lut",
                "tiers": ["smoke"],
                "cad_template_id": "quad_2x2_ocl",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cases_arg": "center:0:0:0:0:0:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {"geometry_source": "CAD template geometry_import.json", "template": "quad_2x2_ocl"},
                "charts": ["subpixel_response_matrix", "convergence_report_card"],
            },
            {
                "id": "cad_nona_3x3_ocl",
                "label": "CAD Nona 3x3 OCL template",
                "runner": "cad_template_lut",
                "tiers": ["smoke"],
                "cad_template_id": "nona_3x3_ocl",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cases_arg": "center:0:0:0:0:0:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {"geometry_source": "CAD template geometry_import.json", "template": "nona_3x3_ocl"},
                "charts": ["subpixel_response_matrix", "convergence_report_card"],
            },
            {
                "id": "cad_qpd_split_pd_2x2",
                "label": "CAD QPD split-PD 2x2 template",
                "runner": "cad_template_lut",
                "tiers": ["smoke"],
                "cad_template_id": "qpd_split_pd_2x2",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cases_arg": "center:0:0:0:0:0:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {"geometry_source": "CAD template geometry_import.json", "template": "qpd_split_pd_2x2", "collection_mode": "split-pd"},
                "charts": ["subpixel_response_matrix", "pdaf_balance", "convergence_report_card"],
            },
            {
                "id": "cad_mixed_ocl_boundary",
                "label": "CAD mixed 1x1/2x2/3x3 OCL boundary template",
                "runner": "cad_template_lut",
                "tiers": ["smoke"],
                "cad_template_id": "mixed_1x1_2x2_3x3_boundary",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cases_arg": "center:0:0:0:0:0:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {"geometry_source": "CAD template geometry_import.json", "template": "mixed_1x1_2x2_3x3_boundary"},
                "charts": ["subpixel_response_matrix", "convergence_report_card"],
            },
        ],
    },
    "material_stack_sensitivity": {
        "id": "material_stack_sensitivity",
        "label": "Material / Stack Sensitivity",
        "category": "Materials",
        "priority": 3,
        "runtime_hint": "1-2 min smoke",
        "decision_goal": "Rank stack knobs that most affect response before measured n,k is available.",
        "recommended_tier": "smoke",
        "tiers": ["smoke", "trend"],
        "cases": [
            {
                "id": "cfa_thin",
                "label": "CFA thickness 0.65um",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-2x2",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
                "resolution": 18,
                "after_source_time": 2,
                "pml_um": 0.45,
                "stack_overrides": {"geometry_um.cfa_thickness": 0.65},
                "design_factors": {"cfa_thickness_um": 0.65},
                "charts": ["material_sensitivity_tornado", "cra_response_curve"],
            },
            {
                "id": "cfa_nominal",
                "label": "CFA thickness 0.80um",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-2x2",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
                "resolution": 18,
                "after_source_time": 2,
                "pml_um": 0.45,
                "stack_overrides": {"geometry_um.cfa_thickness": 0.80},
                "design_factors": {"cfa_thickness_um": 0.80},
                "charts": ["material_sensitivity_tornado", "cra_response_curve"],
            },
            {
                "id": "cfa_thick",
                "label": "CFA thickness 0.95um",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-2x2",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0",
                "resolution": 18,
                "after_source_time": 2,
                "pml_um": 0.45,
                "stack_overrides": {"geometry_um.cfa_thickness": 0.95},
                "design_factors": {"cfa_thickness_um": 0.95},
                "charts": ["material_sensitivity_tornado", "cra_response_curve"],
            },
            {
                "id": "nk_source_swap",
                "label": "CFA n,k source swap",
                "runner": "analysis_only",
                "tiers": ["smoke"],
                "design_factors": {"material_source": "proxy/public/measured"},
                "charts": ["material_source_gate"],
                "decision_notes": [
                    "Measured n,k import path is represented, but no measured file is loaded.",
                    "Do not treat proxy k curves as color-accuracy evidence.",
                ],
            },
        ],
    },
    "pdaf_qpd_balance": {
        "id": "pdaf_qpd_balance",
        "label": "PDAF / QPD Balance",
        "category": "PDAF",
        "priority": 4,
        "runtime_hint": "30-60 sec smoke",
        "decision_goal": "Compare dual-x, dual-z, and QPD phase balance under center/edge illumination.",
        "recommended_tier": "smoke",
        "tiers": ["smoke", "trend"],
        "cases": [
            {
                "id": "dual_x_split",
                "label": "Dual-x split",
                "runner": "example",
                "example_id": "bayer1x1_smoke",
                "tiers": ["smoke"],
                "design_factors": {"split_mode": "dual-x", "cra_axis": "x"},
                "charts": ["cra_response_curve", "pdaf_balance"],
            },
            {
                "id": "dual_z_split",
                "label": "Dual-z split",
                "runner": "example",
                "example_id": "split_pd_dualz_smoke",
                "tiers": ["smoke"],
                "design_factors": {"split_mode": "dual-z", "cra_axis": "z"},
                "charts": ["cra_response_curve", "pdaf_balance"],
            },
            {
                "id": "qpd_2x2",
                "label": "QPD 2x2",
                "runner": "example",
                "example_id": "split_pd_quad_smoke",
                "tiers": ["smoke"],
                "design_factors": {"split_mode": "quad", "pdaf": "QPD"},
                "charts": ["subpixel_response_matrix", "pdaf_balance"],
            },
            {
                "id": "advanced_ocl_qpd_split_collection",
                "label": "Surface-map OCL + CFA polygon + QPD collection",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-layout",
                "layout_nx": 2,
                "layout_nz": 2,
                "ocl_layout": "active_ocl:0:0:2:2",
                "target_lens_id": "active_ocl",
                "collection_mode": "split-pd",
                "split_mode": "quad",
                "ocl_surface_map": "{\"active_ocl\":{\"source\":\"suite_inline_freeform_qpd_smoke\",\"x_um\":[-1.32,0,1.32],\"z_um\":[-1.32,0,1.32],\"height_um\":[[0,0.20,0],[0.18,0.657,0.22],[0,0.21,0]]}}",
                "cfa_polygons": "{\"background\":\"passivation\",\"red\":[[-0.58,-0.58],[0.555,-0.58],[0.58,0.555],[-0.555,0.58]],\"green\":[[-0.58,-0.58],[0.58,-0.58],[0.58,0.58],[-0.58,0.58]],\"blue\":[[-0.555,-0.58],[0.58,-0.555],[0.555,0.58],[-0.58,0.555]]}",
                "ocl_layout_name": "surface_map_2x2_qpd_split_collection",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "quad",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 10,
                "after_source_time": 1,
                "pml_um": 0.45,
                "design_factors": {
                    "ocl": "surface-map 2x2",
                    "cfa": "inset polygon aperture",
                    "split_mode": "quad",
                    "collection_mode": "split-pd",
                    "solver_status": "combined OCL/CFA/QPD FDTD primitive",
                },
                "decision_notes": [
                    "This verifies that advanced OCL/CFA geometry and QPD phase response are solved in the same Meep run.",
                    "Use quantitative resolution and measured stack/material before using phase amplitude as product evidence.",
                ],
                "charts": ["cra_response_curve", "subpixel_response_matrix", "pdaf_balance", "convergence_report_card"],
            },
            {
                "id": "imported_geometry_qpd_split_collection",
                "label": "Imported OCL/CFA geometry + QPD collection",
                "runner": "meep_lut",
                "tiers": ["smoke"],
                "mode": "ocl-layout",
                "layout_nx": 2,
                "layout_nz": 2,
                "ocl_layout": "imported_qpd:0:0:2:2",
                "target_lens_id": "imported_qpd",
                "collection_mode": "split-pd",
                "split_mode": "quad",
                "ocl_polygons": "@configs/pixel_geometry_import_reference.json",
                "ocl_surface_map": "@configs/pixel_geometry_import_reference.json",
                "cfa_polygons": "@configs/pixel_geometry_import_reference.json",
                "ocl_layout_name": "imported_geometry_2x2_qpd_split_collection",
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cfa_pattern": "quad",
                "cases_arg": "center:0:0:0:0:0:0,edge20x:20:0:1:0:0.03:0",
                "resolution": 10,
                "after_source_time": 1,
                "pml_um": 0.45,
                "design_factors": {
                    "ocl": "imported polygon + surface map",
                    "cfa": "imported aperture polygons",
                    "split_mode": "quad",
                    "collection_mode": "split-pd",
                    "geometry_source": "configs/pixel_geometry_import_reference.json",
                    "solver_status": "file-imported OCL/CFA/QPD FDTD primitive",
                },
                "decision_notes": [
                    "This validates the practical file-import path for CAD/profilometry-derived OCL and CFA geometry.",
                    "The reference file is synthetic; replace it with extracted/measured geometry for design-use runs.",
                ],
                "charts": ["cra_response_curve", "subpixel_response_matrix", "pdaf_balance", "convergence_report_card"],
            },
        ],
    },
    "crosstalk_kernel_practical": {
        "id": "crosstalk_kernel_practical",
        "label": "Crosstalk Kernel Practical",
        "category": "Crosstalk",
        "priority": 5,
        "runtime_hint": "instant references + short mixed-boundary FDTD smoke",
        "decision_goal": "Inspect finite-array, x-section, and mixed-OCL boundary crosstalk evidence with solver-backed KPI gates.",
        "recommended_tier": "smoke",
        "tiers": ["smoke", "trend"],
        "cases": [
            {
                "id": "cad_quad_2x2_ocl_5x5_crosstalk_fdtd",
                "label": "CAD Quad 2x2 OCL 5x5 practical kernel",
                "runner": "cad_template_crosstalk",
                "tiers": ["smoke", "trend"],
                "cad_template_id": "quad_2x2_ocl_5x5_crosstalk",
                "neighborhoods": "5",
                "resolutions": "4",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0,field20x:20:0:1:0",
                "color_channel": "green",
                "after_source_time": 1,
                "source_scale": 0.92,
                "source_sigma_scale": 0.34,
                "guard_cells": 0,
                "tier_overrides": {
                    "trend": {
                        "resolutions": "6",
                        "after_source_time": 2,
                        "cases_arg": "center:0:0:0:0,field20x:20:0:1:0,diag20:20:20:1:1",
                    }
                },
                "design_factors": {
                    "geometry_source": "CAD template geometry_import.json",
                    "template": "quad_2x2_ocl_5x5_crosstalk",
                    "ocl": "2x2",
                    "kernel": "5x5 OCL-group practical finite layout",
                    "target_lens": "central 2x2 OCL group",
                },
                "decision_notes": [
                    "This is the practical Quad Bayer 2x2 OCL crosstalk domain: central OCL group plus two OCL-group rings in x/z.",
                    "Use this before camera-system kernel export; smoke resolution is still research/trend evidence, not product accuracy.",
                ],
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "xsection_2x2_reference",
                "label": "2x2 OCL x-section crosstalk",
                "runner": "existing_crosstalk",
                "tiers": ["smoke"],
                "summary_csv": "runs/crosstalk_xsection_2d_ocl2_n5_r72_84/crosstalk_xsection_summary.csv",
                "artifact_png": "runs/crosstalk_xsection_2d_ocl2_n5_r72_84/crosstalk_xsection_kernel_lines.png",
                "design_factors": {"ocl": "2x2", "neighborhood": "5-cell x-section"},
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "xsection_3x3_reference",
                "label": "3x3 OCL x-section crosstalk",
                "runner": "existing_crosstalk",
                "tiers": ["smoke"],
                "summary_csv": "runs/crosstalk_xsection_2d_ocl3_n5_r72_84/crosstalk_xsection_summary.csv",
                "artifact_png": "runs/crosstalk_xsection_2d_ocl3_n5_r72_84/crosstalk_xsection_kernel_lines.png",
                "design_factors": {"ocl": "3x3", "neighborhood": "5-cell x-section"},
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "mixed_ocl_boundary_fdtd",
                "label": "Mixed OCL boundary leakage asymmetry",
                "runner": "crosstalk_fdtd",
                "tiers": ["smoke"],
                "modes": "ocl-layout",
                "layout_nx": 5,
                "layout_nz": 3,
                "ocl_layout": "nona_l:0:0:3:3,quad_r:3:0:2:2,bayer_r0:3:2:1:1,bayer_r1:4:2:1:1",
                "ocl_layout_name": "mixed_3x3_2x2_1x1_boundary_5x3",
                "target_lens_id": "nona_l",
                "neighborhoods": "5",
                "resolutions": "4",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0,field20x:20:0:1:0:0.03:0",
                "color_channel": "green",
                "cfa_pattern": "nona",
                "after_source_time": 1,
                "source_scale": 0.92,
                "source_sigma_scale": 0.34,
                "guard_cells": 0,
                "design_factors": {
                    "ocl": "mixed 3x3/2x2/1x1",
                    "kernel": "local finite layout",
                    "target_lens": "nona_l",
                },
                "decision_notes": [
                    "Use this to inspect leakage asymmetry at OCL topology transitions; it is still not product-ready without measured stack/material calibration.",
                ],
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "polygon_ocl_crosstalk_fdtd",
                "label": "Polygon OCL crosstalk smoke",
                "runner": "crosstalk_fdtd",
                "tiers": ["smoke"],
                "modes": "ocl-layout",
                "layout_nx": 3,
                "layout_nz": 3,
                "ocl_layout": "bottom:1:0:1:1,left:0:1:1:1,poly_center:1:1:1:1,right:2:1:1:1,top:1:2:1:1",
                "ocl_polygons": "{\"poly_center\":[[-0.62,-0.62],[0.62,-0.46],[0.52,0.62],[-0.60,0.50]]}",
                "ocl_layout_name": "custom_polygon_1x1_ocl_crosstalk_3x3",
                "target_lens_id": "poly_center",
                "neighborhoods": "3",
                "resolutions": "4",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0",
                "color_channel": "green",
                "cfa_pattern": "bayer",
                "after_source_time": 1,
                "source_scale": 0.92,
                "source_sigma_scale": 0.34,
                "guard_cells": 0,
                "design_factors": {
                    "ocl": "custom polygon",
                    "kernel": "local finite layout",
                    "target_lens": "poly_center",
                },
                "decision_notes": [
                    "Validates polygon footprint support in the crosstalk FDTD path; smoke grid is intentionally non-quantitative.",
                ],
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "imported_geometry_crosstalk_fdtd",
                "label": "Imported OCL/CFA geometry crosstalk smoke",
                "runner": "crosstalk_fdtd",
                "tiers": ["smoke"],
                "modes": "ocl-layout",
                "layout_nx": 3,
                "layout_nz": 3,
                "ocl_layout": "bottom:1:0:1:1,left:0:1:1:1,imported_center:1:1:1:1,right:2:1:1:1,top:1:2:1:1",
                "ocl_polygons": "@configs/pixel_geometry_import_reference.json",
                "ocl_surface_map": "@configs/pixel_geometry_import_reference.json",
                "cfa_polygons": "@configs/pixel_geometry_import_reference.json",
                "ocl_layout_name": "imported_geometry_crosstalk_smoke",
                "target_lens_id": "imported_center",
                "neighborhoods": "3",
                "resolutions": "4",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0",
                "color_channel": "green",
                "cfa_pattern": "bayer",
                "after_source_time": 1,
                "source_scale": 0.92,
                "source_sigma_scale": 0.34,
                "guard_cells": 0,
                "design_factors": {
                    "ocl": "imported polygon + surface map",
                    "cfa": "imported aperture polygons",
                    "kernel": "local finite layout",
                    "target_lens": "imported_center",
                    "geometry_source": "configs/pixel_geometry_import_reference.json",
                },
                "decision_notes": [
                    "Validates file-imported OCL/CFA geometry in the finite-array crosstalk path.",
                    "Smoke convergence is intentionally non-quantitative; raise resolution for trend/quantitative runs.",
                ],
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "asphere_ocl_crosstalk_fdtd",
                "label": "Asphere OCL crosstalk smoke",
                "runner": "crosstalk_fdtd",
                "tiers": ["smoke"],
                "modes": "ocl-layout",
                "layout_nx": 3,
                "layout_nz": 3,
                "ocl_layout": "bottom:1:0:1:1,left:0:1:1:1,asphere_center:1:1:1:1,right:2:1:1:1,top:1:2:1:1",
                "ocl_sag": "{\"asphere_center\":{\"type\":\"asphere\",\"conic_k\":-0.65,\"a4\":0.018,\"normalize_edge\":true}}",
                "ocl_layout_name": "asphere_1x1_ocl_crosstalk_3x3",
                "target_lens_id": "asphere_center",
                "neighborhoods": "3",
                "resolutions": "4",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0",
                "color_channel": "green",
                "cfa_pattern": "bayer",
                "after_source_time": 1,
                "source_scale": 0.92,
                "source_sigma_scale": 0.34,
                "guard_cells": 0,
                "design_factors": {
                    "ocl": "asphere sag",
                    "kernel": "local finite layout",
                    "target_lens": "asphere_center",
                },
                "decision_notes": [
                    "Validates conic/asphere sag support in the crosstalk FDTD path; smoke grid is intentionally non-quantitative.",
                ],
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
            {
                "id": "surface_map_ocl_crosstalk_fdtd",
                "label": "Surface-map OCL crosstalk smoke",
                "runner": "crosstalk_fdtd",
                "tiers": ["smoke"],
                "modes": "ocl-layout",
                "layout_nx": 3,
                "layout_nz": 3,
                "ocl_layout": "bottom:1:0:1:1,left:0:1:1:1,surface_center:1:1:1:1,right:2:1:1:1,top:1:2:1:1",
                "ocl_surface_map": "{\"surface_center\":{\"source\":\"inline_freeform_smoke\",\"x_um\":[-0.66,0,0.66],\"z_um\":[-0.66,0,0.66],\"height_um\":[[0,0.18,0],[0.20,0.657,0.22],[0,0.19,0]]}}",
                "ocl_layout_name": "surface_map_1x1_ocl_crosstalk_3x3",
                "target_lens_id": "surface_center",
                "neighborhoods": "3",
                "resolutions": "4",
                "wavelengths_nm": "550",
                "cases_arg": "center:0:0:0:0",
                "color_channel": "green",
                "cfa_pattern": "bayer",
                "after_source_time": 1,
                "source_scale": 0.92,
                "source_sigma_scale": 0.34,
                "guard_cells": 0,
                "design_factors": {
                    "ocl": "freeform surface map",
                    "kernel": "local finite layout",
                    "target_lens": "surface_center",
                },
                "decision_notes": [
                    "Validates measured/freeform height-map support in the crosstalk FDTD path; smoke grid is intentionally non-quantitative.",
                ],
                "charts": ["crosstalk_kernel_heatmap", "convergence_report_card"],
            },
        ],
    },
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def json_safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def devsim_dd_solver_gate(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    relative_error = json_safe_number(config.get("dd_relative_error"))
    max_iterations = json_safe_number(config.get("dd_max_iterations"))
    strict_relative_error = 1.0e-9
    if relative_error is None:
        return {
            "gate": "CHECK",
            "reason": "DD solver tolerance is missing from the summary.",
            "dd_relative_error": None,
            "dd_max_iterations": max_iterations,
        }
    if relative_error > strict_relative_error:
        return {
            "gate": "CHECK",
            "reason": f"Relaxed DD smoke tolerance ({relative_error:g}) is looser than strict smoke target ({strict_relative_error:g}).",
            "dd_relative_error": relative_error,
            "dd_max_iterations": max_iterations,
        }
    return {
        "gate": "PASS",
        "reason": f"DD smoke tolerance meets strict target ({relative_error:g} <= {strict_relative_error:g}).",
        "dd_relative_error": relative_error,
        "dd_max_iterations": max_iterations,
    }


def qpd_generation_volume_gate(template_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    generation_volume = summary.get("generation_volume_npz") or summary.get("generation_volume")
    if not generation_volume:
        return {
            "gate": "CHECK",
            "reason": "QPD G*W generation volume path is missing from the summary.",
            "generation_volume_npz": None,
        }
    text = str(generation_volume)
    if f"/{template_id}/" in text or template_id in Path(text).parts or template_id in text:
        return {
            "gate": "PASS",
            "reason": "QPD G*W uses a template-specific generation volume.",
            "generation_volume_npz": text,
        }
    return {
        "gate": "CHECK",
        "reason": f"QPD G*W generation volume is not template-specific for {template_id}.",
        "generation_volume_npz": text,
    }


def split_phase_metric_applicability(capability: dict[str, Any]) -> dict[str, Any]:
    requested_axis = str(capability.get("requested_split_axis") or "none").strip().lower()
    if requested_axis in {"", "none", "null"}:
        return {
            "applicable": False,
            "reason": "This CAD template is an image-pixel template without a requested split axis; DD smoke is a mesh/connectivity proxy, not a split phase metric.",
        }
    return {
        "applicable": True,
        "reason": "This CAD template requests a split axis, so normalized split photocurrent imbalance can be displayed as a phase-proxy metric.",
    }


def split_axis_from_template_parameters(parameters: dict[str, Any]) -> str:
    split_mode = str(parameters.get("split_mode") or "none").strip().lower()
    if split_mode == "dual-x":
        return "x"
    if split_mode == "dual-z":
        return "z"
    if split_mode == "quad":
        return "x_and_z"
    return "none"


def effective_tcad_capability(
    capability: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    effective = dict(capability or {})
    parameter_axis = split_axis_from_template_parameters(parameters)
    report_axis = str(effective.get("requested_split_axis") or "none").strip().lower()
    if parameter_axis == "none":
        effective.setdefault("requested_split_axis", "none")
        return effective
    if report_axis not in {"", "none", "null"}:
        return effective

    represented_axis = "z" if parameter_axis == "z" else "x"
    effective.update(
        {
            "gate": "CHECK",
            "requested_split_axis": parameter_axis,
            "section_axis": effective.get("section_axis") or represented_axis,
            "represented_split_axis": effective.get("represented_split_axis") or represented_axis,
            "contact_axis_labels": effective.get("contact_axis_labels")
            or (
                {"cathode_bottom": "bottom", "cathode_top": "top"}
                if represented_axis == "z"
                else {"cathode_left": "left", "cathode_right": "right"}
            ),
            "supported_phase_axes": effective.get("supported_phase_axes") or [represented_axis],
            "phase_result_scope": effective.get("phase_result_scope")
            or "split capability inferred from CAD template_parameters.json; regenerate TCAD bridge to refresh report metadata",
            "unsupported_outputs": effective.get("unsupported_outputs")
            or (["full_qpd_q1_q4_balance", "orthogonal_axis_pd_balance"] if parameter_axis == "x_and_z" else []),
            "inferred_from_template_parameters": True,
        }
    )
    return effective


def cad_template_simulation_fidelity(
    *,
    fdtd_generation_volume: Path | None,
    tcad_report: dict[str, Any],
    template_parameters: dict[str, Any],
    devsim_dd_report: dict[str, Any],
    qpd_weighting_report: dict[str, Any],
    qpd_gw_report: dict[str, Any],
) -> dict[str, Any]:
    has_3d_optical_volume = bool(fdtd_generation_volume and fdtd_generation_volume.exists())
    has_2d_dd = bool(devsim_dd_report)
    has_3d_weighting = bool(qpd_weighting_report)
    has_3d_gw = bool(qpd_gw_report)
    native_full_cad_mesh = bool(tcad_report.get("native_full_cad_electrical_mesh"))
    preserves_full_cad_connectivity = bool(tcad_report.get("preserves_full_3d_cad_connectivity"))
    capability = effective_tcad_capability(
        tcad_report.get("electrical_capability") if isinstance(tcad_report.get("electrical_capability"), dict) else {},
        template_parameters,
    )
    phase_applicability = split_phase_metric_applicability(capability)
    full_3d_dd = False
    return {
        "schema": "pixel_workbench_simulation_fidelity_v1",
        "summary": "3D CAD + hybrid 2D/3D",
        "accuracy_class": "research_trend",
        "gate": "CHECK",
        "cad_geometry": "3D parametric CAD template",
        "optical_generation": "3D FDTD volume" if has_3d_optical_volume else "not generated for this template",
        "electrical_dd": (
            "2D DEVSIM split-response proxy"
            if has_2d_dd and phase_applicability.get("applicable")
            else "2D DEVSIM connectivity proxy"
            if has_2d_dd
            else "not run"
        ),
        "qpd_weighting": "3D Laplace weighting potential" if has_3d_weighting else "not applicable or not run",
        "qpd_gw": "3D FDTD generation times 3D weighting surrogate" if has_3d_gw else "not applicable or not run",
        "full_3d_drift_diffusion": full_3d_dd,
        "native_full_cad_electrical_mesh": native_full_cad_mesh,
        "preserves_full_3d_cad_connectivity": preserves_full_cad_connectivity,
        "product_accuracy_ready": False,
        "not_full_3d_reason": (
            "Electrical collection uses 2D DEVSIM cross-sections and/or 3D weighting-potential surrogates; "
            "it is not a full 3D drift-diffusion solve on the native 3D CAD mesh."
        ),
    }


def csv_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rel_url(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return "/" + path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def set_nested_value(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override {dotted_path}: {part} is not an object")
        cursor = next_value
    cursor[parts[-1]] = value


def bounded_float(value: Any, name: str, default: float, minimum: float, maximum: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}; got {value!r}")
    return parsed


def bounded_int(value: Any, name: str, default: int, minimum: int, maximum: int) -> int:
    parsed = int(round(bounded_float(value, name, float(default), float(minimum), float(maximum))))
    return parsed


def enum_value(value: Any, name: str, allowed: set[str], default: str) -> str:
    text = str(default if value is None or value == "" else value).strip().lower()
    if text not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}")
    return text


def safe_case_string(value: Any) -> str:
    text = str(value or "center:0:0:0:0:0:0")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:+,-")
    if not text or len(text) > 800 or any(char not in allowed for char in text):
        raise ValueError("solver.cases contains unsupported characters or is too long")
    return text


def safe_layout_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:+,-")
    if len(text) > 1200 or any(char not in allowed for char in text):
        raise ValueError("solver.ocl_layout contains unsupported characters or is too long")
    return text


def safe_optional_id(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    text = str(value)[:80]
    if not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{name} contains unsupported characters")
    return text


def safe_json_import_if_ref(value: Any, root_keys: tuple[str, ...], name: str) -> tuple[str, Any] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text.startswith("@"):
        return None
    path_text = text[1:].strip()
    if not path_text or len(path_text) > 240:
        raise ValueError(f"{name} import path is empty or too long")
    candidate = Path(path_text)
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{name} import path must stay under {ROOT}") from error
    if resolved.suffix.lower() != ".json":
        raise ValueError(f"{name} import path must be a JSON file")
    if not resolved.is_file():
        raise ValueError(f"{name} import file does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} import file is not valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} import file must contain a JSON object")
    for key in root_keys:
        if key in payload:
            return f"@{relative.as_posix()}", payload[key]
    geometry = payload.get("geometry")
    if isinstance(geometry, dict):
        for key in root_keys:
            if key in geometry:
                return f"@{relative.as_posix()}", geometry[key]
    return f"@{relative.as_posix()}", payload


def safe_ocl_polygon_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    imported = safe_json_import_if_ref(value, ("ocl_polygons",), "solver.ocl_polygons")
    import_ref = None
    if imported:
        import_ref, payload = imported
    else:
        payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("solver.ocl_polygons must be a JSON object mapping lens id to points")
    clean: dict[str, list[list[float]]] = {}
    for lens_id, points in payload.items():
        lens_key = str(lens_id)
        if not lens_key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid solver.ocl_polygons lens id: {lens_key!r}")
        if not isinstance(points, list) or len(points) < 3 or len(points) > 24:
            raise ValueError(f"solver.ocl_polygons[{lens_key!r}] must contain 3-24 points")
        clean_points: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"solver.ocl_polygons[{lens_key!r}] points must be [x,z] pairs")
            x = bounded_float(point[0], "solver.ocl_polygons.x", 0.0, -10.0, 10.0)
            z = bounded_float(point[1], "solver.ocl_polygons.z", 0.0, -10.0, 10.0)
            clean_points.append([x, z])
        clean[lens_key] = clean_points
    text = json.dumps(clean, separators=(",", ":"))
    if len(text) > 4000:
        raise ValueError("solver.ocl_polygons is too long")
    return import_ref or text


def safe_ocl_sag_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    imported = safe_json_import_if_ref(value, ("ocl_sag", "ocl_sag_profiles"), "solver.ocl_sag")
    import_ref = None
    if imported:
        import_ref, payload = imported
    else:
        payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("solver.ocl_sag must be a JSON object mapping default/lens id to sag profile")
    clean: dict[str, dict[str, Any]] = {}
    for lens_id, spec in payload.items():
        key = str(lens_id)
        if key != "default" and not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid solver.ocl_sag key: {key!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"solver.ocl_sag[{key!r}] must be an object")
        profile_type = str(spec.get("type", spec.get("profile_type", "asphere"))).lower()
        if profile_type not in {"sphere", "asphere"}:
            raise ValueError("solver.ocl_sag profile type must be sphere or asphere")
        clean_spec: dict[str, Any] = {
            "type": profile_type,
            "conic_k": bounded_float(spec.get("conic_k", spec.get("k", 0.0)), "solver.ocl_sag.conic_k", 0.0, -10.0, 10.0),
            "a4": bounded_float(spec.get("a4", 0.0), "solver.ocl_sag.a4", 0.0, -5.0, 5.0),
            "a6": bounded_float(spec.get("a6", 0.0), "solver.ocl_sag.a6", 0.0, -5.0, 5.0),
            "a8": bounded_float(spec.get("a8", 0.0), "solver.ocl_sag.a8", 0.0, -5.0, 5.0),
            "normalize_edge": bool(spec.get("normalize_edge", True)),
        }
        radius = spec.get("curvature_radius_um", spec.get("radius_um"))
        if radius not in {None, ""}:
            clean_spec["curvature_radius_um"] = bounded_float(radius, "solver.ocl_sag.curvature_radius_um", 1.0, 0.05, 50.0)
        clean[key] = clean_spec
    text = json.dumps(clean, separators=(",", ":"))
    if len(text) > 4000:
        raise ValueError("solver.ocl_sag is too long")
    return import_ref or text


def safe_ocl_surface_map_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    imported = safe_json_import_if_ref(
        value,
        ("ocl_surface_map", "ocl_surface_maps", "surface_maps"),
        "solver.ocl_surface_map",
    )
    import_ref = None
    if imported:
        import_ref, payload = imported
    else:
        payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("solver.ocl_surface_map must be a JSON object mapping default/lens id to a surface grid")
    clean: dict[str, dict[str, Any]] = {}
    for lens_id, spec in payload.items():
        key = str(lens_id)
        if key != "default" and not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid solver.ocl_surface_map key: {key!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"solver.ocl_surface_map[{key!r}] must be an object")
        x_values = spec.get("x_um")
        z_values = spec.get("z_um")
        heights = spec.get("height_um")
        if not isinstance(x_values, list) or not isinstance(z_values, list) or not isinstance(heights, list):
            raise ValueError(f"solver.ocl_surface_map[{key!r}] requires x_um, z_um, and height_um arrays")
        if len(x_values) < 2 or len(z_values) < 2 or len(x_values) > 65 or len(z_values) > 65:
            raise ValueError(f"solver.ocl_surface_map[{key!r}] grid must be 2-65 samples per axis")
        clean_x = [bounded_float(value, "solver.ocl_surface_map.x_um", 0.0, -10.0, 10.0) for value in x_values]
        clean_z = [bounded_float(value, "solver.ocl_surface_map.z_um", 0.0, -10.0, 10.0) for value in z_values]
        if any(clean_x[index] >= clean_x[index + 1] for index in range(len(clean_x) - 1)):
            raise ValueError(f"solver.ocl_surface_map[{key!r}].x_um must be strictly increasing")
        if any(clean_z[index] >= clean_z[index + 1] for index in range(len(clean_z) - 1)):
            raise ValueError(f"solver.ocl_surface_map[{key!r}].z_um must be strictly increasing")
        if len(heights) != len(clean_z):
            raise ValueError(f"solver.ocl_surface_map[{key!r}].height_um row count must match z_um")
        clean_heights: list[list[float]] = []
        for row in heights:
            if not isinstance(row, list) or len(row) != len(clean_x):
                raise ValueError(f"solver.ocl_surface_map[{key!r}].height_um rows must match x_um")
            clean_heights.append(
                [bounded_float(value, "solver.ocl_surface_map.height_um", 0.0, 0.0, 2.0) for value in row]
            )
        clean[key] = {
            "source": str(spec.get("source") or "UI inline surface map")[:120],
            "x_um": clean_x,
            "z_um": clean_z,
            "height_um": clean_heights,
        }
    text = json.dumps(clean, separators=(",", ":"))
    if len(text) > 12000:
        raise ValueError("solver.ocl_surface_map is too long")
    return import_ref or text


def sanitized_stack_overrides(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("solver.stack_overrides must be an object")
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        path = str(key)
        if path not in STACK_OVERRIDE_ALLOWLIST:
            raise ValueError(f"Unsupported stack override: {path}")
        if path == "materials.lens":
            if not isinstance(value, dict):
                raise ValueError("materials.lens override must be an object")
            clean[path] = {
                "n": bounded_float(value.get("n"), "materials.lens.n", 1.30, 1.30, 2.20),
                "k": bounded_float(value.get("k", 0.0), "materials.lens.k", 0.0, 0.0, 0.20),
                "measured": bool(value.get("measured", False)),
                "source": str(value.get("source") or "UI active design override")[:120],
                "usage": str(value.get("usage") or "on-chip microlens")[:120],
            }
            continue
        if path == "shield.mode":
            clean[path] = enum_value(value, path, SHIELD_MODES, "off")
            continue
        minimum, maximum = 0.0, 10.0
        if path == "geometry_um.pitch":
            minimum, maximum = 0.5, 5.0
        elif path == "geometry_um.lens_height":
            minimum, maximum = 0.05, 2.0
        elif path == "geometry_um.lens_edge_gap":
            minimum, maximum = 0.0, 0.5
        elif path == "geometry_um.cfa_thickness":
            minimum, maximum = 0.05, 2.0
        elif path == "geometry_um.passivation_thickness":
            minimum, maximum = 0.0, 0.5
        elif path == "geometry_um.si_thickness":
            minimum, maximum = 0.2, 10.0
        elif path == "shield.mask_edge_width_um":
            minimum, maximum = 0.0, 1.0
        clean[path] = bounded_float(value, path, 0.0, minimum, maximum)
    return clean


def sanitized_cfa_shifts(raw: Any) -> dict[str, dict[str, float]]:
    if not raw:
        return {
            "red": {"x": 0.0, "z": 0.0},
            "green": {"x": 0.0, "z": 0.0},
            "blue": {"x": 0.0, "z": 0.0},
        }
    if not isinstance(raw, dict):
        raise ValueError("solver.cfa_shifts_um must be an object")
    clean: dict[str, dict[str, float]] = {}
    for color in ("red", "green", "blue"):
        item = raw.get(color) or {}
        if not isinstance(item, dict):
            raise ValueError(f"solver.cfa_shifts_um.{color} must be an object")
        clean[color] = {
            "x": bounded_float(item.get("x"), f"solver.cfa_shifts_um.{color}.x", 0.0, -0.30, 0.30),
            "z": bounded_float(item.get("z"), f"solver.cfa_shifts_um.{color}.z", 0.0, -0.30, 0.30),
        }
    return clean


def clean_cfa_polygon_points(raw_points: Any, label: str) -> list[list[float]]:
    if not isinstance(raw_points, list) or len(raw_points) < 3 or len(raw_points) > 48:
        raise ValueError(f"{label} must contain 3-48 [x,z] points")
    clean_points: list[list[float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label} points must be [x,z] pairs")
        clean_points.append(
            [
                bounded_float(point[0], f"{label}.x", 0.0, -10.0, 10.0),
                bounded_float(point[1], f"{label}.z", 0.0, -10.0, 10.0),
            ]
        )
    return clean_points


def clean_cfa_polygon_spec(raw_spec: Any, label: str, default_color: str | None = None) -> dict[str, Any] | list[list[float]]:
    if isinstance(raw_spec, list):
        return clean_cfa_polygon_points(raw_spec, label)
    if not isinstance(raw_spec, dict):
        raise ValueError(f"{label} must be a polygon point list or object")
    color = str(raw_spec.get("color", default_color or "")).lower()
    if color and color not in COLOR_CHANNELS:
        raise ValueError(f"{label}.color must be red, green, or blue")
    points = raw_spec.get("points", raw_spec.get("polygon_um", raw_spec.get("polygon")))
    clean: dict[str, Any] = {
        "points": clean_cfa_polygon_points(points, f"{label}.points"),
        "shift_x_um": bounded_float(raw_spec.get("shift_x_um", raw_spec.get("x_shift_um", 0.0)), f"{label}.shift_x_um", 0.0, -0.5, 0.5),
        "shift_z_um": bounded_float(raw_spec.get("shift_z_um", raw_spec.get("z_shift_um", 0.0)), f"{label}.shift_z_um", 0.0, -0.5, 0.5),
        "source": str(raw_spec.get("source") or "UI inline CFA polygon")[:120],
    }
    if color:
        clean["color"] = color
    if raw_spec.get("ix") not in {None, ""}:
        clean["ix"] = bounded_int(raw_spec.get("ix"), f"{label}.ix", 0, -64, 64)
    if raw_spec.get("iz") not in {None, ""}:
        clean["iz"] = bounded_int(raw_spec.get("iz"), f"{label}.iz", 0, -64, 64)
    if raw_spec.get("id") not in {None, ""}:
        clean_id = str(raw_spec.get("id"))[:80]
        if not clean_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"{label}.id contains unsupported characters")
        clean["id"] = clean_id
    return clean


def safe_cfa_polygon_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    imported = safe_json_import_if_ref(value, ("cfa_polygons",), "solver.cfa_polygons")
    import_ref = None
    if imported:
        import_ref, payload = imported
    else:
        payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("solver.cfa_polygons must be a JSON object")
    clean: dict[str, Any] = {}
    background = str(payload.get("background", "passivation")).lower()
    if background not in {"nearest", "passivation", "air"}:
        raise ValueError("solver.cfa_polygons.background must be nearest, passivation, or air")
    clean["background"] = background
    for color in COLOR_CHANNELS:
        if color in payload:
            clean[color] = clean_cfa_polygon_spec(payload[color], f"solver.cfa_polygons.{color}", color)
    if payload.get("colors"):
        if not isinstance(payload["colors"], dict):
            raise ValueError("solver.cfa_polygons.colors must be an object")
        clean_colors: dict[str, Any] = {}
        for color, spec in payload["colors"].items():
            color_key = str(color).lower()
            if color_key not in COLOR_CHANNELS:
                raise ValueError("solver.cfa_polygons.colors keys must be red, green, or blue")
            clean_colors[color_key] = clean_cfa_polygon_spec(spec, f"solver.cfa_polygons.colors.{color_key}", color_key)
        clean["colors"] = clean_colors
    if payload.get("cells"):
        if not isinstance(payload["cells"], list) or len(payload["cells"]) > 96:
            raise ValueError("solver.cfa_polygons.cells must be a list of at most 96 cells")
        clean["cells"] = [
            clean_cfa_polygon_spec(spec, f"solver.cfa_polygons.cells.{index}")
            for index, spec in enumerate(payload["cells"])
        ]
    if len(clean) == 1:
        raise ValueError("solver.cfa_polygons must define at least one color or cell polygon")
    text = json.dumps(clean, separators=(",", ":"))
    if len(text) > 16000:
        raise ValueError("solver.cfa_polygons is too long")
    return import_ref or text


def normalize_stack_material_paths(config: dict[str, Any]) -> None:
    materials = config.get("materials", {})
    if not isinstance(materials, dict):
        return
    base_dir = BASE_STACK_CONFIG.parent
    for spec in materials.values():
        if not isinstance(spec, dict) or "nk_table" not in spec:
            continue
        table_path = Path(str(spec["nk_table"]))
        if not table_path.is_absolute():
            spec["nk_table"] = str((base_dir / table_path).resolve())


def solver_case_from_request(simulation_request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(simulation_request, dict):
        raise ValueError("simulation_request must be an object")
    solver = simulation_request.get("solver") or {}
    design = simulation_request.get("design") or {}
    condition = simulation_request.get("condition") or {}
    if not isinstance(solver, dict) or not isinstance(design, dict) or not isinstance(condition, dict):
        raise ValueError("simulation_request design/condition/solver must be objects")

    design_cad_template = design.get("cad_template") if isinstance(design.get("cad_template"), dict) else {}
    cad_template_id = safe_optional_id(
        solver.get("cad_template_id") or design_cad_template.get("template_id"),
        "solver.cad_template_id",
    )
    cad_template_defaults: dict[str, Any] | None = None
    ignored_cad_stack_overrides: dict[str, Any] = {}
    if cad_template_id:
        catalog = load_cad_template_catalog()
        template_record = next(
            (item for item in catalog.get("templates", []) if item.get("template_id") == cad_template_id),
            None,
        )
        if not template_record:
            raise ValueError(f"Unknown CAD template: {cad_template_id}")
        cad_template_defaults = template_record.get("solver_defaults")
        if not isinstance(cad_template_defaults, dict) or not cad_template_defaults.get("solver"):
            raise ValueError(f"CAD template is not solver-ready: {cad_template_id}")
        merged_solver = copy.deepcopy(solver)
        for geometry_key in ("layout_nx", "layout_nz", "ocl_layout", "ocl_polygons", "ocl_sag", "ocl_surface_map", "cfa_polygons", "target_lens_id"):
            merged_solver.pop(geometry_key, None)
        merged_solver.update(cad_template_defaults["solver"])
        template_stack = cad_template_defaults.get("stack_overrides") or {}
        allowed_user_stack, ignored_cad_stack_overrides = split_cad_template_stack_overrides(solver.get("stack_overrides"))
        if template_stack or allowed_user_stack:
            merged_solver["stack_overrides"] = {**template_stack, **allowed_user_stack}
        else:
            merged_solver.pop("stack_overrides", None)
        merged_solver["cad_template_id"] = cad_template_id
        solver = merged_solver

    wavelength_nm = bounded_float(
        solver.get("wavelengths_nm") or condition.get("wavelength_nm"),
        "wavelength_nm",
        550.0,
        380.0,
        700.0,
    )
    mode = enum_value(solver.get("mode"), "solver.mode", SOLVER_MODES, "ocl-2x2")
    split_mode = enum_value(solver.get("split_mode"), "solver.split_mode", SPLIT_MODES, "quad")
    collection_mode = enum_value(solver.get("collection_mode"), "solver.collection_mode", COLLECTION_MODES, "auto")
    color_channel = enum_value(solver.get("color_channel") or condition.get("color_channel"), "solver.color_channel", COLOR_CHANNELS, "green")
    shield_mode = enum_value(solver.get("shield_mode"), "solver.shield_mode", SHIELD_MODES, "off")
    cfa_pattern = enum_value(solver.get("cfa_pattern"), "solver.cfa_pattern", CFA_PATTERNS, "uniform")
    ocl_layout = safe_layout_string(solver.get("ocl_layout"))
    ocl_polygons = safe_ocl_polygon_string(solver.get("ocl_polygons"))
    ocl_sag = safe_ocl_sag_string(solver.get("ocl_sag"))
    ocl_surface_map = safe_ocl_surface_map_string(solver.get("ocl_surface_map"))
    cfa_polygons = safe_cfa_polygon_string(solver.get("cfa_polygons"))
    layout_nx = bounded_int(solver.get("layout_nx"), "solver.layout_nx", 1, 1, 12) if solver.get("layout_nx") is not None else None
    layout_nz = bounded_int(solver.get("layout_nz"), "solver.layout_nz", 1, 1, 12) if solver.get("layout_nz") is not None else None
    if mode == "ocl-layout":
        if not ocl_layout:
            raise ValueError("solver.ocl_layout is required when solver.mode is ocl-layout")
        if layout_nx is None or layout_nz is None:
            raise ValueError("solver.layout_nx and solver.layout_nz are required when solver.mode is ocl-layout")
    label = str(design.get("preset_label") or "Active Design Request")[:80]
    return {
        "id": "active_design_request",
        "label": f"Active Design: {label}",
        "preset_hint": str(design.get("preset_id") or "ui_active_design")[:80],
        "mode": mode,
        "split_mode": split_mode,
        "collection_mode": collection_mode,
        "layout_nx": layout_nx,
        "layout_nz": layout_nz,
        "ocl_layout": ocl_layout,
        "ocl_polygons": ocl_polygons,
        "ocl_sag": ocl_sag,
        "ocl_surface_map": ocl_surface_map,
        "ocl_layout_name": str(solver.get("ocl_layout_name") or "")[:80] or None,
        "target_lens_id": safe_optional_id(solver.get("target_lens_id"), "solver.target_lens_id"),
        "wavelengths_nm": f"{wavelength_nm:g}",
        "color_channel": color_channel,
        "cfa_pattern": cfa_pattern,
        "cfa_shifts_um": sanitized_cfa_shifts(solver.get("cfa_shifts_um")),
        "cfa_polygons": cfa_polygons,
        "cases": safe_case_string(solver.get("cases")),
        "resolution": bounded_int(solver.get("resolution"), "solver.resolution", 18, 8, 120),
        "after_source_time": bounded_float(solver.get("after_source_time"), "solver.after_source_time", 2.0, 0.5, 200.0),
        "pml_um": bounded_float(solver.get("pml_um"), "solver.pml_um", 0.45, 0.20, 1.20),
        "shield_mode": shield_mode,
        "shield_mask_edge_width_um": bounded_float(
            solver.get("shield_mask_edge_width_um"),
            "solver.shield_mask_edge_width_um",
            0.12,
            0.0,
            1.0,
        )
        if solver.get("shield_mask_edge_width_um") is not None
        else None,
        "stack_overrides": sanitized_stack_overrides(solver.get("stack_overrides")),
        "description": "Generated from UI Design + Condition state.",
        "cad_template": {
            "template_id": cad_template_defaults.get("template_id"),
            "label": cad_template_defaults.get("label"),
            "source_truth_level": cad_template_defaults.get("source_truth_level"),
            "geometry_import": cad_template_defaults.get("geometry_import"),
            "parameters": cad_template_defaults.get("parameters"),
            "geometry_authority": "cad_template",
            "geometry_override_policy": "Protected geometry stack keys are ignored for CAD-template requests; create a CAD variant instead.",
            "ignored_stack_override_keys": sorted(ignored_cad_stack_overrides),
        }
        if cad_template_defaults
        else None,
    }


def write_stack_override_config(case: dict[str, Any], output_dir: Path) -> Path | None:
    overrides = case.get("stack_overrides") or {}
    if not overrides:
        return None
    config = json.loads(BASE_STACK_CONFIG.read_text(encoding="utf-8"))
    normalize_stack_material_paths(config)
    for path, value in overrides.items():
        set_nested_value(config, str(path), value)
    config.setdefault("calibration_status", {})
    config["calibration_status"]["mode"] = "suite_runtime_override"
    config["calibration_status"]["note"] = "Generated by pixel_workbench_server.py for a local test-suite run."
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{case['id']}_stack.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def command_for_lut_case(case: dict[str, Any], output_dir: Path) -> list[str]:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Meep Python environment not found: {PYTHON}")
    command = [
        str(PYTHON),
        "meep_supercell_lut.py",
        "--mode",
        str(case["mode"]),
        "--wavelengths-nm",
        str(case.get("wavelengths_nm", "550")),
        "--cases",
        str(case.get("cases_arg", case.get("cases", "center:0:0:0:0:0:0"))),
        "--resolution",
        str(case.get("resolution", 18)),
        "--after-source-time",
        str(case.get("after_source_time", 2)),
        "--pml-um",
        str(case.get("pml_um", 0.45)),
        "--grid-snap-y",
        "nearest",
        "--output-dir",
        str(output_dir),
    ]
    if case.get("layout_nx") is not None:
        command.extend(["--layout-nx", str(case["layout_nx"])])
    if case.get("layout_nz") is not None:
        command.extend(["--layout-nz", str(case["layout_nz"])])
    if case.get("ocl_layout"):
        command.extend(["--ocl-layout", str(case["ocl_layout"])])
    if case.get("ocl_polygons"):
        command.extend(["--ocl-polygons", str(case["ocl_polygons"])])
    if case.get("ocl_sag"):
        command.extend(["--ocl-sag", str(case["ocl_sag"])])
    if case.get("ocl_surface_map"):
        command.extend(["--ocl-surface-map", str(case["ocl_surface_map"])])
    if case.get("ocl_layout_name"):
        command.extend(["--ocl-layout-name", str(case["ocl_layout_name"])])
    if case.get("split_mode"):
        command.extend(["--split-mode", str(case["split_mode"])])
    if case.get("collection_mode"):
        command.extend(["--collection-mode", str(case["collection_mode"])])
    if case.get("target_lens_id"):
        command.extend(["--target-lens-id", str(case["target_lens_id"])])
    if case.get("color_channel"):
        command.extend(["--color-channel", str(case["color_channel"])])
    if case.get("cfa_pattern"):
        command.extend(["--cfa-pattern", str(case["cfa_pattern"])])
    if case.get("cfa_polygons"):
        command.extend(["--cfa-polygons", str(case["cfa_polygons"])])
    cfa_shifts = case.get("cfa_shifts_um") or {}
    for color in ("red", "green", "blue"):
        shift = cfa_shifts.get(color) or {}
        for axis in ("x", "z"):
            value = shift.get(axis)
            if value is not None:
                command.extend([f"--cfa-shift-{color}-{axis}-um", str(value)])
    if case.get("shield_mode"):
        command.extend(["--shield-mode", str(case["shield_mode"])])
    if case.get("shield_mask_edge_width_um") is not None:
        command.extend(["--shield-mask-edge-width-um", str(case["shield_mask_edge_width_um"])])
    stack_config = write_stack_override_config(case, output_dir)
    if stack_config:
        command.extend(["--stack-config", str(stack_config)])
    return command


def command_for_crosstalk_case(case: dict[str, Any], output_dir: Path) -> list[str]:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Meep Python environment not found: {PYTHON}")
    command = [
        str(PYTHON),
        "meep_crosstalk_kernel.py",
        "--modes",
        str(case.get("modes", case.get("mode", "ocl-2x2"))),
        "--neighborhoods",
        str(case.get("neighborhoods", case.get("neighborhood", "3"))),
        "--resolutions",
        str(case.get("resolutions", case.get("resolution", "4"))),
        "--wavelengths-nm",
        str(case.get("wavelengths_nm", "550")),
        "--cases",
        str(case.get("cases_arg", case.get("cases", "center:0:0:0:0"))),
        "--after-source-time",
        str(case.get("after_source_time", 1.0)),
        "--source-scale",
        str(case.get("source_scale", 0.92)),
        "--source-profile",
        str(case.get("source_profile", "gaussian")),
        "--source-sigma-scale",
        str(case.get("source_sigma_scale", 0.34)),
        "--guard-cells",
        str(case.get("guard_cells", 1)),
        "--pml-um",
        str(case.get("pml_um", 0.45)),
        "--output-dir",
        str(output_dir),
    ]
    if case.get("color_channel"):
        command.extend(["--color-channel", str(case["color_channel"])])
    if case.get("cfa_pattern"):
        command.extend(["--cfa-pattern", str(case["cfa_pattern"])])
    if case.get("cfa_polygons"):
        command.extend(["--cfa-polygons", str(case["cfa_polygons"])])
    cfa_shifts = case.get("cfa_shifts_um") or {}
    for color in ("red", "green", "blue"):
        shift = cfa_shifts.get(color) or {}
        for axis in ("x", "z"):
            value = shift.get(axis)
            if value is not None:
                command.extend([f"--cfa-shift-{color}-{axis}-um", str(value)])
    if case.get("layout_nx") is not None:
        command.extend(["--layout-nx", str(case["layout_nx"])])
    if case.get("layout_nz") is not None:
        command.extend(["--layout-nz", str(case["layout_nz"])])
    if case.get("ocl_layout"):
        command.extend(["--ocl-layout", str(case["ocl_layout"])])
    if case.get("ocl_polygons"):
        command.extend(["--ocl-polygons", str(case["ocl_polygons"])])
    if case.get("ocl_sag"):
        command.extend(["--ocl-sag", str(case["ocl_sag"])])
    if case.get("ocl_surface_map"):
        command.extend(["--ocl-surface-map", str(case["ocl_surface_map"])])
    if case.get("ocl_layout_name"):
        command.extend(["--ocl-layout-name", str(case["ocl_layout_name"])])
    if case.get("target_lens_id"):
        command.extend(["--target-lens-id", str(case["target_lens_id"])])
    stack_config = write_stack_override_config(case, output_dir)
    if stack_config:
        command.extend(["--stack-config", str(stack_config)])
    return command


def root_import_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Import artifact must stay under {ROOT}: {resolved}") from error
    return f"@{relative.as_posix()}"


def central_ocl_block(blocks: list[dict[str, Any]], nx: int, nz: int) -> dict[str, Any] | None:
    if not blocks:
        return None
    center_x = nx / 2.0
    center_z = nz / 2.0

    def distance(block: dict[str, Any]) -> tuple[float, str]:
        ix = json_safe_number(block.get("ix")) or 0.0
        iz = json_safe_number(block.get("iz")) or 0.0
        sx = json_safe_number(block.get("sx")) or 1.0
        sz = json_safe_number(block.get("sz")) or 1.0
        bx = ix + sx / 2.0
        bz = iz + sz / 2.0
        return ((bx - center_x) ** 2 + (bz - center_z) ** 2, str(block.get("lens_id") or ""))

    return min(blocks, key=distance)


def cad_template_solver_defaults_from_paths(
    template_id: str,
    label: str | None,
    source_truth_level: str | None,
    geometry_path: Path | None,
    parameters_path: Path | None,
) -> dict[str, Any]:
    if geometry_path is None or parameters_path is None:
        raise ValueError(f"CAD template {template_id!r} is missing geometry or parameter path")
    if not geometry_path.is_file():
        raise ValueError(f"CAD template geometry JSON is missing: {geometry_path}")
    if not parameters_path.is_file():
        raise ValueError(f"CAD template parameter JSON is missing: {parameters_path}")
    geometry = read_json_artifact(geometry_path)
    params = read_json_artifact(parameters_path)
    nx = bounded_int(params.get("nx"), f"cad_template.{template_id}.nx", 1, 1, 12)
    nz = bounded_int(params.get("nz"), f"cad_template.{template_id}.nz", 1, 1, 12)
    raw_blocks = params.get("ocl_blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(f"CAD template {template_id!r} must define ocl_blocks")

    layout_parts = []
    clean_blocks: list[dict[str, Any]] = []
    for index, block in enumerate(raw_blocks):
        if not isinstance(block, dict):
            raise ValueError(f"cad_template.{template_id}.ocl_blocks[{index}] must be an object")
        lens_id = safe_optional_id(block.get("lens_id"), f"cad_template.{template_id}.ocl_blocks[{index}].lens_id")
        if not lens_id:
            raise ValueError(f"cad_template.{template_id}.ocl_blocks[{index}].lens_id is required")
        ix = bounded_int(block.get("ix"), f"cad_template.{template_id}.ocl_blocks[{index}].ix", 0, 0, nx - 1)
        iz = bounded_int(block.get("iz"), f"cad_template.{template_id}.ocl_blocks[{index}].iz", 0, 0, nz - 1)
        sx = bounded_int(block.get("sx"), f"cad_template.{template_id}.ocl_blocks[{index}].sx", 1, 1, nx)
        sz = bounded_int(block.get("sz"), f"cad_template.{template_id}.ocl_blocks[{index}].sz", 1, 1, nz)
        if ix + sx > nx or iz + sz > nz:
            raise ValueError(f"cad_template.{template_id}.ocl_blocks[{index}] exceeds template bounds")
        clean_block = {"lens_id": lens_id, "ix": ix, "iz": iz, "sx": sx, "sz": sz}
        clean_blocks.append(clean_block)
        layout_parts.append(f"{lens_id}:{ix}:{iz}:{sx}:{sz}")

    geometry_ref = root_import_ref(geometry_path)
    raw_cfa_pattern = str(params.get("cfa_pattern") or "uniform").strip().lower()
    if raw_cfa_pattern in CFA_PATTERNS:
        cfa_pattern = raw_cfa_pattern
        color_channel = str(params.get("color_channel") or "green").strip().lower()
    elif raw_cfa_pattern in UNIFORM_COLOR_CFA_PATTERNS:
        cfa_pattern = "uniform"
        color_channel = UNIFORM_COLOR_CFA_PATTERNS[raw_cfa_pattern]
    else:
        cfa_pattern = "uniform"
        color_channel = str(params.get("color_channel") or "green").strip().lower()
    if color_channel not in COLOR_CHANNELS:
        color_channel = "green"
    split_mode = str(params.get("split_mode") or "").strip().lower()
    shield_mode = str(params.get("shield_mode") or "off").strip().lower()
    target = central_ocl_block(clean_blocks, nx, nz)
    solver_defaults: dict[str, Any] = {
        "mode": "ocl-layout",
        "layout_nx": nx,
        "layout_nz": nz,
        "ocl_layout": ",".join(layout_parts),
        "ocl_polygons": geometry_ref,
        "cfa_polygons": geometry_ref,
        "ocl_layout_name": template_id[:80],
        "target_lens_id": target.get("lens_id") if target else clean_blocks[0]["lens_id"],
        "cfa_pattern": cfa_pattern,
        "color_channel": color_channel,
        "raw_cfa_pattern": raw_cfa_pattern,
        "collection_mode": "pixel",
    }
    if split_mode in SPLIT_MODES:
        solver_defaults["split_mode"] = split_mode
        solver_defaults["collection_mode"] = "split-pd"
    if shield_mode in SHIELD_MODES:
        solver_defaults["shield_mode"] = shield_mode
    if geometry.get("ocl_surface_maps") or geometry.get("ocl_surface_map") or geometry.get("surface_maps"):
        solver_defaults["ocl_surface_map"] = geometry_ref

    stack_overrides: dict[str, Any] = {}
    numeric_stack_map = {
        "pitch_um": "geometry_um.pitch",
        "lens_height_um": "geometry_um.lens_height",
        "lens_edge_gap_um": "geometry_um.lens_edge_gap",
        "cfa_thickness_um": "geometry_um.cfa_thickness",
        "passivation_thickness_um": "geometry_um.passivation_thickness",
        "si_thickness_um": "geometry_um.si_thickness",
    }
    for source_key, override_key in numeric_stack_map.items():
        if params.get(source_key) not in {None, ""}:
            stack_overrides[override_key] = params[source_key]
    if shield_mode in SHIELD_MODES:
        stack_overrides["shield.mode"] = shield_mode

    return {
        "template_id": template_id,
        "label": label or template_id,
        "source_truth_level": source_truth_level or "parametric_template_not_measured",
        "geometry_import": geometry_ref,
        "parameters": root_import_ref(parameters_path),
        "raw_cfa_pattern": raw_cfa_pattern,
        "ocl_blocks": clean_blocks,
        "solver": solver_defaults,
        "stack_overrides": stack_overrides,
    }


def root_path_from_case(value: Any, name: str) -> Path:
    if value in {None, ""}:
        raise ValueError(f"{name} is required")
    candidate = Path(str(value))
    path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{name} must stay under {ROOT}") from error
    return path


def solver_case_for_gds_import(case: dict[str, Any], geometry_json: Path) -> dict[str, Any]:
    solver_case = copy.deepcopy(case)
    for key in (
        "runner",
        "write_reference_gds",
        "input_gds",
        "gds_filename",
        "gds_map_config",
        "geometry_json_filename",
        "gds_report_filename",
        "gds_preview_filename",
        "generate_gmsh_mesh",
        "gmsh_mesh_dir",
        "gmsh_dimension",
        "gmsh_width_um",
        "gmsh_depth_um",
        "gmsh_z_width_um",
        "gmsh_mesh_um",
        "gmsh_fine_mesh_um",
        "gmsh_bbox_margin_um",
        "gmsh_include_dti_oxide",
        "gmsh_include_fd_contact",
        "gmsh_include_tg_contact",
        "gmsh_include_tg_oxide",
    ):
        solver_case.pop(key, None)
    geometry_ref = root_import_ref(geometry_json)
    solver_case["ocl_polygons"] = geometry_ref
    solver_case["cfa_polygons"] = geometry_ref
    solver_case.setdefault("id", case["id"])
    solver_case.setdefault("label", case["label"])
    return solver_case


def cad_template_defaults_by_id(template_id: str) -> dict[str, Any]:
    catalog = load_cad_template_catalog()
    template_record = next((item for item in catalog.get("templates", []) if item.get("template_id") == template_id), None)
    if not template_record:
        raise ValueError(f"Unknown CAD template: {template_id}")
    defaults = template_record.get("solver_defaults")
    if not isinstance(defaults, dict) or not defaults.get("solver"):
        raise ValueError(f"CAD template is not solver-ready: {template_id}")
    return defaults


def solver_case_for_cad_template(case: dict[str, Any]) -> dict[str, Any]:
    template_id = str(case.get("cad_template_id") or "")
    defaults = cad_template_defaults_by_id(template_id)
    allowed_case_stack, ignored_case_stack = split_cad_template_stack_overrides(case.get("stack_overrides"))
    solver_case: dict[str, Any] = {
        "id": case["id"],
        "label": case["label"],
        "preset_hint": template_id,
        **copy.deepcopy(defaults["solver"]),
        "wavelengths_nm": str(case.get("wavelengths_nm", "550")),
        "color_channel": str(case.get("color_channel", "green")),
        "cases": str(case.get("cases_arg", case.get("cases", "center:0:0:0:0:0:0"))),
        "resolution": case.get("resolution", 8),
        "after_source_time": case.get("after_source_time", 0.5),
        "pml_um": case.get("pml_um", 0.45),
        "stack_overrides": {**(defaults.get("stack_overrides") or {}), **allowed_case_stack},
        "description": "Generated from CAD template solver smoke suite.",
        "cad_template": {
            "template_id": defaults.get("template_id"),
            "label": defaults.get("label"),
            "source_truth_level": defaults.get("source_truth_level"),
            "geometry_import": defaults.get("geometry_import"),
            "parameters": defaults.get("parameters"),
            "geometry_authority": "cad_template",
            "geometry_override_policy": "Protected geometry stack keys are ignored for CAD-template requests; create a CAD variant instead.",
            "ignored_stack_override_keys": sorted(ignored_case_stack),
        },
    }
    return solver_case


def crosstalk_case_for_cad_template(case: dict[str, Any]) -> dict[str, Any]:
    template_id = str(case.get("cad_template_id") or "")
    defaults = cad_template_defaults_by_id(template_id)
    solver_defaults = copy.deepcopy(defaults.get("solver") or {})
    allowed_case_stack, ignored_case_stack = split_cad_template_stack_overrides(case.get("stack_overrides"))
    stack_overrides = {**(defaults.get("stack_overrides") or {}), **allowed_case_stack}
    crosstalk_case: dict[str, Any] = {
        "id": case["id"],
        "label": case["label"],
        "preset_hint": template_id,
        "modes": str(case.get("modes") or solver_defaults.get("mode") or "ocl-layout"),
        "layout_nx": solver_defaults.get("layout_nx"),
        "layout_nz": solver_defaults.get("layout_nz"),
        "ocl_layout": solver_defaults.get("ocl_layout"),
        "ocl_polygons": solver_defaults.get("ocl_polygons"),
        "ocl_layout_name": solver_defaults.get("ocl_layout_name") or template_id[:80],
        "target_lens_id": case.get("target_lens_id") or solver_defaults.get("target_lens_id"),
        "cfa_pattern": solver_defaults.get("cfa_pattern") or "uniform",
        "color_channel": str(case.get("color_channel") or solver_defaults.get("color_channel") or "green"),
        "neighborhoods": str(case.get("neighborhoods", case.get("neighborhood", "5"))),
        "resolutions": str(case.get("resolutions", case.get("resolution", "4"))),
        "wavelengths_nm": str(case.get("wavelengths_nm", "550")),
        "cases_arg": str(case.get("cases_arg", case.get("cases", "center:0:0:0:0"))),
        "after_source_time": case.get("after_source_time", 1),
        "source_scale": case.get("source_scale", 0.92),
        "source_profile": case.get("source_profile", "gaussian"),
        "source_sigma_scale": case.get("source_sigma_scale", 0.34),
        "guard_cells": case.get("guard_cells", 0),
        "pml_um": case.get("pml_um", 0.45),
        "stack_overrides": stack_overrides,
        "cad_template": {
            "template_id": defaults.get("template_id"),
            "label": defaults.get("label"),
            "source_truth_level": defaults.get("source_truth_level"),
            "geometry_import": defaults.get("geometry_import"),
            "parameters": defaults.get("parameters"),
            "geometry_authority": "cad_template",
            "cfa_geometry_policy": "procedural_cfa_pattern_for_large_kernel",
            "geometry_override_policy": "Protected geometry stack keys are ignored for CAD-template crosstalk requests; create a CAD variant instead.",
            "ignored_stack_override_keys": sorted(ignored_case_stack),
        },
    }
    if case.get("import_cfa_polygons"):
        crosstalk_case["cfa_polygons"] = solver_defaults.get("cfa_polygons")
    if solver_defaults.get("ocl_surface_map"):
        crosstalk_case["ocl_surface_map"] = solver_defaults.get("ocl_surface_map")
    return crosstalk_case


def command_for_gds_write_reference(gds_path: Path) -> list[str]:
    return [str(PYTHON), "gds_pixel_geometry_import.py", "--write-reference-gds", str(gds_path)]


def command_for_gds_convert(
    case: dict[str, Any],
    gds_path: Path,
    geometry_json: Path,
    report_json: Path | None = None,
    preview_svg: Path | None = None,
) -> list[str]:
    map_config = root_path_from_case(case.get("gds_map_config"), "gds_map_config")
    command = [
        str(PYTHON),
        "gds_pixel_geometry_import.py",
        str(gds_path),
        "--map-config",
        str(map_config),
        "--output-json",
        str(geometry_json),
    ]
    if report_json is not None:
        command.extend(["--report-json", str(report_json)])
    if preview_svg is not None:
        command.extend(["--preview-svg", str(preview_svg)])
    return command


def read_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def width_from_gds_report(report: dict[str, Any], case: dict[str, Any]) -> tuple[float, float]:
    bbox = report.get("bbox_um") if isinstance(report.get("bbox_um"), dict) else {}
    margin = float(case.get("gmsh_bbox_margin_um", 0.16))
    width = json_safe_number(bbox.get("width")) if isinstance(bbox, dict) else None
    height = json_safe_number(bbox.get("height")) if isinstance(bbox, dict) else None
    width_um = float(case.get("gmsh_width_um", max(1.4, (width or 1.4) + margin)))
    z_width_um = float(case.get("gmsh_z_width_um", max(1.4, (height or width or 1.4) + margin)))
    return width_um, z_width_um


def command_for_gmsh_mesh(case: dict[str, Any], report_json: Path, mesh_dir: Path) -> tuple[list[str], dict[str, Any]]:
    if not TCAD_PYTHON.exists():
        raise FileNotFoundError(f"TCAD Python environment not found: {TCAD_PYTHON}")
    report = read_json_artifact(report_json)
    width_um, z_width_um = width_from_gds_report(report, case)
    depth_um = float(case.get("gmsh_depth_um", 2.8))
    mesh_um = float(case.get("gmsh_mesh_um", 0.18))
    fine_mesh_um = float(case.get("gmsh_fine_mesh_um", 0.06))
    dimension = str(case.get("gmsh_dimension", "2"))
    command = [
        str(TCAD_PYTHON),
        "tcad_gmsh_pixel_mesh.py",
        "--dimension",
        dimension,
        "--output-dir",
        str(mesh_dir),
        "--width-um",
        f"{width_um:.9g}",
        "--depth-um",
        f"{depth_um:.9g}",
        "--z-width-um",
        f"{z_width_um:.9g}",
        "--mesh-um",
        f"{mesh_um:.9g}",
        "--fine-mesh-um",
        f"{fine_mesh_um:.9g}",
    ]
    if case.get("gmsh_include_dti_oxide", False):
        command.append("--include-dti-oxide")
    if case.get("gmsh_include_fd_contact", False):
        command.append("--include-fd-contact")
    if case.get("gmsh_include_tg_contact", False):
        command.append("--include-tg-contact")
    if case.get("gmsh_include_tg_oxide", False):
        command.append("--include-tg-oxide")
    bridge = {
        "schema": "gds_to_gmsh_bridge_v1",
        "status": "PENDING",
        "source_report": str(report_json),
        "mesh_dir": str(mesh_dir),
        "dimension": dimension,
        "width_um": width_um,
        "z_width_um": z_width_um,
        "depth_um": depth_um,
        "mesh_um": mesh_um,
        "fine_mesh_um": fine_mesh_um,
        "native_mask_polygon_mesh": False,
        "bridge_type": "gds_bbox_informed_proxy_tcad_mesh",
        "warning": "This Gmsh mesh uses GDS bbox-derived dimensions and proxy split-pixel/DTI primitives; it is not a native polygon-preserving CAD mesh.",
        "command": command,
    }
    return command, bridge


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def command_for_case(case: dict[str, Any], output_dir: Path) -> list[str] | None:
    runner = case.get("runner")
    if runner == "example":
        example = copy.deepcopy(EXAMPLES[str(case["example_id"])])
        merged = {**example, **case, "id": case["id"]}
        return command_for_lut_case(merged, output_dir)
    if runner == "meep_lut":
        return command_for_lut_case(case, output_dir)
    if runner == "cad_template_lut":
        return command_for_lut_case(solver_case_for_cad_template(case), output_dir)
    if runner == "cad_template_crosstalk":
        return command_for_crosstalk_case(crosstalk_case_for_cad_template(case), output_dir)
    if runner == "crosstalk_fdtd":
        return command_for_crosstalk_case(case, output_dir)
    if runner == "gds_import_lut":
        return None
    return None


def canonical_solver_case_for_suite_case(case: dict[str, Any]) -> dict[str, Any] | None:
    runner = case.get("runner")
    if runner == "example":
        example = copy.deepcopy(EXAMPLES[str(case["example_id"])])
        return {**example, **case, "id": case["id"]}
    if runner == "meep_lut":
        return copy.deepcopy(case)
    if runner == "cad_template_lut":
        return solver_case_for_cad_template(case)
    if runner == "cad_template_crosstalk":
        return crosstalk_case_for_cad_template(case)
    if runner == "gds_import_lut":
        pipeline_path = Path(case.get("_case_dir", "")) / "pipeline_solver_case.json"
        return read_json_artifact(pipeline_path) if pipeline_path.exists() else None
    return None


def write_suite_case_provenance(case: dict[str, Any], case_dir: Path, command: list[str] | None = None) -> dict[str, str | None]:
    case_dir.mkdir(parents=True, exist_ok=True)
    case_input_path = case_dir / "case_input.json"
    command_path = case_dir / "case_command.json"
    solver_case_path = case_dir / "solver_case.json"
    case_input = {
        "schema": "pixel_workbench_suite_case_input_v1",
        "case_id": case.get("id"),
        "label": case.get("label"),
        "runner": case.get("runner"),
        "active_tier": case.get("active_tier"),
        "case": copy.deepcopy({key: value for key, value in case.items() if key != "_case_dir"}),
    }
    write_json_artifact(case_input_path, case_input)
    if command:
        command_payload = {
            "schema": "pixel_workbench_suite_case_command_v1",
            "case_id": case.get("id"),
            "runner": case.get("runner"),
            "cwd": str(ROOT),
            "command": command,
            "artifacts": {},
        }
        write_json_artifact(command_path, command_payload)
        command_payload["artifacts"]["case_command"] = rel_url(command_path)
        write_json_artifact(command_path, command_payload)
    canonical_case = canonical_solver_case_for_suite_case({**case, "_case_dir": str(case_dir)})
    if canonical_case:
        write_json_artifact(solver_case_path, canonical_case)
    return {
        "case_input": rel_url(case_input_path),
        "case_command": rel_url(command_path),
        "solver_case": rel_url(solver_case_path),
    }


def summarize_lut(output_dir: Path) -> dict[str, Any]:
    summary_csv = output_dir / "camera_lut_summary.csv"
    long_csv = output_dir / "camera_lut_long.csv"
    lut_json_path = output_dir / "camera_lut.json"
    convergence_path = output_dir / "convergence_report.json"
    rows = read_csv_rows(summary_csv)
    long_rows = read_csv_rows(long_csv)
    totals = [value for value in (json_safe_number(row.get("total_response")) for row in rows) if value is not None]
    normalized = [
        value
        for value in (json_safe_number(row.get("normalized_total_response_to_first")) for row in rows)
        if value is not None
    ]
    signed_flux = [
        value
        for value in (
            json_safe_number(row.get("signed_flux_si_absorption_fraction_diagnostic")) for row in rows
        )
        if value is not None
    ]
    grid_gates = [value for value in (csv_bool(row.get("grid_resolution_gate_pass")) for row in rows) if value is not None]
    si_pixels = [
        value for value in (json_safe_number(row.get("si_internal_wavelength_pixels")) for row in rows) if value is not None
    ]
    feature_pixels = [
        value for value in (json_safe_number(row.get("minimum_critical_feature_pixels")) for row in rows) if value is not None
    ]
    recommended_resolution = [
        value for value in (json_safe_number(row.get("recommended_min_resolution_px_per_um")) for row in rows) if value is not None
    ]
    grid_notes = sorted({row.get("grid_resolution_notes", "") for row in rows if row.get("grid_resolution_notes")})
    center = next((row for row in rows if row.get("case") == "center"), rows[0] if rows else {})
    edge = next((row for row in rows if "edge" in str(row.get("case", "")).lower()), rows[-1] if rows else {})
    center_total = json_safe_number(center.get("total_response"))
    edge_total = json_safe_number(edge.get("total_response"))
    edge_to_center = edge_total / center_total if center_total and edge_total is not None else None
    edge_delta_pct = (edge_to_center - 1.0) * 100.0 if edge_to_center is not None else None
    region_max = [value for value in (json_safe_number(row.get("max_region_response")) for row in rows) if value is not None]
    region_min = [value for value in (json_safe_number(row.get("min_region_response")) for row in rows) if value is not None]
    imbalance = None
    if region_max and region_min and max(region_max) > 0:
        imbalance = 1.0 - min(region_min) / max(region_max)
    split_phase_x = [
        value for value in (json_safe_number(row.get("split_phase_x_proxy")) for row in rows) if value is not None
    ]
    split_phase_z = [
        value for value in (json_safe_number(row.get("split_phase_z_proxy")) for row in rows) if value is not None
    ]
    phase_amplitudes = [
        math.hypot(x_value or 0.0, z_value or 0.0)
        for x_value, z_value in (
            (
                json_safe_number(row.get("split_phase_x_proxy")),
                json_safe_number(row.get("split_phase_z_proxy")),
            )
            for row in rows
        )
        if x_value is not None or z_value is not None
    ]
    collection_modes = sorted({row.get("collection_mode", "") for row in rows if row.get("collection_mode")})
    target_lens_ids = sorted({row.get("target_lens_id", "") for row in rows if row.get("target_lens_id")})
    cfa_polygon_counts = [
        value for value in (json_safe_number(row.get("cfa_polygon_count")) for row in rows) if value is not None
    ]
    cfa_polygon_backgrounds = sorted(
        {row.get("cfa_polygon_background", "") for row in rows if row.get("cfa_polygon_background")}
    )

    lut_meta: dict[str, Any] = {}
    if lut_json_path.exists():
        try:
            with lut_json_path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            lut_meta = {
                "schema": raw.get("schema"),
                "mode": raw.get("mode"),
                "resolution_px_per_um": raw.get("resolution_px_per_um"),
                "after_source_time": raw.get("after_source_time"),
                "cell_pixels": raw.get("cell_pixels"),
                "collection_mode": raw.get("collection_mode"),
                "split_mode": raw.get("split_mode"),
                "target_lens_id": raw.get("target_lens_id"),
                "cfa": raw.get("cfa", {}),
                "ocl_layout": raw.get("ocl_layout", {}),
                "notes": raw.get("notes", []),
            }
        except json.JSONDecodeError:
            lut_meta = {"json_error": "camera_lut.json could not be parsed"}
    cfa_meta = lut_meta.get("cfa", {}) if isinstance(lut_meta.get("cfa"), dict) else {}
    ocl_meta = lut_meta.get("ocl_layout", {}) if isinstance(lut_meta.get("ocl_layout"), dict) else {}
    geometry_sources = {
        "cfa_polygons": cfa_meta.get("polygons_source"),
        "ocl_polygons": ocl_meta.get("polygons_source"),
        "ocl_sag": ocl_meta.get("sag_source"),
        "ocl_surface_map": ocl_meta.get("surface_map_source"),
    }
    imported_geometry_sources = sorted(
        {
            source
            for source in geometry_sources.values()
            if isinstance(source, str) and source.startswith("@")
        }
    )

    convergence: dict[str, Any] | None = None
    if convergence_path.exists():
        try:
            with convergence_path.open(encoding="utf-8") as handle:
                convergence = json.load(handle)
        except json.JSONDecodeError:
            convergence = {"passed": False, "error": "convergence_report.json could not be parsed"}

    negative_signed_flux_count = sum(1 for value in signed_flux if value < 0)
    grid_gate_fail_count = sum(1 for value in grid_gates if value is False)
    numerical_gate_passed = bool(grid_gates) and grid_gate_fail_count == 0
    kpi_status = (
        "PASS"
        if rows
        and lut_json_path.exists()
        and negative_signed_flux_count == 0
        and numerical_gate_passed
        else "CHECK"
    )

    return {
        "schema": "pixel_workbench_solver_kpi_v1",
        "output_dir": str(output_dir),
        "summary_csv_url": rel_url(summary_csv),
        "lut_json_url": rel_url(lut_json_path),
        "row_count": len(rows),
        "long_row_count": len(long_rows),
        "cases": [row.get("case") for row in rows],
        "wavelengths_nm": sorted({json_safe_number(row.get("wavelength_nm")) for row in rows if row.get("wavelength_nm")}),
        "total_response_min": min(totals) if totals else None,
        "total_response_max": max(totals) if totals else None,
        "center_total_response": center_total,
        "edge_total_response": edge_total,
        "edge_to_center_response": edge_to_center,
        "edge_delta_pct": edge_delta_pct,
        "normalized_response_min": min(normalized) if normalized else None,
        "normalized_response_max": max(normalized) if normalized else None,
        "region_imbalance_proxy": imbalance,
        "split_phase_x_min": min(split_phase_x) if split_phase_x else None,
        "split_phase_x_max": max(split_phase_x) if split_phase_x else None,
        "split_phase_z_min": min(split_phase_z) if split_phase_z else None,
        "split_phase_z_max": max(split_phase_z) if split_phase_z else None,
        "split_phase_amplitude_max": max(phase_amplitudes) if phase_amplitudes else None,
        "collection_modes": collection_modes,
        "target_lens_ids": target_lens_ids,
        "cfa_polygon_count_max": max(cfa_polygon_counts) if cfa_polygon_counts else None,
        "cfa_polygon_backgrounds": cfa_polygon_backgrounds,
        "geometry_sources": geometry_sources,
        "imported_geometry_sources": imported_geometry_sources,
        "imported_geometry": bool(imported_geometry_sources),
        "negative_signed_flux_count": negative_signed_flux_count,
        "signed_flux_available": bool(signed_flux),
        "numerical_gate": {
            "available": bool(grid_gates),
            "passed": numerical_gate_passed,
            "grid_resolution_gate_fail_count": grid_gate_fail_count,
            "min_si_internal_wavelength_pixels": min(si_pixels) if si_pixels else None,
            "min_critical_feature_pixels": min(feature_pixels) if feature_pixels else None,
            "recommended_min_resolution_px_per_um": max(recommended_resolution) if recommended_resolution else None,
            "notes": grid_notes,
        },
        "artifacts": {
            "response_maps": rel_url(output_dir / "response_maps.png"),
            "focal_maps": rel_url(output_dir / "focal_maps.png"),
            "camera_lut_npz": rel_url(output_dir / "camera_lut.npz"),
            "tcad_generation_profile_1d": rel_url(output_dir / "tcad_generation_profile_1d.csv"),
            "tcad_generation_volume_3d": rel_url(output_dir / "tcad_generation_volume_3d.npz"),
        },
        "lut": lut_meta,
        "convergence": convergence,
        "status": kpi_status,
    }


def chart_from_lut(output_dir: Path) -> dict[str, Any]:
    summary_rows = read_csv_rows(output_dir / "camera_lut_summary.csv")
    long_rows = read_csv_rows(output_dir / "camera_lut_long.csv")
    cra_points = []
    for row in summary_rows:
        cra_points.append(
            {
                "case": row.get("case"),
                "cra_x_deg": json_safe_number(row.get("cra_x_deg")),
                "cra_z_deg": json_safe_number(row.get("cra_z_deg")),
                "total_response": json_safe_number(row.get("total_response")),
                "split_phase_x_proxy": json_safe_number(row.get("split_phase_x_proxy")),
                "split_phase_z_proxy": json_safe_number(row.get("split_phase_z_proxy")),
                "edge_to_center": json_safe_number(row.get("normalized_total_response_to_first")),
            }
        )
    first_case = summary_rows[0].get("case") if summary_rows else None
    matrix = []
    for row in long_rows:
        if first_case and row.get("case") != first_case:
            continue
        matrix.append(
            {
                "region_id": row.get("region_id"),
                "ix": json_safe_number(row.get("region_ix")),
                "iz": json_safe_number(row.get("region_iz")),
                "response": json_safe_number(row.get("response")),
                "normalized": json_safe_number(row.get("normalized_region_response_to_first_same_region")),
            }
        )
    return {
        "cra_response_curve": {"type": "line", "x": "cra_x_deg", "y": "total_response", "points": cra_points},
        "subpixel_response_matrix": {"type": "matrix", "case": first_case, "cells": matrix},
    }


def summarize_existing_crosstalk(case: dict[str, Any]) -> dict[str, Any]:
    summary_path = ROOT / str(case["summary_csv"])
    rows = read_csv_rows(summary_path)
    crosstalk_values = [
        value
        for value in (json_safe_number(row.get("output_crosstalk_fraction")) for row in rows)
        if value is not None
    ]
    strongest_neighbor = [
        value
        for value in (json_safe_number(row.get("strongest_neighbor_fraction")) for row in rows)
        if value is not None
    ]
    truncation = [
        value
        for value in (json_safe_number(row.get("truncation_response_fraction")) for row in rows)
        if value is not None
    ]
    grid_gates = [value for value in (csv_bool(row.get("grid_resolution_gate_pass")) for row in rows) if value is not None]
    points = []
    for row in rows:
        points.append(
            {
                "case": row.get("case"),
                "mode": row.get("mode"),
                "resolution_px_per_um": json_safe_number(row.get("resolution_px_per_um")),
                "wavelength_nm": json_safe_number(row.get("wavelength_nm")),
                "cra_x_deg": json_safe_number(row.get("cra_x_deg")),
                "output_crosstalk_fraction": json_safe_number(row.get("output_crosstalk_fraction")),
                "strongest_neighbor_fraction": json_safe_number(row.get("strongest_neighbor_fraction")),
                "truncation_response_fraction": json_safe_number(row.get("truncation_response_fraction")),
                "grid_resolution_gate_pass": csv_bool(row.get("grid_resolution_gate_pass")),
            }
        )
    artifact = ROOT / str(case.get("artifact_png", ""))
    gate_pass = bool(grid_gates) and all(grid_gates)
    return {
        "schema": "pixel_workbench_crosstalk_kpi_v1",
        "status": "PASS" if rows and gate_pass else "CHECK",
        "row_count": len(rows),
        "summary_csv_url": rel_url(summary_path),
        "artifacts": {"crosstalk_plot": rel_url(artifact)},
        "output_crosstalk_fraction_max": max(crosstalk_values) if crosstalk_values else None,
        "strongest_neighbor_fraction_max": max(strongest_neighbor) if strongest_neighbor else None,
        "truncation_fraction_max": max(truncation) if truncation else None,
        "numerical_gate": {
            "available": bool(grid_gates),
            "passed": gate_pass,
            "grid_resolution_gate_fail_count": sum(1 for value in grid_gates if value is False),
        },
        "charts": {
            "crosstalk_kernel_heatmap": {
                "type": "crosstalk_points",
                "points": points,
                "image_url": rel_url(artifact),
            }
        },
    }


def summarize_crosstalk_output(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "crosstalk_kernel_summary.csv"
    output_path = output_dir / "crosstalk_output_kernel.csv"
    manifest_path = output_dir / "crosstalk_kernel.json"
    convergence_path = output_dir / "crosstalk_convergence.json"
    heatmap_path = output_dir / "crosstalk_kernel_heatmap.png"
    rows = read_csv_rows(summary_path)
    output_rows = read_csv_rows(output_path)
    crosstalk_values = [
        value
        for value in (json_safe_number(row.get("output_crosstalk_fraction")) for row in rows)
        if value is not None
    ]
    strongest_neighbor = [
        value
        for value in (json_safe_number(row.get("strongest_neighbor_fraction")) for row in rows)
        if value is not None
    ]
    truncation = [
        value
        for value in (json_safe_number(row.get("truncation_response_fraction")) for row in rows)
        if value is not None
    ]
    grid_gates = [value for value in (csv_bool(row.get("grid_resolution_gate_pass")) for row in rows) if value is not None]
    convergence: dict[str, Any] | None = None
    if convergence_path.exists():
        try:
            with convergence_path.open(encoding="utf-8") as handle:
                convergence = json.load(handle)
        except json.JSONDecodeError:
            convergence = {"status": "FAIL", "error": "crosstalk_convergence.json could not be parsed"}
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError:
            manifest = {"json_error": "crosstalk_kernel.json could not be parsed"}
    manifest_config = manifest.get("configuration", {}) if isinstance(manifest.get("configuration"), dict) else {}
    geometry_sources = {
        "cfa_polygons": manifest_config.get("cfa_polygons_source"),
        "ocl_polygons": manifest_config.get("ocl_polygons_source"),
        "ocl_sag": manifest_config.get("ocl_sag_source"),
        "ocl_surface_map": manifest_config.get("ocl_surface_map_source"),
    }
    imported_geometry_sources = sorted(
        {
            source
            for source in geometry_sources.values()
            if isinstance(source, str) and source.startswith("@")
        }
    )
    points = []
    for row in rows:
        points.append(
            {
                "case": row.get("case"),
                "mode": row.get("mode"),
                "layout_label": row.get("layout_label"),
                "target_lens_id": row.get("target_lens_id"),
                "resolution_px_per_um": json_safe_number(row.get("resolution_px_per_um")),
                "wavelength_nm": json_safe_number(row.get("wavelength_nm")),
                "cra_x_deg": json_safe_number(row.get("cra_x_deg")),
                "cra_z_deg": json_safe_number(row.get("cra_z_deg")),
                "output_crosstalk_fraction": json_safe_number(row.get("output_crosstalk_fraction")),
                "strongest_neighbor_fraction": json_safe_number(row.get("strongest_neighbor_fraction")),
                "truncation_response_fraction": json_safe_number(row.get("truncation_response_fraction")),
                "grid_resolution_gate_pass": csv_bool(row.get("grid_resolution_gate_pass")),
            }
        )
    kernel_cells = []
    for row in output_rows:
        kernel_cells.append(
            {
                "case": row.get("case"),
                "region_id": row.get("region_id"),
                "ocl_lens_id": row.get("ocl_lens_id"),
                "ocl_lens_kind": row.get("ocl_lens_kind"),
                "output_dx": json_safe_number(row.get("output_dx")),
                "output_dz": json_safe_number(row.get("output_dz")),
                "output_dx_um": json_safe_number(row.get("output_dx_um")),
                "output_dz_um": json_safe_number(row.get("output_dz_um")),
                "response_fraction": json_safe_number(row.get("response_fraction")),
            }
        )
    gate_pass = bool(grid_gates) and all(grid_gates) and bool(convergence) and convergence.get("status") == "PASS"
    return {
        "schema": "pixel_workbench_crosstalk_kpi_v1",
        "status": "PASS" if rows and gate_pass else "CHECK",
        "row_count": len(rows),
        "output_row_count": len(output_rows),
        "summary_csv_url": rel_url(summary_path),
        "output_kernel_csv_url": rel_url(output_path),
        "manifest_url": rel_url(manifest_path),
        "artifacts": {"crosstalk_plot": rel_url(heatmap_path)},
        "output_crosstalk_fraction_max": max(crosstalk_values) if crosstalk_values else None,
        "strongest_neighbor_fraction_max": max(strongest_neighbor) if strongest_neighbor else None,
        "truncation_fraction_max": max(truncation) if truncation else None,
        "geometry_sources": geometry_sources,
        "imported_geometry_sources": imported_geometry_sources,
        "imported_geometry": bool(imported_geometry_sources),
        "numerical_gate": {
            "available": bool(grid_gates),
            "passed": gate_pass,
            "grid_resolution_gate_fail_count": sum(1 for value in grid_gates if value is False),
            "convergence_status": convergence.get("status") if convergence else None,
            "convergence": convergence,
        },
        "charts": {
            "crosstalk_kernel_heatmap": {
                "type": "crosstalk_points",
                "points": points,
                "cells": kernel_cells,
                "image_url": rel_url(heatmap_path),
            }
        },
    }


def analysis_only_result(case: dict[str, Any]) -> dict[str, Any]:
    notes = case.get("decision_notes", [])
    return {
        "schema": "pixel_workbench_analysis_only_v1",
        "status": "CHECK",
        "row_count": 0,
        "analysis_only": True,
        "design_factors": case.get("design_factors", {}),
        "decision_notes": notes,
        "gates": {
            "solver_primitive": "missing",
            "product_lut_ready": False,
            "reason": notes[0] if notes else "This test case is a planning/risk item until the solver primitive is implemented.",
        },
        "charts": {
            "boundary_risk_table": {
                "type": "risk_table",
                "rows": [
                    {"risk": "OCL transition discontinuity", "status": "needs mixed-supercell FDTD"},
                    {"risk": "Remosaic boundary artifact", "status": "needs pattern-level camera pipeline model"},
                    {"risk": "Measured material correlation", "status": "blocked until measured n,k import"},
                ],
            }
        },
    }


def case_result_from_output(case: dict[str, Any], case_dir: Path, return_code: int | None = None) -> dict[str, Any]:
    runner = case.get("runner")
    if runner in {"example", "meep_lut", "gds_import_lut", "cad_template_lut"}:
        kpi = summarize_lut(case_dir)
        if runner == "cad_template_lut":
            template_id = str(case.get("cad_template_id") or "")
            defaults = cad_template_defaults_by_id(template_id)
            _allowed_case_stack, ignored_case_stack = split_cad_template_stack_overrides(case.get("stack_overrides"))
            template_record = find_cad_template(load_cad_template_catalog(), template_id)
            template_tcad = template_record.get("tcad_bridge", {}) if isinstance(template_record.get("tcad_bridge"), dict) else {}
            template_artifacts = template_record.get("artifacts", {}) if isinstance(template_record.get("artifacts"), dict) else {}
            coupled_dd_smoke = run_coupled_tcad_dd_smoke_for_case(template_id, case_dir)
            kpi["cad_template"] = {
                "template_id": defaults.get("template_id"),
                "label": defaults.get("label"),
                "source_truth_level": defaults.get("source_truth_level"),
                "geometry_import": defaults.get("geometry_import"),
                "parameters": defaults.get("parameters"),
                "geometry_authority": "cad_template",
                "geometry_override_policy": "Protected geometry stack keys are ignored for CAD-template requests; create a CAD variant instead.",
                "ignored_stack_override_keys": sorted(ignored_case_stack),
                "freecad_validation": template_record.get("freecad_validation", {}),
                "design_rule_validation": template_record.get("design_rule_validation", {}),
                "tcad_bridge": template_tcad,
                "coupled_tcad_dd_smoke": coupled_dd_smoke,
                "artifacts": {
                    "fcstd": template_artifacts.get("fcstd", {}).get("url")
                    if isinstance(template_artifacts.get("fcstd"), dict)
                    else None,
                    "freecad_validation_report": template_artifacts.get("freecad_validation_report", {}).get("url")
                    if isinstance(template_artifacts.get("freecad_validation_report"), dict)
                    else None,
                    "tcad_bridge_report": template_artifacts.get("tcad_bridge_report", {}).get("url")
                    if isinstance(template_artifacts.get("tcad_bridge_report"), dict)
                    else None,
                    "tcad_mesh_2d": template_artifacts.get("tcad_mesh_2d", {}).get("url")
                    if isinstance(template_artifacts.get("tcad_mesh_2d"), dict)
                    else None,
                    "devsim_import_summary": template_artifacts.get("devsim_import_summary", {}).get("url")
                    if isinstance(template_artifacts.get("devsim_import_summary"), dict)
                    else None,
                    "devsim_dd_summary": template_artifacts.get("devsim_dd_summary", {}).get("url")
                    if isinstance(template_artifacts.get("devsim_dd_summary"), dict)
                    else None,
                    "devsim_split_currents_plot": template_artifacts.get("devsim_split_currents_plot", {}).get("url")
                    if isinstance(template_artifacts.get("devsim_split_currents_plot"), dict)
                    else None,
                    "devsim_node_maps_plot": template_artifacts.get("devsim_node_maps_plot", {}).get("url")
                    if isinstance(template_artifacts.get("devsim_node_maps_plot"), dict)
                    else None,
                },
            }
        if runner == "gds_import_lut":
            geometry_json = case_dir / str(case.get("geometry_json_filename", "pixel_geometry_from_gds.json"))
            report_json = case_dir / str(case.get("gds_report_filename", "gds_import_report.json"))
            preview_svg = case_dir / str(case.get("gds_preview_filename", "gds_import_preview.svg"))
            gds_path = case_dir / str(case.get("gds_filename", "reference_pixel_masks.gds"))
            pipeline_solver_case = case_dir / "pipeline_solver_case.json"
            gmsh_mesh_dir = case_dir / str(case.get("gmsh_mesh_dir", "gmsh_mesh"))
            gmsh_bridge_report = case_dir / "gmsh_bridge_report.json"
            gmsh_mesh_metadata = gmsh_mesh_dir / "mesh_metadata.json"
            gmsh_mesh_2d = gmsh_mesh_dir / "split_pixel_2d.msh"
            gmsh_mesh_3d = gmsh_mesh_dir / "split_pixel_3d.msh"
            conversion: dict[str, Any] = {}
            validation_report: dict[str, Any] = read_json_artifact(report_json)
            if geometry_json.exists():
                try:
                    with geometry_json.open(encoding="utf-8") as handle:
                        geometry_payload = json.load(handle)
                    conversion = geometry_payload.get("gds_import", {}) if isinstance(geometry_payload, dict) else {}
                except json.JSONDecodeError:
                    conversion = {"json_error": "Converted GDS geometry JSON could not be parsed"}
            kpi["gds_import_pipeline"] = True
            kpi["gds_import"] = conversion
            if validation_report:
                validation = {
                    "status": validation_report.get("validation_status", "CHECK"),
                    "warning_count": len(validation_report.get("warnings", [])),
                    "error_count": len(validation_report.get("errors", [])),
                    "polygon_count": validation_report.get("polygon_count"),
                    "matched_ocl_polygon_count": validation_report.get("matched_ocl_polygon_count"),
                    "matched_cfa_polygon_count": validation_report.get("matched_cfa_polygon_count"),
                    "bbox_um": validation_report.get("bbox_um"),
                    "warnings": validation_report.get("warnings", [])[:5],
                }
                kpi["gds_import_validation"] = validation
                if validation["status"] == "FAIL":
                    kpi["status"] = "FAIL"
                elif validation["status"] == "CHECK" and kpi.get("status") == "PASS":
                    kpi["status"] = "CHECK"
            if gmsh_mesh_metadata.exists():
                kpi["gmsh_mesh_bridge"] = True
                kpi["gmsh_native_mask_polygon_mesh"] = False
                kpi["gmsh_bridge"] = read_json_artifact(gmsh_bridge_report)
            kpi.setdefault("artifacts", {})
            kpi["artifacts"].update(
                {
                    "input_gds": rel_url(gds_path),
                    "converted_geometry_json": rel_url(geometry_json),
                    "gds_import_report": rel_url(report_json),
                    "gds_import_preview": rel_url(preview_svg),
                    "pipeline_solver_case": rel_url(pipeline_solver_case),
                    "gmsh_bridge_report": rel_url(gmsh_bridge_report),
                    "gmsh_mesh_metadata": rel_url(gmsh_mesh_metadata),
                    "gmsh_mesh_2d": rel_url(gmsh_mesh_2d),
                    "gmsh_mesh_3d": rel_url(gmsh_mesh_3d),
                }
            )
        return {
            "id": case["id"],
            "label": case["label"],
            "runner": runner,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "design_factors": case.get("design_factors", {}),
            "decision_notes": case.get("decision_notes", []),
            "kpi": kpi,
            "charts": chart_from_lut(case_dir),
            "output_url": rel_url(case_dir),
        }
    if runner == "existing_crosstalk":
        kpi = summarize_existing_crosstalk(case)
        return {
            "id": case["id"],
            "label": case["label"],
            "runner": runner,
            "status": "completed",
            "return_code": 0,
            "design_factors": case.get("design_factors", {}),
            "decision_notes": case.get("decision_notes", []),
            "kpi": kpi,
            "charts": kpi.get("charts", {}),
            "output_url": None,
        }
    if runner in {"crosstalk_fdtd", "cad_template_crosstalk"}:
        kpi = summarize_crosstalk_output(case_dir)
        if runner == "cad_template_crosstalk":
            template_id = str(case.get("cad_template_id") or "")
            defaults = cad_template_defaults_by_id(template_id)
            _allowed_case_stack, ignored_case_stack = split_cad_template_stack_overrides(case.get("stack_overrides"))
            template_record = find_cad_template(load_cad_template_catalog(), template_id)
            dimension_summary = (
                template_record.get("dimension_summary", {})
                if isinstance(template_record.get("dimension_summary"), dict)
                else {}
            )
            kpi["cad_template"] = {
                "template_id": defaults.get("template_id"),
                "label": defaults.get("label"),
                "source_truth_level": defaults.get("source_truth_level"),
                "geometry_import": defaults.get("geometry_import"),
                "parameters": defaults.get("parameters"),
                "geometry_authority": "cad_template",
                "cfa_geometry_policy": "procedural_cfa_pattern_for_large_kernel",
                "geometry_override_policy": "Protected geometry stack keys are ignored for CAD-template crosstalk requests; create a CAD variant instead.",
                "ignored_stack_override_keys": sorted(ignored_case_stack),
                "crosstalk_kernel_status": dimension_summary.get("crosstalk_kernel_status"),
                "crosstalk_kernel_label": dimension_summary.get("crosstalk_kernel_label"),
                "effective_ocl_group_pitch_label": dimension_summary.get("effective_ocl_group_pitch_label"),
                "footprint_x_um": dimension_summary.get("footprint_x_um"),
                "footprint_z_um": dimension_summary.get("footprint_z_um"),
            }
        return {
            "id": case["id"],
            "label": case["label"],
            "runner": runner,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "design_factors": case.get("design_factors", {}),
            "decision_notes": case.get("decision_notes", []),
            "kpi": kpi,
            "charts": kpi.get("charts", {}),
            "output_url": rel_url(case_dir),
        }
    if runner == "analysis_only":
        kpi = analysis_only_result(case)
        return {
            "id": case["id"],
            "label": case["label"],
            "runner": runner,
            "status": "completed",
            "return_code": 0,
            "design_factors": case.get("design_factors", {}),
            "decision_notes": case.get("decision_notes", []),
            "kpi": kpi,
            "charts": kpi.get("charts", {}),
            "output_url": None,
        }
    raise ValueError(f"Unsupported suite runner: {runner}")


def cad_variant_delta_rows(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_template: dict[str, dict[str, Any]] = {}
    variants: list[dict[str, Any]] = []
    for result in case_results:
        factors = result.get("design_factors") if isinstance(result.get("design_factors"), dict) else {}
        template_id = str(factors.get("template") or "")
        role = str(factors.get("role") or "")
        kpi = result.get("kpi") or {}
        total = json_safe_number(kpi.get("center_total_response"))
        if not template_id or total is None:
            continue
        cad_template = kpi.get("cad_template") if isinstance(kpi.get("cad_template"), dict) else {}
        tcad_bridge = cad_template.get("tcad_bridge") if isinstance(cad_template.get("tcad_bridge"), dict) else {}
        static_dd_smoke = (
            tcad_bridge.get("devsim_dd_smoke")
            if isinstance(tcad_bridge.get("devsim_dd_smoke"), dict)
            else {}
        )
        coupled_dd = (
            cad_template.get("coupled_tcad_dd_smoke")
            if isinstance(cad_template.get("coupled_tcad_dd_smoke"), dict)
            else {}
        )
        coupled_summary = coupled_dd.get("summary") if isinstance(coupled_dd.get("summary"), dict) else {}
        dd_smoke = coupled_summary if coupled_dd.get("available") else static_dd_smoke
        row = {
            "case_id": result.get("id"),
            "label": result.get("label"),
            "template_id": template_id,
            "center_total_response": total,
            "dd_smoke_available": bool(coupled_dd.get("available") or static_dd_smoke.get("available")),
            "dd_smoke_source": "scaled_fdtd_generation_map_2d_smoke"
            if coupled_dd.get("available")
            else "template_uniform_generation",
            "dd_coupled_status": coupled_dd.get("status"),
            "dd_generation_integral_summary": dd_smoke.get("generation_integral_summary"),
            "dd_split_phase_x_proxy": json_safe_number(dd_smoke.get("photo_split_phase_x_proxy")),
            "dd_left_photo_delta_a_per_cm": json_safe_number(dd_smoke.get("left_photo_delta_a_per_cm")),
            "dd_right_photo_delta_a_per_cm": json_safe_number(dd_smoke.get("right_photo_delta_a_per_cm")),
            "dd_terminal_balance_a_per_cm": json_safe_number(dd_smoke.get("terminal_current_balance_illuminated_a_per_cm")),
            "dd_node_count": json_safe_number(dd_smoke.get("node_count")),
            "dd_electrical_model": dd_smoke.get("electrical_model"),
        }
        if role == "base":
            base_by_template[template_id] = row
        elif role == "variant":
            row["variant_of"] = str(factors.get("variant_of") or "")
            row["overrides"] = factors.get("overrides")
            variants.append(row)

    rows = []
    for variant in variants:
        base = base_by_template.get(str(variant.get("variant_of") or ""))
        base_total = json_safe_number(base.get("center_total_response")) if base else None
        variant_total = json_safe_number(variant.get("center_total_response"))
        base_dd_phase = json_safe_number(base.get("dd_split_phase_x_proxy")) if base else None
        variant_dd_phase = json_safe_number(variant.get("dd_split_phase_x_proxy"))
        delta_abs = variant_total - base_total if variant_total is not None and base_total is not None else None
        delta_pct = (
            (variant_total / base_total - 1.0) * 100.0
            if variant_total is not None and base_total not in {None, 0}
            else None
        )
        dd_delta_abs = (
            variant_dd_phase - base_dd_phase
            if variant_dd_phase is not None and base_dd_phase is not None
            else None
        )
        rows.append(
            {
                "type": "cad_variant_delta",
                "case_id": variant.get("case_id"),
                "label": variant.get("label"),
                "template_id": variant.get("template_id"),
                "variant_of": variant.get("variant_of"),
                "base_case_id": base.get("case_id") if base else None,
                "base_label": base.get("label") if base else None,
                "overrides": variant.get("overrides"),
                "base_center_total_response": base_total,
                "variant_center_total_response": variant_total,
                "delta_abs": delta_abs,
                "delta_pct_vs_base": delta_pct,
                "base_dd_split_phase_x_proxy": base_dd_phase,
                "variant_dd_split_phase_x_proxy": variant_dd_phase,
                "dd_split_phase_delta_abs": dd_delta_abs,
                "dd_smoke_available": bool(base and base.get("dd_smoke_available") and variant.get("dd_smoke_available")),
                "dd_smoke_source": variant.get("dd_smoke_source") or (base.get("dd_smoke_source") if base else None),
                "dd_coupled_status": variant.get("dd_coupled_status"),
                "dd_electrical_model": variant.get("dd_electrical_model") or (base.get("dd_electrical_model") if base else None),
            }
        )
    return rows


def aggregate_suite_result(suite: dict[str, Any], tier: str, case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_responses = []
    edge_ratios = []
    phase_amplitudes = []
    cfa_polygon_case_count = 0
    split_collection_case_count = 0
    imported_geometry_case_count = 0
    gds_import_case_count = 0
    gmsh_bridge_case_count = 0
    gate_failures = []
    chart_cases = []
    tornado_rows = []
    for result in case_results:
        kpi = result.get("kpi") or {}
        if kpi.get("status") != "PASS":
            gate_failures.append({"case_id": result["id"], "status": kpi.get("status", "CHECK")})
        total = json_safe_number(kpi.get("center_total_response"))
        edge = json_safe_number(kpi.get("edge_to_center_response"))
        if total is not None:
            total_responses.append(total)
            tornado_rows.append({"case_id": result["id"], "label": result["label"], "center_total_response": total})
        if edge is not None:
            edge_ratios.append(edge)
        phase = json_safe_number(kpi.get("split_phase_amplitude_max"))
        if phase is not None:
            phase_amplitudes.append(phase)
        if json_safe_number(kpi.get("cfa_polygon_count_max")):
            cfa_polygon_case_count += 1
        if "split-pd" in (kpi.get("collection_modes") or []):
            split_collection_case_count += 1
        if kpi.get("imported_geometry"):
            imported_geometry_case_count += 1
        if kpi.get("gds_import_pipeline"):
            gds_import_case_count += 1
        if kpi.get("gmsh_mesh_bridge"):
            gmsh_bridge_case_count += 1
        if result.get("charts"):
            chart_cases.append({"case_id": result["id"], "charts": result["charts"]})
    nominal = next((row["center_total_response"] for row in tornado_rows if "nominal" in row["case_id"]), None)
    if nominal:
        for row in tornado_rows:
            row["delta_pct_vs_nominal"] = (row["center_total_response"] / nominal - 1.0) * 100.0
    variant_delta_rows = cad_variant_delta_rows(case_results)
    variant_delta_values = [
        abs(row["delta_pct_vs_base"])
        for row in variant_delta_rows
        if isinstance(row.get("delta_pct_vs_base"), (int, float)) and math.isfinite(row["delta_pct_vs_base"])
    ]
    variant_dd_delta_values = [
        abs(row["dd_split_phase_delta_abs"])
        for row in variant_delta_rows
        if isinstance(row.get("dd_split_phase_delta_abs"), (int, float))
        and math.isfinite(row["dd_split_phase_delta_abs"])
    ]
    variant_dd_available_count = sum(1 for row in variant_delta_rows if row.get("dd_smoke_available"))
    suite_status = "PASS" if case_results and not gate_failures else "CHECK"
    return {
        "schema": "pixel_workbench_suite_result_v1",
        "suite_id": suite["id"],
        "suite_label": suite["label"],
        "tier": tier,
        "status": suite_status,
        "product_lut_ready": False,
        "case_count": len(case_results),
        "kpi_summary": {
            "total_response_min": min(total_responses) if total_responses else None,
            "total_response_max": max(total_responses) if total_responses else None,
            "edge_to_center_min": min(edge_ratios) if edge_ratios else None,
            "edge_to_center_max": max(edge_ratios) if edge_ratios else None,
            "split_phase_amplitude_max": max(phase_amplitudes) if phase_amplitudes else None,
            "cfa_polygon_case_count": cfa_polygon_case_count,
            "split_collection_case_count": split_collection_case_count,
            "imported_geometry_case_count": imported_geometry_case_count,
            "gds_import_case_count": gds_import_case_count,
            "gmsh_bridge_case_count": gmsh_bridge_case_count,
            "cad_variant_delta_count": len(variant_delta_rows),
            "cad_variant_max_abs_delta_pct": max(variant_delta_values) if variant_delta_values else None,
            "cad_variant_dd_available_count": variant_dd_available_count,
            "cad_variant_dd_max_abs_split_delta": max(variant_dd_delta_values) if variant_dd_delta_values else None,
            "gate_failure_count": len(gate_failures),
        },
        "gates": {
            "suite_status": suite_status,
            "gate_failures": gate_failures,
            "measured_accuracy": "blocked_proxy_stack",
            "product_lut_ready": False,
        },
        "charts": {
            "case_charts": chart_cases,
            "material_sensitivity_tornado": {"type": "bar", "rows": tornado_rows},
            "cad_variant_deltas": {"type": "table", "rows": variant_delta_rows},
        },
        "decision_notes": [
            suite.get("decision_goal", ""),
            "Smoke/trend results are research evidence only until measured stack/material/device calibration passes.",
        ],
        "cases": case_results,
    }


FIELD_RESPONSE_EXPORT_COLUMNS = [
    "suite_id",
    "tier",
    "case_id",
    "label",
    "runner",
    "source_case",
    "wavelength_nm",
    "cra_x_deg",
    "cra_z_deg",
    "field_x_norm",
    "field_z_norm",
    "total_response",
    "edge_to_center",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "kpi_status",
    "grid_gate_pass",
    "negative_signed_flux_count",
    "product_lut_ready",
]
PDAF_EXPORT_COLUMNS = [
    "suite_id",
    "tier",
    "case_id",
    "label",
    "runner",
    "source_case",
    "wavelength_nm",
    "cra_x_deg",
    "cra_z_deg",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "split_phase_amplitude",
    "collection_mode",
    "kpi_status",
    "grid_gate_pass",
    "product_lut_ready",
]
CROSSTALK_SUMMARY_EXPORT_COLUMNS = [
    "suite_id",
    "tier",
    "case_id",
    "label",
    "runner",
    "source_case",
    "mode",
    "layout_label",
    "target_lens_id",
    "wavelength_nm",
    "resolution_px_per_um",
    "cra_x_deg",
    "cra_z_deg",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "grid_gate_pass",
    "kpi_status",
    "product_lut_ready",
]
CROSSTALK_CELL_EXPORT_COLUMNS = [
    "suite_id",
    "tier",
    "case_id",
    "label",
    "source_case",
    "region_id",
    "ocl_lens_id",
    "ocl_lens_kind",
    "output_dx",
    "output_dz",
    "output_dx_um",
    "output_dz_um",
    "response_fraction",
    "product_lut_ready",
]


def write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def camera_system_export_rows(suite_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    suite_id = str(suite_result.get("suite_id") or "")
    tier = str(suite_result.get("tier") or "")
    field_rows: list[dict[str, Any]] = []
    pdaf_rows: list[dict[str, Any]] = []
    crosstalk_rows: list[dict[str, Any]] = []
    crosstalk_cell_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for case in suite_result.get("cases", []):
        if not isinstance(case, dict):
            continue
        kpi = case.get("kpi") if isinstance(case.get("kpi"), dict) else {}
        charts = case.get("charts") if isinstance(case.get("charts"), dict) else {}
        numerical_gate = kpi.get("numerical_gate") if isinstance(kpi.get("numerical_gate"), dict) else {}
        grid_gate_pass = numerical_gate.get("passed")
        base = {
            "suite_id": suite_id,
            "tier": tier,
            "case_id": case.get("id"),
            "label": case.get("label"),
            "runner": case.get("runner"),
            "kpi_status": kpi.get("status"),
            "grid_gate_pass": grid_gate_pass,
            "negative_signed_flux_count": kpi.get("negative_signed_flux_count"),
            "product_lut_ready": False,
        }
        gate_rows.append(
            {
                **base,
                "case_status": case.get("status"),
                "gate_available": numerical_gate.get("available"),
                "grid_resolution_gate_fail_count": numerical_gate.get("grid_resolution_gate_fail_count"),
                "convergence_status": numerical_gate.get("convergence_status"),
                "measured_accuracy": suite_result.get("gates", {}).get("measured_accuracy")
                if isinstance(suite_result.get("gates"), dict)
                else None,
                "reason": "research_or_smoke_result_not_product_lut",
            }
        )
        cra_points = charts.get("cra_response_curve", {}).get("points") if isinstance(charts.get("cra_response_curve"), dict) else []
        if not cra_points and kpi.get("center_total_response") is not None:
            cra_points = [
                {
                    "case": "center",
                    "wavelength_nm": (kpi.get("wavelengths_nm") or [None])[0]
                    if isinstance(kpi.get("wavelengths_nm"), list)
                    else None,
                    "cra_x_deg": 0,
                    "cra_z_deg": 0,
                    "field_x_norm": 0,
                    "field_z_norm": 0,
                    "total_response": kpi.get("center_total_response"),
                    "edge_to_center": 1.0,
                    "split_phase_x_proxy": kpi.get("split_phase_x_max"),
                    "split_phase_z_proxy": kpi.get("split_phase_z_max"),
                }
            ]
        for point in cra_points or []:
            if not isinstance(point, dict):
                continue
            row = {
                **base,
                "source_case": point.get("case"),
                "wavelength_nm": point.get("wavelength_nm")
                or ((kpi.get("wavelengths_nm") or [None])[0] if isinstance(kpi.get("wavelengths_nm"), list) else None),
                "cra_x_deg": point.get("cra_x_deg"),
                "cra_z_deg": point.get("cra_z_deg"),
                "field_x_norm": point.get("field_x_norm"),
                "field_z_norm": point.get("field_z_norm"),
                "total_response": point.get("total_response"),
                "edge_to_center": point.get("edge_to_center"),
                "split_phase_x_proxy": point.get("split_phase_x_proxy"),
                "split_phase_z_proxy": point.get("split_phase_z_proxy"),
            }
            field_rows.append(row)
            split_x = json_safe_number(row.get("split_phase_x_proxy"))
            split_z = json_safe_number(row.get("split_phase_z_proxy"))
            if split_x is not None or split_z is not None or "split-pd" in (kpi.get("collection_modes") or []):
                pdaf_rows.append(
                    {
                        **base,
                        "source_case": row.get("source_case"),
                        "wavelength_nm": row.get("wavelength_nm"),
                        "cra_x_deg": row.get("cra_x_deg"),
                        "cra_z_deg": row.get("cra_z_deg"),
                        "split_phase_x_proxy": row.get("split_phase_x_proxy"),
                        "split_phase_z_proxy": row.get("split_phase_z_proxy"),
                        "split_phase_amplitude": math.hypot(split_x or 0.0, split_z or 0.0)
                        if split_x is not None or split_z is not None
                        else None,
                        "collection_mode": ",".join(kpi.get("collection_modes") or []),
                    }
                )
        heatmap = charts.get("crosstalk_kernel_heatmap") if isinstance(charts.get("crosstalk_kernel_heatmap"), dict) else {}
        for point in heatmap.get("points", []) if isinstance(heatmap.get("points"), list) else []:
            if not isinstance(point, dict):
                continue
            crosstalk_rows.append(
                {
                    **base,
                    "source_case": point.get("case"),
                    "mode": point.get("mode"),
                    "layout_label": point.get("layout_label"),
                    "target_lens_id": point.get("target_lens_id"),
                    "wavelength_nm": point.get("wavelength_nm"),
                    "resolution_px_per_um": point.get("resolution_px_per_um"),
                    "cra_x_deg": point.get("cra_x_deg"),
                    "cra_z_deg": point.get("cra_z_deg"),
                    "output_crosstalk_fraction": point.get("output_crosstalk_fraction"),
                    "strongest_neighbor_fraction": point.get("strongest_neighbor_fraction"),
                    "truncation_response_fraction": point.get("truncation_response_fraction"),
                    "grid_gate_pass": point.get("grid_resolution_gate_pass"),
                }
            )
        for cell in heatmap.get("cells", []) if isinstance(heatmap.get("cells"), list) else []:
            if not isinstance(cell, dict):
                continue
            crosstalk_cell_rows.append(
                {
                    "suite_id": suite_id,
                    "tier": tier,
                    "case_id": case.get("id"),
                    "label": case.get("label"),
                    "source_case": cell.get("case"),
                    "region_id": cell.get("region_id"),
                    "ocl_lens_id": cell.get("ocl_lens_id"),
                    "ocl_lens_kind": cell.get("ocl_lens_kind"),
                    "output_dx": cell.get("output_dx"),
                    "output_dz": cell.get("output_dz"),
                    "output_dx_um": cell.get("output_dx_um"),
                    "output_dz_um": cell.get("output_dz_um"),
                    "response_fraction": cell.get("response_fraction"),
                    "product_lut_ready": False,
                }
            )
    return {
        "field_response_rows": field_rows,
        "pdaf_rows": pdaf_rows,
        "crosstalk_summary_rows": crosstalk_rows,
        "crosstalk_cell_rows": crosstalk_cell_rows,
        "gate_rows": gate_rows,
    }


def write_camera_system_suite_export(suite_result: dict[str, Any], output_dir: Path, suite_result_path: Path | None = None) -> dict[str, str | None]:
    rows = camera_system_export_rows(suite_result)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_csv = output_dir / "camera_system_field_response.csv"
    pdaf_csv = output_dir / "camera_system_pdaf_response.csv"
    crosstalk_csv = output_dir / "camera_system_crosstalk_summary.csv"
    crosstalk_cell_csv = output_dir / "camera_system_crosstalk_cells.csv"
    gate_csv = output_dir / "camera_system_gate_report.csv"
    export_json = output_dir / "camera_system_suite_export.json"
    summary_json = output_dir / "workbench_camera_system_export_summary.json"
    write_dict_csv(field_csv, rows["field_response_rows"], FIELD_RESPONSE_EXPORT_COLUMNS)
    write_dict_csv(pdaf_csv, rows["pdaf_rows"], PDAF_EXPORT_COLUMNS)
    write_dict_csv(crosstalk_csv, rows["crosstalk_summary_rows"], CROSSTALK_SUMMARY_EXPORT_COLUMNS)
    write_dict_csv(crosstalk_cell_csv, rows["crosstalk_cell_rows"], CROSSTALK_CELL_EXPORT_COLUMNS)
    write_dict_csv(
        gate_csv,
        rows["gate_rows"],
        [
            "suite_id",
            "tier",
            "case_id",
            "label",
            "runner",
            "case_status",
            "kpi_status",
            "grid_gate_pass",
            "gate_available",
            "grid_resolution_gate_fail_count",
            "convergence_status",
            "negative_signed_flux_count",
            "measured_accuracy",
            "product_lut_ready",
            "reason",
        ],
    )
    gate_failures = suite_result.get("gates", {}).get("gate_failures", []) if isinstance(suite_result.get("gates"), dict) else []
    export_payload = {
        "schema": "camera_system_suite_export_v1",
        "source_suite_result": str(suite_result_path) if suite_result_path else None,
        "source_suite_result_url": rel_url(suite_result_path) if suite_result_path else None,
        "suite_id": suite_result.get("suite_id"),
        "suite_label": suite_result.get("suite_label"),
        "tier": suite_result.get("tier"),
        "status": "RESEARCH_ONLY",
        "product_lut_ready": False,
        "usage_scope": "camera_system_research_trend_not_product_accuracy",
        "row_counts": {key: len(value) for key, value in rows.items()},
        "gates": {
            "suite_status": suite_result.get("status"),
            "measured_accuracy": suite_result.get("gates", {}).get("measured_accuracy")
            if isinstance(suite_result.get("gates"), dict)
            else None,
            "gate_failure_count": len(gate_failures),
            "product_lut_ready": False,
        },
        "artifacts": {
            "field_response_csv": rel_url(field_csv),
            "pdaf_response_csv": rel_url(pdaf_csv),
            "crosstalk_summary_csv": rel_url(crosstalk_csv),
            "crosstalk_cells_csv": rel_url(crosstalk_cell_csv),
            "gate_report_csv": rel_url(gate_csv),
            "summary_json": rel_url(summary_json),
        },
        **rows,
        "notes": [
            "This package is for camera-system research/trend simulation only.",
            "It preserves suite gate state and never marks smoke/proxy data as product LUT ready.",
            "Measured stack/material/device calibration and quantitative convergence are required before product LUT use.",
        ],
    }
    write_json_artifact(export_json, export_payload)
    export_payload["artifacts"]["export_json"] = rel_url(export_json)
    write_json_artifact(export_json, export_payload)
    summary = {
        "schema": "workbench_camera_system_export_summary_v1",
        "suite_id": suite_result.get("suite_id"),
        "tier": suite_result.get("tier"),
        "status": export_payload["status"],
        "product_lut_ready": False,
        "row_counts": export_payload["row_counts"],
        "export_json_url": rel_url(export_json),
        "field_response_csv_url": rel_url(field_csv),
        "pdaf_response_csv_url": rel_url(pdaf_csv),
        "crosstalk_summary_csv_url": rel_url(crosstalk_csv),
        "crosstalk_cells_csv_url": rel_url(crosstalk_cell_csv),
        "gate_report_csv_url": rel_url(gate_csv),
    }
    write_json_artifact(summary_json, summary)
    validation_payload = validate_export_package(
        export_json,
        output_dir / "consumer_validation",
        field_x_values=[0.0, 0.5, 1.0],
        field_z_values=[0.0],
    )
    validation_outputs = {
        key: rel_url(Path(value))
        for key, value in (validation_payload.get("outputs") or {}).items()
        if value
    }
    return {
        "camera_system_export": rel_url(export_json),
        "camera_system_export_summary": rel_url(summary_json),
        "camera_system_field_response_csv": rel_url(field_csv),
        "camera_system_pdaf_response_csv": rel_url(pdaf_csv),
        "camera_system_crosstalk_summary_csv": rel_url(crosstalk_csv),
        "camera_system_crosstalk_cells_csv": rel_url(crosstalk_cell_csv),
        "camera_system_gate_report_csv": rel_url(gate_csv),
        "camera_system_validation": validation_outputs.get("validation_json"),
        "camera_system_validation_md": validation_outputs.get("validation_md"),
        "camera_system_field_query_csv": validation_outputs.get("field_query_csv"),
        "camera_system_crosstalk_index_csv": validation_outputs.get("crosstalk_index_csv"),
        "camera_system_gate_summary_csv": validation_outputs.get("gate_summary_csv"),
    }


def camera_system_export_from_request(body: dict[str, Any]) -> dict[str, Any]:
    suite_result_text = str(body.get("suite_result") or body.get("suite_result_json") or "")
    if not suite_result_text:
        raise ValueError("suite_result is required")
    suite_result_path = workspace_safe_path(suite_result_text)
    suite_result = read_json_artifact(suite_result_path)
    if suite_result.get("schema") != "pixel_workbench_suite_result_v1":
        raise ValueError(f"Unsupported suite result schema: {suite_result.get('schema')!r}")
    output_dir_text = str(body.get("output_dir") or "")
    output_dir = (
        workspace_safe_output_path(output_dir_text)
        if output_dir_text
        else suite_result_path.parent / "camera_system_export"
    )
    artifacts = write_camera_system_suite_export(suite_result, output_dir, suite_result_path)
    return {
        "schema": "pixel_workbench_camera_system_export_api_v1",
        "status": "RESEARCH_ONLY",
        "product_lut_ready": False,
        "suite_result": str(suite_result_path),
        "output_url": rel_url(output_dir),
        "artifacts": artifacts,
    }


def camera_system_validate_from_request(body: dict[str, Any]) -> dict[str, Any]:
    export_json_text = str(body.get("export_json") or body.get("camera_system_export") or "")
    if not export_json_text:
        raise ValueError("export_json is required")
    export_json_path = workspace_safe_path(export_json_text)
    output_dir_text = str(body.get("output_dir") or "")
    output_dir = (
        workspace_safe_output_path(output_dir_text)
        if output_dir_text
        else export_json_path.parent / "consumer_validation"
    )
    field_x_values = parse_export_float_list(str(body.get("field_x") or "0,0.5,1"))
    field_z_values = parse_export_float_list(str(body.get("field_z") or "0"))
    wavelength_text = str(body.get("wavelength_nm") or "all")
    wavelength_nm = None if wavelength_text.lower() == "all" else json_safe_number(wavelength_text)
    if wavelength_text.lower() != "all" and wavelength_nm is None:
        raise ValueError("wavelength_nm must be 'all' or a finite number")
    validation = validate_export_package(
        export_json_path,
        output_dir,
        field_x_values=field_x_values,
        field_z_values=field_z_values,
        wavelength_nm=wavelength_nm,
        require_field=bool(body.get("require_field", False)),
        require_pdaf=bool(body.get("require_pdaf", False)),
        require_crosstalk=bool(body.get("require_crosstalk", False)),
    )
    artifacts = {
        key: rel_url(Path(value))
        for key, value in (validation.get("outputs") or {}).items()
        if value
    }
    return {
        "schema": "pixel_workbench_camera_system_validation_api_v1",
        "status": validation.get("status"),
        "product_lut_ready": False,
        "export_json": str(export_json_path),
        "output_url": rel_url(output_dir),
        "validation": validation.get("validation"),
        "query": validation.get("query"),
        "artifacts": artifacts,
    }


def quantitative_evidence_from_request(body: dict[str, Any]) -> dict[str, Any]:
    config_text = str(body.get("config") or "configs/image_sensor_pixel_studio_reference.json")
    output_dir_text = str(body.get("output_dir") or "runs/camera_system_quantitative_evidence_reference")
    config_path = workspace_safe_path(config_text)
    output_dir = workspace_safe_output_path(output_dir_text)
    evidence = build_quantitative_evidence(config_path, output_dir)
    artifacts = {
        key: rel_url(Path(value))
        for key, value in (evidence.get("outputs") or {}).items()
        if value
    }
    return {
        "schema": "pixel_workbench_quantitative_evidence_api_v1",
        "status": evidence.get("status"),
        "framework_ready": evidence.get("framework_ready"),
        "research_lut_ready": evidence.get("research_lut_ready"),
        "quantitative_evidence_pass": evidence.get("quantitative_evidence_pass"),
        "accuracy_ready": evidence.get("accuracy_ready"),
        "product_lut_ready": evidence.get("product_lut_ready"),
        "evidence_count": evidence.get("evidence_count"),
        "blocker_count": evidence.get("blocker_count"),
        "product_blocker_count": evidence.get("product_blocker_count"),
        "framework_blocker_count": evidence.get("framework_blocker_count"),
        "evidence": evidence.get("evidence", []),
        "blockers": evidence.get("blockers", []),
        "artifacts": artifacts,
    }


def command_for_example(example: dict[str, Any], output_dir: Path) -> list[str]:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Meep Python environment not found: {PYTHON}")
    command = [
        str(PYTHON),
        "meep_supercell_lut.py",
        "--mode",
        str(example["mode"]),
        "--wavelengths-nm",
        str(example["wavelengths_nm"]),
        "--cases",
        str(example["cases"]),
        "--resolution",
        str(example["resolution"]),
        "--after-source-time",
        str(example["after_source_time"]),
        "--pml-um",
        str(example["pml_um"]),
        "--grid-snap-y",
        "nearest",
        "--output-dir",
        str(output_dir),
    ]
    if example.get("split_mode"):
        command.extend(["--split-mode", str(example["split_mode"])])
    return command


def update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)


def append_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        tail = JOBS[job_id].setdefault("log_tail", [])
        tail.append(line.rstrip())
        del tail[:-120]


def run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        example = dict(job["example"])
        simulation_request = copy.deepcopy(job.get("simulation_request"))
        solver_case = copy.deepcopy(job.get("solver_case") or example)
        output_dir = Path(job["output_dir"])
    log_path = output_dir / "job.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "simulation_request.json"
    solver_case_path = output_dir / "solver_case.json"
    kpi_path = output_dir / "kpi_summary.json"
    job_summary_path = output_dir / "workbench_job_summary.json"
    try:
        if simulation_request:
            write_json_artifact(request_path, simulation_request)
        write_json_artifact(solver_case_path, solver_case)
        command = command_for_lut_case(solver_case, output_dir)
        update_job(
            job_id,
            status="running",
            started_at=now_iso(),
            command=command,
            log_url=rel_url(log_path),
            request_url=rel_url(request_path) if simulation_request else None,
            solver_case_url=rel_url(solver_case_path),
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                append_log(job_id, line)
            return_code = process.wait()
        kpi = summarize_lut(output_dir)
        if solver_case.get("cad_template"):
            kpi["cad_template"] = solver_case["cad_template"]
        completed_at = now_iso()
        kpi.setdefault("artifacts", {})
        write_json_artifact(kpi_path, kpi)
        kpi["artifacts"]["kpi_summary"] = rel_url(kpi_path)
        write_json_artifact(kpi_path, kpi)
        job_summary = {
            "schema": "pixel_workbench_job_summary_v1",
            "job_id": job_id,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": completed_at,
            "request_url": rel_url(request_path) if simulation_request else None,
            "solver_case_url": rel_url(solver_case_path),
            "kpi_url": rel_url(kpi_path),
            "output_url": job.get("output_url"),
            "cad_template": solver_case.get("cad_template"),
            "kpi_status": kpi.get("status"),
            "numerical_gate": kpi.get("numerical_gate"),
        }
        write_json_artifact(job_summary_path, job_summary)
        update_job(
            job_id,
            status="completed" if return_code == 0 else "failed",
            return_code=return_code,
            completed_at=completed_at,
            kpi=kpi,
            log_url=rel_url(log_path),
            request_url=rel_url(request_path) if simulation_request else None,
            solver_case_url=rel_url(solver_case_path),
            kpi_url=rel_url(kpi_path),
            job_summary_url=rel_url(job_summary_path),
            error=None if return_code == 0 else f"solver exited with {return_code}",
        )
    except Exception as error:  # noqa: BLE001 - return the local job error to the UI.
        update_job(job_id, status="failed", completed_at=now_iso(), error=str(error), log_url=rel_url(log_path))


def run_command_for_case(job_id: str, command: list[str], case_dir: Path, case_label: str) -> int:
    log_path = case_dir / "job.log"
    case_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            append_log(job_id, f"[{case_label}] {line.rstrip()}")
        return process.wait()


def run_logged_case_step(job_id: str, log: Any, command: list[str], case_label: str, step_label: str) -> int:
    log.write(f"\n# {step_label}\n")
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    append_log(job_id, f"[{case_label}] {step_label}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        append_log(job_id, f"[{case_label}] {line.rstrip()}")
    return process.wait()


def run_gds_import_lut_case(job_id: str, case: dict[str, Any], case_dir: Path) -> int:
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "job.log"
    gds_path = (
        case_dir / str(case.get("gds_filename", "reference_pixel_masks.gds"))
        if case.get("write_reference_gds", False)
        else root_path_from_case(case.get("input_gds"), "input_gds")
    )
    geometry_json = case_dir / str(case.get("geometry_json_filename", "pixel_geometry_from_gds.json"))
    report_json = case_dir / str(case.get("gds_report_filename", "gds_import_report.json"))
    preview_svg = case_dir / str(case.get("gds_preview_filename", "gds_import_preview.svg"))
    with log_path.open("w", encoding="utf-8") as log:
        if case.get("write_reference_gds", False):
            return_code = run_logged_case_step(
                job_id,
                log,
                command_for_gds_write_reference(gds_path),
                case["label"],
                "write reference GDS",
            )
            if return_code != 0:
                return return_code
        return_code = run_logged_case_step(
            job_id,
            log,
            command_for_gds_convert(case, gds_path, geometry_json, report_json, preview_svg),
            case["label"],
            "convert GDS to pixel geometry JSON",
        )
        if return_code != 0:
            return return_code
        if case.get("generate_gmsh_mesh", False):
            mesh_dir = case_dir / str(case.get("gmsh_mesh_dir", "gmsh_mesh"))
            gmsh_bridge_report = case_dir / "gmsh_bridge_report.json"
            command, bridge = command_for_gmsh_mesh(case, report_json, mesh_dir)
            write_json_artifact(gmsh_bridge_report, bridge)
            return_code = run_logged_case_step(
                job_id,
                log,
                command,
                case["label"],
                "generate Gmsh TCAD bridge mesh",
            )
            bridge["status"] = "PASS" if return_code == 0 else "FAIL"
            bridge["return_code"] = return_code
            bridge["mesh_metadata"] = str(mesh_dir / "mesh_metadata.json")
            write_json_artifact(gmsh_bridge_report, bridge)
            if return_code != 0:
                return return_code
        solver_case = solver_case_for_gds_import(case, geometry_json)
        write_json_artifact(case_dir / "pipeline_solver_case.json", solver_case)
        return run_logged_case_step(
            job_id,
            log,
            command_for_lut_case(solver_case, case_dir),
            case["label"],
            "run Meep LUT with GDS-derived geometry",
        )


def suite_cases_for_tier(suite: dict[str, Any], tier: str, case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    selected = []
    case_id_set = set(case_ids or [])
    for case in suite.get("cases", []):
        if case_id_set and case["id"] not in case_id_set:
            continue
        case_tiers = case.get("tiers", [])
        tier_supported = tier in case_tiers
        smoke_can_scale = tier in {"trend", "quantitative"} and "smoke" in case_tiers
        if not (tier_supported or smoke_can_scale):
            continue
        tier_case = copy.deepcopy(case)
        overrides = (tier_case.pop("tier_overrides", {}) or {}).get(tier, {})
        if tier in {"trend", "quantitative"} and tier_case.get("runner") in {"example", "meep_lut", "cad_template_lut"}:
            if tier == "trend":
                overrides = {"resolution": 24, "after_source_time": 3, **overrides}
            if tier == "quantitative":
                overrides = {"resolution": 36, "after_source_time": 6, "pml_um": 0.60, **overrides}
        for key, value in overrides.items():
            if key == "stack_overrides":
                merged_overrides = dict(tier_case.get("stack_overrides") or {})
                merged_overrides.update(value)
                tier_case["stack_overrides"] = merged_overrides
            else:
                tier_case[key] = value
        tier_case["active_tier"] = tier
        selected.append(tier_case)
    return selected


def cad_template_variant_suite_from_catalog(catalog: dict[str, Any]) -> dict[str, Any] | None:
    templates = catalog.get("templates", [])
    if not isinstance(templates, list):
        return None
    by_id = {
        str(item.get("template_id")): item
        for item in templates
        if isinstance(item, dict) and item.get("template_id")
    }
    variants = [
        item
        for item in templates
        if isinstance(item, dict) and item.get("variant_of") and item.get("solver_ready")
    ]
    if not variants:
        return None
    cases: list[dict[str, Any]] = []
    included_base_ids: set[str] = set()
    for variant in sorted(variants, key=lambda item: str(item.get("template_id"))):
        variant_id = str(variant.get("template_id"))
        base_id = str(variant.get("variant_of"))
        base = by_id.get(base_id)
        if base and base.get("solver_ready") and base_id not in included_base_ids:
            cases.append(
                {
                    "id": f"cad_variant_base_{base_id}",
                    "label": f"Base: {base.get('label') or base_id}",
                    "runner": "cad_template_lut",
                    "tiers": ["smoke"],
                    "cad_template_id": base_id,
                    "wavelengths_nm": "550",
                    "color_channel": "green",
                    "cases_arg": "center:0:0:0:0:0:0",
                    "resolution": 8,
                    "after_source_time": 0.5,
                    "pml_um": 0.45,
                    "design_factors": {
                        "geometry_source": "CAD template geometry_import.json",
                        "template": base_id,
                        "role": "base",
                    },
                    "charts": ["subpixel_response_matrix", "convergence_report_card"],
                }
            )
            included_base_ids.add(base_id)
        overrides = variant.get("parameter_overrides", {})
        override_text = (
            ", ".join(f"{key}={value}" for key, value in overrides.items())
            if isinstance(overrides, dict) and overrides
            else "none"
        )
        cases.append(
            {
                "id": f"cad_variant_{variant_id}",
                "label": f"Variant: {variant.get('label') or variant_id}",
                "runner": "cad_template_lut",
                "tiers": ["smoke"],
                "cad_template_id": variant_id,
                "wavelengths_nm": "550",
                "color_channel": "green",
                "cases_arg": "center:0:0:0:0:0:0",
                "resolution": 8,
                "after_source_time": 0.5,
                "pml_um": 0.45,
                "design_factors": {
                    "geometry_source": "CAD template variant geometry_import.json",
                    "template": variant_id,
                    "variant_of": base_id,
                    "overrides": override_text,
                    "role": "variant",
                },
                "charts": ["subpixel_response_matrix", "convergence_report_card"],
                "decision_notes": [
                    "Compare against the matching base template before making design decisions.",
                    "Smoke deltas are only wiring/trend evidence because resolution and measured-data gates are not quantitative.",
                ],
            }
        )
    return {
        "id": "cad_template_variant_comparison",
        "label": "CAD Variant Comparison",
        "category": "CAD Geometry",
        "priority": 2.6,
        "runtime_hint": "base + selected variants, 20-60 sec each",
        "decision_goal": "Run FreeCAD-openable CAD variants against their base templates and compare smoke KPI deltas with explicit override provenance.",
        "recommended_tier": "smoke",
        "tiers": ["smoke"],
        "cases": cases,
    }


def test_suite_catalog() -> dict[str, dict[str, Any]]:
    suites = copy.deepcopy(TEST_SUITES)
    try:
        variant_suite = cad_template_variant_suite_from_catalog(load_cad_template_catalog())
    except Exception:  # noqa: BLE001 - keep static suites available if local CAD catalog is broken.
        variant_suite = None
    if variant_suite:
        suites[variant_suite["id"]] = variant_suite
    return suites


def test_suite_by_id(suite_id: str) -> dict[str, Any]:
    suites = test_suite_catalog()
    if suite_id not in suites:
        raise KeyError(f"Unknown simulation suite: {suite_id}")
    return suites[suite_id]


def persist_suite_case_result(case_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    case_result_path = case_dir / "case_result.json"
    case_input_path = case_dir / "case_input.json"
    case_command_path = case_dir / "case_command.json"
    solver_case_path = case_dir / "solver_case.json"
    pipeline_solver_case_path = case_dir / "pipeline_solver_case.json"
    log_path = case_dir / "job.log"
    result.setdefault("schema", "pixel_workbench_suite_case_result_v1")
    result.setdefault("artifacts", {})
    result["artifacts"].update(
        {
            "case_input": rel_url(case_input_path),
            "case_command": rel_url(case_command_path),
            "solver_case": rel_url(solver_case_path) or rel_url(pipeline_solver_case_path),
            "job_log": rel_url(log_path),
        }
    )
    write_json_artifact(case_result_path, result)
    result["artifacts"]["case_result"] = rel_url(case_result_path)
    if not result.get("output_url"):
        result["output_url"] = rel_url(case_dir)
    write_json_artifact(case_result_path, result)
    return result


def run_suite_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        suite = copy.deepcopy(job["suite"])
        tier = str(job["tier"])
        case_ids = list(job.get("case_ids") or [])
        output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_log_path = output_dir / "suite.log"
    suite_result_path = output_dir / "suite_result.json"
    suite_summary_path = output_dir / "workbench_suite_summary.json"
    try:
        cases = suite_cases_for_tier(suite, tier, case_ids)
        update_job(
            job_id,
            status="running",
            started_at=now_iso(),
            log_url=rel_url(suite_log_path),
            progress={"completed": 0, "total": len(cases), "current_case": None},
        )
        case_results: list[dict[str, Any]] = []
        with suite_log_path.open("w", encoding="utf-8") as suite_log:
            suite_log.write(f"Suite {suite['id']} tier={tier} cases={len(cases)}\n")
            suite_log.flush()
            for index, case in enumerate(cases, start=1):
                update_job(
                    job_id,
                    progress={"completed": index - 1, "total": len(cases), "current_case": case["id"]},
                )
                append_log(job_id, f"Starting suite case {index}/{len(cases)}: {case['label']}")
                case_dir = output_dir / case["id"]
                if case.get("runner") == "gds_import_lut":
                    write_suite_case_provenance(case, case_dir)
                    return_code = run_gds_import_lut_case(job_id, case, case_dir)
                    write_suite_case_provenance(case, case_dir)
                    suite_log.write(f"{case['id']} return_code={return_code}\n")
                    suite_log.flush()
                    case_results.append(persist_suite_case_result(case_dir, case_result_from_output(case, case_dir, return_code)))
                    continue
                command = command_for_case(case, case_dir)
                if command:
                    write_suite_case_provenance(case, case_dir, command)
                    return_code = run_command_for_case(job_id, command, case_dir, case["label"])
                    suite_log.write(f"{case['id']} return_code={return_code}\n")
                    suite_log.flush()
                    case_results.append(persist_suite_case_result(case_dir, case_result_from_output(case, case_dir, return_code)))
                else:
                    write_suite_case_provenance(case, case_dir)
                    result = case_result_from_output(case, case_dir, 0)
                    suite_log.write(f"{case['id']} analysis_only status={result['kpi'].get('status')}\n")
                    suite_log.flush()
                    append_log(job_id, f"[{case['label']}] analysis artifact generated without solver execution")
                    case_results.append(persist_suite_case_result(case_dir, result))
        suite_result = aggregate_suite_result(suite, tier, case_results)
        write_json_artifact(suite_result_path, suite_result)
        suite_result.setdefault("artifacts", {})
        suite_result["artifacts"]["suite_result"] = rel_url(suite_result_path)
        suite_result["artifacts"]["suite_summary"] = rel_url(suite_summary_path)
        camera_export_artifacts = write_camera_system_suite_export(
            suite_result,
            output_dir / "camera_system_export",
            suite_result_path,
        )
        suite_result["artifacts"].update(camera_export_artifacts)
        write_json_artifact(suite_result_path, suite_result)
        completed_at = now_iso()
        suite_summary = {
            "schema": "pixel_workbench_suite_summary_v1",
            "job_id": job_id,
            "suite_id": suite.get("id"),
            "suite_label": suite.get("label"),
            "tier": tier,
            "status": suite_result.get("status"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": completed_at,
            "case_count": suite_result.get("case_count"),
            "gate_failure_count": suite_result.get("kpi_summary", {}).get("gate_failure_count"),
            "product_lut_ready": suite_result.get("product_lut_ready"),
            "suite_result_url": rel_url(suite_result_path),
            "output_url": job.get("output_url"),
            "log_url": rel_url(suite_log_path),
            "camera_system_export_url": camera_export_artifacts.get("camera_system_export"),
            "camera_system_field_response_csv_url": camera_export_artifacts.get("camera_system_field_response_csv"),
            "camera_system_pdaf_response_csv_url": camera_export_artifacts.get("camera_system_pdaf_response_csv"),
            "camera_system_crosstalk_summary_csv_url": camera_export_artifacts.get("camera_system_crosstalk_summary_csv"),
            "camera_system_gate_report_csv_url": camera_export_artifacts.get("camera_system_gate_report_csv"),
            "camera_system_validation_url": camera_export_artifacts.get("camera_system_validation"),
            "camera_system_validation_md_url": camera_export_artifacts.get("camera_system_validation_md"),
            "camera_system_field_query_csv_url": camera_export_artifacts.get("camera_system_field_query_csv"),
            "camera_system_crosstalk_index_csv_url": camera_export_artifacts.get("camera_system_crosstalk_index_csv"),
            "camera_system_gate_summary_csv_url": camera_export_artifacts.get("camera_system_gate_summary_csv"),
            "case_artifacts": [
                {
                    "case_id": item.get("id"),
                    "label": item.get("label"),
                    "status": item.get("status"),
                    "kpi_status": (item.get("kpi") or {}).get("status") if isinstance(item.get("kpi"), dict) else None,
                    "output_url": item.get("output_url"),
                    "case_result_url": (item.get("artifacts") or {}).get("case_result")
                    if isinstance(item.get("artifacts"), dict)
                    else None,
                    "case_input_url": (item.get("artifacts") or {}).get("case_input")
                    if isinstance(item.get("artifacts"), dict)
                    else None,
                    "case_command_url": (item.get("artifacts") or {}).get("case_command")
                    if isinstance(item.get("artifacts"), dict)
                    else None,
                    "solver_case_url": (item.get("artifacts") or {}).get("solver_case")
                    if isinstance(item.get("artifacts"), dict)
                    else None,
                }
                for item in suite_result.get("cases", [])
            ],
        }
        write_json_artifact(suite_summary_path, suite_summary)
        suite_result["artifacts"]["suite_summary"] = rel_url(suite_summary_path)
        write_json_artifact(suite_result_path, suite_result)
        update_job(
            job_id,
            status="completed",
            return_code=0,
            completed_at=completed_at,
            progress={"completed": len(cases), "total": len(cases), "current_case": None},
            suite_result=suite_result,
            suite_result_url=rel_url(suite_result_path),
            suite_summary_url=rel_url(suite_summary_path),
            log_url=rel_url(suite_log_path),
            error=None,
        )
    except Exception as error:  # noqa: BLE001 - local job diagnostics.
        update_job(job_id, status="failed", completed_at=now_iso(), error=str(error), log_url=rel_url(suite_log_path))


def create_job(example_id: str) -> dict[str, Any]:
    if example_id not in EXAMPLES:
        raise KeyError(f"Unknown simulation example: {example_id}")
    example = EXAMPLES[example_id]
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_dir = RUN_ROOT / f"{example_id}_{job_id}"
    job = {
        "schema": "pixel_workbench_solver_job_v1",
        "id": job_id,
        "example": example,
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "output_dir": str(output_dir),
        "output_url": "/" + output_dir.relative_to(ROOT).as_posix(),
        "command": None,
        "log_tail": [],
        "log_url": None,
        "request_url": None,
        "solver_case": example,
        "solver_case_url": None,
        "kpi": None,
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()
    return snapshot_job(job_id)


def create_request_job(simulation_request: dict[str, Any]) -> dict[str, Any]:
    solver_case = solver_case_from_request(simulation_request)
    job_id = f"request_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    preset_hint = str(solver_case.get("preset_hint") or "active_design").replace("/", "_")
    output_dir = RUN_ROOT / f"{preset_hint}_{job_id}"
    job = {
        "schema": "pixel_workbench_solver_job_v1",
        "kind": "request",
        "id": job_id,
        "example": solver_case,
        "simulation_request": simulation_request,
        "solver_case": solver_case,
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "output_dir": str(output_dir),
        "output_url": "/" + output_dir.relative_to(ROOT).as_posix(),
        "command": None,
        "log_tail": [],
        "log_url": None,
        "request_url": None,
        "solver_case_url": None,
        "kpi": None,
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()
    return snapshot_job(job_id)


def create_suite_job(suite_id: str, tier: str = "smoke", case_ids: list[str] | None = None) -> dict[str, Any]:
    suite = test_suite_by_id(suite_id)
    if tier not in suite.get("tiers", []):
        raise KeyError(f"Unsupported tier {tier!r} for suite {suite_id!r}")
    selected_cases = suite_cases_for_tier(suite, tier, case_ids)
    if not selected_cases:
        raise KeyError(f"No runnable cases for suite={suite_id!r}, tier={tier!r}")
    job_id = f"suite_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_dir = RUN_ROOT / f"{suite_id}_{tier}_{job_id}"
    job = {
        "schema": "pixel_workbench_solver_job_v1",
        "kind": "suite",
        "id": job_id,
        "suite": suite,
        "tier": tier,
        "case_ids": case_ids or [],
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "output_dir": str(output_dir),
        "output_url": "/" + output_dir.relative_to(ROOT).as_posix(),
        "command": None,
        "log_tail": [],
        "log_url": None,
        "kpi": None,
        "suite_result": None,
        "suite_result_url": None,
        "suite_summary_url": None,
        "error": None,
        "progress": {"completed": 0, "total": len(selected_cases), "current_case": None},
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=run_suite_job, args=(job_id,), daemon=True)
    thread.start()
    return snapshot_job(job_id)


def snapshot_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        return json.loads(json.dumps(JOBS[job_id]))


def list_jobs() -> list[dict[str, Any]]:
    with JOBS_LOCK:
        return [json.loads(json.dumps(job)) for job in sorted(JOBS.values(), key=lambda item: item["created_at"], reverse=True)]


def path_from_record(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    candidate = Path(str(value))
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def file_artifact_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "url": None, "exists": False, "size_bytes": None}
    exists = path.exists()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative = None
    return {
        "path": str(path),
        "relative_path": relative,
        "url": rel_url(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def cad_template_readiness(template: dict[str, Any] | None) -> dict[str, Any]:
    if not template:
        return {
            "status": "MISSING",
            "missing_artifacts": ["template"],
            "failed_checks": ["template_missing"],
        }
    artifacts = template.get("artifacts", {}) if isinstance(template.get("artifacts"), dict) else {}
    required_artifacts = ("step", "brep", "fcstd", "geometry_import", "parameters", "assumption_ledger")
    missing_artifacts = [
        name
        for name in required_artifacts
        if not (isinstance(artifacts.get(name), dict) and artifacts[name].get("exists"))
    ]
    freecad_validation = (
        template.get("freecad_validation", {})
        if isinstance(template.get("freecad_validation"), dict)
        else {}
    )
    design_rule_validation = (
        template.get("design_rule_validation", {})
        if isinstance(template.get("design_rule_validation"), dict)
        else {}
    )
    assumption_ledger = (
        template.get("assumption_ledger", {})
        if isinstance(template.get("assumption_ledger"), dict)
        else {}
    )
    checks = {
        "solver_ready": template.get("solver_ready") is True,
        "freecad_validation_pass": freecad_validation.get("status") == "PASS",
        "design_rules_pass": design_rule_validation.get("status") == "PASS",
        "assumption_ledger_linked": assumption_ledger.get("available") is True,
        "product_accuracy_blocked": assumption_ledger.get("product_accuracy_ready") is False,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    status = "PASS" if not missing_artifacts and not failed_checks else "CHECK"
    return {
        "status": status,
        "checks": checks,
        "missing_artifacts": missing_artifacts,
        "failed_checks": failed_checks,
    }


def cad_starter_template_set_summary(templates: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {
        str(template.get("template_id")): template
        for template in templates
        if isinstance(template, dict) and template.get("template_id")
    }
    rows = []
    for starter in CAD_STARTER_TEMPLATE_SET:
        template_id = starter["template_id"]
        template = by_id.get(template_id)
        readiness = cad_template_readiness(template)
        rows.append(
            {
                **starter,
                "exists": template is not None,
                "variant_of": template.get("variant_of") if template else None,
                "readiness": readiness,
            }
        )
    pass_count = sum(1 for row in rows if row["readiness"]["status"] == "PASS")
    return {
        "schema": "pixel_cad_starter_template_set_v1",
        "status": "PASS" if pass_count == len(rows) else "CHECK",
        "template_count": len(rows),
        "pass_count": pass_count,
        "missing_template_ids": [row["template_id"] for row in rows if not row["exists"]],
        "check_template_ids": [
            row["template_id"]
            for row in rows
            if row["exists"] and row["readiness"]["status"] != "PASS"
        ],
        "templates": rows,
        "notes": [
            "Starter templates are the default FreeCAD review set for repeated pixel design work.",
            "PASS means the parametric CAD source is complete enough for research/trend solver runs, not product accuracy.",
            "Product accuracy still requires measured geometry/material/device calibration and quantitative convergence pass.",
        ],
    }


def qpd_template_comparison(templates: list[dict[str, Any]]) -> dict[str, Any]:
    def design_parameters(template: dict[str, Any]) -> dict[str, Any]:
        parameters = template.get("parameters", {}) if isinstance(template.get("parameters"), dict) else {}
        topology = template.get("topology", {}) if isinstance(template.get("topology"), dict) else {}
        return {**topology, **parameters}

    qpd_templates = [
        template
        for template in templates
        if str(design_parameters(template).get("split_mode") or "").lower() == "quad"
        or "qpd" in str(template.get("template_id") or "").lower()
        or "qpd" in str(template.get("label") or "").lower()
    ]
    baseline = next(
        (template for template in qpd_templates if template.get("template_id") == "qpd_split_pd_2x2"),
        qpd_templates[0] if qpd_templates else None,
    )

    def gw_center_metrics(template: dict[str, Any]) -> dict[str, Any]:
        qpd_gw = template.get("tcad_bridge", {}).get("qpd_gw_3d", {})
        summary = qpd_gw.get("summary", {}) if isinstance(qpd_gw.get("summary"), dict) else {}
        cases = summary.get("cases", []) if isinstance(summary.get("cases"), list) else []
        if cases and isinstance(cases[0], dict) and isinstance(cases[0].get("metrics"), dict):
            return cases[0]["metrics"]
        return {}

    def gw_field_summary(template: dict[str, Any]) -> dict[str, Any]:
        qpd_gw = template.get("tcad_bridge", {}).get("qpd_gw_3d", {})
        summary = qpd_gw.get("summary", {}) if isinstance(qpd_gw.get("summary"), dict) else {}
        field = summary.get("field_response_summary")
        return field if isinstance(field, dict) else {}

    def delta(value: Any, base: Any) -> float | None:
        value_number = json_safe_number(value)
        base_number = json_safe_number(base)
        if value_number is None or base_number is None:
            return None
        return value_number - base_number

    def pct_delta(value: Any, base: Any) -> float | None:
        value_number = json_safe_number(value)
        base_number = json_safe_number(base)
        if value_number is None or base_number in {None, 0.0}:
            return None
        return ((value_number - base_number) / base_number) * 100.0

    base_parameters = design_parameters(baseline) if baseline else {}
    base_metrics = gw_center_metrics(baseline) if baseline else {}
    rows: list[dict[str, Any]] = []
    for template in qpd_templates:
        parameters = design_parameters(template)
        bridge = template.get("tcad_bridge", {}) if isinstance(template.get("tcad_bridge"), dict) else {}
        dd = bridge.get("devsim_dd_smoke", {}) if isinstance(bridge.get("devsim_dd_smoke"), dict) else {}
        weighting = bridge.get("qpd_weighting_3d", {}) if isinstance(bridge.get("qpd_weighting_3d"), dict) else {}
        qpd_gw = bridge.get("qpd_gw_3d", {}) if isinstance(bridge.get("qpd_gw_3d"), dict) else {}
        metrics = gw_center_metrics(template)
        field = gw_field_summary(template)
        case_count = int(qpd_gw.get("case_count") or 0)
        generation_gate = qpd_gw.get("generation_volume_gate")
        gw_gate = "PASS" if qpd_gw.get("full_q1q4_gw_gate") == "PASS" and generation_gate == "PASS" else "CHECK"
        field_curve_gate = "PASS" if case_count >= 4 and json_safe_number(field.get("phase_x_slope_per_deg_max_abs")) is not None else "CHECK"
        metadata_gate = "CHECK" if bridge.get("electrical_capability", {}).get("inferred_from_template_parameters") else "PASS"
        row = {
            "template_id": template.get("template_id"),
            "label": template.get("label"),
            "variant_of": template.get("variant_of"),
            "split_mode": parameters.get("split_mode"),
            "shield_mode": parameters.get("shield_mode"),
            "lens_height_um": json_safe_number(parameters.get("lens_height_um")),
            "lens_height_delta_pct_from_base": pct_delta(parameters.get("lens_height_um"), base_parameters.get("lens_height_um")),
            "dti_width_um": json_safe_number(parameters.get("dti_width_um")),
            "dti_width_delta_nm_from_base": (
                delta(parameters.get("dti_width_um"), base_parameters.get("dti_width_um")) * 1000.0
                if delta(parameters.get("dti_width_um"), base_parameters.get("dti_width_um")) is not None
                else None
            ),
            "dd_phase_metric_applicable": dd.get("phase_metric_applicable"),
            "dd_phase_proxy_x": dd.get("photo_split_phase_x_proxy"),
            "dd_solver_gate": dd.get("solver_gate"),
            "metadata_gate": metadata_gate,
            "weighting_gate": weighting.get("full_q1q4_weighting_gate"),
            "qpd_weighting_phase_x": weighting.get("phase_x_weighting"),
            "qpd_weighting_phase_z": weighting.get("phase_z_weighting"),
            "quadrant_uniformity_weighting": weighting.get("quadrant_uniformity"),
            "gw_gate": gw_gate,
            "qpd_gw_phase_x": metrics.get("phase_x_gw"),
            "qpd_gw_phase_z": metrics.get("phase_z_gw"),
            "quadrant_uniformity_gw": metrics.get("quadrant_uniformity_gw"),
            "generation_weighted_qsum_fraction": metrics.get("generation_weighted_qsum_fraction"),
            "phase_x_delta_from_base": delta(metrics.get("phase_x_gw"), base_metrics.get("phase_x_gw")),
            "phase_z_delta_from_base": delta(metrics.get("phase_z_gw"), base_metrics.get("phase_z_gw")),
            "uniformity_delta_from_base": delta(
                metrics.get("quadrant_uniformity_gw"),
                base_metrics.get("quadrant_uniformity_gw"),
            ),
            "qsum_delta_pct_from_base": pct_delta(
                metrics.get("generation_weighted_qsum_fraction"),
                base_metrics.get("generation_weighted_qsum_fraction"),
            ),
            "field_curve_gate": field_curve_gate,
            "edge_to_center_response_ratio_min": field.get("edge_to_center_response_ratio_min"),
            "edge_to_center_response_ratio_max": field.get("edge_to_center_response_ratio_max"),
            "phase_x_slope_per_deg_max_abs": field.get("phase_x_slope_per_deg_max_abs"),
            "generation_volume_gate": generation_gate,
            "full_q1q4_dd_gate": qpd_gw.get("full_q1q4_dd_gate") or weighting.get("full_q1q4_dd_gate"),
            "case_count": case_count,
            "decision_flags": [
                flag
                for flag, active in {
                    "needs_full_3d_dd": (qpd_gw.get("full_q1q4_dd_gate") or weighting.get("full_q1q4_dd_gate")) == "CHECK",
                    "field_curve_smoke_only": field_curve_gate != "PASS",
                    "metadata_inferred_from_template": metadata_gate == "CHECK",
                    "generation_volume_check": generation_gate != "PASS",
                }.items()
                if active
            ],
        }
        rows.append(row)

    rows.sort(key=lambda row: (row.get("variant_of") is not None, str(row.get("template_id") or "")))
    check_rows = [
        row["template_id"]
        for row in rows
        if row.get("gw_gate") != "PASS"
        or row.get("field_curve_gate") != "PASS"
        or row.get("metadata_gate") != "PASS"
    ]
    return {
        "schema": "pixel_workbench_qpd_template_comparison_v1",
        "status": "PASS" if rows and not check_rows else "CHECK",
        "baseline_template_id": baseline.get("template_id") if baseline else None,
        "row_count": len(rows),
        "check_template_ids": check_rows,
        "rows": rows,
        "notes": [
            "Compares QPD-oriented CAD templates using 3D FDTD generation times 3D weighting-potential surrogate metrics.",
            "Full Q1-Q4 drift-diffusion remains CHECK until a native calibrated 3D DD device solve exists.",
            "Rows with field_curve_smoke_only have center-only or limited CRA coverage and should not drive edge-field decisions.",
        ],
    }


def cad_solver_role_matrix(templates: list[dict[str, Any]], qpd_comparison: dict[str, Any]) -> dict[str, Any]:
    def artifact_count(name: str) -> int:
        return sum(
            1
            for template in templates
            if template.get("artifacts", {}).get(name, {}).get("exists")
        )

    fdtd_geometry_count = artifact_count("geometry_import")
    fdtd_generation_volume_count = artifact_count("fdtd_generation_volume_3d")
    qpd_gw_count = sum(
        1
        for template in templates
        if template.get("tcad_bridge", {}).get("qpd_gw_3d", {}).get("available")
    )
    devsim_dd_count = sum(
        1
        for template in templates
        if template.get("tcad_bridge", {}).get("devsim_dd_smoke", {}).get("available")
    )
    split_proxy_count = sum(
        1
        for template in templates
        if template.get("tcad_bridge", {}).get("devsim_dd_smoke", {}).get("phase_metric_applicable") is True
    )
    rows = [
        {
            "id": "fdtd_optical",
            "label": "FDTD Optical",
            "role": "primary trend solver",
            "availability": "available" if fdtd_geometry_count else "missing",
            "evidence_count": fdtd_generation_volume_count,
            "coverage": "ML/OCL, CFA, CRA, optical crosstalk, Si generation map/volume",
            "current_use": "Use first for optical stack, OCL/CFA, CRA roll-off, and crosstalk design comparisons.",
            "not_for": "carrier collection, TG/FD behavior, lag/noise/readout circuit behavior",
            "product_gate": "CHECK",
            "next_accuracy_requirement": "measured stack geometry, measured n,k, convergence pass",
        },
        {
            "id": "qpd_gw",
            "label": "3D G*W",
            "role": "practical collection surrogate",
            "availability": "available" if qpd_gw_count else "not run",
            "evidence_count": qpd_gw_count,
            "coverage": "FDTD generation volume multiplied by 3D weighting-potential response",
            "current_use": "Use for QPD/PDAF balance screening and variant ranking before calibrated device DD exists.",
            "not_for": "full Q1-Q4 drift-diffusion, charge sharing with implant/trap calibration",
            "product_gate": "CHECK" if qpd_comparison.get("status") != "PASS" else "TREND_PASS",
            "next_accuracy_requirement": "native 3D DD or measured QPD balance calibration",
        },
        {
            "id": "devsim_dd",
            "label": "DEVSIM DD",
            "role": "electrical diagnostic",
            "availability": "available" if devsim_dd_count else "not run",
            "evidence_count": devsim_dd_count,
            "coverage": "2D mesh import, contacts, diode solve, split-current phase-proxy where split mode exists",
            "current_use": "Use to catch broken electrical meshes/contacts and sanity-check split proxy direction.",
            "not_for": "product QE, LUT, true pinned-PD/TG/FD charge transfer, calibrated crosstalk",
            "product_gate": "CHECK",
            "next_accuracy_requirement": "measured implants, TG/FD geometry, traps, mobility/recombination calibration",
            "split_proxy_count": split_proxy_count,
        },
        {
            "id": "circuit_readout",
            "label": "Circuit / Readout",
            "role": "not implemented",
            "availability": "missing",
            "evidence_count": 0,
            "coverage": "conversion gain, full well, lag, noise, source follower, column/ADC, timing",
            "current_use": "Keep outside the current optical/device screening loop.",
            "not_for": "any current KPI produced by this workbench",
            "product_gate": "MISSING",
            "next_accuracy_requirement": "separate compact/device/circuit model and measured readout calibration",
        },
    ]
    return {
        "schema": "pixel_workbench_solver_role_matrix_v1",
        "status": "CHECK",
        "summary": "Current practical decisions should be FDTD/G*W-led; DEVSIM DD is a diagnostic proxy until measured device calibration exists.",
        "primary_decision_path": ["FDTD Optical", "3D G*W"],
        "diagnostic_path": ["DEVSIM DD"],
        "missing_accuracy_track": ["Circuit / Readout", "native calibrated 3D drift-diffusion"],
        "rows": rows,
        "counts": {
            "fdtd_geometry_input_templates": fdtd_geometry_count,
            "fdtd_generation_volume_templates": fdtd_generation_volume_count,
            "qpd_gw_templates": qpd_gw_count,
            "devsim_dd_templates": devsim_dd_count,
            "split_proxy_templates": split_proxy_count,
        },
    }


def cad_template_dimension_summary(parameters: dict[str, Any], freecad_validation: dict[str, Any]) -> dict[str, Any]:
    pitch_um = json_safe_number(parameters.get("pitch_um"))
    nx_number = json_safe_number(parameters.get("nx"))
    nz_number = json_safe_number(parameters.get("nz"))
    nx = int(nx_number) if nx_number is not None else None
    nz = int(nz_number) if nz_number is not None else None
    footprint_x_um = pitch_um * nx if pitch_um is not None and nx is not None else None
    footprint_z_um = pitch_um * nz if pitch_um is not None and nz is not None else None
    bbox = freecad_validation.get("bbox_um") if isinstance(freecad_validation.get("bbox_um"), dict) else {}
    if not bbox and isinstance(freecad_validation.get("step"), dict):
        bbox = (
            freecad_validation.get("step", {}).get("bbox_um")
            if isinstance(freecad_validation.get("step", {}).get("bbox_um"), dict)
            else {}
        )
    raw_ocl_blocks = parameters.get("ocl_blocks") if isinstance(parameters.get("ocl_blocks"), list) else []
    ocl_group_sizes: list[tuple[int, int]] = []
    for block in raw_ocl_blocks:
        if not isinstance(block, dict):
            continue
        sx_number = json_safe_number(block.get("sx"))
        sz_number = json_safe_number(block.get("sz"))
        if sx_number is None or sz_number is None:
            continue
        ocl_group_sizes.append((max(int(sx_number), 1), max(int(sz_number), 1)))
    unique_ocl_group_sizes = sorted(set(ocl_group_sizes))
    max_ocl_sx = max((sx for sx, _ in ocl_group_sizes), default=1)
    max_ocl_sz = max((sz for _, sz in ocl_group_sizes), default=1)
    ocl_pitch_x_um = pitch_um * max_ocl_sx if pitch_um is not None else None
    ocl_pitch_z_um = pitch_um * max_ocl_sz if pitch_um is not None else None
    uniform_ocl_groups = len(unique_ocl_group_sizes) == 1
    ocl_group_count_x = nx // max_ocl_sx if nx is not None and max_ocl_sx and nx % max_ocl_sx == 0 else None
    ocl_group_count_z = nz // max_ocl_sz if nz is not None and max_ocl_sz and nz % max_ocl_sz == 0 else None
    min_ocl_group_count = (
        min(ocl_group_count_x, ocl_group_count_z)
        if uniform_ocl_groups and ocl_group_count_x is not None and ocl_group_count_z is not None
        else None
    )
    if min_ocl_group_count is None:
        crosstalk_kernel_status = "CHECK"
        crosstalk_kernel_label = "mixed/irregular OCL coverage"
        crosstalk_kernel_note = "Mixed or irregular OCL groups need an explicit crosstalk simulation domain definition."
    elif min_ocl_group_count >= 5:
        crosstalk_kernel_status = "PASS"
        crosstalk_kernel_label = f"{ocl_group_count_x}x{ocl_group_count_z} OCL groups · practical kernel"
        crosstalk_kernel_note = "At least 5x5 OCL groups are available, which is practical for checking longer-range leakage and kernel truncation."
    elif min_ocl_group_count >= 3:
        crosstalk_kernel_status = "CHECK"
        crosstalk_kernel_label = f"{ocl_group_count_x}x{ocl_group_count_z} OCL groups · minimum kernel"
        crosstalk_kernel_note = "3x3 OCL groups can estimate central-to-8-neighbor crosstalk, but larger domains are recommended for CRA and long-range leakage."
    else:
        crosstalk_kernel_status = "FAIL"
        crosstalk_kernel_label = f"{ocl_group_count_x}x{ocl_group_count_z} OCL groups · insufficient"
        crosstalk_kernel_note = "Crosstalk kernels need at least a central OCL group plus its 8 neighboring OCL groups; use a 3x3 OCL-group neighborhood or larger."
    if len(unique_ocl_group_sizes) > 1:
        ocl_group_label = (
            f"mixed OCL groups · max {max_ocl_sx}x{max_ocl_sz} · "
            f"{ocl_pitch_x_um:.3f} x {ocl_pitch_z_um:.3f} um"
            if ocl_pitch_x_um is not None and ocl_pitch_z_um is not None
            else "mixed OCL groups"
        )
    elif ocl_pitch_x_um is not None and ocl_pitch_z_um is not None:
        ocl_group_label = f"{max_ocl_sx}x{max_ocl_sz} group · {ocl_pitch_x_um:.3f} x {ocl_pitch_z_um:.3f} um"
    else:
        ocl_group_label = f"{max_ocl_sx}x{max_ocl_sz} group"
    topology_fields = ["nx", "nz", "ocl_blocks", "cfa_pattern", "split_mode", "shield_mode"]
    return {
        "schema": "pixel_workbench_cad_template_dimension_summary_v1",
        "pixel_pitch_um": pitch_um,
        "pixel_area_um2": pitch_um * pitch_um if pitch_um is not None else None,
        "array_nx": nx,
        "array_nz": nz,
        "pixel_count": nx * nz if nx is not None and nz is not None else None,
        "footprint_x_um": footprint_x_um,
        "footprint_z_um": footprint_z_um,
        "freecad_bbox_x_um": json_safe_number(bbox.get("xlen")),
        "freecad_bbox_z_um": json_safe_number(bbox.get("zlen")),
        "ocl_group_sizes": [f"{sx}x{sz}" for sx, sz in unique_ocl_group_sizes],
        "max_ocl_group_sx": max_ocl_sx,
        "max_ocl_group_sz": max_ocl_sz,
        "ocl_group_count_x": ocl_group_count_x,
        "ocl_group_count_z": ocl_group_count_z,
        "effective_ocl_pitch_x_um": ocl_pitch_x_um,
        "effective_ocl_pitch_z_um": ocl_pitch_z_um,
        "effective_ocl_group_pitch_label": ocl_group_label,
        "effective_binning_note": (
            "This is the optical/OCL group pitch. It becomes the effective binned output pitch only when the readout mode combines that group; otherwise subpixels may still be read or remosaiced separately."
        ),
        "crosstalk_kernel_status": crosstalk_kernel_status,
        "crosstalk_kernel_label": crosstalk_kernel_label,
        "crosstalk_kernel_note": crosstalk_kernel_note,
        "topology_signature": (
            f"{nx or '-'}x{nz or '-'} pixels · "
            f"{parameters.get('cfa_pattern') or '-'} · split={parameters.get('split_mode') or '-'} · "
            f"shield={parameters.get('shield_mode') or '-'}"
        ),
        "pitch_variant_policy": "conditional_scalar_variant" if "pitch_um" in SCALAR_OVERRIDE_FIELDS else "new_base_template",
        "pitch_variant_rule": (
            "pitch_um can be a registered variant only when pixel topology stays fixed; regenerate CAD, mesh, FDTD, and TCAD artifacts and review design rules before using the result."
        ),
        "pitch_scaling_policy": "mixed_pitch_lattice_absolute_process",
        "pitch_scaling_label": "mixed: x/z lattice scales, process dimensions stay absolute",
        "pitch_scaled_fields": [
            "pixel centers",
            "array footprint",
            "CFA tile width = pitch - cfa_gap_um",
            "OCL aperture = OCL span * pitch - lens_edge_gap_um",
            "PD lateral width = pitch - 2 * pd_margin_um",
            "PDAF aperture proxy",
        ],
        "pitch_absolute_fields": [
            "si_thickness_um",
            "passivation_thickness_um",
            "cfa_thickness_um",
            "lens_height_um",
            "lens_edge_gap_um",
            "cfa_gap_um",
            "dti_width_um",
            "dti_depth_um",
            "pd_margin_um",
            "pd_depth_min_um",
            "pd_depth_max_um",
        ],
        "requires_new_base_template_for": topology_fields,
        "notes": [
            "Pixel pitch is the single-pixel size; template footprint is array_nx * pitch by array_nz * pitch.",
            "OCL group pitch is OCL span times pixel pitch; it should not be confused with the full template neighborhood span.",
            "Changing pitch is not a uniform scale operation: x/z lattice-derived geometry follows pitch, while process thicknesses, gaps, DTI, and PD depth/margin parameters remain absolute unless overridden.",
            "A new pixel family or topology change should be a new base template, not a hidden variant override.",
        ],
    }


def load_cad_template_catalog() -> dict[str, Any]:
    if not CAD_TEMPLATE_MANIFEST.exists():
        return {
            "schema": "pixel_cad_template_catalog_v1",
            "status": "missing",
            "template_count": 0,
            "templates": [],
            "starter_template_set": cad_starter_template_set_summary([]),
            "manifest": file_artifact_record(CAD_TEMPLATE_MANIFEST),
            "validation_report": file_artifact_record(CAD_TEMPLATE_VALIDATION_REPORT),
            "freecad_validation_report": file_artifact_record(CAD_TEMPLATE_FREECAD_VALIDATION_REPORT),
            "message": "CAD template library has not been generated yet.",
        }
    manifest = read_json_artifact(CAD_TEMPLATE_MANIFEST)
    validation = read_json_artifact(CAD_TEMPLATE_VALIDATION_REPORT)
    validation_by_template = {
        str(item.get("template_id")): item
        for item in validation.get("templates", [])
        if isinstance(item, dict) and item.get("template_id")
    } if isinstance(validation.get("templates"), list) else {}
    freecad_validation = read_json_artifact(CAD_TEMPLATE_FREECAD_VALIDATION_REPORT)
    freecad_by_template = {
        str(item.get("template_id")): item
        for item in freecad_validation.get("templates", [])
        if isinstance(item, dict) and item.get("template_id")
    } if isinstance(freecad_validation.get("templates"), list) else {}
    templates = []
    for item in manifest.get("templates", []):
        files = item.get("files", {}) if isinstance(item.get("files"), dict) else {}
        artifacts = {name: file_artifact_record(path_from_record(path)) for name, path in files.items()}
        parameters_path = path_from_record(files.get("parameters"))
        template_dir = parameters_path.parent if parameters_path else None
        assumption_ledger_path = path_from_record(files.get("assumption_ledger"))
        tcad_bridge_dir = template_dir / "tcad_bridge_2d" if template_dir else None
        fcstd_path = template_dir / "model.FCStd" if template_dir else None
        tcad_bridge_report = tcad_bridge_dir / "tcad_bridge_report.json" if tcad_bridge_dir else None
        devsim_import_summary = (
            tcad_bridge_dir / "devsim_import_smoke" / "gmsh_pixel_2d_import_summary.json"
            if tcad_bridge_dir
            else None
        )
        devsim_dd_summary = tcad_bridge_dir / "devsim_smoke" / "summary.json" if tcad_bridge_dir else None
        axis_pair_dir = template_dir / "tcad_axis_pair_smoke" if template_dir else None
        axis_pair_summary = axis_pair_dir / "summary.json" if axis_pair_dir else None
        axis_pair_plot = axis_pair_dir / "axis_pair_phase.svg" if axis_pair_dir else None
        qpd_weighting_dir = template_dir / "tcad_qpd_weighting_3d" if template_dir else None
        qpd_weighting_summary = qpd_weighting_dir / "summary.json" if qpd_weighting_dir else None
        qpd_weighting_plot = qpd_weighting_dir / "qpd_weighting_3d.svg" if qpd_weighting_dir else None
        qpd_weighting_mesh = qpd_weighting_dir / "qpd_weighting_3d.msh" if qpd_weighting_dir else None
        qpd_gw_dir = template_dir / "tcad_qpd_gw_3d" if template_dir else None
        qpd_gw_summary = qpd_gw_dir / "summary.json" if qpd_gw_dir else None
        qpd_gw_plot = qpd_gw_dir / "qpd_gw_3d_response.svg" if qpd_gw_dir else None
        qpd_gw_csv = qpd_gw_dir / "qpd_gw_3d_response.csv" if qpd_gw_dir else None
        fdtd_smoke_dir = template_dir / "fdtd_smoke" if template_dir else None
        fdtd_generation_volume = fdtd_smoke_dir / "tcad_generation_volume_3d.npz" if fdtd_smoke_dir else None
        fdtd_generation_map = fdtd_smoke_dir / "tcad_generation_map_2d.npz" if fdtd_smoke_dir else None
        fdtd_smoke_kpi = fdtd_smoke_dir / "kpi_summary.json" if fdtd_smoke_dir else None
        artifacts.update(
            {
                "tcad_bridge_report": file_artifact_record(tcad_bridge_report),
                "tcad_axis_pair_summary": file_artifact_record(axis_pair_summary),
                "tcad_axis_pair_plot": file_artifact_record(axis_pair_plot),
                "tcad_qpd_weighting_3d_summary": file_artifact_record(qpd_weighting_summary),
                "tcad_qpd_weighting_3d_plot": file_artifact_record(qpd_weighting_plot),
                "tcad_qpd_weighting_3d_mesh": file_artifact_record(qpd_weighting_mesh),
                "tcad_qpd_gw_3d_summary": file_artifact_record(qpd_gw_summary),
                "tcad_qpd_gw_3d_plot": file_artifact_record(qpd_gw_plot),
                "tcad_qpd_gw_3d_csv": file_artifact_record(qpd_gw_csv),
                "fdtd_generation_volume_3d": file_artifact_record(fdtd_generation_volume),
                "fdtd_generation_map_2d": file_artifact_record(fdtd_generation_map),
                "fdtd_smoke_kpi": file_artifact_record(fdtd_smoke_kpi),
                "fcstd": file_artifact_record(fcstd_path),
                "freecad_validation_report": file_artifact_record(CAD_TEMPLATE_FREECAD_VALIDATION_REPORT),
                "tcad_mesh_2d": file_artifact_record(tcad_bridge_dir / "split_pixel_2d.msh" if tcad_bridge_dir else None),
                "tcad_derived_config": file_artifact_record(tcad_bridge_dir / "derived_tcad_config.json" if tcad_bridge_dir else None),
                "devsim_import_summary": file_artifact_record(devsim_import_summary),
                "devsim_potential_tecplot": file_artifact_record(
                    tcad_bridge_dir / "devsim_import_smoke" / "gmsh_pixel_2d_potential.dat"
                    if tcad_bridge_dir
                    else None
                ),
                "devsim_dd_summary": file_artifact_record(devsim_dd_summary),
                "devsim_split_currents": file_artifact_record(
                    tcad_bridge_dir / "devsim_smoke" / "split_currents.csv" if tcad_bridge_dir else None
                ),
                "devsim_split_currents_plot": file_artifact_record(
                    tcad_bridge_dir / "devsim_smoke" / "split_currents.png" if tcad_bridge_dir else None
                ),
                "devsim_node_maps_plot": file_artifact_record(
                    tcad_bridge_dir / "devsim_smoke" / "node_maps.png" if tcad_bridge_dir else None
                ),
            }
        )
        template_freecad_validation = freecad_by_template.get(str(item.get("template_id") or ""), {})
        template_validation = validation_by_template.get(str(item.get("template_id") or ""), {})
        parameter_payload = read_json_artifact(parameters_path) if parameters_path and parameters_path.exists() else {}
        parameter_summary = {
            key: parameter_payload.get(key)
            for key in sorted(SCALAR_OVERRIDE_FIELDS)
            if key in parameter_payload
        }
        topology_summary = {
            key: parameter_payload.get(key)
            for key in ("nx", "nz", "cfa_pattern", "split_mode", "shield_mode", "ocl_blocks")
            if key in parameter_payload
        }
        dimension_summary = cad_template_dimension_summary(parameter_payload, template_freecad_validation)
        tcad_report = read_json_artifact(tcad_bridge_report) if tcad_bridge_report and tcad_bridge_report.exists() else {}
        raw_tcad_capability = (
            tcad_report.get("electrical_capability")
            if isinstance(tcad_report.get("electrical_capability"), dict)
            else {}
        )
        tcad_capability = effective_tcad_capability(raw_tcad_capability, parameter_payload)
        devsim_report = (
            read_json_artifact(devsim_import_summary)
            if devsim_import_summary and devsim_import_summary.exists()
            else {}
        )
        devsim_dd_report = (
            read_json_artifact(devsim_dd_summary)
            if devsim_dd_summary and devsim_dd_summary.exists()
            else {}
        )
        devsim_dd_gate = devsim_dd_solver_gate(devsim_dd_report) if devsim_dd_report else {}
        phase_metric_applicability = split_phase_metric_applicability(tcad_capability)
        axis_pair_report = (
            read_json_artifact(axis_pair_summary)
            if axis_pair_summary and axis_pair_summary.exists()
            else {}
        )
        qpd_weighting_report = (
            read_json_artifact(qpd_weighting_summary)
            if qpd_weighting_summary and qpd_weighting_summary.exists()
            else {}
        )
        qpd_gw_report = (
            read_json_artifact(qpd_gw_summary)
            if qpd_gw_summary and qpd_gw_summary.exists()
            else {}
        )
        qpd_gw_generation_gate = qpd_generation_volume_gate(str(item.get("template_id") or ""), qpd_gw_report) if qpd_gw_report else {}
        simulation_fidelity = cad_template_simulation_fidelity(
            fdtd_generation_volume=fdtd_generation_volume,
            tcad_report=tcad_report,
            template_parameters=parameter_payload,
            devsim_dd_report=devsim_dd_report,
            qpd_weighting_report=qpd_weighting_report,
            qpd_gw_report=qpd_gw_report,
        )
        assumption_ledger = (
            read_json_artifact(assumption_ledger_path)
            if assumption_ledger_path and assumption_ledger_path.exists()
            else {}
        )
        solver_defaults: dict[str, Any] | None = None
        solver_error: str | None = None
        try:
            solver_defaults = cad_template_solver_defaults_from_paths(
                str(item.get("template_id") or ""),
                item.get("label"),
                item.get("source_truth_level"),
                path_from_record(files.get("geometry_import")),
                path_from_record(files.get("parameters")),
            )
        except Exception as error:  # noqa: BLE001 - expose local template validation to the UI.
            solver_error = str(error)
        templates.append(
            {
                "template_id": item.get("template_id"),
                "label": item.get("label"),
                "status": item.get("status"),
                "variant_of": item.get("variant_of"),
                "starter_template": item.get("template_id") in CAD_STARTER_TEMPLATE_IDS,
                "parameter_overrides": item.get("parameter_overrides", {}),
                "validation_warnings": item.get("validation_warnings", []),
                "source_truth_level": item.get("source_truth_level"),
                "freecad_openable": item.get("freecad_openable"),
                "freecad_validation": {
                    "available": bool(template_freecad_validation),
                    "status": template_freecad_validation.get("status"),
                    "step_status": template_freecad_validation.get("step", {}).get("status")
                    if isinstance(template_freecad_validation.get("step"), dict)
                    else None,
                    "brep_status": template_freecad_validation.get("brep", {}).get("status")
                    if isinstance(template_freecad_validation.get("brep"), dict)
                    else None,
                    "step_solid_count": template_freecad_validation.get("step", {}).get("solid_count")
                    if isinstance(template_freecad_validation.get("step"), dict)
                    else None,
                    "brep_solid_count": template_freecad_validation.get("brep", {}).get("solid_count")
                    if isinstance(template_freecad_validation.get("brep"), dict)
                    else None,
                    "bbox_um": template_freecad_validation.get("step", {}).get("bbox_um")
                    if isinstance(template_freecad_validation.get("step"), dict)
                    else None,
                    "step_bbox_passed": template_freecad_validation.get("step_bbox_checks", {}).get("passed")
                    if isinstance(template_freecad_validation.get("step_bbox_checks"), dict)
                    else None,
                    "brep_bbox_passed": template_freecad_validation.get("brep_bbox_checks", {}).get("passed")
                    if isinstance(template_freecad_validation.get("brep_bbox_checks"), dict)
                    else None,
                    "fcstd": template_freecad_validation.get("fcstd", {})
                    if isinstance(template_freecad_validation.get("fcstd"), dict)
                    else {},
                },
                "design_rule_validation": {
                    "available": bool(template_validation),
                    "status": template_validation.get("design_rule_status"),
                    "fail_count": template_validation.get("design_rule_fail_count"),
                    "rule_count": template_validation.get("design_rule_validation", {}).get("rule_count")
                    if isinstance(template_validation.get("design_rule_validation"), dict)
                    else None,
                    "rules": template_validation.get("design_rule_validation", {}).get("rules", [])
                    if isinstance(template_validation.get("design_rule_validation"), dict)
                    else [],
                },
                "counts": item.get("counts", {}),
                "parameters": parameter_summary,
                "topology": topology_summary,
                "dimension_summary": dimension_summary,
                "notes": item.get("notes", []),
                "simulation_fidelity": simulation_fidelity,
                "artifacts": artifacts,
                "assumption_ledger": {
                    "available": bool(assumption_ledger),
                    "schema": assumption_ledger.get("schema"),
                    "source_truth_level": assumption_ledger.get("source_truth_level"),
                    "product_accuracy_ready": assumption_ledger.get("product_accuracy_ready"),
                    "assumption_count": len(assumption_ledger.get("assumptions", []))
                    if isinstance(assumption_ledger.get("assumptions"), list)
                    else 0,
                    "measured_blocker_count": len(assumption_ledger.get("measured_blockers", []))
                    if isinstance(assumption_ledger.get("measured_blockers"), list)
                    else 0,
                    "solver_mapping": assumption_ledger.get("solver_mapping", {}),
                },
                "tcad_bridge": {
                    "available": bool(tcad_report),
                    "status": tcad_report.get("status"),
                    "mesh_status": tcad_report.get("mesh_status"),
                    "capability_gate": tcad_capability.get("gate"),
                    "electrical_capability": tcad_capability,
                    "bridge_type": tcad_report.get("bridge_type"),
                    "mesh": tcad_report.get("mesh"),
                    "physical_names": tcad_report.get("physical_names"),
                    "native_full_cad_electrical_mesh": tcad_report.get("native_full_cad_electrical_mesh"),
                    "preserves_full_3d_cad_connectivity": tcad_report.get("preserves_full_3d_cad_connectivity"),
                    "devsim_import_smoke": {
                        "available": bool(devsim_report),
                        "node_count": devsim_report.get("node_count"),
                        "regions": devsim_report.get("regions", []),
                        "contacts": devsim_report.get("contacts", []),
                        "solve": devsim_report.get("solve"),
                    },
                    "devsim_dd_smoke": {
                        "available": bool(devsim_dd_report),
                        "node_count": devsim_dd_report.get("node_count"),
                        "photo_split_phase_x_proxy": devsim_dd_report.get("photo_split_phase_x_proxy"),
                        "photo_split_phase_z_proxy": devsim_dd_report.get("photo_split_phase_z_proxy"),
                        "capability_gate": tcad_capability.get("gate"),
                        "solver_gate": devsim_dd_gate.get("gate"),
                        "solver_gate_reason": devsim_dd_gate.get("reason"),
                        "phase_metric_applicable": phase_metric_applicability.get("applicable"),
                        "phase_metric_reason": phase_metric_applicability.get("reason"),
                        "dd_relative_error": devsim_dd_gate.get("dd_relative_error"),
                        "dd_max_iterations": devsim_dd_gate.get("dd_max_iterations"),
                        "phase_result_axis": devsim_dd_report.get("phase_result_axis") or tcad_capability.get("represented_split_axis"),
                        "phase_result_scope": tcad_capability.get("phase_result_scope"),
                        "contact_axis_labels": devsim_dd_report.get("contact_axis_labels") or tcad_capability.get("contact_axis_labels"),
                        "left_photo_delta_a_per_cm": devsim_dd_report.get("left_photo_delta_a_per_cm"),
                        "right_photo_delta_a_per_cm": devsim_dd_report.get("right_photo_delta_a_per_cm"),
                        "bottom_photo_delta_a_per_cm": devsim_dd_report.get("bottom_photo_delta_a_per_cm"),
                        "top_photo_delta_a_per_cm": devsim_dd_report.get("top_photo_delta_a_per_cm"),
                        "terminal_current_balance_illuminated_a_per_cm": devsim_dd_report.get(
                            "terminal_current_balance_illuminated_a_per_cm"
                        ),
                        "electrical_model": devsim_dd_report.get("config", {}).get("electrical_model")
                        if isinstance(devsim_dd_report.get("config"), dict)
                        else None,
                    },
                    "axis_pair_smoke": {
                        "available": bool(axis_pair_report),
                        "status": axis_pair_report.get("status"),
                        "phase_x_proxy": axis_pair_report.get("phase_x_proxy"),
                        "phase_z_proxy": axis_pair_report.get("phase_z_proxy"),
                        "axis_phase_magnitude": axis_pair_report.get("axis_phase_magnitude"),
                        "axis_signal_uniformity": axis_pair_report.get("axis_signal_uniformity"),
                        "full_q1q4_gate": axis_pair_report.get("full_q1q4_gate"),
                        "summary": axis_pair_report,
                    },
                    "qpd_weighting_3d": {
                        "available": bool(qpd_weighting_report),
                        "status": qpd_weighting_report.get("status"),
                        "node_count": qpd_weighting_report.get("node_count"),
                        "full_q1q4_weighting_gate": qpd_weighting_report.get("full_q1q4_weighting_gate"),
                        "full_q1q4_dd_gate": qpd_weighting_report.get("full_q1q4_dd_gate"),
                        "phase_x_weighting": qpd_weighting_report.get("metrics", {}).get("phase_x_weighting")
                        if isinstance(qpd_weighting_report.get("metrics"), dict)
                        else None,
                        "phase_z_weighting": qpd_weighting_report.get("metrics", {}).get("phase_z_weighting")
                        if isinstance(qpd_weighting_report.get("metrics"), dict)
                        else None,
                        "quadrant_uniformity": qpd_weighting_report.get("metrics", {}).get("quadrant_uniformity")
                        if isinstance(qpd_weighting_report.get("metrics"), dict)
                        else None,
                        "summary": qpd_weighting_report,
                    },
                    "qpd_gw_3d": {
                        "available": bool(qpd_gw_report),
                        "status": qpd_gw_report.get("status"),
                        "node_count": qpd_gw_report.get("node_count"),
                        "case_count": qpd_gw_report.get("case_count"),
                        "full_q1q4_gw_gate": qpd_gw_report.get("full_q1q4_gw_gate"),
                        "full_q1q4_dd_gate": qpd_gw_report.get("full_q1q4_dd_gate"),
                        "generation_volume_gate": qpd_gw_generation_gate.get("gate"),
                        "generation_volume_reason": qpd_gw_generation_gate.get("reason"),
                        "generation_volume_npz": qpd_gw_generation_gate.get("generation_volume_npz"),
                        "method": qpd_gw_report.get("method"),
                        "edge_to_center_response_ratio_min": qpd_gw_report.get("field_response_summary", {}).get(
                            "edge_to_center_response_ratio_min"
                        )
                        if isinstance(qpd_gw_report.get("field_response_summary"), dict)
                        else None,
                        "edge_to_center_response_ratio_max": qpd_gw_report.get("field_response_summary", {}).get(
                            "edge_to_center_response_ratio_max"
                        )
                        if isinstance(qpd_gw_report.get("field_response_summary"), dict)
                        else None,
                        "phase_x_slope_per_deg_max_abs": qpd_gw_report.get("field_response_summary", {}).get(
                            "phase_x_slope_per_deg_max_abs"
                        )
                        if isinstance(qpd_gw_report.get("field_response_summary"), dict)
                        else None,
                        "phase_x_gw": qpd_gw_report.get("cases", [{}])[0].get("metrics", {}).get("phase_x_gw")
                        if isinstance(qpd_gw_report.get("cases"), list) and qpd_gw_report.get("cases")
                        else None,
                        "phase_z_gw": qpd_gw_report.get("cases", [{}])[0].get("metrics", {}).get("phase_z_gw")
                        if isinstance(qpd_gw_report.get("cases"), list) and qpd_gw_report.get("cases")
                        else None,
                        "quadrant_uniformity_gw": qpd_gw_report.get("cases", [{}])[0]
                        .get("metrics", {})
                        .get("quadrant_uniformity_gw")
                        if isinstance(qpd_gw_report.get("cases"), list) and qpd_gw_report.get("cases")
                        else None,
                        "generation_weighted_qsum_fraction": qpd_gw_report.get("cases", [{}])[0]
                        .get("metrics", {})
                        .get("generation_weighted_qsum_fraction")
                        if isinstance(qpd_gw_report.get("cases"), list) and qpd_gw_report.get("cases")
                        else None,
                        "summary": qpd_gw_report,
                    },
                },
                "solver_ready": solver_defaults is not None,
                "solver_defaults": solver_defaults,
                "solver_error": solver_error,
            }
        )
    for template in templates:
        template["readiness"] = cad_template_readiness(template)
    starter_template_set = cad_starter_template_set_summary(templates)
    qpd_comparison = qpd_template_comparison(templates)
    solver_role_matrix = cad_solver_role_matrix(templates, qpd_comparison)
    return {
        "schema": "pixel_cad_template_catalog_v1",
        "status": "PASS" if validation.get("status") == "PASS" and starter_template_set.get("status") == "PASS" else "CHECK",
        "generated_with": manifest.get("generated_with"),
        "freecad_role": manifest.get("freecad_role"),
        "mask_role": manifest.get("mask_role"),
        "mesh_role": manifest.get("mesh_role"),
        "accuracy_status": manifest.get("accuracy_status"),
        "template_count": len(templates),
        "starter_template_set": starter_template_set,
        "qpd_template_comparison": qpd_comparison,
        "solver_role_matrix": solver_role_matrix,
        "base_template_count": sum(1 for template in templates if not template.get("variant_of")),
        "variant_count": sum(1 for template in templates if template.get("variant_of")),
        "manifest": file_artifact_record(CAD_TEMPLATE_MANIFEST),
        "validation_report": file_artifact_record(CAD_TEMPLATE_VALIDATION_REPORT),
        "freecad_validation_report": file_artifact_record(CAD_TEMPLATE_FREECAD_VALIDATION_REPORT),
        "freecad_validation": {
            "available": bool(freecad_validation),
            "status": freecad_validation.get("status", "CHECK"),
            "template_count": freecad_validation.get("template_count"),
            "pass_count": sum(
                1
                for item in freecad_validation.get("templates", [])
                if isinstance(item, dict) and item.get("status") == "PASS"
            )
            if isinstance(freecad_validation.get("templates"), list)
            else 0,
            "write_fcstd": freecad_validation.get("write_fcstd"),
            "freecad": freecad_validation.get("freecad", {}),
        },
        "validation": {
            "status": validation.get("status", "CHECK"),
            "template_count": validation.get("template_count"),
            "mesh_expected": validation.get("mesh_expected"),
            "mesh_pass_count": sum(
                1
                for item in validation.get("templates", [])
                if item.get("mesh_exists") and not item.get("mesh_required_groups_missing")
            )
            if isinstance(validation.get("templates"), list)
            else 0,
            "design_rule_pass_count": sum(
                1
                for item in validation.get("templates", [])
                if item.get("design_rule_status") == "PASS"
            )
            if isinstance(validation.get("templates"), list)
            else 0,
            "design_rule_fail_count": sum(
                int(item.get("design_rule_fail_count") or 0)
                for item in validation.get("templates", [])
            )
            if isinstance(validation.get("templates"), list)
            else 0,
            "fdtd_smoke": validation.get("fdtd_smoke", {}),
            "notes": validation.get("notes", []),
        },
        "templates": templates,
    }


def find_cad_template(catalog: dict[str, Any], template_id: str) -> dict[str, Any]:
    for template in catalog.get("templates", []):
        if isinstance(template, dict) and template.get("template_id") == template_id:
            return template
    raise KeyError(f"Unknown CAD template: {template_id}")


def workspace_safe_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if path_text.startswith(("/runs/", "/configs/", "/materials/")):
        candidate = ROOT / path_text.lstrip("/")
    elif not candidate.is_absolute():
        candidate = ROOT / candidate
    path = candidate.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Refusing to open path outside workspace: {path}")
    if not path.exists():
        raise FileNotFoundError(f"CAD artifact does not exist: {path}")
    return path


def workspace_safe_output_path(path_text: str) -> Path:
    if not path_text:
        raise ValueError("output path is required")
    candidate = Path(path_text)
    path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Refusing to write path outside workspace: {path}")
    return path


def open_cad_artifact(template_id: str, artifact: str, prefer_freecad: bool = True) -> dict[str, Any]:
    if artifact not in CAD_OPENABLE_ARTIFACTS:
        raise ValueError(f"Unsupported CAD artifact: {artifact}")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    artifacts = template.get("artifacts", {}) if isinstance(template.get("artifacts"), dict) else {}
    record = artifacts.get(artifact)
    if not isinstance(record, dict) or not record.get("path"):
        raise FileNotFoundError(f"Template {template_id} has no artifact named {artifact}")
    path = workspace_safe_path(str(record["path"]))
    use_freecad = prefer_freecad and artifact in FREECAD_PREFERRED_ARTIFACTS
    command = open_command(path, prefer_freecad=use_freecad) if use_freecad else ["open", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "schema": "pixel_workbench_cad_open_result_v1",
        "status": "launched",
        "template_id": template_id,
        "template_label": template.get("label"),
        "artifact": artifact,
        "path": str(path),
        "used_freecad": use_freecad and bool(freecad_status().get("installed")),
        "command": command,
        "freecad": freecad_status(),
    }


def open_workspace_file(path_text: str, prefer_freecad: bool = True) -> dict[str, Any]:
    if not path_text:
        raise ValueError("path is required")
    path = workspace_safe_path(path_text)
    use_freecad = prefer_freecad and path.suffix.lower() in {".step", ".stp", ".brep", ".fcstd"}
    command = open_command(path, prefer_freecad=use_freecad) if use_freecad else ["open", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "schema": "pixel_workbench_workspace_file_open_result_v1",
        "status": "launched",
        "path": str(path),
        "used_freecad": use_freecad and bool(freecad_status().get("installed")),
        "command": command,
        "freecad": freecad_status(),
    }


def replay_case_from_request(body: dict[str, Any]) -> dict[str, Any]:
    case_command, command, timeout_sec = replay_command_from_request(body)
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=timeout_sec + 30.0)
    try:
        replay = parse_json_from_mixed_stdout(completed.stdout)
    except Exception:
        replay = {
            "schema": "pixel_workbench_case_command_replay_v1",
            "status": "FAIL",
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout.splitlines()[-80:],
        }
    replay["api_status"] = "PASS" if completed.returncode == 0 else "FAIL"
    replay["api_command"] = command
    replay["stderr_tail"] = completed.stderr.splitlines()[-40:]
    if completed.returncode != 0 and replay.get("status") == "PASS":
        replay["status"] = "FAIL"
    return replay_case_api_payload(case_command, command, replay)


def replay_command_from_request(body: dict[str, Any], output_dir: Path | None = None) -> tuple[Path, list[str], float]:
    case_command_text = str(body.get("case_command") or body.get("case_command_json") or "")
    if not case_command_text:
        raise ValueError("case_command is required")
    case_command = workspace_safe_path(case_command_text)
    command = [sys.executable, str(ROOT / "pixel_workbench_replay.py"), str(case_command)]
    output_dir_text = str(body.get("output_dir") or "")
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    elif output_dir_text:
        command.extend(["--output-dir", str(workspace_safe_output_path(output_dir_text))])
    if bool(body.get("compare_source", True)):
        command.append("--compare-source")
    if bool(body.get("allow_original_output", False)):
        command.append("--allow-original-output")
    timeout_sec = json_safe_number(body.get("timeout_sec")) or 300.0
    command.extend(["--timeout-sec", f"{timeout_sec:g}"])
    return case_command, command, timeout_sec


def replay_case_api_payload(case_command: Path, command: list[str], replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "pixel_workbench_replay_case_api_v1",
        "status": replay.get("status", "FAIL"),
        "case_command": str(case_command),
        "replay": replay,
        "api_command": command,
        "replay_manifest_url": rel_url(Path(str(replay.get("replay_manifest") or ""))) if replay.get("replay_manifest") else None,
        "replay_comparison_url": rel_url(Path(str(replay.get("replay_comparison") or ""))) if replay.get("replay_comparison") else None,
        "output_url": rel_url(Path(str(replay.get("output_dir") or ""))) if replay.get("output_dir") else None,
    }


def default_replay_job_output_dir(case_command: Path, job_id: str) -> Path:
    case_id = sanitize_id(case_command.parent.name or "case")
    return ROOT / "runs" / "replay_jobs" / f"{case_id}_{job_id}"


def run_replay_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        body = copy.deepcopy(job.get("replay_request") or {})
        output_dir = Path(job["output_dir"])
    log_path = output_dir / "replay.log"
    summary_path = output_dir / "workbench_replay_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        case_command, command, timeout_sec = replay_command_from_request(body, output_dir=output_dir)
        update_job(
            job_id,
            status="running",
            started_at=now_iso(),
            command=command,
            log_url=rel_url(log_path),
            case_command=str(case_command),
            case_command_url=rel_url(case_command),
        )
        append_log(job_id, "Starting replay job")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 60.0,
        )
        log_lines = ["$ " + " ".join(command), "", completed.stdout or ""]
        if completed.stderr:
            log_lines.extend(["", "# stderr", completed.stderr])
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        for line in (completed.stdout or "").splitlines()[-120:]:
            append_log(job_id, line)
        try:
            replay = parse_json_from_mixed_stdout(completed.stdout)
        except Exception:
            replay = {
                "schema": "pixel_workbench_case_command_replay_v1",
                "status": "FAIL",
                "return_code": completed.returncode,
                "stdout_tail": (completed.stdout or "").splitlines()[-80:],
            }
        replay["api_status"] = "PASS" if completed.returncode == 0 else "FAIL"
        replay["api_command"] = command
        replay["stderr_tail"] = (completed.stderr or "").splitlines()[-40:]
        if completed.returncode != 0 and replay.get("status") == "PASS":
            replay["status"] = "FAIL"
        replay_result = replay_case_api_payload(case_command, command, replay)
        completed_at = now_iso()
        replay_summary = {
            "schema": "pixel_workbench_replay_job_summary_v1",
            "job_id": job_id,
            "status": "completed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": completed_at,
            "case_command": str(case_command),
            "case_command_url": rel_url(case_command),
            "output_url": replay_result.get("output_url") or job.get("output_url"),
            "log_url": rel_url(log_path),
            "replay_manifest_url": replay_result.get("replay_manifest_url"),
            "replay_comparison_url": replay_result.get("replay_comparison_url"),
            "replay_status": replay_result.get("status"),
            "replay_comparison_status": replay.get("replay_comparison_status"),
        }
        write_json_artifact(summary_path, replay_summary)
        update_job(
            job_id,
            status="completed" if completed.returncode == 0 else "failed",
            return_code=completed.returncode,
            completed_at=completed_at,
            replay_result=replay_result,
            replay_summary_url=rel_url(summary_path),
            replay_manifest_url=replay_result.get("replay_manifest_url"),
            replay_comparison_url=replay_result.get("replay_comparison_url"),
            log_url=rel_url(log_path),
            output_url=replay_result.get("output_url") or job.get("output_url"),
            error=None if completed.returncode == 0 else f"replay exited with {completed.returncode}",
        )
    except Exception as error:  # noqa: BLE001 - local replay diagnostics.
        update_job(job_id, status="failed", completed_at=now_iso(), error=str(error), log_url=rel_url(log_path))


def create_replay_job(body: dict[str, Any]) -> dict[str, Any]:
    case_command, _command, _timeout_sec = replay_command_from_request(body)
    job_id = f"replay_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_dir_text = str(body.get("output_dir") or "")
    output_dir = workspace_safe_output_path(output_dir_text) if output_dir_text else default_replay_job_output_dir(case_command, job_id)
    replay_request = copy.deepcopy(body)
    if "output_dir" in replay_request:
        replay_request.pop("output_dir", None)
    job = {
        "schema": "pixel_workbench_solver_job_v1",
        "kind": "replay",
        "id": job_id,
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "completed_at": None,
        "output_dir": str(output_dir),
        "output_url": "/" + output_dir.resolve().relative_to(ROOT).as_posix(),
        "case_command": str(case_command),
        "case_command_url": rel_url(case_command),
        "replay_request": replay_request,
        "command": None,
        "log_tail": [],
        "log_url": None,
        "replay_result": None,
        "replay_summary_url": None,
        "replay_manifest_url": None,
        "replay_comparison_url": None,
        "error": None,
        "progress": {"completed": 0, "total": 1, "current_case": case_command.parent.name},
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=run_replay_job, args=(job_id,), daemon=True)
    thread.start()
    return snapshot_job(job_id)


def validate_freecad_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if template_id:
        find_cad_template(load_cad_template_catalog(), template_id)
    report = validate_freecad_library(
        CAD_TEMPLATE_ROOT,
        template_id="",
        write_fcstd=bool(body.get("write_fcstd", True)),
        output_path=CAD_TEMPLATE_FREECAD_VALIDATION_REPORT,
        tolerance_um=json_safe_number(body.get("tolerance_um")) or 1.0e-6,
    )
    catalog = load_cad_template_catalog()
    return {
        "schema": "pixel_workbench_freecad_validation_result_v1",
        "status": report.get("status"),
        "template_id": template_id or None,
        "report": report,
        "report_url": rel_url(CAD_TEMPLATE_FREECAD_VALIDATION_REPORT),
        "catalog": catalog,
    }


def values_equal_for_override(left: Any, right: Any) -> bool:
    left_number = json_safe_number(left)
    right_number = json_safe_number(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= 1.0e-12
    return str(left) == str(right)


def fcstd_path_for_template(template: dict[str, Any]) -> Path:
    artifacts = template.get("artifacts", {}) if isinstance(template.get("artifacts"), dict) else {}
    fcstd = artifacts.get("fcstd") if isinstance(artifacts.get("fcstd"), dict) else {}
    if not fcstd.get("path"):
        raise FileNotFoundError(f"Template {template.get('template_id')} has no FCStd artifact")
    return workspace_safe_path(str(fcstd["path"]))


def create_fcstd_working_copy_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    source_path = fcstd_path_for_template(template)
    raw_copy_id = str(body.get("copy_id") or "") or f"{template_id}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    copy_id = sanitize_id(raw_copy_id)[:96]
    if not copy_id:
        raise ValueError("copy_id is empty after sanitization")
    target_dir = FCSTD_WORKING_COPY_ROOT / sanitize_id(template_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{copy_id}.FCStd"
    if target_path.exists() and not bool(body.get("overwrite", False)):
        target_path = target_dir / f"{copy_id}_{uuid.uuid4().hex[:8]}.FCStd"
    resolved_target = target_path.resolve()
    if ROOT not in resolved_target.parents:
        raise ValueError(f"Refusing to create FCStd working copy outside workspace: {resolved_target}")
    shutil.copy2(source_path, resolved_target)
    artifact = file_artifact_record(resolved_target)
    result = {
        "schema": "pixel_workbench_fcstd_working_copy_result_v1",
        "status": "created",
        "template_id": template_id,
        "template_label": template.get("label"),
        "source_fcstd_path": str(source_path),
        "fcstd_path": str(resolved_target),
        "fcstd_url": artifact.get("url"),
        "artifact": artifact,
        "freecad": freecad_status(),
        "notes": [
            "Edit this working copy instead of the base model.FCStd.",
            "Only scalar TemplateParameters changes are imported into registered variants.",
        ],
    }
    if bool(body.get("open_freecad", False)):
        open_result = open_workspace_file(str(resolved_target), prefer_freecad=True)
        result["open_result"] = open_result
    return result


def fcstd_parameter_diff(template_id: str, fcstd_path_override: str | None = None) -> dict[str, Any]:
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    fcstd_path = workspace_safe_path(fcstd_path_override) if fcstd_path_override else fcstd_path_for_template(template)
    extraction = extract_fcstd_parameters(fcstd_path)
    parameters = extraction.get("parameters") if isinstance(extraction.get("parameters"), dict) else {}
    artifacts = template.get("artifacts", {}) if isinstance(template.get("artifacts"), dict) else {}
    parameters_artifact = artifacts.get("parameters") if isinstance(artifacts.get("parameters"), dict) else {}
    if not parameters_artifact.get("path"):
        raise FileNotFoundError(f"Template {template_id} has no template_parameters artifact")
    base_parameters = read_json_artifact(workspace_safe_path(str(parameters_artifact["path"])))

    overrides: dict[str, Any] = {}
    unchanged: dict[str, Any] = {}
    missing_scalar_fields: list[str] = []
    for field in sorted(SCALAR_OVERRIDE_FIELDS):
        if field not in parameters:
            missing_scalar_fields.append(field)
            continue
        candidate = parameters[field]
        current = base_parameters.get(field)
        if values_equal_for_override(candidate, current):
            unchanged[field] = current
        else:
            overrides[field] = candidate

    unsupported_changes = []
    for field in sorted(CAD_METADATA_CHANGE_FIELDS | CAD_TOPOLOGY_CHANGE_FIELDS):
        if field in parameters and not values_equal_for_override(parameters.get(field), base_parameters.get(field)):
            reason = (
                "Topology changes require a new base CAD template, not a scalar variant."
                if field in CAD_TOPOLOGY_CHANGE_FIELDS
                else "Template identity/notes metadata is not imported as a registered variant override."
            )
            unsupported_changes.append(
                {
                    "field": field,
                    "current": base_parameters.get(field),
                    "fcstd": parameters.get(field),
                    "reason": reason,
                }
            )
    return {
        "schema": "pixel_workbench_fcstd_parameter_diff_v1",
        "template_id": template_id,
        "template_label": template.get("label"),
        "fcstd_path": str(fcstd_path),
        "extraction": extraction,
        "overrides": overrides,
        "unchanged_scalar_count": len(unchanged),
        "missing_scalar_fields": missing_scalar_fields,
        "unsupported_changes": unsupported_changes,
        "can_create_variant": extraction.get("status") == "PASS" and bool(overrides) and not unsupported_changes,
        "notes": [
            "FCStd round-trip imports scalar TemplateSpec fields only.",
            "Topology fields such as nx/nz, OCL blocks, CFA pattern, split mode, and shield mode must remain explicit base-template changes.",
        ],
    }


def extract_fcstd_parameters_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    fcstd_path = str(body.get("fcstd_path") or "") or None
    diff = fcstd_parameter_diff(template_id, fcstd_path_override=fcstd_path)
    return {
        "schema": "pixel_workbench_fcstd_parameter_extract_result_v1",
        "status": "PASS" if diff["extraction"].get("status") == "PASS" else "FAIL",
        **diff,
    }


def create_cad_variant_from_fcstd_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    fcstd_path = str(body.get("fcstd_path") or "") or None
    diff = fcstd_parameter_diff(template_id, fcstd_path_override=fcstd_path)
    if diff.get("unsupported_changes"):
        raise ValueError("FCStd contains unsupported topology/metadata changes: " + ", ".join(item["field"] for item in diff["unsupported_changes"]))
    overrides = diff.get("overrides", {})
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("FCStd TemplateParameters has no scalar changes versus the selected template")
    create_body = {
        "base_template": template_id,
        "id": body.get("id") or body.get("template_id_out") or variant_id_from_overrides(template_id, overrides),
        "label": body.get("label") or f"{diff.get('template_label') or template_id} FCStd variant",
        "overrides": overrides,
    }
    result = create_cad_variant_from_request(create_body)
    result["schema"] = "pixel_workbench_fcstd_variant_create_result_v1"
    result["fcstd_import"] = diff
    return result


def variant_id_from_overrides(base_template: str, overrides: dict[str, Any]) -> str:
    parts = [base_template, "ui"]
    for key, value in sorted(overrides.items()):
        short_key = key.removesuffix("_um").replace("_", "")
        if isinstance(value, float):
            value_text = f"{value:.4g}"
        else:
            value_text = str(value)
        parts.append(f"{short_key}_{value_text}")
    return sanitize_id("_".join(parts))


def normalize_variant_overrides(raw_overrides: Any) -> dict[str, Any]:
    if not isinstance(raw_overrides, dict):
        raise ValueError("overrides must be a JSON object")
    overrides: dict[str, Any] = {}
    for raw_key, value in raw_overrides.items():
        key = str(raw_key)
        if key not in SCALAR_OVERRIDE_FIELDS:
            raise ValueError(f"Unsupported CAD template override {key!r}")
        if value is None or value == "":
            continue
        overrides[key] = value
    if not overrides:
        raise ValueError("At least one non-empty override is required")
    return overrides


def parse_json_from_mixed_stdout(stdout: str) -> dict[str, Any]:
    starts = [index for index, char in enumerate(stdout) if char == "{"]
    for index in reversed(starts):
        try:
            payload = json.loads(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("CAD variant creator did not emit a JSON object")


def create_cad_variant_from_request(body: dict[str, Any]) -> dict[str, Any]:
    base_template = str(body.get("base_template") or body.get("base_template_id") or "")
    if not base_template:
        raise ValueError("base_template is required")
    find_cad_template(load_cad_template_catalog(), base_template)
    overrides = normalize_variant_overrides(body.get("overrides"))
    variant_id = sanitize_id(str(body.get("id") or body.get("template_id") or variant_id_from_overrides(base_template, overrides)))
    label = str(body.get("label") or f"{base_template} UI variant")
    python_bin = TCAD_PYTHON if TCAD_PYTHON.exists() else Path(sys.executable)
    command = [
        str(python_bin),
        str(ROOT / "cad_template_variant_create.py"),
        "--base-template",
        base_template,
        "--id",
        variant_id,
        "--label",
        label,
        "--library-root",
        str(CAD_TEMPLATE_ROOT),
        "--output-dir",
        str(CAD_TEMPLATE_ROOT),
    ]
    for key, value in sorted(overrides.items()):
        command.extend(["--set", f"{key}={value}"])
    if body.get("allow_warnings"):
        command.append("--allow-warnings")
    if body.get("no_mesh"):
        command.append("--no-mesh")
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=300)
    result = parse_json_from_mixed_stdout(completed.stdout)
    freecad_validation_result: dict[str, Any] | None = None
    freecad_validation_error: str | None = None
    if not body.get("skip_freecad_validation"):
        try:
            freecad_validation_result = validate_freecad_library(
                CAD_TEMPLATE_ROOT,
                template_id="",
                write_fcstd=True,
                output_path=CAD_TEMPLATE_FREECAD_VALIDATION_REPORT,
            )
        except Exception as error:  # noqa: BLE001 - variant still exists; expose review package failure.
            freecad_validation_error = str(error)
    catalog = load_cad_template_catalog()
    return {
        "schema": "pixel_workbench_cad_variant_create_api_v1",
        "status": result.get("status", "CHECK"),
        "variant": result,
        "freecad_validation": {
            "status": freecad_validation_result.get("status") if freecad_validation_result else "FAIL" if freecad_validation_error else None,
            "template_count": freecad_validation_result.get("template_count") if freecad_validation_result else None,
            "error": freecad_validation_error,
        },
        "catalog": catalog,
        "command": command,
        "stderr": completed.stderr[-4000:],
    }


def ocl_blocks_for_base_topology(preset_id: str, preset: dict[str, Any]) -> tuple[OclBlock, ...]:
    nx = int(preset["nx"])
    nz = int(preset["nz"])
    ocl = str(preset.get("ocl") or "unit")
    if ocl == "unit":
        return tuple(OclBlock(f"ocl_{ix}_{iz}", ix, iz, 1, 1) for iz in range(nz) for ix in range(nx))
    if ocl == "single_full":
        return (OclBlock(f"{preset_id}_ocl", 0, 0, nx, nz),)
    if ocl == "block_2x2":
        if nx % 2 or nz % 2:
            raise ValueError(f"Topology preset {preset_id} requires even nx/nz for 2x2 OCL blocks")
        return tuple(OclBlock(f"quad_{ix}_{iz}", ix, iz, 2, 2) for iz in range(0, nz, 2) for ix in range(0, nx, 2))
    if ocl == "block_3x3":
        if nx % 3 or nz % 3:
            raise ValueError(f"Topology preset {preset_id} requires nx/nz multiples of 3 for 3x3 OCL blocks")
        return tuple(OclBlock(f"nona_{ix}_{iz}", ix, iz, 3, 3) for iz in range(0, nz, 3) for ix in range(0, nx, 3))
    if ocl == "mixed_boundary":
        return (
            OclBlock("nona_left", 0, 0, 3, 3),
            OclBlock("quad_right", 3, 0, 2, 2),
            OclBlock("bayer_r0", 3, 2, 1, 1),
            OclBlock("bayer_r1", 4, 2, 1, 1),
        )
    raise ValueError(f"Unsupported base-template OCL topology preset: {ocl}")


def base_template_spec_from_request(body: dict[str, Any]) -> TemplateSpec:
    topology_preset = str(body.get("topology_preset") or body.get("topology") or "qpd_2x2").strip()
    preset = CAD_BASE_TEMPLATE_TOPOLOGY_PRESETS.get(topology_preset)
    if not preset:
        raise ValueError(f"Unknown base-template topology preset: {topology_preset}")
    template_id = sanitize_id(str(body.get("id") or body.get("template_id") or ""))
    if not template_id:
        raise ValueError("id/template_id is required for a new base template")
    label = str(body.get("label") or f"{preset['label']} base").strip()
    raw_parameters = body.get("parameters") if isinstance(body.get("parameters"), dict) else {}
    parameters = normalize_variant_overrides(raw_parameters) if raw_parameters else {}
    template_fields = {
        "template_id": template_id,
        "label": label,
        "nx": int(preset["nx"]),
        "nz": int(preset["nz"]),
        "cfa_pattern": str(preset["cfa_pattern"]),
        "ocl_blocks": ocl_blocks_for_base_topology(topology_preset, preset),
        "split_mode": str(preset["split_mode"]),
        "shield_mode": str(preset["shield_mode"]),
        "notes": (
            f"Base template created from topology preset {topology_preset}.",
            "Topology changes are tracked as a base CAD template, not a scalar variant.",
        ),
        **parameters,
    }
    return TemplateSpec(**template_fields)


def append_or_replace_template_record(manifest: dict[str, Any], record: dict[str, Any], *, replace_existing: bool) -> dict[str, Any]:
    records = [item for item in manifest.get("templates", []) if isinstance(item, dict)]
    existing_index = next((index for index, item in enumerate(records) if item.get("template_id") == record["template_id"]), None)
    if existing_index is not None:
        if not replace_existing:
            raise ValueError(f"CAD base template already exists: {record['template_id']}")
        records[existing_index] = record
    else:
        records.append(record)
    manifest["templates"] = records
    manifest["template_count"] = len(records)
    manifest["accuracy_status"] = "parametric_templates_not_measured"
    return manifest


def create_cad_base_template_from_request(body: dict[str, Any]) -> dict[str, Any]:
    spec = base_template_spec_from_request(body)
    dry_run = bool(body.get("dry_run"))
    if dry_run:
        return {
            "schema": "pixel_workbench_cad_base_template_create_api_v1",
            "status": "DRY_RUN",
            "template_id": spec.template_id,
            "topology_preset": body.get("topology_preset") or body.get("topology") or "qpd_2x2",
            "parameters": {
                "pitch_um": spec.pitch_um,
                "nx": spec.nx,
                "nz": spec.nz,
                "cfa_pattern": spec.cfa_pattern,
                "split_mode": spec.split_mode,
                "shield_mode": spec.shield_mode,
                "ocl_block_count": len(spec.ocl_blocks),
            },
            "notes": [
                "Dry run only; no CAD files were written.",
                "Use this endpoint without dry_run to register a new base template.",
            ],
        }

    manifest = read_json_artifact(CAD_TEMPLATE_MANIFEST) if CAD_TEMPLATE_MANIFEST.exists() else {
        "schema": "pixel_cad_template_library_manifest_v1",
        "output_dir": str(CAD_TEMPLATE_ROOT),
        "generated_with": "Gmsh/OpenCASCADE",
        "freecad_role": "Open generated STEP/BREP files for 3D review; FreeCAD is not required for headless generation.",
        "mask_role": "Use geometry_import.json or downstream GDS export for solver footprints.",
        "mesh_role": "Optional model.msh files are coarse 3D CAD review meshes with physical volume groups; they are not calibrated DEVSIM electrical meshes.",
        "accuracy_status": "parametric_templates_not_measured",
        "templates": [],
    }
    existing_ids = {str(item.get("template_id")) for item in manifest.get("templates", []) if isinstance(item, dict)}
    replace_existing = bool(body.get("replace_existing"))
    if spec.template_id in existing_ids and not replace_existing:
        raise ValueError(f"CAD base template already exists: {spec.template_id}")
    record = write_template(spec, CAD_TEMPLATE_ROOT, mesh=not bool(body.get("no_mesh")))
    record["base_template_source"] = {
        "schema": "pixel_workbench_cad_base_template_source_v1",
        "topology_preset": body.get("topology_preset") or body.get("topology") or "qpd_2x2",
        "created_by": "pixel_workbench_api",
        "topology_changes_allowed_as_variant": False,
    }
    manifest = append_or_replace_template_record(manifest, record, replace_existing=replace_existing)
    CAD_TEMPLATE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mesh_expected = any(
        isinstance(item, dict) and isinstance(item.get("files"), dict) and bool(item["files"].get("mesh"))
        for item in manifest.get("templates", [])
    )
    validation = validate_library(manifest["templates"], CAD_TEMPLATE_ROOT, mesh_expected=mesh_expected)
    CAD_TEMPLATE_VALIDATION_REPORT.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    freecad_validation_result: dict[str, Any] | None = None
    freecad_validation_error: str | None = None
    if not body.get("skip_freecad_validation"):
        try:
            freecad_validation_result = validate_freecad_library(
                CAD_TEMPLATE_ROOT,
                template_id="",
                write_fcstd=True,
                output_path=CAD_TEMPLATE_FREECAD_VALIDATION_REPORT,
            )
        except Exception as error:  # noqa: BLE001 - base template still exists; expose validation failure.
            freecad_validation_error = str(error)
    catalog = load_cad_template_catalog()
    return {
        "schema": "pixel_workbench_cad_base_template_create_api_v1",
        "status": validation.get("status", "CHECK"),
        "template_id": spec.template_id,
        "record": record,
        "validation": validation,
        "freecad_validation": {
            "status": freecad_validation_result.get("status") if freecad_validation_result else "FAIL" if freecad_validation_error else None,
            "template_count": freecad_validation_result.get("template_count") if freecad_validation_result else None,
            "error": freecad_validation_error,
        },
        "catalog": catalog,
    }


def generate_tcad_bridge_artifacts(template_id: str, body: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    template_id = str(template_id or body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    template_dir = CAD_TEMPLATE_ROOT / template_id
    if not (template_dir / "template_parameters.json").exists():
        raise FileNotFoundError(f"CAD template parameters not found for {template_id}")

    python_bin = TCAD_PYTHON if TCAD_PYTHON.exists() else Path(sys.executable)
    mesh_command = [
        str(python_bin),
        str(ROOT / "tcad_mesh_from_cad_template.py"),
        "--template-id",
        template_id,
        "--library-root",
        str(CAD_TEMPLATE_ROOT),
        "--output-dir",
        str(output_dir),
        "--domain",
        str(body.get("domain") or "target-ocl"),
        "--section-axis",
        str(body.get("section_axis") or "auto"),
        "--mesh-um",
        str(body.get("mesh_um") or 0.18),
        "--fine-mesh-um",
        str(body.get("fine_mesh_um") or 0.06),
    ]
    if body.get("include_dti_oxide"):
        mesh_command.append("--include-dti-oxide")
    if body.get("include_fd_contact", True):
        mesh_command.append("--include-fd-contact")
    if body.get("include_tg_contact", True):
        mesh_command.append("--include-tg-contact")
    if body.get("include_tg_oxide"):
        mesh_command.append("--include-tg-oxide")

    mesh_run = subprocess.run(mesh_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=300)
    bridge_report = parse_json_from_mixed_stdout(mesh_run.stdout)

    import_command = [
        str(python_bin),
        str(ROOT / "devsim_gmsh_pixel_import.py"),
        "--mesh",
        str(output_dir / "split_pixel_2d.msh"),
        "--dimension",
        "2",
        "--output-dir",
        str(output_dir / "devsim_import_smoke"),
    ]
    measured_profile = body.get("measured_profile")
    if measured_profile:
        import_command.extend(["--measured-profile", str(measured_profile)])
    import_run = subprocess.run(import_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=300)
    import_summary = parse_json_from_mixed_stdout(import_run.stdout)
    mesh_status = bridge_report.get("mesh_status") or bridge_report.get("status")
    capability = bridge_report.get("electrical_capability") if isinstance(bridge_report.get("electrical_capability"), dict) else {}
    capability_gate = capability.get("gate") or "PASS"
    return {
        "status": "PASS" if mesh_status == "PASS" and capability_gate == "PASS" and import_summary.get("node_count", 0) > 0 else "CHECK",
        "mesh_status": mesh_status,
        "capability_gate": capability_gate,
        "template": template,
        "bridge_report": bridge_report,
        "import_smoke": import_summary,
        "commands": {
            "mesh": mesh_command,
            "import_smoke": import_command,
        },
        "stderr": {
            "mesh": mesh_run.stderr[-4000:],
            "import_smoke": import_run.stderr[-4000:],
        },
    }


def generate_tcad_bridge_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    template_dir = CAD_TEMPLATE_ROOT / template_id
    output_dir = template_dir / "tcad_bridge_2d"
    generated = generate_tcad_bridge_artifacts(template_id, body, output_dir)
    updated_catalog = load_cad_template_catalog()
    updated_template = find_cad_template(updated_catalog, template_id)
    return {
        "schema": "pixel_workbench_tcad_bridge_generate_v1",
        "status": generated["status"],
        "mesh_status": generated["mesh_status"],
        "capability_gate": generated["capability_gate"],
        "template_id": template_id,
        "template_label": generated["template"].get("label"),
        "bridge_report": generated["bridge_report"],
        "import_smoke": generated["import_smoke"],
        "template": updated_template,
        "catalog": updated_catalog,
        "commands": generated["commands"],
        "stderr": generated["stderr"],
        "notes": [
            "This generates a parameter-derived 2D TCAD bridge mesh and verifies DEVSIM import.",
            "It is not a calibrated full product TCAD mesh and does not prove accuracy.",
        ],
    }


def tcad_dd_smoke_command(
    *,
    template: dict[str, Any],
    mesh_path: Path,
    config_path: Path,
    output_dir: Path,
    body: dict[str, Any],
    generation_map_npz: Path | None = None,
) -> list[str]:
    derived = read_json_artifact(config_path)
    mesh_config = derived.get("pixel_mesh_config", {}) if isinstance(derived.get("pixel_mesh_config"), dict) else {}
    params = template.get("parameters", {}) if isinstance(template.get("parameters"), dict) else {}
    width_um = json_safe_number(mesh_config.get("width_um")) or json_safe_number(params.get("pitch_um")) or 1.4
    depth_um = json_safe_number(mesh_config.get("depth_um")) or json_safe_number(params.get("si_thickness_um")) or 2.8
    split_gap_um = json_safe_number(mesh_config.get("split_gap_um")) or 0.04
    dti_width_um = json_safe_number(params.get("dti_width_um")) or 0.06
    python_bin = TCAD_PYTHON if TCAD_PYTHON.exists() else Path(sys.executable)
    command = [
        str(python_bin),
        str(ROOT / "devsim_split_pd_2d.py"),
        "--mesh-source",
        "gmsh",
        "--gmsh-mesh",
        str(mesh_path),
        "--width-um",
        f"{width_um:g}",
        "--depth-um",
        f"{depth_um:g}",
        "--split-gap-um",
        f"{split_gap_um:g}",
        "--dti-width-um",
        f"{dti_width_um:g}",
        "--photo-g0-cm3-s",
        str(body.get("photo_g0_cm3_s") or "1.0e18"),
        "--photo-shift-x-um",
        str(body.get("photo_shift_x_um") or "0.0"),
        "--generation-lateral-mode",
        str(body.get("generation_lateral_mode") or "uniform"),
        "--electrical-model",
        str(body.get("electrical_model") or "proxy-pinned-split-pd"),
        "--dd-max-iterations",
        str(body.get("dd_max_iterations") or "160"),
        "--dd-relative-error",
        str(body.get("dd_relative_error") or "1.0e-9"),
        "--dd-absolute-error",
        str(body.get("dd_absolute_error") or "1.0e10"),
        "--output-dir",
        str(output_dir),
    ]
    if generation_map_npz is not None:
        command.extend(
            [
                "--generation-map-npz",
                str(generation_map_npz),
                "--generation-profile-case",
                str(body.get("generation_profile_case") or "center"),
                "--generation-profile-wavelength-nm",
                str(body.get("generation_profile_wavelength_nm") or "550"),
                "--generation-map-scale",
                str(body.get("generation_map_scale") or "1.0"),
            ]
        )
        if body.get("disable_generation_map_normalization"):
            command.append("--disable-generation-map-normalization")
    measured_profile = body.get("measured_profile")
    if measured_profile:
        command.extend(["--measured-profile", str(measured_profile)])
    return command


def annotate_tcad_dd_summary(summary: dict[str, Any], config_path: Path, summary_path: Path | None = None) -> dict[str, Any]:
    derived_config = read_json_artifact(config_path)
    derivation = derived_config.get("derivation", {}) if isinstance(derived_config.get("derivation"), dict) else {}
    capability = derivation.get("electrical_capability", {}) if isinstance(derivation.get("electrical_capability"), dict) else {}
    solver_gate = devsim_dd_solver_gate(summary)
    represented_axis = capability.get("represented_split_axis")
    annotated = dict(summary)
    annotated["electrical_capability"] = capability
    annotated["capability_gate"] = capability.get("gate") or "PASS"
    annotated["solver_gate"] = solver_gate.get("gate")
    annotated["solver_gate_reason"] = solver_gate.get("reason")
    annotated["phase_result_axis"] = represented_axis
    annotated["phase_result_scope"] = capability.get("phase_result_scope")
    annotated["contact_axis_labels"] = capability.get("contact_axis_labels", {})
    if represented_axis == "z" and annotated.get("photo_split_phase_x_proxy") is not None:
        annotated["photo_split_phase_z_proxy"] = annotated.get("photo_split_phase_x_proxy")
        annotated["bottom_photo_delta_a_per_cm"] = annotated.get("left_photo_delta_a_per_cm")
        annotated["top_photo_delta_a_per_cm"] = annotated.get("right_photo_delta_a_per_cm")
        annotated.setdefault(
            "axis_note",
            "DEVSIM variable names remain x/left/right internally; this run maps the lateral solver axis to template z.",
        )
    elif represented_axis == "x" and annotated.get("photo_split_phase_x_proxy") is not None:
        annotated["left_photo_delta_a_per_cm"] = annotated.get("left_photo_delta_a_per_cm")
        annotated["right_photo_delta_a_per_cm"] = annotated.get("right_photo_delta_a_per_cm")
    if summary_path is not None and summary_path.exists():
        summary_path.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
    return annotated


def run_tcad_dd_smoke_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    template_dir = CAD_TEMPLATE_ROOT / template_id
    bridge_dir = template_dir / "tcad_bridge_2d"
    mesh_path = bridge_dir / "split_pixel_2d.msh"
    config_path = bridge_dir / "derived_tcad_config.json"
    generated_bridge = None
    if not mesh_path.exists() or not config_path.exists():
        generated_bridge = generate_tcad_bridge_from_request({"template_id": template_id})
    if not mesh_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"TCAD bridge mesh/config is missing for {template_id}")

    output_dir = bridge_dir / "devsim_smoke"
    command = tcad_dd_smoke_command(
        template=template,
        mesh_path=mesh_path,
        config_path=config_path,
        output_dir=output_dir,
        body=body,
    )
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
    summary = parse_json_from_mixed_stdout(completed.stdout)
    summary = annotate_tcad_dd_summary(summary, config_path, output_dir / "summary.json")
    capability = summary.get("electrical_capability", {}) if isinstance(summary.get("electrical_capability"), dict) else {}
    capability_gate = capability.get("gate") or "PASS"
    solver_gate = summary.get("solver_gate") or "CHECK"
    updated_catalog = load_cad_template_catalog()
    updated_template = find_cad_template(updated_catalog, template_id)
    raw_status = "PASS" if summary.get("status") in {None, "PASS"} and summary.get("photo_split_phase_x_proxy") is not None else "CHECK"
    status = "PASS" if raw_status == "PASS" and capability_gate == "PASS" and solver_gate == "PASS" else "CHECK"
    return {
        "schema": "pixel_workbench_tcad_dd_smoke_v1",
        "status": status,
        "raw_solver_status": raw_status,
        "capability_gate": capability_gate,
        "solver_gate": solver_gate,
        "electrical_capability": capability,
        "template_id": template_id,
        "template_label": template.get("label"),
        "generated_bridge": generated_bridge is not None,
        "summary": summary,
        "template": updated_template,
        "catalog": updated_catalog,
        "command": command,
        "stderr": completed.stderr[-4000:],
        "notes": [
            "This runs a DEVSIM drift-diffusion smoke solve on the CAD-template-derived 2D Gmsh mesh.",
            "The electrical model remains proxy/calibration-limited unless measured process and transport data are loaded.",
        ],
    }


def signal_sum(summary: dict[str, Any], axis: str) -> float | None:
    if axis == "z":
        first = json_safe_number(summary.get("bottom_photo_delta_a_per_cm"))
        second = json_safe_number(summary.get("top_photo_delta_a_per_cm"))
    else:
        first = json_safe_number(summary.get("left_photo_delta_a_per_cm"))
        second = json_safe_number(summary.get("right_photo_delta_a_per_cm"))
    if first is None or second is None:
        return None
    return abs(first) + abs(second)


def write_axis_pair_svg(report: dict[str, Any], output_path: Path) -> None:
    phase_x = json_safe_number(report.get("phase_x_proxy")) or 0.0
    phase_z = json_safe_number(report.get("phase_z_proxy")) or 0.0
    signal_x = json_safe_number(report.get("signal_sum_x_a_per_cm")) or 0.0
    signal_z = json_safe_number(report.get("signal_sum_z_a_per_cm")) or 0.0
    max_phase = max(abs(phase_x), abs(phase_z), 1.0e-9)
    max_signal = max(abs(signal_x), abs(signal_z), 1.0e-30)

    def bar(value: float, max_value: float, x: float, y: float, width: float, color: str) -> str:
        scaled = min(width, abs(value) / max_value * width)
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{scaled:.1f}" height="24" rx="4" fill="{color}"/>'

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="260" viewBox="0 0 760 260">',
        '<rect width="760" height="260" rx="10" fill="#07131f"/>',
        '<text x="28" y="34" fill="#e2e8f0" font-family="Inter, Arial" font-size="18" font-weight="700">QPD Axis-Pair DD Smoke</text>',
        f'<text x="28" y="58" fill="#94a3b8" font-family="Inter, Arial" font-size="12">phase magnitude {report.get("axis_phase_magnitude", 0):.5g} · full Q1-Q4 gate {report.get("full_q1q4_gate", "CHECK")}</text>',
        '<text x="28" y="98" fill="#cbd5e1" font-family="Inter, Arial" font-size="13">Phase X</text>',
        bar(phase_x, max_phase, 150, 78, 420, "#38bdf8"),
        f'<text x="590" y="96" fill="#e0f2fe" font-family="Inter, Arial" font-size="13">{phase_x:.6g}</text>',
        '<text x="28" y="134" fill="#cbd5e1" font-family="Inter, Arial" font-size="13">Phase Z</text>',
        bar(phase_z, max_phase, 150, 114, 420, "#22c55e"),
        f'<text x="590" y="132" fill="#dcfce7" font-family="Inter, Arial" font-size="13">{phase_z:.6g}</text>',
        '<text x="28" y="178" fill="#cbd5e1" font-family="Inter, Arial" font-size="13">Signal X</text>',
        bar(signal_x, max_signal, 150, 158, 420, "#818cf8"),
        f'<text x="590" y="176" fill="#e0e7ff" font-family="Inter, Arial" font-size="13">{signal_x:.4e}</text>',
        '<text x="28" y="214" fill="#cbd5e1" font-family="Inter, Arial" font-size="13">Signal Z</text>',
        bar(signal_z, max_signal, 150, 194, 420, "#f59e0b"),
        f'<text x="590" y="212" fill="#fef3c7" font-family="Inter, Arial" font-size="13">{signal_z:.4e}</text>',
        '<text x="28" y="242" fill="#94a3b8" font-family="Inter, Arial" font-size="11">Axis-pair projection only; full quadrant-resolved Q1-Q4 requires a coupled 3D solve.</text>',
        '</svg>',
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tcad_axis_pair_smoke_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    params = template.get("parameters", {}) if isinstance(template.get("parameters"), dict) else {}
    split_mode = str(params.get("split_mode") or "")
    template_dir = CAD_TEMPLATE_ROOT / template_id
    output_root = template_dir / "tcad_axis_pair_smoke"
    axes: dict[str, dict[str, Any]] = {}
    commands: dict[str, Any] = {}
    status = "PASS"

    for axis in ("x", "z"):
        axis_dir = output_root / f"axis_{axis}"
        bridge_body = {**body, "template_id": template_id, "section_axis": axis}
        bridge = generate_tcad_bridge_artifacts(template_id, bridge_body, axis_dir)
        mesh_path = axis_dir / "split_pixel_2d.msh"
        config_path = axis_dir / "derived_tcad_config.json"
        dd_dir = axis_dir / "devsim_smoke"
        dd_command = tcad_dd_smoke_command(
            template=template,
            mesh_path=mesh_path,
            config_path=config_path,
            output_dir=dd_dir,
            body=body,
        )
        completed = subprocess.run(dd_command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
        summary = annotate_tcad_dd_summary(parse_json_from_mixed_stdout(completed.stdout), config_path, dd_dir / "summary.json")
        raw_status = "PASS" if summary.get("status") in {None, "PASS"} and summary.get("photo_split_phase_x_proxy") is not None else "CHECK"
        axis_status = "PASS" if bridge.get("status") == "PASS" and raw_status == "PASS" else "CHECK"
        if axis_status != "PASS":
            status = "CHECK"
        axes[axis] = {
            "status": axis_status,
            "bridge_status": bridge.get("status"),
            "capability_gate": bridge.get("capability_gate"),
            "bridge_report": bridge.get("bridge_report"),
            "import_smoke": bridge.get("import_smoke"),
            "summary": summary,
            "artifacts": {
                "bridge_report": file_artifact_record(axis_dir / "tcad_bridge_report.json"),
                "mesh": file_artifact_record(mesh_path),
                "derived_config": file_artifact_record(config_path),
                "dd_summary": file_artifact_record(dd_dir / "summary.json"),
                "dd_plot": file_artifact_record(dd_dir / "split_currents.png"),
            },
        }
        commands[axis] = {
            "bridge": bridge.get("commands"),
            "dd": dd_command,
            "stderr": {
                "bridge": bridge.get("stderr"),
                "dd": completed.stderr[-4000:],
            },
        }

    x_summary = axes["x"]["summary"]
    z_summary = axes["z"]["summary"]
    phase_x = json_safe_number(x_summary.get("photo_split_phase_x_proxy"))
    phase_z = json_safe_number(z_summary.get("photo_split_phase_z_proxy") or z_summary.get("photo_split_phase_x_proxy"))
    signal_x = signal_sum(x_summary, "x")
    signal_z = signal_sum(z_summary, "z")
    max_signal = max(signal_x or 0.0, signal_z or 0.0)
    min_signal = min(signal_x or 0.0, signal_z or 0.0)
    signal_uniformity = (min_signal / max_signal) if max_signal else None
    phase_magnitude = math.sqrt((phase_x or 0.0) ** 2 + (phase_z or 0.0) ** 2)
    phase_angle = math.degrees(math.atan2(phase_z or 0.0, phase_x or 0.0)) if phase_x is not None and phase_z is not None else None
    report = {
        "schema": "pixel_workbench_qpd_axis_pair_smoke_v1",
        "status": status,
        "template_id": template_id,
        "template_label": template.get("label"),
        "split_mode": split_mode,
        "phase_x_proxy": phase_x,
        "phase_z_proxy": phase_z,
        "axis_phase_magnitude": phase_magnitude,
        "axis_phase_angle_deg": phase_angle,
        "signal_sum_x_a_per_cm": signal_x,
        "signal_sum_z_a_per_cm": signal_z,
        "axis_signal_uniformity": signal_uniformity,
        "full_q1q4_gate": "CHECK",
        "full_q1q4_ready": False,
        "axes": axes,
        "artifacts": {
            "summary": file_artifact_record(output_root / "summary.json"),
            "plot": file_artifact_record(output_root / "axis_pair_phase.svg"),
        },
        "notes": [
            "This is an axis-pair projection built from separate x-depth and z-depth 2D DD smoke solves.",
            "It improves QPD/PDAF axis visibility over x-only smoke, but it is not a full coupled 3D Q1-Q4 solve.",
            "Use this for trend checks; product accuracy still requires measured process data and calibrated 3D or validated equivalent modeling.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    plot_path = output_root / "axis_pair_phase.svg"
    report["artifacts"] = {
        "summary": file_artifact_record(summary_path),
        "plot": file_artifact_record(plot_path),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_axis_pair_svg(report, plot_path)
    report["artifacts"] = {
        "summary": file_artifact_record(summary_path),
        "plot": file_artifact_record(plot_path),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    updated_catalog = load_cad_template_catalog()
    return {
        "schema": "pixel_workbench_tcad_axis_pair_smoke_result_v1",
        "status": status,
        "template_id": template_id,
        "summary": report,
        "catalog": updated_catalog,
        "template": find_cad_template(updated_catalog, template_id),
        "commands": commands,
    }


def run_tcad_qpd_weighting_3d_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    params = template.get("parameters", {}) if isinstance(template.get("parameters"), dict) else {}
    split_mode = str(params.get("split_mode") or "")
    if split_mode != "quad":
        raise ValueError(f"3D QPD weighting requires split_mode=quad; {template_id} has split_mode={split_mode!r}")
    pitch_um = json_safe_number(params.get("pitch_um")) or 1.4
    nx = int(json_safe_number(params.get("nx")) or 2)
    nz = int(json_safe_number(params.get("nz")) or 2)
    width_um = json_safe_number(body.get("width_um")) or nx * pitch_um
    z_width_um = json_safe_number(body.get("z_width_um")) or nz * pitch_um
    depth_um = json_safe_number(body.get("depth_um")) or json_safe_number(params.get("si_thickness_um")) or 3.0
    split_gap_um = json_safe_number(body.get("split_gap_um")) or 0.04
    mesh_um = json_safe_number(body.get("mesh_um")) or 0.45
    fine_mesh_um = json_safe_number(body.get("fine_mesh_um")) or 0.28
    output_dir = CAD_TEMPLATE_ROOT / template_id / "tcad_qpd_weighting_3d"
    python_bin = TCAD_PYTHON if TCAD_PYTHON.exists() else Path(sys.executable)
    command = [
        str(python_bin),
        str(ROOT / "devsim_qpd_weighting_3d.py"),
        "--width-um",
        f"{width_um:g}",
        "--z-width-um",
        f"{z_width_um:g}",
        "--depth-um",
        f"{depth_um:g}",
        "--split-gap-um",
        f"{split_gap_um:g}",
        "--mesh-um",
        f"{mesh_um:g}",
        "--fine-mesh-um",
        f"{fine_mesh_um:g}",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
    summary = parse_json_from_mixed_stdout(completed.stdout)
    updated_catalog = load_cad_template_catalog()
    updated_template = find_cad_template(updated_catalog, template_id)
    return {
        "schema": "pixel_workbench_tcad_qpd_weighting_3d_result_v1",
        "status": summary.get("status") or "CHECK",
        "template_id": template_id,
        "summary": summary,
        "catalog": updated_catalog,
        "template": updated_template,
        "command": command,
        "stderr": completed.stderr[-4000:],
        "notes": [
            "This is a 3D QPD terminal weighting-potential smoke solve.",
            "It is not a full calibrated 3D drift-diffusion collection solve.",
        ],
    }


def resolve_qpd_generation_volume(template_id: str, body: dict[str, Any]) -> Path:
    explicit = body.get("generation_volume_npz") or body.get("generation_volume")
    if explicit:
        path = path_from_record(explicit)
        if path is None or not path.exists():
            raise FileNotFoundError(f"generation volume NPZ not found: {explicit}")
        return path
    template_specific_candidates = [
        CAD_TEMPLATE_ROOT / template_id / "fdtd_smoke" / "tcad_generation_volume_3d.npz",
        *sorted(
            (RUN_ROOT).glob(f"**/*{template_id}*/tcad_generation_volume_3d.npz"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        ),
    ]
    for candidate in template_specific_candidates:
        if candidate.exists():
            return candidate
    if not body.get("allow_shared_generation_volume"):
        raise FileNotFoundError(
            f"No template-specific 3D generation volume found for {template_id}. "
            "Run a QPD CAD template FDTD smoke first or pass generation_volume_npz. "
            "Set allow_shared_generation_volume=true only for an intentional control run."
        )
    shared_candidates = [
        *sorted(
            (RUN_ROOT).glob("**/*qpd*/tcad_generation_volume_3d.npz"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        ),
    ]
    for candidate in shared_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No 3D generation volume found for {template_id}. Run a QPD CAD template FDTD smoke first or pass generation_volume_npz."
    )


def run_qpd_gw_3d_from_request(body: dict[str, Any]) -> dict[str, Any]:
    template_id = str(body.get("template_id") or "")
    if not template_id:
        raise ValueError("template_id is required")
    catalog = load_cad_template_catalog()
    template = find_cad_template(catalog, template_id)
    params = template.get("parameters", {}) if isinstance(template.get("parameters"), dict) else {}
    split_mode = str(params.get("split_mode") or "")
    if split_mode != "quad":
        raise ValueError(f"3D QPD G*W requires split_mode=quad; {template_id} has split_mode={split_mode!r}")

    template_dir = CAD_TEMPLATE_ROOT / template_id
    weighting_dir = template_dir / "tcad_qpd_weighting_3d"
    weighting_csv = weighting_dir / "qpd_weighting_3d.csv"
    if bool(body.get("force_weighting")) or not weighting_csv.exists():
        run_tcad_qpd_weighting_3d_from_request(body)
    if not weighting_csv.exists():
        raise FileNotFoundError(f"QPD weighting CSV not found after weighting run: {weighting_csv}")

    generation_volume = resolve_qpd_generation_volume(template_id, body)
    output_dir = template_dir / "tcad_qpd_gw_3d"
    case = str(body.get("case") or "all")
    outside_mode = str(body.get("outside_mode") or "clip")
    integration_grid = str(body.get("integration_grid") or "generation")
    python_bin = TCAD_PYTHON if TCAD_PYTHON.exists() else Path(sys.executable)
    command = [
        str(python_bin),
        str(ROOT / "qpd_3d_gw_response.py"),
        "--weighting-csv",
        str(weighting_csv),
        "--generation-volume-npz",
        str(generation_volume),
        "--output-dir",
        str(output_dir),
        "--case",
        case,
        "--integration-grid",
        integration_grid,
        "--outside-mode",
        outside_mode,
    ]
    wavelength_nm = json_safe_number(body.get("wavelength_nm"))
    if wavelength_nm is not None:
        command.extend(["--wavelength-nm", f"{wavelength_nm:g}"])
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=300)
    summary = parse_json_from_mixed_stdout(completed.stdout)
    generation_gate = qpd_generation_volume_gate(template_id, summary)
    summary["generation_volume_gate"] = generation_gate.get("gate")
    summary["generation_volume_reason"] = generation_gate.get("reason")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    updated_catalog = load_cad_template_catalog()
    updated_template = find_cad_template(updated_catalog, template_id)
    return {
        "schema": "pixel_workbench_qpd_gw_3d_result_v1",
        "status": "PASS" if summary.get("status") == "PASS" and generation_gate.get("gate") == "PASS" else "CHECK",
        "template_id": template_id,
        "summary": summary,
        "catalog": updated_catalog,
        "template": updated_template,
        "command": command,
        "stderr": completed.stderr[-4000:],
        "notes": [
            "This is 3D FDTD generation multiplied by 3D Laplace terminal weighting.",
            "It is a Q1-Q4 response surrogate, not calibrated 3D drift-diffusion.",
        ],
    }


def run_coupled_tcad_dd_smoke_for_case(template_id: str, case_dir: Path) -> dict[str, Any]:
    generation_map = case_dir / "tcad_generation_map_2d.npz"
    if not generation_map.exists():
        return {
            "available": False,
            "status": "CHECK",
            "reason": "tcad_generation_map_2d.npz not found",
        }
    try:
        catalog = load_cad_template_catalog()
        template = find_cad_template(catalog, template_id)
        template_dir = CAD_TEMPLATE_ROOT / template_id
        bridge_dir = template_dir / "tcad_bridge_2d"
        mesh_path = bridge_dir / "split_pixel_2d.msh"
        config_path = bridge_dir / "derived_tcad_config.json"
        if not mesh_path.exists() or not config_path.exists():
            generate_tcad_bridge_from_request({"template_id": template_id})
        if not mesh_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"TCAD bridge mesh/config is missing for {template_id}")

        output_dir = case_dir / "tcad_dd_coupled_smoke"
        smoke_body = {
            "generation_profile_case": "center",
            "generation_profile_wavelength_nm": "550",
            "generation_map_scale": "1.0e-3",
            "dd_relative_error": "1.0e-5",
            "dd_max_iterations": "220",
        }
        command = tcad_dd_smoke_command(
            template=template,
            mesh_path=mesh_path,
            config_path=config_path,
            output_dir=output_dir,
            body=smoke_body,
            generation_map_npz=generation_map,
        )
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
        summary = parse_json_from_mixed_stdout(completed.stdout)
        return {
            "available": True,
            "status": "PASS" if summary.get("photo_split_phase_x_proxy") is not None else "CHECK",
            "generation_map_npz": rel_url(generation_map),
            "output_url": rel_url(output_dir),
            "summary_url": rel_url(output_dir / "summary.json"),
            "split_currents_csv": rel_url(output_dir / "split_currents.csv"),
            "split_currents_plot": rel_url(output_dir / "split_currents.png"),
            "node_maps_plot": rel_url(output_dir / "node_maps.png"),
            "mesh": str(mesh_path),
            "summary": {
                "schema": summary.get("schema"),
                "runtime_tier": "smoke",
                "coupling_mode": "scaled_fdtd_generation_map_2d",
                "generation_map_scale": json_safe_number(smoke_body["generation_map_scale"]),
                "dd_relative_error": json_safe_number(smoke_body["dd_relative_error"]),
                "dd_max_iterations": json_safe_number(smoke_body["dd_max_iterations"]),
                "mesh_source": summary.get("mesh_source"),
                "gmsh_mesh": summary.get("gmsh_mesh"),
                "node_count": summary.get("node_count"),
                "generation_source": summary.get("generation_source"),
                "generation_integral_summary": summary.get("generation_integral_summary"),
                "electrical_model": summary.get("config", {}).get("electrical_model")
                if isinstance(summary.get("config"), dict)
                else summary.get("electrical_model"),
                "photo_split_phase_x_proxy": summary.get("photo_split_phase_x_proxy"),
                "left_photo_delta_a_per_cm": summary.get("left_photo_delta_a_per_cm"),
                "right_photo_delta_a_per_cm": summary.get("right_photo_delta_a_per_cm"),
                "terminal_current_balance_illuminated_a_per_cm": summary.get(
                    "terminal_current_balance_illuminated_a_per_cm"
                ),
            },
            "stderr": completed.stderr[-4000:],
        }
    except Exception as error:  # noqa: BLE001 - suite KPI should expose coupling failure without hiding FDTD output.
        return {
            "available": False,
            "status": "FAIL",
            "generation_map_npz": rel_url(generation_map),
            "error": str(error),
        }


class WorkbenchHandler(SimpleHTTPRequestHandler):
    server_version = "PixelWorkbenchServer/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store" if self.path.startswith("/api/") else "no-cache")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib API
        self.send_response(204)
        self.end_headers()

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            return super().do_GET()
        try:
            if parsed.path == "/api/health":
                return self.send_json(
                    {
                        "ok": True,
                        "schema": "pixel_workbench_backend_health_v1",
                        "root": str(ROOT),
                        "meep_python": str(PYTHON),
                        "meep_python_exists": PYTHON.exists(),
                        "example_count": len(EXAMPLES),
                        "suite_count": len(test_suite_catalog()),
                        "cad_template_count": load_cad_template_catalog().get("template_count", 0),
                    }
                )
            if parsed.path == "/api/simulation/examples":
                return self.send_json({"examples": list(EXAMPLES.values())})
            if parsed.path == "/api/simulation/test-suite":
                suites = sorted(test_suite_catalog().values(), key=lambda item: item.get("priority", 999))
                return self.send_json({"schema": "pixel_workbench_test_suite_catalog_v1", "suites": suites})
            if parsed.path == "/api/cad/templates":
                return self.send_json(load_cad_template_catalog())
            if parsed.path == "/api/cad/tools":
                return self.send_json(
                    {
                        "schema": "pixel_workbench_cad_tools_v1",
                        "freecad": freecad_status(),
                        "allowed_override_fields": sorted(SCALAR_OVERRIDE_FIELDS),
                        "quick_override_fields": sorted(CAD_VARIANT_QUICK_FIELDS & SCALAR_OVERRIDE_FIELDS),
                        "requires_new_base_template_fields": sorted(CAD_TOPOLOGY_CHANGE_FIELDS),
                        "metadata_not_variant_fields": sorted(CAD_METADATA_CHANGE_FIELDS),
                        "base_template_topology_presets": [
                            {"id": key, **value}
                            for key, value in sorted(CAD_BASE_TEMPLATE_TOPOLOGY_PRESETS.items())
                        ],
                        "pitch_variant_policy": "conditional_scalar_variant",
                        "notes": [
                            "pitch_um may be a scalar variant only when topology stays fixed and solver artifacts are regenerated.",
                            "nx/nz, OCL blocks, CFA pattern, split mode, and shield mode define topology and require a new base template.",
                        ],
                    }
                )
            if parsed.path == "/api/simulation/jobs":
                return self.send_json({"jobs": list_jobs()})
            if parsed.path.startswith("/api/simulation/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                if job_id not in JOBS:
                    return self.send_json({"error": "job not found", "job_id": job_id}, status=404)
                return self.send_json(snapshot_job(job_id))
            if parsed.path == "/api/simulation/latest":
                query = parse_qs(parsed.query)
                example_id = query.get("example_id", [""])[0]
                jobs = [
                    job
                    for job in list_jobs()
                    if not example_id or job.get("example", {}).get("id") == example_id
                ]
                return self.send_json(jobs[0] if jobs else {"job": None})
            return self.send_json({"error": "unknown endpoint", "path": parsed.path}, status=404)
        except Exception as error:  # noqa: BLE001 - local diagnostics endpoint.
            return self.send_json({"error": str(error)}, status=500)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        try:
            body = self.read_json_body()
            if parsed.path == "/api/simulation/run":
                if isinstance(body.get("simulation_request"), dict):
                    return self.send_json(create_request_job(body["simulation_request"]), status=202)
                example_id = str(body.get("example_id") or "ocl2x2_smoke")
                return self.send_json(create_job(example_id), status=202)
            if parsed.path == "/api/simulation/resolve-request":
                if not isinstance(body.get("simulation_request"), dict):
                    raise ValueError("simulation_request is required")
                return self.send_json(
                    {
                        "schema": "pixel_workbench_resolved_simulation_request_v1",
                        "solver_case": solver_case_from_request(body["simulation_request"]),
                    }
                )
            if parsed.path == "/api/simulation/run-suite":
                suite_id = str(body.get("suite_id") or "pattern_baseline")
                tier = str(body.get("tier") or "smoke")
                raw_case_ids = body.get("case_ids")
                case_ids = [str(item) for item in raw_case_ids] if isinstance(raw_case_ids, list) else None
                return self.send_json(create_suite_job(suite_id, tier=tier, case_ids=case_ids), status=202)
            if parsed.path == "/api/simulation/replay-case":
                return self.send_json(replay_case_from_request(body), status=201)
            if parsed.path == "/api/simulation/replay-case-job":
                return self.send_json(create_replay_job(body), status=202)
            if parsed.path == "/api/simulation/export-camera-package":
                return self.send_json(camera_system_export_from_request(body), status=201)
            if parsed.path == "/api/simulation/validate-camera-package":
                return self.send_json(camera_system_validate_from_request(body), status=201)
            if parsed.path == "/api/simulation/quantitative-evidence":
                return self.send_json(quantitative_evidence_from_request(body), status=201)
            if parsed.path == "/api/cad/open":
                template_id = str(body.get("template_id") or "")
                artifact = str(body.get("artifact") or "step")
                prefer_freecad = bool(body.get("prefer_freecad", True))
                if not template_id:
                    raise ValueError("template_id is required")
                return self.send_json(open_cad_artifact(template_id, artifact, prefer_freecad=prefer_freecad))
            if parsed.path == "/api/cad/open-file":
                return self.send_json(open_workspace_file(str(body.get("path") or ""), prefer_freecad=bool(body.get("prefer_freecad", True))))
            if parsed.path == "/api/cad/create-fcstd-working-copy":
                return self.send_json(create_fcstd_working_copy_from_request(body), status=201)
            if parsed.path == "/api/cad/create-variant":
                return self.send_json(create_cad_variant_from_request(body), status=201)
            if parsed.path == "/api/cad/create-base-template":
                return self.send_json(create_cad_base_template_from_request(body), status=201)
            if parsed.path == "/api/cad/extract-fcstd-parameters":
                return self.send_json(extract_fcstd_parameters_from_request(body), status=200)
            if parsed.path == "/api/cad/create-variant-from-fcstd":
                return self.send_json(create_cad_variant_from_fcstd_request(body), status=201)
            if parsed.path == "/api/cad/validate-freecad":
                return self.send_json(validate_freecad_from_request(body), status=201)
            if parsed.path == "/api/cad/generate-tcad-bridge":
                return self.send_json(generate_tcad_bridge_from_request(body), status=201)
            if parsed.path == "/api/cad/run-tcad-dd-smoke":
                return self.send_json(run_tcad_dd_smoke_from_request(body), status=201)
            if parsed.path == "/api/cad/run-tcad-axis-pair-smoke":
                return self.send_json(run_tcad_axis_pair_smoke_from_request(body), status=201)
            if parsed.path == "/api/cad/run-tcad-qpd-weighting-3d":
                return self.send_json(run_tcad_qpd_weighting_3d_from_request(body), status=201)
            if parsed.path == "/api/cad/run-qpd-gw-3d":
                return self.send_json(run_qpd_gw_3d_from_request(body), status=201)
            return self.send_json({"error": "unknown endpoint", "path": parsed.path}, status=404)
        except KeyError as error:
            return self.send_json({"error": str(error)}, status=400)
        except ValueError as error:
            return self.send_json({"error": str(error)}, status=400)
        except Exception as error:  # noqa: BLE001 - local diagnostics endpoint.
            return self.send_json({"error": str(error)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--directory", type=Path, default=ROOT)
    args = parser.parse_args()
    handler = partial(WorkbenchHandler, directory=str(args.directory))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Pixel Workbench backend: http://{args.host}:{args.port}")
    print(f"Studio URL: http://{args.host}:{args.port}/runs/image_sensor_pixel_studio_reference/index.html")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
