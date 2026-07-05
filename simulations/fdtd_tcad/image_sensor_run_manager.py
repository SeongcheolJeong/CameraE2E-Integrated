#!/usr/bin/env python3
"""Build Lumerical-style run and dataset manager artifacts.

This script inspects existing local outputs. It does not launch Meep, Gmsh, or
DEVSIM jobs. The goal is to make planned, completed, partial, and missing stages
visible in the Studio, and to expose generated files as structured datasets.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from image_sensor_pixel_studio import build_payload


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_CONFIG = ROOT / "configs" / "image_sensor_pixel_studio_reference.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "image_sensor_design_variants_reference"
DEFAULT_STUDIO_OUTPUT_DIR = ROOT / "runs" / "image_sensor_pixel_studio_reference"
DEFAULT_VARIANT_MANIFEST = DEFAULT_OUTPUT_DIR / "variant_run_manifest.json"

STAGE_ORDER = [
    "meep_fdtd",
    "convergence_gate",
    "gmsh_mesh",
    "devsim_weighting",
    "devsim_electrical",
    "devsim_native_response_sweep",
    "design_viewer",
    "gw_lut",
    "studio",
]

STAGE_EXPECTATIONS: dict[str, list[tuple[str, str, str]]] = {
    "meep_fdtd": [
        ("generation_map_2d", "fdtd_generation", "tcad_generation_map_2d.npz"),
        ("generation_volume_3d", "fdtd_generation", "tcad_generation_volume_3d.npz"),
    ],
    "convergence_gate": [
        ("convergence_report", "convergence", "convergence_report.json"),
    ],
    "gmsh_mesh": [
        ("split_pixel_mesh", "gmsh_mesh", "split_pixel_2d.msh"),
    ],
    "devsim_weighting": [
        ("weighting_summary", "devsim_weighting", "weighting_potential_2d_summary.json"),
        ("weighting_csv", "devsim_weighting", "weighting_potential_2d.csv"),
    ],
    "devsim_electrical": [
        ("center_summary", "devsim_center", "summary.json"),
        ("cra10x_summary", "devsim_cra10x", "summary.json"),
        ("edge20x_summary", "devsim_edge20x", "summary.json"),
    ],
    "devsim_native_response_sweep": [
        ("native_sweep_manifest", "devsim_native_response_sweep", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "devsim_native_response_sweep", "native_response_sweep_summary.csv"),
        ("split_summary_list", "devsim_native_response_sweep", "split_summaries.json"),
    ],
    "design_viewer": [
        ("viewer_manifest", "design_viewer", "manifest.json"),
        ("cross_section_2d", "design_viewer", "viewers/cross_section_2d.html"),
        ("geometry_3d", "design_viewer", "viewers/geometry_3d.html"),
    ],
    "gw_lut": [
        ("gw_manifest", "gw_coupling", "gw_coupling_manifest.json"),
        ("camera_diagnostic_json", "gw_coupling", "camera_system_diagnostic.json"),
        ("native_devsim_response_json", "gw_coupling", "camera_system_native_devsim_response.json"),
        ("research_lut_json", "gw_coupling", "camera_system_research_lut.json"),
        ("research_lut_npz", "gw_coupling", "camera_system_research_lut.npz"),
    ],
    "studio": [
        ("studio_manifest", "studio", "studio_manifest.json"),
        ("studio_index", "studio", "index.html"),
    ],
}

PROJECT_RUNBOOK_EXPECTED_FILES: dict[str, list[tuple[str, str]]] = {
    "mesh_native_split_2d": [("split_pixel_mesh", "split_pixel_2d.msh")],
    "mesh_resolved_dti_2d": [
        ("resolved_dti_mesh", "split_pixel_2d.msh"),
        ("mesh_metadata", "mesh_metadata.json"),
    ],
    "devsim_center_native": [("center_summary", "summary.json")],
    "devsim_edge_native": [("edge_summary", "summary.json")],
    "devsim_resolved_dti_center": [
        ("resolved_dti_summary", "summary.json"),
        ("resolved_dti_currents", "split_currents.csv"),
        ("resolved_dti_node_profile", "node_profile_2d.csv"),
    ],
    "devsim_native_response_sweep": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_convergence": [
        ("native_response_convergence_report", "native_response_convergence_report.json"),
        ("native_response_convergence_summary", "native_response_convergence_summary.csv"),
    ],
    "devsim_tg_fd_transfer_sweep": [
        ("tg_fd_transfer_sweep_report", "tg_fd_transfer_sweep_report.json"),
        ("tg_fd_transfer_sweep_summary", "tg_fd_transfer_sweep_summary.csv"),
    ],
    "devsim_tg_fd_transient": [
        ("tg_fd_transient_report", "tg_fd_transient_report.json"),
        ("tg_fd_transient_timeseries", "tg_fd_transient_timeseries.csv"),
    ],
    "devsim_weighting_native": [
        ("weighting_summary", "weighting_potential_2d_summary.json"),
        ("weighting_csv", "weighting_potential_2d.csv"),
    ],
    "devsim_dd_probe_response": [
        ("dd_probe_summary", "dd_probe_response_2d_summary.json"),
        ("dd_probe_csv", "dd_probe_response_2d.csv"),
    ],
    "gw_coupling_report": [
        ("gw_manifest", "gw_coupling_manifest.json"),
        ("camera_diagnostic_json", "camera_system_diagnostic.json"),
        ("native_devsim_response_json", "camera_system_native_devsim_response.json"),
        ("native_devsim_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "optical_stack_evidence": [
        ("optical_stack_summary", "optical_stack_summary.json"),
        ("optical_stack_materials", "optical_stack_materials.csv"),
    ],
    "optical_nk_interpolation_check": [
        ("optical_nk_interpolation_check_json", "optical_nk_interpolation_check.json"),
    ],
    "fdtd_crosstalk_kernel_baseline": [
        ("crosstalk_manifest", "crosstalk_kernel.json"),
        ("crosstalk_convergence", "crosstalk_convergence.json"),
        ("crosstalk_summary", "crosstalk_kernel_summary.csv"),
    ],
    "fdtd_crosstalk_kernel_split_support": [
        ("crosstalk_manifest", "crosstalk_kernel.json"),
        ("crosstalk_convergence", "crosstalk_convergence.json"),
        ("crosstalk_summary", "crosstalk_kernel_summary.csv"),
    ],
    "fdtd_crosstalk_kernel_ocl_resolution": [
        ("crosstalk_manifest", "crosstalk_kernel.json"),
        ("crosstalk_convergence", "crosstalk_convergence.json"),
        ("crosstalk_summary", "crosstalk_kernel_summary.csv"),
    ],
    "fdtd_crosstalk_kernel_refresh": [
        ("crosstalk_manifest", "crosstalk_kernel.json"),
        ("crosstalk_convergence", "crosstalk_convergence.json"),
        ("crosstalk_summary", "crosstalk_kernel_summary.csv"),
    ],
    "fdtd_crosstalk_xsection_split": [
        ("xsection_manifest", "crosstalk_xsection_kernel.json"),
        ("xsection_convergence", "crosstalk_xsection_convergence.json"),
        ("xsection_summary", "crosstalk_xsection_summary.csv"),
    ],
    "fdtd_crosstalk_xsection_ocl2": [
        ("xsection_manifest", "crosstalk_xsection_kernel.json"),
        ("xsection_convergence", "crosstalk_xsection_convergence.json"),
        ("xsection_summary", "crosstalk_xsection_summary.csv"),
    ],
    "fdtd_crosstalk_xsection_ocl3": [
        ("xsection_manifest", "crosstalk_xsection_kernel.json"),
        ("xsection_convergence", "crosstalk_xsection_convergence.json"),
        ("xsection_summary", "crosstalk_xsection_summary.csv"),
    ],
    "fdtd_crosstalk_xsection_merge": [
        ("xsection_manifest", "crosstalk_xsection_kernel.json"),
        ("xsection_convergence", "crosstalk_xsection_convergence.json"),
        ("xsection_summary", "crosstalk_xsection_summary.csv"),
    ],
    "fdtd_cra5_convergence_r15_r20": [
        ("convergence_report", "convergence_report.json"),
        ("convergence_results", "convergence_results.csv"),
    ],
    "fdtd_center_grid_gate_r20_r25": [
        ("convergence_report", "convergence_report.json"),
        ("convergence_results", "convergence_results.csv"),
    ],
    "fdtd_center_3d_r60_quant_smoke": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("generation_map_2d", "tcad_generation_map_2d.npz"),
        ("generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_edge20x_3d_r60_quant_smoke": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("generation_map_2d", "tcad_generation_map_2d.npz"),
        ("generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_cra10x_3d_r60_quant_smoke": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("generation_map_2d", "tcad_generation_map_2d.npz"),
        ("generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_cra3_convergence_r60_r70": [
        ("convergence_report", "convergence_report.json"),
        ("convergence_results", "convergence_results.csv"),
        ("r70_camera_lut_summary", "r70_t8_pml0p45/camera_lut_summary.csv"),
        ("r70_generation_map_2d", "r70_t8_pml0p45/tcad_generation_map_2d.npz"),
        ("r70_generation_volume_3d", "r70_t8_pml0p45/tcad_generation_volume_3d.npz"),
    ],
    "fdtd_cra3_convergence_r60_r70_r80": [
        ("convergence_report", "convergence_report.json"),
        ("convergence_results", "convergence_results.csv"),
        ("r80_camera_lut_summary", "r80_t8_pml0p45/camera_lut_summary.csv"),
        ("r80_generation_map_2d", "r80_t8_pml0p45/tcad_generation_map_2d.npz"),
        ("r80_generation_volume_3d", "r80_t8_pml0p45/tcad_generation_volume_3d.npz"),
    ],
    "fdtd_cra3_r80_time_convergence": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("generation_map_2d", "tcad_generation_map_2d.npz"),
        ("generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_cra3_r80_pml_convergence": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("generation_map_2d", "tcad_generation_map_2d.npz"),
        ("generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_cra3_full_axes_convergence_report": [
        ("convergence_report", "convergence_report.json"),
        ("convergence_results", "convergence_results.csv"),
    ],
    "devsim_native_response_sweep_center_r60": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra10x_r60": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_edge20x_r60": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra3_r70": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra3_r80": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra3_r80_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra3_r80_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "fdtd_supercell_cra3z_r80_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_supercell_cra_diag_r80_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_supercell_cra_negative_r80_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_supercell_cra3_rgb_r84_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("convergence_report", "convergence_report.json"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_supercell_cra3z_rgb_r84_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_supercell_cra_diag_rgb_r84_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "fdtd_supercell_cra_negative_r84_gridsnap": [
        ("camera_lut_json", "camera_lut.json"),
        ("camera_lut_summary", "camera_lut_summary.csv"),
        ("tcad_generation_map_2d", "tcad_generation_map_2d.npz"),
        ("tcad_generation_profile_1d", "tcad_generation_profile_1d.npz"),
        ("tcad_generation_volume_3d", "tcad_generation_volume_3d.npz"),
    ],
    "devsim_native_response_sweep_cra3_rgb_r84_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra3z_rgb_r84_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra_diag_rgb_r84_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra_negative_r84_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra3z_r80_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra_diag_r80_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "devsim_native_response_sweep_cra_negative_r80_resolved_dti_pd_only": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "native_response_compare_cra3_r80_resolved_dti_pd_only": [
        ("native_response_compare_json", "native_response_compare.json"),
        ("native_response_compare_csv", "native_response_compare.csv"),
        ("native_response_compare_html", "native_response_compare.html"),
    ],
    "native_devsim_research_lut_cra2_r60": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3_r60": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3_r70": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3_r80": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3_r80_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3_rgb_r84_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3z_rgb_r84_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra_diag_rgb_r84_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra3z_r80_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra_diag_r80_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "native_devsim_research_lut_cra_negative_r80_resolved_dti_pd_only": [
        ("native_lut_manifest", "native_devsim_research_lut_manifest.json"),
        ("native_response_json", "camera_system_native_devsim_response.json"),
        ("native_response_npz", "camera_system_native_devsim_response.npz"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "camera_system_uncertainty_lut": [
        ("uncertainty_lut_json", "camera_system_uncertainty_lut.json"),
        ("uncertainty_lut_csv", "camera_system_uncertainty_lut.csv"),
        ("uncertainty_lut_html", "camera_system_uncertainty_lut.html"),
        ("field_lut_json", "camera_system_field_lut.json"),
        ("field_lut_csv", "camera_system_field_lut.csv"),
        ("field_lut_html", "camera_system_field_lut.html"),
        ("field_lut_npz", "camera_system_field_lut.npz"),
    ],
    "camera_system_field_lut_query": [
        ("field_lut_query_json", "field_lut_query.json"),
        ("field_lut_query_csv", "field_lut_query.csv"),
    ],
    "signed_field_symmetry_validation": [
        ("signed_field_symmetry_validation_json", "signed_field_symmetry_validation.json"),
        ("signed_field_symmetry_validation_csv", "signed_field_symmetry_validation.csv"),
        ("signed_field_symmetry_validation_html", "signed_field_symmetry_validation.html"),
    ],
    "camera_lut_spectral_coverage": [
        ("camera_lut_spectral_coverage_json", "camera_lut_spectral_coverage.json"),
        ("camera_lut_spectral_coverage_csv", "camera_lut_spectral_coverage.csv"),
        ("camera_lut_spectral_coverage_html", "camera_lut_spectral_coverage.html"),
    ],
    "measured_profile_interpolation_check": [
        ("interpolation_check_json", "interpolation_check.json"),
    ],
    "tcad_calibration_target_report": [
        ("calibration_target_report_json", "calibration_target_report.json"),
        ("calibration_target_report_csv", "calibration_target_report.csv"),
        ("calibration_target_metric_csv", "calibration_target_metric_residuals.csv"),
    ],
    "tcad_calibration_transport_reference": [
        ("calibration_result_json", "calibration_result.json"),
        ("calibration_history_csv", "calibration_history.csv"),
    ],
    "tcad_calibration_transport_target_report": [
        ("calibration_target_report_json", "calibration_target_report.json"),
        ("calibration_target_report_csv", "calibration_target_report.csv"),
        ("calibration_target_metric_csv", "calibration_target_metric_residuals.csv"),
    ],
    "tcad_transport_sensitivity": [
        ("transport_sensitivity_json", "transport_sensitivity_report.json"),
        ("transport_sensitivity_csv", "transport_sensitivity_report.csv"),
    ],
    "devsim_native_response_sweep_cra5_r20_gridsnap": [
        ("native_sweep_manifest", "native_response_sweep_manifest.json"),
        ("native_sweep_summary", "native_response_sweep_summary.csv"),
        ("split_summary_list", "split_summaries.json"),
    ],
    "gw_coupling_cra5_r20_gridsnap": [
        ("gw_manifest", "gw_coupling_manifest.json"),
        ("camera_diagnostic_json", "camera_system_diagnostic.json"),
        ("native_devsim_response_json", "camera_system_native_devsim_response.json"),
        ("research_lut_json", "camera_system_research_lut.json"),
        ("research_lut_npz", "camera_system_research_lut.npz"),
        ("product_lut_block_json", "camera_system_lut.json"),
    ],
    "accuracy_gate_reference": [("accuracy_gate", "tcad_accuracy_gate.json")],
    "variant_run_plans": [("variant_manifest", "variant_run_manifest.json")],
    "variant_compare": [
        ("variant_comparison_json", "variant_comparison.json"),
        ("variant_comparison_csv", "variant_comparison.csv"),
    ],
    "run_manager_status": [
        ("run_manager_json", "run_manager_status.json"),
        ("dataset_catalog_json", "dataset_catalog.json"),
    ],
    "studio_export": [
        ("studio_manifest", "studio_manifest.json"),
        ("studio_index", "index.html"),
    ],
}

RUN_COLUMNS = [
    "variant_id",
    "variant_label",
    "stage",
    "stage_index",
    "status",
    "completed_outputs",
    "expected_outputs",
    "missing_outputs",
    "blocked_by_missing_upstream",
    "freshness",
    "stale_reason",
    "newest_input_mtime",
    "oldest_output_mtime",
    "freshness_inputs",
    "command_count",
    "first_command",
    "product_lut_ready",
]

DATASET_COLUMNS = [
    "dataset_id",
    "solver",
    "dataset_kind",
    "role",
    "dimensionality",
    "native_mesh",
    "exists",
    "size_bytes",
    "path",
    "viewer",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rel_from_root(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_path(value: str | Path | None, base_dir: Path = ROOT) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (ROOT / path).resolve()


def path_status(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False, "size_bytes": None}
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def mtime_iso(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def expected_stage_paths(variant: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    planned_outputs = variant.get("planned_outputs", {})
    rows: list[dict[str, Any]] = []
    for label, output_key, relative_file in STAGE_EXPECTATIONS.get(stage, []):
        output_dir = planned_outputs.get(output_key)
        path = normalize_path(Path(output_dir) / relative_file) if output_dir else None
        rows.append({"label": label, **path_status(path)})
    return rows


def commands_for_stage(variant: dict[str, Any], stage: str) -> list[dict[str, str]]:
    return [item for item in variant.get("commands", []) if item.get("stage") == stage]


def stage_status(expected: list[dict[str, Any]]) -> str:
    if not expected:
        return "reference"
    existing = [item for item in expected if item["exists"]]
    if len(existing) == len(expected):
        return "complete"
    if existing:
        return "partial"
    return "missing"


def command_input_paths(commands: list[dict[str, str]]) -> list[Path]:
    output_flags = {
        "--output-dir",
        "--output",
        "--output-json",
        "--output-csv",
        "--output-html",
        "--output-path",
        "--log-dir",
    }
    paths: list[Path] = []
    seen: set[str] = set()
    for command in commands:
        try:
            tokens = shlex.split(command.get("command", ""))
        except ValueError:
            tokens = command.get("command", "").split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not token.startswith("--"):
                index += 1
                continue
            flag = token.split("=", 1)[0]
            values: list[str] = []
            if "=" in token:
                values.append(token.split("=", 1)[1])
                index += 1
            else:
                index += 1
                while index < len(tokens) and not tokens[index].startswith("--"):
                    values.append(tokens[index])
                    index += 1
            if flag in output_flags:
                continue
            for value in values:
                path = normalize_path(value)
                if not path or not path.exists() or not path.is_file():
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    paths.append(path.resolve())
    return paths


def command_output_dir(command: str) -> Path | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if token.startswith("--output-dir="):
            return normalize_path(token.split("=", 1)[1])
        if token == "--output-dir" and index + 1 < len(tokens):
            return normalize_path(tokens[index + 1])
    return None


def project_view_output_dir(project: dict[str, Any], command_id: str, config_dir: Path) -> Path | None:
    views = project.get("views", {})
    inputs = project.get("inputs", {})
    if command_id == "design_viewer_export":
        manifest = normalize_path(inputs.get("design_viewer_manifest"), config_dir)
        return manifest.parent if manifest else None
    if command_id == "gw_coupling_report":
        legacy_lut = normalize_path(views.get("camera_system_research_lut_reference"), config_dir)
        if legacy_lut:
            return legacy_lut.parent
        manifest = normalize_path(views.get("gw_coupling_manifest"), config_dir)
        return manifest.parent if manifest else None
    if command_id.startswith("fdtd_crosstalk_kernel"):
        manifest = normalize_path(views.get("crosstalk_kernel_manifest"), config_dir)
        return manifest.parent if manifest else None
    if command_id == "fdtd_crosstalk_xsection_merge":
        manifest = normalize_path(views.get("crosstalk_xsection_manifest"), config_dir)
        return manifest.parent if manifest else None
    if command_id == "variant_run_plans":
        manifest = normalize_path(views.get("variant_run_manifest"), config_dir)
        return manifest.parent if manifest else None
    if command_id == "variant_compare":
        report = normalize_path(views.get("variant_comparison_report"), config_dir)
        return report.parent if report else None
    if command_id == "run_manager_status":
        report = normalize_path(views.get("run_manager_report"), config_dir)
        return report.parent if report else None
    if command_id == "studio_export":
        return DEFAULT_STUDIO_OUTPUT_DIR
    return None


def project_expected_paths(project: dict[str, Any], config_dir: Path, command_row: dict[str, str]) -> list[dict[str, Any]]:
    command_id = str(command_row.get("id", ""))
    command = str(command_row.get("command", ""))
    if command_id == "design_viewer_export":
        expected = [
            ("viewer_manifest", normalize_path(project.get("inputs", {}).get("design_viewer_manifest"), config_dir)),
            ("cross_section_2d", normalize_path(project.get("views", {}).get("cross_section_2d"), config_dir)),
            ("geometry_3d", normalize_path(project.get("views", {}).get("geometry_3d"), config_dir)),
        ]
        return [{"label": label, **path_status(path)} for label, path in expected]

    if command_id == "accuracy_gate_reference":
        path = normalize_path(project.get("inputs", {}).get("accuracy_gate"), config_dir)
        return [{"label": "accuracy_gate", **path_status(path)}]

    output_dir = command_output_dir(command) or project_view_output_dir(project, command_id, config_dir)
    files = PROJECT_RUNBOOK_EXPECTED_FILES.get(command_id, [])
    if not output_dir or not files:
        return []
    return [{"label": label, **path_status(output_dir / relative_file)} for label, relative_file in files]


def freshness_inputs_for_stage(
    variant: dict[str, Any],
    stage: str,
    commands: list[dict[str, str]],
    required_stages: list[str],
) -> list[Path]:
    inputs = command_input_paths(commands)
    if stage == "studio":
        stage_index = required_stages.index(stage)
        for upstream_stage in required_stages[:stage_index]:
            for output in expected_stage_paths(variant, upstream_stage):
                path = Path(output["path"]) if output.get("path") else None
                if path and path.exists() and path.is_file():
                    inputs.append(path.resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in inputs:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    return unique


def stage_freshness(expected: list[dict[str, Any]], inputs: list[Path], status: str) -> dict[str, str]:
    if status == "reference":
        return {
            "freshness": "reference",
            "stale_reason": "no materialized outputs are expected for this runbook row",
            "newest_input_mtime": "",
            "oldest_output_mtime": "",
            "freshness_inputs": "",
        }
    if status != "complete":
        return {
            "freshness": "missing",
            "stale_reason": "stage outputs are not complete",
            "newest_input_mtime": "",
            "oldest_output_mtime": "",
            "freshness_inputs": ",".join(rel_from_root(path) for path in inputs),
        }
    output_paths = [Path(item["path"]) for item in expected if item.get("exists") and item.get("path")]
    output_files = [path for path in output_paths if path.exists() and path.is_file()]
    input_files = [path for path in inputs if path.exists() and path.is_file()]
    if not output_files:
        return {
            "freshness": "unknown",
            "stale_reason": "no tracked output files",
            "newest_input_mtime": "",
            "oldest_output_mtime": "",
            "freshness_inputs": ",".join(rel_from_root(path) for path in input_files),
        }
    if not input_files:
        return {
            "freshness": "fresh",
            "stale_reason": "no tracked input files",
            "newest_input_mtime": "",
            "oldest_output_mtime": mtime_iso(min(output_files, key=lambda path: path.stat().st_mtime)),
            "freshness_inputs": "",
        }
    newest_input = max(input_files, key=lambda path: path.stat().st_mtime)
    oldest_output = min(output_files, key=lambda path: path.stat().st_mtime)
    newest_input_mtime = newest_input.stat().st_mtime
    oldest_output_mtime = oldest_output.stat().st_mtime
    if newest_input_mtime > oldest_output_mtime + 1.0:
        freshness = "stale"
        reason = f"newer input: {rel_from_root(newest_input)}"
    else:
        freshness = "fresh"
        reason = "outputs are newer than tracked inputs"
    return {
        "freshness": freshness,
        "stale_reason": reason,
        "newest_input_mtime": mtime_iso(newest_input),
        "oldest_output_mtime": mtime_iso(oldest_output),
        "freshness_inputs": ",".join(rel_from_root(path) for path in input_files),
    }


def build_project_run_rows(project_config: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    project = read_json(project_config)
    config_dir = project_config.parent
    rows: list[dict[str, str]] = []
    complete_count = 0
    partial_count = 0
    missing_count = 0
    fresh_count = 0
    stale_count = 0
    for index, command_row in enumerate(project.get("runbook", [])):
        command_id = str(command_row.get("id", f"runbook_{index}"))
        expected = project_expected_paths(project, config_dir, command_row)
        status = stage_status(expected)
        commands = [command_row] if command_row.get("command") else []
        freshness = stage_freshness(expected, command_input_paths(commands), status)
        if status == "complete":
            complete_count += 1
        elif status == "partial":
            partial_count += 1
        else:
            missing_count += 1
        if freshness["freshness"] == "fresh":
            fresh_count += 1
        elif freshness["freshness"] == "stale":
            stale_count += 1
        rows.append(
            {
                "variant_id": "reference_project",
                "variant_label": str(project.get("project", {}).get("name", "Reference Project")),
                "stage": command_id,
                "stage_index": str(index),
                "status": status,
                "completed_outputs": ",".join(item["label"] for item in expected if item["exists"]),
                "expected_outputs": ",".join(item["label"] for item in expected),
                "missing_outputs": ",".join(item["label"] for item in expected if not item["exists"]),
                "blocked_by_missing_upstream": "",
                **freshness,
                "command_count": str(len(commands)),
                "first_command": commands[0].get("command", "") if commands else "",
                "product_lut_ready": "false",
            }
        )
    if missing_count == 0 and partial_count == 0:
        state = "stale" if stale_count else "complete"
    elif complete_count or partial_count:
        state = "partial"
    else:
        state = "planned_only"
    summary = {
        "id": "reference_project",
        "label": str(project.get("project", {}).get("name", "Reference Project")),
        "state": state,
        "complete_stage_count": complete_count,
        "missing_stage_count": missing_count,
        "partial_stage_count": partial_count,
        "blocked_stage_count": 0,
        "fresh_stage_count": fresh_count,
        "stale_stage_count": stale_count,
        "product_lut_ready": False,
    }
    return rows, summary


def build_run_rows(variant_manifest_path: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(variant_manifest_path)
    rows: list[dict[str, str]] = []
    variant_summaries: list[dict[str, Any]] = []
    for variant in manifest.get("variants", []):
        required = [stage for stage in STAGE_ORDER if stage in set(variant.get("required_stages", []))]
        if not required:
            variant_summaries.append(
                {
                    "id": variant.get("id"),
                    "label": variant.get("label", variant.get("id")),
                    "state": "reference",
                    "complete_stage_count": 0,
                    "missing_stage_count": 0,
                    "partial_stage_count": 0,
                    "blocked_stage_count": 0,
                    "fresh_stage_count": 0,
                    "stale_stage_count": 0,
                    "product_lut_ready": False,
                }
            )
            continue

        missing_upstream: list[str] = []
        complete_count = 0
        partial_count = 0
        missing_count = 0
        blocked_count = 0
        fresh_count = 0
        stale_count = 0
        for index, stage in enumerate(required):
            expected = expected_stage_paths(variant, stage)
            status = stage_status(expected)
            commands = commands_for_stage(variant, stage)
            freshness_inputs = freshness_inputs_for_stage(variant, stage, commands, required)
            freshness = stage_freshness(expected, freshness_inputs, status)
            if status == "complete":
                complete_count += 1
            elif status == "partial":
                partial_count += 1
            else:
                missing_count += 1
            if freshness["freshness"] == "fresh":
                fresh_count += 1
            elif freshness["freshness"] == "stale":
                stale_count += 1
            if missing_upstream and status != "complete":
                blocked_count += 1
            missing_labels = [item["label"] for item in expected if not item["exists"]]
            completed_labels = [item["label"] for item in expected if item["exists"]]
            rows.append(
                {
                    "variant_id": str(variant.get("id", "")),
                    "variant_label": str(variant.get("label", variant.get("id", ""))),
                    "stage": stage,
                    "stage_index": str(index),
                    "status": status,
                    "completed_outputs": ",".join(completed_labels),
                    "expected_outputs": ",".join(item["label"] for item in expected),
                    "missing_outputs": ",".join(missing_labels),
                    "blocked_by_missing_upstream": ",".join(missing_upstream),
                    **freshness,
                    "command_count": str(len(commands)),
                    "first_command": commands[0].get("command", "") if commands else "",
                    "product_lut_ready": "false",
                }
            )
            if status != "complete":
                missing_upstream.append(stage)

        if missing_count == 0 and partial_count == 0:
            state = "stale" if stale_count else "complete"
        elif complete_count or partial_count:
            state = "partial"
        else:
            state = "planned_only"
        variant_summaries.append(
            {
                "id": variant.get("id"),
                "label": variant.get("label", variant.get("id")),
                "state": state,
                "complete_stage_count": complete_count,
                "missing_stage_count": missing_count,
                "partial_stage_count": partial_count,
                "blocked_stage_count": blocked_count,
                "fresh_stage_count": fresh_count,
                "stale_stage_count": stale_count,
                "product_lut_ready": False,
            }
        )
    return rows, variant_summaries, manifest


def classify_result(result: dict[str, Any]) -> tuple[str, str, str, str]:
    path = Path(str(result.get("path", "")))
    try:
        local_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        local_path = path.as_posix()
    dataset_id = str(result.get("id", "")).lower()
    text = " ".join(str(result.get(key, "")) for key in ("id", "label")).lower()
    text = f"{text} {local_path.lower()}"
    suffix = Path(str(result.get("path", ""))).suffix.lower()
    solver = "Project"
    role = "artifact"
    dimensionality = "n/a"
    if dataset_id.startswith("gmsh_reference") or suffix == ".msh":
        solver = "Gmsh"
        role = "mesh_or_field_view"
        dimensionality = "2D/3D"
    elif "cross_section" in text or "geometry_3d" in text or "parameter_sweep" in text:
        solver = "Design Viewer"
        role = "visualization_or_report"
        dimensionality = "2D/3D" if suffix in {".html", ".png"} else "table"
    elif "crosstalk_xsection" in text:
        solver = "Meep FDTD"
        role = "high_resolution_xsection_crosstalk"
        dimensionality = "2D"
    elif "crosstalk_kernel" in text:
        solver = "Meep FDTD"
        role = "full_array_crosstalk"
        dimensionality = "3D"
    elif "camera_system_diagnostic" in text:
        solver = "Camera Diagnostic"
        role = "camera_system_diagnostic_response"
        dimensionality = "table"
    elif "camera_system_lut" in text or "camera lut" in text:
        solver = "Camera LUT"
        role = "blocked_product_lut_contract"
        dimensionality = "table"
    elif "devsim" in text or "native_summary" in text:
        solver = "DEVSIM"
        role = "electrical_terminal_summary"
        dimensionality = "2D"
    elif "gmsh" in text or suffix in {".msh", ".vtk", ".vtu"}:
        solver = "Gmsh"
        role = "mesh_or_field_view"
        dimensionality = "2D/3D"
    elif "gw" in text:
        solver = "G*W Coupling"
        role = "weighting_or_lut_reduction"
        dimensionality = "table"
    elif "variant" in text:
        solver = "Variant Manager"
        role = "sweep_management"
        dimensionality = "table"
    elif "accuracy" in text:
        solver = "Accuracy Gate"
        role = "gate_report"
        dimensionality = "scalar/table"
    elif "fdtd" in text or "generation" in text:
        solver = "Meep FDTD"
        role = "optical_generation"
        dimensionality = "2D/3D" if suffix == ".npz" else "n/a"
    if suffix == ".csv":
        kind = "table"
    elif suffix == ".json":
        kind = "structured_json"
    elif suffix == ".jsonl":
        kind = "jsonl_log"
    elif suffix == ".npz":
        kind = "array_dataset"
    elif suffix in {".vtk", ".vtu", ".msh"}:
        kind = "mesh_dataset"
    elif suffix == ".html":
        kind = "viewer_report"
    elif suffix == ".png":
        kind = "image_plot"
    elif suffix == ".md":
        kind = "markdown_report"
    else:
        kind = suffix.lstrip(".") or "file"
    return solver, kind, role, dimensionality


def extra_project_datasets(project_config: Path) -> list[dict[str, Any]]:
    project = read_json(project_config)
    config_dir = project_config.parent
    rows: list[dict[str, Any]] = []
    for key in [
        "stack_config",
        "tcad_profile",
        "design_space",
        "generation_map_2d",
        "generation_volume_3d",
        "design_viewer_manifest",
        "accuracy_gate",
        "optical_stack_summary",
    ]:
        value = project.get("inputs", {}).get(key)
        path = normalize_path(value, config_dir)
        if not path:
            continue
        rows.append(
            {
                "id": f"input_{key}",
                "label": key.replace("_", " "),
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "native_mesh": False,
            }
        )
    rows.append(
        {
            "id": "project_config",
            "label": "project config",
            "path": str(project_config),
            "exists": project_config.exists(),
            "size_bytes": project_config.stat().st_size if project_config.exists() else None,
            "native_mesh": False,
        }
    )
    return rows


def variant_file_datasets(variant_manifest_path: Path) -> list[dict[str, Any]]:
    if not variant_manifest_path.exists():
        return []
    manifest = read_json(variant_manifest_path)
    rows: list[dict[str, Any]] = []
    for variant in manifest.get("variants", []):
        variant_id = str(variant.get("id", "variant"))
        for key, value in variant.get("generated_files", {}).items():
            path = normalize_path(value)
            if not path:
                continue
            rows.append(
                {
                    "id": f"variant_{variant_id}_{key}",
                    "label": f"{variant_id} {key}",
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                    "native_mesh": False,
                }
            )
        for output_key, output_value in variant.get("planned_outputs", {}).items():
            output_path = normalize_path(output_value)
            if not output_path or not output_path.exists():
                continue
            if output_path.is_file():
                files = [output_path]
            else:
                files = sorted(path for path in output_path.rglob("*") if path.is_file())
            for path in files:
                rel = path.relative_to(output_path).as_posix() if output_path.is_dir() else path.name
                clean_rel = "".join(ch if ch.isalnum() else "_" for ch in rel).strip("_")
                native_mesh = path.suffix.lower() in {".msh", ".vtk", ".vtu", ".dat"}
                rows.append(
                    {
                        "id": f"variant_{variant_id}_{output_key}_{clean_rel}",
                        "label": f"{variant_id} {output_key} {rel}",
                        "path": str(path),
                        "exists": path.exists(),
                        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                        "native_mesh": native_mesh,
                    }
                )
    return rows


def build_dataset_rows(
    project_config: Path,
    studio_output_dir: Path,
    variant_manifest_path: Path | None = None,
) -> list[dict[str, str]]:
    payload = build_payload(project_config, studio_output_dir)
    raw_results = list(payload.get("results", [])) + extra_project_datasets(project_config)
    if variant_manifest_path:
        raw_results.extend(variant_file_datasets(variant_manifest_path))
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in raw_results:
        path = Path(str(result.get("path", ""))).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        solver, dataset_kind, role, dimensionality = classify_result(result)
        if "orchestrator" in str(result.get("id", "")).lower() or "orchestrator" in str(result.get("path", "")).lower():
            solver = "Variant Orchestrator"
            role = "execution_plan_or_log"
            dimensionality = "table/log"
        elif str(result.get("id", "")).startswith("variant_"):
            solver = "Variant Outputs" if solver == "Project" else solver
            if role == "artifact":
                role = "variant_artifact"
        rows.append(
            {
                "dataset_id": str(result.get("id", path.stem)),
                "solver": solver,
                "dataset_kind": dataset_kind,
                "role": role,
                "dimensionality": dimensionality,
                "native_mesh": "true" if result.get("native_mesh") else "false",
                "exists": "true" if result.get("exists") else "false",
                "size_bytes": "" if result.get("size_bytes") is None else str(result.get("size_bytes")),
                "path": str(path),
                "viewer": "open" if dataset_kind in {"viewer_report", "image_plot", "markdown_report"} else "",
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pill_class(status: str) -> str:
    if status in {"complete", "true", "fresh"}:
        return "ok"
    if status in {"partial", "stale", "unknown"}:
        return "warn"
    if status in {"missing", "false"}:
        return "bad"
    return "warn"


def run_rows_html(rows: list[dict[str, str]]) -> str:
    html_rows = []
    for row in rows:
        status = html.escape(row["status"])
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(row['variant_id'])}</td>"
            f"<td>{html.escape(row['stage'])}</td>"
            f"<td><span class=\"pill {pill_class(row['status'])}\">{status}</span></td>"
            f"<td><span class=\"pill {pill_class(row['freshness'])}\">{html.escape(row['freshness'])}</span></td>"
            f"<td>{html.escape(row['completed_outputs'])}</td>"
            f"<td>{html.escape(row['missing_outputs'])}</td>"
            f"<td>{html.escape(row['blocked_by_missing_upstream'])}</td>"
            f"<td>{html.escape(row['stale_reason'])}</td>"
            f"<td>{html.escape(row['command_count'])}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def dataset_rows_html(rows: list[dict[str, str]]) -> str:
    html_rows = []
    for row in rows:
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(row['dataset_id'])}</td>"
            f"<td>{html.escape(row['solver'])}</td>"
            f"<td>{html.escape(row['dataset_kind'])}</td>"
            f"<td>{html.escape(row['role'])}</td>"
            f"<td>{html.escape(row['dimensionality'])}</td>"
            f"<td><span class=\"pill {pill_class(row['exists'])}\">{html.escape(row['exists'])}</span></td>"
            f"<td>{html.escape(row['native_mesh'])}</td>"
            f"<td>{html.escape(row['size_bytes'])}</td>"
            f"<td>{html.escape(row['path'])}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def write_run_html(path: Path, run_rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Sensor Run Manager</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f4f6f8;font-size:13px}}
header{{padding:18px 22px;background:#111827;color:white}}
h1{{font-size:18px;margin:0 0 6px}}
p{{margin:0;color:#c9d3df}}
main{{padding:16px 22px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}}
.metric{{background:white;border:1px solid #cfd7df;border-radius:8px;padding:10px}}
.label{{color:#61707f;font-size:12px;margin-bottom:5px}}
.value{{font-weight:700;font-size:18px}}
.note{{background:#fff8eb;border:1px solid #f5c27a;color:#7c3f00;border-radius:8px;padding:10px;margin:12px 0}}
.tableWrap{{background:white;border:1px solid #cfd7df;border-radius:8px;overflow:auto}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px}}
th,td{{border-bottom:1px solid #e5e9ee;padding:7px 8px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th{{position:sticky;top:0;background:#f9fafb;z-index:1;color:#4b5563}}
.pill{{display:inline-flex;height:20px;align-items:center;padding:0 6px;border-radius:5px;border:1px solid #cfd7df;background:#fff}}
.ok{{border-color:#b7dec7;color:#177245;background:#f0fbf4}}
.warn{{border-color:#f5c27a;color:#b45309;background:#fff8eb}}
.bad{{border-color:#f0a4a4;color:#b91c1c;background:#fff5f5}}
</style>
</head>
<body>
<header>
  <h1>Image Sensor Run Manager</h1>
  <p>Stage status is inferred from expected output files. No solver jobs are launched by this report.</p>
</header>
<main>
	  <section class="metrics">
	    <div class="metric"><div class="label">Stage Rows</div><div class="value">{summary['stage_row_count']}</div></div>
	    <div class="metric"><div class="label">Complete Stages</div><div class="value">{summary['complete_stage_count']}</div></div>
	    <div class="metric"><div class="label">Stale Stages</div><div class="value">{summary['stale_stage_count']}</div></div>
	    <div class="metric"><div class="label">Product LUT Ready</div><div class="value">No</div></div>
	  </section>
  <div class="note">A complete local stage is trend evidence only. Product LUT readiness still requires measured inputs, calibrated transport, and convergence/accuracy gates.</div>
  <section class="tableWrap">
    <table>
	      <thead><tr><th>Variant</th><th>Stage</th><th>Status</th><th>Freshness</th><th>Completed Outputs</th><th>Missing Outputs</th><th>Blocked By</th><th>Stale Reason</th><th>Commands</th></tr></thead>
      <tbody>{run_rows_html(run_rows)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_dataset_html(path: Path, dataset_rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Sensor Dataset Catalog</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f4f6f8;font-size:13px}}
header{{padding:18px 22px;background:#111827;color:white}}
h1{{font-size:18px;margin:0 0 6px}}
p{{margin:0;color:#c9d3df}}
main{{padding:16px 22px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}}
.metric{{background:white;border:1px solid #cfd7df;border-radius:8px;padding:10px}}
.label{{color:#61707f;font-size:12px;margin-bottom:5px}}
.value{{font-weight:700;font-size:18px}}
.tableWrap{{background:white;border:1px solid #cfd7df;border-radius:8px;overflow:auto}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px}}
th,td{{border-bottom:1px solid #e5e9ee;padding:7px 8px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th{{position:sticky;top:0;background:#f9fafb;z-index:1;color:#4b5563}}
.pill{{display:inline-flex;height:20px;align-items:center;padding:0 6px;border-radius:5px;border:1px solid #cfd7df;background:#fff}}
.ok{{border-color:#b7dec7;color:#177245;background:#f0fbf4}}
.bad{{border-color:#f0a4a4;color:#b91c1c;background:#fff5f5}}
</style>
</head>
<body>
<header>
  <h1>Image Sensor Dataset Catalog</h1>
  <p>Structured catalog of monitor-like outputs, mesh datasets, tables, reports, and project inputs.</p>
</header>
<main>
  <section class="metrics">
    <div class="metric"><div class="label">Datasets</div><div class="value">{summary['dataset_count']}</div></div>
    <div class="metric"><div class="label">Existing</div><div class="value">{summary['existing_dataset_count']}</div></div>
    <div class="metric"><div class="label">Native Mesh/Data</div><div class="value">{summary['native_dataset_count']}</div></div>
    <div class="metric"><div class="label">Missing</div><div class="value">{summary['missing_dataset_count']}</div></div>
  </section>
  <section class="tableWrap">
    <table>
      <thead><tr><th>Dataset</th><th>Solver</th><th>Kind</th><th>Role</th><th>Dim</th><th>Exists</th><th>Native</th><th>Size</th><th>Path</th></tr></thead>
      <tbody>{dataset_rows_html(dataset_rows)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def summarize(run_rows: list[dict[str, str]], dataset_rows: list[dict[str, str]], variant_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_row_count": len(run_rows),
        "complete_stage_count": sum(1 for row in run_rows if row["status"] == "complete"),
        "partial_stage_count": sum(1 for row in run_rows if row["status"] == "partial"),
        "missing_stage_count": sum(1 for row in run_rows if row["status"] == "missing"),
        "blocked_stage_count": sum(1 for row in run_rows if row["blocked_by_missing_upstream"]),
        "fresh_stage_count": sum(1 for row in run_rows if row["freshness"] == "fresh"),
        "stale_stage_count": sum(1 for row in run_rows if row["freshness"] == "stale"),
        "variant_count": len(variant_summaries),
        "complete_variant_count": sum(1 for item in variant_summaries if item["state"] == "complete"),
        "stale_variant_count": sum(1 for item in variant_summaries if item["state"] == "stale"),
        "dataset_count": len(dataset_rows),
        "existing_dataset_count": sum(1 for row in dataset_rows if row["exists"] == "true"),
        "missing_dataset_count": sum(1 for row in dataset_rows if row["exists"] == "false"),
        "native_dataset_count": sum(1 for row in dataset_rows if row["native_mesh"] == "true"),
        "product_lut_ready": False,
        "accuracy_ready": False,
    }


def run(
    project_config: Path,
    variant_manifest: Path,
    output_dir: Path,
    studio_output_dir: Path,
    update_manifest: bool = True,
) -> dict[str, Any]:
    project_rows, project_summary = build_project_run_rows(project_config)
    variant_rows, variant_summaries, manifest = build_run_rows(variant_manifest)
    run_rows = project_rows + variant_rows
    variant_summaries = [project_summary] + variant_summaries
    dataset_rows = build_dataset_rows(project_config, studio_output_dir, variant_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / "run_manager_status.csv"
    run_json = output_dir / "run_manager_status.json"
    run_html = output_dir / "run_manager_status.html"
    dataset_csv = output_dir / "dataset_catalog.csv"
    dataset_json = output_dir / "dataset_catalog.json"
    dataset_html = output_dir / "dataset_catalog.html"
    summary = summarize(run_rows, dataset_rows, variant_summaries)
    data = {
        "schema": "image_sensor_run_manager_v1",
        "source_project_config": str(project_config),
        "source_variant_manifest": str(variant_manifest),
        "summary": summary,
        "variant_summaries": variant_summaries,
        "run_stage_rows": run_rows,
        "dataset_rows": dataset_rows,
        "outputs": {
            "run_csv": str(run_csv),
            "run_json": str(run_json),
            "run_html": str(run_html),
            "dataset_csv": str(dataset_csv),
            "dataset_json": str(dataset_json),
            "dataset_html": str(dataset_html),
        },
        "limitations": [
            "Run status is inferred from expected local output files; it is not a scheduler.",
            "Dataset catalog classifies files by current artifact names and extensions.",
            "Product LUT readiness remains false until measured inputs, calibrated transport, and convergence/accuracy gates pass.",
        ],
    }
    write_csv(run_csv, RUN_COLUMNS, run_rows)
    write_csv(dataset_csv, DATASET_COLUMNS, dataset_rows)
    write_json(run_json, data)
    write_json(dataset_json, {"schema": "image_sensor_dataset_catalog_v1", "summary": summary, "rows": dataset_rows})
    write_run_html(run_html, run_rows, summary)
    write_dataset_html(dataset_html, dataset_rows, summary)
    if update_manifest:
        manifest["run_manager_outputs"] = data["outputs"]
        manifest.setdefault("summary", {})
        manifest["summary"]["run_manager_stage_row_count"] = summary["stage_row_count"]
        manifest["summary"]["dataset_count"] = summary["dataset_count"]
        manifest["summary"]["product_lut_ready"] = False
        write_json(variant_manifest, manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"run_html: {rel_from_root(run_html)}")
    print(f"dataset_html: {rel_from_root(dataset_html)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--variant-manifest", type=Path, default=DEFAULT_VARIANT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--studio-output-dir", type=Path, default=DEFAULT_STUDIO_OUTPUT_DIR)
    parser.add_argument("--no-update-manifest", action="store_true")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.variant_manifest.resolve(),
        args.output_dir.resolve(),
        args.studio_output_dir.resolve(),
        update_manifest=not args.no_update_manifest,
    )


if __name__ == "__main__":
    main()
