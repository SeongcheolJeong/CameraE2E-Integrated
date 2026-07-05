from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyisetcam import (
    CameraE2EDBEntry,
    camerae2e_db_catalog,
    camerae2e_db_get,
    camerae2e_db_lineage,
    camerae2e_db_manifest,
    camerae2e_db_parameters,
    camerae2e_db_search,
    camerae2e_db_summary,
    camerae2e_db_validate,
    camerae2e_physics_pipeline_plan,
)


def test_camerae2e_db_catalog_has_core_families() -> None:
    entries = camerae2e_db_catalog()

    assert entries
    assert all(isinstance(entry, CameraE2EDBEntry) for entry in entries)
    families = {entry.family for entry in entries}

    assert {"lens", "sensor", "isp", "perception", "assets", "parity"} <= families


def test_camerae2e_db_search_finds_runtime_parameter_sources() -> None:
    lens_rows = camerae2e_db_search("lens", include_missing=False)
    fdtd_rows = camerae2e_db_search(tags=["fdtd"], include_missing=False)
    hwisp_rows = camerae2e_db_search(family="isp", include_missing=False)
    perception_rows = camerae2e_db_search("yolo", include_missing=False)

    assert any(row["name"] == "lens_patents_active" for row in lens_rows)
    assert any(row["name"] == "fdtd_sensor_lut_active" for row in fdtd_rows)
    assert any(row["name"] == "hwisp_parameter_profiles" for row in hwisp_rows)
    assert any(row["name"] == "task_perception_model_profiles" for row in perception_rows)


def test_camerae2e_db_parameters_are_directly_usable_paths() -> None:
    lens_params = camerae2e_db_parameters("lens_patents_active")
    hwisp_params = camerae2e_db_parameters("hwisp_parameter_profiles")
    perception_params = camerae2e_db_parameters("task_perception_model_profiles")

    assert Path(lens_params["db_path"]).exists()
    assert Path(lens_params["psf_dir"]).exists()
    assert isinstance(hwisp_params["profile_names"], list)
    assert hwisp_params["profile_names"]
    assert Path(perception_params["profiles_path"]).exists()
    assert isinstance(perception_params["profile_names"], list)
    assert perception_params["profile_names"]


def test_camerae2e_db_get_and_summary_are_json_friendly() -> None:
    row = camerae2e_db_get("lens_patents_active")
    summary = camerae2e_db_summary()

    assert row["name"] == "lens_patents_active"
    assert isinstance(row["path"], str)
    assert isinstance(row["parameters"]["db_path"], str)
    assert summary["total"] >= 6
    assert "lens_patents_active" in summary["active"]


def test_camerae2e_db_manifest_validation_and_lineage_are_json_friendly() -> None:
    manifest = camerae2e_db_manifest()
    validation = camerae2e_db_validate()
    lineage = camerae2e_db_lineage("tcad_sensor_db_active")

    assert manifest["schema_version"] == "camerae2e_db_manifest_v1"
    assert manifest["entries"]
    assert all("readiness_tier" in entry for entry in manifest["entries"])
    assert validation["schema_version"] == "camerae2e_db_validation_v1"
    assert lineage["root"] == "tcad_sensor_db_active"
    assert any(edge["to"] == "fdtd_sensor_lut_active" for edge in lineage["edges"])


def test_camerae2e_db_validate_promotes_active_fdtd_tcad_run_mismatch_to_stale_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fdtd_root = tmp_path / "FDTD"
    fdtd_run = fdtd_root / "runs" / "active_fdtd_run"
    tcad_run = fdtd_root / "runs" / "fdtd_to_tcad_generation_2d_cra_smoke"
    fdtd_run.mkdir(parents=True)
    tcad_run.mkdir(parents=True)
    lut_path = _write_minimal_fdtd_lut(fdtd_run)
    generation_path = _write_minimal_tcad_generation_map(tcad_run)
    _write_default_tcad_summaries(fdtd_root, generation_path)
    _write_default_tcad_gate(fdtd_root)

    monkeypatch.setenv("PYISETCAM_FDTD_ROOT", str(fdtd_root))
    monkeypatch.setenv("PYISETCAM_FDTD_LUT_PATH", str(lut_path))

    tcad = camerae2e_db_get("tcad_sensor_db_active")
    validation = camerae2e_db_validate()

    assert tcad["readiness_tier"] == "calibration_required"
    assert "active artifact mismatch" in tcad["stale_reason"]
    assert validation["stale_dependency_count"] >= 1

    plan = camerae2e_physics_pipeline_plan()
    tcad_action = next(
        action for action in plan["actions"] if action["entry"] == "tcad_sensor_db_active"
    )
    assert tcad_action["kind"] == "stale_dependency"
    assert tcad_action["action"] == "refresh_downstream_from_current_dependency"
    assert plan["active_runs"]["lineage_match"] is False


def _write_minimal_fdtd_lut(run_dir: Path) -> Path:
    summary_csv = run_dir / "camera_lut_summary.csv"
    lut_path = run_dir / "camera_lut.json"
    summary_csv.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "ocl-3x3,450,center,0,0,0,0,0.7,1.0",
                "ocl-3x3,550,center,0,0,0,0,0.8,1.0",
                "ocl-3x3,650,center,0,0,0,0,0.9,1.0",
                "ocl-3x3,550,edge20x,1,0,20,0,0.5,0.625",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lut_path.write_text(
        json.dumps(
            {
                "schema": "camera_supercell_optical_lut_v2",
                "mode": "ocl-3x3",
                "wavelengths_nm": [450.0, 550.0, 650.0],
                "summary_csv": str(summary_csv),
            }
        ),
        encoding="utf-8",
    )
    return lut_path


def _write_minimal_tcad_generation_map(run_dir: Path) -> Path:
    path = run_dir / "tcad_generation_map_2d.npz"
    generation = np.ones((1, 2, 2), dtype=float) * 1.0e21
    np.savez(
        path,
        schema=np.asarray(["tcad_generation_map_2d_x_depth_v1"]),
        generation_cm3_s=generation,
        x_um=np.asarray([0.0, 1.0]),
        depth_um_from_si_top=np.asarray([0.1, 0.2]),
        case=np.asarray(["center"]),
        wavelength_nm=np.asarray([550.0]),
        field_x_norm=np.asarray([0.0]),
        field_z_norm=np.asarray([0.0]),
        cra_x_deg=np.asarray([0.0]),
        cra_z_deg=np.asarray([0.0]),
    )
    return path


def _write_default_tcad_summaries(fdtd_root: Path, generation_path: Path) -> None:
    for relative, case in (
        ("runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json", "center"),
        ("runs/devsim_split_pd_2d_fdtd_map_proxy_edge20x_smoke/summary.json", "center"),
    ):
        path = fdtd_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "devsim_split_pd_2d_smoke_v1",
                    "config": {
                        "generation_profile_case": case,
                        "generation_profile_wavelength_nm": 550.0,
                        "generation_map_npz": str(generation_path),
                    },
                    "generation_source": "imported_2d_map",
                    "electrical_model": "proxy-pinned-split-pd",
                    "left_photo_delta_a_per_cm": 1.0e-6,
                    "right_photo_delta_a_per_cm": 1.0e-6,
                    "photo_split_phase_x_proxy": 0.0,
                    "terminal_current_balance_illuminated_a_per_cm": 0.0,
                }
            ),
            encoding="utf-8",
        )


def _write_default_tcad_gate(fdtd_root: Path) -> None:
    path = fdtd_root / "runs" / "tcad_accuracy_gate_reference_profile" / "tcad_accuracy_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "tcad_accuracy_gate_v1",
                "profile_name": "unit_test_proxy",
                "framework_ready": True,
                "accuracy_ready": False,
                "accuracy_blocking_failure_count": 1,
                "framework_blocking_failure_count": 0,
                "checks": [],
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )
