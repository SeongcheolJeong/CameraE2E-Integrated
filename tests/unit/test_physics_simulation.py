from __future__ import annotations

import json
from pathlib import Path

from pyisetcam import (
    camerae2e_physics_simulation_commands,
    camerae2e_physics_simulation_manifest,
    camerae2e_physics_simulation_validate,
    cameraE2EPhysicsSimulationManifest,
)


def test_camerae2e_physics_simulation_manifest_discovers_external_roots(
    tmp_path: Path,
) -> None:
    fdtd_root = _fake_fdtd_root(tmp_path / "FDTD", same_lineage=True)
    rayoptics_root = _fake_rayoptics_root(tmp_path / "RayOptics")

    manifest = camerae2e_physics_simulation_manifest(
        fdtd_root=fdtd_root,
        rayoptics_root=rayoptics_root,
        output_dir=tmp_path / "reports",
    )
    validation = camerae2e_physics_simulation_validate(manifest)
    local_commands = camerae2e_physics_simulation_commands(
        stages=manifest["stages"],
        include_expensive=False,
    )
    all_commands = camerae2e_physics_simulation_commands(
        stages=manifest["stages"],
        include_expensive=True,
    )

    assert manifest["schema_version"] == "camerae2e_physics_simulation_manifest_v1"
    assert manifest["summary"]["stage_count"] == 6
    assert manifest["active_runs"]["manifest_lineage_match"] is True
    assert Path(manifest["reports"]["json"]).exists()
    assert Path(manifest["reports"]["html"]).exists()
    assert validation["ok"] is True
    assert validation["stage_count"] == 6
    assert len(all_commands) > len(local_commands)
    assert all(command["cost_tier"] != "external_expensive" for command in local_commands)

    stages = {stage["stage_id"]: stage for stage in manifest["stages"]}
    assert stages["fdtd_optical_lut"]["status"] == "available"
    assert stages["tcad_generation_map"]["status"] == "available"
    assert stages["rayoptics_lens_psf"]["status"] == "available"
    assert "fdtd_sensor.py" in stages["fdtd_optical_lut"]["camerae2e_modules"]
    assert "tcad_sensor.py" in stages["tcad_generation_map"]["camerae2e_modules"]
    assert "lens_patents.py" in stages["rayoptics_lens_psf"]["camerae2e_modules"]

    rendered = json.loads(Path(manifest["reports"]["json"]).read_text(encoding="utf-8"))
    assert rendered["merge_policy"]["solver_execution"] == "external_batch_commands_not_unit_tests"


def test_camerae2e_physics_simulation_manifest_flags_tcad_lineage_mismatch(
    tmp_path: Path,
) -> None:
    fdtd_root = _fake_fdtd_root(tmp_path / "FDTD", same_lineage=False)
    manifest = cameraE2EPhysicsSimulationManifest(
        fdtd_root=fdtd_root,
        rayoptics_root=_fake_rayoptics_root(tmp_path / "RayOptics"),
    )
    strict_validation = camerae2e_physics_simulation_validate(manifest, strict=True)

    stages = {stage["stage_id"]: stage for stage in manifest["stages"]}
    assert manifest["active_runs"]["manifest_lineage_match"] is False
    assert stages["tcad_generation_map"]["status"] == "stale_dependency"
    assert strict_validation["ok"] is False
    assert any(
        warning["kind"] == "stale_dependency"
        for warning in strict_validation["warnings"]
    )


def _fake_fdtd_root(root: Path, *, same_lineage: bool) -> Path:
    run = root / "runs/convergence_cra3_rgb_r84_gridsnap_quant"
    tcad_run = run if same_lineage else root / "runs/fdtd_to_tcad_generation_2d_cra_smoke"
    for path in (
        root / "sensor_db/generated_stack_configs",
        root / "configs",
        root / "materials",
        run,
        tcad_run,
        root / "runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for script in (
        "build_sensor_db_from_techinsights.py",
        "build_camera_e2e_sensor_luts.py",
        "export_camera_e2e_handoff_manifest.py",
        "meep_supercell_lut.py",
        "run_camera_e2e_fdtd_field_sweep.py",
        "run_camera_e2e_package_pipeline.py",
        "export_camera_e2e_ingest_luts.py",
        "simulate_camera_e2e_sensor_probe.py",
        "tcad_gmsh_pixel_mesh.py",
        "devsim_split_pd_2d.py",
        "tcad_accuracy_gate.py",
        "tcad_calibration_loop.py",
    ):
        (root / script).write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (root / "sensor_db/sensor_catalog.json").write_text("{}", encoding="utf-8")
    (run / "camera_lut.json").write_text('{"schema": "camera_lut"}', encoding="utf-8")
    (tcad_run / "tcad_generation_map_2d.npz").write_bytes(b"npz")
    summary = root / "runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json"
    summary.write_text("{}", encoding="utf-8")
    gate = root / "runs/tcad_accuracy_gate_reference_profile"
    gate.mkdir(parents=True, exist_ok=True)
    (gate / "tcad_accuracy_gate.json").write_text("{}", encoding="utf-8")
    return root


def _fake_rayoptics_root(root: Path) -> Path:
    package = root / "CameraE2E_Lens_DB_v9_20260627/data/lens_patents"
    (package / "raytrace_psf").mkdir(parents=True, exist_ok=True)
    (package / "raytrace_psf_highres").mkdir(parents=True, exist_ok=True)
    (package / "lens_patent_simulation_v9.sqlite").write_bytes(b"sqlite")
    (package / "raytrace_psf/manifest.json").write_text("{}", encoding="utf-8")
    (package / "raytrace_psf_highres/manifest.json").write_text("{}", encoding="utf-8")
    (root / "backend").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "rayoptics-env/bin").mkdir(parents=True, exist_ok=True)
    (root / "rayoptics-env/bin/python").write_text("#!/bin/sh\n", encoding="utf-8")
    return root
