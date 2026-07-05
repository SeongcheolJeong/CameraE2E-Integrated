"""Unified physics-simulation bridge for CameraE2E.

The integrated repository carries CameraE2E code, selected FDTD/TCAD/RayOptics
scripts, and compact final-result DB artifacts.  Expensive solver execution is
still opt-in; this module standardizes how bundled assets and optional external
workspaces are discovered, validated, and reported.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .db_catalog import camerae2e_db_get
from .physics_pipeline import camerae2e_physics_pipeline_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
MONOREPO_FDTD_ROOT = REPO_ROOT / "simulations/fdtd_tcad"
MONOREPO_RAYOPTICS_ROOT = REPO_ROOT / "simulations/rayoptics"
IN_REPO_CAMERA_DB_ROOT = REPO_ROOT / "camerae2e_db"
SIBLING_CAMERA_DB_ROOT = REPO_ROOT.parent / "CameraE2E-DB"
OPTIONAL_EXTERNAL_FDTD_ROOT = Path(
    os.environ.get("PYISETCAM_EXTERNAL_FDTD_ROOT", str(REPO_ROOT / "external/fdtd_tcad"))
)
OPTIONAL_EXTERNAL_RAYOPTICS_ROOT = Path(
    os.environ.get("PYISETCAM_EXTERNAL_RAYOPTICS_ROOT", str(REPO_ROOT / "external/rayoptics"))
)
DEFAULT_FDTD_ROOT = (
    MONOREPO_FDTD_ROOT if MONOREPO_FDTD_ROOT.exists() else OPTIONAL_EXTERNAL_FDTD_ROOT
)
DEFAULT_RAYOPTICS_ROOT = (
    MONOREPO_RAYOPTICS_ROOT
    if MONOREPO_RAYOPTICS_ROOT.exists()
    else OPTIONAL_EXTERNAL_RAYOPTICS_ROOT
)
DEFAULT_OUTPUT_DIR = Path("reports/camerae2e_goal")

_HASH_LIMIT_BYTES = 128 * 1024 * 1024


def camerae2e_physics_simulation_manifest(
    *,
    fdtd_root: str | Path | None = None,
    rayoptics_root: str | Path | None = None,
    camera_db_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Describe the merged FDTD/TCAD/RayOptics simulation bridge.

    The manifest is the CameraE2E-owned integration surface for external
    simulation workspaces.  It records source scripts, selected artifacts,
    CameraE2E import modules, command templates, and stale-lineage state without
    copying large external databases into the repository.
    """

    fdtd = _resolve_root(fdtd_root, "PYISETCAM_FDTD_ROOT", DEFAULT_FDTD_ROOT)
    rayoptics = _resolve_root(
        rayoptics_root,
        "PYISETCAM_RAYOPTICS_ROOT",
        DEFAULT_RAYOPTICS_ROOT,
    )
    camera_db = _resolve_camera_db_root(
        camera_db_root,
        allow_default=fdtd_root is None and rayoptics_root is None,
    )
    fdtd_artifact_root = camera_db / "fdtd_tcad" if camera_db is not None else fdtd
    entries = _registry_entries()
    pipeline_plan = camerae2e_physics_pipeline_plan(strict=False)
    active_runs = dict(pipeline_plan.get("active_runs", {}))

    fdtd_lut_path = _discover_fdtd_lut_path(fdtd, fdtd_artifact_root, entries)
    tcad_generation_path = _discover_tcad_generation_map_path(
        fdtd,
        fdtd_artifact_root,
        fdtd_lut_path,
        entries,
    )
    rayoptics_package = _discover_rayoptics_package(rayoptics, entries, camera_db)

    stages = [
        _fdtd_stack_catalog_stage(fdtd, fdtd_artifact_root),
        _fdtd_optical_lut_stage(fdtd, fdtd_lut_path, entries),
        _tcad_generation_map_stage(fdtd, fdtd_lut_path, tcad_generation_path, entries),
        _tcad_devsim_collection_stage(fdtd, fdtd_artifact_root, tcad_generation_path, entries),
        _rayoptics_lens_stage(rayoptics, rayoptics_package, entries),
        _camerae2e_import_validation_stage(entries),
    ]
    validation = camerae2e_physics_simulation_validate(
        {
            "schema_version": "camerae2e_physics_simulation_manifest_v1",
            "stages": stages,
            "active_runs": active_runs,
        }
    )
    payload = {
        "schema_version": "camerae2e_physics_simulation_manifest_v1",
        "workspace_roots": {
            "fdtd": _path_info(fdtd, role="external_fdtd_tcad_workspace"),
            "rayoptics": _path_info(rayoptics, role="external_rayoptics_workspace"),
            "camera_db": _path_info(camera_db, role="camerae2e_final_db_repository"),
        },
        "summary": _summary(stages, validation),
        "active_runs": {
            **active_runs,
            "manifest_fdtd_lut_path": _string_path(fdtd_lut_path),
            "manifest_tcad_generation_map_path": _string_path(tcad_generation_path),
            "manifest_lineage_match": _lineage_match(fdtd_lut_path, tcad_generation_path),
        },
        "merge_policy": {
            "large_external_assets": "reference_by_path_and_hash_do_not_vendor",
            "solver_execution": "external_batch_commands_not_unit_tests",
            "camerae2e_boundary": (
                "CameraE2E owns manifest, import adapters, lineage gates, and "
                "reports; external workspaces own expensive solver generation."
            ),
            "final_db_repository": (
                "Lens and sensor DB artifacts are packaged as final-result-only "
                "runtime inputs under camerae2e_db or a sibling CameraE2E-DB repo."
            ),
            "truth_boundary": (
                "FDTD/TCAD/RayOptics simulation artifacts are research inputs until "
                "measured calibration and strict lineage gates are attached."
            ),
        },
        "stages": stages,
        "artifact_graph": _artifact_graph(stages),
        "commands": camerae2e_physics_simulation_commands(stages=stages, include_expensive=True),
        "local_validation_commands": camerae2e_physics_simulation_commands(
            stages=stages,
            include_expensive=False,
        ),
        "validation": validation,
        "physics_pipeline_plan_summary": pipeline_plan.get("summary", {}),
        "truth_boundaries": pipeline_plan.get("truth_boundaries", []),
    }
    payload["summary"] = _summary(stages, validation)

    if output_dir is not None:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "physics_simulation_manifest.json"
        html_path = root / "physics_simulation_manifest.html"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        html_path.write_text(_render_manifest_html(payload), encoding="utf-8")
        payload["reports"] = {"json": str(json_path), "html": str(html_path)}
    return payload


def camerae2e_physics_simulation_validate(
    manifest: str | Path | dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate a physics simulation manifest without launching expensive solvers."""

    payload = _load_manifest(manifest)
    stages = list(payload.get("stages", []))
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for stage in stages:
        stage_id = str(stage.get("stage_id", "unknown"))
        status = str(stage.get("status", "unknown"))
        if status in {"workspace_missing", "missing_output"}:
            warnings.append(
                {
                    "stage_id": stage_id,
                    "kind": status,
                    "message": f"{stage_id} is not fully discoverable in the current workspace.",
                }
            )
        if status == "stale_dependency":
            warnings.append(
                {
                    "stage_id": stage_id,
                    "kind": "stale_dependency",
                    "message": str(stage.get("stale_reason", "stale dependency")),
                }
            )
        for script in stage.get("source_scripts", []):
            if script.get("required") and not script.get("exists"):
                warnings.append(
                    {
                        "stage_id": stage_id,
                        "kind": "missing_source_script",
                        "path": script.get("path"),
                    }
                )
        for output in stage.get("outputs", []):
            if output.get("required") and not output.get("exists"):
                warnings.append(
                    {
                        "stage_id": stage_id,
                        "kind": "missing_output",
                        "path": output.get("path"),
                    }
                )

    if strict:
        issues = [
            warning
            for warning in warnings
            if warning.get("kind")
            in {"workspace_missing", "missing_output", "stale_dependency", "missing_source_script"}
        ]
    status_counts = Counter(str(stage.get("status", "unknown")) for stage in stages)
    return {
        "schema_version": "camerae2e_physics_simulation_validation_v1",
        "strict": bool(strict),
        "ok": not issues,
        "stage_count": len(stages),
        "status_counts": dict(status_counts),
        "warning_count": len(warnings),
        "issue_count": len(issues),
        "warnings": warnings,
        "issues": issues,
    }


def camerae2e_physics_simulation_commands(
    *,
    stages: list[dict[str, Any]] | None = None,
    stage_id: str | None = None,
    include_expensive: bool = False,
) -> list[dict[str, Any]]:
    """Return ordered command templates for the physics simulation bridge."""

    if stages is None:
        stages = camerae2e_physics_simulation_manifest().get("stages", [])
    commands: list[dict[str, Any]] = []
    for stage in stages:
        if stage_id is not None and stage.get("stage_id") != stage_id:
            continue
        for command in stage.get("commands", []):
            cost_tier = str(command.get("cost_tier", "local_report"))
            if not include_expensive and cost_tier == "external_expensive":
                continue
            commands.append(
                {
                    "stage_id": stage.get("stage_id"),
                    "family": stage.get("family"),
                    **dict(command),
                }
            )
    return commands


def _fdtd_stack_catalog_stage(fdtd_root: Path, artifact_root: Path) -> dict[str, Any]:
    catalog = artifact_root / "sensor_db/sensor_catalog.json"
    configs = artifact_root / "sensor_db/generated_stack_configs"
    scripts = [
        _script_info(fdtd_root / "build_sensor_db_from_techinsights.py", required=False),
        _script_info(fdtd_root / "build_camera_e2e_sensor_luts.py", required=False),
        _script_info(fdtd_root / "export_camera_e2e_handoff_manifest.py", required=False),
    ]
    outputs = [
        _path_info(catalog, role="sensor_stack_catalog", required=True),
        _path_info(configs, role="generated_stack_configs", required=False),
    ]
    return _stage(
        stage_id="fdtd_stack_catalog",
        family="sensor",
        readiness_tier="proxy",
        root=fdtd_root,
        source_scripts=scripts,
        inputs=[],
        outputs=outputs,
        camerae2e_modules=["db_catalog.py", "image_sensor_db.py", "fdtd_sensor.py"],
        commands=[
            _command(
                "external_sensor_db_refresh",
                f"cd {fdtd_root} && python build_sensor_db_from_techinsights.py",
                cost_tier="external_expensive",
                requires_review=True,
            ),
            _command(
                "camerae2e_registry_report",
                "python tools/render_camerae2e_asset_registry_report.py",
                cost_tier="local_report",
            ),
        ],
        validation_gates=["catalog_exists", "metadata_proxy_caveat"],
        truth_boundary="metadata_derived_stack_configs_unless_product_cad_listed",
    )


def _fdtd_optical_lut_stage(
    fdtd_root: Path,
    fdtd_lut_path: Path | None,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entry = entries.get("fdtd_sensor_lut_active", {})
    outputs = [_path_info(fdtd_lut_path, role="camera_lut_json", required=True)]
    return _stage(
        stage_id="fdtd_optical_lut",
        family="sensor",
        readiness_tier=str(entry.get("readiness_tier", "proxy")),
        root=fdtd_root,
        source_scripts=[
            _script_info(fdtd_root / "meep_supercell_lut.py", required=False),
            _script_info(fdtd_root / "run_camera_e2e_fdtd_field_sweep.py", required=False),
            _script_info(fdtd_root / "run_camera_e2e_package_pipeline.py", required=False),
        ],
        inputs=[
            _path_info(fdtd_root / "configs", role="fdtd_config_dir", required=False),
            _path_info(fdtd_root / "materials", role="material_nk_tables", required=False),
        ],
        outputs=outputs,
        camerae2e_modules=["fdtd_sensor.py", "sensor.py", "db_catalog.py"],
        commands=[
            _command(
                "external_fdtd_lut_batch",
                f"cd {fdtd_root} && python run_camera_e2e_package_pipeline.py",
                cost_tier="external_expensive",
                requires_review=True,
            ),
            _command(
                "camerae2e_fdtd_report",
                "python tools/render_fdtd_sensor_physics_report.py",
                cost_tier="local_report",
            ),
        ],
        validation_gates=["lut_schema", "physics_sanity", "convergence_metadata"],
        truth_boundary=str(
            entry.get("provenance", {}).get(
                "truth_boundary",
                "optical_absorption_and_regional_response_proxy",
            )
        ),
    )


def _tcad_generation_map_stage(
    fdtd_root: Path,
    fdtd_lut_path: Path | None,
    generation_path: Path | None,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entry = entries.get("tcad_sensor_db_active", {})
    lineage_match = _lineage_match(fdtd_lut_path, generation_path)
    stale_reason = None
    if fdtd_lut_path is not None and generation_path is not None and not lineage_match:
        stale_reason = (
            "FDTD camera_lut.json and TCAD generation map are from different run roots."
        )
    stage = _stage(
        stage_id="tcad_generation_map",
        family="sensor",
        readiness_tier=str(entry.get("readiness_tier", "calibration_required")),
        root=fdtd_root,
        source_scripts=[
            _script_info(fdtd_root / "export_camera_e2e_ingest_luts.py", required=False),
            _script_info(fdtd_root / "simulate_camera_e2e_sensor_probe.py", required=False),
            _script_info(fdtd_root / "tcad_gmsh_pixel_mesh.py", required=False),
        ],
        inputs=[_path_info(fdtd_lut_path, role="source_fdtd_lut", required=True)],
        outputs=[_path_info(generation_path, role="tcad_generation_map_npz", required=True)],
        camerae2e_modules=["tcad_sensor.py", "physics_pipeline.py"],
        commands=[
            _command(
                "external_fdtd_to_tcad_generation_map",
                f"cd {fdtd_root} && python export_camera_e2e_ingest_luts.py",
                cost_tier="external_expensive",
                requires_review=True,
            ),
            _command(
                "camerae2e_fdtd_tcad_report",
                "python tools/render_fdtd_tcad_sensor_report.py",
                cost_tier="local_report",
            ),
        ],
        validation_gates=["generation_map_schema", "lineage_match"],
        truth_boundary="fdtd_generation_map_for_carrier_collection_proxy",
    )
    if stale_reason:
        stage["status"] = "stale_dependency"
        stage["stale_reason"] = stale_reason
    stage["lineage"] = {
        "fdtd_lut_run": None if fdtd_lut_path is None else str(fdtd_lut_path.parent),
        "tcad_generation_map_run": None
        if generation_path is None
        else str(generation_path.parent),
        "lineage_match": lineage_match,
    }
    return stage


def _tcad_devsim_collection_stage(
    fdtd_root: Path,
    artifact_root: Path,
    generation_path: Path | None,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entry = entries.get("tcad_sensor_db_active", {})
    params = dict(entry.get("parameters", {}))
    fixture_summaries = [
        artifact_root / "fixtures/devsim_split_pd_center/summary.json",
        artifact_root / "fixtures/devsim_split_pd_edge/summary.json",
    ]
    fixture_summaries = [path for path in fixture_summaries if path.exists()]
    summary_paths = fixture_summaries or [
        Path(str(path)).expanduser()
        for path in params.get("collection_summary_paths", [])
        if path
    ]
    if not summary_paths:
        summary_paths = sorted(artifact_root.glob("runs/devsim_*/*summary.json"))[:4]
    accuracy_gate = _first_existing(
        [
            artifact_root / "fixtures/tcad_accuracy_gate/tcad_accuracy_gate.json",
            _path_from_params(params, "accuracy_gate_path"),
            artifact_root / "runs/tcad_accuracy_gate_reference_profile/tcad_accuracy_gate.json",
        ]
    )
    outputs = [
        _path_info(path, role="devsim_collection_summary", required=True)
        for path in summary_paths
    ]
    outputs.append(_path_info(accuracy_gate, role="tcad_accuracy_gate", required=False))
    return _stage(
        stage_id="tcad_devsim_collection",
        family="sensor",
        readiness_tier=str(entry.get("readiness_tier", "calibration_required")),
        root=fdtd_root,
        source_scripts=[
            _script_info(fdtd_root / "devsim_split_pd_2d.py", required=False),
            _script_info(fdtd_root / "tcad_accuracy_gate.py", required=False),
            _script_info(fdtd_root / "tcad_calibration_loop.py", required=False),
        ],
        inputs=[_path_info(generation_path, role="tcad_generation_map_npz", required=True)],
        outputs=outputs,
        camerae2e_modules=["tcad_sensor.py", "calibration.py", "physics_pipeline.py"],
        commands=[
            _command(
                "external_devsim_collection_batch",
                f"cd {fdtd_root} && python devsim_split_pd_2d.py",
                cost_tier="external_expensive",
                requires_review=True,
            ),
            _command(
                "camerae2e_physics_validation",
                "python tools/validate_camerae2e_physics_pipeline.py",
                cost_tier="validation",
            ),
        ],
        validation_gates=["devsim_summary_balance", "accuracy_gate", "calibration_evidence"],
        truth_boundary=str(
            entry.get("provenance", {}).get(
                "truth_boundary",
                "carrier_collection_framework_not_product_calibrated_tcad",
            )
        ),
    )


def _rayoptics_lens_stage(
    rayoptics_root: Path,
    package: Path | None,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entry = entries.get("lens_patents_active", {})
    params = dict(entry.get("parameters", {}))
    db_path = _first_existing(
        [
            None
            if package is None
            else package / "data/lens_patents/lens_patent_simulation_v9.sqlite",
            _path_from_params(params, "db_path"),
        ]
    )
    psf_manifest = _first_existing(
        [
            None if package is None else package / "data/lens_patents/raytrace_psf/manifest.json",
            _path_from_params(params, "psf_dir") / "manifest.json"
            if _path_from_params(params, "psf_dir") is not None
            else None,
        ]
    )
    highres_manifest = _first_existing(
        [
            None
            if package is None
            else package / "data/lens_patents/raytrace_psf_highres/manifest.json",
            _path_from_params(params, "highres_psf_dir") / "manifest.json"
            if _path_from_params(params, "highres_psf_dir") is not None
            else None,
        ]
    )
    return _stage(
        stage_id="rayoptics_lens_psf",
        family="lens",
        readiness_tier=str(entry.get("readiness_tier", "proxy")),
        root=rayoptics_root,
        source_scripts=[
            _script_info(rayoptics_root / "backend", required=False),
            _script_info(rayoptics_root / "src", required=False),
            _script_info(rayoptics_root / "rayoptics-env/bin/python", required=False),
        ],
        inputs=[
            _path_info(db_path, role="lens_patent_sqlite", required=True),
            _path_info(psf_manifest, role="raytrace_psf_manifest", required=True),
            _path_info(highres_manifest, role="highres_psf_manifest", required=False),
        ],
        outputs=[
            _path_info(db_path, role="camerae2e_lens_db", required=True),
            _path_info(psf_manifest, role="camerae2e_geometric_psf_manifest", required=True),
        ],
        camerae2e_modules=["lens_patents.py", "optics.py", "db_catalog.py"],
        commands=[
            _command(
                "external_rayoptics_psf_package",
                f"cd {rayoptics_root} && ./rayoptics-env/bin/python -m pytest",
                cost_tier="external_expensive",
                requires_review=True,
            ),
            _command(
                "camerae2e_lens_db_report",
                "python tools/render_lens_db_camerae2e_report.py",
                cost_tier="local_report",
            ),
        ],
        validation_gates=["sqlite_exists", "rayoptics_psf_manifest_exists", "geometric_psf_caveat"],
        truth_boundary=str(
            entry.get("provenance", {}).get(
                "truth_boundary",
                "geometric_psf_not_diffraction_wave_optics",
            )
        ),
    )


def _camerae2e_import_validation_stage(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _stage(
        stage_id="camerae2e_import_validation",
        family="camerae2e",
        readiness_tier="validated",
        root=Path.cwd(),
        source_scripts=[
            _script_info(Path("tools/validate_camerae2e_physics_pipeline.py"), required=True),
            _script_info(Path("tools/render_camerae2e_asset_registry_report.py"), required=True),
            _script_info(Path("tools/render_fdtd_tcad_sensor_report.py"), required=True),
            _script_info(Path("tools/render_lens_db_camerae2e_report.py"), required=True),
        ],
        inputs=[
            _path_info(_entry_path(entries, "fdtd_sensor_lut_active"), role="active_fdtd_lut"),
            _path_info(_entry_path(entries, "tcad_sensor_db_active"), role="active_tcad_root"),
            _path_info(_entry_path(entries, "lens_patents_active"), role="active_lens_db"),
        ],
        outputs=[
            _path_info(Path("reports/camerae2e_goal/asset_registry.json"), role="registry_report"),
            _path_info(
                Path("reports/camerae2e_goal/physics_pipeline_plan.json"),
                role="physics_pipeline_plan",
            ),
        ],
        camerae2e_modules=[
            "db_catalog.py",
            "physics_pipeline.py",
            "physics_simulation.py",
            "fdtd_sensor.py",
            "tcad_sensor.py",
            "lens_patents.py",
        ],
        commands=[
            _command(
                "camerae2e_validate_physics_pipeline",
                "python tools/validate_camerae2e_physics_pipeline.py",
                cost_tier="validation",
            ),
            _command(
                "camerae2e_render_registry",
                "python tools/render_camerae2e_asset_registry_report.py",
                cost_tier="local_report",
            ),
            _command(
                "camerae2e_goal_gate",
                "python tools/run_camerae2e_goal_gate.py",
                cost_tier="validation",
            ),
        ],
        validation_gates=["manifest_schema", "lineage_gate", "goal_gate"],
        truth_boundary="evidence_snapshot_not_product_certification",
    )


def _stage(
    *,
    stage_id: str,
    family: str,
    readiness_tier: str,
    root: Path,
    source_scripts: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    camerae2e_modules: list[str],
    commands: list[dict[str, Any]],
    validation_gates: list[str],
    truth_boundary: str,
) -> dict[str, Any]:
    root_exists = root.exists()
    required_outputs = [output for output in outputs if output.get("required", False)]
    missing_required_outputs = [
        output for output in required_outputs if not output.get("exists", False)
    ]
    if not root_exists:
        status = "workspace_missing"
    elif missing_required_outputs:
        status = "missing_output"
    else:
        status = "available"
    return {
        "stage_id": stage_id,
        "family": family,
        "status": status,
        "readiness_tier": readiness_tier,
        "workspace_root": str(root),
        "source_scripts": source_scripts,
        "inputs": inputs,
        "outputs": outputs,
        "camerae2e_modules": camerae2e_modules,
        "commands": commands,
        "validation_gates": validation_gates,
        "truth_boundary": truth_boundary,
        "merge_policy": {
            "asset_storage": "external_reference",
            "camerae2e_storage": "manifest_and_import_adapters",
            "unit_test_policy": "no_expensive_solver_execution",
        },
    }


def _command(
    command_id: str,
    command: str,
    *,
    cost_tier: str,
    requires_review: bool = False,
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "command": command,
        "cost_tier": cost_tier,
        "requires_review": bool(requires_review),
    }


def _script_info(path: Path, *, required: bool) -> dict[str, Any]:
    return _path_info(path, role="source_script", required=required)


def _path_info(path: str | Path | None, *, role: str, required: bool = False) -> dict[str, Any]:
    if path is None:
        return {
            "role": role,
            "path": None,
            "required": bool(required),
            "exists": False,
            "kind": None,
            "bytes": None,
            "sha256": None,
        }
    resolved = Path(path).expanduser()
    exists = resolved.exists()
    kind = "directory" if resolved.is_dir() else "file" if resolved.is_file() else None
    size = resolved.stat().st_size if resolved.is_file() else None
    digest = _sha256_file(resolved) if resolved.is_file() and size <= _HASH_LIMIT_BYTES else None
    return {
        "role": role,
        "path": str(resolved),
        "required": bool(required),
        "exists": bool(exists),
        "kind": kind,
        "bytes": size,
        "sha256": digest,
    }


def _registry_entries() -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name in (
        "fdtd_sensor_stack_catalog",
        "fdtd_sensor_lut_active",
        "tcad_sensor_db_active",
        "lens_patents_active",
    ):
        try:
            entries[name] = camerae2e_db_get(name)
        except Exception:  # pragma: no cover - defensive for partially installed packages
            entries[name] = {}
    return entries


def _resolve_root(value: str | Path | None, env_var: str, default: Path) -> Path:
    if value is not None:
        return Path(value).expanduser()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser()
    return default.expanduser()


def _resolve_camera_db_root(value: str | Path | None, *, allow_default: bool) -> Path | None:
    if value is not None:
        return Path(value).expanduser()
    env_value = os.environ.get("PYISETCAM_CAMERA_DB_ROOT")
    if env_value:
        return Path(env_value).expanduser()
    if not allow_default:
        return None
    if IN_REPO_CAMERA_DB_ROOT.exists():
        return IN_REPO_CAMERA_DB_ROOT
    if SIBLING_CAMERA_DB_ROOT.exists():
        return SIBLING_CAMERA_DB_ROOT
    return None


def _discover_fdtd_lut_path(
    fdtd_root: Path,
    artifact_root: Path,
    entries: dict[str, dict[str, Any]],
) -> Path | None:
    entry = entries.get("fdtd_sensor_lut_active", {})
    return _first_existing(
        [
            artifact_root / "fixtures/fdtd_active_lut/camera_lut.json",
            artifact_root / "runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json",
            fdtd_root / "fixtures/fdtd_active_lut/camera_lut.json",
            fdtd_root / "runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json",
            fdtd_root / "runs/convergence_cra3z_rgb_r84_gridsnap_quant/camera_lut.json",
            fdtd_root / "runs/fdtd_to_tcad_generation_2d_cra_smoke/camera_lut.json",
            _path_from_params(dict(entry.get("parameters", {})), "lut_path"),
            _path_or_none(entry.get("path")),
        ]
    )


def _discover_tcad_generation_map_path(
    fdtd_root: Path,
    artifact_root: Path,
    fdtd_lut_path: Path | None,
    entries: dict[str, dict[str, Any]],
) -> Path | None:
    entry = entries.get("tcad_sensor_db_active", {})
    same_run = (
        None if fdtd_lut_path is None else fdtd_lut_path.parent / "tcad_generation_map_2d.npz"
    )
    return _first_existing(
        [
            artifact_root / "fixtures/tcad_generation_map/tcad_generation_map_2d.npz",
            artifact_root
            / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz",
            fdtd_root / "fixtures/tcad_generation_map/tcad_generation_map_2d.npz",
            same_run,
            fdtd_root / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz",
            fdtd_root / "runs/fdtd_to_tcad_generation_2d_cra5_smoke/tcad_generation_map_2d.npz",
            _path_from_params(dict(entry.get("parameters", {})), "generation_map_path"),
        ]
    )


def _discover_rayoptics_package(
    rayoptics_root: Path,
    entries: dict[str, dict[str, Any]],
    camera_db_root: Path | None,
) -> Path | None:
    candidates = [
        None
        if camera_db_root is None
        else camera_db_root / "lens_db/CameraE2E_Lens_DB_v9_20260627",
        rayoptics_root / "CameraE2E_Lens_DB_v9_20260627",
        rayoptics_root / "Lens_DB_portable_v9_20260623",
    ]
    entry_path = _entry_path(entries, "lens_patents_active")
    if entry_path is not None:
        for parent in entry_path.parents:
            if parent.name.startswith("CameraE2E_Lens_DB") or parent.name.startswith(
                "Lens_DB_portable"
            ):
                candidates.append(parent)
                break
    return _first_existing(candidates)


def _first_existing(candidates: list[Path | None]) -> Path | None:
    fallback = None
    for candidate in candidates:
        if candidate is None:
            continue
        if fallback is None:
            fallback = candidate
        if candidate.exists():
            return candidate
    return fallback


def _path_from_params(params: dict[str, Any], key: str) -> Path | None:
    return _path_or_none(params.get(key))


def _entry_path(entries: dict[str, dict[str, Any]], name: str) -> Path | None:
    return _path_or_none(entries.get(name, {}).get("path"))


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _lineage_match(fdtd_lut_path: Path | None, generation_path: Path | None) -> bool:
    return bool(
        fdtd_lut_path is not None
        and generation_path is not None
        and fdtd_lut_path.parent == generation_path.parent
    )


def _artifact_graph(stages: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [
        {
            "id": stage["stage_id"],
            "family": stage["family"],
            "status": stage["status"],
            "readiness_tier": stage["readiness_tier"],
        }
        for stage in stages
    ]
    edges = [
        {"from": "fdtd_stack_catalog", "to": "fdtd_optical_lut", "artifact": "stack_config"},
        {"from": "fdtd_optical_lut", "to": "tcad_generation_map", "artifact": "camera_lut"},
        {
            "from": "tcad_generation_map",
            "to": "tcad_devsim_collection",
            "artifact": "generation_map",
        },
        {
            "from": "fdtd_optical_lut",
            "to": "camerae2e_import_validation",
            "artifact": "fdtd_lut",
        },
        {
            "from": "tcad_devsim_collection",
            "to": "camerae2e_import_validation",
            "artifact": "tcad_collection_db",
        },
        {
            "from": "rayoptics_lens_psf",
            "to": "camerae2e_import_validation",
            "artifact": "geometric_psf",
        },
    ]
    return {"nodes": nodes, "edges": edges}


def _summary(stages: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(str(stage.get("status", "unknown")) for stage in stages)
    tier_counts = Counter(str(stage.get("readiness_tier", "available")) for stage in stages)
    return {
        "stage_count": len(stages),
        "status_counts": dict(status_counts),
        "readiness_tiers": dict(tier_counts),
        "warning_count": validation.get("warning_count", 0),
        "issue_count": validation.get("issue_count", 0),
        "strict_ok": camerae2e_physics_simulation_validate(
            {"stages": stages},
            strict=True,
        ).get("ok"),
    }


def _load_manifest(manifest: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return camerae2e_physics_simulation_manifest()
    if isinstance(manifest, dict):
        return manifest
    return json.loads(Path(manifest).expanduser().read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _string_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _render_manifest_html(payload: dict[str, Any]) -> str:
    rows = []
    for stage in payload.get("stages", []):
        command_count = len(stage.get("commands", []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(stage.get('stage_id')))}</td>"
            f"<td>{html.escape(str(stage.get('family')))}</td>"
            f"<td>{html.escape(str(stage.get('status')))}</td>"
            f"<td>{html.escape(str(stage.get('readiness_tier')))}</td>"
            f"<td>{html.escape(str(stage.get('workspace_root')))}</td>"
            f"<td>{command_count}</td>"
            f"<td>{html.escape(str(stage.get('truth_boundary')))}</td>"
            "</tr>"
        )
    summary = html.escape(json.dumps(payload.get("summary", {}), sort_keys=True))
    roots = html.escape(json.dumps(payload.get("workspace_roots", {}), sort_keys=True))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CameraE2E Physics Simulation Manifest</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "margin:24px;color:#172033}table{border-collapse:collapse;width:100%;"
        "font-size:13px}td,th{border:1px solid #ccd3df;padding:7px;vertical-align:top}"
        "th{background:#eef3fb;text-align:left}code{background:#f6f8fb;padding:2px 4px}"
        "</style></head><body>"
        "<h1>CameraE2E Physics Simulation Manifest</h1>"
        f"<p><strong>Summary:</strong> <code>{summary}</code></p>"
        f"<p><strong>Workspace roots:</strong> <code>{roots}</code></p>"
        "<table><thead><tr><th>Stage</th><th>Family</th><th>Status</th>"
        "<th>Tier</th><th>Workspace</th><th>Commands</th><th>Truth Boundary</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


cameraE2EPhysicsSimulationManifest = camerae2e_physics_simulation_manifest  # noqa: N816
cameraE2EPhysicsSimulationValidate = camerae2e_physics_simulation_validate  # noqa: N816
cameraE2EPhysicsSimulationCommands = camerae2e_physics_simulation_commands  # noqa: N816
