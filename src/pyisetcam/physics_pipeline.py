"""CameraE2E external physics pipeline planning helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .db_catalog import camerae2e_db_manifest, camerae2e_db_validate

_PIPELINE_ORDER = (
    "fdtd_sensor_stack_catalog",
    "fdtd_sensor_lut_active",
    "tcad_sensor_db_active",
    "lens_patents_active",
    "hwisp_parameter_profiles",
    "task_perception_model_profiles",
)


def camerae2e_physics_pipeline_plan(
    *, strict: bool = False, include_missing: bool = True
) -> dict[str, Any]:
    """Return a machine-readable refresh and calibration plan for external assets."""

    manifest = camerae2e_db_manifest(include_missing=include_missing)
    validation = camerae2e_db_validate(strict=strict, include_missing=include_missing)
    entries = {entry["name"]: entry for entry in manifest.get("entries", [])}
    active_runs = _active_run_summary(entries)

    ordered_names = [name for name in _PIPELINE_ORDER if name in entries]
    seen = set(ordered_names)
    ordered_names.extend(sorted(name for name in entries if name not in seen))
    actions = [
        _entry_action(entries[name], entries, active_runs=active_runs, strict=strict)
        for name in ordered_names
    ]
    refresh_order = [
        action
        for action in actions
        if action["action"]
        in {
            "provide_or_generate_asset",
            "refresh_downstream_from_current_dependency",
            "attach_calibration_evidence",
        }
    ]
    blocking = [action for action in actions if action["severity"] == "blocking"]
    stale = [action for action in actions if action["kind"] == "stale_dependency"]
    calibration = [
        action
        for action in actions
        if action["readiness_tier"] in {"proxy", "calibration_required"}
    ]
    return {
        "schema_version": "camerae2e_physics_pipeline_plan_v1",
        "strict": bool(strict),
        "ok": not blocking,
        "summary": {
            "action_count": len(actions),
            "refresh_action_count": len(refresh_order),
            "blocking_count": len(blocking),
            "stale_dependency_count": len(stale),
            "calibration_required_count": len(calibration),
        },
        "active_runs": active_runs,
        "manifest_validation": {
            "ok": validation.get("ok"),
            "issue_count": validation.get("issue_count", 0),
            "warning_count": validation.get("warning_count", 0),
            "stale_dependency_count": validation.get("stale_dependency_count", 0),
        },
        "refresh_order": refresh_order,
        "actions": actions,
        "truth_boundaries": _truth_boundaries(actions),
    }


def _entry_action(
    entry: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    *,
    active_runs: Mapping[str, Any],
    strict: bool,
) -> dict[str, Any]:
    name = str(entry["name"])
    readiness = str(entry.get("readiness_tier", "available"))
    stale_reason = entry.get("stale_reason")
    status = str(entry.get("status", "missing"))
    severity = "info"
    kind = "ready"
    action = "validate_current_asset"
    reason = "Asset is discoverable and has no stale dependency marker."
    blocks_strict = False
    remediation: list[dict[str, Any]] = []

    if stale_reason:
        kind = "stale_dependency"
        action = "refresh_downstream_from_current_dependency"
        reason = str(stale_reason)
        severity = "blocking" if strict else "warning"
        blocks_strict = True
        remediation = _stale_remediation(name, entry, entries, active_runs)
    elif readiness == "missing" or status == "missing":
        kind = "missing_asset"
        action = "provide_or_generate_asset"
        reason = "Required or known optional asset is missing from the current workspace."
        severity = "blocking" if strict else "warning"
        blocks_strict = True
        remediation = _missing_remediation(entry)
    elif readiness == "calibration_required":
        kind = "calibration_required"
        action = "attach_calibration_evidence"
        reason = "Framework is connected, but quantitative accuracy needs measured evidence."
        severity = "blocking" if strict else "warning"
        blocks_strict = True
        remediation = _calibration_remediation(entry)
    elif readiness == "proxy":
        kind = "proxy_truth_boundary"
        action = "keep_proxy_label_or_calibrate"
        reason = "Research sweeps may use this asset, but calibrated/sign-off claims are blocked."
        severity = "blocking" if strict else "info"
        blocks_strict = True
        remediation = _calibration_remediation(entry)

    return {
        "entry": name,
        "artifact_id": entry.get("artifact_id"),
        "family": entry.get("family"),
        "role": entry.get("role"),
        "status": status,
        "readiness_tier": readiness,
        "kind": kind,
        "action": action,
        "severity": severity,
        "blocks_strict_validation": blocks_strict,
        "reason": reason,
        "dependencies": list(entry.get("dependencies", [])),
        "env_vars": list(entry.get("env_vars", [])),
        "path": entry.get("path"),
        "refresh_command": entry.get("refresh_command"),
        "validation_gates": list(entry.get("validation_gates", [])),
        "truth_boundary": entry.get("provenance", {}).get("truth_boundary"),
        "inputs": _entry_inputs(entry),
        "outputs": _entry_outputs(entry),
        "remediation": remediation,
    }


def _active_run_summary(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fdtd_entry = entries.get("fdtd_sensor_lut_active", {})
    tcad_entry = entries.get("tcad_sensor_db_active", {})
    fdtd_path = _path_or_none(fdtd_entry.get("path"))
    generation_path = _path_or_none(
        dict(tcad_entry.get("parameters", {})).get("generation_map_path")
    )
    fdtd_run = None if fdtd_path is None else str(fdtd_path.parent)
    generation_run = None if generation_path is None else str(generation_path.parent)
    return {
        "fdtd_lut_path": None if fdtd_path is None else str(fdtd_path),
        "fdtd_lut_run": fdtd_run,
        "tcad_generation_map_path": None if generation_path is None else str(generation_path),
        "tcad_generation_map_run": generation_run,
        "lineage_match": bool(fdtd_run and generation_run and fdtd_run == generation_run),
    }


def _stale_remediation(
    name: str,
    entry: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    active_runs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if name == "tcad_sensor_db_active":
        fdtd_entry = entries.get("fdtd_sensor_lut_active", {})
        return [
            {
                "step": "Regenerate FDTD-to-TCAD generation map from the active FDTD LUT run.",
                "input": active_runs.get("fdtd_lut_path"),
                "expected_output_parent": active_runs.get("fdtd_lut_run"),
                "current_generation_map_parent": active_runs.get("tcad_generation_map_run"),
                "command": entry.get("refresh_command"),
            },
            {
                "step": "Regenerate or re-point DEVSIM summaries to that generation map.",
                "inputs": dict(entry.get("parameters", {})).get("collection_summary_paths", []),
                "dependency": fdtd_entry.get("name", "fdtd_sensor_lut_active"),
            },
            {
                "step": "Re-run CameraE2E registry validation in strict mode before promotion.",
                "command": "python tools/validate_camerae2e_physics_pipeline.py --strict",
            },
        ]
    return [
        {
            "step": "Refresh this artifact from its current dependency chain.",
            "command": entry.get("refresh_command"),
        }
    ]


def _missing_remediation(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": "Create, mount, or point the registry to this asset.",
            "env_vars": list(entry.get("env_vars", [])),
            "expected_path": entry.get("path"),
            "command": entry.get("refresh_command"),
        }
    ]


def _calibration_remediation(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    family = str(entry.get("family", "asset"))
    if family == "lens":
        evidence = "wave-optics/diffraction comparison, measured PSF, tolerance, coating data"
    elif family == "sensor":
        evidence = "measured stack, sensor-specific FDTD LUT, TCAD process deck, QE/noise data"
    elif family == "isp":
        evidence = "board traces, hardware counters, BSP timing, and 3A telemetry"
    elif family == "perception":
        evidence = "dataset-specific training/evaluation split and calibrated task metrics"
    else:
        evidence = "measured or pinned evidence appropriate to the asset"
    return [
        {
            "step": "Attach calibration evidence before raising readiness tier.",
            "required_evidence": evidence,
            "current_truth_boundary": entry.get("provenance", {}).get("truth_boundary"),
        }
    ]


def _entry_inputs(entry: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(entry.get("parameters", {}))
    keys = {
        "catalog_path",
        "lut_path",
        "generation_map_path",
        "collection_summary_paths",
        "accuracy_gate_path",
        "db_path",
        "psf_dir",
        "profile_names",
        "profiles_path",
    }
    return {key: value for key, value in params.items() if key in keys}


def _entry_outputs(entry: Mapping[str, Any]) -> dict[str, Any]:
    name = str(entry.get("name", ""))
    params = dict(entry.get("parameters", {}))
    if name == "fdtd_sensor_lut_active":
        return {"lut_path": params.get("lut_path") or entry.get("path")}
    if name == "tcad_sensor_db_active":
        return {
            "generation_map_path": params.get("generation_map_path"),
            "collection_summary_paths": params.get("collection_summary_paths", []),
            "accuracy_gate_path": params.get("accuracy_gate_path"),
        }
    if name == "lens_patents_active":
        return {
            "db_path": params.get("db_path"),
            "psf_dir": params.get("psf_dir"),
            "highres_psf_dir": params.get("highres_psf_dir"),
        }
    return {"path": entry.get("path")}


def _truth_boundaries(actions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entry": str(action["entry"]),
            "readiness_tier": action.get("readiness_tier"),
            "truth_boundary": action.get("truth_boundary"),
            "blocks_strict_validation": action.get("blocks_strict_validation"),
        }
        for action in actions
        if action.get("truth_boundary")
    ]


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


cameraE2EPhysicsPipelinePlan = camerae2e_physics_pipeline_plan  # noqa: N816
