"""CameraE2E calibration evidence manifests and readiness promotion planning."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .db_catalog import camerae2e_db_manifest, camerae2e_db_validate

_EVIDENCE_ENV = "PYISETCAM_CALIBRATION_EVIDENCE"
_VALIDATION_STATUSES = {"pass", "validated", "calibrated", "warn", "pending", "fail"}
_PROMOTION_STATUSES = {"pass", "validated", "calibrated"}


def camerae2e_calibration_evidence_requirements(
    *, include_missing: bool = True
) -> dict[str, Any]:
    """Return measured-evidence requirements for registry tier promotion."""

    manifest = camerae2e_db_manifest(include_missing=include_missing)
    entries = []
    for entry in manifest.get("entries", []):
        requirement = _entry_evidence_requirement(entry)
        if requirement["required_evidence_types"] or requirement["recommended_evidence_types"]:
            entries.append(requirement)
    required_types = sorted(
        {
            evidence_type
            for item in entries
            for evidence_type in item["required_evidence_types"]
        }
    )
    return {
        "schema_version": "camerae2e_calibration_evidence_requirements_v1",
        "summary": {
            "entry_count": len(entries),
            "required_evidence_types": required_types,
        },
        "entries": entries,
    }


def camerae2e_calibration_evidence_manifest(
    source: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load or normalize a CameraE2E calibration evidence manifest.

    The manifest is intentionally small JSON: it indexes measured artifacts by
    registry entry and evidence type.  It does not copy large calibration files
    into this repository.
    """

    payload, source_path = _load_evidence_source(source)
    entries = [_normalize_evidence_entry(item, source_path) for item in payload.get("entries", [])]
    by_entry: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for item in entries:
        by_entry[str(item.get("entry_name"))] = by_entry.get(str(item.get("entry_name")), 0) + 1
        by_type[str(item.get("evidence_type"))] = (
            by_type.get(str(item.get("evidence_type")), 0) + 1
        )
    return {
        "schema_version": "camerae2e_calibration_evidence_manifest_v1",
        "source": None if source_path is None else str(source_path),
        "source_status": payload.get("source_status", "loaded"),
        "truth_boundary": (
            "Evidence manifest only describes external measured artifacts; "
            "readiness promotion is not applied until validation passes."
        ),
        "summary": {
            "evidence_count": len(entries),
            "by_entry": by_entry,
            "by_type": by_type,
        },
        "entries": entries,
    }


def camerae2e_calibration_evidence_validate(
    source: str | Path | Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate calibration evidence references and hashes."""

    evidence = camerae2e_calibration_evidence_manifest(source)
    registry = {
        entry["name"]: entry
        for entry in camerae2e_db_manifest(include_missing=True).get("entries", [])
    }
    required_types = {
        evidence_type
        for item in camerae2e_calibration_evidence_requirements().get("entries", [])
        for evidence_type in item["required_evidence_types"]
    }
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for index, item in enumerate(evidence.get("entries", [])):
        entry_name = str(item.get("entry_name", ""))
        evidence_id = str(item.get("evidence_id", ""))
        evidence_type = str(item.get("evidence_type", ""))
        if not evidence_id:
            issues.append({"index": index, "kind": "missing_evidence_id"})
        if entry_name not in registry:
            issues.append(
                {
                    "index": index,
                    "kind": "unknown_registry_entry",
                    "entry_name": entry_name,
                }
            )
        if not evidence_type:
            issues.append({"index": index, "kind": "missing_evidence_type"})
        elif evidence_type not in required_types:
            warnings.append(
                {
                    "index": index,
                    "kind": "unknown_evidence_type",
                    "entry_name": entry_name,
                    "evidence_type": evidence_type,
                }
            )
        status = str(item.get("validation_status", "pending"))
        if status not in _VALIDATION_STATUSES:
            issues.append(
                {
                    "index": index,
                    "kind": "invalid_validation_status",
                    "validation_status": status,
                }
            )
        _validate_evidence_path_and_hash(item, index, issues, warnings, strict=strict)
    if strict:
        for item in evidence.get("entries", []):
            if item.get("measured") is not True:
                issues.append(
                    {
                        "entry_name": item.get("entry_name"),
                        "evidence_id": item.get("evidence_id"),
                        "kind": "not_measured",
                    }
                )
            if str(item.get("validation_status", "pending")) not in _PROMOTION_STATUSES:
                issues.append(
                    {
                        "entry_name": item.get("entry_name"),
                        "evidence_id": item.get("evidence_id"),
                        "kind": "not_promotion_ready",
                    }
                )
    return {
        "schema_version": "camerae2e_calibration_evidence_validation_v1",
        "strict": bool(strict),
        "ok": not issues,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "manifest": evidence,
    }


def camerae2e_readiness_promotion_plan(
    evidence_source: str | Path | Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
    include_missing: bool = True,
) -> dict[str, Any]:
    """Return a calibrated-readiness promotion plan from measured evidence."""

    registry_manifest = camerae2e_db_manifest(include_missing=include_missing)
    registry_validation = camerae2e_db_validate(strict=False, include_missing=include_missing)
    evidence = camerae2e_calibration_evidence_manifest(evidence_source)
    evidence_validation = camerae2e_calibration_evidence_validate(evidence, strict=False)
    evidence_by_entry = _passing_evidence_by_entry(evidence.get("entries", []))
    entry_issues = _validation_issues_by_entry(evidence_validation.get("issues", []))
    entry_blocking_warnings = _promotion_blocking_warnings_by_entry(
        evidence_validation.get("warnings", [])
    )
    plans = []
    for entry in registry_manifest.get("entries", []):
        requirement = _entry_evidence_requirement(entry)
        if not requirement["required_evidence_types"]:
            continue
        entry_name = str(entry["name"])
        required = set(requirement["required_evidence_types"])
        present = set(evidence_by_entry.get(entry_name, {}))
        missing = sorted(required - present)
        current_tier = str(entry.get("readiness_tier", "available"))
        blocking_validation = entry_issues.get(entry_name, []) + entry_blocking_warnings.get(
            entry_name,
            [],
        )
        if current_tier == "calibrated":
            status = "already_calibrated"
            target_tier = "calibrated"
        elif blocking_validation:
            status = "blocked_invalid_evidence"
            target_tier = current_tier
        elif missing:
            status = "blocked_missing_evidence"
            target_tier = current_tier
        else:
            status = "promotion_candidate"
            target_tier = "calibrated"
        plans.append(
            {
                "entry": entry_name,
                "artifact_id": entry.get("artifact_id"),
                "family": entry.get("family"),
                "role": entry.get("role"),
                "current_tier": current_tier,
                "target_tier": target_tier,
                "status": status,
                "required_evidence_types": sorted(required),
                "present_evidence_types": sorted(present),
                "missing_evidence_types": missing,
                "evidence_ids": [
                    item.get("evidence_id")
                    for values in evidence_by_entry.get(entry_name, {}).values()
                    for item in values
                ],
                "validation_issues": entry_issues.get(entry_name, []),
                "promotion_blocking_warnings": entry_blocking_warnings.get(entry_name, []),
                "truth_boundary": entry.get("provenance", {}).get("truth_boundary"),
            }
        )
    status_counts: dict[str, int] = {}
    for item in plans:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    return {
        "schema_version": "camerae2e_readiness_promotion_plan_v1",
        "strict": bool(strict),
        "ok": evidence_validation.get("ok") and registry_validation.get("ok"),
        "summary": {
            "plan_count": len(plans),
            "status_counts": status_counts,
            "promotion_candidate_count": status_counts.get("promotion_candidate", 0),
            "blocked_count": sum(
                count for status, count in status_counts.items() if status.startswith("blocked_")
            ),
            "evidence_count": evidence.get("summary", {}).get("evidence_count", 0),
        },
        "evidence_validation": {
            "ok": evidence_validation.get("ok"),
            "issue_count": evidence_validation.get("issue_count", 0),
            "warning_count": evidence_validation.get("warning_count", 0),
            "issues": evidence_validation.get("issues", []),
            "warnings": evidence_validation.get("warnings", []),
        },
        "registry_validation": {
            "ok": registry_validation.get("ok"),
            "warning_count": registry_validation.get("warning_count", 0),
            "stale_dependency_count": registry_validation.get("stale_dependency_count", 0),
        },
        "plans": plans,
    }


def _load_evidence_source(
    source: str | Path | Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return _jsonable(source), None
    resolved = source
    if resolved is None:
        env_value = os.environ.get(_EVIDENCE_ENV)
        resolved = env_value if env_value else None
    if resolved is None:
        return {
            "schema_version": "camerae2e_calibration_evidence_manifest_v1",
            "source_status": "not_provided",
            "entries": [],
        }, None
    path = Path(resolved).expanduser()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        return {
            "schema_version": "camerae2e_calibration_evidence_manifest_v1",
            "source_status": "missing",
            "entries": [],
        }, manifest_path
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def _normalize_evidence_entry(item: Mapping[str, Any], source_path: Path | None) -> dict[str, Any]:
    payload = _jsonable(item)
    if "validation_status" not in payload:
        payload["validation_status"] = "pending"
    if "measured" not in payload:
        payload["measured"] = False
    if payload.get("path") is not None:
        path = Path(str(payload["path"])).expanduser()
        if not path.is_absolute() and source_path is not None:
            path = source_path.parent / path
        payload["path"] = str(path)
    return payload


def _entry_evidence_requirement(entry: Mapping[str, Any]) -> dict[str, Any]:
    family = str(entry.get("family", ""))
    role = str(entry.get("role", ""))
    required: list[str]
    recommended: list[str]
    if family == "lens":
        required = ["measured_psf", "diffraction_or_wave_optics_reference"]
        recommended = ["coating_or_flare_trace", "manufacturing_tolerance_stack"]
    elif family == "sensor" and role == "optical-response-lut":
        required = ["measured_qe_response", "sensor_specific_fdtd_lut", "optical_stack_materials"]
        recommended = ["localized_crosstalk_scan", "cra_response_measurement"]
    elif family == "sensor" and role == "electrical-collection-lut":
        required = [
            "tcad_process_deck",
            "measured_electrical_response",
            "fdtd_tcad_lineage_match",
        ]
        recommended = ["dark_current_measurement", "lag_or_full_well_measurement"]
    elif family == "sensor" and "stack" in role:
        required = ["measured_stack_or_cad", "material_index_reference"]
        recommended = ["process_corner_table"]
    elif family == "isp":
        required = ["board_latency_trace", "hardware_counter_trace", "3a_telemetry_trace"]
        recommended = ["bsp_revision_lock", "thermal_or_bandwidth_stress_trace"]
    elif family == "perception":
        required = ["dataset_split_manifest", "model_evaluation_report"]
        recommended = ["training_config", "domain_shift_report"]
    else:
        required = []
        recommended = []
    return {
        "entry": entry.get("name"),
        "artifact_id": entry.get("artifact_id"),
        "family": family,
        "role": role,
        "current_tier": entry.get("readiness_tier"),
        "required_evidence_types": required,
        "recommended_evidence_types": recommended,
        "truth_boundary": entry.get("provenance", {}).get("truth_boundary"),
    }


def _validate_evidence_path_and_hash(
    item: Mapping[str, Any],
    index: int,
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    strict: bool,
) -> None:
    path_value = item.get("path")
    if path_value in (None, ""):
        payload = {
            "index": index,
            "kind": "missing_evidence_path",
            "entry_name": item.get("entry_name"),
            "evidence_id": item.get("evidence_id"),
        }
        (issues if strict else warnings).append(payload)
        return
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        payload = {
            "index": index,
            "kind": "missing_evidence_path",
            "entry_name": item.get("entry_name"),
            "evidence_id": item.get("evidence_id"),
            "path": str(path),
        }
        (issues if strict else warnings).append(payload)
        return
    expected_hash = item.get("sha256")
    if expected_hash in (None, ""):
        warnings.append(
            {
                "index": index,
                "kind": "missing_sha256",
                "entry_name": item.get("entry_name"),
                "evidence_id": item.get("evidence_id"),
                "path": str(path),
            }
        )
        return
    actual_hash = _sha256_file(path)
    normalized_expected = _normalize_sha(expected_hash)
    if normalized_expected != actual_hash:
        issues.append(
            {
                "index": index,
                "kind": "sha256_mismatch",
                "entry_name": item.get("entry_name"),
                "evidence_id": item.get("evidence_id"),
                "path": str(path),
                "expected": normalized_expected,
                "actual": actual_hash,
            }
        )


def _passing_evidence_by_entry(
    evidence_entries: list[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_entry: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in evidence_entries:
        if item.get("measured") is not True:
            continue
        if str(item.get("validation_status", "pending")) not in _PROMOTION_STATUSES:
            continue
        entry_name = str(item.get("entry_name", ""))
        evidence_type = str(item.get("evidence_type", ""))
        if not entry_name or not evidence_type:
            continue
        by_entry.setdefault(entry_name, {}).setdefault(evidence_type, []).append(dict(item))
    return by_entry


def _validation_issues_by_entry(
    issues: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for item in issues:
        entry_name = item.get("entry_name")
        if entry_name is None:
            continue
        by_entry.setdefault(str(entry_name), []).append(dict(item))
    return by_entry


def _promotion_blocking_warnings_by_entry(
    warnings: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    blocking_kinds = {"missing_evidence_path", "missing_sha256"}
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for item in warnings:
        if item.get("kind") not in blocking_kinds:
            continue
        entry_name = item.get("entry_name")
        if entry_name is None:
            continue
        by_entry.setdefault(str(entry_name), []).append(dict(item))
    return by_entry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _normalize_sha(value: Any) -> str:
    text = str(value).strip()
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


cameraE2ECalibrationEvidenceRequirements = (  # noqa: N816
    camerae2e_calibration_evidence_requirements
)
cameraE2ECalibrationEvidenceManifest = camerae2e_calibration_evidence_manifest  # noqa: N816
cameraE2ECalibrationEvidenceValidate = camerae2e_calibration_evidence_validate  # noqa: N816
cameraE2EReadinessPromotionPlan = camerae2e_readiness_promotion_plan  # noqa: N816
