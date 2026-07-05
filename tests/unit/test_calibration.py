from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pyisetcam import (
    camerae2e_calibration_evidence_manifest,
    camerae2e_calibration_evidence_requirements,
    camerae2e_calibration_evidence_validate,
    camerae2e_readiness_promotion_plan,
)


def test_camerae2e_calibration_evidence_defaults_block_calibrated_promotion() -> None:
    requirements = camerae2e_calibration_evidence_requirements()
    manifest = camerae2e_calibration_evidence_manifest()
    validation = camerae2e_calibration_evidence_validate()
    plan = camerae2e_readiness_promotion_plan()

    assert requirements["schema_version"] == "camerae2e_calibration_evidence_requirements_v1"
    assert manifest["source_status"] == "not_provided"
    assert validation["ok"] is True
    assert plan["schema_version"] == "camerae2e_readiness_promotion_plan_v1"
    assert plan["summary"]["evidence_count"] == 0
    assert plan["summary"]["promotion_candidate_count"] == 0
    assert plan["summary"]["blocked_count"] > 0
    assert any(
        item["entry"] == "hwisp_parameter_profiles"
        and "board_latency_trace" in item["missing_evidence_types"]
        for item in plan["plans"]
    )


def test_camerae2e_readiness_promotion_plan_uses_measured_evidence_bundle(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "hwisp_evidence"
    evidence_dir.mkdir()
    entries = []
    for evidence_type in (
        "board_latency_trace",
        "hardware_counter_trace",
        "3a_telemetry_trace",
    ):
        path = evidence_dir / f"{evidence_type}.json"
        path.write_text(json.dumps({"evidence_type": evidence_type}), encoding="utf-8")
        entries.append(
            {
                "evidence_id": f"unit_{evidence_type}",
                "entry_name": "hwisp_parameter_profiles",
                "evidence_type": evidence_type,
                "path": path.name,
                "sha256": _sha256(path),
                "measured": True,
                "validation_status": "pass",
            }
        )
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "camerae2e_calibration_evidence_manifest_v1",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )

    validation = camerae2e_calibration_evidence_validate(manifest_path)
    plan = camerae2e_readiness_promotion_plan(manifest_path)
    hwisp_plan = next(item for item in plan["plans"] if item["entry"] == "hwisp_parameter_profiles")

    assert validation["ok"] is True
    assert hwisp_plan["status"] == "promotion_candidate"
    assert hwisp_plan["target_tier"] == "calibrated"
    assert hwisp_plan["missing_evidence_types"] == []
    assert set(hwisp_plan["present_evidence_types"]) == {
        "board_latency_trace",
        "hardware_counter_trace",
        "3a_telemetry_trace",
    }


def test_camerae2e_calibration_evidence_validate_detects_hash_mismatch(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "measured_psf.json"
    evidence_path.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": "camerae2e_calibration_evidence_manifest_v1",
        "entries": [
            {
                "evidence_id": "bad_hash",
                "entry_name": "lens_patents_active",
                "evidence_type": "measured_psf",
                "path": str(evidence_path),
                "sha256": "sha256:not-the-real-hash",
                "measured": True,
                "validation_status": "pass",
            }
        ],
    }

    validation = camerae2e_calibration_evidence_validate(manifest)

    assert validation["ok"] is False
    assert any(issue["kind"] == "sha256_mismatch" for issue in validation["issues"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"
