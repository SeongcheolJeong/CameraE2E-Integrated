from __future__ import annotations

from pathlib import Path

from pyisetcam import camerae2e_goal_gate


def test_camerae2e_goal_gate_writes_reports_and_smoke_artifacts(tmp_path: Path) -> None:
    payload = camerae2e_goal_gate(tmp_path / "reports", seed=123)
    checks = {check["name"]: check for check in payload["checks"]}
    requirements = {item["requirement"]: item for item in payload["requirements"]}

    assert payload["schema_version"] == "camerae2e_goal_gate_v1"
    assert payload["ok"] is True
    assert Path(payload["reports"]["json"]).exists()
    assert Path(payload["reports"]["html"]).exists()
    assert Path(payload["artifact_dir"]).exists()
    assert checks["calibration_evidence_policy"]["status"] == "pass"
    assert (
        checks["calibration_evidence_policy"]["evidence"]["promotion_summary"][
            "promotion_candidate_count"
        ]
        == 0
    )
    assert checks["faca_smoke"]["status"] == "pass"
    assert checks["parameter_optimization"]["status"] == "pass"
    assert checks["optimization_escalation_plan"]["status"] == "pass"
    assert checks["dataset_factory_smoke"]["status"] == "pass"
    assert checks["adas_kitti_demo_smoke"]["status"] == "pass"
    assert checks["camera_spec_variant_smoke"]["status"] == "pass"
    assert checks["signoff_claim_guard"]["status"] == "pass"
    assert checks["signoff_claim_guard"]["evidence"]["calibrated_or_signoff_claim_allowed"] is False
    assert requirements["RAW data factory from optimized camera cases"]["status"] == "pass"
    assert requirements["Camera-spec variant re-capture"]["status"] == "pass"
    assert requirements["Calibration evidence and readiness promotion"]["status"] == "pass"
    assert requirements["Optimization-to-physics escalation plan"]["status"] == "pass"
