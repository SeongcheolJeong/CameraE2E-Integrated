from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pyisetcam import camerae2e_ppt_gap_audit, cameraE2EPPTGapAudit


def test_camerae2e_ppt_gap_audit_writes_reports_and_classifies_gaps(
    tmp_path: Path,
) -> None:
    pptx_path = tmp_path / "overview.pptx"
    with ZipFile(pptx_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", "<p:sld/>")
        archive.writestr("ppt/slides/slide2.xml", "<p:sld/>")
        archive.writestr("ppt/media/image1.png", b"not-a-real-png")

    audit = camerae2e_ppt_gap_audit(pptx_path, output_dir=tmp_path / "reports")
    alias_audit = cameraE2EPPTGapAudit(pptx_path)

    assert audit["schema_version"] == "camerae2e_ppt_gap_audit_v1"
    assert audit["source"]["status"] == "loaded"
    assert audit["source"]["slide_count"] == 2
    assert audit["source"]["media_count"] == 1
    assert audit["source"]["sha256"].startswith("sha256:")
    assert audit["summary"]["claim_count"] >= 18
    assert audit["summary"]["gap_count"] >= 10
    assert audit["summary"]["status_counts"]["blocked_external_evidence"] >= 1
    assert Path(audit["reports"]["json"]).exists()
    assert Path(audit["reports"]["html"]).exists()
    assert alias_audit["summary"] == audit["summary"]

    gap_areas = {item["area"] for item in audit["not_implemented_or_lacking"]}
    assert "DNG read/write" in gap_areas
    assert "FDTD/TCAD lineage" in gap_areas
    assert "HW ISP timing" in gap_areas
    assert "RAW data formats" in gap_areas

    rendered = json.loads(Path(audit["reports"]["json"]).read_text(encoding="utf-8"))
    assert rendered["truth_boundary"].startswith("This audit maps")


def test_camerae2e_ppt_gap_audit_handles_missing_source(tmp_path: Path) -> None:
    audit = camerae2e_ppt_gap_audit(tmp_path / "missing.pptx")

    assert audit["source"]["status"] == "missing"
    assert audit["source"]["slide_count"] == 0
    assert audit["summary"]["gap_count"] > 0
