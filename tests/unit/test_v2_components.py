from __future__ import annotations

import json
from pathlib import Path

import pytest

from camerae2e_v2.components import ComponentCatalogService
from camerae2e_v2.engine import CameraEngine
from camerae2e_v2.models import (
    ComponentSelection,
    FidelityLevel,
    JobStatus,
    ModuleBaselineRequest,
    ModuleCompareRequest,
    StudyCreate,
)
from camerae2e_v2.service import CameraE2EService

FRAME_SENSOR_ID = "def_2103_802_stmicroelectronics_vg6640ab1m"
RAY_FRAME_SENSOR_ID = "def_2112_801_omnivision_0x01f10_e48y"
EVENT_SENSOR_ID = "def_2202_802_sony_imx636aamr_es"
DEFAULT_LENS_ID = "p0413:base"
LENS_A_ID = "p0410:base"
LENS_B_ID = "p0411:base"


def test_component_catalog_search_filters_simulation_ready_records() -> None:
    catalog = ComponentCatalogService()

    lenses = catalog.search_lenses(query="US9759892B2", require_psf=True, page_size=200)
    sensors = catalog.search_sensors(
        manufacturer="STMicroelectronics",
        simulation_ready=True,
        page_size=200,
    )

    assert lenses["total"] >= 1
    assert any(item["id"] == LENS_A_ID for item in lenses["items"])
    assert sensors["total"] >= 1
    assert all(item["module_configurable"] for item in sensors["items"])
    assert all(item["frame_model_supported"] for item in sensors["items"])
    assert catalog.search_lenses(page_size=1)["facets"]["psf_count"] > 0


def test_component_compare_requires_two_to_four_candidates() -> None:
    with pytest.raises(ValueError):
        ModuleCompareRequest(
            candidates=[
                ComponentSelection(lens_id=DEFAULT_LENS_ID, sensor_id=FRAME_SENSOR_ID)
            ]
        )


def test_component_catalog_does_not_treat_event_sensor_as_frame_raw() -> None:
    catalog = ComponentCatalogService()
    selection = ComponentSelection(lens_id="p0410:base", sensor_id=EVENT_SENSOR_ID)

    result = catalog.evaluate_modules(StudyCreate().requirements, [selection])["candidates"][0]

    assert result["status"] == "incompatible"
    assert {"frame_sensor_model", "cfa_model"}.issubset(result["failed_gate_ids"])
    assert not result["sensor"]["module_configurable"]
    assert "frame_sensor_model_unsupported" in result["sensor"]["configuration_blockers"]


def test_component_module_preserves_native_geometry_and_caps_simulation_readout() -> None:
    catalog = ComponentCatalogService()
    spec = StudyCreate()
    selection = ComponentSelection(
        lens_id=DEFAULT_LENS_ID,
        sensor_id=FRAME_SENSOR_ID,
        use_geometric_psf=False,
    )

    module, compatibility = catalog.build_module(
        selection,
        spec.baseline,
        spec.requirements,
    )

    assert compatibility["status"] == "compatible"
    assert module.sensor.native_rows == 854
    assert module.sensor.native_cols == 1520
    assert module.sensor.rows <= 360
    assert module.sensor.cols <= 640
    assert module.sensor.simulation_pixel_size_um > module.sensor.pixel_size_um
    assert module.sensor.simulation_fill_factor < module.sensor.pixel_fill_factor
    native_area = module.sensor.pixel_size_um**2 * module.sensor.pixel_fill_factor
    simulation_area = (
        module.sensor.simulation_pixel_size_um**2
        * module.sensor.simulation_fill_factor
    )
    assert simulation_area == pytest.approx(native_area)
    simulation_width = module.sensor.cols * module.sensor.simulation_pixel_size_um
    native_width = module.sensor.native_cols * module.sensor.pixel_size_um
    assert simulation_width == pytest.approx(native_width)
    assert module.metadata["native_sensor_size_rc"] == [854, 1520]
    assert module.metadata["simulation_geometry_policy"] == (
        "full_extent_sparse_sampling_proxy"
    )
    assert module.metadata["sensor_qe_policy"].startswith("existing_baseline_profile")


def test_component_module_attaches_real_geometric_rayoptics_psf() -> None:
    catalog = ComponentCatalogService()
    spec = StudyCreate()
    requirements = spec.requirements.model_copy(
        update={"hfov": spec.requirements.hfov.model_copy(update={"value": 64.0})}
    )
    module, _compatibility = catalog.build_module(
        ComponentSelection(
            lens_id=LENS_A_ID,
            sensor_id=RAY_FRAME_SENSOR_ID,
            use_geometric_psf=True,
        ),
        spec.baseline,
        requirements,
    )
    engine = CameraEngine()
    module = engine.apply_module_overrides(
        module,
        {"sensor.rows": 48, "sensor.cols": 96},
    )

    result = engine.evaluate(
        module,
        spec.scenes[0],
        fidelity=FidelityLevel.ANALYTIC,
        policy=spec.fidelity_policy,
        seed=7,
        include_arrays=False,
    )

    assert result["scenario"]["rayoptics"]["simulation_id"] == LENS_A_ID
    scenario_sensor = result["scenario"]["sensor"]
    simulated_width_mm = scenario_sensor["pixel_size"] * module.sensor.cols * 1e3
    native_width_mm = (
        module.sensor.native_cols * module.sensor.pixel_size_um * 1e-3
    )
    assert simulated_width_mm == pytest.approx(native_width_mm)
    assert scenario_sensor["pixel_fill_factor"] == pytest.approx(
        module.sensor.simulation_fill_factor
    )
    assert "geometric" in result["truth_boundary"].lower()
    assert "diffraction" in result["truth_boundary"].lower()


def test_component_baseline_compare_and_report_are_persisted(tmp_path: Path) -> None:
    service = CameraE2EService(tmp_path / "projects", max_workers=1)
    project = service.create_project("Component study", preset="general")
    study = project.store.list_studies()[0]
    requirements = study.spec.requirements.model_copy(
        update={"hfov": study.spec.requirements.hfov.model_copy(update={"value": 64.0})}
    )
    study = project.update_study(
        study.id,
        study.spec.model_copy(update={"requirements": requirements}),
    )
    first = ComponentSelection(
        lens_id=LENS_A_ID,
        sensor_id=RAY_FRAME_SENSOR_ID,
        use_geometric_psf=False,
    )
    second = ComponentSelection(
        lens_id=LENS_B_ID,
        sensor_id=RAY_FRAME_SENSOR_ID,
        use_geometric_psf=False,
    )

    applied = service.apply_component_module(
        project.info.id,
        study.id,
        ModuleBaselineRequest(selection=first),
    )
    comparison = service.execute_job_now(
        project.info.id,
        study.id,
        "compare_modules",
        {"candidates": [first.model_dump(mode="json"), second.model_dump(mode="json")]},
    )
    report = service.execute_job_now(project.info.id, study.id, "report", {})

    assert applied["study"]["revision"] == study.revision + 1
    assert applied["artifact"]["artifact_type"] == "camera_module_descriptor"
    assert comparison.status == JobStatus.SUCCEEDED
    assert comparison.result and len(comparison.result["evaluations"]) == 2
    assert all(
        item["evaluation"]["preview_artifact_hash"]
        for item in comparison.result["evaluations"]
    )
    assert report.status == JobStatus.SUCCEEDED
    payload = json.loads((Path(report.result["report_dir"]) / "report.json").read_text())
    report_html = (Path(report.result["report_dir"]) / "report.html").read_text()
    assert payload["module_comparison"]["schema_version"] == (
        "camerae2e_component_module_comparison_v1"
    )
    assert "No automatic winner" in payload["module_comparison"]["truth_boundary"]
    assert "Camera module comparison" in report_html
    assert LENS_A_ID in report_html
    service.shutdown()
