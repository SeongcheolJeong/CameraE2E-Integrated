from __future__ import annotations

import json
from pathlib import Path

from pyisetcam import (
    camerae2e_run_scenario,
    image_sensor_db_config,
    image_sensor_db_get,
    image_sensor_db_optimize_camera_parameters,
    image_sensor_db_parameters,
    image_sensor_db_records,
    image_sensor_db_summary,
)


def test_image_sensor_db_records_are_searchable() -> None:
    records = image_sensor_db_records(limit=5)

    assert records
    assert {
        "sensor_id",
        "code",
        "manufacturer",
        "stack_config_path",
        "tcad_profile_path",
    } <= set(records[0])
    assert image_sensor_db_records("sony", limit=1)


def test_image_sensor_db_get_accepts_sensor_id_and_code() -> None:
    first = image_sensor_db_records(limit=1)[0]

    by_id = image_sensor_db_get(first["sensor_id"])
    by_code = image_sensor_db_get(first["code"])

    assert by_id["sensor_id"] == first["sensor_id"]
    assert by_code["code"] == first["code"]


def test_image_sensor_db_parameters_return_existing_paths() -> None:
    first = image_sensor_db_records(limit=1)[0]
    params = image_sensor_db_parameters(first["sensor_id"])

    assert Path(params["catalog_path"]).exists()
    assert Path(params["stack_config_path"]).exists()
    assert Path(params["tcad_profile_path"]).exists()
    assert params["lut_path"] is None or Path(params["lut_path"]).exists()
    assert Path(params["generation_map_path"]).exists()
    assert params["collection_summary_paths"]
    assert Path(params["accuracy_gate_path"]).exists()


def test_image_sensor_db_config_returns_json_safe_hybrid_scenario() -> None:
    first = image_sensor_db_records(limit=1)[0]
    config = image_sensor_db_config(first["sensor_id"])

    assert config["schema_version"] == "camerae2e_image_sensor_db_config_v1"
    assert config["strategy"] == "hybrid"
    assert config["scenario"]["sensor"]
    assert "policy" in config
    assert config["policy"]["component_tiers"]["sensor_db_metadata"] == "proxy"
    assert config["fdtd"] is None or "lut" in config["scenario"]["fdtd"]
    assert config["tcad"] is None or "db" in config["scenario"]["tcad"]
    json.dumps(config)


def test_image_sensor_db_config_analytic_only_runs_faca_scenario() -> None:
    first = image_sensor_db_records(limit=1)[0]
    config = image_sensor_db_config(first["sensor_id"], strategy="analytic_only")

    assert "fdtd" not in config["scenario"]
    assert "tcad" not in config["scenario"]
    result = camerae2e_run_scenario(
        {
            "scene": {"type": "uniform ee", "args": [8]},
            **config["scenario"],
            "sensor": {
                **config["scenario"]["sensor"],
                "noise_flag": 0,
            },
        },
        include_arrays=False,
    )

    assert result["schema_version"] == "camerae2e_faca_scenario_v1"
    assert result["parameter_lineage"]


def test_image_sensor_db_config_maps_quad_bayer_to_shared_ocl_proxy() -> None:
    quad_record = next(
        record
        for record in image_sensor_db_records()
        if record.get("cfa_pattern") in {"quad_bayer", "tetracell_bayer"}
    )
    config = image_sensor_db_config(quad_record["sensor_id"], strategy="analytic_only")

    assert config["sensor"]["cfa_preset"] == "quad_bayer_rgb"
    assert config["sensor"]["ocl_group_shape"] == "2x2"
    assert config["sensor"]["ocl_group_equalization"] == 1.0
    assert config["policy"]["analytic_role"].startswith("Analytic proxy axes")


def test_image_sensor_db_optimize_camera_parameters_preserves_source_lineage() -> None:
    first = image_sensor_db_records(limit=1)[0]
    result = image_sensor_db_optimize_camera_parameters(
        first["sensor_id"],
        strategy="analytic_only",
        base_scenario={"scene": {"type": "uniform ee", "args": [8]}},
        preset="exposure",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        max_cases=2,
        seed=17,
        top_k=1,
    )

    source = result["source_image_sensor_db"]
    assert source["schema_version"] == "camerae2e_optimization_source_image_sensor_db_v1"
    assert source["sensor_id"] == first["sensor_id"]
    assert source["strategy"] == "analytic_only"
    assert result["best_case"]["parameters"]["sensor.integration_time"] == 0.004
    assert result["selected_scenarios"][0]["image_sensor_db"]["sensor_id"] == first["sensor_id"]


def test_image_sensor_db_summary_has_counts() -> None:
    summary = image_sensor_db_summary()

    assert summary["record_count"] >= 100
    assert summary["manufacturer_count"] >= 5
    assert "Sony" in summary["manufacturers"]
