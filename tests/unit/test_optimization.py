from __future__ import annotations

import numpy as np

from pyisetcam import (
    camerae2e_faca_report,
    camerae2e_optimization_config_catalog,
    camerae2e_optimization_escalation_plan,
    camerae2e_optimization_report,
    camerae2e_optimize_camera_parameters,
    camerae2e_optimize_parameters,
    camerae2e_parameter_candidate_plan,
    camerae2e_parameter_space_catalog,
    camerae2e_parameter_space_validate,
    camerae2e_pareto_front,
    camerae2e_run_scenario,
)


def test_camerae2e_scenario_applies_generic_camera_parameters() -> None:
    low = camerae2e_faca_report(
        camerae2e_run_scenario(
            {
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
                "parameters": {"sensor.integration_time": 0.001},
            },
            include_arrays=False,
        )
    )
    high = camerae2e_faca_report(
        camerae2e_run_scenario(
            {
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
                "parameters": {"sensor.integration_time": 0.004},
            },
            include_arrays=False,
        )
    )

    assert high["metrics"]["color"]["rgb_mean"] > low["metrics"]["color"]["rgb_mean"]
    assert high["parameter_lineage"][0]["path"] == "sensor.integration_time"
    assert high["parameter_lineage"][0]["status"] == "applied"


def test_camerae2e_parameter_space_catalog_exposes_presets() -> None:
    catalog = camerae2e_parameter_space_catalog()
    raw_factory = camerae2e_parameter_space_catalog("raw_factory")

    assert catalog["schema_version"] == "camerae2e_parameter_space_catalog_v1"
    assert "sensor.integration_time" in catalog["axes"]
    assert "sensor_geometry" in catalog["presets"]
    assert "raw_factory" in catalog["presets"]
    assert set(raw_factory["parameter_space"]) == {
        "sensor.integration_time",
        "sensor.analog_gain",
        "optics.fnumber",
    }


def test_camerae2e_optimization_config_catalog_lists_configure_targets() -> None:
    catalog = camerae2e_optimization_config_catalog()

    assert catalog["schema_version"] == "camerae2e_optimization_config_catalog_v1"
    assert catalog["registered_axis_count"] >= 20
    assert "sensor.integration_time" in catalog["registered_axes"]
    assert "sensor.pixel_size" in catalog["registered_axes"]
    assert "sensor.cfa_pattern" in catalog["registered_axes"]
    assert "sensor.cfa_preset" in catalog["registered_axes"]
    assert "sensor.binning_factor" in catalog["registered_axes"]
    assert "sensor.ocl_vignetting" in catalog["registered_axes"]
    assert "sensor.ocl_group_shape" in catalog["registered_axes"]
    assert "sensor.ocl_group_equalization" in catalog["registered_axes"]
    assert "sensor.ocl_fnumber" in catalog["registered_axes"]
    assert "optics.si_psf_radius_um" in catalog["registered_axes"]
    assert "fdtd.ocl_shift_um" in catalog["registered_axes"]
    assert "fdtd.crosstalk_strength" in catalog["registered_axes"]
    assert "hw_isp.global_latency_factor" in catalog["registered_axes"]
    assert "metrics.artifact.raw_std" in catalog["objective_metrics"]
    assert any(
        rule["path_pattern"] == "<camera_set dot path>"
        for rule in catalog["custom_path_rules"]
    )
    sensor_rule = next(
        rule for rule in catalog["custom_path_rules"] if rule["path_pattern"] == "sensor.<name>"
    )
    assert "exposure_time" in sensor_rule["allowed_suffixes"]
    assert "pixel_size" in sensor_rule["allowed_suffixes"]
    assert "cfa_preset" in sensor_rule["allowed_suffixes"]
    assert "binning_method" in sensor_rule["allowed_suffixes"]
    assert "binning_factor" in sensor_rule["allowed_suffixes"]
    assert "ocl_vignetting" in sensor_rule["allowed_suffixes"]
    assert "ocl_group_shape" in sensor_rule["allowed_suffixes"]
    assert "ocl_group_equalization" in sensor_rule["allowed_suffixes"]


def test_camerae2e_parameter_space_validate_classifies_axes() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.integration_time": [0.001, 0.004],
            "optics.focal_length": [0.002, 0.004],
            "hw_isp.ae_apply_delay_frames": [0, 2],
            "optics.transmittance_scale": [[1, 1, 1]],
        }
    )

    assert validation["ok"] is True
    assert validation["axes"]["sensor.integration_time"]["status"] == "registered"
    assert validation["axes"]["hw_isp.ae_apply_delay_frames"]["status"] == "registered"
    assert validation["axes"]["optics.focal_length"]["status"] == "registered"
    assert validation["axes"]["optics.transmittance_scale"]["status"] == "custom_passthrough"
    assert validation["warning_count"] == 1


def test_camerae2e_parameter_space_validate_reports_ineffective_axes() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.not_a_real_parameter": [1],
            "fdtd.mode": ["qe"],
            "tcad.collection_mode": ["collection"],
        }
    )
    fdtd_ready = camerae2e_parameter_space_validate(
        {"fdtd.mode": ["qe"]},
        base_scenario={"fdtd": {"lut": "unit_lut.json"}},
    )

    assert validation["ok"] is False
    assert {issue["kind"] for issue in validation["issues"]} == {
        "unsupported_sensor_axis",
        "inactive_fdtd_axis",
        "inactive_tcad_axis",
    }
    assert fdtd_ready["ok"] is True


def test_camerae2e_parameter_space_validate_reports_invalid_values() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.pixel_size": [0.0],
            "sensor.pixel_fill_factor": [1.5],
            "sensor.n_samples_per_pixel": [2],
            "sensor.cfa_pattern": [[[1.2, 2], [2, 3]]],
            "sensor.cfa_preset": ["unsupported"],
            "sensor.binning_method": ["unsupported"],
            "sensor.binning_factor": [4],
            "sensor.ocl_vignetting": ["unsupported"],
            "sensor.ocl_group_shape": ["3x3"],
            "sensor.ocl_group_equalization": [1.5],
            "sensor.ocl_fnumber": [0.0],
            "optics.si_psf_radius_um": [-1.0],
        }
    )

    assert validation["ok"] is False
    issue_kinds = {issue["kind"] for issue in validation["issues"]}
    assert {
        "invalid_pixel_size",
        "invalid_fill_factor",
        "unsupported_samples_per_pixel",
        "invalid_cfa_pattern",
        "unsupported_enum_value",
        "unsupported_binning_factor",
        "unsupported_ocl_group_shape",
        "invalid_fraction_value",
        "invalid_positive_value",
    } <= issue_kinds
    assert validation["axes"]["sensor.n_samples_per_pixel"]["value_issues"][0][
        "kind"
    ] == "unsupported_samples_per_pixel"


def test_camerae2e_parameter_space_validate_reports_value_warnings() -> None:
    validation = camerae2e_parameter_space_validate(
        {
            "sensor.cfa_pattern": [[[1, 4], [2, 3]]],
            "fdtd.crosstalk_strength": [1.5],
        },
        base_scenario={"fdtd": {"lut": "unit_lut.json"}},
    )

    assert validation["ok"] is True
    assert {warning["kind"] for warning in validation["warnings"]} == {
        "cfa_filter_dependency",
        "extrapolating_crosstalk_strength",
    }


def test_camerae2e_parameter_candidate_plan_supports_budgeted_grid() -> None:
    plan = camerae2e_parameter_candidate_plan(
        {
            "sensor.integration_time": [0.001, 0.002, 0.004],
            "sensor.analog_gain": [1.0, 2.0, 4.0],
        },
        max_cases=4,
    )

    assert plan["schema_version"] == "camerae2e_parameter_candidate_plan_v1"
    assert plan["ok"] is True
    assert plan["method"] == "grid"
    assert plan["full_grid_count"] == 9
    assert plan["case_count"] == 4
    assert plan["truncated"] is True
    assert plan["candidates"][0] == {
        "sensor.integration_time": 0.001,
        "sensor.analog_gain": 1.0,
    }
    assert plan["candidates"][-1] == {
        "sensor.integration_time": 0.004,
        "sensor.analog_gain": 4.0,
    }


def test_camerae2e_parameter_candidate_plan_random_is_seeded() -> None:
    parameter_space = {
        "sensor.integration_time": [0.001, 0.002, 0.004],
        "sensor.analog_gain": [1.0, 2.0, 4.0],
        "optics.fnumber": [2.0, 2.8, 4.0],
    }
    left = camerae2e_parameter_candidate_plan(
        parameter_space,
        method="random",
        max_cases=5,
        seed=123,
    )
    right = camerae2e_parameter_candidate_plan(
        parameter_space,
        method="random",
        max_cases=5,
        seed=123,
    )

    assert left["method"] == "random"
    assert left["case_count"] == 5
    assert left["candidates"] == right["candidates"]
    assert len({str(candidate) for candidate in left["candidates"]}) == 5


def test_camerae2e_parameter_candidate_plan_latin_hypercube_defaults_budget() -> None:
    plan = camerae2e_parameter_candidate_plan(
        {
            "sensor.integration_time": [0.001, 0.002, 0.004],
            "sensor.analog_gain": [1.0, 2.0, 4.0],
            "optics.fnumber": [2.0, 2.8, 4.0],
            "sensor.pixel_fill_factor": [0.55, 0.75, 0.95],
        },
        method="lhs",
        seed=77,
    )

    assert plan["method"] == "latin_hypercube"
    assert plan["implicit_default_budget"] is True
    assert plan["max_cases"] == 32
    assert plan["case_count"] == 32
    assert plan["truncated"] is True
    assert plan["warnings"][0]["kind"] == "implicit_default_budget"


def test_camerae2e_parameter_candidate_plan_evolutionary_returns_seed_population() -> None:
    plan = camerae2e_parameter_candidate_plan(
        {
            "sensor.integration_time": [0.001, 0.002, 0.004],
            "sensor.analog_gain": [1.0, 2.0, 4.0],
        },
        method="genetic",
        max_cases=6,
        seed=12,
    )

    assert plan["method"] == "evolutionary"
    assert plan["max_cases"] == 6
    assert plan["case_count"] == 4
    assert plan["search_config"]["population_size"] == 4
    assert plan["warnings"][0]["kind"] == "evolutionary_seed_plan"
    assert plan["candidates"][0] == {
        "sensor.integration_time": 0.001,
        "sensor.analog_gain": 1.0,
    }
    assert plan["candidates"][1] == {
        "sensor.integration_time": 0.004,
        "sensor.analog_gain": 4.0,
    }


def test_camerae2e_parameter_candidate_plan_surrogate_returns_seed_population() -> None:
    plan = camerae2e_parameter_candidate_plan(
        {
            "sensor.integration_time": [0.001, 0.002, 0.004],
            "sensor.analog_gain": [1.0, 2.0, 4.0],
        },
        method="surrogate",
        max_cases=6,
        seed=14,
    )

    assert plan["method"] == "surrogate"
    assert plan["max_cases"] == 6
    assert plan["case_count"] == 4
    assert plan["search_config"]["seed_count"] == 4
    assert "not a calibrated Gaussian-process" in plan["search_config"]["truth_boundary"]
    assert plan["warnings"][0]["kind"] == "surrogate_seed_plan"


def test_camerae2e_parameter_candidate_plan_gaussian_process_returns_seed_population() -> None:
    plan = camerae2e_parameter_candidate_plan(
        {
            "sensor.integration_time": [0.001, 0.002, 0.004],
            "sensor.analog_gain": [1.0, 2.0, 4.0],
        },
        method="bayesian",
        max_cases=6,
        seed=15,
    )

    assert plan["method"] == "gaussian_process"
    assert plan["max_cases"] == 6
    assert plan["case_count"] == 5
    assert plan["search_config"]["seed_count"] == 5
    assert plan["search_config"]["model"] == "gaussian_process_rbf"
    assert plan["search_config"]["acquisition"] == "expected_improvement"
    assert plan["warnings"][0]["kind"] == "gaussian_process_seed_plan"


def test_camerae2e_optimize_parameters_rejects_invalid_values_before_run() -> None:
    try:
        camerae2e_optimize_parameters(
            {"scene": {"type": "uniform ee", "args": [8]}},
            {"sensor.n_samples_per_pixel": [2]},
        )
    except ValueError as exc:
        assert "positive odd integer" in str(exc)
    else:
        raise AssertionError("Expected invalid value to fail before optimization.")


def test_camerae2e_optimize_parameters_uses_budgeted_candidates() -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_budgeted_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {
            "sensor.integration_time": [0.001, 0.002, 0.004],
            "sensor.analog_gain": [1.0, 2.0, 4.0],
        },
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="latin_hypercube",
        max_cases=4,
        seed=200,
        top_k=2,
    )

    assert result["method"] == "budgeted_latin_hypercube"
    assert result["search_method"] == "latin_hypercube"
    assert result["candidate_plan"]["case_count"] == 4
    assert result["candidate_plan"]["truncated"] is True
    assert result["case_count"] == 4
    assert len(result["cases"]) == 4


def test_camerae2e_optimize_parameters_supports_evolutionary_search() -> None:
    parameter_space = {
        "sensor.integration_time": [0.001, 0.002, 0.004],
        "sensor.analog_gain": [1.0, 2.0, 4.0],
    }
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_evolutionary_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        parameter_space,
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="evolutionary",
        max_cases=5,
        seed=210,
        top_k=2,
    )
    repeated = camerae2e_optimize_parameters(
        {
            "name": "unit_evolutionary_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        parameter_space,
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="evolutionary",
        max_cases=5,
        seed=210,
        top_k=2,
    )

    assert result["method"] == "budgeted_evolutionary"
    assert result["search_method"] == "evolutionary"
    assert result["case_count"] == 5
    assert result["candidate_plan"]["case_count"] == 5
    assert result["candidate_plan"]["search_trace"]
    assert result["candidate_plan"]["executed_generation_count"] >= 1
    assert [case["parameters"] for case in result["cases"]] == [
        case["parameters"] for case in repeated["cases"]
    ]
    assert len({str(case["parameters"]) for case in result["cases"]}) == 5


def test_camerae2e_optimize_parameters_supports_surrogate_search() -> None:
    parameter_space = {
        "sensor.integration_time": [0.001, 0.002, 0.004],
        "sensor.analog_gain": [1.0, 2.0, 4.0],
        "optics.fnumber": [2.0, 2.8, 4.0],
    }
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_surrogate_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        parameter_space,
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="surrogate",
        max_cases=6,
        seed=211,
        top_k=2,
    )
    repeated = camerae2e_optimize_parameters(
        {
            "name": "unit_surrogate_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        parameter_space,
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="surrogate",
        max_cases=6,
        seed=211,
        top_k=2,
    )

    assert result["method"] == "budgeted_surrogate"
    assert result["search_method"] == "surrogate"
    assert result["case_count"] == 6
    assert result["candidate_plan"]["case_count"] == 6
    assert result["candidate_plan"]["search_config"]["model"] == (
        "discrete_rbf_inverse_distance_surrogate"
    )
    assert result["candidate_plan"]["search_trace"]
    assert result["candidate_plan"]["executed_generation_count"] >= 1
    assert [case["parameters"] for case in result["cases"]] == [
        case["parameters"] for case in repeated["cases"]
    ]
    assert len({str(case["parameters"]) for case in result["cases"]}) == 6


def test_camerae2e_optimize_parameters_supports_gaussian_process_search() -> None:
    parameter_space = {
        "sensor.integration_time": [0.001, 0.002, 0.004],
        "sensor.analog_gain": [1.0, 2.0, 4.0],
        "optics.fnumber": [2.0, 2.8, 4.0],
    }
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_gp_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        parameter_space,
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="bayesian",
        max_cases=7,
        seed=213,
        top_k=2,
    )
    repeated = camerae2e_optimize_parameters(
        {
            "name": "unit_gp_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        parameter_space,
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="gp",
        max_cases=7,
        seed=213,
        top_k=2,
    )

    assert result["method"] == "budgeted_gaussian_process"
    assert result["search_method"] == "gaussian_process"
    assert result["case_count"] == 7
    assert result["candidate_plan"]["case_count"] == 7
    assert result["candidate_plan"]["search_config"]["model"] == "gaussian_process_rbf"
    assert result["candidate_plan"]["search_config"]["acquisition"] == "expected_improvement"
    assert result["candidate_plan"]["search_trace"]
    assert result["candidate_plan"]["executed_generation_count"] >= 1
    assert result["candidate_plan"]["search_trace"][-1]["model"] == "gaussian_process_rbf"
    assert [case["parameters"] for case in result["cases"]] == [
        case["parameters"] for case in repeated["cases"]
    ]
    assert len({str(case["parameters"]) for case in result["cases"]}) == 7


def test_camerae2e_optimization_escalation_plan_maps_proxy_axes_to_physics() -> None:
    result = camerae2e_optimize_camera_parameters(
        {
            "name": "unit_escalation_source",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        preset="exposure",
        parameter_space={
            "sensor.ocl_group_shape": ["1x1", "2x2"],
            "sensor.ocl_group_equalization": [0.0, 1.0],
            "optics.si_psf_radius_um": [1.0, 2.0],
        },
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        method="latin_hypercube",
        max_cases=4,
        seed=212,
        top_k=2,
    )
    physics_plan = {
        "schema_version": "camerae2e_physics_pipeline_plan_v1",
        "ok": True,
        "summary": {"stale_dependency_count": 0},
        "active_runs": {"lineage_match": True},
        "actions": [
            {
                "entry": "fdtd_sensor_stack_catalog",
                "kind": "proxy_truth_boundary",
                "severity": "info",
                "blocks_strict_validation": True,
            },
            {
                "entry": "fdtd_sensor_lut_active",
                "kind": "proxy_truth_boundary",
                "severity": "info",
                "blocks_strict_validation": True,
            },
            {
                "entry": "tcad_sensor_db_active",
                "kind": "calibration_required",
                "severity": "warning",
                "blocks_strict_validation": True,
            },
            {
                "entry": "lens_patents_active",
                "kind": "proxy_truth_boundary",
                "severity": "info",
                "blocks_strict_validation": True,
            },
        ],
    }

    plan = camerae2e_optimization_escalation_plan(
        result,
        selection="top",
        max_cases=2,
        physics_pipeline_plan=physics_plan,
    )
    stages = {stage["stage_id"]: stage for stage in plan["stages"]}

    assert plan["schema_version"] == "camerae2e_optimization_escalation_plan_v1"
    assert plan["ok"] is True
    assert plan["selected_case_count"] == 2
    assert plan["axis_summary"]["needs"]["fdtd_optical_lut"] is True
    assert plan["axis_summary"]["needs"]["tcad_collection"] is True
    assert plan["axis_summary"]["needs"]["rayoptics_geometric_psf"] is True
    assert stages["fdtd_optical_lut_batch"]["status"] == "ready_proxy"
    assert stages["tcad_collection_batch"]["status"] == "needs_calibration_required"
    assert stages["rayoptics_geometric_psf_batch"]["status"] == "ready_proxy"
    assert plan["validation_jobs"][0]["registry_entries"] == [
        "fdtd_sensor_stack_catalog",
        "fdtd_sensor_lut_active",
        "tcad_sensor_db_active",
        "lens_patents_active",
    ]

    compact_plan = camerae2e_optimization_escalation_plan(
        camerae2e_optimization_report(result),
        selection="top",
        max_cases=1,
        physics_pipeline_plan=physics_plan,
    )
    assert compact_plan["selected_case_count"] == 1
    assert compact_plan["axis_summary"]["axis_count"] >= 3


def test_camerae2e_scenario_applies_extended_configure_axes() -> None:
    result = camerae2e_run_scenario(
        {
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {
                "noise_flag": 0,
                "pixel_size": 3.75e-6,
                "pixel_fill_factor": 0.55,
                "cfa_pattern": [[1, 2], [2, 3]],
                "binning_method": "kodak2008",
                "binning_factor": 2,
                "ocl_fnumber": 1.8,
                "ocl_focal_length_um": 1.4,
                "ocl_refractive_index": 1.55,
                "ocl_vignetting": "centered",
            },
            "parameters": {"optics.si_psf_radius_um": 2.0},
        },
        include_arrays=False,
    )
    lineage = result["parameter_lineage"]

    assert any(item["path"] == "sensor.pixel_size" for item in lineage)
    assert any(item["path"] == "sensor.pixel_fill_factor" for item in lineage)
    assert any(item["path"] == "sensor.cfa_pattern" for item in lineage)
    assert any(item["path"] == "sensor.binning_method" for item in lineage)
    assert any(item["path"] == "sensor.binning_factor" for item in lineage)
    assert any(item["path"] == "sensor.ocl_fnumber" for item in lineage)
    assert any(item["path"] == "sensor.ocl_focal_length_um" for item in lineage)
    assert any(item["path"] == "sensor.ocl_refractive_index" for item in lineage)
    assert any(item["path"] == "sensor.ocl_vignetting" for item in lineage)
    assert any(item["path"] == "optics.si_psf_radius_um" for item in lineage)
    assert result["camera"].fields["sensor"].fields["sensor_compute_method"] == {
        "name": "binning",
        "method": "kodak2008",
        "factor": 2,
    }
    assert result["camera"].fields["sensor"].fields["vignetting"] == 2
    assert result["camera"].fields["sensor"].fields["etendue"] is not None


def test_camerae2e_scenario_applies_quad_bayer_ocl_group_proxy() -> None:
    result = camerae2e_run_scenario(
        {
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {
                "noise_flag": 0,
                "cfa_preset": "quad_bayer_rgb",
                "ocl_group_shape": "2x2",
                "ocl_group_equalization": 1.0,
            },
        },
        include_arrays=False,
    )
    sensor = result["camera"].fields["sensor"]
    expected_quad_bayer = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]],
        dtype=int,
    )

    np.testing.assert_array_equal(sensor.fields["pattern"], expected_quad_bayer)
    assert sensor.fields["ocl_group_proxy"]["shape"] == (2, 2)
    assert sensor.fields["ocl_group_proxy"]["equalization"] == 1.0
    assert sensor.fields["ocl_group_proxy"]["mode"] == "uniform"
    assert any(item["path"] == "sensor.cfa_preset" for item in result["parameter_lineage"])
    assert any(item["path"] == "sensor.ocl_group_shape" for item in result["parameter_lineage"])
    assert any(
        item["path"] == "sensor.ocl_group_equalization"
        for item in result["parameter_lineage"]
    )


def test_camerae2e_optimize_parameters_selects_best_grid_case() -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {
            "sensor.integration_time": [0.001, 0.004],
            "optics.fnumber": [2.8, 4.0],
        },
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        seed=100,
        top_k=2,
    )
    report = camerae2e_optimization_report(result)

    assert result["schema_version"] == "camerae2e_parameter_optimization_v1"
    assert result["case_count"] == 4
    assert result["feasible_count"] == 4
    assert result["pareto_case_count"] == 1
    assert report["best_case"]["parameters"]["sensor.integration_time"] == 0.004
    assert result["parameter_space_validation"]["ok"] is True
    assert report["selected_scenarios"][0]["sensor"]["integration_time"] == 0.004
    assert len(report["top_cases"]) == 2


def test_camerae2e_optimize_camera_parameters_uses_automation_preset() -> None:
    result = camerae2e_optimize_camera_parameters(
        {
            "name": "unit_auto_parameter_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        preset="exposure",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        seed=50,
        top_k=1,
    )

    assert result["automation"]["preset"] == "exposure"
    assert result["best_case"]["parameters"]["sensor.integration_time"] == 0.004
    assert "sensor.analog_gain" in result["automation"]["axes"]


def test_camerae2e_optimize_parameters_supports_constraints() -> None:
    result = camerae2e_optimize_parameters(
        {
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.001, 0.004]},
        {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        constraints=[
            {"metric": "metrics.artifact.raw_std", "op": "<=", "value": 0.002},
        ],
        seed=10,
    )

    assert result["feasible_count"] == 1
    assert result["best_case"]["parameters"]["sensor.integration_time"] == 0.001


def test_camerae2e_optimize_parameters_reports_pareto_front_for_tradeoffs() -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "unit_pareto_optimization",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.001, 0.004]},
        [
            {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
            {"metric": "metrics.artifact.raw_std", "direction": "minimize"},
        ],
        seed=30,
    )
    pareto = camerae2e_pareto_front(result)
    parameter_values = {case["parameters"]["sensor.integration_time"] for case in pareto}

    assert result["pareto_case_count"] == 2
    assert parameter_values == {0.001, 0.004}
    assert all("scenario" in case for case in pareto)
    assert all("objective_utilities" in case for case in pareto)


def test_camerae2e_optimize_parameters_rejects_ineffective_sensor_axis() -> None:
    try:
        camerae2e_optimize_parameters(
            {"scene": {"type": "uniform ee", "args": [8]}},
            {"sensor.not_a_real_parameter": [1, 2]},
        )
    except ValueError as exc:
        assert "Unsupported sensor optimization parameter" in str(exc)
    else:
        raise AssertionError("Expected unsupported sensor axis to fail before optimization.")
