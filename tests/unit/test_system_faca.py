from __future__ import annotations

from pyisetcam import camerae2e_faca_report, camerae2e_run_scenario, camerae2e_run_sweep


def test_camerae2e_run_scenario_returns_stage_metrics_and_report() -> None:
    result = camerae2e_run_scenario(
        {
            "name": "unit_faca",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        seed=11,
        include_arrays=False,
    )
    report = camerae2e_faca_report(result)

    assert result["schema_version"] == "camerae2e_faca_scenario_v1"
    assert report["name"] == "unit_faca"
    assert report["stage_summaries"]["sensor_raw"]["available"] is True
    assert report["metrics"]["color"]["rgb_mean"] is not None
    assert report["parameter_lineage"][0]["path"] == "sensor.noise_flag"
    assert "readiness_tiers" in report["artifact_lineage"]["summary"]


def test_camerae2e_run_sweep_records_axis_values() -> None:
    sweep = camerae2e_run_sweep(
        {
            "name": "unit_sweep",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.002, 0.004]},
        seed=20,
        include_arrays=False,
    )

    assert sweep["schema_version"] == "camerae2e_faca_sweep_v1"
    assert sweep["case_count"] == 2
    assert sweep["cases"][0]["axis_values"]["sensor.integration_time"] == 0.002
    assert sweep["cases"][1]["seed"] == 21


def test_camerae2e_run_scenario_can_include_hw_isp_control_summary() -> None:
    result = camerae2e_run_scenario(
        {
            "name": "unit_hw_faca",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
            "hw_isp": {"enabled": True, "nframes": 2},
        },
        seed=3,
        include_arrays=False,
    )
    report = camerae2e_faca_report(result)

    assert report["metrics"]["control"]["frame_count"] == 2
    assert report["stage_summaries"]["ip_result"]["available"] is True


def test_camerae2e_run_sweep_enables_hw_isp_delay_axes() -> None:
    sweep = camerae2e_run_sweep(
        {
            "name": "unit_hw_delay_sweep",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"hw_isp.ae_apply_delay_frames": [0, 2]},
        seed=7,
        include_arrays=False,
    )

    assert sweep["case_count"] == 2
    assert sweep["cases"][0]["metrics"]["control"]["frame_count"] == 3
    assert sweep["cases"][0]["scenario"]["hw_isp"]["enabled"] is True
    assert sweep["cases"][1]["axis_values"]["hw_isp.ae_apply_delay_frames"] == 2
