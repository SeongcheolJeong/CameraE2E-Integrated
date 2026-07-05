from __future__ import annotations

import json

from pyisetcam import (
    HWIspConfig,
    camera_create,
    hw_isp_config_from_profile,
    hw_isp_parameter_db,
    hw_isp_profile,
    hw_isp_profile_names,
    hw_isp_simulate_frame,
    scene_create,
)
from tools.collect_hwisp_parameter_db import collect
from tools.render_hwisp_implementation_report import render as render_hwisp_implementation_report
from tools.render_hwisp_parameter_requirements_report import (
    render as render_hwisp_parameter_requirements_report,
)
from tools.render_hwisp_technical_report import render as render_hwisp_technical_report


def test_builtin_hwisp_profiles_are_loadable() -> None:
    names = hw_isp_profile_names()
    profiles = hw_isp_parameter_db()

    assert "generic_1080p_30fps" in names
    assert "rpi_vc4_imx219_public_seed" in names
    assert profiles["generic_1080p_30fps"].config.sensor_timing.fps == 30.0
    assert profiles["rpi_vc4_imx219_public_seed"].calibration["black_level"]["value"] == 4096


def test_hwisp_config_from_profile_accepts_nested_overrides() -> None:
    config = hw_isp_config_from_profile(
        "rpi_vc4_imx219_public_seed",
        sensor_timing={"fps": 60.0, "line_time_us": 7.6},
        control_path={"apply_to_image": True, "target_luma": 0.2},
        transport={"request_queue_depth": 2},
    )

    assert isinstance(config, HWIspConfig)
    assert config.sensor_timing.fps == 60.0
    assert config.sensor_timing.line_time_us == 7.6
    assert config.control_path.apply_to_image is True
    assert config.control_path.target_luma == 0.2
    assert config.transport.request_queue_depth == 2
    assert config.stages


def test_hwisp_profile_config_runs_simulation(asset_store) -> None:
    config = hw_isp_config_from_profile("generic_1080p_30fps")
    result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        scene_create("uniform ee", 8, asset_store=asset_store),
        config,
        asset_store=asset_store,
    )

    assert result.timeline.timestamps_us["app_visible"] > result.timeline.timestamps_us["request"]
    assert result.controls_applied["produced_stats"]["ae"]["stats_grid"] == [8, 8]


def test_collector_writes_normalized_profile_from_libcamera_json(tmp_path) -> None:
    source = tmp_path / "imx219.json"
    source.write_text(
        json.dumps(
            {
                "version": 2.0,
                "target": "bcm2835",
                "algorithms": [
                    {"rpi.black_level": {"black_level": 4096}},
                    {
                        "rpi.lux": {
                            "reference_shutter_speed": 27685,
                            "reference_gain": 1.0,
                            "reference_lux": 998,
                            "reference_Y": 12744,
                        }
                    },
                    {"rpi.noise": {"reference_constant": 0, "reference_slope": 3.67}},
                    {"rpi.geq": {"offset": 204, "slope": 0.01633}},
                    {"rpi.agc": {"y_target": [0, 0.16, 1000, 0.165]}},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "profiles"

    written = collect([source], output_dir, overwrite=True)
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["calibration"]["black_level"]["value"] == 4096
    assert payload["config"]["control_path"]["target_luma"] == 0.16

    profile = hw_isp_profile(payload["name"], db_path=output_dir)
    assert profile.calibration["noise"]["reference_slope"] == 3.67


def test_hwisp_implementation_report_renderer_writes_html(tmp_path) -> None:
    output_dir = tmp_path / "hwisp"
    output_dir.mkdir()
    (output_dir / "timeline_summary.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "frame_count": 8.0,
                    "e2e_latency_mean_us": 12000.0,
                    "e2e_latency_max_us": 13000.0,
                    "queue_stall_total_us": 1000.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "three_a_summary.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "ae_settle_frame": 8.0,
                    "ae_final_error_ev": 0.1,
                    "awb_settle_frame": 10.0,
                    "awb_final_rgb_imbalance": 0.15,
                    "max_clip_fraction_before_response": 0.1,
                    "max_clip_fraction_after_response": 0.05,
                    "validation_verdicts": {
                        "ae_settle": True,
                        "awb_settle": True,
                        "clamp_compliance": True,
                        "warmup_delay_mapping": True,
                        "clip_reduction": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = render_hwisp_implementation_report(output_dir)
    html = outputs["html"].read_text(encoding="utf-8")

    assert outputs["html"].exists()
    assert outputs["summary"].exists()
    assert "HW ISP Implementation And Verification Report" in html
    assert "Architecture" in html
    assert "HW ISP Profile" in html

    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c6360000002000100ffff03000006000557bfab00000000"
        "49454e44ae426082"
    )
    (output_dir / "frame_timeline.png").write_bytes(png_bytes)
    embedded_outputs = render_hwisp_implementation_report(output_dir, embed_images=True)
    embedded_html = embedded_outputs["html"].read_text(encoding="utf-8")

    assert embedded_outputs["html"].name == "implementation_verification_report_integrated.html"
    assert "data:image/png;base64," in embedded_html

    technical_outputs = render_hwisp_technical_report(output_dir)
    technical_html = technical_outputs["html"].read_text(encoding="utf-8")

    assert technical_outputs["html"].name == "hwisp_technical_report.html"
    assert "HW ISP Technical Report" in technical_html
    assert "Sensor Timing Model" in technical_html
    assert "ISP Stage Timing Model" in technical_html

    parameter_outputs = render_hwisp_parameter_requirements_report(output_dir)
    parameter_html = parameter_outputs["html"].read_text(encoding="utf-8")

    assert parameter_outputs["html"].name == "hwisp_parameter_requirements_report.html"
    assert "HW ISP Parameter DB Requirements Report" in parameter_html
    assert "Required DB Parameter Matrix" in parameter_html
    assert "Minimum Sign-off Checklist" in parameter_html
