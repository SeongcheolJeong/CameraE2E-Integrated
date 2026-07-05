from __future__ import annotations

import json

import numpy as np

from pyisetcam import (
    HWIspStage,
    camera_compute,
    camera_create,
    camera_get,
    hwISPConfig,
    hw_isp_config,
    hw_isp_export_json,
    hw_isp_latency_summary,
    hw_isp_simulate_frame,
    hw_isp_simulate_sequence,
    hw_isp_timeline_table,
    ip_get,
    scene_create,
)
from tools.render_hwisp_timeline_report import render as render_hwisp_timeline_report


def _small_scene(asset_store):
    return scene_create("uniform ee", 8, asset_store=asset_store)


def _scaled_scene(scene, scale: float):
    result = scene.clone()
    result.data["photons"] = np.asarray(result.data["photons"], dtype=float) * float(scale)
    return result


def _warm_scene(scene):
    result = scene.clone()
    wave = np.asarray(result.fields["wave"], dtype=float)
    spectral_tilt = np.interp(wave, [float(np.min(wave)), float(np.max(wave))], [0.4, 2.2])
    result.data["photons"] = np.asarray(result.data["photons"], dtype=float) * spectral_tilt.reshape(1, 1, -1)
    return result


def _off_center_highlight_scene(scene, *, base_scale: float = 1.0, highlight_scale: float = 80.0):
    result = scene.clone()
    photons = np.asarray(result.data["photons"], dtype=float).copy() * float(base_scale)
    rows, cols = photons.shape[:2]
    photons[: max(rows // 4, 1), : max(cols // 4, 1), :] *= float(highlight_scale)
    result.data["photons"] = photons
    return result


def test_default_config_and_alias_are_deterministic() -> None:
    config = hw_isp_config(fps=60.0, seed=7)
    alias_config = hwISPConfig(fps=60.0, seed=7)

    assert config.sensor_timing.fps == 60.0
    assert alias_config == config
    assert config.stages[0].name == "blc"
    assert config.transport.request_queue_depth > 0


def test_single_frame_timestamps_are_monotonic(asset_store) -> None:
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), _small_scene(asset_store), asset_store=asset_store)
    timestamps = result.timeline.timestamps_us

    assert timestamps["request"] <= timestamps["exposure_start"]
    assert timestamps["exposure_start"] < timestamps["exposure_mid"] < timestamps["readout_start"]
    assert timestamps["readout_start"] < timestamps["readout_end"] <= timestamps["stats_ready"]
    assert timestamps["readout_start"] <= timestamps["isp_start"] < timestamps["isp_done"]
    assert timestamps["isp_done"] < timestamps["dma_done"] < timestamps["app_visible"]
    assert "hw_isp" in result.ip.metadata
    assert "hw_isp" in result.camera.metadata


def test_rolling_shutter_row_timing_formula(asset_store) -> None:
    config = hw_isp_config(line_time_us=10.0, hidden_lines_top=3, exposure_time_us=1000.0)
    result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        asset_store=asset_store,
    )
    timeline = result.timeline

    assert timeline.row_start_us(0) == timeline.timestamps_us["readout_start"] + 30.0
    assert timeline.row_start_us(5) == timeline.timestamps_us["readout_start"] + 80.0


def test_line_buffer_stage_starts_before_frame_readout_end(asset_store) -> None:
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), _small_scene(asset_store), asset_store=asset_store)
    readout_end = result.timeline.timestamps_us["readout_end"]
    line_stages = [stage for stage in result.timeline.stages if stage.buffering in {"stream", "line"}]

    assert line_stages
    assert all(stage.start_us < readout_end for stage in line_stages)


def test_frame_buffer_stage_waits_for_previous_stage(asset_store) -> None:
    config = hw_isp_config(
        stages=[
            HWIspStage("stream_a", "bayer", "stream", window_lines=1, stage_latency_cycles=10),
            HWIspStage("frame_b", "rgb", "frame", stage_latency_cycles=10),
        ]
    )
    result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        asset_store=asset_store,
    )
    stream_stage, frame_stage = result.timeline.stages

    assert frame_stage.start_us >= stream_stage.end_us


def test_latency_factors_change_modeled_timing(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    scene = _small_scene(asset_store)
    nominal = hw_isp_simulate_frame(camera, scene, hw_isp_config(), asset_store=asset_store)
    slow = hw_isp_simulate_frame(camera, scene, hw_isp_config(global_latency_factor=2.0), asset_store=asset_store)

    nominal_isp = nominal.timeline.timestamps_us["isp_done"] - nominal.timeline.timestamps_us["isp_start"]
    slow_isp = slow.timeline.timestamps_us["isp_done"] - slow.timeline.timestamps_us["isp_start"]
    assert slow_isp > nominal_isp
    assert slow.timeline.timestamps_us["readout_end"] == nominal.timeline.timestamps_us["readout_end"]


def test_stage_latency_factor_affects_that_stage_and_downstream(asset_store) -> None:
    base_stages = [
        HWIspStage("first", "bayer", "stream", stage_latency_cycles=10),
        HWIspStage("second", "rgb", "stream", stage_latency_cycles=10),
    ]
    slow_stages = [
        HWIspStage("first", "bayer", "stream", stage_latency_cycles=10),
        HWIspStage("second", "rgb", "stream", stage_latency_cycles=10, latency_factor=3.0),
    ]
    baseline = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        hw_isp_config(stages=base_stages),
        asset_store=asset_store,
    )
    custom = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        hw_isp_config(stages=slow_stages),
        asset_store=asset_store,
    )

    assert custom.timeline.stages[1].cycle_latency_us > custom.timeline.stages[0].cycle_latency_us
    assert custom.timeline.timestamps_us["app_visible"] > baseline.timeline.timestamps_us["app_visible"]


def test_control_delay_maps_stats_frames(asset_store) -> None:
    config = hw_isp_config(control_path={"ae_apply_delay_frames": 2, "awb_apply_delay_frames": 2})
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        nframes=4,
        asset_store=asset_store,
    )

    assert sequence.frames[0].controls_applied["warmup"] is True
    assert sequence.frames[1].controls_applied["warmup"] is True
    assert sequence.frames[2].controls_applied["ae_stats_frame"] == 0
    assert sequence.frames[3].controls_applied["awb_stats_frame"] == 1


def test_queue_depth_creates_stall(asset_store) -> None:
    config = hw_isp_config(
        fps=1000.0,
        transport={"request_queue_depth": 1, "max_buffers": 1, "app_processing_us": 5000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        config,
        nframes=3,
        asset_store=asset_store,
    )

    assert sequence.frames[1].timeline.queue_stall_us > 0.0
    assert sequence.aggregate["queue_stall_total_us"] > 0.0


def test_export_json_and_timeline_table(asset_store, tmp_path) -> None:
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        _small_scene(asset_store),
        hw_isp_config(),
        nframes=2,
        asset_store=asset_store,
    )
    table = hw_isp_timeline_table(sequence)
    output = hw_isp_export_json(sequence, tmp_path / "hwisp.json")
    payload = json.loads(output.read_text())

    assert output.exists()
    assert payload["type"] == "sequence"
    assert len(payload["frames"]) == 2
    assert "produced_stats" in payload["frames"][0]["controls_applied"]
    assert "requested_controls" in payload["frames"][0]["controls_applied"]
    assert "applied_controls" in payload["frames"][0]["controls_applied"]
    assert "ae" in payload["frames"][0]["controls_applied"]["produced_stats"]
    assert "awb" in payload["frames"][0]["controls_applied"]["produced_stats"]
    assert "stats_grid" in payload["frames"][0]["controls_applied"]["produced_stats"]["ae"]
    assert "validation_verdicts" in payload["aggregate"]
    assert "ae_settle_frame" in payload["aggregate"]
    assert any(row["type"] == "stage" for row in table)
    assert hw_isp_latency_summary(sequence)["frame_count"] == 2.0


def test_hw_isp_frame_preserves_camera_compute_image(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)
    scene = _small_scene(asset_store)
    normal = camera_compute(camera.clone(), scene, asset_store=asset_store)
    simulated = hw_isp_simulate_frame(camera, scene, hw_isp_config(), asset_store=asset_store)

    np.testing.assert_allclose(
        ip_get(camera_get(normal, "ip"), "result"),
        ip_get(simulated.ip, "result"),
    )


def test_apply_to_image_false_preserves_sequence_image_behavior(asset_store) -> None:
    scene = _small_scene(asset_store)
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        scene,
        hw_isp_config(control_path={"apply_to_image": False}),
        nframes=2,
        asset_store=asset_store,
    )

    np.testing.assert_allclose(
        ip_get(sequence.frames[0].ip, "result"),
        ip_get(sequence.frames[1].ip, "result"),
    )


def test_ae_mean_metering_mode_matches_full_frame_luma(asset_store) -> None:
    scene = _off_center_highlight_scene(_small_scene(asset_store))
    config = hw_isp_config(control_path={"ae_metering": "mean"})
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), scene, config, asset_store=asset_store)
    ae_stats = result.controls_applied["produced_stats"]["ae"]

    assert ae_stats["ae_metering"] == "mean"
    assert ae_stats["metering_luma"] == ae_stats["mean_luma"]
    assert result.controls_applied["produced_stats"]["mean_sensor_luma_norm"] == ae_stats["mean_luma"]


def test_center_weighted_ae_differs_from_mean_for_off_center_highlight(asset_store) -> None:
    scene = _off_center_highlight_scene(_small_scene(asset_store))
    mean_result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        scene,
        hw_isp_config(control_path={"ae_metering": "mean"}),
        asset_store=asset_store,
    )
    weighted_result = hw_isp_simulate_frame(
        camera_create(asset_store=asset_store),
        scene,
        hw_isp_config(control_path={"ae_metering": "center_weighted"}),
        asset_store=asset_store,
    )

    mean_ae = mean_result.controls_applied["produced_stats"]["ae"]
    weighted_ae = weighted_result.controls_applied["produced_stats"]["ae"]
    assert weighted_ae["metering_luma"] != mean_ae["metering_luma"]
    assert weighted_ae["metering_luma"] < mean_ae["metering_luma"]


def test_highlight_clipping_forces_ae_product_downward(asset_store) -> None:
    scene = _off_center_highlight_scene(_small_scene(asset_store), base_scale=0.02, highlight_scale=3000.0)
    config = hw_isp_config(
        control_path={
            "apply_to_image": True,
            "awb_enabled": False,
            "target_luma": 0.5,
            "ae_highlight_clip": 0.2,
            "ae_highlight_weight": 0.8,
            "max_ae_step_ev": 1.0,
        },
        sensor_timing={"exposure_time_us": 8000.0},
    )
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), scene, config, asset_store=asset_store)
    stats = result.controls_applied["produced_stats"]["ae"]
    requested = result.controls_applied["requested_controls"]["ae"]
    current_product = result.controls_applied["exposure_time_us"] * result.controls_applied["analog_gain"]
    requested_product = requested["exposure_time_us"] * requested["analog_gain"]

    assert stats["metering_luma"] < stats["target_luma"]
    assert stats["clipped_fraction"] > 0.0
    assert requested["highlight_limited"] is True
    assert requested_product < current_product


def test_valid_luma_awb_excludes_dark_and_clipped_pixels(asset_store) -> None:
    scene = _off_center_highlight_scene(_warm_scene(_small_scene(asset_store)), base_scale=1.0, highlight_scale=100.0)
    photons = np.asarray(scene.data["photons"], dtype=float).copy()
    rows, cols = photons.shape[:2]
    photons[-max(rows // 4, 1) :, -max(cols // 4, 1) :, :] = 0.0
    scene.data["photons"] = photons
    config = hw_isp_config(control_path={"awb_stats_roi": "valid_luma", "awb_min_luma": 0.02, "awb_max_luma": 0.5})
    result = hw_isp_simulate_frame(camera_create(asset_store=asset_store), scene, config, asset_store=asset_store)
    awb_stats = result.controls_applied["produced_stats"]["awb"]

    assert awb_stats["valid_pixel_fraction"] < 1.0
    assert not np.allclose(awb_stats["rgb_means"], awb_stats["raw_rgb_means"])


def test_ae_behavior_delay_applies_requested_controls(asset_store) -> None:
    scene = _small_scene(asset_store)
    config = hw_isp_config(
        control_path={"apply_to_image": True, "ae_apply_delay_frames": 2, "awb_enabled": False},
        sensor_timing={"exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        scene,
        config,
        nframes=4,
        asset_store=asset_store,
    )

    requested = sequence.frames[0].controls_applied["requested_controls"]["ae"]
    applied = sequence.frames[2].controls_applied
    assert applied["ae_stats_frame"] == 0
    assert applied["exposure_time_us"] == requested["exposure_time_us"]
    assert applied["analog_gain"] == requested["analog_gain"]


def test_ae_behavior_responds_to_brightness_step(asset_store) -> None:
    base = _small_scene(asset_store)
    bright = _scaled_scene(base, 4.0)
    config = hw_isp_config(
        control_path={"apply_to_image": True, "target_luma": 0.18, "awb_enabled": False},
        sensor_timing={"fps": 30.0, "exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        [base] * 4 + [bright] * 4,
        config,
        asset_store=asset_store,
    )

    product_at_step = (
        sequence.frames[4].controls_applied["exposure_time_us"]
        * sequence.frames[4].controls_applied["analog_gain"]
    )
    product_after_delay = (
        sequence.frames[6].controls_applied["exposure_time_us"]
        * sequence.frames[6].controls_applied["analog_gain"]
    )
    assert sequence.frames[6].controls_applied["ae_stats_frame"] == 4
    assert product_after_delay < product_at_step


def test_ae_settle_metrics_identify_convergence_after_brightness_step(asset_store) -> None:
    base = _small_scene(asset_store)
    bright = _scaled_scene(base, 4.0)
    config = hw_isp_config(
        control_path={"apply_to_image": True, "target_luma": 0.18, "awb_enabled": False},
        sensor_timing={"fps": 30.0, "exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        [base] * 4 + [bright] * 8,
        config,
        asset_store=asset_store,
    )

    assert sequence.aggregate["ae_settle_frame"] >= 4.0
    assert sequence.aggregate["ae_settle_frames_after_step"] >= 0.0
    assert abs(sequence.aggregate["ae_final_error_ev"]) < 0.25
    assert sequence.aggregate["validation_verdicts"]["ae_settle"] is True


def test_ae_behavior_clamps_exposure_and_gain(asset_store) -> None:
    dark = _scaled_scene(_small_scene(asset_store), 0.01)
    config = hw_isp_config(
        control_path={
            "apply_to_image": True,
            "target_luma": 0.5,
            "max_exposure_fraction": 0.25,
            "max_analog_gain": 2.0,
            "awb_enabled": False,
        },
        sensor_timing={"fps": 100.0, "exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        dark,
        config,
        nframes=6,
        asset_store=asset_store,
    )

    max_exposure = (1.0e6 / config.sensor_timing.fps) * config.control_path.max_exposure_fraction
    for frame in sequence.frames:
        assert frame.controls_applied["exposure_time_us"] <= max_exposure
        assert frame.controls_applied["analog_gain"] <= config.control_path.max_analog_gain


def test_awb_behavior_delay_applies_requested_gains(asset_store) -> None:
    warm = _warm_scene(_small_scene(asset_store))
    config = hw_isp_config(
        control_path={"apply_to_image": True, "ae_enabled": False, "awb_apply_delay_frames": 2},
        sensor_timing={"exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        warm,
        config,
        nframes=4,
        asset_store=asset_store,
    )

    requested = sequence.frames[0].controls_applied["requested_controls"]["awb"]["wb_gains_rgb"]
    applied = sequence.frames[2].controls_applied
    assert applied["awb_stats_frame"] == 0
    np.testing.assert_allclose(applied["wb_gains_rgb"], requested)


def test_awb_behavior_reduces_corrected_channel_imbalance(asset_store) -> None:
    warm = _warm_scene(_small_scene(asset_store))
    config = hw_isp_config(
        control_path={"apply_to_image": True, "ae_enabled": False, "awb_apply_delay_frames": 2},
        sensor_timing={"exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        warm,
        config,
        nframes=4,
        asset_store=asset_store,
    )

    frame0_stats = sequence.frames[0].controls_applied["produced_stats"]
    frame2_stats = sequence.frames[2].controls_applied["produced_stats"]
    frame2_gains = np.asarray(sequence.frames[2].controls_applied["wb_gains_rgb"], dtype=float)
    assert frame2_gains[0] < 1.0
    assert frame2_gains[2] > 1.0
    assert frame2_stats["awb_corrected_rgb_imbalance"] < frame0_stats["sensor_rgb_imbalance"]


def test_awb_settle_metrics_identify_reduced_imbalance_after_warm_step(asset_store) -> None:
    base = _small_scene(asset_store)
    warm = _warm_scene(base)
    config = hw_isp_config(
        control_path={"apply_to_image": True, "ae_enabled": False, "awb_apply_delay_frames": 2},
        sensor_timing={"exposure_time_us": 8000.0},
    )
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=asset_store),
        [base] * 8 + [warm] * 6,
        config,
        asset_store=asset_store,
    )

    assert sequence.aggregate["awb_settle_frame"] >= 8.0
    assert sequence.aggregate["awb_final_rgb_imbalance"] < 0.20
    assert sequence.aggregate["validation_verdicts"]["awb_settle"] is True


def test_hw_isp_report_renderer_writes_html(asset_store, tmp_path) -> None:
    outputs = render_hwisp_timeline_report(tmp_path / "hwisp", nframes=3)

    assert outputs["html"].exists()
    assert outputs["details_html"].exists()
    assert outputs["summary"].exists()
    assert outputs["three_a_summary"].exists()
    assert outputs["frame_timeline"].exists()
    assert outputs["ae_convergence"].exists()
    assert outputs["awb_convergence"].exists()
    assert outputs["three_a_thumbnails"].exists()
    html = outputs["html"].read_text(encoding="utf-8")
    details = outputs["details_html"].read_text(encoding="utf-8")
    assert "HW ISP Simulation Report" in html
    assert "frame_timeline.png" in html
    assert "3A AE/AWB Behavior" in html
    assert "3A Validation Verdict" in html
    assert "ae_convergence.png" in html
    assert "HW ISP Frame Details" in details
    assert "Stage Timing" in details
