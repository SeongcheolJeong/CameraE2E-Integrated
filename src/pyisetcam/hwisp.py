"""Hardware ISP latency simulation layer.

The simulator keeps the existing image pipeline as the source of truth for pixel
values and adds deterministic timing metadata for system verification.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_get
from .ip import ip_set
from .scene import scene_create
from .sensor import sensor_get, sensor_set
from .types import Camera, ImageProcessor, OpticalImage, Scene, Sensor


@dataclass(frozen=True)
class HWIspSensorTiming:
    fps: float = 30.0
    line_time_us: float = 15.2
    active_lines: int = 1080
    hidden_lines_top: int = 16
    hidden_lines_bottom: int = 20
    exposure_time_us: float = 8000.0
    rolling_shutter: bool = True


@dataclass(frozen=True)
class HWIspStage:
    name: str
    domain: str
    buffering: str
    window_lines: int = 1
    stage_latency_cycles: float = 0.0
    clock_mhz: float = 400.0
    pixels_per_cycle: float = 2.0
    latency_factor: float = 1.0
    enabled: bool = True
    history_frames: int = 0


@dataclass(frozen=True)
class HWIspControlPath:
    ae_enabled: bool = True
    awb_enabled: bool = True
    stats_ready_at: str = "frame_end"
    ae_apply_delay_frames: int = 2
    awb_apply_delay_frames: int = 2
    apply_to_image: bool = False
    target_luma: float = 0.18
    min_exposure_time_us: float = 100.0
    max_exposure_fraction: float = 0.90
    min_analog_gain: float = 1.0
    max_analog_gain: float = 16.0
    max_ae_step_ev: float = 1.0
    awb_method: str = "gray_world"
    min_wb_gain: float = 0.25
    max_wb_gain: float = 4.0
    max_awb_step_ev: float = 0.5
    stats_grid: tuple[int, int] = (8, 8)
    ae_metering: str = "center_weighted"
    ae_highlight_clip: float = 0.98
    ae_highlight_weight: float = 0.5
    awb_stats_roi: str = "valid_luma"
    awb_min_luma: float = 0.02
    awb_max_luma: float = 0.95


@dataclass(frozen=True)
class HWIspTransport:
    request_queue_depth: int = 4
    max_buffers: int = 6
    dma_submit_us: float = 120.0
    dma_complete_us: float = 320.0
    app_processing_us: float = 500.0
    jitter_us_std: float = 0.0
    drop_policy: str = "stall"


@dataclass(frozen=True)
class HWIspConfig:
    sensor_timing: HWIspSensorTiming = field(default_factory=HWIspSensorTiming)
    stages: tuple[HWIspStage, ...] = field(default_factory=tuple)
    control_path: HWIspControlPath = field(default_factory=HWIspControlPath)
    transport: HWIspTransport = field(default_factory=HWIspTransport)
    global_latency_factor: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class HWIspStageSpan:
    name: str
    domain: str
    buffering: str
    start_us: float
    end_us: float
    latency_us: float
    line_buffer_delay_us: float
    cycle_latency_us: float
    stalled_us: float = 0.0


@dataclass(frozen=True)
class HWIspFrameTimeline:
    frame_id: int
    timestamps_us: dict[str, float]
    stages: tuple[HWIspStageSpan, ...]
    queue_stall_us: float = 0.0
    dropped: bool = False

    def row_start_us(self, row: int) -> float:
        line_time = float(self.timestamps_us["line_time_us"])
        hidden_top = float(self.timestamps_us["hidden_lines_top"])
        readout_start = float(self.timestamps_us["readout_start"])
        return readout_start + (hidden_top + int(row)) * line_time


@dataclass(frozen=True)
class HWIspFrameResult:
    camera: Camera
    oi: OpticalImage
    sensor: Sensor
    ip: ImageProcessor
    timeline: HWIspFrameTimeline
    controls_applied: dict[str, Any]


@dataclass(frozen=True)
class HWIspSequenceResult:
    frames: tuple[HWIspFrameResult, ...]
    aggregate: dict[str, Any]


_AE_SETTLE_EV_THRESHOLD = 0.25
_AWB_SETTLE_IMBALANCE_THRESHOLD = 0.20
_SETTLE_CONSECUTIVE_FRAMES = 2
_AE_STEP_FRAME = 4
_AWB_STEP_FRAME = 8


def _default_stages() -> tuple[HWIspStage, ...]:
    return (
        HWIspStage("blc", "bayer", "stream", window_lines=1, stage_latency_cycles=24),
        HWIspStage("dpc", "bayer", "line", window_lines=3, stage_latency_cycles=80),
        HWIspStage("demosaic", "bayer_to_rgb", "line", window_lines=5, stage_latency_cycles=220),
        HWIspStage("ccm_gamma", "rgb", "stream", window_lines=1, stage_latency_cycles=48),
        HWIspStage(
            "tnr",
            "yuv",
            "frame",
            window_lines=1,
            stage_latency_cycles=0,
            enabled=False,
            history_frames=2,
        ),
    )


def _replace_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    payload = asdict(instance)
    payload.update(updates)
    return type(instance)(**payload)


def _normalize_stage(stage: HWIspStage | dict[str, Any]) -> HWIspStage:
    if isinstance(stage, HWIspStage):
        return stage
    return HWIspStage(**dict(stage))


def hw_isp_config(**overrides: Any) -> HWIspConfig:
    """Create an HW ISP timing configuration with optional nested overrides."""

    config = HWIspConfig(stages=_default_stages())
    payload: dict[str, Any] = {
        "sensor_timing": config.sensor_timing,
        "stages": config.stages,
        "control_path": config.control_path,
        "transport": config.transport,
        "global_latency_factor": config.global_latency_factor,
        "seed": config.seed,
    }

    if "sensor_timing" in overrides:
        value = overrides.pop("sensor_timing")
        payload["sensor_timing"] = value if isinstance(value, HWIspSensorTiming) else HWIspSensorTiming(**dict(value))
    if "control_path" in overrides:
        value = overrides.pop("control_path")
        payload["control_path"] = value if isinstance(value, HWIspControlPath) else HWIspControlPath(**dict(value))
    if "transport" in overrides:
        value = overrides.pop("transport")
        payload["transport"] = value if isinstance(value, HWIspTransport) else HWIspTransport(**dict(value))
    if "stages" in overrides:
        payload["stages"] = tuple(_normalize_stage(stage) for stage in overrides.pop("stages"))

    nested_sensor: dict[str, Any] = {}
    nested_control: dict[str, Any] = {}
    nested_transport: dict[str, Any] = {}
    for key in list(overrides):
        if hasattr(HWIspSensorTiming, key):
            nested_sensor[key] = overrides.pop(key)
        elif hasattr(HWIspControlPath, key):
            nested_control[key] = overrides.pop(key)
        elif hasattr(HWIspTransport, key):
            nested_transport[key] = overrides.pop(key)

    if nested_sensor:
        payload["sensor_timing"] = _replace_dataclass(payload["sensor_timing"], nested_sensor)
    if nested_control:
        payload["control_path"] = _replace_dataclass(payload["control_path"], nested_control)
    if nested_transport:
        payload["transport"] = _replace_dataclass(payload["transport"], nested_transport)

    payload.update(overrides)
    return HWIspConfig(**payload)


def _deterministic_jitter(config: HWIspConfig, frame_id: int) -> float:
    std = float(config.transport.jitter_us_std)
    if std <= 0.0:
        return 0.0
    generator = np.random.default_rng(int(config.seed) + int(frame_id))
    return float(generator.normal(0.0, std))


def _frame_visibility_history(previous_state: Any) -> list[float]:
    if previous_state is None:
        return []
    if isinstance(previous_state, HWIspSequenceResult):
        return [float(frame.timeline.timestamps_us["app_visible"]) for frame in previous_state.frames]
    if isinstance(previous_state, HWIspFrameResult):
        return [float(previous_state.timeline.timestamps_us["app_visible"])]
    if isinstance(previous_state, dict):
        values = previous_state.get("app_visible_times_us", [])
        return [float(value) for value in values]
    return []


def _queue_adjusted_request(
    requested_us: float,
    *,
    previous_state: Any,
    config: HWIspConfig,
) -> tuple[float, float, bool]:
    history = _frame_visibility_history(previous_state)
    if not history:
        return requested_us, 0.0, False

    depth = max(1, min(int(config.transport.request_queue_depth), int(config.transport.max_buffers)))
    if len(history) < depth:
        return requested_us, 0.0, False

    earliest_release = float(history[-depth])
    if requested_us >= earliest_release:
        return requested_us, 0.0, False

    if str(config.transport.drop_policy).lower() == "drop_newest":
        return requested_us, 0.0, True
    return earliest_release, earliest_release - requested_us, False


def _infer_sensor_shape(sensor: Sensor, timing: HWIspSensorTiming) -> tuple[int, int]:
    try:
        size = np.asarray(sensor_get(sensor, "size"), dtype=int).reshape(-1)
    except Exception:
        size = np.asarray([], dtype=int)
    rows = int(size[0]) if size.size >= 1 and int(size[0]) > 0 else int(timing.active_lines)
    cols = int(size[1]) if size.size >= 2 and int(size[1]) > 0 else rows
    return rows, cols


def _stage_cycle_latency_us(stage: HWIspStage, config: HWIspConfig) -> float:
    clock = max(float(stage.clock_mhz), 1.0e-12)
    return (
        float(stage.stage_latency_cycles)
        / clock
        * float(stage.latency_factor)
        * float(config.global_latency_factor)
    )


def _build_timeline(
    *,
    frame_id: int,
    sensor: Sensor,
    config: HWIspConfig,
    previous_state: Any,
) -> HWIspFrameTimeline:
    timing = config.sensor_timing
    frame_interval_us = 1.0e6 / max(float(timing.fps), 1.0e-12)
    requested_us = frame_id * frame_interval_us + _deterministic_jitter(config, frame_id)
    t_request, queue_stall_us, dropped = _queue_adjusted_request(
        requested_us,
        previous_state=previous_state,
        config=config,
    )

    exposure_start = t_request
    exposure_mid = exposure_start + float(timing.exposure_time_us) / 2.0
    readout_start = exposure_start + float(timing.exposure_time_us)
    active_lines, active_cols = _infer_sensor_shape(sensor, timing)
    readout_end = readout_start + (
        int(timing.hidden_lines_top) + active_lines + int(timing.hidden_lines_bottom)
    ) * float(timing.line_time_us)
    stats_ready = readout_end

    prev_start = readout_start + int(timing.hidden_lines_top) * float(timing.line_time_us)
    prev_end = readout_end - int(timing.hidden_lines_bottom) * float(timing.line_time_us)
    spans: list[HWIspStageSpan] = []
    for stage in config.stages:
        if not stage.enabled:
            continue
        buffering = str(stage.buffering).lower()
        cycle_latency = _stage_cycle_latency_us(stage, config)
        if buffering == "frame":
            pixel_count = max(active_lines * active_cols, 1)
            throughput_us = pixel_count / (
                max(float(stage.clock_mhz), 1.0e-12) * max(float(stage.pixels_per_cycle), 1.0e-12)
            )
            history_us = max(int(stage.history_frames), 0) * frame_interval_us
            start = prev_end + cycle_latency
            end = start + throughput_us * float(stage.latency_factor) * float(config.global_latency_factor) + history_us
            line_delay = 0.0
        else:
            line_delay = (
                max(int(stage.window_lines) - 1, 0)
                * float(timing.line_time_us)
                * float(stage.latency_factor)
                * float(config.global_latency_factor)
            )
            start = prev_start + line_delay + cycle_latency
            end = prev_end + line_delay + cycle_latency

        spans.append(
            HWIspStageSpan(
                name=str(stage.name),
                domain=str(stage.domain),
                buffering=buffering,
                start_us=float(start),
                end_us=float(end),
                latency_us=float(end - start),
                line_buffer_delay_us=float(line_delay),
                cycle_latency_us=float(cycle_latency),
            )
        )
        prev_start = start
        prev_end = end

    isp_start = spans[0].start_us if spans else readout_end
    isp_done = spans[-1].end_us if spans else readout_end
    transport_scale = float(config.global_latency_factor)
    dma_done = isp_done + (
        float(config.transport.dma_submit_us) + float(config.transport.dma_complete_us)
    ) * transport_scale
    app_visible = dma_done + float(config.transport.app_processing_us) * transport_scale

    timestamps = {
        "request": float(t_request),
        "requested": float(requested_us),
        "exposure_start": float(exposure_start),
        "exposure_mid": float(exposure_mid),
        "readout_start": float(readout_start),
        "readout_end": float(readout_end),
        "stats_ready": float(stats_ready),
        "isp_start": float(isp_start),
        "isp_done": float(isp_done),
        "dma_done": float(dma_done),
        "app_visible": float(app_visible),
        "frame_interval_us": float(frame_interval_us),
        "line_time_us": float(timing.line_time_us),
        "hidden_lines_top": float(timing.hidden_lines_top),
        "hidden_lines_bottom": float(timing.hidden_lines_bottom),
    }
    return HWIspFrameTimeline(
        frame_id=int(frame_id),
        timestamps_us=timestamps,
        stages=tuple(spans),
        queue_stall_us=float(queue_stall_us),
        dropped=bool(dropped),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _default_control_state(camera: Camera, config: HWIspConfig) -> dict[str, Any]:
    sensor = camera.fields.get("sensor")
    try:
        analog_gain = float(sensor_get(sensor, "analog gain")) if sensor is not None else 1.0
    except Exception:
        analog_gain = 1.0
    return {
        "exposure_time_us": float(config.sensor_timing.exposure_time_us),
        "analog_gain": analog_gain,
        "wb_gains_rgb": [1.0, 1.0, 1.0],
        "ae_stats_frame": None,
        "awb_stats_frame": None,
    }


def _merge_control_update(base: dict[str, Any], update: Any) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    if update is None:
        return merged
    updates = update if isinstance(update, list) else [update]
    for item in updates:
        if not isinstance(item, dict):
            continue
        for key in ("exposure_time_us", "analog_gain", "wb_gains_rgb", "ae_stats_frame", "awb_stats_frame"):
            if key in item:
                value = item[key]
                if key == "wb_gains_rgb":
                    merged[key] = np.asarray(value, dtype=float).reshape(-1)[:3].astype(float).tolist()
                elif key in {"ae_stats_frame", "awb_stats_frame"} and value is None:
                    merged[key] = None
                elif key in {"ae_stats_frame", "awb_stats_frame"}:
                    merged[key] = int(value)
                else:
                    merged[key] = float(value)
    return merged


def _scheduled_update_for_frame(previous_state: Any, frame_id: int) -> Any:
    if not isinstance(previous_state, dict):
        return None
    schedule = previous_state.get("scheduled_controls", {})
    if not isinstance(schedule, dict):
        return None
    return schedule.get(frame_id, schedule.get(str(frame_id)))


def _resolve_applied_controls(
    camera: Camera,
    config: HWIspConfig,
    frame_id: int,
    previous_state: Any,
) -> dict[str, Any]:
    controls = _default_control_state(camera, config)
    if isinstance(previous_state, dict) and isinstance(previous_state.get("current_controls"), dict):
        controls = _merge_control_update(controls, previous_state["current_controls"])
    controls = _merge_control_update(controls, _scheduled_update_for_frame(previous_state, int(frame_id)))
    return _clamp_control_state(controls, config)


def _config_with_exposure(config: HWIspConfig, exposure_time_us: float) -> HWIspConfig:
    return HWIspConfig(
        sensor_timing=_replace_dataclass(
            config.sensor_timing,
            {"exposure_time_us": float(exposure_time_us)},
        ),
        stages=config.stages,
        control_path=config.control_path,
        transport=config.transport,
        global_latency_factor=config.global_latency_factor,
        seed=config.seed,
    )


def _apply_controls_to_camera(camera: Camera, controls: dict[str, Any], config: HWIspConfig) -> Camera:
    if not bool(config.control_path.apply_to_image):
        return camera
    updated = camera.clone()
    sensor = updated.fields.get("sensor")
    if sensor is not None:
        sensor = sensor_set(sensor, "exposure duration", float(controls["exposure_time_us"]) * 1.0e-6)
        sensor = sensor_set(sensor, "analog gain", float(controls["analog_gain"]))
        updated.fields["sensor"] = sensor
    ip = updated.fields.get("ip")
    if ip is not None and config.control_path.awb_enabled:
        wb = np.asarray(controls.get("wb_gains_rgb", [1.0, 1.0, 1.0]), dtype=float).reshape(-1)
        if wb.size < 3:
            wb = np.pad(wb, (0, 3 - wb.size), constant_values=1.0)
        ip = ip_set(ip, "illuminant correction method", "manual")
        ip = ip_set(ip, "illuminant correction matrix", np.diag(wb[:3]))
        updated.fields["ip"] = ip
    return updated


def _sensor_luma_plane_norm(sensor: Sensor) -> np.ndarray:
    volts = np.asarray(sensor.data.get("volts", sensor.data.get("dv", np.empty(0))), dtype=float)
    if volts.size == 0:
        return np.zeros((1, 1), dtype=float)
    if volts.ndim == 1:
        volts = volts.reshape(1, -1)
    elif volts.ndim == 3:
        volts = np.mean(volts[:, :, : min(volts.shape[2], 3)], axis=2)
    try:
        voltage_swing = float(sensor_get(sensor, "pixel voltage swing"))
    except Exception:
        voltage_swing = max(float(np.max(volts)), 1.0)
    luma = volts / max(voltage_swing, 1.0e-12)
    return np.nan_to_num(luma, nan=0.0, posinf=0.0, neginf=0.0).astype(float)


def _stats_grid_shape(config: HWIspConfig) -> tuple[int, int]:
    grid = np.asarray(config.control_path.stats_grid, dtype=int).reshape(-1)
    rows = int(grid[0]) if grid.size >= 1 else 8
    cols = int(grid[1]) if grid.size >= 2 else rows
    return max(rows, 1), max(cols, 1)


def _tile_means(values: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return np.zeros(grid, dtype=float)
    row_splits = np.array_split(array, grid[0], axis=0)
    result = np.zeros(grid, dtype=float)
    for row_index, row_block in enumerate(row_splits):
        col_splits = np.array_split(row_block, grid[1], axis=1)
        for col_index, tile in enumerate(col_splits):
            result[row_index, col_index] = float(np.mean(tile)) if tile.size else 0.0
    return result


def _center_weights(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    y = np.linspace(-1.0, 1.0, rows, dtype=float)
    x = np.linspace(-1.0, 1.0, cols, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    weights = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.55 * 0.55))
    return weights / max(float(np.sum(weights)), 1.0e-12)


def _ae_stats(sensor: Sensor, config: HWIspConfig) -> dict[str, Any]:
    control = config.control_path
    luma = _sensor_luma_plane_norm(sensor)
    grid = _stats_grid_shape(config)
    tile_mean = _tile_means(luma, grid)
    clipped = np.asarray(luma >= float(control.ae_highlight_clip), dtype=float)
    tile_clip = _tile_means(clipped, grid)
    mean_luma = float(np.mean(luma)) if luma.size else 0.0
    weighted_luma = float(np.sum(tile_mean * _center_weights(tile_mean.shape)))
    metering = mean_luma
    metering_mode = str(control.ae_metering).lower().replace("-", "_").replace(" ", "_")
    if metering_mode in {"center", "center_weighted"}:
        metering = weighted_luma
    target = max(float(control.target_luma), 1.0e-9)
    ev_error = float(np.log2(max(metering, 1.0e-9) / target))
    return {
        "stats_grid": [int(grid[0]), int(grid[1])],
        "ae_metering": str(control.ae_metering),
        "tile_mean_luma": tile_mean.astype(float).tolist(),
        "tile_clip_fraction": tile_clip.astype(float).tolist(),
        "mean_luma": float(mean_luma),
        "weighted_luma": float(weighted_luma),
        "metering_luma": float(metering),
        "target_luma": float(control.target_luma),
        "target_error": float(metering - float(control.target_luma)),
        "ev_error": ev_error,
        "clipped_fraction": float(np.mean(clipped)) if clipped.size else 0.0,
        "highlight_clip": float(control.ae_highlight_clip),
    }


def _rgb_array_from_ip(ip: ImageProcessor) -> np.ndarray | None:
    for key in ("sensorspace", "result", "srgb"):
        data = ip.data.get(key)
        if data is None:
            continue
        array = np.asarray(data, dtype=float)
        if array.ndim == 3 and array.shape[2] >= 3 and array.size:
            return np.nan_to_num(array[:, :, :3], nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    return None


def _rgb_means(array: np.ndarray | None, mask: np.ndarray | None = None) -> list[float]:
    if array is None or array.size == 0:
        return [0.0, 0.0, 0.0]
    rgb = np.asarray(array, dtype=float)
    if mask is not None and mask.shape == rgb.shape[:2] and bool(np.any(mask)):
        return np.mean(rgb[mask, :3], axis=0).astype(float).tolist()
    return np.mean(rgb[:, :, :3], axis=(0, 1)).astype(float).tolist()


def _awb_stats(ip: ImageProcessor, config: HWIspConfig, applied_controls: dict[str, Any]) -> dict[str, Any]:
    control = config.control_path
    rgb = _rgb_array_from_ip(ip)
    raw_means = _rgb_means(rgb)
    if rgb is None:
        mask = np.zeros((1, 1), dtype=bool)
        valid_fraction = 0.0
    else:
        luma = np.mean(rgb[:, :, :3], axis=2)
        roi = str(control.awb_stats_roi).lower().replace("-", "_").replace(" ", "_")
        if roi == "valid_luma":
            mask = (
                np.isfinite(luma)
                & (luma >= float(control.awb_min_luma))
                & (luma <= float(control.awb_max_luma))
            )
        else:
            mask = np.isfinite(luma)
        valid_fraction = float(np.mean(mask)) if mask.size else 0.0
        if not bool(np.any(mask)):
            mask = np.isfinite(luma)
            valid_fraction = float(np.mean(mask)) if mask.size else 0.0

    rgb_means = _rgb_means(rgb, mask)
    wb = np.asarray(applied_controls.get("wb_gains_rgb", [1.0, 1.0, 1.0]), dtype=float).reshape(-1)[:3]
    corrected_rgb = (np.asarray(rgb_means, dtype=float).reshape(-1)[:3] * wb).astype(float).tolist()
    raw_corrected_rgb = (np.asarray(raw_means, dtype=float).reshape(-1)[:3] * wb).astype(float).tolist()
    return {
        "stats_roi": str(control.awb_stats_roi),
        "valid_luma_range": [float(control.awb_min_luma), float(control.awb_max_luma)],
        "valid_pixel_fraction": float(valid_fraction),
        "raw_rgb_means": raw_means,
        "rgb_means": rgb_means,
        "corrected_rgb_means": corrected_rgb,
        "raw_corrected_rgb_means": raw_corrected_rgb,
        "raw_rgb_imbalance": _channel_imbalance(raw_means),
        "rgb_imbalance": _channel_imbalance(rgb_means),
        "corrected_rgb_imbalance": _channel_imbalance(corrected_rgb),
        "raw_corrected_rgb_imbalance": _channel_imbalance(raw_corrected_rgb),
    }


def _channel_imbalance(values: Any) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    mean = max(float(np.mean(array)), 1.0e-12)
    return float((np.max(array) - np.min(array)) / mean)


def _frame_stats(sensor: Sensor, ip: ImageProcessor, config: HWIspConfig, applied_controls: dict[str, Any]) -> dict[str, Any]:
    ae = _ae_stats(sensor, config)
    awb = _awb_stats(ip, config, applied_controls)
    output = ip.data.get("result", ip.data.get("srgb"))
    if output is not None and np.asarray(output).ndim == 3:
        output_means = np.mean(np.asarray(output, dtype=float)[:, :, :3], axis=(0, 1)).astype(float).tolist()
    else:
        output_means = [0.0, 0.0, 0.0]
    return {
        "mean_sensor_luma_norm": float(ae["mean_luma"]),
        "weighted_sensor_luma_norm": float(ae["weighted_luma"]),
        "metering_sensor_luma_norm": float(ae["metering_luma"]),
        "target_luma": float(config.control_path.target_luma),
        "clipped_fraction": float(ae["clipped_fraction"]),
        "sensor_rgb_means": list(awb["rgb_means"]),
        "sensor_raw_rgb_means": list(awb["raw_rgb_means"]),
        "awb_corrected_rgb_means": list(awb["corrected_rgb_means"]),
        "sensor_rgb_imbalance": float(awb["rgb_imbalance"]),
        "sensor_raw_rgb_imbalance": float(awb["raw_rgb_imbalance"]),
        "awb_corrected_rgb_imbalance": float(awb["corrected_rgb_imbalance"]),
        "output_rgb_means": output_means,
        "output_rgb_imbalance": _channel_imbalance(output_means),
        "ae": ae,
        "awb": awb,
    }


def _allocate_exposure_gain(
    desired_product: float,
    config: HWIspConfig,
) -> tuple[float, float]:
    control = config.control_path
    frame_interval_us = 1.0e6 / max(float(config.sensor_timing.fps), 1.0e-12)
    min_exposure = max(float(control.min_exposure_time_us), 1.0e-9)
    max_exposure = max(min_exposure, frame_interval_us * float(control.max_exposure_fraction))
    min_gain = max(float(control.min_analog_gain), 1.0e-9)
    max_gain = max(min_gain, float(control.max_analog_gain))
    product = max(float(desired_product), min_exposure * min_gain)

    if product <= max_exposure * min_gain:
        exposure = _clamp(product / min_gain, min_exposure, max_exposure)
        gain = min_gain
    else:
        exposure = max_exposure
        gain = _clamp(product / max(exposure, 1.0e-12), min_gain, max_gain)
    return exposure, gain


def _clamp_control_state(controls: dict[str, Any], config: HWIspConfig) -> dict[str, Any]:
    clamped = copy.deepcopy(controls)
    exposure_us, analog_gain = _allocate_exposure_gain(
        float(clamped["exposure_time_us"]) * float(clamped["analog_gain"]),
        config,
    )
    clamped["exposure_time_us"] = float(exposure_us)
    clamped["analog_gain"] = float(analog_gain)
    wb = np.asarray(clamped.get("wb_gains_rgb", [1.0, 1.0, 1.0]), dtype=float).reshape(-1)
    if wb.size < 3:
        wb = np.pad(wb, (0, 3 - wb.size), constant_values=1.0)
    wb = np.clip(wb[:3], float(config.control_path.min_wb_gain), float(config.control_path.max_wb_gain))
    clamped["wb_gains_rgb"] = wb.astype(float).tolist()
    return clamped


def _requested_ae_controls(
    frame_id: int,
    applied: dict[str, Any],
    stats: dict[str, Any],
    config: HWIspConfig,
) -> dict[str, Any] | None:
    control = config.control_path
    if not control.ae_enabled:
        return None
    ae_stats = stats.get("ae", {}) if isinstance(stats.get("ae"), dict) else {}
    measured = max(float(ae_stats.get("metering_luma", stats["mean_sensor_luma_norm"])), 1.0e-9)
    target = max(float(control.target_luma), 1.0e-9)
    step = max(float(control.max_ae_step_ev), 0.0)
    ratio = _clamp(target / measured, 2.0 ** (-step), 2.0**step)
    clipped_fraction = max(float(ae_stats.get("clipped_fraction", 0.0)), 0.0)
    highlight_limited = False
    if clipped_fraction > 0.0:
        clip_scale = _clamp(clipped_fraction / 0.02, 0.0, 1.0)
        highlight_weight = _clamp(float(control.ae_highlight_weight), 0.0, 1.0)
        highlight_ratio = _clamp(1.0 - highlight_weight * clip_scale, 2.0 ** (-step), 1.0)
        highlight_limited = highlight_ratio < ratio
        ratio = min(ratio, highlight_ratio)
    current_product = float(applied["exposure_time_us"]) * float(applied["analog_gain"])
    exposure_us, analog_gain = _allocate_exposure_gain(current_product * ratio, config)
    return {
        "source_frame": int(frame_id),
        "apply_frame": int(frame_id) + int(control.ae_apply_delay_frames),
        "measured_luma": float(measured),
        "mean_luma": float(ae_stats.get("mean_luma", stats["mean_sensor_luma_norm"])),
        "weighted_luma": float(ae_stats.get("weighted_luma", measured)),
        "clipped_fraction": float(clipped_fraction),
        "ev_error": float(ae_stats.get("ev_error", np.log2(measured / target))),
        "highlight_limited": bool(highlight_limited),
        "target_luma": float(target),
        "ratio": float(ratio),
        "exposure_time_us": float(exposure_us),
        "analog_gain": float(analog_gain),
    }


def _requested_awb_controls(
    frame_id: int,
    applied: dict[str, Any],
    stats: dict[str, Any],
    config: HWIspConfig,
) -> dict[str, Any] | None:
    control = config.control_path
    if not control.awb_enabled:
        return None
    if str(control.awb_method).lower().replace("-", "_").replace(" ", "_") != "gray_world":
        return None
    awb_stats = stats.get("awb", {}) if isinstance(stats.get("awb"), dict) else {}
    source_means = awb_stats.get("rgb_means", stats["sensor_rgb_means"])
    means = np.maximum(np.asarray(source_means, dtype=float).reshape(-1)[:3], 1.0e-9)
    current = np.maximum(np.asarray(applied.get("wb_gains_rgb", [1.0, 1.0, 1.0]), dtype=float).reshape(-1)[:3], 1.0e-9)
    target_mean = float(np.mean(means))
    desired = target_mean / means
    desired = desired / max(float(desired[1]), 1.0e-9)
    step = max(float(control.max_awb_step_ev), 0.0)
    relative = np.clip(desired / current, 2.0 ** (-step), 2.0**step)
    gains = current * relative
    gains = gains / max(float(gains[1]), 1.0e-9)
    gains = np.clip(gains, float(control.min_wb_gain), float(control.max_wb_gain))
    return {
        "source_frame": int(frame_id),
        "apply_frame": int(frame_id) + int(control.awb_apply_delay_frames),
        "rgb_means": means.astype(float).tolist(),
        "valid_pixel_fraction": float(awb_stats.get("valid_pixel_fraction", 1.0)),
        "wb_gains_rgb": gains.astype(float).tolist(),
    }


def _requested_controls(
    frame_id: int,
    applied: dict[str, Any],
    stats: dict[str, Any],
    config: HWIspConfig,
) -> dict[str, Any]:
    return {
        "ae": _requested_ae_controls(frame_id, applied, stats, config),
        "awb": _requested_awb_controls(frame_id, applied, stats, config),
    }


def _control_payload(
    frame_id: int,
    timeline: HWIspFrameTimeline,
    config: HWIspConfig,
    applied_controls: dict[str, Any],
    stats: dict[str, Any],
    requested_controls: dict[str, Any],
) -> dict[str, Any]:
    control = config.control_path
    if bool(control.apply_to_image):
        ae_applied = applied_controls.get("ae_stats_frame") if control.ae_enabled else None
        awb_applied = applied_controls.get("awb_stats_frame") if control.awb_enabled else None
    else:
        ae_source = frame_id - int(control.ae_apply_delay_frames) if control.ae_enabled else None
        awb_source = frame_id - int(control.awb_apply_delay_frames) if control.awb_enabled else None
        ae_applied = ae_source if ae_source is not None and ae_source >= 0 else None
        awb_applied = awb_source if awb_source is not None and awb_source >= 0 else None
    applied = {
        "exposure_time_us": float(applied_controls["exposure_time_us"]),
        "analog_gain": float(applied_controls["analog_gain"]),
        "wb_gains_rgb": np.asarray(applied_controls["wb_gains_rgb"], dtype=float).reshape(-1)[:3].astype(float).tolist(),
        "ae_stats_frame": None if ae_applied is None else int(ae_applied),
        "awb_stats_frame": None if awb_applied is None else int(awb_applied),
    }
    return {
        "ae_enabled": bool(control.ae_enabled),
        "awb_enabled": bool(control.awb_enabled),
        "apply_to_image": bool(control.apply_to_image),
        "ae_stats_frame": applied["ae_stats_frame"],
        "awb_stats_frame": applied["awb_stats_frame"],
        "ae_apply_delay_frames": int(control.ae_apply_delay_frames),
        "awb_apply_delay_frames": int(control.awb_apply_delay_frames),
        "stats_ready_us": float(timeline.timestamps_us["stats_ready"]),
        "warmup": (control.ae_enabled and ae_applied is None) or (control.awb_enabled and awb_applied is None),
        "exposure_time_us": applied["exposure_time_us"],
        "analog_gain": applied["analog_gain"],
        "wb_gains_rgb": applied["wb_gains_rgb"],
        "applied_controls": applied,
        "produced_stats": copy.deepcopy(stats),
        "requested_controls": copy.deepcopy(requested_controls),
    }


def _attach_metadata(
    camera: Camera,
    timeline: HWIspFrameTimeline,
    controls: dict[str, Any],
    config: HWIspConfig,
) -> Camera:
    camera.metadata["hw_isp"] = {
        "timeline": _timeline_to_dict(timeline),
        "controls_applied": copy.deepcopy(controls),
        "config": _config_to_dict(config),
    }
    ip = camera.fields.get("ip")
    if ip is not None:
        ip.metadata["hw_isp"] = copy.deepcopy(camera.metadata["hw_isp"])
    return camera


def hw_isp_simulate_frame(
    camera: Camera,
    scene: Scene | str,
    config: HWIspConfig | None = None,
    frame_id: int = 0,
    previous_state: Any = None,
    *,
    asset_store: AssetStore | None = None,
) -> HWIspFrameResult:
    """Run the normal camera pipeline and attach deterministic HW ISP timing."""

    store = asset_store or AssetStore.default()
    resolved_config = config or hw_isp_config()
    applied_controls = _resolve_applied_controls(camera, resolved_config, int(frame_id), previous_state)
    compute_config = (
        _config_with_exposure(resolved_config, float(applied_controls["exposure_time_us"]))
        if bool(resolved_config.control_path.apply_to_image)
        else resolved_config
    )
    controlled_camera = _apply_controls_to_camera(camera, applied_controls, compute_config)
    working_scene = scene_create(scene, asset_store=store) if isinstance(scene, str) else scene.clone()
    working_camera = camera_compute(controlled_camera.clone(), working_scene, asset_store=store)
    sensor = camera_get(working_camera, "sensor")
    ip = camera_get(working_camera, "ip")
    timeline = _build_timeline(
        frame_id=int(frame_id),
        sensor=sensor,
        config=compute_config,
        previous_state=previous_state,
    )
    stats = _frame_stats(sensor, ip, compute_config, applied_controls)
    requested = _requested_controls(int(frame_id), applied_controls, stats, compute_config)
    controls = _control_payload(int(frame_id), timeline, compute_config, applied_controls, stats, requested)
    working_camera = _attach_metadata(working_camera, timeline, controls, compute_config)
    return HWIspFrameResult(
        camera=working_camera,
        oi=camera_get(working_camera, "oi"),
        sensor=sensor,
        ip=camera_get(working_camera, "ip"),
        timeline=timeline,
        controls_applied=controls,
    )


def _broadcast_frames(scenes: Any, nframes: int | None) -> list[Any]:
    if isinstance(scenes, (list, tuple)):
        scene_list = list(scenes)
    else:
        scene_list = [scenes]
    total = int(nframes) if nframes is not None else len(scene_list)
    total = max(total, 1)
    if len(scene_list) == 1 and total > 1:
        scene_list = scene_list * total
    if len(scene_list) != total:
        raise ValueError("scenes must contain one item or exactly nframes items.")
    return scene_list


def hw_isp_simulate_sequence(
    camera: Camera,
    scenes: Any,
    config: HWIspConfig | None = None,
    exposure_times: Any = None,
    nframes: int | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> HWIspSequenceResult:
    """Simulate a sequence and preserve one timeline per computed frame."""

    store = asset_store or AssetStore.default()
    resolved_config = config or hw_isp_config()
    scene_list = _broadcast_frames(scenes, nframes)
    if exposure_times is None:
        exposure_list = [resolved_config.sensor_timing.exposure_time_us] * len(scene_list)
    else:
        values = np.asarray(exposure_times, dtype=float).reshape(-1).tolist()
        exposure_list = values * len(scene_list) if len(values) == 1 else values
    if len(exposure_list) != len(scene_list):
        raise ValueError("exposure_times must contain one value or one value per frame.")

    frames: list[HWIspFrameResult] = []
    visibility_history: list[float] = []
    current_camera = camera.clone()
    current_controls = _default_control_state(current_camera, resolved_config)
    current_controls["exposure_time_us"] = float(exposure_list[0])
    scheduled_controls: dict[int, dict[str, Any]] = {}
    for frame_id, (scene_item, exposure_us) in enumerate(zip(scene_list, exposure_list, strict=True)):
        if bool(resolved_config.control_path.apply_to_image):
            current_controls = _merge_control_update(current_controls, scheduled_controls.pop(frame_id, None))
            if not resolved_config.control_path.ae_enabled:
                current_controls["exposure_time_us"] = float(exposure_us)
            frame_exposure_us = float(current_controls["exposure_time_us"])
        else:
            frame_exposure_us = float(exposure_us)
        frame_config = HWIspConfig(
            sensor_timing=_replace_dataclass(
                resolved_config.sensor_timing,
                {"exposure_time_us": frame_exposure_us},
            ),
            stages=resolved_config.stages,
            control_path=resolved_config.control_path,
            transport=resolved_config.transport,
            global_latency_factor=resolved_config.global_latency_factor,
            seed=resolved_config.seed,
        )
        if not bool(resolved_config.control_path.apply_to_image):
            current_camera.fields["sensor"] = sensor_set(
                current_camera.fields["sensor"],
                "exposure duration",
                float(exposure_us) * 1.0e-6,
            )
        previous_payload: dict[str, Any] = {"app_visible_times_us": visibility_history}
        if bool(resolved_config.control_path.apply_to_image):
            previous_payload.update(
                {
                    "current_controls": current_controls,
                    "scheduled_controls": scheduled_controls,
                }
            )
        frame_result = hw_isp_simulate_frame(
            current_camera,
            scene_item,
            frame_config,
            frame_id=frame_id,
            previous_state=previous_payload,
            asset_store=store,
        )
        frames.append(frame_result)
        visibility_history.append(float(frame_result.timeline.timestamps_us["app_visible"]))
        current_camera = frame_result.camera
        if bool(resolved_config.control_path.apply_to_image):
            current_controls = copy.deepcopy(frame_result.controls_applied["applied_controls"])
            requested = frame_result.controls_applied.get("requested_controls", {})
            ae_request = requested.get("ae") if isinstance(requested, dict) else None
            if isinstance(ae_request, dict):
                apply_frame = int(ae_request["apply_frame"])
                scheduled_controls.setdefault(apply_frame, {}).update(
                    {
                        "exposure_time_us": float(ae_request["exposure_time_us"]),
                        "analog_gain": float(ae_request["analog_gain"]),
                        "ae_stats_frame": int(ae_request["source_frame"]),
                    }
                )
            awb_request = requested.get("awb") if isinstance(requested, dict) else None
            if isinstance(awb_request, dict):
                apply_frame = int(awb_request["apply_frame"])
                scheduled_controls.setdefault(apply_frame, {}).update(
                    {
                        "wb_gains_rgb": np.asarray(awb_request["wb_gains_rgb"], dtype=float).reshape(-1)[:3].astype(float).tolist(),
                        "awb_stats_frame": int(awb_request["source_frame"]),
                    }
                )

    aggregate = hw_isp_latency_summary(HWIspSequenceResult(tuple(frames), aggregate={}))
    return HWIspSequenceResult(frames=tuple(frames), aggregate=aggregate)


def hw_isp_timeline_table(result: HWIspFrameResult | HWIspSequenceResult) -> list[dict[str, float | int | str]]:
    """Return frame and stage timing rows suitable for reports."""

    frames = result.frames if isinstance(result, HWIspSequenceResult) else (result,)
    rows: list[dict[str, float | int | str]] = []
    for frame in frames:
        timestamps = frame.timeline.timestamps_us
        rows.append(
            {
                "frame_id": int(frame.timeline.frame_id),
                "type": "frame",
                "name": "frame",
                "start_us": float(timestamps["request"]),
                "end_us": float(timestamps["app_visible"]),
                "duration_us": float(timestamps["app_visible"] - timestamps["request"]),
                "queue_stall_us": float(frame.timeline.queue_stall_us),
            }
        )
        for stage in frame.timeline.stages:
            rows.append(
                {
                    "frame_id": int(frame.timeline.frame_id),
                    "type": "stage",
                    "name": stage.name,
                    "start_us": float(stage.start_us),
                    "end_us": float(stage.end_us),
                    "duration_us": float(stage.end_us - stage.start_us),
                    "queue_stall_us": 0.0,
                }
            )
    return rows


def _first_settle_frame(
    values: list[tuple[int, float]],
    *,
    threshold: float,
    consecutive: int,
    start_frame: int,
) -> float:
    if not values:
        return -1.0
    consecutive = max(int(consecutive), 1)
    for index in range(0, max(len(values) - consecutive + 1, 0)):
        frame_id = int(values[index][0])
        if frame_id < int(start_frame):
            continue
        window = values[index : index + consecutive]
        if len(window) == consecutive and all(abs(float(value)) <= float(threshold) for _, value in window):
            return float(frame_id)
    return -1.0


def _control_config_from_frame(frame: HWIspFrameResult) -> dict[str, Any]:
    metadata = frame.camera.metadata.get("hw_isp", {}) if hasattr(frame.camera, "metadata") else {}
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    control = config.get("control_path", {}) if isinstance(config, dict) else {}
    return control if isinstance(control, dict) else {}


def _sensor_config_from_frame(frame: HWIspFrameResult) -> dict[str, Any]:
    metadata = frame.camera.metadata.get("hw_isp", {}) if hasattr(frame.camera, "metadata") else {}
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    sensor = config.get("sensor_timing", {}) if isinstance(config, dict) else {}
    return sensor if isinstance(sensor, dict) else {}


def _three_a_aggregate(frames: tuple[HWIspFrameResult, ...]) -> dict[str, Any]:
    ae_errors: list[tuple[int, float]] = []
    awb_imbalances: list[tuple[int, float]] = []
    clipped: list[tuple[int, float]] = []
    warmup_count = 0
    for frame in frames:
        controls = frame.controls_applied
        stats = controls.get("produced_stats", {})
        ae_stats = stats.get("ae", {}) if isinstance(stats, dict) else {}
        awb_stats = stats.get("awb", {}) if isinstance(stats, dict) else {}
        frame_id = int(frame.timeline.frame_id)
        if bool(controls.get("warmup", False)):
            warmup_count += 1
        ae_errors.append((frame_id, float(ae_stats.get("ev_error", 0.0))))
        awb_imbalances.append((frame_id, float(awb_stats.get("corrected_rgb_imbalance", stats.get("awb_corrected_rgb_imbalance", 0.0)))))
        clipped.append((frame_id, float(ae_stats.get("clipped_fraction", stats.get("clipped_fraction", 0.0)))))

    ae_settle_frame = _first_settle_frame(
        ae_errors,
        threshold=_AE_SETTLE_EV_THRESHOLD,
        consecutive=_SETTLE_CONSECUTIVE_FRAMES,
        start_frame=_AE_STEP_FRAME,
    )
    awb_settle_frame = _first_settle_frame(
        awb_imbalances,
        threshold=_AWB_SETTLE_IMBALANCE_THRESHOLD,
        consecutive=_SETTLE_CONSECUTIVE_FRAMES,
        start_frame=_AWB_STEP_FRAME,
    )
    control_config = _control_config_from_frame(frames[0])
    sensor_config = _sensor_config_from_frame(frames[0])
    fps = float(sensor_config.get("fps", 30.0))
    max_exposure = (1.0e6 / max(fps, 1.0e-12)) * float(control_config.get("max_exposure_fraction", 0.90))
    min_exposure = float(control_config.get("min_exposure_time_us", 100.0))
    min_gain = float(control_config.get("min_analog_gain", 1.0))
    max_gain = float(control_config.get("max_analog_gain", 16.0))
    min_wb = float(control_config.get("min_wb_gain", 0.25))
    max_wb = float(control_config.get("max_wb_gain", 4.0))
    clamp_compliance = True
    for frame in frames:
        controls = frame.controls_applied
        exposure = float(controls.get("exposure_time_us", 0.0))
        gain = float(controls.get("analog_gain", 0.0))
        wb = np.asarray(controls.get("wb_gains_rgb", [1.0, 1.0, 1.0]), dtype=float)
        clamp_compliance = clamp_compliance and min_exposure <= exposure <= max(max_exposure, min_exposure)
        clamp_compliance = clamp_compliance and min_gain <= gain <= max_gain
        clamp_compliance = clamp_compliance and bool(np.all((wb >= min_wb) & (wb <= max_wb)))

    ae_enabled = bool(control_config.get("ae_enabled", True))
    awb_enabled = bool(control_config.get("awb_enabled", True))
    ae_delay = int(control_config.get("ae_apply_delay_frames", 2)) if ae_enabled else 0
    awb_delay = int(control_config.get("awb_apply_delay_frames", 2)) if awb_enabled else 0
    warmup_delay = max(ae_delay, awb_delay)
    warmup_delay_mapping = True
    for frame in frames:
        expected = int(frame.timeline.frame_id) < warmup_delay
        if bool(frame.controls_applied.get("warmup", False)) != expected:
            warmup_delay_mapping = False
            break

    pre_clip = [value for frame_id, value in clipped if _AE_STEP_FRAME <= frame_id < _AE_STEP_FRAME + ae_delay]
    post_clip = [value for frame_id, value in clipped if frame_id >= _AE_STEP_FRAME + ae_delay]
    max_clip_pre = max(pre_clip) if pre_clip else 0.0
    max_clip_post = max(post_clip) if post_clip else 0.0
    clip_reduction = max_clip_pre <= 0.0 or max_clip_post < max_clip_pre

    final_ae_error = float(ae_errors[-1][1]) if ae_errors else 0.0
    final_awb_imbalance = float(awb_imbalances[-1][1]) if awb_imbalances else 0.0
    return {
        "ae_settle_frame": float(ae_settle_frame),
        "ae_settle_frames_after_step": float(ae_settle_frame - _AE_STEP_FRAME) if ae_settle_frame >= 0 else -1.0,
        "ae_final_error_ev": final_ae_error,
        "awb_settle_frame": float(awb_settle_frame),
        "awb_final_rgb_imbalance": final_awb_imbalance,
        "max_clip_fraction": float(max((value for _, value in clipped), default=0.0)),
        "max_clip_fraction_before_response": float(max_clip_pre),
        "max_clip_fraction_after_response": float(max_clip_post),
        "warmup_frame_count": float(warmup_count),
        "validation_verdicts": {
            "ae_settle": bool(ae_settle_frame >= 0),
            "awb_settle": bool(awb_settle_frame >= 0),
            "clamp_compliance": bool(clamp_compliance),
            "warmup_delay_mapping": bool(warmup_delay_mapping),
            "clip_reduction": bool(clip_reduction),
        },
        "validation_thresholds": {
            "ae_settle_ev": float(_AE_SETTLE_EV_THRESHOLD),
            "awb_settle_imbalance": float(_AWB_SETTLE_IMBALANCE_THRESHOLD),
            "settle_consecutive_frames": float(_SETTLE_CONSECUTIVE_FRAMES),
        },
    }


def hw_isp_latency_summary(result: HWIspFrameResult | HWIspSequenceResult) -> dict[str, Any]:
    frames = result.frames if isinstance(result, HWIspSequenceResult) else (result,)
    if not frames:
        return {"frame_count": 0.0}
    e2e = np.asarray(
        [
            frame.timeline.timestamps_us["app_visible"] - frame.timeline.timestamps_us["request"]
            for frame in frames
        ],
        dtype=float,
    )
    queue = np.asarray([frame.timeline.queue_stall_us for frame in frames], dtype=float)
    summary: dict[str, Any] = {
        "frame_count": float(len(frames)),
        "e2e_latency_mean_us": float(np.mean(e2e)),
        "e2e_latency_max_us": float(np.max(e2e)),
        "e2e_latency_min_us": float(np.min(e2e)),
        "queue_stall_total_us": float(np.sum(queue)),
        "queue_stall_max_us": float(np.max(queue)),
    }
    summary.update(_three_a_aggregate(tuple(frames)))
    return summary


def _timeline_to_dict(timeline: HWIspFrameTimeline) -> dict[str, Any]:
    return {
        "frame_id": int(timeline.frame_id),
        "timestamps_us": dict(timeline.timestamps_us),
        "stages": [asdict(stage) for stage in timeline.stages],
        "queue_stall_us": float(timeline.queue_stall_us),
        "dropped": bool(timeline.dropped),
    }


def _config_to_dict(config: HWIspConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["stages"] = [asdict(stage) for stage in config.stages]
    return payload


def _frame_to_dict(frame: HWIspFrameResult) -> dict[str, Any]:
    return {
        "timeline": _timeline_to_dict(frame.timeline),
        "controls_applied": copy.deepcopy(frame.controls_applied),
        "latency_summary": hw_isp_latency_summary(frame),
    }


def hw_isp_export_json(result: HWIspFrameResult | HWIspSequenceResult, path: str | Path) -> Path:
    """Write a stable JSON timing summary."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, HWIspSequenceResult):
        payload = {
            "type": "sequence",
            "aggregate": dict(result.aggregate),
            "frames": [_frame_to_dict(frame) for frame in result.frames],
        }
    else:
        payload = {"type": "frame", **_frame_to_dict(result)}
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


hwISPConfig = hw_isp_config
hwISPSimulateFrame = hw_isp_simulate_frame
hwISPSimulateSequence = hw_isp_simulate_sequence
hwISPTimelineTable = hw_isp_timeline_table
hwISPLatencySummary = hw_isp_latency_summary
hwISPExportJSON = hw_isp_export_json
