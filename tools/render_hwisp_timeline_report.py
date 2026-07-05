"""Render a hardware ISP latency timeline report."""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import AssetStore, camera_create, scene_create
from pyisetcam.hwisp import (
    hw_isp_config,
    hw_isp_export_json,
    hw_isp_latency_summary,
    hw_isp_simulate_sequence,
    hw_isp_timeline_table,
)
from pyisetcam.hwisp_db import hw_isp_config_from_profile

DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "hwisp"


def _render_frame_timeline(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    stage_rows = [row for row in rows if row["type"] == "stage"]
    frame_ids = sorted({int(row["frame_id"]) for row in stage_rows})
    stage_names = list(dict.fromkeys(str(row["name"]) for row in stage_rows))
    y_lookup = {(frame_id, stage): index for index, (frame_id, stage) in enumerate((f, s) for f in frame_ids for s in stage_names)}

    fig_height = max(4.0, 0.35 * max(len(y_lookup), 1))
    fig, axis = plt.subplots(figsize=(11.5, fig_height), constrained_layout=True)
    for row in stage_rows:
        frame_id = int(row["frame_id"])
        name = str(row["name"])
        y = y_lookup[(frame_id, name)]
        start = float(row["start_us"]) / 1000.0
        duration = float(row["duration_us"]) / 1000.0
        axis.broken_barh([(start, duration)], (y - 0.38, 0.76), facecolors=f"C{stage_names.index(name) % 10}")
    axis.set_yticks(list(y_lookup.values()))
    axis.set_yticklabels([f"F{frame_id} {stage}" for frame_id in frame_ids for stage in stage_names])
    axis.set_xlabel("Time (ms)")
    axis.set_title("HW ISP Stage Timeline")
    axis.grid(axis="x", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _render_stage_latency(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    stage_rows = [row for row in rows if row["type"] == "stage"]
    stage_names = list(dict.fromkeys(str(row["name"]) for row in stage_rows))
    means = []
    for name in stage_names:
        durations = [float(row["duration_us"]) / 1000.0 for row in stage_rows if str(row["name"]) == name]
        means.append(float(np.mean(durations)))

    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    axis.bar(stage_names, means, color="#4477aa")
    axis.set_ylabel("Mean latency (ms)")
    axis.set_title("Mean ISP Stage Latency")
    axis.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _render_e2e_latency(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    frame_rows = [row for row in rows if row["type"] == "frame"]
    frame_ids = [int(row["frame_id"]) for row in frame_rows]
    e2e_ms = [float(row["duration_us"]) / 1000.0 for row in frame_rows]
    stalls_ms = [float(row["queue_stall_us"]) / 1000.0 for row in frame_rows]

    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    axis.plot(frame_ids, e2e_ms, marker="o", label="E2E latency")
    axis.bar(frame_ids, stalls_ms, alpha=0.35, label="Queue stall")
    axis.set_xlabel("Frame")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Frame Latency")
    axis.grid(alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _scale_scene(scene, scale: float):
    result = scene.clone()
    result.data["photons"] = np.asarray(result.data["photons"], dtype=float) * float(scale)
    return result


def _warm_scene(scene):
    result = scene.clone()
    wave = np.asarray(result.fields["wave"], dtype=float)
    spectral_tilt = np.interp(wave, [float(np.min(wave)), float(np.max(wave))], [0.4, 2.2])
    result.data["photons"] = np.asarray(result.data["photons"], dtype=float) * spectral_tilt.reshape(1, 1, -1)
    return result


def _highlight_scene(scene):
    result = scene.clone()
    photons = np.asarray(result.data["photons"], dtype=float).copy()
    rows, cols = photons.shape[:2]
    row_end = max(rows // 4, 1)
    col_end = max(cols // 4, 1)
    photons[:row_end, :col_end, :] *= 24.0
    result.data["photons"] = photons
    return result


def _three_a_scenes(store: AssetStore) -> list:
    nominal = scene_create("macbeth d65", 16, asset_store=store)
    bright = _scale_scene(nominal, 4.0)
    highlight = _highlight_scene(bright)
    warm = _warm_scene(bright)
    return [nominal] * 4 + [highlight] * 4 + [warm] * 4


def _three_a_config(profile: str | None = None):
    overrides = {
        "sensor_timing": {"fps": 30.0, "line_time_us": 15.2, "exposure_time_us": 8000.0},
        "control_path": {
            "apply_to_image": True,
            "ae_apply_delay_frames": 2,
            "awb_apply_delay_frames": 2,
            "target_luma": 0.18,
            "max_ae_step_ev": 1.0,
            "max_awb_step_ev": 0.5,
            "ae_metering": "center_weighted",
            "ae_highlight_clip": 0.98,
            "ae_highlight_weight": 0.5,
            "awb_stats_roi": "valid_luma",
        },
        "transport": {"request_queue_depth": 2, "max_buffers": 3},
        "seed": 42,
    }
    if profile:
        return hw_isp_config_from_profile(profile, **overrides)
    return hw_isp_config(**overrides)


def _simulate_three_a_sequence(store: AssetStore, profile: str | None = None):
    config = _three_a_config(profile)
    return hw_isp_simulate_sequence(
        camera_create(asset_store=store),
        _three_a_scenes(store),
        config,
        asset_store=store,
    )


def _render_ae_convergence(sequence, output_path: Path) -> None:
    frames = [frame.timeline.frame_id for frame in sequence.frames]
    stats = [frame.controls_applied["produced_stats"] for frame in sequence.frames]
    controls = [frame.controls_applied for frame in sequence.frames]
    ae_stats = [item.get("ae", {}) for item in stats]
    luma = [float(item.get("metering_luma", stats[index]["mean_sensor_luma_norm"])) for index, item in enumerate(ae_stats)]
    target = [float(item.get("target_luma", stats[index]["target_luma"])) for index, item in enumerate(ae_stats)]
    ev_error = [float(item.get("ev_error", 0.0)) for item in ae_stats]
    clip_fraction = [float(item.get("clipped_fraction", 0.0)) for item in ae_stats]
    exposure_ms = [float(item["exposure_time_us"]) / 1000.0 for item in controls]
    analog_gain = [float(item["analog_gain"]) for item in controls]
    exposure_product = [exposure_ms[index] * analog_gain[index] for index in range(len(frames))]
    ae_settle = float(sequence.aggregate.get("ae_settle_frame", -1.0))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(frames, luma, marker="o", label="metered luma")
    axes[0].plot(frames, target, linestyle="--", label="target luma")
    axes[0].axvline(4, color="#9a5a21", alpha=0.35, label="4x brightness + highlight")
    if ae_settle >= 0:
        axes[0].axvline(ae_settle, color="#338a3e", linestyle=":", alpha=0.8, label="AE settle")
    axes[0].set_ylabel("Normalized luma")
    axes[0].set_title("AE Convergence With H3A-like Metering")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(frames, exposure_ms, marker="o", label="exposure")
    axes[1].plot(frames, analog_gain, marker="s", label="analog gain")
    axes[1].plot(frames, exposure_product, marker="^", label="exposure x gain")
    axes[1].set_ylabel("Exposure ms / gain")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(frames, ev_error, marker="o", label="EV error")
    axes[2].plot(frames, clip_fraction, marker="s", label="clip fraction")
    axes[2].axhline(0.25, color="#338a3e", linestyle=":", alpha=0.7, label="+/-0.25 EV")
    axes[2].axhline(-0.25, color="#338a3e", linestyle=":", alpha=0.7)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("EV / fraction")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _render_awb_convergence(sequence, output_path: Path) -> None:
    frames = [frame.timeline.frame_id for frame in sequence.frames]
    stats = [frame.controls_applied["produced_stats"] for frame in sequence.frames]
    controls = [frame.controls_applied for frame in sequence.frames]
    awb_stats = [item.get("awb", {}) for item in stats]
    corrected = np.asarray(
        [item.get("corrected_rgb_means", stats[index]["awb_corrected_rgb_means"]) for index, item in enumerate(awb_stats)],
        dtype=float,
    )
    raw_imbalance = [float(item.get("rgb_imbalance", stats[index]["sensor_rgb_imbalance"])) for index, item in enumerate(awb_stats)]
    corrected_imbalance = [
        float(item.get("corrected_rgb_imbalance", stats[index]["awb_corrected_rgb_imbalance"]))
        for index, item in enumerate(awb_stats)
    ]
    gains = np.asarray([item["wb_gains_rgb"] for item in controls], dtype=float)
    awb_settle = float(sequence.aggregate.get("awb_settle_frame", -1.0))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    for index, channel in enumerate(("R", "G", "B")):
        axes[0].plot(frames, corrected[:, index], marker="o", label=f"{channel} corrected mean")
        axes[1].plot(frames, gains[:, index], marker="s", label=f"{channel} WB gain")
    axes[0].axvline(8, color="#9a5a21", alpha=0.35, label="warm illuminant step")
    if awb_settle >= 0:
        axes[0].axvline(awb_settle, color="#338a3e", linestyle=":", alpha=0.8, label="AWB settle")
    axes[0].set_title("AWB Convergence With Valid-luma Gray-world Stats")
    axes[0].set_ylabel("Corrected channel mean")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_ylabel("WB gain")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].plot(frames, raw_imbalance, marker="o", label="raw RGB imbalance")
    axes[2].plot(frames, corrected_imbalance, marker="s", label="corrected RGB imbalance")
    axes[2].axhline(0.20, color="#338a3e", linestyle=":", alpha=0.7, label="0.20 settle threshold")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Imbalance")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _frame_preview(frame) -> np.ndarray:
    image = frame.ip.data.get("srgb")
    if image is None:
        image = frame.ip.data.get("result")
    array = np.asarray(image, dtype=float)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        return np.zeros((8, 8, 3), dtype=float)
    return np.clip(array[:, :, :3], 0.0, 1.0)


def _render_three_a_thumbnails(sequence, output_path: Path) -> None:
    selected = [0, 4, 6, 8, 10, 11]
    selected = [index for index in selected if index < len(sequence.frames)]
    fig, axes = plt.subplots(1, len(selected), figsize=(2.2 * len(selected), 2.6), constrained_layout=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, frame_index in zip(axes, selected, strict=True):
        frame = sequence.frames[frame_index]
        controls = frame.controls_applied
        axis.imshow(_frame_preview(frame))
        axis.set_title(
            f"F{frame_index}\nE={controls['exposure_time_us'] / 1000.0:.1f} ms\nWB={controls['wb_gains_rgb'][0]:.2f}/{controls['wb_gains_rgb'][2]:.2f}",
            fontsize=8,
        )
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _ms(value_us: float) -> str:
    return f"{float(value_us) / 1000.0:.3f}"


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #16202a;
      --muted: #5d6b78;
      --line: #d9e0e7;
      --card: #ffffff;
      --bg: #f4f7fa;
      --accent: #215f9a;
      --warn: #9a5a21;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      line-height: 1.15;
      margin: 0 0 14px;
    }}
    h1 {{
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    h2 {{
      margin-top: 34px;
      font-size: 22px;
    }}
    a {{
      color: var(--accent);
    }}
    .lead {{
      max-width: 900px;
      color: var(--muted);
      font-size: 17px;
    }}
    .cards {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin: 26px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgb(22 32 42 / 6%);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .value {{
      display: block;
      margin-top: 6px;
      font-size: 26px;
      font-weight: 760;
      letter-spacing: -0.03em;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    figure {{
      margin: 0;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgb(22 32 42 / 6%);
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }}
    figcaption {{
      padding: 12px 14px 14px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgb(22 32 42 / 6%);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #eaf0f6;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 0;
    }}
    .links a {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      text-decoration: none;
    }}
    .note {{
      border-left: 4px solid var(--warn);
      background: #fff7ea;
      padding: 12px 14px;
      margin: 18px 0;
    }}
    .scroll {{
      overflow-x: auto;
    }}
    .pass {{
      color: #226b32;
      font-weight: 700;
    }}
    .fail {{
      color: #a13b2b;
      font-weight: 700;
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _figure_html(path: Path, title: str, caption: str) -> str:
    return (
        "<figure>"
        f'<img src="{escape(path.name)}" alt="{escape(title)}">'
        f"<figcaption><strong>{escape(title)}</strong><br>{escape(caption)}</figcaption>"
        "</figure>"
    )


def _summary_cards(summary: dict[str, object]) -> str:
    cards = [
        ("Frames", f"{int(summary['frame_count'])}"),
        ("Mean E2E", f"{_ms(summary['e2e_latency_mean_us'])} ms"),
        ("Max E2E", f"{_ms(summary['e2e_latency_max_us'])} ms"),
        ("Total Queue Stall", f"{_ms(summary['queue_stall_total_us'])} ms"),
        ("Max Queue Stall", f"{_ms(summary['queue_stall_max_us'])} ms"),
    ]
    return "\n".join(
        f'<div class="card"><span class="label">{escape(label)}</span>'
        f'<span class="value">{escape(value)}</span></div>'
        for label, value in cards
    )


def _frame_table_html(sequence) -> str:
    rows = []
    for frame in sequence.frames:
        ts = frame.timeline.timestamps_us
        controls = frame.controls_applied
        rows.append(
            "<tr>"
            f"<td>{frame.timeline.frame_id}</td>"
            f"<td>{_ms(ts['requested'])}</td>"
            f"<td>{_ms(ts['request'])}</td>"
            f"<td>{_ms(ts['readout_end'])}</td>"
            f"<td>{_ms(ts['isp_done'])}</td>"
            f"<td>{_ms(ts['app_visible'])}</td>"
            f"<td>{_ms(ts['app_visible'] - ts['request'])}</td>"
            f"<td>{_ms(frame.timeline.queue_stall_us)}</td>"
            f"<td>{escape(str(controls['warmup']))}</td>"
            f"<td>{escape(str(controls['ae_stats_frame']))}</td>"
            f"<td>{escape(str(controls['awb_stats_frame']))}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Frame</th><th>Requested ms</th><th>Actual request ms</th>"
        "<th>Readout end ms</th><th>ISP done ms</th><th>App visible ms</th>"
        "<th>E2E ms</th><th>Queue stall ms</th><th>Warmup</th>"
        "<th>AE stats frame</th><th>AWB stats frame</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def _three_a_table_html(sequence) -> str:
    rows = []
    for frame in sequence.frames:
        controls = frame.controls_applied
        stats = controls["produced_stats"]
        ae_stats = stats.get("ae", {})
        awb_stats = stats.get("awb", {})
        requested = controls["requested_controls"]
        ae_request = requested.get("ae") if isinstance(requested, dict) else None
        awb_request = requested.get("awb") if isinstance(requested, dict) else None
        rows.append(
            "<tr>"
            f"<td>{frame.timeline.frame_id}</td>"
            f"<td>{escape(str(controls['warmup']))}</td>"
            f"<td>{escape(str(controls['ae_stats_frame']))}</td>"
            f"<td>{escape(str(controls['awb_stats_frame']))}</td>"
            f"<td>{controls['exposure_time_us'] / 1000.0:.3f}</td>"
            f"<td>{controls['analog_gain']:.3f}</td>"
            f"<td>{ae_stats.get('metering_luma', stats['mean_sensor_luma_norm']):.4f}</td>"
            f"<td>{ae_stats.get('ev_error', 0.0):.3f}</td>"
            f"<td>{ae_stats.get('clipped_fraction', 0.0):.4f}</td>"
            f"<td>{stats['target_luma']:.4f}</td>"
            f"<td>{controls['wb_gains_rgb'][0]:.3f}</td>"
            f"<td>{controls['wb_gains_rgb'][1]:.3f}</td>"
            f"<td>{controls['wb_gains_rgb'][2]:.3f}</td>"
            f"<td>{awb_stats.get('valid_pixel_fraction', 1.0):.3f}</td>"
            f"<td>{stats['awb_corrected_rgb_imbalance']:.3f}</td>"
            f"<td>{'' if ae_request is None else ae_request['apply_frame']}</td>"
            f"<td>{'' if awb_request is None else awb_request['apply_frame']}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Frame</th><th>Warmup</th><th>AE source</th><th>AWB source</th>"
        "<th>Exposure ms</th><th>Analog gain</th><th>Metered luma</th><th>EV error</th>"
        "<th>Clip fraction</th><th>Target</th>"
        "<th>WB R</th><th>WB G</th><th>WB B</th><th>AWB valid pixels</th><th>Corrected RGB imbalance</th>"
        "<th>Next AE frame</th><th>Next AWB frame</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def _three_a_verdict_html(sequence) -> str:
    aggregate = sequence.aggregate
    verdicts = aggregate.get("validation_verdicts", {})
    thresholds = aggregate.get("validation_thresholds", {})
    rows = [
        (
            "AE settle",
            bool(verdicts.get("ae_settle", False)),
            f"settle frame {aggregate.get('ae_settle_frame', -1):.0f}, final error {aggregate.get('ae_final_error_ev', 0.0):.3f} EV",
        ),
        (
            "AWB settle",
            bool(verdicts.get("awb_settle", False)),
            f"settle frame {aggregate.get('awb_settle_frame', -1):.0f}, final imbalance {aggregate.get('awb_final_rgb_imbalance', 0.0):.3f}",
        ),
        (
            "Clamp compliance",
            bool(verdicts.get("clamp_compliance", False)),
            "exposure, analog gain, and WB gains remain inside configured limits",
        ),
        (
            "Warmup delay mapping",
            bool(verdicts.get("warmup_delay_mapping", False)),
            "frames before the configured control delay are marked warmup",
        ),
        (
            "Clip reduction",
            bool(verdicts.get("clip_reduction", False)),
            f"max clip before response {aggregate.get('max_clip_fraction_before_response', 0.0):.4f}, after response {aggregate.get('max_clip_fraction_after_response', 0.0):.4f}",
        ),
    ]
    body = []
    for name, passed, detail in rows:
        klass = "pass" if passed else "fail"
        status = "PASS" if passed else "FAIL"
        body.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f'<td class="{klass}">{status}</td>'
            f"<td>{escape(detail)}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Validation</th><th>Status</th><th>Evidence</th>"
        "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table></div>"
        f"<p class=\"lead\">Default thresholds: AE |EV error| <= {thresholds.get('ae_settle_ev', 0.25):.2f} for two consecutive frames; "
        f"AWB corrected RGB imbalance <= {thresholds.get('awb_settle_imbalance', 0.20):.2f} for two consecutive frames.</p>"
    )


def _stage_table_html(sequence) -> str:
    rows = []
    for frame in sequence.frames:
        for stage in frame.timeline.stages:
            rows.append(
                "<tr>"
                f"<td>{frame.timeline.frame_id}</td>"
                f"<td>{escape(stage.name)}</td>"
                f"<td>{escape(stage.domain)}</td>"
                f"<td>{escape(stage.buffering)}</td>"
                f"<td>{_ms(stage.start_us)}</td>"
                f"<td>{_ms(stage.end_us)}</td>"
                f"<td>{_ms(stage.latency_us)}</td>"
                f"<td>{_ms(stage.line_buffer_delay_us)}</td>"
                f"<td>{_ms(stage.cycle_latency_us)}</td>"
                "</tr>"
            )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Frame</th><th>Stage</th><th>Domain</th><th>Buffering</th>"
        "<th>Start ms</th><th>End ms</th><th>Latency ms</th>"
        "<th>Line-buffer delay ms</th><th>Cycle latency ms</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def _write_html_reports(
    *,
    sequence,
    summary: dict[str, object],
    report_html: Path,
    details_html: Path,
    frame_timeline: Path,
    stage_latency: Path,
    e2e_latency: Path,
    three_a_sequence,
    ae_convergence: Path,
    awb_convergence: Path,
    three_a_thumbnails: Path,
    three_a_summary_json: Path,
    report_md: Path,
    summary_json: Path,
) -> None:
    dashboard_body = f"""
<h1>HW ISP Simulation Report</h1>
<p class="lead">This report visualizes deterministic timing metadata layered on top of the normal pyisetcam image pipeline. Pixel values still come from the existing camera pipeline; this report focuses on exposure, readout, ISP stage, queue, DMA, and app-visible timing.</p>
<div class="cards">
{_summary_cards(summary)}
</div>
<div class="note">Queue stall is reported separately from per-frame E2E latency. E2E latency is measured from the actual request time after stall to app visibility.</div>
<div class="links">
  <a href="{escape(details_html.name)}">Frame and stage details</a>
  <a href="{escape(summary_json.name)}">Machine-readable JSON</a>
  <a href="{escape(report_md.name)}">Markdown report</a>
</div>
<h2>Timing Figures</h2>
<div class="grid">
  {_figure_html(frame_timeline, "Frame timeline", "Per-frame ISP stage spans in wall-clock time.")}
  {_figure_html(stage_latency, "Stage latency", "Mean modeled latency by HW ISP block.")}
  {_figure_html(e2e_latency, "E2E latency and queue stall", "Application-visible latency and queue back-pressure per frame.")}
</div>
<h2>Frame Summary</h2>
{_frame_table_html(sequence)}
<h2>3A AE/AWB Behavior</h2>
<p class="lead">This sequence enables <code>apply_to_image=True</code>. Frame statistics generate AE/AWB controls that apply two frames later, so the plots show delayed control response rather than instantaneous correction.</p>
<div class="grid">
  {_figure_html(ae_convergence, "AE convergence", "H3A-like metered luma, clipping fraction, EV error, exposure time, and analog gain across a 4x brightness step with a highlight patch.")}
  {_figure_html(awb_convergence, "AWB convergence", "Valid-luma gray-world RGB means, channel imbalance, and white-balance gains across a warm-illuminant step.")}
  {_figure_html(three_a_thumbnails, "3A output thumbnails", "Representative rendered frames before and after AE/AWB delayed control application.")}
</div>
<div class="links">
  <a href="{escape(three_a_summary_json.name)}">3A machine-readable JSON</a>
</div>
<h2>3A Validation Verdict</h2>
{_three_a_verdict_html(three_a_sequence)}
<h2>3A Frame Controls</h2>
{_three_a_table_html(three_a_sequence)}
"""
    details_body = f"""
<h1>HW ISP Frame Details</h1>
<p class="lead">Detailed timestamps for every simulated frame and every enabled ISP stage. All times are milliseconds from the start of the simulation.</p>
<div class="links">
  <a href="{escape(report_html.name)}">Back to dashboard</a>
  <a href="{escape(summary_json.name)}">Machine-readable JSON</a>
</div>
<h2>Frame Timing</h2>
{_frame_table_html(sequence)}
<h2>Stage Timing</h2>
{_stage_table_html(sequence)}
"""
    report_html.write_text(_html_page("HW ISP Simulation Report", dashboard_body), encoding="utf-8")
    details_html.write_text(_html_page("HW ISP Frame Details", details_body), encoding="utf-8")


def render(output_dir: Path = DEFAULT_OUTPUT_DIR, *, nframes: int = 8, profile: str | None = None) -> dict[str, Path]:
    store = AssetStore.default()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_overrides = dict(
        sensor_timing={"fps": 100.0, "line_time_us": 15.2, "exposure_time_us": 8000.0},
        transport={
            "request_queue_depth": 1,
            "max_buffers": 1,
            "dma_submit_us": 120.0,
            "dma_complete_us": 320.0,
            "app_processing_us": 500.0,
        },
        seed=42,
    )
    config = hw_isp_config_from_profile(profile, **config_overrides) if profile else hw_isp_config(**config_overrides)
    scene = scene_create("uniform ee", 8, asset_store=store)
    sequence = hw_isp_simulate_sequence(
        camera_create(asset_store=store),
        scene,
        config,
        nframes=nframes,
        asset_store=store,
    )
    three_a_sequence = _simulate_three_a_sequence(store, profile)
    rows = hw_isp_timeline_table(sequence)

    frame_timeline = output_dir / "frame_timeline.png"
    stage_latency = output_dir / "stage_latency.png"
    e2e_latency = output_dir / "e2e_latency.png"
    ae_convergence = output_dir / "ae_convergence.png"
    awb_convergence = output_dir / "awb_convergence.png"
    three_a_thumbnails = output_dir / "three_a_thumbnails.png"
    summary_json = output_dir / "timeline_summary.json"
    three_a_summary_json = output_dir / "three_a_summary.json"
    report_md = output_dir / "timeline_report.md"
    report_html = output_dir / "index.html"
    details_html = output_dir / "frame_details.html"

    _render_frame_timeline(rows, frame_timeline)
    _render_stage_latency(rows, stage_latency)
    _render_e2e_latency(rows, e2e_latency)
    _render_ae_convergence(three_a_sequence, ae_convergence)
    _render_awb_convergence(three_a_sequence, awb_convergence)
    _render_three_a_thumbnails(three_a_sequence, three_a_thumbnails)
    hw_isp_export_json(sequence, summary_json)
    hw_isp_export_json(three_a_sequence, three_a_summary_json)

    summary = hw_isp_latency_summary(sequence)
    three_a_summary = hw_isp_latency_summary(three_a_sequence)
    _write_html_reports(
        sequence=sequence,
        summary=summary,
        report_html=report_html,
        details_html=details_html,
        frame_timeline=frame_timeline,
        stage_latency=stage_latency,
        e2e_latency=e2e_latency,
        three_a_sequence=three_a_sequence,
        ae_convergence=ae_convergence,
        awb_convergence=awb_convergence,
        three_a_thumbnails=three_a_thumbnails,
        three_a_summary_json=three_a_summary_json,
        report_md=report_md,
        summary_json=summary_json,
    )
    report = [
        "# HW ISP Timeline Report",
        "",
        "## Summary",
        f"- Frames: `{int(summary['frame_count'])}`",
        f"- Mean E2E latency: `{summary['e2e_latency_mean_us'] / 1000.0:.3f} ms`",
        f"- Max E2E latency: `{summary['e2e_latency_max_us'] / 1000.0:.3f} ms`",
        f"- Total queue stall: `{summary['queue_stall_total_us'] / 1000.0:.3f} ms`",
        f"- HW ISP profile: `{profile or 'inline defaults'}`",
        f"- HTML dashboard: [{report_html}]({report_html})",
        f"- HTML details: [{details_html}]({details_html})",
        f"- 3A summary JSON: [{three_a_summary_json}]({three_a_summary_json})",
        "",
        "## 3A Validation",
        f"- AE settle frame: `{three_a_summary['ae_settle_frame']:.0f}`",
        f"- AE final error: `{three_a_summary['ae_final_error_ev']:.3f} EV`",
        f"- AWB settle frame: `{three_a_summary['awb_settle_frame']:.0f}`",
        f"- AWB final RGB imbalance: `{three_a_summary['awb_final_rgb_imbalance']:.3f}`",
        f"- Max clip fraction: `{three_a_summary['max_clip_fraction']:.4f}`",
        "",
        "## Figures",
        f"![frame timeline]({frame_timeline})",
        f"![stage latency]({stage_latency})",
        f"![e2e latency]({e2e_latency})",
        f"![ae convergence]({ae_convergence})",
        f"![awb convergence]({awb_convergence})",
        f"![3a thumbnails]({three_a_thumbnails})",
        "",
        "## Regenerate",
        "- `python tools/render_hwisp_timeline_report.py`",
        f"- JSON: [{summary_json}]({summary_json})",
    ]
    report_md.write_text("\n".join(report) + "\n")
    return {
        "report": report_md,
        "html": report_html,
        "details_html": details_html,
        "summary": summary_json,
        "three_a_summary": three_a_summary_json,
        "frame_timeline": frame_timeline,
        "stage_latency": stage_latency,
        "e2e_latency": e2e_latency,
        "ae_convergence": ae_convergence,
        "awb_convergence": awb_convergence,
        "three_a_thumbnails": three_a_thumbnails,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument("--profile", default=None, help="Named HW ISP parameter profile to use.")
    args = parser.parse_args()
    outputs = render(args.output_dir, nframes=args.nframes, profile=args.profile)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
