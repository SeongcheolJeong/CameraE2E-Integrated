"""Render detailed HW ISP technical report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam.hwisp_db import hw_isp_profile, hw_isp_profile_names


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "hwisp"
DEFAULT_PROFILE = "rpi_vc4_imx219_public_seed"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _architecture_svg() -> str:
    return """
<svg class="diagram" viewBox="0 0 1260 520" role="img" aria-label="Detailed HW ISP architecture">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2d3a45" />
    </marker>
  </defs>
  <g class="lane">
    <text x="48" y="42">Image Data Path</text>
    <rect x="36" y="62" width="150" height="72" rx="14" fill="#f7d794" />
    <text x="111" y="92" text-anchor="middle">Scene</text>
    <text x="111" y="112" text-anchor="middle">photons</text>

    <rect x="234" y="62" width="150" height="72" rx="14" fill="#f7d794" />
    <text x="309" y="92" text-anchor="middle">Optics / OI</text>
    <text x="309" y="112" text-anchor="middle">photons</text>

    <rect x="432" y="62" width="168" height="72" rx="14" fill="#a3d8e6" />
    <text x="516" y="88" text-anchor="middle">Sensor</text>
    <text x="516" y="108" text-anchor="middle">exposure, gain, readout</text>

    <rect x="648" y="62" width="248" height="72" rx="14" fill="#cdb4db" />
    <text x="772" y="88" text-anchor="middle">ISP Core</text>
    <text x="772" y="108" text-anchor="middle">BLC, DPC, LSC, demosaic, CCM, gamma</text>

    <rect x="944" y="62" width="128" height="72" rx="14" fill="#ffafcc" />
    <text x="1008" y="92" text-anchor="middle">DMA</text>
    <text x="1008" y="112" text-anchor="middle">driver queue</text>

    <rect x="1120" y="62" width="104" height="72" rx="14" fill="#ffafcc" />
    <text x="1172" y="92" text-anchor="middle">App</text>
    <text x="1172" y="112" text-anchor="middle">visible</text>
  </g>

  <g class="lane">
    <text x="48" y="214">Control And Metadata Path</text>
    <rect x="432" y="234" width="168" height="72" rx="14" fill="#b7e4c7" />
    <text x="516" y="260" text-anchor="middle">H3A-like Stats</text>
    <text x="516" y="280" text-anchor="middle">tile luma, RGB means</text>

    <rect x="648" y="234" width="168" height="72" rx="14" fill="#b7e4c7" />
    <text x="732" y="260" text-anchor="middle">AE/AWB Logic</text>
    <text x="732" y="280" text-anchor="middle">request controls</text>

    <rect x="864" y="234" width="168" height="72" rx="14" fill="#b7e4c7" />
    <text x="948" y="260" text-anchor="middle">Delay Queue</text>
    <text x="948" y="280" text-anchor="middle">N + apply_delay</text>

    <rect x="234" y="382" width="168" height="72" rx="14" fill="#d8e2dc" />
    <text x="318" y="408" text-anchor="middle">Parameter DB</text>
    <text x="318" y="428" text-anchor="middle">profile, stages, limits</text>

    <rect x="648" y="382" width="248" height="72" rx="14" fill="#d8e2dc" />
    <text x="772" y="408" text-anchor="middle">Timeline Metadata</text>
    <text x="772" y="428" text-anchor="middle">camera.metadata / ip.metadata</text>
  </g>

  <g class="arrows">
    <line x1="186" y1="98" x2="234" y2="98" marker-end="url(#arrow)" />
    <line x1="384" y1="98" x2="432" y2="98" marker-end="url(#arrow)" />
    <line x1="600" y1="98" x2="648" y2="98" marker-end="url(#arrow)" />
    <line x1="896" y1="98" x2="944" y2="98" marker-end="url(#arrow)" />
    <line x1="1072" y1="98" x2="1120" y2="98" marker-end="url(#arrow)" />

    <line x1="516" y1="134" x2="516" y2="234" marker-end="url(#arrow)" />
    <line x1="600" y1="270" x2="648" y2="270" marker-end="url(#arrow)" />
    <line x1="816" y1="270" x2="864" y2="270" marker-end="url(#arrow)" />
    <line x1="948" y1="234" x2="520" y2="134" marker-end="url(#arrow)" />
    <line x1="402" y1="418" x2="648" y2="418" marker-end="url(#arrow)" />
    <line x1="772" y1="382" x2="772" y2="134" marker-end="url(#arrow)" />
  </g>
</svg>
"""


def _sequence_svg() -> str:
    return """
<svg class="diagram compact" viewBox="0 0 980 300" role="img" aria-label="AE/AWB delayed feedback sequence">
  <defs>
    <marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#2d3a45" />
    </marker>
  </defs>
  <text x="34" y="38">Delayed 3A feedback timing</text>
  <g>
    <rect x="40" y="70" width="120" height="52" rx="12" fill="#a3d8e6" />
    <text x="100" y="100" text-anchor="middle">Frame N</text>
    <rect x="230" y="70" width="120" height="52" rx="12" fill="#b7e4c7" />
    <text x="290" y="92" text-anchor="middle">Stats</text>
    <text x="290" y="110" text-anchor="middle">ready</text>
    <rect x="420" y="70" width="120" height="52" rx="12" fill="#b7e4c7" />
    <text x="480" y="92" text-anchor="middle">AE/AWB</text>
    <text x="480" y="110" text-anchor="middle">request</text>
    <rect x="610" y="70" width="120" height="52" rx="12" fill="#b7e4c7" />
    <text x="670" y="92" text-anchor="middle">Delay</text>
    <text x="670" y="110" text-anchor="middle">queue</text>
    <rect x="800" y="70" width="140" height="52" rx="12" fill="#a3d8e6" />
    <text x="870" y="92" text-anchor="middle">Frame N + d</text>
    <text x="870" y="110" text-anchor="middle">controls applied</text>
  </g>
  <g class="arrows">
    <line x1="160" y1="96" x2="230" y2="96" marker-end="url(#arrow2)" />
    <line x1="350" y1="96" x2="420" y2="96" marker-end="url(#arrow2)" />
    <line x1="540" y1="96" x2="610" y2="96" marker-end="url(#arrow2)" />
    <line x1="730" y1="96" x2="800" y2="96" marker-end="url(#arrow2)" />
  </g>
  <text x="40" y="176">Current implementation stores warmup, source stats frame, requested controls, and applied controls per frame.</text>
  <text x="40" y="204">If apply_to_image=False, pixels remain unchanged and controls are metadata only.</text>
  <text x="40" y="232">If apply_to_image=True, delayed exposure/gain and WB gains are applied to cloned sensor/IP objects.</text>
</svg>
"""


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #18232f;
      --muted: #5a6877;
      --line: #d4dde6;
      --bg: #f4f6f8;
      --card: #ffffff;
      --accent: #245d79;
      --code: #eef3f5;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgb(36 93 121 / 10%), transparent 34rem),
        linear-gradient(135deg, #f7f2e8 0%, #eef7f8 100%);
      color: var(--ink);
      font: 15px/1.58 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 24px 56px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      letter-spacing: -0.04em;
      line-height: 1.1;
    }}
    h2 {{
      margin: 36px 0 14px;
      font-size: 24px;
      letter-spacing: -0.015em;
    }}
    h3 {{
      margin: 22px 0 10px;
      font-size: 18px;
    }}
    .lead {{
      max-width: 920px;
      color: var(--muted);
      font-size: 17px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .card, .panel, table {{
      background: rgb(255 255 255 / 94%);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 28px rgb(24 35 47 / 7%);
    }}
    .card, .panel {{
      padding: 17px 18px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .value {{
      display: block;
      margin-top: 6px;
      font-size: 25px;
      font-weight: 780;
      letter-spacing: -0.03em;
    }}
    .diagram {{
      width: 100%;
      min-width: 920px;
      height: auto;
    }}
    .diagram.compact {{
      min-width: 760px;
    }}
    .diagram text {{
      fill: var(--ink);
      font-size: 14px;
      font-weight: 700;
    }}
    .diagram line {{
      stroke: #2d3a45;
      stroke-width: 2.2;
    }}
    .scroll {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eaf0f4;
      color: #344054;
      font-size: 12px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    code, pre {{
      background: var(--code);
      border-radius: 8px;
    }}
    code {{
      padding: 2px 6px;
    }}
    pre {{
      overflow-x: auto;
      padding: 14px 16px;
      border: 1px solid var(--line);
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .links a {{
      color: var(--accent);
      background: white;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      text-decoration: none;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: rgb(255 255 255 / 80%);
      padding: 12px 14px;
      margin: 16px 0;
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


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _code(text: str) -> str:
    return f"<code>{escape(text)}</code>"


def _profile_summary(profile_name: str) -> tuple[list[list[str]], dict[str, Any]]:
    profile = hw_isp_profile(profile_name)
    config = profile.config
    timing = config.sensor_timing
    control = config.control_path
    transport = config.transport
    rows = [
        ["Profile", _code(profile.name), escape(profile.confidence)],
        ["Sensor timing", "fps / line / active lines", f"{timing.fps:.1f} fps, {timing.line_time_us:.3f} us, {timing.active_lines} lines"],
        ["Exposure limits", "min / max fraction", f"{control.min_exposure_time_us:.1f} us / {control.max_exposure_fraction:.2f} frame"],
        ["3A delay", "AE / AWB", f"{control.ae_apply_delay_frames} / {control.awb_apply_delay_frames} frames"],
        ["Stats", "grid / metering", f"{tuple(control.stats_grid)}, {control.ae_metering}"],
        ["Transport", "queue / buffers", f"{transport.request_queue_depth} / {transport.max_buffers}"],
    ]
    return rows, profile.to_dict()


def _stage_rows(profile_name: str) -> list[list[str]]:
    profile = hw_isp_profile(profile_name)
    rows = []
    for stage in profile.config.stages:
        rows.append(
            [
                _code(stage.name),
                escape(stage.domain),
                escape(stage.buffering),
                str(stage.window_lines),
                f"{stage.stage_latency_cycles:.1f}",
                f"{stage.clock_mhz:.1f}",
                f"{stage.pixels_per_cycle:.2f}",
                "yes" if stage.enabled else "no",
            ]
        )
    return rows


def render(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline = _read_json(output_dir / "timeline_summary.json").get("aggregate", {})
    three_a = _read_json(output_dir / "three_a_summary.json").get("aggregate", {})
    profile_rows, profile_payload = _profile_summary(profile)
    generated_at = datetime.now().isoformat(timespec="seconds")
    commit = _git_commit()
    verdicts = three_a.get("validation_verdicts", {})
    verdict_status = "PASS" if verdicts and all(verdicts.values()) else "CHECK"

    html_path = output_dir / "hwisp_technical_report.html"
    summary_path = output_dir / "hwisp_technical_summary.json"

    body = f"""
<h1>HW ISP Technical Report</h1>
<p class="lead">Detailed technical documentation for the pyisetcam HW ISP simulation layer. This report explains the model, data flow, parameterization, timing equations, delayed AE/AWB behavior, and verification hooks.</p>

<div class="cards">
  <div class="card"><span class="label">Git Commit</span><span class="value">{escape(commit)}</span></div>
  <div class="card"><span class="label">Active Profile</span><span class="value">{escape(profile)}</span></div>
  <div class="card"><span class="label">E2E Mean</span><span class="value">{float(timeline.get('e2e_latency_mean_us', 0.0)) / 1000.0:.3f} ms</span></div>
  <div class="card"><span class="label">3A Verdict</span><span class="value">{escape(verdict_status)}</span></div>
</div>

<div class="callout"><strong>Boundary:</strong> this is a system timing/control simulator, not a vendor fixed-point ISP replacement. Pixel values still come from the existing pyisetcam camera pipeline. HW ISP metadata is attached on top.</div>

<h2>1. System Architecture</h2>
<div class="panel">{_architecture_svg()}</div>

<h2>2. Public Surface</h2>
{_table(
    ["API", "Purpose", "Output"],
    [
        [_code("hw_isp_config(**overrides)"), "Create an inline simulator config.", _code("HWIspConfig")],
        [_code("hw_isp_config_from_profile(name, **overrides)"), "Load a named parameter DB profile and override nested fields.", _code("HWIspConfig")],
        [_code("hw_isp_simulate_frame(camera, scene, config)"), "Run one camera frame and attach HW ISP metadata.", _code("HWIspFrameResult")],
        [_code("hw_isp_simulate_sequence(camera, scenes, config)"), "Run frame sequence with queue history and delayed controls.", _code("HWIspSequenceResult")],
        [_code("hw_isp_timeline_table(result)"), "Flatten frame/stage timing into report rows.", _code("list[dict]")],
        [_code("hw_isp_export_json(result, path)"), "Export stable machine-readable timing and 3A metadata.", _code("Path")],
    ],
)}

<h2>3. Parameter DB</h2>
<p>Profiles are normalized JSON seed databases under <code>src/pyisetcam/data/hwisp/</code>. Product-specific profiles can be loaded with <code>PYISETCAM_HWISP_DB</code>.</p>
{_table(["Item", "Field", "Value"], profile_rows)}

<h3>Available Profiles</h3>
{_table(
    ["Profile", "Status"],
    [[_code(name), "active" if name == profile else "available"] for name in hw_isp_profile_names()],
)}

<h2>4. Sensor Timing Model</h2>
<p>The sensor model is deterministic and uses microsecond timestamps. Rolling shutter row timing is derived from configured hidden lines and line time.</p>
<pre>{escape("""frame_interval_us = 1e6 / fps
t_request       = frame_id * frame_interval_us + deterministic_jitter
t_exposure_mid  = t_exposure_start + exposure_time_us / 2
t_readout_start = t_exposure_start + exposure_time_us
t_row_start(r)  = t_readout_start + (hidden_lines_top + r) * line_time_us
t_readout_end   = t_readout_start
                + (hidden_top + active_lines + hidden_bottom) * line_time_us
t_stats_ready   = t_readout_end""")}</pre>

<h2>5. ISP Stage Timing Model</h2>
<p>Stages are modeled as stream, line-buffer, or frame-buffer blocks. Stream and line-buffer stages can begin before full-frame readout ends. Frame-buffer stages wait for full previous-stage completion.</p>
{_table(
    ["Stage", "Domain", "Buffering", "Window lines", "Cycles", "Clock MHz", "Pixels/cycle", "Enabled"],
    _stage_rows(profile),
)}
<pre>{escape("""cycle_latency_us = stage_latency_cycles / clock_mhz
cycle_latency_us *= stage.latency_factor * global_latency_factor

line_delay_us = max(window_lines - 1, 0) * line_time_us
line_delay_us *= stage.latency_factor * global_latency_factor

frame_stage_throughput_us = pixel_count / (clock_mhz * pixels_per_cycle)""")}</pre>

<h2>6. Transport, Queue, And App Visibility</h2>
<p>The transport model separates frame request, ISP completion, DMA completion, and app-visible timing. Queue stalls are recorded when request depth or output buffer limits would be exceeded.</p>
<pre>{escape("""depth = min(request_queue_depth, max_buffers)
if requested_time < app_visible_time_of_frame[-depth]:
    queue_stall_us = app_visible_time_of_frame[-depth] - requested_time

t_dma_done    = t_isp_done + dma_submit_us + dma_complete_us
t_app_visible = t_dma_done + app_processing_us""")}</pre>

<h2>7. AE/AWB Feedback Model</h2>
<div class="panel">{_sequence_svg()}</div>
<p>When <code>apply_to_image=False</code>, controls are recorded as metadata only. When <code>apply_to_image=True</code>, delayed controls are applied to cloned sensor/IP objects before computing the frame.</p>
{_table(
    ["Area", "Current model"],
    [
        ["AE stats", "8x8 H3A-like grid, center-weighted luma, clipped fraction, EV error."],
        ["AE control", "Exposure/gain product moves toward target luma with max EV step and highlight limiting."],
        ["AWB stats", "Valid-luma ROI from sensor-space RGB, excluding dark and clipped pixels."],
        ["AWB control", "Gray-world RGB gain estimate, normalized to green, clamped by WB gain limits."],
        ["Delay", "Stats from frame N apply to frame N + ae/awb_apply_delay_frames."],
    ],
)}

<h2>8. Metadata Contract</h2>
<p>Each simulated frame stores HW ISP metadata under both <code>camera.metadata[\"hw_isp\"]</code> and <code>ip.metadata[\"hw_isp\"]</code>.</p>
<pre>{escape("""{
  "timeline": {
    "frame_id": int,
    "timestamps_us": {...},
    "stages": [...]
  },
  "controls_applied": {
    "warmup": bool,
    "applied_controls": {...},
    "produced_stats": {"ae": {...}, "awb": {...}},
    "requested_controls": {"ae": {...}, "awb": {...}}
  },
  "config": {...}
}""")}</pre>

<h2>9. Verification Hooks</h2>
{_table(
    ["Verification", "Evidence"],
    [
        ["Timeline monotonicity", "Unit tests check sensor, readout, ISP, DMA, and app-visible order."],
        ["Rolling shutter", "Unit tests verify row_start_us formula."],
        ["Latency factors", "Unit tests verify global and per-stage latency scaling."],
        ["Queue model", "Unit tests force high-FPS queue stall."],
        ["AE/AWB delay", "Unit tests verify source frame maps to apply frame."],
        ["3A convergence", "Report scenario checks AE settle, AWB settle, clamp compliance, warmup mapping, clip reduction."],
        ["Profile DB", "Unit tests load built-in profiles and collected libcamera-style profile JSON."],
    ],
)}

<h3>Latest Report Metrics</h3>
{_table(
    ["Metric", "Value"],
    [
        ["Mean E2E latency", f"{float(timeline.get('e2e_latency_mean_us', 0.0)) / 1000.0:.3f} ms"],
        ["Max E2E latency", f"{float(timeline.get('e2e_latency_max_us', 0.0)) / 1000.0:.3f} ms"],
        ["Queue stall total", f"{float(timeline.get('queue_stall_total_us', 0.0)) / 1000.0:.3f} ms"],
        ["AE settle frame", str(three_a.get("ae_settle_frame", "n/a"))],
        ["AWB settle frame", str(three_a.get("awb_settle_frame", "n/a"))],
        ["AWB final imbalance", str(three_a.get("awb_final_rgb_imbalance", "n/a"))],
    ],
)}

<h2>10. Known Gaps And Sign-off Requirements</h2>
<div class="panel">
  <ul>
    <li>Public seed DB values are not silicon sign-off data.</li>
    <li>Block cycle latency, DMA timing, and queue behavior should be replaced by BSP documentation, kernel traces, hardware counters, or bench measurements.</li>
    <li>AF is intentionally out of scope for this phase.</li>
    <li>Fixed-point vendor ISP numerical matching is separate from this timing/control simulation layer.</li>
  </ul>
</div>

<div class="links">
  <a href="implementation_verification_report.html">Implementation report</a>
  <a href="implementation_verification_report_integrated.html">Integrated implementation report</a>
  <a href="index.html">Timeline dashboard</a>
  <a href="frame_details.html">Frame details</a>
  <a href="hwisp_technical_summary.json">Technical summary JSON</a>
</div>
<p class="lead">Generated at {escape(generated_at)}.</p>
"""

    html_path.write_text(_html_page("HW ISP Technical Report", body), encoding="utf-8")
    summary = {
        "generated_at": generated_at,
        "git_commit": commit,
        "profile": profile,
        "profile_payload": profile_payload,
        "timeline_aggregate": timeline,
        "three_a_aggregate": three_a,
        "outputs": {"html": str(html_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"html": html_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    args = parser.parse_args()
    outputs = render(args.output_dir, profile=args.profile)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
