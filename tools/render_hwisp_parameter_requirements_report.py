"""Render required HW ISP parameter DB report."""

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


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _value(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            current = current[int(part)] if part.isdigit() and int(part) < len(current) else None
        else:
            return None
    return current


def _status_for_value(value: Any, source_type: str) -> str:
    if value is None:
        return "Missing"
    if source_type in {"vendor_bsp", "measurement"}:
        return "Needs real data"
    if source_type == "public_seed":
        return "Seed only"
    if source_type == "synthetic_seed":
        return "Synthetic seed"
    return "Available"


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #192733;
      --muted: #627282;
      --line: #d5dee7;
      --card: #ffffff;
      --bg: #f4f7f8;
      --required: #ab3e31;
      --recommended: #a7651c;
      --optional: #2b6f8a;
      --ok: #267545;
      --seed: #8a641d;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 12% 4%, rgb(43 111 138 / 12%), transparent 28rem),
        linear-gradient(135deg, #f8f3e7 0%, #eef7f8 100%);
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1260px;
      margin: 0 auto;
      padding: 34px 24px 54px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 37px;
      letter-spacing: -0.04em;
      line-height: 1.08;
    }}
    h2 {{
      margin: 34px 0 14px;
      font-size: 23px;
    }}
    .lead {{
      max-width: 940px;
      color: var(--muted);
      font-size: 17px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .card, table, .panel {{
      background: rgb(255 255 255 / 94%);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 28px rgb(25 39 51 / 7%);
    }}
    .card, .panel {{
      padding: 16px 18px;
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
      font-size: 26px;
      font-weight: 780;
      letter-spacing: -0.03em;
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
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      min-width: 120px;
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
    code {{
      background: #edf3f5;
      border-radius: 6px;
      padding: 2px 6px;
    }}
    .required {{
      color: var(--required);
      font-weight: 800;
    }}
    .recommended {{
      color: var(--recommended);
      font-weight: 800;
    }}
    .optional {{
      color: var(--optional);
      font-weight: 800;
    }}
    .ok {{
      color: var(--ok);
      font-weight: 800;
    }}
    .seed {{
      color: var(--seed);
      font-weight: 800;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .links a {{
      color: var(--optional);
      background: white;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      text-decoration: none;
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


def _code(value: Any) -> str:
    return f"<code>{escape(str(value))}</code>"


def _priority_label(priority: str) -> str:
    klass = {
        "Required": "required",
        "Recommended": "recommended",
        "Optional": "optional",
    }.get(priority, "optional")
    return f'<span class="{klass}">{escape(priority)}</span>'


def _status_label(status: str) -> str:
    klass = "ok" if status == "Available" else "seed" if "seed" in status.lower() else "required"
    return f'<span class="{klass}">{escape(status)}</span>'


PARAMETERS: list[dict[str, str]] = [
    {
        "group": "Sensor mode",
        "name": "frame rate",
        "field": "config.sensor_timing.fps",
        "priority": "Required",
        "source": "sensor mode table, camera HAL, libcamera mode list",
        "real_source": "vendor BSP or runtime query",
        "use": "frame interval and request cadence",
        "source_type": "public_seed",
    },
    {
        "group": "Sensor mode",
        "name": "line time",
        "field": "config.sensor_timing.line_time_us",
        "priority": "Required",
        "source": "sensor datasheet, mode table, pixel clock / line length",
        "real_source": "sensor datasheet or measurement",
        "use": "rolling shutter and readout timing",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Sensor mode",
        "name": "active lines",
        "field": "config.sensor_timing.active_lines",
        "priority": "Required",
        "source": "selected sensor resolution",
        "real_source": "driver mode table",
        "use": "readout end and frame duration",
        "source_type": "public_seed",
    },
    {
        "group": "Sensor mode",
        "name": "hidden top/bottom lines",
        "field": "config.sensor_timing.hidden_lines_top",
        "priority": "Required",
        "source": "sensor mode table or empirical timing",
        "real_source": "BSP or sensor driver",
        "use": "row start and readout end",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Exposure/gain",
        "name": "default exposure",
        "field": "config.sensor_timing.exposure_time_us",
        "priority": "Required",
        "source": "initial mode setting, AE default",
        "real_source": "HAL or tuning file",
        "use": "exposure start/mid/readout timing",
        "source_type": "public_seed",
    },
    {
        "group": "Exposure/gain",
        "name": "min exposure",
        "field": "config.control_path.min_exposure_time_us",
        "priority": "Required",
        "source": "sensor register limits",
        "real_source": "sensor datasheet or V4L2 controls",
        "use": "AE clamp behavior",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Exposure/gain",
        "name": "max exposure fraction",
        "field": "config.control_path.max_exposure_fraction",
        "priority": "Required",
        "source": "HAL policy or camera tuning",
        "real_source": "vendor tuning or product policy",
        "use": "AE clamp behavior",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Exposure/gain",
        "name": "analog gain range",
        "field": "config.control_path.max_analog_gain",
        "priority": "Required",
        "source": "sensor gain register limits",
        "real_source": "sensor datasheet or V4L2 controls",
        "use": "AE exposure/gain allocation",
        "source_type": "synthetic_seed",
    },
    {
        "group": "ISP stages",
        "name": "stage list/order",
        "field": "config.stages",
        "priority": "Required",
        "source": "SoC ISP block diagram or driver pipeline",
        "real_source": "vendor ISP manual/BSP",
        "use": "stage timeline and reporting",
        "source_type": "synthetic_seed",
    },
    {
        "group": "ISP stages",
        "name": "stage buffering mode",
        "field": "config.stages.*.buffering",
        "priority": "Required",
        "source": "block architecture",
        "real_source": "vendor ISP manual",
        "use": "stream, line-buffer, frame-buffer scheduling",
        "source_type": "synthetic_seed",
    },
    {
        "group": "ISP stages",
        "name": "window lines",
        "field": "config.stages.*.window_lines",
        "priority": "Required",
        "source": "filter kernel/window size",
        "real_source": "vendor ISP manual",
        "use": "line-buffer delay",
        "source_type": "synthetic_seed",
    },
    {
        "group": "ISP stages",
        "name": "stage latency cycles",
        "field": "config.stages.*.stage_latency_cycles",
        "priority": "Required",
        "source": "hardware manual or profiling",
        "real_source": "vendor data or hardware measurement",
        "use": "per-block latency",
        "source_type": "synthetic_seed",
    },
    {
        "group": "ISP stages",
        "name": "clock MHz",
        "field": "config.stages.*.clock_mhz",
        "priority": "Required",
        "source": "SoC clock tree",
        "real_source": "device tree, debugfs, BSP",
        "use": "cycle latency conversion",
        "source_type": "synthetic_seed",
    },
    {
        "group": "ISP stages",
        "name": "pixels per cycle",
        "field": "config.stages.*.pixels_per_cycle",
        "priority": "Recommended",
        "source": "ISP throughput spec",
        "real_source": "vendor ISP manual or measured throughput",
        "use": "frame-buffer stage throughput",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Transport/queue",
        "name": "request queue depth",
        "field": "config.transport.request_queue_depth",
        "priority": "Required",
        "source": "driver/HAL request queue",
        "real_source": "driver source or runtime trace",
        "use": "queue stall model",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Transport/queue",
        "name": "max output buffers",
        "field": "config.transport.max_buffers",
        "priority": "Required",
        "source": "driver/HAL buffer pool",
        "real_source": "driver source or runtime trace",
        "use": "queue stall model",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Transport/queue",
        "name": "DMA submit/complete delay",
        "field": "config.transport.dma_complete_us",
        "priority": "Required",
        "source": "kernel trace, HW counter, driver profiling",
        "real_source": "measurement",
        "use": "app-visible latency",
        "source_type": "synthetic_seed",
    },
    {
        "group": "3A stats/control",
        "name": "AE/AWB apply delay",
        "field": "config.control_path.ae_apply_delay_frames",
        "priority": "Required",
        "source": "3A pipeline behavior",
        "real_source": "HAL/libcamera logs or measurement",
        "use": "delayed feedback simulation",
        "source_type": "synthetic_seed",
    },
    {
        "group": "3A stats/control",
        "name": "stats grid",
        "field": "config.control_path.stats_grid",
        "priority": "Required",
        "source": "H3A/statistics block config",
        "real_source": "vendor ISP manual or tuning",
        "use": "AE/AWB stats model",
        "source_type": "synthetic_seed",
    },
    {
        "group": "3A stats/control",
        "name": "AE target luma",
        "field": "config.control_path.target_luma",
        "priority": "Required",
        "source": "AGC tuning",
        "real_source": "public libcamera seed or vendor tuning",
        "use": "AE convergence",
        "source_type": "public_seed",
    },
    {
        "group": "3A stats/control",
        "name": "highlight clip threshold",
        "field": "config.control_path.ae_highlight_clip",
        "priority": "Recommended",
        "source": "AE tuning policy",
        "real_source": "vendor tuning or product policy",
        "use": "highlight-preserving AE response",
        "source_type": "synthetic_seed",
    },
    {
        "group": "3A stats/control",
        "name": "AWB valid luma range",
        "field": "config.control_path.awb_min_luma",
        "priority": "Recommended",
        "source": "AWB stats ROI policy",
        "real_source": "vendor tuning",
        "use": "AWB dark/clip exclusion",
        "source_type": "synthetic_seed",
    },
    {
        "group": "Calibration/tuning",
        "name": "black level",
        "field": "calibration.black_level.value",
        "priority": "Required",
        "source": "sensor/module calibration, tuning file",
        "real_source": "OTP, tuning file, factory calibration",
        "use": "BLC seed and report provenance",
        "source_type": "public_seed",
    },
    {
        "group": "Calibration/tuning",
        "name": "noise model",
        "field": "calibration.noise.reference_slope",
        "priority": "Recommended",
        "source": "tuning file or sensor characterization",
        "real_source": "lab characterization/vendor tuning",
        "use": "future noise/denoise simulation",
        "source_type": "public_seed",
    },
    {
        "group": "Calibration/tuning",
        "name": "GEQ/lens shading seed",
        "field": "calibration.geq.slope",
        "priority": "Recommended",
        "source": "tuning file",
        "real_source": "factory/tuning data",
        "use": "future green-equalization/LSC simulation",
        "source_type": "public_seed",
    },
    {
        "group": "Calibration/tuning",
        "name": "lux reference",
        "field": "calibration.lux_reference.reference_lux",
        "priority": "Recommended",
        "source": "camera tuning",
        "real_source": "tuning file/lab calibration",
        "use": "AE scale/reference sanity check",
        "source_type": "public_seed",
    },
]


def _parameter_rows(profile_payload: dict[str, Any]) -> list[list[str]]:
    rows = []
    for item in PARAMETERS:
        value = _value(profile_payload, item["field"])
        status = _status_for_value(value, item["source_type"])
        rows.append(
            [
                escape(item["group"]),
                _priority_label(item["priority"]),
                escape(item["name"]),
                _code(item["field"]),
                _code(value) if value is not None else "<em>missing</em>",
                _status_label(status),
                escape(item["real_source"]),
                escape(item["use"]),
            ]
        )
    return rows


def _source_rows() -> list[list[str]]:
    return [
        [
            "Vendor BSP / ISP manual",
            '<span class="required">Required for sign-off</span>',
            "Stage order, buffering, clocks, cycles, queue behavior, DMA timing.",
        ],
        [
            "Sensor datasheet / driver mode table",
            '<span class="required">Required</span>',
            "Line time, blanking, exposure limits, gain limits, active geometry.",
        ],
        [
            "Runtime trace / measurement",
            '<span class="required">Required for latency accuracy</span>',
            "Real request-to-visible latency, queue stall, DMA timing, 3A apply delay.",
        ],
        [
            "Factory calibration / OTP",
            '<span class="recommended">Recommended</span>',
            "Black level, defective pixels, lens shading, module-specific calibration.",
        ],
        [
            "Public libcamera/RPi tuning",
            '<span class="seed">Seed only</span>',
            "Useful starting point for black level, lux reference, noise, GEQ, AGC target.",
        ],
    ]


def render(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_obj = hw_isp_profile(profile)
    profile_payload = profile_obj.to_dict()
    generated_at = datetime.now().isoformat(timespec="seconds")
    commit = _git_commit()
    missing_count = sum(1 for item in PARAMETERS if _value(profile_payload, item["field"]) is None)
    required_count = sum(1 for item in PARAMETERS if item["priority"] == "Required")
    required_seed_count = sum(
        1
        for item in PARAMETERS
        if item["priority"] == "Required" and item["source_type"] != "public_seed"
    )

    html_path = output_dir / "hwisp_parameter_requirements_report.html"
    summary_path = output_dir / "hwisp_parameter_requirements_summary.json"
    body = f"""
<h1>HW ISP Parameter DB Requirements Report</h1>
<p class="lead">This report lists the parameters needed to turn the HW ISP simulator from a public/synthetic seed model into product-grade system verification data. It separates what is already present in the current profile from what must come from vendor BSP, sensor documentation, runtime trace, or factory calibration.</p>

<div class="cards">
  <div class="card"><span class="label">Git Commit</span><span class="value">{escape(commit)}</span></div>
  <div class="card"><span class="label">Active Profile</span><span class="value">{escape(profile)}</span></div>
  <div class="card"><span class="label">Required Items</span><span class="value">{required_count}</span></div>
  <div class="card"><span class="label">Required Real-Data Replacements</span><span class="value">{required_seed_count}</span></div>
  <div class="card"><span class="label">Missing Fields</span><span class="value">{missing_count}</span></div>
</div>

<div class="panel">
  <strong>Conclusion:</strong> the current profile is sufficient for simulator development and report generation, but not for hardware sign-off. The latency-critical fields must be replaced with board-specific BSP or measured values.
</div>

<h2>1. Parameter Source Priority</h2>
{_table(["Source", "Priority", "What It Provides"], _source_rows())}

<h2>2. Required DB Parameter Matrix</h2>
{_table(
    ["Group", "Priority", "Parameter", "Profile field", "Current value", "Current status", "Real source needed", "Simulator use"],
    _parameter_rows(profile_payload),
)}

<h2>3. Minimum Sign-off Checklist</h2>
{_table(
    ["Checklist item", "Pass condition"],
    [
        ["Sensor mode timing", "fps, line time, active/hidden lines match selected hardware mode."],
        ["Exposure/gain limits", "min/max exposure and analog gain verified against sensor controls."],
        ["ISP stage model", "stage order, buffering mode, window lines, clocks, and cycle latency come from vendor docs or profiling."],
        ["Transport timing", "DMA and app-visible timing measured under target driver/HAL configuration."],
        ["3A delay", "AE/AWB stats frame to applied-control frame mapping verified on device logs or capture traces."],
        ["Calibration", "black level, lens shading, noise, and color seeds are tied to module calibration or tuning release."],
        ["Traceability", "every non-seed field has a source file, command, measurement date, or ticket reference."],
    ],
)}

<h2>4. How To Collect More DB Data</h2>
<pre>{escape("""# Public/local libcamera tuning summary
python tools/collect_hwisp_parameter_db.py /usr/share/libcamera/ipa/rpi/vc4 --output-dir configs/hwisp/collected

# Use collected profiles
export PYISETCAM_HWISP_DB=configs/hwisp/collected

# Generate reports with a collected profile
python tools/render_hwisp_timeline_report.py --profile <profile_name>
python tools/render_hwisp_parameter_requirements_report.py --profile <profile_name>""")}</pre>

<h2>5. Available Profiles</h2>
{_table(["Profile", "Status"], [[_code(name), "active" if name == profile else "available"] for name in hw_isp_profile_names()])}

<div class="links">
  <a href="hwisp_technical_report.html">Technical report</a>
  <a href="implementation_verification_report.html">Implementation report</a>
  <a href="index.html">Timeline dashboard</a>
  <a href="hwisp_parameter_requirements_summary.json">Parameter requirements JSON</a>
</div>
<p class="lead">Generated at {escape(generated_at)}.</p>
"""
    html_path.write_text(_html_page("HW ISP Parameter DB Requirements", body), encoding="utf-8")
    summary = {
        "generated_at": generated_at,
        "git_commit": commit,
        "profile": profile,
        "required_count": required_count,
        "required_real_data_replacements": required_seed_count,
        "missing_count": missing_count,
        "parameters": [
            {
                **item,
                "current_value": _value(profile_payload, item["field"]),
                "current_status": _status_for_value(
                    _value(profile_payload, item["field"]),
                    item["source_type"],
                ),
            }
            for item in PARAMETERS
        ],
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
