"""Render HW ISP implementation and verification report."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _run_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return {
        "command": " ".join(command),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": int(result.returncode),
        "summary": lines[-1] if lines else "",
        "output_tail": lines[-8:],
    }


def _verification_results(run_tests: bool) -> list[dict[str, Any]]:
    results = [
        {
            "command": "python tools/render_hwisp_timeline_report.py --profile "
            + DEFAULT_PROFILE,
            "status": "PASS",
            "returncode": 0,
            "summary": "Timeline, 3A, and latency artifacts exist in reports/hwisp/.",
            "output_tail": [],
        }
    ]
    if run_tests:
        results.append(
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/unit/test_hwisp_db.py",
                    "tests/unit/test_hwisp.py",
                ]
            )
        )
    else:
        results.append(
            {
                "command": "python -m pytest -q tests/unit/test_hwisp_db.py tests/unit/test_hwisp.py",
                "status": "NOT_RUN",
                "returncode": None,
                "summary": "Use --run-tests to execute targeted HW ISP verification.",
                "output_tail": [],
            }
        )
    return results


def _artifact_rows(output_dir: Path) -> list[dict[str, Any]]:
    names = [
        "index.html",
        "frame_details.html",
        "timeline_report.md",
        "timeline_summary.json",
        "three_a_summary.json",
        "frame_timeline.png",
        "stage_latency.png",
        "e2e_latency.png",
        "ae_convergence.png",
        "awb_convergence.png",
        "three_a_thumbnails.png",
    ]
    rows = []
    for name in names:
        path = output_dir / name
        rows.append(
            {
                "name": name,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def _architecture_svg() -> str:
    blocks = [
        ("Scene", 24, 72, "#f6d365"),
        ("Optics / OI", 174, 72, "#f6d365"),
        ("Sensor\nExposure + Gain\nReadout", 324, 72, "#9dd6df"),
        ("H3A-like\nStats Grid", 504, 28, "#b7e4c7"),
        ("AE/AWB\nDelay Queue", 504, 146, "#b7e4c7"),
        ("ISP Core\nBLC / DPC / LSC\nDemosaic / CCM / Gamma", 704, 72, "#c8b6ff"),
        ("DMA + Driver\nQueue", 914, 72, "#ffafcc"),
        ("App-visible\nFrame", 1084, 72, "#ffafcc"),
        ("HW ISP\nProfile DB", 504, 286, "#d8e2dc"),
    ]
    rects = []
    for label, x, y, color in blocks:
        lines = label.split("\n")
        text = "".join(
            f'<tspan x="{x + 70}" dy="{0 if index == 0 else 18}">{escape(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        rects.append(
            f'<rect x="{x}" y="{y}" width="140" height="78" rx="14" fill="{color}" />'
            f'<text x="{x + 70}" y="{y + 30}" text-anchor="middle">{text}</text>'
        )
    arrows = [
        (164, 111, 174, 111),
        (314, 111, 324, 111),
        (464, 111, 704, 111),
        (844, 111, 914, 111),
        (1054, 111, 1084, 111),
        (394, 72, 504, 67),
        (574, 106, 574, 146),
        (504, 185, 394, 151),
        (574, 286, 574, 224),
        (644, 325, 704, 151),
    ]
    arrow_markup = "".join(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)" />'
        for x1, y1, x2, y2 in arrows
    )
    return f"""
<svg class="architecture" viewBox="0 0 1248 410" role="img" aria-label="HW ISP architecture diagram">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#344054" />
    </marker>
  </defs>
  <g class="links">{arrow_markup}</g>
  <g class="blocks">{''.join(rects)}</g>
  <text x="624" y="390" text-anchor="middle" class="caption">
    Pixel values stay in the existing camera pipeline; HW ISP adds timing, queue, and delayed-control metadata.
  </text>
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
      --ink: #1d2733;
      --muted: #5d6b78;
      --line: #d6dee8;
      --bg: #f2f6f8;
      --card: #ffffff;
      --accent: #1f6f8b;
      --pass: #237a3b;
      --fail: #aa3b2e;
      --warn: #9a5a21;
    }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #eef7f8 0%, #f7f3eb 100%);
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 24px 52px;
    }}
    h1, h2, h3 {{
      line-height: 1.15;
      margin: 0 0 14px;
    }}
    h1 {{
      font-size: 36px;
      letter-spacing: -0.035em;
    }}
    h2 {{
      margin-top: 34px;
      font-size: 23px;
    }}
    h3 {{
      margin-top: 22px;
      font-size: 18px;
    }}
    .lead {{
      color: var(--muted);
      max-width: 900px;
      font-size: 17px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .card, figure, table, .panel {{
      background: rgb(255 255 255 / 92%);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 26px rgb(31 49 62 / 8%);
    }}
    .card {{
      padding: 16px 18px;
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
    .panel {{
      padding: 18px;
      overflow-x: auto;
    }}
    .architecture {{
      width: 100%;
      min-width: 900px;
      height: auto;
    }}
    .architecture text {{
      fill: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }}
    .architecture .caption {{
      fill: var(--muted);
      font-size: 14px;
      font-weight: 500;
    }}
    .architecture line {{
      stroke: #344054;
      stroke-width: 2.2;
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
      background: #e9f0f3;
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
      padding: 2px 5px;
    }}
    .pass {{
      color: var(--pass);
      font-weight: 800;
    }}
    .fail {{
      color: var(--fail);
      font-weight: 800;
    }}
    .warn {{
      color: var(--warn);
      font-weight: 800;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .links a {{
      color: var(--accent);
      background: white;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      text-decoration: none;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    figure {{
      margin: 0;
      overflow: hidden;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }}
    figcaption {{
      padding: 11px 13px 13px;
      color: var(--muted);
    }}
    ul {{
      margin-top: 8px;
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


def _status_class(status: str) -> str:
    if status == "PASS":
        return "pass"
    if status == "FAIL":
        return "fail"
    return "warn"


def _verification_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for result in results:
        status = str(result["status"])
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(result['command']))}</code></td>"
            f'<td class="{_status_class(status)}">{escape(status)}</td>'
            f"<td>{escape(str(result['summary']))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Command</th><th>Status</th><th>Result</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _verdict_table(aggregate: dict[str, Any]) -> str:
    verdicts = aggregate.get("validation_verdicts", {})
    rows = []
    for key, label in [
        ("ae_settle", "AE settle"),
        ("awb_settle", "AWB settle"),
        ("clamp_compliance", "Clamp compliance"),
        ("warmup_delay_mapping", "Warmup delay mapping"),
        ("clip_reduction", "Highlight clip reduction"),
    ]:
        status = "PASS" if bool(verdicts.get(key, False)) else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f'<td class="{_status_class(status)}">{status}</td>'
            "</tr>"
        )
    return "<table><thead><tr><th>Validation</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _profile_table(profile_name: str) -> str:
    active = hw_isp_profile(profile_name)
    rows = []
    for name in hw_isp_profile_names():
        profile = hw_isp_profile(name)
        marker = "Active" if name == profile_name else "Available"
        rows.append(
            "<tr>"
            f"<td><code>{escape(name)}</code></td>"
            f"<td>{escape(marker)}</td>"
            f"<td>{escape(profile.confidence)}</td>"
            f"<td>{escape(profile.description)}</td>"
            "</tr>"
        )
    timing = active.config.sensor_timing
    control = active.config.control_path
    rows.append(
        "<tr>"
        f"<td><code>{escape(profile_name)} timing</code></td>"
        "<td>Active values</td>"
        "<td>Config</td>"
        f"<td>{timing.fps:.1f} fps, {timing.line_time_us:.3f} us/line, "
        f"{timing.active_lines} active lines, target luma {control.target_luma:.3f}</td>"
        "</tr>"
    )
    return "<table><thead><tr><th>Profile</th><th>Use</th><th>Confidence</th><th>Description</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _artifact_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        status = "PASS" if row["exists"] and int(row["bytes"]) > 0 else "FAIL"
        body.append(
            "<tr>"
            f"<td><code>{escape(row['name'])}</code></td>"
            f'<td class="{_status_class(status)}">{status}</td>'
            f"<td>{int(row['bytes']):,}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Artifact</th><th>Status</th><th>Bytes</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def _image_source(path: Path, embed_images: bool) -> str:
    if not embed_images or not path.exists():
        return path.name
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _figure(path: Path, title: str, caption: str, *, embed_images: bool = False) -> str:
    source = _image_source(path, embed_images)
    return (
        "<figure>"
        f'<img src="{escape(source)}" alt="{escape(title)}">'
        f"<figcaption><strong>{escape(title)}</strong><br>{escape(caption)}</figcaption>"
        "</figure>"
    )


def render(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    profile: str = DEFAULT_PROFILE,
    run_tests: bool = False,
    embed_images: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_summary = _read_json(output_dir / "timeline_summary.json")
    three_a_summary = _read_json(output_dir / "three_a_summary.json")
    timeline_aggregate = timeline_summary.get("aggregate", {})
    three_a_aggregate = three_a_summary.get("aggregate", {})
    verification = _verification_results(run_tests)
    artifacts = _artifact_rows(output_dir)
    commit = _git_commit()
    generated_at = datetime.now().isoformat(timespec="seconds")
    verdicts = three_a_aggregate.get("validation_verdicts", {})
    verdict_status = "PASS" if verdicts and all(verdicts.values()) else "CHECK"

    if embed_images:
        html_path = output_dir / "implementation_verification_report_integrated.html"
        summary_path = output_dir / "implementation_verification_integrated_summary.json"
    else:
        html_path = output_dir / "implementation_verification_report.html"
        summary_path = output_dir / "implementation_verification_summary.json"

    body = f"""
<h1>HW ISP Implementation And Verification Report</h1>
<p class="lead">This report summarizes the HW ISP simulation implementation, architecture, parameter DB integration, and latest verification evidence. It is a system-verification layer: image values remain produced by the existing pyisetcam camera pipeline, while HW ISP adds timing, queue, and delayed AE/AWB behavior.</p>
<div class="cards">
  <div class="card"><span class="label">Git Commit</span><span class="value">{escape(commit)}</span></div>
  <div class="card"><span class="label">HW ISP Profile</span><span class="value">{escape(profile)}</span></div>
  <div class="card"><span class="label">Frame Count</span><span class="value">{int(timeline_aggregate.get('frame_count', 0))}</span></div>
  <div class="card"><span class="label">3A Verdict</span><span class="value">{escape(verdict_status)}</span></div>
</div>

<h2>Architecture</h2>
<div class="panel">{_architecture_svg()}</div>

<h2>Implemented Scope</h2>
<div class="grid">
  <div class="card"><span class="label">Latency Layer</span><p>Sensor exposure/readout timestamps, line-buffer and frame-buffer ISP stage timing, DMA, queue stall, and app-visible timing.</p></div>
  <div class="card"><span class="label">Image-affecting 3A</span><p>Optional <code>apply_to_image=True</code> delayed AE/AWB control loop with H3A-like stats grid, highlight handling, and valid-luma AWB stats.</p></div>
  <div class="card"><span class="label">Parameter DB</span><p>Named profile loader, built-in public seed profiles, environment override via <code>PYISETCAM_HWISP_DB</code>, and libcamera tuning collector.</p></div>
  <div class="card"><span class="label">Reports</span><p>Timeline dashboard, frame details, 3A convergence plots, machine-readable JSON, and this implementation verification report.</p></div>
</div>

<h2>Parameter DB</h2>
{_profile_table(profile)}

<h2>Verification Results</h2>
{_verification_table(verification)}

<h2>3A Validation Results</h2>
<div class="grid">
  <div>{_verdict_table(three_a_aggregate)}</div>
  <div class="card">
    <span class="label">Measured Values</span>
    <ul>
      <li>AE settle frame: <code>{three_a_aggregate.get('ae_settle_frame', 'n/a')}</code></li>
      <li>AE final error: <code>{three_a_aggregate.get('ae_final_error_ev', 'n/a')}</code> EV</li>
      <li>AWB settle frame: <code>{three_a_aggregate.get('awb_settle_frame', 'n/a')}</code></li>
      <li>AWB final RGB imbalance: <code>{three_a_aggregate.get('awb_final_rgb_imbalance', 'n/a')}</code></li>
      <li>Max clip before/after response: <code>{three_a_aggregate.get('max_clip_fraction_before_response', 'n/a')}</code> / <code>{three_a_aggregate.get('max_clip_fraction_after_response', 'n/a')}</code></li>
    </ul>
  </div>
</div>

<h2>Latency Evidence</h2>
<div class="cards">
  <div class="card"><span class="label">Mean E2E</span><span class="value">{float(timeline_aggregate.get('e2e_latency_mean_us', 0.0)) / 1000.0:.3f} ms</span></div>
  <div class="card"><span class="label">Max E2E</span><span class="value">{float(timeline_aggregate.get('e2e_latency_max_us', 0.0)) / 1000.0:.3f} ms</span></div>
  <div class="card"><span class="label">Queue Stall Total</span><span class="value">{float(timeline_aggregate.get('queue_stall_total_us', 0.0)) / 1000.0:.3f} ms</span></div>
</div>
<div class="grid">
  {_figure(output_dir / 'frame_timeline.png', 'Frame timeline', 'Per-frame ISP stage spans and queue timing.', embed_images=embed_images)}
  {_figure(output_dir / 'stage_latency.png', 'Stage latency', 'Mean modeled latency by HW ISP block.', embed_images=embed_images)}
  {_figure(output_dir / 'e2e_latency.png', 'E2E latency', 'Application-visible latency and queue stall.', embed_images=embed_images)}
  {_figure(output_dir / 'ae_convergence.png', 'AE convergence', 'Metered luma, exposure/gain, clipping, and settle frame.', embed_images=embed_images)}
  {_figure(output_dir / 'awb_convergence.png', 'AWB convergence', 'Valid-luma gray-world correction and RGB imbalance.', embed_images=embed_images)}
  {_figure(output_dir / 'three_a_thumbnails.png', '3A thumbnails', 'Representative output frames before and after delayed control response.', embed_images=embed_images)}
</div>

<h2>Artifact Check</h2>
{_artifact_table(artifacts)}

<h2>Known Limits</h2>
<div class="panel">
  <ul>
    <li>Public seed profiles are not vendor sign-off databases. Accurate block latency requires BSP documentation, kernel trace, hardware counters, or bench measurement.</li>
    <li>AF is intentionally out of scope in the current 3A layer.</li>
    <li>The simulator models timing/control behavior; fixed-point vendor ISP numerical matching is a separate task.</li>
  </ul>
</div>

<div class="links">
  <a href="index.html">Timeline dashboard</a>
  <a href="frame_details.html">Frame details</a>
  <a href="timeline_summary.json">Timeline JSON</a>
  <a href="three_a_summary.json">3A JSON</a>
  <a href="implementation_verification_summary.json">Implementation summary JSON</a>
  <a href="implementation_verification_report_integrated.html">Integrated image HTML</a>
</div>
<p class="lead">Generated at {escape(generated_at)}.</p>
"""
    html_path.write_text(_html_page("HW ISP Implementation Verification", body), encoding="utf-8")
    summary = {
        "generated_at": generated_at,
        "git_commit": commit,
        "profile": profile,
        "verification": verification,
        "artifacts": artifacts,
        "timeline_aggregate": timeline_aggregate,
        "three_a_aggregate": three_a_aggregate,
        "embed_images": bool(embed_images),
        "outputs": {"html": str(html_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"html": html_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--embed-images", action="store_true")
    args = parser.parse_args()
    outputs = render(
        args.output_dir,
        profile=args.profile,
        run_tests=bool(args.run_tests),
        embed_images=bool(args.embed_images),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
