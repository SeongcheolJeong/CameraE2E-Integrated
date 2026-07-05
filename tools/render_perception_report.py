"""Render an HTML report for the high-level perception metric surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam.perception import (  # noqa: E402
    PerceptionViewingConfig,
    perception_artifact_metrics,
    perception_compare,
    perception_config,
    perception_sharpness_metrics,
    perception_visible_difference_map,
    pixels_per_degree,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "perception"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _reference_image(size: int = 192) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(float)
    x = x / max(size - 1, 1)
    y = y / max(size - 1, 1)
    image = np.zeros((size, size, 3), dtype=float)
    image[..., 0] = 0.12 + 0.78 * x
    image[..., 1] = 0.10 + 0.72 * y
    image[..., 2] = 0.18 + 0.48 * (1.0 - x)

    checker = ((np.floor(x * 24.0) + np.floor(y * 24.0)) % 2.0).astype(float)
    image[12:76, 12:76, :] = checker[12:76, 12:76, None]

    bars = ((np.floor(x * 48.0) % 2.0) > 0).astype(float)
    image[92:142, 20:172, :] = bars[92:142, 20:172, None]

    patches = np.array(
        [
            [0.85, 0.18, 0.14],
            [0.12, 0.55, 0.22],
            [0.10, 0.26, 0.76],
            [0.86, 0.72, 0.15],
            [0.74, 0.22, 0.68],
            [0.12, 0.72, 0.78],
        ],
        dtype=float,
    )
    patch_h = 28
    patch_w = 28
    for index, color in enumerate(patches):
        row = 154
        col = 12 + index * 30
        image[row : row + patch_h, col : col + patch_w, :] = color
    return np.clip(image, 0.0, 1.0)


def _test_image(reference: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(reference, sigma=(1.1, 1.1, 0.0))
    y, x = np.mgrid[0 : reference.shape[0], 0 : reference.shape[1]].astype(float)
    banding = 0.018 * np.sin(2.0 * np.pi * x / 17.0)
    color_cast = np.array([1.04, 0.98, 0.92], dtype=float)
    test = blurred * color_cast.reshape(1, 1, 3)
    test[..., 1] += banding
    return np.clip(test, 0.0, 1.0)


def _save_image(path: Path, image: np.ndarray, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 5.2))
    plt.imshow(np.clip(image, 0.0, 1.0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=150)
    plt.close()


def _save_difference(path: Path, delta_e: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(delta_e, cmap="magma")
    plt.title("Visible Difference Map (S-CIELAB Delta E)")
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Delta E / JND proxy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_sharpness_plot(path: Path, frequency: np.ndarray, mtf_reference: np.ndarray, mtf_test: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(frequency, mtf_reference, label="Reference MTF", linewidth=2.2)
    plt.plot(frequency, mtf_test, label="Test MTF", linewidth=2.2, linestyle="--")
    plt.xlabel("Spatial frequency (cycles/degree)")
    plt.ylabel("MTF")
    plt.title("Perception-Weighted Sharpness Inputs")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _table(rows: list[tuple[str, Any, str]]) -> str:
    body = "\n".join(
        f"<tr><th>{escape(str(name))}</th><td>{escape(_format_value(value))}</td><td>{escape(units)}</td></tr>"
        for name, value, units in rows
    )
    return f"<table><thead><tr><th>Metric</th><th>Value</th><th>Units</th></tr></thead><tbody>{body}</tbody></table>"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #18222d;
      --muted: #617080;
      --line: #d4dde6;
      --card: #ffffff;
      --accent: #235c6f;
      --bg: #f4f1e9;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgb(35 92 111 / 14%), transparent 36rem),
        linear-gradient(135deg, #fbf7ec 0%, #eef7f4 100%);
      color: var(--ink);
      font: 15px/1.58 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 38px 24px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 38px;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 34px 0 14px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }}
    .lead {{
      max-width: 880px;
      color: var(--muted);
      font-size: 17px;
    }}
    .cards, .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .card, .panel, table {{
      background: rgb(255 255 255 / 94%);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 28px rgb(24 34 45 / 7%);
    }}
    .card, .panel {{
      padding: 17px 18px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 760;
    }}
    img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
      margin: 12px 0 20px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      width: 42%;
      background: #f6f8fa;
    }}
    code {{
      background: #eef3f5;
      border-radius: 6px;
      padding: 2px 5px;
    }}
    .note {{
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      background: rgb(35 92 111 / 8%);
      border-radius: 12px;
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>
"""


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_report(output_dir: Path = DEFAULT_OUTPUT_DIR, config: PerceptionViewingConfig | None = None) -> dict[str, Any]:
    cfg = perception_config() if config is None else config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = _reference_image()
    test = _test_image(reference)
    visible = perception_visible_difference_map(reference, test, cfg, method="scielab")
    summary = perception_compare(reference, test, cfg, visible_method="scielab")
    artifacts = perception_artifact_metrics(reference, test, cfg)

    frequency = np.linspace(0.0, 60.0, 64, dtype=float)
    mtf_reference = np.exp(-np.power(frequency / 52.0, 1.45))
    mtf_test = np.exp(-np.power(frequency / 34.0, 1.45))
    sharp_reference = perception_sharpness_metrics(frequency, mtf_reference, cfg)
    sharp_test = perception_sharpness_metrics(frequency, mtf_test, cfg)

    reference_path = output_dir / "perception_reference.png"
    test_path = output_dir / "perception_test.png"
    difference_path = output_dir / "perception_visible_difference_map.png"
    sharpness_path = output_dir / "perception_sharpness_mtf.png"
    html_path = output_dir / "perception_report.html"
    json_path = output_dir / "perception_summary.json"

    _save_image(reference_path, reference, "Reference image")
    _save_image(test_path, test, "Test image")
    _save_difference(difference_path, np.asarray(visible["delta_e"], dtype=float))
    _save_sharpness_plot(sharpness_path, frequency, mtf_reference, mtf_test)

    git_commit = _git_commit()
    generated_at = datetime.now().isoformat(timespec="seconds")
    image_metrics = summary["image_metrics"]
    color_metrics = summary["color_metrics"]
    visible_metrics = summary["visible_difference"]

    rows_config = [
        ("Viewing distance", cfg.viewing_distance_m, "m"),
        ("Pixel pitch", cfg.pixel_pitch_m * 1e6, "um"),
        ("Pixels per degree", pixels_per_degree(cfg), "px/deg"),
        ("Display peak luminance", cfg.display_luminance_cd_m2, "cd/m^2"),
        ("JND Delta E threshold", cfg.jnd_delta_e, "Delta E"),
        ("Visible Delta E threshold", cfg.visible_delta_e, "Delta E"),
    ]
    rows_visible = [
        ("Mean Delta E", visible_metrics["mean_delta_e"], "Delta E"),
        ("95th percentile Delta E", visible_metrics["p95_delta_e"], "Delta E"),
        ("Max Delta E", visible_metrics["max_delta_e"], "Delta E"),
        ("Mean JND", visible_metrics["mean_jnd"], "JND"),
        ("Visible pixel fraction", visible_metrics["visible_fraction"], "fraction"),
    ]
    rows_quality = [
        ("MAE", image_metrics["mae"], "normalized"),
        ("RMSE", image_metrics["rmse"], "normalized"),
        ("PSNR", image_metrics["psnr"], "dB"),
        ("Mean luminance error", image_metrics["mean_luminance_error_cd_m2"], "cd/m^2"),
        ("RMS luminance error", image_metrics["rms_luminance_error_cd_m2"], "cd/m^2"),
        ("Mean color Delta E", color_metrics["mean_delta_e"], "Delta E"),
        ("Reference ISO acutance", sharp_reference["iso_acutance"], "unitless"),
        ("Test ISO acutance", sharp_test["iso_acutance"], "unitless"),
        ("Reference SQRI", sharp_reference["sqri"], "unitless"),
        ("Test SQRI", sharp_test["sqri"], "unitless"),
        ("High-frequency error fraction", artifacts["high_frequency_error_fraction"], "fraction"),
    ]

    body = f"""
<h1>Perception Metric Implementation Report</h1>
<p class="lead">This report demonstrates the new high-level perception API. It is not a new ISP algorithm; it wraps existing numeric, color, SCIELAB, and sharpness helpers into viewing-condition aware evidence.</p>
<div class="cards">
  <div class="card"><div class="label">Git commit</div><div class="value">{escape(git_commit)}</div></div>
  <div class="card"><div class="label">Generated</div><div class="value">{escape(generated_at)}</div></div>
  <div class="card"><div class="label">Mean JND</div><div class="value">{visible_metrics["mean_jnd"]:.3f}</div></div>
  <div class="card"><div class="label">Visible fraction</div><div class="value">{visible_metrics["visible_fraction"]:.3f}</div></div>
</div>

<h2>Architecture</h2>
<div class="panel">
  <p><code>perception.py</code> sits above <code>metrics.py</code>, <code>scielab.py</code>, and camera quality helpers. The API requires explicit viewing assumptions so that color difference, JND maps, sharpness, and artifact proxies are not reported as context-free scalars.</p>
  <p class="note">Important: a single global "perception score" is intentionally not reported. Color, visible difference, sharpness, and artifacts are separate failure modes and should remain inspectable.</p>
</div>

<h2>Viewing Configuration</h2>
{_table(rows_config)}

<h2>Evidence Images</h2>
<div class="figures">
  <figure><img src="{reference_path.name}" alt="Reference image"><figcaption>Reference image with edges, color patches, gradients, and texture.</figcaption></figure>
  <figure><img src="{test_path.name}" alt="Test image"><figcaption>Test image with blur, small color cast, and banding proxy.</figcaption></figure>
  <figure><img src="{difference_path.name}" alt="Visible difference map"><figcaption>S-CIELAB Delta E map. Brighter areas indicate more visible differences.</figcaption></figure>
  <figure><img src="{sharpness_path.name}" alt="MTF plot"><figcaption>Reference/test MTF curves used for perception-weighted sharpness metrics.</figcaption></figure>
</div>

<h2>Visible Difference</h2>
{_table(rows_visible)}

<h2>Quality Summary</h2>
{_table(rows_quality)}

<h2>Implemented Public API</h2>
<div class="panel">
  <p>Primary functions: <code>perception_config</code>, <code>pixels_per_degree</code>, <code>image_to_luminance</code>, <code>perception_image_metrics</code>, <code>perception_color_metrics</code>, <code>perception_visible_difference_map</code>, <code>perception_sharpness_metrics</code>, <code>perception_artifact_metrics</code>, <code>perception_compare</code>, and <code>perception_report</code>.</p>
  <p>MATLAB-style aliases are also exposed at the package root, for example <code>perceptionConfig</code>, <code>pixelsPerDegree</code>, and <code>perceptionVisibleDifferenceMap</code>.</p>
</div>
"""

    summary_payload = {
        "generated_at": generated_at,
        "git_commit": git_commit,
        "config": _json_ready(cfg.__dict__),
        "figures": {
            "reference": str(reference_path),
            "test": str(test_path),
            "visible_difference": str(difference_path),
            "sharpness": str(sharpness_path),
        },
        "image_metrics": _json_ready(image_metrics),
        "visible_difference": _json_ready(visible_metrics),
        "color_metrics": _json_ready(color_metrics),
        "artifact_metrics": _json_ready(artifacts),
        "sharpness": {
            "reference": _json_ready(sharp_reference),
            "test": _json_ready(sharp_test),
        },
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_html_page("Perception Metric Implementation Report", body), encoding="utf-8")
    return {"html": html_path, "summary": json_path, "figures": summary_payload["figures"], "metrics": summary_payload}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = build_report(args.output_dir)
    print(result["html"])
    return Path(result["html"])


if __name__ == "__main__":
    main()
