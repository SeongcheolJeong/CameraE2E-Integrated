"""Collect public/local ISP tuning files into pyisetcam HW ISP profiles.

The collector intentionally writes a normalized summary profile, not a raw
vendor/libcamera tuning dump. HW timing fields that are absent from tuning files
remain seed values and should be replaced by BSP or measured data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam.hwisp_db import hw_isp_config_from_profile


DEFAULT_SEARCH_DIRS = (
    Path("/usr/share/libcamera/ipa/rpi/vc4"),
    Path("/usr/share/libcamera/ipa/rpi/pisp"),
    Path("/usr/share/libcamera/ipa/raspberrypi"),
)


def _algorithm_map(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("algorithms"), list):
        algorithms: dict[str, Any] = {}
        for item in payload["algorithms"]:
            if isinstance(item, dict):
                algorithms.update(item)
        return algorithms
    return payload


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _target_from_agc(agc: dict[str, Any], default: float = 0.18) -> float:
    y_target = agc.get("y_target")
    if isinstance(y_target, list) and len(y_target) >= 2:
        return _number(y_target[1], default)
    return float(default)


def _pipeline_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "pisp" in parts:
        return "pisp"
    if "vc4" in parts or "raspberrypi" in parts:
        return "vc4"
    return "unknown"


def _profile_from_tuning(path: Path, output_name: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    algorithms = _algorithm_map(payload)
    pipeline = _pipeline_from_path(path)
    stem = path.stem.replace("-", "_")

    lux = algorithms.get("rpi.lux", {})
    black = algorithms.get("rpi.black_level", {})
    noise = algorithms.get("rpi.noise", {})
    geq = algorithms.get("rpi.geq", {})
    agc = algorithms.get("rpi.agc", {})

    target_luma = _target_from_agc(agc, default=0.18)
    reference_shutter = _number(lux.get("reference_shutter_speed"), 8000.0)
    base_profile = hw_isp_config_from_profile(
        "generic_1080p_30fps",
        control_path={"target_luma": target_luma},
        sensor_timing={"exposure_time_us": min(max(reference_shutter, 100.0), 30000.0)},
    )

    config = {
        "sensor_timing": base_profile.sensor_timing.__dict__.copy(),
        "stages": [stage.__dict__.copy() for stage in base_profile.stages],
        "control_path": base_profile.control_path.__dict__.copy(),
        "transport": base_profile.transport.__dict__.copy(),
        "global_latency_factor": base_profile.global_latency_factor,
        "seed": base_profile.seed,
    }
    name = output_name or f"libcamera_{pipeline}_{stem}_collected"
    return {
        "schema_version": 1,
        "name": name,
        "description": f"Collected normalized HW ISP seed profile from {path.name}.",
        "confidence": "collected_tuning_with_seed_timing",
        "source": {
            "type": "local_libcamera_tuning",
            "path": str(path),
            "pipeline": pipeline,
            "notes": [
                "Raw tuning files usually do not contain complete HW latency parameters.",
                "Missing timing is filled from generic_1080p_30fps.",
            ],
        },
        "config": config,
        "calibration": {
            "black_level": {
                "value": black.get("black_level"),
                "units": "DN",
                "source": "rpi.black_level.black_level",
            },
            "lux_reference": {
                "reference_shutter_speed_us": lux.get("reference_shutter_speed"),
                "reference_gain": lux.get("reference_gain"),
                "reference_lux": lux.get("reference_lux"),
                "reference_Y": lux.get("reference_Y"),
                "source": "rpi.lux",
            },
            "noise": {
                "reference_constant": noise.get("reference_constant"),
                "reference_slope": noise.get("reference_slope"),
                "source": "rpi.noise",
            },
            "geq": {
                "offset": geq.get("offset"),
                "slope": geq.get("slope"),
                "source": "rpi.geq",
            },
            "agc": {
                "y_target_seed": target_luma,
                "source": "rpi.agc.y_target",
            },
            "algorithm_keys": sorted(algorithms.keys()),
        },
        "notes": [
            "Review sensor_timing, stage clocks, cycle latency, transport delays, and queues.",
            "Keep raw vendor/NDA tuning files outside the repository unless redistribution is allowed.",
        ],
    }


def collect(
    sources: list[Path],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in sources:
        if source.is_dir():
            files = sorted(source.glob("*.json"))
        else:
            files = [source]
        for path in files:
            profile = _profile_from_tuning(path)
            destination = output_dir / f"{profile['name']}.json"
            if destination.exists() and not overwrite:
                continue
            destination.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
            written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="libcamera tuning JSON files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "hwisp" / "collected",
        help="Directory for normalized profile JSON outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = args.sources or [path for path in DEFAULT_SEARCH_DIRS if path.exists()]
    if not sources:
        raise SystemExit(
            "No source tuning files found. Pass explicit libcamera JSON files or directories."
        )
    written = collect(sources, args.output_dir, overwrite=bool(args.overwrite))
    for path in written:
        print(path)
    if written:
        print(f"Set PYISETCAM_HWISP_DB={args.output_dir} to load these collected profiles.")


if __name__ == "__main__":
    main()
