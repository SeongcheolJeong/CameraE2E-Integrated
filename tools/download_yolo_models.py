"""Download YOLO11n task-perception weights into the local pyisetcam cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam.task_perception import task_model_profile, task_model_profile_names  # noqa: E402


DEFAULT_CACHE_DIR = Path(os.environ.get("PYISETCAM_TASK_MODEL_CACHE", Path.home() / ".cache" / "pyisetcam" / "task_perception" / "yolo"))
YOLO11_PROFILES = (
    "ultralytics_yolo11n_detection",
    "ultralytics_yolo11n_segmentation",
    "ultralytics_yolo11n_classification",
    "ultralytics_yolo11n_pose",
    "ultralytics_yolo11n_obb",
    "ultralytics_yolo11n_bytetrack",
)


def _load_yolo(model_id: str, cache_dir: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install YOLO support with `python -m pip install -e \".[yolo]\"` or `python -m pip install ultralytics`.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / model_id
    if target.exists():
        return target
    previous = Path.cwd()
    try:
        os.chdir(cache_dir)
        YOLO(model_id)
    finally:
        os.chdir(previous)
    if not target.exists():
        raise FileNotFoundError(f"Ultralytics did not create expected model file {target}.")
    return target


def download_yolo_models(cache_dir: Path = DEFAULT_CACHE_DIR, profiles: tuple[str, ...] = YOLO11_PROFILES) -> dict[str, Any]:
    available = set(task_model_profile_names())
    records = []
    seen_model_ids: set[str] = set()
    for profile_name in profiles:
        if profile_name not in available:
            raise KeyError(f"Unknown model profile {profile_name!r}.")
        profile = task_model_profile(profile_name)
        model_id = str(profile.get("model_id", ""))
        if not model_id or model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        path = _load_yolo(model_id, cache_dir)
        records.append(
            {
                "profile": profile_name,
                "model_id": model_id,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    summary = {"cache_dir": str(cache_dir), "models": records}
    (cache_dir / "download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--profiles", nargs="*", default=list(YOLO11_PROFILES))
    args = parser.parse_args(argv)
    summary = download_yolo_models(args.cache_dir, tuple(args.profiles))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
