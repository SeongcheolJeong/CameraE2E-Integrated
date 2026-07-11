"""Dataset source adapters for camera-aware RAW generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import imageio.v3 as iio

from .models import SceneCase

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_SPLITS = ("train", "validation", "val", "test")


def normalize_dataset_split(name: str) -> str:
    key = str(name).strip().lower()
    if key == "val":
        return "validation"
    if key not in {"train", "validation", "test"}:
        raise ValueError("dataset split must be one of: train, validation, val, test")
    return key


def kitti_dataset_inventory(
    root: str | Path,
    *,
    splits: Iterable[str] = ("train",),
    include_records: bool = False,
) -> dict[str, Any]:
    """Inventory a KITTI or YOLO-converted KITTI directory without copying it."""

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"KITTI dataset root does not exist: {source_root}")
    requested = tuple(dict.fromkeys(normalize_dataset_split(item) for item in splits))
    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    label_count = 0
    calibration_count = 0
    fingerprint_rows: list[dict[str, Any]] = []
    for split in requested:
        disk_split = _disk_split(source_root, split)
        image_root = source_root / "images" / disk_split
        if not image_root.is_dir():
            continue
        label_root = source_root / "labels" / disk_split
        for image_path in _image_files(image_root):
            label_path = label_root / f"{image_path.stem}.txt"
            calibration_path = _calibration_path(source_root, disk_split, image_path.stem)
            labels = _label_classes(label_path) if label_path.is_file() else []
            record = {
                "frame_id": image_path.stem,
                "group_id": image_path.stem,
                "split": split,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()) if label_path.is_file() else None,
                "calibration_path": (
                    str(calibration_path.resolve()) if calibration_path is not None else None
                ),
                "classes": labels,
            }
            records.append(record)
            split_counts[split] += 1
            class_counts.update(labels)
            label_count += int(label_path.is_file())
            calibration_count += int(calibration_path is not None)
            fingerprint_rows.append(
                {
                    "split": split,
                    "image": str(image_path.relative_to(source_root)),
                    "image_size": image_path.stat().st_size,
                    "label": (
                        str(label_path.relative_to(source_root)) if label_path.is_file() else None
                    ),
                    "label_sha256": _sha256_file(label_path) if label_path.is_file() else None,
                    "calibration": (
                        str(calibration_path.relative_to(source_root))
                        if calibration_path is not None
                        else None
                    ),
                    "calibration_sha256": (
                        _sha256_file(calibration_path) if calibration_path is not None else None
                    ),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "camerae2e_dataset_inventory_v1",
        "adapter": "kitti",
        "root": str(source_root),
        "requested_splits": list(requested),
        "available_scene_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "label_count": label_count,
        "label_coverage": label_count / max(len(records), 1),
        "calibration_count": calibration_count,
        "calibration_coverage": calibration_count / max(len(records), 1),
        "class_counts": dict(sorted(class_counts.items())),
        "inventory_fingerprint": _canonical_hash(fingerprint_rows),
        "truth_boundary": (
            "KITTI RGB frames are display-derived scene proxies. Inventory calibration "
            "describes source image geometry; it does not recover source sensor RAW."
        ),
    }
    if include_records:
        payload["records"] = records
    return payload


def kitti_dataset_scenes(
    root: str | Path,
    *,
    splits: Iterable[str] = ("train",),
    count: int | None = None,
) -> list[SceneCase]:
    inventory = kitti_dataset_inventory(root, splits=splits, include_records=True)
    records = list(inventory.get("records", []))
    if count is not None:
        records = records[: max(int(count), 0)]
    scenes: list[SceneCase] = []
    for record in records:
        image_path = Path(str(record["image_path"]))
        image = iio.immeta(image_path)
        shape = image.get("shape")
        calibration = _read_kitti_calibration(record.get("calibration_path"))
        metadata = {
            "dataset": "KITTI",
            "adapter": "kitti",
            "frame_id": record["frame_id"],
            "group_id": record["group_id"],
            "split": record["split"],
            "classes": record.get("classes", []),
            "source_intrinsics": calibration,
            "source_image_size_rc": list(shape[:2]) if shape else None,
            "truth_boundary": (
                "RGB-to-scene display proxy; original spectra, photons, and sensor RAW "
                "are not recovered."
            ),
        }
        scenes.append(
            SceneCase(
                id=f"kitti_{record['split']}_{record['frame_id']}",
                name=f"KITTI {record['frame_id']}",
                source_kind="display_rgb_proxy",
                scene_type="rgb_file",
                image_path=str(image_path),
                label_path=record.get("label_path"),
                metadata=metadata,
            )
        )
    return scenes


def _disk_split(root: Path, split: str) -> str:
    candidates = ("val", "validation") if split == "validation" else (split,)
    return next((item for item in candidates if (root / "images" / item).is_dir()), candidates[0])


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)


def _calibration_path(root: Path, split: str, stem: str) -> Path | None:
    candidates = (
        root / "calib" / split / f"{stem}.txt",
        root / "calibration" / split / f"{stem}.txt",
        root / "calib" / f"{stem}.txt",
        root / "calibration" / f"{stem}.txt",
    )
    return next((path for path in candidates if path.is_file()), None)


def _read_kitti_calibration(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, values = line.partition(":")
        if separator and name.strip() in {"P2", "P_rect_02"}:
            numbers = [float(item) for item in values.split()]
            if len(numbers) == 12:
                return {
                    "source": str(path),
                    "projection_key": name.strip(),
                    "projection_matrix": [numbers[0:4], numbers[4:8], numbers[8:12]],
                    "fx_px": numbers[0],
                    "fy_px": numbers[5],
                    "cx_px": numbers[2],
                    "cy_px": numbers[6],
                }
    return {"source": str(path), "projection_key": None}


def _label_classes(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 5:
            values.append(parts[0])
        else:
            values.append(parts[0])
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
