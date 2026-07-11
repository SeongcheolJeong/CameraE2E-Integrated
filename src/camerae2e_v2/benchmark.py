"""Deterministic KITTI benchmark discovery and stratified scene selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import BenchmarkSuite, SceneCase, StudyRecord
from .provenance import canonical_hash, runtime_provenance, sha256_file

ADAS_KITTI_CLASS_IDS = frozenset({0, 1, 2, 3, 4, 5, 6})
ADAS_KITTI_CLASS_NAMES = frozenset(
    {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"}
)
ADAS_CLASS_MAP = {
    "Car": "vehicle",
    "Van": "vehicle",
    "Truck": "vehicle",
    "Tram": "vehicle",
    "Pedestrian": "pedestrian",
    "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
}


def resolve_benchmark_root(study: StudyRecord) -> Path | None:
    configured = study.spec.benchmark.source_root
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.is_dir():
            return root
    for scene in study.spec.scenes:
        if not scene.image_path:
            continue
        image = Path(scene.image_path).expanduser().resolve()
        if image.parent.name == "train" and image.parent.parent.name == "images":
            root = image.parent.parent.parent
            if root.is_dir():
                return root
    return None


def benchmark_scenes(
    study: StudyRecord,
    count: int,
    *,
    suite: BenchmarkSuite | None = None,
) -> list[SceneCase]:
    benchmark = suite or study.spec.benchmark
    root = resolve_benchmark_root(study)
    if root is None:
        return list(study.spec.scenes[:count])
    records = _dataset_records(str(root))
    if benchmark.stratification == "ordered":
        selected = list(records[:count])
    else:
        selected = _round_robin_strata(records, count)
    return [
        SceneCase(
            id=f"kitti_{image.stem}",
            name=f"KITTI {image.stem}",
            source_kind="display_rgb_proxy",
            scene_type="rgb_file",
            image_path=str(image),
            label_path=str(label),
            metadata={
                "dataset": "KITTI",
                "frame_id": image.stem,
                "stratum": stratum,
                "split": "train",
                "group_id": image.stem,
                "truth_boundary": (
                    "RGB-to-scene display proxy; original scene spectra and sensor RAW are not "
                    "recovered."
                ),
            },
        )
        for image, label, stratum in selected
    ]


def benchmark_inventory(study: StudyRecord) -> dict[str, object]:
    root = resolve_benchmark_root(study)
    records = [] if root is None else list(_dataset_records(str(root)))
    strata: dict[str, int] = defaultdict(int)
    for _image, _label, stratum in records:
        strata[stratum] += 1
    return {
        "root": None if root is None else str(root),
        "available_scene_count": len(records),
        "quick_scene_count": study.spec.benchmark.quick_scene_count,
        "final_scene_count": study.spec.benchmark.final_scene_count,
        "metric_version": study.spec.benchmark.metric_version,
        "enough_for_quick": len(records) >= study.spec.benchmark.quick_scene_count,
        "enough_for_final": len(records) >= study.spec.benchmark.final_scene_count,
        "strata": dict(sorted(strata.items())),
    }


def benchmark_manifest(
    study: StudyRecord,
    scenes: Iterable[SceneCase],
) -> dict[str, Any]:
    """Build a path-independent manifest for one exact benchmark selection."""

    selected = list(scenes)
    scene_records = []
    for scene in selected:
        image = Path(scene.image_path).expanduser() if scene.image_path else None
        label = Path(scene.label_path).expanduser() if scene.label_path else None
        scene_records.append(
            {
                "scene_id": scene.id,
                "frame_id": str(scene.metadata.get("frame_id", scene.id)),
                "group_id": str(scene.metadata.get("group_id", scene.id)),
                "split": str(scene.metadata.get("split", "unspecified")),
                "stratum": str(scene.metadata.get("stratum", "unspecified")),
                "source_kind": scene.source_kind,
                "image_sha256": sha256_file(image) if image and image.is_file() else None,
                "label_sha256": sha256_file(label) if label and label.is_file() else None,
            }
        )
    model = (
        Path(study.spec.perception_model_path).expanduser()
        if study.spec.perception_model_path
        else None
    )
    suite = study.spec.benchmark
    payload: dict[str, Any] = {
        "schema_version": "camerae2e_benchmark_manifest_v1",
        "metric_version": suite.metric_version,
        "suite_name": suite.name,
        "study_id": study.id,
        "seed": study.spec.seed,
        "target_profile": study.spec.target_profile,
        "selection": {
            "stratification": suite.stratification,
            "scene_count": len(scene_records),
            "scenes": scene_records,
        },
        "detector": {
            "configured": bool(model and model.is_file()),
            "sha256": sha256_file(model) if model and model.is_file() else None,
            "size_bytes": model.stat().st_size if model and model.is_file() else None,
        },
        "class_contract": {
            "kitti_to_adas_group": ADAS_CLASS_MAP,
            "ignored_classes": ["DontCare", "Misc"],
        },
        "thresholds": {
            "detector_map50_min": suite.detector_map50_min,
            "detector_map50_95_min": suite.detector_map50_95_min,
            "detector_recall_min": suite.detector_recall_min,
            "ideal_recall_retention_min": suite.ideal_recall_retention_min,
            "ideal_ssim_min": suite.ideal_ssim_min,
            "fidelity_recall_retention_min": suite.fidelity_recall_retention_min,
            "fidelity_color_imbalance_max": suite.fidelity_color_imbalance_max,
            "fidelity_ssim_min": suite.fidelity_ssim_min,
        },
        "robustness_cases": [
            item.model_dump(mode="json") for item in suite.robustness_cases
        ],
        "fidelity": study.spec.fidelity_policy.model_dump(mode="json"),
        "runtime": runtime_provenance(),
        "truth_boundary": (
            "The manifest fixes inputs and scoring semantics; it does not convert KITTI RGB "
            "into original scene spectra or measured camera RAW."
        ),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


@lru_cache(maxsize=8)
def _dataset_records(root_text: str) -> tuple[tuple[Path, Path, str], ...]:
    root = Path(root_text)
    image_root = root / "images" / "train"
    label_root = root / "labels" / "train"
    records = []
    for image in sorted(image_root.glob("*.png")):
        label = label_root / f"{image.stem}.txt"
        if label.is_file():
            records.append((image.resolve(), label.resolve(), _label_stratum(label)))
    return tuple(records)


def _round_robin_strata(
    records: Iterable[tuple[Path, Path, str]], count: int
) -> list[tuple[Path, Path, str]]:
    buckets: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for record in records:
        buckets[record[2]].append(record)
    selected: list[tuple[Path, Path, str]] = []
    names = sorted(buckets)
    index = 0
    while names and len(selected) < count:
        name = names[index % len(names)]
        bucket = buckets[name]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            names.remove(name)
            index = 0
        else:
            index += 1
    return selected


def _label_stratum(path: Path) -> str:
    labels: set[str] = set()
    small = False
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 5:
            class_id = int(float(parts[0]))
            if class_id not in ADAS_KITTI_CLASS_IDS:
                continue
            labels.add(_class_group(class_id))
            small = small or float(parts[4]) < 0.09
        elif len(parts) >= 8:
            if parts[0] not in ADAS_KITTI_CLASS_NAMES:
                continue
            labels.add(_name_group(parts[0]))
            small = small or (float(parts[7]) - float(parts[5])) < 32.0
    if not labels:
        return "no_adas_object"
    group = "mixed" if len(labels) > 1 else next(iter(labels))
    return f"{group}_{'small' if small else 'regular'}"


def _class_group(class_id: int) -> str:
    if class_id in {0, 1, 2, 6}:
        return "vehicle"
    if class_id in {3, 4}:
        return "pedestrian"
    return "cyclist"


def _name_group(name: str) -> str:
    if name in {"Car", "Van", "Truck", "Tram"}:
        return "vehicle"
    if name in {"Pedestrian", "Person_sitting"}:
        return "pedestrian"
    return "cyclist"
