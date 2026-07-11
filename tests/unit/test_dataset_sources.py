from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from camerae2e_v2 import CameraE2EService, DatasetEstimateRequest, DatasetInventoryRequest
from camerae2e_v2.dataset_sources import kitti_dataset_inventory, kitti_dataset_scenes
from camerae2e_v2.models import JobStatus


def _kitti_fixture(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "calib" / split).mkdir(parents=True)
        frame_id = "000001" if split == "train" else "000002"
        image = np.zeros((16, 24, 3), dtype=np.uint8)
        image[5:13, 8:18] = np.asarray([170, 90, 40], dtype=np.uint8)
        iio.imwrite(root / "images" / split / f"{frame_id}.png", image)
        (root / "labels" / split / f"{frame_id}.txt").write_text(
            "Car 0 0 0 8 5 18 13 1.5 1.6 4.0 0 0 20 0\n",
            encoding="utf-8",
        )
        (root / "calib" / split / f"{frame_id}.txt").write_text(
            "P2: 20 0 11.5 0 0 20 7.5 0 0 0 1 0\n",
            encoding="utf-8",
        )
    return root


def test_kitti_adapter_inventories_splits_labels_and_calibration(tmp_path: Path) -> None:
    root = _kitti_fixture(tmp_path / "kitti")

    inventory = kitti_dataset_inventory(
        root,
        splits=["train", "validation"],
        include_records=True,
    )
    scenes = kitti_dataset_scenes(root, splits=["validation"])

    assert inventory["available_scene_count"] == 2
    assert inventory["split_counts"] == {"train": 1, "validation": 1}
    assert inventory["label_coverage"] == 1.0
    assert inventory["calibration_coverage"] == 1.0
    assert inventory["inventory_fingerprint"]
    assert scenes[0].metadata["split"] == "validation"
    assert scenes[0].metadata["source_intrinsics"]["fx_px"] == 20.0
    assert scenes[0].source_kind == "display_rgb_proxy"


def test_v2_camera_aware_kitti_export_writes_v3_raw_contract(tmp_path: Path) -> None:
    root = _kitti_fixture(tmp_path / "kitti")
    service = CameraE2EService(tmp_path / "projects", max_workers=1)
    project = service.create_project("Camera-aware RAW", preset="general")
    study = project.store.list_studies()[0]
    request = {
        "selection": "baseline",
        "case_count": 1,
        "scene_count": 1,
        "source_adapter": "kitti",
        "source_root": str(root),
        "source_splits": ["train"],
        "resolution_policy": "source_bounded",
        "exposure_variants_ev": [-1.0, 1.0],
        "noise_repeats": 2,
        "include_raw_uint16": True,
    }

    inventory = service.dataset_inventory(
        project.info.id,
        study.id,
        DatasetInventoryRequest(source_root=str(root), source_splits=["train"]),
    )
    estimate = service.dataset_estimate(
        project.info.id,
        study.id,
        DatasetEstimateRequest.model_validate(request),
    )
    first = service.execute_job_now(project.info.id, study.id, "dataset_export", request)
    second = service.execute_job_now(project.info.id, study.id, "dataset_export", request)

    assert inventory["available_scene_count"] == 1
    assert estimate["sample_count"] == 4
    assert first.status == JobStatus.SUCCEEDED
    assert second.status == JobStatus.SUCCEEDED
    assert first.result is not None and second.result is not None
    first_manifest = first.result["manifest"]
    assert first_manifest["schema_version"] == "camerae2e_raw_dataset_v3"
    assert first_manifest["case_count"] == 4
    assert first_manifest["split_counts"]["train"] == 4
    assert first_manifest["recipe"]["resolution_policy"] == "source_bounded"
    assert "raw_uint16_npy" in first_manifest["formats"]
    assert first_manifest["camera_profile_summaries"][0]["sample_count"] == 4
    root_path = Path(first.result["dataset_root"])
    rows = [json.loads(line) for line in (root_path / "metadata.jsonl").read_text().splitlines()]
    assert {row["exposure_ev"] for row in rows} == {-1.0, 1.0}
    assert {row["noise_repeat"] for row in rows} == {0, 1}
    assert all(row["source_intrinsics"]["fx_px"] == 20.0 for row in rows)
    assert all(row["geometry"]["pinhole"]["mode"] == "pinhole_intrinsics" for row in rows)
    assert all(row["upsampled_scene_proxy"] is False for row in rows)
    labels = json.loads((root_path / rows[0]["labels"]).read_text(encoding="utf-8"))
    transformed_width = labels["objects"][0]["bbox_xyxy"][2] - labels["objects"][0][
        "bbox_xyxy"
    ][0]
    assert transformed_width < 10.0
    with np.load(root_path / rows[0]["raw"], allow_pickle=False) as payload:
        assert set(payload.files) == {
            "raw",
            "sensor_digital",
            "black_level",
            "white_level",
            "bit_depth",
            "cfa_pattern",
        }
        assert payload["raw"].dtype == np.float32
    first_ids = {row["sample_id"] for row in rows}
    second_root = Path(second.result["dataset_root"])
    second_rows = [
        json.loads(line) for line in (second_root / "metadata.jsonl").read_text().splitlines()
    ]
    assert first_ids == {row["sample_id"] for row in second_rows}
    assert first.result["dataset_root"] == second.result["dataset_root"]
    assert second.result["resumed_sample_count"] == 4
    service.shutdown()
