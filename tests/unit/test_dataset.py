from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyisetcam import (
    camerae2e_adas_camera_spec,
    camerae2e_dataset_export,
    camerae2e_dataset_export_adas_kitti_demo,
    camerae2e_dataset_export_camera_spec_variants,
    camerae2e_dataset_export_from_optimization,
    camerae2e_dataset_export_perception_index,
    camerae2e_dataset_validate,
    camerae2e_kitti_yolo_labels,
    camerae2e_optimize_parameters,
    image_sensor_db_optimize_camera_parameters,
    image_sensor_db_records,
)


def test_camerae2e_dataset_export_writes_manifest_raw_preview_and_labels(tmp_path: Path) -> None:
    manifest = camerae2e_dataset_export(
        tmp_path,
        [
            {
                "name": "dataset_case",
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
            }
        ],
        seed=5,
        labels={"objects": [{"label": "chart", "bbox_xyxy": [0, 0, 4, 4]}]},
        split="train",
    )
    record = manifest["records"][0]

    assert manifest["schema_version"] == "camerae2e_dataset_manifest_v1"
    assert Path(record["raw"]).exists()
    assert Path(record["rgb"]).exists()
    assert Path(record["labels"]).exists()
    assert record["split"] == "train"
    assert record["raw_sha256"].startswith("sha256:")
    assert record["raw_content_sha256"].startswith("sha256:")
    assert (tmp_path / "metadata.jsonl").exists()

    raw = np.load(record["raw"])
    labels = json.loads(Path(record["labels"]).read_text(encoding="utf-8"))
    validation = camerae2e_dataset_validate(tmp_path)

    assert raw["raw"].shape == tuple(record["raw_shape"])
    assert labels["labels"]["objects"][0]["label"] == "chart"
    assert manifest["format"]["dng"] == "not emitted as standards-compliant DNG in v1"
    assert validation["ok"] is True


def test_camerae2e_dataset_export_optional_stage_container_and_uncertainty(
    tmp_path: Path,
) -> None:
    manifest = camerae2e_dataset_export(
        tmp_path,
        [
            {
                "name": "dataset_packaged_case",
                "scene": {"type": "uniform ee", "args": [8]},
                "sensor": {"noise_flag": 0},
            }
        ],
        seed=6,
        labels={
            "objects": [{"label": "chart", "bbox_xyxy": [0, 0, 4, 4]}],
            "keypoints": [{"label": "corner", "points_xyc": [[1, 1, 1.0]]}],
            "polylines": [{"label": "edge", "points_xy": [[0, 0], [2, 2]]}],
        },
        split={"train": 0.7, "val": 0.15, "test": 0.15},
        include_rgb=False,
        include_stage_outputs=True,
        include_container=True,
        include_uncertainty=True,
    )
    record = manifest["records"][0]
    validation = camerae2e_dataset_validate(manifest)

    assert manifest["split_policy"]["mode"] == "weighted_deterministic"
    assert manifest["format"]["container"].startswith("deterministic DNG-like")
    assert "not a standards-compliant DNG" in manifest["truth_boundary"]
    assert Path(record["stage_outputs"]).exists()
    assert Path(record["container"]).exists()
    assert Path(record["uncertainty"]).exists()
    assert record["stage_outputs_sha256"].startswith("sha256:")
    assert record["container_sha256"].startswith("sha256:")
    assert record["uncertainty_sha256"].startswith("sha256:")
    assert record["label_summary"]["object_count"] == 1
    assert record["label_summary"]["keypoint_count"] == 1
    assert record["label_summary"]["polyline_count"] == 1
    assert record["uncertainty_model"]["readiness_tier"] == "proxy"
    with np.load(record["uncertainty"], allow_pickle=False) as payload:
        assert payload["uncertainty"].shape == tuple(record["raw_shape"])
        assert payload["confidence"].shape == tuple(record["raw_shape"])
    assert validation["ok"] is True


def test_camerae2e_dataset_export_is_seed_reproducible_and_splits_cases(
    tmp_path: Path,
) -> None:
    scenarios = [
        {"name": f"case_{index}", "scene": {"type": "uniform ee", "args": [8]}}
        for index in range(3)
    ]
    first = camerae2e_dataset_export(
        tmp_path,
        scenarios,
        seed=9,
        split={"train": 2.0, "val": 1.0},
        include_rgb=False,
    )
    second = camerae2e_dataset_export(
        tmp_path,
        scenarios,
        seed=9,
        split={"train": 2.0, "val": 1.0},
        include_rgb=False,
    )

    assert first["splits"] == {"train": 2, "val": 1}
    assert [item["split"] for item in first["records"]] == ["train", "train", "val"]
    assert [item["raw_sha256"] for item in first["records"]] == [
        item["raw_sha256"] for item in second["records"]
    ]
    assert [item["raw_content_sha256"] for item in first["records"]] == [
        item["raw_content_sha256"] for item in second["records"]
    ]
    assert first["integrity"]["metadata_jsonl_sha256"] == second["integrity"][
        "metadata_jsonl_sha256"
    ]


def test_camerae2e_dataset_validate_detects_tampered_raw(tmp_path: Path) -> None:
    manifest = camerae2e_dataset_export(
        tmp_path,
        [{"scene": {"type": "uniform ee", "args": [8]}}],
        seed=2,
        include_rgb=False,
    )
    raw_path = Path(manifest["records"][0]["raw"])
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")

    validation = camerae2e_dataset_validate(manifest)

    assert validation["ok"] is False
    assert any(issue["kind"] == "raw_sha256" for issue in validation["issues"])


def test_camerae2e_dataset_export_from_optimization_uses_pareto_scenarios(
    tmp_path: Path,
) -> None:
    result = camerae2e_optimize_parameters(
        {
            "name": "dataset_from_optimizer",
            "scene": {"type": "uniform ee", "args": [8]},
            "sensor": {"noise_flag": 0},
        },
        {"sensor.integration_time": [0.001, 0.004]},
        [
            {"metric": "metrics.color.rgb_mean", "direction": "maximize"},
            {"metric": "metrics.artifact.raw_std", "direction": "minimize"},
        ],
        seed=31,
    )
    manifest = camerae2e_dataset_export_from_optimization(
        tmp_path,
        result,
        selection="pareto",
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)

    assert manifest["case_count"] == result["pareto_case_count"] == 2
    assert manifest["source_optimization"]["selection"] == "pareto"
    assert manifest["source_optimization"]["selected_case_count"] == 2
    assert {record["split"] for record in manifest["records"]} == {"pareto"}
    assert manifest["records"][0]["parameter_lineage"]
    assert any(
        item["path"] == "sensor.integration_time"
        for item in manifest["records"][0]["parameter_lineage"]
    )
    assert validation["ok"] is True


def test_camerae2e_dataset_export_preserves_sensor_db_optimization_source(
    tmp_path: Path,
) -> None:
    first = image_sensor_db_records(limit=1)[0]
    result = image_sensor_db_optimize_camera_parameters(
        first["sensor_id"],
        strategy="analytic_only",
        base_scenario={"scene": {"type": "uniform ee", "args": [8]}},
        preset="exposure",
        parameter_space={"sensor.integration_time": [0.001, 0.004]},
        objective={"metric": "metrics.color.rgb_mean", "direction": "maximize"},
        max_cases=2,
        seed=37,
        top_k=1,
    )
    manifest = camerae2e_dataset_export_from_optimization(
        tmp_path,
        result,
        selection="best",
        include_rgb=False,
    )

    source = manifest["source_optimization"]["source_image_sensor_db"]
    assert source["sensor_id"] == first["sensor_id"]
    assert source["strategy"] == "analytic_only"
    assert (
        manifest["records"][0]["scenario"]["image_sensor_db"]["sensor_id"]
        == first["sensor_id"]
    )


def test_camerae2e_adas_kitti_demo_exports_raw_and_yolo_labels(tmp_path: Path) -> None:
    spec = camerae2e_adas_camera_spec()
    labels = camerae2e_kitti_yolo_labels(
        ["Car 0.00 0 0.00 10 20 50 60 1.5 1.6 4.0 0 0 20 0"],
        image_size=(96, 320),
    )
    manifest = camerae2e_dataset_export_adas_kitti_demo(
        tmp_path,
        case_count=1,
        seed=77,
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)
    record = manifest["records"][0]
    record_labels = json.loads(Path(record["labels"]).read_text(encoding="utf-8"))

    assert spec["preset"] == "kitti_yolo_demo"
    assert labels["objects"][0]["class_id"] == 0
    assert labels["objects"][0]["yolo_xywhn"][2] > 0.0
    assert manifest["adas_kitti_demo"]["input_mode"] == "synthetic"
    assert manifest["adas_kitti_demo"]["camera_spec"]["readiness_tier"] == "proxy"
    assert record["parameter_lineage"]
    assert any(item["path"] == "optics.focal_length" for item in record["parameter_lineage"])
    assert record["label_coordinate_frame"] == "exported_raw_shape"
    assert record_labels["labels"]["objects"][0]["label"] == "Car"
    assert record_labels["labels"]["objects"][0]["yolo_xywhn"][2] > 0.0
    assert record_labels["labels"]["image_size_rc"] == record["raw_shape"][:2]
    assert validation["ok"] is True


def test_camerae2e_dataset_export_perception_index_writes_raw_and_yolo_views(
    tmp_path: Path,
) -> None:
    manifest = camerae2e_dataset_export_adas_kitti_demo(
        tmp_path / "dataset",
        case_count=1,
        seed=81,
        include_rgb=True,
    )
    index = camerae2e_dataset_export_perception_index(
        manifest,
        output_dir=tmp_path / "perception_index",
        copy_images=True,
    )
    raw_manifest = Path(index["outputs"]["raw_manifest"]["path"])
    yolo_root = Path(index["outputs"]["yolo"]["root"])
    yolo_yaml = Path(index["outputs"]["yolo"]["dataset_yaml"])
    yolo_label = yolo_root / "labels" / "demo" / "case_0000.txt"
    copied_image = yolo_root / "images" / "demo" / "case_0000.png"

    assert index["schema_version"] == "camerae2e_perception_training_index_v1"
    assert index["warning_count"] == 0
    assert raw_manifest.exists()
    assert yolo_yaml.exists()
    assert yolo_label.exists()
    assert copied_image.exists()
    assert "raw_manifest indexes RAW NPZ" in index["truth_boundary"]
    assert "YOLO view uses RGB previews" in index["outputs"]["yolo"]["truth_boundary"]
    assert len(raw_manifest.read_text(encoding="utf-8").splitlines()) == 1
    assert yolo_label.read_text(encoding="utf-8").strip().split()[0] == "0"


def test_camerae2e_camera_spec_variants_recapture_scene_with_target_specs(
    tmp_path: Path,
) -> None:
    manifest = camerae2e_dataset_export_camera_spec_variants(
        tmp_path,
        target_specs=["wide_fov_adas_demo", "narrow_fov_adas_demo"],
        case_count=1,
        seed=88,
        include_rgb=False,
    )
    validation = camerae2e_dataset_validate(manifest)
    presets = [
        record["scenario"]["target_camera_spec"]["preset"]
        for record in manifest["records"]
    ]
    raw_shapes = [tuple(record["raw_shape"][:2]) for record in manifest["records"]]
    focal_lengths = [
        record["scenario"]["target_camera_spec"]["optics"]["focal_length_m"]
        for record in manifest["records"]
    ]
    transforms = [
        record["scenario"]["geometric_scene_transform"]
        for record in manifest["records"]
    ]
    wide_labels = json.loads(Path(manifest["records"][0]["labels"]).read_text(encoding="utf-8"))
    narrow_labels = json.loads(Path(manifest["records"][1]["labels"]).read_text(encoding="utf-8"))
    wide_box_width = wide_labels["labels"]["objects"][0]["yolo_xywhn"][2]
    narrow_box_width = narrow_labels["labels"]["objects"][0]["yolo_xywhn"][2]

    assert manifest["case_count"] == 2
    assert presets == ["wide_fov_adas_demo", "narrow_fov_adas_demo"]
    assert focal_lengths[0] != focal_lengths[1]
    assert manifest["camera_spec_variants"]["geometric_transform"]["mode"] == "pinhole_crop"
    assert transforms[0]["mode"] == "pinhole_crop"
    assert transforms[0]["warp"]["out_of_source_fraction"] > 0.0
    assert transforms[1]["object_scale_x"] > transforms[0]["object_scale_x"]
    assert narrow_box_width > wide_box_width
    assert all(
        any(item["path"] == "optics.focal_length" for item in record["parameter_lineage"])
        for record in manifest["records"]
    )
    assert wide_labels["labels"]["coordinate_frame"] == "exported_raw_shape"
    assert wide_labels["labels"]["image_size_rc"] == list(raw_shapes[0])
    assert "RGB-to-scene proxy" in manifest["camera_spec_variants"]["truth_boundary"]
    assert validation["ok"] is True
