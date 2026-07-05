"""CameraE2E RAW dataset export helpers."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import imageio.v3 as iio
import numpy as np
import tifffile

from .assets import AssetStore
from .scene import scene_from_file, scene_set
from .system_faca import camerae2e_faca_report, camerae2e_run_scenario
from .types import Camera, Scene

_KITTI_CLASS_IDS = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": -1,
}


def camerae2e_dataset_export(
    output_dir: str | Path,
    scenarios: Iterable[Mapping[str, Any]] | None = None,
    *,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    labels: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    include_rgb: bool = True,
    include_tiff: bool = False,
    include_stage_outputs: bool = False,
    include_container: bool = False,
    include_uncertainty: bool = False,
    split: Mapping[str, float] | Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Export reproducible CameraE2E RAW/stage data for perception training.

    The v1 writer intentionally does not emit DNG.  It writes raw arrays and
    metadata in simple, inspectable formats so downstream training code can
    choose its own packing and camera-specific metadata policy.
    """

    root = Path(output_dir).expanduser().resolve()
    raw_dir = root / "raw"
    rgb_dir = root / "rgb"
    label_dir = root / "labels"
    tiff_dir = root / "raw_tiff"
    stage_dir = root / "stage_outputs"
    container_dir = root / "containers"
    uncertainty_dir = root / "uncertainty"
    for directory in (
        raw_dir,
        rgb_dir,
        label_dir,
        tiff_dir if include_tiff else None,
        stage_dir if include_stage_outputs else None,
        container_dir if include_container else None,
        uncertainty_dir if include_uncertainty else None,
    ):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    scenario_list = list(scenarios or [{"name": "camerae2e_dataset_case"}])
    label_list = _normalize_labels(labels, len(scenario_list))
    split_list = _normalize_splits(split, len(scenario_list))
    split_policy = _split_policy(split, len(scenario_list), split_list)
    records: list[dict[str, Any]] = []
    metadata_path = root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_stream:
        for index, scenario_config in enumerate(scenario_list):
            case_id = f"case_{index:04d}"
            result = camerae2e_run_scenario(
                scenario_config,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=int(seed) + index,
                include_arrays=True,
            )
            raw_array = _array_from_stage(result, "sensor_raw")
            rgb_array = _array_from_stage(result, "ip_srgb")
            if rgb_array.size == 0:
                rgb_array = _array_from_stage(result, "ip_result")

            raw_path = raw_dir / f"{case_id}.npz"
            sensor_digital_array = _array_from_stage(result, "sensor_digital")
            _write_npz_deterministic(
                raw_path,
                raw=raw_array,
                sensor_digital=sensor_digital_array,
                seed=np.asarray([int(seed) + index], dtype=np.int64),
            )

            rgb_path = None
            if include_rgb and rgb_array.size:
                rgb_path = rgb_dir / f"{case_id}.png"
                iio.imwrite(rgb_path, _uint8_preview(rgb_array))

            tiff_path = None
            if include_tiff and raw_array.size:
                tiff_path = tiff_dir / f"{case_id}.tiff"
                tifffile.imwrite(tiff_path, raw_array.astype(np.float32), photometric="minisblack")

            stage_outputs_path = None
            if include_stage_outputs:
                stage_outputs_path = stage_dir / f"{case_id}.npz"
                _write_stage_outputs(stage_outputs_path, result)

            uncertainty_path = None
            if include_uncertainty:
                uncertainty_path = uncertainty_dir / f"{case_id}.npz"
                _write_uncertainty_outputs(
                    uncertainty_path,
                    raw_array,
                    sensor_digital_array,
                )

            label_payload = {
                "schema_version": "camerae2e_dataset_labels_v1",
                "case_id": case_id,
                "labels": label_list[index],
                "note": (
                    "Labels are caller-provided or empty; "
                    "CameraE2E does not infer labels in v1."
                ),
            }
            label_path = label_dir / f"{case_id}.json"
            label_path.write_text(
                json.dumps(_jsonable(label_payload), indent=2, sort_keys=True), encoding="utf-8"
            )

            report = camerae2e_faca_report(result)
            record = {
                "case_id": case_id,
                "split": split_list[index],
                "seed": int(seed) + index,
                "scenario_name": report["name"],
                "raw": str(raw_path),
                "rgb": None if rgb_path is None else str(rgb_path),
                "raw_tiff": None if tiff_path is None else str(tiff_path),
                "stage_outputs": None if stage_outputs_path is None else str(stage_outputs_path),
                "container": None,
                "uncertainty": None if uncertainty_path is None else str(uncertainty_path),
                "labels": str(label_path),
                "raw_shape": list(raw_array.shape),
                "raw_dtype": str(raw_array.dtype),
                "rgb_shape": list(rgb_array.shape),
                "rgb_dtype": str(rgb_array.dtype) if rgb_array.size else None,
                "sensor_digital_shape": list(sensor_digital_array.shape),
                "sensor_digital_dtype": str(sensor_digital_array.dtype),
                "raw_sha256": _sha256_file(raw_path),
                "raw_content_sha256": _sha256_array(raw_array),
                "sensor_digital_content_sha256": _sha256_array(sensor_digital_array),
                "rgb_sha256": None if rgb_path is None else _sha256_file(rgb_path),
                "raw_tiff_sha256": None if tiff_path is None else _sha256_file(tiff_path),
                "stage_outputs_sha256": (
                    None if stage_outputs_path is None else _sha256_file(stage_outputs_path)
                ),
                "container_sha256": None,
                "uncertainty_sha256": (
                    None if uncertainty_path is None else _sha256_file(uncertainty_path)
                ),
                "labels_sha256": _sha256_file(label_path),
                "scenario": report.get("scenario", {}),
                "parameter_lineage": report.get("parameter_lineage", []),
                "stage_summaries": report.get("stage_summaries", {}),
                "faca_metrics": report.get("metrics", {}),
                "label_summary": _label_summary(label_list[index]),
                "uncertainty_model": (
                    None if uncertainty_path is None else _uncertainty_model_manifest()
                ),
            }
            if include_container:
                container_path = container_dir / f"{case_id}.zip"
                _write_case_container(
                    container_path,
                    record=record,
                    raw_path=raw_path,
                    label_path=label_path,
                    rgb_path=rgb_path,
                    tiff_path=tiff_path,
                    stage_outputs_path=stage_outputs_path,
                    uncertainty_path=uncertainty_path,
                )
                record["container"] = str(container_path)
                record["container_sha256"] = _sha256_file(container_path)
            records.append(record)
            metadata_stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    metadata_sha256 = _sha256_file(metadata_path)
    split_counts: dict[str, int] = {}
    for name in split_list:
        split_counts[name] = split_counts.get(name, 0) + 1
    manifest = {
        "schema_version": "camerae2e_dataset_manifest_v1",
        "dataset_root": str(root),
        "seed": int(seed),
        "case_count": len(records),
        "splits": split_counts,
        "split_policy": split_policy,
        "format": {
            "raw": "deterministic compressed NumPy .npz with raw and sensor_digital arrays",
            "rgb": "8-bit PNG preview when include_rgb=True",
            "raw_tiff": "float32 TIFF when include_tiff=True",
            "stage_outputs": (
                "deterministic NPZ with selected per-stage arrays when "
                "include_stage_outputs=True"
            ),
            "uncertainty": (
                "deterministic NPZ with proxy uncertainty/confidence maps when "
                "include_uncertainty=True"
            ),
            "labels": "JSON labels supplied by caller; empty by default",
            "container": "deterministic DNG-like ZIP container when include_container=True",
            "dng": "not emitted as standards-compliant DNG in v1",
        },
        "truth_boundary": (
            "RAW factory outputs are research artifacts. The optional container is "
            "DNG-like for packaging only and is not a standards-compliant DNG."
        ),
        "integrity": {
            "hash": "sha256",
            "raw_npz": "deterministic zip container with fixed member timestamps",
            "metadata_jsonl_sha256": metadata_sha256,
        },
        "records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def camerae2e_dataset_export_from_optimization(
    output_dir: str | Path,
    optimization_result: Mapping[str, Any],
    *,
    selection: str = "pareto",
    max_cases: int | None = None,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int | None = None,
    labels: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    include_rgb: bool = True,
    include_tiff: bool = False,
    include_stage_outputs: bool = False,
    include_container: bool = False,
    include_uncertainty: bool = False,
    split: Mapping[str, float] | Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Export RAW data from selected optimization cases.

    This bridges the E2E optimizer and RAW data factory: best/top/Pareto
    camera-parameter candidates can be rendered into training-ready RAW
    artifacts without manually copying scenario dictionaries.
    """

    selected_cases = _optimization_cases(optimization_result, selection=selection)
    if max_cases is not None:
        selected_cases = selected_cases[: max(int(max_cases), 0)]
    scenarios = [
        _scenario_from_optimization_case(case, index)
        for index, case in enumerate(selected_cases)
    ]
    effective_seed = int(seed if seed is not None else optimization_result.get("seed", 0))
    effective_split = split if split is not None else str(selection)
    manifest = camerae2e_dataset_export(
        output_dir,
        scenarios,
        scene=scene,
        camera=camera,
        asset_store=asset_store,
        seed=effective_seed,
        labels=labels,
        include_rgb=include_rgb,
        include_tiff=include_tiff,
        include_stage_outputs=include_stage_outputs,
        include_container=include_container,
        include_uncertainty=include_uncertainty,
        split=effective_split,
    )
    manifest["source_optimization"] = {
        "schema_version": "camerae2e_dataset_source_optimization_v1",
        "selection": str(selection),
        "source_schema_version": optimization_result.get("schema_version"),
        "source_seed": optimization_result.get("seed"),
        "method": optimization_result.get("method"),
        "objective": _jsonable(optimization_result.get("objective", {})),
        "constraints": _jsonable(optimization_result.get("constraints", [])),
        "selected_case_count": len(selected_cases),
        "selected_cases": [_optimization_case_summary(case) for case in selected_cases],
    }
    if optimization_result.get("source_image_sensor_db") is not None:
        manifest["source_optimization"]["source_image_sensor_db"] = _jsonable(
            optimization_result.get("source_image_sensor_db")
        )
    manifest_path = Path(manifest["dataset_root"]) / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def camerae2e_dataset_validate(
    dataset: str | Path | Mapping[str, Any], *, strict: bool = False
) -> dict[str, Any]:
    """Validate a CameraE2E RAW dataset manifest and referenced artifacts."""

    manifest, manifest_path, root = _load_manifest(dataset)
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    records = list(manifest.get("records", []))
    if manifest.get("schema_version") != "camerae2e_dataset_manifest_v1":
        issues.append(
            {
                "kind": "schema_version",
                "message": "Dataset manifest schema_version is not camerae2e_dataset_manifest_v1.",
            }
        )
    if int(manifest.get("case_count", -1)) != len(records):
        issues.append(
            {
                "kind": "case_count",
                "message": "Dataset manifest case_count does not match records length.",
            }
        )

    metadata_path = root / "metadata.jsonl"
    metadata_rows = _load_metadata_rows(metadata_path, issues)
    metadata_by_case = {str(item.get("case_id")): item for item in metadata_rows}
    if len(metadata_rows) != len(records):
        issues.append(
            {
                "kind": "metadata_count",
                "message": "metadata.jsonl row count does not match records length.",
            }
        )

    for index, record in enumerate(records):
        _validate_record(record, index, root, metadata_by_case, issues, warnings)

    split_counts: dict[str, int] = {}
    for record in records:
        split_name = str(record.get("split", "unspecified"))
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
    if dict(manifest.get("splits", {})) != split_counts:
        issues.append(
            {
                "kind": "split_counts",
                "message": "Manifest split counts do not match record split assignments.",
            }
        )

    if strict and warnings:
        issues.extend({**warning, "strict_promoted": True} for warning in warnings)
    return {
        "schema_version": "camerae2e_dataset_validation_v1",
        "ok": not issues,
        "strict": bool(strict),
        "dataset_root": str(root),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "case_count": len(records),
        "split_counts": split_counts,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
    }


def camerae2e_dataset_export_perception_index(
    dataset: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    formats: Iterable[str] | str = ("raw_manifest", "yolo"),
    class_map: Mapping[str, int] | None = None,
    copy_images: bool = False,
) -> dict[str, Any]:
    """Export perception-training indexes from a CameraE2E RAW dataset.

    ``raw_manifest`` keeps RAW NPZ paths, labels, hashes, and FACA metadata for
    custom RAW-aware training pipelines.  ``yolo`` exports a YOLO-style RGB
    preview label view; it is an adapter for detector training and does not mean
    YOLO consumes RAW NPZ directly.
    """

    manifest, _, root = _load_manifest(dataset)
    formats_list = _normalize_index_formats(formats)
    out_root = Path(output_dir).expanduser().resolve() if output_dir else root / "perception_index"
    out_root.mkdir(parents=True, exist_ok=True)
    records = list(manifest.get("records", []))
    resolved_class_map = _dataset_class_map(manifest, root, class_map)
    warnings: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}

    if "raw_manifest" in formats_list:
        outputs["raw_manifest"] = _write_raw_training_index(
            records,
            root,
            out_root,
            resolved_class_map,
            warnings,
        )
    if "yolo" in formats_list:
        outputs["yolo"] = _write_yolo_training_view(
            records,
            root,
            out_root,
            resolved_class_map,
            warnings,
            copy_images=copy_images,
        )

    payload = {
        "schema_version": "camerae2e_perception_training_index_v1",
        "dataset_root": str(root),
        "output_dir": str(out_root),
        "formats": formats_list,
        "case_count": len(records),
        "class_map": resolved_class_map,
        "truth_boundary": (
            "raw_manifest indexes RAW NPZ for RAW-aware training code; "
            "YOLO view uses RGB previews and JSON labels converted to YOLO txt."
        ),
        "outputs": outputs,
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    manifest_path = out_root / "perception_index_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["manifest"] = str(manifest_path)
    return payload


def camerae2e_adas_camera_spec(preset: str = "kitti_yolo_demo") -> dict[str, Any]:
    """Return an ADAS camera preset for RAW demo generation.

    The default preset is tied to KITTI-style geometry, not to measured KITTI
    sensor calibration.  It is intended for reproducible ADAS/YOLO RAW factory
    demos and keeps the truth boundary explicit in the returned metadata.
    """

    key = str(preset).strip().lower()
    if key not in {
        "kitti_yolo_demo",
        "kitti_adas_demo",
        "wide_fov_adas_demo",
        "narrow_fov_adas_demo",
    }:
        raise ValueError(
            "ADAS camera spec preset must be one of: "
            "kitti_yolo_demo, wide_fov_adas_demo, narrow_fov_adas_demo."
        )
    image_size = [375, 1242]
    presets = {
        "kitti_yolo_demo": {
            "focal_px": 721.5377,
            "pixel_pitch_m": 3.75e-6,
            "f_number": 1.8,
            "integration_time_s": 0.004,
        },
        "kitti_adas_demo": {
            "focal_px": 721.5377,
            "pixel_pitch_m": 3.75e-6,
            "f_number": 1.8,
            "integration_time_s": 0.004,
        },
        "wide_fov_adas_demo": {
            "focal_px": 520.0,
            "pixel_pitch_m": 3.0e-6,
            "f_number": 2.0,
            "integration_time_s": 0.003,
        },
        "narrow_fov_adas_demo": {
            "focal_px": 1050.0,
            "pixel_pitch_m": 2.8e-6,
            "f_number": 2.4,
            "integration_time_s": 0.006,
        },
    }
    preset_values = presets[key]
    focal_px = float(preset_values["focal_px"])
    pixel_pitch_m = float(preset_values["pixel_pitch_m"])
    focal_length_m = focal_px * pixel_pitch_m
    hfov_deg = float(2.0 * np.rad2deg(np.arctan2(image_size[1] / 2.0, focal_px)))
    vfov_deg = float(2.0 * np.rad2deg(np.arctan2(image_size[0] / 2.0, focal_px)))
    return {
        "schema_version": "camerae2e_adas_camera_spec_v1",
        "preset": key,
        "readiness_tier": "proxy",
        "truth_boundary": (
            "KITTI camera geometry plus public-style ADAS assumptions; "
            "not measured sensor, lens, ISP, or KITTI raw calibration."
        ),
        "dataset_reference": {
            "name": "KITTI object-detection style",
            "label_format": "KITTI text plus YOLO normalized xywh labels",
            "native_image_size_rc": image_size,
            "p2_focal_px": focal_px,
            "principal_point_px": [609.5593, 172.8540],
        },
        "optics": {
            "focal_length_m": focal_length_m,
            "f_number": float(preset_values["f_number"]),
            "hfov_deg": hfov_deg,
            "vfov_deg": vfov_deg,
        },
        "sensor": {
            "pixel_pitch_m": pixel_pitch_m,
            "cfa_pattern": "bayer_rggb",
            "bits": 12,
            "integration_time_s": float(preset_values["integration_time_s"]),
            "analog_gain": 1.0,
            "noise_flag": 2,
        },
        "hw_isp": {
            "enabled": True,
            "nframes": 3,
            "fps": 30.0,
            "line_time_us": 15.2,
            "ae_apply_delay_frames": 2,
            "awb_apply_delay_frames": 2,
        },
        "demo": {
            "image_size_rc": [96, 320],
            "mean_luminance_cd_m2": 80.0,
            "scene_distance_m": 30.0,
        },
    }


def camerae2e_kitti_yolo_labels(
    source: str | Path | Iterable[str] | None = None,
    *,
    image_size: tuple[int, int] | list[int],
    class_ids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return caller-supplied or demo KITTI labels with YOLO-normalized boxes."""

    rows, cols = _image_size_rc(image_size)
    mapping = dict(_KITTI_CLASS_IDS if class_ids is None else class_ids)
    if source is None:
        lines = _synthetic_kitti_label_lines(rows, cols)
        source_name = "synthetic_kitti_style"
    elif isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        lines = path.read_text(encoding="utf-8").splitlines()
        source_name = str(path)
    else:
        lines = [str(line) for line in source]
        source_name = "inline_kitti_labels"
    objects = [
        _kitti_label_to_object(line, rows=rows, cols=cols, class_ids=mapping)
        for line in lines
        if line.strip()
    ]
    return {
        "schema_version": "camerae2e_kitti_yolo_labels_v1",
        "source": source_name,
        "image_size_rc": [rows, cols],
        "class_map": mapping,
        "objects": objects,
        "masks": [],
    }


def camerae2e_dataset_export_adas_kitti_demo(
    output_dir: str | Path,
    *,
    image_paths: Iterable[str | Path] | None = None,
    label_paths: Iterable[str | Path] | None = None,
    case_count: int = 2,
    spec: Mapping[str, Any] | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_rgb: bool = True,
    include_tiff: bool = False,
    include_stage_outputs: bool = False,
    include_container: bool = False,
    include_uncertainty: bool = False,
    split: Mapping[str, float] | Iterable[str] | str | None = "demo",
) -> dict[str, Any]:
    """Generate a KITTI/YOLO-style ADAS RAW dataset demo.

    If ``image_paths`` and optional ``label_paths`` are supplied, those KITTI
    RGB frames and labels are used.  Otherwise the function creates a small,
    deterministic KITTI-style road scene with YOLO/KITTI boxes so the RAW
    factory can be exercised without bundling the KITTI dataset.
    """

    store = asset_store or AssetStore.default()
    camera_spec = _jsonable(spec or camerae2e_adas_camera_spec())
    demo_shape = _image_size_rc(camera_spec.get("demo", {}).get("image_size_rc", [96, 320]))
    image_list = [] if image_paths is None else [Path(path).expanduser() for path in image_paths]
    label_list = [] if label_paths is None else [Path(path).expanduser() for path in label_paths]
    if label_list and len(label_list) != len(image_list):
        raise ValueError("label_paths must have the same length as image_paths.")

    scenarios: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    if image_list:
        selected_images = image_list[: max(int(case_count), 0)]
        for index, image_path in enumerate(selected_images):
            image = np.asarray(iio.imread(image_path))
            if image.ndim < 2:
                raise ValueError(f"KITTI demo image must be 2D or 3D: {image_path}")
            rows, cols = int(image.shape[0]), int(image.shape[1])
            label_source = label_list[index] if label_list else None
            scene = _adas_scene_from_rgb(
                image,
                camera_spec,
                store,
                f"kitti_frame_{index:04d}",
            )
            scenarios.append(_adas_scenario(scene, camera_spec, index))
            labels.append(camerae2e_kitti_yolo_labels(label_source, image_size=(rows, cols)))
    else:
        for index in range(max(int(case_count), 0)):
            rgb, label_payload = _synthetic_adas_rgb_and_labels(
                demo_shape[0], demo_shape[1], index=index
            )
            scene = _adas_scene_from_rgb(rgb, camera_spec, store, f"synthetic_kitti_{index:04d}")
            scenarios.append(_adas_scenario(scene, camera_spec, index))
            labels.append(label_payload)

    manifest = camerae2e_dataset_export(
        output_dir,
        scenarios,
        asset_store=store,
        seed=seed,
        labels=labels,
        include_rgb=include_rgb,
        include_tiff=include_tiff,
        include_stage_outputs=include_stage_outputs,
        include_container=include_container,
        include_uncertainty=include_uncertainty,
        split=split,
    )
    manifest["adas_kitti_demo"] = {
        "schema_version": "camerae2e_adas_kitti_demo_v1",
        "camera_spec": camera_spec,
        "case_count": len(scenarios),
        "input_mode": "synthetic" if not image_list else "kitti_files",
        "note": (
            "This demo applies KITTI-style ADAS geometry and YOLO/KITTI labels "
            "to the CameraE2E RAW factory. It is not KITTI raw sensor ground truth."
        ),
    }
    _rescale_manifest_labels_to_raw(manifest)
    _rewrite_dataset_metadata_and_manifest(manifest)
    return manifest


def camerae2e_dataset_export_camera_spec_variants(
    output_dir: str | Path,
    *,
    image_paths: Iterable[str | Path] | None = None,
    label_paths: Iterable[str | Path] | None = None,
    source_spec: Mapping[str, Any] | None = None,
    target_specs: Iterable[Mapping[str, Any] | str] | None = None,
    geometric_transform: str = "pinhole_crop",
    out_of_fov_fill: str = "edge",
    case_count: int = 1,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_rgb: bool = True,
    include_tiff: bool = False,
    include_stage_outputs: bool = False,
    include_container: bool = False,
    include_uncertainty: bool = False,
    split: str = "camera_variant",
) -> dict[str, Any]:
    """Export proxy re-captures of one KITTI-style scene under target cameras.

    This is a controlled transformation, not an inverse-physics reconstruction.
    The source RGB frame is interpreted as a display-derived proxy scene, then
    re-rendered through each target camera spec.  By default the source RGB is
    first transformed with a focal-ratio pinhole crop/resize approximation so
    FoV changes are visible in the image geometry and labels.
    """

    targets = list(
        target_specs
        or ["kitti_yolo_demo", "wide_fov_adas_demo", "narrow_fov_adas_demo"]
    )
    target_payloads = [
        camerae2e_adas_camera_spec(item) if isinstance(item, str) else _jsonable(item)
        for item in targets
    ]
    transform_key = str(geometric_transform).strip().lower()
    if transform_key not in {"pinhole_crop", "none"}:
        raise ValueError("geometric_transform must be one of: pinhole_crop, none.")
    source_payload = _jsonable(source_spec or camerae2e_adas_camera_spec())
    store = asset_store or AssetStore.default()
    root = Path(output_dir).expanduser().resolve()
    image_list = [] if image_paths is None else [Path(path).expanduser() for path in image_paths]
    label_list = [] if label_paths is None else [Path(path).expanduser() for path in label_paths]
    if label_list and len(label_list) != len(image_list):
        raise ValueError("label_paths must have the same length as image_paths.")

    scenarios: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_descriptions: list[dict[str, Any]] = []
    if image_list:
        selected_images = image_list[: max(int(case_count), 0)]
        source_frames = []
        for index, image_path in enumerate(selected_images):
            image = np.asarray(iio.imread(image_path))
            rows, cols = int(image.shape[0]), int(image.shape[1])
            label_source = label_list[index] if label_list else None
            source_frames.append(
                (
                    image,
                    camerae2e_kitti_yolo_labels(label_source, image_size=(rows, cols)),
                    f"kitti_file_{index:04d}",
                    str(image_path),
                )
            )
    else:
        demo_shape = _image_size_rc(source_payload.get("demo", {}).get("image_size_rc", [96, 320]))
        source_frames = []
        for index in range(max(int(case_count), 0)):
            image, source_labels = _synthetic_adas_rgb_and_labels(
                demo_shape[0], demo_shape[1], index=index
            )
            source_frames.append(
                (image, source_labels, f"synthetic_kitti_{index:04d}", "synthetic")
            )

    for frame_index, (image, source_labels, frame_name, source_path) in enumerate(source_frames):
        source_descriptions.append(
            {
                "frame_index": frame_index,
                "source": source_path,
                "source_label_frame": source_labels.get("image_size_rc"),
            }
        )
        for target_index, target_spec in enumerate(target_payloads):
            target_image = image
            target_labels = _rescale_label_payload(
                source_labels,
                _image_size_rc(target_spec["dataset_reference"]["native_image_size_rc"]),
                coordinate_frame="target_camera_spec_native_image",
            )
            transform_metadata: dict[str, Any] = {
                "mode": "none",
                "truth_boundary": "no RGB geometric transformation was applied",
            }
            if transform_key == "pinhole_crop":
                target_image, target_labels, transform_metadata = (
                    _camera_spec_geometric_transform(
                        image,
                        source_labels,
                        source_payload,
                        target_spec,
                        out_of_fov_fill=out_of_fov_fill,
                    )
                )
            scene = _adas_scene_from_rgb(
                target_image,
                target_spec,
                store,
                f"{frame_name}_{target_spec.get('preset', target_index)}",
            )
            scenario = _adas_scenario(scene, target_spec, len(scenarios))
            scenario["source_camera_spec"] = source_payload
            scenario["target_camera_spec"] = target_spec
            scenario["geometric_scene_transform"] = transform_metadata
            scenarios.append(scenario)
            labels.append(target_labels)

    manifest = camerae2e_dataset_export(
        root,
        scenarios,
        asset_store=store,
        seed=seed,
        labels=labels,
        include_rgb=include_rgb,
        include_tiff=include_tiff,
        include_stage_outputs=include_stage_outputs,
        include_container=include_container,
        include_uncertainty=include_uncertainty,
        split=split,
    )
    manifest["camera_spec_variants"] = {
        "schema_version": "camerae2e_camera_spec_variants_v1",
        "source_spec": source_payload,
        "target_specs": target_payloads,
        "source_frames": source_descriptions,
        "geometric_transform": {
            "mode": transform_key,
            "out_of_fov_fill": str(out_of_fov_fill),
            "approximation": (
                "2D pinhole focal-ratio crop/resize on the RGB image; "
                "no depth, disocclusion, parallax, or spectral reconstruction."
            ),
        },
        "truth_boundary": (
            "RGB-to-scene proxy re-capture. This does not recover true KITTI "
            "spectral radiance, depth, occlusion, lens flare, ISP inverse, or measured raw."
        ),
    }
    _rescale_manifest_labels_to_raw(manifest)
    _rewrite_dataset_metadata_and_manifest(manifest)
    return manifest


def _optimization_cases(
    optimization_result: Mapping[str, Any], *, selection: str
) -> list[Mapping[str, Any]]:
    key = str(selection).strip().lower()
    if key == "best":
        best = optimization_result.get("best_case")
        return [] if best is None else [dict(best)]
    if key == "top":
        return [dict(item) for item in optimization_result.get("top_cases", [])]
    if key == "pareto":
        return [dict(item) for item in optimization_result.get("pareto_front", [])]
    if key == "selected":
        return [
            {"case_index": index, "scenario": scenario}
            for index, scenario in enumerate(optimization_result.get("selected_scenarios", []))
        ]
    if key == "all":
        return [
            dict(item)
            for item in optimization_result.get("cases", [])
            if item.get("feasible", True)
        ]
    raise ValueError("selection must be one of: best, top, pareto, selected, all.")


def _scenario_from_optimization_case(case: Mapping[str, Any], index: int) -> dict[str, Any]:
    scenario = _deep_dict(dict(case.get("scenario", {})))
    if not scenario:
        scenario = {"parameters": dict(case.get("parameters", {}))}
    base_name = str(scenario.get("name", "camerae2e_optimized_dataset_case"))
    case_index = int(case.get("case_index", index))
    scenario["name"] = f"{base_name}_opt{case_index:04d}"
    scenario.setdefault("optimization_case", _optimization_case_summary(case))
    return scenario


def _optimization_case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "parameters": _jsonable(case.get("parameters", {})),
        "score": case.get("score"),
        "feasible": case.get("feasible", True),
        "objective_values": _jsonable(case.get("objective_values", {})),
        "objective_utilities": _jsonable(case.get("objective_utilities", {})),
        "constraint_results": _jsonable(case.get("constraint_results", [])),
    }


def _normalize_labels(
    labels: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None, count: int
) -> list[dict[str, Any]]:
    empty = {"objects": [], "masks": [], "source": "not_provided"}
    if labels is None:
        return [dict(empty) for _ in range(count)]
    if isinstance(labels, Mapping):
        return [dict(labels) for _ in range(count)]
    values = [dict(item) for item in labels]
    if len(values) != count:
        raise ValueError("labels must contain one item per scenario when provided as an iterable.")
    return values


def _label_summary(labels: Mapping[str, Any]) -> dict[str, Any]:
    objects = list(labels.get("objects", [])) if isinstance(labels.get("objects", []), list) else []
    masks = list(labels.get("masks", [])) if isinstance(labels.get("masks", []), list) else []
    semantic_masks = (
        list(labels.get("semantic_masks", []))
        if isinstance(labels.get("semantic_masks", []), list)
        else []
    )
    instance_masks = (
        list(labels.get("instance_masks", []))
        if isinstance(labels.get("instance_masks", []), list)
        else []
    )
    keypoints = (
        list(labels.get("keypoints", [])) if isinstance(labels.get("keypoints", []), list) else []
    )
    polylines = (
        list(labels.get("polylines", [])) if isinstance(labels.get("polylines", []), list) else []
    )
    stage_tags = (
        list(labels.get("stage_tags", [])) if isinstance(labels.get("stage_tags", []), list) else []
    )
    return {
        "object_count": len(objects),
        "mask_count": len(masks),
        "semantic_mask_count": len(semantic_masks),
        "instance_mask_count": len(instance_masks),
        "keypoint_count": len(keypoints),
        "polyline_count": len(polylines),
        "stage_tag_count": len(stage_tags),
        "supported_label_types": [
            "objects",
            "masks",
            "semantic_masks",
            "instance_masks",
            "keypoints",
            "polylines",
            "stage_tags",
        ],
        "truth_boundary": (
            "Labels are caller-provided or synthetic-helper-provided; CameraE2E "
            "does not infer semantic truth from arbitrary images."
        ),
    }


def _adas_scene_from_rgb(
    rgb: Any, spec: Mapping[str, Any], asset_store: AssetStore, name: str
) -> Scene:
    scene = scene_from_file(
        np.asarray(rgb),
        "rgb",
        float(spec.get("demo", {}).get("mean_luminance_cd_m2", 80.0)),
        "lcdExample.mat",
        asset_store=asset_store,
    )
    scene.name = str(name)
    scene = scene_set(scene, "fov", float(spec.get("optics", {}).get("hfov_deg", 81.4)))
    scene = scene_set(scene, "distance", float(spec.get("demo", {}).get("scene_distance_m", 30.0)))
    return scene


def _adas_scenario(scene: Scene, spec: Mapping[str, Any], index: int) -> dict[str, Any]:
    sensor = dict(spec.get("sensor", {}))
    optics = dict(spec.get("optics", {}))
    hw_isp = dict(spec.get("hw_isp", {}))
    return {
        "name": f"adas_kitti_yolo_demo_{index:04d}",
        "scene": scene,
        "sensor": {
            "noise_flag": sensor.get("noise_flag", 2),
            "integration_time": sensor.get("integration_time_s", 0.004),
            "analog_gain": sensor.get("analog_gain", 1.0),
        },
        "parameters": {
            "pixel.size": [
                sensor.get("pixel_pitch_m", 3.75e-6),
                sensor.get("pixel_pitch_m", 3.75e-6),
            ],
            "sensor.bits": sensor.get("bits", 12),
            "optics.focal_length": optics.get("focal_length_m", 0.0027),
            "optics.fnumber": optics.get("f_number", 1.8),
        },
        "hw_isp": {
            "enabled": bool(hw_isp.get("enabled", True)),
            "nframes": int(hw_isp.get("nframes", 3)),
            "fps": float(hw_isp.get("fps", 30.0)),
            "line_time_us": float(hw_isp.get("line_time_us", 15.2)),
            "ae_apply_delay_frames": int(hw_isp.get("ae_apply_delay_frames", 2)),
            "awb_apply_delay_frames": int(hw_isp.get("awb_apply_delay_frames", 2)),
        },
        "adas_camera_spec": _jsonable(spec),
    }


def _camera_spec_geometric_transform(
    rgb: Any,
    labels: Mapping[str, Any],
    source_spec: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    *,
    out_of_fov_fill: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    source = np.asarray(rgb)
    if source.ndim < 2:
        raise ValueError("camera-spec geometric transform expects a 2D or 3D RGB image.")
    rows, cols = int(source.shape[0]), int(source.shape[1])
    source_geometry = _effective_pinhole_geometry(source_spec, (rows, cols))
    target_geometry = _effective_pinhole_geometry(target_spec, (rows, cols))
    transformed, warp_metadata = _warp_rgb_between_pinhole_specs(
        source,
        source_geometry,
        target_geometry,
        fill_mode=out_of_fov_fill,
    )
    source_labels = _rescale_label_payload(
        labels,
        (rows, cols),
        coordinate_frame="source_image_for_pinhole_crop",
    )
    target_labels = _transform_label_payload_camera_geometry(
        source_labels,
        source_geometry,
        target_geometry,
        target_size_rc=(rows, cols),
    )
    transform_metadata = {
        "mode": "pinhole_crop",
        "source_preset": source_spec.get("preset"),
        "target_preset": target_spec.get("preset"),
        "source_geometry": source_geometry,
        "target_geometry": target_geometry,
        "object_scale_x": target_geometry["fx_px"] / max(source_geometry["fx_px"], 1.0e-12),
        "object_scale_y": target_geometry["fy_px"] / max(source_geometry["fy_px"], 1.0e-12),
        "warp": warp_metadata,
        "truth_boundary": (
            "2D pinhole focal-ratio image crop/resize only; no depth, parallax, "
            "disocclusion, rolling-shutter pose change, or spectral reconstruction."
        ),
    }
    target_labels["geometric_transform"] = transform_metadata
    return transformed, target_labels, transform_metadata


def _effective_pinhole_geometry(
    spec: Mapping[str, Any], image_size_rc: tuple[int, int]
) -> dict[str, Any]:
    rows, cols = image_size_rc
    reference = dict(spec.get("dataset_reference", {}))
    native_rows, native_cols = _image_size_rc(
        reference.get("native_image_size_rc", [rows, cols])
    )
    focal_px = _camera_spec_reference_focal_px(spec, native_cols)
    principal = reference.get(
        "principal_point_px",
        [(native_cols - 1.0) / 2.0, (native_rows - 1.0) / 2.0],
    )
    principal_values = np.asarray(principal, dtype=float).reshape(-1)
    if principal_values.size < 2:
        principal_values = np.asarray(
            [(native_cols - 1.0) / 2.0, (native_rows - 1.0) / 2.0],
            dtype=float,
        )
    x_scale = cols / max(native_cols, 1)
    y_scale = rows / max(native_rows, 1)
    return {
        "image_size_rc": [rows, cols],
        "native_image_size_rc": [native_rows, native_cols],
        "fx_px": float(focal_px * x_scale),
        "fy_px": float(focal_px * y_scale),
        "cx_px": float(principal_values[0] * x_scale),
        "cy_px": float(principal_values[1] * y_scale),
        "reference_focal_px": float(focal_px),
    }


def _camera_spec_reference_focal_px(spec: Mapping[str, Any], native_cols: int) -> float:
    reference = dict(spec.get("dataset_reference", {}))
    if reference.get("p2_focal_px") is not None:
        return float(reference["p2_focal_px"])
    hfov = dict(spec.get("optics", {})).get("hfov_deg")
    if hfov is not None:
        return float((native_cols / 2.0) / np.tan(np.deg2rad(float(hfov)) / 2.0))
    focal_length = dict(spec.get("optics", {})).get("focal_length_m")
    pixel_pitch = dict(spec.get("sensor", {})).get("pixel_pitch_m")
    if focal_length is not None and pixel_pitch is not None:
        return float(focal_length) / max(float(pixel_pitch), 1.0e-12)
    raise ValueError("Camera spec must provide p2_focal_px, hfov_deg, or focal_length/pitch.")


def _warp_rgb_between_pinhole_specs(
    image: np.ndarray,
    source_geometry: Mapping[str, Any],
    target_geometry: Mapping[str, Any],
    *,
    fill_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    key = str(fill_mode).strip().lower()
    if key not in {"edge", "black", "constant"}:
        raise ValueError("out_of_fov_fill must be one of: edge, black, constant.")
    rows, cols = int(image.shape[0]), int(image.shape[1])
    yy, xx = np.indices((rows, cols), dtype=float)
    source_x = float(source_geometry["cx_px"]) + (
        float(source_geometry["fx_px"]) / max(float(target_geometry["fx_px"]), 1.0e-12)
    ) * (xx - float(target_geometry["cx_px"]))
    source_y = float(source_geometry["cy_px"]) + (
        float(source_geometry["fy_px"]) / max(float(target_geometry["fy_px"]), 1.0e-12)
    ) * (yy - float(target_geometry["cy_px"]))
    out_of_source = (
        (source_x < 0.0)
        | (source_x > cols - 1.0)
        | (source_y < 0.0)
        | (source_y > rows - 1.0)
    )
    sampled = _bilinear_sample_image(image, source_x, source_y, out_of_source, fill_mode=key)
    sample_rect = [
        float(np.min(source_x)),
        float(np.min(source_y)),
        float(np.max(source_x)),
        float(np.max(source_y)),
    ]
    return sampled, {
        "source_sample_rect_xyxy": sample_rect,
        "out_of_source_fraction": float(np.mean(out_of_source)),
        "fill_mode": key,
    }


def _bilinear_sample_image(
    image: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    out_of_source: np.ndarray,
    *,
    fill_mode: str,
) -> np.ndarray:
    source = np.asarray(image)
    original_ndim = source.ndim
    if original_ndim == 2:
        source = source[:, :, None]
    rows, cols, channels = source.shape[:3]
    x = np.clip(source_x, 0.0, cols - 1.0)
    y = np.clip(source_y, 0.0, rows - 1.0)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    values = source.astype(float, copy=False)
    top = values[y0, x0, :channels] * (1.0 - dx) + values[y0, x1, :channels] * dx
    bottom = values[y1, x0, :channels] * (1.0 - dx) + values[y1, x1, :channels] * dx
    sampled = top * (1.0 - dy) + bottom * dy
    if fill_mode in {"black", "constant"}:
        sampled[out_of_source, :] = 0.0
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        sampled = np.clip(np.round(sampled), info.min, info.max).astype(image.dtype)
    else:
        sampled = sampled.astype(image.dtype, copy=False)
    if original_ndim == 2:
        return sampled[:, :, 0]
    return sampled


def _transform_label_payload_camera_geometry(
    labels: Mapping[str, Any],
    source_geometry: Mapping[str, Any],
    target_geometry: Mapping[str, Any],
    *,
    target_size_rc: tuple[int, int],
) -> dict[str, Any]:
    payload = _jsonable(labels)
    target_rows, target_cols = target_size_rc
    sx = float(target_geometry["fx_px"]) / max(float(source_geometry["fx_px"]), 1.0e-12)
    sy = float(target_geometry["fy_px"]) / max(float(source_geometry["fy_px"]), 1.0e-12)
    source_cx = float(source_geometry["cx_px"])
    source_cy = float(source_geometry["cy_px"])
    target_cx = float(target_geometry["cx_px"])
    target_cy = float(target_geometry["cy_px"])
    objects = []
    for item in payload.get("objects", []):
        obj = dict(item)
        bbox = obj.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            objects.append(obj)
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        transformed = [
            target_cx + sx * (x1 - source_cx),
            target_cy + sy * (y1 - source_cy),
            target_cx + sx * (x2 - source_cx),
            target_cy + sy * (y2 - source_cy),
        ]
        clipped = [
            min(max(transformed[0], 0.0), float(target_cols)),
            min(max(transformed[1], 0.0), float(target_rows)),
            min(max(transformed[2], 0.0), float(target_cols)),
            min(max(transformed[3], 0.0), float(target_rows)),
        ]
        original_area = max(transformed[2] - transformed[0], 0.0) * max(
            transformed[3] - transformed[1],
            0.0,
        )
        clipped_area = max(clipped[2] - clipped[0], 0.0) * max(clipped[3] - clipped[1], 0.0)
        visibility = 0.0 if original_area <= 0.0 else clipped_area / original_area
        if visibility <= 0.0:
            continue
        width = max(clipped[2] - clipped[0], 0.0)
        height = max(clipped[3] - clipped[1], 0.0)
        obj["bbox_xyxy_unclipped"] = transformed
        obj["bbox_xyxy"] = clipped
        obj["visibility_fraction"] = float(visibility)
        obj["yolo_xywhn"] = [
            (clipped[0] + width / 2.0) / max(target_cols, 1),
            (clipped[1] + height / 2.0) / max(target_rows, 1),
            width / max(target_cols, 1),
            height / max(target_rows, 1),
        ]
        objects.append(obj)
    payload["objects"] = objects
    payload["image_size_rc"] = [target_rows, target_cols]
    payload["coordinate_frame"] = "target_camera_spec_pinhole_crop"
    payload["source_coordinate_frame"] = {
        "image_size_rc": labels.get("image_size_rc", source_geometry.get("image_size_rc")),
        "coordinate_frame": labels.get("coordinate_frame", labels.get("source", "source_image")),
    }
    payload["pinhole_label_transform"] = {
        "scale_x": sx,
        "scale_y": sy,
        "source_principal_point_xy": [source_cx, source_cy],
        "target_principal_point_xy": [target_cx, target_cy],
    }
    return payload


def _synthetic_adas_rgb_and_labels(
    rows: int, cols: int, *, index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    rr, cc = np.indices((rows, cols))
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    horizon = int(round(rows * 0.42))
    sky_grad = np.clip(0.65 + 0.25 * (1.0 - rr / max(horizon, 1)), 0.0, 1.0)
    road_grad = np.clip(0.18 + 0.25 * (rr - horizon) / max(rows - horizon, 1), 0.0, 1.0)
    sky_mask = rr < horizon
    rgb[..., 0] = np.where(sky_mask, 95 * sky_grad, 90 * road_grad).astype(np.uint8)
    rgb[..., 1] = np.where(sky_mask, 145 * sky_grad, 95 * road_grad).astype(np.uint8)
    rgb[..., 2] = np.where(sky_mask, 210 * sky_grad, 85 * road_grad).astype(np.uint8)

    lane_center = cols / 2.0 + (index - 0.5) * cols * 0.03
    for offset in (-cols * 0.14, cols * 0.14):
        x = (lane_center + offset + (rr - horizon) * offset / max(rows - horizon, 1)).astype(int)
        mask = (rr > horizon) & (np.abs(cc - x) <= 1)
        rgb[mask] = np.array([230, 230, 210], dtype=np.uint8)

    boxes = [
        ("Car", [0.42 * cols, 0.55 * rows, 0.66 * cols, 0.82 * rows], [1.55, 1.70, 4.20]),
        ("Car", [0.18 * cols, 0.50 * rows, 0.31 * cols, 0.66 * rows], [1.50, 1.65, 3.90]),
        ("Pedestrian", [0.72 * cols, 0.49 * rows, 0.77 * cols, 0.73 * rows], [1.75, 0.60, 0.50]),
    ]
    lines = []
    for label, box, dims_hwl in boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        color = (
            np.array([170, 35, 30], dtype=np.uint8)
            if label == "Car"
            else np.array([40, 120, 45], dtype=np.uint8)
        )
        rgb[max(y1, 0) : min(y2, rows), max(x1, 0) : min(x2, cols)] = color
        h, w, length = dims_hwl
        lines.append(
            f"{label} 0.00 0 0.00 {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} "
            f"{h:.2f} {w:.2f} {length:.2f} 0.00 0.00 20.00 0.00"
        )
    return rgb, camerae2e_kitti_yolo_labels(lines, image_size=(rows, cols))


def _synthetic_kitti_label_lines(rows: int, cols: int) -> list[str]:
    _, labels = _synthetic_adas_rgb_and_labels(rows, cols, index=0)
    return [
        str(item.get("kitti", {}).get("line", ""))
        for item in labels.get("objects", [])
        if item.get("kitti", {}).get("line")
    ]


def _kitti_label_to_object(
    line: str, *, rows: int, cols: int, class_ids: Mapping[str, int]
) -> dict[str, Any]:
    parts = line.strip().split()
    if len(parts) < 8:
        raise ValueError(f"KITTI label line has too few columns: {line!r}")
    label = parts[0]
    bbox = [float(parts[index]) for index in range(4, 8)]
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    yolo = [
        (x1 + width / 2.0) / max(cols, 1),
        (y1 + height / 2.0) / max(rows, 1),
        width / max(cols, 1),
        height / max(rows, 1),
    ]
    return {
        "label": label,
        "class_id": int(class_ids.get(label, -1)),
        "bbox_xyxy": [x1, y1, x2, y2],
        "yolo_xywhn": yolo,
        "kitti": {
            "line": line,
            "truncation": float(parts[1]) if len(parts) > 1 else None,
            "occlusion": int(float(parts[2])) if len(parts) > 2 else None,
            "alpha": float(parts[3]) if len(parts) > 3 else None,
            "dimensions_hwl_m": [float(value) for value in parts[8:11]] if len(parts) >= 11 else [],
            "location_xyz_m": [float(value) for value in parts[11:14]] if len(parts) >= 14 else [],
            "rotation_y": float(parts[14]) if len(parts) >= 15 else None,
        },
    }


def _rescale_manifest_labels_to_raw(manifest: dict[str, Any]) -> None:
    root = Path(str(manifest.get("dataset_root", "."))).expanduser()
    for record in manifest.get("records", []):
        raw_shape = record.get("raw_shape", [])
        if len(raw_shape) < 2:
            continue
        label_path = _resolve_dataset_path(root, record.get("labels"))
        if label_path is None or not label_path.exists():
            continue
        label_json = json.loads(label_path.read_text(encoding="utf-8"))
        label_json["labels"] = _rescale_label_payload(
            label_json.get("labels", {}),
            _image_size_rc(raw_shape[:2]),
            coordinate_frame="exported_raw_shape",
        )
        label_path.write_text(
            json.dumps(_jsonable(label_json), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        record["labels_sha256"] = _sha256_file(label_path)
        record["label_coordinate_frame"] = "exported_raw_shape"


def _rescale_label_payload(
    labels: Mapping[str, Any],
    target_size_rc: tuple[int, int],
    *,
    coordinate_frame: str,
) -> dict[str, Any]:
    payload = _jsonable(labels)
    source_size = _image_size_rc(payload.get("image_size_rc", target_size_rc))
    target_rows, target_cols = target_size_rc
    y_scale = target_rows / max(source_size[0], 1)
    x_scale = target_cols / max(source_size[1], 1)
    objects = []
    for item in payload.get("objects", []):
        obj = dict(item)
        bbox = obj.get("bbox_xyxy")
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            scaled = [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]
            width = max(scaled[2] - scaled[0], 0.0)
            height = max(scaled[3] - scaled[1], 0.0)
            obj["bbox_xyxy"] = scaled
            obj["yolo_xywhn"] = [
                (scaled[0] + width / 2.0) / max(target_cols, 1),
                (scaled[1] + height / 2.0) / max(target_rows, 1),
                width / max(target_cols, 1),
                height / max(target_rows, 1),
            ]
        objects.append(obj)
    payload["objects"] = objects
    payload["image_size_rc"] = [target_rows, target_cols]
    payload["coordinate_frame"] = coordinate_frame
    payload["source_coordinate_frame"] = {
        "image_size_rc": [source_size[0], source_size[1]],
        "coordinate_frame": labels.get("coordinate_frame", labels.get("source", "source_image")),
    }
    return payload


def _rewrite_dataset_metadata_and_manifest(manifest: dict[str, Any]) -> None:
    root = Path(str(manifest.get("dataset_root", "."))).expanduser()
    metadata_path = root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_stream:
        for record in manifest.get("records", []):
            metadata_stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
    manifest.setdefault("integrity", {})["metadata_jsonl_sha256"] = _sha256_file(metadata_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _image_size_rc(image_size: tuple[int, int] | list[int] | Any) -> tuple[int, int]:
    values = np.asarray(image_size, dtype=int).reshape(-1)
    if values.size < 2 or int(values[0]) <= 0 or int(values[1]) <= 0:
        raise ValueError("image_size must contain positive row and column counts.")
    return int(values[0]), int(values[1])


def _normalize_index_formats(formats: Iterable[str] | str) -> list[str]:
    values = [formats] if isinstance(formats, str) else list(formats)
    normalized = []
    valid = {"raw_manifest", "yolo"}
    for item in values:
        key = str(item).strip().lower()
        if key not in valid:
            raise ValueError("perception index format must be one of: raw_manifest, yolo.")
        if key not in normalized:
            normalized.append(key)
    return normalized


def _dataset_class_map(
    manifest: Mapping[str, Any],
    root: Path,
    override: Mapping[str, int] | None,
) -> dict[str, int]:
    if override is not None:
        return {str(key): int(value) for key, value in override.items()}
    class_map: dict[str, int] = {}
    for record in manifest.get("records", []):
        payload = _label_payload_for_record(record, root)
        for item in payload.get("objects", []):
            label = str(item.get("label", "object"))
            class_id = item.get("class_id")
            if class_id is not None and int(class_id) >= 0:
                class_map[label] = int(class_id)
    if class_map:
        return dict(sorted(class_map.items(), key=lambda item: item[1]))
    return {"object": 0}


def _write_raw_training_index(
    records: list[Mapping[str, Any]],
    root: Path,
    out_root: Path,
    class_map: Mapping[str, int],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    path = out_root / "raw_manifest.jsonl"
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[str]] = {}
    row_count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            case_id = str(record.get("case_id", f"case_{row_count:04d}"))
            split = str(record.get("split", "unspecified"))
            labels = _label_payload_for_record(record, root)
            row = {
                "case_id": case_id,
                "split": split,
                "raw": str(_resolve_dataset_path(root, record.get("raw"))),
                "rgb": _path_or_none(_resolve_dataset_path(root, record.get("rgb"))),
                "labels": str(_resolve_dataset_path(root, record.get("labels"))),
                "raw_shape": record.get("raw_shape", []),
                "raw_dtype": record.get("raw_dtype"),
                "raw_sha256": record.get("raw_sha256"),
                "class_map": dict(class_map),
                "objects": _objects_for_training(labels, class_map, warnings, case_id),
                "faca_metrics": record.get("faca_metrics", {}),
                "parameter_lineage": record.get("parameter_lineage", []),
            }
            stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            split_rows.setdefault(split, []).append(row["raw"])
            row_count += 1
    split_files = {}
    for split, paths in split_rows.items():
        split_path = splits_dir / f"{split}_raw.txt"
        split_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        split_files[split] = str(split_path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": row_count,
        "split_files": split_files,
    }


def _write_yolo_training_view(
    records: list[Mapping[str, Any]],
    root: Path,
    out_root: Path,
    class_map: Mapping[str, int],
    warnings: list[dict[str, Any]],
    *,
    copy_images: bool,
) -> dict[str, Any]:
    yolo_root = out_root / "yolo"
    labels_root = yolo_root / "labels"
    images_root = yolo_root / "images"
    yolo_root.mkdir(parents=True, exist_ok=True)
    split_images: dict[str, list[str]] = {}
    label_count = 0
    object_count = 0
    for index, record in enumerate(records):
        case_id = str(record.get("case_id", f"case_{index:04d}"))
        split = str(record.get("split", "unspecified"))
        rgb_path = _resolve_dataset_path(root, record.get("rgb"))
        if rgb_path is None or not rgb_path.exists():
            warnings.append({"case_id": case_id, "kind": "yolo_missing_rgb_preview"})
            continue
        labels = _label_payload_for_record(record, root)
        rows = _yolo_rows(labels, class_map, warnings, case_id)
        label_dir = labels_root / split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{case_id}.txt"
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        image_reference = rgb_path
        if copy_images:
            image_dir = images_root / split
            image_dir.mkdir(parents=True, exist_ok=True)
            image_reference = image_dir / f"{case_id}{rgb_path.suffix or '.png'}"
            shutil.copyfile(rgb_path, image_reference)
        split_images.setdefault(split, []).append(str(image_reference))
        label_count += 1
        object_count += len(rows)
    split_files = {}
    for split, paths in split_images.items():
        split_file = yolo_root / f"{split}.txt"
        split_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
        split_files[split] = str(split_file)
    names = [None] * (max(class_map.values(), default=-1) + 1)
    for label, class_id in class_map.items():
        if class_id >= 0:
            names[class_id] = label
    dataset_yaml = yolo_root / "dataset.yaml"
    dataset_yaml.write_text(
        _simple_yaml(
            {
                "path": str(yolo_root),
                "names": [
                    item if item is not None else f"class_{index}"
                    for index, item in enumerate(names)
                ],
                **split_files,
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": str(yolo_root),
        "dataset_yaml": str(dataset_yaml),
        "split_files": split_files,
        "label_count": label_count,
        "object_count": object_count,
        "copy_images": bool(copy_images),
        "truth_boundary": "YOLO view uses RGB previews, not RAW NPZ tensors.",
    }


def _label_payload_for_record(record: Mapping[str, Any], root: Path) -> dict[str, Any]:
    label_path = _resolve_dataset_path(root, record.get("labels"))
    if label_path is None or not label_path.exists():
        return {"objects": [], "masks": [], "source": "missing"}
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    labels = payload.get("labels", payload)
    return labels if isinstance(labels, dict) else {"objects": [], "masks": []}


def _objects_for_training(
    labels: Mapping[str, Any],
    class_map: Mapping[str, int],
    warnings: list[dict[str, Any]],
    case_id: str,
) -> list[dict[str, Any]]:
    objects = []
    for item in labels.get("objects", []):
        row = dict(item)
        label = str(row.get("label", "object"))
        class_id = int(row.get("class_id", class_map.get(label, -1)))
        if class_id < 0:
            warnings.append(
                {"case_id": case_id, "kind": "negative_or_unknown_class", "label": label}
            )
        row["class_id"] = class_id
        objects.append(row)
    return objects


def _yolo_rows(
    labels: Mapping[str, Any],
    class_map: Mapping[str, int],
    warnings: list[dict[str, Any]],
    case_id: str,
) -> list[str]:
    rows = []
    image_size = _image_size_rc(labels.get("image_size_rc", [1, 1]))
    for item in labels.get("objects", []):
        row = _yolo_row(item, image_size, class_map)
        if row is None:
            warnings.append(
                {
                    "case_id": case_id,
                    "kind": "skipped_yolo_object",
                    "label": item.get("label"),
                }
            )
            continue
        rows.append(row)
    return rows


def _yolo_row(
    item: Mapping[str, Any],
    image_size_rc: tuple[int, int],
    class_map: Mapping[str, int],
) -> str | None:
    label = str(item.get("label", "object"))
    class_id = int(item.get("class_id", class_map.get(label, -1)))
    if class_id < 0:
        return None
    yolo = item.get("yolo_xywhn")
    if yolo is None and item.get("bbox_xyxy") is not None:
        rows, cols = image_size_rc
        x1, y1, x2, y2 = [float(value) for value in item["bbox_xyxy"]]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        yolo = [
            (x1 + width / 2.0) / max(cols, 1),
            (y1 + height / 2.0) / max(rows, 1),
            width / max(cols, 1),
            height / max(rows, 1),
        ]
    if yolo is None or len(yolo) != 4:
        return None
    x_center, y_center, width, height = [float(value) for value in yolo]
    if width <= 0.0 or height <= 0.0:
        return None
    clipped = np.clip([x_center, y_center, width, height], 0.0, 1.0)
    return f"{class_id} {clipped[0]:.8f} {clipped[1]:.8f} {clipped[2]:.8f} {clipped[3]:.8f}"


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _simple_yaml(payload: Mapping[str, Any]) -> str:
    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[str(key)] = _deep_dict(item) if isinstance(item, Mapping) else item
    return result


def _normalize_splits(
    split: Mapping[str, float] | Iterable[str] | str | None, count: int
) -> list[str]:
    if count <= 0:
        return []
    if split is None:
        return ["unspecified" for _ in range(count)]
    if isinstance(split, str):
        return [split for _ in range(count)]
    if isinstance(split, Mapping):
        names = [str(key) for key in split]
        weights = np.asarray([float(split[key]) for key in split], dtype=float)
        if len(names) == 0 or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("split mapping must contain non-negative weights with positive sum.")
        normalized = weights / float(np.sum(weights))
        exact = normalized * count
        counts = np.floor(exact).astype(int)
        remainder = count - int(np.sum(counts))
        fractions = exact - counts
        for index in np.argsort(-fractions)[:remainder]:
            counts[int(index)] += 1
        result: list[str] = []
        for name, item_count in zip(names, counts, strict=True):
            result.extend([name] * int(item_count))
        return result[:count]
    values = [str(item) for item in split]
    if len(values) != count:
        raise ValueError("split iterable must contain one split name per scenario.")
    return values


def _split_policy(
    split: Mapping[str, float] | Iterable[str] | str | None,
    count: int,
    split_list: list[str],
) -> dict[str, Any]:
    if split is None:
        mode = "unspecified"
        requested: Any = None
    elif isinstance(split, str):
        mode = "single_split"
        requested = split
    elif isinstance(split, Mapping):
        mode = "weighted_deterministic"
        requested = {str(key): float(value) for key, value in split.items()}
    else:
        mode = "explicit_sequence"
        requested = list(split_list)
    counts: dict[str, int] = {}
    for name in split_list:
        counts[name] = counts.get(name, 0) + 1
    fractions = {
        name: (float(value) / max(int(count), 1))
        for name, value in counts.items()
    }
    return {
        "schema_version": "camerae2e_dataset_split_policy_v1",
        "mode": mode,
        "requested": requested,
        "assigned_counts": counts,
        "assigned_fractions": fractions,
        "case_count": int(count),
        "deterministic": True,
    }


def _array_from_stage(result: Mapping[str, Any], stage_name: str) -> np.ndarray:
    stage = dict(result.get("stages", {}).get(stage_name, {}))
    value = stage.get("array")
    if value is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(value)


def _uint8_preview(array: Any) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 2:
        values = np.repeat(values[:, :, None], 3, axis=2)
    if values.ndim == 3 and values.shape[2] > 3:
        values = values[:, :, :3]
    if values.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(finite)) > 1.0 or float(np.min(finite)) < 0.0:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        finite = (finite - lower) / max(upper - lower, 1.0e-12)
    return np.clip(np.round(finite * 255.0), 0.0, 255.0).astype(np.uint8)


def _write_stage_outputs(path: Path, result: Mapping[str, Any]) -> None:
    arrays: dict[str, Any] = {}
    for stage_name, stage in dict(result.get("stages", {})).items():
        if not isinstance(stage, Mapping) or stage.get("array") is None:
            continue
        array = np.asarray(stage.get("array"))
        if array.size == 0:
            continue
        arrays[_safe_npz_member_name(str(stage_name))] = array
    if not arrays:
        arrays["empty"] = np.empty(0, dtype=np.float32)
    _write_npz_deterministic(path, **arrays)


def _write_uncertainty_outputs(path: Path, raw_array: Any, sensor_digital_array: Any) -> None:
    raw = np.asarray(raw_array, dtype=np.float64)
    digital = np.asarray(sensor_digital_array, dtype=np.float64)
    if raw.size == 0:
        uncertainty = np.empty(0, dtype=np.float32)
        confidence = np.empty(0, dtype=np.float32)
    else:
        base = raw
        if digital.shape == raw.shape and np.nanstd(digital) > 0:
            base = 0.5 * raw + 0.5 * digital
        finite = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        center = float(np.median(finite))
        scale = max(float(np.std(finite)), 1.0e-12)
        uncertainty = np.clip(np.abs(finite - center) / (3.0 * scale), 0.0, 1.0).astype(
            np.float32
        )
        confidence = (1.0 - uncertainty).astype(np.float32)
    _write_npz_deterministic(
        path,
        uncertainty=uncertainty,
        confidence=confidence,
    )


def _uncertainty_model_manifest() -> dict[str, Any]:
    return {
        "schema_version": "camerae2e_uncertainty_proxy_v1",
        "readiness_tier": "proxy",
        "model": "normalized_raw_sensor_deviation",
        "outputs": ["uncertainty", "confidence"],
        "truth_boundary": (
            "This is a deterministic proxy map for regression/debugging. It is not "
            "a calibrated sensor noise posterior or model confidence estimate."
        ),
    }


def _safe_npz_member_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def _write_case_container(
    path: Path,
    *,
    record: Mapping[str, Any],
    raw_path: Path,
    label_path: Path,
    rgb_path: Path | None,
    tiff_path: Path | None,
    stage_outputs_path: Path | None,
    uncertainty_path: Path | None,
) -> None:
    payload = {
        "schema_version": "camerae2e_raw_container_manifest_v1",
        "case_id": record.get("case_id"),
        "readiness_tier": "proxy",
        "container_kind": "dng_like_zip",
        "truth_boundary": (
            "This package groups CameraE2E raw/stage/label artifacts. It is not a "
            "standards-compliant DNG and must not be consumed as one."
        ),
        "record": _jsonable(
            {
                key: value
                for key, value in record.items()
                if key not in {"container", "container_sha256"}
            }
        ),
    }
    members: list[tuple[str, Path | None]] = [
        ("raw.npz", raw_path),
        ("labels.json", label_path),
        ("rgb.png", rgb_path),
        ("raw_tiff.tiff", tiff_path),
        ("stage_outputs.npz", stage_outputs_path),
        ("uncertainty.npz", uncertainty_path),
    ]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        info = ZipInfo("container_manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, json.dumps(payload, sort_keys=True).encode("utf-8"))
        for member_name, source_path in members:
            if source_path is None or not source_path.exists():
                continue
            info = ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, source_path.read_bytes())


def _write_npz_deterministic(path: Path, **arrays: Any) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_array(array: Any) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape), sort_keys=True).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _load_manifest(
    dataset: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None, Path]:
    if isinstance(dataset, Mapping):
        manifest = dict(dataset)
        root = Path(str(manifest.get("dataset_root", "."))).expanduser()
        return manifest, None, root
    path = Path(dataset).expanduser()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(str(manifest.get("dataset_root", manifest_path.parent))).expanduser()
    return manifest, manifest_path, root


def _load_metadata_rows(path: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        issues.append({"kind": "missing_metadata", "message": f"Missing metadata.jsonl: {path}"})
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(
                {
                    "kind": "metadata_json",
                    "message": f"metadata.jsonl line {line_number} is invalid JSON: {exc}",
                }
            )
    return rows


def _validate_record(
    record: Mapping[str, Any],
    index: int,
    root: Path,
    metadata_by_case: Mapping[str, Mapping[str, Any]],
    issues: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    case_id = str(record.get("case_id", f"case_{index:04d}"))
    metadata_row = metadata_by_case.get(case_id)
    if metadata_row is None:
        issues.append({"case_id": case_id, "kind": "missing_metadata_row"})
    elif str(metadata_row.get("raw")) != str(record.get("raw")):
        issues.append({"case_id": case_id, "kind": "metadata_record_mismatch"})

    raw_path = _resolve_dataset_path(root, record.get("raw"))
    if raw_path is None or not raw_path.exists():
        issues.append({"case_id": case_id, "kind": "missing_raw", "path": str(raw_path)})
        return
    _check_file_hash(record, "raw_sha256", raw_path, case_id, issues)
    try:
        with np.load(raw_path, allow_pickle=False) as raw_data:
            raw = np.asarray(raw_data["raw"])
            sensor_digital = np.asarray(raw_data["sensor_digital"])
    except Exception as exc:
        issues.append({"case_id": case_id, "kind": "raw_npz_load", "message": str(exc)})
        return
    if list(raw.shape) != list(record.get("raw_shape", [])):
        issues.append({"case_id": case_id, "kind": "raw_shape"})
    if str(raw.dtype) != str(record.get("raw_dtype", raw.dtype)):
        issues.append({"case_id": case_id, "kind": "raw_dtype"})
    if record.get("raw_content_sha256") != _sha256_array(raw):
        issues.append({"case_id": case_id, "kind": "raw_content_sha256"})
    if record.get("sensor_digital_content_sha256") != _sha256_array(sensor_digital):
        issues.append({"case_id": case_id, "kind": "sensor_digital_content_sha256"})

    for path_key, hash_key, missing_kind in (
        ("rgb", "rgb_sha256", "missing_rgb"),
        ("raw_tiff", "raw_tiff_sha256", "missing_raw_tiff"),
        ("stage_outputs", "stage_outputs_sha256", "missing_stage_outputs"),
        ("container", "container_sha256", "missing_container"),
        ("uncertainty", "uncertainty_sha256", "missing_uncertainty"),
        ("labels", "labels_sha256", "missing_labels"),
    ):
        path_value = record.get(path_key)
        if path_value is None:
            continue
        path = _resolve_dataset_path(root, path_value)
        if path is None or not path.exists():
            issues.append({"case_id": case_id, "kind": missing_kind, "path": str(path)})
            continue
        _check_file_hash(record, hash_key, path, case_id, issues)

    label_path = _resolve_dataset_path(root, record.get("labels"))
    if label_path is not None and label_path.exists():
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        if labels.get("case_id") != case_id:
            issues.append({"case_id": case_id, "kind": "label_case_id"})
        if "labels" not in labels:
            warnings.append({"case_id": case_id, "kind": "label_payload_missing"})


def _check_file_hash(
    record: Mapping[str, Any],
    hash_key: str,
    path: Path,
    case_id: str,
    issues: list[dict[str, Any]],
) -> None:
    expected = record.get(hash_key)
    if expected is None:
        return
    actual = _sha256_file(path)
    if actual != expected:
        issues.append(
            {"case_id": case_id, "kind": hash_key, "expected": expected, "actual": actual}
        )


def _resolve_dataset_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


cameraE2EDatasetExport = camerae2e_dataset_export  # noqa: N816
cameraE2EDatasetExportADASKITTIDemo = camerae2e_dataset_export_adas_kitti_demo  # noqa: N816
cameraE2EDatasetExportCameraSpecVariants = (  # noqa: N816
    camerae2e_dataset_export_camera_spec_variants
)
cameraE2EDatasetExportFromOptimization = camerae2e_dataset_export_from_optimization  # noqa: N816
cameraE2EDatasetExportPerceptionIndex = camerae2e_dataset_export_perception_index  # noqa: N816
cameraE2EDatasetValidate = camerae2e_dataset_validate  # noqa: N816
cameraE2EADASCameraSpec = camerae2e_adas_camera_spec  # noqa: N816
cameraE2EKITTIYOLOLabels = camerae2e_kitti_yolo_labels  # noqa: N816
