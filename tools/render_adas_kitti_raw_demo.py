"""Render a KITTI/YOLO-style ADAS RAW dataset demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyisetcam import (
    camerae2e_dataset_export_adas_kitti_demo,
    camerae2e_dataset_export_camera_spec_variants,
    camerae2e_dataset_validate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adas-kitti-yolo-raw-demo"))
    parser.add_argument("--case-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-tiff", action="store_true")
    parser.add_argument(
        "--variants",
        action="store_true",
        help=(
            "Render KITTI-style source scenes through KITTI, wide-FOV, "
            "and narrow-FOV camera specs."
        ),
    )
    args = parser.parse_args()

    if args.variants:
        manifest = camerae2e_dataset_export_camera_spec_variants(
            args.output_dir,
            case_count=args.case_count,
            seed=args.seed,
            include_tiff=args.include_tiff,
        )
        input_mode = "camera_spec_variants"
        truth_boundary = manifest["camera_spec_variants"]["truth_boundary"]
    else:
        manifest = camerae2e_dataset_export_adas_kitti_demo(
            args.output_dir,
            case_count=args.case_count,
            seed=args.seed,
            include_tiff=args.include_tiff,
            split="demo",
        )
        input_mode = manifest["adas_kitti_demo"]["input_mode"]
        truth_boundary = manifest["adas_kitti_demo"]["camera_spec"]["truth_boundary"]
    validation = camerae2e_dataset_validate(manifest)
    summary = {
        "manifest": str(Path(manifest["dataset_root"]) / "manifest.json"),
        "case_count": manifest["case_count"],
        "validation_ok": validation["ok"],
        "input_mode": input_mode,
        "truth_boundary": truth_boundary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
