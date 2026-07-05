#!/usr/bin/env python3
"""Render the CameraE2E FDTD/TCAD/RayOptics simulation bridge manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyisetcam import camerae2e_physics_simulation_manifest

DEFAULT_OUTPUT_DIR = Path("reports/camerae2e_goal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdtd-root", type=Path, default=None)
    parser.add_argument("--rayoptics-root", type=Path, default=None)
    parser.add_argument("--camera-db-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = camerae2e_physics_simulation_manifest(
        fdtd_root=args.fdtd_root,
        rayoptics_root=args.rayoptics_root,
        camera_db_root=args.camera_db_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.get("reports", {}), indent=2, sort_keys=True))
    print(json.dumps(manifest.get("summary", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
