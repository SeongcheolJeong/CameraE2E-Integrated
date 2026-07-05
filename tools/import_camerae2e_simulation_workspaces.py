#!/usr/bin/env python3
"""Import FDTD/TCAD and RayOptics simulation source into the CameraE2E monorepo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FDTD_ROOT = Path("/Users/seongcheoljeong/FDTD")
DEFAULT_RAYOPTICS_ROOT = Path("/Users/seongcheoljeong/RayOptics")
DEFAULT_DEST_ROOT = REPO_ROOT / "simulations"

COMMON_EXCLUDES = {
    ".DS_Store",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".meep-env",
    ".tcad-env",
    "runs",
    "node_modules",
    "dist",
}

FDTD_EXCLUDES = COMMON_EXCLUDES | {
    "FDTD_UX.rtf",
    "f6b68b1b-e4c5-4171-95fb-0370a3e11f8e.png",
    "*.rtf",
    "*.png",
    "*.log",
}

RAYOPTICS_EXCLUDES = COMMON_EXCLUDES | {
    ".rayoptics-workbench",
    "rayoptics-env",
    "*.zip",
    "*.rtf",
    "*.png",
    "*.log",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdtd-root", type=Path, default=DEFAULT_FDTD_ROOT)
    parser.add_argument("--rayoptics-root", type=Path, default=DEFAULT_RAYOPTICS_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = import_workspaces(
        fdtd_root=args.fdtd_root,
        rayoptics_root=args.rayoptics_root,
        dest_root=args.dest_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def import_workspaces(
    *,
    fdtd_root: Path,
    rayoptics_root: Path,
    dest_root: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    dest_root = dest_root.expanduser().resolve()
    imports = [
        _import_spec("fdtd_tcad", fdtd_root.expanduser(), dest_root / "fdtd_tcad", FDTD_EXCLUDES),
        _import_spec(
            "rayoptics",
            rayoptics_root.expanduser(),
            dest_root / "rayoptics",
            RAYOPTICS_EXCLUDES,
        ),
    ]
    for spec in imports:
        if dry_run:
            continue
        source = Path(str(spec["source"]))
        destination = Path(str(spec["destination"]))
        if not source.exists():
            continue
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, ignore=_ignore_patterns(spec["excludes"]))
    return {
        "schema_version": "camerae2e_simulation_workspace_import_v1",
        "dry_run": bool(dry_run),
        "dest_root": str(dest_root),
        "imports": [
            {
                **spec,
                "source_exists": Path(str(spec["source"])).exists(),
                "destination_exists": Path(str(spec["destination"])).exists(),
            }
            for spec in imports
        ],
        "policy": {
            "large_outputs": "excluded",
            "generated_runs": "excluded",
            "runtime_envs": "excluded",
            "source_and_small_fixtures": "included",
        },
    }


def _import_spec(name: str, source: Path, destination: Path, excludes: set[str]) -> dict[str, object]:
    return {
        "name": name,
        "source": str(source),
        "destination": str(destination),
        "excludes": sorted(excludes),
    }


def _ignore_patterns(patterns: set[str]):
    return shutil.ignore_patterns(*sorted(patterns))


if __name__ == "__main__":
    main()
