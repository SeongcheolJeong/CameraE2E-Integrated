"""Build a final-result-only Lens/Sensor DB repository for CameraE2E.

The target repository is intentionally data-only.  It keeps the final lens DB,
sensor catalog/config DB, and small FDTD/TCAD handoff artifacts needed by
CameraE2E runtime APIs, while excluding solver source trees, environments,
large raw runs, and UI/runtime outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT.parent / "CameraE2E-DB"
OPTIONAL_EXTERNAL_FDTD_ROOT = REPO_ROOT / "external/fdtd_tcad"
OPTIONAL_EXTERNAL_RAYOPTICS_ROOT = REPO_ROOT / "external/rayoptics"
LENS_PACKAGE_NAME = "CameraE2E_Lens_DB_v9_20260627"
HASH_CHUNK_SIZE = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--fdtd-root", type=Path, default=None)
    parser.add_argument("--rayoptics-root", type=Path, default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove target contents except .git first.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan the package without copying.")
    args = parser.parse_args()

    package = build_package_plan(
        target=args.target.expanduser().resolve(),
        fdtd_root=args.fdtd_root,
        rayoptics_root=args.rayoptics_root,
    )
    if args.dry_run:
        print(json.dumps(package["plan"], indent=2, sort_keys=True))
        return 0
    write_package(package, clean=args.clean)
    print(json.dumps(package["summary"], indent=2, sort_keys=True))
    return 0


def build_package_plan(
    *,
    target: Path,
    fdtd_root: Path | None = None,
    rayoptics_root: Path | None = None,
) -> dict[str, Any]:
    fdtd = _resolve_fdtd_root(fdtd_root)
    rayoptics = _resolve_rayoptics_root(rayoptics_root)
    lens_package = _first_existing(
        [
            rayoptics / LENS_PACKAGE_NAME,
            REPO_ROOT / "simulations/rayoptics" / LENS_PACKAGE_NAME,
            OPTIONAL_EXTERNAL_RAYOPTICS_ROOT / LENS_PACKAGE_NAME,
        ]
    )
    sensor_db = _first_existing(
        [
            fdtd / "sensor_db",
            REPO_ROOT / "simulations/fdtd_tcad/sensor_db",
            OPTIONAL_EXTERNAL_FDTD_ROOT / "sensor_db",
        ]
    )
    image_sensor_db = _first_existing(
        [
            fdtd / "image_sensor_db",
            REPO_ROOT / "simulations/fdtd_tcad/image_sensor_db",
            OPTIONAL_EXTERNAL_FDTD_ROOT / "image_sensor_db",
        ]
    )
    fixtures = _first_existing(
        [
            fdtd / "fixtures",
            REPO_ROOT / "simulations/fdtd_tcad/fixtures",
        ]
    )
    if lens_package is None:
        raise FileNotFoundError("No CameraE2E RayOptics Lens DB v9 package was found.")
    if sensor_db is None:
        raise FileNotFoundError("No FDTD sensor_db package was found.")
    if fixtures is None:
        raise FileNotFoundError("No FDTD/TCAD fixture package was found.")

    copies: list[dict[str, Any]] = [
        _copy_item(
            source=lens_package / "README_CAMERA_E2E.md",
            target=target / "lens_db" / LENS_PACKAGE_NAME / "README_CAMERA_E2E.md",
            required=False,
            role="lens_package_readme",
        ),
        _copy_item(
            source=lens_package / "data/lens_patents",
            target=target / "lens_db" / LENS_PACKAGE_NAME / "data/lens_patents",
            required=True,
            role="rayoptics_lens_final_data",
        ),
        _copy_item(
            source=sensor_db,
            target=target / "fdtd_tcad/sensor_db",
            required=True,
            role="sensor_catalog_and_stack_configs",
        ),
        _copy_item(
            source=image_sensor_db,
            target=target / "fdtd_tcad/image_sensor_db",
            required=False,
            role="image_sensor_db_enriched_catalog",
        ),
        _copy_item(
            source=fixtures,
            target=target / "fdtd_tcad/fixtures",
            required=True,
            role="small_fdtd_tcad_handoff_fixtures",
        ),
    ]
    aliases = _run_aliases(fixtures=fixtures, target=target)
    copies.extend(aliases)
    missing_required = [item for item in copies if item["required"] and not item["source_exists"]]
    if missing_required:
        raise FileNotFoundError(f"Missing required package inputs: {missing_required}")
    return {
        "target": target,
        "plan": {
            "schema_version": "camerae2e_db_package_plan_v1",
            "target": str(target),
            "source_roots": {
                "fdtd": str(fdtd),
                "rayoptics": str(rayoptics),
                "lens_package": str(lens_package),
                "sensor_db": str(sensor_db),
                "image_sensor_db": None if image_sensor_db is None else str(image_sensor_db),
                "fixtures": str(fixtures),
            },
            "copies": copies,
        },
        "summary": {},
    }


def write_package(package: dict[str, Any], *, clean: bool) -> None:
    target = Path(package["target"])
    if clean and target.exists():
        _clean_target(target)
    target.mkdir(parents=True, exist_ok=True)

    for item in package["plan"]["copies"]:
        if not item["source_exists"]:
            continue
        source = Path(item["source"])
        destination = Path(item["target"])
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_copy_ignore)
        else:
            shutil.copy2(source, destination)

    _write_readme(target)
    manifest = _build_manifest(target, package["plan"])
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    package["summary"] = manifest["summary"]


def _resolve_fdtd_root(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    candidates = [
        REPO_ROOT / "simulations/fdtd_tcad",
        OPTIONAL_EXTERNAL_FDTD_ROOT,
    ]
    return _first_existing(candidates) or candidates[0]


def _resolve_rayoptics_root(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    candidates = [
        REPO_ROOT / "simulations/rayoptics",
        OPTIONAL_EXTERNAL_RAYOPTICS_ROOT,
    ]
    return _first_existing(candidates) or candidates[0]


def _copy_item(*, source: Path | None, target: Path, required: bool, role: str) -> dict[str, Any]:
    exists = bool(source is not None and source.exists())
    return {
        "role": role,
        "source": None if source is None else str(source),
        "target": str(target),
        "required": bool(required),
        "source_exists": exists,
        "source_kind": "directory" if exists and source.is_dir() else "file" if exists else None,
    }


def _run_aliases(*, fixtures: Path, target: Path) -> list[dict[str, Any]]:
    fdtd_lut_alias = (
        target / "fdtd_tcad/runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json"
    )
    generation_alias = (
        target
        / "fdtd_tcad/runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz"
    )
    center_alias = (
        target
        / "fdtd_tcad/runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json"
    )
    edge_alias = (
        target
        / "fdtd_tcad/runs/devsim_split_pd_2d_fdtd_map_proxy_edge20x_smoke/summary.json"
    )
    gate_alias = (
        target / "fdtd_tcad/runs/tcad_accuracy_gate_reference_profile/tcad_accuracy_gate.json"
    )
    return [
        _copy_item(
            source=fixtures / "fdtd_active_lut/camera_lut.json",
            target=fdtd_lut_alias,
            required=True,
            role="fdtd_default_lut_alias",
        ),
        _copy_item(
            source=fixtures / "tcad_generation_map/tcad_generation_map_2d.npz",
            target=generation_alias,
            required=True,
            role="tcad_generation_map_alias",
        ),
        _copy_item(
            source=fixtures / "devsim_split_pd_center/summary.json",
            target=center_alias,
            required=True,
            role="tcad_center_collection_alias",
        ),
        _copy_item(
            source=fixtures / "devsim_split_pd_edge/summary.json",
            target=edge_alias,
            required=True,
            role="tcad_edge_collection_alias",
        ),
        _copy_item(
            source=fixtures / "tcad_accuracy_gate/tcad_accuracy_gate.json",
            target=gate_alias,
            required=True,
            role="tcad_accuracy_gate_alias",
        ),
    ]


def _write_readme(target: Path) -> None:
    body = f"""# CameraE2E Lens/Sensor DB

This repository contains final-result-only data artifacts used by CameraE2E for
camera information lookup and runtime simulation inputs.

## Contents

- `lens_db/{LENS_PACKAGE_NAME}/data/lens_patents/`: RayOptics-derived lens
  patent SQLite DB, CameraE2E manifest, inventory, and geometric PSF NPZ assets.
- `fdtd_tcad/sensor_db/`: image sensor catalog and generated stack configs.
- `fdtd_tcad/image_sensor_db/`: enriched sensor catalog and TCAD candidate report.
- `fdtd_tcad/fixtures/`: small FDTD/TCAD handoff fixtures with known proxy lineage.
- `fdtd_tcad/runs/`: small compatibility aliases so existing CameraE2E default
  path discovery works without copying full solver runs.

## CameraE2E setup

```bash
export PYISETCAM_CAMERA_DB_ROOT=/absolute/path/to/CameraE2E-DB
export PYISETCAM_LENS_DB_ROOT=$PYISETCAM_CAMERA_DB_ROOT/lens_db/{LENS_PACKAGE_NAME}
export PYISETCAM_FDTD_ROOT=$PYISETCAM_CAMERA_DB_ROOT/fdtd_tcad
```

`PYISETCAM_CAMERA_DB_ROOT` is enough for recent CameraE2E code.  The explicit
`PYISETCAM_LENS_DB_ROOT` and `PYISETCAM_FDTD_ROOT` exports are kept for older
tools and scripts.

## Truth Boundary

This is not a product sign-off database.  RayOptics PSFs are geometric
ray-histogram PSFs, not diffraction/wave-optics validation.  FDTD/TCAD assets
are lookup/proxy handoff artifacts unless measured calibration and strict
lineage gates are attached.  The current small TCAD fixture intentionally keeps
the known FDTD-LUT/generation-map lineage mismatch visible.
"""
    (target / "README.md").write_text(body, encoding="utf-8")


def _build_manifest(target: Path, plan: dict[str, Any]) -> dict[str, Any]:
    files = []
    manifest_files = (
        item for item in target.rglob("*") if item.is_file() and item.name != "manifest.json"
    )
    for path in sorted(manifest_files):
        files.append(
            {
                "path": path.relative_to(target).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    role_summaries = []
    for item in plan["copies"]:
        destination = Path(item["target"])
        if not destination.exists():
            role_summaries.append({**item, "target_exists": False, "file_count": 0, "bytes": 0})
            continue
        role_summaries.append(
            {
                **item,
                "target_exists": True,
                "file_count": _count_files(destination),
                "bytes": _count_bytes(destination),
                "tree_sha256": _tree_sha256(destination),
            }
        )
    total_bytes = sum(int(item["bytes"]) for item in files)
    return {
        "schema_version": "camerae2e_db_repository_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "source_plan": plan,
        "summary": {
            "file_count": len(files),
            "bytes": total_bytes,
            "size_mb": round(total_bytes / (1024 * 1024), 3),
            "lens_package": LENS_PACKAGE_NAME,
            "roles": {
                item["role"]: {
                    "file_count": item["file_count"],
                    "bytes": item["bytes"],
                    "target_exists": item["target_exists"],
                }
                for item in role_summaries
            },
        },
        "readiness_tiers": {
            "lens_db": "proxy",
            "sensor_stack_catalog": "proxy",
            "fdtd_lut": "proxy",
            "tcad_collection": "calibration_required",
        },
        "truth_boundaries": [
            "rayoptics_geometric_psf_not_diffraction_wave_optics",
            "fdtd_optical_absorption_and_regional_response_proxy",
            "tcad_collection_framework_not_product_calibrated",
            "small_fixtures_preserve_known_lineage_mismatch",
        ],
        "role_summaries": role_summaries,
        "files": files,
    }


def _clean_target(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        "__pycache__",
        ".DS_Store",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
    }
    return {name for name in names if name in ignored}


def _first_existing(candidates: list[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.expanduser().resolve()
    return None


def _count_files(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _count_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
        return digest.hexdigest()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
