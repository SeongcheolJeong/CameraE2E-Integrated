"""Validate the self-contained CameraE2E integrated repository checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    payload = build_manifest(repo_root)
    if args.write_manifest:
        output = repo_root / "integrated_repository_manifest.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["manifest_path"] = str(output.relative_to(repo_root))

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def build_manifest(repo_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    components = {
        "camerae2e_code": repo_root / "src/pyisetcam",
        "camerae2e_v2_code": repo_root / "src/camerae2e_v2",
        "camerae2e_tests": repo_root / "tests",
        "camerae2e_tools": repo_root / "tools",
        "camerae2e_workbench": repo_root / "camerae2e-workbench",
        "fdtd_tcad_workspace": repo_root / "simulations/fdtd_tcad",
        "rayoptics_workspace": repo_root / "simulations/rayoptics",
        "camera_db_manifest": repo_root / "camerae2e_db/manifest.json",
        "sensor_catalog": repo_root / "camerae2e_db/fdtd_tcad/sensor_db/sensor_catalog.json",
        "lens_db_package": repo_root
        / "camerae2e_db/lens_db/CameraE2E_Lens_DB_v9_20260627/data/lens_patents",
    }
    for name, path in components.items():
        checks.append(
            {
                "name": name,
                "kind": "required_path",
                "path": _rel(repo_root, path),
                "ok": path.exists(),
            }
        )

    checks.extend(_file_size_checks(repo_root))
    checks.extend(_nested_git_checks(repo_root))
    checks.extend(_runtime_path_checks(repo_root))
    checks.extend(_repo_local_default_checks(repo_root))

    summary = _summary(repo_root, checks)
    return {
        "schema_version": "camerae2e_integrated_repository_manifest_v1",
        "repository_root": ".",
        "ok": all(check["ok"] for check in checks),
        "summary": summary,
        "components": {name: _path_info(repo_root, path) for name, path in components.items()},
        "checks": checks,
        "truth_boundary": {
            "claim": "research-grade self-contained checkout",
            "not_claimed": [
                "product sign-off",
                "measured TCAD calibration",
                "vendor ISP latency trace sign-off",
                "wave-optics sign-off for geometric RayOptics PSF",
            ],
        },
    }


def _file_size_checks(repo_root: Path) -> list[dict[str, Any]]:
    largest: list[tuple[int, Path]] = []
    oversized: list[Path] = []
    for path in _tracked_candidate_files(repo_root):
        size = path.stat().st_size
        largest.append((size, path))
        if size >= GITHUB_FILE_LIMIT_BYTES:
            oversized.append(path)
    largest.sort(reverse=True)
    return [
        {
            "name": "github_regular_file_limit",
            "kind": "file_size",
            "ok": not oversized,
            "limit_bytes": GITHUB_FILE_LIMIT_BYTES,
            "oversized": [_rel(repo_root, item) for item in oversized],
            "largest_files": [
                {"path": _rel(repo_root, path), "bytes": size}
                for size, path in largest[:10]
            ],
        }
    ]


def _nested_git_checks(repo_root: Path) -> list[dict[str, Any]]:
    nested = [
        path
        for path in repo_root.rglob(".git")
        if path != repo_root / ".git" and ".pytest_cache" not in path.parts
    ]
    return [
        {
            "name": "no_nested_git_directories",
            "kind": "repository_shape",
            "ok": not nested,
            "paths": [_rel(repo_root, item) for item in nested],
        }
    ]


def _runtime_path_checks(repo_root: Path) -> list[dict[str, Any]]:
    runtime_files = [
        repo_root / "src/pyisetcam/physics_simulation.py",
        repo_root / "src/pyisetcam/fdtd_sensor.py",
        repo_root / "src/pyisetcam/image_sensor_db.py",
        repo_root / "src/pyisetcam/tcad_sensor.py",
        repo_root / "src/camerae2e_v2/service.py",
        repo_root / "tools/package_camerae2e_db_repository.py",
        repo_root / "camerae2e-workbench/backend/app/service.py",
        repo_root / "camerae2e-workbench/package.json",
    ]
    needles = ["/Users/"]
    hits: list[dict[str, Any]] = []
    for path in runtime_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle in text:
                hits.append({"path": _rel(repo_root, path), "needle": needle})
    return [
        {
            "name": "runtime_defaults_are_not_user_absolute_paths",
            "kind": "runtime_path",
            "ok": not hits,
            "hits": hits,
        }
    ]


def _repo_local_default_checks(repo_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    default_paths = {
        "image_sensor_db_root": repo_root / "camerae2e_db/fdtd_tcad/sensor_db",
        "fdtd_sensor_default_lut_path": repo_root
        / "camerae2e_db/fdtd_tcad/runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json",
        "tcad_generation_map_path": repo_root
        / (
            "camerae2e_db/fdtd_tcad/runs/fdtd_to_tcad_generation_2d_cra_smoke/"
            "tcad_generation_map_2d.npz"
        ),
        "tcad_center_collection_summary": repo_root
        / "camerae2e_db/fdtd_tcad/runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json",
        "rayoptics_lens_package": repo_root
        / "camerae2e_db/lens_db/CameraE2E_Lens_DB_v9_20260627/data/lens_patents",
        "physics_simulation_manifest_snapshot": repo_root
        / "reports/camerae2e_goal/physics_simulation_manifest.json",
    }
    for name, path in default_paths.items():
        checks.append(_path_under_repo_check(repo_root, name, path))

    manifest_path = default_paths["physics_simulation_manifest_snapshot"]
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks.append(
            {
                "name": "physics_simulation_manifest_snapshot_schema",
                "kind": "repo_local_default",
                "ok": payload.get("schema_version")
                == "camerae2e_physics_simulation_manifest_v1",
                "stage_count": payload.get("summary", {}).get("stage_count"),
                "status_counts": payload.get("summary", {}).get("status_counts", {}),
            }
        )
    return checks


def _path_under_repo_check(repo_root: Path, name: str, path: Path | None) -> dict[str, Any]:
    resolved = None if path is None else path.expanduser().resolve()
    ok = bool(resolved is not None and resolved.exists() and _is_relative_to(resolved, repo_root))
    return {
        "name": name,
        "kind": "repo_local_default",
        "ok": ok,
        "path": None if resolved is None else _rel(repo_root, resolved),
    }


def _summary(repo_root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    total_files = 0
    total_bytes = 0
    for path in _tracked_candidate_files(repo_root):
        total_files += 1
        total_bytes += path.stat().st_size
    failed = [check["name"] for check in checks if not check["ok"]]
    return {
        "file_count": total_files,
        "bytes": total_bytes,
        "size_mb": round(total_bytes / (1024 * 1024), 3),
        "check_count": len(checks),
        "failed_checks": failed,
    }


def _path_info(repo_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _rel(repo_root, path),
        "exists": path.exists(),
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
        "bytes": _path_size(repo_root, path) if path.exists() else 0,
    }


def _path_size(repo_root: Path, path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in _tracked_candidate_files(repo_root)
        if _is_relative_to(item, path)
    )


def _tracked_candidate_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        result = None
    if result is not None:
        return [
            path
            for item in result.stdout.split(b"\0")
            if item
            and (path := repo_root / item.decode("utf-8", errors="surrogateescape")).is_file()
        ]
    ignored_parts = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    return [
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and not any(part in ignored_parts for part in path.relative_to(repo_root).parts)
    ]


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
