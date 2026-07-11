"""Portable content hashes and runtime provenance for persisted evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_provenance() -> dict[str, str]:
    configured = os.environ.get("CAMERAE2E_CODE_REVISION", "").strip()
    git = _git_provenance()
    revision = configured or git["code_revision"]
    try:
        package_version = version("pyisetcam")
    except PackageNotFoundError:
        package_version = "source-tree"
    return {
        "package": "pyisetcam",
        "package_version": package_version,
        "code_revision": revision,
        "code_state": "configured" if configured else git["code_state"],
    }


def _git_provenance() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {"code_revision": "unknown", "code_state": "unknown"}
    return {
        "code_revision": revision.stdout.strip() or "unknown",
        "code_state": "dirty" if status.stdout.strip() else "clean",
    }
