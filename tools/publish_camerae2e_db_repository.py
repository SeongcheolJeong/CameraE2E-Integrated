"""Validate and push the standalone CameraE2E-DB repository.

This helper cannot create a GitHub repository by itself.  Create
``SeongcheolJeong/CameraE2E-DB`` on GitHub first, then run this command to
verify the local DB package and push it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_REPO = REPO_ROOT.parent / "CameraE2E-DB"
DEFAULT_REMOTE = "git@github.com:SeongcheolJeong/CameraE2E-DB.git"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_DB_REPO)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--skip-remote-check", action="store_true")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    manifest = _load_manifest(target)
    _ensure_git_repo(target, branch=args.branch, remote=args.remote)
    _ensure_clean(target)
    if not args.skip_remote_check:
        _ensure_remote_exists(args.remote)
    _run(["git", "push", "-u", "origin", args.branch], cwd=target)
    print(
        json.dumps(
            {
                "ok": True,
                "target": str(target),
                "remote": args.remote,
                "branch": args.branch,
                "manifest_summary": manifest.get("summary", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_manifest(target: Path) -> dict[str, Any]:
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"CameraE2E-DB manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "camerae2e_db_repository_manifest_v1":
        raise SystemExit(f"Unexpected manifest schema in {manifest_path}")
    return payload


def _ensure_git_repo(target: Path, *, branch: str, remote: str) -> None:
    if not (target / ".git").exists():
        _run(["git", "init"], cwd=target)
    current_branch = _run(["git", "branch", "--show-current"], cwd=target).stdout.strip()
    if current_branch != branch:
        if current_branch:
            _run(["git", "branch", "-m", branch], cwd=target)
        else:
            _run(["git", "checkout", "-B", branch], cwd=target)
    remotes = _run(["git", "remote"], cwd=target).stdout.splitlines()
    if "origin" in remotes:
        _run(["git", "remote", "set-url", "origin", remote], cwd=target)
    else:
        _run(["git", "remote", "add", "origin", remote], cwd=target)


def _ensure_clean(target: Path) -> None:
    status = _run(["git", "status", "--short"], cwd=target).stdout.strip()
    if status:
        commit_hint = (
            "Run `git -C /path/to/CameraE2E-DB add -A && "
            "git -C /path/to/CameraE2E-DB commit ...` first."
        )
        raise SystemExit(
            "CameraE2E-DB has uncommitted changes. "
            + commit_hint
            + "\n"
            + status
        )


def _ensure_remote_exists(remote: str) -> None:
    probe = subprocess.run(
        ["git", "ls-remote", remote],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        message = "Remote repository is not reachable. Create it on GitHub first."
        raise SystemExit(
            message
            + " Then rerun this helper.\n"
            f"Remote: {remote}\n"
            f"git stderr: {probe.stderr.strip()}"
        )


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
