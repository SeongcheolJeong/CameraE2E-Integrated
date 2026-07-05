#!/usr/bin/env python
"""Fetch the pinned upstream ISETCam snapshot into the local cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyisetcam.assets import (  # noqa: E402
    DEFAULT_UPSTREAM_SHA,
    DEFAULT_UPSTREAM_TARBALL_SHA256,
    ensure_upstream_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", default=DEFAULT_UPSTREAM_SHA, help="Upstream ISETCam commit SHA.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / ".cache",
        help="Cache root that will contain upstream/isetcam/<sha>/.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=DEFAULT_UPSTREAM_TARBALL_SHA256,
        help="Expected SHA256 for the downloaded GitHub tarball.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract the snapshot.")
    arguments = parser.parse_args()

    snapshot = ensure_upstream_snapshot(
        sha=arguments.sha,
        cache_root=arguments.cache_root,
        expected_sha256=arguments.expected_sha256,
        force=arguments.force,
    )
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

