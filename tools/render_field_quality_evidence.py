"""Compatibility wrapper for the integrated camera-field parity report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATED_RENDERER = REPO_ROOT / "tools" / "render_parity_evidence.py"
DEFAULT_MARKDOWN = REPO_ROOT / "reports" / "parity" / "camera_field_parity_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="Regenerate reports/parity/latest.json before rendering the integrated report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [sys.executable, str(INTEGRATED_RENDERER)]
    if args.refresh_report:
        command.append("--refresh-report")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(DEFAULT_MARKDOWN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
