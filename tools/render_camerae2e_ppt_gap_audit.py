#!/usr/bin/env python3
"""Render a CameraE2E technical-overview PPT gap audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyisetcam import camerae2e_ppt_gap_audit

DEFAULT_PPTX = Path("/Users/seongcheoljeong/Downloads/camerae2e-technical-overview-images.pptx")
DEFAULT_OUTPUT_DIR = Path("reports/camerae2e_goal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pptx",
        type=Path,
        default=DEFAULT_PPTX,
        help="Path to the CameraE2E technical overview PPTX.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ppt_gap_audit.json/html.",
    )
    args = parser.parse_args()

    audit = camerae2e_ppt_gap_audit(args.pptx, output_dir=args.output_dir)
    print(json.dumps(audit["reports"], indent=2, sort_keys=True))
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
