"""Run the CameraE2E goal-level validation and refresh gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyisetcam import camerae2e_goal_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/camerae2e_goal"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("outputs/camerae2e-goal-gate-smoke"),
        help="Directory for generated smoke RAW datasets and previews.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-demos", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    payload = camerae2e_goal_gate(
        args.output_dir,
        artifact_dir=args.artifact_dir,
        strict=args.strict,
        include_demos=not args.skip_demos,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "strict": payload["strict"],
                "summary": payload["summary"],
                "reports": payload["reports"],
                "artifact_dir": payload["artifact_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
