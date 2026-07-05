"""Validate CameraE2E calibration evidence and readiness promotion candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyisetcam import (
    camerae2e_calibration_evidence_requirements,
    camerae2e_calibration_evidence_validate,
    camerae2e_readiness_promotion_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/camerae2e_goal/calibration_evidence_validation.json"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = {
        "schema_version": "camerae2e_calibration_evidence_cli_v1",
        "requirements": camerae2e_calibration_evidence_requirements(),
        "validation": camerae2e_calibration_evidence_validate(
            args.evidence,
            strict=args.strict,
        ),
        "promotion_plan": camerae2e_readiness_promotion_plan(
            args.evidence,
            strict=args.strict,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": payload["validation"]["ok"],
                "evidence": None if args.evidence is None else str(args.evidence),
                "output": str(args.output),
                "promotion_summary": payload["promotion_plan"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["validation"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
