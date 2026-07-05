"""Regenerate Octave parity baselines for the curated pyisetcam cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyisetcam import ensure_upstream_snapshot
from pyisetcam.exceptions import OctaveExecutionError
from pyisetcam.octave_runner import OCTAVE_BIN_ENV, find_octave_binary, run_octave

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPO_ROOT / "tests" / "parity" / "cases.yaml"
BASELINES_DIR = REPO_ROOT / "tests" / "parity" / "baselines"
OCTAVE_RUNNER = REPO_ROOT / "tools" / "octave" / "run_case.m"


def _case_names() -> list[str]:
    payload = json.loads(CASES_PATH.read_text())
    return [str(case["name"]) for case in payload["cases"]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Case names to regenerate. Defaults to all curated cases.")
    parser.add_argument(
        "--output-dir",
        default=str(BASELINES_DIR),
        help="Directory where exported .mat baselines will be written.",
    )
    parser.add_argument(
        "--upstream-root",
        default="",
        help="Optional explicit upstream ISETCam root. Defaults to the pinned cached snapshot.",
    )
    parser.add_argument(
        "--octave-bin",
        default="",
        help=f"Optional explicit Octave executable. Falls back to {OCTAVE_BIN_ENV} or PATH search.",
    )
    parser.add_argument("--list", action="store_true", help="Print the curated case names and exit.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    available_cases = _case_names()
    if args.list:
        for case_name in available_cases:
            print(case_name)
        return 0

    requested_cases = args.cases or available_cases
    unknown_cases = sorted(set(requested_cases) - set(available_cases))
    if unknown_cases:
        print(f"Unknown parity cases: {', '.join(unknown_cases)}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_root = (
        Path(args.upstream_root).expanduser()
        if args.upstream_root
        else ensure_upstream_snapshot()
    )

    try:
        octave_bin = find_octave_binary(preferred=args.octave_bin or None)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(f"Using Octave binary: {octave_bin}")
    print(f"Using upstream snapshot: {upstream_root}")

    failures = 0
    for case_name in requested_cases:
        destination = output_dir / f"{case_name}.mat"
        arguments = [
            "--no-gui",
            "--no-window-system",
            "--quiet",
            str(OCTAVE_RUNNER),
            case_name,
            str(destination),
            str(upstream_root),
        ]
        try:
            run_octave(arguments, cwd=REPO_ROOT, octave_bin=octave_bin)
        except OctaveExecutionError as error:
            failures += 1
            print(f"[failed] {case_name}: {error}", file=sys.stderr)
            if error.stderr:
                print(error.stderr.strip(), file=sys.stderr)
            if error.stdout:
                print(error.stdout.strip(), file=sys.stderr)
            continue

        print(f"[ok] {case_name} -> {destination}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
