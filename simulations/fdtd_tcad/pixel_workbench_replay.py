#!/usr/bin/env python3
"""Replay a persisted Pixel Workbench suite case command."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
COMPARABLE_CSV_NAMES = (
    "camera_lut_summary.csv",
    "camera_lut_long.csv",
    "crosstalk_kernel_summary.csv",
    "crosstalk_output_kernel.csv",
    "crosstalk_raw_pd_kernel.csv",
)
COMPARABLE_JSON_NAMES = (
    "crosstalk_convergence.json",
    "convergence_report.json",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def replace_option_value(command: list[str], option: str, value: str) -> tuple[list[str], bool]:
    updated = list(command)
    for index, item in enumerate(updated):
        if item == option and index + 1 < len(updated):
            updated[index + 1] = value
            return updated, True
        prefix = f"{option}="
        if item.startswith(prefix):
            updated[index] = f"{prefix}{value}"
            return updated, True
    return updated, False


def command_output_dir(command: list[str]) -> Path | None:
    for index, item in enumerate(command):
        if item == "--output-dir" and index + 1 < len(command):
            return resolve_workspace_path(command[index + 1])
        if item.startswith("--output-dir="):
            return resolve_workspace_path(item.split("=", 1)[1])
    return None


def default_replay_output_dir(command_path: Path, case_id: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_case = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id) or "case"
    return ROOT / "runs" / "replay" / f"{safe_case}_{stamp}"


def build_replay_command(payload: dict[str, Any], command_path: Path, output_dir: Path | None, allow_original_output: bool) -> tuple[list[str], Path | None]:
    command = payload.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("case_command.json must contain a string-list command")
    original_output = command_output_dir(command)
    if output_dir is None and original_output is not None and not allow_original_output:
        output_dir = default_replay_output_dir(command_path, str(payload.get("case_id") or command_path.parent.name))
    if output_dir is not None:
        output_dir = output_dir.resolve()
        command, replaced = replace_option_value(command, "--output-dir", str(output_dir))
        if not replaced:
            raise ValueError("Command does not contain --output-dir; pass --allow-original-output to replay in place")
    elif original_output is not None:
        output_dir = original_output
    return command, output_dir


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def looks_path_like(value: str) -> bool:
    text = str(value)
    return "/" in text or "\\" in text or text.startswith("@runs/") or text.startswith("runs/")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_csv(source: Path, replay: Path, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    source_rows = read_csv_rows(source)
    replay_rows = read_csv_rows(replay)
    source_columns = list(source_rows[0].keys()) if source_rows else []
    replay_columns = list(replay_rows[0].keys()) if replay_rows else []
    common_columns = [column for column in source_columns if column in set(replay_columns)]
    numeric_compared = 0
    string_compared = 0
    string_mismatch_count = 0
    path_string_mismatch_count = 0
    numeric_fail_count = 0
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    examples: list[dict[str, Any]] = []
    for row_index, (left, right) in enumerate(zip(source_rows, replay_rows)):
        for column in common_columns:
            left_value = left.get(column, "")
            right_value = right.get(column, "")
            left_number = maybe_float(left_value)
            right_number = maybe_float(right_value)
            if left_number is not None and right_number is not None:
                numeric_compared += 1
                abs_delta = abs(left_number - right_number)
                denom = max(abs(left_number), abs(right_number), 1.0)
                rel_delta = abs_delta / denom
                max_abs_delta = max(max_abs_delta, abs_delta)
                max_rel_delta = max(max_rel_delta, rel_delta)
                if abs_delta > abs_tol and rel_delta > rel_tol:
                    numeric_fail_count += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "type": "numeric",
                                "row": row_index,
                                "column": column,
                                "source": left_number,
                                "replay": right_number,
                                "abs_delta": abs_delta,
                                "rel_delta": rel_delta,
                            }
                        )
                continue
            string_compared += 1
            if str(left_value) != str(right_value):
                if looks_path_like(str(left_value)) or looks_path_like(str(right_value)):
                    path_string_mismatch_count += 1
                else:
                    string_mismatch_count += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "type": "string",
                                "row": row_index,
                                "column": column,
                                "source": left_value,
                                "replay": right_value,
                            }
                        )
    status = (
        "PASS"
        if len(source_rows) == len(replay_rows)
        and set(source_columns) == set(replay_columns)
        and numeric_fail_count == 0
        and string_mismatch_count == 0
        else "FAIL"
    )
    return {
        "type": "csv",
        "status": status,
        "source": str(source),
        "replay": str(replay),
        "source_rows": len(source_rows),
        "replay_rows": len(replay_rows),
        "source_columns": source_columns,
        "replay_columns": replay_columns,
        "numeric_compared": numeric_compared,
        "string_compared": string_compared,
        "numeric_fail_count": numeric_fail_count,
        "string_mismatch_count": string_mismatch_count,
        "path_string_mismatch_count": path_string_mismatch_count,
        "max_abs_delta": max_abs_delta,
        "max_rel_delta": max_rel_delta,
        "examples": examples,
    }


def compare_json(source: Path, replay: Path) -> dict[str, Any]:
    source_payload = read_json(source)
    replay_payload = read_json(replay)
    checked_keys = [key for key in ("schema", "status", "convergence_status") if key in source_payload or key in replay_payload]
    mismatches = [
        {"key": key, "source": source_payload.get(key), "replay": replay_payload.get(key)}
        for key in checked_keys
        if source_payload.get(key) != replay_payload.get(key)
    ]
    return {
        "type": "json",
        "status": "PASS" if not mismatches else "FAIL",
        "source": str(source),
        "replay": str(replay),
        "checked_keys": checked_keys,
        "mismatches": mismatches,
    }


def compare_replay_outputs(source_dir: Path, replay_dir: Path, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    csv_results = []
    json_results = []
    missing = []
    for name in COMPARABLE_CSV_NAMES:
        source = source_dir / name
        replay = replay_dir / name
        if source.exists() or replay.exists():
            if source.exists() and replay.exists():
                csv_results.append(compare_csv(source, replay, abs_tol, rel_tol))
            else:
                missing.append({"artifact": name, "source_exists": source.exists(), "replay_exists": replay.exists()})
    for name in COMPARABLE_JSON_NAMES:
        source = source_dir / name
        replay = replay_dir / name
        if source.exists() or replay.exists():
            if source.exists() and replay.exists():
                json_results.append(compare_json(source, replay))
            else:
                missing.append({"artifact": name, "source_exists": source.exists(), "replay_exists": replay.exists()})
    failures = [item for item in [*csv_results, *json_results] if item.get("status") != "PASS"]
    status = "PASS" if (csv_results or json_results) and not failures and not missing else "FAIL"
    return {
        "schema": "pixel_workbench_replay_comparison_v1",
        "status": status,
        "source_dir": str(source_dir),
        "replay_dir": str(replay_dir),
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "csv_results": csv_results,
        "json_results": json_results,
        "missing_artifacts": missing,
        "failure_count": len(failures) + len(missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Pixel Workbench case_command.json artifact.")
    parser.add_argument("case_command_json", type=Path, help="Path to a case_command.json artifact.")
    parser.add_argument("--output-dir", type=Path, help="Replay output directory. Defaults to runs/replay/<case>_<timestamp>.")
    parser.add_argument("--allow-original-output", action="store_true", help="Allow replaying into the command's original output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the replay manifest without running the command.")
    parser.add_argument("--timeout-sec", type=float, default=None, help="Optional subprocess timeout in seconds.")
    parser.add_argument("--compare-source", action="store_true", help="Compare replay CSV/JSON outputs against the source case directory.")
    parser.add_argument("--abs-tol", type=float, default=1.0e-12, help="Absolute tolerance for numeric CSV replay comparison.")
    parser.add_argument("--rel-tol", type=float, default=1.0e-9, help="Relative tolerance for numeric CSV replay comparison.")
    args = parser.parse_args()

    command_path = resolve_workspace_path(args.case_command_json)
    payload = read_json(command_path)
    if payload.get("schema") != "pixel_workbench_suite_case_command_v1":
        raise ValueError(f"Unsupported command schema: {payload.get('schema')!r}")
    command, output_dir = build_replay_command(payload, command_path, args.output_dir, args.allow_original_output)
    cwd = resolve_workspace_path(payload.get("cwd") or ROOT)
    manifest = {
        "schema": "pixel_workbench_case_command_replay_v1",
        "source_case_command": str(command_path),
        "source_case_id": payload.get("case_id"),
        "source_runner": payload.get("runner"),
        "cwd": str(cwd),
        "command": command,
        "output_dir": str(output_dir) if output_dir else None,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False)
    manifest.update(
        {
            "return_code": completed.returncode,
            "elapsed_sec": time.time() - started,
            "stdout_tail": (completed.stdout or "").splitlines()[-80:],
            "status": "PASS" if completed.returncode == 0 else "FAIL",
        }
    )
    if output_dir is not None:
        if args.compare_source:
            comparison = compare_replay_outputs(command_path.parent, output_dir, args.abs_tol, args.rel_tol)
            comparison_path = output_dir / "replay_comparison.json"
            write_json(comparison_path, comparison)
            manifest["replay_comparison"] = str(comparison_path)
            manifest["replay_comparison_status"] = comparison["status"]
        manifest_path = output_dir / "replay_manifest.json"
        write_json(manifest_path, manifest)
        manifest["replay_manifest"] = str(manifest_path)
        write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
