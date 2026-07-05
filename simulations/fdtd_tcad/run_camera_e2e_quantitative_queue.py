#!/usr/bin/env python3
"""Run point-sized CameraE2E quantitative queue items with resume support."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"

RUN_LOG_COLUMNS = [
    "queue_id",
    "slug",
    "solver",
    "color",
    "field_case",
    "wavelength_nm",
    "target_resolution_px_per_um",
    "fdtd_cell_volume_um3",
    "estimated_volume_factor",
    "fdtd_domain_factor",
    "status",
    "returncode",
    "duration_s",
    "estimated_hours",
    "command",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def command_option(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return command[index + 1]


def command_with_child_timeout(command: list[str], child_timeout_s: float) -> list[str]:
    """Propagate queue timeout into child sweep scripts that launch Meep jobs."""
    if "--timeout-s" in command:
        return command
    sweep_scripts = {"run_camera_e2e_fdtd_field_sweep.py", "run_camera_e2e_crosstalk_sweep.py"}
    if any(Path(part).name in sweep_scripts for part in command):
        return [*command, "--timeout-s", str(float(child_timeout_s))]
    return command


def child_timeout_s(args: argparse.Namespace) -> float:
    if args.child_timeout_s > 0:
        return args.child_timeout_s
    if args.timeout_s <= 0:
        return 0.0
    return max(1.0, float(args.timeout_s) - float(args.timeout_grace_s))


def infer_run_status(row: dict[str, str], command: list[str], returncode: int | str) -> str:
    if returncode == "TIMEOUT":
        return "TIMEOUT"
    if returncode != 0:
        return "FAIL"
    output_dir_text = command_option(command, "--output-dir")
    if not output_dir_text:
        return "DONE"
    output_dir = Path(output_dir_text)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    solver = row.get("solver", "")
    report_path = output_dir / ("crosstalk_sweep_report.json" if solver == "crosstalk" else "fdtd_field_sweep_report.json")
    if not report_path.exists():
        return "DONE"
    report = read_json(report_path)
    if int(report.get("timeout_job_count") or 0) > 0:
        return "TIMEOUT"
    if int(report.get("failed_job_count") or 0) > 0 and int(report.get("summary_row_count") or 0) <= 0:
        return "FAIL"
    if int(report.get("completed_job_count") or 0) > 0:
        return "DONE"
    if int(report.get("summary_row_count") or 0) <= 0:
        return "CHECK"
    summary_rel = report.get("outputs", {}).get("summary_csv", "")
    summary_path = ROOT / summary_rel if summary_rel else output_dir / (
        "crosstalk_sweep_summary.csv" if solver == "crosstalk" else "fdtd_field_sweep_summary.csv"
    )
    if summary_path.exists():
        for summary_row in read_csv(summary_path):
            if str(summary_row.get("convergence_status", "")).upper() == "RESOURCE_LIMIT":
                return "SKIPPED_RESOURCE"
    return "CHECK"


def completed_queue_ids(package_dir: Path, *, include_failed: bool) -> set[str]:
    path = package_dir / "camera_e2e_quantitative_merged_summary.csv"
    if not path.exists():
        return set()
    completed = set()
    for row in read_csv(path):
        gate = str(row.get("solver_gate", "")).upper()
        if gate == "FAIL" and not include_failed:
            continue
        completed.add(str(row.get("queue_id", "")))
    return completed


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.package_dir / "camera_e2e_quantitative_point_queue.csv")
    if args.queue_ids:
        requested = {item.strip() for item in args.queue_ids.split(",") if item.strip()}
        rows = [row for row in rows if row.get("queue_id") in requested]
    if args.slugs:
        requested = {item.strip() for item in args.slugs.split(",") if item.strip()}
        rows = [row for row in rows if row.get("slug") in requested]
    if args.solvers:
        requested = {item.strip() for item in args.solvers.split(",") if item.strip()}
        rows = [row for row in rows if row.get("solver") in requested]
    if args.colors:
        requested = {item.strip() for item in args.colors.split(",") if item.strip()}
        rows = [row for row in rows if row.get("color") in requested]
    if args.field_cases:
        requested = {item.strip() for item in args.field_cases.split(",") if item.strip()}
        rows = [row for row in rows if row.get("field_case") in requested]
    if args.wavelengths_nm:
        requested = {item.strip() for item in args.wavelengths_nm.split(",") if item.strip()}
        rows = [row for row in rows if row.get("wavelength_nm") in requested]
    if not args.no_skip_completed:
        completed = completed_queue_ids(args.package_dir, include_failed=args.skip_failed)
        rows = [row for row in rows if row.get("queue_id") not in completed]
    if args.max_points > 0:
        rows = rows[: args.max_points]
    return rows


def refresh_package(package_dir: Path) -> None:
    subprocess.run(
        ["python3", "merge_camera_e2e_quantitative_points.py", "--package-dir", str(package_dir)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["python3", "build_camera_e2e_sensor_luts.py", "--major-only", "--output-dir", str(package_dir)],
        cwd=ROOT,
        check=True,
    )


def run_queue(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    logs = []
    start_all = time.time()
    per_job_child_timeout_s = child_timeout_s(args)
    for row in rows:
        if args.max_seconds > 0 and time.time() - start_all >= args.max_seconds:
            break
        command = command_with_child_timeout(shlex.split(row["command"]), per_job_child_timeout_s)
        started = time.time()
        if args.dry_run:
            status = "DRY_RUN"
            returncode: int | str = ""
        else:
            try:
                result = subprocess.run(command, cwd=ROOT, timeout=args.timeout_s, check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = "TIMEOUT"
            status = infer_run_status(row, command, returncode)
            if args.refresh_package:
                refresh_package(args.package_dir)
        logs.append(
            {
                "queue_id": row.get("queue_id", ""),
                "slug": row.get("slug", ""),
                "solver": row.get("solver", ""),
                "color": row.get("color", ""),
                "field_case": row.get("field_case", ""),
                "wavelength_nm": row.get("wavelength_nm", ""),
                "target_resolution_px_per_um": row.get("target_resolution_px_per_um", ""),
                "fdtd_cell_volume_um3": row.get("fdtd_cell_volume_um3", ""),
                "estimated_volume_factor": row.get("estimated_volume_factor", ""),
                "fdtd_domain_factor": row.get("fdtd_domain_factor", ""),
                "status": status,
                "returncode": returncode,
                "duration_s": round(time.time() - started, 3),
                "estimated_hours": row.get("estimated_hours", ""),
                "command": shlex.join(command),
            }
        )
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--queue-ids", default="")
    parser.add_argument("--slugs", default="")
    parser.add_argument("--solvers", default="")
    parser.add_argument("--colors", default="")
    parser.add_argument("--field-cases", default="")
    parser.add_argument("--wavelengths-nm", default="")
    parser.add_argument("--max-points", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    parser.add_argument("--child-timeout-s", type=float, default=0.0)
    parser.add_argument("--timeout-grace-s", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--no-refresh-package", dest="refresh_package", action="store_false")
    parser.set_defaults(refresh_package=True)
    args = parser.parse_args()
    args.package_dir = args.package_dir.resolve()
    rows = selected_rows(args)
    logs = run_queue(args, rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = args.package_dir / "queue_run_logs"
    write_csv(log_dir / f"queue_run_{stamp}.csv", logs, RUN_LOG_COLUMNS)
    report = {
        "schema": "camera_e2e_quantitative_queue_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "timeout_s": args.timeout_s,
        "child_timeout_s": child_timeout_s(args),
        "timeout_grace_s": args.timeout_grace_s,
        "selected_count": len(rows),
        "executed_count": len(logs),
        "run_log_csv": repo_rel(log_dir / f"queue_run_{stamp}.csv"),
        "logs": logs,
    }
    write_json(log_dir / f"queue_run_{stamp}.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
