#!/usr/bin/env python3
"""Run materialized image-sensor variant stages with logging and refresh.

This is a local sequential runner, not a scheduler. It executes command rows
already materialized by image_sensor_variant_builder.py, records logs, verifies
expected output files, and optionally refreshes comparison/run-manager/Studio
artifacts after successful work.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_sensor_run_manager import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_STUDIO_OUTPUT_DIR,
    DEFAULT_VARIANT_MANIFEST,
    STAGE_ORDER,
    commands_for_stage,
    expected_stage_paths,
    freshness_inputs_for_stage,
    stage_status,
    stage_freshness,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "orchestrator_logs"
HEAVY_STAGES = {"meep_fdtd", "convergence_gate"}
OUTPUT_FLAGS = {
    "--output-dir",
    "--output",
    "--output-json",
    "--output-csv",
    "--output-html",
    "--output-path",
    "--log-dir",
}
PATH_SUFFIXES = {
    ".csv",
    ".dat",
    ".h5",
    ".html",
    ".json",
    ".msh",
    ".npz",
    ".py",
    ".txt",
    ".vtk",
    ".vtu",
    ".yaml",
    ".yml",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel_from_root(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_command_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def token_looks_like_path(value: str) -> bool:
    if not value or value.startswith("-") or "://" in value:
        return False
    if "/" in value or value.startswith("."):
        return True
    return Path(value).suffix.lower() in PATH_SUFFIXES


def command_flag_values(tokens: list[str]) -> dict[str, list[str]]:
    values_by_flag: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        flag = token.split("=", 1)[0]
        values: list[str] = []
        if "=" in token:
            values.append(token.split("=", 1)[1])
            index += 1
        else:
            index += 1
            while index < len(tokens) and not tokens[index].startswith("--"):
                values.append(tokens[index])
                index += 1
        values_by_flag.setdefault(flag, []).extend(values)
    return values_by_flag


def add_check(checks: list[dict[str, Any]], name: str, status: str, details: str, evidence: Any = "") -> None:
    checks.append({"name": name, "status": status, "details": details, "evidence": evidence})


def worst_status(checks: list[dict[str, Any]]) -> str:
    statuses = {check.get("status") for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def check_executable(tokens: list[str], checks: list[dict[str, Any]]) -> None:
    if not tokens:
        add_check(checks, "command_parse", "FAIL", "Command is empty.")
        return
    executable = tokens[0]
    if token_looks_like_path(executable):
        path = resolve_command_path(executable)
        add_check(
            checks,
            "executable_exists",
            "PASS" if path.exists() else "FAIL",
            f"Executable path {'exists' if path.exists() else 'is missing'}: {rel_from_root(path)}",
            str(path),
        )
    else:
        found = shutil.which(executable)
        add_check(
            checks,
            "executable_on_path",
            "PASS" if found else "FAIL",
            f"Executable {executable!r} {'was found' if found else 'was not found'} on PATH.",
            found or executable,
        )


def check_python_script(tokens: list[str], checks: list[dict[str, Any]]) -> None:
    script_token = next((token for token in tokens if token.endswith(".py")), "")
    if not script_token:
        return
    path = resolve_command_path(script_token)
    add_check(
        checks,
        "script_exists",
        "PASS" if path.exists() else "FAIL",
        f"Python script {'exists' if path.exists() else 'is missing'}: {rel_from_root(path)}",
        str(path),
    )


def check_micromamba_env(tokens: list[str], checks: list[dict[str, Any]]) -> None:
    if not tokens or Path(tokens[0]).name != "micromamba":
        return
    if "run" not in tokens:
        add_check(checks, "micromamba_run_mode", "WARN", "micromamba command does not include run mode.")
        return
    prefix = ""
    for flag in ("-p", "--prefix"):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                prefix = tokens[index + 1]
            break
    if not prefix:
        add_check(checks, "micromamba_prefix", "WARN", "micromamba run command has no explicit prefix.")
        return
    if not Path(prefix).is_absolute() and "/" not in prefix and "\\" not in prefix:
        add_check(
            checks,
            "micromamba_prefix_path_semantics",
            "FAIL",
            f"micromamba -p value {prefix!r} has no filesystem separator, so micromamba treats it as an env name. Use an absolute path or './{prefix}'.",
            prefix,
        )
        return
    path = resolve_command_path(prefix)
    add_check(
        checks,
        "micromamba_env_exists",
        "PASS" if path.exists() and path.is_dir() else "FAIL",
        f"micromamba environment prefix {'exists' if path.exists() else 'is missing'}: {rel_from_root(path)}",
        str(path),
    )


def check_flag_paths(flags: dict[str, list[str]], checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_paths: list[dict[str, Any]] = []
    output_paths: list[dict[str, Any]] = []
    for flag, values in flags.items():
        for value in values:
            if flag in OUTPUT_FLAGS:
                path = resolve_command_path(value)
                parent = path.parent
                exists = path.exists()
                parent_exists = parent.exists()
                output_paths.append(
                    {
                        "flag": flag,
                        "path": str(path),
                        "exists": exists,
                        "parent_exists": parent_exists,
                    }
                )
                add_check(
                    checks,
                    f"output_path_{flag.lstrip('-').replace('-', '_')}",
                    "PASS" if parent_exists else "WARN",
                    f"Output target parent {'exists' if parent_exists else 'is missing'}: {rel_from_root(parent)}",
                    str(path),
                )
                continue
            if not token_looks_like_path(value):
                continue
            path = resolve_command_path(value)
            exists = path.exists()
            input_paths.append({"flag": flag, "path": str(path), "exists": exists})
            add_check(
                checks,
                f"input_path_{flag.lstrip('-').replace('-', '_')}",
                "PASS" if exists else "FAIL",
                f"Input path {'exists' if exists else 'is missing'}: {rel_from_root(path)}",
                str(path),
            )
            if flag == "--stack-config" and exists:
                check_stack_config_materials(path, checks)
    return input_paths, output_paths


def check_stack_config_materials(stack_config_path: Path, checks: list[dict[str, Any]]) -> None:
    try:
        stack = read_json(stack_config_path)
    except (OSError, json.JSONDecodeError) as exc:
        add_check(
            checks,
            "stack_config_parse",
            "FAIL",
            f"Could not parse stack config for material table checks: {exc}",
            str(stack_config_path),
        )
        return
    for role, spec in stack.get("materials", {}).items():
        if not isinstance(spec, dict) or not spec.get("nk_table"):
            continue
        raw_path = Path(str(spec["nk_table"]))
        table_path = raw_path if raw_path.is_absolute() else (stack_config_path.parent / raw_path).resolve()
        add_check(
            checks,
            f"stack_material_{role}_nk_table",
            "PASS" if table_path.exists() else "FAIL",
            f"Material {role} n,k table {'exists' if table_path.exists() else 'is missing'}: {rel_from_root(table_path)}",
            str(table_path),
        )


def runtime_hints(stage: str, flags: dict[str, list[str]]) -> dict[str, Any]:
    hints: dict[str, Any] = {"stage": stage, "heavy_stage": stage in HEAVY_STAGES}
    wavelengths = ",".join(flags.get("--wavelengths-nm", []))
    cases = ",".join(flags.get("--cases", []))
    if wavelengths:
        hints["wavelength_count"] = len([item for item in wavelengths.split(",") if item])
        hints["wavelengths_nm"] = wavelengths
    if cases:
        hints["case_count"] = len([item for item in cases.split(",") if item])
        hints["cases"] = cases
    for flag, key in [
        ("--resolution", "resolution"),
        ("--after-source-time", "after_source_time"),
        ("--mode", "mode"),
        ("--split-mode", "split_mode"),
    ]:
        if flags.get(flag):
            hints[key] = flags[flag][-1]
    return hints


def preflight_command(command: dict[str, str], stage: str) -> dict[str, Any]:
    command_text = command.get("command", "")
    checks: list[dict[str, Any]] = []
    try:
        tokens = shlex.split(command_text)
        add_check(checks, "command_parse", "PASS", "Command parsed with shlex.")
    except ValueError as exc:
        tokens = command_text.split()
        add_check(checks, "command_parse", "WARN", f"shlex parse failed; fell back to whitespace split: {exc}")
    check_executable(tokens, checks)
    check_micromamba_env(tokens, checks)
    check_python_script(tokens, checks)
    flags = command_flag_values(tokens)
    input_paths, output_paths = check_flag_paths(flags, checks)
    missing_inputs = [item for item in input_paths if not item["exists"]]
    missing_outputs_parent = [item for item in output_paths if not item["parent_exists"]]
    status = worst_status(checks)
    return {
        "command_id": command.get("id", ""),
        "stage": stage,
        "status": status,
        "check_count": len(checks),
        "fail_count": sum(1 for check in checks if check.get("status") == "FAIL"),
        "warn_count": sum(1 for check in checks if check.get("status") == "WARN"),
        "missing_input_count": len(missing_inputs),
        "missing_output_parent_count": len(missing_outputs_parent),
        "runtime_hints": runtime_hints(stage, flags),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "checks": checks,
    }


def summarize_preflight(preflights: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [check for preflight in preflights for check in preflight.get("checks", [])]
    status = worst_status(checks) if checks else "WARN"
    return {
        "status": status,
        "command_count": len(preflights),
        "check_count": len(checks),
        "fail_count": sum(1 for check in checks if check.get("status") == "FAIL"),
        "warn_count": sum(1 for check in checks if check.get("status") == "WARN"),
    }


def select_variants(manifest: dict[str, Any], names: list[str], all_variants: bool) -> list[dict[str, Any]]:
    variants = list(manifest.get("variants", []))
    if all_variants:
        return [variant for variant in variants if variant.get("required_stages")]
    if names:
        selected = []
        missing = []
        by_id = {variant.get("id"): variant for variant in variants}
        for name in names:
            if name in by_id:
                selected.append(by_id[name])
            else:
                missing.append(name)
        if missing:
            raise SystemExit(f"unknown variant id(s): {', '.join(missing)}")
        return selected
    for variant in variants:
        if variant.get("required_stages"):
            return [variant]
    return []


def selected_required_stages(
    variant: dict[str, Any],
    requested_stages: list[str],
    next_missing: bool,
    next_stale: bool,
    next_needed: bool,
) -> list[str]:
    required = [stage for stage in STAGE_ORDER if stage in set(variant.get("required_stages", []))]
    if requested_stages:
        unknown = [stage for stage in requested_stages if stage not in STAGE_ORDER]
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(unknown)}")
        return [stage for stage in required if stage in set(requested_stages)]
    if not (next_missing or next_stale or next_needed):
        return required
    for stage in required:
        state = stage_expected_summary(variant, stage, required)
        if next_missing and state["status"] != "complete":
            return [stage]
        if next_stale and state["freshness"]["freshness"] == "stale":
            return [stage]
        if next_needed and (state["status"] != "complete" or state["freshness"]["freshness"] == "stale"):
            return [stage]
    return []


def stage_expected_summary(variant: dict[str, Any], stage: str, required_stages: list[str]) -> dict[str, Any]:
    expected = expected_stage_paths(variant, stage)
    status = stage_status(expected)
    commands = commands_for_stage(variant, stage)
    freshness_inputs = freshness_inputs_for_stage(variant, stage, commands, required_stages)
    freshness = stage_freshness(expected, freshness_inputs, status)
    return {
        "stage": stage,
        "status": status,
        "freshness": freshness,
        "expected": expected,
        "missing": [item for item in expected if not item.get("exists")],
        "complete": [item for item in expected if item.get("exists")],
        "freshness_inputs": [str(path) for path in freshness_inputs],
    }


def build_plan(
    variants: list[dict[str, Any]],
    requested_stages: list[str],
    next_missing: bool,
    next_stale: bool,
    next_needed: bool,
    include_heavy: bool,
    rerun_complete: bool,
    force_downstream: bool,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for variant in variants:
        blocked_by: list[str] = []
        required = [stage for stage in STAGE_ORDER if stage in set(variant.get("required_stages", []))]
        for stage in selected_required_stages(variant, requested_stages, next_missing, next_stale, next_needed):
            summary = stage_expected_summary(variant, stage, required)
            commands = commands_for_stage(variant, stage)
            preflights = [preflight_command(command, stage) for command in commands]
            preflight_summary = summarize_preflight(preflights)
            reason = ""
            action = "run"
            is_complete = summary["status"] == "complete"
            is_stale = summary["freshness"]["freshness"] == "stale"
            if stage in HEAVY_STAGES and not include_heavy:
                action = "skip"
                reason = "heavy stage skipped; pass --include-heavy to allow"
            elif is_complete and not is_stale and not rerun_complete:
                action = "skip"
                reason = "stage already complete and fresh"
            elif blocked_by and not force_downstream:
                action = "blocked"
                reason = f"missing upstream stage(s): {', '.join(blocked_by)}"
            elif not commands:
                action = "skip"
                reason = "no commands for stage"
            elif is_complete and is_stale:
                reason = summary["freshness"]["stale_reason"]
            elif is_complete and rerun_complete:
                reason = "rerun complete stage requested"
            plan.append(
                {
                    "variant_id": variant.get("id"),
                    "variant_label": variant.get("label", variant.get("id")),
                    "stage": stage,
                    "action": action,
                    "reason": reason,
                    "status_before": summary["status"],
                    "freshness_before": summary["freshness"]["freshness"],
                    "stale_reason_before": summary["freshness"]["stale_reason"],
                    "newest_input_mtime_before": summary["freshness"]["newest_input_mtime"],
                    "oldest_output_mtime_before": summary["freshness"]["oldest_output_mtime"],
                    "freshness_inputs_before": summary["freshness"]["freshness_inputs"],
                    "expected_missing_before": [item["label"] for item in summary["missing"]],
                    "command_count": len(commands),
                    "preflight_summary": preflight_summary,
                    "preflight": preflights,
                    "commands": commands,
                }
            )
            if (summary["status"] != "complete" or is_stale) and action != "run":
                blocked_by.append(stage)
    return plan


def command_log_path(log_dir: Path, run_id: str, variant_id: str, command: dict[str, str]) -> Path:
    safe_command_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in command.get("id", "cmd"))
    return log_dir / run_id / variant_id / f"{safe_command_id}.log"


def execute_command(command: dict[str, str], log_path: Path, timeout_s: int | None) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command['command']}\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command["command"],
                cwd=ROOT,
                shell=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            return_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True
            log.write(f"\nTIMEOUT after {timeout_s} seconds\n")
    elapsed = time.time() - started
    return {
        "command_id": command.get("id", ""),
        "stage": command.get("stage", ""),
        "label": command.get("label", ""),
        "command": command.get("command", ""),
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "log": str(log_path),
    }


def refresh_outputs(project_config: Path, variant_manifest: Path, output_dir: Path, studio_output_dir: Path) -> list[dict[str, Any]]:
    refresh_commands = [
        [
            ".tcad-env/bin/python",
            "image_sensor_variant_compare.py",
            "--config",
            str(project_config),
            "--variant-manifest",
            str(variant_manifest),
            "--output-dir",
            str(output_dir),
        ],
        [
            ".tcad-env/bin/python",
            "image_sensor_run_manager.py",
            "--config",
            str(project_config),
            "--variant-manifest",
            str(variant_manifest),
            "--output-dir",
            str(output_dir),
            "--studio-output-dir",
            str(studio_output_dir),
        ],
        [
            ".tcad-env/bin/python",
            "image_sensor_pixel_studio.py",
            "--config",
            str(project_config),
            "--output-dir",
            str(studio_output_dir),
        ],
    ]
    results = []
    for command in refresh_commands:
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        results.append(
            {
                "command": " ".join(command),
                "return_code": completed.returncode,
                "elapsed_s": time.time() - started,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            break
    return results


def build_run_record(
    run_id: str,
    mode: str,
    variants: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    executed_commands: list[dict[str, Any]],
    refresh_results: list[dict[str, Any]],
    failed: bool,
    project_config: Path,
    variant_manifest: Path,
    output_dir: Path,
    studio_output_dir: Path,
    phase: str,
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "mode": mode,
        "phase": phase,
        "selected_variants": [variant.get("id") for variant in variants],
        "plan_rows": len(plan),
        "planned_run_rows": sum(1 for item in plan if item["action"] == "run"),
        "planned_stale_rows": sum(1 for item in plan if item.get("freshness_before") == "stale"),
        "executed_command_count": len(executed_commands),
        "failed": failed,
        "refresh_ran": bool(refresh_results),
    }
    return {
        "schema": "image_sensor_variant_orchestrator_run_v1",
        "summary": summary,
        "project_config": str(project_config),
        "variant_manifest": str(variant_manifest),
        "output_dir": str(output_dir),
        "studio_output_dir": str(studio_output_dir),
        "plan": plan,
        "executed_commands": executed_commands,
        "refresh_results": refresh_results,
        "limitations": [
            "This is a local sequential runner, not a concurrent scheduler.",
            "Default dry-run mode avoids accidental heavy FDTD execution.",
            "Product LUT readiness remains controlled by the accuracy gate, not by successful orchestration.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_config = args.config.resolve()
    variant_manifest = args.variant_manifest.resolve()
    output_dir = args.output_dir.resolve()
    studio_output_dir = args.studio_output_dir.resolve()
    last_run_path = output_dir / "orchestrator_last_run.json"
    history_path = output_dir / "orchestrator_history.jsonl"
    manifest = read_json(variant_manifest)
    variants = select_variants(manifest, args.variant, args.all)
    if not variants:
        raise SystemExit("no runnable variants selected")

    run_id = timestamp()
    plan = build_plan(
        variants,
        args.stage,
        args.next_missing,
        args.next_stale,
        args.next_needed,
        args.include_heavy,
        args.rerun_complete,
        args.force_downstream,
    )
    executed_commands: list[dict[str, Any]] = []
    failed = False
    if args.execute:
        for item in plan:
            if item["action"] != "run":
                continue
            for command in item["commands"]:
                log_path = command_log_path(args.log_dir.resolve(), run_id, item["variant_id"], command)
                result = execute_command(command, log_path, args.timeout_s)
                executed_commands.append(result)
                if result["return_code"] != 0:
                    failed = True
                    break
            variant = next(variant for variant in variants if variant.get("id") == item["variant_id"])
            required = [stage for stage in STAGE_ORDER if stage in set(variant.get("required_stages", []))]
            after = stage_expected_summary(variant, item["stage"], required)
            item["status_after"] = after["status"]
            item["freshness_after"] = after["freshness"]["freshness"]
            item["stale_reason_after"] = after["freshness"]["stale_reason"]
            if failed and not args.keep_going:
                break
    refresh_results: list[dict[str, Any]] = []
    if args.execute and not failed and args.refresh:
        preliminary = build_run_record(
            run_id,
            "execute",
            variants,
            plan,
            executed_commands,
            refresh_results,
            failed,
            project_config,
            variant_manifest,
            output_dir,
            studio_output_dir,
            "pre_refresh",
        )
        write_json(last_run_path, preliminary)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.touch(exist_ok=True)
        refresh_results = refresh_outputs(project_config, variant_manifest, output_dir, studio_output_dir)
        failed = any(item["return_code"] != 0 for item in refresh_results)

    result = build_run_record(
        run_id,
        "execute" if args.execute else "dry_run",
        variants,
        plan,
        executed_commands,
        refresh_results,
        failed,
        project_config,
        variant_manifest,
        output_dir,
        studio_output_dir,
        "final",
    )
    summary = result["summary"]
    write_json(last_run_path, result)
    append_jsonl(history_path, {"summary": summary, "last_run": str(last_run_path)})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"last_run: {rel_from_root(last_run_path)}")
    if executed_commands:
        print(f"log_dir: {rel_from_root(args.log_dir.resolve() / run_id)}")
    if failed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--variant-manifest", type=Path, default=DEFAULT_VARIANT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--studio-output-dir", type=Path, default=DEFAULT_STUDIO_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--variant", action="append", default=[], help="Variant id to run; can be repeated.")
    parser.add_argument("--stage", action="append", default=[], help="Stage id to run; can be repeated.")
    parser.add_argument("--all", action="store_true", help="Select all materialized non-baseline variants.")
    next_group = parser.add_mutually_exclusive_group()
    next_group.add_argument("--next-missing", action="store_true", help="Only consider the first missing stage per variant.")
    next_group.add_argument("--next-stale", action="store_true", help="Only consider the first stale complete stage per variant.")
    next_group.add_argument(
        "--next-needed",
        action="store_true",
        help="Only consider the first missing, partial, or stale stage per variant.",
    )
    parser.add_argument("--include-heavy", action="store_true", help="Allow Meep/convergence stages.")
    parser.add_argument("--rerun-complete", action="store_true", help="Run complete stages again.")
    parser.add_argument("--force-downstream", action="store_true", help="Allow downstream stages despite missing upstream outputs.")
    parser.add_argument("--execute", action="store_true", help="Execute commands. Omit for dry-run.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a command failure.")
    parser.add_argument("--no-refresh", dest="refresh", action="store_false", help="Skip compare/run-manager/studio refresh.")
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.set_defaults(refresh=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
