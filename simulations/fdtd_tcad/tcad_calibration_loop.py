#!/usr/bin/env python3
"""Calibration loop for the open-source TCAD framework.

The loop treats DEVSIM runs as black-box simulations, reads their summary JSON,
and fits configurable CLI parameters to measured target currents/phases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


TARGET_PREFIX = "target_"
LEGACY_WEIGHT_COLUMNS = {
    "total_photo_current_a_per_cm": "weight_current",
    "split_phase_x_proxy": "weight_phase",
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "tcad_calibration_config_v1":
        raise ValueError("expected tcad_calibration_config_v1")
    return config


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def target_metric_name(column: str) -> str:
    if not column.startswith(TARGET_PREFIX):
        raise ValueError(f"target metric column must start with {TARGET_PREFIX}: {column}")
    return column[len(TARGET_PREFIX) :]


def target_weight(row: dict[str, str], metric: str) -> float:
    for key in (f"weight_{metric}", LEGACY_WEIGHT_COLUMNS.get(metric, "")):
        if key and row.get(key, "") not in {"", None}:
            value = finite_float(row.get(key), 1.0)
            if math.isfinite(value):
                return value
    return 1.0


def load_targets(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            metrics = []
            for column, raw_value in row.items():
                if not column.startswith(TARGET_PREFIX):
                    continue
                if column in {"target_source"} or raw_value in {"", None}:
                    continue
                value = finite_float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"{path}: non-finite {column} for case {row.get('case')}")
                metric = target_metric_name(column)
                metrics.append(
                    {
                        "metric": metric,
                        "column": column,
                        "target": value,
                        "weight": target_weight(row, metric),
                    }
                )
            if not metrics:
                raise ValueError(f"{path}: target row for {row.get('case')} has no target_* metrics")
            rows.append(
                {
                    "case": row["case"],
                    "wavelength_nm": float(row["wavelength_nm"]),
                    "metrics": metrics,
                    "target_source": row.get("target_source", ""),
                    "notes": row.get("notes", ""),
                }
            )
    if not rows:
        raise ValueError(f"empty targets CSV: {path}")
    return rows


def project_root_for(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def resolve_config_path(config_path: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    root = project_root_for(config_path)
    # Keep virtualenv executable symlinks intact. Path.resolve() follows
    # .venv/bin/python -> system python on macOS and loses installed packages.
    return (root / path).absolute()


def parameter_vector(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = config["parameters"]
    initial = np.asarray([float(item["initial"]) for item in params], dtype=float)
    lower = np.asarray([float(item["lower"]) for item in params], dtype=float)
    upper = np.asarray([float(item["upper"]) for item in params], dtype=float)
    return initial, lower, upper


def parameter_values(config: dict[str, Any], vector: np.ndarray) -> dict[str, float]:
    return {
        item["name"]: float(value)
        for item, value in zip(config["parameters"], vector)
    }


def command_for_target(
    config_path: Path,
    config: dict[str, Any],
    target: dict[str, Any],
    params: dict[str, float],
    eval_dir: Path,
) -> list[str]:
    simulator = config["simulator"]
    python = str(resolve_config_path(config_path, simulator["python"]))
    script = str(resolve_config_path(config_path, simulator["script"]))
    command = [
        python,
        script,
        "--generation-map-npz",
        str(resolve_config_path(config_path, simulator["generation_map_npz"])),
        "--generation-profile-case",
        target["case"],
        "--generation-profile-wavelength-nm",
        f"{target['wavelength_nm']:.12g}",
        "--electrical-model",
        simulator.get("electrical_model", "proxy-pinned-split-pd"),
        "--output-dir",
        str(eval_dir),
    ]
    for item in config["parameters"]:
        command.extend([item["cli"], f"{params[item['name']]:.16g}"])
    for key, value in config.get("fixed_cli_args", {}).items():
        if key in {"--generation-profile-case", "--generation-profile-wavelength-nm", "--electrical-model"}:
            continue
        command.extend([key, str(value)])
    return command


def nested_summary_value(summary: dict[str, Any], metric: str) -> float:
    current: Any = summary
    for key in metric.split("__"):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(metric)
        current = current[key]
    value = finite_float(current)
    if not math.isfinite(value):
        raise ValueError(f"metric {metric} is not finite: {current}")
    return value


def summary_metric(summary: dict[str, Any], metric: str) -> float:
    if metric == "total_photo_current_a_per_cm":
        return float(summary["left_photo_delta_a_per_cm"]) + float(
            summary["right_photo_delta_a_per_cm"]
        )
    if metric == "split_phase_x_proxy":
        return float(summary["photo_split_phase_x_proxy"])
    if metric == "dark_total_cathode_current_a_per_cm":
        return float(summary["dark"]["total_cathode_current_a_per_cm"])
    if metric == "dark_total_cathode_current_abs_a_per_cm":
        return abs(float(summary["dark"]["total_cathode_current_a_per_cm"]))
    if metric == "terminal_balance_a_per_cm":
        return float(summary["terminal_current_balance_illuminated_a_per_cm"])
    if metric == "interface_recombination_coeff_max_s1":
        return float(summary["interface_trap_summary"]["recombination_coeff_max_s1"])
    if "__" in metric:
        return nested_summary_value(summary, metric)
    raise KeyError(f"unsupported calibration target metric: {metric}")


def run_one(
    config_path: Path,
    config: dict[str, Any],
    target: dict[str, Any],
    params: dict[str, float],
    eval_root: Path,
    eval_index: int,
) -> dict[str, Any]:
    case_dir = eval_root / f"eval_{eval_index:03d}_{target['case']}"
    command = command_for_target(config_path, config, target, params, case_dir)
    result = subprocess.run(
        command,
        cwd=project_root_for(config_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(f"simulation failed for {target['case']}:\n{tail}")
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sim_metrics = {
        item["metric"]: summary_metric(summary, item["metric"])
        for item in target["metrics"]
    }
    row = {
        "case": target["case"],
        "wavelength_nm": target["wavelength_nm"],
        "summary_json": str(summary_path),
        "sim_metrics": sim_metrics,
        "terminal_balance_a_per_cm": summary["terminal_current_balance_illuminated_a_per_cm"],
    }
    if "total_photo_current_a_per_cm" in sim_metrics:
        row["sim_total_photo_current_a_per_cm"] = sim_metrics["total_photo_current_a_per_cm"]
    if "split_phase_x_proxy" in sim_metrics:
        row["sim_split_phase_x_proxy"] = sim_metrics["split_phase_x_proxy"]
    return row


def residual_scale(config: dict[str, Any], metric: str, target_value: float) -> float:
    residuals = config.get("residuals", {})
    metric_scales = residuals.get("metric_scales", {})
    if metric in metric_scales:
        return max(abs(float(metric_scales[metric])), 1.0e-30)
    if metric == "total_photo_current_a_per_cm":
        floor = float(residuals.get("current_relative_floor_a_per_cm", 1e-12))
        return max(abs(target_value), floor)
    if metric == "dark_total_cathode_current_abs_a_per_cm":
        floor = float(residuals.get("dark_current_relative_floor_a_per_cm", 1e-15))
        return max(abs(target_value), floor)
    if metric in {"split_phase_x_proxy", "split_phase_x_total_current"}:
        return max(abs(float(residuals.get("phase_absolute_scale", 0.05))), 1.0e-30)
    if metric.endswith("_a_per_cm"):
        floor = float(residuals.get("current_relative_floor_a_per_cm", 1e-12))
        return max(abs(target_value), floor)
    return max(abs(target_value), float(residuals.get("generic_relative_floor", 1e-30)))


def residual_vector(
    config_path: Path,
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    vector: np.ndarray,
    eval_root: Path,
    history: list[dict[str, Any]],
) -> np.ndarray:
    params = parameter_values(config, vector)
    eval_index = len(history)
    rows = []
    residuals = []
    residual_terms = []
    for target in targets:
        row = run_one(config_path, config, target, params, eval_root, eval_index)
        rows.append(row)
        for metric_target in target["metrics"]:
            metric = metric_target["metric"]
            target_value = float(metric_target["target"])
            simulated = float(row["sim_metrics"][metric])
            scale = residual_scale(config, metric, target_value)
            residual = (simulated - target_value) / scale * float(metric_target["weight"])
            residuals.append(residual)
            residual_terms.append(
                {
                    "case": target["case"],
                    "wavelength_nm": target["wavelength_nm"],
                    "metric": metric,
                    "target": target_value,
                    "simulated": simulated,
                    "scale": scale,
                    "weight": float(metric_target["weight"]),
                    "residual": float(residual),
                }
            )
    entry = {
        "eval_index": eval_index,
        "parameters": params,
        "rows": rows,
        "residual_norm": float(np.linalg.norm(residuals)),
        "residuals": [float(value) for value in residuals],
        "residual_terms": residual_terms,
    }
    history.append(entry)
    print(json.dumps(entry, indent=2))
    return np.asarray(residuals, dtype=float)


def write_history(output_dir: Path, history: list[dict[str, Any]], result: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in history:
        flat = {
            "eval_index": entry["eval_index"],
            "residual_norm": entry["residual_norm"],
        }
        flat.update(entry["parameters"])
        rows.append(flat)
    with (output_dir / "calibration_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "tcad_calibration_result_v1",
        "success": bool(result.success),
        "message": result.message,
        "best_parameters": {},
        "optimizer_x": [float(value) for value in result.x],
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "history": history,
    }
    (output_dir / "calibration_result.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tcad_calibration_example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tcad_calibration_example"))
    parser.add_argument("--max-evals", type=int, default=6)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    targets = load_targets(resolve_config_path(config_path, config["targets_csv"]))
    initial, lower, upper = parameter_vector(config)
    history: list[dict[str, Any]] = []
    eval_root = resolve_config_path(config_path, config["simulator"]["output_root"])

    result = least_squares(
        lambda vector: residual_vector(
            config_path,
            config,
            targets,
            vector,
            eval_root,
            history,
        ),
        initial,
        bounds=(lower, upper),
        max_nfev=args.max_evals,
        xtol=1e-3,
        ftol=1e-3,
        gtol=1e-3,
    )
    best = parameter_values(config, result.x)
    output_dir = args.output_dir.resolve()
    write_history(output_dir, history, result)
    result_path = output_dir / "calibration_result.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["best_parameters"] = best
    result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
