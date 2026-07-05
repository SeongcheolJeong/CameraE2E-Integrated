#!/usr/bin/env python3
"""Check runtime transport/interface calibration controls.

This is not a measured calibration. It verifies that practical calibration
knobs such as lifetime, mobility, fixed charge, and interface-trap scales are
wired into the DEVSIM profile-PPD equations and materialize in solver outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "tcad_transport_sensitivity_reference"
DEFAULT_SCENARIOS = (
    "baseline;"
    "lifetime_0p25:--lifetime-scale=0.25;"
    "electron_mobility_0p50:--electron-mobility-scale=0.5;"
    "fixed_charge_1p50:--fixed-charge-scale=1.5;"
    "interface_recombination_2p00:--interface-trap-recombination-scale=2.0"
)
CSV_COLUMNS = [
    "scenario",
    "case",
    "wavelength_nm",
    "total_photo_current_a_per_cm",
    "relative_total_delta_to_baseline",
    "split_phase_x",
    "split_phase_delta_to_baseline",
    "electron_mobility_scale",
    "hole_mobility_scale",
    "lifetime_scale",
    "fixed_charge_scale",
    "interface_trap_density_scale",
    "interface_trap_recombination_scale",
    "electron_mobility_min_cm2_v_s",
    "electron_mobility_max_cm2_v_s",
    "tau_n_min_s",
    "tau_n_max_s",
    "recombination_coeff_max_s1",
    "summary_json",
]


def parse_cases(raw: str) -> list[dict[str, Any]]:
    cases = []
    for token in raw.split(","):
        if not token.strip():
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"case must be case:wavelength_nm, got {token}")
        cases.append({"case": parts[0], "wavelength_nm": float(parts[1])})
    if not cases:
        raise ValueError("at least one case is required")
    return cases


def parse_scenarios(raw: str) -> list[dict[str, Any]]:
    scenarios = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, args = token.split(":", 1)
            extra_args = shlex.split(args)
        else:
            name, extra_args = token, []
        scenarios.append({"name": name.strip(), "extra_args": extra_args})
    if not scenarios or scenarios[0]["name"] != "baseline":
        raise ValueError("first scenario must be baseline")
    return scenarios


def scale_from_summary(summary: dict[str, Any], name: str) -> float:
    transport = summary.get("transport_summary", {}).get("runtime_calibration_scales", {})
    interface = summary.get("interface_trap_summary", {}).get("runtime_calibration_scales", {})
    features = (
        summary.get("doping_summary", {})
        .get("feature_summary", {})
        .get("runtime_calibration_scales", {})
    )
    nested_features = (
        summary.get("doping_summary", {})
        .get("feature_summary", {})
        .get("feature_summary", {})
        .get("runtime_calibration_scales", {})
    )
    for source in (transport, interface, features, nested_features):
        if name in source:
            try:
                value = float(source[name])
            except (TypeError, ValueError):
                return math.nan
            return value if math.isfinite(value) else math.nan
    return math.nan


def run_case(args: argparse.Namespace, scenario: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    case_dir = args.output_dir / "runs" / scenario["name"] / case["case"]
    command = [
        str((ROOT / args.python).absolute() if not Path(args.python).is_absolute() else args.python),
        str((ROOT / args.script).absolute() if not Path(args.script).is_absolute() else args.script),
        "--generation-map-npz",
        str(args.generation_map_npz),
        "--generation-profile-case",
        case["case"],
        "--generation-profile-wavelength-nm",
        f"{case['wavelength_nm']:.12g}",
        "--electrical-model",
        "profile-ppd",
        "--measured-profile",
        str(args.measured_profile),
        "--width-um",
        f"{args.width_um:.12g}",
        "--depth-um",
        f"{args.depth_um:.12g}",
        "--dd-relative-error",
        f"{args.dd_relative_error:.12g}",
        "--dd-max-iterations",
        str(args.dd_max_iterations),
        "--output-dir",
        str(case_dir),
    ]
    command.extend(scenario["extra_args"])
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(f"{scenario['name']} {case['case']} failed:\n{tail}")
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    total = float(summary["left_photo_delta_a_per_cm"]) + float(
        summary["right_photo_delta_a_per_cm"]
    )
    transport = summary.get("transport_summary", {})
    return {
        "scenario": scenario["name"],
        "case": case["case"],
        "wavelength_nm": case["wavelength_nm"],
        "total_photo_current_a_per_cm": total,
        "split_phase_x": summary.get("photo_split_phase_x_proxy"),
        "electron_mobility_scale": scale_from_summary(summary, "electron_mobility_scale"),
        "hole_mobility_scale": scale_from_summary(summary, "hole_mobility_scale"),
        "lifetime_scale": scale_from_summary(summary, "lifetime_scale"),
        "fixed_charge_scale": scale_from_summary(summary, "fixed_charge_scale"),
        "interface_trap_density_scale": scale_from_summary(summary, "interface_trap_density_scale"),
        "interface_trap_recombination_scale": scale_from_summary(
            summary, "interface_trap_recombination_scale"
        ),
        "electron_mobility_min_cm2_v_s": transport.get("electron_mobility_min_cm2_v_s"),
        "electron_mobility_max_cm2_v_s": transport.get("electron_mobility_max_cm2_v_s"),
        "tau_n_min_s": transport.get("tau_n_min_s"),
        "tau_n_max_s": transport.get("tau_n_max_s"),
        "recombination_coeff_max_s1": summary.get("interface_trap_summary", {}).get(
            "recombination_coeff_max_s1"
        ),
        "summary_json": str(summary_path),
    }


def expected_scale(scenario: dict[str, Any], option: str) -> float:
    prefix = f"--{option}="
    args = scenario["extra_args"]
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return float(token.split("=", 1)[1])
        if token == f"--{option}" and index + 1 < len(args):
            return float(args[index + 1])
    return 1.0


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (row["case"], row["wavelength_nm"]): row
        for row in rows
        if row["scenario"] == "baseline"
    }
    for row in rows:
        base = baselines[(row["case"], row["wavelength_nm"])]
        denom = max(abs(float(base["total_photo_current_a_per_cm"])), 1.0e-30)
        row["relative_total_delta_to_baseline"] = (
            float(row["total_photo_current_a_per_cm"]) - float(base["total_photo_current_a_per_cm"])
        ) / denom
        row["split_phase_delta_to_baseline"] = float(row["split_phase_x"]) - float(
            base["split_phase_x"]
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = parse_cases(args.cases)
    scenarios = parse_scenarios(args.scenarios)
    rows = [run_case(args, scenario, case) for scenario in scenarios for case in cases]
    add_baseline_deltas(rows)
    wiring_issues = []
    for scenario in scenarios:
        for row in [item for item in rows if item["scenario"] == scenario["name"]]:
            for option, field in (
                ("electron-mobility-scale", "electron_mobility_scale"),
                ("hole-mobility-scale", "hole_mobility_scale"),
                ("lifetime-scale", "lifetime_scale"),
                ("fixed-charge-scale", "fixed_charge_scale"),
                ("interface-trap-density-scale", "interface_trap_density_scale"),
                ("interface-trap-recombination-scale", "interface_trap_recombination_scale"),
            ):
                expected = expected_scale(scenario, option)
                actual = float(row[field])
                if not math.isfinite(actual) or abs(actual - expected) > 1.0e-12:
                    wiring_issues.append(
                        {
                            "scenario": scenario["name"],
                            "case": row["case"],
                            "field": field,
                            "expected": expected,
                            "actual": row[field],
                        }
                    )
    nonbaseline = [row for row in rows if row["scenario"] != "baseline"]
    changed_rows = [
        row
        for row in nonbaseline
        if abs(float(row["relative_total_delta_to_baseline"])) >= args.response_change_rel_tol
        or abs(float(row["split_phase_delta_to_baseline"])) >= args.response_change_abs_tol
    ]
    json_path = args.output_dir / "transport_sensitivity_report.json"
    csv_path = args.output_dir / "transport_sensitivity_report.csv"
    payload = {
        "schema": "tcad_transport_sensitivity_v1",
        "artifact_role": "transport_runtime_calibration_control_check",
        "case_count": len(cases),
        "scenario_count": len(scenarios),
        "row_count": len(rows),
        "solver_parameter_wiring_pass": not wiring_issues,
        "response_sensitivity_pass": bool(changed_rows),
        "changed_row_count": len(changed_rows),
        "wiring_issues": wiring_issues,
        "response_change_rel_tol": args.response_change_rel_tol,
        "response_change_abs_tol": args.response_change_abs_tol,
        "rows": rows,
        "outputs": {"json": str(json_path), "csv": str(csv_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=".tcad-env/bin/python")
    parser.add_argument("--script", default="devsim_split_pd_2d.py")
    parser.add_argument(
        "--generation-map-npz",
        type=Path,
        default=ROOT / "runs" / "fdtd_to_tcad_generation_2d_cra_smoke" / "tcad_generation_map_2d.npz",
    )
    parser.add_argument(
        "--measured-profile",
        type=Path,
        default=ROOT / "measured_profiles" / "reference_cmos_ppd_1p4um" / "profile.json",
    )
    parser.add_argument("--cases", default="center:550,edge20x:550")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--width-um", type=float, default=1.4)
    parser.add_argument("--depth-um", type=float, default=3.0)
    parser.add_argument("--dd-relative-error", type=float, default=1.0e-9)
    parser.add_argument("--dd-max-iterations", type=int, default=160)
    parser.add_argument("--response-change-rel-tol", type=float, default=1.0e-6)
    parser.add_argument("--response-change-abs-tol", type=float, default=1.0e-6)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "solver_parameter_wiring_pass": payload["solver_parameter_wiring_pass"],
                "response_sensitivity_pass": payload["response_sensitivity_pass"],
                "changed_row_count": payload["changed_row_count"],
                "row_count": payload["row_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
