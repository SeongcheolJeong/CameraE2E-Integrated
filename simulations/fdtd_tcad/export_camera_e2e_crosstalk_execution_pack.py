#!/usr/bin/env python3
"""Export runnable crosstalk execution scripts for CameraE2E LUT closure.

The priority CSV already contains the exact solver commands. This exporter
turns those rows into a small handoff package for batch/HPC execution and local
support-discovery checks. It does not run solvers and does not promote any row
to product accuracy.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_crosstalk_execution_pack"
DEFAULT_PRIORITY_CSV = DEFAULT_PACKAGE_DIR / "camera_e2e_crosstalk_batch_priority" / "camera_e2e_crosstalk_batch_priority.csv"
DEFAULT_LOCAL_PROBE_DIR = DEFAULT_PACKAGE_DIR / "crosstalk_support_discovery" / "spectral_green_center_450_620_n15_res20"

JOB_COLUMNS = [
    "job_index",
    "execution_group",
    "priority_rank",
    "priority_class",
    "action_type",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "queue_id",
    "color_channel",
    "wavelength_nm",
    "field_case",
    "mode",
    "recommended_neighborhood",
    "resolution_px_per_um",
    "estimated_voxels",
    "estimated_memory_class",
    "local_feasibility",
    "support_evidence_gate",
    "candidate_support_role",
    "candidate_priority",
    "product_use_gate",
    "expected_artifact",
    "command",
]

SCRIPT_COLUMNS = [
    "script_id",
    "role",
    "path",
    "job_count",
    "execution_policy",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def classify_execution_group(row: dict[str, str]) -> str:
    action = row.get("action_type", "")
    feasibility = row.get("local_feasibility", "")
    if action == "product_resolution_crosstalk_primary":
        return "product_primary_hpc"
    if feasibility == "RUNNABLE_LOCAL_CHECK":
        return "support_discovery_local_candidate"
    return "support_discovery_batch_or_reformulation"


def build_job_rows(priority_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(priority_rows, start=1):
        execution_group = classify_execution_group(row)
        item = {column: row.get(column, "") for column in JOB_COLUMNS}
        item["job_index"] = index
        item["execution_group"] = execution_group
        output.append(item)
    return output


def write_script(path: Path, rows: list[dict[str, Any]], *, title: str, policy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# {title}",
        f"# {policy}",
        f"cd {ROOT}",
        "",
    ]
    for row in rows:
        lines.append(f"echo '[{row.get('job_index')}] {row.get('slug')} {row.get('color_channel')} {row.get('field_case')} {row.get('wavelength_nm')}nm'")
        lines.append(str(row.get("command", "")))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def parse_local_probe_logs(probe_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stdout_path in sorted(probe_dir.glob("**/stdout.log")):
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
        elapsed_match = re.search(r"Elapsed run time = ([0-9.]+) s", text)
        epsilon_match = re.search(r"time for set_epsilon = ([0-9.]+) s", text)
        conductivity_matches = re.findall(r"time for set_conductivity = ([0-9.]+) s", text)
        cell_match = re.search(r"Computational cell is ([^\n]+)", text)
        started_match = re.search(r"running mode=([^,]+), neighborhood=([^,]+), res=([^,]+), wavelength=([^,]+), case=([^\n]+)", text)
        rows.append(
            {
                "stdout": repo_rel(stdout_path),
                "stderr": repo_rel(stdout_path.with_name("stderr.log")),
                "completed_summary_present": stdout_path.with_name("crosstalk_kernel_summary.csv").exists(),
                "elapsed_s": float(elapsed_match.group(1)) if elapsed_match else "",
                "set_epsilon_s": float(epsilon_match.group(1)) if epsilon_match else "",
                "set_conductivity_s_total": sum(float(value) for value in conductivity_matches),
                "computational_cell": cell_match.group(1).strip() if cell_match else "",
                "started_case": started_match.group(0).strip() if started_match else "",
                "interpretation": (
                    "local probe was interrupted before a usable summary; setup cost alone is evidence that broad finite-array "
                    "n15/res20 support discovery should be scheduled selectively or moved to batch/HPC"
                ),
            }
        )
    return rows


def script_rows_for(output_dir: Path, groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "script_id": "product_primary_hpc",
            "role": "product-resolution finite-array crosstalk primary jobs",
            "path": repo_rel(output_dir / "run_product_primary_hpc.sh"),
            "job_count": len(groups["product_primary_hpc"]),
            "execution_policy": "Run on HPC/large workstation/domain-decomposition environment; rerun audits after completion.",
        },
        {
            "script_id": "support_discovery_local_candidate",
            "role": "low-resolution support discovery jobs classified as locally runnable",
            "path": repo_rel(output_dir / "run_support_discovery_local_candidates.sh"),
            "job_count": len(groups["support_discovery_local_candidate"]),
            "execution_policy": "Run selectively; previous local probe shows even n15/res20 can be expensive.",
        },
        {
            "script_id": "support_discovery_batch_or_reformulation",
            "role": "support discovery jobs that should not be run as broad local batches",
            "path": repo_rel(output_dir / "run_support_discovery_batch_or_reformulation.sh"),
            "job_count": len(groups["support_discovery_batch_or_reformulation"]),
            "execution_policy": "Run through batch/HPC or after crosstalk solver reformulation.",
        },
        {
            "script_id": "refresh_after_solver_jobs",
            "role": "package refresh after solver artifacts are produced",
            "path": repo_rel(output_dir / "refresh_after_solver_jobs.sh"),
            "job_count": 0,
            "execution_policy": "Run after selected crosstalk jobs complete.",
        },
    ]


def write_refresh_script(path: Path, package_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        "python3 audit_camera_e2e_crosstalk_support.py "
        f"--package-dir {package_dir} "
        f"--output-dir {package_dir / 'camera_e2e_crosstalk_support_audit'}",
        "python3 export_camera_e2e_crosstalk_batch_priority.py "
        f"--package-dir {package_dir} "
        f"--output-dir {package_dir / 'camera_e2e_crosstalk_batch_priority'}",
        "python3 run_camera_e2e_package_pipeline.py --include-failed --skip-rebuild",
        "python3 audit_camera_e2e_objective_acceptance.py "
        f"--package-dir {package_dir} "
        f"--output-dir {package_dir / 'camera_e2e_objective_acceptance'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def write_html(path: Path, payload: dict[str, Any], jobs: list[dict[str, Any]], scripts: list[dict[str, Any]]) -> None:
    def table(rows: list[dict[str, Any]], columns: list[str], limit: int = 80) -> str:
        if not rows:
            return "<p>No rows.</p>"
        header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
        body = []
        for row in rows[:limit]:
            body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
        if len(rows) > limit:
            body.append(f"<tr><td colspan='{len(columns)}'>... {len(rows) - limit} more rows in CSV</td></tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}.muted{color:#9db6c8}.warn{color:#ffd36e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}
th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#d8fbff}
"""
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Crosstalk Execution Pack</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Crosstalk Execution Pack</h1>
<p class="muted">Generated <code>{html_cell(payload.get("generated_at", ""))}</code>. Product use remains blocked until selected jobs finish with product mesh/convergence PASS and measured inputs.</p>
<div class="grid">
<div class="card"><div class="metric">{html_cell(payload.get("job_count", 0))}</div><div class="muted">jobs</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("product_primary_job_count", 0))}</div><div class="muted">product primary HPC</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("support_local_candidate_count", 0))}</div><div class="muted">local candidates</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_use_gate", ""))}</div><div class="muted">product gate</div></div>
</div>
<h2>Scripts</h2>{table(scripts, SCRIPT_COLUMNS)}
<h2>Jobs</h2>{table(jobs, JOB_COLUMNS)}
</main></body></html>
"""
    path.write_text(html_text, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_crosstalk_execution_pack_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_crosstalk_execution_pack_jobs_csv"] = payload["outputs"]["jobs_csv"]
    outputs["camera_e2e_crosstalk_execution_pack_scripts_csv"] = payload["outputs"]["scripts_csv"]
    outputs["camera_e2e_crosstalk_execution_pack_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_crosstalk_execution_pack"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "job_count": payload["job_count"],
        "product_primary_job_count": payload["product_primary_job_count"],
        "support_local_candidate_count": payload["support_local_candidate_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def export_pack(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    priority_csv = args.priority_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    priority_rows = read_csv_rows(priority_csv)
    jobs = build_job_rows(priority_rows)
    groups = {
        "product_primary_hpc": [row for row in jobs if row["execution_group"] == "product_primary_hpc"],
        "support_discovery_local_candidate": [row for row in jobs if row["execution_group"] == "support_discovery_local_candidate"],
        "support_discovery_batch_or_reformulation": [row for row in jobs if row["execution_group"] == "support_discovery_batch_or_reformulation"],
    }

    write_script(
        output_dir / "run_product_primary_hpc.sh",
        groups["product_primary_hpc"],
        title="Product-primary finite-array crosstalk jobs",
        policy="These are not local smoke jobs. Run on HPC/large workstation and refresh package after completion.",
    )
    write_script(
        output_dir / "run_support_discovery_local_candidates.sh",
        groups["support_discovery_local_candidate"],
        title="Support-discovery jobs marked as local candidates",
        policy="Run selectively. Do not launch the full list blindly on a laptop.",
    )
    write_script(
        output_dir / "run_support_discovery_batch_or_reformulation.sh",
        groups["support_discovery_batch_or_reformulation"],
        title="Support-discovery jobs requiring batch/HPC or solver reformulation",
        policy="These should not be broad local runs.",
    )
    write_refresh_script(output_dir / "refresh_after_solver_jobs.sh", package_dir)

    scripts = script_rows_for(output_dir, groups)
    local_probe_rows = parse_local_probe_logs(args.local_probe_dir.resolve())
    issues: list[dict[str, Any]] = []
    if not jobs:
        issues.append({"severity": "error", "code": "no_jobs"})
    if not groups["product_primary_hpc"]:
        issues.append({"severity": "warning", "code": "no_product_primary_hpc_jobs"})
    if not local_probe_rows:
        issues.append({"severity": "warning", "code": "no_local_probe_evidence"})
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")

    jobs_csv = output_dir / "camera_e2e_crosstalk_execution_jobs.csv"
    scripts_csv = output_dir / "camera_e2e_crosstalk_execution_scripts.csv"
    local_probe_csv = output_dir / "camera_e2e_crosstalk_local_probe_evidence.csv"
    json_path = output_dir / "camera_e2e_crosstalk_execution_pack.json"
    html_path = output_dir / "index.html"
    write_csv(jobs_csv, jobs, JOB_COLUMNS)
    write_csv(scripts_csv, scripts, SCRIPT_COLUMNS)
    write_csv(
        local_probe_csv,
        local_probe_rows,
        [
            "stdout",
            "stderr",
            "completed_summary_present",
            "elapsed_s",
            "set_epsilon_s",
            "set_conductivity_s_total",
            "computational_cell",
            "started_case",
            "interpretation",
        ],
    )

    group_counts = dict(Counter(row.get("execution_group", "") for row in jobs))
    payload = {
        "schema": "camera_e2e_crosstalk_execution_pack_v1",
        "artifact_role": "crosstalk_hpc_and_support_discovery_execution_handoff",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "priority_csv": repo_rel(priority_csv),
        "job_count": len(jobs),
        "product_primary_job_count": len(groups["product_primary_hpc"]),
        "support_local_candidate_count": len(groups["support_discovery_local_candidate"]),
        "support_batch_or_reformulation_count": len(groups["support_discovery_batch_or_reformulation"]),
        "execution_group_counts": group_counts,
        "local_probe_evidence_count": len(local_probe_rows),
        "local_probe_evidence": local_probe_rows,
        "product_use_gate": "FAIL",
        "policy": (
            "Scripts are execution handoff artifacts only. Product crosstalk remains blocked until the resulting solver summaries "
            "pass product mesh/convergence gates and measured stack/material calibration blockers are removed."
        ),
        "validation": {
            "schema": "camera_e2e_crosstalk_execution_pack_validation_v1",
            "pass": error_count == 0,
            "status": "CROSSTALK_EXECUTION_PACK_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL",
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "issues": issues,
        },
        "outputs": {
            "json": repo_rel(json_path),
            "jobs_csv": repo_rel(jobs_csv),
            "scripts_csv": repo_rel(scripts_csv),
            "local_probe_csv": repo_rel(local_probe_csv),
            "html": repo_rel(html_path),
            "product_primary_hpc_script": repo_rel(output_dir / "run_product_primary_hpc.sh"),
            "support_discovery_local_script": repo_rel(output_dir / "run_support_discovery_local_candidates.sh"),
            "support_discovery_batch_script": repo_rel(output_dir / "run_support_discovery_batch_or_reformulation.sh"),
            "refresh_script": repo_rel(output_dir / "refresh_after_solver_jobs.sh"),
        },
    }
    write_json(json_path, payload)
    write_html(html_path, payload, jobs, scripts)
    update_package(package_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--priority-csv", type=Path, default=DEFAULT_PRIORITY_CSV)
    parser.add_argument("--local-probe-dir", type=Path, default=DEFAULT_LOCAL_PROBE_DIR)
    payload = export_pack(parser.parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "job_count": payload["job_count"],
                "product_primary_job_count": payload["product_primary_job_count"],
                "support_local_candidate_count": payload["support_local_candidate_count"],
                "support_batch_or_reformulation_count": payload["support_batch_or_reformulation_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
