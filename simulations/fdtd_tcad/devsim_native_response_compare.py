#!/usr/bin/env python3
"""Compare two DEVSIM native-response sweeps case-by-case.

This is intended for mesh/model A/B checks such as baseline silicon-only DTI
versus resolved oxide DTI. It reads the sweep CSV plus each case summary JSON so
photo response and dark/illuminated terminal currents can be inspected together.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

CSV_COLUMNS = [
    "case",
    "wavelength_nm",
    "cra_x_deg",
    "field_x_norm",
    "baseline_total_photo_delta_a_per_cm",
    "candidate_total_photo_delta_a_per_cm",
    "total_photo_delta_ratio",
    "total_photo_delta_rel_change",
    "baseline_split_phase_x_proxy",
    "candidate_split_phase_x_proxy",
    "split_phase_delta",
    "baseline_left_photo_delta_a_per_cm",
    "candidate_left_photo_delta_a_per_cm",
    "left_photo_delta_ratio",
    "baseline_right_photo_delta_a_per_cm",
    "candidate_right_photo_delta_a_per_cm",
    "right_photo_delta_ratio",
    "baseline_dark_total_cathode_current_a_per_cm",
    "candidate_dark_total_cathode_current_a_per_cm",
    "dark_total_cathode_current_ratio",
    "baseline_dark_signal_current_a_per_cm",
    "candidate_dark_signal_current_a_per_cm",
    "dark_signal_current_ratio",
    "candidate_dark_signal_to_photo_fraction",
    "baseline_illuminated_total_cathode_current_a_per_cm",
    "candidate_illuminated_total_cathode_current_a_per_cm",
    "illuminated_total_cathode_current_ratio",
    "baseline_terminal_balance_illuminated_a_per_cm",
    "candidate_terminal_balance_illuminated_a_per_cm",
    "baseline_node_count",
    "candidate_node_count",
    "mesh_node_delta",
    "candidate_interface_trap_measured",
    "candidate_transport_calibrated",
    "dark_current_alert",
    "dark_signal_alert",
    "baseline_summary_json",
    "candidate_summary_json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in CSV_COLUMNS})


def rel_from_root(path: Path | str | None) -> str:
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


def rel_change(candidate: float | None, baseline: float | None) -> float | None:
    value = ratio(candidate, baseline)
    if value is None:
        return None
    return value - 1.0


def fmt_number(value: Any, digits: int = 6) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def case_key(row: dict[str, str]) -> tuple[str, float | None]:
    return str(row.get("case", "")), safe_float(row.get("wavelength_nm"))


def load_summary(row: dict[str, str]) -> dict[str, Any]:
    path_value = row.get("summary_json")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    data = read_json(path)
    data["_summary_path"] = path
    return data


def photo_total(row: dict[str, str], summary: dict[str, Any]) -> float | None:
    csv_total = safe_float(row.get("photo_total_abs_delta_a_per_cm"))
    if csv_total is not None:
        return csv_total
    left = safe_float(row.get("left_photo_delta_a_per_cm"))
    right = safe_float(row.get("right_photo_delta_a_per_cm"))
    if left is None:
        left = safe_float(summary.get("left_photo_delta_a_per_cm"))
    if right is None:
        right = safe_float(summary.get("right_photo_delta_a_per_cm"))
    if left is None or right is None:
        return None
    return left + right


def nested_float(data: dict[str, Any], section: str, key: str) -> float | None:
    section_data = data.get(section, {})
    if not isinstance(section_data, dict):
        return None
    return safe_float(section_data.get(key))


def all_interface_traps_measured(summary: dict[str, Any]) -> bool | None:
    trap_summary = summary.get("interface_trap_summary", {})
    if not isinstance(trap_summary, dict):
        return None
    traps = trap_summary.get("applied_interface_traps", [])
    if not isinstance(traps, list) or not traps:
        return None
    measured = [safe_bool(item.get("measured")) for item in traps if isinstance(item, dict)]
    if not measured:
        return None
    return all(item is True for item in measured)


def transport_calibrated(summary: dict[str, Any]) -> bool | None:
    transport = summary.get("transport_summary", {})
    if not isinstance(transport, dict):
        return None
    return safe_bool(transport.get("calibrated"))


def build_index(summary_csv: Path) -> dict[tuple[str, float | None], dict[str, Any]]:
    rows = read_csv_rows(summary_csv)
    indexed: dict[tuple[str, float | None], dict[str, Any]] = {}
    for row in rows:
        summary = load_summary(row)
        indexed[case_key(row)] = {"row": row, "summary": summary}
    return indexed


def compare_rows(
    baseline_csv: Path,
    candidate_csv: Path,
    dark_alert_ratio: float,
    dark_signal_fraction_alert: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    baseline = build_index(baseline_csv)
    candidate = build_index(candidate_csv)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in sorted(set(baseline) | set(candidate), key=lambda item: (item[1] or 0.0, item[0])):
        base_item = baseline.get(key)
        cand_item = candidate.get(key)
        if not base_item or not cand_item:
            missing.append(f"{key[0]}:{key[1]}")
            continue
        base_row = base_item["row"]
        cand_row = cand_item["row"]
        base_summary = base_item["summary"]
        cand_summary = cand_item["summary"]
        base_total = photo_total(base_row, base_summary)
        cand_total = photo_total(cand_row, cand_summary)
        base_left = safe_float(base_row.get("left_photo_delta_a_per_cm"))
        cand_left = safe_float(cand_row.get("left_photo_delta_a_per_cm"))
        base_right = safe_float(base_row.get("right_photo_delta_a_per_cm"))
        cand_right = safe_float(cand_row.get("right_photo_delta_a_per_cm"))
        base_split = safe_float(base_row.get("photo_split_phase_x_proxy"))
        cand_split = safe_float(cand_row.get("photo_split_phase_x_proxy"))
        base_dark = nested_float(base_summary, "dark", "total_cathode_current_a_per_cm")
        cand_dark = nested_float(cand_summary, "dark", "total_cathode_current_a_per_cm")
        base_dark_signal = nested_float(base_summary, "dark", "total_cathode_signal_current_a_per_cm")
        cand_dark_signal = nested_float(cand_summary, "dark", "total_cathode_signal_current_a_per_cm")
        base_illum = nested_float(base_summary, "illuminated", "total_cathode_current_a_per_cm")
        cand_illum = nested_float(cand_summary, "illuminated", "total_cathode_current_a_per_cm")
        dark_ratio = ratio(cand_dark, base_dark)
        dark_signal_fraction = None
        if cand_dark_signal is not None and cand_total not in (None, 0):
            dark_signal_fraction = abs(cand_dark_signal) / abs(cand_total)
        base_nodes = safe_float(base_row.get("mesh_node_count")) or safe_float(base_summary.get("node_count"))
        cand_nodes = safe_float(cand_row.get("mesh_node_count")) or safe_float(cand_summary.get("node_count"))
        row = {
            "case": key[0],
            "wavelength_nm": key[1],
            "cra_x_deg": safe_float(cand_row.get("cra_x_deg")),
            "field_x_norm": safe_float(cand_row.get("field_x_norm")),
            "baseline_total_photo_delta_a_per_cm": base_total,
            "candidate_total_photo_delta_a_per_cm": cand_total,
            "total_photo_delta_ratio": ratio(cand_total, base_total),
            "total_photo_delta_rel_change": rel_change(cand_total, base_total),
            "baseline_split_phase_x_proxy": base_split,
            "candidate_split_phase_x_proxy": cand_split,
            "split_phase_delta": None if base_split is None or cand_split is None else cand_split - base_split,
            "baseline_left_photo_delta_a_per_cm": base_left,
            "candidate_left_photo_delta_a_per_cm": cand_left,
            "left_photo_delta_ratio": ratio(cand_left, base_left),
            "baseline_right_photo_delta_a_per_cm": base_right,
            "candidate_right_photo_delta_a_per_cm": cand_right,
            "right_photo_delta_ratio": ratio(cand_right, base_right),
            "baseline_dark_total_cathode_current_a_per_cm": base_dark,
            "candidate_dark_total_cathode_current_a_per_cm": cand_dark,
            "dark_total_cathode_current_ratio": dark_ratio,
            "baseline_dark_signal_current_a_per_cm": base_dark_signal,
            "candidate_dark_signal_current_a_per_cm": cand_dark_signal,
            "dark_signal_current_ratio": ratio(cand_dark_signal, base_dark_signal),
            "candidate_dark_signal_to_photo_fraction": dark_signal_fraction,
            "baseline_illuminated_total_cathode_current_a_per_cm": base_illum,
            "candidate_illuminated_total_cathode_current_a_per_cm": cand_illum,
            "illuminated_total_cathode_current_ratio": ratio(cand_illum, base_illum),
            "baseline_terminal_balance_illuminated_a_per_cm": safe_float(
                base_row.get("terminal_balance_illuminated_a_per_cm")
            )
            or safe_float(base_summary.get("terminal_current_balance_illuminated_a_per_cm")),
            "candidate_terminal_balance_illuminated_a_per_cm": safe_float(
                cand_row.get("terminal_balance_illuminated_a_per_cm")
            )
            or safe_float(cand_summary.get("terminal_current_balance_illuminated_a_per_cm")),
            "baseline_node_count": base_nodes,
            "candidate_node_count": cand_nodes,
            "mesh_node_delta": None if base_nodes is None or cand_nodes is None else cand_nodes - base_nodes,
            "candidate_interface_trap_measured": all_interface_traps_measured(cand_summary),
            "candidate_transport_calibrated": transport_calibrated(cand_summary),
            "dark_current_alert": bool(dark_ratio is not None and abs(dark_ratio) >= dark_alert_ratio),
            "dark_signal_alert": bool(
                dark_signal_fraction is not None and dark_signal_fraction >= dark_signal_fraction_alert
            ),
            "baseline_summary_json": rel_from_root(base_summary.get("_summary_path")),
            "candidate_summary_json": rel_from_root(cand_summary.get("_summary_path")),
        }
        rows.append(row)
    return rows, missing


def write_html(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in rows:
        alert = "bad" if row.get("dark_signal_alert") else "ok"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{fmt_number(row.get('cra_x_deg'))}</td>"
            f"<td>{fmt_number(row.get('baseline_total_photo_delta_a_per_cm'))}</td>"
            f"<td>{fmt_number(row.get('candidate_total_photo_delta_a_per_cm'))}</td>"
            f"<td>{fmt_number(row.get('total_photo_delta_ratio'))}</td>"
            f"<td>{fmt_number(row.get('baseline_split_phase_x_proxy'))}</td>"
            f"<td>{fmt_number(row.get('candidate_split_phase_x_proxy'))}</td>"
            f"<td>{fmt_number(row.get('split_phase_delta'))}</td>"
            f"<td>{fmt_number(row.get('dark_total_cathode_current_ratio'))}</td>"
            f"<td>{fmt_number(row.get('candidate_dark_signal_to_photo_fraction'))}</td>"
            f"<td><span class=\"pill {alert}\">{html.escape(str(row.get('dark_signal_alert')).lower())}</span></td>"
            f"<td>{fmt_number(row.get('candidate_terminal_balance_illuminated_a_per_cm'), 3)}</td>"
            f"<td>{fmt_number(row.get('mesh_node_delta'), 3)}</td>"
            "</tr>"
        )
    html_rows = "\n".join(table_rows)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native DEVSIM Response Compare</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f4f6f8;font-size:13px}}
header{{padding:18px 22px;background:#111827;color:white}}
h1{{font-size:18px;margin:0 0 6px}}
p{{margin:0;color:#c9d3df}}
main{{padding:16px 22px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}}
.metric{{background:white;border:1px solid #cfd7df;border-radius:8px;padding:10px}}
.label{{color:#61707f;font-size:12px;margin-bottom:5px}}
.value{{font-weight:700;font-size:18px}}
.note{{background:#fff8eb;border:1px solid #f5c27a;color:#7c3f00;border-radius:8px;padding:10px;margin:12px 0}}
.tableWrap{{background:white;border:1px solid #cfd7df;border-radius:8px;overflow:auto}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px}}
th,td{{border-bottom:1px solid #e5e9ee;padding:7px 8px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th{{position:sticky;top:0;background:#f9fafb;z-index:1;color:#4b5563}}
.pill{{display:inline-flex;height:20px;align-items:center;padding:0 6px;border-radius:5px;border:1px solid #cfd7df;background:#fff}}
.ok{{border-color:#b7dec7;color:#177245;background:#f0fbf4}}
.bad{{border-color:#f0a4a4;color:#b91c1c;background:#fff5f5}}
</style>
</head>
<body>
<header>
  <h1>Native DEVSIM Response Compare</h1>
  <p>{html.escape(metadata["baseline_label"])} vs {html.escape(metadata["candidate_label"])}</p>
</header>
<main>
  <section class="metrics">
    <div class="metric"><div class="label">Cases</div><div class="value">{len(rows)}</div></div>
    <div class="metric"><div class="label">Max Total Ratio</div><div class="value">{fmt_number(metadata.get("max_total_photo_delta_ratio"))}</div></div>
    <div class="metric"><div class="label">Max |Split Delta|</div><div class="value">{fmt_number(metadata.get("max_abs_split_phase_delta"))}</div></div>
    <div class="metric"><div class="label">Signal Dark Alerts</div><div class="value">{metadata.get("dark_signal_alert_count", 0)}</div></div>
  </section>
  <div class="note">This is a numerical A/B comparison artifact. It does not certify product-LUT accuracy; measured DTI/interface/transport calibration is still required.</div>
  <section class="tableWrap">
    <table>
      <thead><tr><th>Case</th><th>CRA X</th><th>Base Total</th><th>Candidate Total</th><th>Total Ratio</th><th>Base Split</th><th>Candidate Split</th><th>Split Delta</th><th>Total Dark Ratio</th><th>Dark Signal / Photo</th><th>Signal Alert</th><th>Terminal Balance</th><th>Node Delta</th></tr></thead>
      <tbody>{html_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def summarize(rows: list[dict[str, Any]], missing_cases: list[str]) -> dict[str, Any]:
    ratios = [safe_float(row.get("total_photo_delta_ratio")) for row in rows]
    ratios = [value for value in ratios if value is not None]
    split_deltas = [abs(value) for row in rows if (value := safe_float(row.get("split_phase_delta"))) is not None]
    dark_ratios = [safe_float(row.get("dark_total_cathode_current_ratio")) for row in rows]
    dark_ratios = [value for value in dark_ratios if value is not None]
    return {
        "case_count": len(rows),
        "missing_case_count": len(missing_cases),
        "missing_cases": missing_cases,
        "max_total_photo_delta_ratio": max(ratios) if ratios else None,
        "min_total_photo_delta_ratio": min(ratios) if ratios else None,
        "max_abs_split_phase_delta": max(split_deltas) if split_deltas else None,
        "max_dark_total_cathode_current_ratio": max(dark_ratios) if dark_ratios else None,
        "dark_current_alert_count": sum(1 for row in rows if row.get("dark_current_alert")),
        "dark_signal_alert_count": sum(1 for row in rows if row.get("dark_signal_alert")),
        "product_lut_ready": False,
        "accuracy_certified": False,
    }


def run(
    baseline_summary_csv: Path,
    candidate_summary_csv: Path,
    output_dir: Path,
    baseline_label: str,
    candidate_label: str,
    dark_alert_ratio: float,
    dark_signal_fraction_alert: float,
) -> dict[str, Any]:
    rows, missing_cases = compare_rows(
        baseline_summary_csv,
        candidate_summary_csv,
        dark_alert_ratio,
        dark_signal_fraction_alert,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, missing_cases)
    summary.update(
        {
            "schema": "devsim_native_response_compare_v1",
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "baseline_summary_csv": rel_from_root(baseline_summary_csv),
            "candidate_summary_csv": rel_from_root(candidate_summary_csv),
            "dark_alert_ratio": dark_alert_ratio,
            "dark_signal_fraction_alert": dark_signal_fraction_alert,
        }
    )
    outputs = {
        "json": output_dir / "native_response_compare.json",
        "csv": output_dir / "native_response_compare.csv",
        "html": output_dir / "native_response_compare.html",
    }
    data = {
        "schema": "devsim_native_response_compare_v1",
        "summary": summary,
        "rows": rows,
        "outputs": {key: rel_from_root(value) for key, value in outputs.items()},
        "limitations": [
            "Comparison uses existing local DEVSIM sweep artifacts and does not launch a new solve.",
            "Product LUT readiness remains false until measured geometry/n,k/implants/interface/transport calibration targets pass.",
            "A dark-current alert marks a model-calibration priority, not a UI warning condition.",
        ],
    }
    write_json(outputs["json"], data)
    write_csv(outputs["csv"], rows)
    write_html(outputs["html"], rows, summary)
    print(json.dumps({"summary": summary, "outputs": data["outputs"]}, indent=2, ensure_ascii=False))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary-csv", type=Path, required=True)
    parser.add_argument("--candidate-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--dark-alert-ratio", type=float, default=10.0)
    parser.add_argument("--dark-signal-fraction-alert", type=float, default=0.01)
    args = parser.parse_args()
    run(
        args.baseline_summary_csv.resolve(),
        args.candidate_summary_csv.resolve(),
        args.output_dir.resolve(),
        args.baseline_label,
        args.candidate_label,
        args.dark_alert_ratio,
        args.dark_signal_fraction_alert,
    )


if __name__ == "__main__":
    main()
