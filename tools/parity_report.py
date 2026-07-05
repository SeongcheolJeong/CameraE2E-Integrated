"""Generate a machine-readable parity report against stored baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPO_ROOT / "tests" / "parity" / "cases.yaml"
BASELINES_DIR = REPO_ROOT / "tests" / "parity" / "baselines"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "parity" / "latest.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import AssetStore
from pyisetcam.parity import run_python_case_with_context
from pyisetcam.sensor import sensor_compute


def _case_definitions() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())["cases"]


def _field_rules(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rules = case.get("field_overrides", {})
    if not isinstance(raw_rules, dict):
        return {}
    return {str(key): value for key, value in raw_rules.items() if isinstance(value, dict)}


def _load_reference(case_name: str) -> dict[str, Any]:
    baseline_path = BASELINES_DIR / f"{case_name}.mat"
    return {
        key: value
        for key, value in loadmat(baseline_path, squeeze_me=True, struct_as_record=False).items()
        if not key.startswith("__")
    }


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if hasattr(value, "_fieldnames"):
        return {str(field): _normalize(getattr(value, field)) for field in getattr(value, "_fieldnames", [])}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        raw_fields = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        if raw_fields:
            return {str(key): _normalize(item) for key, item in raw_fields.items()}
    if isinstance(value, np.ndarray):
        array = np.squeeze(np.asarray(value))
        if np.iscomplexobj(array):
            array = np.real(array)
        return array
    if np.isscalar(value) and np.iscomplexobj(value):
        return float(np.real(value))
    return value


def _is_string_like_array(value: Any) -> bool:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        return True
    if array.dtype.kind != "O":
        return False
    flat = np.ravel(array)
    return all(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in flat)


def _string_array_payload(value: Any) -> list[str]:
    return [str(item) for item in np.ravel(np.asarray(value, dtype=object))]


def _array_metrics(reference: Any, actual: Any) -> dict[str, Any]:
    if _is_string_like_array(reference) or _is_string_like_array(actual):
        reference_array = np.asarray(reference)
        actual_array = np.asarray(actual)
        return {
            "shape": list(reference_array.shape),
            "string_values_match": _string_array_payload(reference_array) == _string_array_payload(actual_array),
            "expected_values": _string_array_payload(reference_array),
            "actual_values": _string_array_payload(actual_array),
        }
    reference_array = np.asarray(reference, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    difference = np.abs(actual_array - reference_array)
    denominator = np.maximum(np.abs(reference_array), 1e-12)
    relative = difference / denominator
    metrics = {
        "shape": list(reference_array.shape),
        "mae": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(actual_array - reference_array)))),
        "max_abs": float(np.max(difference)),
        "mean_rel": float(np.mean(relative)),
        "max_rel": float(np.max(relative)),
        "normalized_mae": float(np.mean(difference)) / max(float(np.mean(np.abs(reference_array))), 1e-12),
    }
    if reference_array.ndim == 0:
        metrics["max_index"] = []
        metrics["max_index_ref"] = float(reference_array)
        metrics["max_index_actual"] = float(actual_array)
        metrics["max_index_abs"] = float(difference)
        metrics["max_index_rel"] = float(relative)
        return metrics
    max_index = tuple(int(item) for item in np.unravel_index(int(np.argmax(relative)), relative.shape))
    metrics["max_index"] = list(max_index)
    metrics["max_index_ref"] = float(reference_array[max_index])
    metrics["max_index_actual"] = float(actual_array[max_index])
    metrics["max_index_abs"] = float(difference[max_index])
    metrics["max_index_rel"] = float(relative[max_index])

    if reference_array.ndim >= 2:
        border_rows = max(1, int(round(reference_array.shape[0] / 10.0)))
        border_cols = max(1, int(round(reference_array.shape[1] / 10.0)))
        spatial_mask = np.zeros(reference_array.shape[:2], dtype=bool)
        spatial_mask[:border_rows, :] = True
        spatial_mask[-border_rows:, :] = True
        spatial_mask[:, :border_cols] = True
        spatial_mask[:, -border_cols:] = True
        if reference_array.ndim > 2:
            expand = (slice(None), slice(None), *([None] * (reference_array.ndim - 2)))
            border_mask = np.broadcast_to(spatial_mask[expand], reference_array.shape)
        else:
            border_mask = spatial_mask
        interior_mask = ~border_mask
        metrics["border_rows"] = border_rows
        metrics["border_cols"] = border_cols
        metrics["edge_mean_rel"] = float(np.mean(relative[border_mask]))
        if np.any(interior_mask):
            metrics["interior_mean_rel"] = float(np.mean(relative[interior_mask]))

    if reference_array.ndim >= 2 and reference_array.shape[0] >= 2 and reference_array.shape[1] >= 2:
        phase_metrics: dict[str, float] = {}
        row_index, col_index = np.indices(reference_array.shape[:2])
        for row_phase in (0, 1):
            for col_phase in (0, 1):
                phase_mask = (row_index % 2 == row_phase) & (col_index % 2 == col_phase)
                if reference_array.ndim > 2:
                    expand = (slice(None), slice(None), *([None] * (reference_array.ndim - 2)))
                    phase_mask = np.broadcast_to(phase_mask[expand], reference_array.shape)
                phase_metrics[f"r{row_phase}c{col_phase}"] = float(np.mean(relative[phase_mask]))
        metrics["phase_2x2_mean_rel"] = phase_metrics

    return metrics


def _compare(
    reference: Any,
    actual: Any,
    *,
    rtol: float,
    atol: float,
    field_rules: dict[str, dict[str, Any]] | None = None,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    normalized_reference = _normalize(reference)
    normalized_actual = _normalize(actual)
    rule = field_rules.get(path[0]) if field_rules and path else None

    if isinstance(normalized_reference, dict):
        keys = sorted(key for key in normalized_reference if not key.startswith("__"))
        fields: dict[str, Any] = {}
        missing = [key for key in keys if key not in normalized_actual]
        if missing:
            return {"pass": False, "missing_keys": missing}
        overall = True
        for key in keys:
            field_report = _compare(
                normalized_reference[key],
                normalized_actual[key],
                rtol=rtol,
                atol=atol,
                field_rules=field_rules,
                path=(*path, key),
            )
            fields[key] = field_report
            overall = overall and bool(field_report["pass"])
        return {"pass": overall, "fields": fields}

    if _is_string_like_array(normalized_reference) or _is_string_like_array(normalized_actual):
        reference_array = np.asarray(normalized_reference)
        actual_array = np.asarray(normalized_actual)
        expected_values = _string_array_payload(reference_array)
        actual_values = _string_array_payload(actual_array)
        return {
            "pass": bool(reference_array.shape == actual_array.shape and expected_values == actual_values),
            "shape": list(reference_array.shape),
            "expected_values": expected_values,
            "actual_values": actual_values,
        }

    if isinstance(normalized_reference, (str, bytes)) or isinstance(normalized_actual, (str, bytes)):
        passed = normalized_actual == normalized_reference
        return {
            "pass": passed,
            "expected": normalized_reference,
            "actual": normalized_actual,
        }

    if np.asarray(normalized_reference).ndim == 0 and np.asarray(normalized_actual).ndim == 0:
        reference_scalar = float(np.asarray(normalized_reference))
        actual_scalar = float(np.asarray(normalized_actual))
        difference = abs(actual_scalar - reference_scalar)
        rel_diff = float(difference / max(abs(reference_scalar), 1e-12))
        base = {
            "expected": reference_scalar,
            "actual": actual_scalar,
            "abs_diff": difference,
            "rel_diff": rel_diff,
        }
        if rule and rule.get("mode") in {"mean_rel", "normalized_mae", "mae"}:
            if rule["mode"] == "mean_rel":
                threshold_key = "max_mean_rel"
                metric_value = rel_diff
            elif rule["mode"] == "normalized_mae":
                threshold_key = "max_normalized_mae"
                metric_value = rel_diff
            else:
                threshold_key = "max_mae"
                metric_value = difference
            threshold = float(rule[threshold_key])
            return {
                "pass": bool(metric_value <= threshold),
                "comparison_mode": str(rule["mode"]),
                threshold_key: threshold,
                **base,
            }
        if rule and rule.get("mode") == "scale_invariant":
            if abs(actual_scalar) <= 0.0:
                return {
                    "pass": False,
                    "comparison_mode": "scale_invariant",
                    "reason": "Scale-invariant comparison requires nonzero actual data.",
                    **base,
                }
            scale = reference_scalar / actual_scalar
            scaled_actual = scale * actual_scalar
            scaled_rel_diff = float(abs(scaled_actual - reference_scalar) / max(abs(reference_scalar), 1e-12))
            result = {
                "pass": bool(np.isclose(scaled_actual, reference_scalar, rtol=rtol, atol=atol)),
                "comparison_mode": "scale_invariant",
                "scale": scale,
                "scaled_actual": scaled_actual,
                "scaled_rel_diff": scaled_rel_diff,
                **base,
            }
            if "max_mean_rel" in rule:
                threshold = float(rule["max_mean_rel"])
                result["pass"] = bool(scaled_rel_diff <= threshold)
                result["max_mean_rel"] = threshold
            return result
        passed = bool(np.isclose(actual_scalar, reference_scalar, rtol=rtol, atol=atol))
        return {"pass": passed, **base}

    if np.isscalar(normalized_reference) and np.isscalar(normalized_actual):
        reference_scalar = float(normalized_reference)
        actual_scalar = float(normalized_actual)
        difference = abs(actual_scalar - reference_scalar)
        rel_diff = float(difference / max(abs(reference_scalar), 1e-12))
        base = {
            "expected": reference_scalar,
            "actual": actual_scalar,
            "abs_diff": difference,
            "rel_diff": rel_diff,
        }
        if rule and rule.get("mode") in {"mean_rel", "normalized_mae", "mae"}:
            if rule["mode"] == "mean_rel":
                threshold_key = "max_mean_rel"
                metric_value = rel_diff
            elif rule["mode"] == "normalized_mae":
                threshold_key = "max_normalized_mae"
                metric_value = rel_diff
            else:
                threshold_key = "max_mae"
                metric_value = difference
            threshold = float(rule[threshold_key])
            return {
                "pass": bool(metric_value <= threshold),
                "comparison_mode": str(rule["mode"]),
                threshold_key: threshold,
                **base,
            }
        if rule and rule.get("mode") == "scale_invariant":
            if abs(actual_scalar) <= 0.0:
                return {
                    "pass": False,
                    "comparison_mode": "scale_invariant",
                    "reason": "Scale-invariant comparison requires nonzero actual data.",
                    **base,
                }
            scale = reference_scalar / actual_scalar
            scaled_actual = scale * actual_scalar
            scaled_rel_diff = float(abs(scaled_actual - reference_scalar) / max(abs(reference_scalar), 1e-12))
            result = {
                "pass": bool(np.isclose(scaled_actual, reference_scalar, rtol=rtol, atol=atol)),
                "comparison_mode": "scale_invariant",
                "scale": scale,
                "scaled_actual": scaled_actual,
                "scaled_rel_diff": scaled_rel_diff,
                **base,
            }
            if "max_mean_rel" in rule:
                threshold = float(rule["max_mean_rel"])
                result["pass"] = bool(scaled_rel_diff <= threshold)
                result["max_mean_rel"] = threshold
            return result
        passed = bool(np.isclose(actual_scalar, reference_scalar, rtol=rtol, atol=atol))
        return {"pass": passed, **base}

    metrics = _array_metrics(normalized_reference, normalized_actual)
    if rule and rule.get("mode") == "mean_rel":
        threshold = float(rule["max_mean_rel"])
        return {
            "pass": bool(metrics["mean_rel"] <= threshold),
            "comparison_mode": "mean_rel",
            "max_mean_rel": threshold,
            **metrics,
        }
    if rule and rule.get("mode") == "normalized_mae":
        threshold = float(rule["max_normalized_mae"])
        return {
            "pass": bool(metrics["normalized_mae"] <= threshold),
            "comparison_mode": "normalized_mae",
            "max_normalized_mae": threshold,
            **metrics,
        }
    if rule and rule.get("mode") == "mae":
        threshold = float(rule["max_mae"])
        return {
            "pass": bool(metrics["mae"] <= threshold),
            "comparison_mode": "mae",
            "max_mae": threshold,
            **metrics,
        }
    if rule and rule.get("mode") == "scale_invariant":
        reference_array = np.asarray(normalized_reference, dtype=float)
        actual_array = np.asarray(normalized_actual, dtype=float)
        reference_flat = reference_array.reshape(-1)
        actual_flat = actual_array.reshape(-1)
        denominator = float(np.dot(actual_flat, actual_flat))
        if denominator <= 0.0:
            return {
                "pass": False,
                "comparison_mode": "scale_invariant",
                "reason": "Scale-invariant comparison requires nonzero actual data.",
                **metrics,
            }
        scale = float(np.dot(reference_flat, actual_flat)) / denominator
        scaled_actual = scale * actual_array
        scaled_metrics = _array_metrics(reference_array, scaled_actual)
        result = {
            "comparison_mode": "scale_invariant",
            "scale": scale,
            "unscaled_mean_rel": metrics["mean_rel"],
            "unscaled_max_rel": metrics["max_rel"],
            **scaled_metrics,
        }
        if "max_mean_rel" in rule:
            threshold = float(rule["max_mean_rel"])
            result["pass"] = bool(scaled_metrics["mean_rel"] <= threshold)
            result["max_mean_rel"] = threshold
        else:
            result["pass"] = bool(np.allclose(scaled_actual, reference_array, rtol=rtol, atol=atol))
        return result

    passed = bool(
        np.allclose(
            np.asarray(normalized_actual, dtype=float),
            np.asarray(normalized_reference, dtype=float),
            rtol=rtol,
            atol=atol,
        )
    )
    return {"pass": passed, **metrics}


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _context_metadata(context: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    oi = context.get("oi")
    if oi is not None:
        photons = np.asarray(oi.data.get("photons", np.empty((0, 0, 0))), dtype=float)
        metadata["oi_size"] = list(photons.shape)
        metadata["oi_sample_spacing_m"] = float(oi.fields.get("sample_spacing_m") or 0.0)
        metadata["oi_width_m"] = float(oi.fields.get("width_m") or 0.0)
        metadata["oi_height_m"] = float(oi.fields.get("height_m") or 0.0)
        metadata["oi_image_distance_m"] = float(oi.fields.get("image_distance_m") or 0.0)

    sensor = context.get("sensor")
    if sensor is not None:
        metadata["sensor_size"] = [int(sensor.fields["size"][0]), int(sensor.fields["size"][1])]
        integration_time = np.asarray(sensor.fields.get("integration_time", 0.0), dtype=float)
        if integration_time.ndim == 0 or integration_time.size == 1:
            metadata["sensor_integration_time_s"] = float(integration_time.reshape(-1)[0])
        else:
            metadata["sensor_integration_time_s"] = integration_time.reshape(-1).tolist()
        metadata["sensor_noise_flag"] = int(sensor.fields.get("noise_flag", 0))

    return metadata


def _sensor_from_reference_oi(context: dict[str, Any], reference_photons: Any):
    oi = context.get("oi")
    sensor = context.get("sensor")
    if oi is None or sensor is None:
        return None
    reference_oi = oi.clone()
    reference_oi.data["photons"] = np.asarray(reference_photons, dtype=float).copy()
    return sensor_compute(sensor.clone(), reference_oi, seed=0)


def _case_diagnostics(
    case_name: str,
    *,
    reference: dict[str, Any],
    context: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    metadata = _context_metadata(context)
    if metadata:
        diagnostics["context"] = metadata

    if case_name == "sensor_bayer_noiseless" and {"volts", "integration_time"} <= reference.keys():
        oi_reference = _load_reference("oi_diffraction_limited_default")
        if "photons" in oi_reference:
            recomputed_sensor = _sensor_from_reference_oi(context, oi_reference["photons"])
            if recomputed_sensor is not None:
                diagnostics["reference_recompute"] = {
                    "reference_oi_case": "oi_diffraction_limited_default",
                    "sensor_volts": _compare(reference["volts"], recomputed_sensor.data["volts"], rtol=rtol, atol=atol),
                    "integration_time": _compare(
                        reference["integration_time"],
                        recomputed_sensor.fields["integration_time"],
                        rtol=rtol,
                        atol=atol,
                    ),
                }

    if case_name == "camera_default_pipeline" and {"sensor_volts", "oi_photons"} <= reference.keys():
        recomputed_sensor = _sensor_from_reference_oi(context, reference["oi_photons"])
        if recomputed_sensor is not None:
            diagnostics["reference_recompute"] = {
                "sensor_volts_from_reference_oi": _compare(
                    reference["sensor_volts"],
                    recomputed_sensor.data["volts"],
                    rtol=rtol,
                    atol=atol,
                )
            }

    return diagnostics


def build_report(*, asset_store: AssetStore | None = None) -> dict[str, Any]:
    store = asset_store or AssetStore.default()
    cases = _case_definitions()
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped = 0

    for case in cases:
        case_name = str(case["name"])
        field_rules = _field_rules(case)
        baseline_path = BASELINES_DIR / f"{case_name}.mat"
        if not baseline_path.exists():
            skipped += 1
            results.append(
                {
                    "name": case_name,
                    "status": "skipped",
                    "reason": f"Missing baseline: {baseline_path}",
                }
            )
            continue

        reference = _load_reference(case_name)
        case_result = run_python_case_with_context(case_name, asset_store=store)
        comparison = _compare(
            reference,
            case_result.payload,
            rtol=float(case["rtol"]),
            atol=float(case["atol"]),
            field_rules=field_rules,
        )
        diagnostics = _case_diagnostics(
            case_name,
            reference=reference,
            context=case_result.context,
            rtol=float(case["rtol"]),
            atol=float(case["atol"]),
        )
        status = "passed" if comparison["pass"] else "failed"
        if comparison["pass"]:
            passed += 1
        else:
            failed += 1
        result = {
            "name": case_name,
            "status": status,
            "rtol": float(case["rtol"]),
            "atol": float(case["atol"]),
            "comparison": comparison,
        }
        if field_rules:
            result["field_overrides"] = field_rules
        if diagnostics:
            result["diagnostics"] = diagnostics
        results.append(result)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(cases),
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON path. Defaults to reports/parity/latest.json",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when any parity case fails.",
    )
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(
        f"Parity report written to {args.output} "
        f"({summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped)."
    )
    return 1 if args.fail_on_mismatch and int(summary["failed"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
