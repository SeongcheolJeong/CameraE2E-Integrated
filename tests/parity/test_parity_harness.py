from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import loadmat

from pyisetcam import run_python_case

if os.environ.get("PYISETCAM_RUN_PARITY") != "1":
    pytestmark = pytest.mark.skip(reason="Set PYISETCAM_RUN_PARITY=1 to execute Octave parity checks.")


CASES_PATH = Path(__file__).with_name("cases.yaml")
BASELINES_DIR = Path(__file__).with_name("baselines")


def _case_definitions() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())["cases"]


def _field_rules(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rules = case.get("field_overrides", {})
    if not isinstance(raw_rules, dict):
        return {}
    return {str(key): value for key, value in raw_rules.items() if isinstance(value, dict)}


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
        if np.iscomplexobj(value):
            value = np.real(value)
        return np.squeeze(np.asarray(value))
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


def _compare(
    reference: Any,
    actual: Any,
    *,
    rtol: float,
    atol: float,
    field_rules: dict[str, dict[str, Any]] | None = None,
    path: tuple[str, ...] = (),
) -> None:
    reference = _normalize(reference)
    actual = _normalize(actual)
    rule = field_rules.get(path[0]) if field_rules and path else None
    if isinstance(reference, dict):
        for key in reference:
            if key.startswith("__"):
                continue
            if key not in actual:
                raise AssertionError(f"Missing key '{key}' in actual output.")
            _compare(
                reference[key],
                actual[key],
                rtol=rtol,
                atol=atol,
                field_rules=field_rules,
                path=(*path, key),
            )
        return
    if isinstance(reference, (str, bytes)) or isinstance(actual, (str, bytes)):
        assert actual == reference
        return
    if np.isscalar(reference) and np.isscalar(actual):
        reference_scalar = float(reference)
        actual_scalar = float(actual)
        rel_diff = abs(actual_scalar - reference_scalar) / max(abs(reference_scalar), 1e-12)
        if rule and rule.get("mode") in {"mean_rel", "normalized_mae", "mae"}:
            if rule["mode"] == "mean_rel":
                assert rel_diff <= float(rule["max_mean_rel"])
            elif rule["mode"] == "normalized_mae":
                assert rel_diff <= float(rule["max_normalized_mae"])
            else:
                assert difference <= float(rule["max_mae"])
            return
        if rule and rule.get("mode") == "scale_invariant":
            if abs(actual_scalar) <= 0.0:
                raise AssertionError("Scale-invariant comparison requires nonzero actual data.")
            scale = reference_scalar / actual_scalar
            scaled_actual = scale * actual_scalar
            if "max_mean_rel" in rule:
                scaled_rel_diff = abs(scaled_actual - reference_scalar) / max(abs(reference_scalar), 1e-12)
                assert scaled_rel_diff <= float(rule["max_mean_rel"])
            else:
                assert np.isclose(scaled_actual, reference_scalar, rtol=rtol, atol=atol)
            return
        assert np.isclose(actual_scalar, reference_scalar, rtol=rtol, atol=atol)
        return
    if _is_string_like_array(reference) or _is_string_like_array(actual):
        reference_array = np.asarray(reference)
        actual_array = np.asarray(actual)
        assert reference_array.shape == actual_array.shape
        assert _string_array_payload(actual_array) == _string_array_payload(reference_array)
        return
    reference_array = np.asarray(reference, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    assert reference_array.shape == actual_array.shape
    if rule and rule.get("mode") == "mean_rel":
        relative = np.abs(actual_array - reference_array) / np.maximum(np.abs(reference_array), 1e-12)
        assert float(np.mean(relative)) <= float(rule["max_mean_rel"])
        return
    if rule and rule.get("mode") == "normalized_mae":
        normalized_mae = float(np.mean(np.abs(actual_array - reference_array))) / max(float(np.mean(np.abs(reference_array))), 1e-12)
        assert normalized_mae <= float(rule["max_normalized_mae"])
        return
    if rule and rule.get("mode") == "mae":
        mae = float(np.mean(np.abs(actual_array - reference_array)))
        assert mae <= float(rule["max_mae"])
        return
    if rule and rule.get("mode") == "scale_invariant":
        reference_flat = reference_array.reshape(-1)
        actual_flat = actual_array.reshape(-1)
        denominator = float(np.dot(actual_flat, actual_flat))
        if denominator <= 0.0:
            raise AssertionError("Scale-invariant comparison requires nonzero actual data.")
        scale = float(np.dot(reference_flat, actual_flat)) / denominator
        scaled_actual = scale * actual_array
        if "max_mean_rel" in rule:
            relative = np.abs(scaled_actual - reference_array) / np.maximum(np.abs(reference_array), 1e-12)
            assert float(np.mean(relative)) <= float(rule["max_mean_rel"])
        else:
            assert np.allclose(scaled_actual, reference_array, rtol=rtol, atol=atol)
        return
    assert np.allclose(actual_array, reference_array, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", _case_definitions(), ids=lambda case: case["name"])
def test_parity_case(case, asset_store) -> None:
    baseline_path = BASELINES_DIR / f"{case['name']}.mat"
    if not baseline_path.exists():
        pytest.skip(f"Missing baseline: {baseline_path}")
    reference = {key: value for key, value in loadmat(baseline_path, squeeze_me=True, struct_as_record=False).items() if not key.startswith("__")}
    actual = run_python_case(case["name"], asset_store=asset_store)
    _compare(
        reference,
        actual,
        rtol=float(case["rtol"]),
        atol=float(case["atol"]),
        field_rules=_field_rules(case),
    )
