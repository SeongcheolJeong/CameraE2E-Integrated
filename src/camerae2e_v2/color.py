"""Constrained color-correction fitting with deterministic holdout gates."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_constrained_ccm(
    sensor_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    regularization: float = 0.02,
    max_abs: float = 4.0,
    row_sum_tolerance: float = 0.15,
    holdout_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    source = np.asarray(sensor_rgb, dtype=float)
    target = np.asarray(target_rgb, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("CCM fitting requires matching N x 3 sensor and target arrays")
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 12:
        raise ValueError("CCM fitting requires at least 12 finite color samples")
    rng = np.random.default_rng(seed)
    order = rng.permutation(source.shape[0])
    holdout_count = max(3, int(round(source.shape[0] * holdout_fraction)))
    holdout_count = min(holdout_count, source.shape[0] - 6)
    holdout_index = order[:holdout_count]
    train_index = order[holdout_count:]
    train_source = source[train_index]
    train_target = target[train_index]
    identity = np.eye(3, dtype=float)
    normal = train_source.T @ train_source + regularization * np.eye(3)
    rhs = train_source.T @ train_target + regularization * identity
    matrix = np.linalg.solve(normal, rhs)
    matrix = np.clip(matrix, -max_abs, max_abs)
    column_sums = np.sum(matrix, axis=0)
    matrix += (1.0 - column_sums)[None, :] / 3.0
    matrix = np.clip(matrix, -max_abs, max_abs)
    train_metrics = _ccm_metrics(train_source @ matrix, train_target)
    holdout_metrics = _ccm_metrics(source[holdout_index] @ matrix, target[holdout_index])
    identity_holdout = _ccm_metrics(source[holdout_index], target[holdout_index])
    neutral_error = float(np.max(np.abs(np.sum(matrix, axis=0) - 1.0)))
    gates = [
        {
            "id": "holdout_improves_identity",
            "pass": holdout_metrics["rmse"] < identity_holdout["rmse"],
            "value": holdout_metrics["rmse"],
            "limit": identity_holdout["rmse"],
        },
        {
            "id": "neutral_preservation",
            "pass": neutral_error <= row_sum_tolerance,
            "value": neutral_error,
            "limit": row_sum_tolerance,
        },
        {
            "id": "coefficient_bound",
            "pass": float(np.max(np.abs(matrix))) <= max_abs,
            "value": float(np.max(np.abs(matrix))),
            "limit": max_abs,
        },
    ]
    return {
        "schema_version": "camerae2e_constrained_ccm_v2",
        "matrix": matrix.tolist(),
        "sample_count": int(source.shape[0]),
        "train_sample_count": int(train_index.size),
        "holdout_sample_count": int(holdout_index.size),
        "regularization": regularization,
        "constraints": {
            "max_abs": max_abs,
            "neutral_sum_tolerance": row_sum_tolerance,
        },
        "train": train_metrics,
        "holdout": holdout_metrics,
        "identity_holdout": identity_holdout,
        "gates": gates,
        "validated": all(item["pass"] for item in gates),
        "truth_boundary": (
            "The CCM is calibrated only for the supplied color samples and holdout domain."
        ),
    }


def _ccm_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(target, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
    }
