"""Validate CameraE2E external physics pipeline lineage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pyisetcam import (
    camerae2e_db_get,
    camerae2e_db_lineage,
    camerae2e_db_validate,
    camerae2e_physics_pipeline_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = validate_physics_pipeline(strict=args.strict)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not payload["ok"]:
        raise SystemExit(1)


def validate_physics_pipeline(*, strict: bool = False) -> dict[str, Any]:
    validation = camerae2e_db_validate(strict=strict)
    entries = {
        name: camerae2e_db_get(name)
        for name in (
            "fdtd_sensor_lut_active",
            "tcad_sensor_db_active",
            "lens_patents_active",
            "hwisp_parameter_profiles",
        )
    }
    checks = {
        "fdtd_tcad_lineage": _fdtd_tcad_check(
            entries["fdtd_sensor_lut_active"], entries["tcad_sensor_db_active"]
        ),
        "rayoptics_truth_boundary": _rayoptics_check(entries["lens_patents_active"]),
        "hwisp_truth_boundary": _hwisp_check(entries["hwisp_parameter_profiles"]),
    }
    blocking = [name for name, check in checks.items() if check["status"] == "fail"]
    if strict:
        blocking.extend(
            item["entry"]
            for item in validation.get("issues", [])
            if item.get("kind")
            in {"readiness_tier", "missing_path", "unknown_dependency", "stale_dependency"}
        )
    return {
        "schema_version": "camerae2e_physics_pipeline_validation_v1",
        "strict": bool(strict),
        "ok": not blocking,
        "blocking": sorted(set(blocking)),
        "manifest_validation": validation,
        "checks": checks,
        "plan": camerae2e_physics_pipeline_plan(strict=strict),
        "lineage": {
            name: camerae2e_db_lineage(name)
            for name in ("fdtd_sensor_lut_active", "tcad_sensor_db_active", "lens_patents_active")
        },
    }


def _fdtd_tcad_check(fdtd: Mapping[str, Any], tcad: Mapping[str, Any]) -> dict[str, Any]:
    if tcad.get("stale_reason"):
        return {
            "status": "warn",
            "readiness": tcad.get("readiness_tier"),
            "message": tcad["stale_reason"],
        }
    return {
        "status": "pass" if fdtd.get("path") and tcad.get("path") else "warn",
        "readiness": tcad.get("readiness_tier"),
        "message": (
            "Active FDTD and TCAD catalog entries are present; quantitative "
            "accuracy still follows their readiness tiers."
        ),
    }


def _rayoptics_check(entry: Mapping[str, Any]) -> dict[str, Any]:
    boundary = str(entry.get("provenance", {}).get("truth_boundary", ""))
    status = "pass" if "geometric" in boundary and "diffraction" in boundary else "warn"
    return {
        "status": status,
        "readiness": entry.get("readiness_tier"),
        "message": (
            "RayOptics PSF assets are cataloged as geometric PSFs, "
            "not wave-optics sign-off."
        ),
    }


def _hwisp_check(entry: Mapping[str, Any]) -> dict[str, Any]:
    boundary = str(entry.get("provenance", {}).get("truth_boundary", ""))
    status = "pass" if "seed" in boundary and "signoff" in boundary else "warn"
    return {
        "status": status,
        "readiness": entry.get("readiness_tier"),
        "message": (
            "HW ISP profiles are seed/public-derived unless replaced by "
            "board measurements or vendor traces."
        ),
    }


if __name__ == "__main__":
    main()
