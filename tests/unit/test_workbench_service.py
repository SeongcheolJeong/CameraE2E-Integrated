from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBENCH_ROOT = REPO_ROOT / "camerae2e-workbench"
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from backend.app.service import WorkbenchService  # noqa: E402


def test_workbench_simulate_allows_ocl_off_camera_preset() -> None:
    service = WorkbenchService()
    settings = service.default_settings()
    settings.update(
        {
            "cameraPreset": "narrow_fov_adas_demo",
            "fovDeg": 61.2,
            "pixelSizeUm": 2.8,
            "fNumber": 2.4,
            "oclMode": "off",
        }
    )

    result = service.simulate({"settings": settings})

    assert result["preview_png"].startswith("data:image/png;base64,")
    assert any(metric["id"] == "rgb_mean" and metric["value"] is not None for metric in result["metrics"])
