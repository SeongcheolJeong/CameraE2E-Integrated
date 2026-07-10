from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from camerae2e_v2 import CameraE2EService  # noqa: E402

from .v2 import create_v2_router  # noqa: E402

app = FastAPI(
    title="CameraE2E v2 Workbench API",
    version="2.0.0",
    description="Local camera design decision and evidence platform.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4175",
        "http://127.0.0.1:4175",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

v2_service = CameraE2EService()
app.include_router(create_v2_router(v2_service))


@app.on_event("shutdown")
def shutdown_v2_service() -> None:
    v2_service.shutdown(wait=False)
