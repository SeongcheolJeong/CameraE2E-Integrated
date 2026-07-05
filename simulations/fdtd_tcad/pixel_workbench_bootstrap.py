#!/usr/bin/env python3
"""Bootstrap generated local assets for the Pixel Workbench.

The repository intentionally does not commit generated outputs under runs/.
This script rebuilds the local reference assets needed by the backend UI:

1. install UI dependencies when node_modules is missing,
2. build the React UI,
3. generate the CAD template catalog,
4. generate the reference studio under runs/image_sensor_pixel_studio_reference.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "pixel_workbench_ui"
DEFAULT_CAD_PYTHON = ROOT / ".tcad-env" / "bin" / "python"


def run_step(label: str, command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n==> {label}", flush=True)
    print(" ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing executable for step '{label}': {command[0]}") from error


def run_quiet_step(label: str, command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n==> {label}", flush=True)
    print(" ".join(command), flush=True)
    try:
        completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing executable for step '{label}': {command[0]}") from error
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if completed.returncode:
        tail = "\n".join(output.splitlines()[-120:])
        raise SystemExit(f"Step failed: {label}\n{tail}")
    summary_lines = [
        line
        for line in output.splitlines()
        if line.strip().startswith('"manifest"')
        or line.strip().startswith('"validation_report"')
        or line.strip().startswith('"template_count"')
        or line.strip().startswith('"status"')
    ]
    if summary_lines:
        print("\n".join(summary_lines), flush=True)
    else:
        print("done", flush=True)


def resolve_cad_python(raw: str | None) -> str:
    if raw:
        return raw
    if DEFAULT_CAD_PYTHON.exists():
        return str(DEFAULT_CAD_PYTHON)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build generated local assets for the Image Sensor Pixel Workbench."
    )
    parser.add_argument("--skip-ui-deps", action="store_true", help="Do not run npm install when node_modules is missing.")
    parser.add_argument("--skip-ui-build", action="store_true", help="Do not run npm run build.")
    parser.add_argument("--skip-cad", action="store_true", help="Do not generate the CAD template catalog.")
    parser.add_argument("--skip-studio", action="store_true", help="Do not generate the reference studio output.")
    parser.add_argument("--cad-mesh", action="store_true", help="Also generate coarse model.msh CAD review meshes. This is noisier and slower.")
    parser.add_argument("--no-cad-mesh", action="store_false", dest="cad_mesh", help=argparse.SUPPRESS)
    parser.add_argument("--replace-cad-catalog", action="store_true", help="Replace the CAD template manifest instead of appending/updating records.")
    parser.add_argument("--cad-python", help="Python executable with gmsh available. Defaults to .tcad-env/bin/python if present.")
    parser.add_argument("--studio-config", default="configs/image_sensor_pixel_studio_reference.json")
    parser.add_argument("--studio-output-dir", default="runs/image_sensor_pixel_studio_reference")
    parser.add_argument("--cad-output-dir", default="runs/pixel_cad_template_library_reference")
    parser.add_argument("--server-port", type=int, default=8766)
    args = parser.parse_args()

    if not args.skip_ui_deps and not (UI_DIR / "node_modules").exists():
        run_step("Install UI dependencies", ["npm", "install"], cwd=UI_DIR)
    elif not args.skip_ui_deps:
        print("UI dependencies already exist; skipping npm install.")

    if not args.skip_ui_build:
        run_step("Build React UI", ["npm", "run", "build"], cwd=UI_DIR)

    if not args.skip_cad:
        cad_command = [
            resolve_cad_python(args.cad_python),
            "pixel_cad_template_library.py",
            "--output-dir",
            args.cad_output_dir,
        ]
        if not args.replace_cad_catalog:
            cad_command.append("--append")
        if args.cad_mesh:
            cad_command.append("--mesh")
        if args.cad_mesh:
            run_step("Generate CAD template catalog", cad_command)
        else:
            run_quiet_step("Generate CAD template catalog", cad_command)

    if not args.skip_studio:
        run_step(
            "Generate reference studio",
            [
                sys.executable,
                "image_sensor_pixel_studio.py",
                "--config",
                args.studio_config,
                "--output-dir",
                args.studio_output_dir,
            ],
        )

    studio_url = f"http://127.0.0.1:{args.server_port}/{args.studio_output_dir}/index.html"
    print("\nBootstrap complete.")
    print(f"Start backend: python3 pixel_workbench_server.py --port {args.server_port}")
    print(f"Open: {studio_url}")


if __name__ == "__main__":
    # Keep subprocess output unbuffered enough to make long CAD generation visible.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
