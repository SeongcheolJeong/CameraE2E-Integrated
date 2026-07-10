"""Local CameraE2E v2 command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .migration import import_legacy_db, migrate_v1_settings
from .project import Project
from .service import CameraE2EService


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camerae2e", description="CameraE2E v2 local research platform"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a v2 project")
    init.add_argument("path")
    init.add_argument("--name", default="CameraE2E Project")
    init.add_argument("--preset", default="adas", choices=["adas", "general"])

    doctor = sub.add_parser("doctor", help="Validate project artifacts")
    doctor.add_argument("project")

    run = sub.add_parser("run", help="Run one study operation synchronously")
    run.add_argument("project")
    run.add_argument("study_id")
    run.add_argument(
        "kind",
        choices=[
            "evaluate",
            "sensitivity",
            "optimize",
            "validate_candidate",
            "dataset_export",
            "report",
        ],
    )
    run.add_argument("--request", help="JSON request file")

    migrate = sub.add_parser("migrate", help="Migrate v1 settings into a v2 project")
    migrate.add_argument("project")
    migrate.add_argument("source")
    migrate.add_argument("--db", action="store_true", help="Import a v1 camera DB tree")

    artifacts = sub.add_parser("artifacts", help="List project artifacts")
    artifacts.add_argument("project")
    artifacts.add_argument("--type")

    assets = sub.add_parser("assets", help="List canonical camera assets")
    assets.add_argument("project")
    assets.add_argument("--kind")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        path = Path(args.path).expanduser().resolve()
        project = Project.create(path, name=args.name)
        service = CameraE2EService(path.parent)
        project.create_study(service.default_study_spec(args.preset))
        _print(service.project_payload(project))
        service.shutdown()
        return 0
    if args.command == "doctor":
        project = Project.open(args.project)
        payload = project.artifacts.validate()
        _print(payload)
        return 0 if payload["ok"] else 1
    if args.command == "run":
        project = Project.open(args.project)
        request = json.loads(Path(args.request).read_text(encoding="utf-8")) if args.request else {}
        service = CameraE2EService(project.root.parent, max_workers=1)
        job = service.execute_job_now(project.info.id, args.study_id, args.kind, request)
        _print(job.model_dump(mode="json"))
        service.shutdown()
        return 0 if job.status.value == "succeeded" else 1
    if args.command == "migrate":
        project = Project.open(args.project)
        payload = (
            import_legacy_db(project, args.source)
            if args.db
            else migrate_v1_settings(project, args.source)
        )
        _print(payload)
        return 0
    if args.command == "artifacts":
        project = Project.open(args.project)
        _print(
            [
                item.model_dump(mode="json")
                for item in project.store.list_artifacts(artifact_type=args.type)
            ]
        )
        return 0
    if args.command == "assets":
        project = Project.open(args.project)
        _print(
            [
                item.model_dump(mode="json")
                for item in project.store.list_camera_assets(kind=args.kind)
            ]
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
