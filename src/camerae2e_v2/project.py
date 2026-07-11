"""CameraE2E v2 project lifecycle."""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

from .models import ProjectInfo, StudyCreate, StudyRecord, utc_now
from .storage import ArtifactRegistry, ProjectStore


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized.lower() or "camerae2e-project"


def _project_toml(info: ProjectInfo) -> str:
    created = info.created_at.isoformat()
    return (
        'schema_version = "camerae2e_project_v2"\n'
        f'id = "{info.id}"\n'
        f"name = {json.dumps(info.name, ensure_ascii=True)}\n"
        f'created_at = "{created}"\n'
        '\n[storage]\nmetadata = "project.db"\nartifacts = "artifacts"\n'
    )


class Project:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.store = ProjectStore(self.root)
        self.artifacts = ArtifactRegistry(self.store)

    @classmethod
    def create(cls, root: str | Path, *, name: str) -> Project:
        path = Path(root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "project.toml"
        if marker.exists():
            raise FileExistsError(f"CameraE2E project already exists: {path}")
        info = ProjectInfo(name=name, path=str(path))
        project = cls(path)
        project.store.initialize(info)
        marker.write_text(_project_toml(info), encoding="utf-8")
        (path / "runs").mkdir(exist_ok=True)
        (path / "exports").mkdir(exist_ok=True)
        (path / "reports").mkdir(exist_ok=True)
        return project

    @classmethod
    def open(cls, root: str | Path) -> Project:
        path = Path(root).expanduser().resolve()
        marker = path / "project.toml"
        if not marker.is_file():
            raise FileNotFoundError(f"Not a CameraE2E v2 project: {path}")
        with marker.open("rb") as stream:
            payload = tomllib.load(stream)
        if payload.get("schema_version") != "camerae2e_project_v2":
            raise ValueError(
                f"Unsupported CameraE2E project schema: {payload.get('schema_version')}"
            )
        project = cls(path)
        project.store.ensure_schema()
        project.store.project_info()
        return project

    @property
    def info(self) -> ProjectInfo:
        return self.store.project_info()

    def create_study(self, spec: StudyCreate) -> StudyRecord:
        record = StudyRecord(project_id=self.info.id, spec=spec)
        return self.store.put_study(record)

    def update_study(self, study_id: str, spec: StudyCreate) -> StudyRecord:
        current = self.store.get_study(study_id)
        updated = current.model_copy(
            update={
                "revision": current.revision + 1,
                "spec": spec,
                "updated_at": utc_now(),
            }
        )
        return self.store.put_study(updated)


class ProjectManager:
    """Discover and create fully local CameraE2E projects."""

    def __init__(self, root: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parents[2] / "camerae2e-workbench" / "projects"
        self.root = (
            Path(root or os.environ.get("CAMERAE2E_PROJECTS_ROOT", str(default)))
            .expanduser()
            .resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)
        for marker in self.root.glob("*/project.toml"):
            try:
                Project.open(marker.parent).store.recover_interrupted_jobs()
            except (OSError, ValueError):
                continue

    def create(self, name: str, *, directory_name: str | None = None) -> Project:
        base = _slug(directory_name or name)
        path = self.root / base
        if path.exists():
            index = 2
            while (self.root / f"{base}-{index}").exists():
                index += 1
            path = self.root / f"{base}-{index}"
        return Project.create(path, name=name)

    def open(self, project_id_or_path: str | Path) -> Project:
        candidate = Path(project_id_or_path).expanduser()
        if candidate.is_absolute() or candidate.exists():
            return Project.open(candidate)
        for project in self.list():
            if project.info.id == str(project_id_or_path) or project.root.name == str(
                project_id_or_path
            ):
                return project
        raise KeyError(f"Unknown CameraE2E project: {project_id_or_path}")

    def list(self) -> list[Project]:
        projects = []
        for marker in sorted(self.root.glob("*/project.toml")):
            try:
                projects.append(Project.open(marker.parent))
            except (OSError, ValueError):
                continue
        return projects
