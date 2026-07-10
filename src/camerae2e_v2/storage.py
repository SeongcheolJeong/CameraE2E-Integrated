"""Persistent project metadata and content-addressed artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import (
    ArtifactRecord,
    CameraAssetRecord,
    FidelityLevel,
    JobRecord,
    JobStatus,
    ProjectInfo,
    ReadinessTier,
    StudyRecord,
    utc_now,
)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS project (
    id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    study_id TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS jobs_study_idx ON jobs(study_id, created_at);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS artifacts (
    hash TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    fidelity_level TEXT NOT NULL,
    readiness_tier TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS artifacts_type_idx ON artifacts(artifact_type, created_at);

CREATE TABLE IF NOT EXISTS camera_assets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    readiness_tier TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS camera_assets_kind_idx ON camera_assets(kind, name);

CREATE TABLE IF NOT EXISTS job_artifacts (
    job_id TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(job_id, artifact_hash, role),
    FOREIGN KEY(job_id) REFERENCES jobs(id),
    FOREIGN KEY(artifact_hash) REFERENCES artifacts(hash)
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(study_id) REFERENCES studies(id),
    FOREIGN KEY(artifact_hash) REFERENCES artifacts(hash)
);
"""


def _json(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class ProjectStore:
    """SQLite metadata store scoped to one project directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / "project.db"
        self._lock = threading.RLock()

    def initialize(self, info: ProjectInfo) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO project(id, payload_json) VALUES (?, ?)",
                (info.id, _json(info)),
            )

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def project_info(self) -> ProjectInfo:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM project LIMIT 1").fetchone()
        if row is None:
            raise FileNotFoundError(f"Project metadata is missing in {self.db_path}")
        payload = json.loads(str(row["payload_json"]))
        payload["path"] = str(self.root)
        return ProjectInfo.model_validate(payload)

    def put_study(self, study: StudyRecord) -> StudyRecord:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO studies(
                    id, project_id, revision, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study.id,
                    study.project_id,
                    study.revision,
                    study.status,
                    _json(study),
                    study.created_at.isoformat(),
                    study.updated_at.isoformat(),
                ),
            )
        return study

    def get_study(self, study_id: str) -> StudyRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM studies WHERE id = ?", (study_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown study: {study_id}")
        return StudyRecord.model_validate_json(str(row["payload_json"]))

    def list_studies(self) -> list[StudyRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM studies ORDER BY updated_at DESC"
            ).fetchall()
        return [StudyRecord.model_validate_json(str(row["payload_json"])) for row in rows]

    def put_job(self, job: JobRecord) -> JobRecord:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO jobs(
                    id, project_id, study_id, kind, status, progress, payload_json,
                    created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.project_id,
                    job.study_id,
                    job.kind,
                    job.status.value,
                    job.progress,
                    _json(job),
                    job.created_at.isoformat(),
                    None if job.started_at is None else job.started_at.isoformat(),
                    None if job.finished_at is None else job.finished_at.isoformat(),
                ),
            )
        return job

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job: {job_id}")
        return JobRecord.model_validate_json(str(row["payload_json"]))

    def list_jobs(self, study_id: str | None = None, *, limit: int = 100) -> list[JobRecord]:
        query = "SELECT payload_json FROM jobs"
        params: tuple[Any, ...] = ()
        if study_id is not None:
            query += " WHERE study_id = ?"
            params = (study_id,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params += (max(int(limit), 1),)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [JobRecord.model_validate_json(str(row["payload_json"])) for row in rows]

    def recover_interrupted_jobs(self) -> int:
        count = 0
        for job in self.list_jobs(limit=10_000):
            if job.status != JobStatus.RUNNING:
                continue
            job.status = JobStatus.INTERRUPTED
            job.error = "Workbench stopped while this local job was running."
            job.finished_at = utc_now()
            job.log.append("Recovered running job as interrupted during project open.")
            self.put_job(job)
            count += 1
        return count

    def put_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM artifacts WHERE hash = ?", (artifact.hash,)
            ).fetchone()
            if row is None:
                merged = artifact
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        hash, artifact_type, relative_path, size_bytes, media_type,
                        fidelity_level, readiness_tier, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        merged.hash,
                        merged.artifact_type,
                        merged.relative_path,
                        merged.size_bytes,
                        merged.media_type,
                        merged.fidelity_level.value,
                        merged.readiness_tier.value,
                        _json(merged),
                        merged.created_at.isoformat(),
                    ),
                )
            else:
                existing = ArtifactRecord.model_validate_json(str(row["payload_json"]))
                alternate_sources = set(existing.metadata.get("alternate_sources", []))
                alternate_types = set(existing.metadata.get("alternate_artifact_types", []))
                if artifact.source != existing.source:
                    alternate_sources.add(artifact.source)
                if artifact.artifact_type != existing.artifact_type:
                    alternate_types.add(artifact.artifact_type)
                merged_metadata = {**existing.metadata, **artifact.metadata}
                if alternate_sources:
                    merged_metadata["alternate_sources"] = sorted(alternate_sources)
                if alternate_types:
                    merged_metadata["alternate_artifact_types"] = sorted(alternate_types)
                merged = existing.model_copy(
                    update={
                        "dependencies": sorted(
                            set(existing.dependencies) | set(artifact.dependencies)
                        ),
                        "validation": {**existing.validation, **artifact.validation},
                        "metadata": merged_metadata,
                    }
                )
                connection.execute(
                    "UPDATE artifacts SET payload_json = ? WHERE hash = ?",
                    (_json(merged), merged.hash),
                )
        return self.get_artifact(artifact.hash)

    def get_artifact(self, artifact_hash: str) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM artifacts WHERE hash = ?", (artifact_hash,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown artifact: {artifact_hash}")
        return ArtifactRecord.model_validate_json(str(row["payload_json"]))

    def list_artifacts(self, artifact_type: str | None = None) -> list[ArtifactRecord]:
        query = "SELECT payload_json FROM artifacts"
        params: tuple[Any, ...] = ()
        if artifact_type is not None:
            query += " WHERE artifact_type = ?"
            params = (artifact_type,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [ArtifactRecord.model_validate_json(str(row["payload_json"])) for row in rows]

    def put_camera_asset(self, asset: CameraAssetRecord) -> CameraAssetRecord:
        known = {artifact.hash for artifact in self.list_artifacts()}
        missing = [value for value in asset.artifact_hashes if value not in known]
        if missing:
            raise ValueError(f"Camera asset references unknown artifacts: {missing}")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO camera_assets(
                    id, kind, name, version, readiness_tier, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.id,
                    asset.kind.value,
                    asset.name,
                    asset.version,
                    asset.readiness_tier.value,
                    _json(asset),
                    asset.created_at.isoformat(),
                ),
            )
        return asset

    def get_camera_asset(self, asset_id: str) -> CameraAssetRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM camera_assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown camera asset: {asset_id}")
        return CameraAssetRecord.model_validate_json(str(row["payload_json"]))

    def list_camera_assets(self, kind: str | None = None) -> list[CameraAssetRecord]:
        query = "SELECT payload_json FROM camera_assets"
        params: tuple[Any, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            params = (str(kind),)
        query += " ORDER BY kind, name, version"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [CameraAssetRecord.model_validate_json(str(row["payload_json"])) for row in rows]

    def link_job_artifact(self, job_id: str, artifact_hash: str, role: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO job_artifacts(job_id, artifact_hash, role) VALUES (?, ?, ?)",
                (job_id, artifact_hash, role),
            )

    def job_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT ja.role, a.payload_json
                FROM job_artifacts ja JOIN artifacts a ON a.hash = ja.artifact_hash
                WHERE ja.job_id = ? ORDER BY ja.role, a.created_at
                """,
                (job_id,),
            ).fetchall()
        return [
            {"role": str(row["role"]), "artifact": json.loads(str(row["payload_json"]))}
            for row in rows
        ]


class ArtifactRegistry:
    """Immutable, hash-addressed file storage for one project."""

    _EXTENSIONS = {
        "application/json": ".json",
        "application/x-npy": ".npy",
        "application/x-npz": ".npz",
        "image/png": ".png",
        "text/html": ".html",
        "text/plain": ".txt",
    }

    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.root = store.root / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(
        self,
        payload: Any,
        *,
        artifact_type: str,
        fidelity_level: FidelityLevel = FidelityLevel.ANALYTIC,
        readiness_tier: ReadinessTier = ReadinessTier.VALIDATED,
        source: str,
        dependencies: Iterable[str] = (),
        validation: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        content = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return self.put_bytes(
            content,
            artifact_type=artifact_type,
            media_type="application/json",
            fidelity_level=fidelity_level,
            readiness_tier=readiness_tier,
            source=source,
            dependencies=dependencies,
            validation=validation,
            metadata=metadata,
        )

    def put_file(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        media_type: str,
        fidelity_level: FidelityLevel,
        readiness_tier: ReadinessTier,
        source: str,
        dependencies: Iterable[str] = (),
        validation: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        return self.put_bytes(
            source_path.read_bytes(),
            artifact_type=artifact_type,
            media_type=media_type,
            fidelity_level=fidelity_level,
            readiness_tier=readiness_tier,
            source=source,
            dependencies=dependencies,
            validation=validation,
            metadata={**dict(metadata or {}), "original_name": source_path.name},
            extension=source_path.suffix,
        )

    def put_bytes(
        self,
        content: bytes,
        *,
        artifact_type: str,
        media_type: str,
        fidelity_level: FidelityLevel,
        readiness_tier: ReadinessTier,
        source: str,
        dependencies: Iterable[str] = (),
        validation: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        extension: str | None = None,
    ) -> ArtifactRecord:
        dependency_list = [str(value) for value in dependencies]
        missing = [value for value in dependency_list if not self._has_artifact(value)]
        if missing:
            raise ValueError(f"Artifact dependencies are not registered: {missing}")
        digest = hashlib.sha256(content).hexdigest()
        suffix = extension or self._EXTENSIONS.get(media_type, ".bin")
        if suffix and not suffix.startswith("."):
            suffix = "." + suffix
        relative = Path(digest[:2]) / digest[2:4] / f"{digest}{suffix}"
        destination = self.root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        record = ArtifactRecord(
            hash=digest,
            artifact_type=artifact_type,
            relative_path=str(Path("artifacts") / relative),
            size_bytes=len(content),
            media_type=media_type,
            fidelity_level=fidelity_level,
            readiness_tier=readiness_tier,
            source=source,
            dependencies=dependency_list,
            validation=dict(validation or {}),
            metadata=dict(metadata or {}),
        )
        return self.store.put_artifact(record)

    def resolve(self, artifact_hash: str) -> Path:
        record = self.store.get_artifact(artifact_hash)
        path = self.store.root / record.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Artifact {artifact_hash} is registered but missing at {path}")
        return path

    def validate(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        records = self.store.list_artifacts()
        known = {record.hash for record in records}
        for record in records:
            path = self.store.root / record.relative_path
            if not path.is_file():
                issues.append({"hash": record.hash, "kind": "missing_file", "path": str(path)})
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record.hash:
                issues.append({"hash": record.hash, "kind": "hash_mismatch", "actual": digest})
            for dependency in record.dependencies:
                if dependency not in known:
                    issues.append(
                        {"hash": record.hash, "kind": "stale_dependency", "dependency": dependency}
                    )
        return {
            "schema_version": "camerae2e_artifact_validation_v2",
            "ok": not issues,
            "artifact_count": len(records),
            "issue_count": len(issues),
            "issues": issues,
        }

    def import_tree(
        self,
        root: str | Path,
        *,
        artifact_type: str,
        source: str,
        include: tuple[str, ...] = ("*.json", "*.npz", "*.sqlite"),
        fidelity_level: FidelityLevel = FidelityLevel.LUT,
        readiness_tier: ReadinessTier = ReadinessTier.PROXY,
    ) -> list[ArtifactRecord]:
        source_root = Path(root).expanduser().resolve()
        records = []
        for pattern in include:
            for path in sorted(source_root.rglob(pattern)):
                media = "application/json" if path.suffix == ".json" else "application/octet-stream"
                records.append(
                    self.put_file(
                        path,
                        artifact_type=artifact_type,
                        media_type=media,
                        fidelity_level=fidelity_level,
                        readiness_tier=readiness_tier,
                        source=source,
                        metadata={"source_relative_path": str(path.relative_to(source_root))},
                    )
                )
        return records

    def _has_artifact(self, artifact_hash: str) -> bool:
        try:
            self.store.get_artifact(artifact_hash)
        except KeyError:
            return False
        return True


def copy_project_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
