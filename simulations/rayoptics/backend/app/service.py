from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import numpy as np
import rayoptics as ro
import rayoptics.optical.model_constants as mc
import rayoptics.raytr.raytrace as rt
from rayoptics.elem import profiles
from rayoptics.raytr import trace as ro_trace
from opticalglass import glassfactory as gfact
from opticalglass import opticalmedium as om
from rayoptics.gui import appcmds
from rayoptics.gui.appcmds import open_model as ro_open_model
from rayoptics.mpl.analysisplots import FieldCurveFigure
from rayoptics.mpl.axisarrayfigure import Fit, RayFanFigure, SpotDiagramFigure, WavefrontFigure
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.raytr.opticalspec import Field, FieldSpec, FocusRange, PupilSpec, WvlSpec
from rayoptics.util.misc_math import normalize

from .schemas import (
    AnalysisRequest,
    AnalysisSummaryResponse,
    CockpitCountsDTO,
    CockpitMetricDTO,
    CockpitRiskDTO,
    DraftAutosaveResponse,
    ExampleCheckRequest,
    ExampleCheckResponse,
    ExampleCheckResultDTO,
    ExampleCheckStageDTO,
    ModelDTO,
    ModelResponse,
    ModelSettingsPatchRequest,
    NewModelRequest,
    QuickOptimizeMoveDTO,
    QuickOptimizeRequest,
    QuickOptimizeResponse,
    QuickOptimizeResultDTO,
    SensorAssumptionDTO,
    SensorPatchRequest,
    SurfaceCreateRequest,
    SurfaceDTO,
    SurfacePatchRequest,
    SystemDTO,
    SystemPatchRequest,
    ToleranceSweepCaseDTO,
    ToleranceSweepRequest,
    ToleranceSweepResponse,
    ToleranceSweepResultDTO,
)


SUPPORTED_EXTENSIONS = {".roa", ".seq", ".zmx"}
DEFAULT_PATENT_DB_PATH = Path("/Users/seongcheoljeong/Lens_Patent_DB/data/lens_simulation_expanded_v9.sqlite")
DRAFT_KEEP_LIMIT = 20
EXAMPLE_CHECK_PRIORITY = [
    "Sasian Triplet.roa",
    "singlet_f5.roa",
    "telephoto.roa",
    "cell_phone_camera.roa",
    "ag_dblgauss.seq",
    "US00583336-2-scaled.zmx",
]
EXAMPLE_ANALYSIS_KINDS = ["ray-fan", "spot", "wavefront"]
VARIABLE_TOKEN_ORDER = ["R", "T", "SD", "K"]
VARIABLE_TOKEN_ALIASES = {
    "R": "R",
    "RAD": "R",
    "RADIUS": "R",
    "CURVATURE": "R",
    "CV": "R",
    "T": "T",
    "THI": "T",
    "THICKNESS": "T",
    "SD": "SD",
    "SEMI": "SD",
    "SEMI-DIA": "SD",
    "SEMIDIA": "SD",
    "SEMIDIAMETER": "SD",
    "SEMI_DIAMETER": "SD",
    "K": "K",
    "CONIC": "K",
}


@dataclass
class ModelSession:
    model_id: str
    opt_model: Any
    filename: str | None = None
    dirty: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    update_mode: str = "auto"
    last_valid_model: Any | None = None
    undo_stack: list[Any] = field(default_factory=list)
    redo_stack: list[Any] = field(default_factory=list)
    surface_variables: dict[int, set[str]] = field(default_factory=dict)
    sensor_overrides: dict[str, Any] = field(default_factory=dict)
    workbench_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldTraceSummary:
    max_distortion_pct: float | None = None
    max_cra_deg: float | None = None
    max_spot_rms_um: float | None = None
    min_pupil_throughput: float | None = None
    worst_distortion_field: str | None = None
    worst_cra_field: str | None = None
    worst_spot_field: str | None = None
    worst_throughput_field: str | None = None
    worst_throughput_field_radius: float = -1.0
    traced_fields: int = 0
    spot_fields: int = 0
    spot_samples: int = 0
    throughput_fields: int = 0
    trace_failures: list[str] = field(default_factory=list)


@dataclass
class ToleranceRobustnessSummary:
    score: float | None = None
    attempted_cases: int = 0
    passed_cases: int = 0
    stressed_surfaces: list[int] = field(default_factory=list)
    worst_surface: int | None = None
    worst_case: str | None = None
    worst_case_score: float | None = None
    max_spot_rms_um: float | None = None
    min_mtf50_lpmm: float | None = None
    trace_failures: int = 0


@dataclass
class SensorSnrSummary:
    snr_db: float | None = None
    signal_e: float | None = None
    saturation_pct: float | None = None
    image_irradiance_w_m2: float | None = None
    microlens_efficiency: float | None = None


class RayOpticsStore:
    def __init__(self) -> None:
        self.sessions: dict[str, ModelSession] = {}
        self.rayoptics_version = ro.__version__
        self.package_root = Path(ro.__file__).resolve().parent
        self.project_root = Path(__file__).resolve().parents[2]
        self.draft_dir = self.project_root / ".rayoptics-workbench" / "drafts"
        self.runtime_dir = self.project_root / ".rayoptics-workbench" / "runtime"
        self.rayoptics_lock = threading.RLock()
        self._prepare_rayoptics_log_handlers()

    def list_examples(self) -> list[dict[str, str]]:
        patterns = [
            ("RayOptics Model", "models/*.roa"),
            ("Optical Test", "optical/tests/*.roa"),
            ("CODE V", "codev/tests/*.seq"),
            ("Zemax", "zemax/tests/*.zmx"),
            ("Zemax", "zemax/tests/*.ZMX"),
        ]
        examples: list[dict[str, str]] = []
        for kind, pattern in patterns:
            for path in sorted(self.package_root.glob(pattern)):
                examples.append(
                    {
                        "label": f"{kind}: {path.name}",
                        "path": str(path),
                        "kind": kind,
                    }
                )
        return examples

    def patent_db_status(self) -> dict[str, Any]:
        path = self._patent_db_path()
        payload: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            payload["summary"] = None
            return payload
        with self._connect_patent_db() as con:
            schema = self._patent_db_schema(con)
            payload["schema"] = schema
            if schema == "normalized":
                payload["summary"] = self._normalized_patent_db_summary(con)
                return payload
            payload["summary"] = {
                "companies": int(con.execute("SELECT count(*) FROM companies").fetchone()[0]),
                "lenses": int(con.execute("SELECT count(*) FROM lenses").fetchone()[0]),
                "simulationResults": int(con.execute("SELECT count(*) FROM simulation_results").fetchone()[0]),
                "camerae2eReady": int(
                    con.execute("SELECT count(*) FROM simulation_results WHERE simulation_status = 'camerae2e_ready'").fetchone()[0]
                ),
            }
        return payload

    def list_patent_companies(self) -> list[dict[str, Any]]:
        with self._connect_patent_db() as con:
            if self._patent_db_schema(con) == "normalized":
                return self._normalized_patent_companies(con)
            rows = con.execute(
                """
                SELECT
                    c.company,
                    c.company_slug,
                    count(s.simulation_id) AS simulation_results,
                    sum(CASE WHEN s.simulation_status = 'camerae2e_ready' THEN 1 ELSE 0 END) AS camerae2e_ready,
                    sum(CASE WHEN s.simulation_status = 'partial' THEN 1 ELSE 0 END) AS partial,
                    sum(CASE WHEN s.simulation_status = 'metadata_only' THEN 1 ELSE 0 END) AS metadata_only
                FROM companies c
                LEFT JOIN simulation_results s ON s.company_slug = c.company_slug
                GROUP BY c.company, c.company_slug
                ORDER BY c.company
                """
            ).fetchall()
        return [
            {
                "company": row["company"],
                "companySlug": row["company_slug"],
                "simulationResults": int(row["simulation_results"]),
                "camerae2eReady": int(row["camerae2e_ready"]),
                "partial": int(row["partial"]),
                "metadataOnly": int(row["metadata_only"]),
            }
            for row in rows
        ]

    def search_lens_patents(
        self,
        company: str | None = None,
        query: str | None = None,
        status: str = "camerae2e_ready",
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        with self._connect_patent_db() as con:
            if self._patent_db_schema(con) == "normalized":
                return self._normalized_search_lens_patents(con, company=company, query=query, status=status, limit=limit)
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            clauses.append("simulation_status = ?")
            params.append(status)
        if company and company != "all":
            clauses.append("(company_slug = ? OR lower(company) = lower(?))")
            params.extend([_slugify(company), company])
        if query and query.strip():
            like = f"%{query.strip().lower()}%"
            clauses.append(
                "("
                "lower(simulation_id) LIKE ? OR "
                "lower(company) LIKE ? OR "
                "lower(publication_number) LIKE ? OR "
                "lower(example_label) LIKE ? OR "
                "lower(configuration) LIKE ?"
                ")"
            )
            params.extend([like, like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 250)))
        with self._connect_patent_db() as con:
            rows = con.execute(
                f"""
                SELECT
                    simulation_id,
                    lens_id,
                    company,
                    company_slug,
                    publication_number,
                    example_label,
                    configuration,
                    readiness,
                    simulation_status,
                    simulation_model,
                    focal_length_mm,
                    f_number,
                    image_height_mm,
                    half_field_deg,
                    field_of_view_deg,
                    surface_count,
                    asphere_count,
                    notes
                FROM simulation_results
                {where}
                ORDER BY company, publication_number, example_label, configuration
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._patent_row_payload(row) for row in rows]

    def open_lens_patent(self, simulation_id: str) -> ModelResponse:
        with self._connect_patent_db() as con:
            if self._patent_db_schema(con) == "normalized":
                return self._normalized_open_lens_patent(con, simulation_id)
            row = con.execute(
                "SELECT * FROM simulation_results WHERE simulation_id = ?",
                (simulation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown lens patent simulation_id: {simulation_id}")
            surfaces = con.execute(
                """
                SELECT *
                FROM lens_surfaces
                WHERE lens_id = ? AND configuration = ?
                ORDER BY surface_order
                """,
                (row["lens_id"], row["configuration"]),
            ).fetchall()
        if not surfaces:
            raise ValueError(f"No lens surfaces found for patent simulation {simulation_id}")
        warnings: list[str] = [
            "Loaded from Lens Patent DB; verify patent prescriptions before treating as production optics.",
            "Patent glasses use disclosed nd/vd values as synthetic media; catalog glass substitution is not implied.",
        ]
        if row["simulation_status"] != "camerae2e_ready":
            warnings.append(f"Simulation status is {row['simulation_status']}; sequential trace may be incomplete.")
        opm = self._build_lens_patent_model(row, surfaces, warnings)
        try:
            opm.update_model()
        except Exception as exc:
            warnings.append(f"Patent model loaded, but update_model failed: {type(exc).__name__}: {exc}")
        self._append_patent_first_order_validation(row, opm, warnings)
        response = self._register(opm, None, warnings)
        session = self.sessions[response.model.id]
        session.workbench_metadata = {
            "source": "lensPatentDb",
            "lensPatent": self._patent_row_payload(row),
        }
        return self.response(session.model_id)

    def _append_patent_first_order_validation(
        self,
        row: dict[str, Any],
        opm: Any,
        warnings: list[str],
    ) -> None:
        try:
            target_efl = _safe_float(row["focal_length_mm"])
        except (KeyError, IndexError, TypeError):
            target_efl = _safe_float(row.get("focal_length_mm") if isinstance(row, dict) else None)
        if target_efl is None or abs(target_efl) <= 1.0e-9:
            return
        try:
            first_order = self._first_order_values(opm)
        except Exception:
            return
        computed_efl = _safe_float(first_order.get("efl"))
        if computed_efl is None:
            return
        relative_error = abs(abs(computed_efl) - abs(target_efl)) / abs(target_efl) * 100.0
        if relative_error > 3.0:
            warnings.append(
                f"Computed EFL {computed_efl:.4g} mm differs from patent focal length {target_efl:.4g} mm by {relative_error:.1f}%; treat this converted prescription as unvalidated."
            )

    def _patent_db_path(self) -> Path:
        return Path(os.environ.get("LENS_PATENT_DB_PATH", str(DEFAULT_PATENT_DB_PATH))).expanduser()

    def _connect_patent_db(self) -> sqlite3.Connection:
        path = self._patent_db_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Lens Patent DB not found: {path}. Generate CameraE2E lens patent data first."
            )
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        return con

    def _patent_db_schema(self, con: sqlite3.Connection) -> str:
        tables = {
            str(row["name"])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if {"simulation_results", "lens_surfaces", "companies"}.issubset(tables):
            return "camerae2e"
        if {"simulation_prescriptions", "simulation_surfaces", "simulation_aspheres"}.issubset(tables):
            return "normalized"
        raise ValueError(f"Unsupported Lens Patent DB schema. Found tables: {', '.join(sorted(tables))}")

    def _normalized_patent_db_summary(self, con: sqlite3.Connection) -> dict[str, Any]:
        row = con.execute(
            """
            SELECT
                count(DISTINCT company) AS companies,
                count(*) AS prescriptions
            FROM simulation_prescriptions
            """
        ).fetchone()
        surface_count = int(con.execute("SELECT count(*) FROM simulation_surfaces").fetchone()[0])
        asphere_count = int(con.execute("SELECT count(*) FROM simulation_aspheres").fetchone()[0])
        readiness_rows = self._normalized_readiness_index().values()
        if readiness_rows:
            camerae2e_ready = sum(
                1 for item in readiness_rows if item.get("readiness") in {"ready_staging", "ready_configured"}
            )
        else:
            camerae2e_ready = int(row["prescriptions"])
        return {
            "companies": int(row["companies"]),
            "lenses": int(row["prescriptions"]),
            "simulationResults": int(row["prescriptions"]),
            "camerae2eReady": int(camerae2e_ready),
            "surfaces": surface_count,
            "aspheres": asphere_count,
        }

    def _normalized_patent_companies(self, con: sqlite3.Connection) -> list[dict[str, Any]]:
        summary = self._normalized_company_summary()
        if summary:
            return [
                {
                    "company": row["company"],
                    "companySlug": _slugify(row["company"]),
                    "simulationResults": int(_safe_float(row.get("prescriptions", 0)) or 0),
                    "camerae2eReady": int(_safe_float(row.get("ready_staging", 0)) or 0)
                    + int(_safe_float(row.get("ready_configured", 0)) or 0),
                    "partial": int(_safe_float(row.get("needs_variable_distances", 0)) or 0),
                    "metadataOnly": int(_safe_float(row.get("blocked", 0)) or 0),
                }
                for row in summary
            ]
        rows = con.execute(
            """
            SELECT company, count(*) AS prescriptions
            FROM simulation_prescriptions
            GROUP BY company
            ORDER BY company
            """
        ).fetchall()
        return [
            {
                "company": row["company"],
                "companySlug": _slugify(row["company"]),
                "simulationResults": int(row["prescriptions"]),
                "camerae2eReady": int(row["prescriptions"]),
                "partial": 0,
                "metadataOnly": 0,
            }
            for row in rows
        ]

    def _normalized_search_lens_patents(
        self,
        con: sqlite3.Connection,
        company: str | None,
        query: str | None,
        status: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = con.execute(
            """
            SELECT *
            FROM simulation_prescriptions
            ORDER BY company, publication_number, source_table_index, id
            """
        ).fetchall()
        readiness_index = self._normalized_readiness_index()
        status_readiness = {
            "camerae2e_ready": {"ready_staging", "ready_configured"},
            "partial": {"needs_variable_distances"},
            "metadata_only": {"blocked"},
        }
        allowed_readiness = status_readiness.get(status, {status} if status and status not in {"all", "partial"} else None)
        lowered_query = query.strip().lower() if query and query.strip() else ""
        company_slug = _slugify(company) if company and company != "all" else ""
        results: list[dict[str, Any]] = []
        for row in rows:
            readiness = readiness_index.get(int(row["id"]), {})
            readiness_value = readiness.get("readiness", "ready_staging")
            if company_slug and company_slug not in {_slugify(row["company"]), str(row["company"]).lower()}:
                continue
            for configuration in self._normalized_configurations(readiness):
                config_status = self._normalized_configuration_status(con, row, readiness, configuration)
                if status == "camerae2e_ready" and config_status != "camerae2e_ready":
                    continue
                if status == "partial" and config_status != "partial":
                    continue
                if status not in {"all", "camerae2e_ready", "partial"} and allowed_readiness is not None and readiness_value not in allowed_readiness:
                    continue
                if lowered_query:
                    haystack = " ".join(
                        [
                            self._normalized_simulation_id(row, configuration),
                            str(row["company"]),
                            str(row["publication_number"]),
                            str(row["example_label"]),
                            str(row["title"]),
                            configuration,
                        ]
                    ).lower()
                    if lowered_query not in haystack:
                        continue
                results.append(self._normalized_patent_payload(con, row, readiness, configuration))
                if len(results) >= max(1, min(int(limit), 250)):
                    break
            if len(results) >= max(1, min(int(limit), 250)):
                break
        return results

    def _normalized_open_lens_patent(self, con: sqlite3.Connection, simulation_id: str) -> ModelResponse:
        match = re.search(r"p0*(\d+)(?::([A-Za-z0-9_.-]+))?", str(simulation_id), flags=re.IGNORECASE)
        if match is None:
            raise KeyError(f"Unknown lens patent simulation_id: {simulation_id}")
        prescription_id = int(match.group(1))
        requested_configuration = match.group(2) or "base"
        row = con.execute("SELECT * FROM simulation_prescriptions WHERE id = ?", (prescription_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown lens patent simulation_id: {simulation_id}")
        readiness = self._normalized_readiness_index().get(prescription_id, {})
        configurations = self._normalized_configurations(readiness)
        configuration = requested_configuration
        warnings: list[str] = [
            "Loaded from Lens Patent DB normalized schema; verify patent prescriptions before treating as production optics.",
            "Patent glasses use disclosed nd/vd values as synthetic media; catalog glass substitution is not implied.",
        ]
        if configuration == "base" and configurations != ["base"]:
            configuration = configurations[0]
            warnings.append(f"Opened default configured prescription '{configuration}' for {simulation_id}.")
        if configuration not in configurations and configuration != "base":
            raise KeyError(f"Unknown configuration '{configuration}' for lens patent simulation p{prescription_id:04d}.")

        surfaces = self._normalized_configured_surfaces(row, readiness, configuration)
        if surfaces is None:
            surface_rows = con.execute(
                """
                SELECT *
                FROM simulation_surfaces
                WHERE prescription_id = ?
                ORDER BY surface_order
                """,
                (prescription_id,),
            ).fetchall()
            asphere_rows = con.execute(
                """
                SELECT *
                FROM simulation_aspheres
                WHERE prescription_id = ?
                ORDER BY surface_label
                """,
                (prescription_id,),
            ).fetchall()
            if not surface_rows:
                raise ValueError(f"No lens surfaces found for patent simulation {simulation_id}")
            surfaces = self._normalized_surface_rows(surface_rows, asphere_rows)
        self._apply_normalized_variable_distances(con, prescription_id, configuration, surfaces)
        if not surfaces:
            raise ValueError(f"No lens surfaces found for patent simulation {simulation_id}")
        readiness_value = readiness.get("readiness")
        if readiness_value and readiness_value not in {"ready_staging", "ready_configured"}:
            warnings.append(f"Prescription readiness is {readiness_value}; sequential trace may be incomplete.")
        if any(self._patent_surface_kind(surface) == "object" for surface in surfaces):
            warnings.append("Patent object rows were skipped during RayOptics model construction.")
        row_payload = self._normalized_model_row(con, row, readiness, configuration)
        opm = self._build_lens_patent_model(row_payload, surfaces, warnings)
        try:
            opm.update_model()
        except Exception as exc:
            warnings.append(f"Patent model loaded, but update_model failed: {type(exc).__name__}: {exc}")
        self._append_patent_first_order_validation(row_payload, opm, warnings)
        response = self._register(opm, None, warnings)
        session = self.sessions[response.model.id]
        session.workbench_metadata = {
            "source": "lensPatentDb",
            "lensPatent": self._normalized_patent_payload(con, row, readiness, configuration),
        }
        return self.response(session.model_id)

    def _normalized_patent_payload(
        self,
        con: sqlite3.Connection,
        row: sqlite3.Row,
        readiness: dict[str, Any] | None = None,
        configuration: str = "base",
    ) -> dict[str, Any]:
        model_row = self._normalized_model_row(con, row, readiness or {}, configuration)
        return self._patent_row_payload(model_row)

    def _normalized_model_row(
        self,
        con: sqlite3.Connection,
        row: sqlite3.Row,
        readiness: dict[str, Any],
        configuration: str,
    ) -> dict[str, Any]:
        metrics = self._normalized_configuration_metrics(con, int(row["id"]), configuration)
        caption_metrics = self._parse_patent_caption_metrics(row["caption"])
        metrics = {**caption_metrics, **{key: value for key, value in metrics.items() if value is not None}}
        readiness_value = readiness.get("readiness", "ready_staging")
        simulation_status = self._normalized_configuration_status(con, row, readiness, configuration)
        notes = [readiness.get("notes", "")]
        if readiness.get("variable_configurations"):
            notes.append(f"variable configurations: {readiness['variable_configurations']}")
        return {
            "simulation_id": self._normalized_simulation_id(row, configuration),
            "lens_id": f"p{int(row['id']):04d}",
            "company": row["company"],
            "company_slug": _slugify(row["company"]),
            "publication_number": row["publication_number"],
            "example_label": row["example_label"],
            "configuration": configuration,
            "readiness": readiness_value,
            "simulation_status": simulation_status,
            "simulation_model": "normalized_patent",
            "focal_length_mm": metrics.get("focal_length_mm"),
            "f_number": metrics.get("f_number"),
            "image_height_mm": metrics.get("image_height_mm"),
            "half_field_deg": metrics.get("half_field_deg"),
            "field_of_view_deg": metrics.get("field_of_view_deg"),
            "surface_count": int(row["surface_row_count"]),
            "asphere_count": int(row["asphere_row_count"]),
            "notes": "; ".join(note for note in notes if note),
        }

    def _normalized_surface_rows(
        self,
        surfaces: list[sqlite3.Row],
        aspheres: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        asphere_by_label = {str(row["surface_label"]).strip(): row for row in aspheres}
        adapted: list[dict[str, Any]] = []
        for surface in surfaces:
            label = str(surface["surface_label"] or surface["surface_order"]).strip()
            descriptors = self._json_list(surface["descriptors_json"])
            kind = self._normalized_surface_kind(label, descriptors)
            asphere = asphere_by_label.get(label)
            coefficients = {}
            if asphere is not None:
                try:
                    coefficients = json.loads(str(asphere["coefficients_json"] or "{}"))
                except json.JSONDecodeError:
                    coefficients = {}
            conic = _safe_float(surface["conic"])
            thickness = _safe_float(surface["thickness"])
            if thickness is None:
                thickness = self._patent_embedded_surface_thickness(surface["raw"])
            if conic is None:
                conic = _safe_float(coefficients.get("K") if isinstance(coefficients, dict) else None)
            raw_payload = {
                "surface_kind": kind,
                "descriptors": descriptors,
                "source_schema": "normalized",
                "raw": surface["raw"],
            }
            adapted.append(
                {
                    "surface_order": int(surface["surface_order"]),
                    "surface_label": label,
                    "surface_key": label,
                    "radius_mm": _safe_float(surface["radius"]),
                    "thickness_mm": thickness,
                    "nd": _safe_float(surface["nd"]),
                    "vd": _safe_float(surface["vd"]),
                    "material": str(surface["material"] or ""),
                    "effective_aperture_mm": None,
                    "conic": conic,
                    "is_aspheric": int(bool(surface["is_aspheric"]) or bool(asphere) or self._patent_raw_marks_aspheric(surface["raw"])),
                    "coefficients_json": json.dumps(coefficients),
                    "raw_json": json.dumps(raw_payload),
                }
            )
        return adapted

    def _normalized_configured_surfaces(
        self,
        row: sqlite3.Row,
        readiness: dict[str, Any],
        configuration: str,
    ) -> list[dict[str, Any]] | None:
        if configuration == "base":
            return None
        export_dir = self._normalized_prescription_export_dir(row, readiness)
        if export_dir is None:
            return None
        configured_path = self._normalized_configured_surface_path(export_dir, configuration)
        if configured_path is None:
            return None
        with configured_path.open(newline="") as handle:
            return self._normalized_surface_csv_rows(list(csv.DictReader(handle)))

    def _normalized_configured_surface_path(self, export_dir: Path, configuration: str) -> Path | None:
        configured_dir = export_dir / "configured_surfaces"
        names = [
            f"{configuration}.csv",
            f"{configuration.replace('_', '-')}.csv",
            f"{_slugify(configuration)}.csv",
        ]
        for name in dict.fromkeys(names):
            path = configured_dir / name
            if path.exists():
                return path
        return None

    def _normalized_surface_csv_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        adapted: list[dict[str, Any]] = []
        for surface in rows:
            label = str(surface.get("surface_label") or surface.get("surface_order") or "").strip()
            descriptors = self._json_list(surface.get("descriptors_json"))
            kind = str(surface.get("surface_kind") or "").strip().lower() or self._normalized_surface_kind(label, descriptors)
            coefficients = self._normalized_csv_asphere_coefficients(surface)
            if "k" in coefficients and "K" not in coefficients:
                coefficients["K"] = coefficients["k"]
            conic = _safe_float(surface.get("conic"))
            if conic is None:
                conic = _safe_float(coefficients.get("k")) or _safe_float(coefficients.get("K"))
            raw_surface = str(surface.get("raw_surface") or "")
            thickness = _safe_float(surface.get("thickness"))
            if thickness is None:
                thickness = self._patent_embedded_surface_thickness(raw_surface)
            raw_payload = {
                "surface_kind": kind,
                "descriptors": descriptors,
                "source_schema": "normalized_configured_csv",
                "configuration": surface.get("configuration", ""),
                "raw": raw_surface,
            }
            adapted.append(
                {
                    "surface_order": int(_safe_float(surface.get("surface_order")) or len(adapted) + 1),
                    "surface_label": label,
                    "surface_key": str(surface.get("surface_key") or label),
                    "radius_mm": _safe_float(surface.get("radius")),
                    "thickness_mm": thickness,
                    "nd": _safe_float(surface.get("nd")),
                    "vd": _safe_float(surface.get("vd")),
                    "material": str(surface.get("material") or ""),
                    "effective_aperture_mm": _safe_float(surface.get("effective_aperture")),
                    "conic": conic,
                    "is_aspheric": int(self._csv_truthy(surface.get("is_aspheric")) or bool(coefficients) or self._patent_raw_marks_aspheric(raw_surface)),
                    "coefficients_json": json.dumps(coefficients),
                    "raw_json": json.dumps(raw_payload),
                }
            )
        return adapted

    def _patent_embedded_surface_thickness(self, raw: Any) -> float | None:
        match = re.search(
            r"\(ASP\)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
            str(raw or ""),
            flags=re.IGNORECASE,
        )
        return _safe_float(match.group(1)) if match else None

    def _patent_raw_marks_aspheric(self, raw: Any) -> bool:
        return "(asp)" in str(raw or "").lower()

    def _normalized_csv_asphere_coefficients(self, surface: dict[str, Any]) -> dict[str, Any]:
        coefficients = {
            key: surface.get(key, "")
            for key in ("k", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a18", "a20")
            if str(surface.get(key, "")).strip()
        }
        raw_asphere = str(surface.get("raw_asphere") or "")
        for match in re.finditer(
            r"\b(K|A\d+)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
            raw_asphere,
            flags=re.IGNORECASE,
        ):
            key = match.group(1).lower()
            value = match.group(2)
            coefficients[key] = value
            if key == "k":
                coefficients["K"] = value
        return coefficients

    def _normalized_surface_kind(self, label: str, descriptors: list[str]) -> str:
        tokens = " ".join([label, *descriptors]).strip().lower()
        if "object" in tokens:
            return "object"
        if "image" in tokens:
            return "image"
        if "stop" in tokens or "ape." in tokens or "aperture" in tokens:
            return "stop"
        return ""

    def _normalized_simulation_id(self, row: sqlite3.Row, configuration: str = "base") -> str:
        return f"p{int(row['id']):04d}:{configuration}"

    def _normalized_configurations(self, readiness: dict[str, Any]) -> list[str]:
        values = [
            item.strip()
            for item in str(readiness.get("variable_configurations", "")).split(",")
            if item.strip()
        ]
        return values or ["base"]

    def _normalized_configuration_status(
        self,
        con: sqlite3.Connection,
        row: sqlite3.Row,
        readiness: dict[str, Any],
        configuration: str,
    ) -> str:
        readiness_value = readiness.get("readiness", "ready_staging")
        if readiness_value == "blocked":
            return "metadata_only"
        if readiness_value == "needs_variable_distances":
            return "partial"
        if configuration in self._normalized_unresolved_configurations(readiness):
            return "partial"
        if readiness_value == "ready_configured" and configuration != "base":
            export_dir = self._normalized_prescription_export_dir(row, readiness)
            if export_dir is None or self._normalized_configured_surface_path(export_dir, configuration) is None:
                return "partial"
        if self._normalized_missing_front_surfaces(con, row, readiness, configuration):
            return "partial"
        if readiness_value in {"ready_staging", "ready_configured"}:
            return "camerae2e_ready"
        return "partial"

    def _normalized_missing_front_surfaces(
        self,
        con: sqlite3.Connection,
        row: sqlite3.Row,
        readiness: dict[str, Any],
        configuration: str,
    ) -> bool:
        surfaces = self._normalized_configured_surfaces(row, readiness, configuration)
        if surfaces is None:
            surface_rows = con.execute(
                """
                SELECT surface_label, descriptors_json
                FROM simulation_surfaces
                WHERE prescription_id = ?
                ORDER BY surface_order
                """,
                (int(row["id"]),),
            ).fetchall()
            surfaces = [
                {
                    "surface_label": str(surface["surface_label"] or ""),
                    "raw_json": json.dumps(
                        {
                            "surface_kind": self._normalized_surface_kind(
                                str(surface["surface_label"] or ""),
                                self._json_list(surface["descriptors_json"]),
                            )
                        }
                    ),
                }
                for surface in surface_rows
            ]
        for surface in surfaces:
            if self._patent_surface_kind(surface) == "object":
                continue
            label = str(surface.get("surface_label") or "").strip()
            match = re.match(r"(\d+)", label)
            if match is None:
                continue
            return int(match.group(1)) > 1
        return False

    def _normalized_unresolved_configurations(self, readiness: dict[str, Any]) -> set[str]:
        unresolved = str(readiness.get("unresolved_variable_distances", "") or "").strip()
        if not unresolved:
            return set()
        configs: set[str] = set()
        for part in unresolved.split(";"):
            if ":" not in part:
                continue
            name = part.split(":", 1)[0].strip()
            if name:
                configs.add(name)
        return configs

    def _apply_normalized_variable_distances(
        self,
        con: sqlite3.Connection,
        prescription_id: int,
        configuration: str,
        surfaces: list[dict[str, Any]],
    ) -> None:
        distances = self._normalized_variable_distance_values(con, prescription_id, configuration)
        if not distances:
            return
        for surface in surfaces:
            if surface.get("thickness_mm") is not None:
                continue
            for key in self._normalized_surface_distance_keys(surface):
                if key in distances:
                    surface["thickness_mm"] = distances[key]
                    break

        back_focus = distances.get("bf")
        if back_focus is None:
            return
        for surface in reversed(surfaces):
            surface_kind = self._patent_surface_kind(surface)
            if surface_kind == "object":
                continue
            distance_keys = set(self._normalized_surface_distance_keys(surface))
            raw = self._patent_surface_raw_text(surface).lower()
            if not distance_keys.intersection(distances) and (
                surface.get("thickness_mm") is None or surface_kind == "image" or "image" in raw
            ):
                surface["thickness_mm"] = back_focus
            return

    def _normalized_surface_distance_keys(self, surface: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        label = str(surface.get("surface_label") or "").strip()
        match = re.match(r"(\d+)", label)
        if match is not None:
            keys.append(f"d{match.group(1)}")
        surface_order = int(_safe_float(surface.get("surface_order")) or 0)
        if surface_order > 0:
            keys.append(f"d{surface_order}")
        return list(dict.fromkeys(keys))

    def _normalized_variable_distance_values(
        self,
        con: sqlite3.Connection,
        prescription_id: int,
        configuration: str,
    ) -> dict[str, float]:
        if configuration == "base":
            return {}
        rows = con.execute(
            """
            SELECT parameter, configuration, value
            FROM simulation_variable_distances
            WHERE prescription_id = ?
            """,
            (prescription_id,),
        ).fetchall()
        values: dict[str, float] = {}
        for candidate in self._normalized_metric_configuration_candidates(configuration):
            for row in rows:
                if str(row["configuration"]) != candidate:
                    continue
                key = str(row["parameter"]).strip().lower()
                value = _safe_float(row["value"])
                if value is None:
                    continue
                if re.fullmatch(r"d\d+", key) or key in {"bf", "bfl", "back_focus", "back_focal_length"}:
                    values.setdefault("bf" if key in {"bfl", "back_focus", "back_focal_length"} else key, value)
        return values

    def _normalized_configuration_metrics(
        self,
        con: sqlite3.Connection,
        prescription_id: int,
        configuration: str,
    ) -> dict[str, float | None]:
        if configuration == "base":
            return {}
        candidates = self._normalized_metric_configuration_candidates(configuration)
        rows = con.execute(
            """
            SELECT parameter, configuration, value
            FROM simulation_variable_distances
            WHERE prescription_id = ?
            """,
            (prescription_id,),
        ).fetchall()
        values: dict[str, float | None] = {}
        for candidate in candidates:
            for row in rows:
                if str(row["configuration"]) != candidate:
                    continue
                key = str(row["parameter"]).strip().lower()
                if key not in values:
                    values[key] = _safe_float(row["value"])
        focal_length = values.get("focal_length")
        f_number = values.get("f_number")
        image_height = values.get("image_height")
        half_field = values.get("half_angle_of_view") or values.get("half_field_deg") or values.get("hfov")
        if image_height is None and focal_length is not None and half_field is not None:
            image_height = focal_length * math.tan(math.radians(half_field))
        return {
            "focal_length_mm": focal_length,
            "f_number": f_number,
            "image_height_mm": image_height,
            "half_field_deg": half_field,
            "field_of_view_deg": half_field * 2.0 if half_field is not None else None,
        }

    def _normalized_metric_configuration_candidates(self, configuration: str) -> list[str]:
        candidates = [configuration]
        for prefix in ("infinity_", "short_", "close_", "closest_"):
            if configuration.startswith(prefix):
                suffix = configuration[len(prefix) :]
                if suffix:
                    candidates.append(suffix)
        return candidates

    def _parse_patent_caption_metrics(self, caption: str | None) -> dict[str, float | None]:
        text = str(caption or "")
        focal_length = self._caption_float(text, r"(?<![A-Za-z])f\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
        f_number = self._caption_float(text, r"\bF\s*no\.?\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
        half_field = self._caption_float(text, r"\bHFOV\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
        image_height = None
        if focal_length is not None and half_field is not None:
            image_height = focal_length * math.tan(math.radians(half_field))
        return {
            "focal_length_mm": focal_length,
            "f_number": f_number,
            "half_field_deg": half_field,
            "field_of_view_deg": half_field * 2.0 if half_field is not None else None,
            "image_height_mm": image_height,
        }

    def _caption_float(self, text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return _safe_float(match.group(1)) if match else None

    def _json_list(self, value: Any) -> list[str]:
        try:
            parsed = json.loads(str(value or "[]"))
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed]

    def _csv_truthy(self, value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _normalized_readiness_index(self) -> dict[int, dict[str, Any]]:
        path = self._normalized_readiness_path()
        if path is None or not path.exists():
            return {}
        with path.open(newline="") as handle:
            return {
                int(row["prescription_id"]): row
                for row in csv.DictReader(handle)
                if str(row.get("prescription_id", "")).isdigit()
            }

    def _normalized_manifest_index(self) -> dict[int, dict[str, Any]]:
        path = self._normalized_manifest_path()
        if path is None or not path.exists():
            return {}
        with path.open(newline="") as handle:
            return {
                int(row["prescription_id"]): row
                for row in csv.DictReader(handle)
                if str(row.get("prescription_id", "")).isdigit()
            }

    def _normalized_prescription_export_dir(
        self,
        row: sqlite3.Row,
        readiness: dict[str, Any],
    ) -> Path | None:
        root, suffix = self._normalized_db_root_and_suffix()
        if root is None:
            return None
        manifest_row = self._normalized_manifest_index().get(int(row["id"]))
        relative_dir = str(manifest_row.get("relative_dir", "") if manifest_row else "").strip()
        if relative_dir:
            export_name = "prescriptions" if not suffix else f"prescriptions_{suffix}"
            return root / "exports" / export_name / relative_dir
        return None

    def _normalized_company_summary(self) -> list[dict[str, Any]]:
        path = self._normalized_company_summary_path()
        if path is None or not path.exists():
            return []
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def _normalized_readiness_path(self) -> Path | None:
        root, suffix = self._normalized_db_root_and_suffix()
        if root is None:
            return None
        export_name = "prescriptions" if not suffix else f"prescriptions_{suffix}"
        return root / "exports" / export_name / "readiness.csv"

    def _normalized_manifest_path(self) -> Path | None:
        root, suffix = self._normalized_db_root_and_suffix()
        if root is None:
            return None
        export_name = "prescriptions" if not suffix else f"prescriptions_{suffix}"
        return root / "exports" / export_name / "manifest.csv"

    def _normalized_company_summary_path(self) -> Path | None:
        root, suffix = self._normalized_db_root_and_suffix()
        if root is None or not suffix:
            return None
        return root / "data" / suffix / "company_summary.csv"

    def _normalized_db_root_and_suffix(self) -> tuple[Path | None, str]:
        path = self._patent_db_path()
        if path.parent.name != "data":
            return None, ""
        stem = path.stem
        if stem == "lens_simulation":
            return path.parent.parent, ""
        prefix = "lens_simulation_"
        if stem.startswith(prefix):
            return path.parent.parent, stem[len(prefix) :]
        return path.parent.parent, ""

    def _patent_row_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "simulationId": row["simulation_id"],
            "lensId": row["lens_id"],
            "company": row["company"],
            "companySlug": row["company_slug"],
            "publicationNumber": row["publication_number"],
            "exampleLabel": row["example_label"],
            "configuration": row["configuration"],
            "readiness": row["readiness"],
            "simulationStatus": row["simulation_status"],
            "simulationModel": row["simulation_model"],
            "focalLengthMm": _safe_float(row["focal_length_mm"]),
            "fNumber": _safe_float(row["f_number"]),
            "imageHeightMm": _safe_float(row["image_height_mm"]),
            "halfFieldDeg": _safe_float(row["half_field_deg"]),
            "fieldOfViewDeg": _safe_float(row["field_of_view_deg"]),
            "surfaceCount": int(row["surface_count"]),
            "asphereCount": int(row["asphere_count"]),
            "notes": [note.strip() for note in str(row["notes"] or "").split(";") if note.strip()],
        }

    def _build_lens_patent_model(
        self,
        row: sqlite3.Row,
        surfaces: list[sqlite3.Row],
        warnings: list[str],
    ) -> Any:
        opm = OpticalModel()
        opm.radius_mode = True
        opm["system_spec"].title = self._patent_model_title(row)
        sm = opm.seq_model
        osp = opm.optical_spec
        sm.do_apertures = False
        sm.gaps[0].thi = 1.0e10

        f_number = _safe_float(row["f_number"]) or 4.0
        focal_length_mm = _safe_float(row["focal_length_mm"]) or 50.0
        image_height_mm = _safe_float(row["image_height_mm"])
        half_field_deg = _safe_float(row["half_field_deg"])
        if image_height_mm is None and half_field_deg is not None:
            image_height_mm = focal_length_mm * math.tan(math.radians(half_field_deg))
        if image_height_mm is None or image_height_mm <= 0.0:
            image_height_mm = max(focal_length_mm * 0.1, 1.0)
            warnings.append("Patent field height was missing; used 10% of focal length as a field-height proxy.")

        osp.pupil = PupilSpec(osp, key=("image", "f/#"), value=float(f_number))
        osp.field_of_view = FieldSpec(
            osp,
            key=("image", "height"),
            value=float(image_height_mm),
            flds=[0.0, 0.5, 1.0],
            is_relative=True,
        )
        osp.spectral_region = WvlSpec([(450.0, 0.5), (550.0, 1.0), (650.0, 0.5)], ref_wl=1)

        fallback_semi_diameter = self._patent_fallback_semi_diameter(row)
        used_fallback_aperture = False
        stop_surface = None
        min_aperture = math.inf
        skipped_context_rows = 0
        for surface in surfaces:
            surface_kind = self._patent_surface_kind(surface)
            if surface_kind == "object" or (
                surface_kind == "image" and not self._patent_surface_has_interface_data(surface)
            ):
                skipped_context_rows += 1
                continue
            if surface_kind == "stop":
                stop_data = [0.0, _safe_float(surface["thickness_mm"]) or 0.0]
                semi_diameter = _safe_float(surface["effective_aperture_mm"])
                semi_diameter = None if semi_diameter is None or semi_diameter <= 0 else semi_diameter / 2.0
                if semi_diameter is None:
                    semi_diameter = fallback_semi_diameter
                    used_fallback_aperture = True
                sm.add_surface(stop_data, sd=semi_diameter)
                stop_surface = int(sm.cur_surface)
                sm.ifcs[stop_surface].label = str(surface["surface_label"] or "Stop")
                continue
            surface_data = self._patent_surface_data(surface, warnings)
            semi_diameter = _safe_float(surface["effective_aperture_mm"])
            semi_diameter = None if semi_diameter is None or semi_diameter <= 0 else semi_diameter / 2.0
            if semi_diameter is None:
                semi_diameter = fallback_semi_diameter
                used_fallback_aperture = True
            sm.add_surface(surface_data, sd=semi_diameter)
            surface_index = int(sm.cur_surface)
            ifc = sm.ifcs[surface_index]
            ifc.label = str(surface["surface_label"] or surface["surface_order"])
            self._apply_patent_surface_profile(ifc, surface)
            if semi_diameter is not None and semi_diameter < min_aperture:
                min_aperture = semi_diameter
                if stop_surface is None:
                    stop_surface = surface_index
        if skipped_context_rows:
            warnings.append(f"Skipped {skipped_context_rows} patent context rows while constructing the sequential model.")
        if used_fallback_aperture:
            warnings.append(
                f"No reliable clear-aperture data was available for every patent surface; used {fallback_semi_diameter:.3g} mm semi-diameter proxy."
            )

        if stop_surface is not None:
            sm.cur_surface = int(stop_surface)
            sm.set_stop()
        elif sm.get_num_surfaces() > 2:
            fallback_stop = max(1, sm.get_num_surfaces() // 2)
            sm.cur_surface = fallback_stop
            sm.set_stop()
            warnings.append(f"No explicit stop found; used S{fallback_stop} as aperture-stop proxy.")
        return opm

    def _patent_fallback_semi_diameter(self, row: sqlite3.Row | dict[str, Any]) -> float:
        f_number = _safe_float(row["f_number"])
        focal_length_mm = abs(_safe_float(row["focal_length_mm"]) or 0.0)
        image_height_mm = abs(_safe_float(row["image_height_mm"]) or 0.0)
        pupil_radius = focal_length_mm / (2.0 * f_number) if f_number and f_number > 0.0 else 0.0
        paraxial_bundle = math.hypot(image_height_mm, pupil_radius)
        return _clamp(max(3.0, image_height_mm * 1.75, pupil_radius * 2.0, paraxial_bundle * 1.45), 1.5, 12.0)

    def _patent_surface_data(self, surface: sqlite3.Row, warnings: list[str]) -> list[Any]:
        radius = _safe_float(surface["radius_mm"])
        thickness = _safe_float(surface["thickness_mm"])
        if thickness is None:
            thickness = self._patent_embedded_surface_thickness(self._patent_surface_raw_text(surface))
        data: list[Any] = [0.0 if radius is None else radius, 0.0 if thickness is None else thickness]
        nd = _safe_float(surface["nd"])
        vd = _safe_float(surface["vd"])
        material = str(surface["material"] or "").strip()
        if nd is not None and nd > 1.0:
            data.extend([nd, 50.0 if vd is None or vd <= 0.0 else vd])
            if vd is None:
                warnings.append(f"S{surface['surface_order']} has nd without vd; used vd=50 proxy.")
        elif material and material.lower() != "air":
            data.extend([material, "Schott"])
        return data

    def _patent_surface_has_interface_data(self, surface: sqlite3.Row | dict[str, Any]) -> bool:
        if _safe_float(surface["radius_mm"]) is not None:
            return True
        if _safe_float(surface["nd"]) is not None:
            return True
        material = str(surface["material"] or "").strip().lower()
        if material and material != "air":
            return True
        if bool(surface["is_aspheric"]):
            return True
        try:
            coefficients = json.loads(str(surface["coefficients_json"] or "{}"))
        except json.JSONDecodeError:
            coefficients = {}
        return bool(coefficients)

    def _apply_patent_surface_profile(self, ifc: Any, surface: sqlite3.Row) -> None:
        conic = _safe_float(surface["conic"])
        try:
            raw_coefficients = json.loads(str(surface["coefficients_json"] or "{}"))
        except json.JSONDecodeError:
            raw_coefficients = {}
        coefficients = (
            {str(key).lower(): value for key, value in raw_coefficients.items()}
            if isinstance(raw_coefficients, dict)
            else {}
        )
        if conic is None:
            conic = _safe_float(coefficients.get("k"))
        is_aspheric = bool(surface["is_aspheric"])
        if is_aspheric:
            polynomial_orders = self._coefficient_orders(coefficients)
            use_radial_polynomial = any(order % 2 == 1 for order in polynomial_orders)
            ifc.profile = profiles.mutate_profile(
                ifc.profile,
                "RadialPolynomial" if use_radial_polynomial else "EvenPolynomial",
            )
            if conic is not None:
                ifc.profile.cc = conic
            if use_radial_polynomial:
                max_order = max(polynomial_orders, default=0)
                if len(ifc.profile.coefs) < max_order:
                    ifc.profile.coefs = list(ifc.profile.coefs) + [0.0] * (max_order - len(ifc.profile.coefs))
                for order in polynomial_orders:
                    value = _safe_float(coefficients.get(f"a{order}"))
                    if value is not None:
                        ifc.profile.coefs[order - 1] = value
                ifc.profile.update()
                return
            coefficient_map = {
                "a4": 1,
                "a6": 2,
                "a8": 3,
                "a10": 4,
                "a12": 5,
                "a14": 6,
                "a16": 7,
                "a18": 8,
                "a20": 9,
            }
            if len(ifc.profile.coefs) < 10:
                ifc.profile.coefs = list(ifc.profile.coefs) + [0.0] * (10 - len(ifc.profile.coefs))
            for key, index in coefficient_map.items():
                value = _safe_float(coefficients.get(key))
                if value is not None:
                    ifc.profile.coefs[index] = value
            return
        if conic is not None:
            ifc.profile = profiles.mutate_profile(ifc.profile, "Conic")
            ifc.profile.cc = conic

    def _coefficient_orders(self, coefficients: dict[str, Any]) -> list[int]:
        orders: list[int] = []
        for key, value in coefficients.items():
            if not re.fullmatch(r"a\d+", str(key)):
                continue
            if _safe_float(value) is None:
                continue
            orders.append(int(str(key)[1:]))
        return sorted(set(orders))

    def _patent_surface_raw_text(self, surface: sqlite3.Row | dict[str, Any]) -> str:
        raw_json = ""
        try:
            raw_json = str(surface["raw_json"] or "")
        except (KeyError, IndexError, TypeError):
            raw_json = ""
        if raw_json:
            try:
                raw_payload = json.loads(raw_json)
            except json.JSONDecodeError:
                raw_payload = {}
            if isinstance(raw_payload, dict):
                return str(raw_payload.get("raw") or "")
        try:
            return str(surface["raw"] or "")
        except (KeyError, IndexError, TypeError):
            return ""

    def _patent_surface_kind(self, surface: sqlite3.Row) -> str:
        try:
            raw = json.loads(str(surface["raw_json"] or "{}"))
        except json.JSONDecodeError:
            return ""
        return str(raw.get("surface_kind", "")).strip().lower()

    def _patent_model_title(self, row: sqlite3.Row) -> str:
        label = str(row["example_label"] or "").strip()
        parts = [
            str(row["company"]),
            str(row["publication_number"]),
            label,
            str(row["configuration"]),
        ]
        return " ".join(part for part in parts if part).strip()

    def _open_optical_model(self, path: Path) -> Any:
        return self._with_rayoptics_runtime(lambda: ro_open_model(path))

    def _with_rayoptics_runtime(self, action: Any) -> Any:
        with self.rayoptics_lock:
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self._prepare_rayoptics_log_handlers()
            previous_cwd = Path.cwd()
            os.chdir(self.runtime_dir)
            try:
                return action()
            finally:
                os.chdir(previous_cwd)

    def _prepare_rayoptics_log_handlers(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        for module_name, filename in (
            ("rayoptics.codev.cmdproc", "cv_cmd_proc.log"),
            ("rayoptics.zemax.zmxread", "zmx_read_lens.log"),
        ):
            try:
                module = __import__(module_name, fromlist=["_fh"])
                handler = getattr(module, "_fh", None)
                if handler is None or not hasattr(handler, "baseFilename"):
                    continue
                try:
                    handler.close()
                except Exception:
                    pass
                handler.baseFilename = str(self.runtime_dir / filename)
                handler.stream = None
            except Exception:
                pass

    def check_examples(self, payload: ExampleCheckRequest) -> ExampleCheckResponse:
        with self.rayoptics_lock:
            paths = [Path(path).expanduser() for path in payload.paths] if payload.paths else self._default_example_check_paths(payload.limit)
            if not paths:
                raise ValueError("No RayOptics example files were found.")
            checks = [self._check_example_path(path, include_analyses=payload.include_analyses) for path in paths[: payload.limit]]
        return ExampleCheckResponse(
            checked_at=_now().isoformat(),
            total=len(checks),
            passed=sum(1 for check in checks if check.status == "pass"),
            warned=sum(1 for check in checks if check.status == "warn"),
            failed=sum(1 for check in checks if check.status == "fail"),
            checks=checks,
        )

    def _default_example_check_paths(self, limit: int) -> list[Path]:
        examples = self.list_examples()
        selected: list[Path] = []
        selected_keys: set[str] = set()

        for name in EXAMPLE_CHECK_PRIORITY:
            match = next((example for example in examples if Path(example["path"]).name == name), None)
            if match is None:
                continue
            path = Path(match["path"])
            selected.append(path)
            selected_keys.add(str(path))

        for example in examples:
            if len(selected) >= limit:
                break
            path = Path(example["path"])
            key = str(path)
            if key not in selected_keys:
                selected.append(path)
                selected_keys.add(key)
        return selected[:limit]

    def _check_example_path(self, path: Path, include_analyses: bool) -> ExampleCheckResultDTO:
        start = time.perf_counter()
        target = path.expanduser()
        stages: list[ExampleCheckStageDTO] = []
        warnings: list[str] = []
        errors: list[str] = []
        model_name: str | None = None
        surface_count: int | None = None
        field_count: int | None = None
        wavelength_count: int | None = None
        kind = self._example_kind_for_path(target)
        label = target.name

        opm = None
        try:
            if target.suffix.lower() == ".smx":
                raise ValueError(".smx import is experimental in RayOptics and is disabled in this MVP.")
            if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {target.suffix or '(none)'}")
            if not target.exists():
                raise ValueError(f"File does not exist: {target}")
            opm = self._open_optical_model(target)
            if opm is None:
                raise ValueError(f"RayOptics could not open {target}")
            stages.append(_check_stage("open", "pass", "Loaded through rayoptics.gui.appcmds.open_model."))
        except Exception as exc:
            errors.append(f"Open failed: {type(exc).__name__}: {exc}")
            stages.append(_check_stage("open", "fail", errors[-1]))
            return self._example_check_result(
                label,
                target,
                kind,
                stages,
                warnings,
                errors,
                start,
                model_name,
                surface_count,
                field_count,
                wavelength_count,
            )

        try:
            repair_warnings = self._repair_restored_model(opm)
            warnings.extend(repair_warnings)
            stages.append(
                _check_stage(
                    "repair",
                    "warn" if repair_warnings else "pass",
                    "; ".join(repair_warnings) if repair_warnings else "No compatibility repairs required.",
                )
            )
        except Exception as exc:
            errors.append(f"Repair failed: {type(exc).__name__}: {exc}")
            stages.append(_check_stage("repair", "fail", errors[-1]))

        try:
            opm.update_model()
            stages.append(_check_stage("update", "pass", "update_model completed."))
        except Exception as exc:
            errors.append(f"update_model failed: {type(exc).__name__}: {exc}")
            stages.append(_check_stage("update", "fail", errors[-1]))

        try:
            sm = opm.seq_model
            osp = opm.optical_spec
            surface_count = sm.get_num_surfaces()
            field_count = len(osp.field_of_view.fields)
            wavelength_count = len(osp.spectral_region.wavelengths)
            model_name = opm.name() or target.stem
            stages.append(
                _check_stage(
                    "model dto",
                    "pass",
                    f"{surface_count} surfaces, {field_count} fields, {wavelength_count} wavelengths.",
                )
            )
        except Exception as exc:
            errors.append(f"Model DTO failed: {type(exc).__name__}: {exc}")
            stages.append(_check_stage("model dto", "fail", errors[-1]))

        try:
            layout = self._layout_payload(opm)
            layout_warnings = layout.get("warnings", [])
            warnings.extend(str(warning) for warning in layout_warnings)
            stages.append(
                _check_stage(
                    "layout",
                    "warn" if layout_warnings else "pass",
                    f"{len(layout.get('surfaces', []))} layout surfaces, {len(layout.get('rays', []))} ray bundles.",
                )
            )
        except Exception as exc:
            errors.append(f"Layout failed: {type(exc).__name__}: {exc}")
            stages.append(_check_stage("layout", "fail", errors[-1]))

        try:
            first_order = self._first_order_values(opm)
            useful_values = [value for key, value in first_order.items() if key in {"efl", "fno", "bfl", "img_ht"} and value is not None]
            stages.append(
                _check_stage(
                    "first order",
                    "pass" if useful_values else "warn",
                    f"{len(useful_values)} key first-order values available." if useful_values else "No key first-order values available.",
                )
            )
        except Exception as exc:
            errors.append(f"First-order failed: {type(exc).__name__}: {exc}")
            stages.append(_check_stage("first order", "fail", errors[-1]))

        if include_analyses:
            analysis_payload = AnalysisRequest(sampling=9, scale="same")
            for analysis_kind in EXAMPLE_ANALYSIS_KINDS:
                try:
                    svg = self._analysis_svg_for_model(opm, analysis_kind, analysis_payload)
                    if "<svg" not in svg[:500].lower():
                        warnings.append(f"{analysis_kind} returned non-SVG output.")
                        stages.append(_check_stage(analysis_kind, "warn", "Analysis returned output, but it did not look like SVG."))
                    else:
                        stages.append(_check_stage(analysis_kind, "pass", f"Rendered {len(svg):,} SVG characters."))
                except Exception as exc:
                    errors.append(f"{analysis_kind} failed: {type(exc).__name__}: {exc}")
                    stages.append(_check_stage(analysis_kind, "fail", errors[-1]))

        return self._example_check_result(
            label,
            target,
            kind,
            stages,
            warnings,
            errors,
            start,
            model_name,
            surface_count,
            field_count,
            wavelength_count,
        )

    def _example_check_result(
        self,
        label: str,
        target: Path,
        kind: str,
        stages: list[ExampleCheckStageDTO],
        warnings: list[str],
        errors: list[str],
        start: float,
        model_name: str | None,
        surface_count: int | None,
        field_count: int | None,
        wavelength_count: int | None,
    ) -> ExampleCheckResultDTO:
        status = _check_status(stages, warnings, errors)
        return ExampleCheckResultDTO(
            label=label,
            path=str(target),
            kind=kind,
            status=status,
            duration_ms=max(0, round((time.perf_counter() - start) * 1000)),
            model_name=model_name,
            surface_count=surface_count,
            field_count=field_count,
            wavelength_count=wavelength_count,
            stages=stages,
            warnings=warnings,
            errors=errors,
        )

    def _example_kind_for_path(self, path: Path) -> str:
        resolved = path.expanduser()
        for example in self.list_examples():
            if Path(example["path"]) == resolved:
                return example["kind"]
        if resolved.suffix.lower() == ".seq":
            return "CODE V"
        if resolved.suffix.lower() == ".zmx":
            return "Zemax"
        if resolved.suffix.lower() == ".roa":
            return "RayOptics Model"
        return "Unknown"

    def new_model(self, payload: NewModelRequest) -> ModelResponse:
        opm = self._create_default_model(payload)
        return self._register(opm, None, ["Created a new singlet design."])

    def open_model(self, path: Path) -> ModelResponse:
        if path.suffix.lower() == ".smx":
            raise ValueError(".smx import is experimental in RayOptics and is disabled in this MVP. Use .zmx when available.")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix or '(none)'}")
        if not path.exists():
            raise ValueError(f"File does not exist: {path}")
        opm = self._open_optical_model(path)
        if opm is None:
            raise ValueError(f"RayOptics could not open {path}")
        warnings = self._repair_restored_model(opm)
        try:
            opm.update_model()
        except Exception as exc:
            warnings.append(f"Opened file, but update_model failed: {type(exc).__name__}: {exc}")
        response = self._register(opm, str(path), warnings)
        session = self.sessions[response.model.id]
        metadata_warnings = self._load_workbench_metadata(path, session)
        session.warnings = [*warnings, *metadata_warnings]
        return self.response(session.model_id)

    def save_model(self, model_id: str, path: Path | None, overwrite: bool = False, workbench: dict[str, Any] | None = None) -> ModelResponse:
        session = self.sessions[model_id]
        target = path or (Path(session.filename) if session.filename else None)
        if target is None:
            raise ValueError("Save path is required for models without a filename.")
        target = target.expanduser()
        if target.suffix.lower() != ".roa":
            target = target.with_suffix(".roa")
        if not target.parent.exists():
            raise ValueError(f"Save directory does not exist: {target.parent}")
        if not target.parent.is_dir():
            raise ValueError(f"Save parent is not a directory: {target.parent}")
        resolved_target = target.resolve()
        if resolved_target.is_relative_to(self.package_root):
            raise ValueError("Refusing to overwrite bundled RayOptics examples. Save a workspace copy instead.")
        current_file = Path(session.filename).expanduser().resolve() if session.filename else None
        if resolved_target.exists() and current_file != resolved_target and not overwrite:
            raise ValueError(f"File already exists: {target}. Confirm overwrite to replace it.")
        try:
            if workbench is not None:
                session.workbench_metadata = self._json_safe_metadata(workbench)
            session.opt_model.save_model(target, version="0.9.0a1")
            metadata_path = self._write_workbench_metadata(target, session)
        except Exception as exc:
            raise ValueError(f"Save failed: {type(exc).__name__}: {exc}") from exc
        session.filename = str(target)
        session.dirty = False
        session.undo_stack.clear()
        session.redo_stack.clear()
        session.warnings = [f"Saved model to {target}.", f"Saved workbench metadata to {metadata_path}."]
        session.errors = []
        session.last_updated_at = _now()
        return self.response(model_id)

    def autosave_draft(self, model_id: str, workbench: dict[str, Any] | None = None) -> DraftAutosaveResponse:
        session = self.sessions[model_id]
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        target = self.draft_dir / f"{model_id}.roa"
        temp_target = target.with_suffix(".tmp.roa")
        saved_at = _now()
        try:
            if workbench is not None:
                session.workbench_metadata = self._json_safe_metadata(workbench)
            session.opt_model.save_model(temp_target, version="0.9.0a1")
            temp_target.replace(target)
            self._write_workbench_metadata(target, session)
        except Exception as exc:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            raise ValueError(f"Draft autosave failed: {type(exc).__name__}: {exc}") from exc
        pruned_count = self._prune_drafts(model_id)
        draft_count = len(list(self.draft_dir.glob("*.roa")))
        return DraftAutosaveResponse(
            model_id=model_id,
            path=str(target),
            saved_at=saved_at.isoformat(),
            draft_count=draft_count,
            pruned_count=pruned_count,
        )

    def restore_draft(self, path: Path) -> ModelResponse:
        target = path.expanduser()
        if target.suffix.lower() != ".roa":
            raise ValueError("Draft restore only supports .roa draft files.")
        resolved_target = target.resolve()
        if not resolved_target.is_relative_to(self.draft_dir.resolve()):
            raise ValueError("Draft restore path is outside the RayOptics workbench draft directory.")
        if not resolved_target.exists():
            raise ValueError(f"Draft file does not exist: {target}")
        opm = self._open_optical_model(resolved_target)
        if opm is None:
            raise ValueError(f"RayOptics could not restore draft {target}")
        warnings = self._repair_restored_model(opm)
        try:
            opm.update_model()
        except Exception as exc:
            warnings.append(f"Restored draft, but update_model failed: {type(exc).__name__}: {exc}")
        response = self._register(opm, None, ["Restored autosaved draft.", *warnings])
        session = self.sessions[response.model.id]
        metadata_warnings = self._load_workbench_metadata(resolved_target, session)
        session.dirty = True
        session.warnings = ["Restored autosaved draft.", *warnings, *metadata_warnings]
        return self.response(session.model_id)

    def _prune_drafts(self, active_model_id: str) -> int:
        if not self.draft_dir.exists():
            return 0
        pruned_count = 0
        for temp_file in self.draft_dir.glob("*.tmp.roa"):
            try:
                temp_file.unlink(missing_ok=True)
                pruned_count += 1
            except OSError:
                pass
        active_name = f"{active_model_id}.roa"
        drafts = sorted(self.draft_dir.glob("*.roa"), key=lambda path: path.stat().st_mtime, reverse=True)
        kept = 0
        for draft in drafts:
            if draft.name == active_name:
                kept += 1
                continue
            if kept < DRAFT_KEEP_LIMIT:
                kept += 1
                continue
            try:
                draft.unlink(missing_ok=True)
                self._workbench_metadata_path(draft).unlink(missing_ok=True)
                pruned_count += 1
            except OSError:
                pass
        return pruned_count

    def response(self, model_id: str) -> ModelResponse:
        return self._response(self.sessions[model_id])

    def patch_settings(self, model_id: str, payload: ModelSettingsPatchRequest) -> ModelResponse:
        session = self.sessions[model_id]
        provided = payload.model_dump(exclude_none=True)
        if not provided:
            session.warnings = ["No workbench settings changed."]
            session.errors = []
            session.last_updated_at = _now()
            return self.response(model_id)
        if payload.update_mode is not None:
            session.update_mode = payload.update_mode
        session.errors = []
        session.last_updated_at = _now()
        return self.response(model_id)

    def patch_surface(self, model_id: str, surface_index: int, payload: SurfacePatchRequest) -> ModelResponse:
        session = self.sessions[model_id]
        sm = session.opt_model.seq_model
        self._assert_surface_index(sm, surface_index, allow_object_image=True)
        provided = payload.model_dump(exclude_unset=True, by_alias=False)
        if provided and all(value is None for value in provided.values()):
            session.warnings = ["No surface settings changed."]
            session.errors = []
            session.last_updated_at = _now()
            return self.response(model_id)
        for field_name, value in (
            ("Radius", payload.radius),
            ("Curvature", payload.curvature),
            ("Thickness", payload.thickness),
            ("Semi-Diameter", payload.semi_diameter),
            ("Conic", payload.conic),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if payload.semi_diameter is not None and payload.semi_diameter <= 0:
            raise ValueError("Semi-Diameter must be greater than zero.")
        if surface_index == 0 or surface_index == sm.get_num_surfaces() - 1:
            if any(
                value is not None
                for key, value in payload.model_dump(by_alias=False).items()
                if key not in {"label", "stop"}
            ):
                raise ValueError("Object and image rows only allow label or stop edits.")

        variable_update = payload.variable is not None
        optical_update = any(
            value is not None
            for key, value in payload.model_dump(by_alias=False).items()
            if key != "variable"
        )

        if variable_update and not optical_update:
            self._set_surface_variables(session, surface_index, payload.variable or "")
            session.errors = []
            session.warnings = [self._variable_warning(surface_index)]
            session.last_updated_at = _now()
            return self.response(model_id)

        ifc = sm.ifcs[surface_index]
        gap = sm.gaps[surface_index] if surface_index < len(sm.gaps) else None

        before = copy.deepcopy(session.opt_model)
        before_variables = copy.deepcopy(session.surface_variables)
        warnings: list[str] = []
        try:
            if payload.label is not None:
                ifc.label = payload.label
            if payload.radius is not None:
                ifc.profile_cv = _cv_from_radius(payload.radius)
            if payload.curvature is not None:
                ifc.profile_cv = payload.curvature
            if payload.thickness is not None and gap is not None:
                gap.thi = payload.thickness
            if payload.semi_diameter is not None:
                ifc.max_aperture = abs(payload.semi_diameter)
            if payload.conic is not None:
                if hasattr(ifc.profile, "cc"):
                    ifc.profile.cc = payload.conic
                elif hasattr(ifc.profile, "ec"):
                    ifc.profile.ec = payload.conic
                else:
                    warnings.append("Conic was ignored because this surface profile is spherical.")
            if payload.mode is not None:
                ifc.interact_mode = payload.mode
            if (payload.glass is not None or payload.catalog is not None) and gap is not None:
                current_medium = gap.medium
                current_glass = current_medium.name() if current_medium else "air"
                current_catalog = current_medium.catalog_name() if current_medium else ""
                next_glass = payload.glass if payload.glass is not None else current_glass
                next_catalog = payload.catalog if payload.catalog is not None else current_catalog
                if payload.catalog is not None and payload.glass is None and next_glass.strip().lower() in {"", "air"}:
                    warnings.append("Catalog was ignored because the surface medium is air.")
                else:
                    gap.medium = self._make_medium(next_glass, next_catalog)
            if payload.stop is not None:
                sm.stop_surface = surface_index if payload.stop else None
            if variable_update:
                self._set_surface_variables(session, surface_index, payload.variable or "")
                warnings.append(self._variable_warning(surface_index))
            response = self._commit(session, before, dirty=True, warnings=warnings)
            if response.errors:
                session.surface_variables = before_variables
                return self.response(model_id)
            return response
        except Exception as exc:
            session.opt_model = before
            session.surface_variables = before_variables
            session.errors = [f"Edit rejected: {type(exc).__name__}: {exc}"]
            session.last_updated_at = _now()
            return self.response(model_id)

    def patch_system(self, model_id: str, payload: SystemPatchRequest) -> ModelResponse:
        session = self.sessions[model_id]
        provided = payload.model_dump(exclude_none=True)
        if not provided:
            session.warnings = ["No system settings changed."]
            session.errors = []
            session.last_updated_at = _now()
            return self.response(model_id)

        before = copy.deepcopy(session.opt_model)
        osp = session.opt_model.optical_spec
        try:
            if payload.aperture_value is not None:
                if payload.aperture_value <= 0:
                    raise ValueError("Aperture value must be greater than zero.")
                osp.pupil.value = float(payload.aperture_value)
            if payload.field_value is not None:
                if payload.field_value < 0:
                    raise ValueError("Field value must be zero or greater.")
                osp.field_of_view.value = float(payload.field_value)
            if payload.field_x_values is not None or payload.field_y_values is not None or payload.field_weights is not None:
                self._patch_fields(osp, payload)
            if payload.focus_shift is not None or payload.defocus_range is not None:
                focus = getattr(osp, "defocus", None)
                if focus is None:
                    raise ValueError("Focus range is unavailable for this model.")
                if payload.focus_shift is not None:
                    focus.focus_shift = float(payload.focus_shift)
                if payload.defocus_range is not None:
                    if payload.defocus_range < 0:
                        raise ValueError("Defocus range must be zero or greater.")
                    focus.defocus_range = float(payload.defocus_range)
            if (
                payload.wavelength_values is not None
                or payload.wavelength_weights is not None
                or payload.wavelength_reference_index is not None
            ):
                self._patch_wavelengths(osp, payload)
            return self._commit(session, before, dirty=True, warnings=["Updated system settings."])
        except Exception as exc:
            session.opt_model = before
            session.errors = [f"System edit rejected: {type(exc).__name__}: {exc}"]
            session.last_updated_at = _now()
            return self.response(model_id)

    def _patch_fields(self, osp: Any, payload: SystemPatchRequest) -> None:
        fov = osp.field_of_view
        current = fov.fields
        x_values = (
            [float(value) for value in payload.field_x_values]
            if payload.field_x_values is not None
            else [float(getattr(field, "x", 0.0)) for field in current]
        )
        y_values = (
            [float(value) for value in payload.field_y_values]
            if payload.field_y_values is not None
            else [float(getattr(field, "y", 0.0)) for field in current]
        )
        weights = (
            [float(value) for value in payload.field_weights]
            if payload.field_weights is not None
            else [float(getattr(field, "wt", 1.0)) for field in current]
        )

        if not 1 <= len(y_values) <= 12:
            raise ValueError("Field list must contain 1 to 12 entries.")
        if len(x_values) != len(y_values) or len(weights) != len(y_values):
            raise ValueError("Field x, y, and weight lists must have the same length.")
        if any(not math.isfinite(value) for value in x_values + y_values):
            raise ValueError("Field coordinates must be finite values.")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Field weights must be finite values greater than or equal to zero.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one field weight must be greater than zero.")

        if len(current) == len(y_values):
            for field, x_value, y_value, weight in zip(current, x_values, y_values, weights):
                field.x = x_value
                field.y = y_value
                field.wt = weight
                field.fov = fov
        else:
            fov.fields = [Field(x=x_value, y=y_value, wt=weight, fov=fov) for x_value, y_value, weight in zip(x_values, y_values, weights)]
            fov.index_label_type = "auto"

    def _patch_wavelengths(self, osp: Any, payload: SystemPatchRequest) -> None:
        current = osp.spectral_region
        values = (
            [float(value) for value in payload.wavelength_values]
            if payload.wavelength_values is not None
            else [float(value) for value in current.wavelengths]
        )
        weights = (
            [float(weight) for weight in payload.wavelength_weights]
            if payload.wavelength_weights is not None
            else [float(weight) for weight in current.spectral_wts]
        )

        if not 1 <= len(values) <= 7:
            raise ValueError("Wavelength list must contain 1 to 7 entries.")
        if len(weights) != len(values):
            raise ValueError("Wavelength weights must match the wavelength list length.")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Wavelength values must be positive finite nm values.")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Wavelength weights must be finite values greater than or equal to zero.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one wavelength weight must be greater than zero.")
        if any(values[index] >= values[index + 1] for index in range(len(values) - 1)):
            raise ValueError("Wavelength values must be strictly increasing.")

        reference = (
            payload.wavelength_reference_index
            if payload.wavelength_reference_index is not None
            else min(int(current.reference_wvl), len(values) - 1)
        )
        if reference < 0 or reference >= len(values):
            raise ValueError("Reference wavelength index is out of range.")

        coating_wvl = getattr(current, "coating_wvl", 550.0)
        osp.spectral_region = WvlSpec(list(zip(values, weights)), ref_wl=reference)
        osp.spectral_region.coating_wvl = coating_wvl

    def create_surface(self, model_id: str, payload: SurfaceCreateRequest) -> ModelResponse:
        session = self.sessions[model_id]
        sm = session.opt_model.seq_model
        after = payload.after if payload.after is not None else max(0, sm.get_num_surfaces() - 2)
        if after < 0 or after >= sm.get_num_surfaces() - 1:
            raise ValueError("New surfaces must be inserted after an existing non-image surface.")
        before = copy.deepcopy(session.opt_model)
        before_variables = copy.deepcopy(session.surface_variables)
        try:
            sm.set_cur_surface(after)
            radius_mode = session.opt_model.radius_mode
            session.opt_model.radius_mode = True
            mat_data: list[Any] = [payload.radius, payload.thickness]
            if payload.glass and payload.glass.lower() != "air":
                mat_data.extend([payload.glass, payload.catalog or "Schott"])
            mat_data.append(payload.semi_diameter)
            sm.add_surface(mat_data)
            session.opt_model.radius_mode = radius_mode
            self._shift_variables_after_insert(session, after + 1)
            response = self._commit(session, before, dirty=True, warnings=["Inserted a new surface."])
            if response.errors:
                session.surface_variables = before_variables
                return self.response(model_id)
            return response
        except Exception as exc:
            session.opt_model = before
            session.surface_variables = before_variables
            session.errors = [f"Insert rejected: {type(exc).__name__}: {exc}"]
            session.last_updated_at = _now()
            return self.response(model_id)

    def delete_surface(self, model_id: str, surface_index: int) -> ModelResponse:
        session = self.sessions[model_id]
        sm = session.opt_model.seq_model
        self._assert_surface_index(sm, surface_index, allow_object_image=False)
        before = copy.deepcopy(session.opt_model)
        before_variables = copy.deepcopy(session.surface_variables)
        try:
            sm.remove(surface_index)
            self._shift_variables_after_delete(session, surface_index)
            response = self._commit(session, before, dirty=True, warnings=["Deleted surface."])
            if response.errors:
                session.surface_variables = before_variables
                return self.response(model_id)
            return response
        except Exception as exc:
            session.opt_model = before
            session.surface_variables = before_variables
            session.errors = [f"Delete rejected: {type(exc).__name__}: {exc}"]
            session.last_updated_at = _now()
            return self.response(model_id)

    def update_model(self, model_id: str) -> ModelResponse:
        session = self.sessions[model_id]
        return self._commit(session, copy.deepcopy(session.opt_model), dirty=session.dirty, warnings=["Model updated."], record_history=False)

    def tolerance_sweep(self, model_id: str, payload: ToleranceSweepRequest) -> ToleranceSweepResponse:
        session = self.sessions[model_id]
        response = self.response(model_id)
        try:
            result = self._tolerance_sweep_result(session, payload)
        except Exception as exc:
            result = ToleranceSweepResultDTO(
                status="fail",
                scope=payload.scope,
                perturbation_pct=payload.perturbation_pct,
                baseline_score=0.0,
                attempted_cases=0,
                passed_cases=0,
                warned_cases=0,
                failed_cases=0,
                worst_case=None,
                worst_score=None,
                cases=[],
                warnings=[f"Tolerance sweep failed without changing the model: {type(exc).__name__}: {exc}"],
            )
        return ToleranceSweepResponse(
            model=response.model,
            warnings=response.warnings,
            errors=response.errors,
            dirty=response.dirty,
            can_undo=response.can_undo,
            can_redo=response.can_redo,
            last_updated_at=response.last_updated_at,
            result=result,
        )

    def quick_optimize(self, model_id: str, payload: QuickOptimizeRequest) -> QuickOptimizeResponse:
        session = self.sessions[model_id]
        variables = self._optimization_variables(session)
        weights = self._quick_optimize_weights(payload.objective, payload.operand_weights)
        targets = self._quick_optimize_targets(payload.operand_targets)
        if not variables:
            result = QuickOptimizeResultDTO(
                status="failed",
                objective=payload.objective,
                message="No workbench variable flags are active. Mark R, T, SD, or K in the Lens Data Editor first.",
                baseline_score=0.0,
                final_score=0.0,
                improvement=0.0,
                evaluations=0,
                iterations=0,
                variable_count=0,
                operand_weights=weights,
                variables=[],
                moves=[],
            )
            session.warnings = ["Quick Optimize skipped because no variable flags are active."]
            session.errors = []
            session.last_updated_at = _now()
            return self._quick_optimize_response(session, result)

        before = copy.deepcopy(session.opt_model)
        best_model = copy.deepcopy(session.opt_model)
        best_score, baseline_message = self._quick_optimize_score(best_model, weights, targets)
        if not math.isfinite(best_score):
            result = QuickOptimizeResultDTO(
                status="failed",
                objective=payload.objective,
                message=f"Baseline model could not be scored: {baseline_message}",
                baseline_score=0.0,
                final_score=0.0,
                improvement=0.0,
                evaluations=1,
                iterations=0,
                variable_count=len(variables),
                operand_weights=weights,
                variables=[variable["label"] for variable in variables],
                moves=[],
            )
            session.warnings = []
            session.errors = [result.message]
            session.last_updated_at = _now()
            return self._quick_optimize_response(session, result)

        evaluations = 1
        accepted_moves: list[QuickOptimizeMoveDTO] = []
        iterations_done = 0
        max_evaluations = payload.max_evaluations

        for iteration in range(payload.iterations):
            if evaluations >= max_evaluations:
                break
            iterations_done = iteration + 1
            improved_this_iteration = False
            for variable in variables:
                if evaluations >= max_evaluations:
                    break
                current_value = self._read_optimization_variable(best_model, variable)
                if current_value is None:
                    continue
                step = self._optimization_step(best_model, variable, payload.step_scale)
                if step is None or step <= 0:
                    continue

                best_candidate_model = None
                best_candidate_value = current_value
                best_candidate_score = best_score
                for direction in (-1.0, 1.0):
                    if evaluations >= max_evaluations:
                        break
                    candidate = copy.deepcopy(best_model)
                    candidate_value = current_value + direction * step
                    if not self._write_optimization_variable(candidate, variable, candidate_value):
                        continue
                    score, _ = self._quick_optimize_score(candidate, weights, targets)
                    evaluations += 1
                    if score > best_candidate_score + 1.0e-5:
                        best_candidate_score = score
                        best_candidate_value = candidate_value
                        best_candidate_model = candidate

                if best_candidate_model is not None:
                    best_model = best_candidate_model
                    best_score = best_candidate_score
                    improved_this_iteration = True
                    accepted_moves.append(
                        QuickOptimizeMoveDTO(
                            surface_index=variable["surface_index"],
                            token=variable["token"],
                            label=variable["label"],
                            before=current_value,
                            after=best_candidate_value,
                            score=best_score,
                        )
                    )
            if not improved_this_iteration:
                break

        baseline_score = self._quick_optimize_score(before, weights, targets)[0]
        final_score = best_score
        improvement = final_score - baseline_score

        if improvement > 1.0e-5 and accepted_moves:
            session.opt_model = best_model
            response = self._commit(
                session,
                before,
                dirty=True,
                warnings=[
                    f"Quick Optimize accepted {len(accepted_moves)} coordinate-search moves. This is a bounded workbench search, not a full merit-function optimizer."
                ],
            )
            session = self.sessions[response.model.id]
            result = QuickOptimizeResultDTO(
                status="improved" if not response.errors else "failed",
                objective=payload.objective,
                message=(
                    f"Improved {payload.objective} workbench score by {improvement:.4f} using {len(accepted_moves)} moves."
                    if not response.errors
                    else response.errors[0]
                ),
                baseline_score=baseline_score,
                final_score=final_score if not response.errors else baseline_score,
                improvement=improvement if not response.errors else 0.0,
                evaluations=evaluations,
                iterations=iterations_done,
                variable_count=len(variables),
                operand_weights=weights,
                variables=[variable["label"] for variable in variables],
                moves=accepted_moves if not response.errors else [],
            )
            return self._quick_optimize_response(session, result)

        session.warnings = [
            f"Quick Optimize evaluated {evaluations} candidates but did not find an improving coordinate move."
        ]
        session.errors = []
        session.last_updated_at = _now()
        result = QuickOptimizeResultDTO(
            status="no-change",
            objective=payload.objective,
            message="No improving move found within the bounded coordinate search.",
            baseline_score=baseline_score,
            final_score=baseline_score,
            improvement=0.0,
            evaluations=evaluations,
            iterations=iterations_done,
            variable_count=len(variables),
            operand_weights=weights,
            variables=[variable["label"] for variable in variables],
            moves=[],
        )
        return self._quick_optimize_response(session, result)

    def undo(self, model_id: str) -> ModelResponse:
        session = self.sessions[model_id]
        if not session.undo_stack:
            session.warnings = ["Nothing to undo."]
            session.errors = []
            session.last_updated_at = _now()
            return self.response(model_id)
        session.redo_stack.append(copy.deepcopy(session.opt_model))
        session.opt_model = session.undo_stack.pop()
        try:
            self._repair_restored_model(session.opt_model)
            session.opt_model.update_model()
            session.last_valid_model = copy.deepcopy(session.opt_model)
            session.errors = []
            session.warnings = ["Undid last edit."]
            session.dirty = bool(session.undo_stack)
        except Exception as exc:
            session.errors = [f"Undo failed: {type(exc).__name__}: {exc}"]
            session.warnings = []
        session.last_updated_at = _now()
        return self.response(model_id)

    def redo(self, model_id: str) -> ModelResponse:
        session = self.sessions[model_id]
        if not session.redo_stack:
            session.warnings = ["Nothing to redo."]
            session.errors = []
            session.last_updated_at = _now()
            return self.response(model_id)
        session.undo_stack.append(copy.deepcopy(session.opt_model))
        session.opt_model = session.redo_stack.pop()
        try:
            self._repair_restored_model(session.opt_model)
            session.opt_model.update_model()
            session.last_valid_model = copy.deepcopy(session.opt_model)
            session.errors = []
            session.warnings = ["Redid edit."]
            session.dirty = True
        except Exception as exc:
            session.errors = [f"Redo failed: {type(exc).__name__}: {exc}"]
            session.warnings = []
        session.last_updated_at = _now()
        return self.response(model_id)

    def layout(self, model_id: str) -> dict[str, Any]:
        return self._layout_payload(self.sessions[model_id].opt_model)

    def _layout_payload(self, opm: Any) -> dict[str, Any]:
        sm = opm.seq_model
        warnings: list[str] = []
        z_by_surface = self._surface_z_positions(sm)
        surfaces = []
        for idx, ifc in enumerate(sm.ifcs):
            if idx == 0:
                continue
            surfaces.append(
                {
                    "index": idx,
                    "label": ifc.label or ("IMG" if idx == sm.get_num_surfaces() - 1 else str(idx)),
                    "z": z_by_surface[idx],
                    "semiDiameter": _safe_float(ifc.surface_od()) or _safe_float(ifc.max_aperture) or 1.0,
                    "radius": _radius_from_cv(_safe_float(ifc.profile_cv) or 0.0),
                    "mode": ifc.interact_mode,
                    "isStop": idx == sm.stop_surface,
                }
            )
        rays = self._layout_rays(opm, z_by_surface, warnings)
        return {"surfaces": surfaces, "rays": rays, "warnings": warnings}

    def first_order(self, model_id: str) -> dict[str, Any]:
        session = self.sessions[model_id]
        values = self._first_order_values(session.opt_model)
        if not values:
            return {"values": {}, "warnings": ["No paraxial data available."]}
        return {"values": values, "warnings": []}

    def analysis_summary(self, model_id: str) -> AnalysisSummaryResponse:
        session = self.sessions[model_id]
        first_order = self._first_order_values(session.opt_model)
        sensor = self._sensor_assumptions(session)
        counts, metrics, risks = self._cockpit_summary(session, first_order, sensor)
        return AnalysisSummaryResponse(
            model=self._model_dto(session),
            warnings=session.warnings,
            errors=session.errors,
            dirty=session.dirty,
            can_undo=bool(session.undo_stack),
            can_redo=bool(session.redo_stack),
            last_updated_at=session.last_updated_at.isoformat(),
            first_order=first_order,
            counts=counts,
            metrics=metrics,
            risks=risks,
            sensor=sensor,
        )

    def patch_sensor(self, model_id: str, payload: SensorPatchRequest) -> AnalysisSummaryResponse:
        session = self.sessions[model_id]
        for key, value in payload.model_dump(exclude_none=True).items():
            session.sensor_overrides[key] = value
        if payload.model_dump(exclude_none=True):
            session.warnings = ["Updated sensor assumptions for analysis. Optical prescription was not changed."]
        session.errors = []
        session.last_updated_at = _now()
        return self.analysis_summary(model_id)

    def analysis_svg(self, model_id: str, kind: str, payload: AnalysisRequest) -> str:
        return self._analysis_svg_for_model(self.sessions[model_id].opt_model, kind, payload)

    def _analysis_svg_for_model(self, source_model: Any, kind: str, payload: AnalysisRequest) -> str:
        return self._with_rayoptics_runtime(lambda: self._render_analysis_svg(source_model, kind, payload))

    def _render_analysis_svg(self, source_model: Any, kind: str, payload: AnalysisRequest) -> str:
        opm = self._analysis_view_model(source_model, payload)
        try:
            return self._render_analysis_figure_svg(opm, kind, payload)
        except Exception as exc:
            if payload.field_index is None and kind in {"ray-fan", "opd-fan", "spot"}:
                return self._render_filtered_field_analysis_svg(source_model, kind, payload, exc)
            return self._diagnostic_svg(
                f"{self._analysis_label(kind)} unavailable",
                [
                    "The selected field could not be traced by the RayOptics analysis renderer.",
                    f"{type(exc).__name__}: {exc}",
                    "Try a smaller field, lower sampling, or inspect Layout/Wavefront for this model.",
                ],
            )

    def _render_analysis_figure_svg(
        self,
        opm: Any,
        kind: str,
        payload: AnalysisRequest,
        footer: str | None = None,
    ) -> str:
        fit = Fit.All_Same if payload.scale == "same" else Fit.All
        kwargs = {"dpi": 100, "is_dark": False}
        if kind == "ray-fan":
            fig = RayFanFigure(opm, "Ray", num_rays=payload.sampling, scale_type=fit, **kwargs)
        elif kind == "opd-fan":
            fig = RayFanFigure(opm, "OPD", num_rays=payload.sampling, scale_type=fit, **kwargs)
        elif kind == "spot":
            fig = SpotDiagramFigure(opm, num_rays=payload.sampling, scale_type=fit, **kwargs)
        elif kind == "wavefront":
            fig = WavefrontFigure(opm, num_rays=payload.sampling, scale_type=fit, **kwargs)
        elif kind == "field-curves":
            fig = FieldCurveFigure(opm, num_points=payload.sampling, **kwargs)
        else:
            raise ValueError(f"Unsupported analysis: {kind}")
        FigureCanvasAgg(fig)
        fig.plot()
        if footer:
            fig.text(0.01, 0.01, footer, fontsize=7, color="#555555", ha="left", va="bottom")
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        return buf.getvalue()

    def _render_filtered_field_analysis_svg(
        self,
        source_model: Any,
        kind: str,
        payload: AnalysisRequest,
        original_error: Exception,
    ) -> str:
        fields = list(source_model.optical_spec.field_of_view.fields)
        valid_indices: list[int] = []
        failed_indices: list[int] = []
        for idx in range(len(fields)):
            trial_payload = payload.model_copy(update={"field_index": idx})
            try:
                trial_model = self._analysis_view_model(source_model, trial_payload)
                self._render_analysis_figure_svg(trial_model, kind, trial_payload)
                valid_indices.append(idx)
            except Exception:
                failed_indices.append(idx)

        if not valid_indices:
            return self._diagnostic_svg(
                f"{self._analysis_label(kind)} unavailable",
                [
                    "No field produced enough valid traced rays for this analysis.",
                    f"{type(original_error).__name__}: {original_error}",
                    "Layout, First Order, and Wavefront may still be useful for import inspection.",
                ],
            )

        analysis_model = copy.deepcopy(source_model)
        osp = analysis_model.optical_spec
        if payload.wavelength_index is not None:
            wvls = osp.spectral_region
            if payload.wavelength_index >= len(wvls.wavelengths):
                raise ValueError("Analysis wavelength index is out of range.")
            selected_wvl = float(wvls.wavelengths[payload.wavelength_index])
            selected_wt = float(wvls.spectral_wts[payload.wavelength_index])
            coating_wvl = getattr(wvls, "coating_wvl", 550.0)
            osp.spectral_region = WvlSpec([(selected_wvl, selected_wt)], ref_wl=0)
            osp.spectral_region.coating_wvl = coating_wvl

        selected_fields = []
        for idx in valid_indices:
            selected = copy.deepcopy(fields[idx])
            selected.fov = osp.field_of_view
            selected_fields.append(selected)
        osp.field_of_view.fields = selected_fields
        osp.field_of_view.index_label_type = "auto"
        self._repair_restored_model(analysis_model)
        analysis_model.update_model()
        footer = (
            f"Rendered fields {', '.join(f'F{idx + 1}' for idx in valid_indices)}; "
            f"skipped {', '.join(f'F{idx + 1}' for idx in failed_indices)} due to trace instability."
        )
        return self._render_analysis_figure_svg(analysis_model, kind, payload, footer=footer)

    def _diagnostic_svg(self, title: str, lines: list[str]) -> str:
        fig = Figure(figsize=(7.5, 3.2), dpi=100)
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor("#ffffff")
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.04, 0.78, title, fontsize=14, fontweight="bold", color="#111827", transform=ax.transAxes)
        for idx, line in enumerate(lines[:5]):
            ax.text(0.04, 0.58 - idx * 0.15, line, fontsize=9, color="#374151", transform=ax.transAxes)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        return buf.getvalue()

    def _analysis_label(self, kind: str) -> str:
        return {
            "ray-fan": "Ray Fan",
            "opd-fan": "OPD Fan",
            "spot": "Spot Diagram",
            "wavefront": "Wavefront",
            "field-curves": "Field Curves",
        }.get(kind, kind)

    def _analysis_view_model(self, opm: Any, payload: AnalysisRequest) -> Any:
        if payload.field_index is None and payload.wavelength_index is None:
            return opm

        analysis_model = copy.deepcopy(opm)
        osp = analysis_model.optical_spec

        if payload.field_index is not None:
            fields = osp.field_of_view.fields
            if payload.field_index >= len(fields):
                raise ValueError("Analysis field index is out of range.")
            selected = copy.deepcopy(fields[payload.field_index])
            selected.fov = osp.field_of_view
            osp.field_of_view.fields = [selected]
            osp.field_of_view.index_label_type = "auto"

        if payload.wavelength_index is not None:
            wvls = osp.spectral_region
            if payload.wavelength_index >= len(wvls.wavelengths):
                raise ValueError("Analysis wavelength index is out of range.")
            selected_wvl = float(wvls.wavelengths[payload.wavelength_index])
            selected_wt = float(wvls.spectral_wts[payload.wavelength_index])
            coating_wvl = getattr(wvls, "coating_wvl", 550.0)
            osp.spectral_region = WvlSpec([(selected_wvl, selected_wt)], ref_wl=0)
            osp.spectral_region.coating_wvl = coating_wvl

        self._repair_restored_model(analysis_model)
        analysis_model.update_model()
        return analysis_model

    def _first_order_values(self, opm: Any) -> dict[str, float | None]:
        pdata = opm["analysis_results"].get("parax_data")
        if pdata is None:
            return {}
        fod = pdata.fod
        names = [
            "efl",
            "f",
            "ffl",
            "bfl",
            "fno",
            "m",
            "obj_dist",
            "obj_ang",
            "enp_dist",
            "enp_radius",
            "img_dist",
            "img_ht",
            "exp_dist",
            "exp_radius",
            "opt_inv",
        ]
        return {name: _safe_float(getattr(fod, name, None)) for name in names}

    def _field_trace_summary(self, opm: Any, first_order: dict[str, float | None]) -> FieldTraceSummary:
        summary = FieldTraceSummary()
        osp = opm.optical_spec
        fov = osp.field_of_view
        for field_index, _ in enumerate(fov.fields):
            try:
                fld, wvl, foc = osp.lookup_fld_wvl_focus(field_index)
                ray_result = ro_trace.trace_ray(
                    opm,
                    [0.0, 0.0],
                    fld,
                    wvl,
                    output_filter=None,
                    rayerr_filter="summary",
                    use_named_tuples=True,
                )
                if ray_result.err is not None or ray_result.pkg is None:
                    summary.trace_failures.append(f"F{field_index + 1}: {ray_result.err}")
                    continue
                ray = ray_result.pkg.ray
                if len(ray) < 2:
                    summary.trace_failures.append(f"F{field_index + 1}: chief ray returned too few segments")
                    continue
                image_point = ray[-1].p
                image_dir = ray[-2].d
                image_height = math.hypot(_safe_float(image_point[0]) or 0.0, _safe_float(image_point[1]) or 0.0)
                direction_radius = math.hypot(_safe_float(image_dir[0]) or 0.0, _safe_float(image_dir[1]) or 0.0)
                direction_z = abs(_safe_float(image_dir[2]) or 0.0)
                cra_deg = math.degrees(math.atan2(direction_radius, direction_z)) if direction_z > 1.0e-14 else None
                field_label = self._field_label(field_index, fld)
                field_radius = math.hypot(_safe_float(fld.x) or 0.0, _safe_float(fld.y) or 0.0)

                if cra_deg is not None and (summary.max_cra_deg is None or cra_deg > summary.max_cra_deg):
                    summary.max_cra_deg = cra_deg
                    summary.worst_cra_field = field_label

                ideal_image_height = self._ideal_image_height(fov, fld, first_order)
                if ideal_image_height > 1.0e-9:
                    distortion_pct = abs((image_height - ideal_image_height) / ideal_image_height) * 100.0
                    if summary.max_distortion_pct is None or distortion_pct > summary.max_distortion_pct:
                        summary.max_distortion_pct = distortion_pct
                        summary.worst_distortion_field = field_label
                spot_rms_um, sample_count, attempted_count = self._spot_rms_for_field(opm, fld, wvl, foc)
                if attempted_count > 0:
                    throughput = sample_count / attempted_count
                    summary.throughput_fields += 1
                    if (
                        summary.min_pupil_throughput is None
                        or throughput < summary.min_pupil_throughput - 1.0e-9
                        or (
                            abs(throughput - summary.min_pupil_throughput) <= 1.0e-9
                            and field_radius > summary.worst_throughput_field_radius
                        )
                    ):
                        summary.min_pupil_throughput = throughput
                        summary.worst_throughput_field = field_label
                        summary.worst_throughput_field_radius = field_radius
                if spot_rms_um is not None:
                    summary.spot_fields += 1
                    summary.spot_samples += sample_count
                    if summary.max_spot_rms_um is None or spot_rms_um > summary.max_spot_rms_um:
                        summary.max_spot_rms_um = spot_rms_um
                        summary.worst_spot_field = field_label
                else:
                    summary.trace_failures.append(f"{field_label}: spot grid had insufficient traced rays")
                summary.traced_fields += 1
            except Exception as exc:
                summary.trace_failures.append(f"F{field_index + 1}: {type(exc).__name__}: {exc}")
        return summary

    def _tolerance_robustness(self, opm: Any, powered_indices: list[int]) -> ToleranceRobustnessSummary:
        summary = ToleranceRobustnessSummary()
        sm = opm.seq_model
        candidates = sorted(
            [
                idx
                for idx in powered_indices
                if idx < len(sm.ifcs) and abs(_safe_float(sm.ifcs[idx].profile_cv) or 0.0) > 1.0e-8
            ],
            key=lambda idx: abs(_safe_float(sm.ifcs[idx].profile_cv) or 0.0),
            reverse=True,
        )[:4]
        summary.stressed_surfaces = candidates
        if not candidates:
            return summary

        perturbation = 0.005
        for idx in candidates:
            base_cv = _safe_float(sm.ifcs[idx].profile_cv)
            if base_cv is None or abs(base_cv) <= 1.0e-12:
                continue
            for sign in (-1.0, 1.0):
                case_label = f"S{idx} {'+' if sign > 0 else '-'}0.5% curvature"
                summary.attempted_cases += 1
                try:
                    trial = copy.deepcopy(opm)
                    trial.seq_model.ifcs[idx].profile_cv = base_cv * (1.0 + sign * perturbation)
                    trial.update_model()
                    trial_trace = self._field_trace_summary(trial, self._first_order_values(trial))
                    if trial_trace.trace_failures:
                        summary.trace_failures += len(trial_trace.trace_failures)
                    case_score = _tolerance_case_score(trial_trace)
                    if case_score > 0:
                        summary.passed_cases += 1
                    if trial_trace.max_spot_rms_um is not None and (
                        summary.max_spot_rms_um is None or trial_trace.max_spot_rms_um > summary.max_spot_rms_um
                    ):
                        summary.max_spot_rms_um = trial_trace.max_spot_rms_um
                    mtf50 = _mtf50_from_spot_rms_um(trial_trace.max_spot_rms_um)
                    if mtf50 is not None and (summary.min_mtf50_lpmm is None or mtf50 < summary.min_mtf50_lpmm):
                        summary.min_mtf50_lpmm = mtf50
                except Exception:
                    summary.trace_failures += 1
                    case_score = 0.0

                if summary.worst_case_score is None or case_score < summary.worst_case_score:
                    summary.worst_case_score = case_score
                    summary.worst_surface = idx
                    summary.worst_case = case_label

        if summary.attempted_cases > 0:
            summary.score = summary.passed_cases / summary.attempted_cases
        return summary

    def _spot_rms_for_field(self, opm: Any, fld: Any, wvl: float, foc: float) -> tuple[float | None, int, int]:
        samples = _spot_pupil_samples()
        attempted_count = len(samples)
        try:
            ref_sphere, _ = ro_trace.setup_pupil_coords(opm, fld, wvl, foc)
        except Exception:
            return None, 0, 0
        reference_image_pt = np.array(ref_sphere[0])
        points: list[np.ndarray] = []
        for pupil in samples:
            ray_result = ro_trace.trace_ray(
                opm,
                pupil,
                fld,
                wvl,
                output_filter=None,
                rayerr_filter="summary",
                use_named_tuples=True,
                check_apertures=True,
            )
            if ray_result.err is not None or ray_result.pkg is None:
                continue
            ray = ray_result.pkg.ray
            if not ray:
                continue
            last = ray[-1]
            direction_z = _safe_float(last.d[2]) or 0.0
            if abs(direction_z) < 1.0e-12:
                continue
            dist = foc / direction_z
            defocused_pt = np.array(last.p) + dist * np.array(last.d)
            points.append(defocused_pt[:2] - reference_image_pt[:2])
        min_valid_samples = max(12, math.ceil(attempted_count * 0.5))
        if len(points) < min_valid_samples:
            return None, len(points), attempted_count
        arr = np.vstack(points)
        centroid = np.mean(arr, axis=0)
        radial_sq = np.sum((arr - centroid) ** 2, axis=1)
        rms_mm = math.sqrt(float(np.mean(radial_sq)))
        return rms_mm * 1000.0, len(points), attempted_count

    def _ideal_image_height(self, fov: Any, fld: Any, first_order: dict[str, float | None]) -> float:
        field_radius = math.hypot(_safe_float(fld.x) or 0.0, _safe_float(fld.y) or 0.0)
        fov_value = abs(_safe_float(fov.value) or 0.0)
        key = tuple(getattr(fov, "key", ()))
        if key == ("object", "angle"):
            field_angle = field_radius * fov_value if fov.is_relative else field_radius
            efl = abs(first_order.get("efl") or 0.0)
            if efl > 1.0e-12:
                return abs(efl * math.tan(math.radians(field_angle)))
        if key == ("image", "height"):
            return field_radius * fov_value if fov.is_relative else field_radius
        if key == ("object", "height"):
            obj_height = field_radius * fov_value if fov.is_relative else field_radius
            magnification = abs(first_order.get("m") or 0.0)
            if magnification > 1.0e-12:
                return obj_height * magnification
        return abs(first_order.get("img_ht") or 0.0) * self._field_fraction(fov, fld)

    def _field_fraction(self, fov: Any, fld: Any) -> float:
        field_radius = math.hypot(_safe_float(fld.x) or 0.0, _safe_float(fld.y) or 0.0)
        if fov.is_relative:
            return field_radius
        fov_value = abs(_safe_float(fov.value) or 0.0)
        if fov_value > 1.0e-12:
            return field_radius / fov_value
        return field_radius

    def _field_label(self, field_index: int, fld: Any) -> str:
        field_radius = math.hypot(_safe_float(fld.x) or 0.0, _safe_float(fld.y) or 0.0)
        return f"F{field_index + 1} ({field_radius:.3g})"

    def _sensor_assumptions(self, session: ModelSession) -> SensorAssumptionDTO:
        values: dict[str, Any] = {
            "name": "Assumed CMOS reference",
            "pixel_pitch_um": 2.0,
            "quantum_efficiency": 0.55,
            "read_noise_e": 1.2,
            "dark_noise_e": 0.5,
            "full_well_e": 18000.0,
            "exposure_ms": 10.0,
            "scene_luminance_cd_m2": 50.0,
            "optical_transmission": 0.85,
            "microlens_cra_limit_deg": 24.0,
            "reference_wavelength_nm": self._reference_wavelength_nm(session.opt_model),
        }
        values.update(session.sensor_overrides)
        return SensorAssumptionDTO(**values)

    def _reference_wavelength_nm(self, opm: Any) -> float:
        try:
            _, wvl, _ = opm.optical_spec.lookup_fld_wvl_focus(0)
            value = _safe_float(wvl)
            if value is not None and value > 0:
                return value if value > 10 else value * 1000.0
        except Exception:
            pass
        return 550.0

    def _sensor_snr_estimate(
        self,
        sensor: SensorAssumptionDTO,
        fno: float | None,
        pupil_throughput: float,
        cra_deg: float,
    ) -> SensorSnrSummary:
        if fno is None or fno <= 0:
            return SensorSnrSummary()

        wavelength_m = sensor.reference_wavelength_nm * 1.0e-9
        photon_energy_j = 6.62607015e-34 * 299792458.0 / wavelength_m
        exposure_s = sensor.exposure_ms / 1000.0
        pixel_area_m2 = (sensor.pixel_pitch_um * 1.0e-6) ** 2
        cos_cra = max(0.0, math.cos(math.radians(cra_deg)))
        natural_cra_efficiency = cos_cra**4
        cra_over_limit = max(0.0, abs(cra_deg) - sensor.microlens_cra_limit_deg)
        microlens_efficiency = math.exp(-((cra_over_limit / 8.0) ** 2)) if cra_over_limit > 0 else 1.0

        image_lux = (
            sensor.scene_luminance_cd_m2
            * math.pi
            / (4.0 * fno * fno)
            * sensor.optical_transmission
            * _clamp(pupil_throughput, 0.0, 1.0)
            * natural_cra_efficiency
            * microlens_efficiency
        )
        image_irradiance_w_m2 = image_lux / 683.0
        signal_e_uncapped = (
            image_irradiance_w_m2
            * pixel_area_m2
            * exposure_s
            / photon_energy_j
            * sensor.quantum_efficiency
        )
        signal_e = min(max(0.0, signal_e_uncapped), sensor.full_well_e)
        noise_e = math.sqrt(signal_e + sensor.read_noise_e**2 + sensor.dark_noise_e**2)
        snr_linear = signal_e / noise_e if noise_e > 0 else 0.0
        snr_db = 20.0 * math.log10(snr_linear) if snr_linear > 1.0e-12 else -120.0
        return SensorSnrSummary(
            snr_db=snr_db,
            signal_e=signal_e,
            saturation_pct=_clamp(signal_e_uncapped / sensor.full_well_e * 100.0, 0.0, 999.0),
            image_irradiance_w_m2=image_irradiance_w_m2,
            microlens_efficiency=microlens_efficiency,
        )

    def _cockpit_summary(
        self, session: ModelSession, first_order: dict[str, float | None], sensor: SensorAssumptionDTO
    ) -> tuple[CockpitCountsDTO, list[CockpitMetricDTO], list[CockpitRiskDTO]]:
        opm = session.opt_model
        sm = opm.seq_model
        osp = opm.optical_spec
        warnings = session.warnings
        diagnostic_warnings = [warning for warning in warnings if not _is_info_warning(warning)]
        errors = session.errors
        fields = osp.field_of_view.fields
        wavelengths = [w for w in osp.spectral_region.wavelengths if _safe_float(w) is not None]
        real_surface_indices = [idx for idx, _ in enumerate(sm.ifcs) if idx not in (0, sm.get_num_surfaces() - 1)]
        powered_indices = [
            idx for idx, ifc in enumerate(sm.ifcs) if idx not in (0, sm.get_num_surfaces() - 1) and abs(_safe_float(ifc.profile_cv) or 0.0) > 1.0e-6
        ]
        semi_diameters = [
            value
            for idx in real_surface_indices
            for value in [_safe_float(sm.ifcs[idx].surface_od()) or _safe_float(sm.ifcs[idx].max_aperture) or 0.0]
            if 0.0 < value < 1.0e6
        ]
        max_semi = max([1.0, *semi_diameters])
        max_field = max([0.0, *[math.hypot(_safe_float(f.x) or 0.0, _safe_float(f.y) or 0.0) for f in fields]])
        curvature_load = sum(min(2.4, abs(_safe_float(sm.ifcs[idx].profile_cv) or 0.0) * 35.0) for idx in powered_indices)
        complexity = (
            len(powered_indices)
            + curvature_load * 0.3
            + max(0, len(fields) - 3) * 0.35
            + max(0, len(wavelengths) - 3) * 0.25
        )
        warn_count = len(diagnostic_warnings)
        fail_count = len(errors)

        efl = first_order.get("efl")
        fno = first_order.get("fno")
        obj_ang = first_order.get("obj_ang") or _safe_float(osp.field_of_view.value)
        field_trace = self._field_trace_summary(opm, first_order)
        tolerance = self._tolerance_robustness(opm, powered_indices)
        trace_warn_count = warn_count + len(field_trace.trace_failures)
        trace_score = 0.0 if fail_count else 0.68 if trace_warn_count else 1.0
        trace_status = _status_from_count(fail_count, trace_warn_count)

        mtf_proxy = _clamp(0.9 - complexity * 0.038 - warn_count * 0.025 - fail_count * 0.16, 0.08, 0.96)
        illumination_proxy = _clamp(1 - max_field / max(max_semi * 3.6, 1) - max(0, len(powered_indices) - 6) * 0.035, 0.22, 0.98)
        distortion_proxy = _clamp(max_field * 0.16 + curvature_load * 0.085 + warn_count * 0.22, 0.05, 5.5)
        cra_proxy = _clamp((max_field / max(max_semi, 1)) * 18 + len(powered_indices) * 1.1 + (sm.stop_surface or 0) * 0.32, 4, 35)
        distortion_value = field_trace.max_distortion_pct if field_trace.max_distortion_pct is not None else distortion_proxy
        cra_value = field_trace.max_cra_deg if field_trace.max_cra_deg is not None else cra_proxy
        spot_rms_value = field_trace.max_spot_rms_um
        mtf50_lpmm = _mtf50_from_spot_rms_um(spot_rms_value)
        illumination_value = (
            field_trace.min_pupil_throughput
            if field_trace.min_pupil_throughput is not None
            else illumination_proxy
        )
        distortion_source = "computed" if field_trace.max_distortion_pct is not None else "proxy"
        cra_source = "computed" if field_trace.max_cra_deg is not None else "proxy"
        mtf_source = "computed" if mtf50_lpmm is not None else "proxy"
        illumination_source = "computed" if field_trace.min_pupil_throughput is not None else "proxy"
        illumination_warn_threshold = 0.85 if illumination_source == "computed" else 0.62
        illumination_fail_threshold = 0.70 if illumination_source == "computed" else 0.50
        yield_proxy = _clamp(96 - complexity * 2.4 - distortion_value * 1.7 - warn_count * 4.5 - fail_count * 20, 35, 98.5)
        robustness_value = tolerance.score * 100.0 if tolerance.score is not None else yield_proxy
        robustness_source = "computed" if tolerance.score is not None else "proxy"
        snr_proxy = _clamp(44 - max_field * 0.55 - max(0, cra_value - 18) * 0.42 - warn_count * 1.4 - fail_count * 8, 16, 48)
        sensor_snr = self._sensor_snr_estimate(sensor, fno, illumination_value, cra_value)
        snr_value = sensor_snr.snr_db if sensor_snr.snr_db is not None else snr_proxy
        snr_source = "assumption" if sensor_snr.snr_db is not None else "proxy"

        metrics = [
            CockpitMetricDTO(
                key="trace",
                label="Trace Health",
                value=str(fail_count if fail_count else trace_warn_count),
                unit="errors" if fail_count else "warnings",
                target="0 errors",
                status=trace_status,
                score=trace_score,
                source="diagnostic",
                note="Counts update_model, import, field-trace, and analysis diagnostics.",
            ),
            CockpitMetricDTO(
                key="spot",
                label="Spot RMS",
                value=_format_metric(spot_rms_value),
                unit="um",
                target="<= 10 um",
                status=_status_for(spot_rms_value if spot_rms_value is not None else 9999.0, 10.0, 25.0, "low-good"),
                score=_clamp(1 - (spot_rms_value or 100.0) / 100.0, 0, 1),
                source="computed" if spot_rms_value is not None else "unsupported",
                note=(
                    f"Worst-field RMS spot radius from {field_trace.spot_samples} traced pupil samples across {field_trace.spot_fields} fields."
                    if spot_rms_value is not None
                    else "Spot grid could not be traced for enough valid pupil samples."
                ),
            ),
            CockpitMetricDTO(
                key="mtf",
                label="Geom. MTF50" if mtf_source == "computed" else "MTF50 Proxy",
                value=_format_metric(mtf50_lpmm) if mtf50_lpmm is not None else f"{mtf_proxy:.2f}",
                unit="lp/mm" if mtf_source == "computed" else None,
                target=">= 20 lp/mm" if mtf_source == "computed" else ">= 0.30",
                status=(
                    _status_for(mtf50_lpmm, 20.0, 10.0, "high-good")
                    if mtf50_lpmm is not None
                    else _status_for(mtf_proxy, 0.42, 0.30, "high-good")
                ),
                score=_clamp((mtf50_lpmm or 0.0) / 30.0, 0, 1) if mtf50_lpmm is not None else mtf_proxy,
                source=mtf_source,
                note=(
                    f"Gaussian approximation from worst-field RMS spot radius at {field_trace.worst_spot_field}; geometric estimate, not diffraction or polychromatic MTF."
                    if mtf50_lpmm is not None
                    else "Proxy estimated from optical complexity because a dedicated MTF evaluator did not return a computed value."
                ),
            ),
            CockpitMetricDTO(
                key="illumination",
                label="Pupil Throughput" if illumination_source == "computed" else "Rel. Illum. Proxy",
                value=f"{illumination_value:.2f}",
                target=">= 0.85" if illumination_source == "computed" else ">= 0.50",
                status=_status_for(
                    illumination_value,
                    illumination_warn_threshold,
                    illumination_fail_threshold,
                    "high-good",
                ),
                score=illumination_value,
                source=illumination_source,
                note=(
                    "Minimum valid traced pupil-sample fraction across "
                    f"{field_trace.throughput_fields} fields; geometric vignetting only, not radiometric relative illumination."
                    if illumination_source == "computed"
                    else "Estimated from field height, aperture, and surface complexity."
                ),
            ),
            CockpitMetricDTO(
                key="distortion",
                label="Distortion" if distortion_source == "computed" else "Distortion Proxy",
                value=f"{distortion_value:.2f}",
                unit="%",
                target="<= 2.00%",
                status=_status_for(distortion_value, 2.0, 3.25, "low-good"),
                score=_clamp(1 - distortion_value / 5.5, 0, 1),
                source=distortion_source,
                note=(
                    f"Max absolute chief-ray distortion from {field_trace.traced_fields} traced fields."
                    if distortion_source == "computed"
                    else "Proxy estimated from field height and curvature because distortion extraction did not return a computed value."
                ),
            ),
            CockpitMetricDTO(
                key="cra",
                label="CRA Max" if cra_source == "computed" else "CRA Proxy",
                value=f"{cra_value:.1f}",
                unit="deg",
                target="<= 25 deg",
                status=_status_for(cra_value, 25.0, 30.0, "low-good"),
                score=_clamp(1 - cra_value / 35.0, 0, 1),
                source=cra_source,
                note=(
                    f"Max image-space chief ray angle from {field_trace.traced_fields} traced fields."
                    if cra_source == "computed"
                    else "Estimated chief-ray-angle pressure; not a sensor microlens solver."
                ),
            ),
            CockpitMetricDTO(
                key="yield",
                label="Tol. Robustness" if robustness_source == "computed" else "Tol. Proxy",
                value=f"{robustness_value:.1f}",
                unit="%",
                target=">= 86%" if robustness_source == "computed" else ">= 80%",
                status=_status_for(
                    robustness_value,
                    86.0 if robustness_source == "computed" else 86.0,
                    70.0 if robustness_source == "computed" else 80.0,
                    "high-good",
                ),
                score=robustness_value / 100.0,
                source=robustness_source,
                note=(
                    f"Deterministic +/-0.5% curvature stress on {len(tolerance.stressed_surfaces)} powered surfaces; {tolerance.passed_cases}/{tolerance.attempted_cases} cases avoided fail thresholds. Not statistical manufacturing yield."
                    if robustness_source == "computed"
                    else "Proxy estimated from complexity and diagnostics because tolerance stress analysis did not return computed cases."
                ),
            ),
            CockpitMetricDTO(
                key="snr",
                label="Assumed Corner SNR" if snr_source == "assumption" else "Corner SNR Proxy",
                value=f"{snr_value:.1f}",
                unit="dB",
                target=">= 30 dB" if snr_source == "assumption" else ">= 40 dB",
                status=(
                    _status_for(snr_value, 30.0, 20.0, "high-good")
                    if snr_source == "assumption"
                    else _status_for(snr_value, 40.0, 32.0, "high-good")
                ),
                score=_clamp((snr_value - 10.0) / 25.0, 0, 1) if snr_source == "assumption" else _clamp(snr_value / 48.0, 0, 1),
                source=snr_source,
                note=(
                    f"Photon/read-noise estimate from explicit default assumptions: {sensor.scene_luminance_cd_m2:.0f} cd/m^2 scene, {sensor.exposure_ms:.1f} ms, {sensor.pixel_pitch_um:.2f} um pixel, QE {sensor.quantum_efficiency:.2f}, signal {_format_metric(sensor_snr.signal_e)} e-. Not measured scene SNR."
                    if snr_source == "assumption"
                    else "Proxy estimated from field and CRA pressure because explicit sensor SNR calculation did not return a value."
                ),
            ),
            CockpitMetricDTO(
                key="efl",
                label="EFL",
                value=_format_metric(efl),
                unit="mm",
                target="paraxial",
                status="pass" if efl is not None else "warn",
                score=1.0 if efl is not None else 0.0,
                source="computed" if efl is not None else "unsupported",
                note="RayOptics paraxial first-order value.",
            ),
            CockpitMetricDTO(
                key="fno",
                label="F/#",
                value=_format_metric(fno),
                target="paraxial",
                status="pass" if fno and fno > 0 else "warn",
                score=1.0 if fno and fno > 0 else 0.0,
                source="computed" if fno is not None else "unsupported",
                note="RayOptics paraxial first-order value.",
            ),
        ]

        selected_sensitivity = (
            tolerance.worst_surface
            if tolerance.worst_surface is not None
            else powered_indices[min(len(powered_indices) - 1, 2)] if powered_indices else None
        )
        worst_field_value = (
            field_trace.worst_throughput_field
            if illumination_source == "computed" and field_trace.worst_throughput_field
            else f"{max_field:.2f}" if max_field > 0 else "On-axis"
        )
        worst_field_detail = (
            f"min pupil throughput {illumination_value:.2f}; {len(fields) or 1} fields, {len(wavelengths) or 1} wavelengths"
            if illumination_source == "computed"
            else f"{len(fields) or 1} fields, {len(wavelengths) or 1} wavelengths"
        )
        risks = [
            CockpitRiskDTO(
                label="Worst Field",
                value=worst_field_value,
                status=_status_for(
                    illumination_value,
                    illumination_warn_threshold,
                    illumination_fail_threshold,
                    "high-good",
                ),
                detail=worst_field_detail,
                source="computed" if illumination_source == "computed" else "model",
            ),
            CockpitRiskDTO(
                label="Surface Sensitivity",
                value=f"S{selected_sensitivity}" if selected_sensitivity is not None else "n/a",
                status=_status_for(
                    robustness_value,
                    86.0 if robustness_source == "computed" else 86.0,
                    70.0 if robustness_source == "computed" else 80.0,
                    "high-good",
                ),
                detail=(
                    f"{tolerance.worst_case}; {tolerance.passed_cases}/{tolerance.attempted_cases} stress cases survived"
                    if robustness_source == "computed" and tolerance.worst_case
                    else f"{len(powered_indices)} powered surfaces"
                ),
                source=robustness_source,
            ),
            CockpitRiskDTO(
                label="Spot Quality",
                value=f"{_format_metric(spot_rms_value)} um" if spot_rms_value is not None else "n/a",
                status=_status_for(spot_rms_value if spot_rms_value is not None else 9999.0, 10.0, 25.0, "low-good"),
                detail=field_trace.worst_spot_field or "Worst spot field unavailable",
                source="computed" if spot_rms_value is not None else "unsupported",
            ),
            CockpitRiskDTO(
                label="Sensor Coupling",
                value=f"{cra_value:.1f} deg",
                status=_status_for(cra_value, 25.0, 30.0, "low-good"),
                detail=(field_trace.worst_cra_field or "CRA / microlens margin proxy"),
                source=cra_source,
            ),
            CockpitRiskDTO(
                label="Scene Risk",
                value="High" if snr_value < 20 else "Medium" if snr_value < 30 else "Low",
                status=(
                    _status_for(snr_value, 30.0, 20.0, "high-good")
                    if snr_source == "assumption"
                    else _status_for(snr_value, 40.0, 32.0, "high-good")
                ),
                detail=(
                    f"{_format_metric(sensor_snr.signal_e)} e- signal, {_format_metric(sensor_snr.saturation_pct)}% full well under assumed scene"
                    if snr_source == "assumption"
                    else "Generic scene SNR proxy"
                ),
                source=snr_source,
            ),
            CockpitRiskDTO(
                label="First Order",
                value=f"EFL {_format_metric(efl)} mm" if efl is not None else "No paraxial data",
                status="pass" if efl is not None and (obj_ang is not None or fno is not None) else "warn",
                detail=f"F/# {_format_metric(fno)}" if fno is not None else "Paraxial F/# unavailable",
                source="computed" if efl is not None else "unsupported",
            ),
        ]

        counts = CockpitCountsDTO(
            surface_count=sm.get_num_surfaces(),
            powered_surfaces=len(powered_indices),
            wavelength_count=len(wavelengths),
            field_count=len(fields),
            complexity=complexity,
            max_semi_diameter=max_semi,
            max_field=max_field,
        )
        return counts, metrics, risks

    def _register(self, opm: Any, filename: str | None, warnings: list[str] | None = None) -> ModelResponse:
        model_id = uuid.uuid4().hex
        session = ModelSession(
            model_id=model_id,
            opt_model=opm,
            filename=filename,
            dirty=False,
            warnings=warnings or [],
            errors=[],
            last_valid_model=copy.deepcopy(opm),
        )
        self.sessions[model_id] = session
        return self.response(model_id)

    def _create_default_model(self, payload: NewModelRequest) -> Any:
        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        osp.pupil = PupilSpec(osp, key=("object", "epd"), value=payload.epd)
        osp.field_of_view = FieldSpec(
            osp,
            key=("object", "angle"),
            value=payload.fov,
            flds=[0.0, 0.707, 1.0],
            is_relative=True,
        )
        osp.spectral_region = WvlSpec([("F", 0.5), (587.5618, 1.0), ("C", 0.5)], ref_wl=1)
        opm.radius_mode = True
        sm.gaps[0].thi = 1.0e10
        radius = max(abs(payload.efl), 1.0)
        image_distance = max(payload.efl, 1.0)
        sm.add_surface([radius, 5.0, "N-BK7", "Schott", payload.epd / 2])
        sm.add_surface([-radius, image_distance, "air", payload.epd / 2])
        sm.set_stop(1)
        opm.update_model()
        return opm

    def _commit(
        self,
        session: ModelSession,
        before: Any,
        dirty: bool,
        warnings: list[str] | None = None,
        record_history: bool = True,
    ) -> ModelResponse:
        try:
            self._repair_restored_model(session.opt_model)
            session.opt_model.update_model()
            if record_history:
                session.undo_stack.append(copy.deepcopy(before))
                session.undo_stack = session.undo_stack[-40:]
                session.redo_stack.clear()
            session.last_valid_model = copy.deepcopy(session.opt_model)
            session.errors = []
            session.warnings = warnings or []
            session.dirty = dirty
        except Exception as exc:
            session.opt_model = before
            session.errors = [f"Update failed; restored previous valid model: {type(exc).__name__}: {exc}"]
            session.warnings = warnings or []
        session.last_updated_at = _now()
        return self.response(session.model_id)

    def _response(self, session: ModelSession) -> ModelResponse:
        return ModelResponse(
            model=self._model_dto(session),
            warnings=session.warnings,
            errors=session.errors,
            dirty=session.dirty,
            can_undo=bool(session.undo_stack),
            can_redo=bool(session.redo_stack),
            last_updated_at=session.last_updated_at.isoformat(),
            workbench=session.workbench_metadata or None,
        )

    def _model_dto(self, session: ModelSession) -> ModelDTO:
        opm = session.opt_model
        sm = opm.seq_model
        surfaces = []
        for idx, ifc in enumerate(sm.ifcs):
            gap = sm.gaps[idx] if idx < len(sm.gaps) else None
            cv = _safe_float(ifc.profile_cv) or 0.0
            surfaces.append(
                SurfaceDTO(
                    index=idx,
                    label=ifc.label or "",
                    type=ifc.interface_type(),
                    radius=_radius_from_cv(cv),
                    curvature=cv,
                    thickness=_safe_float(gap.thi) if gap else None,
                    glass=gap.medium.name() if gap and gap.medium else "",
                    catalog=gap.medium.catalog_name() if gap and gap.medium else "",
                    semi_diameter=_safe_float(ifc.surface_od()) or _safe_float(ifc.max_aperture) or 0.0,
                    conic=self._conic_value(ifc),
                    mode=ifc.interact_mode,
                    is_stop=idx == sm.stop_surface,
                    variable=self._variable_label(session, idx),
                )
            )
        return ModelDTO(
            id=session.model_id,
            name=Path(session.filename).stem if session.filename else opm.name(),
            filename=session.filename,
            radius_mode=True,
            update_mode=session.update_mode,
            stop_surface=sm.stop_surface,
            surfaces=surfaces,
            system=self._system_dto(opm),
        )

    def _system_dto(self, opm: Any) -> SystemDTO:
        osp = opm.optical_spec
        pupil = osp.pupil
        fov = osp.field_of_view
        wvls = osp.spectral_region
        focus = osp.defocus
        return SystemDTO(
            aperture={"key": list(pupil.key), "value": _safe_float(pupil.value)},
            field={
                "key": list(fov.key),
                "value": _safe_float(fov.value),
                "fields": [{"x": _safe_float(f.x), "y": _safe_float(f.y), "weight": _safe_float(getattr(f, "wt", 1.0))} for f in fov.fields],
                "isRelative": bool(fov.is_relative),
            },
            wavelengths={
                "values": [_safe_float(w) for w in wvls.wavelengths],
                "weights": [_safe_float(w) for w in wvls.spectral_wts],
                "reference": int(wvls.reference_wvl),
            },
            focus={
                "focusShift": _safe_float(getattr(focus, "focus_shift", 0.0)),
                "defocusRange": _safe_float(getattr(focus, "defocus_range", 0.0)),
            },
        )

    def _surface_z_positions(self, sm: Any) -> list[float]:
        count = sm.get_num_surfaces()
        z = [0.0] * count
        if count <= 2:
            return z
        z[1] = 0.0
        for idx in range(2, count):
            prev_gap = sm.gaps[idx - 1] if idx - 1 < len(sm.gaps) else None
            thi = _safe_float(prev_gap.thi) if prev_gap else 0.0
            if thi is None or abs(thi) > 1.0e6:
                thi = 0.0
            z[idx] = z[idx - 1] + thi
        return z

    def _layout_rays(self, opm: Any, z_by_surface: list[float], warnings: list[str]) -> list[list[dict[str, float]]]:
        sm = opm.seq_model
        osp = opm.optical_spec
        pdata = opm["analysis_results"].get("parax_data")
        if pdata is None:
            return []
        fod = pdata.fod
        rays: list[list[dict[str, float]]] = []
        try:
            fields = osp.field_of_view.fields
            field_indices = [0]
            if len(fields) > 1:
                field_indices.append(len(fields) - 1)
            for fi in field_indices:
                fld, wvl, _ = osp.lookup_fld_wvl_focus(fi)
                pt0, _ = osp.obj_coords(fld)
                for py in (-1.0, 0.0, 1.0):
                    vig = fld.apply_vignetting([0.0, py])
                    pt1 = np.array([0.0, fod.enp_radius * vig[1], fod.obj_dist + fod.enp_dist])
                    dir0 = normalize(pt1 - pt0)
                    ray_pkg = rt.trace(sm, pt0, dir0, wvl)
                    ray = ray_pkg[mc.ray]
                    points = []
                    for idx, seg in enumerate(ray):
                        if idx == 0:
                            continue
                        p = seg[mc.p]
                        z = z_by_surface[idx] if idx < len(z_by_surface) else z_by_surface[-1]
                        points.append({"z": _safe_float(z) or 0.0, "y": _safe_float(p[1]) or 0.0})
                    if len(points) > 1:
                        rays.append(points)
        except Exception as exc:
            warnings.append(f"Layout ray trace failed: {type(exc).__name__}: {exc}")
        return rays[:12]

    def _make_medium(self, glass: str, catalog: str) -> Any:
        if glass.strip() == "" or glass.strip().lower() == "air":
            return om.Air()
        return gfact.create_glass(glass.strip(), catalog.strip() or "Schott")

    def _repair_restored_model(self, opm: Any) -> list[str]:
        warnings: list[str] = []
        osp = opm.optical_spec
        if not isinstance(getattr(osp, "pupil", None), PupilSpec):
            pass
        focus = osp.defocus
        repaired = False
        if not hasattr(focus, "focus_shift") and hasattr(focus, "infocus"):
            focus.focus_shift = focus.infocus
            repaired = True
        if not hasattr(focus, "defocus_range") and hasattr(focus, "defocus"):
            focus.defocus_range = focus.defocus
            repaired = True
        if repaired:
            warnings.append("Repaired old FocusRange fields: infocus/defocus -> focus_shift/defocus_range.")
        return warnings

    def _assert_surface_index(self, sm: Any, index: int, allow_object_image: bool) -> None:
        if index < 0 or index >= sm.get_num_surfaces():
            raise ValueError("Surface index out of range.")
        if not allow_object_image and (index == 0 or index == sm.get_num_surfaces() - 1):
            raise ValueError("Object and image surfaces cannot be deleted.")

    def _conic_value(self, ifc: Any) -> float | None:
        if hasattr(ifc.profile, "cc"):
            return _safe_float(ifc.profile.cc)
        if hasattr(ifc.profile, "ec"):
            return _safe_float(ifc.profile.ec)
        return None

    def _set_surface_variables(self, session: ModelSession, surface_index: int, raw: str) -> None:
        tokens = self._parse_variable_tokens(raw)
        if tokens:
            session.surface_variables[surface_index] = tokens
        else:
            session.surface_variables.pop(surface_index, None)

    def _parse_variable_tokens(self, raw: str) -> set[str]:
        tokens: set[str] = set()
        for item in raw.replace("/", ",").replace("|", ",").replace(";", ",").split(","):
            normalized = item.strip().upper().replace(" ", "")
            if not normalized:
                continue
            token = VARIABLE_TOKEN_ALIASES.get(normalized)
            if token is None:
                raise ValueError(f"Unsupported variable token: {item.strip()}. Use R, T, SD, or K.")
            tokens.add(token)
        return tokens

    def _variable_label(self, session: ModelSession, surface_index: int) -> str:
        tokens = session.surface_variables.get(surface_index, set())
        return ",".join(token for token in VARIABLE_TOKEN_ORDER if token in tokens)

    def _variable_warning(self, surface_index: int) -> str:
        return f"Updated S{surface_index} variable flags. These are workbench metadata only; the RayOptics optimizer is not connected yet."

    def _quick_optimize_response(self, session: ModelSession, result: QuickOptimizeResultDTO) -> QuickOptimizeResponse:
        response = self.response(session.model_id)
        return QuickOptimizeResponse(
            model=response.model,
            warnings=response.warnings,
            errors=response.errors,
            dirty=response.dirty,
            can_undo=response.can_undo,
            can_redo=response.can_redo,
            last_updated_at=response.last_updated_at,
            result=result,
        )

    def _tolerance_sweep_result(self, session: ModelSession, payload: ToleranceSweepRequest) -> ToleranceSweepResultDTO:
        opm = session.opt_model
        first_order = self._first_order_values(opm)
        baseline_trace = self._field_trace_summary(opm, first_order)
        baseline_score = _tolerance_case_score(baseline_trace)
        candidates = self._tolerance_sweep_variables(session, payload)
        cases: list[ToleranceSweepCaseDTO] = []
        warnings: list[str] = []
        perturbation = payload.perturbation_pct / 100.0

        if not candidates:
            warnings.append("No tolerance sweep candidates were available for the selected scope.")

        for variable in candidates:
            before = self._read_optimization_variable(opm, variable)
            if before is None:
                continue
            delta = self._tolerance_delta(variable["token"], before, perturbation)
            if delta is None or delta <= 0:
                continue
            for sign in (-1.0, 1.0):
                trial = copy.deepcopy(opm)
                after = before + sign * delta
                perturbation_pct = ((after - before) / before * 100.0) if abs(before) > 1.0e-12 else sign * payload.perturbation_pct
                label = f'{variable["label"]} {"+" if sign > 0 else "-"}{abs(perturbation_pct):.3g}%'
                try:
                    if not self._write_optimization_variable(trial, variable, after):
                        raise ValueError("perturbed value was outside workbench bounds")
                    trial.update_model()
                    trace = self._field_trace_summary(trial, self._first_order_values(trial))
                    score = _tolerance_case_score(trace)
                    status = _tolerance_status(score)
                    cases.append(
                        ToleranceSweepCaseDTO(
                            label=label,
                            surface_index=variable["surface_index"],
                            token=variable["token"],
                            perturbation_pct=perturbation_pct,
                            before=before,
                            after=after,
                            score=score,
                            status=status,
                            spot_rms_um=trace.max_spot_rms_um,
                            mtf50_lpmm=_mtf50_from_spot_rms_um(trace.max_spot_rms_um),
                            throughput=trace.min_pupil_throughput,
                            distortion_pct=trace.max_distortion_pct,
                            cra_deg=trace.max_cra_deg,
                            trace_failures=trace.trace_failures[:6],
                        )
                    )
                except Exception as exc:
                    cases.append(
                        ToleranceSweepCaseDTO(
                            label=label,
                            surface_index=variable["surface_index"],
                            token=variable["token"],
                            perturbation_pct=perturbation_pct,
                            before=before,
                            after=after,
                            score=0.0,
                            status="fail",
                            spot_rms_um=None,
                            mtf50_lpmm=None,
                            throughput=None,
                            distortion_pct=None,
                            cra_deg=None,
                            trace_failures=[f"{type(exc).__name__}: {exc}"],
                        )
                    )

        passed = sum(1 for case in cases if case.status == "pass")
        warned = sum(1 for case in cases if case.status == "warn")
        failed = sum(1 for case in cases if case.status == "fail")
        worst = min(cases, key=lambda case: case.score) if cases else None
        if failed:
            status: str = "fail"
        elif warned:
            status = "warn"
        else:
            status = "pass" if cases else "warn"
        if cases:
            warnings.append(
                f"Quick Tolerance Sweep is deterministic perturbation analysis ({payload.perturbation_pct:.3g}%); it is not statistical manufacturing yield."
            )

        return ToleranceSweepResultDTO(
            status=status,
            scope=payload.scope,
            perturbation_pct=payload.perturbation_pct,
            baseline_score=baseline_score,
            attempted_cases=len(cases),
            passed_cases=passed,
            warned_cases=warned,
            failed_cases=failed,
            worst_case=worst.label if worst else None,
            worst_score=worst.score if worst else None,
            cases=sorted(cases, key=lambda case: case.score),
            warnings=warnings,
        )

    def _tolerance_sweep_variables(self, session: ModelSession, payload: ToleranceSweepRequest) -> list[dict[str, Any]]:
        if payload.scope == "variables":
            return self._optimization_variables(session)[: payload.max_surfaces]
        sm = session.opt_model.seq_model
        candidates = sorted(
            [
                idx
                for idx, ifc in enumerate(sm.ifcs)
                if idx not in (0, sm.get_num_surfaces() - 1) and abs(_safe_float(ifc.profile_cv) or 0.0) > 1.0e-8
            ],
            key=lambda idx: abs(_safe_float(sm.ifcs[idx].profile_cv) or 0.0),
            reverse=True,
        )[: payload.max_surfaces]
        return [{"surface_index": idx, "token": "R", "label": f"S{idx}:R"} for idx in candidates]

    def _tolerance_delta(self, token: str, value: float, perturbation: float) -> float | None:
        if token == "R":
            return max(abs(value) * perturbation, 0.00005)
        if token == "T":
            return max(abs(value) * perturbation, 0.005)
        if token == "SD":
            return max(abs(value) * perturbation, 0.005)
        if token == "K":
            return max(abs(value) * perturbation, 0.001)
        return None

    def _optimization_variables(self, session: ModelSession) -> list[dict[str, Any]]:
        sm = session.opt_model.seq_model
        variables: list[dict[str, Any]] = []
        for surface_index in sorted(session.surface_variables):
            if surface_index <= 0 or surface_index >= sm.get_num_surfaces() - 1:
                continue
            tokens = session.surface_variables.get(surface_index, set())
            for token in VARIABLE_TOKEN_ORDER:
                if token not in tokens:
                    continue
                if token == "T" and surface_index >= len(sm.gaps):
                    continue
                if token == "K" and not self._surface_has_conic(sm.ifcs[surface_index]):
                    continue
                variables.append({"surface_index": surface_index, "token": token, "label": f"S{surface_index}:{token}"})
        return variables

    def _read_optimization_variable(self, opm: Any, variable: dict[str, Any]) -> float | None:
        surface_index = variable["surface_index"]
        token = variable["token"]
        sm = opm.seq_model
        if surface_index <= 0 or surface_index >= sm.get_num_surfaces() - 1:
            return None
        if token == "R":
            return _safe_float(sm.ifcs[surface_index].profile_cv)
        if token == "T" and surface_index < len(sm.gaps):
            return _safe_float(sm.gaps[surface_index].thi)
        if token == "SD":
            return _safe_float(sm.ifcs[surface_index].surface_od()) or _safe_float(sm.ifcs[surface_index].max_aperture)
        if token == "K":
            return self._conic_value(sm.ifcs[surface_index])
        return None

    def _write_optimization_variable(self, opm: Any, variable: dict[str, Any], value: float) -> bool:
        if not math.isfinite(value):
            return False
        surface_index = variable["surface_index"]
        token = variable["token"]
        sm = opm.seq_model
        if surface_index <= 0 or surface_index >= sm.get_num_surfaces() - 1:
            return False
        if token == "R":
            if abs(value) > 2.0:
                return False
            sm.ifcs[surface_index].profile_cv = value
            return True
        if token == "T" and surface_index < len(sm.gaps):
            if abs(value) > 1.0e6 or abs(value) < 1.0e-8:
                return False
            sm.gaps[surface_index].thi = value
            return True
        if token == "SD":
            if value <= 1.0e-8 or value > 1.0e5:
                return False
            sm.ifcs[surface_index].max_aperture = abs(value)
            return True
        if token == "K":
            if abs(value) > 1.0e4:
                return False
            if hasattr(sm.ifcs[surface_index].profile, "cc"):
                sm.ifcs[surface_index].profile.cc = value
                return True
            if hasattr(sm.ifcs[surface_index].profile, "ec"):
                sm.ifcs[surface_index].profile.ec = value
                return True
        return False

    def _optimization_step(self, opm: Any, variable: dict[str, Any], step_scale: float) -> float | None:
        current_value = self._read_optimization_variable(opm, variable)
        if current_value is None:
            return None
        token = variable["token"]
        if token == "R":
            return max(abs(current_value) * 0.02, 0.00025) * step_scale
        if token == "T":
            return max(abs(current_value) * 0.02, 0.05) * step_scale
        if token == "SD":
            return max(abs(current_value) * 0.03, 0.05) * step_scale
        if token == "K":
            return max(abs(current_value) * 0.10, 0.05) * step_scale
        return None

    def _quick_optimize_score(self, opm: Any, weights: dict[str, float], targets: dict[str, float]) -> tuple[float, str]:
        try:
            self._repair_restored_model(opm)
            opm.update_model()
            first_order = self._first_order_values(opm)
            field_trace = self._field_trace_summary(opm, first_order)
            spot = field_trace.max_spot_rms_um
            distortion = field_trace.max_distortion_pct
            throughput = field_trace.min_pupil_throughput
            cra = field_trace.max_cra_deg

            spot_target = max(0.1, targets.get("spot", 25.0))
            distortion_target = max(0.01, targets.get("distortion", 2.0))
            throughput_target = _clamp(targets.get("throughput", 0.85), 0.05, 1.0)
            cra_target = max(0.1, targets.get("cra", 25.0))

            spot_component = 1.0 / (1.0 + (spot if spot is not None else spot_target * 20.0) / spot_target)
            distortion_component = 1.0 / (1.0 + (distortion if distortion is not None else distortion_target * 2.5) / distortion_target)
            throughput_component = _clamp((throughput if throughput is not None else throughput_target * 0.58) / throughput_target, 0.0, 1.0)
            cra_component = 1.0 / (1.0 + max(0.0, (cra if cra is not None else cra_target * 1.35) - cra_target) / max(1.0, cra_target * 0.6))
            first_order_component = 1.0 if first_order.get("efl") is not None and first_order.get("fno") is not None else 0.35
            trace_penalty = min(0.45, len(field_trace.trace_failures) * 0.045)

            score = (
                weights["spot"] * spot_component
                + weights["distortion"] * distortion_component
                + weights["throughput"] * throughput_component
                + weights["cra"] * cra_component
                + weights["first_order"] * first_order_component
                - trace_penalty
            )
            detail = (
                f"spot={_format_metric(spot)}um distortion={_format_metric(distortion)}% "
                f"throughput={_format_metric(throughput)} cra={_format_metric(cra)}deg"
            )
            return score, detail
        except Exception as exc:
            return -1.0e9, f"{type(exc).__name__}: {exc}"

    def _quick_optimize_weights(self, objective: str, overrides: dict[str, float] | None = None) -> dict[str, float]:
        if objective == "spot":
            base = {"spot": 0.62, "distortion": 0.12, "throughput": 0.08, "cra": 0.08, "first_order": 0.10}
        elif objective == "distortion":
            base = {"spot": 0.20, "distortion": 0.48, "throughput": 0.10, "cra": 0.10, "first_order": 0.12}
        elif objective == "throughput":
            base = {"spot": 0.18, "distortion": 0.10, "throughput": 0.52, "cra": 0.10, "first_order": 0.10}
        elif objective == "cra":
            base = {"spot": 0.18, "distortion": 0.10, "throughput": 0.10, "cra": 0.50, "first_order": 0.12}
        else:
            base = {"spot": 0.42, "distortion": 0.20, "throughput": 0.16, "cra": 0.12, "first_order": 0.10}
        weights = dict(base)
        if overrides:
            for key in list(weights):
                try:
                    weights[key] = _clamp(float(overrides.get(key, weights[key])), 0.0, 1.0)
                except (TypeError, ValueError):
                    weights[key] = base[key]
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 1.0e-12:
            return base
        return {key: max(0.0, value) / total for key, value in weights.items()}

    def _quick_optimize_targets(self, overrides: dict[str, float] | None = None) -> dict[str, float]:
        targets = {"spot": 25.0, "distortion": 2.0, "throughput": 0.85, "cra": 25.0}
        if not overrides:
            return targets
        limits = {
            "spot": (0.1, 500.0),
            "distortion": (0.01, 50.0),
            "throughput": (0.05, 1.0),
            "cra": (0.1, 90.0),
        }
        for key, (minimum, maximum) in limits.items():
            try:
                targets[key] = _clamp(float(overrides.get(key, targets[key])), minimum, maximum)
            except (TypeError, ValueError):
                pass
        return targets

    def _surface_has_conic(self, ifc: Any) -> bool:
        return hasattr(ifc.profile, "cc") or hasattr(ifc.profile, "ec")

    def _shift_variables_after_insert(self, session: ModelSession, inserted_index: int) -> None:
        shifted: dict[int, set[str]] = {}
        for index, tokens in session.surface_variables.items():
            shifted[index + 1 if index >= inserted_index else index] = set(tokens)
        session.surface_variables = shifted

    def _shift_variables_after_delete(self, session: ModelSession, deleted_index: int) -> None:
        shifted: dict[int, set[str]] = {}
        for index, tokens in session.surface_variables.items():
            if index == deleted_index:
                continue
            shifted[index - 1 if index > deleted_index else index] = set(tokens)
        session.surface_variables = shifted

    def _workbench_metadata_path(self, model_path: Path) -> Path:
        return model_path.with_suffix(".workbench.json")

    def _write_workbench_metadata(self, model_path: Path, session: ModelSession) -> Path:
        metadata_path = self._workbench_metadata_path(model_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        surface_variables = {
            str(index): self._variable_label(session, index)
            for index in sorted(session.surface_variables)
            if self._variable_label(session, index)
        }
        workbench = self._json_safe_metadata(session.workbench_metadata)
        if not isinstance(workbench, dict):
            workbench = {}
        workbench["surfaceVariables"] = surface_variables
        payload = {
            "schemaVersion": 2,
            "savedAt": _now().isoformat(),
            "modelPath": str(model_path),
            "updateMode": session.update_mode,
            "surfaceVariables": surface_variables,
            "workbench": workbench,
        }
        session.workbench_metadata = workbench
        metadata_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return metadata_path

    def _load_workbench_metadata(self, model_path: Path, session: ModelSession) -> list[str]:
        metadata_path = self._workbench_metadata_path(model_path)
        if not metadata_path.exists():
            return []
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [f"Workbench metadata could not be read from {metadata_path}: {type(exc).__name__}: {exc}"]
        schema_version = payload.get("schemaVersion")
        if not isinstance(payload, dict) or schema_version not in {1, 2}:
            return [f"Workbench metadata ignored because {metadata_path} has an unsupported schema."]

        raw_variables = payload.get("surfaceVariables", {})
        if not isinstance(raw_variables, dict):
            return [f"Workbench metadata ignored invalid surfaceVariables in {metadata_path}."]
        raw_workbench = payload.get("workbench", {}) if schema_version == 2 else {}
        workbench = self._json_safe_metadata(raw_workbench if isinstance(raw_workbench, dict) else {})
        if not isinstance(workbench, dict):
            workbench = {}

        sm = session.opt_model.seq_model
        restored = 0
        skipped: list[str] = []
        session.surface_variables = {}
        for raw_index, raw_tokens in raw_variables.items():
            try:
                index = int(raw_index)
            except Exception:
                skipped.append(str(raw_index))
                continue
            if index <= 0 or index >= sm.get_num_surfaces() - 1:
                skipped.append(str(raw_index))
                continue
            try:
                tokens = self._parse_variable_tokens(str(raw_tokens))
            except ValueError:
                skipped.append(str(raw_index))
                continue
            if tokens:
                session.surface_variables[index] = tokens
                restored += len(tokens)

        warnings: list[str] = []
        if restored:
            warnings.append(f"Restored {restored} workbench variable flags from {metadata_path}.")
        if skipped:
            warnings.append(f"Ignored incompatible workbench variable rows from {metadata_path}: {', '.join(skipped[:8])}.")
        workbench["surfaceVariables"] = {
            str(index): self._variable_label(session, index)
            for index in sorted(session.surface_variables)
            if self._variable_label(session, index)
        }
        session.workbench_metadata = workbench
        return warnings

    def _json_safe_metadata(self, value: Any, depth: int = 0) -> Any:
        if depth > 8:
            return None
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            return value[:20000]
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in list(value.items())[:500]:
                if isinstance(key, (str, int, float, bool)):
                    safe[str(key)[:200]] = self._json_safe_metadata(item, depth + 1)
            return safe
        if isinstance(value, (list, tuple)):
            return [self._json_safe_metadata(item, depth + 1) for item in list(value)[:1000]]
        return str(value)[:20000]


def _check_stage(name: str, status: str, detail: str) -> ExampleCheckStageDTO:
    return ExampleCheckStageDTO(name=name, status=status, detail=detail)


def _check_status(stages: list[ExampleCheckStageDTO], warnings: list[str], errors: list[str]) -> str:
    if errors or any(stage.status == "fail" for stage in stages):
        return "fail"
    if warnings or any(stage.status == "warn" for stage in stages):
        return "warn"
    return "pass"


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(value))
    return "-".join(part for part in cleaned.split())


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max_value, max(min_value, value))


def _mtf50_from_spot_rms_um(spot_rms_um: float | None) -> float | None:
    if spot_rms_um is None or spot_rms_um <= 0:
        return None
    rms_radius_mm = spot_rms_um / 1000.0
    return math.sqrt(math.log(2.0)) / (math.pi * rms_radius_mm)


def _tolerance_case_score(trace: FieldTraceSummary) -> float:
    if trace.trace_failures or trace.max_spot_rms_um is None:
        return 0.0
    mtf50 = _mtf50_from_spot_rms_um(trace.max_spot_rms_um)
    if mtf50 is None:
        return 0.0
    components = [
        _score_low_good(trace.max_spot_rms_um, 10.0, 25.0),
        _score_high_good(mtf50, 20.0, 10.0),
    ]
    if trace.min_pupil_throughput is not None:
        components.append(_score_high_good(trace.min_pupil_throughput, 0.85, 0.70))
    if trace.max_distortion_pct is not None:
        components.append(_score_low_good(trace.max_distortion_pct, 2.0, 3.25))
    if trace.max_cra_deg is not None:
        components.append(_score_low_good(trace.max_cra_deg, 25.0, 30.0))
    return min(components)


def _tolerance_status(score: float) -> str:
    if score >= 0.86:
        return "pass"
    if score >= 0.70:
        return "warn"
    return "fail"


def _score_high_good(value: float, pass_value: float, fail_value: float) -> float:
    if value <= fail_value:
        return 0.0
    if value >= pass_value:
        return 1.0
    return (value - fail_value) / (pass_value - fail_value)


def _score_low_good(value: float, pass_value: float, fail_value: float) -> float:
    if value >= fail_value:
        return 0.0
    if value <= pass_value:
        return 1.0
    return (fail_value - value) / (fail_value - pass_value)


def _status_for(value: float, warn: float, fail: float, direction: str) -> str:
    if direction == "high-good":
        if value <= fail:
            return "fail"
        if value <= warn:
            return "warn"
        return "pass"
    if value >= fail:
        return "fail"
    if value >= warn:
        return "warn"
    return "pass"


def _status_from_count(fail_count: int, warn_count: int) -> str:
    if fail_count > 0:
        return "fail"
    if warn_count > 0:
        return "warn"
    return "pass"


def _is_info_warning(message: str) -> bool:
    info_prefixes = (
        "Created a new",
        "Model updated",
        "Saved model",
        "Inserted a new surface",
        "Deleted surface",
    )
    return message.startswith(info_prefixes)


def _spot_pupil_samples() -> list[list[float]]:
    samples: list[list[float]] = []
    for x in np.linspace(-1.0, 1.0, 7):
        for y in np.linspace(-1.0, 1.0, 7):
            if x * x + y * y <= 1.000001:
                samples.append([float(x), float(y)])
    return samples


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _radius_from_cv(cv: float) -> float | None:
    if abs(cv) < 1.0e-14:
        return None
    return 1.0 / cv


def _cv_from_radius(radius: float) -> float:
    if abs(radius) < 1.0e-14:
        return 0.0
    return 1.0 / radius


def _now() -> datetime:
    return datetime.now(timezone.utc)
