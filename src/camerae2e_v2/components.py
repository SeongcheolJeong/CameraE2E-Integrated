"""Lens/sensor exploration and camera-module compatibility analysis."""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any

from pyisetcam.image_sensor_db import image_sensor_db_get, image_sensor_db_records
from pyisetcam.lens_patents import (
    lens_patent_companies,
    lens_patent_get,
    lens_patent_raytrace_psf_search,
    lens_patent_search,
    lens_patent_surfaces,
)

from .models import CameraModule, ComponentSelection, RequirementSet

_FRAME_SENSOR_MODALITIES = {
    "cmos_image_sensor",
    "global_shutter_cis",
    "image_sensor",
}


class ComponentCatalogService:
    """Read-only normalized view over the bundled lens and sensor databases."""

    @cached_property
    def lenses(self) -> list[dict[str, Any]]:
        psf_ids = {
            str(row.get("simulation_id"))
            for row in lens_patent_raytrace_psf_search(status="generated")
        }
        return [self._normalize_lens(row, psf_ids) for row in lens_patent_search()]

    @cached_property
    def sensors(self) -> list[dict[str, Any]]:
        return [self._normalize_sensor(row) for row in image_sensor_db_records()]

    @cached_property
    def lens_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item["id"]): item for item in self.lenses}

    @cached_property
    def sensor_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item["id"]): item for item in self.sensors}

    @cached_property
    def psf_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("simulation_id")): dict(item)
            for item in lens_patent_raytrace_psf_search(status="generated")
        }

    def search_lenses(
        self,
        *,
        query: str = "",
        company: str | None = None,
        focal_min: float | None = None,
        focal_max: float | None = None,
        f_number_max: float | None = None,
        fov_min: float | None = None,
        fov_max: float | None = None,
        readiness: str | None = None,
        require_psf: bool | None = None,
        simulation_ready: bool | None = None,
        sort: str = "company",
        order: str = "asc",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        needle = query.strip().casefold()
        rows = []
        for item in self.lenses:
            haystack = " ".join(
                str(item.get(key, ""))
                for key in ("id", "lens_id", "company", "publication_number", "configuration")
            ).casefold()
            if needle and needle not in haystack:
                continue
            if company and str(item["company"]).casefold() != company.casefold():
                continue
            if readiness and item["readiness"] != readiness:
                continue
            if require_psf is not None and bool(item["psf_available"]) != require_psf:
                continue
            if (
                simulation_ready is not None
                and bool(item["module_configurable"]) != simulation_ready
            ):
                continue
            if not _range_match(item.get("focal_length_mm"), focal_min, focal_max):
                continue
            if f_number_max is not None and not _upper_match(item.get("f_number"), f_number_max):
                continue
            if not _range_match(item.get("field_of_view_deg"), fov_min, fov_max):
                continue
            rows.append(item)
        aliases = {
            "company": "company",
            "focal_length": "focal_length_mm",
            "f_number": "f_number",
            "fov": "field_of_view_deg",
            "image_height": "image_height_mm",
            "complexity": "surface_count",
        }
        rows = _sort_rows(rows, aliases.get(sort, "company"), order)
        return self._page(
            rows,
            page,
            page_size,
            facets={
                "companies": [item["company"] for item in lens_patent_companies()],
                "readiness": sorted({str(item["readiness"]) for item in self.lenses}),
                "psf_count": sum(bool(item["psf_available"]) for item in self.lenses),
                "configurable_count": sum(
                    bool(item["module_configurable"]) for item in self.lenses
                ),
            },
        )

    def lens_detail(self, simulation_id: str) -> dict[str, Any]:
        if simulation_id not in self.lens_by_id:
            raise KeyError(f"Unknown lens component: {simulation_id}")
        item = dict(self.lens_by_id[simulation_id])
        source = lens_patent_get(simulation_id)
        item["optics"] = source.get("optics", {})
        item["surfaces"] = lens_patent_surfaces(
            str(source["lens_id"]), configuration=str(source["configuration"])
        )
        item["geometric_psf"] = self.psf_by_id.get(simulation_id)
        item["truth_boundary"] = (
            "Patent-derived prescription. Bundled RayOptics PSFs are geometric ray "
            "histograms and do not include diffraction or measured lens performance."
        )
        return item

    def search_sensors(
        self,
        *,
        query: str = "",
        manufacturer: str | None = None,
        pixel_min: float | None = None,
        pixel_max: float | None = None,
        resolution_min: float | None = None,
        cfa: str | None = None,
        shutter: str | None = None,
        has_dti: bool | None = None,
        has_pdaf: bool | None = None,
        has_hdr: bool | None = None,
        has_lofic: bool | None = None,
        is_stacked: bool | None = None,
        simulation_ready: bool | None = None,
        sort: str = "manufacturer",
        order: str = "asc",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        needle = query.strip().casefold()
        rows = []
        for item in self.sensors:
            haystack = " ".join(
                str(item.get(key, ""))
                for key in ("id", "code", "manufacturer", "device_name", "title")
            ).casefold()
            if needle and needle not in haystack:
                continue
            if manufacturer and str(item["manufacturer"]).casefold() != manufacturer.casefold():
                continue
            if not _range_match(item.get("pixel_pitch_um"), pixel_min, pixel_max):
                continue
            if resolution_min is not None and not _lower_match(
                item.get("resolution_mp"), resolution_min
            ):
                continue
            if cfa and cfa.casefold() not in str(item.get("cfa_pattern") or "").casefold():
                continue
            if shutter and shutter.casefold() != str(item.get("shutter") or "").casefold():
                continue
            if (
                simulation_ready is not None
                and bool(item["module_configurable"]) != simulation_ready
            ):
                continue
            requested_flags = {
                "has_dti": has_dti,
                "has_pdaf": has_pdaf,
                "has_hdr": has_hdr,
                "has_lofic": has_lofic,
                "is_stacked": is_stacked,
            }
            if any(
                value is not None and item.get(key) is not value
                for key, value in requested_flags.items()
            ):
                continue
            rows.append(item)
        aliases = {
            "manufacturer": "manufacturer",
            "pixel_pitch": "pixel_pitch_um",
            "resolution": "resolution_mp",
            "year": "analysis_year",
            "evidence": "evidence_completeness",
        }
        rows = _sort_rows(rows, aliases.get(sort, "manufacturer"), order)
        return self._page(
            rows,
            page,
            page_size,
            facets={
                "manufacturers": sorted({str(item["manufacturer"]) for item in self.sensors}),
                "shutters": sorted(
                    {str(item["shutter"]) for item in self.sensors if item.get("shutter")}
                ),
                "configurable_count": sum(
                    bool(item["module_configurable"]) for item in self.sensors
                ),
            },
        )

    def sensor_detail(self, sensor_id: str) -> dict[str, Any]:
        if sensor_id not in self.sensor_by_id:
            raise KeyError(f"Unknown sensor component: {sensor_id}")
        item = dict(self.sensor_by_id[sensor_id])
        source = image_sensor_db_get(sensor_id)
        raw = dict(source.get("raw", {}))
        item["derived_specs"] = dict(raw.get("derived_specs", {}))
        item["evidence"] = dict(raw.get("derived_evidence", {}))
        item["generated_files"] = dict(raw.get("generated_files", {}))
        item["truth_boundary"] = (
            "Source-derived sensor structure record. Generated stack/TCAD files are "
            "proxy inputs; sensor-specific measured QE, PTC, full-well and noise are not implied."
        )
        return item

    def evaluate_modules(
        self,
        requirements: RequirementSet,
        selections: list[ComponentSelection],
        *,
        baseline: CameraModule | None = None,
    ) -> dict[str, Any]:
        if not 1 <= len(selections) <= 4:
            raise ValueError("Module comparison requires one to four candidates")
        candidates = [
            self._evaluate_selection(requirements, selection, baseline=baseline)
            for selection in selections
        ]
        _mark_pareto(candidates)
        counts = {
            status: sum(item["status"] == status for item in candidates)
            for status in ("compatible", "conditional", "incompatible")
        }
        return {
            "schema_version": "camerae2e_component_module_evaluation_v1",
            "requirements": requirements.model_dump(mode="json"),
            "candidate_count": len(candidates),
            "status_counts": counts,
            "candidates": candidates,
            "truth_boundary": (
                "Compatibility uses source-derived specifications and explicit analytic "
                "geometry. Unknown measurements remain not_evaluable; Pareto membership is "
                "not a product recommendation."
            ),
        }

    def build_module(
        self,
        selection: ComponentSelection,
        baseline: CameraModule,
        requirements: RequirementSet,
    ) -> tuple[CameraModule, dict[str, Any]]:
        evaluation = self._evaluate_selection(requirements, selection, baseline=baseline)
        lens = evaluation["lens"]
        sensor = evaluation["sensor"]
        if not lens["module_configurable"]:
            blockers = ", ".join(lens["configuration_blockers"])
            raise ValueError(
                f"Lens {selection.lens_id} cannot run in the optics model: {blockers}"
            )
        if not sensor["module_configurable"]:
            blockers = ", ".join(sensor["configuration_blockers"])
            raise ValueError(
                f"Sensor {selection.sensor_id} cannot run in the frame-RAW model: {blockers}"
            )
        native_rows, native_cols = sensor["native_size_rc"]
        sim_rows, sim_cols = _simulation_size(native_rows, native_cols)
        native_pitch_um = float(sensor["pixel_pitch_um"])
        simulation_pitch_um = native_pitch_um * native_cols / sim_cols
        simulation_fill_factor = baseline.sensor.pixel_fill_factor * (
            native_pitch_um / simulation_pitch_um
        ) ** 2
        ray_psf = bool(selection.use_geometric_psf and lens["psf_available"])
        cfa_preset = _cfa_preset(sensor.get("cfa_pattern"))
        if cfa_preset is None:
            raise ValueError(f"Sensor {selection.sensor_id} has no supported CFA model")
        module = baseline.model_copy(
            update={
                "name": selection.name
                or f"{lens['company']} + {sensor['manufacturer']} {sensor['device_name']}",
                "lens": baseline.lens.model_copy(
                    update={
                        "model_id": selection.lens_id,
                        "hfov_deg": evaluation["derived"]["hfov_deg"],
                        "f_number": lens["f_number"],
                        "focal_length_mm": lens["focal_length_mm"],
                        "psf_radius_um": max(
                            float(lens["airy_disk_diameter_um_550"] or 4.0) / 2.0, 0.1
                        ),
                    }
                ),
                "sensor": baseline.sensor.model_copy(
                    update={
                        "model_id": selection.sensor_id,
                        "rows": sim_rows,
                        "cols": sim_cols,
                        "native_rows": native_rows,
                        "native_cols": native_cols,
                        "geometry_source": sensor["geometry_source"],
                        "pixel_size_um": native_pitch_um,
                        "simulation_pixel_size_um": simulation_pitch_um,
                        "simulation_fill_factor": simulation_fill_factor,
                        "cfa_preset": cfa_preset,
                    }
                ),
                "metadata": {
                    **baseline.metadata,
                    "component_selection": selection.model_dump(mode="json"),
                    "component_compatibility": evaluation,
                    "native_sensor_size_rc": [native_rows, native_cols],
                    "simulation_readout_size_rc": [sim_rows, sim_cols],
                    "simulation_pixel_size_um": simulation_pitch_um,
                    "simulation_fill_factor": simulation_fill_factor,
                    "simulation_geometry_policy": (
                        "native_readout"
                        if (sim_rows, sim_cols) == (native_rows, native_cols)
                        else "full_extent_sparse_sampling_proxy"
                    ),
                    "sensor_qe_policy": "existing_baseline_profile_not_sensor_specific_measurement",
                    "rayoptics_simulation_id": selection.lens_id if ray_psf else None,
                    "use_geometric_psf": ray_psf,
                },
            }
        )
        return module, evaluation

    def _evaluate_selection(
        self,
        requirements: RequirementSet,
        selection: ComponentSelection,
        *,
        baseline: CameraModule | None,
    ) -> dict[str, Any]:
        try:
            lens = self.lens_by_id[selection.lens_id]
        except KeyError as exc:
            raise KeyError(f"Unknown lens component: {selection.lens_id}") from exc
        try:
            sensor = self.sensor_by_id[selection.sensor_id]
        except KeyError as exc:
            raise KeyError(f"Unknown sensor component: {selection.sensor_id}") from exc
        native = sensor.get("native_size_rc")
        pitch = sensor.get("pixel_pitch_um")
        focal = lens.get("focal_length_mm")
        f_number = lens.get("f_number")
        width_mm = native[1] * pitch * 1e-3 if native and pitch else None
        height_mm = native[0] * pitch * 1e-3 if native and pitch else None
        diagonal_mm = math.hypot(width_mm, height_mm) if width_mm and height_mm else None
        hfov = math.degrees(2.0 * math.atan2(width_mm / 2.0, focal)) if width_mm and focal else None
        diagonal_fov = (
            math.degrees(2.0 * math.atan2(diagonal_mm / 2.0, focal))
            if diagonal_mm and focal
            else None
        )
        object_pixels = (
            focal
            * requirements.reference_object_height.value
            / max(requirements.detection_range.value, 1e-12)
            / (pitch * 1e-3)
            if focal and pitch
            else None
        )
        airy_um = (
            2.44 * requirements.reference_wavelength_nm * 1e-3 * f_number if f_number else None
        )
        airy_pixels = airy_um / pitch if airy_um and pitch else None
        fps = baseline.sensor.frame_rate_fps if baseline else 30.0
        row_time_us = baseline.sensor.row_time_us if baseline else 20.0
        pixel_rate = native[0] * native[1] * fps / 1e6 if native else None
        rolling_ms = native[0] * row_time_us / 1000.0 if native else None
        image_circle_mm = 2.0 * lens["image_height_mm"] if lens.get("image_height_mm") else None
        gates: list[dict[str, Any]] = []

        def gate(
            gate_id: str, value: Any, limit: str, passed: bool | None, unit: str, note: str = ""
        ) -> None:
            gates.append(
                {
                    "id": gate_id,
                    "value": value,
                    "limit": limit,
                    "unit": unit,
                    "pass": passed,
                    "hard": True,
                    "status": "not_evaluable" if passed is None else ("pass" if passed else "fail"),
                    "note": note,
                }
            )

        gate(
            "lens_parameter_model",
            lens.get("module_configurable"),
            "positive focal length, F-number, image height and field domain",
            bool(lens.get("module_configurable")),
            "capability",
            "Incomplete patent rows remain searchable but are not simulated.",
        )
        gate(
            "frame_sensor_model",
            sensor.get("sensor_modality"),
            "frame image sensor modality",
            bool(sensor.get("frame_model_supported")),
            "enum",
            "Event, NIR and SWIR sensor physics need dedicated acquisition models.",
        )
        gate(
            "cfa_model",
            sensor.get("cfa_pattern"),
            "Bayer, RGGB Bayer, Quad Bayer or Tetracell Bayer",
            bool(sensor.get("cfa_model_supported")),
            "enum",
            "Unknown CFA values are not replaced with a default Bayer pattern.",
        )
        gate(
            "lens_image_circle",
            diagonal_mm,
            f"<= {image_circle_mm}",
            None
            if diagonal_mm is None or image_circle_mm is None
            else diagonal_mm <= image_circle_mm,
            "mm",
        )
        gate(
            "lens_field_domain",
            diagonal_fov,
            f"<= {lens.get('field_of_view_deg')}",
            None
            if diagonal_fov is None or lens.get("field_of_view_deg") is None
            else diagonal_fov <= float(lens["field_of_view_deg"]),
            "deg diagonal",
            "Uses the patent/RayOptics maximum analyzed field domain.",
        )
        tolerance = max(1.0, requirements.hfov.value * 0.02)
        gate(
            "hfov",
            hfov,
            f"{requirements.hfov.value} +/- {tolerance}",
            None if hfov is None else abs(hfov - requirements.hfov.value) <= tolerance,
            "deg",
        )
        gate(
            "object_pixels_at_range",
            object_pixels,
            f">= {requirements.min_object_height.value}",
            None
            if object_pixels is None
            else object_pixels >= requirements.min_object_height.value,
            "px",
        )
        if requirements.max_sensor_diagonal is not None:
            gate(
                "sensor_diagonal",
                diagonal_mm,
                f"<= {requirements.max_sensor_diagonal.value}",
                None
                if diagonal_mm is None
                else diagonal_mm <= requirements.max_sensor_diagonal.value,
                "mm",
            )
        gate(
            "diffraction_sampling",
            airy_pixels,
            "<= 2.0",
            None if airy_pixels is None else airy_pixels <= 2.0,
            "px",
        )
        gate(
            "pixel_bandwidth",
            pixel_rate,
            f"<= {requirements.max_pixel_rate.value}",
            None if pixel_rate is None else pixel_rate <= requirements.max_pixel_rate.value,
            "Mpixel/s",
        )
        gate(
            "rolling_shutter",
            rolling_ms,
            f"<= {requirements.max_rolling_shutter.value}",
            None if rolling_ms is None else rolling_ms <= requirements.max_rolling_shutter.value,
            "ms",
            "Uses current study row-time assumption; sensor-specific timing is unavailable.",
        )
        failed = [item for item in gates if item["pass"] is False]
        unknown = [item for item in gates if item["pass"] is None]
        status = "incompatible" if failed else ("conditional" if unknown else "compatible")
        complexity = (lens.get("surface_count") or 0) + 2 * (lens.get("asphere_count") or 0)
        low_light = pitch**2 / f_number**2 if pitch and f_number else None
        return {
            "id": f"{selection.lens_id}::{selection.sensor_id}",
            "selection": selection.model_dump(mode="json"),
            "status": status,
            "pareto": False,
            "lens": lens,
            "sensor": sensor,
            "gates": gates,
            "failed_gate_ids": [str(item["id"]) for item in failed],
            "not_evaluable_ids": [str(item["id"]) for item in unknown],
            "derived": {
                "sensor_width_mm": width_mm,
                "sensor_height_mm": height_mm,
                "sensor_diagonal_mm": diagonal_mm,
                "image_circle_mm": image_circle_mm,
                "hfov_deg": hfov,
                "diagonal_fov_deg": diagonal_fov,
                "hfov_error_deg": None if hfov is None else abs(hfov - requirements.hfov.value),
                "object_pixels_at_range": object_pixels,
                "airy_diameter_um": airy_um,
                "airy_diameter_pixels": airy_pixels,
                "pixel_rate_mpix_s": pixel_rate,
                "rolling_shutter_ms": rolling_ms,
                "relative_photon_area_proxy": low_light,
                "complexity_index": complexity,
                "evidence_completeness": min(
                    lens["evidence_completeness"], sensor["evidence_completeness"]
                ),
            },
            "truth_boundary": (
                "Low-light support is pixel-area/f-number only; QE, transmission and "
                "measured noise are not included."
            ),
        }

    @staticmethod
    def _page(
        rows: list[dict[str, Any]], page: int, page_size: int, *, facets: dict[str, Any]
    ) -> dict[str, Any]:
        safe_size = min(max(int(page_size), 1), 200)
        safe_page = max(int(page), 1)
        start = (safe_page - 1) * safe_size
        return {
            "schema_version": "camerae2e_component_search_v1",
            "total": len(rows),
            "page": safe_page,
            "page_size": safe_size,
            "items": rows[start : start + safe_size],
            "facets": facets,
        }

    @staticmethod
    def _normalize_lens(row: dict[str, Any], psf_ids: set[str]) -> dict[str, Any]:
        fields = (
            "focal_length_mm",
            "f_number",
            "image_height_mm",
            "field_of_view_deg",
            "airy_disk_diameter_um_550",
            "diffraction_cutoff_lpmm_550",
        )
        completeness = sum(row.get(key) is not None for key in fields) / len(fields)
        blockers = [
            f"{key}_missing_or_nonpositive"
            for key in ("focal_length_mm", "f_number", "image_height_mm", "field_of_view_deg")
            if not _positive_float(row.get(key))
        ]
        return {
            "id": str(row["simulation_id"]),
            "lens_id": str(row["lens_id"]),
            "company": str(row["company"]),
            "publication_number": str(row["publication_number"]),
            "configuration": str(row["configuration"]),
            "readiness": str(row["readiness"]),
            "simulation_status": str(row["simulation_status"]),
            "simulation_model": str(row["simulation_model"]),
            "focal_length_mm": row.get("focal_length_mm"),
            "f_number": row.get("f_number"),
            "image_height_mm": row.get("image_height_mm"),
            "field_of_view_deg": row.get("field_of_view_deg"),
            "airy_disk_diameter_um_550": row.get("airy_disk_diameter_um_550"),
            "diffraction_cutoff_lpmm_550": row.get("diffraction_cutoff_lpmm_550"),
            "surface_count": row.get("surface_count"),
            "asphere_count": row.get("asphere_count"),
            "psf_available": str(row["simulation_id"]) in psf_ids,
            "module_configurable": not blockers,
            "configuration_blockers": blockers,
            "readiness_tier": "proxy",
            "value_kind": "patent_derived",
            "evidence_completeness": round(completeness, 3),
        }

    @staticmethod
    def _normalize_sensor(row: dict[str, Any]) -> dict[str, Any]:
        specs = dict(row.get("raw", {}).get("derived_specs", {}))
        sensor_modality = str(specs.get("sensor_modality") or "").strip() or None
        frame_model_supported = sensor_modality in _FRAME_SENSOR_MODALITIES
        cfa_model_supported = _cfa_preset(row.get("cfa_pattern")) is not None
        native_rows = _positive_int(specs.get("resolution_y"))
        native_cols = _positive_int(specs.get("resolution_x"))
        geometry_source = "source_resolution"
        if not native_rows or not native_cols:
            native_rows, native_cols = _resolution_proxy(row.get("resolution_mp"))
            geometry_source = "resolution_mp_16_9_proxy" if native_rows else "unknown"
        blockers = []
        if not native_rows or not native_cols:
            blockers.append("sensor_geometry_missing")
        if not row.get("pixel_pitch_um"):
            blockers.append("pixel_pitch_missing")
        if not frame_model_supported:
            blockers.append("frame_sensor_model_unsupported")
        if not cfa_model_supported:
            blockers.append("cfa_model_unsupported")
        fields = (
            row.get("pixel_pitch_um"),
            row.get("resolution_mp"),
            native_rows,
            native_cols,
            sensor_modality,
            row.get("cfa_pattern"),
            specs.get("shutter"),
            row.get("optical_stack_height_um"),
            row.get("active_si_thickness_um"),
        )
        completeness = sum(value is not None for value in fields) / len(fields)
        return {
            "id": str(row["sensor_id"]),
            "code": str(row["code"]),
            "manufacturer": str(row["manufacturer"]),
            "device_name": str(row["device_name"]),
            "title": str(row["title"]),
            "analysis_year": row.get("analysis_year"),
            "pixel_pitch_um": row.get("pixel_pitch_um"),
            "resolution_mp": row.get("resolution_mp"),
            "native_size_rc": [native_rows, native_cols] if native_rows and native_cols else None,
            "geometry_source": geometry_source,
            "sensor_modality": sensor_modality,
            "frame_model_supported": frame_model_supported,
            "cfa_model_supported": cfa_model_supported,
            "module_configurable": not blockers,
            "configuration_blockers": blockers,
            "cfa_pattern": row.get("cfa_pattern"),
            "shutter": specs.get("shutter"),
            "illumination": row.get("illumination"),
            "optical_format": row.get("optical_format"),
            "microlens_type": row.get("microlens_type"),
            "pixel_architecture": row.get("pixel_architecture"),
            "has_dti": row.get("has_dti"),
            "has_pdaf": row.get("has_pdaf"),
            "has_hdr": row.get("has_hdr"),
            "has_lofic": row.get("has_lofic"),
            "is_stacked": specs.get("is_stacked"),
            "is_nir": specs.get("is_nir"),
            "stack_config_available": bool(row.get("stack_config_path")),
            "tcad_profile_available": bool(row.get("tcad_profile_path")),
            "readiness_tier": "proxy",
            "value_kind": "source_derived",
            "evidence_completeness": round(completeness, 3),
        }


def _range_match(value: Any, minimum: float | None, maximum: float | None) -> bool:
    if minimum is None and maximum is None:
        return True
    if value is None:
        return False
    number = float(value)
    return (minimum is None or number >= minimum) and (maximum is None or number <= maximum)


def _upper_match(value: Any, maximum: float) -> bool:
    return value is not None and float(value) <= maximum


def _lower_match(value: Any, minimum: float) -> bool:
    return value is not None and float(value) >= minimum


def _sort_rows(rows: list[dict[str, Any]], key: str, order: str) -> list[dict[str, Any]]:
    reverse = order.casefold() == "desc"
    return sorted(
        rows,
        key=lambda item: (
            item.get(key) is None,
            str(item.get(key)).casefold() if isinstance(item.get(key), str) else item.get(key) or 0,
        ),
        reverse=reverse,
    )


def _positive_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _resolution_proxy(value: Any) -> tuple[int | None, int | None]:
    try:
        pixels = float(value) * 1e6
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(pixels) or pixels <= 0:
        return None, None
    cols = max(2, int(round(math.sqrt(pixels * 16.0 / 9.0))) // 2 * 2)
    rows = max(2, int(round(cols * 9.0 / 16.0)) // 2 * 2)
    return rows, cols


def _simulation_size(rows: int, cols: int) -> tuple[int, int]:
    scale = min(1.0, 640.0 / cols, 360.0 / rows)
    sim_rows = max(2, int(rows * scale) // 2 * 2)
    sim_cols = max(2, int(cols * scale) // 2 * 2)
    return sim_rows, sim_cols


def _cfa_preset(value: Any) -> str | None:
    normalized = str(value or "").casefold().replace("-", " ")
    if normalized in {"quad bayer", "tetracell bayer"}:
        return "quad_bayer_rgb"
    if normalized in {"bayer", "rggb bayer"}:
        return "bayer_rgb"
    return None


def _mark_pareto(candidates: list[dict[str, Any]]) -> None:
    eligible = [item for item in candidates if item["status"] != "incompatible"]

    def vector(item: dict[str, Any]) -> tuple[float, ...]:
        values = item["derived"]
        return (
            float(values.get("object_pixels_at_range") or -math.inf),
            float(values.get("relative_photon_area_proxy") or -math.inf),
            float(values.get("evidence_completeness") or 0.0),
            -float(
                values.get("hfov_error_deg")
                if values.get("hfov_error_deg") is not None
                else math.inf
            ),
            -float(
                values.get("complexity_index")
                if values.get("complexity_index") is not None
                else math.inf
            ),
        )

    for candidate in eligible:
        current = vector(candidate)
        candidate["pareto"] = not any(
            other is not candidate
            and all(left >= right for left, right in zip(vector(other), current, strict=True))
            and any(left > right for left, right in zip(vector(other), current, strict=True))
            for other in eligible
        )
