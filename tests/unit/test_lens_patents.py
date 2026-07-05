from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyisetcam import (
    camera_compute,
    camera_create,
    camera_get,
    camera_set,
    lens_patent_camerae2e_manifest,
    lens_patent_companies,
    lens_patent_company_db_path,
    lens_patent_company_sets_manifest,
    lens_patent_db_summary,
    lens_patent_default_data_dir,
    lens_patent_default_db_path,
    lens_patent_downsample_psf,
    lens_patent_get,
    lens_patent_optics,
    lens_patent_raytrace_optics,
    lens_patent_raytrace_psf_manifest,
    lens_patent_raytrace_psf_search,
    lens_patent_search,
    lens_patent_surfaces,
    oi_compute,
    oi_create,
    oi_set,
    scene_create,
)


def test_lens_patent_default_package_uses_matching_db_and_assets() -> None:
    data_dir = lens_patent_default_data_dir()
    db_path = lens_patent_default_db_path()
    psf_manifest = lens_patent_raytrace_psf_manifest()

    assert data_dir == db_path.parent
    if (data_dir / "camerae2e_manifest.json").exists():
        package_manifest = lens_patent_camerae2e_manifest()
        assert package_manifest["files"]["master_db"] == db_path.name
    assert Path(str(psf_manifest["camerae2e_db"])).name == db_path.name
    assert psf_manifest["summary"]["generated"] > 0


def test_lens_patent_db_summary_and_companies_are_loadable() -> None:
    summary = lens_patent_db_summary()
    companies = lens_patent_companies()

    assert summary["companies"] >= 10
    assert summary["simulation_results"] >= 300
    assert summary["status_counts"]["camerae2e_ready"] > 0
    assert any(row["company"] == "Canon" for row in companies)


def test_lens_patent_search_returns_camerae2e_ready_canon_optics() -> None:
    rows = lens_patent_search(company="Canon", require_camerae2e=True, limit=3)

    assert rows
    assert rows[0]["simulation_status"] == "camerae2e_ready"
    assert rows[0]["focal_length_mm"] > 0
    assert rows[0]["f_number"] > 0


def test_lens_patent_company_db_sets_are_loadable() -> None:
    manifest = lens_patent_company_sets_manifest()
    canon_db = lens_patent_company_db_path("Canon")
    canon_rows = lens_patent_search(db_path=canon_db, require_camerae2e=True, limit=2)

    assert manifest["summary"]["companies"] >= 10
    assert manifest["summary"]["simulation_results"] >= 300
    assert canon_db.name.endswith("_canon.sqlite")
    assert len(canon_rows) == 2
    assert {row["company"] for row in canon_rows} == {"Canon"}


def test_lens_patent_optics_can_be_attached_to_optical_image() -> None:
    row = lens_patent_search(company="Canon", require_camerae2e=True, limit=1)[0]
    optics = lens_patent_optics(row["simulation_id"])

    assert optics["model"] == "diffractionlimited"
    assert optics["focal_length_m"] > 0
    assert optics["f_number"] > 0
    assert optics["lens_patent"]["simulation_id"] == row["simulation_id"]

    oi = oi_set(oi_create(), "optics", optics)
    assert oi.fields["optics"]["focal_length_m"] == pytest.approx(optics["focal_length_m"])
    assert oi.fields["optics"]["f_number"] == pytest.approx(optics["f_number"])


def test_lens_patent_get_and_surfaces_match_simulation_surface_count() -> None:
    row = lens_patent_search(require_camerae2e=True, limit=1)[0]
    result = lens_patent_get(row["simulation_id"])
    surfaces = lens_patent_surfaces(row["lens_id"], row["configuration"])

    assert result["simulation_id"] == row["simulation_id"]
    assert result["optics"]["lens_patent"]["lens_id"] == row["lens_id"]
    assert len(surfaces) == row["surface_count"]
    assert {"surface_order", "radius_mm", "thickness_mm", "coefficients", "raw"} <= set(surfaces[0])


def test_lens_patent_generated_raytrace_psf_grid_is_loadable() -> None:
    manifest = lens_patent_raytrace_psf_manifest()
    generated = lens_patent_raytrace_psf_search(company="Canon", status="generated")

    assert manifest["summary"]["generated"] >= 1
    assert generated
    optics = lens_patent_raytrace_optics(generated[0]["simulation_id"])
    psf = np.asarray(optics["raytrace"]["psf"]["function"], dtype=float)

    assert optics["model"] == "raytrace"
    assert psf.ndim == 4
    assert np.isclose(float(np.sum(psf[:, :, 0, 0])), 1.0)
    assert np.asarray(optics["raytrace"]["psf"]["field_height_mm"], dtype=float).size >= 1


def test_lens_patent_raytrace_psf_downsample_preserves_energy_and_spacing() -> None:
    generated = lens_patent_raytrace_psf_search(company="Canon", status="generated")
    assert generated

    source = lens_patent_raytrace_optics(generated[0]["simulation_id"])
    downsampled = lens_patent_raytrace_optics(generated[0]["simulation_id"], target_psf_size=32)
    source_psf = np.asarray(source["raytrace"]["psf"]["function"], dtype=float)
    downsampled_psf = np.asarray(downsampled["raytrace"]["psf"]["function"], dtype=float)
    direct_downsampled = lens_patent_downsample_psf(source_psf, 32)

    assert source_psf.shape[0] >= 32
    assert downsampled_psf.shape[:2] == (32, 32)
    assert np.allclose(downsampled_psf, direct_downsampled)
    assert np.isclose(float(np.sum(downsampled_psf[:, :, 0, 0])), 1.0)
    assert np.isclose(
        float(downsampled["raytrace"]["psf"]["sample_spacing_mm"][0]),
        float(source["raytrace"]["psf"]["sample_spacing_mm"][0]) * (source_psf.shape[0] / 32),
    )
    assert downsampled["raytrace"]["source_psf_shape"] == list(source_psf.shape)


def test_lens_patent_highres_production_sample_is_loadable() -> None:
    psf_dir = lens_patent_default_db_path().parent / "raytrace_psf_highres"
    manifest = lens_patent_raytrace_psf_manifest(psf_dir)

    assert manifest["settings"]["preset"] == "production"
    assert manifest["settings"]["psf_size"] == 512
    assert manifest["summary"]["generated"] >= 1

    optics = lens_patent_raytrace_optics(
        "p0014:intermediate",
        psf_dir=psf_dir,
        target_psf_size=128,
    )
    psf = np.asarray(optics["raytrace"]["psf"]["function"], dtype=float)

    assert psf.shape == (128, 128, 5, 3)
    assert np.isclose(float(np.sum(psf[:, :, 0, 0])), 1.0)
    assert optics["raytrace"]["build_settings"]["preset"] == "production"


def test_lens_patent_raytrace_optics_runs_through_oi_compute() -> None:
    psf_dir = lens_patent_default_db_path().parent / "raytrace_psf_highres"
    scene = scene_create("uniform ee", 16, np.array([550.0], dtype=float))
    optics = lens_patent_raytrace_optics(
        "p0014:intermediate",
        psf_dir=psf_dir,
        target_psf_size=64,
    )
    oi = oi_set(oi_create("ray trace"), "optics", optics)

    result = oi_compute(oi, scene, crop=True)
    photons = np.asarray(result.data["photons"], dtype=float)

    assert result.fields["optics"]["model"] == "raytrace"
    assert result.fields["optics"]["raytrace"]["psf"]["function"].shape == (64, 64, 5, 3)
    assert photons.shape == (16, 16, 1)
    assert np.all(np.isfinite(photons))
    assert float(np.max(photons)) > 0.0


def test_lens_patent_raytrace_optics_runs_full_camera_pipeline() -> None:
    psf_dir = lens_patent_default_db_path().parent / "raytrace_psf_highres"
    scene = scene_create("uniform ee", 16, np.array([550.0], dtype=float))
    optics = lens_patent_raytrace_optics(
        "p0014:intermediate",
        psf_dir=psf_dir,
        target_psf_size=64,
    )
    camera = camera_create()
    camera = camera_set(camera, "oi", oi_create("ray trace"))
    camera = camera_set(camera, "optics", optics)
    camera = camera_set(camera, "sensor size", [16, 16])

    result = camera_compute(camera, scene, sensor_resize=False)
    image = np.asarray(camera_get(result, "image"), dtype=float)
    sensor_volts = np.asarray(camera_get(result, "sensor volts"), dtype=float)

    assert camera_get(result, "optics")["model"] == "raytrace"
    assert camera_get(result, "optics")["raytrace"]["psf"]["function"].shape == (64, 64, 5, 3)
    assert sensor_volts.shape == (16, 16)
    assert image.shape == (16, 16, 3)
    assert np.all(np.isfinite(sensor_volts))
    assert np.all(np.isfinite(image))
    assert float(np.max(sensor_volts)) > 0.0
    assert float(np.max(image)) > 0.0
