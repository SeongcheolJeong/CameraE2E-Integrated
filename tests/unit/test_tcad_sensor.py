from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyisetcam import (
    DEFAULT_WAVE,
    OpticalImage,
    fdtd_sensor_lut_load,
    sensor_attach_physics_lut,
    sensor_attach_tcad_lut,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
    tcad_sensor_apply_collection_response,
    tcad_sensor_collection_efficiency,
    tcad_sensor_config,
    tcad_sensor_db_load,
    tcad_sensor_generation_map_slice,
    tcad_sensor_split_phase,
    tcad_sensor_summary,
    tcad_sensor_validate,
)


def _write_fdtd_lut(tmp_path: Path) -> Path:
    summary_csv = tmp_path / "camera_lut_summary.csv"
    json_path = tmp_path / "camera_lut.json"
    summary_csv.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "ocl-3x3,550,center,0,0,0,0,1.0,1.0",
                "ocl-3x3,550,edge20x,1,0,20,0,0.5,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "schema": "camera_supercell_optical_lut_v2",
                "mode": "ocl-3x3",
                "wavelengths_nm": [550.0],
                "cases": [
                    {"name": "center", "field_x_norm": 0.0, "field_z_norm": 0.0, "cra_x_deg": 0.0, "cra_z_deg": 0.0},
                    {"name": "edge20x", "field_x_norm": 1.0, "field_z_norm": 0.0, "cra_x_deg": 20.0, "cra_z_deg": 0.0},
                ],
                "summary_csv": str(summary_csv),
            }
        ),
        encoding="utf-8",
    )
    return json_path


def _write_tcad_db(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    generation_path = tmp_path / "tcad_generation_map_2d.npz"
    generation = np.zeros((2, 3, 4), dtype=float)
    generation[0] = np.array(
        [
            [1.0, 0.6, 0.25, 0.1],
            [1.2, 0.7, 0.3, 0.1],
            [1.0, 0.6, 0.25, 0.1],
        ],
        dtype=float,
    )
    generation[1] = generation[0] * np.array([[0.8], [1.0], [1.4]], dtype=float)
    np.savez(
        generation_path,
        schema=np.asarray(["tcad_generation_map_2d_x_depth_v1"]),
        generation_cm3_s=generation * 1.0e21,
        absorption_fraction_per_um2=generation * 1.0e-3,
        x_um=np.array([-0.5, 0.0, 0.5], dtype=float),
        depth_um_from_si_top=np.array([0.1, 0.4, 0.8, 1.2], dtype=float),
        y_um=np.array([0.0, -0.3, -0.7, -1.1], dtype=float),
        case=np.asarray(["center", "edge20x"]),
        wavelength_nm=np.asarray([550.0, 550.0]),
        field_x_norm=np.asarray([0.0, 1.0]),
        field_z_norm=np.asarray([0.0, 0.0]),
        cra_x_deg=np.asarray([0.0, 20.0]),
        cra_z_deg=np.asarray([0.0, 0.0]),
        color_channel=np.asarray(["green"]),
        incident_photon_flux_cm2_s=np.asarray([1.0e20]),
        method=np.asarray(["synthetic unit-test map"]),
    )

    summary_paths: list[Path] = []
    for case, left, right, phase in (
        ("center", 1.0e-6, 2.0e-6, -0.01),
        ("edge20x", 2.5e-6, 3.5e-6, 0.04),
    ):
        path = tmp_path / f"{case}_summary.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "devsim_split_pd_2d_smoke_v1",
                    "devsim_version": "2.10.0",
                    "config": {
                        "generation_profile_case": case,
                        "generation_profile_wavelength_nm": 550.0,
                        "generation_map_npz": str(generation_path),
                    },
                    "generation_source": "imported_2d_map",
                    "electrical_model": "proxy-pinned-split-pd",
                    "left_photo_delta_a_per_cm": left,
                    "right_photo_delta_a_per_cm": right,
                    "photo_split_phase_x_proxy": phase,
                    "terminal_current_balance_illuminated_a_per_cm": 1.0e-15,
                    "dark": {},
                    "illuminated": {},
                    "notes": ["synthetic TCAD smoke result"],
                }
            ),
            encoding="utf-8",
        )
        summary_paths.append(path)

    gate_path = tmp_path / "tcad_accuracy_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema": "tcad_accuracy_gate_v1",
                "profile_name": "synthetic_reference_not_measured",
                "framework_ready": True,
                "accuracy_ready": False,
                "accuracy_blocking_failure_count": 3,
                "framework_blocking_failure_count": 0,
                "inputs": {},
                "checks": [
                    {"name": "profile_calibration_status", "status": "FAIL", "accuracy_blocking": True},
                    {"name": "terminal_current_balance", "status": "PASS", "accuracy_blocking": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return generation_path, summary_paths, gate_path


def _synthetic_oi(rows: int = 12, cols: int = 12) -> OpticalImage:
    wave = np.asarray(DEFAULT_WAVE, dtype=float)
    photons = np.ones((rows, cols, wave.size), dtype=float) * 1.0e12
    oi = OpticalImage(name="synthetic tcad oi")
    spacing = 2.8e-6
    oi.fields.update(
        {
            "wave": wave,
            "sample_spacing_m": spacing,
            "width_m": cols * spacing,
            "height_m": rows * spacing,
            "fov_deg": 1.0,
            "vfov_deg": 1.0,
            "optics": {"model": "skip", "focal_length_m": 0.004},
        }
    )
    oi.data["photons"] = photons
    return oi


def test_tcad_sensor_db_load_validate_and_summary(tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    db = tcad_sensor_db_load(
        generation_map_path=generation_path,
        collection_summary_paths=summary_paths,
        accuracy_gate_path=gate_path,
    )

    validation = tcad_sensor_validate(db)
    summary = tcad_sensor_summary(db)

    assert validation["ok"] is True
    assert validation["status"] == "proxy-framework"
    assert validation["framework_ready"] is True
    assert validation["accuracy_ready"] is False
    assert summary["generation_map"]["shape"] == [2, 3, 4]
    assert summary["accuracy_gate"]["accuracy_blocking_failure_count"] == 3


def test_tcad_collection_efficiency_split_phase_and_generation_slice(tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    db = tcad_sensor_db_load(
        generation_map_path=generation_path,
        collection_summary_paths=summary_paths,
        accuracy_gate_path=gate_path,
    )

    x_um, depth_um, generation = tcad_sensor_generation_map_slice(db, case="edge20x")

    assert tcad_sensor_collection_efficiency(db, case="center") == pytest.approx(1.0)
    assert tcad_sensor_collection_efficiency(db, case="edge20x") == pytest.approx(2.0)
    assert tcad_sensor_split_phase(db, case="edge20x") == pytest.approx(0.04)
    assert x_um.shape == (3,)
    assert depth_um.shape == (4,)
    assert generation.shape == (3, 4)


def test_tcad_collection_response_requires_explicit_config(tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    db = tcad_sensor_db_load(
        generation_map_path=generation_path,
        collection_summary_paths=summary_paths,
        accuracy_gate_path=gate_path,
    )
    values = np.ones((3, 3), dtype=float)

    unchanged = tcad_sensor_apply_collection_response(values, None)
    changed = tcad_sensor_apply_collection_response(values, tcad_sensor_config(db, case="edge20x"))

    assert np.allclose(unchanged, values)
    assert np.allclose(changed, values * 2.0)
    with pytest.raises(ValueError):
        tcad_sensor_apply_collection_response(
            values,
            tcad_sensor_config(db, case="edge20x", allow_proxy_accuracy=False),
        )


def test_tcad_config_accepts_top_level_path_mapping(tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    config = tcad_sensor_config(
        {
            "generation_map_path": str(generation_path),
            "collection_summary_paths": [str(path) for path in summary_paths],
            "accuracy_gate_path": str(gate_path),
        },
        case="edge20x",
    )

    response = tcad_sensor_apply_collection_response(np.ones((2, 2), dtype=float), config)

    assert np.allclose(response, 2.0)


def test_sensor_compute_with_tcad_collection_hook_changes_only_when_attached(asset_store, tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    db = tcad_sensor_db_load(
        generation_map_path=generation_path,
        collection_summary_paths=summary_paths,
        accuracy_gate_path=gate_path,
    )
    oi = _synthetic_oi()
    base = sensor_set(sensor_create("monochrome", asset_store=asset_store), "size", oi.data["photons"].shape[:2])
    base = sensor_set(base, "noise flag", 0)
    base = sensor_set(base, "integration time", 0.01)
    tcad = sensor_attach_tcad_lut(base, db, case="edge20x")

    base_result = sensor_compute(base, oi, seed=0)
    tcad_result = sensor_compute(tcad, oi, seed=0)

    base_volts = np.asarray(sensor_get(base_result, "volts"), dtype=float)
    tcad_volts = np.asarray(sensor_get(tcad_result, "volts"), dtype=float)
    assert np.allclose(tcad_volts, base_volts * 2.0)
    assert sensor_get(base_result, "tcad sensor") is None
    assert sensor_get(tcad, "tcad sensor") is not None


def test_sensor_attach_physics_lut_can_attach_tcad_without_fdtd(asset_store, tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    db = tcad_sensor_db_load(
        generation_map_path=generation_path,
        collection_summary_paths=summary_paths,
        accuracy_gate_path=gate_path,
    )
    sensor = sensor_create("monochrome", asset_store=asset_store)

    updated = sensor_attach_physics_lut(sensor, tcad_db=db, tcad_kwargs={"case": "edge20x"})

    assert sensor_get(sensor, "tcad sensor") is None
    assert sensor_get(updated, "tcad sensor") is not None


def test_sensor_attach_physics_lut_combines_fdtd_and_tcad_layers(asset_store, tmp_path: Path) -> None:
    generation_path, summary_paths, gate_path = _write_tcad_db(tmp_path)
    db = tcad_sensor_db_load(
        generation_map_path=generation_path,
        collection_summary_paths=summary_paths,
        accuracy_gate_path=gate_path,
    )
    fdtd_lut = fdtd_sensor_lut_load(_write_fdtd_lut(tmp_path))
    oi = _synthetic_oi(rows=16, cols=16)
    base = sensor_set(sensor_create("monochrome", asset_store=asset_store), "size", oi.data["photons"].shape[:2])
    base = sensor_set(base, "noise flag", 0)
    base = sensor_set(base, "integration time", 0.01)
    combined = sensor_attach_physics_lut(
        base,
        fdtd_lut=fdtd_lut,
        tcad_db=db,
        fdtd_kwargs={"mode": "field"},
        tcad_kwargs={"case": "edge20x"},
    )

    base_result = sensor_compute(base, oi, seed=0)
    combined_result = sensor_compute(combined, oi, seed=0)

    assert sensor_get(combined, "fdtd sensor") is not None
    assert sensor_get(combined, "tcad sensor") is not None
    assert not np.allclose(sensor_get(base_result, "volts"), sensor_get(combined_result, "volts"))
