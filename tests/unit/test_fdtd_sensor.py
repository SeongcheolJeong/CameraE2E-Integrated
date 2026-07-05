from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyisetcam import (
    DEFAULT_WAVE,
    OpticalImage,
    fdtd_sensor_apply_optical_response,
    fdtd_sensor_config,
    fdtd_sensor_cos4_relative_illumination,
    fdtd_sensor_default_lut_path,
    fdtd_sensor_field_response_map,
    fdtd_sensor_lut_crosstalk_kernel,
    fdtd_sensor_lut_load,
    fdtd_sensor_lut_response,
    fdtd_sensor_lut_validate,
    fdtd_sensor_physics_validate,
    sensor_attach_fdtd_lut,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
)


def _write_synthetic_lut(tmp_path: Path) -> Path:
    long_csv = tmp_path / "camera_lut_long.csv"
    summary_csv = tmp_path / "camera_lut_summary.csv"
    json_path = tmp_path / "camera_lut.json"

    summary_csv.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "ocl-3x3,550,center,0,0,0,0,1.0,1.0",
                "ocl-3x3,550,edge20,1,0,20,0,0.50,0.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = [
        ("center", 0, 0, 0.02),
        ("center", 1, 0, 0.03),
        ("center", 2, 0, 0.02),
        ("center", 0, 1, 0.03),
        ("center", 1, 1, 0.80),
        ("center", 2, 1, 0.03),
        ("center", 0, 2, 0.02),
        ("center", 1, 2, 0.03),
        ("center", 2, 2, 0.02),
        ("edge20", 0, 0, 0.06),
        ("edge20", 1, 0, 0.08),
        ("edge20", 2, 0, 0.06),
        ("edge20", 0, 1, 0.08),
        ("edge20", 1, 1, 0.28),
        ("edge20", 2, 1, 0.08),
        ("edge20", 0, 2, 0.06),
        ("edge20", 1, 2, 0.08),
        ("edge20", 2, 2, 0.06),
    ]
    long_lines = [
        "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,region_id,region_kind,region_ix,region_iz,response"
    ]
    for case, ix, iz, response in rows:
        field = 0 if case == "center" else 1
        cra = 0 if case == "center" else 20
        long_lines.append(f"ocl-3x3,550,{case},{field},0,{cra},0,pix_x{ix}_z{iz},pixel,{ix},{iz},{response}")
    long_csv.write_text("\n".join(long_lines) + "\n", encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "schema": "camera_supercell_optical_lut_v1",
                "mode": "ocl-3x3",
                "cell_pixels": {"x": 3, "z": 3},
                "wavelengths_nm": [550.0],
                "cases": [
                    {"name": "center", "field_x_norm": 0.0, "field_z_norm": 0.0, "cra_x_deg": 0.0, "cra_z_deg": 0.0},
                    {"name": "edge20", "field_x_norm": 1.0, "field_z_norm": 0.0, "cra_x_deg": 20.0, "cra_z_deg": 0.0},
                ],
                "regions": [
                    {"region_id": f"pix_x{ix}_z{iz}", "kind": "pixel", "ix": ix, "iz": iz}
                    for iz in range(3)
                    for ix in range(3)
                ],
                "long_csv": str(long_csv),
                "summary_csv": str(summary_csv),
                "notes": ["unit-test optical absorption proxy"],
            }
        ),
        encoding="utf-8",
    )
    return json_path


def _write_synthetic_split_pd_lut(tmp_path: Path) -> Path:
    long_csv = tmp_path / "camera_lut_long.csv"
    summary_csv = tmp_path / "camera_lut_summary.csv"
    json_path = tmp_path / "camera_lut.json"
    summary_csv.write_text(
        "\n".join(
            [
                "schema,mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "camera_supercell_optical_lut_v2,split-pd-1x1,550,center,0,0,0,0,1.0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    long_csv.write_text(
        "\n".join(
            [
                "schema,mode,wavelength_nm,case,region_id,region_kind,region_ix,region_iz,response",
                "camera_supercell_optical_lut_v2,split-pd-1x1,550,center,pd_left,subpd,-1,0,0.5",
                "camera_supercell_optical_lut_v2,split-pd-1x1,550,center,pd_right,subpd,1,0,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "schema": "camera_supercell_optical_lut_v2",
                "response_model": "si_volume_absorption_flux_calibrated_v1",
                "mode": "split-pd-1x1",
                "wavelengths_nm": [550.0],
                "long_csv": str(long_csv),
                "summary_csv": str(summary_csv),
            }
        ),
        encoding="utf-8",
    )
    return json_path


def _write_root_relative_lut(tmp_path: Path) -> Path:
    root = tmp_path / "FDTD"
    run_dir = root / "runs" / "relative_case"
    run_dir.mkdir(parents=True)
    summary_csv = run_dir / "camera_lut_summary.csv"
    long_csv = run_dir / "camera_lut_long.csv"
    json_path = run_dir / "camera_lut.json"
    summary_csv.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "split-pd-1x1,550,center,0,0,0,0,1.0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    long_csv.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,region_id,region_kind,region_ix,region_iz,response",
                "split-pd-1x1,550,center,pd_left,subpd,-1,0,0.5",
                "split-pd-1x1,550,center,pd_right,subpd,1,0,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "schema": "camera_supercell_optical_lut_v2",
                "mode": "split-pd-1x1",
                "wavelengths_nm": [550.0],
                "long_csv": "runs/relative_case/camera_lut_long.csv",
                "summary_csv": "runs/relative_case/camera_lut_summary.csv",
            }
        ),
        encoding="utf-8",
    )
    return json_path


def _synthetic_oi(rows: int = 24, cols: int = 24) -> OpticalImage:
    wave = np.asarray(DEFAULT_WAVE, dtype=float)
    edge = np.zeros((rows, cols), dtype=float)
    edge[:, cols // 2 :] = 2.0e12
    edge[:, : cols // 2] = 0.5e12
    cube = edge[:, :, None] * np.ones((1, 1, wave.size), dtype=float)

    oi = OpticalImage(name="synthetic fdtd sensor edge")
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
    oi.data["photons"] = cube
    return oi


def test_fdtd_sensor_lut_load_validate_and_response(tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_synthetic_lut(tmp_path))
    validation = fdtd_sensor_lut_validate(lut)

    assert validation["ok"] is True
    assert validation["kernel_shape"] == [3, 3]
    assert validation["n_cases"] == 2
    assert fdtd_sensor_lut_response(lut, case="center", wavelength_nm=550) == pytest.approx(1.0)
    assert fdtd_sensor_lut_response(lut, case="edge20", wavelength_nm=550) == pytest.approx(0.5)


def test_fdtd_sensor_lut_load_resolves_fdtd_root_relative_sidecars(tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_root_relative_lut(tmp_path))

    assert len(lut.summary_rows) == 1
    assert len(lut.long_rows) == 2
    assert fdtd_sensor_lut_validate(lut)["ok"] is True


def test_split_pd_lut_is_not_treated_as_pixel_crosstalk(tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_synthetic_split_pd_lut(tmp_path))

    kernel = fdtd_sensor_lut_crosstalk_kernel(lut, case="center", wavelength_nm=550)

    assert kernel.shape == (1, 1)
    assert kernel[0, 0] == pytest.approx(1.0)


def test_fdtd_sensor_physics_validation_flags_bad_ocl_and_sparse_sweep(tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_synthetic_lut(tmp_path))
    physics = fdtd_sensor_physics_validate(lut)

    assert fdtd_sensor_cos4_relative_illumination(20.0) == pytest.approx(np.cos(np.deg2rad(20.0)) ** 4)
    assert physics["status"] in {"warn", "fail"}
    assert physics["checks"]["wavelength"]["status"] == "warn"
    assert physics["checks"]["symmetry"]["status"] == "warn"
    assert physics["checks"]["relative_illumination"]["comparisons"][0]["cos4_response"] == pytest.approx(
        np.cos(np.deg2rad(20.0)) ** 4
    )


def test_fdtd_sensor_physics_validation_detects_ocl_regression(tmp_path: Path) -> None:
    path = _write_synthetic_lut(tmp_path)
    summary_path = path.with_name("camera_lut_summary.csv")
    summary_path.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "ocl-3x3,550,center,0,0,0,0,1.0,1.0",
                "ocl-3x3,550,edge20_uncomp,1,0,20,0,0.70,0.70",
                "ocl-3x3,550,edge20_ocl,1,0,20,0,0.55,0.55",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lut = fdtd_sensor_lut_load(path)
    physics = fdtd_sensor_physics_validate(lut)

    assert physics["checks"]["ocl_shift"]["status"] == "fail"
    assert physics["checks"]["ocl_shift"]["pairs"][0]["improvement"] < 0.0


def test_fdtd_sensor_field_response_map_uses_center_to_edge_rolloff(tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_synthetic_lut(tmp_path))

    response_map = fdtd_sensor_field_response_map(lut, (9, 9), wavelength_nm=550)

    assert response_map.shape == (9, 9)
    assert response_map[4, 4] == pytest.approx(1.0)
    assert response_map[0, 0] == pytest.approx(0.5)


def test_fdtd_sensor_crosstalk_kernel_spreads_impulse(tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_synthetic_lut(tmp_path))
    kernel = fdtd_sensor_lut_crosstalk_kernel(lut, case="center", wavelength_nm=550)
    impulse = np.zeros((7, 7), dtype=float)
    impulse[3, 3] = 1.0

    response = fdtd_sensor_apply_optical_response(
        impulse,
        fdtd_sensor_config(lut, mode="crosstalk", crosstalk_strength=1.0),
    )

    assert kernel.shape == (3, 3)
    assert np.isclose(float(np.sum(kernel)), 1.0)
    assert response[3, 3] < 1.0
    assert response[3, 2] > 0.0
    assert np.isclose(float(np.sum(response)), 1.0)


def test_sensor_compute_without_fdtd_config_is_unchanged(asset_store) -> None:
    oi = _synthetic_oi()
    base = sensor_set(sensor_create("monochrome", asset_store=asset_store), "size", oi.data["photons"].shape[:2])
    base = sensor_set(base, "noise flag", 0)
    base = sensor_set(base, "integration time", 0.01)
    disabled = sensor_set(base.clone(), "fdtd sensor", {"enabled": False})

    base_result = sensor_compute(base, oi, seed=0)
    disabled_result = sensor_compute(disabled, oi, seed=0)

    assert np.allclose(sensor_get(base_result, "volts"), sensor_get(disabled_result, "volts"))


def test_sensor_compute_with_fdtd_lut_changes_sensor_volts(asset_store, tmp_path: Path) -> None:
    lut = fdtd_sensor_lut_load(_write_synthetic_lut(tmp_path))
    oi = _synthetic_oi()
    sensor = sensor_set(sensor_create("monochrome", asset_store=asset_store), "size", oi.data["photons"].shape[:2])
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "integration time", 0.01)
    fdtd_sensor = sensor_attach_fdtd_lut(sensor, lut, mode="field+crosstalk", crosstalk_strength=1.0)

    base_result = sensor_compute(sensor, oi, seed=0)
    fdtd_result = sensor_compute(fdtd_sensor, oi, seed=0)

    base_volts = np.asarray(sensor_get(base_result, "volts"), dtype=float)
    fdtd_volts = np.asarray(sensor_get(fdtd_result, "volts"), dtype=float)
    assert fdtd_volts.shape == base_volts.shape
    assert not np.allclose(fdtd_volts, base_volts)
    assert float(np.mean(fdtd_volts[:2, :2])) < float(np.mean(base_volts[:2, :2]))


def test_real_fdtd_default_lut_smoke_if_available(asset_store) -> None:
    path = fdtd_sensor_default_lut_path()
    if path is None:
        pytest.skip("external FDTD LUT repo is not available")
    lut = fdtd_sensor_lut_load(path)
    validation = fdtd_sensor_lut_validate(lut)

    assert validation["ok"] is True
    assert validation["n_wavelengths"] >= 1
    assert validation["n_summary_rows"] >= 1

    oi = _synthetic_oi(rows=12, cols=12)
    sensor = sensor_set(sensor_create("monochrome", asset_store=asset_store), "size", oi.data["photons"].shape[:2])
    sensor = sensor_set(sensor, "noise flag", 0)
    sensor = sensor_set(sensor, "integration time", 0.01)
    fdtd_sensor = sensor_attach_fdtd_lut(sensor, lut, mode="qe+field")

    base_result = sensor_compute(sensor, oi, seed=0)
    fdtd_result = sensor_compute(fdtd_sensor, oi, seed=0)

    base_volts = np.asarray(sensor_get(base_result, "volts"), dtype=float)
    fdtd_volts = np.asarray(sensor_get(fdtd_result, "volts"), dtype=float)
    assert fdtd_volts.shape == base_volts.shape
    assert np.all(np.isfinite(fdtd_volts))
