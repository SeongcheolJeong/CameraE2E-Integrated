from __future__ import annotations

import numpy as np
import pytest

from pyisetcam import (
    airyDisk,
    blackbody,
    fisePlotDefaults,
    hist2d,
    ieFigureFormat,
    ieFigureResize,
    ieFormatFigure,
    ieHistImage,
    iePlaneFromVectors,
    iePlot,
    iePlotJitter,
    iePlotSet,
    iePlotShadeBackground,
    ieShape,
    ieXYZFromEnergy,
    identityLine,
    ipPlot,
    plotContrastHistogram,
    plotDisplayColor,
    plotDisplayGamut,
    plotDisplayLine,
    plotDisplaySPD,
    plotEtendueRatio,
    plotGaussianSpectrum,
    plotMetrics,
    plotML,
    plotNormal,
    plotOI,
    plotPixelSNR,
    plotRadiance,
    plotReflectance,
    plotSensorEtendue,
    plotSensorSNR,
    plotSetUpWindow,
    plotSpectrumLocus,
    plotTextString,
    scatplot,
    ip_create,
    ip_get,
    ip_set,
    pixel_snr,
    oiPlot,
    oi_compute,
    oi_create,
    oi_get,
    oi_set,
    plotScene,
    plotSensor,
    plotSensorFFT,
    plotSensorHist,
    scenePlot,
    sensorPlot,
    sensorPlotHist,
    sensorPlotLine,
    scene_create,
    scene_get,
    scene_set,
    sensorPlotColor,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_snr,
    sensor_set,
    ml_radiance,
    mlens_create,
    mlens_get,
    vc_get_roi_data,
    wvf_compute,
    wvf_create,
    wvf_get,
    wvfPlot,
    wvf_to_oi,
    xaxisLine,
    xyz_to_lab,
    xyz_to_luv,
    xyz_to_srgb,
    yaxisLine,
)
from pyisetcam.exceptions import UnsupportedOptionError


def _expected_shot_noise_map(electrons: np.ndarray, conversion_gain: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    electron_image = np.clip(np.asarray(electrons, dtype=float), 0.0, None)
    electron_noise = np.sqrt(electron_image) * rng.standard_normal(electron_image.shape)
    low_count = electron_image < 25.0
    if np.any(low_count):
        poisson_counts = rng.poisson(electron_image[low_count])
        electron_noise[low_count] = poisson_counts - electron_image[low_count]
    return conversion_gain * electron_noise


def test_plot_scene_radiance_photons_roi_and_chromaticity(asset_store) -> None:
    scene = scene_create("uniform ee", 4, asset_store=asset_store)
    scene = scene_set(scene, "wave", np.array([500.0, 600.0], dtype=float))
    scene = scene_set(
        scene,
        "photons",
        np.array(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ],
            dtype=float,
        ),
    )
    roi = np.array([1, 1, 1, 1], dtype=int)

    radiance_udata, handle = plotScene(scene, "radiance photons roi", roi, asset_store=asset_store)
    chromaticity_udata, chroma_handle = plotScene(scene, "chromaticity", roi, asset_store=asset_store)

    expected_photons = np.mean(np.asarray(scene_get(scene, "roi photons", roi, asset_store=asset_store), dtype=float), axis=0)
    expected_xy = np.asarray(scene_get(scene, "chromaticity", roi, asset_store=asset_store), dtype=float)

    assert handle is None
    assert chroma_handle is None
    assert np.allclose(radiance_udata["wave"], np.array([500.0, 600.0], dtype=float))
    assert np.allclose(radiance_udata["photons"], expected_photons)
    assert np.allclose(chromaticity_udata["x"], expected_xy[:, 0])
    assert np.allclose(chromaticity_udata["y"], expected_xy[:, 1])


def test_plot_scene_illuminant_energy(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)

    udata, handle = plotScene(scene, "illuminant energy")

    assert handle is None
    assert udata["comment"] == "D65.mat"
    assert np.allclose(udata["wave"], np.asarray(scene_get(scene, "wave"), dtype=float))
    assert np.allclose(udata["energy"], np.asarray(scene_get(scene, "illuminant energy"), dtype=float))


def test_plot_scene_illuminant_image_for_spectral_illuminant(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)

    udata, handle = plotScene(scene, "illuminant image", asset_store=asset_store)

    wave = np.asarray(scene_get(scene, "wave"), dtype=float)
    energy = np.asarray(scene_get(scene, "illuminant energy", asset_store=asset_store), dtype=float)
    rows, cols = scene_get(scene, "size")
    expected_energy = np.broadcast_to(energy.reshape(1, 1, -1), (rows, cols, wave.size))
    expected_srgb = xyz_to_srgb(np.asarray(ieXYZFromEnergy(expected_energy, wave, asset_store=asset_store), dtype=float))

    assert handle is None
    assert udata["srgb"].shape == (rows, cols, 3)
    assert np.allclose(udata["srgb"], expected_srgb)
    assert np.allclose(udata["srgb"], udata["srgb"][0, 0][None, None, :])


def test_plot_scene_illuminant_image_for_spatial_spectral_illuminant(asset_store) -> None:
    wave = np.array([500.0, 600.0], dtype=float)
    scene = scene_create("uniform ee", 4, asset_store=asset_store)
    scene = scene_set(scene, "wave", wave)
    scene = scene_set(scene, "photons", np.ones((2, 3, wave.size), dtype=float))
    illuminant_energy = np.array(
        [
            [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
            [[0.9, 0.1], [0.1, 0.9], [0.4, 0.6]],
        ],
        dtype=float,
    )
    scene = scene_set(scene, "illuminant energy", illuminant_energy)

    udata, handle = plotScene(scene, "illuminant image", asset_store=asset_store)

    expected_srgb = xyz_to_srgb(np.asarray(ieXYZFromEnergy(illuminant_energy, wave, asset_store=asset_store), dtype=float))

    assert handle is None
    assert udata["srgb"].shape == (2, 3, 3)
    assert np.allclose(udata["srgb"], expected_srgb)
    assert np.max(np.abs(udata["srgb"][0, 0] - udata["srgb"][1, 1])) > 1e-6


def test_oi_plot_roi_and_line_data(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=float,
    )
    oi = oi_set(oi, "photons", photons)
    roi = np.array([1, 1, 2, 1], dtype=int)
    line = np.array([1, 1], dtype=int)

    roi_udata, roi_handle = oiPlot(oi, "irradiance photons roi", roi)
    line_udata, line_handle = oiPlot(oi, "illuminance hline", line)

    expected_y = np.mean(np.asarray(oi_get(oi, "roi photons", roi), dtype=float), axis=0)
    expected_line = oi_get(oi, "illuminance hline", line)

    assert roi_handle is None
    assert line_handle is None
    assert np.allclose(roi_udata["x"], np.array([500.0, 600.0], dtype=float))
    assert np.allclose(roi_udata["y"], expected_y)
    assert np.array_equal(roi_udata["roiLocs"], roi)
    assert np.allclose(line_udata["pos"], np.asarray(expected_line["pos"], dtype=float))
    assert np.allclose(line_udata["data"], np.asarray(expected_line["data"], dtype=float))
    assert np.array_equal(line_udata["roiLocs"], line)


def test_oi_plot_psf_and_axes_for_custom_psf_oi(asset_store) -> None:
    oi = oi_create("psf", asset_store=asset_store)

    psf_udata, psf_handle = oiPlot(oi, "psf550")
    xaxis_udata, xaxis_handle = oiPlot(oi, "psfxaxis", None, 550, "um")
    yaxis_udata, yaxis_handle = oiPlot(oi, "psfyaxis", None, 550, "um")

    assert psf_handle is None
    assert xaxis_handle is None
    assert yaxis_handle is None
    assert psf_udata["units"] == "um"
    assert np.isclose(psf_udata["wave"], 550.0)
    assert psf_udata["psf"].shape == psf_udata["x"].shape == psf_udata["y"].shape
    assert psf_udata["psf"].ndim == 2
    assert np.isclose(np.sum(psf_udata["psf"]), 1.0, rtol=1e-4)
    assert np.allclose(xaxis_udata["samp"], np.asarray(psf_udata["x"][0, :], dtype=float))
    assert np.allclose(yaxis_udata["samp"], np.asarray(psf_udata["y"][:, 0], dtype=float))
    assert xaxis_udata["data"].shape == xaxis_udata["samp"].shape
    assert yaxis_udata["data"].shape == yaxis_udata["samp"].shape


def test_oi_plot_psf_for_diffraction_limited_oi(asset_store) -> None:
    oi = oi_create("diffraction limited", asset_store=asset_store)

    udata, handle = oiPlot(oi, "psf", None, 550, "um")

    assert handle is None
    assert np.isclose(udata["wave"], 550.0)
    assert udata["units"] == "um"
    assert udata["psf"].shape == (50, 50)
    assert udata["x"].shape == (50, 50)
    assert udata["y"].shape == (50, 50)
    assert np.max(udata["psf"]) > 0.0


def test_oi_plot_psf_for_computed_wvf_oi(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create("wvf", asset_store=asset_store), scene, crop=True)

    udata, handle = oiPlot(oi, "psf", None, 550, "um")

    assert handle is None
    assert np.isclose(udata["wave"], 550.0)
    assert udata["psf"].shape == udata["x"].shape == udata["y"].shape
    assert udata["psf"].ndim == 2
    assert np.max(udata["psf"]) > 0.0


def test_oi_plot_lswavelength_and_otfwavelength_for_diffraction_limited_oi(asset_store) -> None:
    oi = oi_create("diffraction limited", asset_store=asset_store)

    ls_udata, ls_handle = oiPlot(oi, "ls wavelength")
    otf_udata, otf_handle = oiPlot(oi, "otf wavelength")

    assert ls_handle is None
    assert otf_handle is None
    assert ls_udata["x"].ndim == 1
    assert ls_udata["wavelength"].ndim == 1
    assert ls_udata["lsWave"].shape == (ls_udata["wavelength"].size, ls_udata["x"].size)
    assert np.all(ls_udata["lsWave"] >= 0.0)
    assert otf_udata["fSupport"].ndim == 1
    assert otf_udata["wavelength"].ndim == 1
    assert otf_udata["otf"].shape == (otf_udata["fSupport"].size, otf_udata["wavelength"].size)
    assert np.all(otf_udata["otf"] >= 0.0)


def test_oi_plot_otfwavelength_for_shift_invariant_oi(asset_store) -> None:
    scene = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    oi = oi_compute(oi_create("shift invariant", asset_store=asset_store), scene, crop=True)

    udata, handle = oiPlot(oi, "mtf wavelength")

    assert handle is None
    assert udata["fSupport"].ndim == 1
    assert udata["wavelength"].ndim == 1
    assert udata["otf"].shape == (udata["fSupport"].size, udata["wavelength"].size)
    assert np.all(udata["otf"] >= 0.0)


def test_wvf_plot_psf_views() -> None:
    wvf = wvf_compute(wvf_create(wave=np.array([550.0], dtype=float)))

    psf_udata, psf_handle = wvfPlot(wvf, "psf normalized", "unit", "um", "wave", 550.0, "plot range", 10.0)
    line_udata, line_handle = wvfPlot(wvf, "1d psf normalized", "unit", "um", "wave", 550.0, "plot range", 10.0)
    xaxis_udata, xaxis_handle = wvfPlot(wvf, "psfxaxis", "unit", "um", "wave", 550.0, "plot range", 10.0)
    yaxis_udata, yaxis_handle = wvfPlot(wvf, "psfyaxis", "unit", "um", "wave", 550.0, "plot range", 10.0)

    assert psf_handle is None
    assert line_handle is None
    assert xaxis_handle is None
    assert yaxis_handle is None
    assert np.isclose(psf_udata["wave"], 550.0)
    assert psf_udata["unit"] == "um"
    assert psf_udata["z"].ndim == 2
    assert psf_udata["x"].ndim == 1
    assert psf_udata["y"].ndim == 1
    assert psf_udata["z"].shape == (psf_udata["y"].size, psf_udata["x"].size)
    assert np.isclose(float(np.max(psf_udata["z"])), 1.0)
    assert np.isclose(float(np.max(line_udata["y"])), 1.0)
    expected_index = np.abs(xaxis_udata["samp"]) < 10.0
    assert np.allclose(xaxis_udata["samp"][expected_index], line_udata["x"])
    assert xaxis_udata["data"].shape == xaxis_udata["samp"].shape
    assert yaxis_udata["data"].shape == yaxis_udata["samp"].shape
    assert np.asarray(wvf_get(wvf, "psf")).ndim == 3
    assert np.asarray(wvf_get(wvf, "psf", 550.0)).ndim == 2


def test_airy_disk_helper_and_overlay_payloads() -> None:
    radius_um, image = airyDisk(550.0, 3.0, "units", "um", return_image=True)

    assert radius_um == pytest.approx(1.22 * 3.0 * 550.0e-3)
    assert image is not None
    assert image["data"].ndim == 2
    assert image["data"].shape[0] == image["data"].shape[1]
    assert image["x"].ndim == 1
    assert image["y"].ndim == 1

    wvf = wvf_compute(wvf_create(wave=np.array([550.0], dtype=float)))
    expected_radius = float(airyDisk(550.0, float(wvf_get(wvf, "fnumber")), "units", "um"))
    psf_udata, _ = wvfPlot(wvf, "psf", "unit", "um", "wave", 550.0, "plot range", 10.0, "airy disk", True)
    xaxis_udata, _ = wvfPlot(wvf, "psfxaxis", "unit", "um", "wave", 550.0, "plot range", 10.0, "airy disk", True)

    assert psf_udata["airyDisk"] is True
    assert psf_udata["airyDiskRadius"] == pytest.approx(expected_radius)
    assert psf_udata["airyDiskDiameter"] == pytest.approx(expected_radius * 2.0)
    assert psf_udata["airyDiskCircle"]["x"].shape == psf_udata["airyDiskCircle"]["y"].shape
    assert xaxis_udata["airyDisk"] is True
    assert xaxis_udata["airyDiskRadius"] == pytest.approx(expected_radius)

    oi = wvf_to_oi(wvf)
    oi_udata, _ = oiPlot(oi, "psf", None, 550.0, "um", "airy disk", True)
    assert oi_udata["airyDisk"] is True
    assert oi_udata["airyDiskRadius"] > 0.0


def test_wvf_plot_pupil_and_wavefront_images() -> None:
    wvf = wvf_compute(
        wvf_create(
            wave=np.array([550.0], dtype=float),
            zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.12], dtype=float),
        )
    )

    amp_udata, amp_handle = wvfPlot(wvf, "image pupil amp", "unit", "mm", "wave", 550.0, "plot range", 1.5)
    phase_udata, phase_handle = wvfPlot(wvf, "image pupil phase", "unit", "mm", "wave", 550.0, "plot range", 1.5)
    wavefront_udata, wavefront_handle = wvfPlot(
        wvf,
        "image wavefront aberrations",
        "unit",
        "mm",
        "wave",
        550.0,
        "plot range",
        1.5,
    )

    assert amp_handle is None
    assert phase_handle is None
    assert wavefront_handle is None
    assert amp_udata["z"].shape == phase_udata["z"].shape == wavefront_udata["z"].shape
    assert amp_udata["z"].ndim == 2
    assert np.all(amp_udata["z"] >= 0.0)
    assert np.max(np.abs(phase_udata["z"])) <= np.pi + 1e-12
    assert np.max(np.abs(wavefront_udata["z"])) > 0.0
    assert np.asarray(wvf_get(wvf, "pupil function", 550.0)).ndim == 2
    assert np.asarray(wvf_get(wvf, "wavefront aberrations", 550.0)).ndim == 2


def test_wvf_plot_angle_and_otf_views() -> None:
    wvf = wvf_compute(
        wvf_create(
            wave=np.array([550.0], dtype=float),
            zcoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.08], dtype=float),
        )
    )

    psf_angle_udata, psf_angle_handle = wvfPlot(
        wvf, "psf angle", "unit", "min", "wave", 550.0, "plot range", 1.0
    )
    image_psf_angle_udata, image_psf_angle_handle = wvfPlot(
        wvf, "image psf angle", "unit", "min", "wave", 550.0, "plot range", 1.0
    )
    line_angle_udata, line_angle_handle = wvfPlot(
        wvf, "1d psf angle normalized", "unit", "min", "wave", 550.0, "plot range", 1.0
    )
    otf_udata, otf_handle = wvfPlot(wvf, "otf", "unit", "mm", "wave", 550.0, "plot range", 200.0)
    otf_1d_udata, otf_1d_handle = wvfPlot(wvf, "1d otf", "unit", "mm", "wave", 550.0, "plot range", 200.0)
    otf_1d_angle_udata, otf_1d_angle_handle = wvfPlot(
        wvf, "1d otf angle", "unit", "deg", "wave", 550.0, "plot range", 60.0
    )

    assert psf_angle_handle is None
    assert image_psf_angle_handle is None
    assert line_angle_handle is None
    assert otf_handle is None
    assert otf_1d_handle is None
    assert otf_1d_angle_handle is None
    assert psf_angle_udata["z"].shape == image_psf_angle_udata["z"].shape
    assert psf_angle_udata["x"].ndim == 1
    assert np.isclose(float(np.max(line_angle_udata["y"])), 1.0)
    assert otf_udata["otf"].ndim == 2
    assert otf_udata["otf"].shape == (otf_udata["fy"].size, otf_udata["fx"].size)
    assert otf_1d_udata["fx"].ndim == 1
    assert otf_1d_udata["fy"].shape == otf_1d_udata["fx"].shape
    assert otf_1d_udata["otf"].shape == (otf_1d_udata["fy"].size, otf_1d_udata["fx"].size)
    assert otf_1d_angle_udata["fx"].ndim == 1
    assert otf_1d_angle_udata["fy"].shape == otf_1d_angle_udata["fx"].shape
    assert otf_1d_angle_udata["otf"].shape == (otf_1d_angle_udata["fy"].size, otf_1d_angle_udata["fx"].size)
    assert np.asarray(wvf_get(wvf, "otf", 550.0)).ndim == 2


def test_plot_sensor_line_data(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(
        sensor,
        "dv",
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=float,
        ),
    )

    volts_udata, volts_handle = plotSensor(sensor, "volts hline", np.array([1, 2], dtype=int))
    electrons_udata, electrons_handle = plotSensor(sensor, "electrons hline", np.array([2, 2], dtype=int))
    dv_udata, dv_handle = plotSensor(sensor, "dv vline", np.array([2, 1], dtype=int))

    expected_volts = sensor_get(sensor, "hline volts", 2)
    expected_electrons = sensor_get(sensor, "hline electrons", 2)
    expected_dv = sensor_get(sensor, "vline dv", 2)

    assert volts_handle is None
    assert electrons_handle is None
    assert dv_handle is None
    assert np.array_equal(volts_udata["xy"], np.array([1, 2], dtype=int))
    assert volts_udata["ori"] == "h"
    assert volts_udata["dataType"] == "volts"
    assert np.array_equal(volts_udata["pixColor"], np.array([1], dtype=int))
    assert volts_udata["filterPlotColors"] == sensor_get(sensor, "filter plot colors")
    assert volts_udata["xLabel"] == "Position (um)"
    assert volts_udata["yLabel"] == "volts"
    assert volts_udata["titleString"] == "Horizontal line 2"
    assert np.allclose(volts_udata["data"][0], np.asarray(expected_volts["data"][0], dtype=float))
    assert np.allclose(volts_udata["pos"][0], 1e6 * np.asarray(expected_volts["pos"][0], dtype=float))
    assert np.array_equal(electrons_udata["pixColor"], np.array([1], dtype=int))
    assert electrons_udata["yLabel"] == "electrons"
    assert electrons_udata["titleString"] == "Horizontal line 2"
    assert np.allclose(electrons_udata["data"][0], np.asarray(expected_electrons["data"][0], dtype=float))
    assert np.allclose(electrons_udata["pixPos"][0], 1e6 * np.asarray(expected_electrons["pixPos"][0], dtype=float))
    assert dv_udata["ori"] == "v"
    assert dv_udata["dataType"] == "dv"
    assert np.array_equal(dv_udata["pixColor"], np.array([1], dtype=int))
    assert dv_udata["xLabel"] == "Position (um)"
    assert dv_udata["yLabel"] == "digital value"
    assert dv_udata["titleString"] == "Vertical line 2"
    assert np.allclose(dv_udata["data"][0], np.asarray(expected_dv["data"][0], dtype=float))


def test_sensor_plot_line_matches_matlab_style_line_contract(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 4)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=float,
        ),
    )

    fig_num, udata = sensorPlotLine(sensor, "h", "volts", "space", np.array([1, 2], dtype=int))
    fft_fig_num, fft_udata = sensorPlotLine(sensor, "h", "volts", "fft", np.array([1, 2], dtype=int))
    expected_line = sensor_get(sensor, "hline volts", 2)
    expected_fft, _ = plotSensorFFT(sensor, "h", "volts", np.array([1, 2], dtype=int))

    assert fig_num is None
    assert fft_fig_num is None
    assert np.allclose(udata["pixPos"], 1e6 * np.asarray(expected_line["pos"][0], dtype=float))
    assert np.allclose(udata["pixData"], np.asarray(expected_line["data"][0], dtype=float))
    assert np.allclose(fft_udata["freq"], np.asarray(expected_fft["cpd"], dtype=float))
    assert np.allclose(fft_udata["amp"], np.asarray(expected_fft["ampPlot"], dtype=float))
    assert np.isclose(float(fft_udata["mean"]), float(expected_fft["mean"]))
    assert np.isclose(float(fft_udata["peakContrast"]), float(expected_fft["peakContrast"]))


def test_plot_sensor_two_lines_wrapper(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=float,
        ),
    )

    udata, handle = plotSensor(sensor, "volts hline", np.array([1, 1], dtype=int), "two lines", True)

    first_line = sensor_get(sensor, "hline volts", 1)
    second_line = sensor_get(sensor, "hline volts", 2)
    expected_pos: list[np.ndarray] = []
    expected_data: list[np.ndarray] = []
    expected_color: list[int] = []
    for profile in (first_line, second_line):
        for color_index, (positions, values) in enumerate(zip(profile["pos"], profile["data"]), start=1):
            values_array = np.asarray(values, dtype=float)
            if values_array.size == 0:
                continue
            expected_pos.append(np.asarray(positions, dtype=float))
            expected_data.append(values_array)
            expected_color.append(color_index)

    assert handle is None
    assert np.array_equal(udata["xy"], np.array([1, 1], dtype=int))
    assert np.array_equal(udata["xy2"], np.array([1, 2], dtype=int))
    assert udata["ori"] == "h"
    assert udata["dataType"] == "volts"
    assert udata["filterPlotColors"] == "rgb"
    assert udata["titleString"] == "Horizontal line 1"
    assert udata["xLabel"] == "Position (um)"
    assert udata["yLabel"] == "volts"
    assert np.array_equal(udata["pixColor"], np.array(expected_color, dtype=int))
    assert len(udata["pixPos"]) == len(expected_pos)
    assert len(udata["pixData"]) == len(expected_data)
    for actual, expected in zip(udata["pixPos"], expected_pos):
        assert np.allclose(actual, expected)
    for actual, expected in zip(udata["pixData"], expected_data):
        assert np.allclose(actual, expected)


def test_plot_sensor_chromaticity_wrapper(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [2.0, 4.0, 2.0, 4.0],
                [6.0, 2.0, 6.0, 2.0],
                [2.0, 4.0, 2.0, 4.0],
                [6.0, 2.0, 6.0, 2.0],
            ],
            dtype=float,
        ),
    )
    roi = np.array([1, 1, 3, 3], dtype=int)

    chroma_udata, chroma_handle = plotSensor(sensor, "chromaticity", roi)

    expected_rg = np.asarray(sensor_get(sensor, "chromaticity", roi), dtype=float)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    expected_locus = spectral_qe[:, :2] / np.sum(spectral_qe, axis=1, keepdims=True)

    assert chroma_handle is None
    assert np.array_equal(chroma_udata["rect"], roi)
    assert np.allclose(chroma_udata["rg"], expected_rg)
    assert np.allclose(chroma_udata["spectrumlocus"], expected_locus)
    assert chroma_udata["xLabel"] == "r-chromaticity"
    assert chroma_udata["yLabel"] == "g-chromaticity"
    assert chroma_udata["titleString"] == "rg sensor chromaticity"


def test_ip_plot_line_chromaticity_and_luminance(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[1.0, 10.0, 2.0], [2.0, 20.0, 3.0], [3.0, 30.0, 4.0]],
            [[4.0, 40.0, 5.0], [5.0, 50.0, 6.0], [6.0, 60.0, 7.0]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)
    ip.data["xyz"] = xyz

    chroma_roi = np.array([1, 1, 1, 1], dtype=int)
    horizontal_xy = np.array([1, 2], dtype=int)
    vertical_xy = np.array([2, 1], dtype=int)

    line_udata, line_handle = ipPlot(ip, "horizontal line", horizontal_xy)
    chroma_udata, chroma_handle = ipPlot(ip, "chromaticity", chroma_roi)
    luminance_udata, luminance_handle = ipPlot(ip, "vertical line luminance", vertical_xy)

    expected_chroma = np.asarray(ip_get(ip, "chromaticity", chroma_roi), dtype=float)
    expected_luminance = np.asarray(ip_get(ip, "data luminance"), dtype=float)[:, 1]

    assert line_handle is None
    assert chroma_handle is None
    assert luminance_handle is None
    assert np.array_equal(line_udata["xy"], horizontal_xy)
    assert line_udata["ori"] == "h"
    assert np.allclose(line_udata["pos"], np.array([1.0, 2.0, 3.0], dtype=float))
    assert np.allclose(line_udata["values"], result[1, :, :])
    assert np.allclose(chroma_udata["x"], expected_chroma[:, 0])
    assert np.allclose(chroma_udata["y"], expected_chroma[:, 1])
    assert np.array_equal(chroma_udata["roiLocs"], chroma_roi)
    assert np.array_equal(luminance_udata["xy"], vertical_xy)
    assert luminance_udata["ori"] == "v"
    assert np.allclose(luminance_udata["pos"], np.array([1.0, 2.0], dtype=float))
    assert np.allclose(luminance_udata["data"], expected_luminance)


def test_plot_sensor_histogram_data(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(
        sensor,
        "dv",
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=float,
        ),
    )
    roi = np.array([1, 1, 1, 1], dtype=int)

    volts_udata, volts_handle = plotSensor(sensor, "volts histogram", roi)
    electrons_udata, electrons_handle = plotSensor(sensor, "electrons hist", roi)
    dv_udata, dv_handle = plotSensor(sensor, "dv hist", roi)

    expected_volts = np.asarray(vc_get_roi_data(sensor, roi, "volts"), dtype=float)
    expected_electrons = np.asarray(vc_get_roi_data(sensor, roi, "electrons"), dtype=float)
    expected_dv = np.asarray(vc_get_roi_data(sensor, roi, "dv"), dtype=float)

    assert volts_handle is None
    assert electrons_handle is None
    assert dv_handle is None
    assert np.array_equal(volts_udata["rect"], roi)
    assert volts_udata["unitType"] == "volts"
    assert volts_udata["filterPlotColors"] == "k"
    assert volts_udata["xLabel"] == "Volts"
    assert volts_udata["yLabel"] == "Count"
    assert np.allclose(volts_udata["data"], expected_volts)
    assert electrons_udata["xLabel"] == "Electrons"
    assert electrons_udata["yLabel"] == "Count"
    assert np.allclose(electrons_udata["data"], expected_electrons)
    assert dv_udata["xLabel"] == "Digital value"
    assert dv_udata["yLabel"] == "Count"
    assert np.allclose(dv_udata["data"], expected_dv)


def test_plot_sensor_capture_selection(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    volts = np.array(
        [
            [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
            [[0.4, 1.4], [0.5, 1.5], [0.6, 1.6]],
        ],
        dtype=float,
    )
    dv = np.array(
        [
            [[10.0, 110.0], [20.0, 120.0], [30.0, 130.0]],
            [[40.0, 140.0], [50.0, 150.0], [60.0, 160.0]],
        ],
        dtype=float,
    )
    sensor = sensor_set(sensor, "volts", volts)
    sensor = sensor_set(sensor, "dv", dv)
    sensor.fields["integration_time"] = np.array([0.01, 0.02], dtype=float)
    roi = np.array([1, 1, 2, 1], dtype=int)

    line_udata, line_handle = plotSensor(sensor, "volts hline", np.array([1, 2], dtype=int), "capture", 2)
    hist_udata, hist_handle = plotSensor(sensor, "dv hist", roi, "capture", 2)

    selected_sensor = sensor.clone()
    selected_sensor.data["volts"] = volts[:, :, 1]
    selected_sensor.data["dv"] = dv[:, :, 1]
    selected_sensor.fields["integration_time"] = float(sensor.fields["integration_time"][1])
    expected_line = sensor_get(selected_sensor, "hline volts", 2)
    expected_hist = np.asarray(vc_get_roi_data(selected_sensor, roi, "dv"), dtype=float)

    assert line_handle is None
    assert hist_handle is None
    assert sensor_get(sensor, "n captures") == 2
    assert np.allclose(line_udata["data"][0], np.asarray(expected_line["data"][0], dtype=float))
    assert np.allclose(line_udata["pos"][0], 1e6 * np.asarray(expected_line["pos"][0], dtype=float))
    assert np.allclose(hist_udata["data"], expected_hist)
    assert np.isclose(float(sensor.fields["integration_time"][1]), 0.02)


def test_plot_sensor_capture_selection_rejects_invalid_index(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(sensor, "volts", np.ones((2, 2, 2), dtype=float))

    with pytest.raises(IndexError):
        plotSensor(sensor, "volts hline", np.array([1, 1], dtype=int), "capture", 3)


def test_ip_plot_rgb_histogram_rgb3d_and_luminance(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[1.0, 10.0, 2.0], [2.0, 20.0, 3.0], [3.0, 30.0, 4.0]],
            [[4.0, 40.0, 5.0], [5.0, 50.0, 6.0], [6.0, 60.0, 7.0]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)
    ip.data["xyz"] = xyz
    roi = np.array([1, 1, 1, 1], dtype=int)

    rgb_hist_udata, rgb_hist_handle = ipPlot(ip, "rgbhistogram", roi)
    rgb3d_udata, rgb3d_handle = ipPlot(ip, "rgb3d", roi)
    luminance_udata, luminance_handle = ipPlot(ip, "luminance", roi)

    expected_rgb = np.asarray(ip_get(ip, "roidata", roi), dtype=float)
    expected_xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    expected_luminance = expected_xyz[:, 1]

    assert rgb_hist_handle is None
    assert rgb3d_handle is None
    assert luminance_handle is None
    assert np.array_equal(rgb_hist_udata["rect"], roi)
    assert np.allclose(rgb_hist_udata["RGB"], expected_rgb)
    assert np.allclose(rgb_hist_udata["meanRGB"], np.mean(expected_rgb, axis=0))
    assert np.allclose(rgb3d_udata["RGB"], expected_rgb)
    assert np.allclose(luminance_udata["luminance"], expected_luminance)
    assert np.isclose(luminance_udata["meanL"], float(np.mean(expected_luminance)))
    assert np.isclose(luminance_udata["stdLum"], float(np.std(expected_luminance)))


def test_ip_plot_cielab_and_cieluv(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[1.0, 10.0, 2.0], [2.0, 20.0, 3.0]],
            [[4.0, 40.0, 5.0], [5.0, 50.0, 6.0]],
        ],
        dtype=float,
    )
    white_point = np.array([95.047, 100.0, 108.883], dtype=float)
    ip = ip_set(ip, "result", result)
    ip.data["xyz"] = xyz
    ip = ip_set(ip, "data white point", white_point)
    roi = np.array([1, 1, 1, 1], dtype=int)

    lab_udata, lab_handle = ipPlot(ip, "cielab", roi)
    luv_udata, luv_handle = ipPlot(ip, "cieluv", roi)

    expected_xyz = np.asarray(ip_get(ip, "roixyz", roi), dtype=float)
    expected_lab = np.asarray(xyz_to_lab(expected_xyz, white_point), dtype=float)
    expected_luv = np.asarray(xyz_to_luv(expected_xyz, white_point), dtype=float)

    assert lab_handle is None
    assert luv_handle is None
    assert np.array_equal(lab_udata["rect"], roi)
    assert np.allclose(lab_udata["whitePoint"], white_point)
    assert np.allclose(lab_udata["LAB"], expected_lab)
    assert np.allclose(lab_udata["meanLAB"], np.mean(expected_lab, axis=0))
    assert np.allclose(luv_udata["whitePoint"], white_point)
    assert np.allclose(luv_udata["LUV"], expected_luv)
    assert np.allclose(luv_udata["meanLUV"], np.mean(expected_luv, axis=0))


def test_pixel_and_sensor_snr_helpers_and_plot_wrappers(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 2.0e-4
    sensor.fields["pixel"]["voltage_swing"] = 1.2
    sensor.fields["pixel"]["read_noise_v"] = 4.0e-4
    sensor.fields["pixel"]["dsnu_sigma_v"] = 2.0e-4
    sensor.fields["pixel"]["prnu_sigma"] = 0.01
    volts = np.array([1.0e-3, 1.0e-2], dtype=float)

    pixel_curve, pixel_volts, pixel_shot, pixel_read = pixel_snr(sensor, volts)
    sensor_curve, sensor_volts, sensor_shot, sensor_read, sensor_dsnu, sensor_prnu = sensor_snr(sensor, volts)

    conv_gain = 2.0e-4
    read_sd = 4.0e-4 / conv_gain
    dsnu_sd = 2.0e-4 / conv_gain
    prnu_gain_sd = 0.01
    shot_sd = np.sqrt(volts / conv_gain)
    signal_power = (volts / conv_gain) ** 2
    expected_pixel = 10.0 * np.log10(signal_power / (read_sd**2 + shot_sd**2))
    expected_sensor = 10.0 * np.log10(signal_power / (shot_sd**2 + read_sd**2 + dsnu_sd**2 + (prnu_gain_sd * (volts / conv_gain)) ** 2))

    pixel_udata, pixel_handle = plotSensor(sensor, "pixel snr")
    sensor_udata, sensor_handle = plotSensor(sensor, "sensor snr")

    assert pixel_handle is None
    assert sensor_handle is None
    assert np.allclose(pixel_volts, volts)
    assert np.allclose(sensor_volts, volts)
    assert np.allclose(pixel_curve, expected_pixel)
    assert np.allclose(sensor_curve, expected_sensor)
    assert np.allclose(pixel_shot, 10.0 * np.log10(signal_power / (shot_sd**2)))
    assert np.allclose(sensor_shot, 10.0 * np.log10(signal_power / (shot_sd**2)))
    assert np.allclose(pixel_read, 10.0 * np.log10(signal_power / (read_sd**2)))
    assert np.allclose(sensor_read, 10.0 * np.log10(signal_power / (read_sd**2)))
    assert np.allclose(sensor_dsnu, 10.0 * np.log10(signal_power / (dsnu_sd**2)))
    assert np.allclose(sensor_prnu, 10.0 * np.log10(signal_power / ((prnu_gain_sd * (volts / conv_gain)) ** 2)))
    assert np.allclose(pixel_udata["snr"], pixel_snr(sensor)[0])
    assert np.allclose(pixel_udata["volts"], pixel_snr(sensor)[1])
    assert pixel_udata["xLabel"] == "Signal (V)"
    assert pixel_udata["yLabel"] == "SNR (db)"
    assert pixel_udata["titleString"] == "Pixel SNR over response range"
    assert pixel_udata["legend"] == ["Total pixel SNR", "Shot noise SNR", "Read noise SNR"]
    assert np.allclose(sensor_udata["snr"], sensor_snr(sensor)[0])
    assert np.allclose(sensor_udata["volts"], sensor_snr(sensor)[1])
    assert sensor_udata["xLabel"] == "Signal (V)"
    assert sensor_udata["yLabel"] == "SNR (db)"
    assert sensor_udata["titleString"] == "Sensor SNR over response range"
    assert sensor_udata["legend"] == ["Total", "Shot", "Read", "DSNU", "PRNU"]


def test_plot_sensor_spectral_wrappers(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "wave", np.array([500.0, 600.0], dtype=float))
    sensor = sensor_set(
        sensor,
        "color filters",
        np.array(
            [
                [0.9, 0.5, 0.1],
                [0.6, 0.2, 0.8],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(sensor, "pixel spectral qe", np.array([0.5, 0.25], dtype=float))
    sensor = sensor_set(sensor, "ir filter", np.array([0.8, 0.4], dtype=float))

    color_udata, color_handle = plotSensor(sensor, "color filters")
    ir_udata, ir_handle = plotSensor(sensor, "ir filter")
    pixel_qe_udata, pixel_qe_handle = plotSensor(sensor, "pixel spectral qe")
    pixel_sr_udata, pixel_sr_handle = plotSensor(sensor, "pixel spectral sr")
    qe_udata, qe_handle = plotSensor(sensor, "sensor spectral qe")
    sensor_sr_udata, sensor_sr_handle = plotSensor(sensor, "sensor spectral sr")

    expected_wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    expected_filters = np.asarray(sensor_get(sensor, "color filters"), dtype=float)
    expected_ir = np.array([[0.8], [0.4]], dtype=float)
    expected_pixel_qe = np.array([[0.5], [0.25]], dtype=float)
    expected_pixel_sr = (
        (expected_wave.reshape(-1, 1) * 1e-9 * 1.602177e-19) / (6.62607015e-34 * 2.99792458e8)
    ) * expected_pixel_qe
    expected_qe = expected_filters * expected_pixel_qe * expected_ir
    expected_sensor_sr = expected_filters * expected_ir * expected_pixel_sr
    expected_names = list(sensor_get(sensor, "filter color letters cell"))

    assert color_handle is None
    assert ir_handle is None
    assert pixel_qe_handle is None
    assert pixel_sr_handle is None
    assert qe_handle is None
    assert sensor_sr_handle is None
    assert np.allclose(color_udata["x"], expected_wave)
    assert np.allclose(color_udata["y"], expected_filters)
    assert color_udata["filterNames"] == expected_names
    assert color_udata["xLabel"] == "Wavelength (nm)"
    assert color_udata["yLabel"] == "Transmittance"
    assert color_udata["nameString"] == "ISET: colorfilters"
    assert np.allclose(ir_udata["x"], expected_wave)
    assert np.allclose(ir_udata["y"], expected_ir)
    assert ir_udata["filterNames"] == ["o"]
    assert ir_udata["xLabel"] == "Wavelength (nm)"
    assert ir_udata["yLabel"] == "Transmittance"
    assert ir_udata["nameString"] == "ISET: irfilter"
    assert np.allclose(pixel_qe_udata["x"], expected_wave)
    assert np.allclose(pixel_qe_udata["y"], expected_pixel_qe)
    assert pixel_qe_udata["filterNames"] == ["k"]
    assert pixel_qe_udata["xLabel"] == "Wavelength (nm)"
    assert pixel_qe_udata["yLabel"] == "QE"
    assert pixel_qe_udata["nameString"] == "ISET: pixelspectralqe"
    assert np.allclose(pixel_sr_udata["x"], expected_wave)
    assert np.allclose(pixel_sr_udata["y"], expected_pixel_sr)
    assert pixel_sr_udata["filterNames"] == ["k"]
    assert pixel_sr_udata["xLabel"] == "Wavelength (nm)"
    assert pixel_sr_udata["yLabel"] == "Responsivity:  Volts/Watt"
    assert pixel_sr_udata["nameString"] == "ISET: pixelspectralsr"
    assert np.allclose(qe_udata["x"], expected_wave)
    assert np.allclose(qe_udata["y"], expected_qe)
    assert qe_udata["filterNames"] == expected_names
    assert qe_udata["xLabel"] == "Wavelength (nm)"
    assert qe_udata["yLabel"] == "Quantum efficiency"
    assert qe_udata["nameString"] == "ISET: sensorspectralqe"
    assert np.allclose(sensor_get(sensor, "sensor spectral sr"), expected_sensor_sr)
    assert np.allclose(sensor_sr_udata["x"], expected_wave)
    assert np.allclose(sensor_sr_udata["y"], expected_sensor_sr)
    assert sensor_sr_udata["filterNames"] == expected_names
    assert sensor_sr_udata["xLabel"] == "Wavelength (nm)"
    assert sensor_sr_udata["yLabel"] == "Responsivity:  Volts/Watt"
    assert sensor_sr_udata["nameString"] == "ISET: sensorspectralsr"


def test_plot_sensor_cfa_wrappers(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)

    block_udata, block_handle = plotSensor(sensor, "cfa")
    full_udata, full_handle = plotSensor(sensor, "cfafull")

    expected_unit_pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    expected_pattern_colors = np.asarray(sensor_get(sensor, "pattern colors"))
    expected_name = str(sensor_get(sensor, "name"))

    assert block_handle is None
    assert full_handle is None
    assert block_udata["mode"] == "block"
    assert full_udata["mode"] == "full"
    assert block_udata["nameString"] == expected_name
    assert full_udata["nameString"] == expected_name
    assert block_udata["scale"] == 8
    assert full_udata["scale"] == 8
    assert np.array_equal(block_udata["unitPattern"], expected_unit_pattern)
    assert np.array_equal(block_udata["pattern"], expected_unit_pattern)
    assert np.array_equal(block_udata["patternColors"], expected_pattern_colors)
    assert np.array_equal(block_udata["unitPatternColors"], expected_pattern_colors)
    assert block_udata["filterNames"] == ["r", "g", "b"]
    assert block_udata["imgSmall"].shape == (2, 2, 3)
    assert block_udata["img"].shape == (16, 16, 3)
    assert np.allclose(block_udata["imgSmall"][0, 0], np.array([0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(block_udata["imgSmall"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(block_udata["imgSmall"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))
    assert np.allclose(block_udata["imgSmall"][1, 1], np.array([0.0, 1.0, 0.0], dtype=float))

    assert np.array_equal(full_udata["unitPattern"], expected_unit_pattern)
    assert full_udata["imgSmall"].shape == (4, 4, 3)
    assert full_udata["img"].shape == (32, 32, 3)
    assert np.array_equal(full_udata["patternColors"][:2, :2], expected_pattern_colors)
    assert np.array_equal(full_udata["patternColors"][2:, 2:], expected_pattern_colors)
    assert np.allclose(full_udata["imgSmall"][0, 0], np.array([0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(full_udata["imgSmall"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(full_udata["imgSmall"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))


def test_plot_sensor_true_size_and_cfa_image_wrappers(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
    )

    true_udata, true_handle = plotSensor(sensor, "true size")
    cfa_image_udata, cfa_image_handle = plotSensor(sensor, "cfa image")
    expected_name = str(sensor_get(sensor, "name"))

    assert true_handle is None
    assert cfa_image_handle is None
    assert true_udata["dataType"] == "volts"
    assert cfa_image_udata["dataType"] == "volts"
    assert true_udata["nameString"] == expected_name
    assert cfa_image_udata["nameString"] == expected_name
    assert true_udata["gamma"] == 1.0
    assert true_udata["scaleMax"] is False
    assert cfa_image_udata["gamma"] == 1.0
    assert cfa_image_udata["scaleMax"] is False
    assert true_udata["img"].shape == (2, 2, 3)
    assert cfa_image_udata["img"].shape == (2, 2, 3)
    assert np.allclose(true_udata["img"][0, 0], np.array([0.0, 0.0, 0.0], dtype=float))
    assert np.allclose(true_udata["img"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(true_udata["img"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))
    assert np.allclose(true_udata["img"][1, 1], np.array([0.0, 0.0, 0.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][0, 0], np.array([0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][0, 1], np.array([1.0, 0.0, 0.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][1, 0], np.array([0.0, 0.0, 1.0], dtype=float))
    assert np.allclose(cfa_image_udata["img"][1, 1], np.array([0.0, 1.0, 0.0], dtype=float))


def test_plot_sensor_true_size_respects_render_controls(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 1)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(sensor, "volts", np.array([[0.25, 0.5]], dtype=float))
    sensor = sensor_set(sensor, "gamma", 2.0)
    sensor = sensor_set(sensor, "scale max", True)

    true_udata, true_handle = plotSensor(sensor, "true size")

    assert true_handle is None
    assert sensor_get(sensor, "gamma") == 2.0
    assert sensor_get(sensor, "scale max") is True
    assert true_udata["gamma"] == 2.0
    assert true_udata["scaleMax"] is True
    assert np.allclose(true_udata["img"], np.array([[0.25, 1.0]], dtype=float))
    assert np.allclose(sensor_get(sensor, "rgb", "volts", 2.0, True), true_udata["img"])


def test_sensor_get_rgb_and_display_ranges(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 1)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(sensor, "zero level", 4)
    sensor = sensor_set(sensor, "dv", np.array([[260.0, 516.0]], dtype=float))

    rgb = np.asarray(sensor_get(sensor, "rgb", "dv or volts", 1.0, False), dtype=float)

    assert sensor_get(sensor, "max output") == sensor_get(sensor, "voltage swing")
    assert sensor_get(sensor, "zero level") == 4.0
    assert sensor_get(sensor, "max digital value") == 1020.0
    assert np.allclose(rgb, np.array([[260.0 / 1020.0, 516.0 / 1020.0]], dtype=float))


def test_plot_sensor_channels_wrapper(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 4)
    sensor = sensor_set(sensor, "cols", 4)
    volts = np.arange(1, 17, dtype=float).reshape(4, 4) / 16.0
    sensor = sensor_set(sensor, "volts", volts)

    channels_udata, channels_handle = plotSensor(sensor, "channels")
    expected_name = str(sensor_get(sensor, "name"))

    assert channels_handle is None
    assert channels_udata["dataType"] == "volts"
    assert channels_udata["nameString"] == expected_name
    assert channels_udata["filterNames"] == ["r", "g", "b"]
    assert len(channels_udata["channelData"]) == 3
    assert len(channels_udata["channelImages"]) == 3
    assert len(channels_udata["masks"]) == 3
    assert channels_udata["pattern"].shape == (4, 4)
    red_mask = channels_udata["masks"][0]
    green_mask = channels_udata["masks"][1]
    blue_mask = channels_udata["masks"][2]
    assert np.array_equal(red_mask, channels_udata["pattern"] == 1)
    assert np.array_equal(green_mask, channels_udata["pattern"] == 2)
    assert np.array_equal(blue_mask, channels_udata["pattern"] == 3)
    assert np.isnan(channels_udata["channelData"][0][0, 0])
    assert np.isclose(channels_udata["channelData"][0][0, 1], volts[0, 1])
    assert np.isclose(channels_udata["channelData"][1][0, 0], volts[0, 0])
    assert np.isclose(channels_udata["channelData"][2][1, 0], volts[1, 0])
    assert np.allclose(channels_udata["channelImages"][0][0, 1, 1:], np.array([0.0, 0.0], dtype=float))
    assert channels_udata["channelImages"][0][0, 1, 0] > 0.0
    assert np.allclose(channels_udata["channelImages"][1][0, 0, [0, 2]], np.array([0.0, 0.0], dtype=float))
    assert channels_udata["channelImages"][1][0, 0, 1] > 0.0
    assert np.allclose(channels_udata["channelImages"][2][1, 0, :2], np.array([0.0, 0.0], dtype=float))
    assert channels_udata["channelImages"][2][1, 0, 2] > 0.0


def test_plot_sensor_etendue_wrapper(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    etendue = np.array(
        [
            [1.0, 0.8, 0.6],
            [0.9, 0.7, 0.5],
        ],
        dtype=float,
    )
    sensor = sensor_set(sensor, "sensor etendue", etendue)

    udata, handle = plotSensor(sensor, "etendue")
    expected_support = sensor_get(sensor, "spatial support", "um")

    assert handle is None
    assert sensor_get(sensor, "vignetting name") == "skip"
    assert udata["nameString"] == "ISET: Etendue (skip)"
    assert udata["xLabel"] == "Position (um)"
    assert udata["yLabel"] == "Position (um)"
    assert udata["zLabel"] == "Relative illumination"
    assert np.allclose(udata["sensorEtendue"], etendue)
    assert np.allclose(udata["support"]["x"], expected_support["x"])
    assert np.allclose(udata["support"]["y"], expected_support["y"])


def test_plot_sensor_noise_wrappers_from_levels(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    sensor = sensor_set(sensor, "dsnu level", 0.01)
    sensor = sensor_set(sensor, "prnu sigma", 5.0)

    dsnu_udata, dsnu_handle = plotSensor(sensor, "dsnu")
    prnu_udata, prnu_handle = plotSensor(sensor, "prnu")

    expected_dsnu = np.random.default_rng(0).normal(0.0, 0.01, size=(2, 3))
    expected_prnu = 1.0 + np.random.default_rng(0).normal(0.0, 0.05, size=(2, 3))

    assert dsnu_handle is None
    assert prnu_handle is None
    assert np.allclose(sensor_get(sensor, "fpn parameters"), np.array([0.01, 5.0], dtype=float))
    assert np.isclose(sensor_get(sensor, "dsnu sigma"), 0.01)
    assert np.isclose(sensor_get(sensor, "prnu sigma"), 5.0)
    assert dsnu_udata["noiseType"] == "dsnu"
    assert dsnu_udata["nameString"] == "ISET:  DSNU"
    assert np.allclose(dsnu_udata["theNoise"], expected_dsnu)
    assert np.allclose(dsnu_udata["noisyImage"], expected_dsnu)
    assert dsnu_udata["titleString"] == (
        f"Max/min: [{float(np.max(expected_dsnu)):.2E},{float(np.min(expected_dsnu)):.2E}] on voltage swing 1.00"
    )
    assert prnu_udata["noiseType"] == "prnu"
    assert prnu_udata["nameString"] == "ISET:  PRNU"
    assert np.allclose(prnu_udata["theNoise"], expected_prnu)
    assert np.allclose(prnu_udata["noisyImage"], expected_prnu)
    assert prnu_udata["titleString"] == f"Max/min: [{float(np.max(expected_prnu)):.2E},{float(np.min(expected_prnu)):.2E}] slope"


def test_plot_sensor_noise_wrappers_use_stored_images_and_shot_noise(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [4.0e-4, 2.5e-3],
                [3.6e-3, 9.0e-4],
            ],
            dtype=float,
        ),
    )
    dsnu_image = np.array([[0.1, -0.2], [0.05, 0.0]], dtype=float)
    prnu_image = np.array([[1.02, 0.98], [1.01, 0.97]], dtype=float)
    sensor = sensor_set(sensor, "dsnu image", dsnu_image)
    sensor = sensor_set(sensor, "prnu image", prnu_image)

    dsnu_udata, _ = plotSensor(sensor, "dsnu")
    prnu_udata, _ = plotSensor(sensor, "prnu")
    shot_udata, shot_handle = plotSensor(sensor, "shot noise")

    electrons = np.asarray(sensor_get(sensor, "electrons"), dtype=float)
    expected_shot = _expected_shot_noise_map(
        electrons,
        float(sensor.fields["pixel"]["conversion_gain_v_per_electron"]),
    )

    assert np.allclose(sensor_get(sensor, "dsnu image"), dsnu_image)
    assert np.allclose(sensor_get(sensor, "prnu image"), prnu_image)
    assert np.allclose(dsnu_udata["theNoise"], dsnu_image)
    assert np.allclose(prnu_udata["theNoise"], prnu_image)
    assert shot_handle is None
    assert shot_udata["noiseType"] == "shotnoise"
    assert shot_udata["nameString"] == "ISET:  Shot noise"
    assert np.allclose(shot_udata["signal"], electrons)
    assert np.allclose(shot_udata["theNoise"], expected_shot)
    assert np.allclose(
        shot_udata["noisyImage"],
        float(sensor.fields["pixel"]["conversion_gain_v_per_electron"]) * np.rint(electrons + (expected_shot / float(sensor.fields["pixel"]["conversion_gain_v_per_electron"]))),
    )


def test_plot_sensor_fft_wrapper(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 3)
    sensor = sensor_set(sensor, "cols", 5)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.5, 0.4, 0.3, 0.2, 0.1],
                [0.2, 0.3, 0.5, 0.7, 0.9],
            ],
            dtype=float,
        ),
    )
    sensor = sensor_set(
        sensor,
        "dv",
        np.array(
            [
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [50.0, 40.0, 30.0, 20.0, 10.0],
                [15.0, 25.0, 35.0, 45.0, 55.0],
            ],
            dtype=float,
        ),
    )

    volts_udata, volts_handle = plotSensorFFT(sensor, "h", "volts", np.array([2, 2], dtype=int))
    dv_udata, dv_handle = plotSensorFFT(sensor, "vertical", "dv", np.array([3, 1], dtype=int))

    h_line = np.asarray(sensor_get(sensor, "volts"), dtype=float)[1, :]
    h_fov = float(sensor_get(sensor, "fov"))
    expected_h_cpd = np.arange(0, round((h_line.size - 1) / 2) + 1, dtype=float) / h_fov
    expected_h_amp = np.abs(np.fft.fft(h_line - np.mean(h_line))) / expected_h_cpd.size

    v_line = np.asarray(sensor_get(sensor, "dv"), dtype=float)[:, 2]
    expected_v_cpd = np.arange(0, round((v_line.size - 1) / 2) + 1, dtype=float) / h_fov
    expected_v_amp = np.abs(np.fft.fft(v_line - np.mean(v_line))) / expected_v_cpd.size

    assert volts_handle is None
    assert dv_handle is None
    assert np.array_equal(volts_udata["xy"], np.array([2, 2], dtype=int))
    assert volts_udata["ori"] == "h"
    assert volts_udata["dataType"] == "volts"
    assert volts_udata["titleString"] == "ISET:  Horizontal fft 2"
    assert volts_udata["xLabel"] == "Cycles/deg (col)"
    assert volts_udata["yLabel"] == "Abs(fft(data))"
    assert np.allclose(volts_udata["cpd"], expected_h_cpd)
    assert np.allclose(volts_udata["amp"], expected_h_amp)
    assert np.allclose(volts_udata["ampPlot"], expected_h_amp[: expected_h_cpd.size])
    assert np.isclose(volts_udata["mean"], float(np.mean(h_line)))
    assert np.isclose(volts_udata["peakContrast"], float(np.max(expected_h_amp) / np.mean(h_line)))

    assert np.array_equal(dv_udata["xy"], np.array([3, 1], dtype=int))
    assert dv_udata["ori"] == "v"
    assert dv_udata["dataType"] == "dv"
    assert dv_udata["titleString"] == "ISET:  Vertical fft 3"
    assert dv_udata["xLabel"] == "Cycles/deg (row)"
    assert dv_udata["yLabel"] == "Abs(fft(data))"
    assert np.allclose(dv_udata["cpd"], expected_v_cpd)
    assert np.allclose(dv_udata["amp"], expected_v_amp)
    assert np.allclose(dv_udata["ampPlot"], expected_v_amp[: expected_v_cpd.size])


def test_plot_sensor_fft_capture_selection(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 4)
    volts = np.array(
        [
            [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4]],
            [[0.4, 1.4], [0.3, 1.3], [0.2, 1.2], [0.1, 1.1]],
        ],
        dtype=float,
    )
    sensor = sensor_set(sensor, "volts", volts)

    udata, handle = plotSensorFFT(sensor, "h", "volts", np.array([2, 1], dtype=int), "capture", 2)

    line = volts[0, :, 1]
    fov = float(sensor_get(sensor, "fov"))
    expected_cpd = np.arange(0, round((line.size - 1) / 2) + 1, dtype=float) / fov
    expected_amp = np.abs(np.fft.fft(line - np.mean(line))) / expected_cpd.size

    assert handle is None
    assert np.array_equal(udata["xy"], np.array([2, 1], dtype=int))
    assert udata["titleString"] == "ISET:  Horizontal fft 1"
    assert udata["yLabel"] == "Abs(fft(data))"
    assert np.allclose(udata["cpd"], expected_cpd)
    assert np.allclose(udata["amp"], expected_amp)
    assert np.allclose(udata["ampPlot"], expected_amp[: expected_cpd.size])
    assert np.isclose(udata["mean"], float(np.mean(line)))


def test_plot_sensor_fft_rejects_color_sensors(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(sensor, "volts", np.ones((2, 2), dtype=float))

    with pytest.raises(UnsupportedOptionError):
        plotSensorFFT(sensor, "h", "volts", np.array([1, 1], dtype=int))


def test_plot_sensor_fft_rejects_invalid_capture(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(sensor, "volts", np.ones((2, 2, 2), dtype=float))

    with pytest.raises(IndexError):
        plotSensorFFT(sensor, "h", "volts", np.array([1, 1], dtype=int), "capture", 3)


def test_plot_display_spd_and_gamut(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)

    spd_udata, spd_handle = plotDisplaySPD(ip)
    gamut_udata, gamut_handle = plotDisplayGamut(ip)

    expected_wave = np.asarray(ip_get(ip, "display wave"), dtype=float)
    expected_spd = np.asarray(ip_get(ip, "display spd"), dtype=float)
    expected_rgb2xyz = np.asarray(ip_get(ip, "display rgb2xyz"), dtype=float)
    expected_xy = expected_rgb2xyz[:, :2] / np.sum(expected_rgb2xyz, axis=1, keepdims=True)

    assert spd_handle is None
    assert gamut_handle is None
    assert np.allclose(spd_udata["wave"], expected_wave)
    assert np.allclose(spd_udata["spd"], expected_spd)
    assert np.allclose(gamut_udata["xy"], expected_xy)
    assert np.isclose(gamut_udata["peakLuminance"], float(ip_get(ip, "display max luminance")))


def test_plot_display_line_and_color_wrappers(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    result = np.array(
        [
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60], [0.70, 0.80, 0.90]],
            [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]],
        ],
        dtype=float,
    )
    xyz = np.array(
        [
            [[0.20, 0.30, 0.10], [0.50, 0.60, 0.20], [0.80, 0.90, 0.30]],
            [[0.25, 0.35, 0.15], [0.55, 0.65, 0.25], [0.85, 0.95, 0.35]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)
    ip = ip_set(ip, "quantization", {"method": "8 bit", "bits": 8})
    ip = ip_set(ip, "data white point", np.array([0.95, 1.0, 1.09], dtype=float))
    ip.data["xyz"] = xyz

    roi = np.array([1, 1, 3, 2], dtype=int)
    line_udata, line_handle = plotDisplayLine(ip, "h", np.array([1, 2], dtype=int))
    hist_udata, hist_handle = plotDisplayColor(ip, "rgb histogram", roi)
    chroma_udata, chroma_handle = plotDisplayColor(ip, "chromaticity", roi)
    luminance_udata, luminance_handle = plotDisplayColor(ip, "luminance", roi)
    lab_udata, lab_handle = plotDisplayColor(ip, "cielab", roi)

    expected_rgb = np.asarray(ip_get(ip, "roi data", roi), dtype=float)
    expected_xyz = np.asarray(ip_get(ip, "roi xyz", roi), dtype=float)
    expected_line = result[1, :, :] * 256.0
    expected_luminance = expected_xyz[:, 1]

    assert line_handle is None
    assert hist_handle is None
    assert chroma_handle is None
    assert luminance_handle is None
    assert lab_handle is None

    assert np.array_equal(line_udata["xy"], np.array([1, 2], dtype=int))
    assert line_udata["ori"] == "h"
    assert line_udata["dataType"] == "digital"
    assert np.allclose(line_udata["values"], expected_line)
    assert np.allclose(line_udata["pos"], np.array([1.0, 2.0, 3.0], dtype=float))

    assert np.array_equal(hist_udata["rect"], roi)
    assert np.allclose(hist_udata["RGB"], expected_rgb)
    assert np.allclose(hist_udata["meanRGB"], np.mean(expected_rgb, axis=0))

    assert np.array_equal(chroma_udata["rect"], roi)
    assert chroma_udata["xy"].shape == (expected_rgb.shape[0], 2)
    assert np.allclose(chroma_udata["XYZ"], expected_xyz)

    assert np.array_equal(luminance_udata["rect"], roi)
    assert np.allclose(luminance_udata["luminance"], expected_luminance)
    assert np.isclose(luminance_udata["meanL"], float(np.mean(expected_luminance)))
    assert np.isclose(luminance_udata["stdLum"], float(np.std(expected_luminance)))

    assert lab_udata.shape == (expected_rgb.shape[0], 3)


def test_plot_aliases_and_sensor_hist_wrapper(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)
    scene_alias, scene_alias_handle = scenePlot(scene, "illuminant energy")
    scene_base, scene_base_handle = plotScene(scene, "illuminant energy")

    oi = oi_create("diffraction limited", asset_store=asset_store)
    oi_alias, oi_alias_handle = plotOI(oi, "otf wavelength")
    oi_base, oi_base_handle = oiPlot(oi, "otf wavelength")

    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 2)
    sensor = sensor_set(sensor, "volts", np.array([[0.10, 0.20], [0.30, 0.40]], dtype=float))
    roi = np.array([1, 1, 2, 2], dtype=int)
    hist_udata, hist_handle = plotSensorHist(sensor, "volts", roi)
    hist_alias_udata, hist_alias_handle = sensorPlotHist(sensor, "volts", roi)
    sensor_alias_udata, sensor_alias_handle = sensorPlot(sensor, "volts histogram", roi)

    assert scene_alias_handle is None
    assert scene_base_handle is None
    assert np.allclose(scene_alias["wave"], scene_base["wave"])
    assert np.allclose(scene_alias["energy"], scene_base["energy"])

    assert oi_alias_handle is None
    assert oi_base_handle is None
    assert np.allclose(oi_alias["fSupport"], oi_base["fSupport"])
    assert np.allclose(oi_alias["otf"], oi_base["otf"])

    assert hist_handle is None
    assert hist_alias_handle is None
    assert sensor_alias_handle is None
    assert np.array_equal(hist_udata["roiLocs"], roi)
    assert np.array_equal(hist_alias_udata["roiLocs"], roi)
    assert np.array_equal(sensor_alias_udata["roiLocs"], roi)
    assert np.allclose(hist_udata["data"], hist_alias_udata["data"], equal_nan=True)
    assert np.allclose(hist_udata["data"], sensor_alias_udata["data"], equal_nan=True)


def test_identity_and_axis_line_wrappers() -> None:
    ax = {"xlim": np.array([-2.0, 3.0]), "ylim": np.array([-1.0, 4.0]), "zlim": np.array([-5.0, 6.0])}

    ident2d = identityLine(ax)
    ident3d = identityLine(ax, True)
    xline = xaxisLine(ax, 0.5, ":")
    yline = yaxisLine(ax, -0.25)

    assert np.allclose(ident2d["x"], np.array([-2.0, 4.0]))
    assert np.allclose(ident2d["y"], np.array([-2.0, 4.0]))
    assert ident2d["linestyle"] == "--"
    assert np.isclose(ident2d["linewidth"], 2.0)

    assert np.allclose(ident3d["x"], np.array([-5.0, 6.0]))
    assert np.allclose(ident3d["y"], np.array([-5.0, 6.0]))
    assert np.allclose(ident3d["z"], np.array([-5.0, 6.0]))

    assert np.allclose(xline["x"], np.array([-2.0, 3.0]))
    assert np.allclose(xline["y"], np.array([0.5, 0.5]))
    assert xline["linestyle"] == ":"

    assert np.allclose(yline["x"], np.array([-0.25, -0.25]))
    assert np.allclose(yline["y"], np.array([-1.0, 4.0]))
    assert yline["linestyle"] == "--"


def test_plot_text_string_wrapper() -> None:
    ax = {
        "xlim": np.array([10.0, 30.0]),
        "ylim": np.array([5.0, 25.0]),
        "xscale": "log",
        "yscale": "linear",
    }

    upper_left = plotTextString("Hello", "ul", [0.1, 0.25], 14, ax=ax)
    lower_right = plotTextString("World", "lr", 0.2, ax=ax)

    assert upper_left["text"] == "Hello"
    assert upper_left["position"] == "ul"
    assert np.isclose(upper_left["x"], 12.0)
    assert np.isclose(upper_left["y"], 20.0)
    assert np.isclose(upper_left["fontSize"], 14.0)
    assert upper_left["background"] == "w"
    assert np.allclose(upper_left["delta"], np.array([0.1, 0.25]))
    assert upper_left["xscale"] == "log"
    assert upper_left["yscale"] == "linear"

    assert lower_right["text"] == "World"
    assert np.isclose(lower_right["x"], 26.0)
    assert np.isclose(lower_right["y"], 9.0)
    assert np.isclose(lower_right["fontSize"], 12.0)
    assert np.allclose(lower_right["delta"], np.array([0.2, 0.2]))


def test_plot_gaussian_spectrum_wrapper(asset_store) -> None:
    wave = np.array([400.0, 500.0, 800.0], dtype=float)

    payload = plotGaussianSpectrum(wave, 500.0, 50.0, asset_store=asset_store)

    expected = np.exp(-((wave - 500.0) ** 2) / (2.0 * 50.0**2))

    assert np.allclose(payload["wavelength"], wave)
    assert np.allclose(payload["transmittance"], expected)
    assert payload["supportRGB"].shape == (wave.size, 3)
    assert np.allclose(payload["supportRGB"][-1], np.array([0.3, 0.3, 0.3]))
    assert payload["xTick"].size == 0
    assert payload["yTick"].size == 0


def test_plot_spectrum_locus_wrapper(asset_store) -> None:
    payload = plotSpectrumLocus(asset_store=asset_store)

    assert payload["wave"].shape == (361,)
    assert payload["wave"][0] == 370.0
    assert payload["wave"][-1] == 730.0
    assert payload["xy"].shape == (361, 2)
    assert np.all(np.isfinite(payload["xy"]))
    assert np.allclose(payload["closingLine"][0], payload["xy"][0])
    assert np.allclose(payload["closingLine"][1], payload["xy"][-1])
    assert payload["axisEqual"] is True
    assert payload["grid"] is True
    assert payload["linestyle"] == "--"


def test_plot_contrast_histogram_wrapper() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    counts, centers = plotContrastHistogram(data)
    expected_counts, edges = np.histogram((data.reshape(-1) - np.mean(data)) / np.mean(data), bins=10)
    expected_centers = 0.5 * (edges[:-1] + edges[1:])

    assert np.allclose(counts, expected_counts)
    assert np.allclose(centers, expected_centers)
    assert int(np.sum(counts)) == data.size


def test_plot_etendue_ratio_wrapper(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "rows", 2)
    sensor = sensor_set(sensor, "cols", 3)
    optimal = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=float)
    bare = np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=float)

    payload = plotEtendueRatio(sensor, optimal, bare, "Gain (%)")
    expected_support = sensor_get(sensor, "spatial support", "microns")

    assert payload["zLabel"] == "Gain (%)"
    assert payload["Ratio"].shape == optimal.shape
    assert np.allclose(payload["Ratio"], (optimal / bare - 1.0) * 100.0)
    assert np.allclose(payload["support"]["x"], np.asarray(expected_support["x"], dtype=float))
    assert np.allclose(payload["support"]["y"], np.asarray(expected_support["y"], dtype=float))


def test_ie_plane_from_vectors_wrapper() -> None:
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    x, y, z = iePlaneFromVectors(basis, [-2.0, 2.0], 6)

    assert x.shape == (6, 10)
    assert y.shape == (6, 10)
    assert z.shape == (6, 10)
    assert np.allclose(z, x + y)


def test_ie_plot_jitter_wrapper() -> None:
    payload = iePlotJitter([0.0, 1.0], [2.0, 3.0], fig=-1, p_symbol="or")
    expected_rng = np.random.default_rng(0)
    expected_jx = np.array([0.0, 1.0], dtype=float) + expected_rng.random(2) * (1.0 / 200.0)
    expected_jy = np.array([2.0, 3.0], dtype=float) + expected_rng.random(2) * (1.0 / 200.0)

    assert np.isclose(payload["f"], 1.0 / 200.0)
    assert payload["figure"] == -1
    assert payload["pSymbol"] == "or"
    assert np.allclose(payload["jx"], expected_jx)
    assert np.allclose(payload["jy"], expected_jy)


def test_ie_shape_circle_wrapper() -> None:
    x, y, z = ieShape("circle", 4, 2.0)

    theta = (2.0 * np.pi * np.arange(1, 5, dtype=float)) / 4.0
    assert np.allclose(x, 2.0 * np.cos(theta))
    assert np.allclose(y, 2.0 * np.sin(theta))
    assert np.allclose(z, np.zeros(4, dtype=float))


def test_ie_plot_and_set_wrappers() -> None:
    payload, handle = iePlot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), "r-")
    updated = iePlotSet(payload, "linewidth", 3.0)

    assert handle is None
    assert payload["figureFactory"] == "ieFigure"
    assert len(payload["series"]) == 1
    assert np.allclose(payload["series"][0]["x"], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(payload["series"][0]["y"], np.array([4.0, 5.0, 6.0]))
    assert payload["series"][0]["style"] == "r-"
    assert np.isclose(updated["series"][0]["lineWidth"], 3.0)


def test_ie_plot_shade_background_wrapper() -> None:
    ax = {"xlim": np.array([1.0, 4.0]), "ylim": np.array([-2.0, 3.0])}
    colors = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.2, 0.3, 0.4]], dtype=float)

    payload = iePlotShadeBackground(ax, "grid", "on", "vertexcolors", colors)

    assert np.allclose(
        payload["background"]["vertices"],
        np.array([[1.0, -2.0], [4.0, -2.0], [4.0, 3.0], [1.0, 3.0]], dtype=float),
    )
    assert np.allclose(payload["background"]["faceVertexCData"], colors)
    assert payload["background"]["faceColor"] == "interp"
    assert payload["background"]["edgeColor"] == "none"
    assert payload["background"]["stack"] == "bottom"
    assert payload["layer"] == "top"
    assert payload["grid"] is True


def test_ie_figure_format_and_resize_wrappers() -> None:
    figure = {"units": "inches", "position": np.array([1.0, 2.0, 3.0, 4.0])}

    formatted = ieFigureFormat(figure, "Arial", [16.0, 12.0], [6.0, 4.0], [0.5, 0.25])
    alias_formatted = ieFormatFigure(figure, "Arial", [16.0, 12.0], [6.0, 4.0], [0.5, 0.25])
    resized = ieFigureResize({"type": "axes"}, [0.1, 0.2, 0.3, 0.4], "pixels")

    assert np.allclose(formatted["position"], np.array([1.0, 2.0, 6.0, 4.0]))
    assert np.allclose(formatted["paperPosition"], np.array([1.25, 3.5, 6.0, 4.0]))
    assert np.allclose(formatted["axes"]["position"], np.array([0.5, 0.5, 5.25, 3.25]))
    assert formatted["axes"]["fontName"] == "Arial"
    assert np.isclose(formatted["axes"]["titleFontSize"], 16.0)
    assert np.isclose(formatted["axes"]["tickFontSize"], 12.0)
    assert np.allclose(alias_formatted["position"], formatted["position"])
    assert resized["windowStyle"] == "normal"
    assert resized["units"] == "pixels"
    assert resized["resize"] == "off"
    assert np.allclose(resized["position"], np.array([0.1, 0.2, 0.3, 0.4]))


def test_fise_plot_defaults_wrapper() -> None:
    defaults = fisePlotDefaults()

    assert defaults["DefaultAxesFontName"] == "Georgia"
    assert np.isclose(defaults["DefaultAxesFontSize"], 16.0)
    assert defaults["DefaultAxesBox"] == "off"
    assert defaults["DefaultAxesTickDir"] == "out"
    assert np.isclose(defaults["DefaultAxesLineWidth"], 1.2)
    assert np.allclose(defaults["DefaultTextColor"], np.array([0.3, 0.3, 0.3]))
    assert defaults["DefaultLegendBox"] == "off"


def test_hist2d_wrapper_counts_nearest_bins() -> None:
    data = np.array([[0.0, 0.0], [1.0, 1.0], [0.9, 1.1]], dtype=float)

    counts = hist2d(data, 3, 3, [0.0, 1.0], [0.0, 1.0])

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    assert np.allclose(counts, expected)


def test_scatplot_wrapper_square_density() -> None:
    payload = scatplot(
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
        "squares",
        0.25,
        5,
        3,
        2,
        4,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
    )

    assert np.allclose(payload["dd"], np.array([8.0, 8.0, 4.0]))
    assert payload["xi"].shape == (5, 5)
    assert payload["yi"].shape == (5, 5)
    assert payload["zi"].shape == (5, 5)
    assert payload["zif"].shape == (5, 5)
    assert "contour" in payload
    assert len(payload["hs"]) >= 1


def test_ie_hist_image_histcn_wrapper() -> None:
    samples = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]], dtype=float)

    img, xy, cmap, fhandle = ieHistImage(samples, "hist type", "histcn", "edges", [2, 2], "plot flag", False)

    assert img.shape == (2, 2)
    assert np.allclose(img, np.ones((2, 2), dtype=float))
    assert np.allclose(xy[0], np.array([0.3, 0.7]))
    assert np.allclose(xy[1], np.array([0.3, 0.7]))
    assert cmap.shape == (256, 3)
    assert fhandle is None


def test_ie_hist_image_scatplot_wrapper() -> None:
    samples = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [1.0, 1.0], [1.1, 1.05]], dtype=float)

    img, xy, cmap, fhandle = ieHistImage(
        samples,
        "hist type",
        "scatplot",
        "scatmethod",
        "squares",
        "scatradius",
        0.25,
        "plot flag",
        False,
    )

    assert img.shape == (100, 100)
    assert xy[0].shape == (100,)
    assert xy[1].shape == (100,)
    assert np.max(img) > 0.0
    assert cmap.shape == (256, 3)
    assert fhandle is None


def test_plot_set_up_window_wrapper() -> None:
    default_window = plotSetUpWindow()
    selected_window = plotSetUpWindow(5)

    assert default_window["figureId"] == "GRAPHWIN"
    assert default_window["units"] == "Normalized"
    assert np.allclose(default_window["position"], np.array([0.5769, 0.0308, 0.4200, 0.4200]))
    assert default_window["name"] == "ISET GraphWin"
    assert default_window["numberTitle"] == "off"
    assert default_window["colormap"] == "default"
    assert default_window["clear"] is True

    assert selected_window["figureId"] == 5
    assert selected_window["colormap"] == "default"
    assert selected_window["clear"] is True


def test_plot_normal_wrapper() -> None:
    payload = plotNormal([1.0, 2.0], [0.5, 1.0], color=["k", "r"])

    assert payload["x"].shape == (200,)
    assert payload["pdf"].shape == (2, 200)
    assert np.all(payload["pdf"] >= 0.0)
    assert np.allclose(payload["mean"], np.array([1.0, 2.0]))
    assert np.allclose(payload["sigma"], np.array([0.5, 1.0]))
    assert payload["color"] == ["k", "r"]
    assert payload["xlabel"] == "x"
    assert payload["ylabel"] == "Probability Density"
    assert payload["grid"] is True
    assert payload["axisTight"] is True


def test_plot_radiance_wrapper_transposes_columnwise_input() -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    radiance = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]], dtype=float)

    payload, handle = plotRadiance(wave, radiance.T, color="r", linewidth=3.0, title="Demo")

    assert handle is None
    assert np.allclose(payload["wavelength"], wave)
    assert np.allclose(payload["radiance"], radiance)
    assert payload["title"] == "Demo"
    assert payload["color"] == "r"
    assert np.isclose(payload["linewidth"], 3.0)
    assert payload["ylabel"] == "Radiance (watts/sr/nm/m^2)"
    assert payload["grid"] is True


def test_plot_reflectance_wrapper_transposes_rowwise_input() -> None:
    wave = np.array([500.0, 600.0, 700.0], dtype=float)
    reflectance = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=float,
    )

    payload, handle = plotReflectance(wave, reflectance.T, color=[0.1, 0.2, 0.3], linewidth=1.5, linestyle=":")

    assert handle is None
    assert np.allclose(payload["wavelength"], wave)
    assert np.allclose(payload["reflectance"], reflectance)
    assert np.allclose(payload["color"], np.array([0.1, 0.2, 0.3]))
    assert np.isclose(payload["linewidth"], 1.5)
    assert payload["linestyle"] == ":"
    assert payload["ylabel"] == "Reflectance"
    assert payload["grid"] is True


def test_direct_sensor_plot_wrappers_match_plot_sensor(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)

    pixel_payload, pixel_handle = plotPixelSNR(sensor)
    expected_pixel, _ = plotSensor(sensor, "pixel snr")
    etendue_payload, etendue_handle = plotSensorEtendue(sensor)
    expected_etendue, _ = plotSensor(sensor, "etendue")
    sensor_snr_payload, sensor_snr_handle = plotSensorSNR(sensor)
    expected_sensor_snr, _ = plotSensor(sensor, "sensor snr")

    assert pixel_handle is None
    assert etendue_handle is None
    assert sensor_snr_handle is None

    assert np.allclose(pixel_payload["volts"], expected_pixel["volts"])
    assert np.allclose(pixel_payload["snr"], expected_pixel["snr"])
    assert np.allclose(pixel_payload["snrShot"], expected_pixel["snrShot"])
    assert np.allclose(pixel_payload["snrRead"], expected_pixel["snrRead"])
    assert pixel_payload["legend"] == expected_pixel["legend"]
    assert pixel_payload["titleString"] == expected_pixel["titleString"]

    assert np.allclose(etendue_payload["support"]["x"], expected_etendue["support"]["x"])
    assert np.allclose(etendue_payload["support"]["y"], expected_etendue["support"]["y"])
    assert np.allclose(etendue_payload["sensorEtendue"], expected_etendue["sensorEtendue"])
    assert etendue_payload["zLabel"] == "Relative illumination"

    assert np.allclose(sensor_snr_payload["volts"], expected_sensor_snr["volts"])
    assert np.allclose(sensor_snr_payload["snr"], expected_sensor_snr["snr"])
    assert np.allclose(sensor_snr_payload["snrShot"], expected_sensor_snr["snrShot"])
    assert np.allclose(sensor_snr_payload["snrRead"], expected_sensor_snr["snrRead"])
    assert np.allclose(sensor_snr_payload["snrDSNU"], expected_sensor_snr["snrDSNU"])
    assert np.allclose(sensor_snr_payload["snrPRNU"], expected_sensor_snr["snrPRNU"])
    assert sensor_snr_payload["legend"] == expected_sensor_snr["legend"]


def test_plot_ml_wrapper_payloads(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    oi = oi_create(asset_store=asset_store)
    microlens = ml_radiance(mlens_create(sensor, oi, asset_store=asset_store), sensor, asset_store=asset_store)

    offsets_payload, offsets_handle = plotML(microlens, "offsets", sensor)
    mesh_payload, mesh_handle = plotML(microlens, "mesh pixel irradiance")
    image_payload, image_handle = plotML(microlens, "image pixel irradiance")

    expected_support = sensor_get(sensor, "spatial support", "mm")
    expected_offsets = np.asarray(mlens_get(microlens, "optimal offsets", sensor), dtype=float)
    expected_irradiance = np.asarray(mlens_get(microlens, "pixel irradiance"), dtype=float)
    expected_x = np.asarray(mlens_get(microlens, "x coordinate"), dtype=float)
    pixel_width_um = float(mlens_get(microlens, "diameter", "microns"))
    positive_index = int(np.argmin(np.abs(expected_x - (pixel_width_um / 2.0))))
    negative_index = int(np.argmin(np.abs((-expected_x) - (pixel_width_um / 2.0))))

    assert offsets_handle is None
    assert mesh_handle is None
    assert image_handle is None
    np.testing.assert_allclose(offsets_payload["support"]["x"], np.asarray(expected_support["x"], dtype=float))
    np.testing.assert_allclose(offsets_payload["support"]["y"], np.asarray(expected_support["y"], dtype=float))
    np.testing.assert_allclose(offsets_payload["optimalOffsets"], expected_offsets)
    assert offsets_payload["command"] == "mesh(support.y, support.x, optimalOffsets)"

    np.testing.assert_allclose(mesh_payload["x"], expected_x)
    np.testing.assert_allclose(mesh_payload["y"], expected_x)
    np.testing.assert_allclose(mesh_payload["pixelIrradiance"], expected_irradiance)
    assert mesh_payload["colormap"].shape == (191, 3)

    np.testing.assert_allclose(image_payload["pixelIrradiance"], expected_irradiance)
    assert image_payload["image"][positive_index, :].min() == pytest.approx(1.0)
    assert image_payload["image"][:, positive_index].min() == pytest.approx(1.0)
    assert image_payload["image"][negative_index, :].min() == pytest.approx(1.0)
    assert image_payload["image"][:, negative_index].min() == pytest.approx(1.0)
    np.testing.assert_array_equal(image_payload["boundaryIndices"], np.array([negative_index + 1, positive_index + 1]))
    np.testing.assert_allclose(image_payload["colorbarTicks"], np.arange(0.0, 1.01, 0.25))


def test_plot_metrics_wrapper_histogram_payload() -> None:
    metric_image = np.arange(1.0, 17.0, dtype=float).reshape(4, 4)
    handles = {
        "image1_name": "first",
        "image2_name": "second",
        "current_metric": "dEab",
        "metric_image": metric_image,
        "img1_rect": np.array([2, 2, 1, 1], dtype=int),
    }

    payload, handle = plotMetrics(handles)

    expected_data = np.array([6.0, 7.0, 10.0, 11.0], dtype=float)
    expected_hist, expected_edges = np.histogram(expected_data, bins=10)

    assert handle is None
    np.testing.assert_array_equal(payload["rect"], np.array([2, 2, 1, 1]))
    np.testing.assert_allclose(payload["data"], expected_data)
    assert payload["metricName"] == "dEab"
    assert payload["titleString"] == "ROI: first and second "
    assert payload["nBins"] == 10
    np.testing.assert_allclose(payload["histogram"], expected_hist)
    np.testing.assert_allclose(payload["binEdges"], expected_edges)
    assert payload["stats"]["mean"] == pytest.approx(float(np.mean(expected_data)))
    assert "Mean" in payload["annotation"]["text"]
    assert payload["grid"] is True


def test_sensor_plot_color_wrapper(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6, 0.7],
        ],
        dtype=float,
    )
    sensor = sensor_set(sensor, "volts", volts)

    payload, handle = sensorPlotColor(sensor, "rg")

    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    d = float(payload["xlim"][1])
    rgb_6500 = np.asarray(spectral_qe.T @ np.asarray(blackbody(wave, 6500.0, kind="quanta"), dtype=float), dtype=float)
    rgb_6500 = 0.9 * d * (rgb_6500 / np.linalg.norm(rgb_6500[:2]))

    assert handle is None
    assert payload["type"] == "rg"
    assert payload["labels"] == ["Red sensor", "Green sensor"]
    assert payload["x"].size == payload["y"].size == 16
    assert len(payload["referencePoints"]) == 9
    assert payload["titleString"] == "Sensor Color Balance"
    np.testing.assert_allclose(payload["xlim"], payload["ylim"])
    assert payload["axisEqual"] is True
    np.testing.assert_allclose(payload["referencePoints"][6]["point"], rgb_6500[:2])


def test_plot_scene_test_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    roi_rect = np.array([6, 6, 4, 4], dtype=int)
    line_selector = np.array([1, max(int(scene_get(scene, "rows")) // 2, 1)], dtype=int)

    luminance_line, line_handle = plotScene(scene, "luminance hline", line_selector, asset_store=asset_store)
    photons_roi, photons_handle = plotScene(scene, "illuminant photons roi", asset_store=asset_store)
    reflectance_roi, reflectance_handle = plotScene(scene, "reflectance roi", roi_rect, asset_store=asset_store)
    chromaticity_roi, chroma_handle = plotScene(scene, "chromaticity", roi_rect, asset_store=asset_store)
    illuminant_photons, illuminant_handle = plotScene(scene, "illuminant photons", asset_store=asset_store)

    assert line_handle is None
    assert photons_handle is None
    assert reflectance_handle is None
    assert chroma_handle is None
    assert illuminant_handle is None
    assert "data" in luminance_line
    assert "photons" in photons_roi
    assert "reflectance" in reflectance_roi
    assert chromaticity_roi["x"].size == chromaticity_roi["y"].size
    np.testing.assert_allclose(illuminant_photons["wave"], np.asarray(scene_get(scene, "wave"), dtype=float))


def test_plot_sensor_test_workflow(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    scene = scene_set(scene, "fov", 4.0)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "qmethod", "10 bit")
    sensor = sensor_compute(sensor, oi)
    roi_rect = np.array([4, 4, 3, 3], dtype=int)
    line_selector = np.array([20, 20], dtype=int)

    electrons_line, _ = plotSensor(sensor, "electrons hline", line_selector)
    volts_vline, _ = plotSensor(sensor, "volts vline", line_selector)
    volts_hline, _ = plotSensor(sensor, "volts hline", line_selector)
    dv_hline, _ = plotSensor(sensor, "dv hline", line_selector)
    sensor_snr_payload, _ = plotSensor(sensor, "sensor snr")
    electrons_hist, _ = plotSensor(sensor, "electrons hist", roi_rect)
    pixel_snr_payload, _ = plotSensor(sensor, "pixel snr")
    cfa_block, _ = plotSensor(sensor, "cfa block")
    cfa_full, _ = plotSensor(sensor, "cfa full")
    etendue_payload, _ = plotSensor(sensor, "etendue")
    color_filters, _ = plotSensor(sensor, "color filters")

    np.testing.assert_array_equal(electrons_line["xy"], line_selector)
    assert volts_vline["ori"] == "v"
    assert volts_hline["ori"] == "h"
    assert dv_hline["dataType"] == "dv"
    assert sensor_snr_payload["titleString"] == "Sensor SNR over response range"
    assert "data" in electrons_hist
    assert pixel_snr_payload["titleString"] == "Pixel SNR over response range"
    assert "pattern" in cfa_block
    assert cfa_full["mode"] == "full"
    assert "sensorEtendue" in etendue_payload
    assert "x" in color_filters and "y" in color_filters
