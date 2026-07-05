"""Display creation and accessors."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.io import savemat
from scipy.ndimage import zoom
from scipy.spatial import ConvexHull

from .assets import AssetStore, ie_read_spectra
from .color import xyz_color_matching, xyz_to_lms
from .exceptions import UnsupportedOptionError
from .metrics import chromaticity_xy, xyz_from_energy, xyz_to_lab
from .session import track_session_object
from .types import Display, SessionContext
from .utils import blackbody, ie_lut_digital, ie_unit_scale_factor, image_linear_transform, interp_spectra, invert_gamma_table, param_format, spectral_step, srgb_to_xyz, xyz_to_srgb


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _is_empty_dispatch_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip() == ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value).size == 0
    return False


def _struct_field(value: Any, field: str, default: Any = ...,) -> Any:
    if isinstance(value, dict):
        if field in value:
            return value[field]
    elif hasattr(value, field):
        return getattr(value, field)
    if default is ...:
        raise KeyError(field)
    return default


def _flatten_struct_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return [item for item in value.reshape(-1, order="F")]
    return [value]


def _normalize_render_function(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            return value
    function_handle = getattr(value, "function_handle", None)
    if function_handle is not None:
        function = getattr(function_handle, "function", None)
        if function:
            return str(function)
    function = getattr(value, "function", None)
    if function:
        return str(function)
    return value


def _display_to_mat_payload(display: Display) -> dict[str, Any]:
    payload = {
        "name": display.name,
        "type": display.type,
        "wave": np.asarray(display.fields["wave"], dtype=float),
        "spd": np.asarray(display_get(display, "spd"), dtype=float),
        "gamma": np.asarray(display_get(display, "gamma"), dtype=float),
        "dpi": float(display_get(display, "dpi")),
        "dist": float(display_get(display, "dist")),
        "isEmissive": bool(display.fields.get("is_emissive", True)),
    }
    if display.fields.get("refresh_rate_hz") is not None:
        payload["refreshRate"] = float(display.fields["refresh_rate_hz"])
    if display.fields.get("psfs") is not None:
        payload["psfs"] = np.asarray(display.fields["psfs"], dtype=float)
    if display.fields.get("dacsize") is not None:
        payload["dacsize"] = int(display.fields["dacsize"])
    return payload


def _mat_display_to_display(display_struct: Any) -> Display:
    display = Display(name=str(getattr(display_struct, "name", "display")))
    dixel = None
    if hasattr(display_struct, "dixel"):
        dixel_struct = getattr(display_struct, "dixel")
        dixel = {}
        for matlab_key, field_key in {
            "intensitymap": "intensity_map",
            "controlmap": "control_map",
            "nPixels": "n_pixels",
            "renderFunc": "render_function",
        }.items():
            if hasattr(dixel_struct, matlab_key):
                field_value = getattr(dixel_struct, matlab_key)
                if field_key == "render_function":
                    field_value = _normalize_render_function(field_value)
                dixel[field_key] = field_value
    display.fields["wave"] = np.asarray(getattr(display_struct, "wave"), dtype=float)
    display.fields["spd"] = np.asarray(getattr(display_struct, "spd"), dtype=float)
    display.fields["gamma"] = np.asarray(getattr(display_struct, "gamma"), dtype=float)
    display.fields["dpi"] = float(getattr(display_struct, "dpi", 96.0))
    display.fields["dist"] = float(getattr(display_struct, "dist", 0.5))
    display.fields["is_emissive"] = bool(getattr(display_struct, "isEmissive", True))
    display.fields["ambient_spd"] = np.zeros(display.fields["wave"].shape, dtype=float)
    display.fields["size_m"] = None
    display.fields["dixel"] = dixel
    display.fields["comment"] = None
    display.fields["refresh_rate_hz"] = None
    display.fields["image"] = None
    if hasattr(display_struct, "psfs"):
        display.fields["psfs"] = np.asarray(getattr(display_struct, "psfs"), dtype=float)
    if hasattr(display_struct, "dacsize"):
        display.fields["dacsize"] = int(getattr(display_struct, "dacsize"))
    return display


def _display_default() -> Display:
    display = Display(name="default")
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    spd = np.full((wave.size, 3), 1.0 / 700.0, dtype=float)
    gamma = np.repeat(np.linspace(0.0, 1.0, 256, dtype=float)[:, None], 3, axis=1)
    display.fields.update(
        {
            "wave": wave,
            "spd": spd,
            "gamma": gamma,
            "dpi": 96.0,
            "dist": 0.5,
            "is_emissive": True,
            "ambient_spd": np.zeros(wave.shape, dtype=float),
            "size_m": None,
            "dixel": None,
            "comment": None,
            "refresh_rate_hz": None,
            "image": None,
            "psfs": None,
            "dacsize": int(round(np.log2(gamma.shape[0]))),
        }
    )
    return display


def display_convert(
    ct_disp: Any,
    sample_wave: Any | None = None,
    output_filename: str | None = None,
    overwrite: bool = False,
    display_name: str | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> Display:
    """Convert a ctToolbox-style display structure into a pyISETCam display."""

    if ct_disp is None:
        raise ValueError("ctToolbox display structure required")

    store = _store(asset_store)
    source = ct_disp
    if isinstance(ct_disp, str):
        loaded = store.load_mat(ct_disp)
        source = loaded.get("vDisp")
        if source is None:
            raise KeyError("displayConvert expected a vDisp payload")

    cdixel = _struct_field(_struct_field(source, "sPhysicalDisplay"), "m_objCDixelStructure")
    display = _display_default()
    display = display_set(display, "name", str(_struct_field(source, "m_strDisplayName")))
    display = display_set(display, "wave", np.asarray(_struct_field(cdixel, "m_aWaveLengthSamples"), dtype=float).reshape(-1))

    spd = np.asarray(_struct_field(cdixel, "m_aSpectrumOfPrimaries"), dtype=float)
    if spd.ndim == 1:
        spd = spd.reshape(-1, 1)
    if spd.shape[0] != display_get(display, "nwave") and spd.shape[1] == display_get(display, "nwave"):
        spd = spd.T
    display = display_set(display, "spd", spd)

    gamma_structs = _flatten_struct_sequence(_struct_field(cdixel, "m_cellGammaStructure"))
    gamma_columns = [
        np.asarray(_struct_field(struct, "vGammaRampLUT"), dtype=float).reshape(-1)
        for struct in gamma_structs
    ]
    if gamma_columns:
        display = display_set(display, "gTable", np.column_stack(gamma_columns))

    psf_structs = _flatten_struct_sequence(_struct_field(cdixel, "m_cellPSFStructure", None))
    if psf_structs:
        psf_planes = []
        for struct in psf_structs:
            raw = np.asarray(_struct_field(_struct_field(struct, "sCustomData"), "aRawData"), dtype=float)
            if raw.shape != (20, 20):
                row_idx = np.linspace(0.0, raw.shape[0] - 1.0, 20, dtype=float)
                col_idx = np.linspace(0.0, raw.shape[1] - 1.0, 20, dtype=float)
                raw = interp_spectra(np.arange(raw.shape[0], dtype=float), raw, row_idx)
                raw = interp_spectra(np.arange(raw.shape[1], dtype=float), raw.T, col_idx).T
            total = float(np.sum(raw, dtype=float))
            psf_planes.append(raw / max(total, 1.0e-12))
        display = display_set(display, "psfs", np.stack(psf_planes, axis=2))

    pixel_size_mm = float(_struct_field(cdixel, "m_fPixelSizeInMmX"))
    display = display_set(display, "dpi", 25.4 / max(pixel_size_mm, 1.0e-12))
    display = display_set(display, "viewing distance", float(_struct_field(_struct_field(source, "sViewingContext"), "m_fViewingDistance")))
    display = display_set(display, "refresh rate", float(_struct_field(_struct_field(source, "sPhysicalDisplay"), "m_fVerticalRefreshRate")))

    if sample_wave is not None and np.asarray(sample_wave).size:
        target_wave = np.asarray(sample_wave, dtype=float).reshape(-1)
        display = display_set(display, "wave", target_wave)

    if display_name not in {None, ""}:
        display = display_set(display, "name", str(display_name))

    if output_filename not in {None, ""}:
        save_path = Path(str(output_filename))
        if save_path.exists() and not overwrite:
            warnings.warn("File already exists. Try other name or overwrite flag", stacklevel=2)
        else:
            savemat(save_path, {"d": _display_to_mat_payload(display)})

    return display


def display_pt2iset(
    fname: str,
    i_wave: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> Display:
    """Convert a PsychToolbox calibration file to a pyISETCam display."""

    if fname in {None, ""}:
        raise ValueError("No calibration file supplied")

    store = _store(asset_store)
    data = store.load_mat(fname)
    calibration = _flatten_struct_sequence(data["cals"])[0]

    display = _display_default()
    display = display_set(display, "name", str(fname))

    s_device = np.asarray(_struct_field(calibration, "S_device"), dtype=float).reshape(-1)
    wave = (np.arange(int(s_device[2]), dtype=float) * float(s_device[1])) + float(s_device[0])
    display = display_set(display, "wave", wave)

    p_device = np.asarray(_struct_field(calibration, "P_device"), dtype=float)
    if p_device.ndim == 1:
        p_device = p_device.reshape(-1, 1)
    if i_wave is None:
        display = display_set(display, "spd", p_device)
    else:
        target_wave = np.asarray(i_wave, dtype=float).reshape(-1)
        display = display_set(display, "spd", p_device)
        display = display_set(display, "wave", target_wave)

    gamma = np.asarray(_struct_field(calibration, "gammaTable"), dtype=float)
    if gamma.ndim == 1:
        gamma = gamma.reshape(-1, 1)
    display = display_set(display, "gamma", gamma)
    display.fields["dacsize"] = int(_struct_field(_struct_field(calibration, "describe"), "dacsize"))
    return display


def display_reflectance(
    ctemp: float,
    wave: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[Display, np.ndarray, np.ndarray]:
    """Create the legacy theoretical reflectance display for an illuminant temperature."""

    store = _store(asset_store)
    wave_array = np.arange(400.0, 701.0, 1.0, dtype=float) if wave is None else np.asarray(wave, dtype=float).reshape(-1)

    basis = np.asarray(ie_read_spectra("reflectanceBasis.mat", wave_array, asset_store=store), dtype=float)
    basis[:, 0] = -basis[:, 0]
    ill_energy = np.asarray(blackbody(wave_array, float(ctemp), kind="energy"), dtype=float).reshape(-1)
    radiance_basis = ill_energy[:, None] * basis[:, :3]

    srgb_primaries = np.eye(3, dtype=float).reshape(1, 3, 3)
    lxyz_in_cols = np.asarray(srgb_to_xyz(srgb_primaries), dtype=float).reshape(3, 3).T
    xyz = np.asarray(ie_read_spectra("XYZEnergy.mat", wave_array, asset_store=store), dtype=float)
    transform = np.linalg.pinv(xyz.T @ radiance_basis) @ lxyz_in_cols
    rgb_primaries = radiance_basis @ transform

    display = _display_default()
    display = display_set(display, "wave", wave_array)
    display = display_set(display, "spd", rgb_primaries)

    peak_luminance = float(display_get(display, "peak luminance"))
    scale = 100.0 / max(peak_luminance, 1.0e-12)
    rgb_primaries = rgb_primaries * scale
    display = display_set(display, "spd", rgb_primaries)
    ill_energy = ill_energy * scale

    apple_display = display_create("LCD-Apple", asset_store=store)
    display = display_set(display, "gamma", np.asarray(display_get(apple_display, "gamma"), dtype=float))
    display = display_set(display, "name", f"Natural (ill {int(np.rint(float(ctemp)))}K)")
    return display, np.asarray(rgb_primaries, dtype=float), np.asarray(ill_energy, dtype=float)


def _display_ambient(display: Display) -> np.ndarray:
    ambient = display.fields.get("ambient_spd")
    wave = np.asarray(display.fields["wave"], dtype=float)
    if ambient is None:
        return np.zeros(wave.shape, dtype=float)
    return np.asarray(ambient, dtype=float).reshape(wave.shape)


def _display_rgb2xyz(display: Display) -> np.ndarray:
    wave = np.asarray(display.fields["wave"], dtype=float).reshape(-1)
    spd = np.asarray(display_get(display, "rgb spd"), dtype=float)
    xyz_energy = xyz_color_matching(wave, energy=True)
    return np.asarray(spd.T @ (xyz_energy * float(spectral_step(wave)) * 683.0), dtype=float)


def _display_size_m(display: Display) -> np.ndarray | None:
    size = display.fields.get("size_m")
    if size is not None:
        return np.asarray(size, dtype=float).reshape(2)
    image = display.fields.get("image")
    if image is None:
        return None
    image_array = np.asarray(image)
    if image_array.ndim < 2:
        return None
    meters_per_dot = float(display_get(display, "meters per dot"))
    rows, cols = image_array.shape[:2]
    return np.array([cols * meters_per_dot, rows * meters_per_dot], dtype=float)


def _inverse_gamma_table(gamma_table: np.ndarray, n_steps: int) -> np.ndarray:
    linear_axis = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    linear_rgb = np.repeat(linear_axis, gamma_table.shape[1], axis=1)
    return invert_gamma_table(linear_rgb, gamma_table)


def mperdot2dpi(mpd: Any) -> float:
    """Convert microns-per-dot to dots-per-inch."""

    microns_per_dot = float(np.asarray(mpd, dtype=float).reshape(-1)[0])
    if microns_per_dot <= 0.0:
        raise ValueError("mperdot2dpi requires a positive microns-per-dot value.")
    return float((1.0 / microns_per_dot) * (2.54 * 1e4))


def ie_calculate_monitor_dpi(
    monitor_size_x_cm: float,
    monitor_size_y_cm: float,
    num_pixels_x: int,
    num_pixels_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MATLAB-style monitor DPI and dot pitch from dimensions in cm."""

    if num_pixels_x <= 0 or num_pixels_y <= 0:
        raise ValueError("ieCalculateMonitorDPI requires positive pixel counts.")
    pixel_size_x_mm = float(monitor_size_x_cm) * 10.0 / float(num_pixels_x)
    pixel_size_y_mm = float(monitor_size_y_cm) * 10.0 / float(num_pixels_y)
    dpi_x = mperdot2dpi(pixel_size_x_mm * 1e3)
    dpi_y = mperdot2dpi(pixel_size_y_mm * 1e3)
    return (
        np.array([dpi_x, dpi_y], dtype=float),
        np.array([pixel_size_x_mm, pixel_size_y_mm], dtype=float),
    )


def display_max_contrast(signal_dir: Any, back_dir: Any) -> float:
    """Return the maximum scalar contrast that keeps RGB within [0, 1]."""

    signal = np.asarray(signal_dir, dtype=float).reshape(-1)
    background = np.asarray(back_dir, dtype=float).reshape(-1)
    if signal.size != 3 or background.size != 3:
        raise ValueError("displayMaxContrast requires three-element signal/background vectors.")
    bounds = np.empty(3, dtype=float)
    for index in range(3):
        if np.isclose(signal[index], 0.0):
            bounds[index] = np.inf
        elif signal[index] > 0.0:
            bounds[index] = (1.0 - background[index]) / signal[index]
        else:
            bounds[index] = abs(-background[index] / signal[index])
    return float(np.min(bounds))


def display_create(
    display_name: str = "LCD-Apple",
    *args: Any,
    asset_store: AssetStore | None = None,
    wave: np.ndarray | None = None,
    session: SessionContext | None = None,
) -> Display:
    """Create a supported display."""

    store = _store(asset_store)
    resolved_name = "LCD-Apple" if _is_empty_dispatch_placeholder(display_name) else display_name
    normalized = param_format(resolved_name)
    if normalized == "default":
        display = _display_default()
    elif normalized == "equalenergy":
        display = _display_default()
        display.name = "equalenergy"
        display.fields["spd"] = np.ones_like(display.fields["spd"], dtype=float) * 1e-3
    else:
        name = resolved_name
        if not name.lower().endswith(".mat"):
            name = f"{name}.mat"
        try:
            display = _mat_display_to_display(store.load_display_struct(name))
        except Exception as error:  # pragma: no cover - exercised via tests against actual assets
            raise UnsupportedOptionError("displayCreate", resolved_name) from error

    target_wave = wave
    if target_wave is None and len(args) == 1:
        target_wave = np.asarray(args[0], dtype=float)
    if target_wave is not None:
        target_wave = np.asarray(target_wave, dtype=float).reshape(-1)
        display.fields["spd"] = interp_spectra(
            np.asarray(display.fields["wave"], dtype=float),
            np.asarray(display.fields["spd"], dtype=float),
            target_wave,
        )
        display.fields["wave"] = target_wave
    display.fields.setdefault("image", None)
    return track_session_object(session, display)


def display_list(
    *,
    type: str = "",
    show: bool = True,
    asset_store: AssetStore | None = None,
) -> list[str]:
    """List vendored display calibration files, optionally filtered by prefix."""

    store = _store(asset_store)
    display_root = store.ensure() / "data" / "displays"
    pattern = f"{type}*.mat" if str(type).strip() else "*.mat"
    names = sorted(path.name for path in display_root.glob(pattern))
    if show:
        for name in names:
            print(name)
    return names


def display_description(display: Display | None) -> str:
    """Return a concise text summary of a display."""

    if display is None:
        return "No display structure"

    description = f"Name:\t{display_get(display, 'name')}\n"
    wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    spacing = int(np.rint(float(display_get(display, "binwidth"))))
    description += f"Wave:\t{int(np.min(wave))}:{spacing}:{int(np.max(wave))} nm\n"
    description += f"# primaries:\t{int(display_get(display, 'nprimaries'))}\n"
    description += f"Color bit depth:\t{int(display_get(display, 'bits'))}\n"
    rgb = display_get(display, "image")
    if rgb is not None:
        rgb_array = np.asarray(rgb)
        if rgb_array.ndim >= 2:
            description += f"Image width: {rgb_array.shape[1]}\t Height: {rgb_array.shape[0]}"
    return description


def display_show_image(
    display: Display,
    app: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Render the display's stored RGB image headlessly through the display model."""

    del app
    from .scene import scene_create, scene_from_file, scene_get

    rgb = display_get(display, "image")
    if rgb is None:
        scene = scene_create(asset_store=_store(asset_store))
        rgb = np.asarray(scene_get(scene, "rgb"), dtype=float)
        display_set(display, "image", rgb)
        return rgb
    scene = scene_from_file(rgb, "rgb", None, display, asset_store=_store(asset_store))
    return np.asarray(scene_get(scene, "rgb"), dtype=float)


def display_plot(
    display: Display,
    param: str,
    *args: Any,
) -> tuple[Any, None]:
    """Return MATLAB-style displayPlot user-data without opening a figure."""

    if display is None:
        raise ValueError("Display required")

    key = param_format(param)
    if key in {"primaries", "spd"}:
        wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
        spd = np.asarray(display_get(display, "spd primaries"), dtype=float)
        return {"wave": wave.copy(), "spd": spd.copy()}, None
    if key in {"gammatable", "gamma"}:
        return np.asarray(display_get(display, "gamma table"), dtype=float).copy(), None
    if key == "gamut":
        xy = np.asarray(display_get(display, "primaries xy"), dtype=float)
        if xy.shape[0] > 0:
            black_like = np.sum(xy, axis=1) < 0.1
            xy = xy[~black_like]
        if xy.shape[0] > 0:
            xy = np.vstack((xy, xy[0, :]))
        return {"xy": np.asarray(xy, dtype=float)}, None
    if key == "gamut3d":
        n_primaries = int(display_get(display, "n primaries"))
        if n_primaries > 3:
            warnings.warn("Display has more than 3 primaries; only the first 3 are used.", RuntimeWarning, stacklevel=2)
        gtable = np.asarray(display_get(display, "gamma table"), dtype=float)
        n_samples = 30
        dac = np.rint(np.linspace(0.0, gtable.shape[0] - 1.0, n_samples + 1, dtype=float)[1:]).astype(int)
        rr, gg, bb = np.meshgrid(dac, dac, dac, indexing="ij")
        digital_rgb = np.stack((gg, rr, bb), axis=-1).reshape(-1, 1, 3)
        rgb = np.asarray(ie_lut_digital(digital_rgb, gtable[:, : min(3, gtable.shape[1])]), dtype=float).reshape(-1, 3)
        xyz = np.asarray(rgb @ np.asarray(display_get(display, "rgb2xyz"), dtype=float), dtype=float)
        lab = np.asarray(xyz_to_lab(xyz, np.asarray(display_get(display, "white xyz"), dtype=float).reshape(3)), dtype=float)
        hull = ConvexHull(lab)
        return {
            "LAB": lab.copy(),
            "rgb": rgb.copy(),
            "hull": np.asarray(hull.simplices, dtype=int).copy(),
        }, None
    if key == "psf":
        psf = np.asarray(display_get(display, "dixel image"), dtype=float)
        if psf.size == 0:
            raise ValueError("No psf for this display")
        n_primaries = min(int(display_get(display, "n primaries")), 3)
        dixel_size = np.asarray(display_get(display, "dixel size"), dtype=float).reshape(2)
        spacing_mm = float(display_get(display, "meters per dot", "mm"))
        x = (np.arange(1, int(dixel_size[1]) + 1, dtype=float) * spacing_mm)
        y = (np.arange(1, int(dixel_size[0]) + 1, dtype=float) * spacing_mm)
        x = x - np.mean(x)
        y = y - np.mean(y)
        srgb = np.asarray(display_get(display, "primaries rgb"), dtype=float).T
        normalized_psf = np.asarray(psf[:, :, :n_primaries], dtype=float).copy()
        for index in range(normalized_psf.shape[2]):
            peak = float(np.max(normalized_psf[:, :, index]))
            if peak > 0.0:
                normalized_psf[:, :, index] /= peak
        return {
            "x": x.copy(),
            "y": y.copy(),
            "psf": normalized_psf,
            "srgb": srgb[:, :n_primaries].copy(),
        }, None
    raise UnsupportedOptionError("displayPlot", param)


def _normalize_target_size(value: Any) -> tuple[int, int]:
    target = np.asarray(value, dtype=int).reshape(-1)
    if target.size == 1:
        target = np.repeat(target, 2)
    if target.size < 2:
        raise ValueError("Target size must provide one or two integer values.")
    return (max(int(target[0]), 1), max(int(target[1]), 1))


def _resize_dixel_image(dixel_image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(dixel_image, dtype=float)
    if image.shape[:2] == target_size:
        return image.copy()
    factors = (target_size[0] / max(image.shape[0], 1), target_size[1] / max(image.shape[1], 1), 1.0)
    resized = np.asarray(zoom(image, factors, order=1), dtype=float)
    resized[resized < 0.0] = 0.0
    scale = np.prod(target_size) / np.maximum(np.sum(resized, axis=(0, 1), keepdims=True), 1.0e-12)
    return resized * scale


def _display_render_function(value: Any) -> Any:
    if callable(value):
        return value
    normalized = param_format(str(value))
    if normalized in {"renderoledsamsung", "render_oled_samsung"}:
        return render_oled_samsung
    if normalized in {"renderlcdsamsungrgbw", "render_lcd_samsung_rgbw"}:
        return render_lcd_samsung_rgbw
    raise UnsupportedOptionError("displayCompute", str(value))


def _display_compute_input(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        payload = np.asarray(iio.imread(Path(image)), dtype=float)
    else:
        payload = np.asarray(image)
    if np.issubdtype(payload.dtype, np.integer):
        info = np.iinfo(payload.dtype)
        return np.asarray(payload, dtype=float) / float(info.max)
    return np.asarray(payload, dtype=float)


def display_compute(
    display: Display | str,
    image: Any,
    sz: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, Display]:
    """Compute a MATLAB-style upsampled display image from RGB input."""

    if display is None:
        raise ValueError("display required")
    if image is None:
        raise ValueError("Input image required")

    current_display = display_create(display, asset_store=asset_store) if isinstance(display, str) else display
    image_array = _display_compute_input(image)
    n_primaries = int(display_get(current_display, "n primaries"))

    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, None], n_primaries, axis=2)
    elif image_array.ndim != 3:
        raise ValueError("displayCompute expects a 2-D grayscale image or 3-D image cube.")
    elif image_array.shape[2] == 1 and n_primaries > 1:
        image_array = np.repeat(image_array, n_primaries, axis=2)
    elif image_array.shape[2] != n_primaries:
        raise ValueError("displayCompute image channels must match the number of display primaries.")

    pixels_per_dixel = np.asarray(display_get(current_display, "pixels per dixel"), dtype=int).reshape(2)
    render_function = display_get(current_display, "render function")
    target_size = display_get(current_display, "dixel size") if sz is None else _normalize_target_size(sz)
    target_size = tuple(np.asarray(target_size, dtype=int).reshape(2))
    oversample = np.rint(np.asarray(target_size, dtype=float) / np.maximum(pixels_per_dixel.astype(float), 1.0)).astype(int)
    if np.any(oversample <= 0):
        raise ValueError("bad up-sampling sz")

    dixel_image = np.asarray(display_get(current_display, "dixel image", target_size), dtype=float)
    if dixel_image.size == 0:
        raise ValueError("psf not defined for display")
    if np.min(dixel_image) < 0.0:
        raise ValueError("psfs values should be non-negative")

    if np.any(pixels_per_dixel > 1) and not render_function:
        raise ValueError("Render algorithm is required")

    if render_function:
        renderer = _display_render_function(render_function)
        out_image = np.asarray(renderer(image_array, current_display, target_size, asset_store=asset_store), dtype=float)
    else:
        out_image = np.repeat(np.repeat(image_array, int(oversample[0]), axis=0), int(oversample[1]), axis=1)

    rows, cols = image_array.shape[:2]
    expected_rows = rows * int(oversample[0])
    expected_cols = cols * int(oversample[1])
    if out_image.shape[0] != expected_rows or out_image.shape[1] != expected_cols:
        raise ValueError("bad outImage size")

    if rows % max(int(pixels_per_dixel[0]), 1) != 0 or cols % max(int(pixels_per_dixel[1]), 1) != 0:
        raise ValueError("Input image dimensions must be divisible by pixels per dixel.")

    tiled_dixel = np.tile(
        dixel_image,
        (rows // max(int(pixels_per_dixel[0]), 1), cols // max(int(pixels_per_dixel[1]), 1), 1),
    )
    return np.asarray(out_image * tiled_dixel, dtype=float), current_display


def _display_control_map(
    display: Display,
    *,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pixels_per_dixel = np.asarray(display_get(display, "pixels per dixel"), dtype=int).reshape(2)
    control_map = np.asarray(display_get(display, "dixel control map"))
    if control_map.ndim != 3:
        raise ValueError("Display dixel control map must be a 3-D array.")
    if target_size is None:
        return pixels_per_dixel, control_map
    target_rows, target_cols = (int(target_size[0]), int(target_size[1]))
    row_idx = np.clip(np.rint(np.linspace(0.0, control_map.shape[0] - 1.0, target_rows)).astype(int), 0, control_map.shape[0] - 1)
    col_idx = np.clip(np.rint(np.linspace(0.0, control_map.shape[1] - 1.0, target_cols)).astype(int), 0, control_map.shape[1] - 1)
    return pixels_per_dixel, control_map[row_idx][:, col_idx, :]


def render_oled_samsung(
    in_img: Any,
    display: Display | None = None,
    sz: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    if in_img is None:
        raise ValueError("input image required")
    current_display = display_create("OLED-Samsung", asset_store=asset_store) if display is None else display
    image = np.asarray(in_img, dtype=float)
    if image.ndim != 3:
        raise ValueError("render_oled_samsung expects an HxWxC image.")
    target_size = None if sz is None else tuple(np.asarray(sz, dtype=int).reshape(2))
    pixels_per_dixel, control_map = _display_control_map(current_display, target_size=target_size)
    if image.shape[0] % pixels_per_dixel[0] != 0 or image.shape[1] % pixels_per_dixel[1] != 0:
        raise ValueError("Input image dimensions must be divisible by pixels per dixel.")
    tile_rows, tile_cols = control_map.shape[:2]
    output = np.zeros(
        (
            (image.shape[0] // pixels_per_dixel[0]) * tile_rows,
            (image.shape[1] // pixels_per_dixel[1]) * tile_cols,
            image.shape[2],
        ),
        dtype=float,
    )
    for primary in range(image.shape[2]):
        control = np.asarray(control_map[:, :, primary], dtype=int) - 1
        for row in range(0, image.shape[0], pixels_per_dixel[0]):
            out_row = (row // pixels_per_dixel[0]) * tile_rows
            for col in range(0, image.shape[1], pixels_per_dixel[1]):
                out_col = (col // pixels_per_dixel[1]) * tile_cols
                block = image[row : row + pixels_per_dixel[0], col : col + pixels_per_dixel[1], primary]
                output[out_row : out_row + tile_rows, out_col : out_col + tile_cols, primary] = block.reshape(-1, order="F")[control]
    return output


def render_lcd_samsung_rgbw(
    in_img: Any,
    display: Display | None = None,
    sz: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    if in_img is None:
        raise ValueError("input image required")
    current_display = display_create("LCD-Samsung-RGBW", asset_store=asset_store) if display is None else display
    image = np.asarray(in_img, dtype=float)
    if image.ndim != 3:
        raise ValueError("render_lcd_samsung_rgbw expects an HxWxC image.")
    target_size = None if sz is None else tuple(np.asarray(sz, dtype=int).reshape(2))
    pixels_per_dixel, control_map = _display_control_map(current_display, target_size=target_size)
    if image.shape[0] % pixels_per_dixel[0] != 0 or image.shape[1] % pixels_per_dixel[1] != 0:
        raise ValueError("Input image dimensions must be divisible by pixels per dixel.")
    tile_rows, tile_cols = control_map.shape[:2]
    n_primaries = image.shape[2]
    output = np.zeros(
        (
            (image.shape[0] // pixels_per_dixel[0]) * tile_rows,
            (image.shape[1] // pixels_per_dixel[1]) * tile_cols,
            n_primaries,
        ),
        dtype=float,
    )
    for row in range(0, image.shape[0], pixels_per_dixel[0]):
        out_row = (row // pixels_per_dixel[0]) * tile_rows
        for col in range(0, image.shape[1], pixels_per_dixel[1]):
            out_col = (col // pixels_per_dixel[1]) * tile_cols
            block = image[row : row + pixels_per_dixel[0], col : col + pixels_per_dixel[1], :]
            rgb_levels = block[:, :, : min(3, n_primaries)].reshape(-1, order="F")[:3]
            white_level = float(np.min(rgb_levels))
            for primary in range(n_primaries):
                control = np.asarray(control_map[:, :, primary], dtype=float)
                if primary < 3:
                    output[out_row : out_row + tile_rows, out_col : out_col + tile_cols, primary] = (rgb_levels[primary] - white_level) * control
                else:
                    output[out_row : out_row + tile_rows, out_col : out_col + tile_cols, primary] = white_level * control
    return output


def display_set_max_luminance(display: Display, max_luminance: float) -> Display:
    """Scale primary SPDs so the display white point reaches the requested luminance."""

    current = display.clone()
    current_white = np.asarray(display_get(current, "white point"), dtype=float).reshape(3)
    current_y = float(current_white[1])
    scale = float(max_luminance) / max(current_y, 1e-12)
    spd = np.asarray(display_get(current, "spd"), dtype=float)
    return display_set(current, "spd", spd * scale)


def display_set_white_point(
    display: Display,
    white_xy: Any,
    format: str = "xyz",
) -> Display:
    """Scale display primaries so the white point chromaticity matches the target xy."""

    normalized = param_format(format or "xyz")
    if normalized != "xyz":
        raise UnsupportedOptionError("displaySetWhitePoint", format)

    chromaticity = np.asarray(white_xy, dtype=float).reshape(-1)
    if chromaticity.size != 2:
        raise ValueError("displaySetWhitePoint requires [x, y] chromaticity.")

    current = display.clone()
    current_y = float(np.asarray(display_get(current, "white point"), dtype=float).reshape(3)[1])
    x, y = float(chromaticity[0]), float(chromaticity[1])
    if y <= 0.0:
        raise ValueError("displaySetWhitePoint requires positive y chromaticity.")
    xyz_white = np.array([x * current_y / y, current_y, (1.0 - x - y) * current_y / y], dtype=float)
    rgb2xyz = np.asarray(display_get(current, "rgb2xyz"), dtype=float)
    scale = xyz_white @ np.linalg.inv(rgb2xyz)
    spd = np.asarray(display_get(current, "spd"), dtype=float)
    return display_set(current, "spd", spd @ np.diag(scale))


def display_get(display: Display, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    if key == "type":
        return display.type
    if key == "name":
        return display.name
    if key in {"isemissive"}:
        return bool(display.fields.get("is_emissive", True))
    if key in {"gtable", "dv2intensity", "gamma", "gammatable", "gammatable"}:
        return np.asarray(display.fields["gamma"], dtype=float)
    if key == "psfs":
        psfs = display.fields.get("psfs")
        return None if psfs is None else np.asarray(psfs, dtype=float)
    if key in {"inversegamma", "inversegammatable"}:
        gamma = np.asarray(display_get(display, "gamma table"), dtype=float)
        n_steps = int(args[0]) if args else gamma.shape[0]
        return _inverse_gamma_table(gamma[:, : min(3, gamma.shape[1])], n_steps)
    if key in {"bits", "dacsize"}:
        if key == "dacsize" and display.fields.get("dacsize") is not None:
            return int(display.fields["dacsize"])
        gamma = np.asarray(display_get(display, "gamma table"), dtype=float)
        return int(round(np.log2(gamma.shape[0])))
    if key in {"nlevels"}:
        return int(2 ** int(display_get(display, "bits")))
    if key in {"levels"}:
        return np.arange(int(display_get(display, "nlevels")), dtype=int)
    if key == "darklevel":
        gamma = np.asarray(display_get(display, "gamma table"), dtype=float)
        return np.asarray(gamma[0, :], dtype=float)
    if key in {"wave", "wavelength"}:
        return np.asarray(display.fields["wave"], dtype=float)
    if key == "binwidth":
        wave = np.asarray(display.fields["wave"], dtype=float)
        if wave.size < 2:
            return 1.0
        return float(wave[1] - wave[0])
    if key == "nwave":
        return int(np.asarray(display.fields["wave"], dtype=float).size)
    if key == "nprimaries":
        return int(np.asarray(display_get(display, "spd"), dtype=float).shape[1])
    if key in {"spd", "spdprimaries"}:
        spd = np.asarray(display.fields["spd"], dtype=float)
        if spd.shape[0] != int(display_get(display, "nwave")):
            spd = spd.T
        if args:
            wave = np.asarray(args[0], dtype=float).reshape(-1)
            spd = interp_spectra(np.asarray(display.fields["wave"], dtype=float), spd, wave)
        return spd
    if key == "rgbspd":
        spd = np.asarray(display_get(display, "spd", *(args[:1] if args else ())), dtype=float)
        return spd[:, : min(3, spd.shape[1])]
    if key in {"rgb2xyz", "lrgb2xyz"}:
        return _display_rgb2xyz(display)
    if key == "drgb2xyz":
        if not args:
            raise ValueError("displayGet(..., 'drgb2xyz') requires a digital RGB image.")
        digital_rgb = np.asarray(args[0], dtype=float)
        if digital_rgb.ndim != 3 or digital_rgb.shape[2] < 3:
            raise ValueError("displayGet(..., 'drgb2xyz') expects an RGB-format digital image.")
        linear_rgb = np.asarray(ie_lut_digital(digital_rgb[:, :, :3], display_get(display, "gamma table")), dtype=float)
        return np.asarray(image_linear_transform(linear_rgb, display_get(display, "rgb2xyz")), dtype=float)
    if key == "rgb2lms":
        return np.asarray(xyz_to_lms(np.asarray(display_get(display, "rgb2xyz"), dtype=float)), dtype=float)
    if key == "xyz2rgb":
        return np.linalg.inv(np.asarray(display_get(display, "rgb2xyz"), dtype=float))
    if key == "whitespd":
        spd = np.asarray(display_get(display, "spd", *(args[:1] if args else ())), dtype=float)
        return np.sum(spd, axis=1)
    if key in {"whitepoint", "whitexyz"}:
        white_xyz = np.sum(np.asarray(display_get(display, "rgb2xyz"), dtype=float), axis=0, dtype=float)
        return np.asarray(white_xyz, dtype=float).reshape(3)
    if key == "blackxyz":
        wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
        black_spd = np.asarray(display_get(display, "black spd"), dtype=float).reshape(1, -1)
        return np.asarray(xyz_from_energy(black_spd, wave), dtype=float).reshape(3)
    if key in {"whitexy", "whitechromaticity"}:
        white_xyz = np.asarray(display_get(display, "white point"), dtype=float).reshape(3)
        return np.asarray(white_xyz[:2] / max(float(np.sum(white_xyz)), 1e-12), dtype=float)
    if key == "whitelms":
        return np.sum(np.asarray(display_get(display, "rgb2lms"), dtype=float), axis=0, dtype=float)
    if key == "primariesxyz":
        wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
        spd = np.asarray(display_get(display, "spd primaries"), dtype=float).T
        return np.asarray(xyz_from_energy(spd, wave), dtype=float)
    if key in {"primariesrgb", "primariessrgb"}:
        xyz = np.asarray(display_get(display, "primaries xyz"), dtype=float)
        return np.asarray(xyz_to_srgb(xyz.reshape(1, xyz.shape[0], 3)), dtype=float).reshape(xyz.shape[0], 3)
    if key == "primariesxy":
        return np.asarray(chromaticity_xy(np.asarray(display_get(display, "primaries xyz"), dtype=float)), dtype=float)
    if key in {"blackspd", "blackradiance", "ambientspd"}:
        ambient = _display_ambient(display)
        if args:
            wave = np.asarray(args[0], dtype=float).reshape(-1)
            ambient = interp_spectra(np.asarray(display.fields["wave"], dtype=float), ambient[:, None], wave).reshape(-1)
        return ambient
    if key in {"dpi", "ppi"}:
        return float(display.fields["dpi"])
    if key in {"metersperdot"}:
        value = 0.0254 / max(float(display.fields["dpi"]), 1e-12)
        if args:
            value *= float(ie_unit_scale_factor(str(args[0])))
        return float(value)
    if key in {"dotspermeter"}:
        value = 1.0 / max(float(display_get(display, "meters per dot")), 1e-12)
        if args:
            value *= float(ie_unit_scale_factor(str(args[0])))
        return float(value)
    if key in {"dotsperdeg", "sampperdeg"}:
        dist = float(display_get(display, "viewing distance"))
        meters_per_dot = float(display_get(display, "meters per dot"))
        deg_per_dot = np.degrees(np.arctan2(meters_per_dot, max(dist, 1.0e-12)))
        return float(np.round(1.0 / max(deg_per_dot, 1.0e-12)))
    if key in {"degperdot", "degperpixel"}:
        dist = float(display_get(display, "viewing distance"))
        meters_per_dot = float(display_get(display, "meters per dot"))
        return float(np.degrees(np.arctan2(meters_per_dot, max(dist, 1.0e-12))))
    if key in {"dist", "distance", "viewingdistance"}:
        return float(display.fields["dist"])
    if key == "size":
        return _display_size_m(display)
    if key == "dixel":
        return display.fields.get("dixel")
    if key == "pixelsperdixel":
        dixel = display.fields.get("dixel") or {}
        return dixel.get("n_pixels")
    if key == "dixelsize":
        dixel = display.fields.get("dixel") or {}
        intensity = dixel.get("intensity_map")
        return None if intensity is None else intensity.shape[:2]
    if key in {"oversample", "osample"}:
        dixel_size = np.asarray(display_get(display, "dixel size"), dtype=float).reshape(2)
        pixels_per_dixel = np.asarray(display_get(display, "pixels per dixel"), dtype=float).reshape(2)
        return dixel_size / np.maximum(pixels_per_dixel, 1.0)
    if key == "samplespacing":
        sample_spacing = np.asarray(display_get(display, "meters per dot"), dtype=float) / np.asarray(
            display_get(display, "dixel size"),
            dtype=float,
        )
        sample_spacing = sample_spacing * np.asarray(display_get(display, "pixels per dixel"), dtype=float)
        if args:
            sample_spacing = sample_spacing * float(ie_unit_scale_factor(str(args[0])))
        return np.asarray(sample_spacing, dtype=float).reshape(2)
    if key in {"fillfactor", "fillingfactor", "subpixelfilling"}:
        dixel_image = np.asarray(display_get(display, "dixel image"), dtype=float)
        max_per_primary = np.max(dixel_image, axis=(0, 1), keepdims=True)
        normalized = np.divide(dixel_image, np.maximum(max_per_primary, 1.0e-12))
        filled = normalized > 0.2
        return np.sum(filled, axis=(0, 1), dtype=float) / float(dixel_image.shape[0] * dixel_image.shape[1])
    if key == "subpixelspd":
        spd = np.asarray(display_get(display, "spd"), dtype=float)
        fill_factor = np.asarray(display_get(display, "fill factor"), dtype=float).reshape(1, -1)
        return spd / np.maximum(fill_factor, 1.0e-12)
    if key in {"dixelintensitymap", "dixelimage"}:
        dixel = display.fields.get("dixel") or {}
        intensity = dixel.get("intensity_map")
        if intensity is None:
            return None
        if args:
            return _resize_dixel_image(np.asarray(intensity, dtype=float), _normalize_target_size(args[0]))
        return intensity
    if key == "dixelcontrolmap":
        dixel = display.fields.get("dixel") or {}
        control_map = dixel.get("control_map")
        if control_map is None:
            return None
        if args:
            target_size = _normalize_target_size(args[0])
            factors = (
                target_size[0] / max(control_map.shape[0], 1),
                target_size[1] / max(control_map.shape[1], 1),
                1.0,
            )
            return np.asarray(zoom(np.asarray(control_map, dtype=float), factors, order=0), dtype=float)
        return control_map
    if key == "renderfunction":
        dixel = display.fields.get("dixel") or {}
        return dixel.get("render_function")
    if key == "comment":
        return display.fields.get("comment")
    if key == "refreshrate":
        return display.fields.get("refresh_rate_hz")
    if key == "image":
        return display.fields.get("image")
    if key in {"maxluminance", "peakluminance", "peakdisplayluminance"}:
        return float(np.asarray(display_get(display, "white point"), dtype=float).reshape(3)[1])
    if key in {"contrast", "peakcontrast"}:
        peak_luminance = float(display_get(display, "peak luminance"))
        dark_luminance = float(display_get(display, "dark luminance"))
        return float(np.inf) if dark_luminance <= 0.0 else peak_luminance / dark_luminance
    if key in {"darkluminance", "blackluminance"}:
        return float(np.asarray(display_get(display, "black xyz"), dtype=float).reshape(3)[1])
    raise KeyError(f"Unsupported displayGet parameter: {parameter}")


def display_set(display: Display, parameter: str, value: Any, *args: Any) -> Display:
    key = param_format(parameter)
    if key == "name":
        display.name = str(value)
        return display
    if key == "type":
        display.type = str(value)
        return display
    if key in {"gtable", "dv2intensity", "gamma", "gammatable"}:
        if isinstance(value, str) and param_format(value) == "linear":
            size = int(args[0]) if args else np.asarray(display.fields["gamma"], dtype=float).shape[0]
            axis = np.linspace(0.0, 1.0, size, dtype=float)[:, None]
            value = np.repeat(axis, int(display_get(display, "nprimaries")), axis=1)
        display.fields["gamma"] = np.asarray(value, dtype=float)
        return display
    if key == "psfs":
        display.fields["psfs"] = np.asarray(value, dtype=float)
        return display
    if key == "dacsize":
        display.fields["dacsize"] = int(value)
        return display
    if key in {"wave", "wavelength"}:
        new_wave = np.asarray(value, dtype=float).reshape(-1)
        if "wave" in display.fields and not np.array_equal(new_wave, np.asarray(display.fields["wave"], dtype=float)):
            display.fields["spd"] = interp_spectra(
                np.asarray(display.fields["wave"], dtype=float),
                np.asarray(display_get(display, "spd"), dtype=float),
                new_wave,
            )
            ambient = _display_ambient(display)
            display.fields["ambient_spd"] = interp_spectra(
                np.asarray(display.fields["wave"], dtype=float),
                ambient[:, None],
                new_wave,
            ).reshape(-1)
        display.fields["wave"] = new_wave
        return display
    if key in {"spd", "spdprimaries"}:
        spd = np.asarray(value, dtype=float)
        if spd.ndim != 2:
            raise ValueError("Display SPD must be a 2D matrix.")
        if spd.shape[0] != int(display_get(display, "nwave")) and spd.shape[1] == int(display_get(display, "nwave")):
            spd = spd.T
        display.fields["spd"] = spd
        return display
    if key in {"dpi", "ppi"}:
        display.fields["dpi"] = float(value)
        return display
    if key == "size":
        display.fields["size_m"] = np.asarray(value, dtype=float).reshape(2)
        return display
    if key in {"viewingdistance", "dist"}:
        display.fields["dist"] = float(value)
        return display
    if key == "refreshrate":
        display.fields["refresh_rate_hz"] = float(value)
        return display
    if key == "dixel":
        display.fields["dixel"] = dict(value)
        return display
    if key in {"dixelintensitymap", "dixelimage"}:
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["intensity_map"] = np.asarray(value, dtype=float)
        return display
    if key == "dixelcontrolmap":
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["control_map"] = np.asarray(value, dtype=float)
        return display
    if key == "pixelsperdixel":
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["n_pixels"] = value
        return display
    if key == "renderfunction":
        display.fields.setdefault("dixel", {})
        display.fields["dixel"]["render_function"] = value
        return display
    if key == "comment":
        display.fields["comment"] = str(value)
        return display
    if key in {"rgb", "image"}:
        display.fields["image"] = value
        return display
    if key == "ambientspd":
        display.fields["ambient_spd"] = np.asarray(value, dtype=float).reshape(-1)
        return display
    raise KeyError(f"Unsupported displaySet parameter: {parameter}")


displayConvert = display_convert
displayPT2ISET = display_pt2iset
displayReflectance = display_reflectance
displayMaxContrast = display_max_contrast
ieCalculateMonitorDPI = ie_calculate_monitor_dpi
ieLUTDigital = ie_lut_digital
mperdot2dpi = mperdot2dpi
displayCreate = display_create
displayList = display_list
displayDescription = display_description
displayShowImage = display_show_image
displayPlot = display_plot
displayCompute = display_compute
displaySetMaxLuminance = display_set_max_luminance
displaySetWhitePoint = display_set_white_point
ieReadSpectra = ie_read_spectra
ieUnitScaleFactor = ie_unit_scale_factor
srgb2xyz = srgb_to_xyz
displayGet = display_get
displaySet = display_set
xyz2lms = xyz_to_lms
xyz2srgb = xyz_to_srgb
