"""Scene creation and manipulation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import imageio.v3 as iio
import numpy as np
from scipy.io import savemat
from scipy.ndimage import map_coordinates, zoom
from scipy.signal import convolve2d

from .assets import AssetStore, ie_read_spectra
from .color import init_default_spectrum, lrgb_to_srgb, luminance_from_photons
from .exceptions import MissingAssetError, UnsupportedOptionError
from .metrics import chromaticity_xy, delta_e_ab, xyz_from_energy
from .session import track_session_object
from .types import Scene, SessionContext
from .utils import (
    DEFAULT_WAVE,
    blackbody,
    energy_to_quanta,
    hc_basis,
    image_increase_image_rgb_size,
    interp_spectra,
    param_format,
    quanta_to_energy,
    rgb_to_xw_format,
    srgb_to_xyz,
    unit_frequency_list,
    xw_to_rgb_format,
    xyz_to_srgb,
)

DEFAULT_DISTANCE_M = 1.2
DEFAULT_FOV_DEG = 10.0
_MACBETH_GRID = (4, 6)
_SPATIAL_UNIT_SCALE = {
    "meters": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "millimeters": 1e3,
    "millimeter": 1e3,
    "mm": 1e3,
    "microns": 1e6,
    "micron": 1e6,
    "um": 1e6,
}
_SCENE_SDR_DEPOSIT_URLS = {
    "isetcambitterli": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/bitterli",
    "isetcampharr": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/pbrtv4",
    "isetcamiset3d": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/iset3d",
}
_SCENE_SDR_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "scenes" / "web"

_SCENE_LIST_TEXT = """ISETCam scene types with optional parameters

Supported Python sceneCreate families include:
  list / scenelist
  empty
  macbeth d65 / d50 / illc / fluorescent / tungsten / ee_ir
  reflectance chart
  monochrome / unispectral / multispectral / hyperspectral / rgb
  rings rays / harmonic / sweep frequency / freq orient pattern / moire orient
  line d65 / line ee / line ep / bar / vernier
  point array / grid lines / radial lines / slanted edge / checkerboard
  zone plate / linear intensity ramp
  uniform equal energy / uniform equal photon / uniform d65 / uniform bb
  white noise / letter / scene from file
"""


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _wave_or_default(wave: Any | None) -> np.ndarray:
    if wave is None:
        return DEFAULT_WAVE.copy()
    array = np.asarray(wave, dtype=float).reshape(-1)
    if array.size == 0:
        return DEFAULT_WAVE.copy()
    return array


def _macbeth_patch_size_arg(value: Any | None) -> int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return int(np.rint(float(array.reshape(-1)[0])))


def _is_empty_scene_dispatch_placeholder(value: Any | None) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _is_empty_scene_dispatch_collection_placeholder(value: Any | None) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _scene_dispatch_int_arg(value: Any | None, default: int) -> int:
    if _is_empty_scene_dispatch_placeholder(value):
        return int(default)
    return int(np.rint(np.asarray(value, dtype=float).reshape(-1)[0]))


def _scene_dispatch_float_arg(value: Any | None, default: float) -> float:
    if _is_empty_scene_dispatch_placeholder(value):
        return float(default)
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _scene_dispatch_text_arg(value: Any | None, default: str) -> str:
    if _is_empty_scene_dispatch_placeholder(value):
        return str(default)
    return str(value)


def _scene_dispatch_path_arg(value: Any | None, default: str) -> str:
    if _is_empty_scene_dispatch_placeholder(value):
        return str(default)
    return str(value)


def _looks_like_scene_size_arg(value: Any | None) -> bool:
    if value is None:
        return False
    if np.isscalar(value):
        return float(np.asarray(value, dtype=float)) > 0.0
    array = np.asarray(value, dtype=float).reshape(-1)
    return 0 < array.size <= 2 and bool(np.all(array > 0.0))


def _looks_like_wave_arg(value: Any | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return False
    return bool(np.all(array >= 100.0))


def _looks_like_monochromatic_wavelength_arg(value: Any | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return False
    return bool(np.all(array >= 300.0))


def _scene_image_input(input_data: Any, *, asset_store: AssetStore | None = None) -> tuple[np.ndarray, str, str]:
    if isinstance(input_data, (str, Path)):
        path = Path(input_data).expanduser()
        if not path.exists() and asset_store is not None:
            candidates = [
                path,
                Path("data/images/rgb") / path,
                Path("data/images") / path,
            ]
            resolved = None
            for candidate in candidates:
                try:
                    resolved = asset_store.resolve(candidate)
                    break
                except MissingAssetError:
                    continue
            if resolved is None:
                snapshot_root = asset_store.ensure()
                matches = list(snapshot_root.rglob(path.name))
                if matches:
                    resolved = matches[0]
            if resolved is not None:
                path = resolved
        image = np.asarray(iio.imread(path), dtype=float)
        return image, str(path), path.stem
    image = np.asarray(input_data, dtype=float)
    return image, "numerical", "numerical"


def _mat_struct_field(structure: Any, field: str, default: Any = None) -> Any:
    if structure is None:
        return default
    return getattr(structure, field, default)


def _scene_struct_value(structure: Any, field: str, default: Any = None) -> Any:
    if isinstance(structure, dict):
        return structure.get(field, default)
    return _mat_struct_field(structure, field, default)


def _resample_wave_last(values: np.ndarray, source_wave_nm: np.ndarray, target_wave_nm: np.ndarray) -> np.ndarray:
    wave_first = np.moveaxis(np.asarray(values, dtype=float), -1, 0)
    resampled = interp_spectra(np.asarray(source_wave_nm, dtype=float), wave_first, np.asarray(target_wave_nm, dtype=float))
    return np.moveaxis(np.asarray(resampled, dtype=float), 0, -1)


def _multispectral_scene_input(
    input_data: Any,
    wave: np.ndarray | None,
    *,
    asset_store: AssetStore,
) -> dict[str, Any]:
    if not isinstance(input_data, (str, Path)):
        raise ValueError("scene_from_file(..., 'multispectral', ...) requires a MAT file path.")

    path = asset_store.resolve(Path(input_data).expanduser())
    data = asset_store.load_mat(path)
    target_wave = None if wave is None else np.asarray(wave, dtype=float).reshape(-1)

    if "mcCOEF" in data:
        basis_struct = data["basis"]
        source_wave = np.asarray(_mat_struct_field(basis_struct, "wave"), dtype=float).reshape(-1)
        basis_matrix = np.asarray(_mat_struct_field(basis_struct, "basis"), dtype=float)
        if basis_matrix.ndim == 1:
            basis_matrix = basis_matrix.reshape(-1, 1)
        if basis_matrix.shape[0] != source_wave.size and basis_matrix.shape[1] == source_wave.size:
            basis_matrix = basis_matrix.T
        wave_nm = source_wave if target_wave is None else target_wave
        if target_wave is not None and not np.array_equal(source_wave, target_wave):
            basis_matrix = interp_spectra(source_wave, basis_matrix, target_wave)
        mc_coef = np.asarray(data["mcCOEF"], dtype=float)
        if mc_coef.ndim == 2:
            mc_coef = mc_coef[:, :, np.newaxis]
        photons = np.tensordot(mc_coef, np.asarray(basis_matrix, dtype=float).T, axes=([2], [0]))
        if "imgMean" in data:
            image_mean = np.asarray(data["imgMean"], dtype=float).reshape(-1)
            if image_mean.size == 0:
                image_mean = np.array([], dtype=float)
            elif image_mean.size != source_wave.size:
                raise ValueError("imgMean wavelength length does not match basis wavelength samples.")
            if image_mean.size > 0 and target_wave is not None and not np.array_equal(source_wave, target_wave):
                image_mean = interp_spectra(source_wave, image_mean, target_wave).reshape(-1)
            if image_mean.size > 0:
                photons = photons + image_mean.reshape(1, 1, -1)
    else:
        if "photons" in data:
            photons = np.asarray(data["photons"], dtype=float)
        elif "data" in data:
            photons = np.asarray(data["data"], dtype=float)
        else:
            raise ValueError("Multispectral MAT file must contain either 'mcCOEF', 'photons', or 'data'.")
        if "wave" in data:
            source_wave = np.asarray(data["wave"], dtype=float).reshape(-1)
        elif "wavelength" in data:
            source_wave = np.asarray(data["wavelength"], dtype=float).reshape(-1)
        else:
            raise ValueError("Multispectral MAT file is missing wavelength samples.")
        if photons.ndim == 3 and photons.shape[-1] != source_wave.size and photons.shape[0] == source_wave.size:
            photons = np.moveaxis(photons, 0, -1)
        wave_nm = source_wave if target_wave is None else target_wave
        if target_wave is not None and not np.array_equal(source_wave, target_wave):
            photons = _resample_wave_last(photons, source_wave, target_wave)

    illuminant = data.get("illuminant")
    illuminant_photons: np.ndarray | None = None
    illuminant_energy: np.ndarray | None = None
    illuminant_format = "spectral"
    if illuminant is not None:
        spectrum_struct = _mat_struct_field(illuminant, "spectrum")
        if spectrum_struct is not None:
            illuminant_wave = np.asarray(_mat_struct_field(spectrum_struct, "wave", wave_nm), dtype=float).reshape(-1)
        else:
            illuminant_wave = np.asarray(
                _mat_struct_field(illuminant, "wave", _mat_struct_field(illuminant, "wavelength", wave_nm)),
                dtype=float,
            ).reshape(-1)
        illuminant_data = _mat_struct_field(illuminant, "data")
        stored_photons = _mat_struct_field(illuminant_data, "photons")
        stored_energy = _mat_struct_field(illuminant_data, "energy")
        if stored_energy is None and isinstance(illuminant_data, np.ndarray):
            stored_energy = illuminant_data
        if stored_energy is None:
            stored_energy = _mat_struct_field(illuminant, "data")
        if stored_energy is None:
            stored_energy = _mat_struct_field(illuminant, "energy")
        if stored_photons is not None:
            illuminant_photons = np.asarray(stored_photons, dtype=float)
            if illuminant_photons.ndim == 3 and illuminant_photons.shape[-1] != illuminant_wave.size and illuminant_photons.shape[0] == illuminant_wave.size:
                illuminant_photons = np.moveaxis(illuminant_photons, 0, -1)
            if not np.array_equal(illuminant_wave, wave_nm):
                if illuminant_photons.ndim == 1:
                    illuminant_photons = interp_spectra(illuminant_wave, illuminant_photons, wave_nm).reshape(-1)
                else:
                    illuminant_photons = _resample_wave_last(illuminant_photons, illuminant_wave, wave_nm)
            illuminant_format = "spatial spectral" if illuminant_photons.ndim == 3 else "spectral"
            illuminant_energy = quanta_to_energy(illuminant_photons, wave_nm)
            if np.asarray(illuminant_energy).ndim == 3:
                illuminant_energy = np.mean(np.asarray(illuminant_energy, dtype=float), axis=(0, 1))
        elif stored_energy is not None:
            illuminant_energy = np.asarray(stored_energy, dtype=float)
            if illuminant_energy.ndim == 3 and illuminant_energy.shape[-1] != illuminant_wave.size and illuminant_energy.shape[0] == illuminant_wave.size:
                illuminant_energy = np.moveaxis(illuminant_energy, 0, -1)
            if not np.array_equal(illuminant_wave, wave_nm):
                if illuminant_energy.ndim == 1:
                    illuminant_energy = interp_spectra(illuminant_wave, illuminant_energy, wave_nm).reshape(-1)
                else:
                    illuminant_energy = _resample_wave_last(illuminant_energy, illuminant_wave, wave_nm)
            illuminant_format = "spatial spectral" if illuminant_energy.ndim == 3 else "spectral"
            illuminant_photons = energy_to_quanta(illuminant_energy, wave_nm)
            if np.asarray(illuminant_energy).ndim == 3:
                illuminant_energy = np.mean(np.asarray(illuminant_energy, dtype=float), axis=(0, 1))

    if illuminant_photons is None:
        illuminant_photons = np.maximum(np.mean(np.asarray(photons, dtype=float), axis=(0, 1)), 1e-12)
        illuminant_energy = quanta_to_energy(illuminant_photons, wave_nm)
        illuminant_format = "spectral"

    comment = data.get("comment")
    if comment is None:
        illuminant_comment = path.name
    elif hasattr(comment, "_fieldnames"):
        illuminant_comment = path.name
    else:
        illuminant_comment = str(comment)

    return {
        "photons": np.maximum(np.asarray(photons, dtype=float), 0.0),
        "wave": np.asarray(wave_nm, dtype=float).reshape(-1),
        "illuminant_photons": np.asarray(illuminant_photons, dtype=float),
        "illuminant_energy": np.asarray(illuminant_energy, dtype=float),
        "illuminant_format": illuminant_format,
        "illuminant_comment": illuminant_comment,
        "filename": str(path),
        "source_name": str(data.get("name", path.stem)),
        "distance_m": float(np.asarray(data.get("dist", DEFAULT_DISTANCE_M), dtype=float).reshape(-1)[0]),
        "fov_deg": float(np.asarray(data.get("fov", DEFAULT_FOV_DEG), dtype=float).reshape(-1)[0]),
    }


def _scene_display(display: Any, wave: np.ndarray | None, *, asset_store: AssetStore) -> Any:
    from .display import display_create, display_set

    if display is None:
        return display_create("default", asset_store=asset_store, wave=wave)
    if isinstance(display, str):
        return display_create(display, asset_store=asset_store, wave=wave)

    current = display
    if wave is None:
        return current
    display_wave = np.asarray(current.fields.get("wave", wave), dtype=float).reshape(-1)
    if not np.array_equal(display_wave, wave):
        current = display_set(current.clone(), "wave", wave)
    return current


def _prepare_display_image(image: np.ndarray, im_type: str, n_primaries: int) -> np.ndarray:
    current = np.asarray(image, dtype=float)
    if current.ndim == 2:
        current = current[:, :, None]
    if current.ndim != 3:
        raise ValueError("scene_from_file expects a 2D or 3D image array.")
    if current.shape[2] == 4:
        current = current[:, :, :3]

    normalized_type = param_format(im_type)
    if normalized_type in {"monochrome", "unispectral"}:
        if current.shape[2] > 1:
            current = np.mean(current[:, :, : min(3, current.shape[2])], axis=2, keepdims=True)
        current = np.repeat(current, n_primaries, axis=2)
    elif normalized_type == "rgb":
        if current.shape[2] == 1:
            current = np.repeat(current, n_primaries, axis=2)
        elif current.shape[2] < n_primaries:
            current = np.pad(current, ((0, 0), (0, 0), (0, n_primaries - current.shape[2])), mode="constant")
        elif current.shape[2] > n_primaries:
            current = current[:, :, :n_primaries]
    else:
        raise UnsupportedOptionError("sceneFromFile", im_type)
    return current


def _display_linear_rgb(image: np.ndarray, gamma_table: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma_table, dtype=float)
    if gamma.ndim != 2:
        raise ValueError("Display gamma table must be a 2D matrix.")

    current = np.asarray(image, dtype=float)
    max_value = float(np.max(current)) if current.size else 0.0
    if max_value <= 1.0:
        digital = np.rint(current * (gamma.shape[0] - 1))
    elif max_value <= 255.0:
        digital = np.rint(current / 255.0 * (gamma.shape[0] - 1))
    else:
        digital = np.rint(current / max(max_value, 1e-12) * (gamma.shape[0] - 1))

    digital = np.clip(digital.astype(int), 0, gamma.shape[0] - 1)
    linear = np.empty_like(current, dtype=float)
    for channel in range(current.shape[2]):
        gamma_channel = min(channel, gamma.shape[1] - 1)
        linear[:, :, channel] = gamma[digital[:, :, channel], gamma_channel]
    return linear


def _invalidate_scene_caches(scene: Scene) -> None:
    scene.fields.pop("luminance", None)
    scene.fields.pop("mean_luminance", None)


def _spatial_unit_scale(unit: Any) -> float:
    if unit is None:
        return 1.0
    return _SPATIAL_UNIT_SCALE.get(param_format(unit), 1.0)


def _scene_frequency_support(scene: Scene, unit: Any = "cyclesPerDegree") -> dict[str, np.ndarray]:
    normalized_unit = param_format(unit or "cyclesPerDegree")
    rows = int(scene.fields.get("rows", 0))
    cols = int(scene.fields.get("cols", 0))
    fov_width = float(scene.fields.get("fov_deg", DEFAULT_FOV_DEG))
    fov_height = float(scene.fields.get("vfov_deg", DEFAULT_FOV_DEG))

    if cols <= 0 or rows <= 0:
        cols = 128
        rows = 128
    if fov_width <= 0.0 or fov_height <= 0.0:
        fov_width = 30.0
        fov_height = 30.0

    max_frequency_cpd = np.array(
        [
            (cols / 2.0) / fov_width,
            (rows / 2.0) / fov_height,
        ],
        dtype=float,
    )

    if normalized_unit in {"cyclesperdegree", "cycperdeg", "cpd"}:
        max_frequency = max_frequency_cpd
    elif normalized_unit in _SPATIAL_UNIT_SCALE:
        deg_per_dist = float(scene_get(scene, "deg per dist", normalized_unit))
        max_frequency = max_frequency_cpd * deg_per_dist
    else:
        raise ValueError(f"Unknown scene spatial frequency units: {unit}")

    return {
        "fx": unit_frequency_list(cols) * max_frequency[0],
        "fy": unit_frequency_list(rows) * max_frequency[1],
    }


def _update_scene_geometry(scene: Scene) -> Scene:
    photons = np.asarray(scene.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    if photons.size == 0:
        return scene
    rows, cols = photons.shape[:2]
    distance_m = float(scene.fields.get("distance_m", DEFAULT_DISTANCE_M))
    fov_deg = float(scene.fields.get("fov_deg", DEFAULT_FOV_DEG))
    width_m = 2.0 * distance_m * np.tan(np.deg2rad(fov_deg) / 2.0)
    height_m = width_m * rows / max(cols, 1)
    vfov_deg = np.rad2deg(2.0 * np.arctan2(height_m / 2.0, distance_m))
    scene.fields.update(
        {
            "wave": np.asarray(scene.fields.get("wave", DEFAULT_WAVE), dtype=float),
            "distance_m": distance_m,
            "fov_deg": fov_deg,
            "vfov_deg": float(vfov_deg),
            "width_m": float(width_m),
            "height_m": float(height_m),
            "rows": int(rows),
            "cols": int(cols),
        }
    )
    return scene


def _scene_resize_array(data: np.ndarray, new_rows: int, new_cols: int) -> np.ndarray:
    current = np.asarray(data, dtype=float)
    if current.ndim == 2:
        factors = (new_rows / max(current.shape[0], 1), new_cols / max(current.shape[1], 1))
    elif current.ndim == 3:
        factors = (
            new_rows / max(current.shape[0], 1),
            new_cols / max(current.shape[1], 1),
            1.0,
        )
    else:
        raise ValueError("Scene resize only supports 2D or 3D arrays.")
    return zoom(current, factors, order=1)


def _scene_resize(scene: Scene, new_size: Any) -> Scene:
    values = np.asarray(new_size, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("scene resize requires a target size.")
    if values.size == 1:
        values = np.repeat(values, 2)
    new_rows = max(int(np.rint(values[0])), 1)
    new_cols = max(int(np.rint(values[1])), 1)

    resized = scene.clone()
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    resized = scene_set(resized, "photons", _scene_resize_array(photons, new_rows, new_cols))

    depth_map = scene.fields.get("depth_map_m")
    if depth_map is not None:
        resized.fields["depth_map_m"] = _scene_resize_array(np.asarray(depth_map, dtype=float), new_rows, new_cols)

    illuminant_photons = scene.fields.get("illuminant_photons")
    if illuminant_photons is not None:
        illuminant_array = np.asarray(illuminant_photons, dtype=float)
        if illuminant_array.ndim >= 2:
            resized.fields["illuminant_photons"] = _scene_resize_array(illuminant_array, new_rows, new_cols)

    return resized


def _support_resample_positions(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("Resample step must be positive.")
    if stop <= start:
        return np.array([float(start)], dtype=float)
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return float(start) + float(step) * np.arange(max(count, 1), dtype=float)


def _scene_resample_plane_on_support(
    plane: np.ndarray,
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    normalized_method = param_format(method or "linear")
    if x_support.size <= 1:
        col_coords = np.zeros_like(x_query, dtype=float)
    else:
        col_coords = (np.asarray(x_query, dtype=float) - float(x_support[0])) / float(x_support[1] - x_support[0])
    if y_support.size <= 1:
        row_coords = np.zeros_like(y_query, dtype=float)
    else:
        row_coords = (np.asarray(y_query, dtype=float) - float(y_support[0])) / float(y_support[1] - y_support[0])

    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing="ij")
    if normalized_method in {"nearest", "nearestneighbor"}:
        return map_coordinates(np.asarray(plane, dtype=float), [row_grid, col_grid], order=0, mode="nearest", prefilter=False)
    if normalized_method in {"linear", "bilinear"}:
        return map_coordinates(np.asarray(plane, dtype=float), [row_grid, col_grid], order=1, mode="nearest", prefilter=False)
    if normalized_method in {"cubic", "spline"}:
        return map_coordinates(np.asarray(plane, dtype=float), [row_grid, col_grid], order=3, mode="nearest", prefilter=True)
    raise ValueError(f"Unsupported scene resample method: {method}")


def scene_combine(scene1: Scene, scene2: Scene, *args: Any) -> Scene:
    direction = "horizontal"
    if args:
        if len(args) != 2 or param_format(args[0]) != "direction":
            raise ValueError("scene_combine expects MATLAB-style 'direction', value arguments.")
        direction = str(args[1]).lower()
    normalized = param_format(direction)

    wave1 = np.asarray(scene_get(scene1, "wave"), dtype=float)
    wave2 = np.asarray(scene_get(scene2, "wave"), dtype=float)
    if not np.array_equal(wave1, wave2):
        raise ValueError("scene_combine requires matching wavelength samples.")

    if normalized == "horizontal":
        if int(scene_get(scene1, "rows")) != int(scene_get(scene2, "rows")):
            raise ValueError("Horizontal scene_combine requires matching row counts.")
        photons = np.concatenate(
            (
                np.asarray(scene_get(scene1, "photons"), dtype=float),
                np.asarray(scene_get(scene2, "photons"), dtype=float),
            ),
            axis=1,
        )
        combined = scene_set(scene1.clone(), "photons", photons)
        return scene_set(
            combined,
            "hfov",
            float(scene_get(scene1, "fov")) + float(scene_get(scene2, "fov")),
        )

    if normalized == "vertical":
        if int(scene_get(scene1, "cols")) != int(scene_get(scene2, "cols")):
            raise ValueError("Vertical scene_combine requires matching column counts.")
        photons = np.concatenate(
            (
                np.asarray(scene_get(scene1, "photons"), dtype=float),
                np.asarray(scene_get(scene2, "photons"), dtype=float),
            ),
            axis=0,
        )
        return scene_set(scene1.clone(), "photons", photons)

    if normalized == "both":
        return scene_combine(scene_combine(scene1, scene2, "direction", "horizontal"), scene_combine(scene1, scene2, "direction", "horizontal"), "direction", "vertical")

    if normalized == "centered":
        scene_mid = scene_combine(scene2, scene_combine(scene1, scene2, "direction", "horizontal"), "direction", "horizontal")
        scene_edge = scene_combine(scene_combine(scene2, scene2, "direction", "horizontal"), scene2, "direction", "horizontal")
        return scene_combine(scene_combine(scene_edge, scene_mid, "direction", "vertical"), scene_edge, "direction", "vertical")

    raise ValueError(f"Unsupported scene_combine direction: {direction}")


def _validate_scene_add_pair(scene1: Scene, scene2: Scene) -> tuple[np.ndarray, np.ndarray]:
    wave1 = np.asarray(scene_get(scene1, "wave"), dtype=float)
    wave2 = np.asarray(scene_get(scene2, "wave"), dtype=float)
    if not np.array_equal(wave1, wave2):
        raise ValueError("scene_add requires matching wavelength samples.")

    photons1 = np.asarray(scene_get(scene1, "photons"), dtype=float)
    photons2 = np.asarray(scene_get(scene2, "photons"), dtype=float)
    if photons1.shape != photons2.shape:
        raise ValueError("scene_add requires matching scene geometry.")
    return photons1, photons2


def _remove_scene_spatial_mean(photons: np.ndarray) -> np.ndarray:
    centered = np.asarray(photons, dtype=float).copy()
    centered -= np.mean(centered, axis=(0, 1), keepdims=True, dtype=float)
    return centered


def scene_add(in1: Scene | list[Scene] | tuple[Scene, ...], in2: Scene | Any, add_flag: str = "add") -> Scene:
    normalized = param_format(add_flag)

    if not isinstance(in1, (list, tuple)):
        if not isinstance(in2, Scene):
            raise ValueError("scene_add with a single Scene input requires another Scene.")
        photons1, photons2 = _validate_scene_add_pair(in1, in2)
        if normalized == "add":
            photons = photons1 + photons2
        elif normalized == "average":
            photons = (photons1 + photons2) / 2.0
        elif normalized == "removespatialmean":
            photons = photons1 + _remove_scene_spatial_mean(photons2)
        else:
            raise ValueError(f"Unsupported scene_add flag: {add_flag}")
        return scene_set(in1.clone(), "photons", photons)

    scenes = list(in1)
    if not scenes:
        raise ValueError("scene_add requires at least one input scene.")

    weights = np.asarray(in2, dtype=float).reshape(-1)
    if weights.size != len(scenes):
        raise ValueError("scene_add scene-list mode requires one weight per input scene.")

    reference = scenes[0]
    photon_stack: list[np.ndarray] = []
    for scene in scenes:
        photons_ref, photons_current = _validate_scene_add_pair(reference, scene)
        photon_stack.append(photons_current if scene is not reference else photons_ref)

    if normalized == "add":
        photons = weights[0] * photon_stack[0]
        for idx in range(1, len(photon_stack)):
            photons = photons + weights[idx] * photon_stack[idx]
    elif normalized == "average":
        photons = np.mean(np.stack(photon_stack, axis=0), axis=0, dtype=float)
    elif normalized == "removespatialmean":
        photons = weights[0] * photon_stack[0]
        for idx in range(1, len(photon_stack)):
            photons = photons + weights[idx] * _remove_scene_spatial_mean(photon_stack[idx])
    else:
        raise ValueError(f"Unsupported scene_add flag: {add_flag}")

    return scene_set(reference.clone(), "photons", photons)


def scene_add_grid(
    scene: Scene,
    p_size: Any | None = None,
    g_width: Any = 1,
) -> Scene:
    """Add black grid lines to scene photons following MATLAB sceneAddGrid()."""

    rows, cols = scene_get(scene, "size")
    if p_size is None:
        p_size_array = np.array([rows / 2.0, cols / 2.0], dtype=float)
    else:
        p_size_array = np.asarray(p_size, dtype=float).reshape(-1)
        if p_size_array.size == 1:
            p_size_array = np.repeat(p_size_array, 2)
    if p_size_array.size != 2:
        raise ValueError("sceneAddGrid pSize must be a scalar or [row, col].")

    step_rows = max(int(np.rint(p_size_array[0])), 1)
    step_cols = max(int(np.rint(p_size_array[1])), 1)
    edge_width = max(int(np.rint(np.asarray(g_width, dtype=float).reshape(-1)[0])), 1)

    photons = np.asarray(scene_get(scene, "photons"), dtype=float).copy()
    photons[:edge_width, :, :] = 0.0
    photons[-edge_width:, :, :] = 0.0
    photons[:, :edge_width, :] = 0.0
    photons[:, -edge_width:, :] = 0.0

    for start in range(step_rows - 1, rows - 1, step_rows):
        stop = min(start + edge_width, rows)
        photons[start:stop, :, :] = 0.0
    for start in range(step_cols - 1, cols - 1, step_cols):
        stop = min(start + edge_width, cols)
        photons[:, start:stop, :] = 0.0

    return scene_set(scene.clone(), "photons", photons)


def scene_adjust_pixel_size(
    scene: Scene,
    oi: Any,
    pixel_size: Any,
) -> tuple[Scene, float]:
    """Adjust scene distance so scene sample spacing matches a target pixel size."""

    del oi
    pixel_array = np.asarray(pixel_size, dtype=float).reshape(-1)
    if pixel_array.size == 0:
        raise ValueError("sceneAdjustPixelSize requires a pixel size.")
    current_distance = float(scene_get(scene, "distance"))
    sample_spacing = np.asarray(scene_get(scene, "sample spacing", "m"), dtype=float).reshape(-1)
    new_distance = current_distance * (float(pixel_array[0]) / max(float(sample_spacing[0]), 1e-12))
    adjusted = scene_set(scene.clone(), "distance", new_distance)
    return adjusted, float(new_distance)


def scene_illuminant_scale(scene: Scene) -> Scene:
    """Rescale scene illuminant so scene radiance implies a plausible reflectance."""

    illuminant_spd = np.asarray(scene_get(scene, "illuminant photons"), dtype=float).reshape(-1)
    if illuminant_spd.size == 0:
        raise ValueError("Scene requires an illuminant for sceneIlluminantScale.")

    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    known_reflectance = np.asarray(scene_get(scene, "known reflectance"), dtype=float).reshape(-1)
    current = scene.clone()

    if known_reflectance.size == 0:
        peak_radiance, peak_wave = np.asarray(scene_get(scene, "peak radiance and wave"), dtype=float).reshape(-1)
        wave_index = int(np.argmin(np.abs(wave - float(peak_wave))))
        photon_slice = np.asarray(scene_get(scene, "photons", float(peak_wave)), dtype=float)
        max_index = int(np.argmax(photon_slice))
        row_index, col_index = np.unravel_index(max_index, photon_slice.shape)
        known_reflectance = np.array([0.9, row_index + 1, col_index + 1, wave_index + 1], dtype=float)
        current = scene_set(current, "known reflectance", known_reflectance)
        scene = current

    reflectance = float(known_reflectance[0])
    row_index = int(np.rint(known_reflectance[1])) - 1
    col_index = int(np.rint(known_reflectance[2])) - 1
    wave_index = int(np.rint(known_reflectance[3])) - 1
    photon_slice = np.asarray(scene_get(scene, "photons", float(wave[wave_index])), dtype=float)
    scene_radiance = float(photon_slice[row_index, col_index])
    scale = (scene_radiance / max(reflectance, 1e-12)) / max(float(illuminant_spd[wave_index]), 1e-12)
    return scene_set(scene.clone(), "illuminant photons", scale * illuminant_spd)


def scene_spd_scale(
    scene: Scene,
    full_name: Any,
    op: str,
    skip_illuminant: Any = False,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[Scene, Any]:
    """Apply a spectral arithmetic operation to a scene cube using energy-domain SPD data."""

    if full_name is None:
        raise ValueError("sceneSPDScale requires a spectrum source in headless mode.")

    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    energy = np.asarray(scene_get(scene, "energy"), dtype=float)
    if isinstance(full_name, (str, Path)):
        spd = np.asarray(ie_read_spectra(full_name, wave, asset_store=_store(asset_store)), dtype=float).reshape(-1)
    else:
        spd = np.asarray(full_name, dtype=float).reshape(-1)
    if spd.size != wave.size:
        raise ValueError("sceneSPDScale spectrum must match the scene wavelength sampling.")

    operator = param_format(op)
    scale = spd.reshape(1, 1, -1)
    updated_energy = energy.copy()
    illuminant_energy = None

    if operator in {"/", "divide"}:
        updated_energy = np.divide(updated_energy, scale, out=np.zeros_like(updated_energy), where=scale != 0.0)
        if not bool(skip_illuminant):
            current_illuminant = np.asarray(scene_get(scene, "illuminant energy"), dtype=float)
            illuminant_energy = np.divide(current_illuminant, spd, out=np.zeros_like(current_illuminant, dtype=float), where=spd != 0.0)
    elif operator in {"*", "multiply"}:
        updated_energy = updated_energy * scale
        if not bool(skip_illuminant):
            illuminant_energy = np.asarray(scene_get(scene, "illuminant energy"), dtype=float) * spd
    elif operator in {"+", "add", "sum", "plus"}:
        updated_energy = updated_energy + scale
    elif operator in {"-", "subtract", "minus"}:
        updated_energy = updated_energy - scale
    else:
        raise ValueError(f"Unknown sceneSPDScale operation: {op}")

    updated = scene_set(scene.clone(), "energy", updated_energy)
    if illuminant_energy is not None:
        updated = scene_set(updated, "illuminant energy", illuminant_energy)
    scene_calculate_luminance(updated, asset_store=asset_store)
    return updated, full_name


def scene_adjust_reflectance(
    scene: Scene,
    new_reflectance: Any,
) -> Scene:
    """Replace scene reflectance while keeping the illuminant unchanged."""

    reflectance = np.asarray(new_reflectance, dtype=float)
    if reflectance.size == 0:
        raise ValueError("sceneAdjustReflectance requires a reflectance input.")
    if np.max(reflectance) > 1.0 or np.min(reflectance) < 0.0:
        raise ValueError("sceneAdjustReflectance reflectance values must lie in [0, 1].")

    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    illuminant = np.asarray(scene_get(scene, "illuminant photons"), dtype=float)
    illuminant_format = param_format(scene_get(scene, "illuminant format"))

    if reflectance.ndim == 1:
        if reflectance.size != wave.size:
            raise ValueError("Reflectance vector must match the scene wavelength samples.")
        if illuminant_format == "spectral":
            rows, cols = scene_get(scene, "size")
            photons = np.broadcast_to((reflectance * illuminant).reshape(1, 1, -1), (rows, cols, wave.size)).copy()
        elif illuminant_format == "spatialspectral":
            photons = np.asarray(illuminant, dtype=float) * reflectance.reshape(1, 1, -1)
        else:
            raise ValueError(f"Unknown illuminant format {scene_get(scene, 'illuminant format')}.")
    elif reflectance.ndim == 3:
        if reflectance.shape[-1] != wave.size:
            raise ValueError("Reflectance cube must match the scene wavelength samples.")
        if illuminant_format == "spectral":
            photons = reflectance * np.asarray(illuminant, dtype=float).reshape(1, 1, -1)
        elif illuminant_format == "spatialspectral":
            photons = reflectance * np.asarray(illuminant, dtype=float)
        else:
            raise ValueError(f"Unknown illuminant format {scene_get(scene, 'illuminant format')}.")
    else:
        raise ValueError("sceneAdjustReflectance expects a reflectance vector or scene-sized reflectance cube.")

    return scene_set(scene.clone(), "photons", photons)


def _normalize_scene_illuminant_array(value: Any, wave: np.ndarray) -> tuple[np.ndarray, str]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if array.size != wave.size:
            raise ValueError("Scene illuminant spectra must match the scene wavelength samples.")
        return array.reshape(-1), "spectral"
    if array.ndim == 3:
        if array.shape[-1] != wave.size and array.shape[0] == wave.size:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != wave.size:
            raise ValueError("Spatial-spectral illuminant cubes must use the scene wavelength samples.")
        return array, "spatial spectral"
    raise ValueError("Scene illuminants must be either a 1D spectrum or a 3D spatial-spectral cube.")


def _set_scene_illuminant_photons(scene: Scene, value: Any) -> Scene:
    wave = np.asarray(scene.fields["wave"], dtype=float)
    illuminant, illuminant_format = _normalize_scene_illuminant_array(value, wave)
    if illuminant.ndim == 3 and tuple(illuminant.shape[:2]) != tuple(scene_get(scene, "size")):
        raise ValueError("Spatial-spectral illuminant photons must match the scene size.")
    scene.fields["illuminant_photons"] = np.asarray(illuminant, dtype=float)
    scene.fields["illuminant_energy"] = np.asarray(quanta_to_energy(illuminant, wave), dtype=float)
    scene.fields["illuminant_format"] = illuminant_format
    _invalidate_scene_caches(scene)
    return scene


def _set_scene_illuminant_energy(scene: Scene, value: Any) -> Scene:
    wave = np.asarray(scene.fields["wave"], dtype=float)
    illuminant, illuminant_format = _normalize_scene_illuminant_array(value, wave)
    if illuminant.ndim == 3 and tuple(illuminant.shape[:2]) != tuple(scene_get(scene, "size")):
        raise ValueError("Spatial-spectral illuminant energy must match the scene size.")
    scene.fields["illuminant_energy"] = np.asarray(illuminant, dtype=float)
    scene.fields["illuminant_photons"] = np.asarray(energy_to_quanta(illuminant, wave), dtype=float)
    scene.fields["illuminant_format"] = illuminant_format
    _invalidate_scene_caches(scene)
    return scene


def _resize_scene_pattern(pattern: Any, size: tuple[int, int]) -> np.ndarray:
    array = np.asarray(pattern, dtype=float)
    if array.ndim != 2:
        raise ValueError("Illuminant spatial patterns must be 2D arrays.")
    if array.shape != tuple(size):
        return _scene_resize_array(array, int(size[0]), int(size[1]))
    return array


def scene_illuminant_ss(scene: Scene, pattern: Any | None = None) -> Scene:
    normalized = param_format(scene_get(scene, "illuminant format"))
    current = scene
    if normalized == "spectral":
        rows, cols = scene_get(scene, "size")
        illuminant = np.asarray(scene_get(scene, "illuminant photons"), dtype=float).reshape(1, 1, -1)
        current = scene_set(current, "illuminant photons", np.broadcast_to(illuminant, (rows, cols, illuminant.shape[2])).copy())
    elif normalized != "spatialspectral":
        raise ValueError(f"Unknown illuminant format: {scene_get(scene, 'illuminant format')}")

    if pattern is None or np.asarray(pattern).size == 0:
        return current
    return scene_illuminant_pattern(current, pattern)


def scene_illuminant_pattern(scene: Scene, pattern: Any) -> Scene:
    current = scene if param_format(scene_get(scene, "illuminant format")) == "spatialspectral" else scene_illuminant_ss(scene)
    rows, cols = scene_get(current, "size")
    pattern_array = _resize_scene_pattern(pattern, (rows, cols))
    photons = np.asarray(scene_get(current, "photons"), dtype=float) * pattern_array[:, :, None]
    illuminant = np.asarray(scene_get(current, "illuminant photons"), dtype=float) * pattern_array[:, :, None]
    current = scene_set(current, "photons", photons)
    current = scene_set(current, "illuminant photons", illuminant)
    return current


def _scene_rotation_degrees(value: Any) -> float:
    if isinstance(value, str):
        normalized = param_format(value)
        if normalized in {"cw", "clockwise"}:
            return -90.0
        if normalized in {"ccw", "counterclockwise"}:
            return 90.0
        raise ValueError(f"Unsupported scene rotation parameter: {value}")
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _rotate_image_loose(image: np.ndarray, angle_deg: float) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    squeeze_channel = False
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
        squeeze_channel = True
    if array.ndim != 3:
        raise ValueError("Loose scene rotation expects a 2-D image or 3-D image cube.")
    if abs(angle_deg) < np.finfo(float).eps:
        return image.copy()

    rows, cols, channels = array.shape
    cx = (cols + 1.0) / 2.0
    cy = (rows + 1.0) / 2.0
    theta = np.deg2rad(float(angle_deg))
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    corners = np.array(
        [
            [1.0, 1.0],
            [float(cols), 1.0],
            [float(cols), float(rows)],
            [1.0, float(rows)],
        ],
        dtype=float,
    )
    centered = corners - np.array([cx, cy], dtype=float)
    rotated_corners = centered @ rotation.T
    rotated_corners = rotated_corners + np.array([cx, cy], dtype=float)

    x_min = int(np.floor(np.min(rotated_corners[:, 0])))
    x_max = int(np.ceil(np.max(rotated_corners[:, 0])))
    y_min = int(np.floor(np.min(rotated_corners[:, 1])))
    y_max = int(np.ceil(np.max(rotated_corners[:, 1])))

    x_out, y_out = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=float),
        np.arange(y_min, y_max + 1, dtype=float),
    )
    x_shift = x_out - cx
    y_shift = y_out - cy

    x_in = np.cos(theta) * x_shift + np.sin(theta) * y_shift + cx
    y_in = -np.sin(theta) * x_shift + np.cos(theta) * y_shift + cy
    sample_coords = np.vstack(((y_in - 1.0).reshape(1, -1), (x_in - 1.0).reshape(1, -1)))

    rotated = np.empty((y_out.shape[0], x_out.shape[1], channels), dtype=float)
    for channel in range(channels):
        rotated[:, :, channel] = map_coordinates(
            array[:, :, channel],
            sample_coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(y_out.shape)

    if squeeze_channel:
        return rotated[:, :, 0]
    return rotated


def scene_rotate(scene: Scene, degrees: Any) -> Scene:
    angle_deg = _scene_rotation_degrees(degrees)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    if photons.ndim != 3:
        raise ValueError("scene_rotate requires a scene photons cube.")

    rotated = scene.clone()
    rotated_photons = _rotate_image_loose(photons, angle_deg)
    rotated = scene_set(rotated, "photons", rotated_photons)

    if param_format(scene_get(scene, "illuminant format")) == "spatialspectral":
        illuminant = np.asarray(scene_get(scene, "illuminant photons"), dtype=float)
        if illuminant.ndim < 2:
            raise ValueError("Spatial-spectral illuminant photons must be at least 2-D.")
        rotated.fields["illuminant_photons"] = _rotate_image_loose(illuminant, angle_deg)
        _invalidate_scene_caches(rotated)

    return rotated


def _macbeth_cube(
    reflectances: np.ndarray,
    illuminant_photons: np.ndarray,
    patch_size: int,
    *,
    black_border: bool,
) -> np.ndarray:
    rows, cols = _MACBETH_GRID
    wave_count = reflectances.shape[0]

    # MATLAB uses reshape(transpose(macbethChart), 4, 6, nWaves), which keeps
    # the patch numbering column-major: 1:4 in the first column, 5:8 in the second.
    chart = np.reshape(reflectances.T, (rows, cols, wave_count), order="F")
    chart = chart * illuminant_photons.reshape(1, 1, -1)
    cube = np.repeat(np.repeat(chart, patch_size, axis=0), patch_size, axis=1)

    if not black_border:
        return cube

    data = cube.copy()
    border_px = int(np.floor(0.2 * patch_size))
    if border_px <= 0:
        return data
    for col in range(1, cols + 1):
        start = int(np.floor(col * patch_size - border_px))
        stop = min(col * patch_size, data.shape[1] - 1)
        data[:, start : stop + 1, :] = 0.0
    for row in range(1, rows + 1):
        start = int(np.floor(row * patch_size - border_px))
        stop = min(row * patch_size, data.shape[0] - 1)
        data[start : stop + 1, :, :] = 0.0

    pad = ((border_px, 0), (border_px, 0), (0, 0))
    return np.pad(data, pad, mode="constant")


def _load_d65(wave: np.ndarray, asset_store: AssetStore) -> tuple[np.ndarray, np.ndarray]:
    wave_nm, energy = asset_store.load_illuminant("D65.mat", wave_nm=wave)
    return wave_nm, energy


def macbeth_read_reflectance(
    wave: Any | None = None,
    patch_list: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Read Macbeth chart reflectances using MATLAB macbethReadReflectance() semantics."""

    wave_nm = np.arange(400.0, 701.0, 1.0, dtype=float) if wave is None else np.asarray(wave, dtype=float).reshape(-1)
    reflectance = np.asarray(ie_read_spectra("macbethChart.mat", wave_nm, asset_store=_store(asset_store)), dtype=float)
    if patch_list is None:
        return reflectance

    patch_array = np.asarray(patch_list, dtype=int).reshape(-1)
    if patch_array.size == 0:
        return reflectance[:, :0]
    if np.min(patch_array) >= 1:
        patch_array = patch_array - 1
    if np.min(patch_array) < 0 or np.max(patch_array) >= reflectance.shape[1]:
        raise IndexError("Macbeth patch indices are out of range.")
    return reflectance[:, patch_array]


def macbeth_rectangles(corner_points: Any) -> tuple[np.ndarray, int, int]:
    """Return Macbeth patch centers plus ROI delta/patch size from 4 chart corners."""

    corners = np.asarray(corner_points, dtype=float).reshape(4, 2)
    corners = np.fliplr(corners)

    offset = corners[0, :]
    current = np.column_stack(
        (
            (corners[1, :] - offset).reshape(-1),
            (corners[2, :] - offset).reshape(-1),
            (corners[3, :] - offset).reshape(-1),
        )
    )
    ideal = np.array([[6.0, 6.0, 0.0], [0.0, 4.0, 4.0]], dtype=float)
    linear = current @ np.linalg.pinv(ideal)

    x_loc, y_loc = np.meshgrid(np.arange(0.5, 6.0, 1.0, dtype=float), np.arange(0.5, 4.0, 1.0, dtype=float))
    ideal_locs = np.vstack((x_loc.reshape(-1), y_loc.reshape(-1)))
    patch_locs = np.rint(linear @ ideal_locs + offset.reshape(2, 1)).astype(int)
    flip_order = np.array([4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17, 24, 23, 22, 21], dtype=int) - 1
    patch_locs = patch_locs[:, flip_order]

    if abs(corners[0, 1] - corners[1, 1]) > abs(corners[0, 0] - corners[1, 0]):
        delta_x = int(np.rint(abs(corners[0, 1] - corners[1, 1]) / 6.0))
        delta_y = int(np.rint(abs(corners[0, 0] - corners[3, 1]) / 4.0))
    else:
        delta_y = int(np.rint(abs(corners[0, 0] - corners[1, 0]) / 6.0))
        delta_x = int(np.rint(abs(corners[0, 1] - corners[3, 1]) / 4.0))
    delta = int(np.rint(min(delta_x, delta_y) / 3.0))
    patch_size = int((2 * delta) + 1)
    return patch_locs, delta, patch_size


def macbeth_rois(current_loc: Any, delta: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-style ROI locations and rect from a Macbeth patch center."""

    from .roi import ie_rect2_locs

    center = np.rint(np.asarray(current_loc, dtype=float).reshape(-1)).astype(int)
    if center.size != 2:
        raise ValueError("current_loc must contain [row, col].")
    rect = np.array(
        [
            int(center[1] - np.rint(float(delta) / 2.0)),
            int(center[0] - np.rint(float(delta) / 2.0)),
            int(delta),
            int(delta),
        ],
        dtype=int,
    )
    return ie_rect2_locs(rect), rect


def macbeth_patch_data(
    obj: Any,
    m_locs: Any,
    delta: int,
    full_data: bool = False,
    data_type: str | None = None,
) -> tuple[Any, np.ndarray]:
    """Return Macbeth patch data from a scene/OI/sensor/IP using headless ROI extraction."""

    from .roi import vc_get_roi_data

    locs = np.asarray(m_locs, dtype=int)
    if locs.shape == (24, 2):
        locs = locs.T
    if locs.shape != (2, 24):
        raise ValueError("m_locs must be a 2x24 matrix of [row, col] patch centers.")

    if data_type is None:
        obj_type = param_format(getattr(obj, "type", type(obj).__name__))
        if obj_type == "scene":
            data_type = "photons"
        elif obj_type == "opticalimage":
            data_type = "photons"
        elif obj_type in {"sensor", "isa"}:
            data_type = "dvorvolts"
        else:
            data_type = "result"

    if full_data:
        all_patch_data: list[np.ndarray] = []
        for patch_index in range(24):
            roi_locs, _ = macbeth_rois(locs[:, patch_index], delta)
            all_patch_data.append(np.asarray(vc_get_roi_data(obj, roi_locs, data_type), dtype=float))
        return all_patch_data, np.asarray([], dtype=float)

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for patch_index in range(24):
        roi_locs, _ = macbeth_rois(locs[:, patch_index], delta)
        patch_data = np.asarray(vc_get_roi_data(obj, roi_locs, data_type), dtype=float)
        means.append(np.nanmean(patch_data, axis=0))
        stds.append(np.nanstd(patch_data, axis=0))
    return np.vstack(means), np.vstack(stds)


def chart_patch_data(
    obj: Any,
    m_locs: Any,
    delta: Any,
    full_data: bool = False,
    data_type: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Return chart ROI values from a vcimage-like object or sensor."""

    from .roi import ie_rect2_locs, vc_get_roi_data

    if obj is None:
        raise ValueError("vcimage or sensor required")

    locs = np.asarray(m_locs, dtype=float)
    if locs.ndim != 2:
        raise ValueError("Mid locations required")
    if locs.shape[0] != 2 and locs.shape[1] == 2:
        locs = locs.T
    if locs.shape[0] != 2:
        raise ValueError("m_locs must be a 2xN matrix of [row, col] patch centers.")

    delta_value = float(np.asarray(delta, dtype=float).reshape(-1)[0])
    roi_delta = _matlab_round_scalar(delta_value)
    half_delta = _matlab_round_scalar(delta_value / 2.0)
    resolved_type = "result" if data_type is None else str(data_type)
    obj_type = param_format(getattr(obj, "type", type(obj).__name__))

    if not full_data and obj_type in {"sensor", "isa"}:
        raise ValueError("Use fullData = 1")

    patch_data: list[np.ndarray] = []
    for index in range(locs.shape[1]):
        current = np.asarray(locs[:, index], dtype=float).reshape(2)
        rect = np.array(
            [
                _matlab_round_scalar(current[1]) - half_delta,
                _matlab_round_scalar(current[0]) - half_delta,
                roi_delta,
                roi_delta,
            ],
            dtype=int,
        )
        roi_locs = ie_rect2_locs(rect)
        current_patch = np.asarray(vc_get_roi_data(obj, roi_locs, resolved_type), dtype=float)
        if current_patch.ndim == 1:
            current_patch = current_patch.reshape(-1, 1)
        patch_data.append(current_patch)

    if full_data:
        return patch_data

    if not patch_data:
        return np.zeros((0, 0), dtype=float)

    means = np.zeros((locs.shape[1], patch_data[0].shape[1]), dtype=float)
    for index, current_patch in enumerate(patch_data):
        means[index, :] = np.mean(current_patch, axis=0, dtype=float)
    return means


def ie_cook_torrance(
    surface_normal: Any,
    view_direction: Any,
    light_direction: Any,
    base_reflectance: Any,
    roughness: float,
) -> float | np.ndarray:
    """Compute the Cook-Torrance BRDF value using the upstream GGX/Schlick form."""

    def _normalize(direction: Any, name: str) -> np.ndarray:
        vector = np.asarray(direction, dtype=float).reshape(-1)
        if vector.size != 3:
            raise ValueError(f"{name} must contain exactly 3 elements.")
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 0.0:
            raise ValueError(f"{name} must be non-zero.")
        return vector / magnitude

    surface = _normalize(surface_normal, "surface_normal")
    view = _normalize(view_direction, "view_direction")
    light = _normalize(light_direction, "light_direction")

    half_vector = view + light
    half_norm = float(np.linalg.norm(half_vector))
    if half_norm <= 0.0:
        half_vector = surface.copy()
    else:
        half_vector = half_vector / half_norm

    n_dot_v = max(float(np.dot(surface, view)), 1.0e-5)
    n_dot_l = max(float(np.dot(surface, light)), 1.0e-5)
    n_dot_h = max(float(np.dot(surface, half_vector)), 1.0e-5)
    v_dot_h = max(float(np.dot(view, half_vector)), 1.0e-5)

    alpha = max(float(roughness) ** 2, 1.0e-8)
    alpha_sq = alpha**2
    denom = max((n_dot_h**2 * (alpha_sq - 1.0) + 1.0) ** 2, 1.0e-12)
    distribution = alpha_sq / (np.pi * denom)

    reflectance = np.asarray(base_reflectance, dtype=float)
    fresnel = reflectance + (1.0 - reflectance) * (1.0 - v_dot_h) ** 5

    def _g1(direction: np.ndarray) -> float:
        n_dot_w = float(np.dot(surface, direction))
        g1_denom = n_dot_w + np.sqrt(alpha_sq + (1.0 - alpha_sq) * n_dot_w**2)
        return (2.0 * n_dot_w) / max(float(g1_denom), 1.0e-12)

    geometry = _g1(view) * _g1(light)
    result = (distribution * fresnel * geometry) / (4.0 * n_dot_l * n_dot_v)
    if reflectance.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=float).reshape(reflectance.shape)


def macbeth_ideal_color(
    illuminant: Any = "D65",
    color_space: str = "XYZ",
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    """Return Macbeth target values for a given illuminant and color space."""

    from .illuminant import illuminant_create, illuminant_get
    from .metrics import xyz_to_lab
    from .utils import rgb_to_xw_format, xw_to_rgb_format, xyz_to_linear_srgb

    store = _store(asset_store)
    if isinstance(illuminant, dict):
        spectrum = illuminant.get("spectrum", {})
        wave = np.asarray(spectrum.get("wave", illuminant.get("wave", DEFAULT_WAVE)), dtype=float).reshape(-1)
    else:
        wave = DEFAULT_WAVE.copy()
    reflectance = macbeth_read_reflectance(wave, np.arange(1, 25, dtype=int), asset_store=store)

    if isinstance(illuminant, str):
        illuminant_obj = illuminant_create(illuminant, wave, 100.0, asset_store=store)
        illuminant_energy = np.asarray(illuminant_get(illuminant_obj, "energy", asset_store=store), dtype=float).reshape(-1)
    elif isinstance(illuminant, dict):
        illuminant_name = str(illuminant.get("name", "D65"))
        luminance = float(illuminant.get("luminance", 100.0))
        if "energy" in illuminant:
            illuminant_energy = np.asarray(illuminant["energy"], dtype=float).reshape(-1)
        elif "photons" in illuminant:
            illuminant_energy = quanta_to_energy(np.asarray(illuminant["photons"], dtype=float).reshape(-1), wave)
        elif param_format(illuminant_name) == "blackbody":
            temperature = float(illuminant.get("temperature", 5000.0))
            illuminant_obj = illuminant_create("blackbody", wave, temperature, luminance, asset_store=store)
            illuminant_energy = np.asarray(illuminant_get(illuminant_obj, "energy", asset_store=store), dtype=float).reshape(-1)
        elif param_format(illuminant_name) == "daylight":
            temperature = float(illuminant.get("temperature", illuminant.get("cct", 6500.0)))
            illuminant_obj = illuminant_create("daylight", wave, temperature, luminance, asset_store=store)
            illuminant_energy = np.asarray(illuminant_get(illuminant_obj, "energy", asset_store=store), dtype=float).reshape(-1)
        else:
            illuminant_obj = illuminant_create(illuminant_name, wave, luminance, asset_store=store)
            illuminant_energy = np.asarray(illuminant_get(illuminant_obj, "energy", asset_store=store), dtype=float).reshape(-1)
    else:
        illuminant_energy = np.asarray(illuminant_get(illuminant, "energy", asset_store=store), dtype=float).reshape(-1)

    color_signal = reflectance * illuminant_energy.reshape(-1, 1)
    normalized_space = param_format(color_space)
    if normalized_space == "xyz":
        target = xyz_from_energy(color_signal.T, wave, asset_store=store)
        return 100.0 * (target / max(float(np.max(target[:, 1])), 1e-12))
    if normalized_space == "lab":
        macbeth_xyz = macbeth_ideal_color(illuminant, "xyz", asset_store=store)
        return xyz_to_lab(macbeth_xyz, macbeth_xyz[3, :])
    if normalized_space == "lrgb":
        macbeth_xyz = macbeth_ideal_color(illuminant, "xyz", asset_store=store) / 100.0
        linear_rgb = xyz_to_linear_srgb(xw_to_rgb_format(macbeth_xyz, 1, 24))
        return np.asarray(rgb_to_xw_format(np.clip(linear_rgb, 0.0, 1.0))[0], dtype=float)
    if normalized_space == "srgb":
        macbeth_xyz = macbeth_ideal_color(illuminant, "xyz", asset_store=store)
        return np.asarray(rgb_to_xw_format(xyz_to_srgb(xw_to_rgb_format(macbeth_xyz, 1, 24)))[0], dtype=float)
    raise UnsupportedOptionError("macbethIdealColor", color_space)


def _macbeth_chart_parameters(obj: Any) -> dict[str, Any]:
    if not hasattr(obj, "fields"):
        raise TypeError("Macbeth helpers require an ISET object with `.fields` storage.")
    obj_type = param_format(getattr(obj, "type", type(obj).__name__))
    field_name = "chart_parameters" if obj_type == "scene" else "chartP"
    stored = obj.fields.get(field_name)
    if not isinstance(stored, dict):
        stored = {}
        obj.fields[field_name] = stored
    return stored


def _macbeth_object_type(obj: Any) -> str:
    return param_format(getattr(obj, "type", type(obj).__name__))


def _macbeth_object_data_type(obj: Any) -> str:
    obj_type = _macbeth_object_type(obj)
    if obj_type in {"scene", "opticalimage"}:
        return "photons"
    if obj_type in {"sensor", "isa"}:
        return "dvorvolts"
    if obj_type == "vcimage":
        return "result"
    raise ValueError(f"Unsupported Macbeth object type: {obj_type}")


def _macbeth_chart_data(reflectances: np.ndarray, patch_size: int, black_border: bool) -> np.ndarray:
    reflectance_array = np.asarray(reflectances, dtype=float)
    patch_count = int(reflectance_array.shape[1])
    wave_count = int(reflectance_array.shape[0])

    if patch_count == 24:
        base = np.reshape(reflectance_array.T, (_MACBETH_GRID[0], _MACBETH_GRID[1], wave_count), order="F")
    else:
        base = reflectance_array.T.reshape(1, patch_count, wave_count)

    data = np.repeat(np.repeat(base, int(patch_size), axis=0), int(patch_size), axis=1)
    if not black_border:
        return np.asarray(data, dtype=float)

    bordered = np.asarray(data, dtype=float).copy()
    border_px = int(np.floor(0.2 * float(patch_size)))
    if border_px <= 0:
        return bordered

    rows, cols, _ = bordered.shape
    patch_rows = max(int(base.shape[0]), 1)
    patch_cols = max(int(base.shape[1]), 1)
    row_size = max(int(rows // patch_rows), 1)
    col_size = max(int(cols // patch_cols), 1)

    for col in range(1, patch_cols + 1):
        start = max(int(np.floor(col * col_size - border_px)), 0)
        stop = min(int(col * col_size), bordered.shape[1])
        bordered[:, start:stop, :] = 0.0
    for row in range(1, patch_rows + 1):
        start = max(int(np.floor(row * row_size - border_px)), 0)
        stop = min(int(row * row_size), bordered.shape[0])
        bordered[start:stop, :, :] = 0.0

    return np.pad(bordered, ((border_px, 0), (border_px, 0), (0, 0)), mode="constant")


def macbeth_chart_create(
    patch_size: int | None = None,
    patch_list: Any | None = None,
    spectrum: Any | None = None,
    surface_file: str | None = None,
    black_border: bool = False,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Create the legacy Macbeth chart reflectance object used by scene wrappers."""

    patch_size_value = 16 if patch_size is None else int(np.rint(float(patch_size)))
    if patch_size_value <= 0:
        raise ValueError("patch_size must be positive.")
    patch_list_value = np.arange(1, 25, dtype=int) if patch_list is None else np.asarray(patch_list, dtype=int).reshape(-1)
    if patch_list_value.size == 0:
        patch_list_value = np.arange(1, 25, dtype=int)

    chart = Scene(name="Macbeth Chart")
    chart = init_default_spectrum(chart, "hyperspectral")
    if isinstance(spectrum, dict):
        wave = np.asarray(spectrum.get("wave", chart.fields["wave"]), dtype=float).reshape(-1)
        chart.fields["spectrum"] = {"wave": wave.copy(), **dict(spectrum)}
        chart.fields["spectrum"]["wave"] = wave.copy()
        chart.fields["wave"] = wave.copy()
    elif spectrum is not None:
        wave = np.asarray(spectrum, dtype=float).reshape(-1)
        chart.fields["spectrum"] = {"wave": wave.copy()}
        chart.fields["wave"] = wave.copy()

    wave = np.asarray(chart.fields["wave"], dtype=float).reshape(-1)
    file_name = "macbethChart.mat" if surface_file is None else str(surface_file)
    reflectance = np.asarray(ie_read_spectra(file_name, wave, asset_store=_store(asset_store)), dtype=float)
    if patch_list_value.size:
        patch_indices = patch_list_value.copy()
        if np.min(patch_indices) >= 1:
            patch_indices = patch_indices - 1
        reflectance = reflectance[:, patch_indices]
    chart.data["data"] = _macbeth_chart_data(reflectance, patch_size_value, bool(black_border))
    chart.fields["patch_size"] = patch_size_value
    chart.fields["patch_list"] = patch_list_value.copy()
    chart.fields["surface_file"] = file_name
    chart.fields["black_border"] = bool(black_border)
    return chart


def macbeth_draw_rects(obj: Any, onoff: str = "on") -> dict[str, Any]:
    """Return headless Macbeth-rectangle payloads from stored chart corner points."""

    if obj is None:
        raise ValueError("Structure required.")
    mode = param_format(onoff or "on")
    if mode == "off":
        return {"mode": "off"}
    if mode != "on":
        raise UnsupportedOptionError("macbethDrawRects", onoff)

    chart = _macbeth_chart_parameters(obj)
    corner_points = chart.get("cornerPoints")
    if corner_points is None:
        raise ValueError("No chart corner points.")
    patch_locs, delta, patch_size = macbeth_rectangles(corner_points)
    half_size = int(np.rint(patch_size / 2.0))
    rects = np.column_stack(
        [
            patch_locs[1, :] - half_size,
            patch_locs[0, :] - half_size,
            np.full(patch_locs.shape[1], patch_size, dtype=int),
            np.full(patch_locs.shape[1], patch_size, dtype=int),
        ]
    ).astype(int)
    chart["rects"] = rects.copy()
    chart["currentRect"] = rects[0].copy()
    return {
        "mode": "on",
        "corner_points": np.asarray(corner_points, dtype=float).copy(),
        "m_locs": patch_locs.copy(),
        "delta": int(delta),
        "patch_size": int(patch_size),
        "rects": rects,
    }


def macbeth_select(
    obj: Any,
    show_selection: bool = True,
    full_data: bool = False,
    corner_points: Any | None = None,
) -> tuple[Any, np.ndarray, int, np.ndarray, np.ndarray]:
    """Extract Macbeth patch data headlessly from a scene, sensor, or IP object."""

    if obj is None:
        raise ValueError("Object required.")
    chart = _macbeth_chart_parameters(obj)
    if corner_points is None:
        stored = chart.get("cornerPoints")
        if stored is None:
            raise ValueError("macbethSelect requires chart corner points in headless mode.")
        corner_array = np.asarray(stored, dtype=float).reshape(4, 2)
    else:
        corner_array = np.asarray(corner_points, dtype=float).reshape(4, 2)
        chart["cornerPoints"] = corner_array.copy()

    patch_locs, delta, patch_size = macbeth_rectangles(corner_array)
    patch_data, patch_std = macbeth_patch_data(
        obj,
        patch_locs,
        delta,
        full_data=bool(full_data),
        data_type=_macbeth_object_data_type(obj),
    )
    if show_selection:
        macbeth_draw_rects(obj, "on")
    return patch_data, patch_locs, int(patch_size), corner_array.copy(), np.asarray(patch_std, dtype=float)


def macbeth_sensor_values(
    sensor: Any,
    show_selection: bool = True,
    corner_points: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and standard deviation of sensor Macbeth patches."""

    full_rgb, _, _, resolved_corners, _ = macbeth_select(
        sensor,
        show_selection=show_selection,
        full_data=True,
        corner_points=corner_points,
    )

    patch_arrays = [np.asarray(patch, dtype=float) for patch in full_rgb]
    n_sensors = 1 if patch_arrays[0].ndim == 1 else int(patch_arrays[0].shape[1])
    sensor_img = np.zeros((24, n_sensors), dtype=float)
    sensor_sd = np.zeros((24, n_sensors), dtype=float)

    for ii, patch in enumerate(patch_arrays):
        values = patch.reshape(-1, n_sensors)
        for band in range(n_sensors):
            valid = values[:, band][~np.isnan(values[:, band])]
            sensor_img[ii, band] = float(np.mean(valid)) if valid.size else np.nan
            sensor_sd[ii, band] = float(np.std(valid)) if valid.size else np.nan

    return sensor_img, sensor_sd, resolved_corners


def macbeth_evaluation_graphs(
    L: Any | None,
    sensorRGB: Any,
    idealRGB: Any | None = None,
    sName: str = "sensor",
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    """Evaluate a linear Macbeth fit and return the legacy figure payload headlessly."""

    transform = np.eye(3, dtype=float) if L is None else np.asarray(L, dtype=float)
    sensor_rgb = np.asarray(sensorRGB, dtype=float)
    if sensor_rgb.ndim != 2 or sensor_rgb.shape[1] != 3:
        raise ValueError("sensorRGB must be an Nx3 XW-format array.")
    ideal_rgb = (
        np.asarray(macbeth_ideal_color("D65", "lRGB", asset_store=asset_store), dtype=float)
        if idealRGB is None
        else np.asarray(idealRGB, dtype=float)
    )
    if ideal_rgb.shape != sensor_rgb.shape:
        raise ValueError("idealRGB must match sensorRGB shape.")

    rgb_l = np.asarray(sensor_rgb @ transform, dtype=float)
    rgb_l_srgb = xw_to_rgb_format(lrgb_to_srgb(np.clip(rgb_l, 0.0, 1.0)), 4, 6)
    rgb_l_xyz = np.asarray(rgb_to_xw_format(srgb_to_xyz(rgb_l_srgb))[0], dtype=float)

    ideal_srgb = xw_to_rgb_format(lrgb_to_srgb(np.clip(ideal_rgb, 0.0, 1.0)), 4, 6)
    ideal_xyz = np.asarray(rgb_to_xw_format(srgb_to_xyz(ideal_srgb))[0], dtype=float)
    white_xyz = np.asarray(ideal_xyz[3, :], dtype=float).reshape(-1)
    d_e = np.asarray(delta_e_ab(rgb_l_xyz, ideal_xyz, white_xyz), dtype=float).reshape(-1)

    user_data = {
        "idealXYZ": ideal_xyz.copy(),
        "rgbLXYZ": rgb_l_xyz.copy(),
        "idealRGB": ideal_rgb.copy(),
        "rgbL": rgb_l.copy(),
        "dE": d_e.copy(),
    }
    return {
        "figure_name": str(sName),
        "sensor_name": str(sName),
        "rgbL": rgb_l,
        "idealRGB": ideal_rgb.copy(),
        "idealXYZ": ideal_xyz,
        "rgbLXYZ": rgb_l_xyz,
        "whiteXYZ": white_xyz,
        "deltaEab": d_e,
        "meanDeltaEab": float(np.mean(d_e)),
        "userData": user_data,
    }


def macbeth_luminance_noise(
    ip: Any,
    cp: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[Any, np.ndarray, list[np.ndarray]]:
    """Analyze luminance noise of the Macbeth gray series without opening a figure."""

    from .ip import image_rgb_to_xyz, ip_set

    if ip is None:
        raise ValueError("ip required.")
    chart = _macbeth_chart_parameters(ip)
    if cp is None:
        stored = chart.get("cornerPoints")
        if stored is None:
            raise ValueError("macbethLuminanceNoise requires chart corner points in headless mode.")
        corner_points = np.asarray(stored, dtype=float).reshape(4, 2)
    else:
        corner_points = np.asarray(cp, dtype=float).reshape(4, 2)
    ip = ip_set(ip, "chart corner points", corner_points)

    _, m_locs, patch_size, _, _ = macbeth_select(ip, show_selection=False, full_data=False, corner_points=corner_points)
    full_delta = max(int(np.rint(0.6 * float(patch_size))), 1)
    m_rgb, _ = macbeth_patch_data(ip, m_locs, full_delta, full_data=True, data_type="result")

    g_series = np.arange(3, 24, 4, dtype=int)
    y_noise = np.zeros(g_series.size, dtype=float)
    for jj, patch_index in enumerate(g_series):
        rgb = np.asarray(m_rgb[int(patch_index)], dtype=float)
        macbeth_xyz = np.asarray(image_rgb_to_xyz(ip, rgb, asset_store=asset_store), dtype=float)
        y = np.asarray(macbeth_xyz, dtype=float)[:, 1]
        y_noise[jj] = 100.0 * (float(np.std(y)) / max(float(np.mean(y)), 1.0e-12))

    return ip, y_noise, [np.asarray(patch, dtype=float) for patch in m_rgb]


def macbeth_gretag_sg_create(
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Create the larger Gretag SG Macbeth scene under an equal-energy illuminant."""

    store = _store(asset_store)
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    reflectance = np.asarray(ie_read_spectra("gretagDigitalColorSG.mat", wave, asset_store=store), dtype=float)
    rows = 10
    cols = 14
    n_wave = int(wave.size)

    img = np.reshape(reflectance.T, (rows, cols, n_wave), order="F")
    patch_size = 30
    img = np.repeat(np.repeat(img, patch_size, axis=0), patch_size, axis=1)
    n_black = 5

    for ii in range(patch_size - n_black, rows * patch_size, patch_size):
        img[ii : min(ii + n_black + 1, img.shape[0]), :, :] = 0.0
    for jj in range(patch_size - n_black, cols * patch_size, patch_size):
        img[:, jj : min(jj + n_black + 1, img.shape[1]), :] = 0.0

    padded = np.zeros((img.shape[0] + 5, img.shape[1] + 5, n_wave), dtype=float)
    padded[5:, 5:, :] = img

    scene = Scene(name="GretagSC")
    scene.fields["wave"] = wave.copy()
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = np.ones(n_wave, dtype=float)
    scene.fields["illuminant_photons"] = energy_to_quanta(scene.fields["illuminant_energy"], wave)
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.fields["known_reflectance"] = np.array([float(padded[9, 9, 9]), 10.0, 10.0, 10.0], dtype=float)
    scene.data["photons"] = padded * scene.fields["illuminant_photons"].reshape(1, 1, -1)
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=store)


def _scale_energy_to_luminance(
    energy: np.ndarray,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
    luminance_cd_m2: float = 100.0,
) -> np.ndarray:
    _, xyz_energy = asset_store.load_xyz(wave_nm=wave, energy=True)
    y_bar = np.asarray(xyz_energy, dtype=float)[:, 1]
    delta = np.mean(np.diff(wave)) if wave.size > 1 else 1.0
    current = 683.0 * float(np.sum(np.asarray(energy, dtype=float) * y_bar * delta))
    if current <= 0.0:
        return np.asarray(energy, dtype=float)
    return np.asarray(energy, dtype=float) * (float(luminance_cd_m2) / current)


def _scene_size_2d(size: Any, *, default: int) -> tuple[int, int]:
    if size is None:
        return (int(default), int(default))
    if np.isscalar(size):
        side = int(size)
        return (side, side)
    values = np.asarray(size, dtype=int).reshape(-1)
    if values.size == 0:
        return (int(default), int(default))
    if values.size == 1:
        side = int(values[0])
        return (side, side)
    return (int(values[0]), int(values[1]))


def _parse_image_size(size: Any) -> tuple[int, int]:
    return _scene_size_2d(size, default=32)


def _matlab_round_scalar(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _matlab_round(values: Any) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    return np.sign(numeric) * np.floor(np.abs(numeric) + 0.5)


def _spectral_illuminant(
    spectral_type: str,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
    blackbody_temperature_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spectral_key = param_format(spectral_type)
    if spectral_key == "d65":
        _, illuminant_energy = _load_d65(wave, asset_store)
        return illuminant_energy, energy_to_quanta(illuminant_energy, wave)
    if spectral_key in {"ee", "equalenergy"}:
        illuminant_energy = np.ones(wave.size, dtype=float)
        return illuminant_energy, energy_to_quanta(illuminant_energy, wave)
    if spectral_key in {"ep", "equalphoton", "equalphotons"}:
        illuminant_photons = np.ones(wave.size, dtype=float)
        return quanta_to_energy(illuminant_photons, wave), illuminant_photons
    if spectral_key in {"bb", "blackbody"}:
        temperature_k = 5000.0 if blackbody_temperature_k is None else float(blackbody_temperature_k)
        illuminant_energy = np.asarray(blackbody(wave, temperature_k, kind="energy"), dtype=float).reshape(-1)
        return illuminant_energy, energy_to_quanta(illuminant_energy, wave)
    raise UnsupportedOptionError("sceneCreate", f"spectralType={spectral_type}")


def _create_macbeth_scene(
    patch_size: int | None,
    wave: np.ndarray,
    surface_file: str,
    black_border: bool,
    illuminant_type: str = "d65",
    *,
    asset_store: AssetStore,
) -> Scene:
    patch_size_value = 16 if patch_size is None else int(np.rint(float(patch_size)))
    _, reflectances = asset_store.load_reflectances(surface_file, wave_nm=wave)
    illuminant_key = param_format(illuminant_type)
    if illuminant_key == "d65":
        _, illuminant_energy = _load_d65(wave, asset_store)
        scene_name = "Macbeth D65"
        illuminant_comment = "D65.mat"
    else:
        from .illuminant import illuminant_create, illuminant_get

        if illuminant_key == "d50":
            illuminant_name = "d50"
            scene_name = "Macbeth D50"
            illuminant_comment = "D50.mat"
        elif illuminant_key in {"c", "illc", "illuminantc"}:
            illuminant_name = "illuminantc"
            scene_name = "Macbeth Ill C"
            illuminant_comment = "illuminantC.mat"
        elif illuminant_key in {"fluorescent", "fluor"}:
            illuminant_name = "fluorescent"
            scene_name = "Macbeth Fluorescent"
            illuminant_comment = "Fluorescent.mat"
        elif illuminant_key == "tungsten":
            illuminant_name = "tungsten"
            scene_name = "Macbeth Tungsten"
            illuminant_comment = "tungsten"
        elif illuminant_key in {"ir", "eeir", "equalenergyinfrared"}:
            illuminant_name = "equalEnergy"
            scene_name = "Macbeth IR"
            illuminant_comment = "equalEnergy"
        else:
            illuminant_name = illuminant_type
            scene_name = f"Macbeth {illuminant_type}"
            illuminant_comment = str(illuminant_type)
        illuminant = illuminant_create(illuminant_name, wave, 100.0, asset_store=asset_store)
        illuminant_energy = np.asarray(illuminant_get(illuminant, "energy"), dtype=float).reshape(-1)
    illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    scene = Scene(name=scene_name)
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = illuminant_comment
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = _macbeth_cube(
        reflectances,
        illuminant_photons,
        patch_size_value,
        black_border=black_border,
    )
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _scene_shell(
    name: str,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
    illuminant_type: str = "d65",
) -> Scene:
    illuminant_key = param_format(illuminant_type)
    if illuminant_key == "d65":
        illuminant_energy, _ = _load_d65(wave, asset_store)
        illuminant_comment = "D65.mat"
    else:
        from .illuminant import illuminant_create, illuminant_get

        illuminant = illuminant_create(illuminant_type, wave, 100.0, asset_store=asset_store)
        illuminant_energy = np.asarray(illuminant_get(illuminant, "energy"), dtype=float).reshape(-1)
        illuminant_comment = str(illuminant_get(illuminant, "name"))

    scene = Scene(name=name)
    scene.fields["wave"] = np.asarray(wave, dtype=float).reshape(-1)
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = np.asarray(illuminant_energy, dtype=float).reshape(-1)
    scene.fields["illuminant_photons"] = energy_to_quanta(scene.fields["illuminant_energy"], scene.fields["wave"])
    scene.fields["illuminant_comment"] = illuminant_comment
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = np.zeros((1, 1, scene.fields["wave"].size), dtype=float)
    _update_scene_geometry(scene)
    return scene


def _seeded_scene_shell(
    name: str,
    spectral_type: str,
    seed: Scene | None,
    *,
    asset_store: AssetStore,
    illuminant_type: str = "d65",
) -> Scene:
    if seed is None:
        if param_format(spectral_type) == "monochrome":
            wave = np.array([550.0], dtype=float)
        else:
            wave = _wave_or_default(None)
        return _scene_shell(name, wave, asset_store=asset_store, illuminant_type=illuminant_type)

    current = init_default_spectrum(seed, spectral_type)
    current.name = str(name)
    current.type = "scene"
    wave = np.asarray(current.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)

    illuminant_key = param_format(illuminant_type)
    if illuminant_key == "d65":
        illuminant_energy, _ = _load_d65(wave, asset_store)
        illuminant_comment = "D65.mat"
    else:
        from .illuminant import illuminant_create, illuminant_get

        illuminant = illuminant_create(illuminant_type, wave, 100.0, asset_store=asset_store)
        illuminant_energy = np.asarray(illuminant_get(illuminant, "energy"), dtype=float).reshape(-1)
        illuminant_comment = str(illuminant_get(illuminant, "name"))

    current.fields["illuminant_format"] = "spectral"
    current.fields["illuminant_energy"] = illuminant_energy
    current.fields["illuminant_photons"] = energy_to_quanta(illuminant_energy, wave)
    current.fields["illuminant_comment"] = illuminant_comment
    current.fields["distance_m"] = float(current.fields.get("distance_m", DEFAULT_DISTANCE_M))
    current.fields["fov_deg"] = float(current.fields.get("fov_deg", DEFAULT_FOV_DEG))

    photons = current.data.get("photons")
    if photons is None:
        current.data["photons"] = np.zeros((1, 1, wave.size), dtype=float)
    else:
        photon_array = np.asarray(photons, dtype=float)
        if photon_array.ndim != 3 or photon_array.shape[2] != wave.size:
            current.data["photons"] = np.zeros((1, 1, wave.size), dtype=float)
        else:
            current.data["photons"] = photon_array.copy()

    _update_scene_geometry(current)
    return current


def _uniform_scene(
    name: str,
    size: Any,
    wave: np.ndarray,
    illuminant_energy: np.ndarray,
    illuminant_comment: str | None = None,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _parse_image_size(size)
    scene = Scene(name=name)
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = energy_to_quanta(illuminant_energy, wave)
    if illuminant_comment is not None:
        scene.fields["illuminant_comment"] = str(illuminant_comment)
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    photons = np.broadcast_to(scene.fields["illuminant_photons"], (rows, cols, wave.size)).copy()
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _checkerboard_scene(
    pixels_per_check: int,
    number_of_checks: int,
    wave: np.ndarray,
    spectral_type: str,
    *,
    asset_store: AssetStore,
) -> Scene:
    size = 2 * pixels_per_check * number_of_checks
    yy, xx = np.indices((size, size))
    tiles = 1 - ((yy // pixels_per_check + xx // pixels_per_check) % 2)
    pattern = np.clip(tiles.astype(float), 1e-6, 1.0)

    spectral_key = param_format(spectral_type)
    if spectral_key == "d65":
        _, illuminant_energy = _load_d65(wave, asset_store)
        illuminant_energy = _scale_energy_to_luminance(illuminant_energy, wave, asset_store=asset_store)
        illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    elif spectral_key in {"ee", "equalenergy"}:
        illuminant_energy = _scale_energy_to_luminance(
            np.ones(wave.size, dtype=float),
            wave,
            asset_store=asset_store,
        )
        illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    elif spectral_key in {"ep", "equalphoton", "equalphotons"}:
        illuminant_photons = np.ones(wave.size, dtype=float)
        illuminant_energy = _scale_energy_to_luminance(
            quanta_to_energy(illuminant_photons, wave),
            wave,
            asset_store=asset_store,
        )
        illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    else:
        raise UnsupportedOptionError("sceneCreate", f"checkerboard spectralType={spectral_type}")

    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="Checkerboard")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _slanted_bar_scene(
    image_size: int,
    edge_slope: float,
    wave: np.ndarray,
    fov_deg: float,
    dark_level: float,
    *,
    asset_store: AssetStore,
) -> Scene:
    half_size = int(np.rint(float(image_size) / 2.0))
    sample = np.arange(-half_size, half_size + 1, dtype=float)
    xx, yy = np.meshgrid(sample, sample, indexing="xy")
    bar = np.where(yy > float(edge_slope) * xx, 1.0, max(float(dark_level), 1.0e-6))
    illuminant_energy, illuminant_photons = _spectral_illuminant("equal energy", wave, asset_store=asset_store)
    photons = bar[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="Slanted Bar")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = float(fov_deg)
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _uniform_blackbody_scene(
    size: Any,
    temperature_k: float,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    illuminant_energy = np.asarray(blackbody(wave, float(temperature_k), kind="energy"), dtype=float).reshape(-1)
    return _uniform_scene(
        f"Uniform BB {int(round(float(temperature_k)))}K",
        _scene_size_2d(size, default=32)[0],
        wave,
        illuminant_energy,
        illuminant_comment=f"blackbody-{int(round(float(temperature_k)))}K",
        asset_store=asset_store,
    )


def _uniform_monochromatic_scene(
    size: Any,
    wavelength: Any,
    *,
    asset_store: AssetStore,
) -> Scene:
    wave = _wave_or_default(wavelength)
    scene = _uniform_scene(
        "Narrow Band",
        _scene_size_2d(size, default=128)[0],
        wave,
        np.ones(wave.size, dtype=float),
        illuminant_comment="equal-energy",
        asset_store=asset_store,
    )
    scene.name = "Narrow Band"
    return scene


def _line_scene(
    spectral_type: str,
    size: Any,
    offset: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=64)
    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    line_pos = _matlab_round_scalar(cols / 2.0) - 1 + int(offset)
    line_pos = min(max(line_pos, 0), cols - 1)
    photons = np.full((rows, cols, wave.size), 1e-4, dtype=float)
    photons[:, line_pos, :] = 1.0
    photons = photons * illuminant_photons.reshape(1, 1, -1)

    scene = Scene(name=f"line-{param_format(spectral_type)}")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _bar_scene(
    size: Any,
    width: int,
    wave: np.ndarray,
    spectral_type: str = "ep",
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=64)
    bar_width = max(int(width), 1)
    start = _matlab_round_scalar((cols - bar_width) / 2.0)
    stop = min(start + bar_width, cols)
    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = np.full((rows, cols, wave.size), 1e-8, dtype=float)
    photons[:, start:stop, :] = 1.0
    photons = photons * illuminant_photons.reshape(1, 1, -1)

    scene = Scene(name=f"bar-{bar_width}")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _point_array_scene(
    size: Any,
    spacing: int,
    spectral_type: str,
    point_size: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=128)
    spacing_px = max(int(spacing), 1)
    pattern = np.zeros((rows, cols), dtype=float)
    x_positions = np.arange(max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0), cols, spacing_px, dtype=int)
    y_positions = np.arange(max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0), rows, spacing_px, dtype=int)
    pattern[np.ix_(y_positions, x_positions)] = 1.0
    if int(point_size) > 1:
        kernel = np.ones((int(point_size), int(point_size)), dtype=float)
        pattern = convolve2d(pattern, kernel, mode="same", boundary="fill")

    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="pointarray")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 40.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _disk_array_image(size: Any, radius: int, array_size: Any) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    array_values = np.asarray(array_size if array_size is not None else [1, 1], dtype=int).reshape(-1)
    if array_values.size == 0:
        array_rows, array_cols = 1, 1
    elif array_values.size == 1:
        array_rows = array_cols = max(int(array_values[0]), 1)
    else:
        array_rows = max(int(array_values[0]), 1)
        array_cols = max(int(array_values[1]), 1)
    disk_radius = max(int(radius), 1)

    pattern = np.zeros((rows, cols), dtype=float)
    yy, xx = np.meshgrid(
        np.arange(-disk_radius, disk_radius + 1, dtype=float),
        np.arange(-disk_radius, disk_radius + 1, dtype=float),
        indexing="ij",
    )
    disk = (np.sqrt(xx**2 + yy**2) < float(disk_radius)).astype(float)

    delta_row = max(_matlab_round_scalar(rows / float(array_rows + 1)), 1)
    delta_col = max(_matlab_round_scalar(cols / float(array_cols + 1)), 1)
    row_centers = np.arange(delta_row - 1, rows - (delta_row - 1), delta_row, dtype=int)
    col_centers = np.arange(delta_col - 1, cols - (delta_col - 1), delta_col, dtype=int)

    for row_center in row_centers:
        for col_center in col_centers:
            row_start = int(row_center - _matlab_round_scalar(disk_radius))
            col_start = int(col_center - _matlab_round_scalar(disk_radius))
            row_end = row_start + disk.shape[0]
            col_end = col_start + disk.shape[1]
            if row_start < 0 or col_start < 0 or row_end > rows or col_end > cols:
                continue
            pattern[row_start:row_end, col_start:col_end] = disk

    return pattern


def _disk_array_scene(
    size: Any,
    radius: int,
    array_size: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    pattern = _disk_array_image(size, radius, array_size)
    illuminant_energy, illuminant_photons = _spectral_illuminant("ep", wave, asset_store=asset_store)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="Disk Array")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 40.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _square_array_image(size: Any, square_size: int, array_size: Any) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    array_values = np.asarray(array_size if array_size is not None else [1, 1], dtype=int).reshape(-1)
    if array_values.size == 0:
        array_rows, array_cols = 1, 1
    elif array_values.size == 1:
        array_rows = array_cols = max(int(array_values[0]), 1)
    else:
        array_rows = max(int(array_values[0]), 1)
        array_cols = max(int(array_values[1]), 1)
    square_width = max(int(square_size), 1)

    pattern = np.zeros((rows, cols), dtype=float)
    square = np.ones((square_width, square_width), dtype=float)

    delta_row = max(_matlab_round_scalar(rows / float(array_rows + 1)), 1)
    delta_col = max(_matlab_round_scalar(cols / float(array_cols + 1)), 1)
    row_centers = np.arange(delta_row - 1, rows - (delta_row - 1), delta_row, dtype=int)
    col_centers = np.arange(delta_col - 1, cols - (delta_col - 1), delta_col, dtype=int)
    half_width = _matlab_round_scalar(square_width / 2.0)

    for row_center in row_centers:
        for col_center in col_centers:
            row_start = int(row_center - half_width)
            col_start = int(col_center - half_width)
            row_end = row_start + square.shape[0]
            col_end = col_start + square.shape[1]
            if row_start < 0 or col_start < 0 or row_end > rows or col_end > cols:
                continue
            pattern[row_start:row_end, col_start:col_end] = square

    return pattern


def _square_array_scene(
    size: Any,
    square_size: int,
    array_size: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    pattern = _square_array_image(size, square_size, array_size)
    illuminant_energy, illuminant_photons = _spectral_illuminant("ep", wave, asset_store=asset_store)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="Square Array")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 40.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _grid_lines_scene(
    size: Any,
    spacing: int,
    spectral_type: str,
    thickness: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=128)
    spacing_px = max(int(spacing), 1)
    line_thickness = max(int(thickness), 1)
    pattern = np.zeros((rows, cols), dtype=float)
    row_start = max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0)
    col_start = max(_matlab_round_scalar(spacing_px / 2.0) - 1, 0)
    for delta in range(line_thickness):
        row_positions = np.arange(row_start + delta, rows, spacing_px, dtype=int)
        col_positions = np.arange(col_start + delta, cols, spacing_px, dtype=int)
        pattern[row_positions, :] = 1.0
        pattern[:, col_positions] = 1.0
    pattern[pattern == 0.0] = 1e-5

    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="gridlines")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 40.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _star_pattern_scene(
    image_size: int,
    spectral_type: str,
    n_lines: int,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    im_size = max(int(image_size), 1)
    n_rays = max(int(n_lines), 1)
    radians = np.pi * np.arange(n_rays, dtype=float) / float(n_rays)
    end_points = np.column_stack((np.cos(radians), np.sin(radians))) * (im_size / 2.0)
    end_points = _matlab_round(end_points).astype(int)

    image = np.zeros((im_size, im_size), dtype=float)
    center = im_size / 2.0
    for x, y in end_points:
        u = -int(x)
        v = -int(y)
        if x > 0:
            x, y, u, v = u, v, int(x), int(y)
        if u != x:
            slope = (float(y) - float(v)) / (float(u) - float(x))
            jj_values = np.arange(float(x), float(u) + 0.2001, 0.2, dtype=float)
            kk_values = _matlab_round(jj_values * slope).astype(int)
            row_idx = _matlab_round(kk_values + center).astype(int)
            col_idx = _matlab_round(jj_values + center).astype(int)
            valid = (
                (row_idx >= 0)
                & (row_idx < im_size)
                & (col_idx >= 0)
                & (col_idx < im_size)
            )
            image[row_idx[valid], col_idx[valid]] = 1.0
        else:
            image[:, int(im_size / 2)] = 1.0

    image[image == 0.0] = 1e-4
    image = image / max(float(np.max(image)), 1e-12)

    illuminant_energy, illuminant_photons = _spectral_illuminant(spectral_type, wave, asset_store=asset_store)
    photons = image[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name=f"radialLine-{param_format(spectral_type)}")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _zone_plate_image(size: Any, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=256)
    origin = (np.array([rows, cols], dtype=float) + 1.0) / 2.0
    x_ramp, y_ramp = np.meshgrid(
        np.arange(1.0, float(cols) + 1.0, dtype=float) - origin[1],
        np.arange(1.0, float(rows) + 1.0, dtype=float) - origin[0],
    )
    radial = x_ramp * x_ramp + y_ramp * y_ramp
    max_size = float(max(rows, cols))
    return float(amplitude) * np.cos((np.pi / max_size) * radial + float(phase)) + 1.0


def _zone_plate_scene(
    size: Any,
    wave: np.ndarray,
    fov_deg: float = 4.0,
    *,
    asset_store: AssetStore,
) -> Scene:
    image = np.clip(_zone_plate_image(size), 1.0e-4, 1.0)
    illuminant_energy, illuminant_photons = _spectral_illuminant("ep", wave, asset_store=asset_store)
    scene = Scene(name="zonePlate")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = "equal photons"
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = float(fov_deg)
    scene.data["photons"] = image[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _dead_leaves_sample_matrix(n_rows: int, n_cols: int, seed: int = 1) -> np.ndarray:
    modulus = 2147483647
    multiplier = 16807
    state = int(seed) % modulus
    if state <= 0:
        state = 1
    samples = np.empty((int(n_rows), int(n_cols)), dtype=float)
    for row in range(int(n_rows)):
        for col in range(int(n_cols)):
            state = (multiplier * state) % modulus
            samples[row, col] = float(state) / float(modulus)
    return samples


def _dead_leaves_options(value: Any | None) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    options: dict[str, Any] = {
        "rmin": float(normalized.get("rmin", 0.01)),
        "rmax": float(normalized.get("rmax", 1.0)),
        "nbr_iter": max(int(np.rint(normalized.get("nbr_iter", 5000))), 1),
        "shape": str(normalized.get("shape", "disk")),
    }
    if "seed" in normalized:
        options["seed"] = int(np.rint(normalized["seed"]))
    elif "rseed" in normalized:
        options["seed"] = int(np.rint(normalized["rseed"]))
    if "random_samples" in normalized and normalized["random_samples"] is not None:
        options["random_samples"] = np.asarray(normalized["random_samples"], dtype=float)
    return options


def _dead_leaves_image(size: Any, sigma: float, options: dict[str, Any] | None = None) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=256)
    config = _dead_leaves_options(options)
    sigma = float(sigma)
    shape = param_format(config["shape"])
    if shape not in {"disk", "square"}:
        raise UnsupportedOptionError("sceneCreate", f"dead leaves shape={config['shape']}")

    image = np.full((rows, cols), np.inf, dtype=float)
    x_support = np.linspace(0.0, 1.0, cols, dtype=float)
    y_support = np.linspace(0.0, 1.0, rows, dtype=float)
    y_grid, x_grid = np.meshgrid(y_support, x_support, indexing="ij")

    radius_list = np.linspace(config["rmin"], config["rmax"], 200, dtype=float)
    radius_dist = 1.0 / np.maximum(radius_list, 1.0e-12) ** sigma
    if sigma > 0.0:
        radius_dist = radius_dist - 1.0 / (float(config["rmax"]) ** sigma)
    radius_dist = _scale_range(np.cumsum(radius_dist, dtype=float), 0.0, 1.0)

    sample_matrix = config.get("random_samples")
    if sample_matrix is None:
        seed = config.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
        sample_matrix = rng.random((int(config["nbr_iter"]), 4), dtype=float)
    else:
        sample_matrix = np.asarray(sample_matrix, dtype=float)
        if sample_matrix.ndim != 2 or sample_matrix.shape[1] < 4:
            raise ValueError("dead leaves random_samples must be an (n, 4) array.")
        if sample_matrix.shape[0] < int(config["nbr_iter"]):
            raise ValueError("dead leaves random_samples does not contain enough rows.")

    remaining = rows * cols
    for index in range(int(config["nbr_iter"])):
        radius_u, x_center, y_center, albedo = sample_matrix[index, :4]
        radius = radius_list[int(np.argmin(np.abs(float(radius_u) - radius_dist)))]
        if shape == "disk":
            mask = np.isinf(image) & (((x_grid - float(x_center)) ** 2 + (y_grid - float(y_center)) ** 2) < float(radius) ** 2)
        else:
            mask = np.isinf(image) & (np.abs(x_grid - float(x_center)) < float(radius)) & (np.abs(y_grid - float(y_center)) < float(radius))
        covered = int(np.count_nonzero(mask))
        if covered <= 0:
            continue
        remaining -= covered
        image[mask] = float(albedo)
        if remaining <= 0:
            break

    image[np.isinf(image)] = 0.0
    return image


def _dead_leaves_scene(
    size: Any,
    sigma: float,
    options: dict[str, Any] | None,
    wave: np.ndarray | None,
    *,
    asset_store: AssetStore,
) -> Scene:
    display = _scene_display("OLED-Sony.mat", wave, asset_store=asset_store)
    image = _dead_leaves_image(size, sigma, options)
    scene = scene_from_file(image, "rgb", 100.0, display, asset_store=asset_store)
    scene = scene_set(scene, "fov", 10.0)
    scene = scene_set(scene, "name", "Dead leaves")
    return scene


def _mackay_image(radial_frequency: float, image_size: Any) -> np.ndarray:
    radial_freq = float(radial_frequency)
    im_size = max(int(np.asarray(image_size, dtype=float).reshape(-1)[0]), 1)

    mx = _matlab_round_scalar(im_size / 2.0)
    mn = -(mx - 1)
    coords = np.arange(mn, mx + 1, dtype=float)
    if coords.size != im_size:
        coords = np.linspace(-(im_size - 1.0) / 2.0, (im_size - 1.0) / 2.0, im_size, dtype=float)

    x, y = np.meshgrid(coords, coords)
    x[x == 0.0] = np.finfo(float).eps
    image = np.cos(np.arctan(y / x) * 2.0 * radial_freq)
    image = _scale_range(image, 1.0, 256.0)

    radius = _matlab_round_scalar(2.0 * radial_freq / np.pi)
    mask_x, mask_y = np.meshgrid(
        np.arange(1, im_size + 1, dtype=float),
        np.arange(1, im_size + 1, dtype=float),
    )
    mask_x = mask_x - np.mean(mask_x)
    mask_y = mask_y - np.mean(mask_y)
    image[np.sqrt(mask_x * mask_x + mask_y * mask_y) < radius] = 128.0
    return image / 256.0


def _mackay_scene(
    radial_frequency: float,
    image_size: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    image = _mackay_image(radial_frequency, image_size)
    scene = _equal_photon_pattern_scene("mackay", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=asset_store)
    scene.name = "mackay"
    return scene


def _white_noise_scene(
    size: Any,
    contrast: float,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    rows, cols = _scene_size_2d(size, default=128)
    sigma = float(contrast) / 100.0 if float(contrast) > 1.0 else float(contrast)
    rng = np.random.default_rng(0)
    pattern = np.maximum(0.0, rng.normal(loc=1.0, scale=sigma, size=(rows, cols)))
    illuminant_energy = _scale_energy_to_luminance(_load_d65(wave, asset_store)[1], wave, asset_store=asset_store)
    illuminant_photons = energy_to_quanta(illuminant_energy, wave)
    photons = pattern[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    scene = Scene(name="white noise")
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = 1.0
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return scene


def _lstar_steps_scene(
    bar_size: Any,
    n_bars: int,
    delta_e: float,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> Scene:
    if np.isscalar(bar_size):
        bar_height = 128
        bar_width = max(int(np.rint(bar_size)), 1)
    else:
        size_values = np.asarray(bar_size, dtype=float).reshape(-1)
        if size_values.size != 2:
            raise ValueError("L* scenes require a scalar bar width or a [height, width] bar size.")
        bar_height = max(int(np.rint(size_values[0])), 1)
        bar_width = max(int(np.rint(size_values[1])), 1)

    n_bars = max(int(n_bars), 1)
    l_values = np.arange(n_bars, dtype=float) * float(delta_e)
    l_values = l_values + 50.0 - ((n_bars - 1) * float(delta_e) / 2.0)
    f_y = (l_values + 16.0) / 116.0
    y_values = np.where(l_values > 8.0, f_y**3, l_values / 903.3)
    y_values = y_values / max(float(np.max(y_values)), 1.0e-12)

    image = np.repeat(y_values.reshape(1, -1), bar_height, axis=0)
    image = np.repeat(image, bar_width, axis=1)

    scene = _equal_photon_pattern_scene("L-star", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=asset_store)
    scene.name = f"L-star ({int(np.rint(delta_e))})"
    return scene


_HDR_LIGHTS_COLOR_MAP = {
    "white": np.array([1.0, 1.0, 1.0], dtype=float),
    "green": np.array([0.0, 1.0, 0.0], dtype=float),
    "blue": np.array([0.0, 0.0, 1.0], dtype=float),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=float),
    "magenta": np.array([1.0, 0.0, 1.0], dtype=float),
    "red": np.array([1.0, 0.0, 0.0], dtype=float),
    "cyan": np.array([0.0, 1.0, 1.0], dtype=float),
    "black": np.array([0.0, 0.0, 0.0], dtype=float),
}


def _hdr_light_color(name: Any) -> np.ndarray:
    normalized = param_format(name)
    return _HDR_LIGHTS_COLOR_MAP.get(normalized, _HDR_LIGHTS_COLOR_MAP["white"]).copy()


def _draw_filled_circle(image: np.ndarray, center_x: float, center_y: float, radius_px: float, color: np.ndarray) -> None:
    rows, cols = image.shape[:2]
    yy, xx = np.ogrid[:rows, :cols]
    x_center = float(center_x) - 1.0
    y_center = float(center_y) - 1.0
    mask = (xx - x_center) ** 2 + (yy - y_center) ** 2 <= float(radius_px) ** 2
    image[mask] = color.reshape(1, 3)


def _draw_filled_rectangle(
    image: np.ndarray,
    left: float,
    top: float,
    width: float,
    height: float,
    color: np.ndarray,
) -> None:
    rows, cols = image.shape[:2]
    x0 = int(np.clip(np.rint(left - 1.0), 0, cols))
    y0 = int(np.clip(np.rint(top - 1.0), 0, rows))
    x1 = int(np.clip(np.rint(left - 1.0 + width), 0, cols))
    y1 = int(np.clip(np.rint(top - 1.0 + height), 0, rows))
    if x1 <= x0:
        x1 = min(x0 + 1, cols)
    if y1 <= y0:
        y1 = min(y0 + 1, rows)
    image[y0:y1, x0:x1, :] = color.reshape(1, 1, 3)


def _hdr_lights_parameters(value: Any | None) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    return {
        "imagesize": max(int(np.rint(normalized.get("imagesize", 384))), 1),
        "ncircles": max(int(np.rint(normalized.get("ncircles", 4))), 1),
        "radius": np.asarray(normalized.get("radius", [0.01, 0.035, 0.07, 0.1]), dtype=float).reshape(-1),
        "circlecolors": list(normalized.get("circlecolors", ["white", "green", "blue", "yellow", "magenta", "white"])),
        "nlines": max(int(np.rint(normalized.get("nlines", 4))), 1),
        "linelength": float(normalized.get("linelength", 0.02)),
        "linecolors": list(normalized.get("linecolors", ["white", "green", "blue", "yellow", "magenta", "white"])),
    }


def _hdr_lights_scene(params: dict[str, Any], *, asset_store: AssetStore) -> Scene:
    im_size = int(params["imagesize"])
    image = np.zeros((im_size, im_size, 3), dtype=float)

    n_circles = int(params["ncircles"])
    radii = np.asarray(params["radius"], dtype=float).reshape(-1)
    if radii.size == 1:
        radii = np.repeat(radii, n_circles)
    elif radii.size < n_circles:
        radii = np.pad(radii, (0, n_circles - radii.size), mode="edge")
    y_circle = int(np.rint(im_size * 0.25))
    x_circle = np.rint(np.linspace(0.2, 0.8, n_circles) * im_size).astype(int)
    for index, x_value in enumerate(x_circle):
        color_index = (index + 1) % max(len(params["circlecolors"]), 1)
        _draw_filled_circle(
            image,
            float(x_value),
            float(y_circle),
            float(radii[index] * im_size),
            _hdr_light_color(params["circlecolors"][color_index]),
        )

    n_lines = int(params["nlines"])
    line_length = int(np.rint(float(params["linelength"]) * im_size))
    hw = np.rint(
        np.array(
            [
                [1.0, 7.0 * line_length],
                [1.0, 3.0 * line_length],
                [3.0 * line_length, 1.0],
                [8.0 * line_length, 1.0],
            ],
            dtype=float,
        )
    ).astype(int)
    if n_lines > hw.shape[0]:
        hw = np.vstack((hw, np.repeat(hw[-1:, :], n_lines - hw.shape[0], axis=0)))
    y_line = int(np.rint(im_size * 0.5))
    x_line = np.rint(np.linspace(0.1, 0.8, n_lines) * im_size).astype(int)
    for index, x_value in enumerate(x_line):
        color_index = (index + 1) % max(len(params["linecolors"]), 1)
        _draw_filled_rectangle(
            image,
            float(x_value),
            float(y_line),
            float(hw[index, 0]),
            float(hw[index, 1]),
            _hdr_light_color(params["linecolors"][color_index]),
        )

    y_square = int(np.rint(im_size * 0.75))
    x_square = np.rint(np.linspace(0.1, 0.7, 3) * im_size).astype(int)
    square_edge = float(im_size) / 64.0
    square_sizes = np.array([[2.0, 2.0], [5.0, 5.0], [9.0, 9.0]], dtype=float) * square_edge
    for index, x_value in enumerate(x_square):
        _draw_filled_rectangle(
            image,
            float(x_value),
            float(y_square),
            float(square_sizes[index, 0]),
            float(square_sizes[index, 1]),
            _HDR_LIGHTS_COLOR_MAP["white"],
        )

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    scene = scene_from_file(
        image,
        "rgb",
        1.0e5,
        _scene_display("LCD-Apple", wave, asset_store=asset_store),
        wave,
        asset_store=asset_store,
    )
    background = scene_create("uniform", im_size, wave, asset_store=asset_store)
    background = scene_adjust_luminance(background, 1.0e-2, asset_store=asset_store)
    scene = scene_set(
        scene,
        "photons",
        np.asarray(scene_get(scene, "photons"), dtype=float) + np.asarray(scene_get(background, "photons"), dtype=float),
    )
    scene = scene_adjust_luminance(scene, 100.0, asset_store=asset_store)
    scene.name = "hdr lights"
    scene.fields["illuminant_comment"] = "hdr lights"
    return scene


def _normalized_parameter_dict(value: Any) -> dict[str, Any]:
    if _is_empty_scene_dispatch_placeholder(value):
        return {}
    if isinstance(value, dict):
        raw = value
    elif hasattr(value, "items"):
        raw = dict(value.items())
    elif hasattr(value, "__dict__"):
        raw = vars(value)
    else:
        return {"imagesize": value}
    return {
        param_format(str(key)): item
        for key, item in raw.items()
        if not _is_empty_scene_dispatch_collection_placeholder(item)
    }


def _normalized_key_value_args(args: tuple[Any, ...]) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError("Legacy key/value arguments must come in pairs.")
    normalized: dict[str, Any] = {}
    for index in range(0, len(args), 2):
        normalized[param_format(str(args[index]))] = args[index + 1]
    return normalized


def _scene_scale_peak_luminance(
    scene: Scene,
    target_luminance: float,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    current_peak = float(np.max(luminance)) if luminance.size else 0.0
    if current_peak <= 0.0:
        return scene
    scale = float(target_luminance) / current_peak
    scene.data["photons"] = np.asarray(scene.data["photons"], dtype=float) * scale
    if "illuminant_photons" in scene.fields:
        scene.fields["illuminant_photons"] = np.asarray(scene.fields["illuminant_photons"], dtype=float) * scale
    if "illuminant_energy" in scene.fields:
        scene.fields["illuminant_energy"] = np.asarray(scene.fields["illuminant_energy"], dtype=float) * scale
    _invalidate_scene_caches(scene)
    scene_calculate_luminance(scene, asset_store=asset_store)
    return scene


def _broadcast_parameter_vector(values: np.ndarray, count: int, name: str) -> np.ndarray:
    if values.size == count:
        return values.astype(float, copy=False)
    if values.size == 1:
        return np.repeat(values.astype(float, copy=False), count)
    raise ValueError(f"Harmonic parameter '{name}' must be scalar or length {count}.")


def _scale_range(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    current = np.asarray(values, dtype=float)
    current_min = float(np.min(current))
    current_max = float(np.max(current))
    if current_max <= current_min:
        return np.full_like(current, float(out_min), dtype=float)
    scaled = (current - current_min) / (current_max - current_min)
    return scaled * (float(out_max) - float(out_min)) + float(out_min)


def _harmonic_parameters(value: Any | None) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    default = {
        "name": "harmonicP",
        "ang": np.array([0.0], dtype=float),
        "contrast": np.array([1.0], dtype=float),
        "freq": np.array([1.0], dtype=float),
        "ph": np.array([np.pi / 2.0], dtype=float),
        "row": 65,
        "col": 65,
        "center": np.array([0.0, 0.0], dtype=float),
        "gaborflag": 0.0,
    }

    imagesize = normalized.get("imagesize")
    row = int(np.rint(normalized.get("row", default["row"])))
    col = int(np.rint(normalized.get("col", default["col"])))
    if imagesize is not None:
        image_size = np.asarray(imagesize, dtype=float).reshape(-1)
        if image_size.size == 1:
            row = int(np.rint(image_size[0]))
            col = int(np.rint(image_size[0]))
        elif image_size.size >= 2:
            row = int(np.rint(image_size[0]))
            col = int(np.rint(image_size[1]))

    angle = np.asarray(normalized.get("orientation", normalized.get("ang", default["ang"])), dtype=float).reshape(-1)
    contrast = np.asarray(normalized.get("contrast", default["contrast"]), dtype=float).reshape(-1)
    frequency = np.asarray(normalized.get("frequency", normalized.get("freq", default["freq"])), dtype=float).reshape(-1)
    phase = np.asarray(normalized.get("phase", normalized.get("ph", default["ph"])), dtype=float).reshape(-1)
    count = max(angle.size, contrast.size, frequency.size, phase.size)

    center = np.asarray(normalized.get("center", default["center"]), dtype=float).reshape(-1)
    if center.size == 1:
        center = np.repeat(center, 2)
    if center.size != 2:
        raise ValueError("Harmonic 'center' must contain one or two values.")

    return {
        "name": str(normalized.get("name", default["name"])),
        "ang": _broadcast_parameter_vector(angle, count, "ang"),
        "contrast": _broadcast_parameter_vector(contrast, count, "contrast"),
        "freq": _broadcast_parameter_vector(frequency, count, "freq"),
        "ph": _broadcast_parameter_vector(phase, count, "ph"),
        "row": max(row, 1),
        "col": max(col, 1),
        "center": center.astype(float, copy=False),
        "gaborflag": float(normalized.get("gaborflag", default["gaborflag"])),
    }


def _image_harmonic(params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    row = int(params["row"])
    col = int(params["col"])
    center = np.asarray(params["center"], dtype=float)

    x = np.arange(col, dtype=float) / max(col, 1)
    y = np.arange(row, dtype=float) / max(row, 1)
    x = x - (x[-1] / 2.0)
    y = y - (y[-1] / 2.0)
    x = x - center[0] / max(col, 1)
    y = y - center[1] / max(row, 1)
    xx, yy = np.meshgrid(x, y)

    x_pixels = np.arange(col, dtype=float) - ((col - 1) / 2.0) - center[0]
    y_pixels = np.arange(row, dtype=float) - ((row - 1) / 2.0) - center[1]
    xx_pixels, yy_pixels = np.meshgrid(x_pixels, y_pixels)

    gabor_flag = float(params["gaborflag"])
    if gabor_flag > 0.0:
        sigma = gabor_flag * min(row, col)
        window = np.exp(-((xx_pixels**2 + yy_pixels**2) / (2.0 * sigma**2)))
        window /= max(float(np.max(window)), 1e-12)
    elif gabor_flag < 0.0:
        radius = -gabor_flag * min(row, col)
        x_arg = np.pi * xx_pixels / (2.0 * radius)
        y_arg = np.pi * yy_pixels / (2.0 * radius)
        window = np.cos(x_arg) * np.cos(y_arg)
        mask = (np.abs(x_arg) <= (np.pi / 2.0)) & (np.abs(y_arg) <= (np.pi / 2.0))
        window = np.where(mask, window, 0.0)
        window /= max(float(np.max(window)), 1e-12)
    else:
        window = np.ones((row, col), dtype=float)

    image = np.zeros((row, col), dtype=float)
    count = len(params["freq"])
    for ii in range(count):
        image += (
            params["contrast"][ii]
            * window
            * np.cos(
                2.0
                * np.pi
                * params["freq"][ii]
                * (np.cos(params["ang"][ii]) * xx + np.sin(params["ang"][ii]) * yy)
                + params["ph"][ii]
            )
            + 1.0
        )
    image /= max(count, 1)
    return image, params


def _frequency_orientation_parameters(value: Any | None) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    angles = np.asarray(normalized.get("angles", np.linspace(0.0, np.pi / 2.0, 8)), dtype=float).reshape(-1)
    freqs = np.asarray(normalized.get("freqs", np.arange(1.0, 9.0, dtype=float)), dtype=float).reshape(-1)
    contrast = float(normalized.get("contrast", 1.0))
    block_size = normalized.get("blocksize")
    if block_size is None:
        requested_size = normalized.get("imagesize")
        if requested_size is None:
            block_size = 32
        else:
            requested = np.asarray(requested_size, dtype=float).reshape(-1)
            if requested.size == 1:
                block_size = max(int(np.rint(requested[0] / max(len(angles), len(freqs)))), 1)
            else:
                block_size = max(int(np.rint(min(requested[0] / max(len(freqs), 1), requested[1] / max(len(angles), 1)))), 1)
    return {
        "angles": angles,
        "freqs": freqs,
        "contrast": contrast,
        "blocksize": max(int(np.rint(block_size)), 1),
    }


def _frequency_orientation_image(params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    block_size = int(params["blocksize"])
    x = np.arange(block_size, dtype=float) / block_size
    xx, yy = np.meshgrid(x, x)
    rows: list[np.ndarray] = []
    for frequency in params["freqs"]:
        blocks: list[np.ndarray] = []
        for angle in params["angles"]:
            block = 0.5 * (
                1.0
                + params["contrast"]
                * np.sin(2.0 * np.pi * frequency * (np.cos(angle) * xx + np.sin(angle) * yy))
            )
            blocks.append(block)
        rows.append(np.concatenate(blocks, axis=1))
    image = np.concatenate(rows, axis=0).T
    return image, params


def _sweep_frequency_image(
    size: Any,
    max_frequency: float,
    y_contrast: Any | None = None,
) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    x = np.arange(1.0, cols + 1.0, dtype=float) / cols
    frequency = (x**2) * float(max_frequency)
    x_image = np.sin(2.0 * np.pi * (frequency * x))
    if y_contrast is None:
        contrast = np.linspace(1.0, 0.0, rows, dtype=float)
    else:
        contrast = np.asarray(y_contrast, dtype=float).reshape(-1)
        if contrast.size == 1:
            contrast = np.repeat(contrast, rows)
        elif contrast.size != rows:
            raise ValueError("Sweep-frequency yContrast must be scalar or length rows.")
    image = contrast[:, None] * x_image[None, :] + 0.5
    image = _scale_range(image, 1.0, 256.0)
    return image / max(float(np.max(image)), 1e-12)


def _linear_intensity_ramp_image(size: Any, dynamic_range: float) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    max_column = _matlab_round_scalar(cols / 2.0)
    min_column = -(max_column - 1)
    x_image = np.arange(min_column, min_column + cols, dtype=float)
    y_contrast = np.arange(rows, 0, -1, dtype=float) / max(rows, 1)
    image = (y_contrast[:, None] * x_image[None, :]) + 0.5
    image = _scale_range(image, 1.0, max(float(dynamic_range), 1.0))
    return image / max(float(np.max(image)), 1e-12)


def _exponential_intensity_ramp_image(size: Any, dynamic_range: float) -> np.ndarray:
    rows, cols = _scene_size_2d(size, default=128)
    if float(dynamic_range) <= 1.0:
        ramp = np.ones(cols, dtype=float)
    else:
        ramp = np.logspace(0.0, np.log10(float(dynamic_range)), cols, dtype=float)
    image = np.tile(ramp.reshape(1, -1), (rows, 1))
    image = _scale_range(image, 1.0, max(float(dynamic_range), 1.0))
    return image / max(float(np.max(image)), 1e-12)


def img_deadleaves(n: Any = 256, sigma: float = 3.0, options: Any | None = None) -> np.ndarray:
    """Create the legacy MATLAB dead-leaves grayscale image."""

    return _dead_leaves_image(n, sigma, options)


def image_dead_leaves(
    n: Any = 256,
    sigma: float = 3.0,
    options: Any | None = None,
) -> tuple[np.ndarray, int | None]:
    """Create the legacy MATLAB dead-leaves image plus replay seed."""

    normalized = _normalized_parameter_dict(options)
    returned_seed: int | None
    if "seed" in normalized:
        returned_seed = int(np.rint(normalized["seed"]))
    elif "rseed" in normalized:
        returned_seed = int(np.rint(normalized["rseed"]))
    elif normalized.get("random_samples") is None:
        returned_seed = int(np.random.SeedSequence().entropy)
        normalized["seed"] = returned_seed
    else:
        returned_seed = None
    return _dead_leaves_image(n, sigma, normalized), returned_seed


def img_disk_array(img_size: Any = 512, disk_radius: int = 16, array_size: Any | None = None) -> np.ndarray:
    """Create the legacy MATLAB disk-array grayscale image."""

    return _disk_array_image(img_size, disk_radius, array_size)


def img_mackay(radial_frequency: float = 8.0, im_size: Any = 128) -> np.ndarray:
    """Create the legacy MATLAB MacKay chart image."""

    return _mackay_image(radial_frequency, im_size)


def img_radial_ramp(sz: Any | None = None, expt: float = 1.0, origin: Any | None = None) -> np.ndarray:
    """Create the legacy MATLAB radial-ramp image."""

    rows, cols = _scene_size_2d([256, 256] if sz is None else sz, default=256)
    if origin is None:
        origin_array = (np.array([rows, cols], dtype=float) + 1.0) / 2.0
    else:
        origin_array = np.asarray(origin, dtype=float).reshape(-1)
        if origin_array.size == 1:
            origin_array = np.repeat(origin_array, 2)
        if origin_array.size != 2:
            raise ValueError("imgRadialRamp origin must be a scalar or [row, col].")
    xramp, yramp = np.meshgrid(
        np.arange(1.0, float(cols) + 1.0, dtype=float) - float(origin_array[1]),
        np.arange(1.0, float(rows) + 1.0, dtype=float) - float(origin_array[0]),
    )
    return (xramp * xramp + yramp * yramp) ** (float(expt) / 2.0)


def img_ramp(im_size: Any = 128, dynamic_range: float = 256.0) -> np.ndarray:
    """Create the legacy MATLAB linear-intensity ramp image."""

    return _linear_intensity_ramp_image(im_size, dynamic_range)


def img_square_array(img_size: Any = 512, square_size: int = 16, array_size: Any | None = None) -> np.ndarray:
    """Create the legacy MATLAB square-array grayscale image."""

    return _square_array_image(img_size, square_size, array_size)


def img_sweep(im_size: Any | None = None, max_freq: float | None = None, y_contrast: Any | None = None) -> np.ndarray:
    """Create the legacy MATLAB sweep-frequency grayscale image."""

    size = [128, 128] if im_size is None else im_size
    if max_freq is None:
        rows, cols = _scene_size_2d(size, default=128)
        max_freq = float(cols) / 16.0
        size = [rows, cols]
    return _sweep_frequency_image(size, float(max_freq), y_contrast)


def img_zone_plate(sz: Any | None = None, amp: float = 1.0, ph: float = 0.0) -> np.ndarray:
    """Create the legacy MATLAB zone-plate image."""

    return _zone_plate_image([256, 256] if sz is None else sz, amp, ph)


imgDeadleaves = img_deadleaves
imageDeadLeaves = image_dead_leaves
imgDiskArray = img_disk_array
imgMackay = img_mackay
imgRadialRamp = img_radial_ramp
imgRamp = img_ramp
imgSquareArray = img_square_array
imgSweep = img_sweep
imgZonePlate = img_zone_plate


def fot_params() -> dict[str, Any]:
    """Return the legacy MATLAB FOTParams defaults."""

    return {
        "angles": np.linspace(0.0, np.pi / 2.0, 8, dtype=float),
        "freqs": np.arange(1.0, 9.0, dtype=float),
        "blockSize": 32,
        "contrast": 1.0,
    }


def gabor_p(value: Any | None = None, /, **kwargs: Any) -> dict[str, Any]:
    """Return the legacy MATLAB gaborP parameter structure."""

    normalized = _normalized_parameter_dict(value)
    if kwargs:
        normalized.update(_normalized_parameter_dict(kwargs))
    return {
        "orientation": float(normalized.get("orientation", 0.0)),
        "contrast": float(normalized.get("contrast", 1.0)),
        "frequency": float(normalized.get("frequency", 1.0)),
        "phase": float(normalized.get("phase", np.pi / 2.0)),
        "imagesize": int(np.rint(normalized.get("imagesize", 65))),
        "spread": float(normalized.get("spread", 10.0)),
    }


def ie_checkerboard(check_period: int = 16, n_check_pairs: int = 8) -> np.ndarray:
    """Create the legacy MATLAB checkerboard pattern image."""

    period = max(int(np.rint(check_period)), 1)
    pairs = max(int(np.rint(n_check_pairs)), 1)
    basic_pattern = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float), np.ones((period, period), dtype=float))
    return np.tile(basic_pattern, (pairs, pairs))


def mo_target(pattern: str = "sinusoidalim", parms: Any | None = None) -> np.ndarray:
    """Create the legacy MATLAB moire-orientation RGB target image."""

    normalized = _normalized_parameter_dict(parms)
    scene_size = max(int(np.rint(normalized.get("scenesize", 512))), 1)
    frequency = float(normalized.get("f", 1.0 / scene_size / 10.0))
    x, y = np.meshgrid(np.arange(1.0, scene_size + 1.0, dtype=float), np.arange(1.0, scene_size + 1.0, dtype=float))
    distance = np.sqrt(x * x + y * y)

    normalized_pattern = param_format(pattern)
    if normalized_pattern == "sinusoidalim":
        image = np.sin(np.pi * frequency * distance * distance)
    elif normalized_pattern == "squareim":
        image = np.sin(np.pi * frequency * distance * distance)
        image = (1.0 + np.sign(image - 0.5)) / 2.0
    elif normalized_pattern in {"sinusoidalimline", "squareimline"}:
        line_frequency = float(normalized.get("fline", 0.001))
        theta_line = float(normalized.get("thetaline", np.pi / 2.0))
        spacing = normalized.get("spacingline")
        if spacing is None:
            spacing_array = np.arange(1.0, 501.0, dtype=float)
        else:
            spacing_array = np.asarray(spacing, dtype=float).reshape(-1)
        x_line, y_line = np.meshgrid((2.0 * np.pi / max(line_frequency, 1e-30)) * spacing_array, spacing_array)
        image = np.sin(line_frequency * (np.cos(theta_line) * x_line + np.sin(theta_line) * y_line) ** 2)
        if normalized_pattern == "squareimline":
            image = (1.0 + np.sign(image - 0.5)) / 2.0
    elif normalized_pattern == "flat":
        image = np.full((500, 500), 255.0, dtype=float)
    else:
        raise UnsupportedOptionError("MOTarget", pattern)

    image = np.asarray(image, dtype=float).T
    return np.repeat(image[:, :, None], 3, axis=2)


def _mo_target_scene(parms: Any | None, *, asset_store: AssetStore) -> Scene:
    wave = _wave_or_default(None)
    target = np.asarray(mo_target("sinusoidalim", parms), dtype=float)
    image = np.clip(target[:, :, 1], 1.0e-4, 1.0)
    scene = _equal_photon_pattern_scene("MOTarget", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=asset_store)
    scene.fields["mo_target_params"] = _normalized_parameter_dict(parms)
    return scene


def scene_hdr_chart(
    d_range: float = 1.0e4,
    n_levels: int = 12,
    cols_per_level: int = 8,
    max_l: float | None = None,
    il: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Create the legacy MATLAB HDR strip chart scene."""

    from .illuminant import illuminant_create, illuminant_get

    scene = scene_create("empty", asset_store=_store(asset_store))
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    if il is None:
        illuminant = illuminant_create("d65", wave, 100.0, asset_store=_store(asset_store))
        ill_photons = np.asarray(illuminant_get(illuminant, "photons", asset_store=_store(asset_store)), dtype=float).reshape(-1)
        illuminant_comment = str(illuminant_get(illuminant, "name", asset_store=_store(asset_store)))
    elif isinstance(il, dict):
        if "photons" in il:
            ill_photons = np.asarray(il["photons"], dtype=float).reshape(-1)
        elif "energy" in il:
            ill_photons = np.asarray(energy_to_quanta(np.asarray(il["energy"], dtype=float).reshape(-1), wave), dtype=float)
        else:
            raise ValueError("HDR chart illuminant dict must contain photons or energy.")
        illuminant_comment = str(il.get("name", "user-defined"))
    else:
        ill_photons = np.asarray(illuminant_get(il, "photons", asset_store=_store(asset_store)), dtype=float).reshape(-1)
        illuminant_comment = str(illuminant_get(il, "name", asset_store=_store(asset_store)))

    cols = int(n_levels) * int(cols_per_level)
    rows = cols
    reflectances = np.logspace(0.0, np.log10(1.0 / float(d_range)), int(n_levels), dtype=float)
    photons_by_level = reflectances[:, None] * ill_photons.reshape(1, -1)
    img = np.zeros((rows, cols, wave.size), dtype=float)
    for ll in range(int(n_levels)):
        these_cols = slice(ll * int(cols_per_level), (ll + 1) * int(cols_per_level))
        img[:, these_cols, :] = photons_by_level[ll, :].reshape(1, 1, -1)

    scene = scene_set(scene, "photons", img)
    scene = scene_set(scene, "illuminant photons", ill_photons)
    scene = scene_set(scene, "illuminant comment", illuminant_comment)
    scene.name = "HDR Chart"
    scene = scene_set(
        scene,
        "chart parameters",
        {
            "dRange": float(d_range),
            "nLevels": int(n_levels),
            "colsPerLevel": int(cols_per_level),
            "cornerPoints": np.array([[1.0, float(rows)], [float(cols), float(rows)], [float(cols), 1.0], [1.0, 1.0]], dtype=float),
        },
    )
    if max_l is not None:
        scene = _scene_scale_peak_luminance(scene, float(max_l), asset_store=_store(asset_store))
    return scene


def scene_hdr_image(
    n_patches: int,
    *,
    background: str | Path | np.ndarray | None = "PsychBuilding.png",
    image_size: Any | None = None,
    dynamic_range: float = 3.0,
    patch_shape: str = "square",
    patch_size: int | None = None,
    row: int | None = None,
    asset_store: AssetStore | None = None,
) -> tuple[Scene, Scene]:
    """Create the legacy MATLAB HDR image scene and return the background scene too."""

    from .display import display_create

    store = _store(asset_store)
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    display = display_create(wave=wave, asset_store=store)

    if background is None or (isinstance(background, str) and background == ""):
        if image_size is None:
            image_size = [512, 512]
        background_scene = scene_create("uniformee", image_size, wave, asset_store=store)
        background_scene = scene_adjust_luminance(background_scene, 0.0, asset_store=store)
        im_size = np.asarray(scene_get(background_scene, "size"), dtype=int)
    else:
        input_background = background
        if isinstance(background, str):
            input_background = background
        background_scene = scene_from_file(input_background, "rgb" if not str(background).lower().endswith(".mat") else "spectral", 1.0, display, wave, asset_store=store)
        if image_size is not None:
            background_scene = scene_set(background_scene, "resize", image_size)
        background_scene = scene_adjust_luminance(background_scene, 1.0, asset_store=store)
        im_size = np.asarray(scene_get(background_scene, "size"), dtype=int)

    im_height = int(im_size[0])
    im_width = int(im_size[1])
    patch_levels = np.flip(np.logspace(0.0, float(dynamic_range), int(n_patches), dtype=float))
    composite_scene: Scene | None = None
    normalized_shape = param_format(patch_shape)

    for ii in range(int(n_patches)):
        patch_image = np.zeros((im_height, im_width), dtype=float)
        if normalized_shape == "square":
            patch_width = max(int(np.floor(im_width / (2 * int(n_patches)))), 1) if patch_size is None else int(patch_size)
            patch_height = patch_width
            spacing = im_width / max(int(n_patches) + 1, 1)
            start_cols = np.rint(np.arange(1, int(n_patches) + 1, dtype=float) * spacing).astype(int)
            start_row = int(np.rint((im_height - patch_height) / 2.0)) if row is None else int(row)
            row_slice = slice(max(start_row, 0), min(start_row + patch_height, im_height))
            start_col = int(start_cols[ii] - np.rint(patch_width / 2.0))
            col_slice = slice(max(start_col, 0), min(start_col + patch_width, im_width))
            patch_image[row_slice, col_slice] = 1.0
        elif normalized_shape == "circle":
            radius = max(int(np.floor(im_width / (4 * int(n_patches)))), 1) if patch_size is None else int(patch_size)
            spacing = im_width / max(int(n_patches) + 1, 1)
            center_cols = np.arange(1, int(n_patches) + 1, dtype=float) * spacing
            center_row = int(np.rint(im_height / 2.0)) if row is None else int(row)
            xx, yy = np.meshgrid(np.arange(1, im_width + 1, dtype=float), np.arange(1, im_height + 1, dtype=float))
            dist = np.sqrt((xx - center_cols[ii]) ** 2 + (yy - float(center_row)) ** 2)
            patch_image = (dist < radius).astype(float)
        else:
            raise UnsupportedOptionError("sceneHDRImage", patch_shape)

        patch_rgb = np.repeat(patch_image[:, :, None], 3, axis=2)
        patch_scene = scene_from_file(patch_rgb, "rgb", 1.0, display, wave, asset_store=store)
        patch_scene = _scene_scale_peak_luminance(patch_scene, float(patch_levels[ii]), asset_store=store)
        composite_scene = patch_scene if composite_scene is None else scene_add(composite_scene, patch_scene)

    assert composite_scene is not None
    combined = scene_add(background_scene, composite_scene)
    combined.name = "HDR Image"
    combined = scene_set(
        combined,
        "chart parameters",
        {
            "nPatches": int(n_patches),
            "imageSize": np.asarray(im_size, dtype=int).copy(),
            "dynamicRange": float(dynamic_range),
            "patchShape": str(patch_shape),
            "patchSize": None if patch_size is None else int(patch_size),
            "row": None if row is None else int(row),
        },
    )
    return combined, background_scene


def scene_radiance_chart(
    wave: Any,
    radiance: Any,
    *,
    rowcol: Any | None = None,
    patch_size: int = 10,
    gray_fill: bool = True,
    sampling: str = "r",
    illuminant: Any | None = None,
    asset_store: AssetStore | None = None,
) -> tuple[Scene, np.ndarray]:
    """Create the legacy MATLAB radiance chart scene."""

    chart_wave = np.asarray(wave, dtype=float).reshape(-1)
    radiance_array = np.asarray(radiance, dtype=float)
    if radiance_array.ndim != 2 or radiance_array.shape[0] != chart_wave.size:
        raise ValueError("radiance must be an nWave x nSamples matrix.")

    scene = scene_create("empty", asset_store=_store(asset_store))
    scene = scene_set(scene, "wave", chart_wave)
    n_samples = int(radiance_array.shape[1])
    if rowcol is None:
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / max(rows, 1)))
    else:
        rowcol_array = np.asarray(rowcol, dtype=int).reshape(-1)
        rows = int(rowcol_array[0])
        cols = int(rowcol_array[1])

    expanded = radiance_array.copy()
    if rows * cols > n_samples:
        n_missing = rows * cols - n_samples
        extra = expanded[:, np.random.default_rng(0).integers(0, n_samples, size=n_missing)]
        expanded = np.concatenate((expanded, extra), axis=1)

    illuminant_photons: np.ndarray
    if gray_fill:
        mean_luminance = np.mean(luminance_from_photons(expanded.T, chart_wave, asset_store=_store(asset_store)))
        unit_luminance = float(luminance_from_photons(np.ones(chart_wave.size, dtype=float), chart_wave, asset_store=_store(asset_store)))
        gray_scale = np.linspace(0.2, 3.0, rows, dtype=float)
        gray_column = np.ones((chart_wave.size, rows), dtype=float) * (mean_luminance / max(unit_luminance, 1e-12))
        gray_column = gray_column * gray_scale.reshape(1, -1)
        expanded = np.concatenate((expanded, gray_column), axis=1)
        cols += 1
        illuminant_photons = gray_column[:, -1]
    else:
        if illuminant is None:
            illuminant_photons = np.mean(expanded, axis=1) * 5.0
        else:
            illuminant_photons = np.asarray(illuminant, dtype=float).reshape(-1)

    rc_size = np.array([rows, cols], dtype=int)
    patch_cube = xw_to_rgb_format(expanded.T, rows, cols)
    s_data = image_increase_image_rgb_size(patch_cube, patch_size)

    scene = scene_set(scene, "photons", s_data)
    scene = scene_set(scene, "illuminant photons", illuminant_photons)
    scene = scene_set(scene, "illuminant comment", "scene radiance chart")
    scene.name = "Radiance Chart (EE)"
    scene = scene_set(
        scene,
        "chart parameters",
        {
            "patchSize": int(patch_size),
            "grayFill": bool(gray_fill),
            "sampling": str(sampling),
            "rowcol": rc_size.copy(),
            "cornerPoints": np.array(
                [[1.0, float(rows * patch_size)], [float(cols * patch_size), float(rows * patch_size)], [float(cols * patch_size), 1.0], [1.0, 1.0]],
                dtype=float,
            ),
        },
    )
    return scene, rc_size


def _vernier_image(params: dict[str, Any]) -> np.ndarray:
    return image_vernier(params)[0]


def _coerce_rgb_triplet(value: Any, default: float) -> np.ndarray:
    if value is None:
        return np.repeat(float(default), 3)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.repeat(float(array[0]), 3)
    if array.size != 3:
        raise ValueError("Color parameters must be scalar or RGB triplets.")
    return array.astype(float, copy=False)


def _vernier_parameter_payload(params: Any | None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(params)
    if len(args) % 2 != 0:
        raise ValueError("imageVernier expects key/value pairs after the parameter struct.")
    for index in range(0, len(args), 2):
        normalized[param_format(str(args[index]))] = args[index + 1]
    if kwargs:
        normalized.update(_normalized_parameter_dict(kwargs))

    scene_size_value = normalized.get("scenesz", 64)
    if np.isscalar(scene_size_value):
        scene_size = np.repeat(int(scene_size_value), 2)
    else:
        scene_size_array = np.asarray(scene_size_value, dtype=int).reshape(-1)
        if scene_size_array.size == 1:
            scene_size = np.repeat(int(scene_size_array[0]), 2)
        elif scene_size_array.size >= 2:
            scene_size = scene_size_array[:2].astype(int, copy=False)
        else:
            raise ValueError("sceneSz must be scalar or length two.")

    return {
        "sceneSz": scene_size.copy(),
        "barWidth": int(normalized.get("barwidth", 1)),
        "barLength": int(normalized.get("barlength", scene_size[0])),
        "offset": int(normalized.get("offset", 1)),
        "gap": int(normalized.get("gap", 0)),
        "barColor": _coerce_rgb_triplet(normalized.get("barcolor", 1.0), 1.0),
        "bgColor": _coerce_rgb_triplet(normalized.get("bgcolor", 0.0), 0.0),
        "pattern": normalized.get("pattern"),
    }


def _insert_vernier_gap(image: np.ndarray, gap: int) -> np.ndarray:
    if gap == 0:
        return image
    rows = image.shape[0]
    if (rows % 2) != (gap % 2):
        raise ValueError(f"Bad row size {rows}, gap size {gap} pair.")
    if rows % 2 == 1:
        middle = (rows - 1) // 2
        offsets = np.arange(gap, dtype=int) - gap // 2
        gap_rows = middle + offsets
    else:
        middle = rows // 2
        gap_rows = np.arange(gap, dtype=int) - (gap // 2) + middle
    with_gap = image.copy()
    with_gap[gap_rows, :, :] = np.nan
    return with_gap


def image_vernier(
    params: Any | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create the legacy MATLAB Vernier image and return the resolved parameters."""

    resolved = _vernier_parameter_payload(params, *args, **kwargs)
    scene_size = np.asarray(resolved["sceneSz"], dtype=int).reshape(2)
    rows = int(scene_size[0])
    cols = int(scene_size[1])
    bar_width = int(resolved["barWidth"])
    bar_length = int(resolved["barLength"])
    gap = int(resolved["gap"])
    offset = int(resolved["offset"])
    bar_color = np.asarray(resolved["barColor"], dtype=float).reshape(3)
    bg_color = np.asarray(resolved["bgColor"], dtype=float).reshape(3)

    pattern = resolved["pattern"]
    if pattern is None:
        base = np.broadcast_to(bg_color.reshape(1, 1, 3), (1, cols, 3)).copy()
        bar_start = _matlab_round_scalar((cols - bar_width) / 2.0)
        bar_stop = bar_start + bar_width
        base[:, max(bar_start, 0) : min(bar_stop, cols), :] = bar_color.reshape(1, 1, 3)
    else:
        pattern_array = np.asarray(pattern, dtype=float)
        if pattern_array.ndim == 1:
            base = np.repeat(pattern_array.reshape(1, -1, 1), 3, axis=2)
        elif pattern_array.ndim == 2:
            base = np.repeat(pattern_array[:, :, np.newaxis], 3, axis=2)
        elif pattern_array.ndim == 3:
            base = pattern_array.astype(float, copy=True)
        else:
            raise ValueError("pattern must be 1-D, 2-D, or 3-D.")
    image = np.tile(base, (bar_length, 1, 1))
    image = _insert_vernier_gap(image, gap)

    half_rows = int(np.rint(image.shape[0] / 2.0))
    image[:half_rows, :, :] = np.roll(image[:half_rows, :, :], shift=offset, axis=1)

    pad_rows = max(int(np.ceil((rows - image.shape[0]) / 2.0)), 0)
    if pad_rows > 0:
        image = np.pad(image, ((pad_rows, pad_rows), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
    image = image[:rows, :, :]
    if image.shape[0] < rows:
        missing = rows - image.shape[0]
        image = np.pad(image, ((0, missing), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
    for channel in range(3):
        channel_view = image[:, :, channel]
        channel_view[np.isnan(channel_view)] = bg_color[channel]
        image[:, :, channel] = channel_view
    return image, resolved


imageVernier = image_vernier


def scene_vernier(
    scene: Scene,
    type: str = "display",
    params: dict[str, Any] | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Create the legacy MATLAB Vernier scene from display or object parameters."""

    from .display import display_create
    from .illuminant import illuminant_create, illuminant_get

    if not isinstance(scene, Scene):
        raise ValueError("sceneVernier requires a scene input.")
    parameters = _normalized_parameter_dict(params)
    normalized_type = param_format(type or "display")

    if normalized_type == "display":
        display_value = parameters.get("display", "LCD-Apple")
        display = display_value if not isinstance(display_value, str) else display_create(display_value, asset_store=_store(asset_store))
        image = _vernier_image(parameters)
        result = scene_from_file(image, "rgb", None, display, asset_store=_store(asset_store))
    elif normalized_type == "object":
        current = scene.clone()
        scene_size = parameters.get("scenesz", 64)
        if np.isscalar(scene_size):
            rows = cols = int(scene_size)
        else:
            size_array = np.asarray(scene_size, dtype=int).reshape(-1)
            rows = int(size_array[0])
            cols = int(size_array[1])
        if cols % 2 == 0:
            cols += 1
        bar_width = int(parameters.get("barwidth", 1))
        offset = int(parameters.get("offset", 1))
        line_reflectance = float(parameters.get("barreflect", 0.6))
        back_reflectance = float(parameters.get("bgreflect", 0.3))
        wave = np.asarray(current.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
        if "wave" not in current.fields:
            current = init_default_spectrum(current, "hyperspectral")
            wave = np.asarray(current.fields["wave"], dtype=float).reshape(-1)

        illuminant_param = parameters.get("il")
        if illuminant_param is None:
            illuminant = illuminant_create("equal photons", wave, 100.0, asset_store=_store(asset_store))
            ill_photons = np.asarray(illuminant_get(illuminant, "photons", asset_store=_store(asset_store)), dtype=float).reshape(-1)
            illuminant_comment = str(illuminant_get(illuminant, "name", asset_store=_store(asset_store)))
        elif isinstance(illuminant_param, str):
            illuminant = illuminant_create(illuminant_param, wave, 100.0, asset_store=_store(asset_store))
            ill_photons = np.asarray(illuminant_get(illuminant, "photons", asset_store=_store(asset_store)), dtype=float).reshape(-1)
            illuminant_comment = str(illuminant_get(illuminant, "name", asset_store=_store(asset_store)))
        elif isinstance(illuminant_param, dict):
            if "photons" in illuminant_param:
                ill_photons = np.asarray(illuminant_param["photons"], dtype=float).reshape(-1)
            elif "energy" in illuminant_param:
                ill_photons = np.asarray(energy_to_quanta(np.asarray(illuminant_param["energy"], dtype=float).reshape(-1), wave), dtype=float)
            else:
                raise ValueError("sceneVernier object illuminant dict must contain photons or energy.")
            illuminant_comment = str(illuminant_param.get("name", "user-defined"))
        else:
            ill_photons = np.asarray(illuminant_get(illuminant_param, "photons", asset_store=_store(asset_store)), dtype=float).reshape(-1)
            illuminant_comment = str(illuminant_get(illuminant_param, "name", asset_store=_store(asset_store)))

        photons = np.ones((rows, cols, wave.size), dtype=float)
        photons *= back_reflectance * ill_photons.reshape(1, 1, -1)
        top_cols = np.arange(int(np.rint((cols - bar_width) / 2.0)) - int(np.floor(offset / 2.0)), int(np.rint((cols - bar_width) / 2.0)) - int(np.floor(offset / 2.0)) + bar_width)
        bot_cols = top_cols + offset
        top_cols = np.clip(top_cols, 0, cols - 1)
        bot_cols = np.clip(bot_cols, 0, cols - 1)
        top_half = int(np.rint(rows / 2.0))
        photons[:top_half, top_cols, :] = line_reflectance * ill_photons.reshape(1, 1, -1)
        photons[top_half:, bot_cols, :] = line_reflectance * ill_photons.reshape(1, 1, -1)
        current.name = f"vernier-{offset}"
        current = scene_set(current, "photons", photons)
        current = scene_set(current, "illuminant photons", ill_photons)
        current = scene_set(current, "illuminant comment", illuminant_comment)
        result = current
    else:
        raise UnsupportedOptionError("sceneVernier", type)

    mean_lum = parameters.get("meanlum")
    if mean_lum is not None:
        result = scene_adjust_luminance(result, float(mean_lum), asset_store=_store(asset_store))
    return result


def scene_ramp(
    scene: Scene,
    sz: Any = 128,
    dynamic_range: float = 256.0,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Apply the legacy MATLAB sceneRamp contract to a scene object."""

    if not isinstance(scene, Scene):
        raise ValueError("sceneRamp requires a scene input.")

    current = scene.clone()
    wave = current.fields.get("wave")
    if wave is None:
        wave = _wave_or_default(None)
    wave_array = np.asarray(wave, dtype=float).reshape(-1)
    ramp = _equal_photon_pattern_scene(
        f"ramp DR {float(dynamic_range):.1f}",
        _linear_intensity_ramp_image(sz, dynamic_range),
        wave_array,
        fov_deg=float(current.fields.get("fov_deg", DEFAULT_FOV_DEG)),
        asset_store=_store(asset_store),
    )
    if "distance_m" in current.fields:
        ramp.fields["distance_m"] = float(current.fields["distance_m"])
        _update_scene_geometry(ramp)
    ramp.metadata = dict(current.metadata)
    return ramp


def _equal_photon_pattern_scene(
    name: str,
    image: np.ndarray,
    wave: np.ndarray,
    *,
    fov_deg: float,
    asset_store: AssetStore,
    illuminant_comment: str = "equal photons",
) -> Scene:
    illuminant_energy, illuminant_photons = _spectral_illuminant("ep", wave, asset_store=asset_store)
    scene = Scene(name=name)
    scene.fields["wave"] = wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = illuminant_comment
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = float(fov_deg)
    scene.data["photons"] = np.asarray(image, dtype=float)[:, :, None] * illuminant_photons.reshape(1, 1, -1)
    _update_scene_geometry(scene)
    return scene_adjust_luminance(scene, 100.0, asset_store=asset_store)


def _default_reflectance_chart_files() -> list[str]:
    return [
        "MunsellSamples_Vhrel.mat",
        "Food_Vhrel.mat",
        "skin/HyspexSkinReflectance.mat",
    ]


def _default_reflectance_sample_files() -> list[str]:
    return [
        "MunsellSamples_Vhrel.mat",
        "Food_Vhrel.mat",
        "DupontPaintChip_Vhrel.mat",
        "skin/HyspexSkinReflectance.mat",
    ]


def _reflectance_chart_sources(value: Any | None) -> list[Any]:
    if _is_empty_scene_dispatch_placeholder(value):
        return _default_reflectance_chart_files()
    if isinstance(value, (str, Path, np.ndarray)):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value.copy()
    return [value]


def _load_reflectance_source(
    source: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.exists():
            data = asset_store.load_mat(source_path)
        else:
            try:
                data = asset_store.load_mat(Path("data/surfaces/reflectances") / source_path)
            except MissingAssetError:
                surfaces_root = asset_store.ensure() / "data" / "surfaces"
                matches = sorted(surfaces_root.rglob(source_path.name))
                if not matches:
                    raise
                data = asset_store.load_mat(matches[0])
        reflectance = np.asarray(data["data"], dtype=float)
        source_wave = np.asarray(data["wavelength"], dtype=float).reshape(-1)
        if reflectance.ndim == 1:
            reflectance = reflectance.reshape(-1, 1)
        if not np.array_equal(source_wave, wave):
            from .utils import interp_spectra

            reflectance = interp_spectra(source_wave, reflectance, wave)
        return np.asarray(reflectance, dtype=float)
    reflectance = np.asarray(source, dtype=float)
    if reflectance.ndim == 1:
        reflectance = reflectance.reshape(-1, 1)
    if reflectance.shape[0] != wave.size:
        raise ValueError("Reflectance matrices must be provided on the target wavelength grid.")
    return reflectance


def _reflectance_sample_lists(
    reflectance_sets: list[np.ndarray],
    sample_spec: Any,
    sampling: str,
) -> list[np.ndarray]:
    normalized_sampling = param_format(sampling)
    with_replacement = normalized_sampling.startswith("r")
    use_all = normalized_sampling == "all"

    if isinstance(sample_spec, (list, tuple)) and len(sample_spec) == len(reflectance_sets):
        explicit = []
        is_explicit = True
        for item in sample_spec:
            if np.isscalar(item):
                is_explicit = False
                break
            values = np.asarray(item, dtype=int).reshape(-1)
            if values.size == 0:
                explicit.append(values)
                continue
            explicit.append(values)
        if is_explicit:
            return explicit

    if sample_spec is None:
        counts = [24] * len(reflectance_sets)
    else:
        counts_array = np.asarray(sample_spec, dtype=int).reshape(-1)
        if counts_array.size != len(reflectance_sets):
            raise ValueError("Reflectance chart sample counts must match the number of reflectance sources.")
        counts = counts_array.tolist()

    rng = np.random.default_rng(0)
    sample_lists: list[np.ndarray] = []
    for count, reflectance in zip(counts, reflectance_sets, strict=True):
        n_reflectances = reflectance.shape[1]
        if use_all:
            sample_lists.append(np.arange(1, n_reflectances + 1, dtype=int))
            continue
        count = int(count)
        if with_replacement:
            sample_lists.append(rng.integers(1, n_reflectances + 1, size=count, endpoint=False, dtype=int))
        else:
            if count > n_reflectances:
                raise ValueError("Requested more reflectance samples than available without replacement.")
            perm = rng.permutation(n_reflectances)[:count]
            sample_lists.append(np.asarray(perm + 1, dtype=int))
    return sample_lists


def ie_reflectance_samples(
    s_files: Any | None = None,
    s_samples: Any | None = None,
    wave: Any | None = None,
    sampling: str | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Return sampled surface reflectances following MATLAB's ieReflectanceSamples()."""

    store = _store(asset_store)
    if s_files is None or (isinstance(s_files, (list, tuple)) and len(s_files) == 0):
        source_files = _default_reflectance_sample_files()
        if s_samples is None:
            sample_spec = np.array([24, 24, 24, 24], dtype=int)
        else:
            sample_spec = s_samples
    else:
        source_files = _reflectance_chart_sources(s_files)
        if s_samples is None:
            sample_spec = np.zeros(len(source_files), dtype=int)
        else:
            sample_spec = s_samples

    if len(source_files) == 0:
        raise ValueError("ie_reflectance_samples requires at least one reflectance source.")

    sample_wave = np.arange(400.0, 701.0, 10.0, dtype=float) if wave is None else np.asarray(wave, dtype=float).reshape(-1)
    if sample_wave.size == 0:
        raise ValueError("ie_reflectance_samples wavelength sampling must not be empty.")
    sampling_mode = "r" if sampling is None else str(sampling)

    reflectance_sets = [_load_reflectance_source(source, sample_wave, asset_store=store) for source in source_files]
    sample_lists = _reflectance_sample_lists(reflectance_sets, sample_spec, sampling_mode)

    total_samples = int(sum(sample_list.size for sample_list in sample_lists))
    reflectances = np.zeros((sample_wave.size, total_samples), dtype=float)
    last = 0
    for reflectance_set, sample_list in zip(reflectance_sets, sample_lists, strict=True):
        if sample_list.size == 0:
            continue
        this_count = int(sample_list.size)
        reflectances[:, last : last + this_count] = reflectance_set[:, sample_list.astype(int) - 1]
        last += this_count

    if reflectances.size and float(np.max(reflectances)) > 1.0:
        raise ValueError("Reflectance samples must not exceed 1.0.")
    return reflectances, [np.asarray(item, dtype=int).copy() for item in sample_lists], sample_wave.copy()


def _reflectance_chart_parameters(
    value: Any | None,
    *,
    wave: np.ndarray | None = None,
) -> dict[str, Any]:
    normalized = _normalized_parameter_dict(value)
    files = _reflectance_chart_sources(normalized.get("sfiles"))
    samples = normalized.get("ssamples", np.array([50, 40, 10], dtype=int))
    if _is_empty_scene_dispatch_placeholder(samples):
        samples = np.array([50, 40, 10], dtype=int)
    patch_size = int(np.rint(normalized.get("psize", 24)))
    chart_wave = _wave_or_default(normalized.get("wave", wave))
    gray_flag = bool(normalized.get("grayflag", 1))
    sampling = str(normalized.get("sampling", "r"))
    return {
        "sfiles": files,
        "ssamples": samples,
        "psize": max(patch_size, 1),
        "wave": chart_wave,
        "grayflag": gray_flag,
        "sampling": sampling,
    }


def _reflectance_chart_scene(
    source_files: list[Any],
    sample_spec: Any,
    patch_size: int,
    wave: np.ndarray,
    gray_flag: bool,
    sampling: str,
    *,
    asset_store: AssetStore,
) -> Scene:
    scene, _, _, _ = scene_reflectance_chart(
        source_files,
        sample_spec,
        patch_size,
        wave,
        gray_flag,
        sampling,
        asset_store=asset_store,
    )
    return scene


def scene_reflectance_chart(
    s_files: Any,
    s_samples: Any,
    p_size: int | float = 32,
    wave: Any | None = None,
    gray_flag: bool = True,
    sampling: str = "r",
    *,
    asset_store: AssetStore | None = None,
) -> tuple[Scene, list[np.ndarray], np.ndarray, np.ndarray]:
    """Create a reflectance chart using MATLAB sceneReflectanceChart() semantics."""

    if s_files is None:
        raise ValueError("scene_reflectance_chart requires s_files.")
    if s_samples is None:
        raise ValueError("scene_reflectance_chart requires s_samples.")

    store = _store(asset_store)
    source_files = _reflectance_chart_sources(s_files)
    chart_wave = _wave_or_default(wave)
    patch_size = max(int(np.rint(p_size)), 1)

    reflectance_sets = [_load_reflectance_source(source, chart_wave, asset_store=store) for source in source_files]
    sample_lists = _reflectance_sample_lists(reflectance_sets, s_samples, sampling)
    sampled_blocks = []
    for reflectance, sample_list in zip(reflectance_sets, sample_lists, strict=True):
        if sample_list.size == 0:
            continue
        sampled_blocks.append(reflectance[:, sample_list.astype(int) - 1])
    if sampled_blocks:
        reflectances = np.concatenate(sampled_blocks, axis=1)
    else:
        reflectances = np.zeros((chart_wave.size, 0), dtype=float)

    n_samples = reflectances.shape[1]
    rows = int(np.ceil(np.sqrt(n_samples))) if n_samples > 0 else 1
    cols = int(np.ceil(n_samples / max(rows, 1))) if n_samples > 0 else 1
    if gray_flag:
        gray_strip = np.ones((chart_wave.size, rows), dtype=float) * np.logspace(0.0, np.log10(0.05), rows, dtype=float)
        reflectances = np.concatenate((reflectances, gray_strip), axis=1)
        cols += 1

    illuminant_energy, illuminant_photons = _spectral_illuminant("ee", chart_wave, asset_store=store)
    radiance = reflectances * illuminant_photons.reshape(-1, 1)
    patch_cube = np.zeros((rows, cols, chart_wave.size), dtype=float)
    index_map = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            idx = row + col * rows
            if idx < radiance.shape[1]:
                patch_cube[row, col, :] = radiance[:, idx]
                index_map[row, col] = idx + 1
            else:
                patch_cube[row, col, :] = 0.2 * illuminant_photons

    xyz = xyz_from_energy(
        quanta_to_energy(patch_cube.reshape(-1, chart_wave.size), chart_wave),
        chart_wave,
        asset_store=store,
    ).reshape(rows, cols, 3)

    photons = np.repeat(np.repeat(patch_cube, patch_size, axis=0), patch_size, axis=1)
    scene = Scene(name="Reflectance Chart (EE)")
    scene.fields["wave"] = chart_wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["illuminant_energy"] = illuminant_energy
    scene.fields["illuminant_photons"] = illuminant_photons
    scene.fields["illuminant_comment"] = "Equal energy"
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    scene.fields["chart_parameters"] = {
        "sFiles": [str(item) for item in source_files],
        "sSamples": [np.asarray(item, dtype=int).copy() for item in sample_lists],
        "grayFlag": bool(gray_flag),
        "sampling": str(sampling),
        "pSize": int(patch_size),
        "wave": np.asarray(chart_wave, dtype=float).copy(),
        "XYZ": xyz.copy(),
        "rowcol": np.array([rows, cols], dtype=int),
        "rIdxMap": np.repeat(np.repeat(index_map, patch_size, axis=0), patch_size, axis=1),
    }
    scene.data["photons"] = photons
    _update_scene_geometry(scene)
    return (
        scene_adjust_luminance(scene, 100.0, asset_store=store),
        [np.asarray(item, dtype=int).copy() for item in sample_lists],
        np.asarray(reflectances, dtype=float),
        np.array([rows, cols], dtype=int),
    )


def scene_create(
    scene_name: str = "default",
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Scene:
    """Create a supported milestone-one scene."""

    store = _store(asset_store)
    resolved_name = (
        "default"
        if _is_empty_scene_dispatch_placeholder(scene_name)
        or (isinstance(scene_name, str) and scene_name.strip() == "")
        else scene_name
    )
    name = param_format(resolved_name)

    if name in {"list", "scenelist"}:
        return scene_list()

    if name in {"default", "macbeth", "macbethd65", "macbethcustomreflectance"}:
        patch_size = _macbeth_patch_size_arg(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = _scene_dispatch_path_arg(args[2] if len(args) > 2 else None, "macbethChart.mat")
        black_border = (
            False if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else bool(args[3])
        )
        return track_session_object(
            session,
            _create_macbeth_scene(patch_size, wave, surface_file, black_border, asset_store=store),
        )

    if name == "macbethd50":
        patch_size = _macbeth_patch_size_arg(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = _scene_dispatch_path_arg(args[2] if len(args) > 2 else None, "macbethChart.mat")
        black_border = (
            False if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else bool(args[3])
        )
        return track_session_object(
            session,
            _create_macbeth_scene(
                patch_size,
                wave,
                surface_file,
                black_border,
                illuminant_type="d50",
                asset_store=store,
            ),
        )

    if name in {"macbethc", "macbethillc"}:
        patch_size = _macbeth_patch_size_arg(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = _scene_dispatch_path_arg(args[2] if len(args) > 2 else None, "macbethChart.mat")
        black_border = (
            False if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else bool(args[3])
        )
        return track_session_object(
            session,
            _create_macbeth_scene(
                patch_size,
                wave,
                surface_file,
                black_border,
                illuminant_type="illuminantc",
                asset_store=store,
            ),
        )

    if name in {"macbethfluorescent", "macbethfluor"}:
        patch_size = _macbeth_patch_size_arg(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = _scene_dispatch_path_arg(args[2] if len(args) > 2 else None, "macbethChart.mat")
        black_border = (
            False if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else bool(args[3])
        )
        return track_session_object(
            session,
            _create_macbeth_scene(
                patch_size,
                wave,
                surface_file,
                black_border,
                illuminant_type="fluorescent",
                asset_store=store,
            ),
        )

    if name in {"macbethtungsten", "macbethtung"}:
        patch_size = _macbeth_patch_size_arg(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = _scene_dispatch_path_arg(args[2] if len(args) > 2 else None, "macbethChart.mat")
        black_border = (
            False if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else bool(args[3])
        )
        return track_session_object(
            session,
            _create_macbeth_scene(
                patch_size,
                wave,
                surface_file,
                black_border,
                illuminant_type="tungsten",
                asset_store=store,
            ),
        )

    if name in {"macbethee_ir", "macbethequalenergyinfrared"}:
        patch_size = _macbeth_patch_size_arg(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        surface_file = _scene_dispatch_path_arg(args[2] if len(args) > 2 else None, "macbethChart.mat")
        black_border = (
            False if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else bool(args[3])
        )
        return track_session_object(
            session,
            _create_macbeth_scene(
                patch_size,
                wave,
                surface_file,
                black_border,
                illuminant_type="ir",
                asset_store=store,
            ),
        )

    if name in {"monochrome", "unispectral"}:
        return track_session_object(
            session,
            _seeded_scene_shell(
                "monochrome",
                "monochrome",
                args[0] if args and isinstance(args[0], Scene) else None,
                asset_store=store,
            ),
        )

    if name in {"multispectral", "hyperspectral"}:
        return track_session_object(
            session,
            _seeded_scene_shell(
                "multispectral",
                "multispectral",
                args[0] if args and isinstance(args[0], Scene) else None,
                asset_store=store,
            ),
        )

    if name == "rgb":
        return track_session_object(
            session,
            _seeded_scene_shell(
                "rgb",
                "hyperspectral",
                args[0] if args and isinstance(args[0], Scene) else None,
                asset_store=store,
            ),
        )

    if name == "lstar":
        bar_size = (
            np.array([128, 20], dtype=float)
            if len(args) == 0 or _is_empty_scene_dispatch_placeholder(args[0])
            else args[0]
        )
        n_bars = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 10)
        delta_e = _scene_dispatch_float_arg(args[2] if len(args) > 2 else None, 10.0)
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        return track_session_object(session, _lstar_steps_scene(bar_size, n_bars, delta_e, wave, asset_store=store))

    if name == "empty":
        wave_arg = args[1] if len(args) > 1 else (args[0] if len(args) > 0 else None)
        wave = _wave_or_default(wave_arg)
        scene = _create_macbeth_scene(16, wave, "macbethChart.mat", False, asset_store=store)
        return track_session_object(session, scene_clear_data(scene))

    if name in {"uniformd65", "uniform d65".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        _, illuminant_energy = _load_d65(wave, store)
        return track_session_object(
            session,
            _uniform_scene("Uniform D65", size, wave, illuminant_energy, illuminant_comment="D65.mat", asset_store=store),
        )

    if name in {"uniform", "uniformee", "uniformequalenergy", "uniformeespecify"}:
        size = args[0] if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        return track_session_object(
            session,
            _uniform_scene(
                "Uniform EE",
                size,
                wave,
                np.ones(wave.size, dtype=float),
                illuminant_comment="equal-energy",
                asset_store=store,
            ),
        )

    if name in {"mackay", "rayimage", "ringsrays"}:
        radial_frequency = _scene_dispatch_float_arg(args[0] if len(args) > 0 else None, 8.0)
        image_size = 256 if len(args) < 2 or _is_empty_scene_dispatch_placeholder(args[1]) else args[1]
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _mackay_scene(radial_frequency, image_size, wave, asset_store=store))

    if name in {"uniformep", "uniformephoton", "uniformequalphoton", "uniformequalphotons"}:
        size = args[0] if len(args) > 0 else 32
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        illuminant_photons = np.ones(wave.size, dtype=float)
        illuminant_energy = quanta_to_energy(illuminant_photons, wave)
        return track_session_object(
            session,
            _uniform_scene(
                "Uniform EP",
                size,
                wave,
                illuminant_energy,
                illuminant_comment="equal-photons",
                asset_store=store,
            ),
        )

    if name in {"uniformbb", "uniformblackbody"}:
        size = args[0] if len(args) > 0 else 32
        temperature_k = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 5000.0)
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _uniform_blackbody_scene(size, temperature_k, wave, asset_store=store))

    if name in {"uniformmonochromatic", "narrowband"}:
        default_wavelength = 500.0
        default_size = 128
        if len(args) >= 2:
            first_arg = args[0]
            second_arg = args[1]
            first_empty = _is_empty_scene_dispatch_placeholder(first_arg)
            second_empty = _is_empty_scene_dispatch_placeholder(second_arg)
            first_size_like = _looks_like_scene_size_arg(first_arg)
            first_wave_like = _looks_like_monochromatic_wavelength_arg(first_arg)
            second_wave_like = _looks_like_monochromatic_wavelength_arg(second_arg)

            if first_empty and second_wave_like:
                size = default_size
                wavelength = second_arg
            elif first_size_like and (second_wave_like or second_empty) and not first_wave_like:
                size = first_arg
                wavelength = default_wavelength if second_empty else second_arg
            else:
                wavelength = default_wavelength if first_empty else first_arg
                size = default_size if second_empty else second_arg
        else:
            wavelength = default_wavelength if len(args) == 0 or _is_empty_scene_dispatch_placeholder(args[0]) else args[0]
            size = default_size
        return track_session_object(session, _uniform_monochromatic_scene(size, wavelength, asset_store=store))

    if name in {"hdrlights", "highdynamicrange", "hdr"}:
        if args and isinstance(args[0], str):
            params = _hdr_lights_parameters(_normalized_key_value_args(args))
        else:
            params = _hdr_lights_parameters(args[0] if len(args) > 0 else None)
        return track_session_object(session, _hdr_lights_scene(params, asset_store=store))

    if name in {"hdrchart", "hdr chart".replace(" ", "")}:
        if args and isinstance(args[0], str):
            params = _normalized_key_value_args(args)
            d_range = _scene_dispatch_float_arg(params["drange"], 1.0e4) if "drange" in params else 10.0**3.5
            n_levels = _scene_dispatch_int_arg(params["nlevels"], 12) if "nlevels" in params else 16
            cols_per_level = _scene_dispatch_int_arg(params["colsperlevel"], 8) if "colsperlevel" in params else 12
            max_l = None if "maxl" not in params or _is_empty_scene_dispatch_placeholder(params["maxl"]) else float(params["maxl"])
            illuminant = None if "illuminant" not in params or _is_empty_scene_dispatch_placeholder(params["illuminant"]) else params["illuminant"]
        else:
            d_range = _scene_dispatch_float_arg(args[0], 1.0e4) if len(args) > 0 and _is_empty_scene_dispatch_placeholder(args[0]) else (float(args[0]) if len(args) > 0 else 10.0**3.5)
            n_levels = _scene_dispatch_int_arg(args[1], 12) if len(args) > 1 and _is_empty_scene_dispatch_placeholder(args[1]) else (int(args[1]) if len(args) > 1 else 16)
            cols_per_level = _scene_dispatch_int_arg(args[2], 8) if len(args) > 2 and _is_empty_scene_dispatch_placeholder(args[2]) else (int(args[2]) if len(args) > 2 else 12)
            max_l = None if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else float(args[3])
            illuminant = None if len(args) <= 4 or _is_empty_scene_dispatch_placeholder(args[4]) else args[4]
        return track_session_object(session, scene_hdr_chart(d_range, n_levels, cols_per_level, max_l, illuminant, asset_store=store))

    if name in {"hdrimage", "hdr image".replace(" ", "")}:
        if args and isinstance(args[0], str):
            params = _normalized_key_value_args(args)
            n_patches = _scene_dispatch_int_arg(params["npatches"], 8) if "npatches" in params else 8
        else:
            n_patches = _scene_dispatch_int_arg(args[0], 8) if len(args) > 0 else 8
            params = _normalized_parameter_dict(args[1]) if len(args) > 1 and hasattr(args[1], "items") else {}
            if len(args) > 1 and not hasattr(args[1], "items"):
                params["background"] = None if _is_empty_scene_dispatch_placeholder(args[1]) else args[1]
        scene, _ = scene_hdr_image(
            n_patches,
            background="PsychBuilding.png" if "background" not in params else (None if _is_empty_scene_dispatch_placeholder(params.get("background")) else params.get("background")),
            image_size=params.get("imagesize"),
            dynamic_range=3.0 if _is_empty_scene_dispatch_placeholder(params.get("dynamicrange")) else float(params.get("dynamicrange", 3.0)),
            patch_shape="square" if _is_empty_scene_dispatch_placeholder(params.get("patchshape")) else str(params.get("patchshape", "square")),
            patch_size=None if _is_empty_scene_dispatch_placeholder(params.get("patchsize")) else int(params.get("patchsize")),
            row=None if _is_empty_scene_dispatch_placeholder(params.get("row")) else int(params.get("row")),
            asset_store=store,
        )
        return track_session_object(session, scene)

    if name in {"radiancechart", "radiance chart".replace(" ", "")}:
        if len(args) < 2:
            raise ValueError("sceneCreate('radiance chart', ...) requires wave and radiance.")
        if len(args) > 2 and hasattr(args[2], "items"):
            params = _normalized_parameter_dict(args[2])
        elif len(args) > 2 and isinstance(args[2], str):
            params = _normalized_key_value_args(tuple(args[2:]))
        elif len(args) > 2:
            params = {
                "rowcol": None if _is_empty_scene_dispatch_placeholder(args[2]) else args[2],
                "patchsize": [] if len(args) <= 3 else args[3],
                "grayfill": [] if len(args) <= 4 else args[4],
                "sampling": [] if len(args) <= 5 else args[5],
                "illuminant": [] if len(args) <= 6 else args[6],
            }
        else:
            params = {}
        scene, _ = scene_radiance_chart(
            args[0],
            args[1],
            rowcol=None if _is_empty_scene_dispatch_placeholder(params.get("rowcol")) else params.get("rowcol"),
            patch_size=10 if _is_empty_scene_dispatch_placeholder(params.get("patchsize")) else int(params.get("patchsize", 10)),
            gray_fill=True if _is_empty_scene_dispatch_placeholder(params.get("grayfill")) else bool(params.get("grayfill", True)),
            sampling="r" if _is_empty_scene_dispatch_placeholder(params.get("sampling")) else str(params.get("sampling", "r")),
            illuminant=None if _is_empty_scene_dispatch_placeholder(params.get("illuminant")) else params.get("illuminant"),
            asset_store=store,
        )
        return track_session_object(session, scene)

    if name == "vernier":
        if args and isinstance(args[0], Scene):
            base_scene = args[0]
            vernier_type = str(args[1]) if len(args) > 1 else "display"
            params = args[2] if len(args) > 2 and hasattr(args[2], "items") else {}
            return track_session_object(session, scene_vernier(base_scene, vernier_type, params, asset_store=store))
        if args and isinstance(args[0], str):
            vernier_type = str(args[0])
            params = args[1] if len(args) > 1 and hasattr(args[1], "items") else {}
            return track_session_object(
                session,
                scene_vernier(scene_create("empty", asset_store=store), vernier_type, params, asset_store=store),
            )

        size = 64 if len(args) > 0 and _is_empty_scene_dispatch_placeholder(args[0]) else (args[0] if len(args) > 0 else 65)
        width = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 0) if len(args) > 1 else 3
        offset = _scene_dispatch_int_arg(args[2] if len(args) > 2 else None, 1) if len(args) > 2 else 3
        line_reflectance = _scene_dispatch_float_arg(args[3] if len(args) > 3 else None, 0.6) if len(args) > 3 else 0.6
        back_reflectance = _scene_dispatch_float_arg(args[4] if len(args) > 4 else None, 0.3) if len(args) > 4 else 0.3
        params = {
            "sceneSz": size,
            "barWidth": width,
            "offset": offset,
            "barReflect": line_reflectance,
            "bgReflect": back_reflectance,
        }
        return track_session_object(
            session,
            scene_vernier(scene_create("empty", asset_store=store), "object", params, asset_store=store),
        )

    if name in {"reflectancechart", "reflectance"}:
        if args and isinstance(args[0], dict):
            params = _reflectance_chart_parameters(args[0])
        elif args and isinstance(args[0], str):
            params = _reflectance_chart_parameters(_normalized_key_value_args(args))
        elif len(args) == 1 and hasattr(args[0], "items"):
            params = _reflectance_chart_parameters(args[0])
        else:
            params = _reflectance_chart_parameters(None)
            if len(args) > 0 and not _is_empty_scene_dispatch_placeholder(args[0]):
                params["psize"] = max(int(np.rint(args[0])), 1)
            if len(args) > 1 and not _is_empty_scene_dispatch_placeholder(args[1]):
                params["ssamples"] = args[1]
            if len(args) > 2 and not _is_empty_scene_dispatch_placeholder(args[2]):
                params["sfiles"] = _reflectance_chart_sources(args[2])
            if len(args) > 3 and not _is_empty_scene_dispatch_placeholder(args[3]):
                params["wave"] = _wave_or_default(args[3])
            if len(args) > 4 and not _is_empty_scene_dispatch_placeholder(args[4]):
                params["grayflag"] = bool(args[4])
            if len(args) > 5 and not _is_empty_scene_dispatch_placeholder(args[5]):
                params["sampling"] = str(args[5])
        return track_session_object(
            session,
            _reflectance_chart_scene(
                params["sfiles"],
                params["ssamples"],
                params["psize"],
                params["wave"],
                params["grayflag"],
                params["sampling"],
                asset_store=store,
            ),
        )

    if name in {"line", "lined65", "impulse1dd65"}:
        size = args[0] if len(args) > 0 else 64
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        return track_session_object(session, _line_scene("d65", size, 0, wave, asset_store=store))

    if name in {"lineee", "impulse1dee"}:
        size = args[0] if len(args) > 0 else 64
        offset = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 0)
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _line_scene("ee", size, offset, wave, asset_store=store))

    if name in {"lineequalphoton", "lineep"}:
        size = args[0] if len(args) > 0 else 64
        offset = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 0)
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _line_scene("ep", size, offset, wave, asset_store=store))

    if name in {"frequencyorientation", "demosaictarget", "freqorientpattern", "freqorient"}:
        params = _frequency_orientation_parameters(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        image, params = _frequency_orientation_image(params)
        image = np.clip(image, 1e-4, 1.0)
        image = image / max(float(np.max(image)), 1e-12)
        scene = _equal_photon_pattern_scene("FOTarget", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store)
        scene.fields["frequency_orientation_params"] = params
        return track_session_object(session, scene)

    if name == "moireorient":
        if args and (hasattr(args[0], "items") or hasattr(args[0], "__dict__")):
            params = args[0]
        elif args:
            params = {}
            if len(args) > 0 and not _is_empty_scene_dispatch_placeholder(args[0]):
                params["sceneSize"] = args[0]
            if len(args) > 1 and not _is_empty_scene_dispatch_placeholder(args[1]):
                params["f"] = args[1]
        else:
            params = None
        return track_session_object(session, _mo_target_scene(params, asset_store=store))

    if name in {"harmonic", "sinusoid"}:
        params = _harmonic_parameters(args[0] if len(args) > 0 else None)
        wave = _wave_or_default(args[1] if len(args) > 1 else None)
        image, params = _image_harmonic(params)
        image = np.where(image == 0.0, 1e-4, image)
        image = image / (2.0 * max(float(np.max(image)), 1e-12))
        scene = _equal_photon_pattern_scene("harmonic", image, wave, fov_deg=1.0, asset_store=store)
        scene.fields["harmonic_params"] = params
        return track_session_object(session, scene)

    if name in {"sweep", "sweepfrequency"}:
        size = 128 if len(args) == 0 or _is_empty_scene_dispatch_placeholder(args[0]) else args[0]
        if np.isscalar(size):
            default_max_frequency = float(size) / 16.0
        else:
            size_vec = np.asarray(size, dtype=float).reshape(-1)
            default_max_frequency = float(size_vec[1] if size_vec.size > 1 else size_vec[0]) / 16.0
        max_frequency = (
            default_max_frequency
            if len(args) <= 1 or _is_empty_scene_dispatch_placeholder(args[1])
            else float(args[1])
        )
        wave_arg = None if len(args) <= 2 or _is_empty_scene_dispatch_placeholder(args[2]) else args[2]
        wave = _wave_or_default(wave_arg)
        y_contrast = None if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else args[3]
        image = _sweep_frequency_image(size, max_frequency, y_contrast)
        scene = _equal_photon_pattern_scene("sweep", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store)
        scene.fields["sweep_params"] = {
            "size": _scene_size_2d(size, default=128),
            "max_frequency": max_frequency,
            "wave": wave.copy(),
        }
        return track_session_object(session, scene)

    if name in {"ramp", "rampequalphoton"}:
        size = 128 if len(args) > 0 and _is_empty_scene_dispatch_placeholder(args[0]) else (args[0] if len(args) > 0 else 256)
        dynamic_range = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 256.0) if len(args) > 1 else 256.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        base_scene = Scene(name="ramp")
        base_scene.fields["wave"] = wave.copy()
        base_scene.fields["distance_m"] = DEFAULT_DISTANCE_M
        base_scene.fields["fov_deg"] = DEFAULT_FOV_DEG
        return track_session_object(session, scene_ramp(base_scene, size, dynamic_range, asset_store=store))

    if name in {"linearintensityramp", "linearramp"}:
        size = 128 if len(args) > 0 and _is_empty_scene_dispatch_placeholder(args[0]) else (args[0] if len(args) > 0 else 256)
        dynamic_range = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 256.0) if len(args) > 1 else 256.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        image = _linear_intensity_ramp_image(size, dynamic_range)
        return track_session_object(
            session,
            _equal_photon_pattern_scene("linearRamp", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store),
        )

    if name in {"exponentialintensityramp", "expintensityramp", "expramp"}:
        size = 128 if len(args) > 0 and _is_empty_scene_dispatch_placeholder(args[0]) else (args[0] if len(args) > 0 else 256)
        dynamic_range = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 256.0) if len(args) > 1 else 256.0
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        image = _exponential_intensity_ramp_image(size, dynamic_range)
        return track_session_object(
            session,
            _equal_photon_pattern_scene("expRamp", image, wave, fov_deg=DEFAULT_FOV_DEG, asset_store=store),
        )

    if name in {"starpattern", "radiallines"}:
        image_size = _scene_dispatch_int_arg(args[0] if len(args) > 0 else None, 256)
        spectral_type = _scene_dispatch_text_arg(args[1] if len(args) > 1 else None, "ep")
        n_lines = _scene_dispatch_int_arg(args[2] if len(args) > 2 else None, 8)
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        return track_session_object(
            session,
            _star_pattern_scene(image_size, spectral_type, n_lines, wave, asset_store=store),
        )

    if name == "deadleaves":
        size = _scene_dispatch_int_arg(args[0] if len(args) > 0 else None, 256)
        sigma = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 2.0)
        options = args[2] if len(args) > 2 and hasattr(args[2], "items") else None
        if len(args) > 2 and _is_empty_scene_dispatch_placeholder(args[2]):
            wave_arg = args[3] if len(args) > 3 else None
        else:
            wave_arg = args[3] if len(args) > 3 else (args[2] if len(args) > 2 and not hasattr(args[2], "items") else None)
        wave = None if wave_arg is None else _wave_or_default(wave_arg)
        return track_session_object(session, _dead_leaves_scene(size, sigma, options, wave, asset_store=store))

    if name == "zoneplate":
        size = args[0] if len(args) > 0 else 384
        second_arg = args[1] if len(args) > 1 else None
        if _looks_like_wave_arg(second_arg):
            fov_deg = 4.0
            wave_arg = second_arg
        else:
            if second_arg is None or (isinstance(second_arg, (list, tuple, np.ndarray)) and np.asarray(second_arg).size == 0):
                fov_source = None
            else:
                fov_source = second_arg
            fov_deg = 4.0 if fov_source is None else float(np.asarray(fov_source, dtype=float).reshape(-1)[0])
            wave_arg = args[2] if len(args) > 2 else None
        wave = _wave_or_default(wave_arg)
        return track_session_object(session, _zone_plate_scene(size, wave, fov_deg=fov_deg, asset_store=store))

    if name == "bar":
        size = args[0] if len(args) > 0 else 64
        width = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 5) if len(args) > 1 else 3
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _bar_scene(size, width, wave, "ep", asset_store=store))

    if name == "baree":
        size = args[0] if len(args) > 0 else 64
        width = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 5) if len(args) > 1 else 3
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _bar_scene(size, width, wave, "ee", asset_store=store))

    if name in {"whitenoise", "noise"}:
        size = args[0] if len(args) > 0 else 128
        contrast = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 20.0)
        wave = _wave_or_default(args[2] if len(args) > 2 else None)
        return track_session_object(session, _white_noise_scene(size, contrast, wave, asset_store=store))

    if name in {"pointarray", "manypoints", "point array".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        spacing = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 16)
        spectral_type = (
            "d65"
            if len(args) > 2 and _is_empty_scene_dispatch_placeholder(args[2])
            else _scene_dispatch_text_arg(args[2] if len(args) > 2 else None, "ep")
        )
        point_size = _scene_dispatch_int_arg(args[3] if len(args) > 3 else None, 1)
        wave = _wave_or_default(args[4] if len(args) > 4 else None)
        return track_session_object(
            session,
            _point_array_scene(size, spacing, spectral_type, point_size, wave, asset_store=store),
        )

    if name in {"diskarray", "disk array".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        radius = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 128)
        array_size = args[2] if len(args) > 2 else np.array([1, 1], dtype=int)
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        return track_session_object(
            session,
            _disk_array_scene(size, radius, array_size, wave, asset_store=store),
        )

    if name in {"squarearray", "square array".replace(" ", ""), "squares"}:
        size = args[0] if len(args) > 0 else 128
        square_size = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 16)
        array_size = args[2] if len(args) > 2 else np.array([1, 1], dtype=int)
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        return track_session_object(
            session,
            _square_array_scene(size, square_size, array_size, wave, asset_store=store),
        )

    if name in {"gridlines", "distortiongrid", "grid lines".replace(" ", "")}:
        size = args[0] if len(args) > 0 else 128
        spacing = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 16)
        spectral_type = _scene_dispatch_text_arg(args[2] if len(args) > 2 else None, "ep")
        thickness = _scene_dispatch_int_arg(args[3] if len(args) > 3 else None, 1)
        wave = _wave_or_default(args[4] if len(args) > 4 else None)
        return track_session_object(
            session,
            _grid_lines_scene(size, spacing, spectral_type, thickness, wave, asset_store=store),
        )

    if name == "checkerboard":
        pixels_per_check = _scene_dispatch_int_arg(args[0] if len(args) > 0 else None, 16)
        number_of_checks = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 8)
        spectral_type = str(args[2]) if len(args) > 2 and isinstance(args[2], str) else "ep"
        wave_arg = args[3] if len(args) > 3 else (args[2] if len(args) > 2 and not isinstance(args[2], str) else None)
        wave = _wave_or_default(wave_arg)
        return track_session_object(
            session,
            _checkerboard_scene(
                pixels_per_check,
                number_of_checks,
                wave,
                spectral_type,
                asset_store=store,
            ),
        )

    if name in {"slantedbar", "slantededge", "iso12233", "slanted bar".replace(" ", ""), "slanted edge".replace(" ", "")}:
        image_size = _scene_dispatch_int_arg(args[0] if len(args) > 0 else None, 384)
        edge_slope = _scene_dispatch_float_arg(args[1] if len(args) > 1 else None, 2.6)
        fov_arg = args[2] if len(args) > 2 else None
        if fov_arg is None or (isinstance(fov_arg, (list, tuple, np.ndarray)) and np.asarray(fov_arg).size == 0):
            fov_deg = 2.0
        else:
            fov_deg = float(fov_arg)
        wave = _wave_or_default(args[3] if len(args) > 3 else None)
        dark_level = _scene_dispatch_float_arg(args[4] if len(args) > 4 else None, 0.0)
        return track_session_object(
            session,
            _slanted_bar_scene(image_size, edge_slope, wave, fov_deg, dark_level, asset_store=store),
        )

    if name in {"letter", "font"}:
        from .fonts import font_create, scene_from_font

        if args and isinstance(args[0], str):
            letter = str(args[0])
            font_size = _scene_dispatch_int_arg(args[1] if len(args) > 1 else None, 14)
            font_name = _scene_dispatch_text_arg(args[2] if len(args) > 2 else None, "Georgia")
            display = None if len(args) <= 3 or _is_empty_scene_dispatch_placeholder(args[3]) else args[3]
            font = font_create(letter, font_name, font_size, asset_store=store)
        else:
            font = font_create(asset_store=store) if len(args) == 0 or _is_empty_scene_dispatch_placeholder(args[0]) else args[0]
            display = None if len(args) <= 1 or _is_empty_scene_dispatch_placeholder(args[1]) else args[1]
        return track_session_object(session, scene_from_font(font, display, asset_store=store))

    raise UnsupportedOptionError("sceneCreate", scene_name)


def scene_from_file(
    input_data: Any,
    im_type: str,
    mean_luminance: float | None = None,
    display: Any = None,
    wave: np.ndarray | None = None,
    illuminant_energy: Any | None = None,
    scale_reflectance: bool = True,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Scene:
    """Create a scene from RGB, monochrome, or multispectral image data."""

    store = _store(asset_store)
    normalized_type = param_format(im_type)
    if normalized_type not in {"rgb", "monochrome", "unispectral", "spectral", "multispectral", "hyperspectral"}:
        raise UnsupportedOptionError("sceneFromFile", im_type)

    requested_wave = None if wave is None else _wave_or_default(wave)

    if normalized_type in {"spectral", "multispectral", "hyperspectral"}:
        multispectral = _multispectral_scene_input(input_data, requested_wave, asset_store=store)
        scene = _scene_shell(str(multispectral["source_name"]), np.asarray(multispectral["wave"], dtype=float), asset_store=store)
        scene.fields["illuminant_format"] = str(multispectral["illuminant_format"])
        scene.fields["illuminant_energy"] = np.asarray(multispectral["illuminant_energy"], dtype=float)
        scene.fields["illuminant_photons"] = np.asarray(multispectral["illuminant_photons"], dtype=float)
        scene.fields["illuminant_comment"] = str(multispectral["illuminant_comment"])
        scene.fields["distance_m"] = float(multispectral["distance_m"])
        scene.fields["fov_deg"] = float(multispectral["fov_deg"])
        scene.fields["filename"] = str(multispectral["filename"])
        scene.fields["source_type"] = "multispectral"
        scene.data["photons"] = np.asarray(multispectral["photons"], dtype=float)
        _update_scene_geometry(scene)
        if mean_luminance is not None:
            scene = scene_adjust_luminance(scene, float(mean_luminance), asset_store=store)
        return track_session_object(session, scene)

    del scale_reflectance
    current_display = _scene_display(display, requested_wave, asset_store=store)

    from .display import display_get

    if not bool(display_get(current_display, "is emissive")):
        raise UnsupportedOptionError("sceneFromFile", "reflective display")

    wave_nm = np.asarray(display_get(current_display, "wave"), dtype=float).reshape(-1)
    image, filename, source_name = _scene_image_input(input_data, asset_store=store)
    spd = np.asarray(display_get(current_display, "spd"), dtype=float)
    n_primaries = spd.shape[1]
    prepared = _prepare_display_image(image, normalized_type, n_primaries)
    linear_rgb = _display_linear_rgb(prepared, np.asarray(display_get(current_display, "gamma table"), dtype=float))
    energy = linear_rgb.reshape(-1, n_primaries) @ spd.T
    photons = energy_to_quanta(energy, wave_nm).reshape(prepared.shape[0], prepared.shape[1], wave_nm.size)

    scene = _scene_shell(f"{source_name} - {current_display.name}", wave_nm, asset_store=store)
    if illuminant_energy is None:
        source_illuminant_energy = np.sum(spd, axis=1)
        illuminant_comment = current_display.name
    else:
        source_illuminant_energy, illuminant_comment = _resolve_illuminant_input(
            illuminant_energy,
            wave_nm,
            asset_store=store,
        )
    scene.fields["illuminant_energy"] = np.asarray(source_illuminant_energy, dtype=float).reshape(-1)
    scene.fields["illuminant_photons"] = energy_to_quanta(scene.fields["illuminant_energy"], wave_nm)
    scene.fields["illuminant_comment"] = str(illuminant_comment)
    scene.fields["distance_m"] = float(display_get(current_display, "viewing distance"))
    scene.fields["fov_deg"] = float(prepared.shape[1]) * float(display_get(current_display, "deg per dot"))
    scene.fields["filename"] = filename
    scene.fields["source_type"] = normalized_type
    scene.fields["display_name"] = current_display.name
    scene.data["photons"] = photons
    _update_scene_geometry(scene)

    if mean_luminance is not None:
        scene = scene_adjust_luminance(scene, float(mean_luminance), asset_store=store)
    return track_session_object(session, scene)


def _scene_ddf_depth_map(file_name: Any) -> np.ndarray | None:
    path = Path(file_name).expanduser()
    info = exiftool_info(path, format="json")
    depth_payload = _exiftool_depth_payload(path)
    if info is None or depth_payload is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            handle.write(depth_payload)
            handle.flush()
            depth_map = np.asarray(iio.imread(handle.name), dtype=np.float32)
    except Exception:
        return None

    if depth_map.ndim > 2:
        depth_map = depth_map[:, :, 0]
    if depth_map.size == 0:
        return None

    depth_max = float(np.max(depth_map))
    if depth_max <= 0.0:
        return None

    try:
        units = str(info.get("DepthMapUnits", ""))
        near_value = float(info["DepthMapNear"])
        far_value = float(info["DepthMapFar"])
        if units == "Meters":
            depth_map = near_value + (depth_map / depth_max) * (far_value - near_value)
        elif units == "Diopters":
            far_diopters = far_value
            near_diopters = near_value
            d_normal = depth_map / depth_max
            denominator = far_diopters - d_normal * (far_diopters - near_diopters)
            depth_map = (far_diopters * near_diopters) / np.maximum(denominator, 1.0e-12)

        image_height = int(info.get("ImageHeight", depth_map.shape[0]))
        image_width = int(info.get("ImageWidth", depth_map.shape[1]))
        if depth_map.shape != (image_height, image_width):
            depth_map = zoom(
                depth_map,
                (
                    image_height / max(depth_map.shape[0], 1),
                    image_width / max(depth_map.shape[1], 1),
                ),
                order=1,
            )
    except Exception:
        return None

    return np.asarray(depth_map, dtype=float)


def exiftool_info(fname: Any, *args: Any, format: str = "text") -> Any:
    path = Path(fname).expanduser()
    exiftool = shutil.which("exiftool")
    if exiftool is None or not path.is_file():
        return None

    normalized = param_format(format)
    if args:
        for index in range(0, len(args), 2):
            key = param_format(args[index])
            if key == "format" and index + 1 < len(args):
                normalized = param_format(args[index + 1])

    try:
        if normalized == "json":
            result = subprocess.run(
                [exiftool, "-j", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            info = json.loads(result.stdout)
            if not info:
                return None
            return info[0]

        option = "-v"
        result = subprocess.run(
            [exiftool, option, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None


def _exiftool_depth_payload(path: Path) -> bytes | None:
    exiftool = shutil.which("exiftool")
    if exiftool is None or not path.is_file():
        return None

    try:
        trailer = subprocess.run(
            [exiftool, "-b", "-trailer", str(path)],
            check=False,
            capture_output=True,
        )
        if trailer.returncode != 0 or not trailer.stdout:
            return None
        depth_payload = subprocess.run(
            [exiftool, "-", "-b", "-trailer"],
            check=False,
            input=trailer.stdout,
            capture_output=True,
        )
        if depth_payload.returncode != 0 or not depth_payload.stdout:
            return None
        return bytes(depth_payload.stdout)
    except Exception:
        return None


def exiftool_depth_from_file(f_name: Any, *args: Any, type: str = "GooglePixel") -> np.ndarray | None:
    normalized_type = param_format(type)
    if args:
        for index in range(0, len(args), 2):
            key = param_format(args[index])
            if key == "type" and index + 1 < len(args):
                normalized_type = param_format(args[index + 1])
    if normalized_type not in {"googlepixel", "pixel", "google"}:
        return None
    return _scene_ddf_depth_map(f_name)


def scene_from_ddf_file(
    f_name: Any,
    im_type: str = "rgb",
    mean_luminance: float | None = None,
    disp_cal: Any = None,
    w_list: Any | None = None,
    *scene_from_file_args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Scene:
    """Read a Dynamic Depth Format image using the legacy MATLAB wrapper contract."""

    scene = scene_from_file(
        f_name,
        im_type,
        mean_luminance,
        disp_cal,
        w_list,
        *scene_from_file_args,
        asset_store=_store(asset_store),
        session=session,
    )

    depth_map = _scene_ddf_depth_map(scene.fields.get("filename", f_name))
    if depth_map is None:
        return scene

    scene_size = np.asarray(scene_get(scene, "size"), dtype=int).reshape(-1)
    if depth_map.shape != tuple(scene_size):
        depth_map = zoom(
            depth_map,
            (
                scene_size[0] / max(depth_map.shape[0], 1),
                scene_size[1] / max(depth_map.shape[1], 1),
            ),
            order=1,
        )
    return scene_set(scene, "depth map", np.asarray(depth_map, dtype=float))


def scene_sdr(
    deposit_name: str,
    scene_name: str,
    *,
    download_dir: str | Path | None = None,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Scene | np.ndarray:
    """Download or reuse a cached Stanford Digital Repository scene/image."""

    deposit_key = param_format(deposit_name)
    if deposit_key not in _SCENE_SDR_DEPOSIT_URLS:
        valid = ", ".join(sorted(_SCENE_SDR_DEPOSIT_URLS))
        raise ValueError(f"Invalid deposit name. Please choose from: {valid}")

    scene_path = Path(scene_name)
    suffix = scene_path.suffix.lower() or ".mat"
    local_name = f"{scene_path.stem}{suffix}"
    cache_dir = Path(download_dir).expanduser() if download_dir is not None else _SCENE_SDR_DEFAULT_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / local_name

    if not local_file.exists():
        remote_url = f"{_SCENE_SDR_DEPOSIT_URLS[deposit_key].rstrip('/')}/{quote(local_name)}"
        request = Request(remote_url, headers={"User-Agent": "pyisetcam/0.1.0"})
        with urlopen(request) as response, local_file.open("wb") as handle:
            handle.write(response.read())

    if suffix == ".mat":
        return scene_from_file(local_file, "multispectral", asset_store=_store(asset_store), session=session)
    if suffix == ".png":
        return np.asarray(iio.imread(local_file))
    raise ValueError(f"Unknown file extension '{suffix}'.")


def scene_from_basis(
    scene_s: Any,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Scene:
    if isinstance(scene_s, (str, Path)):
        return scene_from_file(scene_s, "multispectral", asset_store=_store(asset_store), session=session)

    mc_coef = _scene_struct_value(scene_s, "mcCOEF")
    basis_struct = _scene_struct_value(scene_s, "basis")
    illuminant = _scene_struct_value(scene_s, "illuminant")
    if mc_coef is None:
        raise ValueError("sceneFromBasis requires mcCOEF.")
    if basis_struct is None:
        raise ValueError("sceneFromBasis requires basis.")

    basis_wave = np.asarray(_scene_struct_value(basis_struct, "wave"), dtype=float).reshape(-1)
    basis_matrix = np.asarray(_scene_struct_value(basis_struct, "basis"), dtype=float)
    if basis_matrix.ndim == 1:
        basis_matrix = basis_matrix.reshape(-1, 1)
    if basis_matrix.shape[0] != basis_wave.size and basis_matrix.shape[1] == basis_wave.size:
        basis_matrix = basis_matrix.T
    if basis_matrix.shape[0] != basis_wave.size:
        raise ValueError("Basis wavelength samples must align with the basis matrix.")

    coefficients = np.asarray(mc_coef, dtype=float)
    if coefficients.ndim == 2:
        coefficients = coefficients[:, :, np.newaxis]
    photons = np.tensordot(coefficients, basis_matrix.T, axes=([2], [0]))

    img_mean = _scene_struct_value(scene_s, "imgMean")
    if img_mean is not None:
        mean_vector = np.asarray(img_mean, dtype=float).reshape(-1)
        if mean_vector.size not in {0, basis_wave.size}:
            raise ValueError("imgMean wavelength length does not match basis wavelength samples.")
        if mean_vector.size:
            photons = photons + mean_vector.reshape(1, 1, -1)

    photons = np.maximum(np.asarray(photons, dtype=float), 0.0)
    scene = Scene(name=str(_scene_struct_value(scene_s, "name", "scene")))
    scene.fields["wave"] = basis_wave
    scene.fields["illuminant_format"] = "spectral"
    scene.fields["distance_m"] = float(np.asarray(_scene_struct_value(scene_s, "dist", DEFAULT_DISTANCE_M), dtype=float).reshape(-1)[0])
    scene.fields["fov_deg"] = float(np.asarray(_scene_struct_value(scene_s, "fov", DEFAULT_FOV_DEG), dtype=float).reshape(-1)[0])
    scene.data["photons"] = photons

    if illuminant is not None:
        illuminant_wave = np.asarray(
            _scene_struct_value(illuminant, "wave", _scene_struct_value(illuminant, "wavelength", basis_wave)),
            dtype=float,
        ).reshape(-1)
        illuminant_energy = _scene_struct_value(illuminant, "energy", _scene_struct_value(illuminant, "data"))
        illuminant_photons = _scene_struct_value(illuminant, "photons")
        if illuminant_photons is not None:
            illuminant_photons = np.asarray(illuminant_photons, dtype=float)
            if illuminant_photons.ndim == 3 and illuminant_photons.shape[0] == illuminant_wave.size and illuminant_photons.shape[-1] != illuminant_wave.size:
                illuminant_photons = np.moveaxis(illuminant_photons, 0, -1)
            if not np.array_equal(illuminant_wave, basis_wave):
                if illuminant_photons.ndim == 1:
                    illuminant_photons = interp_spectra(illuminant_wave, illuminant_photons, basis_wave).reshape(-1)
                else:
                    illuminant_photons = _resample_wave_last(illuminant_photons, illuminant_wave, basis_wave)
            scene.fields["illuminant_photons"] = illuminant_photons
            scene.fields["illuminant_energy"] = quanta_to_energy(illuminant_photons, basis_wave)
            if np.asarray(scene.fields["illuminant_energy"]).ndim == 3:
                scene.fields["illuminant_energy"] = np.mean(np.asarray(scene.fields["illuminant_energy"], dtype=float), axis=(0, 1))
            scene.fields["illuminant_format"] = "spatial spectral" if np.asarray(illuminant_photons).ndim == 3 else "spectral"
        elif illuminant_energy is not None:
            illuminant_energy = np.asarray(illuminant_energy, dtype=float)
            if illuminant_energy.ndim == 3 and illuminant_energy.shape[0] == illuminant_wave.size and illuminant_energy.shape[-1] != illuminant_wave.size:
                illuminant_energy = np.moveaxis(illuminant_energy, 0, -1)
            if not np.array_equal(illuminant_wave, basis_wave):
                if illuminant_energy.ndim == 1:
                    illuminant_energy = interp_spectra(illuminant_wave, illuminant_energy, basis_wave).reshape(-1)
                else:
                    illuminant_energy = _resample_wave_last(illuminant_energy, illuminant_wave, basis_wave)
            scene.fields["illuminant_energy"] = illuminant_energy
            scene.fields["illuminant_photons"] = energy_to_quanta(illuminant_energy, basis_wave)
            if np.asarray(illuminant_energy).ndim == 3:
                scene.fields["illuminant_energy"] = np.mean(np.asarray(illuminant_energy, dtype=float), axis=(0, 1))
            scene.fields["illuminant_format"] = "spatial spectral" if np.asarray(illuminant_energy).ndim == 3 else "spectral"
        scene.fields["illuminant_comment"] = str(_scene_struct_value(illuminant, "comment", scene.name))
    else:
        scene.fields["illuminant_photons"] = np.maximum(np.mean(photons, axis=(0, 1), dtype=float), 1.0e-12)
        scene.fields["illuminant_energy"] = quanta_to_energy(scene.fields["illuminant_photons"], basis_wave)
        scene.fields["illuminant_comment"] = scene.name

    _update_scene_geometry(scene)
    return track_session_object(session, scene)


def scene_insert(scene1: Scene, scene2: Scene, position: Any) -> Scene:
    wave1 = np.asarray(scene_get(scene1, "wave"), dtype=float).reshape(-1)
    wave2 = np.asarray(scene_get(scene2, "wave"), dtype=float).reshape(-1)
    if not np.array_equal(wave1, wave2):
        raise ValueError("sceneInsert requires matching wavelength samples.")

    anchor = np.asarray(position, dtype=int).reshape(-1)
    if anchor.size != 2:
        raise ValueError("sceneInsert position must be a 2-vector.")

    base = np.asarray(scene_get(scene1, "photons"), dtype=float).copy()
    inset = np.asarray(scene_get(scene2, "photons"), dtype=float)
    row0 = int(anchor[0]) - 1
    col0 = int(anchor[1]) - 1
    row1 = row0 + int(inset.shape[0])
    col1 = col0 + int(inset.shape[1])
    if row0 < 0 or col0 < 0 or row1 > base.shape[0] or col1 > base.shape[1]:
        raise ValueError("sceneInsert position places the inserted scene outside the base scene.")

    base[row0:row1, col0:col1, ...] = inset
    return scene_set(scene1.clone(), "photons", base)


def scene_to_file(
    fname: str | Path,
    scene: Scene,
    b_type: Any | None = None,
    m_type: str = "mean svd",
    comment: str | None = None,
) -> tuple[float, int]:
    path = Path(fname).expanduser()
    if path.suffix.lower() != ".mat":
        path = path.with_suffix(".mat")
    path.parent.mkdir(parents=True, exist_ok=True)

    if comment is None:
        comment = f"Scene: {scene_get(scene, 'name')}"

    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    illuminant = scene_get(scene, "illuminant")
    fov = float(scene_get(scene, "fov"))
    dist = float(scene_get(scene, "distance"))
    name = str(scene_get(scene, "name"))

    b_type_array = np.asarray(b_type, dtype=float).reshape(-1) if b_type is not None else np.array([], dtype=float)
    if b_type is None or b_type_array.size == 0:
        savemat(
            path,
            {
                "photons": photons,
                "wave": wave,
                "comment": str(comment),
                "illuminant": {
                    "wave": np.asarray(illuminant["wave"], dtype=float).reshape(-1),
                    "data": np.asarray(illuminant["energy"], dtype=float),
                },
                "fov": fov,
                "dist": dist,
                "name": name,
                "type": scene.type,
                "data": dict(scene.data),
            },
            do_compression=True,
        )
        return 1.0, int(wave.size)

    from .fileio import ie_save_multispectral_image

    sampled = photons[::3, ::3, :]
    img_mean, basis_data, _, var_explained = hc_basis(sampled, float(b_type_array[0]), m_type)
    photons_xw, rows, cols, _ = rgb_to_xw_format(photons)
    normalized_mean_type = param_format(m_type)
    if normalized_mean_type == "canonical":
        coefficients_xw = photons_xw @ basis_data
        saved_mean = None
    elif normalized_mean_type == "meansvd":
        coefficients_xw = (photons_xw - img_mean.reshape(1, -1)) @ basis_data
        saved_mean = img_mean
    else:
        raise ValueError(f"Unknown sceneToFile mean type '{m_type}'.")

    coefficients = xw_to_rgb_format(coefficients_xw, rows, cols)
    ie_save_multispectral_image(
        path,
        coefficients,
        {"basis": basis_data, "wave": wave},
        comment,
        saved_mean,
        illuminant,
        fov,
        dist,
        name,
    )
    return float(var_explained), int(coefficients.shape[2])


def scene_wb_create(scene_all: Scene, work_dir: str | Path | None = None) -> str:
    from .fileio import vc_save_object

    output_dir = Path.cwd() / str(scene_get(scene_all, "name")) if work_dir is None else Path(work_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    wave = np.asarray(scene_get(scene_all, "wave"), dtype=float).reshape(-1)
    photons = np.asarray(scene_get(scene_all, "photons"), dtype=float)
    illuminant_photons = np.asarray(scene_get(scene_all, "illuminant photons"), dtype=float)
    illuminant_energy = np.asarray(scene_get(scene_all, "illuminant energy"), dtype=float)

    for index, wavelength in enumerate(wave):
        scene_band = scene_all.clone()
        scene_band.fields["wave"] = np.array([float(wavelength)], dtype=float)
        scene_band.data["photons"] = np.asarray(photons[:, :, index : index + 1], dtype=float)
        if illuminant_photons.ndim == 1:
            scene_band.fields["illuminant_photons"] = np.array([float(illuminant_photons[index])], dtype=float)
        else:
            scene_band.fields["illuminant_photons"] = np.asarray(illuminant_photons[:, :, index : index + 1], dtype=float)
        if illuminant_energy.ndim == 1:
            scene_band.fields["illuminant_energy"] = np.array([float(illuminant_energy[index])], dtype=float)
        else:
            scene_band.fields["illuminant_energy"] = np.asarray(illuminant_energy[:, :, index : index + 1], dtype=float)
        scene_band.fields["illuminant_format"] = "spatial spectral" if np.asarray(scene_band.fields["illuminant_photons"]).ndim == 3 else "spectral"
        _invalidate_scene_caches(scene_band)
        _update_scene_geometry(scene_band)
        vc_save_object(scene_band, output_dir / f"scene{int(np.rint(wavelength))}.mat")

    return str(output_dir)


def scene_make_video(
    scene_list: list[Scene] | tuple[Scene, ...],
    full_name: str | Path | None = None,
    render_flag: int = -3,
    gam: float = 2.2,
    fps: float = 3.0,
    *,
    asset_store: AssetStore | None = None,
) -> str:
    scenes = list(scene_list)
    if not scenes:
        raise ValueError("sceneMakeVideo requires at least one scene.")

    path = Path(full_name) if full_name is not None else (Path.cwd() / "Video-output.gif")
    if path.suffix == "":
        path = path.with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    store = _store(asset_store)
    frames = []
    for scene in scenes:
        rgb = np.asarray(scene_show_image(scene, render_flag, gam, asset_store=store), dtype=float)
        frames.append(np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8))

    duration = 1.0 / max(float(fps), 1.0e-12)
    frame_stack = np.stack(frames, axis=0)
    if path.suffix.lower() == ".gif":
        iio.imwrite(path, frame_stack, duration=duration, loop=0)
    else:
        iio.imwrite(path, frame_stack, fps=float(fps))
    return str(path)


def scene_calculate_luminance(scene: Scene, *, asset_store: AssetStore | None = None) -> np.ndarray:
    store = _store(asset_store)
    luminance = luminance_from_photons(scene.data["photons"], np.asarray(scene.fields["wave"], dtype=float), asset_store=store)
    scene.fields["luminance"] = luminance
    scene.fields["mean_luminance"] = float(np.mean(luminance))
    return luminance


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.copy()
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=float)
    return (array - minimum) / (maximum - minimum)


def _hist_equalize_global(values: np.ndarray, bins: int = 255) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.reshape(-1)
    if flat.size == 0:
        return array.copy()
    normalized = _minmax_normalize(flat)
    if float(np.max(normalized)) <= 0.0:
        return np.zeros_like(array, dtype=float)
    hist, edges = np.histogram(normalized, bins=int(bins), range=(0.0, 1.0))
    cdf = np.cumsum(hist, dtype=float)
    if cdf[-1] <= 0.0:
        return np.zeros_like(array, dtype=float)
    cdf /= cdf[-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    equalized = np.interp(normalized, centers, cdf, left=0.0, right=1.0)
    return equalized.reshape(array.shape)


def _pad_reflect(image: np.ndarray, xsize: int, ysize: int | None = None) -> np.ndarray:
    if ysize is None:
        ysize = xsize
    xsize = int(xsize)
    ysize = int(ysize)
    if xsize <= 0 and ysize <= 0:
        return np.asarray(image, dtype=float).copy()
    return np.pad(np.asarray(image, dtype=float), ((ysize, ysize), (xsize, xsize)), mode="reflect")


def _pad_reflect_neg(image: np.ndarray, xsize: int, ysize: int | None = None, hor_neg: int = 0, ver_neg: int = 0) -> np.ndarray:
    if ysize is None:
        ysize = xsize
    image = np.asarray(image, dtype=float)
    xsize = int(xsize)
    ysize = int(ysize)
    if xsize <= 0 and ysize <= 0:
        return image.copy()

    top = image[1 : ysize + 1, :][::-1, :]
    bottom = image[-ysize - 1 : -1, :][::-1, :]
    left = image[:, 1 : xsize + 1][:, ::-1]
    right = image[:, -xsize - 1 : -1][:, ::-1]
    top_left = image[1 : ysize + 1, 1 : xsize + 1][::-1, ::-1]
    top_right = image[1 : ysize + 1, -xsize - 1 : -1][::-1, ::-1]
    bottom_left = image[-ysize - 1 : -1, 1 : xsize + 1][::-1, ::-1]
    bottom_right = image[-ysize - 1 : -1, -xsize - 1 : -1][::-1, ::-1]

    if hor_neg == 1 and ver_neg == 0:
        signs = ((-1, 1, -1), (-1, 1, -1), (-1, 1, -1))
    elif hor_neg == 0 and ver_neg == 1:
        signs = ((-1, -1, -1), (1, 1, 1), (-1, -1, -1))
    elif hor_neg == 1 and ver_neg == 1:
        signs = ((1, -1, 1), (-1, 1, -1), (1, -1, 1))
    else:
        signs = ((1, 1, 1), (1, 1, 1), (1, 1, 1))

    return np.block(
        [
            [signs[0][0] * top_left, signs[0][1] * top, signs[0][2] * top_right],
            [signs[1][0] * left, signs[1][1] * image, signs[1][2] * right],
            [signs[2][0] * bottom_left, signs[2][1] * bottom, signs[2][2] * bottom_right],
        ]
    )


def _gaussian_row(width: int, sigma: float) -> np.ndarray:
    center = (float(width) + 1.0) / 2.0
    coords = np.arange(1, width + 1, dtype=float)
    kernel = np.exp(-np.square(coords - center) / (2.0 * sigma * sigma))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.zeros(width, dtype=float)
    return kernel / kernel_sum


def _haar_pyramid(image: np.ndarray, nlevels: int) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    height, width = image.shape
    pyramid = np.zeros((height, width, 3 * nlevels + 1), dtype=float)
    lowpass_prev = image
    band = 0

    for level in range(1, nlevels + 1):
        extend_space = max(1, 2 ** (level - 2))
        step = 2 ** (level - 1)
        extended = _pad_reflect(lowpass_prev, extend_space)
        if level > 1:
            shift_1 = extended[0:height, 0:width]
            shift_2 = extended[0:height, step : width + step]
            shift_3 = extended[step : height + step, 0:width]
            shift_4 = extended[step : height + step, step : width + step]
        else:
            shift_1 = extended[1 : height + 1, 1 : width + 1]
            shift_2 = extended[1 : height + 1, 1 + step : width + 1 + step]
            shift_3 = extended[1 + step : height + 1 + step, 1 : width + 1]
            shift_4 = extended[1 + step : height + 1 + step, 1 + step : width + 1 + step]

        lowpass_prev = (shift_1 + shift_2 + shift_3 + shift_4) / 4.0
        pyramid[:, :, band] = (shift_1 + shift_2 - shift_3 - shift_4) / 4.0
        pyramid[:, :, band + 1] = (shift_1 - shift_2 + shift_3 - shift_4) / 4.0
        pyramid[:, :, band + 2] = (shift_1 - shift_2 - shift_3 + shift_4) / 4.0
        band += 3

    pyramid[:, :, band] = lowpass_prev
    return pyramid


def _recons_haar_pyramid(pyramid: np.ndarray) -> np.ndarray:
    pyramid = np.asarray(pyramid, dtype=float)
    height, width, nbands = pyramid.shape
    band_idx = nbands - 2
    lowpass_prev = pyramid[:, :, nbands - 1]
    nlevels = nbands // 3

    for level in range(nlevels, 0, -1):
        step = 2 ** (level - 1)
        extend_space = 1 if level == 1 else step // 2
        extend_low = _pad_reflect(lowpass_prev, extend_space)
        extend_3 = _pad_reflect_neg(pyramid[:, :, band_idx], extend_space, extend_space, 1, 1)
        band_idx -= 1
        extend_2 = _pad_reflect_neg(pyramid[:, :, band_idx], extend_space, extend_space, 1, 0)
        band_idx -= 1
        extend_1 = _pad_reflect_neg(pyramid[:, :, band_idx], extend_space, extend_space, 0, 1)
        band_idx -= 1

        row_1 = slice(0, height)
        col_1 = slice(0, width)
        col_2 = slice(step, width + step)
        row_3 = slice(step, height + step)

        lowpass = (
            extend_low[row_1, col_1]
            + extend_low[row_1, col_2]
            + extend_low[row_3, col_1]
            + extend_low[row_3, col_2]
        )
        band_1 = (
            -extend_1[row_1, col_1]
            - extend_1[row_1, col_2]
            + extend_1[row_3, col_1]
            + extend_1[row_3, col_2]
        )
        band_2 = (
            -extend_2[row_1, col_1]
            + extend_2[row_1, col_2]
            - extend_2[row_3, col_1]
            + extend_2[row_3, col_2]
        )
        band_3 = (
            extend_3[row_1, col_1]
            - extend_3[row_1, col_2]
            - extend_3[row_3, col_1]
            + extend_3[row_3, col_2]
        )
        lowpass_prev = (lowpass + band_1 + band_2 + band_3) / 4.0

    return lowpass_prev


def _range_compression_lum(
    image: np.ndarray,
    filt_type: str = "haar",
    beta: float = 0.6,
    alpha_a: float = 0.2,
    ifsharp: int = 0,
) -> np.ndarray:
    if param_format(filt_type) != "haar":
        raise UnsupportedOptionError("hdrRender currently supports only filt_type='haar'.")

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("range compression requires a 2-D luminance image.")
    if image.size == 0 or float(np.mean(image)) <= 0.0:
        return np.zeros_like(image, dtype=float)

    epsilon = 0.002
    log_image = np.log((image / float(np.mean(image))) + 1e-6)
    log_image = log_image - float(np.min(log_image))

    nlevels = int(np.floor(np.log2(float(min(log_image.shape))))) - 3
    if nlevels < 1:
        return np.exp(log_image)

    pyramid = _haar_pyramid(log_image, nlevels)
    _, _, pyr_layers = pyramid.shape
    lowpass = pyramid[:, :, pyr_layers - 1]

    width = 65
    gauss = _gaussian_row(width, width / 2.0)
    gauss2d = np.outer(gauss, gauss)
    extend = _pad_reflect(np.abs(lowpass), width // 2)
    lowpass_blur = convolve2d(extend, gauss2d, mode="valid")

    height, width_px = lowpass.shape
    band_blur_sum = np.zeros((height, width_px, nlevels), dtype=float)
    band_blur_sum_sum = np.zeros((height, width_px), dtype=float)
    width_init = 4
    filt_num = 3

    for level in range(1, nlevels + 1):
        kernel_width = width_init * (2 ** (level - 1)) + 1
        extend_space = int(round((kernel_width - 1) / 2))
        gauss = _gaussian_row(kernel_width, kernel_width / 2.0)
        gauss2d = np.outer(gauss, gauss)
        temp = np.zeros((height + 2 * extend_space, width_px + 2 * extend_space), dtype=float)
        for band in range(filt_num):
            band_idx = (level - 1) * filt_num + band
            temp += _pad_reflect(np.abs(pyramid[:, :, band_idx]), extend_space)
        band_blur = convolve2d(temp, gauss2d, mode="valid") / float(filt_num)
        band_blur_sum[:, :, level - 1] = band_blur
        band_blur_sum_sum += band_blur

    band_blur_sum_sum = (band_blur_sum_sum + lowpass_blur) / float(nlevels + 1)
    alpha = float(np.median(band_blur_sum_sum)) * float(alpha_a)
    gain = np.power(band_blur_sum_sum + epsilon, beta - 1.0) * (alpha ** (1.0 - beta))

    endrate = 0.4 if int(ifsharp) == 1 else 0.6
    for index in range(pyr_layers):
        level_here = (index // filt_num) + 1
        gain_amount = max(1.0 - (level_here - 1) * 0.15, endrate)
        pyramid[:, :, index] = pyramid[:, :, index] * gain * gain_amount

    result = np.exp(_recons_haar_pyramid(np.real(pyramid)))
    return np.asarray(result, dtype=float)


def hdr_render(
    image: np.ndarray,
    filt_type: str = "haar",
    s_sat: float = 0.7,
    bbeta: float = 0.6,
    aalpha: float = 0.2,
    ifsharp: int = 0,
) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        result = _range_compression_lum(np.squeeze(image), filt_type=filt_type, beta=bbeta, alpha_a=aalpha, ifsharp=ifsharp)
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("hdr_render expects a grayscale image or an RGB image.")
        luminance = np.max(image, axis=2)
        ratios = image / (luminance[:, :, np.newaxis] + 1e-9)
        result_luminance = _range_compression_lum(luminance, filt_type=filt_type, beta=bbeta, alpha_a=aalpha, ifsharp=ifsharp)
        result_luminance = _minmax_normalize(result_luminance)
        result = np.empty_like(image, dtype=float)
        result[:, :, 0] = result_luminance * np.power(ratios[:, :, 0], float(s_sat))
        result[:, :, 1] = result_luminance * np.power(ratios[:, :, 1], float(s_sat))
        result[:, :, 2] = result_luminance * np.power(ratios[:, :, 2], float(s_sat))

    low_end = float(np.percentile(result, 2.0))
    high_end = float(np.percentile(result, 99.0))
    if high_end > low_end:
        result = (result - low_end) / (high_end - low_end)
    else:
        result = np.zeros_like(result, dtype=float)
    result = np.clip(result, 0.0, 1.0)
    result = _minmax_normalize(np.asarray(result, dtype=float) + 0.15 * _hist_equalize_global(np.real(result)))
    return np.clip(result, 0.0, 1.0)


def modulate_flip_shift(lfilt: Any) -> np.ndarray:
    """Construct the QMF high-pass filter from a low-pass prototype."""

    filt = np.asarray(lfilt, dtype=float).reshape(-1)
    size = int(filt.size)
    center = int(np.ceil(size / 2.0))
    one_based = np.arange(size, 0, -1, dtype=int)
    return np.asarray(filt[::-1] * np.power(-1.0, one_based - center), dtype=float)


def qmf_pyramid(pic: np.ndarray, nlevels: int, qmf_length: int = 9) -> np.ndarray:
    """Build an oversampled QMF pyramid using the vendored MATLAB contract."""

    image = np.asarray(pic, dtype=float)
    if image.ndim != 2:
        raise ValueError("qmf_pyramid expects a 2-D image.")

    if qmf_length == 9:
        filt_low = np.asarray(
            [0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934, 0.41472545, -0.073386624, -0.060944743, 0.02807382],
            dtype=float,
        ) / np.sqrt(2.0)
    elif qmf_length == 13:
        filt_low = np.asarray(
            [
                -0.014556438,
                0.021651438,
                0.039045125,
                -0.09800052,
                -0.057827797,
                0.42995453,
                0.7737113,
                0.42995453,
                -0.057827797,
                -0.09800052,
                0.039045125,
                0.021651438,
                -0.014556438,
            ],
            dtype=float,
        ) / np.sqrt(2.0)
    else:
        raise ValueError("qmf_length can only be 9 or 13.")

    height, width = image.shape
    lowpass_prev = image
    pyramid = np.zeros((height, width, 3 * int(nlevels) + 1), dtype=float)
    band = 0
    qmf_halflen = int(np.floor(qmf_length / 2))
    plus_filt = filt_low[::2]
    minus_filt = filt_low[1::2]

    for level in range(1, int(nlevels) + 1):
        if level == 1:
            extend_space = qmf_halflen
            step = 1
        else:
            extend_space = qmf_halflen * (2 ** (level - 1))
            step = 2 ** (level - 1)

        extended = _pad_reflect(lowpass_prev, extend_space)
        ext_height, _ = extended.shape

        plus_row = np.zeros((ext_height, width), dtype=float)
        minus_row = np.zeros((ext_height, width), dtype=float)
        plus_col_low = np.zeros((height, width), dtype=float)
        minus_col_low = np.zeros((height, width), dtype=float)
        plus_col_high = np.zeros((height, width), dtype=float)
        minus_col_high = np.zeros((height, width), dtype=float)

        start = 0
        for coeff in plus_filt:
            plus_row += extended[:, start : start + width] * float(coeff)
            start += step * 2

        start = step
        for coeff in minus_filt:
            minus_row += extended[:, start : start + width] * float(coeff)
            start += step * 2

        lowpass_row = plus_row + minus_row
        hipass_row = plus_row - minus_row

        start = 0
        for coeff in plus_filt:
            plus_col_low += lowpass_row[start : start + height, :] * float(coeff)
            plus_col_high += hipass_row[start : start + height, :] * float(coeff)
            start += step * 2

        start = step
        for coeff in minus_filt:
            minus_col_low += lowpass_row[start : start + height, :] * float(coeff)
            minus_col_high += hipass_row[start : start + height, :] * float(coeff)
            start += step * 2

        lowpass_prev = plus_col_low + minus_col_low
        pyramid[:, :, band] = plus_col_low - minus_col_low
        pyramid[:, :, band + 1] = plus_col_high + minus_col_high
        pyramid[:, :, band + 2] = plus_col_high - minus_col_high
        band += 3

    pyramid[:, :, band] = lowpass_prev
    return pyramid


def recons_qmf_pyramid(pyr: np.ndarray, qmf_length: int = 9) -> np.ndarray:
    """Reconstruct an image from a non-decimated QMF pyramid."""

    pyramid = np.asarray(pyr, dtype=float)
    if pyramid.ndim != 3:
        raise ValueError("recons_qmf_pyramid expects a 3-D pyramid volume.")

    if qmf_length == 9:
        filt_low = np.asarray(
            [0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934, 0.41472545, -0.073386624, -0.060944743, 0.02807382],
            dtype=float,
        ) / np.sqrt(2.0)
    elif qmf_length == 13:
        filt_low = np.asarray(
            [
                -0.014556438,
                0.021651438,
                0.039045125,
                -0.09800052,
                -0.057827797,
                0.42995453,
                0.7737113,
                0.42995453,
                -0.057827797,
                -0.09800052,
                0.039045125,
                0.021651438,
                -0.014556438,
            ],
            dtype=float,
        ) / np.sqrt(2.0)
    else:
        raise ValueError("qmf_length can only be 9 or 13.")

    filt_high = modulate_flip_shift(filt_low)
    qmf_halflen = int(np.floor(qmf_length / 2))
    height, width, nbands = pyramid.shape
    lowpass_prev = pyramid[:, :, nbands - 1]
    band_idx = nbands - 2
    nlevels = int(np.floor(nbands / 3))

    for level in range(nlevels, 0, -1):
        if level == 1:
            extend_space = qmf_halflen
            step = 1
        else:
            extend_space = qmf_halflen * (2 ** (level - 1))
            step = 2 ** (level - 1)

        extend_lo_lo = _pad_reflect(lowpass_prev, extend_space)
        extend_hi_hi = _pad_reflect(pyramid[:, :, band_idx], extend_space)
        band_idx -= 1
        extend_hi_lo = _pad_reflect(pyramid[:, :, band_idx], extend_space)
        band_idx -= 1
        extend_lo_hi = _pad_reflect(pyramid[:, :, band_idx], extend_space)
        band_idx -= 1

        ext_height, _ = extend_lo_lo.shape
        lo_lo_row = np.zeros((ext_height, width), dtype=float)
        hi_hi_row = np.zeros((ext_height, width), dtype=float)
        hi_lo_row = np.zeros((ext_height, width), dtype=float)
        lo_hi_row = np.zeros((ext_height, width), dtype=float)

        start = 0
        for idx in range(int(qmf_length)):
            lo_lo_row += extend_lo_lo[:, start : start + width] * filt_low[idx]
            hi_hi_row += extend_hi_hi[:, start : start + width] * filt_high[idx]
            hi_lo_row += extend_hi_lo[:, start : start + width] * filt_high[idx]
            lo_hi_row += extend_lo_hi[:, start : start + width] * filt_low[idx]
            start += step

        lo_lo = np.zeros((height, width), dtype=float)
        lo_hi = np.zeros((height, width), dtype=float)
        hi_lo = np.zeros((height, width), dtype=float)
        hi_hi = np.zeros((height, width), dtype=float)
        start = 0
        for idx in range(int(qmf_length)):
            lo_lo += lo_lo_row[start : start + height, :] * filt_low[idx]
            lo_hi += lo_hi_row[start : start + height, :] * filt_high[idx]
            hi_lo += hi_lo_row[start : start + height, :] * filt_low[idx]
            hi_hi += hi_hi_row[start : start + height, :] * filt_high[idx]
            start += step

        lowpass_prev = lo_lo + lo_hi + hi_lo + hi_hi

    return lowpass_prev


def build_pyramid(im: np.ndarray, nlevels: int, filt_type: str = "haar") -> tuple[np.ndarray, int]:
    """Build a non-decimated HDR pyramid for the supported filter families."""

    normalized = param_format(filt_type)
    if normalized == "haar":
        return _haar_pyramid(im, nlevels), 3
    if normalized == "qmf":
        return qmf_pyramid(im, nlevels), 3
    if normalized == "steerable":
        raise UnsupportedOptionError("buildPyramid", f"filt_type {filt_type}")
    raise ValueError('filt_type can only be "haar", "qmf", or "steerable".')


def recons_pyramid(pyr: np.ndarray, filt_num: int | None = None, filt_type: str = "haar") -> np.ndarray:
    """Reconstruct a non-decimated HDR pyramid for the supported filter families."""

    del filt_num
    normalized = param_format(filt_type)
    if normalized == "haar":
        return _recons_haar_pyramid(pyr)
    if normalized == "qmf":
        return recons_qmf_pyramid(pyr)
    if normalized == "steerable":
        raise UnsupportedOptionError("reconsPyramid", f"filt_type {filt_type}")
    raise ValueError('filt_type can only be "haar", "qmf", or "steerable".')


def im_norm(im: np.ndarray) -> np.ndarray:
    """Normalize an array to the [0, 1] range."""

    image = np.asarray(im, dtype=float)
    low = float(np.min(image))
    high = float(np.max(image))
    if high <= low:
        return np.zeros_like(image, dtype=float)
    return np.asarray((image - low) / (high - low), dtype=float)


def final_touch(im: np.ndarray) -> np.ndarray:
    """Apply the vendored final HDR touch-up layer."""

    image = np.asarray(im, dtype=float)
    return np.asarray(image + 0.15 * _hist_equalize_global(np.real(image)), dtype=float)


def getpfmraw(filename: str | Path) -> np.ndarray:
    """Read a color PFM raw image using the vendored MATLAB contract."""

    path = Path(filename)
    with path.open("rb") as handle:
        code = handle.readline().strip()
        if code != b"P7":
            raise ValueError("Not a PFM (raw) image")

        def _next_token() -> bytes:
            while True:
                line = handle.readline()
                if line == b"":
                    raise ValueError("Unexpected end of PFM header")
                stripped = line.strip()
                if not stripped or stripped.startswith(b"#"):
                    continue
                return stripped

        width_height = _next_token().split()
        if len(width_height) >= 2:
            width = int(width_height[0])
            height = int(width_height[1])
        else:
            width = int(width_height[0])
            height = int(_next_token().split()[0])
        _ = float(_next_token().split()[0])
        payload = handle.read()

    values = np.frombuffer(payload, dtype=np.float32)
    expected = 3 * width * height
    if values.size != expected:
        raise ValueError("PFM payload size does not match header dimensions.")

    channels = values.reshape(3, width * height)
    image = np.empty((height, width, 3), dtype=np.float32)
    for channel_index in range(3):
        plane = np.reshape(channels[channel_index], (width, height), order="F").T
        image[:, :, channel_index] = plane
    return image[::-1, :, :]


pad_reflect = _pad_reflect
pad_reflect_neg = _pad_reflect_neg
haar_pyramid = _haar_pyramid
recons_haar_pyramid = _recons_haar_pyramid
range_compression_lum = _range_compression_lum

buildPyramid = build_pyramid
finalTouch = final_touch
getPFMraw = getpfmraw
haarPyramid = haar_pyramid
imNorm = im_norm
modulateFlip = modulate_flip_shift
modulateFlipShift = modulate_flip_shift
padReflect = pad_reflect
padReflectNeg = pad_reflect_neg
qmfPyramid = qmf_pyramid
rangeCompressionLum = range_compression_lum
reconsHaarPyramid = recons_haar_pyramid
reconsPyramid = recons_pyramid
reconsQmfPyramid = recons_qmf_pyramid


def _scene_rgb_from_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("scene RGB rendering requires an XYZ image cube.")
    return xyz_to_srgb(xyz)


def _scene_rgb_render(scene: Scene, *, asset_store: AssetStore | None = None) -> np.ndarray:
    xyz = np.asarray(scene_get(scene, "xyz", asset_store=asset_store), dtype=float)
    return _scene_rgb_from_xyz(xyz)


def scene_radiance_from_vector(radiance: Any, row: Any, col: Any) -> np.ndarray:
    spectral = np.asarray(radiance, dtype=float).reshape(-1)
    rows = int(row)
    cols = int(col)
    return np.broadcast_to(spectral.reshape(1, 1, -1), (rows, cols, spectral.size)).copy()


def scene_photons_from_vector(radiance: Any, row: Any, col: Any) -> np.ndarray:
    return scene_radiance_from_vector(radiance, row, col)


def scene_energy_from_vector(energy: Any, row: Any, col: Any) -> np.ndarray:
    spectral = np.asarray(energy, dtype=float).reshape(-1)
    rows = int(row)
    cols = int(col)
    return np.broadcast_to(spectral.reshape(1, 1, -1), (rows, cols, spectral.size)).copy()


def scene_crop(
    scene: Scene,
    rect: Any,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[Scene, np.ndarray]:
    """Crop scene data to an ISET rect and return the cropped scene plus rect."""

    rect_array = np.rint(np.asarray(rect, dtype=float).reshape(-1)).astype(int)
    if rect_array.size != 4:
        raise ValueError("sceneCrop expects [col, row, width, height].")

    from .roi import ie_rect2_locs, vc_get_roi_data

    roi_locs = ie_rect2_locs(rect_array)
    cropped_rows = int(rect_array[3]) + 1
    cropped_cols = int(rect_array[2]) + 1
    photons = np.asarray(vc_get_roi_data(scene, roi_locs, "photons"), dtype=float).reshape(cropped_rows, cropped_cols, -1)

    cropped = scene_set(scene.clone(), "photons", photons)

    depth_map = scene.fields.get("depth_map_m")
    if depth_map is not None:
        depth = np.asarray(depth_map, dtype=float)
        row_index = np.clip(roi_locs[:, 0] - 1, 0, depth.shape[0] - 1)
        col_index = np.clip(roi_locs[:, 1] - 1, 0, depth.shape[1] - 1)
        cropped.fields["depth_map_m"] = depth[row_index, col_index].reshape(cropped_rows, cropped_cols)

    if param_format(scene_get(scene, "illuminant format")) == "spatialspectral":
        illuminant = np.asarray(vc_get_roi_data(scene, roi_locs, "illuminant photons"), dtype=float).reshape(cropped_rows, cropped_cols, -1)
        cropped = scene_set(cropped, "illuminant photons", illuminant)

    scene_calculate_luminance(cropped, asset_store=asset_store)
    cropped.metadata["rect"] = rect_array.copy()
    cropped.metadata.pop("coordinates", None)
    return cropped, rect_array.copy()


def scene_extract_waveband(
    scene: Scene,
    wave_list: Any,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    """Extract or interpolate a scene onto a target wavelength list."""

    target_wave = np.asarray(wave_list, dtype=float).reshape(-1)
    if target_wave.size == 0:
        raise ValueError("sceneExtractWaveband requires a non-empty wavelength list.")
    return scene_interpolate_w(scene.clone(), target_wave, preserve_luminance=False, asset_store=asset_store)


def scene_translate(
    scene: Scene,
    dxy: Any,
    fill_values: Any = 0.0,
) -> Scene:
    """Translate scene photons by `[x, y]` degrees using MATLAB-style fill semantics."""

    dxy_array = np.asarray(dxy, dtype=float).reshape(-1)
    if dxy_array.size != 2:
        raise ValueError("sceneTranslate requires [x, y] displacement in degrees.")

    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    deg_per_pixel = float(scene_get(scene, "degree per sample"))
    if deg_per_pixel <= 0.0:
        raise ValueError("sceneTranslate requires a positive scene angular sample spacing.")

    row_shift = float(dxy_array[1]) / deg_per_pixel
    col_shift = float(dxy_array[0]) / deg_per_pixel
    rows, cols, nwave = photons.shape
    fill = np.asarray(fill_values, dtype=float).reshape(-1)
    if fill.size not in {1, nwave}:
        raise ValueError("sceneTranslate fillValues must be a scalar or match the number of wavelength samples.")

    row_coords, col_coords = np.meshgrid(
        np.arange(rows, dtype=float) - row_shift,
        np.arange(cols, dtype=float) - col_shift,
        indexing="ij",
    )
    translated = np.empty_like(photons, dtype=float)
    for band_index in range(nwave):
        cval = float(fill[0] if fill.size == 1 else fill[band_index])
        translated[:, :, band_index] = map_coordinates(
            photons[:, :, band_index],
            [row_coords, col_coords],
            order=1,
            mode="constant",
            cval=cval,
            prefilter=False,
        )

    return scene_set(scene.clone(), "photons", translated)


def scene_spatial_support(scene: Scene, units: Any = "meters") -> dict[str, np.ndarray]:
    """Return the legacy MATLAB sceneSpatialSupport() structure."""

    return {
        axis: np.asarray(values, dtype=float).copy()
        for axis, values in _scene_spatial_support_linear(scene, units).items()
    }


def scene_frequency_support(scene: Scene, units: Any = "cyclesPerDegree") -> dict[str, np.ndarray]:
    """Return the legacy MATLAB sceneFrequencySupport() structure."""

    return {
        axis: np.asarray(values, dtype=float).copy()
        for axis, values in _scene_frequency_support(scene, units).items()
    }


def scene_init_geometry(scene: Scene) -> Scene:
    """Initialize missing scene distance metadata to the MATLAB default 1.2 meters."""

    if scene.fields.get("distance_m") is not None:
        return scene
    scene.fields["distance_m"] = DEFAULT_DISTANCE_M
    return _update_scene_geometry(scene)


def scene_init_spatial(scene: Scene) -> Scene:
    """Initialize missing scene FOV metadata to the MATLAB default 10 degrees."""

    if scene.fields.get("fov_deg") is not None:
        return scene
    scene.fields["fov_deg"] = DEFAULT_FOV_DEG
    return _update_scene_geometry(scene)


def scene_spatial_resample(
    scene: Scene,
    dx: Any,
    units: Any = "m",
    method: str = "linear",
) -> Scene:
    """Resample a scene cube to a new spatial sample spacing."""

    sample_spacing_m = float(dx) / max(_spatial_unit_scale(units), 1e-12)
    if sample_spacing_m <= 0.0:
        raise ValueError("sceneSpatialResample requires a positive sample spacing.")

    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    if photons.size == 0:
        return scene.clone()

    support = _scene_spatial_support_linear(scene, "m")
    x_support = np.asarray(support["x"], dtype=float)
    y_support = np.asarray(support["y"], dtype=float)
    x_query = _support_resample_positions(float(x_support[0]), float(x_support[-1]), sample_spacing_m)
    y_query = _support_resample_positions(float(y_support[0]), float(y_support[-1]), sample_spacing_m)

    resampled_cube = np.empty((y_query.size, x_query.size, photons.shape[2]), dtype=float)
    for band_index in range(photons.shape[2]):
        resampled_cube[:, :, band_index] = _scene_resample_plane_on_support(
            photons[:, :, band_index],
            x_support,
            y_support,
            x_query,
            y_query,
            method=method,
        )

    resampled = scene.clone()
    original_fov = float(scene_get(scene, "fov"))
    resampled = scene_set(resampled, "photons", resampled_cube)

    depth_map = scene.fields.get("depth_map_m")
    if depth_map is not None:
        resampled.fields["depth_map_m"] = _scene_resample_plane_on_support(
            np.asarray(depth_map, dtype=float),
            x_support,
            y_support,
            x_query,
            y_query,
            method="nearest",
        )

    illuminant_photons = scene.fields.get("illuminant_photons")
    if illuminant_photons is not None:
        illuminant_array = np.asarray(illuminant_photons, dtype=float)
        if illuminant_array.ndim == 3:
            resampled_illuminant = np.empty((y_query.size, x_query.size, illuminant_array.shape[2]), dtype=float)
            for band_index in range(illuminant_array.shape[2]):
                resampled_illuminant[:, :, band_index] = _scene_resample_plane_on_support(
                    illuminant_array[:, :, band_index],
                    x_support,
                    y_support,
                    x_query,
                    y_query,
                    method=method,
                )
            resampled = scene_set(resampled, "illuminant photons", resampled_illuminant)

    distance_m = float(scene_get(resampled, "distance"))
    target_width_m = sample_spacing_m * max(int(scene_get(resampled, "cols")), 1)
    if distance_m > 0.0 and target_width_m > 0.0:
        target_fov = np.rad2deg(2.0 * np.arctan2(target_width_m / 2.0, distance_m))
        resampled = scene_set(resampled, "fov", target_fov)
    elif original_fov > 0.0:
        resampled = scene_set(resampled, "fov", original_fov)

    if resampled.name:
        resampled.name = f"{resampled.name}-{param_format(method or 'linear')}"
    return resampled


def scene_pad(
    oi: Any,
    pad_size: Any,
    s_dist: Any | None = None,
) -> Any:
    """Deprecated MATLAB shim retained for compatibility."""

    raise ValueError("Deprecated. You probably want oiPad.")


def scene_photon_noise(
    scene: Scene,
    rect_or_locs: Any | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add MATLAB-style photon noise to a full scene cube or ROI sample matrix."""

    if rect_or_locs is None:
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    else:
        from .roi import vc_get_roi_data

        photons = np.asarray(vc_get_roi_data(scene, rect_or_locs, "photons"), dtype=float)

    rng = np.random.default_rng(None if seed is None else int(seed))
    the_noise = np.sqrt(np.clip(photons, 0.0, None)) * rng.standard_normal(photons.shape)
    noisy_photons = np.rint(photons + the_noise)

    poisson_mask = photons < 15.0
    if np.any(poisson_mask):
        poisson_samples = rng.poisson(np.clip(photons[poisson_mask], 0.0, None))
        the_noise = np.asarray(the_noise, dtype=float)
        the_noise[poisson_mask] = np.asarray(poisson_samples, dtype=float)
        noisy_photons[poisson_mask] = np.asarray(poisson_samples, dtype=float)

    return np.asarray(noisy_photons, dtype=float), np.asarray(the_noise, dtype=float)


def scene_description(
    scene: Scene | None,
    *,
    asset_store: AssetStore | None = None,
) -> str:
    """Return a headless text description for a scene."""

    if scene is None:
        return "No scene"

    rows = int(scene_get(scene, "rows"))
    cols = int(scene_get(scene, "cols"))
    height_m = float(scene_get(scene, "height"))
    width_m = float(scene_get(scene, "width"))
    sample_size_m = width_m / max(cols, 1)
    deg_per_sample = float(scene_get(scene, "fov")) / max(cols, 1)
    wave = np.asarray(scene_get(scene, "wave"), dtype=float).reshape(-1)
    spacing = float(np.mean(np.diff(wave))) if wave.size >= 2 else 0.0
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    mx = float(np.max(luminance)) if luminance.size else 0.0
    mn = float(np.min(luminance)) if luminance.size else 0.0

    size_order = round(np.log10(max(height_m, 1e-12)))
    if size_order >= 0:
        size_unit, size_scale = "m", 1.0
    elif size_order >= -3:
        size_unit, size_scale = "mm", 1e3
    else:
        size_unit, size_scale = "um", 1e6

    if sample_size_m >= 1.0:
        sample_unit, sample_scale = "m", 1.0
    elif sample_size_m >= 1e-3:
        sample_unit, sample_scale = "mm", 1e3
    else:
        sample_unit, sample_scale = "um", 1e6

    lines = [
        f"Row,Col:\t{rows:.0f} by {cols:.0f} ",
        f"Hgt,Wdth:\t({height_m * size_scale:3.2f}, {width_m * size_scale:3.2f}) {size_unit}",
        f"Sample:\t{sample_size_m * sample_scale:3.2f} {sample_unit}",
        f"Deg/samp: {deg_per_sample:2.2f}",
    ]
    if wave.size:
        lines.append(f"Wave:\t{float(np.min(wave)):.0f}:{spacing:.0f}:{float(np.max(wave)):.0f} nm")
    if mn == 0.0:
        lines.append(f"DR: Inf\n  (max {mx:.0f}, min {mn:.2f} cd/m2)")
    else:
        lines.append(f"DR: {20.0 * np.log10(mx / mn):.2f} dB (max {mx:.0f} cd/m2)")
    if scene.fields.get("depth_map_m") is not None:
        depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
        lines.append(f"Depth range: [{float(np.min(depth_map)):.1f} {float(np.max(depth_map)):.1f}]m")
    return "\n".join(lines)


def scene_list() -> str:
    """Return a headless summary of supported sceneCreate families."""

    return _SCENE_LIST_TEXT


def scene_save_image(
    scene: Scene,
    f_name: str | Path,
    *,
    render_flag: int = 1,
    gam: float = 1.0,
    asset_store: AssetStore | None = None,
) -> str:
    """Save a rendered scene image to an 8-bit PNG without opening a window."""

    image = np.asarray(scene_show_image(scene, -abs(int(render_flag)), float(gam), asset_store=asset_store), dtype=float)
    output_path = Path(f_name).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.clip(np.round(np.clip(image, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)
    iio.imwrite(output_path, payload)
    return str(output_path)


def scene_thumbnail(
    scene: Scene,
    *args: Any,
    asset_store: AssetStore | None = None,
    **kwargs: Any,
) -> str:
    """Write a headless scene thumbnail PNG using the legacy MATLAB shape."""

    options: dict[str, Any] = {}
    if len(args) % 2 != 0:
        raise ValueError("sceneThumbnail expects key/value arguments after the scene.")
    for index in range(0, len(args), 2):
        options[param_format(str(args[index]))] = args[index + 1]
    for key, value in kwargs.items():
        options[param_format(str(key))] = value

    row_size = max(int(np.rint(options.get("rowsize", 192))), 1)
    force_square = bool(options.get("forcesquare", False))
    output_filename = options.get("outputfilename")

    rgb = np.asarray(scene_show_image(scene, -1, 1.0, asset_store=asset_store), dtype=float)
    rows, cols = rgb.shape[:2]
    col_size = max(int(np.rint((row_size / max(rows, 1)) * cols)), 1)
    resized = zoom(rgb, (row_size / max(rows, 1), col_size / max(cols, 1), 1.0), order=1)
    resized = np.clip(np.asarray(resized, dtype=float), 0.0, 1.0)

    if force_square:
        pad_size = row_size - resized.shape[1]
        if pad_size > 0:
            resized = np.pad(resized, ((0, 0), (0, pad_size), (0, 0)), mode="constant", constant_values=0.3)
        elif pad_size < 0:
            resized = resized[:, :row_size, :]

    output_name = scene.name if output_filename in {None, ""} else str(output_filename)
    output_path = Path(output_name).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = np.clip(np.round(resized * 255.0), 0.0, 255.0).astype(np.uint8)
    iio.imwrite(output_path, payload)
    return str(output_path)


def scene_show_image(
    scene: Scene,
    render_flag: int = 1,
    gam: float = 1.0,
    app: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    del app
    method = abs(int(render_flag))
    clip_level = 90.0 if method == 5 else 99.5
    if method == 5:
        method = 4

    if method in {0, 1}:
        rgb = _scene_rgb_render(scene, asset_store=asset_store)
    elif method == 2:
        photons = np.asarray(scene_get(scene, "photons"), dtype=float)
        gray = np.mean(photons, axis=2, dtype=float)
        rgb = np.repeat(_minmax_normalize(gray)[:, :, np.newaxis], 3, axis=2)
    elif method == 3:
        rgb = hdr_render(_scene_rgb_render(scene, asset_store=asset_store))
    elif method == 4:
        xyz = np.asarray(scene_get(scene, "xyz", asset_store=asset_store), dtype=float)
        y_channel = xyz[:, :, 1]
        y_clip = float(np.percentile(y_channel, clip_level))
        xyz = np.clip(xyz, 0.0, y_clip)
        rgb = hdr_render(_scene_rgb_from_xyz(xyz))
    else:
        raise UnsupportedOptionError(f"sceneShowImage renderFlag={render_flag} is not supported.")

    if float(gam) != 1.0:
        rgb = np.power(np.clip(np.asarray(rgb, dtype=float), 0.0, None), float(gam))
    return np.asarray(rgb, dtype=float)



def scene_adjust_luminance(
    scene: Scene,
    target_luminance: float,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    current_luminance = scene_get(scene, "mean luminance", asset_store=asset_store)
    if current_luminance <= 0.0:
        return scene
    scale = float(target_luminance) / current_luminance
    scene.data["photons"] = np.asarray(scene.data["photons"], dtype=float) * scale
    if "illuminant_photons" in scene.fields:
        scene.fields["illuminant_photons"] = np.asarray(scene.fields["illuminant_photons"], dtype=float) * scale
    if "illuminant_energy" in scene.fields:
        scene.fields["illuminant_energy"] = np.asarray(scene.fields["illuminant_energy"], dtype=float) * scale
    _invalidate_scene_caches(scene)
    scene_calculate_luminance(scene, asset_store=asset_store)
    return scene


def scene_interpolate_w(
    scene: Scene,
    wave: Any,
    preserve_luminance: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    source_wave = np.asarray(scene.fields.get("wave", DEFAULT_WAVE), dtype=float).reshape(-1)
    target_wave = np.asarray(wave, dtype=float).reshape(-1)

    if target_wave.size == 0:
        raise ValueError("Target wavelength samples must not be empty.")
    if source_wave.size == 0 or np.array_equal(source_wave, target_wave):
        scene.fields["wave"] = target_wave
        _invalidate_scene_caches(scene)
        return scene
    if float(np.min(target_wave)) < float(np.min(source_wave)) or float(np.max(target_wave)) > float(np.max(source_wave)):
        raise ValueError("sceneInterpolateW does not support extrapolation outside the current wavelength support.")

    store = _store(asset_store)
    scene_photons = np.asarray(scene.data.get("photons", np.empty((0, 0, 0))), dtype=float)
    original_mean: float | None = None
    if scene_photons.size > 0:
        if preserve_luminance:
            original_mean = float(scene_get(scene, "mean luminance", asset_store=store))
        scene.data["photons"] = _resample_wave_last(scene_photons, source_wave, target_wave)

    illuminant_photons = scene.fields.get("illuminant_photons")
    illuminant_energy = scene.fields.get("illuminant_energy")
    if illuminant_photons is not None:
        illuminant_array = np.asarray(illuminant_photons, dtype=float)
        if illuminant_array.ndim == 1:
            resampled_illuminant = np.asarray(interp_spectra(source_wave, illuminant_array, target_wave), dtype=float).reshape(-1)
        else:
            resampled_illuminant = _resample_wave_last(illuminant_array, source_wave, target_wave)
        scene.fields["illuminant_photons"] = resampled_illuminant
        scene.fields["illuminant_energy"] = np.asarray(quanta_to_energy(resampled_illuminant, target_wave), dtype=float)
    elif illuminant_energy is not None:
        illuminant_array = np.asarray(illuminant_energy, dtype=float)
        if illuminant_array.ndim == 1:
            resampled_energy = np.asarray(interp_spectra(source_wave, illuminant_array, target_wave), dtype=float).reshape(-1)
        else:
            resampled_energy = _resample_wave_last(illuminant_array, source_wave, target_wave)
        scene.fields["illuminant_energy"] = np.asarray(resampled_energy, dtype=float)
        scene.fields["illuminant_photons"] = np.asarray(energy_to_quanta(resampled_energy, target_wave), dtype=float)

    scene.fields["wave"] = target_wave
    _invalidate_scene_caches(scene)
    if original_mean is not None:
        scene_adjust_luminance(scene, original_mean, asset_store=store)
    return scene


def _scene_roi_xyz(scene: Scene, roi_locs: Any | None = None, *, asset_store: AssetStore | None = None) -> np.ndarray:
    store = _store(asset_store)
    wave = np.asarray(scene.fields["wave"], dtype=float)
    if roi_locs is None:
        photons = np.asarray(scene.data["photons"], dtype=float).reshape(-1, wave.size)
    else:
        photons = np.asarray(scene_get(scene, "roi photons", roi_locs, asset_store=store), dtype=float)
    energy = quanta_to_energy(photons, wave)
    return xyz_from_energy(energy, wave, asset_store=store)


def _scene_spatial_support_linear(scene: Scene, unit: Any | None = None) -> dict[str, np.ndarray]:
    rows, cols = scene_get(scene, "size")
    if rows <= 0 or cols <= 0:
        return {"x": np.empty(0, dtype=float), "y": np.empty(0, dtype=float)}
    width_m = float(scene.fields["width_m"])
    height_m = float(scene.fields["height_m"])
    dx = width_m / cols
    dy = height_m / rows
    x = ((np.arange(cols, dtype=float) + 0.5) - (cols / 2.0)) * dx
    y = ((np.arange(rows, dtype=float) + 0.5) - (rows / 2.0)) * dy
    scale = _spatial_unit_scale(unit)
    return {"x": x * scale, "y": y * scale}


def _line_index(line_arg: Any, orientation: str) -> int:
    values = np.asarray(line_arg, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Line location must be a scalar or [col, row] locator.")
    if values.size == 1:
        return int(np.rint(values[0]))
    return int(np.rint(values[1 if orientation == "h" else 0]))


def _scene_line_profile(
    scene: Scene,
    data_type: str,
    orientation: str,
    line_arg: Any,
    *,
    unit: Any | None = "mm",
    asset_store: AssetStore | None = None,
) -> dict[str, np.ndarray]:
    wave = np.asarray(scene.fields["wave"], dtype=float)
    support = _scene_spatial_support_linear(scene, unit)
    line_index = _line_index(line_arg, orientation)
    rows, cols = scene_get(scene, "size")
    if data_type == "photons":
        data = np.asarray(scene.data["photons"], dtype=float)
    elif data_type == "illuminant_photons":
        illuminant = np.asarray(scene_get(scene, "illuminant photons", asset_store=asset_store), dtype=float)
        if illuminant.ndim == 1:
            illuminant = np.broadcast_to(illuminant.reshape(1, 1, -1), (rows, cols, illuminant.size)).copy()
        data = illuminant
    elif data_type == "illuminant_energy":
        illuminant = np.asarray(scene_get(scene, "illuminant energy", asset_store=asset_store), dtype=float)
        if illuminant.ndim == 1:
            illuminant = np.broadcast_to(illuminant.reshape(1, 1, -1), (rows, cols, illuminant.size)).copy()
        data = illuminant
    elif data_type == "luminance":
        data = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    else:
        raise KeyError(f"Unsupported scene line profile data type: {data_type}")

    if orientation == "h":
        if line_index < 1 or line_index > data.shape[0]:
            raise IndexError("Horizontal scene line index is out of range.")
        pos = np.asarray(support["x"], dtype=float)
        line = np.asarray(data[line_index - 1, ...], dtype=float)
    else:
        if line_index < 1 or line_index > data.shape[1]:
            raise IndexError("Vertical scene line index is out of range.")
        pos = np.asarray(support["y"], dtype=float)
        line = np.asarray(data[:, line_index - 1, ...], dtype=float)

    if line.ndim == 1:
        return {"pos": pos.copy(), "data": line.copy(), "unit": str(unit or "m")}
    return {"pos": pos.copy(), "wave": wave.copy(), "data": line.T.copy(), "unit": str(unit or "m")}


def _resolve_illuminant_input(
    ill_energy: Any,
    wave: np.ndarray,
    *,
    asset_store: AssetStore,
) -> tuple[np.ndarray, str]:
    if isinstance(ill_energy, str):
        path = Path(ill_energy)
        if path.exists():
            data = asset_store.load_mat(path)
            name = path.name
        else:
            try:
                _, data_values = asset_store.load_illuminant(ill_energy, wave_nm=wave)
                name = path.name if path.suffix else f"{ill_energy}.mat"
                return data_values, name
            except MissingAssetError:
                data = asset_store.load_mat(ill_energy)
                name = path.name
        energy = np.asarray(data["data"], dtype=float)
        source_wave = np.asarray(data["wavelength"], dtype=float).reshape(-1)
        if energy.ndim == 1:
            return np.interp(wave, source_wave, energy.reshape(-1), left=0.0, right=0.0), name
        if energy.shape[-1] != source_wave.size and energy.shape[0] == source_wave.size:
            energy = np.moveaxis(energy, 0, -1)
        if np.array_equal(source_wave, wave):
            return energy, name
        return _resample_wave_last(energy, source_wave, wave), name
    if isinstance(ill_energy, dict) and "energy" in ill_energy:
        return np.asarray(ill_energy["energy"], dtype=float), str(ill_energy.get("name", "custom"))
    if ill_energy is None:
        return blackbody(wave, 6500.0, kind="energy"), "blackbody-6500K"
    return np.asarray(ill_energy, dtype=float), "custom"


def scene_adjust_illuminant(
    scene: Scene,
    ill_energy: Any,
    preserve_mean: bool = True,
    *,
    asset_store: AssetStore | None = None,
) -> Scene:
    store = _store(asset_store)
    wave = np.asarray(scene.fields["wave"], dtype=float)
    original_mean = float(scene_get(scene, "mean luminance", asset_store=store))
    new_energy, comment = _resolve_illuminant_input(ill_energy, wave, asset_store=store)
    new_energy = np.asarray(new_energy, dtype=float)
    if new_energy.ndim == 2 and 1 in new_energy.shape and new_energy.size == wave.size:
        new_energy = new_energy.reshape(-1)
    new_photons = energy_to_quanta(new_energy, wave)
    current_illuminant = np.asarray(
        scene.fields.get("illuminant_photons", np.ones_like(new_photons)),
        dtype=float,
    )
    if current_illuminant.ndim == 2 and 1 in current_illuminant.shape and current_illuminant.size == wave.size:
        current_illuminant = current_illuminant.reshape(-1)
    factor = new_photons / np.maximum(current_illuminant, 1e-12)
    scene.data["photons"] = np.asarray(scene.data["photons"], dtype=float) * factor
    scene = _set_scene_illuminant_energy(scene, new_energy)
    scene.fields["illuminant_comment"] = comment
    if preserve_mean:
        scene_adjust_luminance(scene, original_mean, asset_store=store)
    return scene


def scene_clear_data(scene: Scene) -> Scene:
    cleared = scene.clone()
    if "photons" in cleared.data:
        cleared.data["photons"] = np.zeros_like(cleared.data["photons"], dtype=float)
    _invalidate_scene_caches(cleared)
    return cleared


def scene_get(scene: Scene, parameter: str, *args: Any, asset_store: AssetStore | None = None) -> Any:
    key = param_format(parameter)
    if key == "type":
        return scene.type
    if key == "name":
        return scene.name
    if key == "filename":
        return scene.fields.get("filename")
    if key == "sourcetype":
        return scene.fields.get("source_type")
    if key == "metadata":
        return scene.metadata
    if key == "wave":
        return np.asarray(scene.fields["wave"], dtype=float).reshape(-1)
    if key == "nwave":
        return int(np.asarray(scene.fields["wave"], dtype=float).size)
    if key == "photons":
        photons = np.asarray(scene.data["photons"], dtype=float)
        if args:
            wavelength = float(np.asarray(args[0], dtype=float).reshape(-1)[0])
            wave = np.asarray(scene.fields["wave"], dtype=float).reshape(-1)
            return np.asarray(photons[:, :, int(np.argmin(np.abs(wave - wavelength)))], dtype=float)
        return photons
    if key == "energy":
        photons = np.asarray(scene.data["photons"], dtype=float)
        wave = np.asarray(scene.fields["wave"], dtype=float).reshape(-1)
        energy = np.asarray(quanta_to_energy(photons, wave), dtype=float)
        if args:
            wavelength = float(np.asarray(args[0], dtype=float).reshape(-1)[0])
            return np.asarray(energy[:, :, int(np.argmin(np.abs(wave - wavelength)))], dtype=float)
        return energy
    if key == "data":
        return scene.data
    if key == "rows":
        return int(scene.fields["rows"])
    if key == "cols":
        return int(scene.fields["cols"])
    if key == "size":
        return (int(scene.fields["rows"]), int(scene.fields["cols"]))
    if key in {"distance", "distancem"}:
        scale = _spatial_unit_scale(args[0] if args else "m")
        return float(scene.fields["distance_m"]) * scale
    if key == "depthmap":
        depth_map = scene.fields.get("depth_map_m")
        if depth_map is None:
            depth_map = np.full(scene_get(scene, "size"), float(scene.fields["distance_m"]), dtype=float)
        scale = _spatial_unit_scale(args[0] if args else None)
        return np.asarray(depth_map, dtype=float) * scale
    if key == "depthrange":
        depth_map = np.asarray(scene_get(scene, "depth map"), dtype=float)
        positive = depth_map[depth_map > 0.0]
        if positive.size == 0:
            return np.empty(0, dtype=float)
        scale = _spatial_unit_scale(args[0] if args else None)
        return np.array([positive.min(), positive.max()], dtype=float) * scale
    if key in {"fov", "hfov", "fovhorizontal", "wangular"}:
        return float(scene.fields["fov_deg"])
    if key in {"vfov", "fovvertical"}:
        return float(scene.fields["vfov_deg"])
    if key == "width":
        return float(scene.fields["width_m"])
    if key == "height":
        return float(scene.fields["height_m"])
    if key in {"distperdeg", "distanceperdegree"}:
        scale = _spatial_unit_scale(args[0] if args else None)
        return float(scene.fields["width_m"]) / max(float(scene.fields["fov_deg"]), 1e-12) * scale
    if key in {"degreesperdistance", "degperdist"}:
        scale = _spatial_unit_scale(args[0] if args else None)
        return float(scene.fields["fov_deg"]) / max(float(scene.fields["width_m"]), 1e-12) / max(scale, 1e-12)
    if key in {"degreepersample", "degpersamp", "degreespersample"}:
        return float(scene.fields["fov_deg"]) / max(int(scene.fields["cols"]), 1)
    if key in {"samplespacing", "samplespacingmeters", "samplespacingm", "sample spacing"}:
        scale = _spatial_unit_scale(args[0] if args else "m")
        cols = max(int(scene.fields["cols"]), 1)
        rows = max(int(scene.fields["rows"]), 1)
        return np.array(
            [
                float(scene.fields["width_m"]) * scale / cols,
                float(scene.fields["height_m"]) * scale / rows,
            ],
            dtype=float,
        )
    if key == "spatialsupportlinear":
        return _scene_spatial_support_linear(scene, args[0] if args else None)
    if key == "spatialsupport":
        support = _scene_spatial_support_linear(scene, args[0] if args else None)
        xx, yy = np.meshgrid(support["x"], support["y"])
        return np.stack((xx, yy), axis=2)
    if key in {"frequencyresolution", "freqres"}:
        units = args[0] if args else "cyclesPerDegree"
        return _scene_frequency_support(scene, units)
    if key in {"maxfrequencyresolution", "maxfreqres"}:
        units = args[0] if args else "cyclesPerDegree"
        frequency_resolution = _scene_frequency_support(scene, units)
        return float(max(np.max(frequency_resolution["fx"]), np.max(frequency_resolution["fy"])))
    if key in {"frequencysupport", "fsupportxy", "fsupport2d", "fsupport"}:
        units = args[0] if args else "cyclesPerDegree"
        frequency_resolution = _scene_frequency_support(scene, units)
        xx, yy = np.meshgrid(frequency_resolution["fx"], frequency_resolution["fy"])
        return np.stack((xx, yy), axis=2)
    if key in {"frequencysupportcol", "fsupportx"}:
        units = args[0] if args else "cyclesPerDegree"
        fx = np.asarray(_scene_frequency_support(scene, units)["fx"], dtype=float)
        zero_index = int(np.argmin(np.abs(fx)))
        return fx[zero_index:].copy()
    if key in {"frequencysupportrow", "fsupporty"}:
        units = args[0] if args else "cyclesPerDegree"
        fy = np.asarray(_scene_frequency_support(scene, units)["fy"], dtype=float)
        zero_index = int(np.argmin(np.abs(fy)))
        return fy[zero_index:].copy()
    if key == "illuminantformat":
        return scene.fields.get("illuminant_format", "spectral")
    if key == "illuminantcomment":
        return scene.fields.get("illuminant_comment")
    if key == "illuminant":
        return {
            "wave": np.asarray(scene.fields["wave"], dtype=float).reshape(-1),
            "data": np.asarray(scene.fields["illuminant_energy"], dtype=float),
            "energy": np.asarray(scene.fields["illuminant_energy"], dtype=float),
            "photons": np.asarray(scene.fields["illuminant_photons"], dtype=float),
            "comment": scene.fields.get("illuminant_comment"),
        }
    if key == "illuminantxyz":
        return xyz_from_energy(
            np.asarray(scene.fields["illuminant_energy"], dtype=float),
            np.asarray(scene.fields["wave"], dtype=float),
            asset_store=_store(asset_store),
        )
    if key == "reflectance":
        photons = np.asarray(scene.data["photons"], dtype=float)
        illuminant = np.asarray(scene.fields["illuminant_photons"], dtype=float)
        if illuminant.ndim == 1:
            illuminant = np.broadcast_to(illuminant.reshape(1, 1, -1), photons.shape).copy()
        return np.divide(photons, illuminant, out=np.zeros_like(photons), where=illuminant > 0.0)
    if key == "chartparameters":
        return scene.fields.get("chart_parameters")
    if key in {"cornerpoints", "chartcornerpoints", "chartcorners"}:
        chart = scene.fields.get("chart_parameters", {})
        value = chart.get("cornerPoints")
        return None if value is None else np.asarray(value).copy()
    if key == "mcccornerpoints":
        return scene_get(scene, "chart corner points")
    if key in {"chartrects", "chartrectangles"}:
        chart = scene.fields.get("chart_parameters", {})
        value = chart.get("rects")
        return None if value is None else np.asarray(value).copy()
    if key in {"currentrect", "chartcurrentrect"}:
        chart = scene.fields.get("chart_parameters", {})
        value = chart.get("currentRect")
        return None if value is None else np.asarray(value).copy()
    if key == "mccrecthandles":
        return scene.fields.get("mccRectHandles")
    if key == "knownreflectance":
        value = scene.fields.get("known_reflectance")
        if value is None:
            return np.array([], dtype=float)
        return np.asarray(value, dtype=float).reshape(-1).copy()
    if key == "peakradianceandwave":
        photons = np.asarray(scene.data["photons"], dtype=float)
        peak_index = int(np.argmax(photons))
        row_index, col_index, wave_index = np.unravel_index(peak_index, photons.shape)
        return np.array(
            [
                float(photons[row_index, col_index, wave_index]),
                float(np.asarray(scene.fields["wave"], dtype=float).reshape(-1)[wave_index]),
            ],
            dtype=float,
        )
    if key == "illuminantphotons":
        return np.asarray(scene.fields["illuminant_photons"], dtype=float)
    if key == "illuminantenergy":
        return np.asarray(scene.fields["illuminant_energy"], dtype=float)
    if key in {"roiilluminantphotons", "illuminantphotonsroi"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi illuminant photons').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "illuminant photons")
    if key in {"roimeanilluminantphotons", "illuminantphotonsroimean"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean illuminant photons').")
        return np.mean(np.asarray(scene_get(scene, "roi illuminant photons", args[0], asset_store=asset_store), dtype=float), axis=0).reshape(-1)
    if key in {"roiilluminantenergy", "illuminantenergyroi"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi illuminant energy').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "illuminant energy")
    if key in {"roimeanilluminantenergy", "illuminantenergyroimean"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean illuminant energy').")
        return np.mean(np.asarray(scene_get(scene, "roi illuminant energy", args[0], asset_store=asset_store), dtype=float), axis=0).reshape(-1)
    if key in {"illuminanthlinephotons", "hlineilluminantphotons"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant hline photons').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_photons", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"illuminantvlinephotons", "vlineilluminantphotons"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant vline photons').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_photons", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"illuminanthlineenergy", "hlineilluminantenergy"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant hline energy').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_energy", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"illuminantvlineenergy", "vlineilluminantenergy"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'illuminant vline energy').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "illuminant_energy", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"radiancehline", "hlineradiance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'radiance hline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "photons", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"radiancevline", "vlineradiance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'radiance vline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "photons", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"luminancehline", "hlineluminance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'luminance hline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "luminance", "h", args[0], unit=unit, asset_store=asset_store)
    if key in {"luminancevline", "vlineluminance"}:
        if not args:
            raise ValueError("Line location required for sceneGet(..., 'luminance vline').")
        unit = args[1] if len(args) >= 2 else "mm"
        return _scene_line_profile(scene, "luminance", "v", args[0], unit=unit, asset_store=asset_store)
    if key in {"roiphotons", "roiphotonsspd"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi photons').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "photons")
    if key == "roienergy":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi energy').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "energy")
    if key == "roimeanenergy":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean energy').")
        return np.mean(np.asarray(scene_get(scene, "roi energy", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roimeanphotons":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean photons').")
        return np.mean(np.asarray(scene_get(scene, "roi photons", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roireflectance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi reflectance').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "reflectance")
    if key == "roimeanreflectance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean reflectance').")
        return np.mean(np.asarray(scene_get(scene, "roi reflectance", args[0]), dtype=float), axis=0).reshape(-1)
    if key == "roiluminance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi luminance').")
        from .roi import vc_get_roi_data

        return vc_get_roi_data(scene, args[0], "luminance")
    if key == "roimeanluminance":
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi mean luminance').")
        return float(np.mean(np.asarray(scene_get(scene, "roi luminance", args[0], asset_store=asset_store), dtype=float)))
    if key == "xyz":
        wave = np.asarray(scene.fields["wave"], dtype=float)
        photons = np.asarray(scene.data["photons"], dtype=float)
        energy = quanta_to_energy(photons, wave)
        return xyz_from_energy(energy, wave, asset_store=_store(asset_store))
    if key in {"rgb", "rgbimage", "srgb"}:
        return _scene_rgb_render(scene, asset_store=asset_store)
    if key in {"chromaticity", "roichromaticity"}:
        roi_locs = args[0] if args else None
        return chromaticity_xy(_scene_roi_xyz(scene, roi_locs, asset_store=asset_store))
    if key in {"roichromaticitymean", "roimeanchromaticity"}:
        if not args:
            raise ValueError("ROI required for sceneGet(..., 'roi chromaticity mean').")
        return np.mean(
            np.asarray(scene_get(scene, "chromaticity", args[0], asset_store=asset_store), dtype=float),
            axis=0,
        ).reshape(-1)
    if key == "luminance":
        if "luminance" not in scene.fields:
            scene_calculate_luminance(scene, asset_store=asset_store)
        return np.asarray(scene.fields["luminance"], dtype=float)
    if key in {"meanluminance", "luminancemean"}:
        if "mean_luminance" not in scene.fields:
            scene_calculate_luminance(scene, asset_store=asset_store)
        return float(scene.fields["mean_luminance"])
    raise KeyError(f"Unsupported sceneGet parameter: {parameter}")


def scene_set(scene: Scene, parameter: str, value: Any) -> Scene:
    key = param_format(parameter)
    if key == "name":
        scene.name = str(value)
        return scene
    if key == "filename":
        scene.fields["filename"] = str(value)
        return scene
    if key == "metadata":
        scene.metadata = dict(value)
        return scene
    if key == "wave":
        return scene_interpolate_w(scene, value)
    if key == "photons":
        scene.data["photons"] = np.asarray(value, dtype=float)
        _invalidate_scene_caches(scene)
        return _update_scene_geometry(scene)
    if key == "energy":
        wave = np.asarray(scene.fields["wave"], dtype=float).reshape(-1)
        scene.data["photons"] = np.asarray(energy_to_quanta(np.asarray(value, dtype=float), wave), dtype=float)
        _invalidate_scene_caches(scene)
        return _update_scene_geometry(scene)
    if key == "depthmap":
        depth_map = np.asarray(value, dtype=float)
        if depth_map.shape != scene_get(scene, "size"):
            raise ValueError("Scene depth map must match the scene size.")
        scene.fields["depth_map_m"] = depth_map
        return scene
    if key in {"distance", "distancem"}:
        scene.fields["distance_m"] = float(value)
        return _update_scene_geometry(scene)
    if key in {"fov", "hfov", "wangular"}:
        scene.fields["fov_deg"] = float(value)
        return _update_scene_geometry(scene)
    if key == "resize":
        return _scene_resize(scene, value)
    if key in {"meanluminance", "luminancemean"}:
        return scene_adjust_luminance(scene, float(value))
    if key == "illuminantenergy":
        return _set_scene_illuminant_energy(scene, value)
    if key == "illuminantphotons":
        return _set_scene_illuminant_photons(scene, value)
    if key == "illuminantcomment":
        scene.fields["illuminant_comment"] = str(value)
        return scene
    if key == "chartparameters":
        scene.fields["chart_parameters"] = dict(value)
        return scene
    if key in {"chartcornerpoints", "cornerpoints", "chartcorners"}:
        scene.fields.setdefault("chart_parameters", {})
        scene.fields["chart_parameters"]["cornerPoints"] = np.asarray(value).copy()
        return scene
    if key == "mcccornerpoints":
        scene.fields.setdefault("chart_parameters", {})
        scene.fields["chart_parameters"]["cornerPoints"] = np.asarray(value).copy()
        return scene
    if key in {"chartrects", "chartrectangles"}:
        scene.fields.setdefault("chart_parameters", {})
        scene.fields["chart_parameters"]["rects"] = np.asarray(value).copy()
        return scene
    if key in {"chartcurrentrect", "currentrect"}:
        scene.fields.setdefault("chart_parameters", {})
        scene.fields["chart_parameters"]["currentRect"] = np.asarray(value).copy()
        return scene
    if key == "mccrecthandles":
        scene.fields["mccRectHandles"] = value
        return scene
    if key == "knownreflectance":
        scene.fields["known_reflectance"] = np.asarray(value, dtype=float).reshape(-1)
        return scene
    raise KeyError(f"Unsupported sceneSet parameter: {parameter}")


exiftoolDepthFromFile = exiftool_depth_from_file
exiftoolInfo = exiftool_info
FOTParams = fot_params
gaborP = gabor_p
hdrRender = hdr_render
ieCookTorrance = ie_cook_torrance
ieCheckerboard = ie_checkerboard
ieReflectanceSamples = ie_reflectance_samples
macbethChartCreate = macbeth_chart_create
macbethDrawRects = macbeth_draw_rects
macbethEvaluationGraphs = macbeth_evaluation_graphs
macbethGretagSGCreate = macbeth_gretag_sg_create
macbethIdealColor = macbeth_ideal_color
macbethLuminanceNoise = macbeth_luminance_noise
macbethPatchData = macbeth_patch_data
macbethReadReflectance = macbeth_read_reflectance
macbethRectangles = macbeth_rectangles
macbethROIs = macbeth_rois
macbethSelect = macbeth_select
macbethSensorValues = macbeth_sensor_values
MOTarget = mo_target
chartPatchData = chart_patch_data
sceneCreate = scene_create
sceneAdd = scene_add
sceneAddGrid = scene_add_grid
sceneAdjustIlluminant = scene_adjust_illuminant
sceneAdjustLuminance = scene_adjust_luminance
sceneAdjustPixelSize = scene_adjust_pixel_size
sceneAdjustReflectance = scene_adjust_reflectance
sceneCalculateLuminance = scene_calculate_luminance
sceneClearData = scene_clear_data
sceneCombine = scene_combine
sceneCrop = scene_crop
sceneFromFile = scene_from_file
sceneFromBasis = scene_from_basis
sceneFromDDFFile = scene_from_ddf_file
sceneGet = scene_get
sceneDescription = scene_description
sceneEnergyFromVector = scene_energy_from_vector
sceneExtractWaveband = scene_extract_waveband
sceneFrequencySupport = scene_frequency_support
sceneHDRChart = scene_hdr_chart
sceneHDRImage = scene_hdr_image
sceneIlluminantPattern = scene_illuminant_pattern
sceneIlluminantScale = scene_illuminant_scale
sceneIlluminantSS = scene_illuminant_ss
sceneInitSpatial = scene_init_spatial
sceneInitGeometry = scene_init_geometry
sceneInsert = scene_insert
sceneInterpolateW = scene_interpolate_w
sceneList = scene_list
sceneMakeVideo = scene_make_video
scenePhotonNoise = scene_photon_noise
scenePhotonsFromVector = scene_photons_from_vector
sceneRadianceChart = scene_radiance_chart
sceneRadianceFromVector = scene_radiance_from_vector
sceneReflectanceChart = scene_reflectance_chart
sceneRamp = scene_ramp
sceneRotate = scene_rotate
sceneSaveImage = scene_save_image
sceneSDR = scene_sdr
sceneSet = scene_set
sceneShowImage = scene_show_image
scenePad = scene_pad
sceneSPDScale = scene_spd_scale
sceneSpatialResample = scene_spatial_resample
sceneSpatialSupport = scene_spatial_support
sceneThumbnail = scene_thumbnail
sceneToFile = scene_to_file
sceneTranslate = scene_translate
sceneVernier = scene_vernier
sceneWBCreate = scene_wb_create
