"""Headless illuminant object helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .assets import AssetStore
from .color import daylight, luminance_from_energy
from .exceptions import UnsupportedOptionError
from .types import BaseISETObject
from .utils import DEFAULT_WAVE, blackbody, energy_to_quanta, interp_spectra, param_format, quanta_to_energy

_ILLUMINANT_FILE_MAP = {
    "d65": "D65.mat",
    "d50": "D50.mat",
    "tungsten": "Tungsten.mat",
    "fluorescent": "Fluorescent.mat",
    "illuminantc": "illuminantC.mat",
}


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


def _wave_or_default(wave: Any | None) -> np.ndarray:
    if wave is None:
        return np.asarray(DEFAULT_WAVE, dtype=float).reshape(-1)
    wave_array = np.asarray(wave, dtype=float).reshape(-1)
    return np.asarray(DEFAULT_WAVE, dtype=float).reshape(-1) if wave_array.size == 0 else wave_array


def _scale_energy_to_luminance(
    energy: Any,
    wave: np.ndarray,
    luminance_cd_m2: float,
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    energy_array = np.asarray(energy, dtype=float)
    current_luminance = np.asarray(luminance_from_energy(energy_array, wave, asset_store=asset_store), dtype=float)
    target = float(luminance_cd_m2)
    return energy_array * (target / np.maximum(current_luminance, np.finfo(float).tiny))


def _resample_wave_last(values: np.ndarray, source_wave: np.ndarray, target_wave: np.ndarray) -> np.ndarray:
    wave_first = np.moveaxis(np.asarray(values, dtype=float), -1, 0)
    resampled = interp_spectra(np.asarray(source_wave, dtype=float), wave_first, np.asarray(target_wave, dtype=float))
    return np.moveaxis(np.asarray(resampled, dtype=float), 0, -1)


def _base_illuminant(name: str, wave: np.ndarray) -> BaseISETObject:
    illuminant = BaseISETObject(name=name, type="illuminant")
    illuminant.fields["wave"] = np.asarray(wave, dtype=float).reshape(-1)
    illuminant.fields["comment"] = ""
    illuminant.data["photons"] = np.zeros(wave.size, dtype=float)
    return illuminant


def illuminant_create(
    il_name: str = "d65",
    wave: Any | None = None,
    *args: Any,
    asset_store: AssetStore | None = None,
) -> BaseISETObject:
    """Create a MATLAB-style illuminant object."""

    store = _store(asset_store)
    if _is_empty_dispatch_placeholder(il_name):
        il_name = "d65"
    wave_nm = _wave_or_default(wave)
    name_key = param_format(il_name)
    illuminant = _base_illuminant(str(il_name), wave_nm)

    def _scalar_arg(index: int, default: float) -> float:
        if len(args) <= index or _is_empty_dispatch_placeholder(args[index]):
            return default
        return float(args[index])

    if name_key in _ILLUMINANT_FILE_MAP:
        luminance = _scalar_arg(0, 100.0)
        _, energy = store.load_illuminant(_ILLUMINANT_FILE_MAP[name_key], wave_nm=wave_nm)
        energy = _scale_energy_to_luminance(energy, wave_nm, luminance, asset_store=store)
        illuminant.name = str(il_name)
    elif name_key in {"equalenergy", "white", "uniform"}:
        luminance = _scalar_arg(0, 100.0)
        energy = _scale_energy_to_luminance(np.ones(wave_nm.size, dtype=float), wave_nm, luminance, asset_store=store)
        illuminant.name = str(il_name)
    elif name_key in {"equalphoton", "equalphotons"}:
        luminance = _scalar_arg(0, 100.0)
        energy = _scale_energy_to_luminance(
            quanta_to_energy(np.ones(wave_nm.size, dtype=float), wave_nm),
            wave_nm,
            luminance,
            asset_store=store,
        )
        illuminant.name = str(il_name)
    elif name_key in {"blackbody"}:
        temperature_k = _scalar_arg(0, 5000.0)
        luminance = _scalar_arg(1, 100.0)
        energy = np.asarray(blackbody(wave_nm, temperature_k, kind="energy"), dtype=float).reshape(-1)
        energy = _scale_energy_to_luminance(energy, wave_nm, luminance, asset_store=store)
        illuminant.name = f"blackbody-{temperature_k:.0f}"
    elif name_key in {"555nm", "monochromatic"}:
        luminance = _scalar_arg(0, 100.0)
        energy = np.zeros(wave_nm.size, dtype=float)
        idx = int(np.argmin(np.abs(wave_nm - 555.0)))
        energy[idx] = 1.0
        energy = _scale_energy_to_luminance(energy, wave_nm, luminance, asset_store=store)
        illuminant.name = str(il_name)
    elif name_key in {"gaussian"}:
        center = 550.0
        sd = 20.0
        peak_energy = 25.0
        if args:
            if len(args) == 1 and _is_empty_dispatch_placeholder(args[0]):
                args = ()
            if len(args) == 1 and isinstance(args[0], dict):
                params = args[0]
                if not _is_empty_dispatch_placeholder(params.get("center")):
                    center = float(params.get("center"))
                if not _is_empty_dispatch_placeholder(params.get("sd")):
                    sd = float(params.get("sd"))
                peak_field = params.get("peakEnergy", params.get("peak energy"))
                if not _is_empty_dispatch_placeholder(peak_field):
                    peak_energy = float(peak_field)
            else:
                if len(args) % 2 != 0:
                    raise ValueError("illuminantCreate('gaussian', ...) optional arguments must be a params dict or key/value pairs.")
                for index in range(0, len(args), 2):
                    key = param_format(str(args[index]))
                    raw_value = args[index + 1]
                    if _is_empty_dispatch_placeholder(raw_value):
                        continue
                    if key == "center":
                        center = float(raw_value)
                    elif key == "sd":
                        sd = float(raw_value)
                    elif key == "peakenergy":
                        peak_energy = float(raw_value)
                    else:
                        raise KeyError(f"Unknown gaussian illuminant parameter: {args[index]}")
        energy = peak_energy * np.exp(-0.5 * np.square((wave_nm - center) / sd))
        illuminant.name = f"Gaussian {center:.0f}"
    elif name_key in {"daylight"}:
        cct_k = _scalar_arg(0, 6500.0)
        luminance = _scalar_arg(1, 100.0)
        energy = np.asarray(daylight(wave_nm, cct_k, "energy", asset_store=store), dtype=float).reshape(-1)
        energy = _scale_energy_to_luminance(energy, wave_nm, luminance, asset_store=store)
        illuminant.name = f"daylight-{cct_k:.0f}"
    else:
        raise UnsupportedOptionError("illuminantCreate", il_name)

    illuminant.fields["wave"] = wave_nm
    illuminant.data["photons"] = np.asarray(energy_to_quanta(energy, wave_nm), dtype=float)
    return illuminant


def illuminant_get(
    illuminant: BaseISETObject,
    param: str,
    *args: Any,
    asset_store: AssetStore | None = None,
) -> Any:
    """Get a parameter value from a MATLAB-style illuminant object."""

    key = param_format(param)
    wave = np.asarray(illuminant.fields["wave"], dtype=float).reshape(-1)
    photons = np.asarray(illuminant.data.get("photons", np.zeros(wave.size, dtype=float)), dtype=float)

    if key == "name":
        return illuminant.name
    if key == "type":
        return illuminant.type
    if key == "photons":
        if len(args) == 0:
            return photons.copy()
        target_wave = np.asarray(args[0], dtype=float).reshape(-1)
        if photons.ndim == 1:
            return np.asarray(interp_spectra(wave, photons, target_wave), dtype=float).reshape(-1)
        return _resample_wave_last(photons, wave, target_wave)
    if key == "energy":
        if photons.ndim == 3:
            return np.asarray(quanta_to_energy(photons, wave), dtype=float)
        return np.asarray(quanta_to_energy(photons.reshape(-1), wave), dtype=float).reshape(-1)
    if key == "wave":
        return wave.copy()
    if key == "nwave":
        return int(wave.size)
    if key == "luminance":
        value = np.asarray(luminance_from_energy(illuminant_get(illuminant, "energy"), wave, asset_store=_store(asset_store)), dtype=float)
        return float(value) if value.ndim == 0 else value
    if key == "spatialsize":
        return np.asarray(photons.shape, dtype=int)
    if key == "comment":
        return illuminant.fields.get("comment", "")
    if key in {"format", "illuminantformat"}:
        if photons.ndim == 3 and photons.shape[-1] == wave.size:
            return "spatial spectral"
        return "spectral"
    raise UnsupportedOptionError("illuminantGet", param)


def illuminant_set(
    illuminant: BaseISETObject,
    param: str,
    value: Any,
    *args: Any,
    asset_store: AssetStore | None = None,
) -> BaseISETObject:
    """Set a parameter value on a MATLAB-style illuminant object."""

    key = param_format(param)
    updated = illuminant.clone()

    if key == "name":
        updated.name = str(value)
        return updated
    if key == "type":
        if param_format(str(value)) != "illuminant":
            raise ValueError("Type must be illuminant.")
        updated.type = "illuminant"
        return updated
    if key == "photons":
        updated.data["photons"] = np.asarray(value, dtype=float)
        return updated
    if key == "energy":
        wave = np.asarray(updated.fields["wave"], dtype=float).reshape(-1)
        updated.data["photons"] = np.asarray(energy_to_quanta(np.asarray(value, dtype=float), wave), dtype=float)
        return updated
    if key in {"wave", "wavelength"}:
        old_wave = np.asarray(updated.fields["wave"], dtype=float).reshape(-1)
        new_wave = np.asarray(value, dtype=float).reshape(-1)
        if np.array_equal(old_wave, new_wave):
            return updated
        photons = np.asarray(updated.data.get("photons", np.zeros(old_wave.size, dtype=float)), dtype=float)
        updated.fields["wave"] = new_wave
        if photons.size != 0:
            if photons.ndim == 1:
                fill = float(np.min(photons)) * 1e-3
                updated.data["photons"] = np.asarray(
                    interp_spectra(old_wave, photons, new_wave, left=fill, right=fill),
                    dtype=float,
                ).reshape(-1)
            elif photons.ndim == 3:
                updated.data["photons"] = _resample_wave_last(photons, old_wave, new_wave)
            else:
                raise ValueError("Unsupported illuminant photon shape for wave interpolation.")
        return updated
    if key == "comment":
        updated.fields["comment"] = str(value)
        return updated
    raise UnsupportedOptionError("illuminantSet", param)


def illuminant_modernize(
    illuminant: Any,
    *,
    asset_store: AssetStore | None = None,
) -> BaseISETObject:
    """Convert a legacy illuminant struct into the current headless object format."""

    if isinstance(illuminant, BaseISETObject):
        return illuminant.clone()
    if not isinstance(illuminant, dict):
        raise ValueError("illuminantModernize expects an illuminant object or legacy illuminant dict.")

    if "data" not in illuminant:
        raise ValueError("No illuminant data.")

    wave = illuminant.get("wavelength")
    if wave is None:
        spectrum = illuminant.get("spectrum")
        if isinstance(spectrum, dict):
            wave = spectrum.get("wave", spectrum.get("wavelength"))
    if wave is None:
        raise ValueError("No wavelength or spectrum.wavelength slot.")

    data = illuminant["data"]
    if isinstance(data, dict):
        if "photons" in data:
            energy = np.asarray(data["photons"], dtype=float)
        elif "energy" in data:
            energy = np.asarray(data["energy"], dtype=float)
        else:
            raise ValueError("No illuminant data.")
    else:
        energy = np.asarray(data, dtype=float)

    wave_array = np.asarray(wave, dtype=float).reshape(-1)
    modern = _base_illuminant(str(illuminant.get("name", "illuminant")), wave_array)
    modern.data["photons"] = np.asarray(energy_to_quanta(energy, wave_array), dtype=float)

    comment = illuminant.get("comment")
    if comment is not None:
        modern.fields["comment"] = str(comment)

    name = illuminant.get("name")
    if name is None:
        from .metrics import spd_to_cct

        cct = float(np.asarray(spd_to_cct(wave_array, energy, asset_store=_store(asset_store)), dtype=float).reshape(-1)[0])
        modern.name = f"CCT {cct:.0f}"
    else:
        modern.name = str(name)

    return modern


def illuminant_read(
    ill_p: Any | None = None,
    light_name: str | None = None,
    wave: Any | None = None,
    luminance: float | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the requested illuminant spectral radiance in energy units."""

    store = _store(asset_store)
    if _is_empty_dispatch_placeholder(ill_p):
        name = "d65" if _is_empty_dispatch_placeholder(light_name) else str(light_name)
        wave_nm = _wave_or_default(wave)
        luminance_cd_m2 = 100.0 if _is_empty_dispatch_placeholder(luminance) else float(luminance)
        if param_format(name) == "blackbody":
            illuminant = illuminant_create(name, wave_nm, 6500.0, luminance_cd_m2, asset_store=store)
        else:
            illuminant = illuminant_create(name, wave_nm, luminance_cd_m2, asset_store=store)
    else:
        if isinstance(ill_p, BaseISETObject):
            illuminant = ill_p.clone()
        elif isinstance(ill_p, dict):
            fallback_name = "d65" if _is_empty_dispatch_placeholder(light_name) else str(light_name)
            raw_name = ill_p.get("name", fallback_name)
            name = fallback_name if _is_empty_dispatch_placeholder(raw_name) else str(raw_name)
            fallback_luminance = 100.0 if _is_empty_dispatch_placeholder(luminance) else float(luminance)
            raw_luminance = ill_p.get("luminance", fallback_luminance)
            luminance_cd_m2 = fallback_luminance if _is_empty_dispatch_placeholder(raw_luminance) else float(raw_luminance)
            spectrum = ill_p.get("spectrum", {})
            if not isinstance(spectrum, dict):
                spectrum = {}
            raw_wave = spectrum.get("wave", wave)
            wave_nm = _wave_or_default(raw_wave)
            if param_format(name) == "blackbody":
                raw_temperature = ill_p.get("temperature", 6500.0)
                temperature_k = 6500.0 if _is_empty_dispatch_placeholder(raw_temperature) else float(raw_temperature)
                illuminant = illuminant_create(name, wave_nm, temperature_k, luminance_cd_m2, asset_store=store)
            else:
                illuminant = illuminant_create(name, wave_nm, luminance_cd_m2, asset_store=store)
        else:
            raise ValueError("illuminantRead expects an illuminant parameter dict or illuminant object.")

    return (
        np.asarray(illuminant_get(illuminant, "energy", asset_store=store), dtype=float),
        np.asarray(illuminant_get(illuminant, "wave"), dtype=float).reshape(-1),
    )


illuminantCreate = illuminant_create
illuminantGet = illuminant_get
illuminantSet = illuminant_set
illuminantModernize = illuminant_modernize
illuminantRead = illuminant_read
