"""Shared numerical utilities."""

from __future__ import annotations

import base64
import copy
import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, rotate as ndi_rotate, shift as ndi_shift, zoom
from scipy.optimize import fmin
from scipy.signal import convolve2d

DEFAULT_WAVE = np.arange(400.0, 701.0, 10.0, dtype=float)

_PLANCK = 6.626176e-34
_LIGHT_SPEED = 299792458.0
_BOLTZMANN = 1.380662e-23

_PARAMETER_OTYPE_PREFIXES = {
    "scene": "scene",
    "oi": "oi",
    "optics": "optics",
    "wvf": "wvf",
    "sensor": "sensor",
    "pixel": "pixel",
    "vci": "ip",
    "ip": "ip",
    "display": "display",
    "l3": "l3",
}

_PARAMETER_OTYPE_UNIQUE = {
    "scene": {
        "objectdistance",
        "meanluminance",
        "luminance",
        "spatialsupportlinear",
        "spatialsupport",
        "radiancehline",
        "radiancevline",
        "hlineradiance",
        "vlineradiance",
        "luminancehline",
        "luminancevline",
        "hlineluminance",
        "vlineluminance",
        "illuminantcomment",
        "roiilluminantphotons",
        "roimeanilluminantphotons",
        "roiilluminantenergy",
        "roimeanilluminantenergy",
        "illuminanthlinephotons",
        "illuminantvlinephotons",
        "hlineilluminantphotons",
        "vlineilluminantphotons",
        "illuminanthlineenergy",
        "illuminantvlineenergy",
        "hlineilluminantenergy",
        "vlineilluminantenergy",
        "roimeanluminance",
        "chromaticity",
        "roichromaticitymean",
        "illuminant",
        "illuminantname",
        "illuminantenergy",
        "illuminantphotons",
        "illuminantxyz",
        "illuminantwave",
        "illuminantcomment",
        "illuminantformat",
    },
    "oi": {
        "optics",
        "opticsmodel",
        "irradiancehline",
        "irradiancevline",
        "irradianceenergyhline",
        "irradianceenergyvline",
        "hline",
        "vline",
        "hlineirradiance",
        "vlineirradiance",
        "hlineenergy",
        "vlineenergy",
        "hlineirradianceenergy",
        "vlineirradianceenergy",
        "illuminancehline",
        "illuminancevline",
        "hlineilluminance",
        "vlineilluminance",
        "chromaticity",
        "roichromaticitymean",
        "roiilluminance",
        "roimeanilluminance",
        "diffusermethod",
        "diffuserblur",
        "psfstruct",
        "sampledrtpsf",
        "psfsampleangles",
        "psfanglestep",
        "psfimageheights",
        "raytraceopticsname",
    },
    "optics": {
        "fnumber",
        "effectivefnumber",
        "focallength",
        "power",
        "imagedistance",
        "imageheight",
        "imagewidth",
        "numericalaperture",
        "aperturedameter",
        "apertureradius",
        "magnification",
        "pupilmagnification",
        "offaxismethod",
        "cos4thmethod",
        "cos4thdata",
        "otfdata",
        "otfsize",
        "otffx",
        "otffy",
        "otfsupport",
        "psfdata",
        "psfspacing",
        "psfsupport",
        "incoherentcutoffspatialfrequency",
        "maxincoherentcutoffspatialfrequency",
        "rtname",
        "raytrace",
        "rtopticsprogram",
        "rtlensfile",
        "rteffectivefnumber",
        "rtfnumber",
        "rtmagnification",
        "rtreferencewavelength",
        "rtobjectdistance",
        "rtfieldofview",
        "rteffectivefocallength",
        "rtpsf",
        "rtpsfdata",
        "rtpsfsize",
        "rtpsfwavelength",
        "rtpsffieldheight",
        "rtpsfsamplespacing",
        "rtpsfsupport",
        "rtpsfsupportrow",
        "rtpsfsupportcol",
        "rtotfdata",
        "rtrelillum",
        "rtrifunction",
        "rtriwavelength",
        "rtrifieldheight",
        "rtgeometry",
        "rtgeomfunction",
        "rtgeomwavelength",
        "rtgeomfieldheight",
        "rtgeommaxfieldheight",
    },
    "wvf": {
        "zcoeffs",
        "constantsampleintervaldomain",
        "refsizeoffieldmm",
    },
    "sensor": {
        "matchoi",
        "chiefrayangle",
        "chiefrayangledegrees",
        "sensoretendue",
        "microlens",
        "ml",
        "mlens",
        "ulens",
        "microlensoffset",
        "mloffset",
        "microlensoffsetmicrons",
        "volts",
        "voltage",
        "voltimages",
        "digitalvalue",
        "digitalvalues",
        "electrons",
        "electron",
        "electronsperarea",
        "dvorvolts",
        "digitalorvolts",
        "dvorvoltsroielectrons",
        "roivolts",
        "roidata",
        "roidatav",
        "roidatavolts",
        "roielectrons",
        "roidatae",
        "roidataelectrons",
        "roidv",
        "roidigitalcount",
        "roivoltsmean",
        "roielectronsmean",
        "chromaticity",
        "roichromaticitymean",
        "hlinevolts",
        "hlineelectrons",
        "hlinedv",
        "vlinevolts",
        "vlineelectrons",
        "vlinedv",
        "responseratio",
        "volts2maxratio",
        "responsedr",
        "dr",
        "drdb20",
        "dynamicrange",
        "diffusionmtf",
        "analoggain",
        "ag",
        "analogoffset",
        "ao",
        "sensordynamicrange",
        "quantization",
        "quantizationstructure",
        "nbits",
        "maxdigital",
        "maxdigitalvalue",
        "bits",
        "maxoutput",
        "roi",
        "roilocs",
        "roirect",
        "lut",
        "quantizatonlut",
        "quantizationlut",
        "qmethod",
        "quantizationmethod",
        "color",
        "filterspectra",
        "colorfilters",
        "filtertransmissivities",
        "infraredfilter",
        "irfilter",
        "spatialsupport",
        "cfapattern",
        "cfaname",
        "unitblockrows",
        "unitblockcols",
        "cfasize",
        "unitblockconfig",
        "sensorspectrum",
        "wavelengthresolution",
        "binwidth",
        "numberofwavelengthsamples",
        "filternames",
        "filternamescellarray",
        "filtercolornamescellarray",
        "filternamescell",
        "nfilters",
        "filtercolorletters",
        "filtercolorletterscell",
        "filterplotcolors",
        "patterncolors",
        "spectralqe",
        "sensorspectralsr",
        "pattern",
        "dsnusigma",
        "dsnulevel",
        "sigmaoffsetfpn",
        "offsetfpn",
        "offset",
        "offsetsd",
        "offsetnoisevalue",
        "sigmadsnu",
        "prnusigma",
        "prnulevel",
        "sigmagainfpn",
        "gainfpn",
        "gain",
        "gainsd",
        "gainnoisevalue",
        "sigmaprnu",
        "fpnparameters",
        "fpn",
        "fpnoffsetgain",
        "fpnoffsetandgain",
        "dsnuimage",
        "offsetfpnimage",
        "prnuimage",
        "gainfpnimage",
        "columnfpnparameters",
        "columnfpn",
        "columnfixedpatternnoise",
        "colfpn",
        "columndsnu",
        "columnfpnoffset",
        "colfpnoffset",
        "coldsnu",
        "columnprnu",
        "columnfpngain",
        "colfpngain",
        "colprnu",
        "coloffsetfpnvector",
        "coloffsetfpn",
        "coloffset",
        "colgainfpnvector",
        "colgainfpn",
        "colgain",
        "blacklevel",
        "zerolevel",
        "noiseflag",
        "shotnoiseflag",
        "reusenoise",
        "noiseseed",
        "pixel",
        "integrationtime",
        "integrationtimes",
        "exptime",
        "exptimes",
        "exposuretimes",
        "exposuretime",
        "expduration",
        "exposureduration",
        "exposuredurations",
        "uniqueintegrationtimes",
        "uniqueexptime",
        "uniqueexptimes",
        "centralexposure",
        "geometricmeanexposuretime",
        "exposuremethod",
        "expmethod",
        "nexposures",
        "exposureplane",
        "cds",
        "correlateddoublesampling",
        "autoexp",
        "autoexposure",
        "automaticexposure",
        "autoexpsoure",
        "exposuretime",
        "uniqueexptime",
        "exposureplane",
        "cds",
        "vignetting",
        "vignettingflag",
        "pixelvignetting",
        "sensorvignetting",
        "bareetendue",
        "sensorbareetendue",
        "nomicrolensetendue",
        "vignettingname",
        "ngridsamples",
        "pixelsamples",
        "nsamplesperpixel",
        "npixelsamplesforcomputing",
        "spatialsamplesperpixel",
        "responsetype",
        "sensormovement",
        "eyemovement",
        "movementpositions",
        "sensorpositions",
        "framesperposition",
        "framesperpositions",
        "exposuretimesperposition",
        "etimeperpos",
        "sensorpositionsx",
        "sensorpositionsy",
        "human",
        "conetype",
        "humanconetype",
        "densities",
        "humanconedensities",
        "conexy",
        "conelocs",
        "xy",
        "humanconelocs",
        "rseed",
        "humanrseed",
        "humanconeseed",
        "chartparameters",
        "cornerpoints",
        "chartcornerpoints",
        "chartcorners",
        "chartrects",
        "chartrectangles",
        "currentrect",
        "chartcurrentrect",
        "mccrecthandles",
        "mcccornerpoints",
        "consistency",
        "sensorconsistency",
        "sensorcompute",
        "sensorcomputemethod",
        "metadatascenename",
        "scenename",
        "scene_name",
        "metadataopticsname",
        "metadatalensname",
        "metadatalens",
        "lens",
        "metadatasensorname",
        "metadatacrop",
    },
    "pixel": {
        "pixelsize",
        "pixelsizeconstantfillfactor",
        "pixelsizekeepfillfactor",
        "pixelsizesamefillfactor",
        "widthheight",
        "widthandheight",
        "sizeconstantfillfactor",
        "sizekeepfillfactor",
        "sizesamefillfactor",
        "pixelwidth",
        "pixelwidthmeters",
        "pixelheight",
        "pixelheightmeters",
        "pixelwidthgap",
        "widthbetweenpixels",
        "pixelheightgap",
        "heightbetweenpixels",
        "widthgap",
        "heightgap",
        "xyspacing",
        "pixelarea",
        "pixeldepth",
        "pixeldepthmeters",
        "layerthickness",
        "layerthicknesses",
        "stackheight",
        "refractiveindex",
        "refractiveindices",
        "n",
        "pixelspectrum",
        "pixelwavelength",
        "pixelwavelengthsamples",
        "pixelbinwidth",
        "pixelnwave",
        "pdwidth",
        "pdheight",
        "photodetectorwidth",
        "photodetectorheight",
        "photodetectorsize",
        "pdwidthandheight",
        "pdxpos",
        "photodetectorxposition",
        "pdypos",
        "photodetectoryposition",
        "pdposition",
        "pddimension",
        "pdsize",
        "fillfactor",
        "pdarea",
        "pixelspectralqe",
        "pdspectralqe",
        "spectralsr",
        "quantumefficiency",
        "pixelqe",
        "pixelquantumefficiency",
        "photodetectorquantumefficiency",
        "photodetectorspectralquantumefficiency",
        "conversiongain",
        "conversiongainvpelectron",
        "conversiongainvperelectron",
        "voltsperelectron",
        "voltageswing",
        "vswing",
        "saturationvoltage",
        "maxvoltage",
        "wellcapacity",
        "darkcurrentdensity",
        "darkcurrent",
        "darkcurrentperpixel",
        "darkvolt",
        "darkvoltage",
        "darkvolts",
        "darkvoltageperpixelpersec",
        "darkvoltageperpixel",
        "voltspersecond",
        "darkelectrons",
        "readnoise",
        "readnoiseelectrons",
        "readstandarddeviationelectrons",
        "readnoisevolts",
        "readstandarddeviationvolts",
        "readnoisestdvolts",
        "readnoisemillivolts",
        "pdspectralsr",
        "pixelspectralsr",
        "sr",
        "pixeldr",
        "pixeldynamicrange",
    },
    "ip": {
        "chromaticity",
        "roichromaticitymean",
        "render",
        "colorbalance",
        "colorbalancemethod",
        "demosaic",
        "demosaicmethod",
        "colorconversion",
        "colorconversionmethod",
        "internalcolorspace",
        "internalcolormatchingfunciton",
        "display",
        "displayxyz",
        "displayxy",
        "displaywhitepoint",
        "displaymaxluminance",
        "displayspd",
        "displaygamma",
        "displaymaxrgb",
        "displaydpi",
        "displayviewingdistance",
        "l3",
    },
    "l3": {
        "trainingilluminant",
        "clusters",
        "filters",
        "sensorpatches",
    },
    "asset": {
        "assetobject",
        "assetbranch",
        "assetlight",
    },
}


def param_format(value: Any) -> Any:
    """Mirror ISETCam's ieParamFormat behavior."""

    if isinstance(value, (int, float, complex, np.number, np.ndarray, bool)):
        return value
    if isinstance(value, str):
        return value.lower().replace(" ", "")
    if isinstance(value, list):
        formatted = list(value)
        for index in range(0, len(formatted), 2):
            formatted[index] = param_format(formatted[index])
        return formatted
    if isinstance(value, tuple):
        formatted = list(value)
        for index in range(0, len(formatted), 2):
            formatted[index] = param_format(formatted[index])
        return tuple(formatted)
    return value


def ie_parameter_otype(param: str) -> tuple[str, str | None]:
    """Infer the ISET object type associated with a parameter string."""

    if not param:
        raise ValueError("param is required")

    normalized = param_format(param)
    direct = _PARAMETER_OTYPE_PREFIXES.get(normalized)
    if direct is not None:
        return direct, None

    exact_unique_match: tuple[str, str] | None = None
    for object_type, unique_params in _PARAMETER_OTYPE_UNIQUE.items():
        if normalized in unique_params:
            exact_unique_match = (object_type, normalized)
            break

    positions = [idx for idx in (param.find(" "), param.find("/"), param.find("_")) if idx >= 0]
    if positions:
        pos = min(positions)
        prefix = _PARAMETER_OTYPE_PREFIXES.get(param_format(param[:pos]))
        if prefix is not None:
            remainder = param_format(param[(pos + 1) :])
            prefix_unique = _PARAMETER_OTYPE_UNIQUE.get(prefix)
            if (
                prefix_unique is None
                or remainder in prefix_unique
                or exact_unique_match is None
                or exact_unique_match[0] == prefix
            ):
                return prefix, remainder
            return exact_unique_match
    if exact_unique_match is not None:
        return exact_unique_match
    return "", normalized


def split_prefixed_parameter(parameter: Any, prefixes: tuple[str, ...]) -> tuple[str | None, str]:
    """Split MATLAB-style object-prefixed parameter names.

    Examples:
        ``display spd`` -> (``display``, ``spd``)
        ``display/spd`` -> (``display``, ``spd``)
        ``displayspd`` -> (``display``, ``spd``)
    """

    if not isinstance(parameter, str):
        return None, ""

    normalized = param_format(parameter)
    ordered_prefixes = sorted((param_format(prefix) for prefix in prefixes), key=len, reverse=True)
    for prefix in ordered_prefixes:
        if normalized == prefix:
            return prefix, ""
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix) :]
            remainder = re.sub(r"^[/\\-]+", "", remainder)
            return prefix, remainder
    return None, normalized


def spectral_step(wave_nm: NDArray[np.float64]) -> float:
    wave_nm = np.asarray(wave_nm, dtype=float).reshape(-1)
    if wave_nm.size < 2:
        return 1.0
    return float(np.mean(np.diff(wave_nm)))


def unit_frequency_list(sample_count: int) -> NDArray[np.float64]:
    """Mirror ISETCam's unitFrequencyList() normalization and DC placement."""

    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    if count % 2:
        middle = (count + 1) // 2
    else:
        middle = (count // 2) + 1
    coordinates = np.arange(1, count + 1, dtype=float)
    coordinates = coordinates - coordinates[middle - 1]
    max_coordinate = np.max(np.abs(coordinates))
    if max_coordinate <= 0.0:
        return np.zeros_like(coordinates)
    return coordinates / max_coordinate


def ie_unit_scale_factor(unit_name: str) -> float:
    """Mirror MATLAB ieUnitScaleFactor() conversions from meters/seconds/radians."""

    if unit_name is None or str(unit_name).strip() == "":
        raise ValueError("Unit name must be defined.")

    normalized = param_format(unit_name)
    scale_map = {
        "nm": 1.0e9,
        "nanometer": 1.0e9,
        "nanometers": 1.0e9,
        "micron": 1.0e6,
        "micrometer": 1.0e6,
        "um": 1.0e6,
        "microns": 1.0e6,
        "mm": 1.0e3,
        "millimeter": 1.0e3,
        "millimeters": 1.0e3,
        "cm": 1.0e2,
        "centimeter": 1.0e2,
        "centimeters": 1.0e2,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "km": 1.0e-3,
        "kilometer": 1.0e-3,
        "kilometers": 1.0e-3,
        "inch": 39.37007874,
        "inches": 39.37007874,
        "foot": 3.280839895,
        "feet": 3.280839895,
        "s": 1.0,
        "second": 1.0,
        "sec": 1.0,
        "ms": 1.0e3,
        "millisecond": 1.0e3,
        "us": 1.0e6,
        "microsecond": 1.0e6,
        "degrees": 180.0 / np.pi,
        "deg": 180.0 / np.pi,
        "arcmin": (180.0 / np.pi) * 60.0,
        "minutes": (180.0 / np.pi) * 60.0,
        "min": (180.0 / np.pi) * 60.0,
        "arcsec": (180.0 / np.pi) * 3600.0,
    }
    if normalized not in scale_map:
        raise ValueError("Unknown spatial unit specification.")
    return float(scale_map[normalized])


def dpi2mperdot(dpi: Any, unit: str = "um") -> float | NDArray[np.float64]:
    """Convert dots-per-inch to meters-per-dot scaled to the requested MATLAB unit."""

    dpi_values = np.asarray(dpi, dtype=float)
    if np.any(dpi_values <= 0.0):
        raise ValueError("dpi must be positive.")
    scaled = (0.0254 / dpi_values) * ie_unit_scale_factor(unit)
    if scaled.ndim == 0:
        return float(scaled)
    return np.asarray(scaled, dtype=float)


def ie_dpi2_mperdot(dpi: Any, unit: str = "um") -> float | NDArray[np.float64]:
    """Alias for MATLAB ieDpi2Mperdot()."""

    return dpi2mperdot(dpi, unit)


def ie_space_to_amp(
    pos: NDArray[np.float64] | list[float] | tuple[float, ...],
    data: NDArray[np.float64] | list[float] | tuple[float, ...],
    scale_data: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror MATLAB ieSpace2Amp() FFT amplitude support."""

    pos_array = np.asarray(pos, dtype=float).reshape(-1)
    data_array = np.asarray(data, dtype=float).reshape(-1)
    if pos_array.size == 0:
        raise ValueError("You must define positions.")
    if data_array.size == 0:
        raise ValueError("You must define a vector of data.")
    if pos_array.size != data_array.size:
        raise ValueError("pos and data must have the same number of samples.")

    f_data = np.abs(np.fft.fft(data_array))
    if scale_data:
        peak = float(np.max(f_data)) if f_data.size else 0.0
        if peak > 0.0:
            f_data = f_data / peak

    unit_per_image = float(np.max(pos_array) - np.min(pos_array))
    if unit_per_image <= 0.0:
        raise ValueError("pos must span a non-zero spatial support.")

    n_samp = data_array.size
    freq = np.arange(n_samp, dtype=float) / unit_per_image
    n_freq = int(np.rint((n_samp - 1) / 2.0))
    return freq[:n_freq].copy(), np.asarray(f_data[:n_freq], dtype=float)


def sample2space(
    r_samples: NDArray[np.float64] | list[float] | tuple[float, ...],
    c_samples: NDArray[np.float64] | list[float] | tuple[float, ...],
    row_delta: float,
    col_delta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror MATLAB sample2space() centered-support behavior."""

    row_samples = np.asarray(r_samples, dtype=float).reshape(-1)
    col_samples = np.asarray(c_samples, dtype=float).reshape(-1)
    r_center = float(np.mean(row_samples))
    c_center = float(np.mean(col_samples))
    return (
        np.asarray((row_samples - r_center) * float(row_delta), dtype=float),
        np.asarray((col_samples - c_center) * float(col_delta), dtype=float),
    )


def space2sample(
    r_microns: NDArray[np.float64] | list[float] | tuple[float, ...],
    c_microns: NDArray[np.float64] | list[float] | tuple[float, ...],
    pixel_height: float,
    pixel_width: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror MATLAB's obsolete space2sample() zero-based offset convention."""

    row_positions = np.asarray(r_microns, dtype=float).reshape(-1)
    col_positions = np.asarray(c_microns, dtype=float).reshape(-1)
    row = row_positions / float(pixel_height)
    col = col_positions / float(pixel_width)
    return (
        np.asarray(row - row[0], dtype=float),
        np.asarray(col - col[0], dtype=float),
    )


def _normalize_legacy_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError("Legacy key/value arguments must come in pairs.")
    normalized: dict[str, Any] = {}
    for index in range(0, len(args), 2):
        normalized[param_format(str(args[index]))] = args[index + 1]
    for key, value in kwargs.items():
        if value is None:
            continue
        normalized[param_format(str(key))] = value
    return normalized


def _coerce_matlab_shape(
    size_spec: Any,
    *,
    scalar_means_square: bool = True,
) -> tuple[int, ...]:
    """Convert MATLAB size arguments into a Python shape tuple."""

    if isinstance(size_spec, tuple):
        if len(size_spec) == 0:
            return ()
        if len(size_spec) == 1:
            size_spec = size_spec[0]
        else:
            values = np.asarray(size_spec, dtype=int).reshape(-1)
            if np.any(values < 0):
                raise ValueError("Size arguments must be non-negative.")
            return tuple(int(value) for value in values)

    if np.isscalar(size_spec):
        values = np.asarray([size_spec], dtype=int)
    else:
        values = np.asarray(size_spec, dtype=int).reshape(-1)

    if np.any(values < 0):
        raise ValueError("Size arguments must be non-negative.")
    if values.size == 0:
        return ()
    if values.size == 1 and scalar_means_square:
        value = int(values[0])
        return (value, value)
    return tuple(int(value) for value in values)


def _load_image_array(image_or_path: Any) -> NDArray[Any]:
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as image:
            return np.asarray(image)
    return np.asarray(image_or_path)


def ie_find_files(root_dir: str | Path, ext: str) -> list[str]:
    """Recursively find files with a given extension under a root directory."""

    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    suffix = str(ext)
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    matches = [
        str(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name.endswith(suffix)
    ]
    return matches


ieFindFiles = ie_find_files


def ie_tone(*args: Any, frequency: float = 256.0, amplitude: float = 0.2, duration: float = 0.25) -> tuple[dict[str, float], NDArray[np.float64]]:
    """Synthesize MATLAB-style ieTone() output headlessly without playback."""

    options: dict[str, Any] = {}
    if len(args) == 1 and isinstance(args[0], dict):
        for key, value in args[0].items():
            options[param_format(key)] = value
    else:
        options = _normalize_legacy_kwargs(
            args,
            {
                "Frequency": None,
                "Amplitude": None,
                "Duration": None,
            },
        )

    frequency_value = float(options.get("frequency", frequency))
    amplitude_value = float(options.get("amplitude", amplitude))
    duration_value = float(options.get("duration", duration))

    fs = 8192.0
    n_samples = max(int(np.floor(fs * duration_value)), 0)
    times = np.arange(n_samples + 1, dtype=float) / fs
    tone = amplitude_value * np.sin(2.0 * np.pi * frequency_value * times)
    params = {
        "Amplitude": amplitude_value,
        "Duration": duration_value,
        "Frequency": frequency_value,
    }
    return params, np.asarray(tone, dtype=float)


ieTone = ie_tone


def max2(matrix: Any, userows: Any | None = None, usecols: Any | None = None) -> tuple[float, NDArray[np.int_]]:
    array = np.asarray(matrix)
    if array.ndim > 2:
        raise ValueError("M must be a 2-d array or a vector")
    if array.ndim == 1:
        array = array.reshape(1, -1)

    n_rows, n_cols = array.shape
    row_indices = (
        np.arange(1, n_rows + 1, dtype=int)
        if userows is None or np.asarray(userows).size == 0
        else np.unique(np.asarray(userows, dtype=int).reshape(-1))
    )
    col_indices = (
        np.arange(1, n_cols + 1, dtype=int)
        if usecols is None or np.asarray(usecols).size == 0
        else np.unique(np.asarray(usecols, dtype=int).reshape(-1))
    )

    if np.any(row_indices < 1) or np.any(row_indices > n_rows):
        raise ValueError("userows must be a valid set of indices into the rows of M")
    if np.any(col_indices < 1) or np.any(col_indices > n_cols):
        raise ValueError("usecols must be a valid set of indices into the columns of M")

    restricted = array[row_indices - 1][:, col_indices - 1]
    column_maxima = np.max(restricted, axis=0)
    row_argmax = np.argmax(restricted, axis=0)
    column_argmax = int(np.argmax(column_maxima))
    value = float(column_maxima[column_argmax])
    ij = np.array([row_indices[row_argmax[column_argmax]], col_indices[column_argmax]], dtype=int)
    return value, ij


def min2(matrix: Any, userows: Any | None = None, usecols: Any | None = None) -> tuple[float, NDArray[np.int_]]:
    array = np.asarray(matrix)
    if array.ndim > 2:
        raise ValueError("M must be a 2-d array or a vector")
    if array.ndim == 1:
        array = array.reshape(1, -1)

    n_rows, n_cols = array.shape
    row_indices = (
        np.arange(1, n_rows + 1, dtype=int)
        if userows is None or np.asarray(userows).size == 0
        else np.unique(np.asarray(userows, dtype=int).reshape(-1))
    )
    col_indices = (
        np.arange(1, n_cols + 1, dtype=int)
        if usecols is None or np.asarray(usecols).size == 0
        else np.unique(np.asarray(usecols, dtype=int).reshape(-1))
    )

    if np.any(row_indices < 1) or np.any(row_indices > n_rows):
        raise ValueError("userows must be a valid set of indices into the rows of M")
    if np.any(col_indices < 1) or np.any(col_indices > n_cols):
        raise ValueError("usecols must be a valid set of indices into the columns of M")

    restricted = array[row_indices - 1][:, col_indices - 1]
    column_minima = np.min(restricted, axis=0)
    row_argmin = np.argmin(restricted, axis=0)
    column_argmin = int(np.argmin(column_minima))
    value = float(column_minima[column_argmin])
    ij = np.array([row_indices[row_argmin[column_argmin]], col_indices[column_argmin]], dtype=int)
    return value, ij


def _is_struct_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_struct_array(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(_is_struct_mapping(item) for item in value)


def _is_empty_struct_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, list):
        return all(_is_empty_struct_value(item) for item in value)
    return False


def _copy_struct_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return copy.deepcopy(value)


def _callable_signature(value: Any) -> Any:
    code = getattr(value, "__code__", None)
    closure = getattr(value, "__closure__", None)
    closure_values = None
    if closure is not None:
        closure_values = tuple(cell.cell_contents for cell in closure)
    return (
        getattr(value, "__module__", None),
        getattr(value, "__qualname__", None),
        getattr(code, "co_code", None),
        getattr(code, "co_consts", None),
        getattr(code, "co_names", None),
        getattr(value, "__defaults__", None),
        closure_values,
    )


def _values_equal_with_tol(value1: Any, value2: Any, tol: float) -> bool:
    if _is_struct_mapping(value1) or _is_struct_mapping(value2):
        return False
    if _is_struct_array(value1) or _is_struct_array(value2):
        return False

    if isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
        if not (isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray)):
            return False
        if value1.shape != value2.shape or value1.dtype != value2.dtype:
            return False
        if np.issubdtype(value1.dtype, np.floating) or np.issubdtype(value1.dtype, np.complexfloating):
            return bool(np.allclose(value1, value2, atol=tol, rtol=0.0, equal_nan=True))
        return bool(np.array_equal(value1, value2))

    if isinstance(value1, (list, tuple)) or isinstance(value2, (list, tuple)):
        if type(value1) is not type(value2):
            return False
        if len(value1) != len(value2):
            return False
        return all(_values_equal_with_tol(left, right, tol) for left, right in zip(value1, value2))

    if callable(value1) or callable(value2):
        if not (callable(value1) and callable(value2)):
            return False
        return _callable_signature(value1) == _callable_signature(value2)

    if isinstance(value1, np.generic) or isinstance(value2, np.generic):
        if not (isinstance(value1, np.generic) and isinstance(value2, np.generic)):
            return False
        if value1.dtype != value2.dtype:
            return False
        if np.issubdtype(value1.dtype, np.floating) or np.issubdtype(value1.dtype, np.complexfloating):
            return bool(abs(value1.item() - value2.item()) < tol)
        return bool(value1.item() == value2.item())

    if isinstance(value1, (float, complex)) or isinstance(value2, (float, complex)):
        if type(value1) is not type(value2):
            return False
        return bool(abs(value1 - value2) < tol)

    if type(value1) is not type(value2):
        return False
    return bool(value1 == value2)


def _comp_struct_recursive(s1: Any, s2: Any, tol: float) -> tuple[Any, Any, Any]:
    if _is_struct_mapping(s1) and _is_struct_mapping(s2):
        common: dict[str, Any] = {}
        d1: dict[str, Any] = {}
        d2: dict[str, Any] = {}

        for key in s1.keys():
            if key not in s2:
                d1[key] = _copy_struct_value(s1[key])
                continue
            child_common, child_d1, child_d2 = _comp_struct_recursive(s1[key], s2[key], tol)
            if not _is_empty_struct_value(child_common):
                common[key] = child_common
            if not _is_empty_struct_value(child_d1):
                d1[key] = child_d1
            if not _is_empty_struct_value(child_d2):
                d2[key] = child_d2

        for key in s2.keys():
            if key not in s1:
                d2[key] = _copy_struct_value(s2[key])

        return (
            None if len(common) == 0 else common,
            None if len(d1) == 0 else d1,
            None if len(d2) == 0 else d2,
        )

    if _is_struct_array(s1) and _is_struct_array(s2):
        max_length = max(len(s1), len(s2))
        common_items: list[Any] = []
        d1_items: list[Any] = []
        d2_items: list[Any] = []
        for index in range(max_length):
            if index < len(s1) and index < len(s2):
                child_common, child_d1, child_d2 = _comp_struct_recursive(s1[index], s2[index], tol)
            elif index < len(s1):
                child_common, child_d1, child_d2 = None, _copy_struct_value(s1[index]), None
            else:
                child_common, child_d1, child_d2 = None, None, _copy_struct_value(s2[index])
            common_items.append(child_common)
            d1_items.append(child_d1)
            d2_items.append(child_d2)

        return (
            None if all(_is_empty_struct_value(item) for item in common_items) else common_items,
            None if all(_is_empty_struct_value(item) for item in d1_items) else d1_items,
            None if all(_is_empty_struct_value(item) for item in d2_items) else d2_items,
        )

    if _values_equal_with_tol(s1, s2, tol):
        return _copy_struct_value(s1), None, None
    return None, _copy_struct_value(s1), _copy_struct_value(s2)


def comp_struct(
    s1: Any,
    s2: Any,
    prt: int | None = 0,
    pse: int | None = 0,
    tol: float | None = 1e-20,
    n1: str | None = None,
    n2: str | None = None,
) -> tuple[Any, Any, Any]:
    """Compare nested struct-like Python data using the legacy MATLAB contract."""

    del prt, pse, n1, n2
    if s2 is None:
        raise ValueError("s2 is required")
    tolerance = 1e-20 if tol is None else float(tol)
    return _comp_struct_recursive(s1, s2, tolerance)


compStruct = comp_struct


def _stringify_list_struct_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value)
    return str(value)


def list_struct(s1: Any, v: int = 0, n1: str = "s1") -> list[str]:
    """Return headless field listings for the legacy MATLAB list_struct() helper."""

    if v not in (0, 1):
        v = 0

    lines: list[str] = []
    if _is_struct_mapping(s1):
        for key in s1.keys():
            lines.extend(list_struct(s1[key], v, f"{n1}.{key}"))
        return lines

    if _is_struct_array(s1):
        for index, item in enumerate(s1, start=1):
            lines.extend(list_struct(item, v, f"{n1}({index})"))
        return lines

    if isinstance(s1, list) and s1 and all(_is_struct_mapping(item) for item in s1):
        for index, item in enumerate(s1, start=1):
            for key in item.keys():
                lines.extend(list_struct(item[key], v, f"{n1}{{{index}}}.{key}"))
        return lines

    if v:
        lines.append(f"Field:\t{n1} = {_stringify_list_struct_value(s1)}")
    else:
        lines.append(f"Field:\t{n1}")
    return lines


listStruct = list_struct


def _prepare_zernike_nm(n: Any, m: Any) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    n_array = np.asarray(n)
    m_array = np.asarray(m)
    if n_array.ndim > 1 or m_array.ndim > 1:
        raise ValueError("N and M must be vectors.")
    n_vector = np.asarray(n_array, dtype=int).reshape(-1)
    m_vector = np.asarray(m_array, dtype=int).reshape(-1)
    if n_vector.size != m_vector.size:
        raise ValueError("N and M must be the same length.")
    if np.any((n_vector - m_vector) % 2 != 0):
        raise ValueError("All N and M must differ by multiples of 2 (including 0).")
    if np.any(np.abs(m_vector) > n_vector):
        raise ValueError("Each M must be less than or equal to its corresponding N.")
    return n_vector, m_vector


def _prepare_zernike_radius(r: Any, *, allow_theta: bool = False, theta: Any | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    r_array = np.asarray(r, dtype=float)
    if r_array.ndim > 1:
        raise ValueError("R must be a vector.")
    r_vector = r_array.reshape(-1)
    if np.any((r_vector < 0.0) | (r_vector > 1.0)):
        raise ValueError("All R must be between 0 and 1.")
    if not allow_theta:
        return r_vector, None

    theta_array = np.asarray(theta, dtype=float)
    if theta_array.ndim > 1:
        raise ValueError("R and THETA must be vectors.")
    theta_vector = theta_array.reshape(-1)
    if r_vector.size != theta_vector.size:
        raise ValueError("The number of R- and THETA-values must be equal.")
    return r_vector, theta_vector


def _zernike_rpowers(n_vector: NDArray[np.int_], m_vector: NDArray[np.int_]) -> NDArray[np.int_]:
    powers: list[int] = []
    for n_value, m_value in zip(n_vector.tolist(), m_vector.tolist()):
        powers.extend(range(abs(m_value), n_value + 1, 2))
    return np.asarray(sorted(set(powers)), dtype=int)


def _zernike_power_matrix(r_vector: NDArray[np.float64], powers: NDArray[np.int_]) -> NDArray[np.float64]:
    if powers.size == 0:
        return np.zeros((r_vector.size, 0), dtype=float)
    matrix = np.empty((r_vector.size, powers.size), dtype=float)
    for index, power in enumerate(powers.tolist()):
        matrix[:, index] = 1.0 if power == 0 else np.power(r_vector, power)
    return matrix


def _parse_zernike_norm_flag(nflag: Any | None) -> bool:
    if nflag is None:
        return False
    if isinstance(nflag, str) and nflag.lower() == "norm":
        return True
    raise ValueError("Unrecognized normalization flag.")


def zernpol(n: Any, m: Any, r: Any, nflag: Any | None = None) -> NDArray[np.float64]:
    """Evaluate radial Zernike polynomials using the vendored MATLAB contract."""

    n_vector, m_vector = _prepare_zernike_nm(n, m)
    if np.any(m_vector < 0):
        raise ValueError("All M must be positive.")
    r_vector, _ = _prepare_zernike_radius(r)
    isnorm = _parse_zernike_norm_flag(nflag)

    powers = _zernike_rpowers(n_vector, m_vector)
    power_matrix = _zernike_power_matrix(r_vector, powers)
    z = np.zeros((r_vector.size, n_vector.size), dtype=float)

    for column, (n_value, m_value) in enumerate(zip(n_vector.tolist(), m_vector.tolist())):
        s_values = range(0, (n_value - m_value) // 2 + 1)
        pows = list(range(n_value, m_value - 1, -2))
        for s_value, power in zip(reversed(tuple(s_values)), reversed(pows)):
            coefficient = (
                (-1) ** s_value
                * math.factorial(n_value - s_value)
                / (
                    math.factorial(s_value)
                    * math.factorial((n_value - m_value) // 2 - s_value)
                    * math.factorial((n_value + m_value) // 2 - s_value)
                )
            )
            power_index = int(np.where(powers == power)[0][0])
            z[:, column] = z[:, column] + coefficient * power_matrix[:, power_index]
        if isnorm:
            z[:, column] = z[:, column] * math.sqrt(2.0 * (n_value + 1.0))

    return z


def zernfun(n: Any, m: Any, r: Any, theta: Any, nflag: Any | None = None) -> NDArray[np.float64]:
    """Evaluate Zernike functions on the unit circle using the vendored MATLAB contract."""

    n_vector, m_vector = _prepare_zernike_nm(n, m)
    r_vector, theta_vector = _prepare_zernike_radius(r, allow_theta=True, theta=theta)
    isnorm = _parse_zernike_norm_flag(nflag)

    radial = zernpol(n_vector, np.abs(m_vector), r_vector, "norm" if isnorm else None)
    if isnorm:
        radial = radial / np.sqrt(2.0 * (n_vector.reshape(1, -1) + 1.0))

    z = np.asarray(radial, dtype=float)
    positive = m_vector > 0
    negative = m_vector < 0
    if np.any(positive):
        z[:, positive] = z[:, positive] * np.cos(np.outer(theta_vector, np.abs(m_vector[positive])))
    if np.any(negative):
        z[:, negative] = z[:, negative] * np.sin(np.outer(theta_vector, np.abs(m_vector[negative])))
    if isnorm:
        scale = np.sqrt((1.0 + (m_vector != 0).astype(float)) * (n_vector + 1.0) / math.pi)
        z = z * scale.reshape(1, -1)
    return z


def zernfun2(p: Any, r: Any, theta: Any, nflag: Any | None = None) -> NDArray[np.float64]:
    """Evaluate the first 36 single-index Zernike functions."""

    p_array = np.asarray(p)
    if p_array.ndim > 1:
        raise ValueError("Input P must be vector.")
    p_vector = np.asarray(p_array, dtype=int).reshape(-1)
    if np.any(p_vector < 0) or np.any(p_vector > 35):
        raise ValueError("ZERNFUN2 only computes the first 36 Zernike functions (P = 0 to 35).")
    n_vector = np.ceil((-3.0 + np.sqrt(9.0 + 8.0 * p_vector)) / 2.0).astype(int)
    m_vector = (2 * p_vector - n_vector * (n_vector + 2)).astype(int)
    return zernfun(n_vector, m_vector, r, theta, nflag)


_IE_HASH_MISSING = object()
_IE_HASH_VERSION = 4
_IE_HASH_DATE = [2018, 5, 19]
_IE_HASH_METHOD_ALIASES = {
    "MD2": "md2",
    "MD5": "md5",
    "SHA1": "sha1",
    "SHA-1": "sha1",
    "SHA224": "sha224",
    "SHA-224": "sha224",
    "SHA256": "sha256",
    "SHA-256": "sha256",
    "SHA384": "sha384",
    "SHA-384": "sha384",
    "SHA512": "sha512",
    "SHA-512": "sha512",
}
_IE_HASH_CANONICAL_METHODS = [
    ("MD2", "md2"),
    ("MD5", "md5"),
    ("SHA-1", "sha1"),
    ("SHA-224", "sha224"),
    ("SHA-256", "sha256"),
    ("SHA-384", "sha384"),
    ("SHA-512", "sha512"),
]


def _ie_hash_version() -> dict[str, Any]:
    available = {name.lower() for name in hashlib.algorithms_available}
    methods = [label for label, algorithm in _IE_HASH_CANONICAL_METHODS if algorithm in available]
    return {
        "HashVersion": _IE_HASH_VERSION,
        "Date": list(_IE_HASH_DATE),
        "HashMethod": methods,
    }


def _ie_hash_normalize_method(method: str) -> tuple[str, str]:
    key = str(method).strip().upper()
    algorithm = _IE_HASH_METHOD_ALIASES.get(key)
    if algorithm is None:
        normalized = key.replace("_", "").replace("-", "")
        algorithm = _IE_HASH_METHOD_ALIASES.get(normalized)
    if algorithm is None:
        raise ValueError(f"Invalid hashing algorithm: [{method}]")
    try:
        hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Invalid hashing algorithm: [{method}]") from exc
    for label, candidate in _IE_HASH_CANONICAL_METHODS:
        if candidate == algorithm:
            return label, algorithm
    return key, algorithm


def _ie_hash_scalar_array(data: Any) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 0:
        return array.reshape((1, 1))
    if array.ndim == 1:
        return array.reshape((1, array.size))
    return array


def _ie_hash_class_name(array: np.ndarray) -> str:
    dtype = np.asarray(array).dtype
    if dtype == np.bool_:
        return "logical"
    if dtype.kind == "u":
        return f"uint{dtype.itemsize * 8}"
    if dtype.kind == "i":
        return f"int{dtype.itemsize * 8}"
    if dtype.kind == "f":
        if dtype.itemsize == 4:
            return "single"
        if dtype.itemsize == 8:
            return "double"
    if dtype.kind == "c":
        if dtype.itemsize == 8:
            return "single"
        if dtype.itemsize == 16:
            return "double"
    raise ValueError(f"Data type not handled: {dtype}")


def _ie_hash_shape(array: np.ndarray) -> tuple[int, ...]:
    if array.ndim == 0:
        return (1, 1)
    if array.ndim == 1:
        return (1, int(array.size))
    return tuple(int(value) for value in array.shape)


def _ie_hash_metadata_bytes(class_name: str, shape: tuple[int, ...]) -> bytes:
    dimensions = np.asarray([len(shape), *shape], dtype="<u8")
    return class_name.encode("ascii") + dimensions.tobytes()


def _ie_hash_numeric_bytes(array: np.ndarray) -> bytes:
    flat = np.ravel(np.asarray(array), order="F")
    native = flat.astype(flat.dtype.newbyteorder("="), copy=False)
    return native.tobytes()


def _ie_hash_char_units(text: str) -> np.ndarray:
    return np.frombuffer(str(text).encode("utf-16le"), dtype="<u2")


def _ie_hash_function_payload(func: Any) -> dict[str, Any]:
    return {
        "function": getattr(func, "__name__", repr(func)),
        "module": getattr(func, "__module__", ""),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", repr(func))),
    }


def _ie_hash_update_array(hasher: Any, data: Any) -> None:
    if isinstance(data, Mapping):
        hasher.update(_ie_hash_metadata_bytes("struct", (1, 1)))
        for key in sorted(str(name) for name in data.keys()):
            hasher.update(key.encode("utf-8"))
            _ie_hash_update_array(hasher, data[key])
        return

    if isinstance(data, (list, tuple)):
        values = list(data)
        hasher.update(_ie_hash_metadata_bytes("cell", (1, len(values))))
        for item in values:
            _ie_hash_update_array(hasher, item)
        return

    if isinstance(data, (str, np.str_)):
        units = _ie_hash_char_units(str(data))
        hasher.update(_ie_hash_metadata_bytes("char", (1, int(units.size))))
        if units.size:
            hasher.update(units.tobytes())
        return

    if isinstance(data, (bytes, bytearray, memoryview)):
        values = np.frombuffer(bytes(data), dtype=np.uint8)
        hasher.update(_ie_hash_metadata_bytes("uint8", (1, int(values.size))))
        if values.size:
            hasher.update(values.tobytes())
        return

    if isinstance(data, np.ndarray):
        if data.dtype == object or data.dtype.kind in {"U", "S"}:
            shape = _ie_hash_shape(data)
            hasher.update(_ie_hash_metadata_bytes("cell", shape))
            for item in np.ravel(data, order="F"):
                _ie_hash_update_array(hasher, item.item() if hasattr(item, "item") else item)
            return
        array = _ie_hash_scalar_array(data)
        class_name = _ie_hash_class_name(array)
        shape = _ie_hash_shape(array)
        hasher.update(_ie_hash_metadata_bytes(class_name, shape))
        if array.size == 0:
            return
        if np.iscomplexobj(array):
            hasher.update(_ie_hash_numeric_bytes(np.real(array)))
            hasher.update(_ie_hash_numeric_bytes(np.imag(array)))
        elif array.dtype == np.bool_:
            hasher.update(np.ravel(array.astype(np.uint8, copy=False), order="F").tobytes())
        else:
            hasher.update(_ie_hash_numeric_bytes(array))
        return

    if isinstance(data, (bool, np.bool_)):
        _ie_hash_update_array(hasher, np.asarray(data, dtype=np.bool_))
        return

    if isinstance(data, (int, np.integer, float, np.floating, complex, np.complexfloating)):
        _ie_hash_update_array(hasher, np.asarray(data))
        return

    if callable(data):
        hasher.update(_ie_hash_metadata_bytes("function_handle", (1, 1)))
        _ie_hash_update_array(hasher, _ie_hash_function_payload(data))
        return

    if hasattr(data, "__dict__"):
        hasher.update(_ie_hash_metadata_bytes(data.__class__.__name__, (1, 1)))
        _ie_hash_update_array(hasher, vars(data))
        return

    raise ValueError(f"Cannot create elementary array for type: {type(data).__name__}")


def _ie_hash_update_bin(hasher: Any, data: Any) -> None:
    if isinstance(data, (str, np.str_)):
        units = _ie_hash_char_units(str(data))
        if units.size:
            hasher.update(units.tobytes())
        return

    if isinstance(data, (bytes, bytearray, memoryview)):
        payload = bytes(data)
        if payload:
            hasher.update(payload)
        return

    array = np.asarray(data)
    if array.dtype.kind in {"U", "S", "O"}:
        raise ValueError("[Bin] input needs data type: numeric, CHAR, LOGICAL, STRING.")
    if array.size == 0:
        return
    if np.iscomplexobj(array):
        hasher.update(_ie_hash_numeric_bytes(np.real(array)))
        hasher.update(_ie_hash_numeric_bytes(np.imag(array)))
        return
    if array.dtype == np.bool_:
        hasher.update(np.ravel(array.astype(np.uint8, copy=False), order="F").tobytes())
        return
    hasher.update(_ie_hash_numeric_bytes(array))


def _ie_hash_prepare_ascii_data(data: Any) -> np.ndarray:
    if not isinstance(data, (str, np.str_)):
        raise ValueError("ASCII method: Data must be a CHAR or scalar STRING.")
    text = str(data)
    values = np.asarray([ord(character) & 0xFF for character in text], dtype=np.uint8)
    return values.reshape((1, values.size))


def _ie_hash_parse_input(data: Any, *args: Any) -> tuple[str, str, bool, bool, Any]:
    method = "MD5"
    out_format = "hex"
    is_file = False
    is_bin = False

    options: list[Any] = list(args)
    if len(options) == 1 and isinstance(options[0], Mapping):
        struct_options = options[0]
        options = []
        for key in ("Input", "Method", "OutFormat"):
            if key in struct_options:
                options.append(struct_options[key])

    for option in options:
        if not isinstance(option, str):
            raise ValueError("[Opt] must be a struct or chars.")
        lowered = option.lower()
        if lowered == "file":
            is_file = True
            continue
        if lowered in {"bin", "binary"}:
            array = np.asarray(data)
            if isinstance(data, (bytes, bytearray, memoryview)):
                pass
            elif array.dtype.kind in {"U", "S", "O"} and not isinstance(data, (str, np.str_)):
                raise ValueError("[Bin] input needs data type: numeric, CHAR, LOGICAL, STRING.")
            is_bin = True
            continue
        if lowered == "array":
            is_bin = False
            continue
        if lowered in {"asc", "ascii"}:
            data = _ie_hash_prepare_ascii_data(data)
            is_bin = True
            continue
        if lowered == "hex":
            out_format = "HEX" if option.startswith("H") else "hex"
            continue
        if lowered in {"double", "uint8", "short", "base64"}:
            out_format = lowered
            continue
        method = option.upper()

    return method, out_format, is_file, is_bin, data


def ie_hash(data: Any = _IE_HASH_MISSING, *args: Any) -> Any:
    """Create a MATLAB-compatible hash for arrays, structs, cells, or files."""

    if data is _IE_HASH_MISSING:
        return _ie_hash_version()

    method_label, out_format, is_file, is_bin, prepared = _ie_hash_parse_input(data, *args)
    _, algorithm = _ie_hash_normalize_method(method_label)
    hasher = hashlib.new(algorithm)

    if is_file:
        file_path = Path(prepared)
        with file_path.open("rb") as stream:
            while True:
                chunk = stream.read(1_000_000)
                if not chunk:
                    break
                hasher.update(chunk)
    elif is_bin:
        _ie_hash_update_bin(hasher, prepared)
    else:
        _ie_hash_update_array(hasher, prepared)

    digest = hasher.digest()
    if out_format == "hex":
        return digest.hex()
    if out_format == "HEX":
        return digest.hex().upper()
    if out_format == "double":
        return np.frombuffer(digest, dtype=np.uint8).astype(float)
    if out_format == "uint8":
        return np.frombuffer(digest, dtype=np.uint8).copy()
    if out_format == "base64":
        return base64.b64encode(digest).decode("ascii")
    if out_format == "short":
        return base64.b64encode(digest).decode("ascii").rstrip("=")
    raise ValueError("[Opt.Format] must be: HEX, hex, uint8, double, base64.")


ieHash = ie_hash


def ie_contains(value: Any, pattern: str) -> bool | NDArray[np.bool_]:
    """Return whether a string or string collection contains a substring."""

    if isinstance(value, (list, tuple)):
        return np.asarray(
            [
                isinstance(item, (str, np.str_)) and pattern in str(item)
                for item in value
            ],
            dtype=bool,
        )
    if not isinstance(value, (str, np.str_)):
        return False
    return pattern in str(value)


ieContains = ie_contains


def replace_nan(data: Any, value: float | int) -> NDArray[Any]:
    """Replace NaN entries with a constant value."""

    array = np.array(data, copy=True)
    mask = np.isnan(np.asarray(array, dtype=float))
    array[mask] = value
    return array


replaceNaN = replace_nan


def struct2pairs(structure: Any) -> list[Any]:
    """Convert a scalar struct-like mapping into alternating key/value pairs."""

    if isinstance(structure, (list, tuple)):
        return list(structure)
    if not isinstance(structure, Mapping):
        raise ValueError("Input must be a scalar struct or cell")
    pairs: list[Any] = []
    for key, value in structure.items():
        pairs.extend([key, value])
    return pairs


def append_struct(a_struct: Mapping[str, Any] | None, b_struct: Mapping[str, Any] | None) -> dict[str, Any]:
    """Append two mappings while keeping values from the latter mapping on duplicates."""

    if not a_struct:
        return {} if b_struct is None else dict(b_struct)
    if not b_struct:
        return dict(a_struct)
    combined = dict(a_struct)
    for key, value in b_struct.items():
        combined[key] = value
    return combined


appendStruct = append_struct


def cell_delete(cell_values: list[Any] | tuple[Any, ...], delete_list: Any) -> list[Any]:
    """Delete 1-based entries from a cell-like Python list."""

    values = list(cell_values)
    indices = np.asarray(delete_list, dtype=int).reshape(-1)
    if indices.size == 0:
        return values
    if np.any(indices < 1) or np.any(indices > len(values)):
        raise ValueError("Bad dList")
    for index in sorted(set(indices.tolist()), reverse=True):
        del values[index - 1]
    return values


cellDelete = cell_delete


def cell_merge(*args: Any) -> list[Any]:
    """Merge cell-like Python sequences into a single list."""

    if len(args) == 0:
        return []
    merged: list[Any] = []
    for value in args:
        if value is None or value == []:
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError("All arguments must be cell arrays")
        merged.extend(list(value))
    return merged


cellMerge = cell_merge


def checkfields(structure: Any, *fields: str) -> bool:
    """Check whether nested mapping or attribute fields exist."""

    if len(fields) == 0:
        raise ValueError("At least one field is required.")
    current = structure
    for field in fields:
        if isinstance(current, Mapping):
            if field not in current:
                return False
            current = current[field]
            continue
        if hasattr(current, field):
            current = getattr(current, field)
            continue
        return False
    return True


def compare_fields(a: Any, b: Any) -> bool:
    """Compare top-level mapping fields for exact equality."""

    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False
    if sorted(a.keys()) != sorted(b.keys()):
        return False
    for key in sorted(a.keys()):
        if not _values_equal_with_tol(a[key], b[key], 0.0):
            return False
    return True


compareFields = compare_fields


def _gather_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _gather_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_gather_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_gather_value(item) for item in value)
    if hasattr(value, "get") and callable(getattr(value, "get")) and not isinstance(value, (str, bytes)):
        try:
            return value.get()
        except Exception:
            return value
    return value


def gather_struct(obj: Any) -> Any:
    """Recursively gather struct-like data into ordinary Python containers."""

    return _gather_value(obj)


gatherStruct = gather_struct


def _value_summary(value: Any) -> str | None:
    if isinstance(value, str):
        return repr(value.strip())
    if isinstance(value, np.ndarray) and value.size == 1:
        return str(value.reshape(-1)[0])
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    return None


def _ie_struct_compare_recursive(struct_a: Any, struct_b: Any, prefix: str) -> list[str]:
    differences: list[str] = []

    if hasattr(struct_a, "__dict__") and not isinstance(struct_a, Mapping):
        struct_a = vars(struct_a)
    if hasattr(struct_b, "__dict__") and not isinstance(struct_b, Mapping):
        struct_b = vars(struct_b)

    if not isinstance(struct_a, Mapping) or not isinstance(struct_b, Mapping):
        if not _values_equal_with_tol(struct_a, struct_b, 0.0):
            summary_a = _value_summary(struct_a)
            summary_b = _value_summary(struct_b)
            message = f"Path: {prefix} | Difference: Values are not equal"
            if summary_a is not None and summary_b is not None:
                message = f"{message} (Value A: {summary_a}, Value B: {summary_b})"
            differences.append(message)
        return differences

    fields_a = set(struct_a.keys())
    fields_b = set(struct_b.keys())
    missing_in_b = sorted(fields_a - fields_b)
    missing_in_a = sorted(fields_b - fields_a)
    if missing_in_b:
        differences.append(
            f"Path: {prefix} | Difference: Field(s) missing in second struct: {', '.join(missing_in_b)}"
        )
    if missing_in_a:
        differences.append(
            f"Path: {prefix} | Difference: Field(s) missing in first struct: {', '.join(missing_in_a)}"
        )

    for field in sorted(fields_a & fields_b):
        new_prefix = f"{prefix}.{field}"
        value_a = struct_a[field]
        value_b = struct_b[field]
        if isinstance(value_a, Mapping) and isinstance(value_b, Mapping):
            differences.extend(_ie_struct_compare_recursive(value_a, value_b, new_prefix))
            continue
        if isinstance(value_a, list) and isinstance(value_b, list):
            if len(value_a) != len(value_b):
                differences.append(
                    f"Path: {new_prefix} | Difference: Cell array size mismatch ({len(value_a)} vs {len(value_b)})"
                )
            else:
                for index, (item_a, item_b) in enumerate(zip(value_a, value_b), start=1):
                    if isinstance(item_a, Mapping) and isinstance(item_b, Mapping):
                        differences.extend(
                            _ie_struct_compare_recursive(item_a, item_b, f"{new_prefix}{{{index}}}")
                        )
                    elif not _values_equal_with_tol(item_a, item_b, 0.0):
                        differences.append(
                            f"Path: {new_prefix}{{{index}}} | Difference: Cell element values are not equal"
                        )
            continue
        if not _values_equal_with_tol(value_a, value_b, 0.0):
            summary_a = _value_summary(value_a)
            summary_b = _value_summary(value_b)
            message = f"Path: {new_prefix} | Difference: Values are not equal"
            if summary_a is not None and summary_b is not None:
                message = f"{message} (Value A: {summary_a}, Value B: {summary_b})"
            differences.append(message)

    return differences


def ie_struct_compare(struct_a: Any, struct_b: Any, prefix: str | None = None) -> tuple[list[str], Any, Any, Any]:
    """Recursively compare two struct-like values and return difference summaries."""

    root_prefix = "structA" if prefix is None else str(prefix)
    differences = _ie_struct_compare_recursive(struct_a, struct_b, root_prefix)
    common, d1, d2 = comp_struct(struct_a, struct_b)
    return differences, common, d1, d2


ieStructCompare = ie_struct_compare


def _is_programming_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def ie_struct_remove_empty_field(structure: Mapping[str, Any]) -> dict[str, Any]:
    """Remove empty-valued fields from a struct-like mapping."""

    return {key: value for key, value in structure.items() if not _is_programming_empty(value)}


ieStructRemoveEmptyField = ie_struct_remove_empty_field


def _to_grayscale(image: Any) -> NDArray[np.float64]:
    image_array = np.asarray(image, dtype=float)
    if image_array.ndim == 2:
        return image_array
    if image_array.ndim == 3 and image_array.shape[2] == 1:
        return np.asarray(image_array[:, :, 0], dtype=float)
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError("Image data must be grayscale or RGB.")
    return np.asarray(
        0.2989 * image_array[:, :, 0] + 0.5870 * image_array[:, :, 1] + 0.1140 * image_array[:, :, 2],
        dtype=float,
    )


def _binarize_grayscale(image: Any) -> NDArray[np.float64]:
    gray = _to_grayscale(image)
    if gray.dtype == np.bool_:
        return np.asarray(gray, dtype=float)
    threshold = 0.5 * (float(np.min(gray)) + float(np.max(gray)))
    return np.asarray(gray > threshold, dtype=float)


def ie_light_list(
    *args: Any,
    wave: NDArray[np.float64] | list[float] | tuple[float, ...] | None = None,
    lightdir: str | Path | None = None,
) -> tuple[list[str], list[NDArray[np.float64]], NDArray[np.int64]]:
    """Mirror MATLAB ieLightList() over the vendored light asset tree."""

    options = _normalize_legacy_kwargs(args, {"wave": wave, "lightdir": lightdir})
    wave_nm = np.arange(400.0, 701.0, 10.0, dtype=float) if options.get("wave") is None else np.asarray(
        options["wave"], dtype=float
    ).reshape(-1)

    from .assets import AssetStore, ie_read_spectra

    store = AssetStore.default()
    root = store.ensure()
    if options.get("lightdir") is None:
        search_root = root / "data" / "lights"
    else:
        candidate = Path(options["lightdir"])
        search_root = candidate if candidate.is_absolute() else root / candidate
    light_files = sorted(search_root.rglob("*.mat"))

    names: list[str] = []
    data: list[NDArray[np.float64]] = []
    n_samples: list[int] = []
    for light_file in light_files:
        if light_file.name == "cct.mat":
            continue
        spectra = np.asarray(ie_read_spectra(light_file.name, wave_nm, asset_store=store), dtype=float)
        spectra[spectra == 0.0] = 1.0e-8
        names.append(light_file.name)
        data.append(spectra)
        n_samples.append(int(spectra.shape[1]))
    return names, data, np.asarray(n_samples, dtype=int)


def ie_reflectance_list(
    *args: Any,
    wave: NDArray[np.float64] | list[float] | tuple[float, ...] | None = None,
) -> tuple[list[str], list[NDArray[np.float64]], NDArray[np.int64]]:
    """Mirror MATLAB ieReflectanceList() over the vendored reflectance asset tree."""

    options = _normalize_legacy_kwargs(args, {"wave": wave})
    wave_nm = np.arange(400.0, 701.0, 10.0, dtype=float) if options.get("wave") is None else np.asarray(
        options["wave"], dtype=float
    ).reshape(-1)

    from .assets import AssetStore, ie_read_spectra

    store = AssetStore.default()
    root = store.ensure()
    search_files = [
        *(root / "data" / "surfaces" / "reflectances").glob("*.mat"),
        *(root / "data" / "surfaces" / "reflectances" / "skin").glob("*.mat"),
        *(root / "data" / "surfaces" / "reflectances" / "esser").glob("esserChart.mat"),
    ]

    names: list[str] = []
    data: list[NDArray[np.float64]] = []
    n_samples: list[int] = []
    for reflectance_file in sorted(search_files):
        if reflectance_file.name == "reflectanceBasis.mat":
            continue
        spectra = np.asarray(ie_read_spectra(reflectance_file.name, wave_nm, asset_store=store), dtype=float)
        if spectra.size == 0 or float(np.max(spectra)) > 1.0:
            continue
        names.append(reflectance_file.name)
        data.append(spectra)
        n_samples.append(int(spectra.shape[1]))
    return names, data, np.asarray(n_samples, dtype=int)


def ie_data_list(
    data_type: str,
    *args: Any,
    wave: NDArray[np.float64] | list[float] | tuple[float, ...] | None = None,
) -> tuple[list[str], list[NDArray[np.float64]], NDArray[np.int64]]:
    """Mirror the implemented MATLAB ieDataList() dispatch surface."""

    options = _normalize_legacy_kwargs(args, {"wave": wave})
    wave_nm = options.get("wave")
    normalized = param_format(data_type)
    if normalized in {"refl", "reflectance"}:
        return ie_reflectance_list(wave=wave_nm)
    if normalized == "light":
        return ie_light_list(wave=wave_nm)
    raise ValueError(f"Unsupported ieDataList type: {data_type}")


def get_middle_matrix(
    m: Any,
    sz: Any,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Mirror MATLAB getMiddleMatrix() centered extraction behavior."""

    matrix = np.asarray(m, dtype=float)
    size_array = np.asarray(matrix.shape, dtype=float)
    center = _matlab_round(size_array / 2.0).astype(int)

    half_sizes = np.asarray(sz, dtype=float).reshape(-1)
    half_sizes = _matlab_round(half_sizes / 2.0).astype(int)
    if half_sizes.size == 1:
        half_sizes = np.array([half_sizes[0], half_sizes[0]], dtype=int)

    row_min = max(1, int(center[0] - half_sizes[0]))
    row_max = min(matrix.shape[0], int(center[0] + half_sizes[0]))
    col_min = max(1, int(center[1] - half_sizes[1]))
    col_max = min(matrix.shape[1], int(center[1] + half_sizes[1]))

    row_slice = slice(row_min - 1, row_max)
    col_slice = slice(col_min - 1, col_max)
    if matrix.ndim >= 3:
        middle = np.asarray(matrix[row_slice, col_slice, :], dtype=float)
    else:
        middle = np.asarray(matrix[row_slice, col_slice], dtype=float)
    return middle, center


def ie_clip(im: Any, lower_bound: Any | None = None, upper_bound: Any | None = None) -> NDArray[np.float64]:
    """Mirror MATLAB ieClip() clipping conventions."""

    clipped = np.asarray(im, dtype=float).copy()
    if lower_bound is None and upper_bound is None:
        lower = 0.0
        upper = 1.0
    elif upper_bound is None:
        bound = abs(float(lower_bound))
        lower = -bound
        upper = bound
    else:
        lower = None if lower_bound is None else float(lower_bound)
        upper = None if upper_bound is None else float(upper_bound)

    if lower is not None:
        clipped[clipped < lower] = lower
    if upper is not None:
        clipped[clipped > upper] = upper
    return clipped


def ie_hwhm_to_sd(h: float, g_dim: int = 2) -> float:
    """Convert half-width-half-max to Gaussian standard deviation."""

    if int(g_dim) == 1:
        return float(h) / (2.0 * np.sqrt(np.log(2.0)))
    if int(g_dim) == 2:
        return float(h) / np.sqrt(2.0 * np.log(2.0))
    raise ValueError(f"Not implemented for {g_dim} dimensional Gaussian.")


def ie_scale(im: Any, b1: Any | None = None, b2: Any | None = None) -> tuple[NDArray[np.float64], float, float]:
    """Mirror MATLAB ieScale() range and peak scaling behavior."""

    data = np.asarray(im, dtype=float)
    mx = float(np.max(data))
    mn = float(np.min(data))

    if b1 is not None and b2 is None:
        scaled = np.asarray(data * (float(b1) / mx), dtype=float)
        return scaled, mn, mx

    if np.isclose(mx, mn):
        normalized = np.zeros_like(data, dtype=float)
    else:
        normalized = (data - mn) / (mx - mn)

    low = 0.0 if b1 is None else float(b1)
    high = 1.0 if b2 is None else float(b2)
    if low >= high:
        raise ValueError("ieScale: bad bounds values.")
    scaled = (high - low) * normalized + low
    return np.asarray(scaled, dtype=float), mn, mx


def ie_scale_columns(X: Any, b1: Any = 1, b2: Any | None = None) -> NDArray[np.float64]:
    """Mirror MATLAB ieScaleColumns() by scaling each column independently."""

    array = np.asarray(X, dtype=float)
    scaled = np.zeros_like(array, dtype=float)
    for column in range(array.shape[1]):
        if b2 is None:
            scaled[:, column] = ie_scale(array[:, column], b1)[0]
        else:
            scaled[:, column] = ie_scale(array[:, column], b1, b2)[0]
    return scaled


def isodd(x: Any) -> bool | NDArray[np.bool_]:
    """Return whether values are odd, following MATLAB isodd() semantics."""

    values = np.asarray(x)
    result = np.mod(values, 2) != 0
    if result.ndim == 0:
        return bool(result)
    return np.asarray(result, dtype=bool)


def rotation_matrix_3d(angle_list: Any, scale: Any | None = None) -> NDArray[np.float64]:
    """Mirror MATLAB rotationMatrix3d() axis-order and scaling behavior."""

    angles = np.asarray(angle_list, dtype=float).reshape(-1)
    if angles.size != 3:
        raise ValueError("Must have 3 angles in the angle list.")

    if scale is None:
        scale_matrix = np.eye(3, dtype=float)
    else:
        scale_values = np.asarray(scale, dtype=float).reshape(-1)
        if scale_values.size == 1:
            scale_matrix = np.eye(3, dtype=float) * float(scale_values[0])
        elif scale_values.size == 3:
            scale_matrix = np.diag(scale_values.astype(float))
        else:
            raise ValueError("Scale must be a scalar or 1x3.")

    tx, ty, tz = [float(value) for value in angles]
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tx), -np.sin(tx)], [0.0, np.sin(tx), np.cos(tx)]],
        dtype=float,
    )
    rot_y = np.array(
        [[np.cos(ty), 0.0, -np.sin(ty)], [0.0, 1.0, 0.0], [np.sin(ty), 0.0, np.cos(ty)]],
        dtype=float,
    )
    rot_z = np.array(
        [[np.cos(tz), -np.sin(tz), 0.0], [np.sin(tz), np.cos(tz), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rot_x @ rot_y @ rot_z @ scale_matrix


def unpadarray(in_array: Any, unpad_size: Any) -> NDArray[np.float64]:
    """Mirror MATLAB unpadarray() symmetric crop behavior."""

    array = np.asarray(in_array, dtype=float)
    padding = np.asarray(unpad_size, dtype=int).reshape(-1)
    if padding.size < 2:
        padding = np.array([padding[0], 0], dtype=int)
    row_slice = slice(int(padding[0]), array.shape[0] - int(padding[0]))
    col_slice = slice(int(padding[1]), array.shape[1] - int(padding[1]))
    if array.ndim == 3:
        return np.asarray(array[row_slice, col_slice, :], dtype=float)
    return np.asarray(array[row_slice, col_slice], dtype=float)


def upper_quad_to_full_matrix(upper_right: Any, n_rows: int, n_cols: int) -> NDArray[np.float64]:
    """Mirror MATLAB upperQuad2FullMatrix() quadrant reflection rules."""

    upper = np.asarray(upper_right, dtype=float)
    _, cols = upper.shape
    upper_left = np.fliplr(upper[:, 1:cols]) if isodd(n_cols) else np.fliplr(upper)
    lower_right = np.flipud(upper[:-1, :]) if isodd(n_rows) else np.flipud(upper)
    lower_left = np.flipud(upper_left[:-1, :]) if isodd(n_rows) else np.flipud(upper_left)
    return np.asarray(np.block([[upper_left, upper], [lower_left, lower_right]]), dtype=float)


def vector_length(m: Any, dim: int | None = None) -> float | NDArray[np.float64]:
    """Mirror MATLAB vectorLength() with NaN-as-zero handling."""

    values = np.asarray(m, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    safe = values.copy()
    safe[np.isnan(safe)] = 0.0
    if dim is None:
        return float(np.sqrt(np.dot(safe.reshape(-1), safe.reshape(-1))))
    return np.sqrt(np.sum(np.square(safe), axis=int(dim) - 1))


def ffndgrid(
    x: Any,
    f: Any,
    delta: Any | None = None,
    limits: Any | None = None,
    aver: Any | None = 1,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Mirror MATLAB ffndgrid() for headless uneven-sample gridding."""

    x_array = np.asarray(x, dtype=float)
    was_row_vector = x_array.ndim == 2 and x_array.shape[0] == 1
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    elif x_array.ndim == 2 and x_array.shape[0] == 1:
        x_array = x_array.reshape(-1, 1)
    elif x_array.ndim != 2:
        raise ValueError("x must be a vector or an N x D coordinate matrix.")

    n_samples, n_dims = x_array.shape
    f_array = np.asarray(f, dtype=float).reshape(-1)
    if f_array.size == 1:
        f_array = np.repeat(f_array, n_samples)
    elif f_array.size != n_samples:
        raise ValueError("The length of f must equal size(x,1).")

    dx = np.full(n_dims, -75.0, dtype=float)
    pad = 0.0
    xyz = np.zeros((2 * n_dims) + 3, dtype=float)
    xyz[0 : 2 * n_dims : 2] = np.min(x_array, axis=0)
    xyz[1 : 2 * n_dims : 2] = np.max(x_array, axis=0)
    xyz[-3] = float(np.min(f_array))
    xyz[-2] = float(np.max(f_array))
    xyz[-1] = 0.0

    if limits is not None:
        limit_array = np.asarray(limits, dtype=float).reshape(-1)
        count = min(limit_array.size, xyz.size)
        valid = ~np.isnan(limit_array[:count])
        xyz[:count][valid] = limit_array[:count][valid]

    if delta is not None:
        delta_array = np.asarray(delta).reshape(-1)
        if delta_array.size == 1:
            delta_array = np.repeat(delta_array, n_dims)
        delta_array = delta_array[: min(delta_array.size, n_dims)]
        valid = ~(np.isnan(np.real(delta_array)) | (np.real(delta_array) == 0))
        if np.any(valid):
            dx[: delta_array.size][valid] = np.real(delta_array)[valid]
            pad = float(np.imag(delta_array[0]))

    x_lower = xyz[0 : 2 * n_dims : 2]
    x_upper = xyz[1 : 2 * n_dims : 2]
    f_lower = float(xyz[-3])
    f_upper = float(xyz[-2])
    min_count = float(xyz[-1])

    negative = dx < 0.0
    if np.any(negative):
        if not np.allclose(dx[negative], _matlab_round(dx[negative])):
            raise ValueError("Some of Nx1,...NxD in delta are not an integer.")
        dx[negative] = (x_upper[negative] - x_lower[negative]) / (np.abs(dx[negative]) - 1.0)

    bin_index = _matlab_round((x_array - x_lower.reshape(1, -1)) / dx.reshape(1, -1)).astype(int) + 1

    xvec: list[NDArray[np.float64]] = []
    grid_shape = np.ones(max(n_dims, 2), dtype=int)
    for dim in range(n_dims):
        step = float(dx[dim])
        count = int(np.floor(((x_upper[dim] - x_lower[dim]) / step) + 0.5)) + 1
        axis = x_lower[dim] + (np.arange(count, dtype=float) * step)
        xvec.append(np.asarray(axis, dtype=float))
        grid_shape[dim] = axis.size

    valid = np.all((bin_index >= 1) & (bin_index <= grid_shape[:n_dims].reshape(1, -1)), axis=1)
    valid &= (f_array >= f_lower) & (f_array <= f_upper)
    bin_index = bin_index[valid, :]
    f_array = f_array[valid]
    n_valid = int(bin_index.shape[0])

    if n_valid == 0:
        empty = np.zeros(tuple(grid_shape[:n_dims]), dtype=float)
        if n_dims == 2:
            empty = empty.T
        elif n_dims == 3:
            empty = np.transpose(empty, (1, 0, 2))
        if was_row_vector:
            empty = empty.reshape(1, -1)
        return np.asarray(empty, dtype=float), xvec

    accum = np.zeros(tuple(grid_shape[:n_dims]), dtype=float)
    counts = np.zeros(tuple(grid_shape[:n_dims]), dtype=float)
    zero_based = tuple((bin_index - 1).T)
    np.add.at(accum, zero_based, f_array)
    np.add.at(counts, zero_based, 1.0)

    if min_count != 0.0 or bool(aver):
        if min_count < 0.0:
            min_count = -min_count * n_valid
        if min_count != 0.0:
            reject = counts <= min_count
            accum[reject] = 0.0
            counts[reject] = 0.0
        if bool(aver):
            nonzero = counts > 0.0
            accum[nonzero] = accum[nonzero] / counts[nonzero]

    if pad != 0.0:
        accum[counts == 0.0] = pad

    if was_row_vector:
        output = accum.reshape(1, -1)
    else:
        output = accum.reshape(tuple(grid_shape[:n_dims]))
        if n_dims == 2:
            output = output.T
        elif n_dims == 3:
            output = np.transpose(output, (1, 0, 2))
    return np.asarray(output, dtype=float), xvec


def ie_compress_data(
    data: Any,
    bit_depth: int,
    mn: float | None = None,
    mx: float | None = None,
) -> tuple[NDArray[np.uint16] | NDArray[np.uint32], float, float]:
    """Mirror MATLAB ieCompressData() quantized compression behavior."""

    data_array = np.asarray(data, dtype=float)
    min_value = float(np.min(data_array)) if mn is None else float(mn)
    max_value = float(np.max(data_array)) if mx is None else float(mx)
    if min_value > max_value:
        raise ValueError("Min/Max error.")

    max_compress = (2**int(bit_depth)) - 1
    if np.isclose(min_value, max_value):
        compressed_values = np.zeros_like(data_array, dtype=float)
    else:
        scale = max_value - min_value
        compressed_values = _matlab_round(max_compress * (data_array - min_value) / scale)

    if int(bit_depth) == 32:
        return np.asarray(compressed_values, dtype=np.uint32), min_value, max_value
    if int(bit_depth) == 16:
        return np.asarray(compressed_values, dtype=np.uint16), min_value, max_value
    raise ValueError("Unknown bit depth.")


def ie_uncompress_data(
    c_data: Any,
    mn: float,
    mx: float,
    bit_depth: int,
) -> NDArray[np.float64]:
    """Mirror MATLAB ieUncompressData() inverse quantization behavior."""

    min_value = float(mn)
    max_value = float(mx)
    if min_value > max_value:
        raise ValueError("Min/Max error.")

    max_compress = float((2**int(bit_depth)) - 1)
    if min_value == max_value:
        scale = max_value
    else:
        scale = max_value - min_value

    values = np.asarray(c_data)
    try:
        return np.asarray((float(scale) / max_compress) * values.astype(float) + float(min_value), dtype=float)
    except MemoryError:
        return np.asarray((np.float32(scale) / np.float32(max_compress)) * values.astype(np.float32) + np.float32(min_value), dtype=np.float32)


ieUncompressData = ie_uncompress_data


def ie_line_align(D1: Any, D2: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror MATLAB ieLineAlign() scale-and-shift alignment."""

    x1 = np.asarray(D1["x"], dtype=float).reshape(-1)
    y1 = np.asarray(D1["y"], dtype=float).reshape(-1)
    x2 = np.asarray(D2["x"], dtype=float).reshape(-1)
    y2 = np.asarray(D2["y"], dtype=float).reshape(-1)

    def _estimate(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        transformed = np.asarray(p[0], dtype=float) * (x2 - float(p[1]))
        order = np.argsort(transformed)
        transformed = transformed[order]
        observed = y2[order]
        unique_x, unique_index = np.unique(transformed, return_index=True)
        observed = observed[unique_index]
        est = np.interp(x1, unique_x, observed, left=np.nan, right=np.nan)
        valid = ~np.isnan(est)
        return float(np.sum((y1[valid] - est[valid]) ** 2)), est

    start = np.array([1.0, 0.0], dtype=float)
    est_p = np.asarray(fmin(lambda p: _estimate(np.asarray(p, dtype=float))[0], start, disp=False), dtype=float)
    _, est_y = _estimate(est_p)
    return est_p, np.asarray(est_y, dtype=float)


def ie_tikhonov(
    A: Any,
    b: Any,
    *args: Any,
    minnorm: float = 0.0,
    smoothness: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror MATLAB ieTikhonov() ridge and smoothness regularization."""

    options = {"minnorm": float(minnorm), "smoothness": float(smoothness)}
    if len(args) % 2 != 0:
        raise ValueError("Legacy key/value arguments must come in pairs.")
    for index in range(0, len(args), 2):
        options[param_format(str(args[index]))] = float(args[index + 1])
    lambda1 = float(options.get("minnorm", 0.0))
    lambda2 = float(options.get("smoothness", 0.0))
    matrix = np.asarray(A, dtype=float)
    vector = np.asarray(b, dtype=float).reshape(-1)
    d2 = np.diff(np.eye(matrix.shape[1], dtype=float), 2, axis=0)
    system = (matrix.T @ matrix) + (lambda1 * np.eye(matrix.shape[1], dtype=float)) + (lambda2 * (d2.T @ d2))
    rhs = matrix.T @ vector
    x = np.linalg.solve(system, rhs)
    x_ols = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    return np.asarray(x, dtype=float), np.asarray(x_ols, dtype=float)


def qinterp2(
    X: Any,
    Y: Any,
    Z: Any,
    xi: Any,
    yi: Any,
    methodflag: int | None = None,
) -> NDArray[np.float64]:
    """Mirror MATLAB qinterp2() for nearest, triangular, and bilinear interpolation."""

    xi_array = np.asarray(xi, dtype=float)
    yi_array = np.asarray(yi, dtype=float)
    if xi_array.shape != yi_array.shape:
        if xi_array.ndim == 1 and yi_array.ndim == 1:
            xi_array, yi_array = np.meshgrid(xi_array, yi_array)
        else:
            raise ValueError(f"{'xi'} and {'yi'} must be equal size")

    x_grid = np.asarray(X, dtype=float)
    y_grid = np.asarray(Y, dtype=float)
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    if x_grid.shape != y_grid.shape:
        raise ValueError(f"{'X'} and {'Y'} must have the same size")

    z_grid = np.asarray(Z, dtype=float)
    library_rows, library_cols = x_grid.shape
    method = 2 if methodflag is None else int(methodflag)

    ndx = 1.0 / float(x_grid[0, 1] - x_grid[0, 0])
    ndy = 1.0 / float(y_grid[1, 0] - y_grid[0, 0])
    xi_index = (xi_array - float(x_grid[0, 0])) * ndx
    yi_index = (yi_array - float(y_grid[0, 0])) * ndy
    zi = np.full(xi_index.shape, np.nan, dtype=float)

    if method == 0:
        rxi = _matlab_round(xi_index).astype(int) + 1
        ryi = _matlab_round(yi_index).astype(int) + 1
        valid = (
            (rxi > 0)
            & (rxi <= library_cols)
            & np.isfinite(rxi)
            & (ryi > 0)
            & (ryi <= library_rows)
            & np.isfinite(ryi)
        )
        zi[valid] = z_grid[ryi[valid] - 1, rxi[valid] - 1]
        return zi

    fxi = np.floor(xi_index).astype(int) + 1
    fyi = np.floor(yi_index).astype(int) + 1
    dfxi = xi_index - fxi + 1
    dfyi = yi_index - fyi + 1
    valid = (
        (fxi > 0)
        & (fxi < library_cols)
        & np.isfinite(fxi)
        & (fyi > 0)
        & (fyi < library_rows)
        & np.isfinite(fyi)
    )

    if method == 1:
        compare = dfxi >= dfyi
        flag1 = valid & compare
        flag2 = valid & ~compare
        zi[flag1] = (
            z_grid[fyi[flag1] - 1, fxi[flag1] - 1] * (1.0 - dfxi[flag1])
            + z_grid[fyi[flag1] - 1, fxi[flag1]] * (dfxi[flag1] - dfyi[flag1])
            + z_grid[fyi[flag1], fxi[flag1]] * dfyi[flag1]
        )
        zi[flag2] = (
            z_grid[fyi[flag2] - 1, fxi[flag2] - 1] * (1.0 - dfyi[flag2])
            + z_grid[fyi[flag2], fxi[flag2] - 1] * (dfyi[flag2] - dfxi[flag2])
            + z_grid[fyi[flag2], fxi[flag2]] * dfxi[flag2]
        )
        return zi

    if method == 2:
        zi[valid] = (
            z_grid[fyi[valid] - 1, fxi[valid] - 1] * (1.0 - dfxi[valid]) * (1.0 - dfyi[valid])
            + z_grid[fyi[valid] - 1, fxi[valid]] * dfxi[valid] * (1.0 - dfyi[valid])
            + z_grid[fyi[valid], fxi[valid] - 1] * (1.0 - dfxi[valid]) * dfyi[valid]
            + z_grid[fyi[valid], fxi[valid]] * dfxi[valid] * dfyi[valid]
        )
        return zi

    raise ValueError("Invalid method flag")


def ie_fit_line(
    x: NDArray[np.float64] | list[float] | tuple[float, ...],
    y: NDArray[np.float64] | list[float] | tuple[float, ...],
    method: str = "leastSquares",
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Mirror ISETCam's ieFitLine() least-squares line fitting behavior."""

    normalized_method = param_format(method)
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if x_array.ndim == 0 or y_array.ndim == 0:
        raise ValueError("x and y must contain at least one sample.")
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must have the same number of rows.")

    n_data = y_array.shape[1]
    if n_data > 1 and x_array.shape[1] == 1:
        x_array = np.repeat(x_array, n_data, axis=1)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have matching shapes after MATLAB-style replication.")

    if normalized_method in {"oneline", "onelineleastsquares", "leastsquares"}:
        x_flat = x_array.reshape(-1, order="F")
        y_flat = y_array.reshape(-1, order="F")
        design = np.column_stack((x_flat, np.ones(x_flat.size, dtype=float)))
        fit = np.linalg.pinv(design) @ y_flat
        return float(fit[0]), float(fit[1])

    if normalized_method in {"multiplelines", "multiplelinesleastsquares"}:
        slopes = np.empty(n_data, dtype=float)
        offsets = np.empty(n_data, dtype=float)
        ones_col = np.ones(x_array.shape[0], dtype=float)
        for index in range(n_data):
            design = np.column_stack((x_array[:, index], ones_col))
            fit = np.linalg.pinv(design) @ y_array[:, index]
            slopes[index] = float(fit[0])
            offsets[index] = float(fit[1])
        return slopes, offsets

    raise ValueError(f"Unsupported ieFitLine method: {method}")


def ie_mvnrnd(
    mu: Any = 0.0,
    sigma: Any = 1.0,
    k: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    standard_normal_samples: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Mirror ISETCam's ieMvnrnd() multivariate-normal sampling behavior."""

    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)

    if mu_array.ndim == 0:
        mu_array = mu_array.reshape(1, 1)
    elif mu_array.ndim == 1:
        mu_array = mu_array.reshape(1, -1)
    elif mu_array.ndim != 2:
        raise ValueError("mu must be scalar, vector, or a 2D matrix.")

    if mu_array.shape[1] == 1 and sigma_array.ndim == 2 and sigma_array.size > 1:
        mu_array = mu_array.T

    if k is not None:
        sample_count = int(k)
        if sample_count <= 0:
            raise ValueError("k must be positive when provided.")
        mu_array = np.repeat(mu_array, sample_count, axis=0)
    elif standard_normal_samples is not None:
        requested = np.asarray(standard_normal_samples, dtype=float)
        if requested.ndim != 2:
            raise ValueError("standard_normal_samples must be a 2D matrix.")
        if mu_array.shape[0] == 1 and requested.shape[1] == mu_array.shape[1] and requested.shape[0] > 1:
            mu_array = np.repeat(mu_array, requested.shape[0], axis=0)

    n_rows, n_dims = mu_array.shape

    if sigma_array.ndim == 0:
        sigma_array = np.array([[float(sigma_array)]], dtype=float)
    elif sigma_array.ndim != 2:
        raise ValueError("Sigma must be scalar or a 2D covariance matrix.")

    if sigma_array.shape != (n_dims, n_dims):
        raise ValueError("Sigma must have dimensions d x d where mu is n x d.")

    try:
        upper = np.linalg.cholesky(sigma_array).T
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_array)
        if float(np.min(eigenvalues)) < -1e-10:
            raise ValueError("Sigma must be positive semi-definite.") from None
        upper = np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T

    if standard_normal_samples is None:
        if rng is None:
            standard_normal = np.random.standard_normal((n_rows, n_dims))
        else:
            standard_normal = rng.standard_normal((n_rows, n_dims))
    else:
        standard_normal = np.asarray(standard_normal_samples, dtype=float)
        if standard_normal.shape != (n_rows, n_dims):
            raise ValueError("standard_normal_samples must match the expanded mu shape.")

    return np.asarray(standard_normal, dtype=float) @ upper + mu_array


def bi_normal(
    xSpread: float,
    ySpread: float,
    theta: float = 0.0,
    N: int = 128,
) -> NDArray[np.float64]:
    """Mirror ISETCam's biNormal() separable bivariate Gaussian helper."""

    size = int(N)
    if size <= 0:
        raise ValueError("N must be positive.")

    def _gaussian_kernel(length: int, sigma: float) -> NDArray[np.float64]:
        if float(sigma) <= 0.0:
            kernel = np.zeros(length, dtype=float)
            kernel[(length - 1) // 2] = 1.0
            return kernel
        coords = np.arange(length, dtype=float) - (float(length) - 1.0) / 2.0
        kernel = np.exp(-(coords**2) / (2.0 * float(sigma) ** 2))
        kernel /= float(np.sum(kernel))
        return kernel

    x_kernel = _gaussian_kernel(size, float(xSpread))
    y_kernel = _gaussian_kernel(size, float(ySpread))
    gaussian = np.outer(y_kernel, x_kernel)
    if float(theta) != 0.0:
        gaussian = ndi_rotate(gaussian, float(theta), reshape=False, order=1, mode="constant", cval=0.0)
    return np.asarray(gaussian, dtype=float)


biNormal = bi_normal


def exp_rand(
    mn: float = 1.0,
    S: Any = 1,
    *,
    rng: np.random.Generator | None = None,
    uniform_samples: Any | None = None,
) -> NDArray[np.float64]:
    """Mirror the MATLAB expRand() inverse-CDF exponential sampler."""

    shape = _coerce_matlab_shape(S)
    if uniform_samples is None:
        generator = np.random.default_rng() if rng is None else rng
        uniform = generator.random(shape)
    else:
        uniform = np.asarray(uniform_samples, dtype=float)
        if uniform.shape != shape:
            raise ValueError("uniform_samples must match the requested MATLAB-style size.")
    return np.asarray(-float(mn) * np.log(uniform), dtype=float)


expRand = exp_rand


def gamma_pdf(t: Any, n: int, tau: Any) -> NDArray[np.float64]:
    """Mirror ISETCam's gammaPDF() helper, including its last-tau overwrite behavior."""

    t_array = np.asarray(t, dtype=float)
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    tau_values = np.asarray(tau, dtype=float).reshape(-1)
    if tau_values.size == 0:
        raise ValueError("tau must contain at least one value.")

    result = np.zeros_like(t_array, dtype=float)
    factorial = float(math.factorial(int(n) - 1))
    for tau_value in tau_values:
        result = ((t_array / float(tau_value)) ** (int(n) - 1)) * np.exp(-t_array / float(tau_value))
        result = result / (float(tau_value) * factorial)
    total = float(np.sum(result))
    if total != 0.0:
        result = result / total
    return np.asarray(result, dtype=float)


gammaPDF = gamma_pdf


def get_gaussian(cov: Any, rf_support: Any) -> NDArray[np.float64]:
    """Mirror the MATLAB getGaussian() RF-support helper."""

    covariance = np.asarray(cov, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("cov must be a 2x2 covariance matrix.")

    if isinstance(rf_support, dict):
        x_support = rf_support.get("X", rf_support.get("x"))
        y_support = rf_support.get("Y", rf_support.get("y"))
    else:
        x_support = getattr(rf_support, "X", getattr(rf_support, "x", None))
        y_support = getattr(rf_support, "Y", getattr(rf_support, "y", None))
    if x_support is None or y_support is None:
        raise ValueError("rf_support must expose X and Y support vectors.")

    x_values = np.asarray(x_support, dtype=float).reshape(-1)
    y_values = np.asarray(y_support, dtype=float).reshape(-1)
    X, Y = np.meshgrid(x_values, y_values)
    xy = np.column_stack((X.reshape(-1, order="F"), Y.reshape(-1, order="F")))
    inv_covariance = np.linalg.inv(covariance)
    exponent = -0.5 * np.sum((xy @ inv_covariance) * xy, axis=1)
    gaussian = np.exp(exponent) / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))
    gaussian = gaussian.reshape((x_values.size, y_values.size), order="F")

    dx = float(np.mean(np.diff(x_values))) if x_values.size > 1 else 1.0
    dy = float(np.mean(np.diff(y_values))) if y_values.size > 1 else 1.0
    volume = float(np.sum(gaussian)) * dx * dy
    if volume != 0.0:
        gaussian = gaussian / volume
    return np.asarray(gaussian, dtype=float)


getGaussian = get_gaussian


def ie_exprnd(
    mu: Any,
    *varargin: Any,
    rng: np.random.Generator | None = None,
    uniform_samples: Any | None = None,
) -> NDArray[np.float64]:
    """Mirror ISETCam's ieExprnd() exponential sampler."""

    mu_array = np.asarray(mu, dtype=float)
    if np.any(mu_array < 0.0):
        raise ValueError("Exponential parameter less than zero.")

    if len(varargin) == 0:
        shape = mu_array.shape if mu_array.ndim > 0 else ()
    else:
        shape = _coerce_matlab_shape(varargin)

    if uniform_samples is None:
        generator = np.random.default_rng() if rng is None else rng
        uniform = generator.random(shape)
    else:
        uniform = np.asarray(uniform_samples, dtype=float)
        if uniform.shape != shape:
            raise ValueError("uniform_samples must match the requested MATLAB-style size.")
    return np.asarray(-mu_array * np.log(uniform), dtype=float)


ieExprnd = ie_exprnd


def ie_normpdf(x: Any, mu: Any = 0.0, sigma: Any = 1.0) -> NDArray[np.float64]:
    """Mirror the documented ieNormpdf() broadcasting behavior headlessly."""

    x_array = np.asarray(x, dtype=float)
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    try:
        x_b, mu_b, sigma_b = np.broadcast_arrays(x_array, mu_array, sigma_array)
    except ValueError as exc:
        raise ValueError("Requires arguments to match in size.") from exc
    if np.any(sigma_b <= 0.0):
        raise ValueError("Sigma must be > 0")
    xn = (x_b - mu_b) / sigma_b
    return np.asarray(np.exp(-0.5 * xn**2) / (np.sqrt(2.0 * np.pi) * sigma_b), dtype=float)


ieNormpdf = ie_normpdf


def ie_one_over_f(rgb_image: Any, gamma: float = 2.2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror ISETCam's radial FFT spectrum helper for 1/f image analysis."""

    image = np.asarray(rgb_image, dtype=float)
    if image.ndim == 2:
        gray_image = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        linear_rgb = image**float(gamma)
        gray_image = (
            0.2989 * linear_rgb[:, :, 0] + 0.5870 * linear_rgb[:, :, 1] + 0.1140 * linear_rgb[:, :, 2]
        )
    else:
        raise ValueError("rgb_image must be a 2D grayscale image or an RGB image.")

    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(gray_image, dtype=float))))
    rows, cols = gray_image.shape
    x_coords, y_coords = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    center_x = int(np.ceil(cols / 2.0))
    center_y = int(np.ceil(rows / 2.0))
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    distance_bins = np.floor(distances + 0.5).astype(int)
    max_distance = int(np.floor(min(rows, cols) / 2.0))

    amplitude_spectrum = np.zeros(max_distance, dtype=float)
    counts = np.zeros(max_distance, dtype=float)
    valid = (distance_bins > 0) & (distance_bins <= max_distance)
    for radius in range(1, max_distance + 1):
        mask = valid & (distance_bins == radius)
        if np.any(mask):
            amplitude_spectrum[radius - 1] = float(np.sum(amplitude[mask]))
            counts[radius - 1] = float(np.sum(mask))
    amplitude_spectrum = np.divide(
        amplitude_spectrum,
        counts,
        out=np.zeros_like(amplitude_spectrum),
        where=counts > 0,
    )
    spatial_frequencies = np.arange(1, max_distance + 1, dtype=float) / float(max(rows, cols))
    return np.asarray(spatial_frequencies, dtype=float), np.asarray(amplitude_spectrum, dtype=float)


ieOneOverF = ie_one_over_f


def ie_poisson(
    lambda_: Any,
    *args: Any,
    n_samp: Any = 1,
    noise_flag: str = "random",
    seed: Any = 1,
) -> tuple[NDArray[np.float64], Any]:
    """Mirror ISETCam's iePoisson() sampler and seed bookkeeping."""

    options = _normalize_legacy_kwargs(
        args,
        {
            "nSamp": None,
            "noiseFlag": None,
            "seed": None,
        },
    )
    lambda_array = np.asarray(lambda_, dtype=float)
    n_samp_value = options.get("nsamp", n_samp)
    noise_mode = str(options.get("noiseflag", noise_flag)).lower()
    seed_value = options.get("seed", seed)

    if noise_mode == "frozen":
        generator = np.random.default_rng(int(seed_value))
        seed_out: Any = int(seed_value)
    elif noise_mode == "random":
        generator = np.random.default_rng()
        seed_out = generator.bit_generator.state
    elif noise_mode == "donotset":
        generator = np.random.default_rng()
        seed_out = seed_value
    else:
        raise ValueError("noiseFlag must be 'random', 'frozen', or 'donotset'.")

    if lambda_array.ndim == 0:
        size = _coerce_matlab_shape(n_samp_value)
        values = generator.poisson(float(lambda_array), size=size)
    else:
        values = generator.poisson(lambda_array)
    return np.asarray(values, dtype=float), seed_out


iePoisson = ie_poisson


def ie_prcomp(
    data: Any,
    flag: str = "basic",
    n: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mirror ISETCam's iePrcomp() covariance-SVD principal components."""

    data_array = np.asarray(data, dtype=float)
    mean = np.array([], dtype=float)
    mode = param_format(flag)

    if mode == "basic":
        covariance = data_array @ data_array.T
    elif mode == "removemean":
        mean = np.mean(data_array, axis=1)
        centered = data_array - mean[:, np.newaxis]
        covariance = centered @ centered.T
    else:
        raise ValueError(f"Unknown flag {flag}")

    components, _, _ = np.linalg.svd(covariance, full_matrices=True)
    if n is not None:
        components = components[:, : int(n)]
    return np.asarray(components, dtype=float), np.asarray(mean, dtype=float)


iePrcomp = ie_prcomp


def ie_prctile(x: Any, p: Any) -> float | NDArray[np.float64]:
    """Mirror the fallback percentile interpolation in ISETCam's iePrctile()."""

    p_array = np.asarray(p, dtype=float)
    if p_array.ndim > 2 or (p_array.ndim == 2 and min(p_array.shape) != 1):
        raise ValueError("P must be a scalar or a vector.")
    if np.any((p_array < 0.0) | (p_array > 100.0)):
        raise ValueError("P must take values between 0 and 100")
    p_vector = p_array.reshape(-1)

    x_array = np.asarray(x, dtype=float)
    if x_array.ndim == 1 or 1 in x_array.shape:
        values = x_array.reshape(-1, order="F")
        m = values.size
        if m == 1:
            result = np.full((p_vector.size,), float(values[0]), dtype=float)
        else:
            sorted_values = np.sort(values)
            q = 100.0 * np.arange(0.5, float(m), 1.0) / float(m)
            support = np.concatenate(([0.0], q, [100.0]))
            padded = np.concatenate(([float(np.min(values))], sorted_values, [float(np.max(values))]))
            result = np.interp(p_vector, support, padded)
        if p_array.ndim == 0:
            return float(result[0])
        return np.asarray(result, dtype=float)

    sorted_values = np.sort(x_array, axis=0)
    m = x_array.shape[0]
    q = 100.0 * np.arange(0.5, float(m), 1.0) / float(m)
    support = np.concatenate(([0.0], q, [100.0]))
    padded = np.vstack((np.min(x_array, axis=0), sorted_values, np.max(x_array, axis=0)))
    result = np.empty((p_vector.size, x_array.shape[1]), dtype=float)
    for column in range(x_array.shape[1]):
        result[:, column] = np.interp(p_vector, support, padded[:, column])
    if p_array.ndim == 0:
        return np.asarray(result[0, :], dtype=float)
    return np.asarray(result, dtype=float)


iePrctile = ie_prctile


def lorentz_sum(x: Any, params: Any) -> NDArray[np.float64]:
    """Mirror ISETCam's lorentzSum() helper."""

    x_array = np.asarray(x, dtype=float)
    parameter_rows = np.abs(np.asarray(params, dtype=float))
    if parameter_rows.ndim == 1:
        parameter_rows = parameter_rows.reshape(1, -1)
    if parameter_rows.shape[1] != 3:
        raise ValueError("params must contain [f, S, n] rows.")

    result = np.zeros_like(x_array, dtype=float)
    for frequency, scale, power in parameter_rows:
        result = result + scale / (1.0 + (x_array / frequency) ** 2) ** power
    return np.asarray(result, dtype=float)


lorentzSum = lorentz_sum


def ie_fractal_drawgrid(image_or_path: Any, boxwidth: int) -> NDArray[Any]:
    """Mirror the legacy fractal drawgrid helper without opening a figure."""

    step = int(boxwidth)
    if step <= 0:
        raise ValueError("boxwidth must be positive.")

    image = _load_image_array(image_or_path)
    grid_image = np.array(image, copy=True)
    if grid_image.ndim == 2:
        grid_image = np.repeat(grid_image[:, :, np.newaxis], 3, axis=2)
    elif grid_image.ndim == 3 and grid_image.shape[2] == 1:
        grid_image = np.repeat(grid_image, 3, axis=2)
    elif grid_image.ndim != 3:
        raise ValueError("Image data must be 2D grayscale or RGB.")

    if np.issubdtype(grid_image.dtype, np.integer):
        max_value = np.iinfo(grid_image.dtype).max
    else:
        max_value = 1.0

    grid_image[::step, :, :] = max_value
    grid_image[::step, :, 1] = 0
    grid_image[:, ::step, :] = max_value
    grid_image[:, ::step, 1] = 0
    return grid_image


ieFractalDrawgrid = ie_fractal_drawgrid


def ie_fractal_dim(
    image_or_path: Any,
    boxwidth_start: int,
    boxwidth_end: int,
    boxwidth_incr: int,
) -> float:
    """Mirror the box-counting slope in ISETCam's ieFractaldim()."""

    start = int(boxwidth_start)
    stop = int(boxwidth_end)
    incr = int(boxwidth_incr)
    if start <= 0 or stop <= 0 or incr <= 0:
        raise ValueError("Box widths must be positive.")

    image = _load_image_array(image_or_path)
    binary = _binarize_grayscale(image)
    input_matrix = 1.0 - binary
    sum_matrix = np.cumsum(np.cumsum(input_matrix, axis=0), axis=1)

    def _submatrix_sum(i: int, j: int, k: int, l: int) -> float:
        total = float(sum_matrix[k, l])
        if i > 0:
            total -= float(sum_matrix[i - 1, l])
        if j > 0:
            total -= float(sum_matrix[k, j - 1])
        if i > 0 and j > 0:
            total += float(sum_matrix[i - 1, j - 1])
        return total

    rows, cols = input_matrix.shape
    x_values: list[float] = []
    y_values: list[float] = []
    for boxwidth in range(start, stop + 1, incr):
        count = 0
        for row in range(0, rows, boxwidth):
            row_end = min(row + boxwidth - 1, rows - 1)
            for col in range(0, cols, boxwidth):
                col_end = min(col + boxwidth - 1, cols - 1)
                if _submatrix_sum(row, col, row_end, col_end) > 0.0:
                    count += 1
        x_values.append(float(np.log(1.0 / float(boxwidth))))
        y_values.append(float(np.log(float(count))))
    coefficients = np.polyfit(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), 1)
    return float(coefficients[0])


ieFractaldim = ie_fractal_dim


def ie_n_to_megapixel(n: Any, precision: int = 1) -> float | NDArray[np.float64]:
    """Convert a sample count into megapixels using MATLAB ieN2MegaPixel() rounding."""

    values = np.asarray(n, dtype=float)
    scale = 10.0 ** int(precision)
    scaled = values * 1.0e-6 * scale
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)) / scale
    if rounded.ndim == 0:
        return float(rounded)
    return np.asarray(rounded, dtype=float)


def interp_spectra(
    source_wave_nm: NDArray[np.float64],
    values: NDArray[np.float64],
    target_wave_nm: NDArray[np.float64],
    *,
    left: float = 0.0,
    right: float = 0.0,
) -> NDArray[np.float64]:
    """Interpolate spectral columns onto a new wavelength basis."""

    source_wave_nm = np.asarray(source_wave_nm, dtype=float).reshape(-1)
    target_wave_nm = np.asarray(target_wave_nm, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    flattened = values.reshape(source_wave_nm.size, -1)
    if source_wave_nm.size > 1 and np.any(np.diff(source_wave_nm) < 0):
        order = np.argsort(source_wave_nm)
        source_wave_nm = source_wave_nm[order]
        flattened = flattened[order, :]
    interpolated = np.empty((target_wave_nm.size, flattened.shape[1]), dtype=float)
    for column in range(flattened.shape[1]):
        interpolated[:, column] = np.interp(
            target_wave_nm,
            source_wave_nm,
            flattened[:, column],
            left=left,
            right=right,
        )
    return interpolated.reshape((target_wave_nm.size, *values.shape[1:]))


def quanta_to_energy(quanta: NDArray[np.float64], wave_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert photons/quanta to energy."""

    wave_m = np.asarray(wave_nm, dtype=float) * 1e-9
    quanta_array = np.asarray(quanta, dtype=float)
    scale = _wave_scale_like(quanta_array, wave_m)
    return quanta_array * (_PLANCK * _LIGHT_SPEED / scale)


def energy_to_quanta(energy: NDArray[np.float64], wave_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert energy to photons/quanta."""

    wave_m = np.asarray(wave_nm, dtype=float) * 1e-9
    energy_array = np.asarray(energy, dtype=float)
    scale = _wave_scale_like(energy_array, wave_m)
    return energy_array * scale / (_PLANCK * _LIGHT_SPEED)


def _wave_scale_like(values: NDArray[np.float64], wave_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Broadcast wavelength samples onto either wave-last or wave-first arrays."""

    if values.ndim == 0:
        raise ValueError("values must not be scalar.")
    if values.ndim == 1:
        if values.shape[0] != wave_m.size:
            raise ValueError("1D values must match the wavelength vector length.")
        return wave_m

    if values.shape[-1] == wave_m.size:
        return wave_m.reshape((1,) * (values.ndim - 1) + (wave_m.size,))

    if values.shape[0] == wave_m.size:
        return wave_m.reshape((wave_m.size,) + (1,) * (values.ndim - 1))

    raise ValueError("values must match the wavelength vector on the first or last axis.")


def blackbody(
    wave_nm: NDArray[np.float64],
    temperature_k: float | NDArray[np.float64],
    *,
    kind: str = "energy",
    eq_wave_nm: float = 550.0,
) -> NDArray[np.float64]:
    """Approximate ISETCam's blackbody() scaling and unit conventions."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    temps = np.asarray(temperature_k, dtype=float).reshape(-1)
    wave_m = wave[:, None] * 1e-9

    # ISETCam uses pre-combined constants in Planck's law and reports
    # spectral radiance in W / (m^2 nm sr) before optional photon conversion.
    c1 = 3.741832e-16
    c2 = 1.438786e-2
    spec_emit = c1 / (wave_m**5 * (np.exp(c2 / (wave_m * temps[None, :])) - 1.0))
    spec_rad = spec_emit * 1e-9 / np.pi

    idx = int(np.argmin(np.abs(wave - float(eq_wave_nm))))

    from .assets import AssetStore

    store = AssetStore.default()
    _, xyz_energy = store.load_xyz(wave_nm=wave, energy=True)
    y_bar = np.asarray(xyz_energy, dtype=float)[:, 1]
    delta = spectral_step(wave)

    if param_format(kind) in {"watts", "energy"}:
        luminance = 683.0 * float(np.sum(spec_rad[:, 0] * y_bar * delta))
        spec_rad = spec_rad * (100.0 / max(luminance, 1e-12))
        scale = spec_rad[idx, 0] / np.maximum(spec_rad[idx, :], 1e-12)
        spec_rad = spec_rad * scale.reshape(1, -1)
        return spec_rad[:, 0] if temps.size == 1 else spec_rad

    if param_format(kind) in {"photons", "quanta"}:
        scale = spec_rad[idx, 0] / np.maximum(spec_rad[idx, :], 1e-12)
        spec_rad = spec_rad * scale.reshape(1, -1)
        luminance = 683.0 * float(np.sum(spec_rad[:, 0] * y_bar * delta))
        spec_rad = spec_rad * (100.0 / max(luminance, 1e-12))
        photons = energy_to_quanta(spec_rad, wave)
        return photons[:, 0] if temps.size == 1 else photons

    raise ValueError(f"Unsupported blackbody unit type: {kind}")


def resample_cube(cube: NDArray[np.float64], new_shape: tuple[int, int]) -> NDArray[np.float64]:
    """Resample a spectral image cube over rows and columns."""

    if cube.shape[:2] == new_shape:
        return np.asarray(cube, dtype=float).copy()
    factors = (
        new_shape[0] / cube.shape[0],
        new_shape[1] / cube.shape[1],
        1.0,
    )
    return zoom(np.asarray(cube, dtype=float), factors, order=1)


def gaussian_sigma_pixels(
    f_number: float,
    wavelength_nm: float,
    sample_spacing_m: float,
    *,
    extra_blur_pixels: float = 0.0,
) -> float:
    """Approximate diffraction blur as a Gaussian sigma in pixel units."""

    airy_radius_m = 1.22 * wavelength_nm * 1e-9 * f_number
    sigma_pixels = airy_radius_m / max(sample_spacing_m, 1e-12) / 2.355
    return float(max(0.0, sigma_pixels + extra_blur_pixels))


def apply_channelwise_gaussian(
    cube: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    *,
    mode: str = "constant",
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Blur each wavelength slice with its own sigma."""

    output = np.empty_like(cube, dtype=float)
    for index, sigma in enumerate(np.asarray(sigmas, dtype=float).reshape(-1)):
        output[:, :, index] = gaussian_filter(cube[:, :, index], sigma=sigma, mode=mode, cval=cval)
    return output


def _matlab_round(values: Any) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.floor(np.abs(array) + 0.5)


def _scale_to_range(values: Any, out_min: float, out_max: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    in_min = float(np.min(array))
    in_max = float(np.max(array))
    if np.isclose(in_min, in_max):
        return np.full_like(array, float(out_min))
    scaled = (array - in_min) / (in_max - in_min)
    return scaled * (float(out_max) - float(out_min)) + float(out_min)


def half_tone_image(cell: Any, image: Any) -> NDArray[np.bool_]:
    """Apply the legacy MATLAB HalfToneImage threshold-cell algorithm."""

    image_array = np.asarray(image, dtype=float)
    cell_array = np.asarray(cell, dtype=float)
    if image_array.ndim != 2:
        raise ValueError("HalfToneImage expects a 2D grayscale image.")
    if cell_array.ndim != 2 or cell_array.size == 0:
        raise ValueError("HalfToneImage expects a non-empty 2D halftone cell.")

    if float(np.max(cell_array)) > 1.0:
        low = 0.5 / max(float(np.max(cell_array)), 1e-12)
        high = 1.0 - low
        half_tone_cell = _scale_to_range(cell_array, low, high)
    else:
        half_tone_cell = cell_array.copy()

    row_tiles = int(np.ceil(image_array.shape[0] / half_tone_cell.shape[0]))
    col_tiles = int(np.ceil(image_array.shape[1] / half_tone_cell.shape[1]))
    half_tone_mask = np.tile(half_tone_cell, (row_tiles, col_tiles))
    half_tone_mask = half_tone_mask[: image_array.shape[0], : image_array.shape[1]]
    return np.asarray(half_tone_mask < image_array, dtype=bool)


HalfToneImage = half_tone_image


def floyd_steinberg(fs: Any, image: Any) -> NDArray[np.float64]:
    """Apply the legacy MATLAB FloydSteinberg error-diffusion loop."""

    fs_array = np.asarray(fs, dtype=float)
    image_array = np.asarray(image, dtype=float)
    if fs_array.ndim != 2 or fs_array.size == 0:
        raise ValueError("FloydSteinberg expects a non-empty 2D error kernel.")
    if image_array.ndim != 2:
        raise ValueError("FloydSteinberg expects a 2D grayscale image.")

    img_rows, img_cols = image_array.shape
    fs_rows, fs_cols = fs_array.shape
    fs_col_radius = fs_cols // 2

    temp = np.zeros((img_rows + fs_rows, img_cols + 2 * fs_col_radius), dtype=float)
    temp[:img_rows, fs_col_radius : fs_col_radius + img_cols] = image_array

    for row in range(img_rows):
        for col in range(fs_col_radius, img_cols):
            error = float(temp[row, col])
            temp[row, col] = float(_matlab_round(error))
            error -= float(temp[row, col])
            temp[row : row + fs_rows, col - fs_col_radius : col + fs_col_radius + 1] += error * fs_array

        temp[row : row + fs_rows, img_cols : img_cols + fs_col_radius] += temp[
            row + 1 : row + fs_rows + 1, :fs_col_radius
        ]

        for col in range(img_cols, img_cols + fs_col_radius):
            error = float(temp[row, col])
            temp[row, col] = float(_matlab_round(error))
            error -= float(temp[row, col])
            temp[row : row + fs_rows, col - fs_col_radius : col + fs_col_radius + 1] += error * fs_array

        temp[row + 1 : row + fs_rows + 1, fs_col_radius : 2 * fs_col_radius] += temp[
            row : row + fs_rows, img_cols + fs_col_radius : img_cols + 2 * fs_col_radius
        ]
        temp[:, :fs_col_radius] = 0.0
        temp[:, img_cols + fs_col_radius : img_cols + 2 * fs_col_radius] = 0.0

    return np.asarray(temp[:img_rows, fs_col_radius : fs_col_radius + img_cols], dtype=float)


FloydSteinberg = floyd_steinberg


def least_squares_matrix(
    source_sensitivities: NDArray[np.float64],
    target_sensitivities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the linear transform that maps one spectral basis to another."""

    matrix, _, _, _ = np.linalg.lstsq(
        np.asarray(source_sensitivities, dtype=float),
        np.asarray(target_sensitivities, dtype=float),
        rcond=None,
    )
    return matrix


def rgb_to_xw_format(im_rgb: Any) -> tuple[NDArray[np.float64], int, int, int]:
    """Convert an image cube from RGB/r,c,w format to XW/space,w format."""

    array = np.asarray(im_rgb, dtype=float)
    shape = list(array.shape)
    if array.ndim < 2:
        raise ValueError("RGB2XWFormat expects at least a 2D array.")
    if array.ndim < 3:
        shape.append(1)
    if array.ndim > 4:
        raise ValueError("RGB2XWFormat supports up to 4D inputs.")
    rows, cols, waves = int(shape[0]), int(shape[1]), int(shape[2])
    if array.ndim < 4:
        xw = np.reshape(array, (rows * cols, waves), order="F")
    else:
        xw = np.reshape(array, (rows * cols, waves, int(shape[3])), order="F")
    return np.asarray(xw, dtype=float), rows, cols, waves


def xw_to_rgb_format(im_xw: Any, rows: int, cols: int) -> NDArray[np.float64]:
    """Convert XW/space,w data back to RGB/r,c,w format."""

    array = np.asarray(im_xw, dtype=float)
    if array.ndim != 2:
        raise ValueError("XW2RGBFormat expects a 2D XW array.")
    if int(rows) * int(cols) != int(array.shape[0]):
        raise ValueError("XW2RGBFormat row/col values do not match the number of samples.")
    return np.reshape(array, (int(rows), int(cols), int(array.shape[1])), order="F")


def ie_find_wave_index(
    wave: Any,
    wave_val: Any,
    perfect: bool = True,
) -> NDArray[np.bool_]:
    """Return a boolean wavelength-membership vector."""

    wave_samples = np.asarray(wave, dtype=float).reshape(-1)
    if wave_samples.size == 0:
        raise ValueError("Must define list of all wavelengths.")
    requested = np.asarray(wave_val, dtype=float).reshape(-1)
    if requested.size == 0:
        raise ValueError("Must define wavelength values.")

    if perfect:
        return np.isin(wave_samples, requested)

    found = np.zeros(wave_samples.size, dtype=bool)
    for value in requested:
        found[int(np.argmin(np.abs(wave_samples - float(value))))] = True
    return found


ieFindWaveIndex = ie_find_wave_index


def ie_wave2_index(
    wave_list: Any,
    wave: Any,
    *,
    bounding: bool = False,
) -> int | tuple[int, int]:
    """Convert a wavelength to a 1-based index into the wave list."""

    wave_samples = np.asarray(wave_list, dtype=float).reshape(-1)
    if wave_samples.size == 0:
        raise ValueError("wave_list must contain at least one wavelength.")

    target = float(wave)
    idx1 = int(np.argmin(np.abs(wave_samples - target)))
    if not bounding:
        return idx1 + 1

    if float(wave_samples[idx1]) > target:
        idx2 = max(0, idx1 - 1)
        idx1, idx2 = idx2, idx1
    elif float(wave_samples[idx1]) < target:
        idx2 = min(wave_samples.size - 1, idx1 + 1)
    else:
        idx2 = idx1
    return idx1 + 1, idx2 + 1


ieWave2Index = ie_wave2_index


def ie_radial_matrix(
    nx: int,
    ny: int,
    centerx: float,
    centery: float | None = None,
) -> NDArray[np.float64]:
    """Return the radial distance matrix from a MATLAB-style image center."""

    x_center = float(centerx)
    y_center = x_center if centery is None else float(centery)
    x = np.arange(1, int(nx) + 1, dtype=float) - x_center
    y = np.arange(1, int(ny) + 1, dtype=float) - y_center
    return np.sqrt(np.square(y)[:, np.newaxis] + np.square(x)[np.newaxis, :])


ieRadialMatrix = ie_radial_matrix


def image_bounding_box(image: Any) -> NDArray[np.float64]:
    """Return a MATLAB-style square bounding box around non-zero support."""

    support = np.asarray(image)
    if support.ndim == 0:
        raise ValueError("image must be at least two-dimensional.")
    if support.ndim > 2:
        support = np.any(support != 0, axis=tuple(range(2, support.ndim)))

    row, col = np.nonzero(support)
    if row.size == 0 or col.size == 0:
        raise ValueError("image must contain at least one non-zero element.")

    min_row = float(np.min(row) + 1)
    max_row = float(np.max(row) + 1)
    min_col = float(np.min(col) + 1)
    max_col = float(np.max(col) + 1)

    center_x = (min_col + max_col) / 2.0
    center_y = (min_row + max_row) / 2.0
    max_diff = max(abs(max_row - center_y), abs(max_col - center_x))
    return np.array(
        [center_x - max_diff, center_y - max_diff, 2.0 * max_diff, 2.0 * max_diff],
        dtype=float,
    )


imageBoundingBox = image_bounding_box


def image_centroid(img: Any) -> tuple[int, int]:
    """Calculate the rounded centroid of a 2-D image in 1-based pixels."""

    image = np.asarray(img, dtype=float)
    if image.ndim != 2:
        raise ValueError("imageCentroid expects a 2-D image.")

    col_sum = np.sum(image, axis=0)
    row_sum = np.sum(image, axis=1)
    if float(np.sum(col_sum)) == 0.0 or float(np.sum(row_sum)) == 0.0:
        raise ValueError("imageCentroid requires non-zero image mass.")

    col_sum = col_sum / np.sum(col_sum)
    row_sum = row_sum / np.sum(row_sum)
    x_pos = np.arange(1, image.shape[1] + 1, dtype=float)
    y_pos = np.arange(1, image.shape[0] + 1, dtype=float)
    x = int(_matlab_round(np.dot(col_sum, x_pos)).item())
    y = int(_matlab_round(np.dot(row_sum, y_pos)).item())
    return x, y


imageCentroid = image_centroid


def image_circular(im: Any) -> NDArray[Any]:
    """Zero values outside the centered circular aperture."""

    image = np.array(im, copy=True)
    if image.ndim != 2:
        raise ValueError("Need a 2-D image array.")

    image_size = np.array(image.shape, dtype=float)
    center_point = image_size / 2.0 + 1.0
    radius = (float(np.min(image_size)) - 1.0) / 2.0
    x = np.arange(1, image.shape[1] + 1, dtype=float) - center_point[1]
    y = np.arange(1, image.shape[0] + 1, dtype=float) - center_point[0]
    image_radius = np.sqrt(np.square(y)[:, np.newaxis] + np.square(x)[np.newaxis, :])
    image[image_radius > radius] = 0
    return image


imageCircular = image_circular


def image_contrast(data: Any) -> NDArray[np.float64]:
    """Compute per-channel mean-normalized image contrast."""

    original = np.asarray(data)
    image = np.asarray(data, dtype=float)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError("imageContrast expects a 2-D or 3-D image array.")

    contrast = np.zeros_like(image, dtype=float)
    for channel in range(image.shape[2]):
        plane = image[:, :, channel]
        mean_value = float(np.mean(plane))
        contrast[:, :, channel] = (plane - mean_value) / mean_value
    return contrast[:, :, 0] if original.ndim == 2 else contrast


imageContrast = image_contrast


def ie_cmap(c_name: str = "rg", num: int = 256, gam: float = 1.0) -> NDArray[np.float64]:
    """Prepare simple MATLAB-style color maps."""

    name = param_format(c_name)
    count = int(num)
    if count <= 0:
        raise ValueError("num must be positive.")

    a = np.linspace(0.0, 1.0, count, dtype=float)
    if name in {"redgreen", "rg"}:
        return np.column_stack((a, a[::-1], 0.5 * np.ones_like(a)))
    if name in {"blueyellow", "by"}:
        return np.column_stack((a, a, a[::-1]))
    if name in {"luminance", "blackwhite", "bw"}:
        gray = np.repeat(a[:, np.newaxis], 3, axis=1)
        return gray**float(gam)
    raise ValueError(f"Unknown color map name {c_name}")


ieCmap = ie_cmap


def _matlab_round_scalar(value: float) -> int:
    return int(np.sign(value) * np.floor(abs(float(value)) + 0.5))


def _extract_rt_object_distance_m(oi: Any) -> float:
    try:
        from .optics import oi_get

        return float(oi_get(oi, "optics rt object distance"))
    except Exception:
        pass

    fields = oi.fields if hasattr(oi, "fields") else oi
    if not isinstance(fields, dict):
        raise TypeError("ieCropRect expects an ISET-like optical image or dictionary.")
    optics = fields.get("optics", {})
    if not isinstance(optics, dict):
        raise ValueError("Optics data are required.")
    raytrace = optics.get("raytrace", optics)
    if not isinstance(raytrace, dict):
        raise ValueError("Ray-trace optics data are required.")
    object_distance = raytrace.get("objectDistance")
    if object_distance is None:
        object_distance = raytrace.get("object_distance_m")
    if object_distance is None:
        raise ValueError("Ray-trace object distance is required.")
    distance = float(object_distance)
    return distance / 1e3 if distance > 100.0 else distance


def ie_crop_rect(
    oi: Any,
    scenesize: Any,
    fov: float,
    new_fov: float,
    *,
    center: Any | None = None,
) -> NDArray[np.int_]:
    """Calculate a MATLAB-style crop rect for reducing scene field of view."""

    scene_size = np.asarray(scenesize, dtype=float).reshape(-1)
    if scene_size.size != 2:
        raise ValueError("scenesize must be a two-element row/col vector.")
    current_fov = float(fov)
    target_fov = float(new_fov)
    if target_fov > current_fov:
        raise ValueError(f"New field of view ({target_fov}) must not exceed current field of view {current_fov}.")

    center_rc = (
        np.floor(scene_size / 2.0)
        if center is None
        else np.asarray(center, dtype=float).reshape(-1)
    )
    if center_rc.size != 2:
        raise ValueError("center must be a two-element row/col vector.")

    aspect_ratio = float(scene_size[1] / scene_size[0])
    object_distance = _extract_rt_object_distance_m(oi)
    width_new = 2.0 * object_distance * np.tan(np.deg2rad(target_fov / 2.0))
    width_full = 2.0 * object_distance * np.tan(np.deg2rad(current_fov / 2.0))
    n_col_new = _matlab_round_scalar(scene_size[1] * width_new / width_full)
    n_row_new = _matlab_round_scalar(n_col_new / aspect_ratio)
    start_row = _matlab_round_scalar(center_rc[0] - (n_row_new / 2.0))
    start_col = _matlab_round_scalar(center_rc[1] - (n_col_new / 2.0))
    return np.array([start_col + 1, start_row + 1, n_col_new - 1, n_row_new - 1], dtype=int)


ieCropRect = ie_crop_rect


def ie_lut_digital(dac: Any, g_table: Any = 2.2) -> NDArray[np.float64]:
    """Convert DAC values to linear RGB through a gamma table."""

    dac_array = np.asarray(dac, dtype=float)
    gamma = np.asarray(g_table, dtype=float)
    if gamma.ndim == 0:
        return np.power(dac_array, float(gamma))

    if np.max(dac_array) > gamma.shape[0]:
        raise ValueError("Max DAC value exceeds the row dimension of the gTable.")
    if np.max(gamma) > 1.0 or np.min(gamma) < 0.0:
        raise ValueError("gTable entries should be between 0 and 1.")

    if dac_array.ndim == 2:
        dac_array = dac_array[:, :, np.newaxis]
        squeeze = True
    elif dac_array.ndim == 3:
        squeeze = False
    else:
        raise ValueError("ieLUTDigital expects a 2-D or 3-D DAC array.")

    if gamma.ndim == 1:
        gamma = gamma.reshape(-1, 1)
    if gamma.shape[1] == 1:
        gamma = np.repeat(gamma, dac_array.shape[2], axis=1)

    output = np.zeros_like(dac_array, dtype=float)
    indices = np.clip(dac_array.astype(int), 0, gamma.shape[0] - 1)
    for channel in range(dac_array.shape[2]):
        output[:, :, channel] = gamma[:, channel][indices[:, :, channel]]
    return output[:, :, 0] if squeeze else output


ieLUTDigital = ie_lut_digital


def ie_lut_invert(in_lut: Any, n_steps: int = 2048) -> NDArray[np.float64]:
    """Calculate inverse lookup tables from linear RGB to DAC values."""

    lut_in = np.asarray(in_lut, dtype=float)
    if lut_in.ndim == 1:
        lut_in = lut_in.reshape(-1, 1)
    if lut_in.ndim != 2 or lut_in.shape[0] == 0:
        raise ValueError("input lut required")

    steps = int(n_steps)
    if steps <= 0:
        raise ValueError("nSteps must be positive.")

    n_in_steps = lut_in.shape[0]
    y = np.arange(1.0, n_in_steps + 1.0, dtype=float)
    iy = np.linspace(0.0, (steps - 1.0) / steps, steps, dtype=float)
    lut = np.zeros((steps, lut_in.shape[1]), dtype=float)
    for channel in range(lut_in.shape[1]):
        x, indx = np.unique(lut_in[:, channel], return_index=True)
        lut[:, channel] = np.interp(iy, x, y[indx])
        lut[iy < np.min(x), channel] = 0.0
        lut[iy > np.max(x), channel] = float(n_in_steps)
    return np.clip(lut, 0.0, float(np.max(y)))


ieLUTInvert = ie_lut_invert


def ie_lut_linear(rgb: Any, g_table: Any = 2.2) -> NDArray[np.float64]:
    """Convert linear RGB values through an inverse gamma table to DAC values."""

    rgb_array = np.asarray(rgb, dtype=float)
    gamma = np.asarray(g_table, dtype=float)
    if gamma.ndim == 0:
        return np.power(rgb_array, 1.0 / float(gamma))

    if rgb_array.ndim == 2:
        rgb_array = rgb_array[:, :, np.newaxis]
        squeeze = True
    elif rgb_array.ndim == 3:
        squeeze = False
    else:
        raise ValueError("ieLUTLinear expects a 2-D or 3-D RGB array.")

    if gamma.ndim == 1:
        gamma = gamma.reshape(-1, 1)
    if gamma.shape[1] == 1:
        gamma = np.repeat(gamma, rgb_array.shape[2], axis=1)

    t_max = gamma.shape[0]
    scaled = np.floor(rgb_array * t_max) + 1
    scaled[scaled > t_max] = t_max
    scaled = np.clip(scaled.astype(int) - 1, 0, t_max - 1)

    output = np.zeros_like(rgb_array, dtype=float)
    for channel in range(rgb_array.shape[2]):
        output[:, :, channel] = gamma[:, channel][scaled[:, :, channel]]
    return output[:, :, 0] if squeeze else output


ieLUTLinear = ie_lut_linear


def rgb_to_dac(rgb: Any, inv_gamma_table: Any = 2.2) -> NDArray[np.float64]:
    """Convert linear RGB values through an inverse gamma table to DAC values."""

    rgb_array = np.asarray(rgb, dtype=float)
    gamma = np.asarray(inv_gamma_table, dtype=float)
    if gamma.ndim == 0:
        return np.power(rgb_array, 1.0 / float(gamma))

    scaled = np.rint(rgb_array * (gamma.shape[0] - 1)).astype(int)
    scaled = np.clip(scaled, 0, gamma.shape[0] - 1)
    if gamma.ndim == 1:
        result = gamma[scaled]
    else:
        reshaped = scaled.reshape(-1, gamma.shape[1])
        result = np.column_stack([gamma[reshaped[:, ii], ii] for ii in range(gamma.shape[1])]).reshape(rgb_array.shape)
    return np.rint(result).astype(float)


rgb2dac = rgb_to_dac


def image_transpose(im: Any) -> NDArray[np.float64]:
    """Transpose each plane of a multispectral or RGB image."""

    array = np.asarray(im, dtype=float)
    if array.ndim != 3:
        raise ValueError("Input must be 3-dimensional: row x col x w")
    return np.transpose(array, (1, 0, 2)).astype(float, copy=False)


imageTranspose = image_transpose


def image_translate(img: Any, shift: Any, fill_values: float = 0.0) -> NDArray[np.float64]:
    """Translate image data using MATLAB-style x/y pixel shifts."""

    array = np.asarray(img, dtype=float)
    if array.ndim not in {2, 3}:
        raise ValueError("Image required")
    shift_xy = np.asarray(shift, dtype=float).reshape(-1)
    if shift_xy.size != 2:
        raise ValueError("(x,y) Displacement required")
    offset = (-float(shift_xy[1]), -float(shift_xy[0])) if array.ndim == 2 else (-float(shift_xy[1]), -float(shift_xy[0]), 0.0)
    return np.asarray(ndi_shift(array, shift=offset, order=1, mode="constant", cval=float(fill_values), prefilter=False), dtype=float)


imageTranslate = image_translate


def image_interpolate(in_img: Any, r: int, c: int) -> NDArray[np.float64]:
    """Resample a 2-D or 3-D image to the requested row/col size."""

    array = np.asarray(in_img, dtype=float)
    if array.ndim == 2:
        row0, col0 = array.shape
        zoom_factors = (float(r) / float(row0), float(c) / float(col0))
        return np.asarray(zoom(array, zoom_factors, order=1), dtype=float)
    if array.ndim == 3:
        row0, col0, _ = array.shape
        zoom_factors = (float(r) / float(row0), float(c) / float(col0), 1.0)
        return np.asarray(zoom(array, zoom_factors, order=1), dtype=float)
    raise ValueError("Input image required.")


imageInterpolate = image_interpolate


def image_hparams() -> dict[str, Any]:
    """Return the legacy MATLAB default harmonic-parameter structure."""

    return {
        "freq": 2,
        "contrast": 1,
        "ang": 0,
        "ph": 1.5708,
        "row": 128,
        "col": 128,
        "GaborFlag": 0,
    }


imageHparams = image_hparams


def _f_image_to_f_pixel(n_cycles: float, image_size: Any, theta: float = 0.0) -> float:
    size = np.asarray(image_size, dtype=float).reshape(-1)
    if size.size == 1:
        height = width = float(size[0])
    elif size.size >= 2:
        height = float(size[0])
        width = float(size[1])
    else:
        raise ValueError("imageSize must contain at least one value.")
    span = width * abs(np.cos(theta)) + height * abs(np.sin(theta))
    if span <= 0.0:
        raise ValueError("Image span must be positive.")
    return float(n_cycles) / span


def image_gabor(value: Any | None = None, /, **kwargs: Any) -> NDArray[np.float64]:
    """Create a 2-D Gabor image using the legacy MATLAB parameter contract."""

    normalized: dict[str, Any] = {}
    if value is not None:
        if isinstance(value, dict):
            normalized.update({param_format(k): v for k, v in value.items()})
        else:
            normalized.update({param_format(k): v for k, v in dict(value).items()})
    if kwargs:
        normalized.update({param_format(k): v for k, v in kwargs.items()})

    freq = float(normalized.get("freq", normalized.get("frequency", 5.0)))
    phase = float(normalized.get("ph", normalized.get("phase", 0.0)))
    sigma = float(normalized.get("gaborflag", normalized.get("spread", 0.2)))
    theta = float(normalized.get("ang", normalized.get("orientation", 0.0)))
    image_size = normalized.get("row", normalized.get("imagesize", 128))
    contrast = float(normalized.get("contrast", 1.0))

    half_size = max(1, int(np.rint(float(image_size) / 2.0)))
    axis = np.arange(-half_size, half_size + 1, dtype=float)
    x, y = np.meshgrid(axis, axis)
    sample_count = axis.size
    freq_per_pixel = _f_image_to_f_pixel(freq, [sample_count, sample_count], theta)
    if sigma < 1.0:
        sigma = round(sigma * half_size)
    sigma = max(float(sigma), 1.0)

    g_env = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    harmonic = np.cos((2.0 * np.pi * freq_per_pixel * x_prime) + phase)
    gabor = contrast * g_env * harmonic
    gabor = 0.5 * gabor + 0.5
    return np.clip(np.asarray(gabor, dtype=float), 0.0, 1.0)


imageGabor = image_gabor


def image_make_montage(
    hc: Any,
    slice_list: Any | None = None,
    n_cols: int | None = None,
    back_val: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Create a montage image from the slices of a hypercube."""

    cube = np.asarray(hc)
    if cube.ndim != 3:
        raise ValueError("hypercube data required.")
    if any(int(dim) > 10000 for dim in cube.shape):
        raise ValueError("At least one dimension of input image is >10,000- refusing to continue...")

    rows, cols, waves = cube.shape
    slices = np.arange(1, waves + 1, dtype=int) if slice_list is None else np.asarray(slice_list, dtype=int).reshape(-1)
    if slices.size == 0:
        slices = np.arange(1, waves + 1, dtype=int)
    count = int(slices.size)
    num_cols = int(np.ceil(np.sqrt(count) * np.sqrt(rows / cols))) if n_cols in {None, 0} else int(n_cols)
    num_cols = max(num_cols, 1)
    num_rows = int(np.ceil(count / num_cols))

    montage = np.ones((rows * num_rows, cols * num_cols), dtype=cube.dtype) * back_val
    coords = np.zeros((count, 2), dtype=int)
    for ii, cur_slice in enumerate(slices.tolist(), start=1):
        x = ((ii - 1) % num_cols) * cols
        y = ((ii - 1) // num_cols) * rows
        montage[y : y + rows, x : x + cols] = cube[:, :, int(cur_slice) - 1]
        coords[ii - 1, :] = np.array([x + 1, y + 1], dtype=int)
    return np.asarray(montage, dtype=float), coords


imageMakeMontage = image_make_montage


def image_montage(
    hc: Any,
    slices: Any | None = None,
    num_cols: int | None = None,
    fig_num: Any | None = None,
) -> tuple[None, NDArray[np.float64], None]:
    """Headless montage wrapper that returns the montage image and placeholder handles."""

    del fig_num
    montage, _ = image_make_montage(hc, slices, num_cols, 0.0)
    return None, montage, None


imageMontage = image_montage


def convolve_circ(x: Any, h: Any) -> NDArray[np.float64]:
    """Perform MATLAB-style 2-D circular convolution with a same-size result."""

    image = np.asarray(x, dtype=float)
    kernel = np.asarray(h, dtype=float)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolvecirc expects 2-D image and kernel inputs.")

    rows, cols = image.shape
    result = np.asarray(convolve2d(image, kernel, mode="full"), dtype=float)
    full_rows, full_cols = result.shape
    result[: full_rows - rows, :] += result[rows:, :]
    result[:, : full_cols - cols] += result[:, cols:]
    return np.asarray(result[:rows, :cols], dtype=float)


convolvecirc = convolve_circ


def image_slanted_edge(
    im_size: Any | None = None,
    slope: float = 2.6,
    darklevel: float = 0.0,
) -> NDArray[np.float64]:
    """Create a MATLAB-style slanted-edge target image."""

    if im_size is None:
        size = np.array([384.0, 384.0], dtype=float)
    else:
        size = np.asarray(im_size, dtype=float).reshape(-1)
        if size.size == 0:
            size = np.array([384.0, 384.0], dtype=float)
        elif size.size == 1:
            size = np.repeat(size, 2)
        else:
            size = size[:2]

    half_size = np.rint(size / 2.0).astype(int)
    x, y = np.meshgrid(
        np.arange(-half_size[1], half_size[1] + 1, dtype=float),
        np.arange(-half_size[0], half_size[0] + 1, dtype=float),
    )
    image = np.full(x.shape, float(darklevel), dtype=float)
    image[y > float(slope) * x] = 1.0
    return np.asarray(image, dtype=float)


imageSlantedEdge = image_slanted_edge


def _clip_and_scale_unit(image: Any) -> NDArray[np.float64]:
    scaled = np.clip(np.asarray(image, dtype=float), 0.0, None)
    maximum = float(np.max(scaled)) if scaled.size else 0.0
    if maximum > 0.0:
        scaled = scaled / maximum
    return np.asarray(scaled, dtype=float)


def imagesc_rgb(rgbim: Any, *args: Any) -> tuple[None, NDArray[np.float64]]:
    """Scale an RGB or XW image to unit range for headless MATLAB-style display."""

    scaled = _clip_and_scale_unit(rgbim)
    if scaled.ndim == 2:
        if len(args) < 2:
            raise ValueError("2-D input requires row and col arguments.")
        row = int(args[0])
        col = int(args[1])
        gamma = None if len(args) < 3 else float(args[2])
        scaled = xw_to_rgb_format(scaled, row, col)
    elif scaled.ndim == 3:
        gamma = None if len(args) < 1 else float(args[0])
    else:
        raise ValueError("Bad image input")

    if gamma is not None:
        scaled = np.power(np.clip(scaled, 0.0, 1.0), float(gamma))
    return None, np.asarray(scaled, dtype=float)


imagescRGB = imagesc_rgb


def imagesc_opp(
    opp_img: Any,
    gam: float = 0.3,
    n_table: int = 256,
    *args: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create MATLAB-style opponent-image display payloads without GUI drawing."""

    del args
    opp = np.asarray(opp_img, dtype=float)
    if opp.ndim != 3 or opp.shape[2] != 3:
        raise ValueError("imagescOPP expects an RGB-format opponent image.")

    n_colors = max(int(n_table), 1)
    result = np.zeros_like(opp, dtype=float)
    cmap = np.zeros((n_colors, 3, 3), dtype=float)
    color_maps = ("bw", "rg", "by")
    for idx, name in enumerate(color_maps):
        cmap[:, :, idx] = ie_cmap(name, n_colors, gam)
        plane = opp[:, :, idx]
        maximum = float(np.max(np.abs(plane))) if plane.size else 0.0
        if maximum <= 0.0:
            result[:, :, idx] = 0.0 if idx == 0 else 0.5 * n_colors
            continue
        if idx == 0:
            result[:, :, idx] = (plane / maximum) * n_colors
        else:
            result[:, :, idx] = (0.5 * (plane / maximum) + 0.5) * n_colors
    return np.asarray(result, dtype=float), np.asarray(cmap, dtype=float)


imagescOPP = imagesc_opp


def imagesc_m(
    img: Any,
    mp: Any | None = None,
    bar_dir: str = "none",
    no_scale: bool | int = False,
) -> dict[str, Any] | None:
    """Return a headless MATLAB-style monochrome-display payload."""

    array = np.asarray(img, dtype=float)
    if array.size == 0:
        return None

    if mp is None:
        values = np.linspace(0.0, 1.0, 256, dtype=float)
        cmap = np.repeat(values[:, np.newaxis], 3, axis=1)
    else:
        cmap = np.asarray(mp, dtype=float)

    colorbar_direction = str(bar_dir).strip().lower()
    return {
        "image": np.asarray(array, dtype=float),
        "colormap": np.asarray(cmap, dtype=float),
        "scaled": not bool(no_scale),
        "colorbar": None if colorbar_direction == "none" else {"direction": colorbar_direction},
    }


imagescM = imagesc_m


def _default_spd_wave_list(n_wave: int) -> NDArray[np.float64]:
    if n_wave == 31:
        return np.arange(400.0, 701.0, 10.0, dtype=float)
    if n_wave == 301:
        return np.arange(400.0, 701.0, 1.0, dtype=float)
    if n_wave == 37:
        return np.arange(370.0, 731.0, 10.0, dtype=float)
    raise ValueError("wList is required when SPD band count is not a supported legacy default.")


def image_spd(
    spd: Any,
    w_list: Any | None = None,
    gam: float = 1.0,
    row: int | None = None,
    col: int | None = None,
    display_flag: int = 1,
    xcoords: Any | None = None,
    ycoords: Any | None = None,
    this_w: Any | None = None,
) -> NDArray[np.float64]:
    """Render spectral photon data into a headless RGB image."""

    del xcoords, ycoords, this_w
    photons = np.asarray(spd, dtype=float)
    if photons.ndim == 2:
        if row is None or col is None:
            raise ValueError("XW-format SPD input requires row and col arguments.")
        cube = xw_to_rgb_format(photons, int(row), int(col))
    elif photons.ndim == 3:
        cube = photons
        row = int(cube.shape[0])
        col = int(cube.shape[1])
    else:
        raise ValueError("imageSPD expects XW or RGB-format spectral data.")

    wave = _default_spd_wave_list(int(cube.shape[2])) if w_list is None else np.asarray(w_list, dtype=float).reshape(-1)
    if wave.size != int(cube.shape[2]):
        raise ValueError("SPD wavelength list must match the spectral dimension.")

    mode = abs(int(display_flag))
    clip_level = 90.0 if mode == 5 else 99.5
    if mode == 5:
        mode = 4

    if mode in {0, 1}:
        from .color import ie_xyz_from_photons

        xyz = np.asarray(ie_xyz_from_photons(cube, wave), dtype=float)
        maximum = float(np.max(xyz)) if xyz.size else 0.0
        if maximum > 0.0:
            xyz = xyz / maximum
        rgb = xyz_to_srgb(xyz)
    elif mode == 2:
        gray = np.mean(cube, axis=2)
        maximum = float(np.max(gray)) if gray.size else 0.0
        if maximum > 0.0:
            gray = gray / maximum
        rgb = np.repeat(np.asarray(gray, dtype=float)[:, :, np.newaxis], 3, axis=2)
    elif mode in {3, 4}:
        from .color import ie_xyz_from_photons
        from .scene import hdr_render

        xyz = np.asarray(ie_xyz_from_photons(cube, wave), dtype=float)
        if mode == 4:
            clip_value = float(np.percentile(np.asarray(xyz[:, :, 1], dtype=float), clip_level))
            xyz = np.clip(xyz, 0.0, clip_value)
        maximum = float(np.max(xyz)) if xyz.size else 0.0
        if maximum > 0.0:
            xyz = xyz / maximum
        rgb = np.asarray(hdr_render(xyz_to_srgb(xyz)), dtype=float)
    else:
        raise ValueError(f"Unknown display flag value: {display_flag}")

    rgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    if float(gam) != 1.0:
        rgb = np.power(rgb, float(gam))
    return np.asarray(rgb, dtype=float)


imageSPD = image_spd


def image_spd2rgb(spd: Any, w_list: Any, gam: float = 1.0) -> NDArray[np.float64]:
    """Convert spectral photon data to XW-format visible RGB."""

    photons = np.asarray(spd, dtype=float)
    wave = np.asarray(w_list, dtype=float).reshape(-1)
    if photons.ndim == 3:
        rgb = image_spd(photons, wave, gam, display_flag=-1)
        rgb_xw, _, _, _ = rgb_to_xw_format(rgb)
        return np.asarray(rgb_xw, dtype=float)
    if photons.ndim != 2:
        raise ValueError("imageSPD2RGB expects XW or RGB-format spectral data.")

    from .color import ie_xyz_from_photons

    xyz = np.asarray(ie_xyz_from_photons(photons, wave), dtype=float)
    maximum = float(np.max(xyz)) if xyz.size else 0.0
    if maximum > 0.0:
        xyz = xyz / maximum
    rgb = linear_to_srgb(np.clip(xyz_to_linear_srgb(xyz), 0.0, None))
    if float(gam) != 1.0:
        rgb = np.power(np.asarray(rgb, dtype=float), float(gam))
    return np.asarray(rgb, dtype=float)


imageSPD2RGB = image_spd2rgb


def image_hc2rgb(
    obj: Any,
    n_bands: int = 5,
    delta_percent: Any = (10, 10),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create MATLAB-style waveband RGB images and an overlay composite."""

    if obj is None:
        from .scene import scene_create

        obj = scene_create()

    if not hasattr(obj, "type"):
        raise ValueError("imagehc2rgb expects a scene or optical image object.")

    object_type = str(getattr(obj, "type", "")).lower()
    if object_type == "scene":
        from .scene import scene_get, scene_interpolate_w

        wave = np.asarray(scene_get(obj, "wave"), dtype=float)
        rows = int(scene_get(obj, "rows"))
        cols = int(scene_get(obj, "cols"))
        interpolator = scene_interpolate_w
        photon_getter = lambda current: np.asarray(scene_get(current, "photons"), dtype=float)
    elif object_type == "opticalimage":
        from .optics import oi_get, oi_interpolate_w

        wave = np.asarray(oi_get(obj, "wave"), dtype=float)
        rows = int(oi_get(obj, "rows"))
        cols = int(oi_get(obj, "cols"))
        interpolator = oi_interpolate_w
        photon_getter = lambda current: np.asarray(oi_get(current, "photons"), dtype=float)
    else:
        raise ValueError(f"Bad object type {getattr(obj, 'type', object_type)!r}")

    bands = max(int(n_bands), 1)
    wave_starts = np.zeros(bands, dtype=float)
    band_width = max(int(np.floor(wave.size / bands)), 1)
    for index in range(bands):
        wave_starts[index] = wave[min(index * band_width, wave.size - 1)]

    rgb_images = np.zeros((rows, cols, 3, bands), dtype=float)
    delta = np.asarray(delta_percent, dtype=float).reshape(-1)
    if delta.size == 1:
        delta = np.repeat(delta, 2)

    wave_step = float(wave[1] - wave[0]) if wave.size > 1 else 10.0
    for index in range(bands):
        if index == bands - 1:
            wave_list = np.arange(wave_starts[index], wave[-1] + 0.5 * wave_step, wave_step, dtype=float)
        else:
            wave_list = np.arange(wave_starts[index], wave_starts[index + 1] + 0.5 * wave_step, wave_step, dtype=float)
        current = interpolator(obj.clone(), wave_list)
        rgb_images[:, :, :, index] = image_spd(photon_getter(current), wave_list, 1.0, rows, cols, -1)

    row_step = int(np.rint(rows * float(delta[0]) / 100.0))
    col_step = int(np.rint(rows * float(delta[1]) / 100.0))
    overlay_rows = rows + (bands + 1) * row_step
    overlay_cols = cols + (bands + 1) * col_step
    overlay = np.ones((overlay_rows, overlay_cols, 3), dtype=float)
    for index in range(bands - 1, -1, -1):
        row_start = int((bands - index - 1) * row_step + row_step)
        col_start = int(index * col_step + col_step)
        overlay[row_start : row_start + rows, col_start : col_start + cols, :] = rgb_images[:, :, :, index]

    return np.asarray(rgb_images, dtype=float), np.asarray(overlay, dtype=float)


imagehc2rgb = image_hc2rgb


def image_linear_transform(image: Any, transform: Any) -> NDArray[np.float64]:
    """Apply a linear color transform to an image cube."""

    array = np.asarray(image, dtype=float)
    if array.ndim != 3:
        raise ValueError("imageLinearTransform expects a 3D image cube.")
    rows, cols, channels = array.shape
    xw = array.reshape(rows * cols, channels)
    transformed = xw @ np.asarray(transform, dtype=float)
    return np.asarray(transformed, dtype=float).reshape(rows, cols, -1)


def dac_to_rgb(dac: Any, gamma_table: Any = 2.2) -> NDArray[np.float64]:
    """Convert DAC values to linear RGB using MATLAB dac2rgb() semantics."""

    array = np.asarray(dac, dtype=float)
    if array.ndim == 2:
        data = array
        rgb_format = False
        rows = cols = 0
    elif array.ndim == 3:
        data, rows, cols, _ = rgb_to_xw_format(array)
        rgb_format = True
    else:
        raise ValueError("dac2rgb expects XW or RGB-format data.")

    output = np.zeros_like(data, dtype=float)
    gamma_array = np.asarray(gamma_table, dtype=float)

    if gamma_array.ndim == 0:
        output = np.power(data, float(gamma_array))
    elif gamma_array.ndim == 1 and gamma_array.size == 3:
        for channel in range(min(int(data.shape[1]), 3)):
            output[:, channel] = np.power(data[:, channel], float(gamma_array[channel]))
    elif gamma_array.ndim == 2:
        lookup = gamma_array
        if lookup.shape[1] == 1:
            lookup = np.repeat(lookup, data.shape[1], axis=1)
        indices = data
        if float(np.max(indices)) <= 1.0:
            indices = np.floor(indices * float(lookup.shape[0] - 1) + 0.5) + 1.0
        indices = np.clip(indices.astype(int) - 1, 0, lookup.shape[0] - 1)
        for channel in range(min(int(data.shape[1]), lookup.shape[1])):
            output[:, channel] = lookup[indices[:, channel], channel]
    else:
        raise ValueError("Could not parse GammaTable.")

    if rgb_format:
        return xw_to_rgb_format(output, rows, cols)
    return output


def image_flip(im: Any, flip_type: Any = "l") -> NDArray[np.float64]:
    """Flip RGB-format image data using MATLAB imageFlip() semantics."""

    array = np.asarray(im, dtype=float)
    if array.ndim != 3:
        raise ValueError("Input must be rgb image (row x col x w).")

    key = str(flip_type).strip().lower()[:1] if str(flip_type).strip() else "l"
    if key == "u":
        return np.flip(array, axis=0).astype(float, copy=False)
    if key == "l":
        return np.flip(array, axis=1).astype(float, copy=False)
    raise ValueError(f"Unsupported image flip type '{flip_type}'.")


def image_increase_image_rgb_size(im: Any, s: Any) -> NDArray[np.float64]:
    """Increase an image/cube by MATLAB-style pixel replication."""

    array = np.asarray(im, dtype=float)
    if array.ndim not in {2, 3}:
        raise ValueError("Unexpected input matrix dimension.")

    scale = np.asarray(s, dtype=float).reshape(-1)
    if scale.size == 0:
        raise ValueError("Scale factor required.")
    if scale.size == 1:
        scale = np.repeat(scale, 2)
    row_scale = max(int(np.rint(scale[0])), 1)
    col_scale = max(int(np.rint(scale[1])), 1)

    expanded = np.repeat(np.repeat(array, row_scale, axis=0), col_scale, axis=1)
    return np.asarray(expanded, dtype=float)


def hc_basis(
    hc: Any,
    b_type: float | int = 0.995,
    m_type: str = "canonical",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Approximate a hypercube with spectral bases following MATLAB's hcBasis()."""

    cube = np.asarray(hc, dtype=float)
    if cube.ndim == 3:
        xw, rows, cols, _ = rgb_to_xw_format(cube)
    elif cube.ndim == 2:
        xw = cube
        rows = int(cube.shape[0])
        cols = 1
    else:
        raise ValueError("hc_basis expects a 2D XW array or a 3D hypercube.")

    normalized_mean_type = param_format(m_type)
    if normalized_mean_type not in {"canonical", "meansvd"}:
        raise ValueError(f"Unknown hc_basis method '{m_type}'.")

    if normalized_mean_type == "meansvd":
        img_mean = np.mean(xw, axis=0, dtype=float)
        svd_input = xw - img_mean.reshape(1, -1)
    else:
        img_mean = np.array([], dtype=float)
        svd_input = xw

    _, singular_values, right_vectors_t = np.linalg.svd(svd_input, full_matrices=False)
    variance = np.square(singular_values)
    relative_variance = np.cumsum(variance, dtype=float) / max(float(np.sum(variance, dtype=float)), 1e-12)

    if float(b_type) < 1.0:
        matches = np.flatnonzero(relative_variance > float(b_type))
        n_bases = int(matches[0] + 1) if matches.size else int(relative_variance.size)
    else:
        n_bases = min(int(np.rint(float(b_type))), int(relative_variance.size))

    basis = np.asarray(right_vectors_t.T[:, :n_bases], dtype=float)
    coefficients_xw = np.asarray(svd_input @ basis, dtype=float)
    coefficients = xw_to_rgb_format(coefficients_xw, rows, cols)
    var_explained = float(relative_variance[n_bases - 1]) if n_bases > 0 else 0.0
    return np.asarray(img_mean, dtype=float), basis, coefficients, var_explained


def hc_blur(hc: Any, sd: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Blur each hypercube plane with MATLAB-style Gaussian convolution."""

    cube = np.asarray(hc)
    if cube.ndim != 3:
        raise ValueError("Hypercube data required.")
    size = int(sd)
    if size <= 0:
        raise ValueError("sd must be positive.")

    sigma = 0.5
    coords = np.arange(size, dtype=float) - (float(size) - 1.0) / 2.0
    x_grid, y_grid = np.meshgrid(coords, coords)
    blur = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
    blur = blur / float(np.sum(blur))

    blurred = np.empty(cube.shape, dtype=float)
    for index in range(cube.shape[2]):
        blurred[:, :, index] = convolve2d(np.asarray(cube[:, :, index], dtype=float), blur, mode="same")
    return np.asarray(blurred, dtype=float), np.asarray(blur, dtype=float)


hcBlur = hc_blur


def hc_illuminant_scale(hc_illuminant: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate relative illuminant scale across a hypercube image."""

    cube = np.asarray(hc_illuminant, dtype=float)
    if cube.ndim != 3:
        raise ValueError("hypercube illuminant required")
    xw, rows, cols, _ = rgb_to_xw_format(cube)
    mean_spd = np.mean(xw, axis=0, dtype=float)
    ill_scale = np.asarray(xw @ np.linalg.pinv(mean_spd.reshape(-1, 1).T), dtype=float).reshape(rows, cols, order="F")
    maximum = float(np.max(ill_scale))
    if maximum > 0.0:
        ill_scale = ill_scale / maximum
        mean_spd = mean_spd * maximum
    return np.asarray(ill_scale, dtype=float), np.asarray(mean_spd, dtype=float)


hcIlluminantScale = hc_illuminant_scale


_ENVI_DATA_TYPE_MAP: dict[int, np.dtype[Any]] = {
    1: np.dtype("u1"),
    2: np.dtype("i2"),
    3: np.dtype("i4"),
    4: np.dtype("f4"),
    5: np.dtype("f8"),
    12: np.dtype("u2"),
    13: np.dtype("u4"),
    14: np.dtype("i8"),
    15: np.dtype("u8"),
}


def hc_read_hyspex_imginfo(filename: str | Path) -> dict[str, Any]:
    """Read ENVI header metadata in the format expected by hcReadHyspex()."""

    path = Path(filename)
    if path.suffix.lower() != ".hdr":
        path = path.with_suffix(".hdr")
    if not path.exists():
        raise ValueError(f"No file {path} found")

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ENVI":
        raise ValueError(f"{path} is not an ENVI header file!")

    info: dict[str, Any] = {}
    index = 1
    while index < len(lines):
        current = lines[index].strip()
        index += 1
        if not current or "=" not in current:
            continue
        field, value = current.split("=", 1)
        field_name = field.strip().replace(" ", "_")
        value = value.strip()

        if value.startswith("{"):
            collected = [value[1:]]
            while "}" not in collected[-1] and index < len(lines):
                collected.append(lines[index].strip())
                index += 1
            joined = " ".join(collected)
            joined = joined.split("}", 1)[0].strip()
            if field_name.lower() == "description":
                parsed_value: Any = joined
            else:
                parts = [part.strip() for part in joined.split(",") if part.strip()]
                numeric_parts: list[Any] = []
                all_numeric = True
                for part in parts:
                    try:
                        numeric_parts.append(int(part))
                    except ValueError:
                        try:
                            numeric_parts.append(float(part))
                        except ValueError:
                            numeric_parts.append(part)
                            all_numeric = False
                if all_numeric:
                    parsed_value = np.asarray(numeric_parts)
                else:
                    parsed_value = numeric_parts
        else:
            scalar = value
            try:
                parsed_value = int(scalar)
            except ValueError:
                try:
                    parsed_value = float(scalar)
                except ValueError:
                    parsed_value = scalar
        info[field_name] = parsed_value

    if "data_type" in info:
        code = int(np.asarray(info["data_type"]).reshape(-1)[0])
        if code not in _ENVI_DATA_TYPE_MAP:
            raise ValueError("Data type not supported!")
        dtype = _ENVI_DATA_TYPE_MAP[code]
        byte_order = int(np.asarray(info.get("byte_order", 0)).reshape(-1)[0])
        dtype = dtype.newbyteorder(">" if byte_order == 1 else "<")
        info["data_type"] = dtype
    if "byte_order" in info:
        byte_order = int(np.asarray(info["byte_order"]).reshape(-1)[0])
        info["byte_order"] = "ieee-be" if byte_order == 1 else "ieee-le"
    return info


hcReadHyspexImginfo = hc_read_hyspex_imginfo


def hc_read_hyspex(
    filename: str | Path,
    lines: Any | None = None,
    samples: Any | None = None,
    bands: Any | None = None,
) -> tuple[NDArray[Any], dict[str, Any]]:
    """Read ENVI image data using the legacy hcReadHyspex() contract."""

    data_path = Path(filename)
    info = hc_read_hyspex_imginfo(data_path)

    n_lines = int(info["lines"])
    n_samples = int(info["samples"])
    n_bands = int(info["bands"])
    dtype = np.dtype(info["data_type"])
    header_offset = int(info.get("header_offset", 0))
    interleave = str(info.get("interleave", "bsq")).strip().lower()

    total = n_lines * n_samples * n_bands
    data = np.fromfile(data_path, dtype=dtype, count=total, offset=header_offset)
    if data.size != total:
        raise ValueError("Unexpected ENVI payload size.")

    if interleave == "bsq":
        cube = data.reshape((n_bands, n_lines, n_samples)).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = data.reshape((n_lines, n_bands, n_samples)).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = data.reshape((n_lines, n_samples, n_bands))
    else:
        raise ValueError(f"Unsupported interleave format {interleave!r}.")

    if lines is None:
        row_indices = np.arange(n_lines, dtype=int)
    else:
        row_indices = np.asarray(lines, dtype=int).reshape(-1) - 1
    if samples is None:
        col_indices = np.arange(n_samples, dtype=int)
    else:
        col_indices = np.asarray(samples, dtype=int).reshape(-1) - 1
    if bands is None:
        band_indices = np.arange(n_bands, dtype=int)
    elif isinstance(bands, str) and bands.lower() == "default":
        default_bands = np.asarray(info.get("default_bands", np.arange(1, n_bands + 1)), dtype=int).reshape(-1)
        band_indices = default_bands - 1
    else:
        band_indices = np.asarray(bands, dtype=int).reshape(-1) - 1

    image = cube[np.ix_(row_indices, col_indices, band_indices)]
    return np.asarray(image), info


hcReadHyspex = hc_read_hyspex


def hc_image(
    hc: Any,
    display_type: str = "mean gray",
    slices: Any | None = None,
) -> Any:
    """Return headless payloads for the legacy hcimage() display helper."""

    cube = np.asarray(hc, dtype=float)
    if cube.ndim != 3:
        raise ValueError("hypercube image data required")
    dtype_key = param_format(display_type)

    if dtype_key == "meangray":
        return np.asarray(np.mean(cube, axis=2, dtype=float), dtype=float)
    if dtype_key in {"imagemontage", "montage"}:
        selection = np.arange(1, cube.shape[2] + 1, dtype=int) if slices is None else np.asarray(slices, dtype=int).reshape(-1)
        return image_montage(cube, selection)
    if dtype_key == "movie":
        max_value = float(np.max(cube))
        scaled = np.zeros_like(cube, dtype=float) if max_value == 0.0 else 256.0 * cube / max_value
        return {"frames": np.asarray(scaled, dtype=float), "title": f"Hypercube wavebands: {cube.shape[2]}", "type": "movie"}
    raise ValueError(f"Unknown hc image display type: {display_type}")


hcimage = hc_image


def hc_image_crop(
    img: Any,
    rect: Any | None = None,
    cPlane: int = 1,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    """Crop a hypercube image using MATLAB rect semantics."""

    cube = np.asarray(img)
    if cube.ndim != 3:
        raise ValueError("hyper cube image required")
    _ = int(cPlane)
    rows, cols, _ = cube.shape

    if rect is None:
        crop_rect = np.array([1, 1, cols - 1, rows - 1], dtype=int)
    else:
        crop_rect = np.rint(np.asarray(rect, dtype=float).reshape(-1)).astype(int)
        if crop_rect.size != 4:
            raise ValueError("ROI rect must contain [col, row, width, height].")

    col_min = max(int(crop_rect[0]), 1)
    row_min = max(int(crop_rect[1]), 1)
    col_max = min(col_min + int(crop_rect[2]), cols)
    row_max = min(row_min + int(crop_rect[3]), rows)
    cropped = cube[row_min - 1 : row_max, col_min - 1 : col_max, ...]
    normalized_rect = np.array([col_min, row_min, max(col_max - col_min, 0), max(row_max - row_min, 0)], dtype=int)
    return np.asarray(cropped), normalized_rect


hcimageCrop = hc_image_crop


def hc_image_rotate_clip(
    hc: Any,
    clipPrctile: float = 99.9,
    nRot: int = 1,
) -> tuple[NDArray[Any], NDArray[np.float64]]:
    """Rotate and percentile-clip each plane of a hypercube image."""

    cube = np.asarray(hc)
    if cube.ndim != 3:
        raise ValueError("hyper cube image required")

    rotated_first = np.rot90(np.asarray(cube[:, :, 0], dtype=float), int(nRot)) if int(nRot) != 0 else np.asarray(cube[:, :, 0], dtype=float)
    out_shape = (*rotated_first.shape, cube.shape[2])
    if np.issubdtype(cube.dtype, np.integer):
        clipped_cube = np.zeros(out_shape, dtype=cube.dtype)
    else:
        clipped_cube = np.zeros(out_shape, dtype=float)
    clipped_pixels = np.zeros(rotated_first.shape, dtype=float)

    for index in range(cube.shape[2]):
        plane = np.asarray(cube[:, :, index], dtype=float)
        if int(nRot) != 0:
            plane = np.rot90(plane, int(nRot))
        if float(clipPrctile) < 100.0:
            maximum = float(np.percentile(plane.reshape(-1), float(clipPrctile)))
            mask = plane > maximum
            clipped_pixels = clipped_pixels + mask.astype(float)
            plane = plane.copy()
            plane[mask] = 0.0
        if np.issubdtype(clipped_cube.dtype, np.integer):
            clipped_cube[:, :, index] = np.rint(plane).astype(clipped_cube.dtype)
        else:
            clipped_cube[:, :, index] = plane
    return np.asarray(clipped_cube), np.asarray(clipped_pixels, dtype=float)


hcimageRotateClip = hc_image_rotate_clip


def hc_viewer(image_cube: Any, slice_map: Any | None = None) -> dict[str, Any]:
    """Return a headless payload for the legacy hcViewer() slider UI."""

    cube = np.asarray(image_cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("Image cube required.")
    mapping = (
        np.arange(1, cube.shape[2] + 1, dtype=float)
        if slice_map is None
        else np.asarray(slice_map, dtype=float).reshape(-1)
    )
    if mapping.size != cube.shape[2]:
        raise ValueError("slice_map must match the number of planes in image_cube.")
    return {
        "cube": cube.copy(),
        "current_slice": 1,
        "image": np.asarray(cube[:, :, 0], dtype=float),
        "slice_map": mapping.copy(),
        "label": f"Slice: {mapping[0]:g}",
        "colormap": "gray",
        "title": "Image Cube Viewer",
    }


hcViewer = hc_viewer


imageFlip = image_flip
imageIncreaseImageRGBSize = image_increase_image_rgb_size


def xyz_to_linear_srgb(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIE XYZ to linear sRGB."""

    transform = np.array(
        [
            [3.2410, -1.5374, -0.4986],
            [-0.9692, 1.8760, 0.0416],
            [0.0556, -0.2040, 1.0570],
        ],
        dtype=float,
    )
    reshaped = np.asarray(xyz, dtype=float).reshape(-1, 3)
    rgb = reshaped @ transform.T
    return rgb.reshape(np.asarray(xyz).shape)


def linear_to_srgb(linear_rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the standard sRGB transfer curve."""

    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, None)
    threshold = 0.0031308
    srgb = np.where(
        linear_rgb <= threshold,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def srgb_to_linear(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the inverse sRGB transfer curve."""

    srgb = np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0)
    threshold = 0.04045
    linear = np.where(
        srgb <= threshold,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )
    return np.clip(linear, 0.0, 1.0)


def xyz_to_srgb(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert CIE XYZ to sRGB using MATLAB xyz2srgb() scaling rules."""

    xyz_array = np.asarray(xyz, dtype=float)
    if xyz_array.ndim != 3 or xyz_array.shape[2] != 3:
        raise ValueError("xyz2srgb expects an RGB-format XYZ image.")

    scaled_xyz = xyz_array.copy()
    max_y = float(np.max(scaled_xyz[:, :, 1])) if scaled_xyz.size else 1.0
    if max_y > 1.0:
        scaled_xyz = scaled_xyz / max_y
    if float(np.min(scaled_xyz)) < 0.0:
        scaled_xyz = np.clip(scaled_xyz, 0.0, 1.0)

    linear_rgb = xyz_to_linear_srgb(scaled_xyz)
    return linear_to_srgb(np.clip(linear_rgb, 0.0, 1.0))


xyz2srgb = xyz_to_srgb


def srgb_to_xyz(srgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert sRGB to CIE XYZ."""

    srgb_array = np.asarray(srgb, dtype=float)
    if srgb_array.ndim != 3 or srgb_array.shape[2] != 3:
        raise ValueError("srgb2xyz expects an RGB-format sRGB image.")

    linear_rgb = srgb_to_linear(srgb_array)
    transform = np.linalg.inv(
        np.array(
            [
                [3.2410, -1.5374, -0.4986],
                [-0.9692, 1.8760, 0.0416],
                [0.0556, -0.2040, 1.0570],
            ],
            dtype=float,
        )
    )
    reshaped = linear_rgb.reshape(-1, 3)
    xyz = reshaped @ transform.T
    return xyz.reshape(srgb_array.shape)


srgb2xyz = srgb_to_xyz


def invert_gamma_table(linear_rgb: NDArray[np.float64], gamma_table: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a display gamma LUT."""

    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=float), 0.0, 1.0)
    gamma_table = np.asarray(gamma_table, dtype=float)
    digital_axis = np.linspace(0.0, 1.0, gamma_table.shape[0])
    output = np.empty_like(linear_rgb, dtype=float)
    for channel in range(linear_rgb.shape[-1]):
        output[..., channel] = np.interp(
            linear_rgb[..., channel],
            gamma_table[:, channel],
            digital_axis,
            left=0.0,
            right=1.0,
        )
    return output


def tile_pattern(pattern: NDArray[np.int_], rows: int, cols: int) -> NDArray[np.int_]:
    """Tile a CFA pattern to image size."""

    pattern = np.asarray(pattern, dtype=int)
    tiled = np.tile(
        pattern,
        (
            int(np.ceil(rows / pattern.shape[0])),
            int(np.ceil(cols / pattern.shape[1])),
        ),
    )
    return tiled[:rows, :cols]


def ensure_multiple(value: int, multiple: int) -> int:
    """Round up to the next multiple."""

    if multiple <= 1:
        return int(value)
    return int(np.ceil(value / multiple) * multiple)


def array_percentile(value: NDArray[np.float64], percentile: float) -> float:
    return float(np.percentile(np.asarray(value, dtype=float), percentile))
