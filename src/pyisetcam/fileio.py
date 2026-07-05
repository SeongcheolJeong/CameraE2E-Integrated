"""Headless MATLAB-style object save/load helpers."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import io
import os
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.io import loadmat, savemat, whosmat
from tifffile import TiffFile

from .assets import AssetStore
from .exceptions import UnsupportedOptionError
from .session import ie_add_object, session_add_object, session_replace_object
from .types import BaseISETObject, Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor, SessionContext
from .utils import interp_spectra, param_format, quanta_to_energy

_SAVE_KEY_BY_TYPE = {
    "scene": "scene",
    "oi": "opticalimage",
    "opticalimage": "opticalimage",
    "sensor": "isa",
    "isa": "isa",
    "ip": "vcimage",
    "vcimage": "vcimage",
    "display": "display",
    "camera": "camera",
}

_LOAD_KEYS_BY_TYPE = {
    "scene": ("scene",),
    "oi": ("opticalimage",),
    "opticalimage": ("opticalimage",),
    "sensor": ("isa", "isa_", "sensor"),
    "isa": ("isa", "isa_", "sensor"),
    "ip": ("vcimage",),
    "vcimage": ("vcimage",),
    "display": ("display",),
    "camera": ("camera",),
}

_OBJECT_CLASS_BY_TYPE = {
    "scene": Scene,
    "oi": OpticalImage,
    "opticalimage": OpticalImage,
    "sensor": Sensor,
    "isa": Sensor,
    "ip": ImageProcessor,
    "vcimage": ImageProcessor,
    "display": Display,
    "camera": Camera,
}

_IE_WEB_GET_RESOURCES = {
    "bitterli": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/bitterli",
    "isetcambitterli": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/bitterli",
    "pbrtv4": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/pbrtv4",
    "isetcampharr": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/pbrtv4",
    "isetcamiset3d": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/iset3d",
    "iset3d": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/iset3d",
    "iset3dscenes": "https://stacks.stanford.edu/file/rq335tn9587/ISETCam%20scenes%20rendered/iset3d",
}

_IE_WEB_GET_COLLECTIONS = {
    "vistalabcollection": "https://purl.stanford.edu/rq335tn9587",
    "isetmultispectralcollection": "https://purl.stanford.edu/hx650kw3903",
    "isethyperspectralcollection": "https://purl.stanford.edu/kh752sm9123",
}


def _default_save_path(obj: BaseISETObject) -> Path:
    name = str(getattr(obj, "name", "") or param_format(obj.type) or "object")
    return Path.cwd() / f"{name}.mat"


def _normalize_save_path(full_name: str | Path | None, obj: BaseISETObject | None = None) -> Path:
    if full_name is None:
        if obj is None:
            raise ValueError("full_name is required when no object is available for a default path.")
        path = _default_save_path(obj)
    else:
        path = Path(full_name)
    if path.suffix.lower() != ".mat":
        path = path.with_suffix(".mat")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_value(value: Any) -> Any:
    if value is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(value, BaseISETObject):
        return {
            "name": str(value.name),
            "type": str(value.type),
            "metadata": _serialize_value(value.metadata),
            "fields": _serialize_value(value.fields),
            "data": _serialize_value(value.data),
        }
    if isinstance(value, dict):
        return {str(key): _serialize_value(val) for key, val in value.items() if val is not None}
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deserialize_value(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {field: _deserialize_value(getattr(value, field)) for field in value._fieldnames}
    if isinstance(value, dict):
        return {str(key): _deserialize_value(val) for key, val in value.items() if not str(key).startswith("__")}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.shape == ():
                return _deserialize_value(value.item())
            return [_deserialize_value(item) for item in value.tolist()]
        if value.shape == ():
            return value.item()
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reconstruct_object(value: Any, inferred_type: str | None = None) -> Any:
    if isinstance(value, dict):
        normalized_type = param_format(value.get("type", inferred_type or ""))
        if normalized_type in _OBJECT_CLASS_BY_TYPE and any(key in value for key in ("fields", "data", "metadata")):
            cls = _OBJECT_CLASS_BY_TYPE[normalized_type]
            metadata = _reconstruct_object(value.get("metadata", {}))
            fields = _reconstruct_object(value.get("fields", {}))
            data = _reconstruct_object(value.get("data", {}))
            name = str(value.get("name", normalized_type or cls().name))
            object_type = str(value.get("type", cls().type))
            return cls(name=name, type=object_type, metadata=metadata, fields=fields, data=data)
        return {str(key): _reconstruct_object(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_reconstruct_object(item) for item in value]
    return value


def _save_key_for_object(obj: BaseISETObject) -> str:
    normalized_type = param_format(obj.type)
    key = _SAVE_KEY_BY_TYPE.get(normalized_type)
    if key is None:
        raise ValueError(f"Unsupported object type for vcSaveObject: {obj.type}")
    return key


def _load_payload(path: Path, obj_type: str) -> Any:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    normalized_type = param_format(obj_type or "scene")
    candidates = _LOAD_KEYS_BY_TYPE.get(normalized_type, (normalized_type,))
    for key in candidates:
        if key in data:
            return data[key], normalized_type
    available = [key for key in data if not key.startswith("__")]
    if len(available) == 1:
        return data[available[0]], normalized_type
    raise KeyError(f"No saved object matching type {obj_type} found in {path}.")


def vc_save_object(obj: BaseISETObject, full_name: str | Path | None = None) -> str:
    """Save a core ISET object to a MATLAB `.mat` file."""

    path = _normalize_save_path(full_name, obj)
    save_key = _save_key_for_object(obj)
    savemat(path, {save_key: _serialize_value(obj)}, do_compression=True)
    return str(path)


def vc_export_object(
    obj: BaseISETObject,
    full_name: str | Path | None = None,
    clear_data_flag: bool = False,
) -> str:
    """Export an ISET object, optionally clearing cached data first."""

    export_obj = obj.clone()
    if clear_data_flag:
        export_obj.data = {}
    return vc_save_object(export_obj, full_name)


def vc_load_object(
    obj_type: str = "scene",
    full_name: str | Path | None = None,
    val: int | None = None,
    *,
    session: SessionContext | None = None,
) -> tuple[Any, str]:
    """Load a saved ISET object from a MATLAB `.mat` file."""

    if full_name is None:
        raise ValueError("full_name is required in the headless Python port of vcLoadObject.")
    path = _normalize_save_path(full_name)
    payload, normalized_type = _load_payload(path, obj_type)
    loaded = _reconstruct_object(_deserialize_value(payload), inferred_type=normalized_type)
    if not isinstance(loaded, BaseISETObject):
        cls = _OBJECT_CLASS_BY_TYPE.get(normalized_type)
        if cls is None:
            raise ValueError(f"Unsupported object type for vcLoadObject: {obj_type}")
        loaded = cls(name=path.stem, type=normalized_type, fields=_reconstruct_object(loaded))
    loaded.name = path.stem

    if session is None:
        return loaded, str(path)

    if val is None:
        slot = ie_add_object(session, loaded)
    else:
        if isinstance(loaded, (Camera, ImageProcessor)):
            session_replace_object(session, loaded, int(val), select=True)
        else:
            session_add_object(session, loaded, select=True, object_id=int(val))
        slot = int(val)
    return slot, str(path)


def ie_image_type(full_name: str | Path) -> str:
    path = Path(full_name)
    full_name_str = str(path).lower()
    image_path = str(path.parent).lower()
    ext = path.suffix.lower()

    if ext in {".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png"}:
        if os.path.join("data", "images", "targets").replace("\\", "/") in full_name_str.replace("\\", "/"):
            return "monochrome"
        if "monochrome" in full_name_str:
            return "monochrome"
        return "rgb"

    if "monochrome" in image_path:
        return "monochrome"
    if "multispectral" in image_path or "hyperspectral" in image_path:
        return "multispectral"
    if "rgb" in image_path:
        return "rgb"
    return ""


def ie_tempfile(ext: str | None = None) -> tuple[str, str]:
    suffix = ""
    if ext:
        normalized = str(ext)
        suffix = normalized if normalized.startswith(".") else f".{normalized}"
    descriptor, full_name = tempfile.mkstemp(prefix="ie_", suffix=suffix)
    os.close(descriptor)
    os.unlink(full_name)
    path = Path(full_name)
    return str(path), str(path.parent)


def ie_var_in_file(fullname: str | Path | Any, var_name: str) -> bool:
    if isinstance(fullname, (str, Path)):
        variables = whosmat(str(fullname))
    else:
        variables = fullname

    for variable in variables:
        if isinstance(variable, dict):
            name = variable.get("name")
        elif hasattr(variable, "name"):
            name = getattr(variable, "name")
        elif isinstance(variable, tuple) and variable:
            name = variable[0]
        else:
            name = None
        if str(name) == var_name:
            return True
    return False


def path_to_linux(input_path: str | Path) -> str:
    source = str(input_path)
    if len(source) > 2 and source[1:3] == ":\\":
        source = source[2:]
    return source.replace("\\", "/")


def _xml_decode_name(value: str) -> str:
    decoded = str(value)
    decoded = decoded.replace("-", "_dash_")
    decoded = decoded.replace(":", "_colon_")
    decoded = decoded.replace(".", "_dot_")
    return decoded


def _xml_encode_name(value: str) -> str:
    encoded = str(value)
    encoded = encoded.replace("_dash_", "-")
    encoded = encoded.replace("_colon_", ":")
    encoded = encoded.replace("_dot_", ".")
    return encoded


def _xml_scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _xml_scalar_text(value.item())
        flattened = np.asarray(value).reshape(-1)
        return " ".join(_xml_scalar_text(item) for item in flattened.tolist())
    if isinstance(value, (list, tuple)):
        return " ".join(_xml_scalar_text(item) for item in value)
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _xml_namespace_prefixes(payload: Any) -> set[str]:
    prefixes: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "Attributes" and isinstance(value, Mapping):
                for attr_name in value:
                    raw = _xml_encode_name(str(attr_name))
                    if ":" in raw:
                        prefixes.add(raw.split(":", 1)[0])
                continue
            raw = _xml_encode_name(str(key))
            if key not in {"Attributes", "Text"} and ":" in raw:
                prefixes.add(raw.split(":", 1)[0])
            prefixes.update(_xml_namespace_prefixes(value))
    elif isinstance(payload, list):
        for item in payload:
            prefixes.update(_xml_namespace_prefixes(item))
    return prefixes


def _xml_qualified_name(value: str, namespace_map: Mapping[str, str]) -> str:
    encoded = _xml_encode_name(value)
    if ":" not in encoded:
        return encoded
    prefix, local = encoded.split(":", 1)
    uri = namespace_map[prefix]
    return f"{{{uri}}}{local}"


def _xml_decode_name_with_namespaces(value: str, uri_to_prefix: Mapping[str, str]) -> str:
    raw = str(value)
    if raw.startswith("{") and "}" in raw:
        uri, local = raw[1:].split("}", 1)
        prefix = uri_to_prefix.get(uri, "")
        if prefix:
            return _xml_decode_name(f"{prefix}:{local}")
        return _xml_decode_name(local)
    return _xml_decode_name(raw)


def _xml_parse_with_namespaces(source: str | bytes | Path) -> tuple[ET.Element, dict[str, str]]:
    uri_to_prefix: dict[str, str] = {}
    if isinstance(source, Path):
        iterator = ET.iterparse(source, events=("start", "start-ns"))
    else:
        data = source if isinstance(source, bytes) else str(source).encode("utf-8")
        iterator = ET.iterparse(io.BytesIO(data), events=("start", "start-ns"))

    root: ET.Element | None = None
    for event, value in iterator:
        if event == "start-ns":
            prefix, uri = value
            uri_to_prefix[str(uri)] = str(prefix or "")
        elif event == "start" and root is None:
            root = value
    if root is None:
        raise ValueError("XML payload does not contain a root element.")
    return root, uri_to_prefix


def _xml_element_to_struct(element: ET.Element, uri_to_prefix: Mapping[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if element.attrib:
        payload["Attributes"] = {
            _xml_decode_name_with_namespaces(name, uri_to_prefix): str(value) for name, value in element.attrib.items()
        }

    child_elements = [child for child in list(element) if isinstance(child.tag, str)]
    if child_elements:
        text_parts: list[str] = []
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
        for child in child_elements:
            child_name = _xml_decode_name_with_namespaces(child.tag, uri_to_prefix)
            child_value = _xml_element_to_struct(child, uri_to_prefix)
            existing = payload.get(child_name)
            if existing is None:
                payload[child_name] = child_value
            elif isinstance(existing, list):
                existing.append(child_value)
            else:
                payload[child_name] = [existing, child_value]
            if child.tail and child.tail.strip():
                text_parts.append(child.tail.strip())
        if text_parts:
            payload["Text"] = "".join(text_parts)
        return payload

    if element.text and element.text.strip():
        payload["Text"] = element.text.strip()
    return payload


def _xml_append_payload(element: ET.Element, payload: Any, namespace_map: Mapping[str, str]) -> None:
    if isinstance(payload, Mapping):
        attributes = payload.get("Attributes")
        if isinstance(attributes, Mapping):
            for name, value in attributes.items():
                element.set(_xml_qualified_name(str(name), namespace_map), _xml_scalar_text(value))
        text_value = payload.get("Text")
        if text_value is not None:
            element.text = _xml_scalar_text(text_value)
        for key, value in payload.items():
            if key in {"Attributes", "Text"}:
                continue
            child_name = _xml_qualified_name(str(key), namespace_map)
            items = value if isinstance(value, list) else [value]
            for item in items:
                child = ET.SubElement(element, child_name)
                _xml_append_payload(child, item, namespace_map)
        return

    element.text = _xml_scalar_text(payload)


def ie_xml2struct(file: str | Path | ET.ElementTree | ET.Element) -> dict[str, Any]:
    """Convert XML into a MATLAB-style nested structure payload."""

    root: ET.Element
    uri_to_prefix: dict[str, str]
    if isinstance(file, ET.ElementTree):
        root, uri_to_prefix = _xml_parse_with_namespaces(ET.tostring(file.getroot(), encoding="utf-8"))
    elif isinstance(file, ET.Element):
        root, uri_to_prefix = _xml_parse_with_namespaces(ET.tostring(file, encoding="utf-8"))
    else:
        source = str(file)
        candidate = Path(source)
        if candidate.exists():
            root, uri_to_prefix = _xml_parse_with_namespaces(candidate)
        else:
            with_xml = candidate.with_suffix(".xml") if candidate.suffix.lower() != ".xml" else candidate
            if with_xml.exists():
                root, uri_to_prefix = _xml_parse_with_namespaces(with_xml)
            else:
                try:
                    root, uri_to_prefix = _xml_parse_with_namespaces(source)
                except ET.ParseError as exc:
                    raise FileNotFoundError(f"The file {source} could not be found") from exc

    return {
        _xml_decode_name_with_namespaces(root.tag, uri_to_prefix): _xml_element_to_struct(root, uri_to_prefix)
    }


def ie_struct2xml(payload: Mapping[str, Any], file: str | Path | None = None) -> str:
    """Convert a MATLAB-style nested structure payload into XML text or a file."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    names = list(payload.keys())
    if len(names) != 1:
        raise ValueError("There should be a single field in the main structure.")

    root_name = str(names[0])
    namespace_map = {
        prefix: f"urn:pyisetcam:{prefix}"
        for prefix in sorted(_xml_namespace_prefixes(payload))
    }
    for prefix, uri in namespace_map.items():
        ET.register_namespace(prefix, uri)

    root = ET.Element(_xml_qualified_name(root_name, namespace_map))
    _xml_append_payload(root, payload[root_name], namespace_map)
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    xml_text = ET.tostring(root, encoding="unicode")
    xml_text = "<?xml version='1.0' encoding='utf-8'?>\n" + xml_text

    if file is None:
        return xml_text

    path = Path(file)
    if not str(path):
        raise ValueError("Filename can not be empty")
    if path.suffix.lower() != ".xml":
        path = path.with_suffix(".xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml_text, encoding="utf-8")
    return str(path)


def ie_save_spectral_file(
    wavelength: Any,
    data: Any,
    comment: str | None = None,
    fullpathname: str | Path | None = None,
    d_format: str = "double",
) -> str:
    wave = np.asarray(wavelength, dtype=float).reshape(-1)
    payload = np.asarray(data)
    if payload.ndim == 3:
        if payload.shape[2] != wave.size:
            raise ValueError("The 3rd dimension of data must match number of wavelengths.")
    elif payload.ndim <= 2:
        if payload.ndim == 1:
            payload = payload.reshape(-1, 1)
        if payload.shape[0] != wave.size:
            raise ValueError("The row dimension of data must match number of wavelengths.")
    else:
        raise ValueError("data must be 1D, 2D, or 3D.")

    format_key = param_format(d_format)
    if format_key == "double":
        stored = np.asarray(payload, dtype=float)
    elif format_key == "single":
        stored = np.asarray(payload, dtype=np.float32)
    else:
        raise UnsupportedOptionError("ieSaveSpectralFile", d_format)

    if fullpathname is None:
        fullpathname, _ = ie_tempfile("mat")
    path = _normalize_save_path(fullpathname)
    savemat(
        path,
        {
            "wavelength": wave,
            "data": stored,
            "comment": str(comment or ""),
            "dFormat": str(d_format),
        },
        do_compression=True,
    )
    return str(path)


def _resolve_spectra_path(
    fname: str | Path,
    *,
    asset_store: AssetStore | None = None,
) -> Path:
    path = Path(fname)
    if path.exists():
        return path
    store = asset_store or AssetStore.default()
    return store._resolve_spectra_path(fname)


def _coerce_comment_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _coerce_comment_text(value.item())
        flattened = [_coerce_comment_text(item) for item in value.reshape(-1).tolist()]
        return "\n".join(text for text in flattened if text)
    if isinstance(value, (list, tuple)):
        flattened = [_coerce_comment_text(item) for item in value]
        return "\n".join(text for text in flattened if text)
    return str(value)


def _column_index_from_a1(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha()).upper()
    index = 0
    for character in letters:
        index = index * 26 + (ord(character) - ord("A") + 1)
    return max(index - 1, 0)


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        xml_bytes = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml_bytes)
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for item in root.findall("main:si", namespace):
        fragments = [node.text or "" for node in item.findall(".//main:t", namespace)]
        strings.append("".join(fragments))
    return strings


def _read_xlsx_cells(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        worksheet_name = "xl/worksheets/sheet1.xml"
        if worksheet_name not in archive.namelist():
            worksheets = sorted(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))
            if not worksheets:
                raise ValueError("No worksheet found in spreadsheet.")
            worksheet_name = worksheets[0]
        root = ET.fromstring(archive.read(worksheet_name))

    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[str]] = []
    for row in root.findall(".//main:sheetData/main:row", namespace):
        values: dict[int, str] = {}
        max_col = -1
        for cell in row.findall("main:c", namespace):
            reference = cell.attrib.get("r", "")
            col_index = _column_index_from_a1(reference)
            max_col = max(max_col, col_index)
            cell_type = cell.attrib.get("t", "")
            if cell_type == "s":
                raw = cell.findtext("main:v", default="", namespaces=namespace)
                text = shared_strings[int(raw)] if raw else ""
            elif cell_type == "inlineStr":
                text = cell.findtext("main:is/main:t", default="", namespaces=namespace)
            else:
                text = cell.findtext("main:v", default="", namespaces=namespace)
            values[col_index] = text
        if max_col >= 0:
            rows.append([values.get(index, "") for index in range(max_col + 1)])
    return rows


def _read_delimited_cells(path: Path) -> list[list[str]]:
    sample = path.read_text(encoding="utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample[:2048], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return [list(row) for row in csv.reader(handle, dialect)]


def _spreadsheet_cells(path: Path) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _read_xlsx_cells(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return _read_delimited_cells(path)
    raise ValueError(f"Unsupported spreadsheet format '{path.suffix}'.")


def _spreadsheet_numeric_and_text(path: Path) -> tuple[np.ndarray, list[list[str]]]:
    cells = _spreadsheet_cells(path)
    if not cells:
        return np.zeros((0, 0), dtype=float), []

    width = max(len(row) for row in cells)
    numeric = np.full((len(cells), width), np.nan, dtype=float)
    for row_index, row in enumerate(cells):
        for col_index, value in enumerate(row):
            text = str(value).strip()
            if not text:
                continue
            try:
                numeric[row_index, col_index] = float(text)
            except ValueError:
                continue
    return numeric, cells


def vc_read_spectra(
    fname: str | Path,
    wave: Any | None,
    extrap_val: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    path = _resolve_spectra_path(fname, asset_store=asset_store)
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    source_wave = np.asarray(raw["wavelength"], dtype=float).reshape(-1)
    spectra = np.asarray(raw["data"], dtype=float)
    if spectra.ndim == 1:
        spectra = spectra.reshape(-1, 1)
    elif spectra.ndim == 2 and spectra.shape[0] != source_wave.size and spectra.shape[1] == source_wave.size:
        spectra = spectra.T

    target_wave = np.asarray(wave, dtype=float).reshape(-1) if wave is not None else np.asarray([], dtype=float)
    if target_wave.size:
        left_right = 0.0 if extrap_val is None else float(np.asarray(extrap_val, dtype=float).reshape(-1)[0])
        spectra = interp_spectra(
            source_wave,
            spectra,
            target_wave,
            left=left_right,
            right=left_right,
        )
        source_wave = target_wave

    comment = raw.get("comment", "")
    if isinstance(comment, np.ndarray) and comment.shape == ():
        comment = comment.item()
    return np.asarray(spectra, dtype=float), np.asarray(source_wave, dtype=float), str(comment)


def vc_save_multispectral_image(
    img_dir: str | Path,
    fname: str | None,
    mc_coef: Any,
    basis: Mapping[str, Any],
    basis_lights: Any | None = None,
    illuminant: Mapping[str, Any] | None = None,
    comment: str | None = None,
    img_mean: Any | None = None,
) -> str:
    output_dir = Path(img_dir)
    if not output_dir.exists():
        raise ValueError("No such directory.")
    full_name = output_dir / (fname or "multispectralImage.mat")
    return ie_save_multispectral_image(
        full_name,
        mc_coef,
        basis,
        comment=comment,
        img_mean=img_mean,
        illuminant=illuminant,
        basis_lights=basis_lights,
    )


def vc_import_object(
    obj_type: str = "scene",
    full_name: str | Path | None = None,
    preserve_data_flag: bool | None = None,
    *,
    session: SessionContext | None = None,
) -> tuple[Any, str]:
    del preserve_data_flag
    normalized = param_format(obj_type)
    if normalized not in {"scene", "opticalimage", "oi", "isa", "sensor", "vcimage", "ip", "display", "camera"}:
        raise UnsupportedOptionError("vcImportObject", obj_type)
    return vc_load_object(normalized, full_name, session=session)


def ie_web_get(*args: Any, **kwargs: Any) -> tuple[str, list[str]] | dict[str, list[str]] | str:
    """Download a file from a known remote resource using a MATLAB-style contract."""

    if args and isinstance(args[0], str):
        command = param_format(args[0])
        if command == "list":
            return {
                "deposits": sorted(_IE_WEB_GET_RESOURCES),
                "collections": sorted(_IE_WEB_GET_COLLECTIONS),
            }
        if command == "browse":
            deposit_name = str(args[1]) if len(args) > 1 else "pbrtv4"
            deposit_key = param_format(deposit_name)
            if deposit_key in _IE_WEB_GET_RESOURCES:
                return str(_IE_WEB_GET_RESOURCES[deposit_key])
            if deposit_key in _IE_WEB_GET_COLLECTIONS:
                return str(_IE_WEB_GET_COLLECTIONS[deposit_key])
            valid = ", ".join(sorted(set(_IE_WEB_GET_RESOURCES) | set(_IE_WEB_GET_COLLECTIONS)))
            raise ValueError(f"Invalid deposit name. Please choose from: {valid}")

    if len(args) % 2 != 0:
        raise ValueError("ieWebGet expects key/value pairs.")
    options = {param_format(args[index]): args[index + 1] for index in range(0, len(args), 2)}
    options.update({param_format(key): value for key, value in kwargs.items()})

    deposit_name = str(options.get("depositname", options.get("resourcetype", "pbrtv4")))
    deposit_key = param_format(deposit_name)
    if deposit_key not in _IE_WEB_GET_RESOURCES:
        valid = ", ".join(sorted(_IE_WEB_GET_RESOURCES))
        raise ValueError(f"Invalid deposit name. Please choose from: {valid}")

    deposit_file = str(options.get("depositfile", options.get("resourcefile", ""))).strip()
    if not deposit_file:
        raise ValueError("ieWebGet requires a deposit file/resource file in headless mode.")

    base_url = str(_IE_WEB_GET_RESOURCES[deposit_key]).rstrip("/")
    remote_name = Path(deposit_file).name
    local_name = str(options.get("localname", remote_name) or remote_name)
    download_dir = Path(options.get("downloaddir", Path.cwd())).expanduser()
    unzip_flag = bool(options.get("unzip", True))
    remove_zip_flag = bool(options.get("removezipfile", True))

    download_dir.mkdir(parents=True, exist_ok=True)
    local_file = download_dir / local_name
    if not local_file.exists():
        remote_url = f"{base_url}/{quote(remote_name)}"
        request = Request(remote_url, headers={"User-Agent": "pyisetcam/0.1.0"})
        with urlopen(request) as response, local_file.open("wb") as handle:
            handle.write(response.read())

    if unzip_flag and local_file.suffix.lower() == ".zip":
        target_dir = download_dir / local_file.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(local_file) as archive:
            archive.extractall(target_dir)
            filenames = sorted(str((target_dir / name).resolve()) for name in archive.namelist() if not name.endswith("/"))
        if remove_zip_flag and local_file.exists():
            local_file.unlink()
        return str(target_dir), filenames

    return str(local_file), []


def ie_xl2_color_filter(
    xl_fname: str | Path,
    vc_fname: str | Path | None = None,
    d_type: str = "colorfilter",
    filter_names: Any | None = None,
) -> tuple[str, np.ndarray, np.ndarray, str]:
    """Convert spreadsheet-like color-filter or spectral data into a MATLAB file."""

    path = Path(xl_fname).expanduser()
    numeric, cells = _spreadsheet_numeric_and_text(path)
    if numeric.size == 0 or numeric.shape[1] < 2:
        raise ValueError("Spreadsheet must contain wavelength plus at least one data column.")

    wavelength = np.asarray(numeric[:, 0], dtype=float)
    valid_wave = ~np.isnan(wavelength)
    wavelength = wavelength[valid_wave]

    data_columns: list[np.ndarray] = []
    data_indices: list[int] = []
    for col_index in range(1, numeric.shape[1]):
        column = np.asarray(numeric[:, col_index], dtype=float)
        valid = ~np.isnan(column)
        if int(np.sum(valid)) > 0:
            trimmed = column[valid]
            if trimmed.size != wavelength.size:
                raise ValueError("Data and wavelength must be the same length.")
            data_columns.append(trimmed)
            data_indices.append(col_index)
    if not data_columns:
        raise ValueError("Spreadsheet does not contain spectral data columns.")
    data = np.column_stack(data_columns).astype(float)

    header = cells[0] if cells else []
    inferred_names = [
        str(header[index]).strip()
        for index in data_indices
        if index < len(header) and str(header[index]).strip()
    ]
    comment = str(path.name)

    normalized_type = param_format(d_type)
    if normalized_type == "colorfilter":
        if np.max(data) > 1.0:
            data = data / 100.0
        if filter_names is None:
            if len(inferred_names) == data.shape[1]:
                resolved_names = inferred_names
            else:
                from .sensor import sensor_color_order

                ordering, _ = sensor_color_order("cell")
                resolved_names = [f"{ordering[index % len(ordering)]}Filter" for index in range(data.shape[1])]
        else:
            resolved_names = [str(value) for value in np.atleast_1d(filter_names).tolist()]
        if len(resolved_names) != data.shape[1]:
            raise ValueError("filter_names must match the number of spectral columns.")

        output = _normalize_save_path(vc_fname or path.with_suffix(".mat"))
        savemat(
            output,
            {
                "wavelength": wavelength,
                "data": data,
                "comment": comment,
                "filterNames": np.asarray(resolved_names, dtype=object),
            },
            do_compression=True,
        )
        return str(output), wavelength, data, comment

    if normalized_type in {"spectraldata", "spectral"}:
        output = ie_save_spectral_file(wavelength, data, comment, vc_fname or path.with_suffix(".mat"))
        return str(output), wavelength, data, comment

    raise UnsupportedOptionError("ieXL2ColorFilter", d_type)


def vc_read_image(
    fullname: str | Path | Any,
    image_type: str = "rgb",
    *args: Any,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None, dict[str, Any] | None, str, np.ndarray | None]:
    """Read image or multispectral data and return MATLAB-style photons plus metadata."""

    if fullname is None or (isinstance(fullname, str) and fullname == ""):
        return np.asarray([]), None, None, "", None

    normalized = param_format(image_type or "rgb")
    from .scene import scene_from_file

    if normalized in {"rgb", "unispectral", "monochrome"}:
        display = args[0] if args else None
        requested_type = "monochrome" if normalized in {"unispectral", "monochrome"} else "rgb"
        scene = scene_from_file(fullname, requested_type, None, display, asset_store=asset_store)
        illuminant = {
            "wave": np.asarray(scene.fields["wave"], dtype=float).copy(),
            "data": np.asarray(scene.fields["illuminant_energy"], dtype=float).copy(),
            "photons": np.asarray(scene.fields["illuminant_photons"], dtype=float).copy(),
            "comment": str(scene.fields.get("illuminant_comment", "")),
        }
        return np.asarray(scene.data["photons"], dtype=float), illuminant, None, "", None

    if normalized not in {"spectral", "multispectral", "hyperspectral"}:
        raise UnsupportedOptionError("vcReadImage", image_type)

    wave = args[0] if args else None
    scene = scene_from_file(fullname, normalized, None, None, wave, asset_store=asset_store)
    illuminant = {
        "wave": np.asarray(scene.fields["wave"], dtype=float).copy(),
        "data": np.asarray(scene.fields["illuminant_energy"], dtype=float).copy(),
        "photons": np.asarray(scene.fields["illuminant_photons"], dtype=float).copy(),
        "comment": str(scene.fields.get("illuminant_comment", "")),
    }

    basis = None
    comment = ""
    mc_coef = None
    if isinstance(fullname, (str, Path)):
        raw = loadmat(str(fullname), squeeze_me=True, struct_as_record=False)
        if "basis" in raw:
            basis = _deserialize_value(raw["basis"])
        if "comment" in raw:
            comment = _coerce_comment_text(raw["comment"])
        if "mcCOEF" in raw:
            mc_coef = np.asarray(raw["mcCOEF"], dtype=float)
        if "illuminant" in raw:
            illuminant = _deserialize_value(raw["illuminant"])

    return np.asarray(scene.data["photons"], dtype=float), illuminant, basis, comment, mc_coef


def ie_scp(
    user: str,
    host: str,
    src: str | Path,
    destination_path: str | Path,
    *args: Any,
    quiet: bool = False,
) -> tuple[str, int]:
    """Mirror MATLAB ieSCP() with a thin headless `scp -r` wrapper."""

    if len(args) % 2 != 0:
        raise ValueError("ieSCP expects key/value pairs.")
    for index in range(0, len(args), 2):
        parameter = param_format(args[index])
        value = args[index + 1]
        if parameter == "quiet":
            quiet = bool(value)
        else:
            raise UnsupportedOptionError("ieSCP", str(args[index]))

    command_parts = ["scp", "-r"]
    if quiet:
        command_parts.append("-q")
    command_parts.extend(
        [
            f"{str(user)}@{str(host)}:{Path(src).as_posix() if isinstance(src, Path) else str(src)}",
            str(destination_path),
        ]
    )
    command = shlex.join(command_parts)

    try:
        status = int(subprocess.run(command_parts, check=False).returncode)
    except FileNotFoundError:
        status = 127

    if not quiet:
        if status == 0:
            print("File copied successfully!")
        else:
            print("Error during remote secure copy. Check connection and paths.")
    return command, status


# MATLAB-style aliases.
ieImageType = ie_image_type
ieSCP = ie_scp
ieSaveSpectralFile = ie_save_spectral_file
ieStruct2XML = ie_struct2xml
ieTempfile = ie_tempfile
ieVarInFile = ie_var_in_file
ieWebGet = ie_web_get
ieXML2struct = ie_xml2struct
struct2xml = ie_struct2xml
xml2struct = ie_xml2struct
ieXL2ColorFilter = ie_xl2_color_filter
pathToLinux = path_to_linux
vcSaveObject = vc_save_object
vcExportObject = vc_export_object
vcImportObject = vc_import_object
vcLoadObject = vc_load_object
vcReadImage = vc_read_image
vcReadSpectra = vc_read_spectra
vcSaveMultiSpectralImage = vc_save_multispectral_image


def ie_save_si_data_file(
    psf: Any,
    wave: Any,
    um_per_samp: Any,
    f_name: str | Path | None = None,
) -> str:
    """Write MATLAB-style shift-invariant PSF data for `siSynthetic('custom', ...)`."""

    if psf is None:
        raise ValueError("psf volume required")
    if wave is None:
        raise ValueError("wavelength samples required (nm)")
    if um_per_samp is None:
        raise ValueError("Microns per sample(2-vector) required")

    path = _normalize_save_path(f_name or (Path.cwd() / "siSynthetic.mat"))
    psf_array = np.asarray(psf, dtype=float)
    wave_array = np.asarray(wave, dtype=float).reshape(-1)
    um_per_samp_array = np.asarray(um_per_samp, dtype=float).reshape(-1)
    if um_per_samp_array.size == 1:
        um_per_samp_array = np.repeat(um_per_samp_array, 2)
    if um_per_samp_array.size != 2:
        raise ValueError("umPerSamp must be a scalar or 2-vector.")

    notes = {"timeStamp": datetime.now().isoformat()}
    savemat(
        path,
        {
            "psf": psf_array,
            "wave": wave_array,
            "umPerSamp": um_per_samp_array,
            "notes": notes,
        },
        do_compression=True,
    )
    return str(path)


ieSaveSIDataFile = ie_save_si_data_file


def ie_save_multispectral_image(
    full_name: str | Path | None,
    mc_coef: Any,
    basis: Mapping[str, Any],
    comment: str | None = None,
    img_mean: Any | None = None,
    illuminant: Mapping[str, Any] | None = None,
    fov: float = 10.0,
    dist: float = 1.2,
    name: str | None = None,
    basis_lights: Any | None = None,
) -> str:
    """Write MATLAB-style basis-coded multispectral image data."""

    if mc_coef is None:
        raise ValueError("Coefficients required.")
    if basis is None:
        raise ValueError("Basis function required.")
    if illuminant is None:
        raise ValueError("Illuminant required.")

    path = _normalize_save_path(full_name or (Path.cwd() / "multispectralImage.mat"))
    basis_wave = np.asarray(basis["wave"], dtype=float).reshape(-1)
    basis_matrix = np.asarray(basis["basis"], dtype=float)
    if basis_matrix.shape[0] != basis_wave.size and basis_matrix.shape[1] == basis_wave.size:
        basis_matrix = basis_matrix.T
    if basis_matrix.shape[0] != basis_wave.size:
        raise ValueError("Basis matrix must align with the basis wavelength samples.")

    illuminant_wave = None
    illuminant_data = None
    if "wave" in illuminant:
        illuminant_wave = np.asarray(illuminant["wave"], dtype=float).reshape(-1)
    elif "wavelength" in illuminant:
        illuminant_wave = np.asarray(illuminant["wavelength"], dtype=float).reshape(-1)
    if "data" in illuminant:
        illuminant_data = np.asarray(illuminant["data"], dtype=float)
    elif "energy" in illuminant:
        illuminant_data = np.asarray(illuminant["energy"], dtype=float)
    elif "photons" in illuminant:
        source_wave = illuminant_wave if illuminant_wave is not None else basis_wave
        illuminant_data = quanta_to_energy(np.asarray(illuminant["photons"], dtype=float), source_wave)
    if illuminant_wave is None or illuminant_data is None:
        raise ValueError("Illuminant must provide wave and data/energy/photons fields.")

    payload: dict[str, Any] = {
        "mcCOEF": np.asarray(mc_coef, dtype=float),
        "basis": {
            "basis": basis_matrix,
            "wave": basis_wave,
        },
        "comment": str(comment or f"Date: {datetime.now().strftime('%Y-%m-%d')}"),
        "illuminant": {
            "wave": illuminant_wave,
            "data": illuminant_data,
        },
        "fov": float(fov),
        "dist": float(dist),
        "name": str(name or path.stem),
    }
    if img_mean is not None:
        payload["imgMean"] = np.asarray(img_mean, dtype=float).reshape(-1)
    if basis_lights is not None:
        payload["basisLights"] = np.asarray(basis_lights, dtype=float)

    savemat(path, payload, do_compression=True)
    return str(path)


ieSaveMultiSpectralImage = ie_save_multispectral_image


def ie_save_color_filter(
    in_data: Mapping[str, Any] | Sensor,
    full_file_name: str | Path | None = None,
) -> str:
    """Write MATLAB-style color filter data from a sensor or filter structure."""

    path = _normalize_save_path(full_file_name or (Path.cwd() / "colorFilter.mat"))

    if isinstance(in_data, Sensor):
        from .sensor import sensor_get

        payload: dict[str, Any] = {
            "wavelength": np.asarray(sensor_get(in_data, "wavelength"), dtype=float).reshape(-1),
            "data": np.asarray(sensor_get(in_data, "colorfilters"), dtype=float),
            "filterNames": np.asarray(list(sensor_get(in_data, "filternames")), dtype=object),
            "comment": str(in_data.metadata.get("comment", "No comment")),
        }
    elif isinstance(in_data, Mapping):
        if not {"wavelength", "data", "filterNames"} <= set(in_data.keys()):
            raise ValueError("Input data missing fields. No file written.")
        payload = {
            "wavelength": np.asarray(in_data["wavelength"], dtype=float).reshape(-1),
            "data": np.asarray(in_data["data"], dtype=float),
            "filterNames": np.asarray(list(np.atleast_1d(in_data["filterNames"])), dtype=object),
            "comment": str(in_data.get("comment", "No comment")),
        }
        for key, value in in_data.items():
            if key in {"wavelength", "data", "filterNames", "comment"}:
                continue
            payload[str(key)] = _serialize_value(value)
    else:
        raise ValueError("Input data missing fields. No file written.")

    savemat(path, payload, do_compression=True)
    return str(path)


ieSaveColorFilter = ie_save_color_filter


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _decode_tiff_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_tiff_value(value.item())
        return np.asarray([_decode_tiff_value(item) for item in value.tolist()])
    if isinstance(value, Mapping):
        return {str(key): _decode_tiff_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_tiff_value(item) for item in value]
    if isinstance(value, tuple):
        if len(value) == 2 and all(_is_numeric_scalar(item) for item in value):
            numerator = float(value[0])
            denominator = float(value[1])
            return numerator / denominator if denominator != 0 else np.nan
        if len(value) > 2 and len(value) % 2 == 0 and all(_is_numeric_scalar(item) for item in value):
            pairs = np.asarray(value, dtype=float).reshape(-1, 2)
            if np.all(pairs[:, 1] != 0):
                return pairs[:, 0] / pairs[:, 1]
        return tuple(_decode_tiff_value(item) for item in value)
    return value


def _dng_scalar(value: Any) -> Any:
    decoded = _decode_tiff_value(value)
    if isinstance(decoded, np.ndarray) and decoded.shape == (1,):
        return decoded.reshape(-1)[0].item()
    return decoded


def _select_raw_series(tif: TiffFile) -> Any:
    candidates = [series for series in tif.series if len(series.shape) == 2]
    if not candidates:
        raise ValueError("No raw mosaic image found in DNG file.")
    return max(candidates, key=lambda series: int(np.prod(series.shape)))


def _select_rgb_series(tif: TiffFile) -> Any:
    candidates = [series for series in tif.series if len(series.shape) == 3 and series.shape[-1] >= 3]
    if candidates:
        return max(candidates, key=lambda series: int(np.prod(series.shape)))
    if not tif.series:
        raise ValueError("No image series found in DNG file.")
    return tif.series[0]


def _extract_dng_info(tif: TiffFile, path: Path) -> dict[str, Any]:
    preview_page = tif.pages[0]
    raw_series = _select_raw_series(tif)
    raw_page = raw_series.pages[0]

    make_tag = preview_page.tags.get("Make")
    model_tag = preview_page.tags.get("Model")
    orientation_tag = preview_page.tags.get("Orientation")
    exif_tag = preview_page.tags.get("ExifTag")

    subifd_entry: dict[str, Any] = {}
    for tag_name in (
        "BlackLevel",
        "CFARepeatPatternDim",
        "CFAPattern",
        "ActiveArea",
        "DefaultCropOrigin",
        "DefaultCropSize",
    ):
        tag = raw_page.tags.get(tag_name)
        if tag is not None:
            subifd_entry[tag_name] = _decode_tiff_value(tag.value)

    digital_camera = _decode_tiff_value(exif_tag.value if exif_tag is not None else {})
    info: dict[str, Any] = {
        "Filename": str(path),
        "Make": str(_dng_scalar(make_tag.value)) if make_tag is not None else "",
        "Model": str(_dng_scalar(model_tag.value)) if model_tag is not None else "",
        "Orientation": int(_dng_scalar(orientation_tag.value)) if orientation_tag is not None else 1,
        "DigitalCamera": digital_camera,
        "SubIFDs": [subifd_entry],
        "ImageLength": int(raw_series.shape[0]),
        "ImageWidth": int(raw_series.shape[1]),
    }

    if isinstance(digital_camera, dict):
        if "ISOSpeedRatings" in digital_camera:
            info["ISOSpeedRatings"] = int(round(float(_dng_scalar(digital_camera["ISOSpeedRatings"]))))
        if "ExposureTime" in digital_camera:
            info["ExposureTime"] = float(_dng_scalar(digital_camera["ExposureTime"]))
    if "BlackLevel" in subifd_entry:
        info["BlackLevel"] = np.asarray(subifd_entry["BlackLevel"], dtype=float).reshape(-1)
    return info


def ie_dng_simple_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reduced MATLAB-style DNG metadata summary."""

    digital_camera = info.get("DigitalCamera")
    if isinstance(digital_camera, Mapping):
        iso_speed = digital_camera.get("ISOSpeedRatings")
        exposure_time = digital_camera.get("ExposureTime")
        subifds = info.get("SubIFDs", [])
        black_level = None
        if isinstance(subifds, list) and subifds:
            first_subifd = subifds[0]
            if isinstance(first_subifd, Mapping):
                black_level = first_subifd.get("BlackLevel")
        if black_level is None:
            black_level = info.get("BlackLevel")
        orientation = info.get("Orientation", 1)
    else:
        iso_speed = info.get("ISOSpeedRatings")
        exposure_time = info.get("ExposureTime")
        black_level = info.get("BlackLevel")
        orientation = info.get("Orientation", 1)

    black_level_array = np.asarray([] if black_level is None else black_level, dtype=float).reshape(-1)
    return {
        "isoSpeed": int(round(float(_dng_scalar(iso_speed)))) if iso_speed is not None else None,
        "exposureTime": float(_dng_scalar(exposure_time)) if exposure_time is not None else None,
        "blackLevel": black_level_array,
        "orientation": int(_dng_scalar(orientation)),
    }


def ie_dng_read(
    fname: str | Path,
    *args: Any,
    only_info: bool = False,
    simple_info: bool = False,
    rgb: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read a DNG file as raw mosaic or rendered RGB plus MATLAB-style metadata."""

    if len(args) % 2 != 0:
        raise ValueError("ieDNGRead expects key/value pairs.")
    for index in range(0, len(args), 2):
        parameter = param_format(args[index])
        value = args[index + 1]
        if parameter == "onlyinfo":
            only_info = bool(value)
        elif parameter == "simpleinfo":
            simple_info = bool(value)
        elif parameter == "rgb":
            rgb = bool(value)
        else:
            raise UnsupportedOptionError("ieDNGRead", str(args[index]))

    path = Path(fname)
    with TiffFile(path) as tif:
        info = _extract_dng_info(tif, path)
        reduced_info = ie_dng_simple_info(info) if simple_info else info
        if only_info:
            return None, reduced_info
        series = _select_rgb_series(tif) if rgb else _select_raw_series(tif)
        image = np.asarray(series.asarray())
    return image, reduced_info


def _dng_orientation_pattern(orientation: int) -> np.ndarray:
    mapping = {
        1: np.array([[1, 2], [2, 3]], dtype=int),
        3: np.array([[3, 2], [2, 1]], dtype=int),
        6: np.array([[2, 1], [3, 2]], dtype=int),
        8: np.array([[2, 3], [1, 2]], dtype=int),
    }
    if orientation not in mapping:
        raise ValueError(f"Unknown DNG orientation value: {orientation}")
    return mapping[orientation].copy()


def _normalize_sensor_dng_crop(crop: Any, size: tuple[int, int]) -> np.ndarray:
    crop_array = np.asarray(crop, dtype=float).reshape(-1)
    if crop_array.size == 4:
        row, col, height, width = crop_array[:4]
        return np.rint(np.array([col, row, width, height], dtype=float)).astype(int)
    if crop_array.size == 1:
        fraction = float(crop_array[0])
        if not (0.0 < fraction < 1.0):
            raise ValueError(f"Bad crop value {fraction}")
        sensor_size = np.asarray(size, dtype=float)
        middle_position = sensor_size / 2.0
        rowcol = fraction * sensor_size
        row = middle_position[0] - rowcol[0] / 2.0
        col = middle_position[1] - rowcol[1] / 2.0
        height = rowcol[0]
        width = rowcol[1]
        return np.rint(np.array([col, row, width, height], dtype=float)).astype(int)
    raise ValueError(f"Bad crop value {crop}")


def sensor_dng_read(
    fname: str | Path,
    *args: Any,
    asset_store: Any | None = None,
) -> tuple[Sensor, dict[str, Any]]:
    """Read a DNG file into an IMX363 sensor with MATLAB-style metadata handling."""

    full_info = True
    crop: Any = None
    if len(args) % 2 != 0:
        raise ValueError("sensorDNGRead expects key/value pairs.")
    for index in range(0, len(args), 2):
        parameter = param_format(args[index])
        value = args[index + 1]
        if parameter == "fullinfo":
            full_info = bool(value)
        elif parameter == "crop":
            crop = value
        else:
            raise UnsupportedOptionError("sensorDNGRead", str(args[index]))

    from .sensor import sensor_create, sensor_crop, sensor_get, sensor_set

    image, info = ie_dng_read(fname)
    if image is None:
        raise ValueError("ieDNGRead returned no image data.")
    simple_info = ie_dng_simple_info(info)
    black_level = int(np.ceil(float(simple_info["blackLevel"][0]))) if simple_info["blackLevel"].size else 0
    exposure_time = float(simple_info["exposureTime"]) if simple_info["exposureTime"] is not None else 0.0
    iso_speed = float(simple_info["isoSpeed"]) if simple_info["isoSpeed"] is not None else 1.0

    clipped_image = np.clip(np.asarray(image, dtype=float), black_level, None)
    sensor = sensor_create("IMX363", None, "isospeed", iso_speed, asset_store=asset_store)
    sensor = sensor_set(sensor, "size", clipped_image.shape[:2])
    sensor = sensor_set(sensor, "exp time", exposure_time)
    sensor = sensor_set(sensor, "black level", black_level)
    sensor = sensor_set(sensor, "name", str(Path(fname)))
    sensor = sensor_set(sensor, "digital values", clipped_image)
    sensor = sensor_set(sensor, "pattern", _dng_orientation_pattern(int(simple_info["orientation"])))

    if crop is not None:
        crop_rect = _normalize_sensor_dng_crop(crop, tuple(int(value) for value in sensor_get(sensor, "size")))
        sensor = sensor_crop(sensor, crop_rect)

    return sensor, info if full_info else simple_info


ieDNGRead = ie_dng_read
ieDNGSimpleInfo = ie_dng_simple_info
sensorDNGRead = sensor_dng_read
