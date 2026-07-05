"""Camera orchestration helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d

from .assets import AssetStore, ie_read_spectra
from .display import display_get
from .exceptions import UnsupportedOptionError
from .ip import ip_clear_data, ip_compute, ip_create, ip_get, ip_set
from .iso import iso12233, iso_find_slanted_bar
from .metrics import delta_e_ab, iso_acutance, xyz_from_energy, xyz_to_lab
from .optics import oi_clear_data, oi_compute, oi_create, oi_get, oi_set
from .scene import Scene, scene_adjust_illuminant, scene_adjust_luminance, scene_create, scene_from_file, scene_get, scene_set
from .scielab import scielab
from .session import ie_get_object, track_camera_session_state, track_session_object
from .sensor import (
    _chart_rectangles,
    _chart_roi,
    _macbeth_ideal_linear_rgb,
    sensor_compute,
    sensor_compute_full_array,
    sensor_create,
    sensor_create_ideal,
    sensor_clear_data,
    sensor_get,
    sensor_set,
    sensor_set_size_to_fov,
)
from .types import Camera, ImageProcessor, OpticalImage, Sensor, SessionContext
from .utils import (
    ie_parameter_otype,
    image_increase_image_rgb_size,
    linear_to_srgb,
    param_format,
    rgb_to_xw_format,
    split_prefixed_parameter,
    srgb_to_linear,
    xw_to_rgb_format,
    xyz_to_linear_srgb,
    xyz_to_srgb,
)


def _is_empty_dispatch_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip() == ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value).size == 0
    return False


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _track_camera_sequence(
    session: SessionContext | None,
    cameras: list[Camera],
) -> list[Camera]:
    if session is None:
        return cameras
    tracked: list[Camera] = []
    last_index = len(cameras) - 1
    for index, camera in enumerate(cameras):
        tracked.append(track_camera_session_state(session, camera, select=index == last_index))
    return tracked


def _l3_items(l3: Any) -> list[tuple[str, Any]]:
    if hasattr(l3, "items"):
        return [(str(key), value) for key, value in l3.items()]
    if hasattr(l3, "__dict__"):
        return [(str(key), value) for key, value in vars(l3).items()]
    raise ValueError("cameraCreate('L3', ...) requires a mapping-like L3 payload.")


def _l3_get(l3: Any, *names: str) -> Any:
    aliases = {param_format(name) for name in names}
    for key, value in _l3_items(l3):
        if param_format(key) in aliases:
            return value
    return None


def _l3_set(l3: Any, name: str, value: Any) -> None:
    target = param_format(name)
    if hasattr(l3, "items"):
        for key in list(l3.keys()):
            if param_format(str(key)) == target:
                l3[key] = value
                return
        l3[name] = value
        return
    if hasattr(l3, "__dict__"):
        for key in vars(l3):
            if param_format(str(key)) == target:
                setattr(l3, key, value)
                return
        setattr(l3, name, value)
        return
    raise ValueError("cameraCreate('L3', ...) requires a mapping-like L3 payload.")


def _l3_clear_data(l3: Any) -> Any:
    cleared = copy.deepcopy(l3)
    oi = _l3_get(cleared, "oi")
    if isinstance(oi, OpticalImage):
        _l3_set(cleared, "oi", oi_clear_data(oi))
    return cleared


def _canonical_camera_metric_name(metric_name: Any) -> str:
    normalized = param_format(metric_name)
    if normalized == "visiblesnr":
        return "vsnr"
    return normalized


@dataclass
class CameraMTFResult:
    """Headless payload returned by camera_mtf()."""

    freq: NDArray[np.float64]
    mtf: NDArray[np.float64]
    nyquistf: float
    lsf: NDArray[np.float64]
    lsfx: NDArray[np.float64]
    mtf50: float
    aliasingPercentage: float
    rect: NDArray[np.int_]
    vci: ImageProcessor
    esf: NDArray[np.float64] | None = None
    fitme: NDArray[np.float64] | None = None
    win: None = None


@dataclass
class CameraVSNRResult:
    """Headless payload returned by camera_vsnr()."""

    lightLevels: NDArray[np.float64]
    vSNR: NDArray[np.float64]
    eTime: NDArray[np.float64]
    rect: NDArray[np.int_]
    ip: list[ImageProcessor]
    oi: OpticalImage
    sensor: Sensor


@dataclass
class CameraFullReferenceResult:
    """Headless payload returned by camera_full_reference()."""

    sceneNames: list[str]
    meanLuminances: NDArray[np.float64]
    scielab: NDArray[np.float64]
    ssim: NDArray[np.float64]


@dataclass
class CameraMoireResult:
    """Headless payload returned by camera_moire()."""

    data_abIdeal_mean: NDArray[np.float64]
    data_abim_mean: NDArray[np.float64]
    cpd_mean: NDArray[np.float64]


def _pixel_get(sensor: Sensor, parameter: str, *args: Any) -> Any:
    pixel = sensor.fields["pixel"]
    key = param_format(parameter)
    mapping = {
        "size": np.asarray(pixel["size_m"], dtype=float),
        "pixelsize": np.asarray(pixel["size_m"], dtype=float),
        "fillfactor": float(pixel["fill_factor"]),
        "conversiongain": float(pixel["conversion_gain_v_per_electron"]),
        "conversiongainvpelectron": float(pixel["conversion_gain_v_per_electron"]),
        "voltageswing": float(pixel["voltage_swing"]),
        "darkvoltage": float(pixel["dark_voltage_v_per_sec"]),
        "darkvoltagevpersec": float(pixel["dark_voltage_v_per_sec"]),
        "readnoise": float(pixel["read_noise_v"]),
        "readnoisev": float(pixel["read_noise_v"]),
        "dsnu": float(pixel["dsnu_sigma_v"]),
        "dsnuv": float(pixel["dsnu_sigma_v"]),
        "prnu": float(pixel["prnu_sigma"]),
    }
    if key in mapping:
        return mapping[key]
    return sensor_get(sensor, f"pixel {parameter}", *args)


def _pixel_set(sensor: Sensor, parameter: str, value: Any) -> Sensor:
    key = param_format(parameter)
    if key in {"size", "pixelsize"}:
        size = np.asarray(value, dtype=float)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor.fields["pixel"]["size_m"] = size
        return sensor
    if key == "fillfactor":
        sensor.fields["pixel"]["fill_factor"] = float(value)
        return sensor
    if key in {"conversiongain", "conversiongainvpelectron"}:
        sensor.fields["pixel"]["conversion_gain_v_per_electron"] = float(value)
        return sensor
    if key == "voltageswing":
        sensor.fields["pixel"]["voltage_swing"] = float(value)
        return sensor
    if key in {"darkvoltage", "darkvoltagevpersec"}:
        sensor.fields["pixel"]["dark_voltage_v_per_sec"] = float(value)
        return sensor
    if key in {"readnoise", "readnoisev"}:
        sensor.fields["pixel"]["read_noise_v"] = float(value)
        return sensor
    if key in {"dsnu", "dsnuv"}:
        sensor.fields["pixel"]["dsnu_sigma_v"] = float(value)
        return sensor
    if key == "prnu":
        sensor.fields["pixel"]["prnu_sigma"] = float(value)
        return sensor
    return sensor_set(sensor, f"pixel {parameter}", value)


def camera_create(
    camera_type: str = "default",
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Camera | list[Camera]:
    """Create a supported camera."""

    store = _store(asset_store)
    resolved_type = "default" if _is_empty_dispatch_placeholder(camera_type) else camera_type
    normalized = param_format(resolved_type)
    camera = Camera(name=str(resolved_type))

    if normalized in {"default"}:
        oi = oi_create(session=session)
        sensor = sensor_create(asset_store=store, session=session)
    elif normalized in {"current"}:
        if session is None:
            oi = oi_create()
            sensor = sensor_create(asset_store=store)
            ip = ip_create(asset_store=store)
        else:
            current_oi = ie_get_object(session, "oi")
            oi = oi_create(session=session) if current_oi is None else current_oi

            current_sensor = ie_get_object(session, "sensor")
            sensor = sensor_create(asset_store=store, session=session) if current_sensor is None else current_sensor

            current_ip = ie_get_object(session, "ip")
            ip = ip_create(asset_store=store, session=session) if current_ip is None else current_ip

        camera.fields["oi"] = oi
        camera.fields["sensor"] = sensor
        camera.fields["ip"] = ip
        return track_camera_session_state(session, camera)
    elif normalized == "l3":
        if not args:
            raise FileNotFoundError(
                "cameraCreate('L3') default camera asset is not vendored; pass an L3 payload explicitly."
            )
        if len(args) != 1:
            raise ValueError("cameraCreate('L3', ...) accepts a single L3 payload.")
        l3 = args[0]
        oi_value = _l3_get(l3, "oi")
        sensor_value = _l3_get(l3, "design sensor", "design_sensor", "designsensor")
        if not isinstance(oi_value, OpticalImage):
            raise ValueError("cameraCreate('L3', ...) requires an OpticalImage at L3['oi'].")
        if not isinstance(sensor_value, Sensor):
            raise ValueError("cameraCreate('L3', ...) requires a Sensor at L3['design sensor'].")
        camera.name = "L3"
        camera.fields["oi"] = oi_clear_data(oi_value)
        camera.fields["sensor"] = sensor_value
        camera.fields["ip"] = ip_create(
            "L3",
            sensor_value,
            None,
            _l3_clear_data(l3),
            asset_store=store,
            session=session,
        )
        return track_camera_session_state(session, camera)
    elif normalized in {"ideal"}:
        oi = oi_create(session=session)
        sensor = sensor_create_ideal(asset_store=store, session=session)
    elif normalized in {"monochrome"}:
        oi = oi_create(session=session)
        sensor = sensor_create("monochrome", asset_store=store, session=session)
    elif normalized in {"idealmonochrome"}:
        camera.name = "ideal monochrome"
        oi = oi_create(session=session)
        sensor = sensor_create_ideal("monochrome", asset_store=store, session=session)
    else:
        try:
            oi = oi_create(session=session)
            sensor = sensor_create(camera_type, *args, asset_store=store, session=session)
        except UnsupportedOptionError as exc:
            raise UnsupportedOptionError("cameraCreate", camera_type) from exc

    if isinstance(sensor, list):
        cameras: list[Camera] = []
        for sensor_item in sensor:
            current = Camera(name=str(camera_type))
            current.fields["oi"] = oi_create(session=session)
            current.fields["sensor"] = sensor_item
            current.fields["ip"] = ip_create(sensor=sensor_item, asset_store=store, session=session)
            cameras.append(current)
        return _track_camera_sequence(session, cameras)

    camera.fields["oi"] = oi
    camera.fields["sensor"] = sensor
    camera.fields["ip"] = ip_create(sensor=sensor, asset_store=store, session=session)
    return track_camera_session_state(session, camera)


def camera_get(camera: Camera, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    if key == "vcitype":
        ip_name = param_format(str(camera.fields["ip"].name))
        return "l3" if ip_name in {"l3", "l3global"} else "default"

    prefix, remainder = split_prefixed_parameter(parameter, ("oi", "optics", "sensor", "pixel", "ip", "vci", "l3"))
    if prefix == "oi":
        if not remainder:
            return camera.fields["oi"]
        return oi_get(camera.fields["oi"], remainder, *args)
    if prefix == "optics":
        if not remainder:
            return camera.fields["oi"].fields["optics"]
        optics_param = "optics wvf" if remainder == "wvf" else f"optics {remainder}"
        return oi_get(camera.fields["oi"], optics_param, *args)
    if prefix == "sensor":
        if not remainder:
            return camera.fields["sensor"]
        return sensor_get(camera.fields["sensor"], remainder, *args)
    if prefix == "pixel":
        if not remainder:
            return camera.fields["sensor"].fields["pixel"]
        return _pixel_get(camera.fields["sensor"], remainder, *args)
    if prefix in {"ip", "vci"}:
        if not remainder:
            return camera.fields["ip"]
        return ip_get(camera.fields["ip"], remainder, *args)
    if prefix == "l3":
        l3 = camera.fields["ip"].fields.get("l3")
        if not remainder:
            return l3
        if l3 is None:
            return None
        return _l3_get(l3, remainder)

    if key == "type":
        return camera.type
    if key == "name":
        return camera.name
    if key == "oi":
        return camera.fields["oi"]
    if key == "sensor":
        return camera.fields["sensor"]
    if key in {"ip", "vci"}:
        return camera.fields["ip"]
    if key in {"metric", "metrics"}:
        metrics = camera.fields.get("metrics", {})
        if not args:
            return copy.deepcopy(metrics)
        requested_metric_name = param_format(args[0])
        metric_name = _canonical_camera_metric_name(args[0])
        if metric_name in metrics:
            return copy.deepcopy(metrics[metric_name])
        if requested_metric_name in metrics:
            return copy.deepcopy(metrics[requested_metric_name])
        from .metrics import metrics_camera

        return metrics_camera(camera, metric_name)
    if key == "image":
        return ip_get(camera.fields["ip"], "display data")

    object_type, object_param = ie_parameter_otype(parameter)
    if object_param is not None:
        if object_type == "oi":
            return oi_get(camera.fields["oi"], object_param, *args)
        if object_type == "optics":
            return oi_get(camera.fields["oi"], f"optics {object_param}", *args)
        if object_type == "wvf":
            return oi_get(camera.fields["oi"], f"wvf {object_param}", *args)
        if object_type in {"sensor", "pixel"}:
            return sensor_get(camera.fields["sensor"], object_param, *args)
        if object_type == "ip":
            return ip_get(camera.fields["ip"], object_param, *args)
        if object_type == "display":
            return ip_get(camera.fields["ip"], f"display {object_param}", *args)
        if object_type == "l3":
            l3 = camera.fields["ip"].fields.get("l3")
            if l3 is None:
                return None
            return _l3_get(l3, object_param)
    raise KeyError(f"Unsupported cameraGet parameter: {parameter}")


def camera_set(
    camera: Camera,
    parameter: str,
    value: Any,
    *args: Any,
    session: SessionContext | None = None,
) -> Camera:
    prefix, remainder = split_prefixed_parameter(parameter, ("oi", "optics", "sensor", "pixel", "ip", "vci", "l3"))
    if prefix == "oi":
        if not remainder:
            camera.fields["oi"] = track_session_object(session, value)
        else:
            camera.fields["oi"] = oi_set(camera.fields["oi"], remainder, value)
        return track_camera_session_state(session, camera)
    if prefix == "optics":
        if not remainder:
            camera.fields["oi"] = oi_set(camera.fields["oi"], "optics", value)
        else:
            optics_param = "optics wvf" if remainder == "wvf" else f"optics {remainder}"
            camera.fields["oi"] = oi_set(camera.fields["oi"], optics_param, value)
        return track_camera_session_state(session, camera)
    if prefix == "sensor":
        if not remainder:
            camera.fields["sensor"] = track_session_object(session, value)
        else:
            camera.fields["sensor"] = sensor_set(camera.fields["sensor"], remainder, value)
        return track_camera_session_state(session, camera)
    if prefix == "pixel":
        if not remainder:
            camera.fields["sensor"].fields["pixel"] = dict(value)
        else:
            camera.fields["sensor"] = _pixel_set(camera.fields["sensor"], remainder, value)
        return track_camera_session_state(session, camera)
    if prefix in {"ip", "vci"}:
        if not remainder:
            camera.fields["ip"] = value
        else:
            camera.fields["ip"] = ip_set(camera.fields["ip"], remainder, value, *args, session=session)
        return track_camera_session_state(session, camera)
    if prefix == "l3":
        normalized_remainder = param_format(remainder)
        if normalized_remainder == "sensorsize":
            camera.fields["sensor"] = sensor_set(camera.fields["sensor"], "size", value)
            l3 = camera.fields["ip"].fields.get("l3")
            if l3 is not None:
                l3 = copy.deepcopy(l3)
                design_sensor = _l3_get(l3, "design sensor", "design_sensor", "designsensor")
                if isinstance(design_sensor, Sensor):
                    _l3_set(l3, "design sensor", sensor_set(design_sensor, "size", value))
                    camera.fields["ip"] = ip_set(camera.fields["ip"], "l3", l3, session=session)
            return track_camera_session_state(session, camera)
        if normalized_remainder == "sensorfov":
            camera.fields["sensor"] = sensor_set_size_to_fov(camera.fields["sensor"], value, camera.fields["oi"])
            l3 = camera.fields["ip"].fields.get("l3")
            if l3 is not None:
                l3 = copy.deepcopy(l3)
                design_sensor = _l3_get(l3, "design sensor", "design_sensor", "designsensor")
                if isinstance(design_sensor, Sensor):
                    resized = sensor_set_size_to_fov(design_sensor, value, camera.fields["oi"])
                    _l3_set(l3, "design sensor", resized)
                    camera.fields["ip"] = ip_set(camera.fields["ip"], "l3", l3, session=session)
            return track_camera_session_state(session, camera)
        if remainder:
            raise KeyError(f"Unsupported cameraSet parameter: {parameter}")
        camera.fields["ip"] = ip_set(camera.fields["ip"], "l3", value, session=session)
        return track_camera_session_state(session, camera)

    key = param_format(parameter)
    if key == "name":
        camera.name = str(value)
        return track_camera_session_state(session, camera)
    if key == "type":
        camera.type = str(value)
        return track_camera_session_state(session, camera)
    if key == "oi":
        camera.fields["oi"] = track_session_object(session, value)
        return track_camera_session_state(session, camera)
    if key == "sensor":
        camera.fields["sensor"] = track_session_object(session, value)
        return track_camera_session_state(session, camera)
    if key in {"ip", "vci"}:
        camera.fields["ip"] = value
        return track_camera_session_state(session, camera)
    if key == "l3sensorsize":
        return camera_set(camera, "l3 sensor size", value, session=session)
    if key == "l3sensorfov":
        return camera_set(camera, "l3 sensor fov", value, session=session)
    if key in {"metric", "metrics"}:
        if not args:
            raise ValueError("cameraSet(..., 'metric', value, metric_name) requires a metric name.")
        metric_name = _canonical_camera_metric_name(args[0])
        metrics = dict(camera.fields.get("metrics", {}))
        metrics[metric_name] = copy.deepcopy(value)
        camera.fields["metrics"] = metrics
        return track_camera_session_state(session, camera)

    object_type, object_param = ie_parameter_otype(parameter)
    if object_param is not None:
        if object_type == "oi":
            camera.fields["oi"] = oi_set(camera.fields["oi"], object_param, value, *args)
            return track_camera_session_state(session, camera)
        if object_type == "optics":
            camera.fields["oi"] = oi_set(camera.fields["oi"], f"optics {object_param}", value, *args)
            return track_camera_session_state(session, camera)
        if object_type == "wvf":
            camera.fields["oi"] = oi_set(camera.fields["oi"], f"wvf {object_param}", value, *args)
            return track_camera_session_state(session, camera)
        if object_type in {"sensor", "pixel"}:
            camera.fields["sensor"] = sensor_set(camera.fields["sensor"], object_param, value, *args)
            return track_camera_session_state(session, camera)
        if object_type == "ip":
            camera.fields["ip"] = ip_set(camera.fields["ip"], object_param, value, *args, session=session)
            return track_camera_session_state(session, camera)
        if object_type == "display":
            camera.fields["ip"] = ip_set(camera.fields["ip"], f"display {object_param}", value, *args, session=session)
            return track_camera_session_state(session, camera)
    raise KeyError(f"Unsupported cameraSet parameter: {parameter}")


def camera_compute(
    camera: Camera,
    p_type: str | Scene = "sensor",
    mode: Any = "normal",
    sensor_resize: bool = True,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Camera:
    """Run the supported camera pipeline."""

    store = _store(asset_store)
    if mode is None:
        adjust_scale = None
        normalized_mode = "normal"
    elif isinstance(mode, str):
        adjust_scale = None
        normalized_mode = "normal" if not mode.strip() else param_format(mode)
    else:
        mode_array = np.asarray(mode)
        if mode_array.size == 0:
            adjust_scale = None
            normalized_mode = "normal"
        else:
            adjust_scale = np.asarray(mode, dtype=float)
            normalized_mode = "normal"
    if normalized_mode not in {"normal", "idealxyz"}:
        raise UnsupportedOptionError("cameraCompute", mode)

    if isinstance(p_type, Scene):
        scene = p_type
        start_type = "scene"
    elif p_type is None:
        scene = None
        start_type = "sensor"
    elif isinstance(p_type, str) and not p_type.strip():
        scene = None
        start_type = "sensor"
    else:
        p_type_array = None
        if isinstance(p_type, (list, tuple, np.ndarray)):
            p_type_array = np.asarray(p_type)
        scene = None
        start_type = "sensor" if p_type_array is not None and p_type_array.size == 0 else param_format(p_type)

    if sensor_resize is None:
        sensor_resize_flag = True
    elif isinstance(sensor_resize, str):
        sensor_resize_flag = True if not sensor_resize.strip() else bool(sensor_resize)
    elif isinstance(sensor_resize, (list, tuple, np.ndarray)):
        sensor_resize_array = np.asarray(sensor_resize)
        sensor_resize_flag = True if sensor_resize_array.size == 0 else bool(sensor_resize_array.reshape(-1)[0])
    else:
        sensor_resize_flag = bool(sensor_resize)

    oi: OpticalImage = camera.fields["oi"]
    sensor: Sensor = camera.fields["sensor"]
    ip = camera.fields["ip"]

    if normalized_mode == "idealxyz":
        if start_type == "scene":
            if scene is None:
                raise ValueError("A Scene object is required when starting from scene.")
            camera, xyz_ideal = _camera_ideal_xyz(
                camera,
                scene,
                asset_store=store,
                session=session,
            )
        elif start_type in {"oi", "sensor"}:
            camera = camera.clone()
            oi = camera.fields["oi"]
            sensor = sensor_set(camera_get(camera, "sensor").clone(), "noise flag", -1)
            wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
            sensor = sensor_set(sensor, "wave", wave)
            xyz_quanta = np.asarray(ie_read_spectra("XYZQuanta", wave, asset_store=store), dtype=float)
            xyz_ideal, _ = sensor_compute_full_array(sensor, oi, xyz_quanta)
            camera = camera_set(camera, "oi", oi, session=session)
            camera = camera_set(camera, "sensor", sensor, session=session)
        else:
            raise UnsupportedOptionError("cameraCompute", str(p_type))

        xyz_ideal = np.asarray(xyz_ideal, dtype=float)
        srgb, linear_rgb = _xyz_to_srgb_pair(xyz_ideal)
        ip = ip.clone()
        ip = ip_set(ip, "wave", np.asarray(oi_get(camera.fields["oi"], "wave"), dtype=float), session=session)
        ip = ip_set(ip, "internal cs", "XYZ", session=session)
        ip = ip_set(ip, "conversion method sensor", "None", session=session)
        ip = ip_set(ip, "correction method illuminant", "None", session=session)
        ip = ip_set(ip, "sensor conversion matrix", np.eye(3, dtype=float), session=session)
        ip = ip_set(ip, "illuminant correction matrix", np.eye(3, dtype=float), session=session)
        ip = ip_set(ip, "ics2display transform", np.eye(3, dtype=float), session=session)
        ip.data["input"] = xyz_ideal.copy()
        ip.data["xyz"] = xyz_ideal.copy()
        ip.data["ics"] = xyz_ideal.copy()
        ip.data["sensorspace"] = xyz_ideal.copy()
        ip.data["display_rgb"] = srgb.copy()
        ip.data["srgb"] = srgb.copy()
        ip.data["result"] = linear_rgb.copy()
        camera.fields["ip"] = ip
        camera.data["result"] = ip.data.get("result")
        return track_camera_session_state(session, camera)

    if start_type == "scene":
        if scene is None:
            raise ValueError("A Scene object is required when starting from scene.")
        if sensor_resize_flag:
            scene_hfov = float(scene.fields["fov_deg"])
            scene_vfov = float(scene.fields["vfov_deg"])
            sensor_hfov = float(sensor_get(sensor, "fov horizontal", scene, oi))
            sensor_vfov = float(sensor_get(sensor, "fov vertical", scene, oi))
            if (
                abs((scene_hfov - sensor_hfov) / max(scene_hfov, 1e-12)) > 0.01
                or abs((scene_vfov - sensor_vfov) / max(scene_vfov, 1e-12)) > 0.01
            ):
                sensor = sensor_set_size_to_fov(sensor.clone(), (scene_hfov, scene_vfov), oi)
        oi = oi_compute(oi, scene, session=session)
        sensor = sensor_compute(sensor, oi, session=session)
        ip = ip_compute(ip, sensor, asset_store=store, session=session)
    elif start_type == "oi":
        sensor = sensor_compute(sensor, oi, session=session)
        ip = ip_compute(ip, sensor, asset_store=store, session=session)
    elif start_type == "sensor":
        ip = ip_compute(ip, sensor, asset_store=store, session=session)
    else:
        raise UnsupportedOptionError("cameraCompute", str(p_type))

    if adjust_scale is not None:
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        mean_result = _camera_center_mean(result)
        mean_scale = _camera_center_mean(adjust_scale)
        if mean_result > 0.0:
            result = result * (mean_scale / mean_result)
            ip = ip_set(ip, "result", result, session=session)

    camera.fields["oi"] = oi
    camera.fields["sensor"] = sensor
    camera.fields["ip"] = ip
    camera.data["result"] = ip.data.get("result")
    return track_camera_session_state(session, camera)


def _camera_crop_border(image: Any, border: int = 10) -> np.ndarray:
    data = np.asarray(image)
    if data.ndim < 2:
        return np.asarray(data)
    if data.shape[0] <= (2 * border) or data.shape[1] <= (2 * border):
        return np.asarray(data)
    slices = [slice(border, data.shape[0] - border), slice(border, data.shape[1] - border)]
    slices.extend([slice(None)] * (data.ndim - 2))
    return np.asarray(data[tuple(slices)])


def _camera_center_mean(image: Any, border: int = 10) -> float:
    cropped = _camera_crop_border(image, border=border)
    return float(np.mean(np.asarray(cropped, dtype=float)))


def _camera_scene_input(scene_name: str | Scene, *, asset_store: AssetStore) -> Scene:
    if isinstance(scene_name, Scene):
        return scene_name.clone()

    requested = Path(str(scene_name)).expanduser()
    candidates = [requested]
    if requested.suffix:
        candidates.append(Path("data/images/multispectral") / requested.name)
        candidates.append(Path("data/images/multispectral") / requested)
    else:
        candidates.append(Path(f"{requested}.mat"))
        candidates.append(Path("data/images/multispectral") / requested)
        candidates.append(Path("data/images/multispectral") / f"{requested}.mat")

    resolved = None
    for candidate in candidates:
        try:
            resolved = asset_store.resolve(candidate)
            break
        except Exception:
            continue
    if resolved is None:
        matches = list(asset_store.ensure().rglob(f"{requested.stem}.mat"))
        if not matches:
            raise ValueError(f"Unable to resolve multispectral scene {scene_name!r}.")
        resolved = matches[0]
    return scene_from_file(resolved, "multispectral", asset_store=asset_store)


def _camera_ideal_xyz(
    camera: Camera,
    scene: Scene,
    *,
    asset_store: AssetStore,
    session: SessionContext | None = None,
) -> tuple[Camera, np.ndarray]:
    working = camera.clone()
    oi = oi_compute(camera_get(working, "oi"), scene, session=session)
    sensor = sensor_set(camera_get(working, "sensor").clone(), "noise flag", -1)
    wave = np.asarray(oi_get(oi, "wave"), dtype=float).reshape(-1)
    sensor = sensor_set(sensor, "wave", wave)
    xyz_quanta = np.asarray(ie_read_spectra("XYZQuanta", wave, asset_store=asset_store), dtype=float)
    xyz_ideal, _ = sensor_compute_full_array(sensor, oi, xyz_quanta)
    working = camera_set(working, "oi", oi, session=session)
    working = camera_set(working, "sensor", sensor, session=session)
    return working, np.asarray(xyz_ideal, dtype=float)


def _xyz_to_srgb_pair(xyz: Any) -> tuple[np.ndarray, np.ndarray]:
    xyz_image = np.asarray(xyz, dtype=float)
    scaled_xyz = xyz_image.copy()
    if scaled_xyz.ndim != 3 or scaled_xyz.shape[2] != 3:
        raise ValueError("XYZ image must be rows x cols x 3.")
    max_y = float(np.max(scaled_xyz[:, :, 1])) if scaled_xyz.size else 1.0
    if max_y > 1.0:
        scaled_xyz = scaled_xyz / max_y
    if float(np.min(scaled_xyz)) < 0.0:
        scaled_xyz = np.clip(scaled_xyz, 0.0, 1.0)
    linear_rgb = np.clip(np.asarray(xyz_to_linear_srgb(scaled_xyz), dtype=float), 0.0, 1.0)
    return np.asarray(xyz_to_srgb(xyz_image), dtype=float), linear_rgb


def camera_clear_data(camera: Camera, *, session: SessionContext | None = None) -> Camera:
    """Clear computed OI/sensor/IP payloads from a camera."""

    cleared = camera.clone()
    cleared.fields["oi"] = oi_clear_data(camera_get(cleared, "oi"))
    cleared.fields["sensor"] = sensor_clear_data(camera_get(cleared, "sensor"))
    cleared.fields["ip"] = ip_clear_data(camera_get(cleared, "ip"))
    if "l3" in cleared.fields["ip"].fields:
        cleared.fields["ip"].fields["l3"] = None
    cleared.data = {}
    return track_camera_session_state(session, cleared)


def camera_compute_srgb(
    camera: Camera,
    scene_name: str | Scene,
    mean_luminance: float = 100.0,
    sz: Any | None = None,
    scenefov: float | None = None,
    scaleoutput: float = 1.0,
    plot_flag: int = 0,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Camera]:
    """Compute MATLAB-style camera result and ideal sRGB images."""

    del plot_flag
    store = _store(asset_store)
    working = camera.clone()
    scene = _camera_scene_input(scene_name, asset_store=store)
    scene = scene_adjust_illuminant(scene, "D65.mat", asset_store=store)

    mean_luminance_is_omitted = mean_luminance is None
    if isinstance(mean_luminance, str) and not mean_luminance.strip():
        mean_luminance_is_omitted = True
    elif isinstance(mean_luminance, (list, tuple, np.ndarray)):
        mean_luminance_array = np.asarray(mean_luminance)
        mean_luminance_is_omitted = mean_luminance_array.size == 0
    mean_luminance_value = 100.0 if mean_luminance_is_omitted else float(mean_luminance)

    sz_is_omitted = sz is None
    if isinstance(sz, str) and not sz.strip():
        sz_is_omitted = True
    elif isinstance(sz, (list, tuple, np.ndarray)):
        sz_array = np.asarray(sz)
        sz_is_omitted = sz_array.size == 0

    if not sz_is_omitted:
        size = np.asarray(sz, dtype=int).reshape(-1)
        if size.size == 1:
            size = np.repeat(size, 2)
        sensor = sensor_set(camera_get(working, "sensor").clone(), "size", tuple((size[:2] + 20).tolist()))
        working = camera_set(working, "sensor", sensor, session=session)

    scenefov_is_omitted = scenefov is None
    if isinstance(scenefov, str) and not scenefov.strip():
        scenefov_is_omitted = True
    elif isinstance(scenefov, (list, tuple, np.ndarray)):
        scenefov_array = np.asarray(scenefov)
        scenefov_is_omitted = scenefov_array.size == 0

    if scenefov_is_omitted:
        oi = camera_get(working, "oi")
        sensor = camera_get(working, "sensor")
        scene_distance = float(scene_get(scene, "distance"))
        scenefov = float(sensor_get(sensor, "fov", scene_distance, oi))
    scene = scene_set(scene, "fov", float(scenefov))

    scaleoutput_is_omitted = False
    if isinstance(scaleoutput, str) and not scaleoutput.strip():
        scaleoutput_is_omitted = True
    elif isinstance(scaleoutput, (list, tuple, np.ndarray)):
        scaleoutput_array = np.asarray(scaleoutput)
        scaleoutput_is_omitted = scaleoutput_array.size == 0
    scaleoutput_value = 1.0 if scaleoutput_is_omitted else float(scaleoutput)

    scene = scene_adjust_luminance(scene, mean_luminance_value, asset_store=store)

    working, xyz_ideal = _camera_ideal_xyz(working, scene, asset_store=store, session=session)
    xyz_ideal = xyz_ideal / max(float(np.max(xyz_ideal)), 1.0e-12) * scaleoutput_value
    srgb_ideal, lrgb_ideal = _xyz_to_srgb_pair(xyz_ideal)

    working = camera_compute(working, scene, asset_store=store, session=session)
    lrgb_result = np.asarray(camera_get(working, "image"), dtype=float)
    mean_result = _camera_center_mean(lrgb_result)
    mean_ideal = _camera_center_mean(lrgb_ideal)
    if mean_result > 0.0:
        lrgb_result = lrgb_result * (mean_ideal / mean_result)
        working = camera_set(working, "ip result", lrgb_result, session=session)

    srgb_result = linear_to_srgb(np.clip(lrgb_result, 0.0, 1.0))
    raw = np.asarray(camera_get(working, "sensor volts"), dtype=float)
    return (
        _camera_crop_border(srgb_result),
        _camera_crop_border(srgb_ideal),
        _camera_crop_border(raw),
        working,
    )


def camera_compute_sequence(
    camera: Camera,
    *args: Any,
    scenes: Any = None,
    exposuretimes: Any = None,
    nframes: int | None = None,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[Camera, list[np.ndarray]]:
    """Compute one or more frames with scene/exposure sequences."""

    store = _store(asset_store)
    if args:
        if len(args) % 2 != 0:
            raise ValueError("cameraComputeSequence expects MATLAB-style key/value pairs.")
        for index in range(0, len(args), 2):
            key = param_format(args[index])
            value = args[index + 1]
            if key == "scenes" and scenes is None:
                scenes = value
            elif key == "exposuretimes" and exposuretimes is None:
                exposuretimes = value
            elif key == "nframes" and nframes is None:
                nframes = value
            elif key not in {"scenes", "exposuretimes", "nframes"}:
                raise ValueError(f"cameraComputeSequence does not support parameter {args[index]!r}.")
    if isinstance(scenes, str) and not scenes.strip():
        scenes = None
    elif isinstance(scenes, (list, tuple, np.ndarray)) and np.asarray(scenes).size == 0:
        scenes = None
    if scenes is None:
        scenes = ie_get_object(session, "scene") if session is not None else None
    if scenes is None:
        raise ValueError("cameraComputeSequence requires one or more scenes.")

    if isinstance(exposuretimes, str) and not exposuretimes.strip():
        exposuretimes = None
    elif isinstance(exposuretimes, (list, tuple, np.ndarray)) and np.asarray(exposuretimes).size == 0:
        exposuretimes = None
    if exposuretimes is None:
        exposuretimes = 1.0

    if isinstance(nframes, str) and not nframes.strip():
        nframes = None
    elif isinstance(nframes, (list, tuple, np.ndarray)) and np.asarray(nframes).size == 0:
        nframes = None

    scene_list = list(scenes) if isinstance(scenes, (list, tuple)) else [scenes]
    exposure_list = list(np.asarray(exposuretimes, dtype=float).reshape(-1)) if not np.isscalar(exposuretimes) else [float(exposuretimes)]
    total_frames = int(max(len(scene_list), len(exposure_list), 1 if nframes is None else nframes))

    if len(scene_list) == 1 and total_frames > 1:
        scene_list = scene_list * total_frames
    if len(exposure_list) == 1 and total_frames > 1:
        exposure_list = exposure_list * total_frames
    if len(scene_list) != len(exposure_list):
        raise ValueError("cameraComputeSequence requires scenes and exposuretimes to broadcast to the same length.")

    working = camera.clone()
    images: list[np.ndarray] = []
    for scene_item, exposure_time in zip(scene_list, exposure_list, strict=True):
        current_scene = _camera_scene_input(scene_item, asset_store=store) if not isinstance(scene_item, Scene) else scene_item.clone()
        sensor = sensor_set(camera_get(working, "sensor").clone(), "exposure time", float(exposure_time))
        working = camera_set(working, "sensor", sensor, session=session)
        working = camera_compute(working, current_scene, asset_store=store, session=session)
        images.append(np.asarray(camera_get(working, "image"), dtype=float).copy())
    return working, images


def camera_full_reference(
    camera: Camera,
    scene_names: Any | None = None,
    mean_luminances: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraFullReferenceResult:
    """Compute MATLAB-style full-reference camera metrics without GUI output."""

    store = _store(asset_store)
    scene_names_is_omitted = scene_names is None
    if isinstance(scene_names, str) and not scene_names.strip():
        scene_names_is_omitted = True
    elif isinstance(scene_names, (list, tuple, np.ndarray)):
        scene_names_array = np.asarray(scene_names, dtype=object)
        scene_names_is_omitted = scene_names_array.size == 0

    scene_list = list(scene_names) if isinstance(scene_names, (list, tuple)) else [scene_names]
    if scene_names_is_omitted:
        scene_list = ["StuffedAnimals_tungsten-hdrs"]

    mean_luminances_is_omitted = mean_luminances is None
    if isinstance(mean_luminances, str) and not mean_luminances.strip():
        mean_luminances_is_omitted = True
    elif isinstance(mean_luminances, (list, tuple, np.ndarray)):
        mean_luminances_array = np.asarray(mean_luminances)
        mean_luminances_is_omitted = mean_luminances_array.size == 0

    luminances = np.asarray(
        [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 400.0] if mean_luminances_is_omitted else mean_luminances,
        dtype=float,
    ).reshape(-1)
    if luminances.size == 0:
        raise ValueError("cameraFullReference requires one or more mean luminances.")

    scielab_metric = np.zeros((len(scene_list), luminances.size), dtype=float)
    ssim_metric = np.zeros_like(scielab_metric)
    resolved_names: list[str] = []

    for scene_index, scene_name in enumerate(scene_list):
        source_scene = _camera_scene_input(scene_name, asset_store=store)
        resolved_names.append(source_scene.name if source_scene.name else str(scene_name))
        base_scene = scene_adjust_illuminant(source_scene, "D65.mat", asset_store=store)

        oi = camera_get(camera, "oi")
        sensor = camera_get(camera, "sensor")
        scene_distance = float(scene_get(base_scene, "distance"))
        fov = float(sensor_get(sensor, "fov", scene_distance, oi))
        base_scene = scene_set(base_scene, "fov", 1.26 * fov)

        for luminance_index, mean_luminance in enumerate(luminances):
            working_scene = scene_adjust_luminance(base_scene.clone(), float(mean_luminance), asset_store=store)

            working_camera, xyz_ideal = _camera_ideal_xyz(camera.clone(), working_scene, asset_store=store, session=session)
            xyz_ideal = np.asarray(xyz_ideal, dtype=float)
            xyz_ideal /= max(float(np.max(xyz_ideal)), 1.0e-12)

            _, lrgb_ideal = _xyz_to_srgb_pair(xyz_ideal)
            working_camera = camera_compute(working_camera, working_scene, asset_store=store, session=session)
            lrgb_result = np.asarray(camera_get(working_camera, "image"), dtype=float)

            mean_result = _camera_center_mean(lrgb_result)
            mean_ideal = _camera_center_mean(lrgb_ideal)
            if mean_result > 0.0:
                lrgb_result = lrgb_result * (mean_ideal / mean_result)

            xyz_ideal_cropped = _camera_crop_border(xyz_ideal)
            lrgb_ideal_cropped = _camera_crop_border(lrgb_ideal)
            lrgb_result_cropped = _camera_crop_border(lrgb_result)
            target_shape = np.minimum(
                np.asarray(xyz_ideal_cropped.shape[:2], dtype=int),
                np.asarray(lrgb_result_cropped.shape[:2], dtype=int),
            )
            xyz_ideal_cropped = _center_crop_to_shape(xyz_ideal_cropped, target_shape)
            lrgb_ideal_cropped = _center_crop_to_shape(lrgb_ideal_cropped, target_shape)
            lrgb_result_cropped = _center_crop_to_shape(lrgb_result_cropped, target_shape)
            srgb_ideal = linear_to_srgb(np.clip(lrgb_ideal_cropped, 0.0, 1.0))
            srgb_result = linear_to_srgb(np.clip(lrgb_result_cropped, 0.0, 1.0))
            xyz_result = _linear_srgb_to_xyz(np.asarray(srgb_to_linear(srgb_result), dtype=float))

            white_point = np.asarray(scene_get(working_scene, "illuminant xyz"), dtype=float).reshape(-1)
            white_point /= max(float(np.max(white_point)), 1.0e-12)
            delta_e_image, _, _, _ = scielab(
                xyz_ideal_cropped,
                np.asarray(xyz_result, dtype=float),
                white_point,
                {"imageFormat": "xyz"},
            )
            scielab_metric[scene_index, luminance_index] = float(np.mean(np.asarray(delta_e_image, dtype=float), dtype=float))

            gray_ideal = np.uint8(np.clip(_rgb_to_gray(srgb_ideal) * 255.0, 0.0, 255.0))
            gray_result = np.uint8(np.clip(_rgb_to_gray(srgb_result) * 255.0, 0.0, 255.0))
            ssim_metric[scene_index, luminance_index] = _ssim_index(gray_ideal, gray_result)

    return CameraFullReferenceResult(
        sceneNames=resolved_names,
        meanLuminances=luminances,
        scielab=scielab_metric,
        ssim=ssim_metric,
    )


def camera_moire(
    camera: Camera,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraMoireResult:
    """Compute a headless MATLAB-style moire metric summary."""

    store = _store(asset_store)
    mean_luminance = 100.0
    fov_scene = 10.0

    working_camera = camera.clone()
    sensor = sensor_set_size_to_fov(camera_get(working_camera, "sensor").clone(), fov_scene, camera_get(working_camera, "oi"))
    working_camera = camera_set(working_camera, "sensor", sensor, session=session)

    scene = scene_create("moire orient", asset_store=store, session=session)
    scene = scene_set(scene, "wave", np.asarray(camera_get(working_camera, "sensor wave"), dtype=float))
    scene = scene_adjust_illuminant(scene, "D65.mat", asset_store=store)
    scene = scene_set(scene, "fov", fov_scene)
    scene = scene_adjust_luminance(scene, mean_luminance, asset_store=store)

    white_point = np.asarray(scene_get(scene, "illuminant xyz"), dtype=float).reshape(-1)
    white_point /= max(float(np.max(white_point)), 1.0e-12)

    ideal_camera, xyz_ideal = _camera_ideal_xyz(working_camera, scene, asset_store=store, session=session)
    xyz_ideal = np.asarray(xyz_ideal, dtype=float)
    xyz_ideal /= max(float(np.max(xyz_ideal)), 1.0e-12)
    _, lrgb_ideal = _xyz_to_srgb_pair(xyz_ideal)

    result_camera = camera_set(camera.clone(), "oi", camera_get(ideal_camera, "oi"), session=session)
    result_camera = camera_set(result_camera, "sensor", camera_get(camera, "sensor").clone(), session=session)
    result_camera = camera_compute(result_camera, "oi", lrgb_ideal, asset_store=store, session=session)
    lrgb_result = np.asarray(camera_get(result_camera, "image"), dtype=float)

    xyz_ideal_cropped = _camera_crop_border(xyz_ideal)
    lrgb_result_cropped = _camera_crop_border(lrgb_result)
    target_shape = np.minimum(
        np.asarray(xyz_ideal_cropped.shape[:2], dtype=int),
        np.asarray(lrgb_result_cropped.shape[:2], dtype=int),
    )
    xyz_ideal_cropped = _center_crop_to_shape(xyz_ideal_cropped, target_shape)
    lrgb_result_cropped = _center_crop_to_shape(lrgb_result_cropped, target_shape)
    if xyz_ideal_cropped.shape[0] > 4 and xyz_ideal_cropped.shape[1] > 4:
        xyz_ideal_cropped = xyz_ideal_cropped[2:-2, 2:-2, :]
        lrgb_result_cropped = lrgb_result_cropped[2:-2, 2:-2, :]

    xyz_result = _linear_srgb_to_xyz(np.asarray(np.clip(lrgb_result_cropped, 0.0, 1.0), dtype=float))
    lab_ideal = np.asarray(xyz_to_lab(xyz_ideal_cropped, white_point), dtype=float)
    lab_result = np.asarray(xyz_to_lab(xyz_result, white_point), dtype=float)
    ab_ideal = np.sqrt(np.sum(np.square(lab_ideal[..., 1:3]), axis=2, dtype=float))
    ab_result = np.sqrt(np.sum(np.square(lab_result[..., 1:3]), axis=2, dtype=float))

    profile_ideal = np.mean(ab_ideal, axis=0, dtype=float)
    profile_result = np.mean(ab_result, axis=0, dtype=float)
    center = int(profile_ideal.size // 2)
    profile_ideal = profile_ideal[center:]
    profile_result = profile_result[center:]

    mo_params = scene.fields.get("mo_target_params", {})
    frequency = float(mo_params.get("f", 1.0 / max(float(scene_get(scene, "cols")), 1.0) / 10.0))
    pixels_per_degree = float(np.asarray(xyz_ideal.shape[:2], dtype=float)[1]) / max(float(fov_scene), 1.0e-12)
    radii = np.arange(profile_ideal.size, dtype=float)
    cpd_mean = frequency * radii * pixels_per_degree

    return CameraMoireResult(
        data_abIdeal_mean=np.asarray(profile_ideal, dtype=float),
        data_abim_mean=np.asarray(profile_result, dtype=float),
        cpd_mean=np.asarray(cpd_mean, dtype=float),
    )


def camera_vsnr_sl(
    camera: Camera,
    light_levels: Any | None = None,
    exposure_time: float = 0.01,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraVSNRResult:
    """Legacy wrapper name for camera_vsnr()."""

    return camera_vsnr(
        camera,
        light_levels,
        exposure_time=exposure_time,
        asset_store=asset_store,
        session=session,
    )


def _matlab_round_scalar(value: float) -> int:
    value = float(value)
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def _whole_chart_corner_points(rows: int, cols: int) -> np.ndarray:
    return np.array(
        [
            [1.0, float(rows)],
            [float(cols), float(rows)],
            [float(cols), 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )


def _chart_rects_data(
    obj: ImageProcessor | Sensor,
    m_locs: Any,
    delta: Any,
    *,
    full_data: bool = False,
    data_type: str = "result",
) -> np.ndarray | list[np.ndarray]:
    from .roi import vc_get_roi_data

    locs = np.asarray(m_locs, dtype=float)
    if locs.ndim != 2 or locs.shape[0] != 2:
        raise ValueError("Chart midpoint locations must be a 2xN array in [row; col] order.")

    patch_data: list[np.ndarray] = []
    for index in range(locs.shape[1]):
        roi_locs, _ = _chart_roi(locs[:, index], delta)
        patch_data.append(np.asarray(vc_get_roi_data(obj, roi_locs, data_type), dtype=float))

    if full_data:
        return patch_data

    n_channels = int(patch_data[0].shape[1]) if patch_data else 0
    mean_values = np.zeros((locs.shape[1], n_channels), dtype=float)
    for index, data in enumerate(patch_data):
        mean_values[index, :] = np.mean(np.asarray(data, dtype=float), axis=0, dtype=float)
    return mean_values


def _display_white_xyz(ip: ImageProcessor, *, asset_store: AssetStore) -> np.ndarray:
    display = ip.fields["display"]
    wave = np.asarray(display_get(display, "wave"), dtype=float).reshape(-1)
    white_spd = np.asarray(display_get(display, "white spd"), dtype=float).reshape(-1)
    white_xyz = np.asarray(xyz_from_energy(white_spd, wave, asset_store=asset_store), dtype=float).reshape(-1)
    return 100.0 * (white_xyz / max(float(white_xyz[1]), 1.0e-12))


def _middle_matrix(m: Any, sz: Any) -> np.ndarray:
    data = np.asarray(m, dtype=float)
    target = np.asarray(sz, dtype=float).reshape(-1)
    if target.size == 1:
        target = np.repeat(target, 2)
    half = np.asarray([_matlab_round_scalar(float(value) / 2.0) for value in target[:2]], dtype=int)
    center = np.asarray([_matlab_round_scalar(float(value) / 2.0) for value in data.shape[:2]], dtype=int)

    row_min = max(1, int(center[0] - half[0]))
    row_max = min(int(data.shape[0]), int(center[0] + half[0]))
    col_min = max(1, int(center[1] - half[1]))
    col_max = min(int(data.shape[1]), int(center[1] + half[1]))
    return np.asarray(data[row_min - 1 : row_max, col_min - 1 : col_max, ...], dtype=float)


def _center_crop_to_shape(image: Any, shape: Any) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    target = np.asarray(shape, dtype=int).reshape(-1)
    rows = min(int(target[0]), int(data.shape[0]))
    cols = min(int(target[1]), int(data.shape[1]))
    row_start = max((int(data.shape[0]) - rows) // 2, 0)
    col_start = max((int(data.shape[1]) - cols) // 2, 0)
    return np.asarray(data[row_start : row_start + rows, col_start : col_start + cols, ...], dtype=float)


def _default_vsnr_rect(ip: ImageProcessor) -> np.ndarray:
    size = np.asarray(ip_get(ip, "size"), dtype=int).reshape(-1)
    border = np.asarray([_matlab_round_scalar(0.1 * float(value)) for value in size[:2]], dtype=int)
    return np.array(
        [
            int(border[1]),
            int(border[0]),
            int(size[1] - (2 * border[1])),
            int(size[0] - (2 * border[0])),
        ],
        dtype=int,
    )


def _ip_vsnr(
    ip: ImageProcessor,
    rect: Any,
    *,
    asset_store: AssetStore,
) -> float:
    rect_array = np.asarray(rect, dtype=int).reshape(-1)
    roi_xyz = np.asarray(ip_get(ip, "roixyz", rect_array), dtype=float)
    rows = int(rect_array[3]) + 1
    cols = int(rect_array[2]) + 1
    roi_xyz = xw_to_rgb_format(roi_xyz, rows, cols)

    white_xyz = _display_white_xyz(ip, asset_store=asset_store)
    lab = np.asarray(xyz_to_lab(roi_xyz, white_xyz), dtype=float)
    lab = _middle_matrix(lab, 0.8 * np.asarray(lab.shape[:2], dtype=float))
    channels = lab.reshape(-1, 3)
    variance = np.var(channels, axis=0, ddof=1, dtype=float)
    return 1.0 / max(float(np.sqrt(np.sum(variance, dtype=float))), 1.0e-12)


def _linear_srgb_to_xyz(rgb: Any) -> np.ndarray:
    rgb_array = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    if rgb_array.ndim == 2:
        xw = rgb_array.reshape(-1, 3)
        rows = cols = None
    elif rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
        xw, rows, cols, _ = rgb_to_xw_format(rgb_array)
    else:
        raise ValueError("RGB data must be XW Nx3 or RGB-format rows x cols x 3.")

    transform = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    )
    xyz = np.asarray(xw @ transform.T, dtype=float)
    if rows is None or cols is None:
        return xyz
    return xw_to_rgb_format(xyz, rows, cols)


def _rgb_to_gray(rgb: Any) -> np.ndarray:
    rgb_array = np.asarray(rgb, dtype=float)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("RGB data must be rows x cols x 3.")
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=float)
    return np.tensordot(rgb_array, weights, axes=([2], [0]))


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=float) - ((size - 1) / 2.0)
    kernel_1d = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d /= np.sum(kernel_1d, dtype=float)
    return np.outer(kernel_1d, kernel_1d)


def _ssim_index(gray1: Any, gray2: Any, *, window_size: int = 11, sigma: float = 1.5) -> float:
    first = np.asarray(gray1, dtype=float)
    second = np.asarray(gray2, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("SSIM expects same-shaped 2D images.")

    min_dim = max(1, min(first.shape))
    size = min(int(window_size), min_dim)
    if size % 2 == 0:
        size = max(1, size - 1)
    window = _gaussian_window(size, sigma if size > 1 else 1.0)

    mu1 = convolve2d(first, window, mode="same", boundary="symm")
    mu2 = convolve2d(second, window, mode="same", boundary="symm")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(first * first, window, mode="same", boundary="symm") - mu1_sq
    sigma2_sq = convolve2d(second * second, window, mode="same", boundary="symm") - mu2_sq
    sigma12 = convolve2d(first * second, window, mode="same", boundary="symm") - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    l_value = 255.0
    c1 = (0.01 * l_value) ** 2
    c2 = (0.03 * l_value) ** 2
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / np.maximum(denominator, np.finfo(float).eps)
    return float(np.mean(ssim_map, dtype=float))


def _macbeth_ideal_xyz(
    wave_nm: Any,
    illuminant_name: str = "D65",
    *,
    asset_store: AssetStore,
) -> np.ndarray:
    from .metrics import xyz_from_energy

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    _, reflectances = asset_store.load_reflectances("macbethChart.mat", wave_nm=wave)
    _, illuminant = asset_store.load_illuminant(illuminant_name, wave_nm=wave)
    color_signal = np.asarray(reflectances, dtype=float) * np.asarray(illuminant, dtype=float).reshape(-1, 1)
    macbeth_xyz = np.asarray(xyz_from_energy(color_signal.T, wave, asset_store=asset_store), dtype=float)
    return 100.0 * (macbeth_xyz / max(float(np.max(macbeth_xyz[:, 1])), 1.0e-12))


def _macbeth_color_error(
    ip: ImageProcessor,
    illuminant_name: str = "D65",
    corner_points: Any | None = None,
    *,
    asset_store: AssetStore,
) -> dict[str, Any]:
    if corner_points is None:
        corner_points = ip_get(ip, "chart corner points")
    if corner_points is None or np.asarray(corner_points).size == 0:
        rows, cols = np.asarray(ip_get(ip, "size"), dtype=int)[:2]
        corner_points = _whole_chart_corner_points(int(rows), int(cols))

    corner_points_array = np.asarray(corner_points, dtype=float).reshape(4, 2)
    rects, m_locs, p_size = _chart_rectangles(corner_points_array, 4, 6, 0.3)
    rgb_data = np.asarray(_chart_rects_data(ip, m_locs, float(p_size[0]), full_data=False, data_type="result"), dtype=float)
    macbeth_xyz = np.asarray(_linear_srgb_to_xyz(xw_to_rgb_format(rgb_data, 4, 6)), dtype=float)
    macbeth_xyz_xw, _, _, _ = rgb_to_xw_format(macbeth_xyz)

    ideal_xyz = _macbeth_ideal_xyz(np.asarray(ip_get(ip, "wave"), dtype=float), illuminant_name, asset_store=asset_store)
    white_xyz = np.asarray(macbeth_xyz_xw[3, :], dtype=float).reshape(-1)
    white_ideal_xyz = np.asarray(ideal_xyz[3, :], dtype=float).reshape(-1)
    scale = float(white_ideal_xyz[1]) / max(float(white_xyz[1]), 1.0e-12)
    macbeth_xyz_xw = macbeth_xyz_xw * scale
    macbeth_lab = np.asarray(xyz_to_lab(macbeth_xyz_xw, white_ideal_xyz), dtype=float)
    delta_e = np.asarray(delta_e_ab(macbeth_xyz_xw, ideal_xyz, white_ideal_xyz), dtype=float).reshape(-1)

    return {
        "macbethLAB": macbeth_lab,
        "macbethXYZ": macbeth_xyz_xw,
        "deltaE": delta_e,
        "idealXYZ": np.asarray(ideal_xyz, dtype=float),
        "whiteXYZ": np.asarray(macbeth_xyz_xw[3, :], dtype=float),
        "idealWhiteXYZ": white_ideal_xyz,
        "cornerPoints": corner_points_array,
        "rects": np.asarray(rects, dtype=int),
        "mLocs": np.asarray(m_locs, dtype=int),
        "pSize": np.asarray(p_size, dtype=int),
        "vci": ip,
    }


def macbeth_color_error(
    ip: ImageProcessor,
    illuminant_name: str = "D65",
    corner_points: Any | None = None,
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ImageProcessor]:
    """Headless MATLAB-style `macbethColorError` wrapper."""

    result = _macbeth_color_error(
        ip,
        illuminant_name,
        corner_points,
        asset_store=_store(asset_store),
    )
    return (
        np.asarray(result["macbethLAB"], dtype=float),
        np.asarray(result["macbethXYZ"], dtype=float),
        np.asarray(result["deltaE"], dtype=float),
        result["vci"],
    )


def macbeth_compare_ideal(
    ip: ImageProcessor,
    m_rgb: Any | None = None,
    illuminant_name: str = "d65",
    *,
    asset_store: AssetStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the headless Macbeth comparison image used by the MATLAB script."""

    store = _store(asset_store)
    current_ip = ip

    patch_size: np.ndarray
    if m_rgb is None or np.asarray(m_rgb).size == 0:
        corner_points = ip_get(current_ip, "chart corner points")
        if corner_points is None or np.asarray(corner_points).size == 0:
            rows, cols = np.asarray(ip_get(current_ip, "size"), dtype=int)[:2]
            corner_points = _whole_chart_corner_points(int(rows), int(cols))
            current_ip = ip_set(current_ip, "chart corner points", corner_points)
        rects, m_locs, patch_size = _chart_rectangles(corner_points, 4, 6, 0.5)
        current_ip = ip_set(current_ip, "chart rectangles", rects)
        m_rgb = _chart_rects_data(current_ip, m_locs, 0.6 * float(patch_size[0]), full_data=False, data_type="result")
    else:
        size = np.asarray(ip_get(current_ip, "size"), dtype=int)
        patch_scalar = _matlab_round_scalar((float(size[0]) / 4.0) * 0.6)
        patch_size = np.array([patch_scalar, patch_scalar], dtype=int)

    patch_rgb = np.asarray(m_rgb, dtype=float)
    if patch_rgb.ndim == 2:
        patch_rgb = xw_to_rgb_format(patch_rgb, 4, 6)
    patch_rgb = patch_rgb / max(float(np.max(patch_rgb)), 1.0e-12)

    ideal_patch = xw_to_rgb_format(
        _macbeth_ideal_linear_rgb(np.asarray(ip_get(current_ip, "wave"), dtype=float), asset_store=store),
        4,
        6,
    )
    ideal_patch = ideal_patch / max(float(np.max(ideal_patch)), 1.0e-12)
    full_ideal_rgb = image_increase_image_rgb_size(ideal_patch, patch_size)
    embedded_rgb = np.asarray(full_ideal_rgb, dtype=float).copy()

    patch_extent = int(np.asarray(patch_size, dtype=int).reshape(-1)[0])
    window = patch_extent + np.array(
        [_matlab_round_scalar(value) for value in np.arange(-patch_extent / 3.0, 1.0, 1.0)],
        dtype=int,
    )
    for row_index in range(4):
        rows = (row_index * patch_extent) + window - 1
        for col_index in range(6):
            cols = (col_index * patch_extent) + window - 1
            embedded_rgb[np.ix_(rows, cols, np.arange(3))] = patch_rgb[row_index, col_index, :]

    return (
        linear_to_srgb(embedded_rgb),
        linear_to_srgb(patch_rgb),
        np.asarray(patch_size, dtype=int).reshape(-1),
    )


def camera_color_accuracy(
    camera: Camera,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> tuple[dict[str, Any], Camera]:
    """Run the headless Macbeth color-accuracy workflow from `s_metricsColorAccuracy.m`."""

    store = _store(asset_store)

    scene_distance_m = 1000.0
    oi = camera_get(camera, "oi")
    sensor = camera_get(camera, "sensor")
    scene_fov = float(sensor_get(sensor, "fov", scene_distance_m, oi))

    macbeth_scene = scene_create("macbeth d65", asset_store=store, session=session)
    macbeth_scene = scene_adjust_luminance(macbeth_scene, 100.0, asset_store=store)
    macbeth_scene = scene_set(macbeth_scene, "distance", scene_distance_m)
    macbeth_scene = scene_set(macbeth_scene, "fov", scene_fov)
    camera = camera_compute(camera, macbeth_scene, asset_store=store, session=session)

    ip = camera_get(camera, "ip")
    rows, cols = np.asarray(ip_get(ip, "size"), dtype=int)[:2]
    corner_points = _whole_chart_corner_points(int(rows), int(cols))
    ip = ip_set(ip, "chart corner points", corner_points, session=session)
    color_accuracy = _macbeth_color_error(ip, "D65", corner_points, asset_store=store)

    camera = camera_set(camera, "ip", ip, session=session)
    camera = camera_set(camera, "ip chart corner points", corner_points, session=session)
    color_accuracy["vci"] = camera_get(camera, "ip")
    return color_accuracy, camera


def camera_mtf(
    camera: Camera,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraMTFResult:
    """Compute the ISO 12233 camera MTF using the upstream slanted-edge workflow."""

    store = _store(asset_store)

    scene = scene_create("slanted bar", 256, asset_store=store, session=session)
    scene = scene_adjust_luminance(scene, 100.0, asset_store=store)
    scene = scene_set(scene, "fov", 5.0)

    oi = camera_get(camera, "oi")
    sensor = camera_get(camera, "sensor")
    sensor = sensor_set(sensor, "fov", 5.0, oi)
    camera = camera_set(camera, "sensor", sensor, session=session)
    camera = camera_compute(camera, scene, asset_store=store, session=session)

    ip = camera_get(camera, "ip")
    result = np.clip(np.asarray(ip_get(ip, "result"), dtype=float), 0.0, None)
    ip = ip_set(ip, "result", result, session=session)
    camera = camera_set(camera, "ip", ip, session=session)

    rect = np.asarray(iso_find_slanted_bar(ip), dtype=int).reshape(-1)
    if rect.size != 4:
        raise ValueError("ISOFindSlantedBar did not return a [col, row, width, height] rect.")
    col_min, row_min, width, height = rect
    bar_image = np.asarray(result[row_min - 1 : row_min + height, col_min - 1 : col_min + width, :], dtype=float)
    delta_x = float(camera_get(camera, "pixel width", "mm"))
    mtf = iso12233(bar_image, delta_x=delta_x, plot_options="none")

    return CameraMTFResult(
        freq=np.asarray(mtf.freq, dtype=float),
        mtf=np.asarray(mtf.mtf, dtype=float),
        nyquistf=float(mtf.nyquistf),
        lsf=np.asarray(mtf.lsf, dtype=float),
        lsfx=np.asarray(mtf.lsfx, dtype=float),
        mtf50=float(mtf.mtf50),
        aliasingPercentage=float(mtf.aliasingPercentage),
        rect=np.asarray(rect, dtype=int),
        vci=ip,
        esf=None if mtf.esf is None else np.asarray(mtf.esf, dtype=float),
        fitme=None if mtf.fitme is None else np.asarray(mtf.fitme, dtype=float),
        win=mtf.win,
    )


def camera_acutance(
    camera: Camera,
    plot_flag: bool = True,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> float:
    """Compute ISO acutance from the camera luminance MTF."""

    del plot_flag

    cmtf = camera_mtf(camera, asset_store=asset_store, session=session)
    mtf = np.asarray(cmtf.mtf, dtype=float)
    luminance_mtf = mtf[:, 3] if mtf.ndim == 2 and mtf.shape[1] >= 4 else mtf.reshape(-1)
    oi = camera_get(camera, "oi")
    deg_per_mm = float(camera_get(camera, "sensor h deg per distance", "mm", None, oi))
    cpd = np.asarray(cmtf.freq, dtype=float) / max(deg_per_mm, 1.0e-12)
    return iso_acutance(cpd, luminance_mtf)


def camera_vsnr(
    camera: Camera,
    light_levels: Any | None = None,
    exposure_time: float = 0.01,
    *,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> CameraVSNRResult:
    """Run the headless VSNR workflow from `s_metricsVSNR.m`."""

    store = _store(asset_store)
    light_levels_is_omitted = light_levels is None
    if isinstance(light_levels, str) and not light_levels.strip():
        light_levels_is_omitted = True
    elif isinstance(light_levels, (list, tuple, np.ndarray)):
        light_levels_array = np.asarray(light_levels)
        light_levels_is_omitted = light_levels_array.size == 0
    levels = (
        np.asarray([1.0, 10.0, 100.0], dtype=float)
        if light_levels_is_omitted
        else np.asarray(light_levels, dtype=float).reshape(-1)
    )
    exposure_time_is_omitted = exposure_time is None
    if isinstance(exposure_time, str) and not exposure_time.strip():
        exposure_time_is_omitted = True
    elif isinstance(exposure_time, (list, tuple, np.ndarray)):
        exposure_time_array = np.asarray(exposure_time)
        exposure_time_is_omitted = exposure_time_array.size == 0
    exposure_time_value = 0.01 if exposure_time_is_omitted else float(exposure_time)
    vsnr = np.full(levels.shape, np.nan, dtype=float)
    e_time = np.zeros(levels.shape, dtype=float)
    rect: np.ndarray | None = None
    ip_results: list[ImageProcessor] = []

    for index, level in enumerate(levels):
        scene = scene_create("uniform d65", asset_store=store, session=session)
        scene = scene_set(scene, "fov", 5.0)
        scene = scene_adjust_luminance(scene, float(level), asset_store=store)

        working_camera = camera_set(camera.clone(), "sensor exp time", exposure_time_value, session=session)
        working_camera = camera_compute(working_camera, scene, asset_store=store, session=session)

        ip = camera_get(working_camera, "ip")
        result = np.asarray(ip_get(ip, "result"), dtype=float)
        result_max = float(np.max(result)) if result.size else 0.0
        sensor_max = float(ip_get(ip, "sensormax"))

        if (sensor_max - result_max) < (10.0 * np.finfo(float).eps):
            ip_results.append(ip)
            continue

        scaled_ip = ip_set(ip.clone(), "result", result * (sensor_max / max(result_max, 1.0e-12)), session=session)
        if rect is None:
            rect = _default_vsnr_rect(scaled_ip)
        vsnr[index] = _ip_vsnr(scaled_ip, rect, asset_store=store)
        ip_results.append(scaled_ip)

    if rect is None:
        rect = np.zeros(4, dtype=int)

    return CameraVSNRResult(
        lightLevels=levels,
        vSNR=vsnr,
        eTime=e_time,
        rect=np.asarray(rect, dtype=int),
        ip=ip_results,
        oi=camera_get(camera, "oi"),
        sensor=camera_get(camera, "sensor"),
    )


cameraColorAccuracy = camera_color_accuracy
cameraMoire = camera_moire
cameraFullReference = camera_full_reference
cameraVSNR_SL = camera_vsnr_sl
cameraVSNR = camera_vsnr
cameraCreate = camera_create
cameraCompute = camera_compute
cameraComputeSequence = camera_compute_sequence
cameraComputesrgb = camera_compute_srgb
cameraClearData = camera_clear_data
cameraMTF = camera_mtf
cameraAcutance = camera_acutance
cameraGet = camera_get
cameraSet = camera_set
macbethColorError = macbeth_color_error
macbethCompareIdeal = macbeth_compare_ideal
