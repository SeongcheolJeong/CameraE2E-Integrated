"""Optional vcSESSION-style object registry helpers."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from .types import BaseISETObject, Camera, Display, ImageProcessor, OpticalImage, Scene, Sensor, SessionContext
from .utils import param_format

_T = TypeVar("_T", bound=BaseISETObject)

_SESSION_TYPE_ALIASES = {
    "scene": "scene",
    "scenes": "scene",
    "oi": "oi",
    "opticalimage": "oi",
    "opticalimages": "oi",
    "optics": "oi",
    "sensor": "sensor",
    "sensors": "sensor",
    "isa": "sensor",
    "pixel": "sensor",
    "display": "display",
    "displays": "display",
    "ip": "ip",
    "imgproc": "ip",
    "ipdisplay": "ip",
    "vcimage": "ip",
    "vci": "ip",
    "imageprocessor": "ip",
    "camera": "camera",
    "cameras": "camera",
}

_IE_NESTED_OBJECT_TYPES = {
    "optics": ("oi", "optics"),
    "pixel": ("sensor", "pixel"),
    "ipdisplay": ("ip", "display"),
}

_VC_EQUIVALENT_OBJTYPE_MAP = {
    "scene": "SCENE",
    "oi": "OPTICALIMAGE",
    "opticalimage": "OPTICALIMAGE",
    "sensor": "ISA",
    "isa": "ISA",
    "imgproc": "VCIMAGE",
    "ip": "VCIMAGE",
    "vci": "VCIMAGE",
    "vcimage": "VCIMAGE",
    "ipdisplay": "VCIMAGE",
    "display": "DISPLAY",
    "camera": "CAMERA",
}


def _session_type_name(value: str) -> str:
    normalized = param_format(value)
    if normalized not in _SESSION_TYPE_ALIASES:
        raise KeyError(f"Unsupported session object type: {value}")
    return _SESSION_TYPE_ALIASES[normalized]


def _ie_object_type(value: str | BaseISETObject | dict[str, Any]) -> str:
    if isinstance(value, BaseISETObject):
        return param_format(value.type)
    if isinstance(value, dict) and "type" in value:
        return param_format(str(value["type"]))
    return param_format(str(value))


def vc_equivalent_objtype(this_obj: str | BaseISETObject | dict[str, Any] | Any) -> str:
    if isinstance(this_obj, BaseISETObject):
        key = param_format(this_obj.type)
        return _VC_EQUIVALENT_OBJTYPE_MAP.get(key, key.upper())
    if isinstance(this_obj, dict) and "type" in this_obj:
        key = param_format(str(this_obj["type"]))
        return _VC_EQUIVALENT_OBJTYPE_MAP.get(key, key.upper())
    if isinstance(this_obj, str):
        key = param_format(this_obj)
        return _VC_EQUIVALENT_OBJTYPE_MAP.get(key, key.upper())
    return type(this_obj).__name__


def ie_equivalent_objtype(obj_type: str) -> str:
    return vc_equivalent_objtype(obj_type)


def _object_session_type(obj: BaseISETObject) -> str:
    if isinstance(obj, Scene):
        return "scene"
    if isinstance(obj, OpticalImage):
        return "oi"
    if isinstance(obj, Sensor):
        return "sensor"
    if isinstance(obj, Display):
        return "display"
    if isinstance(obj, ImageProcessor):
        return "ip"
    if isinstance(obj, Camera):
        return "camera"
    return _session_type_name(obj.type)


def session_create(name: str = "vcSESSION") -> SessionContext:
    return SessionContext(name=name, directory=str(Path.cwd()))


def iset_root_path() -> str:
    return str(Path(__file__).resolve().parents[2])


def iset_path(
    iset_dir: str | Path | None = None,
    *,
    apply: bool = False,
) -> list[str]:
    root = Path(iset_root_path() if iset_dir is None else iset_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ISET root directory does not exist: {root}")

    directories = [root]
    directories.extend(path for path in sorted(root.rglob("*")) if path.is_dir())
    path_list = [
        str(path)
        for path in directories
        if not any(part in {".git", ".svn"} for part in path.parts)
    ]

    if apply:
        for directory in reversed(path_list):
            if directory not in sys.path:
                sys.path.insert(0, directory)

    return path_list


def ie_init_session(
    *,
    name: str | None = None,
    directory: str | Path | None = None,
) -> SessionContext:
    session_name = name or f"iset-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    session_dir = str(Path.cwd() if directory is None else directory)
    return SessionContext(name=session_name, directory=session_dir)


def session_object_id(obj: BaseISETObject) -> int | None:
    session_id = obj.metadata.get("session_id")
    return int(session_id) if session_id is not None else None


def session_add_object(
    session: SessionContext,
    obj: _T,
    *,
    select: bool = True,
    object_id: int | None = None,
) -> int:
    object_type = _object_session_type(obj)
    bucket = session.objects.setdefault(object_type, {})
    session.next_ids.setdefault(object_type, 1)

    if object_id is None:
        existing_type = obj.metadata.get("session_type")
        existing_id = obj.metadata.get("session_id")
        if existing_type == object_type and existing_id in bucket:
            object_id = int(existing_id)
        else:
            object_id = int(session.next_ids[object_type])
            session.next_ids[object_type] = object_id + 1
    else:
        object_id = int(object_id)
        session.next_ids[object_type] = max(int(session.next_ids[object_type]), object_id + 1)

    bucket[object_id] = obj
    obj.metadata["session_type"] = object_type
    obj.metadata["session_id"] = object_id
    if select:
        session.selected[object_type] = object_id
    return object_id


def track_session_object(session: SessionContext | None, obj: _T, *, select: bool = True) -> _T:
    if session is not None:
        session_add_object(session, obj, select=select)
    return obj


def track_ip_session_state(
    session: SessionContext | None,
    ip: ImageProcessor,
    *,
    select: bool = True,
) -> ImageProcessor:
    if session is None:
        return ip
    display = ip.fields.get("display")
    if isinstance(display, Display):
        ip.fields["display"] = track_session_object(session, display, select=select)
    return track_session_object(session, ip, select=select)


def track_camera_session_state(
    session: SessionContext | None,
    camera: Camera,
    *,
    select: bool = True,
) -> Camera:
    if session is None:
        return camera
    oi = camera.fields.get("oi")
    if isinstance(oi, OpticalImage):
        camera.fields["oi"] = track_session_object(session, oi, select=select)
    sensor = camera.fields.get("sensor")
    if isinstance(sensor, Sensor):
        camera.fields["sensor"] = track_session_object(session, sensor, select=select)
    ip = camera.fields.get("ip")
    if isinstance(ip, ImageProcessor):
        camera.fields["ip"] = track_ip_session_state(session, ip, select=select)
    return track_session_object(session, camera, select=select)


def _latest_iset_session_name(directory: str | Path | None = None) -> str:
    session_dir = Path.cwd() if directory is None else Path(directory)
    candidates = sorted(
        path.name for path in session_dir.glob("iset-*.mat") if path.is_file()
    )
    if candidates:
        return candidates[-1]
    return "isetSession.mat"


def session_get_object(
    session: SessionContext,
    object_type: str,
    object_id: int | None = None,
) -> BaseISETObject | None:
    normalized_type = _session_type_name(object_type)
    if object_id is None:
        object_id = session.selected.get(normalized_type)
    if object_id is None:
        return None
    return session.objects.get(normalized_type, {}).get(int(object_id))


def session_get_selected(session: SessionContext, object_type: str) -> BaseISETObject | None:
    return session_get_object(session, object_type)


def session_get_selected_pair(
    session: SessionContext,
    object_type: str,
) -> tuple[int | None, BaseISETObject | None]:
    object_id = session_get_selected_id(session, object_type)
    return object_id, session_get_object(session, object_type, object_id)


def session_get_object_with_id(
    session: SessionContext,
    object_type: str,
    object_id: int | None = None,
) -> tuple[BaseISETObject | None, int | None]:
    if object_id is None:
        object_id = session_get_selected_id(session, object_type)
    return session_get_object(session, object_type, object_id), object_id


def session_get_selected_id(session: SessionContext, object_type: str) -> int | None:
    normalized_type = _session_type_name(object_type)
    selected = session.selected.get(normalized_type)
    return int(selected) if selected is not None else None


def session_set_selected(
    session: SessionContext,
    object_type: str,
    value: int | BaseISETObject | None,
) -> None:
    normalized_type = _session_type_name(object_type)
    if value is None:
        session.selected[normalized_type] = None
        return
    if isinstance(value, BaseISETObject):
        session_add_object(session, value, select=True)
        return
    object_id = int(value)
    if object_id <= 0:
        session.selected[normalized_type] = None
        return
    if object_id not in session.objects.get(normalized_type, {}):
        raise KeyError(f"Session object id {object_id} is not registered for type {normalized_type}.")
    session.selected[normalized_type] = object_id


def session_list_objects(
    session: SessionContext,
    object_type: str | None = None,
) -> dict[int, BaseISETObject] | dict[str, dict[int, BaseISETObject]]:
    if object_type is None:
        return {key: dict(value) for key, value in session.objects.items()}
    normalized_type = _session_type_name(object_type)
    return dict(session.objects.get(normalized_type, {}))


def session_get_objects(session: SessionContext, object_type: str) -> list[BaseISETObject]:
    objects = session_list_objects(session, object_type)
    return [obj for _, obj in sorted(objects.items())]


def session_count_objects(session: SessionContext, object_type: str) -> int:
    return len(session_get_objects(session, object_type))


def session_get_object_names(
    session: SessionContext,
    object_type: str = "scene",
    make_unique: bool = False,
) -> list[str]:
    names = [str(obj.name) for obj in session_get_objects(session, object_type)]
    if make_unique:
        return [f"{index + 1}-{name}" for index, name in enumerate(names)]
    return names


def session_find_object_by_name(
    session: SessionContext,
    object_type: str,
    object_name: str,
) -> int | None:
    names = session_get_object_names(session, object_type)
    for index, name in enumerate(names, start=1):
        if name == object_name:
            return index
    return None


def session_set_objects(
    session: SessionContext,
    object_type: str,
    objects: dict[int, BaseISETObject] | list[BaseISETObject] | tuple[BaseISETObject, ...],
) -> list[BaseISETObject]:
    normalized_type = _session_type_name(object_type)
    if isinstance(objects, dict):
        ordered_objects = [obj for _, obj in sorted(objects.items())]
    else:
        ordered_objects = list(objects)

    new_bucket: dict[int, BaseISETObject] = {}
    for object_id, obj in enumerate(ordered_objects, start=1):
        if isinstance(obj, Camera):
            obj = track_camera_session_state(session, obj, select=False)
        elif isinstance(obj, ImageProcessor):
            obj = track_ip_session_state(session, obj, select=False)
        obj.metadata["session_type"] = normalized_type
        obj.metadata["session_id"] = object_id
        new_bucket[object_id] = obj

    session.objects[normalized_type] = new_bucket
    session.next_ids[normalized_type] = len(new_bucket) + 1
    selected_id = session.selected.get(normalized_type)
    if selected_id is not None and int(selected_id) not in new_bucket:
        session.selected[normalized_type] = None
    return ordered_objects


def session_add_and_select_object(
    session: SessionContext,
    object_type: str | BaseISETObject,
    obj: BaseISETObject | None = None,
) -> int:
    if isinstance(object_type, BaseISETObject):
        obj = object_type
        object_type = _object_session_type(obj)
    if obj is None:
        raise ValueError("An object must be provided.")

    if isinstance(obj, Camera):
        camera = track_camera_session_state(session, obj, select=True)
        object_id = session_object_id(camera)
        if object_id is None:
            raise RuntimeError("Failed to assign a session id to camera.")
        return object_id
    if isinstance(obj, ImageProcessor):
        ip = track_ip_session_state(session, obj, select=True)
        object_id = session_object_id(ip)
        if object_id is None:
            raise RuntimeError("Failed to assign a session id to image processor.")
        return object_id
    return session_add_object(session, obj, select=True)


def ie_add_object(session: SessionContext, obj: BaseISETObject) -> int | tuple[int, int, int]:
    if isinstance(obj, Camera):
        camera = track_camera_session_state(session, obj, select=True)
        oi_id = session_object_id(camera.fields["oi"])
        sensor_id = session_object_id(camera.fields["sensor"])
        ip_id = session_object_id(camera.fields["ip"])
        if oi_id is None or sensor_id is None or ip_id is None:
            raise RuntimeError("Failed to assign session ids to camera pipeline objects.")
        return oi_id, sensor_id, ip_id
    return session_add_and_select_object(session, obj)


def session_new_object_value(
    session: SessionContext,
    object_type: str,
) -> int | tuple[int, int, int]:
    normalized_type = _session_type_name(object_type)
    if normalized_type == "camera":
        return (
            session_count_objects(session, "oi") + 1,
            session_count_objects(session, "sensor") + 1,
            session_count_objects(session, "ip") + 1,
        )
    return session_count_objects(session, normalized_type) + 1


def session_get_object_type(obj: BaseISETObject) -> str:
    object_type = getattr(obj, "type", None)
    if object_type is None:
        raise ValueError("Object does not have a type field.")
    return str(object_type)


def session_new_object_name(session: SessionContext, object_type: str) -> str:
    normalized_type = _session_type_name(object_type)
    return f"{normalized_type}{session_count_objects(session, normalized_type) + 1}"


def _normalize_on_off(value: Any) -> Any:
    if isinstance(value, str):
        normalized = param_format(value)
        if normalized == "on":
            return True
        if normalized == "off":
            return False
    return value


_WINDOW_KEY_ALIASES = {
    "mainwindow": "main_window",
    "mainfigure": "main_window",
    "mainfigures": "main_window",
    "scenewindow": "scene_window",
    "scenefigure": "scene_window",
    "sceneimagefigure": "scene_window",
    "sceneimagefigures": "scene_window",
    "oiwindow": "oi_window",
    "oifigure": "oi_window",
    "oifigures": "oi_window",
    "opticalimagefigure": "oi_window",
    "opticalimagefigures": "oi_window",
    "sensorwindow": "sensor_window",
    "sensorfigure": "sensor_window",
    "sensorfigures": "sensor_window",
    "isafigure": "sensor_window",
    "isafigures": "sensor_window",
    "isawindow": "sensor_window",
    "ipwindow": "ip_window",
    "ipfigure": "ip_window",
    "vcimagewindow": "ip_window",
    "vcimagefigure": "ip_window",
    "vcimagefigures": "ip_window",
    "displaywindow": "display_window",
    "metricswindow": "metrics_window",
    "metricsfigure": "metrics_window",
    "metricsfigures": "metrics_window",
    "camdesignwindow": "camdesign_window",
    "imageexplorewindow": "imageexplore_window",
}

_APP_AXIS_FIELDS = {
    "scene": ("scene_window", "sceneImage"),
    "oi": ("oi_window", "oiImage"),
    "opticalimage": ("oi_window", "oiImage"),
    "sensor": ("sensor_window", "imgMain"),
    "isa": ("sensor_window", "imgMain"),
    "ip": ("ip_window", "ipImage"),
    "vcimage": ("ip_window", "ipImage"),
    "display": ("display_window", "displayImage"),
}

_WINDOW_STATE_PARAM_MAP = {
    "scenewindowhandle": ("scene_window", "figure"),
    "scenewindow": ("scene_window", "app"),
    "scenewindowhandle": ("scene_window", "figure"),
    "sceneimagehandle": ("scene_window", "axis"),
    "oiwindowhandles": ("oi_window", "handles"),
    "oiguidata": ("oi_window", "handles"),
    "sensorwindowhandles": ("sensor_window", "handles"),
    "sensorguidata": ("sensor_window", "handles"),
    "sensorimagehandle": ("sensor_window", "axis"),
    "vcimagehandles": ("ip_window", "handles"),
    "metricshandles": ("metrics_window", "handles"),
    "oicomputelist": ("custom", "oi_compute_list"),
    "sensorgamma": ("render_state", "sensor_gamma"),
    "scenegamma": ("render_state", "scene_gamma"),
    "oigamma": ("render_state", "oi_gamma"),
    "ipgamma": ("render_state", "ip_gamma"),
    "scenedisplayflag": ("render_state", "scene_display_flag"),
    "oidisplayflag": ("render_state", "oi_display_flag"),
}

_WINDOW_POSITION_SLOTS = (
    ("main_window", "figure1"),
    ("scene_window", "figure1"),
    ("oi_window", "figure1"),
    ("sensor_window", "figure1"),
    ("ip_window", "figure1"),
    ("camdesign_window", "figure1"),
    ("imageexplore_window", "UIFigure"),
)


def _session_window_key(parameter: str) -> str | None:
    return _WINDOW_KEY_ALIASES.get(param_format(parameter))


def _get_nested_value(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _set_nested_value(obj: Any, attr: str, value: Any) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[attr] = value
        return
    setattr(obj, attr, value)


def _window_container(session: SessionContext, window_key: str, container_attr: str) -> Any:
    state = session.gui.get(window_key)
    if state is None:
        return None
    if isinstance(state, dict):
        app = state.get("app")
        container = _get_nested_value(app, container_attr)
        if container is not None:
            return container
        figure = state.get("hObject")
        nested = _get_nested_value(figure, container_attr)
        if nested is not None:
            return nested
        return figure
    container = _get_nested_value(state, container_attr)
    if container is not None:
        return container
    return state


def _window_slot_defaults() -> list[Any]:
    return [None] * len(_WINDOW_POSITION_SLOTS)


def _normalize_window_slots(values: Any) -> list[Any]:
    normalized = list(values)
    slot_count = len(_WINDOW_POSITION_SLOTS)
    if len(normalized) < slot_count:
        normalized.extend([None] * (slot_count - len(normalized)))
    return normalized[:slot_count]


def _extract_app_axis(app: Any, axis_name: str) -> Any:
    if app is None:
        return None
    if isinstance(app, dict):
        if axis_name in app:
            return app[axis_name]
        return app.get("current_axes")
    if hasattr(app, axis_name):
        return getattr(app, axis_name)
    if hasattr(app, "current_axes"):
        return getattr(app, "current_axes")
    if hasattr(app, "CurrentAxes"):
        return getattr(app, "CurrentAxes")
    return None


def _extract_app_figure(app: Any) -> Any:
    if app is None:
        return None
    if isinstance(app, dict):
        return app.get("figure1", app.get("hObject", app))
    if hasattr(app, "figure1"):
        return getattr(app, "figure1")
    if hasattr(app, "hObject"):
        return getattr(app, "hObject")
    return app


def _window_state_value(session: SessionContext, window_key: str, value_kind: str) -> Any:
    state = session.gui.get(window_key)
    if not isinstance(state, dict):
        return state
    if value_kind == "app":
        return state.get("app", state.get("hObject"))
    if value_kind == "figure":
        return state.get("hObject", _extract_app_figure(state.get("app")))
    if value_kind == "axis":
        app = state.get("app", state.get("hObject"))
        for obj_type, (candidate_window, axis_name) in _APP_AXIS_FIELDS.items():
            if candidate_window == window_key:
                del obj_type
                return _extract_app_axis(app, axis_name)
        return _extract_app_axis(app, "current_axes")
    if value_kind == "handles":
        return state.get("handles", state.get("app", state.get("hObject")))
    return state.get(value_kind)


def ie_session_get(session: SessionContext, parameter: str, *args: Any) -> Any:
    key = param_format(parameter)
    if key == "version":
        return session.version
    if key in {"name", "sessionname"}:
        return session.name
    if key in {"dir", "sessiondir"}:
        return session.directory
    if key in {"help", "inithelp"}:
        return bool(session.init_help)
    if key == "prefs":
        return dict(session.preferences)
    if key == "fontsize":
        return int(session.preferences.get("fontSize", 12))
    if key == "waitbar":
        return int(bool(session.gui.get("waitbar", session.preferences.get("waitbar", 0))))
    if key in {"windowpositions", "wpos"}:
        return _normalize_window_slots(session.preferences.get("wPos", _window_slot_defaults()))
    if key in {"windowstates", "wstate"}:
        return _normalize_window_slots(session.preferences.get("wState", _window_slot_defaults()))
    if key == "initclear":
        return bool(session.preferences.get("initclear", False))

    window_key = _session_window_key(parameter)
    if window_key is not None:
        return _window_state_value(session, window_key, "app")

    window_state_alias = _WINDOW_STATE_PARAM_MAP.get(key)
    if window_state_alias is not None:
        store_name, store_key = window_state_alias
        if store_name == "custom":
            return list(session.custom.get(store_key, []))
        if store_name == "render_state":
            return session.render_state.get(store_key)
        return _window_state_value(session, store_name, store_key)

    if key in {"graphwindow", "graphwinfigure"}:
        return session.graphwin.get("hObject")
    if key in {"graphwinstructure", "graphwinval"}:
        return dict(session.graphwin)
    if key == "graphwinhandle":
        return session.graphwin.get("handle")

    if key in {"scene", "oi", "opticalimage", "sensor", "isa", "vcimage", "ip", "display"}:
        return ie_get_object(session, key)
    if key == "selected":
        if not args:
            raise ValueError("ieSessionGet('selected', objType) requires an object type.")
        return ie_get_selected_object(session, str(args[0]))
    if key == "nobjects":
        if not args:
            raise ValueError("ieSessionGet('nobjects', objType) requires an object type.")
        return session_count_objects(session, str(args[0]))
    if key == "names":
        if not args:
            raise ValueError("ieSessionGet('names', objType) requires an object type.")
        return session_get_object_names(session, str(args[0]))
    if key in {"gpu", "gpucompute", "gpucomputing"}:
        return bool(session.gpu_compute)
    if key == "imagesizethreshold":
        return float(session.image_size_threshold)
    raise KeyError(f"Unknown ieSessionGet parameter: {parameter}")


def ie_session_set(session: SessionContext, parameter: str, value: Any, *args: Any) -> SessionContext:
    key = param_format(parameter)
    if key == "version":
        session.version = None if value is None else str(value)
        return session
    if key in {"name", "sessionname"}:
        session.name = str(value)
        return session
    if key in {"dir", "sessiondir"}:
        session.directory = str(value)
        return session
    if key in {"help", "inithelp"}:
        session.init_help = bool(_normalize_on_off(value))
        return session
    if key == "fontsize":
        session.preferences["fontSize"] = int(value)
        return session
    if key == "waitbar":
        normalized = _normalize_on_off(value)
        waitbar = int(bool(normalized))
        session.preferences["waitbar"] = waitbar
        session.gui["waitbar"] = waitbar
        return session
    if key in {"windowpositions", "wpos"}:
        session.preferences["wPos"] = _normalize_window_slots(value)
        return session
    if key in {"windowstates", "wstate"}:
        session.preferences["wState"] = _normalize_window_slots(value)
        return session
    if key == "initclear":
        session.preferences["initclear"] = bool(_normalize_on_off(value))
        return session

    window_key = _session_window_key(parameter)
    if window_key is not None:
        existing = session.gui.get(window_key, {})
        state = dict(existing) if isinstance(existing, dict) else {}
        state["app"] = value
        if value is None:
            state["hObject"] = None
        elif len(args) >= 2:
            state["hObject"] = value
        else:
            state["hObject"] = _extract_app_figure(value)
        if len(args) >= 1:
            state["eventdata"] = args[0]
        if len(args) >= 2:
            state["handles"] = args[1]
        elif "handles" not in state and isinstance(value, dict):
            state["handles"] = value.get("handles")
        session.gui[window_key] = state
        return session

    window_state_alias = _WINDOW_STATE_PARAM_MAP.get(key)
    if window_state_alias is not None:
        store_name, store_key = window_state_alias
        if store_name == "custom":
            session.custom[store_key] = list(value)
            return session
        if store_name == "render_state":
            session.render_state[store_key] = value
            return session
        state = dict(session.gui.get(store_name, {})) if isinstance(session.gui.get(store_name), dict) else {}
        state[store_key] = value
        session.gui[store_name] = state
        return session

    if key in {"graphwindow", "graphwinfigure"}:
        session.graphwin["hObject"] = value
        return session
    if key in {"graphwinstructure", "graphwinval"}:
        session.graphwin = dict(value)
        return session
    if key == "graphwinhandle":
        session.graphwin["handle"] = value
        return session

    if key in {"scene", "oi", "opticalimage", "sensor", "isa", "vcimage", "ip", "display"}:
        session_set_selected(session, key, value)
        return session
    if key in {"gpu", "gpucompute", "gpucomputing"}:
        session.gpu_compute = bool(_normalize_on_off(value))
        return session
    if key == "imagesizethreshold":
        session.image_size_threshold = float(value)
        return session
    raise KeyError(f"Unknown ieSessionSet parameter: {parameter}")


def ie_app_get(
    session: SessionContext,
    obj: str | BaseISETObject | dict[str, Any] | Any,
    *,
    select: bool = True,
) -> tuple[Any, Any]:
    del select
    if isinstance(obj, (str, BaseISETObject)) or (isinstance(obj, dict) and "type" in obj):
        obj_type = _ie_object_type(obj)
        window_info = _APP_AXIS_FIELDS.get(obj_type)
        if window_info is None:
            raise ValueError(f"Unknown object type for ieAppGet: {obj_type}")
        window_key, axis_name = window_info
        app_state = session.gui.get(window_key)
        app = app_state.get("app") if isinstance(app_state, dict) else app_state
        if app is None:
            raise ValueError(f"Undefined {obj_type} app.")
        return app, _extract_app_axis(app, axis_name)

    if isinstance(obj, dict):
        return obj, obj.get("current_axes")

    return obj, _extract_app_axis(obj, "current_axes")


def vc_get_figure(
    session: SessionContext,
    obj: BaseISETObject,
    select: bool = True,
) -> tuple[Any, Any]:
    return ie_app_get(session, obj, select=select)


def vc_set_figure_handles(
    session: SessionContext,
    fig_type: str,
    app: Any,
    eventdata: Any | None = None,
    handles: Any | None = None,
) -> SessionContext:
    key = param_format(fig_type)
    if key == "main":
        return ie_session_set(session, "mainwindow", app, eventdata, handles)
    if key == "scene":
        return ie_session_set(session, "scenewindow", app)
    if key in {"oi", "opticalimage"}:
        return ie_session_set(session, "oiwindow", app, eventdata, handles)
    if key in {"isa", "sensor"}:
        return ie_session_set(session, "sensorwindow", app, eventdata, handles)
    if key in {"vcimage", "ip", "imgproc"}:
        return ie_session_set(session, "vcimagewindow", app, eventdata, handles)
    if key == "metrics":
        return ie_session_set(session, "metricswindow", app, eventdata, handles)
    if key == "camdesign":
        return ie_session_set(session, "camdesignwindow", app)
    if key in {"imageexplorer", "imageexplore"}:
        return ie_session_set(session, "imageexplorewindow", app)
    raise ValueError(f"Unknown figure type: {fig_type}")


def _window_figure_from_session(session: SessionContext, object_type: str) -> Any:
    normalized = _session_type_name(object_type)
    if normalized == "scene":
        return _window_state_value(session, "scene_window", "figure")
    if normalized == "oi":
        return _window_state_value(session, "oi_window", "figure")
    if normalized == "sensor":
        return _window_state_value(session, "sensor_window", "figure")
    if normalized == "ip":
        return _window_state_value(session, "ip_window", "figure")
    if normalized == "display":
        return _window_state_value(session, "display_window", "figure")
    raise KeyError(f"Unknown window type: {object_type}")


def _next_headless_window_index(session: SessionContext, key: str) -> int:
    counters = session.custom.setdefault("_headless_window_counts", {})
    next_index = int(counters.get(key, 0)) + 1
    counters[key] = next_index
    return next_index


def _create_headless_window_placeholder(
    session: SessionContext,
    object_type: str,
) -> Any:
    normalized = _session_type_name(object_type)
    if normalized == "scene":
        window_key, axis_key = "scene_window", "sceneImage"
    elif normalized == "oi":
        window_key, axis_key = "oi_window", "oiImage"
    elif normalized == "sensor":
        window_key, axis_key = "sensor_window", "imgMain"
    elif normalized == "ip":
        window_key, axis_key = "ip_window", "ipImage"
    elif normalized == "display":
        window_key, axis_key = "display_window", "displayImage"
    else:
        raise KeyError(f"Unknown window type: {object_type}")

    index = _next_headless_window_index(session, normalized)
    figure_handle = f"{normalized}-figure-{index}"
    axis_handle = f"{normalized}-axis-{index}"
    app = {
        "figure1": figure_handle,
        axis_key: axis_handle,
        "headless_placeholder": True,
    }
    session.gui[window_key] = {
        "app": app,
        "hObject": figure_handle,
        "handles": app,
        "headless_placeholder": True,
    }
    return figure_handle


def _create_headless_graphwin_placeholder(session: SessionContext) -> Any:
    index = _next_headless_window_index(session, "graphwin")
    figure_handle = f"graphwin-figure-{index}"
    session.graphwin["hObject"] = figure_handle
    session.graphwin["handle"] = f"graphwin-handle-{index}"
    return figure_handle


def _create_headless_main_window_placeholder(session: SessionContext) -> dict[str, Any]:
    index = _next_headless_window_index(session, "main")
    figure_handle = f"main-figure-{index}"
    app = {
        "figure1": figure_handle,
        "headless_placeholder": True,
    }
    session.gui["main_window"] = {
        "app": app,
        "hObject": figure_handle,
        "handles": app,
        "headless_placeholder": True,
    }
    return app


def iset(
    *,
    directory: str | Path | None = None,
    name: str | None = None,
    version: str | float = "4.0",
    create_main_window: bool = True,
) -> SessionContext:
    session_dir = Path.cwd() if directory is None else Path(directory)
    session = ie_init_session(
        name=name or _latest_iset_session_name(session_dir),
        directory=session_dir,
    )
    session.version = str(version)
    session.directory = str(session_dir)
    if create_main_window:
        _create_headless_main_window_placeholder(session)
    return session


def ie_init(
    session: SessionContext | None = None,
    *,
    directory: str | Path | None = None,
    name: str | None = None,
    version: str | float | None = None,
    create_main_window: bool = False,
) -> SessionContext:
    """Headless ieInit() compatibility wrapper."""

    session_dir = Path.cwd() if directory is None else Path(directory)
    init_clear = False
    resolved_version: str | float = "4.0" if version is None else version
    if session is not None:
        init_clear = bool(ie_session_get(session, "init clear"))
        resolved_version = session.version or resolved_version
        ie_main_close(session)

    fresh = iset(
        directory=session_dir,
        name=name,
        version=resolved_version,
        create_main_window=create_main_window,
    )
    fresh.preferences["initclear"] = init_clear
    return fresh


def main_open(
    session: SessionContext,
    h_object: Any | None = None,
    eventdata: Any | None = None,
    handles: Any | None = None,
) -> bool:
    app = ie_session_get(session, "main window")
    if app is None:
        app = _create_headless_main_window_placeholder(session)

    if h_object is not None:
        if isinstance(h_object, dict) or hasattr(h_object, "figure1"):
            app = h_object
        else:
            app = {"figure1": _extract_app_figure(h_object)}

    figure_handle = _extract_app_figure(app)

    if handles is None:
        if isinstance(app, dict):
            handles_state: Any = dict(app)
        else:
            handles_state = {"figure1": figure_handle}
    else:
        handles_state = handles

    if isinstance(handles_state, dict):
        handles_state = dict(handles_state)
        handles_state.setdefault("figure1", figure_handle)
        handles_state.setdefault("output", figure_handle)
    elif handles_state is not None and getattr(handles_state, "output", None) is None:
        setattr(handles_state, "output", figure_handle)

    session.gui["main_window"] = {
        "app": app,
        "hObject": figure_handle,
        "eventdata": eventdata,
        "handles": handles_state,
        "headless_placeholder": bool(
            isinstance(app, dict) and app.get("headless_placeholder", False)
        ),
    }
    return True


def ie_main_w(session: SessionContext) -> Any:
    main_open(session)
    return ie_session_get(session, "main window")


def ie_print_session_info(
    session: SessionContext,
    obj_type: str = "all",
) -> str:
    key = param_format(obj_type)
    section_map = {
        "all": ("scene", "oi", "sensor", "ip"),
        "scene": ("scene",),
        "scenes": ("scene",),
        "oi": ("oi",),
        "ois": ("oi",),
        "sensor": ("sensor",),
        "sensors": ("sensor",),
        "isa": ("sensor",),
        "ip": ("ip",),
        "ips": ("ip",),
        "vcimage": ("ip",),
    }
    if key not in section_map:
        raise KeyError(f"Unknown session info object type: {obj_type}")

    labels = {
        "scene": ("Scenes:  (* selected)", "No Scenes."),
        "oi": ("OIs  (* selected):", "No OIs."),
        "sensor": ("Sensors (* selected):", "No Sensors."),
        "ip": ("IPs  (* selected):", "No IP."),
    }

    blocks: list[str] = []
    for section in section_map[key]:
        title, empty_message = labels[section]
        count = session_count_objects(session, section)
        if count == 0:
            blocks.append(empty_message)
            continue

        selected_id = session_get_selected_id(session, section)
        names = session_get_object_names(session, section)
        lines = [title, "------------"]
        for index, name in enumerate(names, start=1):
            marker = "*" if selected_id == index else ""
            lines.append(f"{index:2d} {name} {marker}".rstrip())
        blocks.append("\n".join(lines))

    return "\n\n" + "\n\n".join(blocks) + "\n\n"


def vc_select_figure(
    session: SessionContext,
    object_type: str,
    no_new_win: bool = False,
) -> Any:
    normalized = param_format(object_type)
    if normalized in {"graphwin", "graphwinfigure", "graphwindow"}:
        figure_handle = ie_session_get(session, "graph win figure")
        if figure_handle is None and not no_new_win:
            figure_handle = _create_headless_graphwin_placeholder(session)
        return figure_handle

    figure_handle = _window_figure_from_session(session, normalized)
    if figure_handle is None and not no_new_win:
        figure_handle = _create_headless_window_placeholder(session, normalized)
    return figure_handle


def ie_refresh_window(
    session: SessionContext,
    obj_type: str | BaseISETObject | dict[str, Any],
) -> Any:
    canonical = vc_equivalent_objtype(obj_type)
    session_type: str
    if canonical == "SCENE":
        session_type = "scene"
        app = ie_session_get(session, "scene window")
    elif canonical == "OPTICALIMAGE":
        session_type = "oi"
        app = ie_session_get(session, "oi window")
    elif canonical == "ISA":
        session_type = "sensor"
        app = ie_session_get(session, "sensor window")
    elif canonical == "VCIMAGE":
        session_type = "ip"
        app = ie_session_get(session, "ip window")
    else:
        raise ValueError(f"Unknown object type for ieRefreshWindow: {obj_type}")

    if app is None:
        vc_select_figure(session, session_type)
        if canonical == "SCENE":
            app = ie_session_get(session, "scene window")
        elif canonical == "OPTICALIMAGE":
            app = ie_session_get(session, "oi window")
        elif canonical == "ISA":
            app = ie_session_get(session, "sensor window")
        else:
            app = ie_session_get(session, "ip window")

    refresh = app.get("refresh") if isinstance(app, dict) else getattr(app, "refresh", None)
    if callable(refresh):
        refresh()
    return app


def ie_find_object_by_name(
    session: SessionContext,
    obj_type: str,
    obj_name: str,
) -> int | None:
    return session_find_object_by_name(session, obj_type, obj_name)


def ie_main_close(session: SessionContext) -> SessionContext:
    for window_name in ("scene window", "oi window", "sensor window", "ip window", "display window", "metrics window"):
        ie_session_set(session, window_name, None)
    session.gui = {"waitbar": int(bool(session.gui.get("waitbar", session.preferences.get("waitbar", 0))))}
    return session


def ie_windows_get(
    session: SessionContext,
    save_flag: bool = False,
) -> tuple[list[Any], list[Any]]:
    w_pos: list[Any] = []
    w_state: list[Any] = []
    for window_key, container_attr in _WINDOW_POSITION_SLOTS:
        container = _window_container(session, window_key, container_attr)
        position = _get_nested_value(container, "Position")
        if isinstance(position, tuple):
            position = list(position)
        elif isinstance(position, list):
            position = list(position)
        window_state = _get_nested_value(container, "WindowState")
        w_pos.append(position)
        w_state.append(window_state)
    if save_flag:
        session.preferences["wPos"] = list(w_pos)
        session.preferences["wState"] = list(w_state)
    return w_pos, w_state


def ie_windows_set(
    session: SessionContext,
    w_pos: list[Any] | tuple[Any, ...] | None = None,
    w_state: list[Any] | tuple[Any, ...] | bool | None = None,
) -> list[Any]:
    positions = _normalize_window_slots(session.preferences.get("wPos", _window_slot_defaults()) if w_pos is None else w_pos)
    restore_states = isinstance(w_state, (list, tuple)) or w_state is True
    states = _window_slot_defaults()
    if isinstance(w_state, (list, tuple)):
        states = _normalize_window_slots(w_state)
    elif w_state is True:
        states = _normalize_window_slots(session.preferences.get("wState", _window_slot_defaults()))

    for idx, (window_key, container_attr) in enumerate(_WINDOW_POSITION_SLOTS):
        container = _window_container(session, window_key, container_attr)
        if container is None:
            continue
        position = positions[idx]
        if position is not None:
            _set_nested_value(container, "Position", list(position) if isinstance(position, tuple) else position)
        if restore_states and states[idx] is not None:
            _set_nested_value(container, "WindowState", states[idx])

    session.preferences["wPos"] = list(positions)
    if restore_states:
        session.preferences["wState"] = list(states)
    return positions


def ie_get_object(
    session: SessionContext,
    object_type: str | BaseISETObject | dict[str, Any],
    object_id: int | None = None,
    *,
    with_id: bool = False,
) -> Any:
    requested_type = _ie_object_type(object_type)
    nested = _IE_NESTED_OBJECT_TYPES.get(requested_type)
    if nested is not None:
        parent_type, field_name = nested
        if object_id is None:
            object_id = session_get_selected_id(session, parent_type)
        parent = session_get_object(session, parent_type, object_id)
        value = None if parent is None else parent.fields.get(field_name)
        return (value, object_id) if with_id else value

    obj = session_get_object(session, requested_type, object_id)
    if object_id is None:
        object_id = session_get_selected_id(session, requested_type)
    return (obj, object_id) if with_id else obj


def ie_get_selected_object(
    session: SessionContext,
    object_type: str | BaseISETObject | dict[str, Any],
    *,
    with_object: bool = False,
) -> int | None | tuple[int | None, Any]:
    requested_type = _ie_object_type(object_type)
    object_id = session_get_selected_id(session, requested_type)
    if not with_object:
        return object_id
    return object_id, ie_get_object(session, requested_type, object_id)


def ie_select_object(
    session: SessionContext,
    object_type: str | BaseISETObject | dict[str, Any],
    value: int | BaseISETObject | None,
) -> None:
    session_set_selected(session, _ie_object_type(object_type), value)


def ie_delete_object(
    session: SessionContext,
    object_type: str | BaseISETObject | dict[str, Any],
    object_id: int | None = None,
) -> int:
    return session_delete_object(session, _ie_object_type(object_type), object_id)


def ie_replace_object(
    session: SessionContext,
    obj: BaseISETObject,
    object_id: int | None = None,
) -> BaseISETObject:
    return session_replace_object(session, obj, object_id, select=True)


def session_replace_object(
    session: SessionContext,
    obj: _T,
    object_id: int | None = None,
    *,
    select: bool = True,
) -> _T:
    object_type = _object_session_type(obj)
    bucket = session.objects.setdefault(object_type, {})
    if object_id is None:
        existing_id = session_object_id(obj)
        if existing_id is not None and existing_id in bucket:
            object_id = existing_id
        else:
            selected_id = session_get_selected_id(session, object_type)
            object_id = 1 if selected_id is None else selected_id
    object_id = int(object_id)

    if isinstance(obj, Camera):
        obj = track_camera_session_state(session, obj, select=select)
    elif isinstance(obj, ImageProcessor):
        obj = track_ip_session_state(session, obj, select=select)

    bucket[object_id] = obj
    obj.metadata["session_type"] = object_type
    obj.metadata["session_id"] = object_id
    session.next_ids[object_type] = max(int(session.next_ids.get(object_type, 1)), object_id + 1)
    if select:
        session.selected[object_type] = object_id
    return obj


def session_replace_and_select_object(
    session: SessionContext,
    obj: _T,
    object_id: int | None = None,
) -> _T:
    return session_replace_object(session, obj, object_id, select=True)


def _session_reindex_bucket(bucket: dict[int, BaseISETObject]) -> dict[int, BaseISETObject]:
    reindexed: dict[int, BaseISETObject] = {}
    for new_id, obj in enumerate((bucket[key] for key in sorted(bucket)), start=1):
        obj.metadata["session_id"] = new_id
        reindexed[new_id] = obj
    return reindexed


def session_delete_selected_object(session: SessionContext, object_type: str) -> int:
    normalized_type = _session_type_name(object_type)
    selected_id = session_get_selected_id(session, normalized_type)
    if selected_id is None:
        return 0
    return session_delete_object(session, normalized_type, selected_id)


def session_delete_object(
    session: SessionContext,
    object_type: str,
    object_id: int | None = None,
) -> int:
    normalized_type = _session_type_name(object_type)
    bucket = dict(session.objects.get(normalized_type, {}))
    if object_id is None:
        object_id = session_get_selected_id(session, normalized_type)
    if object_id is None or int(object_id) not in bucket:
        return len(bucket)

    target_id = int(object_id)
    del bucket[target_id]
    if bucket:
        bucket = _session_reindex_bucket(bucket)
        session.objects[normalized_type] = bucket
        session.selected[normalized_type] = max(1, target_id - 1)
    else:
        session.objects[normalized_type] = {}
        session.selected[normalized_type] = None
    session.next_ids[normalized_type] = len(bucket) + 1
    return len(bucket)


def session_delete_some_objects(
    session: SessionContext,
    object_type: str,
    delete_list: list[int] | tuple[int, ...] | None = None,
) -> int:
    normalized_type = _session_type_name(object_type)
    if not delete_list:
        return session_count_objects(session, normalized_type)

    remaining = session_count_objects(session, normalized_type)
    for object_id in sorted({int(value) for value in delete_list}, reverse=True):
        remaining = session_delete_object(session, normalized_type, object_id)
    return remaining


# MATLAB-style aliases.
ISET = iset
isetPath = iset_path
isetRootPath = iset_root_path

vcAddAndSelectObject = session_add_and_select_object
vcAddObject = session_add_object
vcCountObjects = session_count_objects
vcDeleteObject = session_delete_object
vcDeleteSomeObjects = session_delete_some_objects
vcDeleteSelectedObject = session_delete_selected_object
vcGetObjectType = session_get_object_type
vcGetObjectNames = session_get_object_names
vcGetObject = session_get_object_with_id
vcGetObjects = session_get_objects
vcGetSelectedObject = session_get_selected_pair
vcGetSelectedObjectID = session_get_selected_id
vcNewObjectValue = session_new_object_value
vcNewObjectName = session_new_object_name
vcReplaceAndSelectObject = session_replace_and_select_object
vcReplaceObject = session_replace_object
vcSetObjects = session_set_objects
vcSetSelectedObject = session_set_selected

ieAddObject = ie_add_object
ieDeleteObject = ie_delete_object
ieGetObject = ie_get_object
ieGetSelectedObject = ie_get_selected_object
ieAppGet = ie_app_get
ieEquivalentObjtype = ie_equivalent_objtype
ieFindObjectByName = ie_find_object_by_name
ieInit = ie_init
ieInitSession = ie_init_session
ieMainW = ie_main_w
ieMainClose = ie_main_close
iePrintSessionInfo = ie_print_session_info
ieReplaceObject = ie_replace_object
ieRefreshWindow = ie_refresh_window
ieSessionGet = ie_session_get
ieSessionSet = ie_session_set
ieWindowsGet = ie_windows_get
ieWindowsSet = ie_windows_set
ieSelectObject = ie_select_object
mainOpen = main_open
vcEquivalentObjtype = vc_equivalent_objtype
vcGetFigure = vc_get_figure
vcSetFigureHandles = vc_set_figure_handles
vcSelectFigure = vc_select_figure
