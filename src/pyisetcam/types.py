"""Public dataclass-backed ISET object types."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


def _default_session_preferences() -> dict[str, Any]:
    return {
        "fontSize": 12,
        "waitbar": 0,
        "wPos": [None, None, None, None, None, None, None],
        "wState": [None, None, None, None, None, None, None],
        "initclear": False,
    }


def _default_session_selected() -> dict[str, int | None]:
    return {
        "scene": None,
        "oi": None,
        "sensor": None,
        "ip": None,
        "display": None,
        "camera": None,
        "graphwin": None,
    }


def _default_session_gui() -> dict[str, Any]:
    return {"waitbar": 0}


def _default_session_render_state() -> dict[str, Any]:
    return {
        "scene_gamma": 1.0,
        "scene_display_flag": 1,
        "oi_gamma": 1.0,
        "oi_display_flag": 1,
        "sensor_gamma": 1.0,
        "ip_gamma": 1.0,
    }


@dataclass
class BaseISETObject:
    """Common mutable storage for MATLAB-like extensibility."""

    name: str
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "BaseISETObject":
        return copy.deepcopy(self)


@dataclass
class Scene(BaseISETObject):
    name: str = "scene"
    type: str = "scene"


@dataclass
class OpticalImage(BaseISETObject):
    name: str = "opticalimage"
    type: str = "opticalimage"


@dataclass
class Sensor(BaseISETObject):
    name: str = "sensor"
    type: str = "sensor"


@dataclass
class ImageProcessor(BaseISETObject):
    name: str = "vcimage"
    type: str = "vcimage"


@dataclass
class Display(BaseISETObject):
    name: str = "display"
    type: str = "display"


@dataclass
class Camera(BaseISETObject):
    name: str = "camera"
    type: str = "camera"


@dataclass
class SessionContext:
    """Optional vcSESSION-style object registry for compatibility workflows."""

    name: str = "vcSESSION"
    directory: str = ""
    version: str | None = None
    init_help: bool = False
    objects: dict[str, dict[int, BaseISETObject]] = field(default_factory=dict)
    selected: dict[str, int | None] = field(default_factory=_default_session_selected)
    next_ids: dict[str, int] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=_default_session_preferences)
    gui: dict[str, Any] = field(default_factory=_default_session_gui)
    custom: dict[str, Any] = field(default_factory=dict)
    render_state: dict[str, Any] = field(default_factory=_default_session_render_state)
    graphwin: dict[str, Any] = field(default_factory=dict)
    gpu_compute: bool = False
    image_size_threshold: float = 1e6
