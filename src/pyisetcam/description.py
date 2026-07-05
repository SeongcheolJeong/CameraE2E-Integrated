"""Headless description helpers modeled on MATLAB *Description functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ptable import IEPTable, ie_p_table
from .types import OpticalImage, Sensor
from .utils import param_format


@dataclass
class HeadlessDescriptionHandle:
    """Minimal headless stand-in for a MATLAB figure handle."""

    title: str
    table: IEPTable
    position: tuple[int, int, int, int] | None = None
    visible: bool = True
    closed: bool = False


def _parse_description_options(
    args: tuple[Any, ...],
    *,
    show: bool,
    close_window: bool,
    reference_oi: OpticalImage | None,
) -> tuple[bool, bool, OpticalImage | None]:
    if len(args) % 2 != 0:
        raise ValueError("Description helpers expect key/value pairs.")
    show_value = bool(show)
    close_value = bool(close_window)
    reference = reference_oi
    for index in range(0, len(args), 2):
        key = param_format(args[index])
        value = args[index + 1]
        if key == "show":
            show_value = bool(value)
        elif key in {"closewindow", "closewindowflag", "closefigure"}:
            close_value = bool(value)
        elif key in {"referenceoi", "oi"}:
            if value is not None and not isinstance(value, OpticalImage):
                raise TypeError("reference oi must be an OpticalImage or None.")
            reference = value
        else:
            raise ValueError(f"Unsupported description option: {args[index]}")
    return show_value, close_value, reference


def _table_to_string_array(table: IEPTable) -> np.ndarray:
    return np.asarray([[str(item) for item in row] for row in table.data], dtype=object)


def sensor_description(
    sensor: Sensor,
    *args: Any,
    show: bool = True,
    close_window: bool = True,
    reference_oi: OpticalImage | None = None,
) -> tuple[IEPTable, np.ndarray, HeadlessDescriptionHandle | None]:
    """Return a headless sensor summary modeled on MATLAB ``sensorDescription``."""

    show_value, close_value, reference = _parse_description_options(
        args,
        show=show,
        close_window=close_window,
        reference_oi=reference_oi,
    )
    table, _ = ie_p_table(sensor, format="window", reference_oi=reference)
    string_table = _table_to_string_array(table)
    if show_value:
        for row in string_table:
            print(" | ".join(row.tolist()))
    handle = None
    if not close_value:
        handle = HeadlessDescriptionHandle(title=table.title, table=table)
    return table, string_table, handle


# MATLAB-style alias.
sensorDescription = sensor_description
