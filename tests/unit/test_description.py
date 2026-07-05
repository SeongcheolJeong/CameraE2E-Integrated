from __future__ import annotations

from pyisetcam import HeadlessDescriptionHandle, sensor_create, sensor_set, sensorDescription, sensor_description


def test_sensor_description_returns_headless_handle(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)

    table, string_table, handle = sensor_description(sensor, show=False, close_window=False)

    assert table.title == "ISET Parameter Table for a Sensor"
    assert string_table.ndim == 2
    assert string_table.shape[1] == 3
    assert isinstance(handle, HeadlessDescriptionHandle)
    assert handle.title == table.title
    handle.position = (1, 759, 627, 607)
    assert handle.position == (1, 759, 627, 607)


def test_sensor_description_supports_t_sensor_fpn_summary(asset_store) -> None:
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "dsnu sigma", 0.05)
    sensor = sensor_set(sensor, "prnu sigma", 1.0)
    sensor = sensor_set(sensor, "read noise volts", 0.1)

    table, string_table, handle = sensorDescription(sensor, show=False, close_window=False)

    assert handle is not None
    rows = {str(row[0]): str(row[1]) for row in string_table.tolist()}
    assert rows["Read noise (V)"] == "0.100"
    assert rows["Analog gain"] == "1"
    assert rows["Exposure time"] == "0"
