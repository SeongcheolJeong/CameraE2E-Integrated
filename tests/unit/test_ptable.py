from __future__ import annotations

import numpy as np

from pyisetcam import IEPTable, camera_create, iePTable, ieTableGet, ie_p_table, ie_table_get, oi_create, scene_create


def test_ie_p_table_scene_window_format(asset_store) -> None:
    scene = scene_create("uniform ee", 8, asset_store=asset_store)

    table, window = ie_p_table(scene, format="window")

    assert isinstance(table, IEPTable)
    assert window is None
    assert table.columns == ("Property", "Value", "Units")
    labels = [row[0] for row in table.data]
    assert "Name" in labels
    assert "Mean luminance" in labels
    assert table.title.endswith("for a Scene")


def test_ie_p_table_oi_embed_includes_optics_rows() -> None:
    oi = oi_create()

    table, window = iePTable(oi, format="embed")

    assert window is None
    assert table.columns == ("Property", "Value")
    labels = [row[0] for row in table.data]
    assert "Optics model" in labels
    assert "Compute method" in labels
    assert table.title.endswith("for an Optical Image")


def test_ie_p_table_camera_combines_pipeline_sections(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)

    table, _ = ie_p_table(camera, format="embed")

    labels = [row[0] for row in table.data]
    assert "Display name" in labels
    assert "Size (mm)" in labels
    assert "Compute method" in labels
    assert table.title.endswith("for a Camera")


def test_ie_table_get_filters_column_tables_with_and_conditions() -> None:
    table = {
        "subject": np.array(["J", "J", "Z"], dtype=object),
        "substrate": np.array(["tongue", "skin", "tongue"], dtype=object),
        "ewave": np.array([405, 415, 405], dtype=int),
        "file": np.array(["j_tongue_405.mat", "j_skin_415.mat", "z_tongue_405.mat"], dtype=object),
    }

    files, rows = ie_table_get(table, "subject", "J", "substrate", "tongue")

    assert files == ["j_tongue_405.mat"]
    assert np.array_equal(rows["subject"], np.array(["J"], dtype=object))
    assert np.array_equal(rows["substrate"], np.array(["tongue"], dtype=object))
    assert np.array_equal(rows["ewave"], np.array([405], dtype=int))


def test_ie_table_get_supports_or_filters_on_repeated_fields() -> None:
    rows = [
        {"subject": "J", "ewave": 405, "file": "j_405.mat"},
        {"subject": "J", "ewave": 415, "file": "j_415.mat"},
        {"subject": "Z", "ewave": 450, "file": "z_450.mat"},
    ]

    files, filtered = ieTableGet(rows, "ewave", 405, "ewave", 450, "operator", "or")

    assert files == ["j_405.mat", "z_450.mat"]
    assert filtered == [rows[0], rows[2]]


def test_ie_table_get_accepts_return_parameter_without_changing_tuple_contract() -> None:
    table = {
        "subject": ["J", "K"],
        "file": ["j.mat", "k.mat"],
    }

    files, rows = ieTableGet(table, "subject", "K", "return", "files")

    assert files == ["k.mat"]
    assert rows["subject"] == ["K"]
    assert rows["file"] == ["k.mat"]
