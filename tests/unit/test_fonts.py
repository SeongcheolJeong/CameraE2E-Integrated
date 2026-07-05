from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pyisetcam import (
    AddTextToImage,
    AddTextToImageWithBorder,
    BitmapFont,
    RasterizeText,
    add_text_to_image,
    add_text_to_image_with_border,
    bitmap_font,
    display_create,
    display_get,
    font_create,
    font_get,
    font_set,
    rasterize_text,
    scene_from_font,
    scene_get,
)

fonts_module = importlib.import_module("pyisetcam.fonts")


def test_fonts_module_helper_matlab_aliases() -> None:
    assert fonts_module.addTextToImage is fonts_module.add_text_to_image
    assert fonts_module.addTextToImageWithBorder is fonts_module.add_text_to_image_with_border
    assert fonts_module.bitmapFont is fonts_module.bitmap_font
    assert fonts_module.displayCreate is fonts_module.display_create
    assert fonts_module.displayGet is fonts_module.display_get
    assert fonts_module.displaySet is fonts_module.display_set
    assert fonts_module.Energy2Quanta is fonts_module.energy_to_quanta
    assert fonts_module.rasterizeText is fonts_module.rasterize_text
    assert fonts_module.RGB2XWFormat is fonts_module.rgb_to_xw_format
    assert fonts_module.sceneAdjustLuminance is fonts_module.scene_adjust_luminance
    assert fonts_module.sceneCreate is fonts_module.scene_create
    assert fonts_module.sceneGet is fonts_module.scene_get
    assert fonts_module.sceneSet is fonts_module.scene_set
    assert fonts_module.XW2RGBFormat is fonts_module.xw_to_rgb_format


def _expected_cached_bitmap(asset_store, relative_path: str) -> np.ndarray:
    path = asset_store.resolve(relative_path)
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    bitmap_plane = 1.0 - np.asarray(data["bmSrc"].dataIndex, dtype=float)
    pad_cols = 3 * int(np.ceil(bitmap_plane.shape[1] / 3.0)) - int(bitmap_plane.shape[1])
    if pad_cols > 0:
        bitmap_plane = np.pad(bitmap_plane, ((0, 0), (pad_cols, 0)), constant_values=1.0)
    bitmap = np.ones((bitmap_plane.shape[0], int(np.ceil(bitmap_plane.shape[1] / 3.0)), 3), dtype=float)
    for channel in range(3):
        bitmap[:, :, channel] = bitmap_plane[:, channel::3]
    return bitmap


def test_font_create_loads_cached_bitmap(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)
    expected = _expected_cached_bitmap(asset_store, "data/fonts/g-georgia-14-96.mat")

    assert font_get(font, "type") == "font"
    assert font_get(font, "name") == "g-georgia-14-96"
    np.testing.assert_allclose(font_get(font, "bitmap"), expected, rtol=0.0, atol=0.0)


def test_font_create_accepts_empty_and_blank_placeholders(asset_store) -> None:
    default_font = font_create(asset_store=asset_store)
    empty_placeholder = font_create([], [], [], [], [], asset_store=asset_store)
    blank_placeholder = font_create("", "", "", "", "", asset_store=asset_store)

    for candidate in (empty_placeholder, blank_placeholder):
        assert font_get(candidate, "name") == font_get(default_font, "name")
        assert font_get(candidate, "character") == font_get(default_font, "character")
        assert font_get(candidate, "family") == font_get(default_font, "family")
        assert font_get(candidate, "size") == font_get(default_font, "size")
        assert font_get(candidate, "dpi") == font_get(default_font, "dpi")
        assert font_get(candidate, "style") == font_get(default_font, "style")
        np.testing.assert_allclose(font_get(candidate, "bitmap"), font_get(default_font, "bitmap"), rtol=0.0, atol=0.0)


def test_font_get_supports_padded_and_inverted_bitmaps(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)

    bitmap = np.asarray(font_get(font, "bitmap"), dtype=float)
    padded = np.asarray(font_get(font, "padded bitmap", [2, 1], 0), dtype=float)
    inverted = np.asarray(font_get(font, "i bitmap"), dtype=float)

    assert padded.shape == (bitmap.shape[0] + 4, bitmap.shape[1] + 2, bitmap.shape[2])
    np.testing.assert_allclose(inverted, 1.0 - bitmap, rtol=0.0, atol=0.0)
    assert np.all(padded[:2, :, :] == 0.0)


def test_font_create_falls_back_to_headless_rendering(asset_store) -> None:
    font = font_create("A", "Georgia", 17, 97, asset_store=asset_store)
    bitmap = np.asarray(font_get(font, "bitmap"), dtype=float)

    assert font_get(font, "name") == "a-georgia-17-97"
    assert bitmap.ndim == 3
    assert bitmap.shape[2] == 3
    assert bitmap.shape[0] > 0 and bitmap.shape[1] > 0
    assert set(np.unique(bitmap)).issubset({0.0, 1.0})


def test_font_set_rebuilds_bitmap_from_updated_cached_parameters(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)
    updated = font_set(font, "dpi", 300, asset_store=asset_store)
    expected = _expected_cached_bitmap(asset_store, "data/fonts/g-georgia-14-300.mat")

    assert font_get(updated, "name") == "g-georgia-14-300"
    np.testing.assert_allclose(font_get(updated, "bitmap"), expected, rtol=0.0, atol=0.0)


def test_scene_from_font_matches_display_wave_and_geometry(asset_store) -> None:
    font = font_create("g", "Georgia", 14, 96, asset_store=asset_store)
    display = display_create("LCD-Apple.mat", asset_store=asset_store)
    scene = scene_from_font(font, display, None, [2, 3], [1, 2], 1, asset_store=asset_store)

    padded_bitmap = np.asarray(font_get(font, "padded bitmap", [1, 2], 1), dtype=float)
    expected_wave = np.asarray(display_get(display, "wave"), dtype=float)
    expected_size = (padded_bitmap.shape[0] * 2, padded_bitmap.shape[1] * 3)
    expected_fov = float(np.degrees(np.arctan2((0.0254 / 96.0) * padded_bitmap.shape[1], 0.5)))

    assert scene_get(scene, "name") == font_get(font, "name")
    np.testing.assert_allclose(scene_get(scene, "wave"), expected_wave, rtol=0.0, atol=0.0)
    assert scene_get(scene, "size") == expected_size
    assert scene_get(scene, "distance") == 0.5
    assert scene_get(scene, "fov") == expected_fov
    assert float(scene_get(scene, "mean luminance", asset_store=asset_store)) > 0.0


def test_bitmap_font_and_rasterize_text_replay_package_contract() -> None:
    font = bitmap_font("Courier New", 12, "AB", 2)
    alias_font = BitmapFont("Courier New", 12, "AB", 2)

    assert font["Name"] == "Courier New"
    assert font["Size"] == 12
    assert font["Characters"] == "AB"
    assert len(font["Bitmaps"]) == 2
    assert font["Bitmaps"][0].dtype == bool
    assert np.all(font["Bitmaps"][0][:, -2:] == 0)
    assert alias_font["Characters"] == font["Characters"]

    mask = rasterize_text("A B\nB", font)
    alias_mask = RasterizeText("A B\nB", alias_font)
    assert mask.dtype == bool
    assert mask.ndim == 2
    assert mask.shape[0] >= 2 * font["Size"]
    assert mask.shape[1] >= font["Bitmaps"][0].shape[1] + int(np.ceil(0.33 * font["Size"])) + font["Bitmaps"][1].shape[1]
    np.testing.assert_array_equal(alias_mask, mask)

    with pytest.raises(ValueError):
        rasterize_text("C", font)


def test_add_text_to_image_overlays_zero_based_mask() -> None:
    font = bitmap_font("Courier New", 12, "A", 0)
    mask = rasterize_text("A", font)
    image = np.zeros((mask.shape[0] + 2, mask.shape[1] + 3, 3), dtype=float)

    result = add_text_to_image(image, "A", [0, 1], [1.0, 0.5, 0.25], font)
    alias_result = AddTextToImage(image, "A", [0, 1], [1.0, 0.5, 0.25], font)
    expected = np.zeros_like(image)
    expected_slice = expected[0 : mask.shape[0], 1 : 1 + mask.shape[1], :]
    expected_slice[:, :, 0][mask] = 1.0
    expected_slice[:, :, 1][mask] = 0.5
    expected_slice[:, :, 2][mask] = 0.25

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(alias_result, expected, rtol=0.0, atol=0.0)


def test_add_text_to_image_with_border_colors_outline_only() -> None:
    font = bitmap_font("Courier New", 12, "A", 0)
    mask = rasterize_text("A", font)
    image = np.zeros((mask.shape[0] + 4, mask.shape[1] + 4, 3), dtype=float)

    plain = add_text_to_image(image, "A", [1, 1], [1.0, 0.0, 0.0], font)
    bordered = add_text_to_image_with_border(
        image,
        "A",
        [1, 1],
        [1.0, 0.0, 0.0],
        font,
        border_width=1,
        border_color=[0.0, 0.0, 1.0],
    )
    alias_bordered = AddTextToImageWithBorder(
        image,
        "A",
        [1, 1],
        [1.0, 0.0, 0.0],
        font,
        border_width=1,
        border_color=[0.0, 0.0, 1.0],
    )

    text_pixels = plain[:, :, 0] > 0
    border_pixels = bordered[:, :, 2] > 0
    assert np.any(border_pixels & ~text_pixels)
    assert not np.any(border_pixels & text_pixels)
    np.testing.assert_allclose(bordered[:, :, 0][text_pixels], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(alias_bordered, bordered, rtol=0.0, atol=0.0)
