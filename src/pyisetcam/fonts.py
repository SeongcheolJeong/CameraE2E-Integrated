"""Headless font helpers and scene rendering."""

from __future__ import annotations

import copy
import string
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import convolve2d

from .assets import AssetStore
from .display import display_create, display_get, display_set
from .exceptions import MissingAssetError
from .scene import scene_adjust_luminance, scene_create, scene_get, scene_set
from .session import track_session_object
from .types import SessionContext
from .utils import energy_to_quanta, param_format, rgb_to_xw_format, xw_to_rgb_format


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


def _font_name(letter: str, family: str, size: int, dpi: int) -> str:
    return str(f"{letter}-{family}-{int(size)}-{int(dpi)}").lower()


def _font_asset_path(font: dict[str, Any]) -> Path:
    return Path("data/fonts") / f"{font_get(font, 'name')}.mat"


def _font_bitmap_from_asset(font: dict[str, Any], store: AssetStore) -> np.ndarray:
    data = store.load_mat(_font_asset_path(font))
    bm_src = data["bmSrc"]
    bitmap_plane = 1.0 - np.asarray(bm_src.dataIndex, dtype=float)
    pad_cols = 3 * int(np.ceil(bitmap_plane.shape[1] / 3.0)) - int(bitmap_plane.shape[1])
    if pad_cols > 0:
        bitmap_plane = np.pad(bitmap_plane, ((0, 0), (pad_cols, 0)), constant_values=1.0)
    bitmap = np.ones((bitmap_plane.shape[0], int(np.ceil(bitmap_plane.shape[1] / 3.0)), 3), dtype=float)
    for channel in range(3):
        bitmap[:, :, channel] = bitmap_plane[:, channel::3]
    return bitmap


def _load_pillow_font(family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_names = [
        family,
        f"{family}.ttf",
        f"{family}.otf",
        family.lower(),
        f"{family.lower()}.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidate_names:
        try:
            return ImageFont.truetype(candidate, max(int(size), 1))
        except OSError:
            continue
    return ImageFont.load_default()


def _font_bitmap_from_pillow(font: dict[str, Any]) -> np.ndarray:
    character = str(font_get(font, "character"))
    family = str(font_get(font, "family"))
    size = int(font_get(font, "size"))
    dpi = int(font_get(font, "dpi"))
    scaled_size = max(int(round(size * max(float(dpi) / 72.0, 1.0))), 1)
    pil_font = _load_pillow_font(family, scaled_size)

    canvas_side = max(scaled_size * 6, 64)
    image = Image.new("L", (canvas_side, canvas_side), color=255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), character, font=pil_font)
    offset_x = max((canvas_side - (bbox[2] - bbox[0])) // 2 - bbox[0], 0)
    offset_y = max((canvas_side - (bbox[3] - bbox[1])) // 2 - bbox[1], 0)
    draw.text((offset_x, offset_y), character, fill=0, font=pil_font)

    bitmap_gray = np.asarray(image, dtype=float)
    occupied = bitmap_gray < 255.0
    if np.any(occupied):
        rows = np.where(np.any(occupied, axis=1))[0]
        cols = np.where(np.any(occupied, axis=0))[0]
        bitmap_gray = bitmap_gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    bitmap = (bitmap_gray > 127.0).astype(float)
    return np.repeat(bitmap[:, :, np.newaxis], 3, axis=2)


def _bitmap_font_characters_to_string(characters: Any | None) -> str:
    if characters is None:
        return string.ascii_letters + string.digits + string.punctuation
    if isinstance(characters, str):
        return characters
    if isinstance(characters, np.ndarray):
        flat = characters.reshape(-1).tolist()
        return "".join(str(char) for char in flat)
    if isinstance(characters, (list, tuple)):
        return "".join(str(char) for char in characters)
    return str(characters)


def _bitmap_font_default_padding(size: int, padding: Any | None) -> int:
    if padding is None:
        return int(np.ceil(0.02 * float(size)))
    return int(np.ceil(float(padding)))


def _bitmap_font_struct(font: dict[str, Any]) -> dict[str, Any]:
    if "Characters" in font and "Bitmaps" in font and "Size" in font:
        return {
            "Name": str(font.get("Name", font.get("name", "Courier New"))),
            "Size": int(font["Size"]),
            "Characters": _bitmap_font_characters_to_string(font["Characters"]),
            "Bitmaps": [np.asarray(bitmap, dtype=bool) for bitmap in font["Bitmaps"]],
        }
    if font.get("type") == "font":
        return bitmap_font(
            str(font.get("family", "Georgia")),
            int(font.get("size", 14)),
            str(font.get("character", "g")),
            0,
        )
    raise TypeError("Font must be a BitmapFont struct or font name.")


def _bitmap_font_render_character(name: str, size: int, character: str, padding: int) -> np.ndarray:
    pil_font = _load_pillow_font(name, size)
    canvas_side = max(int(size) * 6, 64)
    image = Image.new("L", (canvas_side, canvas_side), color=255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), character, font=pil_font)
    offset_x = max(8 - bbox[0], 0)
    offset_y = max(8 - bbox[1], 0)
    draw.text((offset_x, offset_y), character, fill=0, font=pil_font)

    bitmap = np.asarray(image, dtype=float)
    occupied = bitmap < 255.0
    if np.any(occupied):
        rows = np.where(np.any(occupied, axis=1))[0]
        cols = np.where(np.any(occupied, axis=0))[0]
        bitmap = bitmap[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    else:
        bitmap = np.zeros((int(size), 1), dtype=float)

    mask = bitmap < 160.0
    if padding > 0:
        mask = np.pad(mask, ((0, 0), (0, padding)), constant_values=False)
    return np.asarray(mask, dtype=bool)


def _oversample_pair(value: Any | None) -> tuple[int, int]:
    if value is None:
        return (20, 20)
    samples = np.asarray(value, dtype=int).reshape(-1)
    if samples.size == 1:
        samples = np.repeat(samples, 2)
    if samples.size < 2:
        raise ValueError("Oversample must provide one or two integer values.")
    return (max(int(samples[0]), 1), max(int(samples[1]), 1))


def _display_compute_simple(display: Any, image: np.ndarray, oversample: tuple[int, int]) -> np.ndarray:
    n_primaries = int(display_get(display, "n primaries"))
    if image.shape[2] < n_primaries:
        image = np.pad(image, ((0, 0), (0, 0), (0, n_primaries - image.shape[2])), constant_values=0.0)
    elif image.shape[2] > n_primaries:
        image = image[:, :, :n_primaries]
    image = np.repeat(image, oversample[0], axis=0)
    image = np.repeat(image, oversample[1], axis=1)
    return np.asarray(image, dtype=float)


def bitmap_font(
    name: str = "Courier New",
    size: int = 32,
    characters: Any | None = None,
    padding: Any | None = None,
) -> dict[str, Any]:
    font_size = int(np.ceil(float(size)))
    font_padding = _bitmap_font_default_padding(font_size, padding)
    glyphs = _bitmap_font_characters_to_string(characters)
    bitmaps = [
        _bitmap_font_render_character(str(name), font_size, character, font_padding)
        for character in glyphs
    ]
    return {
        "Name": str(name),
        "Size": font_size,
        "Characters": glyphs,
        "Bitmaps": bitmaps,
    }


def rasterize_text(
    text: str = "No string specified.",
    font: dict[str, Any] | str = "Arial",
    font_size: int = 32,
) -> np.ndarray:
    normalized = str(text).replace("\t", "    ").replace("\r\n", "\n")
    control_chars = {chr(index) for index in list(range(0, 10)) + list(range(11, 32)) + [127]}
    normalized = "".join(character for character in normalized if character not in control_chars)

    characters = "".join(
        dict.fromkeys(character for character in normalized if character not in {" ", "\n"})
    )
    if isinstance(font, dict):
        font_struct = _bitmap_font_struct(font)
    else:
        font_struct = bitmap_font(str(font), font_size, characters)

    missing = set(characters) - set(font_struct["Characters"])
    if missing:
        raise ValueError("The font provided is missing some of the necessary characters.")

    line = 0
    x_position = 0
    space_size = int(np.ceil(0.33 * float(font_struct["Size"])))
    placements: list[tuple[int, int, np.ndarray]] = []
    height = 0
    width = 0

    for character in normalized:
        if character == " ":
            x_position += space_size
            height = max(height, line * int(font_struct["Size"]) + int(font_struct["Size"]))
            width = max(width, x_position)
            continue
        if character == "\n":
            line += 1
            x_position = 0
            height = max(height, line * int(font_struct["Size"]) + int(font_struct["Size"]))
            continue

        index = font_struct["Characters"].index(character)
        bitmap = np.asarray(font_struct["Bitmaps"][index], dtype=bool)
        placements.append((line * int(font_struct["Size"]), x_position, bitmap))
        x_position += int(bitmap.shape[1])
        height = max(height, line * int(font_struct["Size"]) + int(bitmap.shape[0]))
        width = max(width, x_position)

    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=bool)

    image = np.zeros((height, width), dtype=bool)
    for row, col, bitmap in placements:
        row_end = row + bitmap.shape[0]
        col_end = col + bitmap.shape[1]
        image[row:row_end, col:col_end] |= bitmap
    return image


def add_text_to_image(
    image: np.ndarray | None = None,
    text: str = "No string specified.",
    position: Any | None = None,
    color: Any | None = None,
    font: dict[str, Any] | str = "Arial",
    font_size: int = 32,
) -> np.ndarray:
    if image is None:
        sample = np.outer(np.linspace(0.0, 1.0, 500), np.linspace(0.0, 1.0, 500))
        image = np.dstack((sample, np.rot90(sample), np.rot90(sample, 2)))
    if position is None:
        position = [1, 1]
    if color is None:
        color = [1, 1, 0]

    output = np.array(image, copy=True)
    image_rows, image_cols = output.shape[:2]
    text_mask = np.asarray(rasterize_text(text, font, font_size), dtype=bool)
    if text_mask.size == 0:
        return output

    row_offset, col_offset = np.asarray(position, dtype=int).reshape(-1)[:2]
    row_start = max(int(row_offset), 0)
    col_start = max(int(col_offset), 0)
    src_row_start = max(-int(row_offset), 0)
    src_col_start = max(-int(col_offset), 0)
    row_end = min(int(row_offset) + text_mask.shape[0], image_rows)
    col_end = min(int(col_offset) + text_mask.shape[1], image_cols)
    if row_start >= row_end or col_start >= col_end:
        return output

    mask = text_mask[src_row_start : src_row_start + (row_end - row_start), src_col_start : src_col_start + (col_end - col_start)]
    scale_factor = 255 if output.dtype == np.uint8 else 1
    color_values = np.asarray(color, dtype=float).reshape(-1)
    if output.ndim == 2:
        color_values = np.asarray([float(np.mean(color_values))], dtype=float)
    elif color_values.size == 1:
        color_values = np.repeat(color_values, output.shape[2])

    for channel_index in range(color_values.size):
        if output.ndim == 2:
            plane = output
        else:
            plane = output[:, :, channel_index]
        region = plane[row_start:row_end, col_start:col_end]
        region[mask] = scale_factor * float(color_values[channel_index])
        if output.ndim == 2:
            output = plane
        else:
            output[:, :, channel_index] = plane
    return output


def add_text_to_image_with_border(
    image: np.ndarray | None = None,
    text: str = "No string specified.",
    position: Any | None = None,
    color: Any | None = None,
    font: dict[str, Any] | str = "Arial",
    font_size: int = 32,
    border_width: int = 1,
    border_color: Any = 0,
) -> np.ndarray:
    if int(border_width) == 0:
        return add_text_to_image(image, text, position, color, font, font_size)
    if image is None:
        image = np.zeros((64, 64, 3), dtype=float)

    base_shape = image.shape[:2]
    mask = add_text_to_image(
        np.zeros(base_shape, dtype=bool),
        text,
        position,
        1,
        font,
        font_size,
    )
    width = int(border_width)
    coordinates_x, coordinates_y = np.meshgrid(np.arange(1, 2 * width + 2), np.arange(1, 2 * width + 2))
    kernel = ((coordinates_x - (width + 0.5) - 0.5) ** 2 + (coordinates_y - (width + 0.5) - 0.5) ** 2) <= (width + 0.5) ** 2
    outline = np.logical_xor(convolve2d(mask.astype(float), kernel.astype(float), mode="same") != 0, mask)

    output = add_text_to_image(image, text, position, color, font, font_size)
    scale_factor = 255 if output.dtype == np.uint8 else 1
    border_values = np.asarray(border_color, dtype=float).reshape(-1)
    if output.ndim == 2:
        border_values = np.asarray([float(np.mean(border_values))], dtype=float)
    elif border_values.size == 1:
        border_values = np.repeat(border_values, output.shape[2])

    for channel_index in range(border_values.size):
        if output.ndim == 2:
            plane = output
        else:
            plane = output[:, :, channel_index]
        plane[outline] = scale_factor * float(border_values[channel_index])
        if output.ndim == 2:
            output = plane
        else:
            output[:, :, channel_index] = plane
    return output


def font_bitmap_get(
    font: dict[str, Any],
    *,
    asset_store: AssetStore | None = None,
) -> np.ndarray:
    if font is None:
        raise ValueError("font required")

    store = _store(asset_store)
    try:
        return _font_bitmap_from_asset(font, store)
    except MissingAssetError:
        return _font_bitmap_from_pillow(font)


def font_create(
    letter: str = "g",
    family: str = "Georgia",
    sz: int = 14,
    dpi: int = 96,
    style: str = "NORMAL",
    *,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    resolved_letter = "g" if _is_empty_dispatch_placeholder(letter) else str(letter)
    resolved_family = "Georgia" if _is_empty_dispatch_placeholder(family) else str(family)
    resolved_size = 14 if _is_empty_dispatch_placeholder(sz) else int(sz)
    resolved_dpi = 96 if _is_empty_dispatch_placeholder(dpi) else int(dpi)
    resolved_style = "NORMAL" if _is_empty_dispatch_placeholder(style) else str(style)
    font = {
        "type": "font",
        "name": _font_name(resolved_letter, resolved_family, resolved_size, resolved_dpi),
        "character": resolved_letter,
        "size": resolved_size,
        "family": resolved_family,
        "style": resolved_style,
        "dpi": resolved_dpi,
    }
    font["bitmap"] = font_bitmap_get(font, asset_store=asset_store)
    return font


def font_get(font: dict[str, Any], param: str, *args: Any) -> Any:
    if param is None:
        raise ValueError("Parameter must be defined.")

    key = param_format(param)
    if key == "type":
        return font["type"]
    if key == "name":
        return font["name"]
    if key == "character":
        return font["character"]
    if key == "size":
        return font["size"]
    if key == "family":
        return font["family"]
    if key == "style":
        return font["style"]
    if key == "dpi":
        return font["dpi"]
    if key == "bitmap":
        return np.asarray(font["bitmap"], dtype=float)
    if key == "ibitmap":
        return 1.0 - np.asarray(font_get(font, "bitmap"), dtype=float)
    if key == "paddedbitmap":
        pad_size = np.asarray(args[0] if args else [7, 7], dtype=int).reshape(-1)
        if pad_size.size == 1:
            pad_size = np.repeat(pad_size, 2)
        pad_val = float(args[1]) if len(args) > 1 else 1.0
        bitmap = np.asarray(font_get(font, "bitmap"), dtype=float)
        return np.pad(
            bitmap,
            ((int(pad_size[0]), int(pad_size[0])), (int(pad_size[1]), int(pad_size[1])), (0, 0)),
            constant_values=pad_val,
        )
    if key == "ipaddedbitmap":
        return 1.0 - np.asarray(font_get(font, "padded bitmap", *(args[:2])), dtype=float)
    raise KeyError(f"Unknown parameter: {param}")


def font_set(
    font: dict[str, Any],
    param: str,
    val: Any,
    *args: Any,
    asset_store: AssetStore | None = None,
) -> dict[str, Any]:
    if param is None:
        raise ValueError("Parameter must be defined.")

    updated = copy.deepcopy(font)
    key = param_format(param)
    if key == "type":
        return updated
    if key == "name":
        updated["name"] = str(val)
        return updated
    if key == "character":
        updated["character"] = str(val)
    elif key == "size":
        updated["size"] = int(val)
    elif key == "family":
        updated["family"] = str(val)
    elif key == "style":
        updated["style"] = str(val)
    elif key == "dpi":
        updated["dpi"] = int(val)
    elif key == "bitmap":
        updated["bitmap"] = np.asarray(val, dtype=float)
        return updated
    else:
        raise KeyError(f"Unknown parameter: {param}")

    rebuilt = font_create(
        updated["character"],
        updated["family"],
        int(updated["size"]),
        int(updated["dpi"]),
        updated.get("style", "NORMAL"),
        asset_store=asset_store,
    )
    return rebuilt


def scene_from_font(
    font: dict[str, Any] | None = None,
    dsp: Any | None = None,
    scene: Any | None = None,
    o_sample: Any | None = None,
    *args: Any,
    asset_store: AssetStore | None = None,
    session: SessionContext | None = None,
) -> Any:
    store = _store(asset_store)
    current_font = font_create(asset_store=store) if font is None else copy.deepcopy(font)
    display = display_create("LCD-Apple", asset_store=store) if dsp is None else (
        display_create(dsp, asset_store=store) if isinstance(dsp, str) else dsp
    )

    if scene is None:
        current_scene = scene_create("default", 16, np.asarray(display_get(display, "wave"), dtype=float), asset_store=store)
    else:
        current_scene = scene.clone() if hasattr(scene, "clone") else copy.deepcopy(scene)

    oversample = _oversample_pair(o_sample)
    pad_size = args[0] if len(args) > 0 else None
    pad_val = args[1] if len(args) > 1 else None

    if float(display_get(display, "dpi")) != float(font_get(current_font, "dpi")):
        warnings.warn("Adjusting display dpi to match font", stacklevel=2)
        display = display_set(display, "dpi", float(font_get(current_font, "dpi")))

    if pad_size is None and pad_val is None:
        padded_bitmap = np.asarray(font_get(current_font, "padded bitmap"), dtype=float)
    elif pad_val is None:
        padded_bitmap = np.asarray(font_get(current_font, "padded bitmap", pad_size), dtype=float)
    else:
        padded_bitmap = np.asarray(font_get(current_font, "padded bitmap", pad_size, pad_val), dtype=float)
    display_image = _display_compute_simple(display, padded_bitmap, oversample)
    display_xw, rows, cols, _ = rgb_to_xw_format(display_image)
    spd = np.asarray(display_get(display, "spd"), dtype=float)
    energy = xw_to_rgb_format(display_xw @ spd.T, rows, cols)
    wave = np.asarray(display_get(display, "wave"), dtype=float)
    photons = energy_to_quanta(energy, wave)

    current_scene = scene_set(current_scene, "photons", photons)

    white_point = np.asarray(display_get(display, "white point"), dtype=float).reshape(-1)
    channel_index = min(1, padded_bitmap.shape[2] - 1)
    support_fraction = float(np.mean(padded_bitmap[:, :, channel_index]))
    current_scene = scene_adjust_luminance(current_scene, float(white_point[1]) * support_fraction, asset_store=store)

    distance_m = 0.5
    current_scene = scene_set(current_scene, "distance", distance_m)

    dot_size_m = 0.0254 / max(float(font_get(current_font, "dpi")), 1.0e-12)
    width_m = dot_size_m * float(padded_bitmap.shape[1])
    current_scene = scene_set(current_scene, "fov", float(np.degrees(np.arctan2(width_m, distance_m))))
    current_scene = scene_set(current_scene, "name", str(font_get(current_font, "name")))
    return track_session_object(session, current_scene)


AddTextToImage = add_text_to_image
AddTextToImageWithBorder = add_text_to_image_with_border
addTextToImage = add_text_to_image
addTextToImageWithBorder = add_text_to_image_with_border
BitmapFont = bitmap_font
bitmapFont = bitmap_font
displayCreate = display_create
displayGet = display_get
displaySet = display_set
Energy2Quanta = energy_to_quanta
RasterizeText = rasterize_text
rasterizeText = rasterize_text
fontBitmapGet = font_bitmap_get
fontCreate = font_create
fontGet = font_get
fontSet = font_set
RGB2XWFormat = rgb_to_xw_format
sceneAdjustLuminance = scene_adjust_luminance
sceneCreate = scene_create
sceneGet = scene_get
sceneSet = scene_set
sceneFromFont = scene_from_font
XW2RGBFormat = xw_to_rgb_format
