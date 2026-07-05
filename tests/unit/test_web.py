from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyisetcam.web as web_module
from pyisetcam import (
    scene_create,
    scene_get,
    scene_to_file,
    webCreateThumbnails,
    webData,
    webFlickr,
    webLOC,
    webPixabay,
)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_web_data_supports_search_thumbnail_and_scene_download(tmp_path, monkeypatch, asset_store) -> None:
    cached_scene = scene_create("uniform ee", 8, asset_store=asset_store)
    mat_path = tmp_path / "cached_scene.mat"
    scene_to_file(mat_path, cached_scene)
    mat_bytes = mat_path.read_bytes()
    thumb_bytes = _png_bytes(np.full((5, 7, 3), [32, 96, 160], dtype=np.uint8))

    def fake_urlopen(request):
        url = getattr(request, "full_url", str(request))
        if url.endswith(".mat"):
            return _Response(mat_bytes)
        if url.endswith(".jpg") or url.endswith(".png"):
            return _Response(thumb_bytes)
        raise AssertionError(url)

    monkeypatch.setattr(web_module, "urlopen", fake_urlopen)

    client = webData("Hyperspectral", asset_store=asset_store, cache_dir=tmp_path / "web-cache")
    results = client.search("Hyperspectral, Fruit")

    assert len(results) == 2
    selected = next(result for result in results if client.getImageTitle(result) == "Fruit Scene")
    thumbnail = client.getImage(selected, "thumbnail")
    assert isinstance(thumbnail, np.ndarray)
    assert thumbnail.shape == (5, 7, 3)

    scene = client.displayScene(selected, asset_store=asset_store)

    assert scene_get(scene, "name") == "Fruit Scene"
    assert scene.fields["source_type"] == "multispectral"
    assert not (tmp_path / "web-cache" / "FruitMCC.mat").exists()


def test_web_wrappers_support_flickr_loc_and_pixabay_workflows(monkeypatch, asset_store) -> None:
    image_bytes = _png_bytes(np.full((4, 6, 3), [120, 64, 16], dtype=np.uint8))

    def fake_urlopen(request):
        url = getattr(request, "full_url", str(request))
        if "flickr.photos.search" in url:
            payload = {
                "photos": {
                    "photo": [
                        {"title": "demo", "farm": 1, "server": "srv", "id": "42", "secret": "abc"}
                    ]
                }
            }
            return _Response(json.dumps(payload).encode("utf-8"))
        if "staticflickr.com" in url:
            return _Response(image_bytes)
        if "loc.gov/pictures/search/" in url:
            payload = {
                "results": [
                    {
                        "title": "Skip me",
                        "image": {"alt": "item not digitized thumbnail", "thumb": "//thumb.invalid", "full": "//full.invalid"},
                    },
                    {
                        "title": "Digitized poster",
                        "image": {"alt": "poster", "thumb": "//cdn.loc.gov/thumb.png", "full": "//cdn.loc.gov/full.png"},
                    },
                ]
            }
            return _Response(json.dumps(payload).encode("utf-8"))
        if "cdn.loc.gov" in url:
            return _Response(image_bytes)
        if "pixabay.com/api/" in url:
            payload = {
                "hits": [
                    {"id": 7, "previewURL": "https://pixabay.invalid/thumb.png", "largeImageURL": "https://pixabay.invalid/full.png"}
                ]
            }
            return _Response(json.dumps(payload).encode("utf-8"))
        if "pixabay.invalid" in url:
            return _Response(image_bytes)
        raise AssertionError(url)

    monkeypatch.setattr(web_module, "urlopen", fake_urlopen)

    flickr = webFlickr()
    flickr_results = flickr.search("cat")
    flickr_photo = flickr_results["photos"]["photo"][0]
    assert flickr.getImageTitle(flickr_photo) == "demo"
    assert flickr.getImageURL(flickr_photo, "thumbnail").endswith("_q.jpg")
    assert flickr.getImageURL(flickr_photo, "large").endswith("_b.jpg")
    flickr_scene = flickr.displayScene(flickr_photo, asset_store=asset_store)
    assert scene_get(flickr_scene, "name") == "demo"

    loc = webLOC()
    loc_results = loc.search("poster, train")
    assert len(loc_results) == 1
    assert loc.getImageTitle(loc_results[0]) == "Digitized poster"
    assert loc.getImageURL(loc_results[0], "large") == "https://cdn.loc.gov/full.png"
    loc_scene = loc.displayScene(loc_results[0], asset_store=asset_store)
    assert scene_get(loc_scene, "name") == "Digitized poster"

    pixabay = webPixabay()
    pixabay_results = pixabay.search("flowers, yellow")
    pixabay_photo = pixabay_results["hits"][0]
    assert pixabay.getImageTitle(pixabay_photo) == "7"
    assert pixabay.getImageURL(pixabay_photo, "thumbnail") == "https://pixabay.invalid/thumb.png"
    pixabay_scene = pixabay.displayScene(pixabay_photo, asset_store=asset_store)
    assert scene_get(pixabay_scene, "name") == "7"


def test_web_create_thumbnails_saves_pngs_for_multispectral_scene_files(tmp_path, asset_store) -> None:
    scene = scene_create("uniform ee", 8, asset_store=asset_store)
    mat_path = tmp_path / "demo_scene.mat"
    scene_to_file(mat_path, scene)

    outputs = webCreateThumbnails(tmp_path, asset_store=asset_store)

    assert outputs == [str(mat_path.with_suffix(".png").resolve())]
    assert mat_path.with_suffix(".png").exists()
