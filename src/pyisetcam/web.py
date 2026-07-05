"""Headless web-search and remote-scene helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .assets import AssetStore
from .scene import scene_from_file, scene_save_image, scene_set
from .types import Scene, SessionContext
from .utils import param_format


def _store(asset_store: AssetStore | None) -> AssetStore:
    return asset_store or AssetStore.default()


def _request_bytes(url: str) -> bytes:
    request = Request(str(url), headers={"User-Agent": "pyisetcam/0.1.0"})
    with urlopen(request) as response:
        return response.read()


def _request_json(url: str) -> dict[str, Any]:
    payload = _request_bytes(url)
    return json.loads(payload.decode("utf-8"))


def _request_image(url: str) -> np.ndarray | bytes:
    payload = _request_bytes(url)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:
        return payload


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _nested_get(record: Any, *keys: str, default: Any = None) -> Any:
    current = record
    for key in keys:
        current = _record_get(current, key, default)
        if current is default:
            return default
    return current


class webData:
    """Headless wrapper over the upstream JSON-backed scene catalog."""

    _TYPE_MAP = {
        "hyperspectral": "Hyperspectral",
        "multispectral": "Multispectral",
        "hdr": "HDR",
        "rgb": "RGB",
    }

    def __init__(
        self,
        forType: str,
        *,
        asset_store: AssetStore | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.dataType = str(forType)
        self.defaultWavelist = np.arange(400.0, 701.0, 10.0, dtype=float)
        self.webDataCache = Path(cache_dir) if cache_dir is not None else Path.cwd() / "local" / "webData"
        self.webDataCache.mkdir(parents=True, exist_ok=True)
        store = _store(asset_store)
        catalog_path = store.resolve("web/webISETData.json")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        type_key = self._TYPE_MAP.get(param_format(forType))
        if type_key is None:
            valid = ", ".join(sorted(self._TYPE_MAP.values()))
            raise ValueError(f"Unknown webData type '{forType}'. Choose from: {valid}")
        self.ourDataStruct = [entry for entry in catalog.get(type_key, []) if _record_get(entry, "Name")]

    def search(self, ourTags: str) -> list[dict[str, Any]]:
        keywords = [token.strip().lower() for token in str(ourTags).split(",") if token.strip()]
        results: list[dict[str, Any]] = []
        for entry in self.ourDataStruct:
            entry_keywords = [str(token).strip().lower() for token in _record_get(entry, "Keywords", [])]
            if all(keyword in entry_keywords for keyword in keywords):
                results.append(dict(entry))
        return results

    def displayScene(
        self,
        fPhoto: Any,
        sceneType: str | None = None,
        *,
        keep_downloads: bool = False,
        asset_store: AssetStore | None = None,
        session: SessionContext | None = None,
    ) -> Scene:
        url = str(self.getImageURL(fPhoto, "large"))
        local_path = self.webDataCache / Path(url).name
        local_path.write_bytes(_request_bytes(url))
        current_type = param_format(sceneType or self.dataType)
        try:
            if current_type in {"hyperspectral", "multispectral"}:
                scene = scene_from_file(local_path, "multispectral", asset_store=_store(asset_store), session=session)
            elif current_type == "hdr":
                scene = scene_from_file(
                    local_path,
                    "multispectral",
                    None,
                    None,
                    self.defaultWavelist,
                    asset_store=_store(asset_store),
                    session=session,
                )
            else:
                scene = scene_from_file(
                    local_path,
                    "rgb",
                    None,
                    None,
                    self.defaultWavelist,
                    asset_store=_store(asset_store),
                    session=session,
                )
            return scene_set(scene, "name", self.getImageTitle(fPhoto))
        finally:
            if not keep_downloads and local_path.exists():
                local_path.unlink()

    def getImageURL(self, fPhoto: Any, wantSize: str) -> str:
        return str(_record_get(fPhoto, "Icon") if param_format(wantSize) == "thumbnail" else _record_get(fPhoto, "URL"))

    def getImageTitle(self, fPhoto: Any) -> str:
        return str(_record_get(fPhoto, "Name", ""))

    def getImage(self, fPhoto: Any, wantSize: str = "large") -> np.ndarray | bytes:
        return _request_image(self.getImageURL(fPhoto, wantSize))


class webFlickr:
    """Headless Flickr API wrapper with the legacy MATLAB surface."""

    def __init__(self) -> None:
        self.api_key = "a6365f14201cd3c5f34678e671b9ab8d"
        self.search_url = "https://www.flickr.com/services/rest/"
        self.format = "json"
        self.tag_mode = "all"
        self.nojsoncallback = "1"
        self.defaultPerPage = 20
        self.licenses = "1,2,3,4,5,6,7,8,9,10"
        self.sort = "relevance"
        self.defaultWavelist = np.arange(400.0, 701.0, 10.0, dtype=float)

    def search(self, ourTags: str) -> dict[str, Any]:
        query = urlencode(
            {
                "method": "flickr.photos.search",
                "api_key": self.api_key,
                "tags": ourTags,
                "format": self.format,
                "nojsoncallback": self.nojsoncallback,
                "content_type": 1,
                "sort": self.sort,
                "per_page": self.defaultPerPage,
                "tag_mode": self.tag_mode,
                "license": self.licenses,
            }
        )
        return _request_json(f"{self.search_url}?{query}")

    def getImageTitle(self, fPhoto: Any) -> str:
        return str(_record_get(fPhoto, "title", ""))

    def displayScene(
        self,
        fPhoto: Any,
        sceneType: str | None = None,
        *,
        asset_store: AssetStore | None = None,
        session: SessionContext | None = None,
    ) -> Scene:
        del sceneType
        image_data = self.getImage(fPhoto, "large")
        scene = scene_from_file(
            image_data,
            "rgb",
            None,
            None,
            self.defaultWavelist,
            asset_store=_store(asset_store),
            session=session,
        )
        return scene_set(scene, "name", self.getImageTitle(fPhoto))

    def getImageURL(self, fPhoto: Any, wantSize: str) -> str:
        size_suffix = "q" if param_format(wantSize) == "thumbnail" else "b"
        return (
            f"https://farm{_record_get(fPhoto, 'farm')}.staticflickr.com/"
            f"{_record_get(fPhoto, 'server')}/{_record_get(fPhoto, 'id')}_"
            f"{_record_get(fPhoto, 'secret')}_{size_suffix}.jpg"
        )

    def getImage(self, fPhoto: Any, wantSize: str = "large") -> np.ndarray | bytes:
        return _request_image(self.getImageURL(fPhoto, wantSize))


class webLOC:
    """Headless Library of Congress API wrapper with the legacy MATLAB surface."""

    def __init__(self) -> None:
        self.search_url = "https://loc.gov/pictures/search/"
        self.tag_mode = "all"
        self.defaultPerPage = 20
        self.sort = "date_desc"
        self.defaultWavelist = np.arange(400.0, 701.0, 10.0, dtype=float)

    def search(self, ourTags: str) -> list[dict[str, Any]]:
        query = urlencode({"fo": "json", "q": str(ourTags).replace(",", " "), "c": self.defaultPerPage * 3})
        payload = _request_json(f"{self.search_url}?{query}")
        return self.filterResults(_record_get(payload, "results", []))

    def filterResults(self, listResults: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for record in listResults or []:
            alt_text = str(_nested_get(record, "image", "alt", default=""))
            if alt_text.lower() == "item not digitized thumbnail":
                continue
            if _nested_get(record, "image", "full", default=None) is None:
                continue
            results.append(dict(record))
        return results

    def getImageTitle(self, fPhoto: Any) -> str:
        return str(_record_get(fPhoto, "title", ""))

    def displayScene(
        self,
        fPhoto: Any,
        sceneType: str | None = None,
        *,
        asset_store: AssetStore | None = None,
        session: SessionContext | None = None,
    ) -> Scene:
        del sceneType
        image_data = self.getImage(fPhoto, "large")
        scene = scene_from_file(
            image_data,
            "rgb",
            None,
            None,
            self.defaultWavelist,
            asset_store=_store(asset_store),
            session=session,
        )
        return scene_set(scene, "name", self.getImageTitle(fPhoto))

    def getImageURL(self, fPhoto: Any, wantSize: str) -> str:
        if param_format(wantSize) == "thumbnail":
            url = str(_nested_get(fPhoto, "image", "thumb", default=""))
        else:
            url = str(_nested_get(fPhoto, "image", "full", default=""))
        if url.startswith("//"):
            url = f"https:{url}"
        return url

    def getImage(self, fPhoto: Any, wantSize: str = "large") -> np.ndarray | bytes:
        return _request_image(self.getImageURL(fPhoto, wantSize))


class webPixabay:
    """Headless Pixabay API wrapper with the legacy MATLAB surface."""

    def __init__(self) -> None:
        self.key = "18230017-1d12c1c7c5182cfa172a39807"
        self.search_url = "https://pixabay.com/api/"
        self.defaultPerPage = 20
        self.defaultWavelist = np.arange(400.0, 701.0, 10.0, dtype=float)

    def search(self, ourTags: str) -> dict[str, Any]:
        tags = " ".join(token.strip() for token in str(ourTags).split(",") if token.strip())
        query = urlencode(
            {
                "key": self.key,
                "q": tags,
                "image_type": "photo",
                "pretty": "true",
                "order": "popular",
                "per_page": self.defaultPerPage,
            }
        )
        return _request_json(f"{self.search_url}?{query}")

    def getImageTitle(self, fPhoto: Any) -> str:
        return str(_record_get(fPhoto, "id", ""))

    def displayScene(
        self,
        fPhoto: Any,
        sceneType: str | None = None,
        *,
        asset_store: AssetStore | None = None,
        session: SessionContext | None = None,
    ) -> Scene:
        del sceneType
        image_data = self.getImage(fPhoto, "large")
        scene = scene_from_file(
            image_data,
            "rgb",
            None,
            None,
            self.defaultWavelist,
            asset_store=_store(asset_store),
            session=session,
        )
        return scene_set(scene, "name", self.getImageTitle(fPhoto))

    def getImageURL(self, fPhoto: Any, wantSize: str) -> str:
        if param_format(wantSize) == "thumbnail":
            return str(_record_get(fPhoto, "previewURL", ""))
        return str(_record_get(fPhoto, "largeImageURL", ""))

    def getImage(self, fPhoto: Any, wantSize: str = "large") -> np.ndarray | bytes:
        return _request_image(self.getImageURL(fPhoto, wantSize))


def webCreateThumbnails(
    folderPath: str | Path,
    *,
    useHDR: bool = False,
    asset_store: AssetStore | None = None,
) -> list[str]:
    """Create PNG thumbnails next to multispectral MAT scene files."""

    store = _store(asset_store)
    folder = Path(folderPath).expanduser()
    outputs: list[str] = []
    for mat_path in sorted(folder.glob("*.mat")):
        scene = scene_from_file(mat_path, "multispectral", asset_store=store)
        output_path = mat_path.with_suffix(".png")
        outputs.append(
            scene_save_image(
                scene,
                output_path,
                render_flag=3 if bool(useHDR) else 1,
                asset_store=store,
            )
        )
    return outputs


def web_create_thumbnails(
    folder_path: str | Path,
    *,
    use_hdr: bool = False,
    asset_store: AssetStore | None = None,
) -> list[str]:
    return webCreateThumbnails(folder_path, useHDR=use_hdr, asset_store=asset_store)


__all__ = [
    "webData",
    "webFlickr",
    "webLOC",
    "webPixabay",
    "webCreateThumbnails",
    "web_create_thumbnails",
]
