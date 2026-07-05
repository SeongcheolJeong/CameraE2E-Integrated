from __future__ import annotations

import pytest

from pyisetcam.assets import AssetStore


@pytest.fixture(scope="session")
def asset_store() -> AssetStore:
    return AssetStore.default()

