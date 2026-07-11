from __future__ import annotations

import numpy as np
from scipy.io import savemat

from pyisetcam.assets import AssetStore


def test_spectra_resolution_is_portable_across_filename_casing(tmp_path) -> None:
    asset_dir = tmp_path / "data" / "human"
    asset_dir.mkdir(parents=True)
    savemat(
        asset_dir / "xyzQuanta.mat",
        {"wavelength": np.array([500.0, 600.0]), "data": np.ones((2, 3))},
    )

    wave, spectra = AssetStore(tmp_path).load_spectra("XYZQuanta")

    np.testing.assert_array_equal(wave, [500.0, 600.0])
    assert spectra.shape == (2, 3)
