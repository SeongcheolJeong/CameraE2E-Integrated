from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

from pyisetcam.fileio import ie_dng_read, ie_dng_simple_info


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: decode_dng_for_octave.py <input.dng> <output.mat>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    image, info = ie_dng_read(input_path)
    if image is None:
        raise RuntimeError(f"Failed to decode DNG image data from {input_path}")

    simple = ie_dng_simple_info(info)
    payload = {
        "img": np.asarray(image),
        "isoSpeed": float(simple["isoSpeed"]),
        "exposureTime": float(simple["exposureTime"]),
        "blackLevel": np.asarray(simple["blackLevel"], dtype=float),
        "orientation": int(simple["orientation"]),
    }
    savemat(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
