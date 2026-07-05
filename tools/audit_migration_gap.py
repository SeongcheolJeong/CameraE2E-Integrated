"""Generate a machine-readable migration gap ledger and summary report.

The ledger is written as JSON-compatible YAML so it stays dependency-free while
remaining readable by YAML tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / ".cache" / "upstream" / "isetcam"
CASES_PATH = REPO_ROOT / "tests" / "parity" / "cases.yaml"
LEDGER_PATH = REPO_ROOT / "docs" / "migration-gap-ledger.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "migration-gap" / "latest.json"
README_PATH = REPO_ROOT / "README.md"
IMPLEMENTATION_PLAN_PATH = REPO_ROOT / "docs" / "implementation-plan.md"
SRC_ROOT = REPO_ROOT / "src" / "pyisetcam"
TEST_ROOT = REPO_ROOT / "tests" / "unit"
SUPPORTED_CODE_SUFFIXES = {".m", ".mlapp", ".fig"}
SHORT_SIGNAL_PREFIXES = (
    "dl",
    "oi",
    "ip",
    "lf",
    "rt",
    "wvf",
    "psf",
    "si",
    "ie",
    "sc",
    "ml",
    "vc",
    "hc",
    "xyy",
    "xyz",
    "lms",
)

FAMILY_SURFACE_MAP = {
    "camera": ["pyisetcam.camera"],
    "color": ["pyisetcam.color", "pyisetcam.illuminant"],
    "data/human": [
        "pyisetcam.assets",
        "pyisetcam.color",
        "pyisetcam.display",
        "pyisetcam.camera",
        "pyisetcam.sensor",
        "pyisetcam.optics",
    ],
    "data/displays": ["pyisetcam.assets", "pyisetcam.display", "pyisetcam.scene", "pyisetcam.parity"],
    "data/fonts": ["pyisetcam.fonts"],
    "data/images": ["pyisetcam.scene", "pyisetcam.fileio", "pyisetcam.camera", "pyisetcam.parity"],
    "data/lens": ["pyisetcam.optics"],
    "data/lights": ["pyisetcam.assets", "pyisetcam.color", "pyisetcam.illuminant", "pyisetcam.scene"],
    "data/optics": ["pyisetcam.assets", "pyisetcam.optics", "pyisetcam.utils"],
    "data/safetystandards": ["pyisetcam.metrics", "pyisetcam.assets"],
    "data/sensor": ["pyisetcam.assets", "pyisetcam.sensor", "pyisetcam.camera"],
    "data/surfaces": ["pyisetcam.assets", "pyisetcam.scene", "pyisetcam.color", "pyisetcam.ip", "pyisetcam.parity"],
    "displays": ["pyisetcam.display"],
    "fonts": ["pyisetcam.fonts", "pyisetcam.display", "pyisetcam.scene"],
    "human": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "imgproc": ["pyisetcam.ip"],
    "imgproc/binning": ["pyisetcam.ip"],
    "imgproc/demosaic": ["pyisetcam.ip"],
    "imgproc/illuminant": ["pyisetcam.ip", "pyisetcam.illuminant"],
    "main": ["pyisetcam.session"],
    "metrics": ["pyisetcam.metrics", "pyisetcam.scielab", "pyisetcam.iso"],
    "metrics/cielab": ["pyisetcam.scielab"],
    "metrics/iso": ["pyisetcam.iso"],
    "metrics/scielab": ["pyisetcam.scielab"],
    "opticalimage": ["pyisetcam.optics"],
    "opticalimage/depth": ["pyisetcam.optics"],
    "opticalimage/optics": ["pyisetcam.optics"],
    "opticalimage/raytrace": ["pyisetcam.optics"],
    "opticalimage/wavefront": ["pyisetcam.optics"],
    "scene": ["pyisetcam.scene"],
    "scene/depth": ["pyisetcam.scene"],
    "scene/illumination": ["pyisetcam.scene", "pyisetcam.illuminant"],
    "scene/imgtargets": ["pyisetcam.scene"],
    "scene/macbeth": ["pyisetcam.scene"],
    "scene/pattern": ["pyisetcam.scene"],
    "scene/reflectance": ["pyisetcam.scene", "pyisetcam.fileio"],
    "scripts/color": ["pyisetcam.color", "pyisetcam.metrics", "pyisetcam.scene"],
    "scripts/display": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    "scripts/human": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "scripts/image": ["pyisetcam.ip", "pyisetcam.fileio"],
    "scripts/metrics": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "scripts/optics": ["pyisetcam.optics"],
    "scripts/scene": ["pyisetcam.scene"],
    "scripts/sensor": ["pyisetcam.sensor", "pyisetcam.camera"],
    "scripts/utility": ["pyisetcam.fileio", "pyisetcam.plotting", "pyisetcam.session"],
    "sensor": ["pyisetcam.sensor"],
    "sensor/binning": ["pyisetcam.sensor"],
    "sensor/cfadesign": ["pyisetcam.sensor"],
    "sensor/microlens": ["pyisetcam.sensor"],
    "sensor/models": ["pyisetcam.sensor"],
    "sensor/pixel": ["pyisetcam.sensor"],
    "sensor/simulation": ["pyisetcam.sensor"],
    "tutorials/camera": ["pyisetcam.camera"],
    "tutorials/color": ["pyisetcam.color", "pyisetcam.metrics"],
    "tutorials/display": ["pyisetcam.display"],
    "tutorials/image": ["pyisetcam.ip", "pyisetcam.fileio"],
    "tutorials/metrics": ["pyisetcam.metrics", "pyisetcam.scielab"],
    "tutorials/oi": ["pyisetcam.optics"],
    "tutorials/optics": ["pyisetcam.optics"],
    "tutorials/scene": ["pyisetcam.scene"],
    "tutorials/sensor": ["pyisetcam.sensor"],
    "utility": ["pyisetcam.utils"],
    "utility/file": ["pyisetcam.fileio"],
    "utility/image": ["pyisetcam.utils", "pyisetcam.fileio"],
    "utility/plots": ["pyisetcam.plotting"],
    "utility/tablebase": ["pyisetcam.ptable"],
    "iset.m": ["pyisetcam.session"],
    "isetpath.m": ["pyisetcam.session"],
    "isetrootpath.m": ["pyisetcam.session"],
}

FAMILY_TEST_MAP = {
    "camera": ["tests/unit/test_pipeline.py"],
    "color": ["tests/unit/test_pipeline.py", "tests/unit/test_metrics.py"],
    "data/human": ["tests/unit/test_human.py", "tests/unit/test_pipeline.py", "tests/unit/test_ip.py"],
    "data/displays": ["tests/unit/test_display.py", "tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "data/fonts": ["tests/unit/test_fonts.py"],
    "data/images": ["tests/unit/test_pipeline.py", "tests/unit/test_scene.py", "tests/unit/test_fileio.py", "tests/parity/test_parity_harness.py"],
    "data/lens": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "data/lights": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "data/optics": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "data/safetystandards": ["tests/unit/test_human.py"],
    "data/sensor": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "data/surfaces": ["tests/unit/test_pipeline.py", "tests/unit/test_scene.py", "tests/unit/test_ip.py", "tests/parity/test_parity_harness.py"],
    "displays": ["tests/unit/test_display.py", "tests/unit/test_pipeline.py"],
    "fonts": ["tests/unit/test_fonts.py"],
    "human": ["tests/unit/test_human.py"],
    "imgproc": ["tests/unit/test_pipeline.py", "tests/unit/test_ip.py"],
    "main": ["tests/unit/test_session.py"],
    "metrics": ["tests/unit/test_metrics.py", "tests/unit/test_pipeline.py"],
    "opticalimage": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "scene": [
        "tests/unit/test_scene.py",
        "tests/unit/test_pipeline.py",
        "tests/parity/test_parity_harness.py",
    ],
    "scripts": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "sensor": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "tutorials": ["tests/unit/test_pipeline.py", "tests/parity/test_parity_harness.py"],
    "utility/file": ["tests/unit/test_fileio.py"],
    "utility/plots": ["tests/unit/test_plotting.py"],
    "utility/tablebase": ["tests/unit/test_ptable.py"],
    "iset.m": ["tests/unit/test_session.py"],
    "isetpath.m": ["tests/unit/test_session.py"],
    "isetrootpath.m": ["tests/unit/test_session.py"],
}

GUI_KEYWORDS = ("plot", "figure", "window", "handle", "refresh", "show", "app")
HEADLESS_KEYWORDS = GUI_KEYWORDS + ("session", "roi", "vc", "ie")
CORE_PRIORITY_FAMILIES = {
    "camera",
    "color",
    "displays",
    "imgproc",
    "metrics",
    "opticalimage",
    "scene",
    "sensor",
    "scripts/metrics",
    "scripts/optics",
    "scripts/scene",
    "scripts/sensor",
    "tutorials/metrics",
    "tutorials/oi",
    "tutorials/optics",
    "tutorials/scene",
    "tutorials/sensor",
}
MID_PRIORITY_FAMILIES = {
    "human",
    "scripts/display",
    "scripts/human",
    "scripts/image",
    "scripts/utility",
    "tutorials/camera",
    "tutorials/color",
    "tutorials/display",
    "tutorials/image",
    "utility",
    "utility/file",
    "utility/image",
    "utility/plots",
    "utility/tablebase",
}

UPSTREAM_STATUS_OVERRIDES: dict[str, dict[str, Any]] = {
    "data/human/": {
        "status": "parity",
        "note": "The vendored human-data asset bundle is already exercised by the current headless runtime through `AssetStore`, spectral readers, and unit coverage over XYZ or XYZQuanta, luminosity, rods, Stockman cone fundamentals, and macular-pigment assets.",
        "module_hits": [
            "pyisetcam.assets",
            "pyisetcam.color",
            "pyisetcam.display",
            "pyisetcam.camera",
            "pyisetcam.sensor",
            "pyisetcam.optics",
        ],
    },
    "data/human/absoluteEfficiency.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-authoring notebook that derives archived absolute-efficiency tables from vendored Stockman fundamentals, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/human/conemosaic/convertFiles2M.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-conversion notebook that imports external Hofer or Williams spreadsheets and text PSFs into archived cone-mosaic MAT assets, rather than a supported headless runtime API surface.",
        "module_hits": [],
    },
    "data/human/conemosaic/s_HoferPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB exploratory cone-mosaic plotting script rather than a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/human/cones/innerSegmentDTutten.fig": {
        "status": "out_of_scope",
        "note": "The upstream file is a saved MATLAB figure artifact for cone inner-segment visualization, not a runtime asset or supported headless API surface.",
        "module_hits": [],
    },
    "data/human/ieRodSpectralSensitivity.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-generation script that derives the vendored rod-sensitivity MAT asset already consumed by the current runtime, rather than a supported headless API surface.",
        "module_hits": [],
    },
    "data/human/lensDensity.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-authoring helper that writes the vendored lens-density MAT asset, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/human/luminosityJuddCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-generation script for luminosity or Judd luminosity tables already vendored in the runtime bundle, rather than a supported headless API surface.",
        "module_hits": [],
    },
    "data/human/macular.m": {
        "status": "parity",
        "note": "The current headless runtime already covers the upstream macular-pigment math through `_macular_profile(...)` and `humanMacularTransmittance(...)`, both backed by the same vendored `macularPigment.mat` asset and focused unit regression coverage.",
        "module_hits": ["pyisetcam.optics"],
    },
    "data/human/macularPigmentCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-authoring script that embeds the macular-pigment source table and writes the vendored `macularPigment.mat` asset already used by the runtime.",
        "module_hits": [],
    },
    "data/human/melanopsin/s_melanopsinCIE.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-preparation notebook that converts CIE melanopsin source tables into archived MAT assets, not a supported headless runtime API surface.",
        "module_hits": [],
    },
    "data/human/stockmanQuantaCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-generation script that derives the vendored `stockmanQuanta.mat` cone-fundamental table already consumed by the current runtime.",
        "module_hits": [],
    },
    "data/human/xyzQuantaCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-generation script that derives vendored XYZ or luminosity quanta tables already consumed by the current runtime.",
        "module_hits": [],
    },
    "data/fluorescence/": {
        "status": "out_of_scope",
        "note": "The pinned upstream snapshot only vendors fluorophore MAT assets, while the corresponding fluorescence-model helpers such as `fluorescenceSignal` and `fluorescenceWeights` are not part of the supported ISETCam migration target; the bundle is therefore tracked as external fluorescence-model data rather than active headless debt.",
        "module_hits": [],
    },
    "data/displays/": {
        "status": "parity",
        "note": "The vendored display bundle is already exercised by the current headless runtime through `displayCreate(...)`, scene and IP display-backed workflows, and focused unit or parity coverage over CRT, LCD, OLED, and reflectance display assets.",
        "module_hits": ["pyisetcam.assets", "pyisetcam.display", "pyisetcam.scene", "pyisetcam.parity"],
    },
    "data/displays/ieBarcoSign.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB authoring notebook that derives the vendored `LED-BarcoC8.mat` display from an existing OLED calibration, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/displays/render_lcd_samsung_rgbw.m": {
        "status": "ported",
        "note": "The upstream display render helper now has a direct headless wrapper in `pyisetcam.display.render_lcd_samsung_rgbw(...)`, with focused regression coverage over the RGBW white-extraction contract and normalized render-function metadata from the vendored display MAT files.",
        "module_hits": ["pyisetcam.display"],
    },
    "data/displays/render_oled_samsung.m": {
        "status": "ported",
        "note": "The upstream display render helper now has a direct headless wrapper in `pyisetcam.display.render_oled_samsung(...)`, with focused regression coverage over the MATLAB control-map replay semantics and normalized render-function metadata from the vendored display MAT files.",
        "module_hits": ["pyisetcam.display"],
    },
    "data/fonts/": {
        "status": "parity",
        "note": "The vendored font bitmap cache is now exercised directly by the headless `fontBitmapGet(...)`, `fontCreate(...)`, `fontSet(...)`, and `sceneFromFont(...)` compatibility surface, with focused unit coverage over the cached Georgia glyph payloads and derived scene geometry.",
        "module_hits": ["pyisetcam.fonts"],
    },
    "data/images/": {
        "status": "parity",
        "note": "The vendored image bundle is already exercised broadly by the current headless runtime through `sceneFromFile(...)`, raw-image and multispectral import paths, camera scene loading, and focused unit or parity coverage over RGB, raw DNG, and multispectral scene assets such as StuffedAnimals, Feng Office, eagle, faceMale, and MCC-centered.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.fileio", "pyisetcam.camera", "pyisetcam.parity"],
    },
    "data/images/faces/s_dataFaces.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB download and visualization notebook for spectral face assets fetched through `ieWebGet`, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/microlens/": {
        "status": "out_of_scope",
        "note": "The pinned upstream snapshot only vendors standalone microlens design JSON presets here, while the supported headless migration target uses the current analytic `mlens_create(...)` and microlens-radiance workflow instead of importing these external preset files.",
        "module_hits": [],
    },
    "data/scenes/": {
        "status": "out_of_scope",
        "note": "The pinned upstream snapshot does not vendor any reusable scene assets in this directory; it only preserves a placeholder folder for externally hosted scene collections, so there is no headless runtime bundle to migrate here.",
        "module_hits": [],
    },
    "data/scenes/d_sceneICVL.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB local-path or web-ingest notebook for the external ICVL hyperspectral repository, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/surfaces/": {
        "status": "parity",
        "note": "The vendored surface-reflectance bundle is already exercised by the current headless runtime through `surfaceReflectance(...)`, `sceneCreate(...)`, chart and skin-reflectance loaders, Esser-chart paths, and focused unit or parity coverage over Munsell, Food, skin, and Esser assets.",
        "module_hits": ["pyisetcam.assets", "pyisetcam.scene", "pyisetcam.color", "pyisetcam.ip", "pyisetcam.parity"],
    },
    "data/surfaces/charts/esser/esserReflectance.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB provenance notebook that derives the vendored Esser chart reflectance table from archived radiance captures, rather than a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/surfaces/reflectances/skin/absorbances/d_hemoglobin.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-authoring notebook that converts external hemoglobin extinction tables into vendored absorbance assets already consumed by the current runtime.",
        "module_hits": [],
    },
    "data/safetyStandards/": {
        "status": "parity",
        "note": "The vendored safety-standard spectra are already exercised by the current headless runtime through `humanUVSafety(...)`, which reads the Actinic and blue-light hazard curves directly and is covered by focused human-safety regressions.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.assets"],
    },
    "data/safetyStandards/s_safetyStandards.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-authoring notebook that transcribes EN62471 safety-standard tables into vendored MAT assets already consumed by the current runtime, rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "data/validation/": {
        "status": "out_of_scope",
        "note": "The pinned upstream snapshot only vendors archived MATLAB validation payloads in this directory, such as `v_data_sceneFromRGB.mat`, which are reference artifacts rather than runtime assets or supported headless APIs.",
        "module_hits": [],
    },
    "data/lights/": {
        "status": "parity",
        "note": "The vendored illuminant-data bundle is already exercised by the current headless runtime through `AssetStore`, daylight generation, illuminant creation, scene illuminant adjustment, and focused unit or parity coverage over CIE daylight basis data, Gretag illuminants, and named daylight or horizon spectra.",
        "module_hits": [
            "pyisetcam.assets",
            "pyisetcam.color",
            "pyisetcam.illuminant",
            "pyisetcam.scene",
            "pyisetcam.parity",
        ],
    },
    "data/lights/daylight/d_daylightBasis.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-management notebook that imports or compares CIE daylight basis tables before rewriting the vendored `cieDaylightBasis.mat` asset already consumed by the current runtime.",
        "module_hits": [],
    },
    "data/lights/daylight/d_daylightStanford.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB daylight-database preparation notebook for the external Stanford DiCarlo daylight corpus rather than a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/lights/daylight/d_granada.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-conversion notebook that imports Granada daylight measurements into a vendored MAT asset, not a supported headless runtime API surface.",
        "module_hits": [],
    },
    "data/lights/daylight/d_rochester.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-conversion notebook that imports Rochester daylight spreadsheet data into a vendored MAT asset, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/lights/gretag/s_dataGretag.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB illuminant-inspection notebook that plots or compares already-vendored Gretag lightbox spectra rather than providing a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/lights/s_dataLamps.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a large MATLAB asset-authoring notebook that pastes external lamp spectra into vendored MAT files, not a supported headless runtime API surface.",
        "module_hits": [],
    },
    "data/lights/solar/d_solarFraunhofer.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB solar-spectrum preparation notebook that imports CSV source data and annotates Fraunhofer lines before writing the vendored solar asset bundle, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/lens/": {
        "status": "parity",
        "note": "The vendored lens-data bundle is already exercised by the current headless runtime through `lensList(...)`, the ray-trace optics import and filename-normalization path, and focused unit or parity coverage over pinned upstream lens JSON descriptors.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.parity"],
    },
    "data/lens/s_closestFocalDistance.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB lens-analysis notebook that scans precomputed focus tables and plots the closest focal distance across vendored lens files, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/lens/s_focusLensTable.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-generation notebook that precomputes legacy `.FL.mat` focus tables from vendored lens JSON descriptors, and its own header marks the workflow as obsolete in favor of on-the-fly focus calculation.",
        "module_hits": [],
    },
    "data/lens/s_lensUpdateFormat.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB metadata-maintenance script that rewrites vendored lens JSON descriptors in place, rather than a supported headless runtime API surface.",
        "module_hits": [],
    },
    "data/optics/": {
        "status": "parity",
        "note": "The vendored optics-data bundle is already exercised by the current headless runtime through `AssetStore`, ray-trace optics loaders, flare assets, and the Thibos virtual-eye path covered by `wvfLoadThibosVirtualEyes(...)`, `ieMvnrnd(...)`, and focused unit or parity regressions.",
        "module_hits": ["pyisetcam.assets", "pyisetcam.optics", "pyisetcam.utils", "pyisetcam.parity"],
    },
    "data/optics/thibosvirtualeyes/VirtualEyes.m": {
        "status": "parity",
        "note": "The upstream Thibos virtual-eye sampler is already covered numerically by `wvfLoadThibosVirtualEyes(...)` together with `ieMvnrnd(...)`, plus focused unit and parity coverage over the same vendored IAS statistics tables and multivariate-normal sampling contract.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.utils", "pyisetcam.parity"],
    },
    "data/optics/thibosvirtualeyes/VirtualEyesDemo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB demonstration script for plotting means and confidence ranges of sampled Thibos virtual eyes, not a reusable headless runtime API surface.",
        "module_hits": [],
    },
    "data/sensor/": {
        "status": "parity",
        "note": "The vendored sensor asset bundle is already exercised by the current headless runtime through `AssetStore`, sensor-model loaders, and parity or unit regressions over MT9V024, AR0132AT, IMX490, IMEC, IR-filter, and MCC sensor assets.",
        "module_hits": ["pyisetcam.assets", "pyisetcam.sensor", "pyisetcam.camera", "pyisetcam.parity"],
    },
    "data/sensor/auto/MT9V024Create.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB authoring script that derives vendored MT9V024 filter and sensor MAT assets already consumed by the current runtime; recreating the provenance notebook is outside the headless migration target.",
        "module_hits": [],
    },
    "data/sensor/auto/ar0132atCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB authoring script that derives vendored AR0132AT filter and sensor MAT assets already consumed by the current runtime; recreating the provenance notebook is outside the headless migration target.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/CMYGCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB color-filter authoring script for vendored sensor-filter assets, not a runtime API required by the headless migration target.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/OVT/s_ovtColorfilters.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB data-preparation notebook for OVT filter assets already vendored in the runtime bundle, rather than a supported runtime API surface.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/auto/SONY/s_imx490QEdata.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB QE-data preparation script for the vendored IMX490 filter assets already exercised by the current headless runtime.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/fourChannelCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB color-filter authoring script for vendored sensor-filter assets, not a runtime API required by the headless migration target.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/gaussianCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB filter-generation notebook for archived sensor-filter assets, not a required runtime API surface.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/s_radiometerCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB authoring script for vendored radiometer-filter assets rather than a runtime API required by the headless port.",
        "module_hits": [],
    },
    "data/sensor/colorfilters/sixChannelCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB color-filter authoring script for vendored sensor-filter assets, not a runtime API required by the headless migration target.",
        "module_hits": [],
    },
    "data/sensor/imec/cornell_thomas.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an IMEC asset-analysis notebook rather than a reusable runtime API surface for the headless migration target.",
        "module_hits": [],
    },
    "data/sensor/imec/generateImecQEfile.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB authoring script that derives the vendored IMEC QE asset already consumed by the current runtime.",
        "module_hits": [],
    },
    "data/sensor/imec/multispectral_pbrt.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an IMEC asset-generation notebook rather than a reusable runtime API required by the headless port.",
        "module_hits": [],
    },
    "data/sensor/imec/s_imecSensorTestScene.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB exploratory script for IMEC scene preparation, not a supported runtime API surface.",
        "module_hits": [],
    },
    "data/sensor/imec/v_imec.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB visualization notebook for IMEC sensor assets rather than a reusable runtime API surface.",
        "module_hits": [],
    },
    "data/sensor/irfilters/irPassFilterSave.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB authoring script that generates a vendored IR-pass filter asset already consumed by the current runtime.",
        "module_hits": [],
    },
    "data/sensor/mccGBRGSensorData.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB asset-conversion script that produces the vendored GBRG MCC TIFF used by existing parity coverage, rather than a runtime API surface.",
        "module_hits": [],
    },
    "web/dngImport.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly labeled a scratch DNG-import notebook and redirects callers to `ieDNGRead`-style helpers rather than exposing a maintained web/runtime API surface.",
        "module_hits": ["pyisetcam.fileio"],
    },
    "web/webCreateThumbnails.m": {
        "status": "ported",
        "note": "The multispectral thumbnail helper is covered by the Python `webCreateThumbnails(...)` / `web_create_thumbnails(...)` wrapper, which replays `sceneFromFile(...)` plus `sceneSaveImage(...)` headlessly for local MAT scenes.",
        "module_hits": ["pyisetcam.web", "pyisetcam.__init__"],
    },
    "web/webData.m": {
        "status": "ported",
        "note": "The JSON-backed scene-catalog helper is covered by the Python `webData` compatibility wrapper, including keyword search, thumbnail fetch, remote MAT download, and `sceneFromFile(...)` replay for the vendored catalog entries.",
        "module_hits": ["pyisetcam.web", "pyisetcam.__init__"],
    },
    "web/webFlickr.m": {
        "status": "ported",
        "note": "The Flickr search helper is covered by the Python `webFlickr` compatibility wrapper, including search, image URL construction, image fetch, and display-scene replay through `sceneFromFile(...)`.",
        "module_hits": ["pyisetcam.web", "pyisetcam.__init__"],
    },
    "web/webLOC.m": {
        "status": "ported",
        "note": "The Library of Congress search helper is covered by the Python `webLOC` compatibility wrapper, including result filtering, URL normalization, image fetch, and display-scene replay through `sceneFromFile(...)`.",
        "module_hits": ["pyisetcam.web", "pyisetcam.__init__"],
    },
    "web/webPixabay.m": {
        "status": "ported",
        "note": "The Pixabay search helper is covered by the Python `webPixabay` compatibility wrapper, including search, preview or large-image URL access, image fetch, and display-scene replay through `sceneFromFile(...)`.",
        "module_hits": ["pyisetcam.web", "pyisetcam.__init__"],
    },
    "cp/cpBurstCamera.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a prototype computational-camera orchestrator for multi-frame HDR, burst, focus-stack, and video capture on top of the external ISET3d/PBRT `cpScene` rendering path rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "cp/cpBurstIP.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an experimental multi-frame computational-photography IP prototype tied to the external `cp` burst stack, including HDR registration, `makehdr`, and focus-stack behavior rather than a supported standalone pyisetcam surface.",
        "module_hits": [],
    },
    "cp/cpCModule.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a PBRT/ISET3d computational-camera module wrapper that couples sensors, optics, scene rendering, and focus-stack defocus replay rather than a supported standalone pyisetcam runtime API.",
        "module_hits": [],
    },
    "cp/cpCamera.m": {
        "status": "out_of_scope",
        "note": "The upstream file is the base class for the experimental PBRT/ISET3d computational-camera prototype, not a supported standalone ISETCam headless API surface for pyisetcam.",
        "module_hits": [],
    },
    "cp/cpIP.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a thin wrapper around MATLAB `ipCompute(...)` for the experimental `cp` prototype, including multi-sensor merge semantics specific to the external computational-camera stack rather than a supported standalone pyisetcam API.",
        "module_hits": [],
    },
    "cp/cpScene.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an ISET3d/PBRT scene-orchestration layer built around `piRecipe`, Docker rendering, camera or object motion, and external recipe assets rather than a supported standalone pyisetcam scene API surface.",
        "module_hits": [],
    },
    "cp/s_cpAssembleScene.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a PBRT asset-assembly demo built around `piRecipeMerge`, `piAssetLoad`, and interactive preview windows rather than a reusable supported pyisetcam workflow.",
        "module_hits": [],
    },
    "cp/s_cpImageRegistration.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an exploratory image-registration notebook generated from MATLAB Registration Estimator and depends on the Computer Vision Toolbox rather than a supported standalone pyisetcam workflow.",
        "module_hits": [],
    },
    "cp/s_demoVideo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GPU or PBRT demo-video notebook built around ISET3d rendering, camera-motion recipes, and MATLAB `VideoWriter` rather than a supported standalone pyisetcam workflow.",
        "module_hits": [],
    },
    "cp/utilities/cpCompareImages.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a simple MATLAB figure helper for side-by-side image display, built around `imshowpair` and text overlays rather than a reusable supported headless pyisetcam API surface.",
        "module_hits": [],
    },
    "utility/file/ieImageType.m": {
        "status": "ported",
        "note": "The legacy MATLAB image-type probe is covered by the Python `ie_image_type(...)` / `ieImageType(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieDNGRead.m": {
        "status": "ported",
        "note": "The legacy MATLAB DNG reader is covered by the Python `ie_dng_read(...)` / `ieDNGRead(...)` compatibility wrapper for raw, RGB, and metadata-only replay.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieDNGSimpleInfo.m": {
        "status": "ported",
        "note": "The reduced DNG metadata helper is covered by the Python `ie_dng_simple_info(...)` / `ieDNGSimpleInfo(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieReadColorFilter.m": {
        "status": "ported",
        "note": "The legacy MATLAB color-filter reader is covered by the Python `ie_read_color_filter(...)` / `ieReadColorFilter(...)` compatibility wrapper on top of the current asset pipeline.",
        "module_hits": ["pyisetcam.assets", "pyisetcam.__init__"],
    },
    "utility/file/ieReadMultipleFileNames.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an interactive multi-file chooser built around MATLAB directory/list dialogs rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "utility/file/ieReadSpectra.m": {
        "status": "ported",
        "note": "The legacy MATLAB spectral-reader helper is covered by the Python `ie_read_spectra(...)` / `ieReadSpectra(...)` compatibility wrapper on top of the current asset pipeline.",
        "module_hits": ["pyisetcam.assets", "pyisetcam.__init__"],
    },
    "utility/file/ieSCP.m": {
        "status": "ported",
        "note": "The remote-copy helper is covered by the Python `ie_scp(...)` / `ieSCP(...)` compatibility wrapper over a headless recursive `scp` call.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieSaveColorFilter.m": {
        "status": "ported",
        "note": "The legacy MATLAB color-filter writer is covered by the Python `ie_save_color_filter(...)` / `ieSaveColorFilter(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieSaveMultiSpectralImage.m": {
        "status": "ported",
        "note": "The basis-coded multispectral-image writer is covered by the Python `ie_save_multispectral_image(...)` / `ieSaveMultiSpectralImage(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieSaveSIDataFile.m": {
        "status": "ported",
        "note": "The shift-invariant PSF-data writer is covered by the Python `ie_save_si_data_file(...)` / `ieSaveSIDataFile(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieSaveSpectralFile.m": {
        "status": "ported",
        "note": "The legacy MATLAB spectral-data writer is covered by the Python `ie_save_spectral_file(...)` / `ieSaveSpectralFile(...)` compatibility wrapper for headless uncompressed spectral MAT payloads.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieTempfile.m": {
        "status": "ported",
        "note": "The legacy MATLAB temp-file helper is covered by the Python `ie_tempfile(...)` / `ieTempfile(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieVarInFile.m": {
        "status": "ported",
        "note": "The legacy MATLAB MAT-variable membership helper is covered by the Python `ie_var_in_file(...)` / `ieVarInFile(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieWebGet.m": {
        "status": "ported",
        "note": "The legacy MATLAB Stanford-repository download helper is covered headlessly by the Python `ie_web_get(...)` / `ieWebGet(...)` compatibility wrapper, including list/browse metadata plus local download/unzip behavior without opening a browser.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/ieXL2ColorFilter.m": {
        "status": "ported",
        "note": "The legacy MATLAB spreadsheet-to-filter converter is covered by the Python `ie_xl2_color_filter(...)` / `ieXL2ColorFilter(...)` compatibility wrapper for headless CSV/XLSX spectral and color-filter payloads.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/pathToLinux.m": {
        "status": "ported",
        "note": "The Windows-to-Linux path-normalization helper is covered by the Python `path_to_linux(...)` / `pathToLinux(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcImportObject.m": {
        "status": "ported",
        "note": "The legacy MATLAB object-import helper is covered for the headless core object types by the Python `vc_import_object(...)` / `vcImportObject(...)` compatibility wrapper on top of `vc_load_object(...)`.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcExportObject.m": {
        "status": "ported",
        "note": "The legacy MATLAB object-export helper is covered by the Python `vc_export_object(...)` / `vcExportObject(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcLoadObject.m": {
        "status": "ported",
        "note": "The legacy MATLAB object-loader helper is covered by the Python `vc_load_object(...)` / `vcLoadObject(...)` compatibility wrapper for the core ISET object types.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcReadSpectra.m": {
        "status": "ported",
        "note": "The deprecated MATLAB spectral-reader alias is covered by the Python `vc_read_spectra(...)` / `vcReadSpectra(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcReadImage.m": {
        "status": "ported",
        "note": "The legacy MATLAB image-reader helper is covered by the Python `vc_read_image(...)` / `vcReadImage(...)` compatibility wrapper on top of the current `scene_from_file(...)` multispectral and emissive-display paths.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcSaveMultiSpectralImage.m": {
        "status": "ported",
        "note": "The legacy MATLAB multispectral-image writer is covered by the Python `vc_save_multispectral_image(...)` / `vcSaveMultiSpectralImage(...)` compatibility wrapper on top of the existing basis-coded save path.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/file/vcSaveObject.m": {
        "status": "ported",
        "note": "The legacy MATLAB object-save helper is covered by the Python `vc_save_object(...)` / `vcSaveObject(...)` compatibility wrapper for the core ISET object types.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/publish/s_publishScripts.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB batch publishing notebook that runs `publish(...)` over script directories to generate local HTML or PDF artifacts, not a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/publish/s_publishTutorials.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB batch publishing notebook that runs `publish(...)` over tutorial directories to generate local HTML or PDF artifacts, not a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/video/ieMovie.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a generic MATLAB figure or video writer for array movies, built around `axes`, `imagesc`, `drawnow`, `getframe`, and `VideoWriter`, while the supported pyisetcam runtime already handles object-pipeline animation through higher-level headless helpers such as `scene_make_video(...)` and `oi_preview_video(...)`.",
        "module_hits": [],
    },
    "utility/ieFindFiles.m": {
        "status": "ported",
        "note": "The recursive file-finder helper is covered by the Python `ie_find_files(...)` / `ieFindFiles(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/ieInit.m": {
        "status": "ported",
        "note": "The legacy MATLAB startup script is covered by the Python `ie_init(...)` / `ieInit(...)` compatibility wrapper on top of the current headless session bootstrap, including `ieMainClose(...)` replay, fresh session creation, and default hidden-main-window behavior.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/ieRadiance2IP.m": {
        "status": "ported",
        "note": "The scene-or-OI convenience wrapper is covered by the Python `ie_radiance_to_ip(...)` / `ieRadiance2IP(...)` compatibility surface in `pyisetcam.ip`, which replays the headless OI -> sensor -> IP path with default auto exposure, default sensor tuning, and sensor-metadata carry-through.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__"],
    },
    "utility/ieManualViewer.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a browser-launch helper for GitHub wiki and source-tree pages, which is documentation navigation rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/ieTikhonovRidge.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an exploratory MATLAB ridge-regression notebook with plotting and handwritten examples rather than a reusable runtime API; the supported headless solver surface is `ie_tikhonov(...)` / `ieTikhonov(...)`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/ieTone.m": {
        "status": "ported",
        "note": "The MATLAB tone helper is covered headlessly by the Python `ie_tone(...)` / `ieTone(...)` compatibility wrapper, which replays the waveform synthesis and parameter payload without depending on MATLAB audio playback.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/ieUncompressData.m": {
        "status": "ported",
        "note": "The inverse quantization helper is covered by the Python `ie_uncompress_data(...)` / `ieUncompressData(...)` compatibility wrapper in `pyisetcam.utils`, matching the `ieCompressData(...)` inversion formula for scene or OI payloads.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/JSONio/jsonread.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only the MATLAB help stub for the external compiled `jsonread.c` MEX parser; the supported pyisetcam runtime relies on Python's built-in JSON tooling instead of vendoring MATLAB's third-party JSONio integration layer.",
        "module_hits": [],
    },
    "utility/external/JSONio/jsonwrite.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a generic third-party JSON serializer from JSONio rather than an ISET-specific runtime surface; the supported headless runtime already uses Python's built-in JSON support where JSON serialization is needed.",
        "module_hits": [],
    },
    "utility/external/JSONio/test_jsonread.m": {
        "status": "out_of_scope",
        "note": "The upstream file is the vendored JSONio package's own MATLAB unit test suite, which is package-maintenance coverage for a third-party dependency rather than a pyisetcam runtime target.",
        "module_hits": [],
    },
    "utility/external/JSONio/test_jsonwrite.m": {
        "status": "out_of_scope",
        "note": "The upstream file is the vendored JSONio package's own MATLAB unit test suite, which is package-maintenance coverage for a third-party dependency rather than a pyisetcam runtime target.",
        "module_hits": [],
    },
    "utility/external/dcraw/dcrawInfo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a thin MATLAB wrapper around an external platform-specific `dcraw` executable; the supported pyisetcam runtime covers the checked-in DNG workflows through `ie_dng_read(...)` and `sensor_dng_read(...)` instead of vendoring generic host-binary raw decoders.",
        "module_hits": [],
    },
    "utility/external/dcraw/dcrawInit.m": {
        "status": "out_of_scope",
        "note": "The upstream file registers external `dcraw` executables into MATLAB `imformats`, which is host-environment integration for a third-party binary rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/dcraw/dcrawRead.m": {
        "status": "out_of_scope",
        "note": "The upstream file shells out to an external `dcraw` executable to decode arbitrary camera RAW files, while the supported pyisetcam runtime only maintains direct headless coverage for the vendored DNG path through `ie_dng_read(...)` and `sensor_dng_read(...)`.",
        "module_hits": [],
    },
    "utility/external/dcraw/t_dcraw.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB demonstration notebook for the external `dcraw` integration, not a supported pyisetcam runtime API surface.",
        "module_hits": [],
    },
    "utility/external/AddTextToImage/AddTextToImage.m": {
        "status": "ported",
        "note": "The upstream text-overlay helper is covered headlessly by the Python `add_text_to_image(...)` / `AddTextToImage(...)` compatibility wrapper in `pyisetcam.fonts`, including zero-based placement, grayscale-or-RGB color replay, and cropped overlay semantics.",
        "module_hits": ["pyisetcam.fonts", "pyisetcam.__init__"],
    },
    "utility/external/AddTextToImage/AddTextToImageDemo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB demonstration notebook for the AddTextToImage package, not a supported pyisetcam runtime API surface.",
        "module_hits": [],
    },
    "utility/external/AddTextToImage/AddTextToImageWithBorder.m": {
        "status": "ported",
        "note": "The upstream bordered text-overlay helper is covered headlessly by the Python `add_text_to_image_with_border(...)` / `AddTextToImageWithBorder(...)` compatibility wrapper in `pyisetcam.fonts`, including circular outline growth and per-channel border-color replay.",
        "module_hits": ["pyisetcam.fonts", "pyisetcam.__init__"],
    },
    "utility/external/AddTextToImage/AddTextToImageWithBorderDemo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB demonstration notebook for the bordered AddTextToImage workflow, not a supported pyisetcam runtime API surface.",
        "module_hits": [],
    },
    "utility/external/AddTextToImage/BitmapFont.m": {
        "status": "ported",
        "note": "The upstream bitmap-font builder is covered by the Python `bitmap_font(...)` / `BitmapFont(...)` compatibility wrapper in `pyisetcam.fonts`, which rasterizes per-character boolean glyph masks without depending on MATLAB figures or screenshots.",
        "module_hits": ["pyisetcam.fonts", "pyisetcam.__init__"],
    },
    "utility/external/AddTextToImage/RasterizeText.m": {
        "status": "ported",
        "note": "The upstream rasterization helper is covered by the Python `rasterize_text(...)` / `RasterizeText(...)` compatibility wrapper in `pyisetcam.fonts`, including control-character cleanup, space-and-newline layout, and missing-character validation against bitmap-font payloads.",
        "module_hits": ["pyisetcam.fonts", "pyisetcam.__init__"],
    },
    "utility/external/ImageConvFrequencyDomain.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a generic standalone FFT-convolution utility from an external source rather than an ISETCam runtime API surface; the supported headless runtime already uses native NumPy or SciPy convolution primitives where needed instead of vendoring this helper as public compatibility debt.",
        "module_hits": [],
    },
    "utility/external/arrow3/arrow3.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a third-party MATLAB 2D or 3D arrow plotting helper for figures, so it is treated as visualization-only external baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/bluewhitered/bluewhitered.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a third-party diverging colormap helper for MATLAB figures, so it is treated as visualization-only external baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/cprintf.m": {
        "status": "out_of_scope",
        "note": "The upstream file only colors text in the MATLAB command window, which is host-console UI behavior rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/cpuinfo/cpuinfo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a host-machine CPU probe for MATLAB environments, which is deployment or diagnostics baggage rather than a supported pyisetcam runtime API surface.",
        "module_hits": [],
    },
    "utility/external/DataHash_20190519/DataHash.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a vendored third-party generic hashing implementation retained as provenance for the canonical ISET `ieHash` helper, so it is tracked as external package baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/DataHash_20190519/uTest_DataHash.m": {
        "status": "out_of_scope",
        "note": "The upstream file is the vendored DataHash package's own MATLAB unit-test suite, which is third-party package maintenance coverage rather than a pyisetcam runtime target.",
        "module_hits": [],
    },
    "utility/external/Inpaint_nans/doc/methods_of_inpaint_nans.m": {
        "status": "out_of_scope",
        "note": "The upstream file is the third-party Inpaint_nans package's own methods documentation notebook, which is external package prose rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/Inpaint_nans/inpaint_nans.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a vendored third-party NaN inpainting solver that is not referenced by the supported pyisetcam runtime, so it is tracked as external numerical baggage rather than compatibility debt.",
        "module_hits": [],
    },
    "utility/external/Inpaint_nans/inpaint_nans_bc.m": {
        "status": "out_of_scope",
        "note": "The upstream file is the boundary-condition variant of the third-party Inpaint_nans solver and is treated the same way as external numerical baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/comp_struct/comp_struct.m": {
        "status": "ported",
        "note": "The upstream nested-struct comparison helper is covered by the Python `comp_struct(...)` / `compStruct(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over nested common-field extraction, tolerance-based numeric equality, and mismatch extraction.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/comp_struct/comp_struct_test.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a manual MATLAB exploratory script that opens figures and prompts through menu dialogs while exercising the comp_struct package, so it is tracked as interactive maintenance coverage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/comp_struct/list_struct.m": {
        "status": "ported",
        "note": "The upstream recursive struct-listing helper is covered by the Python `list_struct(...)` / `listStruct(...)` compatibility wrapper in `pyisetcam.utils`, which returns headless field-path payloads instead of printing to a MATLAB console.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/exiftool/exiftoolDepthFromFile.m": {
        "status": "ported",
        "note": "The upstream Dynamic Depth Format extractor is covered by the Python `exiftool_depth_from_file(...)` / `exiftoolDepthFromFile(...)` compatibility wrapper in `pyisetcam.scene`, together with the existing `sceneFromDDFFile(...)` workflow and focused regression coverage over meter-scaled embedded depth payload replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/exiftool/exiftoolInfo.m": {
        "status": "ported",
        "note": "The upstream exiftool metadata helper is covered by the Python `exiftool_info(...)` / `exiftoolInfo(...)` compatibility wrapper in `pyisetcam.scene`, which replays the JSON or verbose text query contract against the host `exiftool` binary headlessly.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/build_pyramid.m": {
        "status": "ported",
        "note": "The upstream non-decimated pyramid dispatcher is covered by the Python `build_pyramid(...)` / `buildPyramid(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the supported Haar and QMF branches.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/buildFullSFpyr2_o.m": {
        "status": "out_of_scope",
        "note": "The upstream oversampled steerable-pyramid builder depends on external Simoncelli MatlabPyrTools routines such as `rcosFn`, `pointOp`, and `steer2HarmMtx`, and the vendored HDR README explicitly requires installing that toolbox for the `steerable` path, so this branch is treated as external-toolbox scope rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/hdr/buildSFpyrLevs_o.m": {
        "status": "out_of_scope",
        "note": "The upstream recursive steerable-pyramid level builder depends on the external Simoncelli MatlabPyrTools masking utilities referenced by the vendored HDR README, so it is treated as external-toolbox scope rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/hdr/final_touch.m": {
        "status": "ported",
        "note": "The upstream histogram-equalized HDR finishing helper is covered by the Python `final_touch(...)` / `finalTouch(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the vendored global equalization replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/getpfmraw.m": {
        "status": "ported",
        "note": "The upstream color PFM reader is covered by the Python `getpfmraw(...)` / `getPFMraw(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the vendored channel packing and vertical-flip replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/haar_pyramid.m": {
        "status": "ported",
        "note": "The upstream oversampled Haar-pyramid builder is covered by the Python `haar_pyramid(...)` / `haarPyramid(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over exact reconstruction against `recons_haar_pyramid(...)`.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/im_norm.m": {
        "status": "ported",
        "note": "The upstream min-max normalization helper is covered by the Python `im_norm(...)` / `imNorm(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over exact [0, 1] scaling.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/modulateFlip.m": {
        "status": "ported",
        "note": "The upstream QMF high-pass filter constructor is covered by the Python `modulate_flip_shift(...)` / `modulateFlip(...)` / `modulateFlipShift(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the sign-flipped reverse-order replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/pad_reflect.m": {
        "status": "ported",
        "note": "The upstream reflection-padding helper is covered by the Python `pad_reflect(...)` / `padReflect(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the vendored edge-reflection replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/pad_reflect_neg.m": {
        "status": "ported",
        "note": "The upstream signed reflection-padding helper is covered by the Python `pad_reflect_neg(...)` / `padReflectNeg(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the vendored horizontal and vertical sign-flip replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/qmf_pyramid.m": {
        "status": "ported",
        "note": "The upstream oversampled QMF-pyramid builder is covered by the Python `qmf_pyramid(...)` / `qmfPyramid(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over exact reconstruction against `recons_qmf_pyramid(...)`.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/range_compression_lum.m": {
        "status": "ported",
        "note": "The upstream grayscale HDR range-compression helper is covered by the Python `range_compression_lum(...)` / `rangeCompressionLum(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the current headless Haar path.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/reconFullSFpyr2_o.m": {
        "status": "out_of_scope",
        "note": "The upstream oversampled steerable-pyramid reconstructor depends on external Simoncelli MatlabPyrTools routines such as `rcosFn`, `pointOp`, and `pyrBand`, and the vendored HDR README explicitly requires installing that toolbox for the `steerable` path, so this branch is treated as external-toolbox scope rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/hdr/reconSFpyrLevs_o.m": {
        "status": "out_of_scope",
        "note": "The upstream recursive steerable-pyramid reconstruction helper depends on external Simoncelli MatlabPyrTools masking and band-access utilities referenced by the vendored HDR README, so it is treated as external-toolbox scope rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/hdr/recons_haar_pyramid.m": {
        "status": "ported",
        "note": "The upstream oversampled Haar-pyramid reconstructor is covered by the Python `recons_haar_pyramid(...)` / `reconsHaarPyramid(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over exact inverse replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/recons_pyramid.m": {
        "status": "ported",
        "note": "The upstream non-decimated pyramid reconstructor is covered by the Python `recons_pyramid(...)` / `reconsPyramid(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over the supported Haar and QMF branches.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/recons_qmf_pyramid.m": {
        "status": "ported",
        "note": "The upstream oversampled QMF-pyramid reconstructor is covered by the Python `recons_qmf_pyramid(...)` / `reconsQmfPyramid(...)` compatibility wrapper in `pyisetcam.scene`, with focused regression coverage over exact inverse replay.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/external/hdr/recons_steerable_pyramid_full.m": {
        "status": "out_of_scope",
        "note": "The upstream steerable-pyramid compatibility wrapper delegates straight into the external Simoncelli MatlabPyrTools reconstruction stack required by the vendored HDR README, so it is treated as external-toolbox scope rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/hdr/steerable_pyramid_full.m": {
        "status": "out_of_scope",
        "note": "The upstream steerable-pyramid compatibility wrapper delegates straight into the external Simoncelli MatlabPyrTools build stack required by the vendored HDR README, so it is treated as external-toolbox scope rather than a supported standalone pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/fstack/fstackdemo.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB demonstration notebook for the external focus-stack package, not a supported pyisetcam runtime API surface.",
        "module_hits": [],
    },
    "utility/external/fstack/showimages.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUI-only MATLAB image preview helper built around `imshow`, `text`, `rectangle`, and `pause`, so it is treated as visualization baggage rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/freezeColors/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only the index page for the third-party freezeColors figure-helper package, which is GUI-only plotting baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/freezeColors/freezeColors.m": {
        "status": "out_of_scope",
        "note": "The upstream file rewrites MATLAB figure and axis CData so multiple colormaps can coexist in one interactive figure, making it GUI-only visualization baggage rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/freezeColors/test/t_freezeColors.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an interactive MATLAB figure test for the third-party freezeColors package, so it is treated as GUI maintenance coverage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/freezeColors/unfreezeColors.m": {
        "status": "out_of_scope",
        "note": "The upstream file restores indexed colors for interactive MATLAB plot handles and depends entirely on figure-state mutation, so it is treated as GUI-only visualization baggage rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/insertInImage/insertInImage.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB figure and Image Processing Toolbox annotation helper that rasterizes plot or text objects into images, so it is treated as GUI-heavy external baggage rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/external/max/max2.m": {
        "status": "ported",
        "note": "The upstream restricted maximum helper is covered by the Python `max2(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over row or column subset selection and MATLAB-style first-hit index replay.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/max/min2.m": {
        "status": "ported",
        "note": "The upstream restricted minimum helper is covered by the Python `min2(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over row or column subset selection and MATLAB-style first-hit index replay.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/zernike/zernfun.m": {
        "status": "ported",
        "note": "The upstream unit-circle Zernike basis helper is covered by the Python `zernfun(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over known polynomial values, cosine-or-sine angular replay, and the single-index mapping used by `zernfun2(...)`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/zernike/zernfun2.m": {
        "status": "ported",
        "note": "The upstream single-index Zernike helper is covered by the Python `zernfun2(...)` compatibility wrapper in `pyisetcam.utils`, which replays the vendored `P -> (N,M)` mapping for the first 36 modes.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/external/zernike/zernpol.m": {
        "status": "ported",
        "note": "The upstream radial Zernike polynomial helper is covered by the Python `zernpol(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over known low-order polynomial values and normalization handling.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/appendStruct.m": {
        "status": "ported",
        "note": "The upstream struct-append helper is covered by the Python `append_struct(...)` / `appendStruct(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over duplicate-key overwrite behavior from the latter mapping.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/cellDelete.m": {
        "status": "ported",
        "note": "The upstream 1-based cell-array deletion helper is covered by the Python `cell_delete(...)` / `cellDelete(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over MATLAB-style index deletion.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/cellMerge.m": {
        "status": "ported",
        "note": "The upstream cell-array concatenation helper is covered by the Python `cell_merge(...)` / `cellMerge(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over empty-input handling and ordered merge replay.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/checkToolbox.m": {
        "status": "out_of_scope",
        "note": "The upstream file queries installed MATLAB toolboxes through `ver`, which is host-environment bookkeeping specific to MATLAB rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/checkfields.m": {
        "status": "ported",
        "note": "The upstream nested-struct field-presence helper is covered by the Python `checkfields(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over nested mapping traversal.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/clx.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB workspace and figure cleanup script built around `clear`, `close all`, and hidden-window teardown, so it is treated as environment-reset baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/compareFields.m": {
        "status": "ported",
        "note": "The upstream top-level struct-field equality helper is covered by the Python `compare_fields(...)` / `compareFields(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over field-name and value equality replay.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/gatherStruct.m": {
        "status": "ported",
        "note": "The upstream recursive gather helper is covered by the Python `gather_struct(...)` / `gatherStruct(...)` compatibility wrapper in `pyisetcam.utils`, which recursively normalizes struct-like containers headlessly.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/hiddenHandle.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB `classdef` shim that hides `handle` methods from tab completion, which is MATLAB class-system ergonomics rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/ieAddObject.m": {
        "status": "ported",
        "note": "The upstream session-registration helper is covered by the Python `ie_add_object(...)` / `ieAddObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over session slot assignment and camera subobject registration.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieAppGet.m": {
        "status": "ported",
        "note": "The upstream window-handle lookup helper is covered by the Python `ie_app_get(...)` / `ieAppGet(...)` wrapper in `pyisetcam.session`, with direct regression coverage over scene, OI, sensor, IP, and display window payload lookup.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieContains.m": {
        "status": "ported",
        "note": "The upstream pre-`contains()` string helper is covered by the Python `ie_contains(...)` / `ieContains(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over scalar and list-like string replay.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/ieDeleteObject.m": {
        "status": "ported",
        "note": "The upstream session deletion helper is covered by the Python `ie_delete_object(...)` / `ieDeleteObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over object removal and remaining-count replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieEquivalentObjtype.m": {
        "status": "ported",
        "note": "The upstream object-type canonicalizer is covered by the Python `ie_equivalent_objtype(...)` / `ieEquivalentObjtype(...)` wrapper in `pyisetcam.session`, with regression coverage over MATLAB-style ISA or VCI aliases.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieFindCallback.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUI-only MATLAB menu-callback inspection helper and is marked `Not working` upstream, so it is treated as unsupported figure-debugging baggage rather than a pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/ieGetObject.m": {
        "status": "ported",
        "note": "The upstream object lookup helper is covered by the Python `ie_get_object(...)` / `ieGetObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over returned-object and object-id replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieGetSelectedObject.m": {
        "status": "ported",
        "note": "The upstream selected-object lookup helper is covered by the Python `ie_get_selected_object(...)` / `ieGetSelectedObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over selected object-id and object replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieHash.m": {
        "status": "ported",
        "note": "The upstream recursive hashing helper is covered by the Python `ie_hash(...)` / `ieHash(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over the upstream array, binary, ASCII, file, and output-format examples.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/ieMemoryCheck.m": {
        "status": "out_of_scope",
        "note": "The upstream file sums `whos` results from the MATLAB caller or base workspace, which is workspace introspection specific to MATLAB rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/ieMemorySize.m": {
        "status": "out_of_scope",
        "note": "The upstream file estimates object size through MATLAB property reflection and `whos(varname(...))`, which is MATLAB workspace introspection rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/ieNotDefined.m": {
        "status": "out_of_scope",
        "note": "The upstream file inspects the MATLAB caller workspace by variable name string through `evalin`, so it is treated as MATLAB introspection baggage rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/iePTable.m": {
        "status": "ported",
        "note": "The upstream property-table helper is covered by the Python `ie_p_table(...)` / `iePTable(...)` wrapper in `pyisetcam.ptable`, with direct regression coverage over embedded headless table construction.",
        "module_hits": ["pyisetcam.ptable", "pyisetcam.__init__"],
    },
    "utility/programming/ieParamFormat.m": {
        "status": "ported",
        "note": "The upstream parameter-normalization helper is covered by the Python `param_format(...)` behavior exported as `ieParamFormat(...)`, with focused regression coverage over string and alternating key/value list normalization.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/ieParameterOtype.m": {
        "status": "ported",
        "note": "The upstream object-type parameter parser is covered by the Python `ie_parameter_otype(...)` / `ieParameterOtype(...)` helper in `pyisetcam.utils`, with focused regression coverage over split object or parameter parsing.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/ieQuestdlg.m": {
        "status": "out_of_scope",
        "note": "The upstream file opens an interactive MATLAB dialog and waits on user input through `uiwait`, so it is treated as GUI-only baggage rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/ieRemovePathDir.m": {
        "status": "out_of_scope",
        "note": "The upstream file edits MATLAB path strings for startup bootstrap and VCS-folder filtering, which is superseded by the current Python path bootstrap layer rather than a standalone runtime surface.",
        "module_hits": [],
    },
    "utility/programming/ieReplaceObject.m": {
        "status": "ported",
        "note": "The upstream session replacement helper is covered by the Python `ie_replace_object(...)` / `ieReplaceObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over slot-preserving replacement.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieSelectObject.m": {
        "status": "ported",
        "note": "The upstream session-selection helper is covered by the Python `ie_select_object(...)` / `ieSelectObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over MATLAB-style current-object selection.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/ieStructCompare.m": {
        "status": "ported",
        "note": "The upstream recursive struct-difference helper is covered by the Python `ie_struct_compare(...)` / `ieStructCompare(...)` wrapper in `pyisetcam.utils`, with focused regression coverage over nested field-path difference reporting and `comp_struct(...)` passthrough outputs.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/ieStructRemoveEmptyField.m": {
        "status": "ported",
        "note": "The upstream empty-field pruning helper is covered by the Python `ie_struct_remove_empty_field(...)` / `ieStructRemoveEmptyField(...)` wrapper in `pyisetcam.utils`, with focused regression coverage over empty-string, empty-list, and empty-array field removal.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/notDefined.m": {
        "status": "out_of_scope",
        "note": "The upstream file inspects the MATLAB caller workspace and optional nested struct fields by variable name string, which is MATLAB introspection rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/programming/replaceNaN.m": {
        "status": "ported",
        "note": "The upstream NaN replacement helper is covered by the Python `replace_nan(...)` / `replaceNaN(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over numeric array replacement.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/struct2pairs.m": {
        "status": "ported",
        "note": "The upstream struct-to-pairs helper is covered by the Python `struct2pairs(...)` compatibility wrapper in `pyisetcam.utils`, with focused regression coverage over alternating key/value replay.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/programming/vcAddAndSelectObject.m": {
        "status": "ported",
        "note": "The upstream add-and-select session helper is covered by the Python `vcAddAndSelectObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over returned ids and selected-object replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcCountObjects.m": {
        "status": "ported",
        "note": "The upstream session counting helper is covered by the Python `vcCountObjects(...)` wrapper in `pyisetcam.session`, with direct regression coverage over current-object family counts.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcDeleteObject.m": {
        "status": "ported",
        "note": "The upstream session deletion helper is covered by the Python `vcDeleteObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over deletion and remaining-count replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcDeleteSelectedObject.m": {
        "status": "ported",
        "note": "The upstream selected-object deletion helper is covered by the Python `vcDeleteSelectedObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over selected-slot deletion replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcDeleteSomeObjects.m": {
        "status": "ported",
        "note": "The upstream partial deletion helper is covered by the Python `vcDeleteSomeObjects(...)` wrapper in `pyisetcam.session`, with direct regression coverage over MATLAB-style selected-id list deletion.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcEquivalentObjtype.m": {
        "status": "ported",
        "note": "The upstream object-type alias helper is covered by the Python `vcEquivalentObjtype(...)` wrapper in `pyisetcam.session`, with regression coverage over canonical ISET object-type names.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcGetObject.m": {
        "status": "ported",
        "note": "The upstream object lookup helper is covered by the Python `vcGetObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over returned-object and object-id replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcGetObjectNames.m": {
        "status": "ported",
        "note": "The upstream object-name listing helper is covered by the Python `vcGetObjectNames(...)` wrapper in `pyisetcam.session`, with direct regression coverage over current object-name replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcGetObjectType.m": {
        "status": "ported",
        "note": "The upstream object-type helper is covered by the Python `vcGetObjectType(...)` wrapper in `pyisetcam.session`, with direct regression coverage over canonical type naming for IP objects.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcGetObjects.m": {
        "status": "ported",
        "note": "The upstream object-list lookup helper is covered by the Python `vcGetObjects(...)` wrapper in `pyisetcam.session`, with direct regression coverage over ordered object-list replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcGetSelectedObject.m": {
        "status": "ported",
        "note": "The upstream selected-object lookup helper is covered by the Python `vcGetSelectedObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over selected object-id and object replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcNewObjectName.m": {
        "status": "ported",
        "note": "The upstream new-object naming helper is covered by the Python `vcNewObjectName(...)` wrapper in `pyisetcam.session`, with direct regression coverage over MATLAB-style incremental name generation.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcNewObjectValue.m": {
        "status": "ported",
        "note": "The upstream new-object slot helper is covered by the Python `vcNewObjectValue(...)` wrapper in `pyisetcam.session`, with direct regression coverage over scene and camera tuple replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcReplaceAndSelectObject.m": {
        "status": "ported",
        "note": "The upstream replace-and-select helper is covered by the Python `vcReplaceAndSelectObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over replacement and selection replay.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcReplaceObject.m": {
        "status": "ported",
        "note": "The upstream object replacement helper is covered by the Python `vcReplaceObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over slot-preserving replacement.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcSetObjects.m": {
        "status": "ported",
        "note": "The upstream bulk-object replacement helper is covered by the Python `vcSetObjects(...)` wrapper in `pyisetcam.session`, with direct regression coverage over reindexing and stale-selection clearing.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/programming/vcSetSelectedObject.m": {
        "status": "ported",
        "note": "The upstream selection helper is covered by the Python `vcSetSelectedObject(...)` wrapper in `pyisetcam.session`, with direct regression coverage over current selected-id mutation and clearing.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "utility/xml/ieStruct2XML.m": {
        "status": "ported",
        "note": "The legacy MATLAB XML writer is covered by the Python `ie_struct2xml(...)` / `ieStruct2XML(...)` compatibility wrapper, including MATLAB-style `_dash_`, `_colon_`, and `_dot_` name escaping plus nested repeated-element replay.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/xml/ieXML2struct.m": {
        "status": "ported",
        "note": "The legacy MATLAB XML reader is covered by the Python `ie_xml2struct(...)` / `ieXML2struct(...)` compatibility wrapper, including MATLAB-style `_dash_`, `_colon_`, and `_dot_` name decoding plus repeated-element list replay.",
        "module_hits": ["pyisetcam.fileio", "pyisetcam.__init__"],
    },
    "utility/list/ieDataList.m": {
        "status": "ported",
        "note": "The lightweight MATLAB data-list dispatcher is covered by the Python `ie_data_list(...)` / `ieDataList(...)` compatibility wrapper, which replays the implemented reflectance and light branches headlessly.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/list/ieLightList.m": {
        "status": "ported",
        "note": "The legacy MATLAB illuminant catalog helper is covered by the Python `ie_light_list(...)` / `ieLightList(...)` compatibility wrapper over the vendored light asset tree, including `cct.mat` skipping and zero-floor replacement.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/list/ieReflectanceList.m": {
        "status": "ported",
        "note": "The legacy MATLAB reflectance catalog helper is covered by the Python `ie_reflectance_list(...)` / `ieReflectanceList(...)` compatibility wrapper over the vendored surface datasets, including basis-file skipping and reflectance-range filtering.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/getMiddleMatrix.m": {
        "status": "ported",
        "note": "The legacy MATLAB centered matrix-extraction helper is covered by the Python `get_middle_matrix(...)` / `getMiddleMatrix(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieClip.m": {
        "status": "ported",
        "note": "The legacy MATLAB clipping helper is covered by the Python `ie_clip(...)` / `ieClip(...)` compatibility wrapper in `pyisetcam.utils`, including default [0,1], symmetric single-bound, and upper-only clipping behavior.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieCXcorr.m": {
        "status": "ported",
        "note": "The legacy MATLAB cross-correlation helper is covered directly by the Python `ie_cxcorr(...)` / `ieCXcorr(...)` compatibility wrapper in `pyisetcam.iso`.",
        "module_hits": ["pyisetcam.iso", "pyisetcam.__init__"],
    },
    "utility/numerical/ieFitLine.m": {
        "status": "ported",
        "note": "The legacy MATLAB least-squares line-fit helper is covered directly by the Python `ie_fit_line(...)` / `ieFitLine(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieHwhm2SD.m": {
        "status": "ported",
        "note": "The legacy MATLAB half-width-half-max conversion helper is covered by the Python `ie_hwhm_to_sd(...)` / `ieHwhm2SD(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieScale.m": {
        "status": "ported",
        "note": "The legacy MATLAB range-scaling helper is covered by the Python `ie_scale(...)` / `ieScale(...)` compatibility wrapper in `pyisetcam.utils`, including peak scaling and bounded remapping behavior.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieScaleColumns.m": {
        "status": "ported",
        "note": "The legacy MATLAB per-column scaling helper is covered by the Python `ie_scale_columns(...)` / `ieScaleColumns(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/isodd.m": {
        "status": "ported",
        "note": "The legacy MATLAB oddness helper is covered by the Python `isodd(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/rotationMatrix3d.m": {
        "status": "ported",
        "note": "The legacy MATLAB 3D rotation helper is covered by the Python `rotation_matrix_3d(...)` / `rotationMatrix3d(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ffndgrid.m": {
        "status": "ported",
        "note": "The legacy MATLAB uneven-sample gridding helper is covered by the Python `ffndgrid(...)` compatibility wrapper in `pyisetcam.utils`, including averaged bin filling, negative-grid-count spacing, and MATLAB meshgrid orientation for 2-D outputs.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieCompressData.m": {
        "status": "ported",
        "note": "The deprecated MATLAB quantization helper is covered by the Python `ie_compress_data(...)` / `ieCompressData(...)` compatibility wrapper in `pyisetcam.utils`, including uint16 and uint32 compression paths.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieLineAlign.m": {
        "status": "ported",
        "note": "The legacy MATLAB line-alignment helper is covered by the Python `ie_line_align(...)` / `ieLineAlign(...)` compatibility wrapper in `pyisetcam.utils`, using the same shift-and-scale interpolation objective.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/ieTikhonov.m": {
        "status": "ported",
        "note": "The legacy MATLAB ridge and smoothness regularizer is covered by the Python `ie_tikhonov(...)` / `ieTikhonov(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/qinterp2.m": {
        "status": "ported",
        "note": "The legacy MATLAB fast 2-D interpolation helper is covered by the Python `qinterp2(...)` compatibility wrapper in `pyisetcam.utils`, including nearest-neighbor, triangular, and bilinear modes.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/unpadarray.m": {
        "status": "ported",
        "note": "The legacy MATLAB padding-inverse helper is covered by the Python `unpadarray(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/upperQuad2FullMatrix.m": {
        "status": "ported",
        "note": "The legacy MATLAB quadrant-mirroring helper is covered by the Python `upper_quad_to_full_matrix(...)` / `upperQuad2FullMatrix(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/numerical/vectorLength.m": {
        "status": "ported",
        "note": "The legacy MATLAB vector-norm helper is covered by the Python `vector_length(...)` / `vectorLength(...)` compatibility wrapper in `pyisetcam.utils`, including NaN-as-zero behavior.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcBlur.m": {
        "status": "ported",
        "note": "The legacy MATLAB per-plane hypercube blur helper is covered by the Python `hc_blur(...)` / `hcBlur(...)` compatibility wrapper in `pyisetcam.utils`, including the default `fspecial('gaussian',[sd sd])` kernel and same-sized convolution behavior.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcIlluminantScale.m": {
        "status": "ported",
        "note": "The legacy MATLAB hypercube illuminant-scaling helper is covered by the Python `hc_illuminant_scale(...)` / `hcIlluminantScale(...)` compatibility wrapper in `pyisetcam.utils`, including the mean-SPD pseudoinverse projection and normalized scale-map contract.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcReadHyspex.m": {
        "status": "ported",
        "note": "The legacy MATLAB ENVI hypercube reader is covered by the Python `hc_read_hyspex(...)` / `hcReadHyspex(...)` compatibility wrapper in `pyisetcam.utils`, including line, sample, and band subsetting plus `default bands` selection.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcReadHyspexImginfo.m": {
        "status": "ported",
        "note": "The legacy MATLAB ENVI header parser is covered by the Python `hc_read_hyspex_imginfo(...)` / `hcReadHyspexImginfo(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcRedFISh.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a standalone external-dataset notebook for the Grenoble ReDFISh HDF5 bundle and scene-construction demos, not a reusable headless runtime API surface within the pinned ISETCam snapshot.",
        "module_hits": [],
    },
    "utility/hypercube/hcViewer.m": {
        "status": "ported",
        "note": "The legacy MATLAB slider-based hypercube viewer is covered by the Python `hc_viewer(...)` / `hcViewer(...)` compatibility wrapper in `pyisetcam.utils`, returning the initial slice payload headlessly instead of opening a UI control.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcimage.m": {
        "status": "ported",
        "note": "The legacy MATLAB hypercube display helper is covered by the Python `hc_image(...)` / `hcimage(...)` compatibility wrapper in `pyisetcam.utils`, including mean-gray, montage, and movie payload modes.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcimageCrop.m": {
        "status": "ported",
        "note": "The legacy MATLAB hypercube crop helper is covered by the Python `hc_image_crop(...)` / `hcimageCrop(...)` compatibility wrapper in `pyisetcam.utils`, using the same `[col,row,width,height]` rect semantics headlessly.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/hypercube/hcimageRotateClip.m": {
        "status": "ported",
        "note": "The legacy MATLAB hypercube rotate-and-clip helper is covered by the Python `hc_image_rotate_clip(...)` / `hcimageRotateClip(...)` compatibility wrapper in `pyisetcam.utils`, including `rot90` replay, percentile clipping, and clipped-pixel counting.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/dll70/dll70Path.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB deployment-path helper for selecting legacy `dll70` MEX directories by MATLAB version, not a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/dll70/ieCInterp3/ieCinterp3.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only the MATLAB help stub for a legacy compiled `ieCinterp3` MEX interpolation routine; the actual implementation is external compiled code rather than a vendored headless API surface in this snapshot.",
        "module_hits": [],
    },
    "utility/dll70/ieCompileMex.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a local MATLAB MEX build and install script for legacy `md5` and `ieGetMACAddress` binaries, not a supported headless runtime API surface.",
        "module_hits": [],
    },
    "utility/dll70/ieGetMACAddress.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a license-era host MAC-address probe that shells out to `ipconfig` or `ifconfig`; it is deployment-specific infrastructure rather than a supported pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/dll70/ieGetMACAddress2.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an alternate Windows-only MAC-address probe for legacy licensing workflows, not a supported headless pyisetcam runtime API.",
        "module_hits": [],
    },
    "utility/dll70/ieInstall.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB verification and installer script for legacy `md5` and `ieGetMACAddress` MEX dependencies, not a reusable headless runtime surface.",
        "module_hits": [],
    },
    "utility/dll70/ieVCRedistribution.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a platform-specific Microsoft Visual C++ redistributable installer for legacy MATLAB MEX binaries, outside the supported pyisetcam runtime target.",
        "module_hits": [],
    },
    "utility/dll70/md5Mex.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB MEX build script for the legacy `md5` binary, not a supported headless runtime API surface.",
        "module_hits": [],
    },
    "utility/gif/animatedGif.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a generic MATLAB grayscale-volume GIF writer around `imwrite`, while the supported pyisetcam runtime already handles object-pipeline GIF export through higher-level headless paths such as `scene_make_video(...)` and `oi_preview_video(...)`.",
        "module_hits": [],
    },
    "utility/gif/gif/doc/gif_documentation.m": {
        "status": "out_of_scope",
        "note": "The upstream file is standalone documentation for an external generic MATLAB `gif` helper, not a reusable headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/gif/gif/gif.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a generic MATLAB figure-to-GIF helper built around `getframe`, `export_fig`, and persistent figure state, not a supported pyisetcam runtime API.",
        "module_hits": [],
    },
    "utility/gif/ie3dGIF.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-rotation export helper for oscillating 3-D MATLAB plots into GIFs, built around figure handles and `exportgraphics`, rather than a supported headless pyisetcam runtime surface.",
        "module_hits": [],
    },
    "utility/statistics/biNormal.m": {
        "status": "ported",
        "note": "The legacy MATLAB separable bivariate-Gaussian helper is covered by the Python `bi_normal(...)` / `biNormal(...)` compatibility wrapper in `pyisetcam.utils`, including optional in-place rotation over cropped support.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/expRand.m": {
        "status": "ported",
        "note": "The legacy MATLAB exponential random sampler is covered by the Python `exp_rand(...)` / `expRand(...)` compatibility wrapper in `pyisetcam.utils`, including MATLAB-style size parsing and inverse-CDF sampling.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/fractal/ieFractalDrawgrid.m": {
        "status": "ported",
        "note": "The legacy MATLAB fractal-grid overlay helper is covered by the Python `ie_fractal_drawgrid(...)` / `ieFractalDrawgrid(...)` compatibility wrapper in `pyisetcam.utils`, returning the magenta grid image headlessly instead of opening a figure window.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/fractal/ieFractaldim.m": {
        "status": "ported",
        "note": "The legacy MATLAB box-count fractal-dimension helper is covered by the Python `ie_fractal_dim(...)` / `ieFractaldim(...)` compatibility wrapper in `pyisetcam.utils`, including grayscale binarization, integral-image box counting, and best-fit slope calculation headlessly.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/gammaPDF.m": {
        "status": "ported",
        "note": "The legacy MATLAB gamma-kernel helper is covered by the Python `gamma_pdf(...)` / `gammaPDF(...)` compatibility wrapper in `pyisetcam.utils`, including the normalized discrete gamma-form curve construction used upstream.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/getGaussian.m": {
        "status": "ported",
        "note": "The legacy MATLAB receptive-field Gaussian helper is covered by the Python `get_gaussian(...)` / `getGaussian(...)` compatibility wrapper in `pyisetcam.utils`, including RF-support mesh construction and unit-volume normalization.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/ieExprnd.m": {
        "status": "ported",
        "note": "The legacy MATLAB exponential-distribution sampler is covered by the Python `ie_exprnd(...)` / `ieExprnd(...)` compatibility wrapper in `pyisetcam.utils`, including MATLAB-style size parsing and inverse-CDF sampling.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/ieMvnrnd.m": {
        "status": "ported",
        "note": "The legacy MATLAB multivariate-normal sampler is covered directly by the Python `ie_mvnrnd(...)` / `ieMvnrnd(...)` compatibility wrapper in `pyisetcam.utils`, with focused deterministic regression coverage over the Cholesky sampling contract.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__", "pyisetcam.parity"],
    },
    "utility/statistics/ieNormpdf.m": {
        "status": "ported",
        "note": "The legacy MATLAB normal-density helper is covered by the Python `ie_normpdf(...)` / `ieNormpdf(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/ieOneOverF.m": {
        "status": "ported",
        "note": "The legacy MATLAB radial 1/f spectrum helper is covered by the Python `ie_one_over_f(...)` / `ieOneOverF(...)` compatibility wrapper in `pyisetcam.utils`, including gamma-linearization, BT.601 grayscale conversion, FFT centering, and radial amplitude averaging.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/iePoisson.m": {
        "status": "ported",
        "note": "The legacy MATLAB Poisson sampler is covered by the Python `ie_poisson(...)` / `iePoisson(...)` compatibility wrapper in `pyisetcam.utils`, including scalar-vs-matrix lambda handling and frozen/random seed bookkeeping.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/iePrcomp.m": {
        "status": "ported",
        "note": "The legacy MATLAB principal-component helper is covered by the Python `ie_prcomp(...)` / `iePrcomp(...)` compatibility wrapper in `pyisetcam.utils`, including the basic and remove-mean covariance-SVD modes.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/iePrctile.m": {
        "status": "ported",
        "note": "The legacy MATLAB percentile helper is covered by the Python `ie_prctile(...)` / `iePrctile(...)` compatibility wrapper in `pyisetcam.utils`, including the fallback sorted-sample interpolation path used without the Statistics Toolbox.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/statistics/lorentzSum.m": {
        "status": "ported",
        "note": "The legacy MATLAB Lorentzian-sum helper is covered by the Python `lorentz_sum(...)` / `lorentzSum(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieFindWaveIndex.m": {
        "status": "ported",
        "note": "The legacy MATLAB wavelength-membership helper is covered by the Python `ie_find_wave_index(...)` / `ieFindWaveIndex(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieCmap.m": {
        "status": "ported",
        "note": "The legacy MATLAB color-map helper is covered by the Python `ie_cmap(...)` / `ieCmap(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieCropRect.m": {
        "status": "ported",
        "note": "The legacy MATLAB FOV crop-rectangle helper is covered by the Python `ie_crop_rect(...)` / `ieCropRect(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieFieldHeight2Index.m": {
        "status": "ported",
        "note": "The legacy MATLAB field-height index helper is covered by the Python `ie_field_height_to_index(...)` / `ieFieldHeight2Index(...)` compatibility wrapper in `pyisetcam.optics`.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "utility/image/ieLUTDigital.m": {
        "status": "ported",
        "note": "The legacy MATLAB DAC-to-linear-RGB helper is covered by the Python `ie_lut_digital(...)` / `ieLUTDigital(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieLUTInvert-deprecate.m": {
        "status": "ported",
        "note": "The deprecated MATLAB inverse-gamma helper is covered by the Python `ie_lut_invert(...)` / `ieLUTInvert(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieLUTInvert.m": {
        "status": "ported",
        "note": "The legacy MATLAB inverse-gamma helper is covered by the Python `ie_lut_invert(...)` / `ieLUTInvert(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieLUTLinear.m": {
        "status": "ported",
        "note": "The legacy MATLAB linear-RGB-to-DAC helper is covered by the Python `ie_lut_linear(...)` / `ieLUTLinear(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/convolvecirc.m": {
        "status": "ported",
        "note": "The legacy MATLAB circular-convolution helper is covered by the Python `convolve_circ(...)` / `convolvecirc(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageHparams.m": {
        "status": "ported",
        "note": "The legacy MATLAB harmonic-parameter default helper is covered by the Python `image_hparams(...)` / `imageHparams(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageGabor.m": {
        "status": "ported",
        "note": "The legacy MATLAB Gabor-pattern helper is covered by the Python `image_gabor(...)` / `imageGabor(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageInterpolate.m": {
        "status": "ported",
        "note": "The legacy MATLAB image-resampling helper is covered by the Python `image_interpolate(...)` / `imageInterpolate(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageMakeMontage.m": {
        "status": "ported",
        "note": "The legacy MATLAB hypercube montage builder is covered by the Python `image_make_montage(...)` / `imageMakeMontage(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieMontages.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a local PNG report-generator that captures labeled montage pages from the current folder rather than a reusable numeric API surface for the headless pipeline.",
        "module_hits": [],
    },
    "utility/image/imageMontage.m": {
        "status": "ported",
        "note": "The legacy MATLAB montage wrapper is covered headlessly by the Python `image_montage(...)` / `imageMontage(...)` compatibility wrapper, which returns the montage image and placeholder figure/colorbar handles instead of MATLAB GUI objects.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageDeadLeaves.m": {
        "status": "ported",
        "note": "The deprecated MATLAB dead-leaves helper is covered by the Python `image_dead_leaves(...)` / `imageDeadLeaves(...)` compatibility wrapper on top of the current dead-leaves generator.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/image/imageMultiview.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a MATLAB session/object-browser window orchestrator built around `listdlg`, figure creation, and subplot layout rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "utility/image/imageSlantedEdge.m": {
        "status": "ported",
        "note": "The legacy MATLAB slanted-edge target helper is covered by the Python `image_slanted_edge(...)` / `imageSlantedEdge(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageSPD.m": {
        "status": "ported",
        "note": "The legacy MATLAB spectral-image renderer is covered headlessly by the Python `image_spd(...)` / `imageSPD(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageSPD2RGB.m": {
        "status": "ported",
        "note": "The obsolete MATLAB SPD-to-RGB helper is covered by the Python `image_spd2rgb(...)` / `imageSPD2RGB(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imagehc2rgb.m": {
        "status": "ported",
        "note": "The legacy MATLAB multispectral-waveband visualizer is covered by the Python `image_hc2rgb(...)` / `imagehc2rgb(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageVernier.m": {
        "status": "ported",
        "note": "The legacy MATLAB Vernier-image helper is covered by the Python `image_vernier(...)` / `imageVernier(...)` compatibility wrapper in `pyisetcam.scene`.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.__init__"],
    },
    "utility/plots/ipPlot.m": {
        "status": "ported",
        "note": "The MATLAB image-processor plotting gateway is covered by the headless Python `ip_plot(...)` / `ipPlot(...)` wrapper with direct plotting regressions over line, chromaticity, RGB, luminance, and CIELAB/LUV payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__", "pyisetcam.parity"],
    },
    "utility/plots/oiPlot.m": {
        "status": "ported",
        "note": "The MATLAB optical-image plotting gateway is covered by the headless Python `oi_plot(...)` / `oiPlot(...)` wrapper with direct plotting regressions over ROI, line, PSF, and wavelength-domain payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__", "pyisetcam.parity"],
    },
    "utility/plots/plotOI.m": {
        "status": "ported",
        "note": "The legacy MATLAB `plotOI(...)` entry point is covered by the Python `plotOI(...)` compatibility alias on top of `oi_plot(...)` / `oiPlot(...)`.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotScene.m": {
        "status": "ported",
        "note": "The MATLAB scene plotting gateway is covered by the headless Python `scene_plot(...)` / `plotScene(...)` wrapper with direct plotting regressions over radiance, illuminant, and chromaticity payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__", "pyisetcam.parity"],
    },
    "utility/plots/scenePlot.m": {
        "status": "ported",
        "note": "The legacy MATLAB `scenePlot(...)` entry point is covered by the Python `scenePlot(...)` compatibility alias on top of `scene_plot(...)` / `plotScene(...)`.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSensor.m": {
        "status": "ported",
        "note": "The MATLAB sensor plotting gateway is covered by the headless Python `sensor_plot(...)` / `plotSensor(...)` wrapper with direct plotting regressions over line, histogram, SNR, FFT, and CFA payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__", "pyisetcam.parity"],
    },
    "utility/plots/sensorPlot.m": {
        "status": "ported",
        "note": "The legacy MATLAB `sensorPlot(...)` entry point is covered by the Python `sensorPlot(...)` compatibility alias on top of `sensor_plot(...)` / `plotSensor(...)`.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSensorFFT.m": {
        "status": "ported",
        "note": "The MATLAB sensor FFT plotting gateway is covered by the Python `sensor_plot_fft(...)` / `plotSensorFFT(...)` compatibility wrapper with direct FFT payload regressions.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSensorHist.m": {
        "status": "ported",
        "note": "The MATLAB sensor histogram plotting helper is covered by the Python `sensor_plot_hist(...)` / `plotSensorHist(...)` compatibility wrapper over ROI volts, electrons, and DV histogram payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/sensorPlotHist.m": {
        "status": "ported",
        "note": "The legacy MATLAB `sensorPlotHist(...)` entry point is covered by the Python `sensorPlotHist(...)` compatibility alias on top of `sensor_plot_hist(...)` / `plotSensorHist(...)`.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/sensorPlotLine.m": {
        "status": "ported",
        "note": "The MATLAB sensor line plotting helper is covered by the Python `sensor_plot_line(...)` / `sensorPlotLine(...)` compatibility wrapper with direct space-domain and FFT payload regressions.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__", "pyisetcam.parity"],
    },
    "utility/plots/plotDisplaySPD.m": {
        "status": "ported",
        "note": "The MATLAB display primary-SPD plotting helper is covered by the Python `plot_display_spd(...)` / `plotDisplaySPD(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotDisplayLine.m": {
        "status": "ported",
        "note": "The MATLAB display line plotting helper is covered by the Python `plot_display_line(...)` / `plotDisplayLine(...)` compatibility wrapper, including MATLAB-style analog or quantized line payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotDisplayColor.m": {
        "status": "ported",
        "note": "The MATLAB display color-analysis helper is covered by the Python `plot_display_color(...)` / `plotDisplayColor(...)` compatibility wrapper over RGB, chromaticity, luminance, and CIELAB/LUV payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotDisplayGamut.m": {
        "status": "ported",
        "note": "The MATLAB display-gamut plotting helper is covered by the Python `plot_display_gamut(...)` / `plotDisplayGamut(...)` compatibility wrapper on top of display primary XYZ data.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotPixelSNR.m": {
        "status": "ported",
        "note": "The MATLAB pixel-SNR plotting helper is covered by the Python `plot_pixel_snr(...)` / `plotPixelSNR(...)` compatibility wrapper on top of the existing `plotSensor(..., 'pixel snr')` payload.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSensorEtendue.m": {
        "status": "ported",
        "note": "The MATLAB sensor-etendue plotting helper is covered by the Python `plot_sensor_etendue(...)` / `plotSensorEtendue(...)` compatibility wrapper on top of the existing `plotSensor(..., 'etendue')` payload.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSensorSNR.m": {
        "status": "ported",
        "note": "The MATLAB sensor-SNR plotting helper is covered by the Python `plot_sensor_snr(...)` / `plotSensorSNR(...)` compatibility wrapper on top of the existing `plotSensor(..., 'sensor snr')` payload.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSetUpWindow.m": {
        "status": "ported",
        "note": "The MATLAB graph-window setup helper is covered by the Python `plot_set_up_window(...)` / `plotSetUpWindow(...)` wrapper, which returns the graph-window defaults headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/fise_plotDefaults.m": {
        "status": "ported",
        "note": "The MATLAB root-defaults script is covered by the Python `fise_plot_defaults(...)` / `fisePlotDefaults(...)` wrapper, which returns the same graphics-default property map headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/hist2d.m": {
        "status": "ported",
        "note": "The MATLAB 2-D histogram helper is covered by the Python `hist2d(...)` wrapper in `pyisetcam.plotting`, which returns the same nearest-support count matrix headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/ieFigureFormat.m": {
        "status": "ported",
        "note": "The MATLAB figure-format helper is covered by the Python `ie_figure_format(...)` / `ieFigureFormat(...)` headless wrapper, which returns the formatted figure and axes property payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/ieFigureResize.m": {
        "status": "ported",
        "note": "The MATLAB figure-resize helper is covered by the Python `ie_figure_resize(...)` / `ieFigureResize(...)` headless wrapper, which returns the requested units and figure-position payload.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/ieFormatFigure.m": {
        "status": "ported",
        "note": "The deprecated MATLAB `ieFormatFigure(...)` entry point is covered by the Python `ieFormatFigure(...)` compatibility alias on top of `ie_figure_format(...)` / `ieFigureFormat(...)`.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/ieHistImage.m": {
        "status": "ported",
        "note": "The MATLAB histogram-image helper is covered by the Python `ie_hist_image(...)` / `ieHistImage(...)` wrapper, which returns histogram-bin or scatter-density image payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/iePlot.m": {
        "status": "ported",
        "note": "The MATLAB generic `iePlot(...)` helper is covered by the Python `ie_plot(...)` / `iePlot(...)` headless wrapper, which returns parsed line-series payloads instead of opening a figure.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/iePlotSet.m": {
        "status": "ported",
        "note": "The MATLAB `iePlotSet(...)` helper is covered by the Python `ie_plot_set(...)` / `iePlotSet(...)` wrapper, which applies line-width updates to headless line-series payloads.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/iePlotShadeBackground.m": {
        "status": "ported",
        "note": "The MATLAB shaded-background helper is covered by the Python `ie_plot_shade_background(...)` / `iePlotShadeBackground(...)` wrapper, which returns the background patch and layer payload headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/ieShape.m": {
        "status": "ported",
        "note": "The MATLAB analytic-shape helper is covered by the Python `ie_shape(...)` / `ieShape(...)` wrapper for the current circle workflow.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotContrastHistogram.m": {
        "status": "ported",
        "note": "The MATLAB contrast-histogram helper is covered by the Python `plot_contrast_histogram(...)` / `plotContrastHistogram(...)` wrapper, which returns histogram counts and bin centers headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotEtendueRatio.m": {
        "status": "ported",
        "note": "The MATLAB etendue-ratio mesh helper is covered by the Python `plot_etendue_ratio(...)` / `plotEtendueRatio(...)` wrapper, which returns sensor support and ratio payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotGaussianSpectrum.m": {
        "status": "ported",
        "note": "The MATLAB Gaussian-spectrum plotting helper is covered by the Python `plot_gaussian_spectrum(...)` / `plotGaussianSpectrum(...)` wrapper, which returns transmittance and wavelength-color payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotML.m": {
        "status": "ported",
        "note": "The MATLAB microlens plotting gateway is covered by the Python `plot_ml(...)` / `plotML(...)` compatibility wrapper, including offsets, mesh pixel-irradiance, and image pixel-irradiance payloads without opening a figure.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotMetrics.m": {
        "status": "ported",
        "note": "The MATLAB metrics histogram helper is covered by the Python `plot_metrics(...)` / `plotMetrics(...)` compatibility wrapper, which returns ROI histogram counts, bin edges, and summary-stat annotation payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSpectrumLocus.m": {
        "status": "ported",
        "note": "The MATLAB spectrum-locus plotting helper is covered by the Python `plot_spectrum_locus(...)` / `plotSpectrumLocus(...)` wrapper, which returns chromaticity locus and closing-line payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/iePlaneFromVectors.m": {
        "status": "ported",
        "note": "The MATLAB plane-from-vectors helper is covered by the Python `ie_plane_from_vectors(...)` / `iePlaneFromVectors(...)` wrapper in `pyisetcam.plotting`.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/iePlotJitter.m": {
        "status": "ported",
        "note": "The MATLAB jittered-scatter helper is covered by the Python `ie_plot_jitter(...)` / `iePlotJitter(...)` wrapper, which returns the jittered point payload headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotNormal.m": {
        "status": "ported",
        "note": "The MATLAB normal-distribution plotting helper is covered by the Python `plot_normal(...)` / `plotNormal(...)` wrapper, which returns the sampled PDF payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotRadiance.m": {
        "status": "ported",
        "note": "The MATLAB spectral-radiance plotting helper is covered by the Python `plot_radiance(...)` / `plotRadiance(...)` wrapper, which returns wavelength-aligned line payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotReflectance.m": {
        "status": "ported",
        "note": "The MATLAB spectral-reflectance plotting helper is covered by the Python `plot_reflectance(...)` / `plotReflectance(...)` wrapper, which returns wavelength-aligned line payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/identityLine.m": {
        "status": "ported",
        "note": "The MATLAB axis-identity helper is covered by the Python `identity_line(...)` / `identityLine(...)` headless wrapper, which returns the canonical line geometry and styling payload.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotTextString.m": {
        "status": "ported",
        "note": "The MATLAB canonical text-placement helper is covered by the Python `plot_text_string(...)` / `plotTextString(...)` headless wrapper, which returns text position and style payloads from axis limits.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/scatplot.m": {
        "status": "ported",
        "note": "The MATLAB scatter-density helper is covered by the Python `scatplot(...)` wrapper in `pyisetcam.plotting`, which returns data-density, gridded-density, contour, and grouped-point payloads headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/plotSceneTest.m": {
        "status": "parity",
        "note": "The legacy scene-plot workflow is now exercised directly by the focused `test_plot_scene_test_workflow(...)` regression in `tests/unit/test_plotting.py`, which covers the supported headless `plotScene(...)` luminance-line, illuminant, reflectance, and chromaticity paths used by the upstream test script.",
        "module_hits": ["pyisetcam.plotting"],
    },
    "utility/plots/plotSensorTest.m": {
        "status": "parity",
        "note": "The legacy sensor-plot workflow is now exercised directly by the focused `test_plot_sensor_test_workflow(...)` regression in `tests/unit/test_plotting.py`, covering the supported headless `plotSensor(...)` line, histogram, SNR, CFA, etendue, and color-filter paths from the upstream test script.",
        "module_hits": ["pyisetcam.plotting"],
    },
    "utility/plots/sensorPlotColor.m": {
        "status": "ported",
        "note": "The MATLAB sensor cross-correlation plotting helper is covered by the Python `sensor_plot_color(...)` / `sensorPlotColor(...)` compatibility wrapper, which returns demosaiced channel-scatter payloads plus blackbody reference loci headlessly.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/xaxisLine.m": {
        "status": "ported",
        "note": "The MATLAB x-axis guide-line helper is covered by the Python `xaxis_line(...)` / `xaxisLine(...)` headless wrapper.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/plots/yaxisLine.m": {
        "status": "ported",
        "note": "The MATLAB y-axis guide-line helper is covered by the Python `yaxis_line(...)` / `yaxisLine(...)` headless wrapper.",
        "module_hits": ["pyisetcam.plotting", "pyisetcam.__init__"],
    },
    "utility/image/imageSetHarmonic.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE-only harmonic-parameter dialog and remains outside the headless migration target.",
        "module_hits": [],
    },
    "utility/image/imageShowImage.m": {
        "status": "ported",
        "note": "The legacy MATLAB IP-image renderer is covered headlessly by the Python `image_show_image(...)` / `imageShowImage(...)` compatibility wrapper on top of the existing image-processor render data.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__"],
    },
    "utility/image/imageTranslate.m": {
        "status": "ported",
        "note": "The legacy MATLAB image-translation helper is covered by the Python `image_translate(...)` / `imageTranslate(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageTranspose.m": {
        "status": "ported",
        "note": "The legacy MATLAB per-plane transpose helper is covered by the Python `image_transpose(...)` / `imageTranspose(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imagescM.m": {
        "status": "ported",
        "note": "The legacy MATLAB monochrome-display helper is covered headlessly by the Python `imagesc_m(...)` / `imagescM(...)` compatibility wrapper, which returns a display payload instead of MATLAB graphics handles.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imagescOPP.m": {
        "status": "ported",
        "note": "The legacy MATLAB opponent-image display helper is covered headlessly by the Python `imagesc_opp(...)` / `imagescOPP(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imagescRGB.m": {
        "status": "ported",
        "note": "The legacy MATLAB RGB display helper is covered headlessly by the Python `imagesc_rgb(...)` / `imagescRGB(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieRadialMatrix.m": {
        "status": "ported",
        "note": "The legacy MATLAB radial-distance helper is covered by the Python `ie_radial_matrix(...)` / `ieRadialMatrix(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/ieWave2Index.m": {
        "status": "ported",
        "note": "The legacy MATLAB wavelength-index helper is covered by the Python `ie_wave2_index(...)` / `ieWave2Index(...)` compatibility wrapper in `pyisetcam.utils`, preserving the MATLAB 1-based return contract.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageBoundingBox.m": {
        "status": "ported",
        "note": "The legacy MATLAB non-zero-support bounding-box helper is covered by the Python `image_bounding_box(...)` / `imageBoundingBox(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageCentroid.m": {
        "status": "ported",
        "note": "The legacy MATLAB weighted-centroid helper is covered by the Python `image_centroid(...)` / `imageCentroid(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageCircular.m": {
        "status": "ported",
        "note": "The legacy MATLAB centered circular-aperture helper is covered by the Python `image_circular(...)` / `imageCircular(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/imageContrast.m": {
        "status": "ported",
        "note": "The legacy MATLAB per-channel contrast helper is covered by the Python `image_contrast(...)` / `imageContrast(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/image/rgb2dac.m": {
        "status": "ported",
        "note": "The legacy MATLAB RGB-to-DAC helper is covered by the Python `rgb_to_dac(...)` / `rgb2dac(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/dpi2mperdot.m": {
        "status": "ported",
        "note": "The legacy MATLAB dots-per-inch conversion helper is covered by the Python `dpi2mperdot(...)` compatibility wrapper in `pyisetcam.utils`, including the default microns-per-dot return scale.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/ieDpi2Mperdot.m": {
        "status": "ported",
        "note": "The legacy MATLAB DPI conversion alias is covered by the Python `ie_dpi2_mperdot(...)` / `ieDpi2Mperdot(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/ieN2MegaPixel.m": {
        "status": "ported",
        "note": "The display-resolution sizing helper is covered directly by the Python `ie_n_to_megapixel(...)` / `ieN2MegaPixel(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/ieSpace2Amp.m": {
        "status": "ported",
        "note": "The legacy MATLAB spatial-FFT helper is covered by the Python `ie_space_to_amp(...)` / `ieSpace2Amp(...)` compatibility wrapper in `pyisetcam.utils`, including the truncated mean-to-Nyquist support contract.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/ieUnitScaleFactor.m": {
        "status": "ported",
        "note": "The legacy MATLAB unit-scaling helper is covered by the Python `ie_unit_scale_factor(...)` / `ieUnitScaleFactor(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/sample2space.m": {
        "status": "ported",
        "note": "The legacy MATLAB centered-support helper is covered by the Python `sample2space(...)` compatibility wrapper in `pyisetcam.utils`.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/units/space2sample.m": {
        "status": "ported",
        "note": "The obsolete MATLAB inverse support helper is covered by the Python `space2sample(...)` compatibility wrapper in `pyisetcam.utils`, preserving the zero-based offset contract used upstream.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/file/vcSelectDataFile.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a persistent GUI file-picker around MATLAB `uigetfile` / `uiputfile` dialogs and remains outside the headless migration target.",
        "module_hits": [],
    },
    "utility/file/vcSelectImage.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUI image-picker wrapper around MATLAB file dialogs and remains outside the headless migration target.",
        "module_hits": [],
    },
    "main/ieInitSession.m": {
        "status": "ported",
        "note": "The legacy MATLAB session initializer is covered by the Python `ie_init_session(...)` / `ieInitSession(...)` compatibility wrapper on top of the headless `SessionContext` object model.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "main/ieMainClose.m": {
        "status": "ported",
        "note": "The legacy MATLAB main-window close helper is covered by the Python `ie_main_close(...)` / `ieMainClose(...)` compatibility wrapper, which clears the tracked window slots headlessly instead of deleting GUIDE figures.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "main/ieMainW.m": {
        "status": "ported",
        "note": "The GUIDE main-window launcher is covered headlessly by the Python `ie_main_w(...)` / `ieMainW(...)` compatibility wrapper, which creates or reuses the tracked main-window placeholder rather than opening a MATLAB figure.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "main/iePrintSessionInfo.m": {
        "status": "ported",
        "note": "The legacy MATLAB session-summary printer is covered by the Python `ie_print_session_info(...)` / `iePrintSessionInfo(...)` compatibility wrapper, which returns the same object-list text payload headlessly.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "main/ieSessionGet.m": {
        "status": "ported",
        "note": "The legacy MATLAB session getter is covered by the Python `ie_session_get(...)` / `ieSessionGet(...)` compatibility wrapper over the headless `SessionContext` state.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "main/ieSessionSet.m": {
        "status": "ported",
        "note": "The legacy MATLAB session setter is covered by the Python `ie_session_set(...)` / `ieSessionSet(...)` compatibility wrapper over the headless `SessionContext` state.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "main/mainOpen.m": {
        "status": "ported",
        "note": "The GUIDE main-window bootstrap helper is covered headlessly by the Python `main_open(...)` / `mainOpen(...)` compatibility wrapper, which records the main-window placeholder and handle state without MATLAB license dialogs or GUI callbacks.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "ISET.m": {
        "status": "ported",
        "note": "The legacy MATLAB startup script is covered by the headless Python `iset(...)` / `ISET(...)` compatibility wrapper on top of the optional session layer, without reproducing the MATLAB GUI launch.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "isetPath.m": {
        "status": "ported",
        "note": "The MATLAB path bootstrap helper is covered by the Python `iset_path(...)` / `isetPath(...)` compatibility wrapper, including recursive non-VCS path discovery and optional `sys.path` injection.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "isetRootPath.m": {
        "status": "ported",
        "note": "The MATLAB root-path helper is covered by the Python `iset_root_path(...)` / `isetRootPath(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "imgproc/demosaic/bayerIndices.m": {
        "status": "ported",
        "note": "The low-level MATLAB Bayer-index helper is already covered by the shared internal `_bayer_indices` implementation.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/demosaic/ieBilinear.m": {
        "status": "ported",
        "note": "The low-level MATLAB bilinear demosaic helper is already covered by the shared internal `_ie_bilinear` implementation.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/demosaic/mosaicConverter.m": {
        "status": "ported",
        "note": "The low-level MATLAB Bayer-pattern conversion helper is already covered by the shared internal `_mosaic_converter` implementation.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/demosaic/Pocs.m": {
        "status": "ported",
        "note": "The legacy MATLAB POCS demosaic entry point is covered by the Python `pocs(...)` / `Pocs(...)` compatibility wrapper on top of the existing Bayer/CFA helpers.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__"],
    },
    "imgproc/binning/binAnalog2digital.m": {
        "status": "ported",
        "note": "The legacy MATLAB binning quantizer is already covered by the Python `analog_to_digital(...)` / `analog2digital(...)` compatibility surface.",
        "module_hits": ["pyisetcam.sensor", "pyisetcam.__init__"],
    },
    "imgproc/imageColorBalanceDeprecated.m": {
        "status": "ported",
        "note": "The deprecated MATLAB color-balance gateway is already covered by the Python `image_color_balance(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.ip"],
    },
    "utility/printing/FloydSteinberg.m": {
        "status": "ported",
        "note": "The legacy MATLAB Floyd-Steinberg error-diffusion helper is covered by the Python `floyd_steinberg(...)` / `FloydSteinberg(...)` compatibility wrapper and direct utility regressions.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "utility/printing/HalfToneImage.m": {
        "status": "ported",
        "note": "The legacy MATLAB halftone-cell helper is covered by the Python `half_tone_image(...)` / `HalfToneImage(...)` compatibility wrapper and direct utility regressions.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "imgproc/demosaic/faultypixel/FaultyPixelCorrection.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as requiring a rewrite, so it is treated as obsolete rather than actionable headless API debt.",
        "module_hits": [],
    },
    "imgproc/demosaic/lcc1.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked not fully implemented and questions whether the patented method should even remain in the simulator.",
        "module_hits": [],
    },
    "imgproc/demosaic/shtlin.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked not fully implemented and left with TODO placeholders, so it is treated as legacy experimental code rather than supported headless API debt.",
        "module_hits": [],
    },
    "imgproc/demosaic/shtlog.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked as needing a complete rewrite, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "imgproc/lightfield/LFDispMousePan.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a pure Light Field Toolbox display interaction helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/lightfield/LFDispSetup.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a pure Light Field Toolbox display setup helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/lightfield/LFDispVidCirc.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a display/video viewer helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/imageIlluminantCorrectionDeprecated.m": {
        "status": "ported",
        "note": "The deprecated MATLAB illuminant-correction file is already covered by the Python `image_illuminant_correction(...)` helper.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/lightfield/LFMicrolensGeometry.m": {
        "status": "out_of_scope",
        "note": "The upstream file opens a visualization figure for microlens geometry and is treated as GUI-only rather than a headless compute surface.",
        "module_hits": [],
    },
    "imgproc/openexr/exrreadchannels.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB help shim for a MEX-backed OpenEXR binding, which remains outside the supported headless Python migration target.",
        "module_hits": [],
    },
    "imgproc/openexr/exrwritechannels.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB help shim for a MEX-backed OpenEXR binding, which remains outside the supported headless Python migration target.",
        "module_hits": [],
    },
    "imgproc/openexr/make.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a local MATLAB MEX build script for OpenEXR bindings rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "imgproc/ipGet.m": {
        "status": "ported",
        "note": "The direct MATLAB image-processor getter surface is already covered by `ip_get(...)` / `ipGet(...)`.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/ipKeyPress.m": {
        "status": "out_of_scope",
        "note": "GUI keypress handlers remain outside the headless migration target.",
        "module_hits": [],
    },
    "imgproc/ipSet.m": {
        "status": "ported",
        "note": "The direct MATLAB image-processor setter surface is already covered by `ip_set(...)` / `ipSet(...)`.",
        "module_hits": ["pyisetcam.ip"],
    },
    "imgproc/vcimageKeyPress.m": {
        "status": "out_of_scope",
        "note": "GUI keypress handlers remain outside the headless migration target.",
        "module_hits": [],
    },
    "camera/cameraFullReference.m": {
        "status": "ported",
        "note": "The legacy MATLAB full-reference camera benchmark is covered by the Python `camera_full_reference(...)` / `cameraFullReference(...)` wrapper.",
        "module_hits": ["pyisetcam.camera"],
    },
    "camera/cameraPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a plotting gateway stub rather than a reusable supported headless camera API surface.",
        "module_hits": [],
    },
    "camera/cameraMoire.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an exploratory figure-driven moire-analysis workflow with interactive plots rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "camera/cameraVSNR_SL.m": {
        "status": "ported",
        "note": "The legacy MATLAB vSNR wrapper name is covered by the Python `camera_vsnr(...)` / `camera_vsnr_sl(...)` compatibility surface.",
        "module_hits": ["pyisetcam.camera"],
    },
    "camera/cameraWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file only opens MATLAB object windows and remains outside the headless migration target.",
        "module_hits": [],
    },
    "metrics/scielab/ApplyFilters.m": {
        "status": "ported",
        "note": "The legacy MATLAB filter-application helper is covered by the Python `sc_apply_filters(...)` / `ApplyFilters(...)` / `scApplyFilters(...)` compatibility surface.",
        "module_hits": ["pyisetcam.scielab", "pyisetcam.__init__"],
    },
    "metrics/scielab/gauss.m": {
        "status": "ported",
        "note": "The direct MATLAB Gaussian-kernel helper is covered by the Python `gauss(...)` compatibility wrapper in `pyisetcam.scielab`.",
        "module_hits": ["pyisetcam.scielab", "pyisetcam.__init__"],
    },
    "metrics/scielab/scApplyFilters.m": {
        "status": "ported",
        "note": "The legacy MATLAB S-CIELAB filter-application helper is covered by the Python `sc_apply_filters(...)` / `ApplyFilters(...)` / `scApplyFilters(...)` compatibility surface.",
        "module_hits": ["pyisetcam.scielab", "pyisetcam.__init__"],
    },
    "metrics/iePSNR.m": {
        "status": "ported",
        "note": "The direct MATLAB PSNR helper is covered by the Python `peak_signal_to_noise_ratio(...)` / `iePSNR(...)` compatibility surface.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "metrics/ISO/ISOspeedNoise.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a teaching script that expects GUI-managed scene/OI/sensor state rather than a standalone supported headless ISO API.",
        "module_hits": [],
    },
    "utility/tablebase/ieTableGet.m": {
        "status": "ported",
        "note": "The MATLAB metadata-table filter helper is covered by the Python `ie_table_get(...)` / `ieTableGet(...)` compatibility wrapper in `pyisetcam.ptable`.",
        "module_hits": ["pyisetcam.ptable", "pyisetcam.__init__"],
    },
    "color/ieColorTransform.m": {
        "status": "ported",
        "note": "The MATLAB sensor-to-target color transform helper is covered by the Python `ie_color_transform(...)` / `ieColorTransform(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.color", "pyisetcam.__init__"],
    },
    "color/ieLuminanceFromEnergy.m": {
        "status": "ported",
        "note": "The direct MATLAB luminance-from-energy helper is covered by the Python `luminance_from_energy(...)` / `ieLuminanceFromEnergy(...)` compatibility surface.",
        "module_hits": ["pyisetcam.color", "pyisetcam.__init__"],
    },
    "color/ieLuminanceFromPhotons.m": {
        "status": "ported",
        "note": "The direct MATLAB luminance-from-photons helper is covered by the Python `luminance_from_photons(...)` / `ieLuminanceFromPhotons(...)` compatibility surface.",
        "module_hits": ["pyisetcam.color", "pyisetcam.__init__"],
    },
    "color/transforms/colorTransformMatrixCreate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a script used to derive static transform matrices, not a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/optics/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only an index listing for the optics scripts and does not define a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/metrics/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only an index listing for the metrics scripts and does not define a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/development/s_sensorIRSimulation.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked `UNDER DEVELOPMENT` and depends on session/window-driven exploratory IR workflows rather than a supported reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/development/s_spectraNatural.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a scratch exploratory notebook-style script with no stable API contract, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "scripts/development/s_stereoFundamentals.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an unfinished figure-driven class tutorial marked as still not working, so it is treated as exploratory teaching code rather than actionable headless migration debt.",
        "module_hits": [],
    },
    "scripts/oneoverf/s_oneOverF1D.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a research-style exploratory synthesis notebook for 1/f intuition with interactive figures rather than a stable reusable MATLAB API surface.",
        "module_hits": [],
    },
    "scripts/oneoverf/s_oneoverf2D_tinker.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an explicit tinker script for FFT/cropping experiments and remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "scripts/oneoverf/s_oneoverf_tinker.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an explicit tinker script for dead-leaves-style spectrum experiments rather than a supported reusable API surface.",
        "module_hits": [],
    },
    "scripts/human/s_humanColorBlind.m": {
        "status": "parity",
        "note": "The Brettel/Vienot/Mollon color-blind rendering workflow is covered by the focused headless regression for `xyz2lms(..., cbType, whiteXYZ)`, `colorTransformMatrix('lms2xyz')`, `imageLinearTransform(...)`, and `xyz2srgb(...)` on a scene XYZ image.",
        "module_hits": ["pyisetcam.color", "pyisetcam.utils"],
    },
    "scripts/human/s_humanConeAbsorptions.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly deprecated and redirects users to later calibration/ISETBio workflows instead of exposing a maintained standalone ISETCam script surface.",
        "module_hits": [],
    },
    "scripts/human/s_humanDisplayPSF.m": {
        "status": "parity",
        "note": "The display point-spread workflow is covered by the focused headless regression that replays `displayCreate(...)`, `sceneFromFile(..., 'rgb', ..., display)`, `oiCreate('wvf')`, `sensorCreateConeMosaic(...)`, and `sensorCompute(...)` on a point-primary display scene.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.optics", "pyisetcam.sensor"],
    },
    "scripts/human/s_humanPhotonCalculator.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly deprecated and only prints a redirect to `s_calibrationPugh`, so it is treated as retired teaching material rather than actionable headless migration debt.",
        "module_hits": [],
    },
    "scripts/human/s_humanPupilSizeBlur.m": {
        "status": "out_of_scope",
        "note": "The upstream script immediately exits without ISETBio and is framed as exploratory analysis of a known pupil-size limitation, so it is treated as ISETBio-dependent notebook-style material rather than a supported standalone headless workflow.",
        "module_hits": [],
    },
    "scripts/human/s_humanSafetyBlueLight.m": {
        "status": "parity",
        "note": "The blue-light and related lamp-safety workflow is covered headlessly by the current `humanUVSafety(...)`, `ieLuminance2Radiance(...)`, `ieLuminanceFromEnergy(...)`, and `blackbody(...)` regression surface against the vendored safety-standard spectra.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.color", "pyisetcam.utils"],
    },
    "scripts/human/s_humanSafetyLuminance.m": {
        "status": "parity",
        "note": "The monochromatic-luminance safety workflow is covered headlessly by the current `ieLuminance2Radiance(...)` plus `humanUVSafety(...)` regression surface against the vendored actinic hazard spectra.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.color"],
    },
    "scripts/human/s_humanSafetyThermal.m": {
        "status": "parity",
        "note": "The thermal-hazard script path is covered headlessly by the current `humanUVSafety(...)` regression surface, which exercises the thermal safety modes together with the same radiance-to-irradiance contract used by the MATLAB workflow.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.color"],
    },
    "scripts/human/s_humanSafetyUVExposure.m": {
        "status": "parity",
        "note": "The UV-exposure workflow is covered headlessly by the current `humanUVSafety(...)`, `ieLuminance2Radiance(...)`, `ieLuminanceFromEnergy(...)`, and `blackbody(...)` regression surface across the actinic, eye, blue-hazard, and thermal safety branches.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.color", "pyisetcam.utils"],
    },
    "scripts/human/s_humanSceneStatistics.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately exits unless ISETBio's `Lens` class is available, so it is treated as an ISETBio-dependent exploratory script rather than a supported standalone ISETCam workflow.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/dctAlgorithm.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven JPEG/DCT teaching notebook that saves TIFFs and visualizes intermediate blocks rather than a reusable supported pyisetcam workflow.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/dctidct.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/entropy_file.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/jpegCoef.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/jpegRGB.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/jpeg_qtables.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/jpgread.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB shim over external JPEG MEX bindings, so it remains outside the supported headless Python migration target.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/jpgwrite.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only a MATLAB shim over external JPEG MEX bindings, so it remains outside the supported headless Python migration target.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/makeDctMatrix.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/jpegFiles/makeQTable.m": {
        "status": "out_of_scope",
        "note": "The upstream file is part of the standalone JPEG/DCT teaching helper bundle used by the out-of-scope JPEG tutorials, not the supported ISET object pipeline.",
        "module_hits": [],
    },
    "scripts/image/s_ipCircleMTF.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven exploratory Mackay-circle analysis script built around interactive FFT/circle inspection rather than a stable supported headless workflow surface.",
        "module_hits": [],
    },
    "scripts/image/s_ipDisplayResolution.m": {
        "status": "parity",
        "note": "The display-resolution sizing workflow is already covered by the current `ieN2MegaPixel(...)` regression surface and the documented sensor-versus-display megapixel sweep in the headless test suite.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "scripts/image/s_ipGamutReflectance.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an exploratory GUI-heavy reflectance-gamut experiment built around `imageMultiview(...)` and interactive viewing rather than a supported reusable headless workflow.",
        "module_hits": [],
    },
    "scripts/image/s_ipHC2RGB.m": {
        "status": "parity",
        "note": "The multispectral-waveband visualization workflow is already covered by the current `image_hc2rgb(...)` / `imagehc2rgb(...)` regression surface on both scene and optical-image objects.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "scripts/image/s_ipIlluminantCorrection.m": {
        "status": "parity",
        "note": "The illuminant-correction workflow is already covered by the current `imageSensorConversion(...)` and `imageIlluminantCorrection(...)` regression surface, including computed transform replay through headless IP objects.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__"],
    },
    "scripts/image/s_ipSRGB.m": {
        "status": "parity",
        "note": "The sRGB reference-display workflow is already covered by the current `xyz2srgb(...)`, `srgb2xyz(...)`, and related headless color-transform regression surface on scene XYZ data.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.color", "pyisetcam.__init__"],
    },
    "scripts/image/s_ipSensorConversion.m": {
        "status": "parity",
        "note": "The sensor-conversion workflow is already covered by the current `imageSensorConversion(...)` regression surface, including transform estimation and corrected-versus-desired sensor/XYZ comparisons.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__"],
    },
    "scripts/image/s_ipWrite.m": {
        "status": "out_of_scope",
        "note": "The upstream file is primarily a MATLAB `imwrite` usage example; the pyisetcam-specific part is already covered by headless `ipGet(..., 'srgb'/'result')`, while the file-write step itself is ordinary host-language image I/O rather than library migration debt.",
        "module_hits": [],
    },
    "tutorials/color/t_colorChromaticity.m": {
        "status": "parity",
        "note": "The chromaticity tutorial workflow is already covered by the current headless regression surface through `display_create/get(...)`, `chromaticity_xy(...)`, `xyz_from_energy(...)`, and `scene_create/get(...)` on the default Macbeth scene.",
        "module_hits": ["pyisetcam.color", "pyisetcam.metrics", "pyisetcam.display", "pyisetcam.scene"],
    },
    "tutorials/color/t_colorEnergyQuanta.m": {
        "status": "parity",
        "note": "The energy-versus-quanta tutorial workflow is already covered by the current headless regression surface through `quanta_to_energy(...)`, `energy_to_quanta(...)`, `ie_read_spectra(...)`, `xyz_from_energy(...)`, and `ie_xyz_from_photons(...)`.",
        "module_hits": ["pyisetcam.color", "pyisetcam.metrics"],
    },
    "tutorials/code/t_codeObjects.m": {
        "status": "parity",
        "note": "The deprecated object-database tutorial is covered by the optional `pyisetcam.session` compatibility layer and direct regressions for `ieInitSession(...)`, `ieAddObject(...)`, `ieGetObject(...)`, and `ieReplaceObject(...)`; MATLAB windows remain intentionally headless.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "tutorials/code/t_codeRendering.m": {
        "status": "parity",
        "note": "The spectral rendering tutorial is covered by the current headless regression surface through `blackbody(...)`, `xyz_from_energy(...)`, `XW2RGBFormat(...)`, `xyz2srgb(...)`, and `imageIncreaseImageRGBSize(...)`.",
        "module_hits": ["pyisetcam.color", "pyisetcam.utils", "pyisetcam.__init__"],
    },
    "tutorials/code/t_codeSESSION.m": {
        "status": "parity",
        "note": "The vcSESSION add/get tutorial is covered by the optional `pyisetcam.session` compatibility layer and direct regressions for `ieInitSession(...)`, `ieAddObject(...)`, `ieGetObject(...)`, and selected-object bookkeeping; window display remains intentionally headless.",
        "module_hits": ["pyisetcam.session", "pyisetcam.__init__"],
    },
    "tutorials/code/t_codeStartup.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a deprecated MATLAB path-management note about `startup.m`, not a reusable supported headless API surface.",
        "module_hits": [],
    },
    "tutorials/printing/t_printingHalftone.m": {
        "status": "parity",
        "note": "The printing tutorial is covered by the focused headless halftoning regression exercising `HalfToneImage(...)`, the Bayer-style threshold-cell workflow, and Floyd-Steinberg error diffusion.",
        "module_hits": ["pyisetcam.utils", "pyisetcam.__init__"],
    },
    "scripts/data/ieScratchData.m": {
        "status": "parity",
        "note": "The scratch-data bootstrap script is covered by the focused headless `scene -> oi -> sensor -> ip` initialization regression using default slanted-bar, optical-image, sensor, and IP objects.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.optics", "pyisetcam.sensor", "pyisetcam.ip"],
    },
    "scripts/faces/s_faceDetectionDemo.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately prints a TODO and returns, and the remaining code depends on MATLAB Vision Toolbox face-detection UI workflows rather than a supported reusable headless ISETCam surface.",
        "module_hits": [],
    },
    "tutorials/camera/t_SystemSimulate.m": {
        "status": "parity",
        "note": "The end-to-end system tutorial is covered by the current headless `scene -> oi -> sensor -> ip` workflow regressions together with the camera-introduction regression and the existing scene/OI/sensor/IP compatibility surface.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.optics", "pyisetcam.sensor", "pyisetcam.ip", "pyisetcam.camera"],
    },
    "tutorials/camera/t_cameraAntiAliasing.m": {
        "status": "parity",
        "note": "The anti-alias tutorial is covered by the headless frequency-orientation camera regression plus the underlying `oi_diffuser(...)` and `oiBirefringentDiffuser(...)` compatibility coverage, including diffuser replay through `oi_compute(...)`.",
        "module_hits": ["pyisetcam.scene", "pyisetcam.optics", "pyisetcam.sensor", "pyisetcam.ip"],
    },
    "tutorials/camera/t_cameraIntroduction.m": {
        "status": "parity",
        "note": "The camera-introduction tutorial is covered by the current `camera_create/get/set/compute(...)` regression surface, including direct `cameraGet(..., 'ip data srgb')` access and compute replay starting from `scene`, `oi`, and `sensor`.",
        "module_hits": ["pyisetcam.camera", "pyisetcam.ip"],
    },
    "tutorials/camera/t_cameraNoise.m": {
        "status": "parity",
        "note": "The camera-noise tutorial is covered by the seeded headless camera-noise regression, including `noise flag` control, luminance changes, and the default camera pipeline on an RGB scene file.",
        "module_hits": ["pyisetcam.camera", "pyisetcam.scene", "pyisetcam.display"],
    },
    "tutorials/image/t_ip.m": {
        "status": "parity",
        "note": "The introductory IP tutorial workflow is already covered by the current headless `scene -> oi -> sensor -> ip` regression path, including `MCC Optimized` sensor conversion, `gray world` illuminant correction, adaptive-Laplacian demosaic, and display-backed IP state.",
        "module_hits": ["pyisetcam.ip", "pyisetcam.__init__", "pyisetcam.display"],
    },
    "tutorials/image/t_ipJPEGMonochrome.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a standalone educational JPEG/DCT tutorial built around the legacy `scripts/image/jpegFiles` teaching helpers rather than the ISET object pipeline, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "tutorials/image/t_ipJPEGcolor.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a standalone educational JPEG color-compression tutorial built around the legacy `scripts/image/jpegFiles` teaching helpers rather than the ISET object pipeline, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "scripts/display/s_displayCompare.m": {
        "status": "parity",
        "note": "The display-comparison workflow is already covered by the current headless display/scene rendering surface and the existing display-driven `scene_from_file(...)` regression coverage across `OLED-Sony`, `LCD-Apple`, and `CRT-Dell`.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    },
    "scripts/display/s_displayReflectanceCtemp.m": {
        "status": "parity",
        "note": "The theoretical reflectance-display workflow is already covered by the current `display_reflectance(...)` / `displayReflectance(...)`, `scene_from_file(...)`, and `scene_adjust_illuminant(...)` compatibility surface with direct regression coverage.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    },
    "scripts/display/s_displaySurfaceReflectance.m": {
        "status": "parity",
        "note": "The MATLAB reflectance-display construction walkthrough is already exercised by the current headless `display_create/get/set(...)` and display-backed `scene_from_file(...)` workflow coverage, so the script is tracked as covered tutorial evidence rather than open API debt.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene"],
    },
    "scripts/utility/s_initSO.m": {
        "status": "out_of_scope",
        "note": "The upstream file only seeds `vcSESSION` with default scene/OI objects through `ieAddObject(...)`, so it remains a session-population script outside the explicit-object headless migration target.",
        "module_hits": [],
    },
    "tutorials/display/t_displayIntroduction.m": {
        "status": "parity",
        "note": "The introductory display tutorial is already covered by the current headless `display_create/get/set(...)` helper surface plus display-backed `scene_from_file(...)` and `scene_adjust_illuminant(...)` regression coverage.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene"],
    },
    "tutorials/display/t_displayRendering.m": {
        "status": "parity",
        "note": "The display-rendering tutorial is already covered by the current headless display rendering/accessor surface together with the multi-display `scene_from_file(...)` rendering workflow coverage used by the existing unit and parity-backed display regressions.",
        "module_hits": ["pyisetcam.display", "pyisetcam.scene", "pyisetcam.ip"],
    },
    "scripts/optics/chromAb/ChromAb.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a shell-driven external-binary workflow around `otf` and `pr_mat`, so it is treated as a legacy launcher rather than a supported headless Python API target.",
        "module_hits": [],
    },
    "scripts/optics/chromAb/makeCAplots.m": {
        "status": "out_of_scope",
        "note": "The upstream file is effectively a plotting/comment stub for the chromatic-aberration workflow and remains outside the headless migration target.",
        "module_hits": [],
    },
    "scripts/sensor/Contents.m": {
        "status": "out_of_scope",
        "note": "The upstream file is only an index listing for the sensor scripts and does not define a reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/sensor/pixel/s_pixelSizeDR.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a teaching script that explores pixel dynamic-range trends rather than a standalone reusable headless API surface.",
        "module_hits": [],
    },
    "scripts/sensor/readrawsensor/s_Raw2ISET.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an interactive file-picker walkthrough layered on top of LoadRawSensorData rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "metrics/ssim/s_metricsSSIM.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a teaching/regression script that drives an interactive camera-to-metrics walkthrough rather than a standalone reusable headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtBlockPartition.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a script-style diagnostic prototype with display side effects rather than a reusable headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtAngleLUT.m": {
        "status": "ported",
        "note": "The MATLAB angle-lookup helper is covered by the Python `rt_angle_lut(...)` / `rtAngleLUT(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtBlockCenter.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-center helper is covered by the Python `rt_block_center(...)` / `rtBlockCenter(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtChooseBlockSize.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-sizing helper is covered by the Python `rt_choose_block_size(...)` / `rtChooseBlockSize(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtDIInterp.m": {
        "status": "ported",
        "note": "The MATLAB geometric-distortion interpolation helper is covered by the Python `rt_di_interp(...)` / `rtDIInterp(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtExtractBlock.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-extraction helper is covered by the Python `rt_extract_block(...)` / `rtExtractBlock(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtFileNames.m": {
        "status": "ported",
        "note": "The MATLAB Zemax filename helper is covered by the Python `rt_file_names(...)` / `rtFileNames(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtGeometry.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace geometry stage is covered by the Python `rt_geometry(...)` / `rtGeometry(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtImagePSFFieldHeight.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked as not functional yet, so it remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtImageRotate.m": {
        "status": "ported",
        "note": "The MATLAB PSF image-rotation helper is covered by the Python `rt_image_rotate(...)` / `rtImageRotate(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtImportData.m": {
        "status": "ported",
        "note": "The MATLAB Zemax import helper is covered by the Python `rt_import_data(...)` / `rtImportData(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtInsertBlock.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace block-insertion helper is covered by the Python `rt_insert_block(...)` / `rtInsertBlock(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtOTF.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace OTF builder is covered by the Python `rt_otf(...)` / `rtOTF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFApply.m": {
        "status": "ported",
        "note": "The MATLAB shift-variant PSF application entry point is covered by the Python `rt_psf_apply(...)` / `rtPSFApply(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFEdit.m": {
        "status": "ported",
        "note": "The MATLAB PSF centering/rotation helper is covered by the Python `rt_psf_edit(...)` / `rtPSFEdit(...)` compatibility wrapper; visualization remains intentionally headless.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFGrid.m": {
        "status": "ported",
        "note": "The MATLAB PSF support-grid helper is covered by the Python `rt_psf_grid(...)` / `rtPSFGrid(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFInterp.m": {
        "status": "ported",
        "note": "The MATLAB ray-trace PSF interpolation helper is covered by the Python `rt_psf_interp(...)` / `rtPSFInterp(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPSFVisualize.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure/movie viewer for ray-trace PSFs and remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an interactive plotting gateway with dialogs and figure side effects rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "opticalimage/raytrace/rtPrecomputePSF.m": {
        "status": "ported",
        "note": "The MATLAB precomputed shift-variant PSF builder is covered by the Python `rt_precompute_psf(...)` / `rtPrecomputePSF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtPrecomputePSFApply.m": {
        "status": "ported",
        "note": "The MATLAB precomputed shift-variant PSF application helper is covered by the Python `rt_precompute_psf_apply(...)` / `rtPrecomputePSFApply(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtRIInterp.m": {
        "status": "ported",
        "note": "The MATLAB relative-illumination interpolation helper is covered by the Python `rt_ri_interp(...)` / `rtRIInterp(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtRootPath.m": {
        "status": "ported",
        "note": "The MATLAB vendored ray-trace root helper is covered by the Python `rt_root_path(...)` / `rtRootPath(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtSampleHeights.m": {
        "status": "ported",
        "note": "The MATLAB sample-height selection helper is covered by the Python `rt_sample_heights(...)` / `rtSampleHeights(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/rtSynthetic.m": {
        "status": "ported",
        "note": "The MATLAB synthetic ray-trace optics generator is covered by the Python `rt_synthetic(...)` / `rtSynthetic(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/zemax/zemax2json.m": {
        "status": "out_of_scope",
        "note": "The upstream file depends on a live Windows OpticStudio/.NET connection and external JSON-writing workflow, so it remains outside the supported headless migration target.",
        "module_hits": [],
    },
    "opticalimage/raytrace/zemax/zemaxLoad.m": {
        "status": "ported",
        "note": "The MATLAB Zemax PSF text-loader is covered by the Python `zemax_load(...)` / `zemaxLoad(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/raytrace/zemax/zemaxReadHeader.m": {
        "status": "ported",
        "note": "The MATLAB Zemax header parser is covered by the Python `zemax_read_header(...)` / `zemaxReadHeader(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "opticalimage/oiImageInitCustomStrings.m": {
        "status": "out_of_scope",
        "note": "The upstream file only initializes GUI/window label strings and remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/oiExtractMask.m": {
        "status": "out_of_scope",
        "note": "The pinned upstream file is empty, so it is treated as a non-actionable stub rather than a remaining headless API gap.",
        "module_hits": [],
    },
    "opticalimage/depth/oiDepthOverlay.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as obsolete and directs callers to `oiPlot(..., 'depth map contour', ...)` instead.",
        "module_hits": [],
    },
    "opticalimage/wavefront/RandomDirtyApertureDeprecate.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a deprecated Computer Vision Toolbox dirty-aperture generator rather than a reusable headless WVF API surface.",
        "module_hits": [],
    },
    "opticalimage/wavefront/psf/psfPlotrange.m": {
        "status": "out_of_scope",
        "note": "The upstream file only adjusts figure axis limits and title text for PSF plots, so it remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/wavefront/underDevelopment_wavefront/wvfComputeOptimizedConePSF.m": {
        "status": "out_of_scope",
        "note": "The upstream file lives under underDevelopment_wavefront and remains an experimental prototype outside the supported headless target.",
        "module_hits": [],
    },
    "opticalimage/wavefront/underDevelopment_wavefront/wvfComputeOptimizedPSF.m": {
        "status": "out_of_scope",
        "note": "The upstream file lives under underDevelopment_wavefront and remains an experimental prototype outside the supported headless target.",
        "module_hits": [],
    },
    "opticalimage/optics/airyDiskPlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly deprecated and only adjusts live MATLAB figure state for an existing PSF plot.",
        "module_hits": [],
    },
    "opticalimage/optics/defocusMTF.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as obsolete and directs callers to newer defocus/OTF helpers instead.",
        "module_hits": [],
    },
    "opticalimage/optics/psfMovie.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven animation helper, which remains outside the headless migration target.",
        "module_hits": [],
    },
    "opticalimage/optics/defocus/opticsReducedSFandW20.m": {
        "status": "out_of_scope",
        "note": "The upstream helper is internally incomplete, references undefined state, and does not expose a usable supported headless API surface.",
        "module_hits": [],
    },
    "human/humanAchromaticOTF.m": {
        "status": "ported",
        "note": "The legacy MATLAB human achromatic OTF helper is covered by the Python `human_achromatic_otf(...)` / `humanAchromaticOTF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanCore.m": {
        "status": "ported",
        "note": "The legacy MATLAB wavelength-dependent human OTF core helper is covered by the Python `human_core(...)` / `humanCore(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanLSF.m": {
        "status": "ported",
        "note": "The legacy MATLAB human line-spread helper is covered by the Python `human_lsf(...)` / `humanLSF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanOTF.m": {
        "status": "ported",
        "note": "The legacy MATLAB chromatic human OTF helper is covered by the Python `human_otf(...)` / `humanOTF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanPupilSize.m": {
        "status": "ported",
        "note": "The legacy MATLAB pupil-size helper is covered by the Python `human_pupil_size(...)` / `humanPupilSize(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/humanSpaceTime.m": {
        "status": "ported",
        "note": "The legacy MATLAB spatiotemporal sensitivity dispatcher is covered by the Python `human_space_time(...)` / `humanSpaceTime(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/ijspeert.m": {
        "status": "ported",
        "note": "The legacy MATLAB IJspeert ocular-scatter helper is covered by the Python `ijspeert(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/kellySpaceTime.m": {
        "status": "ported",
        "note": "The legacy MATLAB Kelly spatiotemporal sensitivity helper is covered by the Python `kelly_space_time(...)` / `kellySpaceTime(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/poirsonSpatioChromatic.m": {
        "status": "ported",
        "note": "The legacy MATLAB Poirson spatiochromatic sensitivity helper is covered by the Python `poirson_spatio_chromatic(...)` / `poirsonSpatioChromatic(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/watsonImpulseResponse.m": {
        "status": "ported",
        "note": "The legacy MATLAB Watson temporal impulse-response helper is covered by the Python `watson_impulse_response(...)` / `watsonImpulseResponse(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/watsonRGCSpacing.m": {
        "status": "ported",
        "note": "The legacy MATLAB Watson retinal-ganglion-cell spacing helper is covered by the Python `watson_rgc_spacing(...)` / `watsonRGCSpacing(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/westheimerLSF.m": {
        "status": "ported",
        "note": "The legacy MATLAB Westheimer line-spread approximation is covered by the Python `westheimer_lsf(...)` / `westheimerLSF(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/humanConeContrast.m": {
        "status": "ported",
        "note": "The legacy MATLAB Stockman cone-contrast helper is covered by the Python `human_cone_contrast(...)` / `humanConeContrast(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/humanConeIsolating.m": {
        "status": "ported",
        "note": "The legacy MATLAB display cone-isolating helper is covered by the Python `human_cone_isolating(...)` / `humanConeIsolating(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/humanMacularTransmittance.m": {
        "status": "ported",
        "note": "The legacy MATLAB macular-pigment transmittance updater is covered by the Python `human_macular_transmittance(...)` / `humanMacularTransmittance(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanOpticalDensity.m": {
        "status": "ported",
        "note": "The legacy MATLAB human optical-density parameter helper is covered by the Python `human_optical_density(...)` / `humanOpticalDensity(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/humanOTF_ibio.m": {
        "status": "ported",
        "note": "The legacy MATLAB ISETBio OTF-storage variant is covered by the Python `human_otf_ibio(...)` / `humanOTF_ibio(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanOI.m": {
        "status": "ported",
        "note": "The deprecated MATLAB human optical-image helper is covered by the Python `human_oi(...)` / `humanOI(...)` compatibility wrapper on top of the headless optical-image compute path.",
        "module_hits": ["pyisetcam.optics", "pyisetcam.__init__"],
    },
    "human/humanUVSafety.m": {
        "status": "ported",
        "note": "The legacy MATLAB UV and blue-light safety helper is covered by the Python `human_uv_safety(...)` / `humanUVSafety(...)` compatibility wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "human/ieConePlot.m": {
        "status": "ported",
        "note": "The legacy MATLAB cone-mosaic image helper is covered by the Python `ie_cone_plot(...)` / `ieConePlot(...)` headless payload wrapper.",
        "module_hits": ["pyisetcam.metrics", "pyisetcam.__init__"],
    },
    "sensor/sensorUnitBlock.m": {
        "status": "out_of_scope",
        "note": "The upstream file is explicitly marked obsolete and does not implement a usable headless API surface.",
        "module_hits": [],
    },
    "sensor/simulation/noise/noiseDarkCurrent.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors as obsolete and does not implement a usable headless API surface.",
        "module_hits": [],
    },
    "sensor/human/sensorConePlot.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven visualization helper for cone mosaics and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/sensor2EXR.m": {
        "status": "out_of_scope",
        "note": "The upstream file depends on MATLAB OpenEXR I/O and a network-training export workflow rather than a supported headless Python API surface.",
        "module_hits": [],
    },
    "sensor/models/s_sensorIMX490Test.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven exploratory IMX490 demo script rather than a reusable supported headless API surface.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesign.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE figure entry point for interactive CFA editing and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesignCallbacks.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE callback/controller layer for the CFA design UI and does not define a reusable supported headless API surface.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesignUI.m": {
        "status": "out_of_scope",
        "note": "The upstream file constructs an interactive GUIDE CFA design window and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/cfaDesign/cfaDesignUtilities.m": {
        "status": "out_of_scope",
        "note": "The upstream file is an explicitly unimplemented example stub for CFA-design scripting rather than a usable supported headless API surface.",
        "module_hits": [],
    },
    "sensor/pixel/pixelGUI/pixelGeometryWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a legacy MATLAB figure window controller and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/pixel/pixelGUI/pixelOEWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a legacy MATLAB figure window controller and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/microLensWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE figure window controller and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlOpen.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUIDE window-opening callback and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlRefresh.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a GUI refresh helper for the microlens window and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlFillMLFromWindow.m": {
        "status": "out_of_scope",
        "note": "The upstream file only pulls values from GUIDE controls into a microlens structure and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlFillWindowFromML.m": {
        "status": "out_of_scope",
        "note": "The upstream file only pushes microlens values into GUIDE controls and remains outside the headless migration target.",
        "module_hits": [],
    },
    "sensor/microlens/mlIrradianceImage.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a figure-driven irradiance-image display helper and remains outside the headless migration target.",
        "module_hits": [],
    },
    "scene/pattern/_tests_/sceneFluorescenceChart_test.m": {
        "status": "out_of_scope",
        "note": "The upstream file is a private MATLAB regression script for the fluorescence-chart helper rather than a standalone headless API surface.",
        "module_hits": [],
    },
    "scene/pattern/sceneFluorescenceChart.m": {
        "status": "out_of_scope",
        "note": "The pinned upstream snapshot does not vendor the required `fluorescenceSignal` / `fluorescenceWeights` model helpers, so this chart constructor depends on external fluorescence-model code outside the supported ISETCam migration target.",
        "module_hits": [],
    },
    "scene/depth/sceneDepthOverlay.m": {
        "status": "out_of_scope",
        "note": "The upstream file immediately errors and redirects callers to `scenePlot`, so it is treated as obsolete rather than actionable headless API debt.",
        "module_hits": [],
    },
}


@dataclass(frozen=True)
class InventoryEntry:
    upstream_path: str
    family: str
    kind: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _canonical_signal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _signal_min_length(token: str) -> int:
    normalized = _canonical_signal(token)
    if normalized.startswith(SHORT_SIGNAL_PREFIXES):
        return 4
    return 6


def _json_yaml_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False) + "\n"


def _case_definitions() -> list[dict[str, Any]]:
    return json.loads(_read_text(CASES_PATH)).get("cases", [])


def _latest_upstream_snapshot(root: Path) -> Path:
    candidates = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not candidates:
        raise FileNotFoundError(f"No upstream snapshots found in {root}")
    return candidates[-1]


def _family_for_relpath(rel_path: Path) -> str:
    parts = [part.lower() for part in rel_path.parts]
    first = parts[0]
    if first in {"scripts", "tutorials", "data"} and len(parts) > 1:
        return f"{first}/{parts[1]}"
    if (
        first in {"opticalimage", "scene", "sensor", "imgproc", "metrics", "utility"}
        and len(parts) > 1
    ):
        second = parts[1]
        if second in {
            "binning",
            "cielab",
            "cfadesign",
            "depth",
            "file",
            "gui",
            "illumination",
            "image",
            "imgtargets",
            "iso",
            "macbeth",
            "metricsgui",
            "microlens",
            "models",
            "openexr",
            "optics",
            "pattern",
            "pixel",
            "plots",
            "raytrace",
            "reflectance",
            "scielab",
            "simulation",
            "tablebase",
            "wavefront",
        }:
            return f"{first}/{second}"
    return first


def _kind_for_relpath(rel_path: Path, family: str) -> str:
    suffix = rel_path.suffix.lower()
    parts = [part.lower() for part in rel_path.parts]
    if family.startswith("data/"):
        return "asset"
    if suffix in {".mlapp", ".fig"}:
        return "gui"
    if "gui" in parts or family.endswith("/gui") or parts[0] == "gui":
        return "gui"
    if parts[0] == "scripts":
        return "script"
    if parts[0] == "tutorials":
        return "tutorial"
    if parts[0] in {"utility", "main", "local", "cp"}:
        return "helper"
    return "api"


def _inventory(snapshot: Path) -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        rel_path = path.relative_to(snapshot)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_CODE_SUFFIXES:
            continue
        family = _family_for_relpath(rel_path)
        kind = _kind_for_relpath(rel_path, family)
        entries.append(InventoryEntry(upstream_path=rel_path.as_posix(), family=family, kind=kind))

    data_root = snapshot / "data"
    if data_root.exists():
        for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
            rel = child.relative_to(snapshot)
            entries.append(
                InventoryEntry(
                    upstream_path=f"{rel.as_posix()}/",
                    family=_family_for_relpath(rel),
                    kind="asset",
                )
            )
    return entries


def _family_surface(family: str) -> list[str]:
    for candidate in sorted(FAMILY_SURFACE_MAP, key=len, reverse=True):
        if family == candidate or family.startswith(f"{candidate}/"):
            return FAMILY_SURFACE_MAP[candidate]
    return []


def _family_tests(family: str) -> list[str]:
    for candidate in sorted(FAMILY_TEST_MAP, key=len, reverse=True):
        if family == candidate or family.startswith(f"{candidate}/"):
            return FAMILY_TEST_MAP[candidate]
    return []


def _module_texts() -> dict[str, dict[str, str]]:
    module_texts: dict[str, dict[str, str]] = {}
    for path in sorted(SRC_ROOT.glob("*.py")):
        raw_text = _read_text(path).lower()
        module_texts[path.stem] = {
            "raw": raw_text,
            "canonical": _canonical_signal(raw_text),
        }
    return module_texts


def _module_hits(stem: str, module_texts: dict[str, dict[str, str]]) -> list[str]:
    if len(_canonical_signal(stem)) < _signal_min_length(stem):
        return []
    canonical_stem = _canonical_signal(stem)
    hits = [
        f"pyisetcam.{name}"
        for name, text_bundle in module_texts.items()
        if canonical_stem in text_bundle["canonical"]
    ]
    return hits[:4]


def _combined_test_text() -> str:
    chunks = [_read_text(CASES_PATH).lower()]
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        chunks.append(_read_text(path).lower())
    return "\n".join(chunks)


def _docs_text() -> str:
    return "\n".join((_read_text(README_PATH), _read_text(IMPLEMENTATION_PLAN_PATH))).lower()


def _contains_signal(text: str, token: str) -> bool:
    if len(_canonical_signal(token)) < _signal_min_length(token):
        return False
    if token in text:
        return True
    return _canonical_signal(token) in _canonical_signal(text)


def _status_and_note(
    entry: InventoryEntry,
    *,
    docs_text: str,
    tests_text: str,
    case_text: str,
    module_texts: dict[str, dict[str, str]],
) -> tuple[str, str, list[str]]:
    basename = Path(entry.upstream_path).name.lower()
    stem = Path(entry.upstream_path.rstrip("/")).stem.lower()
    surfaces = _family_surface(entry.family)
    module_hits = _module_hits(stem, module_texts)
    direct_doc_hit = basename in docs_text
    explicit_hit = (
        direct_doc_hit
        or _contains_signal(docs_text, stem)
        or _contains_signal(tests_text, stem)
        or _contains_signal(case_text, stem)
    )
    has_surface = bool(surfaces)

    override = UPSTREAM_STATUS_OVERRIDES.get(entry.upstream_path)
    if override is not None:
        return (
            str(override["status"]),
            str(override["note"]),
            list(override.get("module_hits", [])),
        )

    if entry.kind == "gui":
        return (
            "out_of_scope",
            "GUI/App Designer and figure fidelity remain outside the headless migration target.",
            module_hits,
        )
    if entry.kind == "asset":
        return (
            "partial",
            "Runtime still relies on the upstream asset bundle rather than a fully migrated "
            "Python-native asset pipeline.",
            module_hits,
        )
    if entry.upstream_path.startswith("metrics/ISO/sfrmat4v5/"):
        if stem in {"sfrmat4", "sfrmat4simplified", "clip"}:
            return (
                "ported",
                "The MATLAB-private sfrmat4v5 algorithm helper is already absorbed into the "
                "current headless slanted-edge implementation in `pyisetcam.iso`.",
                ["pyisetcam.iso"],
            )
        return (
            "out_of_scope",
            "The remaining sfrmat4v5 file-I/O and GUI support helpers are not migrated as "
            "standalone headless Python APIs.",
            module_hits,
        )
    if entry.upstream_path.startswith("utility/external/facetracker/"):
        return (
            "out_of_scope",
            "The vendored Oxford VGG facetracker package is a standalone external MATLAB "
            "video-processing stack built around ffmpeg frame extraction, classdef packages, "
            "pretrained detector models, `parfor`, `VideoReader`, and multiple MEX/C++ "
            "extensions, so it is treated as third-party toolbox infrastructure rather than a "
            "supported pyisetcam runtime surface.",
            module_hits,
        )
    if entry.family.startswith("opticalimage/raytrace"):
        if explicit_hit and entry.kind in {"script", "tutorial"}:
            return (
                "parity",
                "Explicit ray-trace script parity exists, but the broader ray-trace family "
                "remains partial.",
                module_hits,
            )
        if module_hits or has_surface:
            return (
                "partial",
                "Ray-trace support exists in Python, but the family is still documented as "
                "partial.",
                module_hits,
            )
        return (
            "missing",
            "No direct Python ray-trace surface was detected for this upstream path.",
            module_hits,
        )
    if explicit_hit:
        if entry.kind in {"script", "tutorial"}:
            return (
                "parity",
                "Curated parity or direct script/tutorial evidence exists for this upstream "
                "workflow.",
                module_hits,
            )
        if entry.kind == "helper" and any(keyword in stem for keyword in HEADLESS_KEYWORDS):
            return (
                "headless_only",
                "The MATLAB workflow is covered through headless/session wrappers instead of "
                "GUI fidelity.",
                module_hits,
            )
        return (
            "parity",
            "The upstream API surface is referenced by the current parity-backed migration "
            "coverage.",
            module_hits,
        )
    if module_hits:
        if (
            entry.kind == "helper"
            or any(keyword in stem for keyword in HEADLESS_KEYWORDS)
            or entry.family.startswith("main")
        ):
            return (
                "headless_only",
                "A headless Python/session wrapper exists, but not full MATLAB GUI parity.",
                module_hits,
            )
        return (
            "ported",
            "A direct Python implementation surface was detected for this upstream path.",
            module_hits,
        )
    if has_surface:
        return (
            "partial",
            "The family has a Python surface, but this specific path lacks explicit parity or "
            "direct implementation evidence.",
            module_hits,
        )
    return ("missing", "No usable Python surface was detected for this upstream path.", module_hits)


def _priority_for(entry: InventoryEntry, status: str) -> str:
    if status in {"parity", "ported", "out_of_scope"}:
        return "P3"
    if status == "headless_only":
        return "P2"
    family = entry.family
    if family in CORE_PRIORITY_FAMILIES or any(
        family.startswith(f"{prefix}/") for prefix in CORE_PRIORITY_FAMILIES
    ):
        return "P0"
    if family in MID_PRIORITY_FAMILIES or any(
        family.startswith(f"{prefix}/") for prefix in MID_PRIORITY_FAMILIES
    ):
        return "P1"
    if "raytrace" in family or family.startswith("main") or family.startswith("utility/"):
        return "P2"
    if entry.kind in {"script", "tutorial"}:
        return "P1"
    return "P2"


def _python_surface(entry: InventoryEntry, module_hits: list[str]) -> str:
    surfaces = _family_surface(entry.family)
    merged: list[str] = []
    for item in surfaces + module_hits:
        if item not in merged:
            merged.append(item)
    return ", ".join(merged)


def _tests_for(entry: InventoryEntry, status: str) -> list[str]:
    tests = list(_family_tests(entry.family))
    if status == "parity":
        if "tests/parity/cases.yaml" not in tests:
            tests.append("tests/parity/cases.yaml")
        if "tests/parity/test_parity_harness.py" not in tests:
            tests.append("tests/parity/test_parity_harness.py")
    return tests


def _status_counter(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(entry["status"] for entry in entries)
    return dict(sorted(counter.items()))


def _kind_counter(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(entry["kind"] for entry in entries)
    return dict(sorted(counter.items()))


def _priority_counter(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(entry["priority"] for entry in entries)
    return dict(sorted(counter.items()))


def _family_summary(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["family"]].append(entry)
    summary: dict[str, dict[str, Any]] = {}
    for family, family_entries in sorted(grouped.items()):
        summary[family] = {
            "total": len(family_entries),
            "by_status": _status_counter(family_entries),
            "by_kind": _kind_counter(family_entries),
            "highest_open_priority": min(
                (
                    entry["priority"]
                    for entry in family_entries
                    if entry["status"] not in {"parity", "ported", "out_of_scope"}
                ),
                default="P3",
            ),
        }
    return summary


def _delta_counts(current: dict[str, int], previous: dict[str, int] | None) -> dict[str, int]:
    previous = previous or {}
    keys = sorted(set(current) | set(previous))
    return {key: int(current.get(key, 0) - previous.get(key, 0)) for key in keys}


def _load_previous_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError:
        return None


def _exception_site_summary() -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for path in sorted(SRC_ROOT.glob("*.py")):
        text = _read_text(path)
        summary[path.stem] = {
            "unsupported_option_error_sites": len(re.findall(r"UnsupportedOptionError\(", text)),
            "not_implemented_error_sites": len(re.findall(r"NotImplementedError\(", text)),
        }
    return dict(sorted(summary.items()))


def _top_open_families(
    family_summary: dict[str, dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = []
    for family, summary in family_summary.items():
        open_count = sum(
            count
            for status, count in summary["by_status"].items()
            if status not in {"parity", "ported", "out_of_scope"}
        )
        rows.append(
            {
                "family": family,
                "open_entries": open_count,
                "highest_open_priority": summary["highest_open_priority"],
                "by_status": summary["by_status"],
            }
        )
    rows.sort(
        key=lambda item: (
            item["highest_open_priority"],
            -item["open_entries"],
            item["family"],
        )
    )
    return rows[:limit]


def _ledger_entries(snapshot: Path) -> list[dict[str, Any]]:
    docs_text = _docs_text()
    tests_text = _combined_test_text()
    case_text = _read_text(CASES_PATH).lower()
    module_texts = _module_texts()
    entries: list[dict[str, Any]] = []
    for inventory_entry in _inventory(snapshot):
        status, notes, module_hits = _status_and_note(
            inventory_entry,
            docs_text=docs_text,
            tests_text=tests_text,
            case_text=case_text,
            module_texts=module_texts,
        )
        entries.append(
            {
                "upstream_path": inventory_entry.upstream_path,
                "family": inventory_entry.family,
                "kind": inventory_entry.kind,
                "status": status,
                "python_surface": _python_surface(inventory_entry, module_hits),
                "tests": _tests_for(inventory_entry, status),
                "notes": notes,
                "priority": _priority_for(inventory_entry, status),
            }
        )
    return entries


def _build_outputs(
    snapshot: Path,
    previous_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = _ledger_entries(snapshot)
    family_summary = _family_summary(entries)
    status_counts = _status_counter(entries)
    kind_counts = _kind_counter(entries)
    priority_counts = _priority_counter(entries)
    previous_summary = {} if previous_report is None else previous_report.get("summary", {})
    previous_status_counts = previous_summary.get("status_counts")
    previous_kind_counts = previous_summary.get("kind_counts")
    previous_priority_counts = previous_summary.get("priority_counts")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream_snapshot": snapshot.as_posix(),
        "entry_count": len(entries),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "priority_counts": priority_counts,
        "delta_from_previous": {
            "status_counts": _delta_counts(status_counts, previous_status_counts),
            "kind_counts": _delta_counts(kind_counts, previous_kind_counts),
            "priority_counts": _delta_counts(priority_counts, previous_priority_counts),
        },
        "family_summary": family_summary,
        "top_open_families": _top_open_families(family_summary),
        "exception_sites_by_module": _exception_site_summary(),
    }
    ledger = {
        "generated_at": summary["generated_at"],
        "upstream_snapshot": summary["upstream_snapshot"],
        "summary": {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "generated_at",
                "upstream_snapshot",
            }
        },
        "entries": entries,
    }
    report = {
        "generated_at": summary["generated_at"],
        "upstream_snapshot": summary["upstream_snapshot"],
        "summary": summary,
    }
    return ledger, report


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Migration gap audit")
    print(f"  upstream snapshot: {report['upstream_snapshot']}")
    print(f"  entries: {summary['entry_count']}")
    print(f"  status counts: {summary['status_counts']}")
    print(f"  kind counts: {summary['kind_counts']}")
    print(f"  priority counts: {summary['priority_counts']}")
    print("  top open families:")
    for row in summary["top_open_families"][:10]:
        print(
            f"    - {row['family']}: {row['open_entries']} open, "
            f"highest {row['highest_open_priority']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-snapshot", type=Path, default=None)
    parser.add_argument("--ledger-out", type=Path, default=LEDGER_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    parser.add_argument("--no-write-ledger", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    snapshot = args.upstream_snapshot or _latest_upstream_snapshot(UPSTREAM_ROOT)
    previous_report = _load_previous_report(args.report_out)
    ledger, report = _build_outputs(snapshot, previous_report)

    if not args.no_write_ledger:
        args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_out.write_text(_json_yaml_dump(ledger), encoding="utf-8")
    if not args.no_write_report:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
