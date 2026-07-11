# pyisetcam Development Status Archive

> This document preserves the detailed pyisetcam porting history that previously
> occupied the repository front page. New users should start with the
> [CameraE2E Integrated README](../README.md).

This checkout is also packaged as the CameraE2E integrated repository. See
[INTEGRATED_REPOSITORY.md](../INTEGRATED_REPOSITORY.md) for the self-contained
FDTD/TCAD/RayOptics/Camera-DB layout and validation contract.

`pyisetcam` is a greenfield Python port of the core ISETCam numerical pipeline:

- `scene -> optical image -> sensor -> image processor -> camera`
- pinned to upstream ISETCam commit `412b9f9bdb3262f2552b96f0e769b5ad6cdff821`
- validated through a GNU Octave parity harness for curated cases

## Start Here

- [Install Guide](install-guide.md)
- [Getting Started Tutorial](tutorial.md)
- [MATLAB To Python Mapping](migration.md)
- [HW ISP Simulation](hw-isp-simulation.md)
- [FDTD/TCAD-Informed Sensor Block](fdtd-sensor.md)
- [CameraE2E Technical Overview](camerae2e-technical-overview.md)
- [CameraE2E Goal Roadmap](camerae2e-goal-roadmap.md)
- [Perception Metrics](perception.md)
- [Task Perception](task-perception.md)

Current expansion notes:

- with the broad-parity expansion now started by adding post-core scene-family cases, utility-helper cases for `unitFrequencyList`, `Energy2Quanta/Quanta2Energy` in vector and matrix form, `blackbody` energy/quanta, and `ieParamFormat`, plus initial metrics-family cases for `ieXYZFromEnergy`, `xyz2luv`, `ieXYZ2LAB`, `xyz2uv`, `cct`, `deltaEab` (1976), and `metricsSPD` angle/CIELAB/mired
- with Phase 3 scene-script coverage now started by adding MATLAB-style `spd_to_cct` / `spd2cct` plus curated Octave parity on the `s_sceneCCT.m` blackbody-CCT workflow
- and now also MATLAB-style `daylight(...)` plus curated Octave parity on the `s_sceneDaylight.m` daylight-SPD and daylight-basis workflow
- and now also headless `illuminantCreate/Get/Set`, `illuminantModernize`, and `illuminantRead` plus curated Octave parity on the `s_sceneIlluminant.m` illuminant-structure workflow
- and now also headless `sceneIlluminantSS(...)`, spatial-spectral `sceneAdjustIlluminant(...)`, and `sceneCreate('macbeth tungsten')` coverage plus curated Octave parity on the `s_sceneIlluminantMixtures.m` mixed-illuminant workflow
- and now also curated Octave parity on the `s_sceneIlluminantSpace.m` spatial-spectral illuminant workflow, covering the row-temperature gradient and the column-harmonic illuminant modulation path
- and now also headless `scenePlot(..., 'illuminant image')`, matching MATLAB's spectral-to-uniform-image fallback plus spatial-spectral illuminant rendering through returned `srgb` payloads without opening a figure window
- and now also `sceneGet(..., 'xyz')` plus MATLAB-style `RGB2XWFormat` / `XW2RGBFormat` coverage with curated Octave parity on the `s_sceneXYZilluminantTransforms.m` illuminant-transfer workflow, locking down normalized XYZ balance plus the fitted transfer matrices and relative error trend
- and now also curated Octave parity on the `s_sceneFromRGB.m` display-calibrated RGB-image workflow, covering `displayCreate('LCD-Apple.mat')`, the display white point, `sceneFromFile(..., 'rgb', ..., 'LCD-Apple.mat')`, the 6500 K illuminant swap, and the beak ROI reflectance contract
- and now also curated Octave parity on the `s_sceneFromMultispectral.m` multispectral-file workflow, covering `sceneFromFile(..., 'multispectral', ..., 400:10:700)` on `StuffedAnimals_tungsten-hdrs`, the loaded scene geometry and mean luminance, and normalized mean/center scene spectra
- and now also headless `sceneGet(..., 'rgb')` plus curated Octave parity on the `s_sceneFromRGBvsMultispectral.m` roundtrip workflow, covering the `StuffedAnimals_tungsten-hdrs` multispectral scene rendered to RGB and reconstructed through `LCD-Apple.mat`, with preserved geometry/luminance, preserved 6500 K illuminant chromaticity, and strong per-channel RGB/XYZ roundtrip correlations
- and now also headless `ie_reflectance_samples(...)` / `ieReflectanceSamples(...)` plus curated Octave parity on the `s_sceneReflectanceSamples.m` workflow, covering no-replacement reflectance sampling, exact sample-list replay, and the deterministic normalized-mean / singular-value statistics on explicit reflectance lists
- and now also headless `hc_basis(...)` / `hcBasis(...)` plus curated Octave parity on the `s_sceneReflectanceChartBasisFunctions.m` workflow, covering canonical reflectance-chart basis extraction at `bType=0.999`, `0.95`, and `5`, with parity on the selected basis counts, variance explained, sign-canonicalized basis functions, and coefficient summary statistics
- and now also headless `scene_reflectance_chart(...)` / `sceneReflectanceChart(...)` plus curated Octave parity on the `s_sceneReflectanceCharts.m` workflow, covering the default Natural-100 chart, the explicit Munsell/Food/Dupont/Hyspex no-replacement chart, the D65 illuminant swap, the gray-strip variant, and exact replay of the replica chart from returned stored samples
- and now also curated Octave parity on the `s_sceneChangeIlluminant.m` workflow, covering the default Macbeth-to-tungsten swap together with the StuffedAnimals multispectral scene converted to equal-energy and Horizon illuminants, with parity on preserved mean luminance, normalized illuminant spectra, and rendered-RGB mean summaries
- and now also curated Octave parity on the `s_sceneDataExtractionAndPlotting.m` workflow, covering the default `macbethd65` scene line/ROI extraction path, the returned luminance-hline support and profile, the attached illuminant-energy spectrum, the yellow-patch radiance and reflectance ROI summaries, and exact agreement between `scenePlot(...)` ROI spectra and manual `vcGetROIData(...)` means
- and now also curated Octave parity on the `s_sceneMonochrome.m` workflow, covering `displayCreate('crt')`, `sceneFromFile('cameraman.tif', 'monochrome', 100, 'crt')`, the source display-white illuminant and monochrome scene spectral summaries, and the D65-adjusted unispectral scene plus rendered-RGB mean contract
- and now also `scene_plot(..., 'illuminant energy roi')` support without an explicit ROI, matching upstream `scenePlot(..., [], ...)` fallback behavior, plus curated Octave parity on the `s_sceneSlantedBar.m` workflow, covering the equal-photons and D65 slanted-bar illuminant spectra together with the two slanted-bar constructions’ geometry, mean luminance, and normalized center-row/center-column luminance profiles
- and now also curated Octave parity on the `s_sceneHarmonics.m` workflow, covering the four exact harmonic constructions from the script: the basic 1 cpi pattern, the two-frequency sum with phase offset, the crossed-orientation 2-and-5 cpi pair, and the symmetric 5-and-5 cpi pair, with parity on scene geometry, preserved mean luminance, and normalized center-row/center-column luminance profiles
- and now also headless `sceneCreate('zone plate', ...)` coverage plus curated Octave parity on the `s_sceneZonePlate.m` workflow, covering the equal-photon zone-plate generator, its default `4 deg` field of view, and the clipped hyperspectral photon cube
- and now also headless `sceneCreate('dead leaves', ...)` coverage plus curated Octave parity on the dead-leaves scene family, covering the OLED-Sony display-rendered grayscale target, default `10 deg` field of view, and deterministic dead-leaves photon cube under a fixed sample stream
- and now also MATLAB-style empty-placeholder handling in the public `sceneCreate('l star', ...)` and `sceneCreate('dead leaves', ...)` wrappers, so forms like `sceneCreate('l star', [], [], [])` and `sceneCreate('dead leaves', [], [], [])` reuse the documented dispatcher defaults instead of crashing on `int([])` or `float([])`
- and now also MATLAB-style empty-list placeholder handling inside dict-backed scene parameter payloads, so calls like `sceneCreate('frequency orientation', struct('blockSize',[]))`, `sceneCreate('harmonic', struct('row',[]))`, `sceneCreate('hdr lights', struct('radius',[]))`, and `sceneCreate('moire orient', struct('sceneSize',[]))` fall back to the same defaults as the no-argument wrappers instead of crashing
- and now also MATLAB-style empty-placeholder handling in the human sensor parameter lane, so `sensorCreate('human', struct('sz',[],'rgbDensities',[],'coneAperture',[],'rSeed',[],'wave',[]))` and `sensorCreateConeMosaic(sensor,[],[],[],[])` reuse the upstream defaults instead of crashing or producing an empty wave grid
- and now also MATLAB-style empty-wave placeholder handling in `pixelCreate(...)`, so calls like `pixelCreate('human', [])` and `pixelCreate('ideal', [], pixelSizeM)` reuse the default `400:10:700` wavelength grid instead of creating an empty spectrum
- and now also standalone `human_cone_mosaic(...)` / `humanConeMosaic(...)` coverage, including MATLAB-style empty-placeholder replay for optional densities, cone width, and seed inputs plus the upstream three-element LMS-density patch-up to `[K,L,M,S]`
- and now also standalone `human_wave_defocus(...)` / `humanWaveDefocus(...)` coverage, exposing the existing Thibos chromatic-defocus formula directly as a public helper instead of only through downstream human-optics code paths
- and now also standalone `human_cones(...)` / `humanCones(...)` coverage, exposing the vendored Stockman cone-fundamental assets together with the MATLAB macular-density correction contract and empty-placeholder defaults
- and now also direct module-level `pyisetcam.metrics` alias coverage for `humanCones`, so callers can import the MATLAB-style cone-fundamental helper name from the metrics module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.optics` for the adjacent human helper family, so imports like `from pyisetcam.optics import humanWaveDefocus`, `humanAchromaticOTF`, `humanOTF`, `humanLSF`, `humanMacularTransmittance`, and `humanOI` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.optics` for the adjacent native optics-helper surface, so imports like `opticsBuild2Dotf`, `opticsCoC`, `opticsDefocusCore`, `opticsDefocusDisplacement`, `opticsDepthDefocus`, `opticsDoF`, `opticsPSF2OTF`, and `opticsRayTrace` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.optics` for the adjacent OI helper surface, so imports like `oiCreate`, `oiCompute`, `oiCustomCompute`, `oiClearData`, `oiCrop`, `oiCombineDepths`, `oiBirefringentDiffuser`, `oiCalculateOTF`, `oiCameraMotion`, `oiDepthCombine`, `oiDepthSegmentMap`, `oiGet`, `oiFrequencyResolution`, `oiPadDepthMap`, `oiPreviewVideo`, `oiSaveImage`, `oiShowImage`, `oiSpace`, `oiSpatialSupport`, `oiSpatialResample`, `oiSet`, `oiWBCompute`, `oiCalculateIlluminance`, and `oiDiffuser` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.optics` for the adjacent WVF helper surface, so imports like `wvfCreate`, `wvfSet`, `wvfGet`, `wvfCompute`, `wvfComputePSF`, `wvfAperture`, `wvfApertureP`, `wvfPupilFunction`, `wvfPupilAmplitude`, `wvf2oi`, `wvf2optics`, `wvf2PSF`, `wvfClearData`, `wvfLoadThibosVirtualEyes`, and the defocus microns/diopters conversion helpers work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.optics` for the adjacent ray-trace helper surface, so imports like `rtFileNames`, `rtRootPath`, `rtImportData`, `rtSynthetic`, `rtPSFInterp`, `rtPSFGrid`, `rtPSFEdit`, `rtPSFApply`, `rtPrecomputePSF`, `rtPrecomputePSFApply`, `rtGeometry`, `rtOTF`, `rtAngleLUT`, and the supporting block/height interpolation helpers work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.optics` for the residual helper surface, so imports like `airyDisk`, `ieFieldHeight2Index`, `siSynthetic`, `sceCreate`, `sceGet`, `zemaxLoad`, and `zemaxReadHeader` work without going through the package root
- and now also upstream MATLAB replay for `oiCreate('valid')`, so the optics wrapper returns the legacy normalized constructor-name list instead of failing dispatch on that helper query
- and now also upstream MATLAB replay for `opticsCreate('valid')`, so the headless optics wrapper returns its supported normalized constructor-name list instead of falling through into `oiCreate('valid')` and breaking on the returned list payload
- and now also upstream MATLAB replay for `oiCreate('uniform d65'/'uniform ee'/'black')`, so those compact optical-image constructors build through the legacy uniform-scene wrappers instead of being advertised in the valid-type list but rejected at dispatch
- and now also upstream MATLAB replay for the human OI constructor family, so `oiCreate('human mw')` and `oiCreate('human'/'wvf human')` build through the existing human OTF/WVF helpers instead of being advertised in the valid-type list but rejected at dispatch
- and now also MATLAB-style top-level defaulting in `oiCreate(...)`, so an explicit empty `oiType` placeholder falls back to the default diffraction-limited constructor and the upstream `uniform ee specify` alias routes through the existing `uniform ee` wrapper
- and now also MATLAB-style top-level defaulting in `cameraCreate(...)`, so an explicit empty `cameraType` placeholder falls back to the default camera constructor instead of being treated as a literal unsupported type token
- and now also MATLAB-style top-level defaulting in `displayCreate(...)`, so an explicit empty `displayName` placeholder falls back to the default display constructor instead of being treated as a literal unsupported display token
- and now also MATLAB-style top-level defaulting in `sceneCreate(...)`, so an explicit empty `sceneName` placeholder falls back to the default Macbeth constructor instead of being treated as a literal unsupported scene token
- and now also MATLAB-style top-level defaulting in `sensorCreate(...)`, so an explicit empty `sensorType` placeholder falls back to the default Bayer constructor instead of being treated as a literal unsupported sensor token
- and now also MATLAB-style top-level defaulting in `ipCreate(...)`, so an explicit empty `ipName` placeholder falls back to the default image-processor constructor instead of being treated as a literal empty IP name
- and now also blank-string top-level defaulting in `sceneCreate(...)`, so `sceneCreate('')` follows the same default Macbeth constructor path as the existing empty-array placeholder form instead of being treated as a literal scene token
- and now also blank-string top-level defaulting in `sensorCreate(...)`, so `sensorCreate('')` follows the same default Bayer constructor path as the existing empty-array placeholder form instead of being treated as a literal sensor token
- and now also blank-string top-level defaulting in `oiCreate(...)` and `opticsCreate(...)`, so `oiCreate('')` and `opticsCreate('')` follow the same default diffraction-limited constructor path as the existing empty-array placeholder form instead of being treated as literal type tokens
- and now also blank-string top-level defaulting in `illuminantCreate(...)`, so `illuminantCreate('')` follows the same default D65 constructor path as the existing empty-array placeholder form instead of being treated as a literal illuminant token
- and now also MATLAB-style empty and blank placeholder defaulting in `fontCreate(...)`, so direct calls like `fontCreate([], [], [], [], [])` and `fontCreate('', '', '', '', '')` reuse the default Georgia 14 pt font instead of stringifying placeholders or failing on `int([])` / `int('')`
- and now also MATLAB-style empty-array and blank-string top-level defaulting in `pixelCreate(...)`, so direct calls like `pixelCreate([], wave)` and `pixelCreate('', wave)` follow the default APS constructor path instead of tripping over Python truthiness on placeholder pixel types
- and now also blank-string optional placeholder defaulting across the shared sensor helper lane, so calls like `humanConeMosaic([6, 8], '', '', '')` reuse the same defaults as the existing empty-array forms instead of tripping over scalar parsing
- and now also the deprecated `scenePad(...)` compatibility shim, so the public surface exists again and replays the upstream `Deprecated. You probably want oiPad.` error instead of remaining missing from `pyisetcam.scene`
- and now also the missing module-local sensor aliases `sensorDescription` and `sensorDNGRead`, so those existing helper surfaces are available directly from `pyisetcam.sensor` instead of only through `pyisetcam.description` or `pyisetcam.fileio`
- and now also upstream MATLAB replay for `opticsCreate('human')`, so the optics-wrapper alias routes through the existing Marimont-Wandell `human mw` builder instead of incorrectly inheriting `oiCreate('human')` Thibos/WVF semantics
- and now also MATLAB-style empty-placeholder defaulting for the optional wave slot in `oiCreate('default', [])` and `oiCreate('pinhole', [])`, so those wrappers keep the standard default wavelength grid instead of leaking an empty spectrum into the returned OI shell
- and now also upstream MATLAB replay for `opticsCreate('empty')`, so the headless optics wrapper returns the expected minimal empty shell instead of incorrectly inheriting the default diffraction-limited OI optics payload
- and now also upstream MATLAB replay for the human optics-wrapper bookkeeping, so `opticsCreate('human mw')` and `opticsCreate('wvf human'/'human wvf')` strip camera transmittance and the WVF branch restores the expected `humanwvf` optics name
- and now also upstream MATLAB replay for default optics-wrapper bookkeeping, so `opticsCreate()` restores the legacy `standard (1/4-inch)` name and base `vignetting=0` metadata instead of returning an anonymous struct
- and now also upstream MATLAB replay for the base optics wrapper type metadata, so `opticsCreate()` plus the human wrapper variants all return the expected `type='optics'` field instead of only the explicit empty-shell path carrying that tag
- and now also upstream MATLAB replay for top-level OI bookkeeping, so `oiCreate(...)` and `oiSet(oi, 'name', ...)` keep the legacy `name` and `type='opticalimage'` fields mirrored into the returned OI payload instead of leaving those struct fields empty
- and now also upstream MATLAB replay for human-lens bookkeeping, so `oiCreate('human mw')`, `oiCreate('human'/'wvf human')`, and the inherited `opticsCreate(...)` human wrappers attach a lightweight `lens` payload while still dropping the camera-style `transmittance` field
- and now also upstream MATLAB replay for the human-lens getter surface, so `oiGet/opticsGet(..., 'lens')`, `'transmittance'`, and `'transmittance wave'` fall back to the stored human lens payload when the camera-style `transmittance` slot is absent instead of synthesizing a unity curve
- and now also upstream MATLAB replay for the human-lens setter surface, so `oiSet/opticsSet(..., 'lens', lensPayload)` normalizes the stored lens payload and removes the camera-style `transmittance` slot instead of keeping both representations live at once
- and now also upstream MATLAB replay for adjacent transmittance setters on human-lens optics, so `oiSet/opticsSet(..., 'transmittance', ...)` and `'transmittance wave'` keep updating the stored lens payload instead of silently recreating a camera-style `transmittance` slot
- and now also upstream MATLAB replay for stored wavefront bookkeeping on the public optics wrappers, so `opticsCreate('wvf human')`, `opticsCreate('wvf')`, and `opticsCreate('shift invariant')` expose the legacy top-level `wvf` payload alongside the existing `wavefront` field
- and now also upstream MATLAB replay for the Marimont-Wandell human optics wrappers, so `opticsCreate('human')` and `opticsCreate('human mw')` no longer leak the shift-invariant `wavefront`/`wvf` payload that belongs only on the wavefront-backed optics branches
- and now also upstream MATLAB replay for the human OI constructor bookkeeping, so `oiCreate('human mw')` and `oiCreate('human'/'wvf human')` drop the camera transmittance slot and the WVF branch keeps the legacy `humanwvf` optics name instead of collapsing it to `human`
- and now also upstream MATLAB replay for `oiCreate('empty')`, so the no-argument path returns the basic optical-image shell with empty wavelength sampling instead of seeding diffraction-limited optics/transmittance, while the practical `oi_create('empty', wave)` override still works
- and now also upstream MATLAB replay for `oiCreate('pinhole')` naming, so both the returned OI and its optics struct keep the legacy `pinhole` name instead of falling back to the default `opticalimage` / unnamed optics labels
- and now also upstream MATLAB replay for inherited OI optics names, so `oiCreate('default'/'diffraction')` keeps `standard (1/4-inch)` and `oiCreate('wvf'/'shift invariant')` keeps `wvf`, with `uniform d65` and `black` inheriting those same legacy labels
- and now also upstream MATLAB replay for the bare `opticsCreate('empty')` shell, so the empty optics wrapper no longer invents `offaxis_method` or `vignetting` fields beyond the legacy `type` and `name`
- and now also upstream MATLAB replay for shift-invariant optics-wrapper naming, so `opticsCreate('wvf')` and `opticsCreate('shift invariant')` restore the expected `wvf` optics name instead of returning anonymous shift-invariant structs
- and now also upstream MATLAB replay for the accepted `opticsCreate('diffraction')` alias, so it restores the same `standard (1/4-inch)` default optics metadata as `opticsCreate()` instead of returning an anonymous diffraction-limited struct
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scene` for the adjacent helper surface, so imports like `sceneAdd`, `sceneAdjustIlluminant`, `sceneFromFile`, `sceneInitGeometry`, `sceneList`, `scenePhotonNoise`, `sceneSpatialResample`, and `sceneToFile` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scene` for the adjacent native scene-utility surface, so imports like `sceneAddGrid`, `sceneAdjustLuminance`, `sceneAdjustPixelSize`, `sceneAdjustReflectance`, `sceneCalculateLuminance`, `sceneClearData`, `sceneCombine`, `sceneIlluminantScale`, `sceneRamp`, `sceneRotate`, `sceneSaveImage`, `sceneSDR`, `sceneShowImage`, `sceneSPDScale`, `sceneSpatialSupport`, `sceneThumbnail`, `sceneTranslate`, `sceneVernier`, and `sceneWBCreate` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scene` for the adjacent constructor and scene-data helper surface, so imports like `sceneCrop`, `sceneDescription`, `sceneEnergyFromVector`, `sceneExtractWaveband`, `sceneFrequencySupport`, `sceneFromBasis`, `sceneFromDDFFile`, `sceneHDRChart`, `sceneHDRImage`, `sceneIlluminantPattern`, `sceneIlluminantSS`, `sceneInitSpatial`, `sceneInsert`, `sceneInterpolateW`, `sceneMakeVideo`, `scenePhotonsFromVector`, `sceneRadianceChart`, `sceneRadianceFromVector`, and `sceneReflectanceChart` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scene` for the adjacent Macbeth and EXIF helper surface, so imports like `macbethReadReflectance`, `macbethChartCreate`, `macbethDrawRects`, `macbethEvaluationGraphs`, `macbethGretagSGCreate`, `macbethIdealColor`, `macbethLuminanceNoise`, `macbethPatchData`, `macbethRectangles`, `macbethROIs`, `macbethSelect`, `macbethSensorValues`, `exiftoolInfo`, and `exiftoolDepthFromFile` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scene` for the adjacent pattern/HDR helper surface, so imports like `hdrRender`, `FOTParams`, `gaborP`, `ieCheckerboard`, and `MOTarget` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scene` for the residual chart/reflectance helper surface, so imports like `chartPatchData`, `ieCookTorrance`, and `ieReflectanceSamples` work without going through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.color` for the shared spectral-conversion helpers, so imports like `Energy2Quanta`, `Quanta2Energy`, and `ieUnitScaleFactor` work from the color module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.illuminant` for the core illuminant helpers, so `illuminantCreate`, `illuminantGet`, and `illuminantSet` work from the module itself instead of only through the package root
- and now also upstream MATLAB-style key/value replay for `illuminantCreate('gaussian', wave, ...)`, so `center`, `sd`, and `peak energy` pairs work alongside the existing params-struct path instead of being ignored outside the struct form
- and now also MATLAB-style empty-placeholder replay in the public illuminant wrappers, so `illuminantCreate([], [])` and `illuminantRead([], [], [], [])` fall back to the default D65 / default-wave contract instead of failing on empty array payloads
- and now also direct module-level MATLAB alias coverage in `pyisetcam.color` for the adjacent photometry and color-conversion helpers, so imports like `ieXYZFromPhotons`, `ieLuminanceFromEnergy`, `ieLuminanceFromPhotons`, `ieLuminance2Radiance`, `ieScotopicLuminanceFromEnergy`, `ieResponsivityConvert`, `xyz2srgb`, and `xyy2xyz` work from the color module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.color` for the adjacent color-space helper family, so imports like `srgbParameters`, `adobergbParameters`, `srgb2lrgb`, `lrgb2srgb`, `Y2Lstar`, `ieLAB2XYZ`, `lms2srgb`, `lms2xyz`, and `xyz2lms` work from the color module itself instead of only through the package root
- and now also direct module-level plus package-root MATLAB alias coverage for the shared chromaticity helper, so `chromaticity` and `chromaticityXY` resolve through the existing `chromaticity_xy(...)` implementation instead of only being available in snake_case
- and now also direct module-level MATLAB alias coverage in `pyisetcam.color` for the adjacent spectral/image-transform helper family, so imports like `RGB2XWFormat`, `XW2RGBFormat`, `srgb2xyz`, `ieXYZFromEnergy`, `ieXYZ2LAB`, `xyz2luv`, `xyz2uv`, `spd2cct`, `srgb2colortemp`, and `colorTransformMatrix` work from the color module itself instead of only through the package root or adjacent modules
- and now also direct module-level MATLAB alias coverage for the residual closed-form `colorBlockMatrix` helper, so both `pyisetcam.color` and `pyisetcam.sensor` expose the existing sensor-display block-matrix path instead of leaving it buried as an internal helper
- and now also direct module-level MATLAB alias coverage in `pyisetcam.session` for the environment path helpers, so imports like `ISET`, `isetPath`, and `isetRootPath` work from the session module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.scielab` for the native SCIELAB helper surface, so imports like `getPlanes`, `scParams`, `separableConv`, `separableFilters`, and `scielabRGB` work from the module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.display` for the adjacent calibration helper surface, so imports like `ieCalculateMonitorDPI`, `ieLUTDigital`, `mperdot2dpi`, `ieReadSpectra`, `ieUnitScaleFactor`, `srgb2xyz`, `xyz2lms`, and `xyz2srgb` work from the module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.roi` for the adjacent extraction helper surface, so imports like `sceneGet`, `oiGet`, `sensorGet`, `ipGet`, `Quanta2Energy`, and `imageDataXYZ` work from the module itself, and `vcGetROIData(..., 'xyz')` now reuses the same XYZ conversion path without a missing-import failure
- and now also direct module-level MATLAB alias coverage in `pyisetcam.fonts` for the adjacent scene/display helper surface, so imports like `displayCreate`, `displayGet`, `displaySet`, `Energy2Quanta`, `RGB2XWFormat`, `sceneAdjustLuminance`, `sceneCreate`, `sceneGet`, `sceneSet`, and `XW2RGBFormat` work from the module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.fonts` for the native text helper surface, so `bitmapFont`, `rasterizeText`, `addTextToImage`, and `addTextToImageWithBorder` work from the module itself instead of only through the snake_case helpers or the package root uppercase aliases
- and now also direct module-level MATLAB alias coverage in `pyisetcam.metrics` for the native CCT helper, so `cct(...)` works from the metrics module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.metrics` for the adjacent spectral/color helper surface, so `cctFromUV`, `xyz2uv`, `chromaticityXY`, `spectralAngle`, `comparisonMetrics`, and `exampleSPDPair` work from the metrics module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.metrics` for the adjacent temperature/error helper surface, so `correlatedColorTemperature`, `miredDifference`, `meanAbsoluteError`, `meanRelativeError`, `rootMeanSquaredError`, and `peakSignalToNoiseRatio` work from the metrics module itself instead of only through the snake_case helpers
- and now also direct module-level MATLAB alias coverage in `pyisetcam.camera` for the native camera metric helpers, so `cameraMTF(...)` and `cameraAcutance(...)` work from the camera module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.sensor` for the native pixel-helper surface, so `pixelCreate`, `pixelDescription`, `pixelGet`, `pixelIdeal`, `pixelSet`, `pixelSR`, `pixelSNR`, `pixelSNRluxsec`, and `pixelVperLuxSec` work from the sensor module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.sensor` for the adjacent pixel geometry/electrical helpers, so `pixelCenterFillPD`, `pixelPositionPD`, `iePixelWellCapacity`, `SignalCurrentDensity`, and `signalCurrent` work from the sensor module itself instead of only through the package root
- and now also direct module-level MATLAB alias coverage in `pyisetcam.sensor` for the remaining native sensor-helper surface, so `sensorAddFilter`, `sensorCfaSave`, `sensorCFANameList`, `sensorCrop`, `sensorDeleteFilter`, `sensorFromFile`, `sensorPixelCoord`, `sensorReadColorFilters`, `sensorReadFilter`, `sensorReplaceFilter`, `sensorRescale`, `sensorSNR`, and `spatialIntegration` work from the sensor module itself instead of only through the package root
- and now also direct `cone_plot(...)` / `conePlot(...)` alias coverage on top of the existing headless `ieConePlot(...)` mosaic renderer, so the upstream human-vision plotting helper name is available without changing the returned payload contract
- and now also direct `sensor_create_imx490(...)` / `sensorCreateIMX490(...)` coverage, exposing the upstream four-capture IMX490 constructor with the legacy `large-HCG`, `large-LCG`, `small-HCG`, and `small-LCG` naming and per-capture gain/noise settings
- and now also direct `sensor_roi(...)` / `sensorROI(...)` coverage, exposing the upstream centered-sensor ROI helper that returns MATLAB-style `roiLocs` on top of the existing headless ROI utilities
- and now also direct `sensor_vignetting(...)` / `sensorVignetting(...)` coverage, exposing the upstream etendue gateway wrapper across the `skip`, `bare`, `centered`, and `optimal` microlens-vignetting lanes
- and now also direct `pixel_dr(...)` / `pixelDR(...)` coverage, exposing the upstream pixel dynamic-range helper on top of the existing internal `pixel dynamic range` formula and MATLAB-style empty-placeholder handling
- and now also direct `sensor_compute_noise(...)` / `sensorComputeNoise(...)` coverage, exposing the legacy noise-stage gateway over the existing `sensorAddNoise(...)`, gain/offset, clip, and quantization path
- and now also direct `auto_exposure(...)` / `autoExposure(...)` coverage for the supported headless sensor-exposure methods, exposing the upstream `default` / `luminance`, `full`, `mean`, `cfa`, `specular`, `weighted` / `center`, and `video` lanes on top of the existing bright-patch, ROI crop, and sensor-image compute paths
- and now also direct `sensor_macbeth_daylight_estimate(...)` / `sensorMacbethDaylightEstimate(...)` coverage, exposing the upstream daylight-basis illuminant estimator on top of the existing Macbeth chart ROI helpers and sensor spectral-QE path
- and now also direct `sensor_cone_plot(...)` / `sensorConePlot(...)` coverage, exposing the upstream human-sensor cone-mosaic plot wrapper with headless `ieConePlot(...)` payloads and a `sensorShowCFA(...)` fallback for block-layout human mosaics
- and now also corrected `sceneCreate('bar', ...)` mean-luminance normalization plus curated Octave parity on the equal-photon bar scene family, covering the centered three-pixel bar geometry and `100 cd/m^2` luminance contract
- and now also corrected the shared `sceneCreate('line ...', ...)` mean-luminance normalization plus curated Octave parity on `sceneCreate('line ee', ...)`, `sceneCreate('line ep', ...)`, and `sceneCreate('lined65', ...)`, covering the equal-energy, equal-photon, and D65 vertical-line geometry with the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniform monochromatic', ...)`, covering the single-wave `550 nm` narrow-band scene, its `12x12` geometry, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniform d65', ...)`, covering the `24x24` D65 uniform scene, its flat spatial layout with non-flat D65 spectrum, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('empty')`, covering the cleared Macbeth-D65 geometry, the preserved illuminant spectrum, the all-zero photon cube, and the `0 cd/m^2` mean-luminance contract
- and now also curated Octave parity on `sceneCreate('lstar', ...)`, covering the `80x200` stepped L* chart, its monotonically increasing 20-bar luminance sequence, the full photon cube, and the shared `100 cd/m^2` mean-luminance contract
- and now also corrected `sceneCreate('hdr')` mean-luminance normalization plus curated Octave parity on the default `384x384` HDR-lights constructor, covering the normalized mean spectrum, sampled normalized luminance rows through the circle/line/square bands, and the returned `100 cd/m^2` contract
- and now also curated Octave parity on `sceneCreate('macbethtungsten')`, covering the `64x96` Macbeth chart under tungsten illumination together with the returned `100 cd/m^2` photon-cube contract
- and now also curated Octave parity on `sceneCreate('uniform', ...)` / `sceneCreate('uniformEE', ...)`, covering the direct `24x24` equal-energy constructor alias, its flat spatial layout with wavelength-varying photon spectrum, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniform equal photon', ...)`, covering the `24x24` equal-photon uniform scene, its flat spectral-spatial photon cube, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('uniformbb', ...)`, covering the compact blackbody alias and its exact equivalence to `sceneCreate('uniform bb', 16, 4500, ...)`
- and now also Python `sceneCreate('uniformblackbody', ...)` alias coverage, with curated Octave parity against the canonical upstream `sceneCreate('uniformbb', 16, 4500, ...)` constructor and exact equivalence to `sceneCreate('uniform bb', 16, 4500, ...)`
- and now also added the missing upstream `sceneCreate('uniformephoton')` alias in Python, with curated Octave parity on the compact equal-photon alias and exact equivalence to `sceneCreate('uniform ep', ...)`
- and now also curated Octave parity on Python `sceneCreate('uniformep', ...)`, using the canonical upstream `sceneCreate('uniformephoton', ...)` constructor and locking exact equivalence back to `sceneCreate('uniform ep', ...)`
- and now also curated Octave parity on Python `sceneCreate('uniformequalphoton', ...)` and `sceneCreate('uniformequalphotons', ...)`, using the canonical upstream singular `sceneCreate('uniformequalphoton', ...)` constructor and locking exact equivalence back to `sceneCreate('uniform ep', ...)`
- and now also curated Octave parity on `sceneCreate('line', ...)`, covering the direct D65 line-scene alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('lined65', ...)`
- and now also curated Octave parity on Python `sceneCreate('lineequalphoton', ...)`, using the upstream `sceneCreate('lineequalphoton', ...)` constructor and locking exact equivalence back to `sceneCreate('line ep', ...)`
- and now also curated Octave parity on `sceneCreate('lineee', ...)`, covering the compact equal-energy line alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('line ee', ...)`
- and now also curated Octave parity on `sceneCreate('impulse1dd65', ...)`, covering the impulse-style D65 line alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('lined65', ...)`
- and now also curated Octave parity on `sceneCreate('impulse1dee', ...)`, covering the impulse-style equal-energy line alias, the shared `64x64` vertical-line photon cube, and its exact equivalence to `sceneCreate('line ee', ...)`
- and now also curated Octave parity on `sceneCreate('sinusoid', ...)`, covering the direct harmonic-scene alias, the shared `64x64` multi-frequency photon cube, and its exact equivalence to `sceneCreate('harmonic', ...)`
- and now also curated Octave parity on `sceneCreate('uniformEESpecify', ...)`, covering the explicit `380:10:720 nm` equal-energy uniform scene, its `128x128` geometry, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('exponential intensity ramp', ...)`, covering the `64x64` exponential ramp with dynamic range `256`, exact row-uniformity, monotonic column growth, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('linear intensity ramp', ...)`, covering the `64x64` linear ramp with dynamic range `256`, per-row monotonic growth, the top-row full-scale ratio, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('rings rays', ...)`, covering the `64x64` Mackay-style radial scene at frequency `8`, its row/column center symmetry, and the same `100 cd/m^2` luminance contract
- and now also corrected shared `sceneCreate('point array', ...)` / `sceneCreate('grid lines', ...)` luminance normalization, with curated Octave parity on `sceneCreate('point array', ...)` covering the `64x64` equal-photon point lattice with `16 px` spacing, its exact `4x4` bright-point grid, and the sparse scene geometry used by downstream optics and sensor scripts, and on `sceneCreate('grid lines', ...)` covering the matching `64x64` equal-photon distortion grid with `16 px` spacing, exact `4x4` full-bright line support, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on `sceneCreate('white noise', ...)`, covering the deterministic `128x128` D65 white-noise scene at `20%` contrast through its `1 deg` FOV, photon-cube geometry, normalized spatial distribution summaries, and normalized mean spectrum rather than brittle raw random pixels
- and now also corrected `sceneCreate('disk array', ...)` mean-luminance normalization plus curated Octave parity on the deterministic `64x64` equal-photon `2x2` disk lattice with radius `8`, covering its exact four connected components and centroids, the two-level photon cube, and the same `100 cd/m^2` luminance contract
- and now also implemented `sceneCreate('square array', ...)` plus curated Octave parity on the deterministic `64x64` equal-photon `2x2` square lattice with side `8`, covering its exact four connected components and centroids, the two-level photon cube, and the same `100 cd/m^2` luminance contract
- and now also curated Octave parity on the `s_surfaceMunsell.m` workflow, covering explicit `munsell.mat` loading from `data/surfaces/charts`, the illuminant-C wavelength/support vector, the reshaped XYZ-to-sRGB chip rendering, chromaticity and LAB summary bounds, and the first 45 Munsell hue/value/angle records printed by the script
- and now also curated Octave parity on the `s_sceneDemo.m` workflow, covering the `macbethd65` scene creation path, luminance/photons/wave extraction with the script’s max-photon and mean-spectrum invariants, the FOV update to `20 deg`, and the `freq orient pattern` bottom-row luminance plus `550 nm` radiance-line summaries
- and now also curated Octave parity on the `s_sceneExamples.m` workflow, covering the built-in example scene catalog across `rings rays`, frequency-orientation, harmonic, checkerboard, line, slanted bar, grid, point-array, Macbeth tungsten, `uniformEESpecify`, `lstar`, and exponential-ramp scenes, with deterministic center-row/center-column luminance summaries and geometry-only coverage for the default random reflectance chart
- and now also curated Octave parity on `t_sceneIntroduction.m`, covering the introductory `macbeth d65` create/get/set flow, the dependent sample-spacing updates under distance/FOV changes, and the `StuffedAnimals_tungsten-hdrs` multispectral scene before and after the 5500 K blackbody illuminant swap
- and now also curated Octave parity on the `s_sceneRoi.m` workflow, covering ROI photons/energy/illuminant/reflectance extraction on the default scene, the returned ROI-mean spectra, and exact manual-versus-direct reflectance agreement
- and now also curated Octave parity on the `s_sceneRotate.m` workflow, covering the default star-pattern rotation movie through selected rotated-frame sizes, luminance summaries, and canonicalized center-row/center-column luminance profiles
- and now also curated Octave parity on the `s_sceneWavelength.m` workflow, covering default-scene wavelength resampling through `sceneSet(..., 'wave', ...)`, preserved geometry/luminance at 10 nm, 5 nm, and 2 nm narrowband supports, and the normalized mean/center spectral radiance trends across those three scene states
- and now also headless `ie_save_multispectral_image(...)` / `ieSaveMultiSpectralImage(...)` plus curated Octave parity on the `s_sceneHCCompress.m` workflow, covering `hcBasis(...)`-driven 95% and 99% basis compression of `StuffedAnimals_tungsten-hdrs`, the save/reload roundtrip through basis-coded multispectral MAT files, preserved scene geometry, and the reconstructed 5 nm mean/center spectral trends
- and now also headless `image_increase_image_rgb_size(...)` / `imageIncreaseImageRGBSize(...)` plus curated Octave parity on the `s_sceneIncreaseSize.m` workflow, covering exact pixel-replication resizing by `[2 3]`, `[1 2]`, and `[3 1]`, the preserved mean luminance / normalized mean scene SPD contract across those scene states, and exact replay back to the previous photon cube when striding by the requested enlargement factors
- and now also headless `scene_show_image(...)` / `sceneShowImage(...)`, Haar-path `hdr_render(...)` / `hdrRender(...)`, and MATLAB-style column-vector illuminant adjustment for the `s_sceneRender.m` workflow, with curated Octave parity on the D75-adjusted StuffedAnimals render plus the Feng Office and StuffedAnimals HDR-companded render summaries, center-pixel RGB values, and canonicalized center-row luminance profiles
- and now also headless `scene_radiance_from_vector(...)` / `sceneRadianceFromVector(...)`, `scene_photons_from_vector(...)` / `scenePhotonsFromVector(...)`, and `scene_energy_from_vector(...)` / `sceneEnergyFromVector(...)` for the direct MATLAB vector-to-scene-cube helper mini-family used by older scene setup code
- and now also headless `scene_init_geometry(...)` / `sceneInitGeometry(...)`, `scene_spatial_resample(...)` / `sceneSpatialResample(...)`, and `scene_photon_noise(...)` / `scenePhotonNoise(...)`, extending the legacy scene geometry/support/export helper surface on top of the existing `sceneGet(...)`, ROI, and `sceneShowImage(...)` paths
- and now also headless `scene_add_grid(...)` / `sceneAddGrid(...)`, `scene_adjust_pixel_size(...)` / `sceneAdjustPixelSize(...)`, `scene_illuminant_scale(...)` / `sceneIlluminantScale(...)`, `scene_spd_scale(...)` / `sceneSPDScale(...)`, and `scene_adjust_reflectance(...)` / `sceneAdjustReflectance(...)`, plus the remaining low-level `sceneGet/Set` support they need for `sample spacing`, `energy`, `known reflectance`, and `peak radiance and wave`
- and now also headless `FOTParams(...)`, `gaborP(...)`, `ieCheckerboard(...)`, `MOTarget(...)`, and `sceneRamp(...)`, plus direct `sceneCreate('ramp', ...)` and `sceneCreate('ramp equal photon', ...)` support for the lightweight MATLAB scene-pattern helper layer already backed by the existing frequency-orientation, harmonic, checkerboard, and ramp generators
- and now also the legacy `scene/imgtargets` grayscale helper layer: `img_deadleaves(...)` / `imgDeadleaves(...)`, `img_disk_array(...)` / `imgDiskArray(...)`, `img_mackay(...)` / `imgMackay(...)`, `img_radial_ramp(...)` / `imgRadialRamp(...)`, `img_ramp(...)` / `imgRamp(...)`, `img_square_array(...)` / `imgSquareArray(...)`, `img_sweep(...)` / `imgSweep(...)`, and `img_zone_plate(...)` / `imgZonePlate(...)`, all routed through the same underlying pattern generators already used by the scene constructors
- and now also the headless display helper layer: `display_list(...)` / `displayList(...)`, `display_description(...)` / `displayDescription(...)`, `display_show_image(...)` / `displayShowImage(...)`, `display_set_max_luminance(...)` / `displaySetMaxLuminance(...)`, `display_set_white_point(...)` / `displaySetWhitePoint(...)`, `display_max_contrast(...)` / `displayMaxContrast(...)`, `ie_calculate_monitor_dpi(...)` / `ieCalculateMonitorDPI(...)`, and `mperdot2dpi(...)`, matching the older MATLAB display utility surface without GUI prompts or windows
- and now also headless `display_convert(...)` / `displayConvert(...)`, `display_pt2iset(...)` / `displayPT2ISET(...)`, and `display_reflectance(...)` / `displayReflectance(...)`, matching the remaining MATLAB display-calibration conversion and theoretical reflectance-display helper surface in headless form
- and now also the headless Macbeth helper layer: `macbeth_rectangles(...)` / `macbethRectangles(...)`, `macbeth_rois(...)` / `macbethROIs(...)`, `macbeth_patch_data(...)` / `macbethPatchData(...)`, and `macbeth_ideal_color(...)` / `macbethIdealColor(...)`, matching the older MATLAB chart-geometry, ROI extraction, patch statistics, and ideal-target utility surface
- and now also the adjacent headless `scene/reflectance` helper layer: `chart_patch_data(...)` / `chartPatchData(...)` and `ie_cook_torrance(...)` / `ieCookTorrance(...)`, matching the older MATLAB chart ROI extraction wrapper for vcimage/sensor analysis plus the standalone Cook-Torrance BRDF utility
- and now also the adjacent headless `scene/macbeth` compatibility layer: `macbeth_chart_create(...)` / `macbethChartCreate(...)`, `macbeth_draw_rects(...)` / `macbethDrawRects(...)`, `macbeth_select(...)` / `macbethSelect(...)`, `macbeth_sensor_values(...)` / `macbethSensorValues(...)`, `macbeth_luminance_noise(...)` / `macbethLuminanceNoise(...)`, `macbeth_evaluation_graphs(...)` / `macbethEvaluationGraphs(...)`, and `macbeth_gretag_sg_create(...)` / `macbethGretagSGCreate(...)`, matching the older Macbeth chart-construction, patch-selection, sensor/IP analysis, evaluation, and Gretag SG helper surface with headless payload returns
- and now also the adjacent headless `scene/pattern` wrapper layer: `scene_hdr_chart(...)` / `sceneHDRChart(...)`, `scene_hdr_image(...)` / `sceneHDRImage(...)`, `scene_radiance_chart(...)` / `sceneRadianceChart(...)`, and `scene_vernier(...)` / `sceneVernier(...)`, matching the older MATLAB HDR-strip, HDR-patch, radiance-chart, and Vernier pattern helpers on top of the current scene construction core
- and now also the adjacent headless `scene` export/helper layer: `scene_from_basis(...)` / `sceneFromBasis(...)`, `scene_insert(...)` / `sceneInsert(...)`, `scene_to_file(...)` / `sceneToFile(...)`, `scene_wb_create(...)` / `sceneWBCreate(...)`, and `scene_make_video(...)` / `sceneMakeVideo(...)`, matching the older MATLAB basis-reconstruction, scene insertion, multispectral export, per-waveband export, and scene-sequence video helper surface without GUI dependencies
- and now also the adjacent headless `scene` file-access layer: `scene_from_ddf_file(...)` / `sceneFromDDFFile(...)` and `scene_sdr(...)` / `sceneSDR(...)`, matching the older MATLAB Dynamic Depth Format wrapper and Stanford Digital Repository local-first scene fetch helper on top of the existing `sceneFromFile(...)` path
- and now also headless `sensorComputeNoiseFree(...)`, `sensorAddNoise(...)`, `sensorComputeImage(...)`, and `sensorComputeFullArray(...)`, matching the legacy MATLAB sensor-simulation wrapper layer on top of the existing Python sensor compute path
- and now also headless `regrid_oi_to_isa(...)` / `regridOI2ISA(...)`, `plane2mosaic(...)` / `plane2rgb(...)`, `sensor_compute_mev(...)` / `sensorComputeMEV(...)`, and `sensor_compute_sv_filters(...)` / `sensorComputeSVFilters(...)`, matching the remaining legacy MATLAB sensor-simulation helper layer for current-density resampling, CFA-plane expansion, multi-exposure voltage combination, and space-varying IR-filter replay
- and now also headless `bin_sensor_compute(...)` / `binSensorCompute(...)` and `bin_sensor_compute_image(...)` / `binSensorComputeImage(...)`, matching the legacy MATLAB pixel-binning sensor wrapper layer on top of the existing binning, noise, and quantization helpers
- and now also headless `camera_clear_data(...)` / `cameraClearData(...)`, `camera_compute_srgb(...)` / `cameraComputesrgb(...)`, and `camera_compute_sequence(...)` / `cameraComputeSequence(...)`, matching the older MATLAB camera wrapper layer for computed-state reset, ideal-versus-rendered sRGB replay, and multi-frame exposure sweeps on top of the existing Python camera pipeline
- and now also headless `scene_description(...)` / `sceneDescription(...)`, `scene_list(...)` / `sceneList(...)`, and `scene_thumbnail(...)` / `sceneThumbnail(...)` for the legacy scene description/listing/thumbnail helper mini-family built on the existing headless scene render path
- and now also headless `scene_crop(...)` / `sceneCrop(...)`, `scene_extract_waveband(...)` / `sceneExtractWaveband(...)`, and `scene_translate(...)` / `sceneTranslate(...)` for the legacy scene crop/spectral-extraction/translation helper mini-family built on the existing scene ROI, wavelength-interpolation, and geometry paths
- and now also MATLAB-style upstream RGB asset lookup in `scene_from_file(..., 'rgb', ...)` plus curated Octave parity on the `t_sceneRGB2Radiance.m` workflow, covering `sceneFromFile('macbeth.tif', 'rgb', ..., display)` across `OLED-Sony`, `LCD-Apple`, and `CRT-Dell`, together with the three display white points, primary chromaticities, scene luminance/SPD summaries, and display-dependent rendered RGB means
- and now also headless `macbeth_read_reflectance(...)` / `macbethReadReflectance(...)`, `xyz_to_srgb(...)` / `xyz2srgb(...)`, and `image_flip(...)` / `imageFlip(...)` plus curated Octave parity on the `t_sceneSurfaceModels.m` tutorial, covering Macbeth SVD basis extraction, low-rank reflectance reconstruction error, the normalized D65 render path, and the rendered 4x6 chart RGB summaries for the 1-D, 2-D, 3-D, 4-D, and full-basis cases
- and now also curated Octave parity on `s_reflectanceBasis.m`, covering the upstream two-file reflectance aggregation path, the 8-D SVD basis subspace, the first four sign-canonicalized basis vectors, and the reduced-dimensional reconstruction RMSE/statistics
- and now also curated Octave parity on `s_colorIlluminantTransforms.m`, covering the blackbody `3500:500:8000 K` transform bank on the default reflectance chart plus the two fixed-matrix cosine-similarity tables from the script
- and now also curated Octave parity on `s_chromaticSpatialChart.m`, covering the synthesized row-chromatic / column-frequency RGB chart, its white-border construction, the `sceneFromFile(..., 'rgb', 100, 'LCD-Apple')` conversion, and the resulting scene luminance/spectrum center-line summaries
- and now also curated Octave parity on `s_colorConstancy.m`, covering the stuffed-animals and uniform-D65 blackbody sweeps across the script's 15 reciprocal-temperature samples together with the mean-luminance and normalized RGB render summaries for each frame
- and now also headless `srgb_to_color_temp(...)` / `srgb2colortemp(...)` plus curated Octave parity on `s_rgbColorTemperature.m`, covering the exact Macbeth tungsten/D65 camera pipeline, the returned coarse temperature buckets, the shared `2500:500:10500 K` chromaticity lookup table, the normalized rendered-RGB means used by the script, and the upstream MATLAB-style `('method', ...)` optional key/value call shape
- and now also headless `scene_reflectance_chart(...)` / `sceneReflectanceChart(...)` plus `srgb_parameters(...)` / `adobergb_parameters(...)`, with curated Octave parity on `s_srgbGamut.m`, covering the natural/synthetic reflectance charts, the sRGB and Adobe RGB gamut polygons, and the D65/tungsten chromaticity clouds computed from the script's explicit surface sets
- and now also headless `ie_xyz_from_photons(...)` / `ieXYZFromPhotons(...)`, `ie_luminance_to_radiance(...)` / `ieLuminance2Radiance(...)`, `ie_scotopic_luminance_from_energy(...)` / `ieScotopicLuminanceFromEnergy(...)`, `ie_responsivity_convert(...)` / `ieResponsivityConvert(...)`, `srgb_to_lrgb(...)` / `srgb2lrgb(...)`, `lrgb_to_srgb(...)` / `lrgb2srgb(...)`, `y_to_lstar(...)` / `Y2Lstar(...)`, `xyy_to_xyz(...)` / `xyy2xyz(...)`, `ie_lab_to_xyz(...)` / `ieLAB2XYZ(...)`, and the direct Stockman transform helpers `xyz_to_lms(...)` / `xyz2lms(...)`, `lms_to_xyz(...)` / `lms2xyz(...)`, and `lms_to_srgb(...)` / `lms2srgb(...)` for the remaining direct color-transform helper surface
- and now also headless `cct_to_sun(...)` / `cct2sun(...)`, `ie_ctemp_to_srgb(...)` / `ieCTemp2SRGB(...)`, `ie_circle_points(...)` / `ieCirclePoints(...)`, and `mk_inv_gamma_table(...)` / `mkInvGammaTable(...)` for the remaining direct color temperature / geometry / gamma helper surface
- and now also headless `init_default_spectrum(...)` / `initDefaultSpectrum(...)`, `ie_cov_ellipsoid(...)` / `ieCovEllipsoid(...)`, `ie_spectra_sphere(...)` / `ieSpectraSphere(...)`, and `xyz_to_vsnr(...)` / `xyz2vSNR(...)`, covering the remaining direct color-spectrum initialization, covariance-ellipsoid, spectra-sphere, and visual-SNR helper surface
- and now also headless `delta_e_2000(...)` / `deltaE2000(...)`, `delta_e_94(...)` / `deltaE94(...)`, and `delta_e_uv(...)` / `deltaEuv(...)`, covering the remaining direct MATLAB CIELAB/CIELUV Delta-E helper surface with the same component-return contract for the 2000 and 1994 variants
- and now also Phase 4 metrics coverage has started with new `iso.py`, headless `ISOFindSlantedBar(...)` / `ieCXcorr(...)` / `edge_to_mtf(...)`, curated Octave parity on `s_metricsEdge2MTF.m`, and a MATLAB-faithful odd-sized `sceneCreate('slanted bar', ...)` generator that matches the upstream edge orientation and geometry
- and now also headless `ISO12233(...)` / `ieISO12233(...)` in `iso.py`, with curated Octave parity on `s_metricsMTFSlantedBar.m` covering the direct RGB slanted-bar path, the `ieISO12233(ip, sensor, ...)` sensor-space path, the monochrome direct path, and their stable ESF/LSF/MTF50/aliasing contracts under case-scoped parity tolerances
- and now also legacy `ISO12233v1(...)` / `ieISO12233v1(...)` compatibility names in `iso.py`, matching the older MATLAB file surface on top of the current headless slanted-edge implementation rather than treating the superseded `v1` path as a separate algorithm fork
- and now also curated Octave parity on `s_metricsMTFPixelSize.m`, covering the fixed-die-size monochrome slanted-bar sweep across 2/3/5/9 um pixels, the per-pixel-size sensor geometry and ROI replay, Nyquist and MTF50 trends, and the normalized luminance MTF profiles under case-scoped parity tolerances
- and now also headless `pixelVperLuxSec(...)` / `pixelSNRluxsec(...)` in `sensor.py`, with curated Octave parity on `s_metricsSNRPixelSizeLuxsec.m` covering the monochrome 2/4/6/9/10 um pixel sweep, its lux-sec sensitivity curves, and the returned SNR/read-noise/shot-noise summaries
- and now also curated Octave parity on `s_metricsMTFSlantedBarInfrared.m`, including `ieReadColorFilter(..., 'IRBlocking')` compatibility, the NikonD200IR multispectral slanted-bar setup, the fixed-ROI direct ISO12233 branch, and the IR-blocked auto-ROI `ieISO12233(...)` line-spread/MTF branch
- and now also headless `cameraMTF(...)` / `cameraAcutance(...)` plus `cpiqCSF(...)` / `ISOAcutance(...)`, with curated Octave parity on `s_metricsAcutance.m` covering the default camera slanted-edge workflow, the luminance MTF and CPIQ weighting profiles, the sensor degrees-per-distance conversion, and the final acutance scalar
- and now also headless `cameraColorAccuracy(...)` plus `macbethCompareIdeal(...)`, with curated Octave parity on `s_metricsColorAccuracy.m` covering the default Macbeth-camera workflow, normalized white-point recovery, per-patch Delta E / LAB results, and the returned sRGB Macbeth comparison patches
- and now also headless `macbethColorError(...)`, with curated Octave parity on `s_metricsMacbethDeltaE.m` covering the default scene/OI/sensor/IP current-matrix workflow, recovered sensor CCM, normalized white-point recovery, per-patch Delta E / LAB outputs, and processed-image RGB summary statistics
- and now also curated Octave parity on `s_metricsSPD.m`, covering the daylight `4000:500:7000 K` comparison sweep for both the D4000-reference and fixed-D65-white-point branches, including the returned angle, CIELAB Delta E, and mired curves, while `metrics_spd(...)` / `metricsSPD(...)` also now accept the upstream MATLAB-style optional key/value form for `metric`, `wave`, `luminance`, and `white point`
- and now also headless `exposure_value(...)` / `exposureValue(...)`, `photometric_exposure(...)` / `photometricExposure(...)`, and `chart_patch_compare(...)` / `chartPatchCompare(...)`, matching the direct MATLAB exposure/chart helper mini-family used around the older metrics workflows
- and now also curated Octave parity on `t_metricsSQRI.m`, covering the perfect-display SQRI width and luminance sweeps plus the display-MTF replay for `OLED-Samsung-Note3` and `CRT-HP`
- and now also the headless metrics-window helper layer: `metrics_camera(...)` / `metricsCamera(...)`, `metrics_compute(...)` / `metricsCompute(...)`, `metrics_get(...)` / `metricsGet(...)`, `metrics_set(...)` / `metricsSet(...)`, `metrics_description(...)` / `metricsDescription(...)`, `metrics_get_vci_pair(...)` / `metricsGetVciPair(...)`, `metrics_masked_error(...)` / `metricsMaskedError(...)`, `metrics_show_image(...)` / `metricsShowImage(...)`, `metrics_show_metric(...)` / `metricsShowMetric(...)`, `metrics_save_image(...)` / `metricsSaveImage(...)`, `metrics_save_data(...)` / `metricsSaveData(...)`, `metrics_close(...)` / `metricsClose(...)`, `metrics_refresh(...)` / `metricsRefresh(...)`, and `metrics_key_press(...)` / `metricsKeyPress(...)`, matching the older MATLAB metrics GUI helper surface with headless payload returns
- and now also the adjacent metrics ROI/SQRI helpers: `metrics_roi(...)` / `metricsROI(...)`, `metrics_compare_roi(...)` / `metricsCompareROI(...)`, and `ie_sqri(...)` / `ieSQRI(...)`, matching the older MATLAB ROI-picking/Delta-E comparison and Barten SQRI helper surface in headless form while treating the interactive `s_metricsSSIM.m` teaching script as out of scope
- and now also headless `cameraVSNR(...)`, with curated Octave parity on `s_metricsVSNR.m` covering the default camera uniform-field sweep, the stable ROI geometry, normalized VSNR / reciprocal-Delta-E curves across the three light levels, and processed-image RGB summary statistics
- and now also headless `camera_full_reference(...)` / `cameraFullReference(...)` together with the legacy `camera_vsnr_sl(...)` / `cameraVSNR_SL(...)` alias, matching the older MATLAB camera full-reference benchmarking helper on top of the current headless camera/scene/SCIElab stack while treating the GUI-only `cameraWindow.m` and exploratory `cameraMoire.m` workflow as out of scope
- and now also headless `scielabRGB(...)` / `scielab(...)` in a new `scielab.py`, plus MATLAB-style display `rgb2xyz` / `white point` support, with curated Octave parity on `t_metricsScielab.m` covering the `hats.jpg` versus `hatsC.jpg` LCD-Apple tutorial workflow, returned scene geometry, display white point, and the stable S-CIELAB error-map summaries
- and now also curated Octave parity on `s_rgb2scielab.m`, covering the `crt.mat` S-CIELAB script path for `hats.jpg` versus `hatsC.jpg`, including the mean Delta E, the mean/percent above the script’s `Delta E > 2` threshold, scene geometry, display white point, and a normalized error-profile slice
- and now also headless `dac2rgb(...)` / `imageLinearTransform(...)`, with curated Octave parity on `s_scielabExample.m` covering both the script’s `scielabRGB(...)` branch and its explicit `dac2rgb -> displayGet('rgb2xyz') -> imageLinearTransform -> scielab(...)` walkthrough on `hats.jpg` versus `hatsC.jpg`, including the returned scene geometry, white point, mean Delta E summaries, and canonicalized filter/error profiles
- and now also headless `scPrepareFilters(...)` / `sc_prepare_filters(...)`, with curated Octave parity on `s_scielabFilters.m` covering the 101-sample point-spread filters, the 512-sample MTF branch, and the `distribution` / `original` / `hires` Gaussian-parameter variants through normalized center-row and peak summaries
- and now also the legacy SCIELAB helper layer: `changeColorSpace(...)`, `cmatrix(...)`, `gauss(...)`, `getPlanes(...)`, `ieConv2FFT(...)`, `pad4conv(...)`, `preSCIELAB(...)`, `scResize(...)`, `separableConv(...)`, `separableFilters(...)`, and `visualAngle(...)`, matching the older MATLAB preprocessing/filter glue on top of the current headless `scielab.py` implementation
- and now also direct public `ApplyFilters(...)` / `scApplyFilters(...)` aliases on top of the existing `sc_apply_filters(...)` implementation, so the remaining reusable MATLAB SCIELAB filter-application helper names are tracked as covered compatibility surface rather than residual audit debt
- and now also headless `sceneAdd(...)` / `scene_add(...)` plus `sceneGet(..., 'illuminant xyz')`, with curated Octave parity on `s_scielabMTF.m` covering the harmonic `1:32 cpd` sweep, the `remove spatial mean` scene-add path, the returned illuminant XYZ/SPD contract, and the CIELAB versus S-CIELAB Delta E curves
- and now also curated Octave parity on `s_scielabPatches.m`, covering the 7x7 uniform-patch illuminant-perturbation sweep, the returned illuminant XYZ/SPD setup for `sceneCreate('uniform')`, and the paired CIELAB / S-CIELAB Delta E surfaces together with the script’s quantized S-CIELAB summary vector
- and now also curated Octave parity on `s_scielabMasking.m`, covering the fixed `4 cpd` / `0.8` harmonic-mask setup, the `remove spatial mean` target-combination path, the doubled white-point convention from the script, and the returned CIELAB / S-CIELAB target-contrast curves
- and now also headless `colorTransformMatrix(...)` / `scOpponentFilter(...)` / `scComputeSCIELAB(...)` plus `srgb2xyz(...)`, with curated Octave parity on `s_scielabTutorial.m` covering the multispectral Stuffed Animals scene through OI/sensor/IP, the Gray World rendered RGB handoff, the 50 samples-per-degree SCIELAB filter setup, the opponent/filtered-image summaries, and the final SCIELAB LAB output
- and now also curated Octave parity on `s_scielabHarmonicExperiments.m`, covering the sweep-frequency scene through diffuser-blurred OI, gray-world IP rendering, the script’s three opponent-channel scaling branches, and the resulting padded S-CIELAB error-surface summaries

## Status

This repository implements milestone-one scope:

- explicit-object APIs only, no `vcSESSION`
- core scene/display/oi/sensor/ip/camera objects and functions
- upstream asset fetching into `.cache/upstream/isetcam/<sha>/`
- unit tests and a parity harness scaffold

Out of scope for this milestone:

- GUI/App Designer windows
- ray-trace optics
- OpenEXR/MEX integrations
- facetracker, printing, and most peripheral modules

## Quickstart

Start with the user-facing docs:

- [Install Guide](install-guide.md)
- [Getting Started Tutorial](tutorial.md)

Recommended setup:

```bash
conda env create -f environment.yml
conda activate isetcam-py
python -m pip install -e .
```

Run the tutorial examples:

```bash
python examples/end_to_end.py
python examples/explicit_pipeline.py
python examples/scene_from_file.py
python examples/quality_metrics.py
```

These examples write outputs under `reports/tutorial/`.

Optional developer checks:

```bash
pytest
python tools/parity_report.py
python tools/audit_migration_gap.py
python tools/regenerate_parity_baselines.py
```

Run the initial metrics helpers:

```python
from pyisetcam import comparison_metrics, metrics_spd

value, params = metrics_spd(spd1, spd2, metric="cielab", wave=wave, return_params=True)
summary = comparison_metrics(reference_image, rendered_image, data_range=1.0)
```

The report includes max-error locations plus edge/interior and 2x2 phase
diagnostics for array outputs, plus case-specific context and
reference-recompute checks for remaining sensor/camera gaps.
For the curated WVF case, the harness also records strict pre-PSF / PSF
stage parity and treats the final Octave delta-PSF convolution as a
documented relaxed `mean_rel` diagnostic, because that path collapses to a
single-precision FFT artifact in Octave.

## Layout

- `src/pyisetcam/`: package source
- `tools/fetch_upstream.py`: fetch/extract the pinned upstream snapshot
- `tools/octave/`: Octave parity export helpers
- `tests/unit/`: deterministic unit tests
- `tests/parity/`: curated parity case definitions and baseline comparator
- `docs/migration.md`: MATLAB-to-Python name mapping

## Current Expansion

Post-milestone expansion now covers broader scene support and early sensor
preset growth. `scene_from_file` / `sceneFromFile` supports RGB and
monochrome inputs backed by emissive display calibration, the classic test
scene catalog has been expanded with script-driven patterns such as
`harmonic`, `frequency orientation` / `freq orient`, and
`sweep frequency`, `star pattern`, plus script-driven `reflectance chart` support with
explicit sample-list recreation. The first post-milestone `sceneCreate(...)`
expansion slice is now in place too: extra Macbeth illuminant variants
(`d50`, `illc`, `fluorescent`, `ee_ir`, and `custom reflectance`), the
legacy `moire orient` target, and `letter` / `font` scenes now dispatch
through the same headless scene core. The adjacent file-backed constructor
slice is now landed too: `sceneCreate('list'/'scenelist')`,
`sceneCreate('rgb')`, `sceneCreate('multispectral'/'hyperspectral')`, and
`sceneCreate('monochrome'/'unispectral')` now expose the same safe shell
objects the headless `sceneFromFile(...)` path uses before filling in
radiance data; that same shell-constructor lane now also accepts a passed
seed scene in the upstream one-argument forms like `sceneCreate('rgb',
scene)`, `sceneCreate('multispectral', scene)`, and
`sceneCreate('monochrome', scene)` instead of discarding the supplied
scene shell. That same scene wrapper surface now also exposes the core
MATLAB-style gateway aliases directly from `pyisetcam.scene`, so imports
like `from pyisetcam.scene import sceneCreate`, `sceneGet`, and `sceneSet`
work without routing through the package root. That same module now also
exposes the adjacent helper aliases `sceneAdd`, `sceneAdjustIlluminant`,
`sceneFromFile`, `sceneInitGeometry`, `sceneList`, `scenePhotonNoise`,
`sceneSpatialResample`, and `sceneToFile` directly from `pyisetcam.scene`.
The adjacent display getter
lane now also replays the
upstream MATLAB color and contrast aliases `rgb2lms`, `white lms`,
`black xyz`, `primaries xyz` / `primaries xy`, `dark luminance`, and
`peak contrast` on top of the existing headless display calibration
surface; that same display lane now also covers the derived dixel and
subpixel getters `dark level`, `oversample`, `sample spacing`, `fill factor`,
and `subpixel spd`, including MATLAB-style spatial-unit scaling on
`meters per dot` / `dots per meter`, plus the remaining alias names
`lrgb2xyz`, `primaries rgb` / `primaries srgb`, `black radiance`,
`distance`, and `samp per deg`; that same display lane now also exposes the
last vendored `displayGet(...)` conversion case `drgb2xyz` on top of the
existing DAC lookup and display `rgb2xyz` transform, and now also exposes
headless `display_compute(...)` / `displayCompute(...)` for the upstream
subpixel-upsampling wrapper on top of the existing dixel image, oversample,
and Samsung render-function paths. That same display utility lane now also
exposes headless `display_plot(...)` / `displayPlot(...)` for the reusable
non-GUI `spd`, `gamma table`, `gamut`, `gamut 3d`, and `psf` cases,
returning the same MATLAB-style user-data payloads without opening figures;
the `pyisetcam.display` module now also exposes the corresponding MATLAB-style
camelCase aliases directly, so imports like `from pyisetcam.display import
displayPlot` and `displayCompute` work without going through the package root.
Both
`sceneCreate('empty', wave)` and
`sceneCreate('empty', [], wave)` now replay the legacy wavelength override
instead of forcing the default 400:10:700 grid. The same Macbeth/default
scene dispatcher now also accepts placeholder patch-size slots such as
`sceneCreate('default', [], wave)` and `sceneCreate('macbeth', [], wave)`
without losing the default 16-pixel patch size, while the same Macbeth lane
now also treats empty `surfaceFile` and `blackBorder` slots such as
`sceneCreate('default', 16, wave, [], [])` as the documented defaults
instead of coercing `[]` into a bad asset path. The same scene dispatcher now also accepts MATLAB-style
key/value forms for `sceneCreate('hdr chart', ...)` and
`sceneCreate('hdr image', ...)`, and the same compatibility lane now also
replays MATLAB-style key/value dispatch for `sceneCreate('hdr lights', ...)`
instead of only positional or dict-based Python calls. The adjacent
`scene_create('radiance chart', wave, radiance, ...)` and
`sceneCreate('reflectance chart', ...)` wrappers now accept the same
MATLAB-style key/value arguments too, and the positional
`sceneCreate('reflectance chart', pSize, sSamples, sFiles, wave, grayFlag, sampling)`
form now also honors MATLAB-style empty placeholders for the optional
patch-size, wavelength, gray-strip, and sampling slots. The same
`sceneCreate('radiance chart', wave, radiance, rowcol, patchSize, grayFill, ...)`
wrapper now also accepts positional optional slots with MATLAB-style empty
placeholders instead of forcing only dict or key/value extras. `sceneCreate('bar ee', imageSize,
width)` now replays the documented equal-energy bar alias,
`sceneCreate('squares', imageSize, squareSize, arraySize)` now replays the
documented square-array shorthand, and the slanted-bar dispatcher now also
accepts empty field-of-view placeholders in documented forms like
`sceneCreate('iso12233', imageSize, slope, [], wave, darklevel)`, and
the same scene-size parser now also replays empty size placeholders for the
uniform-family constructors such as `sceneCreate('uniform', [], wave)`,
`sceneCreate('uniform d65', [], wave)`, `sceneCreate('uniform ep', [], wave)`,
and `sceneCreate('uniform bb', [], cTemp, wave)`,
preserving the upstream default `32 x 32` scene size, while
`sceneCreate('uniform monochromatic', sz, wave)` now also replays the
documented size-first shorthand in addition to the existing wave-first form,
and now also honors MATLAB-style empty placeholders in that size-first
shorthand, so `sceneCreate('uniform monochromatic', 12, [])` preserves the
default `500 nm` wavelength and `sceneCreate('uniform monochromatic', [], 550)`
preserves the default `128 x 128` size,
while `sceneCreate('bar', 64, [])` and `sceneCreate('bar ee', 64, [])` now
replay the lower-level helper default bar width of `5` pixels instead of
reusing the wrapper's omitted-argument default of `3`,
while `sceneCreate('point array', 128, 16, [], [], wave)` now replays the
lower-level helper default spectral type `d65` instead of reusing the
wrapper's omitted-argument default of `ep`,
and
`sceneCreate('sweep frequency', [], maxFrequency, [], yContrast)` now
honors MATLAB-style empty placeholders for the size and wavelength slots
instead of crashing on `[]`,
and the same empty-wave placeholder handling now also replays correctly
across the common optional-wave pattern constructors such as
`sceneCreate('line ee', ...)`, `sceneCreate('line ep', ...)`,
`sceneCreate('bar', ...)`, `sceneCreate('bar ee', ...)`,
`sceneCreate('point array', ...)`, `sceneCreate('grid lines', ...)`,
`sceneCreate('checkerboard', ...)`, and `sceneCreate('star pattern', ...)`,
while the same uniform-scene lane now also honors MATLAB-style empty
color-temperature placeholders in `sceneCreate('uniform bb', sceneSize, [], wave)`,
preserving the upstream default `5000 K` blackbody illuminant,
while the same scene-pattern lane now also honors MATLAB-style empty
placeholders in optional positional slots such as
`sceneCreate('line ee', imageSize, [], wave)`,
`sceneCreate('bar', imageSize, [], wave)`,
`sceneCreate('point array', imageSize, spacing, [], [], wave)`,
`sceneCreate('grid lines', imageSize, spacing, [], [], wave)`, and
`sceneCreate('checkerboard', [], nCheckPairs, spectralType, wave)`,
`sceneCreate('checkerboard', checkPeriod, [], spectralType, wave)`, and
`sceneCreate('star pattern', imageSize, [], [], wave)`, reusing the
documented default offset, width, spectral type, thickness, point size, and
line-count values instead of crashing on `[]`,
and the same array-pattern lane now also honors MATLAB-style empty size
placeholders in `sceneCreate('disk array', imageSize, [], arraySize, wave)`
and `sceneCreate('square array', imageSize, [], arraySize, wave)`, preserving
the upstream default disk radius of `128` pixels and square size of `16`
pixels,
while the same helper-backed pattern lane now also honors MATLAB-style empty
placeholders in `sceneCreate('whitenoise', imageSize, [], wave)`,
`sceneCreate('rings rays', [], imageSize, wave)`,
`sceneCreate('rings rays', radialFreq, [], wave)`, and
`sceneCreate('slanted bar', [], [], [], wave, [])`, preserving the upstream
default contrast, radial frequency, image size, slope, field of view, and
dark level instead of crashing on `[]`,
and the same ramp/vernier dispatch lane now also preserves MATLAB's
distinction between omitted arguments and explicit empty placeholders, so
`sceneCreate('ramp', [], [], wave)`, `sceneCreate('linear intensity ramp', [], [], wave)`,
and `sceneCreate('exponential intensity ramp', [], [], wave)` now replay the
helper-level `128 x 128` and `256` dynamic-range defaults, while
`sceneCreate('vernier', [], [], [], [], [])` and
`sceneCreate('vernier', 65, [], [], [], [])` now reuse the lower-level
`sceneVernier(...)` placeholder defaults for size, bar width, and offset
instead of crashing on `[]`,
while the adjacent HDR constructor lane now also replays the upstream
`sceneCreate(...)` defaults for `hdr chart` and `hdr image`, so no-argument
calls match the MATLAB dispatcher’s `10^3.5` / `16` / `12` strip-chart
defaults and `8`-patch HDR-image default, while explicit empty placeholders
in the public wrapper forms reuse the helper-level defaults instead of
crashing,
and
`sceneCreate('moire orient', imageSize, f)` now maps the positional
arguments onto the same headless parameter path as the existing struct-style
moire target form,
and
`sceneCreate('zone plate', imageSize, fieldOfView, wave)` now accepts the
optional field-of-view positional slot while preserving the existing
wave-only shorthand and the upstream MATLAB distinction between the
omitted-size wrapper default `384 x 384` and the empty-placeholder helper
default `256 x 256`, both at `4 deg`,
and
the numeric `sceneCreate('vernier', ...)` shortcut now replays the upstream
MATLAB defaults of `65` pixels, `3`-pixel bars, and `3`-pixel offset instead
of the lower-level helper defaults,
and
the plain `sceneCreate('bar')` shortcut now replays MATLAB's default
3-pixel bar width instead of inheriting the wider helper-level default,
while the documented `sceneCreate('bar ee', ...)` alias now uses the
equal-energy bar spectrum instead of collapsing onto plain `bar`,
while the vendor sensor dispatcher now also unwraps single-entry
MATLAB-style cell/list variant payloads for calls such as
`sensorCreate('MT9V024', [], {'rgbw'})` and
`sensorCreate('ar0132at', [], {'rccc'})` onto the existing vendor preset
builders,
while the custom multispectral sensor lane now preserves the explicit pixel
object and MATLAB-style empty placeholders in
`sensorCreate('custom', pixel, filterPattern, filterFile, [], [])`, so the
default sensor size comes from the CFA pattern and the wavelength grid stays
attached to the provided pixel instead of crashing or falling back to the
default pixel,
and
the `ramp` / `linear intensity ramp` / `exponential intensity ramp`
shortcuts now replay the upstream default `256 x 256` scene size instead of
the smaller helper-era default,
and
the plain `sceneCreate('disk array')` shortcut now replays MATLAB's default
`128`-pixel disk radius instead of the smaller helper-era default,
and
`sceneCreate('letter', 'g', fontSize, fontName, display)` now replays the
documented text shorthand on top of the existing font-object path. The
same scene lane now also accepts the documented `sceneCreate('vernier',
type, params)` shorthand on top of the lower-level
`sceneVernier(scene, type, params)` helper path. It now also preserves the
documented default reflectance-chart sources and sample counts when
MATLAB-style empty placeholders are used in calls like
`sceneCreate('reflectance chart', 24, [], [], wave, [], [])`, instead of
coercing `[]` into invalid `sSamples` or `sFiles` payloads. The same text
shorthand now also honors MATLAB-style empty placeholders in its optional
`fontSize`, `fontName`, and `display` slots, so
`sceneCreate('letter', 'g', [], [], [])` reuses the default Georgia 14 pt
font on `LCD-Apple` instead of crashing or stringifying `[]` into bad
font metadata. The same font-object lane now also treats
`sceneCreate('letter', [], [])` as “use the default font/display” instead
of passing `[]` through as a bogus font object. The same positional moire-target shorthand now also treats
an empty second argument as “use the default frequency”, so
`sceneCreate('moire orient', imageSize, [])` reuses the helper's default
`f` parameter instead of attempting `float([])`.
That same frequency-orientation scene lane now also treats an explicit
empty parameter payload `sceneCreate('frequency orientation', [])` as
“use the default `FOTParams` struct”, matching the upstream
`sceneFOTarget(...)` fallback instead of coercing `[]` into a bogus empty
image-size override. The adjacent helper-backed scene constructors now
also inherit that same empty-payload fallback through the shared
parameter normalizer, so `sceneCreate('moire orient', [])`,
`sceneCreate('hdr lights', [])`, and `sceneCreate('harmonic', [])`
now replay the same defaults as their no-argument upstream forms.
The adjacent `sceneCreate('dead leaves', imageSize, nFactor, [])`
wrapper now also ignores that explicit empty third slot the same way the
vendored MATLAB dispatcher does, instead of misreading it as a wavelength
override and changing the scene spectrum.
The same `sceneCreate('radiance chart', wave, radiance, ...)` wrapper now
also honors MATLAB-style empty placeholders in optional `rowcol`,
`patch size`, and `gray fill` slots for both dict and key/value forms,
reusing the helper defaults instead of attempting `int([])` or passing an
empty layout through to the chart core.
That same `sceneCreate('sweep frequency', size, maxFreq, wave, yContrast)`
wrapper now also honors MATLAB-style empty placeholders in the optional
`maxFreq` slot, so `sceneCreate('sweep frequency', 64, [], [], [])`
reuses the helper default of `size/16` instead of attempting `float([])`.
`sensor_create` now includes generic
`rgbw` / `grbc` / `rccc` presets plus upstream-backed `mt9v024` and `ar0132at`
RGBW/RCCC variants. The next sensorCreate dispatcher slice is now landed
too: `sensorCreate('light field', oi)` and `sensorCreate('light field',
pixel, oi)` reuse the headless light-field helper with MATLAB-style OI
name replay, `sensorCreate('dual pixel', [], oi, nMicrolens)` now mirrors
the upstream OI-sampled split-pixel geometry contract, the documented
multi-variant payloads for `sensorCreate('MT9V024', [], {'rgb','mono','rccc'})`
and `sensorCreate('ar0132at', [], {'rgb','rgbw','rccc'})` now return
tracked sensor lists in the requested order, and direct
`sensorCreate('ovt-large')` / `sensorCreate('ovt-small')` now expose the
existing OVT vendor presets. `sensorCreate('imec44', rowcol)` is now
mirrored by the adjacent camera wrapper too, so `cameraCreate(...)` now
accepts those multi-variant vendor payloads and the `ovt-large` dual-sensor
preset by returning per-sensor camera lists with matching OI, sensor, and IP
objects instead of crashing on list-backed sensor constructors. The same
sensor wrapper surface now also exposes the core MATLAB-style gateway aliases
directly from `pyisetcam.sensor`, so imports like `from pyisetcam.sensor
import sensorCreate`, `sensorCreateIdeal`, `sensorCompute`, `sensorGet`, and
`sensorSet` work without routing through the package root. The same
camera wrapper lane now also accepts `cameraCreate('current')`, reusing the
currently selected session `oi`, `sensor`, and `ip` independently and falling
back to defaults only for the missing pieces. It now also accepts explicit
`cameraCreate('L3', payload)` construction by replaying the supplied L3
payload's `oi` and `design sensor` entries into a headless camera/IP bundle;
the upstream no-argument default `L3defaultcamera` asset branch remains
unsupported because that vendored camera asset is not present in this repo. It
is also routed through the existing IMEC 4x4 multispectral builder, and
`sensorCreate('custom', ...)` / `sensorCreate('fourcolor', ...)` now
reuse the current headless custom filter-pattern helper. The same vendor
lane now also accepts the practical overload
`sensor_create('imx363', 'row col', [rows, cols], ...)` on top of the
already-supported explicit placeholder form.
`sensorCreate('human', params)` plus the explicit
`sensorCreate('human', pixel, params)` form now also replay the upstream
human-cone parameter-struct flow on top of the existing
`pixelCreate('human')` and `sensorCreateConeMosaic(...)` helpers. The same
dispatcher lane now also replays the documented empty-placeholder `[]`
forms for `sensorCreate('imec44', [], rowcol)`,
`sensorCreate('monochrome array', [], N)`, and the `custom` /
`fourcolor` constructors. Direct `sensorCreateIdeal('match', sensor)` and
`sensorCreateIdeal('match xyz', sensor)` now also replay the upstream
array-of-monochrome contract, preserving the example pixel and shared
exposure time while splitting filters into per-channel ideal sensors.
Direct `sensorCreateIdeal('xyz')` now also replays the upstream
three-sensor `CIE-X/Y/Z-ideal` array, while `cameraCreate('ideal')` now
again follows the vendored MATLAB wrapper and returns the default
monochrome ideal camera. Bare `sensorCreateIdeal()` now also follows the upstream default
again by returning the ideal monochrome sensor unless an explicit ideal
type is requested, and the documented MATLAB placeholder forms
`sensorCreateIdeal('monochrome', [], pixelSizeM)` and
`sensorCreateIdeal('xyz', [], pixelSizeM)` now honor the empty
sensor-example slot instead of crashing. The monochrome ideal sensor now
also reuses the upstream `Monochrome` sensor name instead of the older
Python-only `ideal-monochrome` label, and both ideal constructors now
accept either scalar or `[row, col]` pixel-size vectors like the vendored
MATLAB `idealPixel(...)` helper. The adjacent standalone
`pixelCreate('ideal', wave, pixelSizeM)` path now also accepts either
form instead of only square scalar pixels. Non-empty
`sensorCreateIdeal('monochrome', sensorExample, ...)` calls now also match
upstream by failing instead of silently templating from that example.
`sensorCreateArray(...)` now also supports the
ideal-array families
`'xyz'`, `'matchxyz'`, `'match'`, and `'monochrome'` instead of stopping at
the OVT split-pixel lane, and now also honors MATLAB-style empty
`sensor example` placeholders in those ideal-array forms instead of
failing early at the Python type gate. The deprecated top-level
`sensorCreate('ideal')` shim now also matches upstream by throwing an
explicit redirect to `sensorCreateIdeal(...)` instead of silently building
an XYZ ideal sensor. Adjacent camera creation now also replays the
upstream `cameraCreate('idealmonochrome')` naming contract, returning an
`ideal monochrome` camera instead of leaking the compact dispatcher token
into `camera.name`. The adjacent split-pixel array layer now also replays
the upstream `sensorCreateArray('array type','imx490', ...)` /
`sensorCreateSplitPixel('array type','imx490', ...)` four-capture
constructor, including the `large LCG`, `large HCG`, `small LCG`, `small HCG`
ordering and the 4x high-conversion-gain replay. That same split-pixel and
sensor-array lane now also treats `sensorCreateArray('array type', [])`
and `sensorCreateSplitPixel('array type', [])` as omitted array-type
slots, so the explicit empty placeholder falls back to the default `ovt`
path instead of throwing `UnsupportedOptionError`. The same ideal-array
wrapper now also surfaces the upstream error for
`sensorCreateArray('array type','monochrome','sensor example', sensor)`
instead of silently ignoring the provided example.
the dispatcher also exposes the legacy
`sensorCreate('Nikon D100')` preset from the vendored Nikon MAT asset. The
dispatcher also now accepts the remaining parenthesized Bayer CFA aliases
from upstream, including `bayer(grbg)`, `bayer(rggb)`, and `bayer(bggr)`.
It also now replays the upstream `sensorCreate('monochrome array', [], N)`
shortcut, including the default 3-sensor case and the practical Python
overload. That same monochrome-array lane now also treats
`sensorCreate('monochrome array', [], [])` as an omitted sensor-count slot, so
the explicit empty placeholder falls back to the same default 3-sensor bundle
instead of crashing on an empty array index. The same IMEC lane now also treats
`sensorCreate('imec44', [], [])` and `sensorCreateIMECSSM4x4vis('row col', [])`
as omitted `rowcol` inputs, so they fall through to the direct helper's
full-resolution default instead of crashing on the empty placeholder. The scene-constructor lane now also replays the older numeric
`sceneCreate('vernier', sz, width, offset, lineReflectance, backReflectance)`
shortcut on top of the current headless `sceneVernier(...)` object path.
`camera_create(...)` now forwards those expanded
sensor presets too, so camera constructors can directly target `rgbw`,
`rccc`, `mt9v024`, and `ar0132at` sensor variants. The first
metrics/validation slice is also in place:
`metrics_spd`, CIELAB helpers, and generic MAE/RMSE/PSNR utilities.
The same sensor/tutorial wave now also includes headless raw DNG support
through `ie_dng_read` / `ieDNGRead`, `ie_dng_simple_info` /
`ieDNGSimpleInfo`, and `sensor_dng_read` / `sensorDNGRead`, covering the
checked-in Pixel 4a `MCC-centered.dng` workflow from
`t_sensorReadRaw.m` and upstream-backed `sensorCreate('IMX363')` crop
flows. The raw DNG parity case now matches decoded `digital_values`
exactly and checks the rendered RGB result with a bounded normalized-MAE
comparator to absorb the remaining legacy display-gain mismatch between
the Python and Octave render paths. That same tutorial path now also includes headless
`sensor_plot_line` / `sensorPlotLine` coverage for the
`t_sensorSpatialResolution.m` line-profile workflow, including the exact
`sensorSet(..., 'pixel size Constant Fill Factor', ...)` tutorial path and
curated parity on the coarse/fine sensor-vs-OI line data, and headless
`sensor_description` / `sensorDescription` coverage for the
`t_sensorFPN.m` summary path, with curated parity on the returned summary
table fields for the fixed-noise setup used by that tutorial, plus direct
curated parity on the four `noise flag` modes and two-line volts plot used
by the rest of that tutorial, plus ROI-DV summary parity for
`s_sensorPoissonNoise.m`. The same sensor tutorial wave now also
includes `signal_current` / `signalCurrent` coverage for the
`t_sensorInputRefer.m` current-to-electrons workflow, plus
headless `signal_current_density` / `SignalCurrentDensity` and
`spatial_integration` / `spatialIntegration` coverage for the adjacent
legacy sensor-simulation current-density and default grid-integration
helpers, plus direct `analog2digital` / `noiseFPN` / `noiseColumnFPN`
coverage for the same legacy sensor-simulation quantization and fixed-pattern-noise helper surface, plus direct
`sensorRGB2Plane` / `sensorStats` / `sensorCheckArray` coverage for the
adjacent MATLAB CFA-packing, ROI-summary, and CFA-visibility helper
surface, plus
`ie_read_spectra` / `ieReadSpectra` coverage for the core spectral-loading
path used by `t_sensorEstimation.m`, together with curated parity on the
Macbeth/D65/scanner pseudoinverse estimation workflow,
plus direct curated parity on the daylight-basis Macbeth illuminant solve
from `s_sensorMacbethDaylightEstimate.m`,
plus `sensor_color_filter` / `sensorColorFilter`,
`ie_read_color_filter` / `ieReadColorFilter`, and
`ie_save_color_filter` / `ieSaveColorFilter` coverage for the Gaussian
color-filter roundtrip workflow from `s_sensorGaussianFilter.m`,
plus direct curated parity on calibrated `ieReadColorFilter('NikonD100')`
asset loading from `s_sensorPlotColorFilters.m`,
plus direct curated parity on the simulated filter-responsivity and
spectral-filter recovery workflow from `s_sensorSpectralEstimation.m`,
plus headless `sensor_mpe30(...)` / `sensorMPE30(...)`,
`sensor_pd_array(...)` / `sensorPDArray(...)`,
`sensor_jiggle(...)` / `sensorJiggle(...)`, and
`sensor_wb_compute(...)` / `sensorWBCompute(...)` for the remaining
legacy sensor utility helpers built on top of the existing SNR, pixel,
MAT-object load, and sensor-compute paths,
plus headless `bin_noise_fpn(...)` / `binNoiseFPN(...)`,
`bin_noise_column_fpn(...)` / `binNoiseColumnFPN(...)`, and
`bin_noise_read(...)` / `binNoiseRead(...)` for the legacy sensor
binning-noise helpers layered directly on the existing fixed-pattern and
read-noise simulation paths,
plus direct curated parity on the CFA construction workflow from
`s_sensorCFA.m`, including `sensorCreate('ycmy')` and
`sensorSet(..., 'pattern and size', pattern)` behavior,
plus
`filter transmissivities` and overexposed `ipCompute` color-shift coverage
from `t_sensorExposureColor.m`, and `ieFitLine`-driven dark-voltage
estimation coverage for `s_sensorAnalyzeDarkVoltage.m`, plus slope-based
PRNU estimation coverage for `s_sensorSpatialNoisePRNU.m`, plus DSNU
estimation coverage for `s_sensorSpatialNoiseDSNU.m`, plus the
`s_sensorCountingPhotons.m` workflow with MATLAB-style
`sceneCreate('uniform equal photon', [rows cols])`,
`sceneSet(..., 'mean luminance', ...)`, and
`oiGet(..., 'optics aperture diameter', unit)` support, plus headless
`sensor_formats` / `sensorFormats` and `ie_n_to_megapixel` /
`ieN2MegaPixel` coverage for `s_sensorSizeResolution.m`, plus direct
curated parity on the `s_sensorCFA.m` zebra-scene workflow through the
default Bayer, explicit Bayer, YCMY, custom 3x3 RGB, RGBW, and quad-CFA
branches, including MATLAB-style
`cameraSet(..., 'pixel size constant fill factor', ...)` routing, exact
pattern/size metadata, and normalized sensor-RGB summaries, plus direct
curated parity on the cropped CFA point-image workflow from
`s_sensorCFAPointSpread.m`, comparing the 12x12 CFA-image crops,
rendered RGB summaries, and canonical center-line profiles across the
`f/#` sweep, plus direct
curated parity for the `s_sensorSNR.m` sensor-noise component curves
returned by `sensorSNR(...)`, plus direct curated parity for the
`s_sensorExposureBracket.m` stacked multi-exposure sensor workflow, plus
direct curated parity for the CFA-matched exposure-duration workflow from
`s_sensorExposureCFA.m`, plus headless `sensor_create_array` /
`sensorCreateArray`, `sensor_create_split_pixel` / `sensorCreateSplitPixel`,
and `sensor_compute_array` / `sensorComputeArray` coverage for the
OVT split-pixel saturated-combine workflow that underlies
`s_sensorSplitPixel.m`. That same sensor wave now also includes
sensor-array `sensor_compute(...)` support together with
`ie_read_spectra('Foveon'/'NikonD1')` sensor-color-filter asset loading,
covering the stacked monochrome-plane and triple-well image-processing
workflow from `s_sensorStackedPixels.m` with curated parity on the stable
stacked-plane and Bayer-comparison summaries. That same Phase 1 sensor
wave now also includes direct `mlensCreate` / `mlensSet` / `mlensGet` /
`mlRadiance` / `mlAnalyzeArrayEtendue` coverage for the array-efficiency
and optimal-offset workflow from `s_sensorMicrolens.m`, with curated
Octave parity on the no-microlens/centered/optimal etendue maps, the
optimal-offset curve and offset maps, and the single-pixel irradiance
midlines used in the script’s chief-ray-angle illustration. That same
microlens block now also includes headless `ml_get_current(...)` /
`mlGetCurrent(...)`, `ml_set_current(...)` / `mlSetCurrent(...)`,
`ml_import_params(...)` / `mlImportParams(...)`,
`ml_description(...)` / `mlDescription(...)`, and
`ml_print(...)` / `mlPrint(...)`, while the legacy GUIDE microlens window
controllers are now explicitly treated as out of scope in the migration
audit. That same
sensor wave now also includes MATLAB-style `sceneSet(..., 'resize', ...)`,
`sceneCombine(...)`, and `sensorSet(..., 'hfov'/'vfov', ...)` coverage for
the mixed-sensor comparison workflow from `s_sensorComparison.m`, with
curated Octave parity on the combined-scene geometry plus the IMX363,
MT9V024 RCCC, and CYYM small-pixel/large-pixel response summaries. That
same sensor wave now also includes headless `sensorComputeSamples(...)`
for the repeated-capture workflow from `s_sensorNoise.m`, with curated
Octave parity on stable sample-stack noise statistics instead of raw
RNG-matched captures. That same sensor compatibility surface now also
includes headless `sensor_clear_data(...)` / `sensorClearData(...)`,
matching the direct MATLAB computed-payload reset path for sensor
objects, plus headless `sensor_show_image(...)` / `sensorShowImage(...)`
as a direct wrapper over the existing sensor RGB render path using the
MATLAB default `dv or volts` selection, plus headless
`sensor_save_image(...)` / `sensorSaveImage(...)` for the direct MATLAB
PNG-export helper on top of the same rendered sensor RGB path, plus
headless `sensor_show_cfa(...)` / `sensorShowCFA(...)` for the direct
MATLAB CFA-pattern render helper on top of the same sensor rendering
surface, plus headless `sensor_image_color_array(...)` /
`sensorImageColorArray(...)` for the legacy CFA-letter to color-order map
helper used by the older sensor CFA utilities, plus headless
`sensor_show_cfa_weights(...)` / `sensorShowCFAWeights(...)` for the
weighted CFA-visualization helper used by the same legacy sensor CFA
utilities, plus headless `sensor_color_order(...)` / `sensorColorOrder(...)`
and `sensor_determine_cfa(...)` / `sensorDetermineCFA(...)` for the legacy
CFA color-hint and tiled-pattern helpers used across the older sensor CFA
utilities, plus headless `sensor_snr_luxsec(...)` / `sensorSNRluxsec(...)`
for the direct MATLAB lux-sec sensor-SNR helper on top of the existing
`sensorSNR(...)` / `pixelVperLuxSec(...)` path, plus headless
`sensor_display_transform(...)` / `sensorDisplayTransform(...)` for the
direct sensor-channel-to-display transform, `sensor_equate_transmittances(...)`
/ `sensorEquateTransmittances(...)` for the legacy equal-area filter
normalizer, and `sensor_filter_rgb(...)` / `sensorFilterRGB(...)` for the
legacy CFA filter-color approximation helper, plus headless
`sensor_cfa_name_list(...)` / `sensorCFANameList(...)` for the legacy CFA
popup-name list and `sensor_pixel_coord(...)` / `sensorPixelCoord(...)`
for the direct MATLAB pixel-center coordinate helper, plus headless
`sensor_no_noise(...)` / `sensorNoNoise(...)`,
`sensor_gain_offset(...)` / `sensorGainOffset(...)`, and
`sensor_resample_wave(...)` / `sensorResampleWave(...)` for the legacy
sensor noise-reset, analog gain/offset, and spectral-resampling helper
mini-family, plus headless `sensor_rescale(...)` / `sensorRescale(...)`,
`sensor_cfa_save(...)` / `sensorCfaSave(...)`, and
`sensor_from_file(...)` / `sensorFromFile(...)`, and
`load_raw_sensor_data(...)` / `LoadRawSensorData(...)` for the legacy
sensor geometry/CFA-persistence/file-load/raw-loader helper mini-family,
plus headless `sensor_check_human(...)` / `sensorCheckHuman(...)`,
`sensor_create_cone_mosaic(...)` / `sensorCreateConeMosaic(...)`,
`sensor_human_resize(...)` / `sensorHumanResize(...)`, and
`sensor_light_field(...)` / `sensorLightField(...)` for the remaining
human-cone and light-field sensor helper surface, while treating the
plot-only `sensorConePlot.m` and OpenEXR export helper `sensor2EXR.m` as
out of scope,
plus headless `sensor_create_imec_ssm_4x4_vis(...)` /
`sensorCreateIMECSSM4x4vis(...)`, `sensor_imx363_v2(...)` /
`sensorIMX363V2(...)`, `sensor_interleaved(...)` /
`sensorInterleaved(...)`, and `sensor_mt9v024(...)` /
`sensorMT9V024(...)` for the remaining reusable MATLAB sensor-model
helper surface, while treating the exploratory `s_sensorIMX490Test.m`
demo script as out of scope,
and the migration-gap audit now also treats the GUIDE-only
`sensor/cfadesign` family (`cfaDesign.m`, `cfaDesignCallbacks.m`,
`cfaDesignUI.m`) plus the unimplemented `cfaDesignUtilities.m` example
stub as out of scope rather than remaining headless API debt,
plus headless
`pixel_create(...)` / `pixelCreate(...)`,
`pixel_get(...)` / `pixelGet(...)`, `pixel_set(...)` / `pixelSet(...)`,
`pixel_ideal(...)` / `pixelIdeal(...)`, `pixel_sr(...)` / `pixelSR(...)`,
and `ie_pixel_well_capacity(...)` / `iePixelWellCapacity(...)` for the
legacy standalone pixel helper mini-family built on top of the same
internal sensor-pixel state, plus headless
`pv_full_overlap(...)` / `pvFullOverlap(...)`,
`pv_reduction(...)` / `pvReduction(...)`, `bin_pixel(...)` /
`binPixel(...)`, and `bin_pixel_post(...)` / `binPixelPost(...)` for the
remaining reusable pixel-vignetting and binning helper surface, plus
headless
`pixel_position_pd(...)` / `pixelPositionPD(...)`,
`pixel_center_fill_pd(...)` / `pixelCenterFillPD(...)`, and
`pixel_description(...)` / `pixelDescription(...)` for the legacy
photodetector-positioning and pixel-summary helper layer, plus headless
`pixel_transmittance(...)` / `pixelTransmittance(...)` together with the
direct dielectric-stack helpers `ptSnellsLaw(...)`,
`ptReflectionAndTransmission(...)`, `ptInterfaceMatrix(...)`,
`ptPropagationMatrix(...)`, `ptScatteringMatrix(...)`,
`ptPoyntingFactor(...)`, and `ptTransmittance(...)` for the buried
photodetector tunnel-transmission path, plus headless
`sensor_read_color_filters(...)` / `sensorReadColorFilters(...)`,
`sensor_read_filter(...)` / `sensorReadFilter(...)`,
`sensor_add_filter(...)` / `sensorAddFilter(...)`,
`sensor_replace_filter(...)` / `sensorReplaceFilter(...)`, and
`sensor_delete_filter(...)` / `sensorDeleteFilter(...)` for the legacy
sensor filter-management helper mini-family on top of the same filter
spectra and asset-loading paths. The migration-gap
audit now also treats the explicitly obsolete upstream `sensorUnitBlock.m`
stub as out of scope rather than a remaining headless API target. That
same Phase 1 sensor wave now also includes
public `sceneRotate(...)` / `oiCrop(...)` support for the deterministic
rolling-shutter assembly path from `s_sensorRollingShutter.m`, with
curated Octave parity on the per-frame mean-voltage trace, final
row summaries, and normalized final RGB summaries. That same Phase 1
sensor wave now also includes `sensorCreate('imx490-large')`,
`sensorCreate('imx490-small')`, `oiCrop(..., 'border')`,
`oiSpatialResample(...)`, and headless `imx490Compute(...)` coverage for
the HDR four-capture combine workflow from `s_sensorIMX490.m`, with
curated Octave parity on the deterministic uniform-field best-SNR path,
including capture names, capture mean electrons/volts/DV, combined-volts
summaries, and selected pixel-source counts. That same Phase 1 sensor
wave now also includes `sceneFromFile(..., 'multispectral', ...)`
support for basis-coded HDR scene assets plus the mixed pixel-size
monochrome workflow from `s_sensorHDR_PixelSize.m`, with curated Octave
parity on the Feng Office HDR scene path, comparing the three sensor
sizes, mean/p95 volts, mean electrons, and grayscale output summaries
for the 1/2/4 micron pixel-size sweep. That same Phase 1 sensor wave now
also includes `sceneCreate('linear intensity ramp')`,
`sceneCreate('exponential intensity ramp')`, public `sensor_dr` /
`sensorDR`, and headless log-response sensor compute support for
`s_sensorLogAR0132AT.m`, with curated Octave parity on the stable
noise-free log-response line-shape workflow and the script’s 1-second
dynamic-range check. That same Phase 1 sensor wave now also includes the
full `s_sensorAliasing.m` workflow, with curated Octave parity on the
noise-free sweep-frequency aliasing line profiles for the fine-pixel,
coarse-pixel, and blurred-lens cases plus canonicalized slanted-bar
sensor captures for the sharp and anti-aliased large-pixel paths. That
same Phase 1 sensor wave now also includes the asset-driven DUT setup
workflow from `s_sensorExternalAnalysis.m`, with curated Octave parity on
the configured wave/filter/QE payload, CFA pattern, pixel metadata, and
imported `dutData.mat` voltage summaries. That same Phase 1 sensor wave
now also includes the spectral-radiometer workflow from
`s_spectralRadiometer.m`, with curated Octave parity on the radiometer
filter geometry, deterministic noise-free electron line, theoretical
shot-noise curve, and stable photon-noise summary statistics.
Phase 2 optics coverage now also includes direct `optics_coc` /
`opticsCoC` support for the thin-lens circle-of-confusion workflow from
`s_opticsCoC.m`, with curated Octave parity on the 50 mm F/2 and F/8
diameter-versus-distance sweeps at 0.5 m and 3 m focus.
That same Phase 2 optics wave now also includes the wide-field
`s_opticsCos4th.m` vignetting workflow, with curated Octave parity on the
default and 4x-focal-length illuminance hline profiles and their mean
illuminance summaries for a uniform D65 scene at 80 degrees FOV.
That same optics wave now also includes MATLAB-style
`scene_interpolate_w` / `sceneInterpolateW` behavior through
`sceneSet(..., 'wave', ...)`, plus curated parity for the
`s_opticsGaussianPSF.m` point-array workflow on the normalized blurred
center row and column profiles across 450 nm, 550 nm, and 650 nm.
That same Phase 2 optics wave now also includes the diffraction-limited
`s_opticsPSFPlot.m` workflow, with curated parity on the 600 nm, `f/#=12`,
`nSamp=100` PSF surface and Airy-disk radius returned by the underlying
headless PSF data contract.
That same optics wave now also includes the `s_opticsPadCrop.m` workflow,
with curated parity on the padded OI size, MATLAB crop rectangle, cropped
OI geometry, and the deterministic sensor line-profile outputs for both
the scene-FOV crop path and the expanded padded-FOV path.
That same Phase 2 optics wave now also includes the `s_opticsMicrolens.m`
workflow, with curated parity on the microlens getter/setter contract,
default 30-degree sensor FOV setup, and the radiance-generated source and
pixel irradiance center-line outputs plus etendue.
That same Phase 2 optics wave now also includes the
`s_opticsFlare.m` workflow, with upstream-style `sceneCreate('hdr')`
support via the HDR-lights scene family, deterministic seeded flare
apertures for the Python parity path, and curated Octave parity on the
dirty-aperture summaries, normalized 550 nm PSF rows, blur-width
statistics, and HDR/point-scene OI geometry returned by the script.
That same Phase 2 optics wave now also includes the
`s_opticsFlare2.m` workflow, with curated Octave parity on the stable
dirty-aperture script contract: aperture sums, point-scene blur-width
summaries, HDR-scene geometry, and normalized HDR irradiance ratios
across the six-sided, five-sided, and defocused branches.
That same Phase 2 wavefront block now also includes `ie_mvnrnd` /
`ieMvnrnd` together with the `s_wvfThibosModel.m` workflow, with curated
Octave parity on deterministic virtual-eye coefficient draws, the
sample-mean subject PSFs across 450/550/650 nm, and the selected
example-subject PSF center rows at 450 nm and 550 nm.
That same Phase 2 wavefront block now also includes the
`s_wvfZernikeSet.m` workflow, with curated Octave parity on the
defocus-plus-vertical-astigmatism loop, comparing the returned Zernike
coefficients, 550 nm PSF center rows, and uncropped 550 nm OI center
rows for `A = [-1, 0, 1]` at `D = 2` microns.
The optics expansion now also includes script-driven
`oi_create('psf')` / `oiCreate('psf')` support for shift-invariant PSF optics,
including a MATLAB-style default synthetic PSF constructor and explicit
`psf` / `wave` / `umPerSamp` custom PSF data.
The same shift-invariant optics surface now also includes
`si_synthetic` / `siSynthetic` for the script-driven synthetic PSF
families from `s_opticsSIExamples.m`, including MATLAB-style Gaussian,
Lorentzian, pillbox, and custom-PSF workflows that return
shift-invariant optics ready for `oiSet(..., 'optics', optics)`.
The same shift-invariant optics surface now also includes
`optics_psf_to_otf` / `opticsPSF2OTF` and
`oi_set(..., 'optics otfstruct', ...)` for MATLAB-style custom OTF
injection from flare PSFs, with a curated Octave parity case on the
upstream flare image; that case compares `photons` with a scale-invariant
rule because the upstream Octave path preserves image shape but returns a
constant-factor magnitude offset.
The adjacent custom shift-invariant file workflow now also includes
`ie_save_si_data_file` / `ieSaveSIDataFile` plus direct
`siSynthetic('custom', file)` parity, with the case comparing the saved
input PSF row directly and treating the downstream OI `photons` field
scale-invariantly for the same stable custom-OTF magnitude offset.
The same raw OTF workflow now also includes direct
`oi_get(..., 'optics OTF')` / `oi_set(..., 'optics OTF', ...)` support
for MATLAB-style shift-invariant OTF access and replacement, with a
curated Octave parity case that replaces the stored OTF with an ideal
all-ones OTF and compares the resulting image using a normalized-MAE rule
that stays stable in dark regions.
The script-driven synthetic-PSF workflow is also parity-covered by a
Lorentzian `siSynthetic` case that compares the final irradiance cube
scale-invariantly, because the upstream Octave path preserves image shape
while retaining a stable magnitude offset.
That same Lorentzian `siSynthetic(...)` workflow now also has direct
Octave parity on `oiPlot(..., 'psf', 550)`, comparing the returned PSF
surface and spatial support grids on the shift-invariant optics object
before image formation.
That same synthetic-PSF surface now also has direct Octave parity on the
pillbox `siSynthetic(...)` workflow from `s_opticsSIExamples.m`,
including the returned wave axis and the 550 nm input PSF center row.
That same synthetic-PSF surface now also has direct Octave parity on the
anisotropic Gaussian `siSynthetic(...)` workflow from
`s_opticsSIExamples.m`, including the returned wave axis plus the 550 nm
input PSF center row.
That same anisotropic Gaussian workflow now also has direct Octave parity
on `oiPlot(..., 'psf', 550)`, comparing the returned PSF surface and
spatial support grids at 550 nm.
That same anisotropic Gaussian workflow now also has direct Octave parity
on the `oiPlot(..., 'illuminance hline'/'illuminance vline', ..., 'nofigure')`
comparison used in `s_opticsSIExamples.m`, comparing the returned
horizontal and vertical illuminance line profiles through the image center.
The adjacent wavefront helper surface now also includes
`wvf_set` / `wvfSet`, `wvf_get` / `wvfGet`,
`wvf_compute` / `wvfCompute`,
`wvf_defocus_diopters_to_microns` /
`wvfDefocusDioptersToMicrons`, `wvf_to_oi` / `wvf2oi`, direct
`oi_compute(wvf, scene)` support, MATLAB-style `oi_get(..., 'wvf ...')`,
and rebuilding `oi_set(..., 'wvf ...')` / `oi_set(..., 'optics wvf', ...)`
for script-driven WVF defocus/Zernike workflows.
That same WVF-to-OI bridge now also has direct Octave parity on the
`s_wvfOI.m` PSF-slice workflow, comparing both
`wvfGet(..., 'psf xaxis', ...)` / `wvfGet(..., 'psf yaxis', ...)` against
`oiGet(wvf2oi(wvf), 'optics psf xaxis'/'optics psf yaxis', ...)` at
550 nm.
That same explicit wavefront surface now also includes
`wvf_pupil_function` / `wvfPupilFunction` and
`wvf_compute_psf` / `wvfComputePSF`, including the MATLAB-style
stored-aperture workflow and direct parity coverage on the standalone
pupil-function/PSF path.
That same explicit WVF helper surface now also includes
`psf_to_zcoeff_error` / `psf2zcoeff`, plus unit-aware/index-aware
`wvfGet(..., 'wave', unit, idx)`, unit-aware
`wvfGet(..., 'measured wavelength', unit)`, and MATLAB-style
`wvfGet(..., 'z pupil diameter')` support for the upstream
`s_opticsPSF2Zcoeffs.m` workflow.
The same WVF backend now also includes
`wvf_aperture` / `wvfAperture` plus `wvf_aperture_params` / `wvfApertureP`
and now includes `wvf_load_thibos_virtual_eyes` /
`wvfLoadThibosVirtualEyes` for the upstream pupil-size scripts, with the
MATLAB default mean-vector return plus an explicit full-output path for
covariance and measured-subject coefficient matrices.
That same pupil-size workflow now also supports MATLAB key/value
`wvfCreate(...)` construction together with `lcaMethod='human'`, with
direct Octave parity on the Thibos pupil-size compute path.
That same pupil-size workflow now also has direct Octave parity on the
measured-vs-calculated pupil comparison path from `s_wvfPupilSize.m`,
holding the calculated pupil fixed while varying the measured Thibos
coefficient set.
The same WVF surface now also supports the `s_wvfPSFSpacing.m` numeric
path through `wvfSet(..., 'psf sample spacing', ...)`, with direct Octave
parity on the resulting field-size and sampling relationship.
That same script-driven WVF surface now also has direct Octave parity on
the defocus/vertical-astigmatism sweep from `s_wvfAstigmatism.m`,
comparing the returned normalized PSF center rows and columns at 550 nm
across the full 3x3 Zernike grid.
That same script-driven WVF surface now also has direct Octave parity on
the diffraction workflow from `s_wvfDiffraction.m`, comparing the initial
Airy-disk match and `wvf2oi` f-number handoff, the pupil-size sweeps at
550 nm and 400 nm, the human-LCA wavelength sweep, and the focal-length
`um per degree` scaling.
That same script-driven WVF helper surface now also includes
`wvfOSAIndexToZernikeNM` / `wvfZernikeNMToOSAIndex`, with direct Octave
parity on the OSA index round-trip used by `s_wvfWavefronts.m`.
That same script-driven WVF helper surface now also includes headless
`wvf_clear_data(...)` / `wvfClearData(...)`, `wvf_wave_to_idx(...)` /
`wvfWave2idx(...)`, and `wvf_osa_index_to_vector_index(...)` /
`wvfOSAIndexToVectorIndex(...)` for the legacy MATLAB wavefront utility
layer built on top of the existing WVF core.
That same script-driven WVF helper surface now also includes headless
`wvf_osa_index_to_name(...)` / `wvfOSAIndexToName(...)`, covering the
legacy OSA-index-to-aberration-name lookup used by the MATLAB wavefront
toolbox.
That same script-driven WVF helper surface now also includes headless
`sce_create(...)` / `sceCreate(...)` and `sce_get(...)` / `sceGet(...)`,
plus `wvfGet(..., 'sce'/'sce rho'/'scex0'/'scey0'/'sce wavelengths')`
support for the standalone Stiles-Crawford parameter helpers used by the
legacy wavefront toolbox.
That same script-driven WVF helper surface now also includes headless
`wvf_root_path(...)` / `wvfRootPath(...)`, `wvf_summarize(...)` /
`wvfSummarize(...)`, and `wvf_print(...)` / `wvfPrint(...)` for the
legacy toolbox-location and text-summary helpers built on top of the same
WVF core metadata.
That same script-driven WVF helper surface now also includes headless
`wvf_compute_cone_psf(...)` / `wvfComputeConePSF(...)`,
`wvf_compute_cone_average_criterion_radius(...)` /
`wvfComputeConeAverageCriterionRadius(...)`,
`wvf_compute_pupil_function_custom_lca(...)` /
`wvfComputePupilFunctionCustomLCA(...)`,
`wvf_compute_pupil_function_custom_lca_from_master(...)` /
`wvfComputePupilFunctionCustomLCAFromMaster(...)`, and
`wvf_compute_pupil_function_from_master(...)` /
`wvfComputePupilFunctionFromMaster(...)`, together with direct
`wvfGet(..., 'calc cone psf info'/'cone psf'/'sce fraction'/'cone sce fraction')`
support for the remaining legacy wavefront cone-weighting and pupil-function
compatibility helpers.
That same script-driven WVF helper surface now also includes headless
`wvf_key_synonyms(...)` / `wvfKeySynonyms(...)`, `wvf_to_si_psf(...)` /
`wvf2SiPsf(...)`, and `wvf_apply(...)` / `wvfApply(...)`, matching the
legacy key-canonicalization, shift-invariant PSF export, and deprecated
scene-to-OI compatibility wrappers on top of the existing WVF compute path.
That same script-driven WVF helper surface now also includes headless
`psf_find_peak(...)` / `psfFindPeak(...)`, `psf_volume(...)` /
`psfVolume(...)`, `psf_center(...)` / `psfCenter(...)`, and
`psf_find_criterion_radius(...)` / `psfFindCriterionRadius(...)` for the
legacy MATLAB PSF utility layer built on top of the existing WVF/PSF
compute path.
That same script-driven WVF helper surface now also includes headless
`psf2lsf(...)`, `lsf2circularpsf(...)`, `psfCircularlyAverage(...)`, and
`psfAverageMultiple(...)` for the legacy MATLAB PSF/LSF conversion and
multi-PSF averaging helpers layered on top of the existing WVF/PSF
compute path.
That same script-driven WVF helper surface now also includes headless
`wvf_to_psf(...)` / `wvf2PSF(...)`, `wvf_to_optics(...)` /
`wvf2optics(...)`, and `wvf_pupil_amplitude(...)` /
`wvfPupilAmplitude(...)` for the direct MATLAB PSF/optics conversion
helpers layered on top of the existing WVF compute path.
That same script-driven WVF surface now also has direct Octave parity on
the 16-coefficient OSA-index sweep from `s_wvfWavefronts.m`, comparing
the returned spatial support axis, normalized wavefront center rows and
columns, and peak absolute wavefront amplitude for indices 1:16 at a
2 mm pupil.
That same Phase 2 wavefront block now also has direct Octave parity on
`s_zernikeInterpolation.m`, comparing the interpolated-vs-ground-truth
Zernike coefficient vector at the target field height and the normalized
center-row PSFs for Zernike-space interpolation, ground truth, and the
script's direct space-interpolation path.
That same script-driven WVF plotting block now also has direct Octave
parity on the end-to-end sequence in `s_wvfPlot.m`, comparing the 550 nm
mixed-unit 1D PSF payloads and the recalculated 460 nm PSF-angle /
pupil-phase payloads after `wvfSet(..., 'wave', 460)` and `wvfCompute(...)`.
for script-driven synthetic aperture generation, including deterministic
clean polygon support used by the upstream flare scripts and direct
Octave parity on that clean-aperture path.
That same explicit WVF compute surface now also supports the MATLAB
key/value workflow `wvfCompute(wvf, 'aperture', aperture)`, with direct
parity on the script-driven polygon-aperture compute path used by the
upstream flare scripts.
The same script-driven optics surface now also includes headless
`oiPlot(..., 'psf'/'psf550'/'psfxaxis'/'psfyaxis')` support, backed by
computed `oiGet(..., 'psf data'/'psf xaxis'/'psf yaxis')` access for
diffraction-limited, custom-PSF, and computed WVF shift-invariant optics.
That same focus/DoF optics block now also has direct Octave parity on
`s_opticsDefocus.m`, comparing the blur/astigmatism/pupil-diameter WVF
sequence through mean-photon conservation, recovered Zernike coefficients,
pupil-diameter changes, and normalized 550 nm center-row irradiance slices.
That same focus/DoF optics block now also includes public
`optics_defocus_displacement` / `opticsDefocusDisplacement`, with direct
Octave parity on `s_opticsDefocusDisplacement.m` for the base-power sweep,
image-plane displacement curves, and constant displacement-to-focal-length
ratio check.
That same focus/DoF optics block now also has direct Octave parity on
`s_opticsDoF.m`, comparing the 100 mm F/2 thin-lens DOF formula result,
the CoC-derived DOF crossing estimate, and the object-distance by f-number
DOF sweep surface.
That same focus/DoF optics block now also includes public
`optics_depth_defocus` / `opticsDepthDefocus`, with direct Octave parity
on `s_opticsDepthDefocus.m` for the default-lens object-distance sweep,
the focal-plane and 1.1x-focal-length defocus curves, the in-focus object
distance for the shifted image plane, and the pupil-scaled Hopkins `w20`
surface.
That same focus/DoF optics block now also includes public
`optics_defocus_core` / `opticsDefocusCore` and
`optics_build_2d_otf` / `opticsBuild2Dotf`, with direct Octave parity on
`s_opticsDefocusScene.m` for the multispectral D65-adjusted defocus-image
workflow, the derived 10 um and 40 um sensor-plane defocus values, and
the normalized 550 nm center-row blur states for the focused and defocused
OI outputs.
That same focus/DoF optics block now also includes direct Octave parity on
`s_opticsDefocusWVF.m`, comparing the explicit diffraction-limited and
1.5-diopter WVF branches with the alternate `oiCreate('wvf')` update path
through their returned f-numbers, defocus coefficients, and normalized
550 nm PSF / OI center rows.
The adjacent diffraction-limited optics plotting surface now also includes
headless `oiPlot(..., 'ls wavelength'/'lswavelength'/'otf wavelength'/'mtf wavelength')`
support for the script-driven line-spread and OTF-by-wavelength workflows
used in the upstream diffraction PSF tutorials.
That same line-spread plotting surface now also has direct Octave parity on
the WVF-backed `oiPlot(..., 'ls wavelength')` workflow, comparing the
returned spatial axis, wavelength vector, and line-spread-by-wavelength
surface after `wvf2oi`.
That same OTF-by-wavelength plotting surface now also has direct Octave
parity on the diffraction-limited `oiPlot(..., 'otf wavelength')`
workflow, comparing the returned frequency axis, wavelength vector, and
OTF-by-wavelength surface.
That same WVF-backed OI plotting surface now also has direct Octave parity
on `oiPlot(..., 'otf wavelength')`, comparing the returned frequency axis,
wavelength vector, and OTF-by-wavelength surface after `wvf2oi`.
That same diffraction image-formation plotting surface now also has
direct Octave parity on `oiPlot(..., 'irradiance hline', [80 80])` from
`s_opticsImageFormation.m`, comparing the returned spatial axis,
wavelength vector, and irradiance mesh data for the diffraction-limited
line scene.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf', [], 550)` parity coverage for the upstream
`s_opticsDLPsf.m` workflow.
That same diffraction plotting surface now also includes the script-level
`s_opticsDiffraction.m` workflow, with direct Octave parity on the
default-to-`f/12` diffraction-limited point-array path, including the
unit-aware `oiGet(..., 'optics focal length'/'optics pupil diameter',
...)` parameter checks, the returned `psf 550` surface, the
`ls wavelength` payload, and stable 550 nm image-blur width summaries.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf xaxis', [], 550, 'um')` parity coverage for the
upstream diffraction and WVF spatial scripts.
That same diffraction plotting surface now also includes direct
`oiPlot(..., 'psf yaxis', [], 550, 'um')` parity coverage for the
upstream diffraction and WVF spatial scripts.
The script-driven custom optics surface now also includes direct
`opticsPSF2OTF(...)` parity coverage for the upstream flare-image
workflow from `s_opticsPSF2OTF.m`.
The adjacent `s_wvfOI.m` bridge is now covered on the direct OTF path
too: `wvf2oi(...)`-backed OIs can synthesize `oiGet(..., 'optics otf')`
from the embedded wavefront, and curated parity checks that against
`wvfGet(..., 'otf')` after the same `ifftshift` relationship used by the
MATLAB script.
The adjacent wavefront plotting surface now also includes headless
`wvfPlot(...)` support for script-driven PSF, 1D PSF, pupil amplitude,
pupil phase, wavefront-aberration, PSF-angle, and OTF views, backed by
wavelength-aware `wvfGet(...)` support for `psf`, `pupil function`,
`wavefront aberrations`, `psf spatial samples`, `psf angular samples`,
`pupil spatial samples`, `1d psf`, and OTF support.
That same WVF plotting surface now also has direct Octave parity on the
`wvfPlot(..., '2d otf', ...)` workflow from `s_wvfPlot.m`, including the
returned OTF support axis and center-row magnitude data.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '2d otf normalized', ...)`, including the returned OTF
support axis and normalized center-row magnitude data.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf', ...)`, including the returned cropped OTF support
axis and OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf normalized', ...)`, including the returned cropped
OTF support axis and normalized OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf angle', ...)`, including the returned cropped
frequency support axis and OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d otf angle normalized', ...)`, including the returned
cropped frequency support axis and normalized OTF center row.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '1d psf space', ...)`, including the returned spatial
support axis and 1D PSF line data.
That same WVF plotting surface now also has direct Octave parity on
`wvfPlot(..., '2d wavefront aberrations space', ...)`, including the
returned spatial support axis and wavefront-aberration center row.
The shift-invariant optics script coverage now also has direct Octave
parity on the circular Gaussian `siSynthetic('gaussian', ..., xyRatio=1)`
workflow from `s_opticsSIExamples.m`.
The same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image pupil amp', ...)`, including the returned
spatial support axis and pupil-amplitude center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d pupil amplitude space', ...)`, including the
returned spatial support axis and pupil-amplitude center row for the
exact legacy alias.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image pupil phase', ...)`, including the
returned spatial support axis and pupil-phase center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d pupil phase space', ...)`, including the
returned spatial support axis and pupil-phase center row for the exact
legacy plot name used by the upstream Zernike tutorial.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image wavefront aberrations', ...)`, including
the returned spatial support axis and wavefront-aberration center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf', ...)`, including the returned
spatial support axis and PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf', ..., 'airy disk', true)`, including
the returned spatial support axis, PSF center row, and Airy-disk radius.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf normalized', ...)`, including the
returned spatial support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf', 'unit', 'mm', ...)`, including the
returned cropped spatial support axis and PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf normalized', 'unit', 'mm', ...)`, including
the returned cropped spatial support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf xaxis', ..., 'airy disk', true)`, including
the full returned support/data payload and Airy-disk radius from
`s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf xaxis', ...)`, including the full returned
support/data payload from `s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf yaxis', ..., 'airy disk', true)`, including
the full returned support/data payload and Airy-disk radius from
`s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'psf yaxis', ...)`, including the full returned
support/data payload from `s_wvfSpatial.m`.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., 'image psf angle', ...)`, including the returned
angular support axis and PSF center row.
That same script-driven WVF plotting surface now also has Octave-backed
parity on the normalized `image psf angle` numerical contract,
including the returned angular support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d psf angle', ...)`, including the returned
angular support axis and PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '2d psf angle normalized', ...)`, including the
returned angular support axis and normalized PSF center row.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf', ...)`, including the returned cropped
spatial support axis and raw PSF line slice.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf normalized', ...)`, including the
returned cropped spatial support axis and normalized line slice.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf angle', ...)`, including the returned
cropped angular support axis and raw line slice.
That same script-driven WVF plotting surface now also has direct Octave
parity on `wvfPlot(..., '1d psf angle normalized', ...)`, including the
returned cropped angular support axis and normalized line slice.
That same optics helper surface now also includes `airy_disk` /
`airyDisk` plus headless airy-disk overlay payloads for `wvfPlot(...)`
and `oiPlot(...)`, with direct Octave parity on the scalar/image helper
contract used by the upstream diffraction plotting workflows.
That same script-driven WVF surface now also includes the spatial getter
family used by `s_wvfSpatial.m`, including `calc nwave`, `psf sample spacing`,
`ref psf sample interval`, `pupil sample spacing`, `pupil positions`,
`pupil function amplitude`, and `pupil function phase`.
That same script-driven WVF surface now also has direct Octave parity on
the scalar control workflow from `s_wvfSpatial.m`, comparing how `npixels`,
`pupil plane size`, and `focal length` change PSF sample spacing and
`um per degree`.
Initial ray-trace optics support is also in place: `oi_create('ray trace')`
loads the pinned upstream Zemax-derived optics asset, and `oi_compute`
applies upstream-backed geometric distortion, relative illumination, and
angle-aware precomputed ray-trace PSFs with cached reuse on repeated
computes, plus MATLAB-style uncropped padding/output bookkeeping and
`oiGet/oiSet` support for ray-trace PSF metadata, including MATLAB-style
`psf struct` normalization, sample angles, image heights, wavelengths, and
optics-name accessors, plus unit-aware raw ray-trace PSF spacing / field
height getters and derived support / frequency axes. Raw ray-trace geometry
and relative-illumination tables are also exposed now, including scalar
ray-trace metadata, unit-aware field-height getters, and MATLAB-style raw
table/function accessors. Raw scalar ray-trace metadata now round-trips
through `oiGet/oiSet` as well, including object-distance aliases and the
legacy `rtcomputespacing` field. MATLAB-style `oiGet/oiSet('optics ...')`
delegation now routes through the same compatibility surface too, and
whole-struct `optics` / `raytrace` replacement now normalizes MATLAB-style
payloads while invalidating stale cached PSFs. Raw ray-trace table fidelity
now also includes MATLAB-style `rtpsfsize` reporting of the underlying 4-D
PSF table, indexed `oiSet` updates for `rtpsfdata` and `rtgeomfunction`,
MATLAB-style raw struct export for `raytrace`, `rtpsf`, `rtgeometry`, and
`rtrelillum`, and MATLAB-style whole-optics export via `oiGet('optics')`
that round-trips through `oiSet('optics', ...)`. The precomputed ray-trace
cache now also exports as MATLAB-style `psfStruct` / `sampledRTpsf`, and
plain `rtpsfsize` now tracks the precomputed PSF kernel size separately
from raw `optics rtpsfsize`. The upstream `rtPSFInterp` helper is also
ported now as `rt_psf_interp` / `rtPSFInterp` for direct interpolation of
raw ray-trace PSFs by field height, angle, and wavelength. The upstream
`rtDIInterp` / `rtRIInterp` helpers are also ported as `rt_di_interp` /
`rtDIInterp` and `rt_ri_interp` / `rtRIInterp`, and the ray-trace geometry
path now routes through those helper ports. The MATLAB helper surface now
also includes `rt_sample_heights` / `rtSampleHeights`, `rt_psf_grid` /
`rtPSFGrid`, and `rt_angle_lut` / `rtAngleLUT`. The staged ray-trace
pipeline is now partially exposed as well through `rt_geometry` /
`rtGeometry`, `rt_precompute_psf` / `rtPrecomputePSF`, and
`rt_precompute_psf_apply` / `rtPrecomputePSFApply`. The legacy
`rt_psf_apply` / `rtPSFApply` name is now exposed too and routes through
that validated cached-PSF path. The MATLAB-style top-level wrapper
`optics_ray_trace` / `opticsRayTrace` is now exposed as the staged
end-to-end ray-trace entry point as well, including cached OI illuminance
bookkeeping. The supporting OI helpers `oi_calculate_illuminance` /
`oiCalculateIlluminance` and `oi_diffuser` / `oiDiffuser` are now public
too, and `opticsRayTrace` routes its blur/illuminance path through them.
That same headless OI utility surface now also includes
`oi_clear_data(...)` / `oiClearData(...)`,
`oi_show_image(...)` / `oiShowImage(...)`, and
`oi_save_image(...)` / `oiSaveImage(...)` for the legacy MATLAB
computed-payload reset, direct RGB render, and PNG-export helpers.
That same headless OI utility surface now also includes
`oi_frequency_resolution(...)` / `oiFrequencyResolution(...)`,
`oi_spatial_support(...)` / `oiSpatialSupport(...)`, and
`oi_space(...)` / `oiSpace(...)` for the direct MATLAB OI geometry
helper mini-family.
That same headless OI utility surface now also includes
`oi_psf(...)` / `oiPSF(...)` for the legacy MATLAB OI point-spread
summary helper, covering thresholded PSF area and diameter
measurements.
That same headless OI utility surface now also includes
`oi_calculate_irradiance(...)` / `oiCalculateIrradiance(...)`,
`oi_adjust_illuminance(...)` / `oiAdjustIlluminance(...)`,
`oi_interpolate_w(...)` / `oiInterpolateW(...)`,
`oi_extract_waveband(...)` / `oiExtractWaveband(...)`, and
`oi_add(...)` / `oiAdd(...)` for the legacy MATLAB OI spectral and
irradiance helper mini-family.
That same headless OI utility surface now also includes
`oi_extract_bright(...)` / `oiExtractBright(...)` and
`oi_from_file(...)` / `oiFromFile(...)`, matching the legacy MATLAB
brightest-patch extraction helper and the scene-to-ray-trace OI wrapper
on top of the existing `sceneFromFile(...)` / `sceneAdjustIlluminant(...)`
path.
That same headless OI utility surface now also includes
`oi_illuminant_ss(...)` / `oiIlluminantSS(...)`,
`oi_illuminant_pattern(...)` / `oiIlluminantPattern(...)`, and
`oi_photon_noise(...)` / `oiPhotonNoise(...)`, together with
`oiGet/oiSet(..., 'illuminant'/'illuminant photons')` support for the
legacy MATLAB OI illuminant/noise helper mini-family.
That same headless OI utility surface now also includes
`oi_pad_value(...)` / `oiPadValue(...)`, `oi_pad(...)` / `oiPad(...)`,
and `oi_make_even_row_col(...)` / `oiMakeEvenRowCol(...)` for the
legacy MATLAB OI padding and even-dimension helper mini-family.
That same headless OI utility surface now also includes
`oi_pad_depth_map(...)` / `oiPadDepthMap(...)`,
`oi_depth_segment_map(...)` / `oiDepthSegmentMap(...)`,
`oi_depth_combine(...)` / `oiDepthCombine(...)`, and
`oi_combine_depths(...)` / `oiCombineDepths(...)` for the legacy MATLAB
OI depth-map padding, segmentation, and plane-combination helper
mini-family.
That same headless OI depth surface now also includes
`oi_depth_edges(...)` / `oiDepthEdges(...)`,
`s3d_render_depth_defocus(...)` / `s3dRenderDepthDefocus(...)`, and
`oi_depth_compute(...)` / `oiDepthCompute(...)`, covering the legacy
MATLAB depth-edge selection, depth-slab defocus rendering, and per-depth
OI stack helpers.
That same headless OI utility surface now also includes
`oi_calculate_otf(...)` / `oiCalculateOTF(...)`,
`oi_custom_compute(...)` / `oiCustomCompute(...)`,
`oi_preview_video(...)` / `oiPreviewVideo(...)`, and
`oi_wb_compute(...)` / `oiWBCompute(...)`, covering the legacy MATLAB
OTF export, custom-compute detection, preview-animation rendering, and
waveband-scene batch-compute helper surface.
That same headless OI utility surface now also includes
`oi_birefringent_diffuser(...)` / `oiBirefringentDiffuser(...)` and
`oi_camera_motion(...)` / `oiCameraMotion(...)`, covering the legacy
MATLAB birefringent anti-alias filter and depth-map-driven camera-motion
burst helpers.
That same headless optics-object compatibility surface now also includes
`optics_create(...)` / `opticsCreate(...)`,
`optics_get(...)` / `opticsGet(...)`, `optics_set(...)` / `opticsSet(...)`,
`optics_clear_data(...)` / `opticsClearData(...)`,
`optics_description(...)` / `opticsDescription(...)`,
`lens_list(...)` / `lensList(...)`, and `optics_to_wvf(...)` /
`optics2wvf(...)` for the legacy MATLAB optics-object helper mini-family.
That same headless optics helper surface now also includes
`optics_defocus_depth(...)` / `opticsDefocusDepth(...)`,
`optics_dl_compute(...)` / `opticsDLCompute(...)`,
`optics_si_compute(...)` / `opticsSICompute(...)`, and
`optics_plot_transmittance(...)` / `opticsPlotTransmittance(...)`
for the legacy MATLAB optics depth/compute/transmittance wrapper family.
That same headless optics helper surface now also includes
`dl_core(...)` / `dlCore(...)`, `dl_mtf(...)` / `dlMTF(...)`,
`optics_plot_defocus(...)` / `opticsPlotDefocus(...)`,
`optics_plot_off_axis(...)` / `opticsPlotOffAxis(...)`, and
`si_convert_rt_data(...)` / `siConvertRTdata(...)`, covering the
remaining diffraction OTF core/export, defocus-surface, off-axis
falloff, and single-field ray-trace-to-shift-invariant conversion
helpers from the upstream MATLAB optics family.
The audit rules now also recognize short diffraction helper names such as
`dlMTF` as covered compatibility surfaces, while obsolete redirect/index
files like `sceneDepthOverlay.m` and `scripts/metrics/Contents.m` are now
classified out of scope instead of headless API debt.
That same optics helper surface now also includes
`make_combined_otf(...)` / `makeCombinedOtf(...)`,
`make_cmatrix(...)` / `makeCmatrix(...)`, and
`retinal_image(...)` / `retinalImage(...)`, covering the reusable
chromatic-aberration OTF weighting, per-frequency calibration-matrix,
and retinal-strip replay helpers from the older `scripts/optics/chromAb`
workflow while leaving the external `ChromAb.m` launcher and plotting
stub out of scope.
The `rtOTF` groundwork is also starting to land through public helper ports
for `rt_block_center` / `rtBlockCenter`, `rt_extract_block` /
`rtExtractBlock`, `rt_insert_block` / `rtInsertBlock`, and
`rt_choose_block_size` / `rtChooseBlockSize`. An initial public
`rt_otf` / `rtOTF` wrapper is now available as the block-wise ray-trace
OTF path, together with MATLAB-style `rtBlocksPerFieldHeight` control.
The underlying filtered-block support helper is now public too as
`rt_filtered_block_support` / `rtFilteredBlockSupport`.
That same ray-trace helper wave now also includes
`rt_root_path` / `rtRootPath` and the obsolete but still script-visible
`rt_image_rotate` / `rtImageRotate`, and the migration audit now treats
the upstream `rtBlockPartition.m` and `rtImagePSFFieldHeight.m` stubs as
out of scope because they are diagnostic/nonfunctional rather than usable
headless APIs.
`rt_synthetic` / `rtSynthetic` is now available as a synthetic ray-trace
optics generator for testing and controlled local experiments.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTSynthetic.m`, comparing the synthetic geometry and relative
illumination curves, normalized 550 nm center/edge PSF slices, and the
resulting point-array OI summary outputs from the synthetic increasing-Gaussian
ray-trace workflow.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTGridLines.m`, comparing the staged `rtGeometry` /
`rtPrecomputePSF` / `rtPrecomputePSFApply` path against automated
ray-trace and matched diffraction-limited outputs through the returned
gridline image sizes, sampled PSF support metadata, and normalized 550 nm
center-row image slices for the wide-field and reduced-FOV branches.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTPSF.m`, comparing the stored sampled ray-trace PSF metadata,
normalized 550 nm center/edge PSF slices from the computed `sampledRTpsf`
cache, and the rendered point-array ray-trace versus diffraction-limited
image summaries used by the upstream PSF comparison workflow.
That same ray-trace block now also has direct Octave parity on
`s_opticsRTPSFView.m`, comparing the stored sampled ray-trace PSF view
metadata, the normalized 550 nm field-height and angle-sweep PSF rows,
their stable blur-width summaries, and the canonicalized center/edge
`rtPlot(..., 'psf', ...)` row profiles used by the upstream PSF-view
workflow.
The shared MATLAB field-height lookup rule is now exposed too as
`ie_field_height_to_index` / `ieFieldHeight2Index`, and the ray-trace
block-size logic now uses that public helper directly.
The initial Zemax text-import surface is now ported too through
`rt_file_names` / `rtFileNames`, `zemax_read_header` / `zemaxReadHeader`,
`zemax_load` / `zemaxLoad`, and a limited `rt_import_data` /
`rtImportData` path for standard ISETPARAMS + DI/RI/PSF bundles. That
bundle format is now accepted directly by both `rt_import_data(...)` and
`oi_create('ray trace', ...)`, either as the `ISETPARAMS.txt` /
`ISETPARMS.TXT` file itself or as a containing directory, and the
importer now normalizes Windows-style `lensFile` / `baseLensFileName`
paths the way the upstream MATLAB path does. The parameter parser now
also accepts multiline MATLAB vector syntax,
including `...` continuations, bracketed row/column wavelength lists, and
transpose-style vectors such as `[500 600]'`, plus legacy single-line
assignments without trailing semicolons like the older Zemax macros emit
for `baseLensFileName`.
When importing into an existing optics object, `rt_import_data` now also
preserves the existing name/transmittance/compute settings while updating
the top-level ray-trace focal-length / f-number state to the imported
effective values, and it now enforces the upstream `psfSize` evenness
check from the parameter file before loading PSF data. Imported Zemax
bundles now also carry `psfSpacing` through to the ray-trace computation
metadata, so `rtcomputespacing` is populated on imported optics, and an
existing computation spacing is preserved when a bundle omits that field.
Imported ray-trace optics now also retain the requested `rtProgram`
label instead of forcing the lower-case default. Directory-based bundle
loading now also preserves real import failures for malformed Zemax
bundles instead of collapsing them into an unsupported-option error.
Raw MATLAB-style optics structs now also preserve top-level control
fields like `computeMethod`, `offaxis`, and `aberrationScale` through the
import path. When importing into an existing optics object, a distinct
existing ray-trace name is now preserved independently from the top-level
optics name, and a raw `rayTrace.name` now seeds the top-level optics
name when no explicit top-level name is present. Existing
`blocks_per_field_height` settings are now preserved across Zemax
imports too, and raw MATLAB-style ray-trace export now carries
`blocksPerFieldHeight` for round-tripping through `oi_get('optics')`.
Normalized ray-trace key detection now also recognizes
`blocks_per_field_height` directly, plus scalar normalized fields like
`f_number` and `magnification`, keeping raw/normalized struct handling
aligned for those cases too. Partial normalized nested `raytrace`
updates now also merge into the current ray-trace state instead of
dropping existing geometry / PSF tables, and the same preservation now
applies to partial MATLAB-style nested `rayTrace` updates. Raw nested
`rayTrace.fNumber` updates now also override the exported top-level
`fNumber` correctly instead of being clobbered during normalization, and
top-level `focalLength` edits now likewise override the stale exported
nested `rayTrace.effectiveFocalLength`. Whole-optics updates now also
honor normalized top-level aliases like `f_number`, `focal_length_m`,
and `nominal_focal_length_m` in that same export-and-edit workflow, and
raw exported `nominalFocalLength` and `offaxis` now round-trip too.
Whole-optics edits that change only `transmittance.wave` now preserve the
existing scale by interpolation instead of resetting it, and explicit
whole-optics `transmittance.scale` edits now reject wrong-length vectors
the same way the dedicated setter does.
An initial optional `SessionContext` compatibility layer is now in place
too, including session-aware core create/compute entry points and
MATLAB-style helpers for registering and retrieving the current
scene/OI/sensor/IP/camera objects without changing the explicit-object
default workflow. Session-aware `camera_set(...)` and `ip_set(...)`
replacement flows now keep that registry coherent when camera, sensor,
OI, IP, or display objects are swapped out after creation, and the
session layer now also exposes MATLAB-style generic replacement and
listing helpers such as `vcReplaceObject`, `vcReplaceAndSelectObject`,
`vcGetObjects`, `vcGetObjectNames`, and `vcCountObjects`, plus
MATLAB-style pair-return wrappers for `vcGetSelectedObject` /
`vcGetObject`, delete helpers that renumber session slots the MATLAB
way, bulk object-list replacement through `vcSetObjects`, slot-allocation
queries through `vcNewObjectValue`, add-and-select helpers through
`vcAddAndSelectObject`, name generation through `vcNewObjectName`,
batch deletion through `vcDeleteSomeObjects`, nonpositive selected-slot
clearing through `vcSetSelectedObject`, and type reporting through
`vcGetObjectType`. The session layer now also exposes the matching
`ieAddObject` / `ieDeleteObject` / `ieGetObject` /
`ieGetSelectedObject` / `ieReplaceObject` / `ieSelectObject`
wrappers, including nested `optics`, `pixel`, and `ipdisplay`
getter semantics, plus a minimal `ieInitSession` /
`ieSessionGet` / `ieSessionSet` port for session metadata,
preferences, GUI slots, selected-object queries, and
image-size/GPU session flags. Window/figure aliases and
`ieAppGet` are now supported too for the scene/OI/sensor/IP/display
app lookup path, together with the concrete GUI-side aliases that
upstream still reads such as `scene image handle`,
`sensorimagehandle`, `oiwindowHandles`, `vcimagehandles`,
`metricshandles`, `oicomputelist`, and `sensor gamma`. The
session GUI layer now also exposes `vcGetFigure`, `vcSelectFigure`,
and a minimal `ieMainClose` cleanup wrapper over the stored session
windows. Stored window geometry/state compatibility is now exposed too
through `ieWindowsGet` / `ieWindowsSet`, and the adjacent lightweight
GUI/session wrappers `vcEquivalentObjtype`, `vcSetFigureHandles`, and
`ieRefreshWindow` are now available on top of that session state.
Missing session windows now fall back to lightweight headless
placeholders so MATLAB-style `vcSelectFigure(...)` and
`ieRefreshWindow(...)` calls do not fail just because no GUI app has been
created.
The same session compatibility surface now also exposes headless
`iset(...)` / `ISET(...)`, `iset_path(...)` / `isetPath(...)`, and
`iset_root_path(...)` / `isetRootPath(...)`, so the legacy startup and
path-bootstrap helpers resolve the repository root, enumerate recursive
non-VCS subdirectories for optional `sys.path` injection, and initialize a
headless main-window session placeholder without relying on MATLAB GUI
startup.
Name-based lookup is now covered too through `ieEquivalentObjtype` and
`ieFindObjectByName`.
The shared MATLAB-style programming helper `ieParameterOtype` is now
ported too for object-type inference from parameter strings.
Headless `.mat` object persistence is now available through
`vcSaveObject`, `vcExportObject`, and `vcLoadObject` when explicit file
paths are provided.
`iePTable` is now ported in a headless form too, returning structured
parameter-table rows for the core object types without depending on a GUI
table widget.
The adjacent metadata-table helper `ieTableGet` is now ported too, covering
MATLAB-style row/file filtering over headless table-like Python mappings or
row records.
`vcGetROIData` is now ported in a headless form too for scene/OI/sensor/IP
objects, including MATLAB-style rect-to-location conversion, clipped ROI
bounds, XW-style row output, and NaN-filled sensor mosaic planes.
The adjacent headless ROI conversion helpers are now public too:
`ieRect2Locs`, `vcRect2Locs`, `ieRoi2Locs`, `ieLocs2Rect`, and
`ieRect2Vertices`.
The ROI-driven getter surface is now partially ported too: sensor ROI
state plus `sensorGet(..., 'roi volts/electrons/dv')`,
`sensorGet(..., 'roi volts mean')`, `sensorGet(..., 'roi electrons mean')`,
and `ipGet(..., 'roidata'/'roixyz')`, backed by headless
`imageDataXYZ`.
Scene and OI ROI getters are now covered too through
`sceneGet(..., 'roi photons/energy/luminance/reflectance')`,
`sceneGet(..., 'roi mean photons/energy/reflectance')`, and
`oiGet(..., 'roi photons/energy')` plus their ROI-mean variants.
The adjacent sensor compatibility surface now covers direct
`sensorGet(..., 'electrons')` as well as MATLAB-style line-profile
queries through `sensorGet(..., 'hline volts/electrons/dv')` and
`sensorGet(..., 'vline volts/electrons/dv')`, returning per-filter `data`,
`pos`, and `pixPos` vectors.
Sensor ROI/plot compatibility now also includes
`sensorGet(..., 'chromaticity')` and
`sensorGet(..., 'roi chromaticity mean')`, backed by the existing
headless demosaic path.
The adjacent image-processing demosaic surface now also supports
MATLAB-style `ieNearestNeighbor`, `Laplacian`, and
`AdaptiveLaplacian` routing through
`ipSet(..., 'demosaic method', ...)` for RGB Bayer sensors, with the
same bilinear fallback that upstream `Demosaic.m` uses for the
unsupported `gbrg` adaptive-Laplacian branch.
That same headless image-processing surface now also exposes
`demosaic(...)` / `Demosaic(...)`, matching the tutorial-visible
`Demosaic(ip, sensor)` entry point and preferring the current
`ip.data['input']` mosaic when it is present.
That same image-processing surface now also exposes
`faulty_list(...)` / `faultyList(...)`,
`faulty_insert(...)` / `faultyInsert(...)`,
`faulty_nearest_neighbor(...)` / `FaultyNearestNeighbor(...)`, and
`faulty_bilinear(...)` / `FaultyBilinear(...)`, matching the legacy
MATLAB faulty-pixel list, insertion, and GRBG repair helpers.
That same image-processing surface now also exposes
`demosaic_multichannel(...)` / `demosaicMultichannel(...)` and
`pocs(...)` / `Pocs(...)`, matching the remaining MATLAB multichannel
CFA-plane interpolation and sample-preserving POCS demosaic helper
surface on top of the existing Bayer/CFA utilities.
That same image-processing surface now also exposes
`lf_default_val(...)` / `LFDefaultVal(...)`,
`lf_default_field(...)` / `LFDefaultField(...)`,
`lf_convert_to_float(...)` / `LFConvertToFloat(...)`,
`lf_buffer_to_image(...)` / `LFbuffer2image(...)`,
`lf_image_to_buffer(...)` / `LFImage2buffer(...)`,
`lf_buffer_to_sub_aperture_views(...)` / `LFbuffer2SubApertureViews(...)`,
`lf_toolbox_version(...)` / `LFToolboxVersion(...)`, and
`ip_to_lightfield(...)` / `ip2lightfield(...)` for the legacy
Light Field Toolbox utility and IP-export helper family.
That same image-processing surface now also exposes
`lf_filt_shift_sum(...)` / `LFFiltShiftSum(...)`,
`lf_autofocus(...)` / `LFAutofocus(...)`, and
`demosaic_rccc(...)` / `demosaicRCCC(...)`, covering the remaining
headless Light Field Toolbox planar-focus/autofocus helpers plus the
RCCC monochrome demosaic path while treating upstream `lcc1.m`,
`shtlin.m`, and `shtlog.m` as incomplete legacy algorithms outside the
supported migration target.
That same image-processing surface now also exposes
`image_sensor_conversion(...)` / `imageSensorConversion(...)` and
`image_sensor_correction(...)` / `imageSensorCorrection(...)`, matching
the direct MATLAB sensor-to-ICS entry points on top of the existing IP
sensor-conversion path.
That same image-processing surface now also exposes
`image_illuminant_correction(...)` / `imageIlluminantCorrection(...)`,
matching the direct MATLAB ICS white-balance entry point on top of the
existing IP illuminant-correction path.
That same image-processing surface now also exposes
`image_rgb_to_xyz(...)` / `imageRGB2XYZ(...)`, matching the direct MATLAB
display-linear RGB-to-XYZ entry point used by Macbeth and display-plot
helpers while preserving RGB or XW data format.
That same image-processing surface now also exposes
`display_render(...)` / `displayRender(...)`, matching the direct MATLAB
ICS-to-linear-display entry point on top of the existing IP display-render
stage.
That same image-processing surface now also exposes
`image_sensor_transform(...)` / `imageSensorTransform(...)` and
`image_esser_transform(...)` / `imageEsserTransform(...)`, matching the
direct MATLAB sensor-to-target linear transform calculators for the
multisurface and Esser-optimized training sets.
That same image-processing surface now also exposes
`ie_internal_to_display(...)` / `ieInternal2Display(...)`, matching the
direct MATLAB internal-color-space to linear-display transform helper used
under `displayRender(...)` and related IP workflows.
That same image-processing surface now also exposes
`ip_hdr_white(...)` / `ipHDRWhite(...)`, matching the direct MATLAB HDR
highlight-whitening helper on top of the current IP `input` and `result`
payloads.
That same image-processing surface now also exposes
`image_distort(...)` / `imageDistort(...)`, matching the direct MATLAB
image-distortion gateway for Gaussian noise, JPEG compression, and simple
contrast scaling.
That same image-processing surface now also exposes
`ip_clear_data(...)` / `ipClearData(...)`, matching the direct MATLAB
computed-payload reset path and the `ipSet(..., 'data', [])` behavior.
That same image-processing surface now also exposes
`vcimage_clear_data(...)` / `vcimageClearData(...)`, matching the legacy
MATLAB alias for the same computed-payload reset path.
That same image-processing surface now also exposes
`image_color_balance(...)` / `imageColorBalance(...)`, matching the
deprecated MATLAB alias for the same illuminant-correction stage.
That same image-processing surface now also exposes
`vcimage_srgb(...)` / `vcimageSRGB(...)`, matching the legacy MATLAB
scene-to-sRGB convenience pipeline and accepting MATLAB-style
`colorBalanceMethod` / `colorconversionmethod` IP parameter aliases.
That same image-processing surface now also exposes
`vcimage_iso_mtf(...)` / `vcimageISOMTF(...)`, matching the legacy MATLAB
slanted-edge ISO 12233 convenience wrapper on top of the current
`camera_mtf(...)` workflow.
That same image-processing surface now also exposes
`vcimage_vsnr(...)` / `vcimageVSNR(...)`, matching the legacy MATLAB
ROI-based visual-SNR convenience wrapper on top of the current headless
VSNR ROI computation.
That same image-processing surface now also exposes
`ip_mcc_xyz(...)` / `ipMCCXYZ(...)` and the legacy alias
`vcimage_mcc_xyz(...)` / `vcimageMCCXYZ(...)`, matching the Macbeth chart
patch-XYZ helper for both `sRGB` and display-model (`custom`) conversion
paths.
Scene/OI/IP ROI plotting helpers now expose the matching chromaticity and
summary getters too: `sceneGet(..., 'chromaticity'/'roi chromaticity mean')`,
`sceneGet(..., 'roi mean luminance')`, `oiGet(..., 'chromaticity'/'roi chromaticity mean')`,
`oiGet(..., 'roi illuminance'/'roi mean illuminance')`, and
`ipGet(..., 'chromaticity'/'roi chromaticity mean')`.
The adjacent headless line-profile surface is now covered too:
`sceneGet(..., 'radiance hline/vline')`, `sceneGet(..., 'luminance hline/vline')`,
`sceneGet(..., 'spatial support linear')`, `oiGet(..., 'irradiance hline/vline')`,
`oiGet(..., 'irradiance energy hline/vline')`, and
`oiGet(..., 'illuminance hline/vline')`.
The remaining scene illuminant getter surface around that plotting path is
now exposed too: `sceneGet(..., 'illuminant comment')`,
`sceneGet(..., 'roi illuminant photons/energy')`,
`sceneGet(..., 'roi mean illuminant photons/energy')`, and
`sceneGet(..., 'illuminant hline/vline photons/energy')`.
Minimal headless `plotScene` / `oiPlot` wrappers now sit on top of those
getters too. The supported subset returns MATLAB-style `uData` payloads
and `None` instead of opening figures, which makes the plot APIs usable in
tests and scripts.
That same headless wrapper layer now also includes `plotSensor` for
volts/electrons/dv line plots and ROI histograms, and `ipPlot` for RGB
line data, chromaticity, ROI RGB summaries (`rgbhistogram` / `rgb3d`),
luminance summaries, and ROI `cielab` / `cieluv` payloads. The shared
metrics layer now exposes `xyz_to_luv` / `xyz2luv` alongside the existing
XYZ, LAB, and chromaticity helpers, and the sensor layer now exposes
`pixel_snr` / `pixelSNR` plus `sensor_snr` / `sensorSNR` with matching
headless `plotSensor('pixel snr'/'sensor snr')` payloads. The current
headless `plotSensor` subset also includes spectral wrappers for
`'color filters'` and `'sensor spectral qe'` on top of the stored sensor
spectral bundle, plus headless CFA views for `'cfa'` / `'cfa block'` and
`'cfa full'` with MATLAB-style unit-block/full-array tiling and default
display scaling. The adjacent headless sensor image-view slice is now in
place too through `plotSensor('cfa image')`, `plotSensor('true size')`,
and `plotSensor('channels')`, returning image and per-channel payloads
with MATLAB-style sensor-name metadata instead of opening GUI windows.
The explicit sensor render surface now also includes `sensorGet/Set(...,
'gamma'/'scale max')`, shared `sensorGet('rgb', ...)` rendering with
upstream-style `max output` / `max digital value` scaling semantics, and
headless true-size / CFA-image payloads report the gamma / scale-max
settings they use.
The adjacent raw CFA compatibility surface now also includes
`sensorGet/Set('cfa')`, `sensorGet('cfa pattern')`, `sensorGet('cfa name')`,
and `sensorGet('unit block config')` with MATLAB-style unit-block metadata.
The neighboring spectral metadata surface now also includes
`sensorGet/Set('spectrum')`, wavelength aliases, `sensorGet('bin width')`,
and MATLAB-style wave updates that keep stored filter spectra, pixel QE,
and IR-filter samples interpolated onto the new wavelength grid.
The adjacent raw chart/metadata block is now covered too through
`sensorGet/Set('chart parameters')`, `sensorGet/Set('chart corner points')`,
`sensorGet/Set('chart rectangles')`, `sensorGet/Set('current rect')`, and
the nearby `metadata sensor/scene/optics name` plus `metadata crop`
round-trip fields, along with the adjacent sensor movement surface through
`sensorGet/Set('sensor movement'/'eye movement')`,
`sensorGet/Set('movement positions'/'sensor positions')`, and
`sensorGet/Set('frames per position'/'exposure times per position')`. The
neighboring legacy human-cone storage surface now round-trips too through
`sensorGet/Set('human')`, `sensorGet/Set('cone type')`,
`sensorGet/Set('densities')`, `sensorGet/Set('xy')`, and the
`human cone seed` / `rseed` aliases. The
neighboring legacy MCC aliases now round-trip too through
`sensorGet/Set('mcc corner points')`, which reuses the chart corner-point
field, plus raw `sensorGet/Set('mcc rect handles')` storage. The
neighboring noise-summary aliases now also include
`sensorGet/Set('black level')`, mapped onto the same stored zero-level
state used by `zero level` / `zero`. The neighboring sensor data aliases
now also include `sensorGet/Set('digital value')` and
`sensorGet/Set('digital values')`, routed through the existing `dv`
storage path, and `sensorGet('dv or volts'/'digital or volts')` now uses
the same preferred-DV fallback behavior as upstream. The adjacent
headless sensor data-view surface now also includes
`sensorGet('volt images')`, returning per-filter plane stacks with
`NaN`-filled empty locations like MATLAB `plane2rgb(...)`. The same
adjacent voltage/response block now also includes the upstream short-form
aliases `sensorGet/Set('voltage')`, `sensorGet('electron')`, and
`sensorGet/Set('ag'/'ao')` for voltage, electron, analog-gain, and
analog-offset access, and the same `volts` / `dv` / `electrons` getters
now accept a channel index to return MATLAB-style per-filter vectors. The
same adjacent sensor-response geometry surface now also includes
`sensorGet('pixel pd area')` and `sensorGet('electrons per area', unit, channel)`
with MATLAB-style area-unit scaling and optional channel extraction. The
same neighboring quantization block now also includes `sensorGet/Set('bits')`,
`sensorGet/Set('qMethod')`, `sensorGet/Set('lut')`, and
`sensorGet/Set('quantization structure')` while preserving the existing
string-returning `sensorGet('quantization')` path used locally. The same
adjacent raw color/filter block now also includes `sensorGet/Set('color')`
plus the MATLAB cell-style aliases `filter names cell array`,
`filter color names cell array`, and `filter names cell`. The same
sensor-facing pixel passthrough now also routes supported pixel electrical
and geometric names such as `fill factor`, `pd area`, `conversion gain`,
`well capacity`, `dark current`, and `read noise volts` through the
stored pixel model, including whole-`pixel` replacement via
`sensorSet('pixel', ...)`. The same passthrough now also covers pixel spatial
geometry such as `pixel width` / `pixel height`, `width gap` / `height gap`,
`pd width` / `pd height`, `pd size`, `pd dimension`, and `pixel area`.
The neighboring pixel metadata surface now also covers `pixel name`,
`pixel type`, `pixel layer thicknesses`, `pixel refractive indices`,
`pixel spectrum` / `pixel wavelength`, `pixel bin width` / `pixel nwave`,
and MATLAB-style `pixel dr`.
The same pixel passthrough now also covers `pd xpos`, `pd ypos`, and
`pd position`, and that metadata now feeds the internal photodetector array
placement instead of being stored as a dead field.
The neighboring pixel electrical alias surface now also matches MATLAB for
`volts per electron`, `saturation voltage` / `max voltage`,
`dark current per pixel`, `dark voltage per pixel`, `volts per second`,
and the `read noise` family, including the important setter rule that
`sensorSet(..., 'read noise', value)` is interpreted in electrons rather
than volts.
The adjacent MATLAB pair-setter surface now also matches upstream ordering
for `pixel width and height` and `pixel pd width and height`, so those
aliases accept `[width, height]` inputs without changing the existing local
`pixel size` / `pd size` semantics.
The neighboring long-form pixel alias surface now also covers
`photodetector x position` / `photodetector y position` and the remaining
QE aliases such as `pixel quantum efficiency`,
`photodetector quantum efficiency`, and
`photodetector spectral quantum efficiency`.
The same pixel geometry alias layer now also covers the MATLAB gap-spacing
names `width between pixels` and `height between pixels`.
The direct unprefixed sensor-routing path now also recognizes the remaining
unique pixel setters such as `width and height`, `pd width and height`,
`size same fill factor`, and `dark voltage per pixel per sec`, so those
MATLAB-style calls no longer require an explicit `pixel` prefix.
The same adjacent unit/electrical alias surface now also covers
`pixel width meters`, `pixel height meters`, `pixel depth meters`, and the
direct `conversion gain v per electron` form.
The neighboring normalized electrical shorthand surface now also supports
direct sensor-level `conversiongainvpelectron`, `vswing`, `darkvolt`, and
`darkvolts` access without requiring the explicit `pixel` prefix.
The adjacent QE alias surface now also supports direct sensor-level
`quantum efficiency` access without requiring the explicit `pixel` prefix.
The neighboring FPN setter surface now also supports the upstream
`offset noise value` and `gain noise value` aliases.
The same FPN block now also has parser-level support for the broader
MATLAB alias family around DSNU/PRNU and column FPN names such as
`sigma offset fpn`, `gain fpn`, `column fixed pattern noise`, and
`col gain`.
That same FPN parser surface now also recognizes the setter-only
`column fpn parameters` alias.
The neighboring sensor metadata block now also has parser-level support
for `scene_name`, `metadata scene name`, `lens`, `metadata lensname`,
`metadata optics name`, `metadata sensor name`, and `metadata crop`.
The adjacent microlens block now also has parser-level support for
`ml`, `mlens`, `ulens`, `microlens offset`, `mloffset`, and
`microlens offset microns`.
The adjacent consistency block now also has parser-level routing for
`consistency`, `sensor consistency`, and `sensor compute`.
The neighboring exposure block now also has parser-level routing for
`integration time`, `exptimes`, `unique exptimes`, `central exposure`,
`exposure method`, `n exposures`, `exposure plane`, `correlated double sampling`,
`auto exposure`, and `automatic exposure`.
That same exposure coverage now also exercises the exact no-space aliases
`integrationtime`, `uniqueexptimes`, `centralexposure`, `exposuremethod`,
`nexposures`, `exposureplane`, `correlateddoublesampling`, `autoexp`,
`autoexposure`, and `automaticexposure`.
The neighboring sampling/noise-control block now also has parser-level routing for
`pixel samples`, `n pixel samples for computing`, `spatial samples per pixel`,
and `response type`.
That same sampling/noise-control coverage now also exercises the exact no-space aliases
`pixelsamples`, `npixelsamplesforcomputing`, `spatialsamplesperpixel`,
`reusenoise`, `noiseseed`, and `responsetype`.
The adjacent movement block now also has parser-level routing for
`eye movement`, `frames per position`, `exposure times per position`,
and `etime per pos`.
The same movement alias surface now also has focused parser/runtime coverage for
`movement positions` plus the prefixed `sensor positions` and `sensor positions x/y` forms.
That same movement coverage now also exercises the exact no-space aliases
`eyemovement`, `movementpositions`, `sensorpositions`, `sensorpositionsx`,
`sensorpositionsy`, `framesperposition`, `exposuretimesperposition`, and `etimeperpos`.
The neighboring legacy human block now also has parser-level routing for
`human`, `cone type`, `densities`, `xy`, and `rseed`.
That same block now also routes the legacy location aliases `cone xy` and
`cone locs`.
That same legacy human surface now also has focused parser/runtime coverage for
the long-form aliases `human cone type` and `human cone locs`.
The same test coverage now also exercises `human cone densities` and
`human cone seed` directly on the runtime path.
That same runtime/parser coverage now also exercises the exact legacy alias
`humanrseed`, and the same runtime coverage now also exercises the exact
location aliases `conexy` and `conelocs`.
The same legacy human coverage now also exercises the exact no-space aliases
`humanconetype`, `humanconedensities`, and `humanconelocs`, with parser
coverage also including `humanconeseed`.
The adjacent chart block now also has parser-level routing for
`chart parameters`, `corner points`, `chart corners`, `chart rects`,
`chart rectangles`, `current rect`, and `chart current rect`.
That same chart surface now also has focused parser/runtime coverage for the
natural long-form setter alias `chart corner points`.
The same runtime coverage now also exercises the long-form setter path for
`chart rectangles` and `current rect`, plus the adjacent `mcc corner points`
setter alias.
That same chart coverage now also exercises the exact no-space aliases
`chartparameters`, `cornerpoints`, `chartcorners`, `chartrects`,
`currentrect`, and `mccrecthandles`.
The same runtime coverage now also exercises the exact long no-space chart
aliases `chartcornerpoints`, `chartrectangles`, and `chartcurrentrect`.
The neighboring metadata block now also has focused parser/runtime coverage for
`scene name`, `metadata scene name`, `metadata lens name`, `metadata lens`,
`metadata optics name`, `metadata sensor name`, and `metadata crop`.
That same runtime coverage now also exercises the exact aliases `scene_name`,
`scenename`, `metadatalensname`, `metadatalens`, `metadatasensorname`, and
`metadatacrop`.
The adjacent chart/metadata coverage now also exercises the exact no-space MCC
and metadata aliases `mcccornerpoints`, `mccrecthandles`,
`metadatascenename`, `metadataopticsname`, `metadatasensorname`, and
`metadatacrop`.
The neighboring pixel material alias surface now also supports direct
sensor-level `refractive index` and `n` access without requiring the
explicit `pixel` prefix.
The adjacent setter semantics now also match MATLAB for
`pixel size same fill factor` / `pixel size constant fill factor`, scaling the
photodetector geometry with the pixel size instead of treating that alias like
a plain pixel-size replacement.
The neighboring sensor model surface now also includes raw
`sensorGet/Set('diffusion MTF')` storage for legacy compatibility. The
neighboring legacy scene/lens aliases now round-trip too through
`sensorGet/Set('scene_name')`, `sensorGet/Set('lens')`, and
`sensorGet('metadata lensname'/'metadata lens')`. The adjacent raw
microlens storage surface now round-trips as well through
`sensorGet/Set('microlens'/'ml')` and
`sensorGet/Set('microlens offset'/'mloffset')`. The remaining scalar hooks
in that upstream block now round-trip too through
`sensorGet/Set('consistency'/'sensor consistency')` and
`sensorGet/Set('sensor compute'/'sensor compute method')`. The adjacent
exposure-control surface now includes `sensorGet('n exposures')`,
`sensorGet/Set('exposure plane')`, `sensorGet/Set('cds')`, and MATLAB-style
`auto exposure` string handling for `'on'` / `'off'`. The neighboring
exposure-summary surface now also includes `sensorGet/Set('exposure method')`,
array-capable `integration time` / `exptimes` storage, `unique exptimes`,
and `central exposure`, with `sensorCompute` explicitly rejecting multi-
exposure arrays until bracketed/CFA capture compute is ported. The nearby
sampling and vignetting aliases now also round-trip through
`sensorGet/Set('pixel samples'/'n pixel samples for computing')` and
`sensorGet/Set('sensor vignetting'/'sensor bare etendue'/'no microlens etendue')`.
The same vignetting block now also has parser-level support for
`vignetting flag`, `pixel vignetting`, and `vignetting name`.
That same vignetting coverage now also exercises the exact no-space aliases
`vignettingflag`, `pixelvignetting`, `sensorvignetting`,
`sensorbareetendue`, `nomicrolensetendue`, and `vignettingname`.
The adjacent sensor-data coverage now also exercises the exact no-space
aliases `digitalvalue`, `digitalvalues`, `dvorvolts`, `digitalorvolts`,
`voltimages`, `voltage`, `electron`, `ag`, and `ao`.
The neighboring quantization and color/filter coverage now also
exercises the exact no-space aliases `qmethod`,
`quantizationstructure`, `quantizationlut`, `filternamescellarray`,
`filtercolornamescellarray`, and `filternamescell`.
The adjacent CFA and spectral-metadata coverage now also exercises the
exact no-space aliases `cfapattern`, `cfaname`, `unitblockconfig`,
`sensorspectrum`, `wavelengthresolution`, `binwidth`, and
`numberofwavelengthsamples`.
The neighboring filter/response metadata coverage now also exercises the
exact no-space aliases `nfilters`, `filtercolorletters`,
`filtercolorletterscell`, `filterplotcolors`, `patterncolors`,
`spectralqe`, and `sensorspectralsr`.
The adjacent spectral-filter and pixel-response coverage now also
exercises the exact no-space aliases `filterspectra`, `colorfilters`,
`infraredfilter`, `irfilter`, `pixelspectralqe`, `pdspectralqe`,
`pixelqe`, `spectralsr`, `pdspectralsr`, `pixelspectralsr`, and `sr`.
The neighboring geometry/block metadata coverage now also exercises the
exact no-space aliases `unitblockrows`, `unitblockcols`, and `cfasize`,
while the runtime surface now also uses the exact sensor getter form
`spatialsupport`.
The adjacent control/summary coverage now also exercises the exact
no-space aliases `autoexposure`, `integrationtime`,
`integrationtimes`, `nsamplesperpixel`, `sensorconsistency`,
`sensorcompute`, `sensorcomputemethod`, `dynamicrange`,
`sensordynamicrange`, `maxdigitalvalue`, `analoggain`, and
`analogoffset`.
The neighboring pixel passthrough coverage now also exercises the exact
no-space aliases `fillfactor`, `pixelsize`, `pixelwidth`,
`pixelheight`, `pixelwidthmeters`, `pixelheightmeters`, `widthgap`,
`heightgap`, `widthbetweenpixels`, `heightbetweenpixels`,
`xyspacing`, `pdarea`, `voltsperelectron`, `wellcapacity`,
`maxvoltage`, `darkvoltageperpixel`, `darkelectrons`,
`darkcurrentdensity`, `readnoisevolts`, `readnoiseelectrons`,
`readstandarddeviationvolts`, `readnoisemillivolts`,
`pixelspectrum`, `pixelwavelength`, `pixelbinwidth`, `pixelnwave`,
`layerthicknesses`, `refractiveindices`, `stackheight`,
`pixeldepthmeters`, `pdposition`, `pdxpos`, `pdypos`,
`photodetectorwidth`, `photodetectorheight`,
`photodetectorsize`, `photodetectorxposition`,
`photodetectoryposition`, `pdwidthandheight`,
`pixeldynamicrange`, `darkvoltageperpixelpersec`,
`voltspersecond`, `readnoisestdvolts`, and
`readstandarddeviationelectrons`, `widthandheight`,
`widthheight`, `sizeconstantfillfactor`,
`sizekeepfillfactor`, `sizesamefillfactor`, and
`pddimension`. The neighboring direct sensor geometry/data runtime
coverage now also exercises `rows`, `cols`, `size`, `dimension`,
`arraywidth`, `arrayheight`, `wspatialresolution`,
`hspatialresolution`, `deltax`, and `deltay`; parser coverage for this
block remains on the unambiguous prefixed forms like `sensor rows`
because the bare names intentionally stay object-generic in
`ie_parameter_otype(...)`. The adjacent direct sensor data-access
coverage now also explicitly exercises `volts`, `voltage`, `dv`,
`digitalvalue`, `digitalvalues`, `electrons`, `electron`,
`dvorvolts`, `digitalorvolts`, `voltimages`, and channel-select access
through the exact no-space aliases. The adjacent ROI/response block now
also explicitly exercises `roivolts`, `roielectrons`, `roidv`,
`roivoltsmean`, `roielectronsmean`, `responseratio`, and
`volts2maxratio`. That same ROI/response block now also exercises the
exact ROI-data synonym family `roidata`, `roidatav`, `roidatavolts`,
`roidatae`, `roidataelectrons`, and `roidigitalcount`, plus the exact
line-profile aliases `hlinevolts`, `hlineelectrons`, `hlinedv`,
`vlinevolts`, `vlineelectrons`, and `vlinedv`. That same sensor ROI and
line-profile block now also exercises the exact getter form
`roichromaticitymean`, with prefixed parser coverage for
`sensor chromaticity` and `sensor roichromaticitymean`. The adjacent ROI
geometry block now also exercises the exact forms `roi`, `roilocs`, and
`roirect`. The neighboring microlens/vignetting/etendue block now also
exercises the exact forms `microlens`, `microlensoffset`,
`microlensoffsetmicrons`, `vignetting`, `ngridsamples`, and
`sensoretendue`. The adjacent quantization block now also exercises the
exact forms `quantization`, `quantizationmethod`, `nbits`,
`quantizatonlut`, `maxdigital`, and `maxoutput`. The neighboring
CFA/filter metadata block now also exercises the exact forms `pattern`,
`filternames`, and `diffusionmtf`. The adjacent FPN/noise-control block
now also exercises the exact forms `dsnusigma`, `offsetnoisevalue`,
`prnusigma`, `gainnoisevalue`, `fpnparameters`,
`columnfixedpatternnoise`, `coloffsetfpnvector`,
`colgainfpnvector`, and `noiseflag`, and the same neighboring FPN-image
surface now also supports exact `dsnuimage`, `offsetfpnimage`,
`prnuimage`, and `gainfpnimage` access, including clearing those stored
images with `None`. The neighboring exposure-summary
block now also exercises the exact forms `exptime`, `exposuretimes`,
`exposuretime`, `expduration`, `exposureduration`,
`exposuredurations`, `uniqueintegrationtimes`, `uniqueexptime`,
`uniqueexptimes`, `geometricmeanexposuretime`, `expmethod`, and `cds`.
The adjacent movement and legacy-human block now also exercises the
exact forms `sensormovement`, `framesperpositions`, `conetype`,
`conexy`, and `conelocs`.
The adjacent noise-control storage surface now also includes
`sensorGet/Set('reuse noise')`, `sensorGet/Set('noise seed')`, and
`sensorGet/Set('response type')`, with stored noise seeds now feeding
`sensorCompute(...)` whenever no explicit seed argument is passed. The
neighboring column fixed-pattern-noise storage surface now also
round-trips through `sensorGet/Set('column fpn')`,
`sensorGet('column dsnu'/'column prnu')`, and
`sensorGet/Set('col offset fpn vector'/'col gain fpn vector')`. The same
adjacent response/noise summary block now also includes
`sensorGet('response dr')`, `sensorGet('dr'/'dynamic range'/'sensor dynamic range')`,
and the upstream `shot noise flag` getter alias.
That same noise-summary coverage now also exercises the exact no-space aliases
`columnfpnparameters`, `columnfpn`, `columnfpnoffset`, `columnfpngain`,
`coloffsetfpn`, `coloffset`, `colgainfpn`, `colgain`, `responsedr`,
`drdb20`, `shotnoiseflag`, `blacklevel`, and `zerolevel`.
The same headless plotting surface now
also covers `plotSensor('etendue')`, returning MATLAB-style support and
relative-illumination payloads without opening a mesh plot window. Sensor
spectral plotting now also includes `plotSensor('ir filter')`,
`plotSensor('pixel spectral qe')`, `plotSensor('pixel spectral sr')`, and
`plotSensor('sensor spectral sr')`, backed by explicit `ir filter` and
`pixel spectral qe` state on the sensor model plus combined sensor
responsivity derived from the stored QE/filter bundle. The same headless
sensor plotting layer now also includes deterministic noise-map payloads
for `plotSensor('shot noise')`, `plotSensor('dsnu')`, and
`plotSensor('prnu')`. A matching headless `plotSensorFFT(...)` wrapper is
now available too for monochrome horizontal/vertical line FFT payloads,
and `plotSensor('chromaticity')` now returns MATLAB-style `rg` and
spectrum-locus payloads for sensor ROI plots without opening a figure.
Line-profile plots also now support MATLAB-style adjacent-line aggregation
through `plotSensor(..., 'two lines', True)` on the supported sensor
horizontal/vertical line cases, and headless `plotSensor(...)` now also
accepts MATLAB-style `capture` selection for multi-capture sensor stacks.
The adjacent line/FFT metadata is tighter too: single-line sensor plots now
return plot labels and filter-color metadata, and `plotSensorFFT(...)` now
accepts the same capture selection path for multi-capture stacks.
The remaining histogram/chromaticity wrappers now return the matching
headless plot labels too, including histogram axis labels/filter colors and
the MATLAB-style `rg` chromaticity axis/title strings.
The spectral and etendue wrappers now return the same core plot metadata as
well, including spectral wavelength-axis labels plus MATLAB-style window
names and the etendue axis labels/name string.
The headless pixel/sensor SNR wrappers now expose the matching plot labels
too, including signal/SNR axis labels, title strings, and legend entries
for the returned component curves.
Multispectral / EXR / reflective-display cases and the rest of the vendor
sensor catalog remain explicitly out of scope until they are ported
deliberately.
The script-driven sensor slice now also covers the headless
`s_sensorMCC.m` workflow: `sensorCreate('bayer (gbrg)')`, headless
`sensorCCM(...)` chart fitting from stored corner points, and the fixed
`ipSet(..., 'conversion method sensor', 'current matrix')` path now all
run end to end with an Octave-backed parity case on the stable CCM and
corrected-render summaries. The adjacent `s_sensorRollingShutter.m`
workflow is now also covered headlessly through `sceneRotate(...)`,
`oiCrop(...)`, and a deterministic rolling-shutter assembly parity case
that compares the temporal sensor trace plus the final assembled render.
The ray-trace helper audit now counts the already-landed data/math wrapper
layer individually instead of holding the entire family at `partial`:
`rtAngleLUT` through `rtSynthetic` are recognized as direct Python
compatibility surfaces, the new headless `rtPSFEdit(...)` port covers the
legacy PSF centering/rotation helper, and the remaining `rtPlot` /
`rtPSFVisualize` movie-window routines are explicitly treated as GUI-only
out-of-scope files.
The small color/metrics cleanup layer now also exposes legacy
`ieColorTransform(...)` on top of the existing sensor-transform math, and
the direct alias wrappers `ieLuminanceFromEnergy(...)`,
`ieLuminanceFromPhotons(...)`, and `iePSNR(...)` are counted as covered
compatibility surfaces. Plot-only stubs like `cameraPlot.m` and
`psfPlotrange.m` are now classified as GUI-only out of scope.
The remaining `tutorials/color` audit tail is closed too: the
chromaticity and energy/quanta walkthroughs are now tracked as covered
workflow evidence through focused headless regressions, while
`t_colorSpectrum.m` remains backed by curated parity.
The `tutorials/code` tail is closed as well: the deprecated rendering
walkthrough is now backed by a focused headless regression, the
`vcSESSION` object-database tutorials are counted against the existing
session compatibility layer, and the old `startup.m` path-management note
is explicitly out of scope.
The `tutorials/camera` family is now closed too: the camera introduction,
noise, anti-alias, and full system walkthroughs are tracked through
focused headless camera regressions plus a small diffuser fix that now
replays blur and birefringent anti-alias filters through non-raytrace
`oi_compute(...)`.
The `tutorials/printing` tail is now closed too: the halftoning walkthrough
is backed by direct `HalfToneImage(...)` and `FloydSteinberg(...)`
compatibility wrappers plus a focused printing tutorial regression over
clustered-dot, Bayer, and error-diffusion output.
The tiny `scripts/data` / `scripts/faces` tail is now cleaned up too: the
scratch-data bootstrap script is covered by a focused headless
`scene -> oi -> sensor -> ip` initialization regression, while the face
detection demo is explicitly out of scope because the vendored upstream file
immediately returns and depends on MATLAB Vision Toolbox UI behavior.
The small `utility/file` helper tail is cleaner too: headless wrappers now
cover `ieImageType(...)`, `ieSaveSpectralFile(...)`, `ieTempfile(...)`,
`ieVarInFile(...)`, `pathToLinux(...)`, `vcImportObject(...)`,
`vcReadSpectra(...)`, and `vcSaveMultiSpectralImage(...)`, while the pure
MATLAB file-picker helpers `ieReadMultipleFileNames.m`,
`vcSelectDataFile.m`, and `vcSelectImage.m` are explicitly tracked as GUI
out of scope rather than residual headless debt.
That same compatibility layer now also covers `ieWebGet(...)`,
`ieXL2ColorFilter(...)`, and `vcReadImage(...)`: `ieWebGet(...)` stays
headless by returning list or browse metadata plus local download or unzip
results without opening a browser, the spreadsheet importer now supports CSV
plus simple XLSX color-filter and spectral payloads, and the image reader
replays legacy RGB and basis-coded multispectral contracts on top of
`sceneFromFile(...)`.
That `utility/file` family is now fully closed: direct wrappers also cover
`ieDNGRead(...)`, `ieDNGSimpleInfo(...)`, `ieReadColorFilter(...)`,
`ieReadSpectra(...)`, `ieSCP(...)`, `ieSaveColorFilter(...)`,
`ieSaveMultiSpectralImage(...)`, `ieSaveSIDataFile(...)`, `vcExportObject(...)`,
`vcLoadObject(...)`, and `vcSaveObject(...)`.
The neighboring `utility/plots` gateway slice is cleaner too:
`plotDisplaySPD(...)`, `plotDisplayLine(...)`, `plotDisplayColor(...)`,
`plotDisplayGamut(...)`, and `plotSensorHist(...)` now have direct headless
wrapper coverage in `pyisetcam.plotting`, and the legacy alias entry points
`plotOI(...)`, `scenePlot(...)`, and `sensorPlot(...)` are exported directly
on top of the existing headless plotting gateways.
That same plotting family now also covers the small axis-annotation helpers
`identityLine(...)`, `xaxisLine(...)`, `yaxisLine(...)`, and
`plotTextString(...)` through direct headless wrappers that return the line
geometry, text placement, and styling payloads instead of MATLAB figure
handles.
The next standalone plotting helpers are covered too:
`plotGaussianSpectrum(...)`, `plotSpectrumLocus(...)`,
`plotContrastHistogram(...)`, and `plotEtendueRatio(...)` now return the
same core spectral, chromaticity, histogram, and etendue payloads
headlessly instead of opening MATLAB graph windows.
The adjacent standalone plotting helpers are cleaner too:
`iePlaneFromVectors(...)`, `iePlotJitter(...)`, `plotNormal(...)`,
`plotRadiance(...)`, and `plotReflectance(...)` now return the same plane,
jitter, PDF, and wavelength-line payloads headlessly instead of opening
MATLAB figure windows.
That same sensor plotting family now also exposes the direct MATLAB entry
points `plotPixelSNR(...)`, `plotSensorEtendue(...)`, and
`plotSensorSNR(...)` on top of the existing `plotSensor(...)` payloads.
The adjacent figure-style plotting helpers are cleaner too:
`fisePlotDefaults(...)`, `ieFigureFormat(...)`, `ieFigureResize(...)`,
`iePlot(...)`, `iePlotSet(...)`, `iePlotShadeBackground(...)`, and
`ieShape(...)` now return the same graphics-default, figure-layout, line,
background-patch, and analytic-shape payloads headlessly instead of
opening or mutating MATLAB figure windows.
The neighboring density/helper plotting slice is smaller too:
`hist2d(...)`, `scatplot(...)`, `ieHistImage(...)`, and
`plotSetUpWindow(...)` now return the same histogram-count, scatter-density,
histogram-image, and graph-window-default payloads headlessly instead of
opening MATLAB figures.
That same `utility/plots` family now also covers the last direct plotting
tail: `plotML(...)`, `plotMetrics(...)`, and `sensorPlotColor(...)` now have
direct headless compatibility wrappers in `pyisetcam.plotting`, returning
microlens irradiance/offset meshes, metrics-ROI histogram summaries, and
demosaiced channel-scatter payloads plus blackbody reference loci. The
legacy `plotSceneTest.m` and `plotSensorTest.m` workflows are now tracked
against focused plotting regressions instead of remaining as anonymous open
audit debt.
The adjacent `utility/image` numeric helper tail is shrinking too:
`ieFindWaveIndex(...)`, `ieWave2Index(...)`, `ieRadialMatrix(...)`,
`imageBoundingBox(...)`, `imageCentroid(...)`, `imageCircular(...)`, and
`imageContrast(...)` now have direct headless compatibility wrappers in
`pyisetcam.utils` with focused regressions over MATLAB-style 1-based
indexing, radial support, centered aperture masking, centroid rounding, and
per-channel contrast normalization.
That `utility/image` family is now fully closed too: the remaining direct
surface is covered by `ie_field_height_to_index(...)` /
`ieFieldHeight2Index(...)` in `pyisetcam.optics` plus the new
`image_dead_leaves(...)` / `imageDeadLeaves(...)` and
`image_vernier(...)` / `imageVernier(...)` wrappers in `pyisetcam.scene`.
The adjacent `data/sensor` family is now cleaned up too: the vendored sensor
asset bundle is counted directly against the current `AssetStore` and
sensor-model runtime coverage, while the old MATLAB asset-authoring scripts
that generated those MAT or TIFF files are explicitly treated as provenance
notebooks outside the headless migration target.
The neighboring `data/human` family is now cleaned up the same way: the
vendored human spectral assets are counted directly against the current
`AssetStore` readers plus the XYZ, luminosity, rods, Stockman, and macular
runtime coverage, while the old MATLAB creator notebooks and plotting
artifacts are explicitly treated as provenance assets outside the headless
migration target.
The adjacent `data/lights` family is now cleaned up the same way as well:
the vendored illuminant bundle is counted directly against the current
`AssetStore`, daylight generation, illuminant creation, scene illuminant
adjustment, and Gretag or daylight regression coverage, while the MATLAB
data-import and comparison notebooks that authored those MAT files are
explicitly treated as provenance assets outside the headless migration
target.
The neighboring `data/lens` family is now cleaned up in the same style:
the vendored lens JSON bundle is counted directly against the current
`lensList(...)` and ray-trace optics import coverage, while the MATLAB
focus-table and metadata-maintenance notebooks around those lens files are
explicitly treated as provenance or analysis assets outside the headless
migration target.
The adjacent `data/optics` family is now cleaned up the same way too: the
vendored optics bundle is counted directly against the current ray-trace
loaders, flare assets, and Thibos virtual-eye coverage, while the old
`VirtualEyesDemo.m` plotting script is explicitly treated as a provenance
demo outside the headless migration target.
The tiny `data/fluorescence` bundle is now explicitly tracked as out of
scope: the pinned upstream snapshot only vendors fluorophore MAT assets,
while the required fluorescence-model helpers live outside the supported
ISETCam migration target.
The adjacent `data/safetystandards` bundle is now counted directly too:
the vendored hazard spectra are already exercised by the current
`humanUVSafety(...)` path, while the old MATLAB EN62471 transcription
notebook that produced those MAT files is explicitly treated as provenance
outside the headless migration target.
The neighboring `data/images` bundle is now counted directly too: the
vendored RGB, raw, and multispectral image assets are already exercised
through the current `sceneFromFile(...)`, raw-image import, and camera or
parity workflows, while the old spectral-face download notebook is
explicitly treated as a provenance demo outside the headless migration
target.
The adjacent `data/surfaces` bundle is now counted directly too: the
vendored reflectance and chart assets are already exercised through the
current `surfaceReflectance(...)`, `sceneCreate(...)`, Esser-chart, skin,
and curated `s_surfaceMunsell.m` parity workflows, while the old Esser and
hemoglobin authoring notebooks are explicitly treated as provenance data
generation outside the headless migration target.
The neighboring `data/scenes` family is now explicitly tracked as out of
scope: the pinned upstream snapshot does not vendor reusable scene assets
there at all, only an empty placeholder plus the old `d_sceneICVL.m`
local-path ingest notebook for the external ICVL repository.
The adjacent `data/displays` bundle is now counted directly too: the
vendored CRT, LCD, OLED, and reflectance display calibrations are already
exercised through `displayCreate(...)` plus scene or IP display-backed
workflows, `render_oled_samsung(...)` and `render_lcd_samsung_rgbw(...)`
now exist as direct headless wrappers for the Samsung dixel helpers, and
the old `ieBarcoSign.m` calibration-authoring notebook is explicitly
treated as provenance outside the headless migration target.
The tiny `data/microlens` and `data/validation` bundles are now explicitly
tracked as out of scope too: the former only vendors standalone microlens
design JSON presets that the current analytic microlens workflow does not
import, and the latter only vendors archived MATLAB validation payloads
rather than runtime assets or headless APIs.
The neighboring `fonts` and `data/fonts` pair are now covered together:
`fontBitmapGet(...)`, `fontCreate(...)`, `fontGet(...)`, `fontSet(...)`,
and `sceneFromFont(...)` now exist as direct headless wrappers, and the
vendored Georgia glyph bitmaps are exercised directly as the runtime font
cache rather than left as standalone asset debt.
That same utility-image slice now also covers `ieCmap(...)`,
`ieCropRect(...)`, `ieLUTDigital(...)`, `ieLUTInvert(...)`, and
`ieLUTLinear(...)`, with direct regressions for the simple MATLAB color-map
forms, field-of-view crop-rectangle math, and forward/inverse gamma-table
lookup behavior.
The adjacent resampling/geometry utility tail is smaller too:
`rgb2dac(...)`, `imageTranspose(...)`, `imageTranslate(...)`,
`imageInterpolate(...)`, and `imageHparams(...)` now have direct headless
compatibility wrappers in `pyisetcam.utils`, with focused regressions over
DAC lookup-table mapping, per-plane transpose, MATLAB-style left/up
translation, bilinear resize shape/corner behavior, and the default
harmonic-parameter structure.
The remaining utility-image harmonic/montage tail is smaller now too:
`imageGabor(...)`, `imageMakeMontage(...)`, and `imageMontage(...)` now
have direct headless compatibility wrappers in `pyisetcam.utils`, while
`imageShowImage(...)` is covered by a headless IP-render wrapper in
`pyisetcam.ip`; the GUIDE-only `imageSetHarmonic.m` dialog is explicitly
tracked as out of scope.
That same utility-image display/spectral slice is smaller now as well:
`convolvecirc(...)`, `imageSlantedEdge(...)`, `imageSPD(...)`,
`imageSPD2RGB(...)`, `imagehc2rgb(...)`, `imagescRGB(...)`,
`imagescOPP(...)`, and `imagescM(...)` now have direct headless
compatibility wrappers in `pyisetcam.utils`, while the local PNG report
generator `ieMontages.m` and the session/window browser `imageMultiview.m`
are explicitly treated as out of scope.
The legacy startup/path trio is now covered too: `ISET.m`, `isetPath.m`,
and `isetRootPath.m` are tracked against the optional headless session
bootstrap wrappers `iset(...)`, `isetPath(...)`, and `isetRootPath(...)`
rather than remaining as anonymous top-level audit debt.
The adjacent `main` family is closed too: `ieInitSession(...)`,
`ieMainClose(...)`, `ieSessionGet(...)`, and `ieSessionSet(...)` are now
counted directly as ported headless session wrappers, while
`ieMainW(...)`, `iePrintSessionInfo(...)`, and `mainOpen(...)` now have
explicit compatibility wrappers in `pyisetcam.session` for main-window
placeholder bootstrapping, session-summary text output, and headless
GUI-state bookkeeping.
The neighboring `human` helper slice is smaller now too:
`humanPupilSize(...)`, `watsonImpulseResponse(...)`,
`watsonRGCSpacing(...)`, `kellySpaceTime(...)`,
`poirsonSpatioChromatic(...)`, `westheimerLSF(...)`,
`humanSpaceTime(...)`, `humanAchromaticOTF(...)`, `humanCore(...)`,
`humanOTF(...)`, `humanLSF(...)`, and `ijspeert(...)` now have direct
headless compatibility wrappers across `pyisetcam.metrics` and
`pyisetcam.optics`, with focused regressions covering pupil-size models,
temporal impulse normalization, Kelly/Poirson dispatch, Westheimer line
spread, human OTF/LSF payloads, and IJspeert MTF/PSF/LSF replay.
That same `human` family now also covers the neighboring cone/macular
helpers: `humanOpticalDensity(...)`, `humanConeContrast(...)`,
`humanConeIsolating(...)`, `humanMacularTransmittance(...)`, and
`humanOTF_ibio(...)` now have direct headless compatibility wrappers, with
focused regressions over Stockman density defaults, energy/quanta cone
contrast equivalence, display cone-isolating directions, macular
transmittance updates on OIs, and the ISETBio `ifftshift` OTF storage
contract.
That same `human` family is now fully closed too: the deprecated
`humanOI(...)` retinal-irradiance path is replayed through a thin
headless wrapper over the existing OI compute stack, `humanUVSafety(...)`
now covers the actinic/eye/blue-hazard and thermal threshold calculations
against the vendored safety-standard spectra, and `ieConePlot(...)`
returns the cone-grid and blurred RGB mosaic payload headlessly instead of
opening a MATLAB figure window.
The adjacent `scripts/human` family is closed too: the Brettel
color-blind workflow is now covered by direct `xyz2lms(..., cbType,
whiteXYZ)` regression coverage, the display-point-spread script is backed
by a headless `display -> scene -> wvf OI -> cone mosaic sensor`
regression, the lamp-safety scripts are counted against the existing
`humanUVSafety(...)` / radiance-conversion coverage, and the deprecated or
ISETBio-gated human notebooks are explicitly treated as out of scope.
The adjacent `scripts/image` family is closed too: the remaining
headless IP workflows are now counted directly against the current
`ieN2MegaPixel(...)`, `imagehc2rgb(...)`, `imageSensorConversion(...)`,
`imageIlluminantCorrection(...)`, and sRGB roundtrip regression surface,
while the old standalone JPEG/DCT classroom helpers plus the GUI-heavy
exploratory image scripts are explicitly treated as out of scope.
The small `web` family is now closed too: `webData`, `webFlickr`,
`webLOC`, `webPixabay`, and `webCreateThumbnails(...)` now have direct
headless compatibility coverage for catalog search, remote image fetch,
scene replay, and local thumbnail generation, while the scratch
`dngImport.m` notebook is explicitly treated as out of scope because the
supported DNG runtime path already lives under `ieDNGRead(...)`.
The neighboring `cp` family is now closed as audit-only scope: the
remaining `cpCamera`, `cpScene`, `cpCModule`, burst-IP, and demo-script
entries are all prototype computational-photography wrappers built around
external ISET3d or PBRT recipe rendering, MATLAB video export, or toolbox
registration workflows rather than supported standalone pyisetcam runtime
surfaces.
The broad `utility` family is smaller too: the `utility/list` catalog
helpers `ieDataList(...)`, `ieLightList(...)`, and
`ieReflectanceList(...)` now have direct headless compatibility wrappers
in `pyisetcam.utils`, and the neighboring `utility/units` slice now also
covers `dpi2mperdot(...)`, `ieDpi2Mperdot(...)`, `ieN2MegaPixel(...)`,
`ieSpace2Amp(...)`, `ieUnitScaleFactor(...)`, `sample2space(...)`, and
`space2sample(...)`, including the MATLAB default-microns scaling,
truncated mean-to-Nyquist FFT support, centered sample geometry, and the
obsolete zero-based inverse-spacing contract. That same utility-units lane
now also exposes the direct package-root `unit_frequency_list(...)` helper
plus the MATLAB-style alias `unitFrequencyList(...)` instead of leaving the
helper only in `pyisetcam.utils`. The same broad utility-helper surface now
also exposes `energy_to_quanta(...)`, `quanta_to_energy(...)`, and the
MATLAB-style package-root aliases `Energy2Quanta(...)` /
`Quanta2Energy(...)` instead of leaving those conversions only in
`pyisetcam.utils`. That same package-root helper surface now also exposes
`interp_spectra(...)`, `spectral_step(...)`, `array_percentile(...)`, and
`spectral_angle(...)` instead of leaving those small spectral utilities
buried in `pyisetcam.utils` or `pyisetcam.metrics`. The same root helper
surface now also exposes `linear_to_srgb(...)`, `xyz_to_linear_srgb(...)`,
and `invert_gamma_table(...)` so the small sRGB conversion helpers are
available directly from `pyisetcam` as well. The same utility-root surface
now also exposes `resample_cube(...)`, `gaussian_sigma_pixels(...)`,
`apply_channelwise_gaussian(...)`, `least_squares_matrix(...)`, and
`tile_pattern(...)` so the small image-resampling and CFA-layout helpers are
available directly from `pyisetcam` instead of only through
`pyisetcam.utils`. The same package-root helper surface now also exposes
`xyz_color_matching(...)`, `sensor_to_target_matrix(...)`,
`sensor_to_xyz_matrix(...)`, `internal_to_display_matrix(...)`,
`sc_gaussian_parameters(...)`, and `sc_compute_difference(...)` instead of
leaving those reusable color and SCIELAB helpers only under
`pyisetcam.color` or `pyisetcam.scielab`.
The same color-metric path now also replays the upstream component modes for
`delta_e_ab(...)` and `sc_compute_difference(...)`, so `'luminance'`,
`'chrominance'`, `'hue'`, and `'all'` now work on top of the existing
CIEDE2000 component math instead of raising unsupported-option errors, and
`delta_e_ab(...)` now also accepts paired white points like the neighboring
`delta_e_uv(...)` path. The same metrics-helper lane now preserves those paired
per-image white points through the CIELAB `metricsCompute(...)` and
`metricsCompareROI(...)` paths instead of collapsing them into a single average
white. The neighboring
`metricsCamera(...)` gateway now also accepts the upstream `fullreference`
metric name on top of the existing headless `cameraFullReference(...)` path,
and the adjacent camera wrapper now also replays `cameraGet/Set('metric', ...)`
for stored or on-demand camera metrics.
The same package-root helper surface now also exposes
`example_spd_pair(...)` instead of leaving that deterministic
documentation/example SPD generator only under `pyisetcam.metrics`.
That same SCIELAB surface now also exposes the upstream MATLAB-style aliases
`scGaussianParameters(...)` and `scComputeDifference(...)` alongside the
snake-case helpers, instead of leaving only the mixed alias set around
`scApplyFilters(...)` and `scComputeSCIELAB(...)`.
That same broad `utility` family is smaller again: the adjacent
`utility/numerical` helper slice now also covers
`getMiddleMatrix(...)`, `ieClip(...)`, `ieHwhm2SD(...)`, `ieScale(...)`,
`ieScaleColumns(...)`, `ieCXcorr(...)`, `ieFitLine(...)`, `isodd(...)`,
`rotationMatrix3d(...)`, `unpadarray(...)`, `upperQuad2FullMatrix(...)`,
and `vectorLength(...)`, with direct headless wrappers for centered
matrix extraction, bounded clipping/scaling, Gaussian-width conversion,
rotation matrices, quadrant mirroring, and NaN-tolerant vector norms.
That same `utility/numerical` family is now closed too: the remaining
tail is covered by direct headless wrappers for `ffndgrid(...)`,
`ieCompressData(...)`, `ieLineAlign(...)`, `ieTikhonov(...)`, and
`qinterp2(...)`, which handle uneven-sample gridding, uint16/uint32 data
compression, shift-and-scale line alignment, ridge-plus-smoothness
regularization, and nearest/triangular/bilinear 2-D interpolation
without relying on MATLAB figures or toolbox state.
The neighboring `utility/statistics` family is now closed too: direct
headless wrappers now cover `biNormal(...)`, `expRand(...)`,
`gammaPDF(...)`, `getGaussian(...)`, `ieExprnd(...)`, `ieMvnrnd(...)`,
`ieNormpdf(...)`, `ieOneOverF(...)`, `iePoisson(...)`, `iePrcomp(...)`,
`iePrctile(...)`, `lorentzSum(...)`, `ieFractalDrawgrid(...)`, and
`ieFractaldim(...)`, including separable bivariate Gaussians, RF-support
Gaussian normalization, deterministic exponential and multivariate-normal
sampling, percentile fallback interpolation, radial 1/f spectra,
box-count fractal slopes, and magenta grid overlays without opening
MATLAB figures.
The adjacent `utility/hypercube` family is now closed too: direct
headless wrappers now cover `hcBlur(...)`, `hcIlluminantScale(...)`,
`hcReadHyspex(...)`, `hcReadHyspexImginfo(...)`, `hcViewer(...)`,
`hcimage(...)`, `hcimageCrop(...)`, and `hcimageRotateClip(...)`,
including ENVI header parsing and band-subset replay, per-plane Gaussian
blur, illuminant scale-map estimation, headless mean-gray/montage/movie
payloads, rect-based hypercube cropping, and percentile-clipped
rotation. The external ReDFISh notebook is explicitly treated as out of
scope.
The neighboring `utility/dll70` family is now closed as audit-only scope:
the remaining entries are deployment-era MATLAB MEX path selectors,
compiler helpers, Visual C++ redistribution installers, and host
MAC-address or licensing probes rather than supported standalone
pyisetcam runtime surfaces.
The adjacent `utility/gif` family is now closed as audit-only scope too:
the remaining files are generic MATLAB GIF or figure-export helpers built
around `imwrite`, `getframe`, `export_fig`, or `exportgraphics`, while the
supported pyisetcam runtime already handles object-pipeline GIF export
through higher-level headless video paths.
The neighboring `utility/xml` family is now closed too: direct headless
wrappers now cover `ieXML2struct(...)` / `xml2struct(...)` and
`ieStruct2XML(...)` / `struct2xml(...)`, including MATLAB-style
`_dash_`, `_colon_`, and `_dot_` field-name escaping, repeated-element
list replay, XML-string or file parsing, and `.xml` extension fallback on
save/load without any Java or MATLAB DOM dependencies.
The adjacent `utility/publish` family is now closed as audit-only scope:
the remaining files are MATLAB batch `publish(...)` notebooks for
generating local HTML or PDF artifacts from script and tutorial
directories, which is documentation-build behavior rather than a
supported pyisetcam runtime API.
The neighboring `utility/video` family is now closed as audit-only scope
too: the remaining `ieMovie.m` helper is a generic MATLAB figure/video
writer built around `imagesc`, `drawnow`, `getframe`, and `VideoWriter`,
while the supported pyisetcam runtime already handles object-pipeline
animation through higher-level headless helpers such as
`scene_make_video(...)` and `oi_preview_video(...)`.
That same top-level utility helper tail is now fully closed too: direct
headless wrappers now cover `ieFindFiles(...)`, `ieTone(...)`,
`ieUncompressData(...)`, `ieInit(...)`, and `ieRadiance2IP(...)`. The
browser-only `ieManualViewer.m` and ridge-regression notebook
`ieTikhonovRidge.m` are explicitly treated as out of scope, so the
remaining utility debt is now concentrated in the broader
`utility/programming` and `utility/external` families instead of the
standalone top-level helpers. Inside `utility/external`, the vendored
`JSONio` package is now explicitly tracked as out of scope because it is a
generic third-party MATLAB JSON integration layer rather than an ISETCam
runtime surface. The neighboring `dcraw` package is tracked the same way:
it is only MATLAB glue around external platform-specific `dcraw`
executables plus a demo script, while pyisetcam keeps direct headless
coverage only for the checked-in DNG path via `ie_dng_read(...)` and
`sensor_dng_read(...)`. The smaller `AddTextToImage` package is now
covered directly instead: `BitmapFont(...)`, `RasterizeText(...)`,
`AddTextToImage(...)`, and `AddTextToImageWithBorder(...)` all have
headless compatibility wrappers in `pyisetcam.fonts`, while the two demo
scripts remain out of scope. The next tier of tiny one-off external
helpers is now tracked as out of scope too: `ImageConvFrequencyDomain.m`,
`arrow3.m`, `bluewhitered.m`, `cprintf.m`, `cpuinfo.m`, and
`insertInImage.m` are generic third-party plotting, host-diagnostics, or
figure-annotation utilities rather than supported pyisetcam runtime
surfaces. The neighboring micro-packages are smaller too: the vendored
`DataHash_20190519` package and the GUI-only `fstack` preview or demo
files are explicitly out of scope, while `exiftoolInfo(...)`,
`exiftoolDepthFromFile(...)`, `max2(...)`, and `min2(...)` now have direct
headless compatibility coverage in the current runtime. The next tier is
smaller too: `comp_struct(...)`, `list_struct(...)`, `zernfun(...)`,
`zernfun2(...)`, and `zernpol(...)` now have direct headless compatibility
wrappers in `pyisetcam.utils`, while the third-party `Inpaint_nans`
numerical package and the GUI-only `freezeColors` figure helpers are
explicitly treated as out of scope.
The neighboring `utility/programming` family is now fully closed too:
direct headless compatibility coverage now explicitly tracks
`appendStruct(...)`, `cellDelete(...)`, `cellMerge(...)`, `checkfields(...)`,
`compareFields(...)`, `gatherStruct(...)`, `ieContains(...)`,
`replaceNaN(...)`, `struct2pairs(...)`, `ieStructCompare(...)`,
`ieStructRemoveEmptyField(...)`, `ieHash(...)`, and the broader `vc*` /
`ie*` session-object helper surface against the current
`pyisetcam.utils`, `pyisetcam.session`, and `pyisetcam.ptable` runtime.
The remaining environment-only helpers such as `checkToolbox.m`,
`clx.m`, `hiddenHandle.m`, `ieFindCallback.m`, `ieMemoryCheck.m`,
`ieMemorySize.m`, `ieQuestdlg.m`, and `notDefined.m` remain explicitly
tracked as out of scope. The last large external utility lane, the
vendored Oxford VGG facetracker package, is now explicitly treated as
out of scope too because it is a standalone ffmpeg or `VideoReader`,
`classdef`, `parfor`, pretrained-model, and MEX-backed video-processing
stack rather than a supported pyisetcam runtime surface.
The adjacent `utility/external/hdr` lane is now fully closed too:
`build_pyramid(...)`, `recons_pyramid(...)`, the Haar/QMF builders and
reconstructors, `pad_reflect(...)`, `pad_reflect_neg(...)`,
`modulateFlip(...)`, `imNorm(...)`, `finalTouch(...)`, `getPFMraw(...)`,
and `rangeCompressionLum(...)` now have direct headless compatibility
coverage in `pyisetcam.scene`, with focused regressions over padding
replay, Haar/QMF reconstruction, normalization, and PFM channel packing.
The remaining steerable-pyramid files in the vendored HDR package are now
explicitly tracked as out of scope because the upstream README requires
Eero Simoncelli's external MatlabPyrTools toolbox for that branch, and
the pinned ISETCam snapshot does not vendor the required helpers such as
`rcosFn`, `pointOp`, `steer2HarmMtx`, or `pyrBand`. With the facetracker
package also closed as external-toolbox scope, the pinned migration-gap
ledger is now fully closed at `703` parity entries, `328` ported
wrappers, and `412` out-of-scope paths, with no remaining open audit
entries.
Post-milestone compatibility work now also covers the public
`cameraCompute(..., 'ideal xyz')` lane: the headless camera wrapper reuses
the existing ideal-XYZ helper path and now returns consistent `ip xyz`,
`ip srgb`, and linear display-result payloads for both the direct
scene-start call form and the restart-from-`oi` form.
The neighboring explicit-`L3` camera path is tighter too: `cameraGet(...)`
and `ipGet(...)` now both support field-addressable `l3 ...` lookups on
top of the existing whole-payload retrieval, so vendored or synthetic L3
payloads can be queried through the same public gateway shape as upstream.
That same L3 path now also replays the upstream camera-set bookkeeping:
`cameraSet('L3 sensor size', ...)` and `cameraSet('L3 sensor fov', ...)`
update both the active camera sensor and the stored L3 design sensor
together instead of leaving the payload out of sync.
The public camera compute wrapper is tighter too: `cameraCompute(camera,
'oi', lrgbScale)` now accepts the upstream non-string third-argument form
and rescales the rendered linear RGB result to match the reference image
mean instead of treating that input as an unsupported mode token.
The neighboring camera-metrics gateway now also covers the upstream
`moire` branch: `cameraMoire(...)` returns a headless center-to-edge chroma
profile summary for the legacy `moire orient` scene, and
`metricsCamera(camera, 'moire')` now routes through that wrapper instead of
raising an unsupported-option error.
That same metrics gateway now also accepts the descriptive `visible snr`
name on top of the legacy short `vsnr` token, so both direct
`metricsCamera(...)` calls and `cameraGet(camera, 'metric', ...)` reuse the
same visual-SNR wrapper path.
The neighboring `cameraVSNR(...)` wrapper is tighter too: MATLAB-style
empty placeholders in the optional `lightLevels` and `exposureTime` slots
now fall back to the upstream defaults `[1 10 100]` and `0.01 s` instead
of being treated as literal empty arrays.
The adjacent `cameraFullReference(...)` wrapper now follows the same rule:
explicit empty `sceneNames` and `meanLuminances` placeholders fall back to
the upstream default Stuffed Animals benchmark scene plus the standard
`[3 6 12 25 50 100 200 400]` luminance sweep instead of producing an empty
benchmark request.
The same public camera gateway is tighter too: `cameraGet(...)` and
`cameraSet(...)` now reuse `ieParameterOtype(...)` for unprefixed unique
subobject parameters, so calls like `cameraSet(camera, 'fill factor', ...)`,
`cameraSet(camera, 'f number', ...)`, `cameraSet(camera, 'display dpi', ...)`,
and `cameraGet(camera, 'training illuminant')` follow the upstream
delegation shape without requiring explicit `pixel` / `optics` / `ip` / `l3`
prefixes.
The neighboring IP gateway is tighter too: `ipGet(..., 'result max')` and
`ipGet(..., 'max sensor')` now replay the upstream aliases used by legacy
camera utilities like `cameraVSNR.m`, and the same values are reachable
through `cameraGet(camera, 'ip result max')` and `cameraGet(camera, 'ip max sensor')`.
The public camera compute wrapper is tighter too: MATLAB-style empty mode
placeholders are now treated as omitted values, so
`cameraCompute(camera, scene, [], false)` replays the normal compute path
with `sensorResize=false` instead of misreading the empty third argument as
an output-scaling image.
That same parser now also treats empty `pType` placeholders as omitted, so
`cameraCompute(camera, [])` and `cameraCompute(camera, [], [], false)` both
fall back to the upstream default `'sensor'` restart path instead of
failing dispatch on an empty array payload.
The fourth positional slot now follows the same rule: empty or `None`
`sensorResize` placeholders are treated as omitted values, so
`cameraCompute(camera, scene, [], [])` still replays the default resize-to-scene
behavior instead of silently disabling sensor resizing.
The `pType` parser is tighter too: empty-string placeholders are now treated
as omitted values, so `cameraCompute(camera, '')` and
`cameraCompute(camera, '   ')` both fall back to the upstream default
`'sensor'` restart path instead of failing dispatch on blank strings.
The adjacent mode parser now follows the same rule too: blank-string mode
placeholders are treated as omitted values, so
`cameraCompute(camera, scene, '')` and
`cameraCompute(camera, scene, '   ')` both fall back to the upstream
default `'normal'` compute path instead of failing dispatch on blank
strings.
The fourth positional slot is now equally strict too: blank-string
`sensorResize` placeholders are treated as omitted values, so
`cameraCompute(camera, scene, [], '')` and
`cameraCompute(camera, scene, [], '   ')` both preserve the upstream
default resize-to-scene behavior instead of being interpreted through
Python string truthiness.
The adjacent `cameraComputesrgb(...)` wrapper is tighter too: MATLAB-style
empty placeholders in the optional `sz`, `scenefov`, and `scaleoutput`
slots now fall back to the same default sensor size, camera-matched scene
FOV, and unit output scale as the omitted-argument path instead of
crashing on empty arrays.
That same wrapper now also treats an empty `meanLuminance` placeholder as
an omitted value, so `cameraComputesrgb(camera, scene, [])` falls back to
the upstream default `100 cd/m^2` scene mean luminance instead of trying
to coerce an empty array to a scalar.
The neighboring `cameraComputeSequence(...)` wrapper now also accepts the
upstream MATLAB-style key/value form, and empty `exposuretimes` /
`nframes` placeholders in that form now fall back to the existing default
single-exposure behavior instead of being treated as literal empty arrays.
That same camera wrapper surface now also exposes the core MATLAB-style
gateway aliases directly from `pyisetcam.camera`, so imports like
`from pyisetcam.camera import cameraCreate`, `cameraCompute`,
`cameraComputesrgb`, `cameraComputeSequence`, `cameraClearData`,
`cameraGet`, and `cameraSet` work without routing through the package root.
That same sequence wrapper now also reuses the selected session scene when
`scenes` is omitted or passed as an empty placeholder, matching the
upstream current-scene default instead of requiring an explicit scene list
in every call.
The neighboring IP gateway is tighter too: `ipGet(...)` now accepts the
upstream `white point` / `wp` / `image white point` aliases for the stored
data white point, and `data or display white point` now falls back to the
attached display white point when no data-specific white point has been
set.
That same IP getter cluster now also replays the upstream `scaled result`,
`data ics illuminant corrected`, and `result primary` aliases on top of
the existing headless render and corrected-ICS payloads.
The adjacent quantization lane is tighter too: `ipGet(...)` now exposes the
stored quantization struct, method, bit depth, `max digital value`, and
the `data intensities dv` aliases instead of leaving those MATLAB-era
getters unsupported in headless mode.
That same IP bookkeeping surface now also replays the upstream
`transform list`, `gamma`, `data sensor`, and sensor-channel-count alias
families on top of the existing transform, render, and sensor-space
payloads.
The neighboring render-control lane is tighter too: `ipGet/Set(...,
'render whitept', ...)` now replays the vendored sensor-conversion
rewrite path headlessly, updating the current sensor conversion matrix and
locking `transform method` to `current`.
The adjacent IP geometry helpers are now in place too: `ipGet(...)`
supports the upstream `center`, `image grid`, `distance2center`, and
`angle` surfaces on top of the current result image size.
The neighboring bracketed-IP bookkeeping surface is tighter too:
`ipGet/Set(..., 'combination method', ...)` and the `combine exposures`
alias now replay the vendored default `longest` behavior and shared stored
method field.
That same IP wrapper surface now also exposes the core MATLAB-style
gateway aliases directly from `pyisetcam.ip`, so imports like
`from pyisetcam.ip import ipCreate`, `ipCompute`, `ipGet`, and `ipSet`
work without routing through the package root.
