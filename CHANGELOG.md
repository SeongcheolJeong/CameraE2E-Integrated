# Changelog

## Unreleased

- Added a camera-aware KITTI RAW factory with split/calibration inventory, storage
  estimation, source-bounded and target-readout policies, exposure/noise recipes, and
  resumable content-addressed samples.
- Added KITTI P2-to-target pinhole image and label transforms plus RAW NPZ metadata for
  sensor digital values, CFA, bit depth, black level, and white level.
- Added v3 dataset manifests, inventory/estimate APIs, integrity validation, and a
  production Workbench workflow for inspecting and exporting camera-specific RAW data.

## 0.2.0-research

- Added the Lens/Sensor Component Explorer with normalized search, detail inspection,
  simulation-readiness filters, and explicit source/proxy boundaries.
- Added requirement-driven module compatibility gates, Pareto screening, baseline
  application, and two-to-four candidate same-scene comparison.
- Connected bundled RayOptics geometric PSFs to real CameraE2E execution without
  presenting them as diffraction or measured MTF evidence.
- Separated native sensor geometry from bounded local execution and added a full-extent
  sparse-sampling proxy that preserves FOV and native photodiode area.
- Added component APIs, decision-report integration, desktop/mobile Workbench UX,
  regression tests, and English/Korean manuals.

This release is research-grade. Event/NIR/SWIR acquisition, sensor-specific measured
QE/noise, wave-optics sign-off, and product calibration remain outside its validated scope.

## 0.1.0-research

- Integrated CameraE2E, Camera DB, FDTD/TCAD assets, RayOptics assets, and Workbench.
- Added persistent requirement, study, job, artifact, optimization, dataset, calibration,
  and report workflows.
- Added analytic/LUT fidelity routing with explicit research and calibration boundaries.
- Added KITTI/YOLO ADAS preflight and perception-first optimization without fake fallback
  scores.
- Added deterministic virtual RAW export and content-addressed evidence manifests.
- Added user manuals, troubleshooting, contribution guidance, CI, and local launcher.

This release is research-grade and calibration-ready. It does not claim product sign-off.
