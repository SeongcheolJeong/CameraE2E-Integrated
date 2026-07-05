# CameraE2E Goal Roadmap

이 문서는 `camerae2e-technical-overview-images.pptx`가 설명한 목표와 현재 저장소 구현을 대조해, Research-grade E2E 최적화 플랫폼과 Perception 학습용 RAW 데이터 팩토리까지 가는 실행 기준을 정리한다.

중요한 경계는 명확하다. 현재 목표는 제품 sign-off가 아니다. FDTD, TCAD, RayOptics, HW ISP profile은 CameraE2E 파이프라인에 연결되어 있지만, measured calibration 없이 `calibrated` 또는 sign-off claim으로 승격하지 않는다.

## Readiness Tier

| Tier | 의미 |
|---|---|
| `missing` | 알려진 asset/API지만 현재 workspace에서 찾을 수 없음 |
| `available` | 로드 가능하지만 강한 검증 claim은 없음 |
| `proxy` | 연구 탐색에는 사용 가능하지만 물리/시스템 proxy가 포함됨 |
| `validated` | 저장소 검증 gate를 통과한 research-use 기능 |
| `calibration_required` | framework는 연결됐지만 정량 정확도에는 측정 calibration이 필요함 |
| `calibrated` | measured calibration evidence가 붙어 정량 분석에 쓸 수 있음 |

## Capability Matrix

| Area | Current tier | 구현된 것 | 남은 일 |
|---|---:|---|---|
| Scene | `validated` | spectral scene, chart scene, RGB/multispectral import, illuminant control | 실제 scene/illumination capture calibration |
| Optics / RayOptics | `proxy` | Lens DB 검색, RayOptics geometric PSF, CameraE2E optics 실행 | diffraction/wave-optics sign-off, flare/ghost/coating, 제조 오차 |
| Image Sensor | `proxy` | CFA/pixel/exposure/noise/RAW, CFA preset/Quad Bayer selector, analytic shared-OCL group equalization proxy, image-sensor selector DB | per-sensor calibrated process deck, measured n/k, CAD/GDS, sensor-specific LUT |
| FDTD optical LUT | `proxy` | QE/field/crosstalk proxy LUT, physics sanity checks | full convergence, localized crosstalk, measured optical stack |
| TCAD / DEVSIM | `calibration_required` | generation-map ingestion, split-PD current proxy, accuracy gate | active FDTD/TCAD lineage closure, carrier calibration, dark/noise/lag/full-well |
| HW ISP | `proxy` | rolling shutter, stage latency, queue, DMA, delayed AE/AWB | board/vendor trace calibration, AF/HDR/TNR detail |
| Metrics | `validated` | MTF, ISO12233, Delta E, SCIELAB, VSNR, SQRI | product-specific weighting and pass/fail gates |
| Optimization | `validated` | dot-path camera parameter grid/random/Latin-hypercube/evolutionary/surrogate/GP expected-improvement search, pixel geometry/CFA preset/Quad Bayer/readout/noise/optics-PSF/analytic OCL/FDTD-OCL configure catalog, preset parameter-space catalog, FACA objective scoring, hard constraints, Pareto front, selected scenarios, parameter-lineage evidence | hardware-in-loop calibration |
| Perception | `available` | task adapters, detection/segmentation/classification/pose/tracking metrics, robustness sweep | training loop, dataset-specific model calibration |
| RAW data factory | `validated` | manifest, metadata JSONL, deterministic RAW NPZ, split policy, checksum, labels JSON, optional per-stage NPZ, optional uncertainty/confidence NPZ, optional DNG-like ZIP container, validation, RAW-aware perception index, YOLO view export, ADAS/KITTI YOLO demo export, proxy camera-spec variant re-capture | standards-compliant DNG writer, automatic label synthesis, calibrated uncertainty posterior |
| DB/LUT registry | `validated` / `calibration_required` | manifest, readiness tier, provenance, dependency lineage, stale detection, calibration evidence manifest, readiness promotion plan, final-result-only Lens/Sensor DB package in `camerae2e_db/` and standalone `CameraE2E-DB` repo | actual measured evidence attachment and calibrated promotion |
| External pipeline | `calibration_required` | FDTD, TCAD, RayOptics source/config/small fixtures are merged under `simulations/`; runtime DB artifacts are split under `camerae2e_db/`/`CameraE2E-DB`; active assets are discoverable in one registry and simulation manifest | regenerated run lineage closure and calibrated end-to-end asset generation |

## Implemented API Surface

DB/LUT registry:

- `camerae2e_db_manifest()`
- `camerae2e_db_validate(strict=False)`
- `camerae2e_db_lineage(name)`
- `camerae2e_physics_pipeline_plan(strict=False)`
- `camerae2e_physics_simulation_manifest(...)`
- `camerae2e_physics_simulation_validate(...)`
- `camerae2e_physics_simulation_commands(...)`
- `camerae2e_goal_gate(...)`
- `camerae2e_calibration_evidence_requirements(...)`
- `camerae2e_calibration_evidence_manifest(...)`
- `camerae2e_calibration_evidence_validate(...)`
- `camerae2e_readiness_promotion_plan(...)`
- `image_sensor_db_config(...)`
- `image_sensor_db_optimize_camera_parameters(...)`

PPT gap audit:

- `camerae2e_ppt_gap_audit(...)`

`camerae2e_goal_gate(...)` is the top-level research-platform evidence gate.
It regenerates a machine-readable pass/warn/fail matrix over registry,
physics-pipeline lineage, calibration evidence policy, image-sensor DB
hybrid/analytic configuration policy, FACA smoke, camera-parameter optimization,
RAW dataset export, ADAS/KITTI YOLO demo export, camera-spec variant
re-capture, and the strict sign-off claim guard. Non-strict mode is the normal
research gate. Strict mode is expected to fail while proxy or
calibration-required assets remain active.

The calibration evidence APIs define the measured artifacts required before a
registry entry can be promoted to `calibrated`. For example HW ISP promotion
requires board latency traces, hardware counter traces, and 3A telemetry traces;
sensor promotion requires measured optical/electrical evidence and lineage
closure. Without those artifacts the promotion plan remains blocked, which is
the intended behavior.

Validate a calibration evidence bundle with:

```bash
python tools/validate_camerae2e_calibration_evidence.py path/to/manifest.json
```

Default output:

- `reports/camerae2e_goal/calibration_evidence_validation.json`

Final Lens/Sensor DB repository:

- `camerae2e_db/` is the small in-repository mirror.
- `/Users/seongcheoljeong/Documents/CameraE2E-DB` is the standalone DB repo.
- `tools/package_camerae2e_db_repository.py` regenerates both from the curated
  final artifacts.
- `tools/publish_camerae2e_db_repository.py` validates and pushes the standalone
  repo after `SeongcheolJeong/CameraE2E-DB` has been created on GitHub.
- Runtime discovery accepts `PYISETCAM_CAMERA_DB_ROOT`; older scripts can set
  `PYISETCAM_LENS_DB_ROOT` and `PYISETCAM_FDTD_ROOT` explicitly.

Standalone DB GitHub publish command:

```bash
python3 tools/publish_camerae2e_db_repository.py \
  --target /Users/seongcheoljeong/Documents/CameraE2E-DB
```

If that command reports `Repository not found`, create an empty GitHub repo
named `SeongcheolJeong/CameraE2E-DB` first. The helper intentionally does not
create repositories because repository creation requires authenticated GitHub
account state that is outside the local CameraE2E codebase.

System FACA:

- `camerae2e_run_scenario(...)`
- `camerae2e_run_sweep(...)`
- `camerae2e_faca_report(...)`

Optimization:

- `camerae2e_optimize_parameters(...)`
- `camerae2e_optimize_camera_parameters(...)`
- `camerae2e_parameter_space_catalog(...)`
- `camerae2e_parameter_candidate_plan(...)`
- `camerae2e_optimization_config_catalog(...)`
- `camerae2e_parameter_space_validate(...)`
- `camerae2e_optimization_report(...)`
- `camerae2e_pareto_front(...)`
- `camerae2e_optimization_escalation_plan(...)`

The first optimizer is deterministic grid search over dot-path camera
parameters such as `sensor.integration_time`, `sensor.analog_gain`,
`sensor.pixel_size`, `sensor.pixel_fill_factor`, `sensor.cfa_preset`,
`sensor.cfa_pattern`, `sensor.binning_method`, `sensor.binning_factor`,
`sensor.ocl_vignetting`, `sensor.ocl_group_shape`,
`sensor.ocl_group_equalization`, `sensor.ocl_fnumber`, `sensor.ocl_focal_length_um`,
`sensor.ocl_refractive_index`, `sensor.pixel_read_noise_v`, `optics.fnumber`,
`optics.focal_length`, `optics.si_psf_radius_um`, `fdtd.crosstalk_strength`,
`fdtd.ocl_shift_um`, `fdtd.cra_x_deg`, `fdtd.cra_z_deg`,
`ip.demosaic_method`, and HW ISP control-delay parameters.
`camerae2e_parameter_space_catalog(...)` exposes validated preset search spaces
such as `exposure`, `raw_factory`, `sensor_geometry`, `sensor_spectral`,
`sensor_readout`, `sensor_ocl`, `optics_psf`, `raytrace_psf`,
`physics_proxy`, `adas_camera`, `isp`, and `hw_isp_control`.
`camerae2e_optimization_config_catalog(...)` lists every registered configure
axis, custom dot-path assignment rule, and supported FACA objective metric path.
`camerae2e_parameter_space_validate(...)` classifies a caller parameter space as
`registered`, `assignable`, `custom_passthrough`, or blocked before any expensive
sweep is launched. It also validates high-value axis values: positive exposure,
gain, pixel size, f-number, focal length, PSF radius, odd
`sensor.n_samples_per_pixel`, CFA preset/matrix shape/integer constraints, supported
binning/demosaic/OCL group/OCL mode tokens, supported 1/off or 2x legacy
`sensor.binning_factor`, nonnegative FDTD crosstalk strength, finite FDTD CRA/OCL
selectors, and HW ISP delay integer constraints. This prevents false optimization runs where an axis
is syntactically accepted but does not affect the camera pipeline or fails deep
inside sensor/optics computation.
`camerae2e_parameter_candidate_plan(...)` turns the validated space into a
reproducible candidate list before running FACA. The default `grid` method keeps
the previous Cartesian behavior. Passing `max_cases` turns it into a budgeted
grid by sampling evenly across the Cartesian index range. `method="random"` and
`method="latin_hypercube"` sample discrete axis values with a fixed seed and
default to a bounded budget when `max_cases` is omitted. `method="evolutionary"`
starts from a deterministic seed population, evaluates FACA objective fitness,
then expands remaining budget through score-ranked elite selection, uniform
discrete crossover, and mutation. `method="surrogate"` starts from a deterministic
seed population and then chooses unevaluated discrete-axis candidates from a
bounded pool using an RBF/inverse-distance expected-improvement proxy plus
uncertainty. `method="gaussian_process"`, `method="gp"`, or `method="bayesian"`
uses an RBF Gaussian-process posterior over the discrete FACA objective surface
and expected improvement acquisition. This is an optimizer for research
ranking, not a calibrated physical sensor, lens, or ISP model. The optimizer
accepts the same `method` and `max_cases` arguments and records the resulting
candidate-plan summary and generation trace in optimization reports.
`camerae2e_optimize_camera_parameters(...)` runs those presets directly while
still allowing caller overrides. The optimizer maximizes objective paths from
the FACA report, supports hard metric constraints, reports the feasible Pareto
front, and emits selected scenario configs that can be passed into the RAW data
factory. FACA and dataset records include parameter-lineage entries with
requested/before/after values so a RAW export can be traced back to the actual
camera parameters applied. This is the reproducible baseline for later
hardware-in-loop calibration and measured-objective fitting.
`camerae2e_optimization_escalation_plan(...)` takes a completed optimization
result and plans the next fidelity step for selected best/top/Pareto candidates:
DB/LUT anchoring, FDTD optical LUT batch generation, TCAD/DEVSIM collection
checks, RayOptics geometric PSF comparison, HW ISP trace replay, RAW export,
and perception-index evaluation. This is the intended combination path:
analytic/proxy search is used to reduce the space, and expensive physics or
measured calibration evidence is reserved for short-listed candidates.

Important boundary: `sensor.n_samples_per_pixel` is sub-pixel integration
sampling, not readout binning. `sensor.binning_method` and
`sensor.binning_factor` currently route to the legacy 2x binning wrapper and
remain `proxy` until charge-domain, readout-domain, and ISP-domain binning are
separated. `sensor.ocl_vignetting`/microlens axes use the current
etendue/vignetting model. `sensor.cfa_preset` expands named layouts such as
Bayer and Quad Bayer into CFA matrices while reusing the current filter
spectra/QE. `sensor.ocl_group_shape` and `sensor.ocl_group_equalization` are
analytic shared-aperture proxies: they equalize the spatial optical sample
within a group per spectral/current plane before CFA selection and therefore
model "no extra optical resolution under one OCL" without mixing color-filter
outputs. OCL offset/height/process-stack optimization should not be registered
as validated until fixed-offset etendue or FDTD/TCAD-backed OCL LUTs are
attached. `optics.si_psf_radius_um` is a synthetic
shift-invariant pillbox PSF radius for blur sensitivity sweeps; it must not be
interpreted as a RayOptics geometric PSF radius or diffraction/wave-optics
sign-off. FDTD/OCL axes such as `fdtd.crosstalk_strength`,
`fdtd.ocl_shift_um`, and FDTD CRA selectors only affect the pipeline when an
FDTD LUT is attached in the base scenario.

Sensor DB and free-configuration runs intentionally use different defaults.
When a sensor DB record is selected, DB/LUT artifacts should override or
calibrate the proxy defaults. When the goal is to configure many hypothetical
sensors, analytic proxy axes are the fast search engine. FDTD/TCAD/RayOptics
then act as offline LUT sources, calibration anchors, or validation batches for
top candidates, not as per-candidate solvers inside every optimization loop.
The escalation planner makes that policy machine-readable so downstream tools
can see which candidates still need LUT refresh, stale-dependency closure, or
measured evidence before stronger claims are made.

Optimization configure priority:

| Group | High-value axes | Current handling |
|---|---|---|
| Exposure/noise/readout | `sensor.integration_time`, `sensor.analog_gain`, `sensor.pixel_read_noise_v`, `sensor.pixel_dark_voltage`, `sensor.pixel_voltage_swing`, `sensor.pixel_conversion_gain`, `sensor.noise_flag`, `sensor.binning_method`, `sensor.binning_factor` | registered validated/proxy axes; binning is legacy 2x proxy |
| Sensor geometry/sampling | `sensor.pixel_size`, `sensor.pixel_fill_factor`, `sensor.n_samples_per_pixel`, sensor resolution/FOV custom paths | pixel/sampling validated; FOV/size custom paths require lineage checks |
| CFA/spectral | `sensor.cfa_preset`, `sensor.cfa_pattern`, `sensor.filter_names`, `sensor.filter_spectra`, IR/QE custom paths | Bayer/Quad Bayer presets registered; spectra/name changes require consistency checks |
| OCL/microlens | `sensor.ocl_vignetting`, `sensor.ocl_group_shape`, `sensor.ocl_group_equalization`, `sensor.ocl_fnumber`, `sensor.ocl_focal_length_um`, `sensor.ocl_refractive_index`; future `ocl_offset/height/stack` via FDTD/TCAD LUT | etendue/vignetting and shared-aperture analytic proxies; refractive index calibration-required |
| Optics/PSF | `optics.focal_length`, `optics.fnumber`, `optics.si_psf_radius_um`, `optics.psf_angle_step`, `optics.rt_compute_spacing`, distortion/relative-illumination custom paths | focal/f-number validated; PSF radius/RayOptics sampling proxy |
| FDTD/TCAD/OCL | `fdtd.mode`, `fdtd.crosstalk_strength`, `fdtd.ocl_shift_um`, `fdtd.cra_x_deg`, `fdtd.cra_z_deg`, `tcad.collection_mode` | guarded by attached LUT/DB; proxy or calibration-required |
| ISP/control | `ip.demosaic_method`, black level/tone/gamma/custom IP paths, HW ISP AE/AWB delay and latency axes | demosaic validated; HW ISP proxy |
| Perception/data | selected camera-spec scenario, RAW split/export, label preservation, YOLO view, robustness metrics | validated factory outputs; no automatic label synthesis |

RAW data factory:

- `camerae2e_dataset_export(...)`
- `camerae2e_dataset_export_adas_kitti_demo(...)`
- `camerae2e_dataset_export_camera_spec_variants(...)`
- `camerae2e_adas_camera_spec(...)`
- `camerae2e_kitti_yolo_labels(...)`
- `camerae2e_dataset_export_from_optimization(...)`
- `camerae2e_dataset_export_perception_index(...)`
- `camerae2e_dataset_validate(...)`

`camerae2e_dataset_export(...)` can also emit optional per-stage NPZ bundles,
proxy uncertainty/confidence maps, and a deterministic DNG-like ZIP container
that groups RAW, labels, previews, stage outputs, and metadata. That container
is deliberately not a standards-compliant DNG file. A valid DNG writer remains
a separate milestone because TIFF/DNG tag compliance, CFA metadata, color
matrices, black/white levels, and reader interoperability need their own
validation.

ADAS/KITTI demo export is intentionally labeled `proxy`: it applies KITTI-style
object-detection geometry and YOLO/KITTI label metadata to the RAW factory, but
does not claim KITTI raw sensor ground truth or measured ADAS camera
calibration. Regenerate the demo with:

```bash
python tools/render_adas_kitti_raw_demo.py
```

For camera-spec transformation, use:

```bash
python tools/render_adas_kitti_raw_demo.py --variants \
  --output-dir outputs/adas-kitti-camera-variants-demo
```

This performs a proxy re-capture of the same KITTI-style RGB scene through
target camera specs such as `wide_fov_adas_demo` and `narrow_fov_adas_demo`.
The default `geometric_transform="pinhole_crop"` applies a focal-ratio
center crop/resize approximation to the source RGB and remaps/clips labels in
the same coordinate transform. Narrow FoV therefore behaves like a zoomed crop;
wide FoV exposes source-outside regions that are explicitly filled and counted
in metadata. It is useful for controlled robustness experiments, but it does not
recover true KITTI spectral radiance, depth, occlusion, lens flare, ISP inverse,
or measured RAW.

## External Pipeline Policy

FDTD/TCAD and RayOptics simulation source is now monorepo-visible under
`simulations/fdtd_tcad/` and `simulations/rayoptics/`. The import policy keeps
source, configs, sensor/lens DB fixtures, and small canonical loader fixtures in
Git, while excluding `runs/`, virtual environments, `node_modules`, zip bundles,
and generated web/build artifacts. Refresh the monorepo copy with:

```bash
python tools/import_camerae2e_simulation_workspaces.py
python tools/render_camerae2e_physics_simulation_manifest.py
```

FDTD and TCAD artifacts must still be checked as a connected lineage, not as
isolated files. If the active FDTD LUT is from one run and the TCAD generation
map is from another, the registry and simulation manifest mark the TCAD bridge
with `stale_dependency`. This does not block research runs by default, but it
blocks strict validation.

RayOptics PSF assets are kept at `proxy` tier because they are geometric ray histograms. Diffraction and wavefront analysis remain separate CameraE2E optics paths and should be compared explicitly when a design decision depends on diffraction.

HW ISP seed profiles are system-simulation inputs. They become calibrated only when replaced or fitted with board traces, hardware counters, BSP timing, or equivalent measured evidence.

## Report Outputs

Audit the current image-only technical overview PPT claim ledger with:

```bash
python tools/render_camerae2e_ppt_gap_audit.py \
  --pptx /Users/seongcheoljeong/Downloads/camerae2e-technical-overview-images.pptx
```

Outputs:

- `reports/camerae2e_goal/ppt_gap_audit.json`
- `reports/camerae2e_goal/ppt_gap_audit.html`

Regenerate the current goal-readiness reports with:

```bash
python tools/render_camerae2e_asset_registry_report.py
```

Outputs:

- `reports/camerae2e_goal/readiness.json`
- `reports/camerae2e_goal/readiness.html`
- `reports/camerae2e_goal/asset_registry.json`
- `reports/camerae2e_goal/asset_registry.html`

Run the goal-level evidence gate with:

```bash
python tools/run_camerae2e_goal_gate.py
```

Outputs:

- `reports/camerae2e_goal/goal_gate.json`
- `reports/camerae2e_goal/goal_gate.html`
- smoke RAW artifacts under `outputs/camerae2e-goal-gate-smoke/`

Validate external physics pipeline lineage with:

```bash
python tools/validate_camerae2e_physics_pipeline.py
```

The validation payload includes a `plan` section with refresh/calibration
actions, and the registry report command also writes:

- `reports/camerae2e_goal/physics_pipeline_plan.json`
- `reports/camerae2e_goal/physics_simulation_manifest.json`
- `reports/camerae2e_goal/physics_simulation_manifest.html`
