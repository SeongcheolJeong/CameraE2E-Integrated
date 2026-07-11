export type FidelityLevel = "L0_analytic" | "L1_lut" | "L2_solver" | "L3_calibrated";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";

export type Quantity = { value: number; unit: string };

export type RequirementSet = {
  mission: "adas" | "mobile" | "general";
  hfov: Quantity;
  target_classes: string[];
  min_object_height: Quantity;
  detection_range: Quantity;
  illuminance_range: [number, number];
  max_latency: Quantity;
  max_sensor_diagonal: Quantity | null;
  reference_object_height: Quantity;
  min_frame_rate: Quantity;
  max_pixel_rate: Quantity;
  max_rolling_shutter: Quantity;
  max_clip_fraction: number;
  min_snr_db: number;
  reference_wavelength_nm: number;
  notes: string;
};

export type CameraModule = {
  id: string;
  name: string;
  lens: {
    model_id: string | null;
    hfov_deg: number;
    f_number: number;
    focal_length_mm: number | null;
    psf_radius_um: number;
    distortion_model: string;
    relative_illumination_model: string;
  };
  sensor: {
    model_id: string | null;
    rows: number;
    cols: number;
    native_rows: number | null;
    native_cols: number | null;
  geometry_source: string;
  pixel_size_um: number;
  pixel_fill_factor: number;
  simulation_pixel_size_um: number | null;
  simulation_fill_factor: number | null;
  cfa_preset: string;
    qe_profile: string;
    exposure_ms: number;
    analog_gain: number;
    read_noise_e: number | null;
    full_well_e: number | null;
    binning_factor: number;
    ocl_mode: string;
    ocl_group_shape: string;
    ocl_equalization: number;
    bit_depth: number;
    frame_rate_fps: number;
    row_time_us: number;
  };
  isp: {
    demosaic_method: string;
    ccm_method: string;
    ccm_matrix: number[][] | null;
    tone_method: string;
    gamma: number;
    ccm_regularization: number;
    ccm_row_sum_tolerance: number;
    ccm_max_abs: number;
  };
  hw_isp_enabled: boolean;
  hw_isp_frames: number;
  metadata: Record<string, unknown>;
};

export type SceneCase = {
  id: string;
  name: string;
  source_kind: "physical" | "measured_proxy" | "display_rgb_proxy" | "synthetic";
  scene_type: "macbeth" | "slanted_bar" | "uniform" | "rgb_file" | "multispectral_file";
  image_path: string | null;
  label_path: string | null;
  mean_luminance_cd_m2: number;
  illuminance_lux: number | null;
  weather: string;
  metadata: Record<string, unknown>;
};

export type DesignVariable = {
  path: string;
  unit: string;
  values: unknown[] | null;
  minimum: number | null;
  maximum: number | null;
  steps: number;
  readiness_tier: string;
  enabled: boolean;
};

export type Objective = {
  id: string;
  metric: string;
  direction: "maximize" | "minimize" | "target";
  weight: number;
  target: number | null;
  display_name: string | null;
};

export type Constraint = {
  id: string;
  metric: string;
  operator: "<=" | ">=" | "<" | ">" | "==";
  value: number;
  hard: boolean;
  display_name: string | null;
};

export type BenchmarkSuite = {
  name: string;
  metric_version: string;
  source_root: string | null;
  quick_scene_count: number;
  final_scene_count: number;
  halving_scene_counts: number[];
  final_candidate_count: number;
  stratification: "adas_class_and_object_size" | "ordered";
  robustness_cases: Array<{
    name: string;
    kind: "exposure_ev" | "psf_scale" | "ocl_equalization" | "low_light";
    amount: number;
  }>;
  detector_map50_min: number;
  detector_map50_95_min: number;
  detector_recall_min: number;
  ideal_recall_retention_min: number;
  ideal_ssim_min: number;
  fidelity_scene_count: number;
  fidelity_recall_retention_min: number;
  fidelity_color_imbalance_max: number;
  fidelity_ssim_min: number;
  uncertainty_bootstrap_samples: number;
  confidence_level: number;
  minimum_meaningful_score_delta: number;
};

export type StudySpec = {
  name: string;
  requirements: RequirementSet;
  baseline: CameraModule;
  scenes: SceneCase[];
  design_variables: DesignVariable[];
  objectives: Objective[];
  constraints: Constraint[];
  fidelity_policy: {
    search_level: FidelityLevel;
    validation_level: FidelityLevel;
    allow_proxy: boolean;
    require_fresh_lineage: boolean;
    require_calibration_for_signoff: boolean;
    solver_timeout_s: number;
  };
  benchmark: BenchmarkSuite;
  target_profile: "raw_quality" | "adas_yolo_perception";
  perception_model_path: string | null;
  seed: number;
  search_budget: number;
  search_method: "grid" | "random" | "latin_hypercube" | "evolutionary" | "surrogate" | "gaussian_process";
};

export type StudyRecord = {
  id: string;
  project_id: string;
  revision: number;
  status: string;
  spec: StudySpec;
  created_at: string;
  updated_at: string;
};

export type ArtifactRecord = {
  hash: string;
  artifact_type: string;
  relative_path: string;
  size_bytes: number;
  media_type: string;
  fidelity_level: FidelityLevel;
  readiness_tier: string;
  source: string;
  dependencies: string[];
  validation: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type ProjectPayload = {
  info: {
    id: string;
    name: string;
    path: string;
    schema_version: string;
    created_at: string;
    updated_at: string;
  };
  studies: StudyRecord[];
  job_summary: { total: number; running: number; failed: number };
  artifact_summary: {
    count: number;
    validation: { ok: boolean; artifact_count: number; issue_count: number; issues: unknown[] };
  };
  camera_asset_summary: { count: number };
};

export type JobRecord = {
  id: string;
  project_id: string;
  study_id: string | null;
  kind: string;
  status: JobStatus;
  progress: number;
  request: Record<string, unknown>;
  result: Record<string, any> | null;
  error: string | null;
  log: string[];
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  artifacts?: Array<{ role: string; artifact: ArtifactRecord }>;
};

export type JobSubmitResponse = { job: JobRecord; poll_url: string };

export type ComponentSelection = {
  lens_id: string;
  sensor_id: string;
  name?: string | null;
  use_geometric_psf: boolean;
};

export type LensComponent = {
  id: string;
  lens_id: string;
  company: string;
  publication_number: string;
  configuration: string;
  readiness: string;
  simulation_status: string;
  simulation_model: string;
  focal_length_mm: number | null;
  f_number: number | null;
  image_height_mm: number | null;
  field_of_view_deg: number | null;
  airy_disk_diameter_um_550: number | null;
  diffraction_cutoff_lpmm_550: number | null;
  surface_count: number | null;
  asphere_count: number | null;
  psf_available: boolean;
  module_configurable: boolean;
  configuration_blockers: string[];
  readiness_tier: string;
  value_kind: string;
  evidence_completeness: number;
};

export type SensorComponent = {
  id: string;
  code: string;
  manufacturer: string;
  device_name: string;
  title: string;
  analysis_year: number | null;
  pixel_pitch_um: number | null;
  resolution_mp: number | null;
  native_size_rc: [number, number] | null;
  geometry_source: string;
  sensor_modality: string | null;
  frame_model_supported: boolean;
  cfa_model_supported: boolean;
  module_configurable: boolean;
  configuration_blockers: string[];
  cfa_pattern: string | null;
  shutter: string | null;
  illumination: string | null;
  optical_format: string | null;
  microlens_type: string | null;
  pixel_architecture: string | null;
  has_dti: boolean | null;
  has_pdaf: boolean | null;
  has_hdr: boolean | null;
  has_lofic: boolean | null;
  is_stacked: boolean | null;
  is_nir: boolean | null;
  stack_config_available: boolean;
  tcad_profile_available: boolean;
  readiness_tier: string;
  value_kind: string;
  evidence_completeness: number;
};

export type ComponentSearchResponse<T> = {
  schema_version: string;
  total: number;
  page: number;
  page_size: number;
  items: T[];
  facets: Record<string, any>;
};

export type ModuleCompatibilityCandidate = {
  id: string;
  selection: ComponentSelection;
  status: "compatible" | "conditional" | "incompatible";
  pareto: boolean;
  lens: LensComponent;
  sensor: SensorComponent;
  gates: Array<{
    id: string;
    value: unknown;
    limit: string;
    unit: string;
    pass: boolean | null;
    status: "pass" | "fail" | "not_evaluable";
    note: string;
  }>;
  failed_gate_ids: string[];
  not_evaluable_ids: string[];
  derived: Record<string, number | string | boolean | null>;
  truth_boundary: string;
};

export type ModuleCompatibilityResponse = {
  schema_version: string;
  requirements: RequirementSet;
  candidate_count: number;
  status_counts: Record<string, number>;
  candidates: ModuleCompatibilityCandidate[];
  truth_boundary: string;
};

export type ModuleApplicationResponse = {
  schema_version: string;
  study: StudyRecord;
  module: CameraModule;
  compatibility: ModuleCompatibilityCandidate;
  asset: Record<string, any>;
  artifact: ArtifactRecord;
};

export type AssetStatus = {
  schema_version: string;
  manifest_summary: Record<string, unknown>;
  validation: {
    ok: boolean;
    warning_count: number;
    stale_dependency_count: number;
    warnings: Array<Record<string, unknown>>;
  };
  fidelity_levels: FidelityLevel[];
  project_artifacts?: { ok: boolean; artifact_count: number; issue_count: number };
  solver_adapters: Record<string, { available: boolean; execution: string }>;
  qe_profiles?: Record<string, { available: boolean; readiness: string }>;
  benchmark?: {
    root: string | null;
    available_scene_count: number;
    quick_scene_count: number;
    final_scene_count: number;
    enough_for_quick: boolean;
    enough_for_final: boolean;
    strata: Record<string, number>;
  };
  perception?: {
    model_path: string | null;
    model_ready: boolean;
    label_ready: boolean;
  };
  calibration_pack?: {
    schema_version: string;
    complete: boolean;
    readiness_tier: string;
    missing_kinds: string[];
    product_signoff_ready: boolean;
    stages: Record<string, {
      complete: boolean;
      readiness_tier: string;
      required_kinds: string[];
    }>;
  };
};

export type BenchmarkStatus = {
  schema_version: string;
  study_id: string;
  inventory: NonNullable<AssetStatus["benchmark"]>;
  preflight: Record<string, any> | null;
  optimization_ready: boolean;
  latest_benchmark: Record<string, any> | null;
  latest_requirements: Record<string, any> | null;
  latest_optimization: Record<string, any> | null;
};
