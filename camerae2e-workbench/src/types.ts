export type Settings = {
  goal: string;
  cameraPreset: string;
  sceneType: string;
  seed: number;
  outputDir: string;
  fovDeg: number;
  pixelSizeUm: number;
  cfaPreset: string;
  oclMode: string;
  oclGroupShape: string;
  oclEqualization: number;
  exposureMs: number;
  analogGain: number;
  fNumber: number;
  lensPsfRadiusUm: number;
  fdtdEnabled: boolean;
  fdtdMode: string;
  fdtdCrosstalkStrength: number;
  tcadEnabled: boolean;
  hwIspEnabled: boolean;
  hwIspFrames: number;
  binningFactor: number;
  demosaicMethod: string;
  optimizationPreset: string;
  optimizationMethod: string;
  maxCandidates: number;
  datasetCaseCount: number;
  datasetSelection: string;
  includeTiff: boolean;
  includeStageOutputs: boolean;
};

export type PresetsResponse = {
  default_settings: Settings;
  goals: Array<{ id: string; name: string; description: string }>;
  camera_presets: Array<{
    id: string;
    name: string;
    hfov_deg: number;
    pixel_pitch_um: number;
    f_number: number;
    readiness_tier: string;
    truth_boundary: string;
  }>;
  optimization: {
    objective_presets: Array<{ id: string; name: string; metrics: unknown[] }>;
    presets: Record<string, string[]>;
  };
};

export type FidelityBadge = {
  label: string;
  active: boolean;
  available: boolean;
  tier: string;
  tone: string;
};

export type AssetStatus = {
  ok: boolean;
  assets: Record<
    string,
    {
      available: boolean;
      readiness_tier: string;
      badge: string;
      path?: string;
      root?: string;
      stale_reason?: string | null;
      truth_boundary?: string;
    }
  >;
  validation: {
    warning_count?: number;
    stale_dependency_count?: number;
    warnings?: Array<{ entry: string; kind: string; message: string }>;
  };
  fidelity_badges: FidelityBadge[];
};

export type MetricCard = {
  id: string;
  label: string;
  value: number | string | null;
  unit: string;
  area: string;
};

export type StageSummary = {
  name: string;
  available: boolean;
  shape: number[];
  dtype: string | null;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
};

export type SimulationResult = {
  elapsed_ms: number;
  stage_summaries: StageSummary[];
  metrics: MetricCard[];
  raw_metrics: Record<string, unknown>;
  parameter_lineage: Array<Record<string, unknown>>;
  artifact_lineage: Record<string, unknown>;
  preview_png: string | null;
  fidelity_badges: FidelityBadge[];
  truth_boundary: string;
};

export type Candidate = {
  case_index: number;
  seed: number;
  score: number | null;
  feasible: boolean;
  parameters: Record<string, unknown>;
  objective_values: Record<string, number | null>;
};

export type OptimizationResult = {
  elapsed_ms: number;
  method: string;
  search_method: string;
  case_count: number;
  feasible_count: number;
  pareto_case_count: number;
  best_case: Candidate | null;
  top_cases: Candidate[];
  pareto_points: Array<{
    case_index: number;
    score: number | null;
    x: number | null;
    y: number | null;
    clip: number | null;
  }>;
  objective: unknown;
  parameter_space: Record<string, unknown[]>;
  fidelity_badges: FidelityBadge[];
  truth_boundary: string;
};

export type DatasetResult = {
  ok: boolean;
  source: string;
  dataset_root: string;
  manifest_path: string;
  case_count: number;
  records: Array<{
    case_id: string;
    split: string;
    raw: string;
    rgb: string | null;
    labels: string;
    raw_shape: number[];
    raw_dtype: string;
    raw_sha256: string;
  }>;
  validation: { ok: boolean; issue_count: number; warning_count: number };
  truth_boundary: string;
};

export type ReportResult = {
  json_path: string;
  html_path: string;
};

