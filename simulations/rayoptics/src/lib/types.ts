export type UpdateMode = "auto" | "manual" | "layout-only" | "editors-only";
export type WorkflowStage =
  | "project"
  | "optical-model"
  | "analysis"
  | "optimization"
  | "tolerance"
  | "sensor"
  | "scene"
  | "compare"
  | "report";
export type KpiStatus = "pass" | "warn" | "fail";
export type MetricSource = "computed" | "model" | "proxy" | "diagnostic" | "unsupported" | "assumption";
export type AnalysisScale = "all" | "same";
export type QuickOptimizeObjective = "balanced" | "spot" | "distortion" | "throughput" | "cra" | "custom";
export type QuickOptimizeOperandKey = "spot" | "distortion" | "throughput" | "cra" | "first_order";
export type QuickOptimizeWeights = Record<QuickOptimizeOperandKey, number>;
export type QuickOptimizeTargets = Partial<Record<Exclude<QuickOptimizeOperandKey, "first_order">, number>>;
export type WorkbenchMetadata = Record<string, unknown>;

export interface RecentModelFile {
  path: string;
  label: string;
  kind: "open" | "save" | "example";
  touchedAt: string;
  hasWorkbench: boolean;
}

export interface WorkbenchFileStatus {
  state: "new" | "opened" | "restored" | "saved";
  label: string;
  detail: string;
  path: string | null;
  at: string;
  hasWorkbench: boolean;
}

export interface CockpitMetric {
  key: string;
  label: string;
  value: string;
  unit?: string | null;
  target: string;
  status: KpiStatus;
  score: number;
  source: MetricSource;
  note?: string | null;
}

export interface CockpitRisk {
  label: string;
  value: string;
  status: KpiStatus;
  detail: string;
  source: MetricSource;
}

export interface SurfaceRow {
  index: number;
  label: string;
  type: string;
  radius: number | null;
  curvature: number;
  thickness: number | null;
  glass: string;
  catalog: string;
  semiDiameter: number;
  conic: number | null;
  mode: "transmit" | "reflect" | "dummy" | "phantom";
  isStop: boolean;
  variable: string;
}

export interface SystemPatch {
  apertureValue?: number;
  fieldValue?: number;
  fieldXValues?: number[];
  fieldYValues?: number[];
  fieldWeights?: number[];
  focusShift?: number;
  defocusRange?: number;
  wavelengthValues?: number[];
  wavelengthWeights?: number[];
  wavelengthReferenceIndex?: number;
}

export interface NewModelSpec {
  efl: number;
  epd: number;
  fov: number;
}

export interface OpticalModel {
  id: string;
  name: string;
  filename: string | null;
  radiusMode: boolean;
  updateMode: UpdateMode;
  stopSurface: number | null;
  surfaces: SurfaceRow[];
  system: {
    aperture: { key: string[]; value: number | null };
    field: { key: string[]; value: number | null; fields: Array<{ x: number | null; y: number | null; weight?: number | null }>; isRelative: boolean };
    wavelengths: { values: Array<number | null>; weights: Array<number | null>; reference: number };
    focus: { focusShift: number | null; defocusRange: number | null };
  };
}

export interface ModelResponse {
  model: OpticalModel;
  warnings: string[];
  errors: string[];
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  lastUpdatedAt: string;
  workbench?: WorkbenchMetadata | null;
}

export interface QuickOptimizeMove {
  surfaceIndex: number;
  token: string;
  label: string;
  before: number;
  after: number;
  score: number;
}

export interface QuickOptimizeResult {
  status: "improved" | "no-change" | "failed";
  objective: QuickOptimizeObjective;
  message: string;
  baselineScore: number;
  finalScore: number;
  improvement: number;
  evaluations: number;
  iterations: number;
  variableCount: number;
  operandWeights: QuickOptimizeWeights;
  variables: string[];
  moves: QuickOptimizeMove[];
}

export interface QuickOptimizeResponse extends ModelResponse {
  result: QuickOptimizeResult;
}

export type ToleranceSweepScope = "powered" | "variables";

export interface ToleranceSweepCase {
  label: string;
  surfaceIndex: number;
  token: string;
  perturbationPct: number;
  before: number;
  after: number;
  score: number;
  status: KpiStatus;
  spotRmsUm: number | null;
  mtf50LpMm: number | null;
  throughput: number | null;
  distortionPct: number | null;
  craDeg: number | null;
  traceFailures: string[];
}

export interface ToleranceSweepResult {
  status: KpiStatus;
  scope: ToleranceSweepScope;
  perturbationPct: number;
  baselineScore: number;
  attemptedCases: number;
  passedCases: number;
  warnedCases: number;
  failedCases: number;
  worstCase: string | null;
  worstScore: number | null;
  cases: ToleranceSweepCase[];
  warnings: string[];
}

export interface ToleranceSweepResponse extends ModelResponse {
  result: ToleranceSweepResult;
}

export type VariantSnapshotSource = "manual" | "quick-optimize-before" | "quick-optimize-after" | "open" | "new";

export interface VariantSnapshot {
  id: string;
  name: string;
  createdAt: string;
  source: VariantSnapshotSource;
  modelName: string;
  filename: string | null;
  surfaceCount: number;
  variableSummary: string;
  firstOrder: Record<string, number | null>;
  metrics: CockpitMetric[];
  risks: CockpitRisk[];
  surfaces: SurfaceRow[];
}

export interface CockpitCounts {
  surfaceCount: number;
  poweredSurfaces: number;
  wavelengthCount: number;
  fieldCount: number;
  complexity: number;
  maxSemiDiameter: number;
  maxField: number;
}

export interface SensorAssumptions {
  name: string;
  pixelPitchUm: number;
  quantumEfficiency: number;
  readNoiseE: number;
  darkNoiseE: number;
  fullWellE: number;
  exposureMs: number;
  sceneLuminanceCdM2: number;
  opticalTransmission: number;
  microlensCraLimitDeg: number;
  referenceWavelengthNm: number;
}

export type SensorPatch = Partial<SensorAssumptions>;

export interface AnalysisSummary extends ModelResponse {
  firstOrder: Record<string, number | null>;
  counts: CockpitCounts;
  metrics: CockpitMetric[];
  risks: CockpitRisk[];
  sensor: SensorAssumptions;
}

export interface ExampleFile {
  label: string;
  path: string;
  kind: string;
}

export interface PatentDbStatus {
  path: string;
  exists: boolean;
  summary: {
    companies: number;
    lenses: number;
    simulationResults: number;
    camerae2eReady: number;
  } | null;
}

export interface PatentCompany {
  company: string;
  companySlug: string;
  simulationResults: number;
  camerae2eReady: number;
  partial: number;
  metadataOnly: number;
}

export interface PatentSearchResult {
  simulationId: string;
  lensId: string;
  company: string;
  companySlug: string;
  publicationNumber: string;
  exampleLabel: string;
  configuration: string;
  readiness: string;
  simulationStatus: string;
  simulationModel: string;
  focalLengthMm: number | null;
  fNumber: number | null;
  imageHeightMm: number | null;
  halfFieldDeg: number | null;
  fieldOfViewDeg: number | null;
  surfaceCount: number;
  asphereCount: number;
  notes: string[];
}

export interface ExampleCheckStage {
  name: string;
  status: KpiStatus;
  detail: string;
}

export interface ExampleCheckResult {
  label: string;
  path: string;
  kind: string;
  status: KpiStatus;
  durationMs: number;
  modelName: string | null;
  surfaceCount: number | null;
  fieldCount: number | null;
  wavelengthCount: number | null;
  stages: ExampleCheckStage[];
  warnings: string[];
  errors: string[];
}

export interface ExampleCheckResponse {
  checkedAt: string;
  total: number;
  passed: number;
  warned: number;
  failed: number;
  checks: ExampleCheckResult[];
}

export interface LayoutSurface {
  index: number;
  label: string;
  z: number;
  semiDiameter: number;
  radius: number | null;
  mode: string;
  isStop: boolean;
}

export interface LayoutRayPoint {
  z: number;
  y: number;
}

export interface LayoutPayload {
  surfaces: LayoutSurface[];
  rays: LayoutRayPoint[][];
  warnings: string[];
}

export type AnalysisTab = "layout" | "ray-fan" | "opd-fan" | "spot" | "wavefront" | "field-curves" | "first-order";
