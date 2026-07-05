import {
  defaultRequirementProfile,
  requirementProfiles,
  requirementsForProfile,
  sanitizeRequirementTarget,
  type ProjectRequirementKey,
  type ProjectRequirements,
  type RequirementProfileId,
  type RequirementProfileSelection
} from "./requirements";
import type {
  AnalysisScale,
  AnalysisTab,
  CockpitMetric,
  CockpitRisk,
  KpiStatus,
  MetricSource,
  QuickOptimizeObjective,
  QuickOptimizeResult,
  QuickOptimizeWeights,
  RecentModelFile,
  SensorAssumptions,
  SurfaceRow,
  ToleranceSweepResult,
  ToleranceSweepScope,
  VariantSnapshot,
  VariantSnapshotSource
} from "./types";
import { quickOptimizeWeightPresets, sanitizeQuickOptimizeWeights } from "./quickOptimize";

const STORAGE_KEY = "rayoptics.workbench.v1";
const RECENT_FILES_KEY = "rayoptics.recentFiles.v1";
export const WORKBENCH_METADATA_VERSION = 1;
const STORAGE_VERSION = WORKBENCH_METADATA_VERSION;
const MAX_PERSISTED_VARIANTS = 12;
const MAX_RECENT_FILES = 10;
const profileIds = new Set(requirementProfiles.map((profile) => profile.id));
const analysisTabs = new Set<AnalysisTab>(["layout", "ray-fan", "opd-fan", "spot", "wavefront", "field-curves", "first-order"]);
const analysisScales = new Set<AnalysisScale>(["same", "all"]);
const quickOptimizeObjectives = new Set<QuickOptimizeObjective>(["balanced", "spot", "distortion", "throughput", "cra", "custom"]);
const kpiStatuses = new Set<KpiStatus>(["pass", "warn", "fail"]);
const metricSources = new Set<MetricSource>(["computed", "model", "proxy", "diagnostic", "unsupported", "assumption"]);
const variantSources = new Set<VariantSnapshotSource>(["manual", "quick-optimize-before", "quick-optimize-after", "open", "new"]);
const toleranceSweepScopes = new Set<ToleranceSweepScope>(["powered", "variables"]);

export interface PersistedWorkbench {
  version: number;
  modelId: string | null;
  autosavePath: string | null;
  autosaveSavedAt: string | null;
  draftCount: number | null;
  surfaceVariables: Record<string, string>;
  selectedSurface: number | null;
  activeTab: AnalysisTab;
  radiusDisplay: "radius" | "curvature";
  sampling: number;
  scale: AnalysisScale;
  analysisFieldIndex: number | null;
  analysisWavelengthIndex: number | null;
  quickOptimizeObjective: QuickOptimizeObjective;
  quickOptimizeIterations: number;
  quickOptimizeMaxEvaluations: number;
  quickOptimizeStepScale: number;
  quickOptimizeWeights: QuickOptimizeWeights;
  quickOptimizeResult: QuickOptimizeResult | null;
  toleranceSweepScope: ToleranceSweepScope;
  toleranceSweepPerturbationPct: number;
  toleranceSweepMaxSurfaces: number;
  toleranceSweepResult: ToleranceSweepResult | null;
  sensorAssumptions: SensorAssumptions | null;
  variants: VariantSnapshot[];
  requirementProfile: RequirementProfileSelection;
  requirementProfileBase: RequirementProfileId;
  requirements: ProjectRequirements;
}

export type PersistableWorkbenchState = Omit<PersistedWorkbench, "version">;

export function loadPersistedWorkbench(): PersistedWorkbench | null {
  const storage = browserStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return sanitizePersistedWorkbenchPayload(JSON.parse(raw));
  } catch {
    return null;
  }
}

export function sanitizePersistedWorkbenchPayload(value: unknown): PersistedWorkbench | null {
  if (!value || typeof value !== "object") return null;
  const parsed = value as Partial<PersistedWorkbench>;
  if (parsed.version !== STORAGE_VERSION) return null;
  const baseProfile = validProfileId(parsed.requirementProfileBase) ? parsed.requirementProfileBase : defaultRequirementProfile;
  const profile = parsed.requirementProfile === "custom" || validProfileId(parsed.requirementProfile) ? parsed.requirementProfile : baseProfile;
  const objective = quickOptimizeObjectives.has(parsed.quickOptimizeObjective as QuickOptimizeObjective)
    ? (parsed.quickOptimizeObjective as QuickOptimizeObjective)
    : "balanced";
  const requirements = sanitizeRequirements(parsed.requirements, baseProfile);
  return {
    version: STORAGE_VERSION,
    modelId: typeof parsed.modelId === "string" && parsed.modelId ? parsed.modelId : null,
    autosavePath: typeof parsed.autosavePath === "string" && parsed.autosavePath ? parsed.autosavePath : null,
    autosaveSavedAt: typeof parsed.autosaveSavedAt === "string" && parsed.autosaveSavedAt ? parsed.autosaveSavedAt : null,
    draftCount: nullableInteger(parsed.draftCount),
    surfaceVariables: sanitizeSurfaceVariables(parsed.surfaceVariables),
    selectedSurface: nullableInteger(parsed.selectedSurface),
    activeTab: analysisTabs.has(parsed.activeTab as AnalysisTab) ? (parsed.activeTab as AnalysisTab) : "layout",
    radiusDisplay: parsed.radiusDisplay === "curvature" ? "curvature" : "radius",
    sampling: clampInteger(parsed.sampling, 5, 65, 21),
    scale: analysisScales.has(parsed.scale as AnalysisScale) ? (parsed.scale as AnalysisScale) : "same",
    analysisFieldIndex: nullableInteger(parsed.analysisFieldIndex),
    analysisWavelengthIndex: nullableInteger(parsed.analysisWavelengthIndex),
    quickOptimizeObjective: objective,
    quickOptimizeIterations: clampInteger(parsed.quickOptimizeIterations, 1, 5, 2),
    quickOptimizeMaxEvaluations: clampInteger(parsed.quickOptimizeMaxEvaluations, 4, 160, 48),
    quickOptimizeStepScale: clampNumber(parsed.quickOptimizeStepScale, 0.1, 5, 1),
    quickOptimizeWeights: sanitizeQuickOptimizeWeights(parsed.quickOptimizeWeights, quickOptimizeWeightPresets[objective]),
    quickOptimizeResult: sanitizeQuickOptimizeResult(parsed.quickOptimizeResult),
    toleranceSweepScope: toleranceSweepScopes.has(parsed.toleranceSweepScope as ToleranceSweepScope) ? (parsed.toleranceSweepScope as ToleranceSweepScope) : "powered",
    toleranceSweepPerturbationPct: clampNumber(parsed.toleranceSweepPerturbationPct, 0.05, 5, 0.5),
    toleranceSweepMaxSurfaces: clampInteger(parsed.toleranceSweepMaxSurfaces, 1, 20, 6),
    toleranceSweepResult: sanitizeToleranceSweepResult(parsed.toleranceSweepResult),
    sensorAssumptions: sanitizeSensorAssumptions(parsed.sensorAssumptions),
    variants: sanitizeVariants(parsed.variants),
    requirementProfile: profile,
    requirementProfileBase: baseProfile,
    requirements
  };
}

export function savePersistedWorkbench(state: PersistableWorkbenchState) {
  const storage = browserStorage();
  if (!storage) return;
  const payload: PersistedWorkbench = {
    version: STORAGE_VERSION,
    ...state
  };
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore storage quota and privacy-mode failures; runtime state remains authoritative.
  }
}

export function loadRecentFiles(): RecentModelFile[] {
  const storage = browserStorage();
  if (!storage) return [];
  try {
    const raw = storage.getItem(RECENT_FILES_KEY);
    if (!raw) return [];
    return sanitizeRecentFiles(JSON.parse(raw));
  } catch {
    return [];
  }
}

export function saveRecentFiles(files: RecentModelFile[]) {
  const storage = browserStorage();
  if (!storage) return;
  try {
    storage.setItem(RECENT_FILES_KEY, JSON.stringify(sanitizeRecentFiles(files)));
  } catch {
    // Recent files are a convenience cache; ignore storage failures.
  }
}

export function recordRecentFile(files: RecentModelFile[], entry: Omit<RecentModelFile, "label" | "touchedAt"> & { label?: string; touchedAt?: string }) {
  const path = entry.path.trim();
  if (!path) return sanitizeRecentFiles(files);
  const nextEntry: RecentModelFile = {
    path,
    label: entry.label?.trim() || path.split("/").pop() || path,
    kind: entry.kind,
    touchedAt: entry.touchedAt ?? new Date().toISOString(),
    hasWorkbench: entry.hasWorkbench
  };
  return sanitizeRecentFiles([nextEntry, ...files.filter((item) => item.path !== path)]);
}

function sanitizeSurfaceVariables(value: unknown) {
  const result: Record<string, string> = {};
  if (!value || typeof value !== "object") return result;
  for (const [key, rawTokens] of Object.entries(value as Record<string, unknown>)) {
    const index = Number(key);
    if (!Number.isInteger(index) || index <= 0) continue;
    const tokens = String(rawTokens)
      .split(",")
      .map((token) => token.trim().toUpperCase())
      .filter((token) => ["R", "T", "SD", "K"].includes(token));
    if (tokens.length) result[String(index)] = ["R", "T", "SD", "K"].filter((token) => tokens.includes(token)).join(",");
  }
  return result;
}

function sanitizeRecentFiles(value: unknown): RecentModelFile[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  return value.slice(0, MAX_RECENT_FILES * 2).flatMap((raw): RecentModelFile[] => {
    if (!raw || typeof raw !== "object") return [];
    const item = raw as Partial<RecentModelFile>;
    const path = typeof item.path === "string" ? item.path.trim() : "";
    if (!path || seen.has(path) || !/\.(roa|seq|zmx)$/i.test(path)) return [];
    seen.add(path);
    return [
      {
        path,
        label: typeof item.label === "string" && item.label.trim() ? item.label.trim().slice(0, 120) : path.split("/").pop() || path,
        kind: item.kind === "save" || item.kind === "example" ? item.kind : "open",
        touchedAt: typeof item.touchedAt === "string" && item.touchedAt ? item.touchedAt : new Date().toISOString(),
        hasWorkbench: Boolean(item.hasWorkbench)
      }
    ];
  }).slice(0, MAX_RECENT_FILES);
}

function sanitizeQuickOptimizeResult(value: unknown): QuickOptimizeResult | null {
  if (!value || typeof value !== "object") return null;
  const item = value as Partial<QuickOptimizeResult>;
  if (!["improved", "no-change", "failed"].includes(String(item.status))) return null;
  const objective = quickOptimizeObjectives.has(item.objective as QuickOptimizeObjective) ? (item.objective as QuickOptimizeObjective) : "balanced";
  return {
    status: item.status as QuickOptimizeResult["status"],
    objective,
    message: typeof item.message === "string" ? item.message : "",
    baselineScore: finiteNumber(item.baselineScore, 0),
    finalScore: finiteNumber(item.finalScore, 0),
    improvement: finiteNumber(item.improvement, 0),
    evaluations: clampInteger(item.evaluations, 0, 100000, 0),
    iterations: clampInteger(item.iterations, 0, 100000, 0),
    variableCount: clampInteger(item.variableCount, 0, 100000, 0),
    operandWeights: sanitizeQuickOptimizeWeights(item.operandWeights, quickOptimizeWeightPresets[objective]),
    variables: Array.isArray(item.variables) ? item.variables.map((variable) => String(variable)).slice(0, 200) : [],
    moves: Array.isArray(item.moves)
      ? item.moves.slice(0, 200).flatMap((move) => {
          if (!move || typeof move !== "object") return [];
          const entry = move as Partial<QuickOptimizeResult["moves"][number]>;
          return [
            {
              surfaceIndex: clampInteger(entry.surfaceIndex, 0, 10000, 0),
              token: typeof entry.token === "string" ? entry.token : "",
              label: typeof entry.label === "string" ? entry.label : "",
              before: finiteNumber(entry.before, 0),
              after: finiteNumber(entry.after, 0),
              score: finiteNumber(entry.score, 0)
            }
          ];
        })
      : []
  };
}

function sanitizeToleranceSweepResult(value: unknown): ToleranceSweepResult | null {
  if (!value || typeof value !== "object") return null;
  const item = value as Partial<ToleranceSweepResult>;
  if (!kpiStatuses.has(item.status as KpiStatus)) return null;
  const cases = Array.isArray(item.cases)
    ? item.cases.slice(0, 80).flatMap((testCase) => {
        if (!testCase || typeof testCase !== "object") return [];
        const entry = testCase as Partial<ToleranceSweepResult["cases"][number]>;
        return [
          {
            label: typeof entry.label === "string" ? entry.label : "",
            surfaceIndex: clampInteger(entry.surfaceIndex, 0, 10000, 0),
            token: typeof entry.token === "string" ? entry.token : "",
            perturbationPct: finiteNumber(entry.perturbationPct, 0),
            before: finiteNumber(entry.before, 0),
            after: finiteNumber(entry.after, 0),
            score: clampNumber(entry.score, 0, 1, 0),
            status: kpiStatuses.has(entry.status as KpiStatus) ? (entry.status as KpiStatus) : "fail",
            spotRmsUm: nullableNumber(entry.spotRmsUm),
            mtf50LpMm: nullableNumber(entry.mtf50LpMm),
            throughput: nullableNumber(entry.throughput),
            distortionPct: nullableNumber(entry.distortionPct),
            craDeg: nullableNumber(entry.craDeg),
            traceFailures: Array.isArray(entry.traceFailures) ? entry.traceFailures.map((failure) => String(failure)).slice(0, 20) : []
          }
        ];
      })
    : [];
  const scope = toleranceSweepScopes.has(item.scope as ToleranceSweepScope) ? (item.scope as ToleranceSweepScope) : "powered";
  return {
    status: item.status as KpiStatus,
    scope,
    perturbationPct: clampNumber(item.perturbationPct, 0.05, 5, 0.5),
    baselineScore: clampNumber(item.baselineScore, 0, 1, 0),
    attemptedCases: clampInteger(item.attemptedCases, 0, 100000, cases.length),
    passedCases: clampInteger(item.passedCases, 0, 100000, 0),
    warnedCases: clampInteger(item.warnedCases, 0, 100000, 0),
    failedCases: clampInteger(item.failedCases, 0, 100000, 0),
    worstCase: typeof item.worstCase === "string" ? item.worstCase : null,
    worstScore: nullableNumber(item.worstScore),
    cases,
    warnings: Array.isArray(item.warnings) ? item.warnings.map((warning) => String(warning)).slice(0, 20) : []
  };
}

function sanitizeVariants(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, MAX_PERSISTED_VARIANTS).flatMap((raw) => {
    if (!raw || typeof raw !== "object") return [];
    const item = raw as Partial<VariantSnapshot>;
    if (!item.id || !item.name || !item.createdAt) return [];
    const surfaces = sanitizeSurfaces(item.surfaces);
    return [
      {
        id: String(item.id),
        name: String(item.name).slice(0, 80),
        createdAt: String(item.createdAt),
        source: variantSources.has(item.source as VariantSnapshotSource) ? (item.source as VariantSnapshotSource) : "manual",
        modelName: typeof item.modelName === "string" && item.modelName ? item.modelName : "Optical Design Cockpit",
        filename: typeof item.filename === "string" && item.filename ? item.filename : null,
        surfaceCount: clampInteger(item.surfaceCount, 0, 500, surfaces.length),
        variableSummary: typeof item.variableSummary === "string" && item.variableSummary ? item.variableSummary : "No variables",
        firstOrder: sanitizeFirstOrder(item.firstOrder),
        metrics: sanitizeMetrics(item.metrics),
        risks: sanitizeRisks(item.risks),
        surfaces
      }
    ];
  });
}

function sanitizeMetrics(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.flatMap((raw): CockpitMetric[] => {
    if (!raw || typeof raw !== "object") return [];
    const item = raw as Partial<CockpitMetric>;
    if (!item.key || !item.label) return [];
    return [
      {
        key: String(item.key),
        label: String(item.label),
        value: String(item.value ?? "n/a"),
        unit: typeof item.unit === "string" ? item.unit : null,
        target: String(item.target ?? "n/a"),
        status: kpiStatuses.has(item.status as KpiStatus) ? (item.status as KpiStatus) : "warn",
        score: clampNumber(item.score, 0, 1, 0),
        source: metricSources.has(item.source as MetricSource) ? (item.source as MetricSource) : "proxy",
        note: typeof item.note === "string" ? item.note : null
      }
    ];
  });
}

function sanitizeRisks(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.flatMap((raw): CockpitRisk[] => {
    if (!raw || typeof raw !== "object") return [];
    const item = raw as Partial<CockpitRisk>;
    if (!item.label) return [];
    return [
      {
        label: String(item.label),
        value: String(item.value ?? "n/a"),
        status: kpiStatuses.has(item.status as KpiStatus) ? (item.status as KpiStatus) : "warn",
        detail: String(item.detail ?? ""),
        source: metricSources.has(item.source as MetricSource) ? (item.source as MetricSource) : "proxy"
      }
    ];
  });
}

function sanitizeSurfaces(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.flatMap((raw): SurfaceRow[] => {
    if (!raw || typeof raw !== "object") return [];
    const item = raw as Partial<SurfaceRow>;
    const index = Number(item.index);
    if (!Number.isInteger(index) || index < 0) return [];
    return [
      {
        index,
        label: typeof item.label === "string" ? item.label : "",
        type: typeof item.type === "string" ? item.type : "surface",
        radius: nullableNumber(item.radius),
        curvature: Number.isFinite(Number(item.curvature)) ? Number(item.curvature) : 0,
        thickness: nullableNumber(item.thickness),
        glass: typeof item.glass === "string" ? item.glass : "",
        catalog: typeof item.catalog === "string" ? item.catalog : "",
        semiDiameter: Number.isFinite(Number(item.semiDiameter)) ? Number(item.semiDiameter) : 0,
        conic: nullableNumber(item.conic),
        mode: ["transmit", "reflect", "dummy", "phantom"].includes(String(item.mode)) ? (item.mode as SurfaceRow["mode"]) : "transmit",
        isStop: Boolean(item.isStop),
        variable: typeof item.variable === "string" ? item.variable : ""
      }
    ];
  });
}

function sanitizeFirstOrder(value: unknown) {
  const result: Record<string, number | null> = {};
  if (!value || typeof value !== "object") return result;
  for (const [key, rawValue] of Object.entries(value as Record<string, unknown>)) {
    result[key] = nullableNumber(rawValue);
  }
  return result;
}

function sanitizeSensorAssumptions(value: unknown): SensorAssumptions | null {
  if (!value || typeof value !== "object") return null;
  const item = value as Partial<SensorAssumptions>;
  return {
    name: typeof item.name === "string" && item.name ? item.name.slice(0, 80) : "Assumed CMOS reference",
    pixelPitchUm: clampNumber(item.pixelPitchUm, 0.2, 30, 2),
    quantumEfficiency: clampNumber(item.quantumEfficiency, 0.01, 1, 0.55),
    readNoiseE: clampNumber(item.readNoiseE, 0, 100, 1.2),
    darkNoiseE: clampNumber(item.darkNoiseE, 0, 1000, 0.5),
    fullWellE: clampNumber(item.fullWellE, 100, 1.0e7, 18000),
    exposureMs: clampNumber(item.exposureMs, 0.001, 60000, 10),
    sceneLuminanceCdM2: clampNumber(item.sceneLuminanceCdM2, 0.001, 1.0e6, 50),
    opticalTransmission: clampNumber(item.opticalTransmission, 0.01, 1, 0.85),
    microlensCraLimitDeg: clampNumber(item.microlensCraLimitDeg, 0.1, 90, 24),
    referenceWavelengthNm: clampNumber(item.referenceWavelengthNm, 100, 30000, 550)
  };
}

function sanitizeRequirements(requirements: unknown, baseProfile: RequirementProfileId): ProjectRequirements {
  const fallback = requirementsForProfile(baseProfile);
  if (!requirements || typeof requirements !== "object") return fallback;
  const candidate = requirements as Partial<Record<ProjectRequirementKey, unknown>>;
  let current = fallback;
  for (const key of Object.keys(fallback) as ProjectRequirementKey[]) {
    const rawValue = Number(candidate[key]);
    current = {
      ...current,
      [key]: sanitizeRequirementTarget(key, rawValue, current)
    };
  }
  return current;
}

function validProfileId(value: unknown): value is RequirementProfileId {
  return typeof value === "string" && profileIds.has(value as RequirementProfileId);
}

function nullableInteger(value: unknown) {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
}

function nullableNumber(value: unknown) {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function finiteNumber(value: unknown, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clampInteger(value: unknown, min: number, max: number, fallback: number) {
  const numeric = Number(value);
  return Number.isInteger(numeric) ? Math.min(max, Math.max(min, numeric)) : fallback;
}

function clampNumber(value: unknown, min: number, max: number, fallback: number) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.min(max, Math.max(min, numeric)) : fallback;
}

function browserStorage() {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}
