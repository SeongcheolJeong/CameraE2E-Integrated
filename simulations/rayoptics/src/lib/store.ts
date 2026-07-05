import { create } from "zustand";
import type {
  AnalysisScale,
  AnalysisSummary,
  AnalysisTab,
  ExampleCheckResponse,
  ExampleFile,
  LayoutPayload,
  ModelResponse,
  NewModelSpec,
  OpticalModel,
  QuickOptimizeObjective,
  QuickOptimizeResult,
  QuickOptimizeOperandKey,
  QuickOptimizeWeights,
  RecentModelFile,
  SensorPatch,
  SystemPatch,
  ToleranceSweepResult,
  ToleranceSweepScope,
  UpdateMode,
  WorkbenchFileStatus,
  WorkbenchMetadata,
  VariantSnapshot,
  VariantSnapshotSource
} from "./types";
import { api } from "./api";
import {
  defaultRequirementProfile,
  requirementsForProfile,
  sanitizeRequirementTarget,
  type ProjectRequirementKey,
  type ProjectRequirements,
  type RequirementProfileId,
  type RequirementProfileSelection
} from "./requirements";
import {
  loadRecentFiles,
  loadPersistedWorkbench,
  recordRecentFile,
  sanitizePersistedWorkbenchPayload,
  saveRecentFiles,
  savePersistedWorkbench,
  WORKBENCH_METADATA_VERSION,
  type PersistableWorkbenchState,
  type PersistedWorkbench
} from "./workspacePersistence";
import { deriveMetrics } from "./uxMetrics";
import {
  quickOptimizeObjectiveLabel,
  quickOptimizeTargetsFromRequirements,
  quickOptimizeWeightPresets,
  sanitizeQuickOptimizeWeights,
  weightsForQuickOptimizeObjective
} from "./quickOptimize";

const MAX_VARIANT_SNAPSHOTS = 12;

interface WorkbenchState {
  model: OpticalModel | null;
  summary: AnalysisSummary | null;
  examples: ExampleFile[];
  compatibility: ExampleCheckResponse | null;
  quickOptimizeResult: QuickOptimizeResult | null;
  quickOptimizeObjective: QuickOptimizeObjective;
  quickOptimizeIterations: number;
  quickOptimizeMaxEvaluations: number;
  quickOptimizeStepScale: number;
  quickOptimizeWeights: QuickOptimizeWeights;
  toleranceSweepResult: ToleranceSweepResult | null;
  toleranceSweepScope: ToleranceSweepScope;
  toleranceSweepPerturbationPct: number;
  toleranceSweepMaxSurfaces: number;
  variants: VariantSnapshot[];
  recentFiles: RecentModelFile[];
  workbenchFileStatus: WorkbenchFileStatus | null;
  warnings: string[];
  errors: string[];
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  analysisStale: boolean;
  selectedSurface: number | null;
  updateMode: UpdateMode;
  activeTab: AnalysisTab;
  radiusDisplay: "radius" | "curvature";
  requirements: ProjectRequirements;
  requirementProfile: RequirementProfileSelection;
  requirementProfileBase: RequirementProfileId;
  autosavePath: string | null;
  autosaveSavedAt: string | null;
  draftCount: number | null;
  layout: LayoutPayload | null;
  svg: string | null;
  firstOrder: Record<string, number | null>;
  sampling: number;
  scale: AnalysisScale;
  analysisFieldIndex: number | null;
  analysisWavelengthIndex: number | null;
  isBusy: boolean;
  isCheckingExamples: boolean;
  setActiveTab: (tab: AnalysisTab) => void;
  setSelectedSurface: (index: number | null) => void;
  setUpdateMode: (mode: UpdateMode) => Promise<void>;
  setRadiusDisplay: (mode: "radius" | "curvature") => void;
  setRequirementTarget: (key: ProjectRequirementKey, value: number) => void;
  setRequirementProfile: (profile: RequirementProfileId) => void;
  resetRequirements: () => void;
  setSampling: (sampling: number) => void;
  setScale: (scale: AnalysisScale) => void;
  setAnalysisFieldIndex: (index: number | null) => void;
  setAnalysisWavelengthIndex: (index: number | null) => void;
  bootstrap: () => Promise<void>;
  newModel: (spec?: NewModelSpec) => Promise<void>;
  openModel: (path: string) => Promise<boolean>;
  openPatent: (simulationId: string) => Promise<boolean>;
  saveModel: (path?: string, overwrite?: boolean) => Promise<boolean>;
  refresh: () => Promise<void>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  patchSurface: (surfaceIndex: number, patch: Record<string, unknown>) => Promise<void>;
  patchSystem: (patch: SystemPatch) => Promise<void>;
  patchSensor: (patch: SensorPatch) => Promise<void>;
  insertSurface: (after: number) => Promise<void>;
  deleteSurface: (surfaceIndex: number) => Promise<void>;
  refreshAnalysis: () => Promise<void>;
  runExampleChecks: () => Promise<void>;
  setQuickOptimizeObjective: (objective: QuickOptimizeObjective) => void;
  setQuickOptimizeIterations: (iterations: number) => void;
  setQuickOptimizeMaxEvaluations: (evaluations: number) => void;
  setQuickOptimizeStepScale: (scale: number) => void;
  setQuickOptimizeWeight: (key: QuickOptimizeOperandKey, weight: number) => void;
  resetQuickOptimizeWeights: () => void;
  setToleranceSweepScope: (scope: ToleranceSweepScope) => void;
  setToleranceSweepPerturbationPct: (value: number) => void;
  setToleranceSweepMaxSurfaces: (value: number) => void;
  runToleranceSweep: () => Promise<void>;
  captureVariant: (name?: string, source?: VariantSnapshotSource) => void;
  deleteVariant: (id: string) => void;
  clearVariants: () => void;
  quickOptimize: () => Promise<void>;
}

function applyResponse(
  set: (partial: Partial<WorkbenchState>) => void,
  response: ModelResponse,
  options: { preserveSummary?: boolean; analysisStale?: boolean } = {}
) {
  const partial: Partial<WorkbenchState> = {
    model: response.model,
    warnings: response.warnings,
    errors: response.errors,
    dirty: response.dirty,
    canUndo: response.canUndo,
    canRedo: response.canRedo,
    updateMode: response.model.updateMode,
    analysisStale: options.analysisStale ?? false
  };
  if (!options.preserveSummary) {
    partial.summary = null;
  }
  set(partial);
}

function applySummary(set: (partial: Partial<WorkbenchState>) => void, summary: AnalysisSummary) {
  set({
    summary,
    model: summary.model,
    warnings: summary.warnings,
    errors: summary.errors,
    dirty: summary.dirty,
    canUndo: summary.canUndo,
    canRedo: summary.canRedo,
    updateMode: summary.model.updateMode,
    analysisStale: false,
    firstOrder: summary.firstOrder
  });
}

async function refreshSummary(get: () => WorkbenchState, set: (partial: Partial<WorkbenchState>) => void) {
  const model = get().model;
  if (!model) return;
  const summary = await api.analysisSummary(model.id);
  applySummary(set, summary);
}

async function updateViews(get: () => WorkbenchState, set: (partial: Partial<WorkbenchState>) => void, force = false) {
  const { model, activeTab, updateMode, sampling, scale, analysisFieldIndex, analysisWavelengthIndex } = get();
  if (!model) return;
  if (!force && (updateMode === "manual" || updateMode === "editors-only")) return;
  if (activeTab === "layout" || updateMode === "layout-only" || force) {
    const layout = await api.layout(model.id);
    set({ layout, svg: null });
    if (activeTab === "layout" || updateMode === "layout-only") return;
  }
  if (activeTab === "first-order") {
    const payload = await api.firstOrder(model.id);
    set({ firstOrder: payload.values, warnings: [...get().warnings, ...payload.warnings] });
    return;
  }
  const fieldIndex = analysisFieldIndex !== null && model.system.field.fields[analysisFieldIndex] ? analysisFieldIndex : null;
  const wavelengthIndex = analysisWavelengthIndex !== null && model.system.wavelengths.values[analysisWavelengthIndex] !== undefined ? analysisWavelengthIndex : null;
  const svg = await api.analysisSvg(model.id, activeTab, sampling, scale, fieldIndex, wavelengthIndex);
  set({ svg });
}

async function updateViewsSafely(get: () => WorkbenchState, set: (partial: Partial<WorkbenchState>) => void, force = false) {
  try {
    await updateViews(get, set, force);
  } catch (error) {
    set({
      svg: null,
      errors: [error instanceof Error ? error.message : String(error)]
    });
  }
}

function defersAnalysis(mode: UpdateMode) {
  return mode === "manual" || mode === "layout-only" || mode === "editors-only";
}

function clampedAnalysisScope(state: WorkbenchState, model: OpticalModel) {
  return {
    analysisFieldIndex:
      state.analysisFieldIndex !== null && model.system.field.fields[state.analysisFieldIndex] ? state.analysisFieldIndex : null,
    analysisWavelengthIndex:
      state.analysisWavelengthIndex !== null && model.system.wavelengths.values[state.analysisWavelengthIndex] !== undefined
        ? state.analysisWavelengthIndex
        : null
  };
}

async function applyMutationResponse(
  get: () => WorkbenchState,
  set: (partial: Partial<WorkbenchState>) => void,
  response: ModelResponse,
  selection?: number | null
) {
  const mode = get().updateMode;
  const stale = defersAnalysis(mode) && response.dirty;
  applyResponse(set, response, { preserveSummary: defersAnalysis(mode), analysisStale: stale });
  if (selection !== undefined) {
    set({ selectedSurface: clampSurfaceSelection(selection, response.model.surfaces.length) });
  }
  set(clampedAnalysisScope(get(), response.model));
  if (mode === "auto") {
    await refreshSummary(get, set);
    await updateViews(get, set);
    await autosaveDraft(get, set);
    persistCurrentState(get);
    return;
  }
  if (mode === "layout-only") {
    await updateViews(get, set);
  }
  await autosaveDraft(get, set);
  persistCurrentState(get);
}

function clampSurfaceSelection(selected: number | null, surfaceCount: number) {
  if (selected === null) return surfaceCount > 1 ? 1 : null;
  return Math.max(0, Math.min(selected, Math.max(0, surfaceCount - 1)));
}

function persistCurrentState(get: () => WorkbenchState) {
  savePersistedWorkbench(persistableWorkbenchState(get()));
}

function persistableWorkbenchState(state: WorkbenchState): PersistableWorkbenchState {
  return {
    modelId: state.model?.id ?? null,
    autosavePath: state.autosavePath,
    autosaveSavedAt: state.autosaveSavedAt,
    draftCount: state.draftCount,
    surfaceVariables: surfaceVariablesFromModel(state.model),
    selectedSurface: state.selectedSurface,
    activeTab: state.activeTab,
    radiusDisplay: state.radiusDisplay,
    sampling: state.sampling,
    scale: state.scale,
    analysisFieldIndex: state.analysisFieldIndex,
    analysisWavelengthIndex: state.analysisWavelengthIndex,
    quickOptimizeObjective: state.quickOptimizeObjective,
    quickOptimizeIterations: state.quickOptimizeIterations,
    quickOptimizeMaxEvaluations: state.quickOptimizeMaxEvaluations,
    quickOptimizeStepScale: state.quickOptimizeStepScale,
    quickOptimizeWeights: state.quickOptimizeWeights,
    quickOptimizeResult: state.quickOptimizeResult,
    toleranceSweepScope: state.toleranceSweepScope,
    toleranceSweepPerturbationPct: state.toleranceSweepPerturbationPct,
    toleranceSweepMaxSurfaces: state.toleranceSweepMaxSurfaces,
    toleranceSweepResult: state.toleranceSweepResult,
    sensorAssumptions: state.summary?.sensor ?? null,
    variants: state.variants,
    requirementProfile: state.requirementProfile,
    requirementProfileBase: state.requirementProfileBase,
    requirements: state.requirements
  };
}

function workbenchMetadataFromState(state: WorkbenchState): WorkbenchMetadata {
  return {
    version: WORKBENCH_METADATA_VERSION,
    ...persistableWorkbenchState(state),
    modelId: null,
    autosavePath: null,
    autosaveSavedAt: null,
    draftCount: null
  };
}

function surfaceVariablesFromModel(model: OpticalModel | null) {
  const variables: Record<string, string> = {};
  for (const surface of model?.surfaces ?? []) {
    if (surface.variable.trim()) {
      variables[String(surface.index)] = surface.variable;
    }
  }
  return variables;
}

async function restorePersistedSurfaceVariables(response: ModelResponse, variables: Record<string, string> | undefined) {
  if (!variables || !Object.keys(variables).length) return response;
  let current = response;
  for (const [rawIndex, variable] of Object.entries(variables)) {
    const surfaceIndex = Number(rawIndex);
    const surface = current.model.surfaces.find((item) => item.index === surfaceIndex);
    if (!surface || surface.index === 0 || surface.index === current.model.surfaces.length - 1) continue;
    if (surface.variable === variable) continue;
    current = await api.patchSurface(current.model.id, surfaceIndex, { variable } as Partial<OpticalModel["surfaces"][number]>);
  }
  return current;
}

function sidecarWorkbench(response: ModelResponse): PersistedWorkbench | null {
  return sanitizePersistedWorkbenchPayload(response.workbench);
}

function countSidecarPayload(workbench: PersistedWorkbench | null) {
  if (!workbench) return 0;
  return [
    Object.keys(workbench.surfaceVariables).length > 0,
    Boolean(workbench.quickOptimizeResult),
    Boolean(workbench.toleranceSweepResult),
    Boolean(workbench.sensorAssumptions),
    workbench.variants.length > 0,
    workbench.requirementProfile === "custom"
  ].filter(Boolean).length;
}

function setRecentFiles(set: (partial: Partial<WorkbenchState>) => void, files: RecentModelFile[]) {
  saveRecentFiles(files);
  set({ recentFiles: files });
}

function recordModelFile(
  get: () => WorkbenchState,
  set: (partial: Partial<WorkbenchState>) => void,
  path: string | null | undefined,
  kind: RecentModelFile["kind"],
  hasWorkbench: boolean
) {
  if (!path) return;
  const files = recordRecentFile(get().recentFiles, {
    path,
    kind,
    hasWorkbench
  });
  setRecentFiles(set, files);
}

function workbenchStatus(
  response: ModelResponse,
  state: WorkbenchFileStatus["state"],
  hasWorkbench: boolean,
  restoredItems = 0
): WorkbenchFileStatus {
  const path = response.model.filename;
  const basename = path?.split("/").pop() ?? response.model.name;
  const detail =
    state === "restored"
      ? `Restored ${restoredItems || "saved"} workbench state groups from sidecar.`
      : state === "saved"
        ? "Saved ROA model and workbench sidecar metadata."
        : state === "opened"
          ? "Opened optical model without workbench sidecar metadata."
          : "Started a new unsaved model.";
  return {
    state,
    label: basename,
    detail,
    path,
    at: new Date().toISOString(),
    hasWorkbench
  };
}

async function applySidecarWorkbench(
  get: () => WorkbenchState,
  set: (partial: Partial<WorkbenchState>) => void,
  response: ModelResponse
) {
  const workbench = sidecarWorkbench(response);
  if (!workbench) return response;
  let current = await restorePersistedSurfaceVariables(response, workbench.surfaceVariables);
  applyResponse(set, current, { preserveSummary: true, analysisStale: get().analysisStale });
  set({
    activeTab: workbench.activeTab,
    radiusDisplay: workbench.radiusDisplay,
    sampling: workbench.sampling,
    scale: workbench.scale,
    analysisFieldIndex: workbench.analysisFieldIndex,
    analysisWavelengthIndex: workbench.analysisWavelengthIndex,
    quickOptimizeObjective: workbench.quickOptimizeObjective,
    quickOptimizeIterations: workbench.quickOptimizeIterations,
    quickOptimizeMaxEvaluations: workbench.quickOptimizeMaxEvaluations,
    quickOptimizeStepScale: workbench.quickOptimizeStepScale,
    quickOptimizeWeights: workbench.quickOptimizeWeights,
    quickOptimizeResult: workbench.quickOptimizeResult,
    toleranceSweepScope: workbench.toleranceSweepScope,
    toleranceSweepPerturbationPct: workbench.toleranceSweepPerturbationPct,
    toleranceSweepMaxSurfaces: workbench.toleranceSweepMaxSurfaces,
    toleranceSweepResult: workbench.toleranceSweepResult,
    variants: workbench.variants,
    requirementProfile: workbench.requirementProfile,
    requirementProfileBase: workbench.requirementProfileBase,
    requirements: workbench.requirements,
    selectedSurface: clampSurfaceSelection(workbench.selectedSurface ?? current.model.stopSurface ?? 1, current.model.surfaces.length)
  });
  set(clampedAnalysisScope(get(), current.model));
  if (workbench.sensorAssumptions) {
    const summary = await api.patchSensor(current.model.id, workbench.sensorAssumptions);
    current = summary;
    applySummary(set, summary);
  }
  return current;
}

async function autosaveDraft(get: () => WorkbenchState, set: (partial: Partial<WorkbenchState>) => void) {
  const model = get().model;
  if (!model) return;
  try {
    const draft = await api.autosaveDraft(model.id, workbenchMetadataFromState(get()));
    set({ autosavePath: draft.path, autosaveSavedAt: draft.savedAt, draftCount: draft.draftCount });
    persistCurrentState(get);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    set({ warnings: [...get().warnings.filter((item) => !item.startsWith("Draft autosave failed:")), `Draft autosave failed: ${message}`] });
    persistCurrentState(get);
  }
}

function captureCurrentVariant(
  get: () => WorkbenchState,
  set: (partial: Partial<WorkbenchState>) => void,
  name?: string,
  source: VariantSnapshotSource = "manual"
) {
  const state = get();
  const model = state.model;
  if (!model) return null;
  const derived = deriveMetrics(model, state.warnings, state.errors, state.summary, state.requirements);
  const snapshot: VariantSnapshot = {
    id: `variant-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: normalizedVariantName(name, source, state.variants.length),
    createdAt: new Date().toISOString(),
    source,
    modelName: model.name || "Optical Design Cockpit",
    filename: model.filename,
    surfaceCount: model.surfaces.length,
    variableSummary: summarizeVariables(model),
    firstOrder: { ...(state.summary?.firstOrder ?? state.firstOrder) },
    metrics: derived.metrics.map((metric) => ({ ...metric })),
    risks: derived.risks.map((risk) => ({ ...risk })),
    surfaces: model.surfaces.map((surface) => ({ ...surface }))
  };
  set({ variants: [snapshot, ...state.variants].slice(0, MAX_VARIANT_SNAPSHOTS) });
  persistCurrentState(get);
  return snapshot;
}

function normalizedVariantName(name: string | undefined, source: VariantSnapshotSource, count: number) {
  const trimmed = name?.trim();
  if (trimmed) return trimmed.slice(0, 80);
  if (source === "quick-optimize-before") return "Before Quick Optimize";
  if (source === "quick-optimize-after") return "After Quick Optimize";
  if (source === "open") return "Opened Model";
  if (source === "new") return "New Model";
  return `Manual Snapshot ${count + 1}`;
}

function summarizeVariables(model: OpticalModel) {
  const variables = model.surfaces.filter((surface) => surface.variable.trim());
  if (!variables.length) return "No variables";
  return variables
    .slice(0, 6)
    .map((surface) => `S${surface.index}:${surface.variable}`)
    .join(" · ");
}

export const useWorkbench = create<WorkbenchState>((set, get) => ({
  model: null,
  summary: null,
  examples: [],
  compatibility: null,
  quickOptimizeResult: null,
  quickOptimizeObjective: "balanced",
  quickOptimizeIterations: 2,
  quickOptimizeMaxEvaluations: 48,
  quickOptimizeStepScale: 1,
  quickOptimizeWeights: quickOptimizeWeightPresets.balanced,
  toleranceSweepResult: null,
  toleranceSweepScope: "powered",
  toleranceSweepPerturbationPct: 0.5,
  toleranceSweepMaxSurfaces: 6,
  variants: [],
  recentFiles: [],
  workbenchFileStatus: null,
  warnings: [],
  errors: [],
  dirty: false,
  canUndo: false,
  canRedo: false,
  analysisStale: false,
  selectedSurface: null,
  updateMode: "auto",
  activeTab: "layout",
  radiusDisplay: "radius",
  requirements: requirementsForProfile(defaultRequirementProfile),
  requirementProfile: defaultRequirementProfile,
  requirementProfileBase: defaultRequirementProfile,
  autosavePath: null,
  autosaveSavedAt: null,
  draftCount: null,
  layout: null,
  svg: null,
  firstOrder: {},
  sampling: 21,
  scale: "same",
  analysisFieldIndex: null,
  analysisWavelengthIndex: null,
  isBusy: false,
  isCheckingExamples: false,
  setActiveTab: (tab) => {
    set({ activeTab: tab, svg: null });
    persistCurrentState(get);
    void updateViewsSafely(get, set);
  },
  setSelectedSurface: (index) => {
    set({ selectedSurface: index });
    persistCurrentState(get);
  },
  setUpdateMode: async (mode) => {
    const model = get().model;
    set({
      updateMode: mode,
      model: model ? { ...model, updateMode: mode } : model
    });
    if (!model) return;
    try {
      const response = await api.patchSettings(model.id, mode);
      applyResponse(set, response, { preserveSummary: true, analysisStale: get().analysisStale });
      if (mode === "auto" && get().analysisStale) {
        await refreshSummary(get, set);
        await updateViews(get, set, true);
      } else if (mode === "layout-only") {
        await updateViews(get, set, true);
      }
      persistCurrentState(get);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    }
  },
  setRadiusDisplay: (mode) => {
    set({ radiusDisplay: mode });
    persistCurrentState(get);
  },
  setRequirementTarget: (key, value) => {
    const requirements = get().requirements;
    set({
      requirements: {
        ...requirements,
        [key]: sanitizeRequirementTarget(key, value, requirements)
      },
      requirementProfile: "custom"
    });
    persistCurrentState(get);
  },
  setRequirementProfile: (profile) => {
    set({
      requirementProfile: profile,
      requirementProfileBase: profile,
      requirements: requirementsForProfile(profile)
    });
    persistCurrentState(get);
  },
  resetRequirements: () => {
    const profile = get().requirementProfileBase;
    set({
      requirementProfile: profile,
      requirements: requirementsForProfile(profile)
    });
    persistCurrentState(get);
  },
  setSampling: (sampling) => {
    set({ sampling: Math.max(5, Math.min(65, Math.trunc(sampling))), svg: null });
    persistCurrentState(get);
    void updateViewsSafely(get, set, true);
  },
  setScale: (scale) => {
    set({ scale, svg: null });
    persistCurrentState(get);
    void updateViewsSafely(get, set, true);
  },
  setAnalysisFieldIndex: (index) => {
    set({ analysisFieldIndex: index, svg: null });
    persistCurrentState(get);
    void updateViewsSafely(get, set, true);
  },
  setAnalysisWavelengthIndex: (index) => {
    set({ analysisWavelengthIndex: index, svg: null });
    persistCurrentState(get);
    void updateViewsSafely(get, set, true);
  },
  bootstrap: async () => {
    set({ isBusy: true });
    try {
      const persisted = loadPersistedWorkbench();
      const recentFiles = loadRecentFiles();
      set({ recentFiles });
      if (persisted) {
        set({
          activeTab: persisted.activeTab,
          radiusDisplay: persisted.radiusDisplay,
          sampling: persisted.sampling,
          scale: persisted.scale,
          analysisFieldIndex: persisted.analysisFieldIndex,
          analysisWavelengthIndex: persisted.analysisWavelengthIndex,
          quickOptimizeObjective: persisted.quickOptimizeObjective,
          quickOptimizeIterations: persisted.quickOptimizeIterations,
          quickOptimizeMaxEvaluations: persisted.quickOptimizeMaxEvaluations,
          quickOptimizeStepScale: persisted.quickOptimizeStepScale,
          quickOptimizeWeights: persisted.quickOptimizeWeights,
          toleranceSweepScope: persisted.toleranceSweepScope,
          toleranceSweepPerturbationPct: persisted.toleranceSweepPerturbationPct,
          toleranceSweepMaxSurfaces: persisted.toleranceSweepMaxSurfaces,
          variants: persisted.variants,
          requirementProfile: persisted.requirementProfile,
          requirementProfileBase: persisted.requirementProfileBase,
          requirements: persisted.requirements,
          autosavePath: persisted.autosavePath,
          autosaveSavedAt: persisted.autosaveSavedAt,
          draftCount: persisted.draftCount
        });
      }
      const examplesPromise = api.examples();
      let modelResponse: ModelResponse | null = null;
      let shouldRestoreSurfaceVariables = false;
      if (persisted?.modelId) {
        try {
          modelResponse = await api.getModel(persisted.modelId);
          shouldRestoreSurfaceVariables = true;
        } catch {
          modelResponse = null;
        }
      }
      if (!modelResponse && persisted?.autosavePath) {
        try {
          modelResponse = await api.restoreDraft(persisted.autosavePath);
          shouldRestoreSurfaceVariables = true;
        } catch {
          modelResponse = null;
        }
      }
      if (!modelResponse) {
        modelResponse = await api.newModel();
        set({
          autosavePath: null,
          autosaveSavedAt: null,
          draftCount: null,
          workbenchFileStatus: workbenchStatus(modelResponse, "new", false)
        });
      }
      if (shouldRestoreSurfaceVariables) {
        modelResponse = await restorePersistedSurfaceVariables(modelResponse, persisted?.surfaceVariables);
      }
      const examples = await examplesPromise;
      applyResponse(set, modelResponse);
      if (!get().workbenchFileStatus) {
        const hasSidecar = Boolean(sidecarWorkbench(modelResponse));
        set({ workbenchFileStatus: workbenchStatus(modelResponse, hasSidecar ? "restored" : "opened", hasSidecar, countSidecarPayload(sidecarWorkbench(modelResponse))) });
      }
      if (shouldRestoreSurfaceVariables && persisted) {
        set({
          quickOptimizeResult: persisted.quickOptimizeResult,
          toleranceSweepResult: persisted.toleranceSweepResult
        });
      } else {
        set({
          quickOptimizeResult: null,
          toleranceSweepResult: null
        });
      }
      set({
        examples: examples.examples,
        selectedSurface: clampSurfaceSelection(persisted?.selectedSurface ?? modelResponse.model.stopSurface ?? 1, modelResponse.model.surfaces.length),
        analysisFieldIndex: persisted?.analysisFieldIndex ?? null,
        analysisWavelengthIndex: persisted?.analysisWavelengthIndex ?? null
      });
      set(clampedAnalysisScope(get(), modelResponse.model));
      if (shouldRestoreSurfaceVariables && persisted?.sensorAssumptions) {
        const summary = await api.patchSensor(modelResponse.model.id, persisted.sensorAssumptions);
        applySummary(set, summary);
      } else {
        await refreshSummary(get, set);
      }
      await updateViews(get, set, true);
      await autosaveDraft(get, set);
      persistCurrentState(get);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  newModel: async (spec) => {
    set({ isBusy: true });
    try {
      const response = await api.newModel(spec);
      applyResponse(set, response);
      set({
        selectedSurface: 1,
        analysisFieldIndex: null,
        analysisWavelengthIndex: null,
        autosavePath: null,
        autosaveSavedAt: null,
        draftCount: null,
        variants: [],
        quickOptimizeResult: null,
        toleranceSweepResult: null,
        workbenchFileStatus: workbenchStatus(response, "new", false)
      });
      await refreshSummary(get, set);
      await updateViews(get, set, true);
      await autosaveDraft(get, set);
      persistCurrentState(get);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  openModel: async (path) => {
    set({ isBusy: true, errors: [] });
    try {
      let response = await api.openModel(path);
      const sidecar = sidecarWorkbench(response);
      const hasSidecarWorkbench = Boolean(sidecar);
      applyResponse(set, response);
      set({
        selectedSurface: hasSidecarWorkbench ? get().selectedSurface : response.model.stopSurface ?? 1,
        analysisFieldIndex: null,
        analysisWavelengthIndex: null,
        autosavePath: null,
        autosaveSavedAt: null,
        draftCount: null,
        variants: hasSidecarWorkbench ? get().variants : [],
        quickOptimizeResult: hasSidecarWorkbench ? get().quickOptimizeResult : null,
        toleranceSweepResult: hasSidecarWorkbench ? get().toleranceSweepResult : null
      });
      if (hasSidecarWorkbench) {
        response = await applySidecarWorkbench(get, set, response);
      } else {
        set({
          selectedSurface: response.model.stopSurface ?? 1,
          analysisFieldIndex: null,
          analysisWavelengthIndex: null,
          variants: [],
          quickOptimizeResult: null,
          toleranceSweepResult: null
        });
      }
      await refreshSummary(get, set);
      await updateViews(get, set, true);
      await autosaveDraft(get, set);
      set({ workbenchFileStatus: workbenchStatus(response, hasSidecarWorkbench ? "restored" : "opened", hasSidecarWorkbench, countSidecarPayload(sidecar)) });
      recordModelFile(get, set, response.model.filename ?? path, path.includes("/site-packages/rayoptics/") || path.includes("/rayoptics-env/lib/") ? "example" : "open", hasSidecarWorkbench);
      persistCurrentState(get);
      return true;
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
      return false;
    } finally {
      set({ isBusy: false });
    }
  },
  openPatent: async (simulationId) => {
    set({ isBusy: true, errors: [] });
    try {
      const response = await api.openPatent(simulationId);
      applyResponse(set, response);
      set({
        selectedSurface: response.model.stopSurface ?? 1,
        analysisFieldIndex: null,
        analysisWavelengthIndex: null,
        autosavePath: null,
        autosaveSavedAt: null,
        draftCount: null,
        variants: [],
        quickOptimizeResult: null,
        toleranceSweepResult: null,
        workbenchFileStatus: workbenchStatus(response, "opened", false)
      });
      await refreshSummary(get, set);
      await updateViews(get, set, true);
      await autosaveDraft(get, set);
      persistCurrentState(get);
      return true;
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
      return false;
    } finally {
      set({ isBusy: false });
    }
  },
  saveModel: async (path, overwrite = false) => {
    const model = get().model;
    if (!model) return false;
    set({ isBusy: true, errors: [] });
    try {
      const response = await api.saveModel(model.id, path, overwrite, workbenchMetadataFromState(get()));
      applyResponse(set, response);
      await refreshSummary(get, set);
      await autosaveDraft(get, set);
      set({ workbenchFileStatus: workbenchStatus(response, "saved", Boolean(response.workbench), countSidecarPayload(sidecarWorkbench(response))) });
      recordModelFile(get, set, response.model.filename ?? path, "save", Boolean(response.workbench));
      persistCurrentState(get);
      return true;
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
      return false;
    } finally {
      set({ isBusy: false });
    }
  },
  refresh: async () => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true });
    try {
      const response = await api.refreshModel(model.id);
      applyResponse(set, response);
      await refreshSummary(get, set);
      await updateViews(get, set, true);
      await autosaveDraft(get, set);
      persistCurrentState(get);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  setQuickOptimizeObjective: (objective) => {
    set({
      quickOptimizeObjective: objective,
      quickOptimizeWeights: weightsForQuickOptimizeObjective(objective)
    });
    persistCurrentState(get);
  },
  setQuickOptimizeIterations: (iterations) => {
    set({ quickOptimizeIterations: Math.max(1, Math.min(5, Math.trunc(iterations))) });
    persistCurrentState(get);
  },
  setQuickOptimizeMaxEvaluations: (evaluations) => {
    set({ quickOptimizeMaxEvaluations: Math.max(4, Math.min(160, Math.trunc(evaluations))) });
    persistCurrentState(get);
  },
  setQuickOptimizeStepScale: (scale) => {
    set({ quickOptimizeStepScale: Math.max(0.1, Math.min(5, scale)) });
    persistCurrentState(get);
  },
  setQuickOptimizeWeight: (key, weight) => {
    set({
      quickOptimizeObjective: "custom",
      quickOptimizeWeights: sanitizeQuickOptimizeWeights({
        ...get().quickOptimizeWeights,
        [key]: weight
      })
    });
    persistCurrentState(get);
  },
  resetQuickOptimizeWeights: () => {
    const objective = get().quickOptimizeObjective === "custom" ? "balanced" : get().quickOptimizeObjective;
    set({
      quickOptimizeObjective: objective,
      quickOptimizeWeights: weightsForQuickOptimizeObjective(objective)
    });
    persistCurrentState(get);
  },
  setToleranceSweepScope: (scope) => {
    set({ toleranceSweepScope: scope });
    persistCurrentState(get);
  },
  setToleranceSweepPerturbationPct: (value) => {
    set({ toleranceSweepPerturbationPct: Math.max(0.05, Math.min(5, value)) });
    persistCurrentState(get);
  },
  setToleranceSweepMaxSurfaces: (value) => {
    set({ toleranceSweepMaxSurfaces: Math.max(1, Math.min(20, Math.trunc(value))) });
    persistCurrentState(get);
  },
  runToleranceSweep: async () => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true, errors: [] });
    try {
      const state = get();
      const response = await api.toleranceSweep(model.id, {
        scope: state.toleranceSweepScope,
        perturbationPct: state.toleranceSweepPerturbationPct,
        maxSurfaces: state.toleranceSweepMaxSurfaces
      });
      applyResponse(set, response, { preserveSummary: true, analysisStale: get().analysisStale });
      set({ toleranceSweepResult: response.result });
      persistCurrentState(get);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  captureVariant: (name, source = "manual") => {
    captureCurrentVariant(get, set, name, source);
  },
  deleteVariant: (id) => {
    set({ variants: get().variants.filter((variant) => variant.id !== id) });
    persistCurrentState(get);
  },
  clearVariants: () => {
    set({ variants: [] });
    persistCurrentState(get);
  },
  quickOptimize: async () => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true, errors: [] });
    try {
      const state = get();
      const objectiveName = quickOptimizeObjectiveLabel(state.quickOptimizeObjective);
      captureCurrentVariant(get, set, `Before Quick Optimize · ${objectiveName}`, "quick-optimize-before");
      const response = await api.quickOptimize(model.id, {
        objective: state.quickOptimizeObjective,
        iterations: state.quickOptimizeIterations,
        maxEvaluations: state.quickOptimizeMaxEvaluations,
        stepScale: state.quickOptimizeStepScale,
        operandWeights: state.quickOptimizeWeights,
        operandTargets: quickOptimizeTargetsFromRequirements(state.requirements)
      });
      set({ quickOptimizeResult: response.result });
      await applyMutationResponse(get, set, response, get().selectedSurface);
      captureCurrentVariant(get, set, `After Quick Optimize · ${objectiveName} (${response.result.status})`, "quick-optimize-after");
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  undo: async () => {
    const model = get().model;
    if (!model || !get().canUndo) return;
    set({ isBusy: true });
    try {
      const response = await api.undoModel(model.id);
      await applyMutationResponse(get, set, response, get().selectedSurface);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  redo: async () => {
    const model = get().model;
    if (!model || !get().canRedo) return;
    set({ isBusy: true });
    try {
      const response = await api.redoModel(model.id);
      await applyMutationResponse(get, set, response, get().selectedSurface);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  patchSurface: async (surfaceIndex, patch) => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true });
    try {
      const response = await api.patchSurface(model.id, surfaceIndex, patch);
      await applyMutationResponse(get, set, response);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  patchSystem: async (patch) => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true });
    try {
      const response = await api.patchSystem(model.id, patch);
      await applyMutationResponse(get, set, response, get().selectedSurface);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  patchSensor: async (patch) => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true, errors: [] });
    try {
      const summary = await api.patchSensor(model.id, patch);
      applySummary(set, summary);
      persistCurrentState(get);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  insertSurface: async (after) => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true });
    try {
      const response = await api.insertSurface(model.id, after);
      await applyMutationResponse(get, set, response, after + 1);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  deleteSurface: async (surfaceIndex) => {
    const model = get().model;
    if (!model) return;
    set({ isBusy: true });
    try {
      const response = await api.deleteSurface(model.id, surfaceIndex);
      await applyMutationResponse(get, set, response, Math.min(surfaceIndex, response.model.surfaces.length - 2));
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isBusy: false });
    }
  },
  refreshAnalysis: async () => {
    try {
      await refreshSummary(get, set);
      await updateViews(get, set, true);
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    }
  },
  runExampleChecks: async () => {
    set({ isCheckingExamples: true });
    try {
      const compatibility = await api.checkExamples({ limit: 6, includeAnalyses: true });
      set({ compatibility });
    } catch (error) {
      set({ errors: [error instanceof Error ? error.message : String(error)] });
    } finally {
      set({ isCheckingExamples: false });
    }
  }
}));
