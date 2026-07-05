import type {
  AnalysisScale,
  AnalysisSummary,
  ExampleCheckResponse,
  ExampleFile,
  LayoutPayload,
  ModelResponse,
  NewModelSpec,
  PatentCompany,
  PatentDbStatus,
  PatentSearchResult,
  QuickOptimizeObjective,
  QuickOptimizeResponse,
  QuickOptimizeTargets,
  QuickOptimizeWeights,
  SensorPatch,
  SurfaceRow,
  SystemPatch,
  ToleranceSweepResponse,
  ToleranceSweepScope,
  UpdateMode,
  WorkbenchMetadata
} from "./types";

const jsonHeaders = { "Content-Type": "application/json" };

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // keep default detail
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; rayoptics: string }>("/api/health"),
  examples: () => request<{ examples: ExampleFile[] }>("/api/examples"),
  patentStatus: () => request<PatentDbStatus>("/api/patents/status"),
  patentCompanies: () => request<{ companies: PatentCompany[] }>("/api/patents/companies"),
  patentSearch: (options: { company?: string; query?: string; status?: string; limit?: number } = {}) => {
    const params = new URLSearchParams();
    if (options.company) params.set("company", options.company);
    if (options.query) params.set("query", options.query);
    params.set("status", options.status ?? "camerae2e_ready");
    params.set("limit", String(options.limit ?? 80));
    return request<{ results: PatentSearchResult[] }>(`/api/patents/search?${params.toString()}`);
  },
  openPatent: (simulationId: string) =>
    request<ModelResponse>("/api/patents/open", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ simulationId })
    }),
  checkExamples: (options: { paths?: string[]; limit?: number; includeAnalyses?: boolean } = {}) =>
    request<ExampleCheckResponse>("/api/examples/check", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({
        paths: options.paths,
        limit: options.limit ?? 6,
        includeAnalyses: options.includeAnalyses ?? true
      })
    }),
  newModel: (spec: NewModelSpec = { efl: 50, epd: 12.5, fov: 20 }) =>
    request<ModelResponse>("/api/models/new", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(spec)
    }),
  getModel: (modelId: string) => request<ModelResponse>(`/api/models/${modelId}`),
  openModel: (path: string) =>
    request<ModelResponse>("/api/models/open", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ path })
    }),
  saveModel: (modelId: string, path?: string, overwrite = false, workbench?: WorkbenchMetadata | null) =>
    request<ModelResponse>("/api/models/save", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ modelId, path, overwrite, workbench })
    }),
  autosaveDraft: (modelId: string, workbench?: WorkbenchMetadata | null) =>
    request<{ modelId: string; path: string; savedAt: string; draftCount: number; prunedCount: number }>(`/api/models/${modelId}/draft`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ workbench })
    }),
  restoreDraft: (path: string) =>
    request<ModelResponse>("/api/models/drafts/restore", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ path })
    }),
  refreshModel: (modelId: string) =>
    request<ModelResponse>(`/api/models/${modelId}/update`, {
      method: "POST"
    }),
  quickOptimize: (
    modelId: string,
    options: {
      objective?: QuickOptimizeObjective;
      iterations?: number;
      maxEvaluations?: number;
      stepScale?: number;
      operandWeights?: QuickOptimizeWeights;
      operandTargets?: QuickOptimizeTargets;
    } = {}
  ) =>
    request<QuickOptimizeResponse>(`/api/models/${modelId}/optimize/quick`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({
        objective: options.objective ?? "balanced",
        iterations: options.iterations ?? 2,
        maxEvaluations: options.maxEvaluations ?? 48,
        stepScale: options.stepScale ?? 1,
        operandWeights: options.operandWeights,
        operandTargets: options.operandTargets
      })
    }),
  toleranceSweep: (
    modelId: string,
    options: { scope?: ToleranceSweepScope; perturbationPct?: number; maxSurfaces?: number } = {}
  ) =>
    request<ToleranceSweepResponse>(`/api/models/${modelId}/analysis/tolerance-sweep`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({
        scope: options.scope ?? "powered",
        perturbationPct: options.perturbationPct ?? 0.5,
        maxSurfaces: options.maxSurfaces ?? 6
      })
    }),
  patchSettings: (modelId: string, updateMode: UpdateMode) =>
    request<ModelResponse>(`/api/models/${modelId}/settings`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify({ updateMode })
    }),
  undoModel: (modelId: string) =>
    request<ModelResponse>(`/api/models/${modelId}/undo`, {
      method: "POST"
    }),
  redoModel: (modelId: string) =>
    request<ModelResponse>(`/api/models/${modelId}/redo`, {
      method: "POST"
    }),
  patchSurface: (modelId: string, surfaceIndex: number, patch: Partial<SurfaceRow>) =>
    request<ModelResponse>(`/api/models/${modelId}/surfaces/${surfaceIndex}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(patch)
    }),
  patchSystem: (modelId: string, patch: SystemPatch) =>
    request<ModelResponse>(`/api/models/${modelId}/system`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(patch)
    }),
  patchSensor: (modelId: string, patch: SensorPatch) =>
    request<AnalysisSummary>(`/api/models/${modelId}/sensor`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(patch)
    }),
  insertSurface: (modelId: string, after: number) =>
    request<ModelResponse>(`/api/models/${modelId}/surfaces`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ after, radius: 0, thickness: 1, glass: "air", catalog: "", semiDiameter: 5 })
    }),
  deleteSurface: (modelId: string, surfaceIndex: number) =>
    request<ModelResponse>(`/api/models/${modelId}/surfaces/${surfaceIndex}`, {
      method: "DELETE"
    }),
  layout: (modelId: string) => request<LayoutPayload>(`/api/models/${modelId}/layout`),
  firstOrder: (modelId: string) => request<{ values: Record<string, number | null>; warnings: string[] }>(`/api/models/${modelId}/analysis/first-order`),
  analysisSummary: (modelId: string) => request<AnalysisSummary>(`/api/models/${modelId}/analysis/summary`),
  analysisSvg: async (
    modelId: string,
    kind: string,
    sampling: number,
    scale: AnalysisScale,
    fieldIndex: number | null,
    wavelengthIndex: number | null
  ) => {
    const response = await fetch(`/api/models/${modelId}/analysis/${kind}`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ sampling, scale, fieldIndex, wavelengthIndex })
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.text();
  }
};
