import type {
  AssetStatus,
  BenchmarkStatus,
  JobRecord,
  JobSubmitResponse,
  ComponentSearchResponse,
  ComponentSelection,
  DatasetEstimate,
  DatasetInventory,
  LensComponent,
  ModuleApplicationResponse,
  ModuleCompatibilityResponse,
  ProjectPayload,
  RequirementSet,
  SensorComponent,
  StudyRecord,
  StudySpec
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail ?? payload));
  }
  return (await response.json()) as T;
}

export function bootstrapProject(): Promise<ProjectPayload> {
  return request<ProjectPayload>("/api/v2/bootstrap", { method: "POST", body: "{}" });
}

export function fetchProject(projectId: string): Promise<ProjectPayload> {
  return request<ProjectPayload>(`/api/v2/projects/${projectId}`);
}

export function fetchAssetStatus(projectId: string): Promise<AssetStatus> {
  return request<AssetStatus>(`/api/v2/projects/${projectId}/assets/status`);
}

export function fetchBenchmarkStatus(
  projectId: string,
  studyId: string
): Promise<BenchmarkStatus> {
  return request<BenchmarkStatus>(
    `/api/v2/projects/${projectId}/studies/${studyId}/benchmark/status`
  );
}

export function inspectDatasetSource(
  projectId: string,
  studyId: string,
  requestBody: Record<string, unknown>
): Promise<DatasetInventory> {
  return request<DatasetInventory>(
    `/api/v2/projects/${projectId}/studies/${studyId}/datasets/inventory`,
    { method: "POST", body: JSON.stringify(requestBody) }
  );
}

export function estimateDatasetExport(
  projectId: string,
  studyId: string,
  requestBody: Record<string, unknown>
): Promise<DatasetEstimate> {
  return request<DatasetEstimate>(
    `/api/v2/projects/${projectId}/studies/${studyId}/datasets/estimate`,
    { method: "POST", body: JSON.stringify(requestBody) }
  );
}

export function updateStudy(projectId: string, studyId: string, spec: StudySpec): Promise<StudyRecord> {
  return request<StudyRecord>(`/api/v2/projects/${projectId}/studies/${studyId}`, {
    method: "PUT",
    body: JSON.stringify(spec)
  });
}

export function fetchJobs(projectId: string, studyId: string): Promise<JobRecord[]> {
  return request<JobRecord[]>(`/api/v2/projects/${projectId}/jobs?study_id=${encodeURIComponent(studyId)}&limit=100`);
}

export function fetchJob(projectId: string, jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/api/v2/projects/${projectId}/jobs/${jobId}`);
}

const operationPath: Record<string, string> = {
  evaluate: "evaluate",
  benchmark_preflight: "benchmark/preflight",
  benchmark_run: "benchmark/run",
  train_detector: "benchmark/train-detector",
  requirements_evaluate: "requirements/evaluate",
  sensitivity: "explore",
  optimize: "optimize",
  validate_candidate: "candidates/validate",
  dataset_export: "datasets",
  calibrate: "calibrations",
  report: "reports"
};

export function submitOperation(
  projectId: string,
  studyId: string,
  kind: string,
  operationRequest: Record<string, unknown> = {}
): Promise<JobSubmitResponse> {
  const suffix = operationPath[kind];
  if (!suffix) throw new Error(`Unsupported operation: ${kind}`);
  return request<JobSubmitResponse>(`/api/v2/projects/${projectId}/studies/${studyId}/${suffix}`, {
    method: "POST",
    body: JSON.stringify({ request: operationRequest })
  });
}

export function artifactUrl(projectId: string, hash: string): string {
  return `/api/v2/projects/${projectId}/artifacts/${hash}`;
}

export function scenePreviewUrl(projectId: string, studyId: string, sceneId: string): string {
  return `/api/v2/projects/${projectId}/studies/${studyId}/scenes/${sceneId}/preview`;
}

function queryString(values: Record<string, string | number | boolean | null | undefined>): string {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") params.set(key, String(value));
  });
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

export function searchLensComponents(
  filters: Record<string, string | number | boolean | null | undefined>
): Promise<ComponentSearchResponse<LensComponent>> {
  return request<ComponentSearchResponse<LensComponent>>(`/api/v2/catalog/lenses${queryString(filters)}`);
}

export function fetchLensComponent(lensId: string): Promise<Record<string, any>> {
  return request<Record<string, any>>(`/api/v2/catalog/lenses/${encodeURIComponent(lensId)}`);
}

export function searchSensorComponents(
  filters: Record<string, string | number | boolean | null | undefined>
): Promise<ComponentSearchResponse<SensorComponent>> {
  return request<ComponentSearchResponse<SensorComponent>>(`/api/v2/catalog/sensors${queryString(filters)}`);
}

export function fetchSensorComponent(sensorId: string): Promise<Record<string, any>> {
  return request<Record<string, any>>(`/api/v2/catalog/sensors/${encodeURIComponent(sensorId)}`);
}

export function evaluateComponentModules(
  requirements: RequirementSet,
  candidates: ComponentSelection[]
): Promise<ModuleCompatibilityResponse> {
  return request<ModuleCompatibilityResponse>("/api/v2/catalog/modules/evaluate", {
    method: "POST",
    body: JSON.stringify({ requirements, candidates })
  });
}

export function applyComponentModule(
  projectId: string,
  studyId: string,
  selection: ComponentSelection
): Promise<ModuleApplicationResponse> {
  return request<ModuleApplicationResponse>(`/api/v2/projects/${projectId}/studies/${studyId}/module`, {
    method: "POST",
    body: JSON.stringify({ selection, allow_incompatible: false })
  });
}

export function compareComponentModules(
  projectId: string,
  studyId: string,
  candidates: ComponentSelection[],
  sceneId?: string
): Promise<JobSubmitResponse> {
  return request<JobSubmitResponse>(`/api/v2/projects/${projectId}/studies/${studyId}/modules/compare`, {
    method: "POST",
    body: JSON.stringify({ candidates, scene_id: sceneId ?? null, allow_incompatible: false })
  });
}
