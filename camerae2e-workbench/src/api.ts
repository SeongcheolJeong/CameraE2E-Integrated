import type {
  AssetStatus,
  BenchmarkStatus,
  JobRecord,
  JobSubmitResponse,
  ProjectPayload,
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
